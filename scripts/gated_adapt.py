"""Round C.3 — validation-gated adaptation.

Every cohort so far shows head-only adaptation is negative on average, which means
the honest deployment rule is not "adapt" but "adapt only when you can verify it
helped". This implements that rule with no access to the evaluation set:

1. Hold out 25% of the *calibration* set as an internal validation split
   (stratified). If that split would have fewer than ``--val-floor`` trials, the
   condition is reported as **ungateable** rather than gated on noise — a gate
   fitted to 2 trials is not a gate.
2. Train the head with early stopping on internal validation AUC.
3. Accept the adapted head only if it beats the pretrained head on internal
   validation. Otherwise keep the pretrained head unchanged.

Two numbers matter. ``delta_auc`` is the realised gain of the *deployed* head,
which is exactly 0 whenever the gate rejects. ``accepted`` is the acceptance rate —
how often a patient's calibration data actually supports adapting at all. That rate
is a result in its own right, not diagnostics.

Usage
-----
    python scripts/gated_adapt.py --paradigm als --folds version_31 ... version_40
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

VAL_FRACTION = 0.25


def _train_with_early_stop(head, f_tr, y_tr, f_val, y_val, n_epochs, lr, seed,
                           patience, batch_size=32):
    """Train the head, keeping the state with the best internal-validation AUC."""
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    from src.training.losses import weighted_bce

    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    ds = TensorDataset(f_tr, torch.from_numpy(np.asarray(y_tr)).float())
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    n_pos = int(np.sum(y_tr == 1))
    pw = torch.tensor([(len(y_tr) - n_pos) / max(n_pos, 1)])

    best_auc, best_state, best_epoch, since = -np.inf, copy.deepcopy(head.state_dict()), 0, 0
    for ep in range(n_epochs):
        head.train()
        for fb, yb in loader:
            opt.zero_grad()
            weighted_bce(head(fb).squeeze(-1), yb, pos_weight=pw).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(f_val).squeeze(-1)).numpy()
        auc = float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else np.nan
        if np.isfinite(auc) and auc > best_auc + 1e-6:
            best_auc, best_state, best_epoch, since = auc, copy.deepcopy(head.state_dict()), ep, 0
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, best_auc, best_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", choices=["mi", "p300", "als"], required=True)
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--splits", nargs="+", default=["stratified", "temporal_array"])
    ap.add_argument("--calib-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--montage-ks", nargs="+", type=int, default=[0],
                    help="MI only: permutation-k conditions to run.")
    ap.add_argument("--permute-seed", type=int, default=101)
    ap.add_argument("--adapt-epochs", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-floor", type=int, default=8,
                    help="Minimum internal-validation trials. Below this the "
                         "condition is reported ungateable rather than gated.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split

    from scripts.adapt_stage3 import (
        _apply_ea,
        _features,
        _head_auc,
        _load_model_frozen,
    )
    from src.datasets.degradation import apply_permutation, permutation_for_k
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.datasets.registry import get_dataset
    from src.datasets.splits import make_calibration_split
    from src.models.checkpoints import load_folds, loso_fold_for_subject, spec_from_config

    spec = spec_from_config(args.paradigm)
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    contract_spec = spec_from_config("p300") if args.paradigm == "als" else spec
    spec_noea = dataclasses.replace(spec, euclidean_alignment=False)

    sizes = args.calib_sizes or ([10, 20, 50, 100, 150, 200] if args.paradigm == "mi"
                                else [24, 48, 96, 240, 480])
    ks = args.montage_ks if args.paradigm == "mi" else [0]

    wrapper = get_dataset(spec_noea)
    subjects = (wrapper.subject_list if args.paradigm == "als"
                else wrapper.subject_list[-10:])
    refs = None
    if args.paradigm != "p300":
        refs = load_folds(args.log_dir, args.folds or [], expect_channels=n_channels)

    rows = []
    for subj in subjects:
        X_clean, y, meta = wrapper.load_epochs([subj])
        order, source = np.asarray(meta["order"]), np.asarray(meta["source"])
        subj_refs = (refs if refs is not None
                     else [loso_fold_for_subject(args.log_dir, subj,
                                                 n_channels=n_channels, strict=True)])
        for ref in subj_refs:
            montage = load_montage(ref.version_dir, contract_spec, n_times)
            assert_montage_matches(montage, meta["channels"],
                                   where=f"{ref.version} vs subject {subj}")
            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)
            model = _load_model_frozen(ref.ckpt_path, args.paradigm, n_channels,
                                       n_times, reinit_head=False)
            model.eval()

            for k in ks:
                perm = permutation_for_k(n_channels, k, args.permute_seed)
                X_deg = apply_permutation(X_clean, perm) if k else X_clean
                feats = _features(model, pp.transform(_apply_ea(X_deg, fit_idx=None)))

                for arm in args.splits:
                    for N in sizes:
                        for sd in args.seeds:
                            if N >= len(y):
                                continue
                            try:
                                sp = make_calibration_split(arm, y, N, sd, order,
                                                            source, purge_k=5)
                            except ValueError:
                                continue
                            y_cal = np.asarray(y)[sp.calib_idx]
                            y_ev = np.asarray(y)[sp.eval_idx]
                            if len(np.unique(y_ev)) < 2 or len(np.unique(y_cal)) < 2:
                                continue

                            base = copy.deepcopy(model.head); base.eval()
                            base_auc = _head_auc(base, feats[sp.eval_idx], y_ev)

                            n_val = int(round(N * VAL_FRACTION))
                            row = {
                                "paradigm": args.paradigm, "subject": subj,
                                "fold": ref.version, "permute_k": k, "split": arm,
                                "calib_size": N, "seed": sd, "n_val": n_val,
                                "baseline_auc": base_auc,
                            }
                            if n_val < args.val_floor:
                                rows.append({**row, "status": "ungateable",
                                             "accepted": "", "adapted_auc": "",
                                             "deployed_auc": base_auc,
                                             "delta_auc": 0.0, "internal_val_auc": "",
                                             "best_epoch": ""})
                                continue

                            # nested stratified split inside the calibration set
                            try:
                                tr, va = train_test_split(
                                    sp.calib_idx, test_size=n_val, random_state=sd,
                                    stratify=y_cal)
                            except ValueError:
                                rows.append({**row, "status": "ungateable_stratify",
                                             "accepted": "", "adapted_auc": "",
                                             "deployed_auc": base_auc,
                                             "delta_auc": 0.0, "internal_val_auc": "",
                                             "best_epoch": ""})
                                continue
                            y_tr, y_va = np.asarray(y)[tr], np.asarray(y)[va]
                            if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
                                rows.append({**row, "status": "ungateable_one_class_val",
                                             "accepted": "", "adapted_auc": "",
                                             "deployed_auc": base_auc,
                                             "delta_auc": 0.0, "internal_val_auc": "",
                                             "best_epoch": ""})
                                continue

                            ad = copy.deepcopy(model.head)
                            ad, val_auc, best_ep = _train_with_early_stop(
                                ad, feats[tr], y_tr, feats[va], y_va,
                                args.adapt_epochs, args.adapt_lr, sd, args.patience)

                            base_val = _head_auc(base, feats[va], y_va)
                            accept = bool(np.isfinite(val_auc) and val_auc > base_val)

                            adapted_auc = _head_auc(ad, feats[sp.eval_idx], y_ev)
                            deployed = adapted_auc if accept else base_auc
                            rows.append({**row, "status": "ok",
                                         "accepted": int(accept),
                                         "adapted_auc": adapted_auc,
                                         "deployed_auc": deployed,
                                         "delta_auc": deployed - base_auc,
                                         "internal_val_auc": val_auc,
                                         "best_epoch": best_ep})
            log.info("  subject %s | %s | %d rows", subj, ref.version, len(rows))

    tag = args.paradigm if args.paradigm != "mi" else f"mi_k{'_'.join(map(str, ks))}"
    out = Path(args.out) if args.out else ROOT / "experiments" / "roundc" / f"{tag}_gated.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("Wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
