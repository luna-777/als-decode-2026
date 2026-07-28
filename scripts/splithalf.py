"""Round C.1 — split-half test of the baseline/ΔAUC correlation.

§0.12 reported that ALS patients whose backbone fits worse gain more from
adaptation (ρ ≈ −0.79). Both quantities were estimated on the *same* evaluation
epochs, so they share noise: a patient whose baseline AUC happens to be
underestimated will have ΔAUC = adapted − baseline correspondingly
overestimated. That alone produces a negative correlation with no underlying
relationship — the classic same-set artefact.

This estimates them on disjoint halves. Each subject's evaluation set is split in
half (stratified, seeded); the baseline is estimated on half A and the baseline,
adapted score and Δ on half B. Correlating Δ_B against baseline_A removes the
shared-noise channel entirely, because the two estimates come from different
epochs.

Reported alongside the same-set correlation (Δ_B vs baseline_B), which should show
the inflated value.

Usage
-----
    python scripts/splithalf.py --paradigm als
    python scripts/splithalf.py --paradigm mi --montage-ks 0 17
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def stratified_halves(y: np.ndarray, idx: np.ndarray, seed: int):
    """Split *idx* into two class-balanced halves."""
    from sklearn.model_selection import train_test_split

    yy = np.asarray(y)[idx]
    if len(np.unique(yy)) < 2 or len(idx) < 4:
        return None, None
    a, b = train_test_split(idx, train_size=0.5, random_state=seed, stratify=yy)
    return np.sort(a), np.sort(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", choices=["mi", "als", "p300"], default="als")
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--splits", nargs="+", default=["stratified"])
    ap.add_argument("--calib-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--montage-ks", nargs="+", type=int, default=None,
                    help="MI only: run within each of these permutation-k conditions.")
    ap.add_argument("--permute-seed", type=int, default=101)
    ap.add_argument("--adapt-epochs", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.adapt_stage3 import (
        _adapt_head_on_features,
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

    sizes = args.calib_sizes or ([10, 50, 100, 200] if args.paradigm == "mi"
                                 else [24, 48, 96, 240, 480])
    ks = args.montage_ks if (args.paradigm == "mi" and args.montage_ks) else [0]

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
                            A, B = stratified_halves(y, sp.eval_idx, sd)
                            if A is None or len(np.unique(y[A])) < 2 or len(np.unique(y[B])) < 2:
                                continue
                            y_cal = np.asarray(y)[sp.calib_idx]
                            if len(np.unique(y_cal)) < 2:
                                continue

                            base_head = copy.deepcopy(model.head); base_head.eval()
                            bA = _head_auc(base_head, feats[A], y[A])
                            bB = _head_auc(base_head, feats[B], y[B])

                            ad = copy.deepcopy(model.head)
                            _adapt_head_on_features(ad, feats[sp.calib_idx], y_cal,
                                                    n_epochs=args.adapt_epochs,
                                                    lr=args.adapt_lr, seed=sd)
                            aB = _head_auc(ad, feats[B], y[B])

                            rows.append({
                                "paradigm": args.paradigm, "subject": subj,
                                "fold": ref.version, "permute_k": k, "split": arm,
                                "calib_size": N, "seed": sd,
                                "n_half_a": len(A), "n_half_b": len(B),
                                "baseline_a": bA, "baseline_b": bB,
                                "adapted_b": aB, "delta_b": aB - bB,
                            })
            log.info("  subject %s | %s done (%d rows)", subj, ref.version, len(rows))

    out = Path(args.out) if args.out else ROOT / "experiments" / "roundc" / f"{args.paradigm}_splithalf.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("Wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
