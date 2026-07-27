"""Stage 3 per-patient adaptation — freeze backbone, train head on calibration data.

Protocol
--------
For each held-out test subject:
  1. Load epochs with the montage read from configs/dataset/*.yaml — the same
     config the training path uses — and assert it matches the montage recorded
     for the checkpoint.
  2. Apply the cross-subject preprocessor (fit on training data, loaded from
     the fold's preprocessor.pkl).
  3. Draw the calibration set with a stratified random split
     (train_test_split(train_size=N, stratify=y, random_state=seed)); the rest is
     the evaluation set. Note: the module docstring previously described a
     "first N epochs" temporal split, which the code has not done since bf23ef9.
  4. Clone the pre-trained model, freeze the backbone, keep the pretrained head.
  5. Train only the head on the calibration set.
  6. Report AUC vs calibration size, with the checkpoint and montage recorded in
     every output row.

Checkpoints are named explicitly. There is no best-of-N discovery; see
docs/AUDIT.md §0.4.

Usage
-----
    python scripts/adapt_stage3.py --paradigm mi --folds version_28
    python scripts/adapt_stage3.py --paradigm mi --folds version_27 version_28 version_29 version_30 --fold-mode average
    python scripts/adapt_stage3.py --paradigm p300
"""
from __future__ import annotations

import argparse
import copy
import glob
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Checkpoint discovery has moved to src/models/checkpoints.py, which requires folds
# to be named explicitly. The previous _find_best_checkpoint() ranked whatever was on
# disk by the val AUC in the filename; see docs/AUDIT.md §0.4 for why that made every
# MI result a function of the state of lightning_logs/ at run time.


# ---------------------------------------------------------------------------
# Model loading + head reset
# ---------------------------------------------------------------------------

def _load_model_frozen(
    ckpt_path: Path,
    paradigm: str,
    n_channels: int,
    n_times: int,
    reinit_head: bool = True,
) -> nn.Module:
    """Load checkpoint, freeze backbone. Re-initialises head only when reinit_head=True.

    reinit_head=False → pretrained head intact (baseline and warm-start adapted path).
    reinit_head=True  → fresh random head (cold-start ablation only).

    n_channels/n_times come from the dataset config, not from literals — the two
    hardcoded shape tables this function used to carry were a second place for the
    montage to drift out of sync with training (docs/AUDIT.md §0.1).
    """
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    from src.training.lit_module import LitEEG

    backbone = BackboneEncoder(n_channels=n_channels, n_times=n_times)
    head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
    pretrained = EEGDecoder(backbone, head)

    class _Cfg:
        def get(self, k, d=None): return d

    lit = LitEEG.load_from_checkpoint(
        str(ckpt_path), model=pretrained, cfg=_Cfg(), map_location="cpu"
    )

    # Rebuild as stage3_adapted so train() override keeps backbone in eval
    from src.models.backbone import EEGDecoder as _EEGDecoder
    adapted = _EEGDecoder(
        copy.deepcopy(lit.model.backbone),
        copy.deepcopy(lit.model.head),
        stage="stage3_adapted",
    )
    if reinit_head:
        nn.init.xavier_uniform_(adapted.head.linear.weight)
        nn.init.zeros_(adapted.head.linear.bias)

    return adapted


# ---------------------------------------------------------------------------
# Head-only training
# ---------------------------------------------------------------------------

def _adapt_head(
    model: nn.Module,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> nn.Module:
    """Train only the decoder head on calibration data. Mutates and returns model."""
    from src.training.losses import weighted_bce

    model.train()
    # Only head parameters have requires_grad=True
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-3)

    X_t = torch.from_numpy(X_calib).float()
    y_t = torch.from_numpy(y_calib).float()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    n_pos = int(y_calib.sum())
    n_neg = len(y_calib) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)])

    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = weighted_bce(logits, yb, pos_weight=pw)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _epoch_auc(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(model(torch.from_numpy(X).float()).squeeze(-1)).numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


# ---------------------------------------------------------------------------
# Main calibration-curve routine
# ---------------------------------------------------------------------------

def calibration_curve(
    paradigm: str,
    log_dir: str = "lightning_logs",
    calib_sizes: list[int] | None = None,
    n_adapt_epochs: int = 100,
    adapt_lr: float = 1e-3,
    seed: int = 42,
    folds: list[str] | None = None,
    fold_mode: str = "pinned",
    strict_montage: bool = False,
) -> list[dict]:
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.datasets.registry import get_dataset
    from src.models.checkpoints import load_folds, loso_fold_for_subject, spec_from_config

    # Channel list, sfreq, band and window all come from the same config the training
    # path reads (configs/dataset/*.yaml). No second copy lives here.
    spec = spec_from_config(paradigm)
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    log.info("Montage from config: %d channels %s", n_channels, list(spec.channels))

    if calib_sizes is None:
        calib_sizes = [24, 48, 96, 240, 480] if paradigm == "p300" else [10, 20, 50, 100, 200]

    wrapper = get_dataset(spec)
    test_subjects = wrapper.subject_list[-10:]
    log.info("Test subjects for Stage 3: %s", test_subjects)

    # Folds are named explicitly; nothing is selected by ranking (docs/AUDIT.md §0.4).
    mi_folds = None
    if paradigm != "p300":
        mi_folds = load_folds(log_dir, folds or [], expect_channels=n_channels)
        if fold_mode == "pinned" and len(mi_folds) != 1:
            raise ValueError(
                f"fold_mode='pinned' needs exactly one --folds entry, got {len(mi_folds)}. "
                f"Use --fold-mode average to aggregate across folds."
            )

    results: list[dict] = []

    for subj in test_subjects:
        log.info("Subject %s — loading epochs…", subj)

        if paradigm == "p300":
            subj_folds = [loso_fold_for_subject(log_dir, subj, strict=True)]
        else:
            subj_folds = mi_folds

        X_raw, y_all, meta = wrapper.load_epochs([subj])

        for ref in subj_folds:
            # The montage contract: what this checkpoint was trained on must equal
            # what we just loaded, name for name and position for position.
            montage = load_montage(ref.version_dir, spec, n_times, strict=strict_montage)
            assert_montage_matches(
                montage,
                meta["channels"],
                where=f"{ref.version} vs {paradigm} data for subject {subj}",
            )

            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)
            X_all = pp.transform(X_raw)

            for n_calib in calib_sizes:
                if n_calib >= len(X_all) - 10:
                    log.warning("  Skipping calib_size=%d (too large for subject %s)", n_calib, subj)
                    continue

                from sklearn.model_selection import train_test_split
                X_calib, X_eval, y_calib, y_eval = train_test_split(
                    X_all, y_all, train_size=n_calib, random_state=seed, stratify=y_all
                )

                if len(np.unique(y_eval)) < 2:
                    log.warning("  Skipping calib_size=%d (eval set has only one class)", n_calib)
                    continue

                # Baseline: pretrained head (no re-init) on the same eval split.
                # Must be computed here, not before the loop, so it shares the eval
                # denominator with the adapted model.
                model_base = _load_model_frozen(
                    ref.ckpt_path, paradigm, n_channels, n_times, reinit_head=False
                )
                model_base.eval()
                baseline_auc = _epoch_auc(model_base, X_eval, y_eval)

                # Adapted: pretrained head fine-tuned on calib (warm-start), evaluated
                # on the same eval split. Keeping pretrained weights avoids having to
                # relearn the linear map from a handful of calibration samples.
                model_adapted = _load_model_frozen(
                    ref.ckpt_path, paradigm, n_channels, n_times, reinit_head=False
                )
                _adapt_head(model_adapted, X_calib, y_calib,
                            n_epochs=n_adapt_epochs, lr=adapt_lr)
                adapted_auc = _epoch_auc(model_adapted, X_eval, y_eval)

                delta_auc = adapted_auc - baseline_auc
                log.info(
                    "  Subject %s | %s | calib=%3d | baseline=%.3f | adapted=%.3f (Δ=%+.3f)",
                    subj, ref.version, n_calib, baseline_auc, adapted_auc, delta_auc,
                )
                results.append({
                    "subject": subj,
                    "calib_size": n_calib,
                    "baseline_auc": baseline_auc,
                    "adapted_auc": adapted_auc,
                    "delta_auc": delta_auc,
                    "seed": seed,
                    "fold": ref.version,
                    "fold_mode": fold_mode,
                    "checkpoint": str(ref.ckpt_path),
                    "checkpoint_val_auc": ref.val_auc,
                    "montage_source": montage.source,
                    "n_channels": n_channels,
                    "channels": "|".join(spec.channels),
                })

    # Persist results so they can be re-read without re-running
    import csv
    out_dir = Path("experiments") / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{paradigm}_results.csv"
    fieldnames = [
        "paradigm", "subject", "calib_size", "baseline_auc", "adapted_auc", "delta_auc",
        "seed", "fold", "fold_mode", "checkpoint", "checkpoint_val_auc",
        "montage_source", "n_channels", "channels",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({"paradigm": paradigm, **r})
    log.info("Results saved → %s", csv_path)

    try:
        import os
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "als-decode"),
            name=f"stage3_{paradigm}",
            job_type="stage3_adaptation",
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={"paradigm": paradigm, "calib_sizes": calib_sizes, "seed": seed},
        )
        for r in results:
            wandb.log({"paradigm": paradigm, **r})
        wandb.finish()
    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", choices=["mi", "p300"], required=True)
    parser.add_argument("--log-dir", default="lightning_logs")
    parser.add_argument("--calib-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--adapt-epochs", type=int, default=100)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument(
        "--folds", nargs="+", default=None,
        help="Explicit MI fold version dirs, e.g. --folds version_28. Required for "
             "--paradigm mi; P300 resolves folds per subject from val_subjects.json.",
    )
    parser.add_argument(
        "--fold-mode", choices=["pinned", "average"], default="pinned",
        help="pinned: exactly one fold, recorded in every row. average: score every "
             "named fold and aggregate downstream. Best-of-N selection is not offered — "
             "it is selection optimism in the baseline (docs/AUDIT.md §0.4).",
    )
    parser.add_argument(
        "--strict-montage", action="store_true",
        help="Refuse checkpoints that carry no recorded montage.json instead of "
             "inferring the montage from the training config.",
    )
    args = parser.parse_args()

    results = calibration_curve(
        paradigm=args.paradigm,
        log_dir=args.log_dir,
        calib_sizes=args.calib_sizes,
        n_adapt_epochs=args.adapt_epochs,
        adapt_lr=args.adapt_lr,
        folds=args.folds,
        fold_mode=args.fold_mode,
        strict_montage=args.strict_montage,
    )

    paradigm_label = "MI (Stage 1)" if args.paradigm == "mi" else "P300 (Stage 2)"
    print(f"\n{'='*72}")
    print(f"STAGE 3 ADAPTATION — {paradigm_label}")
    print(f"{'='*72}")
    print(f"{'Subject':<10} {'Calib N':>8} {'Baseline':>10} {'Adapted':>10} {'ΔAUC':>8}")
    print(f"{'-'*72}")
    for r in results:
        print(
            f"{r['subject']:<10} {r['calib_size']:>8}"
            f" {r['baseline_auc']:>10.3f} {r['adapted_auc']:>10.3f}"
            f" {r['delta_auc']:>+8.3f}"
        )

    # Summary: mean ΔAUC at each calib size
    import collections
    by_size: dict[int, list[float]] = collections.defaultdict(list)
    for r in results:
        by_size[r["calib_size"]].append(r["delta_auc"])
    if by_size:
        print(f"\n{'Calib N':>8} {'Mean ΔAUC':>12}")
        print("-" * 22)
        for size in sorted(by_size):
            print(f"{size:>8} {np.mean(by_size[size]):>+.3f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
