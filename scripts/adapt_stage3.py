"""Stage 3 per-patient adaptation — freeze backbone, train head on calibration data.

Protocol
--------
For each held-out test subject:
  1. Apply the cross-subject preprocessor (fit on training data, loaded from checkpoint).
  2. Take the first N epochs as the calibration set; the rest as the evaluation set.
  3. Clone the pre-trained model, freeze the backbone, re-initialise the head.
  4. Train only the head on the calibration set.
  5. Report AUC (and character accuracy for P300) vs calibration size.

Usage
-----
    python scripts/adapt_stage3.py --paradigm mi
    python scripts/adapt_stage3.py --paradigm p300
    python scripts/adapt_stage3.py --paradigm mi --log-dir lightning_logs --calib-sizes 10 20 50 100
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

# Checkpoint detection key
_STAGE2_KEY = "model.backbone.net.conv_spatial.parametrizations.weight.original"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _n_channels_from_ckpt(path: str) -> int:
    try:
        sd = torch.load(path, map_location="cpu")["state_dict"]
        return int(sd[_STAGE2_KEY].shape[2])
    except Exception:
        return -1


def _val_auc_from_path(path: str) -> float:
    try:
        return float(Path(path).stem.split("=")[-1])
    except Exception:
        return 0.0


def _find_best_checkpoint(log_dir: str, paradigm: str) -> Path:
    """Return the fold checkpoint with highest val AUC for the given paradigm."""
    target_ch = 8 if paradigm == "p300" else 17
    pattern = str(Path(log_dir) / "version_*" / "checkpoints" / "best-epoch=*" / "*.ckpt")
    ckpts = sorted(glob.glob(pattern))
    candidates = [
        c for c in ckpts
        if _n_channels_from_ckpt(c) == target_ch
        and _val_auc_from_path(c) > 0.55
        and (Path(c).parents[2] / "preprocessor.pkl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No {paradigm} checkpoints found in {log_dir}")
    best = max(candidates, key=_val_auc_from_path)
    log.info("Using checkpoint: %s (val AUC=%.4f)", best, _val_auc_from_path(best))
    return Path(best)


# ---------------------------------------------------------------------------
# Model loading + head reset
# ---------------------------------------------------------------------------

def _load_model_frozen(ckpt_path: Path, paradigm: str) -> nn.Module:
    """Load checkpoint, freeze backbone, re-initialise head. Returns EEGDecoder."""
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    from src.training.lit_module import LitEEG

    if paradigm == "p300":
        n_channels, n_times = 8, 102
    else:
        n_channels, n_times = 17, 320

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
    # Re-initialise head (fresh weights for patient-specific calibration)
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
    lr: float = 1e-2,
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
    adapt_lr: float = 1e-2,
) -> list[dict]:
    from src.datasets.registry import DatasetSpec, get_dataset

    if paradigm == "p300":
        if calib_sizes is None:
            calib_sizes = [24, 48, 96, 240, 480]   # 0.5/1/2/5/10 characters × 48 epochs
        spec = DatasetSpec(
            moabb_name="BNCI2014_009", paradigm="p300",
            channels=["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
            sfreq_target=128.0, exclude_subjects=[],
            band=(1.0, 24.0), epoch_window=(0.0, 0.8),
        )
    else:
        if calib_sizes is None:
            calib_sizes = [10, 20, 50, 100, 200]
        spec = DatasetSpec(
            moabb_name="PhysionetMI", paradigm="mi",
            channels=[
                "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP5", "CP3", "CP1",
            ],
            sfreq_target=160.0, exclude_subjects=[88, 92, 100],
            band=(8.0, 30.0), epoch_window=(0.0, 2.0),
        )

    wrapper = get_dataset(spec)
    test_subjects = wrapper.subject_list[-2:] if paradigm == "p300" else wrapper.subject_list[-10:]
    log.info("Test subjects for Stage 3: %s", test_subjects)

    ckpt_path = _find_best_checkpoint(log_dir, paradigm)
    pp_path = ckpt_path.parents[2] / "preprocessor.pkl"
    with open(pp_path, "rb") as f:
        pp = pickle.load(f)
    log.info("Preprocessor loaded from %s", pp_path)

    results: list[dict] = []

    for subj in test_subjects:
        log.info("Subject %s — loading epochs…", subj)
        X_all, y_all, _ = wrapper.load_epochs([subj])
        X_all = pp.transform(X_all)

        # Baseline: cross-subject model, no adaptation
        model_base = _load_model_frozen(ckpt_path, paradigm)
        model_base.eval()
        baseline_auc = _epoch_auc(model_base, X_all, y_all)
        log.info("  Subject %s baseline AUC = %.3f", subj, baseline_auc)

        results.append({
            "subject": subj,
            "calib_size": 0,
            "auc": baseline_auc,
            "adapted": False,
        })

        for n_calib in calib_sizes:
            if n_calib >= len(X_all) - 10:
                log.warning("  Skipping calib_size=%d (too large for subject %s)", n_calib, subj)
                continue

            X_calib = X_all[:n_calib]
            y_calib = y_all[:n_calib]
            X_eval = X_all[n_calib:]
            y_eval = y_all[n_calib:]

            if len(np.unique(y_eval)) < 2:
                log.warning("  Skipping calib_size=%d (eval set has only one class)", n_calib)
                continue

            # Fresh adapted model for each calib size (independent runs)
            model_adapted = _load_model_frozen(ckpt_path, paradigm)
            _adapt_head(model_adapted, X_calib, y_calib,
                        n_epochs=n_adapt_epochs, lr=adapt_lr)

            adapted_auc = _epoch_auc(model_adapted, X_eval, y_eval)
            log.info(
                "  Subject %s | calib=%3d | adapted AUC=%.3f (Δ=%.3f)",
                subj, n_calib, adapted_auc, adapted_auc - baseline_auc,
            )
            results.append({
                "subject": subj,
                "calib_size": n_calib,
                "auc": adapted_auc,
                "delta_auc": adapted_auc - baseline_auc,
                "adapted": True,
            })

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
    parser.add_argument("--adapt-lr", type=float, default=1e-2)
    args = parser.parse_args()

    results = calibration_curve(
        paradigm=args.paradigm,
        log_dir=args.log_dir,
        calib_sizes=args.calib_sizes,
        n_adapt_epochs=args.adapt_epochs,
        adapt_lr=args.adapt_lr,
    )

    paradigm_label = "MI (Stage 1)" if args.paradigm == "mi" else "P300 (Stage 2)"
    print(f"\n{'='*62}")
    print(f"STAGE 3 ADAPTATION — {paradigm_label}")
    print(f"{'='*62}")
    print(f"{'Subject':<10} {'Calib N':>8} {'AUC':>8} {'ΔAUC':>8}")
    print(f"{'-'*62}")
    for r in results:
        delta = f"{r.get('delta_auc', 0.0):>+.3f}" if r["adapted"] else "baseline"
        print(f"{r['subject']:<10} {r['calib_size']:>8} {r['auc']:>8.3f} {delta:>8}")

    # Summary: mean ΔAUC at each calib size
    adapted = [r for r in results if r["adapted"]]
    if adapted:
        import collections
        by_size: dict[int, list[float]] = collections.defaultdict(list)
        for r in adapted:
            by_size[r["calib_size"]].append(r.get("delta_auc", 0.0))
        print(f"\n{'Calib N':>8} {'Mean ΔAUC':>12}")
        print("-" * 22)
        for size in sorted(by_size):
            mean_d = np.mean(by_size[size])
            print(f"{size:>8} {mean_d:>+.3f}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
