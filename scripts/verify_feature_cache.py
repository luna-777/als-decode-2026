"""Assert the frozen-backbone feature cache is equivalent to running end-to-end.

The Stage 3 runner trains the head on cached backbone outputs instead of forwarding
epochs through the backbone on every step. That is only legitimate if it changes
nothing. This script runs one real condition both ways against the same checkpoint
and preprocessor and compares.

Baseline AUC involves no training and must agree to floating-point noise. Adapted
AUC additionally requires the two head-training loops to see the same batches, so
both are seeded identically before training.

Usage
-----
    python scripts/verify_feature_cache.py --fold version_28 --subject 99
"""
from __future__ import annotations

import argparse
import copy
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

TOL = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="version_28")
    ap.add_argument("--subject", type=int, default=99)
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--arm", default="stratified")
    ap.add_argument("--calib-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapt-epochs", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    args = ap.parse_args()

    import dataclasses

    from scripts.adapt_stage3 import (
        _adapt_head,
        _adapt_head_on_features,
        _apply_ea,
        _epoch_auc,
        _features,
        _head_auc,
        _load_model_frozen,
    )
    from src.datasets.registry import get_dataset
    from src.datasets.splits import make_calibration_split
    from src.models.checkpoints import load_fold, spec_from_config

    spec = spec_from_config("mi")
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    spec = dataclasses.replace(spec, euclidean_alignment=False)

    ref = load_fold(args.log_dir, args.fold)
    wrapper = get_dataset(spec)
    X_noea, y, meta = wrapper.load_epochs([args.subject])

    X = _apply_ea(X_noea, fit_idx=None)  # --ea-ref session
    with open(ref.preprocessor_path, "rb") as f:
        pp = pickle.load(f)
    X = pp.transform(X)

    split = make_calibration_split(
        args.arm, y, args.calib_size, args.seed,
        np.asarray(meta["order"]), np.asarray(meta["source"]),
    )
    y_calib = np.asarray(y)[split.calib_idx]
    y_eval = np.asarray(y)[split.eval_idx]
    X_calib = X[split.calib_idx]
    X_eval = X[split.eval_idx]

    model = _load_model_frozen(ref.ckpt_path, "mi", n_channels, n_times, reinit_head=False)
    model.eval()

    # ---------------------------------------------------------------- cached path
    feats = _features(model, X)
    base_head = copy.deepcopy(model.head)
    base_head.eval()
    cached_baseline = _head_auc(base_head, feats[split.eval_idx], y_eval)

    cached_head = copy.deepcopy(model.head)
    _adapt_head_on_features(
        cached_head, feats[split.calib_idx], y_calib,
        n_epochs=args.adapt_epochs, lr=args.adapt_lr, seed=args.seed,
    )
    cached_adapted = _head_auc(cached_head, feats[split.eval_idx], y_eval)

    # ------------------------------------------------------------------ live path
    live_base = _load_model_frozen(ref.ckpt_path, "mi", n_channels, n_times, reinit_head=False)
    live_base.eval()
    live_baseline = _epoch_auc(live_base, X_eval, y_eval)

    live_model = _load_model_frozen(ref.ckpt_path, "mi", n_channels, n_times, reinit_head=False)
    torch.manual_seed(args.seed)
    _adapt_head(live_model, X_calib, y_calib, n_epochs=args.adapt_epochs, lr=args.adapt_lr)
    live_adapted = _epoch_auc(live_model, X_eval, y_eval)

    # --------------------------------------------------------------------- report
    print("\n" + "=" * 72)
    print("FEATURE CACHE EQUIVALENCE")
    print(f"  fold={args.fold} subject={args.subject} arm={args.arm} "
          f"N={args.calib_size} seed={args.seed}")
    print(f"  n_calib={len(split.calib_idx)} n_eval={len(split.eval_idx)}")
    print("=" * 72)
    print(f"{'quantity':<18} {'cached':>12} {'live':>12} {'|diff|':>12}")
    print("-" * 58)
    dbase = abs(cached_baseline - live_baseline)
    dadapt = abs(cached_adapted - live_adapted)
    print(f"{'baseline AUC':<18} {cached_baseline:>12.9f} {live_baseline:>12.9f} {dbase:>12.2e}")
    print(f"{'adapted AUC':<18} {cached_adapted:>12.9f} {live_adapted:>12.9f} {dadapt:>12.2e}")
    print(f"{'delta AUC':<18} {cached_adapted - cached_baseline:>12.9f} "
          f"{live_adapted - live_baseline:>12.9f} "
          f"{abs((cached_adapted-cached_baseline)-(live_adapted-live_baseline)):>12.2e}")
    print("=" * 72)

    ok = True
    if dbase > TOL:
        print(f"FAIL: baseline AUC differs by {dbase:.3e} > {TOL:.0e}")
        ok = False
    if dadapt > TOL:
        print(f"FAIL: adapted AUC differs by {dadapt:.3e} > {TOL:.0e}")
        ok = False
    if ok:
        print(f"PASS: both agree within {TOL:.0e}. The cache is equivalent.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
