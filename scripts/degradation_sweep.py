"""Round A — montage degradation dose-response.

docs/AUDIT.md §0.1/§0.6 gave one point: the historical montage cost 0.0586 of
held-out AUC, and repairing it made the reported adaptation gain disappear. The
obvious question is whether those two facts are the same fact — whether apparent
adaptation gain is a monotone function of how badly the backbone's input is
scrambled. This sweeps the dose.

Conditions
----------
* ``k`` channel positions permuted, k ∈ {0, 2, 4, 8, 17}, 3 permutation seeds each
  (k=0 is the identity, so it has one condition, not three). The channel *set* is
  held fixed; only electrode-to-position assignment changes.
* ``preaudit`` — the exact historical hardcoded list. This is **not** a permutation
  (it swaps FC5/FC6/CP5 in for CPz/CP2/CP4), so it needs a genuine reload and is
  reported separately. It is the only condition that can answer "does the sweep
  recover +0.0885?" literally.

Output always lands in experiments/degradation/. The montage contract is asserted
on every load; the degradation is applied to verified data afterwards and recorded
per row (ADR-24).

Usage
-----
    python scripts/degradation_sweep.py --folds version_27 version_28 version_29 version_30
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

OUT_DIR = ROOT / "experiments" / "degradation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+",
                    default=["version_27", "version_28", "version_29", "version_30"])
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 2, 4, 8, 17])
    ap.add_argument("--permute-seeds", nargs="+", type=int, default=[101, 102, 103])
    ap.add_argument("--splits", nargs="+", default=["stratified", "temporal_array"])
    ap.add_argument("--calib-sizes", nargs="+", type=int, default=[10, 50, 100, 200])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--adapt-epochs", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--no-preaudit", action="store_true",
                    help="Skip the historical-montage reference condition.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.adapt_stage3 import _features, _load_model_frozen, _run_one
    from src.datasets.degradation import (
        PREAUDIT_MI_MONTAGE,
        apply_permutation,
        n_displaced,
        permutation_for_k,
        permuted_channel_names,
    )
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.datasets.registry import get_dataset
    from src.models.checkpoints import load_folds, spec_from_config

    spec = spec_from_config("mi")
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    clean_channels = list(spec.channels)
    spec_noea = dataclasses.replace(spec, euclidean_alignment=False)

    refs = load_folds(args.log_dir, args.folds, expect_channels=n_channels)
    wrapper = get_dataset(spec_noea)
    subjects = wrapper.subject_list[-10:]

    # (label, k, permute_seed, perm | None) — perm None means "reload this montage"
    conditions: list[tuple] = []
    for k in args.ks:
        seeds = [args.permute_seeds[0]] if k == 0 else args.permute_seeds
        for ps in seeds:
            conditions.append((f"perm_k{k}", k, ps, permutation_for_k(n_channels, k, ps)))
    if not args.no_preaudit:
        conditions.append(("preaudit_hardcoded", -1, -1, None))

    log.info("Conditions: %s", [c[0] for c in conditions])
    log.info("Folds: %s | subjects: %s", [r.version for r in refs], subjects)

    results: list[dict] = []
    for subj in subjects:
        # Verified load under the config montage; every degradation is derived from it.
        X_clean, y, meta = wrapper.load_epochs([subj])
        order, source = np.asarray(meta["order"]), np.asarray(meta["source"])

        # The historical montage swapped electrodes, so it needs its own load.
        X_pre = None
        if not args.no_preaudit:
            spec_pre = dataclasses.replace(spec_noea, channels=list(PREAUDIT_MI_MONTAGE))
            X_pre, y_pre, _ = get_dataset(spec_pre).load_epochs([subj])
            assert np.array_equal(y, y_pre), "label mismatch between montage loads"

        for ref in refs:
            montage = load_montage(ref.version_dir, spec, n_times)
            # Contract still enforced: the *load* must match what was trained on.
            assert_montage_matches(montage, meta["channels"],
                                   where=f"{ref.version} vs subject {subj}")
            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)
            model = _load_model_frozen(ref.ckpt_path, "mi", n_channels, n_times,
                                       reinit_head=False)
            model.eval()

            for label, k, ps, perm in conditions:
                if perm is None:
                    X_deg = X_pre
                    eff_channels = list(PREAUDIT_MI_MONTAGE)
                    disp = 17
                else:
                    # Permute BEFORE the preprocessor: its per-channel scalers are
                    # positional, and that mismatch is part of the defect.
                    X_deg = apply_permutation(X_clean, perm)
                    eff_channels = permuted_channel_names(clean_channels, perm)
                    disp = n_displaced(perm)

                from scripts.adapt_stage3 import _apply_ea
                feats = _features(model, pp.transform(_apply_ea(X_deg, fit_idx=None)))

                for arm in args.splits:
                    for N in args.calib_sizes:
                        for sd in args.seeds:
                            row = _run_one(
                                model=model, pp=pp, X_noea=X_deg, y_all=y,
                                order=order, source=source, arm=arm, n_calib=N,
                                seed=sd, ea_ref="session", purge_k=5,
                                feats_session=feats,
                                n_adapt_epochs=args.adapt_epochs,
                                adapt_lr=args.adapt_lr,
                            )
                            if row is None:
                                continue
                            row.update({
                                "paradigm": "mi", "subject": subj,
                                "montage_condition": label,
                                "permute_k": k, "permute_seed": ps,
                                "n_displaced": disp,
                                "effective_montage": "|".join(eff_channels),
                                "calib_size": N, "seed": sd, "ea_ref": "session",
                                "fold": ref.version, "fold_mode": "average",
                                "checkpoint_val_auc": ref.val_auc,
                                "montage_source": montage.source,
                            })
                            results.append(row)
                log.info("  subj %s | %s | %-18s (disp=%2d) done", subj, ref.version, label, disp)

    out = Path(args.out) if args.out else OUT_DIR / "mi_montage_degradation.csv"
    if not out.resolve().is_relative_to(OUT_DIR.resolve()):
        raise SystemExit(f"Degradation output must live under {OUT_DIR}; got {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["paradigm", "subject", "montage_condition", "permute_k", "permute_seed",
              "n_displaced", "split", "calib_size", "seed", "ea_ref",
              "baseline_auc", "adapted_auc", "delta_auc", "n_calib", "n_eval",
              "calib_pos_rate", "eval_pos_rate", "calib_baseline_frac",
              "eval_baseline_frac", "n_purged", "status", "fold", "fold_mode",
              "checkpoint_val_auc", "montage_source", "effective_montage"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Wrote %d rows → %s", len(results), out)


if __name__ == "__main__":
    main()
