"""Discriminating test: is the config montage the one version_28 was trained on?

version_28 predates the montage contract (docs/AUDIT.md §0.4), so its electrode
order cannot be read back from the checkpoint. It can only be *inferred* from the
dataset config the training path read at the time — which is unambiguous, since
configs/dataset/physionet_mi.yaml has never been edited. This script tests that
inference empirically rather than accepting it on documentary grounds.

Method: evaluate the pretrained model on the 10 held-out MI subjects with **no
adaptation at all**, using the config's sensorimotor montage, EA applied as it
currently is. Compare the resulting mean baseline AUC against:

  * the published wrong-montage baseline (~0.68, from experiments/stage3/mi_results.csv)
  * the fold validation AUC range the checkpoint was selected under

If the correct montage is the config list, feeding the model correctly-ordered
electrodes should recover performance toward the range it achieved on validation
data. If the number does not move, the config list is not the montage that was
trained on, and Stage 1/2 must be retrained before anything else is believed.

Usage
-----
    python scripts/eval_heldout_mi.py --folds version_28
    python scripts/eval_heldout_mi.py --folds version_27 version_28 version_29 version_30
"""
from __future__ import annotations

import argparse
import csv
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


@torch.no_grad()
def _auc(model, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    probs = torch.sigmoid(model(torch.from_numpy(X).float()).squeeze(-1)).numpy()
    return float(roc_auc_score(y, probs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict-montage", action="store_true")
    ap.add_argument(
        "--montage-override", nargs="+", default=None,
        help="AUDIT CONTROL ONLY. Load data with this channel list instead of the "
             "config's, bypassing the montage assertion. Used to re-measure the "
             "pre-audit wrong-montage condition under an otherwise identical "
             "pipeline, so the montage is the only difference. Never use for results.",
    )
    args = ap.parse_args()

    from scripts.adapt_stage3 import _load_model_frozen
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.datasets.registry import get_dataset
    from src.models.checkpoints import load_folds, spec_from_config

    spec = spec_from_config("mi")
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    log.info("Config montage (%d ch): %s", len(spec.channels), list(spec.channels))

    montage_label = "config"
    if args.montage_override:
        import dataclasses

        montage_label = "override"
        # An overridden montage bypasses the ADR-14 assertion, so its output is not a
        # result and must never land where the paper's tables are read from.
        if args.out is None:
            args.out = "experiments/audit/mi_heldout_montage_override.csv"
        if Path(args.out).resolve().is_relative_to(
            (ROOT / "experiments" / "stage3").resolve()
        ):
            raise SystemExit(
                f"--montage-override may not write to experiments/stage3/ "
                f"(got {args.out}). Output from a bypassed montage assertion is audit "
                f"evidence, not a result; route it to experiments/audit/."
            )
        log.warning(
            "AUDIT CONTROL: overriding the montage with %s — the montage assertion "
            "is bypassed. This measures the pre-audit condition, not a result.",
            args.montage_override,
        )
        spec = dataclasses.replace(spec, channels=list(args.montage_override))

    n_channels = len(spec.channels)

    refs = load_folds(args.log_dir, args.folds, expect_channels=n_channels)
    wrapper = get_dataset(spec)
    subjects = wrapper.subject_list[-10:]
    log.info("Held-out subjects: %s", subjects)

    rows: list[dict] = []
    for subj in subjects:
        X_raw, y, meta = wrapper.load_epochs([subj])
        log.info("Subject %s: %d epochs, %d pos / %d neg",
                 subj, len(y), int(y.sum()), int((y == 0).sum()))

        for ref in refs:
            montage = load_montage(ref.version_dir, spec, n_times, strict=args.strict_montage)
            if args.montage_override:
                log.warning("  montage assertion skipped (audit control)")
            else:
                assert_montage_matches(
                    montage, meta["channels"], where=f"{ref.version} vs subject {subj}"
                )

            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)
            X = pp.transform(X_raw)

            model = _load_model_frozen(ref.ckpt_path, "mi", n_channels, n_times, reinit_head=False)
            model.eval()
            auc = _auc(model, X, y)
            log.info("  %s | subject %s | held-out AUC = %.4f", ref.version, subj, auc)
            rows.append({
                "paradigm": "mi",
                "subject": subj,
                "fold": ref.version,
                "checkpoint_val_auc": ref.val_auc,
                "heldout_auc": auc,
                "n_epochs": len(y),
                "n_pos": int(y.sum()),
                "montage_source": montage.source,
                "montage_label": montage_label,
                "n_channels": n_channels,
                "channels": "|".join(spec.channels),
            })

    out = Path(args.out or "experiments/stage3/mi_heldout_montage_check.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s", out)

    print("\n" + "=" * 72)
    print("HELD-OUT MI BASELINE AUC — config sensorimotor montage, no adaptation")
    print("=" * 72)
    for ref in refs:
        vals = np.array([r["heldout_auc"] for r in rows if r["fold"] == ref.version], dtype=float)
        vals = vals[np.isfinite(vals)]
        print(f"\n{ref.version} (fold val AUC {ref.val_auc:.4f})")
        for r in [r for r in rows if r["fold"] == ref.version]:
            print(f"    subject {r['subject']:>4}: {r['heldout_auc']:.4f}")
        print(f"    mean {vals.mean():.4f} | median {np.median(vals):.4f} "
              f"| min {vals.min():.4f} | max {vals.max():.4f}")

    allv = np.array([r["heldout_auc"] for r in rows], dtype=float)
    allv = allv[np.isfinite(allv)]
    print(f"\nOverall mean across {len(refs)} fold(s): {allv.mean():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
