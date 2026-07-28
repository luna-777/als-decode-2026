"""Round B orthogonalized — vary negative provenance with class ratio held fixed.

§0.11 found that baseline-run coverage and calibration class ratio are collinear at
r ≈ −1.0 across the five split arms, because baseline runs contain no positive-class
epochs. So the arms cannot say which of the two drives ΔAUC.

This breaks the collinearity by construction. N and the positive count are held
fixed; only the split of the *negatives* between the two provenances varies:

* **task-run negatives** — T0 rest intervals inside the imagined-movement runs
* **baseline-run negatives** — the eyes-open / eyes-closed runs

Coverage f = n_baseline / n_negatives sweeps {0, 0.25, 0.5, 0.75, 1.0} while
``calib_pos_rate`` stays constant at P/N. Any residual effect of f is attributable
to provenance and not to class balance.

Attainability is a hard constraint: a typical subject has 90 positives, 84 task-run
negatives and 60 baseline-run negatives, so large N leaves no room to reach the
extreme coverage levels. Cells are reported as unattainable rather than approximated
(see --report-only).

Usage
-----
    python scripts/orthogonal_negatives.py --report-only
    python scripts/orthogonal_negatives.py --folds version_27 ... version_41
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

POS_RATE = 90.0 / 234.0   # population positive rate; matches the `stratified` arm


def plan(n_pos_pool, n_task_neg_pool, n_base_neg_pool, N, f):
    """Return (n_pos, n_task_neg, n_base_neg) or None if unattainable."""
    P = int(round(N * POS_RATE))
    neg = N - P
    nb = int(round(f * neg))
    nt = neg - nb
    if P > n_pos_pool or nb > n_base_neg_pool or nt > n_task_neg_pool:
        return None
    return P, nt, nb


def draw(rng, pool, k):
    return rng.choice(pool, size=k, replace=False) if k else np.empty(0, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--calib-sizes", nargs="+", type=int, default=[50, 100, 150, 200])
    ap.add_argument("--coverage", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--adapt-epochs", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--report-only", action="store_true",
                    help="Print the attainability grid and exit without running.")
    ap.add_argument("--out", default="experiments/roundc/mi_orthogonal_negatives.csv")
    args = ap.parse_args()

    from scripts.adapt_stage3 import (
        _adapt_head_on_features,
        _apply_ea,
        _features,
        _head_auc,
        _load_model_frozen,
    )
    from src.datasets.montage import assert_montage_matches, load_montage
    from src.datasets.registry import get_dataset
    from src.models.checkpoints import load_folds, spec_from_config

    spec = spec_from_config("mi")
    n_times = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
    n_channels = len(spec.channels)
    wrapper = get_dataset(dataclasses.replace(spec, euclidean_alignment=False))
    subjects = wrapper.subject_list[-10:]

    if args.report_only:
        print("Attainability with the typical pool (90 pos / 84 task-neg / 60 base-neg):")
        print(f"{'N':>5} {'P':>4} {'neg':>5} " +
              " ".join(f"{('f='+str(f)):>14}" for f in args.coverage))
        for N in args.calib_sizes:
            cells = []
            for f in args.coverage:
                pl = plan(90, 84, 60, N, f)
                cells.append(f"{pl[2]}b/{pl[1]}t" if pl else "UNATTAINABLE")
            P = int(round(N * POS_RATE))
            print(f"{N:>5} {P:>4} {N - P:>5} " + " ".join(f"{c:>14}" for c in cells))
        return

    refs = load_folds(args.log_dir, args.folds or [], expect_channels=n_channels)

    rows = []
    for subj in subjects:
        X, y, meta = wrapper.load_epochs([subj])
        y = np.asarray(y)
        source = np.asarray(meta["source"])
        order = np.asarray(meta["order"])
        pos_pool = np.flatnonzero(y == 1)
        task_neg_pool = np.flatnonzero((y == 0) & (source == "task"))
        base_neg_pool = np.flatnonzero((y == 0) & (source == "baseline"))
        log.info("subject %s: pools pos=%d task_neg=%d base_neg=%d",
                 subj, len(pos_pool), len(task_neg_pool), len(base_neg_pool))

        for ref in refs:
            montage = load_montage(ref.version_dir, spec, n_times)
            assert_montage_matches(montage, meta["channels"],
                                   where=f"{ref.version} vs subject {subj}")
            with open(ref.preprocessor_path, "rb") as f:
                pp = pickle.load(f)
            model = _load_model_frozen(ref.ckpt_path, "mi", n_channels, n_times,
                                       reinit_head=False)
            model.eval()
            feats = _features(model, pp.transform(_apply_ea(X, fit_idx=None)))

            for N in args.calib_sizes:
                for f in args.coverage:
                    pl = plan(len(pos_pool), len(task_neg_pool), len(base_neg_pool), N, f)
                    if pl is None:
                        rows.append({
                            "paradigm": "mi", "subject": subj, "fold": ref.version,
                            "calib_size": N, "coverage_target": f, "seed": "",
                            "status": "unattainable", "n_pos": "", "n_task_neg": "",
                            "n_base_neg": "", "calib_pos_rate": "",
                            "calib_baseline_frac": "", "n_eval": "",
                            "baseline_auc": "", "adapted_auc": "", "delta_auc": "",
                        })
                        continue
                    P, nt, nb = pl
                    for sd in args.seeds:
                        rng = np.random.default_rng(sd * 1000 + N + int(f * 100))
                        cal = np.concatenate([
                            draw(rng, pos_pool, P),
                            draw(rng, task_neg_pool, nt),
                            draw(rng, base_neg_pool, nb),
                        ]).astype(int)
                        ev = np.setdiff1d(np.arange(len(y)), cal)
                        if len(np.unique(y[ev])) < 2 or len(np.unique(y[cal])) < 2:
                            continue
                        base = copy.deepcopy(model.head); base.eval()
                        b = _head_auc(base, feats[ev], y[ev])
                        ad = copy.deepcopy(model.head)
                        _adapt_head_on_features(ad, feats[cal], y[cal],
                                                n_epochs=args.adapt_epochs,
                                                lr=args.adapt_lr, seed=sd)
                        a = _head_auc(ad, feats[ev], y[ev])
                        rows.append({
                            "paradigm": "mi", "subject": subj, "fold": ref.version,
                            "calib_size": N, "coverage_target": f, "seed": sd,
                            "status": "ok", "n_pos": P, "n_task_neg": nt,
                            "n_base_neg": nb,
                            "calib_pos_rate": P / N,
                            "calib_baseline_frac": nb / N,
                            "n_eval": len(ev),
                            "baseline_auc": b, "adapted_auc": a, "delta_auc": a - b,
                        })
            log.info("  subject %s | %s done (%d rows)", subj, ref.version, len(rows))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("Wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
