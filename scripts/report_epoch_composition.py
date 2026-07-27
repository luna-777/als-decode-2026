"""Epoch counts and evaluation-set composition per arm and size — no model involved.

Answers, before any adaptation run: how many epochs does each held-out subject
have, how do they divide into task and baseline-run sub-populations, and what does
each split arm leave in the evaluation set at each calibration size. This decides
whether a condition is measurable at all — notably temporal_array at N=200, where
the evaluation set is whatever is left at the end of the array.

Usage
-----
    python scripts/report_epoch_composition.py --paradigm mi
    python scripts/report_epoch_composition.py --paradigm mi --out experiments/audit/mi_composition.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", choices=["mi", "p300"], default="mi")
    ap.add_argument("--calib-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--purge-k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import dataclasses

    from src.datasets.registry import get_dataset
    from src.datasets.splits import SPLIT_ARMS, make_calibration_split
    from src.models.checkpoints import spec_from_config

    spec = spec_from_config(args.paradigm)
    spec = dataclasses.replace(spec, euclidean_alignment=False)  # composition only
    sizes = args.calib_sizes or (
        [24, 48, 96, 240, 480] if args.paradigm == "p300" else [10, 20, 50, 100, 200]
    )

    wrapper = get_dataset(spec)
    subjects = wrapper.subject_list[-10:]

    per_subject = []
    rows = []
    for subj in subjects:
        _, y, meta = wrapper.load_epochs([subj])
        source = np.asarray(meta["source"])
        order = np.asarray(meta["order"])
        run = np.asarray(meta["run"])
        n = len(y)
        n_task = int((source == "task").sum())
        n_base = int((source == "baseline").sum())
        per_subject.append({
            "subject": subj, "n_epochs": n, "n_task": n_task, "n_baseline": n_base,
            "baseline_frac": n_base / n,
            "n_pos": int(np.sum(y == 1)), "n_neg": int(np.sum(y == 0)),
            "pos_rate": float(np.mean(y == 1)),
            "runs": ",".join(str(r) for r in sorted(set(run.tolist()))),
        })

        for arm in SPLIT_ARMS:
            for N in sizes:
                base = {
                    "paradigm": args.paradigm, "subject": subj, "split": arm,
                    "calib_size": N, "n_epochs_total": n,
                }
                if N >= n:
                    rows.append({**base, "status": "impossible_n_calib_ge_n_epochs",
                                 "n_calib": N, "n_eval": 0, "calib_pos_rate": "",
                                 "eval_pos_rate": "", "calib_baseline_frac": "",
                                 "eval_baseline_frac": "", "n_purged": 0})
                    continue
                try:
                    s = make_calibration_split(
                        arm, y, N, args.seed, order, source, purge_k=args.purge_k
                    )
                except ValueError as e:
                    rows.append({**base, "status": f"impossible: {e}", "n_calib": N,
                                 "n_eval": 0, "calib_pos_rate": "", "eval_pos_rate": "",
                                 "calib_baseline_frac": "", "eval_baseline_frac": "",
                                 "n_purged": 0})
                    continue
                d = s.describe(y, source)
                y_eval = np.asarray(y)[s.eval_idx]
                status = "ok"
                if len(s.eval_idx) == 0:
                    status = "degenerate_empty_eval"
                elif len(np.unique(y_eval)) < 2:
                    status = "degenerate_eval_single_class"
                rows.append({**base, **d, "status": status})

    # ------------------------------------------------------------------ print
    print("\n" + "=" * 100)
    print(f"PER-SUBJECT EPOCH COUNTS — {args.paradigm.upper()}, held-out subjects")
    print("=" * 100)
    print(f"{'subj':>5} {'epochs':>7} {'task':>6} {'baseline':>9} {'base%':>7} "
          f"{'pos':>6} {'neg':>6} {'pos%':>7}  runs")
    print("-" * 100)
    for p in per_subject:
        print(f"{p['subject']:>5} {p['n_epochs']:>7} {p['n_task']:>6} {p['n_baseline']:>9} "
              f"{p['baseline_frac']*100:>6.1f}% {p['n_pos']:>6} {p['n_neg']:>6} "
              f"{p['pos_rate']*100:>6.1f}%  {p['runs']}")
    arr = np.array([p["n_epochs"] for p in per_subject])
    print("-" * 100)
    print(f"{'mean':>5} {arr.mean():>7.1f} "
          f"{np.mean([p['n_task'] for p in per_subject]):>6.1f} "
          f"{np.mean([p['n_baseline'] for p in per_subject]):>9.1f} "
          f"{np.mean([p['baseline_frac'] for p in per_subject])*100:>6.1f}%")
    print(f"  range: {arr.min()}–{arr.max()} epochs")

    print("\n" + "=" * 100)
    print("EVALUATION-SET COMPOSITION BY ARM AND SIZE (mean over subjects)")
    print("=" * 100)
    print(f"{'arm':<22} {'N':>5} {'n_eval':>8} {'eval base%':>11} {'eval pos%':>10} "
          f"{'calib base%':>12} {'purged':>7}  status")
    print("-" * 100)
    for arm in SPLIT_ARMS:
        for N in sizes:
            sel = [r for r in rows if r["split"] == arm and r["calib_size"] == N]
            oks = [r for r in sel if r["status"] == "ok"]
            bad = [r for r in sel if r["status"] != "ok"]
            if not oks:
                st = bad[0]["status"] if bad else "?"
                print(f"{arm:<22} {N:>5} {'-':>8} {'-':>11} {'-':>10} {'-':>12} {'-':>7}  "
                      f"{st} (all {len(sel)} subjects)")
                continue
            ne = np.mean([r["n_eval"] for r in oks])
            eb = np.mean([r["eval_baseline_frac"] for r in oks])
            ep = np.mean([r["eval_pos_rate"] for r in oks])
            cb = np.mean([r["calib_baseline_frac"] for r in oks])
            pg = np.mean([r["n_purged"] for r in oks])
            note = "ok" if not bad else f"{len(bad)}/{len(sel)} degenerate: {bad[0]['status']}"
            print(f"{arm:<22} {N:>5} {ne:>8.1f} {eb*100:>10.1f}% {ep*100:>9.1f}% "
                  f"{cb*100:>11.1f}% {pg:>7.1f}  {note}")
    print("=" * 100)

    out = Path(args.out or f"experiments/audit/{args.paradigm}_epoch_composition.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["paradigm", "subject", "split", "calib_size", "n_epochs_total", "n_calib",
              "n_eval", "calib_pos_rate", "eval_pos_rate", "calib_baseline_frac",
              "eval_baseline_frac", "n_purged", "status"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    subj_out = out.with_name(out.stem + "_per_subject.csv")
    with open(subj_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_subject[0].keys()))
        w.writeheader()
        w.writerows(per_subject)
    log.info("Wrote %s and %s", out, subj_out)


if __name__ == "__main__":
    main()
