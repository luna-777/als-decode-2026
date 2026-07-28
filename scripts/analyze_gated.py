"""Round C.3 analysis — realised gain and acceptance rate under validation gating.

Two numbers per condition:

* **ΔAUC (deployed)** — the gain of the head that would actually be used. Exactly 0
  whenever the gate rejects, so this is a realised-benefit figure, not a
  best-case one.
* **acceptance rate** — how often a patient's own calibration data supports
  adapting at all. This is a result in its own right: a gate that almost never
  fires is telling you the method does not transfer, and a gate that fires often
  but yields no gain is telling you the internal validation split is too small to
  discriminate.

For comparison the ungated ΔAUC from the Phase 2 grid is the relevant contrast:
gating can only help if rejecting is better than adapting, which is exactly what
the Phase 2 negative means predict.

Usage
-----
    python scripts/analyze_gated.py --results experiments/roundc/*_gated.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/roundc/gated_summary.csv")
    args = ap.parse_args()

    rows = []
    for p in args.results:
        rows += list(csv.DictReader(open(p, newline="")))

    from scipy.stats import wilcoxon

    out = []
    keyf = lambda r: (r["paradigm"], int(r.get("permute_k", 0) or 0),
                      r["split"], int(r["calib_size"]))
    groups = defaultdict(list)
    for r in rows:
        groups[keyf(r)].append(r)

    print(f"{'cohort':<8} {'k':>3} {'arm':<16} {'N':>5} {'nsub':>5} "
          f"{'accept%':>8} {'ΔAUC(dep)':>10} {'95% CI':>20} {'p':>8} {'status':<22}")
    print("-" * 120)
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        cohort, k, arm, N = key
        g = groups[key]
        statuses = {r["status"] for r in g}
        ok = [r for r in g if r["status"] == "ok"]

        # subject-level means over folds and seeds
        per = defaultdict(list)
        for r in g:
            per[r["subject"]].append(r)
        subs = sorted(per)

        if not ok:
            st = sorted(statuses)[0]
            print(f"{cohort:<8} {k:>3} {arm:<16} {N:>5} {len(subs):>5} "
                  f"{'—':>8} {0.0:>+10.4f} {'—':>20} {'—':>8} {st:<22}")
            out.append({"paradigm": cohort, "permute_k": k, "split": arm,
                        "calib_size": N, "n_subjects": len(subs),
                        "acceptance_rate": "", "mean_delta_deployed": 0.0,
                        "ci_lo": "", "ci_hi": "", "p_two_sided": "",
                        "status": st, "n_ungateable": len(g)})
            continue

        acc_rate = float(np.mean([int(r["accepted"]) for r in ok]))
        dep = np.array([np.mean([float(x["delta_auc"]) for x in per[s]]) for s in subs])
        lo, hi = boot_ci(dep)
        try:
            _, p2 = wilcoxon(dep) if np.any(dep != 0) else (np.nan, np.nan)
        except ValueError:
            p2 = float("nan")
        st = "ok" if statuses == {"ok"} else "mixed:" + ",".join(sorted(statuses - {"ok"}))
        print(f"{cohort:<8} {k:>3} {arm:<16} {N:>5} {len(subs):>5} "
              f"{acc_rate:>7.1%} {dep.mean():>+10.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{p2:>8.4f} {st:<22}")
        out.append({"paradigm": cohort, "permute_k": k, "split": arm,
                    "calib_size": N, "n_subjects": len(subs),
                    "acceptance_rate": round(acc_rate, 4),
                    "mean_delta_deployed": round(float(dep.mean()), 6),
                    "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                    "p_two_sided": round(float(p2), 6) if np.isfinite(p2) else "",
                    "status": st,
                    "n_ungateable": sum(1 for r in g if r["status"] != "ok")})

    print()
    print("=" * 120)
    print("ACCEPTANCE RATE by cohort — how often calibration data supports adapting")
    print("=" * 120)
    by_cohort = defaultdict(list)
    for o in out:
        if o["acceptance_rate"] != "":
            by_cohort[(o["paradigm"], o["permute_k"])].append(o["acceptance_rate"])
    for kk in sorted(by_cohort):
        v = np.array(by_cohort[kk])
        print(f"  {kk[0]:<8} k={kk[1]:<3} mean acceptance {v.mean():.1%} "
              f"(range {v.min():.1%}–{v.max():.1%}) over {len(v)} conditions")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
