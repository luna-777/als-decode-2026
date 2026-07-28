"""Round C.1 analysis — does the baseline/ΔAUC correlation survive independence?

Compares two correlations across subjects, per calibration size (and per montage-k
for MI):

* **same-set** — ρ(baseline_B, Δ_B). Both terms come from the same epochs, so a
  subject whose baseline is underestimated has Δ = adapted − baseline
  correspondingly overestimated. This is the number §0.12 reported.
* **split-half** — ρ(baseline_A, Δ_B). Disjoint epochs, so no shared noise.

If the relationship is real, both are negative and comparable. If it was a
same-set artefact, the split-half correlation collapses toward zero.

p-values are exact permutation p at these sample sizes (8 ALS patients, 10 MI
subjects), since the asymptotic approximation is not valid there.

Usage
-----
    python scripts/analyze_splithalf.py --results experiments/roundc/als_splithalf.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import permutations
from math import factorial
from pathlib import Path

import numpy as np


def _rho(a, b):
    from scipy.stats import rankdata

    a, b = rankdata(a).astype(float), rankdata(b).astype(float)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def spearman_exact_or_mc(x, y, max_exact=9, n_mc=20000, seed=0):
    """Exact permutation p when n! is tractable, Monte-Carlo otherwise."""
    obs = _rho(x, y)
    n = len(x)
    yy = np.asarray(y, float)
    if factorial(n) <= factorial(max_exact):
        cnt = tot = 0
        for p in permutations(range(n)):
            tot += 1
            if abs(_rho(x, yy[list(p)])) >= abs(obs) - 1e-12:
                cnt += 1
        return obs, cnt / tot, "exact"
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_mc):
        if abs(_rho(x, rng.permutation(yy))) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_mc + 1), "monte-carlo"


def boot_ci_rho(x, y, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for _ in range(n_boot):
        i = rng.choice(len(x), size=len(x), replace=True)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        out.append(_rho(x[i], y[i]))
    if len(out) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.results, newline="")))
    ks = sorted({int(r["permute_k"]) for r in rows})
    sizes = sorted({int(r["calib_size"]) for r in rows})
    arms = sorted({r["split"] for r in rows})

    out = []
    for arm in arms:
        for k in ks:
            print("=" * 100)
            print(f"arm={arm}   permute_k={k}")
            print("=" * 100)
            print(f"{'N':>6} {'n':>4}  {'same-set ρ':>11} {'p':>8}   "
                  f"{'split-half ρ':>13} {'p':>8} {'95% CI':>18}  verdict")
            print("-" * 100)
            for N in sizes:
                sel = [r for r in rows if r["split"] == arm
                       and int(r["permute_k"]) == k and int(r["calib_size"]) == N]
                if not sel:
                    continue
                agg = defaultdict(list)
                for r in sel:
                    agg[r["subject"]].append(r)
                subs = sorted(agg)
                if len(subs) < 5:
                    continue
                bA = [np.mean([float(x["baseline_a"]) for x in agg[s]]) for s in subs]
                bB = [np.mean([float(x["baseline_b"]) for x in agg[s]]) for s in subs]
                dB = [np.mean([float(x["delta_b"]) for x in agg[s]]) for s in subs]

                r_same, p_same, _ = spearman_exact_or_mc(bB, dB)
                r_split, p_split, mode = spearman_exact_or_mc(bA, dB)
                lo, hi = boot_ci_rho(bA, dB)

                survives = (r_split < 0) and (hi < 0)
                verdict = ("SURVIVES" if survives else
                           "collapses" if abs(r_split) < abs(r_same) / 2
                           else "inconclusive (CI spans 0)")
                print(f"{N:>6} {len(subs):>4}  {r_same:>+11.3f} {p_same:>8.4f}   "
                      f"{r_split:>+13.3f} {p_split:>8.4f} "
                      f"[{lo:+.2f},{hi:+.2f}]  {verdict}")
                out.append({
                    "split": arm, "permute_k": k, "calib_size": N,
                    "n_subjects": len(subs), "p_mode": mode,
                    "rho_same_set": round(r_same, 4), "p_same_set": round(p_same, 5),
                    "rho_split_half": round(r_split, 4),
                    "p_split_half": round(p_split, 5),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "survives": survives,
                    "mean_baseline_a": round(float(np.mean(bA)), 4),
                    "mean_delta_b": round(float(np.mean(dB)), 4),
                })
            print()

    if out:
        p = Path(args.out or Path(args.results).with_name(
            Path(args.results).stem + "_summary.csv"))
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"Wrote {p}")
        n_sur = sum(1 for o in out if o["survives"])
        print(f"\nSurvives independent estimation in {n_sur}/{len(out)} conditions.")


if __name__ == "__main__":
    main()
