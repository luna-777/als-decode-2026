"""Round A analysis — does adaptation gain track montage degradation?

Reports, per montage condition: mean held-out baseline AUC (how badly the backbone
is broken) and mean ΔAUC with bootstrap CI and improvement count (how much
head-only adaptation appears to help). If apparent adaptation gain is an artefact
of a mismatched backbone, the two should move together, and the historical montage
should sit near the pre-audit +0.0885.

Usage
-----
    python scripts/analyze_degradation.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PREAUDIT_DELTA = 0.0885   # stratified, N=200, mi_wilcoxon.csv (archived)
PREAUDIT_BASE = 0.6949    # mean baseline AUC over archived mi_results.csv


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def subject_level(rows, keys):
    """Collapse folds, permutation seeds and calibration seeds to per-subject means."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] != "ok":
            continue
        acc[tuple(r[k] for k in keys)][r["subject"]].append(r)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="experiments/degradation/mi_montage_degradation.csv")
    ap.add_argument("--out", default="experiments/degradation/degradation_summary.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.results, newline="")))
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(rows)} rows, {len(ok)} usable, "
          f"{len(rows) - len(ok)} degenerate\n")

    from scipy.stats import wilcoxon

    out = []
    for arm in sorted({r["split"] for r in ok}):
        for N in sorted({int(r["calib_size"]) for r in ok}):
            print("=" * 104)
            print(f"ARM = {arm}   N = {N}")
            print("=" * 104)
            print(f"{'condition':<22} {'disp':>5} {'base AUC':>9} {'meanΔ':>9} "
                  f"{'95% CI':>20} {'impr':>7} {'p2':>8}")
            print("-" * 104)
            sel = [r for r in ok if r["split"] == arm and int(r["calib_size"]) == N]
            groups = subject_level(sel, ["montage_condition"])
            # order: k ascending, then the historical montage last
            def sort_key(c):
                return (99, 0) if c[0] == "preaudit_hardcoded" else (
                    int([r for r in sel if r["montage_condition"] == c[0]][0]["permute_k"]), 0)
            for cond in sorted(groups, key=sort_key):
                per = groups[cond]
                base = np.array([np.mean([float(x["baseline_auc"]) for x in v])
                                 for v in per.values()])
                delt = np.array([np.mean([float(x["delta_auc"]) for x in v])
                                 for v in per.values()])
                disp = int([r for r in sel if r["montage_condition"] == cond[0]][0]["n_displaced"])
                lo, hi = boot_ci(delt)
                try:
                    _, p2 = wilcoxon(delt)
                except ValueError:
                    p2 = float("nan")
                print(f"{cond[0]:<22} {disp:>5} {base.mean():>9.4f} {delt.mean():>+9.4f} "
                      f"[{lo:+.4f},{hi:+.4f}] {int((delt > 0).sum()):>3}/{len(delt):<3} "
                      f"{p2:>8.4f}")
                out.append({
                    "split": arm, "calib_size": N, "montage_condition": cond[0],
                    "n_displaced": disp, "n_subjects": len(delt),
                    "mean_baseline_auc": round(float(base.mean()), 6),
                    "mean_delta_auc": round(float(delt.mean()), 6),
                    "ci95_lo": round(lo, 6), "ci95_hi": round(hi, 6),
                    "n_improved": int((delt > 0).sum()),
                    "p_two_sided": round(float(p2), 6),
                })
            print()

    # dose-response across k, for the arm/size that the paper's headline used
    print("=" * 104)
    print("DOSE-RESPONSE: baseline degradation vs apparent adaptation gain")
    print(f"reference — pre-audit published: baseline {PREAUDIT_BASE:.4f}, "
          f"ΔAUC {PREAUDIT_DELTA:+.4f} (stratified, N=200)")
    print("=" * 104)
    for arm in sorted({r["split"] for r in ok}):
        sub = [o for o in out if o["split"] == arm]
        if not sub:
            continue
        print(f"\n{arm}:")
        print(f"{'N':>5} " + " ".join(f"{c:>13}" for c in
              ["k=0", "k=2", "k=4", "k=8", "k=17", "preaudit"]))
        for N in sorted({o["calib_size"] for o in sub}):
            cells = []
            for lab in ["perm_k0", "perm_k2", "perm_k4", "perm_k8", "perm_k17",
                        "preaudit_hardcoded"]:
                m = [o for o in sub if o["calib_size"] == N and o["montage_condition"] == lab]
                cells.append(f"{m[0]['mean_delta_auc']:+.4f}" if m else "—")
            print(f"{N:>5} " + " ".join(f"{c:>13}" for c in cells))
        print(f"{'base':>5} " + " ".join(
            f"{np.mean([o['mean_baseline_auc'] for o in sub if o['montage_condition']==lab]):>13.4f}"
            if [o for o in sub if o["montage_condition"] == lab] else f"{'—':>13}"
            for lab in ["perm_k0", "perm_k2", "perm_k4", "perm_k8", "perm_k17",
                        "preaudit_hardcoded"]))

    # correlation between degradation and apparent gain
    print("\n" + "=" * 104)
    print("Correlation of mean ΔAUC with mean baseline AUC across montage conditions")
    print("=" * 104)
    from scipy.stats import pearsonr, spearmanr

    for arm in sorted({o["split"] for o in out}):
        for N in sorted({o["calib_size"] for o in out}):
            g = [o for o in out if o["split"] == arm and o["calib_size"] == N]
            if len(g) < 4:
                continue
            b = [o["mean_baseline_auc"] for o in g]
            d = [o["mean_delta_auc"] for o in g]
            r, pr = pearsonr(b, d)
            rho, ps = spearmanr(b, d)
            print(f"  {arm:<22} N={N:<5} n_conditions={len(g):<3} "
                  f"pearson r={r:+.3f} (p={pr:.4f})   spearman ρ={rho:+.3f} (p={ps:.4f})")

    if out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
