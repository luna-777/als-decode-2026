"""Round C.2 — recovery-target analysis. No new runs.

Round A showed apparent adaptation gain rises monotonically with montage damage.
Two mechanisms predict that equally well:

* **Repair.** Adaptation partly undoes the damage, so adapted AUC should climb
  toward the undamaged k=0 baseline as N grows.
* **Headroom.** The damaged model sits lower on the AUC scale, so any
  zero-mean-ish perturbation has more room to appear helpful; adapted AUC would
  then plateau well below the k=0 baseline.

The discriminator is the asymptote. Adapted AUC is fitted against 1/N per k —
linear in the regressor, two parameters, which is as much as four calibration
sizes support — and the intercept is the estimated N→∞ asymptote.

Usage
-----
    python scripts/analyze_recovery.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

K_ORDER = ["perm_k0", "perm_k2", "perm_k4", "perm_k8", "perm_k17",
           "preaudit_hardcoded"]
K_LABEL = {"perm_k0": "k=0", "perm_k2": "k=2", "perm_k4": "k=4",
           "perm_k8": "k=8", "perm_k17": "k=17",
           "preaudit_hardcoded": "historical"}


def fit_asymptote(n_vals, auc_vals):
    """OLS of adapted AUC on 1/N. Returns (asymptote, slope, r2)."""
    n = np.asarray(n_vals, float)
    a = np.asarray(auc_vals, float)
    X = np.column_stack([np.ones(len(n)), 1.0 / n])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ beta
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2 = 1.0 - float(((a - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(beta[0]), float(beta[1]), r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="experiments/degradation/mi_montage_degradation.csv")
    ap.add_argument("--out", default="experiments/degradation/recovery_summary.csv")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.results, newline=""))
            if r["status"] == "ok"]

    # per (arm, condition, N, subject): mean adapted / baseline over folds+seeds
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["split"], r["montage_condition"], int(r["calib_size"]))
        acc[key][r["subject"]].append(
            (float(r["adapted_auc"]), float(r["baseline_auc"])))
    per = {k: {s: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
               for s, v in d.items()} for k, d in acc.items()}

    out = []
    for arm in sorted({k[0] for k in per}):
        sizes = sorted({k[2] for k in per if k[0] == arm})
        k0_base = np.mean([b for _, b in per[(arm, "perm_k0", sizes[0])].values()])

        print("=" * 100)
        print(f"ARM = {arm}   —   undamaged (k=0) baseline AUC = {k0_base:.4f}")
        print("=" * 100)
        print(f"{'condition':<13} {'base':>7} " +
              " ".join(f"{('adapt N='+str(n)):>12}" for n in sizes) +
              f" {'asymptote':>10} {'gap to k=0':>11} {'r2':>6}")
        print("-" * 100)

        for cond in K_ORDER:
            keys = [(arm, cond, n) for n in sizes if (arm, cond, n) in per]
            if len(keys) < 3:
                continue
            ns = [k[2] for k in keys]
            subs = sorted(set.intersection(*[set(per[k]) for k in keys]))
            base = np.mean([per[keys[0]][s][1] for s in subs])
            adapt_mean = [np.mean([per[k][s][0] for s in subs]) for k in keys]

            asy, slope, r2 = fit_asymptote(ns, adapt_mean)

            # bootstrap the asymptote over subjects
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(args.boot):
                pick = rng.choice(subs, size=len(subs), replace=True)
                am = [np.mean([per[k][s][0] for s in pick]) for k in keys]
                boots.append(fit_asymptote(ns, am)[0])
            lo, hi = np.percentile(boots, [2.5, 97.5])

            print(f"{K_LABEL[cond]:<13} {base:>7.4f} " +
                  " ".join(f"{v:>12.4f}" for v in adapt_mean) +
                  f" {asy:>10.4f} {asy - k0_base:>+11.4f} {r2:>6.3f}")
            print(f"{'':<13} {'':>7} " + " ".join(f"{'':>12}" for _ in ns) +
                  f" [{lo:.4f},{hi:.4f}]")
            out.append({
                "split": arm, "montage_condition": cond, "n_subjects": len(subs),
                "baseline_auc": round(float(base), 6),
                "k0_baseline_auc": round(float(k0_base), 6),
                **{f"adapted_N{n}": round(float(v), 6) for n, v in zip(ns, adapt_mean)},
                "asymptote": round(asy, 6),
                "asymptote_ci_lo": round(float(lo), 6),
                "asymptote_ci_hi": round(float(hi), 6),
                "gap_to_k0_baseline": round(asy - float(k0_base), 6),
                # 0 = adaptation recovers nothing (pure headroom: the asymptote sits
                # at the damaged baseline); 1 = full repair (asymptote reaches the
                # undamaged baseline). The point estimate is the discriminator; the
                # asymptote CI from four calibration sizes is too wide to settle it.
                "recovery_fraction": (round((asy - float(base)) / (float(k0_base) - float(base)), 4)
                                      if abs(float(k0_base) - float(base)) > 1e-6 else ""),
                "recovers_k0": bool(lo <= k0_base <= hi),
                "fit_r2": round(r2, 4),
            })
        print()

    print("=" * 100)
    print("RECOVERY FRACTION — the discriminator")
    print("  0 = asymptote sits at the damaged baseline (pure headroom, no repair)")
    print("  1 = asymptote reaches the undamaged k=0 baseline (full repair)")
    print("  The asymptote CI from four sizes is too wide to settle this by itself, so")
    print("  the point estimates and their ordering carry the argument.")
    print("=" * 100)
    print(f"  {'arm':<16} {'cond':<12} {'damaged':>8} {'asympt':>8} {'k=0':>8} "
          f"{'recovered':>10} {'CI incl k=0':>12}")
    for o in out:
        rf = o["recovery_fraction"]
        print(f"  {o['split']:<16} {K_LABEL[o['montage_condition']]:<12} "
              f"{o['baseline_auc']:>8.4f} {o['asymptote']:>8.4f} "
              f"{o['k0_baseline_auc']:>8.4f} "
              f"{(f'{rf:.0%}' if isinstance(rf, float) else '—'):>10} "
              f"{('yes' if o['recovers_k0'] else 'no'):>12}")

    if out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for o in out for k in o})
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(out)
        print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
