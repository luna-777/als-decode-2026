"""Round B — does baseline-idle coverage predict ΔAUC independently of class ratio?

The Phase 2 mechanism reading (docs/AUDIT.md §0.9) is that calibration sets which
*cover the baseline-run idle sub-population* adapt better. But baseline-run epochs
are all label 0, so drawing more of them mechanically lowers the calibration
positive rate. Coverage and class balance are therefore confounded by
construction, and the mechanism claim only stands if coverage predicts ΔAUC with
class ratio held fixed.

This regresses, on the existing Phase 2 rows (no re-run):

    delta_auc ~ 1 + calib_baseline_frac + calib_pos_rate

per (arm, size) where the arm has variation in coverage, and pooled across arms at
each size where it does not. Subject fixed effects are included in the pooled fit
so the comparison is within-subject.

Collinearity is reported, not assumed away: if the two regressors are close to
collinear the design cannot separate them, and the correct output is that
statement rather than a coefficient. Variance inflation factors and the raw
correlation are printed for every fit.

Usage
-----
    python scripts/analyze_confound.py --results experiments/stage3/mi_results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

VIF_ALARM = 10.0
CORR_ALARM = 0.95


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (beta, se, r2) for y ~ X (X must already include an intercept column)."""
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - p
    if dof <= 0:
        return beta, np.full(p, np.nan), float("nan")
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2


def _vif(X: np.ndarray, j: int) -> float:
    """VIF of column j regressed on the others, with an intercept in the auxiliary fit.

    Without that intercept the auxiliary R² can go negative and VIF comes out below
    1, which is impossible and hides the collinearity it is meant to expose.
    """
    others = [c for c in range(X.shape[1]) if c != j]
    if not others:
        return 1.0
    A = np.column_stack([np.ones(len(X)), X[:, others]])
    _, _, r2 = _ols(X[:, j], A)
    return float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)


def _t_p(beta, se):
    from scipy import stats

    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return t, 2 * (1 - stats.norm.cdf(np.abs(t)))


def load(path):
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r.get("status") == "ok" and r.get("ea_ref") == "session"]


def aggregate(rows):
    """Collapse fold and seed replicates to one observation per (subject, arm, size).

    Rows differing only in fold or calibration seed are repeated measurements of the
    same subject under the same condition, not independent samples. Regressing on
    the raw rows would divide the standard errors by ~sqrt(25) and manufacture
    significance. Aggregation first is the honest n.
    """
    acc = defaultdict(list)
    for r in rows:
        acc[(r["subject"], r["split"], int(r["calib_size"]))].append(r)
    out = []
    for (subj, arm, N), g in acc.items():
        out.append({
            "subject": subj, "split": arm, "calib_size": str(N),
            "delta_auc": str(np.mean([float(x["delta_auc"]) for x in g])),
            "calib_baseline_frac": str(np.mean([float(x["calib_baseline_frac"]) for x in g])),
            "calib_pos_rate": str(np.mean([float(x["calib_pos_rate"]) for x in g])),
            "n_replicates": len(g),
        })
    return out


def cluster_ols(rows, label):
    """Pooled fit on raw rows with subject FE and subject-clustered (CR1) SEs.

    The strongest form of the test: it exploits the within-arm seed jitter that
    subject-level aggregation averages away, while still respecting the fact that
    fold and seed replicates within a subject are dependent.
    """
    from scipy import stats

    y = np.array([float(r["delta_auc"]) for r in rows])
    bf = np.array([float(r["calib_baseline_frac"]) for r in rows])
    pr = np.array([float(r["calib_pos_rate"]) for r in rows])
    subs = sorted({r["subject"] for r in rows})
    cols = [np.ones(len(y)), bf, pr] + [
        np.array([1.0 if r["subject"] == s else 0.0 for r in rows]) for s in subs[1:]
    ]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)

    gidx = defaultdict(list)
    for i, r in enumerate(rows):
        gidx[r["subject"]].append(i)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in gidx.values():
        s = X[g].T @ resid[g]
        meat += np.outer(s, s)
    G = len(gidx)
    cov = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    corr = float(np.corrcoef(bf, pr)[0, 1])

    print(f"  {label:<26} n={len(y):<5} clusters={G}  r(bf,pr)={corr:+.4f}")
    print(f"      calib_baseline_frac  β={beta[1]:+.4f}  se={se[1]:.4f}  p={p[1]:.4f}")
    print(f"      calib_pos_rate       β={beta[2]:+.4f}  se={se[2]:.4f}  p={p[2]:.4f}")
    return {"condition": f"clustered {label}", "n": len(y), "corr_bf_pr": round(corr, 4),
            "vif_baseline_frac": "", "vif_pos_rate": "", "r2": "",
            "separable": abs(corr) < CORR_ALARM,
            "beta_baseline_frac": round(float(beta[1]), 6),
            "se_baseline_frac": round(float(se[1]), 6),
            "p_baseline_frac": round(float(p[1]), 6),
            "beta_pos_rate": round(float(beta[2]), 6),
            "se_pos_rate": round(float(se[2]), 6),
            "p_pos_rate": round(float(p[2]), 6)}


def fit_and_report(rows, label, subject_fe=False):
    """One regression. Returns a summary dict or None if not estimable."""
    y = np.array([float(r["delta_auc"]) for r in rows])
    bf = np.array([float(r["calib_baseline_frac"]) for r in rows])
    pr = np.array([float(r["calib_pos_rate"]) for r in rows])

    if np.std(bf) < 1e-9:
        print(f"  {label:<44} coverage is constant ({bf[0]:.3f}) — not estimable")
        return None

    cols = [np.ones(len(y)), bf, pr]
    names = ["intercept", "calib_baseline_frac", "calib_pos_rate"]
    if subject_fe:
        subs = sorted({r["subject"] for r in rows})
        for s in subs[1:]:
            cols.append(np.array([1.0 if r["subject"] == s else 0.0 for r in rows]))
            names.append(f"subj_{s}")
    X = np.column_stack(cols)

    corr = float(np.corrcoef(bf, pr)[0, 1])
    beta, se, r2 = _ols(y, X)
    t, p = _t_p(beta, se)
    vif_bf = _vif(X[:, 1:3] if not subject_fe else X[:, 1:], 0)
    vif_pr = _vif(X[:, 1:3] if not subject_fe else X[:, 1:], 1)

    sep = abs(corr) < CORR_ALARM and max(vif_bf, vif_pr) < VIF_ALARM
    print(f"  {label:<44} n={len(y):<5} r(bf,pr)={corr:+.3f}  "
          f"VIF={vif_bf:.1f}/{vif_pr:.1f}  R²={r2:.3f}  "
          f"{'SEPARABLE' if sep else 'COLLINEAR — not separable'}")
    print(f"      calib_baseline_frac  β={beta[1]:+.4f} (se {se[1]:.4f}, p={p[1]:.4f})")
    print(f"      calib_pos_rate       β={beta[2]:+.4f} (se {se[2]:.4f}, p={p[2]:.4f})")
    return {
        "condition": label, "n": len(y), "corr_bf_pr": round(corr, 4),
        "vif_baseline_frac": round(vif_bf, 3), "vif_pos_rate": round(vif_pr, 3),
        "r2": round(r2, 4), "separable": sep,
        "beta_baseline_frac": round(float(beta[1]), 6),
        "se_baseline_frac": round(float(se[1]), 6),
        "p_baseline_frac": round(float(p[1]), 6),
        "beta_pos_rate": round(float(beta[2]), 6),
        "se_pos_rate": round(float(se[2]), 6),
        "p_pos_rate": round(float(p[2]), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/stage3/mi_results.csv")
    ap.add_argument("--out", default="experiments/stage3/mi_confound_regression.csv")
    args = ap.parse_args()

    raw = load(args.results)
    # purge_k splits purged_stratified into distinct conditions; keep only the
    # specified default so it is not triple-counted in pooled fits.
    raw = [r for r in raw
           if r["split"] != "purged_stratified" or int(r.get("purge_k", 0) or 0) == 5]
    rows = aggregate(raw)
    log.info("%d raw rows -> %d subject-level observations", len(raw), len(rows))
    sizes = sorted({int(r["calib_size"]) for r in rows})
    out = []

    print("\n" + "=" * 96)
    print("WITHIN-ARM: does coverage vary at all inside an arm?")
    print("=" * 96)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["split"], int(r["calib_size"]))].append(r)
    for key in sorted(by_arm):
        arm, N = key
        res = fit_and_report(by_arm[key], f"{arm} N={N}")
        if res:
            out.append(res)

    print("\n" + "=" * 96)
    print("POOLED ACROSS ARMS at each size, with subject fixed effects")
    print("(this is where coverage genuinely varies — between arms, not within them)")
    print("=" * 96)
    by_size = defaultdict(list)
    for r in rows:
        by_size[int(r["calib_size"])].append(r)
    for N in sizes:
        res = fit_and_report(by_size[N], f"ALL ARMS pooled N={N}", subject_fe=True)
        if res:
            out.append(res)

    print("\n" + "=" * 96)
    print("POOLED, raw rows, subject FE, subject-CLUSTERED standard errors")
    print("=" * 96)
    by_size_raw = defaultdict(list)
    for r in raw:
        by_size_raw[int(r["calib_size"])].append(r)
    for N in sizes:
        out.append(cluster_ols(by_size_raw[N], f"N={N}"))

    print("\n" + "=" * 96)
    print("COVERAGE / CLASS-RATIO GRID — where the confound comes from")
    print("=" * 96)
    print(f"{'arm':<22} {'N':>5} {'calib_baseline_frac':>20} {'calib_pos_rate':>16} {'meanΔ':>9}")
    print("-" * 96)
    for key in sorted(by_arm):
        arm, N = key
        g = by_arm[key]
        print(f"{arm:<22} {N:>5} "
              f"{np.mean([float(r['calib_baseline_frac']) for r in g]):>20.3f} "
              f"{np.mean([float(r['calib_pos_rate']) for r in g]):>16.3f} "
              f"{np.mean([float(r['delta_auc']) for r in g]):>+9.4f}")

    if out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        log.info("Wrote %s", p)


if __name__ == "__main__":
    main()
