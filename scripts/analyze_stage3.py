"""Stage 3 statistics — the committed replacement for the ad-hoc analysis that
produced mi_wilcoxon.csv and p300_sign_test.csv.

Emits, per (paradigm, arm, calibration size): mean ΔAUC across subjects, a
bootstrap 95% CI, the number of subjects improved, the Wilcoxon signed-rank
statistic with its two-sided p, and a paired effect size.

Two defects in the previous ad-hoc output are corrected here (docs/AUDIT.md §0.5):

1. **Sign error.** mi_wilcoxon.csv reported mean_delta_auc = +0.0008 at n=10 where
   the per-subject values give −0.000759. The magnitude was right and the sign was
   dropped. Nothing here takes an absolute value of a signed mean.

2. **Mislabelled p-value columns.** The old table's `p_one_sided` column held the
   true two-sided p, and `p_two_sided` was that value doubled a second time — so
   the headline n=200 p was published as 0.0077 when it is 0.0039. This script
   emits exactly one p-value column, `p_two_sided`, taken directly from
   scipy.stats.wilcoxon's two-sided result. A one-sided p is reported separately
   and derived independently, never by halving or doubling the other.

Aggregation: when multiple seeds are present, each subject's ΔAUC is first
averaged across seeds, then the test is run over the per-subject means. Seeds are
not independent samples and must not inflate n.

Usage
-----
    python scripts/analyze_stage3.py --results experiments/stage3/mi_results.csv
    python scripts/analyze_stage3.py --results experiments/stage3/*.csv --out summary.csv
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

BOOTSTRAP_N = 10_000


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Returns (nan, nan) for n < 2."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def _effect_size(values: np.ndarray) -> float:
    """Paired Cohen's d_z = mean / sd of the per-subject differences."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float("nan")
    sd = v.std(ddof=1)
    return float(v.mean() / sd) if sd > 0 else float("nan")


def _wilcoxon(values: np.ndarray) -> tuple[float, float, float]:
    """Return (W, p_two_sided, p_one_sided_greater), computed independently.

    Each p comes from its own scipy call. Neither is derived from the other by
    doubling or halving — that conflation is exactly the §0.5 defect.
    """
    from scipy.stats import wilcoxon

    v = np.asarray(values, dtype=float)
    if v.size < 1 or np.all(v == 0):
        return float("nan"), float("nan"), float("nan")
    try:
        W, p2 = wilcoxon(v, alternative="two-sided")
        _, p1 = wilcoxon(v, alternative="greater")
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    return float(W), float(p2), float(p1)


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """One summary row per (paradigm, split, ea_ref, calib_size).

    Aggregation order, innermost first:

    1. **Across folds** — with ``--fold-mode average`` the runner writes one row per
       fold, each a complete independent adaptation (its own baseline, its own
       adapted head, its own evaluation). Their ``delta_auc`` values are averaged
       here per (subject, split, ea_ref, size, seed). This is an average of
       measurements, *not* a prediction ensemble: no logits or probabilities are
       ever combined.
    2. **Across seeds** — then averaged per subject, because seeds are re-draws of
       the same subject's calibration set and are not independent samples.
    3. **Across subjects** — only at this level does n enter the statistics.

    Rows whose ``status`` is not "ok" are degenerate conditions recorded by the
    runner with NaN metrics; they are excluded from the statistics but counted in
    ``n_degenerate`` so the table shows where they occurred.
    """
    # (key, subject, seed) -> [delta per fold]
    per_fold: dict[tuple, list[float]] = defaultdict(list)
    degenerate: dict[tuple, int] = defaultdict(int)

    for r in rows:
        key = (
            r.get("paradigm", "?"),
            r.get("split", "stratified"),
            r.get("ea_ref", "session"),
            int(r["calib_size"]),
        )
        if r.get("status", "ok") != "ok":
            degenerate[key] += 1
            continue
        delta = r.get("delta_auc")
        if delta in (None, "", "nan") or not np.isfinite(float(delta)):
            degenerate[key] += 1
            continue
        per_fold[(key, str(r["subject"]), str(r.get("seed", "")))].append(float(delta))

    # collapse folds -> (key, subject) -> [one value per seed]
    per_subject: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_folds_seen: dict[tuple, set] = defaultdict(set)
    for (key, subj, seed), fold_vals in per_fold.items():
        per_subject[key][subj].append(float(np.mean(fold_vals)))
        n_folds_seen[key].add(len(fold_vals))

    out = []
    for key in sorted(set(per_subject) | set(degenerate), key=lambda k: (k[0], k[1], k[2], k[3])):
        paradigm, split, ea_ref, size = key
        subj_map = per_subject.get(key, {})
        if not subj_map:
            out.append({
                "paradigm": paradigm, "split": split, "ea_ref": ea_ref,
                "calib_size": size, "n_subjects": 0, "n_seeds": 0, "n_folds": 0,
                "mean_delta_auc": "", "median_delta_auc": "", "ci95_lo": "",
                "ci95_hi": "", "n_improved": 0, "wilcoxon_W": "", "p_two_sided": "",
                "p_one_sided_greater": "", "cohens_dz": "",
                "n_degenerate": degenerate.get(key, 0),
            })
            continue
        subj_means = np.array([np.mean(v) for _, v in sorted(subj_map.items())], dtype=float)
        n_seeds = max(len(v) for v in subj_map.values())
        n_folds = max(n_folds_seen[key]) if n_folds_seen[key] else 0

        finite = subj_means[np.isfinite(subj_means)]
        if finite.size == 0:
            continue
        lo, hi = _bootstrap_ci(finite)
        W, p2, p1 = _wilcoxon(finite)

        out.append({
            "paradigm": paradigm,
            "split": split,
            "ea_ref": ea_ref,
            "calib_size": size,
            "n_subjects": int(finite.size),
            "n_seeds": int(n_seeds),
            "n_folds": int(n_folds),
            "mean_delta_auc": round(float(finite.mean()), 6),
            "median_delta_auc": round(float(np.median(finite)), 6),
            "ci95_lo": round(lo, 6),
            "ci95_hi": round(hi, 6),
            "n_improved": int((finite > 0).sum()),
            "wilcoxon_W": round(W, 4) if np.isfinite(W) else "",
            "p_two_sided": round(p2, 6) if np.isfinite(p2) else "",
            "p_one_sided_greater": round(p1, 6) if np.isfinite(p1) else "",
            "cohens_dz": round(_effect_size(finite), 4),
            "n_degenerate": degenerate.get(key, 0),
        })
    return out


FIELDS = [
    "paradigm", "split", "ea_ref", "calib_size", "n_subjects", "n_seeds", "n_folds",
    "mean_delta_auc", "median_delta_auc", "ci95_lo", "ci95_hi", "n_improved",
    "wilcoxon_W", "p_two_sided", "p_one_sided_greater", "cohens_dz", "n_degenerate",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="Stage 3 results CSV(s)")
    ap.add_argument("--out", default=None, help="Write summary CSV here")
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit("No rows loaded.")
    summary = summarize(rows)

    hdr = f"{'paradigm':<8} {'split':<22} {'ea':<6} {'N':>5} {'nsub':>5} {'meanΔ':>10} {'95% CI':>20} {'impr':>6} {'p2':>9} {'dz':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        ci = f"[{s['ci95_lo']:+.4f},{s['ci95_hi']:+.4f}]"
        p2 = s["p_two_sided"]
        print(
            f"{s['paradigm']:<8} {s['split']:<22} {s['ea_ref']:<6} {s['calib_size']:>5} "
            f"{s['n_subjects']:>5} {s['mean_delta_auc']:>+10.4f} {ci:>20} "
            f"{s['n_improved']:>3}/{s['n_subjects']:<2} {p2 if p2 == '' else f'{p2:>9.4f}'} "
            f"{s['cohens_dz']:>+7.3f}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(summary)
        log.info("Summary → %s", args.out)


if __name__ == "__main__":
    main()
