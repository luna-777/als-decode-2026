"""Stage 3 figures — IEEE two-column, vector output, colourblind-safe.

Sizing: 3.5 in column width (IEEE single column), 7.16 in for full width. Text is
set so that at final print size body labels are ~7 pt and tick labels ~6 pt, which
stays legible at 3.5 in without rescaling.

Palette: Okabe-Ito subset, ordered so that no adjacent pair collides under
deuteranopia/protanopia. Validated with the dataviz skill's checker — all CVD and
lightness checks pass; the two lightest hues fall below 3:1 contrast against the
surface, which is why every series is additionally direct-labelled and given its
own marker. Colour is never the only channel carrying identity.

Usage
-----
    python scripts/make_figures.py --mi experiments/stage3/mi_results.csv \
        --p300 experiments/stage3/p300_results.csv --out results/figures
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

COL_W, FULL_W = 3.5, 7.16

# Okabe-Ito, reordered so adjacent pairs stay separable under CVD.
ARM_COLOR = {
    "temporal_array":       "#0072B2",
    "temporal_chrono":      "#D55E00",
    "stratified":           "#009E73",
    "stratified_task_only": "#E69F00",
    "purged_stratified":    "#CC79A7",
}
ARM_MARKER = {
    "temporal_array": "o", "temporal_chrono": "s", "stratified": "^",
    "stratified_task_only": "D", "purged_stratified": "v",
}
ARM_LABEL = {
    "temporal_array": "temporal (array)",
    "temporal_chrono": "temporal (chrono)",
    "stratified": "stratified",
    "stratified_task_only": "stratified, task-only",
    "purged_stratified": "purged stratified",
}
ARM_ORDER = list(ARM_COLOR)

INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#d6d6d6"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def subject_means(rows, ea_ref="session", purge_k=None):
    """(arm, size) -> {subject: delta}, folds then seeds collapsed. Mirrors analyze_stage3."""
    per = defaultdict(lambda: defaultdict(list))  # (arm,size) -> subj -> [delta per fold*seed]
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("ea_ref", "session") != ea_ref:
            continue
        if purge_k is not None and r["split"] == "purged_stratified":
            if int(r.get("purge_k", 0) or 0) != purge_k:
                continue
        d = _f(r["delta_auc"])
        if d is None:
            continue
        per[(r["split"], int(r["calib_size"]))][str(r["subject"])].append(d)
    return {k: {s: float(np.mean(v)) for s, v in m.items()} for k, m in per.items()}


def boot_ci(vals, n=10000, seed=0):
    v = np.asarray(vals, float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------- fig 1

def fig_pipeline(out: Path):
    """Schematic locating the calibration-split decision in the pipeline."""
    fig, ax = plt.subplots(figsize=(FULL_W, 2.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, w, y, h, label, sub="", fc="#f2f2f2", ec=INK2, lw=0.7, bold=False):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), label, ha="center",
                va="center", fontsize=6.4, color=INK, zorder=3,
                fontweight="bold" if bold else "normal")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 3.0, sub, ha="center", va="center",
                    fontsize=5.4, color=INK2, zorder=3)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=0.7,
                                     color=INK2, zorder=1))

    y, h = 18, 11
    box(0, 15, y, h, "MOABB", "raw epochs")
    arrow(15, 19, y + h / 2)
    box(19, 16, y, h, "montage", "config, asserted")
    arrow(35, 39, y + h / 2)
    box(39, 15, y, h, "EA", "session / calib")
    arrow(54, 58, y + h / 2)
    box(58, 16, y, h, "preprocessor", "fold-fitted")
    arrow(74, 78, y + h / 2)
    box(78, 22, y, h, "frozen backbone", "features cached")

    # the decision
    box(30, 40, 1, 11, "CALIBRATION SPLIT", "5 arms × size × seed", fc="#eaf2f8",
        ec="#0072B2", lw=1.1, bold=True)
    ax.add_patch(FancyArrowPatch((50, 12), (50, 18), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=1.0,
                                 color="#0072B2", zorder=1))
    ax.text(51.5, 14.6, "decides which epochs calibrate", fontsize=5.4,
            color="#0072B2", va="center")

    ax.add_patch(FancyArrowPatch((89, 18), (89, 12), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=0.7, color=INK2))
    box(78, 22, 1, 11, "head adaptation", "AUC on eval set")

    ax.text(0, 37, "Stage 3 per-patient adaptation", fontsize=7.5,
            fontweight="bold", color=INK)
    ax.text(0, 33, "The split arm is the experimental variable; everything "
            "upstream is held fixed.", fontsize=5.6, color=INK2)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"fig1_pipeline.{ext}")
    plt.close(fig)
    log.info("fig1_pipeline")


# --------------------------------------------------------------------------- fig 2

def fig_delta_vs_size(rows, out: Path, paradigm="mi", ea_ref="session", purge_k=5):
    sm = subject_means(rows, ea_ref=ea_ref, purge_k=purge_k)
    sizes = sorted({k[1] for k in sm})
    fig, ax = plt.subplots(figsize=(COL_W, 2.65))
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=0.6, zorder=1)

    for arm in ARM_ORDER:
        xs, ys, los, his = [], [], [], []
        for s in sizes:
            v = list(sm.get((arm, s), {}).values())
            if len(v) < 2:
                continue
            lo, hi = boot_ci(v)
            xs.append(s); ys.append(float(np.mean(v))); los.append(lo); his.append(hi)
        if not xs:
            continue
        c = ARM_COLOR[arm]
        ax.fill_between(xs, los, his, color=c, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=c, marker=ARM_MARKER[arm], markersize=3,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3,
                label=ARM_LABEL[arm])
        ax.annotate(ARM_LABEL[arm], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(3, 0), fontsize=5.2, color=c, va="center", zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("calibration epochs (N)")
    ax.set_ylabel(r"$\Delta$AUC (adapted $-$ baseline)")
    ax.set_title(f"{paradigm.upper()}: adaptation gain by split arm",
                 loc="left", fontweight="bold")
    ax.set_xlim(min(sizes) * 0.85, max(sizes) * 1.9)
    ax.legend(loc="upper left", ncol=1, handlelength=1.4, borderpad=0.2,
              labelspacing=0.25)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"fig2_delta_vs_size_{paradigm}.{ext}")
    plt.close(fig)
    log.info("fig2_delta_vs_size_%s", paradigm)


# --------------------------------------------------------------------------- fig 3

def fig_per_subject(rows, out: Path, size, paradigm="mi", ea_ref="session"):
    sm = subject_means(rows, ea_ref=ea_ref, purge_k=5)
    a, b = "stratified", "temporal_array"
    A, B = sm.get((a, size), {}), sm.get((b, size), {})
    subs = sorted(set(A) | set(B), key=lambda s: -A.get(s, -np.inf))
    if not subs:
        log.warning("fig3: no data at N=%s", size)
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=0.6, zorder=1)
    x = np.arange(len(subs)); w = 0.38

    for off, arm, src in ((-w / 2 - 0.01, a, A), (w / 2 + 0.01, b, B)):
        vals = [src.get(s, np.nan) for s in subs]
        ax.bar(x + off, vals, width=w, color=ARM_COLOR[arm], linewidth=0,
               zorder=3, label=ARM_LABEL[arm])

    missing = [i for i, s in enumerate(subs) if s not in B or not np.isfinite(B.get(s, np.nan))]
    if missing:
        ax.text(0.5, 0.04, f"{ARM_LABEL[b]} unmeasurable at N={size} "
                f"(single-class evaluation set)", transform=ax.transAxes,
                ha="center", fontsize=5.2, color=ARM_COLOR[b], style="italic")

    ax.set_xticks(x); ax.set_xticklabels(subs, rotation=0)
    ax.set_xlabel("subject"); ax.set_ylabel(r"$\Delta$AUC")
    ax.set_title(f"{paradigm.upper()}: per-subject gain at N={size}",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", handlelength=1.2, borderpad=0.2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"fig3_per_subject_N{size}_{paradigm}.{ext}")
    plt.close(fig)
    log.info("fig3_per_subject_N%s_%s", size, paradigm)


# --------------------------------------------------------------------------- fig 4

def fig_eval_composition(rows, out: Path, paradigm="mi", ea_ref="session"):
    """Fraction of the evaluation set that is baseline-run idle — the mechanism figure."""
    per = defaultdict(list)
    status = {}
    for r in rows:
        if r.get("ea_ref", "session") != ea_ref:
            continue
        if r["split"] == "purged_stratified" and int(r.get("purge_k", 0) or 0) != 5:
            continue
        key = (r["split"], int(r["calib_size"]))
        f = _f(r.get("eval_baseline_frac"))
        if r.get("status", "ok") != "ok":
            status.setdefault(key, r["status"])
        if f is not None:
            per[key].append(f)

    sizes = sorted({k[1] for k in set(per) | set(status)})
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    _style(ax)
    x = np.arange(len(sizes)); n = len(ARM_ORDER); w = 0.82 / n

    for i, arm in enumerate(ARM_ORDER):
        off = (i - (n - 1) / 2) * w
        vals, hatch_at = [], []
        for j, s in enumerate(sizes):
            v = per.get((arm, s))
            if v:
                vals.append(float(np.mean(v)) * 100)
            else:
                vals.append(0.0); hatch_at.append(j)
        ax.bar(x + off, vals, width=w * 0.9, color=ARM_COLOR[arm], linewidth=0,
               zorder=3, label=ARM_LABEL[arm])
        for j in hatch_at:
            ax.plot(x[j] + off, 2, marker="x", markersize=3, color=ARM_COLOR[arm],
                    markeredgewidth=0.8, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("calibration epochs (N)")
    ax.set_ylabel("evaluation set that is\nbaseline-run idle (%)")
    ax.set_title(f"{paradigm.upper()}: evaluation-set composition",
                 loc="left", fontweight="bold")
    ax.set_ylim(0, 105)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="x", linestyle="none", color=INK2,
                          markersize=3, markeredgewidth=0.8))
    labels.append("unmeasurable")
    ax.legend(handles, labels, loc="upper left", ncol=2, handlelength=1.2,
              borderpad=0.2, columnspacing=0.8, labelspacing=0.25)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"fig4_eval_composition_{paradigm}.{ext}")
    plt.close(fig)
    log.info("fig4_eval_composition_%s", paradigm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mi", default="experiments/stage3/mi_results.csv")
    ap.add_argument("--p300", default="experiments/stage3/p300_results.csv")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--mi-size", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_pipeline(out)

    if Path(args.mi).exists():
        mi = load(args.mi)
        fig_delta_vs_size(mi, out, "mi")
        fig_per_subject(mi, out, args.mi_size, "mi")
        fig_eval_composition(mi, out, "mi")
    else:
        log.warning("missing %s", args.mi)

    if Path(args.p300).exists():
        p3 = load(args.p300)
        fig_delta_vs_size(p3, out, "p300")
    else:
        log.warning("missing %s", args.p300)

    log.info("figures → %s", out)


if __name__ == "__main__":
    main()
