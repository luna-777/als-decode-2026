"""Aggregation order and corrected statistics in scripts/analyze_stage3.py."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_stage3 import summarize


def _row(subject, delta, *, size=100, seed=42, fold="version_28", split="stratified",
         ea_ref="session", status="ok", paradigm="mi"):
    return {
        "paradigm": paradigm, "subject": str(subject), "split": split,
        "calib_size": str(size), "seed": str(seed), "ea_ref": ea_ref,
        "delta_auc": str(delta), "fold": fold, "status": status,
    }


def test_folds_are_averaged_not_treated_as_subjects():
    """4 folds x 3 subjects must give n_subjects=3, not 12."""
    rows = []
    for subj in (1, 2, 3):
        for fold in ("version_27", "version_28", "version_29", "version_30"):
            rows.append(_row(subj, 0.05, fold=fold))
    s = summarize(rows)[0]
    assert s["n_subjects"] == 3
    assert s["n_folds"] == 4


def test_fold_average_is_the_mean_of_fold_deltas():
    rows = [
        _row(1, 0.00, fold="version_27"),
        _row(1, 0.10, fold="version_28"),
        _row(2, 0.20, fold="version_27"),
        _row(2, 0.40, fold="version_28"),
    ]
    s = summarize(rows)[0]
    # subject 1 -> 0.05, subject 2 -> 0.30, mean 0.175
    assert s["mean_delta_auc"] == pytest.approx(0.175)


def test_seeds_are_averaged_not_treated_as_subjects():
    rows = [_row(subj, 0.05, seed=sd) for subj in (1, 2) for sd in (42, 43, 44, 45, 46)]
    s = summarize(rows)[0]
    assert s["n_subjects"] == 2
    assert s["n_seeds"] == 5


def test_seed_and_fold_collapse_together():
    rows = [
        _row(subj, 0.1, seed=sd, fold=f)
        for subj in (1, 2, 3)
        for sd in (42, 43)
        for f in ("version_27", "version_28")
    ]
    s = summarize(rows)[0]
    assert s["n_subjects"] == 3
    assert s["n_seeds"] == 2
    assert s["n_folds"] == 2


def test_arms_are_kept_separate():
    rows = [_row(1, 0.1, split="stratified"), _row(1, -0.1, split="temporal_array")]
    out = {s["split"]: s for s in summarize(rows)}
    assert set(out) == {"stratified", "temporal_array"}
    assert out["stratified"]["mean_delta_auc"] == pytest.approx(0.1)
    assert out["temporal_array"]["mean_delta_auc"] == pytest.approx(-0.1)


def test_ea_refs_are_kept_separate():
    rows = [_row(1, 0.1, ea_ref="session"), _row(1, 0.02, ea_ref="calibration")]
    out = {s["ea_ref"]: s for s in summarize(rows)}
    assert set(out) == {"session", "calibration"}


# --------------------------------------------------------------------------- §0.5 corrections

def test_negative_mean_keeps_its_sign():
    """The n=10 defect: a small negative mean was published as positive."""
    deltas = [-0.02, 0.01, -0.03, 0.02, -0.01, 0.015, -0.025, 0.005, -0.005, 0.01]
    rows = [_row(i, d, size=10) for i, d in enumerate(deltas)]
    s = summarize(rows)[0]
    assert s["mean_delta_auc"] < 0
    assert s["mean_delta_auc"] == pytest.approx(np.mean(deltas), abs=1e-6)


def test_two_sided_p_comes_from_scipy_not_from_doubling():
    """The mislabelling defect: p_two_sided was scipy's two-sided p, doubled again.

    The published table reported 0.0077 at n=200 where the true two-sided p is
    0.0039 — its `p_one_sided` column held the two-sided value and `p_two_sided`
    was twice that. Note that p_two == 2 * p_one is the *correct* relationship for
    this null, so the test cannot check for its absence; it checks that we emit
    scipy's two-sided value and not that value doubled.
    """
    deltas = [0.09, 0.08, 0.10, 0.07, 0.11, 0.06, 0.12, 0.05, 0.13, -0.01]
    rows = [_row(i, d, size=200) for i, d in enumerate(deltas)]
    s = summarize(rows)[0]
    from scipy.stats import wilcoxon

    _, p2 = wilcoxon(np.array(deltas), alternative="two-sided")
    _, p1 = wilcoxon(np.array(deltas), alternative="greater")
    assert s["p_two_sided"] == pytest.approx(p2, abs=1e-6)
    assert s["p_one_sided_greater"] == pytest.approx(p1, abs=1e-6)
    # the defect would have emitted 2 * p2 in the two-sided column
    assert s["p_two_sided"] != pytest.approx(2 * p2, abs=1e-9)


def test_ci_brackets_the_mean():
    rows = [_row(i, d) for i, d in enumerate([0.01, 0.05, -0.02, 0.08, 0.03])]
    s = summarize(rows)[0]
    assert s["ci95_lo"] <= s["mean_delta_auc"] <= s["ci95_hi"]


def test_n_improved_counts_positive_subjects():
    rows = [_row(i, d) for i, d in enumerate([0.1, -0.1, 0.2, -0.2, 0.3])]
    s = summarize(rows)[0]
    assert s["n_improved"] == 3


# --------------------------------------------------------------------------- degenerate rows

def test_degenerate_rows_excluded_but_counted():
    rows = [_row(1, 0.1), _row(2, 0.2)]
    rows.append(_row(3, float("nan"), status="degenerate_eval_single_class"))
    s = summarize(rows)[0]
    assert s["n_subjects"] == 2
    assert s["n_degenerate"] == 1


def test_all_degenerate_still_emits_a_row():
    """A fully degenerate condition must appear in the table, not vanish from it."""
    rows = [_row(i, float("nan"), size=200, status="degenerate_eval_single_class")
            for i in range(3)]
    out = summarize(rows)
    assert len(out) == 1
    assert out[0]["n_subjects"] == 0
    assert out[0]["n_degenerate"] == 3
    assert out[0]["mean_delta_auc"] == ""


def test_nan_delta_counted_as_degenerate():
    rows = [_row(1, 0.1), _row(2, "nan")]
    s = summarize(rows)[0]
    assert s["n_subjects"] == 1
    assert s["n_degenerate"] == 1
