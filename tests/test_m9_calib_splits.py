"""Stage 3 calibration split protocols. Phase 1.2.

temporal_chrono and stratified_task_only decide the paper's mechanism claim, so
they carry the heaviest coverage here.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.splits import SPLIT_ARMS, make_calibration_split
from src.datasets.wrapper import chronological_order


def _physionet_like(n_task=180, n_baseline=60, seed=0):
    """Array layout matching MoabbDatasetWrapper: task runs first, baseline last.

    Chronologically the baseline runs (EDF 1, 2) come FIRST, and the task runs are
    keyed by MOABB out of acquisition order (4, 8, 12, 6, 10, 14).
    """
    rng = np.random.default_rng(seed)
    moabb_key_runs = [4, 8, 12, 6, 10, 14]
    per_run = n_task // 6

    y, source, run, within = [], [], [], []
    for r in moabb_key_runs:
        for i in range(per_run):
            y.append(int(rng.integers(0, 2)))
            source.append("task")
            run.append(r)
            within.append(i)
    for r in (1, 2):
        for i in range(n_baseline // 2):
            y.append(0)
            source.append("baseline")
            run.append(r)
            within.append(i)

    y = np.array(y)
    source = np.array(source)
    run = np.array(run)
    within = np.array(within)
    subjects = np.zeros(len(y), dtype=int)
    order = chronological_order(subjects, run, within)
    return y, source, order


# --------------------------------------------------------------------------- order

def test_chronological_order_puts_baseline_first():
    y, source, order = _physionet_like()
    chrono = np.argsort(order)
    # first epochs in acquisition order are baseline-run idle
    assert set(source[chrono[:30]]) == {"baseline"}
    # last are task
    assert set(source[chrono[-30:]]) == {"task"}


def test_chronological_order_is_a_permutation():
    _, _, order = _physionet_like()
    assert sorted(order.tolist()) == list(range(len(order)))


def test_array_order_is_not_chronological():
    """The whole point of recording `order` separately."""
    _, _, order = _physionet_like()
    assert not np.array_equal(order, np.arange(len(order)))


def test_moabb_key_order_is_not_run_order():
    """Task runs 4,8,12 then 6,10,14 — chronology must reorder them."""
    y, source, order = _physionet_like()
    task = np.flatnonzero(source == "task")
    # In array order the 2nd task run is EDF 8; chronologically EDF 6 precedes it.
    chrono_task = task[np.argsort(order[task])]
    assert not np.array_equal(task, chrono_task)


def test_order_is_per_subject():
    subjects = np.array([1, 1, 1, 2, 2, 2])
    runs = np.array([4, 1, 2, 2, 4, 1])
    within = np.zeros(6, dtype=int)
    order = chronological_order(subjects, runs, within)
    assert sorted(order[subjects == 1].tolist()) == [0, 1, 2]
    assert sorted(order[subjects == 2].tolist()) == [0, 1, 2]


# --------------------------------------------------------------------------- arms

@pytest.mark.parametrize("arm", SPLIT_ARMS)
def test_calib_and_eval_are_disjoint(arm):
    y, source, order = _physionet_like()
    s = make_calibration_split(arm, y, 50, 42, order, source)
    assert len(np.intersect1d(s.calib_idx, s.eval_idx)) == 0


@pytest.mark.parametrize("arm", SPLIT_ARMS)
def test_calib_size_is_exact(arm):
    y, source, order = _physionet_like()
    s = make_calibration_split(arm, y, 50, 42, order, source)
    assert len(s.calib_idx) == 50


def test_temporal_array_takes_the_literal_first_n():
    y, source, order = _physionet_like()
    s = make_calibration_split("temporal_array", y, 40, 42, order, source)
    np.testing.assert_array_equal(s.calib_idx, np.arange(40))
    # array order is task-first, so no baseline epochs land in calibration
    assert np.all(source[s.calib_idx] == "task")


def test_temporal_chrono_takes_baseline_first():
    """The decisive property: chronologically, baseline runs come first."""
    y, source, order = _physionet_like(n_baseline=60)
    s = make_calibration_split("temporal_chrono", y, 40, 42, order, source)
    assert np.all(source[s.calib_idx] == "baseline")
    # and it is NOT the array prefix
    assert not np.array_equal(s.calib_idx, np.arange(40))


def test_temporal_chrono_is_the_reverse_of_temporal_array():
    """The two temporal arms must load opposite sub-populations into calibration."""
    y, source, order = _physionet_like()
    a = make_calibration_split("temporal_array", y, 40, 42, order, source)
    c = make_calibration_split("temporal_chrono", y, 40, 42, order, source)
    frac_a = np.mean(source[a.calib_idx] == "baseline")
    frac_c = np.mean(source[c.calib_idx] == "baseline")
    assert frac_a == 0.0
    assert frac_c == 1.0


def test_temporal_chrono_calibration_is_contiguous_in_time():
    y, source, order = _physionet_like()
    s = make_calibration_split("temporal_chrono", y, 55, 42, order, source)
    pos = np.sort(order[s.calib_idx])
    np.testing.assert_array_equal(pos, np.arange(55))


def test_stratified_preserves_class_ratio():
    y, source, order = _physionet_like()
    s = make_calibration_split("stratified", y, 60, 42, order, source)
    assert abs(y[s.calib_idx].mean() - y.mean()) < 0.06


def test_stratified_is_seed_deterministic():
    y, source, order = _physionet_like()
    a = make_calibration_split("stratified", y, 50, 7, order, source)
    b = make_calibration_split("stratified", y, 50, 7, order, source)
    np.testing.assert_array_equal(a.calib_idx, b.calib_idx)


def test_stratified_varies_with_seed():
    y, source, order = _physionet_like()
    a = make_calibration_split("stratified", y, 50, 1, order, source)
    b = make_calibration_split("stratified", y, 50, 2, order, source)
    assert not np.array_equal(a.calib_idx, b.calib_idx)


def test_stratified_draws_from_both_sub_populations():
    y, source, order = _physionet_like()
    s = make_calibration_split("stratified", y, 80, 42, order, source)
    frac = np.mean(source[s.calib_idx] == "baseline")
    assert 0.0 < frac < 1.0


# --------------------------------------------------------------------------- mechanism control

def test_stratified_task_only_never_draws_baseline():
    """The mechanism control: calibration covers task epochs only."""
    y, source, order = _physionet_like()
    s = make_calibration_split("stratified_task_only", y, 60, 42, order, source)
    assert np.all(source[s.calib_idx] == "task")
    assert np.mean(source[s.calib_idx] == "baseline") == 0.0


def test_stratified_task_only_leaves_evaluation_set_intact():
    """Evaluation must stay comparable to `stratified` — it is everything not drawn.

    If this arm also shrank the evaluation set to task epochs, a difference against
    `stratified` could be explained by the evaluation denominator rather than by
    calibration coverage, and the arm would answer nothing.
    """
    y, source, order = _physionet_like()
    strat = make_calibration_split("stratified", y, 60, 42, order, source)
    task = make_calibration_split("stratified_task_only", y, 60, 42, order, source)
    assert len(task.eval_idx) == len(strat.eval_idx) == len(y) - 60
    # baseline epochs are still present in the evaluation set
    assert np.mean(source[task.eval_idx] == "baseline") > 0


def test_stratified_task_only_differs_from_stratified():
    y, source, order = _physionet_like()
    a = make_calibration_split("stratified", y, 60, 42, order, source)
    b = make_calibration_split("stratified_task_only", y, 60, 42, order, source)
    assert not np.array_equal(a.calib_idx, b.calib_idx)


def test_stratified_task_only_raises_when_pool_too_small():
    y, source, order = _physionet_like(n_task=30, n_baseline=200)
    with pytest.raises(ValueError, match="pool of"):
        make_calibration_split("stratified_task_only", y, 60, 42, order, source)


# --------------------------------------------------------------------------- purging

def test_purged_removes_temporal_neighbours():
    y, source, order = _physionet_like()
    plain = make_calibration_split("stratified", y, 50, 42, order, source)
    purged = make_calibration_split("purged_stratified", y, 50, 42, order, source, purge_k=5)
    assert len(purged.eval_idx) < len(plain.eval_idx)
    assert purged.n_purged == len(plain.eval_idx) - len(purged.eval_idx)


def test_purged_leaves_no_eval_epoch_within_k():
    y, source, order = _physionet_like()
    k = 5
    s = make_calibration_split("purged_stratified", y, 40, 42, order, source, purge_k=k)
    cal = order[s.calib_idx]
    for i in s.eval_idx:
        assert np.all(np.abs(cal - order[i]) > k)


def test_purge_k_zero_keeps_everything():
    y, source, order = _physionet_like()
    plain = make_calibration_split("stratified", y, 50, 42, order, source)
    s = make_calibration_split("purged_stratified", y, 50, 42, order, source, purge_k=0)
    assert len(s.eval_idx) == len(plain.eval_idx)
    assert s.n_purged == 0


def test_larger_k_purges_more():
    y, source, order = _physionet_like()
    a = make_calibration_split("purged_stratified", y, 40, 42, order, source, purge_k=2)
    b = make_calibration_split("purged_stratified", y, 40, 42, order, source, purge_k=10)
    assert b.n_purged > a.n_purged


def test_purged_calibration_matches_stratified():
    """Purging only touches the evaluation side."""
    y, source, order = _physionet_like()
    a = make_calibration_split("stratified", y, 50, 42, order, source)
    b = make_calibration_split("purged_stratified", y, 50, 42, order, source)
    np.testing.assert_array_equal(a.calib_idx, b.calib_idx)


# --------------------------------------------------------------------------- describe / errors

def test_describe_reports_composition():
    y, source, order = _physionet_like()
    s = make_calibration_split("temporal_chrono", y, 40, 42, order, source)
    d = s.describe(y, source)
    assert d["split"] == "temporal_chrono"
    assert d["n_calib"] == 40
    assert d["calib_baseline_frac"] == 1.0
    assert 0.0 <= d["eval_baseline_frac"] <= 1.0
    assert 0.0 <= d["eval_pos_rate"] <= 1.0


def test_unknown_arm_raises():
    y, source, order = _physionet_like()
    with pytest.raises(ValueError, match="Unknown split arm"):
        make_calibration_split("nope", y, 10, 42, order, source)


def test_n_calib_too_large_raises():
    y, source, order = _physionet_like()
    with pytest.raises(ValueError, match="only"):
        make_calibration_split("stratified", y, len(y) + 1, 42, order, source)
