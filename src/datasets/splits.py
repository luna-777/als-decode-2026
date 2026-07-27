"""Group-aware splitters — subjects are the grouping unit. §11.1.

No subject may appear in two splits. The split is enforced at the subject level
*before* sliding-window epoching so overlapping windows never straddle boundaries.

This module also holds the Stage 3 **calibration split protocols**. Which epochs a
patient's calibration set is drawn from turned out to be the experimental variable
in Stage 3, so every arm lives here, in one place, rather than inline in the script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


class GroupKFoldSplitter:
    """Group-aware k-fold cross-validation over subjects.

    Subjects in *held_out* are excluded from all folds (reserved for final test).
    Used for hyperparameter selection; each fold is one train+val experiment.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def split(
        self,
        subjects: list[int],
        held_out: list[int] | None = None,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Yield (train_subjects, val_subjects) for each fold.

        Neither list overlaps; subjects in held_out are excluded entirely.
        """
        from sklearn.model_selection import GroupKFold

        avail = sorted(s for s in subjects if s not in (held_out or []))
        n = len(avail)
        kf = GroupKFold(n_splits=min(self.n_splits, n))
        # Each subject is its own group — prevents any subject from crossing the boundary.
        groups = list(range(n))
        for train_idx, val_idx in kf.split(groups, groups=groups):
            yield ([avail[i] for i in train_idx], [avail[i] for i in val_idx])


class LOSOSplitter:
    """Leave-one-subject-out cross-validation. §11.2.

    Primary evaluation protocol for Stage 1 subject-independent results.
    Each iteration holds out exactly one subject as the test set.
    """

    def split(
        self,
        subjects: list[int],
        held_out: list[int] | None = None,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Yield (train_subjects, [test_subject]) for each subject.

        Subjects in held_out are tested but never appear in train.
        """
        avail = sorted(s for s in subjects if s not in (held_out or []))
        for test_subj in avail:
            train = [s for s in avail if s != test_subj]
            yield train, [test_subj]


# ---------------------------------------------------------------------------
# Stage 3 calibration split protocols
# ---------------------------------------------------------------------------

SPLIT_ARMS = (
    "temporal_array",
    "temporal_chrono",
    "stratified",
    "stratified_task_only",
    "purged_stratified",
)


@dataclass(frozen=True)
class CalibrationSplit:
    """Indices into a subject's epoch array, plus the composition of each side."""

    calib_idx: np.ndarray
    eval_idx: np.ndarray
    arm: str
    n_purged: int = 0

    def describe(self, y: np.ndarray, source: np.ndarray) -> dict:
        """Composition statistics recorded in the output CSV for every run."""
        def _frac_baseline(idx):
            if len(idx) == 0:
                return float("nan")
            return float(np.mean(np.asarray(source)[idx] == "baseline"))

        def _pos_rate(idx):
            if len(idx) == 0:
                return float("nan")
            return float(np.mean(np.asarray(y)[idx] == 1))

        return {
            "split": self.arm,
            "n_calib": int(len(self.calib_idx)),
            "n_eval": int(len(self.eval_idx)),
            "calib_pos_rate": _pos_rate(self.calib_idx),
            "eval_pos_rate": _pos_rate(self.eval_idx),
            "calib_baseline_frac": _frac_baseline(self.calib_idx),
            "eval_baseline_frac": _frac_baseline(self.eval_idx),
            "n_purged": int(self.n_purged),
        }


def _stratified_indices(y: np.ndarray, n_calib: int, seed: int,
                        pool: np.ndarray | None = None) -> np.ndarray:
    """Stratified draw of n_calib indices, optionally restricted to *pool*."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y)) if pool is None else np.asarray(pool)
    if n_calib >= len(idx):
        raise ValueError(
            f"Cannot draw {n_calib} calibration epochs from a pool of {len(idx)}."
        )
    y_pool = np.asarray(y)[idx]
    if len(np.unique(y_pool)) < 2:
        raise ValueError("Stratified draw needs both classes present in the pool.")
    calib, _ = train_test_split(
        idx, train_size=n_calib, random_state=seed, stratify=y_pool
    )
    return np.sort(np.asarray(calib))


def make_calibration_split(
    arm: str,
    y: np.ndarray,
    n_calib: int,
    seed: int,
    order: np.ndarray,
    source: np.ndarray,
    purge_k: int = 5,
) -> CalibrationSplit:
    """Build the calibration/evaluation index split for one arm.

    Arms
    ----
    temporal_array
        First N epochs in the array's current order (task runs first, baseline runs
        last). Reproduces the original temporal result.
    temporal_chrono
        First N epochs in true acquisition order, so the eyes-open/eyes-closed
        baseline runs — which were recorded first — come first. The reverse of
        temporal_array with respect to the baseline sub-population, and therefore an
        independent test of the same mechanism question.
    stratified
        train_test_split(train_size=N, stratify=y, random_state=seed).
    stratified_task_only
        Stratified draw restricted to source == "task" epochs. The evaluation set is
        unchanged (everything not drawn), so this isolates *which sub-population the
        calibration set covers* from *how it is spread in time*.
    purged_stratified
        Stratified, then drop evaluation epochs within ±purge_k of any calibration
        epoch in chronological order. Leakage control for temporally adjacent,
        partially redundant epochs.
    """
    y = np.asarray(y)
    order = np.asarray(order)
    source = np.asarray(source)
    n = len(y)
    if arm not in SPLIT_ARMS:
        raise ValueError(f"Unknown split arm {arm!r}; expected one of {SPLIT_ARMS}")
    if n_calib >= n:
        raise ValueError(f"n_calib={n_calib} but the subject has only {n} epochs.")

    all_idx = np.arange(n)

    if arm == "temporal_array":
        calib = all_idx[:n_calib]
        evl = all_idx[n_calib:]

    elif arm == "temporal_chrono":
        chrono = np.argsort(order, kind="stable")
        calib = np.sort(chrono[:n_calib])
        evl = np.sort(chrono[n_calib:])

    elif arm == "stratified":
        calib = _stratified_indices(y, n_calib, seed)
        evl = np.setdiff1d(all_idx, calib)

    elif arm == "stratified_task_only":
        pool = all_idx[source == "task"]
        calib = _stratified_indices(y, n_calib, seed, pool=pool)
        # Evaluation set is everything not drawn — including baseline epochs. The
        # evaluation denominator therefore matches `stratified`, and the only
        # difference is where calibration was allowed to draw from.
        evl = np.setdiff1d(all_idx, calib)

    elif arm == "purged_stratified":
        calib = _stratified_indices(y, n_calib, seed)
        evl = np.setdiff1d(all_idx, calib)
        calib_pos = order[calib]
        # Drop evaluation epochs whose chronological position is within ±k of any
        # calibration epoch's.
        keep = np.array([
            bool(np.all(np.abs(calib_pos - order[i]) > purge_k)) for i in evl
        ], dtype=bool)
        n_purged = int((~keep).sum())
        return CalibrationSplit(np.sort(calib), np.sort(evl[keep]), arm, n_purged)

    return CalibrationSplit(np.sort(calib), np.sort(evl), arm, 0)
