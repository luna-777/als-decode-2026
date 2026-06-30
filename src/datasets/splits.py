"""Group-aware splitters — subjects are the grouping unit. §11.1.

No subject may appear in two splits. The split is enforced at the subject level
*before* sliding-window epoching so overlapping windows never straddle boundaries.
"""
from __future__ import annotations

from typing import Iterator


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
