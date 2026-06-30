"""Group-aware splitters — subjects are the grouping unit. §11.1.

No subject may appear in two splits. The split is enforced at the subject level
*before* sliding-window epoching so overlapping windows never straddle boundaries.
"""
from __future__ import annotations

from typing import Iterator


class GroupKFoldSplitter:
    """Group-aware k-fold cross-validation over subjects.

    Subjects in *held_out* are excluded from all folds (reserved for final test).
    Used for hyperparameter selection (Milestone 3).
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
        raise NotImplementedError("Milestone 3")


class LOSOSplitter:
    """Leave-one-subject-out cross-validation. §11.2.

    Primary evaluation protocol for Stage 1 subject-independent results.
    """

    def split(
        self,
        subjects: list[int],
        held_out: list[int] | None = None,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Yield (train_subjects, [test_subject]) for each subject.

        Subjects in held_out are tested but never appear in train.
        """
        raise NotImplementedError("Milestone 3")
