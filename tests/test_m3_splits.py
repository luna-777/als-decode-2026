"""Milestone 3 checkpoint — splits, leakage guard, DataModule wiring.

All tests use synthetic data; no MOABB download required.
Run with: pytest tests/test_m3_splits.py -v
"""
from __future__ import annotations

import numpy as np
import pytest


SUBJECTS = list(range(1, 31))  # 30 synthetic subjects


# ---------------------------------------------------------------------------
# GroupKFoldSplitter
# ---------------------------------------------------------------------------

class TestGroupKFoldSplitter:
    def test_yields_n_splits(self):
        from src.datasets.splits import GroupKFoldSplitter
        folds = list(GroupKFoldSplitter(n_splits=5).split(SUBJECTS))
        assert len(folds) == 5

    def test_no_overlap(self):
        from src.datasets.splits import GroupKFoldSplitter
        for train, val in GroupKFoldSplitter(n_splits=5).split(SUBJECTS):
            assert set(train) & set(val) == set(), "Subject appears in both train and val"

    def test_all_subjects_covered(self):
        """Every subject appears in val exactly once across all folds."""
        from src.datasets.splits import GroupKFoldSplitter
        seen_in_val: set[int] = set()
        for _, val in GroupKFoldSplitter(n_splits=5).split(SUBJECTS):
            seen_in_val.update(val)
        assert seen_in_val == set(SUBJECTS)

    def test_held_out_excluded(self):
        from src.datasets.splits import GroupKFoldSplitter
        held_out = [1, 2, 3]
        for train, val in GroupKFoldSplitter(n_splits=5).split(SUBJECTS, held_out=held_out):
            assert not any(s in held_out for s in train)
            assert not any(s in held_out for s in val)

    def test_fewer_splits_than_subjects(self):
        from src.datasets.splits import GroupKFoldSplitter
        folds = list(GroupKFoldSplitter(n_splits=3).split([1, 2, 3, 4]))
        assert len(folds) == 3

    def test_train_larger_than_val(self):
        from src.datasets.splits import GroupKFoldSplitter
        for train, val in GroupKFoldSplitter(n_splits=5).split(SUBJECTS):
            assert len(train) > len(val)


# ---------------------------------------------------------------------------
# LOSOSplitter
# ---------------------------------------------------------------------------

class TestLOSOSplitter:
    def test_yields_one_test_subject(self):
        from src.datasets.splits import LOSOSplitter
        for _, val in LOSOSplitter().split(SUBJECTS[:10]):
            assert len(val) == 1

    def test_no_overlap(self):
        from src.datasets.splits import LOSOSplitter
        for train, val in LOSOSplitter().split(SUBJECTS[:10]):
            assert set(train) & set(val) == set()

    def test_all_subjects_tested(self):
        from src.datasets.splits import LOSOSplitter
        tested = [val[0] for _, val in LOSOSplitter().split(SUBJECTS[:10])]
        assert set(tested) == set(SUBJECTS[:10])

    def test_held_out_excluded_from_train(self):
        from src.datasets.splits import LOSOSplitter
        held_out = {1, 2}
        for train, _ in LOSOSplitter().split(SUBJECTS[:10], held_out=list(held_out)):
            assert not any(s in held_out for s in train)


# ---------------------------------------------------------------------------
# Leakage guard (the critical anti-leakage test §11)
# ---------------------------------------------------------------------------

def test_no_subject_leakage_kfold():
    """No subject ID may appear in both train and val in any GroupKFold fold."""
    from src.datasets.splits import GroupKFoldSplitter
    splitter = GroupKFoldSplitter(n_splits=5)
    for train_s, val_s in splitter.split(SUBJECTS):
        overlap = set(train_s) & set(val_s)
        assert overlap == set(), f"Leakage detected! Subjects in both splits: {overlap}"


def test_no_subject_leakage_loso():
    """No subject may appear in both train and test in LOSO."""
    from src.datasets.splits import LOSOSplitter
    for train_s, test_s in LOSOSplitter().split(SUBJECTS[:15]):
        overlap = set(train_s) & set(test_s)
        assert overlap == set(), f"Leakage detected! Subjects in both splits: {overlap}"


# ---------------------------------------------------------------------------
# EEGDataModule (smoke — using tiny synthetic wrapper)
# ---------------------------------------------------------------------------

class _FakeWrapper:
    """Minimal stand-in for MoabbDatasetWrapper using random data."""

    def __init__(self, n_ch=17, n_times=320):
        self.n_ch, self.n_times = n_ch, n_times

    @property
    def subject_list(self):
        return list(range(1, 11))

    def load_epochs(self, subjects):
        rng = np.random.default_rng(seed=sum(subjects))
        n = len(subjects) * 20
        X = rng.standard_normal((n, self.n_ch, self.n_times)).astype(np.float32) * 5.0
        y = rng.integers(0, 2, size=n).astype(np.int64)
        meta = {"subjects": [s for s in subjects for _ in range(20)],
                "channels": [f"C{i}" for i in range(self.n_ch)],
                "sfreq": 160.0, "n_times": self.n_times}
        return X, y, meta


@pytest.fixture
def fake_cfg():
    """Minimal config dict understood by EEGDataModule."""
    class Cfg:
        class preprocessing:
            @staticmethod
            def get(k, d=None):
                return {"band": [8.0, 30.0], "keep_sfreq": 160.0,
                        "robust_scale": True, "amplitude_reject_uv": 150.0,
                        "euclidean_alignment": False}.get(k, d)
        class training:
            @staticmethod
            def get(k, d=None):
                return {"batch_size": 8, "max_epochs": 2}.get(k, d)
    return Cfg()


def test_datamodule_setup(fake_cfg):
    from src.training.datamodule import EEGDataModule
    wrapper = _FakeWrapper()
    dm = EEGDataModule(fake_cfg, wrapper, [1, 2, 3], [4], [5])
    dm.setup()
    assert dm.preprocessor is not None
    assert dm.preprocessor.is_fitted
    assert dm.pos_weight is not None
    assert dm._train_ds is not None
    assert dm._val_ds is not None
    assert dm._test_ds is not None


def test_datamodule_train_loader_shape(fake_cfg):
    from src.training.datamodule import EEGDataModule
    wrapper = _FakeWrapper()
    dm = EEGDataModule(fake_cfg, wrapper, [1, 2, 3], [4])
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.ndim == 3  # (B, C, T)
    assert y.ndim == 1


def test_datamodule_no_leakage(fake_cfg):
    """Wrapper is only called with the subjects it was given; no cross-split loading."""
    from src.training.datamodule import EEGDataModule

    loaded: list[list[int]] = []

    class TrackedWrapper(_FakeWrapper):
        def load_epochs(self, subjects):
            loaded.append(list(subjects))
            return super().load_epochs(subjects)

    dm = EEGDataModule(fake_cfg, TrackedWrapper(), [1, 2], [3])
    dm.setup()
    # train and val should not share subjects
    all_loaded = [s for call in loaded for s in call]
    train_set = set(loaded[0])
    val_set = set(loaded[1])
    assert train_set & val_set == set()
