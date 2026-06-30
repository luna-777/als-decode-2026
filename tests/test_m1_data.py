"""Milestone 1 checkpoint — load 2 subjects, verify shapes and class balance.

Run with:
    conda run -n als-decode pytest tests/test_m1_data.py -v -m data

Requires the MOABB cache to be warm (run scripts/download_data.py first,
or the tests will download on demand — slow on first run).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.registry import DatasetSpec, get_dataset

_SENSORIMOTOR_17 = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def spec() -> DatasetSpec:
    return DatasetSpec(
        moabb_name="PhysionetMI",
        paradigm="mi",
        channels=_SENSORIMOTOR_17,
        sfreq_target=160.0,
        exclude_subjects=[88, 92, 100],
        band=(8.0, 30.0),
        epoch_window=(0.0, 2.0),
    )


@pytest.fixture(scope="module")
def epochs(spec):
    """Load subjects 1–2; cached for this module."""
    wrapper = get_dataset(spec)
    X, y, meta = wrapper.load_epochs(subjects=[1, 2])
    return X, y, meta


# ---------------------------------------------------------------------------
# Shape / dtype checks
# ---------------------------------------------------------------------------

def test_x_shape(epochs):
    X, _, _ = epochs
    assert X.ndim == 3, "X must be 3-D (epochs, channels, times)"
    n_ep, n_ch, n_t = X.shape
    print(f"\nX shape: {X.shape}  dtype: {X.dtype}")
    assert n_ch == 17, f"Expected 17 sensorimotor channels, got {n_ch}"
    assert n_t == 320, f"Expected 320 time-points (2 s @ 160 Hz), got {n_t}"


def test_x_dtype(epochs):
    X, _, _ = epochs
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"


def test_y_dtype(epochs):
    _, y, _ = epochs
    assert y.dtype == np.int64, f"Expected int64, got {y.dtype}"


def test_y_shape(epochs):
    X, y, _ = epochs
    assert y.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# Label content
# ---------------------------------------------------------------------------

def test_label_values(epochs):
    _, y, _ = epochs
    assert set(y.tolist()) <= {0, 1}, f"Unexpected label values: {set(y.tolist())}"


def test_both_classes_present(epochs):
    _, y, _ = epochs
    n_ctrl = int((y == 1).sum())
    n_idle = int((y == 0).sum())
    print(
        f"\nClass balance — total: {len(y)}, control: {n_ctrl}, "
        f"idle: {n_idle}, idle/ctrl ratio: {n_idle / max(n_ctrl, 1):.2f}"
    )
    assert n_ctrl > 0, "No control (motor imagery) epochs found"
    assert n_idle > 0, "No idle (rest) epochs found"


# ---------------------------------------------------------------------------
# Metadata / subject integrity
# ---------------------------------------------------------------------------

def test_subject_metadata(epochs):
    _, _, meta = epochs
    assert set(meta["subjects"]) == {1, 2}, (
        f"Expected subjects {{1, 2}}, got {set(meta['subjects'])}"
    )


def test_channel_order(epochs):
    _, _, meta = epochs
    assert meta["channels"] == _SENSORIMOTOR_17, "Channel order does not match canonical spec"


def test_sfreq_metadata(epochs):
    _, _, meta = epochs
    assert meta["sfreq"] == 160.0


# ---------------------------------------------------------------------------
# Dataset isolation guard
# ---------------------------------------------------------------------------

def test_only_physionet_mi_loaded(spec):
    """get_dataset must return a wrapper backed exclusively by PhysionetMI."""
    from moabb.datasets import PhysionetMI
    wrapper = get_dataset(spec)
    assert isinstance(wrapper._moabb_ds, PhysionetMI)
