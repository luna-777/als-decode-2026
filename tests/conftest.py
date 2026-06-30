"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
import torch

from src.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def fixed_seed():
    """Reset the global seed before every test for reproducibility."""
    seed_everything(42)
    yield


@pytest.fixture
def tiny_eeg_batch() -> torch.Tensor:
    """(2, 17, 320) — minimal batch: 2 epochs, 17 sensorimotor channels, 320 samples (2s @160 Hz)."""
    return torch.randn(2, 17, 320)


@pytest.fixture
def tiny_eeg_numpy():
    """(2, 17, 320) float32 numpy array — same shape as tiny_eeg_batch."""
    import numpy as np

    return np.random.randn(2, 17, 320).astype(np.float32)
