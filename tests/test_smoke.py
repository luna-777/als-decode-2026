"""Milestone 0 CI smoke tests — all should pass immediately after env setup.

Verifies: every src module is importable, seed reproducibility works,
and the DatasetSpec dataclass is constructable.
No data download or model training is attempted here.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

MODULES = [
    "src.utils.seed",
    "src.utils.logging",
    "src.utils.io",
    "src.datasets.registry",
    "src.datasets.wrapper",
    "src.datasets.splits",
    "src.preprocessing.pipeline",
    "src.preprocessing.filters",
    "src.preprocessing.artifacts",
    "src.models.backbone",
    "src.models.classical",
    "src.training.lit_module",
    "src.training.datamodule",
    "src.training.losses",
    "src.training.callbacks",
    "src.decoding.intent_switch",
    "src.evaluation.metrics",
    "src.evaluation.streaming",
    "src.evaluation.report",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name: str) -> None:
    """Every Stage 1 module must import without error."""
    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_python_reproducible() -> None:
    import random

    from src.utils.seed import seed_everything

    seed_everything(42)
    a = random.random()
    seed_everything(42)
    assert random.random() == a


def test_seed_numpy_reproducible() -> None:
    from src.utils.seed import seed_everything

    seed_everything(42)
    a = np.random.random()
    seed_everything(42)
    assert np.random.random() == a


def test_seed_torch_reproducible() -> None:
    from src.utils.seed import seed_everything

    seed_everything(42)
    t1 = torch.randn(8)
    seed_everything(42)
    t2 = torch.randn(8)
    assert torch.allclose(t1, t2)


def test_seed_different_values_differ() -> None:
    from src.utils.seed import seed_everything

    seed_everything(0)
    t0 = torch.randn(8)
    seed_everything(1)
    t1 = torch.randn(8)
    assert not torch.allclose(t0, t1)


# ---------------------------------------------------------------------------
# DatasetSpec construction
# ---------------------------------------------------------------------------

def test_dataset_spec_constructable() -> None:
    from src.datasets.registry import DatasetSpec

    spec = DatasetSpec(
        moabb_name="PhysionetMI",
        paradigm="mi",
        channels=["Cz", "C3", "C4"],
        sfreq_target=160.0,
        exclude_subjects=[88, 92, 100],
        band=(8.0, 30.0),
        epoch_window=(0.0, 2.0),
    )
    assert spec.moabb_name == "PhysionetMI"
    assert 88 in spec.exclude_subjects


# ---------------------------------------------------------------------------
# Preprocessor: transform before fit raises
# ---------------------------------------------------------------------------

def test_preprocessor_raises_if_not_fitted(tiny_eeg_numpy) -> None:
    from src.preprocessing.pipeline import Preprocessor

    pp = Preprocessor(cfg={})
    assert not pp.is_fitted
    with pytest.raises((RuntimeError, NotImplementedError)):
        pp.transform(tiny_eeg_numpy)
