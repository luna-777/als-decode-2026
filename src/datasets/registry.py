"""DatasetSpec dataclass and get_dataset factory. §8.3 — sole entry point for dataset creation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetSpec:
    """Immutable descriptor for a single dataset configuration.

    Consumed by MoabbDatasetWrapper / BnciP300Wrapper and EEGDataModule.
    All fields correspond to config keys in configs/dataset/*.yaml.
    """

    moabb_name: str
    paradigm: Literal["mi", "p300"]
    channels: list[str]
    sfreq_target: float
    exclude_subjects: list[int] = field(default_factory=list)
    band: tuple[float, float] = (8.0, 30.0)
    epoch_window: tuple[float, float] = (0.0, 2.0)


def get_dataset(spec: DatasetSpec):
    """Instantiate and return the appropriate wrapper for *spec*.

    paradigm='mi'   → MoabbDatasetWrapper  (Stage 1 / PhysionetMI)
    paradigm='p300' → BnciP300Wrapper      (Stage 2 / BNCI2014_009)
    """
    if spec.paradigm == "p300":
        from src.datasets.wrapper import BnciP300Wrapper
        return BnciP300Wrapper(spec)
    from src.datasets.wrapper import MoabbDatasetWrapper
    return MoabbDatasetWrapper(spec)
