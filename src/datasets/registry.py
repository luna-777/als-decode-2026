"""DatasetSpec dataclass and get_dataset factory. §8.3 — sole entry point for dataset creation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.datasets.wrapper import MoabbDatasetWrapper


@dataclass
class DatasetSpec:
    """Immutable descriptor for a single dataset configuration.

    Consumed by MoabbDatasetWrapper and EEGDataModule.
    All fields correspond to config keys in configs/dataset/*.yaml.
    """

    moabb_name: str
    paradigm: Literal["mi", "p300"]
    channels: list[str]
    sfreq_target: float
    exclude_subjects: list[int] = field(default_factory=list)
    band: tuple[float, float] = (8.0, 30.0)
    epoch_window: tuple[float, float] = (0.0, 2.0)


def get_dataset(spec: DatasetSpec) -> MoabbDatasetWrapper:
    """Instantiate and return a MoabbDatasetWrapper for *spec*."""
    from src.datasets.wrapper import MoabbDatasetWrapper  # avoid circular at import time

    return MoabbDatasetWrapper(spec)
