"""Checkpoint bundling — save/load weights + fitted Preprocessor + channel order + meta.

§10.4 contract: every checkpoint must include all four components so inference is
reproducible and leakage-free without external config.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.preprocessing.pipeline import Preprocessor


def save_checkpoint(
    path: Path | str,
    weights_state_dict: dict[str, Any],
    preprocessor: Preprocessor,
    channel_order: list[str],
    meta: dict[str, Any],
) -> None:
    """Save a complete checkpoint bundle to *path* (`.ckpt`).

    The bundle contains: model weights, the fitted Preprocessor (picklable),
    canonical channel order, and a metadata dict (seed, git SHA, config hash, etc.).
    """
    raise NotImplementedError("Milestone 5")


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a checkpoint bundle.

    Returns a dict with keys: 'weights', 'preprocessor', 'channel_order', 'meta'.
    Raises ValueError if any required key is missing (corrupted bundle).
    """
    raise NotImplementedError("Milestone 5")
