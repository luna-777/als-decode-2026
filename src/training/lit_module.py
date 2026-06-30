"""LitEEG — PyTorch Lightning module wrapping EEGDecoder. §8.3."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


class LitEEG(pl.LightningModule):
    """Wraps EEGDecoder + loss + optimizer/scheduler + torchmetrics.

    Reads everything from cfg (Hydra DictConfig).
    Hooks: training_step, validation_step, test_step, configure_optimizers.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # super().__init__() called first in Milestone 5 implementation.
        raise NotImplementedError("Milestone 5")

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        raise NotImplementedError("Milestone 5")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError("Milestone 5")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError("Milestone 5")

    def configure_optimizers(self) -> Any:
        raise NotImplementedError("Milestone 5")
