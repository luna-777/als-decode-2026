"""Training callbacks — early stopping, checkpointing, LR monitor, W&B image logging."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    """Return the standard callback stack from config.

    Always includes:
    - EarlyStopping on val/roc_auc (patience from cfg.training.early_stop.patience)
    - ModelCheckpoint (best val/roc_auc, saves weights + preprocessor bundle)
    - LearningRateMonitor
    - WandbCallback (if W&B is active)
    """
    raise NotImplementedError("Milestone 5")
