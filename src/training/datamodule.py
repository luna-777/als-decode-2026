"""EEGDataModule — owns split logic, samplers, and per-fold Preprocessor application.

§8.3 contract: applies the Preprocessor inside each fold without leakage.
WeightedRandomSampler handles the idle-heavy class imbalance (§5.4).
"""
from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class EEGDataModule(pl.LightningDataModule):
    """Lightning DataModule for subject-independent EEG experiments.

    Responsibilities:
    - Group-aware splitting (GroupKFold / LOSO / held-out test set).
    - Per-fold Preprocessor.fit() on train subjects only, then transform all splits.
    - WeightedRandomSampler to counteract class imbalance.
    - DataLoader construction.
    """

    def __init__(self, cfg: DictConfig) -> None:
        raise NotImplementedError("Milestone 3")

    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError("Milestone 3")

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError("Milestone 3")

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("Milestone 3")

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Milestone 3")
