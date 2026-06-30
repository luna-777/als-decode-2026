"""LitEEG — PyTorch Lightning module wrapping EEGDecoder. §8.3."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC


class LitEEG(pl.LightningModule):
    """Wraps EEGDecoder + loss + optimizer/scheduler + torchmetrics.

    Reads optimisation hyper-parameters from cfg.training.
    pos_weight (idle/ctrl ratio) is computed by EEGDataModule.setup() and
    passed here so the loss correctly compensates for class imbalance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Any,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model", "cfg"])

        tr = getattr(cfg, "training", {})
        self._loss_name: str = str(_get(tr, "loss", "weighted_bce"))
        self._focal_gamma: float = float(_get(tr, "focal_gamma", 2.0))

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        if pw is not None:
            self.register_buffer("_pos_weight", pw)
        else:
            self._pos_weight = None

        self._train_auc = BinaryAUROC()
        self._val_auc = BinaryAUROC()
        self._test_auc = BinaryAUROC()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        from src.training.losses import focal_loss, weighted_bce

        pw = self._pos_weight
        if self._loss_name == "focal":
            return focal_loss(logits, y.float(), gamma=self._focal_gamma, pos_weight=pw)
        return weighted_bce(logits, y.float(), pos_weight=pw)

    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self._compute_loss(logits, y)
        probs = torch.sigmoid(logits)
        self._train_auc.update(probs, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/roc_auc", self._train_auc, on_step=False, on_epoch=True, prog_bar=False
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self._compute_loss(logits, y)
        probs = torch.sigmoid(logits)
        self._val_auc.update(probs, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/roc_auc", self._val_auc, prog_bar=True, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x).squeeze(-1)
        self._test_auc.update(torch.sigmoid(logits), y)
        self.log("test/roc_auc", self._test_auc, on_epoch=True)

    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        tr = getattr(self.cfg, "training", {})
        lr = float(_get(tr, "lr", 1e-3))
        wd = float(_get(tr, "weight_decay", 1e-2))
        max_epochs = int(_get(tr, "max_epochs", 200))
        warmup = int(_get(tr, "warmup_epochs", 10))

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_epochs - warmup, 1), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _get(cfg, key: str, default):
    try:
        v = cfg.get(key, default)
        return v if v is not None else default
    except AttributeError:
        return default
