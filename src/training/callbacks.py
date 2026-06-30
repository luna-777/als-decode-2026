"""Training callbacks — early stopping, checkpointing, LR monitor."""
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_callbacks(cfg) -> list[pl.Callback]:
    """Return the standard callback stack from config.

    Always includes:
    - EarlyStopping on val/roc_auc (patience from cfg.training.early_stop.patience)
    - ModelCheckpoint (best val/roc_auc, filename encodes epoch+metric)
    - LearningRateMonitor
    """
    tr = getattr(cfg, "training", {})
    es_cfg = _get(tr, "early_stop", {})
    monitor = str(_get(es_cfg, "monitor", "val/roc_auc"))
    patience = int(_get(es_cfg, "patience", 20))
    mode = str(_get(es_cfg, "mode", "max"))
    min_delta = float(_get(es_cfg, "min_delta", 0.001))

    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            verbose=True,
        ),
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            filename="best-{epoch:03d}-{val/roc_auc:.4f}",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def _get(cfg, key: str, default):
    try:
        v = cfg.get(key, default)
        return v if v is not None else default
    except AttributeError:
        return default
