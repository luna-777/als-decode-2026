"""Training callbacks — early stopping, checkpointing, LR monitor."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


class PreprocessorCheckpoint(pl.Callback):
    """Saves the fitted Preprocessor to <log_dir>/preprocessor.pkl on fit start.

    Stored alongside the Lightning version directory so evaluate.py can load
    the exact scaler that was fit on training data for each fold.
    """

    def __init__(self, preprocessor) -> None:
        self._preprocessor = preprocessor

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        save_path = Path(trainer.log_dir) / "preprocessor.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self._preprocessor, f)
        pl_module.print(f"Preprocessor saved → {save_path}")


class MontageCheckpoint(pl.Callback):
    """Records the electrode montage (names *and* order) to <log_dir>/montage.json.

    ADR-10: the montage is part of the checkpoint's identity. Stage 3 asserts the
    data it loads matches this file before using the checkpoint. See
    src/datasets/montage.py.
    """

    def __init__(self, spec, n_times: int) -> None:
        self._spec = spec
        self._n_times = int(n_times)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from src.datasets.montage import write_montage

        path = write_montage(Path(trainer.log_dir), self._spec, self._n_times)
        pl_module.print(f"Montage saved → {path} ({len(self._spec.channels)} ch)")


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
