"""EEGDataModule — owns split logic, samplers, and per-fold Preprocessor application.

§8.3 contract: fits the Preprocessor inside each fold on training subjects only.
WeightedRandomSampler handles the idle-heavy class imbalance (§5.4).
"""
from __future__ import annotations

import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

log = logging.getLogger(__name__)


class EEGDataModule(pl.LightningDataModule):
    """Lightning DataModule for subject-independent EEG experiments.

    Responsibilities:
    - Loads epochs for train/val/test subject lists via MoabbDatasetWrapper.
    - Per-fold Preprocessor.fit_transform() on train, transform() on val/test.
    - Amplitude rejection on train only (never at inference).
    - WeightedRandomSampler to counteract class imbalance in training batches.
    - DataLoader construction (num_workers=0 for Mac/portability).
    """

    def __init__(
        self,
        cfg,
        wrapper,
        train_subjects: list[int],
        val_subjects: list[int],
        test_subjects: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.wrapper = wrapper
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects or []

        # Set after setup():
        self.preprocessor = None
        self.pos_weight: float | None = None
        self._train_ds: TensorDataset | None = None
        self._val_ds: TensorDataset | None = None
        self._test_ds: TensorDataset | None = None
        self._train_sampler: WeightedRandomSampler | None = None

    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Load epochs, fit preprocessor on train, build TensorDatasets."""
        pp_cfg = getattr(self.cfg, "preprocessing", {})
        tr_cfg = getattr(self.cfg, "training", {})
        threshold = float(_cfg_get(pp_cfg, "amplitude_reject_uv", 150.0))
        batch_size = int(_cfg_get(tr_cfg, "batch_size", 64))  # noqa: F841

        # --- Load ---
        log.info("Loading train epochs (%d subjects)…", len(self.train_subjects))
        X_tr, y_tr, _ = self.wrapper.load_epochs(self.train_subjects)
        log.info("Loading val epochs (%d subjects)…", len(self.val_subjects))
        X_val, y_val, _ = self.wrapper.load_epochs(self.val_subjects)

        # --- Preprocess (fit-on-train-only) ---
        from src.preprocessing.pipeline import Preprocessor

        self.preprocessor = Preprocessor(pp_cfg)
        X_tr = self.preprocessor.fit_transform(X_tr)
        X_val = self.preprocessor.transform(X_val)

        # --- Amplitude rejection — training only ---
        from src.preprocessing.artifacts import reject_by_amplitude

        X_tr, mask = reject_by_amplitude(X_tr, threshold_uv=threshold)
        y_tr = y_tr[mask]
        log.info(
            "After rejection: %d train epochs kept (%.1f%% pass)",
            len(y_tr),
            100.0 * mask.mean(),
        )

        # --- Class balance ---
        n_ctrl = int((y_tr == 1).sum())
        n_idle = int((y_tr == 0).sum())
        self.pos_weight = n_idle / max(n_ctrl, 1)
        log.info("pos_weight = %.3f (idle=%d, ctrl=%d)", self.pos_weight, n_idle, n_ctrl)

        # --- TensorDatasets ---
        self._train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        self._val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        if self.test_subjects:
            log.info("Loading test epochs (%d subjects)…", len(self.test_subjects))
            X_te, y_te, _ = self.wrapper.load_epochs(self.test_subjects)
            X_te = self.preprocessor.transform(X_te)
            self._test_ds = TensorDataset(
                torch.from_numpy(X_te), torch.from_numpy(y_te)
            )

        # --- Weighted sampler (balances classes in each batch) ---
        counts = np.bincount(y_tr.astype(np.int64))
        sample_w = (1.0 / counts)[y_tr.astype(np.int64)]
        self._train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).float(),
            num_samples=len(y_tr),
            replacement=True,
        )

    # ------------------------------------------------------------------

    def _batch_size(self) -> int:
        return int(_cfg_get(getattr(self.cfg, "training", {}), "batch_size", 64))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size(),
            sampler=self._train_sampler,
            num_workers=0,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size(),
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("No test subjects configured.")
        return DataLoader(
            self._test_ds,
            batch_size=self._batch_size(),
            shuffle=False,
            num_workers=0,
        )


# ---------------------------------------------------------------------------

def _cfg_get(cfg, key: str, default):
    """Retrieve *key* from an OmegaConf DictConfig or plain dict."""
    try:
        v = cfg.get(key, default)
        return v if v is not None else default
    except AttributeError:
        return default


def build_spec_from_cfg(cfg_dataset):
    """Construct a DatasetSpec from the Hydra dataset config node."""
    from src.datasets.registry import DatasetSpec

    ch_config = str(getattr(cfg_dataset, "channel_config", "sensorimotor"))
    channels_node = cfg_dataset.channels
    ch_list = channels_node[ch_config]
    if ch_list is None:
        raise ValueError(
            f"channels.{ch_config} is null — provide an explicit channel list."
        )
    return DatasetSpec(
        moabb_name=str(cfg_dataset.moabb_name),
        paradigm=str(cfg_dataset.paradigm),
        channels=list(ch_list),
        sfreq_target=float(cfg_dataset.sfreq_target),
        exclude_subjects=list(cfg_dataset.get("exclude_subjects", [])),
        band=(float(cfg_dataset.band[0]), float(cfg_dataset.band[1])),
        epoch_window=(
            float(cfg_dataset.epoch_window[0]),
            float(cfg_dataset.epoch_window[1]),
        ),
    )
