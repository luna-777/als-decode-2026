"""Preprocessor — config-driven preprocessing pipeline. §8.3.

Single source of truth for all preprocessing. Stages differ only by config, not code.
Must be picklable so it can be saved inside checkpoint bundles.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.preprocessing.artifacts import RobustChannelScaler
from src.preprocessing.filters import bandpass_filter


class Preprocessor:
    """Executes a preprocessing recipe from a Hydra config dict / OmegaConf DictConfig.

    Fit-on-train-only invariant: call fit() on training data, then transform() on all splits.
    Never call fit() on validation or test data.

    Stage 1 recipe (mi_8_30, §C.1):
      1. Band-pass 8–30 Hz (zero-phase FIR)
      2. Keep native 160 Hz (no resampling)
      3. Robust per-channel scaling  [fit-on-train]
      4. (optional) Euclidean Alignment  [fit-on-train, disabled by default in Stage 1]

    Artifact rejection (reject_by_amplitude) is intentionally NOT part of transform()
    — it is a training-only step called explicitly by the training loop (M5) so that
    inference code paths never reject windows silently.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._fitted: bool = False
        self._scaler: RobustChannelScaler | None = None

    # ------------------------------------------------------------------
    # Config helpers — accept both OmegaConf DictConfig and plain dict
    # ------------------------------------------------------------------

    def _get(self, key: str, default: Any) -> Any:
        try:
            v = self.cfg.get(key, default)
            return v if v is not None else default
        except AttributeError:
            return default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        channel_names: list[str] | None = None,
    ) -> Preprocessor:
        """Fit scaler (and optional EA reference) on *training* data only.

        X: (n_epochs, n_channels, n_times)
        Returns self for chaining.
        """
        if self._get("robust_scale", True):
            self._scaler = RobustChannelScaler()
            self._scaler.fit(X)

        # Euclidean Alignment is applied per-subject inside load_epochs(),
        # before the Preprocessor sees any data. Nothing to fit here.

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply band-pass filter then robust scaling. Raises if not fitted."""
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit()")

        band = self._get("band", [8.0, 30.0])
        sfreq = self._get("keep_sfreq", 160.0)

        X_out = bandpass_filter(X, sfreq=float(sfreq), l_freq=float(band[0]), h_freq=float(band[1]))

        if self._scaler is not None:
            X_out = self._scaler.transform(X_out)

        return X_out.astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Convenience: fit then transform on the same data (training set only)."""
        return self.fit(X, channel_names).transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
