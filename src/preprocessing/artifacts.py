"""Robust scaling and amplitude-based artifact rejection. Used by Preprocessor."""
from __future__ import annotations

import numpy as np

_EPS = 1e-8  # guard against zero IQR (flat channels)


class RobustChannelScaler:
    """Per-channel robust scaling using median and IQR. [fit-on-train]

    Fit on training data; apply unchanged to val/test.
    Picklable so it can be saved in checkpoint bundles.
    """

    def __init__(self) -> None:
        self._median: np.ndarray | None = None
        self._iqr: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> RobustChannelScaler:
        """Compute per-channel median and IQR from X: (n_epochs, n_channels, n_times).

        Flattens epochs and time to compute channel-wise statistics, consistent
        with 'fit-on-train-only' invariant (§11).
        """
        n_ch = X.shape[1]
        flat = X.transpose(1, 0, 2).reshape(n_ch, -1)  # (n_ch, n_epochs*n_times)
        self._median = np.median(flat, axis=1).astype(np.float32)
        q75, q25 = np.percentile(flat, [75.0, 25.0], axis=1)
        self._iqr = (q75 - q25).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtract per-channel median and divide by IQR. Raises if not fitted."""
        if not self.is_fitted:
            raise RuntimeError("RobustChannelScaler.transform called before fit()")
        med = self._median[np.newaxis, :, np.newaxis]  # (1, C, 1)
        iqr = np.maximum(self._iqr[np.newaxis, :, np.newaxis], _EPS)
        return ((X - med) / iqr).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._median is not None


def reject_by_amplitude(
    X: np.ndarray,
    threshold_uv: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_clean, keep_mask) for epochs below peak-to-peak threshold.

    Training only — never called at inference. §5.2, §C.1 step 6.

    X         : (n_epochs, n_channels, n_times) float32
    keep_mask : bool array (n_epochs,) — True = epoch passes the threshold
    """
    # Peak-to-peak amplitude per (epoch, channel); avoid deprecated .ptp()
    ptp = X.max(axis=-1) - X.min(axis=-1)          # (n_epochs, n_channels)
    worst_channel_ptp = ptp.max(axis=-1)             # (n_epochs,)
    keep_mask = worst_channel_ptp <= threshold_uv
    return X[keep_mask], keep_mask
