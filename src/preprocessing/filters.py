"""Band-pass, notch, and resampling wrappers over MNE. Used by Preprocessor."""
from __future__ import annotations

import warnings

import numpy as np


def bandpass_filter(
    X: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    method: str = "fir",
    phase: str = "zero",
    fir_window: str = "hamming",
) -> np.ndarray:
    """Zero-phase FIR band-pass filter via MNE.

    X: (..., n_times) — last axis is time.
    Returns filtered float32 array of same shape.
    """
    import mne.filter as mnef

    X_in = np.asarray(X, dtype=np.float64)  # MNE needs float64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_out = mnef.filter_data(
            X_in,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            phase=phase,
            fir_window=fir_window,
            verbose=False,
        )
    return X_out.astype(np.float32)


def resample(X: np.ndarray, orig_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Resample X from orig_sfreq to target_sfreq using MNE's resampler.

    X: (..., n_times) — last axis is time.
    Not used in Stage 1 (keep_sfreq=160 Hz); implemented for Stage 2.
    """
    raise NotImplementedError("Milestone 2 — resample needed for Stage 2 (P300 → 128 Hz)")
