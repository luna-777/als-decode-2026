"""Euclidean Alignment (He et al., 2020) — per-subject spatial covariance whitening.

Applied per-subject in load_epochs() before the cross-subject Preprocessor so the
model learns from a covariance-normalised feature space. At Stage 3 test time the
same class is fitted on the test patient's available data, producing a patient-
specific reference that brings new data into the training distribution.

Reference
---------
He, H. & Wu, D. (2020). Transfer learning for brain-computer interfaces: A
Euclidean space data alignment approach. IEEE TNSRE, 28(8), 1817–1830.
"""
from __future__ import annotations

import numpy as np


class EuclideanAligner:
    """Fit R^{-1/2} on a subject's epochs; apply to any epoch set of the same shape.

    fit(X)         — compute reference from X: (N, C, T)
    transform(X)   — return R^{-1/2} @ X[i] for each epoch; shape unchanged
    fit_transform  — convenience: fit then transform on the same array

    The aligner is picklable (stores only a single (C, C) float32 array).
    """

    def __init__(self) -> None:
        self._ref: np.ndarray | None = None  # R^{-1/2}, shape (C, C)

    def fit(self, X: np.ndarray) -> "EuclideanAligner":
        """Estimate the whitening matrix from X: (N, C, T)."""
        if X.ndim != 3:
            raise ValueError(f"Expected (N, C, T), got shape {X.shape}")
        n_times = X.shape[2]
        # Per-epoch covariance: C_i = X_i @ X_i.T / T  →  mean over epochs = R
        covs = np.einsum("nct,ndt->ncd", X, X) / n_times
        R = covs.mean(axis=0)  # (C, C)
        # Symmetric eigendecomposition — eigh guarantees real eigenvalues for SPD R
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-10)  # guard against numerical noise
        # R^{-1/2} = V @ diag(λ^{-1/2}) @ V.T
        self._ref = (eigvecs * (eigvals ** -0.5)).dot(eigvecs.T).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply whitening: R^{-1/2} @ X[i] for each epoch in X: (N, C, T)."""
        if self._ref is None:
            raise RuntimeError("EuclideanAligner.transform() called before fit()")
        return np.einsum("cd,ndt->nct", self._ref, X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
