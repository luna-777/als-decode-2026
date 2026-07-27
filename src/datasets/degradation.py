"""Controlled montage degradation — a first-class experimental instrument.

docs/AUDIT.md §0.1 established that Stage 3 loaded MI data under a montage sharing
zero array positions with the one the checkpoints were trained on, and §0.6 that
repairing it raises held-out AUC by +0.0586. That is a single point. This module
turns montage corruption into a *dose*, so the relationship between backbone
degradation and apparent adaptation gain can be measured rather than inferred.

Design
------
A degraded montage is produced by **permuting k of the 17 channel positions** of
the verified config montage, leaving 17−k in place. The channel *set* is held
fixed; only the assignment of electrode to array position changes. That isolates
spatial-filter misalignment from any change in which electrodes are present.

The permutation is applied to the epoch array **after** the montage contract has
been asserted on the load, not by loading a different channel list. Permuting the
channel axis of an array loaded under montage C is exactly equivalent to loading
under the permuted list, because:

* channel selection is a pure gather (``mne.Raw.pick`` preserves requested order),
* Euclidean Alignment commutes with channel permutation — for ``X' = P X`` the
  reference is ``R' = P R Pᵀ`` and ``R'^{-1/2} = P R^{-1/2} Pᵀ``, so
  ``R'^{-1/2} X' = P R^{-1/2} X``.

So the assertion still runs and still passes; the degradation is an explicit,
recorded transformation applied to verified data. The permutation must be applied
*before* the preprocessor, whose per-channel scalers are positional — that
positional mismatch is part of the defect being reproduced.

The one condition this cannot express is the historical montage itself, which
substituted three electrodes (FC5, FC6, CP5 in; CPz, CP2, CP4 out) rather than
permuting. That requires a genuine reload and is handled by an explicit override.
"""
from __future__ import annotations

import numpy as np

# The exact list scripts/adapt_stage3.py hardcoded from b4cd19f until the audit.
# Not a permutation of the config montage: it swaps three electrodes.
PREAUDIT_MI_MONTAGE: list[str] = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1",
]


def derangement(k: int, rng: np.random.Generator, max_tries: int = 1000) -> np.ndarray:
    """A permutation of range(k) with no fixed point. k=0 and k=1 have none; k=0 → empty."""
    if k == 0:
        return np.empty(0, dtype=int)
    if k == 1:
        raise ValueError("A derangement of one element does not exist; use k >= 2.")
    for _ in range(max_tries):
        p = rng.permutation(k)
        if np.all(p != np.arange(k)):
            return p
    raise RuntimeError(f"Could not draw a derangement of {k} elements")


def permutation_for_k(n_channels: int, k: int, seed: int) -> np.ndarray:
    """Index array `perm` such that degraded[:, i, :] = clean[:, perm[i], :].

    Exactly *k* positions move; the remaining n_channels-k are fixed points.
    k=0 returns the identity regardless of seed.
    """
    if not 0 <= k <= n_channels:
        raise ValueError(f"k must be in [0, {n_channels}], got {k}")
    if k == 1:
        raise ValueError("k=1 is impossible: a single position cannot move alone.")
    perm = np.arange(n_channels)
    if k == 0:
        return perm
    rng = np.random.default_rng(seed)
    positions = rng.choice(n_channels, size=k, replace=False)
    perm[positions] = positions[derangement(k, rng)]
    return perm


def n_displaced(perm: np.ndarray) -> int:
    """How many positions actually moved."""
    return int(np.sum(perm != np.arange(len(perm))))


def apply_permutation(X: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder the channel axis of (N, C, T) epochs."""
    if X.shape[1] != len(perm):
        raise ValueError(f"Permutation length {len(perm)} != n_channels {X.shape[1]}")
    return X[:, perm, :]


def permuted_channel_names(channels: list[str], perm: np.ndarray) -> list[str]:
    """The montage the model effectively sees: position i now carries channels[perm[i]]."""
    return [channels[i] for i in perm]
