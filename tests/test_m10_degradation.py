"""Montage degradation instrument. Round A."""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.degradation import (
    PREAUDIT_MI_MONTAGE,
    apply_permutation,
    derangement,
    n_displaced,
    permutation_for_k,
    permuted_channel_names,
)

CFG = ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
       "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4"]


@pytest.mark.parametrize("k", [0, 2, 4, 8, 17])
def test_exactly_k_positions_move(k):
    perm = permutation_for_k(17, k, seed=0)
    assert n_displaced(perm) == k


@pytest.mark.parametrize("k", [0, 2, 4, 8, 17])
def test_result_is_a_permutation(k):
    perm = permutation_for_k(17, k, seed=3)
    assert sorted(perm.tolist()) == list(range(17))


def test_k0_is_identity_for_every_seed():
    for s in range(5):
        np.testing.assert_array_equal(permutation_for_k(17, 0, s), np.arange(17))


def test_k1_is_rejected():
    with pytest.raises(ValueError, match="k=1 is impossible"):
        permutation_for_k(17, 1, seed=0)


def test_k_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"k must be in"):
        permutation_for_k(17, 18, seed=0)


def test_seeds_give_different_permutations():
    a = permutation_for_k(17, 8, seed=1)
    b = permutation_for_k(17, 8, seed=2)
    assert not np.array_equal(a, b)


def test_seed_is_deterministic():
    np.testing.assert_array_equal(
        permutation_for_k(17, 8, seed=7), permutation_for_k(17, 8, seed=7)
    )


def test_k17_has_no_fixed_point():
    perm = permutation_for_k(17, 17, seed=0)
    assert np.all(perm != np.arange(17))


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for k in (2, 3, 5, 17):
        d = derangement(k, rng)
        assert np.all(d != np.arange(k))


# --------------------------------------------------------------------------- array

def test_apply_permutation_gathers_correctly():
    X = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
    perm = permutation_for_k(17, 8, seed=1)
    out = apply_permutation(X, perm)
    for i in range(17):
        np.testing.assert_array_equal(out[:, i, :], X[:, perm[i], :])


def test_identity_permutation_is_a_noop():
    X = np.random.RandomState(0).randn(4, 17, 5).astype(np.float32)
    np.testing.assert_array_equal(apply_permutation(X, permutation_for_k(17, 0, 0)), X)


def test_length_mismatch_raises():
    X = np.zeros((2, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Permutation length"):
        apply_permutation(X, permutation_for_k(17, 2, 0))


def test_permuted_names_track_the_array():
    perm = permutation_for_k(17, 4, seed=2)
    names = permuted_channel_names(CFG, perm)
    assert len(names) == 17
    assert sorted(names) == sorted(CFG)          # same electrode set
    assert sum(a != b for a, b in zip(names, CFG)) == 4


# --------------------------------------------------------------------------- EA commutes

def test_ea_commutes_with_channel_permutation():
    """The identity that lets us permute a verified load instead of reloading."""
    from src.preprocessing.alignment import EuclideanAligner

    rng = np.random.default_rng(0)
    X = (rng.standard_normal((80, 17, 64)) * rng.uniform(0.5, 2, (1, 17, 1))).astype(np.float32)
    perm = permutation_for_k(17, 8, seed=5)

    # permute then align
    a = EuclideanAligner().fit_transform(apply_permutation(X, perm))
    # align then permute
    b = apply_permutation(EuclideanAligner().fit_transform(X), perm)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- pre-audit list

def test_preaudit_montage_is_not_a_permutation_of_config():
    """It swapped three electrodes, so it cannot be reached by permuting."""
    assert len(PREAUDIT_MI_MONTAGE) == 17
    assert sorted(PREAUDIT_MI_MONTAGE) != sorted(CFG)
    assert set(PREAUDIT_MI_MONTAGE) - set(CFG) == {"FC5", "FC6", "CP5"}
    assert set(CFG) - set(PREAUDIT_MI_MONTAGE) == {"CPz", "CP2", "CP4"}


def test_preaudit_montage_shares_no_position_with_config():
    assert sum(a == b for a, b in zip(PREAUDIT_MI_MONTAGE, CFG)) == 0


# --------------------------------------------------------------------------- Phase 4 montage

def test_als_montage_is_a_strict_subset_of_the_healthy_montage():
    """§4.3's transfer claim, checked against config rather than against design.md.

    docs/design.md §4.2 states BNCI2014_009's channel order incorrectly
    (docs/AUDIT.md §0.2), so the subset property is asserted from the configs that
    the code actually reads and, at run time, by the by-name channel resolution.
    """
    from src.models.checkpoints import spec_from_config

    als = spec_from_config("als")
    healthy = spec_from_config("p300")
    assert als.moabb_name == "BNCI2014_008"
    assert set(als.channels).issubset(set(healthy.channels))
    # identical order, so a Stage 2 checkpoint transfers with no reindexing at all
    assert als.channels == healthy.channels
    assert als.sfreq_target == healthy.sfreq_target
    assert tuple(als.epoch_window) == tuple(healthy.epoch_window)
    assert tuple(als.band) == tuple(healthy.band)
