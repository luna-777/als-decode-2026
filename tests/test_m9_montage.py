"""Montage contract and channel-selection regression tests. docs/AUDIT.md §0.1, §0.2."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.datasets.montage import (
    Montage,
    assert_montage_matches,
    load_montage,
    write_montage,
)
from src.datasets.registry import DatasetSpec
from src.datasets.wrapper import _select_channels_by_name

# The two lists that actually diverged in the pre-audit repo (docs/AUDIT.md §0.1).
TRAINED = [
    "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
]
STAGE3_HARDCODED = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3",
    "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1",
]


def _spec(channels):
    return DatasetSpec(
        moabb_name="PhysionetMI", paradigm="mi", channels=list(channels),
        sfreq_target=160.0, band=(8.0, 30.0), epoch_window=(0.0, 2.0),
    )


# --------------------------------------------------------------------------- §0.1

def test_equal_length_different_order_raises():
    """The core ADR-10 case: same length, different order — must never reindex."""
    montage = Montage(tuple(TRAINED), 160.0, 320, "mi", "recorded")
    with pytest.raises(ValueError) as exc:
        assert_montage_matches(montage, STAGE3_HARDCODED)
    msg = str(exc.value)
    # Both lists appear in the message.
    assert "FC3" in msg and "FC5" in msg
    assert "17" in msg
    # It must say explicitly that no shape error would have caught this.
    assert "NOT have raised a shape error" in msg


def test_equal_length_reordered_only_raises():
    """Same electrode set, permuted order — the silent case."""
    montage = Montage(tuple(TRAINED), 160.0, 320, "mi", "recorded")
    permuted = list(reversed(TRAINED))
    with pytest.raises(ValueError, match="Montage mismatch"):
        assert_montage_matches(montage, permuted)


def test_single_position_swap_raises():
    montage = Montage(tuple(TRAINED), 160.0, 320, "mi", "recorded")
    swapped = list(TRAINED)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError) as exc:
        assert_montage_matches(montage, swapped)
    assert "2/17 positions disagree" in str(exc.value)


def test_identical_montage_passes():
    montage = Montage(tuple(TRAINED), 160.0, 320, "mi", "recorded")
    assert_montage_matches(montage, list(TRAINED))  # no raise


def test_mismatch_message_names_the_disjoint_electrodes():
    montage = Montage(tuple(TRAINED), 160.0, 320, "mi", "recorded")
    with pytest.raises(ValueError) as exc:
        assert_montage_matches(montage, STAGE3_HARDCODED)
    msg = str(exc.value)
    assert "CPz" in msg  # trained on, absent from data
    assert "FC5" in msg  # in data, never trained on


def test_different_length_also_raises():
    montage = Montage(tuple(TRAINED), 160.0, 320, "mi", "recorded")
    with pytest.raises(ValueError, match="Montage mismatch"):
        assert_montage_matches(montage, TRAINED[:8])


# --------------------------------------------------------------------------- round-trip

def test_write_then_load_round_trip(tmp_path):
    spec = _spec(TRAINED)
    write_montage(tmp_path, spec, n_times=320)
    loaded = load_montage(tmp_path, spec, 320)
    assert loaded.source == "recorded"
    assert list(loaded.channels) == TRAINED
    assert loaded.n_channels == 17
    assert_montage_matches(loaded, TRAINED)


def test_recorded_montage_beats_a_wrong_config(tmp_path):
    """A recorded montage must be checked against the data, not against the config.

    Writing the trained montage then loading with a *different* spec must still
    surface the trained list — otherwise the config could paper over a mismatch.
    """
    write_montage(tmp_path, _spec(TRAINED), n_times=320)
    loaded = load_montage(tmp_path, _spec(STAGE3_HARDCODED), 320)
    assert list(loaded.channels) == TRAINED
    with pytest.raises(ValueError):
        assert_montage_matches(loaded, STAGE3_HARDCODED)


def test_missing_montage_infers_from_config(tmp_path):
    loaded = load_montage(tmp_path, _spec(TRAINED), 320)
    assert loaded.source == "inferred_from_config"
    assert list(loaded.channels) == TRAINED


def test_missing_montage_raises_under_strict(tmp_path):
    with pytest.raises(ValueError, match="predates the montage contract"):
        load_montage(tmp_path, _spec(TRAINED), 320, strict=True)


def test_montage_json_is_readable(tmp_path):
    write_montage(tmp_path, _spec(TRAINED), n_times=320)
    d = json.loads((tmp_path / "montage.json").read_text())
    assert d["channels"] == TRAINED
    assert d["n_times"] == 320
    assert d["paradigm"] == "mi"


# --------------------------------------------------------------------------- §0.2

BNCI009_RETURNED = [
    "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4",
]
ALS_SUBSET = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


def test_selects_by_name_not_position():
    X = np.arange(2 * 16 * 5, dtype=np.float32).reshape(2, 16, 5)
    out = _select_channels_by_name(X, BNCI009_RETURNED, ALS_SUBSET)
    assert out.shape == (2, 8, 5)
    # For this dataset positions 0-7 happen to be the wanted set, so a positional
    # slice coincides — that coincidence is exactly what §0.2 flagged.
    np.testing.assert_array_equal(out, X[:, :8, :])


def test_reordered_source_still_selects_correct_electrodes():
    """If MOABB ever reorders, name-based selection must still be right."""
    reordered = BNCI009_RETURNED[8:] + BNCI009_RETURNED[:8]
    X = np.random.RandomState(0).randn(3, 16, 4).astype(np.float32)
    X_re = X[:, [BNCI009_RETURNED.index(c) for c in reordered], :]
    a = _select_channels_by_name(X, BNCI009_RETURNED, ALS_SUBSET)
    b = _select_channels_by_name(X_re, reordered, ALS_SUBSET)
    np.testing.assert_allclose(a, b)
    # ...and a naive positional slice would NOT have been right.
    assert not np.allclose(b, X_re[:, :8, :])


def test_missing_channel_raises():
    X = np.zeros((1, 16, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="not present in the data"):
        _select_channels_by_name(X, BNCI009_RETURNED, ALS_SUBSET + ["T7"])


def test_requested_order_is_preserved():
    X = np.arange(1 * 16 * 3, dtype=np.float32).reshape(1, 16, 3)
    want = ["PO8", "Fz", "CPz"]
    out = _select_channels_by_name(X, BNCI009_RETURNED, want)
    for i, ch in enumerate(want):
        np.testing.assert_array_equal(out[:, i, :], X[:, BNCI009_RETURNED.index(ch), :])


def test_duplicate_source_names_raise():
    dupes = ["Fz"] + BNCI009_RETURNED[:15]
    X = np.zeros((1, 16, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate channel names"):
        _select_channels_by_name(X, dupes, ["Fz"])
