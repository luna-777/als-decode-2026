"""Deterministic checkpoint resolution. docs/AUDIT.md §0.4 (items 5 and 6)."""
from __future__ import annotations

import json

import pytest

from src.models.checkpoints import (
    discover_versions,
    load_fold,
    load_folds,
    loso_fold_for_subject,
    spec_from_config,
)


def _make_version(root, name, auc, *, preprocessor=True, val_subjects=None, n_ckpts=1):
    v = root / name
    for i in range(n_ckpts):
        d = v / "checkpoints" / f"best-epoch={i:03d}-val"
        d.mkdir(parents=True)
        (d / f"roc_auc={auc:.4f}.ckpt").write_text("stub")
    if preprocessor:
        (v / "preprocessor.pkl").write_text("stub")
    if val_subjects is not None:
        (v / "val_subjects.json").write_text(json.dumps(val_subjects))
    return v


# --------------------------------------------------------------------------- pinning

def test_named_fold_loads(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_28", 0.8016)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 17)
    ref = load_fold(tmp_path, "version_28")
    assert ref.version == "version_28"
    assert ref.val_auc == pytest.approx(0.8016)
    assert ref.preprocessor_path.exists()


def test_unknown_version_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such version directory"):
        load_fold(tmp_path, "version_999")


def test_missing_preprocessor_raises(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_1", 0.77, preprocessor=False)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 17)
    with pytest.raises(FileNotFoundError, match="No preprocessor.pkl"):
        load_fold(tmp_path, "version_1")


def test_ambiguous_multiple_checkpoints_raises(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_1", 0.77, n_ckpts=2)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 17)
    with pytest.raises(ValueError, match="expected exactly"):
        load_fold(tmp_path, "version_1")


def test_no_folds_named_raises(tmp_path):
    """The old code silently picked the best on disk; now it must be told."""
    with pytest.raises(ValueError, match="named explicitly"):
        load_folds(tmp_path, [])


def test_folds_disagreeing_on_channels_raise(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_1", 0.77)
    _make_version(tmp_path, "version_2", 0.78)
    monkeypatch.setattr(
        "src.models.checkpoints.n_channels_of",
        lambda p: 17 if "version_1" in str(p) else 8,
    )
    with pytest.raises(ValueError, match="disagree on channel count"):
        load_folds(tmp_path, ["version_1", "version_2"])


def test_channel_count_must_match_config(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_1", 0.77)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 8)
    with pytest.raises(ValueError, match="disagree on the montage size"):
        load_folds(tmp_path, ["version_1"], expect_channels=17)


def test_selection_is_independent_of_what_else_is_on_disk(tmp_path, monkeypatch):
    """Item 5: adding a higher-AUC checkpoint must not change the resolved fold."""
    _make_version(tmp_path, "version_28", 0.8016)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 17)
    before = load_folds(tmp_path, ["version_28"])[0].ckpt_path

    _make_version(tmp_path, "version_99", 0.9999)  # would have won the old scan
    after = load_folds(tmp_path, ["version_28"])[0].ckpt_path
    assert before == after


# --------------------------------------------------------------------------- LOSO

def test_loso_resolves_the_fold_that_held_the_subject_out(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_31", 0.85, val_subjects=[1])
    _make_version(tmp_path, "version_32", 0.77, val_subjects=[2])
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 8)
    assert loso_fold_for_subject(tmp_path, 1).version == "version_31"
    assert loso_fold_for_subject(tmp_path, 2).version == "version_32"


def test_loso_raises_rather_than_falling_back(tmp_path, monkeypatch):
    """The old fallback scored a subject against a model that may have trained on them."""
    _make_version(tmp_path, "version_31", 0.85, val_subjects=[1])
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 8)
    with pytest.raises(FileNotFoundError, match="No fold held out subject 7"):
        loso_fold_for_subject(tmp_path, 7)


def test_loso_with_no_metadata_raises(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_1", 0.85)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 8)
    with pytest.raises(FileNotFoundError, match="val_subjects.json"):
        loso_fold_for_subject(tmp_path, 1)


def test_loso_ambiguous_assignment_raises(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_31", 0.85, val_subjects=[1, 2])
    _make_version(tmp_path, "version_32", 0.77, val_subjects=[2, 3])
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 8)
    with pytest.raises(ValueError, match="multiple folds"):
        loso_fold_for_subject(tmp_path, 2)


# --------------------------------------------------------------------------- config sourcing

def test_spec_from_config_matches_training_channel_list():
    """Item 1: Stage 3 must read the same list the training path reads."""
    spec = spec_from_config("mi")
    assert spec.channels == [
        "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
    ]
    assert len(spec.channels) == 17
    assert spec.euclidean_alignment is True


def test_spec_from_config_never_returns_the_old_hardcoded_list():
    spec = spec_from_config("mi")
    assert "FC5" not in spec.channels
    assert "FC6" not in spec.channels
    assert "CP5" not in spec.channels


def test_p300_spec_from_config():
    spec = spec_from_config("p300")
    assert spec.channels == ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


def test_discover_is_inspection_only(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_1", 0.70)
    _make_version(tmp_path, "version_2", 0.90)
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 17)
    assert discover_versions(tmp_path, 17) == ["version_1", "version_2"]


def test_loso_filters_by_paradigm_channel_count(tmp_path, monkeypatch):
    """MI and P300 subject IDs are both small integers and WILL collide.

    lightning_logs holds folds from both paradigms. An MI fold whose val subjects
    include 2 must not be returned when resolving P300 subject 2.
    """
    _make_version(tmp_path, "version_34", 0.79, val_subjects=[2])          # P300 fold
    _make_version(tmp_path, "version_41", 0.76, val_subjects=[2, 7, 12])   # MI fold
    monkeypatch.setattr(
        "src.models.checkpoints.n_channels_of",
        lambda p: 17 if "version_41" in str(p) else 8,
    )
    # Unfiltered, this is genuinely ambiguous and must raise rather than guess.
    with pytest.raises(ValueError, match="multiple folds"):
        loso_fold_for_subject(tmp_path, 2)
    # Filtered by paradigm, it resolves to the P300 fold.
    assert loso_fold_for_subject(tmp_path, 2, n_channels=8).version == "version_34"
    assert loso_fold_for_subject(tmp_path, 2, n_channels=17).version == "version_41"


def test_loso_channel_filter_can_exclude_everything(tmp_path, monkeypatch):
    _make_version(tmp_path, "version_41", 0.76, val_subjects=[2])
    monkeypatch.setattr("src.models.checkpoints.n_channels_of", lambda p: 17)
    with pytest.raises(FileNotFoundError, match="No fold held out subject 2"):
        loso_fold_for_subject(tmp_path, 2, n_channels=8)
