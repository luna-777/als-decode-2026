"""Milestone 7 checkpoint — Stage 2 BNCI2014_009 P300 pipeline.

All tests use mocked MOABB calls; no data download required.
Run with: pytest tests/test_m7_stage2.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spec(channels=None, exclude_subjects=None):
    from src.datasets.registry import DatasetSpec
    return DatasetSpec(
        moabb_name="BNCI2014_009",
        paradigm="p300",
        channels=channels or ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
        sfreq_target=128.0,
        exclude_subjects=exclude_subjects or [],
        band=(1.0, 24.0),
        epoch_window=(0.0, 0.8),
    )


def _make_moabb_p300_output(n_epochs=100, n_channels=8, n_times=102, n_targets=17):
    """Synthetic output matching moabb.paradigms.P300.get_data() format."""
    import pandas as pd

    X = np.random.randn(n_epochs, n_channels, n_times).astype(np.float32)
    y_str = np.array(
        ["Target"] * n_targets + ["NonTarget"] * (n_epochs - n_targets)
    )
    meta_df = pd.DataFrame({"subject": np.ones(n_epochs, dtype=int)})
    return X, y_str, meta_df


# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------

class TestDatasetSpec:
    def test_p300_paradigm_accepted(self):
        spec = _make_spec()
        assert spec.paradigm == "p300"

    def test_epoch_window_stored(self):
        spec = _make_spec()
        assert spec.epoch_window == (0.0, 0.8)

    def test_band_stored(self):
        spec = _make_spec()
        assert spec.band == (1.0, 24.0)

    def test_channels_stored(self):
        spec = _make_spec()
        assert "Pz" in spec.channels
        assert len(spec.channels) == 8


# ---------------------------------------------------------------------------
# registry.get_dataset routing
# ---------------------------------------------------------------------------

class TestGetDatasetRouting:
    def test_p300_returns_bnci_wrapper(self):
        from src.datasets.registry import get_dataset
        from src.datasets.wrapper import BnciP300Wrapper

        spec = _make_spec()
        mock_ds = MagicMock()
        mock_ds.subject_list = list(range(1, 11))

        # BNCI2014_009 is imported lazily inside __init__; patch at the source
        with patch("moabb.datasets.BNCI2014_009", return_value=mock_ds):
            wrapper = get_dataset(spec)
        assert isinstance(wrapper, BnciP300Wrapper)

    def test_mi_still_returns_moabb_wrapper(self):
        from src.datasets.registry import get_dataset, DatasetSpec
        from src.datasets.wrapper import MoabbDatasetWrapper

        spec = DatasetSpec(
            moabb_name="PhysionetMI",
            paradigm="mi",
            channels=["C3", "Cz", "C4"],
            sfreq_target=160.0,
            exclude_subjects=[88, 92, 100],
        )
        mock_ds = MagicMock()
        mock_ds.subject_list = list(range(1, 10))

        with patch("moabb.datasets.PhysionetMI", return_value=mock_ds):
            wrapper = get_dataset(spec)
        assert isinstance(wrapper, MoabbDatasetWrapper)


# ---------------------------------------------------------------------------
# BnciP300Wrapper
# ---------------------------------------------------------------------------

class TestBnciP300Wrapper:
    def _make_wrapper(self, channels=None, exclude_subjects=None):
        from src.datasets.wrapper import BnciP300Wrapper

        spec = _make_spec(channels=channels, exclude_subjects=exclude_subjects)
        mock_ds = MagicMock()
        mock_ds.subject_list = list(range(1, 11))

        # BNCI2014_009 imported lazily inside __init__; patch at source
        with patch("moabb.datasets.BNCI2014_009", return_value=mock_ds):
            wrapper = BnciP300Wrapper(spec)
        return wrapper

    def test_subject_list_length(self):
        wrapper = self._make_wrapper()
        assert len(wrapper.subject_list) == 10

    def test_exclude_subjects_applied(self):
        wrapper = self._make_wrapper(exclude_subjects=[1, 2])
        assert 1 not in wrapper.subject_list
        assert 2 not in wrapper.subject_list
        assert len(wrapper.subject_list) == 8

    def test_subject_list_returns_copy(self):
        wrapper = self._make_wrapper()
        lst = wrapper.subject_list
        lst.append(999)
        assert 999 not in wrapper.subject_list

    def test_load_epochs_shape(self):
        wrapper = self._make_wrapper()
        X_mock, y_str, meta_df = _make_moabb_p300_output(n_epochs=100)

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            X, y, meta = wrapper.load_epochs([1])

        assert X.ndim == 3
        assert X.shape[0] == 100
        assert X.shape[1] == 8  # all 8 channels
        assert X.shape[2] == 102

    def test_load_epochs_dtype(self):
        wrapper = self._make_wrapper()
        X_mock, y_str, meta_df = _make_moabb_p300_output()

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            X, y, meta = wrapper.load_epochs([1])

        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_target_label_is_one(self):
        wrapper = self._make_wrapper()
        n_targets = 20
        X_mock, y_str, meta_df = _make_moabb_p300_output(n_epochs=80, n_targets=n_targets)

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            _, y, _ = wrapper.load_epochs([1])

        assert int((y == 1).sum()) == n_targets
        assert int((y == 0).sum()) == 80 - n_targets

    def test_only_binary_labels(self):
        wrapper = self._make_wrapper()
        X_mock, y_str, meta_df = _make_moabb_p300_output()

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            _, y, _ = wrapper.load_epochs([1])

        unique = set(y.tolist())
        assert unique.issubset({0, 1})

    def test_channel_selection_reorders(self):
        """Requesting a subset of channels returns the correct columns."""
        subset = ["Cz", "Pz"]
        wrapper = self._make_wrapper(channels=subset)

        # Full 8-channel data with distinctive values per channel
        X_mock = np.zeros((10, 8, 102), dtype=np.float32)
        for i in range(8):
            X_mock[:, i, :] = i  # channel i has value i
        _, y_str, meta_df = _make_moabb_p300_output(n_epochs=10)
        y_str[:] = "NonTarget"

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        from src.datasets.wrapper import _BNCI009_CHANNELS
        cz_idx = _BNCI009_CHANNELS.index("Cz")  # = 1
        pz_idx = _BNCI009_CHANNELS.index("Pz")  # = 2

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            X, _, _ = wrapper.load_epochs([1])

        assert X.shape[1] == 2
        np.testing.assert_allclose(X[:, 0, :], cz_idx)  # first output = Cz
        np.testing.assert_allclose(X[:, 1, :], pz_idx)  # second output = Pz

    def test_metadata_contains_subjects(self):
        wrapper = self._make_wrapper()
        X_mock, y_str, meta_df = _make_moabb_p300_output(n_epochs=50)

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            _, _, meta = wrapper.load_epochs([1])

        assert "subjects" in meta
        assert len(meta["subjects"]) == 50

    def test_metadata_sfreq(self):
        wrapper = self._make_wrapper()
        X_mock, y_str, meta_df = _make_moabb_p300_output()

        mock_paradigm = MagicMock()
        mock_paradigm.get_data.return_value = (X_mock, y_str, meta_df)

        with patch("moabb.paradigms.P300", return_value=mock_paradigm):
            _, _, meta = wrapper.load_epochs([1])

        assert meta["sfreq"] == 128.0


# ---------------------------------------------------------------------------
# n_times consistency: wrapper vs train.py formula
# ---------------------------------------------------------------------------

class TestNTimesConsistency:
    """Ensure the number of time samples from BnciP300Wrapper matches train.py's formula."""

    def test_n_times_matches_train_formula(self):
        spec = _make_spec()
        n_times_train = int(round(spec.sfreq_target * (spec.epoch_window[1] - spec.epoch_window[0])))
        # 128 * 0.8 = 102.4 → round → 102
        assert n_times_train == 102

    def test_tmax_adjustment_yields_correct_n_times(self):
        """Verify the tmax - 1/sfreq trick gives MNE exactly n_times samples."""
        sfreq = 128.0
        epoch_end = 0.8
        tmin = 0.0
        tmax_adjusted = epoch_end - 1.0 / sfreq  # 0.7921875
        # MNE formula: round((tmax - tmin) * sfreq) + 1
        mne_n_times = int(round((tmax_adjusted - tmin) * sfreq)) + 1
        # train.py formula
        train_n_times = int(round(sfreq * (epoch_end - tmin)))
        assert mne_n_times == train_n_times == 102


# ---------------------------------------------------------------------------
# Config loading (build_spec_from_cfg)
# ---------------------------------------------------------------------------

class TestBnciConfigParsing:
    def test_build_spec_from_bnci_yaml(self):
        """build_spec_from_cfg should parse bnci_009.yaml without errors."""
        from omegaconf import OmegaConf
        from src.training.datamodule import build_spec_from_cfg

        cfg_dict = {
            "moabb_name": "BNCI2014_009",
            "paradigm": "p300",
            "sfreq_target": 128.0,
            "exclude_subjects": [],
            "channel_config": "default",
            "channels": {
                "default": ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
            },
            "epoch_window": [0.0, 0.8],
            "band": [1.0, 24.0],
        }
        cfg = OmegaConf.create(cfg_dict)
        spec = build_spec_from_cfg(cfg)

        assert spec.paradigm == "p300"
        assert spec.moabb_name == "BNCI2014_009"
        assert spec.sfreq_target == 128.0
        assert len(spec.channels) == 8
        assert spec.epoch_window == (0.0, 0.8)
        assert spec.band == (1.0, 24.0)
        assert spec.exclude_subjects == []

    def test_channel_order_preserved(self):
        from omegaconf import OmegaConf
        from src.training.datamodule import build_spec_from_cfg

        expected = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
        cfg = OmegaConf.create({
            "moabb_name": "BNCI2014_009",
            "paradigm": "p300",
            "sfreq_target": 128.0,
            "exclude_subjects": [],
            "channel_config": "default",
            "channels": {"default": expected},
            "epoch_window": [0.0, 0.8],
            "band": [1.0, 24.0],
        })
        spec = build_spec_from_cfg(cfg)
        assert list(spec.channels) == expected


# ---------------------------------------------------------------------------
# BackboneEncoder: Stage 2 shape
# ---------------------------------------------------------------------------

class TestStage2BackboneShape:
    def test_backbone_forward_8ch_102t(self):
        """EEGDecoder forward pass with Stage 2 dimensions should not error."""
        import torch
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder

        backbone = BackboneEncoder(n_channels=8, n_times=102)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        model = EEGDecoder(backbone, head)

        x = torch.zeros(4, 8, 102)
        out = model(x)
        assert out.shape == (4, 1)

    def test_feature_dim_positive(self):
        from src.models.backbone import BackboneEncoder

        backbone = BackboneEncoder(n_channels=8, n_times=102)
        assert backbone.feature_dim > 0
