"""Milestone 8 checkpoint — Stage 3 per-patient adaptation.

All tests use synthetic data; no checkpoint or MOABB download required.
Run with: pytest tests/test_m8_stage3.py -v
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(n_channels=8, n_times=102):
    from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
    backbone = BackboneEncoder(n_channels=n_channels, n_times=n_times)
    head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
    return EEGDecoder(backbone, head)


def _synthetic_calib(n=40, n_channels=8, n_times=102, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_channels, n_times)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Backbone freeze / head reset
# ---------------------------------------------------------------------------

class TestFreezeAndReset:
    def test_stage3_adapted_freezes_backbone(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        backbone = BackboneEncoder(n_channels=8, n_times=102)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        model = EEGDecoder(backbone, head, stage="stage3_adapted")

        for name, p in model.backbone.named_parameters():
            assert not p.requires_grad, f"Backbone param {name} should be frozen"

    def test_head_still_trainable_after_freeze(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        backbone = BackboneEncoder(n_channels=8, n_times=102)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        model = EEGDecoder(backbone, head, stage="stage3_adapted")

        for name, p in model.head.named_parameters():
            assert p.requires_grad, f"Head param {name} should be trainable"

    def test_freeze_does_not_affect_forward(self):
        model = _make_model()
        model.backbone.freeze()
        x = torch.zeros(2, 8, 102)
        out = model(x)
        assert out.shape == (2, 1)

    def test_backbone_weights_unchanged_after_head_training(self):
        """Gradient update on head should not touch frozen backbone weights or buffers."""
        from scripts.adapt_stage3 import _adapt_head
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder

        backbone = BackboneEncoder(n_channels=8, n_times=102)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        model = EEGDecoder(backbone, head, stage="stage3_adapted")

        # Snapshot all parameters AND buffers (e.g. BN running stats)
        backbone_before = {k: v.clone() for k, v in model.backbone.state_dict().items()}

        X, y = _synthetic_calib(n=20)
        _adapt_head(model, X, y, n_epochs=5, lr=1e-2)

        for k, v_before in backbone_before.items():
            v_after = model.backbone.state_dict()[k]
            assert torch.allclose(v_before, v_after), \
                f"Backbone tensor {k} changed during head training (BN should be in eval mode)"


# ---------------------------------------------------------------------------
# _adapt_head
# ---------------------------------------------------------------------------

class TestAdaptHead:
    def test_adapt_head_runs(self):
        from scripts.adapt_stage3 import _adapt_head
        model = _make_model()
        model.backbone.freeze()
        X, y = _synthetic_calib(n=30)
        _adapt_head(model, X, y, n_epochs=3, lr=1e-2)

    def test_adapt_head_returns_model(self):
        from scripts.adapt_stage3 import _adapt_head
        model = _make_model()
        model.backbone.freeze()
        X, y = _synthetic_calib(n=30)
        returned = _adapt_head(model, X, y, n_epochs=3)
        assert returned is model  # same object

    def test_adapt_head_model_in_eval_after(self):
        from scripts.adapt_stage3 import _adapt_head
        model = _make_model()
        model.backbone.freeze()
        X, y = _synthetic_calib(n=20)
        _adapt_head(model, X, y, n_epochs=3)
        assert not model.training

    def test_adapt_head_small_batch(self):
        """Should work even when n_calib < batch_size."""
        from scripts.adapt_stage3 import _adapt_head
        model = _make_model()
        model.backbone.freeze()
        X, y = _synthetic_calib(n=5)
        _adapt_head(model, X, y, n_epochs=3, batch_size=32)

    def test_adapt_head_changes_head_weights(self):
        from scripts.adapt_stage3 import _adapt_head
        model = _make_model()
        model.backbone.freeze()

        head_before = model.head.linear.weight.clone()
        X, y = _synthetic_calib(n=40)
        _adapt_head(model, X, y, n_epochs=10, lr=1e-1)
        head_after = model.head.linear.weight.clone()

        assert not torch.allclose(head_before, head_after), "Head weights should change"

    def test_adapt_handles_class_imbalance(self):
        """Works when calibration set is heavily imbalanced."""
        from scripts.adapt_stage3 import _adapt_head
        model = _make_model()
        model.backbone.freeze()
        X = np.random.randn(30, 8, 102).astype(np.float32)
        y = np.zeros(30, dtype=np.int64)
        y[:2] = 1  # only 2 positives
        _adapt_head(model, X, y, n_epochs=3)


# ---------------------------------------------------------------------------
# _epoch_auc
# ---------------------------------------------------------------------------

class TestEpochAuc:
    def test_random_model_near_chance(self):
        from scripts.adapt_stage3 import _epoch_auc
        torch.manual_seed(0)
        model = _make_model()
        model.eval()
        X, y = _synthetic_calib(n=200, seed=42)
        auc = _epoch_auc(model, X, y)
        assert 0.3 < auc < 0.7, f"Random model AUC should be near 0.5, got {auc}"

    def test_returns_nan_for_single_class(self):
        from scripts.adapt_stage3 import _epoch_auc
        model = _make_model()
        X = np.zeros((10, 8, 102), dtype=np.float32)
        y = np.zeros(10, dtype=np.int64)  # all negative
        auc = _epoch_auc(model, X, y)
        assert np.isnan(auc)


# ---------------------------------------------------------------------------
# EEGDecoder stage modes
# ---------------------------------------------------------------------------

class TestEEGDecoderStage:
    def test_stage1_all_params_trainable(self):
        model = _make_model()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total

    def test_stage3_adapted_only_head_trains(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        backbone = BackboneEncoder(n_channels=8, n_times=102)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        model = EEGDecoder(backbone, head, stage="stage3_adapted")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in model.head.parameters())
        assert trainable == head_params

    def test_stage3_forward_same_shape_as_stage1(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        backbone = BackboneEncoder(n_channels=8, n_times=102)
        head = DecoderHead(feature_dim=backbone.feature_dim, n_classes=1)
        m1 = EEGDecoder(backbone, head, stage="stage1")
        m3 = EEGDecoder(
            BackboneEncoder(n_channels=8, n_times=102),
            DecoderHead(feature_dim=backbone.feature_dim, n_classes=1),
            stage="stage3_adapted",
        )
        x = torch.zeros(4, 8, 102)
        assert m1(x).shape == m3(x).shape == (4, 1)
