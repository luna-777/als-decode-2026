"""Milestone 4 checkpoint — models: EEGNet backbone + CSP-LDA pipeline.

No data download required.
Run with: pytest tests/test_m4_models.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


N_CH, N_T = 17, 320
B = 4


# ---------------------------------------------------------------------------
# BackboneEncoder
# ---------------------------------------------------------------------------

class TestBackboneEncoder:
    def test_output_shape(self):
        from src.models.backbone import BackboneEncoder
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        x = torch.randn(B, N_CH, N_T)
        out = enc(x)
        assert out.ndim == 2
        assert out.shape[0] == B

    def test_feature_dim_property(self):
        from src.models.backbone import BackboneEncoder
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        x = torch.randn(B, N_CH, N_T)
        out = enc(x)
        assert out.shape[1] == enc.feature_dim

    def test_freeze_stops_gradients(self):
        from src.models.backbone import BackboneEncoder
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        enc.freeze()
        assert all(not p.requires_grad for p in enc.parameters())

    def test_insert_adapters_raises(self):
        from src.models.backbone import BackboneEncoder
        with pytest.raises(NotImplementedError):
            BackboneEncoder(n_channels=N_CH, n_times=N_T, insert_adapters=True)


# ---------------------------------------------------------------------------
# DecoderHead
# ---------------------------------------------------------------------------

class TestDecoderHead:
    def test_binary_output_shape(self):
        from src.models.backbone import BackboneEncoder, DecoderHead
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        head = DecoderHead(feature_dim=enc.feature_dim, n_classes=1)
        feats = torch.randn(B, enc.feature_dim)
        out = head(feats)
        assert out.shape == (B, 1)

    def test_max_norm_applied(self):
        """After a forward pass with large weights, norms must be clipped."""
        from src.models.backbone import DecoderHead
        head = DecoderHead(feature_dim=8, n_classes=1, max_norm=0.25)
        # Force weights above max_norm
        with torch.no_grad():
            head.linear.weight.fill_(10.0)
        _ = head(torch.randn(2, 8))
        norms = head.linear.weight.norm(dim=1)
        assert (norms <= 0.25 + 1e-5).all(), f"Max-norm violated: {norms}"


# ---------------------------------------------------------------------------
# EEGDecoder (backbone + head)
# ---------------------------------------------------------------------------

class TestEEGDecoder:
    def test_end_to_end_shape(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        head = DecoderHead(feature_dim=enc.feature_dim, n_classes=1)
        model = EEGDecoder(enc, head)
        x = torch.randn(B, N_CH, N_T)
        out = model(x)
        assert out.shape == (B, 1)

    def test_stage3_freezes_backbone(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        head = DecoderHead(feature_dim=enc.feature_dim, n_classes=1)
        model = EEGDecoder(enc, head, stage="stage3_adapted")
        assert all(not p.requires_grad for p in model.backbone.parameters())
        # Head should still be trainable
        assert any(p.requires_grad for p in model.head.parameters())

    def test_stage1_all_trainable(self):
        from src.models.backbone import BackboneEncoder, DecoderHead, EEGDecoder
        enc = BackboneEncoder(n_channels=N_CH, n_times=N_T)
        head = DecoderHead(feature_dim=enc.feature_dim, n_classes=1)
        model = EEGDecoder(enc, head, stage="stage1")
        assert any(p.requires_grad for p in model.backbone.parameters())


# ---------------------------------------------------------------------------
# CSP-LDA classical pipeline
# ---------------------------------------------------------------------------

class TestCSPPipeline:
    def _make_data(self, n=60):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, N_CH, N_T)).astype(np.float32)
        y = rng.integers(0, 2, size=n)
        return X, y

    def test_pipeline_fits_and_predicts(self):
        from src.models.classical import build_mi_csp_pipeline
        pipe = build_mi_csp_pipeline(n_components=4)
        X, y = self._make_data()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds.tolist()) <= {0, 1}

    def test_pipeline_predict_proba(self):
        from src.models.classical import build_mi_csp_pipeline
        pipe = build_mi_csp_pipeline(n_components=4)
        X, y = self._make_data()
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_tangent_space_variant(self):
        from src.models.classical import build_mi_csp_pipeline
        pipe = build_mi_csp_pipeline(tangent_space=True)
        X, y = self._make_data()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(y),)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLosses:
    def test_weighted_bce_shape(self):
        from src.training.losses import weighted_bce
        logits = torch.randn(16)
        targets = torch.randint(0, 2, (16,))
        loss = weighted_bce(logits, targets)
        assert loss.shape == ()  # scalar

    def test_weighted_bce_with_pos_weight(self):
        from src.training.losses import weighted_bce
        logits = torch.randn(16)
        targets = torch.randint(0, 2, (16,))
        pw = torch.tensor([2.0])
        loss_weighted = weighted_bce(logits, targets, pos_weight=pw)
        loss_plain = weighted_bce(logits, targets)
        # Weighted loss differs from plain loss
        assert not torch.isclose(loss_weighted, loss_plain)

    def test_focal_loss_shape(self):
        from src.training.losses import focal_loss
        logits = torch.randn(16)
        targets = torch.randint(0, 2, (16,)).float()
        loss = focal_loss(logits, targets, gamma=2.0)
        assert loss.shape == ()

    def test_focal_leq_bce_on_easy_examples(self):
        """On very easy examples (logit=10 for correct class) focal < BCE."""
        from src.training.losses import focal_loss, weighted_bce
        # All positive, very high logit → very easy
        logits = torch.full((8,), 10.0)
        targets = torch.ones(8)
        fl = focal_loss(logits, targets, gamma=2.0)
        bce = weighted_bce(logits, targets)
        assert fl.item() < bce.item()
