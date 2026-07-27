"""EA reference scope and the frozen-backbone feature cache. Phase 1.5 / caching.

Two things must hold before any Phase 2 run is trustworthy:

1. `--ea-ref session` must reproduce the wrapper's previous in-load EA exactly,
   so the new default is not silently a different experiment.
2. Training the head on cached backbone features must be equivalent to running the
   whole model end-to-end, so the cache is a speedup and not a change of method.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.adapt_stage3 import _apply_ea, _features, _head_auc
from src.preprocessing.alignment import EuclideanAligner


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _epochs(rng, n=60, c=8, t=64):
    return (rng.standard_normal((n, c, t)) * rng.uniform(0.5, 2.0, (1, c, 1))).astype(np.float32)


# --------------------------------------------------------------------------- EA scope

def test_session_ea_matches_wrapper_behaviour(rng):
    """--ea-ref session == EuclideanAligner().fit_transform over all the epochs.

    The wrapper applied exactly this, per subject, inside load_epochs.
    """
    X = _epochs(rng)
    wrapper_style = EuclideanAligner().fit_transform(X)
    session_style = _apply_ea(X, fit_idx=None)
    np.testing.assert_allclose(wrapper_style, session_style, rtol=1e-6, atol=1e-6)


def test_calibration_ea_differs_from_session(rng):
    X = _epochs(rng)
    calib = np.arange(10)
    assert not np.allclose(_apply_ea(X, None), _apply_ea(X, calib))


def test_calibration_ea_uses_only_calibration_epochs(rng):
    """Changing evaluation epochs must not change a calibration-fit reference."""
    X = _epochs(rng)
    calib = np.arange(10)
    a = _apply_ea(X, calib)

    X2 = X.copy()
    X2[20:] *= 5.0  # perturb only non-calibration epochs
    b = _apply_ea(X2, calib)

    # the transform applied to the calibration epochs is unchanged
    np.testing.assert_allclose(a[calib], b[calib], rtol=1e-5, atol=1e-6)


def test_session_ea_is_affected_by_evaluation_epochs(rng):
    """The transductive property that makes session EA unachievable at deployment."""
    X = _epochs(rng)
    calib = np.arange(10)
    a = _apply_ea(X, None)
    X2 = X.copy()
    X2[20:] *= 5.0
    b = _apply_ea(X2, None)
    assert not np.allclose(a[calib], b[calib], rtol=1e-3)


def test_ea_whitens_covariance(rng):
    X = _epochs(rng, n=200, c=6, t=128)
    Xa = _apply_ea(X, None)
    cov = np.einsum("nct,ndt->cd", Xa, Xa) / (Xa.shape[0] * Xa.shape[2])
    np.testing.assert_allclose(cov, np.eye(6), atol=0.05)


# --------------------------------------------------------------------------- feature cache

class _Backbone(torch.nn.Module):
    def __init__(self, c, t, d=16):
        super().__init__()
        self.lin = torch.nn.Linear(c * t, d)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        return self.drop(self.lin(x.flatten(1)))


class _Model(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x))


def _make_model(c=8, t=64, d=16):
    torch.manual_seed(0)
    m = _Model(_Backbone(c, t, d), torch.nn.Linear(d, 1))
    m.eval()
    for p in m.backbone.parameters():
        p.requires_grad_(False)
    return m


def test_cached_features_match_live_forward(rng):
    """backbone(x) computed in batches == computed in one pass."""
    X = _epochs(rng, n=100)
    m = _make_model()
    a = _features(m, X, batch_size=256)
    b = _features(m, X, batch_size=7)
    torch.testing.assert_close(a, b)


def test_head_on_cached_features_equals_end_to_end(rng):
    """head(cached_features) == model(x), the identity the cache relies on."""
    X = _epochs(rng, n=50)
    m = _make_model()
    feats = _features(m, X)
    with torch.no_grad():
        via_cache = m.head(feats)
        end_to_end = m(torch.from_numpy(X).float())
    torch.testing.assert_close(via_cache, end_to_end)


def test_cache_auc_equals_live_auc(rng):
    X = _epochs(rng, n=80)
    y = (rng.random(80) > 0.5).astype(int)
    m = _make_model()
    feats = _features(m, X)
    from sklearn.metrics import roc_auc_score

    with torch.no_grad():
        live = float(roc_auc_score(y, torch.sigmoid(m(torch.from_numpy(X).float()).squeeze(-1)).numpy()))
    cached = _head_auc(m.head, feats, y)
    assert cached == pytest.approx(live, abs=1e-9)


def test_features_are_deterministic_despite_dropout(rng):
    """The backbone stays in eval, so dropout must not perturb cached features."""
    X = _epochs(rng, n=40)
    m = _make_model()
    torch.testing.assert_close(_features(m, X), _features(m, X))


def test_slicing_cache_equals_computing_on_slice(rng):
    """features[idx] == features(X[idx]) — what makes one cache serve every split."""
    X = _epochs(rng, n=60)
    m = _make_model()
    feats = _features(m, X)
    idx = np.array([3, 17, 42, 5])
    torch.testing.assert_close(feats[idx], _features(m, X[idx]))
