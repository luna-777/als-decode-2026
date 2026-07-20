"""Milestone 2 checkpoint — preprocessing pipeline on synthetic data.

All tests use synthetic arrays; no MOABB data download required.
Run with: pytest tests/test_m2_preprocessing.py -v
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_EP, N_CH, N_T = 40, 17, 320  # batch shape: 40 epochs × 17 ch × 320 samples (2 s @ 160 Hz)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_X(rng) -> np.ndarray:
    """Clean synthetic EEG-like data — amplitude kept small so all epochs pass 150 µV."""
    return (rng.standard_normal((N_EP, N_CH, N_T)) * 5.0).astype(np.float32)


@pytest.fixture
def dirty_X(rng) -> np.ndarray:
    """Same low-amplitude base but 5 epochs spiked clearly above 150 µV threshold."""
    X = (rng.standard_normal((N_EP, N_CH, N_T)) * 5.0).astype(np.float32)
    for i in range(5):
        X[i, 0, 10] = 200.0  # spike on first channel
    return X


@pytest.fixture
def cfg_mi8_30() -> dict:
    return {
        "band": [8.0, 30.0],
        "keep_sfreq": 160.0,
        "robust_scale": True,
        "amplitude_reject_uv": 150.0,
        "euclidean_alignment": False,
    }


# ---------------------------------------------------------------------------
# RobustChannelScaler
# ---------------------------------------------------------------------------

class TestRobustChannelScaler:
    def test_fit_sets_stats(self, synthetic_X):
        from src.preprocessing.artifacts import RobustChannelScaler
        sc = RobustChannelScaler()
        assert not sc.is_fitted
        sc.fit(synthetic_X)
        assert sc.is_fitted
        assert sc._median.shape == (N_CH,)
        assert sc._iqr.shape == (N_CH,)

    def test_transform_shape_dtype(self, synthetic_X):
        from src.preprocessing.artifacts import RobustChannelScaler
        sc = RobustChannelScaler()
        X_out = sc.fit_transform(synthetic_X)
        assert X_out.shape == synthetic_X.shape
        assert X_out.dtype == np.float32

    def test_transform_before_fit_raises(self, synthetic_X):
        from src.preprocessing.artifacts import RobustChannelScaler
        sc = RobustChannelScaler()
        with pytest.raises(RuntimeError, match="fit"):
            sc.transform(synthetic_X)

    def test_scaled_values_differ(self, synthetic_X):
        from src.preprocessing.artifacts import RobustChannelScaler
        sc = RobustChannelScaler()
        X_out = sc.fit_transform(synthetic_X)
        # After robust scaling the median of each channel should be ~0
        medians = np.median(X_out.transpose(1, 0, 2).reshape(N_CH, -1), axis=1)
        np.testing.assert_allclose(medians, 0.0, atol=1e-3)

    def test_flat_channel_no_nan(self, rng):
        """Zero-IQR channel must not produce NaN (eps guard)."""
        from src.preprocessing.artifacts import RobustChannelScaler
        X = rng.standard_normal((10, 3, 64)).astype(np.float32)
        X[:, 1, :] = 0.0  # completely flat channel
        sc = RobustChannelScaler()
        X_out = sc.fit_transform(X)
        assert not np.any(np.isnan(X_out))


# ---------------------------------------------------------------------------
# reject_by_amplitude
# ---------------------------------------------------------------------------

class TestRejectByAmplitude:
    def test_clean_data_all_kept(self, synthetic_X):
        from src.preprocessing.artifacts import reject_by_amplitude
        X_clean, mask = reject_by_amplitude(synthetic_X, threshold_uv=150.0)
        assert mask.all(), "Clean synthetic data should pass the 150 µV threshold"
        assert X_clean.shape == synthetic_X.shape

    def test_spiked_epochs_rejected(self, dirty_X):
        from src.preprocessing.artifacts import reject_by_amplitude
        _, mask = reject_by_amplitude(dirty_X, threshold_uv=150.0)
        n_rejected = int((~mask).sum())
        assert n_rejected == 5, f"Expected 5 rejections, got {n_rejected}"

    def test_output_shape(self, dirty_X):
        from src.preprocessing.artifacts import reject_by_amplitude
        X_clean, mask = reject_by_amplitude(dirty_X, threshold_uv=150.0)
        assert X_clean.shape == (mask.sum(), N_CH, N_T)

    def test_mask_dtype(self, synthetic_X):
        from src.preprocessing.artifacts import reject_by_amplitude
        _, mask = reject_by_amplitude(synthetic_X, threshold_uv=150.0)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape(self, synthetic_X):
        from src.preprocessing.filters import bandpass_filter
        X_out = bandpass_filter(synthetic_X, sfreq=160.0, l_freq=8.0, h_freq=30.0)
        assert X_out.shape == synthetic_X.shape

    def test_output_dtype(self, synthetic_X):
        from src.preprocessing.filters import bandpass_filter
        X_out = bandpass_filter(synthetic_X, sfreq=160.0, l_freq=8.0, h_freq=30.0)
        assert X_out.dtype == np.float32

    def test_attenuates_dc(self, rng):
        """A pure DC signal should be near-zero after band-pass."""
        X_dc = np.ones((4, 3, 320), dtype=np.float32) * 50.0
        from src.preprocessing.filters import bandpass_filter
        X_out = bandpass_filter(X_dc, sfreq=160.0, l_freq=8.0, h_freq=30.0)
        # Edges may have transient; check the middle quarter
        mid = X_out[:, :, 80:240]
        assert np.abs(mid).max() < 1.0, "DC should be suppressed below 1 µV in the passband"

    def test_preserves_in_band_energy(self, rng):
        """A 15 Hz sine (in-band) should retain substantial energy."""
        t = np.linspace(0, 2.0, 320, endpoint=False)
        sine = np.sin(2 * np.pi * 15 * t).astype(np.float32)
        X_sine = np.tile(sine[np.newaxis, np.newaxis, :], (4, 3, 1))
        from src.preprocessing.filters import bandpass_filter
        X_out = bandpass_filter(X_sine, sfreq=160.0, l_freq=8.0, h_freq=30.0)
        # Energy in the middle (away from edges) should be > 80% of original
        mid_in = X_sine[:, :, 80:240]
        mid_out = X_out[:, :, 80:240]
        ratio = (mid_out ** 2).mean() / (mid_in ** 2).mean()
        assert ratio > 0.8, f"In-band sine lost too much energy: ratio={ratio:.3f}"


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_transform_before_fit_raises(self, synthetic_X):
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg={})
        assert not pp.is_fitted
        with pytest.raises(RuntimeError, match="fit"):
            pp.transform(synthetic_X)

    def test_fit_sets_fitted(self, synthetic_X, cfg_mi8_30):
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg=cfg_mi8_30)
        pp.fit(synthetic_X)
        assert pp.is_fitted

    def test_fit_transform_shape(self, synthetic_X, cfg_mi8_30):
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg=cfg_mi8_30)
        X_out = pp.fit_transform(synthetic_X)
        assert X_out.shape == synthetic_X.shape

    def test_fit_transform_dtype(self, synthetic_X, cfg_mi8_30):
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg=cfg_mi8_30)
        X_out = pp.fit_transform(synthetic_X)
        assert X_out.dtype == np.float32

    def test_transform_changes_values(self, synthetic_X, cfg_mi8_30):
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg=cfg_mi8_30)
        X_out = pp.fit_transform(synthetic_X)
        assert not np.allclose(X_out, synthetic_X), "transform should change the data"

    def test_separate_fit_transform_equiv_fit_transform(self, synthetic_X, cfg_mi8_30):
        """fit() then transform() must equal fit_transform()."""
        from src.preprocessing.pipeline import Preprocessor
        pp1 = Preprocessor(cfg=cfg_mi8_30)
        out1 = pp1.fit(synthetic_X).transform(synthetic_X)
        pp2 = Preprocessor(cfg=cfg_mi8_30)
        out2 = pp2.fit_transform(synthetic_X)
        np.testing.assert_array_equal(out1, out2)

    def test_ea_flag_does_not_raise(self, synthetic_X):
        """euclidean_alignment flag in Preprocessor config is accepted without error.

        EA is handled at the wrapper level (load_epochs); the Preprocessor itself
        is agnostic to it and must not raise when the flag is set.
        """
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg={"euclidean_alignment": True})
        pp.fit(synthetic_X)  # must not raise

    def test_no_scale_cfg(self, synthetic_X):
        """With robust_scale=False the scaler should not be created."""
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg={"robust_scale": False, "band": [8.0, 30.0], "keep_sfreq": 160.0})
        pp.fit(synthetic_X)
        assert pp._scaler is None

    def test_picklable(self, synthetic_X, cfg_mi8_30):
        """Preprocessor must survive a pickle round-trip (checkpoint bundle requirement)."""
        import pickle
        from src.preprocessing.pipeline import Preprocessor
        pp = Preprocessor(cfg=cfg_mi8_30)
        pp.fit(synthetic_X)
        blob = pickle.dumps(pp)
        pp2 = pickle.loads(blob)
        assert pp2.is_fitted
        out1 = pp.transform(synthetic_X)
        out2 = pp2.transform(synthetic_X)
        np.testing.assert_array_equal(out1, out2)
