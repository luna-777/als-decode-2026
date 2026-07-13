"""Milestone 6 — streaming evaluation unit tests. No data download required."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# IntentSwitch
# ---------------------------------------------------------------------------

class TestIntentSwitch:
    def _sw(self, **kw):
        from src.decoding.intent_switch import IntentSwitch
        defaults = dict(tau=0.6, debounce_k=3, refractory_s=2.0, window_stride_s=0.1)
        return IntentSwitch(**{**defaults, **kw})

    def test_fires_after_debounce(self):
        sw = self._sw()
        for i in range(2):
            assert sw.step(0.9, float(i) * 0.1) is None
        ev = sw.step(0.9, 0.2)
        assert ev is not None
        assert ev.confidence == pytest.approx(0.9)

    def test_no_fire_below_tau(self):
        sw = self._sw()
        for i in range(10):
            assert sw.step(0.5, float(i) * 0.1) is None

    def test_refractory_suppresses(self):
        sw = self._sw(refractory_s=5.0)
        # Fire once
        for i in range(3):
            sw.step(0.9, float(i) * 0.1)
        # Immediately after — should be suppressed
        assert sw.step(0.9, 0.3) is None

    def test_fires_again_after_refractory(self):
        sw = self._sw(refractory_s=1.0)
        # First fire at t≈0.2
        for i in range(3):
            sw.step(0.9, float(i) * 0.1)
        # After refractory expires, should fire again within debounce_k steps
        fired = False
        for i in range(4):
            ev = sw.step(0.9, 2.0 + float(i) * 0.1)
            if ev is not None:
                fired = True
                break
        assert fired

    def test_reset_clears_state(self):
        sw = self._sw()
        sw.step(0.9, 0.0)
        sw.step(0.9, 0.1)
        sw.reset()
        # After reset needs full debounce again
        assert sw.step(0.9, 0.2) is None

    def test_calibrate_threshold(self):
        from src.decoding.intent_switch import IntentSwitch
        rng = np.random.default_rng(0)
        idle_probs = rng.uniform(0, 0.5, 1000).tolist()
        tau = IntentSwitch.calibrate_threshold(idle_probs, target_fpr=0.05)
        # At most 5% of idle probs should exceed tau
        fpr = np.mean(np.array(idle_probs) > tau)
        assert fpr <= 0.06  # small tolerance for discrete quantile


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestEpochMetrics:
    def _data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        # Noisy but informative scores
        scores = y * 0.6 + rng.uniform(0, 0.4, n)
        return y, np.clip(scores, 0, 1)

    def test_roc_auc_above_chance(self):
        from src.evaluation.metrics import epoch_metrics
        y, s = self._data()
        m = epoch_metrics(y, s)
        assert m["roc_auc"] > 0.5

    def test_threshold_yields_classification_metrics(self):
        from src.evaluation.metrics import epoch_metrics
        y, s = self._data()
        m = epoch_metrics(y, s, threshold=0.5)
        assert "precision" in m and "recall" in m and "f1" in m

    def test_tpr_at_fixed_fpr(self):
        from src.evaluation.metrics import tpr_at_fixed_fpr
        y, s = self._data()
        tpr, tau = tpr_at_fixed_fpr(y, s, target_fpr=0.05)
        assert 0.0 <= tpr <= 1.0
        assert 0.0 <= tau <= 1.0


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        from src.evaluation.metrics import bootstrap_ci
        rng = np.random.default_rng(1)
        vals = rng.normal(loc=0.7, scale=0.05, size=50)
        lo, hi = bootstrap_ci(vals, n_bootstrap=500)
        assert lo < vals.mean() < hi

    def test_ci_width_positive(self):
        from src.evaluation.metrics import bootstrap_ci
        vals = np.array([0.6, 0.7, 0.8, 0.75, 0.65])
        lo, hi = bootstrap_ci(vals, n_bootstrap=200)
        assert hi > lo


class TestAsynchronousMetrics:
    def _gt(self):
        return [
            {"onset_s": 0.0,  "duration_s": 2.0, "label": "control"},
            {"onset_s": 3.0,  "duration_s": 2.0, "label": "idle"},
            {"onset_s": 6.0,  "duration_s": 2.0, "label": "control"},
            {"onset_s": 9.0,  "duration_s": 2.0, "label": "idle"},
        ]

    def test_perfect_detection(self):
        from src.evaluation.metrics import asynchronous_metrics
        events = [{"timestamp_s": 1.0}, {"timestamp_s": 7.0}]
        m = asynchronous_metrics(events, self._gt(), idle_duration_s=4.0)
        assert m["tpr"] == pytest.approx(1.0)
        assert m["n_false_activations"] == 0

    def test_false_activation_counted(self):
        from src.evaluation.metrics import asynchronous_metrics
        events = [{"timestamp_s": 4.0}]  # inside idle trial
        m = asynchronous_metrics(events, self._gt(), idle_duration_s=4.0)
        assert m["n_false_activations"] == 1
        assert m["tpr"] == pytest.approx(0.0)

    def test_fa_per_min(self):
        from src.evaluation.metrics import asynchronous_metrics
        events = [{"timestamp_s": 4.0}, {"timestamp_s": 10.5}]  # both in idle
        m = asynchronous_metrics(events, self._gt(), idle_duration_s=60.0)
        assert m["false_activations_per_min"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# StreamSimulator
# ---------------------------------------------------------------------------

class TestStreamSimulator:
    def test_produces_events(self):
        from src.decoding.intent_switch import IntentSwitch
        from src.evaluation.streaming import StreamSimulator

        switch = IntentSwitch(tau=0.4, debounce_k=2, refractory_s=1.0)
        # model always returns 0.9
        sim = StreamSimulator(lambda w: 0.9, switch, sfreq=16.0,
                              window_len_s=1.0, stride_s=0.5)
        n_ch, n_samples = 17, 160
        X = np.zeros((n_ch, n_samples), dtype=np.float32)
        ts = np.linspace(0, 10.0, n_samples)
        events = sim.run(X, ts)
        assert len(events) > 0

    def test_silent_model_no_events(self):
        from src.decoding.intent_switch import IntentSwitch
        from src.evaluation.streaming import StreamSimulator

        switch = IntentSwitch(tau=0.9, debounce_k=3, refractory_s=1.0)
        sim = StreamSimulator(lambda w: 0.0, switch, sfreq=16.0,
                              window_len_s=1.0, stride_s=0.5)
        X = np.zeros((17, 160), dtype=np.float32)
        ts = np.linspace(0, 10.0, 160)
        assert sim.run(X, ts) == []

    def test_simulate_from_epochs_runs(self):
        from src.decoding.intent_switch import IntentSwitch
        from src.evaluation.streaming import simulate_from_epochs

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 17, 32)).astype(np.float32)
        y = np.array([1, 0] * 10)
        switch = IntentSwitch(tau=0.4, debounce_k=2, refractory_s=0.5)
        events, gt, idle_s = simulate_from_epochs(
            X, y, lambda w: 0.8, switch, sfreq=16.0
        )
        assert len(gt) == 20
        assert idle_s > 0
