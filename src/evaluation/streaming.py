"""StreamSimulator — replays a recording sample-by-sample for event-stream evaluation. §8.3.

ADR-7: the only valid measure of a self-paced switch is event-stream scoring,
not shuffled-epoch accuracy. This class implements the §5.7 harness.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.decoding.intent_switch import IntentEvent, IntentSwitch


class StreamSimulator:
    """Replays an EEG recording through a ring buffer and the IntentSwitch.

    Protocol (§5.7):
    1. Feed samples one-at-a-time into a ring buffer of window_len_s.
    2. Every stride_s, extract the current window and score it with the model.
    3. Pass the probability to IntentSwitch.step(); collect emitted IntentEvents.
    4. Return the full event stream for scoring by asynchronous_metrics().
    """

    def __init__(
        self,
        model_fn: callable,
        intent_switch: IntentSwitch,
        sfreq: float = 160.0,
        window_len_s: float = 2.0,
        stride_s: float = 0.1,
    ) -> None:
        self.model_fn = model_fn
        self.intent_switch = intent_switch
        self.sfreq = sfreq
        self.window_samples = int(round(window_len_s * sfreq))
        self.stride_samples = max(1, int(round(stride_s * sfreq)))

    def run(
        self,
        X_continuous: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[IntentEvent]:
        """Replay X_continuous; return list of fired IntentEvents.

        X_continuous: (n_channels, n_samples)
        timestamps:   (n_samples,) float seconds since recording start.
        """
        n_ch, n_samples = X_continuous.shape
        self.intent_switch.reset()
        events: list[IntentEvent] = []

        # Start scoring once we have a full window
        start = self.window_samples
        for sample_idx in range(start, n_samples, self.stride_samples):
            window = X_continuous[:, sample_idx - self.window_samples: sample_idx]
            prob = float(self.model_fn(window))
            t = float(timestamps[sample_idx - 1])
            ev = self.intent_switch.step(prob, t)
            if ev is not None:
                events.append(ev)

        return events


def simulate_from_epochs(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: callable,
    intent_switch: IntentSwitch,
    sfreq: float = 160.0,
    gap_s: float = 0.5,
) -> tuple[list[IntentEvent], list[dict], float]:
    """Convenience wrapper: concatenate epochs into a pseudo-continuous stream.

    Concatenates epochs end-to-end with a `gap_s` silence between them,
    assigns ground-truth trial boundaries, then runs StreamSimulator.

    Returns (events, ground_truth, idle_duration_s) ready for asynchronous_metrics().
    """
    n_epochs, n_ch, n_times = X.shape
    gap_samples = int(round(gap_s * sfreq))
    gap_buf = np.zeros((n_ch, gap_samples), dtype=np.float32)

    segments: list[np.ndarray] = []
    ground_truth: list[dict] = []
    t_cursor = 0.0
    epoch_s = n_times / sfreq
    idle_duration_s = 0.0

    for i, (epoch, label) in enumerate(zip(X, y)):
        onset = t_cursor
        label_str = "control" if int(label) == 1 else "idle"
        ground_truth.append({
            "onset_s": onset,
            "duration_s": epoch_s,
            "label": label_str,
        })
        if label_str == "idle":
            idle_duration_s += epoch_s
        segments.append(epoch)
        t_cursor += epoch_s
        if i < n_epochs - 1:
            segments.append(gap_buf)
            t_cursor += gap_s

    X_cont = np.concatenate(segments, axis=1)  # (n_ch, total_samples)
    total_samples = X_cont.shape[1]
    timestamps = np.linspace(0.0, t_cursor, total_samples, endpoint=False)

    sim = StreamSimulator(model_fn, intent_switch, sfreq=sfreq)
    events = sim.run(X_cont, timestamps)

    return events, ground_truth, idle_duration_s
