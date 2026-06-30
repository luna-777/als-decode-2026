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
        model_fn: callable,      # (window: np.ndarray) → float probability
        intent_switch: IntentSwitch,
        sfreq: float = 160.0,
        window_len_s: float = 2.0,
        stride_s: float = 0.1,
    ) -> None:
        raise NotImplementedError("Milestone 6")

    def run(
        self,
        X_continuous: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[IntentEvent]:
        """Replay X_continuous sample-by-sample; return list of fired IntentEvents.

        X_continuous: (n_channels, n_samples) — raw continuous recording.
        timestamps: (n_samples,) float — seconds since recording start.
        """
        raise NotImplementedError("Milestone 6")
