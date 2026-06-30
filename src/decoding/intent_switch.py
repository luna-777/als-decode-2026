"""Stage 1 decision layer — threshold + debounce + refractory period. §5.5.

Converts per-window probabilities into a single `intent` event (timestamp, confidence).
All parameters exposed in configs/evaluation/streaming.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentEvent:
    """Emitted by IntentSwitch when a communication-intent is detected. §2.4 contract."""

    timestamp_s: float
    confidence: float   # p_t at the triggering window


class IntentSwitch:
    """Stateful decision layer for the self-paced brain switch.

    Algorithm (§5.5):
    1. Threshold: p_t > τ (τ set on validation to a target idle FPR).
    2. Debounce: require k consecutive windows above τ before firing.
    3. Refractory: after firing, suppress for R seconds.
    """

    def __init__(
        self,
        tau: float = 0.5,
        debounce_k: int = 3,
        refractory_s: float = 5.0,
        window_stride_s: float = 0.1,
    ) -> None:
        raise NotImplementedError("Milestone 6")

    def reset(self) -> None:
        """Reset internal state (consecutive-window counter, refractory timer)."""
        raise NotImplementedError("Milestone 6")

    def step(
        self, prob: float, timestamp_s: float
    ) -> IntentEvent | None:
        """Feed one window probability; return IntentEvent if the switch fires, else None."""
        raise NotImplementedError("Milestone 6")

    @classmethod
    def calibrate_threshold(
        cls,
        probs_idle: list[float],
        target_fpr: float = 0.01,
    ) -> float:
        """Find τ such that FPR on idle windows ≈ target_fpr. Used on validation data."""
        raise NotImplementedError("Milestone 6")
