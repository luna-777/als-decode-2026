"""Metrics for Stage 1 evaluation. §8.3, §5.6, §11.3.

epoch_metrics: ROC-AUC, balanced accuracy, P/R/F1 over shuffled epochs.
asynchronous_metrics: TPR@FPR, false-activations/min, detection latency — event-stream scoring.
"""
from __future__ import annotations

import numpy as np


def epoch_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute epoch-level classification metrics.

    y_true: (n,) int — ground truth labels (0 = idle, 1 = control).
    y_score: (n,) float — model output probabilities.
    threshold: if given, compute precision/recall/F1 at this operating point.
    Returns dict: roc_auc, balanced_acc, precision, recall, f1.
    """
    raise NotImplementedError("Milestone 6")


def asynchronous_metrics(
    intent_events: list[dict],
    ground_truth: list[dict],
    idle_duration_s: float,
) -> dict[str, float]:
    """Score a self-paced event stream. §5.6, §8.3.

    intent_events: list of dicts with 'timestamp_s' (IntentSwitch output).
    ground_truth:  list of dicts with 'onset_s', 'duration_s', 'label' ('control'|'idle').
    idle_duration_s: total idle recording time (denominator for false-activation rate).

    Returns dict: tpr_at_fpr_1pct, tpr_at_fpr_5pct, false_activations_per_min,
                  mean_latency_s, median_latency_s.
    """
    raise NotImplementedError("Milestone 6")


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of *values*.

    Returns (lower, upper) for the given CI level.
    """
    raise NotImplementedError("Milestone 6")
