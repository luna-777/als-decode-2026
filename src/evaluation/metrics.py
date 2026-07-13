"""Metrics for Stage 1 evaluation. §8.3, §5.6, §11.3.

epoch_metrics: ROC-AUC, balanced accuracy, P/R/F1 over shuffled epochs.
asynchronous_metrics: TPR@FPR, false-activations/min, detection latency — event-stream scoring.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


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
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    result: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }

    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        result["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    return result


def asynchronous_metrics(
    intent_events: list[dict],
    ground_truth: list[dict],
    idle_duration_s: float,
) -> dict[str, float]:
    """Score a self-paced event stream. §5.6, §8.3.

    intent_events: list of dicts with 'timestamp_s'.
    ground_truth:  list of dicts with 'onset_s', 'duration_s', 'label' ('control'|'idle').
    idle_duration_s: total idle recording time (denominator for false-activation rate).

    Returns dict: tpr_at_fpr_1pct, tpr_at_fpr_5pct, false_activations_per_min,
                  mean_latency_s, median_latency_s.
    """
    ctrl_trials = [g for g in ground_truth if g["label"] == "control"]
    n_ctrl = len(ctrl_trials)

    true_positives: list[float] = []   # latencies of detected control trials
    false_activations = 0

    for ev in intent_events:
        t = ev["timestamp_s"]
        matched = False
        for trial in ctrl_trials:
            onset = trial["onset_s"]
            offset = onset + trial["duration_s"]
            if onset <= t <= offset:
                true_positives.append(t - onset)
                matched = True
                break
        if not matched:
            false_activations += 1

    tpr = len(true_positives) / n_ctrl if n_ctrl > 0 else 0.0
    fa_per_min = (false_activations / idle_duration_s * 60.0) if idle_duration_s > 0 else 0.0

    latencies = true_positives if true_positives else [float("nan")]

    return {
        "tpr": tpr,
        "false_activations_per_min": fa_per_min,
        "mean_latency_s": float(np.nanmean(latencies)),
        "median_latency_s": float(np.nanmedian(latencies)),
        "n_detections": len(true_positives),
        "n_false_activations": false_activations,
        "n_ctrl_trials": n_ctrl,
    }


def tpr_at_fixed_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.05,
) -> tuple[float, float]:
    """Return (tpr, threshold) at the operating point closest to target_fpr.

    Uses the ROC curve — appropriate for epoch-level evaluation where each
    window is independently scored. For streaming evaluation use asynchronous_metrics.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # sklearn prepends a sentinel threshold of max(score)+1; clip to valid range
    thresholds = np.clip(thresholds, 0.0, 1.0)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx]), float(thresholds[idx])


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of *values*.

    Returns (lower, upper) for the given CI level.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
