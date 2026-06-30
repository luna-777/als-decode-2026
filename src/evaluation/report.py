"""Per-subject tables, distribution plots, and bootstrap CI reporting. §11.3–11.4."""
from __future__ import annotations

import numpy as np


def subject_metric_table(
    per_subject_metrics: list[dict],
    ci_level: float = 0.95,
) -> dict:
    """Aggregate per-subject metric dicts into a summary table with bootstrap CIs.

    per_subject_metrics: list of dicts, one per test subject (output of epoch_metrics /
        asynchronous_metrics).
    Returns: dict mapping metric_name → {mean, std, ci_lower, ci_upper, per_subject}.
    """
    raise NotImplementedError("Milestone 6")


def print_stage1_report(summary: dict) -> None:
    """Pretty-print the Stage 1 evaluation summary to stdout."""
    raise NotImplementedError("Milestone 6")
