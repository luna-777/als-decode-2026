"""Loss functions. §5.4, §B.1.

weighted_bce: binary cross-entropy with per-class weighting (idle is over-represented).
focal_loss: focal loss (γ=2.0) for the high-imbalance config.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with optional positive-class weight.

    logits: (batch,) or (batch, 1) — raw (pre-sigmoid) scores.
    targets: (batch,) float — 0.0 or 1.0.
    pos_weight: scalar tensor; upweights the minority (control) class.
    """
    raise NotImplementedError("Milestone 5")


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss: (1 - p_t)^gamma * BCE, down-weights easy negatives.

    gamma: focusing parameter (default 2.0 per §5.4).
    """
    raise NotImplementedError("Milestone 5")
