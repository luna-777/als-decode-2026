"""Loss functions. §5.4, §B.1.

weighted_bce: binary cross-entropy with per-class weighting (idle is over-represented).
focal_loss: focal loss (γ=2.0) for the high-imbalance config.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with optional positive-class weight.

    logits  : (B,) or (B, 1) — raw pre-sigmoid scores.
    targets : (B,) float — 0.0 or 1.0.
    pos_weight: scalar tensor on the same device; upweights minority (control) class.
    """
    logits = logits.view(-1)
    targets = targets.float().view(-1)
    if pos_weight is not None and pos_weight.device != logits.device:
        pos_weight = pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss: (1 - p_t)^γ · BCE, down-weighting easy negatives.

    gamma: focusing parameter (default 2.0 per §5.4).
    pos_weight applied before the focal modulation so class imbalance
    and difficulty-weighting are both addressed.
    """
    logits = logits.view(-1)
    targets = targets.float().view(-1)
    if pos_weight is not None and pos_weight.device != logits.device:
        pos_weight = pos_weight.to(logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1.0 - p_t) ** gamma * bce).mean()
