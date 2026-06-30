"""BackboneEncoder, Adapter, DecoderHead, EEGDecoder. §8.3 — the §2.2 invariant.

Single source of truth for the encoder across all three stages.
Uses Braindecode's EEGNet reference implementation; do not re-implement.
Stage flag controls adapter insertion (off for Stage 1, on for Stage 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BackboneEncoder(nn.Module):
    """Shared EEGNet-based feature extractor.

    Braindecode's EEGNetv4 body (temporal + depthwise spatial + separable convolutions).
    Accepts insert_adapters=True for Stage 3 adaptation (bottleneck adapters after each block).
    freeze() locks all backbone parameters during Stage 3 PEFT.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        pool_size_1: int = 4,
        pool_size_2: int = 8,
        dropout: float = 0.25,
        max_norm_spatial: float = 1.0,
        insert_adapters: bool = False,
        adapter_reduction: int = 8,
    ) -> None:
        raise NotImplementedError("Milestone 4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_channels, n_times) → features: (batch, feature_dim)"""
        raise NotImplementedError("Milestone 4")

    def freeze(self) -> None:
        """Freeze all backbone parameters in-place. Called by Stage 3 adaptation."""
        raise NotImplementedError("Milestone 4")

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector produced by forward()."""
        raise NotImplementedError("Milestone 4")


class Adapter(nn.Module):
    """Bottleneck adapter: Linear(d→d//r) → activation → Linear(d//r→d) + residual.

    Inserted after convolutional blocks for Stage 3 parameter-efficient fine-tuning.
    reduction controls the bottleneck size; only adapters + head train on patient data.
    """

    def __init__(self, d_in: int, reduction: int = 8) -> None:
        raise NotImplementedError("Milestone 4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Milestone 4")


class DecoderHead(nn.Module):
    """Linear classification head applied to BackboneEncoder features.

    Max-norm constraint on weights (§A.4 max_norm_dense = 0.25).
    """

    def __init__(self, feature_dim: int, n_classes: int, max_norm: float = 0.25) -> None:
        raise NotImplementedError("Milestone 4")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, feature_dim) → logits: (batch, n_classes)"""
        raise NotImplementedError("Milestone 4")


class EEGDecoder(nn.Module):
    """BackboneEncoder + DecoderHead.

    stage='stage1': adapters off, all parameters trainable.
    stage='stage3_adapted': adapters on, backbone frozen, only adapters+head train.
    """

    def __init__(
        self,
        backbone: BackboneEncoder,
        head: DecoderHead,
        stage: str = "stage1",
    ) -> None:
        raise NotImplementedError("Milestone 4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Milestone 4")
