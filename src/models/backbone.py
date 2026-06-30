"""BackboneEncoder, Adapter, DecoderHead, EEGDecoder. §8.3 — the §2.2 invariant.

Single source of truth for the encoder across all three stages.
Uses Braindecode's EEGNetv4 reference implementation; do not re-implement.
Stage flag controls adapter insertion (off for Stage 1, on for Stage 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BackboneEncoder(nn.Module):
    """Shared EEGNet-based feature extractor.

    Wraps Braindecode's EEGNetv4 body (temporal + depthwise spatial + separable
    convolutions + pooling) with the classifier head replaced by a Flatten layer,
    producing a flat feature vector of shape (batch, feature_dim).

    insert_adapters=True is reserved for Stage 3 PEFT and raises in Stage 1.
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
        adapter_reduction: int = 8,  # noqa: ARG002 — reserved for Stage 3
    ) -> None:
        super().__init__()
        if insert_adapters:
            raise NotImplementedError(
                "Bottleneck adapters are a Stage 3 feature. "
                "Set insert_adapters=False for Stage 1."
            )

        try:
            from braindecode.models import EEGNet as _EEGNetCls
        except ImportError:  # <1.12
            from braindecode.models import EEGNetv4 as _EEGNetCls  # type: ignore[no-redef]

        _net = _EEGNetCls(
            n_chans=n_channels,
            n_outputs=2,  # placeholder; head is replaced below
            n_times=n_times,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            pool1_kernel_size=pool_size_1,
            pool2_kernel_size=pool_size_2,
            drop_prob=dropout,
            conv_spatial_max_norm=int(max_norm_spatial),
        )
        # Replace the Braindecode classifier with a plain Flatten.
        # The conv body ends with shape (B, F2, 1, W); flatten → (B, F2*W).
        _net.final_layer = nn.Flatten(start_dim=1)
        self.net = _net

        # Probe the actual feature dimension with a dry-run (no grad).
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_times)
            self._feature_dim: int = int(self.net(dummy).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → features: (B, feature_dim)"""
        return self.net(x)

    def freeze(self) -> None:
        """Freeze all backbone parameters in-place. Called by Stage 3 adaptation."""
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector produced by forward()."""
        return self._feature_dim


class Adapter(nn.Module):
    """Bottleneck adapter: Linear(d→d//r) → GELU → Linear(d//r→d) + residual.

    Inserted after convolutional blocks for Stage 3 parameter-efficient fine-tuning.
    Stage 1 does not use adapters — this class is an extension point only.
    """

    def __init__(self, d_in: int, reduction: int = 8) -> None:
        raise NotImplementedError(
            "Bottleneck adapters are used in Stage 3 only. "
            "Implement when building the Stage 3 PEFT path."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Stage 3 only")


class DecoderHead(nn.Module):
    """Linear classification head with max-norm weight constraint.

    Max-norm on the weight rows prevents the head from dominating the gradient
    signal early in training (§A.4 max_norm_dense = 0.25).
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        max_norm: float = 0.25,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, n_classes)
        self.max_norm = max_norm

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, feature_dim) → logits: (B, n_classes)"""
        self._apply_max_norm()
        return self.linear(features)

    def _apply_max_norm(self) -> None:
        with torch.no_grad():
            w = self.linear.weight
            norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (norms / self.max_norm).clamp(min=1.0)
            w.div_(scale)


class EEGDecoder(nn.Module):
    """BackboneEncoder + DecoderHead — the full Stage 1 classifier.

    stage='stage1': all parameters trainable.
    stage='stage3_adapted': backbone frozen, only adapters+head train (Stage 3).
    """

    def __init__(
        self,
        backbone: BackboneEncoder,
        head: DecoderHead,
        stage: str = "stage1",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.stage = stage
        if stage == "stage3_adapted":
            backbone.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
