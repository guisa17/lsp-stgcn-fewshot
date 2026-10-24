"""ST-GCN layers in plain PyTorch (no PyTorch Geometric).

The graph topology is fixed, so the spatial convolution reduces to a 1x1 channel convolution
followed by a dense matmul with the partitioned adjacency, and the temporal convolution is a Conv2d
over the time axis (Yan et al., 2018). This is faster on CPU than sparse message passing.

Tensor convention: ``x`` has shape ``(B, C, T, V)`` (batch, channels, time, nodes) and ``A`` has
shape ``(K, V, V)`` (K spatial partitions).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution: aggregate per partition and mix channels."""

    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, C_out*K, T, V)
        B, CK, T, V = x.shape
        x = x.view(B, self.K, CK // self.K, T, V)  # (B, K, C_out, T, V)
        x = torch.einsum("bkctv,kvw->bctw", x, A)  # aggregate neighbours per partition
        return x.contiguous()


def temporal_conv(channels: int, t_kernel: int, stride: int, dropout: float) -> nn.Sequential:
    """Temporal convolution over the time axis, followed by BatchNorm and Dropout."""
    pad = ((t_kernel - 1) // 2, 0)
    return nn.Sequential(
        nn.Conv2d(channels, channels, (t_kernel, 1), (stride, 1), pad),
        nn.BatchNorm2d(channels),
        nn.Dropout(dropout),
    )


def residual_branch(in_channels: int, out_channels: int, stride: int, residual: bool):
    """Identity, 1x1 projection, or no residual connection (returns 0)."""
    if not residual:
        return lambda x: 0
    if in_channels == out_channels and stride == 1:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
        nn.BatchNorm2d(out_channels),
    )


class STGCNBlock(nn.Module):
    """ST-GCN block: spatial graph convolution + temporal convolution with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        t_kernel: int,
        stride: int,
        dropout: float,
        residual: bool = True,
    ):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, K)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.tcn = temporal_conv(out_channels, t_kernel, stride, dropout)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual_branch(in_channels, out_channels, stride, residual)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.gcn_bn(self.gcn(x, A)))
        x = self.tcn(x)
        return self.relu(x + res)


def apply_input_bn(data_bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """BatchNorm over the input features flattened per node; maps (B, T, V, F) -> (B, F, T, V).

    A function rather than a module so the parameter names in saved checkpoints stay ``data_bn.*``.
    """
    x = x.permute(0, 3, 1, 2).contiguous()  # (B, F, T, V)
    B, Cf, T, V = x.shape
    x = x.permute(0, 2, 1, 3).reshape(B, T, Cf * V)
    x = data_bn(x.transpose(1, 2)).transpose(1, 2)
    return x.reshape(B, T, Cf, V).permute(0, 2, 1, 3).contiguous()


def block_strides(in_channels: int, channels: tuple[int, ...]) -> list[tuple[int, int, int, bool]]:
    """(in, out, stride, residual) per block: stride 2 when the width changes after block 0,
    and no residual on the first block."""
    plan = []
    c_prev = in_channels
    for i, c in enumerate(channels):
        stride = 2 if (i > 0 and c != c_prev) else 1
        plan.append((c_prev, c, stride, i != 0))
        c_prev = c
    return plan
