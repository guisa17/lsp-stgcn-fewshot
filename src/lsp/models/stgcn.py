"""ST-GCN encoder (Yan et al., 2018) in plain PyTorch.

Deliberately small (~0.1M parameters) to avoid overfitting a few hundred sequences. Input shape is
``(B, T, V, F)``; the model permutes it to ``(B, F, T, V)`` internally.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .layers import STGCNBlock, apply_input_bn, block_strides


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        A: np.ndarray,
        channels: tuple[int, ...] = (32, 64, 64),
        t_kernel: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        K, V, _ = A.shape
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.V = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.blocks = nn.ModuleList(
            STGCNBlock(
                c_in, c_out, K, t_kernel=t_kernel, stride=stride, dropout=dropout, residual=residual
            )
            for c_in, c_out, stride, residual in block_strides(in_channels, tuple(channels))
        )
        self.fc = nn.Linear(channels[-1], num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.fc.in_features

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V, F) -> (B, C) embedding after global average pooling, before ``fc``.

        Used by few-shot calibration, which classifies by distance between embeddings.
        """
        x = apply_input_bn(self.data_bn, x)
        for blk in self.blocks:
            x = blk(x, self.A)
        return x.mean(dim=(2, 3))  # global average pooling over T and V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
