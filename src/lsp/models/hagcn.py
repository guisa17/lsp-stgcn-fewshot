"""HA-GCN (Hand-Aware GCN): ST-GCN with a learnable, hand-focused adjacency and DropGraph.

Per-partition adaptive adjacency::

    A_eff = A_base + B + alpha * (M_hand * C)

  - ``A_base``: fixed skeletal adjacency (spatial partition, K=3).
  - ``B``: learnable global refinement (initialized at 0).
  - ``C``: learnable term masked to the hands (``M_hand`` is 1 only on the 42x42 hand block),
    weighted by a learnable scalar ``alpha`` (initialized at 0). This is the hand-aware part.

DropGraph drops whole nodes during training. With ``B = 0`` and ``alpha = 0`` at initialization,
``A_eff == A_base``, so HA-GCN starts out identical to the ST-GCN baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data import canonical as C
from .layers import apply_input_bn, block_strides, residual_branch, temporal_conv


def hand_mask() -> np.ndarray:
    """(N, N) mask with ones on the hand-node block (indices 11..52) and zeros elsewhere."""
    M = np.zeros((C.N, C.N), dtype=np.float32)
    idx = np.arange(C.LH_OFF, C.RH_OFF + C.N_HAND)  # 11..52 (42 nodes)
    M[np.ix_(idx, idx)] = 1.0
    return M


class HandAwareGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A_base: np.ndarray, M_hand: np.ndarray):
        super().__init__()
        K, V, _ = A_base.shape
        self.K = K
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)
        self.register_buffer("A_base", torch.tensor(A_base, dtype=torch.float32))
        self.register_buffer("M_hand", torch.tensor(M_hand, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(K, V, V))
        self.C = nn.Parameter(torch.zeros(K, V, V))
        self.alpha = nn.Parameter(torch.zeros(1))

    def effective_adjacency(self) -> torch.Tensor:
        return self.A_base + self.B + self.alpha * (self.M_hand.unsqueeze(0) * self.C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.effective_adjacency()
        x = self.conv(x)
        B, CK, T, V = x.shape
        x = x.view(B, self.K, CK // self.K, T, V)
        x = torch.einsum("bkctv,kvw->bctw", x, A)
        return x.contiguous()


class DropGraph(nn.Module):
    """Structural dropout: drops whole nodes during training and rescales the rest."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T, V)
        if not self.training or self.p <= 0:
            return x
        V = x.shape[-1]
        keep = (torch.rand(x.shape[0], 1, 1, V, device=x.device) > self.p).float()
        return x * keep / (1.0 - self.p)


class HAGCNBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        A_base: np.ndarray,
        M_hand: np.ndarray,
        t_kernel: int,
        stride: int,
        dropout: float,
        drop_graph: float,
        residual: bool = True,
    ):
        super().__init__()
        self.gcn = HandAwareGraphConv(in_c, out_c, A_base, M_hand)
        self.gcn_bn = nn.BatchNorm2d(out_c)
        self.dropg = DropGraph(drop_graph)
        self.tcn = temporal_conv(out_c, t_kernel, stride, dropout)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual_branch(in_c, out_c, stride, residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.gcn_bn(self.gcn(x)))
        x = self.dropg(x)
        x = self.tcn(x)
        return self.relu(x + res)


class HAGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        A: np.ndarray,
        M_hand: np.ndarray,
        channels: tuple[int, ...] = (32, 64, 64),
        t_kernel: int = 9,
        dropout: float = 0.3,
        drop_graph: float = 0.1,
    ):
        super().__init__()
        K, V, _ = A.shape
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.blocks = nn.ModuleList(
            HAGCNBlock(
                c_in,
                c_out,
                A,
                M_hand,
                t_kernel=t_kernel,
                stride=stride,
                dropout=dropout,
                drop_graph=drop_graph,
                residual=residual,
            )
            for c_in, c_out, stride, residual in block_strides(in_channels, tuple(channels))
        )
        self.fc = nn.Linear(channels[-1], num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.fc.in_features

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V, F) -> (B, C) embedding after global average pooling, before ``fc``."""
        x = apply_input_bn(self.data_bn, x)
        for blk in self.blocks:
            x = blk(x)
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))
