"""Bundle loading and the PyTorch dataset for sign sequences.

A bundle is the ``.npz`` written by ``scripts/prepare_lsp.py`` or ``scripts/prepare_lsa64.py``:
``X: (M, T, N, 3)`` float32, ``y: (M,)`` int64, ``classes: (C,)`` str and ``signers: (M,)`` str.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import Augmentor
from .streams import feature_dim, stack_streams


class SignDataset(Dataset):
    """Yields ``(x, label)`` with ``x`` of shape ``(T, N, F)``; augmentation runs on landmarks."""

    def __init__(
        self,
        X: np.ndarray,
        y: Sequence[int],
        indices: Sequence[int],
        streams: Sequence[str] = ("landmarks", "bones", "motion"),
        augmentor: Augmentor | None = None,
    ):
        self.X = X
        self.y = np.asarray(y)
        self.indices = list(indices)
        self.streams = tuple(streams)
        self.augmentor = augmentor

    @property
    def feature_dim(self) -> int:
        return feature_dim(self.streams)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = self.indices[i]
        coords = self.X[idx]  # (T, N, 3), already normalized
        if self.augmentor is not None:
            coords = self.augmentor(coords)
        x = stack_streams(coords, self.streams)  # (T, N, F)
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return x, int(self.y[idx])


def load_bundle(path: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load a bundle. Returns ``(X, y, classes, signers)``.

    Bundles written before per-sample signer ids existed return ``"unknown"`` for every sample.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bundle not found: {path}. Build it with scripts/prepare_lsp.py or "
            "scripts/prepare_lsa64.py, or pass data.bundle=<path> on the command line."
        )
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    classes = [str(c) for c in data["classes"]]
    signers = data["signers"] if "signers" in data else np.array(["unknown"] * len(y))
    return X, y, classes, signers
