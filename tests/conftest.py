"""Shared synthetic data for the tests. No test needs real LSP or LSA64 data."""

from __future__ import annotations

import os

import numpy as np
import pytest

from lsp.data import canonical as C

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def synthetic_embeddings_dataset(n_classes=8, n_signers=4, reps=6, T=10, V=6, F=3, seed=0):
    """Sequences with well-separated per-class centres, labelled by class and signer."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(n_classes, V, F)).astype(np.float32)
    X, y, signers = [], [], []
    for c in range(n_classes):
        for s in range(n_signers):
            for _ in range(reps):
                X.append(centers[c][None] + rng.normal(0, 0.1, size=(T, V, F)).astype(np.float32))
                y.append(c)
                signers.append(f"signer_{s}")
    return np.stack(X), np.array(y), np.array(signers)


def tiny_adjacency(V: int) -> np.ndarray:
    return np.stack([np.eye(V, dtype=np.float32)] * 3)


def synthetic_raw_take(label_idx: int, rng: np.random.Generator, T: int = 30) -> np.ndarray:
    """A plausible raw take (T, 285): fixed body, hands moving along a class-specific path."""
    pts = np.full((T, C.RAW_TOTAL, 3), 0.5, dtype=np.float32)
    pose = pts[:, C.RAW_POSE_OFF : C.RAW_POSE_OFF + C.RAW_POSE]
    pose[:, C.P_LSHO], pose[:, C.P_RSHO] = [0.62, 0.32, 0.0], [0.38, 0.32, 0.0]
    pose[:, C.P_LHIP], pose[:, C.P_RHIP] = [0.57, 0.62, 0.0], [0.43, 0.62, 0.0]
    t = np.linspace(0, 1, T, dtype=np.float32)[:, None]
    direction = np.array([np.cos(label_idx), np.sin(label_idx), 0.0], dtype=np.float32) * 0.2
    for off in (C.RAW_LH_OFF, C.RAW_RH_OFF):
        pts[:, off : off + C.RAW_HAND] = 0.5 + t[:, :, None] * direction
    pts += rng.normal(0, 0.005, size=pts.shape).astype(np.float32)
    return pts.reshape(T, C.RAW_FEATURES)


@pytest.fixture
def raw_dataset(tmp_path):
    """Two signer folders with 4 classes x 8 takes of synthetic raw keypoints."""
    rng = np.random.default_rng(0)
    dirs = []
    for signer in ("signer_a", "signer_b"):
        d = tmp_path / "raw" / signer
        d.mkdir(parents=True)
        for c, label in enumerate(["ALPHA", "BRAVO", "CHARLIE", "DELTA"]):
            for i in range(8):
                np.save(d / f"{label}_{1000 + i}.npy", synthetic_raw_take(c, rng))
        dirs.append(str(d))
    return dirs
