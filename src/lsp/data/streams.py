"""Enriched representation: the three per-node feature streams.

From normalized canonical coordinates ``(T, N, 3)`` this module builds:
  - landmarks: the coordinates themselves.
  - bones:     ``p_child - p_parent`` along the kinematic tree (root = 0).
  - motion:    temporal difference ``p_t - p_{t-1}`` (first frame = 0).

``stack_streams`` concatenates the requested streams on the feature axis (early fusion):
all three -> ``(T, N, 9)``; landmarks only -> ``(T, N, 3)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .graph import PARENT

VALID_STREAMS = ("landmarks", "bones", "motion")


def build_landmarks(coords: np.ndarray) -> np.ndarray:
    return coords.astype(np.float32)


def build_bones(coords: np.ndarray) -> np.ndarray:
    """(T, N, 3) -> (T, N, 3). For every node, p - p_parent; the root stays at 0."""
    bones = np.zeros_like(coords, dtype=np.float32)
    for i in range(coords.shape[1]):
        p = int(PARENT[i])
        if p >= 0:
            bones[:, i, :] = coords[:, i, :] - coords[:, p, :]
    return bones


def build_motion(coords: np.ndarray) -> np.ndarray:
    """(T, N, 3) -> (T, N, 3). Temporal difference; the first frame is 0."""
    motion = np.zeros_like(coords, dtype=np.float32)
    motion[1:] = coords[1:] - coords[:-1]
    return motion


_BUILDERS = {"landmarks": build_landmarks, "bones": build_bones, "motion": build_motion}


def stack_streams(coords: np.ndarray, streams: Sequence[str] = VALID_STREAMS) -> np.ndarray:
    """Concatenate the requested streams on the feature axis. (T, N, 3) -> (T, N, 3 * len)."""
    parts = []
    for s in streams:
        if s not in _BUILDERS:
            raise ValueError(f"Unknown stream: {s!r}. Valid: {VALID_STREAMS}")
        parts.append(_BUILDERS[s](coords))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def stack_streams_batch(X: np.ndarray, streams: Sequence[str]) -> np.ndarray:
    """(M, T, N, 3) -> (M, T, N, F): ``stack_streams`` applied to every sequence."""
    return np.stack([stack_streams(seq, streams) for seq in X]).astype(np.float32)


def feature_dim(streams: Sequence[str] = VALID_STREAMS) -> int:
    return 3 * len(streams)
