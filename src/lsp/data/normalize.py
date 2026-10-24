"""Per-sample normalization of canonical sequences ``(T, N, 3)``.

Order: impute missing nodes -> temporal smoothing -> torso centring + biometric scaling.
Normalization is per sample (no global statistics), so there is no train->test leakage and
inference needs no stored state.

A "missing" node is one that MediaPipe did not detect and was stored as (0, 0, 0).
"""

from __future__ import annotations

import numpy as np

from . import canonical as C

_EPS = 1e-6


def detect_mask(coords: np.ndarray) -> np.ndarray:
    """(T, N, 3) -> (T, N) bool. True means the node is present (not ~zero)."""
    return np.abs(coords).sum(axis=-1) > _EPS


def impute_missing(coords: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Linearly interpolate missing nodes over time, independently for each node.

    A node that is missing in every frame is left at zero.
    """
    coords = coords.astype(np.float32).copy()
    if mask is None:
        mask = detect_mask(coords)
    T, Nn, _ = coords.shape
    t_idx = np.arange(T)
    for n in range(Nn):
        present = mask[:, n]
        if present.all() or not present.any():
            continue
        valid_t = t_idx[present]
        for c in range(3):
            coords[:, n, c] = np.interp(t_idx, valid_t, coords[present, n, c])
    return coords


def smooth_moving_average(coords: np.ndarray, w: int = 3) -> np.ndarray:
    """Centred temporal moving average (window ``w``) per node and coordinate."""
    if w <= 1:
        return coords
    T = coords.shape[0]
    pad = w // 2
    padded = np.pad(coords, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / w
    out = np.empty_like(coords)
    for n in range(coords.shape[1]):
        for c in range(3):
            out[:, n, c] = np.convolve(padded[:, n, c], kernel, mode="valid")[:T]
    return out


def center_and_scale(coords: np.ndarray) -> np.ndarray:
    """Centre on the per-frame torso centroid; scale by the median shoulder-to-shoulder distance."""
    coords = coords.astype(np.float32).copy()
    torso = coords[:, C.TORSO_IDX, :]  # (T, 4, 3)
    centroid = torso.mean(axis=1, keepdims=True)  # (T, 1, 3)
    coords = coords - centroid

    ls = coords[:, C.L_SHOULDER, :]
    rs = coords[:, C.R_SHOULDER, :]
    shoulder_dist = np.linalg.norm(ls - rs, axis=-1)  # (T,)
    scale = float(np.median(shoulder_dist))
    if scale < _EPS:
        scale = 1.0
    return (coords / scale).astype(np.float32)


def normalize_sequence(coords: np.ndarray, smooth_w: int = 3) -> np.ndarray:
    """Full pipeline: impute -> smooth -> centre/scale. (T, N, 3) -> (T, N, 3)."""
    coords = impute_missing(coords)
    coords = smooth_moving_average(coords, w=smooth_w)
    coords = center_and_scale(coords)
    return coords
