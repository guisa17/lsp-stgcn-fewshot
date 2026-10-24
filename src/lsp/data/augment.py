"""Skeleton augmentation (training only), applied to normalized landmarks ``(T, N, 3)``.

Transformations: temporal rescaling (speed change), small 3D rotation, per-node jitter and an
optional left/right mirror. The mirror can change the meaning of asymmetric signs, so it is disabled
in every released config.
"""

from __future__ import annotations

import numpy as np

from . import canonical as C
from .temporal import resample_to_T


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def random_temporal_scale(
    coords: np.ndarray, max_ratio: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Slightly change the speed by stretching or shrinking the window and resampling back to T."""
    rng = _rng(rng)
    T = coords.shape[0]
    ratio = 1.0 + rng.uniform(-max_ratio, max_ratio)
    new_len = max(2, int(round(T * ratio)))
    stretched = resample_to_T(coords, new_len)
    return resample_to_T(stretched, T)


def random_rotation(
    coords: np.ndarray, max_deg: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Small 3D rotation about the origin (the data are already torso-centred)."""
    rng = _rng(rng)
    angles = np.radians(rng.uniform(-max_deg, max_deg, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)
    return coords @ R.T


def random_jitter(
    coords: np.ndarray, sigma: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = _rng(rng)
    return coords + rng.normal(0.0, sigma, size=coords.shape).astype(np.float32)


# Left/right symmetric Face Mesh points (eyebrows, eyes, mouth corners). Midline points (inner and
# outer lips) map to themselves.
_FACE_MIRROR_MP = [
    (70, 300), (105, 334), (107, 336),  # eyebrows
    (33, 263), (133, 362), (159, 386), (145, 374),  # eyes
    (61, 291),  # mouth corners
]  # fmt: skip
_BODY_MIRROR = [
    (C.L_SHOULDER, C.R_SHOULDER),
    (C.L_ELBOW, C.R_ELBOW),
    (C.L_WRIST, C.R_WRIST),
    (C.L_HIP, C.R_HIP),
]
MIRROR_PAIRS = _BODY_MIRROR + [
    (C.FACE_OFF + C.FACE_IDX.index(a), C.FACE_OFF + C.FACE_IDX.index(b)) for a, b in _FACE_MIRROR_MP
]


def mirror_lr(coords: np.ndarray) -> np.ndarray:
    """Horizontal mirror: negate x, swap the two hand blocks and every left/right node pair.

    Nose, neck, mid-hip and the lip midline stay in place. Applying it twice is the identity.
    """
    out = coords.copy()
    out[..., 0] = -out[..., 0]
    lh = out[:, C.LH_SLICE, :].copy()
    out[:, C.LH_SLICE, :] = out[:, C.RH_SLICE, :]
    out[:, C.RH_SLICE, :] = lh
    for a, b in MIRROR_PAIRS:
        out[:, [a, b], :] = out[:, [b, a], :]
    return out


class Augmentor:
    """Configurable composition of augmentations, called once per sample and epoch."""

    def __init__(
        self,
        temporal_scale: bool = True,
        rotation: bool = True,
        jitter: bool = True,
        mirror: bool = False,
        max_ratio: float = 0.15,
        max_deg: float = 5.0,
        sigma: float = 0.01,
        mirror_p: float = 0.5,
        seed: int | None = None,
    ):
        self.temporal_scale = temporal_scale
        self.rotation = rotation
        self.jitter = jitter
        self.mirror = mirror
        self.max_ratio = max_ratio
        self.max_deg = max_deg
        self.sigma = sigma
        self.mirror_p = mirror_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        x = coords.astype(np.float32)
        if self.temporal_scale:
            x = random_temporal_scale(x, self.max_ratio, self.rng)
        if self.rotation:
            x = random_rotation(x, self.max_deg, self.rng)
        if self.mirror and self.rng.random() < self.mirror_p:
            x = mirror_lr(x)
        if self.jitter:
            x = random_jitter(x, self.sigma, self.rng)
        return x
