"""Temporal resampling to a fixed number of frames by linear interpolation.

Used by the speed augmentation during training and by the LSA64 adapter, whose videos have variable
length and must be brought to the fixed ``T`` of the LSP pipeline.
"""

from __future__ import annotations

import numpy as np


def resample_to_T(coords: np.ndarray, T: int) -> np.ndarray:
    """(L, ...) -> (T, ...) by linear interpolation along the time axis (axis 0).

    Each output frame is a weighted mix of its two nearest input frames, so frames are never
    duplicated or skipped. Works for both shrinking (L > T) and stretching (L < T).

    Edge cases: ``L == T`` returns the input unchanged; ``L == 0`` returns zeros; ``L == 1``
    repeats the single frame ``T`` times.
    """
    L = coords.shape[0]
    if L == T:
        return coords
    if L == 0:
        return np.zeros((T, *coords.shape[1:]), dtype=np.float32)

    dst_t = np.linspace(0, L - 1, T)
    idx_lo = np.floor(dst_t).astype(np.int64)
    idx_hi = np.minimum(idx_lo + 1, L - 1)
    frac = (dst_t - idx_lo).astype(np.float32).reshape((T,) + (1,) * (coords.ndim - 1))

    lo = coords[idx_lo].astype(np.float32)
    hi = coords[idx_hi].astype(np.float32)
    return lo * (1.0 - frac) + hi * frac
