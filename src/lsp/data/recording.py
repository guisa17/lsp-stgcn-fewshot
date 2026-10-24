"""Per-frame keypoint extraction for recording, testable without a camera.

Assembles the 95-point raw vector expected by ``canonical.py``::

    pose(33) + left_hand(21) + right_hand(21) + face(20 selected with FACE_IDX)  =  285 values
"""

from __future__ import annotations

import numpy as np

from . import canonical as C

# A take is flagged as low quality when the pose is missing too often or both hands are rarely seen.
LOW_QUALITY_MIN_POSE = 0.8
LOW_QUALITY_MIN_HAND = 0.3


def _landmarks_to_array(landmarks, n_expected: int) -> np.ndarray:
    """MediaPipe landmark list -> (n_expected, 3); zeros when the part was not detected."""
    if landmarks is None:
        return np.zeros((n_expected, 3), dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
    if pts.shape[0] < n_expected:
        pad = np.zeros((n_expected - pts.shape[0], 3), dtype=np.float32)
        pts = np.concatenate([pts, pad], axis=0)
    return pts[:n_expected]


def extract_points(results) -> np.ndarray:
    """MediaPipe Holistic results -> raw vector (285,) = 95 points (x, y, z)."""
    pose = _landmarks_to_array(getattr(results, "pose_landmarks", None), C.RAW_POSE)
    lh = _landmarks_to_array(getattr(results, "left_hand_landmarks", None), C.RAW_HAND)
    rh = _landmarks_to_array(getattr(results, "right_hand_landmarks", None), C.RAW_HAND)

    face_lm = getattr(results, "face_landmarks", None)
    if face_lm is None:
        face = np.zeros((C.RAW_FACE, 3), dtype=np.float32)
    else:
        full = np.array([[p.x, p.y, p.z] for p in face_lm.landmark], dtype=np.float32)
        if full.shape[0] > max(C.FACE_IDX):
            face = full[C.FACE_IDX]
        else:
            face = np.zeros((C.RAW_FACE, 3), np.float32)

    return np.concatenate([pose, lh, rh, face], axis=0).reshape(-1).astype(np.float32)


def detection_rates(seq: np.ndarray) -> dict[str, float]:
    """(T, 285) -> fraction of frames in which each part was detected (not all zeros)."""
    pts = seq.reshape(seq.shape[0], C.RAW_TOTAL, 3)
    blocks = {
        "pose": pts[:, C.RAW_POSE_OFF : C.RAW_POSE_OFF + C.RAW_POSE],
        "lh": pts[:, C.RAW_LH_OFF : C.RAW_LH_OFF + C.RAW_HAND],
        "rh": pts[:, C.RAW_RH_OFF : C.RAW_RH_OFF + C.RAW_HAND],
        "face": pts[:, C.RAW_FACE_OFF : C.RAW_FACE_OFF + C.RAW_FACE],
    }
    return {k: float((np.abs(b).sum(axis=(1, 2)) > 1e-6).mean()) for k, b in blocks.items()}


def is_low_quality(rates: dict[str, float]) -> bool:
    return rates["pose"] < LOW_QUALITY_MIN_POSE or (
        rates["lh"] < LOW_QUALITY_MIN_HAND and rates["rh"] < LOW_QUALITY_MIN_HAND
    )
