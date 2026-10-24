"""Canonical skeleton layout: the single source of truth for node indices.

The canonical skeleton has 73 nodes::

    11 upper body  +  21 left hand  +  21 right hand  +  20 face  =  73 nodes

Each raw frame written by the recorder (``scripts/record_keypoints.py``) contains, in order::

    pose(33) + left_hand(21) + right_hand(21) + face(20 selected Face Mesh points)  =  95 points

i.e. 285 values (x, y, z). The 73 canonical nodes are derived from those 95 points: 11 body nodes
are selected or derived from the pose block (including a synthetic neck and mid-hip); the 42 hand
points and the 20 face points are used as they are.
"""

from __future__ import annotations

import numpy as np

# --- Raw layout (what the recorder stores per frame) ---
RAW_POSE = 33
RAW_HAND = 21
RAW_FACE = 20  # already selected at capture time with FACE_IDX
RAW_TOTAL = RAW_POSE + 2 * RAW_HAND + RAW_FACE  # 95
RAW_FEATURES = RAW_TOTAL * 3  # 285

# Offsets inside the 95-point raw vector
RAW_POSE_OFF = 0
RAW_LH_OFF = RAW_POSE  # 33
RAW_RH_OFF = RAW_POSE + RAW_HAND  # 54
RAW_FACE_OFF = RAW_POSE + 2 * RAW_HAND  # 75

# --- MediaPipe Pose (BlazePose, 33 points) indices used for the body ---
P_NOSE = 0
P_LSHO, P_RSHO = 11, 12
P_LELB, P_RELB = 13, 14
P_LWRI, P_RWRI = 15, 16
P_LHIP, P_RHIP = 23, 24

# --- MediaPipe Face Mesh (468 points) indices for the 20 face nodes ---
# Eyebrows (6), eyes (8), mouth (6). The recorder uses this list to select points at capture time.
FACE_IDX = [
    70,
    105,
    107,  # left eyebrow
    336,
    334,
    300,  # right eyebrow
    33,
    133,
    159,
    145,  # left eye: outer, inner, top, bottom
    362,
    263,
    386,
    374,  # right eye: inner, outer, top, bottom
    61,
    291,
    13,
    14,
    0,
    17,  # mouth: left/right corner, inner upper/lower lip, outer upper/lower lip
]
assert len(FACE_IDX) == RAW_FACE
FACE_NAMES = [
    "Lbrow_a", "Lbrow_b", "Lbrow_c",
    "Rbrow_a", "Rbrow_b", "Rbrow_c",
    "Leye_outer", "Leye_inner", "Leye_top", "Leye_bot",
    "Reye_inner", "Reye_outer", "Reye_top", "Reye_bot",
    "mouth_L", "mouth_R", "mouth_up_in", "mouth_lo_in", "mouth_up_out", "mouth_lo_out",
]  # fmt: skip

# --- Canonical counts and ranges (73 nodes) ---
N_BODY = 11
N_HAND = RAW_HAND  # 21
N_FACE = RAW_FACE  # 20
N = N_BODY + 2 * N_HAND + N_FACE  # 73

BODY_SLICE = slice(0, N_BODY)  # 0..10
LH_SLICE = slice(N_BODY, N_BODY + N_HAND)  # 11..31
RH_SLICE = slice(N_BODY + N_HAND, N_BODY + 2 * N_HAND)  # 32..52
FACE_SLICE = slice(N_BODY + 2 * N_HAND, N)  # 53..72

LH_OFF = N_BODY  # 11
RH_OFF = N_BODY + N_HAND  # 32
FACE_OFF = N_BODY + 2 * N_HAND  # 53

# --- Canonical body indices (order of the BODY block) ---
NOSE = 0
NECK = 1  # synthetic: midpoint of the shoulders
L_SHOULDER, R_SHOULDER = 2, 3
L_ELBOW, R_ELBOW = 4, 5
L_WRIST, R_WRIST = 6, 7
L_HIP, R_HIP = 8, 9
MID_HIP = 10  # synthetic: midpoint of the hips

LH_WRIST = LH_OFF + 0  # 11
RH_WRIST = RH_OFF + 0  # 32

# Nodes used for the torso centre and the biometric scale (normalization)
TORSO_IDX = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


def raw_to_points(arr: np.ndarray) -> np.ndarray:
    """(..., 285) -> (..., 95, 3)."""
    if arr.shape[-1] != RAW_FEATURES:
        raise ValueError(f"Expected {RAW_FEATURES} features (95 points), got {arr.shape[-1]}")
    return arr.reshape(*arr.shape[:-1], RAW_TOTAL, 3)


def select_canonical(pts95: np.ndarray) -> np.ndarray:
    """(..., 95, 3) -> (..., 73, 3): derive body (11) + hands (42) + face (20)."""
    if pts95.shape[-2] != RAW_TOTAL:
        raise ValueError(f"Expected {RAW_TOTAL} points, got {pts95.shape[-2]}")
    pose = pts95[..., RAW_POSE_OFF : RAW_POSE_OFF + RAW_POSE, :]
    lh = pts95[..., RAW_LH_OFF : RAW_LH_OFF + RAW_HAND, :]
    rh = pts95[..., RAW_RH_OFF : RAW_RH_OFF + RAW_HAND, :]
    face = pts95[..., RAW_FACE_OFF : RAW_FACE_OFF + RAW_FACE, :]

    nose = pose[..., P_NOSE, :]
    lsho, rsho = pose[..., P_LSHO, :], pose[..., P_RSHO, :]
    lelb, relb = pose[..., P_LELB, :], pose[..., P_RELB, :]
    lwri, rwri = pose[..., P_LWRI, :], pose[..., P_RWRI, :]
    lhip, rhip = pose[..., P_LHIP, :], pose[..., P_RHIP, :]
    neck = (lsho + rsho) / 2.0
    mid_hip = (lhip + rhip) / 2.0

    body = np.stack([nose, neck, lsho, rsho, lelb, relb, lwri, rwri, lhip, rhip, mid_hip], axis=-2)
    return np.concatenate([body, lh, rh, face], axis=-2)


def raw_to_canonical(arr: np.ndarray) -> np.ndarray:
    """Shortcut: (..., 285) -> (..., 73, 3)."""
    return select_canonical(raw_to_points(arr))
