"""Skeleton graph over the 73-node layout: kinematic tree, edges and partitioned adjacency.

On top of the canonical layout (``canonical.py``) this module defines:
  - ``PARENT``: parent of every node in the kinematic tree (global root = nose). Used for the bone
    stream and for the centripetal/centrifugal spatial partition.
  - ``EDGES``: undirected spatial edges (body + hands + face skeleton), including
      * pose-wrist <-> hand-wrist bridges (connect each hand to the body),
      * face -> body bridges (inner left eye and inner upper lip attach to the nose),
    so the resulting graph is connected (a tree with 73 nodes and 72 edges).
  - ``build_adjacency``: partitioned adjacency ``(3, 73, 73)`` (spatial partition of Yan et al.,
    2018: self / centripetal / centrifugal), row-normalized by degree.
"""

from __future__ import annotations

import numpy as np

from . import canonical as C

NUM_PARTITIONS = 3

# --- Hand tree (local MediaPipe Hands indices 0..20) ---
# 0 = wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 little finger.
HAND_PARENT_LOCAL = [
    -1,  # 0 wrist (hand root -> body wrist)
    0, 1, 2, 3,  # thumb
    0, 5, 6, 7,  # index
    5, 9, 10, 11,  # middle
    9, 13, 14, 15,  # ring
    13, 17, 18, 19,  # little
]  # fmt: skip

# --- Body tree (canonical body indices 0..10) ---
# 0 nose (global root), 1 neck, 2/3 shoulders, 4/5 elbows, 6/7 wrists, 8/9 hips, 10 mid-hip.
BODY_PARENT = [-1, 0, 1, 1, 2, 3, 4, 5, 10, 10, 1]

# --- Face tree (local indices 0..19; -1 = bridge to the body nose) ---
# Order = canonical.FACE_NAMES.
FACE_PARENT_LOCAL = [
    1,  # 0 Lbrow_a -> Lbrow_b
    8,  # 1 Lbrow_b -> Leye_top
    1,  # 2 Lbrow_c -> Lbrow_b
    4,  # 3 Rbrow_a -> Rbrow_b
    12,  # 4 Rbrow_b -> Reye_top
    4,  # 5 Rbrow_c -> Rbrow_b
    7,  # 6 Leye_outer -> Leye_inner
    -1,  # 7 Leye_inner -> (bridge) nose
    7,  # 8 Leye_top -> Leye_inner
    7,  # 9 Leye_bot -> Leye_inner
    7,  # 10 Reye_inner -> Leye_inner (crosses the nose bridge)
    10,  # 11 Reye_outer -> Reye_inner
    10,  # 12 Reye_top -> Reye_inner
    10,  # 13 Reye_bot -> Reye_inner
    16,  # 14 mouth_L -> mouth_up_in
    16,  # 15 mouth_R -> mouth_up_in
    -1,  # 16 mouth_up_in -> (bridge) nose
    16,  # 17 mouth_lo_in -> mouth_up_in
    16,  # 18 mouth_up_out -> mouth_up_in
    17,  # 19 mouth_lo_out -> mouth_lo_in
]


def build_parent() -> np.ndarray:
    """Length-73 ``PARENT`` array (canonical parent of each node; -1 only for the nose root)."""
    parent = np.full(C.N, -1, dtype=np.int64)
    for i, p in enumerate(BODY_PARENT):
        parent[i] = p
    # Each hand wrist hangs from the body wrist (bridge).
    for loc, p in enumerate(HAND_PARENT_LOCAL):
        parent[C.LH_OFF + loc] = C.L_WRIST if p == -1 else C.LH_OFF + p
        parent[C.RH_OFF + loc] = C.R_WRIST if p == -1 else C.RH_OFF + p
    # Face (two bridges to the nose).
    for loc, p in enumerate(FACE_PARENT_LOCAL):
        parent[C.FACE_OFF + loc] = C.NOSE if p == -1 else C.FACE_OFF + p
    return parent


PARENT = build_parent()


def build_edges() -> list[tuple[int, int]]:
    """Undirected spatial edges = (child, parent) for every node that has a parent."""
    return [(i, int(PARENT[i])) for i in range(C.N) if PARENT[i] >= 0]


EDGES = build_edges()


def _normalize_adjacency(a: np.ndarray) -> np.ndarray:
    """Row-degree normalization D^{-1} A (empty rows are left at zero)."""
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return a / deg


def build_adjacency() -> np.ndarray:
    """Partitioned adjacency ``(3, N, N)`` using the spatial partition of Yan et al.

    - k=0: self-loops (identity).
    - k=1: centripetal, each node aggregates from its PARENT (closer to the root).
    - k=2: centrifugal, each node aggregates from its CHILDREN (farther from the root).
    """
    a_self = np.eye(C.N, dtype=np.float32)
    a_cp = np.zeros((C.N, C.N), dtype=np.float32)  # i <- parent(i)
    a_cf = np.zeros((C.N, C.N), dtype=np.float32)  # i <- children(i)
    for i in range(C.N):
        p = int(PARENT[i])
        if p >= 0:
            a_cp[i, p] = 1.0
            a_cf[p, i] = 1.0
    A = np.stack([a_self, _normalize_adjacency(a_cp), _normalize_adjacency(a_cf)], axis=0)
    return A.astype(np.float32)
