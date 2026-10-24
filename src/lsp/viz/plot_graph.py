"""Figure of the 73-node skeleton graph (nodes coloured by body part, spatial edges in grey).

Uses a clean schematic layout by default, or the mean frame of the first sequence of a bundle.
"""

from __future__ import annotations

import os

import numpy as np

from ..data import canonical as C
from ..data.graph import EDGES

# Local template of one hand (21 MediaPipe points), wrist at the origin, fingers pointing up.
_HAND_TEMPLATE = np.array([
    [0.00, 0.00],  # 0 wrist
    [-0.30, 0.20], [-0.50, 0.40], [-0.65, 0.55], [-0.80, 0.72],  # thumb
    [-0.12, 0.62], [-0.12, 0.92], [-0.12, 1.12], [-0.12, 1.32],  # index
    [0.05, 0.66], [0.05, 1.00], [0.05, 1.25], [0.05, 1.46],  # middle
    [0.22, 0.62], [0.22, 0.92], [0.22, 1.16], [0.22, 1.36],  # ring
    [0.38, 0.52], [0.44, 0.76], [0.47, 0.96], [0.50, 1.12],  # little
], dtype=np.float32)  # fmt: skip

# Local template of the 20 face points (order = canonical.FACE_NAMES).
_FACE_TEMPLATE = np.array([
    [-0.18, 0.30], [-0.12, 0.33], [-0.06, 0.31],  # left eyebrow
    [0.06, 0.31], [0.12, 0.33], [0.18, 0.30],  # right eyebrow
    [-0.20, 0.18], [-0.06, 0.18], [-0.13, 0.22], [-0.13, 0.14],  # left eye
    [0.06, 0.18], [0.20, 0.18], [0.13, 0.22], [0.13, 0.14],  # right eye
    [-0.10, -0.05], [0.10, -0.05], [0.0, -0.02], [0.0, -0.06], [0.0, 0.0], [0.0, -0.09],  # mouth
], dtype=np.float32)  # fmt: skip

_COLORS = {"body": "#1f77b4", "lh": "#d62728", "rh": "#2ca02c", "face": "#9467bd"}


def schematic_coords() -> np.ndarray:
    """(73, 2) schematic layout with the y axis pointing up."""
    pts = np.zeros((C.N, 2), dtype=np.float32)
    pts[C.NOSE] = [0.0, 2.7]
    pts[C.NECK] = [0.0, 2.3]
    pts[C.L_SHOULDER], pts[C.R_SHOULDER] = [-0.6, 2.1], [0.6, 2.1]
    pts[C.L_ELBOW], pts[C.R_ELBOW] = [-0.95, 1.5], [0.95, 1.5]
    pts[C.L_WRIST], pts[C.R_WRIST] = [-1.2, 0.9], [1.2, 0.9]
    pts[C.MID_HIP] = [0.0, 1.2]
    pts[C.L_HIP], pts[C.R_HIP] = [-0.35, 1.15], [0.35, 1.15]

    scale = 0.45
    lh = _HAND_TEMPLATE * scale
    lh[:, 0] = -lh[:, 0]
    pts[C.LH_SLICE] = lh + pts[C.L_WRIST]
    pts[C.RH_SLICE] = _HAND_TEMPLATE * scale + pts[C.R_WRIST]
    pts[C.FACE_SLICE] = _FACE_TEMPLATE + np.array([0.0, 2.95], dtype=np.float32)
    return pts


def _bundle_coords(bundle: str) -> np.ndarray:
    """(73, 2) mean frame of the first bundle sequence, flipped so that y points up."""
    X = np.load(bundle, allow_pickle=True)["X"]  # (M, T, N, 3)
    xy = X[0].mean(axis=0)[:, :2].copy()
    xy[:, 1] = -xy[:, 1]  # image coordinates grow downwards
    return xy


def _node_color(i: int) -> str:
    if i < C.N_BODY:
        return _COLORS["body"]
    if i < C.RH_OFF:
        return _COLORS["lh"]
    if i < C.FACE_OFF:
        return _COLORS["rh"]
    return _COLORS["face"]


def plot(out_path: str, bundle: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = _bundle_coords(bundle) if bundle else schematic_coords()

    fig, ax = plt.subplots(figsize=(7, 8))
    for i, j in EDGES:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#999999", linewidth=1.0,
                zorder=1)  # fmt: skip
    for i in range(C.N):
        ax.scatter(xy[i, 0], xy[i, 1], s=40, color=_node_color(i), zorder=2, edgecolors="white")

    ax.set_title(
        f"Skeleton graph: {C.N} nodes, {len(EDGES)} edges\n"
        "(blue: body, red: left hand, green: right hand, purple: face)"
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_path} ({C.N} nodes, {len(EDGES)} edges)")
