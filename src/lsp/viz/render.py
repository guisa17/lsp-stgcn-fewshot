"""Skeleton-only renderings of normalized takes (body and hands; the face is a plain head outline).

Frames are drawn with Matplotlib from canonical coordinates ``(T, 73, 3)`` and written as GIF files
with Pillow, so no video codec is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data import canonical as C
from ..data.graph import EDGES

_BODY_COLOR = "#5b6b7c"
_LEFT_COLOR = "#c98a2b"  # signer's left hand
_RIGHT_COLOR = "#2f7f86"  # signer's right hand
_TEXT_COLOR = "#1d2731"
_OK_COLOR = "#2e7d4f"
_ERR_COLOR = "#b3402e"
_LINE = 0.3  # caption line height, in shoulder-width units
_CAPTION_SPACE = 1.2  # vertical space reserved below the skeleton for three caption lines

# Edges between body and hand nodes only; face nodes are never drawn.
_SKELETON_EDGES = [(i, j) for i, j in EDGES if i < C.FACE_OFF and j < C.FACE_OFF]


@dataclass
class Caption:
    """Text drawn on a clip: the true gloss and, optionally, the prediction."""

    true_label: str
    pred_label: str | None = None
    note: str | None = None


_LEFT_NODES = set(range(C.LH_OFF, C.RH_OFF))
_RIGHT_NODES = set(range(C.RH_OFF, C.FACE_OFF))
_MIN_HAND_SPREAD = 1e-3  # an undetected hand collapses to a single point after normalization


def _visible_edges(xy: np.ndarray) -> list[tuple[int, int]]:
    """Skeleton edges of one frame, without hands that were not detected."""
    hidden = set()
    for nodes in (_LEFT_NODES, _RIGHT_NODES):
        idx = sorted(nodes)
        if np.ptp(xy[idx], axis=0).max() < _MIN_HAND_SPREAD:
            hidden |= nodes
    return [(i, j) for i, j in _SKELETON_EDGES if i not in hidden and j not in hidden]


def _edge_color(i: int, j: int) -> str:
    nodes = {i, j}
    if nodes <= set(range(C.LH_OFF, C.RH_OFF)) | {C.L_WRIST}:
        return _LEFT_COLOR
    if nodes <= set(range(C.RH_OFF, C.FACE_OFF)) | {C.R_WRIST}:
        return _RIGHT_COLOR
    return _BODY_COLOR


def view_limits(takes: Sequence[np.ndarray], margin: float, head_radius: float) -> tuple:
    """Shared axis limits (x0, x1, y0, y1) covering the body and detected hands of all ``takes``."""
    pts = np.concatenate([
        t[f, sorted({n for e in _visible_edges(t[f, :, :2]) for n in e}), :2]
        for t in takes for f in range(t.shape[0])
    ])  # fmt: skip
    x0, y0 = pts.min(axis=0) - margin
    x1, y1 = pts.max(axis=0) + margin
    y0 = min(y0, float(np.min([t[:, C.NOSE, 1].min() for t in takes])) - head_radius - margin)
    return float(x0), float(x1), float(y0), float(y1)


def render_frames(
    coords: np.ndarray,
    caption: Caption,
    *,
    limits: tuple,
    size_px: int,
    head_radius: float,
) -> list[np.ndarray]:
    """Draw every frame of one take ``(T, 73, 3)``; returns RGB uint8 arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = limits
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    frames = []
    for t in range(coords.shape[0]):
        ax.clear()
        ax.set_xlim(x0, x1)
        ax.set_ylim(-y1 - _CAPTION_SPACE, -y0)  # image y grows downwards; room for the caption
        ax.set_aspect("equal")
        ax.axis("off")
        xy = coords[t, :, :2] * np.array([1.0, -1.0])
        for i, j in _visible_edges(xy):
            ax.plot(*xy[[i, j]].T, color=_edge_color(i, j), linewidth=2.2, solid_capstyle="round")
        ax.add_patch(
            plt.Circle(xy[C.NOSE], head_radius, fill=False, color=_BODY_COLOR, linewidth=2.2)
        )
        _draw_caption(ax, caption, x0, -y1 - 0.25)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


def _draw_caption(ax, caption: Caption, x: float, y: float) -> None:
    """Up to three lines, top to bottom from ``y``: true sign, prediction, note."""
    ax.text(x + 0.1, y, f"sign: {caption.true_label}", fontsize=11, color=_TEXT_COLOR, va="top")
    if caption.pred_label is not None:
        ok = caption.pred_label == caption.true_label
        ax.text(x + 0.1, y - _LINE, f"predicted: {caption.pred_label}", fontsize=11,
                weight="bold", color=_OK_COLOR if ok else _ERR_COLOR, va="top")  # fmt: skip
    if caption.note:
        ax.text(x + 0.1, y - 2 * _LINE, caption.note, fontsize=8, color=_BODY_COLOR, va="top")


def save_gif(frames: Sequence[np.ndarray], path: str, fps: int) -> None:
    """Write frames as a looping GIF with a shared adaptive palette."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    palette = images[0].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:], duration=int(1000 / fps), loop=0,
        optimize=True,
    )  # fmt: skip
