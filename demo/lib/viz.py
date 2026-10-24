"""Figures for the demo: canonical skeleton of a recorded frame, detection rates and top-k bars."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from lsp.data import canonical as C
from lsp.data import graph as G

_PART_COLOR = {"body": "#7C93AC", "lh": "#C9A24B", "rh": "#7FAE9B", "face": "#C77B72"}
_EDGE_COLOR = "#8892a0"
_GOOD, _WARN, _BAD = "#7FAE9B", "#C9A24B", "#C77B72"


def _node_colors() -> list[str]:
    colors = [_PART_COLOR["body"]] * C.N
    for i in range(C.LH_OFF, C.LH_OFF + C.N_HAND):
        colors[i] = _PART_COLOR["lh"]
    for i in range(C.RH_OFF, C.RH_OFF + C.N_HAND):
        colors[i] = _PART_COLOR["rh"]
    for i in range(C.FACE_OFF, C.FACE_OFF + C.N_FACE):
        colors[i] = _PART_COLOR["face"]
    return colors


_NODE_COLORS = _node_colors()


def skeleton_figure(seq_raw: np.ndarray, frame_idx: int, *, figsize=(4.2, 4.2)):
    """``seq_raw``: (T, 285) raw take. Draws the canonical skeleton of frame ``frame_idx``."""
    pts73 = C.raw_to_canonical(seq_raw[frame_idx])  # (73, 3)
    x, y = pts73[:, 0], -pts73[:, 1]  # flip y (image coordinates grow downwards)
    present = ~np.all(pts73 == 0, axis=-1)

    fig, ax = plt.subplots(figsize=figsize)
    for child, parent in G.EDGES:
        if present[child] and present[parent]:  # undetected nodes sit at (0, 0)
            ax.plot(
                [x[child], x[parent]],
                [y[child], y[parent]],
                color=_EDGE_COLOR,
                linewidth=1.0,
                zorder=1,
            )
    ax.scatter(
        x[present],
        y[present],
        c=[_NODE_COLORS[i] for i in range(C.N) if present[i]],
        s=18,
        zorder=2,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def detection_bar_figure(rates: dict, *, figsize=(4.0, 1.6)):
    """Horizontal bars with the detection rate of each part (pose / hands / face)."""
    labels = ["body", "left hand", "right hand", "face"]
    values = [rates.get(k, 0.0) for k in ("pose", "lh", "rh", "face")]
    colors = [_GOOD if v >= 0.8 else (_WARN if v >= 0.3 else _BAD) for v in values]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(min(v + 0.02, 0.85), i, f"{v:.0%}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


def topk_bar_figure(
    ranked: list[tuple[str, float]],
    *,
    true_label: str | None = None,
    top_k: int = 5,
    figsize=(4.4, 2.2),
):
    """Horizontal bars with the ``top_k`` most probable classes of a prediction."""
    top = ranked[:top_k]
    labels = [str(c) for c, _ in top]
    values = [p for _, p in top]
    colors = [
        _GOOD if true_label is not None and c == str(true_label) else _PART_COLOR["body"]
        for c in labels
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(min(v + 0.02, 0.85), i, f"{v:.0%}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig
