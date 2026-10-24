"""Draw the method overview figure (docs/figures/method.png).

Four panels: keypoints as a 73-node skeleton graph, per-node features, the HA-GCN encoder, and
few-shot calibration with class prototypes. The skeleton is the schematic layout of
``lsp.viz.plot_graph`` (no recorded person). The prototype panel is an illustration, not data.

Usage:
    python scripts/make_method_figure.py --out docs/figures/method.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from lsp.data import canonical as C
from lsp.data.graph import EDGES
from lsp.viz.plot_graph import schematic_coords

INK = "#1d2731"
MUTED = "#5b6b7c"
LINE = "#c9d1d9"
TEAL = "#2f7f86"
WARM = "#c98a2b"
VIOLET = "#7b6aa8"
FILL = {"in": "#e3f1f2", "gcn": "#ece8f5", "tcn": "#e6eef8", "out": "#f8ecd9", "note": "#f4f6f8"}
WIDTH_IN, HEIGHT_IN, DPI = 16.0, 6.6, 100  # 1600 x 660 px


def box(ax, x, y, w, h, text, fill, size=10.5, weight="normal", dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=fill,
        edgecolor=MUTED, linewidth=1.1, linestyle=(0, (4, 3)) if dashed else "solid",
    ))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight=weight, linespacing=1.35)  # fmt: skip


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.4))  # fmt: skip


def panel_title(ax, x, y, number, title):
    ax.text(x, y, f"{number}  {title}", fontsize=13, weight="bold", color=INK, ha="left")


def draw_skeleton(ax, cx, cy, scale):
    pts = schematic_coords()
    pts = (pts - pts.mean(axis=0)) * scale + np.array([cx, cy])
    for i, j in EDGES:
        ax.plot(*pts[[i, j]].T, color=LINE, linewidth=1.0, zorder=2)
    colors = np.array([MUTED] * C.N, dtype=object)
    colors[C.LH_SLICE] = WARM
    colors[C.RH_SLICE] = TEAL
    colors[C.FACE_SLICE] = VIOLET
    ax.scatter(pts[:, 0], pts[:, 1], s=9, c=list(colors), zorder=3, linewidths=0)


def draw_prototypes(ax, x0, y0):
    rng = np.random.default_rng(3)
    centres = {"HOLA": (x0 + 0.55, y0 + 1.55), "GRACIAS": (x0 + 2.1, y0 + 1.9),
               "AGUA": (x0 + 1.35, y0 + 0.45)}  # fmt: skip
    colors = {"HOLA": MUTED, "GRACIAS": TEAL, "AGUA": WARM}
    for name, (cx, cy) in centres.items():
        pts = rng.normal(0, 0.16, size=(4, 2)) + (cx, cy)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=colors[name], alpha=0.75, linewidths=0)
        ax.scatter([cx], [cy], marker="*", s=170, color=colors[name], edgecolor="white", zorder=4)
        ax.text(cx, cy + 0.3, name, ha="center", fontsize=9, color=colors[name], weight="bold")
    query = (x0 + 1.75, y0 + 1.25)
    ax.scatter([query[0]], [query[1]], marker="D", s=46, color="white", edgecolor=INK, zorder=5)
    ax.annotate("", xy=centres["GRACIAS"], xytext=query,
                arrowprops={"arrowstyle": "-", "color": TEAL, "linewidth": 1.6})  # fmt: skip
    for name in ("HOLA", "AGUA"):
        ax.annotate("", xy=centres[name], xytext=query,
                    arrowprops={"arrowstyle": "-", "color": LINE, "linewidth": 1.0,
                                "linestyle": (0, (3, 3))})  # fmt: skip
    ax.text(query[0] + 0.12, query[1] - 0.28, "new take", fontsize=8.5, color=INK)


def build(out_path: str) -> None:
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    top = 6.1
    # 1. Keypoints -> skeleton graph
    panel_title(ax, 0.3, top, "1", "Keypoints")
    draw_skeleton(ax, 1.75, 3.55, scale=0.95)
    ax.text(1.75, 1.25, "MediaPipe Holistic, 30 frames\n73-node skeleton graph\n"
            "(11 body, 21 + 21 hands, 20 face)", ha="center", fontsize=9.5, color=MUTED,
            linespacing=1.4)  # fmt: skip

    arrow(ax, 3.3, 3.55, 3.85, 3.55)

    # 2. Per-node features
    panel_title(ax, 3.95, top, "2", "Node features")
    box(ax, 4.0, 4.35, 2.55, 0.75, "landmarks\nposition", FILL["in"], size=10)
    box(ax, 4.0, 3.45, 2.55, 0.75, "bones\nvector to parent node", FILL["in"], size=10)
    box(ax, 4.0, 2.55, 2.55, 0.75, "motion\nchange from last frame", FILL["in"], size=10)
    ax.text(5.275, 1.85, "input tensor\n30 frames x 73 nodes x 9", ha="center", fontsize=9.5,
            color=MUTED, linespacing=1.4)  # fmt: skip

    arrow(ax, 6.7, 3.55, 7.25, 3.55)

    # 3. HA-GCN encoder
    panel_title(ax, 7.35, top, "3", "HA-GCN encoder")
    ax.add_patch(FancyBboxPatch((7.35, 2.35), 3.6, 3.15, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor="none", edgecolor=MUTED, linewidth=1.1,
                                linestyle=(0, (4, 3))))  # fmt: skip
    ax.text(10.85, 5.25, "x 3 blocks\n(32, 64, 64 ch.)\nresidual", ha="right", va="top",
            fontsize=8.5, color=MUTED, linespacing=1.3)  # fmt: skip
    box(ax, 7.55, 4.25, 2.3, 0.8, "spatial graph conv\nA + B + α (M_hand ⊙ C)", FILL["gcn"],
        size=9.5)  # fmt: skip
    box(ax, 7.55, 3.35, 2.3, 0.6, "DropGraph (p = 0.1)", FILL["gcn"], size=9.5)
    box(ax, 7.55, 2.5, 2.3, 0.6, "temporal conv (k = 9)", FILL["tcn"], size=9.5)
    arrow(ax, 8.7, 4.25, 8.7, 3.95)
    arrow(ax, 8.7, 3.35, 8.7, 3.1)
    arrow(ax, 9.15, 2.35, 9.15, 2.0)
    box(ax, 7.75, 1.25, 2.8, 0.7, "global average pooling\n64-d embedding", FILL["out"], size=9.5)
    ax.text(9.15, 0.75, "learnable adjacency on the 42 hand nodes;\nequal to ST-GCN at "
            "initialization", ha="center", fontsize=8.5, color=MUTED, linespacing=1.3)  # fmt: skip

    arrow(ax, 11.1, 3.55, 11.65, 3.55)

    # 4. Few-shot calibration
    panel_title(ax, 11.75, top, "4", "Few-shot calibration")
    ax.text(11.8, 5.45, "K takes per sign from the new signer;\nencoder weights stay frozen",
            fontsize=9.5, color=MUTED, va="top", linespacing=1.4)  # fmt: skip
    draw_prototypes(ax, 12.3, 2.15)
    ax.text(13.6, 1.25, "★ prototype = mean embedding of the K takes\n"
            "prediction = nearest prototype (Euclidean)", ha="center", fontsize=9.5, color=INK,
            linespacing=1.4)  # fmt: skip

    # Training strip
    ax.add_patch(FancyBboxPatch((0.3, 0.1), 15.4, 0.4, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=FILL["note"], edgecolor="none"))  # fmt: skip
    ax.text(8.0, 0.3, "Encoder training (before calibration): cross-entropy pre-training, then "
            "episodic meta-training with Prototypical Networks on LSA64 (signer-disjoint "
            "support and query)", ha="center", va="center", fontsize=9.5, color=INK)  # fmt: skip

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # opaque, white background
    plt.close(fig)
    from PIL import Image

    Image.fromarray(rgb).save(out_path, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.out)
    print(f"Figure saved to {args.out}")


if __name__ == "__main__":
    main()
