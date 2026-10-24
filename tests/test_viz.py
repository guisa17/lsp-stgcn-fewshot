"""Skeleton rendering used for the project page: no face nodes, undetected hands hidden."""

import numpy as np

from lsp.data import canonical as C
from lsp.viz.render import Caption, _visible_edges, render_frames, save_gif, view_limits


def _take(T=4, seed=0):
    return np.random.default_rng(seed).normal(0, 0.5, size=(T, C.N, 3)).astype(np.float32)


def test_face_nodes_are_never_drawn():
    edges = _visible_edges(_take()[0, :, :2])
    assert edges and all(i < C.FACE_OFF and j < C.FACE_OFF for i, j in edges)


def test_undetected_hand_is_hidden():
    take = _take()
    take[:, C.LH_SLICE, :] = 0.7  # a missing hand collapses to one point after normalization
    edges = _visible_edges(take[0, :, :2])
    left = set(range(C.LH_OFF, C.RH_OFF))
    assert not any(i in left or j in left for i, j in edges)
    assert any(i in range(C.RH_OFF, C.FACE_OFF) for i, _ in edges)


def test_render_and_save_gif(tmp_path):
    take = _take()
    limits = view_limits([take], margin=0.3, head_radius=0.35)
    frames = render_frames(take, Caption("HOLA", "HOLA", "note"), limits=limits, size_px=120,
                           head_radius=0.35)  # fmt: skip
    assert len(frames) == 4 and frames[0].shape == (120, 120, 3)
    path = tmp_path / "clip.gif"
    save_gif(frames, str(path), fps=10)
    assert path.stat().st_size > 0
