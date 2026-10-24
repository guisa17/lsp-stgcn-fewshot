"""Canonical layout, normalization, streams, temporal resampling and the LSA64 adapter."""

import numpy as np
import pytest

from lsp.data import canonical as C
from lsp.data.augment import MIRROR_PAIRS, mirror_lr
from lsp.data.lsa64 import parse_video_name, process_entry
from lsp.data.normalize import center_and_scale, impute_missing, normalize_sequence
from lsp.data.streams import build_bones, build_motion, feature_dim, stack_streams
from lsp.data.temporal import resample_to_T


def _fake_seq(T=30):
    return np.random.default_rng(0).random((T, C.RAW_FEATURES), dtype=np.float32)


def test_canonical_shapes_and_synthetic_neck():
    arr = _fake_seq()
    coords = C.raw_to_canonical(arr)
    assert coords.shape == (30, C.N, 3) == (30, 73, 3)
    assert (C.N_BODY, C.N_HAND, C.N_FACE) == (11, 21, 20)
    assert C.RAW_FEATURES == 285
    pts = C.raw_to_points(arr)
    expected_neck = (pts[:, C.P_LSHO, :] + pts[:, C.P_RSHO, :]) / 2.0
    assert np.allclose(coords[:, C.NECK, :], expected_neck, atol=1e-5)


def test_canonical_rejects_wrong_width():
    with pytest.raises(ValueError):
        C.raw_to_canonical(np.zeros((30, 225), dtype=np.float32))  # old 53-node raw layout


def test_streams():
    coords = C.raw_to_canonical(_fake_seq())
    assert feature_dim(("landmarks",)) == 3
    assert feature_dim(("landmarks", "bones", "motion")) == 9
    assert stack_streams(coords, ("landmarks", "bones", "motion")).shape == (30, C.N, 9)
    assert np.allclose(build_bones(coords)[:, C.NOSE, :], 0)  # root has a zero bone
    assert np.allclose(build_motion(coords)[0], 0)  # first frame has zero motion
    with pytest.raises(ValueError):
        stack_streams(coords, ("velocity",))


def test_normalize_centers_torso():
    coords = C.raw_to_canonical(_fake_seq())
    out = center_and_scale(coords)
    assert np.allclose(out[:, C.TORSO_IDX, :].mean(axis=1), 0, atol=1e-5)
    assert normalize_sequence(coords).shape == coords.shape


def test_impute_missing_interpolates_gaps():
    coords = np.ones((5, 2, 3), dtype=np.float32)
    coords[:, 0, 0] = [0.0, 1.0, 0.0, 3.0, 4.0]
    coords[2, 0, :] = 0.0  # node 0 missing in frame 2
    out = impute_missing(coords)
    assert np.isclose(out[2, 0, 0], 2.0)


def test_resample_to_T_identity_and_shape():
    coords = np.random.default_rng(0).random((30, 73, 3)).astype(np.float32)
    assert np.array_equal(resample_to_T(coords, 30), coords)
    assert resample_to_T(coords, 45).shape == (45, 73, 3)
    assert resample_to_T(coords, 15).shape == (15, 73, 3)


def test_resample_to_T_linear_ramp():
    ramp = np.arange(10, dtype=np.float32).reshape(10, 1)
    expected = np.linspace(0, 9, 19).astype(np.float32).reshape(19, 1)
    assert np.allclose(resample_to_T(ramp, 19), expected, atol=1e-5)


def test_resample_to_T_edge_cases():
    assert np.array_equal(resample_to_T(np.zeros((0, 3), dtype=np.float32), 30), np.zeros((30, 3)))
    single = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.allclose(resample_to_T(single, 5), np.tile(single, (5, 1)))


def test_mirror_is_an_involution_and_swaps_sides():
    coords = np.random.default_rng(1).random((30, C.N, 3)).astype(np.float32)
    once = mirror_lr(coords)
    assert np.allclose(mirror_lr(once), coords)
    assert np.allclose(once[:, C.LH_SLICE, 1:], coords[:, C.RH_SLICE, 1:])  # hands swapped
    assert np.allclose(once[:, C.L_SHOULDER, 0], -coords[:, C.R_SHOULDER, 0])
    for node in (C.NOSE, C.NECK, C.MID_HIP):  # midline nodes only flip x
        assert np.allclose(once[:, node, 1:], coords[:, node, 1:])
    swapped = {n for pair in MIRROR_PAIRS for n in pair}
    assert len(swapped) == 2 * len(MIRROR_PAIRS)  # every node appears in at most one pair


def test_lsa64_parse_video_name():
    assert parse_video_name("004_003_002.mp4") == ("004", "003", "002")
    assert parse_video_name("064_010_005") == ("064", "010", "005")


def test_lsa64_reorder_frames_right_hand_first():
    # The first LSA64 hand block is the RIGHT hand (verified empirically, see lsa64.py).
    row = np.zeros((5, 1629), dtype=np.float32)
    row[:, 0:99] = 1.0  # pose
    row[:, 99:1503] = 2.0  # full face mesh
    row[:, 1503:1566] = 3.0  # first hand block = right
    row[:, 1566:1629] = 4.0  # second hand block = left
    pts = process_entry(row, T=30).reshape(30, 95, 3)
    assert np.allclose(pts[:, 0:33], 1.0)
    assert np.allclose(pts[:, 33:54], 4.0)  # LSP left-hand slot
    assert np.allclose(pts[:, 54:75], 3.0)  # LSP right-hand slot
    assert np.allclose(pts[:, 75:95], 2.0)  # 20 selected face points
