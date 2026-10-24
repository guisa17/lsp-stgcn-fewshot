"""Adapter from LSA64 (Argentinian Sign Language, 10 signers) to the canonical LSP pipeline.

Source: the Kaggle dataset "SimpleLSA64 Mediapipe Holistic landmarks", already processed with
MediaPipe Holistic (``refine_face_landmarks=True``). Each ``<split>_landmarks.pkl`` is
``{video_name: [[1629 values] per frame]}`` with ``video_name`` of the form ``NNN_SSS_RRR.mp4``
(class, signer, repetition), and 1629 = pose(33) + face(468) + hands(21+21) points in (x, y, z),
in that block order.

This module reorders the 1629 values per frame into the same 95-point raw layout used by the
recorder (pose(33) + left_hand(21) + right_hand(21) + face(20 selected with
``canonical.FACE_IDX``)), resamples each sequence to ``T`` frames by linear interpolation (LSA64
videos have variable length) and then applies exactly the same code as LSP
(``canonical.raw_to_canonical`` + ``normalize.normalize_sequence``).

Hand order was verified empirically because the Kaggle export does not document it: inside the
42-point hand block the RIGHT hand comes first, then the LEFT hand. This is the reverse of the
MediaPipe convention used by the recorder. It was checked by measuring, on a smoke-test subset, the
distance between each hand-block wrist and the matching pose wrist (MediaPipe Pose #15/#16): with
the right-then-left order the same-side wrist is consistently closer than the opposite one (388
comparisons, mean 0.66 vs 0.76 in normalized units); with the other order the opposite holds.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import canonical as C
from .normalize import normalize_sequence
from .temporal import resample_to_T

HAND_ORDER_ASSUMPTION = "right_hand_then_left_hand"  # verified, see module docstring

# --- LSA64 raw layout (1629 values per frame: pose -> full face -> hands) ---
_POSE_N, _FACE_N, _HAND_N = 33, 468, 21
_POSE_OFF = 0
_FACE_OFF = _POSE_N * 3  # 99
_HANDS_OFF = _FACE_OFF + _FACE_N * 3  # 1503
_TOTAL_COLS = _HANDS_OFF + 2 * _HAND_N * 3  # 1629


def parse_video_name(video_name: str) -> tuple[str, str, str]:
    """``"NNN_SSS_RRR.mp4"`` -> ``(class_id, signer_id, repetition)`` as zero-padded strings."""
    stem = os.path.splitext(video_name)[0]
    class_id, signer_id, rep = stem.split("_")
    return class_id, signer_id, rep


def _reorder_frames(raw1629: np.ndarray) -> np.ndarray:
    """``(T, 1629)`` [pose|face468|hands42] -> ``(T, 285)`` LSP raw layout [pose|lh|rh|face20]."""
    T = raw1629.shape[0]
    pose = raw1629[:, _POSE_OFF:_FACE_OFF].reshape(T, _POSE_N, 3)
    face_full = raw1629[:, _FACE_OFF:_HANDS_OFF].reshape(T, _FACE_N, 3)
    hands = raw1629[:, _HANDS_OFF:_TOTAL_COLS].reshape(T, 2 * _HAND_N, 3)
    rh, lh = hands[:, :_HAND_N], hands[:, _HAND_N:]  # verified order: right first
    face_sel = face_full[:, C.FACE_IDX, :]  # (T, 20, 3)

    raw95 = np.concatenate([pose, lh, rh, face_sel], axis=1)  # (T, 95, 3)
    return raw95.reshape(T, -1).astype(np.float32)  # (T, 285)


def process_entry(raw_frames, T: int) -> np.ndarray:
    """One pickle entry ``(L, 1629)`` (variable L) -> ``(T, 285)`` in the LSP raw layout."""
    arr = np.asarray(raw_frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != _TOTAL_COLS:
        raise ValueError(f"Expected (L, {_TOTAL_COLS}), got {arr.shape}")
    resampled = resample_to_T(arr, T)  # (T, 1629)
    return _reorder_frames(resampled)  # (T, 285)


def load_split_raw(lsa64_dir: str, split: str) -> tuple[pd.DataFrame, dict]:
    """Load ``<split>_landmarks.pkl`` and its label CSV. ``split`` is "train" or "test"."""
    csv_candidates = [
        os.path.join(lsa64_dir, f"LSA64_videos_{split}.csv"),
        os.path.join(lsa64_dir, f"{split}_labels.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.isfile(p)), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"No label CSV for '{split}' in {lsa64_dir} (tried {csv_candidates})"
        )

    labels = pd.read_csv(csv_path)
    parsed = labels["video_name"].apply(parse_video_name)
    labels["class_id"] = [p[0] for p in parsed]
    labels["signer_id"] = [p[1] for p in parsed]
    labels["rep"] = [p[2] for p in parsed]

    pkl_path = os.path.join(lsa64_dir, f"{split}_landmarks.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Landmark pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        landmarks = pickle.load(f)

    return labels, landmarks


def build_bundle(
    lsa64_dir: str,
    splits: Iterable[str] = ("train", "test"),
    *,
    T: int,
    smooth: int,
    signers: Iterable[str] | None = None,
    classes: Iterable[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Process LSA64 (or a subset) into the same bundle format as the LSP preparation script.

    ``signers`` / ``classes`` optionally restrict processing to some ``signer_id`` / ``class_id``
    values (strings such as "003"), which is used for a quick smoke test.
    """
    signers = set(signers) if signers is not None else None
    classes = set(classes) if classes is not None else None

    rows = []  # (class_id, signer_id, raw_frames)
    class_names = {}
    for split in splits:
        labels, landmarks = load_split_raw(lsa64_dir, split)
        for _, row in labels.iterrows():
            cid, sid = row["class_id"], row["signer_id"]
            if signers is not None and sid not in signers:
                continue
            if classes is not None and cid not in classes:
                continue
            video_name = row["video_name"]
            if video_name not in landmarks:
                if verbose:
                    print(f"  ! no landmarks for {video_name}, skipped")
                continue
            class_names[cid] = row["class_name"]
            rows.append((cid, sid, landmarks[video_name]))

    if not rows:
        raise ValueError("No record matches the given signers/classes filters")

    class_ids = sorted({cid for cid, _, _ in rows})
    class_to_idx = {c: i for i, c in enumerate(class_ids)}

    X, y, signer_arr = [], [], []
    for cid, sid, raw_frames in rows:
        raw285 = process_entry(raw_frames, T=T)
        coords = C.raw_to_canonical(raw285)
        coords = normalize_sequence(coords, smooth_w=smooth)
        X.append(coords.astype(np.float32))
        y.append(class_to_idx[cid])
        signer_arr.append(f"lsa64_{sid}")

    X = np.stack(X, axis=0)
    y = np.asarray(y, dtype=np.int64)
    classes_out = [class_names.get(c, c) for c in class_ids]

    if verbose:
        print(
            f"LSA64 bundle: samples={len(y)} classes={len(classes_out)} "
            f"signers={len(set(signer_arr))} X={X.shape}"
        )

    return {
        "X": X,
        "y": y,
        "classes": classes_out,
        "signers": np.array(signer_arr),
        "class_ids": class_ids,
        "class_names": class_names,
    }


def save_bundle(bundle: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        X=bundle["X"],
        y=bundle["y"],
        classes=np.array(bundle["classes"]),
        signers=bundle["signers"],
        class_ids=np.array(bundle["class_ids"]),
    )
