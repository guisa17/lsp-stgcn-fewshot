"""Read-only inventory of raw takes (``<raw_dir>/<signer_id>/<CLASS>_<timestamp>.npy``), plus a
reversible delete that moves a take to a trash folder."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass

import numpy as np

from lsp.data import canonical as C
from lsp.data.recording import detection_rates, is_low_quality

TRASH_DIRNAME = "_trash"
_FNAME_RE = re.compile(r"^(?P<label>.+)_(?P<ts>\d+)\.npy$")


@dataclass(frozen=True)
class Sample:
    path: str
    signer_id: str
    label: str
    ts_ms: int

    @property
    def fname(self) -> str:
        return os.path.basename(self.path)


def list_signers(raw_dir: str) -> list[str]:
    if not os.path.isdir(raw_dir):
        return []
    return sorted(
        d
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith("_")
    )


def list_samples(raw_dir: str, signer_id: str) -> list[Sample]:
    signer_dir = os.path.join(raw_dir, signer_id)
    if not os.path.isdir(signer_dir):
        return []
    out = []
    for fname in os.listdir(signer_dir):
        m = _FNAME_RE.match(fname)
        if m:
            out.append(Sample(os.path.join(signer_dir, fname), signer_id, m["label"], int(m["ts"])))
    return sorted(out, key=lambda s: (s.label, s.ts_ms))


def class_counts(raw_dir: str, signer_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in list_samples(raw_dir, signer_id):
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts


def full_inventory(raw_dir: str) -> dict[str, dict[str, int]]:
    """``{signer_id: {class: n_takes}}`` for every signer under ``raw_dir``."""
    return {signer: class_counts(raw_dir, signer) for signer in list_signers(raw_dir)}


def load_sample(sample: Sample) -> np.ndarray:
    return np.load(sample.path)


def sample_quality(sample: Sample) -> dict:
    """Expected shape plus per-part detection rates of a recorded take."""
    seq = load_sample(sample)
    ok_shape = seq.ndim == 2 and seq.shape[1] == C.RAW_FEATURES
    rates = detection_rates(seq) if ok_shape else {"pose": 0.0, "lh": 0.0, "rh": 0.0, "face": 0.0}
    return {
        "shape": seq.shape,
        "ok_shape": ok_shape,
        "rates": rates,
        "low_quality": ok_shape and is_low_quality(rates),
    }


def move_to_trash(sample: Sample, raw_dir: str) -> str:
    """Move a take to ``<raw_dir>/<signer>/_trash/`` instead of deleting it."""
    trash_dir = os.path.join(raw_dir, sample.signer_id, TRASH_DIRNAME)
    os.makedirs(trash_dir, exist_ok=True)
    dest = os.path.join(trash_dir, sample.fname)
    shutil.move(sample.path, dest)
    return dest
