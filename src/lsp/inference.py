"""Load a trained run directory and classify raw recorded takes.

Mirrors the preparation pipeline exactly at inference time:
raw (T, 285) -> canonical (T, 73, 3) -> normalized -> streams -> tensor (1, T, 73, F) -> model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

from .data import canonical as C
from .data.dataset import load_bundle
from .data.normalize import normalize_sequence
from .data.streams import stack_streams
from .training.checkpoint import load_classifier


@dataclass
class LoadedRun:
    name: str
    model: torch.nn.Module
    classes: list[str]
    streams: tuple[str, ...]
    smooth_w: int
    cfg: dict
    metrics: dict = field(default_factory=dict)

    def predict_raw(self, raw_seq: np.ndarray) -> dict:
        """``raw_seq``: (T, 285) as written by the recorder.

        Returns ``{"pred": class, "confidence": float, "ranked": [(class, prob), ...]}``.
        """
        if raw_seq.ndim != 2 or raw_seq.shape[1] != C.RAW_FEATURES:
            raise ValueError(f"Expected (T, {C.RAW_FEATURES}), got {raw_seq.shape}")
        coords = C.raw_to_canonical(raw_seq.astype(np.float32))
        coords = normalize_sequence(coords, smooth_w=self.smooth_w)
        x = stack_streams(coords, self.streams)
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(xt), dim=1)[0].numpy()

        order = np.argsort(-probs)
        return {
            "pred": self.classes[int(order[0])],
            "confidence": float(probs[order[0]]),
            "ranked": [(self.classes[int(i)], float(probs[i])) for i in order],
        }


def list_runs(runs_dir: str) -> list[str]:
    """Sub-directories of ``runs_dir`` that contain ``model.pt`` and ``config.yaml``."""
    if not os.path.isdir(runs_dir):
        return []
    return [
        name
        for name in sorted(os.listdir(runs_dir))
        if os.path.isfile(os.path.join(runs_dir, name, "model.pt"))
        and os.path.isfile(os.path.join(runs_dir, name, "config.yaml"))
    ]


def load_run(runs_dir: str, name: str) -> LoadedRun:
    """Load weights, config and class names of ``runs_dir/<name>/`` (from scripts/train.py)."""
    d = os.path.join(runs_dir, name)
    with open(os.path.join(d, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics = {}
    metrics_path = os.path.join(d, "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    classes = metrics.get("classes")
    if classes is None:  # runs from the original code: class names live in the bundle
        classes = load_bundle(cfg["data"]["bundle"])[2]

    streams = tuple(cfg["data"]["streams"])
    model = load_classifier(os.path.join(d, "model.pt"), cfg["model"], len(classes), streams)
    return LoadedRun(
        name=name,
        model=model,
        classes=classes,
        streams=streams,
        smooth_w=int(cfg["data"].get("smooth_window", 3)),
        cfg=cfg,
        metrics=metrics,
    )
