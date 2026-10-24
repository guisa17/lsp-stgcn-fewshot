"""Saving and loading model checkpoints."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn as nn

from ..models.build import build_model


def save_encoder(
    path: str,
    model: nn.Module,
    *,
    model_cfg: dict,
    streams: Sequence[str],
    classes: Sequence[str],
    test_signers: Sequence[str],
    seed: int,
    val_signers: Sequence[str] | None = None,
) -> None:
    """Save a meta-trained encoder together with what is needed to rebuild it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_type": model_cfg["type"],
            "model_cfg": dict(model_cfg),
            "streams": list(streams),
            "state_dict": model.state_dict(),
            "lsa64_classes": list(classes),
            "test_signers": list(test_signers),
            "val_signers": list(val_signers) if val_signers is not None else None,
            "seed": seed,
        },
        path,
    )


def load_encoder(path: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load an encoder checkpoint in eval mode. Returns ``(model, checkpoint_dict)``.

    Checkpoints written by the original code store only ``model_type``; the model is then
    built with the constructor defaults, which are the values those encoders were trained with.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {path}. See scripts/download_weights.py."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg") or {"type": ckpt["model_type"]}
    model = build_model(
        model_cfg, num_classes=len(ckpt["lsa64_classes"]), streams=tuple(ckpt["streams"])
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_classifier(
    path: str, model_cfg: dict, num_classes: int, streams: Sequence[str]
) -> nn.Module:
    """Load the ``state_dict`` saved by ``scripts/train.py`` into a freshly built model."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model(model_cfg, num_classes=num_classes, streams=streams)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
