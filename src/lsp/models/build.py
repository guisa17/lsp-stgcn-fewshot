"""Model factory: build an encoder from the ``model`` section of a config."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from ..data import canonical as C
from ..data import graph as G
from ..data.streams import feature_dim
from .bilstm import BiLSTM
from .hagcn import HAGCN, hand_mask
from .stgcn import STGCN

MODEL_TYPES = ("bilstm", "stgcn", "hagcn")


def build_model(model_cfg: dict, num_classes: int, streams: Sequence[str]) -> nn.Module:
    """Build the model named by ``model_cfg['type']``; other keys override constructor defaults."""
    mtype = model_cfg["type"].lower()
    kwargs = {k: v for k, v in model_cfg.items() if k != "type"}
    if "channels" in kwargs:
        kwargs["channels"] = tuple(kwargs["channels"])
    F = feature_dim(streams)

    if mtype == "bilstm":
        return BiLSTM(in_features=F * C.N, num_classes=num_classes, **kwargs)
    if mtype == "stgcn":
        return STGCN(in_channels=F, num_classes=num_classes, A=G.build_adjacency(), **kwargs)
    if mtype == "hagcn":
        return HAGCN(
            in_channels=F,
            num_classes=num_classes,
            A=G.build_adjacency(),
            M_hand=hand_mask(),
            **kwargs,
        )
    raise ValueError(f"Unknown model type: {mtype!r}. Valid: {MODEL_TYPES}")
