"""Target vocabulary (``demo/vocab.yaml``) used by the Record and Progress pages."""

from __future__ import annotations

import yaml

from . import paths


def load_vocab() -> dict:
    """``{"classes": [...], "target_takes_per_signer": int}``."""
    with open(paths.VOCAB_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "classes": data.get("classes", []),
        "target_takes_per_signer": int(data["target_takes_per_signer"]),
    }
