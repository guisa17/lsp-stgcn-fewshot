"""YAML configuration loading with ``key.subkey=value`` command-line overrides."""

from __future__ import annotations

import ast
import copy
import os

import yaml


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping at the top level")
    return cfg


def _coerce(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply overrides such as ``train.epochs=200`` (nested keys separated by dots)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Invalid override: {ov!r} (expected key=value)")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def load_with_overrides(path: str, overrides: list[str] | None) -> dict:
    return apply_overrides(load_config(path), overrides)
