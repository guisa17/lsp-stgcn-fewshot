"""Every released config loads, has the required keys and builds its model."""

import glob
import os

import pytest

from lsp.models.build import build_model
from lsp.utils.config import apply_overrides, load_config

from .conftest import REPO_ROOT

CONFIGS = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "**", "*.yaml"), recursive=True))
TRAIN_CONFIGS = [c for c in CONFIGS if os.sep + "lsp" + os.sep in c and "prepare" not in c]
META_CONFIGS = [c for c in CONFIGS if os.sep + "lsa64" + os.sep in c and "prepare" not in c]
TRANSFER_CONFIGS = [c for c in CONFIGS if os.sep + "transfer" + os.sep in c]


def _name(path):
    return os.path.relpath(path, REPO_ROOT)


def test_expected_configs_exist():
    assert len(TRAIN_CONFIGS) == 4 and len(META_CONFIGS) == 4 and len(TRANSFER_CONFIGS) == 3


@pytest.mark.parametrize("path", CONFIGS, ids=_name)
def test_config_loads(path):
    assert isinstance(load_config(path), dict)


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=_name)
def test_train_config(path):
    cfg = load_config(path)
    for key in (
        "exp_name",
        "seed",
        "deterministic",
        "out_root",
        "data",
        "model",
        "train",
        "augment",
        "eval",
    ):
        assert key in cfg, f"{key} missing"
    for key in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "patience",
        "lr_factor",
        "lr_patience",
        "min_lr",
        "num_workers",
        "device",
    ):
        assert key in cfg["train"], f"train.{key} missing"
    assert cfg["augment"]["mirror"] is False  # the reported runs did not use the mirror
    build_model(cfg["model"], num_classes=30, streams=cfg["data"]["streams"])


@pytest.mark.parametrize("path", META_CONFIGS, ids=_name)
def test_meta_train_config(path):
    cfg = load_config(path)
    assert cfg["mode"] == "meta_train"
    assert cfg["validation"]["source"] in ("test_signers", "heldout")
    for section in ("pretrain", "meta", "calibration", "split"):
        assert section in cfg
    build_model(cfg["model"], num_classes=64, streams=cfg["data"]["streams"])


@pytest.mark.parametrize("path", TRANSFER_CONFIGS, ids=_name)
def test_transfer_config(path):
    cfg = load_config(path)
    assert cfg["mode"] == "transfer"
    assert len(cfg["encoders"]) == len(set(cfg["encoders"]))
    assert all("id" in s and "ks" in s for s in cfg["signers"])


def test_corrected_transfer_config_uses_one_encoder_per_split():
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "transfer", "hagcn_lsa64_to_lsp.yaml"))
    assert cfg["allow_duplicate_splits"] is False
    assert not any("singlerun" in e for e in cfg["encoders"])


def test_overrides():
    cfg = apply_overrides(
        {"train": {"epochs": 1}}, ["train.epochs=5", "train.device=cuda", "x.y=[1,2]"]
    )
    assert cfg["train"] == {"epochs": 5, "device": "cuda"} and cfg["x"]["y"] == [1, 2]
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])
