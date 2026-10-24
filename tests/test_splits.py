"""Split protocols, including signer leakage checks on the released split files."""

import glob
import json
import os

import numpy as np
import pytest

from lsp.data.splits import (
    make_intra_subject_split,
    make_subject_disjoint,
    pick_signers,
    random_subject_split,
)

from .conftest import REPO_ROOT


def test_intra_subject_split_has_no_sample_overlap():
    y = np.array([i % 20 for i in range(301)])
    s = make_intra_subject_split(y, seed=42)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) + len(test) == 301


def test_subject_disjoint_split_no_signer_in_train_and_test():
    signers = np.array([f"s{i % 6}" for i in range(120)])
    split = make_subject_disjoint(signers, test_signers=["s0", "s1"])
    train_signers = set(signers[split["train"]].tolist())
    test_signers = set(signers[split["test"]].tolist())
    assert not (train_signers & test_signers)
    assert test_signers == {"s0", "s1"}
    assert len(split["train"]) + len(split["test"]) == 120


def test_random_subject_split_matches_chosen_signers():
    signers = np.array([f"s{i % 6}" for i in range(120)])
    split, chosen = random_subject_split(signers, n_test_signers=2, seed=1)
    assert len(chosen) == 2
    assert set(signers[split["test"]].tolist()) == set(chosen)
    assert not set(signers[split["train"]].tolist()) & set(chosen)


def test_heldout_validation_signers_exclude_test_signers():
    signers = np.array([f"lsa64_{i:03d}" for i in range(1, 11)] * 5)
    _, test = random_subject_split(signers, 2, seed=7)
    val = pick_signers(signers, 1, seed=7, exclude=test)
    assert len(val) == 1 and not set(val) & set(test)


def test_subject_disjoint_rejects_unknown_signers():
    with pytest.raises(ValueError):
        make_subject_disjoint(np.array(["a", "b"]), test_signers=["z"])


LSA64_DIR = os.path.join(REPO_ROOT, "splits", "lsa64")
LSA64_SPLITS = sorted(glob.glob(os.path.join(LSA64_DIR, "subject_split_seed*.json")))


def test_lsa64_split_files_exist():
    assert len(LSA64_SPLITS) == 3


@pytest.mark.parametrize("path", LSA64_SPLITS, ids=os.path.basename)
def test_released_lsa64_splits_are_signer_disjoint(path):
    """``sample_signers.json`` maps every bundle index to its LSA64 signer id."""
    with open(os.path.join(LSA64_DIR, "sample_signers.json"), encoding="utf-8") as f:
        signer_of = json.load(f)["signers"]
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["protocol"] == "subject_disjoint"
    assert len(payload["split"]["train"]) + len(payload["split"]["test"]) == len(signer_of)
    train = {signer_of[i] for i in payload["split"]["train"]}
    test = {signer_of[i] for i in payload["split"]["test"]}
    assert not train & test, f"signer leakage in {os.path.basename(path)}: {train & test}"
    assert test == set(payload["test_signers"])


def test_released_lsp_split_is_labelled_intra_subject():
    """The LSP split is sample-level by design; it must not be presented as signer-disjoint."""
    with open(
        os.path.join(REPO_ROOT, "splits", "lsp", "intra_subject_v2.json"), encoding="utf-8"
    ) as f:
        payload = json.load(f)
    assert payload["protocol"] == "intra_subject"
    s = payload["split"]
    assert not set(s["train"]) & set(s["test"]) and not set(s["val"]) & set(s["test"])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == payload["num_samples"] == 540
