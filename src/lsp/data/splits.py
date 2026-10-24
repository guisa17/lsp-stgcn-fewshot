"""Dataset splits, persisted to JSON. Two protocols:

- ``intra_subject``: stratified 70/15/15 split by class over samples. Samples from the same signer
  can appear in train and test, so it measures vocabulary learning, not signer independence.
- ``subject_disjoint``: split by signer. No signer appears in both train and test. This is the
  protocol used for few-shot evaluation on unseen signers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def make_intra_subject_split(
    labels: Sequence[int], seed: int, val_size: float = 0.15, test_size: float = 0.15
) -> dict[str, list[int]]:
    """Return ``{train, val, test}`` index lists, stratified by class."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))

    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train, test_size=rel_val, random_state=seed, stratify=labels[idx_train]
    )
    return {
        "train": sorted(int(i) for i in idx_train),
        "val": sorted(int(i) for i in idx_val),
        "test": sorted(int(i) for i in idx_test),
    }


def make_subject_disjoint(
    signers: Sequence[str], test_signers: Iterable[str]
) -> dict[str, list[int]]:
    """Split indices into ``{train, test}`` by signer: no signer contributes to both sides."""
    signers = np.asarray(signers)
    test_set = set(test_signers)
    test_mask = np.isin(signers, list(test_set))

    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(test_idx) == 0:
        raise ValueError(f"None of test_signers={sorted(test_set)} appears in 'signers'")
    if len(train_idx) == 0:
        raise ValueError("Every signer is in test; nothing is left for training")

    train_signers = set(signers[train_idx].tolist())
    assert not (train_signers & test_set), "signer leakage: a signer is in both train and test"

    return {
        "train": sorted(int(i) for i in train_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def pick_signers(
    signers: Sequence[str], n: int, seed: int, exclude: Iterable[str] = ()
) -> list[str]:
    """Pick ``n`` signers at random (fixed seed) among those not in ``exclude``."""
    excluded = set(exclude)
    unique = sorted(s for s in set(np.asarray(signers).tolist()) if s not in excluded)
    if n >= len(unique):
        raise ValueError(f"n={n} >= available signers ({len(unique)})")
    rng = np.random.default_rng(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    return sorted(shuffled[:n])


def random_subject_split(
    signers: Sequence[str], n_test_signers: int, seed: int
) -> tuple[dict[str, list[int]], list[str]]:
    """Pick ``n_test_signers`` test signers at random (fixed seed); the rest go to train."""
    test_signers = pick_signers(signers, n_test_signers, seed)
    return make_subject_disjoint(signers, test_signers), test_signers


def save_split(split: dict, path: str, seed: int, protocol: str, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"protocol": protocol, "seed": seed, **(meta or {}), "split": split}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split(path: str) -> dict[str, list[int]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["split"]
