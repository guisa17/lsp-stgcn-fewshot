"""Aggregation of accuracy-vs-K results across runs (seeds or encoders)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

METHODS = ("protonet", "finetune_k")


def aggregate_runs(runs: Sequence[dict], ks: Sequence[int]) -> dict:
    """Pool every per-trial accuracy of every run and report mean and std per method and K.

    Each run is a dict with ``accuracy_vs_k`` (per-K means) and ``accuracy_vs_k_raw`` (per-trial
    values). The per-run means are kept to spot outlier runs.
    """
    agg: dict = {m: {} for m in METHODS}
    for method in METHODS:
        for k in ks:
            pooled = [v for r in runs for v in r["accuracy_vs_k_raw"][method][str(k)]]
            per_run_means = [
                r["accuracy_vs_k"][method][str(k)]
                for r in runs
                if r["accuracy_vs_k"][method].get(str(k)) is not None
            ]
            agg[method][str(k)] = {
                "mean": float(np.mean(pooled)) if pooled else None,
                "std": float(np.std(pooled)) if pooled else None,
                "n_pooled": len(pooled),
                "per_seed_mean": per_run_means,
            }
    return agg


def check_distinct_splits(test_signer_sets: Sequence[Sequence[str]], labels: Sequence[str]) -> None:
    """Raise if two runs were trained with the same held-out signers.

    Averaging two runs of the same split would weight that split twice and make the spread look
    like it came from independent splits.
    """
    seen: dict[tuple[str, ...], str] = {}
    for signers, label in zip(test_signer_sets, labels, strict=False):
        key = tuple(sorted(signers))
        if key in seen:
            raise ValueError(
                f"Runs '{seen[key]}' and '{label}' share the same held-out signers {list(key)}; "
                "aggregate only one run per signer split."
            )
        seen[key] = label
