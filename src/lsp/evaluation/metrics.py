"""Classification metrics, confusion matrix and CPU latency."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Accuracy and macro precision / recall / F1."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def confusion(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def plot_confusion(
    cm: np.ndarray, class_names: Sequence[str], out_path: str, title: str = "Confusion matrix"
) -> None:
    """Save the confusion matrix as an image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    size = max(8, len(class_names) * 0.5), max(6, len(class_names) * 0.5)
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def measure_latency(
    model: nn.Module, sample: torch.Tensor, device: str = "cpu", n_warmup: int = 5, n_runs: int = 50
) -> dict[str, float]:
    """Inference latency for a single sequence ``(T, V, F)``. Returns ms per inference and FPS."""
    model.eval().to(device)
    x = sample.unsqueeze(0).to(device) if sample.dim() == 3 else sample.to(device)
    for _ in range(n_warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    elapsed = time.perf_counter() - t0
    ms = (elapsed / n_runs) * 1000.0
    return {"latency_ms": float(ms), "fps": float(1000.0 / ms) if ms > 0 else float("inf")}
