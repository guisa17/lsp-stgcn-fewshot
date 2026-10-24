"""Few-shot calibration of a frozen encoder and the accuracy-vs-K protocol.

Two calibration methods are evaluated on the same K samples per class of an unseen signer:
  - ProtoNet: class prototypes are the mean embeddings; prediction by nearest prototype.
  - Fine-tune-K (control baseline): a new linear layer trained on the frozen embeddings.
The encoder weights are never updated in either case.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..models.protonet import proto_logits


class Calibrator:
    """Frozen encoder (``.embed(x)``) plus prototypes computed from K samples per class."""

    def __init__(self, encoder: nn.Module, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.encoder = encoder.eval().to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.prototypes: torch.Tensor | None = None
        self.classes: list = []

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, labels: Sequence[int], classes: Sequence) -> None:
        """``x``: (K*C, T, V, F) calibration samples from one new signer.

        ``labels`` are indices ``0..len(classes)-1`` aligned with ``classes``. Replaces any previous
        calibration.
        """
        x = x.to(self.device)
        labels_t = torch.as_tensor(np.asarray(labels), device=self.device)
        embeddings = self.encoder.embed(x)

        n_classes = len(classes)
        prototypes = embeddings.new_zeros((n_classes, embeddings.shape[1]))
        for c in range(n_classes):
            mask = labels_t == c
            if not mask.any():
                raise ValueError(f"No calibration samples for class '{classes[c]}'")
            prototypes[c] = embeddings[mask].mean(dim=0)

        self.prototypes = prototypes
        self.classes = list(classes)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """``x``: (N, T, V, F) -> ``{"pred", "probs" (N, n_classes), "confidence"}``."""
        if self.prototypes is None:
            raise RuntimeError("Calibrator is not calibrated: call calibrate() first")
        embeddings = self.encoder.embed(x.to(self.device))
        probs = torch.softmax(proto_logits(embeddings, self.prototypes), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        return {
            "pred": [self.classes[i] for i in preds],
            "probs": probs,
            "confidence": probs.max(axis=1).tolist(),
        }


def finetune_head(
    model: nn.Module,
    x_calib: torch.Tensor,
    y_calib: Sequence[int],
    n_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Linear:
    """Control baseline: a new linear layer trained on frozen embeddings of the K samples/class."""
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(x_calib.to(device))
    head = nn.Linear(model.embedding_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    y_t = torch.as_tensor(y_calib, device=device, dtype=torch.long)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(head(embeddings), y_t)
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def finetune_predict(
    model: nn.Module, head: nn.Linear, x: torch.Tensor, device: torch.device
) -> np.ndarray:
    model.eval()
    return head(model.embed(x.to(device))).argmax(dim=1).cpu().numpy()


def split_calibration(
    y: np.ndarray, idx_signer: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[list[int], list[int], list[int]]:
    """For each class of this signer, draw K samples for calibration and keep the rest for eval.

    Classes with K or fewer samples are skipped, since nothing would be left to evaluate.
    """
    classes_present = sorted(set(y[idx_signer].tolist()))
    calib_idx, calib_labels, eval_idx = [], [], []
    for c in classes_present:
        c_idx = idx_signer[y[idx_signer] == c]
        if len(c_idx) <= k:
            continue
        chosen = rng.choice(c_idx, k, replace=False)
        rest = np.setdiff1d(c_idx, chosen)
        calib_idx.extend(chosen.tolist())
        calib_labels.extend([c] * k)
        eval_idx.extend(rest.tolist())
    return calib_idx, calib_labels, eval_idx


def evaluate_vs_k(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    signers: np.ndarray,
    test_signers: Sequence[str],
    ks: Sequence[int],
    device: str | torch.device,
    *,
    n_trials: int,
    seed: int,
    finetune_epochs: int,
    finetune_lr: float,
) -> tuple[dict, dict]:
    """Accuracy-vs-K curves for ProtoNet and fine-tune-K on each test signer.

    Returns ``(summary, raw)``: per-K means, and every per-trial accuracy.
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    proto_acc: dict[int, list[float]] = {k: [] for k in ks}
    finetune_acc: dict[int, list[float]] = {k: [] for k in ks}

    for signer in test_signers:
        idx_signer = np.flatnonzero(signers == signer)
        for k in ks:
            for _trial in range(n_trials):
                calib_idx, calib_labels, eval_idx = split_calibration(y, idx_signer, k, rng)
                if not calib_idx or not eval_idx:
                    continue
                calib_classes = sorted(set(calib_labels))
                local_labels = [calib_classes.index(c) for c in calib_labels]
                x_calib = torch.as_tensor(X[calib_idx], dtype=torch.float32)
                x_eval = torch.as_tensor(X[eval_idx], dtype=torch.float32)
                y_eval = y[eval_idx]

                calib = Calibrator(model, device=device)
                calib.calibrate(x_calib, local_labels, calib_classes)
                pred = np.array(calib.predict(x_eval)["pred"])
                proto_acc[k].append(float((pred == y_eval).mean()))

                head = finetune_head(
                    model,
                    x_calib,
                    local_labels,
                    len(calib_classes),
                    device,
                    epochs=finetune_epochs,
                    lr=finetune_lr,
                )
                pred_idx = finetune_predict(model, head, x_eval, device)
                pred_ft = np.array([calib_classes[i] for i in pred_idx])
                finetune_acc[k].append(float((pred_ft == y_eval).mean()))

    summary = {
        "protonet": {str(k): float(np.mean(v)) if v else None for k, v in proto_acc.items()},
        "finetune_k": {str(k): float(np.mean(v)) if v else None for k, v in finetune_acc.items()},
    }
    raw = {
        "protonet": {str(k): v for k, v in proto_acc.items()},
        "finetune_k": {str(k): v for k, v in finetune_acc.items()},
    }
    return summary, raw


def protonet_predictions(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    signers: np.ndarray,
    signer: str,
    ks: Sequence[int],
    device: str | torch.device,
    *,
    n_trials: int,
    seed: int,
) -> list[dict]:
    """ProtoNet predictions for every trial of one signer, drawn exactly as in ``evaluate_vs_k``.

    Returns one dict per trial with ``k``, ``calib_idx``, ``eval_idx``, ``pred`` and ``confidence``.
    """
    rng = np.random.default_rng(seed)
    idx_signer = np.flatnonzero(signers == signer)
    trials = []
    for k in ks:
        for _trial in range(n_trials):
            calib_idx, calib_labels, eval_idx = split_calibration(y, idx_signer, k, rng)
            if not calib_idx or not eval_idx:
                continue
            calib_classes = sorted(set(calib_labels))
            calib = Calibrator(model, device=device)
            calib.calibrate(
                torch.as_tensor(X[calib_idx], dtype=torch.float32),
                [calib_classes.index(c) for c in calib_labels],
                calib_classes,
            )
            out = calib.predict(torch.as_tensor(X[eval_idx], dtype=torch.float32))
            trials.append({
                "k": k,
                "calib_idx": calib_idx,
                "eval_idx": eval_idx,
                "pred": [int(p) for p in out["pred"]],
                "confidence": out["confidence"],
            })  # fmt: skip
    return trials
