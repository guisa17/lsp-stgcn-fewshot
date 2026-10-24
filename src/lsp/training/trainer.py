"""Supervised classification training with early stopping on validation accuracy."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module | None = None
) -> dict:
    """Return loss (if ``criterion`` is given), accuracy and the true/predicted labels."""
    model.eval()
    y_true, y_pred = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    loss = total_loss / n if n else 0.0
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    return {"loss": loss, "accuracy": acc, "y_true": y_true, "y_pred": y_pred}


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    lr_factor: float,
    lr_patience: int,
    min_lr: float,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict:
    """Adam + cross-entropy with ReduceLROnPlateau and early stopping.

    The weights with the best validation accuracy are restored before returning.
    """
    device = torch.device(device)
    model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    best_acc, best_state, best_epoch = -1.0, None, -1
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / seen

        val = evaluate(model, val_loader, device, criterion)
        scheduler.step(val["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_acc": val["accuracy"],
            }
        )

        if val["accuracy"] > best_acc:
            best_acc, best_epoch = val["accuracy"], epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
                f"| val_acc {val['accuracy']:.4f} (best {best_acc:.4f} @ {best_epoch})"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_acc": best_acc, "best_epoch": best_epoch}
