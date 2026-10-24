"""Episodic meta-training of an encoder with Prototypical Networks.

``EpisodicBatchSampler`` builds C-way K-shot Q-query episodes. In ``subject_disjoint`` mode a single
partition of all available signers into a support group and a query group is drawn per episode and
shared by every class of the episode. Drawing it per class instead lets the aggregated support and
query signer sets overlap almost always as soon as an episode has more than one class.

``pretrain_classifier`` optionally warms the encoder up with standard cross-entropy, and
``meta_train`` runs the episodic loop on ``protonet.proto_loss``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.protonet import run_episode


class EpisodicBatchSampler:
    def __init__(
        self,
        labels: Sequence[int],
        signers: Sequence[str],
        n_way: int,
        k_shot: int,
        q_query: int,
        n_episodes: int,
        subject_disjoint: bool = True,
        seed: int | None = None,
    ):
        self.labels = np.asarray(labels)
        self.signers = np.asarray(signers)
        if len(self.labels) != len(self.signers):
            raise ValueError("labels and signers must have the same length")
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        self.subject_disjoint = subject_disjoint
        self.rng = np.random.default_rng(seed)

        # class -> {signer -> [idx, ...]}
        self._by_class_signer: dict = {}
        for idx, (lab, sig) in enumerate(zip(self.labels, self.signers, strict=False)):
            self._by_class_signer.setdefault(lab, {}).setdefault(sig, []).append(idx)
        self._all_signers = sorted({s for m in self._by_class_signer.values() for s in m})

        need = k_shot + q_query
        self._eligible_classes = [
            c
            for c, sig_map in self._by_class_signer.items()
            if sum(len(v) for v in sig_map.values()) >= need
        ]
        if len(self._eligible_classes) < n_way:
            raise ValueError(
                f"Only {len(self._eligible_classes)} classes have >= {need} samples "
                f"(k_shot + q_query); at least n_way={n_way} are needed."
            )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.n_episodes):
            yield self._sample_episode()

    def _sample_episode(self) -> dict:
        classes = self.rng.choice(self._eligible_classes, size=self.n_way, replace=False)

        support_group = query_group = None
        if self.subject_disjoint and len(self._all_signers) >= 2:
            shuffled = list(self._all_signers)
            self.rng.shuffle(shuffled)
            half = max(1, len(shuffled) // 2)
            support_group, query_group = set(shuffled[:half]), set(shuffled[half:])

        support_idx, query_idx, support_lab, query_lab = [], [], [], []
        for way, c in enumerate(classes):
            sig_map = self._by_class_signer[c]
            if support_group is not None:
                s_idx, q_idx = self._split_by_groups(sig_map, support_group, query_group)
            else:
                s_idx, q_idx = self._split_mixed(sig_map)
            support_idx.extend(s_idx)
            query_idx.extend(q_idx)
            support_lab.extend([way] * len(s_idx))
            query_lab.extend([way] * len(q_idx))
        return {
            "support_idx": np.array(support_idx, dtype=np.int64),
            "query_idx": np.array(query_idx, dtype=np.int64),
            "support_labels": np.array(support_lab, dtype=np.int64),
            "query_labels": np.array(query_lab, dtype=np.int64),
            "classes": classes,  # original ids of the n_way classes, in 'way' order
        }

    def _split_by_groups(
        self, sig_map: dict, support_group: set, query_group: set
    ) -> tuple[list, list]:
        """Use the episode's signer partition. If this class lacks enough samples on either side,
        fall back to mixed sampling for this class only rather than dropping the episode."""
        support_pool = [i for s, idxs in sig_map.items() if s in support_group for i in idxs]
        query_pool = [i for s, idxs in sig_map.items() if s in query_group for i in idxs]
        if len(support_pool) >= self.k_shot and len(query_pool) >= self.q_query:
            s_idx = self.rng.choice(support_pool, self.k_shot, replace=False).tolist()
            q_idx = self.rng.choice(query_pool, self.q_query, replace=False).tolist()
            return s_idx, q_idx
        return self._split_mixed(sig_map)

    def _split_mixed(self, sig_map: dict) -> tuple[list, list]:
        pool = [i for idxs in sig_map.values() for i in idxs]
        chosen = self.rng.choice(pool, self.k_shot + self.q_query, replace=False)
        return chosen[: self.k_shot].tolist(), chosen[self.k_shot :].tolist()


def pretrain_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool = True,
) -> nn.Module:
    """Standard cross-entropy over all classes (uses the full ``forward()`` with ``fc``)."""
    device = torch.device(device)
    model.to(device)
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, running = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        if verbose:
            print(
                f"  [pretrain] epoch {epoch}/{epochs} loss={running / total:.4f} "
                f"acc={correct / total:.3f}"
            )
    return model


def _remap_episode(episode: dict, idx_map: np.ndarray) -> dict:
    """Translate an episode's local indices (inside a subset) to global indices."""
    out = dict(episode)
    out["support_idx"] = idx_map[episode["support_idx"]]
    out["query_idx"] = idx_map[episode["query_idx"]]
    return out


def meta_train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    signers: np.ndarray,
    *,
    val_signers: Sequence[str] | None,
    device: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    n_episodes_train: int,
    n_episodes_val: int,
    epochs: int,
    lr: float,
    subject_disjoint: bool = True,
    select_best: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Meta-train ``model`` (it must expose ``.embed(x)``) on the samples of ``X``.

    If ``val_signers`` is given, validation episodes use only those signers and training episodes
    exclude them. If it is ``None``, validation reuses the training pool (for debugging only).

    With ``select_best=True`` the weights with the best episodic validation accuracy are restored;
    with ``False`` the weights after the last epoch are kept (fixed-epoch protocol).

    Returns ``{"model", "history", "best_val_acc", "best_epoch"}``.
    """
    device = torch.device(device)
    model.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_arr = np.asarray(y)
    signers_arr = np.asarray(signers)

    if val_signers is not None:
        val_mask = np.isin(signers_arr, list(val_signers))
        train_mask = ~val_mask
        if not val_mask.any():
            raise ValueError(f"None of val_signers={val_signers} appears in 'signers'")
    else:
        train_mask = np.ones(len(y_arr), dtype=bool)
        val_mask = train_mask

    train_idx_map = np.flatnonzero(train_mask)
    val_idx_map = np.flatnonzero(val_mask)

    train_sampler = EpisodicBatchSampler(
        y_arr[train_mask],
        signers_arr[train_mask],
        n_way,
        k_shot,
        q_query,
        n_episodes_train,
        subject_disjoint=subject_disjoint,
        seed=seed,
    )
    val_sampler = EpisodicBatchSampler(
        y_arr[val_mask],
        signers_arr[val_mask],
        n_way,
        k_shot,
        q_query,
        n_episodes_val,
        subject_disjoint=subject_disjoint,
        seed=seed + 1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_accs = [], []
        for episode in train_sampler:
            ep = _remap_episode(episode, train_idx_map)
            optimizer.zero_grad()
            loss, acc = run_episode(model, X_t, ep, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(acc)

        model.eval()
        val_accs = []
        with torch.no_grad():
            for episode in val_sampler:
                ep = _remap_episode(episode, val_idx_map)
                _, acc = run_episode(model, X_t, ep, device)
                val_accs.append(acc)

        val_acc = float(np.mean(val_accs))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "train_acc": float(np.mean(train_accs)),
                "val_acc": val_acc,
            }
        )
        if verbose:
            h = history[-1]
            print(
                f"  [meta] epoch {epoch}/{epochs} train_loss={h['train_loss']:.4f} "
                f"train_acc={h['train_acc']:.3f} val_acc={val_acc:.3f}"
            )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            if select_best:
                best_state = copy.deepcopy(model.state_dict())

    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_acc": best_acc, "best_epoch": best_epoch}
