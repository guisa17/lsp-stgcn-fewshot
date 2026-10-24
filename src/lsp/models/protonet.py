"""Prototypical Networks (Snell, Swersky & Zemel, 2017).

A class prototype is the mean embedding of its support samples. A query embedding is classified by
negative Euclidean distance to each prototype (a nearest-centroid classifier, trainable end to end).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_way: int) -> torch.Tensor:
    """``embeddings``: (N, D); ``labels``: (N,) in ``[0, n_way)``. Returns ``(n_way, D)``."""
    D = embeddings.shape[1]
    prototypes = embeddings.new_zeros((n_way, D))
    for c in range(n_way):
        mask = labels == c
        if not mask.any():
            raise ValueError(f"No support embeddings for local class {c}")
        prototypes[c] = embeddings[mask].mean(dim=0)
    return prototypes


def proto_logits(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Negative Euclidean distance from each query to each prototype, used as logits: (Q, n_way)."""
    return -torch.cdist(query_embeddings, prototypes)


def proto_loss(
    query_embeddings: torch.Tensor, prototypes: torch.Tensor, query_labels: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over the distance logits."""
    return F.cross_entropy(proto_logits(query_embeddings, prototypes), query_labels)


@torch.no_grad()
def episodic_accuracy(
    query_embeddings: torch.Tensor, prototypes: torch.Tensor, query_labels: torch.Tensor
) -> float:
    preds = proto_logits(query_embeddings, prototypes).argmax(dim=1)
    return (preds == query_labels).float().mean().item()


def run_episode(
    encoder: torch.nn.Module, x: torch.Tensor, episode: dict, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Run one episode: embeddings -> prototypes -> loss and accuracy.

    ``x`` holds every sample of the bundle ``(M, T, V, F)``; ``episode`` is a dict produced by
    ``EpisodicBatchSampler`` (indices into ``x`` and local labels ``0..n_way-1``).
    """
    support_x = x[episode["support_idx"]].to(device)
    query_x = x[episode["query_idx"]].to(device)
    support_labels = torch.as_tensor(episode["support_labels"], device=device)
    query_labels = torch.as_tensor(episode["query_labels"], device=device)
    n_way = int(support_labels.max().item()) + 1

    support_emb = encoder.embed(support_x)
    query_emb = encoder.embed(query_x)

    prototypes = compute_prototypes(support_emb, support_labels, n_way)
    loss = proto_loss(query_emb, prototypes, query_labels)
    acc = episodic_accuracy(query_emb, prototypes, query_labels)
    return loss, acc
