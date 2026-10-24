"""Episodic sampler, Prototypical Networks, meta-training, calibration and result aggregation."""

import numpy as np
import pytest
import torch

from lsp.evaluation.aggregate import aggregate_runs, check_distinct_splits
from lsp.evaluation.fewshot import Calibrator, evaluate_vs_k
from lsp.models.protonet import compute_prototypes, episodic_accuracy, proto_loss, run_episode
from lsp.models.stgcn import STGCN
from lsp.training.episodic import EpisodicBatchSampler, meta_train

from .conftest import synthetic_embeddings_dataset, tiny_adjacency

META_KW = dict(
    device="cpu",
    n_way=3,
    k_shot=2,
    q_query=2,
    n_episodes_train=4,
    n_episodes_val=3,
    epochs=2,
    lr=1e-3,
    seed=7,
    verbose=False,
)


def _tiny_stgcn(n_classes=5):
    return STGCN(in_channels=3, num_classes=n_classes, A=tiny_adjacency(6), channels=(4, 8))


def test_episodic_sampler_shapes():
    _, y, signers = synthetic_embeddings_dataset(n_classes=6, n_signers=4, reps=5)
    sampler = EpisodicBatchSampler(y, signers, 3, 2, 2, n_episodes=10, seed=1)
    assert len(sampler) == 10
    for ep in sampler:
        assert ep["support_idx"].shape == (6,) and ep["query_idx"].shape == (6,)
        assert sorted(set(ep["support_labels"].tolist())) == [0, 1, 2]
        assert not set(ep["support_idx"].tolist()) & set(ep["query_idx"].tolist())


def test_episodic_sampler_support_and_query_signers_are_disjoint():
    _, y, signers = synthetic_embeddings_dataset(n_classes=4, n_signers=4, reps=8)
    sampler = EpisodicBatchSampler(
        y, signers, 4, 3, 3, n_episodes=20, subject_disjoint=True, seed=2
    )
    disjoint = sum(
        not set(signers[ep["support_idx"]].tolist()) & set(signers[ep["query_idx"]].tolist())
        for ep in sampler
    )
    assert disjoint >= 18, f"only {disjoint}/20 episodes were signer-disjoint"


def test_episodic_sampler_rejects_insufficient_classes():
    _, y, signers = synthetic_embeddings_dataset(n_classes=2, n_signers=2, reps=2)
    with pytest.raises(ValueError):
        EpisodicBatchSampler(y, signers, n_way=5, k_shot=1, q_query=1, n_episodes=1)


def test_protonet_prototypes_and_separation():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    support = torch.cat([c.repeat(4, 1) + torch.randn(4, 2) * 0.05 for c in centers])
    prototypes = compute_prototypes(support, torch.tensor([0] * 4 + [1] * 4 + [2] * 4), n_way=3)
    assert torch.allclose(prototypes, centers, atol=0.2)
    query, labels = centers + torch.randn(3, 2) * 0.05, torch.tensor([0, 1, 2])
    assert episodic_accuracy(query, prototypes, labels) == 1.0
    assert torch.isfinite(proto_loss(query, prototypes, labels))


def test_run_episode_end_to_end():
    X, y, signers = synthetic_embeddings_dataset(n_classes=5, n_signers=3, reps=6, T=8)
    sampler = EpisodicBatchSampler(y, signers, 3, 2, 2, n_episodes=3, seed=3)
    model, X_t = _tiny_stgcn(), torch.as_tensor(X)
    for ep in sampler:
        loss, acc = run_episode(model, X_t, ep, device=torch.device("cpu"))
        assert torch.isfinite(loss) and 0.0 <= acc <= 1.0


def test_meta_train_with_heldout_validation_signer():
    X, y, signers = synthetic_embeddings_dataset(n_classes=5, n_signers=4, reps=6, T=8)
    result = meta_train(_tiny_stgcn(), X, y, signers, val_signers=["signer_3"], **META_KW)
    assert set(result) == {"model", "history", "best_val_acc", "best_epoch"}
    assert len(result["history"]) == 2 and 0.0 <= result["best_val_acc"] <= 1.0


def test_meta_train_fixed_epochs_keeps_final_weights():
    X, y, signers = synthetic_embeddings_dataset(n_classes=5, n_signers=4, reps=6, T=8)
    torch.manual_seed(0)
    model = _tiny_stgcn()
    result = meta_train(
        model, X, y, signers, val_signers=["signer_3"], select_best=False, **META_KW
    )
    assert result["model"] is model  # no checkpoint restored; weights are those of the last epoch


def test_calibrator_round_trip():
    X, y, _ = synthetic_embeddings_dataset(n_classes=4, n_signers=1, reps=5, T=8, seed=9)
    calib = Calibrator(_tiny_stgcn(4), device="cpu")
    x_t = torch.as_tensor(X)
    calib.calibrate(x_t, y, ["A", "B", "C", "D"])
    out = calib.predict(x_t[:6])
    assert calib.prototypes.shape == (4, 8)
    assert len(out["pred"]) == 6 and out["probs"].shape == (6, 4)


def test_evaluate_vs_k_is_deterministic_for_protonet():
    X, y, signers = synthetic_embeddings_dataset(n_classes=4, n_signers=2, reps=6, T=8)
    model = _tiny_stgcn(4).eval()
    kw = dict(n_trials=3, seed=5, finetune_epochs=2, finetune_lr=0.01)
    a, _ = evaluate_vs_k(model, X, y, signers, ["signer_1"], [1, 2], "cpu", **kw)
    b, _ = evaluate_vs_k(model, X, y, signers, ["signer_1"], [1, 2], "cpu", **kw)
    assert a["protonet"] == b["protonet"]


def test_aggregate_runs_pools_trials():
    def run(vals):
        return {
            "accuracy_vs_k": {
                "protonet": {"1": float(np.mean(vals))},  # noqa: E731
                "finetune_k": {"1": float(np.mean(vals))},
            },
            "accuracy_vs_k_raw": {"protonet": {"1": vals}, "finetune_k": {"1": vals}},
        }

    agg = aggregate_runs([run([0.5, 0.7]), run([0.9])], [1])
    assert agg["protonet"]["1"]["n_pooled"] == 3
    assert np.isclose(agg["protonet"]["1"]["mean"], 0.7)


def test_duplicate_signer_splits_are_refused():
    check_distinct_splits([["a", "b"], ["c", "d"]], ["e1", "e2"])
    with pytest.raises(ValueError, match="same held-out signers"):
        check_distinct_splits([["a", "b"], ["b", "a"]], ["e1", "e2"])
