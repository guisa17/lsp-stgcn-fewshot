"""Model input/output shapes, HA-GCN initialization and checkpoint round trips."""

import pytest
import torch

from lsp.data import canonical as C
from lsp.data import graph as G
from lsp.models.build import build_model
from lsp.models.hagcn import HandAwareGraphConv, hand_mask
from lsp.training.checkpoint import load_encoder, save_encoder

NUM_CLASSES = 5
STREAMS = ("landmarks", "bones", "motion")  # F = 9


@pytest.mark.parametrize("mtype", ["bilstm", "stgcn", "hagcn"])
def test_forward_shapes(mtype):
    x = torch.randn(4, 30, C.N, 9)
    model = build_model({"type": mtype}, NUM_CLASSES, STREAMS)
    assert model(x).shape == (4, NUM_CLASSES)


@pytest.mark.parametrize("mtype", ["stgcn", "hagcn"])
def test_embed_shapes(mtype):
    model = build_model({"type": mtype, "channels": [16, 32]}, NUM_CLASSES, ("landmarks",))
    emb = model.eval().embed(torch.randn(3, 30, C.N, 3))
    assert emb.shape == (3, model.embedding_dim) == (3, 32)


def test_unknown_model_type():
    with pytest.raises(ValueError):
        build_model({"type": "transformer"}, NUM_CLASSES, STREAMS)


def test_hagcn_equals_base_adjacency_at_init():
    A = G.build_adjacency()
    conv = HandAwareGraphConv(9, 16, A, hand_mask())
    assert torch.allclose(conv.effective_adjacency(), torch.tensor(A), atol=1e-6)


def test_hand_mask_covers_both_hands_only():
    M = hand_mask()
    assert M.shape == (C.N, C.N)
    assert M.sum() == (2 * C.N_HAND) ** 2
    assert M[C.NOSE, C.NOSE] == 0 and M[C.FACE_OFF, C.FACE_OFF] == 0
    assert M[C.LH_OFF, C.RH_OFF] == 1


def test_encoder_checkpoint_round_trip(tmp_path):
    cfg = {"type": "hagcn", "channels": [16, 32], "t_kernel": 9, "dropout": 0.3, "drop_graph": 0.1}
    model = build_model(cfg, 7, STREAMS).eval()
    path = tmp_path / "enc.pt"
    save_encoder(
        str(path),
        model,
        model_cfg=cfg,
        streams=STREAMS,
        classes=list("abcdefg"),
        test_signers=["s1", "s2"],
        seed=3,
    )
    loaded, ckpt = load_encoder(str(path))
    x = torch.randn(2, 30, C.N, 9)
    assert torch.allclose(model.embed(x), loaded.embed(x))
    assert ckpt["test_signers"] == ["s1", "s2"] and ckpt["seed"] == 3
