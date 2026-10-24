"""Skeleton graph: connectivity, bridges and partitioned adjacency."""

import numpy as np

from lsp.data import canonical as C
from lsp.data import graph as G


def test_graph_is_a_connected_tree():
    assert int((G.PARENT < 0).sum()) == 1
    assert len(G.EDGES) == C.N - 1 == 72
    adj = {i: set() for i in range(C.N)}
    for i, j in G.EDGES:
        adj[i].add(j)
        adj[j].add(i)
    seen, stack = {C.NOSE}, [C.NOSE]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert len(seen) == C.N


def test_bridges():
    assert G.PARENT[C.LH_WRIST] == C.L_WRIST
    assert G.PARENT[C.RH_WRIST] == C.R_WRIST
    face_to_nose = [i for i in range(C.FACE_OFF, C.N) if G.PARENT[i] == C.NOSE]
    assert len(face_to_nose) == 2


def test_adjacency_partitions():
    A = G.build_adjacency()
    assert A.shape == (G.NUM_PARTITIONS, C.N, C.N) == (3, 73, 73)
    assert np.allclose(A[0], np.eye(C.N))
    for k in (1, 2):
        row_sums = A[k].sum(axis=1)
        assert np.all(np.isclose(row_sums, 0) | np.isclose(row_sums, 1))
    # centripetal (i <- parent) and centrifugal (parent <- i) are transposed patterns
    assert np.array_equal(A[1] > 0, (A[2] > 0).T)
