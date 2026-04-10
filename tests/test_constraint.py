"""Tests for Burt's constraint decomposition.

Reference values from Burt (2008) Appendix B, Figures B1 and B2.
Burt reports constraint × 100; we use raw [0, 1] scale.
"""

import numpy as np
import networkx as nx
from abm_core import burt_constraint, burt_constraint_decomposed


def _assert_decomposition_sums(G):
    """Verify that c_size + c_density + c_hierarchy == constraint for all nodes."""
    result = burt_constraint_decomposed(G)
    total = result["c_size"] + result["c_density"] + result["c_hierarchy"]
    np.testing.assert_allclose(total, result["constraint"], atol=1e-10)


def _assert_matches_original(G):
    """Verify decomposed total matches the original burt_constraint function."""
    original = burt_constraint(G)
    decomposed = burt_constraint_decomposed(G)["constraint"]
    np.testing.assert_allclose(decomposed, original, atol=1e-10)


# ---- Decomposition sum property ----

def test_decomposition_sums_sparse():
    """Sparse 3-node star: A -- B, A -- C, no B-C edge."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    _assert_decomposition_sums(G)
    _assert_matches_original(G)


def test_decomposition_sums_dense():
    """Complete K5."""
    G = nx.complete_graph(5)
    _assert_decomposition_sums(G)
    _assert_matches_original(G)


def test_decomposition_sums_hierarchical():
    """Star: hub connected to 4 leaves, no leaf-leaf edges."""
    G = nx.star_graph(4)  # node 0 is hub
    _assert_decomposition_sums(G)
    _assert_matches_original(G)


def test_decomposition_sums_broker():
    """Broker graph from reference diagrams."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)])
    _assert_decomposition_sums(G)
    _assert_matches_original(G)


# ---- Burt Figure B2: Small Networks (3 contacts, ego not shown) ----
# Burt's figures show ego's contacts only (ego has edges to all shown nodes).
# "Sparse": ego -- A, ego -- B, ego -- C, no inter-contact edges.
# "Dense": ego -- A, ego -- B, ego -- C, A-B, A-C, B-C (contacts form clique).
# "Hierarchical": ego -- A, ego -- B, ego -- C, A-B, A-C (A is hub among contacts).

def _build_ego_network(n_contacts, contact_edges):
    """Build ego network: ego (node 0) connected to all contacts (1..n),
    plus specified edges among contacts."""
    G = nx.Graph()
    for i in range(1, n_contacts + 1):
        G.add_edge(0, i)
    G.add_edges_from(contact_edges)
    return G


def test_figure_b2_small_sparse():
    """3 contacts, no inter-contact edges. Burt: constraint = 33."""
    G = _build_ego_network(3, [])
    result = burt_constraint_decomposed(G)
    # Ego constraint × 100 should be ~33
    assert abs(result["constraint"][0] * 100 - 33) < 1
    # In sparse network: c_density and c_hierarchy should be 0
    assert abs(result["c_density"][0]) < 1e-10
    assert abs(result["c_hierarchy"][0]) < 1e-10


def test_figure_b2_small_dense():
    """3 contacts, all inter-connected. Burt: constraint = 93."""
    G = _build_ego_network(3, [(1, 2), (1, 3), (2, 3)])
    result = burt_constraint_decomposed(G)
    assert abs(result["constraint"][0] * 100 - 93) < 1


def test_figure_b2_small_hierarchical():
    """3 contacts, A (node 1) connected to B and C. Burt: constraint = 84."""
    G = _build_ego_network(3, [(1, 2), (1, 3)])
    result = burt_constraint_decomposed(G)
    assert abs(result["constraint"][0] * 100 - 84) < 1


def test_figure_b2_medium_sparse():
    """5 contacts, no inter-contact edges. Burt: constraint = 20."""
    G = _build_ego_network(5, [])
    result = burt_constraint_decomposed(G)
    assert abs(result["constraint"][0] * 100 - 20) < 1


def test_figure_b2_medium_dense():
    """5 contacts, all inter-connected (K5 among contacts). Burt: constraint = 65."""
    edges = [(i, j) for i in range(1, 6) for j in range(i + 1, 6)]
    G = _build_ego_network(5, edges)
    result = burt_constraint_decomposed(G)
    assert abs(result["constraint"][0] * 100 - 65) < 1


def test_figure_b2_medium_hierarchical():
    """5 contacts, A (node 1) connected to all others. Burt: constraint = 59."""
    G = _build_ego_network(5, [(1, 2), (1, 3), (1, 4), (1, 5)])
    result = burt_constraint_decomposed(G)
    assert abs(result["constraint"][0] * 100 - 59) < 1


# ---- Figure B1 hand calculation ----

def test_figure_b1_gray_dot():
    """Figure B1: gray dot has 6 contacts (A-F), each contact has 2 edges.
    Burt reports total constraint on gray dot = 39.9."""
    G = nx.Graph()
    # Gray dot = node 0, contacts A-F = nodes 1-6
    for i in range(1, 7):
        G.add_edge(0, i)
    # Inter-contact edges from Figure B1: A-B, E-F (and both connect to gray dot)
    # Looking at the adjacency matrix in Figure B1:
    # A(1): connected to B(2), E(5), F(6), gray(0)
    # B(2): connected to A(1), D(4), F(6), gray(0)
    # C(3): connected to F(6), gray(0)
    # D(4): connected to B(2), F(6), gray(0)
    # E(5): connected to A(1), F(6), gray(0)
    # F(6): connected to A(1), B(2), D(4), E(5), gray(0)  -- wait, that's wrong
    # Actually from the adjacency matrix in the figure:
    #       A  B  C  D  E  F  gray
    # A     .  1  0  0  1  1  1
    # B     1  .  0  1  0  0  1
    # C     0  0  .  0  0  0  1
    # D     0  1  0  .  0  0  1
    # E     1  0  0  0  .  0  1
    # F     1  0  0  0  0  .  1
    # gray  1  1  1  1  1  1  .
    G.add_edges_from([
        (1, 2),  # A-B
        (1, 5),  # A-E
        (1, 6),  # A-F
        (2, 4),  # B-D
    ])
    result = burt_constraint_decomposed(G)
    # Burt reports 39.9 (×100) for the gray dot
    assert abs(result["constraint"][0] * 100 - 39.9) < 1.0


# ---- Isolates ----

def test_isolated_nodes_max_constraint():
    """Isolates get constraint=1.0 (no access to structural holes)."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)
    result = burt_constraint_decomposed(G)
    assert result["constraint"][2] == 1.0
    assert result["c_size"][2] == 1.0
    assert result["c_density"][2] == 0.0
    assert result["c_hierarchy"][2] == 0.0


# ---- Component interpretation ----

def test_sparse_has_zero_density_and_hierarchy():
    """When no inter-contact edges exist, indirect term is zero
    so c_density and c_hierarchy must both be zero."""
    G = _build_ego_network(5, [])
    result = burt_constraint_decomposed(G)
    assert abs(result["c_density"][0]) < 1e-10
    assert abs(result["c_hierarchy"][0]) < 1e-10
    # All constraint comes from size
    np.testing.assert_allclose(
        result["constraint"][0], result["c_size"][0], atol=1e-10
    )


def test_hierarchical_has_high_hierarchy():
    """Hub network should have high c_hierarchy relative to dense network."""
    G_hier = _build_ego_network(5, [(1, 2), (1, 3), (1, 4), (1, 5)])
    G_dense = _build_ego_network(5, [(i, j) for i in range(1, 6) for j in range(i + 1, 6)])
    h_hier = burt_constraint_decomposed(G_hier)["c_hierarchy"][0]
    h_dense = burt_constraint_decomposed(G_dense)["c_hierarchy"][0]
    # In the hierarchical network, indirect constraint is concentrated through the hub
    # so hierarchy component should be larger relative to total
    r_hier = h_hier / burt_constraint_decomposed(G_hier)["constraint"][0]
    r_dense = h_dense / burt_constraint_decomposed(G_dense)["constraint"][0]
    assert r_hier > r_dense
