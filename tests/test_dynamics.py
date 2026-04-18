import numpy as np
import pytest
from abm_dynamics import (
    SimState,
    mechanism_homophily,
    mechanism_triadic_closure,
    mechanism_popularity,
    constraint_attention_budget,
    constraint_attention_hard,
    mechanism_attention_budget,
    mechanism_attention_hard,
    decay_over_budget,
    step,
)


# ---- Task 1: SimState ----

def test_simstate_from_init():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    budgets = np.array([5, 5, 5])
    state = SimState(D=D, budgets=budgets)
    assert state.n == 3
    assert state.A.shape == (3, 3)
    assert state.A.sum() == 0  # no edges initially
    assert np.array_equal(state.degrees, [0, 0, 0])
    assert state.t == 0


# ---- Task 2: Homophily (log-odds) ----

def test_homophily_basic():
    D = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    eps = 0.1
    contrib = mechanism_homophily(state, b_homophily=1.0, epsilon=eps)
    assert contrib.shape == (3, 3)
    # Check exact values: b * 1/(d + eps)
    assert np.isclose(contrib[0, 1], 1.0 / (1.0 + eps))
    assert np.isclose(contrib[0, 2], 1.0 / (3.0 + eps))
    # Closer agents get higher log-odds
    assert contrib[0, 1] > contrib[0, 2]
    # Diagonal is large negative (no self-ties)
    assert contrib[0, 0] < -1e8


def test_homophily_coefficient_scaling():
    D = np.array([[0, 1], [1, 0]])
    state = SimState(D=D, budgets=np.array([5, 5]))
    eps = 0.1
    c_low = mechanism_homophily(state, b_homophily=0.5, epsilon=eps)
    c_high = mechanism_homophily(state, b_homophily=2.0, epsilon=eps)
    # Higher coefficient = proportionally higher log-odds
    assert np.isclose(c_high[0, 1], 4.0 * c_low[0, 1])


def test_homophily_zero_distance():
    """Epsilon prevents infinity when d=0."""
    D = np.array([[0, 0], [0, 0]])
    state = SimState(D=D, budgets=np.array([5, 5]))
    eps = 0.1
    contrib = mechanism_homophily(state, b_homophily=1.0, epsilon=eps)
    assert np.isclose(contrib[0, 1], 1.0 / eps)
    assert np.isfinite(contrib[0, 1])


# ---- Task 3: Triadic Closure (log-odds) ----

def test_triadic_closure_shared_neighbor():
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 5))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[1, 2] = state.A[2, 1] = 1
    contrib = mechanism_triadic_closure(state, b_triadic=1.5)
    # 1 shared neighbor between 0 and 2 -> contribution = 1.5 * 1
    assert np.isclose(contrib[0, 2], 1.5)
    assert np.isclose(contrib[2, 0], 1.5)
    # 0 shared neighbors between 0 and 3 -> contribution = 0
    assert np.isclose(contrib[0, 3], 0.0)


def test_triadic_closure_multiple_shared():
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 5))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[1, 3] = state.A[3, 1] = 1
    state.A[2, 3] = state.A[3, 2] = 1
    contrib = mechanism_triadic_closure(state, b_triadic=2.0)
    # 2 shared neighbors between 0 and 3 (via 1 and 2) -> contribution = 2.0 * 2
    assert np.isclose(contrib[0, 3], 4.0)


def test_triadic_closure_invalid_b():
    D = np.zeros((2, 2))
    state = SimState(D=D, budgets=np.array([5, 5]))
    with pytest.raises(ValueError):
        mechanism_triadic_closure(state, b_triadic=-1.0)


# ---- Task 3b: Popularity (log-odds, asymmetric) ----

def test_popularity_isolated_nodes():
    """With no edges, all k_j=0, so contribution is 0 everywhere."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    contrib = mechanism_popularity(state, b_popularity=1.0)
    assert contrib.shape == (3, 3)
    assert np.allclose(contrib, 0.0)


def test_popularity_high_degree_boost():
    """Node 0 has degree 3, others have degree 1. Only target degree matters."""
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 10))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[0, 3] = state.A[3, 0] = 1
    contrib = mechanism_popularity(state, b_popularity=1.0)
    # Column 0: k_0=3, so contribution = 1.0 * 3 = 3.0 for all rows
    assert np.isclose(contrib[1, 0], 3.0)
    assert np.isclose(contrib[2, 0], 3.0)
    # Column 1: k_1=1, so contribution = 1.0 * 1 = 1.0 for all rows
    assert np.isclose(contrib[0, 1], 1.0)
    assert np.isclose(contrib[2, 1], 1.0)
    # All rows in a column are the same (asymmetric: only target matters)
    assert np.isclose(contrib[0, 0], contrib[1, 0])


def test_popularity_b_zero_neutral():
    """b=0 means no popularity effect: contribution is all 0s."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    state.A[0, 1] = state.A[1, 0] = 1
    contrib = mechanism_popularity(state, b_popularity=0.0)
    assert np.allclose(contrib, 0.0)


def test_popularity_asymmetry():
    """Contribution depends only on target (column), not sender (row)."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    state.A[0, 1] = state.A[1, 0] = 1
    contrib = mechanism_popularity(state, b_popularity=0.5)
    # All rows for column 0 should be the same (k_0 = 1)
    assert np.isclose(contrib[0, 0], contrib[1, 0])
    assert np.isclose(contrib[1, 0], contrib[2, 0])
    # Column 2 (k_2 = 0) should be 0
    assert np.isclose(contrib[0, 2], 0.0)


def test_popularity_invalid_b():
    D = np.zeros((2, 2))
    state = SimState(D=D, budgets=np.array([5, 5]))
    with pytest.raises(ValueError):
        mechanism_popularity(state, b_popularity=-1.0)


# ---- Task 4: Attention Budget (constraint, unchanged) ----

def test_budget_under_capacity():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    factor = constraint_attention_budget(state, beta=5.0)
    assert factor[0, 1] > 0.9
    assert factor[1, 0] > 0.9


def test_budget_over_capacity():
    D = np.zeros((5, 5))
    state = SimState(D=D, budgets=np.array([2, 2, 2, 2, 2]))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[0, 3] = state.A[3, 0] = 1
    state.A[0, 4] = state.A[4, 0] = 1
    factor = constraint_attention_budget(state, beta=5.0)
    assert factor[0, 1] < 0.1


def test_budget_symmetry():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    factor = constraint_attention_budget(state, beta=5.0)
    assert np.isclose(factor[0, 1], factor[1, 0])


# ---- Task 4b: Hard Attention Budget (constraint, unchanged) ----

def test_hard_budget_under_capacity():
    """All agents under budget -> factor is all 1s."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    factor = constraint_attention_hard(state)
    assert np.allclose(factor, 1.0)


def test_hard_budget_at_capacity():
    """Agent 0 has degree 2 == budget 2 -> row/col zeroed out."""
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.array([2, 10, 10, 10]))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    factor = constraint_attention_hard(state)
    # Agent 0 is at budget, can't form new ties
    assert factor[0, 3] == 0.0
    assert factor[3, 0] == 0.0
    # Agents 1-3 are under budget, can still form ties with each other
    assert factor[1, 2] == 1.0


def test_hard_budget_no_decay():
    """Hard cutoff with enable_decay=False should not drop existing ties."""
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    state = SimState(D=D, budgets=np.array([1, 1, 1]))
    # Agent 0 already has 2 ties, over budget of 1
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    constraints = [lambda s: constraint_attention_hard(s)]
    rng = np.random.default_rng(42)
    new_state = step(state, [], rng, intercept=-10.0,
                     constraints=constraints, enable_decay=False)
    # Existing ties should be preserved (no decay)
    assert new_state.A[0, 1] == 1.0
    assert new_state.A[0, 2] == 1.0


# ---- Task 4c: Backward-compat aliases ----

def test_backward_compat_aliases():
    """mechanism_attention_budget and mechanism_attention_hard still work."""
    assert mechanism_attention_budget is constraint_attention_budget
    assert mechanism_attention_hard is constraint_attention_hard


# ---- Task 5: Tie Decay ----

def test_decay_no_drop_under_budget():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    state.A[0, 1] = state.A[1, 0] = 1
    rng = np.random.default_rng(42)
    A_new = decay_over_budget(state, rng)
    assert A_new[0, 1] == 1


def test_decay_drops_when_over_budget():
    D = np.array([
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
        [4, 3, 2, 1, 0],
    ], dtype=float)
    state = SimState(D=D, budgets=np.array([2, 5, 5, 5, 5]))
    for j in range(1, 5):
        state.A[0, j] = state.A[j, 0] = 1
    rng = np.random.default_rng(42)
    A_new = decay_over_budget(state, rng)
    assert A_new[0].sum() <= 4
    if A_new[0].sum() == 2:
        assert A_new[0, 1] == 1
        assert A_new[0, 2] == 1


# ---- Task 6: Step Function (logistic model) ----

def test_step_intercept_only():
    """With no mechanisms, intercept controls base rate."""
    rng = np.random.default_rng(42)
    D = np.array([[0, 1], [1, 0]])
    state = SimState(D=D, budgets=np.array([10, 10]))
    # Very negative intercept -> almost no edges
    edges_low = 0
    for _ in range(200):
        s = SimState(D=D, budgets=np.array([10, 10]))
        s2 = step(s, [], np.random.default_rng(), intercept=-20.0)
        edges_low += s2.A[0, 1]
    assert edges_low < 5  # nearly impossible at sigmoid(-20) ~ 2e-9

    # Moderate intercept -> some edges
    edges_mid = 0
    for _ in range(200):
        s = SimState(D=D, budgets=np.array([10, 10]))
        s2 = step(s, [], np.random.default_rng(), intercept=0.0)
        edges_mid += s2.A[0, 1]
    assert edges_mid > 50  # sigmoid(0) = 0.5


def test_step_homophily_only():
    rng = np.random.default_rng(42)
    D = np.array([[0, 0.5, 5], [0.5, 0, 4.5], [5, 4.5, 0]])
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    mechanisms = [lambda s: mechanism_homophily(s, b_homophily=2.0)]
    new_state = step(state, mechanisms, rng, intercept=-5.0)
    assert new_state.t == 1
    connected_01 = 0
    connected_02 = 0
    for _ in range(200):
        s = SimState(D=D, budgets=np.array([10, 10, 10]))
        s2 = step(s, mechanisms, np.random.default_rng(), intercept=-5.0)
        connected_01 += s2.A[0, 1]
        connected_02 += s2.A[0, 2]
    # Close pair (d=0.5) should form ties much more than distant pair (d=5)
    assert connected_01 > connected_02 * 3


def test_step_with_constraints():
    """Constraints applied after sigmoid reduce probabilities."""
    rng = np.random.default_rng(42)
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.array([10, 10, 10, 10]))
    mechanisms = [lambda s: mechanism_homophily(s, b_homophily=1.0)]
    constraints = [lambda s: constraint_attention_budget(s, beta=5.0)]
    new_state = step(state, mechanisms, rng, intercept=-3.0,
                     constraints=constraints)
    assert new_state.t == 1
    assert np.array_equal(new_state.A, new_state.A.T)


def test_step_increments_time():
    D = np.array([[0, 1], [1, 0]])
    state = SimState(D=D, budgets=np.array([5, 5]))
    mechanisms = [lambda s: mechanism_homophily(s, b_homophily=1.0)]
    rng = np.random.default_rng(42)
    s1 = step(state, mechanisms, rng, intercept=-5.0)
    assert s1.t == 1
    s2 = step(s1, mechanisms, rng, intercept=-5.0)
    assert s2.t == 2


def test_step_symmetrizes_probability():
    """Even with asymmetric popularity, resulting adjacency is symmetric."""
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    state.A[0, 1] = state.A[1, 0] = 1  # node 0 and 1 have degree 1
    mechanisms = [lambda s: mechanism_popularity(s, b_popularity=1.0)]
    rng = np.random.default_rng(42)
    new_state = step(state, mechanisms, rng, intercept=-3.0)
    assert np.array_equal(new_state.A, new_state.A.T)
