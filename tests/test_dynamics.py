import numpy as np
from abm_dynamics import (
    SimState,
    mechanism_homophily,
    mechanism_triadic_closure,
    mechanism_popularity,
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


# ---- Task 2: Homophily ----

def test_homophily_basic():
    D = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    lam = 1.0
    factor = mechanism_homophily(state, lam=lam)
    assert factor.shape == (3, 3)
    assert np.isclose(factor[0, 1], np.exp(-1.0))
    assert np.isclose(factor[0, 2], np.exp(-3.0))
    # Closer agents have higher probability
    assert factor[0, 1] > factor[0, 2]
    # Diagonal is zero (no self-ties)
    assert factor[0, 0] == 0.0


def test_homophily_lambda_scaling():
    D = np.array([[0, 1], [1, 0]])
    state = SimState(D=D, budgets=np.array([5, 5]))
    f_low = mechanism_homophily(state, lam=0.5)
    f_high = mechanism_homophily(state, lam=2.0)
    assert f_low[0, 1] > f_high[0, 1]


# ---- Task 3: Triadic Closure ----

def test_triadic_closure_shared_neighbor():
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 5))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[1, 2] = state.A[2, 1] = 1
    tau = 1.5
    factor = mechanism_triadic_closure(state, tau=tau)
    assert np.isclose(factor[0, 2], tau ** 1)
    assert np.isclose(factor[2, 0], tau ** 1)
    assert np.isclose(factor[0, 3], 1.0)


def test_triadic_closure_multiple_shared():
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 5))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[1, 3] = state.A[3, 1] = 1
    state.A[2, 3] = state.A[3, 2] = 1
    tau = 2.0
    factor = mechanism_triadic_closure(state, tau=tau)
    assert np.isclose(factor[0, 3], tau ** 2)


# ---- Task 3b: Popularity ----

def test_popularity_isolated_nodes():
    """With no edges, all (k+1)^mu = 1^mu = 1, so factor is all 1s."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    factor = mechanism_popularity(state, mu=1.0)
    assert factor.shape == (3, 3)
    assert np.allclose(factor, 1.0)


def test_popularity_high_degree_boost():
    """Node 0 has degree 3, others have degree 1. Node 0 pairs get highest boost."""
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 10))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[0, 3] = state.A[3, 0] = 1
    mu = 1.0
    factor = mechanism_popularity(state, mu=mu)
    # Node 0: k=3, pop=(3+1)^1=4. Nodes 1,2,3: k=1, pop=(1+1)^1=2.
    assert np.isclose(factor[1, 2], 2.0 * 2.0)  # both degree-1
    assert np.isclose(factor[0, 1], 4.0 * 2.0)  # degree-3 * degree-1


def test_popularity_mu_zero_neutral():
    """mu=0 means no popularity effect: factor is all 1s regardless of degree."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    state.A[0, 1] = state.A[1, 0] = 1
    factor = mechanism_popularity(state, mu=0.0)
    assert np.allclose(factor, 1.0)


def test_popularity_symmetry():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    state.A[0, 1] = state.A[1, 0] = 1
    factor = mechanism_popularity(state, mu=0.5)
    assert np.isclose(factor[0, 1], factor[1, 0])
    assert np.isclose(factor[0, 2], factor[2, 0])


def test_popularity_invalid_mu():
    D = np.zeros((2, 2))
    state = SimState(D=D, budgets=np.array([5, 5]))
    try:
        mechanism_popularity(state, mu=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---- Task 4: Attention Budget ----

def test_budget_under_capacity():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    factor = mechanism_attention_budget(state, beta=5.0)
    assert factor[0, 1] > 0.9
    assert factor[1, 0] > 0.9


def test_budget_over_capacity():
    D = np.zeros((5, 5))
    state = SimState(D=D, budgets=np.array([2, 2, 2, 2, 2]))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[0, 3] = state.A[3, 0] = 1
    state.A[0, 4] = state.A[4, 0] = 1
    factor = mechanism_attention_budget(state, beta=5.0)
    assert factor[0, 1] < 0.1


def test_budget_symmetry():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    factor = mechanism_attention_budget(state, beta=5.0)
    assert np.isclose(factor[0, 1], factor[1, 0])


# ---- Task 4b: Hard Attention Budget ----

def test_hard_budget_under_capacity():
    """All agents under budget -> factor is all 1s."""
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    factor = mechanism_attention_hard(state)
    assert np.allclose(factor, 1.0)


def test_hard_budget_at_capacity():
    """Agent 0 has degree 2 == budget 2 -> row/col zeroed out."""
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.array([2, 10, 10, 10]))
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    factor = mechanism_attention_hard(state)
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
    mechanisms = [lambda s: mechanism_attention_hard(s)]
    rng = np.random.default_rng(42)
    new_state = step(state, mechanisms, rng, enable_decay=False)
    # Existing ties should be preserved (no decay)
    assert new_state.A[0, 1] == 1.0
    assert new_state.A[0, 2] == 1.0


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


# ---- Task 6: Step Function ----

def test_step_homophily_only():
    rng = np.random.default_rng(42)
    D = np.array([[0, 0.5, 5], [0.5, 0, 4.5], [5, 4.5, 0]])
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    mechanisms = [lambda s: mechanism_homophily(s, lam=1.0)]
    new_state = step(state, mechanisms, rng)
    assert new_state.t == 1
    connected_01 = 0
    connected_02 = 0
    for _ in range(200):
        s = SimState(D=D, budgets=np.array([10, 10, 10]))
        s2 = step(s, mechanisms, np.random.default_rng())
        connected_01 += s2.A[0, 1]
        connected_02 += s2.A[0, 2]
    assert connected_01 > connected_02 * 5


def test_step_combined_mechanisms():
    rng = np.random.default_rng(42)
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.array([10, 10, 10, 10]))
    mechanisms = [
        lambda s: mechanism_homophily(s, lam=0.1),
        lambda s: mechanism_attention_budget(s, beta=5.0),
    ]
    new_state = step(state, mechanisms, rng)
    assert new_state.t == 1
    assert np.array_equal(new_state.A, new_state.A.T)


def test_step_increments_time():
    D = np.array([[0, 1], [1, 0]])
    state = SimState(D=D, budgets=np.array([5, 5]))
    mechanisms = [lambda s: mechanism_homophily(s, lam=1.0)]
    rng = np.random.default_rng(42)
    s1 = step(state, mechanisms, rng)
    assert s1.t == 1
    s2 = step(s1, mechanisms, rng)
    assert s2.t == 2
