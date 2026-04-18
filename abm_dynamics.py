"""
Modular dynamics engine for the structural holes ABM.

Each mechanism is a function: (SimState, params) -> (N, N) log-odds contribution.
Mechanisms are summed (with an intercept) to form log-odds, then passed through
the sigmoid to produce tie formation probability P.

Constraints (attention budget) are applied as post-hoc multiplicative masks
after the sigmoid, since they represent capacity limits rather than predictors.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """Current state of the simulation.

    D:       (N, N) distance matrix (fixed, from initialization).
    A:       (N, N) binary adjacency matrix (evolves each timestep).
    budgets: (N,) attention budget per agent.
    t:       current timestep.
    """
    D: np.ndarray
    budgets: np.ndarray
    A: np.ndarray = field(default=None)
    t: int = 0

    def __post_init__(self):
        if self.A is None:
            self.A = np.zeros_like(self.D, dtype=np.float64)

    @property
    def n(self) -> int:
        return self.D.shape[0]

    @property
    def degrees(self) -> np.ndarray:
        return self.A.sum(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Mechanisms — each returns an (N, N) additive log-odds contribution
# ---------------------------------------------------------------------------

def mechanism_homophily(
    state: SimState, b_homophily: float, epsilon: float = 0.1
) -> np.ndarray:
    """Homophily log-odds: closer agents get higher tie probability.

    Contribution[i,j] = b_homophily * (1 / (d_ij + epsilon))

    Args:
        state: current simulation state.
        b_homophily: coefficient. Higher = stronger preference for nearby agents.
        epsilon: small constant to prevent division by zero and bound the
                 contribution for very small distances.

    Returns:
        (N, N) additive log-odds matrix. Diagonal is -inf (no self-ties).
    """
    contribution = b_homophily * (1.0 / (state.D + epsilon))
    np.fill_diagonal(contribution, -1e9)
    return contribution


def mechanism_triadic_closure(
    state: SimState, b_triadic: float
) -> np.ndarray:
    """Triadic closure log-odds: boost per shared neighbor.

    Contribution[i,j] = b_triadic * shared_neighbors(i,j)

    Args:
        state: current simulation state.
        b_triadic: coefficient per shared neighbor. Must be >= 0.

    Returns:
        (N, N) additive log-odds matrix. Values >= 0.

    Raises:
        ValueError: if b_triadic < 0.
    """
    if b_triadic < 0:
        raise ValueError(f"b_triadic must be >= 0 (got {b_triadic})")
    shared = state.A @ state.A
    np.fill_diagonal(shared, 0.0)
    return b_triadic * shared


def mechanism_popularity(
    state: SimState, b_popularity: float
) -> np.ndarray:
    """Popularity log-odds: popular targets attract ties (asymmetric).

    Contribution[i,j] = b_popularity * k_j

    Only the target node's degree matters (preferential attachment).

    Args:
        state: current simulation state.
        b_popularity: coefficient. Must be >= 0.

    Returns:
        (N, N) additive log-odds matrix. Each column j has the same value.

    Raises:
        ValueError: if b_popularity < 0.
    """
    if b_popularity < 0:
        raise ValueError(f"b_popularity must be >= 0 (got {b_popularity})")
    k = state.degrees.astype(np.float64)
    contrib = np.broadcast_to(
        b_popularity * k[np.newaxis, :], (state.n, state.n)
    ).copy()
    return contrib


# ---------------------------------------------------------------------------
# Constraints — post-hoc multiplicative masks applied after sigmoid
# ---------------------------------------------------------------------------

def constraint_attention_budget(state: SimState, beta: float) -> np.ndarray:
    """Attention budget constraint: penalize tie formation when over capacity.

    sigma_i = 1 / (1 + exp(beta * (k_i - b_i)))
    Factor[i,j] = sigma_i * sigma_j

    Args:
        state: current simulation state.
        beta: sharpness of the budget cutoff.

    Returns:
        (N, N) factor matrix. Values in [0, 1].
    """
    k = state.degrees.astype(np.float64)
    sigma = 1.0 / (1.0 + np.exp(beta * (k - state.budgets)))
    return sigma[:, np.newaxis] * sigma[np.newaxis, :]


def constraint_attention_hard(state: SimState) -> np.ndarray:
    """Hard attention budget: agents at or over budget cannot form new ties.

    Factor[i,j] = open_i * open_j
    where open_i = 1 if k_i < b_i, else 0.

    Args:
        state: current simulation state.

    Returns:
        (N, N) factor matrix. Values in {0, 1}.
    """
    open_slots = (state.degrees < state.budgets).astype(np.float64)
    return open_slots[:, np.newaxis] * open_slots[np.newaxis, :]


# Keep old names as aliases for backward compatibility in tests/imports
mechanism_attention_budget = constraint_attention_budget
mechanism_attention_hard = constraint_attention_hard


# ---------------------------------------------------------------------------
# Tie decay
# ---------------------------------------------------------------------------

def decay_over_budget(state: SimState, rng: np.random.Generator) -> np.ndarray:
    """Drop ties for agents over their attention budget.

    Two-pass approach to avoid order-dependent bias:
      Pass 1: each agent independently marks its most-distant excess ties.
      Pass 2: an edge is dropped if EITHER endpoint marked it for removal.

    Args:
        state: current simulation state.
        rng: random number generator (used to randomize processing order).

    Returns:
        (N, N) updated adjacency matrix.
    """
    A_new = state.A.copy()
    drop_mask = np.zeros_like(A_new, dtype=bool)

    # Pass 1: each agent independently marks excess ties for removal
    degrees = A_new.sum(axis=1).astype(int)
    order = rng.permutation(state.n)
    for i in order:
        excess = degrees[i] - int(state.budgets[i])
        if excess <= 0:
            continue
        neighbors = np.where(A_new[i] > 0)[0]
        dists_to_neighbors = state.D[i, neighbors]
        drop_order = neighbors[np.argsort(-dists_to_neighbors)]
        for j in drop_order[:excess]:
            drop_mask[i, j] = True
            drop_mask[j, i] = True

    # Pass 2: apply all drops at once
    A_new[drop_mask] = 0.0

    return A_new


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Mechanism = Callable[[SimState], np.ndarray]
Constraint = Callable[[SimState], np.ndarray]


# ---------------------------------------------------------------------------
# Step function — compose mechanisms into one timestep
# ---------------------------------------------------------------------------

def step(
    state: SimState,
    mechanisms: list[Mechanism],
    rng: np.random.Generator,
    intercept: float = -5.0,
    constraints: list[Constraint] | None = None,
    enable_decay: bool = True,
) -> SimState:
    """Execute one simulation timestep using the logistic model.

    1. Sum all mechanism log-odds contributions + intercept.
    2. Apply sigmoid to get P(i,j).
    3. Apply constraint factors (attention budget masks).
    4. Symmetrize P for undirected sampling.
    5. Mask out existing ties (only form NEW edges).
    6. Sample new edges from P.
    7. Optionally apply tie decay for over-budget agents.

    Args:
        state: current simulation state.
        mechanisms: list of mechanism functions, each returns (N, N) log-odds.
        rng: random number generator.
        intercept: base rate in log-odds space. Default -5.0 (~0.7% base rate).
        constraints: list of constraint functions, each returns (N, N) factor
                     in [0, 1]. Applied multiplicatively after sigmoid.
        enable_decay: if True (default), drop excess ties each step.

    Returns:
        New SimState with updated adjacency and t+1.
    """
    n = state.n
    if constraints is None:
        constraints = []

    # Sum all mechanism log-odds contributions + intercept
    logit = np.full((n, n), intercept, dtype=np.float64)
    for mech in mechanisms:
        logit += mech(state)

    # Sigmoid: P = 1 / (1 + exp(-logit)), clamped to avoid overflow
    logit = np.clip(logit, -500, 500)
    P = 1.0 / (1.0 + np.exp(-logit))

    # Apply constraint factors (attention budget, etc.)
    for constraint in constraints:
        P *= constraint(state)

    # Symmetrize for undirected graph (average both directions)
    P = (P + P.T) / 2.0

    # Mask existing ties and self-ties
    P *= (1 - state.A)
    np.fill_diagonal(P, 0.0)

    # Sample new edges (upper triangle, then symmetrize)
    draws = rng.uniform(0, 1, size=(n, n))
    new_edges = np.triu(draws < P, k=1)
    new_edges = new_edges | new_edges.T

    # Combine with existing adjacency
    A_new = np.maximum(state.A, new_edges.astype(np.float64))

    # Optionally apply decay
    if enable_decay:
        decayed_state = SimState(D=state.D, budgets=state.budgets, A=A_new, t=state.t)
        A_new = decay_over_budget(decayed_state, rng)

    return SimState(D=state.D, budgets=state.budgets, A=A_new, t=state.t + 1)
