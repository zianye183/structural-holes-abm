"""
Modular dynamics engine for the structural holes ABM.

Each mechanism is a function: (SimState, params) -> (N, N) factor matrix.
Mechanisms are multiplied together to form the tie formation probability P.
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
# Mechanisms — each returns an (N, N) factor matrix
# ---------------------------------------------------------------------------

def mechanism_homophily(state: SimState, lam: float) -> np.ndarray:
    """Homophily factor: P(i,j) = exp(-lam * d_ij).

    Args:
        state: current simulation state.
        lam: decay rate. Higher = stronger preference for similar agents.

    Returns:
        (N, N) factor matrix. Diagonal is zero.
    """
    factor = np.exp(-lam * state.D)
    np.fill_diagonal(factor, 0.0)
    return factor


def mechanism_triadic_closure(state: SimState, tau: float) -> np.ndarray:
    """Triadic closure factor: boost by tau per shared neighbor.

    shared_neighbors[i,k] = (A @ A)[i,k] = number of shared neighbors.
    Factor = tau ^ shared_neighbors.

    Args:
        state: current simulation state.
        tau: boost factor per shared neighbor. Must be >= 1.0.
             tau > 1 encourages closure; tau == 1 is neutral.

    Returns:
        (N, N) factor matrix. Values >= 1.0 (no penalty, only boost).

    Raises:
        ValueError: if tau < 1.0.
    """
    if tau < 1.0:
        raise ValueError(f"tau must be >= 1.0 (got {tau})")
    shared = state.A @ state.A
    np.fill_diagonal(shared, 0.0)
    return tau ** shared


def mechanism_attention_budget(state: SimState, beta: float) -> np.ndarray:
    """Attention budget factor: penalize tie formation when over capacity.

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
# Step function — compose mechanisms into one timestep
# ---------------------------------------------------------------------------

Mechanism = Callable[[SimState], np.ndarray]


def step(
    state: SimState,
    mechanisms: list[Mechanism],
    rng: np.random.Generator,
) -> SimState:
    """Execute one simulation timestep.

    1. Compute formation probability P by multiplying all mechanism factors.
    2. Mask out existing ties (only form NEW edges).
    3. Sample new edges from P.
    4. Apply tie decay for over-budget agents.
    5. Return new state with updated A and incremented t.

    Args:
        state: current simulation state.
        mechanisms: list of mechanism functions, each returns (N, N) factor.
        rng: random number generator.

    Returns:
        New SimState with updated adjacency and t+1.
    """
    n = state.n

    # Multiply all mechanism factors
    P = np.ones((n, n))
    for mech in mechanisms:
        P *= mech(state)

    # Mask existing ties and self-ties
    P *= (1 - state.A)
    np.fill_diagonal(P, 0.0)

    # Clamp to [0, 1]
    P = np.clip(P, 0.0, 1.0)

    # Sample new edges (upper triangle, then symmetrize)
    draws = rng.uniform(0, 1, size=(n, n))
    new_edges = np.triu(draws < P, k=1)
    new_edges = new_edges | new_edges.T

    # Combine with existing adjacency
    A_new = np.maximum(state.A, new_edges.astype(np.float64))

    # Apply decay
    decayed_state = SimState(D=state.D, budgets=state.budgets, A=A_new, t=state.t)
    A_final = decay_over_budget(decayed_state, rng)

    return SimState(D=state.D, budgets=state.budgets, A=A_final, t=state.t + 1)
