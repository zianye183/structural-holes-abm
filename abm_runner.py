"""
Simulation runner: executes the dynamics loop and records history.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from abm_core import InitResult
from abm_dynamics import SimState, Mechanism, step


@dataclass
class SimHistory:
    """Recorded simulation history.

    init_result: the initialization that produced D and viz_coords.
    params:      dict of simulation parameters for reproducibility.
    frames:      list of sparse CSR adjacency matrices, one per timestep.
    stats:       list of summary dicts, one per timestep.
    """
    init_result: InitResult
    params: dict
    frames: list[sparse.csr_matrix]
    stats: list[dict]

    def get_dense_frame(self, t: int) -> np.ndarray:
        """Return frame t as a dense (N, N) array."""
        return self.frames[t].toarray()


def run_simulation(
    init_result: InitResult,
    mechanisms: list[Mechanism],
    budgets: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> SimHistory:
    """Run the full simulation loop.

    Args:
        init_result: output of any init_* function from abm_core.
        mechanisms: list of mechanism functions to compose.
        budgets: (N,) attention budget per agent.
        n_steps: number of timesteps to simulate.
        rng: random number generator.

    Returns:
        SimHistory with sparse frames and stats for each timestep.
    """
    state = SimState(D=init_result.distance_matrix, budgets=budgets)

    frames = [sparse.csr_matrix(state.A)]
    stats = [_frame_stats(state)]

    for _ in range(n_steps):
        state = step(state, mechanisms, rng)
        frames.append(sparse.csr_matrix(state.A))
        stats.append(_frame_stats(state))

    return SimHistory(
        init_result=init_result,
        params={"n_steps": n_steps, "n": init_result.n},
        frames=frames,
        stats=stats,
    )


def _frame_stats(state: SimState) -> dict:
    """Compute summary statistics for a single frame."""
    degrees = state.degrees
    n_edges = int(state.A.sum()) // 2
    return {
        "t": state.t,
        "n_edges": n_edges,
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
    }
