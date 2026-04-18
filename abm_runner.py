"""
Simulation runner: executes the dynamics loop and records history.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy import sparse

from abm_core import InitResult, burt_constraint_decomposed
from abm_dynamics import SimState, Mechanism, Constraint, step


@dataclass
class SimHistory:
    """Recorded simulation history.

    init_result:  the initialization that produced D and viz_coords.
    params:       dict of simulation parameters for reproducibility.
    frames:       list of sparse CSR adjacency matrices, one per timestep.
    stats:        list of summary dicts, one per timestep.
    node_metrics: list of per-node metric dicts, one per timestep.
                  Each dict contains numpy arrays keyed by metric name.
    """
    init_result: InitResult
    params: dict
    frames: list[sparse.csr_matrix]
    stats: list[dict]
    node_metrics: list[dict] = field(default_factory=list)

    def get_dense_frame(self, t: int) -> np.ndarray:
        """Return frame t as a dense (N, N) array."""
        return self.frames[t].toarray()


def run_simulation(
    init_result: InitResult,
    mechanisms: list[Mechanism],
    budgets: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    intercept: float = -5.0,
    constraints: list[Constraint] | None = None,
    enable_decay: bool = True,
) -> SimHistory:
    """Run the full simulation loop.

    Args:
        init_result: output of any init_* function from abm_core.
        mechanisms: list of mechanism functions (log-odds contributors).
        budgets: (N,) attention budget per agent.
        n_steps: number of timesteps to simulate.
        rng: random number generator.
        intercept: base rate in log-odds space. Default -5.0.
        constraints: list of constraint functions (post-hoc masks).
        enable_decay: if True (default), drop excess ties each step.

    Returns:
        SimHistory with sparse frames, stats, and per-node metrics.
    """
    state = SimState(D=init_result.distance_matrix, budgets=budgets)

    frames = [sparse.csr_matrix(state.A)]
    stats_list = []
    metrics_list = []

    frame_stats, frame_metrics = _compute_frame(state)
    stats_list.append(frame_stats)
    metrics_list.append(frame_metrics)

    for _ in range(n_steps):
        state = step(state, mechanisms, rng, intercept=intercept,
                     constraints=constraints, enable_decay=enable_decay)
        frames.append(sparse.csr_matrix(state.A))
        frame_stats, frame_metrics = _compute_frame(state)
        stats_list.append(frame_stats)
        metrics_list.append(frame_metrics)

    return SimHistory(
        init_result=init_result,
        params={"n_steps": n_steps, "n": init_result.n},
        frames=frames,
        stats=stats_list,
        node_metrics=metrics_list,
    )


def _compute_frame(state: SimState) -> tuple[dict, dict]:
    """Compute summary stats and per-node metrics for a single frame.

    Returns:
        (stats_dict, node_metrics_dict)
    """
    degrees = state.degrees
    n_edges = int(state.A.sum()) // 2

    # Per-node constraint decomposition
    G = nx.from_numpy_array(state.A)
    decomp = burt_constraint_decomposed(G)

    # Aggregate stats (no NaNs — isolates get constraint=1.0)
    mean_constraint = float(decomp["constraint"].mean())
    mean_c_size = float(decomp["c_size"].mean())
    mean_c_density = float(decomp["c_density"].mean())
    mean_c_hierarchy = float(decomp["c_hierarchy"].mean())

    stats = {
        "t": state.t,
        "n_edges": n_edges,
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
        "mean_constraint": mean_constraint,
        "mean_c_size": mean_c_size,
        "mean_c_density": mean_c_density,
        "mean_c_hierarchy": mean_c_hierarchy,
    }

    node_metrics = {
        "degrees": degrees.copy(),
        "constraint": decomp["constraint"],
        "c_size": decomp["c_size"],
        "c_density": decomp["c_density"],
        "c_hierarchy": decomp["c_hierarchy"],
    }

    return stats, node_metrics
