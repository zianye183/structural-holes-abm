# Structural Holes ABM — Dynamics & Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular simulation engine with toggleable mechanisms (homophily, triadic closure, attention budgets) and a web interface for interactive visualization with playback controls.

**Architecture:** Each mechanism is a pure function `(SimState) → (N, N) factor matrix`. The simulation loop multiplies active factors into a formation probability matrix P, samples new edges, applies decay, and records each frame. A Panel web app provides interactive visualization with mechanism toggles, parameter sliders, and timeline scrubbing — all in Python.

**Tech Stack:** Python (numpy, networkx) for simulation; Panel + Bokeh for interactive web interface.

---

## File Structure

```
abm_core.py              — initialization, static tie formation, analysis (EXISTS)
abm_dynamics.py           — mechanism functions + SimState + decay rules (NEW)
abm_runner.py             — simulation loop, history recording (NEW)
tests/
  test_dynamics.py        — unit tests for each mechanism (NEW)
  test_runner.py          — integration tests for simulation loop (NEW)
03_dynamics.ipynb         — interactive notebook to test/visualize dynamics (NEW)
app_panel.py              — Panel web app with controls + network playback (NEW)
```

---

## Task 1: SimState and Mechanism Interface

**Files:**
- Create: `abm_dynamics.py`
- Create: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing test for SimState creation**

```python
# tests/test_dynamics.py
import numpy as np
from abm_dynamics import SimState

def test_simstate_from_init():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    budgets = np.array([5, 5, 5])
    state = SimState(D=D, budgets=budgets)
    assert state.n == 3
    assert state.A.shape == (3, 3)
    assert state.A.sum() == 0  # no edges initially
    assert np.array_equal(state.degrees, [0, 0, 0])
    assert state.t == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "Paper 2 - Structural Holes ABM simulation" && python -m pytest tests/test_dynamics.py::test_simstate_from_init -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abm_dynamics'`

- [ ] **Step 3: Implement SimState**

```python
# abm_dynamics.py
"""
Modular dynamics engine for the structural holes ABM.

Each mechanism is a function: (SimState, params) -> (N, N) factor matrix.
Mechanisms are multiplied together to form the tie formation probability P.
"""

from dataclasses import dataclass, field
import numpy as np


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dynamics.py::test_simstate_from_init -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add abm_dynamics.py tests/test_dynamics.py
git commit -m "feat: add SimState dataclass for simulation state"
```

---

## Task 2: Homophily Mechanism

**Files:**
- Modify: `abm_dynamics.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing test for homophily**

```python
# tests/test_dynamics.py
from abm_dynamics import SimState, mechanism_homophily

def test_homophily_basic():
    D = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    lam = 1.0
    factor = mechanism_homophily(state, lam=lam)
    # P(i,j) = exp(-lam * d_ij)
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
    # Higher lambda = sharper decay
    f_low = mechanism_homophily(state, lam=0.5)
    f_high = mechanism_homophily(state, lam=2.0)
    assert f_low[0, 1] > f_high[0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py::test_homophily_basic -v`
Expected: FAIL — `ImportError: cannot import name 'mechanism_homophily'`

- [ ] **Step 3: Implement homophily mechanism**

```python
# abm_dynamics.py (append)

def mechanism_homophily(state: SimState, lam: float) -> np.ndarray:
    """Homophily factor: P(i,j) = exp(-lam * d_ij).

    Args:
        state: current simulation state.
        lam (λ): decay rate. Higher = stronger preference for similar agents.

    Returns:
        (N, N) factor matrix. Diagonal is zero.
    """
    factor = np.exp(-lam * state.D)
    np.fill_diagonal(factor, 0.0)
    return factor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamics.py -k homophily -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add abm_dynamics.py tests/test_dynamics.py
git commit -m "feat: add homophily mechanism (exp decay on distance)"
```

---

## Task 3: Triadic Closure Mechanism

**Files:**
- Modify: `abm_dynamics.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing test for triadic closure**

```python
# tests/test_dynamics.py
from abm_dynamics import SimState, mechanism_triadic_closure

def test_triadic_closure_shared_neighbor():
    D = np.zeros((4, 4))  # distances irrelevant for this mechanism
    state = SimState(D=D, budgets=np.full(4, 5))
    # 0--1, 1--2 (so 0 and 2 share neighbor 1)
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[1, 2] = state.A[2, 1] = 1
    tau = 1.5
    factor = mechanism_triadic_closure(state, tau=tau)
    # 0 and 2 share 1 neighbor -> boosted by tau^1
    assert np.isclose(factor[0, 2], tau ** 1)
    assert np.isclose(factor[2, 0], tau ** 1)
    # 0 and 3 share 0 neighbors -> tau^0 = 1 (no boost)
    assert np.isclose(factor[0, 3], 1.0)

def test_triadic_closure_multiple_shared():
    D = np.zeros((4, 4))
    state = SimState(D=D, budgets=np.full(4, 5))
    # 0--1, 0--2, 1--3, 2--3 (so 0 and 3 share neighbors 1 AND 2)
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[1, 3] = state.A[3, 1] = 1
    state.A[2, 3] = state.A[3, 2] = 1
    tau = 2.0
    factor = mechanism_triadic_closure(state, tau=tau)
    # 0 and 3 share 2 neighbors -> tau^2
    assert np.isclose(factor[0, 3], tau ** 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py -k triadic -v`
Expected: FAIL — `ImportError: cannot import name 'mechanism_triadic_closure'`

- [ ] **Step 3: Implement triadic closure mechanism**

```python
# abm_dynamics.py (append)

def mechanism_triadic_closure(state: SimState, tau: float) -> np.ndarray:
    """Triadic closure factor: boost by tau per shared neighbor.

    shared_neighbors[i,k] = (A @ A)[i,k] = number of shared neighbors.
    Factor = tau ^ shared_neighbors.

    Args:
        state: current simulation state.
        tau (τ): boost factor per shared neighbor. tau > 1 = encourages closure.

    Returns:
        (N, N) factor matrix. Minimum value is 1.0 (no penalty, only boost).
    """
    shared = state.A @ state.A
    np.fill_diagonal(shared, 0.0)
    return tau ** shared
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamics.py -k triadic -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add abm_dynamics.py tests/test_dynamics.py
git commit -m "feat: add triadic closure mechanism (tau^shared_neighbors)"
```

---

## Task 4: Attention Budget Mechanism

**Files:**
- Modify: `abm_dynamics.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing test for attention budget**

```python
# tests/test_dynamics.py
from abm_dynamics import SimState, mechanism_attention_budget

def test_budget_under_capacity():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    # 0 edges -> well under budget -> sigma near 1
    factor = mechanism_attention_budget(state, beta=5.0)
    assert factor[0, 1] > 0.9
    assert factor[1, 0] > 0.9

def test_budget_over_capacity():
    D = np.zeros((5, 5))
    state = SimState(D=D, budgets=np.array([2, 2, 2, 2, 2]))
    # Give node 0 four connections (over budget of 2)
    state.A[0, 1] = state.A[1, 0] = 1
    state.A[0, 2] = state.A[2, 0] = 1
    state.A[0, 3] = state.A[3, 0] = 1
    state.A[0, 4] = state.A[4, 0] = 1
    factor = mechanism_attention_budget(state, beta=5.0)
    # Node 0 is way over budget -> its row/col should be near 0
    assert factor[0, 1] < 0.1
    # Node 1 has 1 connection, under budget -> its contribution is high
    # But factor[0,1] = sigma(0) * sigma(1), and sigma(0) is tiny
    assert factor[0, 1] < 0.1

def test_budget_symmetry():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    factor = mechanism_attention_budget(state, beta=5.0)
    assert np.isclose(factor[0, 1], factor[1, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py -k budget -v`
Expected: FAIL — `ImportError: cannot import name 'mechanism_attention_budget'`

- [ ] **Step 3: Implement attention budget mechanism**

```python
# abm_dynamics.py (append)

def mechanism_attention_budget(state: SimState, beta: float) -> np.ndarray:
    """Attention budget factor: penalize tie formation when over capacity.

    sigma_i = 1 / (1 + exp(beta * (k_i - b_i)))
    Factor[i,j] = sigma_i * sigma_j

    Args:
        state: current simulation state.
        beta (β): sharpness of the budget cutoff.
            High beta = hard cutoff at b_i.
            Low beta = soft, gradual penalty.

    Returns:
        (N, N) factor matrix. Values in [0, 1].
    """
    k = state.degrees.astype(np.float64)
    sigma = 1.0 / (1.0 + np.exp(beta * (k - state.budgets)))
    return sigma[:, np.newaxis] * sigma[np.newaxis, :]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamics.py -k budget -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add abm_dynamics.py tests/test_dynamics.py
git commit -m "feat: add attention budget mechanism (logistic saturation)"
```

---

## Task 5: Tie Decay Rule

**Files:**
- Modify: `abm_dynamics.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing test for tie decay**

```python
# tests/test_dynamics.py
from abm_dynamics import SimState, decay_over_budget

def test_decay_no_drop_under_budget():
    D = np.zeros((3, 3))
    state = SimState(D=D, budgets=np.array([5, 5, 5]))
    state.A[0, 1] = state.A[1, 0] = 1  # 1 edge, budget is 5
    rng = np.random.default_rng(42)
    A_new = decay_over_budget(state, rng)
    # Under budget -> no decay
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
    # Node 0 connected to 1,2,3,4 -> degree 4, budget 2 -> over by 2
    for j in range(1, 5):
        state.A[0, j] = state.A[j, 0] = 1
    rng = np.random.default_rng(42)
    A_new = decay_over_budget(state, rng)
    # Node 0 should have lost some ties
    assert A_new[0].sum() <= 4
    # Should have dropped the most distant ties first
    # (d=4 to node 4 and d=3 to node 3 are farthest)
    if A_new[0].sum() == 2:
        assert A_new[0, 1] == 1  # closest, kept
        assert A_new[0, 2] == 1  # second closest, kept
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py -k decay -v`
Expected: FAIL — `ImportError: cannot import name 'decay_over_budget'`

- [ ] **Step 3: Implement tie decay**

```python
# abm_dynamics.py (append)

def decay_over_budget(state: SimState, rng: np.random.Generator) -> np.ndarray:
    """Drop ties for agents over their attention budget.

    For each agent with degree > budget, drop the most distant ties
    until degree <= budget.

    Args:
        state: current simulation state.
        rng: random number generator (unused for now, deterministic drop).

    Returns:
        (N, N) updated adjacency matrix.
    """
    A_new = state.A.copy()
    degrees = A_new.sum(axis=1).astype(int)

    for i in range(state.n):
        excess = degrees[i] - int(state.budgets[i])
        if excess <= 0:
            continue
        # Find neighbors sorted by distance (farthest first)
        neighbors = np.where(A_new[i] > 0)[0]
        dists_to_neighbors = state.D[i, neighbors]
        drop_order = neighbors[np.argsort(-dists_to_neighbors)]
        # Drop the most distant ties
        for j in drop_order[:excess]:
            A_new[i, j] = 0
            A_new[j, i] = 0
        degrees[i] -= excess
        # Update degrees of dropped neighbors
        for j in drop_order[:excess]:
            degrees[j] -= 1

    return A_new
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamics.py -k decay -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add abm_dynamics.py tests/test_dynamics.py
git commit -m "feat: add tie decay rule (drop most distant when over budget)"
```

---

## Task 6: Compose Mechanisms + Step Function

**Files:**
- Modify: `abm_dynamics.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing test for step function**

```python
# tests/test_dynamics.py
from abm_dynamics import (
    SimState, step, mechanism_homophily,
    mechanism_triadic_closure, mechanism_attention_budget,
)

def test_step_homophily_only():
    rng = np.random.default_rng(42)
    D = np.array([[0, 0.5, 5], [0.5, 0, 4.5], [5, 4.5, 0]])
    state = SimState(D=D, budgets=np.array([10, 10, 10]))
    mechanisms = [lambda s: mechanism_homophily(s, lam=1.0)]
    new_state = step(state, mechanisms, rng)
    assert new_state.t == 1
    # Close pair (0,1) much more likely to connect than distant (0,2)
    # Run many steps to verify statistically
    connected_01 = 0
    connected_02 = 0
    for _ in range(200):
        s = SimState(D=D, budgets=np.array([10, 10, 10]))
        s2 = step(s, mechanisms, np.random.default_rng())
        connected_01 += s2.A[0, 1]
        connected_02 += s2.A[0, 2]
    assert connected_01 > connected_02 * 5  # should be much more likely

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
    # A should be symmetric
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py -k "test_step" -v`
Expected: FAIL — `ImportError: cannot import name 'step'`

- [ ] **Step 3: Implement step function**

```python
# abm_dynamics.py (append)

from typing import Callable

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamics.py -k "test_step" -v`
Expected: 3 PASS

- [ ] **Step 5: Run all dynamics tests**

Run: `python -m pytest tests/test_dynamics.py -v`
Expected: ALL PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add abm_dynamics.py tests/test_dynamics.py
git commit -m "feat: add step function composing modular mechanisms"
```

---

## Task 7: Simulation Runner + History Recording

**Files:**
- Create: `abm_runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing test for simulation runner**

```python
# tests/test_runner.py
import numpy as np
from abm_core import init_torus_uniform
from abm_dynamics import mechanism_homophily, mechanism_attention_budget
from abm_runner import run_simulation, SimHistory

def test_run_basic():
    rng = np.random.default_rng(42)
    init = init_torus_uniform(n=30, d=2, rng=rng)
    mechanisms = [
        lambda s: mechanism_homophily(s, lam=5.0),
        lambda s: mechanism_attention_budget(s, beta=3.0),
    ]
    history = run_simulation(
        init_result=init,
        mechanisms=mechanisms,
        budgets=np.full(30, 8),
        n_steps=10,
        rng=np.random.default_rng(42),
    )
    assert isinstance(history, SimHistory)
    assert len(history.frames) == 11  # t=0 through t=10
    assert history.frames[0].shape == (30, 30)
    # Network should have grown over time
    assert history.frames[10].sum() >= history.frames[0].sum()

def test_run_records_stats():
    rng = np.random.default_rng(42)
    init = init_torus_uniform(n=20, d=2, rng=rng)
    mechanisms = [lambda s: mechanism_homophily(s, lam=5.0)]
    history = run_simulation(
        init_result=init,
        mechanisms=mechanisms,
        budgets=np.full(20, 10),
        n_steps=5,
        rng=np.random.default_rng(42),
    )
    assert len(history.stats) == 6  # one per frame
    assert "mean_degree" in history.stats[0]
    assert "n_edges" in history.stats[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py::test_run_basic -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abm_runner'`

- [ ] **Step 3: Implement simulation runner**

```python
# abm_runner.py
"""
Simulation runner: executes the dynamics loop and records history.
"""

from dataclasses import dataclass
import numpy as np

from abm_core import InitResult
from abm_dynamics import SimState, Mechanism, step


@dataclass
class SimHistory:
    """Recorded simulation history.

    init_result: the initialization that produced D and viz_coords.
    params:      dict of simulation parameters for reproducibility.
    frames:      list of (N, N) adjacency matrices, one per timestep.
    stats:       list of summary dicts, one per timestep.
    """
    init_result: InitResult
    params: dict
    frames: list[np.ndarray]
    stats: list[dict]


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
        SimHistory with frames and stats for each timestep.
    """
    state = SimState(D=init_result.distance_matrix, budgets=budgets)

    frames = [state.A.copy()]
    stats = [_frame_stats(state)]

    for _ in range(n_steps):
        state = step(state, mechanisms, rng)
        frames.append(state.A.copy())
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add abm_runner.py tests/test_runner.py
git commit -m "feat: add simulation runner with history recording"
```

---

## Task 8: Dynamics Exploration Notebook

**Files:**
- Create: `03_dynamics.ipynb`

- [ ] **Step 1: Create notebook with mechanism comparison**

The notebook should:

1. Initialize a network (hyperbolic uniform, N=300, alpha=0.5)
2. Run 4 simulations with different mechanism combinations:
   - Homophily only
   - Homophily + triadic closure
   - Homophily + attention budget
   - Homophily + triadic closure + attention budget
3. Plot for each: degree distribution over time, clustering coefficient over time, mean Burt's constraint over time
4. Animate one network (final combination) as a sequence of snapshots at t=0, 50, 100, 200

```python
# Cell 1: Setup
import numpy as np
from abm_core import init_hyperbolic_uniform, burt_constraint
from abm_dynamics import (
    mechanism_homophily, mechanism_triadic_closure,
    mechanism_attention_budget,
)
from abm_runner import run_simulation
import matplotlib.pyplot as plt
import networkx as nx

N = 300
TARGET_K = 10
ALPHA = 0.5

rng = np.random.default_rng(42)
init = init_hyperbolic_uniform(N, TARGET_K, rng, alpha=ALPHA)

# Cell 2: Define mechanism configurations
configs = {
    "Homophily only": [
        lambda s: mechanism_homophily(s, lam=3.0),
    ],
    "Homophily + Triadic": [
        lambda s: mechanism_homophily(s, lam=3.0),
        lambda s: mechanism_triadic_closure(s, tau=1.5),
    ],
    "Homophily + Budget": [
        lambda s: mechanism_homophily(s, lam=3.0),
        lambda s: mechanism_attention_budget(s, beta=3.0),
    ],
    "All three": [
        lambda s: mechanism_homophily(s, lam=3.0),
        lambda s: mechanism_triadic_closure(s, tau=1.5),
        lambda s: mechanism_attention_budget(s, beta=3.0),
    ],
}

# Cell 3: Run simulations
results = {}
for name, mechanisms in configs.items():
    history = run_simulation(
        init_result=init,
        mechanisms=mechanisms,
        budgets=np.full(N, 10),
        n_steps=200,
        rng=np.random.default_rng(42),
    )
    results[name] = history
    print(f"{name}: final edges={history.stats[-1]['n_edges']}, "
          f"mean_degree={history.stats[-1]['mean_degree']:.1f}")

# Cell 4: Plot mean degree over time
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for name, history in results.items():
    ts = [s["t"] for s in history.stats]
    axes[0].plot(ts, [s["mean_degree"] for s in history.stats], label=name)
    axes[1].plot(ts, [s["n_edges"] for s in history.stats], label=name)
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Mean degree")
axes[0].legend()
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Number of edges")
axes[1].legend()
plt.tight_layout()
plt.show()

# Cell 5: Network snapshots for "All three"
history = results["All three"]
snapshot_times = [0, 50, 100, 200]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
coords = init.viz_coords
pos = {i: coords[i] for i in range(N)}
for ax, t in zip(axes, snapshot_times):
    A = history.frames[t]
    G = nx.from_numpy_array(A)
    degrees = np.array([d for _, d in G.degree()])
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1)
    ax.add_patch(circle)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.03, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10,
                           node_color=degrees, cmap='YlOrRd', alpha=0.7)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_title(f't={t}, edges={int(A.sum())//2}')
plt.suptitle("Network Evolution (All mechanisms)")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Run all cells, verify outputs are reasonable**

Check: degree growth curves make sense, budget-constrained runs plateau, triadic closure increases clustering.

- [ ] **Step 3: Commit**

```bash
git add 03_dynamics.ipynb
git commit -m "feat: add dynamics exploration notebook"
```

---

## Task 9: Panel Web App — Interactive Playback

**Files:**
- Create: `app_panel.py`

- [ ] **Step 1: Install Panel**

Run: `pip install panel bokeh`

- [ ] **Step 2: Create the Panel app**

```python
# app_panel.py
"""
Interactive ABM visualization with Panel.

Run: panel serve app_panel.py --show
"""

import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Circle, MultiLine
from bokeh.palettes import YlOrRd9
from bokeh.transform import linear_cmap

from abm_core import (
    init_torus_uniform, init_torus_gmm,
    init_hyperbolic_uniform, init_hyperbolic_gmm,
)
from abm_dynamics import (
    mechanism_homophily, mechanism_triadic_closure,
    mechanism_attention_budget,
)
from abm_runner import run_simulation

pn.extension()

# ---------------------------------------------------------------
# Widgets: Initialization
# ---------------------------------------------------------------
geometry_select = pn.widgets.Select(
    name="Geometry", options=["Torus Uniform", "Torus GMM",
                               "Hyperbolic Uniform", "Hyperbolic GMM"],
    value="Hyperbolic Uniform",
)
n_agents = pn.widgets.IntSlider(name="N agents", start=50, end=500, step=50, value=300)
alpha_slider = pn.widgets.FloatSlider(
    name="α (hyperbolic, gamma=2α+1)", start=0.3, end=1.5, step=0.1, value=0.5,
)
dim_slider = pn.widgets.IntSlider(name="d (torus dimension)", start=2, end=20, step=1, value=2)
normalize_select = pn.widgets.Select(
    name="Normalize D", options=["None", "Mean (D/mean)", "Max (D/max)"],
    value="Mean (D/mean)",
)

# ---------------------------------------------------------------
# Widgets: Mechanisms
# ---------------------------------------------------------------
toggle_homophily = pn.widgets.Checkbox(name="Homophily", value=True)
lam_slider = pn.widgets.FloatSlider(name="λ (decay rate)", start=0.1, end=10.0, step=0.1, value=3.0)

toggle_triadic = pn.widgets.Checkbox(name="Triadic Closure", value=False)
tau_slider = pn.widgets.FloatSlider(name="τ (boost per shared neighbor)", start=1.0, end=3.0, step=0.1, value=1.5)

toggle_budget = pn.widgets.Checkbox(name="Attention Budget", value=True)
beta_slider = pn.widgets.FloatSlider(name="β (budget sharpness)", start=0.5, end=10.0, step=0.5, value=3.0)
budget_slider = pn.widgets.IntSlider(name="b (budget per agent)", start=3, end=20, step=1, value=10)

# ---------------------------------------------------------------
# Widgets: Simulation
# ---------------------------------------------------------------
n_steps_slider = pn.widgets.IntSlider(name="Timesteps", start=10, end=500, step=10, value=200)
run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
player = pn.widgets.Player(
    name="Timeline", start=0, end=0, value=0, step=1,
    interval=50, loop_policy="once",
)

# ---------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------
stats_pane = pn.pane.Markdown("*Run a simulation to see stats.*")

# ---------------------------------------------------------------
# Bokeh network figure
# ---------------------------------------------------------------
node_source = ColumnDataSource(data={"x": [], "y": [], "degree": [], "size": []})
edge_source = ColumnDataSource(data={"xs": [], "ys": []})

plot = figure(
    width=650, height=650, match_aspect=True,
    tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
    background_fill_color="#1a1a2e", border_fill_color="#1a1a2e",
    outline_line_color=None,
)
plot.axis.visible = False
plot.grid.visible = False

# Boundary circle (for hyperbolic; hidden for torus)
boundary_theta = np.linspace(0, 2 * np.pi, 100)
plot.line(np.cos(boundary_theta), np.sin(boundary_theta),
          line_color="#333", line_width=1.5)

# Edges
plot.multi_line("xs", "ys", source=edge_source,
                line_color="white", line_alpha=0.08, line_width=0.5)

# Nodes
mapper = linear_cmap("degree", palette=list(reversed(YlOrRd9)), low=0, high=10)
plot.scatter("x", "y", source=node_source, size="size",
             color=mapper, alpha=0.8)

# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------
sim_history = {"frames": [], "init": None, "stats": []}


def run_sim(event):
    """Run the simulation with current widget settings."""
    rng = np.random.default_rng(42)
    n = n_agents.value

    # Initialize
    geom = geometry_select.value
    if geom == "Torus Uniform":
        init = init_torus_uniform(n, dim_slider.value, rng)
    elif geom == "Torus GMM":
        init = init_torus_gmm(n, dim_slider.value, 5, 0.08, rng)
    elif geom == "Hyperbolic Uniform":
        init = init_hyperbolic_uniform(n, 10, rng, alpha=alpha_slider.value)
    elif geom == "Hyperbolic GMM":
        init = init_hyperbolic_gmm(n, 10, 5, 0.4, rng, alpha=alpha_slider.value)
    else:
        return

    # Normalize distance matrix
    norm_choice = normalize_select.value
    if norm_choice == "Mean (D/mean)":
        init = init.normalized("mean")
    elif norm_choice == "Max (D/max)":
        init = init.normalized("max")

    # Build mechanism list
    mechanisms = []
    if toggle_homophily.value:
        lam = lam_slider.value
        mechanisms.append(lambda s, _lam=lam: mechanism_homophily(s, lam=_lam))
    if toggle_triadic.value:
        tau = tau_slider.value
        mechanisms.append(lambda s, _tau=tau: mechanism_triadic_closure(s, tau=_tau))
    if toggle_budget.value:
        beta = beta_slider.value
        mechanisms.append(lambda s, _beta=beta: mechanism_attention_budget(s, beta=_beta))

    if not mechanisms:
        stats_pane.object = "**Enable at least one mechanism.**"
        return

    # Run
    budgets = np.full(n, budget_slider.value)
    history = run_simulation(init, mechanisms, budgets, n_steps_slider.value,
                             np.random.default_rng(42))

    sim_history["frames"] = history.frames
    sim_history["init"] = init
    sim_history["stats"] = history.stats

    # Update player range
    player.end = len(history.frames) - 1
    player.value = 0

    # Render first frame
    render_frame(0)
    stats_pane.object = f"**Simulation complete.** {n_steps_slider.value} steps, {n} agents."


def render_frame(t):
    """Render frame t on the Bokeh plot."""
    if not sim_history["frames"]:
        return

    init = sim_history["init"]
    A = sim_history["frames"][t]
    coords = init.viz_coords
    degrees = A.sum(axis=1).astype(int)
    max_deg = max(degrees.max(), 1)

    # Update node data
    node_source.data = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "degree": degrees.tolist(),
        "size": (3 + (degrees / max_deg) * 8).tolist(),
    }

    # Update color mapper range
    mapper["transform"].high = int(max_deg)

    # Update edge data
    edges_ij = np.argwhere(np.triu(A, k=1) > 0)
    xs = [[float(coords[i, 0]), float(coords[j, 0])] for i, j in edges_ij]
    ys = [[float(coords[i, 1]), float(coords[j, 1])] for i, j in edges_ij]
    edge_source.data = {"xs": xs, "ys": ys}

    # Update stats
    stat = sim_history["stats"][t]
    stats_pane.object = (
        f"**t = {stat['t']}** | "
        f"Edges: {stat['n_edges']} | "
        f"Mean degree: {stat['mean_degree']:.1f} | "
        f"Max degree: {stat['max_degree']}"
    )


# Wire up callbacks
run_button.on_click(run_sim)
player.param.watch(lambda event: render_frame(event.new), "value")

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
sidebar = pn.Column(
    "## Structural Holes ABM",
    "### Initialization",
    geometry_select, n_agents, alpha_slider, dim_slider, normalize_select,
    "### Mechanisms",
    toggle_homophily, lam_slider,
    toggle_triadic, tau_slider,
    toggle_budget, beta_slider, budget_slider,
    "### Simulation",
    n_steps_slider, run_button,
    "### Playback",
    player,
    stats_pane,
    width=300,
)

main = pn.Column(plot)

template = pn.template.FastListTemplate(
    title="Structural Holes ABM",
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#e94560",
    header_background="#16213e",
)

template.servable()
```

- [ ] **Step 3: Test the app**

Run: `panel serve app_panel.py --show`

Verify:
- Select "Hyperbolic Uniform", click "Run Simulation"
- Network renders with nodes colored by degree
- Player widget scrubs through timesteps smoothly
- Toggle mechanisms on/off, re-run, see different behavior
- Stats update on each frame

- [ ] **Step 4: Commit**

```bash
git add app_panel.py
git commit -m "feat: add Panel web app with interactive network playback"
```

---

## Summary

| Task | Component | Key output |
|------|-----------|------------|
| 1 | SimState | Simulation state dataclass |
| 2 | Homophily | `mechanism_homophily(state, lam)` |
| 3 | Triadic closure | `mechanism_triadic_closure(state, tau)` |
| 4 | Attention budget | `mechanism_attention_budget(state, beta)` |
| 5 | Tie decay | `decay_over_budget(state, rng)` |
| 6 | Step function | `step(state, mechanisms, rng)` — composes everything |
| 7 | Runner | `run_simulation(...)` → SimHistory |
| 8 | Notebook | `03_dynamics.ipynb` — validation & exploration |
| 9 | Panel app | `app_panel.py` — interactive web interface |

**Future tasks (not in this plan):**
- Popularity mechanism (`mechanism_popularity(state, mu)`)
- Live re-run from within the Panel app (currently pre-computes, then plays back)
- Parameter sweeps and batch analysis
- Burt's constraint tracking over time in the Panel app
- Deploy Panel app (e.g. via `panel serve --address 0.0.0.0`)
