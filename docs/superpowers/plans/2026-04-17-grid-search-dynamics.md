# Grid Search Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook that runs a 27-cell × 3-replicate grid search over the three dynamics mechanisms on a fixed 5-d torus, recording C_i and its three components every 100 steps for 500 steps.

**Architecture:** Extract a small `experiment_grid_search.py` module with reusable, unit-tested helpers (mechanism builder, snapshot extractor, summary statistics, snapshot saver, single-cell runner). The notebook `04_grid_search_dynamics.ipynb` only contains config, the run loop, sanity checks, and plots — all heavy lifting lives in existing modules (`abm_core`, `abm_dynamics`, `abm_runner`) plus the new helpers.

**Tech Stack:** Python 3, numpy, scipy.sparse, networkx, pandas (for the summary parquet), pyarrow (parquet engine), matplotlib, tqdm, pytest, jupyter.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `experiment_grid_search.py` | Create | Pure helpers: `build_mechanisms`, `extract_snapshot_metrics`, `summarize_metrics`, `save_snapshots`, `run_grid_cell`. No I/O besides `save_snapshots`. No plotting. |
| `tests/test_experiment_grid_search.py` | Create | Unit tests for each helper. |
| `04_grid_search_dynamics.ipynb` | Create | Notebook: config, grid loop, sanity check, histogram plots, trajectory plots. |
| `simulations/grid_search_torus_5d/runs/cell_*_rep*.npz` | Generated | Per-run snapshot outputs (per-node metric arrays + 6 sparse frames). |
| `simulations/grid_search_torus_5d/summary.parquet` | Generated | One row per (cell, replicate, snapshot_t) with 12 stat columns. |

---

## Task 1: `build_mechanisms` — filter zero-coefficient mechanisms

**Files:**
- Create: `experiment_grid_search.py`
- Create: `tests/test_experiment_grid_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
import numpy as np
import pytest
from abm_core import init_torus_uniform
from abm_dynamics import SimState

from experiment_grid_search import build_mechanisms


def _make_state(n=10, d=2, seed=0):
    rng = np.random.default_rng(seed)
    init = init_torus_uniform(n=n, d=d, rng=rng)
    return SimState(D=init.distance_matrix, budgets=np.full(n, 5))


def test_build_mechanisms_omits_zero_coefs():
    mechs = build_mechanisms(b_homophily=0.0, b_triadic=0.0, b_popularity=0.0)
    assert mechs == []


def test_build_mechanisms_keeps_positive_coefs():
    mechs = build_mechanisms(b_homophily=1.0, b_triadic=0.0, b_popularity=0.6)
    assert len(mechs) == 2


def test_build_mechanisms_callables_return_correct_shape():
    state = _make_state(n=10)
    mechs = build_mechanisms(b_homophily=2.0, b_triadic=1.0, b_popularity=0.5)
    for mech in mechs:
        out = mech(state)
        assert out.shape == (10, 10)
        assert out.dtype == np.float64


def test_build_mechanisms_late_binding_safe():
    # closures must bind the coefficient at definition time, not loop variable
    state = _make_state(n=8)
    mechs = build_mechanisms(b_homophily=3.0, b_triadic=1.5, b_popularity=0.6)
    # if late-binding bug, all three would use the same b — call them and check
    # outputs differ across mechanisms
    outs = [m(state) for m in mechs]
    assert not np.allclose(outs[0], outs[1])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "$(pwd)" && python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiment_grid_search'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiment_grid_search.py`:

```python
"""
Helpers for the dynamics grid-search experiment.

Pure functions plus a single-cell runner. No plotting, no notebook-only logic.
"""

from abm_dynamics import (
    Mechanism,
    mechanism_homophily,
    mechanism_triadic_closure,
    mechanism_popularity,
)


def build_mechanisms(
    b_homophily: float,
    b_triadic: float,
    b_popularity: float,
) -> list[Mechanism]:
    """Return mechanism closures for non-zero coefficients only.

    Mechanisms with coefficient 0 are omitted entirely (cleaner than
    contributing zero log-odds). Closures bind the coefficient at
    definition time via default-argument capture to avoid late-binding bugs.
    """
    mechs: list[Mechanism] = []
    if b_homophily > 0:
        mechs.append(lambda s, b=b_homophily: mechanism_homophily(s, b))
    if b_triadic > 0:
        mechs.append(lambda s, b=b_triadic: mechanism_triadic_closure(s, b))
    if b_popularity > 0:
        mechs.append(lambda s, b=b_popularity: mechanism_popularity(s, b))
    return mechs
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): add build_mechanisms helper for grid search"
```

---

## Task 2: `extract_snapshot_metrics` — pick out per-node arrays at snapshot times

**Files:**
- Modify: `experiment_grid_search.py`
- Modify: `tests/test_experiment_grid_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
from abm_dynamics import mechanism_homophily
from abm_runner import run_simulation

from experiment_grid_search import extract_snapshot_metrics


def _tiny_history(n_steps=10):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=15, d=2, rng=rng)
    return run_simulation(
        init_result=init,
        mechanisms=[lambda s: mechanism_homophily(s, b_homophily=4.0)],
        budgets=np.full(15, 5),
        n_steps=n_steps,
        rng=np.random.default_rng(1),
        intercept=-3.0,
    )


def test_extract_snapshot_metrics_returns_correct_times():
    history = _tiny_history(n_steps=10)
    times = [0, 5, 10]
    snap = extract_snapshot_metrics(history, times)
    assert list(snap["times"]) == times


def test_extract_snapshot_metrics_array_shapes():
    history = _tiny_history(n_steps=10)
    times = [0, 5, 10]
    snap = extract_snapshot_metrics(history, times)
    for key in ("constraint", "c_size", "c_density", "c_hierarchy"):
        assert snap[key].shape == (3, 15)


def test_extract_snapshot_metrics_values_match_history():
    history = _tiny_history(n_steps=10)
    snap = extract_snapshot_metrics(history, [0, 10])
    assert np.allclose(snap["constraint"][0], history.node_metrics[0]["constraint"])
    assert np.allclose(snap["constraint"][1], history.node_metrics[10]["constraint"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_experiment_grid_search.py::test_extract_snapshot_metrics_returns_correct_times -v
```

Expected: FAIL with `ImportError: cannot import name 'extract_snapshot_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiment_grid_search.py`:

```python
import numpy as np

from abm_runner import SimHistory


def extract_snapshot_metrics(
    history: SimHistory,
    snapshot_times: list[int],
) -> dict[str, np.ndarray]:
    """Stack per-node metric arrays at the requested timesteps.

    Args:
        history: completed simulation history (frames + node_metrics for every t).
        snapshot_times: list of integer timesteps (must be valid indices into history).

    Returns:
        dict with keys:
            times:        (T,) int array of the snapshot timesteps
            constraint:   (T, N) float array
            c_size:       (T, N) float array
            c_density:    (T, N) float array
            c_hierarchy:  (T, N) float array
    """
    out: dict[str, np.ndarray] = {
        "times": np.array(snapshot_times, dtype=np.int32),
    }
    for name in ("constraint", "c_size", "c_density", "c_hierarchy"):
        out[name] = np.array([history.node_metrics[t][name] for t in snapshot_times])
    return out
```

Add the import at the top if not already present:

```python
import numpy as np
from abm_runner import SimHistory
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): add extract_snapshot_metrics helper"
```

---

## Task 3: `summarize_metrics` — mean/median/std for one snapshot

**Files:**
- Modify: `experiment_grid_search.py`
- Modify: `tests/test_experiment_grid_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
from experiment_grid_search import summarize_metrics


def test_summarize_metrics_returns_12_columns():
    snap_for_one_t = {
        "constraint":  np.array([0.1, 0.2, 0.3, 0.4]),
        "c_size":      np.array([0.05, 0.1, 0.15, 0.2]),
        "c_density":   np.array([0.02, 0.04, 0.06, 0.08]),
        "c_hierarchy": np.array([0.03, 0.06, 0.09, 0.12]),
    }
    summary = summarize_metrics(snap_for_one_t)
    expected_keys = {
        "mean_constraint", "median_constraint", "std_constraint",
        "mean_c_size", "median_c_size", "std_c_size",
        "mean_c_density", "median_c_density", "std_c_density",
        "mean_c_hierarchy", "median_c_hierarchy", "std_c_hierarchy",
    }
    assert set(summary.keys()) == expected_keys


def test_summarize_metrics_correct_values():
    snap = {
        "constraint":  np.array([1.0, 2.0, 3.0, 4.0]),
        "c_size":      np.zeros(4),
        "c_density":   np.zeros(4),
        "c_hierarchy": np.zeros(4),
    }
    summary = summarize_metrics(snap)
    assert summary["mean_constraint"] == pytest.approx(2.5)
    assert summary["median_constraint"] == pytest.approx(2.5)
    assert summary["std_constraint"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))
    assert summary["mean_c_size"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_experiment_grid_search.py::test_summarize_metrics_returns_12_columns -v
```

Expected: FAIL with `ImportError: cannot import name 'summarize_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiment_grid_search.py`:

```python
def summarize_metrics(metrics_at_t: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute mean/median/std for each of the 4 metrics at one timestep.

    Args:
        metrics_at_t: dict with keys constraint, c_size, c_density, c_hierarchy,
                      each value a 1-D array of per-node values.

    Returns:
        dict with 12 float entries: {mean,median,std}_{constraint,c_size,c_density,c_hierarchy}.
    """
    out: dict[str, float] = {}
    for name in ("constraint", "c_size", "c_density", "c_hierarchy"):
        arr = metrics_at_t[name]
        out[f"mean_{name}"] = float(arr.mean())
        out[f"median_{name}"] = float(np.median(arr))
        out[f"std_{name}"] = float(arr.std())
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): add summarize_metrics helper"
```

---

## Task 4: `save_snapshots` — write per-run npz file

**Files:**
- Modify: `experiment_grid_search.py`
- Modify: `tests/test_experiment_grid_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
from pathlib import Path

from experiment_grid_search import save_snapshots


def test_save_snapshots_writes_npz(tmp_path):
    history = _tiny_history(n_steps=10)
    out = tmp_path / "snap.npz"
    save_snapshots(out, history, snapshot_times=[0, 5, 10])
    assert out.exists()
    data = np.load(out)
    assert list(data["times"]) == [0, 5, 10]
    assert data["constraint"].shape == (3, 15)
    assert data["n"].item() == 15
    # Sparse frames stored as concatenated triu COO with frame_ids in {0,1,2}
    assert set(data["frame_ids"].tolist()).issubset({0, 1, 2})


def test_save_snapshots_frames_roundtrip(tmp_path):
    history = _tiny_history(n_steps=10)
    out = tmp_path / "snap.npz"
    save_snapshots(out, history, snapshot_times=[0, 10])
    data = np.load(out)
    # Reconstruct frame at index 1 (t=10) from COO
    mask = data["frame_ids"] == 1
    n = int(data["n"].item())
    rebuilt = np.zeros((n, n), dtype=np.float64)
    rebuilt[data["rows"][mask], data["cols"][mask]] = 1.0
    rebuilt = rebuilt + rebuilt.T
    expected = history.frames[10].toarray()
    assert np.array_equal(rebuilt, expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_experiment_grid_search.py::test_save_snapshots_writes_npz -v
```

Expected: FAIL with `ImportError: cannot import name 'save_snapshots'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiment_grid_search.py`:

```python
from pathlib import Path

from scipy import sparse


def save_snapshots(
    out_path: Path | str,
    history: SimHistory,
    snapshot_times: list[int],
) -> None:
    """Persist per-node metrics + sparse frames at snapshot timesteps to a single npz.

    File layout:
        times       : (T,) int32  — snapshot timesteps
        constraint  : (T, N) float64
        c_size      : (T, N) float64
        c_density   : (T, N) float64
        c_hierarchy : (T, N) float64
        rows, cols  : concatenated triu indices for all T frames
        frame_ids   : (len(rows),) int32 — which snapshot each edge belongs to (0..T-1)
        n           : scalar — number of agents
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    snap = extract_snapshot_metrics(history, snapshot_times)

    rows_list, cols_list, fid_list = [], [], []
    for i, t in enumerate(snapshot_times):
        coo = sparse.triu(history.frames[t], k=1).tocoo()
        rows_list.append(coo.row.astype(np.int32))
        cols_list.append(coo.col.astype(np.int32))
        fid_list.append(np.full(len(coo.row), i, dtype=np.int32))

    np.savez_compressed(
        out,
        times=snap["times"],
        constraint=snap["constraint"],
        c_size=snap["c_size"],
        c_density=snap["c_density"],
        c_hierarchy=snap["c_hierarchy"],
        rows=np.concatenate(rows_list) if rows_list else np.array([], dtype=np.int32),
        cols=np.concatenate(cols_list) if cols_list else np.array([], dtype=np.int32),
        frame_ids=np.concatenate(fid_list) if fid_list else np.array([], dtype=np.int32),
        n=np.int32(history.init_result.n),
    )
```

Add to the imports section if not present:

```python
from pathlib import Path
from scipy import sparse
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): add save_snapshots for per-run persistence"
```

---

## Task 5: `run_grid_cell` — orchestrate one (cell × replicate) run

**Files:**
- Modify: `experiment_grid_search.py`
- Modify: `tests/test_experiment_grid_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
from experiment_grid_search import run_grid_cell


def test_run_grid_cell_writes_file_and_returns_rows(tmp_path):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=20, d=2, rng=rng)
    rows = run_grid_cell(
        init_result=init,
        b_homophily=2.0,
        b_triadic=0.5,
        b_popularity=0.0,
        budget=5,
        n_steps=20,
        snapshot_times=[0, 10, 20],
        sim_seed=42,
        out_path=tmp_path / "test_run.npz",
    )
    # File written
    assert (tmp_path / "test_run.npz").exists()
    # One row per snapshot time
    assert len(rows) == 3
    # Each row has the configured params + stats
    for i, row in enumerate(rows):
        assert row["b_homophily"] == 2.0
        assert row["b_triadic"] == 0.5
        assert row["b_popularity"] == 0.0
        assert row["t"] == [0, 10, 20][i]
        assert "mean_constraint" in row
        assert "std_c_hierarchy" in row


def test_run_grid_cell_reproducible(tmp_path):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=20, d=2, rng=rng)
    rows1 = run_grid_cell(
        init_result=init, b_homophily=2.0, b_triadic=0.0, b_popularity=0.0,
        budget=5, n_steps=10, snapshot_times=[0, 10], sim_seed=99,
        out_path=tmp_path / "a.npz",
    )
    rows2 = run_grid_cell(
        init_result=init, b_homophily=2.0, b_triadic=0.0, b_popularity=0.0,
        budget=5, n_steps=10, snapshot_times=[0, 10], sim_seed=99,
        out_path=tmp_path / "b.npz",
    )
    assert rows1[-1]["mean_constraint"] == rows2[-1]["mean_constraint"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_experiment_grid_search.py::test_run_grid_cell_writes_file_and_returns_rows -v
```

Expected: FAIL with `ImportError: cannot import name 'run_grid_cell'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiment_grid_search.py`:

```python
from abm_core import InitResult
from abm_dynamics import constraint_attention_hard
from abm_runner import run_simulation


def run_grid_cell(
    init_result: InitResult,
    b_homophily: float,
    b_triadic: float,
    b_popularity: float,
    budget: int,
    n_steps: int,
    snapshot_times: list[int],
    sim_seed: int,
    out_path: Path | str,
) -> list[dict]:
    """Run a single (cell, replicate) and persist its snapshots.

    Pipeline:
        1. Build the mechanism list (zero-coef mechanisms omitted).
        2. Apply constraint_attention_hard with uniform `budget`.
        3. Run the simulation for `n_steps` steps with a fresh RNG seeded
           from `sim_seed`.
        4. Save snapshots to `out_path`.
        5. Return one summary-row dict per snapshot timestep.

    The returned rows include the configured coefficients so they can be
    appended directly to a tidy DataFrame across the whole grid.
    """
    mechanisms = build_mechanisms(b_homophily, b_triadic, b_popularity)
    budgets = np.full(init_result.n, budget)
    rng = np.random.default_rng(sim_seed)
    history = run_simulation(
        init_result=init_result,
        mechanisms=mechanisms,
        budgets=budgets,
        n_steps=n_steps,
        rng=rng,
        constraints=[constraint_attention_hard],
        enable_decay=True,
    )
    save_snapshots(out_path, history, snapshot_times)

    rows: list[dict] = []
    for t in snapshot_times:
        row = {
            "b_homophily": b_homophily,
            "b_triadic": b_triadic,
            "b_popularity": b_popularity,
            "t": t,
            **summarize_metrics(history.node_metrics[t]),
        }
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): add run_grid_cell single-cell orchestrator"
```

---

## Task 6: Build the notebook

**Files:**
- Create: `04_grid_search_dynamics.ipynb`

- [ ] **Step 1: Create the notebook with the cells listed below**

Use `jupyter nbconvert` or write the JSON directly. The cells (in order) are:

**Cell 1 — Markdown header**

```markdown
# 04 — Grid Search of Dynamics

Holding geometry fixed (5d torus, uniform), vary the three forming-force
coefficients on a 3×3×3 grid, 3 replicates each = 81 runs × 500 steps.
Records constraint + 3 components every 100 steps.
```

**Cell 2 — Imports + config**

```python
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from abm_core import init_torus_uniform
from experiment_grid_search import run_grid_cell

# Fixed factors
N = 300
D = 5
BUDGET = 10
N_STEPS = 500
SNAPSHOT_TIMES = [0, 100, 200, 300, 400, 500]
REPLICATES = 3

# Varied factors
LEVELS = {
    "b_homophily":  [0.0, 1.0, 3.0],
    "b_triadic":    [0.0, 0.5, 1.5],
    "b_popularity": [0.0, 0.2, 0.6],
}

OUT_DIR = Path("simulations/grid_search_torus_5d")
RUNS_DIR = OUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "summary.parquet"
```

**Cell 3 — Build inits + cell list**

```python
# One init per replicate (block design): all cells share an init within a block,
# different blocks use different inits.
inits = [init_torus_uniform(n=N, d=D, rng=np.random.default_rng(seed))
         for seed in range(REPLICATES)]

cells = list(product(LEVELS["b_homophily"],
                     LEVELS["b_triadic"],
                     LEVELS["b_popularity"]))
print(f"{len(cells)} cells × {REPLICATES} replicates = {len(cells) * REPLICATES} runs")
```

**Cell 4 — Grid loop**

```python
all_rows = []
total = len(cells) * REPLICATES
with tqdm(total=total, desc="grid runs") as pbar:
    for cell_idx, (b_h, b_t, b_p) in enumerate(cells):
        for rep in range(REPLICATES):
            sim_seed = rep * 1000 + cell_idx
            out_path = RUNS_DIR / f"cell_{cell_idx:02d}_rep{rep}.npz"
            rows = run_grid_cell(
                init_result=inits[rep],
                b_homophily=b_h,
                b_triadic=b_t,
                b_popularity=b_p,
                budget=BUDGET,
                n_steps=N_STEPS,
                snapshot_times=SNAPSHOT_TIMES,
                sim_seed=sim_seed,
                out_path=out_path,
            )
            for row in rows:
                row["cell_id"] = cell_idx
                row["replicate"] = rep
            all_rows.extend(rows)
            pbar.update(1)

summary = pd.DataFrame(all_rows)
summary.to_parquet(SUMMARY_PATH)
print(f"Saved {len(summary)} rows → {SUMMARY_PATH}")
summary.head()
```

**Cell 5 — Sanity check**

```python
summary = pd.read_parquet(SUMMARY_PATH)

# Baseline cell: no mechanisms — almost all nodes isolated → mean_constraint ≈ 1
baseline = summary[
    (summary["b_homophily"] == 0)
    & (summary["b_triadic"] == 0)
    & (summary["b_popularity"] == 0)
    & (summary["t"] == N_STEPS)
]
print("Baseline (all-zero) at t=500, mean_constraint per replicate:")
print(baseline[["replicate", "mean_constraint", "std_constraint"]])
assert baseline["mean_constraint"].mean() > 0.9, "baseline should be near 1.0 (mostly isolated)"

# High homophily-only cell — should differ clearly
homophily_only = summary[
    (summary["b_homophily"] == 3.0)
    & (summary["b_triadic"] == 0)
    & (summary["b_popularity"] == 0)
    & (summary["t"] == N_STEPS)
]
print("\nHomophily-only at t=500:")
print(homophily_only[["replicate", "mean_constraint", "std_constraint"]])
```

**Cell 6 — Trajectory plots (mean ± std across replicates, colored by one mechanism)**

```python
def plot_trajectories(summary, color_by, metric="mean_constraint"):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = summary.groupby(["b_homophily", "b_triadic", "b_popularity", "t"])
    agg = grouped[metric].agg(["mean", "std"]).reset_index()
    for level, subset in agg.groupby(color_by):
        for cell_key, line in subset.groupby(
            [c for c in ("b_homophily", "b_triadic", "b_popularity") if c != color_by]
        ):
            ax.plot(line["t"], line["mean"], alpha=0.4)
        # bold mean across cells with this level for the legend
        bold = subset.groupby("t")["mean"].mean()
        ax.plot(bold.index, bold.values, label=f"{color_by}={level}", linewidth=2.5)
    ax.set_xlabel("t")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over time, colored by {color_by}")
    ax.legend()
    plt.tight_layout()
    plt.show()

for color_by in ("b_homophily", "b_triadic", "b_popularity"):
    plot_trajectories(summary, color_by, metric="mean_constraint")
```

**Cell 7 — Histogram plots (final snapshot, one per metric, faceted by one mechanism)**

```python
def plot_histograms_at_t(snapshot_t, metric, slice_by):
    """Pool per-node values across replicates, plot histogram per level of slice_by."""
    levels = sorted(LEVELS[slice_by])
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4), sharey=True)
    for ax, lvl in zip(axes, levels):
        # Pool across all cells with slice_by==lvl and across replicates
        cell_indices = [i for i, c in enumerate(cells)
                        if c[("b_homophily", "b_triadic", "b_popularity").index(slice_by)] == lvl]
        snap_idx = SNAPSHOT_TIMES.index(snapshot_t)
        all_vals = []
        for cell_idx in cell_indices:
            for rep in range(REPLICATES):
                data = np.load(RUNS_DIR / f"cell_{cell_idx:02d}_rep{rep}.npz")
                all_vals.append(data[metric][snap_idx])
        pooled = np.concatenate(all_vals)
        ax.hist(pooled, bins=40, alpha=0.7)
        ax.set_title(f"{slice_by}={lvl}")
        ax.set_xlabel(metric)
    fig.suptitle(f"{metric} histogram at t={snapshot_t}, sliced by {slice_by}")
    plt.tight_layout()
    plt.show()

for metric in ("constraint", "c_size", "c_density", "c_hierarchy"):
    plot_histograms_at_t(snapshot_t=N_STEPS, metric=metric, slice_by="b_homophily")
```

- [ ] **Step 2: Sanity-check the notebook JSON parses**

```bash
python -c "import json; json.load(open('04_grid_search_dynamics.ipynb'))"
```

Expected: no output (valid JSON).

- [ ] **Step 3: Commit the notebook structure (without executing)**

```bash
git add 04_grid_search_dynamics.ipynb
git commit -m "feat(notebook): scaffold 04_grid_search_dynamics"
```

---

## Task 7: Execute the notebook end-to-end and verify outputs

**Files:**
- Execute: `04_grid_search_dynamics.ipynb`
- Generates: `simulations/grid_search_torus_5d/runs/*.npz` (81 files), `summary.parquet`

- [ ] **Step 1: Execute the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace 04_grid_search_dynamics.ipynb --ExecutePreprocessor.timeout=1800
```

Expected: completes within ~5–15 min depending on machine; cells produce no errors.

- [ ] **Step 2: Verify outputs**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

OUT = Path("simulations/grid_search_torus_5d")
runs = sorted((OUT / "runs").glob("*.npz"))
assert len(runs) == 81, f"expected 81 run files, found {len(runs)}"

summary = pd.read_parquet(OUT / "summary.parquet")
# 27 cells × 3 reps × 6 snapshots
assert len(summary) == 27 * 3 * 6, f"expected 486 rows, got {len(summary)}"

cols = set(summary.columns)
required = {"cell_id", "replicate", "t", "b_homophily", "b_triadic", "b_popularity",
            "mean_constraint", "median_constraint", "std_constraint",
            "mean_c_size", "median_c_size", "std_c_size",
            "mean_c_density", "median_c_density", "std_c_density",
            "mean_c_hierarchy", "median_c_hierarchy", "std_c_hierarchy"}
missing = required - cols
assert not missing, f"missing columns: {missing}"
print("OK: 81 runs, 486 summary rows, all 18 columns present.")
PY
```

Expected: `OK: 81 runs, 486 summary rows, all 18 columns present.`

- [ ] **Step 3: Commit the executed notebook + outputs**

```bash
git add 04_grid_search_dynamics.ipynb
# Do NOT add simulations/ — it's already in .gitignore (verify with: git status simulations/)
git status simulations/  # should be empty (ignored) or only show untracked
git commit -m "feat(experiment): run grid search end-to-end"
```

(If `simulations/` is not gitignored and you don't want to check in ~50 MB, add it to `.gitignore` and amend.)

---

## Self-Review Checklist (the implementer should NOT do this; the planner did)

- [x] All 5 spec sections (setup, grid, replicates, recording, notebook structure) are covered by tasks
- [x] No placeholders ("TBD", "implement later") — every code step shows actual code
- [x] Type names consistent across tasks (`SimHistory`, `InitResult`, `Mechanism`)
- [x] Helper functions exist before the notebook task uses them (Tasks 1–5 then Task 6)
- [x] Test naming matches `tests/test_*.py` pytest convention
- [x] All commits use conventional-commit prefixes (feat/docs)
