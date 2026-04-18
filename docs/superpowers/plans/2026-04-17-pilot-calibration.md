# Pilot Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 90-run pilot across six stages (A–F) that fixes concrete values for `LEVELS`, `budget`, `intercept`, `n_steps`, `sim_seed_scheme`, and `geometry_strategy` for the next factorial experiment.

**Design reference:** `docs/superpowers/specs/2026-04-17-pilot-calibration-design.md` — all stage definitions, success criteria, and fallbacks live there. This plan only covers *implementation*.

**Architecture:** Extend the existing `experiment_grid_search.py` helpers minimally (add p10 / p90 / frac statistics to `summarize_metrics`, expose `intercept` through `run_grid_cell`). Build a new notebook `05_pilot_calibration.ipynb` that imports the helpers, runs each stage, and ends with a Decisions cell. Reuse `abm_core.init_torus_uniform` (with `d=2` or `d=5`) and `abm_core.init_hyperbolic_uniform` for Stage F — no new init functions needed.

**Tech Stack:** Python 3, numpy, scipy.sparse, networkx, pandas, pyarrow, matplotlib, tqdm, joblib, pytest, jupyter.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `experiment_grid_search.py` | Modify | Extend `summarize_metrics` (add p10, p90, frac_C_lt_0.1); expose `intercept` in `run_grid_cell`. |
| `tests/test_experiment_grid_search.py` | Modify | Add tests for the new statistics and the new `intercept` parameter. |
| `05_pilot_calibration.ipynb` | Create | Notebook: Config, Stages A–F, Decisions cell. |
| `simulations/pilot/runs/stage{A,B,C,D,E,F}_*.npz` | Generated | Per-run snapshot outputs. |
| `simulations/pilot/pilot_summary.parquet` | Generated | One row per (stage, geometry, cell, rep, t) with all metrics. |

---

## Task 1: Extend `summarize_metrics` with tail statistics

**Files:**
- Modify: `experiment_grid_search.py`
- Modify: `tests/test_experiment_grid_search.py`

Current `summarize_metrics` returns mean/median/std for each of the four metrics. The pilot's primary DV is `p10_constraint` and it also needs `p90_constraint` and `frac_C_lt_0.1` per snapshot.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
def test_summarize_metrics_includes_tail_stats():
    snap = {
        "constraint":  np.linspace(0.0, 1.0, 101),  # p10=0.10, p90=0.90, median=0.50
        "c_size":      np.zeros(101),
        "c_density":   np.zeros(101),
        "c_hierarchy": np.zeros(101),
    }
    summary = summarize_metrics(snap)
    assert summary["p10_constraint"] == pytest.approx(0.10)
    assert summary["p90_constraint"] == pytest.approx(0.90)
    assert summary["frac_constraint_lt_0.1"] == pytest.approx(10 / 101)


def test_summarize_metrics_frac_threshold_boundary():
    snap = {
        "constraint":  np.array([0.05, 0.09, 0.10, 0.11, 0.50]),
        "c_size":      np.zeros(5),
        "c_density":   np.zeros(5),
        "c_hierarchy": np.zeros(5),
    }
    summary = summarize_metrics(snap)
    # strict less-than: 0.05 and 0.09 qualify; 0.10 does not
    assert summary["frac_constraint_lt_0.1"] == pytest.approx(2 / 5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_experiment_grid_search.py::test_summarize_metrics_includes_tail_stats -v
```

Expected: FAIL with `KeyError: 'p10_constraint'`.

- [ ] **Step 3: Write minimal implementation**

Modify `summarize_metrics` in `experiment_grid_search.py`:

```python
def summarize_metrics(metrics_at_t: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute mean/median/std/p10/p90 for each of the 4 metrics at one timestep,
    plus the fraction of agents with constraint < 0.1 (brokerage headcount).

    Args:
        metrics_at_t: dict with keys constraint, c_size, c_density, c_hierarchy,
                      each value a 1-D array of per-node values.

    Returns:
        dict with:
          - mean/median/std/p10/p90 for each of the 4 metrics (20 entries)
          - frac_constraint_lt_0.1 (fraction of agents with constraint < 0.1)
    """
    out: dict[str, float] = {}
    for name in ("constraint", "c_size", "c_density", "c_hierarchy"):
        arr = metrics_at_t[name]
        out[f"mean_{name}"] = float(arr.mean())
        out[f"median_{name}"] = float(np.median(arr))
        out[f"std_{name}"] = float(arr.std())
        out[f"p10_{name}"] = float(np.quantile(arr, 0.10))
        out[f"p90_{name}"] = float(np.quantile(arr, 0.90))
    out["frac_constraint_lt_0.1"] = float((metrics_at_t["constraint"] < 0.1).mean())
    return out
```

- [ ] **Step 4: Run all tests to verify nothing broke**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: all previously-passing tests still pass plus the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): add p10/p90/frac_lt_0.1 to summarize_metrics"
```

---

## Task 2: Expose `intercept` parameter in `run_grid_cell`

**Files:**
- Modify: `experiment_grid_search.py`
- Modify: `tests/test_experiment_grid_search.py`

Stage C of the pilot sweeps `intercept ∈ {−7, −5, −3, −1}`. The current `run_grid_cell` calls `run_simulation` without passing an `intercept`, so it uses the default of −5.0.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_grid_search.py`:

```python
def test_run_grid_cell_intercept_changes_dynamics(tmp_path):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=25, d=2, rng=rng)
    rows_neg7 = run_grid_cell(
        init_result=init, b_homophily=1.0, b_triadic=0.0, b_popularity=0.0,
        budget=5, n_steps=20, snapshot_times=[0, 20], sim_seed=1,
        out_path=tmp_path / "neg7.npz", intercept=-7.0,
    )
    rows_neg1 = run_grid_cell(
        init_result=init, b_homophily=1.0, b_triadic=0.0, b_popularity=0.0,
        budget=5, n_steps=20, snapshot_times=[0, 20], sim_seed=1,
        out_path=tmp_path / "neg1.npz", intercept=-1.0,
    )
    # Higher intercept → more ties → lower constraint
    assert rows_neg1[-1]["mean_constraint"] < rows_neg7[-1]["mean_constraint"]


def test_run_grid_cell_intercept_default_unchanged(tmp_path):
    """Default intercept behavior must remain -5.0 for back-compat with grid search."""
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=25, d=2, rng=rng)
    rows_default = run_grid_cell(
        init_result=init, b_homophily=1.0, b_triadic=0.0, b_popularity=0.0,
        budget=5, n_steps=20, snapshot_times=[0, 20], sim_seed=1,
        out_path=tmp_path / "default.npz",
    )
    rows_explicit = run_grid_cell(
        init_result=init, b_homophily=1.0, b_triadic=0.0, b_popularity=0.0,
        budget=5, n_steps=20, snapshot_times=[0, 20], sim_seed=1,
        out_path=tmp_path / "explicit.npz", intercept=-5.0,
    )
    assert rows_default[-1]["mean_constraint"] == rows_explicit[-1]["mean_constraint"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_experiment_grid_search.py::test_run_grid_cell_intercept_changes_dynamics -v
```

Expected: FAIL with `TypeError: run_grid_cell() got an unexpected keyword argument 'intercept'`.

- [ ] **Step 3: Write minimal implementation**

Modify `run_grid_cell` signature and body in `experiment_grid_search.py`:

```python
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
    intercept: float = -5.0,
) -> list[dict]:
    """Run a single (cell, replicate) and persist its snapshots.

    Args:
        ... (existing args) ...
        intercept: base log-odds. Default -5.0 matches the grid search experiment.

    Pipeline unchanged except `intercept` is now threaded through to `run_simulation`.
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
        intercept=intercept,
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
            "intercept": intercept,
            "t": t,
            **summarize_metrics(history.node_metrics[t]),
        }
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run all tests to verify nothing broke**

```bash
python -m pytest tests/test_experiment_grid_search.py -v
```

Expected: all existing tests still pass plus the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add experiment_grid_search.py tests/test_experiment_grid_search.py
git commit -m "feat(experiment): expose intercept parameter in run_grid_cell"
```

---

## Task 3: Scaffold `05_pilot_calibration.ipynb` (all stages, no execution)

**Files:**
- Create: `05_pilot_calibration.ipynb`

The notebook is a single file with a consistent structure: Config → Stage A → B → C → D → E → F → Decisions. Stages B, C, E need values chosen from earlier stages; these are marked `# TODO: fill from Stage X results` in the scaffold and resolved during Task 4.

- [ ] **Step 1: Create the notebook**

Use `jupyter nbconvert` or write the JSON directly. Cells in order:

**Cell 1 — Markdown header**

```markdown
# 05 — Pilot Calibration

Produces concrete settings (`LEVELS`, `budget`, `intercept`, `n_steps`,
`sim_seed_scheme`, `geometry_strategy`) for the next factorial experiment.

Spec: `docs/superpowers/specs/2026-04-17-pilot-calibration-design.md`.

Stages: A (1D scans on torus-5d) → B (budget) → C (intercept) → D (equilibrium)
→ E (CRN) → F (alt geometries) → Decisions.
```

**Cell 2 — Imports + shared config**

```python
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from abm_core import init_torus_uniform, init_hyperbolic_uniform
from experiment_grid_search import run_grid_cell

# --- Fixed factors ---
N = 300
BUDGET_DEFAULT = 10
INTERCEPT_DEFAULT = -5.0
N_STEPS_DEFAULT = 200
SNAPSHOT_TIMES_DEFAULT = [0, 50, 100, 150, 200]

OUT_DIR = Path("simulations/pilot")
RUNS_DIR = OUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "pilot_summary.parquet"

all_rows: list[dict] = []  # accumulated across stages
```

**Cell 3 — Stage A: 1D marginal scans on torus-5d**

```python
SCANS = {
    "b_homophily":  [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "b_triadic":    [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "b_popularity": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
}

def stage_a_cell(scan_name, level):
    coefs = {"b_homophily": 0.0, "b_triadic": 0.0, "b_popularity": 0.0, scan_name: level}
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=N, d=5, rng=rng)
    out_path = RUNS_DIR / f"stageA_torus5d_{scan_name}_{level:g}.npz"
    rows = run_grid_cell(
        init_result=init,
        b_homophily=coefs["b_homophily"],
        b_triadic=coefs["b_triadic"],
        b_popularity=coefs["b_popularity"],
        budget=BUDGET_DEFAULT,
        n_steps=N_STEPS_DEFAULT,
        snapshot_times=SNAPSHOT_TIMES_DEFAULT,
        sim_seed=1000,
        out_path=out_path,
    )
    for r in rows:
        r.update(stage="A", geometry="torus5d", scan=scan_name, level=level, rep=0)
    return rows

jobs = [(scan, level) for scan, levels in SCANS.items() for level in levels]
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(stage_a_cell)(s, l) for s, l in tqdm(jobs, desc="Stage A")
)
for group in results:
    all_rows.extend(group)

# Plot: p10_constraint vs level, per scan
df_a = pd.DataFrame(all_rows)
df_a_end = df_a[(df_a["stage"] == "A") & (df_a["t"] == N_STEPS_DEFAULT)]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, scan in zip(axes, SCANS):
    sub = df_a_end[df_a_end["scan"] == scan].sort_values("level")
    ax.plot(sub["level"], sub["p10_constraint"], marker="o", label="p10")
    ax.plot(sub["level"], sub["mean_constraint"], marker="s", alpha=0.6, label="mean")
    ax.set_xlabel(scan); ax.set_ylabel("constraint"); ax.set_title(f"Stage A: {scan}")
    ax.legend()
plt.tight_layout(); plt.show()
```

**Cell 4 — Markdown: "Pick Stage-A levels" checkpoint**

```markdown
## Checkpoint: choose Stage-A mid-active level per mechanism

Inspect the plots above. For each of `b_homophily`, `b_triadic`, `b_popularity`,
pick the **slope** level (not the flat, not past the knee). Record them below.
Stages B, C, E use the all-slope setting as the "active" configuration.
```

**Cell 5 — Stage B: budget scan (depends on Stage-A choices)**

```python
# TODO: fill from Stage A plots
MID_ACTIVE = {"b_homophily": 1.0, "b_triadic": 0.5, "b_popularity": 0.25}

BUDGETS = [5, 8, 12, 20, 30, 10_000]  # 10_000 ≈ no cap for N=300

def stage_b_cell(budget):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=N, d=5, rng=rng)
    out_path = RUNS_DIR / f"stageB_budget_{budget}.npz"
    rows = run_grid_cell(
        init_result=init,
        **MID_ACTIVE,
        budget=budget,
        n_steps=N_STEPS_DEFAULT,
        snapshot_times=SNAPSHOT_TIMES_DEFAULT,
        sim_seed=2000,
        out_path=out_path,
    )
    for r in rows:
        r.update(stage="B", geometry="torus5d", scan="budget", level=budget, rep=0)
    return rows

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(stage_b_cell)(b) for b in tqdm(BUDGETS, desc="Stage B")
)
for group in results:
    all_rows.extend(group)

df_b = pd.DataFrame(all_rows)
df_b_end = df_b[(df_b["stage"] == "B") & (df_b["t"] == N_STEPS_DEFAULT)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_b_end["level"], df_b_end["std_c_size"], marker="o")
ax.set_xscale("log"); ax.set_xlabel("budget"); ax.set_ylabel("std_c_size")
ax.set_title("Stage B: c_size variance vs budget")
plt.tight_layout(); plt.show()
```

**Cell 6 — Stage C: intercept scan**

```python
INTERCEPTS = [-7.0, -5.0, -3.0, -1.0]

def stage_c_cell(intercept):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=N, d=5, rng=rng)
    out_path = RUNS_DIR / f"stageC_intercept_{intercept:g}.npz"
    rows = run_grid_cell(
        init_result=init,
        **MID_ACTIVE,
        budget=BUDGET_DEFAULT,
        n_steps=N_STEPS_DEFAULT,
        snapshot_times=SNAPSHOT_TIMES_DEFAULT,
        sim_seed=3000,
        out_path=out_path,
        intercept=intercept,
    )
    for r in rows:
        r.update(stage="C", geometry="torus5d", scan="intercept", level=intercept, rep=0)
    return rows

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(stage_c_cell)(i) for i in tqdm(INTERCEPTS, desc="Stage C")
)
for group in results:
    all_rows.extend(group)
```

**Cell 7 — Stage D: equilibrium check**

```python
EQ_SETTINGS = {
    "baseline":   {"b_homophily": 0.0, "b_triadic": 0.0, "b_popularity": 0.0},
    "mid_active": MID_ACTIVE,
    # TODO: fill from Stage A plots with the knee levels
    "all_high":   {"b_homophily": 4.0, "b_triadic": 2.0, "b_popularity": 1.0},
}
EQ_STEPS = 300
EQ_SNAPSHOTS = list(range(0, EQ_STEPS + 1, 10))

def stage_d_cell(name, coefs):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=N, d=5, rng=rng)
    out_path = RUNS_DIR / f"stageD_{name}.npz"
    rows = run_grid_cell(
        init_result=init, **coefs,
        budget=BUDGET_DEFAULT,
        n_steps=EQ_STEPS,
        snapshot_times=EQ_SNAPSHOTS,
        sim_seed=4000,
        out_path=out_path,
    )
    for r in rows:
        r.update(stage="D", geometry="torus5d", scan=name, level=0, rep=0)
    return rows

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(stage_d_cell)(name, coefs) for name, coefs in tqdm(EQ_SETTINGS.items(), desc="Stage D")
)
for group in results:
    all_rows.extend(group)

df_d = pd.DataFrame(all_rows)
df_d = df_d[df_d["stage"] == "D"]
fig, ax = plt.subplots(figsize=(7, 4))
for name, sub in df_d.groupby("scan"):
    sub = sub.sort_values("t")
    ax.plot(sub["t"], sub["p10_constraint"], marker=".", label=name)
ax.set_xlabel("t"); ax.set_ylabel("p10_constraint"); ax.set_title("Stage D: equilibrium")
ax.legend(); plt.tight_layout(); plt.show()
```

**Cell 8 — Stage E: common random numbers**

```python
CRN_CELLS = {
    "A": {"b_homophily": MID_ACTIVE["b_homophily"], "b_triadic": 0.0, "b_popularity": 0.0},
    "B": {"b_homophily": 2 * MID_ACTIVE["b_homophily"], "b_triadic": 0.0, "b_popularity": 0.0},
}
N_REPS = 5

def stage_e_cell(cell_name, coefs, rep, regime):
    # matched: same seed across cells, differs by rep
    # unmatched: seed differs by (cell, rep)
    if regime == "matched":
        sim_seed = 5000 + rep
    else:
        sim_seed = 5000 + rep * 100 + (1 if cell_name == "B" else 0)
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=N, d=5, rng=rng)
    out_path = RUNS_DIR / f"stageE_{regime}_cell{cell_name}_rep{rep}.npz"
    rows = run_grid_cell(
        init_result=init, **coefs,
        budget=BUDGET_DEFAULT,
        n_steps=N_STEPS_DEFAULT,
        snapshot_times=SNAPSHOT_TIMES_DEFAULT,
        sim_seed=sim_seed,
        out_path=out_path,
    )
    for r in rows:
        r.update(stage="E", geometry="torus5d", scan=f"{regime}_cell{cell_name}", level=rep, rep=rep)
    return rows

jobs = [(cn, coefs, rep, regime)
        for regime in ("matched", "unmatched")
        for cn, coefs in CRN_CELLS.items()
        for rep in range(N_REPS)]
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(stage_e_cell)(*j) for j in tqdm(jobs, desc="Stage E")
)
for group in results:
    all_rows.extend(group)

# Compute paired-contrast std per regime
df_e = pd.DataFrame(all_rows)
df_e = df_e[(df_e["stage"] == "E") & (df_e["t"] == N_STEPS_DEFAULT)]
for regime in ("matched", "unmatched"):
    a = df_e[df_e["scan"] == f"{regime}_cellA"].sort_values("rep")["p10_constraint"].values
    b = df_e[df_e["scan"] == f"{regime}_cellB"].sort_values("rep")["p10_constraint"].values
    print(f"{regime:>10s}: std(p10_A - p10_B) = {(a - b).std():.4f}")
```

**Cell 9 — Stage F: alt geometries (Poincaré + torus-2d)**

```python
def make_init(geometry):
    rng = np.random.default_rng(0)
    if geometry == "torus5d":
        return init_torus_uniform(n=N, d=5, rng=rng)
    if geometry == "torus2d":
        return init_torus_uniform(n=N, d=2, rng=rng)
    if geometry == "poincare":
        return init_hyperbolic_uniform(n=N, spread=1.0, rng=rng)
    raise ValueError(geometry)

def stage_f_cell(geometry, scan_name, level):
    coefs = {"b_homophily": 0.0, "b_triadic": 0.0, "b_popularity": 0.0, scan_name: level}
    init = make_init(geometry)
    out_path = RUNS_DIR / f"stageF_{geometry}_{scan_name}_{level:g}.npz"
    rows = run_grid_cell(
        init_result=init, **coefs,
        budget=BUDGET_DEFAULT,
        n_steps=N_STEPS_DEFAULT,
        snapshot_times=SNAPSHOT_TIMES_DEFAULT,
        sim_seed=6000,
        out_path=out_path,
    )
    for r in rows:
        r.update(stage="F", geometry=geometry, scan=scan_name, level=level, rep=0)
    return rows

jobs = [(g, s, l) for g in ("poincare", "torus2d")
                  for s, levels in SCANS.items()
                  for l in levels]
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(stage_f_cell)(*j) for j in tqdm(jobs, desc="Stage F")
)
for group in results:
    all_rows.extend(group)

# Overlay: for each scan, p10_constraint vs level, one line per geometry
df_f = pd.concat([pd.DataFrame(all_rows)], ignore_index=True)
df_end = df_f[(df_f["stage"].isin(["A", "F"])) & (df_f["t"] == N_STEPS_DEFAULT)]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, scan in zip(axes, SCANS):
    sub = df_end[df_end["scan"] == scan]
    for geo, grp in sub.groupby("geometry"):
        grp = grp.sort_values("level")
        ax.plot(grp["level"], grp["p10_constraint"], marker="o", label=geo)
    ax.set_xlabel(scan); ax.set_ylabel("p10_constraint"); ax.set_title(f"Stage F overlay: {scan}")
    ax.legend()
plt.tight_layout(); plt.show()
```

**Cell 10 — Persist summary**

```python
summary = pd.DataFrame(all_rows)
summary.to_parquet(SUMMARY_PATH)
print(f"Wrote {len(summary)} rows → {SUMMARY_PATH}")
```

**Cell 11 — Markdown: Decisions (filled in Task 4)**

```markdown
## Decisions for the main factorial experiment

Filled in after all stages execute and plots are inspected.

- **LEVELS:**
  - `b_homophily`: `[LOW, MID, HIGH]` (chosen from Stage A / F)
  - `b_triadic`: `[LOW, MID, HIGH]`
  - `b_popularity`: `[LOW, MID, HIGH]`
- **budget:** chosen from Stage B (smallest budget with `std_c_size > 0` non-trivially)
- **intercept:** chosen from Stage C (−5.0 if rank stable, else promoted to a factor)
- **n_steps:** `ceil(1.5 × t*)` where `t*` is Stage D's equilibrium time
- **snapshot_times:** `[0, t*, 1.5·t*]` or similar (3–4 snapshots suffice given equilibrium)
- **sim_seed_scheme:** `rep`-only if Stage E shows CRN ≥ 2× reduction, else `rep*1000 + cell`
- **geometry_strategy:** based on Stage F:
  - curves align → single torus-5d experiment, geometry fixed
  - scale differs → per-geometry experiments with per-geometry levels
  - shape differs → geometry as a factor in the main grid
```

- [ ] **Step 2: Verify notebook JSON is valid**

```bash
python -c "import json; json.load(open('05_pilot_calibration.ipynb'))"
```

Expected: no output.

- [ ] **Step 3: Commit the scaffold (unexecuted)**

```bash
git add 05_pilot_calibration.ipynb
git commit -m "feat(notebook): scaffold 05_pilot_calibration with stages A-F"
```

---

## Task 4: Execute the pilot and resolve the Decisions cell

**Files:**
- Execute: `05_pilot_calibration.ipynb`
- Generates: `simulations/pilot/runs/*.npz` (90 files), `pilot_summary.parquet`

This task is **iterative**: Stages B, D, E reference values chosen from Stage A's output. Execute cell-by-cell, update the TODOs, re-execute downstream cells.

- [ ] **Step 1: Execute Stage A**

```bash
jupyter nbconvert --to notebook --execute --inplace 05_pilot_calibration.ipynb --ExecutePreprocessor.timeout=600
```

Or run interactively in Jupyter up to the end of Cell 3. Inspect the three scan plots.

- [ ] **Step 2: Fill in `MID_ACTIVE` and `EQ_SETTINGS["all_high"]` in Cells 5 and 7**

From the Stage A plots:
- Slope level = the coefficient in the middle of the declining region of `p10_constraint`
- Knee level = the smallest coefficient beyond which `p10_constraint` stops declining meaningfully

Edit the notebook to replace the placeholder values. Commit interim progress:

```bash
git add 05_pilot_calibration.ipynb
git commit -m "feat(notebook): lock Stage-A mid-active and knee levels"
```

- [ ] **Step 3: Execute Stages B–F**

Re-run the notebook end-to-end with the chosen levels:

```bash
jupyter nbconvert --to notebook --execute --inplace 05_pilot_calibration.ipynb --ExecutePreprocessor.timeout=600
```

Expected: completes in ≤ 5 min on 8 cores.

- [ ] **Step 4: Verify outputs**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

OUT = Path("simulations/pilot")
runs = sorted((OUT / "runs").glob("*.npz"))
# 19 (A) + 6 (B) + 4 (C) + 3 (D) + 20 (E) + 38 (F) = 90
assert len(runs) == 90, f"expected 90 run files, found {len(runs)}"

summary = pd.read_parquet(OUT / "pilot_summary.parquet")
expected_cols = {"stage", "geometry", "scan", "level", "rep", "t",
                 "b_homophily", "b_triadic", "b_popularity", "intercept",
                 "mean_constraint", "p10_constraint", "p90_constraint",
                 "frac_constraint_lt_0.1", "std_c_size"}
missing = expected_cols - set(summary.columns)
assert not missing, f"missing columns: {missing}"

stages = set(summary["stage"].unique())
assert stages == {"A", "B", "C", "D", "E", "F"}, f"missing stages: {{'A','B','C','D','E','F'}} - {stages}"
print(f"OK: 90 runs, {len(summary)} summary rows, all stages present.")
PY
```

Expected: `OK: 90 runs, <N> summary rows, all stages present.`

- [ ] **Step 5: Fill in the Decisions markdown cell (Cell 11)**

Concrete values derived from the plots:
- `LEVELS` — from Stage A / F overlays (pick flat / slope / knee per mechanism)
- `budget` — from Stage B (smallest with `std_c_size > 0.01`, or fallback threshold chosen by inspection)
- `intercept` — from Stage C (−5.0 if mechanism rank stable across intercepts)
- `n_steps` — `ceil(1.5 × t*)` from Stage D
- `snapshot_times` — derived from `t*`
- `sim_seed_scheme` — matched if Stage E's `std(matched) ≤ 0.5 × std(unmatched)`, else unmatched
- `geometry_strategy` — align / scale / shape verdict from Stage F

Write the final values directly into the markdown cell.

- [ ] **Step 6: Commit the executed notebook + filled Decisions**

```bash
git add 05_pilot_calibration.ipynb
# simulations/ is gitignored — verify before commit
git status simulations/
git commit -m "feat(experiment): run pilot calibration end-to-end, lock decisions"
```

- [ ] **Step 7: Update the memory file with the resolved decisions**

Edit `memory/project_pilot_calibration.md` to replace the "Decisions locked on 2026-04-17" section's sub-items with the concrete values from the notebook's Decisions cell. Commit the memory update separately (it lives outside the project tree; no code commit needed).

---

## Self-Review Checklist (planner-filled, implementer should NOT re-check)

- [x] All 6 stages from the spec are covered by code cells
- [x] Helper changes (tail stats, intercept parameter) are tested before the notebook uses them
- [x] Tasks run in topological order (helper changes → scaffold → execute)
- [x] Task 4 is explicitly iterative — Stage A must produce plots before Stages B/D placeholders are filled
- [x] Verification script in Task 4 Step 4 checks file count, columns, and stage coverage
- [x] No reinvention — `init_torus_uniform`, `init_hyperbolic_uniform`, `run_grid_cell` are reused as-is (with the one new `intercept` kwarg)
- [x] Commit messages follow the repo's conventional-commit style (`feat(experiment)`, `feat(notebook)`)
