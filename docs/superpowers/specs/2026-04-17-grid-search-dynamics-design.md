# Grid Search of Network Forming Forces — Design

**Date:** 2026-04-17
**Goal:** Quantify how much each social network forming force (homophily, triadic closure, popularity) affects network bonding, holding geometry fixed.

## Motivation

The geometry-only initialization gives us a distance matrix; the dynamics layer translates that into ties through three mechanisms plus an attention budget constraint. We currently don't know which mechanism contributes how much to the resulting structure. This experiment temporarily holds geometry constant and varies the dynamics on a 3×3×3 grid so we can read off marginal and interaction effects from the recorded constraint distributions.

## Experimental Setup

### Fixed factors

| Factor | Value | Source |
|---|---|---|
| Geometry | Torus, 5d | `abm_core.init_torus_uniform` |
| Initial distribution | Uniform on `[0, 1)^5` | same |
| N (agents) | 300 | matches existing notebooks |
| Budget cap | 10 (uniform) | hard cap |
| Constraint | `constraint_attention_hard` | over-budget pairs cannot form ties |
| Decay | Enabled (drops excess ties post-step) | `step(... enable_decay=True)` |
| Intercept | −5.0 (existing default) | `step(... intercept=-5.0)` |
| Steps per run | 500 | |
| Snapshot times | t = 0, 100, 200, 300, 400, 500 | 6 snapshots/run |

### Varied factors (3³ = 27 cells)

```python
LEVELS = {
    "b_homophily":  [0.0, 1.0, 3.0],
    "b_triadic":    [0.0, 0.5, 1.5],
    "b_popularity": [0.0, 0.2, 0.6],
}
```

Levels are calibrated per-mechanism because the multipliers operate on different scales:
- Homophily multiplies `1/(d + 0.1)` where d ∈ ~[0.1, 1] (normalized torus distances)
- Triadic multiplies # shared neighbors (typically 0–5)
- Popularity multiplies target degree (typically 0–10)

A mechanism with coef 0 is **omitted** from the mechanism list (cleaner than passing a zero-contribution function). The all-zero cell `(0, 0, 0)` is a baseline run with only the intercept driving tie formation.

### Replicate / blocking design

**Randomized block design**: 3 replicates × 27 cells = **81 runs**.

- Generate 3 different `InitResult`s (init seeds 0, 1, 2) — one per "block"
- Each cell is run 3 times, once per block, sharing that block's init
- Within a block, the same geometry is used across all 27 cells → enables paired comparisons across cells while still capturing init variance across blocks
- Tie-formation RNG seed = `block_seed * 1000 + cell_index` for reproducibility

## Recording

For every run, after `run_simulation` finishes (in-memory full history):

1. Extract per-node arrays `[constraint, c_size, c_density, c_hierarchy]` at each snapshot time → shape `(6, 300)` per metric.
2. Extract the adjacency frame at each snapshot time (sparse).
3. Discard the rest of the history before the next run.

Outputs under `simulations/grid_search_torus_5d/`:

1. `runs/cell_{idx:02d}_rep{r}.npz` — per-node metric arrays + 6 sparse frames
2. `summary.parquet` — one row per (cell, rep, snapshot_t) with columns:
   - `cell_id, b_homophily, b_triadic, b_popularity, replicate, t`
   - `mean_C, median_C, std_C` and the same triple for `c_size`, `c_density`, `c_hierarchy` (12 stat columns total)

Total disk: ~5 MB summary + ~50 MB per-node arrays.

## Notebook Structure

`04_grid_search_dynamics.ipynb`:

1. **Config** — N, D, BUDGET, N_STEPS, SNAPSHOT_TIMES, REPLICATES, LEVELS, output_dir
2. **Helpers** (~30 lines, inline):
   - `build_mechanisms(b_h, b_t, b_p)` — returns filtered mechanism closures
   - `extract_snapshots(history, times)` — dict of arrays
   - `summarize(node_metrics_at_t)` — dict of mean/median/std per metric
3. **Grid run** — nested loop with `tqdm`, calls `run_simulation`, saves per-run npz + appends summary rows
4. **Load + sanity check** — reads `summary.parquet`, prints baseline `(0, 0, 0)` cell to confirm minimal dynamics
5. **Histogram plots** — for each of the 4 metrics, faceted histogram grid showing distribution at each snapshot time, sliced by mechanism level
6. **Trajectory plots** — for each metric, mean ± std across replicates over t for all 27 cells, colored by one mechanism at a time

The heavy lifting (init, mechanisms, runner, constraint decomposition) stays in the existing modules. The notebook is the experiment script + plots only.

## Out of Scope

- Geometry sweeps (will be run later with the same harness)
- Soft attention budget (`constraint_attention_budget`)
- Decay-disabled comparisons
- Larger N or longer time horizons
- Per-replicate error bars beyond mean ± std (no formal ANOVA in this notebook)

## Success Criteria

- All 81 runs complete without error
- Baseline `(0, 0, 0)` cell shows mean_C close to 1.0 (most nodes isolated)
- Each "high homophily, all else off" cell shows clearly different C distribution from the baseline (sanity check that mechanisms have effect)
- Plots make it visually obvious which mechanism dominates the C distribution shape
