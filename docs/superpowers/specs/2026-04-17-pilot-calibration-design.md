# Pilot Calibration Run — Design

**Date:** 2026-04-17
**Goal:** Calibrate coefficient ranges, budget, intercept, equilibrium time, and RNG strategy before committing to a follow-up factorial experiment.

## Motivation

The first grid search (`2026-04-17-grid-search-dynamics-design.md`) exposed several design flaws that would have been caught by a pilot:

- Budget cap saturated every cell — `c_size` was a mechanical constant.
- `b_homophily = 3.0` already past its knee — high cells silenced the other two mechanisms.
- 500 steps was ~5× overkill — equilibrium by t ≈ 100.
- Paired comparisons between cells carried sim-RNG noise (each cell used a different seed).

This pilot is **calibration, not inference**. It does not attempt to estimate main effects, interactions, or significance. Its job is to produce the settings for the next factorial experiment.

## What the Pilot Must Answer

1. At what coefficient value does each mechanism **turn on**, and where does it **saturate**?
2. What budget lets `c_size` carry signal instead of being pinned at `1/k`?
3. When does the system actually reach equilibrium?
4. Does the intercept (currently fixed at −5.0) materially change findings?
5. Does Common Random Numbers (CRN) across cells reduce paired-contrast noise?

## Resolved Decisions

1. **DV:** record `mean_C`, `p10_C`, and `frac_C_lt_0.1` at every snapshot. **Primary = `p10_constraint`** — it maps directly to the paper's structural-holes thesis (does the low-constraint tail exist under each mechanism?). Turn-on and knee are defined against `p10_C`.
2. **Budget:** fixed (not a factor in the main experiment); value to be chosen from Stage B. **Fallback:** if Stage B reveals that mechanism rank order flips across budgets, budget is promoted to a factor in the main experiment.
3. **Geometry:** Stage A runs on torus-5d. Stage F re-runs Stage A on two alternative geometries — **Poincaré (hyperbolic)** and **torus-2d** — as a smoke test. If all three geometries agree on turn-on and knee, main-experiment levels generalize; if they diverge, geometry must enter the main factorial or be calibrated per-geometry.
4. **N:** 300 (match the main experiment; mismatch would hide degree-distribution effects and make pilot-to-main transfer risky).

## Fixed Factors (unless noted per stage)

| Factor | Value |
|---|---|
| Geometry | Torus 5d, uniform init (except Stage F) |
| N | 300 |
| Budget cap | 10 (except Stage B) |
| Constraint | `constraint_attention_hard` |
| Decay | Enabled |
| Intercept | −5.0 (except Stage C) |
| Init seed | 0 (single init per geometry, except where noted) |
| **Primary DV** | **`p10_constraint`** |

## Stages

### Stage A — 1D Marginal Scans

Vary one mechanism at a time with the other two pinned at 0. This is **three line scans**, not a factorial.

| Scan | Levels | Others | Runs |
|---|---|---|---|
| `b_homophily` | 0, 0.25, 0.5, 1, 2, 4, 8 | `b_triadic = b_popularity = 0` | 7 |
| `b_triadic` | 0, 0.25, 0.5, 1, 2, 4 | `b_homophily = b_popularity = 0` | 6 |
| `b_popularity` | 0, 0.1, 0.25, 0.5, 1, 2 | `b_homophily = b_triadic = 0` | 6 |

- 1 init seed, 1 rep per point → **19 runs**
- `n_steps = 200`, snapshots at t = 0, 50, 100, 150, 200
- Record per-level at t=200: tie density, `mean_C`, `p10_C`, `p50_C`, `p90_C`, `frac_C_lt_0.1`, `mean_c_size`, `mean_c_density`, `mean_c_hierarchy`
- **Deliverable:** three curves (`p10_constraint` vs coefficient; `mean_C` and `frac_C_lt_0.1` overlaid for robustness). From each curve, pick three informative levels for the main factorial: one **before** turn-on (flat), one **in the slope**, one **at the knee** (not past it). Turn-on defined as the first level where `p10_C` departs from baseline by more than the observed single-run noise; knee defined as the level beyond which further increases move `p10_C` by less than that same noise band.

### Stage B — Budget Scan

Establish the budget value at which `c_size` stops being a constant.

- Fix `(b_homophily, b_triadic, b_popularity)` at the mid-active setting chosen from Stage A (e.g. all three at their Stage-A slope levels).
- Sweep `budget ∈ {5, 8, 12, 20, 30, ∞}` (∞ = no `constraint_attention_hard`).
- 1 rep each → **6 runs**
- `n_steps = 200`
- Record `mean_c_size`, `std_c_size`, `mean_degree`, `max_degree`.
- **Deliverable:** plot of `std_c_size` vs budget. Pick the smallest budget where `std_c_size > 0.01` (concrete threshold TBD from data). Decide whether to fix budget at that value in the main experiment or promote it to a factor.

### Stage C — Intercept Scan *(recommended, optional)*

Test whether the intercept is load-bearing.

- Fix mechanism coefficients at Stage-A mid-active.
- Sweep `intercept ∈ {−7, −5, −3, −1}`
- 1 rep each → **4 runs**
- `n_steps = 200`
- Record tie density, `mean_C`, `p10_C`, `frac_C_lt_0.1`.
- **Deliverable:** if mechanism rank order (homophily vs triadic vs popularity, measured via `p10_constraint`) stays stable across intercepts, fix intercept at −5.0. If it flips, intercept must be varied in the main experiment.

### Stage D — Equilibrium Check

Find the first timestep at which per-node constraint is stationary.

- Three settings: `(0, 0, 0)` baseline, Stage-A mid-active all-on, Stage-A high all-on.
- 1 rep each → **3 runs**
- `n_steps = 300`, snapshots every 10 steps (30 snapshots)
- Equilibrium criterion: `|Δ mean_C| < 0.002` for 5 consecutive snapshots.
- **Deliverable:** `t* = max` of the three cell equilibrium times. Set main experiment `n_steps = ceil(1.5 × t*)`.

### Stage E — Common Random Numbers (CRN) Check

Quantify whether matched sim seeds across cells reduce paired-contrast variance.

- Two "adjacent" cells: e.g. `(b_h=1, 0, 0)` vs `(b_h=2, 0, 0)`.
- Two regimes:
  - **Matched:** cell A rep k and cell B rep k use the *same* sim seed; k = 0..4
  - **Unmatched:** independent seeds per (cell, rep)
- 2 cells × 5 reps × 2 regimes → **20 runs**
- `n_steps = 200`
- Compute `std_k(p10_C_A − p10_C_B)` under each regime (primary DV); also compute for `mean_C` as a robustness check.
- **Deliverable:** if matched-regime std is ≥ 2× smaller than unmatched (on the primary DV), adopt CRN for the main experiment (sim seed depends on rep only, not on cell).

### Stage F — Geometry Smoke Test

Re-run Stage A's 1D marginal scans on two alternative geometries to test whether the chosen levels generalize beyond torus-5d.

- **Geometry F1:** Poincaré (hyperbolic disk) — the geometry most relevant to the paper's hub-periphery structural-holes story.
- **Geometry F2:** Torus 2d — tests generalization within the torus family at a lower dimension.
- Same level grids as Stage A (7 + 6 + 6 = 19 per geometry), same `n_steps = 200`, same snapshots.
- 1 init seed per geometry (init seed = 0 on both; init function differs).
- **2 geometries × 19 scans = 38 runs.**
- Record the same per-level quantities as Stage A.
- **Deliverable:** for each mechanism, overlay the three geometry-conditional `p10_C` curves. Three outcomes:
  - **Curves align** (turn-on and knee at similar coefficient values across geometries): main-experiment levels from Stage A transfer; geometry can remain fixed or be added later as an isolated factor.
  - **Curves differ in scale but same shape**: main experiment needs per-geometry levels; keep geometry out of the main factorial and run parallel experiments instead.
  - **Curves differ in shape** (e.g. a knee on torus-5d but no knee on Poincaré): the pilot has revealed a first-order geometry-mechanism interaction; the main experiment must include geometry as a factor, and Stages B–E may need re-planning.

## Recording

Reuse the existing `experiment_grid_search.run_grid_cell` harness (extend if needed to accept a generic `init_result`). Outputs under `simulations/pilot/`:

- `runs/stage{A,B,C,D,E,F}_*.npz` — per-node metric arrays + sparse frames at snapshot times
- `pilot_summary.parquet` — one row per (stage, geometry, cell, rep, t) with:
  - `stage, geometry, scan, level, rep, t`
  - `mean_C, p10_C, p50_C, p90_C, std_C, frac_C_lt_0.1`
  - `mean_c_size, std_c_size, mean_c_density, mean_c_hierarchy`
  - `tie_density, mean_degree, max_degree`

## Notebook Structure

New notebook `05_pilot_calibration.ipynb`:

1. **Config** — N, D, budget, snapshot schedules per stage, geometry list for Stage F
2. **Stage A** — three scans on torus-5d + turn-on/knee plots (primary DV `p10_C`, overlays for `mean_C` and `frac_C_lt_0.1`)
3. **Stage B** — budget scan + `std_c_size` vs budget + mechanism rank-stability check across budgets
4. **Stage C** — intercept scan + rank-stability check
5. **Stage D** — equilibrium plot with stationarity marker
6. **Stage E** — CRN variance-reduction check
7. **Stage F** — three scans × three geometries overlaid (torus-5d reuses Stage A output; Poincaré and torus-2d are fresh)
8. **Decisions** — markdown cell that records the chosen levels, budget, intercept, n_steps, seed scheme, and geometry strategy for the main experiment

Parallelization: reuse `joblib.Parallel(n_jobs=-1, backend="loky")` pattern from cell 4 of `04_grid_search_dynamics.ipynb`.

## Total Cost

**90 runs**, most ≤200 steps. Estimated wall time ≤ 3 minutes on 8 cores.

| Stage | Runs | Steps | Purpose |
|---|---|---|---|
| A | 19 | 200 | Range-finding per mechanism (torus-5d) |
| B | 6 | 200 | Budget saturation |
| C | 4 | 200 | Intercept sensitivity |
| D | 3 | 300 | Equilibrium time |
| E | 20 | 200 | CRN check |
| F | 38 | 200 | Geometry smoke test (Poincaré + torus-2d, Stage A scans each) |

## Out of Scope

- Estimating main effects or interactions (that is the main experiment's job)
- Formal hypothesis tests, preregistration, or ANOVA
- Geometry sweeps (will be folded in at the main experiment stage if decided)
- Heterogeneous budgets

## Success Criteria

- All 90 runs complete without error.
- Stage A produces monotone `p10_constraint` curves with a visible turn-on and knee for each mechanism. If a curve is flat throughout, the chosen level range was wrong and the pilot must be extended.
- Stage B identifies a budget at which `std_c_size > 0` non-trivially.
- Stage D identifies `t* ≤ 200`; if `t* > 200`, Stage A/B/C/F results are under-equilibrated and must be re-run at the new `n_steps`.
- Stage F produces a documented classification (curves align / scale differs / shape differs) for each mechanism × geometry pair, with a recommendation on whether the main experiment can use torus-5d calibration or needs per-geometry calibration.
- The "Decisions" notebook cell at the end contains concrete values for: `LEVELS`, `budget`, `intercept`, `n_steps`, `snapshot_times`, `sim_seed_scheme`, and `geometry_strategy`. These are the inputs to the main experiment's design doc.
