# Structural Holes ABM Simulation

**Course:** SOCIOL 338, Duke University
**Goal:** Simulate how social networks evolve under simple local tie-formation rules, and ask whether Burtian structural holes emerge **spontaneously** from those rules — without any agent trying to become a broker.

## Research Question

Do structural holes emerge endogenously from behavioral mechanisms (homophily, triadic closure, popularity), or do they require intentional brokerage? And how does the assumed geometry of social space — flat vs curved, low-dimensional vs high-dimensional — shape the answer?

---

## Interactive Panel App

The easiest way to understand what this model does is to run it live. `app_panel.py` is a Panel/Bokeh app that runs **four seeded simulations side-by-side** (2×2 grid) so you can see both the signal (mechanism effect) and the noise (replicate variability) in one view.

```bash
pip install panel bokeh numpy scipy networkx
panel serve app_panel.py --show
```

![panel screenshot placeholder — add a PNG of the running app here]

### What you can tinker with (left sidebar)

**Geometry & initialization**
- `Geometry`: `torus_uniform`, `torus_gmm`, `hyperbolic_uniform`, `hyperbolic_gmm`
- `N agents`, torus dimension `d`, hyperbolic `α`, GMM cluster count and spread
- `Normalize`: rescale the distance matrix by mean/median/max — the pivotal choice for cross-geometry comparability
- Per-panel seeds so the 2×2 grid shows 4 independent replicates of the same parameter set

**Tie-formation mechanisms**
Each mechanism is toggleable with its own coefficient slider. Log-odds model:

```
logit P(tie_ij) = intercept + β_h·(−d_ij) + β_t·n_shared_ij + β_p·log(k_j + 1) + attention_penalty
```

- Homophily `β_h`: prefer similar (nearby) alters
- Triadic closure `β_t`: prefer alters who share existing neighbors
- Popularity `β_p`: prefer high-degree alters (Matthew effect)
- Attention budget: hard cap `b` or soft sigmoid penalty — finite social capacity

**Visualization controls**
- `Node color`: Burt constraint (red→green = embedded→broker), Louvain community, clustering coefficient, degree
- `Edge opacity`: constant or weighted by shared-neighbor count
- `Timesteps` slider + **Player** widget to scrub through the simulation frame-by-frame

### What you'll see

Each of the 4 panels renders:
1. **Network plot** — nodes at their initialization coordinates (or unit-disk embedding for hyperbolic), edges from the current adjacency matrix, colored by your chosen metric.
2. **Time-series chart** — how the three Burt-constraint components (size / density / hierarchy) evolve over simulation time.
3. **Histogram** — distribution of per-node constraint at the current timestep, with `p10` (brokerage tail) marked.

The Player widget lets you walk through 200+ timesteps of dynamics without re-running. A **Save Simulation** button dumps the run to `simulations/sim_<timestamp>/` so you can load it later in analysis notebooks.

### What makes it worth running

- **Four seeds at once** makes it immediately visible whether a finding is robust or a seed artifact.
- **Compare geometries live**: switch `torus_uniform` → `hyperbolic_uniform` and see the hub/periphery constraint pattern snap into place.
- **Dial `β_p` from 0 to 1 with the slider** and watch `p10` rise instead of fall — the counterintuitive popularity result (Finding 2 in the paper) is visceral when you see it happen.

---

## Theoretical Background

- **Structural holes** (Burt 1992): gaps between densely connected groups. Agents bridging them have brokerage advantages. Measured via the **constraint index** `C_i`, decomposed into size / density / hierarchy components.
- **Hyperbolic network geometry** (Krioukov et al. 2010): a latent hyperbolic space naturally reproduces heavy-tailed degree distributions, high clustering, and short path lengths together. Radial = popularity, angular = similarity.
- **Popularity vs. similarity** (Papadopoulos et al. 2012): network growth is driven by both similarity (homophily) and popularity (preferential attachment).

## Model Design

### Initialization (geometry-specific → distance matrix `D`)

A 2×2 of init choices:

|                | Toroidal                          | Hyperbolic                                    |
|----------------|-----------------------------------|-----------------------------------------------|
| **Uniform**    | Uniform on `[0,1)^d`, no structure  | Krioukov: `sinh(αr)` radial + uniform θ  |
| **GMM**        | Gaussian mixture, pre-seeded clusters | Krioukov radial + Gaussian mixture on θ    |

**Key decision:** geometry is absorbed at init time. Everything downstream runs on `D` and the adjacency matrix `A` — no coordinates.

### Dynamics (geometry-agnostic)

Each mechanism contributes a term to the logit of tie formation. At each timestep:

1. Compute `logit_ij = intercept + Σ β_k · feature_k(i, j) + attention_penalty(i, j)`
2. Sample new edges from `σ(logit_ij) · (1 − A_ij)`
3. Over-budget agents drop their most-distant ties

### Measurement

- **Burt constraint** `C_i` with size/density/hierarchy decomposition — primary outcome
- **`p10_constraint`**: 10th percentile, captures the broker tail
- Optional: degree distribution, clustering coefficient, modularity

---

## Experiments

The repo contains three ordered experiments, each in its own notebook:

| Notebook | Runs | What it establishes |
|----------|------|---------------------|
| `04_grid_search_dynamics.ipynb` | 27 cells × 3 reps on torus-5d | Budget saturates everyone; homophily dominates; `c_density` carries the signal |
| `05_pilot_calibration.ipynb` | 90 runs across stages A–F | Locks `budget=20`, `intercept=-5`, `n_steps=50`. Finds `c_size` mechanically saturated at `1/budget`. Equilibrium by t≈20. |
| `06_main_factorial.ipynb` | **972 runs**: 6×6×3 mech × 3 geo × 3 rep | Headline experiment. Popularity dominates and saturates at β_p=0.5; geometry–mechanism interaction is real (cross-geometry Stage F prediction fails); torus-5d has ~2.5× less dynamic range than torus-2d/hyperbolic. |

Headline findings from the main factorial (in `docs/` as paper section drafts):
1. **Geometry sets the ceiling on brokerage variation.** Torus-5d compresses all mechanism effects into a tight band; torus-2d and hyperbolic give the mechanisms real teeth.
2. **Popularity dominates but saturates fast and substitutes for homophily.** β_p=0→0.5 does almost all the work; β_p=0.5→1 adds nothing. Under strong homophily, popularity has no additional effect.
3. **Normalization doesn't equalize geometries.** Concentration-of-measure in 5d tightens the distance distribution even after mean-normalizing, so mechanism coefficients are not fully transferable across geometries.

---

## Project Structure

```
abm_core.py                   — Initialization + distance computation.
                                Returns InitResult(distance_matrix, viz_coords, metadata).
                                Also: burt_constraint, burt_constraint_decomposed, .normalized()

abm_dynamics.py               — Mechanism functions (homophily, triadic, popularity),
                                attention-budget constraints, SimState, step()
abm_runner.py                 — Simulation loop + history recording (run_simulation)
abm_storage.py                — Save/load simulations to disk
experiment_grid_search.py     — Batch runner used by the factorial notebooks
app_panel.py                  — Interactive Panel/Bokeh app (see section above)

01_torus_initialization.ipynb       — Torus geometry: uniform/GMM, concentration of measure
02_hyperbolic_initialization.ipynb  — Hyperbolic: Krioukov prescription, α, degree distributions
03_dynamics.ipynb                   — First look at mechanism effects
04_grid_search_dynamics.ipynb       — 27-cell × 3-rep sweep (torus-5d)
05_pilot_calibration.ipynb          — 90-run calibration, locks hyperparameters
06_main_factorial.ipynb             — 972-run mechanism × geometry factorial

make_paper_figures.py         — Regenerates paper figures from saved simulations
make_saturation_figure.py     — Saturation-limit analysis figure
run_paper_dose_response.py    — β_p dose-response sweep
verify_saturation.py          — Sanity-checks for the saturation story

tests/
  test_dynamics.py            — Mechanism unit tests
  test_runner.py              — Simulation integration tests
  test_storage.py             — Save/load roundtrip tests

docs/superpowers/             — Plans and design specs for each experiment
figures/                      — Rendered paper figures
simulations/                  — (gitignored) per-run output data
```

## Key Design Decisions

1. **Common `InitResult` interface**: all initializations return a distance matrix. Dynamics never see coordinates — geometry is fully absorbed at init.
2. **Log-odds mechanism composition**: each mechanism contributes to a logit, not a multiplier. This is the post-April-2026 redesign per professor feedback — cleaner mathematically than multiplicative factors.
3. **Budget cap matters more than it looks**: with `b=20`, preferential attachment can't produce hubs, so the popularity mechanism produces a *dense core* instead. This is the pivotal modeling choice that explains the counterintuitive popularity/`p10` result.
4. **Panel over Streamlit / D3**: Panel's `Player` widget handles timeline scrubbing natively with Bokeh; 4-seed parallel rendering is straightforward.
5. **α=0.5 default** for hyperbolic (power-law exponent γ=2): agents spread across the disk rather than all compressing to the boundary.
6. **Normalization exposed as a live choice** rather than hardcoded — the main-factorial finding that mean-normalization *fails* to equalize geometries is a first-class result, not a caveat.

## Mathematical Notes

### Concentration of Measure (torus)
`Var(d_ij) ~ 1/d`. At high d, all agents become roughly equidistant — distance-based tie formation degenerates, and the mechanism-effect range compresses. Torus-5d sits at ~half the dynamic range of torus-2d.

### Hyperbolic Density (Krioukov Eq. 7/17)
`ρ(r) = α sinh(αr) / (cosh(αR) − 1)`. "Uniform" in hyperbolic area means exponential in the radial coordinate: most agents near the boundary (low degree, periphery), few at the centre (high degree, hubs).

### Attention Budget vs Popularity
Popularity attracts ties to high-degree agents; budget caps everyone's degree. With `b=20`, they don't produce hubs — they produce a dense early-forming core of first-movers whose members fill each other's budgets. Every peripheral connects into that core, raising the density component of constraint across the whole network.

## References

- Burt, R. (1992). *Structural Holes: The Social Structure of Competition*.
- Krioukov, D. et al. (2010). *Hyperbolic Geometry of Complex Networks*. Phys. Rev. E 82, 036106.
- Papadopoulos, F. et al. (2012). *Popularity versus Similarity in Growing Networks*. Nature 489, 537–540.
- Simmel, G. (1908). *Die Kreuzung sozialer Kreise* (The Web of Group Affiliations).

## Running

```bash
# Install
pip install numpy scipy pandas networkx joblib tqdm panel bokeh matplotlib pyarrow

# Interactive app
panel serve app_panel.py --show

# Notebooks (in order of increasing scope)
jupyter notebook 01_torus_initialization.ipynb      # geometry warm-up
jupyter notebook 05_pilot_calibration.ipynb         # calibration
jupyter notebook 06_main_factorial.ipynb            # headline experiment

# Tests
python -m pytest tests/ -v
```
