# Structural Holes ABM Simulation

**Course:** SOCIOL 338, Duke University
**Goal:** Simulate how social networks evolve temporally and whether structural holes (Burt) emerge spontaneously from simple local rules without agents actively seeking them.

## Research Question

Do structural holes emerge endogenously from behavioral mechanisms (homophily, triadic closure, attention budgets), or do they require intentional brokerage? How does the assumed geometry of social space affect this?

## Theoretical Background

- **Structural holes** (Burt): gaps between densely connected groups. Agents bridging these gaps have brokerage advantages.
- **Hyperbolic geometry of networks** (Krioukov et al. 2010): hyperbolic space naturally reproduces heavy-tailed degree distributions, high clustering, and short path lengths simultaneously. Radial position = popularity, angular position = similarity.
- **Popularity vs. similarity** (Papadopoulos et al.): network growth driven by both similarity (homophily) and popularity (preferential attachment).

## Model Design

### Initialization (geometry-specific, produces a distance matrix D)

Four initialization options forming a 2x2:

|                | Toroidal                          | Hyperbolic                                    |
|----------------|-----------------------------------|-----------------------------------------------|
| **Uniform**    | Uniform on [0,1)^d, no structure  | Krioukov Eq. 17: sinh(αr) radial + uniform θ  |
| **GMM**        | Gaussian mixture, pre-seeded clusters | Krioukov radial + Gaussian mixture on θ    |

Key design decision: the geometry is absorbed into the initialization step. All downstream dynamics operate only on the distance matrix D and adjacency matrix A.

- **Toroidal**: flat, no hierarchy, wrap-around eliminates edge effects. "Neutral" geometry — any structure must come from dynamics.
- **Hyperbolic**: radial position creates natural popularity hierarchy (central agents = hubs). Produces realistic degree distributions without engineering.
- **"Uniform" means different things**: toroidal uniform is truly structureless. Hyperbolic "uniform" (uniform in hyperbolic area) already has radial hierarchy baked in via the sinh density.

#### Key parameters
- **α** (hyperbolic): controls power-law exponent γ = 2α + 1. α=1 gives γ=3 (standard Krioukov), α=0.5 gives γ=2 (more spread, heavier tail). Currently set to 0.5.
- **d** (torus): dimension. Higher d causes concentration of measure — pairwise distances converge, making distance-based tie formation degenerate. The hyperbolic model is inherently 2D (H²).

### Dynamics (geometry-agnostic, operates on D and A)

Each mechanism is a function `(SimState) → (N, N) factor matrix`. Mechanisms multiply together:

```
P(i,j) = [homophily factor] * [triadic factor] * [budget factor] * (1 - A_ij)
```

Then new edges are sampled from P, and ties decay for over-budget agents.

#### Mechanisms

| Mechanism | Formula | Parameter | Effect |
|-----------|---------|-----------|--------|
| Homophily | e^{-λ d_ij} | λ (decay rate) | Similar agents more likely to connect |
| Triadic closure | τ^{n_ij} where n_ij = shared neighbors | τ (boost factor) | Friends-of-friends more likely to connect |
| Attention budget | σ(k_i, b_i) · σ(k_j, b_j) where σ is logistic | β (sharpness), b_i (budget) | Penalizes tie formation when over capacity |
| Popularity | (k_j + 1)^μ | μ (attachment strength) | High-degree agents attract more ties |

Popularity is designed but not yet implemented.

#### Tie Decay
When an agent's degree exceeds their budget b_i, the most distant ties are dropped until degree <= b_i. This is deterministic (drop farthest first).

### Measurement
- **Burt's constraint** C_i: lower = more structural holes around agent i
- **Betweenness centrality**: identifies brokers
- **Degree distribution**: should be heavy-tailed (not Poisson)
- **Clustering coefficient**: local clique density
- **Modularity**: community structure strength

## Project Structure

```
abm_core.py                  — Initialization functions + distance computation
                               Returns InitResult(distance_matrix, viz_coords, metadata)
                               Also: static tie formation, calibrate_radius, burt_constraint

abm_dynamics.py              — (TO BUILD) Mechanism functions + SimState + step()
abm_runner.py                — (TO BUILD) Simulation loop + history recording
app_panel.py                 — (TO BUILD) Panel web app for interactive visualization

01_torus_initialization.ipynb    — Explores torus geometry: uniform vs GMM,
                                   dimensionality effects, concentration of measure
02_hyperbolic_initialization.ipynb — Explores hyperbolic geometry: Krioukov prescription,
                                     α parameter, Fermi-Dirac temperature, degree distributions
03_dynamics.ipynb                — (TO BUILD) Mechanism comparison experiments

tests/
  test_dynamics.py           — (TO BUILD) Unit tests for mechanisms
  test_runner.py             — (TO BUILD) Integration tests for simulation

PLAN.md                      — Full implementation plan (9 tasks)
```

## Key Design Decisions Made

1. **Common interface**: All initializations return `InitResult` with a distance matrix. Geometry is fully absorbed at init time — dynamics never see coordinates.
2. **Mechanisms as multipliers**: Each mechanism returns an (N, N) factor matrix. They compose by elementwise multiplication. Toggleable and stackable.
3. **Vectorized computation**: All dynamics are matrix operations (numpy), no Python loops over agent pairs. `A @ A` gives shared neighbor counts.
4. **Panel for visualization**: Chosen over Streamlit (laggy rerun model) and D3.js (separate language). Panel's `Player` widget handles timeline scrubbing natively with Bokeh for interactive network plots.
5. **α = 0.5 default** for hyperbolic: spreads agents away from boundary (γ=2 power law). The standard α=1 (γ=3) compresses almost all agents to the boundary, making visualization poor.
6. **Visual scaling for Poincare disk**: `(r/R)^0.6` instead of `tanh(r/2)` to spread agents across the disk for readability.

## Mathematical Notes

### Concentration of Measure (Torus)
In d-dimensional space, random pairwise distances concentrate: Var(d_ij) ~ 1/d. At high d, all agents are roughly equidistant, making distance-based tie formation degenerate. The torus notebook identifies where this becomes problematic (CV plot).

### Near-Orthogonality
For random unit vectors in d dimensions, E[a·b] = 0, Var[a·b] = 1/d. Moving toward agent A changes distance to agent B by ~a·b, which is near-zero in high d. This means high-dimensional social space is less "zero-sum" — being close to one agent doesn't force you away from another.

### Hyperbolic Density (Krioukov Eq. 7/17)
ρ(r) = α sinh(αr) / (cosh(αR) - 1). "Uniform" in hyperbolic area means exponential in the radial coordinate. Most agents at boundary (large r, low degree), few at center (small r, high degree hubs).

### Attention Budget Tension
Popularity (high degree attracts more ties) and budget (high degree penalizes new ties) pull in opposite directions. This tension may be sufficient to generate structural holes around hubs — they attract diverse connections but can't maintain them all, leaving gaps between their contacts.

## References

- Burt, R. (1992). *Structural Holes*
- Krioukov, D. et al. (2010). *Hyperbolic Geometry of Complex Networks*. [In project folder]
- Papadopoulos, F. et al. (2012). *Popularity versus Similarity in Growing Networks*. [In project folder]

## Running

```bash
# Notebooks (existing)
jupyter notebook 01_torus_initialization.ipynb
jupyter notebook 02_hyperbolic_initialization.ipynb

# Web app (after Tasks 1-9 are complete)
pip install panel bokeh
panel serve app_panel.py --show

# Tests (after Tasks 1-7 are complete)
python -m pytest tests/ -v
```
