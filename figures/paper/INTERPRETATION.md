# Paper figure interpretation

Four figures generated 2026-04-17 from two experiments:
- `simulations/paper_figures/dose_response.parquet` — 162 runs: 3 geometries × 3 mechanisms × 6 levels × 3 reps, each mechanism scanned alone with others pinned at 0.
- `simulations/main_factorial/summary.parquet` — 243 runs: 3³ mechanism grid × 3 geometries × 3 reps at pilot-locked settings.

All runs use `budget=20`, `intercept=-5`, `n_steps=50`, distance-normalized inits.
Primary outcome: **brokerage headcount** = fraction of agents with Burt constraint `C_i < 0.1`.

---

## Figure 1 — Dose-response (`fig1_dose_response`)

> Brokerage headcount vs coefficient, one panel per geometry, one line per mechanism (b_homophily / b_triadic / b_popularity).

**Paper claim**: The three mechanisms erode brokerage at clearly different rates, and the ordering depends on geometry.

### Evidence
- **Euclidean 5D torus** — all three mechanisms keep brokerage at ≈1.0 across the full coefficient range (0 → 4). **Brokerage is geometrically protected.**
- **Euclidean 2D torus** — popularity destroys brokers earliest (drops at b=0.25), then triadic closure (threshold near b=2), then homophily (threshold near b=2 but with a slightly smoother curve).
- **Hyperbolic (Poincaré)** — popularity again the fastest destroyer (action from b=0.25); triadic closure threshold at b=2; homophily is the slowest — brokers remain intact until b=4.

### Mechanism ranking on broker-destroying geometries
**popularity > triadic closure > homophily.**
Popularity acts on **every** eligible target pair via degree; it does not require local structure to be built up first. Triadic closure requires pre-existing ties to amplify (hence its sharp threshold). Homophily acts most locally and is the most gradual.

---

## Figure 2 — Constraint distributions (`fig2_distributions`)

> Per-node constraint histograms for three geometries (rows) × four conditions (baseline + each mechanism alone at b=4). Red dashed line at `C = 0.1` marks the broker threshold.

**Paper claim**: Mechanisms produce a **shifted unimodal** distribution, not a **bimodal** one. This tells us mechanisms embed *everyone* uniformly rather than producing a two-population split of brokers and non-brokers.

### Evidence
- **Baseline** on all three geometries: tight unimodal distribution near the budget floor (`1/20 = 0.05`). 100% brokers.
- **Torus-5d** (top row): all four conditions look nearly identical. Mechanisms cannot shift the distribution on this geometry.
- **Torus-2d** (middle row): baseline near floor, each mechanism at b=4 shifts the entire distribution up to ~0.10–0.12. No left tail survives → brokers drop from 100% to ~4%.
- **Poincaré** (bottom row): baseline near floor; each mechanism at b=4 produces a **slightly wider** distribution centered around 0.10–0.12, with a few agents remaining below 0.1 (~7–11% brokers). Homophily and triadic produce distributions marginally wider than popularity.

### Interpretation
The model does **not** produce the classic structural-holes story of "most people embedded + a few brokers spanning structural holes." It produces **uniform embedding** once mechanisms cross their threshold. This limits what the paper can claim about heterogeneous brokerage.

---

## Figure 3 — Two-way interactions (`fig3_interactions`)

> Main-factorial heatmaps showing `frac_C<0.1` across all pairs of mechanisms, marginalized over the third, for each geometry.

**Paper claim**: On broker-destroying geometries, mechanism effects are **largely non-interactive** — any single mechanism being on is sufficient to collapse brokerage. On torus-5d, no combination of mechanisms can destroy brokerage.

### Evidence
- **Torus-5d** (top row): every cell at 1.00. Brokerage is invariant to mechanism combinations. No interaction effects.
- **Torus-2d** (middle row): the entire grid collapses to 0.03–0.05 *except* the strict all-off corner (~0.36–0.41). Any mechanism on → brokers destroyed. Additive, not multiplicative; no synergy or antagonism.
- **Poincaré** (bottom row): similar pattern with slightly more variation in the low-coefficient cells (e.g., homophily=2 with other mechanisms off preserves ~0.20 brokerage).

### Takeaway
The paper's grid-search is less informative than 1D scans for establishing mechanism contributions — the mechanisms saturate each other rather than interacting multiplicatively. For mechanism characterization, dose-response curves (Fig 1) are the right tool.

---

## Figure 4 — Summary bar (`fig4_summary_bar`)

> For each geometry, the fraction of agents remaining in the broker regime (`C<0.1`) at baseline and under each mechanism alone at its maximum tested coefficient (b=4). Error bars show replicate std.

**Paper claim** — the single-figure takeaway: **Brokerage is a first-order function of geometry; mechanism identity is a second-order refinement.**

### Evidence
- Baseline brokerage = 1.0 on all three geometries (universally preserved without mechanisms).
- At b=4 on torus-5d: all three mechanisms leave brokerage at ~0.99–1.00.
- At b=4 on torus-2d: brokerage collapses to 0.03–0.05 regardless of mechanism.
- At b=4 on Poincaré: brokerage lands at 0.07–0.11 depending on mechanism — small differences, but popularity is consistently the most aggressive and homophily the least.

### Caveats to state in the paper
1. **Brokers are defined by the budget floor.** The threshold `C<0.1` was chosen because the budget cap `k=20` sets the minimum constraint at `1/k = 0.05`; agents near this floor are structural-hole brokers by construction. Changing the budget changes the floor.
2. **Sigmoid saturation** means coefficients above ~4 produce no additional change; the "plateau" at higher coefficients is an instrument artifact, not a feature of the world. A coefficient of b=100 in the input reaches the same equilibrium as b=4.
3. **Mechanism interchangeability at high coefficients** (Fig 3) stems from the decay rule: when mechanisms are strong enough to saturate sigmoid, the final network is determined by `decay_over_budget`'s distance-based tie-breaking. All three mechanisms converge to a geometric k-nearest-neighbor graph.

---

## Suggested narrative for the paper

1. **Lead with the geometry finding** (Fig 4): brokerage is preserved in higher-dimensional Euclidean geometries and destroyed in low-dimensional / hyperbolic ones. This is a structural prediction independent of which mechanism drives the dynamics.
2. **Report the mechanism dose-response** (Fig 1) as the second-order result: where brokerage *can* be destroyed, **popularity destroys it first**, triadic second, homophily last. Attribute this ordering to the structural requirements of each mechanism — popularity acts on any existing tie; triadic needs triangles; homophily needs a local distance gradient.
3. **Show the distributional evidence** (Fig 2) to clarify what "brokerage destruction" looks like: a uniform upward shift of every agent's constraint, not a reorganization into broker and non-broker sub-populations.
4. **Use Fig 3 as a supplement** showing the absence of mechanism interactions — a methodological note that single-mechanism scans suffice to characterize the model's brokerage behavior.

## Files
- Figures (PDF + PNG, 300 DPI) at `figures/paper/`
- Source data: `simulations/paper_figures/dose_response.parquet` (162 runs), `simulations/main_factorial/summary.parquet` (243 runs)
- Generator: `make_paper_figures.py`, `run_paper_dose_response.py`
