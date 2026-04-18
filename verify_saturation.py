"""Verify saturation thresholds empirically.

For each (geometry, mechanism), run at a wide range of coefficients and compute:
  1. Jaccard similarity of the final network to a reference (highest-b) run.
     J = 1.0 means the network is structurally identical to the saturated run.
  2. Change in mean_constraint between consecutive b values.

The saturation coefficient is defined empirically as the smallest b for which
Jaccard to the max-b run exceeds 0.98. That is compared to the analytical
prediction b_sat = 12 * (d_20 + 0.1) for homophily, 12/5 = 2.4 for triadic,
12/budget = 0.6 for popularity.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from abm_core import init_torus_uniform, init_hyperbolic_uniform
from experiment_grid_search import run_grid_cell


N = 300
BUDGET = 20
INTERCEPT = -5.0
N_STEPS = 50
SNAPSHOT_TIMES = [0, N_STEPS]

GEOMETRIES = ["torus5d", "torus2d", "poincare"]
MECHANISMS = ("b_homophily", "b_triadic", "b_popularity")

# Wide coefficient range per mechanism, covering predicted saturation
LEVELS = {
    "b_homophily":  [0.5, 1, 2, 4, 6, 8, 12, 16, 24, 40],
    "b_triadic":    [0.5, 1, 2, 3, 4, 6, 8, 12, 20],
    "b_popularity": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0],
}

OUT_DIR = Path("simulations/saturation")
RUNS_DIR = OUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_init(geometry, seed=0):
    rng = np.random.default_rng(seed)
    if geometry == "torus5d":
        init = init_torus_uniform(n=N, d=5, rng=rng)
    elif geometry == "torus2d":
        init = init_torus_uniform(n=N, d=2, rng=rng)
    elif geometry == "poincare":
        init = init_hyperbolic_uniform(n=N, spread=1.0, rng=rng)
    return init.normalized(method="mean")


def run_one(geo, mech, level, rep=0):
    init = make_init(geo, seed=rep)
    coefs = {"b_homophily": 0.0, "b_triadic": 0.0, "b_popularity": 0.0, mech: level}
    out_path = RUNS_DIR / f"{geo}_{mech}_{level:g}_rep{rep}.npz"
    rows = run_grid_cell(
        init_result=init,
        b_homophily=coefs["b_homophily"],
        b_triadic=coefs["b_triadic"],
        b_popularity=coefs["b_popularity"],
        budget=BUDGET,
        n_steps=N_STEPS,
        snapshot_times=SNAPSHOT_TIMES,
        sim_seed=42,  # fixed: vary only b so Jaccard measures mechanism saturation not RNG
        out_path=out_path,
        intercept=INTERCEPT,
    )
    for r in rows:
        r.update(geometry=geo, mechanism=mech, level=level, rep=rep)
    return rows


def jaccard_edges(path_a, path_b, n):
    """Jaccard similarity of final-frame edge sets from two npz runs."""
    a = np.load(path_a)
    b = np.load(path_b)
    # last snapshot = highest frame_id
    last_a = a["frame_ids"].max()
    last_b = b["frame_ids"].max()
    mask_a = a["frame_ids"] == last_a
    mask_b = b["frame_ids"] == last_b
    edges_a = set(zip(a["rows"][mask_a].tolist(), a["cols"][mask_a].tolist()))
    edges_b = set(zip(b["rows"][mask_b].tolist(), b["cols"][mask_b].tolist()))
    inter = len(edges_a & edges_b)
    union = len(edges_a | edges_b)
    return inter / union if union else 0.0


def main():
    jobs = [(geo, mech, lvl)
            for geo in GEOMETRIES
            for mech in MECHANISMS
            for lvl in LEVELS[mech]]
    print(f"Running {len(jobs)} simulations...")
    Parallel(n_jobs=-1, backend="loky")(
        delayed(run_one)(*j) for j in tqdm(jobs, desc="saturation scan")
    )

    # Build results table: for each (geo, mech), compute Jaccard to max-b run
    rows = []
    for geo in GEOMETRIES:
        for mech in MECHANISMS:
            levels = LEVELS[mech]
            max_lvl = levels[-1]
            ref = RUNS_DIR / f"{geo}_{mech}_{max_lvl:g}_rep0.npz"
            for lvl in levels:
                p = RUNS_DIR / f"{geo}_{mech}_{lvl:g}_rep0.npz"
                j = jaccard_edges(p, ref, N)
                rows.append({"geometry": geo, "mechanism": mech,
                             "level": lvl, "jaccard_to_max": j})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "jaccard.parquet")

    # Predicted thresholds
    d20 = {"torus5d": 0.659, "torus2d": 0.382, "poincare": 0.753}
    predictions = {}
    for geo in GEOMETRIES:
        predictions[(geo, "b_homophily")]  = 12 * (d20[geo] + 0.1)
        predictions[(geo, "b_triadic")]    = 2.4
        predictions[(geo, "b_popularity")] = 0.6

    print("\n=== Saturation: empirical vs predicted ===")
    print(f"{'geometry':<10} {'mechanism':<14} {'b_sat_pred':>10} {'b_sat_emp':>10}  "
          f"{'J(min b -> max b)':>20}")
    for (geo, mech), pred in predictions.items():
        sub = df[(df["geometry"] == geo) & (df["mechanism"] == mech)].sort_values("level")
        # empirical: smallest b with J >= 0.98
        sat_row = sub[sub["jaccard_to_max"] >= 0.98]
        emp = sat_row["level"].min() if not sat_row.empty else float("nan")
        j_min = sub["jaccard_to_max"].iloc[0]
        print(f"{geo:<10} {mech:<14} {pred:>10.2f} {emp:>10.2f}  {j_min:>20.3f}")

    # Full curves
    print("\n=== Jaccard-to-max curves ===")
    for geo in GEOMETRIES:
        print(f"\n-- {geo} --")
        for mech in MECHANISMS:
            sub = df[(df["geometry"] == geo) & (df["mechanism"] == mech)].sort_values("level")
            curve = [f"b={l:g}: J={j:.3f}" for l, j in zip(sub["level"], sub["jaccard_to_max"])]
            print(f"  {mech}: " + " | ".join(curve))


if __name__ == "__main__":
    main()
