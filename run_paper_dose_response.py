"""Paper-figure dose-response experiment.

1D marginal scans per (geometry, mechanism) at pilot-locked settings in the
non-saturated coefficient regime (sigmoid-saturation kicks in around logit>=10;
we stay below that to isolate mechanism signal from decay-dominated behavior).

Outputs:
    simulations/paper_figures/dose_response.parquet
    simulations/paper_figures/runs/*.npz

162 runs: 3 geometries x 3 mechanisms x 6 levels x 3 replicates.
"""
from itertools import product
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
SNAPSHOT_TIMES = [0, 20, 50]
REPLICATES = 3
GEOMETRIES = ["torus5d", "torus2d", "poincare"]
MECHANISMS = ("b_homophily", "b_triadic", "b_popularity")
LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

OUT_DIR = Path("simulations/paper_figures")
RUNS_DIR = OUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "dose_response.parquet"


def make_init(geometry: str, seed: int):
    rng = np.random.default_rng(seed)
    if geometry == "torus5d":
        init = init_torus_uniform(n=N, d=5, rng=rng)
    elif geometry == "torus2d":
        init = init_torus_uniform(n=N, d=2, rng=rng)
    elif geometry == "poincare":
        init = init_hyperbolic_uniform(n=N, spread=1.0, rng=rng)
    else:
        raise ValueError(geometry)
    return init.normalized(method="mean")


def build_coefs(scan_mech: str, level: float) -> dict:
    coefs = {"b_homophily": 0.0, "b_triadic": 0.0, "b_popularity": 0.0}
    coefs[scan_mech] = level
    return coefs


def run_one(geo: str, rep: int, scan_mech: str, level: float) -> list[dict]:
    init = make_init(geo, seed=rep)
    coefs = build_coefs(scan_mech, level)
    out_path = RUNS_DIR / f"{geo}_{scan_mech}_{level:g}_rep{rep}.npz"
    sim_seed = rep * 1000 + hash((geo, scan_mech, level)) % 999
    rows = run_grid_cell(
        init_result=init,
        b_homophily=coefs["b_homophily"],
        b_triadic=coefs["b_triadic"],
        b_popularity=coefs["b_popularity"],
        budget=BUDGET,
        n_steps=N_STEPS,
        snapshot_times=SNAPSHOT_TIMES,
        sim_seed=sim_seed,
        out_path=out_path,
        intercept=INTERCEPT,
    )
    for r in rows:
        r.update(geometry=geo, replicate=rep, scan_mechanism=scan_mech, level=level)
    return rows


def main() -> None:
    jobs = [(geo, rep, mech, lvl)
            for geo in GEOMETRIES
            for rep in range(REPLICATES)
            for mech in MECHANISMS
            for lvl in LEVELS]
    print(f"{len(jobs)} runs scheduled")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_one)(*j) for j in tqdm(jobs, desc="dose-response")
    )
    rows = [r for group in results for r in group]
    df = pd.DataFrame(rows)
    df.to_parquet(SUMMARY_PATH)
    print(f"Wrote {len(df)} rows -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
