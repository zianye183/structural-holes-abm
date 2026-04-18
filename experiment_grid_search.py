"""
Helpers for the dynamics grid-search experiment.

Pure functions plus a single-cell runner. No plotting, no notebook-only logic.
"""

from pathlib import Path

import numpy as np
from scipy import sparse

from abm_core import InitResult
from abm_dynamics import (
    Mechanism,
    mechanism_homophily,
    mechanism_triadic_closure,
    mechanism_popularity,
    constraint_attention_hard,
)
from abm_runner import SimHistory, run_simulation


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


def summarize_metrics(metrics_at_t: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute per-metric distribution summaries at one timestep.

    For each of {constraint, c_size, c_density, c_hierarchy} emits
    mean/median/std/p10/p90 (20 entries total). Also emits
    frac_constraint_lt_0.1 — the share of agents whose constraint is
    strictly below 0.1, a headcount proxy for brokerage used as the
    pilot's primary tail statistic.

    Args:
        metrics_at_t: dict with keys constraint, c_size, c_density, c_hierarchy,
                      each value a 1-D array of per-node values.

    Returns:
        dict with 21 float entries.
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
