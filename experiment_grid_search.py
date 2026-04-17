"""
Helpers for the dynamics grid-search experiment.

Pure functions plus a single-cell runner. No plotting, no notebook-only logic.
"""

import numpy as np

from abm_dynamics import (
    Mechanism,
    mechanism_homophily,
    mechanism_triadic_closure,
    mechanism_popularity,
)
from abm_runner import SimHistory


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
