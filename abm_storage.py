"""
Simulation serialization: save and load SimHistory to/from disk.

Format: a directory containing
    init.npz       — distance matrix, viz coords, budgets
    frames.npz     — all adjacency frames as COO edge lists
    metrics.npz    — per-node metric arrays for all frames
    metadata.json  — params, stats, init metadata
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from abm_core import InitResult
from abm_runner import SimHistory


def save_simulation(history: SimHistory, path: str) -> None:
    """Save a SimHistory to a directory on disk.

    Args:
        history: completed simulation history.
        path: directory path to save to (created if needed).
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    # Init data
    np.savez_compressed(
        out / "init.npz",
        distance_matrix=history.init_result.distance_matrix,
        viz_coords=history.init_result.viz_coords,
    )

    # Frames: store all sparse matrices as concatenated COO
    rows, cols, frame_ids = [], [], []
    for t, frame in enumerate(history.frames):
        coo = sparse.triu(frame, k=1).tocoo()
        rows.append(coo.row.astype(np.int32))
        cols.append(coo.col.astype(np.int32))
        frame_ids.append(np.full(len(coo.row), t, dtype=np.int32))

    np.savez_compressed(
        out / "frames.npz",
        rows=np.concatenate(rows) if rows else np.array([], dtype=np.int32),
        cols=np.concatenate(cols) if cols else np.array([], dtype=np.int32),
        frame_ids=np.concatenate(frame_ids) if frame_ids else np.array([], dtype=np.int32),
        n=history.init_result.n,
        n_frames=len(history.frames),
    )

    # Per-node metrics
    metric_arrays = {}
    if history.node_metrics:
        keys = list(history.node_metrics[0].keys())
        for key in keys:
            stacked = np.array([m[key] for m in history.node_metrics])
            metric_arrays[key] = stacked  # shape (n_frames, n_agents)
    np.savez_compressed(out / "metrics.npz", **metric_arrays)

    # Metadata (JSON-serializable)
    init_meta = {}
    for k, v in history.init_result.metadata.items():
        if isinstance(v, np.ndarray):
            init_meta[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            init_meta[k] = v.item()
        else:
            init_meta[k] = v

    metadata = {
        "params": history.params,
        "stats": history.stats,
        "init_metadata": init_meta,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_simulation(path: str) -> SimHistory:
    """Load a SimHistory from a directory on disk.

    Args:
        path: directory path containing saved simulation files.

    Returns:
        Reconstructed SimHistory.
    """
    src = Path(path)

    # Init data
    init_data = np.load(src / "init.npz")
    distance_matrix = init_data["distance_matrix"]
    viz_coords = init_data["viz_coords"]

    # Metadata
    with open(src / "metadata.json") as f:
        metadata = json.load(f)

    # Reconstruct numpy arrays in init metadata
    init_meta = metadata["init_metadata"]
    for k, v in init_meta.items():
        if isinstance(v, list):
            init_meta[k] = np.array(v)

    init_result = InitResult(
        distance_matrix=distance_matrix,
        viz_coords=viz_coords,
        metadata=init_meta,
    )

    # Frames
    frame_data = np.load(src / "frames.npz")
    n = int(frame_data["n"])
    n_frames = int(frame_data["n_frames"])
    all_rows = frame_data["rows"]
    all_cols = frame_data["cols"]
    all_fids = frame_data["frame_ids"]

    frames = []
    for t in range(n_frames):
        mask = all_fids == t
        r, c = all_rows[mask], all_cols[mask]
        data = np.ones(len(r), dtype=np.float64)
        upper = sparse.coo_matrix((data, (r, c)), shape=(n, n))
        full = upper + upper.T
        frames.append(sparse.csr_matrix(full))

    # Per-node metrics
    metric_data = np.load(src / "metrics.npz")
    node_metrics = []
    if len(metric_data.files) > 0:
        keys = list(metric_data.files)
        for t in range(n_frames):
            node_metrics.append({k: metric_data[k][t] for k in keys})

    return SimHistory(
        init_result=init_result,
        params=metadata["params"],
        frames=frames,
        stats=metadata["stats"],
        node_metrics=node_metrics,
    )
