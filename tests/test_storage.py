"""Tests for simulation save/load round-trip."""

import os
import numpy as np
from abm_core import init_torus_uniform
from abm_dynamics import mechanism_homophily, constraint_attention_hard
from abm_runner import run_simulation
from abm_storage import save_simulation, load_simulation


def _run_small_sim():
    rng = np.random.default_rng(42)
    init = init_torus_uniform(n=20, d=2, rng=rng)
    mechanisms = [
        lambda s: mechanism_homophily(s, b_homophily=5.0),
    ]
    constraints = [
        lambda s: constraint_attention_hard(s),
    ]
    return run_simulation(
        init_result=init,
        mechanisms=mechanisms,
        budgets=np.full(20, 6),
        n_steps=5,
        rng=np.random.default_rng(42),
        intercept=-3.0,
        constraints=constraints,
        enable_decay=False,
    )


def test_save_load_roundtrip(tmp_path):
    original = _run_small_sim()
    save_path = str(tmp_path / "test_sim")
    save_simulation(original, save_path)

    loaded = load_simulation(save_path)

    # Params match
    assert loaded.params == original.params

    # Frame count matches
    assert len(loaded.frames) == len(original.frames)

    # Adjacency matrices match
    for t in range(len(original.frames)):
        orig_dense = original.frames[t].toarray()
        load_dense = loaded.frames[t].toarray()
        np.testing.assert_array_equal(orig_dense, load_dense)

    # Stats match
    assert len(loaded.stats) == len(original.stats)
    for t in range(len(original.stats)):
        for key in original.stats[t]:
            assert abs(loaded.stats[t][key] - original.stats[t][key]) < 1e-6

    # Node metrics match
    assert len(loaded.node_metrics) == len(original.node_metrics)
    for t in range(len(original.node_metrics)):
        for key in original.node_metrics[t]:
            np.testing.assert_allclose(
                loaded.node_metrics[t][key],
                original.node_metrics[t][key],
                atol=1e-10,
            )

    # Init result preserved
    np.testing.assert_allclose(
        loaded.init_result.distance_matrix,
        original.init_result.distance_matrix,
    )
    np.testing.assert_allclose(
        loaded.init_result.viz_coords,
        original.init_result.viz_coords,
    )


def test_saved_files_exist(tmp_path):
    history = _run_small_sim()
    save_path = str(tmp_path / "test_sim2")
    save_simulation(history, save_path)

    assert os.path.exists(os.path.join(save_path, "init.npz"))
    assert os.path.exists(os.path.join(save_path, "frames.npz"))
    assert os.path.exists(os.path.join(save_path, "metrics.npz"))
    assert os.path.exists(os.path.join(save_path, "metadata.json"))
