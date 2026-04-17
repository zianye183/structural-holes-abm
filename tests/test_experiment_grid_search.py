import numpy as np
import pytest
from abm_core import init_torus_uniform
from abm_dynamics import SimState

from experiment_grid_search import build_mechanisms


def _make_state(n=10, d=2, seed=0):
    rng = np.random.default_rng(seed)
    init = init_torus_uniform(n=n, d=d, rng=rng)
    return SimState(D=init.distance_matrix, budgets=np.full(n, 5))


def test_build_mechanisms_omits_zero_coefs():
    mechs = build_mechanisms(b_homophily=0.0, b_triadic=0.0, b_popularity=0.0)
    assert mechs == []


def test_build_mechanisms_keeps_positive_coefs():
    mechs = build_mechanisms(b_homophily=1.0, b_triadic=0.0, b_popularity=0.6)
    assert len(mechs) == 2


def test_build_mechanisms_callables_return_correct_shape():
    state = _make_state(n=10)
    mechs = build_mechanisms(b_homophily=2.0, b_triadic=1.0, b_popularity=0.5)
    for mech in mechs:
        out = mech(state)
        assert out.shape == (10, 10)
        assert out.dtype == np.float64


def test_build_mechanisms_late_binding_safe():
    # If b were captured by reference instead of by default-arg value,
    # both closures would end up using the same coefficient. Verify they don't.
    state = _make_state(n=8)
    mechs_low = build_mechanisms(b_homophily=1.0, b_triadic=0.0, b_popularity=0.0)
    mechs_high = build_mechanisms(b_homophily=10.0, b_triadic=0.0, b_popularity=0.0)
    out_low = mechs_low[0](state)
    out_high = mechs_high[0](state)
    assert not np.allclose(out_low, out_high)


from abm_dynamics import mechanism_homophily
from abm_runner import run_simulation

from experiment_grid_search import extract_snapshot_metrics


def _tiny_history(n_steps=10):
    rng = np.random.default_rng(0)
    init = init_torus_uniform(n=15, d=2, rng=rng)
    return run_simulation(
        init_result=init,
        mechanisms=[lambda s: mechanism_homophily(s, b_homophily=4.0)],
        budgets=np.full(15, 5),
        n_steps=n_steps,
        rng=np.random.default_rng(1),
        intercept=-3.0,
    )


def test_extract_snapshot_metrics_returns_correct_times():
    history = _tiny_history(n_steps=10)
    times = [0, 5, 10]
    snap = extract_snapshot_metrics(history, times)
    assert list(snap["times"]) == times


def test_extract_snapshot_metrics_array_shapes():
    history = _tiny_history(n_steps=10)
    times = [0, 5, 10]
    snap = extract_snapshot_metrics(history, times)
    for key in ("constraint", "c_size", "c_density", "c_hierarchy"):
        assert snap[key].shape == (3, 15)


def test_extract_snapshot_metrics_values_match_history():
    history = _tiny_history(n_steps=10)
    snap = extract_snapshot_metrics(history, [0, 10])
    assert np.allclose(snap["constraint"][0], history.node_metrics[0]["constraint"])
    assert np.allclose(snap["constraint"][1], history.node_metrics[10]["constraint"])
