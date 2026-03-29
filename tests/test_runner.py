import numpy as np
from abm_core import init_torus_uniform
from abm_dynamics import mechanism_homophily, mechanism_attention_budget
from abm_runner import run_simulation, SimHistory


def test_run_basic():
    rng = np.random.default_rng(42)
    init = init_torus_uniform(n=30, d=2, rng=rng)
    mechanisms = [
        lambda s: mechanism_homophily(s, lam=5.0),
        lambda s: mechanism_attention_budget(s, beta=3.0),
    ]
    history = run_simulation(
        init_result=init,
        mechanisms=mechanisms,
        budgets=np.full(30, 8),
        n_steps=10,
        rng=np.random.default_rng(42),
    )
    assert isinstance(history, SimHistory)
    assert len(history.frames) == 11  # t=0 through t=10
    assert history.frames[0].shape == (30, 30)
    assert history.get_dense_frame(10).sum() >= history.get_dense_frame(0).sum()


def test_run_records_stats():
    rng = np.random.default_rng(42)
    init = init_torus_uniform(n=20, d=2, rng=rng)
    mechanisms = [lambda s: mechanism_homophily(s, lam=5.0)]
    history = run_simulation(
        init_result=init,
        mechanisms=mechanisms,
        budgets=np.full(20, 10),
        n_steps=5,
        rng=np.random.default_rng(42),
    )
    assert len(history.stats) == 6
    assert "mean_degree" in history.stats[0]
    assert "n_edges" in history.stats[0]
