"""
Interactive ABM visualization with Panel.

Run: panel serve app_panel.py --show
"""

import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import YlOrRd9
from bokeh.transform import linear_cmap

from scipy import sparse

from abm_core import (
    init_torus_uniform, init_torus_gmm,
    init_hyperbolic_uniform, init_hyperbolic_gmm,
)
from abm_dynamics import (
    mechanism_homophily, mechanism_triadic_closure,
    mechanism_attention_budget,
)
from abm_runner import run_simulation

pn.extension()

# ---------------------------------------------------------------
# Widgets: Initialization
# ---------------------------------------------------------------
geometry_select = pn.widgets.Select(
    name="Geometry", options=["Torus Uniform", "Torus GMM",
                               "Hyperbolic Uniform", "Hyperbolic GMM"],
    value="Hyperbolic Uniform",
)
n_agents = pn.widgets.IntSlider(name="N agents", start=50, end=500, step=50, value=300)
alpha_slider = pn.widgets.FloatSlider(
    name="α (hyperbolic, γ=2α+1)", start=0.3, end=1.5, step=0.1, value=0.5,
)
dim_slider = pn.widgets.IntSlider(name="d (torus dimension)", start=2, end=20, step=1, value=2)
n_clusters_slider = pn.widgets.IntSlider(name="GMM clusters", start=2, end=15, step=1, value=5)
sigma_slider = pn.widgets.FloatSlider(
    name="σ (torus GMM spread)", start=0.01, end=0.3, step=0.01, value=0.08,
)
angular_sigma_slider = pn.widgets.FloatSlider(
    name="σ_θ (hyperbolic GMM angular spread)", start=0.1, end=1.5, step=0.1, value=0.4,
)
normalize_select = pn.widgets.Select(
    name="Normalize D", options=["None", "Mean (D/mean)", "Max (D/max)"],
    value="Mean (D/mean)",
)

# ---------------------------------------------------------------
# Widgets: Mechanisms
# ---------------------------------------------------------------
toggle_homophily = pn.widgets.Checkbox(name="Homophily", value=True)
lam_slider = pn.widgets.FloatSlider(name="λ (decay rate)", start=0.1, end=00.0, step=0.1, value=3.0)

toggle_triadic = pn.widgets.Checkbox(name="Triadic Closure", value=False)
tau_slider = pn.widgets.FloatSlider(name="τ (boost per shared neighbor)", start=1.0, end=3.0, step=0.1, value=1.5)

toggle_budget = pn.widgets.Checkbox(name="Attention Budget", value=True)
beta_slider = pn.widgets.FloatSlider(name="β (budget sharpness)", start=0.5, end=10.0, step=0.5, value=3.0)
budget_slider = pn.widgets.IntSlider(name="b (budget per agent)", start=3, end=20, step=1, value=10)

# ---------------------------------------------------------------
# Widgets: Simulation
# ---------------------------------------------------------------
n_steps_slider = pn.widgets.IntSlider(name="Timesteps", start=10, end=500, step=10, value=200)
run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
player = pn.widgets.Player(
    name="Timeline", start=0, end=0, value=0, step=1,
    interval=50, loop_policy="once",
)

# ---------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------
stats_pane = pn.pane.Markdown("*Run a simulation to see stats.*")

# ---------------------------------------------------------------
# Bokeh network figure
# ---------------------------------------------------------------
node_source = ColumnDataSource(data={"x": [], "y": [], "degree": [], "size": []})
edge_source = ColumnDataSource(data={"xs": [], "ys": []})

plot = figure(
    width=650, height=650, match_aspect=True,
    tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
    background_fill_color="#1a1a2e", border_fill_color="#1a1a2e",
    outline_line_color=None,
)
plot.axis.visible = False
plot.grid.visible = False

# Boundary circle (shown for hyperbolic, hidden for torus)
_btheta = np.linspace(0, 2 * np.pi, 100)
boundary_source = ColumnDataSource(data={
    "x": np.cos(_btheta).tolist(),
    "y": np.sin(_btheta).tolist(),
})
boundary_line = plot.line("x", "y", source=boundary_source,
                          line_color="#333", line_width=1.5)

# Edges
plot.multi_line("xs", "ys", source=edge_source,
                line_color="white", line_alpha=0.08, line_width=0.5)

# Nodes
mapper = linear_cmap("degree", palette=list(reversed(YlOrRd9)), low=0, high=10)
plot.scatter("x", "y", source=node_source, size="size",
             color=mapper, alpha=0.8)

# ---------------------------------------------------------------
# Torus wrap-around edge rendering
# ---------------------------------------------------------------

def _torus_edge_segments(coords, ii, jj):
    """Convert torus edges into drawable segments, splitting wrap-around edges.

    For edges that cross the [0,1) boundary, draws two short segments:
    one from each endpoint toward the nearest boundary (Pac-Man style).
    """
    xs_all, ys_all = [], []

    for idx in range(len(ii)):
        x1, y1 = coords[ii[idx]]
        x2, y2 = coords[jj[idx]]
        dx, dy = x2 - x1, y2 - y1
        wrap_x = abs(dx) > 0.5
        wrap_y = abs(dy) > 0.5

        if not wrap_x and not wrap_y:
            xs_all.append([x1, x2])
            ys_all.append([y1, y2])
            continue

        # Virtual target: shortest-path position (may be outside [0,1])
        vx2 = x2 - np.sign(dx) if wrap_x else x2
        vy2 = y2 - np.sign(dy) if wrap_y else y2

        # Segment A → boundary: find t where ray from A toward virtual B exits [0,1]
        t_exit = _ray_exit_t(x1, y1, vx2 - x1, vy2 - y1)
        bx1 = x1 + t_exit * (vx2 - x1)
        by1 = y1 + t_exit * (vy2 - y1)
        xs_all.append([x1, bx1])
        ys_all.append([y1, by1])

        # Virtual source for B (mirror of A)
        vx1 = x1 + np.sign(dx) if wrap_x else x1
        vy1 = y1 + np.sign(dy) if wrap_y else y1

        # Segment B → boundary: find t where ray from B toward virtual A exits [0,1]
        t_exit2 = _ray_exit_t(x2, y2, vx1 - x2, vy1 - y2)
        bx2 = x2 + t_exit2 * (vx1 - x2)
        by2 = y2 + t_exit2 * (vy1 - y2)
        xs_all.append([x2, bx2])
        ys_all.append([y2, by2])

    return xs_all, ys_all


def _ray_exit_t(ox, oy, dx, dy):
    """Find parameter t > 0 where ray (ox + t*dx, oy + t*dy) exits [0, 1]."""
    t = 1.0  # fallback: full length
    if dx > 0:
        t = min(t, (1.0 - ox) / dx)
    elif dx < 0:
        t = min(t, -ox / dx)
    if dy > 0:
        t = min(t, (1.0 - oy) / dy)
    elif dy < 0:
        t = min(t, -oy / dy)
    return max(t, 0.0)


# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------
sim_history = {"frames": [], "init": None, "stats": [], "is_torus": False}


def run_sim(event):
    """Run the simulation with current widget settings."""
    rng = np.random.default_rng(42)
    n = n_agents.value

    # Initialize
    geom = geometry_select.value
    is_hyperbolic = geom.startswith("Hyperbolic")
    if geom == "Torus Uniform":
        init = init_torus_uniform(n, dim_slider.value, rng)
    elif geom == "Torus GMM":
        init = init_torus_gmm(n, dim_slider.value, n_clusters_slider.value,
                              sigma_slider.value, rng)
    elif geom == "Hyperbolic Uniform":
        init = init_hyperbolic_uniform(n, 10, rng, alpha=alpha_slider.value)
    elif geom == "Hyperbolic GMM":
        init = init_hyperbolic_gmm(n, 10, n_clusters_slider.value,
                                   angular_sigma_slider.value, rng,
                                   alpha=alpha_slider.value)
    else:
        return

    # Normalize distance matrix
    norm_choice = normalize_select.value
    if norm_choice == "Mean (D/mean)":
        init = init.normalized("mean")
    elif norm_choice == "Max (D/max)":
        init = init.normalized("max")

    # Show/hide boundary circle based on geometry
    if is_hyperbolic:
        _btheta = np.linspace(0, 2 * np.pi, 100)
        boundary_source.data = {
            "x": np.cos(_btheta).tolist(),
            "y": np.sin(_btheta).tolist(),
        }
    else:
        boundary_source.data = {"x": [], "y": []}

    # Build mechanism list
    mechanisms = []
    if toggle_homophily.value:
        lam = lam_slider.value
        mechanisms.append(lambda s, _lam=lam: mechanism_homophily(s, lam=_lam))
    if toggle_triadic.value:
        tau = tau_slider.value
        mechanisms.append(lambda s, _tau=tau: mechanism_triadic_closure(s, tau=_tau))
    if toggle_budget.value:
        beta = beta_slider.value
        mechanisms.append(lambda s, _beta=beta: mechanism_attention_budget(s, beta=_beta))

    if not mechanisms:
        stats_pane.object = "**Enable at least one mechanism.**"
        return

    # Run
    budgets = np.full(n, budget_slider.value)
    history = run_simulation(init, mechanisms, budgets, n_steps_slider.value,
                             np.random.default_rng(42))

    sim_history["frames"] = history.frames
    sim_history["init"] = init
    sim_history["stats"] = history.stats
    sim_history["is_torus"] = not is_hyperbolic

    # Update player range
    player.end = len(history.frames) - 1
    player.value = 0

    # Adjust plot range for geometry
    if is_hyperbolic:
        plot.x_range.start, plot.x_range.end = -1.15, 1.15
        plot.y_range.start, plot.y_range.end = -1.15, 1.15
    else:
        plot.x_range.start, plot.x_range.end = -0.1, 1.1
        plot.y_range.start, plot.y_range.end = -0.1, 1.1

    # Render first frame
    render_frame(0)
    stats_pane.object = f"**Simulation complete.** {n_steps_slider.value} steps, {n} agents."


def render_frame(t):
    """Render frame t on the Bokeh plot."""
    if not sim_history["frames"]:
        return

    init = sim_history["init"]
    A_sparse = sim_history["frames"][t]
    coords = init.viz_coords
    degrees = np.asarray(A_sparse.sum(axis=1)).flatten().astype(int)
    max_deg = max(degrees.max(), 1)

    # Update node data
    node_source.data = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "degree": degrees.tolist(),
        "size": (3 + (degrees / max_deg) * 8).tolist(),
    }

    # Update color mapper range
    mapper["transform"].high = int(max_deg)

    # Update edge data
    upper = sparse.triu(A_sparse, k=1).tocoo()
    ii, jj = upper.row, upper.col

    if sim_history["is_torus"]:
        # Pac-Man style: split wrap-around edges at boundaries
        xs, ys = _torus_edge_segments(coords, ii, jj)
    else:
        x0, x1 = coords[ii, 0], coords[jj, 0]
        y0, y1 = coords[ii, 1], coords[jj, 1]
        xs = np.column_stack([x0, x1]).tolist()
        ys = np.column_stack([y0, y1]).tolist()

    edge_source.data = {"xs": xs, "ys": ys}

    # Update stats
    stat = sim_history["stats"][t]
    stats_pane.object = (
        f"**t = {stat['t']}** | "
        f"Edges: {stat['n_edges']} | "
        f"Mean degree: {stat['mean_degree']:.1f} | "
        f"Max degree: {stat['max_degree']}"
    )


# Wire up callbacks
run_button.on_click(run_sim)
player.param.watch(lambda event: render_frame(event.new), "value")

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
sidebar = pn.Column(
    "## Structural Holes ABM",
    "### Initialization",
    geometry_select, n_agents, dim_slider, alpha_slider,
    n_clusters_slider, sigma_slider, angular_sigma_slider, normalize_select,
    "### Mechanisms",
    toggle_homophily, lam_slider,
    toggle_triadic, tau_slider,
    toggle_budget, beta_slider, budget_slider,
    "### Simulation",
    n_steps_slider, run_button,
    "### Playback",
    player,
    stats_pane,
    width=300,
)

main = pn.Column(plot)

template = pn.template.FastListTemplate(
    title="Structural Holes ABM",
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#e94560",
    header_background="#16213e",
)

template.servable()
