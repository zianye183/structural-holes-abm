"""
Interactive ABM visualization with Panel — 2x2 multi-seed grid.

Run: panel serve app_panel.py --show
"""

import datetime

import numpy as np
import networkx as nx
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import YlOrRd9, Category20_20, RdYlGn11

from scipy import sparse

from abm_core import (
    init_torus_uniform, init_torus_gmm,
    init_hyperbolic_uniform, init_hyperbolic_gmm,
    burt_constraint, burt_constraint_decomposed,
)
from abm_dynamics import (
    mechanism_homophily, mechanism_triadic_closure,
    mechanism_popularity, mechanism_attention_budget,
    mechanism_attention_hard,
)
from abm_runner import run_simulation
from abm_storage import save_simulation

pn.extension(raw_css=["""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Global typography ── */
body, .bk, .pn-wrapper {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
}

/* ── Sidebar ── */
#sidebar {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* ── Widget labels ── */
.bk-input-group > label, .bk label {
    font-size: 11px !important;
    color: #475569 !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* ── Slider track ── */
.noUi-connect {
    background: #2563eb !important;
}
.noUi-handle {
    border-color: #2563eb !important;
}
.noUi-target {
    background: #e2e8f0 !important;
    border-color: #cbd5e1 !important;
}

/* ── Buttons ── */
.bk-btn-primary {
    background-color: #1e40af !important;
    border-color: #1e40af !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.bk-btn-primary:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
.bk-btn-default {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
}
.bk-btn-default:hover {
    border-color: #94a3b8 !important;
    color: #1e293b !important;
    background-color: #f1f5f9 !important;
}

/* ── Stats bar ── */
.stats-bar {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 10px 16px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #475569 !important;
    line-height: 1.6 !important;
}
.stats-bar strong {
    color: #1e40af !important;
}

/* ── Legend bar ── */
.legend-bar {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
}

/* ── Header ── */
#header {
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 1px !important;
}

/* ── Player widget ── */
.bk-Player button {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
    color: #475569 !important;
}
.bk-Player button:hover {
    background: #e2e8f0 !important;
    color: #1e293b !important;
}

/* ── Select / Input ── */
.bk-input {
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #1e293b !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}
.bk-input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.2) !important;
}

/* ── Checkbox ── */
input[type="checkbox"] {
    accent-color: #2563eb !important;
}

/* ── Section divider ── */
.sidebar-section-title {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: #94a3b8 !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding-bottom: 4px !important;
    margin-top: 12px !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
}

/* ── Main content ── */
.main .bk-Column {
    background: transparent !important;
}
"""])

# ── Color palette (light theme) ──
BG_PLOT = "#ffffff"
BG_BORDER = "#f8fafc"
GRID_COLOR = "#e2e8f0"
TITLE_COLOR = "#334155"
LABEL_COLOR = "#64748b"
TICK_COLOR = "#94a3b8"
ACCENT = "#2563eb"
BOUNDARY_COLOR = "#cbd5e1"
EDGE_COLOR = "#475569"
NODE_ALPHA = 0.9

TS_CSIZE = "#0891b2"
TS_CDENSITY = "#e11d48"
TS_CHIERARCHY = "#d97706"

NUM_PANELS = 4
DEFAULT_SEEDS = [42, 43, 44, 45]

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
    name="\u03b1 (hyperbolic, \u03b3=2\u03b1+1)", start=0.3, end=1.5, step=0.1, value=0.5,
)
dim_slider = pn.widgets.IntSlider(name="d (torus dimension)", start=2, end=20, step=1, value=2)
n_clusters_slider = pn.widgets.IntSlider(name="GMM clusters", start=2, end=15, step=1, value=5)
sigma_slider = pn.widgets.FloatSlider(
    name="\u03c3 (torus GMM spread)", start=0.01, end=0.3, step=0.01, value=0.08,
)
angular_sigma_slider = pn.widgets.FloatSlider(
    name="\u03c3_\u03b8 (hyperbolic GMM angular spread)", start=0.1, end=1.5, step=0.1, value=0.4,
)
normalize_select = pn.widgets.Select(
    name="Normalize D", options=["None", "Mean (D/mean)", "Max (D/max)"],
    value="Mean (D/mean)",
)

# ---------------------------------------------------------------
# Widgets: Seeds (one per panel)
# ---------------------------------------------------------------
seed_inputs = [
    pn.widgets.IntInput(name=f"Seed {i+1}", value=DEFAULT_SEEDS[i], step=1)
    for i in range(NUM_PANELS)
]

# ---------------------------------------------------------------
# Widgets: Mechanisms
# ---------------------------------------------------------------
toggle_homophily = pn.widgets.Checkbox(name="Homophily", value=True)
lam_slider = pn.widgets.FloatSlider(
    name="\u03bb (decay rate)", start=0.1, end=100.0, step=0.1, value=3.0,
)

toggle_triadic = pn.widgets.Checkbox(name="Triadic Closure", value=False)
tau_slider = pn.widgets.FloatSlider(
    name="\u03c4 (boost per shared neighbor)", start=1.0, end=3.0, step=0.1, value=1.5,
)

toggle_popularity = pn.widgets.Checkbox(name="Popularity", value=False)
mu_slider = pn.widgets.FloatSlider(
    name="\u03bc (popularity exponent)", start=0.0, end=2.0, step=0.05, value=0.5,
)

attention_mode = pn.widgets.Select(
    name="Attention Budget",
    options=["Off", "Hard cutoff", "Sigmoid + decay"],
    value="Hard cutoff",
)
beta_slider = pn.widgets.FloatSlider(
    name="\u03b2 (budget sharpness)", start=0.5, end=10.0, step=0.5, value=3.0,
)
budget_slider = pn.widgets.IntSlider(
    name="b (budget per agent)", start=3, end=20, step=1, value=10,
)

# ---------------------------------------------------------------
# Widgets: Visualization
# ---------------------------------------------------------------
node_color_select = pn.widgets.Select(
    name="Node color",
    options=["Degree", "Burt's Constraint", "Community (Louvain)", "Clustering Coeff"],
    value="Degree",
)
edge_opacity_select = pn.widgets.Select(
    name="Edge opacity", options=["Uniform", "Bridge highlight"],
    value="Uniform",
)

# ---------------------------------------------------------------
# Widgets: Simulation
# ---------------------------------------------------------------
n_steps_slider = pn.widgets.IntSlider(name="Timesteps", start=10, end=500, step=10, value=200)
run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
save_button = pn.widgets.Button(name="Save Simulation", button_type="default")
player = pn.widgets.Player(
    name="Timeline", start=0, end=0, value=0, step=1,
    interval=50, loop_policy="once",
)

# ---------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------
stats_pane = pn.pane.HTML(
    '<div class="stats-bar" style="color:#475569; font-style:italic;">'
    'Run a simulation to see stats.</div>',
)


# ---------------------------------------------------------------
# Per-panel Bokeh network figures
# ---------------------------------------------------------------

def _make_panel_figure(seed_label):
    """Create a Bokeh figure and data sources for one panel."""
    node_src = ColumnDataSource(data={
        "x": [], "y": [], "color_val": [], "size": [], "node_color": [],
    })
    edge_src = ColumnDataSource(data={"xs": [], "ys": [], "alpha": []})

    p = figure(
        width=420, height=420, match_aspect=True,
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        background_fill_color=BG_PLOT, border_fill_color=BG_BORDER,
        outline_line_color=None,
        title=f"Seed {seed_label}",
    )
    p.title.text_color = TITLE_COLOR
    p.title.text_font_size = "10pt"
    p.title.text_font_style = "normal"
    p.axis.visible = False
    p.grid.visible = False
    p.min_border = 8

    _btheta = np.linspace(0, 2 * np.pi, 100)
    boundary_src = ColumnDataSource(data={
        "x": np.cos(_btheta).tolist(),
        "y": np.sin(_btheta).tolist(),
    })
    p.line("x", "y", source=boundary_src, line_color=BOUNDARY_COLOR,
           line_width=1.5, line_dash="dotted")
    p.multi_line("xs", "ys", source=edge_src,
                 line_color=EDGE_COLOR, line_alpha="alpha", line_width=0.6)
    p.scatter("x", "y", source=node_src, size="size",
              color="node_color", alpha=NODE_ALPHA)

    return {
        "plot": p,
        "node_source": node_src,
        "edge_source": edge_src,
        "boundary_source": boundary_src,
    }


panels = [_make_panel_figure(DEFAULT_SEEDS[i]) for i in range(NUM_PANELS)]


# ---------------------------------------------------------------
# Per-panel time-series + histogram charts
# ---------------------------------------------------------------

def _make_ts_chart(seed_label):
    """Create a time-series constraint decomposition chart for one panel."""
    ts_src = ColumnDataSource(data={
        "t": [], "c_size": [], "c_density": [], "c_hierarchy": [], "total": [],
    })
    cursor_src = ColumnDataSource(data={"t": [0, 0], "y": [0, 1]})

    p = figure(
        width=420, height=140,
        tools="", background_fill_color=BG_PLOT, border_fill_color=BG_BORDER,
        outline_line_color=None,
        title=f"Constraint Decomposition — Seed {seed_label}",
    )
    p.title.text_color = TITLE_COLOR
    p.title.text_font_size = "9pt"
    p.title.text_font_style = "normal"
    p.xaxis.axis_label = "t"
    p.xaxis.axis_label_text_color = LABEL_COLOR
    p.xaxis.axis_label_text_font_size = "9pt"
    p.xaxis.major_label_text_color = TICK_COLOR
    p.xaxis.major_label_text_font_size = "8pt"
    p.xaxis.axis_line_color = GRID_COLOR
    p.xaxis.major_tick_line_color = GRID_COLOR
    p.xaxis.minor_tick_line_color = None
    p.yaxis.axis_label = "Mean C"
    p.yaxis.axis_label_text_color = LABEL_COLOR
    p.yaxis.axis_label_text_font_size = "9pt"
    p.yaxis.major_label_text_color = TICK_COLOR
    p.yaxis.major_label_text_font_size = "8pt"
    p.yaxis.axis_line_color = GRID_COLOR
    p.yaxis.major_tick_line_color = GRID_COLOR
    p.yaxis.minor_tick_line_color = None
    p.grid.grid_line_color = GRID_COLOR
    p.grid.grid_line_alpha = 0.5
    p.min_border = 8

    p.varea_stack(
        ["c_size", "c_density", "c_hierarchy"],
        x="t", source=ts_src,
        color=[TS_CSIZE, TS_CDENSITY, TS_CHIERARCHY],
        alpha=0.65,
    )
    p.line("t", "total", source=ts_src, line_color="#1e293b",
           line_width=1.5, line_alpha=0.85)
    p.line("t", "y", source=cursor_src,
           line_color=ACCENT, line_width=2, line_dash="dashed")

    return {"plot": p, "ts_source": ts_src, "cursor_source": cursor_src}


def _make_hist_chart(seed_label):
    """Create a constraint distribution histogram for one panel."""
    hist_src = ColumnDataSource(data={
        "left": [], "right": [], "top": [], "color": [],
    })

    p = figure(
        width=420, height=140,
        tools="", background_fill_color=BG_PLOT, border_fill_color=BG_BORDER,
        outline_line_color=None,
        title=f"C_i Distribution — Seed {seed_label}",
    )
    p.title.text_color = TITLE_COLOR
    p.title.text_font_size = "9pt"
    p.title.text_font_style = "normal"
    p.xaxis.axis_label = "C_i"
    p.xaxis.axis_label_text_color = LABEL_COLOR
    p.xaxis.axis_label_text_font_size = "9pt"
    p.xaxis.major_label_text_color = TICK_COLOR
    p.xaxis.major_label_text_font_size = "8pt"
    p.xaxis.axis_line_color = GRID_COLOR
    p.xaxis.major_tick_line_color = GRID_COLOR
    p.xaxis.minor_tick_line_color = None
    p.yaxis.axis_label = "Count"
    p.yaxis.axis_label_text_color = LABEL_COLOR
    p.yaxis.axis_label_text_font_size = "9pt"
    p.yaxis.major_label_text_color = TICK_COLOR
    p.yaxis.major_label_text_font_size = "8pt"
    p.yaxis.axis_line_color = GRID_COLOR
    p.yaxis.major_tick_line_color = GRID_COLOR
    p.yaxis.minor_tick_line_color = None
    p.grid.grid_line_color = GRID_COLOR
    p.grid.grid_line_alpha = 0.5
    p.min_border = 8

    p.quad(
        top="top", bottom=0, left="left", right="right",
        source=hist_src, fill_color="color", line_color=BG_PLOT,
        alpha=0.85,
    )

    return {"plot": p, "hist_source": hist_src}


ts_charts = [_make_ts_chart(DEFAULT_SEEDS[i]) for i in range(NUM_PANELS)]
hist_charts = [_make_hist_chart(DEFAULT_SEEDS[i]) for i in range(NUM_PANELS)]

ts_legend_pane = pn.pane.HTML(
    '<div class="legend-bar">'
    f'<span style="color:{TS_CSIZE};">\u25cf</span> '
    '<span style="color:#475569;">C<sub>size</sub></span>'
    '&nbsp;&nbsp;&nbsp;'
    f'<span style="color:{TS_CDENSITY};">\u25cf</span> '
    '<span style="color:#475569;">C<sub>density</sub></span>'
    '&nbsp;&nbsp;&nbsp;'
    f'<span style="color:{TS_CHIERARCHY};">\u25cf</span> '
    '<span style="color:#475569;">C<sub>hierarchy</sub></span>'
    '&nbsp;&nbsp;&nbsp;'
    '<span style="color:#1e293b;">\u2014</span> '
    '<span style="color:#475569;">Total</span>'
    '</div>',
)


# ---------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------

def _vals_to_heat_colors(vals, palette=list(reversed(YlOrRd9))):
    v = np.asarray(vals, dtype=float)
    vmax = max(v.max(), 1e-9)
    normed = np.clip(v / vmax, 0, 1)
    indices = (normed * (len(palette) - 1)).astype(int)
    return [palette[i] for i in indices]


def _vals_to_diverging_colors(vals, palette=RdYlGn11):
    v = np.asarray(vals, dtype=float)
    vmax = max(v.max(), 1e-9)
    normed = np.clip(v / vmax, 0, 1)
    indices = (normed * (len(palette) - 1)).astype(int)
    return [palette[i] for i in indices]


def _community_colors(labels):
    palette = Category20_20
    return [palette[int(l) % len(palette)] for l in labels]


# ---------------------------------------------------------------
# Torus wrap-around edge rendering
# ---------------------------------------------------------------

def _torus_edge_segments(coords, ii, jj):
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
        vx2 = x2 - np.sign(dx) if wrap_x else x2
        vy2 = y2 - np.sign(dy) if wrap_y else y2
        t_exit = _ray_exit_t(x1, y1, vx2 - x1, vy2 - y1)
        bx1 = x1 + t_exit * (vx2 - x1)
        by1 = y1 + t_exit * (vy2 - y1)
        xs_all.append([x1, bx1])
        ys_all.append([y1, by1])
        vx1 = x1 + np.sign(dx) if wrap_x else x1
        vy1 = y1 + np.sign(dy) if wrap_y else y1
        t_exit2 = _ray_exit_t(x2, y2, vx1 - x2, vy1 - y2)
        bx2 = x2 + t_exit2 * (vx1 - x2)
        by2 = y2 + t_exit2 * (vy1 - y2)
        xs_all.append([x2, bx2])
        ys_all.append([y2, by2])
    return xs_all, ys_all


def _ray_exit_t(ox, oy, dx, dy):
    t = 1.0
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
# Reference diagrams: broker vs. clique
# ---------------------------------------------------------------

def _make_ref_figure(title):
    node_src = ColumnDataSource(data={
        "x": [], "y": [], "size": [], "node_color": [], "label": [],
    })
    edge_src = ColumnDataSource(data={"xs": [], "ys": []})
    p = figure(
        width=280, height=210, tools="",
        background_fill_color=BG_PLOT, border_fill_color=BG_BORDER,
        outline_line_color=None, title=title,
    )
    p.title.text_color = TITLE_COLOR
    p.title.text_font_size = "9pt"
    p.title.text_font_style = "normal"
    p.axis.visible = False
    p.grid.visible = False
    p.min_border = 8
    p.multi_line("xs", "ys", source=edge_src,
                 line_color=EDGE_COLOR, line_alpha=0.3, line_width=1.5)
    p.scatter("x", "y", source=node_src, size="size",
              color="node_color", alpha=0.9)
    p.text("x", "y", text="label", source=node_src,
           text_color="#475569", text_font_size="7pt",
           text_align="center", text_baseline="middle",
           x_offset=0, y_offset=-14)
    return {"plot": p, "node_source": node_src, "edge_source": edge_src}


def _render_ref(ref_fig, G, pos, color_mode):
    n = G.number_of_nodes()
    degrees = np.array([G.degree(i) for i in range(n)])
    max_deg = max(degrees.max(), 1)
    if color_mode == "Burt's Constraint":
        c_vals = burt_constraint(G)
        vals = c_vals
        colors = _vals_to_diverging_colors(vals)
        labels = [f"C={v:.2f}" for v in vals]
    elif color_mode == "Community (Louvain)":
        comms = nx.community.louvain_communities(G, seed=42)
        lab = np.zeros(n, dtype=int)
        for ci, comm in enumerate(comms):
            for node in comm:
                lab[node] = ci
        colors = _community_colors(lab)
        labels = [f"c{lab[i]}" for i in range(n)]
    elif color_mode == "Clustering Coeff":
        cc = nx.clustering(G)
        vals = np.array([cc[i] for i in range(n)])
        colors = _vals_to_diverging_colors(vals)
        labels = [f"cc={vals[i]:.1f}" for i in range(n)]
    else:
        colors = _vals_to_heat_colors(degrees)
        labels = [f"k={degrees[i]}" for i in range(n)]
    ref_fig["node_source"].data = {
        "x": [pos[i][0] for i in range(n)],
        "y": [pos[i][1] for i in range(n)],
        "size": (7 + (degrees / max_deg) * 7).tolist(),
        "node_color": colors,
        "label": labels,
    }
    ref_fig["edge_source"].data = {
        "xs": [[pos[u][0], pos[v][0]] for u, v in G.edges()],
        "ys": [[pos[u][1], pos[v][1]] for u, v in G.edges()],
    }


# Build reference graphs
G_broker = nx.Graph()
G_broker.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)])
pos_broker = {0: (0, 0), 1: (-1, 0.6), 2: (-1, -0.6), 3: (1, 0.6), 4: (1, -0.6)}

G_clique = nx.complete_graph(5)
_angle = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 5, endpoint=False)
pos_clique = {i: (float(np.cos(a)), float(np.sin(a))) for i, a in enumerate(_angle)}

ref_broker = _make_ref_figure("Broker (structural hole)")
ref_clique = _make_ref_figure("Clique (no hole)")


def _update_references(event=None):
    mode = node_color_select.value
    _render_ref(ref_broker, G_broker, pos_broker, mode)
    _render_ref(ref_clique, G_clique, pos_clique, mode)


_update_references()


# ---------------------------------------------------------------
# Per-panel simulation state
# ---------------------------------------------------------------
sim_states = [
    {
        "history": None,
        "is_torus": False,
        "constraint_cache": {},
        "community_cache": {},
        "clustering_cache": {},
    }
    for _ in range(NUM_PANELS)
]


# ---------------------------------------------------------------
# Cached per-frame computations (panel-aware)
# ---------------------------------------------------------------

def _get_constraint(pi, t):
    cache = sim_states[pi]["constraint_cache"]
    if t not in cache:
        nm = sim_states[pi]["history"].node_metrics[t]
        cache[t] = nm["constraint"]
    return cache[t]


def _get_communities(pi, t):
    cache = sim_states[pi]["community_cache"]
    if t not in cache:
        A_sparse = sim_states[pi]["history"].frames[t]
        G = nx.from_scipy_sparse_array(A_sparse)
        communities = nx.community.louvain_communities(G, seed=42)
        labels = np.zeros(G.number_of_nodes(), dtype=int)
        for idx, comm in enumerate(communities):
            for node in comm:
                labels[node] = idx
        cache[t] = labels
    return cache[t]


def _get_clustering(pi, t):
    cache = sim_states[pi]["clustering_cache"]
    if t not in cache:
        A_sparse = sim_states[pi]["history"].frames[t]
        G = nx.from_scipy_sparse_array(A_sparse)
        cc = nx.clustering(G)
        cache[t] = np.array([cc[i] for i in range(G.number_of_nodes())])
    return cache[t]


# ---------------------------------------------------------------
# Init + mechanism helpers
# ---------------------------------------------------------------

def _build_init(geom, n, rng):
    """Build an InitResult for the given geometry and rng."""
    if geom == "Torus Uniform":
        return init_torus_uniform(n, dim_slider.value, rng)
    elif geom == "Torus GMM":
        return init_torus_gmm(n, dim_slider.value, n_clusters_slider.value,
                              sigma_slider.value, rng)
    elif geom == "Hyperbolic Uniform":
        return init_hyperbolic_uniform(n, 10, rng, alpha=alpha_slider.value)
    elif geom == "Hyperbolic GMM":
        return init_hyperbolic_gmm(n, 10, n_clusters_slider.value,
                                   angular_sigma_slider.value, rng,
                                   alpha=alpha_slider.value)
    return None


def _build_mechanisms():
    """Build the mechanism list from current widget state."""
    mechanisms = []
    if toggle_homophily.value:
        lam = lam_slider.value
        mechanisms.append(lambda s, _lam=lam: mechanism_homophily(s, lam=_lam))
    if toggle_triadic.value:
        tau = tau_slider.value
        mechanisms.append(lambda s, _tau=tau: mechanism_triadic_closure(s, tau=_tau))
    if toggle_popularity.value:
        mu = mu_slider.value
        mechanisms.append(lambda s, _mu=mu: mechanism_popularity(s, mu=_mu))

    att = attention_mode.value
    enable_decay = False
    if att == "Hard cutoff":
        mechanisms.append(lambda s: mechanism_attention_hard(s))
    elif att == "Sigmoid + decay":
        beta = beta_slider.value
        mechanisms.append(lambda s, _beta=beta: mechanism_attention_budget(s, beta=_beta))
        enable_decay = True

    return mechanisms, enable_decay


# ---------------------------------------------------------------
# Run simulation (all 4 panels)
# ---------------------------------------------------------------

def run_sim(event):
    """Run 4 simulations with current widget settings, one per init seed."""
    geom = geometry_select.value
    is_hyperbolic = geom.startswith("Hyperbolic")
    n = n_agents.value

    mechanisms, enable_decay = _build_mechanisms()
    if not mechanisms:
        stats_pane.object = '<div class="stats-bar" style="color:#f43f5e;">Enable at least one mechanism.</div>'
        return

    budgets = np.full(n, budget_slider.value)
    norm_choice = normalize_select.value

    for pi in range(NUM_PANELS):
        seed = seed_inputs[pi].value
        init_rng = np.random.default_rng(seed)
        init = _build_init(geom, n, init_rng)
        if init is None:
            return

        if norm_choice == "Mean (D/mean)":
            init = init.normalized("mean")
        elif norm_choice == "Max (D/max)":
            init = init.normalized("max")

        # Same dynamics seed for all panels
        sim_rng = np.random.default_rng(42)
        history = run_simulation(init, mechanisms, budgets, n_steps_slider.value,
                                 sim_rng, enable_decay=enable_decay)

        sim_states[pi]["history"] = history
        sim_states[pi]["is_torus"] = not is_hyperbolic
        sim_states[pi]["constraint_cache"] = {}
        sim_states[pi]["community_cache"] = {}
        sim_states[pi]["clustering_cache"] = {}

        # Update boundary circle visibility
        panel = panels[pi]
        if is_hyperbolic:
            _btheta = np.linspace(0, 2 * np.pi, 100)
            panel["boundary_source"].data = {
                "x": np.cos(_btheta).tolist(),
                "y": np.sin(_btheta).tolist(),
            }
        else:
            panel["boundary_source"].data = {"x": [], "y": []}

        # Adjust plot range
        p = panel["plot"]
        if is_hyperbolic:
            p.x_range.start, p.x_range.end = -1.15, 1.15
            p.y_range.start, p.y_range.end = -1.15, 1.15
        else:
            p.x_range.start, p.x_range.end = -0.1, 1.1
            p.y_range.start, p.y_range.end = -0.1, 1.1

        # Update title with current seed value
        p.title.text = f"Seed {seed}"

    # Update player range
    player.end = len(sim_states[0]["history"].frames) - 1
    player.value = 0

    # Update time-series charts (all panels)
    for pi in range(NUM_PANELS):
        _update_timeseries(pi)
        ts_charts[pi]["plot"].title.text = f"Constraint Decomposition — Seed {seed_inputs[pi].value}"
        hist_charts[pi]["plot"].title.text = f"C_i Distribution — Seed {seed_inputs[pi].value}"

    render_all_frames(0)
    stats_pane.object = (
        '<div class="stats-bar">'
        f'<strong>Simulation complete.</strong> '
        f'{n_steps_slider.value} steps \u00b7 {n} agents \u00b7 4 seeds'
        '</div>'
    )


# ---------------------------------------------------------------
# Time-series + histogram (driven by panel 0)
# ---------------------------------------------------------------

def _update_timeseries(pi):
    """Populate the time-series chart for panel pi."""
    history = sim_states[pi]["history"]
    if history is None:
        return
    chart = ts_charts[pi]
    stats = history.stats
    ts = [s["t"] for s in stats]
    c_size = [s["mean_c_size"] for s in stats]
    c_density = [s["mean_c_density"] for s in stats]
    c_hierarchy = [s["mean_c_hierarchy"] for s in stats]
    total = [s["mean_constraint"] for s in stats]
    chart["ts_source"].data = {
        "t": ts, "c_size": c_size, "c_density": c_density,
        "c_hierarchy": c_hierarchy, "total": total,
    }
    y_max = max(max(total) if total else 1, 1)
    chart["plot"].y_range.start = 0
    chart["plot"].y_range.end = y_max * 1.1


def _update_histogram(pi, t):
    """Update the histogram for panel pi at frame t."""
    history = sim_states[pi]["history"]
    if history is None:
        return
    chart = hist_charts[pi]
    c_vals = history.node_metrics[t]["constraint"]
    valid = c_vals
    if len(valid) == 0:
        chart["hist_source"].data = {"left": [], "right": [], "top": [], "color": []}
        return
    n_bins = 20
    counts, edges = np.histogram(valid, bins=n_bins)
    palette = list(reversed(YlOrRd9))
    bin_centers = (edges[:-1] + edges[1:]) / 2
    max_center = max(bin_centers.max(), 1e-9)
    normed = np.clip(bin_centers / max_center, 0, 1)
    colors = [palette[int(v * (len(palette) - 1))] for v in normed]
    chart["hist_source"].data = {
        "left": edges[:-1].tolist(),
        "right": edges[1:].tolist(),
        "top": counts.tolist(),
        "color": colors,
    }


def _update_cursor(pi, t):
    """Move the time cursor on the time-series chart for panel pi."""
    chart = ts_charts[pi]
    ts_data = chart["ts_source"].data
    if not ts_data["t"]:
        return
    total = ts_data.get("total", [])
    y_max = max(total) * 1.1 if total else 1
    chart["cursor_source"].data = {"t": [t, t], "y": [0, y_max]}


# ---------------------------------------------------------------
# Render single panel
# ---------------------------------------------------------------

def _render_panel_frame(pi, t):
    """Render frame t for a single panel."""
    hist = sim_states[pi]
    if hist["history"] is None:
        return

    panel = panels[pi]
    history = hist["history"]
    init = history.init_result
    A_sparse = history.frames[t]
    coords = init.viz_coords
    degrees = np.asarray(A_sparse.sum(axis=1)).flatten().astype(int)
    max_deg = max(degrees.max(), 1)

    # --- Node coloring ---
    color_mode = node_color_select.value
    if color_mode == "Burt's Constraint":
        c_vals = _get_constraint(pi, t)
        color_val = np.nan_to_num(c_vals, nan=0.0)
        node_colors = _vals_to_diverging_colors(color_val)
    elif color_mode == "Community (Louvain)":
        labels = _get_communities(pi, t)
        color_val = labels.astype(float)
        node_colors = _community_colors(labels)
    elif color_mode == "Clustering Coeff":
        cc = _get_clustering(pi, t)
        color_val = cc
        node_colors = _vals_to_diverging_colors(cc)
    else:
        color_val = degrees.astype(float)
        node_colors = _vals_to_heat_colors(degrees)

    panel["node_source"].data = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "color_val": color_val.tolist(),
        "size": (3 + (degrees / max_deg) * 8).tolist(),
        "node_color": node_colors,
    }

    # --- Edge data ---
    upper = sparse.triu(A_sparse, k=1).tocoo()
    ii, jj = upper.row, upper.col

    if hist["is_torus"]:
        xs, ys = _torus_edge_segments(coords, ii, jj)
    else:
        x0, x1 = coords[ii, 0], coords[jj, 0]
        y0, y1 = coords[ii, 1], coords[jj, 1]
        xs = np.column_stack([x0, x1]).tolist()
        ys = np.column_stack([y0, y1]).tolist()

    # --- Edge opacity ---
    opacity_mode = edge_opacity_select.value
    if opacity_mode == "Bridge highlight" and len(ii) > 0:
        A_dense = A_sparse.toarray() if sparse.issparse(A_sparse) else A_sparse
        shared = A_dense @ A_dense
        shared_counts = np.array([shared[i, j] for i, j in zip(ii, jj)])
        max_shared = max(shared_counts.max(), 1)
        edge_alpha = 0.03 + 0.35 * (1.0 - shared_counts / max_shared)
        if hist["is_torus"]:
            alpha_list = []
            for idx in range(len(ii)):
                x1, y1 = coords[ii[idx]]
                x2, y2 = coords[jj[idx]]
                dx, dy = x2 - x1, y2 - y1
                wraps = abs(dx) > 0.5 or abs(dy) > 0.5
                a = float(edge_alpha[idx])
                alpha_list.extend([a, a] if wraps else [a])
            alpha_out = alpha_list
        else:
            alpha_out = edge_alpha.tolist()
    else:
        alpha_out = [0.25] * len(xs)

    panel["edge_source"].data = {"xs": xs, "ys": ys, "alpha": alpha_out}


# ---------------------------------------------------------------
# Render all panels + shared charts
# ---------------------------------------------------------------

def render_all_frames(t):
    """Render frame t across all 4 panels and update per-panel charts/stats."""
    for pi in range(NUM_PANELS):
        _render_panel_frame(pi, t)
        _update_histogram(pi, t)
        _update_cursor(pi, t)

    # Stats summary from panel 0
    history = sim_states[0]["history"]
    if history is None:
        return
    stat = history.stats[t]
    color_mode = node_color_select.value
    extra = ""
    if color_mode == "Burt's Constraint":
        c_vals = _get_constraint(0, t)
        extra = f'<span style="color:#64748b;">\u2502</span> Mean C = <strong>{c_vals.mean():.3f}</strong>'
    elif color_mode == "Community (Louvain)":
        labels = _get_communities(0, t)
        n_communities = len(set(labels))
        extra = f'<span style="color:#64748b;">\u2502</span> Communities = <strong>{n_communities}</strong>'
    elif color_mode == "Clustering Coeff":
        cc = _get_clustering(0, t)
        extra = f'<span style="color:#64748b;">\u2502</span> Mean CC = <strong>{cc.mean():.3f}</strong>'
    stats_pane.object = (
        '<div class="stats-bar">'
        f'<strong>t = {stat["t"]}</strong> '
        f'<span style="color:#64748b;">\u2502</span> '
        f'Edges: {stat["n_edges"]} '
        f'<span style="color:#64748b;">\u2502</span> '
        f'k\u0305 = {stat["mean_degree"]:.1f} '
        f'<span style="color:#64748b;">\u2502</span> '
        f'k_max = {stat["max_degree"]} '
        f'<span style="color:#64748b;">\u2502</span> '
        f'C\u0305 = {stat["mean_constraint"]:.3f} '
        f'{extra}'
        '</div>'
    )


# ---------------------------------------------------------------
# Save (saves panel 0)
# ---------------------------------------------------------------

def save_sim(event):
    if sim_states[0]["history"] is None:
        stats_pane.object = '<div class="stats-bar" style="color:#f43f5e;">No simulation to save.</div>'
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"simulations/sim_{timestamp}"
    save_simulation(sim_states[0]["history"], save_path)
    stats_pane.object = f'<div class="stats-bar"><strong>Saved</strong> \u2192 <code>{save_path}/</code></div>'


# ---------------------------------------------------------------
# Wire up callbacks
# ---------------------------------------------------------------
run_button.on_click(run_sim)
save_button.on_click(save_sim)
player.param.watch(lambda event: render_all_frames(event.new), "value")


def _on_color_change(event):
    render_all_frames(player.value)
    _update_references()


node_color_select.param.watch(_on_color_change, "value")
edge_opacity_select.param.watch(lambda event: render_all_frames(player.value), "value")

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
def _section_label(text):
    """Create a styled section label for the sidebar."""
    return pn.pane.HTML(
        f'<div class="sidebar-section-title">{text}</div>',
    )

seed_row = pn.Row(
    pn.Column(seed_inputs[0], seed_inputs[1], margin=0),
    pn.Column(seed_inputs[2], seed_inputs[3], margin=0),
    margin=0,
)

sidebar = pn.Column(
    _section_label("Geometry"),
    geometry_select, n_agents, dim_slider, alpha_slider,
    n_clusters_slider, sigma_slider, angular_sigma_slider, normalize_select,
    _section_label("Seeds"),
    seed_row,
    _section_label("Visualization"),
    node_color_select, edge_opacity_select,
    _section_label("Mechanisms"),
    toggle_homophily, lam_slider,
    pn.layout.Divider(styles={"border-color": "#e2e8f0", "margin": "2px 0"}),
    toggle_triadic, tau_slider,
    pn.layout.Divider(styles={"border-color": "#e2e8f0", "margin": "2px 0"}),
    toggle_popularity, mu_slider,
    pn.layout.Divider(styles={"border-color": "#e2e8f0", "margin": "2px 0"}),
    attention_mode, beta_slider, budget_slider,
    _section_label("Simulation"),
    n_steps_slider,
    pn.Row(run_button, save_button, margin=(4, 0)),
    _section_label("Playback"),
    player,
    stats_pane,
    width=310,
    styles={"padding": "8px 4px"},
)

def _panel_column(pi):
    """Build a column: network plot + ts chart + histogram for one panel."""
    return pn.Column(
        panels[pi]["plot"],
        ts_charts[pi]["plot"],
        hist_charts[pi]["plot"],
        sizing_mode="fixed",
        margin=(0, 4),
    )


grid = pn.GridSpec(width=880, height=1480)
grid[0, 0] = _panel_column(0)
grid[0, 1] = _panel_column(1)
grid[1, 0] = _panel_column(2)
grid[1, 1] = _panel_column(3)

ref_header = pn.pane.HTML(
    '<div style="font-size:10px; text-transform:uppercase; letter-spacing:1.5px; '
    'color:#475569; border-bottom:1px solid #1e2d45; padding-bottom:4px; '
    'margin-top:8px; margin-bottom:8px; font-weight:500;">'
    'Reference Patterns</div>',
)
ref_row = pn.Row(ref_broker["plot"], ref_clique["plot"], margin=(0, 0))

main = pn.Column(
    ts_legend_pane,
    grid,
    ref_header,
    ref_row,
    styles={"padding": "8px 12px"},
)

template = pn.template.FastListTemplate(
    title="Structural Holes ABM",
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#2563eb",
    header_background="#1e293b",
    background_color="#f1f5f9",
    neutral_color="#e2e8f0",
    theme="default",
    font="IBM Plex Sans",
    font_url="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
)

template.servable()
