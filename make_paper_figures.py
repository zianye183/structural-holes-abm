"""Generate paper-quality figures from the dose-response and main-factorial data.

Writes to figures/paper/*.pdf and *.png (300 DPI).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

GEOMETRIES = ["torus5d", "torus2d", "poincare"]
GEO_LABELS = {
    "torus5d": "Euclidean 5D torus",
    "torus2d": "Euclidean 2D torus",
    "poincare": "Hyperbolic (Poincaré)",
}
MECHANISMS = ("b_homophily", "b_triadic", "b_popularity")
MECH_LABELS = {
    "b_homophily": "Homophily",
    "b_triadic": "Triadic closure",
    "b_popularity": "Popularity",
}
MECH_COLORS = {
    "b_homophily": "#1f77b4",
    "b_triadic": "#ff7f0e",
    "b_popularity": "#2ca02c",
}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    print(f"  -> {OUT / name}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 1: Dose-response curves
# ---------------------------------------------------------------------------

def fig_dose_response():
    df = pd.read_parquet("simulations/paper_figures/dose_response.parquet")
    end = df[df["t"] == 50]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for gi, geo in enumerate(GEOMETRIES):
        ax = axes[gi]
        sub = end[end["geometry"] == geo]
        for mech in MECHANISMS:
            mech_sub = sub[sub["scan_mechanism"] == mech]
            agg = (mech_sub.groupby("level")
                           .agg(mean=("frac_constraint_lt_0.1", "mean"),
                                std=("frac_constraint_lt_0.1", "std"))
                           .reset_index())
            ax.errorbar(agg["level"], agg["mean"], yerr=agg["std"],
                        marker="o", markersize=5, capsize=3, linewidth=1.8,
                        label=MECH_LABELS[mech], color=MECH_COLORS[mech])
        ax.set_xlabel("Mechanism coefficient")
        if gi == 0:
            ax.set_ylabel(r"Brokerage headcount (fraction with $C_i < 0.1$)")
        ax.set_title(GEO_LABELS[geo])
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks([0, 1, 2, 3, 4])

    axes[-1].legend(loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="0.8", title="Mechanism", title_fontsize=9.5)
    fig.suptitle("Brokerage erosion by mechanism and geometry",
                 y=1.02, fontsize=13, fontweight="semibold")
    save(fig, "fig1_dose_response")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Constraint distribution histograms
# ---------------------------------------------------------------------------

def fig_distributions():
    """Pool per-node constraint across replicates. Columns = baseline + each
    mechanism alone at level 4. Rows = geometries."""
    runs_dir = Path("simulations/paper_figures/runs")

    conditions = [
        ("Baseline", None, 0.0),
        (MECH_LABELS["b_homophily"], "b_homophily", 4.0),
        (MECH_LABELS["b_triadic"], "b_triadic", 4.0),
        (MECH_LABELS["b_popularity"], "b_popularity", 4.0),
    ]

    fig, axes = plt.subplots(len(GEOMETRIES), len(conditions),
                             figsize=(14, 8), sharex=True, sharey=True)

    for gi, geo in enumerate(GEOMETRIES):
        for ci, (label, mech, level) in enumerate(conditions):
            ax = axes[gi, ci]
            pooled = []
            for rep in range(3):
                if mech is None:
                    # Baseline: all mechanisms at 0. Any mech's level-0 run works.
                    path = runs_dir / f"{geo}_b_homophily_0_rep{rep}.npz"
                else:
                    path = runs_dir / f"{geo}_{mech}_{level:g}_rep{rep}.npz"
                arr = np.load(path)
                pooled.append(arr["constraint"][-1])
            pooled = np.concatenate(pooled)
            color = MECH_COLORS.get(mech, "0.4") if mech else "0.4"
            ax.hist(pooled, bins=40, range=(0, 0.30), alpha=0.85,
                    color=color, edgecolor="white", linewidth=0.3)
            ax.axvline(0.10, color="red", linestyle="--", linewidth=1, alpha=0.7)
            frac_broker = (pooled < 0.10).mean()
            ax.text(0.97, 0.93, f"brokers: {frac_broker:.1%}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.9, edgecolor="0.8"))
            if gi == 0:
                ax.set_title(label, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{GEO_LABELS[geo]}\ncount", fontsize=10)
            if gi == len(GEOMETRIES) - 1:
                ax.set_xlabel("Burt constraint $C_i$", fontsize=10.5)
            ax.set_xlim(0, 0.30)

    fig.suptitle("Per-node constraint distributions at mechanism coefficient = 4",
                 y=1.00, fontsize=13, fontweight="semibold")
    save(fig, "fig2_distributions")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Main-factorial summary — brokerage headcount heatmap
# ---------------------------------------------------------------------------

def fig_factorial_heatmap():
    """3 rows (geometries) x 3 cols (mech-pair heatmaps of frac_C<0.1)."""
    df = pd.read_parquet("simulations/main_factorial/summary.parquet")
    end = df[df["t"] == 50].copy()

    mech_pairs = [
        ("b_homophily", "b_triadic"),
        ("b_homophily", "b_popularity"),
        ("b_triadic", "b_popularity"),
    ]

    fig, axes = plt.subplots(len(GEOMETRIES), len(mech_pairs),
                             figsize=(13, 10))

    for gi, geo in enumerate(GEOMETRIES):
        sub = end[end["geometry"] == geo]
        for pi, (mx, my) in enumerate(mech_pairs):
            ax = axes[gi, pi]
            ax.grid(False)
            grid = (sub.groupby([mx, my])["frac_constraint_lt_0.1"]
                       .mean()
                       .unstack(my))
            im = ax.imshow(grid.values, origin="lower", aspect="auto",
                           cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels([f"{v:g}" for v in grid.columns])
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels([f"{v:g}" for v in grid.index])
            ax.set_xlabel(MECH_LABELS[my], fontsize=10)
            ax.set_ylabel(MECH_LABELS[mx], fontsize=10)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid.values[i, j]
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center",
                            color="black" if 0.3 < v < 0.8 else "white",
                            fontsize=9, fontweight="semibold")
            if pi == 0:
                ax.text(-0.35, 0.5, GEO_LABELS[geo],
                        transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=11.5,
                        fontweight="semibold")

    # Single colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r"fraction with $C_i < 0.1$")

    fig.suptitle("Two-way mechanism interactions: brokerage headcount per geometry",
                 y=0.99, fontsize=13, fontweight="semibold")
    fig.subplots_adjust(right=0.91, hspace=0.35, wspace=0.3)
    save(fig, "fig3_interactions")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Summary bar — "each mechanism alone at max level"
# ---------------------------------------------------------------------------

def fig_summary_bar():
    df = pd.read_parquet("simulations/paper_figures/dose_response.parquet")
    end = df[df["t"] == 50]

    # Compute brokerage at each mech alone at level=4 per geometry + baseline
    rows = []
    for geo in GEOMETRIES:
        sub = end[end["geometry"] == geo]
        # Baseline from any mech at level 0
        base = sub[(sub["scan_mechanism"] == "b_homophily") & (sub["level"] == 0)]
        rows.append({"geometry": geo, "mechanism": "Baseline",
                     "frac": base["frac_constraint_lt_0.1"].mean(),
                     "std":  base["frac_constraint_lt_0.1"].std()})
        for mech in MECHANISMS:
            ms = sub[(sub["scan_mechanism"] == mech) & (sub["level"] == 4.0)]
            rows.append({"geometry": geo, "mechanism": MECH_LABELS[mech],
                         "frac": ms["frac_constraint_lt_0.1"].mean(),
                         "std":  ms["frac_constraint_lt_0.1"].std()})
    bar_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(GEOMETRIES))
    width = 0.2
    mech_order = ["Baseline", MECH_LABELS["b_homophily"],
                  MECH_LABELS["b_triadic"], MECH_LABELS["b_popularity"]]
    bar_colors = ["0.55", MECH_COLORS["b_homophily"],
                  MECH_COLORS["b_triadic"], MECH_COLORS["b_popularity"]]

    for mi, mech in enumerate(mech_order):
        vals = [bar_df[(bar_df["geometry"] == geo) & (bar_df["mechanism"] == mech)]["frac"].iloc[0]
                for geo in GEOMETRIES]
        errs = [bar_df[(bar_df["geometry"] == geo) & (bar_df["mechanism"] == mech)]["std"].iloc[0]
                for geo in GEOMETRIES]
        ax.bar(x + (mi - 1.5) * width, vals, width, yerr=errs, capsize=3,
               label=mech, color=bar_colors[mi], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([GEO_LABELS[g] for g in GEOMETRIES])
    ax.set_ylabel(r"Brokerage headcount (fraction with $C_i < 0.1$)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95, edgecolor="0.8")
    ax.set_title("Brokers remaining at each mechanism's maximum coefficient ($b = 4$)",
                 fontsize=12)
    save(fig, "fig4_summary_bar")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures to figures/paper/")
    fig_dose_response()
    fig_distributions()
    fig_factorial_heatmap()
    fig_summary_bar()
    print("done")
