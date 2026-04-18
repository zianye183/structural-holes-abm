"""Generate Figure 5: saturation curves for the three mechanisms.

Shows Jaccard similarity of final network to max-b reference run,
overlaid with analytical saturation thresholds b_sat.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

GEOMETRIES = ["torus5d", "torus2d", "poincare"]
GEO_LABELS = {"torus5d": "Euclidean 5D torus",
              "torus2d": "Euclidean 2D torus",
              "poincare": "Hyperbolic (Poincaré)"}
MECHANISMS = ("b_homophily", "b_triadic", "b_popularity")
MECH_LABELS = {"b_homophily": "Homophily",
               "b_triadic": "Triadic closure",
               "b_popularity": "Popularity"}
MECH_COLORS = {"b_homophily": "#1f77b4",
               "b_triadic": "#ff7f0e",
               "b_popularity": "#2ca02c"}

# Analytical saturation predictions: b_sat at which the mechanism contribution
# to the critical (budget-boundary) pair reaches logit = +7, i.e. P >= 0.999
D20 = {"torus5d": 0.659, "torus2d": 0.382, "poincare": 0.753}
BUDGET = 20
TYPICAL_SHARED = 5   # see text: estimate of shared neighbors at equilibrium

def predicted_bsat(mech: str, geo: str) -> float:
    if mech == "b_homophily":
        return 12 * (D20[geo] + 0.1)
    if mech == "b_triadic":
        return 12 / TYPICAL_SHARED
    if mech == "b_popularity":
        return 12 / BUDGET
    raise ValueError(mech)


df = pd.read_parquet("simulations/saturation/jaccard.parquet")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)

for gi, geo in enumerate(GEOMETRIES):
    ax = axes[gi]
    for mech in MECHANISMS:
        sub = df[(df["geometry"] == geo) & (df["mechanism"] == mech)].sort_values("level")
        ax.plot(sub["level"], sub["jaccard_to_max"],
                marker="o", markersize=5, linewidth=1.8,
                label=MECH_LABELS[mech], color=MECH_COLORS[mech])
        bsat = predicted_bsat(mech, geo)
        ax.axvline(bsat, color=MECH_COLORS[mech], linestyle=":", linewidth=1.5, alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Mechanism coefficient $b$ (log scale)")
    if gi == 0:
        ax.set_ylabel(r"Jaccard similarity to max-$b$ network")
    ax.set_title(GEO_LABELS[geo])
    ax.set_ylim(-0.05, 1.08)
    ax.axhline(1.0, color="0.7", linestyle="-", linewidth=0.6)

axes[-1].legend(loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="0.8", title="Mechanism", title_fontsize=9.5)

# Annotate predicted thresholds textually in a single panel (the middle one)
mid = axes[1]
tx = 0.02
ty = 0.03
handles = []
for mech in MECHANISMS:
    bsat_text = f"{MECH_LABELS[mech]}: $b_{{\\rm sat}}$ ≈ {predicted_bsat(mech, 'torus2d'):.1f}"
    mid.text(tx, ty, bsat_text, transform=mid.transAxes, fontsize=8,
             color=MECH_COLORS[mech], fontweight="semibold")
    ty += 0.055

fig.suptitle("Saturation curves: empirical Jaccard vs analytical $b_{\\rm sat}$ (dotted lines)",
             y=1.01, fontsize=13, fontweight="semibold")
fig.tight_layout()
fig.savefig(OUT / "fig5_saturation.pdf")
fig.savefig(OUT / "fig5_saturation.png")
print(f"Wrote fig5_saturation.{{pdf,png}}")
