"""
Core ABM module: initialization and analysis.

All initialization functions return an InitResult with a distance matrix.
The geometry is absorbed into the initialization step.
"""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Common output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InitResult:
    """Output of any initialization function.

    distance_matrix: (N, N) pairwise distances — the common currency.
    viz_coords:      (N, 2) coordinates for plotting (geometry-specific).
    metadata:        anything else (labels, raw positions, params, etc.).
    """
    distance_matrix: np.ndarray
    viz_coords: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.distance_matrix.shape[0]

    def normalized(self, method: str = "mean") -> "InitResult":
        """Return a new InitResult with normalized distance matrix.

        Methods:
            "mean": D / mean(D). Mean distance becomes 1.
                    Preserves distribution shape. Good default.
            "max":  D / max(D). Farthest pair becomes 1.
                    All values in [0, 1]. Sensitive to outliers.

        The raw distance matrix is preserved in metadata["D_raw"].
        """
        D = self.distance_matrix
        upper = D[np.triu_indices(D.shape[0], k=1)]

        if method == "mean":
            scale = upper.mean()
        elif method == "max":
            scale = upper.max()
        else:
            raise ValueError(f"Unknown normalization method: {method!r}. Use 'mean' or 'max'.")

        D_norm = D / scale
        new_metadata = {**self.metadata, "D_raw": D, "normalization": method, "norm_scale": scale}
        return InitResult(
            distance_matrix=D_norm,
            viz_coords=self.viz_coords,
            metadata=new_metadata,
        )


# ---------------------------------------------------------------------------
# Torus initialization
# ---------------------------------------------------------------------------

def _torus_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Pairwise toroidal distances for agents in [0, 1)^d."""
    diff = np.abs(positions[:, np.newaxis, :] - positions[np.newaxis, :, :])
    diff = np.minimum(diff, 1.0 - diff)
    return np.sqrt(np.sum(diff ** 2, axis=2))


def init_torus_uniform(
    n: int,
    d: int,
    rng: np.random.Generator,
) -> InitResult:
    """Uniform agents on [0, 1)^d torus."""
    positions = rng.uniform(0, 1, size=(n, d))
    return InitResult(
        distance_matrix=_torus_distance_matrix(positions),
        viz_coords=positions[:, :2],  # project to first 2 dims
        metadata={"geometry": "torus", "init": "uniform", "d": d,
                  "positions": positions},
    )


def init_torus_gmm(
    n: int,
    d: int,
    n_clusters: int,
    sigma: float,
    rng: np.random.Generator,
) -> InitResult:
    """Gaussian mixture on [0, 1)^d torus."""
    centers = rng.uniform(0, 1, size=(n_clusters, d))
    labels = rng.integers(0, n_clusters, size=n)
    positions = (centers[labels] + rng.normal(0, sigma, size=(n, d))) % 1.0
    return InitResult(
        distance_matrix=_torus_distance_matrix(positions),
        viz_coords=positions[:, :2],
        metadata={"geometry": "torus", "init": "gmm", "d": d,
                  "n_clusters": n_clusters, "sigma": sigma,
                  "labels": labels, "positions": positions},
    )


# ---------------------------------------------------------------------------
# Hyperbolic initialization
# ---------------------------------------------------------------------------

def _hyperbolic_distance_matrix(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Pairwise hyperbolic distances (Krioukov Eq. 5, zeta=1)."""
    r1, r2 = r[:, np.newaxis], r[np.newaxis, :]
    t1, t2 = theta[:, np.newaxis], theta[np.newaxis, :]
    delta_theta = np.pi - np.abs(np.pi - np.abs(t1 - t2))
    cosh_x = (
        np.cosh(r1) * np.cosh(r2)
        - np.sinh(r1) * np.sinh(r2) * np.cos(delta_theta)
    )
    return np.arccosh(np.maximum(cosh_x, 1.0))


def _hyperbolic_viz_coords(
    r: np.ndarray,
    theta: np.ndarray,
    R: float,
    stretch: float = 0.6,
) -> np.ndarray:
    """Map (r, theta) to 2D Poincare disk with visual scaling."""
    r_visual = (r / R) ** stretch
    return np.column_stack([r_visual * np.cos(theta),
                            r_visual * np.sin(theta)])


def _hyperbolic_disk_radius(n: int, spread: float, alpha: float) -> float:
    """Disk radius R controlling point spread.

    Based on Krioukov Eq. 13 generalized for arbitrary alpha.
    For alpha=0.5 this reduces to 2*ln(8n/(pi*spread)).

    Args:
        n: number of agents.
        spread: controls how spread-out points are (higher = tighter).
        alpha: radial density exponent (gamma = 2*alpha + 1).
    """
    return (1.0 / alpha) * np.log(8.0 * alpha * n / (np.pi * spread))


def _sample_hyperbolic_r(
    n: int,
    R: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inverse CDF sampling for generalized radial density (Eq. 17)."""
    u = rng.uniform(0, 1, size=n)
    return np.arccosh(1.0 + u * (np.cosh(alpha * R) - 1.0)) / alpha


def init_hyperbolic_uniform(
    n: int,
    spread: float,
    rng: np.random.Generator,
    alpha: float = 0.5,
    viz_stretch: float = 0.6,
) -> InitResult:
    """Krioukov prescription: uniform theta, sinh(alpha*r) radial density.

    alpha controls power-law exponent: gamma = 2*alpha + 1 (zeta=1).
    spread controls how concentrated points are (higher = tighter clustering).
    """
    R = _hyperbolic_disk_radius(n, spread, alpha)
    r = _sample_hyperbolic_r(n, R, alpha, rng)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return InitResult(
        distance_matrix=_hyperbolic_distance_matrix(r, theta),
        viz_coords=_hyperbolic_viz_coords(r, theta, R, viz_stretch),
        metadata={"geometry": "hyperbolic", "init": "uniform",
                  "alpha": alpha, "R": R, "gamma": 2 * alpha + 1,
                  "r": r, "theta": theta},
    )


def init_hyperbolic_gmm(
    n: int,
    spread: float,
    n_clusters: int,
    angular_sigma: float,
    rng: np.random.Generator,
    alpha: float = 0.5,
    viz_stretch: float = 0.6,
) -> InitResult:
    """Krioukov radial density + Gaussian mixture on theta."""
    R = _hyperbolic_disk_radius(n, spread, alpha)
    r = _sample_hyperbolic_r(n, R, alpha, rng)
    centers = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    labels = rng.integers(0, n_clusters, size=n)
    theta = (centers[labels] + rng.normal(0, angular_sigma, size=n)) % (2 * np.pi)
    return InitResult(
        distance_matrix=_hyperbolic_distance_matrix(r, theta),
        viz_coords=_hyperbolic_viz_coords(r, theta, R, viz_stretch),
        metadata={"geometry": "hyperbolic", "init": "gmm",
                  "alpha": alpha, "R": R, "gamma": 2 * alpha + 1,
                  "n_clusters": n_clusters, "angular_sigma": angular_sigma,
                  "labels": labels, "r": r, "theta": theta},
    )


# ---------------------------------------------------------------------------
# Analysis (geometry-agnostic)
# ---------------------------------------------------------------------------

def network_summary(G: nx.Graph) -> dict[str, Any]:
    """Key network statistics."""
    degrees = np.array([d for _, d in G.degree()])
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "mean_degree": float(degrees.mean()),
        "std_degree": float(degrees.std()),
        "clustering": nx.average_clustering(G),
        "n_components": nx.number_connected_components(G),
        "degrees": degrees,
    }


def burt_constraint(G: nx.Graph) -> np.ndarray:
    """Burt's constraint per node (vectorized). Lower = more structural holes."""
    A = nx.to_numpy_array(G)
    deg = A.sum(axis=1, keepdims=True)
    isolated = (deg.flatten() == 0)

    # Avoid division by zero for isolated nodes
    safe_deg = np.where(deg == 0, 1.0, deg)
    P = A / safe_deg  # P[i,j] = proportion of i's relations invested in j

    # c_ij = (p_ij + Σ_{q≠i,j} p_iq·p_qj)², summed over j ∈ N(i) only
    M = P + P @ P
    np.fill_diagonal(M, 0.0)
    constraint = ((M ** 2) * A).sum(axis=1)
    constraint[isolated] = np.nan
    return constraint
