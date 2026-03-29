"""
Core ABM module: initialization, tie formation, and analysis.

All initialization functions return an InitResult with a distance matrix.
All tie formation functions take a distance matrix and return a nx.Graph.
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


def _hyperbolic_disk_radius(n: int, target_mean_degree: float) -> float:
    """Disk radius R from Krioukov Eq. 13."""
    return 2.0 * np.log(8.0 * n / (np.pi * target_mean_degree))


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
    target_mean_degree: float,
    rng: np.random.Generator,
    alpha: float = 0.5,
    viz_stretch: float = 0.6,
) -> InitResult:
    """Krioukov prescription: uniform theta, sinh(alpha*r) radial density.

    alpha controls power-law exponent: gamma = 2*alpha + 1 (zeta=1).
    NOTE: alpha is a key tunable parameter.
    """
    R = _hyperbolic_disk_radius(n, target_mean_degree)
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
    target_mean_degree: float,
    n_clusters: int,
    angular_sigma: float,
    rng: np.random.Generator,
    alpha: float = 0.5,
    viz_stretch: float = 0.6,
) -> InitResult:
    """Krioukov radial density + Gaussian mixture on theta."""
    R = _hyperbolic_disk_radius(n, target_mean_degree)
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
# Tie formation (geometry-agnostic — operates on distance matrix only)
# ---------------------------------------------------------------------------

def form_network_threshold(dist_matrix: np.ndarray, radius: float) -> nx.Graph:
    """Hard threshold: connect if distance < radius."""
    n = dist_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    mask = np.triu((dist_matrix < radius) & (dist_matrix > 0), k=1)
    G.add_edges_from(zip(*np.where(mask)))
    return G


def form_network_fermi_dirac(
    dist_matrix: np.ndarray,
    radius: float,
    temperature: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Soft threshold: p(connect) = 1 / (1 + exp((d - R) / 2T))."""
    n = dist_matrix.shape[0]
    if temperature <= 0:
        return form_network_threshold(dist_matrix, radius)

    prob = 1.0 / (1.0 + np.exp((dist_matrix - radius) / (2.0 * temperature)))
    np.fill_diagonal(prob, 0)
    draws = rng.uniform(0, 1, size=(n, n))
    mask = np.triu(draws < prob, k=1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(*np.where(mask)))
    return G


def form_network_knn(dist_matrix: np.ndarray, k: int) -> nx.Graph:
    """k-nearest neighbors: connect to k closest agents (symmetrized)."""
    n = dist_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        dists_i = dist_matrix[i].copy()
        dists_i[i] = np.inf
        neighbors = np.argpartition(dists_i, k)[:k]
        for j in neighbors:
            G.add_edge(i, j)
    return G


def calibrate_radius(
    dist_matrix: np.ndarray,
    target_mean_degree: float,
) -> float:
    """Find distance threshold that yields target mean degree."""
    n = dist_matrix.shape[0]
    dists = dist_matrix[np.triu_indices(n, k=1)]
    target_quantile = target_mean_degree / (n - 1)
    return float(np.quantile(dists, target_quantile))


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
    """Burt's constraint per node. Lower = more structural holes."""
    n = G.number_of_nodes()
    constraint = np.full(n, np.nan)
    for i in G.nodes():
        neighbors_i = set(G.neighbors(i))
        if len(neighbors_i) == 0:
            continue
        deg_i = len(neighbors_i)
        c_i = 0.0
        for j in neighbors_i:
            p_ij = 1.0 / deg_i
            indirect = 0.0
            for q in neighbors_i:
                if q != j and G.has_edge(q, j):
                    p_iq = 1.0 / deg_i
                    deg_q = G.degree(q)
                    p_qj = 1.0 / deg_q if deg_q > 0 else 0.0
                    indirect += p_iq * p_qj
            c_i += (p_ij + indirect) ** 2
        constraint[i] = c_i
    return constraint
