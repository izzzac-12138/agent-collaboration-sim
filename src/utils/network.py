"""Network construction utilities for the Boid simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from ..core.agent import Agent

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None


def build_interaction_network(
    agents: list[Agent], radius: float
) -> nx.Graph:
    """Build an undirected graph connecting agents within *radius* of each other.

    Nodes are agent unique_ids. Each edge stores a ``weight`` equal to
    ``1.0 / distance`` (clamped with a small epsilon to avoid division by zero).

    When :mod:`scipy.spatial` is available the function uses a KD-tree for
    O(n log n) neighbor queries; otherwise it falls back to a brute-force
    O(n^2) pairwise check.
    """
    g = nx.Graph()
    if not agents:
        return g

    # Register every agent as a node (even if it has no neighbours).
    for a in agents:
        g.add_node(a.unique_id)

    # Collect positions into a contiguous array for fast vectorised ops.
    positions = np.array([a.pos for a in agents], dtype=float)
    n = len(agents)

    if KDTree is not None:
        _build_kdtree(g, agents, positions, radius)
    else:
        _build_bruteforce(g, agents, positions, radius)

    return g


def _build_kdtree(
    g: nx.Graph,
    agents: list[Agent],
    positions: np.ndarray,
    radius: float,
) -> None:
    """Use scipy.spatial.KDTree for efficient neighbour search."""
    tree = KDTree(positions)
    # query_pairs returns a set of frozensets {(i, j), ...} with i < j.
    pairs: set[tuple[int, int]] = tree.query_pairs(radius)  # type: ignore[assignment]
    eps = 1e-12
    for i, j in pairs:
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        g.add_edge(agents[i].unique_id, agents[j].unique_id, weight=1.0 / max(dist, eps))


def _build_bruteforce(
    g: nx.Graph,
    agents: list[Agent],
    positions: np.ndarray,
    radius: float,
) -> None:
    """O(n^2) fallback when scipy is not installed."""
    eps = 1e-12
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            dist = float(np.sqrt(diff @ diff))
            if dist <= radius:
                g.add_edge(
                    agents[i].unique_id,
                    agents[j].unique_id,
                    weight=1.0 / max(dist, eps),
                )


def network_stats(G: nx.Graph) -> dict[str, Any]:
    """Return basic statistics for the interaction network.

    Returns a dict with keys:
        n_nodes, n_edges, avg_degree, density, avg_clustering, n_components.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "avg_degree": 0.0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "n_components": 0,
        }

    avg_degree = (2 * n_edges) / n_nodes
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    n_components = nx.number_connected_components(G)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "density": density,
        "avg_clustering": avg_clustering,
        "n_components": n_components,
    }
