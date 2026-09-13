"""Network construction utilities for the Boid simulation.

Uses toroidal (wrapping) distance to match the simulation's boundary handling,
ensuring agents near opposite edges are correctly identified as neighbors.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def build_interaction_network(
    agents: list, radius: float, width: float = 200.0, height: float = 200.0
) -> nx.Graph:
    """Build an undirected graph connecting agents within *radius* of each other.

    Uses toroidal distance to be consistent with the Boid model's wrapping
    boundary. Nodes are agent unique_ids. Each edge stores a ``weight``
    equal to ``1.0 / distance`` (clamped with a small epsilon to avoid
    division by zero).
    """
    g = nx.Graph()
    if not agents:
        return g

    for a in agents:
        g.add_node(a.unique_id)

    positions = np.array([a.pos for a in agents], dtype=float)
    n = len(agents)
    eps = 1e-12

    for i in range(n):
        for j in range(i + 1, n):
            # Toroidal shortest-distance vector
            diff = positions[j] - positions[i]
            diff[0] -= width * round(diff[0] / width)
            diff[1] -= height * round(diff[1] / height)
            dist = float(np.linalg.norm(diff))
            if dist <= radius:
                g.add_edge(
                    agents[i].unique_id,
                    agents[j].unique_id,
                    weight=1.0 / max(dist, eps),
                )

    return g


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
