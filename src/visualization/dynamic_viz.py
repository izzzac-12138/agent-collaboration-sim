"""
Dynamic visualization for the Boid simulation.

Provides animated real-time rendering of agent movement and interaction
networks using Matplotlib animation and NetworkX.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

from ..models.boid_model import BoidModel


def _extract_state(model):
    """Extract positions, velocities, and interaction graph from the model.

    Returns:
        tuple: (positions, velocities, graph)
            - positions: ndarray of shape (n_agents, 2)
            - velocities: ndarray of shape (n_agents, 2)
            - graph: networkx.Graph
    """
    agents = model.schedule.agents
    positions = np.array([a.pos for a in agents], dtype=float)
    velocities = np.array([a.velocity for a in agents], dtype=float)
    graph = model.interaction_network
    return positions, velocities, graph


def _update_main_plot(ax, model, step_count):
    """Update the main simulation subplot with current agent state.

    Args:
        ax: Matplotlib Axes to draw on.
        model: BoidModel instance with current state.
        step_count: Current simulation step number.
    """
    ax.clear()

    positions, velocities, graph = _extract_state(model)

    # Compute speed for color mapping
    speeds = np.linalg.norm(velocities, axis=1)

    # Draw interaction edges first (behind agents)
    for u, v in graph.edges():
        p1 = positions[u]
        p2 = positions[v]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color='gray', alpha=0.15, linewidth=0.5, zorder=1,
        )

    # Scatter agents colored by speed
    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=speeds, cmap='viridis', s=12, zorder=3, edgecolors='none',
    )

    # Velocity arrows (quiver plot)
    ax.quiver(
        positions[:, 0], positions[:, 1],
        velocities[:, 0], velocities[:, 1],
        color='red', alpha=0.5, scale_units='xy', angles='xy',
        scale=1, width=0.003, zorder=2,
    )

    # Axis setup
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_aspect('equal')
    ax.set_title(f"Boid Simulation  |  step {step_count}", fontsize=12)


def _update_network_plot(ax, model, step_count):
    """Update the interaction network subplot.

    Args:
        ax: Matplotlib Axes to draw on.
        model: BoidModel instance with current state.
        step_count: Current simulation step number.
    """
    ax.clear()

    _, _, graph = _extract_state(model)

    if graph.number_of_nodes() == 0:
        ax.set_title("Interaction Network  |  (no nodes)", fontsize=12)
        return

    pos = nx.spring_layout(graph, seed=42, k=0.8)

    # Node sizes proportional to degree (minimum size to stay visible)
    degrees = dict(graph.degree())
    node_sizes = [max(degrees[n] * 60, 30) for n in graph.nodes()]

    # Edge widths proportional to weight
    if graph.number_of_edges() > 0:
        weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
        max_w = max(weights) if weights else 1.0
        edge_widths = [0.5 + 2.0 * (w / max_w) for w in weights]
    else:
        edge_widths = []

    nx.draw_networkx(
        graph, pos=pos, ax=ax,
        node_size=node_sizes,
        node_color='steelblue',
        edge_color='gray',
        width=edge_widths,
        alpha=0.8,
        with_labels=False,
        font_size=6,
    )

    # Compute stats for title
    n_edges = graph.number_of_edges()
    avg_degree = (
        sum(degrees.values()) / len(degrees) if len(degrees) > 0 else 0
    )
    ax.set_title(
        f"Interaction Network  |  edges={n_edges}  avg_degree={avg_degree:.1f}",
        fontsize=12,
    )


def run_visualization(model, steps=200, interval=50):
    """Run the animated Boid simulation visualization.

    Creates a two-panel display: the left panel shows agent positions and
    velocity vectors, while the right panel shows the evolving interaction
    network.

    Args:
        model: BoidModel instance (already initialized).
        steps: Number of simulation steps to animate.
        interval: Milliseconds between animation frames.
    """
    fig, (ax_main, ax_network) = plt.subplots(1, 2, figsize=(14, 7))

    def animate(frame):
        """Advance model by one step and refresh both panels."""
        model.step()
        _update_main_plot(ax_main, model, frame + 1)
        _update_network_plot(ax_network, model, frame + 1)
        # Returning empty list is fine when using clear/redraw pattern
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=steps, interval=interval, blit=False,
    )

    plt.tight_layout()
    plt.show()

    return anim


def save_snapshot(model, filepath):
    """Save a single-frame snapshot of the current model state.

    Args:
        model: BoidModel instance with current state.
        filepath: Output path for the PNG image.
    """
    fig, (ax_main, ax_network) = plt.subplots(1, 2, figsize=(14, 7))

    step_count = model.schedule.steps if hasattr(model.schedule, 'steps') else 0
    _update_main_plot(ax_main, model, step_count)
    _update_network_plot(ax_network, model, step_count)

    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    model = BoidModel(n_agents=100)
    run_visualization(model, steps=200, interval=50)
