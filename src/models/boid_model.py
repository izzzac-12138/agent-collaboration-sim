"""
BoidModel - Mesa 3.x Model for Boid flocking simulation.

Implements a multi-agent flocking model based on Reynolds' Boids rules
(separation, alignment, cohesion) with dynamic interaction networks.
"""

import mesa
import numpy as np
import networkx as nx

from ..agents.boid_agent import BoidAgent
from ..utils.network import build_interaction_network


class BoidModel(mesa.Model):
    """Mesa model for simulating boid flocking behavior.

    Agents move in a continuous toroidal space following three steering rules:
    separation, alignment, and cohesion. An interaction network is built each
    step based on perception radius proximity, and network metrics are collected.

    Attributes:
        n_agents: Number of boid agents in the simulation.
        space: Continuous 2D toroidal space.
        perception_radius: Radius within which agents perceive neighbors.
        separation_radius: Radius within which agents actively avoid each other.
        max_speed: Maximum agent speed.
        max_force: Maximum steering force magnitude.
        weights: Tuple of (separation, alignment, cohesion) steering weights.
        network: NetworkX graph of current agent interactions.
        step_count: Current simulation step number.
        agents_list: Ordered list of all BoidAgent instances.
        datacollector: Mesa DataCollector for model-level metrics.
    """

    def __init__(
        self,
        n_agents: int = 100,
        width: float = 200.0,
        height: float = 200.0,
        perception_radius: float = 50.0,
        separation_radius: float = 25.0,
        max_speed: float = 2.0,
        max_force: float = 0.3,
        weights: tuple = (1.5, 1.0, 1.0),
    ):
        super().__init__()
        self.n_agents = n_agents
        self.space = mesa.space.ContinuousSpace(width, height, torus=True)
        self.perception_radius = perception_radius
        self.separation_radius = separation_radius
        self.max_speed = max_speed
        self.max_force = max_force
        self.weights = weights
        self.network = nx.Graph()
        self.step_count = 0

        # Create agents with random positions and velocities
        self.agents_list = []
        for i in range(n_agents):
            pos = (
                self.random.uniform(0, width),
                self.random.uniform(0, height),
            )
            angle = self.random.uniform(0, 2 * np.pi)
            speed = self.random.uniform(0.5, max_speed)
            velocity = (speed * np.cos(angle), speed * np.sin(angle))
            agent = BoidAgent(
                model=self,
                unique_id=i,
                pos=pos,
                velocity=velocity,
                perception_radius=perception_radius,
                separation_radius=separation_radius,
                max_speed=max_speed,
                max_force=max_force,
            )
            self.space.place_agent(agent, pos)
            agent.pos = np.array(agent.pos, dtype=float)
            self.agents_list.append(agent)

        # Configure data collection for network metrics
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "n_edges": lambda m: m.network.number_of_edges(),
                "avg_degree": lambda m: (
                    2 * m.network.number_of_edges() / m.network.number_of_nodes()
                    if m.network.number_of_nodes() > 0
                    else 0
                ),
                "n_components": lambda m: (
                    nx.number_connected_components(m.network)
                    if m.network.number_of_nodes() > 0
                    else 0
                ),
                "avg_clustering": lambda m: (
                    nx.average_clustering(m.network)
                    if m.network.number_of_nodes() > 0
                    else 0
                ),
            }
        )

    def step(self) -> None:
        """Execute one simulation step.

        Each agent computes and applies steering forces (separation, alignment,
        cohesion) weighted by model parameters. After all agents move, the
        interaction network is rebuilt and model metrics are recorded.
        """
        self.step_count += 1

        # Each agent applies flocking behavior and moves
        for agent in self.agents_list:
            agent.step(self.weights)

        # Rebuild the interaction network from current positions
        self.network = build_interaction_network(
            self.agents_list, self.perception_radius
        )

        # Collect model-level metrics
        self.datacollector.collect(self)

    def get_agent_positions(self) -> np.ndarray:
        """Return an (N, 2) array of all agent positions."""
        return np.array([a.pos for a in self.agents_list])

    def get_agent_velocities(self) -> np.ndarray:
        """Return an (N, 2) array of all agent velocities."""
        return np.array([a.velocity for a in self.agents_list])

    def get_interaction_network(self) -> nx.Graph:
        """Return a copy of the current interaction network."""
        return self.network.copy()
