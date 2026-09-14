"""CollaborationModel -- the core Mesa model for agent collaboration simulation."""

from __future__ import annotations

from typing import Optional

import mesa
import networkx as nx
import numpy as np

from .agents import CollaborativeAgent
from .communication import CommBus
from .logger import EventLogger
from .resources import ResourcePool
from .tasks import TaskManager


class CollaborationModel(mesa.Model):
    """Central model coordinating agents, tasks, communication, and resources.

    Wraps a ``mesa.space.ContinuousSpace`` world and exposes four
    infrastructure subsystems -- ``TaskManager``, ``CommBus``,
    ``ResourcePool``, and ``EventLogger`` -- that the agents interact with
    each step.  An optional *scenario* object can seed the initial state and
    determine when the simulation is finished.

    Parameters
    ----------
    scenario:
        A scenario object implementing ``setup(model)`` and
        ``check_completion(model) -> bool``.
    n_agents:
        Number of ``CollaborativeAgent`` instances to spawn.
    width:
        Width of the continuous toroidal space.
    height:
        Height of the continuous toroidal space.
    """

    def __init__(
        self,
        scenario: Optional[object] = None,
        n_agents: int = 10,
        width: float = 200.0,
        height: float = 200.0,
    ) -> None:
        super().__init__()

        # Spatial environment --------------------------------------------------
        self.space: mesa.space.ContinuousSpace = mesa.space.ContinuousSpace(
            width, height, torus=True
        )

        # Infrastructure subsystems --------------------------------------------
        self.task_manager: TaskManager = TaskManager()
        self.commbus: CommBus = CommBus()
        self.resource_pool: ResourcePool = ResourcePool()
        self.event_logger: EventLogger = EventLogger()

        # Scenario & bookkeeping -----------------------------------------------
        self.scenario: Optional[object] = scenario
        self.step_count: int = 0
        self.network: nx.Graph = nx.Graph()
        self.agents_list: list[CollaborativeAgent] = []
        self.done: bool = False

        # Agent creation -------------------------------------------------------
        for i in range(n_agents):
            x = self.random.uniform(0, width)
            y = self.random.uniform(0, height)
            agent = CollaborativeAgent(self, unique_id=f"agent_{i}")
            self.space.place_agent(agent, (x, y))
            self.agents_list.append(agent)

        # Let the scenario seed tasks / resources if needed --------------------
        if self.scenario is not None:
            self.scenario.setup(self)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the model by one tick.

        Executes every agent's ``step``, rebuilds the proximity network,
        logs a step summary, and checks for scenario completion.
        """
        self.step_count += 1

        for agent in self.agents_list:
            agent.step()

        self._build_network()
        self.event_logger.log_step_summary(self)

        if self.scenario is not None and self.scenario.check_completion(self):
            self.done = True

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def _build_network(self) -> None:
        """Rebuild the agent proximity network from scratch.

        An undirected edge is created between every pair of agents whose
        toroidal distance is within ``perception_radius``.  The previous
        graph contents are discarded each step so the topology always
        reflects the current positions.

        Uses the torus-aware distance metric:

        ``d = sqrt(min(dx, W-dx)^2 + min(dy, H-dy)^2)``
        """
        self.network.clear()
        perception_radius: float = 50.0
        w = self.space.width
        h = self.space.height

        for i, a in enumerate(self.agents_list):
            self.network.add_node(a.unique_id)
            ax, ay = a.pos

            for j in range(i + 1, len(self.agents_list)):
                b = self.agents_list[j]
                bx, by = b.pos

                dx = abs(ax - bx)
                dy = abs(ay - by)
                # Toroidal wrap-around
                if dx > w / 2:
                    dx = w - dx
                if dy > h / 2:
                    dy = h - dy

                if (dx * dx + dy * dy) <= perception_radius * perception_radius:
                    self.network.add_edge(a.unique_id, b.unique_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_agent_positions(self) -> np.ndarray:
        """Return an ``(N, 2)`` array of current agent positions.

        Each row is ``[x, y]`` corresponding to the agent at the same
        index in ``self.agents_list``.
        """
        return np.array([agent.pos for agent in self.agents_list])

    def get_agent_by_id(self, agent_id: int) -> CollaborativeAgent:
        """Look up an agent by its ``unique_id``.

        Returns
        -------
        CollaborativeAgent
            The agent whose ``unique_id`` equals *agent_id*.

        Raises
        ------
        ValueError
            If no agent with that id exists in the model.
        """
        for agent in self.agents_list:
            if agent.unique_id == agent_id:
                return agent
        raise ValueError(
            f"No agent with unique_id={agent_id} in the model "
            f"(known ids: {[a.unique_id for a in self.agents_list]})"
        )
