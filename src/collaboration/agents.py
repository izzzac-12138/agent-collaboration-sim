"""CollaborativeAgent for the collaboration simulation.

Each agent follows a perceive -> decide -> act -> communicate -> log
cycle every Mesa step. Subclasses override ``decide`` to inject
scenario-specific behaviour; the base class provides the plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mesa
import numpy as np

from .communication import CommBus
from .resources import ResourcePool
from .tasks import TaskManager

if TYPE_CHECKING:
    pass


class CollaborativeAgent(mesa.Agent):
    """A single agent in the collaboration simulation.

    Parameters
    ----------
    model:
        The Mesa ``Model`` this agent belongs to.
    unique_id:
        A unique identifier for this agent.
    role:
        One of ``"leader"``, ``"worker"``, ``"scout"``, or ``"bridge"``.
        Roles affect default decision-making (override ``decide``).
    capabilities:
        Skill tags this agent possesses (e.g. ``["coding", "review"]``).
    """

    VALID_ROLES: frozenset[str] = frozenset(
        {"leader", "worker", "scout", "bridge"}
    )

    def __init__(
        self,
        model: mesa.Model,
        unique_id: int | str,
        role: str = "worker",
        capabilities: list[str] | None = None,
    ) -> None:
        super().__init__(model)
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role {role!r}; expected one of {self.VALID_ROLES}"
            )
        self.role: str = role
        self.capabilities: list[str] = capabilities or []

        # Messaging ----------------------------------------------------------
        self.inbox: list[dict[str, Any]] = []

        # Bookkeeping --------------------------------------------------------
        self.resource_cache: dict[str, Any] = {}
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self.messages_sent: int = 0
        self.messages_received: int = 0

        # Spatial ------------------------------------------------------------
        self.pos: np.ndarray = np.array([0.0, 0.0])

    # ------------------------------------------------------------------
    # Mesa step cycle
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute one full decision cycle.

        Order: perceive -> decide -> act -> communicate -> log.
        """
        self.perceive()
        actions = self.decide()
        self.act(actions)
        self.communicate(actions)
        self.log_step()

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def perceive(self) -> None:
        """Pull pending messages from the communication bus."""
        commbus: CommBus = self.model.commbus
        self.inbox = commbus.receive(self.unique_id)
        self.messages_received += len(self.inbox)

    # ------------------------------------------------------------------
    # Decision-making
    # ------------------------------------------------------------------

    def decide(self) -> list[dict[str, Any]]:
        """Return a list of action dicts to execute this step.

        The base implementation returns an empty list.  Override in
        subclasses or scenario mixins to implement agent behaviour.

        Each action dict must include an ``"action"`` key recognised by
        :meth:`act` and :meth:`communicate`:

        * ``{"action": "complete_task", "task_id": ...}``
        * ``{"action": "fail_task", "task_id": ..., "reason": ...}``
        * ``{"action": "transfer_resource", "target": ..., "resource": ...}``
        * ``{"action": "send_message", "to": ..., "content": ...}``
        """
        return []

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def act(self, actions: list[dict[str, Any]]) -> None:
        """Process non-communication actions.

        Handles ``complete_task``, ``fail_task``, and
        ``transfer_resource`` action dicts.

        Parameters
        ----------
        actions:
            Action dicts produced by :meth:`decide`.
        """
        task_mgr: TaskManager = self.model.task_manager
        resource_pool: ResourcePool = self.model.resource_pool

        for action in actions:
            action_type = action.get("action")

            if action_type == "complete_task":
                task_id = action.get("task_id")
                if task_id is not None:
                    task_mgr.complete_task(task_id, agent=self)
                    self.tasks_completed += 1

            elif action_type == "fail_task":
                task_id = action.get("task_id")
                reason = action.get("reason", "unspecified")
                if task_id is not None:
                    task_mgr.fail_task(task_id, agent=self, reason=reason)
                    self.tasks_failed += 1

            elif action_type == "transfer_resource":
                target_id = action.get("target")
                resource = action.get("resource")
                if target_id is not None and resource is not None:
                    resource_pool.transfer(
                        source=self.unique_id,
                        target=target_id,
                        resource=resource,
                    )

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    def communicate(self, actions: list[dict[str, Any]]) -> None:
        """Send messages queued during decision-making.

        Handles ``send_message`` action dicts by pushing them onto the
        communication bus and updating bookkeeping counters.

        Parameters
        ----------
        actions:
            Action dicts produced by :meth:`decide`.
        """
        commbus: CommBus = self.model.commbus

        for action in actions:
            if action.get("action") == "send_message":
                target = action.get("to")
                content = action.get("content", "")
                if target is not None:
                    commbus.send(
                        sender=self.unique_id,
                        receiver=target,
                        content=content,
                    )
                    self.messages_sent += 1

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_step(self) -> None:
        """Record this step's state via the model's event logger."""
        self.model.event_logger.log_agent_step(self)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.unique_id!r}, "
            f"role={self.role!r}, completed={self.tasks_completed}, "
            f"failed={self.tasks_failed})"
        )
