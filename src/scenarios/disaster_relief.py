"""Disaster relief supply distribution scenario.

Simulates a coordinated disaster response where agents with different roles
(leaders, scouts, workers) collaborate to distribute relief supplies across
resource points.  Leaders prioritise and assign tasks, scouts discover and
report resource locations, and workers carry out the actual distribution.

Completion is reached when every task has either been COMPLETED or FAILED.
"""

from __future__ import annotations

import random
from typing import Any

from .base import ScenarioBase
from ..collaboration.tasks import Task, TaskStatus


# Resource point label template -----------------------------------------------

_RESOURCE_LABELS: list[str] = [
    "Shelter Alpha",
    "Shelter Beta",
    "Medical Center",
    "Distribution Hub",
    "Field Hospital",
]


class DisasterReliefScenario(ScenarioBase):
    """Disaster relief supply distribution scenario.

    The simulation area represents a disaster-struck region.  Multiple resource
    points hold supplies, and a set of distribution tasks must be completed by
    a team of agents organised into leaders, scouts, and workers.

    Parameters
    ----------
    n_tasks:
        Number of distribution tasks to create during setup.
    n_resource_points:
        Number of independent supply depots to place on the map.
    resource_types:
        Kinds of relief物资 (food, water, medicine, blankets, ...).  Each
        task requires a random subset of these; each resource point stocks
        a random amount of each type.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_tasks: int = 20,
        n_resource_points: int = 5,
        resource_types: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_tasks: int = n_tasks
        self.n_resource_points: int = n_resource_points
        self.resource_types: list[str] = resource_types or [
            "food",
            "water",
            "medicine",
            "blankets",
        ]
        self._rng: random.Random = random.Random(seed)

    # ------------------------------------------------------------------
    # ScenarioBase interface
    # ------------------------------------------------------------------

    def setup(self, model: Any) -> None:
        """Seed roles, create distribution tasks, and stock resource points.

        Role assignment (deterministic split based on agent order):

        * First 20 % --> ``"leader"``
        * Next  20 % --> ``"scout"``
        * Rest       --> ``"worker"``

        Each task demands a random non-empty subset of ``resource_types``
        with small random quantities (1-5 units per type).  Resource points
        are created as unowned pool entries so any agent can claim supplies.
        """
        agents = model.agents_list
        n = len(agents)
        n_leaders = max(1, int(n * 0.2))
        n_scouts = max(1, int(n * 0.2))

        for idx, agent in enumerate(agents):
            if idx < n_leaders:
                agent.role = "leader"
            elif idx < n_leaders + n_scouts:
                agent.role = "scout"
            else:
                agent.role = "worker"

        # --- create distribution tasks -----------------------------------
        for _ in range(self.n_tasks):
            # Pick 1-3 resource types this task requires.
            k = self._rng.randint(1, min(3, len(self.resource_types)))
            chosen_types = self._rng.sample(self.resource_types, k)
            required = {
                rtype: self._rng.randint(1, 5) for rtype in chosen_types
            }
            priority = self._rng.randint(0, 5)

            task_type = self._rng.choice(
                [
                    "medical_evacuation",
                    "food_delivery",
                    "water_distribution",
                    "shelter_setup",
                    "blanket_delivery",
                    "medicine_delivery",
                ]
            )
            model.task_manager.create_task(
                task_type=task_type,
                required_resources=required,
                priority=priority,
            )

        # --- create resource points (unowned pool resources) ------------
        for idx in range(self.n_resource_points):
            label = (
                _RESOURCE_LABELS[idx]
                if idx < len(_RESOURCE_LABELS)
                else f"Depot {idx + 1}"
            )
            for rtype in self.resource_types:
                amount = float(self._rng.randint(5, 20))
                model.resource_pool.add_resource(
                    resource_type=rtype,
                    amount=amount,
                    owner=None,  # unowned pool resource
                    location=(
                        self._rng.uniform(10, model.space.width - 10),
                        self._rng.uniform(10, model.space.height - 10),
                    ),
                )

        model.event_logger._emit(
            "scenario_setup", "", {
                "scenario": self.get_name(),
                "n_tasks": self.n_tasks,
                "n_resource_points": self.n_resource_points,
                "resource_types": list(self.resource_types),
            }, model.step_count
        )

    # ------------------------------------------------------------------

    def agent_decide(self, agent: Any, model: Any) -> list[dict[str, Any]]:
        """Produce actions for a single agent this tick.

        Behaviour varies by role:

        **Leaders**
            Scan all PENDING tasks, prioritise by ``task.priority``, and
            assign the highest-priority pending task to a worker that is
            not currently busy (has no ASSIGNED/IN_PROGRESS task).  If
            assignment succeeds the leader broadcasts an informative
            message so scouts can relay resource information to the worker.

        **Scouts**
            Examine unowned resources in the pool and broadcast
            ``resource_share`` messages listing available supplies so
            workers know what they can collect.

        **Workers**
            Check assigned tasks first: if an ASSIGNED task exists,
            start it.  If an IN_PROGRESS task exists and the worker has
            accumulated enough resources in its cache, complete the task;
            otherwise, claim unowned resources from the pool that match
            the task requirements and cache them for the next tick.

        Returns
        -------
        list[dict[str, Any]]
            Action dicts understood by the base ``CollaborativeAgent``.
        """
        actions: list[dict[str, Any]] = []

        if agent.role == "leader":
            actions.extend(self._leader_decide(agent, model))
        elif agent.role == "scout":
            actions.extend(self._scout_decide(agent, model))
        elif agent.role == "worker":
            actions.extend(self._worker_decide(agent, model))

        return actions

    # ------------------------------------------------------------------

    def check_completion(self, model: Any) -> bool:
        """Return ``True`` when every task is COMPLETED or FAILED."""
        for task in model.task_manager.all_tasks():
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return False
        return True

    # ------------------------------------------------------------------
    # Role-specific decision helpers
    # ------------------------------------------------------------------

    def _leader_decide(
        self, agent: Any, model: Any
    ) -> list[dict[str, Any]]:
        """Leader logic: assign pending tasks to idle workers.

        The leader iterates pending tasks in priority order (descending)
        and assigns each one to the first worker that currently has no
        active task assignment.  After assigning, the leader broadcasts
        an ``inform`` message so other agents are aware.
        """
        actions: list[dict[str, Any]] = []
        tm = model.task_manager
        pool = model.resource_pool

        pending = tm.get_pending_tasks()
        # Sort descending by priority so high-priority tasks go first.
        pending.sort(key=lambda t: t.priority, reverse=True)

        # Build a set of worker IDs that already have an active task.
        busy_workers: set[str] = set()
        for task in tm.all_tasks():
            if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                if task.assigned_to is not None:
                    busy_workers.add(task.assigned_to)

        for task in pending:
            target_worker = self._find_idle_worker(
                model.agents_list, busy_workers
            )
            if target_worker is None:
                break  # no idle workers left

            worker_id = str(target_worker.unique_id)
            tm.assign_task(task.task_id, worker_id)
            busy_workers.add(worker_id)

            # Build a human-readable requirement summary for the message.
            req_summary = ", ".join(
                f"{amt} {rtype}"
                for rtype, amt in task.required_resources.items()
            )
            content = (
                f"Task {task.task_id} ({task.task_type}) assigned to you. "
                f"Requires: {req_summary}. Priority: {task.priority}."
            )
            actions.append(
                {
                    "action": "send_message",
                    "to": target_worker.unique_id,
                    "content": content,
                }
            )

        return actions

    def _scout_decide(
        self, agent: Any, model: Any
    ) -> list[dict[str, Any]]:
        """Scout logic: broadcast information about available resources.

        Scouts periodically scan the resource pool for unowned supplies
        and broadcast a summary to all agents.  The broadcast includes
        the resource types and total amounts available so workers can
        plan their resource collection.
        """
        actions: list[dict[str, Any]] = []
        pool = model.resource_pool
        totals = pool.total_by_type()  # all resources (pool-level query)

        if not totals:
            return actions

        lines = [
            f"{rtype}: {amt:.0f} units available"
            for rtype, amt in sorted(totals.items())
        ]
        content = "Resource status — " + "; ".join(lines)
        actions.append(
            {
                "action": "send_message",
                "to": None,  # broadcast to all
                "content": content,
            }
        )

        return actions

    def _worker_decide(
        self, agent: Any, model: Any
    ) -> list[dict[str, Any]]:
        """Worker logic: progress through the task lifecycle.

        1. If the worker has an ASSIGNED task, start it.
        2. If the worker has an IN_PROGRESS task, check the resource
           cache.  If all required resources are cached, deduct them
           from the cache and complete the task.  Otherwise, claim
           matching resources from the pool and add them to the cache.
        3. If the worker has no active task, do nothing.
        """
        actions: list[dict[str, Any]] = []
        tm = model.task_manager
        pool = model.resource_pool
        worker_id = str(agent.unique_id)

        my_tasks = tm.get_tasks_by_agent(worker_id)

        for task in my_tasks:
            # --- ASSIGNED -> IN_PROGRESS --------------------------------
            if task.status == TaskStatus.ASSIGNED:
                tm.start_task(task.task_id)
                # Fall through to check resources this same tick.
                # Re-read after status change.

            # --- IN_PROGRESS: try to complete or gather resources --------
            if task.status == TaskStatus.IN_PROGRESS:
                if self._has_all_resources(agent, task):
                    self._deduct_cached_resources(agent, task)
                    task_id = task.task_id
                    actions.append(
                        {
                            "action": "complete_task",
                            "task_id": task_id,
                        }
                    )
                else:
                    self._claim_resources(agent, task, pool)

        return actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_idle_worker(
        agents: list[Any], busy_workers: set[str]
    ) -> Any | None:
        """Return the first worker-role agent that has no active task.

        Parameters
        ----------
        agents:
            Full agent list from the model.
        busy_workers:
            Set of agent ID strings that currently hold an ASSIGNED or
            IN_PROGRESS task.

        Returns
        -------
        The idle worker agent, or ``None`` if all workers are busy.
        """
        for candidate in agents:
            if candidate.role != "worker":
                continue
            if str(candidate.unique_id) in busy_workers:
                continue
            return candidate
        return None

    @staticmethod
    def _has_all_resources(agent: Any, task: Task) -> bool:
        """Check whether *agent*'s resource cache satisfies *task* requirements.

        Parameters
        ----------
        agent:
            Agent whose ``resource_cache`` dict is inspected.
        task:
            Task whose ``required_resources`` mapping is checked.

        Returns
        -------
        ``True`` if every required resource type is present in sufficient
        quantity in the agent's cache.
        """
        cache: dict[str, float] = agent.resource_cache
        for rtype, needed in task.required_resources.items():
            if cache.get(rtype, 0.0) < needed:
                return False
        return True

    @staticmethod
    def _deduct_cached_resources(agent: Any, task: Task) -> None:
        """Deduct the resources required by *task* from the agent's cache.

        Parameters
        ----------
        agent:
            Agent whose ``resource_cache`` is modified in-place.
        task:
            Task whose ``required_resources`` are subtracted.
        """
        cache: dict[str, float] = agent.resource_cache
        for rtype, needed in task.required_resources.items():
            cache[rtype] = cache.get(rtype, 0.0) - needed

    @staticmethod
    def _claim_resources(
        agent: Any, task: Task, pool: Any
    ) -> None:
        """Claim unowned resources from *pool* that match *task* requirements.

        For each required resource type, the worker searches the pool for
        unowned resources of that type and transfers up to the needed
        amount into the agent's ``resource_cache``.  The pool's
        ``transfer`` method is bypassed in favour of direct quantity
        manipulation because unowned resources do not have an owner to
        transfer *from*.

        Parameters
        ----------
        agent:
            Agent whose ``resource_cache`` is updated.
        task:
            Task whose ``required_resources`` drive the claim.
        pool:
            The ``ResourcePool`` instance to search.
        """
        cache: dict[str, float] = agent.resource_cache

        for rtype, needed in task.required_resources.items():
            still_needed = needed - cache.get(rtype, 0.0)
            if still_needed <= 0:
                continue

            for resource in pool.get_all_resources():
                if resource.owner is not None:
                    continue
                if resource.resource_type != rtype:
                    continue
                if resource.amount <= 0:
                    continue

                take = min(still_needed, resource.amount)
                resource.amount -= take
                cache[rtype] = cache.get(rtype, 0.0) + take
                still_needed -= take

                if still_needed <= 0:
                    break

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the human-readable scenario name."""
        return "DisasterRelief"

    def get_description(self) -> str:
        """Return a brief description of the scenario narrative."""
        return (
            "Disaster relief supply distribution.  Agents coordinate as "
            "leaders, scouts, and workers to prioritise, discover, and "
            "deliver relief supplies (food, water, medicine, blankets) "
            "across multiple resource points in a disaster-struck region."
        )
