"""Software collaborative development scenario.

Agents work together to complete software modules with dependencies,
code reviews, and integration tasks.
"""

from __future__ import annotations

import random
from typing import Any

from .base import ScenarioBase


class SoftwareDevScenario(ScenarioBase):
    """Simulate a team of developers building software modules."""

    MODULE_TYPES = [
        "api_endpoint",
        "database_schema",
        "ui_component",
        "unit_test",
        "documentation",
        "integration_test",
        "refactoring",
        "bug_fix",
    ]

    def __init__(
        self,
        n_tasks: int = 15,
        n_modules: int = 8,
        seed: int | None = None,
    ) -> None:
        self.n_tasks = n_tasks
        self.n_modules = n_modules
        self._rng = random.Random(seed)

    def get_name(self) -> str:
        return "software_dev"

    def setup(self, model: Any) -> None:
        """Initialize the software development scenario."""
        agents = model.agents_list
        n = len(agents)

        # Assign roles: 15% tech-lead, 25% senior, 60% junior
        for i, agent in enumerate(agents):
            ratio = i / max(n, 1)
            if ratio < 0.15:
                agent.role = "leader"
                agent.capabilities = self.MODULE_TYPES
            elif ratio < 0.40:
                agent.role = "worker"
                agent.capabilities = self._rng.sample(
                    self.MODULE_TYPES, k=self._rng.randint(2, 4)
                )
            else:
                agent.role = "worker"
                agent.capabilities = self._rng.sample(
                    self.MODULE_TYPES, k=self._rng.randint(1, 2)
                )

        # Create tasks (modules to build)
        for _ in range(self.n_tasks):
            module_type = self._rng.choice(self.MODULE_TYPES)
            required_skills = {module_type: 1}
            # Some tasks also need tests or docs
            if self._rng.random() < 0.4:
                required_skills["unit_test"] = 1
            if self._rng.random() < 0.3:
                required_skills["documentation"] = 1

            priority = self._rng.randint(0, 5)
            task = model.task_manager.create_task(
                task_type=module_type,
                required_resources=required_skills,
                priority=priority,
            )
            model.event_logger.log_task_created(task, model.step_count)

    def agent_decide(self, agent: Any, model: Any) -> list[dict]:
        """Decision logic for each agent role."""
        if agent.role == "leader":
            return self._lead_decide(agent, model)
        return self._dev_decide(agent, model)

    def _lead_decide(self, agent: Any, model: Any) -> list[dict]:
        """Tech lead: assigns tasks, coordinates team."""
        actions = []
        pending = model.task_manager.get_pending_tasks()
        workers = [
            a for a in model.agents_list
            if a.role == "worker" and a.unique_id != agent.unique_id
        ]

        for task in pending[:3]:  # assign up to 3 tasks per step
            if not workers:
                break
            # Find a worker with matching capabilities
            capable = [
                w for w in workers
                if task.task_type in w.capabilities
            ]
            if not capable:
                capable = workers  # fallback: assign to anyone

            target = self._rng.choice(capable)
            model.task_manager.assign_task(task.task_id, str(target.unique_id))

            content = (
                f"Assigned: {task.task_id} ({task.task_type}). "
                f"Priority {task.priority}. Please start when ready."
            )
            actions.append({
                "action": "send_message",
                "to": target.unique_id,
                "content": content,
                "msg_type": "request",
            })

        return actions

    def _dev_decide(self, agent: any, model: Any) -> list[dict]:
        """Developer: picks up tasks, completes them, requests help."""
        actions = []
        my_tasks = model.task_manager.get_tasks_by_agent(str(agent.unique_id))

        for task in my_tasks:
            if task.status.value == "assigned":
                model.task_manager.start_task(task.task_id)
                actions.append({
                    "action": "send_message",
                    "to": None,  # broadcast
                    "content": f"Starting work on {task.task_id} ({task.task_type}).",
                    "msg_type": "inform",
                })
            elif task.status.value == "in_progress":
                # 60% chance to complete each step
                if self._rng.random() < 0.6:
                    model.task_manager.complete_task(task.task_id)
                    agent.tasks_completed += 1
                    actions.append({
                        "action": "send_message",
                        "to": None,
                        "content": f"Completed {task.task_id} ({task.task_type}). Ready for review.",
                        "msg_type": "confirm",
                    })
                elif self._rng.random() < 0.1:
                    # Occasionally ask for help
                    lead = next(
                        (a for a in model.agents_list if a.role == "leader"),
                        None,
                    )
                    if lead:
                        actions.append({
                            "action": "send_message",
                            "to": lead.unique_id,
                            "content": f"Need guidance on {task.task_id} ({task.task_type}).",
                            "msg_type": "request",
                        })

        # If no tasks assigned, try to pick up a pending one
        if not my_tasks:
            pending = model.task_manager.get_pending_tasks()
            capable_pending = [
                t for t in pending if t.task_type in agent.capabilities
            ]
            if capable_pending:
                task = self._rng.choice(capable_pending)
                model.task_manager.assign_task(task.task_id, str(agent.unique_id))
                actions.append({
                    "action": "send_message",
                    "to": None,
                    "content": f"Picking up {task.task_id} ({task.task_type}).",
                    "msg_type": "confirm",
                })

        return actions

    def check_completion(self, model: Any) -> bool:
        """Check if all tasks are completed or failed."""
        all_tasks = model.task_manager.all_tasks()
        if not all_tasks:
            return False
        return all(
            t.status.value in ("completed", "failed") for t in all_tasks
        )
