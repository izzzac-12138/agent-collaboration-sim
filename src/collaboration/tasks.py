"""Task and TaskManager for the collaboration simulation."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    """Status of a task in its lifecycle."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a unit of work that can be assigned to an agent.

    Attributes:
        task_id: Unique identifier for the task.
        task_type: Category or kind of task.
        required_resources: Resource name to amount mapping needed by this task.
        assigned_to: ID of the agent this task is assigned to, or None.
        status: Current lifecycle status of the task.
        created_at: Timestamp when the task was created.
        completed_at: Timestamp when the task was completed, or None.
        priority: Higher value means higher priority. Default is 0.
    """

    task_id: str
    task_type: str
    required_resources: dict[str, int]
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: int = 0
    completed_at: Optional[int] = None
    priority: int = 0


class TaskManager:
    """Manages the creation, assignment, and lifecycle of tasks.

    Provides methods to create tasks, transition them through statuses,
    and query tasks by various criteria.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._next_id: int = 0

    def _generate_id(self) -> str:
        """Generate the next task ID in T001 format."""
        self._next_id += 1
        return f"T{self._next_id:03d}"

    def create_task(
        self,
        task_type: str,
        required_resources: dict[str, int],
        priority: int = 0,
    ) -> Task:
        """Create a new task with PENDING status.

        Args:
            task_type: Category or kind of task.
            required_resources: Resource name to amount mapping needed.
            priority: Higher value means higher priority.

        Returns:
            The newly created Task.
        """
        task_id = self._generate_id()
        task = Task(
            task_id=task_id,
            task_type=task_type,
            required_resources=required_resources,
            status=TaskStatus.PENDING,
            created_at=int(time.time()),
            priority=priority,
        )
        self._tasks[task_id] = task
        return task

    def assign_task(self, task_id: str, agent_id: str) -> Task:
        """Assign a PENDING task to an agent.

        Transitions the task from PENDING to ASSIGNED.

        Args:
            task_id: The ID of the task to assign.
            agent_id: The ID of the agent to assign the task to.

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task_id does not exist.
            ValueError: If the task is not in PENDING status.
        """
        task = self.get_task(task_id)
        if task.status != TaskStatus.PENDING:
            raise ValueError(
                f"Task {task_id} is {task.status.value}, not PENDING"
            )
        task.assigned_to = agent_id
        task.status = TaskStatus.ASSIGNED
        return task

    def start_task(self, task_id: str) -> Task:
        """Start an ASSIGNED task.

        Transitions the task from ASSIGNED to IN_PROGRESS.

        Args:
            task_id: The ID of the task to start.

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task_id does not exist.
            ValueError: If the task is not in ASSIGNED status.
        """
        task = self.get_task(task_id)
        if task.status != TaskStatus.ASSIGNED:
            raise ValueError(
                f"Task {task_id} is {task.status.value}, not ASSIGNED"
            )
        task.status = TaskStatus.IN_PROGRESS
        return task

    def complete_task(self, task_id: str) -> Task:
        """Complete an IN_PROGRESS task.

        Transitions the task from IN_PROGRESS to COMPLETED and sets
        completed_at to the current timestamp.

        Args:
            task_id: The ID of the task to complete.

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task_id does not exist.
            ValueError: If the task is not in IN_PROGRESS status.
        """
        task = self.get_task(task_id)
        if task.status != TaskStatus.IN_PROGRESS:
            raise ValueError(
                f"Task {task_id} is {task.status.value}, not IN_PROGRESS"
            )
        task.status = TaskStatus.COMPLETED
        task.completed_at = int(time.time())
        return task

    def fail_task(self, task_id: str) -> Task:
        """Fail an IN_PROGRESS task.

        Transitions the task from IN_PROGRESS to FAILED.

        Args:
            task_id: The ID of the task to fail.

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task_id does not exist.
            ValueError: If the task is not in IN_PROGRESS status.
        """
        task = self.get_task(task_id)
        if task.status != TaskStatus.IN_PROGRESS:
            raise ValueError(
                f"Task {task_id} is {task.status.value}, not IN_PROGRESS"
            )
        task.status = TaskStatus.FAILED
        return task

    def get_task(self, task_id: str) -> Task:
        """Retrieve a task by its ID.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            The matching Task.

        Raises:
            KeyError: If no task with that ID exists.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        return self._tasks[task_id]

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Return all tasks with the given status.

        Args:
            status: The TaskStatus to filter by.

        Returns:
            A list of matching tasks.
        """
        return [t for t in self._tasks.values() if t.status == status]

    def get_tasks_by_agent(self, agent_id: str) -> list[Task]:
        """Return all tasks assigned to the given agent.

        Args:
            agent_id: The agent identifier to filter by.

        Returns:
            A list of matching tasks.
        """
        return [t for t in self._tasks.values() if t.assigned_to == agent_id]

    def get_pending_tasks(self) -> list[Task]:
        """Return all tasks with PENDING status.

        Returns:
            A list of pending tasks.
        """
        return self.get_tasks_by_status(TaskStatus.PENDING)

    def all_tasks(self) -> list[Task]:
        """Return all tasks managed by this TaskManager.

        Returns:
            A list of all tasks.
        """
        return list(self._tasks.values())
