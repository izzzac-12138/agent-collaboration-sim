"""EventLogger for recording simulation events as JSONL."""

import json
import os
from datetime import datetime
from typing import Any, Optional


class EventLogger:
    """Records simulation events in memory and optionally persists them as JSONL.

    Events are buffered in ``_events`` and written to disk on :meth:`flush`.
    Each event is a dict with keys ``event_type``, ``agent_id``, ``payload``,
    and ``timestamp``.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """Initialise the logger.

        Args:
            output_dir: Directory for JSONL output.  If *None* events are kept
                in memory only (never flushed to disk).
        """
        self._events: list[dict[str, Any]] = []
        self._output_dir: Optional[str] = output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        agent_id: str,
        payload: dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create an event dict and append it to the buffer.

        Args:
            event_type: Category of the event (e.g. ``"agent_step"``).
            agent_id: Identifier of the agent involved.
            payload: Arbitrary data attached to the event.
            timestamp: ISO-8601 timestamp.  Defaults to
                ``datetime.utcnow().isoformat()``.

        Returns:
            The newly created event dict.
        """
        event: dict[str, Any] = {
            "event_type": event_type,
            "agent_id": agent_id,
            "payload": payload,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
        self._events.append(event)
        return event

    # ------------------------------------------------------------------
    # Log methods
    # ------------------------------------------------------------------

    def log_agent_step(self, agent: Any) -> dict[str, Any]:
        """Record a single agent step.

        Args:
            agent: An agent object expected to expose ``role`` and various
                counter attributes (``tasks_completed``, ``messages_sent``,
                ``resources_acquired``, etc.).

        Returns:
            The emitted event dict.
        """
        payload: dict[str, Any] = {
            "role": getattr(agent, "role", "unknown"),
            "tasks_completed": getattr(agent, "tasks_completed", 0),
            "messages_sent": getattr(agent, "messages_sent", 0),
            "resources_acquired": getattr(agent, "resources_acquired", 0),
        }
        return self._emit("agent_step", agent.unique_id, payload)

    def log_task_created(self, task: Any, timestamp: Optional[str] = None) -> dict[str, Any]:
        """Record task creation.

        Args:
            task: A task object with ``task_id``, ``title``, ``required_role``,
                and ``complexity`` attributes.
            timestamp: Optional ISO-8601 timestamp override.

        Returns:
            The emitted event dict.
        """
        payload: dict[str, Any] = {
            "task_id": getattr(task, "task_id", "unknown"),
            "title": getattr(task, "title", ""),
            "required_role": getattr(task, "required_role", ""),
            "complexity": getattr(task, "complexity", 0),
        }
        return self._emit("task_created", "", payload, timestamp)

    def log_task_assigned(self, task: Any, timestamp: Optional[str] = None) -> dict[str, Any]:
        """Record task assignment to an agent.

        Args:
            task: A task object with ``task_id`` and ``assigned_to`` attributes.
            timestamp: Optional ISO-8601 timestamp override.

        Returns:
            The emitted event dict.
        """
        payload: dict[str, Any] = {
            "task_id": getattr(task, "task_id", "unknown"),
            "assigned_to": getattr(task, "assigned_to", ""),
        }
        return self._emit("task_assigned", getattr(task, "assigned_to", ""), payload, timestamp)

    def log_task_completed(self, task: Any, timestamp: Optional[str] = None) -> dict[str, Any]:
        """Record task completion.

        Args:
            task: A task object with ``task_id``, ``assigned_to``, and
                ``result`` attributes.
            timestamp: Optional ISO-8601 timestamp override.

        Returns:
            The emitted event dict.
        """
        payload: dict[str, Any] = {
            "task_id": getattr(task, "task_id", "unknown"),
            "assigned_to": getattr(task, "assigned_to", ""),
            "result": getattr(task, "result", None),
        }
        return self._emit("task_completed", getattr(task, "assigned_to", ""), payload, timestamp)

    def log_message_sent(self, message: Any, timestamp: Optional[str] = None) -> dict[str, Any]:
        """Record a message sent between agents.

        Args:
            message: A message object with ``sender``, ``receiver``, ``msg_type``,
                and ``content`` attributes.
            timestamp: Optional ISO-8601 timestamp override.

        Returns:
            The emitted event dict.
        """
        payload: dict[str, Any] = {
            "sender": getattr(message, "sender", ""),
            "receiver": getattr(message, "receiver", ""),
            "msg_type": getattr(message, "msg_type", ""),
            "content": getattr(message, "content", ""),
        }
        return self._emit("message_sent", getattr(message, "sender", ""), payload, timestamp)

    def log_resource_transferred(self, info: Any, timestamp: Optional[str] = None) -> dict[str, Any]:
        """Record a resource transfer between agents.

        Args:
            info: An info object with ``from_agent``, ``to_agent``,
                ``resource_type``, and ``amount`` attributes.
            timestamp: Optional ISO-8601 timestamp override.

        Returns:
            The emitted event dict.
        """
        payload: dict[str, Any] = {
            "from_agent": getattr(info, "from_agent", ""),
            "to_agent": getattr(info, "to_agent", ""),
            "resource_type": getattr(info, "resource_type", ""),
            "amount": getattr(info, "amount", 0),
        }
        return self._emit("resource_transferred", getattr(info, "from_agent", ""), payload, timestamp)

    def log_step_summary(self, model: Any) -> dict[str, Any]:
        """Record a per-step summary of the simulation state.

        Args:
            model: A model object expected to expose ``step``,
                ``communication_edges``, ``tasks`` (iterable of tasks),
                and ``messages`` (iterable of messages).

        Returns:
            The emitted event dict.
        """
        tasks = getattr(model, "tasks", [])
        messages = getattr(model, "messages", [])
        payload: dict[str, Any] = {
            "step": getattr(model, "step", 0),
            "edges": getattr(model, "communication_edges", []),
            "tasks_total": len(list(tasks)),
            "tasks_completed": sum(
                1 for t in tasks if getattr(t, "status", "") == "completed"
            ),
            "message_count": len(list(messages)),
        }
        return self._emit("step_summary", "", payload)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_events(self) -> list[dict[str, Any]]:
        """Return all buffered events.

        Returns:
            A list of event dicts.
        """
        return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return events matching *event_type*.

        Args:
            event_type: The ``event_type`` value to filter on.

        Returns:
            Matching event dicts.
        """
        return [e for e in self._events if e["event_type"] == event_type]

    def get_events_by_agent(self, agent_id: str) -> list[dict[str, Any]]:
        """Return events whose ``agent_id`` matches *agent_id*.

        Args:
            agent_id: The agent identifier to filter on.

        Returns:
            Matching event dicts.
        """
        return [e for e in self._events if e["agent_id"] == agent_id]

    def get_events_in_range(
        self, start: str, end: str
    ) -> list[dict[str, Any]]:
        """Return events with timestamps in the half-open range ``[start, end)``.

        Args:
            start: ISO-8601 start timestamp (inclusive).
            end: ISO-8601 end timestamp (exclusive).

        Returns:
            Matching event dicts.
        """
        return [
            e for e in self._events if start <= e["timestamp"] < end
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write buffered events to ``events.jsonl`` and clear the buffer.

        Events are appended as one JSON object per line.  Does nothing if
        ``output_dir`` is *None*.
        """
        if self._output_dir is None:
            self._events.clear()
            return

        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, "events.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events.clear()

    def save_summary(self, model: Any, filepath: str) -> None:
        """Persist a summary dict to a JSON file.

        Args:
            model: A model object whose attributes are serialised.  Expected
                keys include ``step``, ``agents``, ``tasks``, and ``messages``.
            filepath: Destination path for the JSON file.
        """
        summary: dict[str, Any] = {
            "step": getattr(model, "step", 0),
            "agent_count": len(getattr(model, "agents", [])),
            "task_count": len(getattr(model, "tasks", [])),
            "message_count": len(getattr(model, "messages", [])),
            "total_events": len(self._events),
        }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
