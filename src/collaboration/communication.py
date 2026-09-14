"""Message and CommBus for agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message exchanged between agents.

    Attributes:
        msg_id: Unique identifier for the message (e.g. ``"msg_0001"``).
        from_agent: ID of the sending agent.
        to_agent: ID of the receiving agent, or ``None`` for broadcast.
        msg_type: One of ``request``, ``confirm``, ``deny``, ``inform``,
            ``resource_share``, ``alert``.
        content: The body of the message.
        timestamp: Simulation tick or timestamp when the message was created.
    """

    msg_id: str
    from_agent: str
    to_agent: str | None = None
    msg_type: str = ""
    content: str = ""
    timestamp: int = 0


class CommBus:
    """Communication bus that routes messages between agents.

    Agents deposit outbound messages via :meth:`send` and later retrieve
    their inbox via :meth:`receive`.  A ``None`` recipient causes the
    message to be delivered to *every* agent's inbox except the sender.
    """

    def __init__(self) -> None:
        self._message_log: list[Message] = []
        self._inbox: dict[str, list[Message]] = {}

    # ------------------------------------------------------------------
    # Core send / receive
    # ------------------------------------------------------------------

    def send(self, message: Message) -> None:
        """Append *message* to the global log and deliver it.

        If ``message.to_agent`` is ``None`` the message is **broadcast**:
        it is placed in the inbox of every agent that has previously
        registered (i.e. called :meth:`receive` at least once), **except**
        the sender.  Otherwise it is delivered only to the named recipient.
        """
        self._message_log.append(message)

        if message.to_agent is None:
            for agent_id in self._inbox:
                if agent_id != message.from_agent:
                    self._inbox[agent_id].append(message)
        else:
            self._inbox.setdefault(message.to_agent, []).append(message)

    def create_and_send(
        self,
        from_agent: str,
        to_agent: str | None,
        msg_type: str,
        content: str,
        timestamp: int,
    ) -> Message:
        """Build a :class:`Message` with an auto-generated id, send it, and
        return it.

        Message IDs follow the pattern ``msg_0001``, ``msg_0002``, etc.,
        based on the current length of the message log.
        """
        next_num = len(self._message_log) + 1
        msg_id = f"msg_{next_num:04d}"
        message = Message(
            msg_id=msg_id,
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            content=content,
            timestamp=timestamp,
        )
        self.send(message)
        return message

    def receive(self, agent_id: str) -> list[Message]:
        """Return all pending messages for *agent_id* and clear its inbox.

        The agent is automatically registered the first time this method
        is called (enabling future broadcasts to reach it).
        """
        self._inbox.setdefault(agent_id, [])
        pending = self._inbox[agent_id]
        result = list(pending)
        self._inbox[agent_id] = []
        return result

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_all_messages(self) -> list[Message]:
        """Return every message that has been sent through this bus."""
        return list(self._message_log)

    def get_messages_between(self, agent_a: str, agent_b: str) -> list[Message]:
        """Return messages where *agent_a* and *agent_b* appear as sender
        and receiver (in either direction)."""
        return [
            m
            for m in self._message_log
            if {m.from_agent, m.to_agent} == {agent_a, agent_b}
        ]

    def get_messages_by_type(self, msg_type: str) -> list[Message]:
        """Return all messages whose ``msg_type`` matches *msg_type*."""
        return [m for m in self._message_log if m.msg_type == msg_type]
