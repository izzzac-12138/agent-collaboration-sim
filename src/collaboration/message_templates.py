"""Natural language message template system for agent communication.

Provides randomized template-based message generation with optional
LLM integration for more natural dialogue.
"""

from __future__ import annotations

import random
from typing import Any

# --- Template library ---------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "request": [
        "I need help with {task}. Can anyone assist?",
        "Requesting support for {task}. Required: {resources}.",
        "{task} needs attention. Who can contribute {resources}?",
        "Looking for collaborators on {task}. We need {resources}.",
        "Urgent: {task} is pending. Need {resources} immediately.",
    ],
    "confirm": [
        "Got it. I'll handle {task} right away.",
        "Confirmed - starting work on {task} now.",
        "Roger that. {task} is on my way.",
        "Understood. I'll take care of {task}.",
        "On it. {task} will be completed shortly.",
    ],
    "deny": [
        "Sorry, I'm swamped with other tasks right now.",
        "Can't take {task} at the moment - too busy.",
        "I'm unavailable for {task}. Try someone else.",
        "Negative on {task}. I'm at capacity.",
        "Unable to assist with {task} right now.",
    ],
    "inform": [
        "FYI: {info}",
        "Heads up - {info}",
        "Just so you know: {info}",
        "Information update: {info}",
        "Noting for the team: {info}",
    ],
    "resource_share": [
        "Sharing {amount} {resource} with you. Use it wisely.",
        "Sending {amount} {resource} your way.",
        "Here's {amount} {resource} for your task.",
        "Transferring {amount} {resource} - let me know if you need more.",
        "Dispatching {amount} {resource}. Confirm receipt.",
    ],
    "alert": [
        "[!] Alert: {message}",
        "Warning: {message}",
        "Attention all agents: {message}",
        "Critical notice: {message}",
        "[!] {message}. Please act accordingly.",
    ],
}


def generate_message(
    msg_type: str,
    variables: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a natural-language message from a template.

    Parameters
    ----------
    msg_type:
        One of the keys in ``TEMPLATES`` (request, confirm, deny, etc.).
    variables:
        Placeholder values to fill in the template (e.g. ``{"task": "T001"}``).
    rng:
        Optional ``random.Random`` instance for reproducibility.

    Returns
    -------
    str
        A message string with placeholders filled in.
    """
    rng = rng or random
    templates = TEMPLATES.get(msg_type, [f"[{msg_type}]"])
    template = rng.choice(templates)
    variables = variables or {}
    try:
        return template.format(**variables)
    except KeyError:
        return template


def generate_conversation(
    participants: list[str],
    topic: str,
    n_turns: int = 4,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Generate a short simulated conversation between agents.

    Useful for creating richer non-structured communication data.
    Returns a list of message dicts suitable for CommBus.create_and_send.
    """
    rng = rng or random
    messages = []
    types_sequence = ["inform", "request", "confirm", "inform"]

    for i in range(min(n_turns, len(participants) * 2)):
        sender = rng.choice(participants)
        receiver = rng.choice([p for p in participants if p != sender])
        msg_type = types_sequence[i % len(types_sequence)]

        content = generate_message(
            msg_type,
            {"task": topic, "info": topic, "message": topic},
            rng=rng,
        )
        messages.append({
            "from_agent": sender,
            "to_agent": receiver,
            "msg_type": msg_type,
            "content": content,
        })

    return messages
