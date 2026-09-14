"""Feature extraction from simulation event logs for mining algorithms."""

from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd


def extract_agent_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-agent feature vectors from simulation event logs.

    For each unique agent_id, compute behavioral features such as message
    counts, task outcomes, collaboration diversity, response times, and
    resource contributions.

    Args:
        events_df: DataFrame with columns (timestamp, event_type, agent_id, payload).
            The payload column should contain dicts or objects with attributes
            like ``to_agent``, ``from_agent``, ``task_name``, ``status``, etc.

    Returns:
        DataFrame with agent_id as index and numeric feature columns only.
        Returns an empty DataFrame with the expected columns if input is empty.
    """
    numeric_cols = [
        "messages_sent",
        "messages_received",
        "tasks_completed",
        "tasks_failed",
        "unique_collaborators",
        "avg_response_time",
        "resource_contributions",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=numeric_cols).rename_axis("agent_id")

    df = events_df.copy()

    # Normalize payload: if payload is a dict-like, we can use .get() / []
    # For Series of dicts, convert to DataFrame for easier access
    if not df.empty and "payload" in df.columns:
        # Ensure payload is treated as dict-like
        payload_df = pd.json_normalize(df["payload"].apply(
            lambda p: p if isinstance(p, dict) else {}
        ))
        df = pd.concat([df.drop(columns=["payload"]), payload_df], axis=1)
    else:
        # Add empty columns so downstream logic works uniformly
        for col in ("to_agent", "from_agent", "task_name", "status", "role"):
            if col not in df.columns:
                df[col] = None

    # Fill missing structural columns
    for col in ("to_agent", "from_agent", "role"):
        if col not in df.columns:
            df[col] = None
    if "status" not in df.columns:
        df["status"] = None

    all_agent_ids = df["agent_id"].dropna().unique()
    rows = []

    for agent_id in sorted(all_agent_ids):
        # --- messages_sent ---
        sent_mask = (df["event_type"] == "message_sent") & (df["from_agent"] == agent_id)
        messages_sent = int(sent_mask.sum())

        # --- messages_received ---
        received_mask = (df["event_type"] == "message_sent") & (df["to_agent"] == agent_id)
        messages_received = int(received_mask.sum())

        # --- tasks_completed / tasks_failed ---
        tasks_mask = df["agent_id"] == agent_id
        tasks_completed = int(
            ((df["event_type"] == "task_completed") & tasks_mask).sum()
        )
        tasks_failed = int(
            ((df["event_type"] == "task_failed") & tasks_mask).sum()
        )

        # --- unique_collaborators ---
        sent_to = df.loc[sent_mask, "to_agent"].dropna().unique()
        received_from = df.loc[received_mask, "from_agent"].dropna().unique()
        collaborators = set(sent_to) | set(received_from)
        collaborators.discard(agent_id)
        unique_collaborators = len(collaborators)

        # --- avg_response_time ---
        # Approximation: for each message received, find the next message sent
        # by the same agent, compute time diff.
        received_events = (
            df.loc[received_mask, ["timestamp"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        sent_events = (
            df.loc[sent_mask, ["timestamp"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        avg_response_time = 0.0
        if not received_events.empty and not sent_events.empty:
            response_times = []
            for recv_ts in received_events["timestamp"]:
                # Find the next sent message after this received message
                later_sends = sent_events[sent_events["timestamp"] > recv_ts]
                if not later_sends.empty:
                    next_send_ts = later_sends.iloc[0]["timestamp"]
                    delta = next_send_ts - recv_ts
                    diff = (
                        delta.total_seconds()
                        if hasattr(delta, "total_seconds")
                        else float(delta)
                    )
                    response_times.append(diff)
            if response_times:
                avg_response_time = float(np.mean(response_times))

        # --- resource_contributions ---
        resource_mask = (df["event_type"] == "resource_transferred") & (
            df["from_agent"] == agent_id
        )
        resource_contributions = int(resource_mask.sum())

        rows.append(
            {
                "agent_id": agent_id,
                "messages_sent": messages_sent,
                "messages_received": messages_received,
                "tasks_completed": tasks_completed,
                "tasks_failed": tasks_failed,
                "unique_collaborators": unique_collaborators,
                "avg_response_time": avg_response_time,
                "resource_contributions": resource_contributions,
            }
        )

    if not rows:
        return pd.DataFrame(columns=numeric_cols).rename_axis("agent_id")

    result = pd.DataFrame(rows).set_index("agent_id")
    return result[numeric_cols]


def extract_temporal_features(
    events_df: pd.DataFrame, window_size: int = 10
) -> pd.DataFrame:
    """Extract temporal features by splitting events into time windows.

    Args:
        events_df: DataFrame with columns (timestamp, event_type, agent_id, payload).
            May also contain ``step_summary`` events with network density info
            in the payload.
        window_size: Number of steps per window. Defaults to 10.

    Returns:
        DataFrame with window_id as index and columns:
        message_rate, task_completion_rate, active_agents, network_density.
        Returns an empty DataFrame with expected columns if input is empty.
    """
    cols = ["message_rate", "task_completion_rate", "active_agents", "network_density"]

    if events_df.empty:
        return pd.DataFrame(columns=cols).rename_axis("window_id")

    df = events_df.copy()
    if "step" not in df.columns:
        # Try to infer step from timestamp ordering
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["step"] = np.arange(len(df))

    # Normalize payload for network density extraction
    if "payload" in df.columns:
        payload_df = pd.json_normalize(df["payload"].apply(
            lambda p: p if isinstance(p, dict) else {}
        ))
        df = pd.concat([df.drop(columns=["payload"]), payload_df], axis=1)
    else:
        for col in ("edges_formed", "possible_edges"):
            if col not in df.columns:
                df[col] = None

    max_step = df["step"].max()
    rows = []

    window_id = 0
    start = df["step"].min()
    while start <= max_step:
        end = start + window_size
        window_mask = (df["step"] >= start) & (df["step"] < end)
        window_df = df[window_mask]

        if window_df.empty:
            start = end
            window_id += 1
            continue

        message_count = int((window_df["event_type"] == "message_sent").sum())
        task_completed_count = int((window_df["event_type"] == "task_completed").sum())
        active_agents = window_df["agent_id"].nunique()

        # Network density from step_summary events
        network_density = 0.0
        step_summaries = window_df[window_df["event_type"] == "step_summary"]
        if not step_summaries.empty and "edges_formed" in step_summaries.columns:
            total_edges = step_summaries["edges_formed"].sum()
            total_possible = step_summaries["possible_edges"].sum()
            if total_possible > 0:
                network_density = float(total_edges / total_possible)

        rows.append(
            {
                "window_id": window_id,
                "message_rate": message_count / max(window_size, 1),
                "task_completion_rate": task_completed_count / max(window_size, 1),
                "active_agents": active_agents,
                "network_density": network_density,
            }
        )

        start = end
        window_id += 1

    if not rows:
        return pd.DataFrame(columns=cols).rename_axis("window_id")

    result = pd.DataFrame(rows).set_index("window_id")
    return result[cols]


def build_action_sequences(
    events_df: pd.DataFrame, agent_id: Optional[str] = None
) -> List[List[str]]:
    """Extract ordered action sequences per agent from event logs.

    Each action is a descriptive string combining event_type with relevant
    payload info, e.g. ``"task_completed"``, ``"message_sent:confirm"``,
    ``"resource_transferred"``.

    Args:
        events_df: DataFrame with columns (timestamp, event_type, agent_id, payload).
        agent_id: If provided, return sequences only for this agent.
            If None, return sequences for all agents.

    Returns:
        List of lists, where each inner list is an ordered action sequence
        for one agent. Returns an empty list if input is empty or no
        matching agents are found.
    """
    if events_df.empty:
        return []

    df = events_df.copy()

    if "payload" in df.columns:
        payload_df = pd.json_normalize(df["payload"].apply(
            lambda p: p if isinstance(p, dict) else {}
        ))
        df = pd.concat([df.drop(columns=["payload"]), payload_df], axis=1)
    else:
        for col in ("action", "message_type", "task_name", "status"):
            if col not in df.columns:
                df[col] = None

    # Filter by agent_id if specified
    if agent_id is not None:
        # Include events where agent_id matches either agent_id column (direct)
        # or from_agent/to_agent (communication events)
        mask = df["agent_id"] == agent_id
        if "from_agent" in df.columns:
            mask = mask | (df["from_agent"] == agent_id)
        df = df[mask]

    if df.empty:
        return []

    def _safe_str(val) -> str:
        """Return a clean string or empty string for None/NaN."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        s = str(val).strip()
        return "" if s == "nan" else s

    def _make_action(row: pd.Series) -> str:
        """Build an action string from an event row."""
        event_type = str(row.get("event_type", "unknown"))

        # Enrich action string with payload context
        if event_type == "message_sent":
            msg_type = _safe_str(row.get("message_type")) or _safe_str(row.get("action"))
            if msg_type:
                return f"message_sent:{msg_type}"
            return "message_sent"
        elif event_type == "resource_transferred":
            resource = _safe_str(row.get("resource")) or _safe_str(row.get("resource_type"))
            if resource:
                return f"resource_transferred:{resource}"
            return "resource_transferred"
        elif event_type == "task_completed":
            task = row.get("task_name") or ""
            if task:
                return f"task_completed:{task}"
            return "task_completed"
        elif event_type == "task_failed":
            task = row.get("task_name") or ""
            if task:
                return f"task_failed:{task}"
            return "task_failed"
        else:
            return event_type

    # Sort by timestamp for correct ordering
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Group by agent_id (using the main agent_id column)
    sequences: List[List[str]] = []

    if agent_id is not None:
        # Single agent: return one sequence
        actions = df.apply(_make_action, axis=1).tolist()
        if actions:
            sequences.append(actions)
    else:
        # All agents
        for aid, group in df.groupby("agent_id"):
            if pd.isna(aid):
                continue
            actions = group.apply(_make_action, axis=1).tolist()
            if actions:
                sequences.append(actions)

    return sequences
