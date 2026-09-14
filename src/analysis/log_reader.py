"""JSONL log reader - loads simulation events into Pandas DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_events(filepath: str | Path) -> pd.DataFrame:
    """Read a JSONL file and return a DataFrame of simulation events.

    Each line is a JSON object with at least: timestamp, event_type, agent_id, payload.
    Malformed lines are skipped with a warning printed to stderr.

    Parameters
    ----------
    filepath : str or Path
        Path to the .jsonl file.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, event_type, agent_id, payload.
        Empty DataFrame with correct columns if file is missing or empty.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    records: list[dict[str, Any]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"WARN: skipping line {line_no}: {exc}")
                skipped += 1
                continue

            records.append(
                {
                    "timestamp": obj.get("timestamp", 0),
                    "event_type": obj.get("event_type", ""),
                    "agent_id": obj.get("agent_id", ""),
                    "payload": obj.get("payload", {}),
                }
            )

    if skipped:
        print(f"WARN: {skipped} malformed line(s) skipped in {path.name}")

    if not records:
        return pd.DataFrame(columns=["timestamp", "event_type", "agent_id", "payload"])

    return pd.DataFrame(records)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a simulation run directory containing events.jsonl and summary.json.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run directory.

    Returns
    -------
    dict
        ``{"events": pd.DataFrame, "summary": dict}``

    Raises
    ------
    FileNotFoundError
        If the run directory or events.jsonl is missing.
    """
    base = Path(run_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Run directory not found: {base}")

    events_path = base / "events.jsonl"
    events_df = load_events(events_path)

    summary_path = base / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fh:
            try:
                summary = json.load(fh)
            except json.JSONDecodeError:
                print(f"WARN: could not parse {summary_path}")
                summary = {}

    return {"events": events_df, "summary": summary}


def events_to_tasks_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """Filter task-related events and flatten their payloads.

    Identifies rows where ``event_type`` contains the substring ``"task"``
    (case-insensitive) and expands the ``payload`` dict into columns.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame as returned by :func:`load_events`.

    Returns
    -------
    pd.DataFrame
        Task events with payload fields merged as columns.
    """
    if events_df.empty:
        return events_df.copy()

    mask = events_df["event_type"].str.contains("task", case=False, na=False)
    task_events = events_df.loc[mask].copy()

    if task_events.empty:
        return pd.DataFrame(
            columns=["timestamp", "event_type", "agent_id", "payload"]
        )

    payload_expanded = pd.json_normalize(task_events["payload"])
    result = task_events.drop(columns=["payload"]).reset_index(drop=True)
    result = pd.concat([result, payload_expanded], axis=1)
    return result


def events_to_messages_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """Filter message events and extract sender, recipient, and content.

    Identifies rows where ``event_type`` contains the substring ``"message"``
    (case-insensitive) and extracts ``sender``, ``recipient``, and ``content``
    from the payload.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame as returned by :func:`load_events`.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, event_type, agent_id, sender, recipient, content.
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "event_type",
                "agent_id",
                "sender",
                "recipient",
                "content",
            ]
        )

    mask = events_df["event_type"].str.contains("message", case=False, na=False)
    msg_events = events_df.loc[mask].copy()

    if msg_events.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "event_type",
                "agent_id",
                "sender",
                "recipient",
                "content",
            ]
        )

    def _extract(payload: dict[str, Any]) -> pd.Series:
        if isinstance(payload, dict):
            return pd.Series(
                {
                    "sender": payload.get("sender", ""),
                    "recipient": payload.get("recipient", ""),
                    "content": payload.get("content", ""),
                }
            )
        return pd.Series({"sender": "", "recipient": "", "content": ""})

    extracted = msg_events["payload"].apply(_extract)
    result = msg_events.drop(columns=["payload"]).reset_index(drop=True)
    result = pd.concat([result, extracted], axis=1)
    return result


def events_to_network_snapshots(
    events_df: pd.DataFrame,
) -> dict[int, list[tuple[str, str, dict[str, Any]]]]:
    """Group message events by timestamp to form network edge snapshots.

    Each snapshot is a list of ``(sender, recipient, metadata)`` tuples
    representing directed edges at a given timestamp.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame as returned by :func:`load_events`.

    Returns
    -------
    dict[int, list[tuple[str, str, dict[str, Any]]]]
        Mapping from timestamp to list of edges. Timestamps are cast to int.
    """
    snapshots: dict[int, list[tuple[str, str, dict[str, Any]]]] = {}

    if events_df.empty:
        return snapshots

    mask = events_df["event_type"].str.contains("message", case=False, na=False)
    msg_events = events_df.loc[mask]

    for _, row in msg_events.iterrows():
        payload = row["payload"]
        if not isinstance(payload, dict):
            continue

        ts = int(row["timestamp"])
        sender = str(payload.get("sender", ""))
        recipient = str(payload.get("recipient", ""))
        meta = {
            k: v
            for k, v in payload.items()
            if k not in ("sender", "recipient")
        }

        snapshots.setdefault(ts, []).append((sender, recipient, meta))

    return snapshots


def compute_agent_stats(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-agent statistics from the events DataFrame.

    Statistics computed:
    - ``event_count``: total number of events for each agent
    - ``event_type_counts``: dict mapping event_type to count
    - ``first_seen``: earliest timestamp
    - ``last_seen``: latest timestamp
    - ``timespan``: last_seen - first_seen

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame as returned by :func:`load_events`.

    Returns
    -------
    pd.DataFrame
        One row per agent_id with the statistics above.
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "agent_id",
                "event_count",
                "event_type_counts",
                "first_seen",
                "last_seen",
                "timespan",
            ]
        )

    rows: list[dict[str, Any]] = []

    for agent_id, group in events_df.groupby("agent_id"):
        type_counts: dict[str, int] = {}
        for etype in group["event_type"]:
            key = str(etype)
            type_counts[key] = type_counts.get(key, 0) + 1

        timestamps = pd.to_numeric(group["timestamp"], errors="coerce").dropna()
        first_seen = int(timestamps.min()) if not timestamps.empty else 0
        last_seen = int(timestamps.max()) if not timestamps.empty else 0

        rows.append(
            {
                "agent_id": str(agent_id),
                "event_count": len(group),
                "event_type_counts": type_counts,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "timespan": last_seen - first_seen,
            }
        )

    return pd.DataFrame(rows)
