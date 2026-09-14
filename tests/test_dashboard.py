"""Unit tests for the S4 visualization dashboard."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dashboard.charts import (  # noqa: E402
    plot_task_progression,
    plot_agent_activity,
    plot_network_evolution,
    plot_message_type_distribution,
)
from src.dashboard.report import generate_report  # noqa: E402


# ------------------------------------------------------------------
# Helper: synthetic events
# ------------------------------------------------------------------


def _make_mock_events(n_steps: int = 20, n_agents: int = 6) -> pd.DataFrame:
    """Create a DataFrame mimicking a JSONL events log for testing charts."""
    rng = np.random.RandomState(42)
    agents = [f"agent_{i}" for i in range(n_agents)]
    records: list[dict] = []

    for step in range(n_steps):
        # message_sent events
        for _ in range(3):
            frm = rng.choice(agents)
            to = rng.choice([a for a in agents if a != frm])
            records.append({
                "timestamp": step,
                "event_type": "message_sent",
                "agent_id": frm,
                "payload": {
                    "from_agent": frm,
                    "to_agent": to,
                    "content": f"msg from {frm} to {to}",
                    "msg_type": rng.choice(["inform", "request", "confirm"]),
                },
            })

        # task_completed events (intermittent)
        if rng.random() < 0.4:
            records.append({
                "timestamp": step,
                "event_type": "task_completed",
                "agent_id": rng.choice(agents),
                "payload": {"task_id": f"T{step:03d}"},
            })

        # step_summary events
        records.append({
            "timestamp": step,
            "event_type": "step_summary",
            "agent_id": "",
            "payload": {
                "step": step,
                "n_edges": int(rng.randint(3, 15)),
                "tasks_completed": int(rng.randint(0, 5)),
                "network_density": float(rng.uniform(0.05, 0.4)),
            },
        })

        # agent_step events
        for agent in rng.choice(agents, size=min(3, n_agents), replace=False):
            records.append({
                "timestamp": step,
                "event_type": "agent_step",
                "agent_id": agent,
                "payload": {
                    "role": rng.choice(["leader", "scout", "worker"]),
                    "tasks_completed": int(rng.randint(0, 3)),
                    "messages_sent": int(rng.randint(0, 5)),
                },
            })

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# 1. plot_task_progression
# ------------------------------------------------------------------


def test_plot_task_progression():
    """Generate a task progression chart and verify it returns a Figure."""
    df = _make_mock_events()
    fig = plot_task_progression(df)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ------------------------------------------------------------------
# 2. plot_agent_activity
# ------------------------------------------------------------------


def test_plot_agent_activity():
    """Generate an agent activity chart and verify it returns a Figure."""
    df = _make_mock_events()
    fig = plot_agent_activity(df)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ------------------------------------------------------------------
# 3. plot_network_evolution
# ------------------------------------------------------------------


def test_plot_network_evolution():
    """Generate a network evolution chart and verify it returns a Figure."""
    df = _make_mock_events()
    fig = plot_network_evolution(df)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ------------------------------------------------------------------
# 4. plot_message_type_distribution
# ------------------------------------------------------------------


def test_plot_message_type_distribution():
    """Generate a message type distribution chart and verify it returns a Figure."""
    df = _make_mock_events()
    fig = plot_message_type_distribution(df)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ------------------------------------------------------------------
# 5. generate_report (HTML output)
# ------------------------------------------------------------------


def test_generate_report():
    """Write mock events to a temp JSONL, generate HTML report, verify output."""
    df = _make_mock_events()

    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = os.path.join(tmpdir, "events.jsonl")
        with open(events_path, "w", encoding="utf-8") as fh:
            for _, row in df.iterrows():
                fh.write(row.to_json(force_ascii=False) + "\n")

        output_path = os.path.join(tmpdir, "report.html")
        generate_report(tmpdir, output_path)

        assert os.path.exists(output_path), "HTML report file was not created"
        size = os.path.getsize(output_path)
        assert size > 0, "HTML report file is empty"


# ------------------------------------------------------------------
# 6. chart with empty data
# ------------------------------------------------------------------


def test_chart_with_empty_data():
    """Pass an empty DataFrame to chart functions; verify no crash."""
    empty_df = pd.DataFrame(
        columns=["timestamp", "event_type", "agent_id", "payload"]
    )

    # Each chart function should gracefully handle an empty input
    for fn in (
        plot_task_progression,
        plot_agent_activity,
        plot_network_evolution,
        plot_message_type_distribution,
    ):
        fig = fn(empty_df)
        assert isinstance(fig, Figure)
        plt.close(fig)
