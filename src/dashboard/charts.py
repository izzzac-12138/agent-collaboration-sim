"""Chart generators for the dashboard using matplotlib."""

from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_task_progression(events_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot cumulative task completions over time.

    Filters ``task_completed`` events, groups by timestamp, and shows
    a running total as a line chart.

    Parameters
    ----------
    events_df:
        Simulation event log with columns ``event_type``, ``timestamp``.
    ax:
        Optional matplotlib Axes to draw on.  If *None* a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    task_events = events_df[events_df["event_type"] == "task_completed"]

    if task_events.empty:
        ax.set_title("Task Completion Over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cumulative Tasks Completed")
        fig.tight_layout()
        return fig

    timestamps = task_events["timestamp"].sort_values().values
    cumulative = np.arange(1, len(timestamps) + 1)

    ax.plot(timestamps, cumulative, marker="o", markersize=3, linewidth=1.5)
    ax.set_title("Task Completion Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Tasks Completed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_agent_activity(events_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Horizontal bar chart showing the top 10 most active agents.

    Activity is measured as the sum of ``messages_sent`` and
    ``tasks_completed`` from ``agent_step`` events.

    Parameters
    ----------
    events_df:
        Simulation event log.
    ax:
        Optional matplotlib Axes.  If *None* a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    agent_steps = events_df[events_df["event_type"] == "agent_step"]

    if agent_steps.empty:
        ax.set_title("Agent Activity (Top 10)")
        fig.tight_layout()
        return fig

    def _sum_metric(payload, key: str) -> int:
        if isinstance(payload, dict):
            return int(payload.get(key, 0))
        return 0

    activity: dict[str, int] = {}
    for _, row in agent_steps.iterrows():
        aid = str(row.get("agent_id", ""))
        payload = row.get("payload", {})
        msgs = _sum_metric(payload, "messages_sent")
        tasks = _sum_metric(payload, "tasks_completed")
        activity[aid] = activity.get(aid, 0) + msgs + tasks

    sorted_agents = sorted(activity.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_agents:
        ax.set_title("Agent Activity (Top 10)")
        fig.tight_layout()
        return fig

    names = [a[0] for a in sorted_agents][::-1]
    values = [a[1] for a in sorted_agents][::-1]

    ax.barh(names, values, color="#4C72B0")
    ax.set_title("Agent Activity (Top 10)")
    ax.set_xlabel("Messages Sent + Tasks Completed")
    fig.tight_layout()
    return fig


def plot_network_evolution(
    events_df: pd.DataFrame, max_snapshots: int = 4
) -> plt.Figure:
    """Show the communication network at different time windows.

    Creates a 2x2 grid of subplots.  Each subplot contains a spring layout
    graph built from ``message_sent`` events in a roughly equal time window.

    Parameters
    ----------
    events_df:
        Simulation event log.
    max_snapshots:
        Number of time windows (must be <= 4).

    Returns
    -------
    matplotlib.figure.Figure
    """
    max_snapshots = min(max_snapshots, 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    message_events = events_df[events_df["event_type"] == "message_sent"]

    if message_events.empty:
        for i in range(max_snapshots):
            axes_flat[i].set_title(f"Window {i + 1} (no data)")
            axes_flat[i].axis("off")
        fig.suptitle("Network Evolution", fontsize=14)
        fig.tight_layout()
        return fig

    timestamps = message_events["timestamp"].sort_values().values
    t_min, t_max = timestamps[0], timestamps[-1]
    span = t_max - t_min if t_max != t_min else 1
    window_size = span / max_snapshots

    for i in range(max_snapshots):
        ax = axes_flat[i]
        win_start = t_min + i * window_size
        win_end = t_min + (i + 1) * window_size

        win_msgs = message_events[
            (message_events["timestamp"] >= win_start) &
            (message_events["timestamp"] < win_end)
        ]

        G = nx.Graph()
        for _, row in win_msgs.iterrows():
            payload = row.get("payload", {})
            if isinstance(payload, dict):
                sender = payload.get("sender") or payload.get("from_agent", "")
                receiver = payload.get("receiver") or payload.get("to_agent", "")
            else:
                sender = row.get("agent_id", "")
                receiver = ""
            if sender and receiver:
                G.add_node(sender)
                G.add_node(receiver)
                if G.has_edge(sender, receiver):
                    G[sender][receiver]["weight"] += 1
                else:
                    G.add_edge(sender, receiver, weight=1)

        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=42)
            degrees = dict(G.degree())
            node_sizes = [max(degrees[n] * 200, 80) for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="#4C72B0", alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color="gray")
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        else:
            ax.text(0.5, 0.5, "No connections", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"Window {i + 1}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Network Evolution", fontsize=14)
    fig.tight_layout()
    return fig


def plot_clustering_results(
    cluster_profiles: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Bar chart of cluster feature profiles.

    One group of bars per cluster, one bar per numeric feature column.
    The x-axis is labeled with the ``role_label`` column if present.

    Parameters
    ----------
    cluster_profiles:
        DataFrame from ``AgentClusterer.get_cluster_profiles`` with a
        ``role_label`` column and numeric feature columns.
    ax:
        Optional matplotlib Axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if cluster_profiles.empty:
        ax.set_title("Cluster Profiles")
        fig.tight_layout()
        return fig

    numeric_cols = [c for c in cluster_profiles.columns if c not in ("count", "role_label")]
    if not numeric_cols:
        ax.set_title("Cluster Profiles")
        fig.tight_layout()
        return fig

    role_labels = cluster_profiles.get("role_label")
    if role_labels is not None:
        x_labels = [str(v) for v in role_labels.values]
    else:
        x_labels = [str(idx) for idx in cluster_profiles.index]

    n_clusters = len(cluster_profiles)
    n_features = len(numeric_cols)
    x = np.arange(n_clusters)
    width = 0.8 / max(n_features, 1)

    for j, col in enumerate(numeric_cols):
        offset = (j - n_features / 2 + 0.5) * width
        values = cluster_profiles[col].values.astype(float)
        ax.bar(x + offset, values, width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title("Cluster Profiles")
    ax.set_ylabel("Mean Feature Value")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_anomaly_detection(
    temporal_results: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Scatter plot of temporal anomaly scores over time.

    Anomalous points are highlighted in red.

    Parameters
    ----------
    temporal_results:
        DataFrame with columns ``window_id``, ``anomaly_score``,
        and ``is_anomaly``.
    ax:
        Optional matplotlib Axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    if temporal_results.empty:
        ax.set_title("Anomaly Detection Over Time")
        fig.tight_layout()
        return fig

    normal = temporal_results[~temporal_results["is_anomaly"]]
    anomalous = temporal_results[temporal_results["is_anomaly"]]

    ax.scatter(normal["window_id"], normal["anomaly_score"],
               c="#4C72B0", s=30, alpha=0.6, label="Normal", zorder=2)
    if not anomalous.empty:
        ax.scatter(anomalous["window_id"], anomalous["anomaly_score"],
                   c="red", s=60, marker="x", linewidths=2, label="Anomaly", zorder=3)

    ax.set_title("Anomaly Detection Over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Anomaly Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_message_type_distribution(
    events_df: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Pie chart of message types from ``message_sent`` events.

    Parameters
    ----------
    events_df:
        Simulation event log.
    ax:
        Optional matplotlib Axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    message_events = events_df[events_df["event_type"] == "message_sent"]

    if message_events.empty:
        ax.set_title("Message Type Distribution")
        ax.text(0.5, 0.5, "No message events", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    msg_types: dict[str, int] = {}
    for _, row in message_events.iterrows():
        payload = row.get("payload", {})
        if isinstance(payload, dict):
            mt = str(payload.get("msg_type", "unknown"))
        else:
            mt = "unknown"
        msg_types[mt] = msg_types.get(mt, 0) + 1

    labels = list(msg_types.keys())
    sizes = list(msg_types.values())

    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Message Type Distribution")
    fig.tight_layout()
    return fig
