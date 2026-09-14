"""HTML report generator that combines all charts into a single report."""

from __future__ import annotations

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .charts import (  # noqa: E402
    plot_agent_activity,
    plot_anomaly_detection,
    plot_clustering_results,
    plot_message_type_distribution,
    plot_network_evolution,
    plot_task_progression,
)
from src.analysis.information_flow import InformationTracer  # noqa: E402
from src.analysis.semantic import SemanticAnalyzer  # noqa: E402


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def generate_report(run_dir: str, output_path: str | None = None) -> str:
    """Generate a full HTML report from a simulation run directory.

    Loads ``events.jsonl`` (and optionally ``summary.json``) from *run_dir*,
    produces every chart defined in ``charts.py``, runs lightweight
    information-flow and semantic analyses, and writes a self-contained
    HTML file with embedded base64 images.

    Parameters
    ----------
    run_dir:
        Path to a directory containing ``events.jsonl`` and optionally
        ``summary.json``.
    output_path:
        Where to write the HTML file.  Defaults to ``run_dir/report.html``.

    Returns
    -------
    str
        The absolute path of the written report file.
    """
    run_dir = os.path.abspath(run_dir)
    if output_path is None:
        output_path = os.path.join(run_dir, "report.html")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    events_path = os.path.join(run_dir, "events.jsonl")
    records: list[dict] = []
    with open(events_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    events_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["timestamp", "event_type", "agent_id", "payload"]
    )

    summary: dict = {}
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            try:
                summary = json.load(fh)
            except json.JSONDecodeError:
                summary = {}

    # ------------------------------------------------------------------
    # 2. Build Overview section
    # ------------------------------------------------------------------
    if summary:
        rows = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items()
        )
    else:
        n_events = len(events_df)
        n_types = events_df["event_type"].nunique() if not events_df.empty else 0
        n_agents = events_df["agent_id"].nunique() if not events_df.empty else 0
        rows = (
            f"<tr><th>Total Events</th><td>{n_events}</td></tr>"
            f"<tr><th>Event Types</th><td>{n_types}</td></tr>"
            f"<tr><th>Unique Agents</th><td>{n_agents}</td></tr>"
        )

    overview_section = (
        "<h2>Overview</h2>\n"
        f'<table class="overview-table">{rows}</table>'
    )

    # ------------------------------------------------------------------
    # 3. Generate chart images
    # ------------------------------------------------------------------
    chart_images: list[tuple[str, str]] = []

    fig = plot_task_progression(events_df)
    chart_images.append(("Task Progression", _fig_to_base64(fig)))

    fig = plot_agent_activity(events_df)
    chart_images.append(("Agent Activity", _fig_to_base64(fig)))

    fig = plot_network_evolution(events_df)
    chart_images.append(("Network Evolution", _fig_to_base64(fig)))

    fig = plot_message_type_distribution(events_df)
    chart_images.append(("Message Type Distribution", _fig_to_base64(fig)))

    # ------------------------------------------------------------------
    # 4. Communication analysis (information flow + semantic)
    # ------------------------------------------------------------------
    comm_lines: list[str] = []

    if not events_df.empty:
        try:
            tracer = InformationTracer(events_df)
            prop_stats = tracer.get_propagation_stats()
            bottleneck_df = tracer.compute_bottleneck_nodes()

            comm_lines.append("Information Flow Statistics:")
            comm_lines.append(f"  Average Path Length: {prop_stats.get('avg_path_length', 0):.2f}")
            comm_lines.append(f"  Diameter: {prop_stats.get('diameter', 0)}")
            comm_lines.append(
                f"  Average Clustering Coefficient: {prop_stats.get('avg_clustering', 0):.4f}"
            )
            comm_lines.append("")

            if not bottleneck_df.empty:
                comm_lines.append("Top Bottleneck Agents (by betweenness centrality):")
                for _, row in bottleneck_df.head(5).iterrows():
                    comm_lines.append(f"  {row['agent_id']}: {row['betweenness_centrality']:.4f}")
            else:
                comm_lines.append("No bottleneck data available.")
        except Exception:
            comm_lines.append("Information flow analysis could not be computed.")

        try:
            analyzer = SemanticAnalyzer(events_df)
            discussion = analyzer.get_discussion_summary()
            comm_lines.append("")
            comm_lines.append("Semantic Communication Summary:")
            comm_lines.append(f"  Total Messages: {discussion.get('total_messages', 0)}")
            comm_lines.append(f"  Unique Senders: {discussion.get('unique_senders', 0)}")
            comm_lines.append(f"  Unique Receivers: {discussion.get('unique_receivers', 0)}")
            comm_lines.append(
                f"  Avg Content Length: {discussion.get('avg_content_length', 0):.1f} chars"
            )
            comm_lines.append(f"  Most Active Sender: {discussion.get('most_active_sender', 'N/A')}")

            msg_dist = discussion.get("msg_type_distribution", {})
            if msg_dist:
                comm_lines.append("  Message Type Breakdown:")
                for mtype, count in msg_dist.items():
                    comm_lines.append(f"    {mtype}: {count}")
        except Exception:
            comm_lines.append("Semantic analysis could not be computed.")
    else:
        comm_lines.append("No events available for communication analysis.")

    communication_section = (
        "<h2>Communication Analysis</h2>\n"
        f'<pre class="text-block">{chr(10).join(comm_lines)}</pre>'
    )

    # ------------------------------------------------------------------
    # 5. Assemble and write HTML
    # ------------------------------------------------------------------
    chart_sections_html = ""
    for title, img_b64 in chart_images:
        chart_sections_html += (
            f"<h2>{title}</h2>\n"
            f'<img src="data:image/png;base64,{img_b64}" alt="{title}">\n'
        )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "  <title>Simulation Report</title>\n"
        "  <style>\n"
        "    body {\n"
        "      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n"
        "      background: #fff;\n"
        "      color: #222;\n"
        "      margin: 0;\n"
        "      padding: 0;\n"
        "    }\n"
        "    .container {\n"
        "      max-width: 1200px;\n"
        "      margin: 0 auto;\n"
        "      padding: 24px;\n"
        "    }\n"
        "    h1 { text-align: center; margin-bottom: 8px; }\n"
        "    h2 { border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 32px; }\n"
        "    .overview-table { width: 100%; border-collapse: collapse; margin: 16px 0; }\n"
        "    .overview-table td, .overview-table th {\n"
        "      border: 1px solid #ddd; padding: 8px 12px; text-align: left;\n"
        "    }\n"
        "    .overview-table th { background: #f5f5f5; }\n"
        "    img { display: block; margin: 12px auto; max-width: 100%; }\n"
        "    .text-block {\n"
        "      background: #f9f9f9; border: 1px solid #eee; border-radius: 4px;\n"
        "      padding: 12px; font-size: 13px; line-height: 1.5; margin: 12px 0;\n"
        "      overflow-x: auto;\n"
        "    }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <div class="container">\n'
        "    <h1>Agent Collaboration Simulation Report</h1>\n"
        f"    {overview_section}\n"
        f"    {chart_sections_html}\n"
        f"    {communication_section}\n"
        "  </div>\n"
        "</body>\n"
        "</html>"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return os.path.abspath(output_path)
