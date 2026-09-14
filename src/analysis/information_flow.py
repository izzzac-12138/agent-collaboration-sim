"""Information propagation path tracing through the agent network."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


class InformationTracer:
    """Trace how information flows through the agent communication network."""

    def __init__(self, events_df: pd.DataFrame) -> None:
        self.events_df = events_df
        self.network_snapshots: dict[int, nx.Graph] = {}
        self._build_snapshots()

    def _build_snapshots(self) -> None:
        msgs = self.events_df[self.events_df["event_type"] == "message_sent"]
        for _, row in msgs.iterrows():
            ts = int(row["timestamp"])
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            frm = payload.get("from_agent") or row.get("agent_id", "")
            to = payload.get("to_agent", "")
            if not frm or not to:
                continue
            if ts not in self.network_snapshots:
                self.network_snapshots[ts] = nx.Graph()
            g = self.network_snapshots[ts]
            g.add_node(frm)
            g.add_node(to)
            if g.has_edge(frm, to):
                g[frm][to]["weight"] = g[frm][to].get("weight", 0) + 1
            else:
                g.add_edge(frm, to, weight=1)

    def trace_information(
        self,
        source_agent: str,
        keyword: str | None = None,
        start_time: int = 0,
        end_time: int | None = None,
    ) -> dict:
        """BFS propagation from source_agent through the message network."""
        times = sorted(self.network_snapshots.keys())
        if end_time is None:
            end_time = times[-1] if times else 0

        reachable: dict[str, tuple[int, int, list]] = {}
        queue: list[tuple[str, int, int, list]] = [(source_agent, 0, 0, [])]
        visited = {source_agent}

        for ts in times:
            if ts < start_time or ts > end_time:
                continue
            g = self.network_snapshots[ts]
            next_queue: list[tuple[str, int, int, list]] = []
            for agent, hops, arrive_ts, path in queue:
                if agent in g:
                    for neighbor in g.neighbors(agent):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [agent]
                            reachable[neighbor] = (ts, hops + 1, new_path)
                            next_queue.append((neighbor, hops + 1, ts, new_path))
            queue = next_queue

        total_agents = set()
        for g in self.network_snapshots.values():
            total_agents.update(g.nodes())

        n_reached = len(reachable)
        n_total = max(len(total_agents), 1)
        hops_list = [v[1] for v in reachable.values()]

        return {
            "source": source_agent,
            "reached_agents": list(reachable.keys()),
            "total_reached": n_reached,
            "coverage_rate": n_reached / n_total,
            "avg_hops": float(np.mean(hops_list)) if hops_list else 0.0,
            "max_hops": max(hops_list) if hops_list else 0,
            "propagation_timeline": [
                (v[0], agent, v[1]) for agent, v in reachable.items()
            ],
            "paths": {a: v[2] for a, v in reachable.items()},
        }

    def compute_bottleneck_nodes(self) -> pd.DataFrame:
        """Compute betweenness centrality to find information bottleneck nodes."""
        combined = nx.Graph()
        for g in self.network_snapshots.values():
            for u, v in g.edges():
                if combined.has_edge(u, v):
                    combined[u][v]["weight"] += g[u][v].get("weight", 1)
                else:
                    combined.add_edge(u, v, weight=g[u][v].get("weight", 1))

        if combined.number_of_nodes() == 0:
            return pd.DataFrame(columns=["agent_id", "betweenness_centrality"])

        bc = nx.betweenness_centrality(combined, weight="weight")
        df = pd.DataFrame([
            {"agent_id": a, "betweenness_centrality": c}
            for a, c in bc.items()
        ]).sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)
        return df

    def compute_propagation_speed(self, source_agent: str) -> dict:
        """Compute information propagation speed metrics."""
        trace = self.trace_information(source_agent)
        timeline = trace["propagation_timeline"]
        if not timeline:
            return {"avg_speed": 0.0, "max_hops_per_step": 0.0, "total_time": 0}

        timestamps = [t[0] for t in timeline]
        hops = [t[2] for t in timeline]
        total_time = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
        avg_speed = sum(hops) / max(total_time, 1)

        return {
            "avg_speed": avg_speed,
            "max_hops_per_step": max(hops) / max(total_time, 1),
            "total_time": total_time,
            "agents_reached_per_step": len(timeline) / max(total_time, 1),
        }

    def compute_information_distortion(
        self, source_agent: str, keyword: str = ""
    ) -> dict:
        """Measure how much information degrades as it propagates.

        Compares the original message content from the source with messages
        forwarded by intermediate agents.  Distortion is measured as the
        fraction of keyword overlap lost at each hop.
        """
        msgs = self.events_df[self.events_df["event_type"] == "message_sent"]
        if msgs.empty:
            return {"avg_distortion": 0.0, "max_distortion": 0.0, "n_hops_analyzed": 0}

        source_keywords: set[str] = set()
        hop_distortions: list[float] = []

        for _, row in msgs.iterrows():
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            frm = payload.get("from_agent") or row.get("agent_id", "")
            content = payload.get("content", "")
            words = set(content.lower().split())

            if frm == source_agent:
                source_keywords = words
                continue

            if source_keywords and words:
                overlap = len(source_keywords & words) / max(len(source_keywords), 1)
                hop_distortions.append(1.0 - overlap)

        if not hop_distortions:
            return {"avg_distortion": 0.0, "max_distortion": 0.0, "n_hops_analyzed": 0}

        return {
            "avg_distortion": float(np.mean(hop_distortions)),
            "max_distortion": float(np.max(hop_distortions)),
            "n_hops_analyzed": len(hop_distortions),
        }

    def compute_information_value(
        self, source_agent: str, task_completion_events: pd.DataFrame
    ) -> dict:
        """Correlate information propagation with task success."""
        trace = self.trace_information(source_agent)
        informed = set(trace["reached_agents"])

        completed = task_completion_events[
            task_completion_events["event_type"] == "task_completed"
        ]
        if len(completed) == 0:
            return {
                "source": source_agent,
                "info_value_rate": 0.0,
                "agents_informed": len(informed),
                "tasks_completed": 0,
            }

        completed_agents = set(completed["agent_id"].unique())
        overlap = informed & completed_agents
        rate = len(overlap) / max(len(completed_agents), 1)

        return {
            "source": source_agent,
            "info_value_rate": rate,
            "agents_informed": len(informed),
            "tasks_completed": len(completed),
        }

    def get_propagation_stats(self) -> dict:
        """Overall network propagation statistics."""
        combined = nx.Graph()
        for g in self.network_snapshots.values():
            combined = nx.compose(combined, g)

        n_nodes = combined.number_of_nodes()
        if n_nodes == 0:
            return {"avg_path_length": 0, "diameter": 0, "avg_clustering": 0}

        avg_clustering = nx.average_clustering(combined)
        if nx.is_connected(combined):
            avg_path = nx.average_shortest_path_length(combined)
            diameter = nx.diameter(combined)
        else:
            components = list(nx.connected_components(combined))
            longest_path = 0
            for comp in components:
                sub = combined.subgraph(comp)
                if sub.number_of_nodes() > 1:
                    longest_path = max(longest_path, nx.diameter(sub))
            avg_path = longest_path
            diameter = longest_path

        return {
            "avg_path_length": avg_path,
            "diameter": diameter,
            "avg_clustering": avg_clustering,
        }
