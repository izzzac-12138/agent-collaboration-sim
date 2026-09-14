"""Unit tests for the S3 information analysis layer."""

import sys
import os

import numpy as np
import pandas as pd
import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.information_flow import InformationTracer
from src.analysis.causal import CausalAnalyzer
from src.analysis.semantic import SemanticAnalyzer


def _make_message_events(n_agents: int = 8, n_steps: int = 30) -> pd.DataFrame:
    """Create synthetic message and task events for testing."""
    rng = np.random.RandomState(42)
    records = []

    agents = [f"agent_{i}" for i in range(n_agents)]

    for step in range(n_steps):
        # Message events
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
                    "content": f"message from {frm} to {to}",
                    "msg_type": "inform",
                },
            })

        # Step summary
        records.append({
            "timestamp": step,
            "event_type": "step_summary",
            "agent_id": "",
            "payload": {
                "step": step,
                "n_edges": rng.randint(5, 20),
                "tasks_completed": rng.randint(0, 3),
                "network_density": rng.uniform(0.1, 0.5),
            },
        })

        # Task completion
        if rng.random() < 0.3:
            records.append({
                "timestamp": step,
                "event_type": "task_completed",
                "agent_id": rng.choice(agents),
                "payload": {"task_id": f"T{step:03d}"},
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# InformationTracer tests
# ---------------------------------------------------------------------------

class TestInformationTracer:
    def test_trace_reach(self):
        events_df = _make_message_events()
        tracer = InformationTracer(events_df)
        result = tracer.trace_information("agent_0")
        assert result["source"] == "agent_0"
        assert result["total_reached"] >= 0
        assert 0 <= result["coverage_rate"] <= 1

    def test_bottleneck_computation(self):
        events_df = _make_message_events()
        tracer = InformationTracer(events_df)
        df = tracer.compute_bottleneck_nodes()
        assert isinstance(df, pd.DataFrame)
        assert "agent_id" in df.columns
        assert "betweenness_centrality" in df.columns

    def test_propagation_stats(self):
        events_df = _make_message_events()
        tracer = InformationTracer(events_df)
        stats = tracer.get_propagation_stats()
        assert "avg_clustering" in stats
        assert stats["avg_clustering"] >= 0


# ---------------------------------------------------------------------------
# CausalAnalyzer tests
# ---------------------------------------------------------------------------

class TestCausalAnalyzer:
    def test_granger_causality_correlated(self):
        rng = np.random.RandomState(42)
        n = 50
        x = pd.Series(rng.randn(n))
        y = pd.Series(np.roll(x.values, 1) * 0.8 + rng.randn(n) * 0.1)

        events_df = _make_message_events()
        analyzer = CausalAnalyzer(events_df)
        result = analyzer.compute_granger_causality(x, y, max_lag=2)

        assert "p_value" in result
        assert "is_causal" in result
        assert isinstance(result["is_causal"], (bool, np.bool_))

    def test_granger_no_causality(self):
        rng = np.random.RandomState(42)
        x = pd.Series(rng.randn(50))
        y = pd.Series(rng.randn(50))

        events_df = _make_message_events()
        analyzer = CausalAnalyzer(events_df)
        result = analyzer.compute_granger_causality(x, y, max_lag=2)

        assert result["p_value"] > 0.01  # should NOT be causal

    def test_network_impact(self):
        events_df = _make_message_events()
        analyzer = CausalAnalyzer(events_df)
        result = analyzer.analyze_network_structure_impact()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SemanticAnalyzer tests
# ---------------------------------------------------------------------------

class TestSemanticAnalyzer:
    def test_topics(self):
        events_df = _make_message_events()
        analyzer = SemanticAnalyzer(events_df)
        result = analyzer.extract_topics_lda(n_topics=2)
        assert "topics" in result
        assert "dominant_topics_per_agent" in result

    def test_embeddings(self):
        events_df = _make_message_events()
        analyzer = SemanticAnalyzer(events_df)
        embeddings = analyzer.compute_message_embeddings()
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2

    def test_discussion_summary(self):
        events_df = _make_message_events()
        analyzer = SemanticAnalyzer(events_df)
        summary = analyzer.get_discussion_summary()
        assert "total_messages" in summary
        assert summary["total_messages"] > 0
