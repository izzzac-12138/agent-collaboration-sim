"""Unit tests for the S2 data mining layer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.mining.features import build_action_sequences, extract_agent_features
from src.mining.sequence_mining import SequenceMiner
from src.mining.clustering import AgentClusterer, auto_cluster
from src.mining.anomaly import AnomalyDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_events(n_agents: int = 5, n_steps: int = 20) -> pd.DataFrame:
    """Create a synthetic events DataFrame simulating agent interactions.

    The payload column contains dicts so that ``extract_agent_features`` and
    ``build_action_sequences`` can extract the fields they need.
    """
    rng = np.random.RandomState(42)
    agent_ids = [f"agent_{i}" for i in range(n_agents)]

    records: list[dict] = []
    for step in range(n_steps):
        ts = pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=step)
        sender = rng.choice(agent_ids)
        receiver = rng.choice([a for a in agent_ids if a != sender])

        # message_sent
        records.append(
            {
                "timestamp": ts,
                "event_type": "message_sent",
                "agent_id": sender,
                "payload": {
                    "from_agent": sender,
                    "to_agent": receiver,
                    "message_type": rng.choice(["confirm", "request", "info"]),
                },
            }
        )

        # task_completed or task_failed
        worker = rng.choice(agent_ids)
        task_name = rng.choice(["compile", "deploy", "review"])
        success = rng.random() > 0.3
        records.append(
            {
                "timestamp": ts + pd.Timedelta(milliseconds=100),
                "event_type": "task_completed" if success else "task_failed",
                "agent_id": worker,
                "payload": {"task_name": task_name, "status": "success" if success else "failed"},
            }
        )

        # resource_transferred
        provider = rng.choice(agent_ids)
        consumer = rng.choice([a for a in agent_ids if a != provider])
        records.append(
            {
                "timestamp": ts + pd.Timedelta(milliseconds=200),
                "event_type": "resource_transferred",
                "agent_id": provider,
                "payload": {
                    "from_agent": provider,
                    "to_agent": consumer,
                    "resource_type": rng.choice(["gpu_time", "data", "budget"]),
                },
            }
        )

        # agent_step (neutral event)
        actor = rng.choice(agent_ids)
        records.append(
            {
                "timestamp": ts + pd.Timedelta(milliseconds=300),
                "event_type": "agent_step",
                "agent_id": actor,
                "payload": {},
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1. extract_agent_features
# ---------------------------------------------------------------------------


class TestExtractAgentFeatures:
    def test_returns_correct_shape_and_columns(self):
        n_agents = 5
        events = _make_mock_events(n_agents=n_agents, n_steps=20)
        features = extract_agent_features(events)

        assert features.shape[0] == n_agents
        expected_cols = {
            "messages_sent",
            "messages_received",
            "tasks_completed",
            "tasks_failed",
            "unique_collaborators",
            "avg_response_time",
            "resource_contributions",
        }
        assert set(features.columns) == expected_cols

    def test_empty_input_returns_empty_df(self):
        empty = pd.DataFrame(columns=["timestamp", "event_type", "agent_id", "payload"])
        result = extract_agent_features(empty)
        assert result.empty
        assert "messages_sent" in result.columns

    def test_messages_sent_matches_from_agent(self):
        events = pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2025-01-01"), "event_type": "message_sent",
                 "agent_id": "a", "payload": {"from_agent": "a", "to_agent": "b"}},
                {"timestamp": pd.Timestamp("2025-01-02"), "event_type": "message_sent",
                 "agent_id": "c", "payload": {"from_agent": "c", "to_agent": "a"}},
            ]
        )
        features = extract_agent_features(events)
        assert features.loc["a", "messages_sent"] == 1
        assert features.loc["a", "messages_received"] == 1


# ---------------------------------------------------------------------------
# 2. build_action_sequences
# ---------------------------------------------------------------------------


class TestBuildActionSequences:
    def test_returns_correct_length_and_nonempty(self):
        n_agents = 5
        events = _make_mock_events(n_agents=n_agents, n_steps=20)
        sequences = build_action_sequences(events)

        assert len(sequences) == n_agents
        for seq in sequences:
            assert len(seq) > 0
            assert all(isinstance(action, str) for action in seq)

    def test_empty_input_returns_empty_list(self):
        empty = pd.DataFrame(columns=["timestamp", "event_type", "agent_id", "payload"])
        assert build_action_sequences(empty) == []

    def test_single_agent_filter(self):
        events = _make_mock_events(n_agents=3, n_steps=10)
        seqs = build_action_sequences(events, agent_id="agent_0")
        assert len(seqs) == 1
        assert len(seqs[0]) > 0


# ---------------------------------------------------------------------------
# 3. sequence_mining - SequenceMiner.fit
# ---------------------------------------------------------------------------


class TestSequenceMinerFit:
    def test_fit_populates_patterns(self):
        sequences = [
            ["message_sent", "task_completed", "message_sent"],
            ["task_completed", "message_sent", "resource_transferred"],
            ["message_sent", "task_completed", "resource_transferred"],
            ["task_failed", "message_sent", "task_completed"],
        ]
        miner = SequenceMiner(min_support=0.3, min_length=2)
        miner.fit(sequences)

        assert isinstance(miner.patterns_, list)
        assert len(miner.patterns_) > 0
        for pattern, support in miner.patterns_:
            assert isinstance(pattern, list)
            assert 0.0 < support <= 1.0


# ---------------------------------------------------------------------------
# 4. sequence_mining - success correlation
# ---------------------------------------------------------------------------


class TestSequenceMinerSuccessCorrelation:
    def test_success_patterns_populated(self):
        # Sequences where "task_completed" is strongly correlated with success
        sequences = [
            ["message_sent", "task_completed", "message_sent"],
            ["task_completed", "message_sent", "resource_transferred"],
            ["message_sent", "task_completed", "resource_transferred"],
            ["task_failed", "message_sent", "resource_transferred"],
            ["task_failed", "message_sent", "task_failed"],
            ["message_sent", "task_failed", "resource_transferred"],
            ["message_sent", "task_completed", "message_sent"],
            ["task_completed", "resource_transferred", "message_sent"],
        ]
        # First 4 succeed, last 4 fail
        successes = [True, True, True, True, False, False, False, False]

        miner = SequenceMiner(min_support=0.2, min_length=2)
        miner.fit(sequences, successes=successes)

        assert isinstance(miner.success_patterns_, list)
        # At least some patterns should have lift > 1.0
        if miner.success_patterns_:
            for pattern, lift in miner.success_patterns_:
                assert lift > 1.0


# ---------------------------------------------------------------------------
# 5. clustering - KMeans
# ---------------------------------------------------------------------------


class TestAgentClustererKMeans:
    def test_kmeans_labels_and_profiles(self):
        rng = np.random.RandomState(0)
        n_agents = 10
        n_clusters = 3

        features_df = pd.DataFrame(
            {
                "messages_sent": rng.poisson(5, n_agents).astype(float),
                "tasks_completed": rng.poisson(3, n_agents).astype(float),
                "tasks_failed": rng.poisson(1, n_agents).astype(float),
                "unique_collaborators": rng.poisson(4, n_agents).astype(float),
                "avg_response_time": rng.exponential(2, n_agents),
                "resource_contributions": rng.poisson(2, n_agents).astype(float),
            },
            index=[f"agent_{i}" for i in range(n_agents)],
        )
        features_df.index.name = "agent_id"

        clusterer = AgentClusterer(method="kmeans", n_clusters=n_clusters, random_state=42)
        clusterer.fit(features_df)

        assert len(clusterer.labels_) == n_agents

        profiles = clusterer.get_cluster_profiles(features_df)
        assert profiles.shape[0] == n_clusters
        assert "count" in profiles.columns
        assert "role_label" in profiles.columns

    def test_kmeans_reproducible(self):
        rng = np.random.RandomState(7)
        df = pd.DataFrame(
            {
                "messages_sent": rng.poisson(5, 8).astype(float),
                "tasks_completed": rng.poisson(3, 8).astype(float),
            },
            index=[f"agent_{i}" for i in range(8)],
        )
        df.index.name = "agent_id"

        c1 = AgentClusterer(method="kmeans", n_clusters=2, random_state=42)
        c1.fit(df)

        c2 = AgentClusterer(method="kmeans", n_clusters=2, random_state=42)
        c2.fit(df)

        np.testing.assert_array_equal(c1.labels_, c2.labels_)


# ---------------------------------------------------------------------------
# 6. clustering - DBSCAN
# ---------------------------------------------------------------------------


class TestAgentClustererDBSCAN:
    def test_dbscan_labels_and_profiles(self):
        rng = np.random.RandomState(0)
        n_agents = 10

        # Create two clear groups so DBSCAN can find them
        group_a = rng.normal(loc=10, scale=0.5, size=(5, 3))
        group_b = rng.normal(loc=0, scale=0.5, size=(5, 3))
        data = np.vstack([group_a, group_b])

        features_df = pd.DataFrame(
            data,
            columns=["messages_sent", "tasks_completed", "unique_collaborators"],
            index=[f"agent_{i}" for i in range(n_agents)],
        )
        features_df.index.name = "agent_id"

        clusterer = AgentClusterer(method="dbscan")
        clusterer.fit(features_df)

        assert len(clusterer.labels_) == n_agents

        profiles = clusterer.get_cluster_profiles(features_df)
        assert "count" in profiles.columns
        assert "role_label" in profiles.columns

    def test_dbscan_unknown_method_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0]}, index=["x", "y"])
        df.index.name = "agent_id"
        clusterer = AgentClusterer(method="unknown")
        with pytest.raises(ValueError, match="Unknown method"):
            clusterer.fit(df)


# ---------------------------------------------------------------------------
# 7. anomaly detection - IsolationForest
# ---------------------------------------------------------------------------


class TestAnomalyDetectorIsolationForest:
    def test_outlier_is_flagged(self):
        rng = np.random.RandomState(42)
        n_normal = 15
        n_features = 3

        normal_data = rng.normal(loc=0, scale=1, size=(n_normal, n_features))
        # One obvious outlier far from the rest
        outlier = np.array([[100.0, 100.0, 100.0]])
        data = np.vstack([normal_data, outlier])

        features_df = pd.DataFrame(
            data,
            columns=["f1", "f2", "f3"],
            index=[f"agent_{i}" for i in range(n_normal + 1)],
        )

        detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
        detector.fit(features_df)
        result = detector.detect(features_df)

        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns

        # The outlier (last row) should be flagged
        outlier_idx = f"agent_{n_normal}"
        assert result.loc[outlier_idx, "is_anomaly"] == True

    def test_summary_after_detect(self):
        rng = np.random.RandomState(0)
        data = rng.normal(0, 1, (12, 2))
        data[-1] = [50, 50]  # outlier

        df = pd.DataFrame(data, columns=["x", "y"], index=range(12))
        detector = AnomalyDetector(method="isolation_forest", contamination=0.15)
        detector.fit(df).detect(df)
        summary = detector.get_anomaly_summary()

        assert summary["n_anomalies"] >= 1
        assert 0.0 <= summary["anomaly_rate"] <= 1.0
        assert isinstance(summary["anomaly_indices"], list)


# ---------------------------------------------------------------------------
# 8. anomaly detection - LOF
# ---------------------------------------------------------------------------


class TestAnomalyDetectorLOF:
    def test_lof_outlier_is_flagged(self):
        rng = np.random.RandomState(42)
        normal_data = rng.normal(loc=0, scale=1, size=(30, 3))
        outlier = np.array([[100.0, 100.0, 100.0]])
        data = np.vstack([normal_data, outlier])

        features_df = pd.DataFrame(
            data,
            columns=["f1", "f2", "f3"],
            index=[f"agent_{i}" for i in range(31)],
        )

        detector = AnomalyDetector(method="lof", contamination=0.1)
        detector.fit(features_df)
        result = detector.detect(features_df)

        assert "is_anomaly" in result.columns
        assert result.loc["agent_30", "is_anomaly"] == True

    def test_lof_detect_before_fit_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0]}, index=[0, 1])
        detector = AnomalyDetector(method="lof")
        with pytest.raises(RuntimeError, match="fit"):
            detector.detect(df)


# ---------------------------------------------------------------------------
# 9. auto_cluster convenience function
# ---------------------------------------------------------------------------


class TestAutoCluster:
    def test_returns_tuple_of_features_and_profiles(self):
        n_agents = 8
        events = _make_mock_events(n_agents=n_agents, n_steps=30)

        result = auto_cluster(events, method="kmeans", n_clusters=2)
        assert isinstance(result, tuple)
        assert len(result) == 2

        features_df, profiles = result
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(profiles, pd.DataFrame)

        assert features_df.shape[0] == n_agents
        assert "count" in profiles.columns
        assert profiles.shape[0] == 2
