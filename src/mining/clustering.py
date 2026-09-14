"""Agent role clustering using behavioral features."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class AgentClusterer:
    """Cluster agents into role groups based on behavioral feature vectors.

    Supports KMeans and DBSCAN clustering with automatic role labeling
    based on feature profiles.
    """

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> None:
        """Initialize the clusterer.

        Args:
            method: Clustering algorithm, either "kmeans" or "dbscan".
            n_clusters: Number of clusters for KMeans (ignored for DBSCAN).
            random_state: Random seed for reproducibility.
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_: Optional[KMeans | DBSCAN] = None
        self.scaler_ = StandardScaler()
        self.labels_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list[str]] = None

    def fit(self, features_df: pd.DataFrame) -> "AgentClusterer":
        """Fit the clustering model on agent behavioral features.

        Args:
            features_df: DataFrame with agent_id as index and numeric columns
                as features.

        Returns:
            self, for method chaining.
        """
        self.feature_names_ = list(features_df.columns)
        scaled = self.scaler_.fit_transform(features_df.values)

        if self.method == "kmeans":
            self.model_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        elif self.method == "dbscan":
            self.model_ = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'kmeans' or 'dbscan'.")

        self.model_.fit(scaled)
        self.labels_ = self.model_.labels_
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new agent features.

        Args:
            features_df: DataFrame with the same columns as the training data.

        Returns:
            Array of cluster labels.
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        scaled = self.scaler_.transform(features_df.values)
        return self.model_.predict(scaled)

    def get_cluster_profiles(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-cluster feature means and auto-generate role labels.

        Args:
            features_df: DataFrame with agent_id as index, same columns as
                training data.

        Returns:
            DataFrame indexed by cluster number with mean feature values,
            cluster size (count), and a role_label column.
        """
        if self.labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        df = features_df.copy()
        df["cluster"] = self.labels_

        profiles = df.groupby("cluster").mean()
        profiles["count"] = df.groupby("cluster").size()

        profiles["role_label"] = profiles.apply(
            self._assign_role_label, axis=1, columns=features_df.columns
        )
        return profiles

    def evaluate(self, features_df: pd.DataFrame) -> dict:
        """Evaluate clustering quality.

        Args:
            features_df: DataFrame with agent_id as index, same columns as
                training data.

        Returns:
            Dict with keys: silhouette_score, n_clusters, cluster_sizes.
        """
        if self.labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        scaled = self.scaler_.transform(features_df.values)

        result: dict = {
            "silhouette_score": None,
            "n_clusters": len(set(self.labels_) - {-1}),
            "cluster_sizes": {},
        }

        unique, counts = np.unique(self.labels_, return_counts=True)
        for label, count in zip(unique, counts):
            result["cluster_sizes"][int(label)] = int(count)

        # Silhouette score requires at least 2 clusters and no noise-only labels
        non_noise = self.labels_[self.labels_ >= 0]
        if len(set(non_noise)) >= 2:
            non_noise_scaled = scaled[self.labels_ >= 0]
            result["silhouette_score"] = float(
                silhouette_score(non_noise_scaled, non_noise)
            )

        return result

    @staticmethod
    def _assign_role_label(row: pd.Series, columns: list[str]) -> str:
        """Assign a human-readable role label based on feature magnitudes.

        Decision logic (checked in order):
        - High messages_sent + high tasks_completed -> "leader"
        - High resource_contributions -> "provider"
        - High unique_collaborators -> "bridge"
        - Low everything -> "peripheral"
        - Default -> "worker"
        """
        # Use column presence; fall back to 0 for missing columns
        msg = row.get("messages_sent", 0)
        tasks = row.get("tasks_completed", 0)
        resources = row.get("resource_contributions", 0)
        collabs = row.get("unique_collaborators", 0)

        high_msg = msg > 0.5
        high_tasks = tasks > 0.5
        high_resources = resources > 0.5
        high_collabs = collabs > 0.5

        low_everything = not (high_msg or high_tasks or high_resources or high_collabs)

        if high_msg and high_tasks:
            return "leader"
        if high_resources:
            return "provider"
        if high_collabs:
            return "bridge"
        if low_everything:
            return "peripheral"
        return "worker"


def auto_cluster(
    events_df: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function: extract features, cluster, return results.

    Args:
        events_df: Raw event log with columns including at least:
            agent_id, event_type, and optionally task/resource/collaboration
            data.
        method: Clustering algorithm ("kmeans" or "dbscan").
        n_clusters: Number of clusters for KMeans.

    Returns:
        Tuple of (features_df, cluster_profiles).
    """
    features_df = _extract_features(events_df)
    clusterer = AgentClusterer(method=method, n_clusters=n_clusters)
    clusterer.fit(features_df)
    profiles = clusterer.get_cluster_profiles(features_df)
    return features_df, profiles


def _extract_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extract behavioral feature vectors from raw events.

    Produces one row per agent with columns:
        messages_sent, tasks_completed, resource_contributions,
        unique_collaborators, total_events.
    """
    grouped = events_df.groupby("agent_id")

    features = pd.DataFrame(index=events_df["agent_id"].unique())
    features.index.name = "agent_id"

    features["messages_sent"] = grouped.apply(
        lambda g: (g.get("event_type", pd.Series()) == "message").sum()
    )
    features["tasks_completed"] = grouped.apply(
        lambda g: (g.get("event_type", pd.Series()) == "task_complete").sum()
    )
    features["resource_contributions"] = grouped.apply(
        lambda g: (g.get("event_type", pd.Series()) == "resource_share").sum()
    )
    features["unique_collaborators"] = grouped.apply(
        lambda g: g["target_agent"].nunique() if "target_agent" in g.columns else 0
    )
    features["total_events"] = grouped.size()

    features = features.fillna(0).astype(float)
    return features
