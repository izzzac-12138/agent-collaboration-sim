"""Anomaly detection for collaboration risk monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .features import extract_agent_features, extract_temporal_features


class AnomalyDetector:
    """Detect anomalous agents or time-windows in collaboration event streams.

    Parameters
    ----------
    method:
        ``"isolation_forest"`` (default) or ``"lof"`` (Local Outlier Factor).
    contamination:
        Expected proportion of outliers in the training data.
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
    ) -> None:
        self.method = method
        self.contamination = contamination
        self.model_ = None
        self.threshold_ = None
        self.anomalies_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> AnomalyDetector:
        """Fit the anomaly detection model on numeric features.

        Parameters
        ----------
        features_df:
            DataFrame of numeric columns (e.g. output of
            :func:`extract_temporal_features`).  Non-numeric columns are
            dropped before fitting.

        Returns
        -------
        AnomalyDetector
            ``self``, so callers can chain ``fit(...).detect(...)``.
        """
        X = features_df.select_dtypes(include=[np.number])

        if self.method == "isolation_forest":
            self.model_ = IsolationForest(
                contamination=self.contamination,
                random_state=42,
            )
            self.model_.fit(X)
        elif self.method == "lof":
            self.model_ = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
            )
            self.model_.fit(X)
        else:
            raise ValueError(
                f"Unknown method {self.method!r}. "
                "Choose 'isolation_forest' or 'lof'."
            )

        return self

    def detect(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies on *features_df* (must be compatible with fit data).

        The returned DataFrame contains the original columns plus:

        * ``is_anomaly`` (bool) -- ``True`` for detected outliers.
        * ``anomaly_score`` (float) -- the negative decision-function value
          (more negative = more anomalous).

        Parameters
        ----------
        features_df:
            Numeric feature DataFrame.  Non-numeric columns are preserved
            in the output but ignored during prediction.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before detect().")

        X = features_df.select_dtypes(include=[np.number])
        result = features_df.copy()

        labels = self.model_.predict(X)  # -1 = anomaly, 1 = normal
        result["is_anomaly"] = labels == -1

        # Negate so that *lower* values indicate stronger anomalies,
        # matching the convention of ``negative_outlier_factor_`` in LOF.
        result["anomaly_score"] = -self.model_.decision_function(X)

        self.anomalies_ = result
        return result

    def get_anomaly_summary(self) -> dict:
        """Return a summary dict of the last :meth:`detect` run.

        Keys
        ----
        n_anomalies : int
        anomaly_rate : float
        anomaly_indices : list[int]
        mean_score : float
        """
        if self.anomalies_ is None:
            raise RuntimeError("Call detect() before get_anomaly_summary().")

        mask = self.anomalies_["is_anomaly"]
        return {
            "n_anomalies": int(mask.sum()),
            "anomaly_rate": float(mask.mean()),
            "anomaly_indices": self.anomalies_.index[mask].tolist(),
            "mean_score": float(
                self.anomalies_.loc[mask, "anomaly_score"].mean()
            )
            if mask.any()
            else 0.0,
        }

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def detect_agent_anomalies(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous agents based on per-agent feature profiles.

        Parameters
        ----------
        events_df:
            Raw simulation event log (passed to
            :func:`features.extract_agent_features`).

        Returns
        -------
        pd.DataFrame
            Columns: ``agent_id``, each numeric feature produced by
            :func:`extract_agent_features`, ``is_anomaly``, ``anomaly_score``.
        """
        agent_features = extract_agent_features(events_df)
        if agent_features.empty:
            return agent_features

        agent_ids = agent_features.index.to_frame(name="agent_id")
        numeric = agent_features.select_dtypes(include=[np.number])

        self.fit(numeric)
        result = self.detect(numeric)

        result = result.reset_index()
        if "agent_id" not in result.columns:
            # ``extract_agent_features`` may use agent_id as the index
            result = pd.concat([agent_ids.reset_index(drop=True), result], axis=1)

        return result

    def detect_temporal_anomalies(
        self,
        events_df: pd.DataFrame,
        window_size: int = 10,
    ) -> pd.DataFrame:
        """Detect anomalous time windows.

        Parameters
        ----------
        events_df:
            Raw simulation event log.
        window_size:
            Number of events per sliding window.

        Returns
        -------
        pd.DataFrame
            Columns: ``window_id``, each numeric feature produced by
            :func:`extract_temporal_features`, ``is_anomaly``,
            ``anomaly_score``.
        """
        temporal_features = extract_temporal_features(events_df, window_size=window_size)
        if temporal_features.empty:
            return temporal_features

        window_ids = temporal_features.index.to_frame(name="window_id")
        numeric = temporal_features.select_dtypes(include=[np.number])

        self.fit(numeric)
        result = self.detect(numeric)

        result = result.reset_index()
        if "window_id" not in result.columns:
            result = pd.concat(
                [window_ids.reset_index(drop=True), result], axis=1
            )

        return result


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------


def quick_anomaly_check(
    events_df: pd.DataFrame,
    method: str = "isolation_forest",
) -> dict:
    """Detect temporal anomalies and return a compact summary.

    Parameters
    ----------
    events_df:
        Raw simulation event log.
    method:
        ``"isolation_forest"`` or ``"lof"``.

    Returns
    -------
    dict
        Keys: ``n_anomalies``, ``anomaly_rate``, ``anomaly_indices``,
        ``mean_score``, ``total_windows``.
    """
    detector = AnomalyDetector(method=method)
    detector.detect_temporal_anomalies(events_df)

    summary = detector.get_anomaly_summary()
    summary["total_windows"] = (
        len(detector.anomalies_) if detector.anomalies_ is not None else 0
    )
    return summary
