"""Causal analysis: network structure vs efficiency correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

try:
    import shap
except ImportError:
    shap = None


class CausalAnalyzer:
    """Analyze causal relationships between network structure and task efficiency."""

    def __init__(
        self, events_df: pd.DataFrame, network_snapshots: dict | None = None
    ) -> None:
        self.events_df = events_df
        self.network_snapshots = network_snapshots or {}
        self.results_: dict = {}

    def compute_granger_causality(
        self, cause: pd.Series, effect: pd.Series, max_lag: int = 3
    ) -> dict:
        """Simple Granger causality F-test for cause -> effect."""
        best_f, best_p, best_lag = 0.0, 1.0, 1

        for lag in range(1, max_lag + 1):
            n = len(effect) - lag
            if n < lag + 2:
                continue

            y = effect.iloc[lag:].values
            y_lag = effect.iloc[lag - 1 : -1].values
            x_lag = cause.iloc[lag - 1 : -1].values

            X_restricted = np.column_stack([np.ones(n), y_lag])
            X_full = np.column_stack([np.ones(n), y_lag, x_lag])

            try:
                beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                beta_f = np.linalg.lstsq(X_full, y, rcond=None)[0]

                rss_r = np.sum((y - X_restricted @ beta_r) ** 2)
                rss_f = np.sum((y - X_full @ beta_f) ** 2)

                if rss_f == 0:
                    continue

                f_stat = ((rss_r - rss_f) / 1) / (rss_f / (n - 3))
                p_val = 1 - stats.f.cdf(f_stat, 1, n - 3)

                if p_val < best_p:
                    best_f, best_p, best_lag = f_stat, p_val, lag
            except np.linalg.LinAlgError:
                continue

        return {
            "best_lag": best_lag,
            "f_statistic": best_f,
            "p_value": best_p,
            "is_causal": best_p < 0.05,
        }

    def analyze_network_structure_impact(self) -> dict:
        """Granger test: do network metrics predict task completion rate?"""
        step_events = self.events_df[
            self.events_df["event_type"] == "step_summary"
        ].copy()

        if len(step_events) < 5:
            return {"error": "Not enough step_summary events (need >= 5)"}

        task_completions = []
        densities = []
        for _, row in step_events.iterrows():
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            task_completions.append(payload.get("tasks_completed", 0))
            densities.append(payload.get("network_density", 0))

        tc = pd.Series(task_completions, dtype=float)
        dn = pd.Series(densities, dtype=float)

        result = self.compute_granger_causality(dn, tc)
        self.results_["structure_impact"] = result
        return result

    def compute_shap_importance(
        self, features_df: pd.DataFrame, target: pd.Series
    ) -> dict:
        """Feature importance via SHAP values or correlation fallback."""
        if shap is not None:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(features_df, target)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)
            importance = {
                col: float(np.mean(np.abs(shap_values[:, i])))
                for i, col in enumerate(features_df.columns)
            }
        else:
            importance = {
                col: float(abs(features_df[col].corr(target)))
                for col in features_df.columns
                if features_df[col].std() > 0
            }

        self.results_["feature_importance"] = importance
        return importance

    def summarize(self) -> dict:
        """Return all analysis results."""
        return dict(self.results_)
