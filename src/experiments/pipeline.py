"""End-to-end pipeline: run simulation -> mine patterns -> analyze -> report."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.collaboration.runner import SimulationRunner
from src.scenarios.disaster_relief import DisasterReliefScenario

# --- Optional imports with graceful fallback ---
try:
    from src.mining.features import extract_agent_features, build_action_sequences
except ImportError:
    extract_agent_features = None  # type: ignore[assignment]
    build_action_sequences = None  # type: ignore[assignment]

try:
    from src.mining.clustering import auto_cluster
except ImportError:
    auto_cluster = None  # type: ignore[assignment]

try:
    from src.mining.anomaly import AnomalyDetector
except ImportError:
    AnomalyDetector = None  # type: ignore[assignment]

try:
    from src.analysis.log_reader import load_events
except ImportError:
    load_events = None  # type: ignore[assignment]

try:
    from src.analysis.information_flow import InformationTracer
except ImportError:
    InformationTracer = None  # type: ignore[assignment]

try:
    from src.analysis.semantic import SemanticAnalyzer
except ImportError:
    SemanticAnalyzer = None  # type: ignore[assignment]


class ExperimentPipeline:
    """Orchestrate a full experiment: simulation, log analysis, mining, and reporting.

    Parameters
    ----------
    scenario:
        A scenario instance.  Defaults to ``DisasterReliefScenario()``.
    n_agents:
        Number of agents in each simulation run.
    max_steps:
        Maximum ticks per run.
    output_dir:
        Root directory for run artefacts.
    """

    def __init__(
        self,
        scenario: Optional[object] = None,
        n_agents: int = 10,
        max_steps: int = 100,
        output_dir: str = "data/runs",
    ) -> None:
        self.scenario = scenario or DisasterReliefScenario()
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.run_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run_single(self) -> Dict[str, Any]:
        """Execute one simulation and perform full post-hoc analysis.

        Steps:

        1. Run the simulation via ``SimulationRunner``.
        2. Load events from the generated log.
        3. Cluster agents by behaviour (``auto_cluster``).
        4. Detect temporal anomalies (``AnomalyDetector``).
        5. Trace information flow from the first agent.
        6. Run semantic analysis on messages.
        7. Compile and return all results.

        Returns
        -------
        dict
            Keys: ``run_id``, ``summary``, ``clustering``, ``anomaly``,
            ``information_flow``, ``semantic``, ``duration``.
        """
        t0 = time.perf_counter()

        # -- 1. Run simulation --
        runner = SimulationRunner(
            scenario=self.scenario,
            n_agents=self.n_agents,
            output_dir=self.output_dir,
        )
        summary = runner.run(max_steps=self.max_steps)
        run_id: str = summary.get("run_id", "unknown")

        results: Dict[str, Any] = {
            "run_id": run_id,
            "summary": summary,
            "clustering": None,
            "anomaly": None,
            "information_flow": None,
            "semantic": None,
        }

        # -- 2. Load events --
        events_df: Optional[pd.DataFrame] = None
        if load_events is not None:
            events_path = os.path.join(self.output_dir, run_id, "events.jsonl")
            if os.path.isfile(events_path):
                try:
                    events_df = load_events(events_path)
                except Exception as exc:
                    print(f"[pipeline] Failed to load events: {exc}")
            else:
                # Try the summary JSON for an event log path hint.
                summary_path = os.path.join(self.output_dir, run_id, "summary.json")
                if os.path.isfile(summary_path):
                    print(f"[pipeline] events.jsonl not found at {events_path}, "
                          f"skipping log-based analysis.")
        else:
            print("[pipeline] log_reader not available, skipping event-based analysis.")

        if events_df is None or events_df.empty:
            print("[pipeline] No events loaded – skipping mining/analysis stages.")
            results["duration"] = round(time.perf_counter() - t0, 4)
            self.run_results[run_id] = results
            return results

        # -- 3. Clustering --
        if auto_cluster is not None:
            try:
                features_df, cluster_profiles = auto_cluster(events_df)
                results["clustering"] = {
                    "n_clusters": int(cluster_profiles.shape[0]) if cluster_profiles is not None else 0,
                    "profiles": cluster_profiles.to_dict(orient="records") if cluster_profiles is not None else [],
                }
            except Exception as exc:
                print(f"[pipeline] Clustering failed: {exc}")

        # -- 4. Anomaly detection (temporal) --
        if AnomalyDetector is not None:
            try:
                detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
                temporal_anomalies = detector.detect_temporal_anomalies(events_df, window_size=10)
                summary_info = detector.get_anomaly_summary()
                results["anomaly"] = {
                    "n_anomalies": summary_info.get("n_anomalies", 0),
                    "anomaly_rate": summary_info.get("anomaly_rate", 0.0),
                    "details": temporal_anomalies.to_dict(orient="records")
                    if temporal_anomalies is not None and not temporal_anomalies.empty
                    else [],
                }
            except Exception as exc:
                print(f"[pipeline] Anomaly detection failed: {exc}")

        # -- 5. Information flow tracing --
        if InformationTracer is not None and events_df is not None:
            try:
                tracer = InformationTracer(events_df)
                # Identify the first agent present in the events.
                first_agent = (
                    events_df["agent_id"].dropna().iloc[0]
                    if "agent_id" in events_df.columns and not events_df["agent_id"].dropna().empty
                    else None
                )
                if first_agent is not None:
                    flow_result = tracer.trace_information(source_agent=str(first_agent))
                    bottleneck_df = tracer.compute_bottleneck_nodes()
                    results["information_flow"] = {
                        "source": flow_result.get("source"),
                        "coverage_rate": flow_result.get("coverage_rate", 0.0),
                        "total_reached": flow_result.get("total_reached", 0),
                        "avg_hops": flow_result.get("avg_hops", 0.0),
                        "max_hops": flow_result.get("max_hops", 0),
                        "propagation_steps": len(flow_result.get("propagation_timeline", [])),
                        "bottlenecks": bottleneck_df.head(5).to_dict(orient="records")
                        if bottleneck_df is not None and not bottleneck_df.empty
                        else [],
                    }
            except Exception as exc:
                print(f"[pipeline] Information tracing failed: {exc}")

        # -- 6. Semantic analysis --
        if SemanticAnalyzer is not None and events_df is not None:
            try:
                analyzer = SemanticAnalyzer(events_df)
                discussion = analyzer.get_discussion_summary()
                results["semantic"] = {
                    "total_messages": discussion.get("total_messages", 0),
                    "unique_senders": discussion.get("unique_senders", 0),
                    "avg_content_length": discussion.get("avg_content_length", 0.0),
                    "msg_type_distribution": discussion.get("msg_type_distribution", {}),
                    "most_active_sender": discussion.get("most_active_sender"),
                }
            except Exception as exc:
                print(f"[pipeline] Semantic analysis failed: {exc}")

        # -- Finalize --
        results["duration"] = round(time.perf_counter() - t0, 4)
        self.run_results[run_id] = results
        print(f"[pipeline] Run {run_id} complete in {results['duration']:.2f}s")
        return results

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_batch(self, n_runs: int = 5) -> pd.DataFrame:
        """Execute ``run_single`` multiple times and compare results.

        Parameters
        ----------
        n_runs:
            Number of independent runs.

        Returns
        -------
        pd.DataFrame
            One row per run with columns: ``run_id``, ``steps_completed``,
            ``tasks_completed``, ``messages_sent``, ``duration``,
            ``n_clusters``, ``n_anomalies``, ``info_coverage``,
            ``total_messages``.
        """
        rows: List[Dict[str, Any]] = []
        for i in range(n_runs):
            print(f"\n[pipeline] === Batch run {i + 1}/{n_runs} ===")
            results = self.run_single()
            summary = results.get("summary", {})
            rows.append({
                "run_id": results.get("run_id"),
                "steps_completed": summary.get("steps_completed", 0),
                "tasks_completed": summary.get("tasks_completed", 0),
                "messages_sent": summary.get("messages_sent", 0),
                "duration": results.get("duration", 0.0),
                "n_clusters": (
                    results["clustering"].get("n_clusters", 0)
                    if results.get("clustering") is not None
                    else 0
                ),
                "n_anomalies": (
                    results["anomaly"].get("n_anomalies", 0)
                    if results.get("anomaly") is not None
                    else 0
                ),
                "info_coverage": (
                    results["information_flow"].get("coverage_rate", 0.0)
                    if results.get("information_flow") is not None
                    else 0.0
                ),
                "total_messages": (
                    results["semantic"].get("total_messages", 0)
                    if results.get("semantic") is not None
                    else 0
                ),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, run_dir: str) -> None:
        """Generate a summary report for the collected runs.

        Attempts to use the dashboard report generator if available,
        otherwise prints a plain-text summary to stdout.

        Parameters
        ----------
        run_dir:
            Directory containing run artefacts (used by the dashboard module).
        """
        # Try the dashboard report generator first.
        try:
            from src.dashboard import report_generator  # type: ignore[import-not-found]
            if hasattr(report_generator, "generate_report"):
                report_generator.generate_report(run_dir)
                print(f"[pipeline] Dashboard report generated for {run_dir}")
                return
        except (ImportError, AttributeError, Exception) as exc:
            print(f"[pipeline] Dashboard report unavailable ({exc}), falling back to text summary.")

        # Fallback: plain-text summary.
        self._print_summary()

    def _print_summary(self) -> None:
        """Print a human-readable summary of all collected run results."""
        if not self.run_results:
            print("[pipeline] No runs to summarize.")
            return

        print("\n" + "=" * 60)
        print(f"  Experiment Pipeline Summary  ({len(self.run_results)} runs)")
        print("=" * 60)

        for run_id, res in self.run_results.items():
            s = res.get("summary", {})
            print(f"\n--- {run_id} ---")
            print(f"  Steps completed : {s.get('steps_completed', 'N/A')}")
            print(f"  Tasks completed : {s.get('tasks_completed', 'N/A')}")
            print(f"  Messages sent   : {s.get('messages_sent', 'N/A')}")
            print(f"  Duration        : {res.get('duration', 'N/A')}s")

            clust = res.get("clustering")
            if clust:
                print(f"  Clusters found  : {clust.get('n_clusters', 0)}")

            anom = res.get("anomaly")
            if anom:
                print(f"  Anomalies       : {anom.get('n_anomalies', 0)} "
                      f"(rate={anom.get('anomaly_rate', 0):.2%})")

            flow = res.get("information_flow")
            if flow:
                print(f"  Info coverage   : {flow.get('coverage_rate', 0):.1%} "
                      f"(reached {flow.get('total_reached', 0)} agents)")

            sem = res.get("semantic")
            if sem:
                print(f"  Messages (sem)  : {sem.get('total_messages', 0)} "
                      f"from {sem.get('unique_senders', 0)} senders")

        print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate results from all runs into overall statistics.

        Returns
        -------
        dict
            ``total_runs``, ``avg_steps``, ``avg_tasks_completed``,
            ``avg_messages_sent``, ``avg_duration``, ``total_anomalies``,
            ``avg_info_coverage``.
        """
        if not self.run_results:
            return {"total_runs": 0}

        steps_list: List[float] = []
        tasks_list: List[float] = []
        msgs_list: List[float] = []
        dur_list: List[float] = []
        anomaly_total = 0
        coverage_list: List[float] = []

        for res in self.run_results.values():
            s = res.get("summary", {})
            steps_list.append(s.get("steps_completed", 0))
            tasks_list.append(s.get("tasks_completed", 0))
            msgs_list.append(s.get("messages_sent", 0))
            dur_list.append(res.get("duration", 0.0))

            anom = res.get("anomaly")
            if anom:
                anomaly_total += anom.get("n_anomalies", 0)

            flow = res.get("information_flow")
            if flow:
                coverage_list.append(flow.get("coverage_rate", 0.0))

        def _avg(lst: List[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        return {
            "total_runs": len(self.run_results),
            "avg_steps": _avg(steps_list),
            "avg_tasks_completed": _avg(tasks_list),
            "avg_messages_sent": _avg(msgs_list),
            "avg_duration": _avg(dur_list),
            "total_anomalies": anomaly_total,
            "avg_info_coverage": _avg(coverage_list),
        }
