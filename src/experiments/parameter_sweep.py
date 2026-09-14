"""Parameter sweep framework for running experiments with varying parameters."""

from __future__ import annotations

import itertools
import json
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.collaboration.runner import SimulationRunner
from src.scenarios.disaster_relief import DisasterReliefScenario


class ParameterSweep:
    """Run a series of simulations over a grid of parameter values and aggregate results.

    Usage::

        sweep = ParameterSweep()
        sweep.define_sweep("n_agents", [5, 10, 20])
        sweep.define_sweep("max_steps", [50, 100])
        df = sweep.run(n_repeats=3)
        sweep.plot_results("n_agents")
    """

    def __init__(self, base_config: Optional[Dict[str, Any]] = None) -> None:
        self.base_config: Dict[str, Any] = base_config or {
            "n_agents": 10,
            "max_steps": 100,
        }
        self.results: List[Dict[str, Any]] = []

        # Internal sweep definitions: {param_name: [values, ...]}
        self._sweep_params: Dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # Sweep definition
    # ------------------------------------------------------------------

    def define_sweep(self, param_name: str, values: list) -> ParameterSweep:
        """Register a parameter to sweep over.

        Parameters
        ----------
        param_name:
            Name of the parameter to vary.  Supported names:

            * ``"n_agents"`` – number of agents per run
            * ``"max_steps"`` – maximum simulation steps
            * ``"n_tasks"`` – number of tasks (forwarded to the scenario)
            * ``"n_resource_points"`` – number of resource points
        values:
            List of concrete values to try for this parameter.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.
        """
        self._sweep_params[param_name] = list(values)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        n_repeats: int = 3,
        output_base: str = "data/experiments",
    ) -> pd.DataFrame:
        """Execute the full sweep.

        For every combination of swept parameters * n_repeats independent runs,
        a simulation is executed and its metrics are recorded.

        Parameters
        ----------
        n_repeats:
            Number of independent repetitions per parameter combination.
        output_base:
            Root directory for saving per-run output artefacts.

        Returns
        -------
        pd.DataFrame
            One row per individual run with columns for every swept parameter
            plus ``run_id``, ``steps``, ``tasks_completed``, ``messages_sent``,
            and ``duration``.
        """
        self.results.clear()

        # Build the Cartesian product of all swept parameters.
        if self._sweep_params:
            param_names = list(self._sweep_params.keys())
            param_values = [self._sweep_params[k] for k in param_names]
            combinations: List[Dict[str, Any]] = [
                dict(zip(param_names, combo))
                for combo in itertools.product(*param_values)
            ]
        else:
            # No sweep defined – single run with base config.
            combinations = [{}]

        total_runs = len(combinations) * n_repeats
        completed = 0

        for combo in combinations:
            # Merge base config with the current parameter combination.
            run_config = dict(self.base_config)
            run_config.update(combo)

            n_agents = run_config.pop("n_agents", self.base_config.get("n_agents", 10))
            max_steps = run_config.pop("max_steps", self.base_config.get("max_steps", 100))

            # Scenario-level parameters.
            scenario_kwargs: Dict[str, Any] = {}
            for key in ("n_tasks", "n_resource_points", "resource_types", "seed"):
                if key in run_config:
                    scenario_kwargs[key] = run_config.pop(key)

            scenario = DisasterReliefScenario(**scenario_kwargs) if scenario_kwargs else DisasterReliefScenario()

            for repeat_idx in range(n_repeats):
                run_id = f"sweep_{len(self.results):04d}"
                os.makedirs(output_base, exist_ok=True)

                runner = SimulationRunner(
                    scenario=scenario,
                    n_agents=n_agents,
                    output_dir=output_base,
                )

                t0 = time.perf_counter()
                summary = runner.run(max_steps=max_steps)
                duration = time.perf_counter() - t0

                record: Dict[str, Any] = {
                    **combo,
                    "run_id": run_id,
                    "n_agents": n_agents,
                    "max_steps": max_steps,
                    "steps": summary.get("steps_completed", 0),
                    "tasks_completed": summary.get("tasks_completed", 0),
                    "messages_sent": summary.get("messages_sent", 0),
                    "duration": round(duration, 4),
                    "repeat": repeat_idx,
                }
                self.results.append(record)

                completed += 1
                print(
                    f"[sweep] {completed}/{total_runs}  "
                    f"params={combo}  repeat={repeat_idx}  "
                    f"steps={record['steps']}  tasks={record['tasks_completed']}"
                )

        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def analyze(self) -> pd.DataFrame:
        """Aggregate sweep results by parameter values.

        Returns
        -------
        pd.DataFrame
            One row per unique parameter combination with ``mean``, ``std``,
            ``min``, ``max`` for each numeric metric.
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        numeric_metrics = ["steps", "tasks_completed", "messages_sent", "duration"]
        present_metrics = [m for m in numeric_metrics if m in df.columns]
        sweep_cols = [c for c in df.columns if c not in numeric_metrics + ["run_id", "repeat"]]

        if sweep_cols:
            grouped = df.groupby(sweep_cols, dropna=False)
        else:
            grouped = df

        stats_frames: List[pd.DataFrame] = []
        for metric in present_metrics:
            agg = grouped[metric].agg(["mean", "std", "min", "max"])
            agg.columns = [f"{metric}_{stat}" for stat in agg.columns]
            stats_frames.append(agg)

        if stats_frames:
            summary = pd.concat(stats_frames, axis=1)
        else:
            summary = pd.DataFrame(index=grouped.groups.keys())

        return summary.reset_index()

    def plot_results(
        self,
        param_name: str,
        metric: str = "tasks_completed",
    ) -> None:
        """Plot a metric against a single swept parameter with error bars.

        Parameters
        ----------
        param_name:
            The swept parameter on the x-axis.
        metric:
            The metric on the y-axis (default ``"tasks_completed"``).
        """
        if not self.results:
            print("[sweep] No results to plot.")
            return

        df = pd.DataFrame(self.results)
        if param_name not in df.columns:
            print(f"[sweep] Parameter '{param_name}' not found in results.")
            return

        summary = (
            df.groupby(param_name)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            summary[param_name],
            summary["mean"],
            yerr=summary["std"],
            fmt="o-",
            capsize=4,
            linewidth=2,
            markersize=6,
        )
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {param_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, filepath: str) -> None:
        """Save sweep results to a CSV file.

        Parameters
        ----------
        filepath:
            Destination path (parent directories are created automatically).
        """
        if not self.results:
            print("[sweep] No results to save.")
            return

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        pd.DataFrame(self.results).to_csv(filepath, index=False)
        print(f"[sweep] Results saved to {filepath}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sweep configuration and results to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``base_config``, ``sweep_params``, ``results``.
        """
        return {
            "base_config": self.base_config,
            "sweep_params": self._sweep_params,
            "results": self.results,
        }
