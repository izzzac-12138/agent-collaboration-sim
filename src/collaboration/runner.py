"""SimulationRunner -- unified entry point for running collaboration simulations."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from .model import CollaborationModel


class SimulationRunner:
    """High-level wrapper that creates a :class:`CollaborationModel`, runs it,
    and persists the results.

    Parameters
    ----------
    scenario:
        An optional scenario object forwarded to :class:`CollaborationModel`.
    n_agents:
        Number of agents to spawn in each run.
    width:
        Width of the continuous toroidal space.
    height:
        Height of the continuous toroidal space.
    output_dir:
        Directory where run summaries are saved as JSON files.
    """

    def __init__(
        self,
        scenario: Optional[object] = None,
        n_agents: int = 10,
        width: float = 200.0,
        height: float = 200.0,
        output_dir: str = "data/runs",
    ) -> None:
        self.scenario = scenario
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_steps: int = 200) -> Dict[str, Any]:
        """Execute a single simulation run.

        Steps
        -----
        1. Generate a unique ``run_id``.
        2. Create a :class:`CollaborationModel` with the configured parameters.
        3. Advance the model step-by-step until *max_steps* or ``model.done``.
        4. Flush the event logger.
        5. Persist the summary as a JSON file inside ``output_dir``.
        6. Return the summary dictionary.

        Parameters
        ----------
        max_steps:
            Maximum number of simulation ticks.

        Returns
        -------
        dict
            A summary dictionary containing at least ``run_id``,
            ``steps_completed``, ``n_agents``, and ``done``.
        """
        run_id: str = uuid.uuid4().hex[:12]

        model = CollaborationModel(
            scenario=self.scenario,
            n_agents=self.n_agents,
            width=self.width,
            height=self.height,
        )

        steps_completed: int = 0
        for _ in range(max_steps):
            if model.done:
                break
            model.step()
            steps_completed += 1

        # Flush buffered log events (best-effort; logger may not exist yet).
        self._flush_logger(model)

        summary: Dict[str, Any] = self._build_summary(model, run_id, steps_completed)

        self._save_summary(summary, run_id)

        return summary

    def run_batch(
        self, n_runs: int = 5, max_steps: int = 200
    ) -> List[Dict[str, Any]]:
        """Execute multiple independent runs and collect their summaries.

        Parameters
        ----------
        n_runs:
            Number of runs to execute.
        max_steps:
            Maximum number of simulation ticks per run.

        Returns
        -------
        list[dict]
            A list of summary dictionaries, one per run.
        """
        summaries: List[Dict[str, Any]] = []
        for _ in range(n_runs):
            summaries.append(self.run(max_steps=max_steps))
        return summaries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _flush_logger(model: CollaborationModel) -> None:
        """Best-effort flush of the event logger's buffered events.

        The logger may expose ``flush`` or ``save`` methods depending on the
        implementation; if neither exists we silently skip.
        """
        logger = model.event_logger
        for method_name in ("flush", "save"):
            method = getattr(logger, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass  # Non-critical -- log persistence is best-effort.
                break

    @staticmethod
    def _build_summary(
        model: CollaborationModel,
        run_id: str,
        steps_completed: int,
    ) -> Dict[str, Any]:
        """Construct a summary dictionary from the final model state."""
        return {
            "run_id": run_id,
            "steps_completed": steps_completed,
            "n_agents": len(model.agents_list),
            "done": model.done,
            "scenario": type(model.scenario).__name__ if model.scenario else None,
        }

    def _save_summary(self, summary: Dict[str, Any], run_id: str) -> None:
        """Persist *summary* as a JSON file under ``output_dir``.

        Directories are created automatically if they do not yet exist.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"run_{run_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
