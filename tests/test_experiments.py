"""Unit tests for the S5 experiment framework."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.parameter_sweep import ParameterSweep
from src.experiments.pipeline import ExperimentPipeline
from src.scenarios.disaster_relief import DisasterReliefScenario


# ------------------------------------------------------------------
# 1. ParameterSweep -- define sweep parameters
# ------------------------------------------------------------------


def test_parameter_sweep_define():
    """Create a sweep, define parameters, verify stored config."""
    sweep = ParameterSweep()
    sweep.define_sweep("n_agents", [5, 10, 15])
    sweep.define_sweep("max_steps", [50, 100])

    assert "n_agents" in sweep._sweep_params
    assert "max_steps" in sweep._sweep_params
    assert sweep._sweep_params["n_agents"] == [5, 10, 15]
    assert sweep._sweep_params["max_steps"] == [50, 100]


# ------------------------------------------------------------------
# 2. ParameterSweep -- run (small)
# ------------------------------------------------------------------


def test_parameter_sweep_run_small():
    """Run a sweep with 1 parameter, 2 values, 1 repeat; verify DataFrame."""
    sweep = ParameterSweep(base_config={"max_steps": 10})
    sweep.define_sweep("n_agents", [5, 10])

    results_df = sweep.run(n_repeats=1)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) >= 2
    assert "n_agents" in results_df.columns


# ------------------------------------------------------------------
# 3. ExperimentPipeline -- run_single
# ------------------------------------------------------------------


def test_pipeline_run_single():
    """Create a pipeline with DisasterReliefScenario, run_single, verify dict keys."""
    scenario = DisasterReliefScenario(n_tasks=3)
    pipeline = ExperimentPipeline(
        scenario=scenario,
        n_agents=5,
        max_steps=10,
        output_dir="data/test_runs",
    )

    result = pipeline.run_single()

    assert isinstance(result, dict)
    assert "run_id" in result
    assert result.get("n_agents", result.get("agents", 5)) is not None


# ------------------------------------------------------------------
# 4. ExperimentPipeline -- run_batch
# ------------------------------------------------------------------


def test_pipeline_run_batch():
    """Run 2 batch runs, verify the returned DataFrame has 2 rows."""
    scenario = DisasterReliefScenario(n_tasks=3)
    pipeline = ExperimentPipeline(
        scenario=scenario,
        n_agents=5,
        max_steps=10,
        output_dir="data/test_runs",
    )

    results_df = pipeline.run_batch(n_runs=2)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 2


# ------------------------------------------------------------------
# 5. ExperimentPipeline -- get_summary
# ------------------------------------------------------------------


def test_pipeline_get_summary():
    """Run a single experiment then call get_summary; verify dict returned."""
    scenario = DisasterReliefScenario(n_tasks=3)
    pipeline = ExperimentPipeline(
        scenario=scenario,
        n_agents=5,
        max_steps=10,
        output_dir="data/test_runs",
    )

    pipeline.run_single()
    summary = pipeline.get_summary()

    assert isinstance(summary, dict)
    assert "total_runs" in summary
    assert summary["total_runs"] >= 1
