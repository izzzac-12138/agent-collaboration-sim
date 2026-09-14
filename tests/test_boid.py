"""Unit tests for the Boid flocking simulation."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.boid.models.boid_model import BoidModel
from src.boid.agents.boid_agent import BoidAgent


def test_model_creation():
    """Model creates the correct number of agents and a space."""
    model = BoidModel(n_agents=10, width=100, height=100)
    assert len(model.agents_list) == 10
    assert model.space is not None


def test_agent_initialization():
    """Every agent starts with valid position and velocity within bounds."""
    model = BoidModel(n_agents=5)
    for agent in model.agents_list:
        assert agent.pos is not None and len(agent.pos) == 2
        assert agent.velocity is not None and len(agent.velocity) == 2
        assert 0 <= agent.pos[0] <= model.space.width
        assert 0 <= agent.pos[1] <= model.space.height


def test_model_steps():
    """Stepping the model increments step_count correctly (design requires 100 steps)."""
    model = BoidModel(n_agents=20)
    for i in range(100):
        model.step()
    assert model.step_count == 100


def test_separation_force():
    """Separation force points away from nearby agents."""
    model = BoidModel(n_agents=3, width=200, height=200,
                      separation_radius=25, max_speed=2.0, max_force=0.3)
    a0, a1, a2 = model.agents_list

    # Place a1 very close to a0 (within separation radius)
    a0.pos = np.array([100.0, 100.0])
    a1.pos = np.array([105.0, 100.0])  # 5 units away, inside separation_radius
    a2.pos = np.array([10.0, 10.0])    # far away, should not affect separation

    sep = a0.separate([a1])
    # Force should point away from a1: negative x direction
    assert sep[0] < 0, f"Expected negative x separation, got {sep[0]}"
    assert np.linalg.norm(sep) > 0, "Separation force should be non-zero"


def test_alignment_force():
    """Alignment force steers toward average neighbor velocity."""
    model = BoidModel(n_agents=3, width=200, height=200,
                      max_speed=2.0, max_force=0.3)
    a0, a1, a2 = model.agents_list

    # Set a0 velocity to zero, neighbors moving right
    a0.velocity = np.array([0.0, 0.0])
    a1.velocity = np.array([2.0, 0.0])
    a2.velocity = np.array([2.0, 0.0])

    ali = a0.align([a1, a2])
    # Force should push a0 toward positive x (matching neighbors)
    assert ali[0] > 0, f"Expected positive x alignment, got {ali[0]}"
    assert np.linalg.norm(ali) > 0, "Alignment force should be non-zero"


def test_cohesion_force():
    """Cohesion force points toward center of mass of neighbors."""
    model = BoidModel(n_agents=3, width=200, height=200,
                      max_speed=2.0, max_force=0.3)
    a0, a1, a2 = model.agents_list

    # Place a0 at origin, neighbors far to the right
    a0.pos = np.array([50.0, 100.0])
    a1.pos = np.array([150.0, 100.0])
    a2.pos = np.array([150.0, 100.0])
    a0.velocity = np.array([0.0, 0.0])

    coh = a0.cohere([a1, a2])
    # Center of mass is at (150, 100), so force should point right
    assert coh[0] > 0, f"Expected positive x cohesion, got {coh[0]}"
    assert np.linalg.norm(coh) > 0, "Cohesion force should be non-zero"


def test_boundary_wrapping():
    """Agents remain within space boundaries after many steps (toroidal wrap)."""
    model = BoidModel(n_agents=5, width=50, height=50)
    for _ in range(50):
        model.step()
    for agent in model.agents_list:
        assert 0 <= agent.pos[0] <= 50
        assert 0 <= agent.pos[1] <= 50


def test_speed_limiting():
    """Agent speed never exceeds max_speed after stepping."""
    model = BoidModel(n_agents=20, max_speed=2.0)
    for _ in range(20):
        model.step()
    for agent in model.agents_list:
        speed = np.linalg.norm(agent.velocity)
        assert speed <= 2.0 + 1e-6  # small tolerance for float


def test_network_construction():
    """Interaction network has the correct number of nodes."""
    model = BoidModel(n_agents=30, perception_radius=50)
    model.step()
    G = model.get_interaction_network()
    assert G.number_of_nodes() == 30
    assert G.number_of_edges() >= 0  # at least doesn't crash


def test_data_collection():
    """DataCollector records one row per step with expected columns."""
    model = BoidModel(n_agents=10)
    for i in range(5):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert len(df) == 5
    assert 'n_edges' in df.columns
    assert 'avg_degree' in df.columns


def test_get_positions():
    """get_agent_positions returns an (N, 2) numpy array."""
    model = BoidModel(n_agents=10)
    positions = model.get_agent_positions()
    assert positions.shape == (10, 2)
