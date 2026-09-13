"""Unit tests for the Boid flocking simulation."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.boid_model import BoidModel
from src.agents.boid_agent import BoidAgent


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
    """Stepping the model increments step_count correctly."""
    model = BoidModel(n_agents=20)
    for i in range(10):
        model.step()
    assert model.step_count == 10


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
