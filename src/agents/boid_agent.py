"""
Boid Agent for multi-agent flocking simulation.

Implements Craig Reynolds' Boids model (separation, alignment, cohesion)
on a toroidal continuous space using Mesa 3.x.
"""

import math

import mesa
import numpy as np


class BoidAgent(mesa.Agent):
    """An agent that follows the Boids flocking rules.

    Each agent steers based on three forces:
    - Separation: avoid crowding nearby agents
    - Alignment: steer toward average heading of nearby agents
    - Cohesion: steer toward average position of nearby agents

    The simulation space is treated as toroidal (wrapping at boundaries).
    """

    def __init__(
        self,
        model,
        unique_id,
        pos,
        velocity,
        perception_radius=50,
        separation_radius=25,
        max_speed=2.0,
        max_force=0.3,
    ):
        super().__init__(model)
        self.unique_id = unique_id
        # pos will be set by model.space.place_agent(); store as ndarray after placement
        self.velocity = np.array(velocity, dtype=float) # [vx, vy]
        self.perception_radius = perception_radius
        self.separation_radius = separation_radius
        self.max_speed = max_speed
        self.max_force = max_force

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, weights=(1.5, 1.0, 1.0)):
        """Advance one simulation tick.

        Parameters
        ----------
        weights : tuple of float
            (separation_weight, alignment_weight, cohesion_weight)
        """
        w_sep, w_ali, w_coh = weights

        # Gather neighbors within perception radius (toroidal distance).
        neighbors = [
            a
            for a in self.model.agents
            if a is not self
            and np.linalg.norm(self._toroidal_distance(a.pos)) < self.perception_radius
        ]

        if not neighbors:
            # No neighbors -- keep moving with current velocity.
            self._wrap_position()
            self.pos += self.velocity
            self._wrap_position()
            return

        # Compute the three flocking forces.
        sep = self.separate(neighbors)
        ali = self.align(neighbors)
        coh = self.cohere(neighbors)

        # Weighted combination of forces.
        steering = w_sep * sep + w_ali * ali + w_coh * coh

        # Limit the steering force.
        steering = self.limit_vector(steering, self.max_force)

        # Update velocity and limit speed.
        self.velocity += steering
        self.velocity = self.limit_vector(self.velocity, self.max_speed)

        # Move and wrap around boundaries.
        self.pos += self.velocity
        self._wrap_position()

    # ------------------------------------------------------------------
    # Flocking forces
    # ------------------------------------------------------------------

    def separate(self, neighbors):
        """Compute separation steering force.

        Agents within ``separation_radius`` are repulsed, weighted by
        inverse distance so that closer neighbors produce stronger
        repulsion.

        Parameters
        ----------
        neighbors : list of BoidAgent
            Nearby agents (already filtered by perception_radius).

        Returns
        -------
        np.ndarray
            Average repulsion vector, or a zero vector if no neighbor
            is within separation_radius.
        """
        repulsion = np.zeros(2, dtype=float)
        count = 0

        for agent in neighbors:
            diff = self._toroidal_distance(agent.pos)
            dist = np.linalg.norm(diff)

            if dist < self.separation_radius and dist > 0:
                # Point away from neighbor, weighted by inverse distance.
                repulsion += (-diff) / dist
                count += 1

        if count > 0:
            repulsion /= count

        return repulsion

    def align(self, neighbors):
        """Compute alignment steering force.

        Steers toward the average velocity of neighboring agents.

        Parameters
        ----------
        neighbors : list of BoidAgent
            Nearby agents.

        Returns
        -------
        np.ndarray
            Desired velocity change, limited by max_force.
        """
        avg_velocity = np.zeros(2, dtype=float)

        for agent in neighbors:
            avg_velocity += agent.velocity

        avg_velocity /= len(neighbors)

        # Desired change is the difference from current velocity.
        desired = avg_velocity - self.velocity
        return self.limit_vector(desired, self.max_force)

    def cohere(self, neighbors):
        """Compute cohesion steering force.

        Steers toward the center of mass of neighboring agents.

        Parameters
        ----------
        neighbors : list of BoidAgent
            Nearby agents.

        Returns
        -------
        np.ndarray
            Steering vector toward the center of mass, limited by
            max_force.
        """
        center_of_mass = np.zeros(2, dtype=float)

        for agent in neighbors:
            # Use toroidal offset so that wrapping agents still
            # contribute a meaningful center of mass.
            center_of_mass += self._toroidal_distance(agent.pos)

        center_of_mass /= len(neighbors)

        # Steering = desired (toward center) minus current velocity.
        desired = center_of_mass - self.velocity
        return self.limit_vector(desired, self.max_force)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def limit_vector(v, max_mag):
        """Scale *v* to at most *max_mag* while preserving direction.

        Parameters
        ----------
        v : np.ndarray
            The vector to limit.
        max_mag : float
            Maximum allowed magnitude.

        Returns
        -------
        np.ndarray
            The (possibly scaled) vector.
        """
        mag = np.linalg.norm(v)
        if mag > max_mag:
            return (v / mag) * max_mag
        return v

    def _toroidal_distance(self, other_pos):
        """Compute the shortest distance vector on a toroidal space.

        Wraps each axis so that the returned vector always represents
        the shortest path between ``self.pos`` and *other_pos*.

        Parameters
        ----------
        other_pos : array-like
            Target position [x, y].

        Returns
        -------
        np.ndarray
            Shortest toroidal displacement vector from self to other.
        """
        width = self.model.space.width
        height = self.model.space.height

        dx = other_pos[0] - self.pos[0]
        dy = other_pos[1] - self.pos[1]

        if dx > width / 2:
            dx -= width
        elif dx < -width / 2:
            dx += width

        if dy > height / 2:
            dy -= height
        elif dy < -height / 2:
            dy += height

        return np.array([dx, dy])

    def _wrap_position(self):
        """Wrap the agent's position to stay within the toroidal space."""
        self.pos = np.array([
            self.pos[0] % self.model.space.width,
            self.pos[1] % self.model.space.height,
        ])
