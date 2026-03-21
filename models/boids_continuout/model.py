"""
Boids flocking model using Mesa 3.x + mesa.space.ContinuousSpace

Classic Craig Reynolds (1987) Boids algorithm:
  separation  — steer away from boids that are too close
  alignment   — match the average heading of nearby boids
  cohesion    — steer toward the average position of nearby boids

Mesa concepts demonstrated:
  mesa.space.ContinuousSpace — stable continuous 2D toroidal space
  space.place_agent(agent, pos)
  space.move_agent(agent, new_pos)   — handles torus wrapping
  space.get_neighbors(pos, radius)   — radius-based lookup
  self.agents.shuffle_do("step")     — Mesa 3.x activation (no scheduler)
  mesa.DataCollector                 — per-step metrics
"""

import numpy as np
import mesa
import mesa.space


class Boid(mesa.Agent):
    """A single boid. State: pos (managed by space), vel (numpy array)."""

    def __init__(self, model, velocity):
        super().__init__(model)
        self.velocity = np.array(velocity, dtype=float)

    # ------------------------------------------------------------------ #
    # Three steering rules
    # ------------------------------------------------------------------ #

    def _separation(self, neighbors):
        """Push away from boids closer than min_dist."""
        force = np.zeros(2)
        for other in neighbors:
            delta = np.array(self.pos) - np.array(other.pos)
            dist = np.linalg.norm(delta)
            if 0 < dist < self.model.min_dist:
                force += delta / dist
        return force

    def _alignment(self, neighbors):
        """Match the average heading of neighbors."""
        if not neighbors:
            return np.zeros(2)
        avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
        return avg_vel - self.velocity

    def _cohesion(self, neighbors):
        """Steer toward the average position of neighbors."""
        if not neighbors:
            return np.zeros(2)
        avg_pos = np.mean([np.array(n.pos) for n in neighbors], axis=0)
        return avg_pos - np.array(self.pos)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self):
        # Get all boids within vision radius
        neighbors = [
            a for a in self.model.space.get_neighbors(
                self.pos, self.model.vision_radius, include_center=False
            )
            if a is not self
        ]

        # Weighted sum of three steering forces
        accel = (
            self.model.w_sep * self._separation(neighbors)
            + self.model.w_ali * self._alignment(neighbors)
            + self.model.w_coh * self._cohesion(neighbors)
        )

        self.velocity += accel

        # Clamp to max speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.model.max_speed:
            self.velocity = self.velocity / speed * self.model.max_speed

        # move_agent handles torus wrapping automatically
        new_pos = tuple(np.array(self.pos) + self.velocity)
        self.model.space.move_agent(self, new_pos)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #


class BoidsFlock(mesa.Model):
    """
    Boids flocking model on a continuous toroidal 2D space.

    Parameters
    ----------
    n_boids        : int   — number of agents
    width, height  : float — space dimensions
    vision_radius  : float — neighbourhood radius for all three rules
    min_dist       : float — separation kicks in below this distance
    max_speed      : float — speed cap per step
    w_sep          : float — separation weight
    w_ali          : float — alignment weight
    w_coh          : float — cohesion weight
    seed           : int   — random seed for reproducibility
    """

    def __init__(
        self,
        n_boids=50,
        width=100.0,
        height=100.0,
        vision_radius=10.0,
        min_dist=3.0,
        max_speed=2.0,
        w_sep=1.5,
        w_ali=1.0,
        w_coh=1.0,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.vision_radius = vision_radius
        self.min_dist = min_dist
        self.max_speed = max_speed
        self.w_sep = w_sep
        self.w_ali = w_ali
        self.w_coh = w_coh

        # mesa.space.ContinuousSpace — stable API, torus wraps edges
        self.space = mesa.space.ContinuousSpace(width, height, torus=True)

        for _ in range(n_boids):
            x = self.random.uniform(0, width)
            y = self.random.uniform(0, height)
            angle = self.random.uniform(0, 2 * np.pi)
            spd = self.random.uniform(0.5, max_speed)
            vel = (np.cos(angle) * spd, np.sin(angle) * spd)
            boid = Boid(self, velocity=vel)
            self.space.place_agent(boid, (x, y))

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "avg_speed": lambda m: float(
                    np.mean([np.linalg.norm(a.velocity) for a in m.agents])
                ),
                "n_boids": lambda m: len(list(m.agents)),
            }
        )
        self.datacollector.collect(self)

    def step(self):
        # Mesa 3.x activation — replaces RandomActivation from Mesa 2.x
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
