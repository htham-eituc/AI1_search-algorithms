"""
Social Force Optimization (SFO)
================================
Based on Helbing & Molnár (1995) pedestrian dynamics model.

Inherits BaseMetaheuristic from algorithms/base.py.

Each agent experiences
-----------------------
Goal force   : attraction toward the global best solution.
Social force : pairwise exponential repulsion to maintain diversity.
Velocity updated via Euler integration, clamped to v_max.

Demo problem : Rastrigin function (continuous, dim=10).
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from algorithms.base import BaseMetaheuristic


def _diversity(population: np.ndarray) -> float:
    n = len(population)
    if n < 2:
        return 0.0
    diffs = population[:, None, :] - population[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    return float(dists.sum() / (n * (n - 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  SFO
# ══════════════════════════════════════════════════════════════════════════════

class SFO(BaseMetaheuristic):
    """
    Social Force Optimization.

    Parameters
    ----------
    objective_func : callable
        f(population) -> np.ndarray shape (n,). Must be vectorised.
    pop_size : int
    max_iter : int
    bounds : np.ndarray, shape (dim, 2)
    dim : int
    desired_speed : float
        Target speed v^0 toward global best.
    tau : float
        Relaxation time — controls velocity adjustment rate.
    A : float
        Social repulsion amplitude.
    B : float
        Social repulsion decay range.
    r_agent : float
        Personal space radius of each agent.
    dt : float
        Euler integration time step.
    v_max : float
        Maximum velocity magnitude (prevents blow-up).
    seed : int or None
    """

    def __init__(self, objective_func, pop_size: int = 40, max_iter: int = 300,
                 bounds: np.ndarray = None, dim: int = None,
                 desired_speed: float = 0.8, tau: float = 0.5,
                 A: float = 2.0, B: float = 0.3, r_agent: float = 0.5,
                 dt: float = 0.1, v_max: float = 2.0, seed: int = None):
        super().__init__("SFO", objective_func, pop_size, max_iter, bounds, dim)
        self.desired_speed = desired_speed
        self.tau           = tau
        self.A             = A
        self.B             = B
        self.r_agent       = r_agent
        self.dt            = dt
        self.v_max         = v_max
        self.seed          = seed

    def solve(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        t0 = time.time()

        # ── Initialisation ───────────────────────────────────────────────────
        pos     = self.initialize_population()
        vel     = np.zeros((self.pop_size, self.dim))
        fitness = self.evaluate_population(pos)

        best_idx           = np.argmin(fitness)
        self.best_solution = pos[best_idx].copy()
        self.best_fitness  = fitness[best_idx]

        for t in range(self.max_iter):
            forces = np.zeros((self.pop_size, self.dim))

            for i in range(self.pop_size):
                # Goal force
                diff      = self.best_solution - pos[i]
                dist_goal = np.linalg.norm(diff) + 1e-12
                e0        = diff / dist_goal
                F_goal    = (self.desired_speed * e0 - vel[i]) / self.tau

                # Social repulsion
                F_social = np.zeros(self.dim)
                for j in range(self.pop_size):
                    if i == j:
                        continue
                    d_vec = pos[i] - pos[j]
                    d     = np.linalg.norm(d_vec) + 1e-12
                    F_social += self.A * np.exp((self.r_agent - d) / self.B) * (d_vec / d)

                forces[i] = F_goal + F_social

            # Velocity update + clamp
            vel    = vel + self.dt * forces
            speeds = np.linalg.norm(vel, axis=1, keepdims=True)
            mask   = (speeds > self.v_max).flatten()
            vel[mask] = vel[mask] / speeds[mask] * self.v_max

            pos     = np.clip(pos + self.dt * vel, lb, ub)
            fitness = self.evaluate_population(pos)

            # Track metrics
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness  = fitness[best_idx]
                self.best_solution = pos[best_idx].copy()

            self.convergence_curve[t]     = self.best_fitness
            self.average_fitness_curve[t] = fitness.mean()
            self.diversity_curve[t]       = _diversity(pos)

        self.execution_time = time.time() - t0
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def rastrigin(pop: np.ndarray) -> np.ndarray:
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        n = pop.shape[1]
        return 10 * n + np.sum(pop**2 - 10 * np.cos(2 * np.pi * pop), axis=1)

    DIM    = 10
    BOUNDS = np.array([[-5.12, 5.12]] * DIM)

    model = SFO(rastrigin, pop_size=40, max_iter=300, bounds=BOUNDS, dim=DIM, seed=42)
    model.solve()
    r = model.get_results()

    print(f"[{r['algorithm']}] Rastrigin dim={DIM}")
    print(f"  Best fitness   : {r['best_fitness']:.6f}")
    print(f"  Execution time : {r['execution_time_seconds']:.3f}s")
    print(f"  Final diversity: {r['diversity_curve'][-1]:.4f}")
