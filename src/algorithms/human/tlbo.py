"""
Teaching-Learning-Based Optimization (TLBO)
=============================================
Rao, Savsani & Vakharia (2011).

Inherits BaseMetaheuristic from algorithms/base.py.

Two phases per iteration
------------------------
Teacher Phase : Best individual pulls population mean toward itself.
Learner Phase : Each learner moves toward a better peer (or away from a worse one).

Parameter-free: only pop_size and max_iter are required.
Demo problem  : Rastrigin function (continuous, dim=10).
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from algorithms.base import BaseMetaheuristic


# ── Shared diversity metric ───────────────────────────────────────────────────

def _diversity(population: np.ndarray) -> float:
    """Mean pairwise Euclidean distance — high = exploration, low = exploitation."""
    n = len(population)
    if n < 2:
        return 0.0
    diffs = population[:, None, :] - population[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    return float(dists.sum() / (n * (n - 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  TLBO
# ══════════════════════════════════════════════════════════════════════════════

class TLBO(BaseMetaheuristic):
    """
    Teaching-Learning-Based Optimization.

    Parameters
    ----------
    objective_func : callable
        f(population) -> np.ndarray shape (n,). Must be vectorised.
    pop_size : int
        Number of learners.
    max_iter : int
    bounds : np.ndarray, shape (dim, 2)
        Column 0 = lower bounds, column 1 = upper bounds.
    dim : int
        Number of decision variables.
    seed : int or None
    """

    def __init__(self, objective_func, pop_size: int = 30, max_iter: int = 300,
                 bounds: np.ndarray = None, dim: int = None, seed: int = None):
        super().__init__("TLBO", objective_func, pop_size, max_iter, bounds, dim)
        self.seed = seed

    def solve(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        t0 = time.time()

        # ── Initialisation ───────────────────────────────────────────────────
        pop     = self.initialize_population()      # (pop_size, dim)
        fitness = self.evaluate_population(pop)     # (pop_size,)

        best_idx           = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness  = fitness[best_idx]

        for t in range(self.max_iter):

            # ── Teacher Phase ─────────────────────────────────────────────────
            teacher_idx = np.argmin(fitness)
            teacher     = pop[teacher_idx]
            mean        = pop.mean(axis=0)
            T_F         = np.round(1 + np.random.random())    # T_F ∈ {1, 2}

            r       = np.random.random(size=(self.pop_size, self.dim))
            new_pop = np.clip(pop + r * (teacher - T_F * mean), lb, ub)

            new_fitness = self.evaluate_population(new_pop)
            improved    = new_fitness < fitness
            pop[improved]     = new_pop[improved]
            fitness[improved] = new_fitness[improved]

            # ── Learner Phase ─────────────────────────────────────────────────
            indices = np.arange(self.pop_size)
            for i in range(self.pop_size):
                j   = np.random.choice(indices[indices != i])
                r_i = np.random.random(size=self.dim)
                if fitness[i] < fitness[j]:
                    x_new = pop[i] + r_i * (pop[i] - pop[j])
                else:
                    x_new = pop[i] + r_i * (pop[j] - pop[i])
                x_new = np.clip(x_new, lb, ub)
                f_new = self.objective_func(x_new.reshape(1, -1))[0]
                if f_new < fitness[i]:
                    pop[i]     = x_new
                    fitness[i] = f_new

            # ── Track metrics ─────────────────────────────────────────────────
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness  = fitness[best_idx]
                self.best_solution = pop[best_idx].copy()

            self.convergence_curve[t]     = self.best_fitness
            self.average_fitness_curve[t] = fitness.mean()
            self.diversity_curve[t]       = _diversity(pop)

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

    model = TLBO(rastrigin, pop_size=30, max_iter=300, bounds=BOUNDS, dim=DIM, seed=42)
    model.solve()
    r = model.get_results()

    print(f"[{r['algorithm']}] Rastrigin dim={DIM}")
    print(f"  Best fitness   : {r['best_fitness']:.6f}")
    print(f"  Execution time : {r['execution_time_seconds']:.3f}s")
    print(f"  Final diversity: {r['diversity_curve'][-1]:.4f}")
