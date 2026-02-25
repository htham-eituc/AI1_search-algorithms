"""
Cultural Algorithm (CA)
========================
Reynolds (1994). Dual inheritance: population space + belief space.

Two classes
-----------
CA      — continuous optimization, inherits BaseMetaheuristic.
CA_TSP  — TSP (discrete), inherits BaseAlgorithm directly
          (permutation-based solutions don't fit the real-valued population matrix).

Belief space knowledge sources
-------------------------------
Situational : best exemplar solution found so far.
Normative   : per-dimension promising interval [L_j, U_j] (continuous)
              OR city-at-position frequency table (discrete).
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from algorithms.base import BaseMetaheuristic, BaseAlgorithm


def _diversity(population: np.ndarray) -> float:
    n = len(population)
    if n < 2:
        return 0.0
    diffs = population[:, None, :] - population[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    return float(dists.sum() / (n * (n - 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  Internal belief space
# ══════════════════════════════════════════════════════════════════════════════

class _BeliefSpace:
    """
    Stores cultural knowledge extracted from elite individuals.

    mode='continuous' : normative = per-dimension interval [L_j, U_j].
    mode='discrete'   : normative = city-at-position frequency table (EMA).
    """

    def __init__(self, mode: str, dim: int, bounds: np.ndarray = None):
        assert mode in ('continuous', 'discrete')
        self.mode                = mode
        self.dim                 = dim
        self.situational         = None
        self.situational_fitness = float('inf')

        if mode == 'continuous':
            self.norm_lower = bounds[:, 0].copy()
            self.norm_upper = bounds[:, 1].copy()
        else:
            self.norm_freq = np.zeros((dim, dim), dtype=float)

    def update_continuous(self, accepted: np.ndarray, accepted_fitness: np.ndarray):
        best_idx = np.argmin(accepted_fitness)
        if accepted_fitness[best_idx] < self.situational_fitness:
            self.situational         = accepted[best_idx].copy()
            self.situational_fitness = accepted_fitness[best_idx]
        self.norm_lower = accepted.min(axis=0)
        self.norm_upper = accepted.max(axis=0)
        too_small = (self.norm_upper - self.norm_lower) < 1e-8
        self.norm_lower[too_small] -= 1e-4
        self.norm_upper[too_small] += 1e-4

    def update_discrete(self, accepted_tours: np.ndarray, accepted_fitness: np.ndarray):
        best_idx = np.argmin(accepted_fitness)
        if accepted_fitness[best_idx] < self.situational_fitness:
            self.situational         = accepted_tours[best_idx].copy()
            self.situational_fitness = accepted_fitness[best_idx]
        freq = np.zeros((self.dim, self.dim), dtype=float)
        for tour in accepted_tours:
            for pos, city in enumerate(tour):
                freq[pos, city] += 1.0
        self.norm_freq = 0.6 * self.norm_freq + 0.4 * freq


# ══════════════════════════════════════════════════════════════════════════════
#  CA — Continuous
# ══════════════════════════════════════════════════════════════════════════════

class CA(BaseMetaheuristic):
    """
    Cultural Algorithm for continuous optimization.

    Mutation operators
    ------------------
    Normative   (50%) : perturb each variable within its promising interval [L_j, U_j].
    Situational (50%) : step toward the global best with a random scale.

    Parameters
    ----------
    objective_func : callable
        f(population) -> np.ndarray shape (n,). Must be vectorised.
    pop_size : int
    max_iter : int
    bounds : np.ndarray, shape (dim, 2)
    dim : int
    alpha : float
        Acceptance rate — top floor(alpha × pop_size) individuals update belief space.
    seed : int or None
    """

    def __init__(self, objective_func, pop_size: int = 50, max_iter: int = 300,
                 bounds: np.ndarray = None, dim: int = None,
                 alpha: float = 0.2, seed: int = None):
        super().__init__("CA", objective_func, pop_size, max_iter, bounds, dim)
        self.alpha = alpha
        self.seed  = seed

    def solve(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        lb    = self.bounds[:, 0]
        ub    = self.bounds[:, 1]
        n_acc = max(1, int(self.alpha * self.pop_size))
        t0    = time.time()

        belief = _BeliefSpace('continuous', self.dim, self.bounds)

        # ── Initialisation ───────────────────────────────────────────────────
        pop     = self.initialize_population()
        fitness = self.evaluate_population(pop)

        sorted_idx = np.argsort(fitness)
        belief.update_continuous(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

        self.best_solution = pop[sorted_idx[0]].copy()
        self.best_fitness  = fitness[sorted_idx[0]]

        for t in range(self.max_iter):
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(self.pop_size):
                if np.random.random() < 0.5:
                    # Normative influence
                    delta = np.random.uniform(belief.norm_lower, belief.norm_upper)
                    x_new = np.clip(pop[i] + 0.5 * (delta - pop[i]), lb, ub)
                else:
                    # Situational influence
                    if belief.situational is not None:
                        r     = np.random.random(size=self.dim)
                        x_new = np.clip(pop[i] + r * (belief.situational - pop[i]), lb, ub)
                    else:
                        x_new = np.clip(
                            np.random.uniform(belief.norm_lower, belief.norm_upper), lb, ub
                        )

                f_new = self.objective_func(x_new.reshape(1, -1))[0]
                if f_new < fitness[i]:
                    new_pop[i] = x_new
                    new_fit[i] = f_new

            pop     = new_pop
            fitness = new_fit

            sorted_idx = np.argsort(fitness)
            belief.update_continuous(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

            if fitness[sorted_idx[0]] < self.best_fitness:
                self.best_fitness  = fitness[sorted_idx[0]]
                self.best_solution = pop[sorted_idx[0]].copy()

            self.convergence_curve[t]     = self.best_fitness
            self.average_fitness_curve[t] = fitness.mean()
            self.diversity_curve[t]       = _diversity(pop)

        self.execution_time = time.time() - t0
        return self.get_results()


# ══════════════════════════════════════════════════════════════════════════════
#  CA_TSP — Discrete (Traveling Salesman Problem)
# ══════════════════════════════════════════════════════════════════════════════

class CA_TSP(BaseAlgorithm):
    """
    Cultural Algorithm for TSP — inherits BaseAlgorithm directly.

    TSP solutions are permutations, not real-valued vectors, so
    BaseMetaheuristic's population matrix and bounds are not applicable.

    Mutation operators
    ------------------
    Situational (prob=influence_prob) : inherit sub-sequence from best tour.
    Normative   (prob=1-influence_prob): 2-opt swap at weakest cultural position.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n_cities, n_cities)
        Symmetric inter-city distance matrix.
    pop_size : int
    max_iter : int
    alpha : float
        Acceptance rate for belief space update.
    influence_prob : float
        Probability of applying situational (vs normative) influence.
    seed : int or None
    """

    def __init__(self, dist_matrix: np.ndarray, pop_size: int = 50,
                 max_iter: int = 300, alpha: float = 0.2,
                 influence_prob: float = 0.4, seed: int = None):
        super().__init__("CA_TSP")
        self.dist_matrix    = dist_matrix
        self.n_cities       = dist_matrix.shape[0]
        self.pop_size       = pop_size
        self.max_iter       = max_iter
        self.alpha          = alpha
        self.influence_prob = influence_prob
        self.seed           = seed
        self.convergence_curve = np.zeros(max_iter)

    def _tour_length(self, tour: np.ndarray) -> float:
        n = self.n_cities
        return float(sum(
            self.dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n)
        ))

    @staticmethod
    def _two_opt(tour: np.ndarray, i: int, k: int) -> np.ndarray:
        t          = tour.copy()
        t[i:k + 1] = tour[i:k + 1][::-1]
        return t

    def solve(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        n_acc  = max(1, int(self.alpha * self.pop_size))
        belief = _BeliefSpace('discrete', self.n_cities)
        t0     = time.time()

        # ── Initialisation ───────────────────────────────────────────────────
        pop     = np.array([np.random.permutation(self.n_cities)
                             for _ in range(self.pop_size)])
        fitness = np.array([self._tour_length(t) for t in pop])

        sorted_idx = np.argsort(fitness)
        belief.update_discrete(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

        self.best_solution = pop[sorted_idx[0]].copy()
        self.best_fitness  = fitness[sorted_idx[0]]

        for t in range(self.max_iter):
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(self.pop_size):
                tour = pop[i].copy()

                if np.random.random() < self.influence_prob and belief.situational is not None:
                    # Situational: inherit sub-sequence from best tour
                    seg_len   = np.random.randint(2, max(3, self.n_cities // 3))
                    start     = np.random.randint(0, self.n_cities - seg_len)
                    segment   = belief.situational[start:start + seg_len]
                    remaining = [c for c in tour if c not in segment]
                    ins_pos   = np.random.randint(0, len(remaining) + 1)
                    new_tour  = np.array(
                        list(remaining[:ins_pos]) + list(segment) + list(remaining[ins_pos:]),
                        dtype=int
                    )
                else:
                    # Normative: 2-opt at weakest cultural position
                    scores   = np.array([belief.norm_freq[p, c] for p, c in enumerate(tour)])
                    weak_pos = int(np.argmin(scores))
                    other    = np.random.randint(0, self.n_cities)
                    if weak_pos != other:
                        lo, hi   = min(weak_pos, other), max(weak_pos, other)
                        new_tour = self._two_opt(tour, lo, hi)
                    else:
                        new_tour = tour

                f_new = self._tour_length(new_tour)
                if f_new < fitness[i]:
                    new_pop[i] = new_tour
                    new_fit[i] = f_new

            pop     = new_pop
            fitness = new_fit

            sorted_idx = np.argsort(fitness)
            belief.update_discrete(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

            if fitness[sorted_idx[0]] < self.best_fitness:
                self.best_fitness  = fitness[sorted_idx[0]]
                self.best_solution = pop[sorted_idx[0]].copy()

            self.convergence_curve[t] = self.best_fitness

        self.execution_time = time.time() - t0
        return self

    def get_results(self):
        results = super().get_results()
        results.update({"convergence_curve": self.convergence_curve})
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CA continuous ─────────────────────────────────────────────────────────
    def rastrigin(pop: np.ndarray) -> np.ndarray:
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        n = pop.shape[1]
        return 10 * n + np.sum(pop**2 - 10 * np.cos(2 * np.pi * pop), axis=1)

    DIM    = 10
    BOUNDS = np.array([[-5.12, 5.12]] * DIM)

    ca = CA(rastrigin, pop_size=50, max_iter=300, bounds=BOUNDS, dim=DIM,
            alpha=0.2, seed=42)
    ca.solve()
    r = ca.get_results()
    print(f"[{r['algorithm']}] Rastrigin dim={DIM}")
    print(f"  Best fitness   : {r['best_fitness']:.6f}")
    print(f"  Execution time : {r['execution_time_seconds']:.3f}s")
    print(f"  Final diversity: {r['diversity_curve'][-1]:.4f}\n")

    # ── CA_TSP ────────────────────────────────────────────────────────────────
    rng      = np.random.default_rng(7)
    n_cities = 15
    coords   = rng.uniform(0, 100, size=(n_cities, 2))
    dist_mat = np.sqrt(((coords[:, None] - coords[None, :])**2).sum(axis=2))

    ca_tsp = CA_TSP(dist_mat, pop_size=60, max_iter=300, seed=42)
    ca_tsp.solve()
    r2 = ca_tsp.get_results()
    print(f"[{r2['algorithm']}] {n_cities} cities")
    print(f"  Best tour length: {r2['best_fitness']:.4f}")
    print(f"  Tour            : {r2['best_solution'].tolist()}")
    print(f"  Execution time  : {r2['execution_time_seconds']:.3f}s")
