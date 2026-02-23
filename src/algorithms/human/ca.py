"""
Cultural Algorithm (CA)
========================
- NumPy only (no scikit-learn, scipy.optimize, or other high-level libraries)
- Modular, well-documented, Python best practices
- Configurable parameters (population size, acceptance rate, iterations, etc.)
- Supports both continuous and discrete optimization problems

Demo problems:
    Continuous : Ackley function — highly multimodal, tests landscape exploration
                 f(x*) = 0  at  x* = (0,...,0)
    Discrete   : Traveling Salesman Problem (TSP) — shortest closed tour
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Belief Space — shared by both continuous and discrete variants
# ══════════════════════════════════════════════════════════════════════════════

class BeliefSpace:
    """
    Stores and updates cultural knowledge extracted from elite individuals.

    Knowledge sources
    -----------------
    Situational  : best exemplar solution found so far.
    Normative    : per-dimension bounds [L_j, U_j] of promising regions
                   (continuous) or city-at-position frequency table (discrete).
    """

    def __init__(self, mode: str, dim: int, bounds=None):
        """
        Parameters
        ----------
        mode : str
            'continuous' or 'discrete'.
        dim : int
            Number of dimensions (continuous) or cities (discrete/TSP).
        bounds : tuple of (float, float), required for continuous mode.
        """
        assert mode in ('continuous', 'discrete')
        self.mode = mode
        self.dim  = dim

        # Situational knowledge
        self.situational         = None
        self.situational_fitness = np.inf

        if mode == 'continuous':
            lb, ub = bounds
            # Normative: [lower, upper] promising interval per dimension
            self.norm_lower = np.full(dim, lb, dtype=float)
            self.norm_upper = np.full(dim, ub, dtype=float)
        else:
            # Normative: city-at-position frequency table (EMA-smoothed)
            self.norm_freq = np.zeros((dim, dim), dtype=float)

    # ── Continuous update ────────────────────────────────────────────────────
    def update_continuous(self, accepted: np.ndarray, accepted_fitness: np.ndarray):
        """Update belief space from elite individuals (continuous)."""
        best_idx = np.argmin(accepted_fitness)
        if accepted_fitness[best_idx] < self.situational_fitness:
            self.situational         = accepted[best_idx].copy()
            self.situational_fitness = accepted_fitness[best_idx]

        # Normative: shrink intervals toward observed elite range
        self.norm_lower = accepted.min(axis=0)
        self.norm_upper = accepted.max(axis=0)
        # Avoid degenerate intervals
        too_small = (self.norm_upper - self.norm_lower) < 1e-8
        self.norm_lower[too_small] -= 1e-4
        self.norm_upper[too_small] += 1e-4

    # ── Discrete update ──────────────────────────────────────────────────────
    def update_discrete(self, accepted_tours: np.ndarray, accepted_fitness: np.ndarray):
        """Update belief space from elite tours (discrete/TSP)."""
        best_idx = np.argmin(accepted_fitness)
        if accepted_fitness[best_idx] < self.situational_fitness:
            self.situational         = accepted_tours[best_idx].copy()
            self.situational_fitness = accepted_fitness[best_idx]

        # Normative: exponential moving average of city-at-position frequency
        freq = np.zeros((self.dim, self.dim), dtype=float)
        for tour in accepted_tours:
            for pos, city in enumerate(tour):
                freq[pos, city] += 1.0
        self.norm_freq = 0.6 * self.norm_freq + 0.4 * freq


# ══════════════════════════════════════════════════════════════════════════════
#  CA — Continuous Optimization
# ══════════════════════════════════════════════════════════════════════════════

def ca_continuous(
    obj_func,
    dim: int,
    bounds: tuple[float, float],
    n_pop: int = 50,
    max_iter: int = 400,
    alpha: float = 0.2,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    Cultural Algorithm for continuous optimization (minimization).

    Mutation uses both knowledge sources from the belief space:
      - Normative  : perturb each variable within its promising interval [L_j, U_j].
      - Situational: move toward the best known solution with a random step.

    Parameters
    ----------
    obj_func : callable
        Objective function f(x) -> float to minimize.
    dim : int
        Number of decision variables.
    bounds : tuple of (float, float)
        (lower_bound, upper_bound) for all dimensions.
    n_pop : int
        Population size.
    max_iter : int
        Maximum number of generations.
    alpha : float
        Acceptance rate — top floor(alpha * n_pop) individuals update belief space.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print best fitness every 100 iterations.

    Returns
    -------
    dict
        'best_solution' : np.ndarray
        'best_fitness'  : float
        'history'       : list[float]
    """
    rng    = np.random.default_rng(seed)
    lb, ub = bounds
    n_acc  = max(1, int(alpha * n_pop))
    belief = BeliefSpace('continuous', dim, bounds)

    # ── Initialisation ───────────────────────────────────────────────────────
    pop     = rng.uniform(lb, ub, size=(n_pop, dim))
    fitness = np.array([obj_func(x) for x in pop])

    sorted_idx = np.argsort(fitness)
    belief.update_continuous(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

    best_solution = pop[sorted_idx[0]].copy()
    best_fitness  = fitness[sorted_idx[0]]
    history       = []

    for t in range(max_iter):
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(n_pop):
            # 50 % chance: normative influence; 50 %: situational influence
            if rng.random() < 0.5:
                # Normative: perturb within promising intervals
                delta = rng.uniform(belief.norm_lower, belief.norm_upper)
                x_new = np.clip(pop[i] + 0.5 * (delta - pop[i]), lb, ub)
            else:
                # Situational: step toward global best with random scale
                if belief.situational is not None:
                    r     = rng.random(size=dim)
                    x_new = np.clip(
                        pop[i] + r * (belief.situational - pop[i]), lb, ub
                    )
                else:
                    x_new = np.clip(
                        rng.uniform(belief.norm_lower, belief.norm_upper), lb, ub
                    )

            f_new = obj_func(x_new)
            if f_new < fitness[i]:
                new_pop[i] = x_new
                new_fit[i] = f_new

        pop     = new_pop
        fitness = new_fit

        # ── Acceptance & belief update ───────────────────────────────────────
        sorted_idx = np.argsort(fitness)
        belief.update_continuous(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

        if fitness[sorted_idx[0]] < best_fitness:
            best_fitness  = fitness[sorted_idx[0]]
            best_solution = pop[sorted_idx[0]].copy()

        history.append(best_fitness)

        if verbose and (t + 1) % 100 == 0:
            print(f"  [CA] Iter {t+1:4d}/{max_iter} | Best fitness: {best_fitness:.6e}")

    return {
        "best_solution": best_solution,
        "best_fitness" : best_fitness,
        "history"      : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CA — Discrete Optimization (TSP)
# ══════════════════════════════════════════════════════════════════════════════

def _tour_length(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Compute total distance of a closed TSP tour."""
    n = len(tour)
    return float(sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n)))


def _two_opt(tour: np.ndarray, i: int, k: int) -> np.ndarray:
    """Reverse the segment [i, k] of a tour (2-opt move)."""
    new_tour          = tour.copy()
    new_tour[i:k + 1] = tour[i:k + 1][::-1]
    return new_tour


def ca_tsp(
    dist_matrix: np.ndarray,
    n_pop: int = 50,
    max_iter: int = 300,
    alpha: float = 0.2,
    influence_prob: float = 0.4,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    Cultural Algorithm for the Traveling Salesman Problem (TSP).

    Mutation operators guided by the belief space:
      - Situational (prob = influence_prob) : inherit a random sub-sequence
        from the best known tour (segment insertion).
      - Normative   (prob = 1 - influence_prob) : apply a 2-opt swap at the
        position with the weakest cultural alignment score.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        Symmetric matrix of inter-city distances.
    n_pop : int
        Population size.
    max_iter : int
        Maximum generations.
    alpha : float
        Acceptance rate: top floor(alpha * n_pop) tours update the belief space.
    influence_prob : float
        Probability of applying situational (vs. normative) influence.
    seed : int or None
        Random seed.
    verbose : bool
        Print best tour length every 50 iterations.

    Returns
    -------
    dict
        'best_tour'   : np.ndarray – best tour (0-indexed city order)
        'best_length' : float      – total distance of best tour
        'history'     : list[float]
    """
    rng      = np.random.default_rng(seed)
    n_cities = dist_matrix.shape[0]
    n_acc    = max(1, int(alpha * n_pop))
    belief   = BeliefSpace('discrete', n_cities)

    # ── Initialisation ───────────────────────────────────────────────────────
    pop     = np.array([rng.permutation(n_cities) for _ in range(n_pop)])
    fitness = np.array([_tour_length(t, dist_matrix) for t in pop])

    sorted_idx = np.argsort(fitness)
    belief.update_discrete(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

    best_tour   = pop[sorted_idx[0]].copy()
    best_length = fitness[sorted_idx[0]]
    history     = []

    for t in range(max_iter):
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(n_pop):
            tour = pop[i].copy()

            if rng.random() < influence_prob and belief.situational is not None:
                # ── Situational: inherit sub-sequence from best tour ──────────
                seg_len  = rng.integers(2, max(3, n_cities // 3))
                start    = rng.integers(0, n_cities - seg_len)
                segment  = belief.situational[start:start + seg_len]
                remaining = [c for c in tour if c not in segment]
                ins_pos   = rng.integers(0, len(remaining) + 1)
                new_tour  = np.array(
                    list(remaining[:ins_pos]) + list(segment) + list(remaining[ins_pos:]),
                    dtype=int
                )
            else:
                # ── Normative: 2-opt at weakest cultural position ─────────────
                scores  = np.array([
                    belief.norm_freq[pos, city]
                    for pos, city in enumerate(tour)
                ])
                weak_pos  = int(np.argmin(scores))
                other_pos = rng.integers(0, n_cities)
                if weak_pos != other_pos:
                    lo, hi   = min(weak_pos, other_pos), max(weak_pos, other_pos)
                    new_tour = _two_opt(tour, lo, hi)
                else:
                    new_tour = tour

            f_new = _tour_length(new_tour, dist_matrix)
            if f_new < fitness[i]:
                new_pop[i] = new_tour
                new_fit[i] = f_new

        pop     = new_pop
        fitness = new_fit

        # ── Acceptance & belief update ───────────────────────────────────────
        sorted_idx = np.argsort(fitness)
        belief.update_discrete(pop[sorted_idx[:n_acc]], fitness[sorted_idx[:n_acc]])

        if fitness[sorted_idx[0]] < best_length:
            best_length = fitness[sorted_idx[0]]
            best_tour   = pop[sorted_idx[0]].copy()

        history.append(best_length)

        if verbose and (t + 1) % 50 == 0:
            print(f"  [CA-TSP] Iter {t+1:4d}/{max_iter} | Best tour length: {best_length:.4f}")

    return {
        "best_tour"  : best_tour,
        "best_length": best_length,
        "history"    : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Continuous: Ackley ───────────────────────────────────────────────────
    def ackley(x):
        n   = len(x)
        a, b, c = 20, 0.2, 2 * np.pi
        s1  = np.sum(x ** 2)
        s2  = np.sum(np.cos(c * x))
        return float(-a * np.exp(-b * np.sqrt(s1 / n))
                     - np.exp(s2 / n) + a + np.e)

    print("=" * 56)
    print("  CA (Continuous) — Ackley Function, dim=10")
    print("  Global minimum: f(0,...,0) = 0")
    print("=" * 56)
    r1 = ca_continuous(ackley, dim=10, bounds=(-32.768, 32.768),
                       n_pop=50, max_iter=400, alpha=0.2, seed=42, verbose=True)
    print(f"\n  Best fitness : {r1['best_fitness']:.6e}")
    print(f"  Best solution: {np.round(r1['best_solution'], 4)}")

    # ── Discrete: TSP ────────────────────────────────────────────────────────
    rng_tsp  = np.random.default_rng(7)
    n_cities = 15
    coords   = rng_tsp.uniform(0, 100, size=(n_cities, 2))
    dist_mat = np.sqrt(
        ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    )

    print("\n" + "=" * 56)
    print(f"  CA (Discrete) — TSP, {n_cities} cities")
    print("=" * 56)
    print("  City coordinates:")
    for i, (x, y) in enumerate(coords):
        print(f"    City {i:2d}: ({x:.1f}, {y:.1f})")

    r2 = ca_tsp(dist_mat, n_pop=60, max_iter=400,
                alpha=0.2, influence_prob=0.4, seed=42, verbose=True)
    print(f"\n  Best tour   : {r2['best_tour'].tolist()}")
    print(f"  Tour length : {r2['best_length']:.4f}")