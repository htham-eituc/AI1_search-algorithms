"""
Teaching-Learning-Based Optimization (TLBO)
============================================
- NumPy only (no scikit-learn, scipy.optimize, or other high-level libraries)
- Modular, well-documented, Python best practices
- Configurable parameters (population size, iterations, bounds, dimension)
- Supports both continuous and discrete optimization problems

Demo problems:
    Continuous : Sphere function  f(x) = sum(x_i^2),  x* = 0,  f(x*) = 0
    Discrete   : Binary Knapsack — maximize total value subject to weight limit
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Core TLBO — Continuous Optimization
# ══════════════════════════════════════════════════════════════════════════════

def tlbo_continuous(
    obj_func,
    dim: int,
    bounds: tuple[float, float],
    n_pop: int = 30,
    max_iter: int = 500,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    TLBO for continuous optimization (minimization).

    Parameters
    ----------
    obj_func : callable
        Objective function f(x) -> float to minimize. x is a 1-D numpy array.
    dim : int
        Number of decision variables.
    bounds : tuple of (float, float)
        (lower_bound, upper_bound) applied uniformly to all dimensions.
    n_pop : int
        Number of learners (population size).
    max_iter : int
        Maximum number of iterations.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        If True, print best fitness every 100 iterations.

    Returns
    -------
    dict
        'best_solution' : np.ndarray  – best position found
        'best_fitness'  : float       – best objective value
        'history'       : list[float] – best fitness per iteration
    """
    rng = np.random.default_rng(seed)
    lb, ub = bounds

    # ── Initialisation ───────────────────────────────────────────────────────
    pop     = rng.uniform(lb, ub, size=(n_pop, dim))
    fitness = np.array([obj_func(x) for x in pop])

    best_idx      = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_fitness  = fitness[best_idx]
    history       = []

    for t in range(max_iter):

        # ── Teacher Phase ────────────────────────────────────────────────────
        teacher_idx = np.argmin(fitness)
        teacher     = pop[teacher_idx]
        mean        = pop.mean(axis=0)
        T_F         = np.round(1 + rng.random())          # T_F ∈ {1, 2}

        r       = rng.random(size=(n_pop, dim))
        new_pop = np.clip(pop + r * (teacher - T_F * mean), lb, ub)

        new_fitness = np.array([obj_func(x) for x in new_pop])
        improved    = new_fitness < fitness
        pop[improved]     = new_pop[improved]
        fitness[improved] = new_fitness[improved]

        # ── Learner Phase ─────────────────────────────────────────────────────
        indices = np.arange(n_pop)
        for i in range(n_pop):
            j   = rng.choice(indices[indices != i])
            r_i = rng.random(size=dim)

            if fitness[i] < fitness[j]:
                x_new = pop[i] + r_i * (pop[i] - pop[j])   # move away from worse
            else:
                x_new = pop[i] + r_i * (pop[j] - pop[i])   # move toward better

            x_new = np.clip(x_new, lb, ub)
            f_new = obj_func(x_new)
            if f_new < fitness[i]:
                pop[i]     = x_new
                fitness[i] = f_new

        # ── Track best ───────────────────────────────────────────────────────
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness  = fitness[best_idx]
            best_solution = pop[best_idx].copy()

        history.append(best_fitness)

        if verbose and (t + 1) % 100 == 0:
            print(f"  [TLBO] Iter {t+1:4d}/{max_iter} | Best fitness: {best_fitness:.6e}")

    return {
        "best_solution": best_solution,
        "best_fitness" : best_fitness,
        "history"      : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TLBO — Discrete Optimization (Binary Knapsack)
# ══════════════════════════════════════════════════════════════════════════════

def tlbo_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    n_pop: int = 40,
    max_iter: int = 300,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    TLBO adapted for the 0/1 Knapsack Problem (maximization).

    Continuous positions are mapped to binary decisions via a sigmoid
    threshold: item i is selected if sigmoid(x_i) > 0.5.
    Infeasible solutions are repaired by removing lowest value-to-weight
    ratio items until the weight constraint is satisfied.

    Parameters
    ----------
    values : np.ndarray, shape (n_items,)
        Value of each item.
    weights : np.ndarray, shape (n_items,)
        Weight of each item.
    capacity : float
        Maximum total weight allowed.
    n_pop : int
        Population size.
    max_iter : int
        Maximum iterations.
    seed : int or None
        Random seed.
    verbose : bool
        Print progress every 50 iterations.

    Returns
    -------
    dict
        'best_selection' : np.ndarray[bool] – selected items
        'best_value'     : float            – total value of best solution
        'best_weight'    : float            – total weight of best solution
        'history'        : list[float]      – best value per iteration
    """
    rng     = np.random.default_rng(seed)
    n_items = len(values)
    ratio   = values / (weights + 1e-9)             # value-to-weight ratio

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def decode(x: np.ndarray) -> np.ndarray:
        """Map continuous vector to feasible binary selection."""
        selected = sigmoid(x) > 0.5
        # Repair: remove items with lowest ratio until feasible
        while selected.any() and weights[selected].sum() > capacity:
            candidates = np.where(selected)[0]
            worst      = candidates[np.argmin(ratio[candidates])]
            selected[worst] = False
        return selected

    def fitness(x: np.ndarray) -> float:
        """Return total value of the decoded solution (to maximise → negate)."""
        sel = decode(x)
        return -float(values[sel].sum())            # negative for minimization

    # ── Initialisation ───────────────────────────────────────────────────────
    pop     = rng.uniform(-4, 4, size=(n_pop, n_items))
    fit     = np.array([fitness(x) for x in pop])

    best_idx = np.argmin(fit)
    best_sol = pop[best_idx].copy()
    best_fit = fit[best_idx]
    history  = []

    for t in range(max_iter):

        # ── Teacher Phase ────────────────────────────────────────────────────
        teacher = pop[np.argmin(fit)]
        mean    = pop.mean(axis=0)
        T_F     = np.round(1 + rng.random())

        r       = rng.random(size=(n_pop, n_items))
        new_pop = pop + r * (teacher - T_F * mean)

        new_fit = np.array([fitness(x) for x in new_pop])
        improved = new_fit < fit
        pop[improved] = new_pop[improved]
        fit[improved] = new_fit[improved]

        # ── Learner Phase ─────────────────────────────────────────────────────
        indices = np.arange(n_pop)
        for i in range(n_pop):
            j   = rng.choice(indices[indices != i])
            r_i = rng.random(size=n_items)
            if fit[i] < fit[j]:
                x_new = pop[i] + r_i * (pop[i] - pop[j])
            else:
                x_new = pop[i] + r_i * (pop[j] - pop[i])
            f_new = fitness(x_new)
            if f_new < fit[i]:
                pop[i] = x_new
                fit[i] = f_new

        # ── Track best ───────────────────────────────────────────────────────
        best_idx = np.argmin(fit)
        if fit[best_idx] < best_fit:
            best_fit = fit[best_idx]
            best_sol = pop[best_idx].copy()

        history.append(-best_fit)                   # store positive value

        if verbose and (t + 1) % 50 == 0:
            print(f"  [TLBO-Knapsack] Iter {t+1:3d}/{max_iter} | Best value: {-best_fit:.1f}")

    best_sel = decode(best_sol)
    return {
        "best_selection": best_sel,
        "best_value"    : float(values[best_sel].sum()),
        "best_weight"   : float(weights[best_sel].sum()),
        "history"       : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Continuous: Sphere ───────────────────────────────────────────────────
    def sphere(x):
        return float(np.sum(x ** 2))

    print("=" * 56)
    print("  TLBO (Continuous) — Sphere Function, dim=10")
    print("  Global minimum: f(0,...,0) = 0")
    print("=" * 56)
    r1 = tlbo_continuous(sphere, dim=10, bounds=(-5.12, 5.12),
                         n_pop=30, max_iter=500, seed=42, verbose=True)
    print(f"\n  Best fitness : {r1['best_fitness']:.6e}")
    print(f"  Best solution: {np.round(r1['best_solution'], 5)}")

    # ── Discrete: Knapsack ───────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    n_items  = 20
    values_  = rng.integers(10, 100, size=n_items).astype(float)
    weights_ = rng.integers(5,  50,  size=n_items).astype(float)
    capacity_= 100.0

    print("\n" + "=" * 56)
    print("  TLBO (Discrete) — 0/1 Knapsack, 20 items, cap=100")
    print("=" * 56)
    r2 = tlbo_knapsack(values_, weights_, capacity_,
                       n_pop=40, max_iter=300, seed=42, verbose=True)
    print(f"\n  Best value  : {r2['best_value']:.1f}")
    print(f"  Best weight : {r2['best_weight']:.1f} / {capacity_}")
    print(f"  Items chosen: {np.where(r2['best_selection'])[0].tolist()}")