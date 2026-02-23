"""
Social Force Optimization (SFO)
================================
- NumPy only (no scikit-learn, scipy.optimize, or other high-level libraries)
- Modular, well-documented, Python best practices
- Configurable parameters (population, iterations, force params, bounds)
- Supports both continuous and discrete optimization problems

Demo problems:
    Continuous : Rosenbrock function  f(x) = sum[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
                 x* = (1,...,1),  f(x*) = 0
    Discrete   : Job Scheduling — minimize total weighted completion time
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Core SFO — Continuous Optimization
# ══════════════════════════════════════════════════════════════════════════════

def sfo_continuous(
    obj_func,
    dim: int,
    bounds: tuple[float, float],
    n_pop: int = 40,
    max_iter: int = 500,
    desired_speed: float = 0.8,
    tau: float = 0.5,
    A: float = 2.0,
    B: float = 0.3,
    r_agent: float = 0.5,
    dt: float = 0.1,
    v_max: float = 2.0,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    SFO for continuous optimization (minimization).

    Each agent experiences:
      - Goal force    : attraction toward the current global best.
      - Social force  : pairwise repulsion to maintain population diversity.
    Velocity is integrated with Euler method and clamped to v_max.

    Parameters
    ----------
    obj_func : callable
        Objective function f(x) -> float to minimize.
    dim : int
        Number of decision variables.
    bounds : tuple of (float, float)
        (lower_bound, upper_bound) for all dimensions.
    n_pop : int
        Number of agents.
    max_iter : int
        Maximum iterations.
    desired_speed : float
        Target speed v^0 toward global best (goal force strength).
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
        Velocity clamp to prevent numerical blow-up.
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
    rng = np.random.default_rng(seed)
    lb, ub = bounds

    # ── Initialisation ───────────────────────────────────────────────────────
    pos     = rng.uniform(lb, ub, size=(n_pop, dim))
    vel     = np.zeros((n_pop, dim))
    fitness = np.array([obj_func(x) for x in pos])

    best_idx      = np.argmin(fitness)
    best_solution = pos[best_idx].copy()
    best_fitness  = fitness[best_idx]
    history       = []

    for t in range(max_iter):
        forces = np.zeros((n_pop, dim))

        for i in range(n_pop):
            # ── Goal force: steer toward global best ─────────────────────────
            diff      = best_solution - pos[i]
            dist_goal = np.linalg.norm(diff) + 1e-12
            e0        = diff / dist_goal
            F_goal    = (desired_speed * e0 - vel[i]) / tau

            # ── Social repulsion: push agents apart ───────────────────────────
            F_social = np.zeros(dim)
            for j in range(n_pop):
                if i == j:
                    continue
                d_vec = pos[i] - pos[j]
                d     = np.linalg.norm(d_vec) + 1e-12
                F_social += A * np.exp((r_agent - d) / B) * (d_vec / d)

            forces[i] = F_goal + F_social

        # ── Velocity and position update (Euler) ─────────────────────────────
        vel = vel + dt * forces
        for i in range(n_pop):                         # velocity clamping
            speed = np.linalg.norm(vel[i])
            if speed > v_max:
                vel[i] = vel[i] / speed * v_max

        pos     = np.clip(pos + dt * vel, lb, ub)
        fitness = np.array([obj_func(x) for x in pos])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness  = fitness[best_idx]
            best_solution = pos[best_idx].copy()

        history.append(best_fitness)

        if verbose and (t + 1) % 100 == 0:
            print(f"  [SFO] Iter {t+1:4d}/{max_iter} | Best fitness: {best_fitness:.6e}")

    return {
        "best_solution": best_solution,
        "best_fitness" : best_fitness,
        "history"      : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SFO — Discrete Optimization (Job Scheduling)
# ══════════════════════════════════════════════════════════════════════════════

def sfo_scheduling(
    processing_times: np.ndarray,
    weights: np.ndarray,
    n_pop: int = 40,
    max_iter: int = 300,
    desired_speed: float = 0.6,
    tau: float = 0.5,
    A: float = 1.5,
    B: float = 0.5,
    r_agent: float = 0.5,
    dt: float = 0.1,
    v_max: float = 1.5,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    SFO for Job Scheduling — minimize total weighted completion time (TWCT).

    Agents maintain continuous priority vectors; jobs are scheduled in
    descending priority order. Social forces maintain diverse orderings
    while goal forces pull agents toward the best schedule found.

    Parameters
    ----------
    processing_times : np.ndarray, shape (n_jobs,)
        Processing time p_j for each job j.
    weights : np.ndarray, shape (n_jobs,)
        Weight w_j for each job j (importance of completion time).
    n_pop : int
        Number of agents.
    max_iter : int
        Maximum iterations.
    desired_speed, tau, A, B, r_agent, dt, v_max : float
        SFO force parameters (same semantics as sfo_continuous).
    seed : int or None
        Random seed.
    verbose : bool
        Print progress every 50 iterations.

    Returns
    -------
    dict
        'best_schedule'  : list[int]  – job order (0-indexed)
        'best_twct'      : float      – total weighted completion time
        'history'        : list[float]
    """
    rng    = np.random.default_rng(seed)
    n_jobs = len(processing_times)

    def decode(priority_vec: np.ndarray) -> np.ndarray:
        """Map priority vector to a job permutation (descending priority)."""
        return np.argsort(-priority_vec)

    def twct(schedule: np.ndarray) -> float:
        """Compute total weighted completion time for a given schedule."""
        total = 0.0
        C     = 0.0
        for j in schedule:
            C     += processing_times[j]
            total += weights[j] * C
        return total

    def fitness(priority_vec: np.ndarray) -> float:
        return twct(decode(priority_vec))

    # ── Initialisation ───────────────────────────────────────────────────────
    pos     = rng.uniform(0, 1, size=(n_pop, n_jobs))
    vel     = np.zeros((n_pop, n_jobs))
    fit     = np.array([fitness(x) for x in pos])

    best_idx  = np.argmin(fit)
    best_pos  = pos[best_idx].copy()
    best_fit  = fit[best_idx]
    history   = []

    for t in range(max_iter):
        forces = np.zeros((n_pop, n_jobs))

        for i in range(n_pop):
            diff   = best_pos - pos[i]
            dist   = np.linalg.norm(diff) + 1e-12
            e0     = diff / dist
            F_goal = (desired_speed * e0 - vel[i]) / tau

            F_social = np.zeros(n_jobs)
            for j in range(n_pop):
                if i == j:
                    continue
                dv = pos[i] - pos[j]
                d  = np.linalg.norm(dv) + 1e-12
                F_social += A * np.exp((r_agent - d) / B) * (dv / d)

            forces[i] = F_goal + F_social

        vel = vel + dt * forces
        for i in range(n_pop):
            speed = np.linalg.norm(vel[i])
            if speed > v_max:
                vel[i] = vel[i] / speed * v_max

        pos = np.clip(pos + dt * vel, 0, 1)
        fit = np.array([fitness(x) for x in pos])

        best_idx = np.argmin(fit)
        if fit[best_idx] < best_fit:
            best_fit = fit[best_idx]
            best_pos = pos[best_idx].copy()

        history.append(best_fit)

        if verbose and (t + 1) % 50 == 0:
            print(f"  [SFO-Schedule] Iter {t+1:3d}/{max_iter} | Best TWCT: {best_fit:.2f}")

    best_schedule = decode(best_pos).tolist()
    return {
        "best_schedule" : best_schedule,
        "best_twct"     : best_fit,
        "history"       : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Continuous: Rosenbrock ───────────────────────────────────────────────
    def rosenbrock(x):
        return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

    print("=" * 56)
    print("  SFO (Continuous) — Rosenbrock Function, dim=5")
    print("  Global minimum: f(1,...,1) = 0")
    print("=" * 56)
    r1 = sfo_continuous(rosenbrock, dim=5, bounds=(-2.048, 2.048),
                        n_pop=40, max_iter=500, seed=42, verbose=True)
    print(f"\n  Best fitness : {r1['best_fitness']:.6e}")
    print(f"  Best solution: {np.round(r1['best_solution'], 4)}")

    # ── Discrete: Job Scheduling ─────────────────────────────────────────────
    rng = np.random.default_rng(7)
    n_jobs   = 10
    proc     = rng.integers(1, 20, size=n_jobs).astype(float)
    wts      = rng.integers(1, 10, size=n_jobs).astype(float)

    # Optimal rule: Smith's rule — sort by w_j / p_j descending
    smith_order = np.argsort(-wts / proc)
    smith_twct  = float(sum(
        wts[smith_order[k]] * proc[smith_order[:k+1]].sum()
        for k in range(n_jobs)
    ))

    print("\n" + "=" * 56)
    print("  SFO (Discrete) — Job Scheduling, 10 jobs")
    print(f"  Processing times: {proc.astype(int).tolist()}")
    print(f"  Weights         : {wts.astype(int).tolist()}")
    print("=" * 56)
    r2 = sfo_scheduling(proc, wts, n_pop=40, max_iter=300, seed=42, verbose=True)
    print(f"\n  SFO  best TWCT  : {r2['best_twct']:.2f}")
    print(f"  SFO  schedule   : {r2['best_schedule']}")
    print(f"  Smith rule TWCT : {smith_twct:.2f}  (optimal reference)")