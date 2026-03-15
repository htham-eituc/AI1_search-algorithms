"""
dim2_experiment.py
==================
Runs 2D trials for all three problems for GIF/contour visualization.

FAIRNESS
--------
All algorithms run max_iter = MAX_ITER_DIM2 = 500 iterations.
SA cooling_rate and HC step_decay are adjusted by fairness.py to remain
meaningful over 500 steps. Convergence curves are all length 500.

Outputs
-------
  results/dim2_sphere.pkl
  results/dim2_rastrigin.pkl
  results/dim2_rosenbrock.pkl

Each PKL:
  { "<algo>": {
      "algorithm", "best_fitness", "best_solution",
      "convergence_curve"     (len=500),
      "average_fitness_curve" (len=500),
      "diversity_curve"       (len=500),
      "population_history"    (list of (pop_size, 2) arrays),
      "trajectory"            (np.array shape (T, 2)),
      "execution_time_seconds",
      "max_iter", "evals_per_iter", "nfe",
  } }
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.configHelper import load_config
from utils.fairness import (
    build_fair_params,
    evals_per_iter,
    print_fairness_table,
    MAX_ITER_DIM2,
)
from problems.continuous import sphere, rastrigin, rosenbrock

from algorithms.evolution.DE         import DE
from algorithms.evolution.GA         import GA
from algorithms.physics.SA           import SimulatedAnnealing
from algorithms.physics.GSA          import GSA
from algorithms.biology.ABC          import ABC
from algorithms.biology.CS           import CuckooSearch
from algorithms.biology.FA           import FireflyAlgorithm
from algorithms.biology.PSO          import PSO
from algorithms.human.tlbo           import TLBO
from algorithms.human.sfo            import SFO
from algorithms.human.ca             import CA
from algorithms.classic.local_search import HillClimbingContinuous

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DIM = 2

PROBLEM_ALGOS = {
    "sphere"    : ["HC", "SA", "GA", "DE", "GSA", "PSO", "TLBO"],
    "rastrigin" : ["HC", "SA", "GA", "DE", "ABC", "FA",  "CS",  "CA"],
    "rosenbrock": ["HC", "SA", "GA", "DE", "ABC", "PSO", "SFO", "TLBO"],
}

PROBLEM_BOUNDS = {
    "sphere"    : (-5.12, 5.12),
    "rastrigin" : (-5.12, 5.12),
    "rosenbrock": (-2.0,   2.0),
}

PROBLEM_FUNCS = {
    "sphere"    : sphere,
    "rastrigin" : rastrigin,
    "rosenbrock": rosenbrock,
}

RESULTS_DIR = ROOT / "results"
GIFS_DIR    = RESULTS_DIR / "gifs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GIFS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory-tracking shim (zero changes to algorithm source files)
# ─────────────────────────────────────────────────────────────────────────────

def _add_trajectory_tracking(algo_instance):
    """
    Dual-patch trajectory tracker — covers two calling patterns:

    Pattern A — population-based algorithms (GA, DE, PSO, SA, …):
        call self.evaluate_population(array)  → patched here

    Pattern B — HC (HillClimbingContinuous):
        calls self.objective_func(array) directly, bypassing
        evaluate_population entirely → also patched here

    Both patches share a single _record() helper and write into the same
    _tracked_trajectory list so the worker uses one code path for all algos.

    Stores into algo_instance._tracked_trajectory  list of np.array (2,)
    """
    algo_instance._tracked_trajectory = []
    algo_instance._tracked_best_pos   = None
    algo_instance._tracked_best_fit   = float("inf")

    def _record(population, fitness):
        """Update running best and append current best position."""
        pop = np.atleast_2d(population)
        fit = np.atleast_1d(fitness)
        best_idx = int(np.argmin(fit))
        if fit[best_idx] < algo_instance._tracked_best_fit:
            algo_instance._tracked_best_fit = float(fit[best_idx])
            algo_instance._tracked_best_pos = np.copy(pop[best_idx])
        if algo_instance._tracked_best_pos is not None:
            algo_instance._tracked_trajectory.append(
                np.copy(algo_instance._tracked_best_pos)
            )

    # ── Patch A: evaluate_population (GA, DE, PSO, SA, …) ────────────────────
    original_eval = algo_instance.evaluate_population

    def _patched_eval(population):
        fitness = original_eval(population)
        _record(population, fitness)
        return fitness

    algo_instance.evaluate_population = _patched_eval

    # ── Patch B: objective_func (HC calls this directly, bypasses eval_pop) ───
    original_obj = algo_instance.objective_func

    def _patched_obj(population):
        fitness = original_obj(population)
        _record(population, fitness)
        return fitness

    algo_instance.objective_func = _patched_obj

    return algo_instance


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm factory — uses fair max_iter from fairness.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_algo(algo_name, problem_name, config):
    func   = PROBLEM_FUNCS[problem_name]
    lo, hi = PROBLEM_BOUNDS[problem_name]
    bounds = np.array([[lo, hi], [lo, hi]])

    p        = build_fair_params(algo_name, problem_name, DIM, config, MAX_ITER_DIM2)
    pop      = p.get("population_size", 50)
    max_iter = p["max_iterations"]      # always MAX_ITER_DIM2 = 500

    if algo_name == "DE":
        return DE(objective_func=func, bounds=bounds, dim=DIM,
                  pop_size=pop, max_iter=max_iter,
                  F=p.get("mutation_factor", 0.8),
                  CR=p.get("crossover_rate", 0.9))

    if algo_name == "GA":
        return GA(objective_func=func, bounds=bounds, dim=DIM,
                  pop_size=pop, max_iter=max_iter,
                  crossover_rate=p.get("crossover_rate", 0.8),
                  mutation_rate=p.get("mutation_rate", 0.1),
                  mutation_scale=p.get("mutation_scale", 0.1),
                  tournament_size=p.get("tournament_size", 3))

    if algo_name == "SA":
        return SimulatedAnnealing(
                  objective_func=func, bounds=bounds, dim=DIM,
                  pop_size=p.get("population_size", 10),
                  max_iter=max_iter,
                  initial_temperature=p.get("initial_temperature", 1000.0),
                  cooling_rate=p["cooling_rate"],      # ← adjusted
                  cooling_schedule=p.get("cooling_schedule", "geometric"))

    if algo_name == "GSA":
        return GSA(objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=pop, max_iter=max_iter,
                   G0=p.get("gravitational_constant", 100.0),
                   kbest_initial=p.get("alpha", 20))

    if algo_name == "ABC":
        return ABC(objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=pop, max_iter=max_iter,
                   limit=p.get("limit", None))

    if algo_name == "CS":
        return CuckooSearch(
                   objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=pop, max_iter=max_iter,
                   pa=p.get("pa", 0.25),
                   alpha=p.get("alpha", 0.01),
                   lambda_levy=p.get("lambda_levy", 1.5))

    if algo_name == "FA":
        return FireflyAlgorithm(
                   objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=pop, max_iter=max_iter,
                   beta0=p.get("beta0", 1.0),
                   gamma=p.get("gamma", 1.0),
                   alpha=p.get("alpha", 0.5),
                   alpha_decay=p.get("alpha_decay", 0.97))

    if algo_name == "PSO":
        return PSO(objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=pop, max_iter=max_iter,
                   w_max=p.get("w_max", 0.9),
                   w_min=p.get("w_min", 0.4),
                   c1=p.get("c1", 2.0),
                   c2=p.get("c2", 2.0),
                   v_clamp_ratio=p.get("v_clamp_ratio", 0.2))

    if algo_name == "HC":
        return HillClimbingContinuous(
                   objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=p.get("population_size", 10),
                   max_iter=max_iter,
                   step_size=p.get("step_size", 0.5),
                   step_decay=p["step_decay"],          # ← adjusted
                   max_restarts=p.get("max_restarts", 10),
                   patience=p.get("patience", 30))

    if algo_name == "TLBO":
        return TLBO(objective_func=func, bounds=bounds, dim=DIM,
                    pop_size=pop, max_iter=max_iter)

    if algo_name == "SFO":
        return SFO(objective_func=func, bounds=bounds, dim=DIM,
                   pop_size=pop, max_iter=max_iter,
                   desired_speed=p.get("desired_speed", 0.8),
                   tau=p.get("tau", 0.5),
                   A=p.get("A", 2.0),
                   B=p.get("B", 0.3),
                   r_agent=p.get("r_agent", 0.5),
                   dt=p.get("dt", 0.1),
                   v_max=p.get("v_max", 2.0))

    if algo_name == "CA":
        return CA(objective_func=func, bounds=bounds, dim=DIM,
                  pop_size=pop, max_iter=max_iter,
                  alpha=p.get("alpha", 0.2))

    raise ValueError(f"No factory entry for '{algo_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Per-iteration best-position extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_best_per_iter(population_history, func):
    """Return (T, 2) monotonically-improving best-position array."""
    best_positions   = []
    running_best_fit = float("inf")
    running_best_pos = None

    for snap in population_history:
        snap = np.array(snap)
        fits = func(snap)
        idx  = int(np.argmin(fits))
        if fits[idx] < running_best_fit:
            running_best_fit = fits[idx]
            running_best_pos = snap[idx].copy()
        best_positions.append(running_best_pos.copy())

    return np.array(best_positions)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    algo_name, problem_name, config, seed = args
    np.random.seed(seed)

    try:
        algo = _build_algo(algo_name, problem_name, config)
        _add_trajectory_tracking(algo)

        result   = algo.solve()
        pop_hist = result.get("population_history", [])
        func     = PROBLEM_FUNCS[problem_name]

        if len(pop_hist) > 0:
            trajectory = _extract_best_per_iter(pop_hist, func)
        else:
            traj_raw   = getattr(algo, "_tracked_trajectory", [])
            trajectory = np.array(traj_raw) if traj_raw else None

        p        = build_fair_params(algo_name, problem_name, DIM, config, MAX_ITER_DIM2)
        pop_size = p.get("population_size", 50)
        epi      = evals_per_iter(algo_name, DIM, pop_size)

        return algo_name, {
            "algorithm"             : result["algorithm"],
            "best_fitness"          : result["best_fitness"],
            "best_solution"         : result["best_solution"],
            "convergence_curve"     : result["convergence_curve"],
            "average_fitness_curve" : result["average_fitness_curve"],
            "diversity_curve"       : result["diversity_curve"],
            "population_history"    : pop_hist,
            "trajectory"            : trajectory,
            "execution_time_seconds": result["execution_time_seconds"],
            "max_iter"              : MAX_ITER_DIM2,
            "evals_per_iter"        : epi,
            "nfe"                   : epi * MAX_ITER_DIM2,
        }

    except Exception as exc:
        import traceback
        print(f"  [ERROR] {algo_name} on {problem_name}: {exc}")
        traceback.print_exc()
        return algo_name, None


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dim2_experiments(config, problems=None, max_workers=None):
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    problems  = problems or list(PROBLEM_ALGOS.keys())
    seed_base = config["experiment"]["seed"]
    all_results = {}

    for problem in problems:
        algos = PROBLEM_ALGOS[problem]

        print(f"\n{'─'*65}")
        print(f"[{problem.upper()}]  dim={DIM}  max_iter={MAX_ITER_DIM2} (all algos)")
        print_fairness_table(algos, dim=DIM, max_iter=MAX_ITER_DIM2)

        tasks = [
            (algo, problem, config, seed_base + i * 7 + abs(hash(algo)) % 100)
            for i, algo in enumerate(algos)
        ]

        problem_results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                algo_name, data = future.result()
                if data is not None:
                    problem_results[algo_name] = data
                    print(
                        f"  ✓ {algo_name:<8}  "
                        f"best={data['best_fitness']:.4e}  "
                        f"time={data['execution_time_seconds']:.3f}s  "
                        f"max_iter={data['max_iter']}  "
                        f"epi={data['evals_per_iter']}"
                    )
                else:
                    print(f"  ✗ {algo_name}  FAILED")

        all_results[problem] = problem_results

        pkl_path = RESULTS_DIR / f"dim2_{problem}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(problem_results, f)
        print(f"\n  Saved → {pkl_path}  ({pkl_path.stat().st_size / 1024:.0f} KB)")

    return all_results


def print_summary(all_results):
    print(f"\n{'='*80}")
    print("DIM-2 SUMMARY")
    print(f"{'='*80}")
    for problem, prob_results in all_results.items():
        print(f"\n  {problem.upper()}")
        print(f"  {'Algorithm':<10} {'Best Fitness':>15} {'Time':>8} "
              f"{'MaxIter':>9} {'EPI':>6} {'TotalNFE':>10}")
        print(f"  {'─'*62}")
        for algo, data in sorted(prob_results.items(),
                                  key=lambda x: x[1]["best_fitness"]):
            print(
                f"  {algo:<10} {data['best_fitness']:>15.4e} "
                f"{data['execution_time_seconds']:>8.3f}s "
                f"{data['max_iter']:>9} "
                f"{data['evals_per_iter']:>6} "
                f"{data['nfe']:>10}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run dim=2 experiments with fair max_iter for GIF rendering"
    )
    parser.add_argument(
        "problems", nargs="*",
        choices=["sphere", "rastrigin", "rosenbrock"],
        help="Problems to run (default: all three)",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config_path = ROOT / "utils/config.json"
    config      = load_config(config_path)
    np.random.seed(config["experiment"]["seed"])

    print("=" * 65)
    print(f"DIM-2 EXPERIMENT  |  dim={DIM}  max_iter={MAX_ITER_DIM2} (all algos)")
    print("=" * 65)

    results = run_dim2_experiments(
        config,
        problems=args.problems or None,
        max_workers=args.workers,
    )
    print_summary(results)