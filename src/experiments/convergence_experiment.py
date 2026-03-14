"""
convergence_experiment.py
=========================
Runs ONE trial per (algorithm, dimension) for all three continuous problems.

FAIRNESS
--------
All algorithms run max_iter = MAX_ITER_STANDARD = 1000 iterations.
  - GA, DE, PSO, … run 1000 iterations as-is.
  - SA cooling_rate is recomputed so the temperature schedule spans all
    1000 steps meaningfully (not burned out in ~140 iters as with 0.95).
  - HC step_decay is recomputed so the step_size decays gracefully over
    1000 steps (not barely reduced as with 0.995).

Convergence curves are ALL length 1000 → directly comparable on same x-axis.
Actual NFE differs (logged in each result for transparency).

Output files
------------
  results/convergence_sphere.pkl
  results/convergence_rastrigin.pkl
  results/convergence_rosenbrock.pkl

PKL structure:
  { "<algo>": { <dim>: { "algorithm", "dimensions", "best_fitness",
                          "convergence_curve"     (len=1000),
                          "average_fitness_curve" (len=1000),
                          "diversity_curve"       (len=1000),
                          "execution_time_seconds",
                          "max_iter", "evals_per_iter", "nfe" } } }

Usage
-----
  python experiments/convergence_experiment.py
  python experiments/convergence_experiment.py sphere rosenbrock --workers 6
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.configHelper import (
    load_config,
    ALGORITHM_CLASSES,
    PROBLEM_FUNCTIONS,
    get_problem_bounds,
)
from utils.fairness import (
    build_fair_params,
    evals_per_iter,
    print_fairness_table,
    MAX_ITER_STANDARD,
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBLEM_ALGOS = {
    "sphere"    : ["HC", "SA", "GA", "DE", "GSA", "PSO", "TLBO"],
    "rastrigin" : ["HC", "SA", "GA", "DE", "ABC", "FA",  "CS",  "CA"],
    "rosenbrock": ["HC", "SA", "GA", "DE", "ABC", "PSO", "SFO", "TLBO"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker(args):
    algo_name, problem_name, dimensions, config, seed = args
    np.random.seed(seed)

    try:
        objective_func = PROBLEM_FUNCTIONS[problem_name]
        bounds_range   = get_problem_bounds(problem_name, config)
        bounds         = np.array([bounds_range] * dimensions)

        # build_fair_params sets max_iterations=MAX_ITER_STANDARD,
        # adjusts SA cooling_rate and HC step_decay for this iter count
        p        = build_fair_params(algo_name, problem_name,
                                     dimensions, config, MAX_ITER_STANDARD)
        pop      = p.get("population_size", 50)
        max_iter = p["max_iterations"]      # always MAX_ITER_STANDARD

        algo_class = ALGORITHM_CLASSES[algo_name]

        if algo_name == "DE":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                F=p.get("mutation_factor", 0.8),
                CR=p.get("crossover_rate", 0.9))

        elif algo_name == "GA":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                crossover_rate=p.get("crossover_rate", 0.8),
                mutation_rate=p.get("mutation_rate", 0.1),
                mutation_scale=p.get("mutation_scale", 0.1),
                tournament_size=p.get("tournament_size", 3))

        elif algo_name == "SA":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=p.get("population_size", 10),
                max_iter=max_iter,
                initial_temperature=p.get("initial_temperature", 1000.0),
                cooling_rate=p["cooling_rate"],      # ← adjusted by fairness.py
                cooling_schedule=p.get("cooling_schedule", "geometric"))

        elif algo_name == "GSA":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                G0=p.get("gravitational_constant", 100.0),
                kbest_initial=p.get("alpha", 20))

        elif algo_name == "ABC":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                limit=p.get("limit", None))

        elif algo_name == "CS":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                pa=p.get("pa", 0.25),
                alpha=p.get("alpha", 0.01),
                lambda_levy=p.get("lambda_levy", 1.5))

        elif algo_name == "FA":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                beta0=p.get("beta0", 1.0),
                gamma=p.get("gamma", 1.0),
                alpha=p.get("alpha", 0.5),
                alpha_decay=p.get("alpha_decay", 0.97))

        elif algo_name == "PSO":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                w_max=p.get("w_max", 0.9),
                w_min=p.get("w_min", 0.4),
                c1=p.get("c1", 2.0),
                c2=p.get("c2", 2.0),
                v_clamp_ratio=p.get("v_clamp_ratio", 0.2))

        elif algo_name == "HC":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=p.get("population_size", 10),
                max_iter=max_iter,
                step_size=p.get("step_size", 0.5),
                step_decay=p["step_decay"],          # ← adjusted by fairness.py
                max_restarts=p.get("max_restarts", 10),
                patience=p.get("patience", 30))

        elif algo_name == "TLBO":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter)

        elif algo_name == "SFO":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                desired_speed=p.get("desired_speed", 0.8),
                tau=p.get("tau", 0.5),
                A=p.get("A", 2.0),
                B=p.get("B", 0.3),
                r_agent=p.get("r_agent", 0.5),
                dt=p.get("dt", 0.1),
                v_max=p.get("v_max", 2.0))

        elif algo_name == "CA":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                alpha=p.get("alpha", 0.2))

        else:
            raise ValueError(f"No factory entry for '{algo_name}'")

        result = algo.solve()
        result["dimensions"]     = dimensions
        result["max_iter"]       = max_iter
        result["evals_per_iter"] = evals_per_iter(algo_name, dimensions, pop)
        result["nfe"]            = result["evals_per_iter"] * max_iter
        return algo_name, dimensions, result

    except Exception as exc:
        import traceback
        print(f"  [ERROR] {algo_name} {problem_name} {dimensions}D: {exc}", flush=True)
        traceback.print_exc()
        return algo_name, dimensions, None


# ─────────────────────────────────────────────────────────────────────────────
# Per-problem runner
# ─────────────────────────────────────────────────────────────────────────────

def run_convergence_experiment(problem_name, config, max_workers=None):
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    algos           = PROBLEM_ALGOS[problem_name]
    problem_config  = config["problems"][problem_name]
    dimensions_list = problem_config.get("dimensions",
                                         config["experiment"]["dimensions"])
    seed_base       = config["experiment"]["seed"]

    print_fairness_table(algos, dim=dimensions_list[0], max_iter=MAX_ITER_STANDARD)

    tasks = []
    for algo in algos:
        if algo not in config["algorithms"]:
            print(f"[WARN] {algo} not in config — skipping")
            continue
        for dim in dimensions_list:
            seed = seed_base + abs(hash(algo + str(dim))) % 10000
            tasks.append((algo, problem_name, dim, config, seed))

    print(f"[{problem_name.upper()}]  {len(tasks)} tasks  workers={max_workers}")

    results   = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_worker, t) for t in tasks]
        for future in as_completed(futures):
            algo_name, dimensions, result = future.result()
            if result is not None:
                results.setdefault(algo_name, {})[dimensions] = result
            completed += 1
            status = "✓" if result else "✗"
            print(f"  {status} {completed}/{len(tasks)}  "
                  f"{algo_name} {dimensions}D", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved → {path}  ({Path(path).stat().st_size / 1024:.1f} KB)")


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_convergence_summary(results, problem_name):
    print(f"\n{'='*100}")
    print(f"CONVERGENCE — {problem_name.upper()}  "
          f"(max_iter={MAX_ITER_STANDARD} for all algorithms)")
    print(f"{'='*100}")
    print(f"{'Algorithm':<10} {'Dims':<6} {'Best Fitness':>16} "
          f"{'Final AvgFit':>16} {'NFE':>9} {'Time(s)':>9}")
    print("─" * 100)
    for algo, dim_data in sorted(results.items()):
        for dim, res in sorted(dim_data.items()):
            avg = (res["average_fitness_curve"][-1]
                   if res.get("average_fitness_curve") is not None else float("nan"))
            print(f"{algo:<10} {dim:<6} {res['best_fitness']:>16.4e} "
                  f"{avg:>16.4e} {res.get('nfe', '?'):>9} "
                  f"{res['execution_time_seconds']:>9.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", nargs="*",
                        choices=["sphere", "rastrigin", "rosenbrock"],
                        help="Problems to run (default: all)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    problems_to_run = args.problems or ["sphere", "rastrigin", "rosenbrock"]

    config_path = ROOT / "utils/config.json"
    config      = load_config(config_path)
    np.random.seed(config["experiment"]["seed"])

    print("=" * 72)
    print(f"CONVERGENCE EXPERIMENT  |  max_iter={MAX_ITER_STANDARD} (all algorithms)")
    print("=" * 72)

    for problem in problems_to_run:
        results  = run_convergence_experiment(problem, config,
                                              max_workers=args.workers)
        pkl_path = RESULTS_DIR / f"convergence_{problem}.pkl"
        save_pkl(results, pkl_path)
        print_convergence_summary(results, problem)