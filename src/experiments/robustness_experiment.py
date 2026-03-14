"""
robustness_experiment.py
========================
Runs 30 independent trials per (algorithm, dimension) for Rosenbrock.

FAIRNESS
--------
All algorithms run max_iter = MAX_ITER_STANDARD = 1000 iterations.
SA and HC have their decay parameters adjusted by fairness.py so their
schedules remain meaningful over 1000 steps. All convergence curves are
length 1000 → directly comparable on the same x-axis.

Outputs
-------
  results/robustness_rosenbrock.csv   (summary: mean/std/min/max per algo×dim)
  results/robustness_rosenbrock.pkl   (raw runs including convergence curves)

Rosenbrock algorithms:
  Classic : HC, SA
  Evo     : GA, DE
  Biology : ABC, PSO
  Human   : SFO, TLBO
"""

import os
import sys
import csv
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

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

ROSENBROCK_ALGOS = ["HC", "SA", "GA", "DE", "ABC", "PSO", "SFO", "TLBO"]
N_RUNS           = 30
PROBLEM_NAME     = "rosenbrock"
RESULTS_DIR      = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Single-run worker
# ─────────────────────────────────────────────────────────────────────────────

def _single_run(args):
    algo_name, problem_name, dimensions, config, seed = args
    np.random.seed(seed)

    try:
        objective_func = PROBLEM_FUNCTIONS[problem_name]
        bounds_range   = get_problem_bounds(problem_name, config)
        bounds         = np.array([bounds_range] * dimensions)

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

        elif algo_name == "ABC":
            algo = algo_class(
                objective_func=objective_func, bounds=bounds, dim=dimensions,
                pop_size=pop, max_iter=max_iter,
                limit=p.get("limit", None))

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

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        result = algo.solve()

        return {
            "algorithm"        : algo_name,
            "dimensions"       : dimensions,
            "run"              : seed,
            "best_fitness"     : result["best_fitness"],
            "execution_time"   : result["execution_time_seconds"],
            "convergence_curve": result.get("convergence_curve"),
            "max_iter"         : max_iter,
            "evals_per_iter"   : evals_per_iter(algo_name, dimensions, pop),
            "nfe"              : evals_per_iter(algo_name, dimensions, pop) * max_iter,
        }

    except Exception as exc:
        print(f"  [ERROR] {algo_name} dim={dimensions} seed={seed}: {exc}", flush=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness(config, n_runs=N_RUNS, max_workers=None):
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    problem_config  = config["problems"][PROBLEM_NAME]
    dimensions_list = problem_config.get("dimensions",
                                         config["experiment"]["dimensions"])

    # Show fairness table before starting
    print_fairness_table(ROSENBROCK_ALGOS, dim=dimensions_list[0],
                         max_iter=MAX_ITER_STANDARD)

    tasks = []
    for algo in ROSENBROCK_ALGOS:
        if algo not in config["algorithms"]:
            print(f"[WARN] {algo} not in config, skipping.")
            continue
        for dim in dimensions_list:
            for run_idx in range(n_runs):
                # Deterministic but unique seed per (algo, dim, run)
                seed = run_idx * 1000 + abs(hash(algo + str(dim))) % 1000
                tasks.append((algo, PROBLEM_NAME, dim, config, seed))

    total = len(tasks)
    print(f"Robustness: {total} tasks  "
          f"({len(ROSENBROCK_ALGOS)} algos × {len(dimensions_list)} dims "
          f"× {n_runs} runs)  workers={max_workers}\n")

    raw_results = []
    completed   = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_single_run, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                raw_results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}", flush=True)

    # ── Aggregate ─────────────────────────────────────────────────────
    buckets = defaultdict(list)
    for r in raw_results:
        buckets[(r["algorithm"], r["dimensions"])].append(r)

    summary = []
    for (algo, dim), runs in sorted(buckets.items()):
        fitnesses = np.array([r["best_fitness"] for r in runs])
        times     = np.array([r["execution_time"] for r in runs])
        summary.append({
            "algorithm"     : algo,
            "problem"       : PROBLEM_NAME,
            "dimensions"    : dim,
            "n_runs"        : len(runs),
            "max_iter"      : MAX_ITER_STANDARD,
            "nfe"           : runs[0]["nfe"] if runs else 0,
            "mean_fitness"  : float(np.mean(fitnesses)),
            "std_fitness"   : float(np.std(fitnesses)),
            "min_fitness"   : float(np.min(fitnesses)),
            "max_fitness"   : float(np.max(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "mean_time"     : float(np.mean(times)),
        })

    return raw_results, summary


# ─────────────────────────────────────────────────────────────────────────────
# Save / print
# ─────────────────────────────────────────────────────────────────────────────

def save_results(raw_results, summary):
    csv_path   = RESULTS_DIR / "robustness_rosenbrock.csv"
    fieldnames = ["algorithm", "problem", "dimensions", "n_runs", "max_iter", "nfe",
                  "mean_fitness", "std_fitness", "min_fitness",
                  "max_fitness", "median_fitness", "mean_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary CSV → {csv_path}")

    pkl_path = RESULTS_DIR / "robustness_rosenbrock.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"raw": raw_results, "summary": summary}, f)
    print(f"Raw PKL     → {pkl_path}")


def print_robustness_table(summary):
    print(f"\n{'='*108}")
    print(f"ROBUSTNESS — Rosenbrock  (max_iter={MAX_ITER_STANDARD}, 30 runs)")
    print(f"{'='*108}")
    print(f"{'Algorithm':<10} {'Dims':<6} {'NFE':>8} {'Mean':>14} {'Std':>14} "
          f"{'Min':>14} {'Median':>14} {'Time(s)':>9}")
    print("─" * 108)
    for r in summary:
        print(f"{r['algorithm']:<10} {r['dimensions']:<6} {r['nfe']:>8} "
              f"{r['mean_fitness']:>14.4e} {r['std_fitness']:>14.4e} "
              f"{r['min_fitness']:>14.4e} {r['median_fitness']:>14.4e} "
              f"{r['mean_time']:>9.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = ROOT / "utils/config.json"
    config      = load_config(config_path)
    np.random.seed(config["experiment"]["seed"])

    print("=" * 72)
    print(f"ROBUSTNESS EXPERIMENT  |  max_iter={MAX_ITER_STANDARD} (all algorithms)")
    print("=" * 72)

    raw_results, summary = run_robustness(config)
    save_results(raw_results, summary)
    print_robustness_table(summary)