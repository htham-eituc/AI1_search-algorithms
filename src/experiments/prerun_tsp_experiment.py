"""
tsp_experiments.py  (v5)
========================
Three test suites: sparse / clustered / euclidean

Timeout — ACTUALLY kills the subprocess
----------------------------------------
Previous versions used ProcessPoolExecutor + future.result(timeout=N).
That only times out the *wait* in the main process; the worker subprocess
keeps running until it finishes naturally.

This version wraps every task in its own multiprocessing.Process.
On timeout the process is .terminate() + .kill()-ed, so the CPU is
freed immediately and the next task can start.

A semaphore limits how many processes run concurrently (= max_workers).

Status values
-------------
  "ok"      finished, finite best_fitness
  "inf"     finished but returned inf / NaN
  "timeout" process killed after TIMEOUT_SECONDS
  "error"   unhandled exception inside worker

Usage
-----
    python tsp_experiments.py
    python tsp_experiments.py --case sparse
    python tsp_experiments.py --iter 200 --timeout 120 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import pickle
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems.discrete import TSPProblem
from algorithms.human.ca                import CA_TSP
from algorithms.evolution.GA            import GA_TSP
from algorithms.physics.SA              import SA_TSP
from algorithms.biology.ACOR            import ACO_TSP
from algorithms.classic.local_search    import HillClimbing_TSP
from algorithms.classic.informed_search import A_STAR_TSP


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

TIMEOUT_SECONDS: int = 60
MAX_ITER:    int   = 100
POP_SIZE:    int   = 30
HC_RESTARTS: int   = MAX_ITER
SA_INIT_TEMP:    float = 1000.0
SA_COOLING_RATE: float = 0.99
ACO_ALPHA: float = 1.0
ACO_BETA:  float = 2.0
ACO_RHO:   float = 0.1
ACO_Q:     float = 100.0
SEED: int = 42

ALGORITHMS: list[str] = [
    "CA_TSP",
    "GA_TSP",
    "SA_TSP",
    "ACO_TSP",
    "HillClimbing_TSP",
]

CASES: dict[str, str] = {
    "sparse":    "tests/TSP/sparse",
    "clustered": "tests/TSP/clustered",
    "euclidean": "tests/TSP/euclidean",
}

RESULTS_DIR = Path("results")


# ═════════════════════════════════════════════════════════════════════════════
# TASK
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    case:        str
    test_id:     str
    n_cities:    int
    algo_name:   str
    dist_matrix: np.ndarray
    max_iter:    int


# ═════════════════════════════════════════════════════════════════════════════
# WORKER  — runs inside a spawned subprocess, puts result onto a Queue
# ═════════════════════════════════════════════════════════════════════════════

def _worker_fn(task: Task, queue: mp.Queue) -> None:
    """
    Entry point for the subprocess.  Puts exactly one dict onto `queue`:
    either the real result or an error result.
    """
    import tracemalloc

    name  = task.algo_name
    dm    = task.dist_matrix
    n     = task.n_cities
    iters = task.max_iter

    def _err(msg: str) -> dict:
        return {
            "case": task.case, "test_id": task.test_id, "name": name,
            "status": "error",
            "best_fitness": None, "execution_time": 0.0, "wall_time": 0.0,
            "peak_memory_kb": None, "best_solution": None,
            "time_complexity": None, "space_complexity": None,
            "nodes_expanded": None, "convergence": None,
            "max_iter_used": iters, "error": msg,
        }

    try:
        if name == "CA_TSP":
            algo = CA_TSP(dm, pop_size=POP_SIZE, max_iter=iters, seed=SEED)
        elif name == "GA_TSP":
            algo = GA_TSP(dm, pop_size=POP_SIZE, max_iter=iters, seed=SEED)
        elif name == "SA_TSP":
            algo = SA_TSP(dm, initial_temperature=SA_INIT_TEMP,
                          cooling_rate=SA_COOLING_RATE,
                          max_iterations=iters, seed=SEED)
        elif name == "ACO_TSP":
            algo = ACO_TSP(dm, n_ants=n, max_iterations=iters,
                           alpha=ACO_ALPHA, beta=ACO_BETA,
                           rho=ACO_RHO, q=ACO_Q, seed=SEED)
        elif name == "HillClimbing_TSP":
            algo = HillClimbing_TSP(dm, max_restarts=HC_RESTARTS, seed=SEED)
        elif name == "A_STAR_TSP":
            algo = A_STAR_TSP(dm, seed=SEED)
        else:
            queue.put(_err(f"Unknown algorithm: {name}"))
            return

        tracemalloc.start()
        t0 = time.perf_counter()
        algo.solve()
        wall = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kb = peak_bytes / 1024

        if hasattr(algo, "get_results"):
            r = algo.get_results()
            bf   = r.get("best_fitness")
            et   = r.get("execution_time_seconds", wall)
            sol  = r.get("best_solution")
            tc   = r.get("time_complexity")
            sc   = r.get("space_complexity")
            ne   = r.get("nodes_expanded")
            cv   = r.get("convergence_curve")
        else:
            bf   = getattr(algo, "best_fitness",      None)
            et   = getattr(algo, "execution_time",    wall)
            sol  = getattr(algo, "best_solution",     None)
            tc   = None
            sc   = None
            ne   = getattr(algo, "nodes_expanded",    None)
            cv   = getattr(algo, "convergence_curve", None)

        # classify
        is_bad = (bf is None
                  or (isinstance(bf, float) and (bf != bf or bf == float("inf"))))
        status = "inf" if is_bad else "ok"

        queue.put({
            "case": task.case, "test_id": task.test_id, "name": name,
            "status":           status,
            "best_fitness":     bf,
            "execution_time":   et,
            "wall_time":        wall,
            "peak_memory_kb":   peak_kb,
            "best_solution":    sol,
            "time_complexity":  tc,
            "space_complexity": sc,
            "nodes_expanded":   ne,
            "convergence":      cv,
            "max_iter_used":    iters,
            "error":            "",
        })

    except Exception:
        queue.put(_err(traceback.format_exc()))


# ═════════════════════════════════════════════════════════════════════════════
# TIMED RUNNER  — spawns a Process, enforces hard wall-clock kill
# ═════════════════════════════════════════════════════════════════════════════

def _run_with_timeout(task: Task, timeout: int) -> dict[str, Any]:
    """
    Spawn task in a fresh Process.  If it doesn't finish within `timeout`
    seconds, SIGTERM then SIGKILL it and return a timeout result.
    The process is guaranteed to be dead before this function returns.
    """
    queue: mp.Queue = mp.Queue()
    proc  = mp.Process(target=_worker_fn, args=(task, queue), daemon=True)
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        # hard kill — two-stage for robustness on Windows and Unix
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()

        return {
            "case": task.case, "test_id": task.test_id, "name": task.algo_name,
            "status":           "timeout",
            "best_fitness":     None,
            "execution_time":   float(timeout),
            "wall_time":        float(timeout),
            "peak_memory_kb":   None,
            "best_solution":    None,
            "time_complexity":  None,
            "space_complexity": None,
            "nodes_expanded":   None,
            "convergence":      None,
            "max_iter_used":    task.max_iter,
            "error":            f"Killed after {timeout}s",
        }

    # process finished in time — retrieve result from queue
    try:
        return queue.get_nowait()
    except Exception:
        return {
            "case": task.case, "test_id": task.test_id, "name": task.algo_name,
            "status":           "error",
            "best_fitness":     None,
            "execution_time":   0.0,
            "wall_time":        0.0,
            "peak_memory_kb":   None,
            "best_solution":    None,
            "time_complexity":  None,
            "space_complexity": None,
            "nodes_expanded":   None,
            "convergence":      None,
            "max_iter_used":    task.max_iter,
            "error":            "Worker put nothing on queue",
        }


# ═════════════════════════════════════════════════════════════════════════════
# TASK BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_tasks(
    cases: dict[str, str],
    algorithms: list[str],
    max_iter: int,
) -> tuple[list[Task], dict]:
    tasks: list[Task] = []
    meta:  dict       = {}

    for case_name, case_dir in cases.items():
        case_path = Path(case_dir)
        if not case_path.exists():
            print(f"[WARN] Directory not found, skipping: {case_path}")
            continue

        test_files = sorted(
            case_path.glob("test_*.txt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not test_files:
            print(f"[WARN] No test files in {case_path}")
            continue

        meta[case_name] = {}
        for tf in test_files:
            test_id = tf.stem
            try:
                prob = TSPProblem(str(tf))
            except Exception as exc:
                print(f"[ERROR] Cannot load {tf}: {exc}")
                continue

            meta[case_name][test_id] = {
                "n_cities":    prob.n_cities,
                "coordinates": getattr(prob, "coordinates", None),
                "dist_matrix": prob.distance_matrix,
            }
            for algo in algorithms:
                tasks.append(Task(
                    case=case_name,
                    test_id=test_id,
                    n_cities=prob.n_cities,
                    algo_name=algo,
                    dist_matrix=prob.distance_matrix,
                    max_iter=max_iter,
                ))

    return tasks, meta


# ═════════════════════════════════════════════════════════════════════════════
# PARALLEL RUNNER  — semaphore-bounded pool of timed processes
# ═════════════════════════════════════════════════════════════════════════════

def run_experiments(
    cases:           dict[str, str] | None = None,
    algorithms:      list[str]      | None = None,
    max_workers:     int             | None = None,
    max_iter:        int                    = MAX_ITER,
    timeout_seconds: int                    = TIMEOUT_SECONDS,
) -> dict:
    """
    Run every (case × test × algorithm) task.

    Concurrency model
    -----------------
    Each task runs in its own Process.  A threading.Semaphore limits how many
    run at the same time (= max_workers).  This gives us true per-process
    kill on timeout while still keeping all cores busy.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if cases      is None: cases      = CASES
    if algorithms is None: algorithms = ALGORITHMS

    workers = max_workers or os.cpu_count() or 1

    print("Loading problem instances …")
    tasks, meta = build_tasks(cases, algorithms, max_iter)
    n_tasks  = len(tasks)
    n_done   = 0
    t_global = time.perf_counter()

    print(f"Submitting {n_tasks} tasks → {workers} parallel processes  "
          f"(hard timeout={timeout_seconds}s)\n")

    # seed results skeleton
    all_results: dict = {}
    for case_name, tests in meta.items():
        all_results[case_name] = {}
        for test_id, tdata in tests.items():
            all_results[case_name][test_id] = {
                "n_cities":    tdata["n_cities"],
                "coordinates": tdata["coordinates"],
                "dist_matrix": tdata["dist_matrix"],
                "algorithms":  {},
            }

    counts: dict[str, int] = defaultdict(int)

    # Use a ThreadPoolExecutor to drive the spawned processes concurrently.
    # Each thread owns one Process at a time and calls _run_with_timeout,
    # which blocks until the process either finishes or is killed.
    def _thread_fn(task: Task) -> dict:
        return _run_with_timeout(task, timeout_seconds)

    with ThreadPoolExecutor(max_workers=workers) as tex:
        future_map = {tex.submit(_thread_fn, t): t for t in tasks}

        for future in as_completed(future_map):
            try:
                res = future.result()
            except Exception:
                task = future_map[future]
                res  = {
                    "case": task.case, "test_id": task.test_id,
                    "name": task.algo_name, "status": "error",
                    "best_fitness": None, "execution_time": 0.0,
                    "wall_time": 0.0, "peak_memory_kb": None,
                    "best_solution": None, "time_complexity": None,
                    "space_complexity": None, "nodes_expanded": None,
                    "convergence": None, "max_iter_used": task.max_iter,
                    "error": traceback.format_exc(),
                }

            case  = res["case"]
            tid   = res["test_id"]
            aname = res["name"]
            st    = res["status"]

            all_results[case][tid]["algorithms"][aname] = res
            counts[st] += 1

            n_done  += 1
            elapsed  = time.perf_counter() - t_global
            eta      = (elapsed / n_done) * (n_tasks - n_done)

            tag_map = {"ok": " OK ", "inf": " INF", "timeout": "TIME", "error": "FAIL"}
            tag     = tag_map.get(st, " ?? ")

            if st == "ok":
                bf     = res["best_fitness"]
                detail = (f"dist={bf:.1f}  "
                          f"t={res['execution_time']:.3f}s  "
                          f"mem={res['peak_memory_kb']:.0f}KB")
            elif st == "timeout":
                detail = f"KILLED after {timeout_seconds}s"
            elif st == "inf":
                detail = f"returned inf/NaN  t={res['execution_time']:.3f}s"
            else:
                detail = (res.get("error") or "").splitlines()[-1][:55]

            print(f"[{tag}] {n_done:>3}/{n_tasks}  "
                  f"{case:10s} {tid:8s}  {aname:22s}  "
                  f"{detail}  ETA {eta:.0f}s")

    total = time.perf_counter() - t_global
    print(f"\nDone — {n_tasks} tasks in {total:.1f}s "
          f"({total/n_tasks:.2f}s avg)")
    print(f"  ok={counts['ok']}  inf={counts['inf']}  "
          f"timeout={counts['timeout']}  error={counts['error']}")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

def save_pickle(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVED] {path}  ({path.stat().st_size / 1024:.1f} KB)")


def save_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case", "test_id", "n_cities", "algorithm", "status", "max_iter_used",
        "best_fitness", "execution_time", "wall_time", "peak_memory_kb",
        "nodes_expanded", "time_complexity", "space_complexity",
        "has_convergence", "error",
    ]
    rows = []
    for case_name, tests in data.items():
        for test_id, tdata in tests.items():
            for aname, ares in tdata["algorithms"].items():
                rows.append({
                    "case":             case_name,
                    "test_id":          test_id,
                    "n_cities":         tdata["n_cities"],
                    "algorithm":        aname,
                    "status":           ares.get("status", ""),
                    "max_iter_used":    ares.get("max_iter_used"),
                    "best_fitness":     ares.get("best_fitness"),
                    "execution_time":   ares.get("execution_time"),
                    "wall_time":        ares.get("wall_time"),
                    "peak_memory_kb":   ares.get("peak_memory_kb"),
                    "nodes_expanded":   ares.get("nodes_expanded"),
                    "time_complexity":  ares.get("time_complexity"),
                    "space_complexity": ares.get("space_complexity"),
                    "has_convergence":  ares.get("convergence") is not None,
                    "error":            ares.get("error", ""),
                })
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {path}  ({len(rows)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSP experiments — sparse / clustered / euclidean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case",
                        choices=["sparse", "clustered", "euclidean", "all"],
                        default="all")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel processes (default: cpu count)")
    parser.add_argument("--iter",    type=int, default=MAX_ITER, dest="max_iter",
                        help="Unified max iterations for CA/GA/SA/ACO")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS,
                        help="Hard wall-clock kill limit per task (seconds)")
    args = parser.parse_args()

    cases_to_run = (
        CASES if args.case == "all"
        else {args.case: CASES[args.case]}
    )

    print("=" * 70)
    print("TSP Experiments  —  sparse / clustered / euclidean")
    print(f"  Cases      : {list(cases_to_run.keys())}")
    print(f"  MAX_ITER   : {args.max_iter}")
    print(f"  TIMEOUT    : {args.timeout}s  (process is killed if exceeded)")
    print(f"  POP_SIZE   : {POP_SIZE}")
    print(f"  Workers    : {args.workers or os.cpu_count()}")
    print("=" * 70 + "\n")

    results = run_experiments(
        cases=cases_to_run,
        max_workers=args.workers,
        max_iter=args.max_iter,
        timeout_seconds=args.timeout,
    )

    save_pickle(results, RESULTS_DIR / "tsp_results.pkl")
    save_csv(results,    RESULTS_DIR / "tsp_summary.csv")

    print("\n" + "=" * 70)
    print("LEADERBOARD  (ok results only — mean dist across all instances)")
    print("=" * 70)
    for case_name, tests in results.items():
        print(f"\n  [{case_name.upper()}]")
        sums: dict[str, list[float]] = defaultdict(list)
        fail: dict[str, int]         = defaultdict(int)
        for tdata in tests.values():
            for aname, ares in tdata["algorithms"].items():
                if ares.get("status") == "ok":
                    sums[aname].append(ares["best_fitness"])
                else:
                    fail[aname] += 1
        all_algos = set(list(sums) + list(fail))
        ranked = sorted(
            all_algos,
            key=lambda a: (sum(sums[a]) / len(sums[a])) if sums[a] else float("inf"),
        )
        for rank, aname in enumerate(ranked, 1):
            avg   = (sum(sums[aname]) / len(sums[aname])) if sums[aname] else None
            avg_s = f"{avg:.1f}" if avg else "  —  "
            f     = fail[aname]
            print(f"    {rank}. {aname:22s}  avg={avg_s}  "
                  f"failed={f}/{len(tests)}")


# Required on Windows for multiprocessing
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()