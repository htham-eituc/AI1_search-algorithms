"""
fairness.py
===========
Fairness configuration for algorithm comparison experiments.

PROBLEM
-------
Algorithms consume different evaluations per iteration:
  - Population algos (GA, DE, PSO, …): pop_size evals/iter  (e.g. 50)
  - SA:                                 1 eval/iter
  - HC:                                 2 * dim evals/iter

SOLUTION: Equal max_iter + parameter compensation
--------------------------------------------------
All algorithms run the SAME max_iter so convergence curves are the same
length and directly comparable on the same x-axis. SA and HC compensate
by having their decay/cooling parameters automatically adjusted so their
schedules remain meaningful over the shared iteration count:

  SA default was tuned for ~5000 iters (cooling_rate=0.95).
  At max_iter=1000 that would burn out in ~140 iters (0.95^140 ≈ 0.001).
  Fix: recompute cooling_rate = target_reduction^(1/max_iter).

  HC default was tuned for ~5000 iters (step_decay=0.995).
  At max_iter=1000 that would reduce step_size to only ~0.67x.
  Fix: recompute step_decay = target_reduction^(1/max_iter).

This is the standard approach in CEC benchmark competitions.
NFE differs between algorithms but is logged for transparency.

Shared iteration counts
-----------------------
  MAX_ITER_STANDARD = 1000  (convergence + robustness experiments)
  MAX_ITER_DIM2     =  500  (2D GIF experiments)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Shared max_iter  (identical for ALL algorithms in each experiment type)
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITER_STANDARD = 1000
MAX_ITER_DIM2     =  500

DEFAULT_POP_SIZE  =   50


# ─────────────────────────────────────────────────────────────────────────────
# SA cooling_rate adjustment
# ─────────────────────────────────────────────────────────────────────────────

def sa_cooling_rate(max_iter: int, t_reduction: float = 0.001) -> float:
    """
    Cooling rate so that T drops to t_reduction * T_initial over max_iter steps.

        T_final = T_initial * cooling_rate ^ max_iter
        cooling_rate = t_reduction ^ (1 / max_iter)

    Examples
    --------
    max_iter=1000 → cooling_rate ≈ 0.99311  (was 0.95 → burns out at ~140 iters)
    max_iter= 500 → cooling_rate ≈ 0.98623
    """
    return float(t_reduction ** (1.0 / max_iter))


# ─────────────────────────────────────────────────────────────────────────────
# HC step_decay adjustment
# ─────────────────────────────────────────────────────────────────────────────

def hc_step_decay(max_iter: int, step_reduction: float = 0.01) -> float:
    """
    Step decay so that step_size reduces to step_reduction fraction over max_iter.

        step_final = step_initial * step_decay ^ max_iter
        step_decay = step_reduction ^ (1 / max_iter)

    Examples
    --------
    max_iter=1000 → step_decay ≈ 0.99540  (was 0.995 → only reaches 0.67x)
    max_iter= 500 → step_decay ≈ 0.99081
    """
    return float(step_reduction ** (1.0 / max_iter))


# ─────────────────────────────────────────────────────────────────────────────
# Evals per iteration (for NFE logging only — NOT used to set max_iter)
# ─────────────────────────────────────────────────────────────────────────────

def evals_per_iter(algo_name: str, dim: int, pop_size: int) -> int:
    """Number of objective function evaluations consumed per iteration."""
    if algo_name == "SA":
        return 1
    if algo_name == "HC":
        return max(1, 2 * dim)
    return max(1, pop_size)


# ─────────────────────────────────────────────────────────────────────────────
# Main params builder
# ─────────────────────────────────────────────────────────────────────────────

def build_fair_params(
    algo_name: str,
    problem_name: str,
    dim: int,
    config: dict,
    max_iter: int = MAX_ITER_STANDARD,
) -> dict:
    """
    Return algorithm params dict with:
      - max_iterations = max_iter  (same value for every algorithm)
      - SA: cooling_rate adjusted to be meaningful over max_iter steps
      - HC: step_decay  adjusted to be meaningful over max_iter steps
      - All other params from config defaults + problem-specific overrides

    Also injects read-only metadata keys (prefixed _) for logging:
      _max_iter, _evals_per_iter, _nfe
    These must be stripped before passing to algorithm __init__.
    """
    from utils.configHelper import get_algorithm_params

    params   = get_algorithm_params(algo_name, problem_name, dim, config)
    pop_size = params.get("population_size", DEFAULT_POP_SIZE)

    # Enforce shared iteration count
    params["max_iterations"] = max_iter

    # Parameter compensation for non-population algorithms
    if algo_name == "SA":
        params["cooling_rate"] = sa_cooling_rate(max_iter)

    if algo_name == "HC":
        params["step_decay"] = hc_step_decay(max_iter)

    # Metadata (strip these before passing to __init__)
    epi = evals_per_iter(algo_name, dim, pop_size)
    params["_max_iter"]       = max_iter
    params["_evals_per_iter"] = epi
    params["_nfe"]            = epi * max_iter

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

def print_fairness_table(
    algos: list,
    dim: int,
    max_iter: int = MAX_ITER_STANDARD,
    pop_size: int = DEFAULT_POP_SIZE,
) -> None:
    """Print the shared max_iter and actual NFE per algorithm."""
    print(f"\n{'─'*72}")
    print(f"Fairness  |  max_iter={max_iter} (shared)  dim={dim}  pop={pop_size}")
    print(f"{'─'*72}")
    print(f"{'Algorithm':<12} {'Evals/iter':>11} {'max_iter':>10} "
          f"{'Total NFE':>11}  Adjustment")
    print(f"{'─'*72}")
    for algo in algos:
        epi   = evals_per_iter(algo, dim, pop_size)
        total = epi * max_iter
        note  = ""
        if algo == "SA":
            note = f"cooling_rate → {sa_cooling_rate(max_iter):.5f}"
        elif algo == "HC":
            note = f"step_decay   → {hc_step_decay(max_iter):.5f}"
        print(f"{algo:<12} {epi:>11,} {max_iter:>10,} {total:>11,}  {note}")
    print(f"{'─'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ALL = ["HC", "SA", "GA", "DE", "GSA", "ABC", "FA", "CS", "PSO", "TLBO", "SFO", "CA"]
    print_fairness_table(ALL, dim=30,  max_iter=MAX_ITER_STANDARD)
    print_fairness_table(ALL, dim=100, max_iter=MAX_ITER_STANDARD)
    print_fairness_table(ALL, dim=2,   max_iter=MAX_ITER_DIM2)