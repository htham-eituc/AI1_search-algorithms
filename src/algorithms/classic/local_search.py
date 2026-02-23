"""
Local Search Algorithms on Graphs
====================================
- Pure Python only (no high-level libraries)
- Modular, well-documented, Python best practices
- Configurable parameters (max_iter, restarts, step size, etc.)
- Supports both discrete (graph coloring) and continuous (function optimization)

Algorithms:
    1. Hill Climbing — Steepest Ascent (discrete: Graph Coloring)
    2. Hill Climbing — Steepest Ascent (continuous: function minimization)
"""

import random


# ══════════════════════════════════════════════════════════════════════════════
#  1a. Hill Climbing — Steepest Ascent (Discrete: Graph Coloring)
# ══════════════════════════════════════════════════════════════════════════════

def hill_climbing_coloring(
    graph: dict,
    n_colors: int,
    max_iter: int = 1000,
    max_restarts: int = 50,
    seed: int = None,
) -> dict:
    """
    Hill Climbing (Steepest Ascent) for Graph Coloring.

    Goal: assign one of n_colors colors to each node so that no two
    adjacent nodes share the same color (minimize number of conflicts).

    At each step, ALL neighbors of the current state are evaluated;
    the move that reduces conflicts most is taken (steepest ascent).
    Random restarts escape local optima.

    Parameters
    ----------
    graph : dict
        Adjacency list {node: [neighbor, ...]}.
    n_colors : int
        Number of colors available.
    max_iter : int
        Maximum iterations per restart.
    max_restarts : int
        Number of random restarts allowed.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        'coloring'   : dict {node: color}  – best coloring found
        'conflicts'  : int   – number of conflicting edges (0 = valid coloring)
        'solved'     : bool  – True if a valid coloring was found
        'iterations' : int   – total iterations performed
        'history'    : list  – conflict count per iteration
    """
    rng   = random.Random(seed)
    nodes = list(graph.keys())

    def count_conflicts(coloring: dict) -> int:
        """Count edges where both endpoints share the same color."""
        total = 0
        for u, neighbors in graph.items():
            for v in neighbors:
                if coloring[u] == coloring[v]:
                    total += 1
        return total // 2                      # each edge counted twice

    def node_conflicts(node, coloring: dict) -> int:
        """Number of neighbors sharing node's color."""
        return sum(1 for v in graph[node] if coloring[v] == coloring[node])

    best_coloring  = None
    best_conflicts = float('inf')
    total_iters    = 0
    history        = []

    for _ in range(max_restarts):
        # ── Random initialisation ────────────────────────────────────────────
        coloring = {n: rng.randint(0, n_colors - 1) for n in nodes}
        conflicts = count_conflicts(coloring)

        for it in range(max_iter):
            total_iters += 1
            history.append(conflicts)

            if conflicts == 0:                 # valid coloring found
                break

            # ── Steepest ascent: evaluate ALL single-node recolorings ─────────
            best_delta  = 0                    # improvement (reduction in conflicts)
            best_moves  = []                   # (node, new_color) with best_delta

            for node in nodes:
                old_color     = coloring[node]
                old_node_conf = node_conflicts(node, coloring)
                for c in range(n_colors):
                    if c == old_color:
                        continue
                    coloring[node] = c
                    new_node_conf  = node_conflicts(node, coloring)
                    delta = old_node_conf - new_node_conf   # positive = improvement
                    if delta > best_delta:
                        best_delta = delta
                        best_moves = [(node, c)]
                    elif delta == best_delta and delta >= 0:
                        best_moves.append((node, c))
                coloring[node] = old_color     # restore

            if not best_moves or best_delta <= 0:
                break                          # local optimum — trigger restart

            # Apply one best move (random tie-break)
            node, color    = rng.choice(best_moves)
            coloring[node] = color
            conflicts      = count_conflicts(coloring)

        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_coloring  = coloring.copy()

        if best_conflicts == 0:
            break                              # global solution found

    return {
        'coloring'  : best_coloring,
        'conflicts' : best_conflicts,
        'solved'    : best_conflicts == 0,
        'iterations': total_iters,
        'history'   : history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1b. Hill Climbing — Steepest Ascent (Continuous: function minimization)
# ══════════════════════════════════════════════════════════════════════════════

def hill_climbing_continuous(
    obj_func,
    dim: int,
    bounds: tuple[float, float],
    step_size: float = 0.1,
    max_iter: int = 1000,
    max_restarts: int = 20,
    n_neighbors: int = 8,
    step_decay: float = 0.995,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    Hill Climbing (Steepest Ascent) for continuous function minimization.

    At each step, n_neighbors candidate solutions are generated by
    perturbing the current position; the best one (steepest descent)
    is accepted only if it improves the objective.
    Random restarts and step size decay help escape shallow local optima.

    Parameters
    ----------
    obj_func : callable
        f(x) -> float to minimize. x is a list/array of length dim.
    dim : int
        Number of decision variables.
    bounds : tuple of (float, float)
        (lower_bound, upper_bound) for all dimensions.
    step_size : float
        Initial perturbation magnitude.
    max_iter : int
        Maximum iterations per restart.
    max_restarts : int
        Number of random restarts.
    n_neighbors : int
        Number of neighbors sampled per iteration (steepest ascent).
    step_decay : float
        Multiplicative decay applied to step_size each iteration.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print best fitness every 200 iterations.

    Returns
    -------
    dict
        'best_solution' : list[float]
        'best_fitness'  : float
        'history'       : list[float] – best fitness per iteration
        'iterations'    : int
    """
    rng    = random.Random(seed)
    lb, ub = bounds

    def clip(x):
        return [max(lb, min(ub, xi)) for xi in x]

    def random_point():
        return [rng.uniform(lb, ub) for _ in range(dim)]

    def perturb(x, step):
        return clip([xi + rng.gauss(0, step) for xi in x])

    best_solution = random_point()
    best_fitness  = obj_func(best_solution)
    history       = []
    total_iters   = 0

    for restart in range(max_restarts):
        current     = random_point()
        current_fit = obj_func(current)
        step        = step_size

        for it in range(max_iter):
            total_iters += 1
            step        *= step_decay

            # ── Steepest ascent: sample n_neighbors, pick the best ────────────
            neighbors    = [perturb(current, step) for _ in range(n_neighbors)]
            neighbor_fit = [obj_func(n) for n in neighbors]
            best_nb_idx  = min(range(n_neighbors), key=lambda i: neighbor_fit[i])

            if neighbor_fit[best_nb_idx] < current_fit:
                current     = neighbors[best_nb_idx]
                current_fit = neighbor_fit[best_nb_idx]

            history.append(current_fit)

            if current_fit < best_fitness:
                best_fitness  = current_fit
                best_solution = current[:]

            if verbose and total_iters % 200 == 0:
                print(f"  [HC] iter {total_iters:5d} | restart {restart+1} "
                      f"| best fitness: {best_fitness:.6e}")

    return {
        'best_solution': best_solution,
        'best_fitness' : best_fitness,
        'history'      : history,
        'iterations'   : total_iters,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Demo 1: Graph Coloring on Petersen graph ──────────────────────────────
    #
    #   The Petersen graph: 10 nodes, chromatic number = 3
    #   Outer pentagon: 0-1-2-3-4-0
    #   Inner pentagram: 5-7-9-6-8-5
    #   Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    #
    petersen = {
        0: [1, 4, 5],  1: [0, 2, 6],  2: [1, 3, 7],
        3: [2, 4, 8],  4: [3, 0, 9],  5: [0, 7, 8],
        6: [1, 8, 9],  7: [2, 5, 9],  8: [3, 5, 6],
        9: [4, 6, 7],
    }

    print("=" * 60)
    print("  HC (Discrete) — Graph Coloring on Petersen Graph")
    print("  Chromatic number = 3  (needs at least 3 colors)")
    print("=" * 60)

    for n_colors in [2, 3, 4]:
        r = hill_climbing_coloring(
            petersen, n_colors=n_colors,
            max_iter=500, max_restarts=30, seed=42
        )
        status = "✓ VALID" if r['solved'] else f"✗ {r['conflicts']} conflict(s)"
        print(f"\n  Colors = {n_colors} → {status}")
        if r['solved']:
            by_color = {}
            for node, c in sorted(r['coloring'].items()):
                by_color.setdefault(c, []).append(node)
            for c, nodes in sorted(by_color.items()):
                print(f"    Color {c}: nodes {nodes}")

    # ── Demo 2: Continuous — Sphere function ─────────────────────────────────
    def sphere(x):
        return sum(xi**2 for xi in x)

    print("\n" + "=" * 60)
    print("  HC (Continuous) — Sphere Function, dim=10")
    print("  Global minimum: f(0,...,0) = 0")
    print("=" * 60)
    r2 = hill_climbing_continuous(
        sphere, dim=10, bounds=(-5.12, 5.12),
        step_size=0.5, max_iter=500, max_restarts=20,
        n_neighbors=8, seed=42, verbose=True
    )
    print(f"\n  Best fitness : {r2['best_fitness']:.6e}")
    print(f"  Best solution: {[round(x, 5) for x in r2['best_solution']]}")
    print(f"  Total iters  : {r2['iterations']}")