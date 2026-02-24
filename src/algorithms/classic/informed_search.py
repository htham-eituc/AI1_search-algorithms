"""
Informed Search Algorithms
============================
GreedyBestFirst — expand node with lowest h(n) only
AStarSearch     — expand node with lowest f(n) = g(n) + h(n)

All classes inherit BaseGraphSearch from algorithms/base.py.
Default heuristic: Manhattan distance (admissible for 4-directional grids).
"""

import time
import heapq
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from algorithms.base import BaseGraphSearch


# ── Default heuristic ─────────────────────────────────────────────────────────

def _manhattan(node, goal) -> float:
    """Manhattan distance — admissible heuristic for 4-directional grid."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def _reconstruct_path(parent: dict, start, goal) -> list:
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ══════════════════════════════════════════════════════════════════════════════
#  1. Greedy Best-First Search
# ══════════════════════════════════════════════════════════════════════════════

class GreedyBestFirst(BaseGraphSearch):
    """
    Greedy Best-First Search — always expand the node with the lowest h(n).

    Uses only the heuristic estimate to the goal; ignores path cost g(n).
    Very fast but NOT guaranteed to find the optimal path.
    Default heuristic: Manhattan distance.

    Parameters
    ----------
    heuristic : callable, optional
        heuristic(node, goal) -> float.
    """

    def __init__(self, grid, start_node, end_node, heuristic=None):
        super().__init__("GreedyBestFirst", grid, start_node, end_node)
        self.heuristic = heuristic if heuristic is not None else _manhattan

    def solve(self):
        t0    = time.time()
        start = self.start_node
        goal  = self.end_node

        if start == goal:
            self.best_solution  = [start]
            self.best_fitness   = 0.0
            self.execution_time = time.time() - t0
            return self

        heap    = [(self.heuristic(start, goal), start)]
        parent  = {start: None}
        g_cost  = {start: 0.0}
        visited = set()

        while heap:
            _, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            self.explored_path.append(node)
            self.nodes_expanded += 1

            if node == goal:
                path                = _reconstruct_path(parent, start, goal)
                self.best_solution  = path
                self.best_fitness   = g_cost[goal]
                self.execution_time = time.time() - t0
                return self

            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    new_g = g_cost[node] + 1.0
                    if neighbor not in g_cost or new_g < g_cost[neighbor]:
                        g_cost[neighbor] = new_g
                        parent[neighbor] = node
                        heapq.heappush(heap,
                            (self.heuristic(neighbor, goal), neighbor))

        self.best_solution  = []
        self.best_fitness   = float('inf')
        self.execution_time = time.time() - t0
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  2. A* Search
# ══════════════════════════════════════════════════════════════════════════════

class AStarSearch(BaseGraphSearch):
    """
    A* Search — optimal informed search using f(n) = g(n) + h(n).

    Expands the node with the lowest f(n) = actual cost g(n) + heuristic h(n).
    Guaranteed optimal when heuristic is admissible (never overestimates).
    Default heuristic: Manhattan distance (admissible for 4-directional grids).

    Extra attribute
    ---------------
    f_values : dict {node: f(n)} — recorded for every expanded node,
               useful for visualization of the search frontier.

    Parameters
    ----------
    heuristic : callable, optional
        heuristic(node, goal) -> float.
    """

    def __init__(self, grid, start_node, end_node, heuristic=None):
        super().__init__("AStarSearch", grid, start_node, end_node)
        self.heuristic = heuristic if heuristic is not None else _manhattan
        self.f_values  = {}           # exposed for visualization

    def solve(self):
        t0    = time.time()
        start = self.start_node
        goal  = self.end_node

        if start == goal:
            self.best_solution  = [start]
            self.best_fitness   = 0.0
            self.f_values       = {start: 0.0}
            self.execution_time = time.time() - t0
            return self

        g      = {start: 0.0}
        parent = {start: None}
        heap   = [(self.heuristic(start, goal), start)]

        while heap:
            f_val, node = heapq.heappop(heap)

            if node in self.f_values:       # already expanded optimally
                continue
            self.f_values[node] = f_val
            self.explored_path.append(node)
            self.nodes_expanded += 1

            if node == goal:
                path                = _reconstruct_path(parent, start, goal)
                self.best_solution  = path
                self.best_fitness   = g[goal]
                self.execution_time = time.time() - t0
                return self

            for neighbor in self.get_neighbors(node):
                new_g = g[node] + 1.0
                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor]      = new_g
                    parent[neighbor] = node
                    f_new            = new_g + self.heuristic(neighbor, goal)
                    heapq.heappush(heap, (f_new, neighbor))

        self.best_solution  = []
        self.best_fitness   = float('inf')
        self.execution_time = time.time() - t0
        return self

    def get_results(self):
        results = super().get_results()
        results.update({"f_values": self.f_values})
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    np.random.seed(5)
    ROWS, COLS = 15, 15
    grid = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in [(0, 0), (14, 14)] and np.random.random() < 0.25:
                grid[r, c] = 1

    START, GOAL = (0, 0), (14, 14)
    print(f"Grid {ROWS}×{COLS} | Start {START} → Goal {GOAL}\n")

    for Cls in [GreedyBestFirst, AStarSearch]:
        alg = Cls(grid, START, GOAL)
        alg.solve()
        r = alg.get_results()
        found = r['best_fitness'] < float('inf')
        print(f"  [{r['algorithm']:16s}] found={found} | "
              f"path={r['path_length']} | nodes={r['nodes_expanded']} | "
              f"time={r['execution_time_seconds']*1000:.2f}ms")
