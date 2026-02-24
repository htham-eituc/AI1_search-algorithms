"""
Uninformed Search Algorithms
==============================
BFS  — Breadth-First Search   : shortest path by edge count
DFS  — Depth-First Search     : explores deepest path first
UCS  — Uniform Cost Search    : shortest path by total cost

All classes inherit BaseGraphSearch from algorithms/base.py.
Problem: 2-D grid pathfinding (0 = open, 1 = wall).
"""

import time
import heapq
from collections import deque
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from algorithms.base import BaseGraphSearch


# ── Shared utility ────────────────────────────────────────────────────────────

def _reconstruct_path(parent: dict, start, goal) -> list:
    """Trace parent pointers back from goal to start."""
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ══════════════════════════════════════════════════════════════════════════════
#  1. Breadth-First Search (BFS)
# ══════════════════════════════════════════════════════════════════════════════

class BFS(BaseGraphSearch):
    """
    Breadth-First Search — guaranteed shortest path by number of edges.

    Expands nodes level by level via a FIFO queue.
    Time/space complexity: O(V + E).
    Optimal for unweighted grids.
    """

    def __init__(self, grid, start_node, end_node):
        super().__init__("BFS", grid, start_node, end_node)

    def solve(self):
        t0    = time.time()
        start = self.start_node
        goal  = self.end_node

        if start == goal:
            self.best_solution  = [start]
            self.best_fitness   = 0
            self.execution_time = time.time() - t0
            return self

        queue  = deque([start])
        parent = {start: None}

        while queue:
            node = queue.popleft()
            self.explored_path.append(node)
            self.nodes_expanded += 1

            for neighbor in self.get_neighbors(node):
                if neighbor not in parent:
                    parent[neighbor] = node
                    if neighbor == goal:
                        self.explored_path.append(neighbor)
                        path                = _reconstruct_path(parent, start, goal)
                        self.best_solution  = path
                        self.best_fitness   = len(path) - 1
                        self.execution_time = time.time() - t0
                        return self
                    queue.append(neighbor)

        # Unreachable
        self.best_solution  = []
        self.best_fitness   = float('inf')
        self.execution_time = time.time() - t0
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  2. Depth-First Search (DFS)
# ══════════════════════════════════════════════════════════════════════════════

class DFS(BaseGraphSearch):
    """
    Depth-First Search — explores as deep as possible before backtracking.

    Uses an explicit stack (iterative) to avoid Python recursion limits.
    Does NOT guarantee the shortest path.
    Time/space complexity: O(V + E).
    """

    def __init__(self, grid, start_node, end_node):
        super().__init__("DFS", grid, start_node, end_node)

    def solve(self):
        t0    = time.time()
        start = self.start_node
        goal  = self.end_node

        if start == goal:
            self.best_solution  = [start]
            self.best_fitness   = 0
            self.execution_time = time.time() - t0
            return self

        # Stack entries: (node, path_so_far)
        stack = [(start, [start])]
        seen  = set()

        while stack:
            node, path = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            self.explored_path.append(node)
            self.nodes_expanded += 1

            if node == goal:
                self.best_solution  = path
                self.best_fitness   = len(path) - 1
                self.execution_time = time.time() - t0
                return self

            for neighbor in reversed(self.get_neighbors(node)):
                if neighbor not in seen:
                    stack.append((neighbor, path + [neighbor]))

        self.best_solution  = []
        self.best_fitness   = float('inf')
        self.execution_time = time.time() - t0
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  3. Uniform Cost Search (UCS)
# ══════════════════════════════════════════════════════════════════════════════

class UCS(BaseGraphSearch):
    """
    Uniform Cost Search — minimum-cost path on a weighted grid.

    Expands nodes in order of cumulative path cost g(n) using a min-heap.
    Grid cell value is treated as the traversal cost of entering that cell
    (value 0 cells cost 1 by default). Equivalent to Dijkstra for
    single-target queries. Guaranteed optimal for non-negative edge costs.
    """

    def __init__(self, grid, start_node, end_node):
        super().__init__("UCS", grid, start_node, end_node)

    def _step_cost(self, node) -> float:
        """Cost of entering a cell — min cost is 1 to avoid zero-cost loops."""
        return max(float(self.grid[node[0], node[1]]), 1.0)

    def solve(self):
        t0    = time.time()
        start = self.start_node
        goal  = self.end_node

        if start == goal:
            self.best_solution  = [start]
            self.best_fitness   = 0.0
            self.execution_time = time.time() - t0
            return self

        # heap: (cumulative_cost, node, path)
        heap    = [(0.0, start, [start])]
        visited = {}                        # node -> best g(n) confirmed

        while heap:
            cost, node, path = heapq.heappop(heap)

            if node in visited:
                continue
            visited[node] = cost
            self.explored_path.append(node)
            self.nodes_expanded += 1

            if node == goal:
                self.best_solution  = path
                self.best_fitness   = cost
                self.execution_time = time.time() - t0
                return self

            for neighbor in self.get_neighbors(node):
                new_cost = cost + self._step_cost(neighbor)
                if neighbor not in visited:
                    heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))

        self.best_solution  = []
        self.best_fitness   = float('inf')
        self.execution_time = time.time() - t0
        return self


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

    for Cls in [BFS, DFS, UCS]:
        alg = Cls(grid, START, GOAL)
        alg.solve()
        r = alg.get_results()
        found = r['best_fitness'] < float('inf')
        print(f"  [{r['algorithm']:4s}] found={found} | "
              f"path={r['path_length']} | nodes={r['nodes_expanded']} | "
              f"time={r['execution_time_seconds']*1000:.2f}ms")
