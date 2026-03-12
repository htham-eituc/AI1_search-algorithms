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
import numpy as np
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

    def __init__(self, grid, start_node, end_node, record_frontier=False):
        super().__init__("BFS", grid, start_node, end_node)
        self.record_frontier = record_frontier

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
            # Save frontier state for visualization
            if self.record_frontier:
                self._save_frontier(list(queue))
            
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

    def __init__(self, grid, start_node, end_node, record_frontier=False):
        super().__init__("DFS", grid, start_node, end_node)
        self.record_frontier = record_frontier

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
            # Save frontier state for visualization
            if self.record_frontier:
                self._save_frontier([n for n, p in stack])
            
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

    def __init__(self, grid, start_node, end_node, record_frontier=False):
        super().__init__("UCS", grid, start_node, end_node)
        self.record_frontier = record_frontier

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
            # Save frontier state for visualization
            if self.record_frontier:
                self._save_frontier([n for c, n, p in heap])
            
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


# ══════════════════════════════════════════════════════════════════════════════
#  BFS_TSP & DFS_TSP — Uninformed Search for Traveling Salesman Problem
# ══════════════════════════════════════════════════════════════════════════════

class BFS_TSP:
    """
    Breadth-First Search for TSP (branch-and-bound style).
    
    WARNING: Only suitable for VERY small instances (n <= 10) due to 
    exponential state space growth. Explores partial tours level-by-level,
    pruning branches that cannot lead to a better solution.
    
    Not practical but demonstrates exhaustive search bounds.
    """
    
    def __init__(self, dist_matrix, max_depth: int = None, seed: int = None):
        self.dist_matrix = dist_matrix
        self.n = dist_matrix.shape[0]
        self.max_depth = max_depth or self.n
        self.seed = seed
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.execution_time = 0.0
        self.nodes_expanded = 0
    
    def _tour_length(self, tour):
        return float(sum(
            self.dist_matrix[tour[i], tour[(i + 1) % len(tour)]] 
            for i in range(len(tour))
        ))
    
    def solve(self):
        import numpy as np
        if self.seed is not None:
            np.random.seed(self.seed)
        
        t0 = time.time()
        
        # BFS: level-by-level exploration of partial tours
        queue = deque()
        for start in range(self.n):
            queue.append(([start], {start}))
        
        while queue:
            tour, visited = queue.popleft()
            self.nodes_expanded += 1
            
            # Pruning: if partial tour cost exceeds best, skip
            partial_cost = sum(
                self.dist_matrix[tour[i], tour[i + 1]] 
                for i in range(len(tour) - 1)
            )
            if partial_cost >= self.best_fitness:
                continue
            
            # Goal: complete tour
            if len(tour) == self.n:
                total_cost = self._tour_length(tour)
                if total_cost < self.best_fitness:
                    self.best_fitness = total_cost
                    self.best_solution = tuple(tour)
                continue
            
            # Expand: add unvisited city
            last = tour[-1]
            for next_city in range(self.n):
                if next_city not in visited:
                    new_tour = tour + [next_city]
                    new_visited = visited | {next_city}
                    queue.append((new_tour, new_visited))
        
        self.execution_time = time.time() - t0
        if self.best_solution is None:
            self.best_fitness = float('inf')
        return self
    
    def get_results(self):
        return {
            "algorithm": "BFS_TSP",
            "best_fitness": float(self.best_fitness),
            "best_solution": list(self.best_solution) if self.best_solution else None,
            "execution_time_seconds": self.execution_time,
            "nodes_expanded": self.nodes_expanded,
            "note": "Only suitable for n <= 10. Exponential complexity O(n!)",
        }


class DFS_TSP:
    """
    Depth-First Search for TSP (branch-and-bound style).
    
    WARNING: Only suitable for VERY small instances (n <= 10).
    Explores tours depth-first, pruning branches that exceed the current best.
    
    Not practical but exhaustive; demonstrates DFS ordering vs. BFS.
    """
    
    def __init__(self, dist_matrix, max_depth: int = None, seed: int = None):
        self.dist_matrix = dist_matrix
        self.n = dist_matrix.shape[0]
        self.max_depth = max_depth or self.n
        self.seed = seed
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.execution_time = 0.0
        self.nodes_expanded = 0
    
    def _tour_length(self, tour):
        return float(sum(
            self.dist_matrix[tour[i], tour[(i + 1) % len(tour)]] 
            for i in range(len(tour))
        ))
    
    def _dfs(self, tour, visited):
        """Recursive DFS with branch-and-bound."""
        self.nodes_expanded += 1
        
        # Pruning
        partial_cost = sum(
            self.dist_matrix[tour[i], tour[i + 1]] 
            for i in range(len(tour) - 1)
        )
        if partial_cost >= self.best_fitness:
            return
        
        # Goal
        if len(tour) == self.n:
            total_cost = self._tour_length(tour)
            if total_cost < self.best_fitness:
                self.best_fitness = total_cost
                self.best_solution = tuple(tour)
            return
        
        # Expand
        last = tour[-1]
        for next_city in range(self.n):
            if next_city not in visited:
                self._dfs(tour + [next_city], visited | {next_city})
    
    def solve(self):
        import numpy as np
        if self.seed is not None:
            np.random.seed(self.seed)
        
        t0 = time.time()
        
        # Start DFS from each city
        for start in range(self.n):
            self._dfs([start], {start})
        
        self.execution_time = time.time() - t0
        if self.best_solution is None:
            self.best_fitness = float('inf')
        return self
    
    def get_results(self):
        return {
            "algorithm": "DFS_TSP",
            "best_fitness": float(self.best_fitness),
            "best_solution": list(self.best_solution) if self.best_solution else None,
            "execution_time_seconds": self.execution_time,
            "nodes_expanded": self.nodes_expanded,
            "note": "Only suitable for n <= 10. Exponential complexity O(n!)",
        }