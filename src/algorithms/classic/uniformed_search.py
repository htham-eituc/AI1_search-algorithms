"""
Uninformed Search Algorithms on Graphs
========================================
- NumPy / pure Python only
- Modular, well-documented, Python best practices
- Configurable parameters
- Works on both unweighted (BFS, DFS) and weighted (UCS) graphs

Algorithms:
    1. Breadth-First Search (BFS)   — shortest path by number of edges
    2. Depth-First Search  (DFS)    — explores deep paths first
    3. Uniform Cost Search (UCS)    — shortest path by total edge weight

Graph representation:
    Adjacency list as dict[node, list[tuple[neighbor, weight]]]
    For unweighted graphs, weight = 1 by default.
"""

import numpy as np
from collections import deque
import heapq


# ══════════════════════════════════════════════════════════════════════════════
#  Graph helper
# ══════════════════════════════════════════════════════════════════════════════

def make_graph(edges: list[tuple], directed: bool = False) -> dict:
    """
    Build an adjacency-list graph from an edge list.

    Parameters
    ----------
    edges : list of (u, v) or (u, v, weight)
        Edge definitions. If weight is omitted, defaults to 1.
    directed : bool
        If False (default), edges are added in both directions.

    Returns
    -------
    dict[node -> list[(neighbor, weight)]]
    """
    graph = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        w    = edge[2] if len(edge) == 3 else 1
        graph.setdefault(u, []).append((v, w))
        if not directed:
            graph.setdefault(v, []).append((u, w))
        else:
            graph.setdefault(v, [])
    return graph


def reconstruct_path(parent: dict, start, goal) -> list:
    """Trace back through parent pointers to reconstruct the path."""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ══════════════════════════════════════════════════════════════════════════════
#  1. Breadth-First Search (BFS)
# ══════════════════════════════════════════════════════════════════════════════

def bfs(graph: dict, start, goal) -> dict:
    """
    Breadth-First Search — finds the path with fewest edges.

    Explores nodes level by level using a FIFO queue.
    Guaranteed to find the shortest path (by edge count) in unweighted graphs.

    Parameters
    ----------
    graph : dict
        Adjacency list {node: [(neighbor, weight), ...]}.
    start : hashable
        Source node.
    goal : hashable
        Target node.

    Returns
    -------
    dict
        'path'     : list – nodes from start to goal, or [] if unreachable
        'visited'  : list – nodes explored in order
        'cost'     : int  – number of edges in shortest path (-1 if unreachable)
        'found'    : bool
    """
    if start == goal:
        return {'path': [start], 'visited': [start], 'cost': 0, 'found': True}

    queue   = deque([start])
    visited = []
    parent  = {start: None}

    while queue:
        node = queue.popleft()
        visited.append(node)

        for neighbor, _ in graph.get(node, []):
            if neighbor not in parent:
                parent[neighbor] = node
                if neighbor == goal:
                    visited.append(neighbor)
                    path = reconstruct_path(parent, start, goal)
                    return {
                        'path'   : path,
                        'visited': visited,
                        'cost'   : len(path) - 1,
                        'found'  : True,
                    }
                queue.append(neighbor)

    return {'path': [], 'visited': visited, 'cost': -1, 'found': False}


# ══════════════════════════════════════════════════════════════════════════════
#  2. Depth-First Search (DFS)
# ══════════════════════════════════════════════════════════════════════════════

def dfs(graph: dict, start, goal) -> dict:
    """
    Depth-First Search — explores as far as possible before backtracking.

    Uses an explicit stack (iterative). Does NOT guarantee shortest path.
    Useful for reachability, cycle detection, and topological sorting.

    Parameters
    ----------
    graph : dict
        Adjacency list {node: [(neighbor, weight), ...]}.
    start : hashable
        Source node.
    goal : hashable
        Target node.

    Returns
    -------
    dict
        'path'     : list – one path from start to goal (not necessarily shortest)
        'visited'  : list – nodes explored in order
        'cost'     : int  – edge count of returned path (-1 if unreachable)
        'found'    : bool
    """
    if start == goal:
        return {'path': [start], 'visited': [start], 'cost': 0, 'found': True}

    stack   = [(start, [start])]      # (current_node, path_so_far)
    visited = []
    seen    = set()

    while stack:
        node, path = stack.pop()

        if node in seen:
            continue
        seen.add(node)
        visited.append(node)

        if node == goal:
            return {
                'path'   : path,
                'visited': visited,
                'cost'   : len(path) - 1,
                'found'  : True,
            }

        for neighbor, _ in reversed(graph.get(node, [])):
            if neighbor not in seen:
                stack.append((neighbor, path + [neighbor]))

    return {'path': [], 'visited': visited, 'cost': -1, 'found': False}


# ══════════════════════════════════════════════════════════════════════════════
#  3. Uniform Cost Search (UCS)
# ══════════════════════════════════════════════════════════════════════════════

def ucs(graph: dict, start, goal) -> dict:
    """
    Uniform Cost Search — finds the minimum-cost path in a weighted graph.

    Expands the node with the lowest cumulative path cost using a min-heap.
    Equivalent to Dijkstra's algorithm for single-target queries.
    Guaranteed optimal for non-negative edge weights.

    Parameters
    ----------
    graph : dict
        Adjacency list {node: [(neighbor, weight), ...]}.
    start : hashable
        Source node.
    goal : hashable
        Target node.

    Returns
    -------
    dict
        'path'     : list  – optimal path from start to goal
        'visited'  : list  – nodes expanded in order
        'cost'     : float – total path cost (-1 if unreachable)
        'found'    : bool
    """
    # heap entries: (cumulative_cost, node, path)
    heap    = [(0, start, [start])]
    visited = []
    best    = {}                        # node -> best cost seen so far

    while heap:
        cost, node, path = heapq.heappop(heap)

        if node in best:                # already expanded with lower cost
            continue
        best[node] = cost
        visited.append(node)

        if node == goal:
            return {
                'path'   : path,
                'visited': visited,
                'cost'   : cost,
                'found'  : True,
            }

        for neighbor, weight in graph.get(node, []):
            new_cost = cost + weight
            if neighbor not in best:
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))

    return {'path': [], 'visited': visited, 'cost': -1, 'found': False}


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Demo graph (Romania road map classic AI example) ────────────────────
    #
    #   Arad --75-- Zerind --71-- Oradea --151-- Sibiu
    #   |                                         |
    #   140                                      99
    #   |                                         |
    #   Timisoara                               Fagaras --211-- Bucharest
    #   |                                         |                 |
    #   118                                     80               90
    #   |                                         |                 |
    #   Lugoj --111-- Mehadia --70-- Drobeta    Pitesti --101-- Bucharest
    #

    edges = [
        ('Arad',      'Zerind',    75),
        ('Arad',      'Timisoara', 118),
        ('Arad',      'Sibiu',     140),
        ('Zerind',    'Oradea',    71),
        ('Oradea',    'Sibiu',     151),
        ('Timisoara', 'Lugoj',     111),
        ('Lugoj',     'Mehadia',   70),
        ('Mehadia',   'Drobeta',   75),
        ('Drobeta',   'Craiova',   120),
        ('Craiova',   'Pitesti',   138),
        ('Craiova',   'RimnicuVilcea', 146),
        ('Sibiu',     'RimnicuVilcea', 80),
        ('Sibiu',     'Fagaras',   99),
        ('RimnicuVilcea', 'Pitesti', 97),
        ('RimnicuVilcea', 'Craiova', 146),
        ('Fagaras',   'Bucharest', 211),
        ('Pitesti',   'Bucharest', 101),
        ('Bucharest', 'Giurgiu',   90),
        ('Bucharest', 'Urziceni',  85),
    ]

    graph = make_graph(edges, directed=False)
    START, GOAL = 'Arad', 'Bucharest'

    print("=" * 60)
    print(f"  Graph: Romania Road Map")
    print(f"  Start: {START}  →  Goal: {GOAL}")
    print("=" * 60)

    for name, func in [("BFS", bfs), ("DFS", dfs), ("UCS", ucs)]:
        result = func(graph, START, GOAL)
        print(f"\n  [{name}]")
        print(f"    Found   : {result['found']}")
        print(f"    Path    : {' → '.join(result['path'])}")
        print(f"    Cost    : {result['cost']}")
        print(f"    Visited : {len(result['visited'])} nodes")

    print()