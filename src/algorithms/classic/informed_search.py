"""
Informed Search Algorithms on Graphs
======================================
- NumPy / pure Python only
- Modular, well-documented, Python best practices
- Configurable heuristic functions
- Works on any graph with a user-supplied heuristic h(node, goal)

Algorithms:
    1. Greedy Best-First Search — expand node with lowest h(n)
    2. A* Search               — expand node with lowest f(n) = g(n) + h(n)
"""

import heapq


# ══════════════════════════════════════════════════════════════════════════════
#  Graph helper (reused from uninformed_search.py)
# ══════════════════════════════════════════════════════════════════════════════

def make_graph(edges: list[tuple], directed: bool = False) -> dict:
    """Build adjacency list from edge list [(u,v,w), ...] or [(u,v), ...]."""
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
    path, node = [], goal
    while node is not None:
        path.append(node); node = parent[node]
    return path[::-1]


# ══════════════════════════════════════════════════════════════════════════════
#  1. Greedy Best-First Search
# ══════════════════════════════════════════════════════════════════════════════

def greedy_best_first(graph: dict, start, goal, heuristic: dict) -> dict:
    """
    Greedy Best-First Search — always expand the node closest to goal by h(n).

    Uses only the heuristic h(n) to rank nodes; ignores path cost g(n).
    Fast but NOT guaranteed to find the optimal path.

    Parameters
    ----------
    graph : dict
        Adjacency list {node: [(neighbor, weight), ...]}.
    start : hashable
        Source node.
    goal : hashable
        Target node.
    heuristic : dict
        {node: estimated_cost_to_goal}.  h(goal) must equal 0.

    Returns
    -------
    dict
        'path'     : list  – path found (not necessarily optimal)
        'visited'  : list  – expansion order
        'cost'     : float – actual cost of returned path
        'found'    : bool
    """
    if start == goal:
        return {'path': [start], 'visited': [start], 'cost': 0, 'found': True}

    # heap: (h(node), node)
    heap    = [(heuristic.get(start, 0), start)]
    parent  = {start: None}
    g_cost  = {start: 0}          # track real costs for reporting
    visited = []

    while heap:
        h_val, node = heapq.heappop(heap)

        if node in visited:        # skip stale heap entries
            continue
        visited.append(node)

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return {
                'path'   : path,
                'visited': visited,
                'cost'   : g_cost[goal],
                'found'  : True,
            }

        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                tentative_g = g_cost[node] + weight
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    parent[neighbor] = node
                    heapq.heappush(heap, (heuristic.get(neighbor, 0), neighbor))

    return {'path': [], 'visited': visited, 'cost': -1, 'found': False}


# ══════════════════════════════════════════════════════════════════════════════
#  2. A* Search
# ══════════════════════════════════════════════════════════════════════════════

def astar(graph: dict, start, goal, heuristic: dict) -> dict:
    """
    A* Search — optimal informed search using f(n) = g(n) + h(n).

    Expands nodes in order of f(n) = actual cost g(n) + heuristic h(n).
    Guaranteed to find the optimal path when h(n) is admissible
    (i.e., h(n) never overestimates the true cost to goal).

    Parameters
    ----------
    graph : dict
        Adjacency list {node: [(neighbor, weight), ...]}.
    start : hashable
        Source node.
    goal : hashable
        Target node.
    heuristic : dict
        {node: estimated_cost_to_goal}.  Must be admissible for optimality.

    Returns
    -------
    dict
        'path'     : list  – optimal path from start to goal
        'visited'  : list  – expansion order
        'cost'     : float – optimal total path cost (-1 if unreachable)
        'found'    : bool
        'f_values' : dict  – {node: f(n)} for every expanded node
    """
    if start == goal:
        return {'path': [start], 'visited': [start],
                'cost': 0, 'found': True, 'f_values': {start: 0}}

    g       = {start: 0}
    parent  = {start: None}
    f_vals  = {}
    visited = []

    # heap: (f(node), node)
    h_start = heuristic.get(start, 0)
    heap    = [(g[start] + h_start, start)]

    while heap:
        f_val, node = heapq.heappop(heap)

        if node in f_vals:         # already expanded optimally
            continue
        f_vals[node] = f_val
        visited.append(node)

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return {
                'path'   : path,
                'visited': visited,
                'cost'   : g[goal],
                'found'  : True,
                'f_values': f_vals,
            }

        for neighbor, weight in graph.get(node, []):
            tentative_g = g[node] + weight
            if neighbor not in g or tentative_g < g[neighbor]:
                g[neighbor]      = tentative_g
                parent[neighbor] = node
                f_new            = tentative_g + heuristic.get(neighbor, 0)
                heapq.heappush(heap, (f_new, neighbor))

    return {'path': [], 'visited': visited, 'cost': -1,
            'found': False, 'f_values': f_vals}


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Romania road map ─────────────────────────────────────────────────────
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
        ('Fagaras',   'Bucharest', 211),
        ('Pitesti',   'Bucharest', 101),
        ('Bucharest', 'Giurgiu',   90),
        ('Bucharest', 'Urziceni',  85),
    ]
    graph = make_graph(edges, directed=False)

    # Straight-line distance heuristic to Bucharest (from AIMA textbook)
    h_bucharest = {
        'Arad': 366, 'Bucharest': 0,   'Craiova': 160,   'Drobeta': 242,
        'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77,    'Hirsova': 151,
        'Iasi': 226,  'Lugoj': 244,    'Mehadia': 241,   'Neamt': 234,
        'Oradea': 380,'Pitesti': 100,  'RimnicuVilcea': 193,
        'Sibiu': 253, 'Timisoara': 329,'Urziceni': 80,   'Vaslui': 199,
        'Zerind': 374,
    }

    START, GOAL = 'Arad', 'Bucharest'

    print("=" * 62)
    print(f"  Graph : Romania Road Map")
    print(f"  Start : {START}  →  Goal : {GOAL}")
    print(f"  Heuristic : straight-line distance to Bucharest")
    print("=" * 62)

    # Greedy Best-First
    r_gbf = greedy_best_first(graph, START, GOAL, h_bucharest)
    print(f"\n  [Greedy Best-First]")
    print(f"    Found   : {r_gbf['found']}")
    print(f"    Path    : {' → '.join(r_gbf['path'])}")
    print(f"    Cost    : {r_gbf['cost']}  (actual road distance)")
    print(f"    Visited : {len(r_gbf['visited'])} nodes")

    # A*
    r_as = astar(graph, START, GOAL, h_bucharest)
    print(f"\n  [A* Search]")
    print(f"    Found   : {r_as['found']}")
    print(f"    Path    : {' → '.join(r_as['path'])}")
    print(f"    Cost    : {r_as['cost']}  (optimal)")
    print(f"    Visited : {len(r_as['visited'])} nodes")
    print(f"    f-values along path:")
    for node in r_as['path']:
        fv = r_as['f_values'].get(node, '–')
        gv = r_as['cost'] if node == GOAL else '?'
        print(f"      {node:20s}  f = {fv}")

    print()

    # ── Comparison ────────────────────────────────────────────────────────────
    print("  ┌────────────────────┬──────────┬──────────┐")
    print("  │ Algorithm          │  Cost    │ Visited  │")
    print("  ├────────────────────┼──────────┼──────────┤")
    for label, r in [("Greedy Best-First", r_gbf), ("A*", r_as)]:
        print(f"  │ {label:18s} │ {str(r['cost']):8s} │ {len(r['visited']):8d} │")
    print("  └────────────────────┴──────────┴──────────┘")