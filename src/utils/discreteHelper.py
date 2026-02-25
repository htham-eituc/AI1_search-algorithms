"""
Discrete Optimization Problem Helper Functions
===============================================

Provides utilities for loading and testing TSP and SP problem instances.
Similar to configHelper.py but for discrete (combinatorial) problems.
"""

import numpy as np
from pathlib import Path
from problems.discrete import get_tsp_problem, get_sp_problem


def load_discrete_problem(problem_type, test_name="test_1"):
    """
    Load a discrete optimization problem instance.
    
    Parameters
    ----------
    problem_type : str
        Either 'TSP' or 'SP'
    test_name : str
        Name of test file without extension (e.g., 'test_1')
    
    Returns
    -------
    TSPProblem or SPProblem
        The loaded problem instance
    """
    if problem_type.upper() == "TSP":
        return get_tsp_problem(test_name)
    elif problem_type.upper() == "SP":
        return get_sp_problem(test_name)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def get_available_tests(problem_type):
    """
    Get list of available test files for a problem type.
    
    Parameters
    ----------
    problem_type : str
        Either 'TSP' or 'SP'
    
    Returns
    -------
    list
        Sorted list of test names (without .txt extension)
    """
    tests_dir = Path(__file__).parent.parent / "tests" / problem_type.upper()
    if not tests_dir.exists():
        return []
    
    test_files = sorted([f.stem for f in tests_dir.glob("test_*.txt")])
    return test_files


def print_tsp_info(tsp_problem):
    """Print information about a loaded TSP instance."""
    print(f"\n{'='*70}")
    print(f"TSP INSTANCE: {tsp_problem.filepath.name}")
    print(f"{'='*70}")
    print(f"  Number of cities: {tsp_problem.n_cities}")
    print(f"  Distance matrix shape: {tsp_problem.distance_matrix.shape}")
    
    # Calculate some statistics
    valid_distances = tsp_problem.distance_matrix[tsp_problem.distance_matrix != np.inf]
    if len(valid_distances) > 0:
        print(f"  Min distance: {valid_distances.min():.2f}")
        print(f"  Max distance: {valid_distances.max():.2f}")
        print(f"  Avg distance: {valid_distances.mean():.2f}")


def print_sp_info(sp_problem):
    """Print information about a loaded SP instance."""
    print(f"\n{'='*70}")
    print(f"SP INSTANCE: {sp_problem.filepath.name}")
    print(f"{'='*70}")
    print(f"  Number of nodes: {sp_problem.n_nodes}")
    print(f"  Start node: {sp_problem.start_node + 1} (0-indexed: {sp_problem.start_node})")
    print(f"  End node: {sp_problem.end_node + 1} (0-indexed: {sp_problem.end_node})")
    
    # Count edges
    n_edges = np.sum(np.isfinite(sp_problem.adjacency_matrix) & (sp_problem.adjacency_matrix != 0))
    n_edges = n_edges // 2  # Divide by 2 if undirected
    print(f"  Number of edges: {n_edges}")
    
    # Calculate some statistics
    valid_weights = sp_problem.adjacency_matrix[
        (sp_problem.adjacency_matrix != 0) & 
        (sp_problem.adjacency_matrix != np.inf)
    ]
    if len(valid_weights) > 0:
        print(f"  Min weight: {valid_weights.min():.2f}")
        print(f"  Max weight: {valid_weights.max():.2f}")
        print(f"  Avg weight: {valid_weights.mean():.2f}")


def evaluate_tsp_solution(tsp_problem, solution):
    """
    Evaluate a TSP solution (tour).
    
    Parameters
    ----------
    tsp_problem : TSPProblem
        The loaded TSP instance
    solution : array-like
        A tour (permutation of cities)
    
    Returns
    -------
    float
        Total tour distance
    """
    return tsp_problem.evaluate_tour(solution)


def evaluate_sp_solution(sp_problem, solution):
    """
    Evaluate an SP solution (path).
    
    Parameters
    ----------
    sp_problem : SPProblem
        The loaded SP instance
    solution : array-like
        A path (sequence of node indices)
    
    Returns
    -------
    float
        Total path distance
    """
    return sp_problem.evaluate_path(solution)


def generate_random_tsp_solution(tsp_problem):
    """
    Generate a random valid TSP tour.
    
    Parameters
    ----------
    tsp_problem : TSPProblem
        The loaded TSP instance
    
    Returns
    -------
    np.ndarray
        A random permutation of cities
    """
    return np.random.permutation(tsp_problem.n_cities)


def generate_random_sp_solution(sp_problem, max_path_length=None):
    """
    Generate a random valid SP path.
    
    Parameters
    ----------
    sp_problem : SPProblem
        The loaded SP instance
    max_path_length : int or None
        Maximum length of path (affects search complexity)
    
    Returns
    -------
    list
        A random path from start to end (may not be connected)
    """
    if max_path_length is None:
        max_path_length = sp_problem.n_nodes
    
    path = [sp_problem.start_node]
    current = sp_problem.start_node
    
    while current != sp_problem.end_node and len(path) < max_path_length:
        # Get neighbors
        neighbors = []
        for node in range(sp_problem.n_nodes):
            if sp_problem.adjacency_matrix[current, node] != np.inf:
                neighbors.append(node)
        
        if not neighbors:
            break
        
        # Randomly choose next node
        next_node = np.random.choice(neighbors)
        path.append(next_node)
        current = next_node
    
    # Ensure we end at destination
    if path[-1] != sp_problem.end_node:
        if sp_problem.adjacency_matrix[path[-1], sp_problem.end_node] != np.inf:
            path.append(sp_problem.end_node)
    
    return path


# Demo
if __name__ == "__main__":
    print("DISCRETE OPTIMIZATION PROBLEM HELPER - DEMO")
    print("=" * 80)
    
    # Test TSP
    print("\n[LOADING TSP TEST]")
    try:
        tsp = load_discrete_problem("TSP", "test_1")
        print_tsp_info(tsp)
        
        # Evaluate random tour
        random_tour = generate_random_tsp_solution(tsp)
        distance = evaluate_tsp_solution(tsp, random_tour)
        print(f"  Random tour distance: {distance:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test SP
    print("\n[LOADING SP TEST]")
    try:
        sp = load_discrete_problem("SP", "test_1")
        print_sp_info(sp)
        
        # Evaluate random path
        random_path = generate_random_sp_solution(sp)
        distance = evaluate_sp_solution(sp, random_path)
        print(f"  Random path: {random_path[:5]}..." if len(random_path) > 5 else f"  Random path: {random_path}")
        if distance < np.inf:
            print(f"  Random path distance: {distance:.2f}")
        else:
            print(f"  Random path is not connected")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Available tests
    print("\n[AVAILABLE TESTS]")
    print("=" * 80)
    tsp_tests = get_available_tests("TSP")
    sp_tests = get_available_tests("SP")
    
    print(f"TSP: {len(tsp_tests)} test files")
    print(f"  Examples: {', '.join(tsp_tests[:5])}")
    
    print(f"\nSP: {len(sp_tests)} test files")
    print(f"  Examples: {', '.join(sp_tests[:5])}")
