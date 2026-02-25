"""
Discrete Optimization Problems
==============================

TSP (Traveling Salesman Problem) - Combinatorial Route Optimization
  - Format: Complete graph with weighted edges
  - Objective: Find shortest Hamiltonian cycle visiting all cities

SP (Shortest Path Grid) - Exact Pathfinding on Graph
  - Format: Graph with nodes and weighted edges
  - Objective: Find shortest path from source to destination
"""

import numpy as np
from pathlib import Path


# ==============================================================================
#  Traveling Salesman Problem (TSP)
# ==============================================================================

class TSPProblem:
    """
    Traveling Salesman Problem loader and evaluator.
    
    File format:
        Line 1: n (number of cities)
        Lines 2+: city1 city2 distance
    
    Creates a complete distance matrix from edge list.
    """
    
    def __init__(self, filepath):
        """Load TSP instance from file."""
        self.filepath = Path(filepath)
        self.n_cities = 0
        self.distance_matrix = None
        self.best_known_solution = None
        self._load_from_file()
    
    def _load_from_file(self):
        """Parse TSP file and build distance matrix."""
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # First line: number of cities
        self.n_cities = int(lines[0])
        
        # Initialize distance matrix (symmetric)
        self.distance_matrix = np.full((self.n_cities, self.n_cities), np.inf)
        np.fill_diagonal(self.distance_matrix, 0)
        
        # Parse edges: city1 city2 distance
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            city1, city2, dist = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            self.distance_matrix[city1, city2] = dist
            self.distance_matrix[city2, city1] = dist  # Symmetric
    
    def evaluate_tour(self, tour):
        """
        Evaluate a single tour (permutation).
        
        Parameters
        ----------
        tour : np.ndarray or list
            A permutation of cities [0, 1, ..., n_cities-1]
        
        Returns
        -------
        float
            Total distance of the tour (closing back to start)
        """
        tour = np.array(tour) if not isinstance(tour, np.ndarray) else tour
        
        if len(tour) != self.n_cities:
            return np.inf
        
        distance = 0.0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]  # Wrap around
            distance += self.distance_matrix[from_city, to_city]
        
        return distance
    
    def evaluate_population(self, population):
        """
        Evaluate multiple tours (vectorized over population).
        
        Parameters
        ----------
        population : np.ndarray, shape (pop_size, n_cities)
            Population of permutations
        
        Returns
        -------
        np.ndarray, shape (pop_size,)
            Fitness values (tour distances)
        """
        fitness = np.zeros(len(population))
        for i, tour in enumerate(population):
            fitness[i] = self.evaluate_tour(tour)
        return fitness


def load_tsp(filepath):
    """Load a TSP instance and return the problem object."""
    return TSPProblem(filepath)


# ==============================================================================
#  Shortest Path Problem (SP) on General Graph
# ==============================================================================

class SPProblem:
    """
    Shortest Path Problem on general weighted graph.
    
    File format:
        Line 1: n_nodes start end (start and end nodes)
        Lines 2+: node1 node2 weight
    
    Builds adjacency matrix from edge list.
    """
    
    def __init__(self, filepath):
        """Load SP instance from file."""
        self.filepath = Path(filepath)
        self.n_nodes = 0
        self.start_node = 0
        self.end_node = 0
        self.adjacency_matrix = None
        self.best_known_distance = None
        self._load_from_file()
    
    def _load_from_file(self):
        """Parse SP file and build adjacency matrix."""
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # First line: n_nodes n_edges (or n_nodes start end)
        # Handle both formats: if 3 values, assume n_nodes start end; if 2 values, assume n_nodes n_edges
        parts = lines[0].split()
        self.n_nodes = int(parts[0])
        
        if len(parts) == 3:
            # Format: n_nodes start end
            self.start_node = int(parts[1]) - 1  # Convert to 0-indexed
            self.end_node = int(parts[2]) - 1
        else:
            # Format: n_nodes n_edges (default to node 1 to node n)
            self.start_node = 0
            self.end_node = self.n_nodes - 1
        
        # Initialize adjacency matrix
        self.adjacency_matrix = np.full((self.n_nodes, self.n_nodes), np.inf)
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        # Parse edges: node1 node2 weight
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            node1, node2, weight = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            
            # Ensure nodes are within bounds
            if 0 <= node1 < self.n_nodes and 0 <= node2 < self.n_nodes:
                self.adjacency_matrix[node1, node2] = weight
                self.adjacency_matrix[node2, node1] = weight  # Assume undirected
    
    def evaluate_path(self, path):
        """
        Evaluate a single path as a sequence of nodes.
        
        Parameters
        ----------
        path : np.ndarray or list
            Sequence of node indices from start to end
        
        Returns
        -------
        float
            Total path cost
        """
        path = np.array(path) if not isinstance(path, np.ndarray) else path
        
        if len(path) < 2:
            return np.inf
        
        distance = 0.0
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            distance += self.adjacency_matrix[from_node, to_node]
        
        return distance
    
    def evaluate_population(self, population):
        """
        Evaluate multiple paths (vectorized over population).
        
        Parameters
        ----------
        population : list of lists/arrays
            Population of paths
        
        Returns
        -------
        np.ndarray, shape (pop_size,)
            Fitness values (path distances)
        """
        fitness = np.zeros(len(population))
        for i, path in enumerate(population):
            fitness[i] = self.evaluate_path(path)
        return fitness


def load_sp(filepath):
    """Load a SP instance and return the problem object."""
    return SPProblem(filepath)


# ==============================================================================
#  Convenience functions (same signature as continuous.py functions)
# ==============================================================================

def get_tsp_problem(test_name="test_1"):
    """Get a TSP problem by name (e.g., 'test_1')."""
    tests_dir = Path(__file__).parent.parent / "tests" / "TSP"
    filepath = tests_dir / f"{test_name}.txt"
    return load_tsp(filepath)


def get_sp_problem(test_name="test_1"):
    """Get an SP problem by name (e.g., 'test_1')."""
    tests_dir = Path(__file__).parent.parent / "tests" / "SP"
    filepath = tests_dir / f"{test_name}.txt"
    return load_sp(filepath)


