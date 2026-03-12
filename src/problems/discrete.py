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

        if not lines:
            raise ValueError(f"Empty TSP file: {self.filepath}")

        # First line may be either: "n" or "n m" where m is number of edges
        header = lines[0].split()
        self.n_cities = int(header[0])

        # Initialize distance matrix (symmetric) with infinities and zero diagonal
        self.distance_matrix = np.full((self.n_cities, self.n_cities), np.inf)
        np.fill_diagonal(self.distance_matrix, 0)

        # Subsequent lines are edges: u v weight (1-indexed in files)
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            if 0 <= u < self.n_cities and 0 <= v < self.n_cities:
                self.distance_matrix[u, v] = w
                self.distance_matrix[v, u] = w
    
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
    Shortest Path Problem defined on a square grid (maze).

    Two input formats are supported:
    
    **Format 1 (Fixed positions):**
        Line 1: n                     # size of the grid (n x n)
        Lines 2..n+1: grid rows      # each row has n integers (0=open, 1=wall)
        Start: (0, 1), Goal: (n-1, n-2) [hardcoded]
    
    **Format 2 (Dynamic S/E markers):**
        Line 1: n m                   # grid dimensions (n rows, m cols; or just n for square)
        Lines 2..n+1: grid rows      # each row has m integers/chars: 0=open, 1=wall, S=start, E=end
        Start and Goal: determined by S and E in grid

    The class exposes ``grid`` as a NumPy array and ``start_node`` / 
    ``end_node`` as (row, col) tuples so that the various graph-search 
    algorithms in ``algorithms/classic`` can operate directly on it.
    """

    def __init__(self, filepath):
        """Load a grid-based shortest path instance from file."""
        self.filepath = Path(filepath)
        self.n = 0              # grid rows
        self.m = 0              # grid cols
        self.grid = None        # 2D numpy array of 0/1 values (S/E converted to 0)
        self.start_node = None  # tuple (row, col)
        self.end_node = None
        # keep n_nodes for backwards compatibility (number of cells)
        self.n_nodes = 0
        self._load_from_file()

    def _load_from_file(self):
        """Parse the maze file and populate grid, start/end nodes."""
        with open(self.filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"Empty file: {self.filepath}")

        # First line: "n" or "n m"
        header = lines[0].split()
        self.n = int(header[0])
        self.m = int(header[1]) if len(header) > 1 else self.n

        # read exactly n subsequent lines
        if len(lines) < 1 + self.n:
            raise ValueError(f"Expected {self.n} grid rows, got {len(lines)-1}")

        grid_rows = []
        has_s_e = False  # whether S and E markers are present
        
        for idx, line in enumerate(lines[1 : 1 + self.n], start=1):
            parts = line.split()
            if len(parts) != self.m:
                raise ValueError(
                    f"Line {idx+1} must contain {self.m} values, found {len(parts)}"
                )
            
            row = []
            for col, val in enumerate(parts):
                # Handle S/E markers
                if val == 'S':
                    if self.start_node is not None:
                        raise ValueError(f"Multiple S markers found in grid")
                    self.start_node = (idx - 1, col)  # idx-1 because idx is 1-indexed
                    has_s_e = True
                    row.append(0)  # treat S as open
                elif val == 'E':
                    if self.end_node is not None:
                        raise ValueError(f"Multiple E markers found in grid")
                    self.end_node = (idx - 1, col)
                    has_s_e = True
                    row.append(0)  # treat E as open
                else:
                    # Parse as 0 or 1
                    row.append(int(val))
            
            grid_rows.append(row)

        self.grid = np.array(grid_rows, dtype=int)
        self.n_nodes = self.n * self.m

        # If S/E not found, use hardcoded defaults for backward compatibility
        if not has_s_e:
            self.start_node = (0, 1)
            self.end_node = (self.n - 1, self.m - 2)

    @classmethod
    def from_array(cls, grid, start_node, end_node):
        """
        Create an SPProblem instance from a NumPy array grid and start/end nodes.

        Parameters
        ----------
        grid : np.ndarray
            2D array where 0=open, 1=wall.
        start_node : tuple
            (row, col) for start.
        end_node : tuple
            (row, col) for end.

        Returns
        -------
        SPProblem
            Instance with the provided grid and nodes.
        """
        instance = cls.__new__(cls)
        instance.grid = grid.copy()
        instance.n, instance.m = grid.shape
        instance.start_node = start_node
        instance.end_node = end_node
        instance.n_nodes = instance.n * instance.m
        instance.filepath = None  # No file associated
        return instance

    @classmethod
    def random(cls, rows, cols=None, obstacle_prob=0.25, start=None, end=None, seed=None):
        """
        Generate a random grid-based shortest path problem.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int, optional
            Number of columns (default: same as rows).
        obstacle_prob : float
            Probability of a cell being a wall (0-1).
        start : tuple, optional
            (row, col) for start; if None, random open cell.
        end : tuple, optional
            (row, col) for end; if None, random open cell.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        SPProblem
            Randomly generated instance.
        """
        if cols is None:
            cols = rows
        if seed is not None:
            np.random.seed(seed)

        # Generate grid: 0=open, 1=wall
        grid = np.random.choice([0, 1], size=(rows, cols), p=[1 - obstacle_prob, obstacle_prob])

        # Ensure start and end are open
        def find_open_cell():
            while True:
                r, c = np.random.randint(0, rows), np.random.randint(0, cols)
                if grid[r, c] == 0:
                    return (r, c)

        if start is None:
            start = find_open_cell()
        else:
            grid[start] = 0  # Ensure it's open

        if end is None:
            end = find_open_cell()
        else:
            grid[end] = 0  # Ensure it's open

        # Ensure start != end
        while start == end:
            end = find_open_cell()
            grid[end] = 0

        return cls.from_array(grid, start, end)

    def evaluate_path(self, path):
        """Compute cost of a path represented as a sequence of (row,col) tuples.

        Distance is simply the number of valid moves; invalid steps or
        traversing a wall returns ``np.inf``.
        """
        if not path or len(path) < 2:
            return np.inf

        dist = 0
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            # ensure neighboring cells
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                return np.inf
            # ensure target cell is open
            if self.grid[r2, c2] != 0:
                return np.inf
            dist += 1
        return dist

    def evaluate_population(self, population):
        """Vectorized evaluation of multiple paths.

        ``population`` should be an iterable of path lists/arrays.
        """
        fitness = np.zeros(len(population))
        for i, path in enumerate(population):
            fitness[i] = self.evaluate_path(path)
        return fitness
    
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
    """Get an SP problem by name from the test suite.

    The SP instances are currently stored under ``tests/TSP/SP``
    (a relic of earlier structuring).
    """
    tests_dir = Path(__file__).parent.parent / "tests" / "TSP" / "SP"
    filepath = tests_dir / f"{test_name}.txt"
    return load_sp(filepath)


