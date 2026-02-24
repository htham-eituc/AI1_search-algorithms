import numpy as np
import time
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    The ultimate master class. All algorithms must inherit from this to ensure 
    standardized tracking of execution time and universal output formats.
    """
    def __init__(self, name):
        self.name = name
        self.execution_time = 0.0
        self.best_solution = None
        self.best_fitness = float('inf')  # Assuming minimization for all problems

    @abstractmethod
    def solve(self):
        """
        The core logic goes here. Every child class MUST implement this method.
        """
        pass

    def get_results(self):
        """
        Standardized output payload. This guarantees that your visualization 
        notebooks always receive the exact same dictionary structure.
        """
        return {
            "algorithm": self.name,
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "execution_time_seconds": self.execution_time
            
        }


class BaseMetaheuristic(BaseAlgorithm):
    """
    Blueprint for population/agent-based algorithms (GA, PSO, DE, ACO, SA, etc.)
    Used for Sphere, Rastrigin, Rosenbrock, and TSP.
    """
    def __init__(self, name, objective_func, pop_size, max_iter, bounds=None, dim=None):
        super().__init__(name)
        self.objective_func = objective_func
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds  # Shape: (dim, 2) defining [min, max] for each dimension
        self.dim = dim
        
        # Array to track the best fitness at every single iteration for the Convergence plot
        self.convergence_curve = np.zeros(max_iter)
        # For Average Solution Quality
        self.average_fitness_curve = np.zeros(max_iter)
        # For Exploration vs. Exploitation (Spatial Diversity)
        self.diversity_curve = np.zeros(max_iter)
        # To store the positions of the agents for the MP4 video animation
        self.population_history = []

    def initialize_population(self):
        """
        Helper method to generate the starting population uniformly within bounds.
        Returns a NumPy array of shape (pop_size, dim).
        """
        if self.bounds is None or self.dim is None:
            raise ValueError("Bounds and dimensions must be set for continuous problems.")
            
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        
        # Vectorized random initialization
        return np.random.uniform(lower_bounds, upper_bounds, (self.pop_size, self.dim))

    def evaluate_population(self, population):
        """
        Expects the objective_func to handle vectorized NumPy operations.
        Returns a 1D NumPy array of fitness values.
        """
        return self.objective_func(population)
    
    def get_results(self):
        """Upgraded to return the new tracking curves."""
        results = super().get_results()
        results.update({
            "convergence_curve": self.convergence_curve,
            "average_fitness_curve": self.average_fitness_curve,
            "diversity_curve": self.diversity_curve,
            "population_history": self.population_history 
        })
        return results

    # The actual solve() method will be implemented by the specific GA/PSO/etc. classes,
    # but they will call self.initialize_population() and self.evaluate_population() inside it.


class BaseGraphSearch(BaseAlgorithm):
    """
    Blueprint for classic discrete pathfinding algorithms (BFS, DFS, A*, Greedy).
    Used strictly for the Shortest Path Grid problem.
    """
    def __init__(self, name, grid, start_node, end_node):
        super().__init__(name)
        self.grid = grid              # 2D NumPy array (e.g., 0 for open, 1 for wall)
        self.start_node = start_node  # Tuple: (row, col)
        self.end_node = end_node      # Tuple: (row, col)
        
        # Crucial metric for A* vs BFS comparison
        self.nodes_expanded = 0       
        # To visualize the search wave later
        self.explored_path = []       

    def get_neighbors(self, current_node):
        """
        Finds valid (up, down, left, right) neighbors that are not walls.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        rows, cols = self.grid.shape
        
        for dr, dc in directions:
            r, c = current_node[0] + dr, current_node[1] + dc
            # Check bounds and if the cell is passable (assuming 0 is open path)
            if 0 <= r < rows and 0 <= c < cols and self.grid[r, c] == 0:
                neighbors.append((r, c))
                
        return neighbors

    def get_results(self):
        """
        Overrides the master get_results to include pathfinding-specific metrics.
        """
        results = super().get_results()
        results.update({
            "nodes_expanded": self.nodes_expanded,
            "path_length": len(self.best_solution) if self.best_solution else 0,
            "explored_nodes_history": self.explored_path
        })
        return results