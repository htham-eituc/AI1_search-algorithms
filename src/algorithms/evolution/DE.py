import numpy as np
import time
from ..base import BaseMetaheuristic 

class DE(BaseMetaheuristic):
    """
    Differential Evolution (DE)
    A highly efficient population-based optimizer that uses vector differences 
    for mutation. Excellent for continuous, nonlinear, and multimodal landscapes.
    """
    def __init__(self, objective_func, pop_size, max_iter, bounds, dim, F=0.8, CR=0.9):
        """
        Configurable Parameters:
        :param F: Mutation factor [0, 2]. Controls the amplification of the differential variation.
        :param CR: Crossover rate [0, 1]. Controls the fraction of parameter values copied from the mutant.
        """
        super().__init__("Differential Evolution", objective_func, pop_size, max_iter, bounds, dim)
        self.F = F
        self.CR = CR

    def solve(self):
        start_time = time.time()
        
        # 1. Initialize Population and Fitness
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Track the global best
        best_idx = np.argmin(fitness)
        self.best_solution = np.copy(population[best_idx])
        self.best_fitness = fitness[best_idx]
        
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]

        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):
            
            # --- MUTATION PHASE ---
            # For each agent, randomly select 3 distinct other agents (r1, r2, r3)
            # We use a list comprehension for the index generation as it's purely integer selection
            idxs = np.array([
                np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False) 
                for i in range(self.pop_size)
            ])
            r1, r2, r3 = idxs[:, 0], idxs[:, 1], idxs[:, 2]
            
            # Vectorized Mutation: V = x_{r1} + F * (x_{r2} - x_{r3})
            mutant_vectors = population[r1] + self.F * (population[r2] - population[r3])
            
            # Keep mutants strictly inside search boundaries
            mutant_vectors = np.clip(mutant_vectors, lower_bounds, upper_bounds)
            
            # --- CROSSOVER PHASE ---
            # Create a boolean mask of where to crossover based on CR
            cross_points = np.random.rand(self.pop_size, self.dim) <= self.CR
            
            # Ensure at least one dimension is always inherited from the mutant vector
            # to prevent the trial vector from being identical to the target vector
            j_rand = np.random.randint(0, self.dim, size=self.pop_size)
            cross_points[np.arange(self.pop_size), j_rand] = True
            
            # Vectorized Crossover: np.where(condition, true_array, false_array)
            trial_vectors = np.where(cross_points, mutant_vectors, population)
            
            # --- SELECTION PHASE ---
            # Evaluate the new trial vectors
            trial_fitness = self.evaluate_population(trial_vectors)
            
            # Find where the trial vectors are strictly better than the old population
            improved_indices = trial_fitness < fitness
            
            # Replace the old population and fitness with the successful trial vectors
            population[improved_indices] = trial_vectors[improved_indices]
            fitness[improved_indices] = trial_fitness[improved_indices]
            
            # Update Global Best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = np.copy(population[current_best_idx])
                
            # Record best fitness for convergence plots
            self.convergence_curve[iteration] = self.best_fitness
            
        self.execution_time = time.time() - start_time
        return self.get_results()