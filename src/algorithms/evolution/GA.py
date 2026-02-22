import numpy as np
import time
from base import BaseMetaheuristic

class GA(BaseMetaheuristic):
    """
    Continuous Genetic Algorithm (GA)
    Uses Tournament Selection, Arithmetic Crossover, and Gaussian Mutation 
    to optimize continuous landscapes.
    """
    def __init__(self, objective_func, pop_size, max_iter, bounds, dim, 
                 crossover_rate=0.8, mutation_rate=0.1, mutation_scale=0.1, tournament_size=3):
        """
        Configurable Parameters:
        :param crossover_rate: Probability [0, 1] that two parents will mate.
        :param mutation_rate: Probability [0, 1] that a specific gene will mutate.
        :param mutation_scale: The standard deviation of the Gaussian noise applied during mutation.
        :param tournament_size: Number of individuals competing in selection. Higher = more exploitation.
        """
        super().__init__("Genetic Algorithm", objective_func, pop_size, max_iter, bounds, dim)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size

    def solve(self):
        start_time = time.time()
        
        # 1. Initialize Population and evaluate
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Track global best
        best_idx = np.argmin(fitness)
        self.best_solution = np.copy(population[best_idx])
        self.best_fitness = fitness[best_idx]
        
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        range_bounds = upper_bounds - lower_bounds

        # 2. Main Evolution Loop
        for iteration in range(self.max_iter):
            
            # --- SELECTION (Tournament) ---
            # Randomly pick 'tournament_size' competitors for each of the 'pop_size' parent slots
            tournaments = np.random.randint(0, self.pop_size, size=(self.pop_size, self.tournament_size))
            tournament_fitnesses = fitness[tournaments]
            
            # Find the index of the winner (lowest fitness) in each tournament
            winner_indices = np.argmin(tournament_fitnesses, axis=1)
            
            # Map back to the original population indices
            parent_indices = tournaments[np.arange(self.pop_size), winner_indices]
            parents = population[parent_indices]
            
            # --- CROSSOVER (Arithmetic) ---
            # Shuffle parents to pair them up (P1 and P2)
            P1 = parents
            P2 = np.roll(parents, shift=1, axis=0) # Shift array by 1 to create pairs
            
            # Create a boolean mask to decide which pairs actually crossover
            cross_mask = np.random.rand(self.pop_size, 1) < self.crossover_rate
            
            # Arithmetic blend: offspring = beta * P1 + (1 - beta) * P2
            beta = np.random.rand(self.pop_size, self.dim)
            offspring = np.where(cross_mask, beta * P1 + (1.0 - beta) * P2, P1)
            
            # --- MUTATION (Gaussian) ---
            # Create a mask for which specific genes (dimensions) mutate
            mut_mask = np.random.rand(self.pop_size, self.dim) < self.mutation_rate
            
            # Generate Gaussian noise scaled to the search space boundaries
            noise = np.random.normal(loc=0.0, scale=self.mutation_scale * range_bounds, size=(self.pop_size, self.dim))
            
            # Apply mutation where the mask is True
            offspring = offspring + (mut_mask * noise)
            
            # Keep offspring inside bounds
            offspring = np.clip(offspring, lower_bounds, upper_bounds)
            
            # --- EVALUATION & ELITISM ---
            # Evaluate new generation
            offspring_fitness = self.evaluate_population(offspring)
            
            # Elitism: Find the worst offspring and replace it with the best previous solution
            # This ensures we never lose our best-found mathematical coordinate
            worst_offspring_idx = np.argmax(offspring_fitness)
            offspring[worst_offspring_idx] = self.best_solution
            offspring_fitness[worst_offspring_idx] = self.best_fitness
            
            # Update population and fitness for the next generation
            population = offspring
            fitness = offspring_fitness
            
            # --- METRICS TRACKING ---
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = np.copy(population[current_best_idx])
                
            self.convergence_curve[iteration] = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(fitness)
            self.diversity_curve[iteration] = np.mean(np.std(population, axis=0))
            
        self.execution_time = time.time() - start_time
        return self.get_results()