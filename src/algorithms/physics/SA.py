import numpy as np
import time
import math
from ..base import BaseMetaheuristic


class SimulatedAnnealing(BaseMetaheuristic):
    """
    Simulated Annealing (SA)
    A single-solution metaheuristic inspired by the metallurgical annealing process.
    Capable of escaping local optima by accepting worse solutions with decreasing probability.
    Excellent for continuous and discrete optimization problems.
    """
    def __init__(
        self,
        objective_func,
        pop_size,
        max_iter,
        bounds,
        dim,
        initial_temperature=100.0,
        cooling_rate=0.995,
        cooling_schedule="exponential"
    ):
        """
        Configurable Parameters:
        :param initial_temperature: Starting temperature. Controls initial acceptance of worse solutions.
        :param cooling_rate: Temperature decay factor [0, 1). Higher values cool slower.
        :param cooling_schedule: Cooling mechanism: "exponential", "linear", or "logarithmic".
        
        Note: pop_size is used here to maintain interface compatibility but SA is fundamentally 
        single-solution. Internally, pop_size agents are run in parallel trajectories.
        """
        super().__init__("Simulated Annealing", objective_func, pop_size, max_iter, bounds, dim)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.cooling_schedule = cooling_schedule

    def _cool_temperature(self, temperature, iteration):
        """
        Apply cooling schedule to decay temperature over time.
        
        :param temperature: Current temperature
        :param iteration: Current iteration number
        :return: Cooled temperature
        """
        if self.cooling_schedule == "exponential":
            # T(t) = T0 * cooling_rate^t (most common)
            return temperature * self.cooling_rate
        
        elif self.cooling_schedule == "linear":
            # T(t) = T0 * (1 - t/max_iter)
            return self.initial_temperature * (1 - iteration / self.max_iter)
        
        elif self.cooling_schedule == "logarithmic":
            # T(t) = T0 / ln(1 + t)
            return self.initial_temperature / np.log(1.0 + iteration)
        
        else:
            raise ValueError(f"Unknown cooling schedule: {self.cooling_schedule}")

    def _generate_neighbor(self, current_solution, temperature):
        """
        Generate a neighbor solution by adding Gaussian perturbation scaled by temperature.
        Larger temperature → larger perturbations (exploration).
        Smaller temperature → smaller perturbations (exploitation).
        
        :param current_solution: Current solution (1D array or 2D array for multiple agents)
        :param temperature: Current temperature controlling perturbation magnitude
        :return: Neighbor solution(s) clipped within bounds
        """
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        
        # Handle both single solution and vectorized (pop_size solutions)
        if current_solution.ndim == 1:
            perturbation = np.random.normal(0, temperature * 0.5, self.dim)
            neighbor = current_solution + perturbation
        else:
            # Vectorized: (pop_size, dim)
            perturbation = np.random.normal(0, temperature * 0.5, current_solution.shape)
            neighbor = current_solution + perturbation
        
        # Clip to bounds
        neighbor = np.clip(neighbor, lower_bounds, upper_bounds)
        return neighbor

    def _acceptance_probability(self, delta, temperature):
        """
        Calculate Metropolis acceptance probability using Boltzmann distribution.
        
        :param delta: Cost difference (neighbor_cost - current_cost)
        :param temperature: Current temperature
        :return: Acceptance probability [0, 1]
        """
        if delta < 0:
            # Better solution → always accept
            return 1.0
        else:
            # Worse solution → accept with probability based on temperature and delta
            # Prevent overflow by clamping the exponent
            exponent = -delta / temperature if temperature > 0 else -float('inf')
            exponent = np.clip(exponent, -700, 700)  # Avoid overflow/underflow
            return np.exp(exponent)

    def solve(self):
        """
        Run Simulated Annealing with parallel trajectories (one per population member).
        Each agent follows its own path with the same temperature schedule.
        """
        start_time = time.time()
        
        # 1. Initialize Population
        population = self.initialize_population()  # Shape: (pop_size, dim)
        fitness = self.evaluate_population(population)
        
        # Track the global best
        best_idx = np.argmin(fitness)
        self.best_solution = np.copy(population[best_idx])
        self.best_fitness = fitness[best_idx]
        
        # Current temperature (shared across all agents)
        temperature = self.initial_temperature
        
        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):
            
            # --- NEIGHBOR GENERATION ---
            # Generate neighbors for all population members simultaneously
            neighbors = self._generate_neighbor(population, temperature)
            neighbor_fitness = self.evaluate_population(neighbors)
            
            # --- ACCEPTANCE DECISION ---
            # Vectorized acceptance: calculate delta for all agents
            delta_fitness = neighbor_fitness - fitness
            
            # Calculate acceptance probabilities for all agents
            acceptance_probs = np.array([
                self._acceptance_probability(delta, temperature)
                for delta in delta_fitness
            ])
            
            # Stochastic acceptance: generate random numbers and compare
            random_values = np.random.rand(self.pop_size)
            accepted = random_values < acceptance_probs
            
            # --- UPDATE POPULATION ---
            # Replace solutions that were accepted
            population[accepted] = neighbors[accepted]
            fitness[accepted] = neighbor_fitness[accepted]
            
            # --- UPDATE GLOBAL BEST ---
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = np.copy(population[current_best_idx])
            
            # Record best fitness for convergence plots
            self.convergence_curve[iteration] = self.best_fitness
            
            # --- COOL DOWN ---
            temperature = self._cool_temperature(temperature, iteration)
        
        self.execution_time = time.time() - start_time
        return self.get_results()
