import numpy as np
import time
from ..base import BaseMetaheuristic


class GSA(BaseMetaheuristic):
    """
    Gravitational Search Algorithm (GSA)
    A physics-inspired population-based metaheuristic based on Newton's law of gravity.
    Agents (masses) attract each other through gravitational forces, enabling both
    exploration and exploitation through dynamic force interactions.
    """
    def __init__(
        self,
        objective_func,
        pop_size,
        max_iter,
        bounds,
        dim,
        G0=100.0,
        kbest_initial=1.0
    ):
        """
        Configurable Parameters:
        :param G0: Initial gravitational constant. Controls the magnitude of gravitational forces.
        :param kbest_initial: Initial fraction of best agents to consider [0, 1].
                              Decreases to 1 agent over iterations (exploitation phase).
        """
        super().__init__("Gravitational Search Algorithm", objective_func, pop_size, max_iter, bounds, dim)
        self.G0 = G0
        self.kbest_initial = kbest_initial
        
        # Velocities for all agents (required for physics-based movement)
        self.velocities = None

    def _calculate_gravitational_constant(self, iteration):
        """
        Exponentially decay gravitational constant over iterations.
        Late iterations have smaller forces, promoting exploitation.
        
        :param iteration: Current iteration number
        :return: Current gravitational constant G(t)
        """
        # G(t) = G0 * exp(-20 * t / max_iter)
        return self.G0 * np.exp(-20.0 * iteration / self.max_iter)

    def _calculate_kbest(self, iteration):
        """
        Dynamically reduce the number of best agents considered for force calculation.
        Early iterations: consider many agents (exploration).
        Late iterations: consider only the best agent (exploitation).
        
        :param iteration: Current iteration number
        :return: Number of best agents to use for force calculation
        """
        # Linear decay from kbest_initial (usually 1.0) down to 1
        kbest = max(1, int(self.pop_size * self.kbest_initial * (1.0 - iteration / self.max_iter)))
        return kbest

    def _calculate_masses(self, fitness):
        """
        Calculate masses for all agents based on their fitness values.
        Better agents have larger masses (attract others more strongly).
        Uses exponential mapping to avoid numerical issues.
        
        :param fitness: 1D array of fitness values for all agents
        :return: 1D array of normalized masses [0, 1]
        """
        # Find worst fitness (helps normalize exponential mapping)
        worst_idx = np.argmax(fitness)
        worst_fitness = fitness[worst_idx]
        
        # Map fitness to normalized values [0, 1] where 0 = best, ~1 = worst
        fitness_min = np.min(fitness)
        fitness_max = np.max(fitness)
        
        if fitness_max - fitness_min > 1e-10:
            normalized_fitness = (fitness - fitness_min) / (fitness_max - fitness_min)
        else:
            normalized_fitness = np.zeros_like(fitness)
        
        # Apply exponential mapping: better agents (lower normalized_fitness) get higher mass
        std_fitness = np.std(normalized_fitness)
        if std_fitness > 1e-10:
            masses = np.exp(-(normalized_fitness - np.min(normalized_fitness)) / (2 * std_fitness**2))
        else:
            masses = np.ones_like(fitness)
        
        # Normalize masses to sum to 1
        mass_sum = np.sum(masses)
        if mass_sum > 1e-10:
            masses = masses / mass_sum
        
        return masses

    def _calculate_forces(self, population, fitness, masses, G, kbest, iteration):
        """
        Calculate the total gravitational force on each agent from the kbest agents.
        Forces are vectorized for efficiency.
        
        :param population: (pop_size, dim) array of agent positions
        :param fitness: (pop_size,) array of fitness values
        :param masses: (pop_size,) array of normalized masses
        :param G: Current gravitational constant
        :param kbest: Number of best agents to consider
        :param iteration: Current iteration (for random scaling)
        :return: (pop_size, dim) array of acceleration vectors
        """
        forces = np.zeros_like(population)
        
        # Find indices of kbest best agents
        best_indices = np.argsort(fitness)[:kbest]
        
        # For each agent, calculate force from all kbest agents
        for i in range(self.pop_size):
            for j in best_indices:
                if i != j:
                    # Distance between agent i and agent j
                    diff = population[j] - population[i]
                    distance = np.linalg.norm(diff)
                    
                    # Avoid division by zero
                    if distance > 1e-10:
                        # Gravitational force: F = G * (m_j / distance) * diff_direction
                        # Random coefficient to vary force application
                        rand_coeff = np.random.random()
                        force_magnitude = G * masses[j] / (distance + 1e-10)
                        forces[i] += rand_coeff * force_magnitude * (diff / (distance + 1e-10))
        
        return forces

    def solve(self):
        """
        Execute Gravitational Search Algorithm with dynamic mass interactions.
        """
        start_time = time.time()
        
        # 1. Initialize Population and Velocities
        population = self.initialize_population()  # Shape: (pop_size, dim)
        self.velocities = np.zeros_like(population)  # All agents start with zero velocity
        fitness = self.evaluate_population(population)
        
        # Track the global best
        best_idx = np.argmin(fitness)
        self.best_solution = np.copy(population[best_idx])
        self.best_fitness = fitness[best_idx]
        
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        
        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):
            
            # --- CALCULATE PARAMETERS FOR THIS ITERATION ---
            G = self._calculate_gravitational_constant(iteration)
            kbest = self._calculate_kbest(iteration)
            
            # --- CALCULATE MASSES ---
            masses = self._calculate_masses(fitness)
            
            # --- CALCULATE FORCES AND UPDATE VELOCITIES ---
            forces = self._calculate_forces(population, fitness, masses, G, kbest, iteration)
            
            # Update velocities: v(t+1) = rand() * v(t) + a(t) where a = forces
            # The random coefficient prevents agents from getting stuck in local patterns
            rand_inertia = np.random.random((self.pop_size, 1))  # Shape: (pop_size, 1)
            self.velocities = rand_inertia * self.velocities + forces
            
            # --- UPDATE POSITIONS ---
            population = population + self.velocities
            
            # --- APPLY BOUNDARY CONDITIONS ---
            population = np.clip(population, lower_bounds, upper_bounds)
            
            # --- EVALUATE NEW POPULATION ---
            fitness = self.evaluate_population(population)
            
            # --- UPDATE GLOBAL BEST ---
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = np.copy(population[current_best_idx])
            
            # Record best fitness for convergence plots
            self.convergence_curve[iteration] = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(fitness)
            self.diversity_curve[iteration] = np.mean(np.std(population, axis=0))
            self.population_history.append(np.copy(population))
        
        self.execution_time = time.time() - start_time
        return self.get_results()
