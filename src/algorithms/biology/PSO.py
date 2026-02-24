import numpy as np
import time
from ..base import BaseMetaheuristic


class PSO(BaseMetaheuristic):
    """
    Particle Swarm Optimization (PSO)

    Introduced by Kennedy & Eberhart (1995).
    Simulates the social foraging behavior of bird flocks and fish schools.
    Each particle is simultaneously pulled toward its own historical best
    position (cognitive component) and the best position ever found by any
    particle in the swarm (social component). The balance between these two
    forces, modulated by a decaying inertia weight, naturally transitions
    the swarm from broad exploration at the start to fine-grained exploitation
    near convergence.
    """

    def __init__(self, objective_func, pop_size, max_iter, bounds, dim,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, v_clamp_ratio=0.2):
        """
        Configurable Parameters:
        :param w_max: Starting inertia weight (high = strong momentum / exploration).
        :param w_min: Final inertia weight (low = short steps / exploitation).
                      w is linearly decayed from w_max to w_min over all iterations.
        :param c1:    Cognitive coefficient. Scales attraction toward each particle's
                      personal best. Typical value: 2.0.
        :param c2:    Social coefficient. Scales attraction toward the global best.
                      Typical value: 2.0 (c1 + c2 <= 4 is recommended).
        :param v_clamp_ratio: Maximum velocity expressed as a fraction of the
                              search-space width per dimension. Prevents particles
                              from flying out of bounds before position clamping.
        """
        super().__init__("Particle Swarm Optimization",
                         objective_func, pop_size, max_iter, bounds, dim)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_clamp_ratio = v_clamp_ratio

    def solve(self):
        start_time = time.time()

        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        # Velocity clamp: no particle dimension may travel faster than v_max per step
        v_max = self.v_clamp_ratio * (upper_bounds - lower_bounds)

        # 1. Initialize Swarm: positions and velocities
        positions = self.initialize_population()
        # Velocities start uniformly within [-v_max, v_max] per dimension
        velocities = np.random.uniform(-v_max, v_max, (self.pop_size, self.dim))
        fitness = self.evaluate_population(positions)

        # Personal best: each particle remembers its own best-visited location
        personal_best_pos = np.copy(positions)
        personal_best_fit = np.copy(fitness)

        # Global best: the single best position found by any particle so far
        best_idx = np.argmin(personal_best_fit)
        self.best_solution = np.copy(personal_best_pos[best_idx])
        self.best_fitness = personal_best_fit[best_idx]

        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):

            # --- INERTIA WEIGHT LINEAR DECAY ---
            # w decreases from w_max (strong global search) to w_min (tight local search)
            w = self.w_max - (self.w_max - self.w_min) * (iteration / max(self.max_iter - 1, 1))

            # --- VELOCITY UPDATE ---
            # Independent uniform random scalars per particle per dimension
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            cognitive = self.c1 * r1 * (personal_best_pos - positions)
            social    = self.c2 * r2 * (self.best_solution  - positions)

            velocities = w * velocities + cognitive + social

            # Clamp velocities to prevent explosive divergence
            velocities = np.clip(velocities, -v_max, v_max)

            # --- POSITION UPDATE ---
            positions = positions + velocities
            # Enforce hard boundary constraints (absorbing walls)
            positions = np.clip(positions, lower_bounds, upper_bounds)

            # --- FITNESS EVALUATION ---
            fitness = self.evaluate_population(positions)

            # --- PERSONAL BEST UPDATE ---
            # A particle updates its memory only when it finds a strictly better location
            improved = fitness < personal_best_fit
            personal_best_pos[improved] = np.copy(positions[improved])
            personal_best_fit[improved] = fitness[improved]

            # --- GLOBAL BEST UPDATE ---
            current_best_idx = np.argmin(personal_best_fit)
            if personal_best_fit[current_best_idx] < self.best_fitness:
                self.best_fitness = personal_best_fit[current_best_idx]
                self.best_solution = np.copy(personal_best_pos[current_best_idx])

            # Record per-iteration metrics for convergence and diversity plots
            self.convergence_curve[iteration]    = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(fitness)
            self.diversity_curve[iteration]       = np.mean(np.std(positions, axis=0))
            self.population_history.append(np.copy(positions))

        self.execution_time = time.time() - start_time
        return self.get_results()
