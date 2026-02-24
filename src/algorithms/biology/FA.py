import numpy as np
import time
from ..base import BaseMetaheuristic


class FireflyAlgorithm(BaseMetaheuristic):
    """
    Firefly Algorithm (FA)

    Proposed by Xin-She Yang (2008) at Cambridge University.
    Inspired by the bioluminescent signaling of fireflies (Lampyridae):
    each individual emits light whose intensity encodes its solution quality,
    and dimmer fireflies are attracted toward brighter ones. Attractiveness
    decays exponentially with the squared Euclidean distance, modeling light
    absorption in the medium.

    Key property compared to PSO: every firefly can be attracted by any
    brighter firefly (not just the single global best), so the population
    spontaneously self-clusters around multiple local optima simultaneously —
    a natural multimodal search capability without explicit niching.

    Conventions used here (minimization):
      - Lower objective value  <==>  brighter firefly.
      - A firefly i moves toward j whenever f(x_j) < f(x_i).
      - The brightest firefly in the swarm only moves via the random-walk term.
    """

    def __init__(self, objective_func, pop_size, max_iter, bounds, dim,
                 beta0=1.0, gamma=1.0, alpha=0.5, alpha_decay=0.97):
        """
        Configurable Parameters:
        :param beta0:       Maximum attractiveness at zero distance (r = 0).
                            Scales the deterministic pull toward brighter fireflies.
        :param gamma:       Light absorption coefficient. Controls how rapidly
                            attractiveness fades with distance.
                              gamma -> 0 : all fireflies see each other equally
                                          (collapses to a PSO-like behaviour).
                              gamma -> ∞ : attractiveness vanishes instantly
                                          (pure random walk).
                            A practical starting point is gamma = 1.0.
        :param alpha:       Initial random-walk step size. Scales the uniform
                            perturbation applied to all fireflies each generation.
        :param alpha_decay: Multiplicative decay factor in (0, 1) applied to
                            alpha after every generation. Reduces exploration
                            and focuses on exploitation as iterations progress.
        """
        super().__init__("Firefly Algorithm",
                         objective_func, pop_size, max_iter, bounds, dim)
        self.beta0       = beta0
        self.gamma       = gamma
        self.alpha       = alpha
        self.alpha_decay = alpha_decay

    def solve(self):
        start_time = time.time()

        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        # Per-dimension width of the search space, used to scale random perturbations
        range_bounds = upper_bounds - lower_bounds

        # 1. Initialize Firefly Population
        fireflies      = self.initialize_population()
        # Light intensity = objective value; lower is brighter in minimization
        light_intensity = self.evaluate_population(fireflies)

        # Track the globally brightest (best) firefly
        best_idx           = np.argmin(light_intensity)
        self.best_solution = np.copy(fireflies[best_idx])
        self.best_fitness  = light_intensity[best_idx]

        alpha = self.alpha   # mutable step size, decays over time

        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):

            # --- PAIRWISE ATTRACTION PHASE ---
            # For each firefly i, scan all other fireflies j.
            # If j is brighter (lower fitness), i is attracted toward j.
            # We accumulate all movements for i before applying them so that
            # each firefly's original position is used for every comparison
            # within the same generation (synchronous update).
            new_fireflies = np.copy(fireflies)

            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if light_intensity[j] < light_intensity[i]:  # j is brighter

                        # Squared Euclidean distance between the two fireflies
                        r_sq = np.sum((fireflies[i] - fireflies[j]) ** 2)

                        # Attractiveness decays with distance via a Gaussian kernel
                        beta = self.beta0 * np.exp(-self.gamma * r_sq)

                        # Uniform random perturbation centered at zero
                        epsilon = np.random.uniform(-0.5, 0.5, self.dim)

                        # Position update:
                        #   deterministic attraction toward j  +  random exploration
                        new_fireflies[i] = (new_fireflies[i]
                                            + beta * (fireflies[j] - fireflies[i])
                                            + alpha * range_bounds * epsilon)

            # Enforce hard boundary constraints after all movements
            new_fireflies = np.clip(new_fireflies, lower_bounds, upper_bounds)

            # --- FITNESS EVALUATION ---
            new_intensity = self.evaluate_population(new_fireflies)

            # FA accepts all new positions unconditionally (not strictly greedy)
            fireflies      = new_fireflies
            light_intensity = new_intensity

            # --- ALPHA DECAY ---
            # Gradually shrink the random-walk radius to shift from exploration
            # in early iterations to fine exploitation in later iterations
            alpha *= self.alpha_decay

            # --- GLOBAL BEST UPDATE ---
            current_best_idx = np.argmin(light_intensity)
            if light_intensity[current_best_idx] < self.best_fitness:
                self.best_fitness  = light_intensity[current_best_idx]
                self.best_solution = np.copy(fireflies[current_best_idx])

            # Record per-iteration metrics for convergence and diversity plots
            self.convergence_curve[iteration]     = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(light_intensity)
            self.diversity_curve[iteration]       = np.mean(np.std(fireflies, axis=0))
            self.population_history.append(np.copy(fireflies))

        self.execution_time = time.time() - start_time
        return self.get_results()
