import math
import numpy as np
import time
from ..base import BaseMetaheuristic


class CuckooSearch(BaseMetaheuristic):
    """
    Cuckoo Search (CS)

    Proposed by Yang & Deb (2009).
    Fuses two independently motivated biological mechanisms into one optimizer:

    1. Brood parasitism — certain cuckoo species lay their eggs in other birds'
       nests.  Host birds discover and eject alien eggs with probability p_a,
       exerting a survival pressure that continuously discards poor solutions.

    2. Lévy flights — the foraging paths of many animals (albatrosses, fruit
       flies, spider monkeys) follow a heavy-tailed power-law distribution,
       generating many short local steps punctuated by rare, very long jumps.
       Lévy flights explore the search space far more efficiently than a
       Gaussian (Brownian) random walk of equal total path length.

    Combining Lévy-flight exploration with quality-filtered nest abandonment
    gives CS strong global search capability while keeping the population
    diversity high throughout the optimization run.

    Lévy steps are generated via Mantegna's algorithm, which approximates
    the true Lévy distribution efficiently using two independent normal samples.
    """

    def __init__(self, objective_func, pop_size, max_iter, bounds, dim,
                 pa=0.25, alpha=0.01, lambda_levy=1.5):
        """
        Configurable Parameters:
        :param pa:           Nest discovery (abandonment) probability in (0, 1).
                             Fraction of the worst nests replaced per iteration.
                             Default 0.25 is the standard from Yang & Deb (2009).
        :param alpha:        Global step-size scaling factor for the Lévy flight.
                             Should be small relative to the search-space width to
                             prevent overshooting. A common guideline: alpha ≈ 0.01
                             times the domain width.
        :param lambda_levy:  Lévy exponent in (1, 3). Controls the heaviness of
                             the distribution tail — larger lambda means shorter
                             typical jumps with rarer very long ones.
                             Default 1.5 is the most widely used value.
        """
        super().__init__("Cuckoo Search",
                         objective_func, pop_size, max_iter, bounds, dim)
        self.pa           = pa
        self.alpha        = alpha
        self.lambda_levy  = lambda_levy

        # Precompute Mantegna's sigma_u once at construction time to avoid
        # repeating the gamma-function calls inside the hot loop
        lam = lambda_levy
        num = math.gamma(1.0 + lam) * np.sin(np.pi * lam / 2.0)
        den = (math.gamma((1.0 + lam) / 2.0) * lam
               * 2.0 ** ((lam - 1.0) / 2.0))
        self._sigma_u = (num / den) ** (1.0 / lam)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _levy_flight(self, size):
        """
        Generate a Lévy-distributed step vector via Mantegna's algorithm.

        The approximation draws:
            u ~ N(0, sigma_u^2)   (sigma_u precomputed from lambda)
            v ~ N(0, 1)
        and returns   step = u / |v|^(1/lambda).

        The resulting distribution has the power-law tail P(l) ~ l^(-lambda),
        which produces the characteristic mix of short local moves and
        occasional long-range jumps that characterize animal foraging paths.

        :param size: Number of independent step components to generate.
        :returns:    1-D NumPy array of length `size`.
        """
        u    = np.random.normal(0.0, self._sigma_u, size)
        v    = np.random.normal(0.0, 1.0,           size)
        step = u / (np.abs(v) ** (1.0 / self.lambda_levy))
        return step

    # ------------------------------------------------------------------
    # Core solve loop
    # ------------------------------------------------------------------

    def solve(self):
        start_time = time.time()

        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        # Per-dimension range used to scale the Lévy step size
        range_bounds = upper_bounds - lower_bounds

        # 1. Initialize Nests (each nest holds one candidate solution / "egg")
        nests   = self.initialize_population()
        fitness = self.evaluate_population(nests)

        # Track global best; elitism ensures this never deteriorates
        best_idx           = np.argmin(fitness)
        self.best_solution = np.copy(nests[best_idx])
        self.best_fitness  = fitness[best_idx]

        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):

            # --- LÉVY FLIGHT — MECHANISM 1 ---
            # One randomly chosen cuckoo generates a new egg via a Lévy flight
            # and deposits it in a randomly chosen host nest.
            i         = np.random.randint(0, self.pop_size)
            levy_step = self._levy_flight(self.dim)
            # Scale the Lévy step by alpha and the per-dimension domain width
            new_nest  = nests[i] + self.alpha * range_bounds * levy_step
            new_nest  = np.clip(new_nest, lower_bounds, upper_bounds)

            # The new egg competes with the current egg in a randomly chosen nest
            j       = np.random.randint(0, self.pop_size)
            new_fit = self.objective_func(new_nest.reshape(1, -1))[0]
            if new_fit < fitness[j]:
                nests[j]   = new_nest
                fitness[j] = new_fit

            # --- NEST ABANDONMENT — MECHANISM 2 ---
            # Discard and regenerate the worst p_a fraction of nests.
            # Abandoned nests are replaced via a biased differential random walk
            # (difference of two randomly chosen nests), which inherits some
            # directional information from the current population — more
            # informative than a purely random replacement.
            n_abandon    = max(1, int(self.pa * self.pop_size))
            worst_indices = np.argsort(fitness)[-n_abandon:]   # indices of worst nests

            for idx in worst_indices:
                # Select two partners that are distinct from the abandoned nest
                candidates  = np.delete(np.arange(self.pop_size), idx)
                r1, r2      = np.random.choice(candidates, 2, replace=False)
                r           = np.random.rand()   # random step scaling in [0, 1)

                new_abandoned = nests[idx] + r * (nests[r1] - nests[r2])
                new_abandoned = np.clip(new_abandoned, lower_bounds, upper_bounds)

                abandon_fit = self.objective_func(new_abandoned.reshape(1, -1))[0]
                # The abandoned nest is always replaced (forced diversification)
                nests[idx]   = new_abandoned
                fitness[idx] = abandon_fit

            # --- ELITISM: Guarantee the global best is never lost ---
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness  = fitness[current_best_idx]
                self.best_solution = np.copy(nests[current_best_idx])
            else:
                # Re-inject the remembered global best into the worst slot
                worst_idx        = np.argmax(fitness)
                nests[worst_idx]   = np.copy(self.best_solution)
                fitness[worst_idx] = self.best_fitness

            # Record per-iteration metrics for convergence and diversity plots
            self.convergence_curve[iteration]     = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(fitness)
            self.diversity_curve[iteration]       = np.mean(np.std(nests, axis=0))
            self.population_history.append(np.copy(nests))

        self.execution_time = time.time() - start_time
        return self.get_results()
