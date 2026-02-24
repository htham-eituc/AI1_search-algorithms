import numpy as np
import time
from ..base import BaseMetaheuristic


class ABC(BaseMetaheuristic):
    """
    Artificial Bee Colony (ABC)

    Proposed by Karaboga (2005) at Erciyes University.
    Models the foraging division of labor in a honey-bee colony.
    The colony of size 2*SN is split into three functional roles:

      - Employed bees  (SN agents): each is linked to one food source and
        exploits it via single-dimension perturbation.
      - Onlooker bees  (SN agents): observe the waggle dance at the hive and
        probabilistically select sources proportional to their nectar amount
        (fitness), then perform additional neighborhood search.
      - Scout bees      (implicit): any employed bee that has failed to improve
        its source for more than `limit` consecutive trials abandons it and
        is re-assigned to a randomly generated new source.

    This three-phase cycle naturally separates exploitation from exploration
    without any additional control parameter.
    """

    def __init__(self, objective_func, pop_size, max_iter, bounds, dim, limit=None):
        """
        Configurable Parameters:
        :param pop_size: Number of food sources SN (employed = onlooker = SN).
                         Total colony size is 2 * SN.
        :param limit:    Abandonment threshold. A source is scouted (replaced with
                         a fresh random solution) after `limit` consecutive failures.
                         Defaults to pop_size * dim, the most common heuristic.
        """
        super().__init__("Artificial Bee Colony",
                         objective_func, pop_size, max_iter, bounds, dim)
        self.SN = pop_size
        # limit = SN * D is the standard default from Karaboga's original paper
        self.limit = limit if limit is not None else pop_size * dim

    def solve(self):
        start_time = time.time()

        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]

        # 1. Initialize Food Sources and Trial Counters
        food_sources = self.initialize_population()   # shape: (SN, dim)
        fitness = self.evaluate_population(food_sources)
        # trial_counters tracks how many consecutive iterations each source has
        # failed to improve; when it exceeds `limit` the source is abandoned
        trial_counters = np.zeros(self.SN, dtype=int)

        # Track global best
        best_idx = np.argmin(fitness)
        self.best_solution = np.copy(food_sources[best_idx])
        self.best_fitness   = fitness[best_idx]

        # 2. Main Optimization Loop — each iteration represents one full ABC cycle
        for iteration in range(self.max_iter):

            # --- EMPLOYED BEE PHASE ---
            # Every employed bee generates a neighbor candidate by perturbing a
            # single, randomly chosen dimension using a random partner source
            for i in range(self.SN):
                j = np.random.randint(0, self.dim)     # dimension index to perturb
                # Partner k must be distinct from i to create a meaningful difference
                k = np.random.choice(
                    [x for x in range(self.SN) if x != i]
                )
                phi = np.random.uniform(-1.0, 1.0)    # random scaling in [-1, 1]

                candidate = np.copy(food_sources[i])
                candidate[j] = (food_sources[i, j]
                                + phi * (food_sources[i, j] - food_sources[k, j]))
                candidate = np.clip(candidate, lower_bounds, upper_bounds)

                cand_fit = self.objective_func(candidate.reshape(1, -1))[0]

                # Greedy selection: accept only if the candidate strictly improves
                if cand_fit < fitness[i]:
                    food_sources[i] = candidate
                    fitness[i]       = cand_fit
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # --- ONLOOKER BEE PHASE ---
            # Convert raw objective values to nectar amounts using the standard
            # ABC fitness transform (maps minimization fitness to [0, inf])
            fit_scores = np.where(
                fitness >= 0,
                1.0 / (1.0 + fitness),
                1.0 + np.abs(fitness)
            )
            # Roulette-wheel probabilities proportional to nectar amount
            probabilities = fit_scores / fit_scores.sum()

            for _ in range(self.SN):
                # Fitness-proportionate source selection (waggle dance analog)
                i = np.random.choice(self.SN, p=probabilities)
                j = np.random.randint(0, self.dim)
                k = np.random.choice(
                    [x for x in range(self.SN) if x != i]
                )
                phi = np.random.uniform(-1.0, 1.0)

                candidate = np.copy(food_sources[i])
                candidate[j] = (food_sources[i, j]
                                + phi * (food_sources[i, j] - food_sources[k, j]))
                candidate = np.clip(candidate, lower_bounds, upper_bounds)

                cand_fit = self.objective_func(candidate.reshape(1, -1))[0]

                if cand_fit < fitness[i]:
                    food_sources[i] = candidate
                    fitness[i]       = cand_fit
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # --- SCOUT BEE PHASE ---
            # Any source that has exceeded the abandonment limit is discarded and
            # replaced by a uniformly random new solution — this is the exploration
            # injection that prevents the colony from stagnating
            exhausted = np.where(trial_counters > self.limit)[0]
            for i in exhausted:
                food_sources[i] = np.random.uniform(lower_bounds, upper_bounds)
                fitness[i]       = self.objective_func(food_sources[i].reshape(1, -1))[0]
                trial_counters[i] = 0

            # --- GLOBAL BEST UPDATE ---
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness  = fitness[current_best_idx]
                self.best_solution = np.copy(food_sources[current_best_idx])

            # Record per-iteration metrics for convergence and diversity plots
            self.convergence_curve[iteration]     = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(fitness)
            self.diversity_curve[iteration]       = np.mean(np.std(food_sources, axis=0))
            self.population_history.append(np.copy(food_sources))

        self.execution_time = time.time() - start_time
        return self.get_results()
