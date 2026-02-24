import numpy as np
import time
from ..base import BaseMetaheuristic


class ACOR(BaseMetaheuristic):
    """
    Ant Colony Optimization for Continuous Domains (ACO_R)

    Based on Socha & Dorigo (2008) — the canonical extension of ACO to
    real-valued (continuous) search spaces.

    In the original discrete ACO, ants deposit pheromone on graph edges and
    future ants bias their moves toward high-pheromone edges. ACO_R replaces
    graph-edge pheromone with a *solution archive*: a ranked list of the k
    best solutions found so far.

    At each iteration, m new candidate solutions ("ants") are built by
    sampling from a *Gaussian mixture* whose components are the archive
    solutions.  The mean of component l is the l-th archive solution, and its
    standard deviation is proportional to the spread of the archive around
    that solution (scaled by xi, the evaporation analog).  A rank-based
    Gaussian kernel weight biases sampling toward better (lower-ranked)
    archive entries — the continuous analog of preferring high-pheromone
    edges.  After sampling, the archive is updated by keeping only the best k
    solutions from the union of the old archive and the m new ants.

    This archive-update step is the pheromone evaporation-and-reinforcement
    analog: poor solutions are "evaporated" out while good ones are
    "reinforced" by remaining in the archive.
    """

    def __init__(self, objective_func, pop_size, max_iter, bounds, dim,
                 archive_size=50, n_ants=10, q=0.5, xi=0.85):
        """
        Configurable Parameters:
        :param pop_size:     Ignored (set archive_size instead); kept for API
                             compatibility with BaseMetaheuristic.
        :param archive_size: Number of solutions retained in the archive (k).
                             Larger k = more diverse pheromone memory.
        :param n_ants:       Number of new candidate solutions built per
                             iteration (m).  More ants = more exploration per
                             iteration at higher computational cost.
        :param q:            Locality parameter in (0, +∞).
                             Small q (e.g. 0.1) concentrates sampling around
                             the top-ranked archive entries (exploitation).
                             Large q (e.g. 1.0) spreads weight more uniformly
                             across all archive entries (exploration).
        :param xi:           Evaporation-analog in (0, 1].  Scales the
                             standard deviation of each Gaussian kernel.
                             Smaller xi = tighter kernels = stronger
                             exploitation of current archive.
        """
        super().__init__("ACO for Continuous Domains",
                         objective_func, pop_size, max_iter, bounds, dim)
        self.archive_size = archive_size
        self.n_ants       = n_ants
        self.q            = q
        self.xi           = xi

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _compute_kernel_weights(self):
        """
        Compute rank-based Gaussian kernel weights for the solution archive.

        Each archive entry l (1-indexed, sorted best-to-worst) receives weight:

            w_l = 1 / (k * q * sqrt(2*pi)) * exp( -(l-1)^2 / (2*k^2*q^2) )

        The Gaussian is centered on rank 1 (the best solution), so lower-ranked
        (higher quality) entries receive exponentially more weight.  This
        mirrors the way short (high-quality) paths accumulate more pheromone
        than long paths in the original graph-based ACO.

        :returns: Normalized 1-D weight array of length archive_size.
        """
        k     = self.archive_size
        ranks = np.arange(1, k + 1, dtype=float)
        raw   = (1.0 / (k * self.q * np.sqrt(2.0 * np.pi))) * np.exp(
            -((ranks - 1.0) ** 2) / (2.0 * k ** 2 * self.q ** 2)
        )
        return raw / raw.sum()   # normalize to a proper probability distribution

    # ------------------------------------------------------------------
    # Core solve loop
    # ------------------------------------------------------------------

    def solve(self):
        start_time = time.time()

        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]

        # 1. Build Initial Solution Archive
        # Seed with archive_size random solutions and sort by fitness (best first)
        archive = np.random.uniform(
            lower_bounds, upper_bounds, (self.archive_size, self.dim)
        )
        archive_fitness = self.evaluate_population(archive)

        sorted_idx      = np.argsort(archive_fitness)
        archive         = archive[sorted_idx]
        archive_fitness = archive_fitness[sorted_idx]

        # Track global best (archive[0] is always the best after sorting)
        self.best_solution = np.copy(archive[0])
        self.best_fitness  = archive_fitness[0]

        # Precompute kernel weights once (they depend only on k and q)
        weights = self._compute_kernel_weights()

        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):

            # --- SAMPLING PHASE  (Ants construct new solutions) ---
            # Each ant builds its solution dimension-by-dimension by sampling
            # from the Gaussian mixture defined by the current archive.
            new_solutions = np.zeros((self.n_ants, self.dim))

            for ant in range(self.n_ants):
                for d in range(self.dim):

                    # Select which archive solution to use as the Gaussian mean,
                    # weighted by rank-based kernel probabilities (better = more
                    # likely to be chosen, analogous to following a pheromone trail)
                    chosen_l = np.random.choice(self.archive_size, p=weights)
                    mean     = archive[chosen_l, d]

                    # Standard deviation is the weighted average distance of
                    # all archive entries from the chosen mean — this is the
                    # standard ACOR sigma formula; xi is the evaporation factor
                    sigma = self.xi * np.sum(
                        weights * np.abs(archive[:, d] - mean)
                    )
                    # Guard against degenerate distributions when all archive
                    # entries share the same coordinate in dimension d
                    sigma = max(sigma, 1e-10)

                    new_solutions[ant, d] = np.random.normal(mean, sigma)

            # Clip all new solutions to the feasible domain
            new_solutions = np.clip(new_solutions, lower_bounds, upper_bounds)

            # --- ARCHIVE UPDATE PHASE  (Pheromone evaporation & reinforcement) ---
            # Evaluate the m new ants and merge them with the current archive.
            # Keeping only the best k combined solutions is the ACO_R analog of:
            #   - Evaporation  : old poor solutions are dropped from the archive.
            #   - Reinforcement: new good solutions are added and will increase
            #                    sampling density in their neighborhood.
            new_fitness = self.evaluate_population(new_solutions)

            combined         = np.vstack([archive, new_solutions])
            combined_fitness = np.concatenate([archive_fitness, new_fitness])

            best_k_idx      = np.argsort(combined_fitness)[:self.archive_size]
            archive         = combined[best_k_idx]
            archive_fitness = combined_fitness[best_k_idx]

            # --- GLOBAL BEST UPDATE ---
            # archive[0] is always the best after the argsort above
            if archive_fitness[0] < self.best_fitness:
                self.best_fitness  = archive_fitness[0]
                self.best_solution = np.copy(archive[0])

            # Record per-iteration metrics for convergence and diversity plots
            self.convergence_curve[iteration]     = self.best_fitness
            self.average_fitness_curve[iteration] = np.mean(archive_fitness)
            self.diversity_curve[iteration]       = np.mean(np.std(archive, axis=0))
            self.population_history.append(np.copy(archive))

        self.execution_time = time.time() - start_time
        return self.get_results()
