import numpy as np
import time
from ..base import BaseMetaheuristic, BaseAlgorithm


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

# ==============================================================================
#  ACO_TSP — Ant Colony Optimization for Traveling Salesman Problem
# ==============================================================================

import numpy as np
import time


class ACO_TSP:
    """
    Research-grade Ant Colony Optimization for TSP.
    Includes:
    - Iteration-best and global-best pheromone updates
    - Pheromone bounds (anti-stagnation)
    - Convergence tracking
    - Diversity tracking
    - Optional history for visualization
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        n_ants: int = None,
        max_iterations: int = 500,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q: float = 100.0,
        elite_weight: float = 2.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        seed: int = None,
        store_history: bool = False
    ):
        self.dist_matrix = dist_matrix
        self.n = dist_matrix.shape[0]
        self.n_ants = n_ants if n_ants else self.n
        self.max_iterations = max_iterations

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elite_weight = elite_weight

        self.tau_min = tau_min
        self.tau_max = tau_max

        self.seed = seed
        self.store_history = store_history

        # Results
        self.best_solution = None
        self.best_fitness = float("inf")
        self.execution_time = 0.0

        self.convergence_curve = np.zeros(max_iterations)
        self.diversity_curve = np.zeros(max_iterations)

        self.population_history = [] if store_history else None

    # ---------------------------------------------------------
    # Core Methods
    # ---------------------------------------------------------

    def _tour_length(self, tour):
        return float(
            sum(
                self.dist_matrix[tour[i], tour[(i + 1) % self.n]]
                for i in range(self.n)
            )
        )

    def _construct_single_tour(self, pheromone, heuristic):
        tour = []
        visited = np.zeros(self.n, dtype=bool)

        current = np.random.randint(self.n)
        tour.append(current)
        visited[current] = True

        for _ in range(self.n - 1):
            unvisited = np.where(~visited)[0]

            probs = (
                pheromone[current, unvisited] ** self.alpha
                * heuristic[current, unvisited] ** self.beta
            )

            total = probs.sum()
            if total == 0:
                next_city = np.random.choice(unvisited)
            else:
                probs /= total
                next_city = np.random.choice(unvisited, p=probs)

            tour.append(next_city)
            visited[next_city] = True
            current = next_city

        return np.array(tour)

    def _construct_tours(self, pheromone, heuristic):
        tours = []
        fitnesses = []

        for _ in range(self.n_ants):
            tour = self._construct_single_tour(pheromone, heuristic)
            fitness = self._tour_length(tour)
            tours.append(tour)
            fitnesses.append(fitness)

        return np.array(tours), np.array(fitnesses)

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    def solve(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        start_time = time.time()

        pheromone = np.ones((self.n, self.n))
        np.fill_diagonal(pheromone, 0)

        with np.errstate(divide="ignore"):
            heuristic = 1.0 / (self.dist_matrix + 1e-12)
        heuristic[np.isinf(heuristic)] = 0
        np.fill_diagonal(heuristic, 0)

        for iteration in range(self.max_iterations):

            tours, fitnesses = self._construct_tours(pheromone, heuristic)

            # Diversity = std of tour lengths
            self.diversity_curve[iteration] = np.std(fitnesses)

            # Iteration best
            best_idx = np.argmin(fitnesses)
            iter_best_tour = tours[best_idx]
            iter_best_fitness = fitnesses[best_idx]

            # Update global best
            if iter_best_fitness < self.best_fitness:
                self.best_fitness = iter_best_fitness
                self.best_solution = iter_best_tour.copy()

            # Optional history for animation
            if self.store_history:
                self.population_history.append(tours.copy())

            # Evaporation
            pheromone *= (1 - self.rho)

            # Iteration-best deposit
            for i in range(self.n):
                u = iter_best_tour[i]
                v = iter_best_tour[(i + 1) % self.n]

                deposit = self.q / iter_best_fitness
                pheromone[u, v] += deposit
                pheromone[v, u] += deposit

            # Global-best reinforcement (elite)
            if self.best_solution is not None:
                for i in range(self.n):
                    u = self.best_solution[i]
                    v = self.best_solution[(i + 1) % self.n]

                    deposit = self.elite_weight * self.q / self.best_fitness
                    pheromone[u, v] += deposit
                    pheromone[v, u] += deposit

            # Clamp pheromone (prevents stagnation)
            pheromone = np.clip(pheromone, self.tau_min, self.tau_max)

            self.convergence_curve[iteration] = self.best_fitness

        self.execution_time = time.time() - start_time
        return self

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    def get_results(self):
        return {
            "algorithm": "ACO_TSP",
            "best_fitness": float(self.best_fitness),
            "best_solution": self.best_solution.tolist()
            if self.best_solution is not None else None,
            "execution_time_seconds": self.execution_time,
            "convergence_curve": self.convergence_curve,
            "diversity_curve": self.diversity_curve,
            "population_history": self.population_history,
            "time_complexity": "O(max_iter * n_ants * n^2)",
            "space_complexity": "O(n^2)"
        }


# ==============================================================================
#  ACO_Grid — Ant Colony Optimization for Grid Pathfinding
# ==============================================================================

import numpy as np
import time


class ACO_Grid(BaseAlgorithm):

    def __init__(
        self,
        grid,
        start_node,
        end_node,
        n_ants=20,
        max_iterations=100,
        alpha=1.0,
        beta=3.0,
        rho=0.15,
        q=100.0,
        elite_weight=3.0,
        seed=None
    ):
        super().__init__("ACO_Grid")

        self.grid = np.array(grid)
        self.n, self.m = self.grid.shape

        self.start_node = start_node
        self.end_node = end_node

        self.n_ants = n_ants
        self.max_iterations = max_iterations

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elite_weight = elite_weight

        self.seed = seed

        self.best_solution = None
        self.best_fitness = float("inf")

        self.execution_time = 0.0

        self.convergence_curve = np.zeros(max_iterations)
        self.diversity_curve = np.zeros(max_iterations)

        # Manhattan heuristic
        self.heuristic = np.zeros((self.n, self.m))

        for r in range(self.n):
            for c in range(self.m):
                self.heuristic[r, c] = abs(r - end_node[0]) + abs(c - end_node[1])

        # Precompute neighbors
        self.neighbors = {}

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for r in range(self.n):
            for c in range(self.m):

                if self.grid[r, c] != 0:
                    continue

                cell = (r, c)
                self.neighbors[cell] = []

                for dr, dc in directions:

                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < self.n and 0 <= nc < self.m:
                        if self.grid[nr, nc] == 0:
                            self.neighbors[cell].append((nr, nc))


    def _path_length(self, path):
        return float(len(path) - 1)


    def _construct_path(self, pheromone):

        visited = np.zeros((self.n, self.m), dtype=bool)

        path = [self.start_node]

        r, c = self.start_node
        visited[r, c] = True

        max_steps = (self.n + self.m) * 2

        for _ in range(max_steps):

            if (r, c) == self.end_node:
                return path

            neighbors = self.neighbors[(r, c)]

            candidates = []

            for nr, nc in neighbors:

                # allow small chance of revisiting to escape traps
                if not visited[nr, nc] or np.random.rand() < 0.05:
                    candidates.append((nr, nc))

            if not candidates:
                return None

            probs = []

            for nr, nc in candidates:

                tau = pheromone[nr, nc]

                # Manhattan heuristic
                eta = 1.0 / (1.0 + self.heuristic[nr, nc])

                # Direction bias toward goal
                dir_bias = 1.0 / (1.0 + abs(nr - self.end_node[0]) + abs(nc - self.end_node[1]))

                score = (tau ** self.alpha) * (eta ** self.beta) * (1.0 + dir_bias)

                probs.append(score)

            probs = np.array(probs)

            total = probs.sum()

            if total == 0:
                idx = np.random.randint(len(candidates))
            else:
                probs /= total
                idx = np.random.choice(len(candidates), p=probs)

            nr, nc = candidates[idx]

            path.append((nr, nc))
            visited[nr, nc] = True

            r, c = nr, nc

        return None


    def solve(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        start_time = time.time()

        pheromone = np.full((self.n, self.m), 0.1)

        stagnation_counter = 0

        for iteration in range(self.max_iterations):

            paths = []
            fitnesses = []

            for _ in range(self.n_ants):

                path = self._construct_path(pheromone)

                if path is not None:

                    length = self._path_length(path)

                    paths.append(path)
                    fitnesses.append(length)

            if not paths:
                continue

            fitnesses = np.array(fitnesses)

            self.diversity_curve[iteration] = np.std(fitnesses)

            best_idx = np.argmin(fitnesses)

            iter_best_path = paths[best_idx]
            iter_best_fitness = fitnesses[best_idx]

            if iter_best_fitness < self.best_fitness:

                self.best_fitness = iter_best_fitness
                self.best_solution = iter_best_path.copy()

                stagnation_counter = 0

            else:
                stagnation_counter += 1

            # Evaporation
            pheromone *= (1 - self.rho)

            # Deposit pheromone for ALL ants (faster convergence)
            for path, fit in zip(paths, fitnesses):

                deposit = self.q / fit

                for r, c in path:
                    pheromone[r, c] += deposit * 0.2

            # Iteration best reinforcement
            deposit = self.q / iter_best_fitness

            for r, c in iter_best_path:
                pheromone[r, c] += deposit

            # Elite reinforcement
            if self.best_solution is not None:

                deposit = self.elite_weight * self.q / self.best_fitness

                for r, c in self.best_solution:
                    pheromone[r, c] += deposit

            self.convergence_curve[iteration] = self.best_fitness

            # Early stopping (important for speed)
            if stagnation_counter > 15:
                break

        self.execution_time = time.time() - start_time

        return self

    def get_results(self):

        return {
            "algorithm": self.name,
            "best_fitness": float(self.best_fitness),
            "best_solution": self.best_solution,
            "execution_time_seconds": self.execution_time,
            "convergence_curve": self.convergence_curve,
            "diversity_curve": self.diversity_curve,
            "time_complexity": "O(iter * ants * path_length)",
            "space_complexity": "O(grid_size)"
        }