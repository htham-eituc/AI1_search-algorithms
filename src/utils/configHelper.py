import json
import os
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from problems.continuous import sphere, rastrigin, rosenbrock
from problems.discrete import SPProblem
from algorithms.evolution.DE import DE
from algorithms.evolution.GA import GA, GA_Grid
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.physics.GSA import GSA
from algorithms.biology.ABC import ABC
from algorithms.biology.CS import CuckooSearch
from algorithms.biology.FA import FireflyAlgorithm
from algorithms.biology.PSO import PSO
from algorithms.biology.ACOR import ACOR, ACO_Grid
from algorithms.classic.uninformed_search import BFS, DFS, UCS
from algorithms.classic.informed_search import GreedyBestFirst, AStarSearch
from algorithms.classic.local_search import HillClimbingContinuous
from algorithms.human.tlbo import TLBO
from algorithms.human.sfo import SFO
from algorithms.human.ca import CA

# Map problem names to their objective functions
PROBLEM_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock
}

# Map algorithm names to their classes
ALGORITHM_CLASSES = {
    "DE": DE,
    "GA": GA,
    "GA_Grid": GA_Grid,
    "SA": SimulatedAnnealing,
    "GSA": GSA,
    "ABC": ABC,
    "CS": CuckooSearch,
    "FA": FireflyAlgorithm,
    "PSO": PSO,
    "ACOR": ACOR,
    "ACO_Grid": ACO_Grid,
    "BFS": BFS,
    "DFS": DFS,
    "UCS": UCS,
    "Greedy": GreedyBestFirst,
    "A*": AStarSearch,
    "HC": HillClimbingContinuous,
    "TLBO": TLBO,
    "SFO": SFO,
    "CA": CA
}

# Map algorithms to their tested problems according to user specifications
# 1. Sphere: HC, SA, GA, DE, GSA, PSO, TLBO
# 2. Rastrigin: HC, SA, GA, DE, ABC, FA, CS, CA
# 3. Rosenbrock: HC, SA, GA, DE, ABC, PSO, SFO, TLBO
# 4. Grid (Shortest Path): BFS, DFS, UCS, Greedy, A*, GA_Grid, ACO_Grid
ALGORITHM_PROBLEM_MAP = {
    # Sphere tested with
    "HC_sphere": ("HC", "sphere"),
    "SA_sphere": ("SA", "sphere"),
    "GA_sphere": ("GA", "sphere"),
    "DE_sphere": ("DE", "sphere"),
    "GSA_sphere": ("GSA", "sphere"),
    "PSO_sphere": ("PSO", "sphere"),
    "TLBO_sphere": ("TLBO", "sphere"),
    
    # Rastrigin tested with
    "HC_rastrigin": ("HC", "rastrigin"),
    "SA_rastrigin": ("SA", "rastrigin"),
    "GA_rastrigin": ("GA", "rastrigin"),
    "DE_rastrigin": ("DE", "rastrigin"),
    "ABC_rastrigin": ("ABC", "rastrigin"),
    "FA_rastrigin": ("FA", "rastrigin"),
    "CS_rastrigin": ("CS", "rastrigin"),
    "CA_rastrigin": ("CA", "rastrigin"),
    
    # Rosenbrock tested with
    "HC_rosenbrock": ("HC", "rosenbrock"),
    "SA_rosenbrock": ("SA", "rosenbrock"),
    "GA_rosenbrock": ("GA", "rosenbrock"),
    "DE_rosenbrock": ("DE", "rosenbrock"),
    "ABC_rosenbrock": ("ABC", "rosenbrock"),
    "PSO_rosenbrock": ("PSO", "rosenbrock"),
    "SFO_rosenbrock": ("SFO", "rosenbrock"),
    "TLBO_rosenbrock": ("TLBO", "rosenbrock"),
    
    # Grid (Shortest Path) tested with
    "BFS_grid": ("BFS", "grid"),
    "DFS_grid": ("DFS", "grid"),
    "UCS_grid": ("UCS", "grid"),
    "Greedy_grid": ("Greedy", "grid"),
    "A*_grid": ("A*", "grid"),
    "GA_Grid_grid": ("GA_Grid", "grid"),
    "ACO_Grid_grid": ("ACO_Grid", "grid"),
}


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def discover_test_files(problem_name, config):
    """
    Discover all test files for a discrete problem matching the configured pattern.
    
    Returns a list of test file paths (relative to workspace).
    """
    problem_config = config["problems"][problem_name]
    
    if problem_config.get("type") != "discrete":
        return []
    
    test_dir = problem_config.get("test_dir", "tests/SP")
    test_pattern = problem_config.get("test_pattern", "test_*.txt")
    
    # Build full glob pattern
    full_pattern = os.path.join(test_dir, test_pattern)
    
    # Find all matching files
    test_files = sorted(glob.glob(full_pattern))
    
    return test_files


def get_problem_bounds(problem_name, config):
    """Get search space bounds for a problem."""
    problem_config = config["problems"][problem_name]
    bounds = problem_config["search_space"]
    return bounds


def get_algorithm_params(algo_name, problem_name, dimensions, config):
    """Get algorithm parameters with problem-specific overrides."""
    algo_config = config["algorithms"][algo_name]
    params = algo_config["defaults"].copy()
    
    # Apply problem-specific overrides
    if problem_name in algo_config["problem_overrides"]:
        overrides = algo_config["problem_overrides"][problem_name]
        # For problem-specific overrides, we need to handle dimension-specific settings
        params.update(overrides)
    
    return params


def run_experiment(algo_name, problem_name, dimensions=None, config=None, test_file=None, verbose=True):
    """
    Run a single experiment on either continuous or discrete problem.
    
    For continuous: requires dimensions parameter
    For discrete (grid): uses test_file parameter from config
    """
    problem_config = config["problems"][problem_name]
    is_discrete = problem_config.get("type") == "discrete"
    
    if verbose:
        if is_discrete:
            print(f"\n{'='*70}")
            print(f"Running {algo_name} on {problem_config['name']}")
            print(f"Test file: {test_file}")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"Running {algo_name} on {problem_config['name']} ({dimensions}D)")
            print(f"{'='*70}")
    
    # ===== DISCRETE PROBLEM (Grid Shortest Path) =====
    if is_discrete:
        return _run_grid_experiment(algo_name, problem_name, test_file, config, verbose)
    
    # ===== CONTINUOUS PROBLEM =====
    else:
        return _run_continuous_experiment(algo_name, problem_name, dimensions, config, verbose)


def _run_grid_experiment(algo_name, problem_name, test_file, config, verbose):
    """Run experiment on grid shortest path problem."""
    from problems.discrete import SPProblem
    
    # Load grid problem from file
    prob = SPProblem(test_file)
    
    # Get algorithm parameters
    params = get_algorithm_params(algo_name, problem_name, None, config)
    algo_class = ALGORITHM_CLASSES[algo_name]
    
    # Instantiate algorithm based on type
    if algo_name == "BFS":
        algo = algo_class(prob.grid, prob.start_node, prob.end_node)
    elif algo_name == "DFS":
        algo = algo_class(prob.grid, prob.start_node, prob.end_node)
    elif algo_name == "UCS":
        algo = algo_class(prob.grid, prob.start_node, prob.end_node)
    elif algo_name == "Greedy":
        algo = algo_class(prob.grid, prob.start_node, prob.end_node)
    elif algo_name == "A*":
        algo = algo_class(prob.grid, prob.start_node, prob.end_node)
    elif algo_name == "GA_Grid":
        algo = algo_class(
            prob.grid, prob.start_node, prob.end_node,
            pop_size=params.get("pop_size", 30),
            max_iter=params.get("max_iterations", 100),
            mutation_rate=params.get("mutation_rate", 0.2)
        )
    elif algo_name == "ACO_Grid":
        algo = algo_class(
            prob.grid, prob.start_node, prob.end_node,
            n_ants=params.get("n_ants", 20),
            max_iterations=params.get("max_iterations", 100),
            alpha=params.get("alpha", 0.5),
            beta=params.get("beta", 1.5),
            rho=params.get("rho", 0.1),
            q=params.get("q", 100.0)
        )
    
    # Run solver
    results = algo.solve()
    
    # Print results
    if verbose:
        res = results.get_results()
        print(f"Algorithm: {res['algorithm']}")
        print(f"Best Fitness (path length): {res['best_fitness']:.0f}")
        print(f"Execution Time: {res['execution_time_seconds']:.5f} seconds")
        if 'nodes_expanded' in res:
            print(f"Nodes Expanded: {res['nodes_expanded']}")
    
    return results.get_results()


def _run_continuous_experiment(algo_name, problem_name, dimensions, config, verbose):
    """Run experiment on continuous optimization problem."""
    # Get objective function
    objective_func = PROBLEM_FUNCTIONS[problem_name]
    
    # Get search space
    bounds_range = get_problem_bounds(problem_name, config)
    bounds = np.array([bounds_range for _ in range(dimensions)])
    
    # Get algorithm parameters
    params = get_algorithm_params(algo_name, problem_name, dimensions, config)
    
    # Initialize algorithm based on type
    algo_class = ALGORITHM_CLASSES[algo_name]
    
    if algo_name == "DE":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            F=params.get("mutation_factor", 0.8),
            CR=params.get("crossover_rate", 0.9)
        )
    elif algo_name == "GA":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            crossover_rate=params.get("crossover_rate", 0.8),
            mutation_rate=params.get("mutation_rate", 0.1),
            mutation_scale=params.get("mutation_scale", 0.1),
            tournament_size=params.get("tournament_size", 3)
        )
    elif algo_name == "SA":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 5000),
            bounds=bounds,
            dim=dimensions,
            initial_temperature=params.get("initial_temperature", 1000.0),
            cooling_rate=params.get("cooling_rate", 0.95),
            cooling_schedule=params.get("cooling_schedule", "geometric")
        )
    elif algo_name == "GSA":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            G0=params.get("gravitational_constant", 100.0),
            kbest_initial=params.get("alpha", 20)
        )
    elif algo_name == "ABC":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            limit=params.get("limit", None)
        )
    elif algo_name == "CS":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            pa=params.get("pa", 0.25),
            alpha=params.get("alpha", 0.01),
            lambda_levy=params.get("lambda_levy", 1.5)
        )
    elif algo_name == "FA":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            beta0=params.get("beta0", 1.0),
            gamma=params.get("gamma", 1.0),
            alpha=params.get("alpha", 0.5),
            alpha_decay=params.get("alpha_decay", 0.97)
        )
    elif algo_name == "PSO":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            w_max=params.get("w_max", 0.9),
            w_min=params.get("w_min", 0.4),
            c1=params.get("c1", 2.0),
            c2=params.get("c2", 2.0),
            v_clamp_ratio=params.get("v_clamp_ratio", 0.2)
        )
    elif algo_name == "ACOR":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            Ar_k=params.get("archive_size", 20),
            xi=params.get("xi", 0.85),
            q=params.get("q", 2)
        )
    elif algo_name == "HC":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 10),
            max_iter=params.get("max_iterations", 5000),
            bounds=bounds,
            dim=dimensions,
            step_size=params.get("step_size", 0.5),
            step_decay=params.get("step_decay", 0.995),
            max_restarts=params.get("max_restarts", 10),
            patience=params.get("patience", 30)
        )
    elif algo_name == "TLBO":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions
        )
    elif algo_name == "SFO":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            desired_speed=params.get("desired_speed", 0.8),
            tau=params.get("tau", 0.5),
            A=params.get("A", 2.0),
            B=params.get("B", 0.3),
            r_agent=params.get("r_agent", 0.5),
            dt=params.get("dt", 0.1),
            v_max=params.get("v_max", 2.0)
        )
    elif algo_name == "CA":
        algo = algo_class(
            objective_func=objective_func,
            pop_size=params.get("population_size", 50),
            max_iter=params.get("max_iterations", 1000),
            bounds=bounds,
            dim=dimensions,
            alpha=params.get("alpha", 0.2)
        )
    
    # Run solver
    results = algo.solve()
    
    # Print results
    if verbose:
        print(f"Algorithm: {results['algorithm']}")
        print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
        print(f"Best Fitness Achieved: {results['best_fitness']:.5e}")
        print(f"Final Average Fitness: {results['average_fitness_curve'][-1]:.5e}")
        print(f"Final Diversity: {results['diversity_curve'][-1]:.5e}")
        print(f"Best Solution (first 3 dims): {results['best_solution'][:3]}")
        
        problem_config = config["problems"][problem_name]
        global_min = problem_config["global_minimum"]
        tolerance = problem_config["tolerance"]
        print(f"Global Minimum: {global_min}")
        print(f"Tolerance: {tolerance}")
        print(f"Success: {abs(results['best_fitness'] - global_min) < tolerance}")
    
    return results


def run_single_experiment_worker(args):
    """Module-level worker function for multiprocessing (must be picklable)."""
    algo_name, problem_name, config, dimensions, test_file = args
    try:
        result = run_experiment(algo_name, problem_name, dimensions=dimensions, 
                               config=config, test_file=test_file, verbose=False)
        if problem_name == "grid":
            # Extract just the filename from the full path
            test_file_name = os.path.basename(test_file) if test_file else "unknown"
            experiment_data = {
                "algorithm": algo_name,
                "problem": problem_name,
                "test_file": test_file_name,
                "best_fitness": result["best_fitness"],
                "execution_time": result["execution_time_seconds"]
            }
        else:
            experiment_data = {
                "algorithm": algo_name,
                "problem": problem_name,
                "dimensions": dimensions,
                "best_fitness": result["best_fitness"],
                "avg_fitness": result["average_fitness_curve"][-1],
                "diversity": result["diversity_curve"][-1],
                "execution_time": result["execution_time_seconds"]
            }
        return experiment_data
    except Exception as e:
        return None


def run_all_experiments(config, max_workers=None):
    """Run all experiments defined in configuration using multiprocessing."""
    if max_workers is None:
        max_workers = os.cpu_count()
    
    results_summary = []
    tasks = []
    
    # Collect all continuous problem tasks
    for key, (algo_name, problem_name) in ALGORITHM_PROBLEM_MAP.items():
        if algo_name not in config["algorithms"] or problem_name not in config["problems"]:
            continue
        
        problem_config = config["problems"][problem_name]
        
        # Skip grid for now (handle separately)
        if problem_config.get("type") == "discrete":
            continue
        
        dimensions_to_test = problem_config.get("dimensions", config["experiment"]["dimensions"])
        
        for dimensions in dimensions_to_test:
            tasks.append((algo_name, problem_name, config, dimensions, None))
    
    # Add grid problem tasks with automatic test file discovery
    if config["experiment"].get("test_discrete", False):
        discrete_algorithms = config["experiment"].get("discrete_algorithms", [])
        
        # Discover all test files for grid problem
        test_files = discover_test_files("grid", config)
        
        if not test_files:
            print("Warning: No test files found for grid problem matching the configured pattern.")
        
        for test_file in test_files:
            for algo_name in discrete_algorithms:
                tasks.append((algo_name, "grid", config, None, test_file))
    
    # Run experiments in parallel using processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_experiment_worker, task) for task in tasks]
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                results_summary.append(result)
            completed += 1
    
    return results_summary


def print_summary(results_summary):
    """Print a summary of all experiment results."""
    print(f"\n{'='*130}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*130}")
    
    # Separate continuous and discrete results
    continuous_results = [r for r in results_summary if "dimensions" in r]
    discrete_results = [r for r in results_summary if "dimensions" not in r]
    
    # Print continuous results
    if continuous_results:
        print("\nCONTINUOUS OPTIMIZATION RESULTS:")
        print(f"{'Algorithm':<12} {'Problem':<12} {'Dims':<6} {'Best Fit':<15} {'Avg Fit':<15} {'Diversity':<15} {'Time (s)':<10}")
        print("-" * 130)
        
        for result in continuous_results:
            print(f"{result['algorithm']:<12} {result['problem']:<12} {result['dimensions']:<6} "
                  f"{result['best_fitness']:<15.5e} {result['avg_fitness']:<15.5e} {result['diversity']:<15.5e} {result['execution_time']:<10.5f}")
    
    # Print discrete results grouped by test file
    if discrete_results:
        print("\nDISCRETE OPTIMIZATION RESULTS (Grid Shortest Path):")
        
        # Group results by test file
        by_test_file = {}
        for result in discrete_results:
            test_file = result.get("test_file", "unknown")
            if test_file not in by_test_file:
                by_test_file[test_file] = []
            by_test_file[test_file].append(result)
        
        # Print results for each test file
        for test_file in sorted(by_test_file.keys()):
            print(f"\n  Test File: {test_file}")
            print(f"  {'Algorithm':<20} {'Path Length':<20} {'Time (s)':<12}")
            print(f"  {'-'*52}")
            
            # Sort by path length (fitness)
            test_results = sorted(by_test_file[test_file], key=lambda x: x['best_fitness'])
            
            for result in test_results:
                print(f"  {result['algorithm']:<20} {result['best_fitness']:<20.0f} {result['execution_time']:<12.5f}")
