import json
import os
import numpy as np
from problems.continuous import sphere, rastrigin, rosenbrock
from algorithms.evolution.DE import DE
from algorithms.evolution.GA import GA
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.physics.GSA import GSA
from algorithms.biology.ABC import ABC
from algorithms.biology.CS import CuckooSearch
from algorithms.biology.FA import FireflyAlgorithm
from algorithms.biology.PSO import PSO
from algorithms.biology.ACOR import ACOR
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
    "SA": SimulatedAnnealing,
    "GSA": GSA,
    "ABC": ABC,
    "CS": CuckooSearch,
    "FA": FireflyAlgorithm,
    "PSO": PSO,
    "ACOR": ACOR,
    "HC": HillClimbingContinuous,
    "TLBO": TLBO,
    "SFO": SFO,
    "CA": CA
}

# Map algorithms to their tested problems according to user specifications
# 1. Sphere: HC, SA, GA, DE, GSA, PSO, TLBO
# 2. Rastrigin: HC, SA, GA, DE, ABC, FA, CS, CA
# 3. Rosenbrock: HC, SA, GA, DE, ABC, PSO, SFO, TLBO
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
}


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


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


def run_experiment(algo_name, problem_name, dimensions, config, verbose=True):
    """Run a single experiment."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running {algo_name} on {config['problems'][problem_name]['name']} ({dimensions}D)")
        print(f"{'='*70}")
    
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


def run_all_experiments(config):
    """Run all experiments defined in configuration."""
    results_summary = []
    
    # Run all algorithm-problem combinations defined in ALGORITHM_PROBLEM_MAP
    for key, (algo_name, problem_name) in ALGORITHM_PROBLEM_MAP.items():
        if algo_name not in config["algorithms"]:
            print(f"Warning: Algorithm {algo_name} not in config, skipping {key}")
            continue
        if problem_name not in config["problems"]:
            print(f"Warning: Problem {problem_name} not in config, skipping {key}")
            continue
        
        for dimensions in config["experiment"]["dimensions"]:
            try:
                result = run_experiment(algo_name, problem_name, dimensions, config)
                results_summary.append({
                    "algorithm": algo_name,
                    "problem": problem_name,
                    "dimensions": dimensions,
                    "best_fitness": result["best_fitness"],
                    "avg_fitness": result["average_fitness_curve"][-1],
                    "diversity": result["diversity_curve"][-1],
                    "execution_time": result["execution_time_seconds"]
                })
            except Exception as e:
                print(f"Error running {algo_name} on {problem_name} ({dimensions}D): {e}")
                continue
    
    return results_summary


def print_summary(results_summary):
    """Print a summary of all experiment results."""
    print(f"\n{'='*105}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*105}")
    print(f"{'Algorithm':<12} {'Problem':<12} {'Dims':<6} {'Best Fit':<15} {'Avg Fit':<15} {'Diversity':<15} {'Time (s)':<10}")
    print("-" * 105)
    
    for result in results_summary:
        print(f"{result['algorithm']:<12} {result['problem']:<12} {result['dimensions']:<6} "
              f"{result['best_fitness']:<15.5e} {result['avg_fitness']:<15.5e} {result['diversity']:<15.5e} {result['execution_time']:<10.5f}")
