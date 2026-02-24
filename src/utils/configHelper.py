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
    "PSO": PSO
}

# Map algorithms to their test problems (one problem per algorithm for testing)
ALGORITHM_PROBLEM_MAP = {
    "ABC": "sphere",
    "CS": "rastrigin",
    "FA": "rosenbrock",
    "PSO": "sphere",
    "GA": "rastrigin",
    "DE": "sphere",
    "SA": "rosenbrock",
    "GSA": "sphere"
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
    
    # Only run algorithms with their assigned problems
    for algo_name, problem_name in ALGORITHM_PROBLEM_MAP.items():
        if algo_name not in config["algorithms"]:
            continue
        if problem_name not in config["problems"]:
            continue
        
        for dimensions in config["experiment"]["dimensions"]:
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
