import json
import os
import numpy as np
from pathlib import Path
from problems.continuous import sphere, rastrigin, rosenbrock
from algorithms.evolution.DE import DE
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.physics.GSA import GSA

# Map problem names to their objective functions
PROBLEM_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock
}

# Map algorithm names to their classes
ALGORITHM_CLASSES = {
    "DE": DE,
    "SA": SimulatedAnnealing,
    "GSA": GSA
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
    
    # Run solver
    results = algo.solve()
    
    # Print results
    if verbose:
        print(f"Algorithm: {results['algorithm']}")
        print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
        print(f"Best Fitness Achieved: {results['best_fitness']:.5e}")
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
    
    for algo_name in config["algorithms"].keys():
        for problem_name in config["problems"].keys():
            # Skip GSA on rastrigin
            if algo_name == "GSA" and problem_name == "rastrigin":
                continue
            for dimensions in config["experiment"]["dimensions"]:
                result = run_experiment(algo_name, problem_name, dimensions, config)
                results_summary.append({
                    "algorithm": algo_name,
                    "problem": problem_name,
                    "dimensions": dimensions,
                    "best_fitness": result["best_fitness"],
                    "execution_time": result["execution_time_seconds"]
                })
    
    return results_summary


def print_summary(results_summary):
    """Print a summary of all experiment results."""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Algorithm':<15} {'Problem':<15} {'Dimensions':<12} {'Best Fitness':<20} {'Time (s)':<10}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['algorithm']:<15} {result['problem']:<15} {result['dimensions']:<12} "
              f"{result['best_fitness']:<20.5e} {result['execution_time']:<10.5f}")


# OLD TEST FUNCTIONS (COMMENTED OUT - FOR REFERENCE)

# def test_de_sphere():
#     print("--- Testing Differential Evolution on 10D Sphere Function ---")
#     
#     dimensions = 10
#     bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
#     
#     # Initialize the DE Algorithm
#     de_algo = DE(
#         objective_func=sphere, 
#         pop_size=50, 
#         max_iter=100, 
#         bounds=bounds, 
#         dim=dimensions,
#         F=0.8,   # Standard mutation factor
#         CR=0.9   # High crossover rate works well for Sphere
#     )
#     
#     # Run the solver
#     results = de_algo.solve()
#     
#     print(f"Algorithm: {results['algorithm']}")
#     print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
#     # Using scientific notation because DE gets very close to 0
#     print(f"Best Fitness Achieved: {results['best_fitness']:.5e}") 
#     print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")
#
# def test_sa_sphere():
#     print("\n--- Testing Simulated Annealing on 10D Sphere Function ---")
#     
#     dimensions = 10
#     bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
#     
#     # Initialize the SA Algorithm
#     sa_algo = SimulatedAnnealing(
#         objective_func=sphere, 
#         pop_size=50, 
#         max_iter=100, 
#         bounds=bounds, 
#         dim=dimensions,
#         initial_temperature=100.0,
#         cooling_rate=0.995,
#         cooling_schedule="exponential"
#     )
#     
#     # Run the solver
#     results = sa_algo.solve()
#     
#     print(f"Algorithm: {results['algorithm']}")
#     print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
#     print(f"Best Fitness Achieved: {results['best_fitness']:.5e}") 
#     print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")
#
# def test_gsa_sphere():
#     print("\n--- Testing Gravitational Search Algorithm on 10D Sphere Function ---")
#     
#     dimensions = 10
#     bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
#     
#     # Initialize the GSA Algorithm
#     gsa_algo = GSA(
#         objective_func=sphere, 
#         pop_size=50, 
#         max_iter=100, 
#         bounds=bounds, 
#         dim=dimensions,
#         G0=100.0,
#         kbest_initial=1.0
#     )
#     
#     # Run the solver
#     results = gsa_algo.solve()
#     
#     print(f"Algorithm: {results['algorithm']}")
#     print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
#     print(f"Best Fitness Achieved: {results['best_fitness']:.5e}") 
#     print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")
#
# def test_de_rastrigin():
#     print("\n--- Testing Differential Evolution on 10D Rastrigin Function ---")
#     
#     dimensions = 10
#     bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
#     
#     # Initialize the DE Algorithm
#     de_algo = DE(
#         objective_func=rastrigin, 
#         pop_size=50, 
#         max_iter=100, 
#         bounds=bounds, 
#         dim=dimensions,
#         F=0.8,
#         CR=0.9
#     )
#     
#     # Run the solver
#     results = de_algo.solve()
#     
#     print(f"Algorithm: {results['algorithm']}")
#     print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
#     print(f"Best Fitness Achieved: {results['best_fitness']:.5f}") 
#     print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")
#
# def test_de_rosenbrock():
#     print("\n--- Testing Differential Evolution on 10D Rosenbrock Function ---")
#     
#     dimensions = 10
#     bounds = np.array([[-2.0, 2.0] for _ in range(dimensions)]) 
#     
#     # Initialize the DE Algorithm
#     de_algo = DE(
#         objective_func=rosenbrock, 
#         pop_size=50, 
#         max_iter=100, 
#         bounds=bounds, 
#         dim=dimensions,
#         F=0.8,
#         CR=0.9
#     )
#     
#     # Run the solver
#     results = de_algo.solve()
#     
#     print(f"Algorithm: {results['algorithm']}")
#     print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
#     print(f"Best Fitness Achieved: {results['best_fitness']:.5f}") 
#     print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")
#
# if __name__ == "__main__":
#     test_de_sphere()
#     test_sa_sphere()
#     test_gsa_sphere()
#     test_de_rastrigin()
#     test_de_rosenbrock()


if __name__ == "__main__":
    # Get config path (relative to script location)
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    np.random.seed(config["experiment"]["seed"])
    
    # Run all experiments
    results_summary = run_all_experiments(config)
    
    # Print summary
    print_summary(results_summary)

    # Save results to CSV
    import csv
    csv_path = script_dir / "results.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        fieldnames = ["algorithm", "problem", "dimensions", "best_fitness", "execution_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_summary:
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")