import numpy as np
from problems.continuous import sphere, rastrigin, rosenbrock
from algorithms.evolution.DE import DE
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.physics.GSA import GSA

def test_de_sphere():
    print("--- Testing Differential Evolution on 10D Sphere Function ---")
    
    dimensions = 10
    bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
    
    # Initialize the DE Algorithm
    de_algo = DE(
        objective_func=sphere, 
        pop_size=50, 
        max_iter=100, 
        bounds=bounds, 
        dim=dimensions,
        F=0.8,   # Standard mutation factor
        CR=0.9   # High crossover rate works well for Sphere
    )
    
    # Run the solver
    results = de_algo.solve()
    
    print(f"Algorithm: {results['algorithm']}")
    print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
    # Using scientific notation because DE gets very close to 0
    print(f"Best Fitness Achieved: {results['best_fitness']:.5e}") 
    print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")

def test_sa_sphere():
    print("\n--- Testing Simulated Annealing on 10D Sphere Function ---")
    
    dimensions = 10
    bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
    
    # Initialize the SA Algorithm
    sa_algo = SimulatedAnnealing(
        objective_func=sphere, 
        pop_size=50, 
        max_iter=100, 
        bounds=bounds, 
        dim=dimensions,
        initial_temperature=100.0,
        cooling_rate=0.995,
        cooling_schedule="exponential"
    )
    
    # Run the solver
    results = sa_algo.solve()
    
    print(f"Algorithm: {results['algorithm']}")
    print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
    print(f"Best Fitness Achieved: {results['best_fitness']:.5e}") 
    print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")

def test_gsa_sphere():
    print("\n--- Testing Gravitational Search Algorithm on 10D Sphere Function ---")
    
    dimensions = 10
    bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
    
    # Initialize the GSA Algorithm
    gsa_algo = GSA(
        objective_func=sphere, 
        pop_size=50, 
        max_iter=100, 
        bounds=bounds, 
        dim=dimensions,
        G0=100.0,
        kbest_initial=1.0
    )
    
    # Run the solver
    results = gsa_algo.solve()
    
    print(f"Algorithm: {results['algorithm']}")
    print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
    print(f"Best Fitness Achieved: {results['best_fitness']:.5e}") 
    print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")

def test_de_rastrigin():
    print("\n--- Testing Differential Evolution on 10D Rastrigin Function ---")
    
    dimensions = 10
    bounds = np.array([[-5.12, 5.12] for _ in range(dimensions)]) 
    
    # Initialize the DE Algorithm
    de_algo = DE(
        objective_func=rastrigin, 
        pop_size=50, 
        max_iter=100, 
        bounds=bounds, 
        dim=dimensions,
        F=0.8,
        CR=0.9
    )
    
    # Run the solver
    results = de_algo.solve()
    
    print(f"Algorithm: {results['algorithm']}")
    print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
    print(f"Best Fitness Achieved: {results['best_fitness']:.5f}") 
    print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")

def test_de_rosenbrock():
    print("\n--- Testing Differential Evolution on 10D Rosenbrock Function ---")
    
    dimensions = 10
    bounds = np.array([[-2.0, 2.0] for _ in range(dimensions)]) 
    
    # Initialize the DE Algorithm
    de_algo = DE(
        objective_func=rosenbrock, 
        pop_size=50, 
        max_iter=100, 
        bounds=bounds, 
        dim=dimensions,
        F=0.8,
        CR=0.9
    )
    
    # Run the solver
    results = de_algo.solve()
    
    print(f"Algorithm: {results['algorithm']}")
    print(f"Execution Time: {results['execution_time_seconds']:.5f} seconds")
    print(f"Best Fitness Achieved: {results['best_fitness']:.5f}") 
    print(f"Global Minimum (first 3 dims): {results['best_solution'][:3]}")

if __name__ == "__main__":
    test_de_sphere()
    test_sa_sphere()
    test_gsa_sphere()
    test_de_rastrigin()
    test_de_rosenbrock()