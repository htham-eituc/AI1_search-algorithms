import numpy as np
from problems.continuous import sphere
from algorithms.evolution.DE import DE

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

if __name__ == "__main__":
    test_de_sphere()