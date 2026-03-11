"""Quick test of GA_Grid and ACO_Grid integration."""

from problems.discrete import SPProblem
from algorithms.evolution.GA import GA_Grid
from algorithms.biology.ACOR import ACO_Grid

prob = SPProblem('tests/SP/test_1.txt')
print(f"Testing GA_Grid on {prob.n}x{prob.m} grid...")
ga = GA_Grid(prob.grid, prob.start_node, prob.end_node, pop_size=20, max_iter=50)
ga.solve()
res = ga.get_results()
print(f"  Best fitness: {res['best_fitness']}")
print(f"  Has convergence_curve: {'convergence_curve' in res}")
print(f"  Execution time: {res['execution_time_seconds']:.4f}s\n")

print(f"Testing ACO_Grid on {prob.n}x{prob.m} grid...")
aco = ACO_Grid(prob.grid, prob.start_node, prob.end_node, n_ants=15, max_iterations=50)
aco.solve()
res = aco.get_results()
print(f"  Best fitness: {res['best_fitness']}")
print(f"  Has convergence_curve: {'convergence_curve' in res}")
print(f"  Execution time: {res['execution_time_seconds']:.4f}s")
