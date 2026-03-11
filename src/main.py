import sys
import numpy as np
from problems.discrete import TSPProblem
from algorithms.human.ca import CA_TSP
from algorithms.evolution.GA import GA_TSP
from algorithms.physics.SA import SA_TSP
from algorithms.biology.ACOR import ACO_TSP
from algorithms.classic.uninformed_search import BFS_TSP, DFS_TSP
from algorithms.classic.informed_search import A_STAR_TSP
from algorithms.classic.local_search import HillClimbing_TSP

def test_all():
    # Load test instance - use smaller instance for exhaustive search methods
    prob = TSPProblem('tests/TSP/test_1.txt')  # 6 cities - good for BFS/DFS/A*
    print(f"\n{'='*100}")
    print(f"TSP Test Instance: {prob.n_cities} cities")
    print(f"{'='*100}\n")

    results = []
    
    # Test CA_TSP
    print("[1/8] CA_TSP...")
    ca = CA_TSP(prob.distance_matrix, pop_size=30, max_iter=50, seed=42)
    ca.solve()
    r_ca = ca.get_results()
    print(f"  Best fitness: {r_ca['best_fitness']:.2f}")
    print(f"  Time: {r_ca['execution_time_seconds']:.3f}s\n")
    results.append(("CA_TSP", r_ca['best_fitness']))

    # Test GA_TSP
    print("[2/8] GA_TSP...")
    ga = GA_TSP(prob.distance_matrix, pop_size=30, max_iter=50, seed=42)
    ga.solve()
    print(f"  Best fitness: {ga.best_fitness:.2f}")
    print(f"  Time: {ga.execution_time:.3f}s\n")
    results.append(("GA_TSP", ga.best_fitness))

    # Test SA_TSP
    print("[3/8] SA_TSP...")
    sa = SA_TSP(prob.distance_matrix, initial_temperature=1000, 
                cooling_rate=0.99, max_iterations=200, seed=42)
    sa.solve()
    r_sa = sa.get_results()
    print(f"  Best fitness: {r_sa['best_fitness']:.2f}")
    print(f"  Time: {r_sa['execution_time_seconds']:.3f}s\n")
    results.append(("SA_TSP", r_sa['best_fitness']))

    # Test ACO_TSP
    print("[4/8] ACO_TSP...")
    aco = ACO_TSP(prob.distance_matrix, n_ants=20, max_iterations=50, seed=42)
    aco.solve()
    r_aco = aco.get_results()
    print(f"  Best fitness: {r_aco['best_fitness']:.2f}")
    print(f"  Time: {r_aco['execution_time_seconds']:.3f}s\n")
    results.append(("ACO_TSP", r_aco['best_fitness']))

    # Test HillClimbing_TSP (fast, good baseline)
    print("[5/8] HillClimbing_TSP...")
    hc = HillClimbing_TSP(prob.distance_matrix, max_restarts=20, seed=42)
    hc.solve()
    r_hc = hc.get_results()
    print(f"  Best fitness: {r_hc['best_fitness']:.2f}")
    print(f"  Time: {r_hc['execution_time_seconds']:.3f}s\n")
    results.append(("HillClimbing_TSP", r_hc['best_fitness']))

    # Test A*_TSP (small scale only - n=6 is fine)
    print("[6/8] A*_TSP (small scale n <= 20)...")
    astar = A_STAR_TSP(prob.distance_matrix, seed=42)
    astar.solve()
    r_astar = astar.get_results()
    print(f"  Best fitness: {r_astar['best_fitness']:.2f}")
    print(f"  Time: {r_astar['execution_time_seconds']:.3f}s")
    print(f"  Nodes expanded: {r_astar['nodes_expanded']}\n")
    results.append(("A*_TSP", r_astar['best_fitness']))

    # Test BFS_TSP (small scale only - n=6 is fine)
    print("[7/8] BFS_TSP (small scale n <= 10)...")
    bfs = BFS_TSP(prob.distance_matrix, seed=42)
    bfs.solve()
    r_bfs = bfs.get_results()
    print(f"  Best fitness: {r_bfs['best_fitness']:.2f}")
    print(f"  Time: {r_bfs['execution_time_seconds']:.3f}s")
    print(f"  Nodes expanded: {r_bfs['nodes_expanded']}\n")
    results.append(("BFS_TSP", r_bfs['best_fitness']))

    # Test DFS_TSP (small scale only - n=6 is fine)
    print("[8/8] DFS_TSP (small scale n <= 10)...")
    dfs = DFS_TSP(prob.distance_matrix, seed=42)
    dfs.solve()
    r_dfs = dfs.get_results()
    print(f"  Best fitness: {r_dfs['best_fitness']:.2f}")
    print(f"  Time: {r_dfs['execution_time_seconds']:.3f}s")
    print(f"  Nodes expanded: {r_dfs['nodes_expanded']}\n")
    results.append(("DFS_TSP", r_dfs['best_fitness']))

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY (sorted by solution quality)")
    print(f"{'='*100}")
    results.sort(key=lambda x: x[1])
    for i, (name, fitness) in enumerate(results, 1):
        print(f"{i}. {name:20s} : {fitness:.2f}")

if __name__ == "__main__":
    test_all()
