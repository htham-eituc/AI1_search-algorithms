import csv
import numpy as np
from pathlib import Path
from utils.configHelper import load_config, run_all_experiments, print_summary
from utils.discreteHelper import (
    load_discrete_problem, 
    get_available_tests,
    print_tsp_info,
    print_sp_info,
    evaluate_tsp_solution,
    evaluate_sp_solution
)

if __name__ == "__main__":
    # Get config path (relative to script location)
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    np.random.seed(config["experiment"]["seed"])
    
    print("\n" + "="*100)
    print("COMPREHENSIVE ALGORITHM TEST SUITE")
    print("="*100)
    print("""
Testing algorithms on benchmark functions:
    
1. SPHERE (Unimodal, Convex):
   - Classic: Hill Climbing, SA
   - Evolution: GA, DE
   - Physics: GSA
   - Biology: PSO
   - Human: TLBO

2. RASTRIGIN (Highly Multimodal):
   - Classic: Hill Climbing (Expected to fail), SA
   - Evolution: GA, DE
   - Biology: ABC, FA, CS
   - Human: CA

3. ROSENBROCK (Narrow Valley):
   - Classic: Hill Climbing, SA
   - Evolution: GA, DE
   - Biology: ABC, PSO
   - Human: SFO, TLBO
""")
    print("="*100 + "\n")
    
    # Run all experiments
    results_summary = run_all_experiments(config)
    
    # Print summary
    print_summary(results_summary)

    # Save results to CSV
    csv_path = script_dir / "results.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        fieldnames = ["algorithm", "problem", "dimensions", "best_fitness", "avg_fitness", "diversity", "execution_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_summary:
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")
    
    # ========================================================================
    #  DISCRETE OPTIMIZATION PROBLEMS (Optional)
    # ========================================================================
    
    if config["experiment"].get("test_discrete", False):
        print("\n\n" + "="*100)
        print("DISCRETE OPTIMIZATION TEST SUITE")
        print("="*100)
        print("""
Testing algorithms on discrete problems:

4. TRAVELING SALESMAN (TSP) - Discrete, Combinatorial Route:
   - Classic: Hill Climbing (2-opt), SA
   - Evolution: GA
   - Biology: ACO

5. SHORTEST PATH (SP) - Discrete, Exact Pathfinding:
   - Classic: BFS, DFS, UCS, Greedy Best-First, A*
   - Evolution: GA (Adapted for grid)
   - Biology: ACO (Adapted for grid)
""")
        print("="*100 + "\n")
        
        # Test TSP
        print("\n[TESTING TSP PROBLEMS]")
        print("-"*100)
        tsp_tests = get_available_tests("TSP")
        
        if not tsp_tests:
            print("WARNING: No TSP test files found in tests/TSP/")
        else:
            tsp_results = []
            for test_name in tsp_tests[:3]:  # Test first 3 instances
                try:
                    tsp = load_discrete_problem("TSP", test_name)
                    print_tsp_info(tsp)
                    
                    # Evaluate a random solution
                    random_solution = np.random.permutation(tsp.n_cities)
                    distance = evaluate_tsp_solution(tsp, random_solution)
                    print(f"  Random solution distance: {distance:.2f}")
                    
                    tsp_results.append({
                        "problem": "TSP",
                        "test": test_name,
                        "cities": tsp.n_cities,
                        "random_distance": distance
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
        
        # Test SP
        print("\n[TESTING SP PROBLEMS]")
        print("-"*100)
        sp_tests = get_available_tests("SP")
        
        if not sp_tests:
            print("WARNING: No SP test files found in tests/SP/")
        else:
            sp_results = []
            for test_name in sp_tests[:3]:  # Test first 3 instances
                try:
                    sp = load_discrete_problem("SP", test_name)
                    print_sp_info(sp)
                    
                    # Evaluate direct path if exists
                    direct_distance = sp.adjacency_matrix[sp.start_node, sp.end_node]
                    if direct_distance != np.inf:
                        print(f"  Direct path distance: {direct_distance:.2f}")
                    else:
                        print(f"  No direct edge between start and end")
                    
                    sp_results.append({
                        "problem": "SP",
                        "test": test_name,
                        "nodes": sp.n_nodes,
                        "direct_distance": direct_distance if direct_distance != np.inf else None
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
        
        print("\n" + "="*100)
        print("NOTE: Full algorithm integration for discrete problems (ACO, Graph Search)")
        print("      requires implementing specialized algorithm variants.")
        print("="*100)

