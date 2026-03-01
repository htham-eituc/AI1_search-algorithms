import csv
import numpy as np
from pathlib import Path
from utils.configHelper import load_config, run_all_experiments, print_summary

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
    
    # ========================================================================
    #  CONTINUOUS OPTIMIZATION PROBLEMS
    # ========================================================================
    
    if config["experiment"].get("test_continuous", True):
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
    else:
        print("Continuous optimization tests disabled in config.\n")

