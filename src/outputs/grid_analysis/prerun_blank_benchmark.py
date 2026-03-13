import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(r"D:\Uni\IT\IntroToAI\AI1_search-algorithms\src")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.grid_experiment import GridExperiment
from problems.discrete import SPProblem

CASE_NAME = "blank"
CASE_PATH = Path(r"D:\Uni\IT\IntroToAI\AI1_search-algorithms\src\tests\SP\blank")
ALGORITHMS = ['DFS', 'BFS', 'UCS', 'GreedyBestFirst', 'AStarSearch']
OUTPUT_DIR = Path(r"D:\Uni\IT\IntroToAI\AI1_search-algorithms\src\outputs\grid_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running benchmarks for {CASE_NAME}...")
print(f"Case path: {CASE_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# Find all test files (test_0.txt, test_1.txt, etc.)
test_files = sorted(CASE_PATH.glob('test_*.txt'))

if not test_files:
    print(f"ERROR: No test files found in {CASE_PATH}")
    sys.exit(1)

print(f"Found {len(test_files)} test files")

results = []

for i, test_file in enumerate(test_files, 1):
    print(f"\n[{i}/{len(test_files)}] Processing {test_file.name}...", end=' ', flush=True)
    try:
        problem = SPProblem(str(test_file))
        experiment = GridExperiment(problem)
        
        # Run benchmark (returns DataFrame with results)
        benchmark_df = experiment.benchmark(ALGORITHMS, csv_path=None)
        
        # Add case name and test file info
        benchmark_df['case'] = CASE_NAME
        benchmark_df['test_file'] = test_file.name
        
        results.append(benchmark_df)
        print("✓")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        continue

if results:
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Save CSV
    csv_path = OUTPUT_DIR / f"{CASE_NAME}_benchmark.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved benchmark results: {csv_path}")
    print(f"    Total tests processed: {len(results)}")
    print(f"    Total algorithm runs: {len(combined_df)}")
    
    # Save pickle for debugging
    pkl_path = OUTPUT_DIR / f"{CASE_NAME}_benchmark.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(combined_df, f)
    print(f"[OK] Saved pickle: {pkl_path}")
else:
    print("[ERROR] No results collected")
    sys.exit(1)
