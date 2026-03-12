#!/usr/bin/env python3
"""
Grid Pathfinding Experiment Driver
==================================

Supports two modes for grid shortest path problems:

1. VISUAL MODE: Small grids (~50x50) with frontier tracking for animation
   - Records node expansion history and frontier snapshots
   - Exports trace files for notebook visualization
   - Suitable for creating expansion animations and path visualizations

2. BENCHMARK MODE: Large grids (up to 1000x1000) for performance analysis
   - Collects computational metrics (nodes expanded, time, path length)
   - Exports CSV files for statistical analysis and plotting
   - Compares algorithms across multiple metrics

Usage:
    python experiments/grid_experiment.py --mode visual --grid-file tests/SP/test_1.txt --algorithms BFS,A* --out trace.pkl
    python experiments/grid_experiment.py --mode benchmark --rows 500 --cols 500 --algorithms BFS,A*,GA_Grid --out results.csv

The driver integrates with config.json for fair algorithm comparisons and supports
multiple test files via auto-discovery (test_*.txt pattern).
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from problems.discrete import SPProblem
from algorithms.classic.uninformed_search import BFS, DFS, UCS
from algorithms.classic.informed_search import GreedyBestFirst, AStarSearch
from algorithms.evolution.GA import GA_Grid
from algorithms.biology.ACOR import ACO_Grid
from utils.configHelper import load_config


class GridExperiment:
    """
    Experiment driver for grid pathfinding algorithms.

    Supports both visual mode (small grids with expansion tracking) and
    benchmark mode (large grids with performance metrics).
    """

    def __init__(self, problem: SPProblem):
        """
        Initialize experiment with a grid problem.

        Args:
            problem: SPProblem instance with grid, start, and end nodes
        """
        self.problem = problem
        self.config = load_config("../utils/config.json")

    @classmethod
    def from_file(cls, filepath: str) -> 'GridExperiment':
        """Create experiment from grid file."""
        problem = SPProblem(filepath)
        return cls(problem)

    @classmethod
    def random(cls, rows: int, cols: Optional[int] = None,
               obstacle_prob: float = 0.25, seed: Optional[int] = None) -> 'GridExperiment':
        """
        Create experiment with random grid.

        Args:
            rows: Number of rows
            cols: Number of columns (defaults to rows for square grid)
            obstacle_prob: Probability of each cell being a wall (0-1)
            seed: Random seed for reproducibility
        """
        if cols is None:
            cols = rows

        if seed is not None:
            np.random.seed(seed)

        # Generate random grid
        grid = np.random.choice([0, 1], size=(rows, cols), p=[1-obstacle_prob, obstacle_prob])

        # Ensure start and end are open
        start = (0, 0)
        end = (rows-1, cols-1)
        grid[start] = 0
        grid[end] = 0

        # Ensure path exists (simple guarantee: clear a diagonal path)
        for i in range(min(rows, cols)):
            grid[i, i] = 0

        problem = SPProblem.from_array(grid, start, end)
        return cls(problem)

    def run(self, algorithm_name: str, record_frontier: bool = False) -> Dict:
        """
        Run a single algorithm on the grid problem.

        Args:
            algorithm_name: Name of algorithm ('BFS', 'A*', etc.)
            record_frontier: Whether to track frontier snapshots for visualization

        Returns:
            Results dictionary from algorithm
        """
        algo_class = self._get_algorithm_class(algorithm_name)

        # Instantiate algorithm with frontier recording option
        if algorithm_name in ['BFS', 'DFS', 'UCS', 'GreedyBestFirst', 'AStarSearch']:
            algo = algo_class(self.problem.grid, self.problem.start_node,
                            self.problem.end_node, record_frontier=record_frontier)
        elif algorithm_name in ['GA_Grid']:
            # Use config for GA parameters
            ga_config = self.config['algorithms']['GA_Grid']
            algo = algo_class(self.problem.grid, self.problem.start_node, self.problem.end_node,
                            pop_size=ga_config['pop_size'], max_iter=ga_config['max_iter'])
        elif algorithm_name in ['ACO_Grid']:
            # Use config for ACO parameters
            aco_config = self.config['algorithms']['ACO_Grid']
            algo = algo_class(self.problem.grid, self.problem.start_node, self.problem.end_node,
                            n_ants=aco_config['n_ants'], max_iterations=aco_config['max_iterations'])
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        # Run algorithm
        algo.solve()
        return algo.get_results()

    def run_all(self, algorithms: List[str], record_frontier: bool = False) -> Dict[str, Dict]:
        """
        Run multiple algorithms and collect results.

        Args:
            algorithms: List of algorithm names
            record_frontier: Whether to track frontiers for visualization

        Returns:
            Dictionary mapping algorithm names to their results
        """
        results = {}
        for algo_name in algorithms:
            print(f"Running {algo_name}...")
            try:
                results[algo_name] = self.run(algo_name, record_frontier)
                print(f"  ✓ {algo_name} completed")
            except Exception as e:
                print(f"  ✗ {algo_name} failed: {e}")
                results[algo_name] = {'error': str(e)}
        return results

    def visualize(self, algorithms: List[str], trace_path: Optional[str] = None) -> Dict:
        """
        Run algorithms in visual mode with frontier tracking.

        Args:
            algorithms: List of algorithm names to run
            trace_path: Optional path to save trace file (.pkl)

        Returns:
            Trace dictionary containing grid, start/end, and algorithm results
        """
        print(f"Running visual experiment on {self.problem.n}x{self.problem.m} grid...")

        # Run algorithms with frontier recording
        results = self.run_all(algorithms, record_frontier=True)

        # Build trace dictionary
        trace = {
            'grid': self.problem.grid.copy(),
            'start_node': self.problem.start_node,
            'end_node': self.problem.end_node,
            'grid_shape': self.problem.grid.shape,
            'algorithms': results,
            'timestamp': time.time(),
            'mode': 'visual'
        }

        # Save trace if requested
        if trace_path:
            self._save_trace(trace, trace_path)
            print(f"Trace saved to: {trace_path}")

        return trace

    def benchmark(self, algorithms: List[str], csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run algorithms in benchmark mode and collect performance metrics.

        Args:
            algorithms: List of algorithm names to run
            csv_path: Optional path to save CSV results

        Returns:
            DataFrame with algorithm performance metrics
        """
        print(f"Running benchmark on {self.problem.n}x{self.problem.m} grid...")

        # Run algorithms without frontier recording (faster for large grids)
        results = self.run_all(algorithms, record_frontier=False)

        # Convert to DataFrame
        records = []
        for algo_name, result in results.items():
            if 'error' in result:
                record = {
                    'algorithm': algo_name,
                    'error': result['error'],
                    'best_fitness': np.nan,
                    'execution_time_seconds': np.nan,
                    'nodes_expanded': np.nan,
                    'path_length': np.nan
                }
            else:
                record = {
                    'algorithm': algo_name,
                    'best_fitness': result.get('best_fitness', np.nan),
                    'execution_time_seconds': result.get('execution_time_seconds', np.nan),
                    'nodes_expanded': result.get('nodes_expanded', 0),
                    'path_length': result.get('path_length', 0)
                }

                # Add metaheuristic-specific metrics
                if 'convergence_curve' in result:
                    record['convergence_final'] = result['convergence_curve'][-1] if len(result['convergence_curve']) > 0 else np.nan
                    record['iterations'] = len(result['convergence_curve'])

            records.append(record)

        df = pd.DataFrame(records)

        # Add computed metrics
        if 'best_fitness' in df.columns:
            # Find optimal solution (minimum path length)
            optimal = df['best_fitness'].min()
            df['optimality_gap'] = df['best_fitness'] - optimal
            df['optimality_gap_pct'] = (df['optimality_gap'] / optimal * 100).round(2)

        # Save CSV if requested
        if csv_path:
            df.to_csv(csv_path, index=False)
            print(f"Results saved to: {csv_path}")

        # Print summary
        self._print_benchmark_summary(df)

        return df

    def _get_algorithm_class(self, name: str):
        """Get algorithm class by name."""
        algorithms = {
            'BFS': BFS,
            'DFS': DFS,
            'UCS': UCS,
            'GreedyBestFirst': GreedyBestFirst,
            'AStarSearch': AStarSearch,
            'GA_Grid': GA_Grid,
            'ACO_Grid': ACO_Grid
        }
        if name not in algorithms:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")
        return algorithms[name]

    def _save_trace(self, trace: Dict, filepath: str):
        """Save trace dictionary to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(trace, f)

    def _print_benchmark_summary(self, df: pd.DataFrame):
        """Print benchmark results summary."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        print(f"Grid Size: {self.problem.n} x {self.problem.m}")
        print(f"Start: {self.problem.start_node}, Goal: {self.problem.end_node}")
        print()

        # Filter out failed algorithms
        valid_df = df.dropna(subset=['best_fitness'])

        if len(valid_df) == 0:
            print("No algorithms completed successfully.")
            return

        # Sort by path length
        sorted_df = valid_df.sort_values('best_fitness')

        print("Algorithm Performance (sorted by path length):")
        print("-" * 80)
        for _, row in sorted_df.iterrows():
            gap_str = ".0f" if row['optimality_gap'] == 0 else ".1f"
            print("20"
                  "8.0f"
                  "10.4f"
                  "12.0f"
                  "12.0f")

        print("\nNotes:")
        print("- Path length = number of steps in solution")
        print("- Optimality gap = difference from best solution")
        print("- Lower nodes expanded = better computational efficiency")
        print("- A* provides the gold standard: optimal solution with minimal exploration")


def main():
    """Command-line interface for grid experiments."""
    parser = argparse.ArgumentParser(
        description="Grid Pathfinding Experiment Driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visual mode with file
  python experiments/grid_experiment.py --mode visual --grid-file tests/SP/test_1.txt --algorithms BFS,A* --out trace.pkl

  # Visual mode with random grid
  python experiments/grid_experiment.py --mode visual --rows 20 --cols 20 --algorithms BFS,A*,GreedyBestFirst --out trace.pkl

  # Benchmark mode with file
  python experiments/grid_experiment.py --mode benchmark --grid-file tests/SP/test_1.txt --algorithms BFS,A*,GA_Grid --out results.csv

  # Benchmark mode with random large grid
  python experiments/grid_experiment.py --mode benchmark --rows 500 --cols 500 --algorithms BFS,A*,GA_Grid --out results.csv
        """
    )

    # Mode selection
    parser.add_argument('--mode', choices=['visual', 'benchmark'], required=True,
                       help='Experiment mode: visual (small grids) or benchmark (large grids)')

    # Grid specification
    grid_group = parser.add_mutually_exclusive_group(required=True)
    grid_group.add_argument('--grid-file', type=str,
                           help='Path to grid file (e.g., tests/SP/test_1.txt)')
    grid_group.add_argument('--rows', type=int,
                           help='Number of rows for random grid')
    parser.add_argument('--cols', type=int,
                       help='Number of columns for random grid (defaults to rows)')
    parser.add_argument('--obstacle-prob', type=float, default=0.25,
                       help='Obstacle probability for random grids (0-1, default: 0.25)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible grids')

    # Algorithm selection
    parser.add_argument('--algorithms', type=str, required=True,
                       help='Comma-separated list of algorithms (e.g., BFS,A*,GA_Grid)')

    # Output
    parser.add_argument('--out', type=str,
                       help='Output file path (.pkl for visual, .csv for benchmark)')

    args = parser.parse_args()

    # Parse algorithms
    algorithms = [a.strip() for a in args.algorithms.split(',')]

    # Create experiment
    if args.grid_file:
        print(f"Loading grid from file: {args.grid_file}")
        experiment = GridExperiment.from_file(args.grid_file)
    else:
        print(f"Generating random {args.rows}x{args.cols} grid (obstacle prob: {args.obstacle_prob})")
        experiment = GridExperiment.random(
            rows=args.rows,
            cols=args.cols,
            obstacle_prob=args.obstacle_prob,
            seed=args.seed
        )

    # Run experiment based on mode
    if args.mode == 'visual':
        if not args.out:
            args.out = f"grid_visual_{experiment.problem.n}x{experiment.problem.m}_{int(time.time())}.pkl"
        trace = experiment.visualize(algorithms, args.out)
        print(f"\nVisual trace saved. Load in notebook for animation:")
        print(f"  from notebooks.grid_visualization_example import load_trace, create_grid_animation")
        print(f"  trace = load_trace('{args.out}')")
        print(f"  anim = create_grid_animation(trace, '{algorithms[0]}')")

    elif args.mode == 'benchmark':
        if not args.out:
            args.out = f"grid_benchmark_{experiment.problem.n}x{experiment.problem.m}_{int(time.time())}.csv"
        df = experiment.benchmark(algorithms, args.out)
        print(f"\nBenchmark complete. Results saved to CSV for analysis.")


if __name__ == "__main__":
    main()