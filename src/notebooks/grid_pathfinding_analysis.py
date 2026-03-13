# Grid Pathfinding Visualization & Benchmark Analysis
# =====================================================
# Fixed for ACTUAL project structure (NO src/ directory)
# Structure:
#   D:\Uni\IT\IntroToAI\AI1_search-algorithms\
#   ├── algorithms/
#   ├── experiments/
#   ├── notebooks/
#   ├── problems/
#   ├── tests/
#   └── utils/

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple
import time

# ============================================================================
# PATH SETUP (CORRECTED FOR ACTUAL STRUCTURE - NO src/)
# ============================================================================

FILE_PATH = Path(__file__).resolve()

print("\n" + "=" * 80)
print("PATH DETECTION")
print("=" * 80)
print(f"Script location: {FILE_PATH}")

# ACTUAL STRUCTURE (NO src/ directory):
# If script is in: D:\...\AI1_search-algorithms\notebooks\grid_pathfinding_analysis.py
# Then parent is: D:\...\AI1_search-algorithms\
# That parent is PROJECT_ROOT

# Strategy: Go up to parent, verify key directories exist
PROJECT_ROOT = None
current = FILE_PATH.parent

# Check up to 3 levels up for the project root
for i in range(3):
    if (current / 'algorithms').exists() and (current / 'experiments').exists() and (current / 'problems').exists():
        PROJECT_ROOT = current
        print(f"✓ Found project root at: {current}")
        break
    current = current.parent

if PROJECT_ROOT is None:
    print("[ERROR] Could not find project root!")
    print("Expected to find: algorithms/, experiments/, problems/ directories")
    sys.exit(1)

# Set paths (NO src/ directory)
SRC_PATH = PROJECT_ROOT  # The root itself is the source path
TESTS_PATH = PROJECT_ROOT / "tests"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root     : {PROJECT_ROOT}")
print(f"Source path      : {SRC_PATH}")
print(f"Tests path       : {TESTS_PATH}")
print(f"Outputs path     : {OUTPUTS_PATH}")

# Verify critical directories
print("\nDirectory verification:")
for dirname in ['algorithms', 'experiments', 'problems', 'tests']:
    dirpath = PROJECT_ROOT / dirname
    status = "✓" if dirpath.exists() else "✗"
    print(f"  {status} {dirname:20s} {dirpath}")

# Import custom modules
print("\nImporting modules...")
try:
    from experiments.grid_experiment import GridExperiment
    print("  ✓ GridExperiment")
except ImportError as e:
    print(f"  ✗ GridExperiment: {e}")
    sys.exit(1)

try:
    from problems.discrete import SPProblem
    print("  ✓ SPProblem")
except ImportError as e:
    print(f"  ✗ SPProblem: {e}")
    sys.exit(1)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)

print("\n" + "=" * 80)
print("GRID PATHFINDING ANALYSIS - 5 Test Cases")
print("=" * 80)

# ============================================================================
# CONFIGURATION (CORRECTED TEST PATHS)
# ============================================================================

TEST_CASES = {
    'blank': TESTS_PATH / 'SP/blank',
    'maze_loops': TESTS_PATH / 'SP/maze_loops',
    'obstacles': TESTS_PATH / 'SP/obstacles',
    'obstacles_dense': TESTS_PATH / 'SP/obstacles_dense',  # FIXED: was 'dense'
    'perfect_maze': TESTS_PATH / 'SP/perfect_maze'
}

# Algorithms to test (excluding metaheuristics for faster runs)
ALGORITHMS = ['DFS', 'BFS', 'UCS', 'GreedyBestFirst', 'AStarSearch']
ALGORITHMS_DISPLAY = ['DFS', 'BFS', 'Uniform Cost', 'Greedy Best-First', 'A*']

# Verify algorithm count matches display names
if len(ALGORITHMS) != len(ALGORITHMS_DISPLAY):
    print(f"[WARNING] Algorithm count mismatch: {len(ALGORITHMS)} != {len(ALGORITHMS_DISPLAY)}")
    ALGORITHMS_DISPLAY = ALGORITHMS

# Output directory
OUTPUT_DIR = OUTPUTS_PATH / "grid_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nTest case directories:")
for case_name, case_path in TEST_CASES.items():
    status = "✓" if case_path.exists() else "✗"
    print(f"  {status} {case_name:20s} {case_path}")

print(f"\n[OK] Output directory: {OUTPUT_DIR}")
print(f"[OK] Algorithms: {ALGORITHMS}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def display_grid(grid, start_node, end_node, title="Grid Problem", figsize=(8, 8)):
    """Display grid with start and end nodes."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create display grid (RGB)
    display = np.zeros((*grid.shape, 3))
    display[grid == 1] = [0.2, 0.2, 0.2]  # Obstacles in dark gray
    display[grid == 0] = [1, 1, 1]         # Open cells in white
    
    # Mark start and end
    display[start_node] = [0, 1, 0]  # Start in green
    display[end_node] = [1, 0, 0]    # End in red
    
    ax.imshow(display)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0, 1, 0], label='Start'),
        Patch(facecolor=[1, 0, 0], label='End'),
        Patch(facecolor=[0.2, 0.2, 0.2], label='Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_path_on_grid(grid, path, start_node, end_node, title="Path Visualization", figsize=(10, 10)):
    """Plot a single path on the grid."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create display grid
    display = np.zeros((*grid.shape, 3))
    display[grid == 1] = [0.2, 0.2, 0.2]  # Obstacles in dark gray
    display[grid == 0] = [1, 1, 1]         # Open cells in white
    
    # Mark path
    if path and len(path) > 0:
        for node in path[1:-1]:  # Exclude start and end
            display[node] = [0.5, 0.5, 1]  # Path in light blue
    
    # Mark start and end
    display[start_node] = [0, 1, 0]  # Start in green
    display[end_node] = [1, 0, 0]    # End in red
    
    ax.imshow(display)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    if path:
        ax.text(0.02, 0.02, f'Path length: {len(path)}', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def compare_paths(trace, algorithms, case_name, figsize=(16, 12)):
    """Compare paths from multiple algorithms."""
    grid = trace['problem_grid']
    start_node = trace['start_node']
    end_node = trace['end_node']
    
    n_algos = len(algorithms)
    n_cols = 3
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        
        if algo not in trace['algorithms']:
            ax.text(0.5, 0.5, f'{algo}\nNo data', ha='center', va='center')
            ax.set_title(f'{algo}', fontweight='bold')
            ax.axis('off')
            continue
        
        result = trace['algorithms'][algo]
        path = result.get('path', [])
        
        # Create display grid
        display = np.zeros((*grid.shape, 3))
        display[grid == 1] = [0.2, 0.2, 0.2]
        display[grid == 0] = [1, 1, 1]
        
        # Mark path
        if path and len(path) > 0:
            for node in path[1:-1]:
                display[node] = [0.5, 0.5, 1]
        
        display[start_node] = [0, 1, 0]
        display[end_node] = [1, 0, 0]
        
        ax.imshow(display)
        
        path_len = len(path) if path else 'Failed'
        nodes_exp = result.get('nodes_expanded', 'N/A')
        time_taken = result.get('execution_time_seconds', 'N/A')
        
        # Safe time formatting
        if isinstance(time_taken, (int, float)):
            time_str = f"{time_taken:.3f}s"
        else:
            time_str = str(time_taken)
        
        title = f'{algo}\nPath: {path_len} | Nodes: {nodes_exp} | Time: {time_str}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(algorithms), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Stage 1 - Path Comparison ({case_name})', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


# ============================================================================
# STAGE 1: VISUALIZATION (test_0.txt only)
# ============================================================================

def stage1_visualization(case_name, case_path):
    """
    Stage 1: Visualize smallest test with grid display and path comparison.
    """
    print(f"\n{'=' * 80}")
    print(f"CASE: {case_name.upper()}")
    print(f"{'=' * 80}")
    print(f"\n[STAGE 1] Visualization (test_0.txt)")
    print("-" * 80)
    
    test_file = Path(case_path) / 'test_0.txt'
    
    if not test_file.exists():
        print(f"[ERROR] Test file not found: {test_file}")
        return None
    
    try:
        # Change to project root before loading problem to fix relative config paths
        original_cwd = os.getcwd()
        try:
            os.chdir(str(PROJECT_ROOT))
            problem = SPProblem(str(test_file))
        finally:
            os.chdir(original_cwd)
        
        print(f"[OK] Problem loaded: {problem.n} × {problem.m} grid")
        print(f"  Start: {problem.start_node}, End: {problem.end_node}")
        print(f"  Obstacles: {np.sum(problem.grid == 1)}, Open: {np.sum(problem.grid == 0)}")
    except Exception as e:
        print(f"[ERROR] Error loading problem: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Display grid
    print(f"\n-> Displaying grid...")
    try:
        fig_grid = display_grid(
            problem.grid,
            start_node=problem.start_node,
            end_node=problem.end_node,
            title=f"Grid Problem - {case_name}"
        )
        plt.savefig(OUTPUT_DIR / f'{case_name}_grid.png', dpi=150, bbox_inches='tight')
        plt.close(fig_grid)
        print(f"  ✓ Saved: {OUTPUT_DIR / f'{case_name}_grid.png'}")
    except Exception as e:
        print(f"  ✗ Error saving grid: {e}")
    
    # Run algorithms
    print(f"\n-> Running algorithms...")
    
    # Change to project root before creating experiment to fix relative path issues
    original_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT))
        experiment = GridExperiment(problem)
    finally:
        os.chdir(original_cwd)
    
    start_time = time.time()
    trace = None
    try:
        trace = experiment.visualize(ALGORITHMS, trace_path=None)
        elapsed = time.time() - start_time
        print(f"[OK] Algorithms completed in {elapsed:.2f}s")
        
        # Display results summary
        print(f"\n  Results Summary:")
        for algo_name in ALGORITHMS:
            if algo_name in trace['algorithms']:
                result = trace['algorithms'][algo_name]
                path_len = result.get('path_length', 'N/A')
                nodes_exp = result.get('nodes_expanded', 'N/A')
                time_t = result.get('execution_time_seconds', 'N/A')
                if isinstance(time_t, (int, float)):
                    time_str = f"{time_t:.4f}s"
                else:
                    time_str = str(time_t)
                print(f"    {algo_name:20s} - Path: {path_len:4} | Nodes: {nodes_exp:6} | Time: {time_str}")
    except Exception as e:
        print(f"[ERROR] Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if trace is None:
        return None
    
    # Plot path comparison
    print(f"\n-> Creating path comparison plot...")
    try:
        fig_paths = compare_paths(trace, ALGORITHMS, case_name)
        plt.savefig(OUTPUT_DIR / f'{case_name}_paths.png', dpi=150, bbox_inches='tight')
        plt.close(fig_paths)
        print(f"  ✓ Saved: {OUTPUT_DIR / f'{case_name}_paths.png'}")
    except Exception as e:
        print(f"  ✗ Error saving paths plot: {e}")
    
    # Save trace for later reference
    try:
        trace_file = OUTPUT_DIR / f'{case_name}_stage1_trace.pkl'
        with open(trace_file, 'wb') as f:
            pickle.dump(trace, f)
        print(f"  ✓ Saved trace: {trace_file}")
    except Exception as e:
        print(f"  ✗ Error saving trace: {e}")
    
    return trace


# ============================================================================
# PRE-RUN BENCHMARK SCRIPT (for Stage 2)
# ============================================================================

def prerun_benchmark_script(case_name, case_path):
    """
    Generate a standalone script to pre-run benchmarks for all test files.
    This script can be run separately to generate .csv and .pkl files.
    """
    # Convert to absolute paths for the script
    case_path_str = str(Path(case_path).resolve())
    output_dir_str = str(OUTPUT_DIR.resolve())
    project_root_str = str(PROJECT_ROOT.resolve())
    
    script_content = f'''#!/usr/bin/env python
"""
Pre-run benchmark for {case_name} test case.
Generated script - run independently to generate benchmark results.
Structure (NO src/):
  D:\\Uni\\IT\\IntroToAI\\AI1_search-algorithms\\
  ├── algorithms/
  ├── experiments/
  ├── problems/
  └── tests/
"""

import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(r"{project_root_str}")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.grid_experiment import GridExperiment
from problems.discrete import SPProblem

CASE_NAME = "{case_name}"
CASE_PATH = Path(r"{case_path_str}")
ALGORITHMS = {ALGORITHMS}
OUTPUT_DIR = Path(r"{output_dir_str}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running benchmarks for {{CASE_NAME}}...")
print(f"Case path: {{CASE_PATH}}")
print(f"Output directory: {{OUTPUT_DIR}}")

# Find all test files (test_0.txt, test_1.txt, etc.)
test_files = sorted(CASE_PATH.glob('test_*.txt'))

if not test_files:
    print(f"ERROR: No test files found in {{CASE_PATH}}")
    sys.exit(1)

print(f"Found {{len(test_files)}} test files")

results = []

for i, test_file in enumerate(test_files, 1):
    print(f"\\n[{{i}}/{{len(test_files)}}] Processing {{test_file.name}}...", end=' ', flush=True)
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
        print(f"✗ Error: {{e}}")
        continue

if results:
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Save CSV
    csv_path = OUTPUT_DIR / f"{{CASE_NAME}}_benchmark.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\\n[OK] Saved benchmark results: {{csv_path}}")
    print(f"    Total tests processed: {{len(results)}}")
    print(f"    Total algorithm runs: {{len(combined_df)}}")
    
    # Save pickle for debugging
    pkl_path = OUTPUT_DIR / f"{{CASE_NAME}}_benchmark.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(combined_df, f)
    print(f"[OK] Saved pickle: {{pkl_path}}")
else:
    print("[ERROR] No results collected")
    sys.exit(1)
'''
    
    script_path = OUTPUT_DIR / f'prerun_{case_name}_benchmark.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"[OK] Generated pre-run script: {script_path}")
    return script_path


# ============================================================================
# STAGE 2: BENCHMARK ANALYSIS
# ============================================================================

def stage2_benchmark_analysis(case_name, case_path):
    """
    Stage 2: Load and analyze pre-computed benchmark results.
    """
    print(f"\n[STAGE 2] Benchmark Analysis (test_1.txt to test_N)")
    print("-" * 80)
    
    csv_path = OUTPUT_DIR / f'{case_name}_benchmark.csv'
    
    if not csv_path.exists():
        print(f"[WARNING] Benchmark CSV not found: {csv_path}")
        print(f"  To generate real results, run:")
        print(f"    python {OUTPUT_DIR / f'prerun_{case_name}_benchmark.py'}")
        print(f"\n  Generating mock data for demonstration...\n")
        
        benchmark_df = generate_mock_benchmark(case_name, case_path)
    else:
        benchmark_df = pd.read_csv(csv_path)
        print(f"[OK] Loaded benchmark: {len(benchmark_df)} results")
    
    if benchmark_df is None or len(benchmark_df) == 0:
        print(f"[SKIP] No benchmark data available")
        return None
    
    # Ensure required columns exist
    required_cols = ['algorithm', 'best_fitness', 'nodes_expanded', 'execution_time_seconds']
    missing_cols = [col for col in required_cols if col not in benchmark_df.columns]
    if missing_cols:
        print(f"[WARNING] Missing columns: {missing_cols}")
    
    # Analysis 1: Path Length Performance
    print(f"\n-> Analyzing path length performance...")
    try:
        fig_pathlen = analyze_path_length(benchmark_df, case_name)
        plt.savefig(OUTPUT_DIR / f'{case_name}_pathlen_performance.png', dpi=150, bbox_inches='tight')
        plt.close(fig_pathlen)
        print(f"  ✓ Saved: {OUTPUT_DIR / f'{case_name}_pathlen_performance.png'}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Analysis 2: Execution Time
    print(f"\n-> Analyzing execution time...")
    try:
        fig_time = analyze_execution_time(benchmark_df, case_name)
        plt.savefig(OUTPUT_DIR / f'{case_name}_execution_time.png', dpi=150, bbox_inches='tight')
        plt.close(fig_time)
        print(f"  ✓ Saved: {OUTPUT_DIR / f'{case_name}_execution_time.png'}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Analysis 3: Nodes Expanded
    print(f"\n-> Analyzing nodes expanded...")
    try:
        fig_nodes = analyze_nodes_expanded(benchmark_df, case_name)
        plt.savefig(OUTPUT_DIR / f'{case_name}_nodes_expanded.png', dpi=150, bbox_inches='tight')
        plt.close(fig_nodes)
        print(f"  ✓ Saved: {OUTPUT_DIR / f'{case_name}_nodes_expanded.png'}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Analysis 4: Summary Statistics
    print(f"\n-> Summary Statistics:")
    print_summary_stats(benchmark_df, case_name)
    
    return benchmark_df


def generate_mock_benchmark(case_name, case_path):
    """Generate mock benchmark data for demonstration."""
    test_dir = Path(case_path)
    test_files = sorted(test_dir.glob('test_*.txt'))
    
    if not test_files:
        print(f"  [ERROR] No test files found in {test_dir}")
        return None
    
    results = []
    np.random.seed(42)
    
    for test_file in test_files[1:]:  # Skip test_0 (used in Stage 1)
        try:
            # Change to project root before loading problem to fix relative paths
            original_cwd = os.getcwd()
            try:
                os.chdir(str(PROJECT_ROOT))
                problem = SPProblem(str(test_file))
            finally:
                os.chdir(original_cwd)
            
            grid_size = problem.n * problem.m
            
            for algo in ALGORITHMS:
                # Simulate results with realistic patterns
                path_len = grid_size // 2 + np.random.randint(-20, 50)
                
                # Adjust by algorithm
                if algo == 'AStarSearch':
                    path_len = int(path_len * 0.95)
                elif algo == 'GreedyBestFirst':
                    path_len = int(path_len * 0.98)
                elif algo == 'DFS':
                    path_len = int(path_len * 1.3)
                elif algo == 'UCS':
                    path_len = int(path_len * 1.05)
                
                results.append({
                    'algorithm': algo,
                    'test_file': test_file.name,
                    'best_fitness': max(path_len, 1),
                    'nodes_expanded': np.random.randint(int(grid_size * 0.1), int(grid_size * 0.8)),
                    'execution_time_seconds': np.random.uniform(0.001, 0.5)
                })
        except Exception as e:
            print(f"    [DEBUG] {test_file.name}: {e}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        print(f"  ✓ Generated mock data: {len(df)} results from {len(test_files)-1} tests\n")
        return df
    
    return None


def analyze_path_length(df, case_name):
    """Analyze path length performance - % difference from best."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    grouped = df.groupby('algorithm')['best_fitness'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    best_path = grouped['mean'].min()
    grouped['pct_diff'] = ((grouped['mean'] - best_path) / best_path * 100).round(2)
    grouped = grouped.sort_values('pct_diff')
    
    colors = sns.color_palette("husl", len(grouped))
    ax.barh(range(len(grouped)), grouped['pct_diff'], color=colors, alpha=0.8, edgecolor='black')
    
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(row['pct_diff'] + 0.5, i, f"{row['pct_diff']:.1f}%", va='center', fontsize=10)
    
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped['algorithm'])
    ax.set_xlabel('% Difference from Best Solution', fontsize=11, fontweight='bold')
    ax.set_title(f'Path Length Performance - {case_name}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    ax.errorbar(grouped['pct_diff'], range(len(grouped)), 
                xerr=grouped['std'] / best_path * 100, fmt='none', color='black', 
                elinewidth=1, capsize=3, alpha=0.5)
    
    plt.tight_layout()
    return fig


def analyze_execution_time(df, case_name):
    """Analyze execution time across algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    time_stats = df.groupby('algorithm')['execution_time_seconds'].agg(['mean', 'std']).reset_index()
    time_stats = time_stats.sort_values('mean')
    
    colors = sns.color_palette("husl", len(time_stats))
    axes[0].barh(range(len(time_stats)), time_stats['mean'], color=colors, alpha=0.8, edgecolor='black')
    axes[0].errorbar(time_stats['mean'], range(len(time_stats)), 
                     xerr=time_stats['std'], fmt='none', color='black', elinewidth=1, capsize=3)
    
    axes[0].set_yticks(range(len(time_stats)))
    axes[0].set_yticklabels(time_stats['algorithm'])
    axes[0].set_xlabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Execution Time', fontsize=12, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(axis='x', alpha=0.3)
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]['execution_time_seconds']
        axes[1].hist(algo_data, bins=10, alpha=0.6, label=algo)
    
    axes[1].set_xlabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Execution Times', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    fig.suptitle(f'Execution Time Analysis - {case_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def analyze_nodes_expanded(df, case_name):
    """Analyze nodes expanded across algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nodes_stats = df.groupby('algorithm')['nodes_expanded'].agg(['mean', 'std']).reset_index()
    nodes_stats = nodes_stats.sort_values('mean', ascending=False)
    
    colors = sns.color_palette("husl", len(nodes_stats))
    axes[0].barh(range(len(nodes_stats)), nodes_stats['mean'], color=colors, alpha=0.8, edgecolor='black')
    axes[0].errorbar(nodes_stats['mean'], range(len(nodes_stats)), 
                     xerr=nodes_stats['std'], fmt='none', color='black', elinewidth=1, capsize=3)
    
    axes[0].set_yticks(range(len(nodes_stats)))
    axes[0].set_yticklabels(nodes_stats['algorithm'])
    axes[0].set_xlabel('Nodes Expanded', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Nodes Expanded', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    plot_data = [df[df['algorithm'] == algo]['nodes_expanded'].values for algo in df['algorithm'].unique()]
    bp = axes[1].boxplot(plot_data, labels=df['algorithm'].unique(), vert=True, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(plot_data))):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    axes[1].set_ylabel('Nodes Expanded', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Nodes Expanded', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    fig.suptitle(f'Nodes Expanded Analysis - {case_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def print_summary_stats(df, case_name):
    """Print summary statistics table."""
    summary = df.groupby('algorithm').agg({
        'best_fitness': ['mean', 'std', 'min', 'max'],
        'nodes_expanded': ['mean', 'std'],
        'execution_time_seconds': ['mean', 'std']
    }).round(4)
    
    print(f"\n  {case_name.upper()} Summary Statistics:")
    print(summary)
    
    csv_summary = OUTPUT_DIR / f'{case_name}_summary_stats.csv'
    summary.to_csv(csv_summary)
    print(f"\n  ✓ Saved summary: {csv_summary}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all stages for all test cases."""
    
    print("\n" + "=" * 80)
    print("GRID PATHFINDING - COMPLETE ANALYSIS")
    print("=" * 80)
    
    all_results = {}
    processed = 0
    skipped = 0
    
    for case_name, case_path in TEST_CASES.items():
        if not case_path.exists():
            print(f"\n[SKIP] {case_name}: directory not found")
            print(f"       Expected: {case_path}")
            skipped += 1
            continue
        
        try:
            # Stage 1: Visualization
            trace_s1 = stage1_visualization(case_name, case_path)
            all_results[case_name] = {'stage1': trace_s1}
            
            # Generate pre-run script for benchmarks
            prerun_benchmark_script(case_name, case_path)
            
            # Stage 2: Benchmark Analysis
            benchmark_df = stage2_benchmark_analysis(case_name, case_path)
            all_results[case_name]['stage2'] = benchmark_df
            
            processed += 1
            
        except Exception as e:
            print(f"\n[ERROR] Error processing {case_name}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n[OK] Output saved to: {OUTPUT_DIR}")
    print(f"\nResults: {processed} processed, {skipped} skipped")
    
    if all_results:
        print(f"\nTest cases processed:")
        for case_name in sorted(all_results.keys()):
            print(f"  ✓ {case_name}")
    
    print(f"\nGenerated files:")
    output_files = sorted([f.name for f in OUTPUT_DIR.glob('*') if f.is_file()])
    if output_files:
        for fname in output_files:
            print(f"  • {fname}")
    else:
        print("  (none yet)")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    if processed > 0:
        print("\n1. To generate real benchmark results, run the pre-run scripts:")
        for case_name in sorted(TEST_CASES.keys()):
            if case_path.exists():
                print(f"   python {OUTPUT_DIR / f'prerun_{case_name}_benchmark.py'}")
        print("\n2. After generating benchmarks, re-run this script to load results:")
        print(f"   python {FILE_PATH}")
        print("\n3. View outputs in:")
        print(f"   {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()