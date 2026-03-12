#!/usr/bin/env python3
"""
Grid Pathfinding Visualization Notebook
=======================================

This notebook provides visualization functions for grid pathfinding experiments.
It can create animations of algorithm expansion, path visualizations, and
performance comparison plots.

Usage:
    from notebooks.grid_visualization_example import load_trace, create_grid_animation
    trace = load_trace('trace.pkl')
    anim = create_grid_animation(trace, 'BFS')
    plt.show()
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_trace(filepath: str) -> Dict:
    """
    Load a trace file created by GridExperiment.visualize().

    Args:
        filepath: Path to .pkl trace file

    Returns:
        Trace dictionary containing grid and algorithm results
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def create_grid_animation(trace: Dict, algorithm: str, interval: int = 200,
                          save_path: Optional[str] = None) -> animation.FuncAnimation:

    if algorithm not in trace['algorithms']:
        raise ValueError(
            f"Algorithm '{algorithm}' not found. Available: {list(trace['algorithms'].keys())}"
        )

    result = trace['algorithms'][algorithm]
    grid = trace['grid']
    start = trace['start_node']
    end = trace['end_node']

    explored_history = result.get('explored_nodes_history', [])
    frontier_history = result.get('frontier_history', [])
    best_path = result.get('best_solution', [])

    if not explored_history:
        raise ValueError(f"No expansion history found for {algorithm}")

    max_steps = len(explored_history)

    # ---- State values ----
    OPEN = 0
    WALL = 1
    START = 2
    GOAL = 3
    EXPLORED = 4
    FRONTIER = 5
    PATH = 6

    # ---- Colors ----
    cmap = ListedColormap([
        "#ffffff",  # open
        "#000000",  # wall
        "#FDFCD8",  # start
        "#006A8A",  # goal
        "#00C2CC",  # explored
        "#B3E5CD",  # frontier
        "#0081A7"   # path
    ])

    # ---- Setup figure ----
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(right=0.8)

    rows, cols = grid.shape

    # reusable display grid (avoid reallocating)
    display_grid = grid.copy()
    display_grid[start] = START
    display_grid[end] = GOAL

    im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=6, origin="upper")
    ax.set_aspect("equal")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- Draw grid lines once ----
    for x in range(cols + 1):
        ax.axvline(x - 0.5, color="gray", linewidth=0.5)

    for y in range(rows + 1):
        ax.axhline(y - 0.5, color="gray", linewidth=0.5)

    # ---- Legend ----
    legend_elements = [
        Patch(facecolor="#ffffff", edgecolor="black", label="Open"),
        Patch(facecolor="#000000", label="Wall"),
        Patch(facecolor="#FDFCD8", label="Start"),
        Patch(facecolor="#006A8A", label="Goal"),
        Patch(facecolor="#00C2CC", label="Explored"),
        Patch(facecolor="#B3E5CD", label="Frontier"),
        Patch(facecolor="#0081A7", label="Path")
    ]

    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax.set_title(f"{algorithm} Expansion")

    # ---- Animation function ----
    def animate(frame):

        step = min(frame, max_steps - 1)

        # reset grid
        display_grid[:] = grid
        display_grid[start] = START
        display_grid[end] = GOAL

        explored_set = set(explored_history[:step+1])

        # explored nodes
        for node in explored_set:
            if node != start and node != end:
                display_grid[node] = EXPLORED

        # frontier nodes
        if step < len(frontier_history):
            for node in frontier_history[step]:
                if (
                    node != start
                    and node != end
                    and node not in explored_set
                ):
                    display_grid[node] = FRONTIER

        # final path
        if frame >= max_steps - 1 and best_path:
            for node in best_path:
                if node != start and node != end:
                    display_grid[node] = PATH

        im.set_data(display_grid)

        ax.set_title(f"{algorithm} - Step {step+1}/{max_steps}")

        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=max_steps,
        interval=interval,
        blit=True,
        repeat=False,
    )

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
        else:
            print(f"Unsupported save format for {save_path}. Use .gif or .mp4.")
        print(f"Animation saved to: {save_path}")
    else:
        print("No save path provided, animation will not be saved.")

    plt.close(fig)
    return anim

def plot_path_comparison(trace: Dict, algorithms: Optional[List[str]] = None,
                        save_path: Optional[str] = None):

    if algorithms is None:
        algorithms = list(trace['algorithms'].keys())

    grid = trace['grid']
    start = trace['start_node']
    end = trace['end_node']

    # ---- State values (same as animation) ----
    OPEN = 0
    WALL = 1
    START = 2
    GOAL = 3
    EXPLORED = 4
    FRONTIER = 5
    PATH = 6

    # ---- Same colormap as animation ----
    cmap = ListedColormap([
        "#ffffff",  # open
        "#000000",  # wall
        "#FDFCD8",  # start
        "#006A8A",  # goal
        "#00C2CC",  # explored
        "#B3E5CD",  # frontier
        "#0081A7"   # final path
    ])

    n_algorithms = len(algorithms)
    n_cols = min(3, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, algo in enumerate(algorithms):

        ax = axes[i]
        result = trace['algorithms'][algo]

        display_grid = np.copy(grid)
        display_grid[start] = START
        display_grid[end] = GOAL

        best_path = result.get('best_solution', [])

        for node in best_path:
            if node != start and node != end:
                display_grid[node] = PATH

        im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=6, origin="upper")
        ax.set_aspect("equal")

        rows, cols = grid.shape

        # grid lines
        # draw vertical lines
        for x in range(cols + 1):
            ax.axvline(x - 0.5, color="gray", linewidth=0.5)

        # draw horizontal lines
        for y in range(rows + 1):
            ax.axhline(y - 0.5, color="gray", linewidth=0.5)

        ax.set_title(f"{algo}\nPath Length: {len(best_path)-1 if best_path else 'N/A'}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(len(algorithms), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Path comparison plot saved to: {save_path}")

    return fig


def plot_performance_comparison(csv_path: str, metrics: List[str] = None,
                               save_path: Optional[str] = None):
    """
    Create performance comparison plots from benchmark CSV.

    Args:
        csv_path: Path to benchmark CSV file
        metrics: List of metrics to plot (default: ['best_fitness', 'execution_time_seconds', 'nodes_expanded'])
        save_path: Optional path to save plot
    """
    import pandas as pd

    if metrics is None:
        metrics = ['best_fitness', 'execution_time_seconds', 'nodes_expanded']

    df = pd.read_csv(csv_path)

    # Filter out failed algorithms
    df = df.dropna(subset=['best_fitness'])

    if len(df) == 0:
        print("No valid results found in CSV")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    metric_names = {
        'best_fitness': 'Path Length',
        'execution_time_seconds': 'Execution Time (s)',
        'nodes_expanded': 'Nodes Expanded',
        'optimality_gap_pct': 'Suboptimality (%)'
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric "{metric}" not found', ha='center', va='center', transform=ax.transAxes)
            continue

        # Create bar plot
        bars = ax.bar(range(len(df)), df[metric], color=sns.color_palette("husl", len(df)))

        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(df[metric])*0.01,
                   '.2f', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['algorithm'], rotation=45, ha='right')
        ax.set_title(f'{metric_names.get(metric, metric)}')
        ax.set_ylabel(metric_names.get(metric, metric))

        # Add grid
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to: {save_path}")

    return fig


def plot_convergence_comparison(csv_path: str, algorithms: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
    """
    Plot convergence curves for metaheuristic algorithms.

    Note: This requires trace files with convergence_curve data, not CSV.
    For CSV files, this will show iteration count comparison.

    Args:
        csv_path: Path to benchmark CSV (or trace file path)
        algorithms: Algorithms to include
        save_path: Optional path to save plot
    """
    import pandas as pd

    # Try to load as CSV first
    try:
        df = pd.read_csv(csv_path)
        # For CSV, we only have final metrics, not convergence curves
        fig, ax = plt.subplots(figsize=(8, 6))

        meta_algorithms = df[df['algorithm'].isin(['GA_Grid', 'ACO_Grid'])]
        if len(meta_algorithms) == 0:
            ax.text(0.5, 0.5, 'No metaheuristic algorithms found in CSV', ha='center', va='center', transform=ax.transAxes)
        else:
            bars = ax.bar(range(len(meta_algorithms)), meta_algorithms['iterations'],
                          color=sns.color_palette("husl", len(meta_algorithms)))
            ax.set_xticks(range(len(meta_algorithms)))
            ax.set_xticklabels(meta_algorithms['algorithm'], rotation=45, ha='right')
            ax.set_title('Metaheuristic Iterations')
            ax.set_ylabel('Iterations')

        plt.tight_layout()

    except:
        # Assume it's a trace file
        trace = load_trace(csv_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        meta_algorithms = ['GA_Grid', 'ACO_Grid']
        colors = sns.color_palette("husl", len(meta_algorithms))

        for i, algo in enumerate(meta_algorithms):
            if algo in trace['algorithms']:
                result = trace['algorithms'][algo]
                conv_curve = result.get('convergence_curve', [])

                if conv_curve:
                    ax.plot(conv_curve, label=algo, color=colors[i], linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Convergence Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")

    return fig


# Example usage and demonstration
if __name__ == "__main__":
    print("Grid Visualization Example")
    print("=" * 40)

    # Example: Load trace and create animation
    try:
        # This would be run after creating a trace file
        print("Example usage:")
        print("1. Create trace file:")
        print("   python experiments/grid_experiment.py --mode visual --rows 10 --cols 10 --algorithms BFS,A* --out example_trace.pkl")
        print()
        print("2. Load and animate:")
        print("   from notebooks.grid_visualization_example import load_trace, create_grid_animation")
        print("   trace = load_trace('example_trace.pkl')")
        print("   anim = create_grid_animation(trace, 'BFS')")
        print("   plt.show()")
        print()
        print("3. Create performance plots:")
        print("   python experiments/grid_experiment.py --mode benchmark --rows 50 --cols 50 --algorithms BFS,A*,GA_Grid --out benchmark.csv")
        print("   plot_performance_comparison('benchmark.csv')")
        print("   plt.show()")

    except Exception as e:
        print(f"Error in example: {e}")