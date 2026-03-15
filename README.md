# AI1 Search Algorithms

A comprehensive collection of search and optimization algorithms for solving continuous and discrete optimization problems. This project implements classic algorithms (BFS, DFS, A*), evolutionary algorithms (GA, DE), biology-inspired algorithms (PSO, ACO, ABC, FA), physics-inspired algorithms (SA, GSA), and human-inspired algorithms (TLBO, SFO, CA).

## Project Overview

This project contains implementations and benchmarks for multiple categories of search algorithms:

- **Classic Algorithms**: BFS, DFS, A*, UCS, Greedy Search
- **Evolutionary Algorithms**: Genetic Algorithm (GA), Differential Evolution (DE)
- **Physics-Inspired**: Simulated Annealing (SA), Gravitational Search Algorithm (GSA)
- **Biology-Inspired**: Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Artificial Bee Colony (ABC), Firefly Algorithm (FA), Cuckoo Search (CS)
- **Human-Inspired**: Teaching-Learning-Based Optimization (TLBO), Shuffled Frog Leaping (SFO), Cultural Algorithm (CA)

## Project Structure

```
src/
├── algorithms/                 # Core algorithm implementations
│   ├── base.py                 # Abstract base class for all algorithms
│   ├── classic/                # BFS, DFS, A*, UCS, Greedy
│   ├── evolution/              # GA, DE
│   ├── physics/                # SA, GSA
│   ├── biology/                # ACO, PSO, ABC, FA, CS
│   └── human/                  # TLBO, SFO, CA
│
├── problems/                   # Problem definitions
│   ├── continuous.py           # Sphere, Rastrigin, Rosenbrock
│   └── discrete.py             # TSP, Grid Pathfinding
│
├── utils/                      # Helper utilities
│   ├── config.json             # Configuration settings
│   ├── metrics.py              # Performance metrics and statistics
│   ├── fairness.py             # Fairness evaluation functions
│   └── visualizes/             # Visualization utilities
│
├── experiments/                # Prerun experiment scripts
│   ├── prerun_tsp_experiment.py
│   ├── prerun_convergence_experiment.py
│   ├── prerun_robustness_experiment.py
│   └── [other prerun scripts]
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 1_notebook_sphere.ipynb
│   ├── 2_notebook_rastrigin.ipynb
│   ├── 3_notebook_rosenbrock.ipynb
│   ├── 4_notebook_tsp.ipynb
│   └── 5_notebook_grid_pathfinding_analysis.ipynb
│
└── tests/                      # Test data and benchmarks
    └── TSP/                    # TSP test cases (sparse, clustered, euclidean)
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd AI1_search-algorithms
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The following packages are required:
   - numpy (numerical computations)
   - matplotlib (visualization)
   - scipy (scientific computing)
   - pandas (data analysis)
   - seaborn (statistical visualization)
   - ipywidgets (interactive widgets for Jupyter)
   - ipython (enhanced Python shell)
   - imageio (video/animation creation)

## Usage Guide

The project is designed to be run in two stages:

### Stage 1: Generate Benchmark Data

Run all prerun experiment scripts to generate benchmark data and experimental results. This stage computes algorithm performance across various optimization problems.

```bash
# Run from the project root directory
python src/experiments/prerun_blank_benchmark.py
python src/experiments/prerun_convergence_experiment.py
python src/experiments/prerun_dim2_experiment.py
python src/experiments/prerun_grid_experiment.py
python src/experiments/prerun_maze_loops_benchmark.py
python src/experiments/prerun_obstacles_benchmark.py
python src/experiments/prerun_obstacles_dense_benchmark.py
python src/experiments/prerun_perfect_maze_benchmark.py
python src/experiments/prerun_robustness_experiment.py
python src/experiments/prerun_tsp_experiment.py
```

**Or run all at once** (from project root):
```bash
# Unix/macOS/WSL
for script in src/experiments/prerun_*.py; do python "$script"; done

# Windows PowerShell
Get-ChildItem src/experiments/prerun_*.py | ForEach-Object { python $_.FullName }
```

**What these scripts do:**
- Execute algorithms on various optimization problems
- Generate convergence traces and performance metrics
- Save results to `results/` and `src/notebooks/outputs/` directories
- Create visualizations and GIFs of algorithm behavior

⏱️ **Note**: This stage may take significant time (hours) depending on the number of iterations and algorithms. Output messages indicate progress.

### Stage 2: Analyze Results in Notebooks

After all prerun scripts complete, run the Jupyter notebooks in order to analyze the generated data:

```bash
# Start Jupyter
jupyter notebook src/notebooks/
```

Then open and run each notebook sequentially:

1. **1_notebook_sphere.ipynb** - Sphere function benchmark analysis
2. **2_notebook_rastrigin.ipynb** - Rastrigin function benchmark analysis
3. **3_notebook_rosenbrock.ipynb** - Rosenbrock function benchmark analysis
4. **4_notebook_tsp.ipynb** - Traveling Salesman Problem analysis
5. **5_notebook_grid_pathfinding_analysis.ipynb** - Grid pathfinding and maze solving analysis

Each notebook includes:
- Algorithm performance comparisons
- Statistical analysis and metrics
- Visualizations (convergence plots, 3D landscapes, heatmaps)
- Generated GIFs showing algorithm behavior

### Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run experiments (takes time)
python src/experiments/prerun_convergence_experiment.py
python src/experiments/prerun_dim2_experiment.py
python src/experiments/prerun_grid_experiment.py
python src/experiments/prerun_robustness_experiment.py
python src/experiments/prerun_tsp_experiment.py

# 3. View results in notebooks
jupyter notebook src/notebooks/4_notebook_tsp.ipynb
```

## Output Files

Results are saved to:
- `results/` - CSV files with summary statistics
- `results/gifs/` - Animated visualizations of algorithm convergence
- `src/notebooks/outputs/` - Notebook analysis outputs

## Algorithm Categories

### Classic Search Algorithms
Traditional uninformed and informed search methods for graph traversal and pathfinding.

### Evolutionary Algorithms
Population-based algorithms inspired by natural evolution and genetics.

### Physics-Inspired Algorithms
Optimization algorithms based on physical phenomena like temperature and gravity.

### Biology-Inspired Algorithms
Swarm intelligence and natural ecological behaviors applied to optimization.

### Human-Inspired Algorithms
Optimization strategies based on human learning, behavior, and culture.

## Configuration

Algorithm parameters and experiment settings can be configured in:
- `src/utils/config.json` - Global configuration
- `src/utils/configHelper.py` - Configuration helper functions

## Requirements

See `requirements.txt` for the complete list of dependencies and their versions.