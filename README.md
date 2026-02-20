# AI1_search-algorithms

src
│
├── algorithms/                 # Core algorithm implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract Base Class (BaseAlgorithm)
│   ├── classic/                # BFS, DFS, A*, UCS, Greedy
│   ├── evolutionary/           # GA, DE
│   ├── physics/                # SA, GSA
│   ├── biology/                # ACO, PSO, ABC, FA, CS
│   └── human/                  # TLBO, SFO, CA
│
├── problems/                   # Problem definitions
│   ├── __init__.py
│   ├── continuous.py           # Sphere, Rastrigin, Rosenbrock (NumPy math)
│   └── discrete.py             # TSP, Shortest Path graph generators
│
├── utils/                      # Helpers for your metrics
│   ├── __init__.py
│   ├── metrics.py              # Functions to calculate mean, std, time complexity
│   └── visualization.py        # Matplotlib/Seaborn wrappers (3D plots, convergence lines)
│
├── notebooks/                  # Jupyter notebooks for running experiments
│   ├── 01_continuous_benchmarks.ipynb
│   ├── 02_discrete_benchmarks.ipynb
│   ├── 03_parameter_sensitivity.ipynb
│   └── 04_classic_search_visuals.ipynb
│
├── requirements.txt            # numpy, matplotlib, seaborn
├── README.md                   # Project overview and setup instructions
└── main.py                     # Optional: CLI script to run specific tests