import numpy as np

def sphere(x):
    """
    Sphere Function: f(x) = sum(x_i^2)
    Convex, unimodal. Global minimum at x = [0, ..., 0] where f(x) = 0.
    Expects x shape: (pop_size, dimensions)
    """
    return np.sum(x**2, axis=1)


def rastrigin(x):
    """
    Rastrigin Function: f(x) = 10*n + Σ(x_i² - 10*cos(2πx_i))
    Highly multimodal with many local minima. Global minimum at x = [0, ..., 0] where f(x) = 0.
    Search space: typically [-5.12, 5.12] per dimension.
    Expects x shape: (pop_size, dimensions)
    """
    n = x.shape[1]  # Number of dimensions
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def rosenbrock(x):
    """
    Rosenbrock Function: f(x) = Σ(100*(x_{i+1} - x_i²)² + (1 - x_i)²)
    Valley-shaped with a narrow curved minimum. Global minimum at x = [1, ..., 1] where f(x) = 0.
    Search space: typically [-2, 2] per dimension (or [-5, 10]).
    Expects x shape: (pop_size, dimensions)
    """
    # Shift indices: compare consecutive dimensions
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)