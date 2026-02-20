import numpy as np

def sphere(x):
    """
    Sphere Function: f(x) = sum(x_i^2)
    Convex, unimodal. Global minimum at x = [0, ..., 0] where f(x) = 0.
    Expects x shape: (pop_size, dimensions)
    """
    return np.sum(x**2, axis=1)