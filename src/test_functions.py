import numpy as np
from typing import Tuple

class TestFunctions:
    """
    Collection of standard benchmark functions for continuous optimization.
    
    These functions are commonly used in EC literature to test the
    performance of optimization algorithms. We include a mix of unimodal
    (Sphere, Rosenbrock) and multimodal (Rastrigin, Ackley) functions
    to test both local tuning and global exploration capabilities.
    
    Source for formulas and properties:
    - https://www.sfu.ca/~ssurjano/optimization.html
    - Project brief and lecture slides
    """
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        Sphere Function.
        - f(x) = Σ(x_i^2)
        - Unimodal, convex, and simple.
        - Used as a basic benchmark to ensure an algorithm can perform
          simple hill-climbing / local search.
        - Global minimum: f(0, ..., 0) = 0
        """
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Rastrigin Function.
        - f(x) = 10n + Σ(x_i^2 - 10*cos(2*π*x_i))
        - Highly multimodal with a regular, grid-like structure of
          local optima.
        - Very "difficult" function designed to trap local optimizers.
        - Global minimum: f(0, ..., 0) = 0
        """
        n = len(x)
        A = 10
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """
        Rosenbrock Function (aka "Rosenbrock's banana" or "valley").
        - f(x) = Σ(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
        - Unimodal but features a very narrow, curved, parabolic-shaped
          valley.
        - Difficult for algorithms that do not adapt their search direction.
        - Global minimum: f(1, ..., 1) = 0
        """
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """
        Ackley Function.
        - f(x) = -20*exp(-0.2*sqrt(Σ(x_i^2)/n)) - exp(Σ(cos(2*π*x_i))/n) + 20 + e
        - Highly multimodal, with a near-flat outer region that can
          stall algorithms and a deep central basin.
        - Global minimum: f(0, ..., 0) = 0
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + 20 + np.e
    
    @staticmethod
    def get_bounds(func_name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the conventional compact search space (bounds) for a
        given function.
        """
        bounds_dict = {
            'sphere': (-5.12, 5.12),
            'rastrigin': (-5.12, 5.12),
            'rosenbrock': (-2.048, 2.048), # Often [-5, 10], but this is common too
            'ackley': (-32.768, 32.768)
        }
        
        # Default to Rastrigin/Sphere bounds if the name is not found
        lower, upper = bounds_dict.get(func_name, (-5.12, 5.12))
        
        # Return n-dimensional numpy arrays for the bounds
        return np.full(dim, lower), np.full(dim, upper)