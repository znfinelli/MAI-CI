"""
This file makes the 'src' directory a Python package.

The '__all__' list defines the public API of this package, i.e.,
what modules are imported when a user runs:
from src import *
"""

# Import our modules to make them easily accessible at the package level
from .es_params import ESParams
from .test_functions import TestFunctions
from .evolution_strategy import EvolutionStrategy
from .experiment_runner import ExperimentRunner
from .visualization import plot_convergence, plot_boxplots

# Define the public API
__all__ = [
    'ESParams',
    'TestFunctions',
    'EvolutionStrategy',
    'ExperimentRunner',
    'plot_convergence',
    'plot_boxplots'
]