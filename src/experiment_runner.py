import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Any

# Import our custom modules
from src.evolution_strategy import EvolutionStrategy
from src.test_functions import TestFunctions
from src.es_params import ESParams


class ExperimentRunner:
    """
    Manages the execution of multiple independent runs for a given
    ES configuration to ensure statistical rigor, as required by the 
    project brief.
    """
    
    def __init__(self, n_runs: int = 30):
        """
        Args:
            n_runs (int): The number of independent trials to average over.
        """
        self.n_runs = n_runs
        # self.results list was here, but it's better to have 
        # run_experiment be stateless and just return its own results.
        
    def run_experiment(self, func_name: str, dim: int, mu: int, lambda_: int, 
                       strategy: str, max_gen: int = 500) -> Tuple[pd.DataFrame, List[List[float]]]:
        """
        Runs a full experiment (n_runs) for a single ES configuration.

        Args:
            func_name (str): Name of the test function (e.g., 'rastrigin').
            dim (int): Problem dimension (n).
            mu (int): Number of parents.
            lambda_ (int): Number of offspring.
            strategy (str): 'comma' for (μ,λ) or 'plus' for (μ+λ).
            max_gen (int): Maximum generations to run.

        Returns:
            A tuple containing:
            1.  A pandas DataFrame with detailed results for each run.
            2.  A list of lists, where each inner list is the best_fitness_history
                for one of the n_runs.
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {func_name.upper()} (n={dim})")
        print(f"Strategy: ({mu} {strategy} {lambda_})-ES | Runs: {self.n_runs} | Max Gens: {max_gen}")
        print(f"{'='*60}")
        
        # 1. Get the problem definition
        func = getattr(TestFunctions, func_name)
        lower, upper = TestFunctions.get_bounds(func_name, dim)
        
        # 2. Setup Self-Adaptive Parameters
        # We are using Mutation (I) (one sigma per individual).
        tau = 1 / np.sqrt(dim)  # Learning rate for Mutation (I)
        
        # 3. Create the parameter object
        params = ESParams(
            mu=mu,
            lambda_=lambda_,
            dim=dim,
            sigma=0.5,  # Initial mutation strength
            tau=tau, # Pass only the one we are using
            max_generations=max_gen,
            target_fitness=1e-6, # Convergence threshold
            strategy=strategy
        )
        
        # 4. Run multiple independent trials
        trial_results_list = []
        all_run_histories = []
        
        for run_idx in range(self.n_runs):
            if (run_idx + 1) % 10 == 0 or run_idx == 0:
                print(f"  Starting run {run_idx+1}/{self.n_runs}...")
            
            # Create a new ES instance for each run
            es = EvolutionStrategy(params, func, (lower, upper))
            
            # Run the algorithm
            result_dict = es.run(verbose=False)
            
            # Store results for this run
            trial_results_list.append({
                'function': func_name,
                'dimension': dim,
                'mu': mu,
                'lambda': lambda_,
                'strategy': strategy,
                'run': run_idx + 1,
                'best_fitness': result_dict['best_fitness'],
                'generations': result_dict['generations'],
                'function_evals': result_dict['function_evaluations'],
                'time': result_dict['time'],
                'converged': result_dict['converged']
            })
            
            # Store the fitness history for this run
            all_run_histories.append(result_dict['best_history'])
        
        print(f"...Done! ({self.n_runs} runs completed)")
        
        # 5. Collate and summarize results
        df = pd.DataFrame(trial_results_list)
        
        # Print summary statistics for this specific experiment
        print(f"\nResults Summary for this configuration:")
        print(f"  Success Rate: {df['converged'].sum()}/{self.n_runs} ({100*df['converged'].mean():.1f}%)")
        print(f"  Avg. Best Fitness: {df['best_fitness'].mean():.4e} (Std: {df['best_fitness'].std():.4e})")
        print(f"  Avg. Func. Evals: {df['function_evals'].mean():.0f}")
        print(f"  Avg. Time: {df['time'].mean():.3f}s")
        
        return df, all_run_histories