"""
EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION
Master in Artificial Intelligence - Evolutionary Computation
November 2025

This script runs the full comparative experiment:
1.  Evolution Strategy (from-scratch implementation)
2.  L-BFGS-B (derivative-based baseline from SciPy)

It runs 30 independent trials for all configurations and saves
all results and visualizations to the 'outputs/' directory.
"""

import numpy as np
import pandas as pd
import warnings
import os
import time
from scipy.optimize import minimize, Bounds

from src import (
    ExperimentRunner,
    TestFunctions,
    plot_convergence,
    plot_boxplots
)

warnings.filterwarnings('ignore')
np.random.seed(42)

def run_baseline_experiment(func_name, dim, n_runs, max_gen, target_fitness):
    """
    Runs 30 trials for the Scipy L-BFGS-B baseline.
    """
    print(f"\n{'='*60}")
    print(f"Running Baseline: L-BFGS-B on {func_name.upper()} (n={dim})")
    print(f"{'='*60}")
    
    func = getattr(TestFunctions, func_name)
    lower, upper = TestFunctions.get_bounds(func_name, dim)
    scipy_bounds = Bounds(lower, upper)
    
    trial_results = []
    
    for run in range(n_runs):
        if (run + 1) % 10 == 0 or run == 0:
            print(f"  Starting run {run+1}/{n_runs}...")
        
        # Use the same random start for a fair comparison
        x_start = np.random.uniform(lower, upper, dim)
        
        start_time = time.time()
        
        # L-BFGS-B is a quasi-Newton (derivative-based) method.
        # We give it a comparable budget of function evaluations.
        bfgs_result = minimize(
            func, 
            x_start, 
            method='L-BFGS-B', 
            bounds=scipy_bounds,
            options={'maxfun': 100000} # Give it a large budget
        )
        
        elapsed_time = time.time() - start_time
        
        trial_results.append({
            'function': func_name,
            'dimension': dim,
            'mu': 'N/A',
            'lambda': 'N/A',
            'strategy': 'L-BFGS-B',  # Use 'strategy' as the method label
            'run': run + 1,
            'best_fitness': bfgs_result.fun,
            'generations': bfgs_result.nit, # Use iterations as "generations"
            'function_evals': bfgs_result.nfev,
            'time': elapsed_time,
            'converged': bfgs_result.fun <= target_fitness
        })
    
    print("...Done!")
    return pd.DataFrame(trial_results)


def main():
    """Main experimental pipeline"""
    
    print("="*80)
    print("EVOLUTION STRATEGIES VS. DERIVATIVE-BASED OPTIMIZATION")
    print("Master in AI - Evolutionary Computation")
    print("="*80)
    
    N_RUNS = 30
    MAX_GEN = 500
    TARGET_FITNESS = 1e-6
    
    # Initialize the ES runner
    es_runner = ExperimentRunner(n_runs=N_RUNS)
    
    # --- Define experiments ---
    es_experiments = [
        # Sphere 10D
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'plus'},
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        
        # Rastrigin 10D
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'plus'},
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'comma'},
        
        # As defined in CI_work.pdf (Figure 3 & 4)
        {'func': 'sphere', 'dim': 20, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        {'func': 'rastrigin', 'dim': 20, 'mu': 30, 'lambda': 200, 'strategy': 'comma'},
    ]
    
    baseline_experiments = [
        {'func': 'sphere', 'dim': 10},
        {'func': 'rastrigin', 'dim': 10},

        {'func': 'sphere', 'dim': 20},
        {'func': 'rastrigin', 'dim': 20},
    ]
    all_results_list = []
    all_histories_dict = {}
    
    # --- 1. Run ES Experiments ---
    for exp_config in es_experiments:
        df, histories = es_runner.run_experiment(
            func_name=exp_config['func'],
            dim=exp_config['dim'],
            mu=exp_config['mu'],
            lambda_=exp_config['lambda'],
            strategy=exp_config['strategy'],
            max_gen=MAX_GEN
        )
        all_results_list.append(df)
        
        key = f"ES_d{exp_config['dim']}_({exp_config['mu']}+{exp_config['lambda']})_{exp_config['func']}"
        all_histories_dict[key] = histories
        
    # --- 2. Run Baseline Experiments ---
    for exp_config in baseline_experiments:
        df = run_baseline_experiment(
            func_name=exp_config['func'],
            dim=exp_config['dim'],
            n_runs=N_RUNS,
            max_gen=MAX_GEN,
            target_fitness=TARGET_FITNESS
        )
        all_results_list.append(df)

    # --- 3. Save Artifacts (CSVs and Plots) ---
    combined_df = pd.concat(all_results_list, ignore_index=True)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'results.csv')
    combined_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {results_path}")
    
    # --- 4. Generate Visualizations ---
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS...")
    print(f"{'='*80}")
    
    # Convergence plots
    # Convergence plots
    if all_histories_dict:
        # Plot for Sphere
        sphere_hist = {k: v for k, v in all_histories_dict.items() if 'sphere' in k}
        plot_convergence(
            sphere_hist,
            'ES Convergence on Sphere (Avg. of 30 Runs)',
            os.path.join(output_dir, 'convergence_sphere.png')
        )
        
        # Plot for Rastrigin
        rastrigin_hist = {k: v for k, v in all_histories_dict.items() if 'rastrigin' in k}
        plot_convergence(
            rastrigin_hist,
            'ES Convergence on Rastrigin (Avg. of 30 Runs)',
            os.path.join(output_dir, 'convergence_rastrigin.png')
        )
    
    # Box plots
    plot_boxplots(combined_df, os.path.join(output_dir, 'comparison_boxplots.png'))
    
    # --- 5. Generate Summary Statistics Table ---
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (for final report)")
    print(f"{'='*80}")
    
    summary = combined_df.groupby(['function', 'dimension', 'strategy']).agg({
        'best_fitness': ['mean', 'std', 'min'],
        'function_evals': ['mean', 'std'],
        'time': ['mean', 'std'],
        'converged': ['sum', 'mean']
    }).round(6)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(summary)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Find all outputs in: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()