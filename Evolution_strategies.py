import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable, Tuple, List, Dict
import pandas as pd
from dataclasses import dataclass
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


@dataclass
class ESParams:
    """Parameters for Evolution Strategy"""
    mu: int  # Number of parents
    lambda_: int  # Number of offspring
    dim: int  # Problem dimension
    sigma: float  # Initial mutation strength
    tau: float  # Learning rate for sigma adaptation
    tau_prime: float  # Global learning rate
    max_generations: int  # Maximum generations
    target_fitness: float  # Target fitness to stop
    strategy: str  # 'comma' for (μ,λ) or 'plus' for (μ+λ)


class TestFunctions:
    """Collection of benchmark optimization functions"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        Sphere function: f(x) = sum(x_i^2)
        Global minimum: f(0,...,0) = 0
        """
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Rastrigin function: highly multimodal
        Global minimum: f(0,...,0) = 0
        """
        n = len(x)
        A = 10
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """
        Rosenbrock function: narrow valley
        Global minimum: f(1,...,1) = 0
        """
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """
        Ackley function: multimodal with deep valleys
        Global minimum: f(0,...,0) = 0
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    
    @staticmethod
    def get_bounds(func_name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get search space bounds for each function"""
        bounds = {
            'sphere': (-5.12, 5.12),
            'rastrigin': (-5.12, 5.12),
            'rosenbrock': (-2.048, 2.048),
            'ackley': (-32.768, 32.768)
        }
        lower, upper = bounds[func_name]
        return np.full(dim, lower), np.full(dim, upper)


class EvolutionStrategy:
    """
    Implementation of (μ,λ)-ES and (μ+λ)-ES with self-adaptive mutation
    """
    
    def __init__(self, params: ESParams, fitness_func: Callable, bounds: Tuple[np.ndarray, np.ndarray]):
        self.params = params
        self.fitness_func = fitness_func
        self.lower_bounds, self.upper_bounds = bounds
        
        # Initialize population
        self.population = None
        self.sigmas = None
        self.fitness_values = None
        
        # Statistics
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.generations_run = 0
        self.function_evaluations = 0
        
    def initialize_population(self):
        """Initialize population uniformly in search space"""
        self.population = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.params.mu, self.params.dim)
        )
        self.sigmas = np.full(self.params.mu, self.params.sigma)
        # Evaluate and store fitness values for the initialized population
        self.fitness_values = self.evaluate_population(self.population)
        
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for all individuals"""
        fitness = np.array([self.fitness_func(ind) for ind in population])
        self.function_evaluations += len(population)
        return fitness
    
    def mutate(self, parent: np.ndarray, sigma: float) -> Tuple[np.ndarray, float]:
        """
        Apply self-adaptive mutation
        First mutate sigma, then use it to mutate the individual
        """
        # Self-adaptation of sigma
        new_sigma = sigma * np.exp(self.params.tau_prime * np.random.randn() + 
                                    self.params.tau * np.random.randn())
        
        # Mutate individual
        offspring = parent + new_sigma * np.random.randn(self.params.dim)
        
        # Clip to bounds
        offspring = np.clip(offspring, self.lower_bounds, self.upper_bounds)
        
        return offspring, new_sigma
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray, 
                       sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select μ best individuals"""
        indices = np.argsort(fitness)[:self.params.mu]
        return population[indices], sigmas[indices]
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run the Evolution Strategy
        Returns dictionary with results
        """
        start_time = time.time()
        self.initialize_population()
        
        for generation in range(self.params.max_generations):
            # Generate offspring
            offspring_pop = []
            offspring_sigmas = []
            
            for _ in range(self.params.lambda_):
                # Select random parent
                parent_idx = np.random.randint(self.params.mu)
                parent = self.population[parent_idx]
                parent_sigma = self.sigmas[parent_idx]
                
                # Mutate
                offspring, new_sigma = self.mutate(parent, parent_sigma)
                offspring_pop.append(offspring)
                offspring_sigmas.append(new_sigma)
            
            offspring_pop = np.array(offspring_pop)
            offspring_sigmas = np.array(offspring_sigmas)
            
            # Evaluate offspring
            offspring_fitness = self.evaluate_population(offspring_pop)
            
            # Selection strategy
            if self.params.strategy == 'comma':
                # (μ,λ)-ES: select only from offspring
                self.population, self.sigmas = self.select_parents(
                    offspring_pop, offspring_fitness, offspring_sigmas
                )
                self.fitness_values = self.evaluate_population(self.population)
            else:
                # (μ+λ)-ES: select from parents + offspring
                combined_pop = np.vstack([self.population, offspring_pop])
                combined_sigmas = np.concatenate([self.sigmas, offspring_sigmas])
                combined_fitness = np.concatenate([self.fitness_values, offspring_fitness])
                
                self.population, self.sigmas = self.select_parents(
                    combined_pop, combined_fitness, combined_sigmas
                )
                self.fitness_values = self.evaluate_population(self.population)
            
            # Update statistics
            current_best_fitness = np.min(self.fitness_values)
            current_mean_fitness = np.mean(self.fitness_values)
            
            self.best_fitness_history.append(current_best_fitness)
            self.mean_fitness_history.append(current_mean_fitness)
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin(self.fitness_values)].copy()
            
            self.generations_run = generation + 1
            
            # Verbose output
            if verbose and generation % 10 == 0:
                print(f"Gen {generation}: Best={current_best_fitness:.6f}, Mean={current_mean_fitness:.6f}")
            
            # Check convergence
            if current_best_fitness <= self.params.target_fitness:
                if verbose:
                    print(f"Converged at generation {generation}")
                break
        
        end_time = time.time()
        
        return {
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution,
            'generations': self.generations_run,
            'function_evaluations': self.function_evaluations,
            'time': end_time - start_time,
            'best_history': self.best_fitness_history,
            'mean_history': self.mean_fitness_history,
            'converged': self.best_fitness <= self.params.target_fitness
        }


class ExperimentRunner:
    """Run comprehensive experiments with different configurations"""
    
    def __init__(self, n_runs: int = 30):
        self.n_runs = n_runs
        self.results = []
        
    def run_experiment(self, func_name: str, dim: int, mu: int, lambda_: int, 
                       strategy: str, max_gen: int = 500) -> pd.DataFrame:
        """
        Run multiple independent trials
        """
        print(f"\n{'='*60}")
        print(f"Running: {func_name}, dim={dim}, μ={mu}, λ={lambda_}, strategy={strategy}")
        print(f"{'='*60}")
        
        # Get function and bounds
        func = getattr(TestFunctions, func_name)
        lower, upper = TestFunctions.get_bounds(func_name, dim)
        
        # Setup parameters
        tau = 1 / np.sqrt(2 * dim)
        tau_prime = 1 / np.sqrt(2 * np.sqrt(dim))
        
        params = ESParams(
            mu=mu,
            lambda_=lambda_,
            dim=dim,
            sigma=0.5,
            tau=tau,
            tau_prime=tau_prime,
            max_generations=max_gen,
            target_fitness=1e-6,
            strategy=strategy
        )
        
        # Run multiple trials
        trial_results = []
        all_histories = []
        
        for run in range(self.n_runs):
            if run % 10 == 0:
                print(f"Run {run+1}/{self.n_runs}...", end=' ')
            
            es = EvolutionStrategy(params, func, (lower, upper))
            result = es.run(verbose=False)
            
            trial_results.append({
                'function': func_name,
                'dimension': dim,
                'mu': mu,
                'lambda': lambda_,
                'strategy': strategy,
                'run': run + 1,
                'best_fitness': result['best_fitness'],
                'generations': result['generations'],
                'function_evals': result['function_evaluations'],
                'time': result['time'],
                'converged': result['converged']
            })
            
            all_histories.append(result['best_history'])
        
        print("Done!")
        
        # Store results
        df = pd.DataFrame(trial_results)
        
        # Print summary statistics
        print(f"\nResults Summary:")
        print(f"  Mean Best Fitness: {df['best_fitness'].mean():.6e} ± {df['best_fitness'].std():.6e}")
        print(f"  Success Rate: {df['converged'].sum()}/{self.n_runs} ({100*df['converged'].mean():.1f}%)")
        print(f"  Mean Generations: {df['generations'].mean():.1f} ± {df['generations'].std():.1f}")
        print(f"  Mean Time: {df['time'].mean():.3f}s ± {df['time'].std():.3f}s")
        
        return df, all_histories


def plot_convergence(histories_dict: Dict[str, List], title: str, save_path: str):
    """Plot convergence curves for multiple configurations"""
    plt.figure(figsize=(10, 6))
    
    for label, histories in histories_dict.items():
        # Compute mean and std across runs
        max_len = max(len(h) for h in histories)
        
        # Pad histories to same length
        padded = []
        for h in histories:
            if len(h) < max_len:
                padded.append(h + [h[-1]] * (max_len - len(h)))
            else:
                padded.append(h)
        
        histories_array = np.array(padded)
        mean = np.mean(histories_array, axis=0)
        std = np.std(histories_array, axis=0)
        
        generations = np.arange(len(mean))
        
        plt.semilogy(generations, mean, label=label, linewidth=2)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_boxplots(df: pd.DataFrame, save_path: str):
    """Create box plots comparing different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Best fitness comparison
    ax = axes[0, 0]
    data_to_plot = []
    labels = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        data_to_plot.append(group['best_fitness'].values)
        labels.append(f"{func[:3]}\nμ={mu},λ={lam}\n{strat}")
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Best Fitness Distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Generations comparison
    ax = axes[0, 1]
    data_to_plot = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        data_to_plot.append(group['generations'].values)
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Generations', fontsize=11)
    ax.set_title('Generations to Convergence', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # Function evaluations
    ax = axes[1, 0]
    data_to_plot = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        data_to_plot.append(group['function_evals'].values)
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Function Evaluations', fontsize=11)
    ax.set_title('Function Evaluations', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # Success rate
    ax = axes[1, 1]
    success_rates = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        success_rates.append(100 * group['converged'].mean())
    
    ax.bar(range(len(labels)), success_rates, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate (Convergence < 1e-6)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main experimental pipeline"""
    
    print("="*80)
    print("EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION")
    print("Master in AI - Evolutionary Computation")
    print("="*80)
    
    # Create experiment runner
    runner = ExperimentRunner(n_runs=30)
    
    # Define experiments
    experiments = [
        # Sphere function - easy unimodal
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'plus'},
        {'func': 'sphere', 'dim': 20, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        
        # Rastrigin - multimodal
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'comma'},
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'plus'},
        {'func': 'rastrigin', 'dim': 20, 'mu': 30, 'lambda': 200, 'strategy': 'comma'},
    ]
    
    all_results = []
    all_histories = {}
    
    # Run all experiments
    for exp in experiments:
        df, histories = runner.run_experiment(
            func_name=exp['func'],
            dim=exp['dim'],
            mu=exp['mu'],
            lambda_=exp['lambda'],
            strategy=exp['strategy'],
            max_gen=500
        )
        all_results.append(df)
        
        # Store histories for plotting
        key = f"{exp['func']}_d{exp['dim']}_μ{exp['mu']}_λ{exp['lambda']}_{exp['strategy']}"
        all_histories[key] = histories
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    # Prepare local output directory (project-root/outputs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    results_path = os.path.join(output_dir, 'results.csv')
    combined_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path}")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Convergence plots for Sphere
    sphere_histories = {k: v for k, v in all_histories.items() if 'sphere' in k}
    plot_convergence(
        sphere_histories,
        'Convergence on Sphere Function',
        os.path.join(output_dir, 'convergence_sphere.png')
    )
    
    # Convergence plots for Rastrigin
    rastrigin_histories = {k: v for k, v in all_histories.items() if 'rastrigin' in k}
    plot_convergence(
        rastrigin_histories,
        'Convergence on Rastrigin Function',
        os.path.join(output_dir, 'convergence_rastrigin.png')
    )
    
    # Box plots
    plot_boxplots(combined_df, os.path.join(output_dir, 'comparison_boxplots.png'))
    
    # Summary statistics table
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    summary = combined_df.groupby(['function', 'dimension', 'mu', 'lambda', 'strategy']).agg({
        'best_fitness': ['mean', 'std', 'min'],
        'generations': ['mean', 'std'],
        'function_evals': ['mean', 'std'],
        'time': ['mean', 'std'],
        'converged': ['sum', 'mean']
    }).round(6)
    
    print(summary)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print("\nSummary statistics saved to: {}".format(summary_path))
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  - results.csv (detailed results)")
    print("  - summary_statistics.csv (aggregated statistics)")
    print("  - convergence_sphere.png")
    print("  - convergence_rastrigin.png")
    print("  - comparison_boxplots.png")


if __name__ == "__main__":
    main()