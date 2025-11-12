import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List


def plot_convergence(histories_dict: Dict[str, List[List[float]]], title: str, save_path: str):
    """
    Plots the convergence curves (Best Fitness vs. Generation) for
    one or more experimental configurations.
    
    Each configuration's line represents the mean of its 30 independent runs,
    and the shaded area represents +/- one standard deviation.

    Args:
        histories_dict: A dictionary where keys are experiment labels
                        (e.g., 'rastrigin_d10_comma') and values are
                        lists of 30 fitness histories.
        title (str): The main title for the plot.
        save_path (str): The full path to save the resulting PNG file.
    """
    plt.figure(figsize=(10, 6))
    
    for label, histories in histories_dict.items():
        # --- Handle unequal history lengths ---
        # Runs that converge early will have shorter history lists.
        # We must pad them to the length of the longest run (max_gen)
        # to correctly calculate mean/std at each generation.
        
        if not histories:  # Skip if no data
            continue
            
        max_len = max(len(h) for h in histories)
        
        padded_histories = []
        for h in histories:
            # Pad with the *last* fitness value (assuming it converged)
            padding = [h[-1]] * (max_len - len(h))
            padded_histories.append(h + padding)
        
        histories_array = np.array(padded_histories)
        
        # --- Calculate statistics ---
        mean = np.mean(histories_array, axis=0)
        std = np.std(histories_array, axis=0)
        generations = np.arange(max_len)
        
        # --- Plot mean and standard deviation ---
        # We use semilogy for fitness as it often spans orders of magnitude
        plt.semilogy(generations, mean, label=label, linewidth=2)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2,
                         label=f"{label} (±1 std dev)")
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}")
    
def plot_boxplots(df: pd.DataFrame, save_path: str):
    """Create box plots comparing different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of Optimization Methods (30 Runs)', fontsize=16, fontweight='bold')
    
    # --- Prepare data ---
    # Group by function and the *method* (which we've stored in 'strategy')
    grouped = df.groupby(['function', 'dimension', 'strategy'])
    
    labels = []
    data_by_metric = {
        'best_fitness': [],
        'generations': [],
        'function_evals': [],
        'success_rate': []
    }
    
    for (func, dim, strategy), group in grouped:
        labels.append(f"{func.upper()} (n={dim})\n({strategy})")
        data_by_metric['best_fitness'].append(group['best_fitness'].values)
        data_by_metric['generations'].append(group['generations'].values)
        data_by_metric['function_evals'].append(group['function_evals'].values)
        data_by_metric['success_rate'].append(100 * group['converged'].mean())

    # --- 1. Best Fitness Plot (Boxplot) ---
    ax = axes[0, 0]
    ax.boxplot(data_by_metric['best_fitness'], labels=labels)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Final Best Fitness Distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # --- 2. Generations/Iterations (Boxplot) ---
    ax = axes[0, 1]
    ax.boxplot(data_by_metric['generations'], labels=labels)
    ax.set_ylabel('Generations / Iterations', fontsize=11)
    ax.set_title('Generations/Iterations to Convergence', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # --- 3. Function Evaluations (Boxplot) ---
    ax = axes[1, 0]
    ax.boxplot(data_by_metric['function_evals'], labels=labels)
    ax.set_ylabel('Function Evaluations', fontsize=11)
    ax.set_title('Total Function Evaluations', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # --- 4. Success Rate (Bar Plot) ---
    ax = axes[1, 1]
    ax.bar(range(len(labels)), data_by_metric['success_rate'], color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate (Fitness < 1e-6)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}")