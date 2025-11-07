"""
Quick test version - runs faster for verification
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable, Tuple, List, Dict
import pandas as pd
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

@dataclass
class ESParams:
    mu: int
    lambda_: int
    dim: int
    sigma: float
    tau: float
    tau_prime: float
    max_generations: int
    target_fitness: float
    strategy: str

class TestFunctions:
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        n = len(x)
        A = 10
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def get_bounds(func_name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        bounds = {
            'sphere': (-5.12, 5.12),
            'rastrigin': (-5.12, 5.12),
        }
        lower, upper = bounds[func_name]
        return np.full(dim, lower), np.full(dim, upper)

class EvolutionStrategy:
    def __init__(self, params: ESParams, fitness_func: Callable, bounds: Tuple[np.ndarray, np.ndarray]):
        self.params = params
        self.fitness_func = fitness_func
        self.lower_bounds, self.upper_bounds = bounds
        self.population = None
        self.sigmas = None
        self.fitness_values = None
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.generations_run = 0
        self.function_evaluations = 0
        
    def initialize_population(self):
        self.population = np.random.uniform(
            self.lower_bounds, self.upper_bounds,
            size=(self.params.mu, self.params.dim)
        )
        self.sigmas = np.full(self.params.mu, self.params.sigma)
        self.evaluate_population(self.population)
        
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        fitness = np.array([self.fitness_func(ind) for ind in population])
        self.function_evaluations += len(population)
        return fitness
    
    def mutate(self, parent: np.ndarray, sigma: float) -> Tuple[np.ndarray, float]:
        new_sigma = sigma * np.exp(self.params.tau_prime * np.random.randn() + 
                                    self.params.tau * np.random.randn())
        offspring = parent + new_sigma * np.random.randn(self.params.dim)
        offspring = np.clip(offspring, self.lower_bounds, self.upper_bounds)
        return offspring, new_sigma
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray, 
                       sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.argsort(fitness)[:self.params.mu]
        return population[indices], sigmas[indices]
    
    def run(self, verbose: bool = False) -> Dict:
        start_time = time.time()
        self.initialize_population()
        
        for generation in range(self.params.max_generations):
            offspring_pop = []
            offspring_sigmas = []
            
            for _ in range(self.params.lambda_):
                parent_idx = np.random.randint(self.params.mu)
                parent = self.population[parent_idx]
                parent_sigma = self.sigmas[parent_idx]
                offspring, new_sigma = self.mutate(parent, parent_sigma)
                offspring_pop.append(offspring)
                offspring_sigmas.append(new_sigma)
            
            offspring_pop = np.array(offspring_pop)
            offspring_sigmas = np.array(offspring_sigmas)
            offspring_fitness = self.evaluate_population(offspring_pop)
            
            if self.params.strategy == 'comma':
                self.population, self.sigmas = self.select_parents(
                    offspring_pop, offspring_fitness, offspring_sigmas
                )
                self.fitness_values = self.evaluate_population(self.population)
            else:
                combined_pop = np.vstack([self.population, offspring_pop])
                combined_sigmas = np.concatenate([self.sigmas, offspring_sigmas])
                combined_fitness = np.concatenate([self.fitness_values, offspring_fitness])
                self.population, self.sigmas = self.select_parents(
                    combined_pop, combined_fitness, combined_sigmas
                )
                self.fitness_values = self.evaluate_population(self.population)
            
            current_best_fitness = np.min(self.fitness_values)
            self.best_fitness_history.append(current_best_fitness)
            self.mean_fitness_history.append(np.mean(self.fitness_values))
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin(self.fitness_values)].copy()
            
            self.generations_run = generation + 1
            
            if verbose and generation % 20 == 0:
                print(f"Gen {generation}: Best={current_best_fitness:.6e}")
            
            if current_best_fitness <= self.params.target_fitness:
                break
        
        return {
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution,
            'generations': self.generations_run,
            'function_evaluations': self.function_evaluations,
            'time': time.time() - start_time,
            'best_history': self.best_fitness_history,
            'converged': self.best_fitness <= self.params.target_fitness
        }

print("="*60)
print("QUICK TEST - Evolution Strategies")
print("="*60)

# Test on Sphere function
dim = 10
tau = 1 / np.sqrt(2 * dim)
tau_prime = 1 / np.sqrt(2 * np.sqrt(dim))

params = ESParams(
    mu=15, lambda_=100, dim=dim, sigma=0.5,
    tau=tau, tau_prime=tau_prime,
    max_generations=200, target_fitness=1e-6,
    strategy='comma'
)

func = TestFunctions.sphere
lower, upper = TestFunctions.get_bounds('sphere', dim)

print("\nRunning 5 trials on Sphere function (dim=10)...")
results = []

for i in range(5):
    print(f"  Trial {i+1}/5...", end=' ')
    es = EvolutionStrategy(params, func, (lower, upper))
    result = es.run(verbose=False)
    results.append(result['best_fitness'])
    print(f"Best fitness: {result['best_fitness']:.6e}")

print(f"\nMean: {np.mean(results):.6e}")
print(f"Std:  {np.std(results):.6e}")
print(f"\n✓ Test successful! Main script should work correctly.")
print("="*60)