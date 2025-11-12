import numpy as np
import time
from typing import Callable, Tuple, Dict, List
from .es_params import ESParams


class EvolutionStrategy:
    """
    Implements the core (μ,λ)-ES and (μ+λ)-ES algorithms.
    
    This implementation uses a simple self-adaptive mutation strategy
    with one mutation strength (σ) per individual, which is adapted
    using two learning rates (τ and τ') as discussed in the course
    lecture slides (Slide 16, "Mutation (II)").
    """
    
    def __init__(self, params: ESParams, fitness_func: Callable, 
                 bounds: Tuple[np.ndarray, np.ndarray]):
        """
        Initializes the Evolution Strategy optimizer.

        Args:
            params: ESParams dataclass object with all parameters.
            fitness_func: The objective function to minimize.
            bounds: A tuple (lower_bounds, upper_bounds) for the search space.
        """
        self.params = params
        self.fitness_func = fitness_func
        self.lower_bounds, self.upper_bounds = bounds
        
        # --- Algorithm State ---
        # Genotype = [object variables (x), strategy parameter (sigma)]
        self.population: np.ndarray = None  # (μ, n) array of object variables
        self.sigmas: np.ndarray = None      # (μ,) array of mutation strengths
        self.fitness_values: np.ndarray = None # (μ,) array of fitness values
        
        # --- Statistics for Reporting (as required by project brief) ---
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.best_solution: np.ndarray = None
        self.best_fitness: float = float('inf')
        self.generations_run: int = 0
        self.function_evaluations: int = 0
        
    def initialize_population(self):
        """
        Initializes the parent population (μ individuals).
        Individuals are sampled uniformly from the search space.
        """
        # Initialize object variables
        self.population = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.params.mu, self.params.dim)
        )
        
        # Initialize strategy parameters (all start with the same sigma)
        self.sigmas = np.full(self.params.mu, self.params.sigma)
        
        # Evaluate and store fitness for the initial population
        self.fitness_values = self.evaluate_population(self.population)
        
        # Initialize best-so-far tracking
        self.best_fitness = np.min(self.fitness_values)
        self.best_solution = self.population[np.argmin(self.fitness_values)].copy()
        self.best_fitness_history.append(self.best_fitness)
        self.mean_fitness_history.append(np.mean(self.fitness_values))

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of each individual in a given population.
        This is the main bottleneck, so we track the number of calls.
        """
        # Use a list comprehension, which is clean and efficient for this.
        fitness = np.array([self.fitness_func(ind) for ind in population])
        
        # Increment our performance counter
        self.function_evaluations += len(population)
        return fitness
    
    def mutate(self, parent: np.ndarray, sigma: float) -> Tuple[np.ndarray, float]:
        """
        Applies self-adaptive mutation (as per lecture slides).
        
        This implements "Mutation (I)" from the "4._Intro_to_ESs (1).pdf" slides:
        - One sigma (mutation strength) per individual.
        - One learning rate (tau) for the log-normal update.

        1. Mutates the strategy parameter (sigma).
        2. Mutates the object variables (x) using the *new* sigma.
        """
        
        # 1. Mutate sigma: σ' = σ * exp(τ * N_global(0,1))
        # We only use the single global learning rate, tau.
        tau_norm = self.params.tau * np.random.randn()
        new_sigma = sigma * np.exp(tau_norm) # <-- FIX HERE

        # Ensure sigma doesn't become too small (a common issue)
        # We can set a floor, e.g., 1e-8, to prevent stalling.
        new_sigma = max(new_sigma, 1e-8)
        
        # 2. Mutate object variables: x' = x + σ' * N(0,I)
        # We use the *new* sigma for the mutation.
        offspring = parent + new_sigma * np.random.randn(self.params.dim)
        
        # 3. Enforce boundary constraints (clipping)
        offspring = np.clip(offspring, self.lower_bounds, self.upper_bounds)
        
        return offspring, new_sigma
    
    def select_survivors(self, population: np.ndarray, fitness: np.ndarray, 
                         sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic, rank-based selection.
        Selects the μ-best individuals from the given pool.
        
        Returns the surviving individuals, their sigmas, AND their fitness values.
        """
        # Get the indices of the μ-best (lowest fitness) individuals
        indices = np.argsort(fitness)[:self.params.mu]
        
        # Return the slices corresponding to these survivors
        return population[indices], sigmas[indices], fitness[indices]
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Runs the main evolutionary loop for a fixed number of generations
        or until a target fitness is reached.
        """
        start_time = time.time()
        self.initialize_population()
        
        for generation in range(self.params.max_generations):
            
            # --- 1. Reproduction (Create Offspring) ---
            # We will generate λ offspring.
            offspring_pop_list = []
            offspring_sigmas_list = []
            
            for _ in range(self.params.lambda_):
                # Select a parent randomly from the μ parents
                parent_idx = np.random.randint(self.params.mu)
                parent = self.population[parent_idx]
                parent_sigma = self.sigmas[parent_idx]
                
                # Create one offspring via mutation
                offspring, new_sigma = self.mutate(parent, parent_sigma)
                offspring_pop_list.append(offspring)
                offspring_sigmas_list.append(new_sigma)
            
            offspring_pop = np.array(offspring_pop_list)
            offspring_sigmas = np.array(offspring_sigmas_list)
            
            # --- 2. Evaluate Offspring ---
            # We only evaluate the new offspring
            offspring_fitness = self.evaluate_population(offspring_pop)
            
            # --- 3. Selection (Form Next Generation's Parents) ---
            # This is the main ES strategy choice [cite: 680, 682]
            
            if self.params.strategy == 'comma':
                # (μ, λ)-ES: Select μ best *only* from λ offspring [cite: 682]
                # Parents are discarded (limited life time)[cite: 2165].
                self.population, self.sigmas, self.fitness_values = self.select_survivors(
                    offspring_pop, offspring_fitness, offspring_sigmas
                )
            
            elif self.params.strategy == 'plus':
                # (μ + λ)-ES: Select μ best from (μ parents + λ offspring) [cite: 680]
                # This is an elitist strategy[cite: 832, 2197].
                combined_pop = np.vstack([self.population, offspring_pop])
                combined_sigmas = np.concatenate([self.sigmas, offspring_sigmas])
                combined_fitness = np.concatenate([self.fitness_values, offspring_fitness])
                
                self.population, self.sigmas, self.fitness_values = self.select_survivors(
                    combined_pop, combined_fitness, combined_sigmas
                )
            else:
                raise ValueError(f"Unknown strategy: {self.params.strategy}")
            
            # --- 4. Update Statistics ---
            current_best_fitness = np.min(self.fitness_values) # Fitness of the new best parent
            current_mean_fitness = np.mean(self.fitness_values)
            
            self.best_fitness_history.append(current_best_fitness)
            self.mean_fitness_history.append(current_mean_fitness)
            
            # Update the best-ever-found solution
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin(self.fitness_values)].copy()
            
            self.generations_run = generation + 1
            
            if verbose and generation % 10 == 0:
                print(f"Gen {generation}: Best={current_best_fitness:.6e}, Mean={current_mean_fitness:.6e}")
            
            # --- 5. Check Termination Criterion ---
            if current_best_fitness <= self.params.target_fitness:
                if verbose:
                    print(f"Converged at generation {generation}")
                break
        
        end_time = time.time()
        
        # Return a dictionary of results for this single run
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