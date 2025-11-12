from dataclasses import dataclass

@dataclass
class ESParams:
    """
    A dataclass to hold all configuration parameters for a single
    Evolution Strategy experiment. This keeps our function signatures
    clean and makes experiments easy to configure and pass around.
    """
    
    # --- Population Parameters ---
    mu: int       # Number of parents (μ)
    lambda_: int  # Number of offspring (λ)
    
    # --- Problem Parameters ---
    dim: int      # Problem dimension (n)
    
    # --- Mutation Parameters (Self-Adaptation) ---
    sigma: float  # Initial mutation strength (step size)
    
    # Learning rates for self-adaptive sigma, as per "Mutation (II)" (slide 16)
    tau: float    # Learning rate (τ)
    
    # --- Runner Parameters ---
    max_generations: int # Termination criterion: max generations
    target_fitness: float  # Termination criterion: target fitness
    
    # --- Strategy Parameters ---
    strategy: str # 'comma' for (μ,λ) or 'plus' for (μ+λ)