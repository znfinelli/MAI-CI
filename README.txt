======================================================
Evolution Strategies for Function Optimization
======================================================

Master in Artificial Intelligence - Evolutionary Computation
Authors: Toygar Yurt, Zoë Finelli, Onat Bitirgen
Date: November 10, 2025

------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------
This project implements and evaluates Evolution Strategies (ES) for 
continuous function optimization, as required for the CI-MAI 
Evolutionary Computation practical work [2].

We implement two primary selection strategies discussed in the lectures [1]:
- (μ,λ)-ES: Selection only from offspring ("comma").
- (μ+λ)-ES: Selection from both parents and offspring ("plus").

The algorithm is a from-scratch implementation that uses a simple 
self-adaptive mutation mechanism (Mutation I) with one mutation strength (σ) 
per individual, which is adapted using a single global learning rate (τ') [1].

------------------------------------------------------
2. HOW TO RUN
------------------------------------------------------

1.  Language: Python 3.x
2.  Dependencies: See `requirements.txt` (numpy, pandas, matplotlib, scipy)
3.  Installation:
    pip install -r requirements.txt

4.  Run Main Experiment:
    python main.py

------------------------------------------------------
3. EXPECTED OUTPUT
------------------------------------------------------
Running `main.py` will:
1.  Run 30 independent trials for all defined experimental configurations 
    (ES-plus, ES-comma, and L-BFGS-B baseline) for both Sphere and
    Rastrigin functions.
2.  Print summary statistics to the console.
3.  Create an `outputs/` directory containing:
    - `results.csv`: Detailed results for each of the independent runs.
    - `summary_statistics.csv`: Aggregated tables for the report.
    - `convergence_plots.png`: Convergence plot for all ES strategies.
    - `comparison_boxplots.png`: Box plots comparing all methods.

------------------------------------------------------
4. PROJECT STRUCTURE
------------------------------------------------------
evolution_strategies/
│
├── README.txt                     # This file
├── requirements.txt               # Python dependencies
├── main.py                        # Main execution script
│
├── src/                           # Source code package
│   ├── __init__.py                # Package initialization
│   ├── es_params.py               # ES parameters dataclass
│   ├── test_functions.py          # Benchmark functions
│   ├── evolution_strategy.py      # Core ES algorithm
│   ├── experiment_runner.py       # Experiment management
│   └── visualization.py           # Plotting functions
│
└── outputs/                       # Generated results (auto-created)
    ├── ... (csv and png files) ...

------------------------------------------------------
5. ALGORITHM DETAILS
------------------------------------------------------
- Representation: Real-valued vectors for object variables (x) and 
  a single real value for mutation strength (σ).
- Mutation: Self-adaptive mutation (Mutation (I)).
    - σ' = σ * exp(τ' * N(0,1))
    - x' = x + σ' * N(0,I)
- Learning Rates:
    - τ' = 1 / sqrt(2 * n)
- Recombination: None used.
- Selection: Deterministic, rank-based selection of the μ-best 
  individuals.
- Strategies:
    - `comma`: (μ,λ) - Survivors from offspring only.
    - `plus`: (μ+λ) - Survivors from parents + offspring.

6. REFERENCES
------------------------------------------------------
[1] Belanche, L. (2025). "Intro to Evolution Strategies" [Lecture Slides, 4._Intro_to_ESs.pdf].
[2] Belanche, L. (2025). "Evolutionary Computation practical work" [Project Brief, practical-exercises.html].
[4] Google. (2025). "Clean up my code with proper comments and help with efficiency and organization" [Large language model].
[4] Surjanovic, S. & Bingham, D. "Virtual Library of Simulation Experiments".
    https://www.sfu.ca/~ssurjano/optimization.html