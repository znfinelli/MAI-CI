EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION
Master in Artificial Intelligence - Evolutionary Computation
Date: November 2025

===========================================
QUICK START
===========================================

1. Create and activate a virtual environment (recommended) and install dependencies

   macOS / zsh (recommended):

   ```bash
   # create a lightweight venv in the project folder
   python3 -m venv .venv

   # activate the virtual environment (zsh)
   source .venv/bin/activate

   # install required packages from the requirements file
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you prefer to install globally (not recommended), you can instead run:

   ```bash
   pip install numpy matplotlib pandas
   ```

2. Run the experiments:

   ```bash
   # while the virtualenv is active
   python Evolution_strategies.py
   ```

3. Results will be generated in the same directory (or in the configured output path):
   - results.csv
   - summary_statistics.csv
   - convergence_sphere.png
   - convergence_rastrigin.png
   - comparison_boxplots.png

Expected runtime: ~5-10 minutes (depending on your CPU and configuration)

===========================================
PROJECT DESCRIPTION
===========================================

This project implements Evolution Strategies (ES) for continuous function 
optimization. We compare (μ,λ)-ES and (μ+λ)-ES strategies on different 
benchmark functions with varying dimensions.

IMPLEMENTED ALGORITHMS:
- (μ,λ)-ES: Selection only from offspring
- (μ+λ)-ES: Selection from parents + offspring
- Self-adaptive mutation with learning rates τ and τ'

TEST FUNCTIONS:
1. Sphere: f(x) = Σx_i² (unimodal, easy)
2. Rastrigin: multimodal with many local optima (harder)
3. Rosenbrock: narrow valley (challenging)
4. Ackley: multimodal with deep valleys

===========================================
EXPERIMENTAL DESIGN
===========================================

PARAMETERS TESTED:
- Functions: Sphere, Rastrigin
- Dimensions: 10, 20
- Population sizes: μ=15-30, λ=100-200
- Strategies: comma, plus
- Independent runs: 30 per configuration
- Max generations: 500
- Target fitness: 1e-6

PERFORMANCE METRICS:
- Best fitness achieved
- Number of generations to convergence
- Number of function evaluations
- Execution time
- Success rate (convergence to target)

===========================================
FILE STRUCTURE
===========================================

evolution_strategies.py
├── TestFunctions: Benchmark function suite
├── EvolutionStrategy: Core ES implementation
│   ├── Self-adaptive mutation
│   ├── Parent selection
│   ├── Strategy selection (comma/plus)
│   └── Statistics tracking
├── ExperimentRunner: Manages multiple trials
└── Visualization functions

===========================================
ALGORITHM DETAILS
===========================================

INITIALIZATION:
- Population: uniform random in search space
- Mutation strength σ: 0.5 (initial)

SELF-ADAPTATION:
- τ = 1/√(2n) where n = dimension
- τ' = 1/√(2√n)
- σ_new = σ * exp(τ'*N(0,1) + τ*N(0,1))
- x_new = x + σ_new * N(0,I)

SELECTION:
- (μ,λ): Select μ best from λ offspring only
- (μ+λ): Select μ best from μ parents + λ offspring

REPRODUCTION:
- Each offspring created by mutating random parent
- No recombination (crossover) used

===========================================
RESULTS INTERPRETATION
===========================================

OUTPUT FILES:

1. results.csv
   - Detailed results for each run
   - Columns: function, dimension, mu, lambda, strategy, run, 
     best_fitness, generations, function_evals, time, converged

2. summary_statistics.csv
   - Aggregated statistics per configuration
   - Mean, std, min for each metric

3. convergence_sphere.png & convergence_rastrigin.png
   - Fitness evolution over generations
   - Mean ± std across 30 runs
   - Log scale for y-axis

4. comparison_boxplots.png
   - 4 subplots comparing all configurations:
     * Best fitness distribution
     * Generations to convergence
     * Function evaluations
     * Success rate

===========================================
STATISTICAL ANALYSIS
===========================================

Each configuration is run 30 times to ensure statistical significance.

METRICS COMPUTED:
- Mean and standard deviation
- Success rate (% reaching target fitness)
- Minimum achieved fitness
- Median performance

COMPARISONS:
- (μ,λ) vs (μ+λ) strategies
- Different dimensions (10 vs 20)
- Easy (Sphere) vs Hard (Rastrigin) functions
- Impact of population size

===========================================
THEORETICAL BACKGROUND
===========================================

ADVANTAGES OF EVOLUTION STRATEGIES:
1. No gradient information needed
2. Self-adaptation of mutation parameters
3. Robust to noise and discontinuities
4. Simple and efficient implementation

EXPECTED BEHAVIOR:
- Sphere: Fast exponential convergence
- Rastrigin: Slower due to local optima
- Larger populations: More robust but slower
- (μ+λ): More stable, preserves best solutions
- (μ,λ): More exploratory, avoids premature convergence

===========================================
CUSTOMIZATION
===========================================

To test different configurations, modify the 'experiments' list in main():

experiments = [
    {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
    # Add more configurations here
]

Available functions: 'sphere', 'rastrigin', 'rosenbrock', 'ackley'
Recommended λ/μ ratio: 5-10

To change number of runs, modify:
runner = ExperimentRunner(n_runs=30)

===========================================
REQUIREMENTS
===========================================

Python version: 3.7+

Required packages:
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- pandas >= 1.1.0

All packages available via pip:
pip install numpy matplotlib pandas

===========================================
TROUBLESHOOTING
===========================================

ISSUE: "ModuleNotFoundError"
SOLUTION: Install missing packages with pip

ISSUE: Experiments too slow
SOLUTION: Reduce n_runs from 30 to 10-15, or reduce max_generations

ISSUE: Memory error
SOLUTION: Reduce dimension or population size

ISSUE: Poor convergence
SOLUTION: Increase max_generations or adjust λ/μ ratio

===========================================
CITATIONS & REFERENCES
===========================================

Evolution Strategies:
- Rechenberg, I. (1973). Evolutionsstrategie: Optimierung technischer 
  Systeme nach Prinzipien der biologischen Evolution.
- Schwefel, H.-P. (1995). Evolution and Optimum Seeking.
- Hansen, N., & Ostermeier, A. (2001). Completely derandomized 
  self-adaptation in evolution strategies. Evolutionary Computation.

Benchmark Functions:
- https://www.sfu.ca/~ssurjano/optimization.html
- Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark 
  functions for global optimization problems.

===========================================
CONTACT & SUBMISSION
===========================================

Master in Artificial Intelligence
Evolutionary Computation - Practical Work
Delivery: November 10, 2025

For questions about the implementation, refer to the inline documentation
in evolution_strategies.py

===========================================
END OF README
===========================================