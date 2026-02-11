"""
D0FUS Genetic Algorithm Optimization Module

"""

#%% Imports
import sys
import os
import numpy as np
import json
from datetime import datetime

# Plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import DEAP library
try:
    from deap import base, creator, tools, algorithms
    import random
except ImportError:
    print("\n ERROR: DEAP library not found. Please install it:")
    print("    pip install deap")
    sys.exit(1)

# Import D0FUS modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_EXE.D0FUS_run import run, Parameters, CostParameters, save_run_output
from D0FUS_BIB.D0FUS_cost import *
from D0FUS_BIB.D0FUS_cost_data import *

#%% Global Variables
opt_ranges = {}
static_inputs = {}
param_keys = []
current_generation = 0
n_total_generations = 50
current_run_directory = None

PENALTY_VALUE = 1e6

# Cost function for genetic optimization (tunable via input file)
cost_function_choice = 'COE'  # Default: Cost of Electricity

#%% DEAP Setup
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

#%% Helper Functions

def to_serializable(x):
    """Convert to JSON-serializable format"""
    if x is None:
        return None
    if callable(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        if np.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [to_serializable(item) for item in x]
    if isinstance(x, dict):
        return {str(k): to_serializable(v) for k, v in x.items() if not callable(v)}
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except:
        return None

#%% Constraint Handling

def compute_stability_penalty(nbar, nG, betaT, betaN, qstar, q_min=2.0,
                               penalty_strength=1000.0):
    """
    Strong exponential penalty for stability violations.
    
    Args:
        nbar: Average density [10^20 m^-3]
        nG: Greenwald density limit [10^20 m^-3]
        betaT: Toroidal beta (fraction, not %)
        betaN: Normalized beta limit [%]
        qstar: Kink safety factor
        q_min: Minimum safety factor (default 2.0)
        penalty_strength: Penalty multiplier
    """
    violations = {}
    penalty_terms = []
    
    # Density limit: nbar < nG
    if nG > 0:
        density_ratio = nbar / nG
        if density_ratio > 1.0:
            excess = density_ratio - 1.0
            penalty = penalty_strength * (np.exp(10 * excess) - 1)
            penalty_terms.append(penalty)
            violations['density'] = {
                'value': float(nbar), 'limit': float(nG),
                'ratio': float(density_ratio)
            }
        elif density_ratio > 0.90:
            margin_used = (density_ratio - 0.90) / 0.10
            penalty_terms.append(10 * margin_used ** 3)
    
    # Beta limit: betaT (in %) < betaN (in %)
    # betaT from D0FUS is a fraction, multiply by 100 to get %
    betaT_percent = betaT * 100
    if betaN > 0:
        beta_ratio = betaT_percent / betaN
        if beta_ratio > 1.0:
            excess = beta_ratio - 1.0
            penalty = penalty_strength * (np.exp(10 * excess) - 1)
            penalty_terms.append(penalty)
            violations['beta'] = {
                'value': float(betaT_percent), 'limit': float(betaN),
                'ratio': float(beta_ratio)
            }
        elif beta_ratio > 0.90:
            margin_used = (beta_ratio - 0.90) / 0.10
            penalty_terms.append(10 * margin_used ** 3)
    
    # Kink safety factor: qstar > q_min
    if qstar < q_min:
        deficit = (q_min - qstar) / q_min
        penalty = penalty_strength * (np.exp(10 * deficit) - 1)
        penalty_terms.append(penalty)
        violations['qstar'] = {
            'value': float(qstar), 'limit': float(q_min),
            'ratio': float(qstar / q_min)
        }
    elif qstar < q_min * 1.10:
        margin = (qstar - q_min) / (q_min * 0.10)
        penalty_terms.append(10 * (1 - margin) ** 3)
    
    total_penalty = 1.0 + sum(penalty_terms)
    is_stable = len(violations) == 0
    
    return is_stable, total_penalty, violations


def check_radial_build(cost, R0_abcd, qstar, beta, nbar):
    """Check radial build validity"""
    for val in [cost, R0_abcd, qstar, beta, nbar]:
        if val is None:
            return False, "None value"
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return False, "NaN or Inf"
            if val < 0:
                return False, f"Negative value: {val}"
    return True, None

#%% Fitness Evaluation

def evaluate_individual(individual, verbose=False):
    """Evaluate fitness of an individual"""
    try:
        param_dict = {k: individual[i] for i, k in enumerate(param_keys)}
        all_params = {**param_dict, **static_inputs}
        
        # Create Parameters object
        p = Parameters()
        for key, value in all_params.items():
            if key == 'cost_model':
                if p.cost_model != value:
                    p.cost_model = value
                    p.cost = CostParameters(value)

            elif hasattr(p.cost, key):
                setattr(p.cost, key, value)
            elif hasattr(p, key):
                setattr(p, key, value)
        
        # Call run with Parameters object
        output = run(p)
        
        # Unpack 66 values from new run
        (B0, B_CS, B_pol,
         tauE, W_th,
         Q, Volume, Surface,
         Ip, Ib, I_CD, I_Ohm,
         nbar, nG, pbar,
         betaN, betaT, betaP,
         qstar, q95,
         P_CD, P_sep, P_Thresh, eta_CD, P_elec,
         cost, P_Brem, P_syn,
         heat, heat_par, heat_pol, lambda_q, q_target,
         P_wall_H, P_wall_L,
         T_op_limit, C_invest,
         CF, COE,
         Gamma_n,
         f_alpha, tau_alpha,
         J_TF, J_CS,
         c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
         d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS_out, J_CS_out,
         r_minor, r_sep, r_c, r_d,
         kappa, kappa_95, δ, δ_95) = output
        
        # R0_abcd is the radial build validity check (r_d in new version)
        R0_abcd = r_d
        
        is_valid, reason = check_radial_build(cost, R0_abcd, qstar, betaT, nbar)
        if not is_valid:
            return (PENALTY_VALUE,)
        
        is_stable, penalty_multiplier, violations = compute_stability_penalty(
            nbar, nG, betaT, betaN, qstar, q_min=2.0, penalty_strength=1000.0
        )
        
        # Extract cost function value based on user's choice
        try:
            cost_value = get_cost_value(
                cost_function_choice,
                COE=COE,
                C_invest=C_invest,
                Cost=cost
            )
        except ValueError as e:
            if verbose:
                print(f"Error getting cost value: {e}")
            return (PENALTY_VALUE,)
        
        fitness = cost_value * penalty_multiplier
        return (fitness,)
    
    except Exception as e:
        if verbose:
            print(f"Error in evaluate_individual: {e}")
        return (PENALTY_VALUE,)

#%% Cost Function Selection

# Available cost functions with their properties
COST_FUNCTIONS = {
    'COE': {
        'name': 'Cost of Electricity',
        'unit': '€/MWh',
        'minimize': True,
        'description': 'Optimize for lowest cost of electricity'
    },
    'C_invest': {
        'name': 'Invested Capital Cost',
        'unit': 'B€',
        'minimize': True,
        'description': 'Optimize for lowest capital investment'
    },
    'Cost': {
        'name': 'Volume Cost Proxy',
        'unit': 'm³',
        'minimize': True,
        'description': 'Optimize for smallest reactor volume (cost proxy)'
    }
}

def get_cost_value(cost_func_name, **outputs):
    """
    Extract the appropriate cost function value from outputs.
    
    Args:
        cost_func_name: Name of cost function ('COE', 'C_invest', etc.)
        **outputs: Named output values from run()
    
    Returns:
        float: Cost function value (adjusted for minimization)
        
    Raises:
        ValueError: If cost function name is invalid
    """
    if cost_func_name not in COST_FUNCTIONS:
        valid = ', '.join(COST_FUNCTIONS.keys())
        raise ValueError(f"Invalid cost function '{cost_func_name}'. Valid options: {valid}")
    
    # Get the value
    value = outputs.get(cost_func_name)
    if value is None:
        raise ValueError(f"Cost function '{cost_func_name}' not found in outputs")
    
    return value

#%% I/O Management

def initialize_run_directory(base_directory="D0FUS_OUTPUTS/genetic"):
    """Initialize run directory"""
    global current_run_directory
    timestamp = datetime.now().strftime("Genetic_D0FUS_%Y%m%d_%H%M%S")
    current_run_directory = os.path.join(base_directory, timestamp)
    os.makedirs(current_run_directory, exist_ok=True)
    print(f"\n Output directory: {current_run_directory}")
    return current_run_directory

#%% Input File Parsing

def load_input_file(input_file):
    """Load and parse input file"""
    global opt_ranges, static_inputs, param_keys, cost_function_choice
    
    opt_ranges = {}
    static_inputs = {}
    defaults = Parameters()
    
    with open(input_file, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.split('#')[0].strip()
            if not line or ('=' not in line and ':' not in line):
                continue
            
            parts = line.split('=', 1) if '=' in line else line.split(':', 1)
            if len(parts) != 2:
                continue
            
            key, value = parts[0].strip(), parts[1].strip()
            if not key or not value:
                continue
            
            if '[' in value and ']' in value:
                bracket_start = value.index('[')
                bracket_end = value.index(']')
                values_str = value[bracket_start+1:bracket_end].strip()
                values = [v.strip() for v in values_str.replace(';', ',').split(',') if v.strip()]
                
                if len(values) == 2:
                    try:
                        min_val, max_val = float(values[0]), float(values[1])
                        opt_ranges[key] = (min_val, max_val)
                        print(f"  {key} = [{min_val}, {max_val}] (optimize)")
                    except ValueError:
                        pass
            else:
                try:
                    static_inputs[key] = float(value)
                except ValueError:
                    static_inputs[key] = value
    
    # Fill defaults (filter out methods)
    for attr in dir(defaults):
        if attr.startswith('_') or attr in opt_ranges or attr in static_inputs:
            continue
        default_val = getattr(defaults, attr)
        if callable(default_val):
            continue
        if isinstance(default_val, (int, float, str, bool)):
            static_inputs[attr] = default_val
        elif isinstance(default_val, np.ndarray):
            static_inputs[attr] = default_val.tolist()
        elif isinstance(default_val, (list, tuple)):
            static_inputs[attr] = list(default_val)
    
    param_keys = list(opt_ranges.keys())
    
    # Handle cost_function choice
    if 'cost_function' in static_inputs:
        cost_func = static_inputs['cost_function']
        if isinstance(cost_func, str):
            if cost_func in COST_FUNCTIONS:
                cost_function_choice = cost_func
                print(f"\n Cost function: {cost_func} ({COST_FUNCTIONS[cost_func]['name']})")
            else:
                valid = ', '.join(COST_FUNCTIONS.keys())
                print(f"\n WARNING: Invalid cost_function '{cost_func}'")
                print(f"   Valid options: {valid}")
                print(f"   Using default: COE")
                cost_function_choice = 'COE'
        # Remove from static_inputs since it's not a run() parameter
        del static_inputs['cost_function']
    else:
        # Use default
        cost_function_choice = 'COE'
        print(f"\n Cost function: COE (default)")
    
    print(f"\n Optimization parameters: {len(param_keys)}")
    for key, (lo, hi) in opt_ranges.items():
        print(f"    {key}: [{lo}, {hi}]")
    
    return opt_ranges, static_inputs, param_keys

#%% Genetic Algorithm with Enhanced Exploration

def setup_toolbox(opt_ranges, param_keys, n_generations):
    """Setup DEAP toolbox with exploration-focused operators"""
    global toolbox, n_total_generations
    n_total_generations = n_generations
    
    # Latin Hypercube-like initialization for better coverage
    def create_individual_lhs():
        """Create individual with better space coverage"""
        ind = []
        for k in param_keys:
            lo, hi = opt_ranges[k]
            # Add small random perturbation to avoid grid patterns
            val = lo + random.random() * (hi - lo)
            ind.append(val)
        return creator.Individual(ind)
    
    toolbox.register("individual", create_individual_lhs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    
    # Crossover: blend with wider alpha for more exploration
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[opt_ranges[k][0] for k in param_keys],
                     up=[opt_ranges[k][1] for k in param_keys],
                     eta=10.0)  # Lower eta = more spread
    
    # Enhanced mutation for better exploration
    def exploratory_mutation(individual, indpb=0.4):
        """
        Mutation with exploration/exploitation balance.
        Early: large jumps to explore
        Late: small refinements
        """
        progress = current_generation / max(n_total_generations, 1)
        
        # Exploration probability decreases over time
        explore_prob = 0.3 * (1 - progress)
        
        for i, key in enumerate(param_keys):
            if random.random() < indpb:
                lo, hi = opt_ranges[key]
                
                if random.random() < explore_prob:
                    # EXPLORATION: Jump to random location in range
                    individual[i] = lo + random.random() * (hi - lo)
                else:
                    # EXPLOITATION: Polynomial mutation with adaptive eta
                    eta = 5 + 50 * progress  # Start wide, narrow down
                    x = individual[i]
                    delta = hi - lo
                    
                    u = random.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    individual[i] = x + delta_q * delta
                    individual[i] = max(lo, min(hi, individual[i]))
        
        return (individual,)
    
    toolbox.register("mutate", exploratory_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)


def inject_diverse_individuals(pop, n_inject, opt_ranges, param_keys):
    """Inject random individuals to maintain diversity"""
    for _ in range(n_inject):
        new_ind = creator.Individual([
            random.uniform(opt_ranges[k][0], opt_ranges[k][1])
            for k in param_keys
        ])
        # Evaluate the new individual first
        new_ind.fitness.values = toolbox.evaluate(new_ind)
        
        # Find worst individual (only among those with valid fitness)
        valid_indices = [i for i in range(len(pop)) 
                        if pop[i].fitness.valid and len(pop[i].fitness.values) > 0]
        
        if valid_indices:
            worst_idx = max(valid_indices, key=lambda i: pop[i].fitness.values[0])
            # Only replace if new individual is better
            if new_ind.fitness.values[0] < pop[worst_idx].fitness.values[0]:
                pop[worst_idx] = new_ind
        else:
            # If no valid individuals, just append
            pop[0] = new_ind
    
    return pop


def run_genetic_algorithm(pop_size, n_generations, cxpb, mutpb, 
                          patience=20, verbose=True):
    """
    Run GA with enhanced exploration and diversity maintenance
    """
    global current_generation
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)
    
    # Statistics
    def safe_mean(values):
        valid = [v[0] for v in values if v[0] < PENALTY_VALUE / 2]
        return np.mean(valid) if valid else float('nan')
    
    def safe_std(values):
        valid = [v[0] for v in values if v[0] < PENALTY_VALUE / 2]
        return np.std(valid) if len(valid) > 1 else float('nan')
    
    def count_valid(values):
        return sum(1 for v in values if v[0] < PENALTY_VALUE / 2)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: min(v[0] for v in x))
    stats.register("avg", safe_mean)
    stats.register("std", safe_std)
    stats.register("n_valid", count_valid)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'min', 'avg', 'std', 'n_valid']
    
    print(f"\n{'='*70}")
    print(f" Genetic Algorithm - Exploration Mode")
    print(f"{'='*70}")
    print(f"    Population: {pop_size}")
    print(f"    Generations: {n_generations}")
    print(f"    Crossover: {cxpb}, Mutation: {mutpb}")
    print(f"    Diversity injection every 10 generations")
    print(f"{'='*70}\n")
    
    # Initial evaluation
    current_generation = 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    
    if verbose:
        print(logbook.stream)
    
    # Evolution
    mu = pop_size
    lambda_ = int(pop_size * 1.5)  # More offspring for diversity
    
    best_fitness_history = []
    stagnation_counter = 0
    
    for gen in range(1, n_generations + 1):
        current_generation = gen
        
        # Select and create offspring
        offspring = toolbox.select(pop, lambda_)
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values
        
        # Mutation (higher rate for exploration)
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Mu+Lambda selection
        pop[:] = tools.selBest(pop + offspring, mu)
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        if verbose:
            print(logbook.stream)
        
        # Track stagnation
        current_best = record['min']
        best_fitness_history.append(current_best)
        
        if len(best_fitness_history) > 1:
            if abs(best_fitness_history[-1] - best_fitness_history[-2]) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
        
        # Inject diversity if stagnating or periodically
        if stagnation_counter >= 5 or gen % 10 == 0:
            n_inject = max(2, pop_size // 20)
            pop = inject_diverse_individuals(pop, n_inject, opt_ranges, param_keys)
            if stagnation_counter >= 5:
                stagnation_counter = 0
                if verbose:
                    print(f"    [Diversity injection: {n_inject} new individuals]")
        
        # Early stopping
        if stagnation_counter >= patience:
            print(f"\n Converged at generation {gen}")
            break
    
    return hof[0], logbook, hof

#%% Plotting

def plot_convergence(logbook, save_path):
    """Single convergence plot in log scale"""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gens = [record['gen'] for record in logbook]
    mins = [record['min'] for record in logbook]
    n_valid = [record['n_valid'] for record in logbook]
    
    # Main convergence curve
    ax.semilogy(gens, mins, 'o-', color='#2ecc71', linewidth=2.5, 
                markersize=5, label='Best fitness')
    ax.fill_between(gens, mins, alpha=0.2, color='#2ecc71')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (log scale)')
    ax.set_title('D0FUS Genetic Algorithm - Convergence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Annotate final value
    final_fitness = mins[-1]
    ax.annotate(f'Final: {final_fitness:.4f}', 
                xy=(gens[-1], final_fitness),
                xytext=(0.7, 0.9), textcoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='green'),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Add valid count on secondary axis
    ax2 = ax.twinx()
    ax2.bar(gens, n_valid, alpha=0.3, color='blue', width=0.8)
    ax2.set_ylabel('Valid designs per generation', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Convergence plot saved: {save_path}")

#%% Main Interface

def run_genetic_optimization(input_file, 
                             population_size=200,
                             generations=50,
                             crossover_rate=0.7,
                             mutation_rate=0.3,  # Higher for exploration
                             patience=20,
                             seed=None,
                             verbose=True):
    """
    Main optimization function
    """
    global current_run_directory
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Random seed: {seed}")
    
    print("="*80)
    print("D0FUS GENETIC ALGORITHM OPTIMIZATION")
    print("="*80)
    
    # Initialize
    initialize_run_directory()
    opt_ranges, static_inputs, param_keys = load_input_file(input_file)
    
    if len(param_keys) < 1:
        print("\n ERROR: Need at least 1 optimization parameter")
        sys.exit(1)
    
    setup_toolbox(opt_ranges, param_keys, generations)
    
    # Run optimization
    best, logbook, hof = run_genetic_algorithm(
        population_size, generations, crossover_rate, mutation_rate,
        patience, verbose
    )
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    best_params = {k: best[i] for i, k in enumerate(param_keys)}
    print("\n Best parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value:.6f}")
    
    # Get full output for best design
    all_params = {**best_params, **static_inputs}
    
    # Create Parameters object
    p = Parameters()
    for key, value in all_params.items():
        if key == 'cost_model':
            if p.cost_model != value:
                p.cost_model = value
                p.cost = CostParameters(value)
        elif hasattr(p.cost, key):
            setattr(p.cost, key, value)
        elif hasattr(p, key):
            setattr(p, key, value)
    
    # Call run with Parameters object
    final_output = run(p)
    
    # Unpack 66 values from new run
    (B0, B_CS, B_pol,
     tauE, W_th,
     Q, Volume, Surface,
     Ip, Ib, I_CD, I_Ohm,
     nbar, nG, pbar,
     betaN, betaT, betaP,
     qstar, q95,
     P_CD, P_sep, P_Thresh, eta_CD, P_elec,
     cost, P_Brem, P_syn,
     heat, heat_par, heat_pol, lambda_q, q_target,
     P_wall_H, P_wall_L,
     T_op_limit, C_invest,
     CF, COE,
     Gamma_n,
     f_alpha, tau_alpha,
     J_TF, J_CS,
     c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
     d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS_out, J_CS_out,
     r_minor, r_sep, r_c, r_d,
     kappa, kappa_95, δ, δ_95) = final_output
    
    # Check stability (variables already unpacked above)
    is_stable, _, violations = compute_stability_penalty(nbar, nG, betaT, betaN, qstar)
    
    # betaT is fraction, convert to % for display
    betaT_percent = betaT * 100
    
    # Get cost function value for display
    cost_func_value = get_cost_value(
        cost_function_choice,
        COE=COE,
        C_invest=C_invest,
        Cost=cost
    )
    
    print(f"\n Best design metrics:")
    print(f"    Cost function ({cost_function_choice}): {cost_func_value:.4f} {COST_FUNCTIONS[cost_function_choice]['unit']}")
    print(f"    Cost (volume proxy): {cost:.4f} m³")
    print(f"    Q factor: {Q:.2f}")
    print(f"    P_elec: {P_elec:.1f} MW")
    print(f"    n/nG: {nbar/nG:.3f} ({(1-nbar/nG)*100:+.1f}% margin)")
    print(f"    betaT/betaN: {betaT_percent/betaN:.3f} ({(1-betaT_percent/betaN)*100:+.1f}% margin)")
    print(f"    q*/2: {qstar/2:.3f} ({(qstar/2-1)*100:+.1f}% margin)")
    
    if is_stable:
        print("\n STABLE design found!")
    else:
        print("\n WARNING: Design violates constraints:")
        for name, details in violations.items():
            print(f"      {name}: {details['value']:.3f} vs limit {details['limit']:.3f}")
    
    # === Save outputs ===
    
    # 1. Convergence plot
    plot_path = os.path.join(current_run_directory, "convergence.png")
    plot_convergence(logbook, plot_path)
    
    # 2. Use D0FUS save_run_output for complete design documentation
    # Create a Parameters object with the best values
    p = Parameters()
    for key, value in all_params.items():
        if hasattr(p, key):
            setattr(p, key, value)
            
    # Explicitly propagate optimized cost parameters
    for i, key in enumerate(param_keys):
        if hasattr(p.cost, key):
            setattr(p.cost, key, best[i])

    
    # Save using D0FUS format (creates output_results.txt)
    save_run_output(p, final_output, current_run_directory, None)
    
    # 3. Save optimization summary JSON
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "cost_function": {
            "name": cost_function_choice,
            "description": COST_FUNCTIONS[cost_function_choice]['name'],
            "unit": COST_FUNCTIONS[cost_function_choice]['unit'],
            "value": to_serializable(cost_func_value),
            "minimize": COST_FUNCTIONS[cost_function_choice]['minimize']
        },
        "settings": {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate
        },
        "optimized_parameters": {k: to_serializable(v) for k, v in best_params.items()},
        "best_fitness": to_serializable(best.fitness.values[0]),
        "is_stable": is_stable,
        "stability_margins": {
            "n_over_nG": to_serializable(nbar/nG),
            "qstar_over_2": to_serializable(qstar/2),
            "betaT_over_betaN": to_serializable(betaT_percent/betaN) if betaN > 0 else None
        },
        "hall_of_fame": [
            {
                "rank": i+1,
                "fitness": to_serializable(ind.fitness.values[0]),
                "params": {k: to_serializable(ind[j]) for j, k in enumerate(param_keys)}
            }
            for i, ind in enumerate(hof)
        ]
    }
    
    summary_path = os.path.join(current_run_directory, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n All outputs saved to: {current_run_directory}/")
    
    return best, final_output, hof

#%% Command Line Interface

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python D0FUS_genetic.py <input_file> [options]")
        print("\nOptions:")
        print("  --pop N      Population size (default: 200)")
        print("  --gen N      Max generations (default: 50)")
        print("  --mut F      Mutation rate (default: 0.3)")
        print("  --seed N     Random seed for reproducibility")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Parse arguments
    pop_size = 200
    n_gen = 50
    mut_rate = 0.3
    seed = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--pop" and i + 1 < len(sys.argv):
            pop_size = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--gen" and i + 1 < len(sys.argv):
            n_gen = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--mut" and i + 1 < len(sys.argv):
            mut_rate = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    run_genetic_optimization(
        input_file,
        population_size=pop_size,
        generations=n_gen,
        mutation_rate=mut_rate,
        seed=seed,
        verbose=True
    )


if __name__ == "__main__":
    main()