"""
D0FUS Genetic Algorithm Optimization Module
============================================
Adapted to the GlobalConfig / dc_replace architecture of D0FUS_run (v2).

"""

#%% Imports
import sys
import os
import numpy as np
import json
from datetime import datetime
from dataclasses import replace as dc_replace, asdict

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
from D0FUS_BIB.D0FUS_parameterization import GlobalConfig, DEFAULT_CONFIG
from D0FUS_EXE.D0FUS_run import run, save_run_output

# =============================================================================
# run() return-tuple index map  (v2 — 81 outputs)
#
# This dictionary is the SINGLE SOURCE OF TRUTH for mapping symbolic names
# to positional indices in the tuple returned by run().  Both
# evaluate_individual() and debug_single_run() reference it so that a
# change in run()'s return ordering only requires an update here.
# =============================================================================
_IDX = {
    'B0':        0,
    'B_CS':      1,
    'B_pol':     2,
    'tauE':      3,
    'W_th':      4,
    'Q':         5,
    'Volume':    6,
    'Surface':   7,
    'Ip':        8,
    'Ib':        9,
    'I_CD':     10,
    'I_Ohm':    11,
    'nbar':     12,   # Volume-averaged density [10²⁰ m⁻³]
    'nbar_line':13,   # Line-averaged density [10²⁰ m⁻³]
    'nG':       14,   # Greenwald density limit [10²⁰ m⁻³]
    'pbar':     15,
    'betaN':    16,   # Normalized beta [% m T / MA]
    'betaT':    17,   # Toroidal beta (fraction, ×100 for %)
    'betaP':    18,
    'qstar':    19,   # Kink safety factor
    'q95':      20,
    'P_CD':     21,
    'P_sep':    22,
    'P_Thresh': 23,
    'eta_CD':   24,
    'P_elec':   25,
    'P_wallplug': 26,
    'cost':     27,   # Machine volume proxy [m³]
    'P_Brem':   28,
    'P_syn':    29,
    'P_line':   30,
    'P_line_core': 31,
    'heat':     32,
    'heat_par': 33,
    'heat_pol': 34,
    'lambda_q': 35,
    'q_target': 36,
    'P_wall_H': 37,
    'P_wall_L': 38,
    'Gamma_n':  39,
    'f_alpha':  40,
    'tau_alpha': 41,
    'J_TF':     42,
    'J_CS':     43,
    'c':        44,
    'c_WP_TF':  45,
    'c_Nose_TF':46,
    'sigma_z_TF':   47,
    'sigma_th_TF':  48,
    'sigma_r_TF':   49,
    'Steel_frac_TF':50,
    'd':        51,
    'sigma_z_CS':   52,
    'sigma_th_CS':  53,
    'sigma_r_CS':   54,
    'Steel_frac_CS':55,
    'B_CS_dup': 56,
    'J_CS_dup': 57,
    'r_minor':  58,   # R0 - a
    'r_sep':    59,   # R0 - a - b
    'r_c':      60,   # R0 - a - b - c
    'r_d':      61,   # R0 - a - b - c - d  (must be > 0)
    'kappa':    62,
    'kappa_95': 63,
    'delta':    64,
    'delta_95': 65,
    'PsiPI':    66,
    'PsiRU':    67,
    'PsiPlat':  68,
    'PsiPF':    69,
    'PsiCS':    70,
    'eta_LH':   71,
    'eta_EC':   72,
    'eta_NBI':  73,
    'P_LH':     74,
    'P_EC':     75,
    'P_NBI':    76,
    'P_ICR':    77,
    'I_LH':     78,
    'I_EC':     79,
    'I_NBI':    80,
}

#%% Global Variables (default values)
opt_ranges = {}
static_inputs = {}
param_keys = []
current_generation = 0
n_total_generations = 50
current_run_directory = None

PENALTY_VALUE = 1e6

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

def compute_stability_penalty(nbar_line, nG, betaT, betaN, qstar,
                               q_min=DEFAULT_CONFIG.q_limit,
                               penalty_strength=1000.0):
    """
    Strong exponential penalty for stability violations.

    Parameters
    ----------
    nbar_line : float
        Line-averaged density [10²⁰ m⁻³].  The Greenwald limit is defined
        in line-averaged density, so nbar_line (not nbar_vol) must be used.
    nG : float
        Greenwald density limit [10²⁰ m⁻³].
    betaT : float
        Toroidal beta (fraction; multiply by 100 for %).
    betaN : float
        Normalized beta limit [% m T / MA].
    qstar : float
        Kink safety factor.
    q_min : float
        Minimum safety factor.
    penalty_strength : float
        Penalty multiplier.
    """
    violations = {}
    penalty_terms = []
    
    # Density limit: nbar_line < nG
    if nG > 0:
        density_ratio = nbar_line / nG
        if density_ratio > 1.0:
            excess = density_ratio - 1.0
            penalty = penalty_strength * (np.exp(10 * excess) - 1)
            penalty_terms.append(penalty)
            violations['density'] = {
                'value': float(nbar_line), 'limit': float(nG),
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


def check_radial_build(cost, r_d, c_TF, d_CS, qstar, betaT, nbar_line):
    """
    Check radial build validity for a candidate design.

    Parameters
    ----------
    cost      : float  Machine cost proxy [m³].
    r_d       : float  Innermost radial build radius R0 - a - b - c - d [m].
    c_TF      : float  TF inboard radial thickness [m].
    d_CS      : float  CS radial thickness [m].
    qstar     : float  Kink safety factor.
    betaT     : float  Toroidal beta (fraction).
    nbar_line : float  Line-averaged density [10²⁰ m⁻³].

    Returns
    -------
    (bool, str or None)
        (True, None) if the design is geometrically valid,
        (False, reason) otherwise.
    """
    # All key scalars must be finite and non-negative
    for name, val in [('cost', cost), ('r_d', r_d), ('c_TF', c_TF),
                      ('d_CS', d_CS), ('qstar', qstar), ('betaT', betaT),
                      ('nbar_line', nbar_line)]:
        if val is None:
            return False, f"{name} is None"
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return False, f"{name} is NaN/Inf"
            if val < 0:
                return False, f"{name} = {val:.4g} < 0"

    # TF and CS thicknesses must be physically meaningful (> 1 mm)
    if c_TF < 1e-3:
        return False, f"c_TF = {c_TF:.4g} m (too thin)"
    if d_CS < 1e-3:
        return False, f"d_CS = {d_CS:.4g} m (too thin / no CS)"

    return True, None

#%% Fitness Evaluation

def _safe_real(value):
    """
    Coerce a scalar to a real float, returning NaN for complex / non-finite.

    Some physics functions produce complex128 when a square root receives a
    negative argument (unphysical parameter combination).  Rather than
    letting the TypeError propagate, we silently map these to NaN so the
    radial-build validity check rejects the design cleanly.
    """
    if isinstance(value, (complex, np.complexfloating)):
        return np.nan
    try:
        v = float(np.real(value))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def evaluate_individual(individual, verbose=False):
    """
    Evaluate the fitness of a single individual.

    Uses the centralised ``_IDX`` map for all tuple indexing so that any
    change in run()'s return ordering is automatically propagated.
    """
    import warnings

    try:
        param_dict = {k: individual[i] for i, k in enumerate(param_keys)}
        all_params  = {**param_dict, **static_inputs}

        # Build an immutable GlobalConfig, filtering out non-field keys
        config = GlobalConfig(**{k: v for k, v in all_params.items()
                                  if k in GlobalConfig.__dataclass_fields__})

        # Suppress numpy warnings: unphysical parameter combinations
        # routinely produce sqrt(negative), 0/0, overflow — all expected.
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            output = run(config, verbose=0)

        # ── Guard against any residual non-finite output ─────────────────
        cost      = _safe_real(output[_IDX['cost']])
        nbar_line = _safe_real(output[_IDX['nbar_line']])
        nG        = _safe_real(output[_IDX['nG']])
        betaN     = _safe_real(output[_IDX['betaN']])
        betaT     = _safe_real(output[_IDX['betaT']])
        qstar     = _safe_real(output[_IDX['qstar']])
        R0_abcd   = _safe_real(output[_IDX['r_d']])
        c_TF      = _safe_real(output[_IDX['c']])
        d_CS      = _safe_real(output[_IDX['d']])

        # ── Radial build validity check ──────────────────────────────────
        is_valid, reason = check_radial_build(
            cost, R0_abcd, c_TF, d_CS, qstar, betaT, nbar_line)
        if not is_valid:
            return (PENALTY_VALUE,)

        # ── Stability penalty ────────────────────────────────────────────
        q_lim = static_inputs.get('q_limit', DEFAULT_CONFIG.q_limit)
        is_stable, penalty_multiplier, violations = compute_stability_penalty(
            nbar_line, nG, betaT, betaN, qstar,
            q_min=q_lim, penalty_strength=1000.0
        )

        fitness = cost * penalty_multiplier
        return (fitness,)

    except Exception:
        return (PENALTY_VALUE,)

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
    global opt_ranges, static_inputs, param_keys
    
    opt_ranges = {}
    static_inputs = {}

    # Use asdict to iterate over DEFAULT_CONFIG fields cleanly
    default_dict = asdict(DEFAULT_CONFIG)
    
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

    # Fill defaults from GlobalConfig dataclass fields
    for field_name, default_val in default_dict.items():
        if field_name in opt_ranges or field_name in static_inputs:
            continue
        if isinstance(default_val, (int, float, str, bool)):
            static_inputs[field_name] = default_val
        elif isinstance(default_val, np.ndarray):
            static_inputs[field_name] = default_val.tolist()
        elif isinstance(default_val, (list, tuple)):
            static_inputs[field_name] = list(default_val)
        # Skip None and other non-serializable types
    
    param_keys = list(opt_ranges.keys())
    
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

    # ── Diagnostic: abort early if entire initial population is invalid ───
    n_init_valid = sum(1 for f in fitnesses if f[0] < PENALTY_VALUE / 2)
    if n_init_valid == 0:
        print(f"\n  *** WARNING: 0 / {pop_size} initial designs are valid! ***")
        print(f"  The parameter ranges likely produce unphysical configurations")
        print(f"  (complex-valued sqrt, negative radial build, etc.).")
        print(f"  Suggestions:")
        print(f"    1. Narrow the optimisation ranges (especially R0, a, Bmax_TF)")
        print(f"    2. Run debug_single_run() with a known-good parameter set")
        print(f"    3. Check that static_inputs in the input file are consistent")
        print(f"  Attempting to continue with diversity injection...\n")
    elif verbose:
        print(f"  Initial population: {n_init_valid}/{pop_size} valid designs")
    
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

    # Override GA hyperparameters with values parsed from the input file.
    # These keys are removed from static_inputs so they are not passed to
    # GlobalConfig as physics parameters.
    if 'population_size' in static_inputs:
        population_size = int(static_inputs.pop('population_size'))
        print(f"  [input file] population_size = {population_size}")
    if 'generations' in static_inputs:
        generations = int(static_inputs.pop('generations'))
        print(f"  [input file] generations     = {generations}")
    if 'crossover_rate' in static_inputs:
        crossover_rate = float(static_inputs.pop('crossover_rate'))
        print(f"  [input file] crossover_rate  = {crossover_rate}")
    if 'mutation_rate' in static_inputs:
        mutation_rate = float(static_inputs.pop('mutation_rate'))
        print(f"  [input file] mutation_rate   = {mutation_rate}")

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
    
    # ── Full output for best design ──────────────────────────────────────
    all_params = {**best_params, **static_inputs}
    
    config = GlobalConfig(**{k: v for k, v in all_params.items()
                         if k in GlobalConfig.__dataclass_fields__})
    final_output = run(config, verbose=0)
    
    # Extract key metrics via centralised index map
    nbar_line = final_output[_IDX['nbar_line']]
    nG        = final_output[_IDX['nG']]
    betaN     = final_output[_IDX['betaN']]
    betaT     = final_output[_IDX['betaT']]
    qstar     = final_output[_IDX['qstar']]
    cost      = final_output[_IDX['cost']]
    Q         = final_output[_IDX['Q']]
    P_elec    = final_output[_IDX['P_elec']]
    c_TF      = final_output[_IDX['c']]
    d_CS      = final_output[_IDX['d']]
    r_d       = final_output[_IDX['r_d']]
    Ip        = final_output[_IDX['Ip']]

    # Check stability (using line-averaged density for Greenwald comparison)
    is_stable, _, violations = compute_stability_penalty(nbar_line, nG, betaT, betaN, qstar)
    
    # betaT is fraction, convert to % for display
    betaT_percent = betaT * 100
    
    print(f"\n Best design metrics:")
    print(f"    Cost: {cost:.4f}")
    print(f"    Q factor: {Q:.2f}")
    print(f"    Ip: {Ip:.2f} MA")
    print(f"    P_elec: {P_elec:.1f} MW")
    print(f"    n_line/nG: {nbar_line/nG:.3f} ({(1-nbar_line/nG)*100:+.1f}% margin)")
    print(f"    betaT/betaN: {betaT_percent/betaN:.3f} ({(1-betaT_percent/betaN)*100:+.1f}% margin)")
    q_lim = static_inputs.get('q_limit', DEFAULT_CONFIG.q_limit)
    print(f"    q*/q_limit: {qstar/q_lim:.3f} ({(qstar/q_lim-1)*100:+.1f}% margin)")
    print(f"    c_TF: {c_TF:.3f} m   d_CS: {d_CS:.3f} m   r_d: {r_d:.3f} m")

    # Radial build sanity check on final design
    _rb_ok, _rb_reason = check_radial_build(
        cost, r_d, c_TF, d_CS, qstar, betaT, nbar_line)
    if not _rb_ok:
        print(f"\n  WARNING: Best design fails radial build check: {_rb_reason}")
    
    if is_stable and _rb_ok:
        print("\n STABLE + BUILDABLE design found!")
    elif is_stable:
        print("\n STABLE but NOT BUILDABLE design")
    else:
        print("\n WARNING: Design violates constraints:")
        for name, details in violations.items():
            print(f"      {name}: {details['value']:.3f} vs limit {details['limit']:.3f}")
    
    # === Save outputs ===
    
    # 1. Convergence plot
    plot_path = os.path.join(current_run_directory, "convergence.png")
    plot_convergence(logbook, plot_path)
    
    # 2. Use D0FUS save_run_output for complete design documentation
    save_run_output(config, final_output, current_run_directory, None)
    
    # 3. Save optimization summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "settings": {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate
        },
        "optimized_parameters": {k: to_serializable(v) for k, v in best_params.items()},
        "best_fitness": to_serializable(best.fitness.values[0]),
        "is_stable": is_stable,
        "is_buildable": _rb_ok,
        "stability_margins": {
            "n_line_over_nG": to_serializable(nbar_line/nG),
            "qstar_over_qlim": to_serializable(qstar/q_lim),
            "betaT_over_betaN": to_serializable(betaT_percent/betaN) if betaN > 0 else None
        },
        "radial_build": {
            "c_TF_m": to_serializable(c_TF),
            "d_CS_m": to_serializable(d_CS),
            "r_d_m":  to_serializable(r_d),
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



#%% Debug Helper

def debug_single_run(a=2.0, R0=6.0, Bmax_TF=12.0, Tbar=15.0):
    """
    Run a single D0FUS evaluation with verbose output to diagnose
    why evaluate_individual returns PENALTY_VALUE for all individuals.

    Call this from a Python shell BEFORE launching the GA:

        from D0FUS_genetic import debug_single_run, load_input_file
        load_input_file("my_input.txt")
        debug_single_run(a=1.5, R0=5.0, Bmax_TF=14.0, Tbar=12.0)
    """
    import traceback
    test_params = {"a": a, "R0": R0, "Bmax_TF": Bmax_TF, "Tbar": Tbar}
    print(f"\n=== debug_single_run ===")
    print(f"Parameters: {test_params}")
    try:
        all_params = {**test_params, **static_inputs}
        config = GlobalConfig(**{k: v for k, v in all_params.items()
                                  if k in GlobalConfig.__dataclass_fields__})
        print("GlobalConfig created successfully.")
        output = run(config, verbose=0)
        print(f"run() returned {len(output)} values.")

        # Use the centralised index map for diagnostic output
        diag_keys = ['B0', 'Q', 'Volume', 'nbar', 'nbar_line', 'nG',
                     'betaN', 'betaT', 'qstar', 'q95', 'cost', 'J_TF', 'r_d']
        for key in diag_keys:
            idx = _IDX[key]
            print(f"  output[{idx:2d}] {key:15s} = {output[idx]}")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

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