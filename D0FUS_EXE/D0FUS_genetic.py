"""
D0FUS Genetic Algorithm Optimization Module
===========================================
Memetic optimization driver for the D0FUS (Design 0-dimensional for
FUsion Systems) project. Built on the GlobalConfig / dataclasses.replace
architecture of D0FUS_run v2.

Algorithm
---------
Global exploration is handled by a DEAP-based genetic algorithm with
simulated-binary crossover, an adaptive polynomial mutation that shifts
from exploration-wide jumps to exploitation-narrow steps as generations
progress, tournament-based mu+lambda replacement, and unconditional
diversity injection every 7 generations (or whenever stagnation is
detected over 3 consecutive generations). A Hall-of-Fame preserves the
overall best designs across the run, with one minimal-elitism step per
generation guaranteeing the current champion is never lost.

Local exploitation is provided by an optional memetic refinement applied
to the top-k Hall-of-Fame members at the end of the GA and, optionally,
to the current best every N generations during the GA. Three local
optimisers are available, all working in normalised [0,1]^d coordinates
for scale invariance:
  - 'gradient'     projected gradient descent with centered finite
                   differences and Armijo backtracking line search.
                   Pedagogical, transparent, and parallelisable.
  - 'lbfgsb'       scipy.optimize.minimize with method 'L-BFGS-B' and
                   bound constraints. Quasi-Newton, fast in smooth
                   regions; accepts a parallel Jacobian when a pool is
                   available.
  - 'nelder-mead'  scipy.optimize.minimize with method 'Nelder-Mead'
                   and bound support. Derivative-free simplex, robust
                   to the step of the fitness at the constraint
                   boundaries (feasibility band).

Constraints
-----------
Feasibility follows the shared RUN rule (D0FUS_run.evaluate_feasibility),
so a design ranked feasible here is reported FEASIBLE by RUN:
  - Density limit (model-selectable)    nbar_line / nG             <= 1
    where nG is the effective cap of config.density_limit_model
    ('greenwald' | 'giacomin' | 'zanca') scaled by Greenwald_limit
  - Troyon beta limit                   betaN_total / betaN_limit  <= 1
  - Kink safety factor                  q_limit / q_kink           <= 1
    (q_kink = q* or q95 according to config.kink_parameter)
  - Plasma-current ceiling (optional)   Ip / Ip_limit              <= 1
  - Radial build closure                r_d > 0, c_TF and d_CS > 1 mm
A design whose radial build does not close receives the PENALTY_VALUE
sentinel. The other designs are ranked feasible first, on the objective,
then infeasible ones by their total violation (Deb's rules, see
compute_feasibility_fitness). The capital-cost ceiling C_invest_max is
soft: the objective is multiplied by 1 + 3 x^2, x being the relative
overshoot of C_invest.

Fitness objectives
------------------
Selected via 'fitness_objective' in the input file:
  - 'COE'       cost of electricity (Sheffield 2016, EUR/MWh) under the
                soft C_invest_max ceiling. Default.
  - 'volume'    machine-volume proxy (V_BB + V_TF + V_CS) / P_fus.
  - 'C_invest'  total capital cost (Sheffield 2016, M EUR).
  - 'P_elec'    net electric power (maximised).
  - 'R0'        major radius (minimised), i.e. the most compact
                machine satisfying every constraint. R0 must be a
                GA-optimised parameter for this to be meaningful.
  - 'custom'    any arithmetic expression of the design inputs and
                outputs, given by 'fitness_expression' (minimised),
                e.g. R0 + 0.05 * Ip or R0 + 10 * max(0, 20 - Q).

Outputs
-------
Each run creates a timestamped directory under D0FUS_OUTPUTS/genetic/
containing:
  - convergence.png                 best-fitness curve over generations
                                    with valid-design counts.
  - optimization_summary.json       settings, parameters of the best
                                    design, stability margins, full
                                    Hall of Fame, per-call refinement
                                    log, per-generation cumulative-best
                                    trajectory, and per-generation
                                    exploration summary stats.
  - trajectory_<vx>_<vy>.gif        when make_gif is true: a 3D-plus-2D
                                    animated GIF of the cumulative-best
                                    design moving across a chosen
                                    parameter plane (default a, R0)
                                    over the cost surface (in log10 by
                                    default), with the GA evaluations
                                    plotted as a semi-transparent grey
                                    scatter cloud that grows as the GA
                                    progresses.
  - scan_<vx>_<vy>.npz              raw 2D scan and exploration cloud
                                    (gen, x, y, fitness) for re-plotting
                                    without re-running.
  - the standard D0FUS save_run_output of the best design.

Usage (CLI)
-----------
    python D0FUS_genetic.py <input_file> [options]
Example:
    python D0FUS_genetic.py inputs/EU_DEMO.txt \\
        --refine nelder-mead --refine-topk 3 --refine-every 10 \\
        --gif --gif-vars a,R0 --gif-res 30

Input file
----------
One key per line, '#' starts a comment. Scalar values become static
inputs (e.g. 'P_fus = 500'). Square-bracket pairs become GA-optimised
parameters (e.g. 'R0 = [4.0, 9.0]'). All GA, refinement and GIF
hyperparameters (population_size, generations, n_workers,
local_refine_method, local_refine_top_k, gif_var_x, gif_resolution,
...) can be set there too; see run_genetic_optimization for the full
list of recognised keys.

Dependencies
------------
All standard, scientific, plotting and DEAP imports are centralised in
D0FUS_import.py. Project-specific dependencies (D0FUS_BIB physics and
cost modules, D0FUS_EXE.run, D0FUS_EXE.save_run_output) are imported at
the top of this file.

Author
------
Auclair Timothé, CEA-IRFM Cadarache.
Originally created : December 2023.
Last major revision : May 2026 (memetic local refinement, parallel
gradient, exploration cloud, trajectory GIF, centralised imports).
"""

#%% Imports

# Centralised imports — D0FUS_BIB/D0FUS_import.py exports all standard,
# scientific, plotting and DEAP names.
#
# Path resolution strategy:
#   - Normal usage (D0FUS.py is the entry point): D0FUS.py already inserts
#     the project root in sys.path. The first attempt succeeds.
#   - Standalone execution of this module: fall back to inserting the
#     project root using a fully-normalised absolute path. The check
#     ``_project_root not in sys.path`` prevents duplicate entries that
#     would otherwise let the module be discovered under two different
#     qualified names (e.g. ``D0FUS_genetic`` AND ``D0FUS_EXE.D0FUS_genetic``)
#     and break Windows ``multiprocessing`` pickling.
try:
    from D0FUS_BIB.D0FUS_import import *
except ModuleNotFoundError:
    import sys, os
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from D0FUS_BIB.D0FUS_import import *

# ── Supplementary imports specific to this module ────────────────────────────
# The wildcard import above covers most of what D0FUS_genetic needs, but a
# few names used here (parallel pool, dataclass introspection, animation,
# 3D axes registration, traceback in except blocks) are not exported by
# the current D0FUS_BIB/D0FUS_import.py. They are imported explicitly so
# this module is self-contained and does not require any change to
# D0FUS_import.py. If D0FUS_import.py is later extended to cover these,
# the duplicate imports are harmless (Python caches the modules).
# (multiprocessing, traceback, dataclasses.replace/asdict, FuncAnimation,
#  PillowWriter and the Axes3D 3d-projection registration are all provided
#  by the D0FUS_import wildcard above.)

# Project-specific D0FUS imports (kept here because they describe this
# module's direct dependencies inside the D0FUS source tree). DEAP is already
# provided by the centralised D0FUS_import wildcard above, so it is not
# re-imported here. Feasibility (plasma limits and radial-build closure) and
# cost come from the shared RUN evaluators (evaluate_feasibility,
# evaluate_costs), identical in every mode.
from D0FUS_BIB.D0FUS_parameterization import (GlobalConfig, DEFAULT_CONFIG, coerce_input_value,
                                               resolve_deprecated_key, LEGACY_SPLIT_KEYS,
                                               apply_legacy_split_keys)
from D0FUS_BIB.D0FUS_cost_data import *
from D0FUS_EXE.D0FUS_run import (run, save_run_output, resolve_operating_point,
                                  core_plasma_values, evaluate_costs,
                                  evaluate_feasibility, new_output_dir,
                                  default_output_root)

# Backwards-compatible alias: some legacy parts of the file may still use
# the `dc_replace` name (and the docstring references it).
dc_replace = replace

# ── Fitness objective ────────────────────────────────────────────────────────
# Selectable via 'fitness_objective = ...' in the input file.
#
#   'COE'      : minimise cost of electricity (Sheffield 2016)     [EUR/MWh]
#                 with C_invest <= C_invest_max budget constraint   (DEFAULT)
#   'volume'   : minimise (V_blanket + V_TF_one*N_TF + V_CS_geom) / P_fus  [m^3/MW]
#   'C_invest' : minimise total capital cost (Sheffield 2016)      [M EUR]
#   'P_elec'   : maximise net electric power (minimise -P_elec)    [MW]
#   'R0'       : minimise the major radius (most compact machine)  [m]
#                 R0 must be declared as a GA-optimised parameter
#                 (e.g. 'R0 = [4.0, 9.0]') for this objective to be
#                 meaningful; otherwise the fitness is a constant.
#   'custom'   : minimise the expression given by 'fitness_expression',
#                 built from the design inputs and outputs (see
#                 custom_fitness_names). A lower value is better.
#
VALID_FITNESS_OBJECTIVES = ('COE', 'volume', 'C_invest', 'P_elec', 'R0', 'custom')

# =============================================================================
# run() return-tuple index map (indices 0-148 of the 165 outputs)
#
# This dictionary maps symbolic names to positional indices in the tuple
# returned by run(). debug_single_run() and the final report of
# run_genetic_optimization() use it. The feasibility and cost of a design
# come from the shared RUN evaluators (evaluate_feasibility, evaluate_costs).
# =============================================================================
_IDX = {
    # ── Fields and plasma ──────────────────────────────────────────────────
    'B0':          0,
    'B_CS':        1,
    'B_pol':       2,
    'tauE':        3,
    'W_th':        4,   # [J]  — divide by 1e6 for MJ
    'Q':           5,
    'Volume':      6,
    'Surface':     7,
    'Ip':          8,
    'Ib':          9,
    'I_CD':       10,
    'I_Ohm':      11,
    'nbar':       12,   # Volume-averaged density [10²⁰ m⁻³]
    'nbar_line':  13,   # Line-averaged density  [10²⁰ m⁻³]
    'nG':         14,   # Density limit, model-selectable effective cap [10²⁰ m⁻³]
    'pbar':       15,   # Volume-averaged pressure [MPa]
    'betaN':      16,   # Normalized beta [% m T / MA]
    'betaT':      17,   # Toroidal beta (fraction, ×100 for %)
    'betaP':      18,
    'qstar':      19,   # Kink safety factor
    'q95':        20,
    # ── Power balance ─────────────────────────────────────────────────────
    'P_CD':       21,
    'P_sep':      22,
    'P_Thresh':   23,
    'eta_CD':     24,
    'P_elec':     25,
    'P_wallplug': 26,
    'cost':       27,   # Machine volume proxy (V_BB + V_TF + V_CS) / P_fus [m³/MW]
    'P_Brem':     28,
    'P_syn':      29,
    'P_line':     30,
    'P_line_core':31,
    'heat':       32,
    'heat_par':   33,
    'heat_pol':   34,
    'lambda_q':   35,
    'q_target':   36,
    'P_wall_H':   37,
    'P_wall_L':   38,
    'Gamma_n':    39,
    'f_alpha':    40,
    'tau_alpha':  41,
    # ── Coil currents ─────────────────────────────────────────────────────
    'J_TF':       42,
    'J_CS':       43,
    # ── TF radial build ───────────────────────────────────────────────────
    'c':          44,
    'c_WP_TF':    45,
    'c_Nose_TF':  46,
    'sigma_z_TF': 47,
    'sigma_th_TF':48,
    'sigma_r_TF': 49,
    'Steel_frac_TF': 50,
    # ── CS radial build ───────────────────────────────────────────────────
    'd':          51,
    'sigma_z_CS': 52,
    'sigma_th_CS':53,
    'sigma_r_CS': 54,
    'Steel_frac_CS': 55,
    'B_CS_dup':   56,   # B_CS repeated (same value as index 1)
    'J_CS_dup':   57,   # J_CS repeated (same value as index 43)
    # ── Radii ─────────────────────────────────────────────────────────────
    'r_minor':    58,   # R0 − a
    'r_sep':      59,   # R0 − a − b
    'r_c':        60,   # R0 − a − b − c
    'r_d':        61,   # R0 − a − b − c − d  (must be > 0)
    # ── Shape ─────────────────────────────────────────────────────────────
    'kappa':      62,
    'kappa_95':   63,
    'delta':      64,
    'delta_95':   65,
    # ── Magnetic flux ─────────────────────────────────────────────────────
    'PsiPI':      66,
    'PsiRU':      67,
    'PsiPlat':    68,
    'PsiPF':      69,
    'PsiCS':      70,
    # ── New outputs (V_loop, li) ──────────────────────────────────────────
    'Vloop':      71,   # Steady-state loop voltage [V]
    'li':         72,   # Internal inductance li(3) [-]
    # ── Per-source CD efficiencies & powers ───────────────────────────────
    'eta_LH':     73,
    'eta_EC':     74,
    'eta_NBI':    75,
    'P_LH':       76,
    'P_EC':       77,
    'P_NBI':      78,
    'P_ICR':      79,
    'I_LH':       80,
    'I_EC':       81,
    'I_NBI':      82,
    # ── TF cable-level fractions (new in v3) ──────────────────────────────
    'f_sc_TF':    83,
    'f_cu_TF':    84,
    'f_He_pipe_TF': 85,
    'f_void_TF':  86,
    'f_He_TF':    87,
    'f_In_TF':    88,
    # ── CS cable-level fractions (new in v3) ──────────────────────────────
    'f_sc_CS':    89,
    'f_cu_CS':    90,
    'f_He_pipe_CS': 91,
    'f_void_CS':  92,
    'f_He_CS':    93,
    'f_In_CS':    94,
    # ── Fast-alpha outputs ──────────────────────────────────────────────
    'beta_fast_alpha': 95,
    'betaN_total':     96,
    'tau_sd_alpha':    97,
    'W_fast_alpha':    98,
    # ── Coil volumes and cable inventory ───────────────────────────────
    'V_TF_one':       99,    # Single TF coil volume (Pappus centroid) [m³]
    'V_CS_geom':      100,   # Total CS solenoid volume [m³]
    'V_steel_TF':     101,    # TF steel total (all coils) [m³]
    'V_sc_TF':        102,    # TF SC material total [m³]
    'V_cu_TF':        103,    # TF Cu total [m³]
    'V_He_TF':        104,    # TF He total [m³]
    'V_In_TF':        105,    # TF insulation total [m³]
    'V_steel_CS':     106,    # CS steel [m³]
    'V_sc_CS':        107,    # CS SC material [m³]
    'V_cu_CS':        108,    # CS Cu [m³]
    'V_He_CS':        109,    # CS He [m³]
    'V_In_CS':        110,    # CS insulation [m³]
    'L_cable_TF':     111,    # Total TF cable conductor length [m]
    'L_cable_CS':     112,    # Total CS cable conductor length [m]
    'n_sc_TF':        113,    # SC strands/tapes per TF cable [-]
    'n_sc_CS':        114,    # SC strands/tapes per CS cable [-]
    'L_sc_strand_TF': 115,    # Total TF SC strand length [m]
    'L_sc_strand_CS': 116,    # Total CS SC strand length [m]
    # ── Coil masses ─────────────────────────────────────────────────────────
    'M_steel_TF':  117,   # TF steel mass [kg]
    'M_sc_TF':     118,   # TF SC mass [kg]
    'M_cu_TF':     119,   # TF Cu mass [kg]
    'M_In_TF':     120,   # TF insulation mass [kg]
    'M_total_TF':  121,   # TF total mass [kg]
    'M_steel_CS':  122,   # CS steel mass [kg]
    'M_sc_CS':     123,   # CS SC mass [kg]
    'M_cu_CS':     124,   # CS Cu mass [kg]
    'M_In_CS':     125,   # CS insulation mass [kg]
    'M_total_CS':  126,   # CS total mass [kg]
    # ── Blanket volume ───────────────────────────────────────────────────────
    'V_blanket':       127,  # Blanket volume (Miller LCFS → TF) [m³]
    # ── Component lifetimes and availability ─────────────────────────────────
    't_life_bl_fpy':   128,  # [fpy]  # Blanket lifetime [fpy]
    't_life_div_fpy':  129,  # Divertor lifetime [fpy]
    't_life_bl_yr':    130,  # Blanket lifetime [yr]
    't_life_div_yr':   131,  # Divertor lifetime [yr]
    'T_op_limit':      132,  # Operation per replacement cycle [yr]
    'dt_rep_eff':      133,  # Effective replacement downtime [yr]
    'Av':              134,  # Plant availability [-]
    'CF':              135,  # Capacity factor [-]
    # ── Radial build component volumes ───────────────────────────────────────
    'V_rb_SOL':        136,  # SOL / far-SOL volume (all around) [m³]
    'V_rb_FW':         137,  # First wall volume (excl. divertor) [m³]
    'V_rb_BB':         138,  # Breeding blanket volume (excl. divertor) [m³]
    'V_rb_shield':     139,
    'V_rb_VV':         140,
    'V_rb_gap_TF':     141,
    'V_rb_divertor':   142,
    # ── Radial build component masses ────────────────────────────────────────
    'M_rb_FW':         143,  # First wall mass [kg]
    'M_rb_BB':         144,  # Breeding blanket mass [kg]
    'M_rb_shield':     145,  # Neutron shield mass [kg]
    'M_rb_VV':         146,  # Vacuum vessel mass [kg]
    'M_rb_divertor':   147,  # Divertor mass [kg]
    'M_rb_total':      148,  # Total radial build mass [kg]
}

#%% Global Variables (default values)
opt_ranges = {}
static_inputs = {}
param_keys = []
current_generation = 0
n_total_generations = 50
current_run_directory = None

PENALTY_VALUE = 1e6

# ---------------------------------------------------------------------------
# Feasibility-first (Deb) ranking parameters.
#
# With a purely multiplicative penalty a cheap but constraint-violating design
# can outrank an expensive but feasible one (a large enough cost advantage
# overcomes any finite penalty factor). The banded fitness assembled in
# evaluate_individual removes that failure mode: any design whose aggregate
# relative violation does not exceed EPSILON ranks ahead of any design that
# does (feasibility-first, Deb 2000), while still-infeasible designs are
# ordered by their total relative violation (Deb's third rule) so the search
# keeps a smooth gradient toward the feasible boundary. This is the opposite of
# a flat PENALTY_VALUE wall: near-boundary "bridge" designs are preserved.
#
#   EPSILON : aggregate relative violation tolerated inside the acceptable band
#             [-]. Default 0.0 (strict Deb): a design ranked acceptable is
#             FEASIBLE for RUN. A positive value lets the optimum sit up to
#             EPSILON beyond a limit, which RUN then reports as infeasible.
#   FLOOR   : fitness threshold (in objective units) at which the infeasible
#             band starts. It must sit above the largest objective value any
#             feasible design can take and below PENALTY_VALUE. The default,
#             0.1 * PENALTY_VALUE, clears the COE, C_invest, R0 and volume
#             objectives with a wide margin while staying below the sentinel.
#   GAIN    : slope of the violation gradient inside the infeasible band [-].
# ---------------------------------------------------------------------------
DEFAULT_FEASIBILITY_EPSILON   = 0.0                  # tolerated aggregate violation [-]
DEFAULT_INFEASIBLE_FLOOR      = 0.1 * PENALTY_VALUE  # infeasible band threshold (obj. units)
DEFAULT_FEASIBILITY_BAND_GAIN = 1.0                  # in-band violation gradient slope [-]

# ---------------------------------------------------------------------------
# Deck settings that are not GlobalConfig fields: name -> (default, type).
# Precedence: explicit argument of run_genetic_optimization (or CLI option),
# then the deck, then the default below.
#
# GA_SETTINGS are read by the driver. FITNESS_SETTINGS are read by
# evaluate_individual and therefore stay in static_inputs, which travels to
# the parallel workers with the design inputs.
# ---------------------------------------------------------------------------
GA_SETTINGS = {
    'population_size':           (200,   int),
    'generations':               (50,    int),
    'crossover_rate':            (0.7,   float),
    'mutation_rate':             (0.3,   float),
    'patience':                  (20,    int),    # generations without improvement before stopping
    'seed':                      (None,  int),    # None: drawn at random, then recorded
    'n_workers':                 (1,     int),
    'local_refine_method':       (None,  str),    # 'gradient', 'lbfgsb', 'nelder-mead' or none
    'local_refine_top_k':        (3,     int),
    'local_refine_every':        (0,     int),
    'diversity_inject_fraction': (0.30,  float),
    'cloud_refinement_min_dist': (0.05,  float),
    'make_gif':                  (False, bool),
    'gif_var_x':                 ('a',   str),
    'gif_var_y':                 ('R0',  str),
    'gif_resolution':            (30,    int),
    'gif_log_scale':             (True,  bool),
    'gif_fps':                   (8,     int),
}
FITNESS_SETTINGS = {
    'fitness_objective':         ('COE', str),
    'fitness_expression':        (None,  str),    # used when fitness_objective = custom
    'feasibility_epsilon':       (DEFAULT_FEASIBILITY_EPSILON,   float),
    'infeasible_floor':          (DEFAULT_INFEASIBLE_FLOOR,      float),
    'feasibility_band_gain':     (DEFAULT_FEASIBILITY_BAND_GAIN, float),
    'density_violation_weight':  (1.0,   float),
    'beta_violation_weight':     (1.0,   float),
    'q_kink_violation_weight':   (1.0,   float),
    'Ip_violation_weight':       (1.0,   float),
    'Ip_soft_start':             (None,  float),
    'Ip_soft_penalty':           (None,  float),
}
# Local optimiser knobs, written local_refine_<name> in the deck
REFINE_KNOBS = {
    'max_iters': int, 'max_backtrack': int,
    'h': float, 'step_init': float, 'c1': float, 'tol_grad': float,
    'tol_step': float, 'ftol': float, 'gtol': float, 'eps': float,
    'xatol': float, 'fatol': float, 'initial_simplex_scale': float,
}
VALID_REFINE_METHODS = ('gradient', 'lbfgsb', 'nelder-mead')


def _parse_setting(name, value, kind):
    """
    Convert a deck or argument value to the type of a GA setting.

    Integers written as floats (1e3, 200.0) are accepted when they are
    whole numbers. 'none' maps to None. A malformed value raises a
    ValueError naming the setting, at parsing time rather than after the GA.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ('none', 'off', ''):
        return None
    try:
        if kind is bool:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            low = str(value).strip().lower()
            if low in ('1', 'true', 'yes', 'on'):
                return True
            if low in ('0', 'false', 'no', 'off'):
                return False
            raise ValueError
        if kind is int:
            v = float(value)
            if not v.is_integer():
                raise ValueError
            return int(v)
        if kind is float:
            return float(value)
        return str(value).strip()
    except (TypeError, ValueError):
        raise ValueError(f"GA setting '{name}': cannot read {value!r} "
                         f"as {kind.__name__}.") from None


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


def _summarize_cloud_per_gen(ga_cloud):
    """
    Aggregate the GA exploration cloud into per-generation summary stats.

    The full cloud (one entry per evaluation) can hold ~10^4 points after
    a typical run; serialising it inline in JSON makes the file unwieldy.
    The raw cloud is stored separately in the .npz file alongside the
    scan; here we keep only counts and fitness quantiles per generation.

    Returns
    -------
    list of dicts, one per generation present in the cloud, with keys:
        gen, n_evals, f_min, f_median, f_max
    """
    if not ga_cloud:
        return []
    # Group by generation
    by_gen = {}
    for entry in ga_cloud:
        g = entry.get('gen', 0)
        by_gen.setdefault(g, []).append(entry['fitness'])
    out = []
    for g in sorted(by_gen.keys()):
        fs = np.asarray(by_gen[g], dtype=float)
        out.append({
            'gen':      int(g),
            'n_evals':  int(len(fs)),
            'f_min':    float(np.min(fs)),
            'f_median': float(np.median(fs)),
            'f_max':    float(np.max(fs)),
        })
    return out

#%% Constraint Handling

def _opt_float(value):
    """Coerce an optional numeric tuning input to ``float``, preserving ``None``.

    Input-file values for keys that are not ``GlobalConfig`` fields can reach the
    optimiser as plain strings (the parser only type-coerces known dataclass
    fields). This maps a *set* value to ``float`` and leaves an *unset* value
    (``None``) untouched, so the caller's default is used. Any value that cannot
    be parsed as a real number degrades gracefully to ``None``.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_Ip_soft_multiplier(Ip, Ip_soft_start, Ip_limit, Ip_soft_penalty):
    """
    Soft fitness surcharge for a plasma current in the undesirable zone.

    Between Ip_soft_start and Ip_limit the current is admissible but not
    desirable (larger disruption loads, larger runaway-electron avalanche
    gain). The fitness is multiplied by

        m(Ip) = 1 + Ip_soft_penalty * x,
        x     = (Ip - Ip_soft_start) / (Ip_limit - Ip_soft_start),

    with x clipped to [0, 1]. The surcharge is linear, so its marginal price
    is finite at Ip_soft_start: the search enters the zone only where the
    relative objective gain per MA exceeds
    Ip_soft_penalty / (Ip_limit - Ip_soft_start). Above Ip_limit the hard
    constraint (feasibility band) takes over and the multiplier stays at its
    maximum. The multiplier shapes the ranking only: the reported C_invest,
    COE and P_elec of a design are never modified.

    Parameters
    ----------
    Ip              : float  Plasma current [MA].
    Ip_soft_start   : float  Lower edge of the undesirable zone [MA], or None.
    Ip_limit        : float  Hard current ceiling [MA], or None.
    Ip_soft_penalty : float  Relative surcharge reached at Ip_limit [-]
                             (0.10 = +10 %). None or <= 0 disables it.

    Returns
    -------
    float  Multiplier >= 1 (exactly 1.0 when the soft zone is not configured).
    """
    vals = (Ip, Ip_soft_start, Ip_limit, Ip_soft_penalty)
    if any(v is None for v in vals):
        return 1.0
    try:
        Ip, start, lim, k = (float(v) for v in vals)
    except (TypeError, ValueError):
        return 1.0
    if (not all(np.isfinite(v) for v in (Ip, start, lim, k))
            or k <= 0.0 or lim <= start):
        return 1.0
    x = min(max((Ip - start) / (lim - start), 0.0), 1.0)
    return 1.0 + k * x


#%% Fitness Evaluation

def _safe_real(value):
    """
    Real part of a scalar as a float, NaN when it is not finite.

    Same convention as RUN (evaluate_feasibility, evaluate_costs), which
    read float(np.real(x)), so that the GA and RUN judge a design on the
    same numbers.
    """
    try:
        v = float(np.real(value))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# Feasibility ratio of evaluate_feasibility -> deck key of its weight
_VIOLATION_WEIGHT_KEYS = {
    'density': 'density_violation_weight',
    'beta':    'beta_violation_weight',
    'q':       'q_kink_violation_weight',
    'Ip':      'Ip_violation_weight',
}


def _feasibility_violation(ratios, weights=None):
    """
    Total relative violation of the RUN feasibility ratios.

    Each ratio of evaluate_feasibility must stay <= 1 (density, beta, q
    written q_limit / q_kink, and Ip when Ip_limit is set). A ratio r
    contributes w * max(0, r - 1). The sum is exactly 0 for a design that
    RUN reports FEASIBLE (radial build apart) and grows with the violation.

    Parameters
    ----------
    ratios  : dict  evaluate_feasibility(...)['ratios'].
    weights : dict or None  Weight per ratio name (default 1.0). A larger
              weight makes the search cure that violation first.

    Returns
    -------
    float  Total weighted violation >= 0, or NaN if a ratio is not finite.
    """
    total = 0.0
    for name, r in ratios.items():
        r = _safe_real(r)
        if not np.isfinite(r):
            return np.nan
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        total += w * max(0.0, r - 1.0)
    return total


def compute_feasibility_fitness(raw_fitness, total_violation, objective,
                                budget_multiplier=1.0,
                                epsilon=DEFAULT_FEASIBILITY_EPSILON,
                                infeasible_floor=DEFAULT_INFEASIBLE_FLOOR,
                                band_gain=DEFAULT_FEASIBILITY_BAND_GAIN):
    """
    Map (objective value, aggregate violation) to a single MINIMISED fitness
    enforcing feasibility-first ranking (Deb 2000).

    The DEAP fitness is minimised (weights = (-1.0,)), so 'smaller is better'.
    Two bands are produced.

    Acceptable band (total_violation <= epsilon)
        The raw objective, penalised by the soft budget multiplier m >= 1:
        ``fitness = raw + |raw| (m - 1)``. This equals raw * m for a positive
        objective and moves a negative one (-P_elec) towards zero, so a
        budget overshoot always worsens the fitness. With epsilon = 0 (the
        default) this band holds exactly the designs that RUN reports
        feasible. A clamp keeps the value strictly below
        ``infeasible_floor`` so an acceptable design can never enter the
        infeasible band, even if ``infeasible_floor`` were mis-set below the
        objective scale.

    Infeasible band (total_violation > epsilon)
        ``fitness = infeasible_floor * (1 + band_gain * (total_violation - epsilon))``.
        Every value is >= ``infeasible_floor`` and therefore strictly worse
        than any acceptable design. Within the band the value increases
        monotonically with the violation (Deb's third rule), so the search
        still separates a 2 % overshoot from a 40 % one and keeps climbing
        toward the boundary; this is a slope, not a wall. The value is capped
        just below PENALTY_VALUE so structurally invalid designs (returned as
        PENALTY_VALUE elsewhere) remain the worst of all.

    Notes
    -----
    Exact band ordering requires ``infeasible_floor`` to exceed the largest
    objective value any feasible design can take. The default (1e5) clears the
    COE, C_invest, R0 and volume objectives with margin and sits below
    PENALTY_VALUE (1e6); the clamp keeps the ordering correct even outside that
    assumption. Defensive guards also coerce epsilon and band_gain to be
    non-negative and fall back to the default floor if a non-finite or
    non-positive floor is supplied.
    """
    epsilon = max(0.0, float(epsilon))
    band_gain = max(0.0, float(band_gain))
    infeasible_floor = float(infeasible_floor)
    if not (np.isfinite(infeasible_floor) and infeasible_floor > 0.0):
        infeasible_floor = DEFAULT_INFEASIBLE_FLOOR

    if total_violation <= epsilon:
        fitness = raw_fitness + abs(raw_fitness) * (budget_multiplier - 1.0)
        # An acceptable design must never reach the infeasible band.
        return min(fitness, infeasible_floor * (1.0 - 1e-9))

    # Infeasible band: strictly worse, ordered by violation, bounded below the
    # hard PENALTY_VALUE sentinel so structurally invalid designs stay worst.
    overshoot = total_violation - epsilon
    fitness = infeasible_floor * (1.0 + band_gain * overshoot)
    return min(fitness, PENALTY_VALUE * (1.0 - 1e-9))


# ── Custom fitness expression ────────────────────────────────────────────────
# fitness_objective = custom minimises an arithmetic expression written in the
# deck, for instance
#     fitness_expression = R0 + 0.05 * Ip + 10 * max(0, 20 - Q)
# The expression may use numbers, the names of custom_fitness_names() (numeric
# GlobalConfig inputs, run() outputs of _IDX and a few derived quantities),
# the operators + - * / ** %, parentheses and the functions of _EXPR_FUNCS.
# Square brackets and '#' cannot appear in a deck value (range and comment
# markers). The expression is parsed with the ast module and evaluated on a
# closed set of nodes: no Python code is executed.
import ast as _ast
import operator as _operator

_EXPR_FUNCS = {'min': min, 'max': max, 'abs': abs, 'sqrt': np.sqrt,
               'log': np.log, 'log10': np.log10, 'exp': np.exp}
_EXPR_BINOPS = {_ast.Add: _operator.add, _ast.Sub: _operator.sub,
                _ast.Mult: _operator.mul, _ast.Div: _operator.truediv,
                _ast.Pow: _operator.pow, _ast.Mod: _operator.mod}
_EXPR_UNARYOPS = {_ast.UAdd: _operator.pos, _ast.USub: _operator.neg}
# Derived names: A = R0 / a, f_GW = nbar_line / (Ip / (pi a^2)),
# P_aux = P_aux_input [MW], C_invest [M EUR] and COE [EUR/MWh] (Sheffield).
_EXPR_DERIVED = ('A', 'f_GW', 'P_aux', 'C_invest', 'COE')
_EXPR_COST_NAMES = ('C_invest', 'COE')
_EXPR_CACHE = {}        # expression string -> (parsed tree, names used)


def custom_fitness_names():
    """Sorted names usable in fitness_expression."""
    inputs = [k for k in DEFAULT_CONFIG.__dataclass_fields__
              if isinstance(getattr(DEFAULT_CONFIG, k), (int, float))
              and not isinstance(getattr(DEFAULT_CONFIG, k), bool)]
    return sorted(set(inputs) | set(_IDX) | set(_EXPR_DERIVED))


def parse_fitness_expression(expr):
    """
    Parse and check a fitness expression once (cached).

    Returns
    -------
    tuple  (tree, names): the ast.Expression and the frozenset of the
           names it reads. A ValueError names the first unsupported
           element or unknown name.
    """
    if expr in _EXPR_CACHE:
        return _EXPR_CACHE[expr]
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("fitness_objective = custom needs a non-empty "
                         "fitness_expression in the deck.")
    try:
        tree = _ast.parse(expr.strip(), mode='eval')
    except SyntaxError as exc:
        raise ValueError(f"fitness_expression {expr!r}: syntax error ({exc.msg}).") from None
    allowed = set(custom_fitness_names())
    names = set()
    for node in _ast.walk(tree):
        if (isinstance(node, (_ast.Expression, _ast.Load, _ast.BinOp, _ast.UnaryOp))
                or type(node) in _EXPR_BINOPS or type(node) in _EXPR_UNARYOPS):
            continue
        if isinstance(node, _ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                continue
            raise ValueError(f"fitness_expression {expr!r}: constant {node.value!r} "
                             "is not a number.")
        if isinstance(node, _ast.Call):
            if (not isinstance(node.func, _ast.Name) or node.func.id not in _EXPR_FUNCS
                    or node.keywords or not node.args):
                raise ValueError(f"fitness_expression {expr!r}: only the functions "
                                 f"{', '.join(_EXPR_FUNCS)} are allowed, with positional "
                                 "arguments.")
            continue
        if isinstance(node, _ast.Name):
            if node.id in _EXPR_FUNCS:
                continue
            if node.id not in allowed:
                import difflib
                close = difflib.get_close_matches(node.id, allowed, n=3, cutoff=0.6)
                raise ValueError(f"fitness_expression {expr!r}: unknown name "
                                 f"{node.id!r}. Closest names: {close}. Full list: "
                                 "custom_fitness_names().")
            names.add(node.id)
            continue
        raise ValueError(f"fitness_expression {expr!r}: element "
                         f"{type(node).__name__} is not allowed.")
    _EXPR_CACHE[expr] = (tree, frozenset(names))
    return _EXPR_CACHE[expr]


def _eval_expr_node(node, ns):
    """Recursive evaluation of a node checked by parse_fitness_expression."""
    if isinstance(node, _ast.Expression):
        return _eval_expr_node(node.body, ns)
    if isinstance(node, _ast.Constant):
        return float(node.value)
    if isinstance(node, _ast.Name):
        return ns[node.id]
    if isinstance(node, _ast.BinOp):
        return _EXPR_BINOPS[type(node.op)](_eval_expr_node(node.left, ns),
                                           _eval_expr_node(node.right, ns))
    if isinstance(node, _ast.UnaryOp):
        return _EXPR_UNARYOPS[type(node.op)](_eval_expr_node(node.operand, ns))
    if isinstance(node, _ast.Call):
        return _EXPR_FUNCS[node.func.id](*[_eval_expr_node(a, ns) for a in node.args])
    raise ValueError(f"unsupported node {type(node).__name__}")


def custom_fitness_namespace(config, output, costs=None):
    """
    Values of the names usable in fitness_expression for one design.

    Numeric GlobalConfig inputs, then the run() outputs of _IDX (an output
    overrides an input of the same name), then the derived quantities.
    C_invest and COE are NaN when costs is None or empty.
    """
    ns = {}
    for k in config.__dataclass_fields__:
        v = getattr(config, k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ns[k] = float(v)
    ns.update({k: _safe_real(output[i]) for k, i in _IDX.items() if i < len(output)})
    ns['A'] = config.R0 / config.a
    nGW = ns['Ip'] / (np.pi * config.a ** 2)
    ns['f_GW'] = ns['nbar_line'] / nGW if nGW > 0 else np.nan
    ns['P_aux'] = float(config.P_aux_input)
    costs = costs or {}
    ns['C_invest'] = _safe_real(costs.get('C_invest', np.nan))
    ns['COE'] = _safe_real(costs.get('COE', np.nan))
    return ns


def evaluate_fitness_expression(expr, config, output, costs=None):
    """Value of fitness_expression for one design (NaN on a math error)."""
    tree, _ = parse_fitness_expression(expr)
    ns = custom_fitness_namespace(config, output, costs)
    try:
        with np.errstate(all='ignore'):
            return _safe_real(_eval_expr_node(tree, ns))
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return np.nan


def evaluate_individual(individual, verbose=False):
    """
    Evaluate the fitness of a single individual.

    The design is run with run(), then judged with the shared RUN
    evaluators: evaluate_feasibility for the constraints and
    evaluate_costs for the Sheffield cost. The objective is read from
    static_inputs['fitness_objective'] (see VALID_FITNESS_OBJECTIVES).

    Returns
    -------
    tuple  (fitness,), minimised. PENALTY_VALUE when run() fails, the
           radial build does not close, a constraint ratio or the objective
           is not finite.
    """
    try:
        param_dict = {k: individual[i] for i, k in enumerate(param_keys)}
        all_params = {**static_inputs, **param_dict}

        # Build an immutable GlobalConfig, filtering out non-field keys.
        config = GlobalConfig(**{k: v for k, v in all_params.items()
                                  if k in GlobalConfig.__dataclass_fields__})

        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            output = run(config, verbose=0)

        # ── Feasibility: same rule as the RUN verdict ────────────────────
        fz = evaluate_feasibility(config, output)
        if not fz['build_ok']:
            if verbose:
                print(f"  [PENALTY] radial build: {fz['build_reason']}")
            return (PENALTY_VALUE,)
        weights = {name: _opt_float(static_inputs.get(key)) for name, key
                   in _VIOLATION_WEIGHT_KEYS.items()}
        weights = {k: (1.0 if v is None else v) for k, v in weights.items()}
        total_violation = _feasibility_violation(fz['ratios'], weights)
        if not np.isfinite(total_violation):
            if verbose:
                print(f"  [PENALTY] non-finite constraint ratio: {fz['ratios']}")
            return (PENALTY_VALUE,)

        # ── Objective ────────────────────────────────────────────────────
        objective = static_inputs.get('fitness_objective', 'COE')
        _expr = static_inputs.get('fitness_expression')
        _expr_cost = (objective == 'custom' and bool(
            parse_fitness_expression(_expr)[1] & set(_EXPR_COST_NAMES)))
        # Sheffield cost (shared RUN evaluator, exact RUN volumes). Needed by
        # the cost objectives, by the C_invest_max budget of 'P_elec' and by
        # a custom expression that reads C_invest or COE.
        _cres = (evaluate_costs(config, output)
                 if objective in ('COE', 'C_invest', 'P_elec') or _expr_cost else {})
        _C_inv = _safe_real(_cres.get('C_invest', np.nan))

        if objective == 'volume':
            # (V_BB + V_TF + V_CS) / P_fus [m^3/MW]
            raw_fitness = _safe_real(output[_IDX['cost']])
        elif objective == 'R0':
            # Most compact machine satisfying every constraint [m]
            raw_fitness = _safe_real(config.R0)
        elif objective == 'COE':
            raw_fitness = _safe_real(_cres.get('COE', np.nan))
        elif objective == 'C_invest':
            raw_fitness = _C_inv
        elif objective == 'custom':
            # User expression, minimised. Any finite value, sign included.
            raw_fitness = evaluate_fitness_expression(_expr, config, output, _cres)
        else:  # 'P_elec', maximised
            raw_fitness = -_safe_real(output[_IDX['P_elec']])

        if not np.isfinite(raw_fitness) or (raw_fitness <= 0
                                            and objective not in ('P_elec', 'custom')):
            if verbose:
                print(f"  [PENALTY] objective '{objective}' not finite or <= 0")
            return (PENALTY_VALUE,)

        # ── Capital cost ceiling: soft quadratic penalty ─────────────────
        # penalty_budget = 1 + k * (excess)^2      k = 3.0
        #   excess = 0.04 (26 vs 25 B) → ×1.005   (barely noticeable)
        #   excess = 0.5                → ×1.75    (moderate pressure)
        #   excess = 1.0               → ×4.0     (strong, but survives)
        # Skipped when C_invest is not available ('volume', 'R0') or when
        # the user directly minimises C_invest (redundant).
        budget_multiplier = 1.0
        C_max = config.C_invest_max
        if (objective != 'C_invest'
                and np.isfinite(_C_inv) and np.isfinite(C_max)
                and _C_inv > C_max):
            excess = (_C_inv - C_max) / C_max
            budget_multiplier = 1.0 + 3.0 * excess ** 2

        # ── Plasma-current comfort zone: soft linear surcharge ───────────
        # Admissible but undesirable currents (Ip_soft_start < Ip <= Ip_limit)
        # pay a surcharge rising linearly to Ip_soft_penalty at Ip_limit. It
        # rides on the budget multiplier, so it acts inside the acceptable
        # band only and leaves the feasibility-first ranking untouched.
        budget_multiplier *= compute_Ip_soft_multiplier(
            _safe_real(output[_IDX['Ip']]),
            _opt_float(static_inputs.get('Ip_soft_start')),
            _opt_float(config.Ip_limit),
            _opt_float(static_inputs.get('Ip_soft_penalty')))

        # ── Feasibility-first (Deb) fitness ──────────────────────────────
        _eps   = _opt_float(static_inputs.get('feasibility_epsilon'))
        _floor = _opt_float(static_inputs.get('infeasible_floor'))
        _gain  = _opt_float(static_inputs.get('feasibility_band_gain'))
        fitness = compute_feasibility_fitness(
            raw_fitness, total_violation, objective,
            budget_multiplier=budget_multiplier,
            epsilon=(DEFAULT_FEASIBILITY_EPSILON if _eps is None else _eps),
            infeasible_floor=(DEFAULT_INFEASIBLE_FLOOR if _floor is None else _floor),
            band_gain=(DEFAULT_FEASIBILITY_BAND_GAIN if _gain is None else _gain),
        )
        return (fitness,)

    except Exception as _exc:
        # Unphysical parameter combinations routinely fail (NaN in Miller
        # geometry, negative radii, etc.) — this is expected, not alarming.
        # Only print a one-line diagnostic when explicitly requested.
        if verbose:
            print(f"  [PENALTY] {type(_exc).__name__}: {_exc}")
        return (PENALTY_VALUE,)

#%% I/O Management

def initialize_run_directory(base_directory=None):
    """
    Create the run directory base_directory/Genetic_D0FUS_<time>.
    Default base_directory: the genetic/ folder of default_output_root().
    """
    global current_run_directory
    if base_directory is None:
        base_directory = os.path.join(default_output_root(), "genetic")
    current_run_directory, _ = new_output_dir(base_directory, "Genetic_D0FUS")
    print(f"\n Output directory: {current_run_directory}")
    return current_run_directory

#%% Input File Parsing

def load_input_file(input_file):
    """
    Load and parse a GENETIC input file.

    Scalar lines become static inputs (GlobalConfig fields and GA
    settings). Lines with a bracket pair, written 'key = [lo, hi]' or
    'key : [lo, hi]', become GA-optimised parameters. Each optimised key
    must be a numeric GlobalConfig field (deprecated aliases are resolved)
    with lo < hi, otherwise a ValueError is raised before any evaluation.
    """
    global opt_ranges, static_inputs, param_keys
    
    opt_ranges = {}
    static_inputs = {}

    # Use asdict to iterate over DEFAULT_CONFIG fields cleanly
    default_dict = asdict(DEFAULT_CONFIG)

    # Keys consumed by the GA driver itself (not GlobalConfig fields). Used to
    # suppress false warnings when checking deck keys against GlobalConfig.
    _GA_SETTING_KEYS = (set(GA_SETTINGS) | set(FITNESS_SETTINGS)
                        | {f'local_refine_{k}' for k in REFINE_KNOBS}
                        | {'local_refine_kwargs'})
    
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
                key = resolve_deprecated_key(key)
                values_str = value[value.index('[') + 1:value.index(']')].strip()
                values = [v.strip() for v in values_str.replace(';', ',').split(',')
                          if v.strip()]
                where = f"{input_file}, line {line_num}"
                if len(values) != 2:
                    raise ValueError(f"{where}: '{key}' needs a range [min, max] "
                                     f"for the GA, got [{values_str}].")
                if key not in default_dict:
                    import difflib
                    _close = difflib.get_close_matches(key, default_dict.keys(),
                                                       n=3, cutoff=0.6)
                    raise ValueError(f"{where}: '{key}' is not a GlobalConfig "
                                     f"field and cannot be optimised. "
                                     f"Closest fields: {_close}")
                if isinstance(default_dict[key], (bool, str)):
                    raise ValueError(f"{where}: '{key}' is not numeric and "
                                     "cannot be optimised.")
                try:
                    min_val, max_val = float(values[0]), float(values[1])
                except ValueError:
                    raise ValueError(f"{where}: '{key}' range [{values_str}] "
                                     "is not numeric.") from None
                if not min_val < max_val:
                    raise ValueError(f"{where}: '{key}' range needs min < max, "
                                     f"got [{min_val}, {max_val}].")
                opt_ranges[key] = (min_val, max_val)
                # Remove from static_inputs if a scalar line was
                # parsed earlier for the same key — prevents the GA
                # value from being silently overwritten at merge time.
                static_inputs.pop(key, None)
                print(f"  {key} = [{min_val}, {max_val}] (optimize)")
            else:
                key = resolve_deprecated_key(key)
                # Skip scalar lines for keys already declared as GA ranges
                # (e.g. a user might have 'a = 3.0' before 'a = [2, 4]').
                if key in opt_ranges:
                    continue
                # Warn on a key that is neither a GlobalConfig field nor a GA
                # setting: it would otherwise be stored inertly in static_inputs
                # (GlobalConfig construction filters it out), so a renamed or
                # mistyped physics field is dropped without notice and silently
                # falls back to its default. This is the single most common
                # cause of an apparently healthy deck producing all-infeasible
                # designs (e.g. a dropped TF field collapses the radial build).
                if (key not in default_dict and key not in _GA_SETTING_KEYS
                        and key not in LEGACY_SPLIT_KEYS):
                    import difflib
                    _close = difflib.get_close_matches(key, default_dict.keys(),
                                                       n=3, cutoff=0.6)
                    print(f"  [warn] '{key}' is not a GlobalConfig field nor a "
                          f"GA setting -> ignored. Closest fields: {_close}")
                # Type-aware coercion shared with the run / scan loaders, so
                # a boolean such as "False" is not silently kept as the
                # truthy string "False", and optional fields decode "None"
                # to the Python None.
                static_inputs[key] = coerce_input_value(key, value)

    # Old single keys split per coil system (Chosen_Steel, σ_manual)
    static_inputs = apply_legacy_split_keys(static_inputs)

    # A Greenwald closure solves its own unknown: a gene on it would be
    # overwritten by the solve (Tbar) or reduced to an initial guess (P_fus).
    for gene, mode_key in (('Tbar', 'Tbar_mode'), ('P_fus', 'P_fus_mode')):
        mode = str(static_inputs.get(mode_key, getattr(DEFAULT_CONFIG, mode_key)))
        if gene in opt_ranges and mode.lower() == 'greenwald':
            raise ValueError(f"'{gene}' cannot be optimised with {mode_key} = "
                             f"greenwald: the closure solves {gene}. Set "
                             f"{mode_key} = manual or remove the range.")

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

def _lhs_individuals(n, opt_ranges, param_keys):
    """
    n individuals drawn by Latin hypercube sampling over the GA box.

    Each parameter range is cut into n equal strata and every stratum holds
    exactly one individual, which covers the box more evenly than n i.i.d.
    uniform draws. The sampler seed is drawn from the random module, so the
    population is reproducible under the run seed.
    """
    sampler = qmc.LatinHypercube(d=len(param_keys), seed=random.getrandbits(32))
    u = sampler.random(n)
    lo = np.array([opt_ranges[k][0] for k in param_keys])
    hi = np.array([opt_ranges[k][1] for k in param_keys])
    return [creator.Individual([float(v) for v in lo + ui * (hi - lo)])
            for ui in u]


def setup_toolbox(opt_ranges, param_keys, n_generations):
    """Setup DEAP toolbox with exploration-focused operators"""
    global toolbox, n_total_generations
    n_total_generations = n_generations
    
    # Single random individual (uniform in the box)
    def create_individual():
        return creator.Individual([random.uniform(*opt_ranges[k])
                                   for k in param_keys])
    
    toolbox.register("individual", create_individual)
    # Initial population: Latin hypercube over the box
    toolbox.register("population", _lhs_individuals,
                     opt_ranges=opt_ranges, param_keys=param_keys)
    toolbox.register("evaluate", evaluate_individual)
    
    # Crossover: blend with wider alpha for more exploration
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[opt_ranges[k][0] for k in param_keys],
                     up=[opt_ranges[k][1] for k in param_keys],
                     eta=5.0)  # Lower eta = more spread
    
    # Enhanced mutation for better exploration
    def exploratory_mutation(individual, indpb=0.4):
        """
        Mutation with exploration/exploitation balance.
        Early: large jumps to explore
        Late: small refinements
        """
        progress = current_generation / max(n_total_generations, 1)
        
        # Exploration probability decreases over time but keeps a floor
        explore_prob = max(0.10, 0.3 * (1 - progress))
        
        for i, key in enumerate(param_keys):
            if random.random() < indpb:
                lo, hi = opt_ranges[key]
                
                if random.random() < explore_prob:
                    # EXPLORATION: Jump to random location in range
                    individual[i] = lo + random.random() * (hi - lo)
                else:
                    # EXPLOITATION: bounded polynomial mutation (Deb), with
                    # an eta that grows from 5 (wide) to 30 (narrow). The
                    # bounded form keeps the gene inside [lo, hi] without
                    # piling mutants onto the bounds.
                    eta = 5 + 25 * progress
                    gene = [individual[i]]
                    tools.mutPolynomialBounded(gene, eta=eta, low=lo, up=hi,
                                               indpb=1.0)
                    individual[i] = gene[0]
        
        return (individual,)
    
    toolbox.register("mutate", exploratory_mutation)
    toolbox.register("select", tools.selTournament, tournsize=2)


def inject_diverse_individuals(pop, n_inject, opt_ranges, param_keys):
    """Inject new individuals to maintain diversity (UNCONDITIONAL).

    n_inject newcomers are drawn by Latin hypercube sampling, evaluated
    together through toolbox.map (in parallel when n_workers > 1), and
    replace the n_inject worst members of the population, whatever their
    own fitness. This forces the GA to re-explore even once the population
    has converged to one basin.

    Returns
    -------
    pop : list
        Same object as input, modified in place.
    injected : list of creator.Individual
        The newly evaluated individuals, each with its `fitness.values`
        already populated. The caller may use this list to log the
        evaluation events (e.g. for the GA exploration cloud).
    """
    n_inject = min(int(n_inject), len(pop))
    if n_inject < 1:
        return pop, []
    injected = _lhs_individuals(n_inject, opt_ranges, param_keys)
    for ind, fit in zip(injected, toolbox.map(toolbox.evaluate, injected)):
        ind.fitness.values = fit

    # Worst members first (an invalid fitness counts as the worst), ranked
    # once so that each newcomer takes a different slot.
    def _badness(i):
        f = pop[i].fitness
        return f.values[0] if f.valid and len(f.values) > 0 else np.inf
    worst = sorted(range(len(pop)), key=_badness, reverse=True)[:n_inject]
    for slot, new_ind in zip(worst, injected):
        pop[slot] = new_ind
    
    return pop, injected


def _pool_initializer(static_inputs_, param_keys_, opt_ranges_):
    """
    Re-populate module-level globals in each worker process.

    Required on Windows/macOS (spawn start method) where worker processes
    start fresh and do not inherit the parent's global state.
    On Linux (fork), this is a no-op since globals are already inherited.

    Note: kept for backwards compatibility with legacy code paths. The
    joblib-based parallel map below (`_evaluate_for_joblib`) does NOT need
    this initializer because each job carries its own context.
    """
    global static_inputs, param_keys, opt_ranges
    static_inputs = static_inputs_
    param_keys    = param_keys_
    opt_ranges    = opt_ranges_


def _evaluate_for_joblib(args):
    """
    Worker for parallel GA fitness evaluation, mirroring the pattern in
    D0FUS_scan._run_scan_point.

    All dependencies are imported locally to prevent cloudpickle from
    serializing this module's global namespace, which would otherwise be
    polluted with __main__-bound symbols when D0FUS.py is launched via
    %runfile in Spyder. Each individual carries its own context
    (static_inputs, param_keys, opt_ranges) as plain Python data, so the
    worker does NOT rely on module globals being pre-populated by an
    initializer — that is what made multiprocessing.Pool fail across
    successive Spyder runs.

    Parameters
    ----------
    args : tuple
        (individual_values, static_inputs_dict, param_keys_list, opt_ranges_dict)

    Returns
    -------
    tuple of float
        Fitness tuple as returned by evaluate_individual.
    """
    # Bootstrap imports only (see module header); aliased to match the
    # worker idiom of D0FUS_scan._run_scan_point.
    import os as _os
    import sys as _sys

    # Make the D0FUS package importable in the worker process. Identical
    # idiom to D0FUS_scan._run_scan_point.
    _parent = _os.path.normpath(
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)

    ind_values, static_inputs_dict, param_keys_list, opt_ranges_dict = args

    # Set the genuine D0FUS_genetic module's globals before delegating to
    # evaluate_individual. Done per call (rather than once per worker via
    # an initializer) so the worker stays stateless — the call is
    # idempotent and the cost is negligible (a few dict assignments).
    from D0FUS_EXE import D0FUS_genetic as _gen_mod
    _gen_mod.static_inputs = static_inputs_dict
    _gen_mod.param_keys    = param_keys_list
    _gen_mod.opt_ranges    = opt_ranges_dict

    return _gen_mod.evaluate_individual(ind_values)


def run_genetic_algorithm(pop_size, n_generations, cxpb, mutpb,
                          patience=20, verbose=True, n_workers=1,
                          local_refine_method=None,
                          local_refine_top_k=3,
                          local_refine_every=0,
                          local_refine_kwargs=None,
                          diversity_inject_fraction=0.30,
                          cloud_refinement_min_dist=0.05):
    """
    Run GA with enhanced exploration, diversity maintenance, and optional
    memetic local refinement.

    Parameters
    ----------
    pop_size, n_generations, cxpb, mutpb, patience, verbose : see caller.
    patience : int
        Early stop after this many generations without improvement of the
        best fitness found so far.
    n_workers : int
        Number of parallel worker processes for fitness evaluation.
        1 (default) -> sequential map.
        N > 1       -> joblib/loky with N workers, used by toolbox.map for
                       the initial population, the offspring and the
                       diversity injections.
    local_refine_method : {None, 'gradient', 'lbfgsb', 'nelder-mead'}
        If not None, apply local refinement (see refine_individual) to the
        top-k Hall-of-Fame members at the end of the GA. 'gradient' is the
        hand-rolled projected gradient descent, 'lbfgsb' uses scipy's
        L-BFGS-B and 'nelder-mead' the scipy simplex. None disables local
        refinement.
    local_refine_top_k : int
        Number of top HoF members to refine at the end of the GA.
    local_refine_every : int
        If > 0, additionally refine the current best HoF member every N
        generations (memetic mode). 0 disables periodic refinement.
    local_refine_kwargs : dict or None
        Extra kwargs forwarded to the local optimiser
        (e.g. max_iters, h, step_init, ftol, gtol, eps).
    """
    global current_generation

    # ── Optional parallel map via joblib/loky (cloudpickle-based) ─────────────
    # Identical pattern to D0FUS_scan: cloudpickle is robust to module-reload
    # and sys.path pollution under Spyder's %runfile, which the standard
    # multiprocessing.Pool + initializer pattern is not (PicklingError on
    # successive runs, see commit history). joblib is already a dependency
    # of D0FUS_scan, so no new package is introduced here.
    _pool = None  # kept as `None` — refine_topk paths that previously took
                  # a Pool now run gradient/L-BFGS-B sequentially. Nelder-Mead
                  # (the default refinement method) does not use a pool, so
                  # this has no effect on the recommended workflow.
    if n_workers > 1:
        from joblib import Parallel, delayed

        # Local closure: the joblib map ignores its `func` argument because
        # DEAP only ever calls toolbox.map(toolbox.evaluate, pop), and the
        # worker wrapper always invokes evaluate_individual. Each job
        # carries its own context, making the call idempotent across runs.
        def _joblib_map(_unused_func, iterable):
            jobs = [
                (list(ind),
                 dict(static_inputs),
                 list(param_keys),
                 dict(opt_ranges))
                for ind in iterable
            ]
            return Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                delayed(_evaluate_for_joblib)(job) for job in jobs
            )

        toolbox.register("map", _joblib_map)
        if verbose:
            print(f"  Parallel fitness evaluation: {n_workers} workers (joblib/loky)")
    else:
        toolbox.register("map", map)   # built-in sequential map

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)
    
    # Statistics.
    # Fitness values fall into three regimes: feasible designs carry the
    # objective value, infeasible ones are mapped into the band
    # [floor, PENALTY_VALUE) by compute_feasibility_fitness, and structurally
    # invalid ones return PENALTY_VALUE. Population statistics must describe
    # feasible designs only, so the cut is the band floor, not
    # PENALTY_VALUE / 2. The floor is deck-overridable, so it is resolved here
    # exactly as in evaluate_individual.
    _floor_stats = _opt_float(static_inputs.get('infeasible_floor'))
    if _floor_stats is None:
        _floor_stats = DEFAULT_INFEASIBLE_FLOOR

    def _feasible(values):
        return [v[0] for v in values if v[0] < _floor_stats]

    def safe_mean(values):
        valid = _feasible(values)
        return np.mean(valid) if valid else float('nan')
    
    def safe_std(values):
        valid = _feasible(values)
        return np.std(valid) if len(valid) > 1 else float('nan')
    
    def count_valid(values):
        return len(_feasible(values))
    
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
    print(f"    Diversity injection every 7 generations")
    print(f"{'='*70}\n")

    # ── History trackers (defined before any evaluation that may use them) ──
    # `ga_cloud` logs every valid D0FUS evaluation that happens inside the
    # GA loop (initial pop, offspring of each generation, diversity injection).
    # It does NOT include refinement evaluations (those cluster near the
    # best by construction and would distort the exploration picture).
    # Each entry: {gen, params, fitness}.
    ga_cloud = []

    def _append_to_cloud(ind, gen_tag):
        """Append a single individual to ga_cloud if its fitness is valid."""
        if not ind.fitness.valid:
            return
        f = ind.fitness.values[0]
        if f >= PENALTY_VALUE / 2 or not np.isfinite(f):
            return
        ga_cloud.append({
            'gen': gen_tag,
            'params': {k: float(ind[i]) for i, k in enumerate(param_keys)},
            'fitness': float(f),
        })

    # `refinement_log` accumulates the info dicts returned by every
    # refine_topk call (periodic memetic + final). Each entry is tagged
    # with a 'phase' field ('memetic' / 'final') so the JSON summary
    # can break down the contribution per phase.
    refinement_log = []

    # Initial evaluation (toolbox.map: parallel when n_workers > 1)
    current_generation = 0
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        _append_to_cloud(ind, 0)

    # ── Diagnostic: abort early if entire initial population is invalid ───
    # Same feasibility cut as the population statistics above: a design sitting
    # in the infeasible band is not a valid starting point.
    n_init_valid = sum(1 for f in fitnesses if f[0] < _floor_stats)
    if n_init_valid == 0:
        print(f"\n  *** WARNING: 0 / {pop_size} initial designs are valid! ***")
        print(f"  The parameter ranges likely produce unphysical configurations")
        print(f"  (complex-valued sqrt, negative radial build, etc.).")
        # Run the first individual with verbose diagnostics
        print(f"\n  ── Diagnostic: evaluating first individual with tracing ──")
        _diag_params = {k: pop[0][i] for i, k in enumerate(param_keys)}
        for k, v in _diag_params.items():
            print(f"    {k} = {v:.4f}")
        print(f"  static_inputs keys that overlap param_keys: "
              f"{[k for k in param_keys if k in static_inputs]}")
        evaluate_individual(pop[0], verbose=True)
        print(f"  ── End diagnostic ──\n")
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
    # Two counters: stagnation of the population best triggers a diversity
    # injection (and is reset by it), while the number of generations
    # without improvement of the best design found so far drives the early
    # stop against `patience`.
    stagnation_counter = 0
    no_improve_counter = 0
    best_so_far = hof[0].fitness.values[0] if len(hof) > 0 else np.inf

    # ── History trackers for post-run analytics and visualization ──────────
    # `hof_history` records the cumulative-best individual at the end of
    # each generation, in the form {gen, params, fitness}. This is used
    # by animate_ga_trajectory() to render the design-point trajectory.
    hof_history = []
    if len(hof) > 0:
        hof_history.append({
            'gen': 0,
            'params': {k: float(hof[0][i]) for i, k in enumerate(param_keys)},
            'fitness': float(hof[0].fitness.values[0]),
        })
    
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
        
        # Evaluate (toolbox.map: parallel when n_workers > 1)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            _append_to_cloud(ind, gen)
        
        # Mu+Lambda selection — tournament-based, NOT pure elitist.
        # selBest kills diversity: once a basin dominates, exploration dies.
        # Tournament selection (size 2) keeps weaker individuals alive with
        # probability ~25%, allowing the GA to maintain multiple basins.
        combined = pop + offspring
        pop[:] = tools.selTournament(combined, mu, tournsize=3)
        
        # Minimal elitism: guarantee the overall best is never lost.
        # Replace the worst individual in pop with the HoF champion.
        if hof and hof[0].fitness.values[0] < max(ind.fitness.values[0] for ind in pop):
            worst_i = max(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
            pop[worst_i] = toolbox.clone(hof[0])
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        # Record cumulative-best parameters for post-run trajectory plots
        if len(hof) > 0:
            hof_history.append({
                'gen': gen,
                'params': {k: float(hof[0][i]) for i, k in enumerate(param_keys)},
                'fitness': float(hof[0].fitness.values[0]),
            })

        if verbose:
            print(logbook.stream)
        
        # Track stagnation of the population best (diversity trigger)
        current_best = record['min']
        best_fitness_history.append(current_best)
        
        if len(best_fitness_history) > 1:
            if abs(best_fitness_history[-1] - best_fitness_history[-2]) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

        # Track improvement of the best design found so far (early stop)
        hof_best = hof[0].fitness.values[0] if len(hof) > 0 else np.inf
        if hof_best < best_so_far - 1e-9 * max(1.0, abs(best_so_far)):
            best_so_far = hof_best
            no_improve_counter = 0
        else:
            no_improve_counter += 1
        
        # Inject diversity if stagnating or periodically.
        # The injection fraction is controlled by `diversity_inject_fraction`
        # (default 0.30 = 30% of pop). The previous fixed 10% was too small:
        # once the population had collapsed to one basin, 10% of fresh blood
        # could not overpower the basin's gravity.
        if stagnation_counter >= 3 or gen % 7 == 0:
            n_inject = max(5, int(pop_size * diversity_inject_fraction))
            pop, _injected = inject_diverse_individuals(
                pop, n_inject, opt_ranges, param_keys)
            hof.update(_injected)   # a better newcomer is kept even if not selected later
            for ind in _injected:
                _append_to_cloud(ind, gen)
            if stagnation_counter >= 3:
                stagnation_counter = 0
                if verbose:
                    print(f"    [Diversity injection: {n_inject} new individuals]")

        # ── Memetic periodic refinement ──────────────────────────────────
        # If enabled, refine the current best HoF member every N generations
        # and reinsert it. Heavy: keep N >= 5 unless the population is small.
        if (local_refine_method is not None
                and local_refine_every > 0
                and gen % local_refine_every == 0
                and len(hof) > 0):
            try:
                _periodic_info = refine_topk(
                    hof, k=1,
                    method=local_refine_method,
                    pool=_pool,
                    verbose=verbose,
                    **(local_refine_kwargs or {})
                )
            except Exception as _exc:
                # A failed refinement must not stop the GA
                print(f"    [refine] memetic refinement failed at generation "
                      f"{gen}: {type(_exc).__name__}: {_exc}")
                _periodic_info = []
            # Tag the info with the generation tag so we can identify
            # memetic events in the JSON output later.
            for _refind, _info in _periodic_info:
                _info['phase'] = 'memetic'
                _info['gen']   = gen
            refinement_log.extend(_info for _, _info in _periodic_info)
            # Also replace the worst pop member with the refined HoF[0]
            # so the GA picks up the improvement immediately.
            worst_i = max(range(len(pop)),
                          key=lambda i: pop[i].fitness.values[0])
            pop[worst_i] = toolbox.clone(hof[0])
            # The HoF best may have changed; refresh the history entry
            hof_history[-1] = {
                'gen': gen,
                'params': {k: float(hof[0][i]) for i, k in enumerate(param_keys)},
                'fitness': float(hof[0].fitness.values[0]),
            }

        # Early stopping: no improvement of the best design for `patience`
        # generations
        if patience and no_improve_counter >= patience:
            print(f"\n Converged at generation {gen} (no improvement for "
                  f"{patience} generations)")
            break

    # ── Final local refinement on diverse starters from the GA cloud ─────────
    # Rationale: by the end of the GA the HoF has often collapsed to k
    # near-clones of the same point, making `refine top-k of the HoF`
    # equivalent to refining one point k times. Selecting k diverse
    # starters from the cloud (which retains the full exploration
    # history) gives Nelder-Mead k different basins of attraction to
    # polish, drastically improving the chance of finding a globally
    # competitive design.
    if local_refine_method is not None and len(hof) > 0:
        # Build a synthetic HoF from diverse cloud entries
        diverse_starters = _select_diverse_from_cloud(
            ga_cloud, k=local_refine_top_k,
            opt_ranges_loc=opt_ranges, param_keys_loc=param_keys,
            min_dist_norm=cloud_refinement_min_dist,
            verbose=verbose,
        )
        if not diverse_starters:
            # Fallback: cloud was empty (should not happen) - use HoF as before.
            refine_hof = hof
            n_eff = local_refine_top_k
        else:
            # Build a fresh HoF of size = number of diverse starters and
            # populate it with proper Individuals derived from the cloud.
            n_eff = len(diverse_starters)
            refine_hof = tools.HallOfFame(n_eff)
            for d in diverse_starters:
                ind = creator.Individual(
                    [float(d['params'][k_]) for k_ in param_keys])
                ind.fitness.values = (float(d['fitness']),)
                refine_hof.update([ind])

        # Checkpoint of the GA result before the refinement, so that a
        # failure or an interruption during the refinement loses nothing.
        if current_run_directory is not None:
            _ckpt = os.path.join(current_run_directory, "hall_of_fame_ga.json")
            with open(_ckpt, "w", encoding="utf-8") as _fh:
                json.dump([{"fitness": to_serializable(ind.fitness.values[0]),
                            "params": {k_: float(ind[j]) for j, k_
                                       in enumerate(param_keys)}}
                           for ind in hof], _fh, indent=2)
        try:
            _final_info = refine_topk(
                refine_hof, k=n_eff,
                method=local_refine_method,
                pool=_pool,
                verbose=verbose,
                **(local_refine_kwargs or {})
            )
        except Exception as _exc:
            print(f"    [refine] final refinement failed, GA result kept: "
                  f"{type(_exc).__name__}: {_exc}")
            _final_info = []
        for _refind, _info in _final_info:
            _info['phase'] = 'final'
            # Propagate any refined point that beats the main HoF
            hof.update([_refind])
        refinement_log.extend(_info for _, _info in _final_info)
        # Final HoF best may have changed; append a terminal history entry
        hof_history.append({
            'gen': hof_history[-1]['gen'] if hof_history else 0,
            'params': {k: float(hof[0][i]) for i, k in enumerate(param_keys)},
            'fitness': float(hof[0].fitness.values[0]),
            'phase': 'post_refinement',
        })

    # ── No explicit pool teardown needed ──────────────────────────────────────
    # joblib.Parallel manages its own worker lifecycle and shuts the loky
    # backend down automatically after each Parallel(...) invocation.
    # The `_pool = None` placeholder above is retained only for the
    # `pool=_pool` arguments propagated to refine_topk further down.

    return hof[0], logbook, hof, hof_history, refinement_log, ga_cloud

#%% Local Refinement (Memetic Hybrid: gradient descent / L-BFGS-B)
#
# Rationale
# ---------
# The Genetic Algorithm is good at exploring the global parameter space but
# converges slowly inside a single basin of attraction. We graft a local
# search on top of it that takes the best Hall-of-Fame members and refines
# them with a derivative-based optimiser. This is the standard "memetic
# algorithm" pattern (GA = exploration, gradient = exploitation).
#
# All local-search work is performed in *normalised* coordinates u in [0,1]^d
# so that the finite-difference step h is dimensionless and the algorithm
# is scale-invariant across parameters of very different magnitudes
# (e.g. R0 in [3, 10] vs Tbar in [5, 30]).
#
# Two optimisers are exposed:
#   - 'gradient' : hand-rolled projected gradient descent with Armijo
#                  backtracking line search. Centered finite differences.
#                  Pedagogical and transparent.
#   - 'lbfgsb'   : scipy.optimize.minimize, method='L-BFGS-B'. Quasi-Newton
#                  with bound constraints. Generally faster in smooth regions.
#
# Caveats
# -------
# The feasibility band of compute_feasibility_fitness() puts a step in the
# fitness at the constraint boundaries. The local search assumes the starting point
# is feasible (top of HoF), which keeps it inside the smooth interior most
# of the time. A moderate h (1e-3 in unit coordinates) further smooths
# residual micro-discontinuities.

def _to_unit_cube(x_phys, opt_ranges, param_keys):
    """Map a physical parameter vector to the unit hypercube [0,1]^d."""
    return np.array([
        (x_phys[i] - opt_ranges[k][0]) / (opt_ranges[k][1] - opt_ranges[k][0])
        for i, k in enumerate(param_keys)
    ], dtype=float)


def _from_unit_cube(u, opt_ranges, param_keys):
    """Inverse mapping: unit-cube point to physical parameters."""
    return np.array([
        opt_ranges[k][0] + u[i] * (opt_ranges[k][1] - opt_ranges[k][0])
        for i, k in enumerate(param_keys)
    ], dtype=float)


def _evaluate_unit(u, opt_ranges, param_keys):
    """
    Evaluate fitness at a unit-cube point.

    The point is clipped to [0,1]^d, mapped to physical parameters, and
    passed through evaluate_individual(). Returns a scalar (lower is better).
    Uses module-level static_inputs implicitly via evaluate_individual.
    """
    u_clipped = np.clip(u, 0.0, 1.0)
    x_phys = _from_unit_cube(u_clipped, opt_ranges, param_keys)
    return evaluate_individual(list(x_phys))[0]


def _eval_unit_for_pool(u):
    """
    Picklable top-level wrapper used by pool.map for parallel gradient
    computation.

    Relies on the worker-process module globals `opt_ranges` and
    `param_keys` being populated by `_pool_initializer`. When the gradient
    is evaluated outside the GA loop (standalone call to refine_individual),
    the caller is responsible for ensuring the pool was initialised with a
    matching set of ranges and keys.
    """
    return _evaluate_unit(u, opt_ranges, param_keys)


def _gradient_central(u, opt_ranges, param_keys, h=1e-3, pool=None):
    """
    Centered finite-difference gradient at a unit-cube point.

    Requires 2*d function evaluations. The step h is in normalised
    coordinates so it applies uniformly to all parameters.

    Parameters
    ----------
    pool : multiprocessing.Pool or None
        If provided, the 2*d evaluations are dispatched in parallel via
        pool.map. This typically yields a (n_workers)-fold speedup, capped
        by 2*d (no benefit beyond 2*d workers for a single gradient call).

    Returns a (d,) numpy array; NaN entries are replaced with 0 to keep
    the descent direction defined when one component happens to land on
    a discontinuity.
    """
    d = len(u)
    # Build the 2*d perturbed points
    pts_plus, pts_minus, steps = [], [], []
    for i in range(d):
        u_plus = u.copy();  u_plus[i] = min(1.0, u[i] + h)
        u_minus = u.copy(); u_minus[i] = max(0.0, u[i] - h)
        pts_plus.append(u_plus)
        pts_minus.append(u_minus)
        steps.append(u_plus[i] - u_minus[i])

    pts_all = pts_plus + pts_minus
    if pool is not None:
        fvals = pool.map(_eval_unit_for_pool, pts_all)
    else:
        fvals = [_evaluate_unit(p, opt_ranges, param_keys) for p in pts_all]

    f_plus = fvals[:d]
    f_minus = fvals[d:]

    grad = np.zeros(d)
    for i in range(d):
        if steps[i] <= 0:
            grad[i] = 0.0
            continue
        g_i = (f_plus[i] - f_minus[i]) / steps[i]
        grad[i] = g_i if np.isfinite(g_i) else 0.0
    return grad


def gradient_descent_unit(u0, opt_ranges, param_keys,
                          max_iters=30, h=1e-3,
                          step_init=0.1, c1=1e-4,
                          max_backtrack=15,
                          tol_grad=1e-5, tol_step=1e-6,
                          pool=None, verbose=False,
                          **_ignored):
    """
    Box-constrained projected gradient descent on the unit hypercube.

    Algorithm
    ---------
    1. Compute gradient g by centered finite differences (2*d evals,
       optionally dispatched in parallel via `pool`).
    2. Direction d = -g / ||g||  (normalised steepest descent).
    3. Armijo backtracking line search: find the largest alpha such that
           f(u + alpha * d) <= f(u) - c1 * alpha * ||g||
       (the directional derivative along d is exactly -||g||).
    4. Project the new point back to [0,1]^d (clip).
    5. Repeat until ||g|| < tol_grad, step < tol_step, or max_iters reached.

    Parameters
    ----------
    u0 : array-like, shape (d,)
        Starting point in [0,1]^d.
    pool : multiprocessing.Pool or None
        Used only for the gradient computation. The line search itself is
        sequential (each trial point depends on the previous one).
    Other parameters : see _gradient_central and module docstring.

    Returns
    -------
    u, f, info : final point, final fitness, diagnostics dict.
    """
    u = np.array(u0, dtype=float).clip(0.0, 1.0)
    f = _evaluate_unit(u, opt_ranges, param_keys)

    if not np.isfinite(f) or f >= PENALTY_VALUE / 2:
        return u, f, {'iters': 0, 'success': False,
                      'reason': 'invalid_starting_point',
                      'history': [f]}

    history = [f]
    step = step_init

    for it in range(1, max_iters + 1):
        grad = _gradient_central(u, opt_ranges, param_keys, h=h, pool=pool)
        gnorm = np.linalg.norm(grad)

        if gnorm < tol_grad:
            return u, f, {'iters': it - 1, 'success': True,
                          'reason': 'small_gradient', 'history': history}

        # Steepest descent direction (unit length)
        direction = -grad / gnorm

        # Armijo backtracking line search
        alpha = step
        u_new, f_new = u, f
        accepted = False
        for _ in range(max_backtrack):
            u_try = np.clip(u + alpha * direction, 0.0, 1.0)
            f_try = _evaluate_unit(u_try, opt_ranges, param_keys)
            if np.isfinite(f_try) and f_try <= f - c1 * alpha * gnorm:
                u_new, f_new = u_try, f_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return u, f, {'iters': it - 1, 'success': False,
                          'reason': 'line_search_failed', 'history': history}

        step_change = np.linalg.norm(u_new - u)
        u, f = u_new, f_new
        history.append(f)

        # Adaptive step: grow back, capped at 1.0 (full cube edge)
        step = min(alpha * 2.0, 1.0)

        if verbose:
            print(f"    [grad] it {it:3d}  f={f:.6g}  |g|={gnorm:.3g}  "
                  f"alpha={alpha:.3g}")

        if step_change < tol_step:
            return u, f, {'iters': it, 'success': True,
                          'reason': 'small_step', 'history': history}

    return u, f, {'iters': max_iters, 'success': True,
                  'reason': 'max_iters', 'history': history}


def lbfgsb_unit(u0, opt_ranges, param_keys,
                max_iters=50, ftol=1e-7, gtol=1e-5,
                eps=1e-3, pool=None, verbose=False,
                **_ignored):
    """
    L-BFGS-B refinement in unit-cube coordinates.

    Thin wrapper around scipy.optimize.minimize with bounds = [(0,1)]*d.
    When pool is None, the gradient is approximated internally by scipy
    via forward finite differences. When pool is provided, we supply an
    explicit Jacobian computed by `_gradient_central` (centered FD,
    parallelised via the pool), which is both more accurate and faster
    for moderate worker counts.

    Returns
    -------
    (u, f, info) tuple.
    """
    def objective(u):
        v = _evaluate_unit(u, opt_ranges, param_keys)
        # L-BFGS-B does not tolerate NaN; remap to a large finite value
        return v if np.isfinite(v) else PENALTY_VALUE

    bounds = [(0.0, 1.0)] * len(u0)

    minimize_kwargs = dict(
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iters, 'ftol': ftol, 'gtol': gtol},
    )

    if pool is not None:
        # Provide an explicit (parallel) Jacobian; scipy then does not call
        # its own internal finite-difference loop.
        def jac(u):
            return _gradient_central(u, opt_ranges, param_keys,
                                     h=eps, pool=pool)
        minimize_kwargs['jac'] = jac
    else:
        # Fall back to scipy-internal forward FD with step `eps`.
        minimize_kwargs['options']['eps'] = eps

    res = minimize(
        objective,
        np.array(u0, dtype=float).clip(0.0, 1.0),
        **minimize_kwargs,
    )

    u_out = np.clip(res.x, 0.0, 1.0)
    f_out = _evaluate_unit(u_out, opt_ranges, param_keys)

    if verbose:
        print(f"    [lbfgsb] nit={getattr(res, 'nit', '?')}  "
              f"nfev={getattr(res, 'nfev', '?')}  "
              f"success={bool(res.success)}  msg={res.message}")

    return u_out, f_out, {
        'iters':   getattr(res, 'nit', None),
        'nfev':    getattr(res, 'nfev', None),
        'success': bool(res.success),
        'reason':  str(res.message),
        'history': [f_out],
    }


def nelder_mead_unit(u0, opt_ranges, param_keys,
                     max_iters=200, xatol=1e-5, fatol=1e-7,
                     adaptive=True, verbose=False,
                     initial_simplex_scale=0.20,
                     **_ignored):
    """
    Nelder-Mead simplex refinement in unit-cube coordinates.

    Derivative-free, robust to non-smooth and noisy objectives. Good
    fallback when the step of the fitness at the constraint boundaries
    (feasibility band) throws off gradient-based methods.

    Bounds are respected via scipy.optimize.minimize's native bounds
    support (requires scipy >= 1.7). The `adaptive` flag enables the
    Gao-Han adaptive parameter scheme (helpful in higher dimensions).

    The initial simplex is built explicitly with edge length
    `initial_simplex_scale` (in unit-cube coordinates, default 0.20 =
    20% of the search range). Scipy's default (~5% of the starting
    point's value) is too narrow when the GA has converged to a basin
    that is not the global optimum: the simplex never reaches a
    neighbouring basin. A larger initial simplex lets the refinement
    cross small ridges between basins.

    Notes
    -----
    Pool parallelism does not apply: Nelder-Mead is intrinsically
    sequential (each simplex move depends on the previous one).
    The `pool` kwarg is accepted but silently ignored.

    Returns
    -------
    (u, f, info) tuple.
    """
    def objective(u):
        v = _evaluate_unit(u, opt_ranges, param_keys)
        return v if np.isfinite(v) else PENALTY_VALUE

    u0_arr = np.array(u0, dtype=float).clip(0.0, 1.0)
    d = len(u0_arr)
    bounds = [(0.0, 1.0)] * d

    # Build a non-degenerate initial simplex of edge length
    # `initial_simplex_scale`. Each non-base vertex offsets u0 by ±scale
    # in one axis. We pick the sign that stays inside the unit cube.
    step = float(initial_simplex_scale)
    simplex = np.tile(u0_arr, (d + 1, 1))
    for i in range(d):
        if u0_arr[i] + step <= 1.0:
            simplex[i + 1, i] = u0_arr[i] + step
        else:
            simplex[i + 1, i] = max(0.0, u0_arr[i] - step)

    res = minimize(
        objective,
        u0_arr,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': max_iters, 'xatol': xatol, 'fatol': fatol,
                 'adaptive': adaptive,
                 'initial_simplex': simplex},
    )

    u_out = np.clip(res.x, 0.0, 1.0)
    f_out = _evaluate_unit(u_out, opt_ranges, param_keys)

    if verbose:
        print(f"    [NM] nit={getattr(res, 'nit', '?')}  "
              f"nfev={getattr(res, 'nfev', '?')}  "
              f"success={bool(res.success)}  msg={res.message}  "
              f"simplex_scale={step:.3f}")

    return u_out, f_out, {
        'iters':   getattr(res, 'nit', None),
        'nfev':    getattr(res, 'nfev', None),
        'success': bool(res.success),
        'reason':  str(res.message),
        'history': [f_out],
    }


def refine_individual(individual, opt_ranges_loc, param_keys_loc,
                      method='gradient', pool=None, verbose=False, **kwargs):
    """
    Refine a single DEAP individual via local search.

    Parameters
    ----------
    individual : sequence
        Physical parameter vector in the order defined by param_keys_loc.
        Typically a DEAP Individual from the Hall of Fame.
    opt_ranges_loc, param_keys_loc :
        Pass the module-level objects explicitly so the function is
        unit-testable without relying on globals.
    method : {'gradient', 'lbfgsb', 'nelder-mead' or 'nm'}
        Local optimiser. Any other name raises a ValueError.
    pool : multiprocessing.Pool or None
        Forwarded to the underlying optimiser. Used by 'gradient' and
        'lbfgsb' for parallel gradient evaluation; ignored by 'nelder-mead'.
    kwargs : passed through to the underlying optimiser.

    Returns
    -------
    refined : creator.Individual
        New individual with refined parameters and updated fitness.
    info : dict
        Augmented with 'f_before', 'f_after', 'improvement', 'method'.
    """
    x0 = np.array(list(individual), dtype=float)
    u0 = _to_unit_cube(x0, opt_ranges_loc, param_keys_loc)

    if individual.fitness.valid:
        f_before = individual.fitness.values[0]
    else:
        f_before = _evaluate_unit(u0, opt_ranges_loc, param_keys_loc)

    method_norm = str(method).lower().strip()
    if method_norm in ('nm', 'nelder-mead', 'neldermead'):
        method_norm = 'nelder-mead'
    if method_norm not in VALID_REFINE_METHODS:
        raise ValueError(f"Unknown local refinement method '{method}'. "
                         f"Valid: {', '.join(VALID_REFINE_METHODS)}.")

    method_used = method_norm
    result = None

    if method_norm == 'lbfgsb':
        result = lbfgsb_unit(u0, opt_ranges_loc, param_keys_loc,
                             pool=pool, verbose=verbose, **kwargs)
        if result is None:
            method_used = 'gradient'
            if verbose:
                print("    [refine] scipy unavailable, falling back to gradient")

    elif method_norm == 'nelder-mead':
        result = nelder_mead_unit(u0, opt_ranges_loc, param_keys_loc,
                                  verbose=verbose, **kwargs)
        if result is None:
            method_used = 'gradient'
            if verbose:
                print("    [refine] scipy unavailable, falling back to gradient")

    if result is None:  # gradient (either requested or fallback)
        # Filter kwargs to those understood by gradient_descent_unit
        _grad_keys = {'max_iters', 'h', 'step_init', 'c1', 'max_backtrack',
                      'tol_grad', 'tol_step'}
        grad_kwargs = {k: v for k, v in kwargs.items() if k in _grad_keys}
        result = gradient_descent_unit(u0, opt_ranges_loc, param_keys_loc,
                                       pool=pool, verbose=verbose,
                                       **grad_kwargs)

    u_out, f_out, info = result
    x_out = _from_unit_cube(u_out, opt_ranges_loc, param_keys_loc)

    refined = creator.Individual(list(x_out))
    refined.fitness.values = (f_out,)

    info = dict(info)
    info['f_before']    = float(f_before)
    info['f_after']     = float(f_out)
    info['improvement'] = float(f_before - f_out)
    info['method']      = method_used
    return refined, info


def _select_diverse_from_cloud(cloud, k, opt_ranges_loc, param_keys_loc,
                               min_dist_norm=0.05, verbose=False):
    """
    Select k diverse points from the GA exploration cloud.

    Rationale
    ---------
    The Hall of Fame typically collapses by mid-run: once the population
    converges to one basin, the top-k members all snapshot the same
    point with infinitesimal numerical differences. Refining ``top-k of
    a collapsed HoF'' is then equivalent to refining the same point k
    times, which is wasteful and misses other basins. The GA cloud, in
    contrast, contains EVERY valid evaluation across all generations,
    including poorer-but-distinct points sampled before convergence.
    Picking diverse starters from the cloud preserves the chance to
    discover a competitive alternative basin during refinement.

    Algorithm
    ---------
    1. Sort the cloud by fitness ascending (best first).
    2. Add the best point as the first starter.
    3. Walk down the sorted list; add a candidate iff its L_inf distance
       in normalised unit-cube space from EVERY already-selected starter
       is at least ``min_dist_norm`` (default 0.05 = 5% of the range).
    4. Stop when k starters are selected, or when the sorted list is
       exhausted. If fewer than k pass the distance filter, top up with
       the next best candidates regardless of distance.

    Parameters
    ----------
    cloud : list of dict
        Each entry has keys 'gen', 'fitness', 'params' (with 'params' a
        dict {param_name: value}).
    k : int
        Number of diverse starters to select.
    opt_ranges_loc, param_keys_loc : as in refine_individual.
    min_dist_norm : float
        Minimum L_inf distance in unit-cube coordinates between any two
        selected starters. 0.05 (5% of range) is a sensible default for
        4D problems; raise to 0.10 for stronger diversity, lower to
        0.02 for less.

    Returns
    -------
    list of dict
        The selected cloud entries in order of selection. Empty if the
        cloud is empty.
    """
    if not cloud or k < 1:
        return []

    sorted_cloud = sorted(cloud, key=lambda c: c['fitness'])

    def _to_unit(params_dict):
        out = np.zeros(len(param_keys_loc), dtype=float)
        for j, key in enumerate(param_keys_loc):
            lo, hi = opt_ranges_loc[key]
            out[j] = (params_dict[key] - lo) / max(hi - lo, 1e-12)
        return out

    selected = []
    selected_units = []

    for cand in sorted_cloud:
        u_cand = _to_unit(cand['params'])
        if not selected_units:
            selected.append(cand)
            selected_units.append(u_cand)
            continue
        dists = [float(np.max(np.abs(u_cand - u_s))) for u_s in selected_units]
        if min(dists) >= min_dist_norm:
            selected.append(cand)
            selected_units.append(u_cand)
            if len(selected) >= k:
                break

    # Top up with next-best candidates if we couldn't find enough diverse ones.
    if len(selected) < k:
        already_ids = set(id(c) for c in selected)
        for cand in sorted_cloud:
            if id(cand) in already_ids:
                continue
            selected.append(cand)
            if len(selected) >= k:
                break

    if verbose:
        print(f"  [cloud] selected {len(selected)} diverse starters "
              f"(min_dist_norm = {min_dist_norm})")
        for i, s in enumerate(selected, 1):
            print(f"    [{i}] f = {s['fitness']:.2f}  params = "
                  + "  ".join(f"{k}={v:.3g}" for k, v in s['params'].items()))

    return selected


def refine_topk(hof, k=3, method='gradient', pool=None,
                verbose=True, **kwargs):
    """
    Refine the top-k individuals of the Hall of Fame.

    Each refined individual is reinserted via hof.update(), so the HoF
    automatically keeps the best k members (refined or not). Improvements
    are reported per individual.

    Parameters
    ----------
    hof : deap.tools.HallOfFame
        Hall of Fame to refine and update in place.
    k : int
        Number of top members to refine (clamped to len(hof)).
    method : {'gradient', 'lbfgsb', 'nelder-mead'}
        See refine_individual().
    pool : multiprocessing.Pool or None
        Forwarded to refine_individual for parallel gradient evaluation.
    kwargs : forwarded to refine_individual / underlying optimiser.

    Returns
    -------
    list of (refined_individual, info) tuples, in the order they were
    refined (rank 1 first).
    """
    k_eff = min(int(k), len(hof))
    if k_eff < 1:
        return []

    if verbose:
        pool_tag = f" (pool={pool._processes})" if pool is not None else ""
        print(f"\n{'-' * 70}")
        print(f" Local refinement: top-{k_eff} via {method}{pool_tag}")
        print(f"{'-' * 70}")

    out = []
    # Snapshot the starting individuals because hof gets mutated below.
    starters = [hof[i] for i in range(k_eff)]

    for rank, ind in enumerate(starters, start=1):
        f0 = ind.fitness.values[0] if ind.fitness.valid else float('nan')
        if verbose:
            print(f"\n  [#{rank}] starting fitness = {f0:.6g}")

        refined, info = refine_individual(
            ind, opt_ranges, param_keys,
            method=method, pool=pool, verbose=verbose, **kwargs
        )

        if verbose:
            print(f"  [#{rank}] refined fitness  = {info['f_after']:.6g}  "
                  f"(delta = {info['improvement']:+.6g}, "
                  f"iters={info.get('iters')}, "
                  f"reason={info.get('reason')})")

        # Tag the info with the starting rank for downstream reporting
        info['starting_rank'] = rank
        # Snapshot the refined parameters as well (helpful for JSON output)
        info['params'] = {k: float(refined[i])
                          for i, k in enumerate(param_keys)}

        hof.update([refined])
        out.append((refined, info))

    if verbose:
        print(f"{'-' * 70}\n")
    return out


#%% Plotting

def plot_convergence(logbook, save_path):
    """
    Convergence plot: best fitness per generation and number of feasible
    designs. Log scale when every best fitness is positive, linear scale
    otherwise (objective 'P_elec' is stored as -P_elec). The style is local
    to this figure (rc_context), the global rcParams are not changed.
    """
    with plt.rc_context({'font.size': 11, 'axes.labelsize': 12,
                         'axes.titlesize': 14}):
        fig, ax = plt.subplots(figsize=(10, 6))

        gens = [record['gen'] for record in logbook]
        mins = [record['min'] for record in logbook]
        n_valid = [record['n_valid'] for record in logbook]
        log_scale = all(m > 0 for m in mins)

        # Main convergence curve
        ax.plot(gens, mins, 'o-', color='#2ecc71', linewidth=2.5,
                markersize=5, label='Best fitness')
        if log_scale:
            ax.set_yscale('log')
            ax.fill_between(gens, mins, alpha=0.2, color='#2ecc71')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Best fitness (log scale)' if log_scale else 'Best fitness')
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
        ax2.set_ylabel('Feasible designs per generation', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print(f" Convergence plot saved: {save_path}")


#%% Visualization (parameter-space trajectory)
#
# Build a 2D fitness scan over a chosen pair of design parameters (default:
# `a` and `R0`) at fixed values of the other parameters (set to the
# best-found individual). Then animate the cumulative-best trajectory
# from the GA on top of this surface as a GIF.
#
# Two views are stacked in the same animation:
#   - Left  : 3D surface (cost as height), with the trajectory as a 3D
#             polyline and a marker for the current generation.
#   - Right : 2D contour (cost as colour), same trajectory.
#
# Cost is plotted on a log10 scale by default since the GA typically
# spans several orders of magnitude (PENALTY_VALUE is 1e6, valid designs
# can be 1 to 1000 in COE).

def _eval_scan_point_for_pool(args):
    """
    Pool wrapper for the 2D scan. `args` is (idx, params_phys_list).

    Returns (idx, fitness) so results can be reordered after pool.map
    (parallelised mapping is otherwise non-deterministic in order).

    Note: kept for backwards compatibility with sequential code paths.
    The parallel path uses `_eval_scan_point_for_joblib` instead, which
    follows the joblib/loky pattern (cloudpickle-based, robust to module
    reload under Spyder's %runfile).
    """
    idx, params = args
    return idx, evaluate_individual(list(params))[0]


def _eval_scan_point_for_joblib(args):
    """
    Joblib worker for the 2D fitness scan, mirroring _evaluate_for_joblib.

    Imports are local and the full context travels with the job so the
    worker does not depend on module globals being pre-set by an
    initializer. This avoids the multiprocessing.Pool pickling failures
    observed in Spyder after a first run pollutes sys.path.

    Parameters
    ----------
    args : tuple
        (idx, params_phys_list, static_inputs_dict, param_keys_list,
         opt_ranges_dict)

    Returns
    -------
    (idx, fitness) : tuple
    """
    # Bootstrap imports only (see module header); aliased to match the
    # worker idiom of D0FUS_scan._run_scan_point.
    import os as _os
    import sys as _sys

    _parent = _os.path.normpath(
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)

    idx, params, static_inputs_dict, param_keys_list, opt_ranges_dict = args

    from D0FUS_EXE import D0FUS_genetic as _gen_mod
    _gen_mod.static_inputs = static_inputs_dict
    _gen_mod.param_keys    = param_keys_list
    _gen_mod.opt_ranges    = opt_ranges_dict

    return idx, _gen_mod.evaluate_individual(list(params))[0]


def scan_2d_for_animation(best_individual, opt_ranges, param_keys,
                          var_x='a', var_y='R0', n_x=30, n_y=30,
                          n_workers=1, verbose=True):
    """
    Build a 2D fitness scan over (var_x, var_y) with all other parameters
    fixed at the values of `best_individual`.

    Parameters
    ----------
    best_individual : sequence
        Reference design point; var_x and var_y are swept while the
        remaining components stay at their `best_individual` value.
    var_x, var_y : str
        Parameter names to sweep. Both must appear in opt_ranges. The
        sweep range is taken from opt_ranges[var_x] and opt_ranges[var_y].
    n_x, n_y : int
        Grid resolution. Total cost ~= n_x * n_y D0FUS evaluations.
    n_workers : int
        If > 1, dispatches evaluations to a fresh multiprocessing.Pool.
        The GA's own pool is already closed by this point.

    Returns
    -------
    XX, YY : 2D ndarrays of shape (n_y, n_x), in physical units.
    Z : 2D ndarray of fitness values.
    """
    if var_x not in opt_ranges or var_y not in opt_ranges:
        raise ValueError(
            f"var_x={var_x!r} and var_y={var_y!r} must both be in opt_ranges "
            f"({list(opt_ranges.keys())})"
        )

    ix = param_keys.index(var_x)
    iy = param_keys.index(var_y)

    x_vals = np.linspace(*opt_ranges[var_x], n_x)
    y_vals = np.linspace(*opt_ranges[var_y], n_y)
    XX, YY = np.meshgrid(x_vals, y_vals)

    # Build the parameter list for every grid point
    base = list(best_individual)
    work = []
    for i in range(n_y):
        for j in range(n_x):
            params = list(base)
            params[ix] = float(XX[i, j])
            params[iy] = float(YY[i, j])
            work.append((i * n_x + j, params))

    n_pts = len(work)
    if verbose:
        print(f"\n Building 2D scan: {var_x} x {var_y} = {n_x} x {n_y} "
              f"= {n_pts} D0FUS evaluations "
              f"({'parallel' if n_workers > 1 else 'sequential'})")

    fvals = [np.nan] * n_pts
    if n_workers > 1:
        # Same joblib/loky pattern as the main GA loop: cloudpickle handles
        # module-reload pollution that breaks multiprocessing.Pool under
        # Spyder's %runfile on successive executions.
        from joblib import Parallel, delayed

        jobs = [
            (idx, params,
             dict(static_inputs), list(param_keys), dict(opt_ranges))
            for idx, params in work
        ]
        # joblib preserves submission order, so we can use the idx returned
        # by the worker as a consistency check rather than for reordering.
        results = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
            delayed(_eval_scan_point_for_joblib)(job) for job in jobs
        )
        for idx, f in results:
            fvals[idx] = f
    else:
        for idx, params in work:
            fvals[idx] = evaluate_individual(list(params))[0]

    Z = np.array(fvals, dtype=float).reshape(n_y, n_x)
    return XX, YY, Z


def animate_ga_trajectory(XX, YY, Z, hof_history, save_path,
                          var_x='a', var_y='R0',
                          log_scale=True, fps=8, dpi=100,
                          rotate_3d=True,
                          ga_cloud=None,
                          cloud_alpha=0.12, cloud_size=10,
                          cloud_color='0.25',
                          verbose=True):
    """
    Build an animated GIF of the GA trajectory in (var_x, var_y) space.

    Parameters
    ----------
    XX, YY, Z : 2D ndarrays
        Output of scan_2d_for_animation. Physical units for XX, YY;
        raw fitness for Z (log scaling applied here if requested).
    hof_history : list of dicts
        As returned by run_genetic_algorithm. Each entry must contain
        'gen', 'params' (dict keyed by param name), and 'fitness'.
    save_path : str
        Output path for the GIF (must end in .gif).
    log_scale : bool
        If True, plot log10(max(fitness, eps)) so the early generations
        (large fitness due to penalties) remain readable.
    fps : int
        Frames per second of the animation.
    rotate_3d : bool
        If True, slowly rotate the 3D camera over the animation.
    ga_cloud : list of dicts or None
        Every valid individual evaluated during the GA, with the same
        structure as hof_history entries. If provided, plotted as
        semi-transparent grey scatter that grows over frames, revealing
        the GA's exploration pattern in addition to the best trajectory.
    cloud_alpha, cloud_size, cloud_color :
        Visual style of the scatter. Defaults: alpha 0.12, size 10,
        a neutral medium-grey ('0.25' in matplotlib greyscale).

    Returns
    -------
    True on success, False on internal failure.
    """
    # Extract trajectory in physical units
    xs = np.array([h['params'][var_x] for h in hof_history], dtype=float)
    ys = np.array([h['params'][var_y] for h in hof_history], dtype=float)
    fs = np.array([h['fitness']        for h in hof_history], dtype=float)
    n_frames = len(xs)
    if n_frames < 2:
        if verbose:
            print(" [animate] not enough history points to animate")
        return False

    # Prepare colour-mapping data (apply log scale if requested)
    eps = 1e-3
    def transform(z):
        return np.log10(np.maximum(z, eps)) if log_scale else z

    Z_plot  = transform(Z)
    z_traj  = transform(fs)
    zlabel  = 'log10(fitness)' if log_scale else 'fitness'

    # ── Pre-extract the exploration cloud as numpy arrays ─────────────
    have_cloud = ga_cloud is not None and len(ga_cloud) > 0
    if have_cloud:
        cloud_gens = np.array([c.get('gen', 0) for c in ga_cloud], dtype=int)
        cloud_xs   = np.array([c['params'][var_x] for c in ga_cloud], dtype=float)
        cloud_ys   = np.array([c['params'][var_y] for c in ga_cloud], dtype=float)
        cloud_fs   = np.array([c['fitness']        for c in ga_cloud], dtype=float)
        cloud_zs   = transform(cloud_fs)

    # ── Map each animation frame to a 'current generation' threshold ──
    # This is what `cloud_gen <= threshold` will be tested against, so
    # the cloud grows incrementally as the trajectory plays out.
    frame_gens = np.array([h.get('gen', i) for i, h in enumerate(hof_history)],
                          dtype=int)

    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    # Left: 3D surface (set first so the scatter renders on top of it)
    ax3d.plot_surface(
        XX, YY, Z_plot, cmap='viridis', alpha=0.55,
        linewidth=0, antialiased=True,
    )
    ax3d.set_xlabel(var_x)
    ax3d.set_ylabel(var_y)
    ax3d.set_zlabel(zlabel)
    ax3d.set_title('Cost surface and best-design trajectory')

    # Right: 2D filled contour
    levels = np.linspace(np.nanmin(Z_plot), np.nanmax(Z_plot), 25)
    cf = ax2d.contourf(XX, YY, Z_plot, levels=levels, cmap='viridis')
    ax2d.set_xlabel(var_x)
    ax2d.set_ylabel(var_y)
    ax2d.set_title('Best-design trajectory (top-down view)')
    cbar = fig.colorbar(cf, ax=ax2d)
    cbar.set_label(zlabel)

    # ── Cloud artists (semi-transparent scatter, initially empty) ─────
    cloud3d = cloud2d = None
    if have_cloud:
        cloud3d = ax3d.scatter([], [], [], s=cloud_size,
                               c=cloud_color, alpha=cloud_alpha,
                               edgecolors='none', depthshade=False)
        cloud2d = ax2d.scatter([], [], s=cloud_size,
                               c=cloud_color, alpha=cloud_alpha,
                               edgecolors='none')

    # Trajectory artists (will be updated frame-by-frame)
    traj3d, = ax3d.plot([], [], [], color='red', linewidth=2.2, alpha=0.85)
    pt3d,   = ax3d.plot([], [], [], 'o', color='red',
                        markersize=8, markeredgecolor='black')

    traj2d, = ax2d.plot([], [], color='red', linewidth=2.2, alpha=0.9)
    pt2d,   = ax2d.plot([], [], 'o', color='red',
                        markersize=10, markeredgecolor='black')

    suptitle = fig.suptitle('', fontsize=13)

    # Camera angles for 3D rotation
    azim0 = -60
    azim_span = 60.0  # total rotation across the animation

    def init():
        traj3d.set_data([], [])
        traj3d.set_3d_properties([])
        pt3d.set_data([], [])
        pt3d.set_3d_properties([])
        traj2d.set_data([], [])
        pt2d.set_data([], [])
        if have_cloud:
            cloud3d._offsets3d = ([], [], [])
            cloud2d.set_offsets(np.empty((0, 2)))
        artists = [traj3d, pt3d, traj2d, pt2d, suptitle]
        if have_cloud:
            artists.extend([cloud3d, cloud2d])
        return artists

    def update(frame):
        f_idx = frame + 1  # include endpoint
        traj3d.set_data(xs[:f_idx], ys[:f_idx])
        traj3d.set_3d_properties(z_traj[:f_idx])
        pt3d.set_data([xs[frame]], [ys[frame]])
        pt3d.set_3d_properties([z_traj[frame]])

        traj2d.set_data(xs[:f_idx], ys[:f_idx])
        pt2d.set_data([xs[frame]], [ys[frame]])

        # Grow the exploration cloud up to the current frame's generation
        if have_cloud:
            gen_thresh = frame_gens[frame]
            mask = cloud_gens <= gen_thresh
            cloud3d._offsets3d = (cloud_xs[mask], cloud_ys[mask], cloud_zs[mask])
            cloud2d.set_offsets(np.column_stack([cloud_xs[mask],
                                                  cloud_ys[mask]]))

        if rotate_3d:
            azim = azim0 + azim_span * (frame / max(n_frames - 1, 1))
            ax3d.view_init(elev=28, azim=azim)

        gen_label = hof_history[frame].get('gen', frame)
        phase = hof_history[frame].get('phase', '')
        phase_str = f"  [{phase}]" if phase else ""
        # Cloud size annotation (useful for the manuscript)
        cloud_count = int((cloud_gens <= frame_gens[frame]).sum()) if have_cloud else 0
        cloud_str = f"   evals: {cloud_count}" if have_cloud else ""
        suptitle.set_text(
            f"Generation {gen_label}{phase_str}   "
            f"best fitness = {fs[frame]:.4g}   "
            f"{var_x} = {xs[frame]:.3f}   {var_y} = {ys[frame]:.3f}"
            f"{cloud_str}"
        )
        artists = [traj3d, pt3d, traj2d, pt2d, suptitle]
        if have_cloud:
            artists.extend([cloud3d, cloud2d])
        return artists

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1000 // max(fps, 1),
                         blit=False)
    writer = PillowWriter(fps=fps)

    if verbose:
        cloud_info = f" + {len(ga_cloud)} cloud points" if have_cloud else ""
        print(f" Rendering GIF ({n_frames} frames at {fps} fps{cloud_info})...")
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)
    if verbose:
        print(f" Animation saved: {save_path}")
    return True


#%% Main Interface

def resolve_ga_settings(static_inputs, overrides):
    """
    GA settings with the precedence argument > deck > default.

    Parameters
    ----------
    static_inputs : dict  Parsed deck. The GA_SETTINGS and local_refine_*
                          keys are removed from it (they are not design
                          inputs). The FITNESS_SETTINGS keys are typed and
                          kept, evaluate_individual reads them there.
    overrides     : dict  Arguments of run_genetic_optimization or CLI
                          options. None means "not given".

    Returns
    -------
    settings      : dict  Value of every GA_SETTINGS and FITNESS_SETTINGS key.
    sources       : dict  'argument', 'deck' or 'default' for each key.
    refine_kwargs : dict  Local optimiser knobs (local_refine_<knob>).
    """
    settings, sources = {}, {}
    for name, (default, kind) in {**GA_SETTINGS, **FITNESS_SETTINGS}.items():
        in_deck = name in static_inputs
        deck_value = (static_inputs[name] if name in FITNESS_SETTINGS
                      else static_inputs.pop(name, None)) if in_deck else None
        if overrides.get(name) is not None:
            value, src = _parse_setting(name, overrides[name], kind), 'argument'
        elif in_deck:
            value, src = _parse_setting(name, deck_value, kind), 'deck'
        else:
            value, src = default, 'default'
        settings[name], sources[name] = value, src
        if name in FITNESS_SETTINGS and value is not None:
            static_inputs[name] = value

    # Local refinement method: one of VALID_REFINE_METHODS or None
    m = settings['local_refine_method']
    if m is not None:
        m = m.lower()
        m = 'nelder-mead' if m in ('nm', 'neldermead') else m
        if m not in VALID_REFINE_METHODS:
            raise ValueError(f"local_refine_method = {settings['local_refine_method']!r}: "
                             f"valid values are {', '.join(VALID_REFINE_METHODS)} or none.")
        settings['local_refine_method'] = m

    # Optimiser knobs: deck keys local_refine_<knob>, then explicit kwargs
    refine_kwargs = {}
    for knob, kind in REFINE_KNOBS.items():
        key = f'local_refine_{knob}'
        if key in static_inputs:
            refine_kwargs[knob] = _parse_setting(key, static_inputs.pop(key), kind)
    static_inputs.pop('local_refine_kwargs', None)
    for knob, v in (overrides.get('local_refine_kwargs') or {}).items():
        refine_kwargs[knob] = _parse_setting(f'local_refine_{knob}', v,
                                             REFINE_KNOBS.get(knob, float))

    # Range checks
    for name in ('population_size', 'generations', 'n_workers', 'gif_resolution',
                 'gif_fps', 'local_refine_top_k'):
        if settings[name] < 1:
            raise ValueError(f"{name} = {settings[name]}: must be >= 1.")
    for name in ('crossover_rate', 'mutation_rate', 'diversity_inject_fraction'):
        if not 0.0 <= settings[name] <= 1.0:
            raise ValueError(f"{name} = {settings[name]}: must lie in [0, 1].")
    if settings['fitness_objective'] not in VALID_FITNESS_OBJECTIVES:
        raise ValueError(f"fitness_objective = {settings['fitness_objective']!r}: "
                         f"valid values are {', '.join(VALID_FITNESS_OBJECTIVES)}.")
    if settings['fitness_objective'] == 'custom':
        parse_fitness_expression(settings['fitness_expression'])   # fails early
    elif settings['fitness_expression'] is not None:
        print(f"  [warn] fitness_expression is ignored: fitness_objective = "
              f"{settings['fitness_objective']}, not custom.")
    return settings, sources, refine_kwargs


def run_genetic_optimization(input_file,
                             population_size=None,
                             generations=None,
                             crossover_rate=None,
                             mutation_rate=None,
                             patience=None,
                             seed=None,
                             verbose=True,
                             n_workers=None,
                             local_refine_method=None,
                             local_refine_top_k=None,
                             local_refine_every=None,
                             local_refine_kwargs=None,
                             diversity_inject_fraction=None,
                             cloud_refinement_min_dist=None,
                             make_gif=None,
                             gif_var_x=None,
                             gif_var_y=None,
                             gif_resolution=None,
                             gif_log_scale=None,
                             gif_fps=None,
                             output_dir=None,
                             save_figures=True):
    """
    Main optimization function.

    Every GA setting left at None takes the value of the deck, or else
    the default of GA_SETTINGS (population 200, 50 generations, crossover
    0.7, mutation 0.3, patience 20, sequential). An argument given here
    overrides the deck. The seed, when given neither here nor in the deck,
    is drawn at random, printed and recorded in optimization_summary.json,
    so that every run can be reproduced.

    Local refinement options
    ------------------------
    local_refine_method : {None, 'gradient', 'lbfgsb', 'nelder-mead'}
        Activate memetic refinement at the end of the GA (and optionally
        periodically). 'gradient' = hand-rolled projected gradient descent;
        'lbfgsb' = scipy L-BFGS-B with bounds; 'nelder-mead' = derivative-free
        simplex (robust to the step of the fitness at the constraint
        boundaries). None or 'none' disables the feature.
    local_refine_top_k : int
        Number of diverse starters refined after the GA (default: 3).
    local_refine_every : int
        Period (in generations) for periodic memetic refinement. 0 disables.
    local_refine_kwargs : dict or None
        Forwarded to the underlying optimiser (max_iters, h, step_init,
        ftol, gtol, eps, ...). Deck keys local_refine_<knob> do the same.

    Visualization options
    ---------------------
    make_gif : bool
        If True, after the GA build a 2D fitness scan over
        (gif_var_x, gif_var_y) at fixed best-design values for the other
        parameters, and render a GIF of the GA trajectory on the resulting
        surface (3D + 2D contour side-by-side).
    gif_var_x, gif_var_y : str
        Parameter names for the scan axes. Both must be optimised parameters
        (i.e. present in opt_ranges) for the scan range to be meaningful.
    gif_resolution : int
        Grid resolution per axis (total scan = gif_resolution^2 evaluations).
    gif_log_scale : bool
        Plot log10(fitness) so penalty-dominated regions remain readable.
    gif_fps : int
        Animation frame rate.

    Output options
    --------------
    output_dir : str or None
        Parent folder of the genetic/ result folders. Default:
        default_output_root() of D0FUS_run.
    save_figures : bool
        If False, neither convergence.png nor the trajectory GIF is made.
    """
    global current_run_directory
    
    print("="*80)
    print("D0FUS GENETIC ALGORITHM OPTIMIZATION")
    print("="*80)
    
    # Initialize
    initialize_run_directory(os.path.join(output_dir or default_output_root(),
                                          "genetic"))
    opt_ranges, static_inputs, param_keys = load_input_file(input_file)
    if len(param_keys) < 1:
        raise ValueError("The GENETIC mode needs at least one optimised "
                         "parameter written 'key = [min, max]'.")

    # ── Settings: argument > deck > default ──────────────────────────────────
    _args = dict(population_size=population_size, generations=generations,
                 crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                 patience=patience, seed=seed, n_workers=n_workers,
                 local_refine_method=local_refine_method,
                 local_refine_top_k=local_refine_top_k,
                 local_refine_every=local_refine_every,
                 local_refine_kwargs=local_refine_kwargs,
                 diversity_inject_fraction=diversity_inject_fraction,
                 cloud_refinement_min_dist=cloud_refinement_min_dist,
                 make_gif=make_gif, gif_var_x=gif_var_x, gif_var_y=gif_var_y,
                 gif_resolution=gif_resolution, gif_log_scale=gif_log_scale,
                 gif_fps=gif_fps)
    S, sources, refine_kwargs = resolve_ga_settings(static_inputs, _args)
    if S['seed'] is None:
        S['seed'], sources['seed'] = random.SystemRandom().randrange(2**31), 'drawn'
    random.seed(S['seed'])
    np.random.seed(S['seed'])

    objective = S['fitness_objective']
    _cost_model = static_inputs.get('cost_model', DEFAULT_CONFIG.cost_model)
    if objective in ('COE', 'C_invest') and _cost_model == 'None':
        raise ValueError(f"fitness_objective = {objective} needs a cost model: "
                         f"cost_model = None in the deck.")
    if (objective == 'custom' and _cost_model == 'None' and
            parse_fitness_expression(S['fitness_expression'])[1] & set(_EXPR_COST_NAMES)):
        raise ValueError("fitness_expression reads C_invest or COE, which need a "
                         "cost model: cost_model = None in the deck.")
    if objective == 'R0' and 'R0' not in param_keys:
        # Degenerate case: minimising a fixed major radius gives a constant
        # fitness, so the GA would only search the remaining design space
        # for any feasible point. Warn the user to bracket R0.
        print("  [warn] fitness_objective = 'R0' but R0 is not a GA "
              "parameter; declare it as a range (e.g. 'R0 = [4.0, 9.0]') "
              "to actually minimise the major radius.")

    print("\n GA settings (source):")
    for name in list(GA_SETTINGS) + list(FITNESS_SETTINGS):
        if sources[name] != 'default' or name in (
                'population_size', 'generations', 'crossover_rate',
                'mutation_rate', 'patience', 'seed', 'n_workers',
                'fitness_objective', 'feasibility_epsilon'):
            print(f"    {name:<26s} = {S[name]!s:<14s} ({sources[name]})")
    if objective == 'COE':
        print(f"    {'C_invest_max':<26s} = "
              f"{float(static_inputs.get('C_invest_max', DEFAULT_CONFIG.C_invest_max))*1e-3:.1f} "
              f"B EUR (soft budget ceiling)")
    if refine_kwargs:
        print(f"    local_refine_kwargs        = {refine_kwargs}")

    # Copy of the deck next to the results
    shutil.copy2(input_file, os.path.join(current_run_directory, "input_parameters.txt"))

    setup_toolbox(opt_ranges, param_keys, S['generations'])

    best, logbook, hof, hof_history, refinement_log, ga_cloud = run_genetic_algorithm(
        S['population_size'], S['generations'], S['crossover_rate'],
        S['mutation_rate'], S['patience'], verbose, n_workers=S['n_workers'],
        local_refine_method=S['local_refine_method'],
        local_refine_top_k=S['local_refine_top_k'],
        local_refine_every=S['local_refine_every'],
        local_refine_kwargs=refine_kwargs,
        diversity_inject_fraction=S['diversity_inject_fraction'],
        cloud_refinement_min_dist=S['cloud_refinement_min_dist'],
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
    # Resolve Greenwald closures (Tbar_mode / P_fus_mode) so that the summary
    # and save_run_output report the solved Tbar and P_fus, as in main()
    config = resolve_operating_point(config)
    final_output = run(config, verbose=0)
    
    # Extract key metrics via centralised index map
    nbar_line = _safe_real(final_output[_IDX['nbar_line']])
    nG        = _safe_real(final_output[_IDX['nG']])
    betaN_total = _safe_real(final_output[_IDX['betaN_total']])
    cost      = _safe_real(final_output[_IDX['cost']])
    Q         = _safe_real(final_output[_IDX['Q']])
    P_elec    = _safe_real(final_output[_IDX['P_elec']])
    c_TF      = _safe_real(final_output[_IDX['c']])
    d_CS      = _safe_real(final_output[_IDX['d']])
    r_d       = _safe_real(final_output[_IDX['r_d']])
    Ip        = _safe_real(final_output[_IDX['Ip']])
    B0        = _safe_real(final_output[_IDX['B0']])
    _core     = core_plasma_values(config, final_output)

    # Final check with the RUN rule (same verdict as the RUN report)
    _fz = evaluate_feasibility(config, final_output)
    q_kink = _fz['q_kink']
    _kink_param = config.kink_parameter
    _q_label = 'q95' if _kink_param == 'q95' else 'q*'
    q_lim, betaN_lim, Ip_lim = config.q_limit, config.betaN_limit, config.Ip_limit

    # Compute Sheffield COE for the best design (regardless of objective)
    _COE_best = np.nan
    _C_invest_best = np.nan
    _cres_best = evaluate_costs(config, final_output)
    if _cres_best:
        _COE_best      = _cres_best['COE']
        _C_invest_best = _cres_best['C_invest'] * 1e-3  # M EUR -> B EUR

    print(f"\n Fitness objective: {objective}")
    if objective == 'custom':
        _cres_expr = _cres_best if _cres_best else None
        print(f"    Expression: {S['fitness_expression']}")
        print(f"    Value     : {evaluate_fitness_expression(S['fitness_expression'], config, final_output, _cres_expr):.6g}")
    if objective == 'COE':
        _budget = config.C_invest_max
        _within = _C_invest_best * 1e3 <= _budget if np.isfinite(_C_invest_best) else False
        _tag = "WITHIN BUDGET" if _within else "OVER BUDGET"
        print(f"    Budget ceiling:             {_budget*1e-3:.1f} B EUR  [{_tag}]")
    print("\n Best design metrics:")
    print(f"    R0      (major radius)        : {config.R0:9.3f}  [m]")
    print(f"    a       (minor radius)        : {config.a:9.3f}  [m]")
    print(f"    B0      (on-axis field)       : {B0:9.3f}  [T]")
    print(f"    Ip      (plasma current)      : {Ip:9.2f}  [MA]")
    print(f"    Q       (fusion gain)         : {Q:9.2f}")
    print(f"    P_elec  (net electric)        : {P_elec:9.1f}  [MWe]")
    print(f"    COE     (Sheffield)           : {_COE_best:9.1f}  [EUR/MWh]")
    print(f"    C_invest                      : {_C_invest_best:9.2f}  [B EUR]")
    print(f"    V/P_fus (volume proxy)        : {cost:9.4f}  [m^3/MW]")
    print(f"    c_TF / d_CS / r_d             : {c_TF:.3f} / {d_CS:.3f} / {r_d:.3f}  [m]")

    print("\n Core plasma (rho = 0):")
    print(f"    Te0 / Ti0                     : {_core['Te0']:.2f} / {_core['Ti0']:.2f}  [keV]")
    print(f"    ne0 / ni0 (fuel ions)         : {_core['ne0']:.3f} / {_core['ni0']:.3f}  [1e20 m^-3]")

    # Operational limits: margin > 0 means inside the limit
    def _lim_row(label, value, limit, upper=True, note=""):
        if limit is None or not np.isfinite(limit) or limit <= 0:
            print(f"    {label:<26s}{value:10.3f}{'':>10s}{'':>11s}  {note}")
            return
        margin = (1.0 - value / limit) if upper else (value / limit - 1.0)
        print(f"    {label:<26s}{value:10.3f}{limit:10.3f}{margin*100:+9.1f} %  {note}")

    _nG_GW   = Ip / (np.pi * config.a**2)                 # Greenwald density Ip/(pi a^2)
    _dl_mod  = config.density_limit_model
    _dl_fac  = config.Greenwald_limit
    _Ip_lim  = (float(Ip_lim) if isinstance(Ip_lim, (int, float, np.integer, np.floating))
                else None)
    print(f"\n {'Operational limits':<29s}{'value':>10s}{'limit':>10s}{'margin':>11s}")
    _lim_row("n_line / n_limit [1e20]", nbar_line, nG,
             note=f"({_dl_mod} x {_dl_fac:g})")
    _lim_row("n_line / n_G     [1e20]", nbar_line, _nG_GW,
             note="(Greenwald, Ip/pi a^2)")
    _lim_row("betaN / betaN_limit", betaN_total, betaN_lim,
             note="(thermal + fast alpha)")
    _lim_row(f"{_q_label} / q_limit", q_kink, q_lim, upper=False)
    _lim_row("Ip / Ip_limit    [MA]", Ip, _Ip_lim,
             note="" if _Ip_lim is not None else "(no ceiling)")
    _Ip_soft = _opt_float(S['Ip_soft_start'])
    _Ip_k    = _opt_float(S['Ip_soft_penalty'])
    _Ip_mult = compute_Ip_soft_multiplier(Ip, _Ip_soft, _Ip_lim, _Ip_k)
    if _Ip_soft is not None and _Ip_k:
        print(f"    {'Ip soft zone     [MA]':<26s}{Ip:10.3f}{_Ip_soft:10.3f}"
              f"  fitness surcharge {(_Ip_mult - 1.0) * 100:+.1f} %"
              f"  (+{_Ip_k * 100:g} % at Ip_limit)")

    # Verdict of the RUN rule on the final design
    if _fz['feasible']:
        print("\n FEASIBLE design (RUN rule: plasma limits and radial build).")
    elif not _fz['build_ok']:
        print(f"\n WARNING: best design is NOT FEASIBLE, the radial build does "
              f"not close ({_fz['build_reason']}).")
    else:
        _viol = {k: v for k, v in _fz['ratios'].items() if not v <= 1.0}
        print("\n WARNING: no feasible design found. The best design is the "
              "least violating one:")
        for name, ratio in _viol.items():
            print(f"      {name}: ratio {ratio:.4f} (limit 1)")
    
    # === Save outputs ===
    
    # 1. Convergence plot
    if save_figures:
        plot_path = os.path.join(current_run_directory, "convergence.png")
        plot_convergence(logbook, plot_path)
    
    # 2. Use D0FUS save_run_output for complete design documentation
    save_run_output(config, final_output, current_run_directory, None)

    # 2b. Optional: 2D scan + animated GIF of the GA trajectory
    gif_path = None
    scan_data = None
    gif_var_x, gif_var_y = S['gif_var_x'], S['gif_var_y']
    if S['make_gif'] and not save_figures:
        print("\n  [gif] Skipped: figures are disabled.")
    elif S['make_gif']:
        try:
            if gif_var_x not in param_keys or gif_var_y not in param_keys:
                print(f"\n  [gif] WARNING: gif_var_x={gif_var_x!r} or "
                      f"gif_var_y={gif_var_y!r} not in optimised parameters "
                      f"({param_keys}). Skipping GIF generation.")
            elif not hof_history:
                print("\n  [gif] WARNING: no HoF history recorded. Skipping GIF.")
            else:
                XX, YY, Z = scan_2d_for_animation(
                    best_individual=list(best),
                    opt_ranges=opt_ranges, param_keys=param_keys,
                    var_x=gif_var_x, var_y=gif_var_y,
                    n_x=S['gif_resolution'], n_y=S['gif_resolution'],
                    n_workers=S['n_workers'], verbose=verbose,
                )
                scan_data = (XX, YY, Z)
                gif_path = os.path.join(
                    current_run_directory,
                    f"trajectory_{gif_var_x}_{gif_var_y}.gif")
                animate_ga_trajectory(
                    XX, YY, Z, hof_history, gif_path,
                    var_x=gif_var_x, var_y=gif_var_y,
                    log_scale=S['gif_log_scale'], fps=S['gif_fps'],
                    ga_cloud=ga_cloud,
                    verbose=verbose,
                )
                # Save the raw scan + cloud as .npz for re-plotting later.
                # The cloud is stored as parallel arrays so it is trivial
                # to re-load with np.load without any JSON parsing.
                npz_path = os.path.join(
                    current_run_directory,
                    f"scan_{gif_var_x}_{gif_var_y}.npz")
                _cloud_gens = np.array([c.get('gen', 0) for c in ga_cloud], dtype=int)
                _cloud_xs   = np.array([c['params'][gif_var_x] for c in ga_cloud], dtype=float)
                _cloud_ys   = np.array([c['params'][gif_var_y] for c in ga_cloud], dtype=float)
                _cloud_fs   = np.array([c['fitness']            for c in ga_cloud], dtype=float)
                np.savez(npz_path,
                         XX=XX, YY=YY, Z=Z,
                         var_x=gif_var_x, var_y=gif_var_y,
                         cloud_gens=_cloud_gens,
                         cloud_x=_cloud_xs, cloud_y=_cloud_ys,
                         cloud_fitness=_cloud_fs)
                if verbose:
                    print(f" Scan + cloud data saved: {npz_path}  "
                          f"({len(ga_cloud)} cloud points)")
        except Exception as _gif_exc:
            print(f"\n  [gif] Animation failed: "
                  f"{type(_gif_exc).__name__}: {_gif_exc}")
            traceback.print_exc()

    # 3. Save optimization summary JSON (enriched with refinement & trajectory)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": os.path.abspath(input_file),
        "seed": S['seed'],
        "fitness_objective": objective,
        "C_invest_max_MEUR": to_serializable(config.C_invest_max),
        "settings": {k: to_serializable(v) for k, v in S.items()},
        "settings_source": dict(sources),
        "local_refine_kwargs": {k: to_serializable(v)
                                for k, v in refine_kwargs.items()},
        "optimized_parameters": {k: to_serializable(v) for k, v in best_params.items()},
        "best_fitness": to_serializable(best.fitness.values[0]),
        "cost_metrics": {
            "volume_proxy_m3_per_MW": to_serializable(cost),
            "COE_EUR_per_MWh": to_serializable(_COE_best),
            "C_invest_BEUR": to_serializable(_C_invest_best),
            "P_elec_MW": to_serializable(P_elec),
        },
        # Verdict of the RUN rule (evaluate_feasibility) on the final design
        "feasible": bool(_fz['feasible']),
        "is_buildable": bool(_fz['build_ok']),
        "binding_constraint": _fz['binding'],
        "feasibility_ratios": {k: to_serializable(v) for k, v in _fz['ratios'].items()},
        "stability_margins": {
            "n_line_over_nG": to_serializable(nbar_line / nG),
            "q_kink_over_qlim": to_serializable(q_kink / q_lim),
            "kink_parameter": _kink_param,
            "betaN_over_betaN_limit": to_serializable(betaN_total / betaN_lim) if betaN_lim > 0 else None,
            "Ip_over_Ip_limit": to_serializable(Ip / _Ip_lim) if _Ip_lim else None,
            "Ip_soft_multiplier": to_serializable(_Ip_mult),
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
        ],
        # ── Refinement log: every refine_topk call (memetic + final) ────
        "refinement_log": [
            {
                "phase":         entry.get('phase'),
                "gen":           entry.get('gen'),
                "starting_rank": entry.get('starting_rank'),
                "method":        entry.get('method'),
                "iters":         to_serializable(entry.get('iters')),
                "nfev":          to_serializable(entry.get('nfev')),
                "success":       to_serializable(entry.get('success')),
                "reason":        entry.get('reason'),
                "f_before":      to_serializable(entry.get('f_before')),
                "f_after":       to_serializable(entry.get('f_after')),
                "improvement":   to_serializable(entry.get('improvement')),
                "params":        {k: to_serializable(v)
                                  for k, v in (entry.get('params') or {}).items()},
            }
            for entry in refinement_log
        ],
        # ── GA trajectory: cumulative-best per generation ───────────────
        "ga_trajectory": [
            {
                "gen":     entry.get('gen'),
                "fitness": to_serializable(entry.get('fitness')),
                "params":  {k: to_serializable(v)
                            for k, v in (entry.get('params') or {}).items()},
                **({'phase': entry['phase']} if 'phase' in entry else {}),
            }
            for entry in hof_history
        ],
        # ── Exploration cloud: aggregate stats per generation ────────────
        # The full cloud is saved separately in scan_{var_x}_{var_y}.npz to
        # keep the JSON tractable. Here we only record per-generation
        # summary statistics (count and quantiles of the fitness).
        "ga_exploration_summary": _summarize_cloud_per_gen(ga_cloud),
    }
    
    summary_path = os.path.join(current_run_directory, "optimization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
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
    """
    Command line interface. An option given here overrides the deck. A
    setting given neither here nor in the deck takes the GA_SETTINGS
    default.
    """
    if len(sys.argv) < 2:
        print("Usage: python D0FUS_genetic.py <input_file> [options]")
        print("\nOptions (override the deck):")
        print("  --pop N           Population size (default: 200)")
        print("  --gen N           Max generations (default: 50)")
        print("  --mut F           Mutation rate (default: 0.3)")
        print("  --patience N      Generations without improvement before stopping (default: 20)")
        print("  --workers N       Parallel worker processes (default: 1)")
        print("  --seed N          Random seed (default: drawn and recorded)")
        print("  --refine METHOD   Local refinement: 'gradient', 'lbfgsb', "
              "'nelder-mead', or 'none' (default: none)")
        print("  --refine-topk N   Refine N diverse starters at the end (default: 3)")
        print("  --refine-every N  Memetic period in generations (0 = off, default: 0)")
        print("  --gif             Generate the trajectory GIF after the GA")
        print("  --gif-vars X,Y    Parameter axes for the GIF (default: a,R0)")
        print("  --gif-res N       Scan resolution per axis (default: 30)")
        print("  --gif-fps N       Animation frame rate (default: 8)")
        print("  --gif-linear      Use linear cost scale (default: log10)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Options left at None take the deck value, or else the default
    opts = {}
    flags = {'--pop': ('population_size', int), '--gen': ('generations', int),
             '--mut': ('mutation_rate', float), '--patience': ('patience', int),
             '--workers': ('n_workers', int), '--seed': ('seed', int),
             '--refine': ('local_refine_method', str),
             '--refine-topk': ('local_refine_top_k', int),
             '--refine-every': ('local_refine_every', int),
             '--gif-res': ('gif_resolution', int), '--gif-fps': ('gif_fps', int)}
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in flags and i + 1 < len(sys.argv):
            name, kind = flags[arg]
            opts[name] = kind(sys.argv[i + 1]); i += 2
        elif arg == "--gif":
            opts['make_gif'] = True; i += 1
        elif arg == "--gif-linear":
            opts['gif_log_scale'] = False; i += 1
        elif arg == "--gif-vars" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split(',')
            if len(parts) == 2:
                opts['gif_var_x'], opts['gif_var_y'] = (p.strip() for p in parts)
            i += 2
        else:
            print(f"  [warn] unknown option ignored: {arg}")
            i += 1
    
    run_genetic_optimization(input_file, verbose=True, **opts)


if __name__ == "__main__":
    main()