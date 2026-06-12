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
                   to the step discontinuities introduced by the
                   stability penalties.

Constraints
-----------
Stability and engineering constraints are enforced multiplicatively on
the fitness through compute_stability_penalty and check_radial_build:
  - Density limit (model-selectable)    nbar_line < nG
    where nG is the effective cap of config.density_limit_model
    ('greenwald' | 'giacomin' | 'zanca') scaled by Greenwald_limit
  - Normalised beta limit               betaT     < betaN
  - Kink safety factor                  q_kink    > q_limit  (q* or q95)
  - Sheffield capital-cost ceiling      C_invest  < C_invest_max
  - Positive innermost radial build     r_d       > 0
  - Finite TF / CS thicknesses          c_TF, d_CS > 1 mm
The step + quadratic penalty form keeps marginal violations expensive
while bounding penalty growth so the GA can still extract useful
genetic material from severely violated designs.

Fitness objectives
------------------
Selected via 'fitness_objective' in the input file:
  - 'COE'       cost of electricity (Sheffield 2016, EUR/MWh) under a
                hard C_invest_max ceiling. Default.
  - 'volume'    machine-volume proxy (V_BB + V_TF + V_CS) / P_fus.
  - 'C_invest'  total capital cost (Sheffield 2016, M EUR).
  - 'P_elec'    net electric power (maximised).
  - 'R0'        major radius (minimised), i.e. the most compact
                machine satisfying every constraint. R0 must be a
                GA-optimised parameter for this to be meaningful.

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
# module's direct dependencies inside the D0FUS source tree).
from D0FUS_BIB.D0FUS_parameterization import GlobalConfig, DEFAULT_CONFIG, coerce_input_value
from D0FUS_BIB.D0FUS_physical_functions import f_volume
from D0FUS_BIB.D0FUS_cost_functions import f_costs_Sheffield
from D0FUS_BIB.D0FUS_cost_data import *
from D0FUS_EXE.D0FUS_run import run, save_run_output

# Backwards-compatible alias: some legacy parts of the file may still use
# the `dc_replace` name (and the docstring references it).
dc_replace = replace

# ── Fitness objective ────────────────────────────────────────────────────────
# Selectable via 'fitness_objective = ...' in the input file.
#
#   'COE'      : minimise cost of electricity (Sheffield 2016)     [EUR/MWh]
#                 with C_invest <= C_invest_max budget constraint   (DEFAULT)
#   'volume'   : minimise (V_BB + V_TF + V_CS) / P_fus  [m^3/MW]
#   'C_invest' : minimise total capital cost (Sheffield 2016)      [M EUR]
#   'P_elec'   : maximise net electric power (minimise -P_elec)    [MW]
#   'R0'       : minimise the major radius (most compact machine)  [m]
#                 R0 must be declared as a GA-optimised parameter
#                 (e.g. 'R0 = [4.0, 9.0]') for this objective to be
#                 meaningful; otherwise the fitness is a constant.
#
VALID_FITNESS_OBJECTIVES = ('COE', 'volume', 'C_invest', 'P_elec', 'R0')

# =============================================================================
# run() return-tuple index map  (v3 — 93 outputs)
#
# This dictionary is the SINGLE SOURCE OF TRUTH for mapping symbolic names
# to positional indices in the tuple returned by run().  Both
# evaluate_individual() and debug_single_run() reference it so that a
# change in run()'s return ordering only requires an update here.
#
# Indices 0–80  : physics outputs (unchanged from v2)
# Indices 81–92 : cable-level conductor fractions (TF then CS)
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
    'cost':       27,   # Machine volume proxy [m³]
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
# Feasibility-first (epsilon-relaxed Deb) ranking parameters.
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
#             [-]. A design is pushed into the rejected band only once it
#             exceeds a limit by MORE than this margin, so it is not rejected
#             for an infinitesimal overshoot. Set to 0.0 for strict Deb.
#   FLOOR   : fitness threshold (in objective units) at which the infeasible
#             band starts. It must sit above the largest objective value any
#             feasible design can take and below PENALTY_VALUE. The default,
#             0.1 * PENALTY_VALUE, clears the COE, C_invest, R0 and volume
#             objectives with a wide margin while staying below the sentinel.
#   GAIN    : slope of the violation gradient inside the infeasible band [-].
# ---------------------------------------------------------------------------
DEFAULT_FEASIBILITY_EPSILON   = 0.02                 # tolerated aggregate violation [-]
DEFAULT_INFEASIBLE_FLOOR      = 0.1 * PENALTY_VALUE  # infeasible band threshold (obj. units)
DEFAULT_FEASIBILITY_BAND_GAIN = 1.0                  # in-band violation gradient slope [-]

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
    (``None``) untouched, so the caller's strengthened default is used. Any value
    that cannot be parsed as a real number degrades gracefully to ``None``.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_stability_penalty(nbar_line, nG, betaT, betaN, q_kink,
                               q_min=DEFAULT_CONFIG.q_limit,
                               betaN_limit=DEFAULT_CONFIG.betaN_limit,
                               Ip=None, Ip_limit=None,
                               penalty_step=2.0,
                               penalty_lin=10.0,
                               penalty_slope=20.0,
                               margin_width=0.05,
                               margin_amplitude=0.15,
                               # Ip-specific (strengthened) penalty knobs. Left None here;
                               # resolved to a strengthened multiple of the shared
                               # coefficients in the body, and overridable per input deck.
                               Ip_penalty_step=None,
                               Ip_penalty_lin=None,
                               Ip_penalty_slope=None,
                               Ip_margin_width=None,
                               Ip_margin_amplitude=None):
    """
    Step + quadratic penalty for stability violations.

    Plasma-current overshoot (Ip > Ip_limit) is penalised with its OWN,
    stronger coefficients (Ip_penalty_step / Ip_penalty_lin / Ip_penalty_slope,
    defaulting to about 2 to 2.5 times the shared coefficients), so the GA
    stops favouring slightly over-current designs and instead consolidates the
    rare Ip-feasible ones. The penalty keeps the same smooth, finite,
    gradient-preserving step+quadratic shape (it is never a hard kill), so
    near-boundary infeasible designs survive in the gene pool as bridges toward
    the feasible region. All five Ip knobs are overridable from the input deck
    (e.g. ``Ip_penalty_step = 6.0``); calibrate them against a given machine's
    cost-versus-current leverage with ``test_Ip_penalty.py``.

    Design philosophy
    -----------------
    Two requirements must be satisfied simultaneously:

    1. NO MARGINAL CHEATING — crossing a stability boundary must always
       cost more than any COE gain.  This is achieved by a *step*
       penalty at the boundary: fitness is immediately ×3 worse.

    2. BOUNDED GROWTH — even severely violated designs must stay below
       PENALTY_VALUE / 2 so the GA keeps them in the gene pool and can
       extract useful genetic material.  A quadratic (not exponential)
       growth guarantees this.

    3. GENTLE MARGIN PREFERENCE — the margin zone provides a *small*
       nudge toward designs with more headroom, but must NOT create a
       wall that prevents the GA from exploring near-limit designs.
       Good tokamak designs (ITER, DEMO) operate within 5–10% of
       stability limits — this is a feature, not a bug.

       margin_amplitude = 0.15 → at the boundary (0% margin), the
       fitness is multiplied by 1.15 (a 15% cost surcharge).
       At 5% margin, penalty → 0 (no surcharge).

    penalty_per_constraint (violation)  = penalty_step + penalty_lin × excess + penalty_slope × excess²
    penalty_per_constraint (margin zone) = margin_amplitude × (1 − margin)³

    Examples (per constraint, violation side):
      At boundary (ε → 0⁺):  2.0             → multiplier ≈ 3.0
      At 10% excess:         2.0 + 1.0 + 0.2 = 3.2 → multiplier ≈ 4.2

    Examples (per constraint, margin zone):
      At 0% margin (just feasible):  0.15     → multiplier ≈ 1.15
      At 2% margin:                  0.054    → multiplier ≈ 1.054
      At 5% margin:                  0.0      → multiplier = 1.0

    Parameters
    ----------
    nbar_line : float
        Line-averaged density [10²⁰ m⁻³].
    nG : float
        Greenwald density limit [10²⁰ m⁻³].
    betaT : float
        Toroidal beta (fraction; multiply by 100 for %).
    betaN : float
        Normalized beta limit [% m T / MA].
    q_kink : float
        Kink safety factor — either q* or q95 depending on
        config.kink_parameter (selected upstream by the caller).
    q_min : float
        Minimum safety factor threshold (must match the convention
        used for q_kink: ~2–2.5 for q*, ~3–3.5 for q95).
    penalty_step : float
        Discontinuity cost at the stability boundary.  A value of 2.0
        means the fitness is multiplied by ≥ 3 for any violation.
    penalty_lin : float
        Linear growth coefficient beyond the boundary.  Provides a non-zero
        slope AT the boundary so a barely-infeasible design is pulled back to
        the limit; must exceed ~3x the cost elasticity to remove the infeasible
        local minimum.
    penalty_slope : float
        Quadratic growth coefficient beyond the boundary.
    margin_width : float
        Width of the margin zone as a fraction of the limit (default 0.05 = 5%).
    margin_amplitude : float
        Maximum penalty in the margin zone (default 0.15 → ×1.15 at boundary).
    """
    violations = {}
    penalty_terms = []

    # Resolve the Ip-specific coefficients. When not provided explicitly they
    # fall back to a strengthened multiple of the shared coefficients, which
    # keeps a single tuning origin while making the current ceiling markedly
    # harder to cross than the density, beta and kink limits.
    Ip_penalty_step     = (2.0 * penalty_step)  if Ip_penalty_step     is None else Ip_penalty_step
    Ip_penalty_lin      = (2.5 * penalty_lin)   if Ip_penalty_lin      is None else Ip_penalty_lin
    Ip_penalty_slope    = (2.5 * penalty_slope) if Ip_penalty_slope    is None else Ip_penalty_slope
    Ip_margin_width     = margin_width          if Ip_margin_width     is None else Ip_margin_width
    Ip_margin_amplitude = margin_amplitude      if Ip_margin_amplitude is None else Ip_margin_amplitude
    
    # Density limit: nbar_line < nG
    if nG > 0:
        density_ratio = nbar_line / nG
        if density_ratio > 1.0:
            excess = density_ratio - 1.0
            penalty = penalty_step + penalty_lin * excess + penalty_slope * excess ** 2
            penalty_terms.append(penalty)
            violations['density'] = {
                'value': float(nbar_line), 'limit': float(nG),
                'ratio': float(density_ratio)
            }
        elif density_ratio > (1.0 - margin_width):
            margin_used = (density_ratio - (1.0 - margin_width)) / margin_width
            penalty_terms.append(margin_amplitude * margin_used ** 3)
    
    # Troyon beta limit: normalised beta_N must stay below betaN_limit
    if betaN_limit > 0:
        beta_ratio = betaN / betaN_limit
        if beta_ratio > 1.0:
            excess = beta_ratio - 1.0
            penalty = penalty_step + penalty_lin * excess + penalty_slope * excess ** 2
            penalty_terms.append(penalty)
            violations['beta'] = {
                'value': float(betaN), 'limit': float(betaN_limit),
                'ratio': float(beta_ratio)
            }
        elif beta_ratio > (1.0 - margin_width):
            margin_used = (beta_ratio - (1.0 - margin_width)) / margin_width
            penalty_terms.append(margin_amplitude * margin_used ** 3)
    
    # Plasma-current ceiling: Ip must stay below Ip_limit (disruption headroom).
    # Uses the STRONGER Ip-specific coefficients resolved above. The guard now
    # also accepts NumPy scalar types and rejects non-finite limits, so a limit
    # read as np.float64 / np.int64 is not silently treated as "no limit"
    # (which would disable the penalty entirely).
    if (Ip is not None
            and isinstance(Ip_limit, (int, float, np.integer, np.floating))
            and np.isfinite(Ip_limit) and Ip_limit > 0):
        ip_ratio = Ip / Ip_limit
        if ip_ratio > 1.0:
            excess = ip_ratio - 1.0
            penalty = (Ip_penalty_step
                       + Ip_penalty_lin * excess
                       + Ip_penalty_slope * excess ** 2)
            penalty_terms.append(penalty)
            violations['Ip'] = {
                'value': float(Ip), 'limit': float(Ip_limit),
                'ratio': float(ip_ratio)
            }
        elif ip_ratio > (1.0 - Ip_margin_width):
            margin_used = (ip_ratio - (1.0 - Ip_margin_width)) / Ip_margin_width
            penalty_terms.append(Ip_margin_amplitude * margin_used ** 3)
    
    # Kink safety factor: q_kink > q_min  (q_kink = q* or q95)
    if q_kink < q_min:
        deficit = (q_min - q_kink) / q_min
        penalty = penalty_step + penalty_lin * deficit + penalty_slope * deficit ** 2
        penalty_terms.append(penalty)
        violations['q_kink'] = {
            'value': float(q_kink), 'limit': float(q_min),
            'ratio': float(q_kink / q_min)
        }
    elif q_kink < q_min * (1.0 + margin_width):
        margin = (q_kink - q_min) / (q_min * margin_width)
        penalty_terms.append(margin_amplitude * (1 - margin) ** 3)
    
    total_penalty = 1.0 + sum(penalty_terms)
    is_stable = len(violations) == 0
    
    return is_stable, total_penalty, violations


def check_radial_build(cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line):
    """
    Check radial build validity for a candidate design.

    Parameters
    ----------
    cost      : float  Machine cost proxy [m³].
    r_d       : float  Innermost radial build radius R0 - a - b - c - d [m].
    c_TF      : float  TF inboard radial thickness [m].
    d_CS      : float  CS radial thickness [m].
    q_kink    : float  Kink safety factor (q* or q95, see kink_parameter).
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
                      ('d_CS', d_CS), ('q_kink', q_kink), ('betaT', betaT),
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


def _total_relative_violation(violations, weights=None):
    """
    Aggregate constraint violations into a single dimensionless measure.

    Each violated constraint contributes the magnitude of its relative
    departure from the limit, ``|ratio - 1|``. For the upper limits (density,
    beta, Ip) ``ratio`` is value / limit, so an overshoot gives ratio > 1; for
    the lower q limit ``ratio`` is q_kink / q_min, so a deficit gives
    ratio < 1. Taking the absolute value makes every violation contribute
    positively. The contributions are summed (optionally weighted per
    constraint), yielding a scalar that is exactly 0 for a strictly feasible
    design and grows with the severity of the violation.

    Parameters
    ----------
    violations : dict
        The mapping returned by ``compute_stability_penalty``. Keys are
        constraint names ('density', 'beta', 'Ip', 'q_kink'); each value holds
        at least a 'ratio' entry.
    weights : dict or None
        Optional per-constraint weights (default 1.0 each). Increasing a weight
        makes that constraint's violation count more heavily, i.e. the search
        is pushed to cure it first. A weight of 0.0 removes that constraint
        from the feasibility measure entirely (use with care).

    Returns
    -------
    float
        Total (weighted) relative violation, >= 0.0.
    """
    if not violations:
        return 0.0
    total = 0.0
    for name, info in violations.items():
        try:
            ratio = float(info.get('ratio', 1.0))
        except (TypeError, ValueError, AttributeError):
            ratio = 1.0
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        total += w * abs(ratio - 1.0)
    return total


def compute_feasibility_fitness(raw_fitness, total_violation, objective,
                                budget_multiplier=1.0,
                                epsilon=DEFAULT_FEASIBILITY_EPSILON,
                                infeasible_floor=DEFAULT_INFEASIBLE_FLOOR,
                                band_gain=DEFAULT_FEASIBILITY_BAND_GAIN):
    """
    Map (objective value, aggregate violation) to a single MINIMISED fitness
    enforcing epsilon-relaxed feasibility-first ranking.

    The DEAP fitness is minimised (weights = (-1.0,)), so 'smaller is better'.
    Two bands are produced.

    Acceptable band (total_violation <= epsilon)
        ``fitness = raw objective`` times the soft budget multiplier. The
        design is ranked purely on objective merit. This band contains strictly
        feasible designs (violation 0) and designs that exceed a limit by no
        more than epsilon, so a design is not rejected for an infinitesimal
        overshoot (epsilon is the breathing room; epsilon = 0 recovers strict
        Deb). A clamp keeps the value strictly below ``infeasible_floor`` so an
        acceptable design can never enter the infeasible band, even if
        ``infeasible_floor`` were mis-set below the objective scale.

    Infeasible band (total_violation > epsilon)
        ``fitness = infeasible_floor * (1 + band_gain * (total_violation - epsilon))``.
        Every value is >= ``infeasible_floor`` and therefore strictly worse
        than any acceptable design. Within the band the value increases
        monotonically with the violation (Deb's third rule), so the search
        still separates a 2 % overshoot from a 40 % one and keeps climbing
        toward the boundary; this is a slope, not a wall. The value is capped
        just below PENALTY_VALUE so structurally invalid designs (returned as
        PENALTY_VALUE elsewhere) remain the worst of all.

    For the maximisation objective 'P_elec' (stored as raw_fitness = -P_elec,
    a negative number) the acceptable-band value divides by the budget
    multiplier and needs no clamp, a negative value being always below the
    positive floor.

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
        if objective == 'P_elec':
            return raw_fitness / budget_multiplier
        fitness = raw_fitness * budget_multiplier
        # An acceptable design must never reach the infeasible band.
        return min(fitness, infeasible_floor * (1.0 - 1e-9))

    # Infeasible band: strictly worse, ordered by violation, bounded below the
    # hard PENALTY_VALUE sentinel so structurally invalid designs stay worst.
    overshoot = total_violation - epsilon
    fitness = infeasible_floor * (1.0 + band_gain * overshoot)
    return min(fitness, PENALTY_VALUE * (1.0 - 1e-9))


def evaluate_individual(individual, verbose=False):
    """
    Evaluate the fitness of a single individual.

    Uses the centralised ``_IDX`` map for all tuple indexing so that any
    change in run()'s return ordering is automatically propagated.

    The fitness objective is read from ``static_inputs['fitness_objective']``
    (default: 'volume').  See VALID_FITNESS_OBJECTIVES for options.
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

        # ── Guard against any residual non-finite output ─────────────────
        cost      = _safe_real(output[_IDX['cost']])
        nbar_line = _safe_real(output[_IDX['nbar_line']])
        nG        = _safe_real(output[_IDX['nG']])
        betaN     = _safe_real(output[_IDX['betaN']])
        betaT     = _safe_real(output[_IDX['betaT']])
        qstar     = _safe_real(output[_IDX['qstar']])
        q95       = _safe_real(output[_IDX['q95']])
        R0_abcd   = _safe_real(output[_IDX['r_d']])
        c_TF      = _safe_real(output[_IDX['c']])
        d_CS      = _safe_real(output[_IDX['d']])
        Ip        = _safe_real(output[_IDX['Ip']])

        # Select kink parameter (q* or q95) for stability checks.
        _kink_param = static_inputs.get('kink_parameter', DEFAULT_CONFIG.kink_parameter)
        q_kink = q95 if _kink_param == 'q95' else qstar

        # ── Radial build validity check ──────────────────────────────────
        is_valid, reason = check_radial_build(
            cost, R0_abcd, c_TF, d_CS, q_kink, betaT, nbar_line)
        if not is_valid:
            if verbose:
                print(f"  [PENALTY] radial build: {reason}")
                print(f"    cost={cost:.4g}  r_d={R0_abcd:.4g}  "
                      f"c_TF={c_TF:.4g}  d_CS={d_CS:.4g}")
            return (PENALTY_VALUE,)

        # ── Stability penalty ────────────────────────────────────────────
        q_lim = static_inputs.get('q_limit', DEFAULT_CONFIG.q_limit)
        is_stable, penalty_multiplier, violations = compute_stability_penalty(
            nbar_line, nG, betaT, betaN, q_kink,
            q_min=q_lim,
            betaN_limit=static_inputs.get('betaN_limit', DEFAULT_CONFIG.betaN_limit),
            Ip=Ip, Ip_limit=static_inputs.get('Ip_limit', DEFAULT_CONFIG.Ip_limit),
            # Optional per-deck overrides of the strengthened Ip penalty. Unset
            # values arrive as None, so compute_stability_penalty applies its own
            # strengthened Ip defaults. Calibrate with test_Ip_penalty.py.
            Ip_penalty_step=_opt_float(static_inputs.get('Ip_penalty_step')),
            Ip_penalty_lin=_opt_float(static_inputs.get('Ip_penalty_lin')),
            Ip_penalty_slope=_opt_float(static_inputs.get('Ip_penalty_slope')),
            Ip_margin_width=_opt_float(static_inputs.get('Ip_margin_width')),
            Ip_margin_amplitude=_opt_float(static_inputs.get('Ip_margin_amplitude')),
        )

        # ── Compute fitness value based on selected objective ────────────
        objective = static_inputs.get('fitness_objective', 'COE')

        # Capital cost (Sheffield), computed for COE / C_invest / P_elec
        # objectives.  Used both as a fitness metric and for the budget
        # ceiling constraint.  Remains NaN for the 'volume' legacy proxy.
        _C_inv = np.nan

        if objective == 'volume':
            # Legacy cost proxy: (V_BB + V_TF + V_CS) / P_fus [m^3/MW]
            raw_fitness = cost

        elif objective == 'R0':
            # Compact-machine objective: minimise the major radius [m].
            # R0 is a design variable (config.R0), so the GA drives the
            # design toward the smallest major radius that still satisfies
            # every physics and engineering constraint. The stability and
            # radial-build penalties below inflate R0 for infeasible
            # designs, keeping the search inside the feasible region.
            # The C_invest_max ceiling is intentionally left inactive
            # here (as for 'volume'), since _C_inv stays NaN.
            raw_fitness = _safe_real(config.R0)

        elif objective in ('COE', 'C_invest', 'P_elec'):
            # Sheffield (2016) cost model — needs derived quantities
            P_CD    = _safe_real(output[_IDX['P_CD']])
            P_elec  = _safe_real(output[_IDX['P_elec']])
            Gamma_n = _safe_real(output[_IDX['Gamma_n']])
            Surface = _safe_real(output[_IDX['Surface']])
            κ       = _safe_real(output[_IDX['kappa']])

            if any(np.isnan(v) for v in [P_CD, P_elec, Gamma_n, Surface, κ]):
                if verbose:
                    print(f"  [PENALTY] NaN in cost inputs: P_CD={P_CD:.4g} "
                          f"P_elec={P_elec:.4g} Gamma_n={Gamma_n:.4g} "
                          f"Surface={Surface:.4g} kappa={κ:.4g}")
                return (PENALTY_VALUE,)

            P_th = config.P_fus * config.M_blanket + P_CD
            (V_BB, V_TF, V_CS, V_FI) = f_volume(
                config.a, config.b, c_TF, d_CS, config.R0, κ)

            _cres = f_costs_Sheffield(
                discount_rate=config.discount_rate,
                contingency=config.contingency,
                T_life=config.T_life,
                T_build=config.T_build,
                P_t=P_th,
                P_e=max(P_elec, 1.0),
                P_aux=P_CD,
                Gamma_n=Gamma_n,
                Util_factor=config.Util_factor,
                Dwell_factor=config.Dwell_factor,
                dt_rep=config.dt_rep,
                V_FI=V_FI,
                V_pc=V_TF + V_CS,
                V_sg=V_BB,
                V_bl=V_BB,
                S_tt=0.1 * Surface,
                Supra_cost_factor=config.Supra_cost_factor,
            )
            _COE   = _safe_real(_cres[3])   # COE [EUR/MWh]
            _C_inv = _safe_real(_cres[2])    # C_invest [M EUR]

            if objective == 'COE':
                raw_fitness = _COE
            elif objective == 'C_invest':
                raw_fitness = _C_inv
            else:  # P_elec
                raw_fitness = -P_elec if np.isfinite(P_elec) else PENALTY_VALUE

        else:
            # Fallback to volume proxy
            raw_fitness = cost

        if not np.isfinite(raw_fitness) or (raw_fitness <= 0 and objective != 'P_elec'):
            return (PENALTY_VALUE,)

        # ── Capital cost ceiling: soft quadratic penalty ─────────────────
        # Applied as a multiplicative factor (like the stability penalty)
        # so it scales naturally regardless of the fitness objective.
        #
        # penalty_budget = 1 + k * (excess)^2      k = 3.0
        #   excess = 0.04 (26 vs 25 B) → ×1.005   (barely noticeable)
        #   excess = 0.5                → ×1.75    (moderate pressure)
        #   excess = 1.0               → ×4.0     (strong, but survives)
        #
        # Skipped when C_invest is not available (volume objective)
        # or when the user directly minimises C_invest (redundant).
        budget_multiplier = 1.0
        C_max = config.C_invest_max
        if (objective != 'C_invest'
                and np.isfinite(_C_inv) and np.isfinite(C_max)
                and _C_inv > C_max):
            excess = (_C_inv - C_max) / C_max
            budget_multiplier = 1.0 + 3.0 * excess ** 2

        # ── Feasibility-first (epsilon-relaxed Deb) fitness ──────────────
        # The multiplicative stability penalty above is superseded here for
        # RANKING: it is kept only to produce the `violations` map (and for
        # backward-compatible diagnostics). Ranking now uses the banded scheme
        # of compute_feasibility_fitness, so any design within the epsilon
        # tolerance outranks any design outside it, while infeasible designs
        # stay ordered by total relative violation (a smooth gradient, not a
        # flat wall, so the rare near-boundary designs survive).
        #
        # NOTE: because ranking is banded, the Ip_penalty_step/lin/slope knobs
        # no longer influence selection (they only shaped the now-superseded
        # multiplier). To make a given limit count more heavily, weight its
        # violation instead, e.g. `Ip_violation_weight = 2.0` in the deck.
        def _wt(key):
            v = _opt_float(static_inputs.get(key))
            return 1.0 if v is None else v

        _vw = {
            'density': _wt('density_violation_weight'),
            'beta':    _wt('beta_violation_weight'),
            'Ip':      _wt('Ip_violation_weight'),
            'q_kink':  _wt('q_kink_violation_weight'),
        }
        total_violation = _total_relative_violation(violations, weights=_vw)

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
                        # Remove from static_inputs if a scalar line was
                        # parsed earlier for the same key — prevents the GA
                        # value from being silently overwritten at merge time.
                        static_inputs.pop(key, None)
                        print(f"  {key} = [{min_val}, {max_val}] (optimize)")
                    except ValueError:
                        pass
            else:
                # Skip scalar lines for keys already declared as GA ranges
                # (e.g. a user might have 'a = 3.0' before 'a = [2, 4]').
                if key in opt_ranges:
                    continue
                # Type-aware coercion shared with the run / scan loaders, so
                # a boolean such as "False" is not silently kept as the
                # truthy string "False", and optional fields decode "None"
                # to the Python None.
                static_inputs[key] = coerce_input_value(key, value)

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
                    # EXPLOITATION: Polynomial mutation with adaptive eta
                    eta = 5 + 25 * progress  # Start wide (5), cap at 30
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
    toolbox.register("select", tools.selTournament, tournsize=2)


def inject_diverse_individuals(pop, n_inject, opt_ranges, param_keys):
    """Inject random individuals to maintain diversity (UNCONDITIONAL).

    Previous version only replaced the worst individual if the random
    newcomer was better — once the population converges, this condition
    is almost never met, making injection a no-op.

    Now: always replace the n_inject worst individuals, regardless of
    the newcomer's fitness.  This forces the GA to re-explore even if
    the new point is initially poor.

    Returns
    -------
    pop : list
        Same object as input, modified in place.
    injected : list of creator.Individual
        The newly evaluated individuals, each with its `fitness.values`
        already populated. The caller may use this list to log the
        evaluation events (e.g. for the GA exploration cloud).
    """
    injected = []
    for _ in range(n_inject):
        new_ind = creator.Individual([
            random.uniform(opt_ranges[k][0], opt_ranges[k][1])
            for k in param_keys
        ])
        new_ind.fitness.values = toolbox.evaluate(new_ind)
        injected.append(new_ind)

        # Find worst individual (only among those with valid fitness)
        valid_indices = [i for i in range(len(pop)) 
                        if pop[i].fitness.valid and len(pop[i].fitness.values) > 0]
        
        if valid_indices:
            worst_idx = max(valid_indices, key=lambda i: pop[i].fitness.values[0])
            pop[worst_idx] = new_ind
        else:
            pop[0] = new_ind
    
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
    n_workers : int
        Number of parallel worker processes for fitness evaluation.
        1 (default) -> sequential map (original behaviour).
        N > 1       -> multiprocessing.Pool with N workers.
        Works on Linux (fork) and macOS/Windows (spawn) alike, because
        _pool_initializer re-populates the required globals in each worker.
    local_refine_method : {None, 'gradient', 'lbfgsb'}
        If not None, apply local refinement (see refine_individual) to the
        top-k Hall-of-Fame members at the end of the GA. 'gradient' is the
        hand-rolled projected gradient descent; 'lbfgsb' uses scipy's
        L-BFGS-B. None disables local refinement.
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

    # Initial evaluation
    current_generation = 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        _append_to_cloud(ind, 0)

    # ── Diagnostic: abort early if entire initial population is invalid ───
    n_init_valid = sum(1 for f in fitnesses if f[0] < PENALTY_VALUE / 2)
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
    stagnation_counter = 0

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
        
        # Evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
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
        
        # Track stagnation
        current_best = record['min']
        best_fitness_history.append(current_best)
        
        if len(best_fitness_history) > 1:
            if abs(best_fitness_history[-1] - best_fitness_history[-2]) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
        
        # Inject diversity if stagnating or periodically.
        # The injection fraction is controlled by `diversity_inject_fraction`
        # (default 0.30 = 30% of pop). The previous fixed 10% was too small:
        # once the population had collapsed to one basin, 10% of fresh blood
        # could not overpower the basin's gravity.
        if stagnation_counter >= 3 or gen % 7 == 0:
            n_inject = max(5, int(pop_size * diversity_inject_fraction))
            pop, _injected = inject_diverse_individuals(
                pop, n_inject, opt_ranges, param_keys)
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
            _periodic_info = refine_topk(
                hof, k=1,
                method=local_refine_method,
                pool=_pool,
                verbose=verbose,
                **(local_refine_kwargs or {})
            )
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

        # Early stopping
        if stagnation_counter >= patience:
            print(f"\n Converged at generation {gen}")
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

        _final_info = refine_topk(
            refine_hof, k=n_eff,
            method=local_refine_method,
            pool=_pool,
            verbose=verbose,
            **(local_refine_kwargs or {})
        )
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
# The step penalties in compute_stability_penalty() introduce non-smoothness
# at the constraint boundaries. The local search assumes the starting point
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
    fallback when the step penalties in compute_stability_penalty create
    discontinuities that throw off gradient-based methods.

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
        Local optimiser. If a scipy-based method is requested but scipy is
        unavailable, falls back to 'gradient'.
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
    """Single convergence plot in log scale"""
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

def run_genetic_optimization(input_file, 
                             population_size=200,
                             generations=50,
                             crossover_rate=0.7,
                             mutation_rate=0.3,
                             patience=20,
                             seed=None,
                             verbose=True,
                             n_workers=1,
                             local_refine_method=None,
                             local_refine_top_k=3,
                             local_refine_every=0,
                             local_refine_kwargs=None,
                             diversity_inject_fraction=0.30,
                             cloud_refinement_min_dist=0.05,
                             make_gif=False,
                             gif_var_x='a',
                             gif_var_y='R0',
                             gif_resolution=30,
                             gif_log_scale=True,
                             gif_fps=8):
    """
    Main optimization function.

    Local refinement options
    ------------------------
    local_refine_method : {None, 'gradient', 'lbfgsb', 'nelder-mead'}
        Activate memetic refinement at the end of the GA (and optionally
        periodically). 'gradient' = hand-rolled projected gradient descent;
        'lbfgsb' = scipy L-BFGS-B with bounds; 'nelder-mead' = derivative-free
        simplex (robust to step penalties). None disables the feature.
    local_refine_top_k : int
        Number of top HoF members to refine after the GA (default: 3).
    local_refine_every : int
        Period (in generations) for periodic memetic refinement. 0 disables.
    local_refine_kwargs : dict or None
        Forwarded to the underlying optimiser (max_iters, h, step_init,
        ftol, gtol, eps, ...).

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

    # Fitness objective: kept in static_inputs (read by evaluate_individual)
    # but validated here.  Default: 'COE' with C_invest_max budget constraint.
    _obj = static_inputs.get('fitness_objective', 'COE')
    if _obj not in VALID_FITNESS_OBJECTIVES:
        print(f"  [warn] Unknown fitness_objective '{_obj}' — defaulting to 'COE'")
        static_inputs['fitness_objective'] = 'COE'
        _obj = 'COE'
    print(f"  [input file] fitness_objective = {_obj}")
    if _obj == 'COE':
        _budget = static_inputs.get('C_invest_max', DEFAULT_CONFIG.C_invest_max)
        print(f"  [input file] C_invest_max      = {float(_budget)*1e-3:.1f} B EUR (budget ceiling)")
    if _obj == 'R0' and 'R0' not in param_keys:
        # Degenerate case: minimising a fixed major radius gives a constant
        # fitness, so the GA would only search the remaining design space
        # for any feasible point. Warn the user to bracket R0.
        print("  [warn] fitness_objective = 'R0' but R0 is not a GA "
              "parameter; declare it as a range (e.g. 'R0 = [4.0, 9.0]') "
              "to actually minimise the major radius.")

    if len(param_keys) < 1:
        print("\n ERROR: Need at least 1 optimization parameter")
        sys.exit(1)
    
    setup_toolbox(opt_ranges, param_keys, generations)
    
    # Run optimization
    # Override n_workers from input file if provided
    if 'n_workers' in static_inputs:
        n_workers = max(1, int(static_inputs.pop('n_workers')))
        print(f"  [input file] n_workers           = {n_workers}")

    # ── Local refinement options (override from input file if present) ───────
    # Recognised keys:
    #   local_refine_method  : 'gradient', 'lbfgsb', 'none'
    #   local_refine_top_k   : int
    #   local_refine_every   : int  (0 disables periodic memetic)
    #   local_refine_max_iters, local_refine_h, local_refine_step_init,
    #   local_refine_ftol, local_refine_gtol, local_refine_eps : optimiser knobs
    if 'local_refine_method' in static_inputs:
        m = str(static_inputs.pop('local_refine_method')).strip().lower()
        local_refine_method = None if m in ('none', '', 'off', 'disabled') else m
        print(f"  [input file] local_refine_method = {local_refine_method}")
    if 'local_refine_top_k' in static_inputs:
        local_refine_top_k = int(static_inputs.pop('local_refine_top_k'))
        print(f"  [input file] local_refine_top_k  = {local_refine_top_k}")
    if 'local_refine_every' in static_inputs:
        local_refine_every = int(static_inputs.pop('local_refine_every'))
        print(f"  [input file] local_refine_every  = {local_refine_every}")
    if 'diversity_inject_fraction' in static_inputs:
        diversity_inject_fraction = float(
            static_inputs.pop('diversity_inject_fraction'))
        print(f"  [input file] diversity_inject_fraction = {diversity_inject_fraction}")
    if 'cloud_refinement_min_dist' in static_inputs:
        cloud_refinement_min_dist = float(
            static_inputs.pop('cloud_refinement_min_dist'))
        print(f"  [input file] cloud_refinement_min_dist = {cloud_refinement_min_dist}")

    # Collect optimiser-specific knobs into local_refine_kwargs
    if local_refine_kwargs is None:
        local_refine_kwargs = {}
    _refine_knobs = ('max_iters', 'h', 'step_init', 'ftol', 'gtol', 'eps',
                     'tol_grad', 'tol_step', 'c1', 'max_backtrack',
                     'initial_simplex_scale')
    for knob in _refine_knobs:
        key = f'local_refine_{knob}'
        if key in static_inputs:
            local_refine_kwargs[knob] = float(static_inputs.pop(key))
            print(f"  [input file] {key:<25s} = {local_refine_kwargs[knob]}")
    if local_refine_method is not None:
        print(f"  [input file] local_refine_kwargs  = {local_refine_kwargs}")

    # ── GIF / visualization options ──────────────────────────────────────────
    if 'make_gif' in static_inputs:
        _v = static_inputs.pop('make_gif')
        make_gif = bool(_v) if isinstance(_v, bool) else (
            str(_v).strip().lower() in ('1', 'true', 'yes', 'on'))
        print(f"  [input file] make_gif            = {make_gif}")
    if 'gif_var_x' in static_inputs:
        gif_var_x = str(static_inputs.pop('gif_var_x')).strip()
        print(f"  [input file] gif_var_x           = {gif_var_x}")
    if 'gif_var_y' in static_inputs:
        gif_var_y = str(static_inputs.pop('gif_var_y')).strip()
        print(f"  [input file] gif_var_y           = {gif_var_y}")
    if 'gif_resolution' in static_inputs:
        gif_resolution = int(static_inputs.pop('gif_resolution'))
        print(f"  [input file] gif_resolution      = {gif_resolution}")
    if 'gif_log_scale' in static_inputs:
        _v = static_inputs.pop('gif_log_scale')
        gif_log_scale = bool(_v) if isinstance(_v, bool) else (
            str(_v).strip().lower() in ('1', 'true', 'yes', 'on'))
        print(f"  [input file] gif_log_scale       = {gif_log_scale}")
    if 'gif_fps' in static_inputs:
        gif_fps = int(static_inputs.pop('gif_fps'))
        print(f"  [input file] gif_fps             = {gif_fps}")

    best, logbook, hof, hof_history, refinement_log, ga_cloud = run_genetic_algorithm(
        population_size, generations, crossover_rate, mutation_rate,
        patience, verbose, n_workers=n_workers,
        local_refine_method=local_refine_method,
        local_refine_top_k=local_refine_top_k,
        local_refine_every=local_refine_every,
        local_refine_kwargs=local_refine_kwargs,
        diversity_inject_fraction=diversity_inject_fraction,
        cloud_refinement_min_dist=cloud_refinement_min_dist,
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
    q95_val   = final_output[_IDX['q95']]
    cost      = final_output[_IDX['cost']]
    Q         = final_output[_IDX['Q']]
    P_elec    = final_output[_IDX['P_elec']]
    c_TF      = final_output[_IDX['c']]
    d_CS      = final_output[_IDX['d']]
    r_d       = final_output[_IDX['r_d']]
    Ip        = final_output[_IDX['Ip']]

    # Select kink parameter for final reporting (consistent with GA evaluation)
    _kink_param = static_inputs.get('kink_parameter',
                                     DEFAULT_CONFIG.kink_parameter)
    q_kink = q95_val if _kink_param == 'q95' else qstar
    _q_label = 'q95' if _kink_param == 'q95' else 'q*'

    # Check stability (using line-averaged density for Greenwald comparison)
    q_lim = static_inputs.get('q_limit', DEFAULT_CONFIG.q_limit)
    betaN_lim = static_inputs.get('betaN_limit', DEFAULT_CONFIG.betaN_limit)
    Ip_lim = static_inputs.get('Ip_limit', DEFAULT_CONFIG.Ip_limit)
    is_stable, _, violations = compute_stability_penalty(
        nbar_line, nG, betaT, betaN, q_kink, q_min=q_lim, betaN_limit=betaN_lim,
        Ip=Ip, Ip_limit=Ip_lim)

    # Compute Sheffield COE for the best design (regardless of objective)
    _COE_best = np.nan
    _C_invest_best = np.nan
    try:
        P_CD_best    = final_output[_IDX['P_CD']]
        Gamma_n_best = final_output[_IDX['Gamma_n']]
        Surface_best = final_output[_IDX['Surface']]
        κ_best       = final_output[_IDX['kappa']]
        P_th_best    = config.P_fus * config.M_blanket + P_CD_best
        (V_BB_b, V_TF_b, V_CS_b, V_FI_b) = f_volume(
            config.a, config.b, c_TF, d_CS, config.R0, κ_best)
        _cres_best = f_costs_Sheffield(
            discount_rate=config.discount_rate, contingency=config.contingency,
            T_life=config.T_life, T_build=config.T_build,
            P_t=P_th_best, P_e=max(P_elec, 1.0), P_aux=P_CD_best,
            Gamma_n=Gamma_n_best,
            Util_factor=config.Util_factor, Dwell_factor=config.Dwell_factor,
            dt_rep=config.dt_rep,
            V_FI=V_FI_b, V_pc=V_TF_b+V_CS_b, V_sg=V_BB_b, V_bl=V_BB_b,
            S_tt=0.1*Surface_best,
            Supra_cost_factor=config.Supra_cost_factor)
        _COE_best     = _cres_best[3]
        _C_invest_best = _cres_best[2] * 1e-3  # M EUR -> B EUR
    except Exception:
        pass

    objective = static_inputs.get('fitness_objective', 'COE')
    print(f"\n Fitness objective: {objective}")
    if objective == 'COE':
        _budget = config.C_invest_max
        _within = _C_invest_best * 1e3 <= _budget if np.isfinite(_C_invest_best) else False
        _tag = "WITHIN BUDGET" if _within else "OVER BUDGET"
        print(f"    Budget ceiling:             {_budget*1e-3:.1f} B EUR  [{_tag}]")
    print(f"\n Best design metrics:")
    print(f"    R0 (major radius):          {config.R0:.4f} [m]")
    print(f"    Volume proxy (V/P_fus):     {cost:.4f} [m^3/MW]")
    print(f"    COE (Sheffield):            {_COE_best:.1f} [EUR/MWh]")
    print(f"    C_invest:                   {_C_invest_best:.2f} [B EUR]")
    print(f"    Q factor:                   {Q:.2f}")
    print(f"    Ip:                         {Ip:.2f} MA")
    print(f"    P_elec:                     {P_elec:.1f} MW")
    print(f"    n_line/nG: {nbar_line/nG:.3f} ({(1-nbar_line/nG)*100:+.1f}% margin)")
    print(f"    betaN/betaN_limit: {betaN/betaN_lim:.3f} ({(1-betaN/betaN_lim)*100:+.1f}% margin)")
    if Ip_lim is not None:
        print(f"    Ip/Ip_limit: {Ip/Ip_lim:.3f} ({(1-Ip/Ip_lim)*100:+.1f}% margin)")
    q_lim = static_inputs.get('q_limit', DEFAULT_CONFIG.q_limit)
    print(f"    {_q_label}/q_limit: {q_kink/q_lim:.3f} ({(q_kink/q_lim-1)*100:+.1f}% margin)")
    print(f"    c_TF: {c_TF:.3f} m   d_CS: {d_CS:.3f} m   r_d: {r_d:.3f} m")

    # Radial build sanity check on final design
    _rb_ok, _rb_reason = check_radial_build(
        cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line)
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

    # 2b. Optional: 2D scan + animated GIF of the GA trajectory
    gif_path = None
    scan_data = None
    if make_gif:
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
                    n_x=gif_resolution, n_y=gif_resolution,
                    n_workers=n_workers, verbose=verbose,
                )
                scan_data = (XX, YY, Z)
                gif_path = os.path.join(
                    current_run_directory,
                    f"trajectory_{gif_var_x}_{gif_var_y}.gif")
                animate_ga_trajectory(
                    XX, YY, Z, hof_history, gif_path,
                    var_x=gif_var_x, var_y=gif_var_y,
                    log_scale=gif_log_scale, fps=gif_fps,
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
        "seed": seed,
        "fitness_objective": objective,
        "C_invest_max_MEUR": to_serializable(config.C_invest_max),
        "settings": {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "local_refine_method": local_refine_method,
            "local_refine_top_k": local_refine_top_k,
            "local_refine_every": local_refine_every,
            "local_refine_kwargs": {k: to_serializable(v)
                                    for k, v in (local_refine_kwargs or {}).items()},
        },
        "optimized_parameters": {k: to_serializable(v) for k, v in best_params.items()},
        "best_fitness": to_serializable(best.fitness.values[0]),
        "cost_metrics": {
            "volume_proxy_m3_per_MW": to_serializable(cost),
            "COE_EUR_per_MWh": to_serializable(_COE_best),
            "C_invest_BEUR": to_serializable(_C_invest_best),
            "P_elec_MW": to_serializable(P_elec),
        },
        "is_stable": is_stable,
        "is_buildable": _rb_ok,
        "stability_margins": {
            "n_line_over_nG": to_serializable(nbar_line/nG),
            "q_kink_over_qlim": to_serializable(q_kink/q_lim),
            "kink_parameter": _kink_param,
            "betaN_over_betaN_limit": to_serializable(betaN/betaN_lim) if betaN_lim > 0 else None,
            "Ip_over_Ip_limit": to_serializable(Ip/Ip_lim) if Ip_lim is not None else None
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
        print("  --pop N           Population size (default: 200)")
        print("  --gen N           Max generations (default: 50)")
        print("  --mut F           Mutation rate (default: 0.3)")
        print("  --seed N          Random seed for reproducibility")
        print("  --refine METHOD   Local refinement: 'gradient', 'lbfgsb', "
              "'nelder-mead', or 'none' (default: none)")
        print("  --refine-topk N   Refine top-N HoF members at the end (default: 3)")
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
    
    # Parse arguments
    pop_size = 200
    n_gen = 50
    mut_rate = 0.3
    seed = None
    refine_method = None
    refine_topk_val = 3
    refine_every_val = 0
    make_gif_flag = False
    gif_vx, gif_vy = 'a', 'R0'
    gif_res = 30
    gif_fps_val = 8
    gif_log_flag = True
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--pop" and i + 1 < len(sys.argv):
            pop_size = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--gen" and i + 1 < len(sys.argv):
            n_gen = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--mut" and i + 1 < len(sys.argv):
            mut_rate = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--refine" and i + 1 < len(sys.argv):
            m = sys.argv[i + 1].strip().lower()
            refine_method = None if m in ('none', 'off') else m
            i += 2
        elif sys.argv[i] == "--refine-topk" and i + 1 < len(sys.argv):
            refine_topk_val = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--refine-every" and i + 1 < len(sys.argv):
            refine_every_val = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--gif":
            make_gif_flag = True; i += 1
        elif sys.argv[i] == "--gif-vars" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split(',')
            if len(parts) == 2:
                gif_vx, gif_vy = parts[0].strip(), parts[1].strip()
            i += 2
        elif sys.argv[i] == "--gif-res" and i + 1 < len(sys.argv):
            gif_res = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--gif-fps" and i + 1 < len(sys.argv):
            gif_fps_val = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--gif-linear":
            gif_log_flag = False; i += 1
        else:
            i += 1
    
    run_genetic_optimization(
        input_file,
        population_size=pop_size,
        generations=n_gen,
        mutation_rate=mut_rate,
        seed=seed,
        verbose=True,
        local_refine_method=refine_method,
        local_refine_top_k=refine_topk_val,
        local_refine_every=refine_every_val,
        make_gif=make_gif_flag,
        gif_var_x=gif_vx,
        gif_var_y=gif_vy,
        gif_resolution=gif_res,
        gif_log_scale=gif_log_flag,
        gif_fps=gif_fps_val,
    )


if __name__ == "__main__":
    main()