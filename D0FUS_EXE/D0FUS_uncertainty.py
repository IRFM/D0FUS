"""
D0FUS_uncertainty.py -- Uncertainty quantification around a single design point.

Fourth execution mode (RUN / SCAN / OPTIMIZATION / UNCERTAINTY), self-contained
in the same spirit as the other input files: the user pastes a complete RUN deck
(the design point), then appends an [UNCERTAINTY] section listing the parameters
to study and their envelopes, plus a [CONTROLS] section.

  * The design point provides the CENTRAL value of every uncertain law, so the
    study auto-centres on the machine and re-centres if the design changes.
  * A parameter is studied by listing it under [UNCERTAINTY] as a distribution
    tri(lo, hi) / tri(lo, mode, hi) / unif(lo, hi) / norm(...), or as a
    model-form list envelope(A | B). The normal law is a TRUNCATED normal and
    accepts three forms: norm(sigma) centres on the design value, norm(lo, hi)
    centres on the design value with the bounds at about mean +/- 2 sigma, and
    norm(lo, centre, hi) lets the user set both the bounds and the centre.
  * Adding two scan axes  a = [min, max, n]  to the design point turns the study
    into the (a, R0) feasibility map (planned).

Building blocks:
  - evaluate(config): one in-memory solver call (no file I/O, no figures)
    returning the QoIs, the feasibility flag and the constraint margins.
  - triangular / normal / uniform LHS sampler.
  - parse_uq_file() / detect_mode(): the input-file front end.
  - run_uq_from_file(): forward propagation over the model envelope (serial here;
    joblib/loky parallelism is the next step).

Feasibility mirrors D0FUS_scan / D0FUS_genetic (Greenwald, Troyon, kink and
radial-build closure) so "feasible" means exactly what the optimiser means.
"""
#%% Imports

# Centralised imports: D0FUS_BIB/D0FUS_import.py exports all standard, scientific
# and plotting names (os, re, shutil, datetime, numpy, dataclasses replace/asdict,
# tqdm, ...). Path resolution mirrors the other EXE modules: D0FUS.py inserts the
# project root in sys.path in normal usage; the fallback covers a standalone run
# of this module.
try:
    from D0FUS_BIB.D0FUS_import import *
except ModuleNotFoundError:
    import sys, os
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from D0FUS_BIB.D0FUS_import import *

# (itertools, tempfile, Counter, scipy.stats and scipy.stats.qmc are exported
#  by D0FUS_import.py.)

# --- Project-specific D0FUS dependencies -------------------------------------
from D0FUS_EXE import D0FUS_run as RUN
from D0FUS_BIB.D0FUS_physical_functions import f_volume
from D0FUS_BIB.D0FUS_cost_functions import f_costs_Sheffield
from D0FUS_BIB.D0FUS_parameterization import M_blanket_effective
from D0FUS_BIB.D0FUS_radial_build_functions import Number_TF_coils

# Backwards-compatible alias for dataclasses.replace, used throughout the module.
dc_replace = replace


# --- default uncertain set, documented as (family, *params); the file front end
#     re-centres each marginal on the design point, so these are only a fallback.
UNCERTAIN_SPEC = {
    'H':                 ('tri', 0.75, 1.00, 1.50),
    'Tbar':              ('tri', 7.90, 8.90, 9.90),
    'C_Alpha':           ('tri', 3.00, 5.00, 7.00),
    'nu_n_manual':       ('tri', 0.00, 0.01, 0.21),
    'nu_T_manual':       ('tri', 1.80, 2.80, 3.80),
    'rho_ped':           ('tri', 0.90, 0.95, 0.97),
    'n_ped_frac':        ('tri', 0.85, 0.99, 0.99),
    'T_ped_frac':        ('tri', 0.40, 0.55, 0.70),
    'eta_WP_acad':       ('tri', 0.20, 0.30, 0.50),
    'gamma_CD_acad':     ('tri', 0.05, 0.20, 0.50),
    'Ce':                ('tri', 0.20, 0.45, 0.50),
    'betaN_limit':       ('tri', 2.80, 2.80, 3.60),
    'q_limit':           ('tri', 3.00, 3.50, 3.50),
    'Greenwald_limit':   ('tri', 0.80, 1.00, 1.50),
    'Supra_cost_factor': ('tri', 1.50, 2.00, 3.50),
    'discount_rate':     ('tri', 0.05, 0.07, 0.10),
}

# Convenience for the programmatic run_uq() path on a stock RUN deck that is not
# already in the study frame; the file front end does NOT use this (the pasted
# design deck is authoritative).
FIXED_OVERRIDES = {'CD_source': 'Academic', 'kink_parameter': 'q95',
                   'Plasma_profiles': 'Manual'}

DIST_RE = re.compile(r'^(tri|norm|unif)\((.*)\)$', re.IGNORECASE)
ENV_RE = re.compile(r'^envelope\((.*)\)$', re.IGNORECASE)
# Indexed access into a comma-separated string field, e.g. 'f_imp_core[0]'
# samples the FIRST impurity concentration of the deck (species order given by
# impurity_species) while leaving the other entries at their deck values.
IDX_RE = re.compile(r'^(\w+)\[(\d+)\]$')


def design_value(base, name):
    """Design-deck value of a (possibly indexed) uncertain input, or None.

    Plain names read the GlobalConfig attribute directly. Indexed names such as
    'f_imp_core[0]' read element i of the comma-separated string attribute, so
    the truncated normals can auto-centre on the deck value exactly as for
    scalar fields.
    """
    m = IDX_RE.match(name)
    if m is None:
        val = getattr(base, name, None)
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    raw = getattr(base, m.group(1), None)
    if raw is None:
        return None
    try:
        return float(str(raw).split(',')[int(m.group(2))].strip())
    except (ValueError, IndexError):
        return None


# =============================================================================
# Single-configuration evaluation (the engine brick)
# =============================================================================
def _compute_cost(cfg, P_CD, P_elec, Gamma_n, Surface, c, d, kappa,
                  T_op_limit, CF, t_life_bl_yr, t_life_div_yr, V_rb_BB):
    """Sheffield (2016) COE [EUR/MWh] and capital cost [B EUR], as in D0FUS_scan.

    Aligned on the merged exact-volume cost path: f_volume now takes the
    Princeton-D (Delta_TF, H_TF) signature, and f_costs_Sheffield consumes the
    availability schedule (T_op_limit, CF) and component lifetimes
    (t_life_bl_yr, t_life_div_yr) produced by run(), in place of the former
    Util_factor / Dwell_factor / dt_rep inputs.
    """
    try:
        P_th = cfg.P_fus * M_blanket_effective(cfg.Blanket_choice) + P_CD
        _, _, Delta_TF = Number_TF_coils(cfg.R0, cfg.a, cfg.b, cfg.ripple_adm, cfg.L_min)
        H_TF = 2.0 * (kappa * cfg.a + cfg.b + c)
        (V_blanket, V_TF_Pappus, V_CS_geom, V_FI) = f_volume(
            cfg.a, cfg.b, c, d, cfg.R0, kappa, Delta_TF, H_TF)
        cres = f_costs_Sheffield(
            discount_rate=cfg.discount_rate, contingency=cfg.contingency,
            T_life=cfg.T_life, T_build=cfg.T_build,
            P_t=P_th, P_e=max(P_elec, 1.0), P_aux=P_CD, Gamma_n=Gamma_n,
            T_op_limit=T_op_limit, CF=CF,
            t_life_bl_yr=t_life_bl_yr, t_life_div_yr=t_life_div_yr,
            V_FI=V_FI, V_pc=V_TF_Pappus + V_CS_geom, V_sg=V_blanket,
            V_bl=V_rb_BB, S_tt=0.1 * Surface, Supra_cost_factor=cfg.Supra_cost_factor)
        return float(cres[3]), float(cres[2]) * 1e-3
    except Exception:
        return np.nan, np.nan


def _radial_build_ok(cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line):
    """Faithful mirror of D0FUS_genetic.check_radial_build (geometric closure)."""
    for val in (cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line):
        if val is None or isinstance(val, (complex, np.complexfloating)):
            return False
        if not np.isfinite(val) or val < 0:
            return False
    if c_TF < 1e-3 or d_CS < 1e-3:   # TF / CS winding pack too thin to be valid
        return False
    return True


def evaluate(cfg):
    """Run one configuration in memory and return QoIs plus feasibility.

    Non-converged samples are tagged with a 'failure' category so that the
    summary and figures can separate physically distinct outcomes:
      'no_operating_point' : the plasma solver found no solution at the
                             prescribed (P_fus, Tbar) point, typically because
                             radiation exceeds the heating power for that draw;
      'no_closure'         : a plasma solution exists but the engineering chain
                             (radial build, flux budget, cost) returned NaN;
      'crash'              : the solver raised an exception.
    """
    # The Monte-Carlo deliberately visits corners where the solver has no
    # solution (radiation exceeding the heating power, non-closing builds):
    # the diagnostic RuntimeWarnings that D0FUS_run emits there are expected
    # by construction in this mode and are silenced locally. The information
    # is not lost: it comes back through the 'failure' tag of each sample.
    # This filter is process-local (loky workers), so RUN mode keeps its
    # warnings untouched.
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = RUN.run(cfg, verbose=0)
    except Exception:
        return {'converged': False, 'feasible': False, 'failure': 'crash'}

    (B0, B_CS, B_pol, tauE, W_th, Q, Volume, Surface, Ip, Ib, I_CD, I_Ohm,
     nbar, nbar_line, nG, pbar, betaN, betaT, betaP, qstar, q95,
     P_CD, P_sep, P_Thresh, eta_CD, P_elec, P_wallplug, cost, P_Brem, P_syn,
     P_line, P_line_core, heat, heat_par, heat_pol, lambda_q, q_target,
     P_wall_H, P_wall_L, Gamma_n, f_alpha, tau_alpha, J_TF, J_CS,
     c, c_WP_TF, c_Nose_TF, sz_TF, st_TF, sr_TF, Steel_fraction_TF,
     d, sz_CS, st_CS, sr_CS, Steel_fraction_CS, B_CS_out, J_CS_out,
     r_minor, r_sep, r_c, r_d, kappa, kappa_95, delta, delta_95,
     PsiPI, PsiRampUp, Psiplateau, PsiPF, PsiCS, Vloop_sc, li_sc,
     eta_LH, eta_EC, eta_NBI, P_LH, P_EC, P_NBI, P_ICR, I_LH, I_EC, I_NBI,
     f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF,
     f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS,
     beta_fast_alpha, betaN_total, tau_sd_alpha, W_fast_alpha, *_rest) = res

    # Trailing tuple fields appended by the dev-Mat integration (the coil /
    # volume / mass block, then the divertor dict as the very last element).
    # Extract the cost inputs by absolute index, identical to D0FUS_scan.
    _g = lambda i: (res[i] if len(res) > i else np.nan)
    _t_bl_yr, _t_div_yr        = _g(130), _g(131)
    _T_op_limit, _CF, _V_rb_BB = _g(132), _g(135), _g(138)
    _diag = _rest[-1] if _rest else {}

    plasma_ok  = bool(np.isfinite(Q) and np.isfinite(Ip) and Ip > 0)
    closure_ok = bool(np.isfinite(cost))
    converged  = plasma_ok and closure_ok
    if not converged:
        return {'converged': False, 'feasible': False,
                'failure': 'no_operating_point' if not plasma_ok else 'no_closure'}

    c_TF = r_sep - r_c if np.isfinite(r_c) and np.isfinite(r_sep) else np.nan
    d_CS = r_c - r_d   if np.isfinite(r_c) and np.isfinite(r_d)   else np.nan
    f_bs = (Ib / Ip * 100.0) if Ip > 0 else np.nan
    gw   = (nbar_line / nG) if nG > 0 else np.nan
    COE, C_invest = _compute_cost(cfg, P_CD, P_elec, Gamma_n, Surface, c, d, kappa,
                                  _T_op_limit, _CF, _t_bl_yr, _t_div_yr, _V_rb_BB)

    q_kink = q95 if cfg.kink_parameter == 'q95' else qstar

    build_ok  = _radial_build_ok(cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line)
    gw_ok     = bool(np.isfinite(gw)    and gw    <= cfg.Greenwald_limit)
    troyon_ok = bool(np.isfinite(betaN_total) and betaN_total <= cfg.betaN_limit)
    kink_ok   = bool(np.isfinite(q_kink) and q_kink >= cfg.q_limit)
    stable_ok = bool(gw_ok and troyon_ok and kink_ok)
    feasible  = bool(build_ok and stable_ok)

    gw_margin     = (1.0 - gw / cfg.Greenwald_limit) if np.isfinite(gw)    else np.nan
    troyon_margin = (1.0 - betaN_total / cfg.betaN_limit)  if np.isfinite(betaN_total) else np.nan
    kink_margin   = (q_kink / cfg.q_limit - 1.0)     if np.isfinite(q_kink) else np.nan

    binding = None
    if not feasible:
        cand = {'build':     -1.0 if not build_ok else np.inf,
                'greenwald':  gw_margin     if not gw_ok     else np.inf,
                'troyon':     troyon_margin if not troyon_ok else np.inf,
                'kink':       kink_margin   if not kink_ok   else np.inf}
        binding = min(cand, key=lambda k: cand[k] if np.isfinite(cand[k]) else np.inf)

    return {
        'converged': True, 'feasible': feasible,
        'build_ok': build_ok, 'stable_ok': stable_ok, 'binding': binding,
        'Q': Q, 'P_elec': P_elec, 'COE': COE, 'C_invest': C_invest,
        'f_bs': f_bs, 'beta_N': betaN, 'q95': q95, 'B0': B0, 'B_CS': B_CS,
        'P_sep': P_sep, 'd_TF': c_TF, 'd_CS': d_CS,
        'gw_margin': gw_margin, 'troyon_margin': troyon_margin, 'kink_margin': kink_margin,
    }


# =============================================================================
# Operator retuning of the operating point
# =============================================================================
def _retune_ladder(T0, T_lo, T_hi, n_points=4):
    """Candidate operating temperatures covering the admissible window.

    The ladder is built RELATIVE to the window rather than in absolute keV
    steps: each direction is probed at n_points fractions of the distance
    between the starting temperature T0 and the corresponding bound, the last
    fraction landing exactly on the bound. The search therefore spans the full
    window whatever the machine scale (ITER at 7.75 keV, a plant at 9 keV or a
    compact device at 4.5 keV need no per-deck tuning), and the evaluation
    budget stays at most 2 * n_points extra solver calls per failing sample.

    Returns (up, down): the candidate temperatures above and below T0,
    ordered from the nearest to the farthest.
    """
    fractions = [(k + 1) / n_points for k in range(n_points)]
    up = [T0 + f * (T_hi - T0) for f in fractions if T_hi > T0]
    down = [T0 - f * (T0 - T_lo) for f in fractions if T_lo < T0]
    return up, down


def evaluate_retuned(cfg, window):
    """Feasibility WITH operator retuning of the volume-averaged temperature.

    Rationale: the forward Monte-Carlo holds the operating point (P_fus, Tbar)
    frozen, so a favourable confinement draw is converted by the inverse solve
    into a lower plasma current and a higher Greenwald fraction instead of
    extra margin. In reality the operator controls the density, hence
    indirectly the temperature at fixed fusion power, and would move the
    operating point. This function asks the corresponding question: does a
    feasible operating temperature EXIST inside `window` = (T_lo, T_hi)?

    The as-designed point (deck Tbar) is evaluated first; if it is feasible
    the sample is accepted unchanged. Otherwise the temperature is moved along
    RETUNE_LADDER, trying first the direction suggested by the failure
    signature: density-limit, beta-limit and radiative no-operating-point
    failures call for a HIGHER temperature (lower density at fixed fusion
    power), kink and flux-closure failures for a LOWER one; both directions
    are eventually tried. The first feasible point is accepted and tagged
    retuned=True with its Tbar_used. If no temperature in the window is
    feasible, the best converged attempt is returned (so the margin statistics
    stay meaningful), tagged feasible=False. In every case the key
    'feasible_as_designed' preserves the frozen-point verdict, so summaries
    can report both readings side by side.
    """
    T_lo, T_hi = window[0], window[1]
    n_points = int(window[2]) if len(window) > 2 else 4
    r0 = evaluate(cfg)
    r0['feasible_as_designed'] = bool(r0.get('feasible'))
    r0['retuned'] = False
    r0['Tbar_used'] = float(cfg.Tbar)
    if r0['feasible_as_designed']:
        return r0

    up_first = (r0.get('binding') in ('greenwald', 'troyon')
                or r0.get('failure') == 'no_operating_point')
    up, down = _retune_ladder(float(cfg.Tbar), T_lo, T_hi, n_points)
    first, second = (up, down) if up_first else (down, up)
    # Interleave the two directions, preferred one first, nearest steps first.
    steps = [t for pair in zip(first, second) for t in pair]
    steps += first[len(second):] + second[len(first):]

    tried = {round(float(cfg.Tbar), 3)}
    best = r0 if r0.get('converged') else None
    for T in steps:
        key = round(float(T), 3)
        if key in tried:
            continue
        tried.add(key)
        r = evaluate(dc_replace(cfg, Tbar=float(T)))
        r['feasible_as_designed'] = False
        r['Tbar_used'] = T
        if r.get('feasible'):
            r['retuned'] = True
            return r
        if best is None and r.get('converged'):
            r['retuned'] = False
            best = r
    if best is None:
        best = r0
    best.setdefault('retuned', False)
    return best


def evaluate_pulse(cfg, tbar_window, pulse_cfg):
    """Feasibility WITH pulse-length relaxation of the flux budget.

    A design that closes structurally but exhausts the central-solenoid flux
    is not simply infeasible: it can still run, on a shorter flat-top. This
    function first evaluates the point at the full design pulse (with operator
    temperature retuning when tbar_window is given). If that already passes,
    the sample keeps pulse_frac = 1.0. Otherwise, and ONLY when the failure is
    flux/build related (no build-flux closure, or the radial build binds), the
    flat-top duration is walked down a ladder of pulse fractions and, last, an
    absolute floor (e.g. 10 s). The largest pulse fraction that yields a
    feasible design is recorded in 'pulse_frac' (with pulse_limited=True); if
    even the floor fails, the design is genuinely flux-infeasible and keeps
    its full-pulse verdict with pulse_frac = 0.0.

    pulse_cfg = (fractions, floor_s). Pulse tranches are evaluated at the deck
    operating temperature (a well-defined operating point), so the reported
    pulse capability answers 'how much flat-top can this draw sustain on
    flux', decoupled from the plasma-limit retuning of stage one.
    """
    r = evaluate_retuned(cfg, tbar_window) if tbar_window else evaluate(cfg)
    r.setdefault('feasible_as_designed', bool(r.get('feasible')))
    r['pulse_frac'] = 1.0
    r['pulse_limited'] = False
    if r.get('feasible') or pulse_cfg is None:
        return r

    # Only flux / build-closure failures are addressable by a shorter pulse.
    addressable = (r.get('failure') == 'no_closure' or r.get('binding') == 'build')
    if not addressable:
        return r

    fractions, floor_s = pulse_cfg
    t_full = float(cfg.Temps_Plateau_input)
    durations = sorted({f * t_full for f in fractions} | {float(floor_s)},
                       reverse=True)
    durations = [t for t in durations if 0.0 < t < t_full]
    last_closed = None
    for t in durations:
        rp = evaluate(dc_replace(cfg, Temps_Plateau_input=float(t)))
        if rp.get('feasible'):
            rp['feasible_as_designed'] = False
            rp['retuned'] = r.get('retuned', False)
            rp['Tbar_used'] = r.get('Tbar_used', float(cfg.Tbar))
            rp['pulse_frac'] = t / t_full
            rp['pulse_limited'] = True
            rp['Temps_Plateau_used'] = t
            return rp
        if rp.get('converged'):
            last_closed = rp     # closes on flux at this pulse but plasma-limited
    if last_closed is not None:
        # A shorter pulse relieves the CS flux, but the design is still held
        # back by a plasma limit (kink, Greenwald, ...). Report it by that TRUE
        # binding rather than as a flux/build wall: the full-pulse no-closure
        # was only the first symptom of the same (usually high-current) draw.
        last_closed['feasible_as_designed'] = False
        last_closed['retuned'] = r.get('retuned', False)
        last_closed['Tbar_used'] = r.get('Tbar_used', float(cfg.Tbar))
        last_closed['pulse_frac'] = 0.0
        last_closed['pulse_limited'] = False
        last_closed['pulse_relieved'] = True
        return last_closed
    r['pulse_frac'] = 0.0     # genuine CS wall: no closure at any pulse
    return r


# =============================================================================
# Radial-build (central-solenoid) relief ladder
# =============================================================================
def _family_of(r):
    """Coarse feasibility family of a non-feasible evaluate() result:
    'radial_build' when the central solenoid / build does not close,
    'stability' when a plasma limit (Greenwald, Troyon, kink) binds,
    'no_operating_point' when the plasma solver found no solution at all,
    'crash' when the solver raised."""
    if r.get('feasible'):
        return 'feasible'
    if r.get('binding') == 'build' or r.get('failure') == 'no_closure':
        return 'radial_build'
    if r.get('failure') == 'no_operating_point':
        return 'no_operating_point'
    if r.get('failure') == 'crash':
        return 'crash'
    return 'stability'


def _relief_outcome(rp, r0, flux_cut):
    """Interpret one relief attempt (the CS is asked to supply flux_cut less
    inductive flux). Returns a finished result dict when the CS then closes,
    else None to keep escalating. If the CS closes but a plasma limit now binds,
    the sample is reclassified as stability-limited: the CS was only the first
    symptom of a high-current draw, and no CS relief removes a plasma-limit
    wall."""
    if not rp.get('converged'):
        return None
    rp['feasible_as_designed'] = bool(r0.get('feasible_as_designed', False))
    rp['retuned']   = bool(r0.get('retuned', False))
    rp['Tbar_used'] = r0.get('Tbar_used', np.nan)
    rp['flux_cut']  = float(flux_cut)
    rp['build_relieved'] = True
    rp['category'] = 'radial_build' if rp.get('feasible') else 'stability'
    return rp


def evaluate_relief(cfg, tbar_window, relief_cfg):
    """Feasibility with operator Tbar retuning followed by a radial-build relief
    ladder for central-solenoid (flux-closure) failures.

    A design whose CS cannot supply the inductive volt-seconds is not simply
    infeasible: the flux it must provide can be shed. Rather than tracking each
    engineering lever separately, the relief is expressed as ONE transparent
    quantity, the fraction of CS inductive flux that must be removed for the
    design to close. That reduction is realised by shedding the same fraction of
    the two reducible volt-second terms, the current ramp-up (assisted
    non-inductively by H&CD, through f_heat_ramp) and the flat-top (a shorter
    burn); so a reported '25% flux relief' reads as 'obtainable with H&CD ramp
    assist and/or a 25% shorter pulse'. The CS coil field is NOT a relief lever.

    The smallest flux reduction that closes the CS is recorded in 'flux_cut'. If
    closing the CS reveals a plasma-stability limit the sample is reclassified as
    stability-limited. Samples that never close, even at the largest reduction,
    are a genuine radial-build wall. That wall also collects build failures that
    flux relief cannot address, since the shed levers act on the CS volt-seconds
    only: a TF coil that cannot be built (for instance under a peak-field scan)
    is not flux-relievable and falls straight through to the wall.

    relief_cfg = list of flux-reduction fractions to try (e.g. 0.25, 0.5, 0.75);
    None or empty disables relief entirely.
    """
    r = evaluate_retuned(cfg, tbar_window) if tbar_window else evaluate(cfg)
    r.setdefault('feasible_as_designed', bool(r.get('feasible')))
    r.setdefault('retuned', False)
    r.setdefault('Tbar_used', float(cfg.Tbar))
    r['flux_cut'] = 0.0
    r['build_relieved'] = False

    if r.get('feasible'):
        r['category'] = 'feasible'
        return r
    if not relief_cfg:
        r['category'] = _family_of(r)
        return r

    # Only CS flux / build-closure failures are addressable by shedding flux.
    addressable = (r.get('failure') == 'no_closure' or r.get('binding') == 'build')
    if not addressable:
        r['category'] = _family_of(r)   # stability / no_operating_point / crash
        return r

    t_full = float(cfg.Temps_Plateau_input)
    for delta in sorted(f for f in relief_cfg if 0.0 < f < 1.0):
        # Shed the same fraction from both reducible flux terms: assist the
        # ramp-up (f_heat_ramp = delta) and shorten the flat-top (x (1 - delta)),
        # so the reducible CS flux is reduced by delta overall.
        rp = evaluate(dc_replace(cfg, f_heat_ramp=float(delta),
                                 Temps_Plateau_input=float((1.0 - delta) * t_full)))
        out = _relief_outcome(rp, r, flux_cut=delta)
        if out is not None:
            return out

    # genuine CS / radial-build wall: closes at no achievable flux reduction
    r['category']       = 'radial_build'
    r['build_relieved'] = False
    r['flux_cut']       = 1.0
    return r


def parse_relief_controls(controls):
    """Read the CS radial-build relief control.

    [CONTROLS] key:
      cs_relief = 0.25, 0.5, 0.75   fractions of the reducible CS inductive flux
                                    to try shedding (via H&CD ramp assist and/or
                                    a shorter flat-top) to close a non-closing CS.
    Returns the list of fractions, or None when relief is off. Backward
    compatible: a deck that still sets pulse_retune or ramp_retune activates
    relief with the default fractions.
    """
    raw = controls.get('cs_relief', controls.get('flux_relief', None))
    if raw is None:
        if (str(controls.get('pulse_retune', '')).strip() or
                str(controls.get('ramp_retune', '')).strip()):
            raw = '0.25, 0.5, 0.75'
        else:
            return None
    return [float(x) for x in str(raw).split(',')]


def parse_pulse_controls(controls):
    """Read the optional pulse-relaxation controls.

    [CONTROLS] keys:
      pulse_retune    = Temps_Plateau_input   activates the pulse search
      pulse_fractions = 0.75, 0.5, 0.25       tranches below the full pulse
      pulse_floor     = 10.0                  absolute floor [s]
    Returns (fractions, floor_s) or None when pulse relaxation is off.
    """
    if str(controls.get('pulse_retune', '')).strip() not in (
            'Temps_Plateau_input', 'pulse', 'Temps_Plateau'):
        return None
    raw = str(controls.get('pulse_fractions', '0.75, 0.5, 0.25'))
    fractions = [float(t) for t in raw.split(',')]
    floor_s = float(controls.get('pulse_floor', 10.0))
    return (fractions, floor_s)


def parse_retune_controls(controls):
    """Read the optional operator-retuning controls.

    [CONTROLS] keys:
      retune        = Tbar          activates the retuning search
      Tbar_window   = 6.0, 12.0     admissible operating window [keV]
      retune_points = 4             ladder resolution per direction (optional)
    Returns (T_lo, T_hi, n_points) or None when retuning is off. The ladder is
    built relative to the window (see _retune_ladder), so the window bounds are
    always reachable whatever the deck's design temperature.
    """
    if str(controls.get('retune', '')).strip().lower() != 'tbar':
        return None
    win = str(controls.get('Tbar_window', '6.0, 12.0'))
    lo, hi = (float(t) for t in win.split(','))
    return (lo, hi, int(controls.get('retune_points', 4)))


# =============================================================================
# Sampling
# =============================================================================
def _triangular_ppf(u, lo, mode, hi):
    if hi <= lo:
        return np.full_like(u, lo, dtype=float)
    c = min(max((mode - lo) / (hi - lo), 0.0), 1.0)
    return stats.triang.ppf(u, c=c, loc=lo, scale=(hi - lo))


def _split_truncnorm_ppf(u, mu, s_lo, s_hi, lo, hi):
    """Inverse CDF of a SPLIT (two-piece) truncated normal on [lo, hi].

    The density is a half-normal of width s_lo below the mode mu and of width
    s_hi above it, joined continuously at mu (Fechner two-piece normal), then
    truncated to [lo, hi]. This gives an asymmetric belief with a single mode
    at mu: a sharp side and a long tail on the other, which a symmetric normal
    cannot represent. With s_lo = s_hi it reduces to the ordinary truncated
    normal. The two sides are sampled from their own truncnorm, with the split
    weight set so the joined density is continuous (mass on each side is
    proportional to its sigma times the truncated area of that side).
    """
    u = np.asarray(u, dtype=float)
    Phi = stats.norm.cdf
    # truncated area of each half (unnormalised, continuity-weighted by sigma)
    m_lo = s_lo * (0.5 - Phi((lo - mu) / s_lo))
    m_hi = s_hi * (Phi((hi - mu) / s_hi) - 0.5)
    p_lo = m_lo / (m_lo + m_hi)
    out = np.empty_like(u)
    low = u < p_lo
    # lower half: truncnorm on [lo, mu], quantile rescaled into [0, 1]
    if np.any(low):
        a, b = (lo - mu) / s_lo, 0.0
        out[low] = stats.truncnorm.ppf(u[low] / p_lo, a, b, loc=mu, scale=s_lo)
    # upper half: truncnorm on [mu, hi]
    if np.any(~low):
        a, b = 0.0, (hi - mu) / s_hi
        out[~low] = stats.truncnorm.ppf((u[~low] - p_lo) / (1.0 - p_lo),
                                        a, b, loc=mu, scale=s_hi)
    return out


def _marginal_ppf(u, dist):
    """Inverse CDF of one marginal evaluated on u in [0, 1]."""
    fam = dist[0]
    if fam == 'tri':
        _, lo, mode, hi = dist
        return _triangular_ppf(u, lo, mode, hi)
    if fam == 'norm':
        if len(dist) == 6:                      # SPLIT truncated normal
            _, mu, s_lo, s_hi, lo, hi = dist
            if s_lo <= 0 or s_hi <= 0:
                return np.full_like(np.asarray(u, dtype=float), mu)
            return _split_truncnorm_ppf(u, mu, s_lo, s_hi, lo, hi)
        mu, sigma = dist[1], dist[2]
        if sigma <= 0:                          # degenerate width -> point mass at mu
            return np.full_like(u, mu, dtype=float)
        if len(dist) >= 5:                      # bounded -> TRUNCATED normal on [lo, hi]
            # A truncated normal (not a clip): the density stays smooth inside the
            # bounds and is exactly zero outside, instead of piling probability mass
            # on lo and hi as np.clip would. Standardised truncation limits a, b are
            # passed to scipy.stats.truncnorm (Burkardt 2014; scipy.stats docs).
            lo, hi = dist[3], dist[4]
            a, b = (lo - mu) / sigma, (hi - mu) / sigma
            return stats.truncnorm.ppf(u, a, b, loc=mu, scale=sigma)
        return stats.norm.ppf(u, loc=mu, scale=sigma)
    if fam == 'unif':
        _, lo, hi = dist
        return lo + u * (hi - lo)
    raise ValueError(f"unknown distribution family '{fam}'")


def sample_lhs(spec, n, seed=0):
    """Latin-Hypercube sample from the declared marginals."""
    names = list(spec.keys())
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    out = np.empty_like(unit)
    for j, name in enumerate(names):
        out[:, j] = _marginal_ppf(unit[:, j], spec[name])
    return names, out


# =============================================================================
# Configuration assembly
# =============================================================================
def _coerce(val, ref):
    """Coerce a string value to the type of the reference attribute."""
    if isinstance(ref, bool):
        return str(val).strip().lower() in ('true', '1', 'yes')
    if isinstance(ref, int) and not isinstance(ref, bool):
        try:
            return int(float(val))
        except ValueError:
            return val
    if isinstance(ref, float):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def build_config(base, names, row, extra_overrides=None):
    """
    Apply an envelope combo and one sampled row to the base design.

    Uses dataclasses.replace (a shallow clone of the scalar GlobalConfig) instead of
    copy.deepcopy. This is what the scan worker does and is far cheaper, which matters
    when it runs once per Monte-Carlo sample.
    """
    fields = base.__dataclass_fields__
    changes = {}
    if extra_overrides:
        for k, v in extra_overrides.items():
            if k in fields:
                changes[k] = _coerce(v, getattr(base, k))
    for name, val in zip(names, row):
        m = IDX_RE.match(name)
        if m is not None and m.group(1) in fields:
            # Indexed entry of a comma-separated string field: rebuild the
            # string with element i replaced by the sampled value, preserving
            # the other entries (possibly already modified by a previous name).
            fname, i = m.group(1), int(m.group(2))
            parts = [p.strip() for p in
                     str(changes.get(fname, getattr(base, fname))).split(',')]
            if i < len(parts):
                parts[i] = f'{float(val):.6g}'
                changes[fname] = ', '.join(parts)
            continue
        if name in fields:
            changes[name] = float(val)
    return dc_replace(base, **changes)


# =============================================================================
# Input-file front end (self-contained 4th-mode parser + auto-detection)
# =============================================================================
def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _build_marginal(name, fam, args, central):
    """Turn a parsed token into (family, *params), centring on the design value."""
    if fam == 'tri':
        if len(args) == 2:                       # (lo, hi) -> mode = design value
            lo, hi = args
            mode = central if central is not None else 0.5 * (lo + hi)
        elif len(args) == 3:                     # (lo, mode, hi) explicit
            lo, mode, hi = args
        else:
            raise ValueError(f"tri() for '{name}' expects 2 or 3 arguments")
        if not (lo <= mode <= hi):
            print(f"  [UQ] warning: '{name}' central {mode:g} outside "
                  f"[{lo:g}, {hi:g}] -> clamped.")
            mode = min(max(mode, lo), hi)
        return ('tri', lo, mode, hi)
    if fam == 'unif':
        return ('unif', args[0], args[1])
    if fam == 'norm':
        # Truncated normal parameterised by the user as bounds and centre.
        #   norm(sigma)                       -> mean = design value, unbounded
        #   norm(lo, hi)                      -> mean = design value / midpoint
        #   norm(lo, centre, hi)              -> mean = centre, bounds lo/hi
        #   norm(lo, centre, hi, sigma)       -> explicit symmetric sigma
        #   norm(lo, centre, hi, s_lo, s_hi)  -> SPLIT normal: sharp side / long tail
        # For the 2/3-argument forms the standard deviation is set so that lo and hi
        # sit at about mean +/- 2 sigma, i.e. sigma = (hi - lo) / 4. Use the 4th
        # argument to decouple the width from the bounds (e.g. a narrow peak with a
        # far-reaching but rare tail), and the 5th to make the two sides asymmetric.
        if len(args) == 1:                       # (sigma) -> mean = design value
            if central is None:
                raise ValueError(f"norm() for '{name}': sigma-only form needs a "
                                 f"design value to centre on")
            return ('norm', central, args[0])
        if len(args) == 2:                       # (lo, hi) -> mean = design value
            lo, hi = args
            mu = central if central is not None else 0.5 * (lo + hi)
            sig_args = ()
        elif len(args) == 3:                     # (lo, centre, hi) explicit
            lo, mu, hi = args
            sig_args = ()
        elif len(args) == 4:                     # (lo, centre, hi, sigma)
            lo, mu, hi, sig = args
            sig_args = (sig,)
        elif len(args) == 5:                     # (lo, centre, hi, s_lo, s_hi) split
            lo, mu, hi, s_lo, s_hi = args
            sig_args = (s_lo, s_hi)
        else:
            raise ValueError(f"norm() for '{name}' expects 1 to 5 arguments")
        if not (lo <= mu <= hi):
            print(f"  [UQ] warning: '{name}' centre {mu:g} outside "
                  f"[{lo:g}, {hi:g}] -> clamped.")
            mu = min(max(mu, lo), hi)
        if len(sig_args) == 2:                   # split (two-piece) truncated normal
            return ('norm', mu, sig_args[0], sig_args[1], lo, hi)
        sigma = sig_args[0] if sig_args else (hi - lo) / 4.0
        return ('norm', mu, sigma, lo, hi)
    raise ValueError(f"unknown marginal {fam}{tuple(args)}")


def _load_design_config(design_lines):
    """
    Build a GlobalConfig from the design section by reusing the RUN deck parser.

    The temporary deck is kept on disk (not deleted) and its path is returned, so that
    parallel workers can rebuild the base configuration from it instead of receiving the
    object through pickling, which the loky/spawn backend cannot always do on Windows.
    """
    fd, tmp = tempfile.mkstemp(suffix='_uq_design.txt')
    os.close(fd)
    with open(tmp, 'w') as fh:
        fh.write('\n'.join(design_lines) + '\n')
    return RUN.load_config_from_file(tmp, verbose=0), tmp


def parse_uq_file(path):
    """
    Parse a self-contained UNCERTAINTY file.

    Returns base (GlobalConfig built from the design point), spec (dict of
    centred marginals), envelope (dict of model-option lists), controls (dict),
    and deck_path (the persistent design deck the workers rebuild base from).
    """
    section = 'design'
    design, raw_spec, controls = [], {}, {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('[') and s.endswith(']') and '=' not in s:
                section = s[1:-1].strip().lower()
                continue
            if section == 'design':
                design.append(line.rstrip('\n'))
                continue
            body = line.split('#', 1)[0].strip()
            if not body or '=' not in body:
                continue
            key, rhs = (t.strip() for t in body.split('=', 1))
            if section == 'controls':
                controls[key] = int(rhs) if rhs.lstrip('-').isdigit() else rhs
            else:                                # [uncertainty]
                raw_spec[key] = rhs

    # Split the [uncertainty] entries into model envelopes, distributions, and plain
    # scalar overrides. Scalars are folded into the design deck so the base config the
    # workers rebuild from disk is identical to the one used here.
    env_rhs, dist_rhs, scalar_lines = {}, {}, []
    for key, rhs in raw_spec.items():
        if ENV_RE.match(rhs):
            env_rhs[key] = rhs
        elif DIST_RE.match(rhs):
            dist_rhs[key] = rhs
        else:
            scalar_lines.append(f"{key} = {rhs}")

    base, deck_path = _load_design_config(design + scalar_lines)

    spec, envelope = {}, {}
    for key, rhs in env_rhs.items():
        envelope[key] = [o.strip() for o in ENV_RE.match(rhs).group(1).split('|')]
    for key, rhs in dist_rhs.items():
        dist = DIST_RE.match(rhs)
        fam = dist.group(1).lower()
        args = [float(a) for a in dist.group(2).split(',')]
        spec[key] = _build_marginal(key, fam, args, design_value(base, key))
    return base, spec, envelope, controls, deck_path


def detect_mode(path):
    """Auto-detect RUN / SCAN / OPTIMIZATION / UNCERTAINTY from the file syntax."""
    has_dist = has_env = has_scan = has_opt = False
    with open(path) as fh:
        for line in fh:
            s = line.split('#', 1)[0].strip()
            if s.lower() in ('[uncertainty]', '[controls]'):
                return 'UNCERTAINTY'
            if '=' not in s:
                continue
            rhs = s.split('=', 1)[1].strip()
            if DIST_RE.match(rhs) or ENV_RE.match(rhs):
                has_dist = True
            elif rhs.startswith('[') and rhs.endswith(']'):
                inner = rhs[1:-1].strip()
                parts = [p.strip() for p in inner.split('|' if '|' in inner else ',')]
                if '|' in inner or not all(_is_number(p) for p in parts):
                    has_env = True
                elif len(parts) == 3:
                    has_scan = True
                elif len(parts) == 2:
                    has_opt = True
    if has_dist or has_env:
        return 'UNCERTAINTY'
    if has_scan:
        return 'SCAN'
    if has_opt:
        return 'OPTIMIZATION'
    return 'RUN'


# =============================================================================
# Forward propagation
# =============================================================================
# Per-process cache of base configurations, keyed by deck path. Each spawned worker
# rebuilds the base config from the deck once and reuses it for all its samples.
_BASE_CACHE = {}


def _load_base_cached(deck_path):
    base = _BASE_CACHE.get(deck_path)
    if base is None:
        base = RUN.load_config_from_file(deck_path, verbose=0)
        _BASE_CACHE[deck_path] = base
    return base


def _uq_worker(deck_path, names, row, combo, retune=None, relief=None):
    """
    Worker for the parallel Monte-Carlo. Every dependency is imported LOCALLY.

    Local-only imports stop cloudpickle from serialising this module's global
    namespace. When D0FUS.py is launched with %runfile in Spyder, __main__ is
    polluted with scipy symbols (erfc, ...); without local imports the loky workers
    fail to un-serialise the task ("Can't get attribute 'erfc' on __main__"). Every
    argument is a plain type (deck path, list of names, numpy row, dict), so the task
    is always picklable regardless of how the parent imported the modules. This
    mirrors the scan worker (_run_scan_point), which is robust on Windows/Spyder.
    """
    import os
    import sys
    _parent = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from D0FUS_EXE.D0FUS_uncertainty import (_load_base_cached, build_config,
                                             evaluate, evaluate_retuned,
                                             evaluate_relief)
    cfg = build_config(_load_base_cached(deck_path), names, row, combo)
    if relief is not None:
        return evaluate_relief(cfg, retune, relief)
    if retune is not None:
        return evaluate_retuned(cfg, retune)
    return evaluate(cfg)


def run_uq(base_deck_path, spec=UNCERTAIN_SPEC, n=200, seed=0, overrides=None):
    """Programmatic propagation for a single model combo on a stock deck (serial)."""
    base = RUN.load_config_from_file(base_deck_path, verbose=0)
    names, X = sample_lhs(spec, n, seed=seed)
    rows = [evaluate(build_config(base, names, X[i], overrides)) for i in range(n)]
    return names, X, rows


def run_uq_from_file(path, n_override=None, n_jobs=-1, verbose=5):
    """
    Parse a self-contained UNCERTAINTY file and propagate over the model envelope.

    The Monte-Carlo is evaluated in parallel with joblib/loky. On Windows the
    calling script MUST be guarded by  if __name__ == '__main__':  for the loky
    backend to spawn workers.

    Returns names, X, results (combo -> rows), controls.
    """
    from joblib import Parallel, delayed

    base, spec, envelope, controls, deck_path = parse_uq_file(path)
    n = n_override or controls.get('n_samples', 1000)
    retune = parse_retune_controls(controls)
    relief = parse_relief_controls(controls)
    names, X = sample_lhs(spec, n, seed=controls.get('seed', 0))

    if envelope:
        keys = list(envelope.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*envelope.values())]
    else:
        combos = [{}]

    # Flatten (combo, sample) into one task list for balanced core utilisation.
    flat = [(combo, X[i]) for combo in combos for i in range(n)]
    # One updating tqdm bar instead of joblib's per-batch log lines. return_as
    # 'generator' preserves submission order, so the index-based slicing below
    # that maps results back to each model combo stays valid.
    _gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_uq_worker)(deck_path, names, row, combo, retune, relief)
        for combo, row in flat)
    out = list(tqdm(_gen, total=len(flat), desc="UQ Monte-Carlo",
                    unit="run", disable=(verbose == 0)))

    results = {}
    for c_idx, combo in enumerate(combos):
        key = tuple(sorted(combo.items())) if combo else ('nominal',)
        results[key] = out[c_idx * n:(c_idx + 1) * n]
    return names, X, results, controls


def _pct(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(a, [5, 50, 95]))


# =============================================================================
# Entry point for the UNCERTAINTY mode (called by D0FUS.py)
# =============================================================================
def summarize_results(results):
    """Return (n_total, n_converged, n_feasible, binding_counter, failure_counter).

    binding counts the most-violated limit among CONVERGED infeasible samples;
    failures counts the non-convergence categories ('no_operating_point',
    'no_closure', 'crash') so that 'this draw has no solution' is reported
    separately from 'this draw violates an operational limit'.
    """
    all_rows = [r for k in results for r in results[k]]
    conv = [r for r in all_rows if r.get('converged')]
    feas = [r for r in conv if r.get('feasible')]
    binding = Counter(r.get('binding') for r in conv if not r.get('feasible'))
    failures = Counter(r.get('failure', 'unknown')
                       for r in all_rows if not r.get('converged'))
    return len(all_rows), len(conv), len(feas), binding, failures


def _write_summary(path, input_file, results, controls, scans=None):
    """Write a concise human-readable summary of the uncertainty study."""
    n, n_conv, n_feas, binding, failures = summarize_results(results)
    # The headline verdict is taken over the CONVERGED samples: a draw with no
    # operating point is a different (and separately reported) outcome from a
    # converged design that violates an operational limit.
    p_feas_conv = 100.0 * n_feas / max(n_conv, 1)
    p_feas_all  = 100.0 * n_feas / max(n, 1)
    verdict = ('LARGELY FEASIBLE' if p_feas_conv >= 85 else
               'MARGINAL' if p_feas_conv >= 60 else 'AT RISK')
    conv = [r for k in results for r in results[k] if r.get('converged')]

    def pct(key):
        a = np.array([r[key] for r in conv if np.isfinite(r.get(key, np.nan))])
        return tuple(np.percentile(a, [5, 50, 95])) if a.size else (np.nan, np.nan, np.nan)

    fail_txt = ", ".join(f"{k}={v}" for k, v in failures.items()) or "none"
    L = ["D0FUS uncertainty study summary",
         f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         f"Input file : {os.path.basename(input_file)}",
         "",
         f"Samples    : {n} total = {n_conv} converged + "
         f"{n - n_conv} without a solution ({fail_txt})",
         f"Feasible   : {p_feas_conv:.0f}% of converged samples "
         f"({p_feas_all:.0f}% of all samples)",
         f"Verdict    : {verdict}  (thresholds on the converged share)"]
    # Two-reading breakdown when operator retuning was active: the frozen
    # design-point verdict versus the operating-window verdict.
    if any('feasible_as_designed' in r for k in results for r in results[k]):
        n_design = sum(1 for r in conv if r.get('feasible_as_designed'))
        n_ret = sum(1 for r in conv if r.get('feasible') and r.get('retuned'))
        L.append(f"Retuning   : {100.0 * n_design / max(n_conv, 1):.0f}% feasible "
                 f"as designed; +{100.0 * n_ret / max(n_conv, 1):.0f}% recovered "
                 f"by moving Tbar inside the operating window")
    # Pulse-capability breakdown when pulse relaxation was active.
    if any('pulse_frac' in r for k in results for r in results[k]):
        all_rows = [r for k in results for r in results[k]]
        pl = [r for r in conv if r.get('pulse_limited')]
        n_full = sum(1 for r in conv if r.get('feasible')
                     and not r.get('pulse_limited'))
        # Genuinely flux/build-infeasible: went through the pulse ladder and
        # still did not close (pulse_frac forced to 0.0). These live in the
        # non-converged pool, so count them over all rows, as a share of the
        # whole Monte-Carlo.
        n_floor = sum(1 for r in all_rows
                      if r.get('pulse_frac') == 0.0 and not r.get('feasible'))
        buckets = [('>=3/4 pulse', 0.75, 1.0), ('1/2-3/4 pulse', 0.5, 0.75),
                   ('1/4-1/2 pulse', 0.25, 0.5), ('<1/4 pulse', 0.0, 0.25)]
        L.append("")
        L.append(f"Pulse      : of the converged designs, "
                 f"{100.0 * n_full / max(n_conv, 1):.0f}% keep the full flat-top and "
                 f"{100.0 * len(pl) / max(n_conv, 1):.0f}% are feasible only on a "
                 f"shorter one; {100.0 * n_floor / max(n, 1):.0f}% of all samples do "
                 f"not close on flux even at the floor pulse")
        for name, lo, hi in buckets:
            k = sum(1 for r in pl if lo < r.get('pulse_frac', 0.0) <= hi)
            if k:
                L.append(f"  reduced to {name:14s}: {100.0 * k / max(n_conv, 1):.0f}% "
                         f"of converged")
        n_relieved = sum(1 for r in conv if r.get('pulse_relieved'))
        if n_relieved:
            L.append(f"  note: {100.0 * n_relieved / max(n_conv, 1):.0f}% of converged "
                     f"exhaust the CS flux at full pulse but are ultimately held by a "
                     f"plasma limit (counted under that binding, not as a flux wall)")
    if binding:
        L.append("Binding limit among converged infeasible: "
                 + ", ".join(f"{k}={v}" for k, v in binding.items()))
    L += ["",
          "Headroom to each limit (normalised margin, P5 / P50 / P95):"]
    for key, name in [('gw_margin', 'Greenwald'), ('troyon_margin', 'Troyon'),
                      ('kink_margin', 'Kink (q95)')]:
        p5, p50, p95 = pct(key)
        L.append(f"  {name:12s}: {p5:+.3f} / {p50:+.3f} / {p95:+.3f}")

    combos = [k for k in results if k != ('nominal',)]
    if combos:
        L += ["", "Feasibility by model combination:"]
        for k in combos:
            rws = results[k]
            f = sum(1 for r in rws if r.get('converged') and r.get('feasible'))
            label = " · ".join(str(v) for _, v in k)
            L.append(f"  {label:34s}: {100.0 * f / max(len(rws), 1):.0f}%")

    if scans:
        L += ["", "Feasibility at the design value of each scanned parameter:"]
        for p, (xs, pf, nomv) in scans.items():
            L.append(f"  {p:8s} = {nomv:<8g}: {float(np.interp(nomv, xs, pf)):.0f}%")

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(L) + "\n")


def main(input_file, save_figures=True, output_dir=None, n_override=None,
         scan=True, scan_npts=11, scan_n=40, scan_frac=0.4, n_jobs=-1):
    """
    Run the full uncertainty study for an input file. Like the RUN / SCAN / GENETIC
    modes, it writes a timestamped folder under D0FUS_OUTPUTS/uncertainty/ containing a
    copy of the input deck, a synthetic summary, and the figures.
    """
    if output_dir is None:
        output_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(input_file)), '..', 'D0FUS_OUTPUTS'))

    names, X, results, controls = run_uq_from_file(
        input_file, n_override=n_override, n_jobs=n_jobs, verbose=5)

    n, n_conv, n_feas, binding, failures = summarize_results(results)
    p_feas_conv = 100.0 * n_feas / max(n_conv, 1)
    p_feas_all  = 100.0 * n_feas / max(n, 1)
    verdict = ('LARGELY FEASIBLE' if p_feas_conv >= 85 else
               'MARGINAL' if p_feas_conv >= 60 else 'AT RISK')
    print(f"\n  UNCERTAINTY verdict: {p_feas_conv:.0f}% of the {n_conv} converged "
          f"samples feasible ({p_feas_all:.0f}% of all {n})  ->  {verdict}")
    if failures:
        print("  Samples without a solution: "
              + ', '.join(f'{k}={v}' for k, v in failures.items()))
    if binding:
        print("  Binding limit among converged infeasible: "
              + ', '.join(f'{k}={v}' for k, v in binding.items()))

    scans = None
    if save_figures:
        from D0FUS_BIB import D0FUS_figures as FIG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, 'uncertainty',
                                   f"Uncertainty_D0FUS_{timestamp}")
        fig_dir = os.path.join(output_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # base, the centred marginals and the model-form envelope, parsed once.
        base, spec, envelope = parse_uq_file(input_file)[0:3]

        FIG.fig_robustness(results, save_dir=fig_dir)
        FIG.fig_margins(results, save_dir=fig_dir)
        # Distributions of the uncertain inputs (continuous marginals + model
        # switches) and of the main outputs (two separate figures).
        FIG.fig_inputs(names, X, base, spec, envelope, save_dir=fig_dir)
        FIG.fig_outputs(results, save_dir=fig_dir)
        if scan:
            specs = {}
            for p in ('P_fus', 'R0', 'a', 'Tbar'):
                if hasattr(base, p):
                    nomv = float(getattr(base, p))
                    specs[p] = (nomv * (1 - scan_frac), nomv * (1 + scan_frac), scan_npts)
            # Report the scan workload up front: this single Parallel pass over
            # (parameter, point, sample) tasks is the longest step of the UQ study.
            n_scan = sum(s[2] + 1 for s in specs.values()) * scan_n
            print(f"  Parameter feasibility scan: {len(specs)} params, ~{n_scan} runs "
                  f"(longest step; progress bar follows)...", flush=True)
            scans = FIG.scan_feasibility(input_file, specs, n_samples=scan_n, n_jobs=n_jobs)
            FIG.fig_scan(scans, save_dir=fig_dir)

        # copy of the input deck and synthetic summary, alongside the figures
        try:
            shutil.copy2(input_file, os.path.join(output_path, 'input_parameters.txt'))
        except Exception:
            pass
        _write_summary(os.path.join(output_path, 'uncertainty_summary.txt'),
                       input_file, results, controls, scans)
        print(f"  Output written to: {output_path}")

    return results


# =============================================================================
# Self-test
# =============================================================================
if __name__ == '__main__':
    UQ_FILE = 'D0FUS_INPUTS/4_uncertainty_ITER.txt'

    print(f"=== {UQ_FILE} ===")
    print(f"  detected mode: {detect_mode(UQ_FILE)}")
    base, spec, envelope, controls, deck_path = parse_uq_file(UQ_FILE)

    nom = evaluate(base)
    print(f"\n  Nominal design point: converged={nom['converged']} "
          f"feasible={nom['feasible']}  Q={nom['Q']:.3g}  "
          f"C_invest={nom['C_invest']:.3g} B EUR")
    print(f"  margins: greenwald={nom['gw_margin']:.3f} "
          f"troyon={nom['troyon_margin']:.3f} kink={nom['kink_margin']:.3f}")

    print("\n  Marginals auto-centred on the design point (lo, mode, hi):")
    for k in ['H', 'Tbar', 'q_limit', 'eta_WP_acad', 'n_ped_frac', 'betaN_limit']:
        print(f"    {k:14s} {spec[k]}")
    print(f"  envelope: {envelope}")
    print(f"  controls: {controls}")

    N = 64   # capped for the self-test
    names, X, results, controls = run_uq_from_file(
        UQ_FILE, n_override=N, n_jobs=2, verbose=0)
    print(f"\n=== Envelope propagation (parallel), N={N} per combo ===")
    for combo, rows in results.items():
        conv = [r for r in rows if r.get('converged')]
        feas = [r for r in conv if r['feasible']]
        p5, p50, p95 = _pct([r.get('C_invest') for r in feas])
        label = ', '.join(f"{k}={v}" for k, v in combo) if combo != ('nominal',) else 'nominal'
        print(f"  [{label}] converged {len(conv)}/{N}, feasible {len(feas)}/{len(conv)}; "
              f"C_invest P5/P50/P95 = {p5:.3g}/{p50:.3g}/{p95:.3g} B EUR")