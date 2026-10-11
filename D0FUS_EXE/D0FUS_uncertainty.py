"""
D0FUS_uncertainty.py -- Uncertainty quantification around a single design point.

Fourth execution mode (RUN / SCAN / OPTIMIZATION / UNCERTAINTY), self-contained
in the same spirit as the other input files: the user pastes a complete RUN deck
(the design point), then appends the parameters to study, written as
distributions or model envelopes, and the study controls (n_samples, seed,
retune, ...). These entries are recognised by their content, so the titles
[UNCERTAINTY] and [CONTROLS] are optional.

  * The design point provides the CENTRAL value of every uncertain law, so the
    study auto-centres on the machine and re-centres if the design changes.
  * A parameter is studied by writing it as a distribution
    tri(lo, hi) / tri(lo, mode, hi) / unif(lo, hi) / norm(...), or as a
    model-form list envelope(A | B). The normal law is a TRUNCATED normal, see
    _build_marginal for its forms. In norm(lo, hi) the centre is the design
    value and sigma = (hi - lo) / 4: the bounds sit at +/- 2 sigma from the
    midpoint of [lo, hi], so they are asymmetric in sigma units around a
    design value that is not the midpoint.
  * Every uncertain key is checked when the file is read: an unknown
    field, an index out of range or an invalid model option stops the study
    before any sample is run.

Building blocks:
  - evaluate(config): one in-memory solver call (no file I/O, no figures)
    returning the QoIs, the feasibility flag and the constraint margins.
  - triangular / normal / uniform LHS sampler.
  - parse_uq_file(): the input-file front end (the mode itself is detected
    by D0FUS.py).
  - run_uq_from_file(): forward propagation over the model envelope
    (Monte-Carlo evaluated in parallel with joblib/loky).
  - classify() / feasible_share(): the single definition of the feasible
    share used by the console, the summary file and the figures.

Feasibility is the shared RUN rule (D0FUS_run.evaluate_feasibility), so
"feasible" means exactly what RUN, SCAN and GENETIC mean.
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
from D0FUS_BIB.D0FUS_parameterization import coerce_input_value, resolve_deprecated_key

# Backwards-compatible alias for dataclasses.replace, used throughout the module.
dc_replace = replace


# Study controls, recognised by name anywhere in the deck. A key that is
# neither a control nor uncertain belongs to the design point.
KNOWN_CONTROLS = {
    'n_samples', 'seed',                                   # Monte-Carlo
    'retune', 'Tbar_window', 'retune_points', 'retune_bisect',   # Tbar retuning
    'cs_relief', 'flux_relief',                            # CS flux relief
    'pulse_retune', 'ramp_retune',                         # legacy relief switches
}

DIST_RE = re.compile(r'^(tri|norm|unif)\s*\((.*)\)$', re.IGNORECASE)
ENV_RE = re.compile(r'^envelope\s*\((.*)\)$', re.IGNORECASE)
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


    plasma_ok  = bool(np.isfinite(Q) and np.isfinite(Ip) and Ip > 0)
    closure_ok = bool(np.isfinite(cost))
    converged  = plasma_ok and closure_ok
    if not converged:
        return {'converged': False, 'feasible': False,
                'failure': 'no_operating_point' if not plasma_ok else 'no_closure'}

    # TF inboard leg and CS thicknesses from the run() tuple (r_sep - r_c would
    # also count the TF-CS gap in a wedged design)
    c_TF, d_CS = c, d
    f_bs = (Ib / Ip * 100.0) if Ip > 0 else np.nan

    # Cost and feasibility: shared RUN evaluators, identical in every mode.
    # nG already carries Greenwald_limit (and the density_limit_model), so the
    # density ratio is compared to 1, not to Greenwald_limit a second time.
    _cres = RUN.evaluate_costs(cfg, res)
    COE = _cres.get('COE', np.nan)
    C_invest = _cres['C_invest'] * 1e-3 if _cres else np.nan   # M EUR -> B EUR
    _fz = RUN.evaluate_feasibility(cfg, res)
    q_kink    = _fz['q_kink']
    build_ok  = _fz['build_ok']
    stable_ok = bool(all(np.isfinite(v) and v <= 1.0 for v in _fz['ratios'].values()))
    feasible  = _fz['feasible']
    gw_margin     = 1.0 - _fz['ratios']['density']
    troyon_margin = 1.0 - _fz['ratios']['beta']
    kink_margin   = 1.0 / _fz['ratios']['q'] - 1.0
    Ip_margin     = (1.0 - _fz['ratios']['Ip']) if 'Ip' in _fz['ratios'] else np.nan
    binding = {'density': 'greenwald', 'beta': 'troyon', 'q': 'kink',
               'Ip': 'Ip', 'radial_build': 'build'}.get(_fz['binding'], _fz['binding'])

    return {
        'converged': True, 'feasible': feasible,
        'build_ok': build_ok, 'stable_ok': stable_ok, 'binding': binding,
        'Q': Q, 'P_elec': P_elec, 'COE': COE, 'C_invest': C_invest,
        'f_bs': f_bs, 'beta_N': betaN, 'q95': q95, 'B0': B0, 'B_CS': B_CS,
        'P_sep': P_sep, 'd_TF': c_TF, 'd_CS': d_CS,
        'gw_margin': gw_margin, 'troyon_margin': troyon_margin, 'kink_margin': kink_margin,
        'Ip_margin': Ip_margin,
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

    The fractions grow geometrically, (2^(k+1) - 1) / (2^n - 1): 1/15, 3/15,
    7/15 and 1 for n = 4. The first rungs stay close to T0, so a narrow
    feasible band next to the design point is not jumped over (with an
    evenly spaced ladder on a 1-20 keV window, the first rung lies about
    3 keV away).

    Returns (up, down): the candidate temperatures above and below T0,
    ordered from the nearest to the farthest.
    """
    fractions = [(2 ** (k + 1) - 1) / (2 ** n_points - 1) for k in range(n_points)]
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

    The as-designed point (Tbar of cfg) is evaluated first; if it is feasible
    the sample is accepted unchanged. Otherwise the temperature is moved along
    the ladder of _retune_ladder, trying first the direction suggested by the
    failure signature: density-limit, beta-limit and radiative
    no-operating-point failures call for a HIGHER temperature (lower density
    at fixed fusion power), kink and flux-closure failures for a LOWER one;
    both directions are eventually tried. When a rung is feasible, the move
    is refined by bisection between that rung and the last infeasible point
    of the same direction (window[3] steps, default 3), so that Tbar_used is
    the smallest move found rather than the rung itself. If no temperature in
    the window is feasible, the best converged attempt is returned (so the
    margin statistics stay meaningful), tagged feasible=False. In every case
    the key 'feasible_as_designed' preserves the frozen-point verdict, so
    summaries can report both readings side by side.

    cfg must have its operating point resolved (Tbar_mode = 'manual'): with
    the Greenwald closure on, run() would solve Tbar again and a move of Tbar
    would have no effect. _uq_worker resolves it once per draw.
    """
    T_lo, T_hi = window[0], window[1]
    n_points = int(window[2]) if len(window) > 2 else 4
    n_bisect = int(window[3]) if len(window) > 3 else 3
    T0 = float(cfg.Tbar)
    r0 = evaluate(cfg)
    r0['feasible_as_designed'] = bool(r0.get('feasible'))
    r0['retuned'] = False
    r0['Tbar_used'] = T0
    if r0['feasible_as_designed']:
        return r0

    up_first = (r0.get('binding') in ('greenwald', 'troyon')
                or r0.get('failure') == 'no_operating_point')
    up, down = _retune_ladder(T0, T_lo, T_hi, n_points)
    first, second = (up, down) if up_first else (down, up)
    # Interleave the two directions, preferred one first, nearest steps first.
    steps = [t for pair in zip(first, second) for t in pair]
    steps += first[len(second):] + second[len(first):]

    tried = {round(T0, 3)}
    last_bad = {+1: T0, -1: T0}   # last infeasible temperature per direction
    best = r0 if r0.get('converged') else None
    for T in steps:
        key = round(float(T), 3)
        if key in tried:
            continue
        tried.add(key)
        direction = +1 if T > T0 else -1
        r = evaluate(dc_replace(cfg, Tbar=float(T)))
        r['feasible_as_designed'] = False
        r['Tbar_used'] = T
        if r.get('feasible'):
            # Bisection between the last infeasible point and this rung
            lo, hi = last_bad[direction], T
            for _ in range(n_bisect):
                Tm = 0.5 * (lo + hi)
                rm = evaluate(dc_replace(cfg, Tbar=float(Tm)))
                if rm.get('feasible'):
                    hi = Tm
                    rm['feasible_as_designed'] = False
                    rm['Tbar_used'] = Tm
                    r = rm
                else:
                    lo = Tm
            r['retuned'] = True
            return r
        last_bad[direction] = T
        if best is None and r.get('converged'):
            r['retuned'] = False
            best = r
    if best is None:
        best = r0
    best.setdefault('retuned', False)
    return best


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
    if str(raw).strip().lower() in ('none', 'off', ''):
        return None
    fractions = [float(x) for x in str(raw).split(',')]
    if not all(0.0 < f < 1.0 for f in fractions):
        raise ValueError(f"[CONTROLS] cs_relief = {raw}: fractions must lie in (0, 1).")
    return fractions


def parse_retune_controls(controls):
    """Read the optional operator-retuning controls.

    [CONTROLS] keys:
      retune        = Tbar          activates the retuning search
      Tbar_window   = 6.0, 12.0     admissible operating window [keV]
      retune_points = 4             ladder resolution per direction (optional)
      retune_bisect = 3             bisection steps refining a feasible rung
                                    (optional, 0 keeps the rung itself)
    Returns (T_lo, T_hi, n_points, n_bisect) or None when retuning is off. The
    ladder is built relative to the window (see _retune_ladder), so the window
    bounds are always reachable whatever the deck's design temperature.
    """
    value = str(controls.get('retune', '')).strip().lower()
    if value in ('', 'none', 'off'):
        return None
    if value != 'tbar':
        raise ValueError(f"[CONTROLS] retune = {controls.get('retune')!r}: "
                         "the only retuning lever is Tbar (or none).")
    win = str(controls.get('Tbar_window', '6.0, 12.0'))
    lo, hi = (float(t) for t in win.split(','))
    if not lo < hi:
        raise ValueError(f"[CONTROLS] Tbar_window = {win}: needs lo < hi.")
    return (lo, hi, int(controls.get('retune_points', 4)),
            int(controls.get('retune_bisect', 3)))


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
                # Envelope options arrive as deck strings and get the deck
                # typing. Numeric values (scanned design parameters) are
                # kept as they are: casting them to the type of the deck
                # value would truncate a scan of 'a = 2.0' (read as int 2)
                # to whole metres.
                changes[k] = coerce_input_value(k, v) if isinstance(v, str) else v
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
        # For the 2/3-argument forms sigma = (hi - lo) / 4: lo and hi sit at
        # +/- 2 sigma from the MIDPOINT of [lo, hi]. Around a centre that is not
        # the midpoint they are asymmetric in sigma units (H = norm(0.80, 1.60)
        # centred on 1.0: lo at -1 sigma, hi at +3 sigma). Use the 4th argument
        # to decouple the width from the bounds (e.g. a narrow peak with a
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


def _check_uncertain_key(base, key):
    """
    Resolve and check one sampled [UNCERTAINTY] key against the design config.

    A plain key must be a numeric GlobalConfig field (deprecated aliases are
    resolved). An indexed key 'field[i]' must point to entry i of a
    comma-separated field (e.g. f_imp_core[0]). Returns the resolved key.
    """
    fields = base.__dataclass_fields__
    m = IDX_RE.match(key)
    if m is not None:
        fname = resolve_deprecated_key(m.group(1))
        raw = getattr(base, fname, None) if fname in fields else None
        if not isinstance(raw, str):
            raise ValueError(f"[UNCERTAINTY] '{key}': '{fname}' is not a "
                             "comma-separated GlobalConfig field.")
        n_entries = len(raw.split(','))
        if int(m.group(2)) >= n_entries:
            raise ValueError(f"[UNCERTAINTY] '{key}': index {m.group(2)} out of "
                             f"range, '{fname}' has {n_entries} entries in the deck.")
        return f"{fname}[{m.group(2)}]"
    name = resolve_deprecated_key(key)
    if name not in fields:
        import difflib
        close = difflib.get_close_matches(name, fields.keys(), n=3, cutoff=0.6)
        raise ValueError(f"[UNCERTAINTY] '{key}' is not a GlobalConfig field. "
                         f"Closest fields: {close}")
    if isinstance(getattr(base, name), (bool, str)):
        raise ValueError(f"[UNCERTAINTY] '{key}' is not numeric: use "
                         "envelope(A | B) for a model choice.")
    return name


def parse_uq_file(path):
    """
    Parse a self-contained UNCERTAINTY file.

    Each line is classified by its content, with the rule of the mode
    detection (D0FUS_run.deck_entries): a tri(), norm(), unif() or envelope()
    value is an uncertain input, a key of KNOWN_CONTROLS a study control, any
    other line belongs to the design point and is read in memory with the
    RUN deck parser (no temporary file). The titles [UNCERTAINTY] and
    [CONTROLS] are accepted and ignored. Every uncertain key is checked
    (field name, numeric type, index range, envelope options), so that a
    typo stops the study instead of drawing an input that has no effect.

    Returns base (GlobalConfig built from the design point), spec (dict of
    centred marginals), envelope (dict of model-option lists) and controls
    (dict).
    """
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().splitlines()
    mode, _ = RUN.detect_mode_from_lines(lines)
    if mode != 'uncertainty':
        raise ValueError(f"{path} has no uncertain entry: an UNCERTAINTY study "
                         "needs at least one tri(), norm(), unif() or "
                         "envelope() value.")

    # Each entry is classified by its content: an uncertain value, a study
    # control (known by name), or a design-point line for the RUN parser.
    design, raw_spec, controls = [], {}, {}
    for key, kind, rhs in RUN.deck_entries(lines):
        if kind == 'uncertain':
            if key in raw_spec:
                raise ValueError(f"[UNCERTAINTY] '{key}' is defined twice in {path}.")
            raw_spec[key] = rhs
        elif key in KNOWN_CONTROLS:
            if key in controls:
                raise ValueError(f"[CONTROLS] '{key}' is defined twice in {path}.")
            controls[key] = int(rhs) if rhs.lstrip('-').isdigit() else rhs
        else:
            design.append(f"{key} = {rhs}")

    # Split the uncertain entries into model envelopes and distributions
    env_rhs, dist_rhs = {}, {}
    for key, rhs in raw_spec.items():
        if ENV_RE.match(rhs):
            env_rhs[key] = rhs
        elif DIST_RE.match(rhs):
            dist_rhs[key] = rhs
        else:
            raise ValueError(f"[UNCERTAINTY] '{key} = {rhs}': expected "
                             "tri(...), norm(...), unif(...) or envelope(...).")

    base = RUN.load_config_from_lines(design, verbose=0)

    spec, envelope = {}, {}
    for key, rhs in env_rhs.items():
        name = resolve_deprecated_key(key)
        if name not in base.__dataclass_fields__:
            raise ValueError(f"[UNCERTAINTY] envelope key '{key}' is not a "
                             "GlobalConfig field.")
        options = [o.strip() for o in ENV_RE.match(rhs).group(1).split('|')]
        for opt in options:
            try:
                dc_replace(base, **{name: coerce_input_value(name, opt)})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"[UNCERTAINTY] {name} = envelope(...): option "
                                 f"'{opt}' is not valid ({exc}).") from None
        envelope[name] = options
    for key, rhs in dist_rhs.items():
        name = _check_uncertain_key(base, key)
        dist = DIST_RE.match(rhs)
        fam = dist.group(1).lower()
        args = [float(a) for a in dist.group(2).split(',')]
        spec[name] = _build_marginal(name, fam, args, design_value(base, name))
    return base, spec, envelope, controls


# =============================================================================
# Forward propagation
# =============================================================================
def _uq_worker(base_dict, names, row, combo, retune=None, relief=None):
    """
    Worker for the parallel Monte-Carlo. Every dependency is imported LOCALLY.

    Local-only imports stop cloudpickle from serialising this module's global
    namespace. When D0FUS.py is launched with %runfile in Spyder, __main__ is
    polluted with scipy symbols (erfc, ...); without local imports the loky workers
    fail to un-serialise the task ("Can't get attribute 'erfc' on __main__"). Every
    argument is a plain type (dict of the base configuration, list of names, numpy
    row, dict), so the task is always picklable regardless of how the parent
    imported the modules. This mirrors the scan worker (_run_scan_point).

    The Greenwald closure of the draw (Tbar_mode or P_fus_mode = 'greenwald')
    is solved once here. The retuning and relief ladders then move a resolved
    operating point instead of solving it again at every step.
    """
    import os
    import sys
    import warnings
    _parent = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from D0FUS_BIB.D0FUS_parameterization import GlobalConfig
    from D0FUS_EXE.D0FUS_run import resolve_operating_point
    from D0FUS_EXE.D0FUS_uncertainty import (build_config, evaluate,
                                             evaluate_retuned, evaluate_relief)
    cfg = build_config(GlobalConfig(**base_dict), names, row, combo)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cfg = resolve_operating_point(cfg)
    except ValueError:
        # No temperature (or fusion power) reaches the target Greenwald fraction
        return {'converged': False, 'feasible': False, 'failure': 'no_operating_point'}
    except Exception:
        return {'converged': False, 'feasible': False, 'failure': 'crash'}
    if relief is not None:
        return evaluate_relief(cfg, retune, relief)
    if retune is not None:
        return evaluate_retuned(cfg, retune)
    return evaluate(cfg)


def model_combos(envelope):
    """All combinations of the model-form envelopes (one empty combo if none)."""
    if not envelope:
        return [{}]
    keys = list(envelope.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*envelope.values())]


def run_uq_from_file(path, n_override=None, n_jobs=-1, verbose=5, seed=None):
    """
    Parse a self-contained UNCERTAINTY file and propagate over the model envelope.
    seed, when given, overrides the seed of [CONTROLS].

    The Monte-Carlo is evaluated in parallel with joblib/loky. On Windows the
    calling script MUST be guarded by  if __name__ == '__main__':  for the loky
    backend to spawn workers.

    Returns names, X, results (combo -> rows), controls, and the parsed
    (base, spec, envelope) so that callers do not read the file again.
    """
    from joblib import Parallel, delayed

    base, spec, envelope, controls = parse_uq_file(path)
    if seed is not None:
        controls['seed'] = int(seed)
    n = n_override or controls.get('n_samples', 1000)
    retune = parse_retune_controls(controls)
    relief = parse_relief_controls(controls)
    if retune is not None and 'Tbar' in spec:
        raise ValueError("Tbar is both sampled in [UNCERTAINTY] and retuned by "
                         "the operator (retune = Tbar): keep one of the two.")
    names, X = sample_lhs(spec, n, seed=controls.get('seed', 0))
    combos = model_combos(envelope)
    base_dict = asdict(base)

    # Flatten (combo, sample) into one task list for balanced core utilisation.
    flat = [(combo, X[i]) for combo in combos for i in range(n)]
    # One updating tqdm bar instead of joblib's per-batch log lines. return_as
    # 'generator' preserves submission order, so the index-based slicing below
    # that maps results back to each model combo stays valid.
    _gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_uq_worker)(base_dict, names, row, combo, retune, relief)
        for combo, row in flat)
    out = list(tqdm(_gen, total=len(flat), desc="UQ Monte-Carlo",
                    unit="run", disable=(verbose == 0)))

    results = {}
    for c_idx, combo in enumerate(combos):
        key = tuple(sorted(combo.items())) if combo else ('nominal',)
        results[key] = out[c_idx * n:(c_idx + 1) * n]
    return names, X, results, controls, (base, spec, envelope)


def _pct(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(a, [5, 50, 95]))


# =============================================================================
# Single definition of the outcome of a sample and of the feasible share
# =============================================================================
def classify(r):
    """
    Outcome class of one sample, shared by the console, the summary file and
    the figures:
      'feasible_as_designed' : feasible at the deck operating point
      'feasible_retuned'     : feasible after operator Tbar retuning
      'feasible_relieved'    : feasible only with CS flux relief (H&CD ramp
                               assist and/or a shorter flat-top)
      'stability'            : a plasma limit (kink, Greenwald, Troyon, Ip) binds
      'radial_build'         : the radial build does not close, even with relief
      'no_operating_point'   : the plasma solver found no solution
      'crash'                : the solver raised
    """
    if r.get('feasible'):
        if r.get('build_relieved'):
            return 'feasible_relieved'
        if r.get('retuned') and not r.get('feasible_as_designed', False):
            return 'feasible_retuned'
        return 'feasible_as_designed'
    if not r.get('converged'):
        f = r.get('failure')
        if f == 'no_closure':
            return 'radial_build'
        return 'crash' if f == 'crash' else 'no_operating_point'
    if r.get('category') == 'radial_build' or r.get('binding') == 'build':
        return 'radial_build'
    return 'stability'


# Feasible classes counted at each level of recourse
_LEVELS = {
    'as_designed': ('feasible_as_designed',),
    'window':      ('feasible_as_designed', 'feasible_retuned'),
    'relief':      ('feasible_as_designed', 'feasible_retuned', 'feasible_relieved'),
}


def wilson_interval(k, n, z=1.96):
    """Wilson score interval (default 95 %) of a proportion k / n, in [0, 1]."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def feasible_share(rows, level='window'):
    """
    Share of feasible samples over ALL samples (no-solution draws counted as
    failures), with its 95 % Wilson interval.

    level : 'as_designed' (frozen operating point), 'window' (plus operator
            Tbar retuning, the headline of the study) or 'relief' (plus CS
            flux relief, reported separately because it changes the pulse).

    Returns (share, lo, hi) in percent, and the sample count.
    """
    n = len(rows)
    k = sum(1 for r in rows if classify(r) in _LEVELS[level])
    lo, hi = wilson_interval(k, n)
    return (100.0 * k / max(n, 1), 100.0 * lo, 100.0 * hi), n


def verdict_of(share):
    """Verdict on the headline share [%]."""
    return ('LARGELY FEASIBLE' if share >= 85 else
            'MARGINAL' if share >= 60 else 'AT RISK')


# =============================================================================
# Entry point for the UNCERTAINTY mode (called by D0FUS.py)
# =============================================================================
def summarize_results(results):
    """Return (n_total, n_converged, n_feasible, binding_counter, failure_counter).

    n_feasible counts the headline 'window' level (as designed or after
    Tbar retuning, see classify). binding counts the most-violated limit among
    CONVERGED samples that are not feasible at any level (the draws feasible
    with CS flux relief are reported apart). failures counts the
    non-convergence categories ('no_operating_point', 'no_closure', 'crash')
    so that 'this draw has no solution' is reported separately from 'this
    draw violates an operational limit'.
    """
    all_rows = [r for k in results for r in results[k]]
    conv = [r for r in all_rows if r.get('converged')]
    feas = [r for r in all_rows if classify(r) in _LEVELS['window']]
    binding = Counter(r.get('binding') for r in conv
                      if classify(r) in ('stability', 'radial_build'))
    failures = Counter(r.get('failure', 'unknown')
                       for r in all_rows if not r.get('converged'))
    return len(all_rows), len(conv), len(feas), binding, failures


def _share_line(rows, level):
    (p, lo, hi), n = feasible_share(rows, level)
    return f"{p:.0f}% (95% CI {lo:.0f}-{hi:.0f}%)"


def _write_summary(path, input_file, results, controls, scans=None):
    """Write a concise human-readable summary of the uncertainty study."""
    n, n_conv, n_feas, binding, failures = summarize_results(results)
    all_rows = [r for k in results for r in results[k]]
    conv = [r for r in all_rows if r.get('converged')]
    # The headline verdict is read over ALL samples, the no-solution draws
    # counted as failures (thesis convention, Chapter 4), at the 'window'
    # level: as designed or after operator Tbar retuning.
    (p_win, _, _), _ = feasible_share(all_rows, 'window')
    verdict = verdict_of(p_win)

    def pct(key):
        a = np.array([r[key] for r in conv if np.isfinite(r.get(key, np.nan))])
        return tuple(np.percentile(a, [5, 50, 95])) if a.size else (np.nan, np.nan, np.nan)

    fail_txt = ", ".join(f"{k}={v}" for k, v in failures.items()) or "none"
    L = ["D0FUS uncertainty study summary",
         f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         f"Input file : {os.path.basename(input_file)}",
         f"Seed       : {controls.get('seed', 0)}",
         "",
         f"Samples    : {n} total = {n_conv} converged + "
         f"{n - n_conv} without a solution ({fail_txt})",
         f"Feasible   : {_share_line(all_rows, 'window')} of all samples, "
         f"as designed or after Tbar retuning",
         f"Verdict    : {verdict}  (thresholds 85% / 60% on the share above)"]
    if any('feasible_as_designed' in r for r in all_rows):
        L.append(f"  as designed (frozen operating point)    : "
                 f"{_share_line(all_rows, 'as_designed')}")
    if any(classify(r) == 'feasible_relieved' for r in all_rows):
        L.append(f"  including the draws that close only with CS flux relief "
                 f"(ramp assist and/or shorter pulse): {_share_line(all_rows, 'relief')}")
    if binding:
        L.append("Binding limit among converged non-feasible samples: "
                 + ", ".join(f"{k}={v}" for k, v in binding.items()))
    L += ["",
          "Headroom to each limit (normalised margin, P5 / P50 / P95):"]
    for key, name in [('gw_margin', 'Greenwald'), ('troyon_margin', 'Troyon'),
                      ('kink_margin', 'Kink (q95)')]:
        p5, p50, p95 = pct(key)
        L.append(f"  {name:12s}: {p5:+.3f} / {p50:+.3f} / {p95:+.3f}")

    combos = [k for k in results if k != ('nominal',)]
    if combos:
        L += ["", "Feasible share by model combination (as designed or after Tbar retuning):"]
        for k in combos:
            label = " · ".join(str(v) for _, v in k)
            L.append(f"  {label:34s}: {_share_line(results[k], 'window')}")

    if scans:
        L += ["", "Feasible share AS DESIGNED (no retuning, model combos cycled) "
              "at the design value of each scanned parameter:"]
        for p, (xs, pf, nomv, n_s) in scans.items():
            k = int(round(float(np.interp(nomv, xs, pf)) * n_s / 100.0))
            lo, hi = wilson_interval(k, n_s)
            L.append(f"  {p:8s} = {nomv:<8g}: {float(np.interp(nomv, xs, pf)):.0f}% "
                     f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%, {n_s} samples)")

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(L) + "\n")


def main(input_file, save_figures=True, output_dir=None, n_override=None,
         scan=True, scan_npts=11, scan_n=40, scan_frac=0.4, n_jobs=-1,
         seed=None):
    """
    Run the full uncertainty study for an input file. Like the other modes,
    it writes a timestamped folder under <output_dir>/uncertainty/ with a
    copy of the input deck, a synthetic summary and, when save_figures is
    True, the figures and the parameter feasibility scan.

    output_dir : parent folder of the uncertainty/ result folders. Default:
                 default_output_root() of D0FUS_run.
    seed       : sampling seed. Overrides the seed of [CONTROLS].
    """
    if output_dir is None:
        output_dir = RUN.default_output_root()

    names, X, results, controls, (base, spec, envelope) = run_uq_from_file(
        input_file, n_override=n_override, n_jobs=n_jobs, verbose=5, seed=seed)

    n, n_conv, n_feas, binding, failures = summarize_results(results)
    all_rows = [r for k in results for r in results[k]]
    (p_win, lo, hi), _ = feasible_share(all_rows, 'window')
    print(f"\n  UNCERTAINTY verdict: {p_win:.0f}% of all {n} samples feasible "
          f"(95% CI {lo:.0f}-{hi:.0f}%), as designed or after Tbar retuning  "
          f"->  {verdict_of(p_win)}")
    if any(classify(r) == 'feasible_relieved' for r in all_rows):
        print(f"  Including the draws that close only with CS flux relief: "
              f"{_share_line(all_rows, 'relief')}")
    if failures:
        print("  Samples without a solution: "
              + ', '.join(f'{k}={v}' for k, v in failures.items()))
    if binding:
        print("  Binding limit among converged non-feasible samples: "
              + ', '.join(f'{k}={v}' for k, v in binding.items()))

    output_path, _ = RUN.new_output_dir(os.path.join(output_dir, 'uncertainty'),
                                        "Uncertainty_D0FUS")
    scans = None
    if save_figures:
        from D0FUS_BIB import D0FUS_figures as FIG
        fig_dir = os.path.join(output_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

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
            scans = FIG.scan_feasibility(input_file, specs, n_samples=scan_n, n_jobs=n_jobs,
                                         seed=controls.get('seed', 0))
            (p_des, lo_des, hi_des), _ = feasible_share(all_rows, 'as_designed')
            FIG.fig_scan(scans, save_dir=fig_dir,
                         reference=(p_des, lo_des, hi_des))

    # copy of the input deck and synthetic summary, alongside the figures
    shutil.copy2(input_file, os.path.join(output_path, 'input_parameters.txt'))
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
    base, spec, envelope, controls = parse_uq_file(UQ_FILE)

    nom = evaluate(RUN.resolve_operating_point(base))
    print(f"\n  Nominal design point: converged={nom['converged']} "
          f"feasible={nom['feasible']}  Q={nom['Q']:.3g}  "
          f"C_invest={nom['C_invest']:.3g} B EUR")
    print(f"  margins: greenwald={nom['gw_margin']:.3f} "
          f"troyon={nom['troyon_margin']:.3f} kink={nom['kink_margin']:.3f}")

    print("\n  Marginals auto-centred on the design point:")
    for k in spec:
        print(f"    {k:18s} {spec[k]}")
    print(f"  envelope: {envelope}")
    print(f"  controls: {controls}")

    N = 64   # capped for the self-test
    names, X, results, controls, _ = run_uq_from_file(
        UQ_FILE, n_override=N, n_jobs=2, verbose=0)
    print(f"\n=== Envelope propagation (parallel), N={N} per combo ===")
    for combo, rows in results.items():
        feas = [r for r in rows if classify(r) in _LEVELS['window']]
        p5, p50, p95 = _pct([r.get('C_invest') for r in feas])
        label = ', '.join(f"{k}={v}" for k, v in combo) if combo != ('nominal',) else 'nominal'
        print(f"  [{label}] feasible {_share_line(rows, 'window')}; "
              f"C_invest P5/P50/P95 = {p5:.3g}/{p50:.3g}/{p95:.3g} B EUR")
