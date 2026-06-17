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
    """Run one configuration in memory and return QoIs plus feasibility."""
    try:
        res = RUN.run(cfg, verbose=0)
    except Exception:
        return {'converged': False, 'feasible': False}

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

    converged = bool(np.isfinite(Q) and np.isfinite(cost)
                     and np.isfinite(Ip) and Ip > 0)
    if not converged:
        return {'converged': False, 'feasible': False}

    c_TF = r_sep - r_c if np.isfinite(r_c) and np.isfinite(r_sep) else np.nan
    d_CS = r_c - r_d   if np.isfinite(r_c) and np.isfinite(r_d)   else np.nan
    f_bs = (Ib / Ip * 100.0) if Ip > 0 else np.nan
    gw   = (nbar_line / nG) if nG > 0 else np.nan
    COE, C_invest = _compute_cost(cfg, P_CD, P_elec, Gamma_n, Surface, c, d, kappa,
                                  _T_op_limit, _CF, _t_bl_yr, _t_div_yr, _V_rb_BB)

    q_kink = q95 if cfg.kink_parameter == 'q95' else qstar

    build_ok  = _radial_build_ok(cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line)
    gw_ok     = bool(np.isfinite(gw)    and gw    <= cfg.Greenwald_limit)
    troyon_ok = bool(np.isfinite(betaN) and betaN <= cfg.betaN_limit)
    kink_ok   = bool(np.isfinite(q_kink) and q_kink >= cfg.q_limit)
    stable_ok = bool(gw_ok and troyon_ok and kink_ok)
    feasible  = bool(build_ok and stable_ok)

    gw_margin     = (1.0 - gw / cfg.Greenwald_limit) if np.isfinite(gw)    else np.nan
    troyon_margin = (1.0 - betaN / cfg.betaN_limit)  if np.isfinite(betaN) else np.nan
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
# Sampling
# =============================================================================
def _triangular_ppf(u, lo, mode, hi):
    if hi <= lo:
        return np.full_like(u, lo, dtype=float)
    c = min(max((mode - lo) / (hi - lo), 0.0), 1.0)
    return stats.triang.ppf(u, c=c, loc=lo, scale=(hi - lo))


def _marginal_ppf(u, dist):
    """Inverse CDF of one marginal evaluated on u in [0, 1]."""
    fam = dist[0]
    if fam == 'tri':
        _, lo, mode, hi = dist
        return _triangular_ppf(u, lo, mode, hi)
    if fam == 'norm':
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
        #   norm(sigma)            -> mean = design value, unbounded
        #   norm(lo, hi)           -> mean = design value (or midpoint if undefined)
        #   norm(lo, centre, hi)   -> mean = centre, bounds lo/hi
        # In every bounded case the standard deviation is set so that lo and hi sit
        # at about mean +/- 2 sigma, i.e. sigma = (hi - lo) / 4.
        if len(args) == 1:                       # (sigma) -> mean = design value
            if central is None:
                raise ValueError(f"norm() for '{name}': sigma-only form needs a "
                                 f"design value to centre on")
            return ('norm', central, args[0])
        if len(args) == 2:                       # (lo, hi) -> mean = design value
            lo, hi = args
            mu = central if central is not None else 0.5 * (lo + hi)
        elif len(args) == 3:                     # (lo, centre, hi) explicit
            lo, mu, hi = args
        else:
            raise ValueError(f"norm() for '{name}' expects 1, 2 or 3 arguments")
        if not (lo <= mu <= hi):
            print(f"  [UQ] warning: '{name}' centre {mu:g} outside "
                  f"[{lo:g}, {hi:g}] -> clamped.")
            mu = min(max(mu, lo), hi)
        return ('norm', mu, (hi - lo) / 4.0, lo, hi)
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
        spec[key] = _build_marginal(key, fam, args, getattr(base, key, None))
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


def _uq_worker(deck_path, names, row, combo):
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
    from D0FUS_EXE.D0FUS_uncertainty import _load_base_cached, build_config, evaluate
    return evaluate(build_config(_load_base_cached(deck_path), names, row, combo))


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
        delayed(_uq_worker)(deck_path, names, row, combo) for combo, row in flat)
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
    """Return (n_total, n_converged, n_feasible, binding_counter)."""
    all_rows = [r for k in results for r in results[k]]
    conv = [r for r in all_rows if r.get('converged')]
    feas = [r for r in conv if r.get('feasible')]
    binding = Counter(r.get('binding') for r in conv if not r.get('feasible'))
    return len(all_rows), len(conv), len(feas), binding


def _write_summary(path, input_file, results, controls, scans=None):
    """Write a concise human-readable summary of the uncertainty study."""
    n, n_conv, n_feas, binding = summarize_results(results)
    p_feas = 100.0 * n_feas / max(n, 1)
    verdict = ('LARGELY FEASIBLE' if p_feas >= 85 else
               'MARGINAL' if p_feas >= 60 else 'AT RISK')
    conv = [r for k in results for r in results[k] if r.get('converged')]

    def pct(key):
        a = np.array([r[key] for r in conv if np.isfinite(r.get(key, np.nan))])
        return tuple(np.percentile(a, [5, 50, 95])) if a.size else (np.nan, np.nan, np.nan)

    L = ["D0FUS uncertainty study summary",
         f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         f"Input file : {os.path.basename(input_file)}",
         "",
         f"Samples    : {n} total ({n_conv} converged)",
         f"Verdict    : {p_feas:.0f}% feasible  ->  {verdict}"]
    if binding:
        L.append("Binding limit among infeasible: "
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

    n, n_conv, n_feas, binding = summarize_results(results)
    p_feas = 100.0 * n_feas / max(n, 1)
    verdict = ('LARGELY FEASIBLE' if p_feas >= 85 else
               'MARGINAL' if p_feas >= 60 else 'AT RISK')
    print(f"\n  UNCERTAINTY verdict: {p_feas:.0f}% feasible over {n} samples "
          f"({n_conv} converged)  ->  {verdict}")
    if binding:
        print("  Binding limit among infeasible: "
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