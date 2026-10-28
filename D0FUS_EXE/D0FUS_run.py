"""
D0FUS Run Module
=================
Computes a single tokamak design point and writes the full output report.

The module is driven by a GlobalConfig instance (see D0FUS_parameterization.py),
which centralises every input parameter with physically motivated default values.
Partial overrides are achieved via dataclasses.replace(), making the interface
well-suited for parametric scans and genetic-algorithm optimisation.

"""

#%% Standard imports

# Bootstrap imports only: sys/os are required to make the package importable
# before D0FUS_import.py (which centralises every other import) is reachable.
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# D0FUS modules
from D0FUS_BIB.D0FUS_parameterization import (
    GlobalConfig, DEFAULT_CONFIG, coerce_input_value, resolve_deprecated_key,
    LEGACY_SPLIT_KEYS, apply_legacy_split_keys, plant_conversion)
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_BIB.D0FUS_cost_functions import f_costs_Sheffield
from D0FUS_BIB.D0FUS_cost_data import *

# Backwards-compatible alias for dataclasses.replace (exported by D0FUS_import
# through the wildcard chain above), used throughout this module.
dc_replace = replace


#%% Profile preset table
# ---------------------------------------------------------------------------
# plasma profile peaking factors and pedestal
# parameters across run(), _build_run_dict(), and save_run_output().
#
# Calibration references:
#   'L'        : purely parabolic (no pedestal).
#   'H'        : CORSICA simulations of the 15 MA ITER baseline (Kim 2018)
#   'Advanced' : improved-confinement scenarios (peaked profiles)
#   'EU-DEMO'  : EU-DEMO 2017 PROCESS reference profile set
#   'NT'       : negative triangularity, L-mode edge (no pedestal), parabolic
#                profiles fitted on DIII-D NT discharges (Marinoni 2019, Austin
#                2019): n0/<n> = 1.55, T0/<T> = 2.80 in any geometry
# ---------------------------------------------------------------------------
_PROFILE_PRESETS = {
    'L':        {'nu_n': 0.50, 'nu_T': 1.00, 'rho_ped': 1.00,
                 'n_ped_frac': 0.00, 'T_ped_frac': 0.00},
    'H':        {'nu_n': 0.01, 'nu_T': 2.80, 'rho_ped': 0.95,
                 'n_ped_frac': 0.99, 'T_ped_frac': 0.55},
    'Advanced': {'nu_n': 1.50, 'nu_T': 2.00, 'rho_ped': 0.96,
                 'n_ped_frac': 0.95, 'T_ped_frac': 0.55},
    'EU-DEMO': {'nu_n': 1.00, 'nu_T': 1.45, 'rho_ped': 0.95,
                 'n_ped_frac': 0.78, 'T_ped_frac': 0.43},
    'NT':       {'nu_n': 0.55, 'nu_T': 1.80, 'rho_ped': 1.00,
                 'n_ped_frac': 0.00, 'T_ped_frac': 0.00},
}


#%% Input file loader

# (legacy-value remap layer removed; see loader comment)


def load_config_from_lines(lines,
                           base: GlobalConfig = None,
                           verbose: int = 0,
                           ignore_keys=()) -> GlobalConfig:
    """
    Build a GlobalConfig from the lines of a D0FUS deck.

    Same rules as load_config_from_file, for a deck already held in memory
    (for instance the design section of a POPCON or UNCERTAINTY deck), so
    that no temporary file is needed.

    Parameters
    ----------
    lines : iterable of str
        Deck lines. Each non-empty, non-comment line must follow the pattern
        ``key = value``. Lines whose value is a bracketed list
        (e.g. ``R0 = [8, 9, 10]``) are treated as scan declarations and are
        skipped: the field keeps its default value.
    base, verbose, ignore_keys : see load_config_from_file.

    Returns
    -------
    GlobalConfig
        New immutable configuration with the deck overrides applied.
    """
    if base is None:
        base = DEFAULT_CONFIG

    overrides = {}

    for raw_line in lines:
        # Strip inline comments and surrounding whitespace
        line = raw_line.split("#")[0].strip()
        if not line or "=" not in line:
            continue

        key, _, raw_val = line.partition("=")
        key     = key.strip()
        raw_val = raw_val.strip()

        # Skip scan parameters (bracketed lists)
        if raw_val.startswith("[") and raw_val.endswith("]"):
            if verbose >= 2:
                print(f"  [scan]   {key} = {raw_val}  →  skipped (kept at default)")
            continue

        # Coerce the raw token to the correct Python type for this
        # GlobalConfig field via the shared helper (single source of
        # truth across the run, genetic and scan loaders). A boolean
        # field written as "False" therefore decodes to the bool False
        # rather than the truthy string "False", and an optional field
        # written as "None" decodes to the Python None.
        # Resolve deprecated deck keys (f_In -> f_In_cable, f_gap ->
        # f_In_WP) before typing and before the GlobalConfig membership
        # check, so an old deck warns instead of silently falling back to
        # the field default.
        key = resolve_deprecated_key(key)
        if key in ignore_keys:
            continue
        val = coerce_input_value(key, raw_val)

        # Old single keys split per coil system: expanded after the loop,
        # so that a TF or CS key set anywhere in the deck takes precedence.
        if key in LEGACY_SPLIT_KEYS:
            overrides[key] = val
            continue

        # Only accept keys that exist in GlobalConfig
        if key not in GlobalConfig.__dataclass_fields__:
            # Always surface unknown keys: a stale deck silently
            # falling back to defaults is exactly the hidden-
            # assumption failure mode D0FUS aims to avoid.
            print(f"  [warn]   '{key}' is not a recognised GlobalConfig field — ignored")
            continue

        # Legacy-value remapping removed by design: an unrecognised
        # option value must fail loudly in the downstream validator
        # (which lists the valid options) rather than being silently
        # translated. Unknown KEYS still warn and fall back to the
        # documented defaults.
        overrides[key] = val
        if verbose >= 2:
            print(f"  [input]  {key} = {overrides[key]}")

    return dc_replace(base, **apply_legacy_split_keys(overrides))


def load_config_from_file(filepath: str,
                          base: GlobalConfig = None,
                          verbose: int = 0,
                          ignore_keys=()) -> GlobalConfig:
    """
    Parse a D0FUS input file and return a GlobalConfig with overridden fields.

    Parameters
    ----------
    filepath : str
        Path to the input file.  Each non-empty, non-comment line must follow
        the pattern ``key = value``.  Lines whose value is a bracketed list
        (e.g. ``R0 = [8, 9, 10]``) are treated as scan declarations and are
        silently skipped — the field keeps its default value.
    base : GlobalConfig, optional
        Starting configuration.  If None, DEFAULT_CONFIG is used.
    verbose : int, optional
        Verbosity level: 0 = silent, 1 = warnings only, 2 = full trace.
        Default: 0.
    ignore_keys : iterable of str, optional
        Deck keys read by the calling mode itself (for instance the SCAN
        plot settings). They are skipped without the unknown-key warning.

    Returns
    -------
    GlobalConfig
        New immutable configuration with the file overrides applied.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        return load_config_from_lines(fh, base=base, verbose=verbose,
                                      ignore_keys=ignore_keys)


#%% Deck content and execution mode
# ---------------------------------------------------------------------------
# The execution mode follows from the content of the deck, never from a
# section title. tri(), norm(), unif() or envelope() values make an
# UNCERTAINTY study. Two [min, max, n_points] axes nbar_line and Tbar make a
# POPCON map, two other [min, max, n_points] axes a SCAN. [min, max] ranges
# make a GENETIC optimisation. A deck with none of them is a RUN. Section
# titles such as [POPCON] are accepted and ignored. D0FUS.py and the mode
# parsers use the same functions, so they cannot disagree on a deck.
# ---------------------------------------------------------------------------

# Grid axes of a POPCON map. nbar_line is not a design input, so a
# [min, max, n_points] axis on it can only belong to a POPCON map.
POPCON_AXES = ('nbar_line', 'Tbar')

_UNCERTAIN_RE = re.compile(r'^(tri|norm|unif|envelope)\s*\(', re.IGNORECASE)


def deck_entries(lines):
    """
    Entries of a D0FUS deck, in file order.

    Comments (text after #) are removed. Each 'key = value' line, and each
    'key : [...]' line, gives a tuple (key, kind, value) with kind
      'grid'       value [min, max, n_points]
      'range'      value [min, max]
      'uncertain'  value tri(...), norm(...), unif(...) or envelope(...)
      'scalar'     any other value
    Lines without an entry (blank lines, section titles such as [POPCON])
    are skipped.

    Raises
    ------
    ValueError : bracketed value with neither 2 nor 3 items.
    """
    entries = []
    for raw in lines:
        line = raw.split('#', 1)[0].strip()
        if '=' in line:
            key, _, value = line.partition('=')
        elif ':' in line and line.partition(':')[2].strip().startswith('['):
            key, _, value = line.partition(':')
        else:
            continue
        key, value = key.strip(), value.strip()
        if value.startswith('[') and value.endswith(']'):
            items = [v for v in re.split(r'[,;]', value[1:-1]) if v.strip()]
            if len(items) not in (2, 3):
                raise ValueError(
                    f"\n Invalid bracket format for {key}: {value}\n"
                    f"Expected:\n"
                    f"  - [min, max] for optimization\n"
                    f"  - [min, max, n_points] for scan\n")
            kind = 'range' if len(items) == 2 else 'grid'
        elif _UNCERTAIN_RE.match(value):
            kind = 'uncertain'
        else:
            kind = 'scalar'
        entries.append((key, kind, value))
    return entries


def _bracket_numbers(key, value):
    """Numbers of a bracketed deck value, with an explicit error message."""
    items = [v.strip() for v in re.split(r'[,;]', value[1:-1]) if v.strip()]
    form = "[min, max]" if len(items) == 2 else "[min, max, n_points]"
    try:
        return [float(v) for v in items]
    except ValueError as e:
        raise ValueError(f"\n Invalid values for parameter {key}: {value}\n"
                         f"Values must be numeric: {form}\n"
                         f"Error: {e}") from None


def detect_mode_from_lines(lines):
    """
    Execution mode of a deck, read from its content (see deck_entries).

    Returns
    -------
    (mode, details) with
      'uncertainty'  : number of uncertain entries
      'popcon'       : {'nbar_line': (min, max, n), 'Tbar': (min, max, n)}
      'scan'         : [(name, min, max, n_points), (name, min, max, n_points)]
      'optimization' : {name: (min, max)}
      'run'          : []

    Raises
    ------
    ValueError : forms mixed in one deck, a key bracketed twice, or a wrong
                 number of axes or ranges.
    """
    entries = deck_entries(lines)
    uncertain = [k for k, kind, _ in entries if kind == 'uncertain']
    grids, ranges = {}, {}
    for key, kind, value in entries:
        if kind not in ('grid', 'range'):
            continue
        if key in grids or key in ranges:
            raise ValueError(f"\n '{key}' is given twice as a bracketed range.\n")
        nums = _bracket_numbers(key, value)
        if kind == 'grid':
            grids[key] = (nums[0], nums[1], int(nums[2]))
        else:
            ranges[key] = (nums[0], nums[1])

    if uncertain:
        if grids or ranges:
            raise ValueError(
                f"\n Invalid input file: uncertain entries and bracketed ranges "
                f"in the same file.\n"
                f"  - uncertain : {', '.join(uncertain)}\n"
                f"  - bracketed : {', '.join(list(grids) + list(ranges))}\n"
                f"An UNCERTAINTY study varies its inputs with tri(), norm(), "
                f"unif() or envelope() only.\n")
        return 'uncertainty', len(uncertain)

    if grids and ranges:
        raise ValueError(
            f"\n Invalid input file: Mixed parameter formats detected.\n"
            f"  - {len(ranges)} optimization parameter(s) with [min, max]\n"
            f"  - {len(grids)} scan parameter(s) with [min, max, n_points]\n"
            f"\n"
            f"Please choose ONE mode:\n"
            f"  - OPTIMIZATION: Use [min, max] for all variable parameters\n"
            f"  - SCAN: Use [min, max, n_points] for exactly 2 parameters\n"
            f"  - RUN: Remove all brackets for fixed values\n")

    if grids:
        if set(grids) == set(POPCON_AXES):
            return 'popcon', {k: grids[k] for k in POPCON_AXES}
        if 'nbar_line' in grids:
            raise ValueError(
                f"\n nbar_line is not a design input. An axis "
                f"nbar_line = [min, max, n_points] makes a POPCON map,\n"
                f"together with Tbar = [min, max, n_points] and no other axis "
                f"(found: {', '.join(grids)}).\n")
        if len(grids) == 1:
            param_name = list(grids)[0]
            raise ValueError(
                f"\n Invalid scan: Found only 1 parameter ({param_name}).\n"
                f"\n"
                f"SCAN mode requires exactly 2 parameters with [min, max, n_points].\n"
                f"\n"
                f"Example:\n"
                f"  R0 = [3, 9, 25]\n"
                f"  a = [1, 3, 25]\n")
        if len(grids) > 2:
            raise ValueError(
                f"\n Invalid scan: Found {len(grids)} parameters: {', '.join(grids)}.\n"
                f"\n"
                f"SCAN mode requires exactly 2 parameters.\n")
        return 'scan', [(k, lo, hi, n) for k, (lo, hi, n) in grids.items()]

    if ranges:
        if len(ranges) < 2:
            param_name = list(ranges)[0]
            raise ValueError(
                f"\n Invalid optimization: Found only 1 parameter ({param_name}).\n"
                f"\n"
                f"OPTIMIZATION mode requires at least 2 parameters with [min, max].\n"
                f"\n"
                f"Example:\n"
                f"  R0 = [3, 9]\n"
                f"  a = [1, 3]\n"
                f"  Bmax_TF = [10, 16]\n")
        return 'optimization', ranges

    return 'run', []


#%% Core calculation
# ---------------------------------------------------------------------------

# Coronal-equilibrium mean ion charges <Z>.
# SUPERSEDED for the Z_eff / dilution chain by the temperature-dependent
# Mavrin (2018) fits get_Z_mean(species, Tbar) (D0FUS_physical_functions),
# which share the reference and species coverage of the get_Lz cooling-rate
# tables. Kept only as a documented reference of the former fixed values
# (representative of Tbar ~ 5-10 keV; e.g. W: 46 vs Mavrin 53 at 8.9 keV).
_Z_CORONAL_MEAN = {'Be': 4, 'C': 6, 'N': 7, 'Ne': 10, 'Ar': 18, 'Kr': 34, 'Xe': 44, 'W': 46}


def _parse_impurity_inventory(config):
    """
    Parse config.impurity_species / config.f_imp_core into matched lists.

    Returns
    -------
    (species, concentrations) : (list[str], list[float])
        Only entries with a strictly positive concentration n_imp/n_e are kept.
        Empty / 'None' inventory returns ([], []).

    Raises
    ------
    ValueError  When the two lists do not have the same length.
    """
    sp_raw = getattr(config, 'impurity_species', '')
    fc_raw = getattr(config, 'f_imp_core', '')
    if sp_raw is None or str(sp_raw).strip() in ('', 'None', 'none'):
        return [], []
    species = [s.strip() for s in str(sp_raw).split(',') if s.strip()]
    if isinstance(fc_raw, (int, float)):
        conc = [float(fc_raw)]
    else:
        conc = [float(c.strip()) for c in str(fc_raw).split(',') if c.strip()]
    if len(conc) != len(species):
        raise ValueError(
            f"impurity_species = {sp_raw!r} lists {len(species)} species but "
            f"f_imp_core = {fc_raw!r} gives {len(conc)} concentrations.")
    paired = [(s, c) for s, c in zip(species, conc) if c > 0]
    return [p[0] for p in paired], [p[1] for p in paired]


def _compute_Zeff_effective(config, f_alpha):
    """
    Effective plasma charge Z_eff = sum_k n_k Z_k^2 / n_e.

    If config.Zeff is an explicit float, that value is returned unchanged
    (manual override / legacy behaviour).  Otherwise Z_eff is computed
    self-consistently from the prescribed impurity inventory and the helium
    ash fraction f_alpha = n_He/n_e, using the same coronal mean charges
    <Z_j> as the fuel-dilution term:

        Z_eff = 1 + 2 f_He - sum_j <Z_j> c_j + sum_j <Z_j>^2 c_j

    where the leading 1 is the D-T fuel baseline (Z = 1), 2 f_He the helium
    ash (Z_He = 2), and c_j = n_j/n_e the impurity concentrations.  An empty
    inventory therefore gives Z_eff = 1 + 2 f_He (clean D-T+He plasma).

    Note: <Z_j> are fixed coronal values; for high-Z species (W) this is a
    mid-temperature approximation, and the rigorous moment is <Z^2> rather
    than <Z>^2.

    Parameters
    ----------
    config  : GlobalConfig  Configuration (provides Zeff override + inventory).
    f_alpha : float         Helium ash fraction n_He / n_e.
    """
    if getattr(config, 'Zeff', None) is not None:
        return float(config.Zeff)
    species, conc = _parse_impurity_inventory(config)
    # Temperature-dependent coronal mean charges <Z>(Tbar) — Mavrin (2018)
    # fits, consistent with the get_Lz radiation tables.
    Z1 = sum(get_Z_mean(s, config.Tbar) * c for s, c in zip(species, conc))
    Z2 = sum(get_Z_mean(s, config.Tbar) ** 2 * c for s, c in zip(species, conc))
    return 1.0 + 2.0 * float(f_alpha) - Z1 + Z2


def _ion_density_fraction(config, f_alpha):
    """
    Total ion density over electron density, n_i,tot / n_e, from
    quasi-neutrality with the He ash and the impurity inventory:

        n_i,tot / n_e = 1 - f_He - sum_j (<Z_j> - 1) c_j

    (fuel n_e (1 - 2 f_He - sum_j <Z_j> c_j) + He f_He + impurities sum_j c_j),
    with the same coronal <Z_j>(Tbar) as the fuel dilution and Z_eff.
    """
    species, conc = _parse_impurity_inventory(config)
    Z1 = sum(get_Z_mean(s, config.Tbar) * c for s, c in zip(species, conc))
    return 1.0 - float(f_alpha) - Z1 + float(sum(conc))


def _profile_params(config, Vprime_data=None):
    """
    Profile parameters (nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac) of the
    deck, from the Manual inputs or the _PROFILE_PRESETS table.

    The deck and the presets give rho_ped in rho_tor. It is returned as the
    volume label rho_V of the design geometry (Vprime_data, None in Academic
    geometry where rho_V = rho_tor), the coordinate of the profiles.
    """
    if config.Plasma_profiles == 'Manual':
        p = dict(nu_n=config.nu_n_manual, nu_T=config.nu_T_manual,
                 rho_ped=config.rho_ped, n_ped_frac=config.n_ped_frac,
                 T_ped_frac=config.T_ped_frac)
    else:
        p = _PROFILE_PRESETS[config.Plasma_profiles]
    return (p['nu_n'], p['nu_T'],
            float(rho_V_of_rho_tor(p['rho_ped'], Vprime_data)),
            p['n_ped_frac'], p['T_ped_frac'])


def new_output_dir(parent, prefix):
    """
    Create a new output folder parent/<prefix>_<YYYYmmdd_HHMMSS>.

    Two runs started within the same second get distinct folders, the
    second one with the suffix _2, the third with _3, and so on.

    Returns
    -------
    (path, timestamp) : (str, str)
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path, k = os.path.join(parent, f"{prefix}_{stamp}"), 1
    while True:
        try:
            os.makedirs(path)
            return path, stamp
        except FileExistsError:
            k += 1
            path = os.path.join(parent, f"{prefix}_{stamp}_{k}")


def default_output_root():
    """
    Default parent folder of the result folders of every mode.

    In a source checkout (pyproject.toml at the root of the code, editable
    installs included) it is D0FUS_OUTPUTS at that root, whatever the
    working directory. For an installed package it is D0FUS_OUTPUTS in the
    working directory, so that no result is written into site-packages.

    Returns
    -------
    str : absolute path of the folder (not created here).
    """
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(code_root, "pyproject.toml")):
        return os.path.join(code_root, "D0FUS_OUTPUTS")
    return os.path.join(os.getcwd(), "D0FUS_OUTPUTS")


def _cd_deposition(CD_source, I_LH, I_EC, I_NBI, rho_EC, rho_NBI):
    """
    Centre and width (rho_CD, delta_CD) of the Gaussian current-drive
    deposition of the converged q profile, shared by run() and the figures.

    ECCD (rho_EC, 0.05) and NBCD (rho_NBI, 0.15) use their input radius.
    Multi with a non-zero driven current takes the current-weighted average
    of the LH (0.5, 0.20), EC and NBI depositions. LHCD, Academic and Multi
    without driven current use a broad central deposition (0.5, 0.20).

    Parameters
    ----------
    CD_source           : str    Current-drive scheme.
    I_LH, I_EC, I_NBI   : float  Driven currents of each source [MA].
    rho_EC, rho_NBI     : float  Input deposition radii [-].

    Returns
    -------
    (rho_CD, delta_CD) : (float, float)
    """
    I_sources = np.abs([I_LH, I_EC, I_NBI])
    if CD_source == 'Multi' and np.sum(I_sources) > 0:
        w = I_sources / np.sum(I_sources)
        return (float(np.dot(w, [0.5, rho_EC, rho_NBI])),
                float(np.dot(w, [0.20, 0.05, 0.15])))
    if CD_source == 'ECCD':
        return rho_EC, 0.05
    if CD_source == 'NBCD':
        return rho_NBI, 0.15
    return 0.5, 0.20


def fuel_power_split(config, Vprime_data=None):
    """
    (f_charged, f_neutron) of config.Fuel at the deck Tbar and profiles.

    Post-processing helper (reports, cost blocks). The profiles are
    functions of the volume label rho_V, so the volume integrals do not
    depend on the geometry: the Academic weight gives the split of run().
    Vprime_data (the design geometry) only converts the pedestal label of
    the deck from rho_tor to rho_V.
    """
    nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _profile_params(
        config, Vprime_data)
    return f_fuel_power_split(config.Fuel, config.Tbar, nu_T, nu_n,
                              rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                              T_ped_frac=T_ped_frac, Vprime_data=None,
                              tau_i_e=config.tau_i_e)


def house_load_breakdown(config, R_TF_out, R_TF_in, c, H_TF, Z_TF_bore_half,
                         N_TF, E_TF_J, P_cryo_cool_kW):
    """
    House load of the plant, PROCESS model (f_house_load): facilities on the
    effective floor area of the plant buildings, tritium plant (D-T fuel
    only), vacuum pumps and TF power supplies.

    D0FUS has no PF coils: the cryostat radius is the TF outer radius plus
    dr_TF_cryostat. The shield sector spans the TF bore: width at the
    midplane, height of the inner contour.

    Returns
    -------
    dict  P_house, P_facilities, P_tritium, P_vacuum, P_TF_supplies [MWe],
          A_floor [m2], V_buildings [m3].
    """
    P_TF, V_TF = f_tf_power_supplies(config.I_cond, config.R0, N_TF,
                                     config.V_max, E_TF_J)
    bld = f_plant_buildings(
        R_cryostat=R_TF_out + config.dr_TF_cryostat, R_TF_out=R_TF_out,
        R_TF_in=R_TF_in, H_TF=H_TF, W_shield=R_TF_out - R_TF_in - 2.0 * c,
        H_shield=2.0 * Z_TF_bore_half, P_cryo_W=1.0e3 * P_cryo_cool_kW,
        V_TF_supplies=V_TF, pulsed=(config.Operation_mode == 'Pulsed'))
    P_trit = config.P_tritium_plant if config.Fuel == 'DT' else 0.0
    P_fac, P_house = f_house_load(bld['A_floor'], P_TF, config.P_plant_base,
                                  config.pflux_plant_floor, P_trit,
                                  config.P_vacuum_pumps)
    return dict(P_house=P_house, P_facilities=P_fac, P_tritium=P_trit,
                P_vacuum=config.P_vacuum_pumps, P_TF_supplies=P_TF,
                A_floor=bld['A_floor'], V_buildings=bld['V_total'])


def _design_Vprime(config, results):
    """
    Miller V'(rho) of a converged design on the run() grid (N_rho = 500,
    N_theta = 200), so that post-processing integrals match the physics
    chain. None in Academic geometry or when the shaping is not finite.
    """
    kappa_edge, kappa_95, delta_edge, delta_95 = (
        float(np.real(x)) for x in results[62:66])
    if not (config.Plasma_geometry == 'refined'
            and np.isfinite(kappa_edge) and np.isfinite(delta_edge)):
        return None
    return precompute_Vprime(config.R0, config.a, kappa_edge, delta_edge,
                             geometry_model='refined', kappa_95=kappa_95,
                             delta_95=delta_95, N_rho=500, N_theta=200)


def runaway_indicators(config, results, Vprime_data=None):
    """
    Runaway-electron indicators of a converged design.

    compute_RE_indicators is called with the profiles, shaping, l_i and
    Z_eff of the design and with the deck disruption settings (tau_TQ,
    Te_final_eV, pellet_dilution, pellet_dilution_cools). The RUN report
    and the SCAN maps both use this function, so that one design gives the
    same indicators in both modes.

    Parameters
    ----------
    config      : GlobalConfig   Resolved configuration of the design.
    results     : tuple          Return value of run(config).
    Vprime_data : tuple or None  Miller V'(rho) of the design. Built with
                                 _design_Vprime when None.

    Returns
    -------
    dict  Output of compute_RE_indicators.
    """
    if Vprime_data is None:
        Vprime_data = _design_Vprime(config, results)
    nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _profile_params(
        config, Vprime_data)
    g = {k: float(np.real(results[i])) for k, i in
         dict(Volume=6, Ip=8, nbar=12, f_He=40, kappa=62, li=72).items()}
    return compute_RE_indicators(
        Ip=g['Ip'], nbar=g['nbar'], Tbar=config.Tbar,
        a=config.a, R0=config.R0, κ=g['kappa'],
        Z_eff=_compute_Zeff_effective(config, g['f_He']), li=g['li'],
        nu_n=nu_n, nu_T=nu_T, rho_ped=rho_ped, n_ped_frac=n_ped_frac,
        T_ped_frac=T_ped_frac, Te_final_eV=config.Te_final_eV,
        tau_TQ=config.tau_TQ, Vprime_data=Vprime_data, V=g['Volume'],
        pellet_dilution=config.pellet_dilution,
        pellet_dilution_cools=config.pellet_dilution_cools)


def core_plasma_values(config, results):
    """
    On-axis plasma values (rho = 0) of a converged design.

    T_e0 and n_e0 follow f_Tprof / f_nprof with the run() normalisation
    (Miller V' in refined geometry, cylindrical otherwise). T_i0 = tau_i_e
    T_e0. n_i0 is the fuel-ion density from quasi-neutrality with the He
    ash and the impurity inventory: n_i0 = n_e0 (1 - 2 f_He - sum_j <Z_j> c_j).

    Parameters
    ----------
    config  : GlobalConfig  Resolved configuration (Tbar, P_fus prescribed).
    results : tuple         Return value of run(config).

    Returns
    -------
    dict  Te0, Ti0 [keV], ne0, ni0 [1e20 m^-3]. NaN on failed designs.
    """
    nbar, f_He = float(results[12]), float(results[40])
    if not (np.isfinite(nbar) and np.isfinite(f_He)):
        return dict(Te0=np.nan, Ti0=np.nan, ne0=np.nan, ni0=np.nan)
    Vprime_data = _design_Vprime(config, results)
    nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _profile_params(
        config, Vprime_data)
    Te0 = float(f_Tprof(config.Tbar, nu_T, 0.0, rho_ped=rho_ped,
                        T_ped_frac=T_ped_frac, Vprime_data=Vprime_data))
    ne0 = float(f_nprof(nbar, nu_n, 0.0, rho_ped=rho_ped,
                        n_ped_frac=n_ped_frac, Vprime_data=Vprime_data))
    species, conc = _parse_impurity_inventory(config)
    Z1 = sum(get_Z_mean(sp, config.Tbar) * c for sp, c in zip(species, conc))
    return dict(Te0=Te0, Ti0=config.tau_i_e * Te0,
                ne0=ne0, ni0=ne0 * (1.0 - 2.0 * f_He - Z1))


#%% Shared post-convergence evaluation (single source for every execution mode)
# ---------------------------------------------------------------------------
# RUN, SCAN, GENETIC and UNCERTAINTY all derive the Sheffield cost and the
# feasibility verdict from these two functions, so that one design point gives
# the same C_invest, COE and feasibility whatever the mode that evaluated it.

# Absolute indices in the run() results tuple used below.
_RI = dict(Surface=7, Ip=8, nbar_line=13, nG=14, betaT=17, qstar=19, q95=20,
           P_CD=21, P_elec=25, cost=27, Gamma_n=39, c=44, d=51, r_d=61,
           kappa=62, betaN_total=96, V_TF_one=99, V_CS_geom=100,
           t_life_bl_yr=130, t_life_div_yr=131,
           T_op_limit=132, CF=135, V_rb_BB=138, V_rb_shield=139, V_rb_VV=140,
           V_rb_gap_TF=141, P_th=150)


def evaluate_costs(config, results):
    """
    Sheffield (2016) cost breakdown of a converged design, on the RUN geometry.

    Volumes are the exact ones computed by run(): TF on the conductor
    centreline (V_TF_one x N_TF), CS with the resolved height (V_CS_geom),
    breeding blanket (V_bl = V_rb_BB) and shield and gaps (V_sg = shield +
    vacuum vessel + gap layers) from the radial build layers, and the
    fusion-island cylinder of height H_TF from the Princeton-D solve. The
    divertor target area is f_div_area_fraction x plasma surface.
    P_th is read from the results tuple, so it carries the P_fus resolved by
    run() (P_fus_mode = 'greenwald') rather than the deck placeholder.

    A plant without net electric output (P_elec <= 0) has no COE: it is NaN
    and 'net_electric' is False. Its capital cost is still given, with the
    balance of plant taken at P_e = 0.

    Returns
    -------
    dict  All f_costs_Sheffield outputs (C_invest [M EUR], COE [EUR/MWh],
          component costs), the inputs and 'net_electric'. Empty dict when
          cost_model = 'None' or when an input is non-finite.
    """
    if getattr(config, 'cost_model', 'None') == 'None':
        return {}
    g = {k: float(np.real(results[i])) for k, i in _RI.items()}
    need = ('Surface', 'P_CD', 'P_elec', 'Gamma_n', 'c', 'd', 'kappa',
            'V_TF_one', 'V_CS_geom', 't_life_bl_yr', 't_life_div_yr',
            'T_op_limit', 'CF', 'V_rb_BB', 'V_rb_shield', 'V_rb_VV',
            'V_rb_gap_TF', 'P_th')
    if not all(np.isfinite(g[k]) for k in need):
        return {}
    try:
        N_TF, _, Delta_TF = Number_TF_coils(config.R0, config.a, config.b,
                                            config.ripple_adm, config.L_min)
        N_TF, Delta_TF = int(N_TF), float(Delta_TF)
        H_TF = float(f_TF_cross_section(config.a, config.b, config.R0,
                                        g['c'], Delta_TF)[2])
        V_FI = float(f_volume(config.a, config.b, g['c'], g['d'], config.R0,
                              g['kappa'], Delta_TF, H_TF)[3])
    except Exception:
        return {}
    V_sg = g['V_rb_shield'] + g['V_rb_VV'] + g['V_rb_gap_TF']
    inputs = dict(P_t=g['P_th'], P_e=max(g['P_elec'], 0.0), P_aux=g['P_CD'],
                  Gamma_n=g['Gamma_n'], T_op_limit=g['T_op_limit'], CF=g['CF'],
                  t_life_bl_yr=g['t_life_bl_yr'], t_life_div_yr=g['t_life_div_yr'],
                  V_FI=V_FI, V_pc=g['V_TF_one'] * N_TF + g['V_CS_geom'],
                  V_sg=V_sg, V_bl=g['V_rb_BB'],
                  S_tt=config.f_div_area_fraction * g['Surface'])
    out = f_costs_Sheffield(
        discount_rate=config.discount_rate, contingency=config.contingency,
        T_life=config.T_life, T_build=config.T_build,
        Supra_cost_factor=config.Supra_cost_factor, **inputs)
    keys = ('T_op_limit', 'CF', 'C_invest', 'COE', 'C_ind', 'C_Op_waste',
            'C_Op_OM', 'C_Op_F', 'C_syst_other', 'C_syst_BOP', 'C_syst_heat',
            'C_syst_aux', 'C_reac_tt', 'C_reac_bl', 'C_reac_sg', 'C_reac_pc')
    res = {k: float(v) for k, v in zip(keys, out)}
    res.update(inputs, N_TF=N_TF, Delta_TF=Delta_TF, H_TF=H_TF,
               net_electric=bool(g['P_elec'] > 0.0))
    # C_Op_waste is a cost per unit of energy [EUR/MWh], added to the COE.
    # Its annual value [M EUR/yr] is shown with the other operating costs.
    res['C_Op_waste_annual'] = (res['C_Op_waste'] * 8760.0 * res['CF']
                                * res['P_e'] * 1e-6)
    return res


def radial_build_check(cost, r_d, c_TF, d_CS, q_kink, betaT, nbar_line):
    """
    Geometric closure of a design: every key scalar finite and non-negative,
    TF and CS thicker than 1 mm. Returns (ok, reason); reason is None if ok.
    """
    for name, val in (('cost', cost), ('r_d', r_d), ('c_TF', c_TF),
                      ('d_CS', d_CS), ('q_kink', q_kink), ('betaT', betaT),
                      ('nbar_line', nbar_line)):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return False, f"{name} is not a real number"
        if not np.isfinite(v):
            return False, f"{name} is NaN/Inf"
        if v < 0:
            return False, f"{name} = {v:.4g} < 0"
    if c_TF < 1e-3:
        return False, f"c_TF = {c_TF:.4g} m (too thin)"
    if d_CS < 1e-3:
        return False, f"d_CS = {d_CS:.4g} m (too thin / no CS)"
    return True, None


def evaluate_feasibility(config, results):
    """
    Feasibility verdict of a converged design (single definition for all modes).

    Constraints, each expressed as a ratio that must stay <= 1:
      density : nbar_line / nG      (nG already carries Greenwald_limit and the
                                     selected density_limit_model)
      beta    : betaN_total / betaN_limit   (Troyon, fast alphas included)
      q       : q_limit / q_kink    (q_kink = q95 or q* per kink_parameter)
      Ip      : Ip / Ip_limit       (only when Ip_limit is set)
    Radial build: every key scalar finite and non-negative, c_TF and d_CS
    thicker than 1 mm (the GENETIC check_radial_build rule).

    Returns
    -------
    dict  ratios, 'max_ratio', 'build_ok', 'feasible' and 'binding' (name of
          the most violated constraint, None when feasible).
    """
    g = {k: float(np.real(results[i])) for k, i in _RI.items()}
    q_kink = g['q95'] if config.kink_parameter == 'q95' else g['qstar']
    ratios = {
        'density': g['nbar_line'] / g['nG'] if g['nG'] > 0 else np.nan,
        'beta': g['betaN_total'] / config.betaN_limit,
        'q': config.q_limit / q_kink if q_kink > 0 else np.nan,
    }
    Ip_lim = config.Ip_limit
    if Ip_lim is not None and np.isfinite(Ip_lim) and Ip_lim > 0:
        ratios['Ip'] = g['Ip'] / Ip_lim
    build_ok, build_reason = radial_build_check(
        g['cost'], g['r_d'], g['c'], g['d'], q_kink, g['betaT'], g['nbar_line'])
    finite = all(np.isfinite(v) for v in ratios.values())
    max_ratio = max(ratios.values()) if finite else np.nan
    feasible = bool(build_ok and finite and max_ratio <= 1.0)
    if feasible:
        binding = None
    elif not build_ok:
        binding = 'radial_build'
    elif not finite:
        binding = 'non_finite'
    else:
        binding = max(ratios, key=ratios.get)
    return dict(ratios=ratios, max_ratio=max_ratio, build_ok=build_ok,
                build_reason=build_reason, feasible=feasible, binding=binding,
                q_kink=q_kink)


def resolve_Tbar(config: GlobalConfig, verbose: int = 0) -> GlobalConfig:
    """
    Resolve the volume-averaged temperature according to config.Tbar_mode.

    'manual'    : returns the configuration unchanged.
    'greenwald' : solves Tbar with brentq so that the converged design sits
        at the requested Greenwald fraction
            g(Tbar) = nbar_line(Tbar) / (Ip(Tbar) / pi a^2) - f_GW_target = 0.
        The deck Tbar is the initial guess: the bracket is searched around
        it with steps of 0.5 keV, doubled, inside [Tbar_min, Tbar_max].  If
        no sign change is found there, the whole interval and then a coarse
        pre-scan are tried, and a clear error is raised otherwise.  Each
        evaluation is one full run() at Tbar_mode='manual' (5 for the ITER
        deck).

    Returns the configuration with the solved Tbar and Tbar_mode='manual',
    so that downstream consumers (run, printing, figures, the other
    execution modes) see a fully resolved design.
    """
    mode = str(getattr(config, 'Tbar_mode', 'manual')).lower()
    if mode != 'greenwald':
        return config
    target = float(config.f_GW_target)
    T_lo, T_hi = float(config.Tbar_min), float(config.Tbar_max)

    def g(T):
        # Probe evaluations may legitimately fail at extreme temperatures
        # (e.g. radiation exceeding heating near Tbar_max); these failures
        # are handled through the NaN return below, so the corresponding
        # physics RuntimeWarnings are suppressed for the probes only.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                res = run(dc_replace(config, Tbar=float(T), Tbar_mode='manual'),
                          verbose=0)
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.nan
        nbl, Ip = float(res[13]), float(res[8])
        if not (np.isfinite(nbl) and np.isfinite(Ip) and Ip > 0):
            return np.nan
        return nbl / f_nG(Ip, config.a) - target

    # Memoised evaluations: brentq re-evaluates the bracket ends.
    _g = {}
    def g_cached(T):
        if T not in _g:
            _g[T] = g(T)
        return _g[T]

    # Bracket around the deck Tbar: steps of 0.5 keV, doubled, first on the
    # side of the root given by the sign of g (f_GW falls as Tbar rises at
    # fixed P_fus), then on the other side.
    bracket = None
    T0 = min(max(float(config.Tbar), T_lo), T_hi)
    g0 = g_cached(T0)
    if np.isfinite(g0):
        for side in ((1.0, -1.0) if g0 > 0 else (-1.0, 1.0)):
            T_a, g_a, step = T0, g0, 0.5
            while bracket is None:
                T_b = min(max(T_a + side * step, T_lo), T_hi)
                g_b = g_cached(T_b) if T_b != T_a else np.nan
                if not np.isfinite(g_b):
                    break
                if g_a * g_b <= 0:
                    bracket = (min(T_a, T_b), max(T_a, T_b))
                else:
                    T_a, g_a, step = T_b, g_b, 2.0 * step
            if bracket is not None:
                break

    if bracket is None:
        g_lo, g_hi = g_cached(T_lo), g_cached(T_hi)
        if np.isfinite(g_lo) and np.isfinite(g_hi) and g_lo * g_hi < 0:
            bracket = (T_lo, T_hi)
        else:
            # Coarse pre-scan for an enclosing sub-bracket (handles solver
            # failures at the bracket edges and non-monotonic corners).
            Ts = np.linspace(T_lo, T_hi, 13)
            gs = np.array([g_cached(T) for T in Ts])
            ok = np.isfinite(gs)
            for k in range(len(Ts) - 1):
                if ok[k] and ok[k + 1] and gs[k] * gs[k + 1] < 0:
                    bracket = (Ts[k], Ts[k + 1])
                    break
            if bracket is None:
                raise ValueError(
                    f"resolve_Tbar: no Tbar in [{T_lo}, {T_hi}] keV reaches "
                    f"f_GW = {target} (g spans "
                    f"[{np.nanmin(gs):+.3f}, {np.nanmax(gs):+.3f}]). "
                    f"Adjust f_GW_target or the [Tbar_min, Tbar_max] bracket.")
    T_star = brentq(g_cached, *bracket, xtol=1e-3, rtol=1e-5)
    if verbose:
        print(f"resolve_Tbar: f_GW = {target} reached at "
              f"Tbar = {T_star:.3f} keV")
    return dc_replace(config, Tbar=float(T_star), Tbar_mode='manual')


def resolve_P_fus(config: GlobalConfig, verbose: int = 0) -> GlobalConfig:
    """
    Resolve the fusion power according to config.P_fus_mode.

    'manual'    : returns the configuration unchanged.
    'greenwald' : solves P_fus at the deck Tbar so that the converged design
        sits at the requested Greenwald fraction,
            g(P) = nbar_line(P) / (Ip(P) / pi a^2) - f_GW_target = 0.
        The unknown is x = log10(P_fus), since D-D devices sit at kW level
        and D-T devices at 10-1000 MW. The deck P_fus is the initial
        guess. The bracket is widened by one decade per step, up to six
        decades on each side. Only rising crossings (f_GW increasing with
        P_fus) are accepted: at very low P_fus the ohmic branch gives a
        spurious falling crossing. Each iteration is one full run() at
        P_fus_mode='manual'.

    Returns the configuration with the solved P_fus and P_fus_mode='manual'.
    """
    mode = str(getattr(config, 'P_fus_mode', 'manual')).lower()
    if mode == 'manual':
        return config
    if mode != 'greenwald':
        raise ValueError(f"Unknown P_fus_mode: '{config.P_fus_mode}'. "
                         "Valid options: 'manual', 'greenwald'.")
    if str(getattr(config, 'Tbar_mode', 'manual')).lower() == 'greenwald':
        raise ValueError("P_fus_mode and Tbar_mode cannot both be "
                         "'greenwald': f_GW_target closes only one unknown.")
    if not config.P_fus > 0:
        raise ValueError("resolve_P_fus: the deck P_fus is the initial guess "
                         f"and must be > 0 (got {config.P_fus}).")
    target = float(config.f_GW_target)

    def g(x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                res = run(dc_replace(config, P_fus=float(10.0 ** x),
                                     P_fus_mode='manual'), verbose=0)
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.nan
        nbl, Ip = float(res[13]), float(res[8])
        if not (np.isfinite(nbl) and np.isfinite(Ip) and Ip > 0):
            return np.nan
        return nbl / f_nG(Ip, config.a) - target

    # Bracket search, one decade per step on each side of the guess.
    # f_GW grows with P_fus on the physical branch, so the side of the
    # root is searched first and only rising crossings are accepted.
    _g = {}
    def g_cached(x):
        if x not in _g:
            _g[x] = g(x)
        return _g[x]
    x0 = float(np.log10(config.P_fus))
    up_first = not (g_cached(x0) > 0)
    bracket = None
    for k in range(1, 7):
        pairs = ((x0 + k - 1, x0 + k), (x0 - k, x0 - k + 1))
        for x1, x2 in (pairs if up_first else pairs[::-1]):
            g1, g2 = g_cached(x1), g_cached(x2)
            if np.isfinite(g1) and np.isfinite(g2) and g1 <= 0 <= g2:
                bracket = (x1, x2)
                break
        if bracket is not None:
            break
    if bracket is None:
        raise ValueError(
            f"resolve_P_fus: no P_fus in [{10**(x0-6):.3g}, {10**(x0+6):.3g}]"
            f" MW reaches f_GW = {target} at Tbar = {config.Tbar} keV. "
            "Adjust f_GW_target, Tbar or H.")
    x_star = brentq(g_cached, *bracket, xtol=1e-5, rtol=1e-8)
    P_star = float(10.0 ** x_star)
    if verbose:
        print(f"resolve_P_fus: f_GW = {target} reached at "
              f"P_fus = {P_star:.4g} MW (Fuel = {config.Fuel})")
    return dc_replace(config, P_fus=P_star, P_fus_mode='manual')


def apply_Q_target(config: GlobalConfig) -> GlobalConfig:
    """
    Pulsed mode with config.Q_target set: P_aux = P_fus / Q_target.

    The auxiliary power then follows P_fus (GA, scan, Greenwald closure on
    P_fus). With CD_source = 'Multi' the deck source powers are rescaled in
    proportion, so they set the mix only. The reported Q = P_fus /
    (P_aux + P_Ohm) is slightly below Q_target. Idempotent.
    """
    Q_t = getattr(config, 'Q_target', None)
    if Q_t is None:
        return config
    if config.Operation_mode != 'Pulsed':
        raise ValueError("Q_target applies to Operation_mode = 'Pulsed' "
                         "(Steady-State solves Q from the current drive).")
    if not Q_t > 0:
        raise ValueError(f"Q_target must be > 0 (got {Q_t}).")
    P_aux = config.P_fus / Q_t
    if config.CD_source != 'Multi':
        return dc_replace(config, P_aux_input=P_aux)
    P_src = config.P_LH + config.P_ECRH + config.P_NBI + config.P_ICRH
    if not P_src > 0:
        raise ValueError("Q_target with CD_source = 'Multi' needs non-zero "
                         "deck source powers to define the mix.")
    s = P_aux / P_src
    return dc_replace(config, P_LH=config.P_LH * s, P_ECRH=config.P_ECRH * s,
                      P_NBI=config.P_NBI * s, P_ICRH=config.P_ICRH * s)


def resolve_operating_point(config: GlobalConfig,
                            verbose: int = 0) -> GlobalConfig:
    """
    Apply the Greenwald closure selected by the deck: Tbar_mode='greenwald'
    solves Tbar, P_fus_mode='greenwald' solves P_fus. Returns a fully
    prescribed configuration (both modes 'manual'), with P_aux set from
    Q_target on the final P_fus when Q_target is given.
    """
    config = resolve_P_fus(config, verbose=verbose)
    config = resolve_Tbar(config, verbose=verbose)
    return apply_Q_target(config)


def run(config: GlobalConfig = None, verbose: int = 0) -> tuple:
    """
    Compute a single D0FUS design point.

    Parameters
    ----------
    config : GlobalConfig, optional
        Complete design configuration.  If None, DEFAULT_CONFIG is used.
        To override selected parameters only, use dataclasses.replace():
            cfg = dc_replace(DEFAULT_CONFIG, R0=8.0, Bmax_TF=13.0)
            results = run(cfg)
    verbose : int, optional
        Verbosity level: 0 = silent, 1 = convergence summary, 2 = full debug.
        Default: 0.

    Returns
    -------
    tuple of 165 entries, scalars except one array and one dict
        Ordered outputs.  All entries are np.nan if the self-consistent solver
        fails to converge.

        Unit conventions (all other quantities follow standard D0FUS units):
          - W_th  (index 4) : Joules [J]  — divide by 1e6 for MJ.
            All other power-like quantities are in MW.
          - J_TF, J_CS (indices 42–43) : A/m²  — divide by 1e6 for A/mm².
          - betaT (index 17) : dimensionless fraction  — multiply by 100 for %.
          - lambda_q (index 35) : metres [m]  — multiply by 1000 for mm.

        Full ordering: B0, B_CS, B_pol, tauE, W_th[J], Q, Volume, Surface,
        Ip, Ib, I_CD, I_Ohm, nbar, nbar_line, nG, pbar, betaN, betaT, betaP,
        qstar, q95, P_CD, P_sep, P_Thresh, eta_CD, P_elec, P_wallplug, cost,
        P_Brem, P_syn, P_line, P_line_core, heat, heat_par, heat_pol, lambda_q,
        q_target, P_wall_rad, P_wall_div, Gamma_n, f_alpha, tau_alpha, J_TF, J_CS,
        c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_frac_TF,
        d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_frac_CS, B_CS(dup), J_CS(dup),
        r_minor, r_sep, r_c, r_d, κ, κ_95, δ, δ_95,
        ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS, Vloop, li,
        eta_LH, eta_EC, eta_NBI, P_LH, P_EC, P_NBI, P_ICR, I_LH, I_EC, I_NBI,
        f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF,
        f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS.
    """

    # ── Resolve configuration ─────────────────────────────────────────────────
    if config is None:
        config = DEFAULT_CONFIG
    # Resolve Tbar or P_fus when the deck prescribes a Greenwald fraction
    # (Tbar_mode or P_fus_mode = 'greenwald'). The resolved config carries
    # both modes at 'manual', so the recursive run() calls inside the solve
    # do not re-enter this branch.
    if ('greenwald' in (str(getattr(config, 'Tbar_mode', 'manual')).lower(),
                        str(getattr(config, 'P_fus_mode', 'manual')).lower())):
        config = resolve_operating_point(config, verbose=verbose)
    # P_aux = P_fus / Q_target when Q_target is set (pulsed mode)
    config = apply_Q_target(config)

    # Unpack every field into local names so the downstream physics code
    # is unchanged (no 'config.X' references scattered throughout).
    R0                        = config.R0
    a                         = config.a
    b                         = config.b
    Bmax_TF                   = config.Bmax_TF
    Bmax_CS_adm                   = config.Bmax_CS_adm
    P_fus                     = config.P_fus
    Tbar                      = config.Tbar
    tau_i_e                   = config.tau_i_e   # Ion/electron temperature ratio T_i/T_e
    H                         = config.H
    Operation_mode            = config.Operation_mode
    Temps_Plateau_input       = config.Temps_Plateau_input
    P_aux_input               = config.P_aux_input
    Plasma_profiles           = config.Plasma_profiles
    nu_n_manual               = config.nu_n_manual
    nu_T_manual               = config.nu_T_manual
    rho_ped                   = config.rho_ped
    n_ped_frac                = config.n_ped_frac
    T_ped_frac                = config.T_ped_frac
    Scaling_Law               = config.Scaling_Law
    L_H_Scaling_choice        = config.L_H_Scaling_choice
    Bootstrap_choice          = config.Bootstrap_choice
    trapped_fraction_model    = config.trapped_fraction_model
    Option_q95                = config.Option_q95
    q_profile_mode            = config.q_profile_mode
    alpha_J                   = config.alpha_J
    Option_Kappa              = config.Option_Kappa
    κ_manual                  = config.κ_manual
    betaN_limit               = config.betaN_limit   # Feasibility: evaluate_feasibility()
    q_limit                   = config.q_limit        # Feasibility: evaluate_feasibility()
    Greenwald_limit           = config.Greenwald_limit
    density_limit_model       = config.density_limit_model
    ms                        = config.ms
    Fuel                      = config.Fuel
    Atomic_mass               = config.Atomic_mass
    Zeff_override             = config.Zeff   # None -> compute Z_eff from inventory + ash;
                                              # float -> impose (legacy). See
                                              # _compute_Zeff_effective; the local Z_eff used
                                              # below is built from this per f_alpha.
    r_synch                   = config.r_synch
    C_Alpha                   = config.C_Alpha
    Ce                        = config.Ce
    eta_model                 = config.eta_model
    Chosen_Steel_TF           = config.Chosen_Steel_TF
    Chosen_Steel_CS           = config.Chosen_Steel_CS
    # NOTE: nu_Steel, Young_modul_Steel, Young_modul_GF are accessed directly
    # from the config object inside f_CS_refined / f_CS_CIRCE — no local alias needed.
    fatigue_CS                = config.fatigue_CS
    Radial_build_model        = config.Radial_build_model
    Choice_Buck_Wedg          = config.Choice_Buck_Wedg
    Supra_choice              = config.Supra_choice
    J_wost_Manual             = config.J_wost_Manual
    # CS conductor: its own superconductor and manual current density when
    # given, else those of the TF ('same' / None).
    Supra_choice_CS           = (Supra_choice if config.Supra_choice_CS == 'same'
                                 else config.Supra_choice_CS)
    J_wost_Manual_CS          = (J_wost_Manual if config.J_wost_Manual_CS is None
                                 else config.J_wost_Manual_CS)
    coef_inboard_tension      = config.coef_inboard_tension
    F_CClamp                  = config.F_CClamp
    n_shape_TF                = config.n_shape_TF
    c_BP                      = config.c_BP
    TF_grading                = config.TF_grading
    f_TF_steel_mass           = config.f_TF_steel_mass
    Gap                       = config.Gap
    n_shape_CS                = config.n_shape_CS
    N_sub_CS                  = config.N_sub_CS
    T_helium                  = config.T_helium
    Marge_T_He                = config.Marge_T_He
    f_He_pipe                 = config.f_He_pipe
    f_void                    = config.f_void
    f_In_cable                      = config.f_In_cable
    f_In_WP                     = getattr(config, 'f_In_WP', 0.0)
    Marge_T_Nb3Sn             = config.Marge_T_Nb3Sn
    Marge_T_NbTi              = config.Marge_T_NbTi
    Marge_T_REBCO             = config.Marge_T_REBCO
    Eps                       = config.Eps
    Tet                       = config.Tet
    I_cond                    = config.I_cond
    # CS conductor current, the TF one when the deck does not give it
    I_cond_CS = config.I_cond_CS if config.I_cond_CS is not None else I_cond
    V_max                     = config.V_max
    tau_h_LTS                 = config.tau_h_LTS
    tau_h_HTS                 = config.tau_h_HTS
    T_hotspot                 = config.T_hotspot
    RRR                       = config.RRR
    Dump_resistor_subdivision = config.Dump_resistor_subdivision
    # Blanket multiplication, gross conversion efficiency and coolant
    # pumping fraction: deck values, or the blanket concept values
    M_blanket, eta_T, f_pump  = plant_conversion(config)
    eta_WP                    = config.eta_WP_acad if config.CD_source == 'Academic' else config.eta_RF
    theta_deg                 = config.theta_deg
    f_n_sep                   = config.f_n_sep
    f_cooling_div             = config.f_cooling_div
    f_mom_div                 = config.f_mom_div
    q_dep_limit               = config.q_dep_limit
    flux_expansion            = config.flux_expansion
    ripple_adm                = config.ripple_adm
    L_min                     = config.L_min
    # ── Multi-source current drive / heating ──────────────────────────────────
    CD_source                 = config.CD_source
    P_LH                      = config.P_LH
    P_ECRH                    = config.P_ECRH
    P_NBI                     = config.P_NBI
    P_ICRH                    = config.P_ICRH
    f_heat_LH                 = config.f_heat_LH
    f_heat_EC                 = config.f_heat_EC
    f_heat_NBI                = config.f_heat_NBI
    f_heat_ICR                = config.f_heat_ICR
    rho_EC                    = config.rho_EC
    rho_NBI                   = config.rho_NBI
    A_beam                    = config.A_beam
    E_beam_keV                = config.E_beam_keV
    Plasma_geometry           = config.Plasma_geometry

    # Core/edge radiation boundary for P_loss convention.
    # Radiation emitted inside ρ < rho_rad_core is subtracted from P_heat
    # in τ_E and the scaling law inversion (it never crosses the separatrix).
    # Radiation emitted at ρ > rho_rad_core is edge radiation that reduces
    # P_sep but does NOT affect confinement.
    # Value set in GlobalConfig (default 0.6; canonical presets 0.6 and 1.0).
    rho_rad_core = config.rho_rad_core
    coreradiationfraction = config.coreradiationfraction

    # ── Cross-mode consistency checks ─────────────────────────────────────────
    # Raise warnings (verbose >= 1) when the user combines selectors that are
    # mechanically valid but lose physical self-consistency.
    #
    # 1) Bootstrap_choice='Segal' + q_profile_mode='refined'
    #    Segal-Cerfon-Freidberg is a scalar formula (no j_bs(rho)).  In refined
    #    mode the q-profile solver internally rebuilds j_bs(rho) via
    #    Sauter-Redl regardless of Bootstrap_choice; the integral of that
    #    internal j_bs(rho) generally differs from the Segal scalar by 5-15%,
    #    so the global Ip = I_Ohm + I_CD + I_bs balance becomes inconsistent.
    #
    # 2) CD_source='Academic' + q_profile_mode='refined'
    #    The academic CD model returns a scalar I_CD with no deposition
    #    information.  In refined mode this current is deposited via a
    #    fallback Gaussian centred at rho_CD = 0.5, delta_CD = 0.20 (broad,
    #    safe default, _cd_deposition).  Mechanically fine but the location
    #    is arbitrary; the user is informed so they can pick a physical CD
    #    source if desired.
    if (config.Bootstrap_choice == 'Segal'
            and config.q_profile_mode == 'refined' and verbose >= 1):
        print("[warn] Bootstrap_choice='Segal' with q_profile_mode='refined': "
              "the refined solver uses Sauter-Redl internally for j_bs(rho); "
              "the global I_bs (Segal scalar) and the integral of the internal "
              "j_bs(rho) will differ. Use Bootstrap_choice='Sauter-Redl' for "
              "full self-consistency.")
    if (config.CD_source == 'Academic'
            and config.q_profile_mode == 'refined' and verbose >= 1):
        print("[warn] CD_source='Academic' with q_profile_mode='refined': "
              "the academic gamma_CD model gives a scalar I_CD only; the "
              "refined solver deposits it on a fallback Gaussian at rho=0.5, "
              "delta=0.20.  Set CD_source to LHCD/ECCD/NBCD/Multi for a "
              "physically motivated deposition profile.")

    # ── Verbose summary of the configuration choices ──────────────────────────
    # Emitted once at run start so the user can confirm which physics
    # philosophy the calculation is running under.  Silent at verbose=0.
    if verbose >= 1:
        _alpha_J_str = (f", alpha_J={config.alpha_J}"
                        if config.q_profile_mode == 'academic' else "")
        print(
            f"[run] q_profile_mode={config.q_profile_mode!r}{_alpha_J_str}, "
            f"Plasma_geometry={config.Plasma_geometry!r}, "
            f"Bootstrap_choice={config.Bootstrap_choice!r}, "
            f"CD_source={config.CD_source!r}, "
            f"eta_model={config.eta_model!r}")

    # ── Multi-impurity parsing ────────────────────────────────────────────────
    imp_species_list, imp_conc_list = _parse_impurity_inventory(config)

    # Temperature-dependent coronal mean charges <Z>(Tbar) — Mavrin (2018)
    # fits (get_Z_mean), consistent with the get_Lz radiation tables and
    # with _compute_Zeff_effective. Unknown species now raise a clear error
    # instead of silently defaulting to Z = 10.
    f_imp_dilution = sum(get_Z_mean(s, Tbar) * c
                         for s, c in zip(imp_species_list, imp_conc_list))
    # Impurity Z^2 moment sum_j <Z_j>^2 c_j, reused below to build the effective
    # Z_eff = Z_eff_fuel + Zeff_imp_Z2 when config.Zeff is None (computed mode).
    Zeff_imp_Z2 = sum(get_Z_mean(s, Tbar) ** 2 * c
                      for s, c in zip(imp_species_list, imp_conc_list))

    # ── Profile peaking factors ───────────────────────────────────────────────
    # Values are read from the module-level _PROFILE_PRESETS table (single
    # source of truth shared with _build_run_dict and save_run_output).
    if Plasma_profiles == 'Manual':
        nu_n       = nu_n_manual
        nu_T       = nu_T_manual
        # rho_ped, n_ped_frac, T_ped_frac already read from config above
    elif Plasma_profiles in _PROFILE_PRESETS:
        _p         = _PROFILE_PRESETS[Plasma_profiles]
        nu_n       = _p['nu_n']
        nu_T       = _p['nu_T']
        rho_ped    = _p['rho_ped']
        n_ped_frac = _p['n_ped_frac']
        T_ped_frac = _p['T_ped_frac']
    else:
        raise ValueError(
            f"Unknown Plasma_profiles: '{Plasma_profiles}'. "
            "Valid options: 'L', 'H', 'Advanced', 'EU-DEMO', 'NT', 'Manual'."
        )

    # Validate the fuel early. Atomic_mass stays an input (scaling laws,
    # L-H threshold, SOL sound speed). Flag a value inconsistent with Fuel.
    if Fuel not in FUELS:
        raise ValueError(f"Unknown Fuel: '{Fuel}'. "
                         f"Valid options: {sorted(FUELS)}.")
    if Fuel == 'DD' and Operation_mode == 'Steady-State':
        # The SS solver parameterises P_aux = P_fus / Q with Q = O(10-100),
        # which is ill-posed at D-D fusion power (Q ~ 1e-6).
        raise ValueError("Fuel = 'DD' requires Operation_mode = 'Pulsed' "
                         "(heating powers prescribed, P_aux = P_fus / Q "
                         "is ill-posed at D-D fusion power).")
    _A_fuel = {'DT': 2.5, 'DD': 2.0}[Fuel]
    if abs(Atomic_mass - _A_fuel) > 1e-9 and verbose >= 1:
        print(f"[warn] Atomic_mass = {Atomic_mass} with Fuel = '{Fuel}' "
              f"(pure fuel mass: {_A_fuel}).")

    # Validate the bootstrap model early: fail fast at load time,
    # before any physics evaluation, with the valid options listed.
    if Bootstrap_choice not in ('Sauter-Redl', 'Segal'):
        raise ValueError(
            f"Unknown Bootstrap_choice: '{Bootstrap_choice}'. "
            "Valid options: 'Sauter-Redl', 'Segal'."
        )

    # ── Validate manual superconductor option ─────────────────────────────────
    if Supra_choice == 'Manual':
        if J_wost_Manual is None or J_wost_Manual <= 0:
            raise ValueError(
                "When Supra_choice='Manual', a valid J_wost_Manual > 0 must be provided "
                f"(current value: {J_wost_Manual}).  "
                "Example: J_wost_Manual = 25e6  # 25 MA/m²"
            )
    if Supra_choice_CS == 'Manual':
        if J_wost_Manual_CS is None or J_wost_Manual_CS <= 0:
            raise ValueError(
                "When the CS superconductor is 'Manual', a valid J_wost_Manual_CS "
                "(or J_wost_Manual) > 0 must be provided "
                f"(current value: {J_wost_Manual_CS}).")

    # ── Steel stress allowables ───────────────────────────────────────────────
    σ_TF = Steel(Chosen_Steel_TF, config.σ_manual_TF)
    σ_CS = Steel(Chosen_Steel_CS, config.σ_manual_CS)   # Unfatigued allowable [Pa]
    # NOTE: CS fatigue knockdown (config.fatigue_CS) is applied entirely inside
    # the CS solver (f_CS_ACAD / f_CS_refined / f_CS_CIRCE), within the stress
    # residual / f_sigma_diff closure. The effective allowable σ_eff is reduced
    # by fatigue_CS only when Operation_mode == 'Pulsed' AND the light case
    # (CS own electromagnetic load dominant, sigma_theta > 0) governs — which
    # is always the case in Wedging, and conditionally in Bucking (P_CS > 2 P_TF).
    # The structural safety factors config.SF_TF and config.SF_CS are likewise
    # applied inside the solvers at their entry (σ ← σ / SF). They compose
    # multiplicatively with fatigue_CS on the CS side (disjoint mechanisms:
    # primary static stress vs cyclic damage). Default 1.0 preserves the legacy
    # behaviour; ≈ 1.5 captures realistic CICC packing penalties.

    # Fraction of vertical tension carried by the TF winding pack
    if Choice_Buck_Wedg == "Wedging":
        omega_TF = 1 / 2
    elif Choice_Buck_Wedg in ("Bucking", "Plug"):
        omega_TF = 1
    else:
        raise ValueError(
            f"Unknown mechanical configuration: '{Choice_Buck_Wedg}'. "
            "Valid options: 'Wedging', 'Bucking', 'Plug'."
        )

    # ── Confinement scaling law coefficients ──────────────────────────────────
    (C_SL, alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
     alpha_R, alpha_B, alpha_n, alpha_I, alpha_P) = f_Get_parameter_scaling_law(Scaling_Law)
    # Optional Greenwald-fraction correction attached to the law (0 = none)
    fgw_slope, fgw_ref = f_Get_fGW_correction(Scaling_Law)

    # ── Plasma geometry ───────────────────────────────────────────────────────
    κ             = f_Kappa(R0 / a, Option_Kappa, κ_manual, ms)
    κ_95          = f_Kappa_95(κ)
    # Triangularity sign. 'positive' keeps the TREND correlation
    # delta = +0.6*(kappa-1) (ordinary D-shape). 'negative' flips the sign,
    # delta = -0.6*(kappa-1), so negative-triangularity configurations can be
    # requested from the deck; at the LUCIOLE elongation this lands near
    # delta ~ -0.5. The refined Miller geometry already supports delta < 0
    # downstream (arcsin(delta) contour), and delta_95 follows the same ITER
    # 1989 ratio with the sign preserved.
    δ_mag         = f_Delta(κ)
    _tri = getattr(config, 'Triangularity', 'positive')
    if _tri == 'negative':
        δ         = -δ_mag
    elif _tri == 'positive':
        δ         = δ_mag
    else:
        raise ValueError(f"Unknown Triangularity: '{_tri}'. "
                         "Valid options: 'positive', 'negative'.")
    # Optional manual override of the edge triangularity (signed). None keeps
    # the TREND correlation delta = ±0.6*(kappa-1) selected above.
    if getattr(config, 'Delta_manual', None) is not None:
        δ         = float(config.Delta_manual)
    δ_95          = f_Delta_95(δ)

    # Precompute Miller volume derivative V'(ρ) for refined geometry mode.
    # Vprime_data = (rho_grid, Vprime, V_total) is passed to f_nbar, f_pbar,
    # and f_plasma_volume so that all volume integrals use the same shaped
    # flux-surface Jacobian.  In Academic mode, Vprime_data = None triggers
    # the fast cylindrical-torus approximation in every downstream function.
    if Plasma_geometry == 'refined':
        # N_rho=500, N_theta=200 is the production grid.
        # The volume V converges to better than 0.1 % already at N_rho=100,
        # so the radial resolution is driven by dA_pol/drho near rho_95
        # rather than by V itself: the smoothstep5 kappa(rho) and
        # delta(rho) profiles curve sharply over [rho_95, 1], and finite-
        # difference evaluation of the Miller Jacobian on a coarse grid
        # (N_rho=100) would mis-resolve dA at rho=0.95 by approximately
        # 10 %, which propagates to q(0.95).  The earlier PCHIP shaping
        # had a slope discontinuity at rho_95 that made this resolution
        # requirement even tighter; smoothstep5 removed the discontinuity
        # but the fine grid is kept conservatively for the Jacobian.
        Vprime_data = precompute_Vprime(R0, a, κ, δ,
                                        geometry_model='refined',
                                        kappa_95=κ_95, delta_95=δ_95,
                                        N_rho=500, N_theta=200)
    else:
        Vprime_data = None

    # ── Radial labels of the deck: rho_tor -> rho_V ───────────────────────────
    # The deck gives the pedestal top, the deposition radii and the core
    # radiation boundary in the toroidal-flux label rho_tor, the coordinate
    # of transport codes and published profile fits. The profiles are
    # written in the volume label rho_V: the labels are converted once here,
    # every function below receives rho_V. Academic geometry: rho_V = rho_tor.
    rho_ped      = float(rho_V_of_rho_tor(rho_ped, Vprime_data))
    rho_EC       = float(rho_V_of_rho_tor(rho_EC, Vprime_data))
    rho_NBI      = float(rho_V_of_rho_tor(rho_NBI, Vprime_data))
    rho_rad_core = float(rho_V_of_rho_tor(rho_rad_core, Vprime_data))

    Volume_solution  = f_plasma_volume(R0, a, κ, δ, Vprime_data=Vprime_data)
    Surface_solution = f_first_wall_surface(R0, a, κ, δ,
                                            geometry_model=Plasma_geometry)

    # ── Multi-impurity radiation helpers (profile-integrated) ────────────────
    def _compute_P_line_split(nbar_loc, Tbar_loc, V_loc):
        """
        Sum profile-integrated line radiation over all impurity species,
        split into core (ρ < rho_rad_core) and total (ρ ∈ [0, 1]) contributions.

        Returns
        -------
        P_line_core  : float  Core line radiation [MW] — enters P_loss for τ_E.
        P_line_total : float  Total line radiation [MW] — enters P_sep.
        """
        P_core = 0.0
        P_total = 0.0
        for sp, fc in zip(imp_species_list, imp_conc_list):
            # N=150 gives <0.1 % error vs N=500 for smooth Mavrin L_z(T)
            # profiles, while reducing cost by ~3x in the hot solver loop.
            _Pc, _Pt = f_P_line_radiation_profile(
                sp, fc, nbar_loc, Tbar_loc, nu_n, nu_T, V_loc,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, rho_core=rho_rad_core, N=150)
            P_core  += _Pc
            P_total += _Pt
        return P_core, P_total

    # ── Elongation of the confinement scaling law ─────────────────────────────
    # Each law is evaluated with the elongation of its own fit, stored in the
    # registry (f_Get_kappa_convention). Most laws use the area elongation
    # κ_a ≡ V_p / (2π²R₀a²), the elongation of the ellipse with the same
    # cross-section as the shaped plasma: IPB98(y,2), ITPA20 and ITPA20-IL
    # (Verdoolaege et al., NF 61 (2021) 076006, §2.2.1). ITER89-P uses the
    # LCFS elongation.
    #
    # For a shaped plasma with triangularity, κ_a < κ_edge because D-shaping
    # reduces the cross-sectional area.
    #
    # MHD quantities (q*, β_P, etc.) continue to use κ_edge as they describe
    # the actual LCFS geometry, not the scaling-law convention.
    κ_a = Volume_solution / (2.0 * np.pi**2 * R0 * a**2)
    κ_SL = κ_a if f_Get_kappa_convention(Scaling_Law) == 'area' else κ

    # ── TF coil current density (cable-level sizing) ──────────────────────────
    tau_h = tau_h_HTS if Supra_choice == 'REBCO' else tau_h_LTS
    tau_h_CS = tau_h_HTS if Supra_choice_CS == 'REBCO' else tau_h_LTS

    # ── Auto-select f_void based on conductor technology ──────────────────────
    # REBCO tapes are stacked (no interstitial void between round strands).
    # LTS (Nb3Sn, NbTi) use CICC with ~33% void fraction in the strand bundle.
    # If the user explicitly sets f_void in GlobalConfig, that value is used;
    # if left at the default sentinel (None), auto-select based on Supra_choice,
    # and on Supra_choice_CS for the CS cable.
    if f_void is None:
        f_void = default_f_void(Supra_choice)
        f_void_CS = default_f_void(Supra_choice_CS)
        # Update config so downstream functions (f_CS_refined, _unpack_CS_config)
        # that read config.f_void directly also see the resolved value.
        config = dc_replace(config, f_void=f_void)
    else:
        f_void_CS = f_void

    N_TF, ripple, Delta_TF = Number_TF_coils(R0, a, b, ripple_adm, L_min)
    # N_TF is a float from Number_TF_coils; truncate to int before dividing so
    # that N_sub is an integer count of protection subdivisions, not a ratio.
    # A fractional N_sub would be silently rounded inside calculate_cable_current_density
    # (via int(N_sub)), potentially causing ~1 A/m² discontinuities at scan boundaries.
    # Delta_TF: extra outboard radial clearance imposed by port-access constraint [m].
    # Stored and forwarded to _resolve_build() so that R_TF_out is consistent
    # with the ripple model (r2 = R0 + a + b + Delta_TF) rather than R0 + a + b alone.
    N_sub = int(N_TF) // Dump_resistor_subdivision

    def _tf_cable(E_mag):
        """TF cable current density and fractions for a stored energy E_mag [J]."""
        return calculate_cable_current_density(
            sc_type=Supra_choice, B_peak=Bmax_TF, T_op=T_helium, E_mag=E_mag,
            I_cond=I_cond, V_max=V_max, N_sub=N_sub, tau_h=tau_h,
            f_He_pipe=f_He_pipe, f_void=f_void, f_In_cable=f_In_cable, f_In_WP=f_In_WP,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He,
            Marge_T_Nb3Sn=Marge_T_Nb3Sn, Marge_T_NbTi=Marge_T_NbTi,
            Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=J_wost_Manual if Supra_choice == 'Manual' else None)

    # Stored energy in the Princeton-D bore. The bore depends on the coil
    # thickness c, unknown here: thin-coil estimate (c = 0), refined by the
    # fixed point of the TF sizing below.
    E_mag_TF            = calculate_E_mag_TF(Bmax_TF, a, b, R0, 0.0, Delta_TF)
    result_J_TF         = _tf_cable(E_mag_TF)
    J_max_TF_conducteur = result_J_TF['J_wost']

    # ── On-axis magnetic field and alpha power ────────────────────────────────
    B0_solution = f_B0(Bmax_TF, a, b, R0, b_cover=c_BP)
    # Charged-product and neutron fractions of P_fus (constant for D-T,
    # branch-weighted on the deck profiles for D-D).
    f_charged, f_neutron = f_fuel_power_split(
        Fuel, Tbar, nu_T, nu_n, rho_ped=rho_ped, n_ped_frac=n_ped_frac,
        T_ped_frac=T_ped_frac, Vprime_data=Vprime_data, tau_i_e=tau_i_e)
    P_Alpha     = f_P_alpha(P_fus, f_charged)

    # =========================================================================
    #    SELF-CONSISTENT SOLVER  (f_alpha, Q)
    #
    #    Architecture rationale (v2 — refactored):
    #    -----------------------------------------------------------------
    #    In Pulsed mode, P_aux is fixed by the input (sum of heating sources
    #    or P_aux_input).  Q = P_fus / (P_aux + P_Ohm) is therefore *not* a
    #    free variable — it is fully determined once f_alpha (hence nbar,
    #    tau_E, Ip, I_Ohm, P_Ohm) is known.  The solver is thus 1-D on
    #    f_alpha, with an inner loop to converge P_Ohm self-consistently.
    #
    #    In Steady-State mode, P_aux = P_fus / Q, so Q is genuinely a free
    #    variable.  A 2-D solver on (f_alpha, Q) is used, with proper
    #    variable scaling and a finite-difference Jacobian.
    #
    #    Both branches use:
    #      • Absolute residuals (not percentages) for scipy compatibility
    #      • Variable scaling to ~O(1) for well-conditioned Jacobians
    #      • Anderson acceleration (via scipy) as a robust fallback
    #      • Per-method debug counter reset for clear diagnostics
    #      • A q(ρ) profile frozen during the solve, so that the residuals
    #        are pure functions of the unknowns, and an outer fixed point
    #        that updates q(ρ) at each converged solution
    #        (_solve_with_q_loop)
    #      • The chain re-evaluated at the retained solution, stored for
    #        the post-convergence outputs
    # =========================================================================

    # ── Dilution ceiling (shared by all solvers) ──────────────────────────────
    _fa_max = (1.0 - f_imp_dilution) / 2.0 - 0.01

    # ── Debug counter — reset between solver stages for clear output ─────────
    _dbg_counter = [0]
    _dbg_limit   = [5]   # mutable: how many calls to print per solver stage

    # ── Converged chain ───────────────────────────────────────────────────────
    # Physics-chain dict re-evaluated at the retained solution by each
    # solver, so that post-convergence code reads the profile-integrated
    # quantities of that solution (nbar, pbar, Ib, radiation, ...) without
    # recomputing them.
    _converged_chain = [None]

    # ── Solver tolerances ─────────────────────────────────────────────────────
    _P_OHM_ATOL  = 1e-6   # Pulsed inner loop: |P_Ohm(P) - P| [MW]
    _SS_FA_ATOL  = 1e-7   # Steady-State acceptance: |new_f_alpha - f_alpha| [-]
    _SS_Q_RTOL   = 1e-6   # Steady-State acceptance: |Q_phys - Q| / Q [-]
    _Q_LOOP_RTOL = 1e-3   # outer q(ρ) loop: relative change of I_bs [-]
    _Q_LOOP_MAX  = 8      # outer q(ρ) loop: maximum number of inner solves

    # ── q,j profile of the bootstrap ──────────────────────────────────────
    # q(ρ) dict from f_q_profile_academic or f_q_profile_refined (selected
    # by config.q_profile_mode), passed to f_Sauter_Redl_Ib so that the
    # bootstrap sees a realistic q(ρ) instead of the parabolic fallback.
    # The physics chain only reads it: q(ρ) is frozen during the inner
    # (f_alpha[, Q]) solve, whose residuals are then pure functions of the
    # unknowns. The outer fixed point of the solver dispatch updates it at
    # each converged inner solution.
    _q_profile_cache = [None]
    _q_failures      = []        # q-profile solver errors, reported after convergence

    def _solve_q_profile(Ip_loc, I_CD_loc, q95_loc, B0_loc, nbar_loc,
                         rho_CD_loc, delta_CD_loc,
                         n_rho=60, max_iter=10, tol=5e-3, damping=0.5,
                         q_init=None, Zeff_loc=None, n_ion_frac_loc=1.0):
        """
        Mode-aware front end for q,j profile evaluation.

        Routes to f_q_profile_academic or f_q_profile_refined according
        to config.q_profile_mode.  Returned dict has the same key set in
        both branches (q_arr, rho, j_total, j_Ohm, j_CD, j_bs, li, q0,
        I_bs, f_bs, n_iter, converged, kappa_arr) so call sites are
        agnostic to the chosen physics model.
        """
        if q_profile_mode == 'academic':
            return f_q_profile_academic(
                Ip_loc, q95_loc,
                R0, a, κ,
                alpha_J=alpha_J,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, kappa_95=κ_95, rho_95=0.95,
                n_rho=n_rho)
        elif q_profile_mode == 'refined':
            # Pass delta and delta_95 so that the on-the-fly Lp / <1/R^2>
            # path inside f_q_profile_refined (used when Vprime_data is
            # missing the inv_R2 6th element) recovers the same triangularity
            # as the main computation.  When Vprime_data is a full 6-tuple,
            # these parameters are ignored.
            return f_q_profile_refined(
                Ip_loc, I_CD_loc,
                R0, a, B0_loc, κ, nbar_loc, Tbar, Zeff_loc,
                nu_n, nu_T,
                trapped_fraction_model=trapped_fraction_model,
                eta_model=eta_model,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, kappa_95=κ_95, rho_95=0.95,
                delta=δ, delta_95=δ_95,
                rho_CD=rho_CD_loc, delta_CD=delta_CD_loc,
                q_init=q_init,
                n_rho=n_rho, max_iter=max_iter, tol=tol, damping=damping,
                tau_i_e=tau_i_e, n_ion_frac=n_ion_frac_loc)
        else:
            raise ValueError(
                f"Unknown q_profile_mode: '{q_profile_mode}'. "
                f"Valid options: 'academic', 'refined'.")

    def _bootstrap_current(nbar_loc, Ip_loc, q95_loc, Zeff_loc,
                           n_ion_frac_loc, q_profile):
        """
        Bootstrap current [MA] of the physics chain.

        Sauter-Redl reads the q(ρ) profile given as argument (None: parabolic
        fallback): q enters the collisionality nu*_e and the poloidal-flux
        derivative psi' ~ 1/q of j_bs. Segal does not depend on q(ρ).
        """
        if Bootstrap_choice == 'Sauter-Redl':
            return f_Sauter_Redl_Ib(
                R0, a, κ, B0_solution, nbar_loc, Tbar,
                q95_loc, Zeff_loc, nu_n, nu_T,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, kappa_95=κ_95,
                q_profile=q_profile,
                trapped_fraction_model=trapped_fraction_model, tau_i_e=tau_i_e,
                n_ion_frac=n_ion_frac_loc)
        if Bootstrap_choice == 'Segal':
            return f_Segal_Ib(
                nu_n, nu_T, a / R0, κ, nbar_loc, Tbar, R0, Ip_loc,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac, Vprime_data=Vprime_data)
        raise ValueError(
            f"Unknown Bootstrap_choice: '{Bootstrap_choice}'. "
            f"Valid options: 'Segal', 'Sauter-Redl'.")

    def _cd_efficiencies(nbar_loc, Zeff_loc, fa_loc):
        """CD figures of merit (LH, EC, NBI) [MA/(MW m²)] at a chain point."""
        eta_LH_loc  = f_etaCD_LH_physics(Tbar, Zeff_loc)
        eta_EC_loc  = f_etaCD_EC_physics(
            a, R0, Tbar, nbar_loc, Zeff_loc, nu_T, nu_n, rho_EC,
            theta_EC_pol_deg=config.theta_EC_pol_deg,
            rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
            Vprime_data=Vprime_data)
        eta_NBI_loc = f_etaCD_NBI_physics(
            A_beam, E_beam_keV, a, R0, Tbar, nbar_loc, Zeff_loc, nu_T, nu_n,
            rho_NBI, f_alpha=fa_loc, angle_NBI_deg=config.angle_NBI_deg,
            rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
            fuel=Fuel, Vprime_data=Vprime_data)
        return eta_LH_loc, eta_EC_loc, eta_NBI_loc

    def _cd_split(P_CD_loc, nbar_loc, eta_LH_loc, eta_EC_loc, eta_NBI_loc):
        """
        Power [MW] and driven current [MA] of each source. Pulsed Multi: the
        input source powers. Single source or Steady-State: the heating
        fractions of P_CD (f_CD_breakdown).
        """
        if Operation_mode == 'Pulsed' and CD_source == 'Multi':
            denom = R0 * nbar_loc
            return {'P_LH': P_LH, 'P_EC': P_ECRH, 'P_NBI': P_NBI, 'P_ICR': P_ICRH,
                    'I_LH':  eta_LH_loc  * P_LH   / denom if denom > 0 else 0.0,
                    'I_EC':  eta_EC_loc  * P_ECRH / denom if denom > 0 else 0.0,
                    'I_NBI': eta_NBI_loc * P_NBI  / denom if denom > 0 else 0.0}
        return f_CD_breakdown(config, P_CD_loc, R0, nbar_loc,
                              eta_LH_loc, eta_EC_loc, eta_NBI_loc)

    def _q_at_solution(res, fa_loc, q_init):
        """
        q,j profile at a converged chain point res of helium fraction fa_loc.

        Converged Ip, I_CD, q95 and nbar of the chain, Z_eff and ion fraction
        of fa_loc, and the Gaussian CD deposition of the per-source driven
        currents of the report (_cd_deposition).  Moderate resolution
        (n_rho = 80, max_iter = 20): l_i, q_0 and the j decomposition differ
        by less than 0.5 % from n_rho = 200 for smooth current profiles.  The
        figures use the same settings.  In academic mode the CD deposition,
        max_iter and tol are ignored (single deterministic pass).
        """
        Zeff_loc = _compute_Zeff_effective(config, fa_loc)
        bd = _cd_split(res['P_CD'], res['nbar'],
                       *_cd_efficiencies(res['nbar'], Zeff_loc, fa_loc))
        rho_CD_loc, delta_CD_loc = _cd_deposition(
            CD_source, bd['I_LH'], bd['I_EC'], bd['I_NBI'], rho_EC, rho_NBI)
        return _solve_q_profile(
            res['Ip'], res['I_CD'], res['q95'], B0_solution, res['nbar'],
            rho_CD_loc=rho_CD_loc, delta_CD_loc=delta_CD_loc,
            n_rho=80, max_iter=20, tol=1e-3, damping=0.5,
            q_init=q_init, Zeff_loc=Zeff_loc,
            n_ion_frac_loc=1.0 - fa_loc - f_imp_dilution + sum(imp_conc_list))

    def _reset_dbg(limit=5):
        """Reset debug call counter for a new solver stage."""
        _dbg_counter[0] = 0
        _dbg_limit[0]   = limit

    # =========================================================================
    #  PHYSICS CHAIN — shared core for all solvers
    #  Evaluates the full tokamak power balance for a given f_alpha, returning
    #  all quantities needed to close both the f_alpha and Q equations.
    # =========================================================================

    def _evaluate_physics_chain(f_alpha, P_Aux_in, P_Ohm_in, ss_mode=False):
        """
        Run the tokamak physics chain for a given helium ash fraction
        and heating power split.

        Parameters
        ----------
        f_alpha : float
            Helium ash fraction guess (dimensionless, 0 < f_alpha < 0.5).
        P_Aux_in : float
            Auxiliary heating power [MW] (fixed in Pulsed, = P_fus/Q in SS).
        P_Ohm_in : float
            Ohmic heating power guess [MW] (iterated in the inner loop).
        ss_mode : bool
            If True, use Steady-State current drive inversion:
            I_CD = Ip - Ib, P_CD = f_PCD(...), Q = P_fus/P_CD.
            If False (default), use Pulsed logic: P_CD fixed, I_Ohm derived.

        Returns
        -------
        dict with keys:
            nbar, pbar, nbar_line, P_rad_core, P_rad_total, tau_E, Ip, Ib,
            I_CD, I_Ohm, P_Ohm, P_CD, Q, new_f_alpha, P_line_core, P_line_total
            — all in standard D0FUS units (MA, MW, keV, 10^20 m^-3, etc.)
        """
        _dbg_counter[0] += 1
        _dbg = (verbose >= 2 and _dbg_counter[0] <= _dbg_limit[0])

        # Total ion density over n_e (quasi-neutrality with He ash and impurities)
        n_ion_frac_loc = 1.0 - f_alpha - f_imp_dilution + sum(imp_conc_list)

        # Volume-averaged density and pressure
        nbar_loc = f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, κ, δ,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac,
                          Vprime_data=Vprime_data, f_imp=f_imp_dilution,
                          tau_i_e=tau_i_e, fuel=Fuel)
        pbar_loc = f_pbar(nu_n, nu_T, nbar_loc, Tbar,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac,
                          Vprime_data=Vprime_data, tau_i_e=tau_i_e,
                          n_ion_frac=n_ion_frac_loc)
        # Vprime_data is required: nbar_loc is a volume average on the refined
        # Miller Jacobian, so the profile rebuilt inside f_nbar_line must use
        # the same volume element (see the f_nbar_line docstring).
        nbar_line_loc = f_nbar_line(nbar_loc, nu_n, rho_ped, n_ped_frac,
                                    Vprime_data=Vprime_data)

        # Radiative power losses [MW]
        # Fuel bremsstrahlung: uses Z_eff,fuel (D+T+He only) to avoid
        # double-counting impurity brem already in Mavrin L_z.
        #   Z_eff,fuel = (n_D+n_T)*1 + n_He*4) / ne
        #             = (1 - 2*f_He - f_imp) + 4*f_He = 1 + 2*f_He - f_imp
        # Impurity radiation (brem+line+recomb) is in P_line via Mavrin L_z.
        Zeff_fuel = 1.0 + 2.0 * f_alpha - f_imp_dilution
        # Full effective charge for transport/bootstrap/CD/runaway:
        # Z_eff = Z_eff_fuel + sum_j <Z_j>^2 c_j, or the user override.
        Zeff = Zeff_override if Zeff_override is not None else (Zeff_fuel + Zeff_imp_Z2)
        P_Brem_loc = f_P_bremsstrahlung(nbar_loc, Tbar, Zeff_fuel,
                                        Volume_solution, nu_n, nu_T,
                                        rho_ped=rho_ped,
                                        n_ped_frac=n_ped_frac,
                                        T_ped_frac=T_ped_frac,
                                        Vprime_data=Vprime_data)
        # Synchrotron: Albajar (2001) + Fidone (2001) wall-reflection.
        # Pedestal parameters are passed so that T₀ and ne₀ are the ACTUAL
        # on-axis values (from pedestal normalization). The K-factor shape
        # exponents are fitted to the actual core profile when a pedestal is
        # present (the user nu_T differs from the Albajar alpha_T by up to
        # 10 % rms in T/T0), see _albajar_profile_exponents.
        P_syn_loc  = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_loc,
                                     κ, nu_n, nu_T, r_synch,
                                     rho_ped=rho_ped,
                                     n_ped_frac=n_ped_frac,
                                     T_ped_frac=T_ped_frac,
                                     Vprime_data=Vprime_data)
        P_line_core_loc, P_line_total_loc = _compute_P_line_split(
            nbar_loc, Tbar, Volume_solution)

        # P_rad_core  → subtracted from P_heat in τ_E and scaling law (Ip)
        # P_rad_total → subtracted from P_heat to get P_sep (divertor load)
        #
        # coreradiationfraction (default 1.0, identical to PROCESS) scales
        # the core radiation term in P_loss only.  Values < 1 account for
        # partial re-absorption, non-coronal transport effects, and
        # scaling-law convention ambiguity.  Beware: the PROCESS 0.6 often
        # quoted is their core-region RADIUS (radius_plasma_core_norm),
        # analogue of rho_rad_core, not this fraction.
        # P_sep is always computed with the FULL P_rad_total (no scaling).
        _P_rad_core_raw = P_Brem_loc + P_syn_loc + P_line_core_loc
        P_rad_core_loc  = coreradiationfraction * _P_rad_core_raw
        P_rad_total_loc = P_Brem_loc + P_syn_loc + P_line_total_loc

        if _dbg:
            print(f"  [DBG #{_dbg_counter[0]}] f_alpha={f_alpha:.5f}")
            print(f"    nbar={nbar_loc:.4f}, pbar={pbar_loc:.4f} MPa, "
                  f"nbar_line={nbar_line_loc:.4f}")
            print(f"    P_Brem={P_Brem_loc:.1f}, P_syn={P_syn_loc:.1f}, "
                  f"P_line_core={P_line_core_loc:.1f}, "
                  f"P_line_total={P_line_total_loc:.1f} MW")
            print(f"    P_rad_core={P_rad_core_loc:.1f}, "
                  f"P_rad_total={P_rad_total_loc:.1f} MW")

        # Energy confinement time — uses P_rad_core (not total)
        tau_E_loc = f_tauE(pbar_loc, Volume_solution, P_Alpha,
                           P_Aux_in, P_Ohm_in, P_rad_core_loc)

        if _dbg:
            W_th_dbg = f_W_th(pbar_loc, Volume_solution) / 1e6  # MJ
            P_loss_dbg = P_Alpha + P_Aux_in + P_Ohm_in - P_rad_core_loc
            print(f"    W_th={W_th_dbg:.1f} MJ, "
                  f"P_loss={P_Alpha:.0f}+{P_Aux_in:.0f}+{P_Ohm_in:.0f}"
                  f"-{P_rad_core_loc:.0f}={P_loss_dbg:.1f} MW")
            print(f"    τ_E = {tau_E_loc:.3f} s")
            if not np.isfinite(tau_E_loc):
                print(f"    *** τ_E IS NaN/Inf — P_loss ≤ 0! ***")

        # Plasma current from scaling law inversion — uses P_rad_core
        # (same P_loss convention as τ_E for consistency with the scaling law)
        Ip_loc = f_Ip(tau_E_loc, R0, a, κ_SL, δ, nbar_line_loc,
                      B0_solution, Atomic_mass,
                      P_Alpha, P_Ohm_in, P_Aux_in, P_rad_core_loc,
                      H, C_SL,
                      alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
                      alpha_R, alpha_B, alpha_n, alpha_I, alpha_P,
                      fgw_slope=fgw_slope, fgw_ref=fgw_ref)

        if _dbg:
            print(f"    Ip={Ip_loc:.2f} MA")

        # Bootstrap current
        # q95 is needed by Sauter/Redl models and is always useful for the
        # post-convergence cache; f_q95 is algebraic so the overhead is negligible.
        # Sauter (2016) uses LCFS shaping (κ, δ); ITER_1989 uses 95%-surface
        # values (κ_95, δ_95).  Both sets are passed so f_q95 picks the right one.
        q95_loc = f_q95(B0_solution, Ip_loc, R0, a, κ, δ, κ_95, δ_95,
                        Option_q95=Option_q95)

        # The q(ρ) profile is frozen during the inner solve (outer fixed
        # point in _solve_with_q_loop).
        Ib_loc = _bootstrap_current(nbar_loc, Ip_loc, q95_loc, Zeff,
                                    n_ion_frac_loc, _q_profile_cache[0])

        # Current drive — different logic for Pulsed vs Steady-State
        if ss_mode:
            # Steady-State: I_Ohm = 0, all non-bootstrap current is driven.
            # P_CD is derived from the required I_CD and the CD efficiency.
            eta_CD_loc = f_etaCD_effective(
                config, a, R0, B0_solution, nbar_loc, Tbar, nu_n, nu_T,
                Zeff, f_alpha=f_alpha,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, rho_EC=rho_EC, rho_NBI=rho_NBI)
            I_Ohm_loc = 0.0
            I_CD_loc  = f_ICD(Ip_loc, Ib_loc, I_Ohm_loc)
            P_CD_loc  = f_PCD(R0, nbar_loc, I_CD_loc, eta_CD_loc)
            P_Ohm_loc = 0.0
            Q_loc     = f_Q(P_fus, P_CD_loc, P_Ohm_loc)
        else:
            # Pulsed mode: P_CD is fixed by the input source powers.
            if CD_source == 'Multi':
                P_CD_loc = P_LH + P_ECRH + P_NBI + P_ICRH
                eta_LH_loc, eta_EC_loc, eta_NBI_loc = _cd_efficiencies(
                    nbar_loc, Zeff, f_alpha)
                I_CD_loc = (f_I_CD(R0, nbar_loc, eta_LH_loc,  P_LH)
                          + f_I_CD(R0, nbar_loc, eta_EC_loc,  P_ECRH)
                          + f_I_CD(R0, nbar_loc, eta_NBI_loc, P_NBI))
            else:
                P_CD_loc   = P_aux_input
                eta_CD_loc = f_etaCD_effective(
                    config, a, R0, B0_solution, nbar_loc, Tbar, nu_n, nu_T,
                    Zeff, f_alpha=f_alpha,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, rho_EC=rho_EC, rho_NBI=rho_NBI)
                I_CD_loc   = f_I_CD(R0, nbar_loc, eta_CD_loc, P_CD_loc)

            I_Ohm_loc = f_I_Ohm(Ip_loc, Ib_loc, I_CD_loc)
            # Ohmic power from the profile-integrated neoclassical resistance
            # (R_eff from f_Reff), replacing the simplified straight-conductor
            # estimate f_P_Ohm.
            P_Ohm_loc = f_P_Ohm_integrated(
                I_Ohm_loc, a, κ, R0, Tbar, nbar_loc, Zeff, q95_loc, nu_T, nu_n,
                eta_model=eta_model,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data,
                trapped_fraction_model=trapped_fraction_model)
            Q_loc     = f_Q(P_fus, P_CD_loc, P_Ohm_loc)

        if _dbg:
            print(f"    Ib={Ib_loc:.2f}, I_CD={I_CD_loc:.2f}, "
                  f"I_Ohm={I_Ohm_loc:.2f}, P_Ohm={P_Ohm_loc:.3f}, "
                  f"Q={Q_loc:.2f}")

        # Helium ash fraction from confinement time.  The density profile
        # (nu_n, n_ped_frac) and the volume weight (Vprime_data) are the
        # ones used by f_nbar above, so that the alpha source of the ash
        # balance is exactly P_fus / E_fus (v2.8 correction: the former
        # call dropped the n_hat^2 weight and the Miller weight, which
        # under-counted the source by 1.2 to 1.6).
        new_fa_loc = f_He_fraction(
            nbar_loc, Tbar, tau_E_loc, C_Alpha, nu_T,
            rho_ped=rho_ped, T_ped_frac=T_ped_frac, tau_i_e=tau_i_e,
            f_imp=f_imp_dilution, nu_n=nu_n, n_ped_frac=n_ped_frac,
            Vprime_data=Vprime_data, fuel=Fuel)

        if _dbg:
            print(f"    new_f_alpha={new_fa_loc:.6f} (input={f_alpha:.6f})")

        # ── Sanitize: coerce any complex / non-finite scalar to NaN ──────
        # Unphysical parameter combinations (e.g. sqrt of negative argument
        # inside f_Ip, f_nbar, or f_tauE) can produce complex128 values.
        # If these propagate to scipy's solvers, a TypeError is raised at
        # the C level *before* Python's exception handling can intercept it,
        # which produces uncatchable console noise in Spyder/IPython.
        # Converting to real NaN here is clean and lets the existing
        # np.isfinite guards in the solver loops handle it gracefully.
        def _to_real(v):
            """Coerce a scalar to real float; complex / non-finite → NaN."""
            if isinstance(v, (complex, np.complexfloating)):
                return np.nan
            try:
                fv = float(np.real(v))
                return fv if np.isfinite(fv) else np.nan
            except (TypeError, ValueError):
                return np.nan

        return {
            'nbar': _to_real(nbar_loc), 'pbar': _to_real(pbar_loc),
            'nbar_line': _to_real(nbar_line_loc),
            'P_Brem': _to_real(P_Brem_loc), 'P_syn': _to_real(P_syn_loc),
            'P_rad_core': _to_real(P_rad_core_loc),
            'P_rad_total': _to_real(P_rad_total_loc),
            'P_line_core': _to_real(P_line_core_loc),
            'P_line_total': _to_real(P_line_total_loc),
            'tau_E': _to_real(tau_E_loc), 'q95': _to_real(q95_loc),
            'Ip': _to_real(Ip_loc), 'Ib': _to_real(Ib_loc),
            'I_CD': _to_real(I_CD_loc), 'I_Ohm': _to_real(I_Ohm_loc),
            'P_Ohm': _to_real(P_Ohm_loc), 'P_CD': _to_real(P_CD_loc),
            'Q': _to_real(Q_loc), 'new_f_alpha': _to_real(new_fa_loc),
        }

    # =========================================================================
    #  PULSED-MODE SOLVER — 1-D on f_alpha with inner P_Ohm loop
    # =========================================================================

    def _solve_pulsed(fa_hint=None):
        """
        Solve for f_alpha in Pulsed mode, with the q(ρ) profile frozen.

        In Pulsed mode, P_aux is fixed (user input).  The only self-consistent
        unknown is f_alpha.  Q is derived from Q = P_fus / (P_CD + P_Ohm).
        An inner loop on P_Ohm ensures Ohmic power consistency.

        Strategy: Brent's method on r(fa) = new_fa(fa) - fa.  The bracket is
        searched near fa_hint when given, else by a scan of [0, fa_max].  The
        chain evaluated at the root is stored in _converged_chain.

        Parameters
        ----------
        fa_hint : float or None
            Root of the previous q(ρ) iteration.

        Returns
        -------
        tuple (f_alpha, Q) or (nan, nan) on failure.
        """
        # Fixed auxiliary power for Pulsed mode
        if CD_source == 'Multi':
            P_Aux_fixed = abs(P_LH + P_ECRH + P_NBI + P_ICRH)
        else:
            P_Aux_fixed = abs(P_aux_input)

        # Mutable closure for P_Ohm warm-start: stores the converged Ohmic
        # power from the previous outer residual evaluation, close to the new
        # one since fa changes slowly between brentq steps.
        #
        # IMPORTANT: the inner loop converges to _P_OHM_ATOL, so that the
        # warm start has no visible effect and _pulsed_residual remains a pure
        # function of fa, as required by brentq.
        _P_Ohm_warmstart = [0.0]

        # Results of _pulsed_fixed_point by fa.  The map is pure for the
        # frozen q(ρ), so that brentq's first two calls (bracket ends) and
        # the final evaluation at the root reuse earlier evaluations.
        _fp_memo = {}

        def _pulsed_fixed_point(fa):
            """
            Fixed-point map: fa → new_fa, with the inner P_Ohm loop.

            The Ohmic power entering τ_E and Ip must equal the Ohmic power of
            the chain output.  The inner loop solves h(P) = P_Ohm(P) - P = 0
            from the warm start, with one substitution step and then secant
            steps, to _P_OHM_ATOL.  Typically 1 to 3 chain evaluations.

            Parameters
            ----------
            fa : float
                Helium ash fraction guess.

            Returns
            -------
            new_fa : float
                Updated helium ash fraction from the physics chain.
            Q_out  : float
                Derived energy gain factor.
            res    : dict
                Physics-chain dict at the converged Ohmic power.
            """
            fa = max(0.0, min(fa, _fa_max))
            if fa in _fp_memo:
                return _fp_memo[fa]

            # Warm start, or 0 if a NaN from an unphysical scan point (e.g.
            # fa near fa_max where DT dilution kills the reaction and the
            # chain returns NaN) corrupted it.
            P_in = _P_Ohm_warmstart[0] if np.isfinite(_P_Ohm_warmstart[0]) else 0.0
            res = _evaluate_physics_chain(fa, P_Aux_fixed, P_in)
            P_prev = h_prev = None
            for _inner in range(12):
                h = res['P_Ohm'] - P_in
                # Converged, or NaN from the chain
                if not np.isfinite(h) or abs(h) <= _P_OHM_ATOL:
                    break
                if h_prev is None or h == h_prev:
                    P_next = res['P_Ohm']                               # substitution
                else:
                    P_next = P_in - h * (P_in - P_prev) / (h - h_prev)  # secant
                P_prev, h_prev = P_in, h
                P_in = P_next if P_next >= 0.0 else 0.5 * P_prev
                res = _evaluate_physics_chain(fa, P_Aux_fixed, P_in)

            # Update warm-start for the next outer call — guard against NaN
            if np.isfinite(res['P_Ohm']):
                _P_Ohm_warmstart[0] = res['P_Ohm']

            _fp_memo[fa] = (res['new_f_alpha'], res['Q'], res)
            return _fp_memo[fa]

        def _pulsed_residual(fa):
            """Scalar residual r(fa) = new_fa - fa (absolute, not %)."""
            new_fa, _, _ = _pulsed_fixed_point(fa)
            return new_fa - fa

        # ── Brent's method on f_alpha residual ─────────────────────────────────
        # r(fa) = new_fa(fa) - fa is a scalar continuous function on [1e-4, fa_max].
        # Physically, r is positive at small fa (too little dilution → ash builds up)
        # and negative near fa_max (extreme dilution kills the reaction rate →
        # the chain predicts lower ash).  Brent's method combines superlinear
        # convergence (inverse quadratic interpolation / secant) with the safety
        # of bisection, converging in ~8–12 residual evaluations for typical cases.
        _reset_dbg(5)
        if verbose >= 2:
            print("[SOLVER-Pulsed] Brent solver on f_alpha residual")

        def _safe_residual(fa):
            """Evaluate residual, returning NaN on any failure."""
            try:
                return _pulsed_residual(fa)
            except Exception:
                return np.nan

        # -- Find a valid bracket [fa_lo, fa_hi] where r changes sign -----------
        bracket_found = False
        fa_lo = fa_hi = r_lo = r_hi = np.nan

        # Near the root of the previous q(ρ) iteration, which moves little
        # when only q(ρ) changes: steps of 2 %, 4 %, 8 %, ... of fa_hint in
        # the direction of the root, until r changes sign.
        if fa_hint is not None and np.isfinite(fa_hint) and fa_hint > 1e-4:
            _fa_a = fa_hint
            _r_a  = _safe_residual(_fa_a)
            _step = (0.02 if _r_a > 0 else -0.02) * fa_hint
            for _ in range(8):
                if not np.isfinite(_r_a):
                    break
                _fa_b = min(max(_fa_a + _step, 1e-4), _fa_max)
                _r_b  = _safe_residual(_fa_b)
                if np.isfinite(_r_b) and _r_a * _r_b <= 0 and _fa_b != _fa_a:
                    (fa_lo, r_lo), (fa_hi, r_hi) = sorted(
                        [(_fa_a, _r_a), (_fa_b, _r_b)])
                    bracket_found = True
                    break
                if _fa_b in (1e-4, _fa_max):
                    break
                _fa_a, _r_a = _fa_b, _r_b
                _step *= 2.0

        # Otherwise scan [0, fa_max]: evaluate points incrementally and stop
        # as soon as a sign change is detected.  This avoids wasting chain
        # evaluations on the full scan when the bracket is found early
        # (typically at the 2nd–3rd point for reactor-class tokamaks where
        # fa ~ 0.02).
        _scan_pts = np.unique(np.concatenate([
            [1e-4],
            np.linspace(0.01, min(0.20, _fa_max), 10),
            [_fa_max],
        ]))

        if not bracket_found:
            _prev_fa = None
            _prev_r  = None

            for _sfa in _scan_pts:
                _sr = _safe_residual(_sfa)
                if verbose >= 2:
                    _sr_str = f"{_sr:+.3e}" if np.isfinite(_sr) else "NaN"
                    print(f"  [bracket scan] fa={_sfa:.5f}  r={_sr_str}")

                # Ash fraction below the first scan point (D-D devices, or D-T at
                # very low fusion power): the root lies in [0, 1e-4], where
                # r(0) = new_fa(0) > 0 by construction.
                if _sfa == _scan_pts[0] and np.isfinite(_sr) and _sr < 0:
                    _r0 = _safe_residual(0.0)
                    if np.isfinite(_r0) and _r0 > 0:
                        fa_lo, r_lo = 0.0, _r0
                        fa_hi, r_hi = _sfa, _sr
                        bracket_found = True
                        break

                # Check for sign change with previous finite point
                if np.isfinite(_sr) and _prev_r is not None and np.isfinite(_prev_r):
                    if _sr * _prev_r < 0:
                        fa_lo, r_lo = _prev_fa, _prev_r
                        fa_hi, r_hi = _sfa,     _sr
                        bracket_found = True
                        break

                # Update previous finite point
                if np.isfinite(_sr):
                    _prev_fa = _sfa
                    _prev_r  = _sr

        if not bracket_found:
            # No root exists in the physical range — return failure
            if verbose >= 1:
                print("[SOLVER-Pulsed] No sign change found in f_alpha residual. "
                      "No self-consistent solution exists for this configuration.")
                if verbose < 2:
                    # Re-run scan with output for diagnostic
                    print("  Diagnostic scan (re-run with verbose=2 for details):")
                    for _sfa in _scan_pts:
                        _sr = _safe_residual(_sfa)
                        _sr_str = f"{_sr:+.4e}" if np.isfinite(_sr) else "NaN"
                        print(f"  fa={_sfa:.4f} → r={_sr_str}")
            return np.nan, np.nan

        if verbose >= 2:
            print(f"  [bracket] fa ∈ [{fa_lo:.5f}, {fa_hi:.5f}], "
                  f"r ∈ [{r_lo:+.3e}, {r_hi:+.3e}]")

        # -- Call brentq on the validated bracket --------------------------------
        _eval_count = [0]
        def _counted_residual(fa):
            """Wrapper around _pulsed_residual that counts evaluations."""
            _eval_count[0] += 1
            r = _pulsed_residual(fa)
            if verbose >= 2 and _eval_count[0] <= 15:
                print(f"  [Brent {_eval_count[0]:3d}] fa={fa:.6f}  r={r:+.3e}")
            return r

        # Absolute tolerance 1e-6 for reactor-level ash. For a bracket
        # starting at 0 the root is ~r(0), so the tolerance follows it.
        _fa_xtol = 1e-6 if fa_lo > 0 else min(1e-6, 1e-3 * r_lo)
        try:
            fa_sol = brentq(_counted_residual, fa_lo, fa_hi,
                            xtol=_fa_xtol, rtol=1e-8, maxiter=50)
        except (ValueError, RuntimeError) as _brentq_err:
            # brentq failed (bracket lost, NaN in residual, or convergence failure)
            if verbose >= 1:
                print(f"[SOLVER-Pulsed] brentq failed: {_brentq_err}")
            return np.nan, np.nan

        # Final evaluation at the root: the chain of the retained solution is
        # stored for the post-convergence outputs.  The inner P_Ohm loop in
        # _pulsed_fixed_point already guarantees self-consistency, so a
        # single call is sufficient.
        _, Q_out, _converged_chain[0] = _pulsed_fixed_point(fa_sol)

        if verbose >= 1:
            print(f"[SOLVER-Pulsed] Brent converged in {_eval_count[0]} evaluations: "
                  f"f_alpha={fa_sol:.6f}, Q={Q_out:.2f}")

        return fa_sol, Q_out

    # =========================================================================
    #  STEADY-STATE SOLVER — 2-D on (f_alpha, Q) with scaled variables
    # =========================================================================

    def _solve_steady_state(x_hint=None):
        """
        Solve the coupled (f_alpha, Q) system for Steady-State operation,
        with the q(ρ) profile frozen.

        In SS mode, P_aux = P_fus / Q, so Q is a genuine free variable.
        The 2-D system is:
          R1 = new_f_alpha(fa, Q) - fa     = 0
          R2 = Q_physics(fa, Q)   - Q      = 0

        Variables are scaled to ~O(1) for well-conditioned Jacobians:
          x = [fa / fa_scale, Q / Q_scale]

        Strategy, the first accepted solution is returned:
          1. hybr with a finite-difference Jacobian, from x_hint then from
             default guesses
          2. Levenberg-Marquardt from the same guesses
          3. Anderson mixing (2-D fixed-point)
        A solution is accepted when the chain re-evaluated at (fa, Q) closes
        both equations within _SS_FA_ATOL and _SS_Q_RTOL.  That chain is
        stored in _converged_chain.

        Parameters
        ----------
        x_hint : tuple (f_alpha, Q) or None
            Solution of the previous q(ρ) iteration.

        Returns
        -------
        tuple (f_alpha, Q) or (nan, nan) on failure.
        """
        # Characteristic scales for variable normalisation
        fa_scale = 0.1     # f_alpha ~ O(0.1)
        Q_scale  = 50.0    # Q ~ O(10–100) for SS tokamaks

        def _ss_residual_scaled(x_scaled):
            """
            Residual function in scaled coordinates.

            Parameters
            ----------
            x_scaled : array [fa/fa_scale, Q/Q_scale]

            Returns
            -------
            array [R_fa/fa_scale, R_Q/Q_scale] — absolute residuals, scaled.
            """
            fa = x_scaled[0] * fa_scale
            Q  = x_scaled[1] * Q_scale

            # Physical bounds check
            if not (1e-4 <= fa <= _fa_max) or Q < 0.1:
                return np.array([1e6, 1e6])

            try:
                P_Aux_ss = abs(P_fus / Q)
                res = _evaluate_physics_chain(fa, P_Aux_ss, 0.0, ss_mode=True)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                return np.array([1e6, 1e6])

            if not (np.isfinite(res['new_f_alpha']) and np.isfinite(res['Q'])):
                return np.array([1e6, 1e6])

            # Absolute residuals, normalised by the same scales
            R_fa = (res['new_f_alpha'] - fa) / fa_scale
            R_Q  = (res['Q'] - Q)            / Q_scale

            return np.array([R_fa, R_Q])

        def _ss_jacobian_scaled(x_scaled):
            """
            Forward-difference Jacobian in scaled coordinates.

            The Jacobian is nearly diagonal because f_alpha and Q are weakly
            coupled (f_alpha affects Q only through fuel dilution, and Q
            affects f_alpha only through the slight change in P_aux → τ_E):

            J ≈ [[-1,   0 ],    (dominant: ∂R_fa/∂fa ≈ -1)
                 [ 0,  -1 ]]    (dominant: ∂R_Q/∂Q   ≈ -1)

            A diagonal element below 0.1 in magnitude is replaced by -1.
            """
            r0 = _ss_residual_scaled(x_scaled)

            J = np.zeros((2, 2))
            for j in range(2):
                h = max(1e-4 * abs(x_scaled[j]), 1e-5)
                x_pert = x_scaled.copy()
                x_pert[j] += h
                r1 = _ss_residual_scaled(x_pert)
                J[:, j] = (r1 - r0) / h

            # Regularise if any diagonal element is too small
            for j in range(2):
                if abs(J[j, j]) < 0.1:
                    J[j, j] = -1.0

            return J

        def _accept(fa, Q):
            """
            Re-evaluate the chain at (fa, Q).  If both equations close within
            the tolerances, store the chain and return True.
            """
            if not (np.isfinite(fa) and np.isfinite(Q)
                    and 0.0 < fa < _fa_max and Q > 0.0):
                return False
            try:
                res = _evaluate_physics_chain(fa, abs(P_fus / Q), 0.0,
                                              ss_mode=True)
            except Exception:
                return False
            if not (np.isfinite(res['new_f_alpha']) and np.isfinite(res['Q'])):
                return False
            if (abs(res['new_f_alpha'] - fa) <= _SS_FA_ATOL
                    and abs(res['Q'] - Q) <= _SS_Q_RTOL * Q):
                _converged_chain[0] = res
                return True
            return False

        initial_guesses_scaled = [
            np.array([0.05 / fa_scale, 50.0 / Q_scale]),
            np.array([0.05 / fa_scale, 20.0 / Q_scale]),
            np.array([0.05 / fa_scale, 100.0 / Q_scale]),
            np.array([0.10 / fa_scale, 50.0 / Q_scale]),
        ]
        if x_hint is not None:
            initial_guesses_scaled.insert(
                0, np.array([x_hint[0] / fa_scale, x_hint[1] / Q_scale]))

        # ── Step 1: hybr with the finite-difference Jacobian ──────────────────
        _reset_dbg(5)
        if verbose >= 2:
            print("[SOLVER-SS] Step 1: hybr with finite-difference Jacobian")

        for guess in initial_guesses_scaled:
            _reset_dbg(3)
            try:
                result = root(_ss_residual_scaled, guess,
                              method='hybr', jac=_ss_jacobian_scaled,
                              tol=1e-8,
                              options={'maxfev': 300})
            except Exception:
                continue
            fa_sol = result.x[0] * fa_scale
            Q_sol  = result.x[1] * Q_scale
            if _accept(fa_sol, Q_sol):
                if verbose >= 1:
                    print(f"[SOLVER-SS] Converged (hybr+jac): f_alpha={fa_sol:.5f}, "
                          f"Q={Q_sol:.2f}")
                return fa_sol, Q_sol

        # ── Step 2: lm solver (no Jacobian needed) ────────────────────────────
        _reset_dbg(3)
        if verbose >= 2:
            print("[SOLVER-SS] Step 2: Levenberg-Marquardt")

        for guess in initial_guesses_scaled:
            _reset_dbg(2)
            try:
                result = root(_ss_residual_scaled, guess,
                              method='lm', tol=1e-8)
            except Exception:
                continue
            fa_sol = result.x[0] * fa_scale
            Q_sol  = result.x[1] * Q_scale
            if _accept(fa_sol, Q_sol):
                if verbose >= 1:
                    print(f"[SOLVER-SS] Converged (lm): f_alpha={fa_sol:.5f}, "
                          f"Q={Q_sol:.2f}")
                return fa_sol, Q_sol

        # ── Step 3: Anderson-accelerated 2-D fixed-point ──────────────────────
        _reset_dbg(3)
        if verbose >= 2:
            print("[SOLVER-SS] Step 3: Anderson mixing (2-D fixed-point)")

        fa = 0.05
        Q_fp = 50.0
        m_depth = 4
        x_hist = []   # stores x vectors
        g_hist = []   # stores g(x) - x residuals

        for k in range(100):
            P_Aux_ss = abs(P_fus / Q_fp)
            try:
                res = _evaluate_physics_chain(fa, P_Aux_ss, 0.0, ss_mode=True)
            except Exception:
                res = None
            if res is None or not (np.isfinite(res['new_f_alpha'])
                                   and np.isfinite(res['Q'])):
                # Failed or non-finite evaluation: smaller ash fraction and
                # new mixing history.
                fa = max(1e-4, 0.8 * fa)
                x_hist.clear()
                g_hist.clear()
                continue

            new_fa = res['new_f_alpha']
            new_Q  = res['Q']

            r_fa = new_fa - fa
            r_Q  = new_Q  - Q_fp

            if (k < 10 or k % 10 == 0) and verbose >= 2:
                print(f"  [Anderson-2D {k:3d}] fa={fa:.6f}, Q={Q_fp:.2f}  "
                      f"r_fa={r_fa:+.2e}, r_Q={r_Q:+.2e}")

            if abs(r_fa) <= _SS_FA_ATOL and abs(r_Q) <= _SS_Q_RTOL * Q_fp:
                _converged_chain[0] = res
                if verbose >= 1:
                    print(f"[SOLVER-SS] Anderson-2D converged after {k+1} iter: "
                          f"f_alpha={fa:.6f}, Q={Q_fp:.2f}")
                return fa, Q_fp

            # Anderson mixing in 2-D
            x_vec = np.array([fa, Q_fp])
            g_vec = np.array([new_fa, new_Q])
            r_vec = g_vec - x_vec

            x_hist.append(x_vec.copy())
            g_hist.append(r_vec.copy())

            if len(x_hist) > m_depth:
                x_hist.pop(0)
                g_hist.pop(0)

            if len(x_hist) >= 2:
                # Build the Anderson system: min ||r_k - Σ θ_j Δr_j||
                m = len(x_hist) - 1
                dR = np.column_stack([g_hist[j+1] - g_hist[j] for j in range(m)])
                dX = np.column_stack([x_hist[j+1] - x_hist[j] for j in range(m)])

                try:
                    # Solve the least-squares problem for mixing coefficients
                    theta, _, _, _ = np.linalg.lstsq(dR, g_hist[-1], rcond=None)
                    x_next = g_vec - dX @ theta
                except np.linalg.LinAlgError:
                    x_next = g_vec  # fallback to simple Picard

                fa   = max(1e-4, min(x_next[0], _fa_max))
                Q_fp = max(1.0, x_next[1])
            else:
                # Not enough history — simple Picard with damping
                fa   = max(1e-4, min(0.7 * fa + 0.3 * new_fa, _fa_max))
                Q_fp = max(1.0, 0.7 * Q_fp + 0.3 * new_Q)

        # ── Diagnostic ────────────────────────────────────────────────────────
        if verbose >= 1:
            print("[SOLVER-SS] All methods failed. Diagnostic:")
            for _fa in [0.02, 0.05, 0.10]:
                for _Q in [10, 30, 50, 100]:
                    try:
                        P_Aux_ss = P_fus / _Q
                        _res = _evaluate_physics_chain(_fa, P_Aux_ss, 0.0, ss_mode=True)
                        print(f"  fa={_fa:.3f}, Q={_Q:.0f} → new_fa={_res['new_f_alpha']:.4f}, "
                              f"Q_phys={_res['Q']:.1f}")
                    except Exception as e:
                        print(f"  fa={_fa:.3f}, Q={_Q:.0f} → ERROR: {e}")

        return np.nan, np.nan

    # =========================================================================
    #  OUTER FIXED POINT ON q(ρ)
    # =========================================================================

    def _solve_with_q_loop(solve):
        """
        Outer fixed point on the q(ρ) profile of the bootstrap.

        The inner solver finds the operating point with q(ρ) frozen.  q(ρ) is
        then solved at the converged currents (_q_at_solution), warm-started
        from the frozen profile.  The loop stops when the bootstrap of the
        converged chain, re-evaluated on the new profile, changes by less
        than _Q_LOOP_RTOL: the reported chain and the reported profile (l_i)
        then describe the same plasma.  Otherwise the new profile is frozen
        and the operating point solved again, from the previous solution.

        Parameters
        ----------
        solve : callable
            Inner solver, solve(hint) -> (f_alpha, Q), with hint the
            (f_alpha, Q) of the previous iteration or None.

        Returns
        -------
        (f_alpha, Q, q_dict), q_dict being the q(ρ) solved at the retained
        solution, or (nan, nan, None) when the inner solver fails.
        """
        hint = None
        for k in range(1, _Q_LOOP_MAX + 1):
            fa_k, Q_k = solve(hint)
            if not (np.isfinite(fa_k) and np.isfinite(Q_k)):
                return np.nan, np.nan, None
            res = _converged_chain[0]
            q_new = _q_at_solution(res, fa_k, _q_profile_cache[0])
            if not np.all(np.isfinite(q_new['q_arr'])):
                if verbose >= 1:
                    print("[warn] non-finite q(ρ) at the converged solution: "
                          "the bootstrap keeps the previous profile.")
                return fa_k, Q_k, q_new
            Ib_new = _bootstrap_current(
                res['nbar'], res['Ip'], res['q95'],
                _compute_Zeff_effective(config, fa_k),
                1.0 - fa_k - f_imp_dilution + sum(imp_conc_list), q_new)
            dIb = abs(Ib_new - res['Ib'])
            if verbose >= 2:
                print(f"[SOLVER] q(ρ) iteration {k}: I_bs = {res['Ib']:.4f} MA, "
                      f"{Ib_new:.4f} MA on the new profile")
            if dIb <= _Q_LOOP_RTOL * abs(res['Ib']) + 1e-6:
                if verbose >= 1:
                    print(f"[SOLVER] q(ρ) profile converged after {k} "
                          f"inner solve(s)")
                return fa_k, Q_k, q_new
            if k < _Q_LOOP_MAX:
                _q_profile_cache[0] = q_new
                hint = (fa_k, Q_k)
        if verbose >= 1:
            print(f"[warn] q(ρ) outer loop not converged after {_Q_LOOP_MAX} "
                  f"inner solves (last relative change of I_bs "
                  f"{dIb / max(abs(res['Ib']), 1e-6):.1e}).")
        return fa_k, Q_k, q_new

    # =========================================================================
    #  MAIN SOLVER DISPATCH
    # =========================================================================

    # ── Pre-solver validation ─────────────────────────────────────────────────
    _reset_dbg(3)
    if Operation_mode == 'Pulsed':
        _P_aux_est = (P_LH + P_ECRH + P_NBI + P_ICRH) if CD_source == 'Multi' else P_aux_input
        try:
            _pre_res = _evaluate_physics_chain(0.05, _P_aux_est, 0.0)
            _r_fa = _pre_res['new_f_alpha'] - 0.05
            if verbose >= 2:
                print(f"[SOLVER] Pre-check (Pulsed): f_alpha residual={_r_fa:+.4e}, "
                      f"Q_derived={_pre_res['Q']:.2f}")
        except Exception as _pre_exc:
            import traceback as _tb
            if verbose >= 1:
                print("\n[ERROR] Pre-solver validation failed:")
                _tb.print_exc()
            raise RuntimeError(
                f"Physics chain is broken at the initial guess "
                f"({type(_pre_exc).__name__}: {_pre_exc}). "
                "Fix this error before running D0FUS."
            ) from _pre_exc

        def _solve_inner(hint):
            return _solve_pulsed(None if hint is None else hint[0])

    elif Operation_mode == 'Steady-State':
        try:
            _pre_res = _evaluate_physics_chain(0.05, P_fus / 50.0, 0.0, ss_mode=True)
            _r_fa = _pre_res['new_f_alpha'] - 0.05
            _r_Q  = _pre_res['Q'] - 50.0
            if verbose >= 2:
                print(f"[SOLVER] Pre-check (SS): f_alpha res={_r_fa:+.4e}, "
                      f"Q res={_r_Q:+.2e}")
        except Exception as _pre_exc:
            import traceback as _tb
            if verbose >= 1:
                print("\n[ERROR] Pre-solver validation failed:")
                _tb.print_exc()
            raise RuntimeError(
                f"Physics chain is broken at the initial guess "
                f"({type(_pre_exc).__name__}: {_pre_exc}). "
                "Fix this error before running D0FUS."
            ) from _pre_exc

        _solve_inner = _solve_steady_state

    else:
        raise ValueError(
            f"Unknown Operation_mode: '{Operation_mode}'. "
            "Valid options: 'Pulsed', 'Steady-State'."
        )

    # Starting q(ρ) of the outer fixed point, solved at the pre-check point.
    # On failure the bootstrap uses the parabolic fallback.
    if all(np.isfinite(_pre_res[k]) for k in ('Ip', 'I_CD', 'q95', 'nbar')):
        try:
            _q_start = _q_at_solution(_pre_res, 0.05, None)
            if np.all(np.isfinite(_q_start['q_arr'])):
                _q_profile_cache[0] = _q_start
        except Exception as exc:
            _q_failures.append(f"{type(exc).__name__}: {exc}")

    f_alpha_solution, Q_solution, _q_solution = _solve_with_q_loop(_solve_inner)

    # ── Return nan tuple on convergence failure ───────────────────────────────
    _nan = np.nan
    if np.isnan(f_alpha_solution) or np.isnan(Q_solution):
        return (
            _nan, _nan, _nan,                  # B0, B_CS, B_pol
            _nan, _nan,                          # tauE, W_th
            _nan, _nan, _nan,                  # Q, Volume, Surface
            _nan, _nan, _nan, _nan,            # Ip, Ib, I_CD, I_Ohm
            _nan, _nan, _nan, _nan,            # nbar, nbar_line, nG, pbar
            _nan, _nan, _nan,                  # betaN, betaT, betaP
            _nan, _nan,                          # qstar, q95
            _nan, _nan, _nan, _nan, _nan, _nan,  # P_CD, P_sep, P_Thresh, eta_CD, P_elec, P_wallplug
            _nan, _nan, _nan, _nan,            # cost, P_Brem, P_syn, P_line
            _nan,                               # P_line_core
            _nan, _nan, _nan, _nan, _nan,      # heat, heat_par, heat_pol, lambda_q, q_target
            _nan, _nan,                          # P_wall_rad, P_wall_div
            _nan,                               # Gamma_n
            _nan, _nan,                          # f_alpha, tau_alpha
            _nan, _nan,                          # J_TF, J_CS
            _nan, _nan, _nan, _nan, _nan, _nan, _nan,  # TF radial build + stresses
            _nan, _nan, _nan, _nan, _nan, _nan, _nan,  # CS radial build + stresses
            _nan, _nan, _nan, _nan,            # r_minor, r_sep, r_c, r_d
            _nan, _nan, _nan, _nan,            # κ, κ_95, δ, δ_95
            _nan, _nan, _nan, _nan, _nan, _nan, _nan,  # ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS, Vloop, li
            _nan, _nan, _nan,                  # eta_LH, eta_EC, eta_NBI
            _nan, _nan, _nan, _nan,            # P_LH, P_EC, P_NBI, P_ICR
            _nan, _nan, _nan,                  # I_LH, I_EC, I_NBI
            _nan, _nan, _nan, _nan, _nan, _nan,  # f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF
            _nan, _nan, _nan, _nan, _nan, _nan,  # f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS
            _nan, _nan, _nan, _nan,              # beta_fast_alpha, betaN_total, tau_sd_alpha, W_fast_alpha
            _nan, _nan,                          # V_TF_one, V_CS_geom
            _nan, _nan, _nan, _nan, _nan,        # V_steel_TF, V_sc_TF, V_cu_TF, V_He_TF, V_In_TF
            _nan, _nan, _nan, _nan, _nan,        # V_steel_CS, V_sc_CS, V_cu_CS, V_He_CS, V_In_CS
            _nan, _nan,                          # L_cable_TF, L_cable_CS
            _nan, _nan,                          # n_sc_TF, n_sc_CS
            _nan, _nan,                          # L_sc_strand_TF, L_sc_strand_CS
            _nan, _nan, _nan, _nan, _nan,        # M_steel_TF, M_sc_TF, M_cu_TF, M_In_TF, M_total_TF
            _nan, _nan, _nan, _nan, _nan,        # M_steel_CS, M_sc_CS, M_cu_CS, M_In_CS, M_total_CS
            _nan,                                # V_blanket
            _nan, _nan,                          # t_life_bl_fpy, t_life_div_fpy
            _nan, _nan,                          # t_life_bl_yr, t_life_div_yr
            _nan, _nan,                          # T_op_limit, dt_rep_eff
            _nan, _nan,                          # Av, CF
            _nan,                                # V_rb_SOL
            _nan, _nan,                          # V_rb_FW, V_rb_BB
            _nan, _nan, _nan,                    # V_rb_shield, V_rb_VV, V_rb_gap_TF
            _nan,                                # V_rb_divertor
            _nan, _nan, _nan, _nan, _nan, _nan,  # M_rb_FW, M_rb_BB, M_rb_shield, M_rb_VV, M_rb_divertor, M_rb_total
            {},                                  # divertor_solution (empty on failure)
            _nan, _nan, _nan, _nan, _nan,        # plant electrical balance (150-164)
            _nan, _nan, _nan, _nan, _nan,
            _nan, _nan, _nan, _nan, _nan,
            _nan, _nan, _nan, _nan, _nan, _nan,  # plant outputs 165–170
            _nan, _nan, _nan, _nan, _nan,       # phase outputs 171–175
        )

    # =========================================================================
    #    POST-CONVERGENCE OUTPUTS
    #    Full calculation of all derived quantities after (f_alpha, Q) is known.
    # =========================================================================

    # Plasma thermodynamics — read from converged physics-chain cache
    # (avoids recomputing f_nbar, f_pbar, f_P_bremsstrahlung, f_P_synchrotron,
    #  and _compute_P_line_split, all of which involve numerical quadrature
    #  over the radial profile).
    _chain = _converged_chain[0]
    # Effective charge at the converged helium-ash fraction (override-aware).
    # Used by f_beta_fast_alpha, the CD efficiencies, f_P_Ohm and Magnetic_flux
    # below; consistent with the value used inside the converged physics chain.
    Zeff = _compute_Zeff_effective(config, f_alpha_solution)
    nbar_solution        = _chain['nbar']
    pbar_solution        = _chain['pbar']
    W_th_solution        = f_W_th(pbar_solution, Volume_solution)      # [J]
    nbar_line_solution   = _chain['nbar_line']
    P_Brem_solution      = _chain['P_Brem']
    P_syn_solution       = _chain['P_syn']
    P_line_core_solution = _chain['P_line_core']
    P_line_solution      = _chain['P_line_total']
    P_rad_core_solution  = _chain['P_rad_core']
    P_rad_total_solution = _chain['P_rad_total']

    # Heating power partition
    if Operation_mode == 'Steady-State':
        P_Aux_solution = P_fus / Q_solution
        P_Ohm_solution = 0
    elif Operation_mode == 'Pulsed':
        # In Multi-source mode, the actual injected power is the sum of all
        # source powers, NOT the default P_aux_input (which may differ).
        if CD_source == 'Multi':
            P_Aux_solution = P_LH + P_ECRH + P_NBI + P_ICRH
        else:
            P_Aux_solution = P_aux_input
        P_Ohm_solution = P_fus / Q_solution - P_Aux_solution
    else:
        if verbose >= 1:
            print("Choose a valid operation mode")

    # Energy confinement time, plasma current, q95, bootstrap — from cache
    # (avoids recomputing f_tauE, f_Ip, f_q95, and especially the expensive
    #  neoclassical bootstrap model f_Sauter_Redl_Ib which involves
    #  radial profile integration with trapped-particle corrections).
    tauE_solution = _chain['tau_E']
    Ip_solution   = _chain['Ip']
    q95_solution  = _chain['q95']
    Ib_solution   = _chain['Ib']

    # Alpha-particle confinement time — not computed inside the chain
    tau_alpha = f_tau_alpha(nbar_solution, Tbar, tauE_solution, C_Alpha, nu_T,
                            rho_ped=rho_ped, T_ped_frac=T_ped_frac)

    # Kink safety factor — algebraic, not in the chain
    qstar_solution  = f_qstar(a, B0_solution, R0, Ip_solution, κ)
    
    # --- Remaining MHD quantities (qstar/q95 already computed above) ---
    # Ampere convention (mu0 Ip / L_pol), consistent with f_beta_P;
    # the q95-based f_Bpol estimate is biased low by q*/q95 (~28 %).
    B_pol_solution  = f_Bpol_ampere(Ip_solution, a, κ)
    betaT_solution  = f_beta_T(pbar_solution, B0_solution)
    betaP_solution  = f_beta_P(a, κ, pbar_solution, Ip_solution)
    # Troyon normalisation uses the TOROIDAL beta, NOT the total-field beta:
    #   beta_N = beta_T * a * B0 / Ip   (Troyon 1984; Wesson 2011, §6.7).
    # betaN_solution is the thermal beta_N, reported as a diagnostic.
    betaN_solution  = f_beta_N(betaT_solution, a, B0_solution, Ip_solution)

    # Fast-alpha pressure contribution (Stix slowing-down model)
    # For D-D the fast charged products (3He, T, p) are treated as alphas.
    # Their pressure is negligible at D-D power levels.
    beta_fast_alpha, tau_sd_alpha, W_fast_alpha = f_beta_fast_alpha(
        P_Alpha, Tbar, nbar_solution, B0_solution, Volume_solution, nu_n, nu_T,
        rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
        Vprime_data=Vprime_data, tau_i_e=tau_i_e, A_DT=Atomic_mass, fuel=Fuel)
    # Toroidal beta INCLUDING the fast-alpha pressure — the MHD-relevant beta
    # for the Troyon limit (kink / ballooning / NTM modes respond to the TOTAL
    # pressure, thermal + fast).  betaN_total is the quantity compared against
    # betaN_limit in the scan / genetic / uncertainty feasibility checks.
    betaT_total     = betaT_solution + beta_fast_alpha
    betaN_total     = f_beta_N(betaT_total, a, B0_solution, Ip_solution)
    # Greenwald limit is defined in terms of line-averaged density;
    # f_nG returns n_G [1e20 m-3], which must be compared with nbar_line, not nbar_vol.
    nG_solution     = f_nG(Ip_solution, a) * Greenwald_limit
    nG_raw          = f_nG(Ip_solution, a)  # raw Ip/(πa²), no limit factor

    # Current drive and power balance
    # ── Individual CD efficiencies (computed once, reused in breakdown) ────────
    eta_LH_solution, eta_EC_solution, eta_NBI_solution = _cd_efficiencies(
        nbar_solution, Zeff, f_alpha_solution)

    if Operation_mode == 'Steady-State':
        # Steady-State: γ_eff is needed to invert I_CD → P_CD.
        eta_CD_solution = f_etaCD_effective(
            config, a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T, Zeff,
            f_alpha=f_alpha_solution,
            rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, rho_EC=rho_EC, rho_NBI=rho_NBI)
        I_Ohm_solution  = 0
        P_Ohm_solution  = 0
        I_CD_solution   = f_ICD(Ip_solution, Ib_solution, I_Ohm_solution)
        P_CD_solution   = f_PCD(R0, nbar_solution, I_CD_solution, eta_CD_solution)

    elif Operation_mode == 'Pulsed':
        # Pulsed: powers are fixed inputs; I_CD is the direct sum of per-source
        # non-inductive contributions.  ICRH heats but drives no current.
        if CD_source == 'Multi':
            P_CD_solution = P_LH + P_ECRH + P_NBI + P_ICRH
            I_CD_solution = (f_I_CD(R0, nbar_solution, eta_LH_solution,  P_LH  ) +
                             f_I_CD(R0, nbar_solution, eta_EC_solution,  P_ECRH) +
                             f_I_CD(R0, nbar_solution, eta_NBI_solution, P_NBI ))
        else:
            P_CD_solution = P_aux_input
            eta_CD_solution = f_etaCD_effective(
                config, a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T, Zeff,
                f_alpha=f_alpha_solution,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, rho_EC=rho_EC, rho_NBI=rho_NBI)
            I_CD_solution = f_I_CD(R0, nbar_solution, eta_CD_solution, P_CD_solution)
        I_Ohm_solution = f_I_Ohm(Ip_solution, Ib_solution, I_CD_solution)
        # Ohmic power from the profile-integrated neoclassical resistance
        # (R_eff from f_Reff), replacing the simplified straight-conductor
        # estimate f_P_Ohm.
        P_Ohm_solution = f_P_Ohm_integrated(
            I_Ohm_solution, a, κ, R0, Tbar, nbar_solution, Zeff, q95_solution,
            nu_T, nu_n, eta_model=eta_model,
            rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
            Vprime_data=Vprime_data,
            trapped_fraction_model=trapped_fraction_model)
        # Effective γ computed a posteriori for reporting only:
        # γ_eff = I_CD · R₀ · n̄ / P_CD_non_ICRH  (ICRH denominator excluded since it
        # drives no current — using P_CD_total would artifically dilute the figure of merit)
        _P_CD_driving = P_CD_solution - (P_ICRH if CD_source == 'Multi' else 0.0)
        eta_CD_solution = (I_CD_solution * R0 * nbar_solution / _P_CD_driving
                           if _P_CD_driving > 0 else 0.0)

    else:
        if verbose >= 1:
            print("Choose a valid operation mode")

    # ── Per-source breakdown (post-convergence) ────────────────────────────────
    # Pulsed Multi: the input source powers. Single source or Steady-State:
    # the heating fractions of P_CD (f_CD_breakdown).
    cd_bd = _cd_split(P_CD_solution, nbar_solution,
                      eta_LH_solution, eta_EC_solution, eta_NBI_solution)
    P_LH_solution  = cd_bd['P_LH']
    P_EC_solution  = cd_bd['P_EC']
    P_NBI_solution = cd_bd['P_NBI']
    P_ICR_solution = cd_bd['P_ICR']
    I_LH_solution  = cd_bd['I_LH']
    I_EC_solution  = cd_bd['I_EC']
    I_NBI_solution = cd_bd['I_NBI']

    # Separatrix power: P_sep = P_α + P_CD − P_rad_total (thesis Eq. P_sep).
    # The whole of the radiated power is subtracted: wherever inside the
    # LCFS the photons are emitted, they land on the first wall directly
    # and do not cross the separatrix as conducted or convected heat.
    # The divertor input restores the small ohmic term: P_div = P_sep + P_Ω.
    # The variable name P_sep_solution is kept for backward compatibility
    # with the genetic algorithm, scan module and figures module.
    P_sep_solution      = f_P_sep(P_fus, P_CD_solution, P_rad_total_solution,
                                  f_charged=f_charged)
    Gamma_n_solution    = f_Gamma_n(a, P_fus, R0, κ, S_wall=Surface_solution,
                                    f_neutron=f_neutron)
    heat_refined_solution = f_heat_refined(R0, P_sep_solution)
    heat_par_solution   = f_heat_par(R0, B0_solution, P_sep_solution)
    heat_pol_solution   = f_heat_pol(R0, B0_solution, P_sep_solution, a, q95_solution)
    # The Eich width uses the outer-midplane separatrix B_pol, not the
    # perimeter average B_pol_solution (reported, consistent with beta_P).
    B_pol_omp_solution = f_Bpol_omp(Ip_solution, a, κ)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(
        P_sep_solution, B_pol_omp_solution, R0, a / R0, theta_deg,
        B0=B0_solution)

    # ── Refined divertor exhaust: two-point model (Stangeby 2018) ─────────────
    # D0FUS density profiles vanish at the LCFS by construction (tanh edge), so
    # n_sep is NOT read off the profile. It is anchored to the volume-average
    # density: n_sep = f_n_sep · n̄. Anchoring to n̄ rather than the Greenwald
    # limit n_G is essential: at low Greenwald fraction a fixed n_sep/n_G could
    # exceed n̄, which is unphysical. The ITPA H-mode database normalises exactly
    # this way (n_sep/n̄ ≈ 0.1–0.5, arXiv:2406.15693). nbar is in 1e20 m^-3.
    n_sep_solution = f_n_sep * nbar_solution * 1e20        # [m^-3]

    divertor_solution = f_heat_two_point(
        P_sep_solution, B_pol_omp_solution, R0, a / R0, q95_solution,
        n_sep_solution, theta_deg,
        B0=B0_solution,
        f_cooling=f_cooling_div, f_mom=f_mom_div,
        q_dep_limit=q_dep_limit, flux_expansion=flux_expansion,
        m_f=Atomic_mass * 1.67e-27)   # fuel-ion mass, same u as M_F_DT
    divertor_solution['n_sep']            = n_sep_solution
    divertor_solution['B_pol_omp']        = B_pol_omp_solution
    divertor_solution['f_cooling']        = f_cooling_div     # operating point
    divertor_solution['f_mom']            = f_mom_div         # operating point
    divertor_solution['theta_deg']        = theta_deg
    divertor_solution['q_dep_limit']      = q_dep_limit
    divertor_solution['q_par0_attached']  = q_parallel0_Eich  # Academic peak flux
    divertor_solution['m_f']              = Atomic_mass * 1.67e-27  # fuel-ion mass [kg]
    # First-wall load: radiative power distributed over the plasma surface
    P_1rst_wall_rad    = f_P_wall(P_rad_total_solution, Surface_solution)
    # Exhaust power density: P_div / S (useful divertor figure of merit)
    P_1rst_wall_div   = P_sep_solution / Surface_solution if Surface_solution > 0 else 0.0
    P_wallplug_solution = P_CD_solution / eta_WP   # Wall-plug power consumed by heating/CD [MW]

    # ── Loop voltage ──────────────────────────────────────────────────────────
    # V_loop is now computed inside Magnetic_flux (returned as 5th element)
    # to avoid redundant neoclassical conductance integration.

    # ── q,j profile on the converged global quantities ───────────────────
    # Profile solved by the outer fixed point at the retained solution
    # (_q_at_solution): converged Ip and I_CD, CD deposition of the
    # per-source currents above.  The bootstrap of the chain was evaluated on
    # the previous profile, within _Q_LOOP_RTOL of this one.
    if _q_failures and verbose >= 1:
        print(f"[warn] q-profile solve at the starting point failed "
              f"({_q_failures[0]}). The first inner solve used the parabolic "
              "q(ρ) fallback.")
    li_solution      = _q_solution['li']

    # L-H power threshold. All scalings were fitted with the line-averaged
    # density and the area of the last closed flux surface, here the area of
    # the plasma geometry model (Miller contour in refined mode). The TC-26
    # scalings take the nominal divertor factor D = 1; the report gives the
    # upper bound D = TC26_D_VT (Delabie et al., NF 66 (2026) 036016).
    P_Thresh = f_P_thresh(nbar_line_solution, B0_solution, Surface_solution,
                          Atomic_mass, Ip=Ip_solution, R0=R0,
                          Option_PLH=L_H_Scaling_choice)

    # ==============================================================================
    #    TOROIDAL FIELD (TF) COIL RADIAL BUILD
    #    Determines the inboard thickness 'c' from mechanical and magnetic constraints.
    # ==============================================================================
    # config.SF_TF applies the safety factor on the steel mechanical allowable
    # at the solver entry (σ_TF ← σ_TF / SF_TF). Default 1.0 reproduces the
    # legacy behaviour; realistic engineering value ≈ 1.5 covers the unfilled
    # WP envelope, stress concentrations, weld efficiency and manufacturing
    # tolerances not represented in the idealised CICC area model.
    # The quench protection sizes J_TF from the stored energy, and the
    # Princeton-D bore holding that energy grows with c: fixed point on c,
    # 2 to 4 sizings. delta_port = Delta_TF puts the outboard leg where the
    # ripple and port-access sizing placed it, in the vertical force.
    for _ in range(10):
        if Radial_build_model == "academic":
            (c, c_WP_TF, c_Nose_TF,
             σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_academic(
                a, b, R0, σ_TF, J_max_TF_conducteur, Bmax_TF, Choice_Buck_Wedg,
                coef_inboard_tension, F_CClamp, delta_port=Delta_TF,
                SF_TF=config.SF_TF)

        elif Radial_build_model in ("refined", "CIRCE"):
            (c, c_WP_TF, c_Nose_TF,
             σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_refined(
                a, b, R0, σ_TF, J_max_TF_conducteur, Bmax_TF, Choice_Buck_Wedg, omega_TF, n_shape_TF,
                c_BP, coef_inboard_tension, F_CClamp, TF_grading,
                delta_port=Delta_TF, SF_TF=config.SF_TF)

        else:
            raise ValueError(
                f"Unknown radial build model: '{Radial_build_model}'. "
                "Valid options: 'academic', 'refined', 'CIRCE'."
            )

        if not np.isfinite(c):
            break
        E_mag_c = calculate_E_mag_TF(Bmax_TF, a, b, R0, c, Delta_TF)
        if abs(E_mag_c - E_mag_TF) <= 1e-3 * E_mag_TF:
            break
        E_mag_TF            = E_mag_c
        result_J_TF         = _tf_cable(E_mag_TF)
        J_max_TF_conducteur = result_J_TF['J_wost']

    # ── TF cable-level fractions (wost = without steel) ──────────────────────
    f_sc_TF      = result_J_TF['f_sc']
    f_cu_TF      = result_J_TF['f_cu']
    f_He_pipe_TF = result_J_TF['f_He_pipe']
    f_void_TF    = result_J_TF['f_void']
    f_He_TF      = result_J_TF['f_He']
    f_In_TF      = result_J_TF['f_In_cable']

    # ── TF radial build failed: keep the converged plasma, blank only the
    #    magnet / engineering quantities ────────────────────────────────────────
    # When f_TF_* cannot size a feasible inboard leg, c comes back non-finite:
    # the winding pack reaching the requested peak field Bmax_TF within the steel
    # stress allowable does not fit in the available inboard space R0 - a - b.
    # The plasma equilibrium itself is still physical and was already solved
    # above, so returning an all-NaN tuple here would discard a perfectly valid
    # operating point (its density, beta, q95, Ip, Q, ...). That discarding is
    # exactly what makes the plasma operating space appear blank in 2-D scans in
    # regions where no plasma limit is reached but the TF coil is not buildable.
    # We therefore return the plasma quantities and set to NaN only the
    # quantities that genuinely depend on the (failed) coil build: the TF/CS
    # geometry and stresses, the magnetic-flux scenario, and all coil/component
    # volumes, masses, lifetimes and costs. The earlier convergence guard on
    # (f_alpha, Q) still returns an all-NaN tuple, because without a converged
    # burn there is no plasma to report at all.
    _nan = np.nan
    if not np.isfinite(c):
        return (B0_solution,  _nan,  B_pol_solution,          # B0, B_CS(nan), B_pol
                tauE_solution,  W_th_solution,
                Q_solution,  Volume_solution,  Surface_solution,
                Ip_solution,  Ib_solution,  I_CD_solution,  I_Ohm_solution,
                nbar_solution,  nbar_line_solution,  nG_solution,  pbar_solution,
                betaN_solution,  betaT_solution,  betaP_solution,
                qstar_solution,  q95_solution,
                P_CD_solution,  P_sep_solution,  P_Thresh,  eta_CD_solution,
                _nan,  P_wallplug_solution,    # P_elec(nan): plant balance needs the coils
                _nan,  P_Brem_solution,  P_syn_solution,  P_line_solution,  # cost(nan)
                P_line_core_solution,
                heat_refined_solution,  heat_par_solution,  heat_pol_solution,
                lambda_q_Eich_m,  q_target_Eich,
                P_1rst_wall_rad,  P_1rst_wall_div,
                Gamma_n_solution,
                f_alpha_solution,  tau_alpha,
                J_max_TF_conducteur,  _nan,                    # J_TF, J_CS(nan)
                _nan, _nan, _nan, _nan, _nan, _nan, _nan,      # c, c_WP_TF, c_Nose_TF, sigma_TF x3, Steel_TF
                _nan, _nan, _nan, _nan, _nan, _nan, _nan,      # d, sigma_CS x3, Steel_CS, B_CS, J_CS
                R0 - a,  R0 - a - b,  _nan,  _nan,             # r_minor, r_sep kept; r_c, r_d depend on c
                κ,  κ_95,  δ,  δ_95,
                _nan, _nan, _nan, _nan, _nan, _nan,  li_solution,  # Psi PI/RampUp/plateau/PF/CS, Vloop; li kept
                eta_LH_solution,  eta_EC_solution,  eta_NBI_solution,
                P_LH_solution,  P_EC_solution,  P_NBI_solution,  P_ICR_solution,
                I_LH_solution,  I_EC_solution,  I_NBI_solution,
                f_sc_TF,  f_cu_TF,  f_He_pipe_TF,  f_void_TF,  f_He_TF,  f_In_TF,
                _nan, _nan, _nan, _nan, _nan, _nan,            # f_*_CS (CS conductor build)
                beta_fast_alpha,  betaN_total,  tau_sd_alpha,  W_fast_alpha,
                _nan, _nan,                                    # V_TF_one, V_CS_geom
                _nan, _nan, _nan, _nan, _nan,                  # V_steel/sc/cu/He/In_TF
                _nan, _nan, _nan, _nan, _nan,                  # V_steel/sc/cu/He/In_CS
                _nan, _nan,                                    # L_cable_TF, L_cable_CS
                _nan, _nan,                                    # n_sc_TF, n_sc_CS
                _nan, _nan,                                    # L_sc_strand_TF, L_sc_strand_CS
                _nan, _nan, _nan, _nan, _nan,                  # M_steel/sc/cu/In/total_TF
                _nan, _nan, _nan, _nan, _nan,                  # M_steel/sc/cu/In/total_CS
                _nan,                                          # V_blanket
                _nan, _nan,                                    # t_life_bl_fpy, t_life_div_fpy
                _nan, _nan,                                    # t_life_bl_yr, t_life_div_yr
                _nan, _nan,                                    # T_op_limit, dt_rep_eff
                _nan, _nan,                                    # Av, CF
                _nan,                                          # V_rb_SOL
                _nan, _nan,                                    # V_rb_FW, V_rb_BB
                _nan, _nan, _nan,                              # V_rb_shield, V_rb_VV, V_rb_gap_TF
                _nan,                                          # V_rb_divertor
                _nan, _nan, _nan,                              # M_rb_FW, M_rb_BB, M_rb_shield
                _nan, _nan, _nan,                              # M_rb_VV, M_rb_divertor, M_rb_total
                divertor_solution,                             # two-point divertor model (kept)

                # Plant electrical-balance outputs unavailable because the
                # magnet build failed before these quantities could be solved.
                _nan, _nan, _nan, _nan, _nan,
                _nan, _nan, _nan, _nan, _nan,
                _nan, _nan, _nan, _nan, _nan,  # plant outputs 160–164
                _nan, _nan, _nan, _nan, _nan,  # plant outputs 165–169
                _nan, _nan, _nan, _nan, _nan, _nan)  # phase outputs 170–175

    # ==============================================================================
    #    MAGNETIC FLUX REQUIREMENTS (Inductive Scenario)
    #    Volt-seconds budget for plasma initiation, ramp-up, and flat-top.
    #    Also returns the steady-state loop voltage (avoids redundant computation).
    # ==============================================================================
    _gap_eff = Gap if Choice_Buck_Wedg == 'Wedging' else 0.0
    (ΨPI, ΨRampUp, Ψplateau, ΨPF, Vloop_solution) = Magnetic_flux(
    Ip_solution, I_Ohm_solution, R0, a, κ, li_solution,
    Ce, Temps_Plateau_input,
    config.E_BD, betaP_solution,
    nbar_solution, Tbar, Zeff, q95_solution, nu_T, nu_n, eta_model,
    rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
    Vprime_data=Vprime_data, trapped_fraction_model=trapped_fraction_model,
    )
    # Non-inductive current drive during ramp-up reduces the CS volt-second
    # requirement proportionally: both the inductive (Ψ_ind ∝ Ip) and resistive
    # (Ψ_res = Ce μ0 R0 Ip) terms scale with the inductively-ramped current fraction.
    ΨRampUp *= (1.0 - getattr(config, 'f_heat_ramp', 0.0))
    # ── TF Princeton-D cross-section geometry ─────────────────────────────────
    # Solved here rather than after the CS block because the CS height is
    # referenced to the TF winding-pack inside height (see f_H_CS). The ODE is
    # solved once per run and every consumer reuses this result.
    (R_bore_TF, _R_TF_out, _H_TF_D, _A_cross_TF, L_turn_TF,
     _R_out, _Z_out, _R_in, _Z_in) = f_TF_cross_section(a, b, R0, c, Delta_TF)
    Z_TF_bore_half = float(np.max(_Z_in)) if np.all(np.isfinite(_Z_in)) else np.nan

    # ── CS height: single source of truth ────────────────────────────────────
    # An explicit config.H_CS wins, otherwise f_H_CS gives the TF bore inside
    # height. The resolved value is written back into the config so that every
    # consumer (the CS solvers via _CS_geometry_init, the volume, the cable
    # inventory, the report and the figures) sees the same height.
    if config.H_CS is None or not (config.H_CS > 0):
        H_CS = f_H_CS(Z_TF_bore_half)
        config = dc_replace(config, H_CS=H_CS)
    else:
        H_CS = float(config.H_CS)

    # ==============================================================================
    #    CENTRAL SOLENOID (CS) DESIGN
    #    Determines the CS radial thickness 'd' to satisfy the Volt-second budget.
    # ==============================================================================
    cs_args = (ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax_TF, Bmax_CS_adm, σ_CS,
           Supra_choice_CS, J_wost_Manual_CS, T_helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h_CS,
           config if f_void_CS == f_void else dc_replace(config, f_void=f_void_CS))

    if Radial_build_model == "academic":
        (d, σ_z_CS, σ_theta_CS, σ_r_CS,
         Steel_fraction_CS, B_CS, J_CS) = f_CS_ACAD(*cs_args)
    elif Radial_build_model == "refined":
        (d, σ_z_CS, σ_theta_CS, σ_r_CS,
         Steel_fraction_CS, B_CS, J_CS) = f_CS_refined(*cs_args)
    elif Radial_build_model == "CIRCE":
        (d, σ_z_CS, σ_theta_CS, σ_r_CS,
         Steel_fraction_CS, B_CS, J_CS) = f_CS_CIRCE(*cs_args)

    # Total inductive flux swing provided by the CS
    # ΨCS = Breakdown + Ramp-up + Flat-top − External PF contribution
    # (no geometric correction: the null-field breakdown assumption
    # implies that Ψ at R_0 equals the infinite-solenoid Ψ at R_e.)
    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF

    # ── CS coil current density (cable-level fractions) ──────────────────────
    # The CS solver computes J_wost and Steel_fraction internally, but does not
    # expose the sub-fractions (SC, Cu, He pipe, void, insulation).
    # We call calculate_cable_current_density with B_CS to recover them, with
    # the same stored energy as inside the solver (same radii, same H_CS).

    if np.isfinite(B_CS) and B_CS > 0:
        E_mag_CS_post = calculate_E_mag_CS(
            B_CS,
            R0 - a - b - c - _gap_eff - d,   # CS inner bore [m]
            R0 - a - b - c - _gap_eff,         # CS outer face [m]
            H_CS)
        result_J_CS = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS, T_op=T_helium,
            E_mag=E_mag_CS_post,
            I_cond=I_cond_CS, V_max=V_max, N_sub=N_sub_CS, tau_h=tau_h_CS,
            f_He_pipe=f_He_pipe, f_void=f_void_CS, f_In_cable=f_In_cable, f_In_WP=f_In_WP,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He,
            Marge_T_Nb3Sn=Marge_T_Nb3Sn, Marge_T_NbTi=Marge_T_NbTi,
            Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=J_wost_Manual_CS if Supra_choice_CS == 'Manual' else None)
        f_sc_CS      = result_J_CS['f_sc']
        f_cu_CS      = result_J_CS['f_cu']
        f_He_pipe_CS = result_J_CS['f_He_pipe']
        f_void_CS    = result_J_CS['f_void']
        f_He_CS      = result_J_CS['f_He']
        f_In_CS      = result_J_CS['f_In_cable']
    else:
        f_sc_CS = f_cu_CS = f_He_pipe_CS = f_void_CS = f_He_CS = f_In_CS = np.nan
        E_mag_CS_post = np.nan

    # ── Density limit, model-selectable ──────────────────────────────────────
    # Evaluated here (after the power balance and exhaust blocks) because the
    # Giacomin and Zanca limits are power-dependent. nG_solution, consumed by
    # the results tuple and the downstream feasibility constraints
    # (nbar_line < nG), is overridden with the selected model cap so that the
    # GA / scan / figures machinery inherits the choice transparently.
    # The native Greenwald density Ip/(pi a^2) remains available as nG_raw.
    _f_n_sep_line = f_n_sep * (nbar_solution / nbar_line_solution)  # n_sep/n̄_line
    # Giacomin near-separatrix anchoring: n(rho=0.9)/n̄_line from the profile
    _f_n_edge_line = (f_n_edge_ratio(nu_n, rho_ped, n_ped_frac)
                      * (nbar_solution / nbar_line_solution))
    _P_tot_heat   = P_Alpha + P_Aux_solution + P_Ohm_solution       # [MW]
    n_DL_line, n_DL_native, n_DL_convention = f_density_limit(
        density_limit_model, Ip_solution, a,
        P_sol=P_sep_solution, P_tot=_P_tot_heat,
        R0=R0, kappa=κ, B0=B0_solution, q_edge=q95_solution,
        Z_eff=Zeff, f_n_sep_line=_f_n_sep_line,
        f_n_edge_line=_f_n_edge_line,
        A_ion=config.Atomic_mass,
        alpha_GR=config.alpha_giacomin, f0=config.f0_zanca)
    if density_limit_model != 'greenwald':
        nG_solution = n_DL_line * Greenwald_limit
    # Component volumes — only V_FI used; Princeton-D H_TF for the FI cylinder
    (_, _, _, V_FI) = f_volume(a, b, c, d, R0, κ, Delta_TF, _H_TF_D)

    # ── Single-coil TF volume (inline: reuse A_cross and R_bore already computed) ──
    # f_V_TF(a,b,R0,c,N_TF,Delta_TF) = A_cross * 2π * R_bore / N_TF but it would
    # call f_TF_cross_section a second time (ODE solve). Use pre-computed values.
    V_TF_one  = _A_cross_TF * 2.0 * np.pi * R_bore_TF / float(N_TF)
    V_CS_geom = f_V_CS(a, b, c, d, R0, Gap, Choice_Buck_Wedg, H_CS)

    # ── Radial build layer volumes ────────────────────────────────────────────
    # All boundaries are real 3D contours; volumes via exact _moment_area integrals.
    # Concept-specific layer stack (BLANKET_CONCEPTS[...]['radial_layers']) split
    # around the single 'breeder' layer: plasma-side layers use cumulative Miller
    # ellipses, TF-side layers use cumulative Princeton-D offsets, breeder fills
    # the remaining shell.  No Pappus approximation; no post-hoc scaling.
    V_blanket = f_V_blanket(a, b, R0, κ, δ, Delta_TF,
                            R_outer=_R_in, Z_outer=_Z_in)
    _rb = f_radial_build_layers(
        a=a, b=b, κ=κ, δ=δ, R0=R0,
        Delta_TF=Delta_TF,
        f_kappa_SOL=config.f_kappa_SOL,
        f_div_area_fraction=config.f_div_area_fraction,
        R_tf_in=_R_in,
        Z_tf_in=_Z_in,
        Blanket_choice=config.Blanket_choice,
        delta_BB_IB=config.delta_BB_IB, delta_BB_OB=config.delta_BB_OB,
        delta_shield=config.delta_shield,
    )
    # A concept multiplication falls where the breeder is thin: it is
    # averaged over the breeder surface with the local breeder thickness.
    # A deck value is used as given.
    if config.M_blanket is None and _rb['feasible']:
        M_blanket = f_blanket_multiplication(M_blanket, _rb['delta_BB_local'],
                                             config.Blanket_choice,
                                             _rb['w_BB_local'])
    _BB_ROLES     = ('breeder', 'structure', 'multiplier')
    _SHIELD_ROLES = ('shield_HT', 'shield_LT')

    V_rb_SOL      = sum(L['V']     for L in _rb['layers'] if L['role'] == 'SOL')
    V_rb_FW       = sum(L['V_eff'] for L in _rb['layers'] if L['role'] == 'FW')
    V_rb_BB       = sum(L['V_eff'] for L in _rb['layers'] if L['role'] in _BB_ROLES)
    V_rb_shield   = sum(L['V']     for L in _rb['layers'] if L['role'] in _SHIELD_ROLES)
    V_rb_VV       = sum(L['V']     for L in _rb['layers'] if L['role'] == 'VV')
    V_rb_gap_TF   = sum(L['V']     for L in _rb['layers'] if L['role'] == 'gap')
    V_rb_divertor = _rb['V_divertor']
    if not _rb['feasible']:
        # Unbuilt components do not have known zero volumes: mark them missing.
        V_rb_SOL = V_rb_FW = V_rb_BB = V_rb_shield = V_rb_VV = V_rb_gap_TF = np.nan


    # ── Radial build layer masses ─────────────────────────────────────────────
    M_rb_FW       = sum(L['M'] for L in _rb['layers'] if L['role'] == 'FW')
    M_rb_BB       = sum(L['M'] for L in _rb['layers'] if L['role'] in _BB_ROLES)
    M_rb_shield   = sum(L['M'] for L in _rb['layers'] if L['role'] in _SHIELD_ROLES)
    M_rb_VV       = sum(L['M'] for L in _rb['layers'] if L['role'] == 'VV')
    M_rb_divertor = V_rb_divertor * config.rho_divertor
    M_rb_total    = M_rb_FW + M_rb_BB + M_rb_shield + M_rb_VV + M_rb_divertor
    if not _rb['feasible']:
        M_rb_FW = M_rb_BB = M_rb_shield = M_rb_VV = M_rb_divertor = M_rb_total = np.nan


    cost_solution = (V_rb_BB + V_TF_one * N_TF + V_CS_geom) / P_fus   # legacy cost proxy [m^3/MW]

    # ── Material volumes (TF: total = V_TF_one × N_TF; CS: full solenoid) ─────
    # Steel share of the whole TF inboard leg. The refined models return the
    # winding-pack value (conductor jacket), while the nose and the backplate
    # are all steel. The academic model already returns the share of the
    # whole leg (c_Nose / c).
    if Radial_build_model == "academic":
        steel_fraction_leg_TF = Steel_fraction_TF
    else:
        steel_fraction_leg_TF = 1.0 - (c_WP_TF / c) * (1.0 - Steel_fraction_TF)
    (V_steel_TF, V_sc_TF, V_cu_TF, V_He_TF, V_In_TF,
     V_steel_CS, V_sc_CS, V_cu_CS, V_He_CS, V_In_CS) = f_coil_material_volumes(
        V_TF_one, int(N_TF),
        steel_fraction_leg_TF,
        f_sc_TF, f_cu_TF, f_He_TF, f_In_TF,
        V_CS_geom,
        Steel_fraction_CS,
        f_sc_CS, f_cu_CS, f_He_CS, f_In_CS,
        f_In_WP)

    # ── Cable conductor lengths ───────────────────────────────────────────────
    R_CS_ext_val = R0 - a - b - c - _gap_eff
    L_turn_CS    = np.pi * (R_CS_ext_val + R_CS_ext_val - d)   # 2π × R_mean
    _R_TF_in     = R0 - a - b   # plasma-facing inner face, consistent with D0FUS NI convention
    (L_cable_TF, L_cable_CS,
     _Nturns_TF, _Nturns_CS) = f_cable_length(
        int(N_TF), L_turn_TF, Bmax_TF, _R_TF_in,
        I_cond,
        N_sub_CS, L_turn_CS, B_CS, H_CS,   # CS Ampere law uses the resolved H_CS
        I_cond_CS)

    # ── Material masses ───────────────────────────────────────────────────────
    (M_steel_TF, M_sc_TF, M_cu_TF, M_In_TF, M_total_TF,
     M_steel_CS, M_sc_CS, M_cu_CS, M_In_CS, M_total_CS) = f_coil_masses(
        V_steel_TF, V_sc_TF, V_cu_TF, V_In_TF,
        V_steel_CS, V_sc_CS, V_cu_CS, V_In_CS,
        Chosen_Steel_TF, Supra_choice, f_TF_steel_mass, Chosen_Steel_CS,
        Supra_choice_CS)

    # ── SC strand counts and total SC strand lengths ───────────────────────────
    (n_sc_TF, n_sc_CS,
     L_sc_strand_TF, L_sc_strand_CS) = f_cable_strands(
        V_sc_TF, L_cable_TF,
        V_sc_CS, L_cable_CS,
        Supra_choice, Supra_choice_CS)

    # ── Component lifetimes and plant availability ────────────────────────────
    A_div          = config.f_div_area_fraction * Surface_solution
    A_blanket      = Surface_solution * (1.0 - config.f_div_area_fraction)
    t_life_bl_fpy  = f_blanket_lifetime_fpy(P_fus, A_blanket,
                                             config.dpa_lim, config.C_dpa,
                                             f_neutron=f_neutron)
    t_life_div_fpy = f_divertor_lifetime_fpy(P_sep_solution, A_div,
                                              config.epsilon_div, config.f_peak)
    t_life_bl_yr   = f_lifetime_to_years(t_life_bl_fpy,
                                          config.Util_factor, config.Dwell_factor)
    t_life_div_yr  = f_lifetime_to_years(t_life_div_fpy,
                                          config.Util_factor, config.Dwell_factor)
    (T_op_limit, dt_rep_eff,
     Av_solution, CF_solution) = f_availability_schedule(
        t_life_bl_fpy, t_life_div_fpy,
        config.dt_rep_bl, config.dt_rep_div,
        config.Util_factor, config.Dwell_factor)

    # ── Plant electrical balance and pulsed thermal storage ──────────────────
    # Simplified recirculating-power model (M. Fletcher). Cryogenics scaled on
    # EU-DEMO reference values (config.*_DEMO_ref) with the TF + CS cold mass,
    # coolant pumping on P_th, house load from the PROCESS model
    # (house_load_breakdown).
    # Net ELECTRICAL output is smoothed across pulse and dwell. The nominal
    # reactor thermal power also determines coolant pumping, independent of x.
    # CS recharge retains its existing cycle-averaged definition unchanged;
    # its contribution to P_recirc_var is assigned to the pulse balance by
    # the simplified model (not a time-resolved recharge profile).
    P_th_solution = (P_fus * (f_neutron * M_blanket + 1.0 - f_neutron)
                     + P_CD_solution)   # neutron-only blanket multiplication
    # Derive dwell duration here solely for the existing P_coil_CS expression.
    t_dwell_solution = (config.Temps_Plateau_input
                        * (1.0 / config.Dwell_factor - 1.0)
                        if config.Dwell_factor < 1.0 else 0.0)
    # CS recharge: E_mag_CS supplied during the dwell, averaged over the
    # cycle (zero without dwell)
    P_coil_CS_solution = (f_coil_power_supply(
        E_mag_CS_post, config.Temps_Plateau_input + t_dwell_solution)
        if config.Dwell_factor < 1.0 else 0.0)
    I_feeders_solution = f_feeder_current(
        I_cond, I_cond_CS, N_TF=N_TF, N_CS=config.N_sub_CS)
    M_cold_solution = M_total_TF + M_total_CS   # magnet cold mass [kg]
    P_cryo_cool_solution = f_cryo_cooling_power(
        M_cold_solution, config.M_cold_DEMO_ref, config.P_cryo_cool_DEMO_ref)
    P_cryo_electric_solution = f_cryo_electric_power(
        M_cold_solution, config.M_cold_DEMO_ref, config.P_cryo_electric_DEMO_ref)
    (V_cryo_LHe_solution, V_cryo_LN_solution) = f_cryo_LHe_LN_volume(P_cryo_electric_solution)
    P_house_load_solution = house_load_breakdown(
        config, _R_TF_out, R_bore_TF, c, _H_TF_D, Z_TF_bore_half, N_TF,
        calculate_E_mag_TF(Bmax_TF, a, b, R0, c, Delta_TF),
        P_cryo_cool_solution)['P_house']
    # Coolant pumping on the heat extracted during the burn
    P_BoP_solution = f_bop_power(P_th_solution, f_pump)
    # Heating/CD wall-plug power on P_CD, as in P_wallplug_solution, during
    # the burn
    (P_recirc_var_solution, P_recirc_fix_solution,
     P_recirculated_solution) = f_recirculated_power(
        P_CD_solution, eta_WP, P_coil_CS_solution,
        P_cryo_electric_solution, P_BoP_solution, P_house_load_solution)
    electrical_balance = f_pulsed_electrical_quantities(
        P_th_solution, config.Temps_Plateau_input, config.Dwell_factor,
        eta_T, P_recirc_var_solution, P_recirc_fix_solution, config.eta_store)
    E_store_th_solution = electrical_balance['E_store_th']
    # Legacy fields kept for downstream callers. P_th_smoothed now means
    # reactor-to-turbine heat *during pulse*; it is not constant over cycle.
    P_th_smoothed_solution = electrical_balance['P_th_reactor_to_turbine_pulse']
    P_e_gross_solution = electrical_balance['P_e_gross_pulse']
    P_elec_solution = electrical_balance['P_e_net_cycle_average']
    if not electrical_balance['electrical_balance_valid']:
        import warnings
        warnings.warn(
            'Electrical balance unavailable: ' + electrical_balance['electrical_balance_reason']
            + f' (P_th={P_th_solution!r} MWth, eta_turbine={eta_T!r}, '
              f'P_recirc_var={P_recirc_var_solution!r} MWe, '
              f'P_recirc_fix={P_recirc_fix_solution!r} MWe). '
              'Other machine calculations continue; power-dependent economics are undefined.',
            RuntimeWarning, stacklevel=2)
    elif not electrical_balance['smoothing_feasible']:
        import warnings
        warnings.warn(
            'Electrical smoothing infeasible: ' + electrical_balance['electrical_balance_reason']
            + '. Using x=1 (no thermal storage); annual production and economics '
            'use cycle-average net electricity.',
            RuntimeWarning, stacklevel=2)

    CF_solution = Av_solution * config.Util_factor
    E_recirculated_solution = f_recirculated_energy(
        P_recirc_var_solution, P_recirc_fix_solution, CF_solution)
    # Burn time fraction = Av × Util × Dwell. D-D consumes no external tritium.
    (C_tritium_fpy_solution, C_tritium_y_solution) = (f_tritium_consumption(
        P_fus, CF_solution * config.Dwell_factor) if Fuel == 'DT' else (0.0, 0.0))

    # Lift factors use the same bath/operating temperature as conductor sizing
    _lf_params = (config.LF_numerator, config.LF_a0, config.LF_a1,
                  config.LF_b0, config.LF_b1, config.LF_b2)
    LF_TF = f_lift_factor(T_helium, Bmax_TF, *_lf_params)
    LF_CS = f_lift_factor(T_helium, B_CS, *_lf_params)

    return (B0_solution,  B_CS,  B_pol_solution,
            tauE_solution,  W_th_solution,
            Q_solution,  Volume_solution,  Surface_solution,
            Ip_solution,  Ib_solution,  I_CD_solution,  I_Ohm_solution,
            nbar_solution,  nbar_line_solution,  nG_solution,  pbar_solution,
            betaN_solution,  betaT_solution,  betaP_solution,
            qstar_solution,  q95_solution,
            P_CD_solution,  P_sep_solution,  P_Thresh,  eta_CD_solution,
            P_elec_solution,  P_wallplug_solution,
            cost_solution,  P_Brem_solution,  P_syn_solution,  P_line_solution,
            P_line_core_solution,
            heat_refined_solution,  heat_par_solution,  heat_pol_solution,
            lambda_q_Eich_m,  q_target_Eich,
            P_1rst_wall_rad,  P_1rst_wall_div,
            Gamma_n_solution,
            f_alpha_solution,  tau_alpha,
            J_max_TF_conducteur,  J_CS,
            c,  c_WP_TF,  c_Nose_TF,  σ_z_TF,  σ_theta_TF,  σ_r_TF,  Steel_fraction_TF,
            d,  σ_z_CS,  σ_theta_CS,  σ_r_CS,  Steel_fraction_CS,  B_CS,  J_CS,
            # r_c = CS outer face; r_d = CS inner bore.
            # In Wedging the TF-CS gap shifts both inward by Gap; in Bucking/Plug gap = 0.
            R0 - a,  R0 - a - b,
            R0 - a - b - c - _gap_eff,       # CS outer face [m]
            R0 - a - b - c - _gap_eff - d,   # CS inner bore [m]
            κ,  κ_95,  δ,  δ_95,
            ΨPI,  ΨRampUp,  Ψplateau,  ΨPF,  ΨCS,  Vloop_solution,  li_solution,
            eta_LH_solution,  eta_EC_solution,  eta_NBI_solution,
            P_LH_solution,  P_EC_solution,  P_NBI_solution,  P_ICR_solution,
            I_LH_solution,  I_EC_solution,  I_NBI_solution,
            f_sc_TF,  f_cu_TF,  f_He_pipe_TF,  f_void_TF,  f_He_TF,  f_In_TF,
            f_sc_CS,  f_cu_CS,  f_He_pipe_CS,  f_void_CS,  f_He_CS,  f_In_CS,
            beta_fast_alpha,  betaN_total,  tau_sd_alpha,  W_fast_alpha,
            # ── Coil volumes and cable inventory (indices 99–116) ──────────────
            V_TF_one,   V_CS_geom,                                 # 99, 100
            V_steel_TF, V_sc_TF, V_cu_TF, V_He_TF, V_In_TF,      # 101–105
            V_steel_CS, V_sc_CS, V_cu_CS, V_He_CS, V_In_CS,       # 106–110
            L_cable_TF,     L_cable_CS,                            # 111, 112
            n_sc_TF,        n_sc_CS,                               # 113, 114
            L_sc_strand_TF, L_sc_strand_CS,                        # 115, 116
            # ── Coil masses (indices 117–126) ──────────────────────────────────
            M_steel_TF, M_sc_TF, M_cu_TF, M_In_TF, M_total_TF,   # 117–121
            M_steel_CS, M_sc_CS, M_cu_CS, M_In_CS, M_total_CS,    # 122–126
            V_blanket,                                             # 127
            # ── Component lifetimes and availability (indices 128–135) ─────────
            t_life_bl_fpy,  t_life_div_fpy,                        # 128, 129
            t_life_bl_yr,   t_life_div_yr,                         # 130, 131
            T_op_limit,     dt_rep_eff,                            # 132, 133
            Av_solution,    CF_solution,                          # 134, 135
            # ── Radial build component volumes (indices 136–142) ─────────────
            V_rb_SOL,                                              # 136
            V_rb_FW,        V_rb_BB,                              # 137, 138
            V_rb_shield,    V_rb_VV,    V_rb_gap_TF,              # 139, 140, 141
            V_rb_divertor,                                         # 142
            # ── Radial build component masses (indices 143–148) ──────────────
            M_rb_FW,      M_rb_BB,       M_rb_shield,             # 143, 144, 145
            M_rb_VV,      M_rb_divertor, M_rb_total,
            divertor_solution,                         # 149

            # ── Plant electrical balance outputs (150–167) + phase details (168–175)
            P_th_solution,
            P_th_smoothed_solution,
            E_store_th_solution,
            P_e_gross_solution,
            P_coil_CS_solution,
            I_feeders_solution,
            P_cryo_cool_solution,
            P_cryo_electric_solution,
            V_cryo_LHe_solution,
            V_cryo_LN_solution,
            P_house_load_solution,
            P_BoP_solution,
            P_recirc_var_solution,
            P_recirc_fix_solution,
            P_recirculated_solution,
            E_recirculated_solution,
            C_tritium_fpy_solution,
            C_tritium_y_solution,
            # Additional explicitly phase-resolved smoothing metrics.
            electrical_balance['x_storage_split'],
            electrical_balance['P_th_reactor_to_storage_pulse'],
            electrical_balance['P_th_reactor_to_turbine_pulse'],
            electrical_balance['P_th_storage_to_turbine_dwell'],
            electrical_balance['P_e_gross_pulse'],
            electrical_balance['P_e_gross_dwell'],
            electrical_balance['P_e_net_pulse'],
            electrical_balance['P_e_net_dwell'],
            LF_TF, LF_CS,
            electrical_balance['smoothing_feasible']) 


#%% Output writer
# ---------------------------------------------------------------------------

def _build_run_dict(config: GlobalConfig, results: tuple) -> dict:
    """
    Assemble the run-dict consumed by D0FUS_figures.plot_run().

    The dict maps onto the interface documented in D0FUS_figures.py
    (section 'Run Dict Interface').  Geometry, kinetics, power-balance,
    and radial-build keys are filled from the config dataclass and the
    corresponding entries in the results tuple.

    Parameters
    ----------
    config  : GlobalConfig  Input configuration.
    results : tuple         Return value of run().

    Returns
    -------
    dict  Ready to pass to plot_run().
    """
    _plant = {}
    _base_plant_keys = (
        'P_th', 'P_th_smoothed', 'E_store_th', 'P_e_gross',
        'P_coil_CS', 'I_feeders', 'P_cryo_cool', 'P_cryo_electric',
        'V_cryo_LHe', 'V_cryo_LN', 'P_house_load', 'P_BoP',
        'P_recirc_var', 'P_recirc_fix', 'P_recirculated',
        'E_recirculated', 'C_tritium_fpy', 'C_tritium_y')
    _phase_keys = ('x_storage_split', 'P_th_reactor_to_storage_pulse',
                   'P_th_reactor_to_turbine_pulse',
                   'P_th_storage_to_turbine_dwell', 'P_e_gross_pulse',
                   'P_e_gross_dwell', 'P_e_net_pulse', 'P_e_net_dwell')
    if len(results) >= 150 + len(_base_plant_keys):
        _plant.update(zip(_base_plant_keys, results[150:168]))
    if len(results) >= 176:
        _plant.update(zip(_phase_keys, results[168:176]))
        _plant['P_th_reactor_pulse'] = _plant['P_th']
        _plant['P_th_turbine_dwell'] = _plant['P_th_storage_to_turbine_dwell']
        _plant['P_e_net_smoothed'] = results[25]
    if len(results) >= 179:
        _plant['smoothing_feasible'] = bool(results[178])
        _plant['smoothing_fallback'] = not _plant['smoothing_feasible']
    if len(results) >= 176:
        _plant['P_e_net_cycle_average'] = results[25]
        if len(results) >= 179 and not results[178]:
            _plant['P_e_net_smoothed'] = float('nan')
    _lift_factors = {'LF_TF': results[176], 'LF_CS': results[177]} if len(results) >= 178 else {}

    # ── Unpack the results tuple (same ordering as run() return statement) ────
    (B0, _B_CS_r, _B_pol_r,
     _tauE, _W_th,
     Q, _Volume, _Surface,
     Ip, _Ib, _I_CD, _I_Ohm,
     nbar, _nbar_line, _nG, _pbar,
     _betaN, _betaT, _betaP,
     _qstar, q95,
     _P_CD, _P_sep, _P_Thresh, _eta_CD, _P_elec, _P_wallplug,
     _cost, _P_Brem, _P_syn, _P_line,
     _P_line_core,
     _heat, _heat_par, _heat_pol, _lambda_q, _q_target,
     _P_wall_rad, _P_wall_div,
     _Gamma_n,
     _f_alpha, _tau_alpha,
     _J_TF, _J_CS_1,
     c_TF, c_WP_TF, c_Nose_TF, _sz_TF, _st_TF, _sr_TF, _sf_TF,
     c_CS, _sz_CS, _st_CS, _sr_CS, _sf_CS, _B_CS2, _J_CS2,
     r_minor, r_sep, r_c, r_d,
     kappa_edge, kappa_95, delta_edge, delta_95,
     *_rest) = results

    # Refined divertor exhaust (two-point model) is appended as the very last
    # element of the results tuple, after Mat's coil/volume/mass block (which
    # occupies _rest[33:83]). Empty dict on failure or for older callers.
    _divertor = _rest[83] if len(_rest) >= 84 else {}

    # ── Cable-level fractions (explicit positions in the new tuple layout) ─────
    # Expected *_rest layout (55 values):
    #   [0:7]   ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS, Vloop, li
    #   [7:10]  eta_LH, eta_EC, eta_NBI
    #   [10:14] P_LH, P_EC, P_NBI, P_ICR
    #   [14:17] I_LH, I_EC, I_NBI
    #   [17:23] f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF
    #   [23:29] f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS
    #   [29:33] beta_fast_alpha, betaN_total, tau_sd_alpha, W_fast_alpha
    #   [33:35] V_TF_one, V_CS_geom
    #   [35:40] V_steel_TF, V_sc_TF, V_cu_TF, V_He_TF, V_In_TF
    #   [40:45] V_steel_CS, V_sc_CS, V_cu_CS, V_He_CS, V_In_CS
    #   [45:47] L_cable_TF, L_cable_CS
    #   [47:49] n_sc_TF, n_sc_CS
    #   [49:51] L_sc_strand_TF, L_sc_strand_CS
    #   [51:56] M_steel_TF, M_sc_TF, M_cu_TF, M_In_TF, M_total_TF
    #   [56:61] M_steel_CS, M_sc_CS, M_cu_CS, M_In_CS, M_total_CS
    #   [61]    V_blanket
    if len(_rest) >= 29:
        _f_sc_TF      = _rest[17]
        _f_cu_TF      = _rest[18]
        _f_He_pipe_TF = _rest[19]
        _f_void_TF    = _rest[20]
        _f_He_TF      = _rest[21]
        _f_In_TF      = _rest[22]
        _f_sc_CS      = _rest[23]
        _f_cu_CS      = _rest[24]
        _f_He_pipe_CS = _rest[25]
        _f_void_CS    = _rest[26]
        _f_He_CS      = _rest[27]
        _f_In_CS      = _rest[28]
        # Coil volumes and cable inventory (present when len >= 55)
        if len(_rest) >= 61:
            _V_TF_one       = _rest[33];  _V_CS_geom  = _rest[34]
            _V_steel_TF     = _rest[35];  _V_sc_TF    = _rest[36]
            _V_cu_TF        = _rest[37];  _V_He_TF    = _rest[38];  _V_In_TF = _rest[39]
            _V_steel_CS     = _rest[40];  _V_sc_CS    = _rest[41]
            _V_cu_CS        = _rest[42];  _V_He_CS    = _rest[43];  _V_In_CS = _rest[44]
            _L_cable_TF     = _rest[45];  _L_cable_CS = _rest[46]
            _n_sc_TF        = _rest[47];  _n_sc_CS    = _rest[48]
            _L_sc_strand_TF = _rest[49];  _L_sc_strand_CS = _rest[50]
            _M_steel_TF     = _rest[51];  _M_sc_TF    = _rest[52]
            _M_cu_TF        = _rest[53];  _M_In_TF    = _rest[54];  _M_total_TF = _rest[55]
            _M_steel_CS     = _rest[56];  _M_sc_CS    = _rest[57]
            _M_cu_CS        = _rest[58];  _M_In_CS    = _rest[59];  _M_total_CS = _rest[60]
            _V_blanket      = _rest[61] if len(_rest) >= 62 else np.nan
            # Radial build masses (layout [77:83] — after lifetimes [62:70] and V_rb [70:77])
            if len(_rest) >= 83:
                M_rb_FW       = _rest[77];  M_rb_BB       = _rest[78]
                M_rb_shield   = _rest[79];  M_rb_VV       = _rest[80]
                M_rb_divertor = _rest[81];  M_rb_total    = _rest[82]
            else:
                M_rb_FW = M_rb_BB = M_rb_shield = M_rb_VV = M_rb_divertor = M_rb_total = np.nan
        else:
            (_V_TF_one, _V_CS_geom,
             _V_steel_TF, _V_sc_TF, _V_cu_TF, _V_He_TF, _V_In_TF,
             _V_steel_CS, _V_sc_CS, _V_cu_CS, _V_He_CS, _V_In_CS,
             _L_cable_TF, _L_cable_CS,
             _n_sc_TF, _n_sc_CS,
             _L_sc_strand_TF, _L_sc_strand_CS,
             _M_steel_TF, _M_sc_TF, _M_cu_TF, _M_In_TF, _M_total_TF,
             _M_steel_CS, _M_sc_CS, _M_cu_CS, _M_In_CS, _M_total_CS) = (np.nan,) * 28
            _V_blanket = np.nan
            M_rb_FW = M_rb_BB = M_rb_shield = M_rb_VV = M_rb_divertor = M_rb_total = np.nan
    elif len(_rest) >= 23:
        # Intermediate format: TF fracs only
        _f_sc_TF      = _rest[17]
        _f_cu_TF      = _rest[18]
        _f_He_pipe_TF = _rest[19]
        _f_void_TF    = _rest[20]
        _f_He_TF      = _rest[21]
        _f_In_TF      = _rest[22]
        _f_sc_CS = _f_cu_CS = _f_He_pipe_CS = _f_void_CS = _f_He_CS = _f_In_CS = np.nan
        (_V_TF_one, _V_CS_geom,
         _V_steel_TF, _V_sc_TF, _V_cu_TF, _V_He_TF, _V_In_TF,
         _V_steel_CS, _V_sc_CS, _V_cu_CS, _V_He_CS, _V_In_CS,
         _L_cable_TF, _L_cable_CS,
         _n_sc_TF, _n_sc_CS,
         _L_sc_strand_TF, _L_sc_strand_CS,
         _M_steel_TF, _M_sc_TF, _M_cu_TF, _M_In_TF, _M_total_TF,
         _M_steel_CS, _M_sc_CS, _M_cu_CS, _M_In_CS, _M_total_CS) = (np.nan,) * 28
        _V_blanket = np.nan
        M_rb_FW = M_rb_BB = M_rb_shield = M_rb_VV = M_rb_divertor = M_rb_total = np.nan
    else:
        # Old tuple format: no cable fractions
        _f_sc_TF = _f_cu_TF = _f_He_pipe_TF = _f_void_TF = _f_He_TF = _f_In_TF = np.nan
        _f_sc_CS = _f_cu_CS = _f_He_pipe_CS = _f_void_CS = _f_He_CS = _f_In_CS = np.nan
        (_V_TF_one, _V_CS_geom,
         _V_steel_TF, _V_sc_TF, _V_cu_TF, _V_He_TF, _V_In_TF,
         _V_steel_CS, _V_sc_CS, _V_cu_CS, _V_He_CS, _V_In_CS,
         _L_cable_TF, _L_cable_CS,
         _n_sc_TF, _n_sc_CS,
         _L_sc_strand_TF, _L_sc_strand_CS,
         _M_steel_TF, _M_sc_TF, _M_cu_TF, _M_In_TF, _M_total_TF,
         _M_steel_CS, _M_sc_CS, _M_cu_CS, _M_In_CS, _M_total_CS) = (np.nan,) * 28
        _V_blanket = np.nan
        M_rb_FW = M_rb_BB = M_rb_shield = M_rb_VV = M_rb_divertor = M_rb_total = np.nan

    # ── Profile peaking / pedestal parameters and Miller V'(rho) ─────────────
    try:
        Vprime_data = _design_Vprime(config, results)
    except Exception:
        Vprime_data = None   # Non-critical: figures fall back gracefully
    # rho_ped as the volume label of the profiles (deck value in rho_tor)
    nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _profile_params(
        config, Vprime_data)

    # ── Impurity inventory {species: n_imp / n_e}, every species of the deck ─
    _impurities = dict(zip(*_parse_impurity_inventory(config)))

    # ── TF coil count: same deterministic call as in run() ───────────────────
    _N_TF, _ripple, Delta_TF = Number_TF_coils(
        config.R0, config.a, config.b,
        config.ripple_adm, config.L_min,
    )
    N_TF     = int(_N_TF)
    # Delta_TF: extra outboard radial clearance required by port-access
    # constraint [m]. Forwarded to _resolve_build() so that R_TF_out
    # is consistent with the ripple model: r2 = R0 + a + b + Delta_TF.
    Delta_TF = float(Delta_TF)

    # ── Float extraction. A failed design keeps NaN, so that neither the
    # figures nor the report show a made-up value.
    def _f(x):
        try:
            return float(np.real(x))
        except (TypeError, ValueError):
            return np.nan

    # ── TF Princeton-D contour and CS height (single resolution) ─────────────
    # Same rule as run(). Solved once and reused by the cost block below.
    try:
        (_, _, _H_TF_cb, _, _, _, _, _, _Z_in_cb) = f_TF_cross_section(
            config.a, config.b, config.R0, c_TF, Delta_TF)
        _Z_bore_half_cb = (float(np.max(_Z_in_cb))
                           if np.all(np.isfinite(_Z_in_cb)) else np.nan)
    except Exception:
        _H_TF_cb, _Z_bore_half_cb = np.nan, np.nan
    if config.H_CS is not None and config.H_CS > 0:
        _H_CS_rd = float(config.H_CS)
    else:
        _H_CS_rd = f_H_CS(_Z_bore_half_cb)
    _H_CS_mod_rd = f_H_CS_module(_H_CS_rd, config.N_sub_CS)

    # ── Techno-economic breakdown (Sheffield), shared evaluate_costs() so the
    # figure matches the RUN report and every other execution mode; empty dict
    # when cost_model is off or any required input is unavailable.
    _cost_bd = {}
    _cr = evaluate_costs(config, results)
    if _cr:
        _cost_bd = {
            "COE": _cr['COE'], "C_invest": _cr['C_invest'],
            "currency": "M EUR (2025)", "coe_unit": "EUR/MWh",
            "capex": {"TF + CS coils":    _cr['C_reac_pc'],
                      "Breeding blanket":  _cr['C_reac_bl'],
                      "Shield + gaps":     _cr['C_reac_sg'],
                      "Divertor targets":  _cr['C_reac_tt'],
                      "Heat transfer system": _cr['C_syst_heat'],
                      "Balance of plant":  _cr['C_syst_BOP'],
                      "Aux. heating":      _cr['C_syst_aux'],
                      "Buildings & other": _cr['C_syst_other'],
                      "Indirect, interest, contingency": _cr['C_ind']},
            # Annual operating costs [M EUR/yr]. The waste cost is defined
            # per unit of energy (coe_waste, EUR/MWh) and converted here.
            "opex": {"O&M":               _cr['C_Op_OM'],
                     "Fuel / consumables": _cr['C_Op_F'],
                     "Waste disposal":     _cr['C_Op_waste_annual']},
            "coe_waste": _cr['C_Op_waste'],
        }

    return {
        # Plasma geometry
        "R0":              config.R0,
        "a":               config.a,
        "Plasma_geometry": config.Plasma_geometry,
        "kappa_edge":      _f(kappa_edge),
        "delta_edge":      _f(delta_edge),
        "kappa_95":        _f(kappa_95),
        "delta_95":        _f(delta_95),
        "Vprime_data":     Vprime_data,
        # q,j profile philosophy (consumed by D0FUS_figures.plot_q_profile)
        "q_profile_mode":  config.q_profile_mode,
        "alpha_J":         config.alpha_J,
        # On-axis field and plasma current
        "B0":          _f(B0),
        "B_max":       config.Bmax_TF,
        **_lift_factors,
        "Ip":          _f(Ip),
        # Volume-averaged kinetics
        "nbar":        _f(nbar),
        "Tbar":        config.Tbar,
        "tau_i_e":     config.tau_i_e,   # T_i/T_e, pressure closure of the figures
        "n_ion_frac":  _ion_density_fraction(config, _f(_f_alpha)),   # n_i/n_e
        "nu_n":        nu_n,
        "nu_T":        nu_T,
        "rho_ped":     rho_ped,
        "n_ped_frac":  n_ped_frac,
        "T_ped_frac":  T_ped_frac,
        # MHD
        "q95":         _f(q95),
        "Z_eff":       _compute_Zeff_effective(config, _f(_f_alpha)),
        "f_He":        _f(_f_alpha),
        # Power balance
        "P_fus":       config.P_fus,
        "P_aux":       config.P_aux_input,
        "Q":           _f(Q),
        **_plant,
        # Impurity seeding {species: n_imp / n_e}
        "impurities":  _impurities,
        # Radial build — inboard side [m]
        "b":           config.b,
        "c_TF":        _f(c_TF),
        "c_WP":        _f(c_WP_TF),
        "c_nose":      _f(c_Nose_TF),
        "c_CS":        _f(c_CS),
        # CS vertical build [m] — resolved by f_H_CS, consumed by the report
        # and by the figures.
        "H_CS":            _H_CS_rd,
        "H_CS_module":     _H_CS_mod_rd,
        "N_TF":            N_TF,
        "N_sub_CS":        config.N_sub_CS,   # CS axial segmentation (1 = monobloc)
        "Delta_TF":        Delta_TF,   # Extra outboard radial clearance from port-access constraint [m]
        "Gap":             config.Gap,
        # ── Radial build sublayer widths ──────────────────────────────────────
        "f_kappa_SOL": config.f_kappa_SOL,
        "Blanket_choice":   config.Blanket_choice,
        "delta_BB_IB": config.delta_BB_IB,
        "delta_BB_OB": config.delta_BB_OB,
        "delta_shield": config.delta_shield,
        # Mechanical configuration key — consumed by _resolve_build() in figures
        # to decide whether the TF-CS gap is applied to the CS outer radius.
        "Choice_Buck_Wedg": config.Choice_Buck_Wedg,
        # Conductor cable fractions (wost = without steel) — TF
        # n_shape_TF, n_shape_CS: steel asymmetry parameter δ_S1/δ_S2 (1 = square jacket)
        "n_shape_TF":  config.n_shape_TF,
        "n_shape_CS":  config.n_shape_CS,
        "Supra_choice": config.Supra_choice,
        "Supra_choice_CS": (config.Supra_choice if config.Supra_choice_CS == 'same'
                            else config.Supra_choice_CS),
        "Steel_fraction_TF": _f(_sf_TF),
        "f_sc_TF":      _f(_f_sc_TF),
        "f_cu_TF":      _f(_f_cu_TF),
        "f_He_pipe_TF": _f(_f_He_pipe_TF),
        "f_void_TF":    _f(_f_void_TF),
        "f_He_TF":      _f(_f_He_TF),
        "f_In_TF":      _f(_f_In_TF),
        # Conductor cable fractions (wost = without steel) — CS
        "Steel_fraction_CS": _f(_sf_CS),
        "f_sc_CS":      _f(_f_sc_CS),
        "f_cu_CS":      _f(_f_cu_CS),
        "f_He_pipe_CS": _f(_f_He_pipe_CS),
        "f_void_CS":    _f(_f_void_CS),
        "f_He_CS":      _f(_f_He_CS),
        "f_In_CS":      _f(_f_In_CS),
        # ── Coil volumes and cable inventory ─────────────────────────────────
        "V_TF_one":        _f(_V_TF_one),   # Single TF coil [m³]
        "V_CS_geom":       _f(_V_CS_geom),   # Total CS solenoid [m³]
        "V_blanket":       _f(_V_blanket),   # Blanket torus [m³]
        "V_steel_TF":      _f(_V_steel_TF),
        "V_sc_TF":         _f(_V_sc_TF),
        "V_cu_TF":         _f(_V_cu_TF),
        "V_He_TF":         _f(_V_He_TF),
        "V_In_TF":         _f(_V_In_TF),
        "V_steel_CS":      _f(_V_steel_CS),
        "V_sc_CS":         _f(_V_sc_CS),
        "V_cu_CS":         _f(_V_cu_CS),
        "V_He_CS":         _f(_V_He_CS),
        "V_In_CS":         _f(_V_In_CS),
        "L_cable_TF":      _f(_L_cable_TF),
        "L_cable_CS":      _f(_L_cable_CS),
        "n_sc_TF":         _f(_n_sc_TF),
        "n_sc_CS":         _f(_n_sc_CS),
        "L_sc_strand_TF":  _f(_L_sc_strand_TF),
        "L_sc_strand_CS":  _f(_L_sc_strand_CS),
        # ── Coil masses ─────────────────────────────────────────────────────
        "M_steel_TF":  _f(_M_steel_TF),
        "M_sc_TF":     _f(_M_sc_TF),
        "M_cu_TF":     _f(_M_cu_TF),
        "M_In_TF":     _f(_M_In_TF),
        "M_total_TF":  _f(_M_total_TF),
        "M_steel_CS":  _f(_M_steel_CS),
        "M_sc_CS":     _f(_M_sc_CS),
        "M_cu_CS":     _f(_M_cu_CS),
        "M_In_CS":     _f(_M_In_CS),
        "M_total_CS":  _f(_M_total_CS),
        # ── Radial build component masses ────────────────────────────────────
        "M_rb_FW":       _f(M_rb_FW),
        "M_rb_BB":       _f(M_rb_BB),
        "M_rb_shield":   _f(M_rb_shield),
        "M_rb_VV":       _f(M_rb_VV),
        "M_rb_divertor": _f(M_rb_divertor),
        "M_rb_total":    _f(M_rb_total),
        "divertor":     _divertor,
        "cost_breakdown": _cost_bd,
    }


def _write_full_report(config, results, output_path, timestamp, input_file_path=None):
    """Write an exhaustive run report (all inputs + all outputs) to disk.

    Produces ``output_path/output_detailed.txt`` in addition to the
    human-readable ``output_highlight.txt``. It lists every GlobalConfig field
    (including values left at their defaults), followed by the full set of
    computed quantities grouped into labelled sections, under a header that
    records the code version, date, git commit, Python version and platform.

    Outputs are gathered from two complementary sources: the curated run dict
    (_build_run_dict) and the named unpacking of the raw results tuple, so both
    the high-level geometry/build quantities and the lower-level physics scalars
    are covered. Any value that cannot be resolved is written as 'n/a' rather
    than aborting the run, so this never interferes with a calculation.

    Parameters
    ----------
    config          : GlobalConfig  Input configuration.
    results         : tuple         Return value of run().
    output_path     : str           Timestamped run output folder.
    timestamp       : str           Run timestamp (folder suffix).
    input_file_path : str, optional Original input deck (for the header).

    Returns
    -------
    str  Path of the detailed report file.
    """
    import platform as _platform
    import subprocess as _subprocess
    import dataclasses as _dc

    # -- Code version: pyproject.toml of the source tree, else the metadata
    #    of the installed package (pip install without the source tree) ----
    def _code_version():
        pp = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        if os.path.exists(pp):
            for line in open(pp, encoding="utf-8"):
                t = line.strip()
                if t.startswith("version"):
                    return t.split("=", 1)[1].strip().strip('"').strip("'")
        try:
            from importlib.metadata import version, PackageNotFoundError
            return version("d0fus")
        except (ImportError, PackageNotFoundError):
            return "unknown"

    def _git_commit():
        try:
            h = _subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=_subprocess.DEVNULL, timeout=2)
            return h.decode().strip()
        except Exception:
            return "n/a"

    def _fmt(v):
        if v is None:
            return "n/a"
        # NumPy scalars and 0-d arrays are formatted as Python numbers
        if isinstance(v, (np.generic, np.ndarray)) and np.ndim(v) == 0:
            v = v.item()
        if isinstance(v, complex):
            v = v.real
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            if not np.isfinite(v):
                return "n/a"
            av = abs(v)
            if av != 0.0 and (av < 1e-3 or av >= 1e6):
                return f"{v:.6e}"
            return f"{v:.6g}"
        return str(v)

    W = 80
    lines = []
    notes = []   # problems met while gathering the outputs, shown under the header

    def H(title):
        lines.append("=" * W)
        lines.append(" " + title)
        lines.append("=" * W)

    def row(name, val, unit=""):
        u = f" [{unit}]" if unit else ""
        lines.append(f"  {name:<46}: {_fmt(val)}{u}")

    # -- Curated run dict (geometry, build, volumes, masses, divertor, cost) -
    try:
        rd = _build_run_dict(config, results)
    except Exception as e:
        rd = {}
        notes.append(f"  (run dict not available, part of the outputs is n/a: {e})")

    # -- Named unpack of the raw results tuple (physics scalars) -------------
    named = {}
    try:
        (B0, B_CS, B_pol, tauE, W_th, Q, Volume, Surface, Ip, Ib, I_CD, I_Ohm,
         nbar, nbar_line, nG, pbar, betaN, betaT, betaP, qstar, q95,
         P_CD, P_sep, P_Thresh, eta_CD, P_elec, P_wallplug,
         cost, P_Brem, P_syn, P_line, P_line_core,
         heat, heat_par, heat_pol, lambda_q, q_target,
         P_wall_rad, P_wall_div, Gamma_n, f_alpha, tau_alpha,
         J_TF, J_CS_1, c_TF, c_WP_TF, c_Nose_TF, sz_TF, st_TF, sr_TF, sf_TF,
         c_CS, sz_CS, st_CS, sr_CS, sf_CS, B_CS2, J_CS2,
         r_minor, r_sep, r_c, r_d, kappa_edge, kappa_95, delta_edge, delta_95,
         *rest) = results
        named = dict(
            B0=B0, B_CS=B_CS, B_pol=B_pol, tauE=tauE, W_th=W_th, Q=Q,
            Volume=Volume, Surface=Surface, Ip=Ip, Ib=Ib, I_CD=I_CD, I_Ohm=I_Ohm,
            nbar=nbar, nbar_line=nbar_line, nG=nG, pbar=pbar,
            betaN=betaN, betaT=betaT, betaP=betaP, qstar=qstar, q95=q95,
            P_CD=P_CD, P_sep=P_sep, P_Thresh=P_Thresh, eta_CD=eta_CD,
            P_elec=P_elec, P_wallplug=P_wallplug, cost=cost,
            P_Brem=P_Brem, P_syn=P_syn, P_line=P_line, P_line_core=P_line_core,
            heat=heat, heat_par=heat_par, heat_pol=heat_pol,
            lambda_q=lambda_q, q_target=q_target,
            P_wall_rad=P_wall_rad, P_wall_div=P_wall_div, Gamma_n=Gamma_n,
            f_alpha=f_alpha, tau_alpha=tau_alpha,
            J_TF=J_TF, J_CS_1=J_CS_1, sz_TF=sz_TF, st_TF=st_TF, sr_TF=sr_TF,
            sf_TF=sf_TF, sz_CS=sz_CS, st_CS=st_CS, sr_CS=sr_CS, sf_CS=sf_CS,
            B_CS2=B_CS2, J_CS2=J_CS2,
            r_minor=r_minor, r_sep=r_sep, r_c=r_c, r_d=r_d,
            kappa_edge=kappa_edge, kappa_95=kappa_95,
            delta_edge=delta_edge, delta_95=delta_95)
        # Documented head of *_rest (fluxes, CD breakdown, fast-alpha).
        names_rest = [
            "psi_PI", "psi_RampUp", "psi_plateau", "psi_PF", "psi_CS", "Vloop", "li",
            "eta_LH", "eta_EC", "eta_NBI", "P_LH", "P_EC", "P_NBI", "P_ICR",
            "I_LH", "I_EC", "I_NBI",
            "f_sc_TF", "f_cu_TF", "f_He_pipe_TF", "f_void_TF", "f_He_TF", "f_In_TF",
            "f_sc_CS", "f_cu_CS", "f_He_pipe_CS", "f_void_CS", "f_He_CS", "f_In_CS",
            "beta_fast_alpha", "betaN_total", "tau_sd_alpha", "W_fast_alpha"]
        for i, nm in enumerate(names_rest):
            if i < len(rest):
                named[nm] = rest[i]
        # Coil-scale (overall) current densities, derived from Ampere's law
        # and the converged build. J_TF / J_CS_1 returned by the solvers are
        # J_wost, i.e. referenced to the non-steel cross-section only; J_coil
        # is referenced to the full coil cross-section (steel jacket, nose and
        # backplate included). See f_J_coil_TF / f_J_coil_CS.
        named["J_wost_TF"] = J_TF
        named["J_wost_CS"] = J_CS_1
        named["J_coil_TF"] = f_J_coil_TF(config.Bmax_TF,
                                         config.R0 - config.a - config.b,
                                         c_TF)
        named["J_coil_CS"] = f_J_coil_CS(B_CS, c_CS)
    except Exception as e:
        notes.append(f"  (results tuple not fully unpacked, part of the outputs is n/a: {e})")

    if len(results) >= 178:
        named["LF_TF"], named["LF_CS"] = results[176:178]

    # Component lifetime / availability live at fixed absolute tuple indices.
    def _at(idx):
        try:
            return results[idx]
        except Exception:
            return None
    t_life_bl_fpy = _at(128)
    t_life_div_fpy = _at(129)
    t_life_bl_yr = _at(130)
    t_life_div_yr = _at(131)
    T_op_limit = _at(132)
    dt_rep_eff = _at(133)
    Av = _at(134)
    CF = _at(135)
    # Radial build layer volumes, results[136:143]
    for _i, _k in enumerate(("V_rb_SOL", "V_rb_FW", "V_rb_BB", "V_rb_shield",
                             "V_rb_VV", "V_rb_gap_TF", "V_rb_divertor")):
        named[_k] = _at(136 + _i)

    def get(key):
        if key in named:
            return named[key]
        return rd.get(key, None)

    def emit(section, items):
        H(section)
        for label, key, unit in items:
            row(label, get(key), unit)
        lines.append("")

    # ===================================================================
    # HEADER
    # ===================================================================
    H("D0FUS - DETAILED RUN REPORT")
    lines.append(f"  Code version  : {_code_version()}")
    lines.append(f"  Date / time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Git commit    : {_git_commit()}")
    lines.append(f"  Python        : {_platform.python_version()}")
    lines.append(f"  Platform      : {_platform.platform()}")
    _deck = os.path.basename(input_file_path) if input_file_path else "n/a"
    lines.append(f"  Input deck    : {_deck}")
    lines.append(f"  Output folder : {os.path.basename(output_path)}")
    lines.append(f"  Run timestamp : {timestamp}")
    lines.append("")
    lines.append("  This report lists every input (including defaults) and every")
    lines.append("  computed output. It complements input_parameters.txt (inputs as")
    lines.append("  given) and output_highlight.txt (human-readable summary).")
    lines.append("")
    if notes:
        lines.extend(notes + [""])

    # ===================================================================
    # 1. INPUTS  (complete GlobalConfig, including default values)
    # ===================================================================
    # The configuration given to run() has its operating point resolved
    # (Tbar_mode or P_fus_mode = 'greenwald' replaced by the solved value).
    # Each field that differs from the deck is flagged with the deck value.
    H("1. INPUTS - configuration used by run() (including default values)")
    deck = None
    if input_file_path and os.path.exists(input_file_path):
        try:
            deck = load_config_from_file(input_file_path, verbose=0)
        except Exception as e:
            lines.append(f"  (deck values not available for comparison: {e})")
    for fld in _dc.fields(config):
        v = getattr(config, fld.name, None)
        tname = getattr(fld.type, "__name__", str(fld.type))
        note = ""
        if deck is not None and getattr(deck, fld.name) != v:
            note = f"    (deck: {_fmt(getattr(deck, fld.name))})"
        lines.append(f"  {fld.name:<46}: {_fmt(v)}    [{tname}]{note}")
    lines.append("")

    # ===================================================================
    # OUTPUTS
    # ===================================================================
    A_val = None
    try:
        A_val = float(rd.get("R0")) / float(rd.get("a"))
    except Exception:
        A_val = None
    rd_A = dict(rd)
    rd_A["A"] = A_val
    # local shadow so 'A' resolves through get()
    named.setdefault("A", A_val)

    emit("2. OUTPUTS - Plasma geometry & shape", [
        ("Major radius R0", "R0", "m"),
        ("Minor radius a", "a", "m"),
        ("Aspect ratio A = R0/a", "A", "-"),
        ("Geometry model", "Plasma_geometry", ""),
        ("Elongation kappa_edge (LCFS)", "kappa_edge", "-"),
        ("Elongation kappa_95", "kappa_95", "-"),
        ("Triangularity delta_edge (LCFS)", "delta_edge", "-"),
        ("Triangularity delta_95", "delta_95", "-"),
        ("Plasma volume", "Volume", "m^3"),
        ("Plasma surface", "Surface", "m^2"),
        ("Inboard LCFS radius r_minor = R0 - a", "r_minor", "m"),
        ("TF leg plasma-side radius r_sep = R0-a-b", "r_sep", "m"),
        ("CS outer radius r_c", "r_c", "m"),
        ("CS bore radius r_d", "r_d", "m"),
    ])

    emit("3. OUTPUTS - Magnetic field & current", [
        ("Toroidal field on axis B0", "B0", "T"),
        ("Peak field B_max", "B_max", "T"),
        ("CS field B_CS", "B_CS", "T"),
        ("Poloidal field B_pol", "B_pol", "T"),
        ("Plasma current Ip", "Ip", "MA"),
        ("Bootstrap current Ib", "Ib", "MA"),
        ("Current-drive current I_CD", "I_CD", "MA"),
        ("Ohmic current I_Ohm", "I_Ohm", "MA"),
        ("Edge safety factor q95", "q95", "-"),
        ("Cylindrical safety factor q*", "qstar", "-"),
        ("Loop voltage Vloop", "Vloop", "V"),
        ("Internal inductance li", "li", "-"),
    ])

    emit("4. OUTPUTS - Density, temperature, pressure, beta", [
        ("Volume-avg density nbar", "nbar", "1e20 m^-3"),
        ("Line-avg density nbar_line", "nbar_line", "1e20 m^-3"),
        ("Density limit used nG (model x margin)", "nG", "1e20 m^-3"),
        ("Volume-avg temperature Tbar", "Tbar", "keV"),
        ("Volume-avg pressure pbar", "pbar", "MPa"),
        ("Effective charge Z_eff", "Z_eff", "-"),
        ("Helium ash fraction f_He = n_He/n_e", "f_He", "-"),
        ("Normalised beta betaN (thermal)", "betaN", "% m T/MA"),
        ("Toroidal beta betaT", "betaT", "-"),
        ("Poloidal beta betaP", "betaP", "-"),
        ("Total normalised beta betaN_total", "betaN_total", "% m T/MA"),
        ("Fast-alpha beta beta_fast_alpha", "beta_fast_alpha", "-"),
        ("Density profile exponent nu_n", "nu_n", "-"),
        ("Temperature profile exponent nu_T", "nu_T", "-"),
        ("Pedestal radius rho_ped, volume label rho_V", "rho_ped", "-"),
        ("Pedestal density fraction n_ped_frac", "n_ped_frac", "-"),
        ("Pedestal temperature fraction T_ped_frac", "T_ped_frac", "-"),
    ])

    emit("5. OUTPUTS - Confinement & fusion", [
        ("Energy confinement time tauE", "tauE", "s"),
        ("Thermal stored energy W_th", "W_th", "J"),
        ("Fusion power P_fus", "P_fus", "MW"),
        ("Fusion gain Q", "Q", "-"),
        ("He ash confinement tau*_He = C_Alpha tauE", "tau_alpha", "s"),
        ("Fast-alpha slowing time tau_sd_alpha", "tau_sd_alpha", "s"),
        ("Fast-alpha stored energy W_fast_alpha", "W_fast_alpha", "MJ"),
    ])

    emit("6. OUTPUTS - Power balance", [
        ("Auxiliary power input P_aux_input", "P_aux", "MW"),
        ("Heating + CD power P_CD (incl. ICRH)", "P_CD", "MW"),
        ("Power across separatrix P_sep", "P_sep", "MW"),
        ("L-H threshold power P_Thresh", "P_Thresh", "MW"),
        ("Net electric power P_elec", "P_elec", "MWe"),
        ("Wall-plug power P_wallplug", "P_wallplug", "MW"),
        ("CD figure of merit gamma_CD", "eta_CD", "1e20 A W^-1 m^-2"),
        ("Bremsstrahlung P_Brem", "P_Brem", "MW"),
        ("Synchrotron P_syn", "P_syn", "MW"),
        ("Impurity radiation P_line (Mavrin Lz)", "P_line", "MW"),
        ("Impurity radiation, rho < rho_rad_core", "P_line_core", "MW"),
    ])

    emit("7. OUTPUTS - Current-drive breakdown", [
        ("LH figure of merit gamma_LH", "eta_LH", "1e20 A W^-1 m^-2"),
        ("EC figure of merit gamma_EC", "eta_EC", "1e20 A W^-1 m^-2"),
        ("NBI figure of merit gamma_NBI", "eta_NBI", "1e20 A W^-1 m^-2"),
        ("LH power P_LH", "P_LH", "MW"),
        ("EC power P_EC", "P_EC", "MW"),
        ("NBI power P_NBI", "P_NBI", "MW"),
        ("ICRH power P_ICR", "P_ICR", "MW"),
        ("LH-driven current I_LH", "I_LH", "MA"),
        ("EC-driven current I_EC", "I_EC", "MA"),
        ("NBI-driven current I_NBI", "I_NBI", "MA"),
    ])

    emit("8. OUTPUTS - Poloidal flux budget", [
        ("Plasma initiation flux psi_PI", "psi_PI", "Wb"),
        ("Ramp-up flux psi_RampUp", "psi_RampUp", "Wb"),
        ("Plateau flux psi_plateau", "psi_plateau", "Wb"),
        ("PF flux psi_PF", "psi_PF", "Wb"),
        ("CS flux psi_CS", "psi_CS", "Wb"),
    ])

    # Section 9: exhaust scalars + refined divertor dict.
    H("9. OUTPUTS - Divertor & exhaust")
    for label, key, unit in [
        ("SOL elongation factor f_kappa_SOL", "f_kappa_SOL", "-"),
        ("Divertor heat scaling P_sep/R0", "heat", "MW/m"),
        ("Parallel heat flux proxy P_sep B0/R0", "heat_par", "MW T/m"),
        ("Poloidal proxy P_sep B0/(q95 R0 A)", "heat_pol", "MW T/m"),
        ("SOL power width lambda_q (Eich #15)", "lambda_q", "m"),
        ("Attached unmitigated target flux q_target", "q_target", "MW/m^2"),
        ("Radiative wall load P_rad,tot/S", "P_wall_rad", "MW/m^2"),
        ("Mean SOL exhaust flux P_sep/S", "P_wall_div", "MW/m^2"),
        ("Neutron wall load Gamma_n", "Gamma_n", "MW/m^2"),
    ]:
        row(label, get(key), unit)
    # Two-point model dict (f_heat_two_point and the run() operating point).
    # lambda_q is already listed above.
    _div_units = dict(q_pol_omp="MW/m^2", pitch_B_over_Bpol="-",
                      f_outer_target="-", q_par_u="MW/m^2", T_eu="eV",
                      T_et="eV", n_et="m^-3", Gamma_et="m^-2 s^-1",
                      q_dep_t="MW/m^2", f_pwr_loss_req="-", n_sep="m^-3",
                      f_cooling="-", f_mom="-", theta_deg="deg",
                      q_dep_limit="MW/m^2", q_par0_attached="MW/m^2", m_f="kg",
                      B_pol_omp="T")
    _div = rd.get("divertor") or {}
    if _div:
        lines.append("  Refined two-point divertor model (NaN target values when sheath-limited):")
        for k in sorted(_div.keys()):
            if k != "lambda_q":
                row("  " + str(k), _div[k], _div_units.get(k, ""))
    else:
        lines.append("  Refined two-point divertor model: (not computed)")
    lines.append("")

    emit("10. OUTPUTS - TF coil (winding pack, composition, structure)", [
        ("TF lift factor LF_TF", "LF_TF", "-"),
        ("TF current density J_wost_TF (no steel)", "J_wost_TF", "A/m^2"),
        ("TF current density J_coil_TF (full TF leg)", "J_coil_TF", "A/m^2"),
        ("TF radial thickness c_TF", "c_TF", "m"),
        ("TF winding-pack thickness c_WP", "c_WP", "m"),
        ("TF nose thickness c_nose", "c_nose", "m"),
        ("TF conductor shape factor n_shape_TF", "n_shape_TF", "-"),
        ("Number of TF coils N_TF", "N_TF", "-"),
        ("TF steel fraction Steel_fraction_TF", "Steel_fraction_TF", "-"),
        ("TF axial stress sz_TF", "sz_TF", "Pa"),
        ("TF hoop stress st_TF", "st_TF", "Pa"),
        ("TF radial stress sr_TF", "sr_TF", "Pa"),
        ("TF SC fraction f_sc_TF", "f_sc_TF", "-"),
        ("TF Cu fraction f_cu_TF", "f_cu_TF", "-"),
        ("TF He-pipe fraction f_He_pipe_TF", "f_He_pipe_TF", "-"),
        ("TF void fraction f_void_TF", "f_void_TF", "-"),
        ("TF He fraction f_He_TF", "f_He_TF", "-"),
        ("TF insulator fraction f_In_TF", "f_In_TF", "-"),
        ("TF steel volume V_steel_TF", "V_steel_TF", "m^3"),
        ("TF SC volume V_sc_TF", "V_sc_TF", "m^3"),
        ("TF Cu volume V_cu_TF", "V_cu_TF", "m^3"),
        ("TF He volume V_He_TF", "V_He_TF", "m^3"),
        ("TF insulator volume V_In_TF", "V_In_TF", "m^3"),
        ("Single-TF volume V_TF_one", "V_TF_one", "m^3"),
        ("TF cable length L_cable_TF", "L_cable_TF", "m"),
        ("TF equivalent SC strands per cable n_sc_TF", "n_sc_TF", "-"),
        ("TF strand length L_sc_strand_TF", "L_sc_strand_TF", "m"),
        ("TF steel mass M_steel_TF", "M_steel_TF", "kg"),
        ("TF SC mass M_sc_TF", "M_sc_TF", "kg"),
        ("TF Cu mass M_cu_TF", "M_cu_TF", "kg"),
        ("TF insulator mass M_In_TF", "M_In_TF", "kg"),
        ("TF total mass M_total_TF", "M_total_TF", "kg"),
    ])

    emit("11. OUTPUTS - CS coil (winding pack, composition, structure)", [
        ("CS lift factor LF_CS", "LF_CS", "-"),
        ("CS current density J_wost_CS (no steel)", "J_wost_CS", "A/m^2"),
        ("CS current density J_coil_CS (steel incl.)", "J_coil_CS", "A/m^2"),
        ("CS radial thickness c_CS", "c_CS", "m"),
        ("CS total height H_CS", "H_CS", "m"),
        ("CS module height H_CS_module", "H_CS_module", "m"),
        ("CS module count N_sub_CS", "N_sub_CS", "-"),
        ("CS conductor shape factor n_shape_CS", "n_shape_CS", "-"),
        ("CS steel fraction Steel_fraction_CS", "Steel_fraction_CS", "-"),
        ("CS axial stress sz_CS", "sz_CS", "Pa"),
        ("CS hoop stress st_CS", "st_CS", "Pa"),
        ("CS radial stress sr_CS", "sr_CS", "Pa"),
        ("CS SC fraction f_sc_CS", "f_sc_CS", "-"),
        ("CS Cu fraction f_cu_CS", "f_cu_CS", "-"),
        ("CS He-pipe fraction f_He_pipe_CS", "f_He_pipe_CS", "-"),
        ("CS void fraction f_void_CS", "f_void_CS", "-"),
        ("CS He fraction f_He_CS", "f_He_CS", "-"),
        ("CS insulator fraction f_In_CS", "f_In_CS", "-"),
        ("CS steel volume V_steel_CS", "V_steel_CS", "m^3"),
        ("CS SC volume V_sc_CS", "V_sc_CS", "m^3"),
        ("CS Cu volume V_cu_CS", "V_cu_CS", "m^3"),
        ("CS He volume V_He_CS", "V_He_CS", "m^3"),
        ("CS insulator volume V_In_CS", "V_In_CS", "m^3"),
        ("CS solenoid volume V_CS_geom", "V_CS_geom", "m^3"),
        ("CS cable length L_cable_CS", "L_cable_CS", "m"),
        ("CS equivalent SC strands per cable n_sc_CS", "n_sc_CS", "-"),
        ("CS strand length L_sc_strand_CS", "L_sc_strand_CS", "m"),
        ("CS steel mass M_steel_CS", "M_steel_CS", "kg"),
        ("CS SC mass M_sc_CS", "M_sc_CS", "kg"),
        ("CS Cu mass M_cu_CS", "M_cu_CS", "kg"),
        ("CS insulator mass M_In_CS", "M_In_CS", "kg"),
        ("CS total mass M_total_CS", "M_total_CS", "kg"),
    ])

    emit("12. OUTPUTS - Radial build volumes & masses", [
        ("Blanket+shield radial thickness b", "b", "m"),
        ("Outboard clearance Delta_TF", "Delta_TF", "m"),
        ("Assembly gap Gap", "Gap", "m"),
        ("Blanket concept Blanket_choice", "Blanket_choice", ""),
        ("LCFS-to-TF shell volume V_blanket", "V_blanket", "m^3"),
        ("Scrape-off layer volume V_rb_SOL", "V_rb_SOL", "m^3"),
        ("First-wall volume V_rb_FW", "V_rb_FW", "m^3"),
        ("Breeding-blanket volume V_rb_BB", "V_rb_BB", "m^3"),
        ("Shield volume V_rb_shield", "V_rb_shield", "m^3"),
        ("Vacuum-vessel volume V_rb_VV", "V_rb_VV", "m^3"),
        ("Gaps and voids volume V_rb_gap_TF", "V_rb_gap_TF", "m^3"),
        ("Divertor volume V_rb_divertor", "V_rb_divertor", "m^3"),
        ("First-wall mass M_rb_FW", "M_rb_FW", "kg"),
        ("Breeding-blanket mass M_rb_BB", "M_rb_BB", "kg"),
        ("Shield mass M_rb_shield", "M_rb_shield", "kg"),
        ("Vacuum-vessel mass M_rb_VV", "M_rb_VV", "kg"),
        ("Divertor mass M_rb_divertor", "M_rb_divertor", "kg"),
        ("Radial-build total mass M_rb_total", "M_rb_total", "kg"),
    ])

    H("13. OUTPUTS - Component lifetime & availability")
    row("Blanket lifetime t_life_bl_fpy", t_life_bl_fpy, "fpy")
    row("Divertor lifetime t_life_div_fpy", t_life_div_fpy, "fpy")
    row("Blanket calendar lifetime t_life_bl_yr", t_life_bl_yr, "yr")
    row("Divertor calendar lifetime t_life_div_yr", t_life_div_yr, "yr")
    row("Operation time per cycle T_op_limit", T_op_limit, "yr")
    row("Effective replacement downtime dt_rep_eff", dt_rep_eff, "yr")
    row("Plant availability Av", Av, "-")
    row("Capacity factor CF", CF, "-")
    lines.append("")

    H("14. OUTPUTS - Plant electrical balance & pulsed operation")
    row("Instantaneous thermal power P_th", rd.get("P_th"), "MWth")
    row("Pulse reactor-to-turbine thermal power P_th_smoothed (legacy)", rd.get("P_th_smoothed"), "MWth")
    row("Stored thermal energy E_store_th", rd.get("E_store_th"), "MWhth")
    for label, key, unit in (
        ('Electrical smoothing split x', 'x_storage_split', '-'),
        ('Nominal reactor thermal output during pulse', 'P_th_reactor_pulse', 'MWth'),
        ('Reactor to storage during pulse', 'P_th_reactor_to_storage_pulse', 'MWth'),
        ('Reactor to turbine during pulse', 'P_th_reactor_to_turbine_pulse', 'MWth'),
        ('Storage to turbine during dwell', 'P_th_storage_to_turbine_dwell', 'MWth'),
        ('Total turbine thermal input during dwell', 'P_th_turbine_dwell', 'MWth'),
        ('Turbine gross electrical output during pulse', 'P_e_gross_pulse', 'MWe'),
        ('Turbine gross electrical output during dwell', 'P_e_gross_dwell', 'MWe'),
        ('Net electrical output during pulse', 'P_e_net_pulse', 'MWe'),
        ('Net electrical output during dwell', 'P_e_net_dwell', 'MWe'),
        ('Net smoothed electrical output (if feasible)', 'P_e_net_smoothed', 'MWe'),
        ('Cycle-averaged net electrical output', 'P_e_net_cycle_average', 'MWe')):
        row(label, rd.get(key), unit)

    if rd.get('smoothing_feasible') is False:
        lines.append('[!] Electrical smoothing infeasible: x=1, no stored heat; pulse/dwell powers differ.')
    row("Gross electrical power P_e_gross", rd.get("P_e_gross"), "MWe")
    row("CS recharge, cycle average P_coil_CS", rd.get("P_coil_CS"), "MWe")
    row("Total TF + CS feeder current I_feeders", rd.get("I_feeders"), "MA")
    row("Refrigeration load at 4.5 K P_cryo_cool", rd.get("P_cryo_cool"), "kW")
    row("Cryogenic electrical consumption P_cryo_electric", rd.get("P_cryo_electric"), "MWe")
    row("Cryogenic LHe volume V_cryo_LHe", rd.get("V_cryo_LHe"), "L")
    row("Cryogenic LN volume V_cryo_LN", rd.get("V_cryo_LN"), "L")
    row("House load P_house_load (PROCESS model)", rd.get("P_house_load"), "MWe")
    row("Coolant pumping power P_BoP", rd.get("P_BoP"), "MWe")
    row("Variable recirculating power P_recirc_var", rd.get("P_recirc_var"), "MWe")
    row("Fixed recirculating power P_recirc_fix", rd.get("P_recirc_fix"), "MWe")
    row("Total recirculated power P_recirculated", rd.get("P_recirculated"), "MWe")
    row("Annual recirculated electricity E_recirculated", rd.get("E_recirculated"), "MWh/year")
    row("Nominal tritium consumption C_tritium_fpy", rd.get("C_tritium_fpy"), "kg/fpy")
    row("Annual tritium consumption C_tritium_y", rd.get("C_tritium_y"), "kg/year")
    lines.append("")

    H("15. OUTPUTS - Techno-economics (Sheffield 2016)")
    row("Geometric cost proxy (V_build / P_fus)", get("cost"), "m^3/MW")
    _cb = rd.get("cost_breakdown") or {}
    if _cb:
        cur = _cb.get("currency", "M EUR")
        row("Cost of electricity COE", _cb.get("COE"), _cb.get("coe_unit", "EUR/MWh"))
        row("Total constructed capital C_invest", _cb.get("C_invest"), cur)
        lines.append(f"  CapEx components [{cur}]:")
        for k, v in (_cb.get("capex") or {}).items():
            row("  " + str(k), v)
        lines.append(f"  Annual OpEx [{cur}/yr]:")
        for k, v in (_cb.get("opex") or {}).items():
            row("  " + str(k), v)
        row("Waste disposal cost (included in COE)", _cb.get("coe_waste"),
            _cb.get("coe_unit", "EUR/MWh"))
    else:
        lines.append("  (cost model disabled or unavailable)")
    lines.append("")

    H("END OF REPORT")

    out_file = os.path.join(output_path, "output_detailed.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_file


def save_run_output(config: GlobalConfig,
                    results: tuple,
                    output_dir: str,
                    input_file_path: str = None,
                    save_figures: bool = False,
                    verbose: int = 0,
                    view_3d: bool = True) -> str:
    """
    Write a timestamped output directory containing:
      - a copy (or reconstruction) of the input parameters
      - a complete human-readable results report
      - (optional) PNG figures from D0FUS_figures.plot_run()

    Parameters
    ----------
    config          : GlobalConfig  Configuration used for this run.
    results         : tuple         Return value of run().
    output_dir      : str           Base output directory.
    input_file_path : str, optional Original input file to copy verbatim.
    save_figures    : bool          If True, generate and save all run figures
                                    in a ``figures/`` sub-directory alongside
                                    the text report.
    view_3d         : bool          False skips the 3D machine view.

    Returns
    -------
    str : absolute path to the created output directory.
    """
    output_path, timestamp = new_output_dir(os.path.join(output_dir, 'run'),
                                            "Run_D0FUS")
    # Exhaustive machine-readable report (all inputs + all outputs).
    try:
        _write_full_report(config, results, output_path, timestamp, input_file_path)
    except Exception as _e:
        print(f"[detailed report skipped: {_e}]")

    # ── Save input parameters ─────────────────────────────────────────────────
    input_copy = os.path.join(output_path, "input_parameters.txt")
    if input_file_path and os.path.exists(input_file_path):
        shutil.copy2(input_file_path, input_copy)
    else:
        # Reconstruct input file from the GlobalConfig dataclass
        with open(input_copy, "w", encoding="utf-8") as f:
            f.write("# D0FUS Input Parameters\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for key, val in asdict(config).items():
                f.write(f"{key} = {val}\n")

    # ── Unpack results tuple ──────────────────────────────────────────────────
    (B0, B_CS, B_pol,
     tauE, W_th,
     Q, Volume, Surface,
     Ip, Ib, I_CD, I_Ohm,
     nbar, nbar_line, nG, pbar,
     betaN, betaT, betaP,
     qstar, q95,
     P_CD, P_sep, P_Thresh, eta_CD, P_elec, P_wallplug,
     cost, P_Brem, P_syn, P_line,
     P_line_core_solution,
     heat, heat_par, heat_pol, lambda_q, q_target,
     P_wall_rad, P_wall_div,
     Gamma_n,
     f_alpha, tau_alpha,
     J_TF, J_CS,
     c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
     d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS,
     r_minor, r_sep, r_c, r_d,
     κ, κ_95, δ, δ_95,
     ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS_Total, Vloop, li_sc,
     eta_LH, eta_EC, eta_NBI,
     P_LH_out, P_EC_out, P_NBI_out, P_ICR_out,
     I_LH_out, I_EC_out, I_NBI_out,
     f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF,
     f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS,
     beta_fast_alpha, betaN_total, tau_sd_alpha, W_fast_alpha,
     V_TF_one, V_CS_geom,
     V_steel_TF, V_sc_TF, V_cu_TF, V_He_TF, V_In_TF,
     V_steel_CS, V_sc_CS, V_cu_CS, V_He_CS, V_In_CS,
     L_cable_TF, L_cable_CS,
     n_sc_TF, n_sc_CS,
     L_sc_strand_TF, L_sc_strand_CS,
     M_steel_TF, M_sc_TF, M_cu_TF, M_In_TF, M_total_TF,
     M_steel_CS, M_sc_CS, M_cu_CS, M_In_CS, M_total_CS,
     V_blanket,
     t_life_bl_fpy, t_life_div_fpy,
     t_life_bl_yr, t_life_div_yr,
     T_op_limit, dt_rep_eff,
     Av, CF,
     V_rb_SOL,
     V_rb_FW, V_rb_BB,
     V_rb_shield, V_rb_VV, V_rb_gap_TF,
     V_rb_divertor,
     M_rb_FW, M_rb_BB, M_rb_shield,
     M_rb_VV, M_rb_divertor, M_rb_total,
     divertor_out,
     *plant_results) = results

    # ── Plant electrical-balance outputs (return indices 150–175) ───────────
    # The original 150 result positions remain unchanged; the new quantities
    # are appended after them.
    if len(plant_results) >= 18:
        (P_th, P_th_smoothed, E_store_th,
         P_e_gross, P_coil_CS, I_feeders,
         P_cryo_cool, P_cryo_electric, V_cryo_LHe,
         V_cryo_LN, P_house_load,
         P_BoP, P_recirc_var, P_recirc_fix,
         P_recirculated, E_recirculated, C_tritium_fpy, C_tritium_y) = plant_results[:18]
    else:
        (P_th, P_th_smoothed, E_store_th,
         P_e_gross, P_coil_CS, I_feeders,
         P_cryo_cool, P_cryo_electric, V_cryo_LHe,
         V_cryo_LN, P_house_load,
         P_BoP, P_recirc_var, P_recirc_fix,
         P_recirculated, E_recirculated, C_tritium_fpy, C_tritium_y) = (np.nan,) * 18

    # ── Recompute N_TF for display (not stored in results tuple) ──────────
    # Same deterministic call as in run().
    _N_TF_disp, _ripple_disp, _Delta_TF_disp = Number_TF_coils(
        config.R0, config.a, config.b, config.ripple_adm, config.L_min)
    N_TF_disp     = int(_N_TF_disp)
    Delta_TF_disp = float(_Delta_TF_disp)

    # ── Deck as given, to tell solved operating-point values from inputs ──
    # resolve_operating_point() replaces Tbar_mode / P_fus_mode = 'greenwald'
    # by the solved value before run(). Those values are outputs [O].
    _deck = (load_config_from_file(input_file_path, verbose=0)
             if input_file_path and os.path.exists(input_file_path) else config)
    _Tbar_io  = 'O' if _deck.Tbar_mode == 'greenwald' else 'I'
    _P_fus_io = 'O' if _deck.P_fus_mode == 'greenwald' else 'I'

    # ── Blanket concept: TBR and radial-build layer breakdown (display only) ──
    (_R_bore_disp, _R_TF_out_disp, _H_TF_disp, _, _, _, _,
     _R_in_disp, _Z_in_disp) = f_TF_cross_section(
        config.a, config.b, config.R0, c, Delta_TF_disp)
    _rb_disp = f_radial_build_layers(
        a=config.a, b=config.b, κ=κ, δ=δ, R0=config.R0,
        Delta_TF=Delta_TF_disp,
        f_kappa_SOL=config.f_kappa_SOL,
        f_div_area_fraction=config.f_div_area_fraction,
        R_tf_in=_R_in_disp, Z_tf_in=_Z_in_disp,
        Blanket_choice=config.Blanket_choice,
        delta_BB_IB=config.delta_BB_IB, delta_BB_OB=config.delta_BB_OB,
        delta_shield=config.delta_shield,
    )
    if not _rb_disp['feasible']:
        import warnings
        warnings.warn('Mandatory radial build infeasible: available IB '
                      f"{_rb_disp['available_ib']:.4g} m, OB "
                      f"{_rb_disp['available_ob']:.4g} m; "
                      f"total cavity volume {_rb_disp['V_available']:.4g} m³. "
                      'Layer geometry and masses are not calculated.',
                      RuntimeWarning, stacklevel=2)
    _delta_BB_ib  = _rb_disp['delta_BB_ib']
    _delta_BB_ob  = _rb_disp['delta_BB_ob']
    _delta_BB_loc = _rb_disp['delta_BB_local']
    _w_BB_loc     = _rb_disp['w_BB_local']
    _blanket_concept = material_blanket(config.Blanket_choice)
    _RB_layers        = _rb_disp['layers']
    _TBR_achieved     = (f_TBR(_delta_BB_loc, config.Blanket_choice, _w_BB_loc)
                         if _rb_disp['feasible'] else np.nan)
    # Conversion parameters as used in run(): [I] from the deck, [O] from the
    # blanket concept (multiplication reduced for a thin breeder)
    _M_blanket, _eta_T, _f_pump = plant_conversion(config)
    if config.M_blanket is None and _rb_disp['feasible']:
        _M_blanket = f_blanket_multiplication(_M_blanket, _delta_BB_loc,
                                              config.Blanket_choice, _w_BB_loc)
    _conv_src = {k: (('I', 'deck') if getattr(config, k) is not None
                     else ('O', 'concept'))
                 for k in ('M_blanket', 'eta_T', 'f_BoP')}
    if config.M_blanket is None:
        _conv_src['M_blanket'] = ('O', f"concept, {_blanket_concept['M_blanket']:.2f} "
                                       "for a thick breeder")

    # ── Magnetic stored energy and ampere-turns (display only) ────────────
    # Princeton-D bore of the sized coil, as in run()
    _r_in_TF  = config.R0 - config.a - config.b
    E_mag_TF_disp = calculate_E_mag_TF(config.Bmax_TF, config.a, config.b,
                                       config.R0, c, Delta_TF_disp)
    # House load breakdown, as in run()
    _hl = house_load_breakdown(config, _R_TF_out_disp, _R_bore_disp, c,
                               _H_TF_disp, float(np.max(_Z_in_disp)), N_TF_disp,
                               E_mag_TF_disp, P_cryo_cool)
    # Ampere's law on toroidal path at inner bore: NI_total = B_max × 2π r_in / μ₀
    NI_TF_total = config.Bmax_TF * 2 * np.pi * _r_in_TF / μ0
    # Per-coil ampere-turns and winding turns
    NI_TF_coil     = NI_TF_total / N_TF_disp if N_TF_disp > 0 else np.nan
    Nturns_TF_coil = NI_TF_coil / config.I_cond if config.I_cond > 0 else np.nan

    # CS geometry from converged radial build
    # CS height: same resolution as run(), reusing the display cross-section
    # above so no extra ODE solve.
    if config.H_CS is not None and config.H_CS > 0:
        _H_CS, _H_CS_src = float(config.H_CS), 'forced'
    else:
        _Z_bore_half = (float(np.max(_Z_in_disp))
                        if np.all(np.isfinite(_Z_in_disp)) else np.nan)
        _H_CS, _H_CS_src = f_H_CS(_Z_bore_half), 'TF bore'
    _H_CS_mod = f_H_CS_module(_H_CS, config.N_sub_CS)
    if np.isfinite(B_CS) and np.isfinite(r_d) and np.isfinite(r_c) and r_c > r_d:
        E_mag_CS_disp = calculate_E_mag_CS(B_CS, r_d, r_c, _H_CS)
        # Ampere's law on axial path through bore: NI_total = B_max × H / μ₀
        NI_CS_total = B_CS * _H_CS / μ0
        # Per-module ampere-turns and winding turns
        _N_mod_CS = config.N_sub_CS if config.N_sub_CS > 0 else 1
        NI_CS_mod     = NI_CS_total / _N_mod_CS
        _I_cond_CS    = (config.I_cond_CS if config.I_cond_CS is not None
                         else config.I_cond)
        Nturns_CS_mod = NI_CS_mod / _I_cond_CS if _I_cond_CS > 0 else np.nan
    else:
        E_mag_CS_disp = np.nan
        NI_CS_total   = np.nan
        NI_CS_mod     = np.nan
        Nturns_CS_mod = np.nan

    # ── Coil-scale (overall) current densities (display only) ─────────────
    # J_wost, returned by the radial build solvers, is referenced to the
    # non-steel cross-section of the conductor. J_coil is referenced to the
    # full coil cross-section, steel jacket and structural case included.
    # Both follow from Ampere's law and the converged build alone; see
    # f_J_coil_TF / f_J_coil_CS in D0FUS_radial_build_functions.
    J_coil_TF = f_J_coil_TF(config.Bmax_TF, _r_in_TF, c)
    J_coil_CS = f_J_coil_CS(B_CS, d)

    # Impurity inventory for the per-species display, with the same coronal
    # <Z_j>(Tbar) as run() for the fuel dilution
    imp_species_list, imp_conc_list = _parse_impurity_inventory(config)
    f_imp_dilution = sum(get_Z_mean(s, config.Tbar) * c
                         for s, c in zip(imp_species_list, imp_conc_list))

    # Profile parameters and Miller V'(rho) for the per-species display,
    # on the run() grid so that report integrals match the physics chain.
    Vprime_data = _design_Vprime(config, results)
    nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _profile_params(
        config, Vprime_data)
    rho_rad_core = config.rho_rad_core                                  # deck, rho_tor
    rho_rad_core_V = float(rho_V_of_rho_tor(rho_rad_core, Vprime_data))  # profiles

    # ── Write results report (console + file) ─────────────────────────────────
    output_file = os.path.join(output_path, "output_highlight.txt")
    with open(output_file, "w", encoding="utf-8") as f:

        class DualWriter:
            """
            Writes to the UTF-8 report file, then to the console.

            The file is written first, so that a console that cannot encode
            a character (cp1252 terminal, pipe) never truncates the report.
            Such characters are replaced on the console only.
            """
            def __init__(self, report_file, console):
                self._file = report_file
                self._console = console
            def write(self, data):
                self._file.write(data)
                try:
                    self._console.write(data)
                except UnicodeEncodeError:
                    enc = getattr(self._console, "encoding", None) or "ascii"
                    self._console.write(data.encode(enc, "replace").decode(enc))
            def flush(self):
                self._file.flush()
                self._console.flush()

        out = DualWriter(f, sys.stdout)

        print("=========================================================================", file=out)
        print("=== D0FUS Calculation Results ===", file=out)
        print("-------------------------------------------------------------------------", file=out)
        # Configuration choices summary — listed first so every output
        # downstream is interpretable in the right physics framework.
        print("[I] Configuration choices",                                                                file=out)
        print(f"[I]  ├ q_profile_mode    (q,j builder)              : {config.q_profile_mode}",            file=out)
        if config.q_profile_mode == 'academic':
            print(f"[I]  │   alpha_J         (PROCESS / Uckan exponent)  : {config.alpha_J}",              file=out)
        print(f"[I]  ├ Plasma_geometry   (cross-section model)      : {config.Plasma_geometry}",            file=out)
        print(f"[I]  ├ Plasma_profiles   (n,T radial preset)        : {config.Plasma_profiles}",            file=out)
        print(f"[I]  ├ Bootstrap_choice  (j_bs model)               : {config.Bootstrap_choice}",           file=out)
        print(f"[I]  ├ CD_source         (current drive scheme)     : {config.CD_source}",                  file=out)
        print(f"[I]  ├ eta_model         (parallel resistivity)     : {config.eta_model}",                  file=out)
        print(f"[I]  └ Option_q95        (MHD scaling for q95)      : {config.Option_q95}",                 file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] R0 (Major Radius)                               : {config.R0:.3f} [m]",   file=out)
        print(f"[I] a  (Minor Radius)                               : {config.a:.3f} [m]",    file=out)
        print(f"[I] b  (BB & neutron shield thickness)              : {config.b:.3f} [m]", file=out)
        # c and d are read directly from results to avoid any gap-dependent ambiguity
        # when interpreting differences of the radii r_c / r_d.
        print(f"[O] c  (TF inboard leg thickness)                   : {c:.3f} [m]",   file=out)
        if config.Choice_Buck_Wedg == 'Wedging':
            print(f"[I] Gap (TF inner bore -> CS outer face)            : {config.Gap:.3f} [m]", file=out)
        print(f"[O] d  (CS winding-pack radial thickness)           : {d:.3f} [m]",     file=out)
        _H_CS_io = 'I' if _H_CS_src == 'forced' else 'O'
        print(f"{'[' + _H_CS_io + '] H_CS   (CS total height, ' + _H_CS_src + ')':<52}: {_H_CS:.3f} [m]", file=out)
        print(f"[O] H_CS/mod (CS module height)                     : {_H_CS_mod:.3f} [m]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Kappa    (Plasma elongation)                    : {κ:.3f}",     file=out)
        print(f"[O] Kappa_95 (Elongation at 95% flux surface)       : {κ_95:.3f}", file=out)
        print(f"[O] Delta    (Plasma triangularity)                 : {δ:.3f}",     file=out)
        print(f"[O] Delta_95 (Triangularity at 95% flux surface)    : {δ_95:.3f}", file=out)
        print(f"[O] Volume   (Plasma volume)                        : {Volume:.3f} [m³]",   file=out)
        print(f"[O] Surface  (First wall area)                      : {Surface:.3f} [m²]",  file=out)
        print(f"[I] Mechanical configuration                        : {config.Choice_Buck_Wedg}", file=out)
        _sc_CS = (config.Supra_choice if config.Supra_choice_CS == 'same'
                  else config.Supra_choice_CS)
        print(f"[I] Superconductor technology (TF / CS)             : {config.Supra_choice} / {_sc_CS}", file=out)
        print(f"[O] N_TF   (Number of TF coils)                     : {N_TF_disp}",              file=out)
        print(f"[O] Delta_TF (Port-access outboard clearance)       : {Delta_TF_disp:.3f} [m]", file=out)
        print(f"[I] N_CS   (Number of CS modules)                   : {config.N_sub_CS}",         file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] Bmax_TF (Peak field on TF conductor)            : {config.Bmax_TF:.3f} [T]",   file=out)
        print(f"[O] B0   (On-axis magnetic field)                   : {B0:.3f} [T]",             file=out)
        print(f"[O] BCS  (CS peak magnetic field)                   : {B_CS:.3f} [T]",           file=out)
        if len(results) >= 178:
            print(f"[O] LF_TF (TF lift factor - REBCO only)             : {results[176]:.5f} [-]", file=out)
            print(f"[O] LF_CS (CS lift factor - REBCO only)             : {results[177]:.5f} [-]", file=out)
        print(f"[O] Engineering current density, by reference section", file=out)
        print(f"[O]  ├ J_wost-TF (TF cable, steel jacket excluded)  : {J_TF/1e6:.3f} [MA/m²]", file=out)
        print(f"[O]  ├ J_coil-TF (TF leg: WP + nose + backplate)    : {J_coil_TF/1e6:.3f} [MA/m²]", file=out)
        print(f"[O]  ├ J_wost-CS (CS cable, steel jacket excluded)  : {J_CS/1e6:.3f} [MA/m²]", file=out)
        print(f"[O]  └ J_coil-CS (CS pack, steel jacket included)   : {J_coil_CS/1e6:.3f} [MA/m²]", file=out)
        print(f"[O] E_mag_TF (TF stored magnetic energy)            : {E_mag_TF_disp/1e9:.3f} [GJ]", file=out)
        print(f"[O] E_mag_CS (CS stored magnetic energy)            : {E_mag_CS_disp/1e9:.3f} [GJ]", file=out)
        print(f"[I] I_cond (Operating current per TF conductor)     : {config.I_cond/1e3:.1f} [kA]", file=out)
        if config.I_cond_CS is not None:
            print(f"[I] I_cond_CS (Operating current per CS conductor)  : {config.I_cond_CS/1e3:.1f} [kA]", file=out)
        print(f"[O] NI_TF/coil  (Ampere-turns per TF coil)          : {NI_TF_coil/1e6:.3f} [MA·turns]", file=out)
        print(f"[O] N_turns_TF  (Winding turns per TF coil)         : {Nturns_TF_coil:.0f}", file=out)
        print(f"[O] NI_CS/mod   (Ampere-turns per CS module)        : {NI_CS_mod/1e6:.3f} [MA·turns]", file=out)
        print(f"[O] N_turns_CS  (Winding turns per CS module)       : {Nturns_CS_mod:.0f}", file=out)
        print(f"[O] Steel fraction TF                               : {Steel_fraction_TF*100:.3f} [%]", file=out)
        print(f"[O] Cable fraction TF  (1 - Steel)                  : {(1-Steel_fraction_TF)*100:.3f} [%]", file=out)
        print(f"[O]  ├ f_sc      (superconductor)                   : {f_sc_TF*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_cu      (copper stabiliser)                : {f_cu_TF*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_He_pipe (He cooling pipe)                  : {f_He_pipe_TF*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_void    (interstitial He void)             : {f_void_TF*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_He      (total He = pipe + void)           : {f_He_TF*100:.2f} [%]", file=out)
        print(f"[O]  └ f_In_cable (turn insulation)                 : {f_In_TF*100:.2f} [%]", file=out)
        print(f"[O] Steel fraction CS                               : {Steel_fraction_CS*100:.3f} [%]", file=out)
        print(f"[O] Cable fraction CS  (1 - Steel)                  : {(1-Steel_fraction_CS)*100:.3f} [%]", file=out)
        print(f"[O]  ├ f_sc      (superconductor)                   : {f_sc_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_cu      (copper stabiliser)                : {f_cu_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_He_pipe (He cooling pipe)                  : {f_He_pipe_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_void    (interstitial He void)             : {f_void_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_He      (total He = pipe + void)           : {f_He_CS*100:.2f} [%]", file=out)
        print(f"[O]  └ f_In_cable (turn insulation)                 : {f_In_CS*100:.2f} [%]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] L_cable_TF (total cable, all TF coils)          : {L_cable_TF/1e3:.3f} [km]", file=out)
        print(f"[O] L_cable_CS (total cable, all CS modules)        : {L_cable_CS/1e3:.3f} [km]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] SC strands per cable  ({config.Supra_choice} / {_sc_CS})", file=out)
        print(f"[O]  ├ n_sc_TF                                      : {n_sc_TF:.0f}", file=out)
        print(f"[O]  └ n_sc_CS                                      : {n_sc_CS:.0f}", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Total SC strand length", file=out)
        print(f"[O]  ├ L_sc_strand_TF                               : {L_sc_strand_TF/1e3:.0f} [km]", file=out)
        print(f"[O]  └ L_sc_strand_CS                               : {L_sc_strand_CS/1e3:.0f} [km]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] V_TF_one  (Single TF coil volume, Princeton-D)  : {V_TF_one:.4f} [m³]",  file=out)
        print(f"[O] V_CS_geom (Total CS solenoid volume)            : {V_CS_geom:.4f} [m³]", file=out)
        print(f"[O] V_blanket (blanket, Miller LCFS→TF inner face)  : {V_blanket:.4f} [m³]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        V_TF_total = V_TF_one * N_TF_disp if np.isfinite(V_TF_one) else np.nan
        print(f"[O] TF material volumes  (total = V_TF_one × N_TF = {V_TF_one:.3f} × {N_TF_disp})", file=out)
        print(f"[O]  ├ V_steel_TF  (structural steel)               : {V_steel_TF:.3f} [m³]", file=out)
        print(f"[O]  ├ V_sc_TF     (superconductor)                 : {V_sc_TF:.3f} [m³]",   file=out)
        print(f"[O]  ├ V_cu_TF     (copper stabiliser)              : {V_cu_TF:.3f} [m³]",   file=out)
        print(f"[O]  ├ V_He_TF     (helium, pipe+void)              : {V_He_TF:.3f} [m³]",   file=out)
        print(f"[O]  └ V_In_TF     (insulation)                     : {V_In_TF:.3f} [m³]",   file=out)
        print(f"[O] CS material volumes  (total solenoid)", file=out)
        print(f"[O]  ├ V_steel_CS  (structural steel)               : {V_steel_CS:.3f} [m³]", file=out)
        print(f"[O]  ├ V_sc_CS     (superconductor)                 : {V_sc_CS:.3f} [m³]",   file=out)
        print(f"[O]  ├ V_cu_CS     (copper stabiliser)              : {V_cu_CS:.3f} [m³]",   file=out)
        print(f"[O]  ├ V_He_CS     (helium, pipe+void)              : {V_He_CS:.3f} [m³]",   file=out)
        print(f"[O]  └ V_In_CS     (insulation)                     : {V_In_CS:.3f} [m³]",   file=out)
        print(f"[O] Blanket components volumes (LCFS→TF inner face, exact body-of-revolution)", file=out)
        print(f"[O]  ├ V_SOL        (scrape-off layer, all around)  : {V_rb_SOL:.1f} [m³]", file=out)
        print(f"[O]  ├ V_FW         (first wall, excl. divertor)    : {V_rb_FW:.1f} [m³]", file=out)
        print(f"[O]  ├ V_BB         (breeder/structure/multip., excl. div.) : {V_rb_BB:.1f} [m³]", file=out)
        print(f"[O]  ├ V_divertor   (divertor, f_div={config.f_div_area_fraction:.2f})          : {V_rb_divertor:.1f} [m³]", file=out)
        print(f"[O]  ├ V_shield     (HT+LT shield, all around)      : {V_rb_shield:.1f} [m³]", file=out)
        print(f"[O]  ├ V_VV         (vacuum vessel, all around)     : {V_rb_VV:.1f} [m³]", file=out)
        print(f"[O]  └ V_gap        (gaps/voids, all around)        : {V_rb_gap_TF:.1f} [m³]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] TF coil masses  (total all N_TF coils, {config.Chosen_Steel_TF} / {config.Supra_choice})", file=out)
        print(f"[O]  ├ M_steel_TF                                   : {M_steel_TF/1e3:.1f} [t]", file=out)
        print(f"[O]  ├ M_sc_TF                                      : {M_sc_TF/1e3:.1f} [t]",    file=out)
        print(f"[O]  ├ M_cu_TF                                      : {M_cu_TF/1e3:.1f} [t]",    file=out)
        print(f"[O]  ├ M_In_TF                                      : {M_In_TF/1e3:.1f} [t]",    file=out)
        print(f"[O]  └ M_total_TF                                   : {M_total_TF/1e3:.1f} [t]",       file=out)
        print(f"[O] CS coil masses  (total solenoid, {config.Chosen_Steel_CS} / {_sc_CS})", file=out)
        print(f"[O]  ├ M_steel_CS                                   : {M_steel_CS/1e3:.1f} [t]", file=out)
        print(f"[O]  ├ M_sc_CS                                      : {M_sc_CS/1e3:.1f} [t]",   file=out)
        print(f"[O]  ├ M_cu_CS                                      : {M_cu_CS/1e3:.1f} [t]",   file=out)
        print(f"[O]  ├ M_In_CS                                      : {M_In_CS/1e3:.1f} [t]",   file=out)
        print(f"[O]  └ M_total_CS                                   : {M_total_CS/1e3:.1f} [t]", file=out)
        print(f"[O] Blanket components masses", file=out)
        print(f"[O]  ├ M_rb_FW       (first wall, excl. div.)       : {M_rb_FW/1e3:.1f} [t]",      file=out)
        print(f"[O]  ├ M_rb_BB       (breeder/structure/multip., excl. div.) : {M_rb_BB/1e3:.1f} [t]", file=out)
        print(f"[O]  ├ M_rb_shield   (HT+LT shield)                 : {M_rb_shield/1e3:.1f} [t]",   file=out)
        print(f"[O]  ├ M_rb_VV       (vacuum vessel)                : {M_rb_VV/1e3:.1f} [t]",      file=out)
        print(f"[O]  ├ M_rb_divertor (divertor)                     : {M_rb_divertor/1e3:.1f} [t]", file=out)
        print(f"[O]  └ M_rb_total    (FW+BB+shield+VV+div)          : {M_rb_total/1e3:.1f} [t]",   file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] Blanket_choice  ({_blanket_concept['label']})", file=out)
        print(f"[I]  ├ Breeder / Multiplier  : {_blanket_concept['breeder']} / {_blanket_concept['multiplier']}", file=out)
        print(f"[I]  ├ Coolant / Structure   : {_blanket_concept['coolant']} / {_blanket_concept['structure']}", file=out)
        print(f"[I]  ├ 6Li enrichment        : {_blanket_concept['Li6_enrichment']}", file=out)
        print(f"[I]  ├ Magnet shielding      : {_blanket_concept['shield_quality']}", file=out)
        print(f"[I]  ├ VV feasible           : {_blanket_concept['VV_feasible']}", file=out)
        _s = _conv_src['M_blanket']
        print(f"[{_s[0]}]  ├ M_blanket             : {_M_blanket:.3f}  ({_s[1]})", file=out)
        _s = _conv_src['eta_T']
        print(f"[{_s[0]}]  ├ eta_T (gross)         : {_eta_T:.3f}  ({_s[1]})", file=out)
        _s = _conv_src['f_BoP']
        print(f"[{_s[0]}]  └ f_BoP (coolant pumps) : {_f_pump:.4f}  ({_s[1]})", file=out)
        if not _rb_disp['feasible']:
            print('[!] INFEASIBLE MANDATORY RADIAL BUILD — no layers calculated', file=out)
            print(f"[O] Available width: IB={_rb_disp['available_ib']:.3f} m, "
                  f"OB={_rb_disp['available_ob']:.3f} m", file=out)
            print(f"[O] Total available LCFS-to-TF volume: "
                  f"{_rb_disp['V_available']:.3f} m³", file=out)
            print('[!] Layer volumes, masses and TBR are unavailable (not zero).', file=out)
        else:
            print(f"[O] TBR_achieved  (breeder {_delta_BB_loc.min():.2f} to {_delta_BB_loc.max():.2f} m, "
                  f"mean {_rb_disp['delta_BB_eff']:.2f} m) : {_TBR_achieved:.3f}  "
                  f"(TBR_max = {_blanket_concept['TBR_max']:.2f})", file=out)
            if _rb_disp['stack_scale'] < 1.0:
                print(f"[!] b = {config.b:.3f} m is thinner than the fixed layers of the "
                      f"{config.Blanket_choice} stack: they are scaled by "
                      f"{_rb_disp['stack_scale']:.2f} and the inboard breeder vanishes", file=out)
            print(f"[O] Radial build layers  (plasma → TF, delta_BB_ib = {_delta_BB_ib:.3f} m, delta_BB_ob = {_delta_BB_ob:.3f} m)", file=out)
            for _i, _layer in enumerate(_RB_layers):
                _last   = _i == len(_RB_layers) - 1
                _branch = "├" if not _last else "└"
                _cont   = "│" if not _last else " "
                print(f"[O]  {_branch} {_layer['name']:<28s}: "
                      f"d_ib={_layer['delta_ib']:.3f} m  d_ob={_layer['delta_ob']:.3f} m  "
                      f"V={_layer['V']:.1f} m³  M={_layer['M']/1e3:.1f} t  "
                      f"rho={_layer['rho']:.1f} kg/m³", file=out)
                # Compact one-line composition (volume fractions, dominant first).
                # The per-material smeared thicknesses and masses are dropped from
                # the text report to keep it readable; the full breakdown remains
                # available in the run dict (_layer['component_masses']) and in the
                # radial-build assembly figure.
                _comp = _layer['composition']
                if _comp and not (len(_comp) == 1 and next(iter(_comp)) == 'void'):
                    _parts = " · ".join(f"{_mat} {_frac*100:.0f}%" for _mat, _frac
                                        in sorted(_comp.items(), key=lambda kv: -kv[1]))
                    print(f"[O]  {_cont}    └ {_parts}", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Psi_PI      (Breakdown flux)                    : {ΨPI:.3f} [Wb]",      file=out)
        print(f"[O] Psi_RampUp  (Ramp-up flux)                      : {ΨRampUp:.3f} [Wb]",  file=out)
        print(f"[O] Psi_Plateau (Flat-top flux)                     : {Ψplateau:.3f} [Wb]", file=out)
        print(f"[O] Psi_PF      (External PF contribution)          : {ΨPF:.3f} [Wb]",      file=out)
        print(f"[O] Psi_CS      (CS flux requirement)               : {ΨCS_Total:.3f} [Wb]", file=out)
        print(f"[O] V_loop      (Steady-state loop voltage)         : {Vloop*1e3:.1f} [mV]", file=out)
        if not np.isfinite(d):
            print("[!] CS sizing failed: no thickness provides Psi_CS within the field and stress limits", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] Fuel   (Fuel mix)                               : {config.Fuel}",  file=out)
        _pfus_label = f"[{_P_fus_io}] P_fus  (Fusion power{', solved' if _P_fus_io == 'O' else ''})"
        print(f"{_pfus_label:<52}: {config.P_fus:.6g} [MW]",  file=out)
        print(f"[O] P_CD   (Heating + current drive power)          : {P_CD:.3f} [MW]",           file=out)
        if config.CD_source == 'Academic':
            print(f"[I]  \u2514 gamma_CD_acad (no per-source breakdown)       : {config.gamma_CD_acad:.4f} [10\u00b2\u2070 A/(W\u00b7m\u00b2)]", file=out)
        else:
            print(f"[O]  \u251c gamma_LH  (LHCD figure of merit)             : {eta_LH:.4f} [10\u00b2\u2070 A/(W\u00b7m\u00b2)]",  file=out)
            print(f"[O]  \u251c gamma_EC  (ECCD figure of merit)             : {eta_EC:.4f} [10\u00b2\u2070 A/(W\u00b7m\u00b2)]",  file=out)
            print(f"[O]  \u251c gamma_NBI (NBCD figure of merit)             : {eta_NBI:.4f} [10\u00b2\u2070 A/(W\u00b7m\u00b2)]", file=out)
            print(f"[O]  \u251c P_LH  / I_LH  (LHCD)                         : {P_LH_out:.2f} MW / {I_LH_out:.3f} MA",  file=out)
            print(f"[O]  \u251c P_EC  / I_EC  (ECCD)                         : {P_EC_out:.2f} MW / {I_EC_out:.3f} MA",  file=out)
            print(f"[O]  \u251c P_NBI / I_NBI (NBCD)                         : {P_NBI_out:.2f} MW / {I_NBI_out:.3f} MA", file=out)
            print(f"[O]  \u2514 P_ICR        (ICRH, heating only)            : {P_ICR_out:.2f} MW",       file=out)
        print(f"[O] P_syn  (Synchrotron, Albajar-Fidone)            : {P_syn:.3f} [MW]",          file=out)
        print(f"[O] P_Brem (Fuel bremsstrahlung, D+T+He)            : {P_Brem:.3f} [MW]",         file=out)
        print(f"[O] P_imp  (Impurity radiation, Mavrin L_z)         : {P_line:.3f} [MW]",         file=out)
        if imp_species_list:
            for _sp, _fc in zip(imp_species_list, imp_conc_list):
                _Pc, _Pt = f_P_line_radiation_profile(
                    _sp, _fc, nbar, config.Tbar, nu_n, nu_T, Volume,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, rho_core=rho_rad_core_V)
                print(f"[O]  \u251c P_imp_{_sp:<3s} (f={_fc:.1e}, brem+line+recomb)      : {_Pt:.3f} [MW] "
                      f"(core={_Pc:.3f}, edge={_Pt - _Pc:.3f})", file=out)
            print(f"[O]  \u2514 f_imp_dilution (\u03a3 <Z_j> c_j, fuel dilution)  : {f_imp_dilution:.4f} [-]", file=out)
        _P_rad_total = P_syn + P_Brem + P_line
        _P_rad_core  = P_syn + P_Brem + P_line_core_solution
        print(f"[O] P_rad_core  (core radiation, \u03c1_tor<{rho_rad_core})        : {_P_rad_core:.3f} [MW]", file=out)
        print(f"[O] P_rad_total (total radiated power)              : {_P_rad_total:.3f} [MW]", file=out)
        print(f"[I] rho_rad_core (core/edge boundary, rho_tor)      : {rho_rad_core:.2f}", file=out)
        print(f"[O] gamma_CD (CD figure of merit, effective)        : {eta_CD:.3f} [10²⁰ A/(W·m²)]", file=out)
        print(f"[O] Q      (Energy gain factor)                     : {Q:.3f}",                   file=out)
        print(f"[O] P_wallplug (Wall-plug heating/CD power)         : {P_wallplug:.3f} [MW]",     file=out)
        if len(plant_results) >= 26:
            (x_split, p_to_store, p_reactor_turbine, p_tank_turbine,
             pe_pulse, pe_dwell) = plant_results[18:24]
            for _key, _val, _unit in (
                ('x_storage_split', x_split, '-'),
                ('P_th_reactor_pulse', P_th, 'MWth'),
                ('P_th_reactor_to_storage_pulse', p_to_store, 'MWth'),
                ('P_th_reactor_to_turbine_pulse', p_reactor_turbine, 'MWth'),
                ('P_th_storage_to_turbine_dwell', p_tank_turbine, 'MWth'),
                ('P_e_gross_pulse', pe_pulse, 'MWe'),
                ('P_e_gross_dwell', pe_dwell, 'MWe'),
                ('P_e_net_cycle_average', P_elec, 'MWe')):
                print(f'[O] {_key:<48}: {_val:.3f} [{_unit}]', file=out)
        if len(results) >= 179 and not results[178]:
            print('[!] Electrical smoothing INFEASIBLE: x=1 fallback (no storage); pulse and dwell net power differ.', file=out)
        if len(plant_results) >= 26:
            print(f'[O] P_e_net_pulse                                    : {plant_results[24]:.3f} [MWe]', file=out)
            print(f'[O] P_e_net_dwell                                    : {plant_results[25]:.3f} [MWe]', file=out)
        print(f"[O] E_store_th (Stored thermal energy)              : {E_store_th:.3f} [MWhth]", file=out)
        print(f"[O] P_coil_CS (CS recharge, cycle average)          : {P_coil_CS:.3f} [MWe]", file=out)
        print(f"[O] I_feeders (Total TF + CS feeder current)        : {I_feeders:.3f} [MA]", file=out)
        print(f"[O] P_cryo_cool (Refrigeration load at 4.5 K)       : {P_cryo_cool:.3f} [kW]", file=out)
        print(f"[O] P_cryo_electric                                 : {P_cryo_electric:.3f} [MWe]", file=out)
        print(f"[O] V_cryo_LHe                                      : {V_cryo_LHe:.3f} [L]", file=out)
        print(f"[O] V_cryo_LN                                       : {V_cryo_LN:.3f} [L]", file=out)
        print(f"[O] P_house_load                                    : {P_house_load:.3f} [MWe]", file=out)
        _lbl = f"facilities (floor area {_hl['A_floor']:.0f} m²)"
        print(f"[O]  ├ {_lbl:<45s}: {_hl['P_facilities']:.3f} [MWe]", file=out)
        print(f"[I]  ├ {'tritium plant':<45s}: {_hl['P_tritium']:.3f} [MWe]", file=out)
        print(f"[I]  ├ {'vacuum pumps':<45s}: {_hl['P_vacuum']:.3f} [MWe]", file=out)
        print(f"[O]  └ {'TF power supplies (bus bars)':<45s}: {_hl['P_TF_supplies']:.3f} [MWe]", file=out)
        print(f"[O] P_BoP (coolant pumping)                         : {P_BoP:.3f} [MWe]", file=out)
        print(f"[O] P_recirc_var                                    : {P_recirc_var:.3f} [MWe]", file=out)
        print(f"[O] P_recirc_fix                                    : {P_recirc_fix:.3f} [MWe]", file=out)
        print(f"[O] P_recirculated                                  : {P_recirculated:.3f} [MWe]", file=out)
        print(f"[O] E_recirculated                                  : {E_recirculated:.3f} [MWh/year]", file=out)
        print(f"[O] C_tritium_fpy                                   : {C_tritium_fpy:.3f} [kg/fpy]", file=out)
        print(f"[O] C_tritium_y                                     : {C_tritium_y:.3f} [kg/year]", file=out)
        _Q_E = P_e_gross / P_recirculated if P_recirculated > 0 else np.nan
        print(f"[O] Q_E    (Engineering gain, P_e_gross/P_recirc)   : {_Q_E:.3f}", file=out)
        print(f"[O] Cost   ((V_BB+V_TF+V_CS) / P_fus)               : {cost:.3f} [m³/MW]",    file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] t_blanket_fpy  (Blanket lifetime, dpa model)       : {t_life_bl_fpy:.3f} [fpy]", file=out)
        print(f"[O] t_div_fpy      (Divertor lifetime, heat model)     : {t_life_div_fpy:.3f} [fpy]", file=out)
        print(f"[O] t_blanket_yr   (Blanket lifetime, calendar)        : {t_life_bl_yr:.3f} [yr]", file=out)
        print(f"[O] t_div_yr       (Divertor lifetime, calendar)       : {t_life_div_yr:.3f} [yr]", file=out)
        print(f"[O] T_op_limit     (Operation per replacement cycle)   : {T_op_limit:.3f} [yr]", file=out)
        print(f"[O] dt_rep_eff     (Effective replacement downtime)    : {dt_rep_eff:.3f} [yr]", file=out)
        print(f"[O] Av             (Plant availability)                : {Av:.4f} [-]", file=out)
        print(f"[O] CF             (Capacity factor)                   : {CF:.4f} [-]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] H              (H-factor)                       : {config.H:.3f}",             file=out)
        print(f"[I] Operation mode                                  : {config.Operation_mode}",    file=out)
        print(f"[I] t_plateau      (Flat-top duration)              : {config.Temps_Plateau_input:.3f} [s]", file=out)
        print(f"[O] tau_E          (Energy confinement time)        : {tauE:.3f} [s]",             file=out)
        print(f"[O] Ip             (Plasma current)                 : {Ip:.3f} [MA]",              file=out)
        print(f"[O] Ib             (Bootstrap current, {config.Bootstrap_choice}) : {Ib:.3f} [MA]", file=out)
        print(f"[O] I_CD           (Driven current, {config.CD_source})       : {I_CD:.3f} [MA]", file=out)
        print(f"[O] I_Ohm          (Ohmic current)                  : {I_Ohm:.3f} [MA]",           file=out)
        print(f"[O] f_b            (Bootstrap fraction)             : {(Ib/Ip)*100:.3f} [%]",      file=out)
        print(f"[O] l_i(3)         (Internal inductance, {config.q_profile_mode})   : {li_sc:.3f} [-]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        _tbar_label = f"[{_Tbar_io}] Tbar  (Volume-averaged Te{', solved' if _Tbar_io == 'O' else ''})"
        print(f"{_tbar_label:<52}: {config.Tbar:.3f} [keV]",    file=out)
        print(f"[I] tau_i_e (Ion/electron temperature ratio Ti/Te)   : {config.tau_i_e:.3f} [-]",   file=out)
        print(f"[O] Ti    (Volume-averaged ion temperature)          : {config.tau_i_e*config.Tbar:.3f} [keV]", file=out)
        print(f"[{'I' if config.Zeff is not None else 'O'}] Z_eff (Effective charge, {'imposed' if config.Zeff is not None else 'computed from inventory+ash'}) : {_compute_Zeff_effective(config, f_alpha):.3f} [-]", file=out)
        print(f"[O] nbar  (Volume-averaged electron density)        : {nbar:.3f} [10²⁰ m⁻³]",     file=out)
        print(f"[O] nbar_line (Line-averaged electron density)      : {nbar_line:.3f} [10²⁰ m⁻³]", file=out)
        # nG from the results tuple is the effective cap of the SELECTED
        # density-limit model scaled by the margin factor Greenwald_limit
        # (n_DL_line × Greenwald_limit); recompute the physical Greenwald
        # density Ip/πa² for display, regardless of the selected model.
        nG_raw_disp = f_nG(Ip, config.a)
        print(f"[O] nG    (Greenwald density, Ip/πa²)               : {nG_raw_disp:.3f} [10²⁰ m⁻³]", file=out)
        print(f"[O] f_GW  (Greenwald fraction, nbar_line/(Ip/πa²))  : {nbar_line/nG_raw_disp:.3f} [-]",  file=out)
        print(f"[I] density_limit_model                             : {config.density_limit_model}", file=out)
        if config.density_limit_model != 'greenwald':
            # Recompute the selected limit from the tuple values for display
            # (same call as in run(); P_tot = P_alpha + P_aux + P_Ohm with
            # P_aux + P_Ohm = P_fus / Q by the D0FUS definition of Q).
            _Pfus_disp = config.P_fus
            _Ptot_disp = (f_P_alpha(_Pfus_disp,
                                    fuel_power_split(config, Vprime_data)[0])
                          + (_Pfus_disp / Q if Q > 0 else 0.0))
            _fnsl_disp = config.f_n_sep * (nbar / nbar_line)
            _fnel_disp = (f_n_edge_ratio(nu_n, rho_ped, n_ped_frac)
                          * (nbar / nbar_line))
            n_DL_line_d, n_DL_native_d, n_DL_conv_d = f_density_limit(
                config.density_limit_model, Ip, config.a,
                P_sol=P_sep, P_tot=_Ptot_disp,
                R0=config.R0, kappa=κ, B0=B0, q_edge=q95,
                Z_eff=_compute_Zeff_effective(config, f_alpha),
                f_n_sep_line=_fnsl_disp, f_n_edge_line=_fnel_disp,
                A_ion=config.Atomic_mass,
                alpha_GR=config.alpha_giacomin, f0=config.f0_zanca)
            print(f"[O] n_DL  ({config.density_limit_model} limit, native: {n_DL_conv_d}) : {n_DL_native_d:.3f} [10²⁰ m⁻³]", file=out)
            print(f"[O] n_DL  (line-averaged equivalent cap)            : {n_DL_line_d:.3f} [10²⁰ m⁻³]", file=out)
            print(f"[O] f_DL  (nbar_line / n_DL_line)                   : {nbar_line/n_DL_line_d:.3f} [-]", file=out)
        print(f"[O] f_DL / f_DL_limit (feasibility ratio, {config.density_limit_model}) : {nbar_line/nG:.3f} [-]", file=out)
        print(f"[O] pbar  (Volume-averaged pressure)                : {pbar:.3f} [MPa]",            file=out)
        print(f"[O] f_He  (Helium ash fraction)                     : {f_alpha*100:.3f} [%]",      file=out)
        print(f"[O] tau_alpha (He ash confinement time tau*_He)     : {tau_alpha:.3f} [s]",         file=out)
        print(f"[O] W_th  (Plasma thermal energy content)           : {W_th/1e6:.3f} [MJ]",        file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] beta_T (Toroidal beta)                          : {betaT*100:.3f} [%]", file=out)
        print(f"[O] beta_P (Poloidal beta)                          : {betaP:.3f}",         file=out)
        print(f"[O] beta_N (Normalised beta, toroidal, thermal)     : {betaN:.3f}",         file=out)
        print(f"[O] beta_N Troyon (toroidal, incl. fast α)          : {betaN_total:.3f}",   file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] q*   (Kink safety factor)                       : {qstar:.3f}", file=out)
        if config.q_profile_mode == 'academic':
            _q95_note = "imposed = q(rho_95) in academic mode"
        else:
            _q95_note = "scaling only; profile q(rho_95) may differ (refined mode)"
        print(f"[O] q95  (Safety factor at 95% flux surface)        : {q95:.3f}    [{_q95_note}]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] P_sep    (P_α+P_CD−P_rad_tot, separatrix power) : {P_sep:.3f} [MW]",    file=out)
        print(f"[O] P_Thresh (L-H power threshold)                  : {P_Thresh:.3f} [MW]", file=out)
        if config.L_H_Scaling_choice in TC26_D_VT:
            # Upper bound of the divertor-configuration uncertainty
            _D_VT = TC26_D_VT[config.L_H_Scaling_choice]
            _lbl = f"[O] P_Thresh, divertor upper bound (D = {_D_VT:.2f})"
            print(f"{_lbl:<52}: {P_Thresh * _D_VT:.3f} [MW]", file=out)
        print(f"[O] P_rad_tot / S_wall  (FW radiative load)         : {P_wall_rad:.3f} [MW/m²]", file=out)
        print(f"[O] P_sep / S_wall  (mean SOL exhaust flux)         : {P_wall_div:.3f} [MW/m²]", file=out)
        print(f"[O] P_sep / R0  (Divertor heat scaling)             : {heat:.3f} [MW/m]",      file=out)
        print(f"[O] P_sep·B0 / R0  (Parallel heat flux proxy)       : {heat_par:.3f} [MW·T/m]", file=out)
        print(f"[O] P_sep·B0 / (q95·R0·A)  (Poloidal heat flux)     : {heat_pol:.3f} [MW·T/m]", file=out)
        print(f"[O] Gamma_n  (Neutron wall load)                    : {Gamma_n:.3f} [MW/m²]",   file=out)
        # ── Two-point divertor model: SOL dissipation needed for the target ─
        if isinstance(divertor_out, dict) and 'f_pwr_loss_req' in divertor_out:
            print(f"[O] f_loss_req (SOL dissipation required, 2PM)      : "
                  f"{divertor_out['f_pwr_loss_req']:.3f} [-]   [{divertor_out['regime']} at f_cooling = "
                  f"{divertor_out['f_cooling']:.2f}]", file=out)
        # ── Lengyel detachment diagnostic (indicative only) ───────────────
        # Required SOL impurity concentration to radiate the two-point-model
        # power-loss fraction, recomputed from the divertor dict.
        _dimp = getattr(config, 'detachment_impurity', '')
        if _dimp and isinstance(divertor_out, dict) and 'q_par_u' in divertor_out:
            try:
                _fpl = min(max(float(divertor_out['f_pwr_loss_req']), 0.0), 0.9999)
                # Lower integral bound: DESIRED post-seeding target
                # temperature (config), not the operating 2PM T_et, which
                # is capped at T_u in the sheath-limited regime.
                _Tt_des = float(getattr(config, 'lengyel_T_target_eV', 25.0))
                _cz_req = f_lengyel_concentration(
                    float(divertor_out['q_par_u']), float(divertor_out['n_sep']),
                    float(divertor_out['T_eu']), _Tt_des,
                    _dimp, _fpl)
                print(f"[O] c_z_req ({_dimp}, Lengyel, f_loss={_fpl:.3f}, T_t={_Tt_des:.0f} eV) : {_cz_req*100:.3f} [%] of n_e (SOL)", file=out)
                if _cz_req > 0.05:
                    # Trace-impurity assumption: at 5 % the impurity adds
                    # 1.5 to 3 to the SOL Z_eff (Ne, <Z> = 6 to 8).
                    print("[!] c_z_req above 5 % of n_e: not a trace impurity, outside the validity of the Lengyel estimate", file=out)
                _core_inv = dict(zip(*_parse_impurity_inventory(config)))
                if _dimp in _core_inv:
                    print(f"[I] c_{_dimp} core seeding (deck inventory)             : {_core_inv[_dimp]*100:.3f} [%]", file=out)
            except (ValueError, KeyError, TypeError) as _e:
                print(f"[O] c_z_req (Lengyel)                               : n/a ({_e})", file=out)
        print("=========================================================================", file=out)

        # ── Runaway Electron Indicators (post-convergence diagnostic) ─────────
        # Indicative only — does NOT affect any convergence loop.
        # li recalculated here from available profile data (not in results tuple).
        print("=== Runaway Electron Indicators (Indicative) ===", file=out)
        print("-------------------------------------------------------------------------", file=out)
        try:
            # Same function as the SCAN maps (runaway_indicators)
            RE = runaway_indicators(config, results, Vprime_data=Vprime_data)
            print(f"[!] INDICATOR ONLY — order-of-magnitude estimate for design comparison.", file=out)
            print(f"[I] tau_TQ         (Thermal quench time)                : {RE['tau_TQ']*1e3:.1f} [ms]", file=out)
            print(f"[I] Te_final       (Post-TQ temperature)                : {RE['Te_final_eV']:.0f} [eV]", file=out)
            print(f"[I] pellet_dilution (SPI/MGI density factor)            : {RE['pellet_dilution']:.1f} [-]", file=out)
            print(f"[O] nbar_diluted   (Post-pellet density)                : {RE['nbar_diluted']:.3f} [1e20 m-3]", file=out)
            print(f"[O] Tbar_diluted   (Pre-TQ temperature, {'isobaric' if RE['pellet_dilution_cools'] else 'unchanged'})       : {RE['Tbar_diluted']:.2f} [keV]", file=out)
            print(f"[O] f_RE_core      (Hot-tail fraction at rho=0)         : {RE['f_RE_core']:.3e} [-]", file=out)
            print(f"[O] f_RE_avg       (Volume-averaged hot-tail fraction)  : {RE['f_RE_avg']:.3e} [-]", file=out)
            print(f"[O] I_RE_seed      (Hot-tail seed, post-pellet n_e)     : {RE['I_RE_seed']:.3e} [A]", file=out)
            print(f"[O] I_RE_aval      (After avalanche, post-pellet)       : {RE['I_RE_avalanche']*1e-6:.3e} [MA]", file=out)
            print(f"[O] I_RE / Ip      (RE-to-plasma current fraction)      : {RE['f_RE_to_Ip']*100:.3e} [%]", file=out)
            print(f"[O] W_mag_RE       (RE magnetic energy = 1/2 Li I^2_RE) : {RE['W_mag_RE']:.3e} [MJ]", file=out)
            print(f"[O] E_RE_kin       (RE kinetic energy, <gamma>=10)      : {RE['E_RE_kin']:.3e} [MJ]", file=out)
        except Exception as _e:
            print(f"[O] RE indicators could not be computed: {_e}", file=out)
        print("=========================================================================", file=out)

        # ── Feasibility verdict (shared rule of SCAN / GENETIC / UNCERTAINTY) ──
        _fz = evaluate_feasibility(config, results)
        print("=== Feasibility (same rule in every execution mode) ===", file=out)
        print("-------------------------------------------------------------------------", file=out)
        _lbl = {'density': 'nbar_line / n_lim  (density)',
                'beta':    'betaN_total / betaN_limit',
                'q':       f"q_limit / {'q95' if config.kink_parameter == 'q95' else 'q*'}",
                'Ip':      'Ip / Ip_limit'}
        for _k, _v in _fz['ratios'].items():
            print(f"[O] {_lbl[_k]:48s}: {_v:.3f} [-]", file=out)
        print(f"[O] {'Radial build closes (r_d, c, d > 0)':48s}: {_fz['build_ok']}", file=out)
        _verdict = 'FEASIBLE' if _fz['feasible'] else f"INFEASIBLE (binding: {_fz['binding']})"
        print(f"[O] {'Verdict':48s}: {_verdict}", file=out)
        print("=========================================================================", file=out)

        # ── Techno-economic cost assessment (post-convergence) ────────────
        # Sheffield & Milora, Fus. Sci. Technol. 70, 14–35 (2016).
        # Uses D0FUS radial build volumes and power balance outputs.
        if config.cost_model != 'None':
            print("=== Cost Assessment — Sheffield (2016) ===", file=out)
            print("-------------------------------------------------------------------------", file=out)
            try:
                # Shared cost evaluation (identical in every execution mode)
                _cr = evaluate_costs(config, results)
                if not _cr:
                    raise ValueError("non-finite cost inputs")
                P_th, V_FI_c = _cr['P_t'], _cr['V_FI']
                (T_op_limit, CF, C_invest, COE,
                 C_ind, C_Op_waste, C_Op_OM, C_Op_F,
                 C_syst_other, C_syst_BOP, C_syst_heat, C_syst_aux,
                 C_reac_tt, C_reac_bl, C_reac_sg, C_reac_pc) = (
                    _cr[k] for k in ('T_op_limit', 'CF', 'C_invest', 'COE',
                                     'C_ind', 'C_Op_waste', 'C_Op_OM', 'C_Op_F',
                                     'C_syst_other', 'C_syst_BOP', 'C_syst_heat',
                                     'C_syst_aux', 'C_reac_tt', 'C_reac_bl',
                                     'C_reac_sg', 'C_reac_pc'))
                C_D = C_invest - C_ind  # direct cost [M EUR]

                # Inputs (condensed)
                print(f"[I] r={config.discount_rate:.0%}  T_life={config.T_life}yr"
                      f"  T_build={config.T_build}yr"
                      f"  contingency={config.contingency:.0%}"
                      f"  Supra_cost={config.Supra_cost_factor:.1f}x", file=out)
                print(f"[I] Util={config.Util_factor:.2f}"
                      f"  Dwell={config.Dwell_factor:.2f}"
                      f"  dt_rep_bl={config.dt_rep_bl:.1f}yr"
                      f"  dt_rep_div={config.dt_rep_div:.1f}yr", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # Derived inputs from D0FUS
                print(f"[O] P_th  (thermal power to BoP)                    : {P_th:.1f} [MW]", file=out)
                print(f"[O] P_e   (net electric power)                      : {P_elec:.1f} [MWe]", file=out)
                if not _cr['net_electric']:
                    print("[!] No net electric output: the COE is undefined, the capital cost takes P_e = 0", file=out)
                print(f"[O] Gamma_n (neutron wall load)                     : {Gamma_n:.3f} [MW/m^2]", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # Component volumes
                print(f"[O] V_bl  (breeding blanket, radial build)          : {_cr['V_bl']:.1f} [m^3]", file=out)
                print(f"[O] V_sg  (shield + VV + gaps, radial build)        : {_cr['V_sg']:.1f} [m^3]", file=out)
                print(f"[O] V_TF_one*N_TF (TF winding packs, Princeton-D)   : {V_TF_one * N_TF_disp:.1f} [m^3]", file=out)
                print(f"[O] V_CS_geom (central solenoid)                    : {V_CS_geom:.1f} [m^3]", file=out)
                print(f"[O] V_FI      (fusion island envelope)              : {V_FI_c:.1f} [m^3]", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # Availability
                print(f"[O] T_op_limit (time before replacement)            : {T_op_limit:.3f} [yr]", file=out)
                print(f"[O] CF (capacity factor)                            : {CF:.3f}", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # CapEx breakdown
                print(f"[O] C_reac_pc (primary coils)                       : {C_reac_pc:.1f} [M EUR]", file=out)
                print(f"[O] C_reac_bl (blanket)                             : {C_reac_bl:.1f} [M EUR]", file=out)
                print(f"[O] C_reac_sg (shield & gaps)                       : {C_reac_sg:.1f} [M EUR]", file=out)
                print(f"[O] C_reac_tt (divertor targets)                    : {C_reac_tt:.1f} [M EUR]", file=out)
                print(f"[O] C_syst_heat (heat transfer)                     : {C_syst_heat:.1f} [M EUR]", file=out)
                print(f"[O] C_syst_aux (heating & CD)                       : {C_syst_aux:.1f} [M EUR]", file=out)
                print(f"[O] C_syst_BOP (turbine & BoP)                      : {C_syst_BOP:.1f} [M EUR]", file=out)
                print(f"[O] C_syst_other (buildings & auxiliaries)          : {C_syst_other:.1f} [M EUR]", file=out)
                print(f"[O] C_D    (total direct cost)                      : {C_D:.1f} [M EUR]", file=out)
                print(f"[O] C_ind  (indirect + contingency)                 : {C_ind:.1f} [M EUR]", file=out)
                print(f"[O] C_invest (total capital cost)                   : {C_invest*1e-3:.3f} [B EUR]", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # OpEx
                print(f"[O] C_Op_OM (annual O&M)                            : {C_Op_OM:.1f} [M EUR/yr]", file=out)
                print(f"[O] C_Op_F  (annual fuel & replacements)            : {C_Op_F:.1f} [M EUR/yr]", file=out)
                print(f"[O] C_Op_waste (waste disposal)                     : {C_Op_waste:.3f} [EUR/MWh]", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # Bottom line
                if _cr['net_electric']:
                    print(f"[O] COE (cost of electricity)                       : {COE:.3f} [EUR/MWh]", file=out)
                else:
                    print("[O] COE (cost of electricity)                       : undefined (P_elec <= 0)", file=out)

            except Exception as _e:
                print(f"[!] Cost assessment failed: {_e}", file=out)
            print("=========================================================================", file=out)

    print(f"\n✓ Results saved to: {output_path}\n")

    # ── Optional figure generation ────────────────────────────────────────────
    if save_figures:
        _generate_run_figures(config, results, output_path, verbose=verbose,
                              view_3d=view_3d)

    return output_path


#%% Main entry point
# ---------------------------------------------------------------------------

def _generate_run_figures(config: GlobalConfig, results: tuple,
                          output_path: str, verbose: int = 0,
                          view_3d: bool = True) -> None:
    """
    Generate and save run-specific figures (16, 15 without the 3D view) into
    output_path/figures/.

    Calls D0FUS_figures.plot_run() which produces only the figures that
    depend on the current run geometry and results.

    Attempts to import D0FUS_figures from the package first, then falls back
    to a direct file-level import so the function works both when D0FUS is
    installed as a package and when run from the source tree.

    Parameters
    ----------
    config      : GlobalConfig  Configuration used for this run.
    results     : tuple         Return value of run().
    output_path : str           Timestamped run output directory.
    verbose     : int           Verbosity level (0 = silent, 1 = warnings, 2 = full).
    """
    # ── Import D0FUS_figures ──────────────────────────────────────────────────
    figs = None
    try:
        from D0FUS_BIB import D0FUS_figures as figs
    except ImportError:
        try:
            import importlib.util, pathlib
            _fig_path = (
                pathlib.Path(__file__).resolve().parent.parent
                / "D0FUS_BIB" / "D0FUS_figures.py"
            )
            _spec = importlib.util.spec_from_file_location("D0FUS_figures", _fig_path)
            figs  = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(figs)
        except Exception as _e:
            if verbose >= 1:
                print(f"  [warn] Could not import D0FUS_figures — figures skipped ({_e})")
            return

    # ── Build figures output directory ────────────────────────────────────────
    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Assemble run-dict ─────────────────────────────────────────────────────
    run_dict = _build_run_dict(config, results)

    # ── q-profile for figures ─────────────────────────────────────────────────
    # Same call and settings as the post-convergence q profile of run()
    # (shaping at rho_95, CD deposition, n_rho = 80), so that the figure shows
    # the l_i and q_0 of the report. Mode-aware dispatch on q_profile_mode.
    try:
        _Ip, _q95, _f_He = run_dict["Ip"], run_dict["q95"], run_dict["f_He"]
        if config.q_profile_mode == 'academic':
            _q_sc = f_q_profile_academic(
                _Ip, _q95,
                config.R0, config.a, run_dict["kappa_edge"],
                alpha_J=config.alpha_J,
                rho_ped=run_dict["rho_ped"],
                n_ped_frac=run_dict["n_ped_frac"],
                T_ped_frac=run_dict["T_ped_frac"],
                Vprime_data=run_dict.get("Vprime_data"),
                kappa_95=run_dict["kappa_95"], rho_95=0.95, n_rho=80)
        else:  # 'refined'
            # Driven currents I_CD [10] and I_LH, I_EC, I_NBI [80:83] [MA]
            _Vpd_fig = run_dict.get("Vprime_data")
            _rho_CD, _delta_CD = _cd_deposition(
                config.CD_source, results[80], results[81], results[82],
                float(rho_V_of_rho_tor(config.rho_EC, _Vpd_fig)),
                float(rho_V_of_rho_tor(config.rho_NBI, _Vpd_fig)))
            _q_sc = f_q_profile_refined(
                _Ip, float(results[10]),
                config.R0, config.a, run_dict["B0"],
                run_dict["kappa_edge"], run_dict["nbar"],
                config.Tbar, _compute_Zeff_effective(config, _f_He),
                run_dict["nu_n"], run_dict["nu_T"],
                trapped_fraction_model=config.trapped_fraction_model,
                eta_model=config.eta_model,
                rho_ped=run_dict["rho_ped"],
                n_ped_frac=run_dict["n_ped_frac"],
                T_ped_frac=run_dict["T_ped_frac"],
                Vprime_data=run_dict.get("Vprime_data"),
                kappa_95=run_dict["kappa_95"], rho_95=0.95,
                delta=run_dict["delta_edge"], delta_95=run_dict["delta_95"],
                rho_CD=_rho_CD, delta_CD=_delta_CD,
                n_ion_frac=_ion_density_fraction(config, _f_He),
                n_rho=80, max_iter=20, tol=1e-3, damping=0.5,
                tau_i_e=config.tau_i_e)
        run_dict["_q_sc"]        = _q_sc
    except Exception as _e:
        if verbose >= 1:
            print(f"  [warn] Self-consistent q-profile for figures failed: {_e}")

    # ── Generate full figure catalogue ────────────────────────────────────────
    if verbose >= 2:
        print("\n  Generating figures:")
    try:
        figs.plot_run(run_dict, save_dir=fig_dir, view_3d=view_3d)
    except Exception as _e:
        if verbose >= 1:
            print(f"  [warn] Figure generation failed: {_e}")

    if verbose >= 2:
        print(f"\n  ✓ Figures saved to: {fig_dir}\n")


def _parse_save_figures_flag(filepath: str) -> bool:
    """
    Read the ``save_figures`` flag from a D0FUS input file without going
    through GlobalConfig (avoids adding a non-physics field to the dataclass).

    Recognised truthy values: 1, true, yes, on  (case-insensitive).
    Any other value, or absence of the key, returns False.

    Parameters
    ----------
    filepath : str  Path to the input file.

    Returns
    -------
    bool
    """
    if not filepath or not os.path.exists(filepath):
        return False
    _TRUTHY = {'1', 'true', 'yes', 'on'}
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.split('#')[0].strip()
                if not line or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                if key.strip().lower() == 'save_figures':
                    return val.strip().lower() in _TRUTHY
    except OSError:
        pass
    return False


def main(input_file: str = None, save_figures: bool = None,
         verbose: int = 0, output_dir: str = None,
         view_3d: bool = True) -> tuple:
    """
    Load a configuration, run the calculation, and save the output report.

    Parameters
    ----------
    input_file   : str, optional
        Path to a plain-text input file.  If None, the module looks for
        ``D0FUS_INPUTS/default_input.txt`` relative to the package root.
        Missing files trigger a warning and the code runs with DEFAULT_CONFIG.
    save_figures : bool or None, optional
        True or False: generate the PNG figures or not, whatever the input
        file says. None (default): follow the ``save_figures`` line of the
        input file, no figures without it.
    verbose : int, optional
        Verbosity level: 0 = silent, 1 = convergence summary + warnings,
        2 = full debug trace.  Default: 0.
    output_dir : str, optional
        Parent folder of the run/ result folders. Default:
        default_output_root().
    view_3d : bool, optional
        False skips the 3D machine view (command-line option --no-3d).

    Returns
    -------
    tuple : return value of run()
    """
    # ── Resolve input file ────────────────────────────────────────────────────
    input_file_path = input_file   # Preserve original path for archiving

    if input_file is None:
        default_input = os.path.join(
            os.path.dirname(__file__), '..', 'D0FUS_INPUTS', 'default_input.txt')
        if os.path.exists(default_input):
            input_file = default_input

    # ── Build GlobalConfig ────────────────────────────────────────────────────
    if input_file and os.path.exists(input_file):
        if verbose >= 2:
            print(f"\nLoading parameters from: {input_file}")
        config = load_config_from_file(input_file, verbose=verbose)
    else:
        if verbose >= 1:
            print("\nWarning: input file not found — running with DEFAULT_CONFIG.")
        config = DEFAULT_CONFIG
        input_file_path = None

    # An explicit argument wins, otherwise the input-file flag decides
    if save_figures is None:
        save_figures = _parse_save_figures_flag(input_file)

    # ── Launch calculation ────────────────────────────────────────────────────
    if verbose >= 2:
        print("\n" + "=" * 73)
        print("Starting D0FUS calculation...")
        print("=" * 73 + "\n")

    try:
        # Resolve the f_GW-prescribed Tbar or P_fus here (not only inside
        # run) so that save_run_output receives the resolved configuration
        # and reports the solved value instead of the deck placeholder.
        config     = resolve_operating_point(config, verbose=max(verbose, 1))
        results    = run(config, verbose=verbose)
        if output_dir is None:
            output_dir = default_output_root()
        os.makedirs(output_dir, exist_ok=True)
        save_run_output(config, results, output_dir, input_file_path,
                        save_figures=save_figures, verbose=verbose,
                        view_3d=view_3d)
        return results

    except Exception as e:
        # Always surface the error message: a silent exit(1) hides
        # physics infeasibilities (e.g. unreachable f_GW target).
        print(f"\n!!! ERROR during calculation !!!")
        print(f"Error message: {str(e)}")
        if verbose >= 1:
            import traceback
            traceback.print_exc()
        sys.exit(1)


#%% Standalone execution

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)
    print("\nD0FUS_run completed successfully!")
