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

import sys
import os
from dataclasses import replace as dc_replace, asdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# D0FUS modules
from D0FUS_BIB.D0FUS_parameterization import (
    GlobalConfig, DEFAULT_CONFIG, material_rho,
    M_blanket_effective, eta_T_effective)
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_BIB.D0FUS_cost_functions import f_costs_Sheffield
from D0FUS_BIB.D0FUS_cost_data import *
from scipy.optimize import brentq


#%% Profile preset table
# ---------------------------------------------------------------------------
# plasma profile peaking factors and pedestal
# parameters across run(), _build_run_dict(), and save_run_output().
#
# Calibration references:
#   'L'        : purely parabolic (no pedestal).
#   'H'        : ITER CORSICA H-mode study (Doyle et al. 2007, PIPB Ch.2)
#   'Advanced' : EU-DEMO 2017 PROCESS reference run
# ---------------------------------------------------------------------------
_PROFILE_PRESETS = {
    'L':        {'nu_n': 0.50, 'nu_T': 1.00, 'rho_ped': 1.00,
                 'n_ped_frac': 0.00, 'T_ped_frac': 0.00},
    'H':        {'nu_n': 0.01, 'nu_T': 2.80, 'rho_ped': 0.95,
                 'n_ped_frac': 0.99, 'T_ped_frac': 0.55},
    'EU-DEMO': {'nu_n': 1.00, 'nu_T': 1.45, 'rho_ped': 0.95,
                 'n_ped_frac': 0.78, 'T_ped_frac': 0.43},
}


#%% Input file loader

def _remap_legacy_value(key: str, val, verbose: int = 0):
    """
    Translate legacy input-file values that the refactor renamed.

    Older input files used 'D0FUS' as the geometry / radial-build sub-mode
    label and 'Sauter' / 'Redl' / 'Freidberg' as bootstrap labels.  These
    are remapped to the new canonical values to keep legacy decks running
    without manual edits.  A warning is printed at verbose >= 1 so the
    user is reminded to update their files.
    """
    if not isinstance(val, str):
        return val

    legacy_map = {
        # Plasma_geometry / Radial_build_model: 'D0FUS' (legacy) -> 'refined'
        ('Plasma_geometry',    'D0FUS'):     'refined',
        ('Radial_build_model', 'D0FUS'):     'refined',
        # Bootstrap_choice: pure Sauter and pure Redl were merged into the
        # composite 'Sauter-Redl' (Sauter 1999/2002 structure with Redl 2021
        # refit).  Freidberg was removed; the closest surviving option is
        # the Segal analytic fit.
        ('Bootstrap_choice',   'Sauter'):    'Sauter-Redl',
        ('Bootstrap_choice',   'Redl'):      'Sauter-Redl',
        ('Bootstrap_choice',   'Freidberg'): 'Segal',
    }
    new = legacy_map.get((key, val))
    if new is not None:
        if verbose >= 1:
            print(f"  [warn]   legacy value '{val}' for '{key}' "
                  f"-> remapped to '{new}'.  Please update your input file.")
        return new
    return val


def load_config_from_file(filepath: str,
                          base: GlobalConfig = None,
                          verbose: int = 0) -> GlobalConfig:
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
        Default: 2 (backward-compatible).

    Returns
    -------
    GlobalConfig
        New immutable configuration with the file overrides applied.
    """
    if base is None:
        base = DEFAULT_CONFIG

    overrides = {}

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
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

            # Attempt numeric conversion; fall back to string
            try:
                val = float(raw_val)
                val = int(val) if val.is_integer() else val
            except ValueError:
                val = raw_val

            # Only accept keys that exist in GlobalConfig
            if key not in GlobalConfig.__dataclass_fields__:
                if verbose >= 1:
                    print(f"  [warn]   '{key}' is not a recognised GlobalConfig field — ignored")
                continue

            overrides[key] = _remap_legacy_value(key, val, verbose=verbose)
            if verbose >= 2:
                print(f"  [input]  {key} = {overrides[key]}")

    return dc_replace(base, **overrides)


#%% Core calculation
# ---------------------------------------------------------------------------

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
        Default: 2 (backward-compatible).

    Returns
    -------
    tuple of 93 scalars
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
        
    # Unpack every field into local names so the downstream physics code
    # is unchanged (no 'config.X' references scattered throughout).
    R0                        = config.R0
    a                         = config.a
    b                         = config.b
    Bmax_TF                   = config.Bmax_TF
    Bmax_CS_adm                   = config.Bmax_CS_adm
    P_fus                     = config.P_fus
    Tbar                      = config.Tbar
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
    betaN_limit               = config.betaN_limit   # Used in D0FUS_scan.py check_radial_build
    q_limit                   = config.q_limit        # Used in D0FUS_scan.py check_radial_build
    Greenwald_limit           = config.Greenwald_limit
    ms                        = config.ms
    Atomic_mass               = config.Atomic_mass
    Zeff                      = config.Zeff
    r_synch                   = config.r_synch
    C_Alpha                   = config.C_Alpha
    Ce                        = config.Ce
    eta_model                 = config.eta_model
    Chosen_Steel              = config.Chosen_Steel
    # NOTE: nu_Steel, Young_modul_Steel, Young_modul_GF are accessed directly
    # from the config object inside f_CS_refined / f_CS_CIRCE — no local alias needed.
    fatigue_CS                = config.fatigue_CS
    Radial_build_model        = config.Radial_build_model
    Choice_Buck_Wedg          = config.Choice_Buck_Wedg
    Supra_choice              = config.Supra_choice
    J_wost_Manual             = config.J_wost_Manual
    coef_inboard_tension      = config.coef_inboard_tension
    F_CClamp                  = config.F_CClamp
    n_shape_TF                = config.n_shape_TF
    c_BP                      = config.c_BP
    TF_grading                = config.TF_grading
    Gap                       = config.Gap
    n_shape_CS                = config.n_shape_CS
    N_sub_CS                  = config.N_sub_CS
    T_helium                  = config.T_helium
    Marge_T_He                = config.Marge_T_He
    f_He_pipe                 = config.f_He_pipe
    f_void                    = config.f_void
    f_In                      = config.f_In
    Marge_T_Nb3Sn             = config.Marge_T_Nb3Sn
    Marge_T_NbTi              = config.Marge_T_NbTi
    Marge_T_REBCO             = config.Marge_T_REBCO
    Eps                       = config.Eps
    Tet                       = config.Tet
    I_cond                    = config.I_cond
    V_max                     = config.V_max
    tau_h_LTS                 = config.tau_h_LTS
    tau_h_HTS                 = config.tau_h_HTS
    T_hotspot                 = config.T_hotspot
    RRR                       = config.RRR
    Dump_resistor_subdivision = config.Dump_resistor_subdivision
    eta_T                     = eta_T_effective(config.Blanket_choice)
    eta_WP                    = config.eta_WP_acad if config.CD_source == 'Academic' else config.eta_RF
    theta_deg                 = config.theta_deg
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
    M_blanket                 = M_blanket_effective(config.Blanket_choice)

    # Core/edge radiation boundary for P_loss convention.
    # Radiation emitted inside ρ < rho_rad_core is subtracted from P_heat
    # in τ_E and the scaling law inversion (it never crosses the separatrix).
    # Radiation emitted at ρ > rho_rad_core is edge radiation that reduces
    # P_sep but does NOT affect confinement.
    # Value set in GlobalConfig (default 0.7, well inside pedestal top).
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
    #    fallback Gaussian centred at rho_CD = 0.4, delta_CD = 0.15 (broad,
    #    safe default).  Mechanically fine but the location is arbitrary; the
    #    user is informed so they can pick a physical CD source if desired.
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
              "refined solver deposits it on a fallback Gaussian at rho=0.4, "
              "delta=0.15.  Set CD_source to LHCD/ECCD/NBCD/Multi for a "
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
    def _parse_impurity_list(species_raw, conc_raw):
        if species_raw is None or str(species_raw).strip() in ('', 'None', 'none'):
            return [], []
        sp_list = [s.strip() for s in str(species_raw).split(',') if s.strip()]
        if not sp_list:
            return [], []
        if isinstance(conc_raw, (int, float)):
            c_list = [float(conc_raw)]
        elif isinstance(conc_raw, str):
            c_str = [s.strip() for s in conc_raw.split(',') if s.strip()]
            c_list = [float(c) for c in c_str] if c_str else []
        else:
            c_list = []
        if len(sp_list) != len(c_list):
            raise ValueError(
                f"Impurity species/concentration mismatch: "
                f"{len(sp_list)} species ({sp_list}) vs "
                f"{len(c_list)} concentrations ({c_list}).")
        paired = [(s, c) for s, c in zip(sp_list, c_list) if c > 0]
        if not paired:
            return [], []
        return [p[0] for p in paired], [p[1] for p in paired]

    imp_species_list, imp_conc_list = _parse_impurity_list(
        config.impurity_species, config.f_imp_core)

    _Z_CORONAL = {'Be': 4, 'C': 6, 'N': 7, 'Ne': 10, 'Ar': 18, 'Kr': 34, 'Xe': 44, 'W': 46}
    f_imp_dilution = sum(_Z_CORONAL.get(s, 10) * c
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
            "Valid options: 'L', 'H', 'Advanced', 'Manual'."
        )

    # ── Validate manual superconductor option ─────────────────────────────────
    if Supra_choice == 'Manual':
        if J_wost_Manual is None or J_wost_Manual <= 0:
            raise ValueError(
                "When Supra_choice='Manual', a valid J_wost_Manual > 0 must be provided "
                f"(current value: {J_wost_Manual}).  "
                "Example: J_wost_Manual = 25e6  # 25 MA/m²"
            )

    # ── Steel stress allowables ───────────────────────────────────────────────
    σ_TF = Steel(Chosen_Steel, config.σ_manual)
    σ_CS = Steel(Chosen_Steel, config.σ_manual)   # Unfatigued allowable [Pa]
    # NOTE: CS fatigue knockdown (config.fatigue_CS) is applied entirely inside
    # the CS solver (f_CS_ACAD / f_CS_refined / f_CS_CIRCE), within the stress
    # residual / f_sigma_diff closure. The effective allowable σ_eff is reduced
    # by fatigue_CS only when Operation_mode == 'Pulsed' AND the light case
    # (CS own electromagnetic load dominant, sigma_theta > 0) governs — which
    # is always the case in Wedging, and conditionally in Bucking (P_CS > 2 P_TF).

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

    # ── Plasma geometry ───────────────────────────────────────────────────────
    κ             = f_Kappa(R0 / a, Option_Kappa, κ_manual, ms)
    κ_95          = f_Kappa_95(κ)
    δ             = f_Delta(κ)
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

    # ── Area elongation for confinement scaling laws ──────────────────────────
    # The IPB98(y,2), DS03, ITER89-P, and L-mode scaling laws were fitted using
    # the "area elongation" κ_a ≡ V_p / (2π²R₀a²), which is the elongation of
    # the equivalent ellipse with the same cross-sectional area as the actual
    # shaped plasma (ITER Physics Basis, NF 39, 1999, Table 5; Kovari et al.,
    # Fusion Eng. Des. 89, 2014, §2.2 — PROCESS convention).
    #
    # For a shaped plasma with triangularity, κ_a < κ_edge because D-shaping
    # reduces the cross-sectional area.  Numerically κ_a ≈ κ_95 ≈ κ_edge/1.12
    # for ITER-class shaping (the agreement is fortuitous but robust within
    # ~2% across the ITER/EU-DEMO parameter space).
    #
    # In contrast, the ITPA20 scaling (Verdoolaege et al., NF 61, 2021) was
    # fitted with the LCFS elongation and triangularity (κ_edge, δ_edge).
    #
    # MHD quantities (q*, β_P, etc.) continue to use κ_edge as they describe
    # the actual LCFS geometry, not the scaling-law convention.
    κ_a = Volume_solution / (2.0 * np.pi**2 * R0 * a**2)

    _KAPPA_AREA_LAWS = {'IPB98(y,2)', 'DS03', 'L-mode', 'L-mode OK', 'ITER89-P'}
    _KAPPA_EDGE_LAWS = {'ITPA20', 'ITPA20-IL'}
    if Scaling_Law in _KAPPA_AREA_LAWS:
        κ_SL = κ_a
    elif Scaling_Law in _KAPPA_EDGE_LAWS:
        κ_SL = κ
    else:
        # Unknown convention: fall back to area elongation (conservative)
        κ_SL = κ_a

    # ── TF coil current density (cable-level sizing) ──────────────────────────
    H_TF       = 2 * (κ * a + b + 1)
    r_in_TF    = R0 - a - b
    r_out_TF   = R0 + a + b
    E_mag_TF   = calculate_E_mag_TF(Bmax_TF, r_in_TF, r_out_TF, H_TF)

    tau_h = tau_h_HTS if Supra_choice == 'REBCO' else tau_h_LTS

    # ── Auto-select f_void based on conductor technology ──────────────────────
    # REBCO tapes are stacked (no interstitial void between round strands).
    # LTS (Nb3Sn, NbTi) use CICC with ~33% void fraction in the strand bundle.
    # If the user explicitly sets f_void in GlobalConfig, that value is used;
    # if left at the default sentinel (None), auto-select based on Supra_choice.
    if f_void is None:
        f_void = 0.00 if 'REBCO' in Supra_choice else 0.33
        # Update config so downstream functions (f_CS_refined, _unpack_CS_config)
        # that read config.f_void directly also see the resolved value.
        config = dc_replace(config, f_void=f_void)

    N_TF, ripple, Delta_TF = Number_TF_coils(R0, a, b, ripple_adm, L_min)
    # N_TF is a float from Number_TF_coils; truncate to int before dividing so
    # that N_sub is an integer count of protection subdivisions, not a ratio.
    # A fractional N_sub would be silently rounded inside calculate_cable_current_density
    # (via int(N_sub)), potentially causing ~1 A/m² discontinuities at scan boundaries.
    # Delta_TF: extra outboard radial clearance imposed by port-access constraint [m].
    # Stored and forwarded to _resolve_build() so that R_TF_out is consistent
    # with the ripple model (r2 = R0 + a + b + Delta_TF) rather than R0 + a + b alone.
    N_sub = int(N_TF) // Dump_resistor_subdivision

    result_J_TF = calculate_cable_current_density(
        sc_type=Supra_choice, B_peak=Bmax_TF, T_op=T_helium, E_mag=E_mag_TF,
        I_cond=I_cond, V_max=V_max, N_sub=N_sub, tau_h=tau_h,
        f_He_pipe=f_He_pipe, f_void=f_void, f_In=f_In,
        T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn, Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
        J_wost_Manual=J_wost_Manual if Supra_choice == 'Manual' else None)
    J_max_TF_conducteur = result_J_TF['J_wost']

    # ── Cable-level fractions (wost = without steel) ─────────────────────────
    f_sc_TF      = result_J_TF['f_sc']
    f_cu_TF      = result_J_TF['f_cu']
    f_He_pipe_TF = result_J_TF['f_He_pipe']
    f_void_TF    = result_J_TF['f_void']
    f_He_TF      = result_J_TF['f_He']
    f_In_TF      = result_J_TF['f_In']

    # ── On-axis magnetic field and alpha power ────────────────────────────────
    B0_solution = f_B0(Bmax_TF, a, b, R0)
    P_Alpha     = f_P_alpha(P_fus)

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
    #    variable scaling and analytical Jacobian.
    #
    #    Both branches use:
    #      • Absolute residuals (not percentages) for scipy compatibility
    #      • Variable scaling to ~O(1) for well-conditioned Jacobians
    #      • Anderson acceleration (via scipy) as a robust fallback
    #      • Per-method debug counter reset for clear diagnostics
    # =========================================================================

    # ── Dilution ceiling (shared by all solvers) ──────────────────────────────
    _fa_max = (1.0 - f_imp_dilution) / 2.0 - 0.01

    # ── Debug counter — reset between solver stages for clear output ─────────
    _dbg_counter = [0]
    _dbg_limit   = [5]   # mutable: how many calls to print per solver stage

    # ── Converged chain cache ─────────────────────────────────────────────────
    # Stores the full physics-chain dict from the last converged solver
    # evaluation so that post-convergence code can read expensive profile-
    # integrated quantities (nbar, pbar, Ib, radiation, ...) without
    # recomputing them.
    _converged_chain = [None]

    # ── q,j profile cache (lagged evaluation, mode-aware dispatcher) ──────
    # Stores the q(ρ) dict from f_q_profile_academic or f_q_profile_refined
    # (selected by config.q_profile_mode) computed with the previous
    # iteration's (Ip, I_CD).  Passed to f_Sauter_Redl_Ib so the bootstrap
    # coefficients see a realistic q(ρ) instead of the constant q_95 fallback.
    # Updated lazily: only recomputed when I_Ohm or Ip changes by more than 8 %.
    # The 'refined' branch supports warm starts via the q_init argument,
    # which collapses subsequent iterations to a single Picard pass.
    _q_profile_cache = [None]
    _q_cache_Ip      = [0.0]     # Ip [MA] at last cache update
    _q_cache_I_Ohm   = [0.0]     # I_Ohm [MA] at last cache update

    def _solve_q_profile(Ip_loc, I_CD_loc, q95_loc, B0_loc, nbar_loc,
                         rho_CD_loc, delta_CD_loc,
                         n_rho=60, max_iter=10, tol=5e-3, damping=0.5,
                         q_init=None):
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
                R0, a, B0_loc, κ, nbar_loc, Tbar, Zeff,
                nu_n, nu_T,
                trapped_fraction_model=trapped_fraction_model,
                eta_model=eta_model,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, kappa_95=κ_95, rho_95=0.95,
                delta=δ, delta_95=δ_95,
                rho_CD=rho_CD_loc, delta_CD=delta_CD_loc,
                q_init=q_init,
                n_rho=n_rho, max_iter=max_iter, tol=tol, damping=damping)
        else:
            raise ValueError(
                f"Unknown q_profile_mode: '{q_profile_mode}'. "
                f"Valid options: 'academic', 'refined'.")


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

        # Expose f_alpha to the CD dispatcher (used by f_etaCD_NBI_physics)
        config._f_alpha = f_alpha

        # Volume-averaged density and pressure
        nbar_loc = f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, κ,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac,
                          Vprime_data=Vprime_data, f_imp=f_imp_dilution)
        pbar_loc = f_pbar(nu_n, nu_T, nbar_loc, Tbar,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac,
                          Vprime_data=Vprime_data)
        nbar_line_loc = f_nbar_line(nbar_loc, nu_n, rho_ped, n_ped_frac)

        # Radiative power losses [MW]
        # Fuel bremsstrahlung: uses Z_eff,fuel (D+T+He only) to avoid
        # double-counting impurity brem already in Mavrin L_z.
        #   Z_eff,fuel = (n_D+n_T)*1 + n_He*4) / ne
        #             = (1 - 2*f_He - f_imp) + 4*f_He = 1 + 2*f_He - f_imp
        # Impurity radiation (brem+line+recomb) is in P_line via Mavrin L_z.
        Zeff_fuel = 1.0 + 2.0 * f_alpha - f_imp_dilution
        P_Brem_loc = f_P_bremsstrahlung(nbar_loc, Tbar, Zeff_fuel,
                                        Volume_solution, nu_n, nu_T,
                                        rho_ped=rho_ped,
                                        n_ped_frac=n_ped_frac,
                                        T_ped_frac=T_ped_frac,
                                        Vprime_data=Vprime_data)
        # Synchrotron: Albajar (2001) + Fidone (2001) wall-reflection.
        # Pedestal parameters are passed so that T₀ and ne₀ are the ACTUAL
        # on-axis values (from pedestal normalization).  The K factor uses
        # the USER's αn/αT because the core profile (ρ < 0.7, where >91%
        # of synchrotron is emitted) is parabolic in (1-(ρ/ρ_ped)²)^αT
        # to better than 3%.  This matches PROCESS convention (Kovari §10:
        # passes temp_plasma_electron_on_axis_kev with user alphan/alphat).
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
        # coreradiationfraction (default 1.0) scales the core radiation term
        # in P_loss only.  Values < 1 account for partial re-absorption,
        # non-coronal transport effects, and scaling-law convention ambiguity.
        # PROCESS uses 0.6 (Kovari et al., FED 89, 2014).
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
                      alpha_R, alpha_B, alpha_n, alpha_I, alpha_P)

        if _dbg:
            print(f"    Ip={Ip_loc:.2f} MA")

        # Bootstrap current
        # q95 is needed by Sauter/Redl models and is always useful for the
        # post-convergence cache; f_q95 is algebraic so the overhead is negligible.
        # Sauter (2016) uses LCFS shaping (κ, δ); ITER_1989 uses 95%-surface
        # values (κ_95, δ_95).  Both sets are passed so f_q95 picks the right one.
        q95_loc = f_q95(B0_solution, Ip_loc, R0, a, κ, δ, κ_95, δ_95,
                        Option_q95=Option_q95)

        if Bootstrap_choice == 'Sauter-Redl':
            Ib_loc = f_Sauter_Redl_Ib(
                R0, a, κ, B0_solution, nbar_loc, Tbar,
                q95_loc, Zeff, nu_n, nu_T,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, kappa_95=κ_95,
                q_profile=_q_profile_cache[0],
                trapped_fraction_model=trapped_fraction_model)
        elif Bootstrap_choice == 'Segal':
            Ib_loc = f_Segal_Ib(
                nu_n, nu_T, a / R0, κ, nbar_loc, Tbar, R0, Ip_loc,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac)
        else:
            raise ValueError(
                f"Unknown Bootstrap_choice: '{Bootstrap_choice}'. "
                f"Valid options: 'Segal', 'Sauter-Redl'.")

        # Current drive — different logic for Pulsed vs Steady-State
        if ss_mode:
            # Steady-State: I_Ohm = 0, all non-bootstrap current is driven.
            # P_CD is derived from the required I_CD and the CD efficiency.
            eta_CD_loc = f_etaCD_effective(
                config, a, R0, B0_solution, nbar_loc, Tbar, nu_n, nu_T,
                Zeff,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac)
            I_Ohm_loc = 0.0
            I_CD_loc  = f_ICD(Ip_loc, Ib_loc, I_Ohm_loc)
            P_CD_loc  = f_PCD(R0, nbar_loc, I_CD_loc, eta_CD_loc)
            P_Ohm_loc = 0.0
            Q_loc     = f_Q(P_fus, P_CD_loc, P_Ohm_loc)
        else:
            # Pulsed mode: P_CD is fixed by the input source powers.
            if CD_source == 'Multi':
                P_CD_loc    = P_LH + P_ECRH + P_NBI + P_ICRH
                eta_LH_loc  = f_etaCD_LH_physics(Tbar, Zeff)
                eta_EC_loc  = f_etaCD_EC_physics(
                    a, R0, Tbar, nbar_loc, Zeff, nu_T, nu_n,
                    rho_EC,
                    theta_EC_pol_deg=config.theta_EC_pol_deg,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac)
                eta_NBI_loc = f_etaCD_NBI_physics(
                    A_beam, E_beam_keV,
                    a, R0, Tbar, nbar_loc, Zeff, nu_T, nu_n,
                    rho_NBI,
                    f_alpha=f_alpha,
                    angle_NBI_deg=config.angle_NBI_deg,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac)
                I_CD_loc = (f_I_CD(R0, nbar_loc, eta_LH_loc,  P_LH)
                          + f_I_CD(R0, nbar_loc, eta_EC_loc,  P_ECRH)
                          + f_I_CD(R0, nbar_loc, eta_NBI_loc, P_NBI))
            else:
                P_CD_loc   = P_aux_input
                eta_CD_loc = f_etaCD_effective(
                    config, a, R0, B0_solution, nbar_loc, Tbar, nu_n, nu_T,
                    Zeff,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac)
                I_CD_loc   = f_I_CD(R0, nbar_loc, eta_CD_loc, P_CD_loc)

            I_Ohm_loc = f_I_Ohm(Ip_loc, Ib_loc, I_CD_loc)
            P_Ohm_loc = f_P_Ohm(I_Ohm_loc, Tbar, R0, a, κ, Z_eff=Zeff,
                                nbar=nbar_loc, eta_model=eta_model,
                                q95=q95_loc)
            Q_loc     = f_Q(P_fus, P_CD_loc, P_Ohm_loc)

        # ── q,j profile cache for bootstrap collisionality ───────────────
        #
        # WHY: The Sauter-Redl bootstrap coefficients L31, L32, L34 depend
        #   on the electron collisionality nu*_e ~ q(rho).  Using a
        #   constant q = q_95 overestimates nu* at mid-radius by a factor
        #   2 to 3, reducing I_bs by roughly 20 %.  A radial q(rho) solves
        #   this; the dispatcher routes between the two modes.
        #
        # HOW: Lagged solve with lazy update via _solve_q_profile().
        #   - 'academic' mode : single deterministic pass, j ~ (1-rho^2)^alpha_J,
        #                       q(rho) analytic from cylindrical Ampere.
        #                       Cached q(rho) is still passed to the global
        #                       Sauter-Redl call so that the global I_bs
        #                       sees a realistic collisionality profile.
        #   - 'refined'  mode : Picard iteration on q(rho) from the composite
        #                       j_Ohm + j_CD + j_bs (5 to 10 iterations cold,
        #                       1 to 2 with warm start from the cache).
        #   - The cache is refreshed when the current decomposition changes
        #     by more than 8 % in either I_Ohm or Ip.
        #   - On first call (cache=None), the bootstrap functions use a
        #     parabolic fallback q_0 + (q_95 - q_0) rho^2 internally.
        #
        # COST: with vectorised eta_redl/eta_sauter, ~5 to 10 ms per
        #   solve in refined mode (cold), ~2 ms warm; ~5 ms in academic.
        #   Total impact on a 1000-point scan: ~10 to 30 s.
        #
        # ROBUSTNESS: the one-iteration lag is benign because q(rho) is a
        #   slow function of the current decomposition.  If the solve fails
        #   (pathological geometry, non-convergence), the previous cache or
        #   the parabolic fallback is used and the outer solver sees no error.

        _do_q_update = False
        if np.isfinite(Ip_loc) and np.isfinite(I_Ohm_loc):
            if _q_profile_cache[0] is None:
                # First call: always compute to escape the parabolic fallback.
                _do_q_update = True
            elif (abs(I_Ohm_loc - _q_cache_I_Ohm[0])
                      > 0.08 * max(abs(_q_cache_I_Ohm[0]), 0.5)
                  or abs(Ip_loc - _q_cache_Ip[0])
                      > 0.08 * max(abs(_q_cache_Ip[0]), 1.0)):
                # Current decomposition has shifted by >8% in either
                # I_Ohm (pulsed) or Ip (steady-state where I_Ohm ≡ 0).
                # Threshold raised from 5% to 8%: q(ρ) shape is a slow
                # function of the current decomposition, so the extra lag
                # has negligible effect on I_bs accuracy.
                _do_q_update = True

        if _do_q_update:
            # Select CD deposition parameters from the active source.
            # The values below are used only by q_profile_mode='refined' to
            # build j_CD(rho); academic mode ignores them (j_CD = 0).
            #   ECCD  : narrow Gaussian at rho_EC (input)
            #   NBCD  : medium Gaussian at rho_NBI (input)
            #   LHCD  : broad Gaussian centred at rho = 0.5
            #   Multi : broad fallback (current-weighted average is computed
            #           more accurately at the post-convergence call site)
            #   Academic / unknown : same broad fallback as Multi.  Triggers
            #           the warning printed at the top of run() when paired
            #           with q_profile_mode='refined'.
            if CD_source == 'ECCD':
                _rho_CD_q, _delta_CD_q = rho_EC, 0.10
            elif CD_source == 'NBCD':
                _rho_CD_q, _delta_CD_q = rho_NBI, 0.15
            elif CD_source == 'LHCD':
                _rho_CD_q, _delta_CD_q = 0.5, 0.20
            else:  # Multi / Academic / unknown
                _rho_CD_q, _delta_CD_q = 0.4, 0.15

            try:
                _q_profile_cache[0] = _solve_q_profile(
                    Ip_loc, I_CD_loc, q95_loc, B0_solution, nbar_loc,
                    rho_CD_loc=_rho_CD_q, delta_CD_loc=_delta_CD_q,
                    n_rho=60, max_iter=10, tol=5e-3, damping=0.5,
                    q_init=_q_profile_cache[0])
                _q_cache_I_Ohm[0] = I_Ohm_loc
                _q_cache_Ip[0]    = Ip_loc
            except Exception:
                pass  # keep previous cache (or None -> parabolic fallback)

        if _dbg:
            print(f"    Ib={Ib_loc:.2f}, I_CD={I_CD_loc:.2f}, "
                  f"I_Ohm={I_Ohm_loc:.2f}, P_Ohm={P_Ohm_loc:.3f}, "
                  f"Q={Q_loc:.2f}")

        # Helium ash fraction from confinement time
        new_fa_loc = f_He_fraction(
            nbar_loc, Tbar, tau_E_loc, C_Alpha, nu_T,
            rho_ped=rho_ped, T_ped_frac=T_ped_frac)

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

    def _solve_pulsed():
        """
        Solve for f_alpha in Pulsed mode.

        In Pulsed mode, P_aux is fixed (user input).  The only self-consistent
        unknown is f_alpha.  Q is derived from Q = P_fus / (P_CD + P_Ohm).
        An inner loop on P_Ohm ensures Ohmic power consistency.

        Strategy:
          1. Anderson mixing on f_alpha (fast, robust)
          2. Newton with finite-difference Jacobian (fallback)
          3. Bisection on the f_alpha residual (guaranteed convergence)

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
        # power from the previous outer residual evaluation.  Instead of
        # starting the inner P_Ohm loop from 0 (which needs 3–10 iterations),
        # we start from the previous value (which is close to the new one
        # since fa changes slowly between brentq steps).  This typically
        # converges in 1–2 inner iterations instead of 3–10.
        #
        # IMPORTANT: the inner loop is kept (not eliminated) so that
        # _pulsed_residual remains a pure function of fa — no hidden state
        # leaks into the residual, which is required by brentq.
        _P_Ohm_warmstart = [0.0]

        def _pulsed_fixed_point(fa):
            """
            Fixed-point map: fa → new_fa, with warm-started inner P_Ohm loop.

            The inner loop converges P_Ohm self-consistently starting from
            the previous call's converged value.  Typically 1–2 iterations.

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
            """
            fa = max(1e-4, min(fa, _fa_max))

            # Inner P_Ohm loop: warm-started from previous converged value.
            # Fall back to 0 if the warmstart was corrupted by a NaN from an
            # unphysical scan point (e.g. fa near fa_max where DT dilution
            # kills the reaction and the chain returns NaN).
            P_Ohm_inner = _P_Ohm_warmstart[0] if np.isfinite(_P_Ohm_warmstart[0]) else 0.0
            for _inner in range(4):
                res = _evaluate_physics_chain(fa, P_Aux_fixed, P_Ohm_inner)
                P_Ohm_new = res['P_Ohm']

                # If the chain returned NaN, abort the inner loop early
                if not np.isfinite(P_Ohm_new):
                    break

                # Check inner convergence (absolute tolerance 10 kW)
                if abs(P_Ohm_new - P_Ohm_inner) < 0.01:
                    break
                # Damped update to avoid oscillation
                P_Ohm_inner = 0.5 * P_Ohm_inner + 0.5 * P_Ohm_new

            # Update warm-start for the next outer call — guard against NaN
            if np.isfinite(res['P_Ohm']):
                _P_Ohm_warmstart[0] = res['P_Ohm']

            # Cache the chain dict for post-convergence reuse
            _converged_chain[0] = res
            return res['new_f_alpha'], res['Q']

        def _pulsed_residual(fa):
            """Scalar residual r(fa) = new_fa - fa (absolute, not %)."""
            new_fa, _ = _pulsed_fixed_point(fa)
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
        # Strategy: evaluate points incrementally and stop as soon as a sign
        # change is detected.  This avoids wasting chain evaluations on the
        # full scan when the bracket is found early (typically at the 2nd–3rd
        # point for reactor-class tokamaks where fa ~ 0.02).
        _scan_pts = np.unique(np.concatenate([
            [1e-4],
            np.linspace(0.01, min(0.20, _fa_max), 10),
            [_fa_max],
        ]))

        bracket_found = False
        fa_lo = fa_hi = r_lo = r_hi = np.nan
        _prev_fa = None
        _prev_r  = None

        for _sfa in _scan_pts:
            _sr = _safe_residual(_sfa)
            if verbose >= 2:
                _sr_str = f"{_sr:+.3e}" if np.isfinite(_sr) else "NaN"
                print(f"  [bracket scan] fa={_sfa:.5f}  r={_sr_str}")

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

        try:
            fa_sol = brentq(_counted_residual, fa_lo, fa_hi,
                            xtol=1e-6, rtol=1e-8, maxiter=50)
        except (ValueError, RuntimeError) as _brentq_err:
            # brentq failed (bracket lost, NaN in residual, or convergence failure)
            if verbose >= 1:
                print(f"[SOLVER-Pulsed] brentq failed: {_brentq_err}")
            return np.nan, np.nan

        # Final evaluation to populate _converged_chain and get Q.
        # The inner P_Ohm loop in _pulsed_fixed_point already guarantees
        # self-consistency, so a single call is sufficient.
        _, Q_out = _pulsed_fixed_point(fa_sol)

        if verbose >= 1:
            print(f"[SOLVER-Pulsed] Brent converged in {_eval_count[0]} evaluations: "
                  f"f_alpha={fa_sol:.6f}, Q={Q_out:.2f}")

        return fa_sol, Q_out

    # =========================================================================
    #  STEADY-STATE SOLVER — 2-D on (f_alpha, Q) with scaled variables
    # =========================================================================

    def _solve_steady_state():
        """
        Solve the coupled (f_alpha, Q) system for Steady-State operation.

        In SS mode, P_aux = P_fus / Q, so Q is a genuine free variable.
        The 2-D system is:
          R1 = new_f_alpha(fa, Q) - fa     = 0
          R2 = Q_physics(fa, Q)   - Q      = 0

        Variables are scaled to ~O(1) for well-conditioned Jacobians:
          x = [fa / fa_scale, Q / Q_scale]

        Strategy:
          1. hybr with analytical Jacobian estimate
          2. Anderson mixing (2-D fixed-point)
          3. Picard with damped relaxation

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

            # Cache for post-convergence reuse
            _converged_chain[0] = res

            # Absolute residuals, normalised by the same scales
            R_fa = (res['new_f_alpha'] - fa) / fa_scale
            R_Q  = (res['Q'] - Q)            / Q_scale

            return np.array([R_fa, R_Q])

        def _ss_jacobian_scaled(x_scaled):
            """
            Approximate analytical Jacobian in scaled coordinates.

            The Jacobian is nearly diagonal because f_alpha and Q are weakly
            coupled (f_alpha affects Q only through fuel dilution, and Q
            affects f_alpha only through the slight change in P_aux → τ_E).

            J ≈ [[-1,   0 ],    (dominant: ∂R_fa/∂fa ≈ -1)
                 [ 0,  -1 ]]    (dominant: ∂R_Q/∂Q   ≈ -1)

            We refine the diagonal with finite differences at modest cost.
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

        # ── Step 1: scipy root with Jacobian ──────────────────────────────────
        _reset_dbg(5)
        if verbose >= 2:
            print("[SOLVER-SS] Step 1: Newton with analytical Jacobian estimate")

        initial_guesses_scaled = [
            np.array([0.05 / fa_scale, 50.0 / Q_scale]),
            np.array([0.05 / fa_scale, 20.0 / Q_scale]),
            np.array([0.05 / fa_scale, 100.0 / Q_scale]),
            np.array([0.10 / fa_scale, 50.0 / Q_scale]),
        ]

        best_sol = None
        best_rnorm = np.inf

        for guess in initial_guesses_scaled:
            _reset_dbg(3)
            try:
                result = root(_ss_residual_scaled, guess,
                              method='hybr', jac=_ss_jacobian_scaled,
                              tol=1e-8,
                              options={'maxfev': 300})
                fa_sol = result.x[0] * fa_scale
                Q_sol  = result.x[1] * Q_scale
                r_final = _ss_residual_scaled(result.x)
                rnorm = np.sqrt(r_final[0]**2 + r_final[1]**2)

                if (0 < fa_sol < _fa_max and Q_sol > 0 and rnorm < 1e-3):
                    if rnorm < best_rnorm:
                        best_sol   = (fa_sol, Q_sol)
                        best_rnorm = rnorm
            except Exception:
                continue

        if best_sol is not None:
            if verbose >= 1:
                print(f"[SOLVER-SS] Converged (hybr+jac): f_alpha={best_sol[0]:.5f}, "
                      f"Q={best_sol[1]:.2f}, |r|={best_rnorm:.2e}")
            return best_sol

        # ── Step 2: lm solver (no Jacobian needed) ────────────────────────────
        _reset_dbg(3)
        if verbose >= 2:
            print("[SOLVER-SS] Step 2: Levenberg-Marquardt")

        for guess in initial_guesses_scaled:
            _reset_dbg(2)
            try:
                result = root(_ss_residual_scaled, guess,
                              method='lm', tol=1e-8)
                fa_sol = result.x[0] * fa_scale
                Q_sol  = result.x[1] * Q_scale
                r_final = _ss_residual_scaled(result.x)
                rnorm = np.sqrt(r_final[0]**2 + r_final[1]**2)

                if (0 < fa_sol < _fa_max and Q_sol > 0 and rnorm < 1e-3):
                    if rnorm < best_rnorm:
                        best_sol   = (fa_sol, Q_sol)
                        best_rnorm = rnorm
            except Exception:
                continue

        if best_sol is not None:
            if verbose >= 1:
                print(f"[SOLVER-SS] Converged (lm): f_alpha={best_sol[0]:.5f}, "
                      f"Q={best_sol[1]:.2f}, |r|={best_rnorm:.2e}")
            return best_sol

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
                fa *= 0.8
                continue

            # Cache for post-convergence reuse
            _converged_chain[0] = res

            new_fa = res['new_f_alpha']
            new_Q  = res['Q']

            r_fa = new_fa - fa
            r_Q  = new_Q  - Q_fp

            if (k < 10 or k % 10 == 0) and verbose >= 2:
                print(f"  [Anderson-2D {k:3d}] fa={fa:.6f}, Q={Q_fp:.2f}  "
                      f"r_fa={r_fa:+.2e}, r_Q={r_Q:+.2e}")

            if abs(r_fa) < 1e-5 and abs(r_Q) < 0.01:
                if verbose >= 1:
                    print(f"[SOLVER-SS] Anderson-2D converged after {k+1} iter: "
                          f"f_alpha={new_fa:.6f}, Q={new_Q:.2f}")
                return new_fa, new_Q

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

        # Accept approximate solution if residuals are small enough
        if abs(r_fa) < 1e-3 and abs(r_Q) < 1.0:
            if verbose >= 1:
                print(f"[SOLVER-SS] Accepting approximate solution: "
                      f"f_alpha={fa:.6f}, Q={Q_fp:.2f}")
            return fa, Q_fp

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
                "Physics chain is broken at the initial guess. "
                "Fix the error above before running D0FUS."
            ) from _pre_exc

        f_alpha_solution, Q_solution = _solve_pulsed()

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
                "Physics chain is broken at the initial guess. "
                "Fix the error above before running D0FUS."
            ) from _pre_exc

        f_alpha_solution, Q_solution = _solve_steady_state()

    else:
        raise ValueError(
            f"Unknown Operation_mode: '{Operation_mode}'. "
            "Valid options: 'Pulsed', 'Steady-State'."
        )

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
    B_pol_solution  = f_Bpol(q95_solution, B0_solution, a, R0, kappa=κ)
    betaT_solution  = f_beta_T(pbar_solution, B0_solution)
    betaP_solution  = f_beta_P(a, κ, pbar_solution, Ip_solution)
    beta_solution   = f_beta(betaP_solution, betaT_solution)
    betaN_solution  = f_beta_N(beta_solution, a, B0_solution, Ip_solution)

    # Fast-alpha pressure contribution (Stix slowing-down model)
    beta_fast_alpha, tau_sd_alpha, W_fast_alpha = f_beta_fast_alpha(
        P_Alpha, Tbar, nbar_solution, B0_solution, Volume_solution, Z_eff=Zeff)
    # Total beta including fast alphas (for MHD stability comparison)
    betaT_total     = betaT_solution + beta_fast_alpha
    betaP_total     = betaP_solution + beta_fast_alpha * (B0_solution / B_pol_solution)**2
    beta_total      = f_beta(betaP_total, betaT_total)
    betaN_total     = f_beta_N(beta_total, a, B0_solution, Ip_solution)
    # Greenwald limit is defined in terms of line-averaged density;
    # f_nG returns n_G [1e20 m-3], which must be compared with nbar_line, not nbar_vol.
    nG_solution     = f_nG(Ip_solution, a) * Greenwald_limit
    nG_raw          = f_nG(Ip_solution, a)  # raw Ip/(πa²), no limit factor

    # Current drive and power balance
    # ── Individual CD efficiencies (computed once, reused in breakdown) ────────
    eta_LH_solution  = f_etaCD_LH_physics(Tbar, Zeff)
    eta_EC_solution  = f_etaCD_EC_physics(a, R0, Tbar, nbar_solution, Zeff, nu_T, nu_n,
                                   rho_EC,
                                   theta_EC_pol_deg=config.theta_EC_pol_deg,
                                   rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                   T_ped_frac=T_ped_frac)
    eta_NBI_solution = f_etaCD_NBI_physics(
        A_beam, E_beam_keV,
        a, R0, Tbar, nbar_solution, Zeff, nu_T, nu_n,
        rho_NBI,
        f_alpha=f_alpha_solution,
        angle_NBI_deg=config.angle_NBI_deg,
        rho_ped=rho_ped, n_ped_frac=n_ped_frac,
        T_ped_frac=T_ped_frac)

    if Operation_mode == 'Steady-State':
        # Steady-State: γ_eff is needed to invert I_CD → P_CD.
        eta_CD_solution = f_etaCD_effective(
            config, a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T, Zeff,
            rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
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
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
            I_CD_solution = f_I_CD(R0, nbar_solution, eta_CD_solution, P_CD_solution)
        I_Ohm_solution = f_I_Ohm(Ip_solution, Ib_solution, I_CD_solution)
        P_Ohm_solution = f_P_Ohm(I_Ohm_solution, Tbar, R0, a, κ, Z_eff=Zeff,
                                  nbar=nbar_solution, eta_model=eta_model,
                                  q95=q95_solution)
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
    if Operation_mode == 'Pulsed' and CD_source == 'Multi':
        # Pulsed Multi: powers are user-specified inputs, not derived from fractions
        P_LH_solution  = P_LH
        P_EC_solution  = P_ECRH
        P_NBI_solution = P_NBI
        P_ICR_solution = P_ICRH
        denom_cd = R0 * nbar_solution
        I_LH_solution  = eta_LH_solution  * P_LH   / denom_cd if denom_cd > 0 else 0.0
        I_EC_solution  = eta_EC_solution  * P_ECRH  / denom_cd if denom_cd > 0 else 0.0
        I_NBI_solution = eta_NBI_solution * P_NBI   / denom_cd if denom_cd > 0 else 0.0
    else:
        # Single-source or Steady-State: fraction-based breakdown is appropriate
        cd_bd = f_CD_breakdown(config, P_CD_solution, R0, nbar_solution,
                               eta_LH_solution, eta_EC_solution, eta_NBI_solution)
        P_LH_solution  = cd_bd['P_LH']
        P_EC_solution  = cd_bd['P_EC']
        P_NBI_solution = cd_bd['P_NBI']
        P_ICR_solution = cd_bd['P_ICR']
        I_LH_solution  = cd_bd['I_LH']
        I_EC_solution  = cd_bd['I_EC']
        I_NBI_solution = cd_bd['I_NBI']

    # Divertor power: P_α + P_CD − P_rad_total
    # P_rad_total (not P_rad_core) is subtracted because edge/SOL radiation
    # also reduces the power reaching the divertor target plates.
    # Note: this is P_div, not the true P_sep (which uses P_rad_core).
    # The variable name P_sep_solution is kept for backward compatibility
    # with the genetic algorithm, scan module and figures module.
    P_sep_solution      = f_P_sep(P_fus, P_CD_solution, P_rad_total_solution)
    Gamma_n_solution    = f_Gamma_n(a, P_fus, R0, κ, S_wall=Surface_solution)
    heat_refined_solution = f_heat_refined(R0, P_sep_solution)
    heat_par_solution   = f_heat_par(R0, B0_solution, P_sep_solution)
    heat_pol_solution   = f_heat_pol(R0, B0_solution, P_sep_solution, a, q95_solution)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(
        P_sep_solution, B_pol_solution, R0, a / R0, theta_deg)
    # First-wall load: radiative power distributed over the plasma surface
    P_1rst_wall_rad    = f_P_wall(P_rad_total_solution, Surface_solution)
    # Exhaust power density: P_div / S (useful divertor figure of merit)
    P_1rst_wall_div   = P_sep_solution / Surface_solution if Surface_solution > 0 else 0.0
    P_wallplug_solution = P_CD_solution / eta_WP   # Wall-plug power consumed by heating/CD [MW]
    P_elec_solution   = f_P_elec(P_fus, P_CD_solution, eta_T, M_blanket, eta_WP)

    # ── Loop voltage ──────────────────────────────────────────────────────────
    # V_loop is now computed inside Magnetic_flux (returned as 5th element)
    # to avoid redundant neoclassical conductance integration.

    # ── q,j profile on the converged global quantities ───────────────────
    # The dispatcher routes to f_q_profile_academic or f_q_profile_refined
    # according to config.q_profile_mode.  In academic mode this is a single
    # deterministic pass on the parametric ansatz; in refined mode this is a
    # final Picard pass at moderate resolution that returns li, q0, q(rho_95)
    # and the full j-decomposition consistent with the converged Ip, I_CD.

    # CD deposition (rho_CD, delta_CD) selection.  These values are passed
    # to the dispatcher and used only by the refined branch to build the
    # j_CD(rho) Gaussian.  Academic q-profile mode ignores them (j_CD=0).
    if CD_source == 'Multi' and I_CD_solution > 0:
        _I_sources = np.array([abs(I_LH_solution), abs(I_EC_solution), abs(I_NBI_solution)])
        _rho_sources = np.array([0.5, rho_EC, rho_NBI])
        _delta_sources = np.array([0.20, 0.05, 0.15])
        _w = _I_sources / max(np.sum(_I_sources), 1e-10)
        _rho_CD_eff  = float(np.dot(_w, _rho_sources))
        _delta_CD_eff = float(np.dot(_w, _delta_sources))
    elif CD_source == 'ECCD':
        _rho_CD_eff, _delta_CD_eff = rho_EC, 0.05
    elif CD_source == 'NBCD':
        _rho_CD_eff, _delta_CD_eff = rho_NBI, 0.15
    else:  # LHCD, Academic, or unknown — broad central deposition fallback
        _rho_CD_eff, _delta_CD_eff = 0.5, 0.20

    # Post-convergence pass at moderate resolution (n_rho=80, max_iter=20).
    # The solver cache already provides a near-converged starting point in
    # refined mode, so full defaults (n_rho=200, max_iter=50) are
    # unnecessarily expensive here.  The quantities extracted (li, q0, j
    # profiles) converge at < 0.5 % error between n_rho=80 and n_rho=200 for
    # smooth tokamak current profiles.  In academic mode max_iter and tol
    # are ignored (deterministic single pass).
    _q_sc = _solve_q_profile(
        Ip_solution, I_CD_solution, q95_solution,
        B0_solution, nbar_solution,
        rho_CD_loc=_rho_CD_eff, delta_CD_loc=_delta_CD_eff,
        n_rho=80, max_iter=20, tol=1e-3, damping=0.5,
        q_init=_q_profile_cache[0])
    li_solution      = _q_sc['li']

    # L-H power threshold — all Martin/Delabie scalings were fitted with line-averaged density
    if L_H_Scaling_choice == 'Martin':
        P_Thresh = P_Thresh_Martin(nbar_line_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_S':
        P_Thresh = P_Thresh_New_S(nbar_line_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_Ip':
        P_Thresh = P_Thresh_New_Ip(nbar_line_solution, B0_solution, a, R0, κ, Ip_solution, Atomic_mass)
    else:
        if verbose >= 1:
            print('Choose a valid scaling for the L-H transition')

    # ==============================================================================
    #    TOROIDAL FIELD (TF) COIL RADIAL BUILD
    #    Determines the inboard thickness 'c' from mechanical and magnetic constraints.
    # ==============================================================================
    if Radial_build_model == "academic":
        (c, c_WP_TF, c_Nose_TF,
         σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_academic(
            a, b, R0, σ_TF, J_max_TF_conducteur, Bmax_TF, Choice_Buck_Wedg,
            coef_inboard_tension, F_CClamp)

    elif Radial_build_model in ("refined", "CIRCE"):
        (c, c_WP_TF, c_Nose_TF,
         σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_refined(
            a, b, R0, σ_TF, J_max_TF_conducteur, Bmax_TF, Choice_Buck_Wedg, omega_TF, n_shape_TF,
            c_BP, coef_inboard_tension, F_CClamp, TF_grading)

    else:
        raise ValueError(
            f"Unknown radial build model: '{Radial_build_model}'. "
            "Valid options: 'academic', 'refined', 'CIRCE'."
        )

    # ── Return nan tuple if TF radial build failed ────────────────────────────
    _nan = np.nan
    if not np.isfinite(c):
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
        )

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
    Vprime_data=Vprime_data,
    )
    # Non-inductive current drive during ramp-up reduces the CS volt-second
    # requirement proportionally: both the inductive (Ψ_ind ∝ Ip) and resistive
    # (Ψ_res = Ce μ0 R0 Ip) terms scale with the inductively-ramped current fraction.
    ΨRampUp *= (1.0 - getattr(config, 'f_heat_ramp', 0.0))
    # ==============================================================================
    #    CENTRAL SOLENOID (CS) DESIGN
    #    Determines the CS radial thickness 'd' to satisfy the Volt-second budget.
    # ==============================================================================
    cs_args = (ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax_TF, Bmax_CS_adm, σ_CS,
           Supra_choice, J_wost_Manual, T_helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
           config)

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
    # We call calculate_cable_current_density with B_CS to recover them.
    # The LRU cache makes this essentially free (same call was made inside solver).

    if np.isfinite(B_CS) and B_CS > 0:
        Supra_choice_CS = config.Supra_choice   # Same SC for TF and CS
        E_mag_CS_post = calculate_E_mag_CS(
            B_CS,
            R0 - a - b - c - _gap_eff - d,   # CS inner bore [m]
            R0 - a - b - c - _gap_eff,         # CS outer face [m]
            2 * (κ * a + b + 1))
        result_J_CS = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS, T_op=T_helium,
            E_mag=E_mag_CS_post,
            I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS, tau_h=tau_h,
            f_He_pipe=f_He_pipe, f_void=f_void, f_In=f_In,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He,
            Marge_T_Nb3Sn=Marge_T_Nb3Sn, Marge_T_NbTi=Marge_T_NbTi,
            Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=J_wost_Manual if Supra_choice == 'Manual' else None)
        f_sc_CS      = result_J_CS['f_sc']
        f_cu_CS      = result_J_CS['f_cu']
        f_He_pipe_CS = result_J_CS['f_He_pipe']
        f_void_CS    = result_J_CS['f_void']
        f_He_CS      = result_J_CS['f_He']
        f_In_CS      = result_J_CS['f_In']
    else:
        f_sc_CS = f_cu_CS = f_He_pipe_CS = f_void_CS = f_He_CS = f_In_CS = np.nan

    # ── TF Princeton-D cross-section geometry ─────────────────────────────────
    (R_bore_TF, _R_TF_out, _H_TF_D, _A_cross_TF, L_turn_TF,
     _R_out, _Z_out, _R_in, _Z_in) = f_TF_cross_section(a, b, R0, c, Delta_TF)

    # Component volumes — only V_FI used; Princeton-D H_TF for the FI cylinder
    (_, _, _, V_FI) = f_volume(a, b, c, d, R0, κ, Delta_TF, _H_TF_D)

    # ── Single-coil TF volume (inline: reuse A_cross and R_bore already computed) ──
    # f_V_TF(a,b,R0,c,N_TF,Delta_TF) = A_cross * 2π * R_bore / N_TF but it would
    # call f_TF_cross_section a second time (ODE solve). Use pre-computed values.
    V_TF_one  = _A_cross_TF * 2.0 * np.pi * R_bore_TF / float(N_TF)
    V_CS_geom = f_V_CS(a, b, c, d, R0, κ, Gap, Choice_Buck_Wedg)

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
    )
    _BB_ROLES     = ('breeder', 'structure', 'multiplier')
    _SHIELD_ROLES = ('shield_HT', 'shield_LT')

    V_rb_SOL      = sum(L['V']     for L in _rb['layers'] if L['role'] == 'SOL')
    V_rb_FW       = sum(L['V_eff'] for L in _rb['layers'] if L['role'] == 'FW')
    V_rb_BB       = sum(L['V_eff'] for L in _rb['layers'] if L['role'] in _BB_ROLES)
    V_rb_shield   = sum(L['V']     for L in _rb['layers'] if L['role'] in _SHIELD_ROLES)
    V_rb_VV       = sum(L['V']     for L in _rb['layers'] if L['role'] == 'VV')
    V_rb_gap_TF   = sum(L['V']     for L in _rb['layers'] if L['role'] == 'gap')
    V_rb_divertor = _rb['V_divertor']

    # ── Radial build layer masses ─────────────────────────────────────────────
    M_rb_FW       = sum(L['M'] for L in _rb['layers'] if L['role'] == 'FW')
    M_rb_BB       = sum(L['M'] for L in _rb['layers'] if L['role'] in _BB_ROLES)
    M_rb_shield   = sum(L['M'] for L in _rb['layers'] if L['role'] in _SHIELD_ROLES)
    M_rb_VV       = sum(L['M'] for L in _rb['layers'] if L['role'] == 'VV')
    M_rb_divertor = V_rb_divertor * config.rho_divertor
    M_rb_total    = M_rb_FW + M_rb_BB + M_rb_shield + M_rb_VV + M_rb_divertor

    cost_solution = (V_rb_BB + V_TF_one * N_TF + V_CS_geom) / P_fus   # legacy cost proxy [m^3/MW]

    # ── Material volumes (TF: total = V_TF_one × N_TF; CS: full solenoid) ─────
    (V_steel_TF, V_sc_TF, V_cu_TF, V_He_TF, V_In_TF,
     V_steel_CS, V_sc_CS, V_cu_CS, V_He_CS, V_In_CS) = f_coil_material_volumes(
        V_TF_one, int(N_TF),
        Steel_fraction_TF,
        f_sc_TF, f_cu_TF, f_He_TF, f_In_TF,
        V_CS_geom,
        Steel_fraction_CS,
        f_sc_CS, f_cu_CS, f_He_CS, f_In_CS)

    # ── Cable conductor lengths ───────────────────────────────────────────────
    R_CS_ext_val = R0 - a - b - c - _gap_eff
    L_turn_CS    = np.pi * (R_CS_ext_val + R_CS_ext_val - d)   # 2π × R_mean
    _R_TF_in     = R0 - a - b   # plasma-facing inner face, consistent with D0FUS NI convention
    (L_cable_TF, L_cable_CS,
     _Nturns_TF, _Nturns_CS) = f_cable_length(
        int(N_TF), L_turn_TF, Bmax_TF, _R_TF_in,
        I_cond,
        N_sub_CS, L_turn_CS, B_CS, H_TF)   # H_TF = 2(κa+b+1) = H_CS_NI

    # ── Material masses ───────────────────────────────────────────────────────
    (M_steel_TF, M_sc_TF, M_cu_TF, M_In_TF, M_total_TF,
     M_steel_CS, M_sc_CS, M_cu_CS, M_In_CS, M_total_CS) = f_coil_masses(
        V_steel_TF, V_sc_TF, V_cu_TF, V_In_TF,
        V_steel_CS, V_sc_CS, V_cu_CS, V_In_CS,
        Chosen_Steel, Supra_choice)

    # ── SC strand counts and total SC strand lengths ───────────────────────────
    (n_sc_TF, n_sc_CS,
     L_sc_strand_TF, L_sc_strand_CS) = f_cable_strands(
        V_sc_TF, L_cable_TF,
        V_sc_CS, L_cable_CS,
        Supra_choice)

    # ── Component lifetimes and plant availability ────────────────────────────
    A_div          = config.f_div_area_fraction * Surface_solution
    A_blanket      = Surface_solution * (1.0 - config.f_div_area_fraction)
    t_life_bl_fpy  = f_blanket_lifetime_fpy(P_fus, A_blanket,
                                             config.dpa_lim, config.C_dpa)
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
            M_rb_VV,      M_rb_divertor, M_rb_total)              # 146, 147, 148


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

    # ── Profile peaking / pedestal parameters ─────────────────────────────────
    # Use the module-level _PROFILE_PRESETS table (single source of truth).
    if config.Plasma_profiles == 'Manual':
        nu_n       = config.nu_n_manual
        nu_T       = config.nu_T_manual
        rho_ped    = config.rho_ped
        n_ped_frac = config.n_ped_frac
        T_ped_frac = config.T_ped_frac
    else:
        _p         = _PROFILE_PRESETS.get(config.Plasma_profiles, _PROFILE_PRESETS['H'])
        nu_n       = _p['nu_n'];       nu_T       = _p['nu_T']
        rho_ped    = _p['rho_ped'];    n_ped_frac = _p['n_ped_frac']
        T_ped_frac = _p['T_ped_frac']

    # ── Vprime_data (Miller geometry pre-computation, if applicable) ──────────
    Vprime_data = None
    if (config.Plasma_geometry == 'refined'
            and np.isfinite(float(kappa_edge))
            and np.isfinite(float(delta_edge))):
        try:
            Vprime_data = precompute_Vprime(
                config.R0, config.a,
                float(kappa_edge), float(delta_edge),
                geometry_model='refined',
                kappa_95=float(kappa_95),
                delta_95=float(delta_95),
            )
        except Exception:
            Vprime_data = None   # Non-critical — figures fall back gracefully

    # ── Impurity fractions: map species list to f_W / f_Ne / f_Ar keys ───────
    _sp_raw  = config.impurity_species
    _fc_raw  = config.f_imp_core
    _imp_out = {'W': None, 'Ne': None, 'Ar': None}
    if _sp_raw and str(_sp_raw).strip() not in ('', 'None', 'none'):
        _sp_list = [s.strip() for s in str(_sp_raw).split(',') if s.strip()]
        if isinstance(_fc_raw, (int, float)):
            _fc_list = [float(_fc_raw)]
        else:
            _fc_list = [float(c.strip()) for c in str(_fc_raw).split(',') if c.strip()]
        for _sp, _fc in zip(_sp_list, _fc_list):
            if _sp in _imp_out and _fc > 0:
                _imp_out[_sp] = _fc

    # ── TF coil count: recompute to stay consistent with run() ───────────────
    try:
        _N_TF, _ripple, Delta_TF = Number_TF_coils(
            config.R0, config.a, config.b,
            config.ripple_adm, config.L_min,
        )
        N_TF     = int(_N_TF)
        # Delta_TF: extra outboard radial clearance required by port-access
        # constraint [m]. Forwarded to _resolve_build() so that R_TF_out
        # is consistent with the ripple model: r2 = R0 + a + b + Delta_TF.
        Delta_TF = float(Delta_TF)
    except Exception:
        N_TF     = 16                 # Conservative default: 16 TF coils
        Delta_TF = 0.0                # Conservative default: no extra clearance

    # ── Safe float extraction (guard against NaN in radial build) ─────────────
    def _f(x, fallback):
        try:
            v = float(x)
            return v if np.isfinite(v) else fallback
        except (TypeError, ValueError):
            return fallback

    return {
        # Plasma geometry
        "R0":              config.R0,
        "a":               config.a,
        "Plasma_geometry": config.Plasma_geometry,
        "kappa_edge":      _f(kappa_edge,  1.70),
        "delta_edge":      _f(delta_edge,  0.33),
        "kappa_95":        _f(kappa_95,    1.60),
        "delta_95":        _f(delta_95,    0.25),
        "Vprime_data":     Vprime_data,
        # q,j profile philosophy (consumed by D0FUS_figures.plot_q_profile)
        "q_profile_mode":  config.q_profile_mode,
        "alpha_J":         config.alpha_J,
        # On-axis field and plasma current
        "B0":          _f(B0, config.Bmax_TF),
        "B_max":       config.Bmax_TF,
        "Ip":          _f(Ip, 15.0),
        # Volume-averaged kinetics
        "nbar":        _f(nbar, 1.0),
        "Tbar":        config.Tbar,
        "nu_n":        nu_n,
        "nu_T":        nu_T,
        "rho_ped":     rho_ped,
        "n_ped_frac":  n_ped_frac,
        "T_ped_frac":  T_ped_frac,
        # MHD
        "q95":         _f(q95, 3.0),
        "Z_eff":       config.Zeff,
        # Power balance
        "P_fus":       config.P_fus,
        "P_aux":       config.P_aux_input,
        "Q":           _f(Q, float('nan')),
        # Impurity seeding
        "f_W":         _imp_out["W"],
        "f_Ne":        _imp_out["Ne"],
        "f_Ar":        _imp_out["Ar"],
        # Radial build — inboard side [m]
        "b":           config.b,
        "c_TF":        _f(c_TF,      0.56),
        "c_WP":        _f(c_WP_TF,   0.36),
        "c_nose":      _f(c_Nose_TF, 0.20),
        "c_CS":        _f(c_CS,      0.70),
        "N_TF":            N_TF,
        "Delta_TF":        Delta_TF,   # Extra outboard radial clearance from port-access constraint [m]
        "Gap":             config.Gap,
        # ── Radial build sublayer widths ──────────────────────────────────────
        "f_kappa_SOL": config.f_kappa_SOL,
        "Blanket_choice":   config.Blanket_choice,
        # Mechanical configuration key — consumed by _resolve_build() in figures
        # to decide whether the TF-CS gap is applied to the CS outer radius.
        "Choice_Buck_Wedg": config.Choice_Buck_Wedg,
        # First wall / blanket / shield (optional; ITER-like defaults)
        "e_fw":        getattr(config, 'e_fw',      0.05),
        "e_blanket":   getattr(config, 'e_blanket', 0.45),
        "e_shield":    getattr(config, 'e_shield',  0.50),
        # Conductor cable fractions (wost = without steel) — TF
        # n_shape_TF, n_shape_CS: steel asymmetry parameter δ_S1/δ_S2 (1 = square jacket)
        "n_shape_TF":  config.n_shape_TF,
        "n_shape_CS":  config.n_shape_CS,
        "Supra_choice": config.Supra_choice,
        "Steel_fraction_TF": _f(_sf_TF, 0.50),
        "f_sc_TF":      _f(_f_sc_TF, np.nan),
        "f_cu_TF":      _f(_f_cu_TF, np.nan),
        "f_He_pipe_TF": _f(_f_He_pipe_TF, np.nan),
        "f_void_TF":    _f(_f_void_TF, np.nan),
        "f_He_TF":      _f(_f_He_TF, np.nan),
        "f_In_TF":      _f(_f_In_TF, np.nan),
        # Conductor cable fractions (wost = without steel) — CS
        "Steel_fraction_CS": _f(_sf_CS, 0.50),
        "f_sc_CS":      _f(_f_sc_CS, np.nan),
        "f_cu_CS":      _f(_f_cu_CS, np.nan),
        "f_He_pipe_CS": _f(_f_He_pipe_CS, np.nan),
        "f_void_CS":    _f(_f_void_CS, np.nan),
        "f_He_CS":      _f(_f_He_CS, np.nan),
        "f_In_CS":      _f(_f_In_CS, np.nan),
        # ── Coil volumes and cable inventory ─────────────────────────────────
        "V_TF_one":        _f(_V_TF_one,       np.nan),   # Single TF coil [m³]
        "V_CS_geom":       _f(_V_CS_geom,      np.nan),   # Total CS solenoid [m³]
        "V_blanket":       _f(_V_blanket,      np.nan),   # Blanket torus [m³]
        "V_steel_TF":      _f(_V_steel_TF,     np.nan),
        "V_sc_TF":         _f(_V_sc_TF,        np.nan),
        "V_cu_TF":         _f(_V_cu_TF,        np.nan),
        "V_He_TF":         _f(_V_He_TF,        np.nan),
        "V_In_TF":         _f(_V_In_TF,        np.nan),
        "V_steel_CS":      _f(_V_steel_CS,     np.nan),
        "V_sc_CS":         _f(_V_sc_CS,        np.nan),
        "V_cu_CS":         _f(_V_cu_CS,        np.nan),
        "V_He_CS":         _f(_V_He_CS,        np.nan),
        "V_In_CS":         _f(_V_In_CS,        np.nan),
        "L_cable_TF":      _f(_L_cable_TF,     np.nan),
        "L_cable_CS":      _f(_L_cable_CS,     np.nan),
        "n_sc_TF":         _f(_n_sc_TF,        np.nan),
        "n_sc_CS":         _f(_n_sc_CS,        np.nan),
        "L_sc_strand_TF":  _f(_L_sc_strand_TF, np.nan),
        "L_sc_strand_CS":  _f(_L_sc_strand_CS, np.nan),
        # ── Coil masses ─────────────────────────────────────────────────────
        "M_steel_TF":  _f(_M_steel_TF,  np.nan),
        "M_sc_TF":     _f(_M_sc_TF,     np.nan),
        "M_cu_TF":     _f(_M_cu_TF,     np.nan),
        "M_In_TF":     _f(_M_In_TF,     np.nan),
        "M_total_TF":  _f(_M_total_TF,  np.nan),
        "M_steel_CS":  _f(_M_steel_CS,  np.nan),
        "M_sc_CS":     _f(_M_sc_CS,     np.nan),
        "M_cu_CS":     _f(_M_cu_CS,     np.nan),
        "M_In_CS":     _f(_M_In_CS,     np.nan),
        "M_total_CS":  _f(_M_total_CS,  np.nan),
        # ── Radial build component masses ────────────────────────────────────
        "M_rb_FW":       _f(M_rb_FW,       np.nan),
        "M_rb_BB":       _f(M_rb_BB,       np.nan),
        "M_rb_shield":   _f(M_rb_shield,   np.nan),
        "M_rb_VV":       _f(M_rb_VV,       np.nan),
        "M_rb_divertor": _f(M_rb_divertor, np.nan),
        "M_rb_total":    _f(M_rb_total,    np.nan),
    }


def save_run_output(config: GlobalConfig,
                    results: tuple,
                    output_dir: str,
                    input_file_path: str = None,
                    save_figures: bool = False,
                    verbose: int = 0) -> str:
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

    Returns
    -------
    str : absolute path to the created output directory.
    """
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, 'run', f"Run_D0FUS_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

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
     M_rb_VV, M_rb_divertor, M_rb_total) = results

    # ── Recompute N_TF for display (not stored in results tuple) ──────────
    try:
        _N_TF_disp, _ripple_disp, _Delta_TF_disp = Number_TF_coils(
            config.R0, config.a, config.b,
            config.ripple_adm, config.L_min)
        N_TF_disp    = int(_N_TF_disp)
        Delta_TF_disp = float(_Delta_TF_disp)
    except Exception:
        N_TF_disp    = 16
        Delta_TF_disp = 0.0

    # ── Blanket concept: TBR and radial-build layer breakdown (display only) ──
    _, _, _, _, _, _, _, _R_in_disp, _Z_in_disp = f_TF_cross_section(
        config.a, config.b, config.R0, c, Delta_TF_disp)
    _rb_disp = f_radial_build_layers(
        a=config.a, b=config.b, κ=κ, δ=δ, R0=config.R0,
        Delta_TF=Delta_TF_disp,
        f_kappa_SOL=config.f_kappa_SOL,
        f_div_area_fraction=config.f_div_area_fraction,
        R_tf_in=_R_in_disp, Z_tf_in=_Z_in_disp,
        Blanket_choice=config.Blanket_choice,
    )
    _delta_BB_ib = _rb_disp['delta_BB_ib']
    _delta_BB_ob = _rb_disp['delta_BB_ob']
    _blanket_concept = material_blanket(config.Blanket_choice)
    _RB_layers        = _rb_disp['layers']
    _TBR_achieved     = f_TBR(_delta_BB_ib, config.Blanket_choice)
    _M_blanket        = M_blanket_effective(config.Blanket_choice)
    _eta_T            = eta_T_effective(config.Blanket_choice)

    # ── Magnetic stored energy and ampere-turns (display only) ────────────
    # TF bore geometry (same convention as run())
    _r_in_TF  = config.R0 - config.a - config.b
    _r_out_TF = config.R0 + config.a + config.b
    _H_TF     = 2 * (κ * config.a + config.b + 1)
    E_mag_TF_disp = calculate_E_mag_TF(config.Bmax_TF, _r_in_TF, _r_out_TF, _H_TF)
    # Ampere's law on toroidal path at inner bore: NI_total = B_max × 2π r_in / μ₀
    NI_TF_total = config.Bmax_TF * 2 * np.pi * _r_in_TF / μ0
    # Per-coil ampere-turns and winding turns
    NI_TF_coil     = NI_TF_total / N_TF_disp if N_TF_disp > 0 else np.nan
    Nturns_TF_coil = NI_TF_coil / config.I_cond if config.I_cond > 0 else np.nan

    # CS geometry from converged radial build
    _H_CS = _H_TF                       # Same vertical extent as TF
    if np.isfinite(B_CS) and np.isfinite(r_d) and np.isfinite(r_c) and r_c > r_d:
        E_mag_CS_disp = calculate_E_mag_CS(B_CS, r_d, r_c, _H_CS)
        # Ampere's law on axial path through bore: NI_total = B_max × H / μ₀
        NI_CS_total = B_CS * _H_CS / μ0
        # Per-module ampere-turns and winding turns
        _N_mod_CS = config.N_sub_CS if config.N_sub_CS > 0 else 1
        NI_CS_mod     = NI_CS_total / _N_mod_CS
        Nturns_CS_mod = NI_CS_mod / config.I_cond if config.I_cond > 0 else np.nan
    else:
        E_mag_CS_disp = np.nan
        NI_CS_total   = np.nan
        NI_CS_mod     = np.nan
        Nturns_CS_mod = np.nan

    # Re-parse multi-impurity config for per-species display
    _Z_CORONAL = {'Be': 4, 'C': 6, 'N': 7, 'Ne': 10, 'Ar': 18, 'Kr': 34, 'Xe': 44, 'W': 46}
    _sp_raw = config.impurity_species
    _fc_raw = config.f_imp_core
    if _sp_raw is None or str(_sp_raw).strip() in ('', 'None', 'none'):
        imp_species_list, imp_conc_list = [], []
    else:
        imp_species_list = [s.strip() for s in str(_sp_raw).split(',') if s.strip()]
        if isinstance(_fc_raw, (int, float)):
            imp_conc_list = [float(_fc_raw)]
        else:
            imp_conc_list = [float(c.strip()) for c in str(_fc_raw).split(',') if c.strip()]
        paired = [(s, c) for s, c in zip(imp_species_list, imp_conc_list) if c > 0]
        imp_species_list = [p[0] for p in paired]
        imp_conc_list    = [p[1] for p in paired]
    f_imp_dilution = sum(_Z_CORONAL.get(s, 10) * c
                         for s, c in zip(imp_species_list, imp_conc_list))

    # Reconstruct profile params and Vprime_data for per-species display.
    # Use the module-level _PROFILE_PRESETS table (single source of truth).
    if config.Plasma_profiles == 'Manual':
        nu_n = config.nu_n_manual; nu_T = config.nu_T_manual
        rho_ped = config.rho_ped; n_ped_frac = config.n_ped_frac; T_ped_frac = config.T_ped_frac
    elif config.Plasma_profiles in _PROFILE_PRESETS:
        p = _PROFILE_PRESETS[config.Plasma_profiles]
        nu_n = p['nu_n']; nu_T = p['nu_T']; rho_ped = p['rho_ped']
        n_ped_frac = p['n_ped_frac']; T_ped_frac = p['T_ped_frac']
    else:
        nu_n = 0; nu_T = 0; rho_ped = 1.0; n_ped_frac = 0; T_ped_frac = 0
    Vprime_data = None
    if config.Plasma_geometry == 'refined' and np.isfinite(κ) and np.isfinite(δ):
        Vprime_data = precompute_Vprime(config.R0, config.a, κ, δ,
                                         geometry_model='refined',
                                         kappa_95=κ_95, delta_95=δ_95)
    rho_rad_core = config.rho_rad_core

    # ── Write results report (console + file) ─────────────────────────────────
    output_file = os.path.join(output_path, "output_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:

        class DualWriter:
            """Writes simultaneously to stdout and a file."""
            def __init__(self, *files):
                self._files = files
            def write(self, data):
                for ff in self._files:
                    ff.write(data)
            def flush(self):
                for ff in self._files:
                    ff.flush()

        out = DualWriter(sys.stdout, f)

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
        print(f"[I] b  (BB & neutron shield thickness)              : {r_minor-r_sep:.3f} [m]", file=out)
        # c and d are read directly from results to avoid any gap-dependent ambiguity
        # when interpreting differences of the radii r_c / r_d.
        print(f"[O] c  (TF inboard leg thickness)                   : {c:.3f} [m]",   file=out)
        if config.Choice_Buck_Wedg == 'Wedging':
            print(f"[O] Gap (TF inner bore -> CS outer face)            : {config.Gap:.3f} [m]", file=out)
        print(f"[O] d  (CS winding-pack radial thickness)           : {d:.3f} [m]",     file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Kappa    (Plasma elongation)                    : {κ:.3f}",     file=out)
        print(f"[O] Kappa_95 (Elongation at 95% flux surface)       : {κ_95:.3f}", file=out)
        print(f"[O] Delta    (Plasma triangularity)                 : {δ:.3f}",     file=out)
        print(f"[O] Delta_95 (Triangularity at 95% flux surface)    : {δ_95:.3f}", file=out)
        print(f"[O] Volume   (Plasma volume)                        : {Volume:.3f} [m³]",   file=out)
        print(f"[O] Surface  (First wall area)                      : {Surface:.3f} [m²]",  file=out)
        print(f"[I] Mechanical configuration                        : {config.Choice_Buck_Wedg}", file=out)
        print(f"[I] Superconductor technology                       : {config.Supra_choice}",    file=out)
        print(f"[O] N_TF   (Number of TF coils)                     : {N_TF_disp}",              file=out)
        print(f"[O] Delta_TF (Port-access outboard clearance)       : {Delta_TF_disp:.3f} [m]", file=out)
        print(f"[I] N_CS   (Number of CS modules)                   : {config.N_sub_CS}",         file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] Bmax_TF (Peak field on TF conductor)            : {config.Bmax_TF:.3f} [T]",   file=out)
        print(f"[O] B0   (On-axis magnetic field)                   : {B0:.3f} [T]",             file=out)
        print(f"[O] BCS  (CS peak magnetic field)                   : {B_CS:.3f} [T]",           file=out)
        print(f"[O] J_E-TF (TF engineering current density)         : {J_TF/1e6:.3f} [MA/m²]",  file=out)
        print(f"[O] J_E-CS (CS engineering current density)         : {J_CS/1e6:.3f} [MA/m²]",  file=out)
        print(f"[O] E_mag_TF (TF stored magnetic energy)            : {E_mag_TF_disp/1e9:.3f} [GJ]", file=out)
        print(f"[O] E_mag_CS (CS stored magnetic energy)            : {E_mag_CS_disp/1e9:.3f} [GJ]", file=out)
        print(f"[I] I_cond (Operating current per conductor)        : {config.I_cond/1e3:.1f} [kA]", file=out)
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
        print(f"[O]  └ f_In      (insulation)                       : {f_In_TF*100:.2f} [%]", file=out)
        print(f"[O] Steel fraction CS                               : {Steel_fraction_CS*100:.3f} [%]", file=out)
        print(f"[O] Cable fraction CS  (1 - Steel)                  : {(1-Steel_fraction_CS)*100:.3f} [%]", file=out)
        print(f"[O]  ├ f_sc      (superconductor)                   : {f_sc_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_cu      (copper stabiliser)                : {f_cu_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_He_pipe (He cooling pipe)                  : {f_He_pipe_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_void    (interstitial He void)             : {f_void_CS*100:.2f} [%]", file=out)
        print(f"[O]  ├ f_He      (total He = pipe + void)           : {f_He_CS*100:.2f} [%]", file=out)
        print(f"[O]  └ f_In      (insulation)                       : {f_In_CS*100:.2f} [%]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] L_cable_TF (total cable, all TF coils)          : {L_cable_TF/1e3:.3f} [km]", file=out)
        print(f"[O] L_cable_CS (total cable, all CS modules)        : {L_cable_CS/1e3:.3f} [km]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] SC strands per cable  ({config.Supra_choice})", file=out)
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
        _rho = material_rho(config.Chosen_Steel, config.Supra_choice)
        print(f"[O] TF coil masses  (total all N_TF coils, {config.Chosen_Steel} / {config.Supra_choice})", file=out)
        print(f"[O]  ├ M_steel_TF                                   : {M_steel_TF/1e3:.1f} [t]", file=out)
        print(f"[O]  ├ M_sc_TF                                      : {M_sc_TF/1e3:.1f} [t]",    file=out)
        print(f"[O]  ├ M_cu_TF                                      : {M_cu_TF/1e3:.1f} [t]",    file=out)
        print(f"[O]  ├ M_In_TF                                      : {M_In_TF/1e3:.1f} [t]",    file=out)
        print(f"[O]  └ M_total_TF                                   : {M_total_TF/1e3:.1f} [t]",       file=out)
        print(f"[O] CS coil masses  (total solenoid)", file=out)
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
        print(f"[I]  └ M_blanket / eta_T     : {_M_blanket:.2f}  /  {_eta_T:.2f}  (eta_T_range = {_blanket_concept['eta_T_range']})", file=out)
        print(f"[O] TBR_achieved  (delta_BB_ib = {_delta_BB_ib:.3f} m)        : {_TBR_achieved:.3f}  (TBR_max = {_blanket_concept['TBR_max']:.2f})", file=out)
        print(f"[O] Radial build layers  (plasma → TF, delta_BB_ib = {_delta_BB_ib:.3f} m, delta_BB_ob = {_delta_BB_ob:.3f} m)", file=out)
        for _i, _layer in enumerate(_RB_layers):
            _last   = _i == len(_RB_layers) - 1
            _branch = "├" if not _last else "└"
            _cont   = "│" if not _last else " "
            print(f"[O]  {_branch} {_layer['name']:<28s}: "
                  f"d_ib={_layer['delta_ib']:.3f} m  d_ob={_layer['delta_ob']:.3f} m  "
                  f"V={_layer['V']:.1f} m³  M={_layer['M']/1e3:.1f} t  "
                  f"rho={_layer['rho']:.1f} kg/m³", file=out)
            _comp_masses = _layer['component_masses']
            _materials   = list(_comp_masses)
            for _j, _mat in enumerate(_materials):
                _mbranch = "├" if _j < len(_materials) - 1 else "└"
                _frac    = _layer['composition'][_mat]
                print(f"[O]  {_cont}    {_mbranch} {_mat:<12s}: "
                      f"{_frac*100:5.1f} vol%  {_comp_masses[_mat]/1e3:.2f} t", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Psi_PI      (Breakdown flux)                    : {ΨPI:.3f} [Wb]",      file=out)
        print(f"[O] Psi_RampUp  (Ramp-up flux)                      : {ΨRampUp:.3f} [Wb]",  file=out)
        print(f"[O] Psi_Plateau (Flat-top flux)                     : {Ψplateau:.3f} [Wb]", file=out)
        print(f"[O] Psi_PF      (External PF contribution)          : {ΨPF:.3f} [Wb]",      file=out)
        print(f"[O] Psi_CS      (CS flux requirement)               : {ΨCS_Total:.3f} [Wb]", file=out)
        print(f"[O] V_loop      (Steady-state loop voltage)         : {Vloop*1e3:.1f} [mV]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] P_fus  (Fusion power)                           : {config.P_fus:.3f} [MW]",  file=out)
        print(f"[O] P_CD   (Current drive power)                    : {P_CD:.3f} [MW]",           file=out)
        print(f"[O]  \u251c gamma_LH  (LHCD efficiency)                  : {eta_LH:.4f} [MA/MW\u00b7m\u00b2]",  file=out)
        print(f"[O]  \u251c gamma_EC  (ECCD efficiency)                  : {eta_EC:.4f} [MA/MW\u00b7m\u00b2]",  file=out)
        print(f"[O]  \u2514 gamma_NBI (NBCD efficiency)                  : {eta_NBI:.4f} [MA/MW\u00b7m\u00b2]", file=out)
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
                    Vprime_data=Vprime_data, rho_core=rho_rad_core)
                print(f"[O]  \u251c P_imp_{_sp:<3s} (f={_fc:.1e}, brem+line+recomb)      : {_Pt:.3f} [MW] "
                      f"(core={_Pc:.3f}, edge={_Pt - _Pc:.3f})", file=out)
            print(f"[O]  f_imp_dilution (\u03a3 Z_j\u00b7f_j)                     : {f_imp_dilution:.4f} [-]", file=out)
        _P_rad_total = P_syn + P_Brem + P_line
        _P_rad_core  = P_syn + P_Brem + P_line_core_solution
        print(f"[O] P_rad_core  (core radiation, \u03c1<{rho_rad_core})            : {_P_rad_core:.3f} [MW]", file=out)
        print(f"[O] P_rad_total (total radiated power)              : {_P_rad_total:.3f} [MW]", file=out)
        print(f"[I] rho_rad_core (core/edge boundary)               : {rho_rad_core:.2f}", file=out)
        print(f"[O] eta_CD (Current drive efficiency)               : {eta_CD:.3f} [MA/MW·m²]",   file=out)
        print(f"[O] Q      (Energy gain factor)                     : {Q:.3f}",                   file=out)
        print(f"[O] P_elec (Net electrical power)                   : {P_elec:.3f} [MW]",         file=out)
        print(f"[O] P_wallplug (Wall-plug heating/CD power)         : {P_wallplug:.3f} [MW]",     file=out)
        _P_gross = config.eta_T * config.M_blanket * config.P_fus
        print(f"[O] Q_eng  (Engineering gain = P_elec / P_wallplug) : {P_elec / P_wallplug:.3f}", file=out)
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
        print(f"[I] Tbar  (Volume-averaged ion temperature)         : {config.Tbar:.3f} [keV]",    file=out)
        print(f"[O] nbar  (Volume-averaged electron density)        : {nbar:.3f} [10²⁰ m⁻³]",     file=out)
        print(f"[O] nbar_line (Line-averaged electron density)      : {nbar_line:.3f} [10²⁰ m⁻³]", file=out)
        print(f"[O] nG    (Greenwald density limit)                 : {nG:.3f} [10²⁰ m⁻³]",       file=out)
        print(f"[O] f_GW  (Greenwald fraction, nbar_line/(Ip/πa²))  : {nbar_line/(nG/config.Greenwald_limit):.3f} [-]",  file=out)
        print(f"[O] f_GW / f_GW_limit                               : {nbar_line/nG:.3f} [-]",     file=out)
        print(f"[O] pbar  (Volume-averaged pressure)                : {pbar:.3f} [MPa]",            file=out)
        print(f"[O] f_He  (Helium ash fraction)                     : {f_alpha*100:.3f} [%]",      file=out)
        print(f"[O] tau_alpha (Alpha-particle confinement time)     : {tau_alpha:.3f} [s]",         file=out)
        print(f"[O] W_th  (Plasma thermal energy content)           : {W_th/1e6:.3f} [MJ]",        file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] beta_T (Toroidal beta)                          : {betaT*100:.3f} [%]", file=out)
        print(f"[O] beta_P (Poloidal beta)                          : {betaP:.3f}",         file=out)
        print(f"[O] beta_N (Normalised beta, thermal)               : {betaN:.3f}",         file=out)
        print(f"[O] beta_N_total (incl. fast α, Stix model)         : {betaN_total:.3f}",   file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] q*   (Kink safety factor)                       : {qstar:.3f}", file=out)
        if config.q_profile_mode == 'academic':
            _q95_note = "imposed = q(rho_95) in academic mode"
        else:
            _q95_note = "scaling only; profile q(rho_95) may differ (refined mode)"
        print(f"[O] q95  (Safety factor at 95% flux surface)        : {q95:.3f}    [{_q95_note}]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] P_div    (P_α+P_CD−P_rad_tot, divertor power)   : {P_sep:.3f} [MW]",    file=out)
        print(f"[O] P_Thresh (L-H power threshold)                  : {P_Thresh:.3f} [MW]", file=out)
        print(f"[O] P_rad_tot / S_wall  (FW radiative load)         : {P_wall_rad:.3f} [MW/m²]", file=out)
        print(f"[O] P_div / S_wall  (mean SOL exhaust flux)         : {P_wall_div:.3f} [MW/m²]", file=out)
        print(f"[O] P_sep / R0  (Divertor heat scaling)             : {heat:.3f} [MW/m]",      file=out)
        print(f"[O] P_sep·B0 / R0  (Parallel heat flux proxy)       : {heat_par:.3f} [MW·T/m]", file=out)
        print(f"[O] P_sep·B0 / (q95·R0·A)  (Poloidal heat flux)     : {heat_pol:.3f} [MW·T/m]", file=out)
        print(f"[O] Gamma_n  (Neutron wall load)                    : {Gamma_n:.3f} [MW/m²]",   file=out)
        print("=========================================================================", file=out)

        # ── Runaway Electron Indicators (post-convergence diagnostic) ─────────
        # Indicative only — does NOT affect any convergence loop.
        # li recalculated here from available profile data (not in results tuple).
        print("=== Runaway Electron Indicators (Indicative) ===", file=out)
        print("-------------------------------------------------------------------------", file=out)
        try:
            # li is the l_i(3) value returned by the q-profile dispatcher
            # (f_q_profile_academic or f_q_profile_refined depending on
            # config.q_profile_mode), passed through the results tuple.
            RE = compute_RE_indicators(
                Ip=Ip, nbar=nbar, Tbar=config.Tbar,
                a=config.a, R0=config.R0, κ=κ, Z_eff=config.Zeff, li=li_sc,
                nu_n=nu_n, nu_T=nu_T,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Te_final_eV=config.Te_final_eV, tau_TQ=config.tau_TQ,
                Vprime_data=Vprime_data, V=Volume,
                pellet_dilution=config.pellet_dilution,
                pellet_dilution_cools=config.pellet_dilution_cools,
            )
            print(f"[!] INDICATOR ONLY — order-of-magnitude estimate for design comparison.", file=out)
            print(f"[I] tau_TQ         (Thermal quench time)                : {RE['tau_TQ']*1e3:.1f} [ms]", file=out)
            print(f"[I] Te_final       (Post-TQ temperature)                : {RE['Te_final_eV']:.0f} [eV]", file=out)
            print(f"[I] pellet_dilution (SPI/MGI density factor)            : {RE['pellet_dilution']:.1f} [-]", file=out)
            print(f"[I] nbar_diluted   (Post-pellet density)                : {RE['nbar_diluted']:.3f} [1e20 m-3]", file=out)
            print(f"[I] Tbar_diluted   (Pre-TQ temperature, {'isobaric' if RE['pellet_dilution_cools'] else 'unchanged'})       : {RE['Tbar_diluted']:.2f} [keV]", file=out)
            print(f"[O] f_RE_core      (Hot-tail fraction at rho=0)         : {RE['f_RE_core']:.3e} [-]", file=out)
            print(f"[O] f_RE_avg       (Volume-averaged hot-tail fraction)  : {RE['f_RE_avg']:.3e} [-]", file=out)
            print(f"[O] I_RE_seed      (Hot-tail seed, pre-pellet)          : {RE['I_RE_seed']:.3e} [A]", file=out)
            print(f"[O] I_RE_aval      (After avalanche, post-pellet)       : {RE['I_RE_avalanche']*1e-6:.3f} [MA]", file=out)
            print(f"[O] I_RE / Ip      (RE-to-plasma current fraction)      : {RE['f_RE_to_Ip']*100:.1f} [%]", file=out)
            print(f"[O] W_mag_RE       (RE magnetic energy = 1/2 Li I^2_RE) : {RE['W_mag_RE']:.1f} [MJ]", file=out)
            print(f"[O] E_RE_kin       (RE kinetic energy, <gamma>=10)      : {RE['E_RE_kin']:.1f} [MJ]", file=out)
        except Exception as _e:
            print(f"[O] RE indicators could not be computed: {_e}", file=out)
        print("=========================================================================", file=out)

        # ── Techno-economic cost assessment (post-convergence) ────────────
        # Sheffield & Milora, Fus. Sci. Technol. 70, 14–35 (2016).
        # Uses D0FUS radial build volumes and power balance outputs.
        if config.cost_model != 'None':
            print("=== Cost Assessment — Sheffield (2016) ===", file=out)
            print("-------------------------------------------------------------------------", file=out)
            try:
                # Derived quantities from D0FUS convergence
                P_th = config.P_fus * M_blanket_effective(config.Blanket_choice) + P_CD   # total thermal [MW]
                P_e  = max(P_elec, 1.0)                          # net electric [MWe]
                S_FW = Surface                                   # first-wall surface [m^2]

                # Component volumes from D0FUS radial build
                _, _, _H_TF_c, _, _, _, _, _, _ = f_TF_cross_section(
                    config.a, config.b, config.R0, c, Delta_TF_disp)
                (_, _, _, V_FI_c) = f_volume(
                    config.a, config.b, c, d, config.R0, κ, Delta_TF_disp, _H_TF_c)

                _res = f_costs_Sheffield(
                    discount_rate  = config.discount_rate,
                    contingency    = config.contingency,
                    T_life         = config.T_life,
                    T_build        = config.T_build,
                    P_t            = P_th,
                    P_e            = P_e,
                    P_aux          = P_CD,
                    Gamma_n        = Gamma_n,
                    T_op_limit     = T_op_limit,
                    CF             = CF,
                    t_life_bl_yr   = t_life_bl_yr,
                    t_life_div_yr  = t_life_div_yr,
                    V_FI           = V_FI_c,
                    V_pc           = V_TF_one * N_TF_disp + V_CS_geom,
                    V_sg           = V_blanket,
                    V_bl           = V_rb_BB,
                    S_tt           = 0.1 * S_FW,
                    Supra_cost_factor = config.Supra_cost_factor,
                )
                (T_op_limit, CF, C_invest, COE,
                 C_ind, C_Op_waste, C_Op_OM, C_Op_F,
                 C_syst_other, C_syst_BOP, C_syst_heat, C_syst_aux,
                 C_reac_tt, C_reac_bl, C_reac_sg, C_reac_pc) = _res
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
                print(f"[O] P_e   (net electric power)                      : {P_e:.1f} [MWe]", file=out)
                print(f"[O] Gamma_n (neutron wall load)                     : {Gamma_n:.3f} [MW/m^2]", file=out)
                print("-------------------------------------------------------------------------", file=out)

                # Component volumes
                print(f"[O] V_blanket (FW+BB+shield+VV+gaps, Miller)        : {V_blanket:.1f} [m^3]", file=out)
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
                print(f"[O] COE (cost of electricity)                       : {COE:.3f} [EUR/MWh]", file=out)

            except Exception as _e:
                print(f"[!] Cost assessment failed: {_e}", file=out)
            print("=========================================================================", file=out)

    if verbose >= 2:
        print(f"\n✓ Results saved to: {output_path}\n")

    # ── Optional figure generation ────────────────────────────────────────────
    if save_figures:
        _generate_run_figures(config, results, output_path, verbose=verbose)

    return output_path


#%% Main entry point
# ---------------------------------------------------------------------------

def _generate_run_figures(config: GlobalConfig, results: tuple,
                          output_path: str, verbose: int = 0) -> None:
    """
    Generate and save run-specific figures (12) into output_path/figures/.

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

    # ── Self-consistent q-profile for figures ─────────────────────────────────
    # Recompute using the converged global quantities stored in run_dict.
    # Cost is negligible (~50 ms) and avoids modifying the results tuple.
    # Mode-aware dispatch on config.q_profile_mode.
    try:
        _Ip    = run_dict.get("Ip", 15.0)
        _q95   = run_dict.get("q95", 3.0)
        _I_CD  = results[10] if len(results) > 10 else 0.0
        if config.q_profile_mode == 'academic':
            _q_sc = f_q_profile_academic(
                _Ip, _q95,
                config.R0, config.a, run_dict.get("kappa_edge", 1.7),
                alpha_J=config.alpha_J,
                rho_ped=run_dict["rho_ped"],
                n_ped_frac=run_dict["n_ped_frac"],
                T_ped_frac=run_dict["T_ped_frac"],
                Vprime_data=run_dict.get("Vprime_data"),
                kappa_95=None, rho_95=0.95, n_rho=200)
        else:  # 'refined'
            _q_sc = f_q_profile_refined(
                _Ip, _I_CD,
                config.R0, config.a,
                run_dict.get("B0", config.Bmax_TF),
                run_dict.get("kappa_edge", 1.7), run_dict.get("nbar", 1.0),
                config.Tbar, config.Zeff,
                run_dict["nu_n"], run_dict["nu_T"],
                trapped_fraction_model=config.trapped_fraction_model,
                eta_model=config.eta_model,
                rho_ped=run_dict["rho_ped"],
                n_ped_frac=run_dict["n_ped_frac"],
                T_ped_frac=run_dict["T_ped_frac"],
                Vprime_data=run_dict.get("Vprime_data"),
                rho_CD=config.rho_EC, delta_CD=0.15,
                n_rho=120, max_iter=15, tol=1e-3, damping=0.5)
        run_dict["_q_sc"]        = _q_sc
    except Exception as _e:
        if verbose >= 1:
            print(f"  [warn] Self-consistent q-profile for figures failed: {_e}")

    # ── Generate full figure catalogue ────────────────────────────────────────
    if verbose >= 2:
        print("\n  Generating figures:")
    try:
        figs.plot_run(run_dict, save_dir=fig_dir)
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


def main(input_file: str = None, save_figures: bool = False,
         verbose: int = 0) -> tuple:
    """
    Load a configuration, run the calculation, and save the output report.

    Parameters
    ----------
    input_file   : str, optional
        Path to a plain-text input file.  If None, the module looks for
        ``D0FUS_INPUTS/default_input.txt`` relative to the package root.
        Missing files trigger a warning and the code runs with DEFAULT_CONFIG.
    save_figures : bool, optional
        If True, generate and save PNG figures alongside the text report.
        Default: False.  A ``save_figures = 1`` line in the input file
        overrides this argument.
    verbose : int, optional
        Verbosity level: 0 = silent, 1 = convergence summary + warnings,
        2 = full debug trace.  Default: 2 (backward-compatible).

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

    # Input-file flag takes precedence over the function argument
    _file_flag = _parse_save_figures_flag(input_file)
    save_figures = _file_flag or save_figures

    # ── Launch calculation ────────────────────────────────────────────────────
    if verbose >= 2:
        print("\n" + "=" * 73)
        print("Starting D0FUS calculation...")
        print("=" * 73 + "\n")

    try:
        results    = run(config, verbose=verbose)
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        save_run_output(config, results, output_dir, input_file_path,
                        save_figures=save_figures, verbose=verbose)
        return results

    except Exception as e:
        if verbose >= 1:
            print(f"\n!!! ERROR during calculation !!!")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


#%% Standalone execution

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)
    print("\nD0FUS_run completed successfully!")