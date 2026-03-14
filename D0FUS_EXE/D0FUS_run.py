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
from D0FUS_BIB.D0FUS_parameterization import GlobalConfig, DEFAULT_CONFIG
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from scipy.optimize import brentq


#%% Input file loader

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

            overrides[key] = val
            if verbose >= 2:
                print(f"  [input]  {key} = {val}")

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
    tuple
        Ordered scalar outputs (see bottom of function for full list).
        All entries are np.nan if the self-consistent solver fails to converge.
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
    Bmax_CS                   = config.Bmax_CS
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
    Option_Kappa              = config.Option_Kappa
    κ_manual                  = config.κ_manual
    betaN_limit               = config.betaN_limit
    q_limit                   = config.q_limit
    Greenwald_limit           = config.Greenwald_limit
    ms                        = config.ms
    Atomic_mass               = config.Atomic_mass
    Zeff                      = config.Zeff
    r_synch                   = config.r_synch
    C_Alpha                   = config.C_Alpha
    Ce                        = config.Ce
    eta_model                 = config.eta_model
    Chosen_Steel              = config.Chosen_Steel
    nu_Steel                  = config.nu_Steel
    Young_modul_Steel         = config.Young_modul_Steel
    Young_modul_GF            = config.Young_modul_GF
    fatigue_CS                = config.fatigue_CS
    Radial_build_model        = config.Radial_build_model
    Choice_Buck_Wedg          = config.Choice_Buck_Wedg
    Supra_choice              = config.Supra_choice
    J_wost_Manual             = config.J_wost_Manual
    coef_inboard_tension      = config.coef_inboard_tension
    F_CClamp                  = config.F_CClamp
    n_TF                      = config.n_TF
    c_BP                      = config.c_BP
    Gap                       = config.Gap
    n_CS                      = config.n_CS
    cs_axial_stress           = config.cs_axial_stress
    N_sub_CS                  = config.N_sub_CS
    T_helium                  = config.T_helium
    Marge_T_He                = config.Marge_T_He
    f_He                      = config.f_He
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
    eta_T                     = config.eta_T
    eta_RF                    = config.eta_RF
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
    C_EC                      = config.C_EC
    C_NBI                     = config.C_NBI
    Plasma_geometry           = config.Plasma_geometry
    M_blanket                 = config.M_blanket

    # Core/edge radiation boundary for P_loss convention.
    # Radiation emitted inside ρ < rho_rad_core is subtracted from P_heat
    # in τ_E and the scaling law inversion (it never crosses the separatrix).
    # Radiation emitted at ρ > rho_rad_core is edge radiation that reduces
    # P_sep but does NOT affect confinement.
    # Typical value: 0.7 (well inside pedestal top for ITER/DEMO shaping).
    # Set to 1.0 to recover the legacy behaviour (all radiation subtracted).
    rho_rad_core = getattr(config, 'rho_rad_core', 0.7)

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
    # L-mode  : purely parabolic (no pedestal)
    # H-mode  : parabola-with-pedestal, calibrated on EU-DEMO 2017 PROCESS run
    # Advanced: highly peaked core
    # Manual  : user-defined from GlobalConfig
    profiles = {
        'L':        {'nu_n': 0.5,  'nu_T': 1.75, 'rho_ped': 1.00, 'n_ped_frac': 0.00, 'T_ped_frac': 0.00},
        'H':        {'nu_n': 1.0,  'nu_T': 1.45, 'rho_ped': 0.94, 'n_ped_frac': 0.80, 'T_ped_frac': 0.40},
        'Advanced': {'nu_n': 1.5,  'nu_T': 2.00, 'rho_ped': 0.94, 'n_ped_frac': 0.80, 'T_ped_frac': 0.40},
    }
    if Plasma_profiles == 'Manual':
        nu_n       = nu_n_manual
        nu_T       = nu_T_manual
        # rho_ped, n_ped_frac, T_ped_frac already read from config above
    elif Plasma_profiles in profiles:
        nu_n       = profiles[Plasma_profiles]['nu_n']
        nu_T       = profiles[Plasma_profiles]['nu_T']
        rho_ped    = profiles[Plasma_profiles]['rho_ped']
        n_ped_frac = profiles[Plasma_profiles]['n_ped_frac']
        T_ped_frac = profiles[Plasma_profiles]['T_ped_frac']
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
    # CS fatigue knockdown applies only in pulsed wedging configuration
    if Choice_Buck_Wedg == 'Wedging' and Operation_mode == 'Pulsed':
        σ_CS = Steel(Chosen_Steel, config.σ_manual) / fatigue_CS
    else:
        σ_CS = Steel(Chosen_Steel, config.σ_manual)

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

    # Precompute Miller volume derivative V'(ρ) for D0FUS geometry mode.
    # Vprime_data = (rho_grid, Vprime, V_total) is passed to f_nbar, f_pbar,
    # and f_plasma_volume so that all volume integrals use the same shaped
    # flux-surface Jacobian.  In Academic mode, Vprime_data = None triggers
    # the fast cylindrical-torus approximation in every downstream function.
    if Plasma_geometry == 'D0FUS':
        Vprime_data = precompute_Vprime(R0, a, κ, δ,
                                        geometry_model='D0FUS',
                                        kappa_95=κ_95, delta_95=δ_95)
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
            _Pc, _Pt = f_P_line_radiation_profile(
                sp, fc, nbar_loc, Tbar_loc, nu_n, nu_T, V_loc,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data, rho_core=rho_rad_core)
            P_core  += _Pc
            P_total += _Pt
        return P_core, P_total

    def _compute_P_line_total(nbar_loc, Tbar_loc, V_loc):
        """Sum profile-integrated line radiation (total only) [MW].
        Retained for backward compatibility and per-species reporting."""
        P_total = 0.0
        for sp, fc in zip(imp_species_list, imp_conc_list):
            P_total += f_P_line_radiation_profile(
                sp, fc, nbar_loc, Tbar_loc, nu_n, nu_T, V_loc,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                Vprime_data=Vprime_data)
        return P_total

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

    N_TF, ripple, Delta = Number_TF_coils(R0, a, b, ripple_adm, L_min)
    N_sub = N_TF / Dump_resistor_subdivision

    result_J_TF = calculate_cable_current_density(
        sc_type=Supra_choice, B_peak=Bmax_TF, T_op=T_helium, E_mag=E_mag_TF,
        I_cond=I_cond, V_max=V_max, N_sub=N_sub, tau_h=tau_h, f_He=f_He, f_In=f_In,
        T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn, Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
        J_wost_Manual=J_wost_Manual if Supra_choice == 'Manual' else None)
    J_max_TF_conducteur = result_J_TF['J_wost']

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
        # Bremsstrahlung and synchrotron are volumetric, dominated by the hot
        # dense core (profiles ~ n²T^{1/2} and ~ nT^{3.5}) → counted entirely
        # as core radiation (> 95% for typical reactor profiles).
        # Line radiation is split at ρ = rho_rad_core: the core fraction
        # enters P_loss (affects τ_E), the total enters P_sep.
        P_Brem_loc = f_P_bremsstrahlung(nbar_loc, Tbar, Zeff,
                                        Volume_solution, nu_n, nu_T)
        P_syn_loc  = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_loc,
                                     κ, nu_n, nu_T, r_synch,
                                     rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                     T_ped_frac=T_ped_frac)
        P_line_core_loc, P_line_total_loc = _compute_P_line_split(
            nbar_loc, Tbar, Volume_solution)

        # P_rad_core  → subtracted from P_heat in τ_E and scaling law (Ip)
        # P_rad_total → subtracted from P_heat to get P_sep (divertor load)
        P_rad_core_loc  = P_Brem_loc + P_syn_loc + P_line_core_loc
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
            W_th_dbg = 1.5 * pbar_loc * 1e6 * Volume_solution / 1e6  # MJ
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
        q95_loc = f_q95(B0_solution, Ip_loc, R0, a, κ_95, δ_95)

        if Bootstrap_choice in ('Sauter', 'Redl'):
            if Bootstrap_choice == 'Sauter':
                Ib_loc = f_Sauter_Ib(
                    R0, a, κ, B0_solution, nbar_loc, Tbar,
                    q95_loc, Zeff, nu_n, nu_T,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, kappa_95=κ_95)
            else:
                Ib_loc = f_Redl_Ib(
                    R0, a, κ, B0_solution, nbar_loc, Tbar,
                    q95_loc, Zeff, nu_n, nu_T,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, kappa_95=κ_95)
        elif Bootstrap_choice == 'Freidberg':
            Ib_loc = f_Freidberg_Ib(R0, a, κ, pbar_loc, Ip_loc)
        elif Bootstrap_choice == 'Segal':
            Ib_loc = f_Segal_Ib(
                nu_n, nu_T, a / R0, κ, nbar_loc, Tbar, R0, Ip_loc,
                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                T_ped_frac=T_ped_frac)
        else:
            raise ValueError(
                f"Unknown Bootstrap_choice: '{Bootstrap_choice}'.")

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
                eta_LH_loc  = f_etaCD_LH(
                    a, R0, B0_solution, nbar_loc, Tbar, nu_n, nu_T,
                    Z_eff=Zeff,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac)
                eta_EC_loc  = f_etaCD_EC(
                    a, R0, Tbar, nbar_loc, Zeff, nu_T, nu_n,
                    rho_EC, C_EC, rho_ped, n_ped_frac, T_ped_frac)
                eta_NBI_loc = f_etaCD_NBI(
                    A_beam, E_beam_keV,
                    a, R0, Tbar, nbar_loc, Zeff, nu_T, nu_n,
                    rho_NBI, C_NBI, rho_ped, n_ped_frac, T_ped_frac)
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
            P_Ohm_loc = f_P_Ohm(I_Ohm_loc, Tbar, R0, a, κ)
            Q_loc     = f_Q(P_fus, P_CD_loc, P_Ohm_loc)

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
            _nan, _nan,                          # P_wall_H, P_wall_L
            _nan,                               # Gamma_n
            _nan, _nan,                          # f_alpha, tau_alpha
            _nan, _nan,                          # J_TF, J_CS
            _nan, _nan, _nan, _nan, _nan, _nan, _nan,  # TF radial build + stresses
            _nan, _nan, _nan, _nan, _nan, _nan, _nan,  # CS radial build + stresses
            _nan, _nan, _nan, _nan,            # r_minor, r_sep, r_c, r_d
            _nan, _nan, _nan, _nan,            # κ, κ_95, δ, δ_95
            _nan, _nan, _nan, _nan, _nan,      # ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS
            _nan, _nan, _nan,                  # eta_LH, eta_EC, eta_NBI
            _nan, _nan, _nan, _nan,            # P_LH, P_EC, P_NBI, P_ICR
            _nan, _nan, _nan,                  # I_LH, I_EC, I_NBI
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
    W_th_solution        = 1.5 * pbar_solution * 1e6 * Volume_solution   # [J]
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
    #  neoclassical bootstrap models f_Sauter_Ib / f_Redl_Ib which involve
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
    B_pol_solution  = f_Bpol(q95_solution, B0_solution, a, R0)
    betaT_solution  = f_beta_T(pbar_solution, B0_solution)
    betaP_solution  = f_beta_P(a, κ, pbar_solution, Ip_solution)
    beta_solution   = f_beta(betaP_solution, betaT_solution)
    betaN_solution  = f_beta_N(beta_solution, a, B0_solution, Ip_solution)
    # Greenwald limit is defined in terms of line-averaged density;
    # f_nG returns n_G [1e20 m-3], which must be compared with nbar_line, not nbar_vol.
    nG_solution     = f_nG(Ip_solution, a) * Greenwald_limit

    # Current drive and power balance
    # ── Individual CD efficiencies (computed once, reused in breakdown) ────────
    eta_LH_solution  = f_etaCD_LH(a, R0, B0_solution, nbar_solution, Tbar,
                                   nu_n, nu_T,
                                   rho_ped=rho_ped, n_ped_frac=n_ped_frac)
    eta_EC_solution  = f_etaCD_EC(a, R0, Tbar, nbar_solution, Zeff, nu_T, nu_n,
                                   rho_EC, C_EC,
                                   rho_ped, n_ped_frac, T_ped_frac)
    eta_NBI_solution = f_etaCD_NBI(A_beam, E_beam_keV,
                                    a, R0, Tbar, nbar_solution, Zeff, nu_T, nu_n,
                                    rho_NBI, C_NBI,
                                    rho_ped, n_ped_frac, T_ped_frac)

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
        P_Ohm_solution = f_P_Ohm(I_Ohm_solution, Tbar, R0, a, κ)
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

    # Separatrix power: P_sep = P_alpha + P_CD - P_rad_total
    # P_rad_total is used here (not P_rad_core) because edge radiation
    # reduces the power reaching the divertor target plates.
    P_sep_solution      = f_P_sep(P_fus, P_CD_solution, P_rad_total_solution)
    Gamma_n_solution    = f_Gamma_n(a, P_fus, R0, κ, S_wall=Surface_solution)
    heat_D0FUS_solution = f_heat_D0FUS(R0, P_sep_solution)
    heat_par_solution   = f_heat_par(R0, B0_solution, P_sep_solution)
    heat_pol_solution   = f_heat_pol(R0, B0_solution, P_sep_solution, a, q95_solution)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(
        P_sep_solution, B_pol_solution, R0, a / R0, theta_deg)
    P_1rst_wall_Hmod = f_P_wall(P_sep_solution, P_CD_solution, Surface_solution,
                                H_mode=True)
    P_1rst_wall_Lmod  = f_P_wall(P_sep_solution, P_CD_solution, Surface_solution,
                                 H_mode=False)
    P_wallplug_solution = P_CD_solution / eta_RF   # Wall-plug power consumed by heating/CD [MW]
    P_elec_solution   = f_P_elec(P_fus, P_CD_solution, eta_T, M_blanket, eta_RF)
    li_solution = f_li(
        nu_T, nu_n, Tbar, nbar_solution,
        Zeff, a, R0, q95_solution,
        eta_model=eta_model,
        rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac
    )

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

    elif Radial_build_model in ("D0FUS", "CIRCE"):
        (c, c_WP_TF, c_Nose_TF,
         σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_D0FUS(
            a, b, R0, σ_TF, J_max_TF_conducteur, Bmax_TF, Choice_Buck_Wedg, omega_TF, n_TF,
            c_BP, coef_inboard_tension, F_CClamp)

    else:
        raise ValueError(
            f"Unknown radial build model: '{Radial_build_model}'. "
            "Valid options: 'academic', 'D0FUS', 'CIRCE'."
        )

    # ==============================================================================
    #    MAGNETIC FLUX REQUIREMENTS (Inductive Scenario)
    #    Volt-seconds budget for plasma initiation, ramp-up, and flat-top.
    # ==============================================================================
    (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(
    Ip_solution, I_Ohm_solution, Bmax_TF, a, b, c, R0, κ,
    nbar_solution, Tbar, Ce, Temps_Plateau_input, li_solution,
    Choice_Buck_Wedg, Gap,
    Zeff, q95_solution, nu_T, nu_n, eta_model,
    E_phi_BD=config.E_phi_BD,
    t_BD=config.t_BD,
    C_PF=config.C_PF,
    rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
    )
    # ==============================================================================
    #    CENTRAL SOLENOID (CS) DESIGN
    #    Determines the CS radial thickness 'd' to satisfy the Volt-second budget.
    # ==============================================================================
    cs_args = (ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax_TF, Bmax_CS, σ_CS,
           Supra_choice, J_wost_Manual, T_helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
           config)

    if Radial_build_model == "academic":
        (d, σ_z_CS, σ_theta_CS, σ_r_CS,
         Steel_fraction_CS, B_CS, J_CS) = f_CS_ACAD(*cs_args)
    elif Radial_build_model == "D0FUS":
        (d, σ_z_CS, σ_theta_CS, σ_r_CS,
         Steel_fraction_CS, B_CS, J_CS) = f_CS_D0FUS(*cs_args)
    elif Radial_build_model == "CIRCE":
        (d, σ_z_CS, σ_theta_CS, σ_r_CS,
         Steel_fraction_CS, B_CS, J_CS) = f_CS_CIRCE(*cs_args)

    # Total inductive flux swing provided by the CS
    # ΨCS = Breakdown + Ramp-up + Flat-top − External PF contribution
    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF

    # Volume-based machine cost proxy
    cost_solution = f_cost(a, b, c, d, R0, κ, P_fus)

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
            heat_D0FUS_solution,  heat_par_solution,  heat_pol_solution,
            lambda_q_Eich_m,  q_target_Eich,
            P_1rst_wall_Hmod,  P_1rst_wall_Lmod,
            Gamma_n_solution,
            f_alpha_solution,  tau_alpha,
            J_max_TF_conducteur,  J_CS,
            c,  c_WP_TF,  c_Nose_TF,  σ_z_TF,  σ_theta_TF,  σ_r_TF,  Steel_fraction_TF,
            d,  σ_z_CS,  σ_theta_CS,  σ_r_CS,  Steel_fraction_CS,  B_CS,  J_CS,
            R0 - a,  R0 - a - b,  R0 - a - b - c,  R0 - a - b - c - d,
            κ,  κ_95,  δ,  δ_95,
            ΨPI,  ΨRampUp,  Ψplateau,  ΨPF,  ΨCS,
            eta_LH_solution,  eta_EC_solution,  eta_NBI_solution,
            P_LH_solution,  P_EC_solution,  P_NBI_solution,  P_ICR_solution,
            I_LH_solution,  I_EC_solution,  I_NBI_solution)


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
     _P_wall_H, _P_wall_L,
     _Gamma_n,
     _f_alpha, _tau_alpha,
     _J_TF, _J_CS_1,
     c_TF, c_WP_TF, c_Nose_TF, _sz_TF, _st_TF, _sr_TF, _sf_TF,
     c_CS, _sz_CS, _st_CS, _sr_CS, _sf_CS, _B_CS2, _J_CS2,
     r_minor, r_sep, r_c, r_d,
     kappa_edge, kappa_95, delta_edge, delta_95,
     *_rest) = results

    # ── Profile peaking / pedestal parameters ─────────────────────────────────
    _prof_presets = {
        'L':        {'nu_n': 0.5,  'nu_T': 1.75, 'rho_ped': 1.00,
                     'n_ped_frac': 0.00, 'T_ped_frac': 0.00},
        'H':        {'nu_n': 1.0,  'nu_T': 1.45, 'rho_ped': 0.94,
                     'n_ped_frac': 0.80, 'T_ped_frac': 0.40},
        'Advanced': {'nu_n': 1.5,  'nu_T': 2.00, 'rho_ped': 0.94,
                     'n_ped_frac': 0.80, 'T_ped_frac': 0.40},
    }
    if config.Plasma_profiles == 'Manual':
        nu_n       = config.nu_n_manual
        nu_T       = config.nu_T_manual
        rho_ped    = config.rho_ped
        n_ped_frac = config.n_ped_frac
        T_ped_frac = config.T_ped_frac
    else:
        _p         = _prof_presets.get(config.Plasma_profiles, _prof_presets['H'])
        nu_n       = _p['nu_n'];       nu_T       = _p['nu_T']
        rho_ped    = _p['rho_ped'];    n_ped_frac = _p['n_ped_frac']
        T_ped_frac = _p['T_ped_frac']

    # ── Vprime_data (Miller geometry pre-computation, if applicable) ──────────
    Vprime_data = None
    if (config.Plasma_geometry == 'D0FUS'
            and np.isfinite(float(kappa_edge))
            and np.isfinite(float(delta_edge))):
        try:
            Vprime_data = precompute_Vprime(
                config.R0, config.a,
                float(kappa_edge), float(delta_edge),
                geometry_model='D0FUS',
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
        _N_TF, *_ = Number_TF_coils(
            config.R0, config.a, config.b,
            config.ripple_adm, config.L_min,
        )
        N_TF = int(_N_TF)
    except Exception:
        N_TF = int(config.n_TF)   # Fall back to config hint

    # ── Safe float extraction (guard against NaN in radial build) ─────────────
    def _f(x, fallback):
        try:
            v = float(x)
            return v if np.isfinite(v) else fallback
        except (TypeError, ValueError):
            return fallback

    return {
        # Plasma geometry
        "R0":          config.R0,
        "a":           config.a,
        "kappa_edge":  _f(kappa_edge,  1.70),
        "delta_edge":  _f(delta_edge,  0.33),
        "kappa_95":    _f(kappa_95,    1.60),
        "delta_95":    _f(delta_95,    0.25),
        "Vprime_data": Vprime_data,
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
        "N_TF":        N_TF,
        "Gap":         config.Gap,
        # First wall / blanket / shield (optional; ITER-like defaults)
        "e_fw":        getattr(config, 'e_fw',      0.05),
        "e_blanket":   getattr(config, 'e_blanket', 0.45),
        "e_shield":    getattr(config, 'e_shield',  0.50),
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
     P_wall_H, P_wall_L,
     Gamma_n,
     f_alpha, tau_alpha,
     J_TF, J_CS,
     c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
     d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS,
     r_minor, r_sep, r_c, r_d,
     κ, κ_95, δ, δ_95,
     ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS_Total,
     eta_LH, eta_EC, eta_NBI,
     P_LH_out, P_EC_out, P_NBI_out, P_ICR_out,
     I_LH_out, I_EC_out, I_NBI_out) = results

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

    # Reconstruct profile params and Vprime_data for per-species display
    profiles = {
        'L':        {'nu_n': 0.5,  'nu_T': 1.75, 'rho_ped': 1.00, 'n_ped_frac': 0.00, 'T_ped_frac': 0.00},
        'H':        {'nu_n': 1.0,  'nu_T': 1.45, 'rho_ped': 0.94, 'n_ped_frac': 0.80, 'T_ped_frac': 0.40},
        'Advanced': {'nu_n': 1.5,  'nu_T': 2.00, 'rho_ped': 0.94, 'n_ped_frac': 0.80, 'T_ped_frac': 0.40},
    }
    if config.Plasma_profiles == 'Manual':
        nu_n = config.nu_n_manual; nu_T = config.nu_T_manual
        rho_ped = config.rho_ped; n_ped_frac = config.n_ped_frac; T_ped_frac = config.T_ped_frac
    elif config.Plasma_profiles in profiles:
        p = profiles[config.Plasma_profiles]
        nu_n = p['nu_n']; nu_T = p['nu_T']; rho_ped = p['rho_ped']
        n_ped_frac = p['n_ped_frac']; T_ped_frac = p['T_ped_frac']
    else:
        nu_n = 0; nu_T = 0; rho_ped = 1.0; n_ped_frac = 0; T_ped_frac = 0
    Vprime_data = None
    if config.Plasma_geometry == 'D0FUS' and np.isfinite(κ) and np.isfinite(δ):
        Vprime_data = precompute_Vprime(config.R0, config.a, κ, δ,
                                         geometry_model='D0FUS',
                                         kappa_95=κ_95, delta_95=δ_95)
    rho_rad_core = getattr(config, 'rho_rad_core', 0.7)

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
        print(f"[I] R0 (Major Radius)                               : {config.R0:.3f} [m]",   file=out)
        print(f"[I] a  (Minor Radius)                               : {config.a:.3f} [m]",    file=out)
        print(f"[I] b  (BB & neutron shield thickness)              : {r_minor-r_sep:.3f} [m]", file=out)
        print(f"[O] c  (TF coil thickness)                          : {r_sep-r_c:.3f} [m]",   file=out)
        print(f"[O] d  (CS thickness)                               : {r_c-r_d:.3f} [m]",     file=out)
        print(f"[O] R0-a-b-c-d (CS inner bore radius)               : {r_d:.3f} [m]",         file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Kappa    (Plasma elongation)                    : {κ:.3f}",     file=out)
        print(f"[O] Kappa_95 (Elongation at 95% flux surface)       : {κ_95:.3f}", file=out)
        print(f"[O] Delta    (Plasma triangularity)                 : {δ:.3f}",     file=out)
        print(f"[O] Delta_95 (Triangularity at 95% flux surface)    : {δ_95:.3f}", file=out)
        print(f"[O] Volume   (Plasma volume)                        : {Volume:.3f} [m³]",   file=out)
        print(f"[O] Surface  (First wall area)                      : {Surface:.3f} [m²]",  file=out)
        print(f"[I] Mechanical configuration                        : {config.Choice_Buck_Wedg}", file=out)
        print(f"[I] Superconductor technology                       : {config.Supra_choice}",    file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] Bmax_TF (Peak field on TF conductor)            : {config.Bmax_TF:.3f} [T]",   file=out)
        print(f"[O] B0   (On-axis magnetic field)                   : {B0:.3f} [T]",             file=out)
        print(f"[O] BCS  (CS peak magnetic field)                   : {B_CS:.3f} [T]",           file=out)
        print(f"[O] J_E-TF (TF engineering current density)         : {J_TF/1e6:.3f} [MA/m²]",  file=out)
        print(f"[O] J_E-CS (CS engineering current density)         : {J_CS/1e6:.3f} [MA/m²]",  file=out)
        print(f"[O] Steel fraction TF                               : {Steel_fraction_TF*100:.3f} [%]", file=out)
        print(f"[O] Steel fraction CS                               : {Steel_fraction_CS*100:.3f} [%]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] Psi_PI      (Breakdown flux)                    : {ΨPI:.3f} [Wb]",      file=out)
        print(f"[O] Psi_RampUp  (Ramp-up flux)                      : {ΨRampUp:.3f} [Wb]",  file=out)
        print(f"[O] Psi_Plateau (Flat-top flux)                     : {Ψplateau:.3f} [Wb]", file=out)
        print(f"[O] Psi_PF      (External PF contribution)          : {ΨPF:.3f} [Wb]",      file=out)
        print(f"[O] Psi_CS      (CS flux requirement)               : {ΨCS_Total:.3f} [Wb]", file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] P_fus  (Fusion power)                           : {config.P_fus:.3f} [MW]",  file=out)
        print(f"[O] P_CD   (Current drive power)                    : {P_CD:.3f} [MW]",           file=out)
        print(f"[O]  \u251c gamma_LH  (LHCD efficiency)                 : {eta_LH:.4f} [MA/MW\u00b7m\u00b2]",  file=out)
        print(f"[O]  \u251c gamma_EC  (ECCD efficiency)                 : {eta_EC:.4f} [MA/MW\u00b7m\u00b2]",  file=out)
        print(f"[O]  \u2514 gamma_NBI (NBCD efficiency)                 : {eta_NBI:.4f} [MA/MW\u00b7m\u00b2]", file=out)
        print(f"[O]  \u251c P_LH  / I_LH  (LHCD)                       : {P_LH_out:.2f} MW / {I_LH_out:.3f} MA",  file=out)
        print(f"[O]  \u251c P_EC  / I_EC  (ECCD)                        : {P_EC_out:.2f} MW / {I_EC_out:.3f} MA",  file=out)
        print(f"[O]  \u251c P_NBI / I_NBI (NBCD)                        : {P_NBI_out:.2f} MW / {I_NBI_out:.3f} MA", file=out)
        print(f"[O]  \u2514 P_ICR        (ICRH, heating only)           : {P_ICR_out:.2f} MW",       file=out)
        print(f"[O] P_syn  (Synchrotron radiation power)            : {P_syn:.3f} [MW]",          file=out)
        print(f"[O] P_Brem (Bremsstrahlung power)                   : {P_Brem:.3f} [MW]",         file=out)
        print(f"[O] P_line (Impurity line radiation, profile-int.)  : {P_line:.3f} [MW]",         file=out)
        if imp_species_list:
            for _sp, _fc in zip(imp_species_list, imp_conc_list):
                _Pc, _Pt = f_P_line_radiation_profile(
                    _sp, _fc, nbar, config.Tbar, nu_n, nu_T, Volume,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                    Vprime_data=Vprime_data, rho_core=rho_rad_core)
                print(f"[O]  \u251c P_line_{_sp:<3s} (f={_fc:.1e})                     : {_Pt:.3f} [MW] "
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
        print(f"[O] Cost   ((V_BB+V_TF+V_CS) / P_fus)               : {cost:.3f} [m³/MW]",        file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] H              (H-factor)                       : {config.H:.3f}",             file=out)
        print(f"[I] Operation mode                                  : {config.Operation_mode}",    file=out)
        print(f"[I] t_plateau      (Flat-top duration)              : {config.Temps_Plateau_input:.3f} [s]", file=out)
        print(f"[O] tau_E          (Energy confinement time)        : {tauE:.3f} [s]",             file=out)
        print(f"[O] Ip             (Plasma current)                 : {Ip:.3f} [MA]",              file=out)
        print(f"[O] Ib             (Bootstrap current)              : {Ib:.3f} [MA]",              file=out)
        print(f"[O] I_CD           (Driven current)                 : {I_CD:.3f} [MA]",            file=out)
        print(f"[O] I_Ohm          (Ohmic current)                  : {I_Ohm:.3f} [MA]",           file=out)
        print(f"[O] f_b            (Bootstrap fraction)             : {(Ib/Ip)*100:.3f} [%]",      file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[I] Tbar  (Volume-averaged ion temperature)         : {config.Tbar:.3f} [keV]",    file=out)
        print(f"[O] nbar  (Volume-averaged electron density)        : {nbar:.3f} [10²⁰ m⁻³]",     file=out)
        print(f"[O] nbar_line (Line-averaged electron density)      : {nbar_line:.3f} [10²⁰ m⁻³]", file=out)
        print(f"[O] nG    (Greenwald density limit)                 : {nG:.3f} [10²⁰ m⁻³]",       file=out)
        print(f"[O] nbar_line/nG (Greenwald fraction)               : {nbar_line/nG:.3f} [-]",     file=out)
        print(f"[O] pbar  (Volume-averaged pressure)                : {pbar:.3f} [MPa]",            file=out)
        print(f"[O] f_He  (Helium ash fraction)                     : {f_alpha*100:.3f} [%]",      file=out)
        print(f"[O] tau_alpha (Alpha-particle confinement time)     : {tau_alpha:.3f} [s]",         file=out)
        print(f"[O] W_th  (Plasma thermal energy content)           : {W_th/1e6:.3f} [MJ]",        file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] beta_T (Toroidal beta)                          : {betaT*100:.3f} [%]", file=out)
        print(f"[O] beta_P (Poloidal beta)                          : {betaP:.3f}",         file=out)
        print(f"[O] beta_N (Normalised beta / Troyon factor)        : {betaN:.3f}",         file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] q*   (Kink safety factor)                       : {qstar:.3f}", file=out)
        print(f"[O] q95  (Safety factor at 95% flux surface)        : {q95:.3f}",   file=out)
        print("-------------------------------------------------------------------------", file=out)
        print(f"[O] P_sep    (Power crossing the separatrix)        : {P_sep:.3f} [MW]",    file=out)
        print(f"[O] P_Thresh (L-H power threshold)                  : {P_Thresh:.3f} [MW]", file=out)
        print(f"[O] (P_sep - P_thresh) / S_wall                     : {P_wall_H:.3f} [MW/m²]", file=out)
        print(f"[O] P_sep / S_wall                                  : {P_wall_L:.3f} [MW/m²]", file=out)
        print(f"[O] P_sep / R0  (Divertor heat scaling)             : {heat:.3f} [MW/m]",      file=out)
        print(f"[O] P_sep·B0 / R0  (Parallel heat flux proxy)       : {heat_par:.3f} [MW·T/m]", file=out)
        print(f"[O] P_sep·B0 / (q95·R0·A)  (Poloidal heat flux)     : {heat_pol:.3f} [MW·T/m]", file=out)
        print(f"[O] Gamma_n  (Neutron wall load)                    : {Gamma_n:.3f} [MW/m²]",   file=out)
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
    Generate and save run-specific figures (11) into output_path/figures/.

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