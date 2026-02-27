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


#%% Input file loader

def load_config_from_file(filepath: str,
                          base: GlobalConfig = None) -> GlobalConfig:
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
                print(f"  [warn]   '{key}' is not a recognised GlobalConfig field — ignored")
                continue

            overrides[key] = val
            print(f"  [input]  {key} = {val}")

    return dc_replace(base, **overrides)


#%% Core calculation
# ---------------------------------------------------------------------------

def run(config: GlobalConfig = None) -> tuple:
    """
    Compute a single D0FUS design point.

    Parameters
    ----------
    config : GlobalConfig, optional
        Complete design configuration.  If None, DEFAULT_CONFIG is used.
        To override selected parameters only, use dataclasses.replace():
            cfg = dc_replace(DEFAULT_CONFIG, R0=8.0, Bmax_TF=13.0)
            results = run(cfg)

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
    Volume_solution  = f_plasma_volume(R0, a, κ, δ)
    Surface_solution = f_surface_premiere_paroi(κ, R0, a)

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
    P_Alpha     = f_P_alpha(P_fus, E_ALPHA, E_N)

    # =========================================================================
    #    SELF-CONSISTENT SOLVER  (f_alpha, Q)
    #    Jointly determines the helium ash fraction and the energy gain factor
    #    through a fixed-point iteration on the power balance.
    # =========================================================================

    def to_solve_f_alpha_and_Q(vars):
        """
        Residual function for the coupled (f_alpha, Q) system.

        Returns the relative residuals [%] of:
          - helium ash fraction self-consistency
          - energy gain factor self-consistency
        """
        f_alpha, Q = vars

        # Sanity checks — return large residuals outside physical bounds
        if not (0 <= f_alpha <= 1) or Q < 0:
            return [1e10, 1e10]

        # Volume-averaged density and pressure
        nbar_alpha = f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, κ,
                              rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
        pbar_alpha = f_pbar(nu_n, nu_T, nbar_alpha, Tbar,
                              rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)

        # Radiative power losses
        P_Brem_alpha = f_P_bremsstrahlung(Volume_solution, nbar_alpha, Tbar, Zeff, R0, a)
        P_syn_alpha  = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_alpha, κ, nu_n, nu_T, r_synch,
                                        rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
        P_rad_alpha  = P_Brem_alpha + P_syn_alpha

        # Heating power initialisation
        if Operation_mode == 'Steady-State':
            P_Aux_alpha_init = abs(P_fus / Q)
            P_Ohm_alpha_init = 0
        elif Operation_mode == 'Pulsed':
            P_Aux_alpha_init = abs(P_aux_input)
            P_Ohm_alpha_init = abs(P_fus / Q - P_Aux_alpha_init)
        else:
            print("Choose a valid operation mode")

        # Energy confinement time (scaling-law estimate)
        tau_E_alpha = f_tauE(pbar_alpha, Volume_solution, P_Alpha,
                             P_Aux_alpha_init, P_Ohm_alpha_init, P_rad_alpha)

        # Plasma current from the scaling law inversion
        Ip_alpha = f_Ip(tau_E_alpha, R0, a, κ, δ, nbar_alpha, B0_solution, Atomic_mass,
                        P_Alpha, P_Ohm_alpha_init, P_Aux_alpha_init, P_rad_alpha,
                        H, C_SL,
                        alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
                        alpha_R, alpha_B, alpha_n, alpha_I, alpha_P)

        # Bootstrap current
        if Bootstrap_choice == 'Freidberg':
            Ib_alpha = f_Freidberg_Ib(R0, a, κ, pbar_alpha, Ip_alpha)
        
        elif Bootstrap_choice == 'Segal':
            Ib_alpha = f_Segal_Ib(nu_n, nu_T, a / R0, κ, nbar_alpha, Tbar, R0, Ip_alpha,
                                          rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
        
        elif Bootstrap_choice in ('Sauter', 'Redl'):
            # q95 and B0 must be estimated locally — not yet converged at this stage
            B0_alpha  = f_B0(Bmax_TF, a, b, R0)
            q95_alpha = f_q95(B0_alpha, Ip_alpha, R0, a, κ_95, δ_95)
            if Bootstrap_choice == 'Sauter':
                Ib_alpha = f_Sauter_Ib(R0, a, κ, B0_alpha, nbar_alpha, Tbar,
                                        q95_alpha, Zeff, nu_n, nu_T,
                                        rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
            else:
                Ib_alpha = f_Redl_Ib(R0, a, κ, B0_alpha, nbar_alpha, Tbar,
                                      q95_alpha, Zeff, nu_n, nu_T,
                                      rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
        
        else:
            raise ValueError(f"Unknown Bootstrap_choice: '{Bootstrap_choice}'. "
                             "Valid options: 'Freidberg', 'Segal', 'Sauter', 'Redl'.")

        # Current drive and Q evaluation
        eta_CD_alpha = f_etaCD(a, R0, B0_solution, nbar_alpha, Tbar, nu_n, nu_T,
                                 rho_ped=rho_ped, n_ped_frac=n_ped_frac)

        if Operation_mode == 'Steady-State':
            I_Ohm_alpha = 0
            P_Ohm_alpha = 0
            I_CD_alpha  = f_ICD(Ip_alpha, Ib_alpha, I_Ohm_alpha)
            P_CD_alpha  = f_PCD(R0, nbar_alpha, I_CD_alpha, eta_CD_alpha)
            Q_alpha     = f_Q(P_fus, P_CD_alpha, P_Ohm_alpha)

        elif Operation_mode == 'Pulsed':
            P_CD_alpha  = P_aux_input
            I_CD_alpha  = f_I_CD(R0, nbar_alpha, eta_CD_alpha, P_CD_alpha)
            I_Ohm_alpha = f_I_Ohm(Ip_alpha, Ib_alpha, I_CD_alpha)
            P_Ohm_alpha = f_P_Ohm(I_Ohm_alpha, Tbar, R0, a, κ)
            Q_alpha     = f_Q(P_fus, P_CD_alpha, P_Ohm_alpha)

        else:
            print("Choose a valid operation mode")

        # Self-consistent helium ash fraction
        new_f_alpha = f_He_fraction(nbar_alpha, Tbar, tau_E_alpha, C_Alpha, nu_T,
                                     rho_ped=rho_ped, T_ped_frac=T_ped_frac)

        epsilon = 1e-10  # Avoid division by zero
        f_alpha_residual = abs(new_f_alpha - f_alpha)  / (abs(new_f_alpha) + epsilon) * 100
        Q_residual       = abs(Q           - Q_alpha)  / (abs(Q_alpha)     + epsilon) * 100

        return [f_alpha_residual, Q_residual]

    def solve_f_alpha_Q():
        """
        Solve for (f_alpha, Q) with a progressive robustness strategy:
          1. Fast solver    — 'hybr'
          2. Robust solver  — 'lm'
          3. Safeguard      — 'df-sane'
          4. Grid search    — 'df-sane' (last resort)

        Returns the solution with the highest Q among all converged candidates.
        """

        def verify_solution(f_alpha, Q, residuals, tolerance=1e-2):
            """Return True if the candidate is physically valid and numerically converged."""
            if not (0 <= f_alpha <= 1 and Q >= 0):
                return False
            if abs(residuals[0]) > tolerance or abs(residuals[1]) > tolerance:
                return False
            return True

        def is_duplicate_solution(sol, solutions_list, tol_f_alpha=1e-2, tol_Q=1.0):
            """Return True if the candidate is already present in the solution list."""
            for existing in solutions_list:
                if (abs(existing['f_alpha'] - sol['f_alpha']) < tol_f_alpha and
                        abs(existing['Q']       - sol['Q'])       < tol_Q):
                    return True
            return False

        def select_best_solution(valid_solutions):
            """Return the (f_alpha, Q) pair that maximises Q."""
            if not valid_solutions:
                return None
            best = max(valid_solutions, key=lambda x: x['Q'])
            return (best['f_alpha'], best['Q'])

        def try_method_with_guesses(method_name, initial_guesses, tolerance=1):
            """
            Attempt root-finding with a given scipy method and multiple initial guesses.

            Parameters
            ----------
            method_name    : str   — scipy.optimize.root method identifier
            initial_guesses: list  — list of [f_alpha, Q] starting points
            tolerance      : float — maximum acceptable relative residual [%]

            Returns
            -------
            list of valid solution dictionaries
            """
            valid_solutions = []
            for guess in initial_guesses:
                try:
                    result = root(to_solve_f_alpha_and_Q, guess,
                                  method=method_name, tol=1e-8)
                    if result.success:
                        f_alpha, Q  = result.x
                        residuals   = to_solve_f_alpha_and_Q([f_alpha, Q])
                        if verify_solution(f_alpha, Q, residuals, tolerance=tolerance):
                            sol_dict = {
                                'f_alpha'      : f_alpha,
                                'Q'            : Q,
                                'residuals'    : residuals,
                                'residual_norm': np.sqrt(residuals[0]**2 + residuals[1]**2),
                                'method'       : method_name,
                                'guess'        : guess,
                            }
                            if not is_duplicate_solution(sol_dict, valid_solutions):
                                valid_solutions.append(sol_dict)
                except Exception:
                    continue
            return valid_solutions

        # Canonical starting points covering a wide range of physically expected Q values
        initial_guesses = [
            [0.05,   50],
            [0.05, 1000],
            [0.05,    1],
            [0.05, 5000],
        ]

        # ── Step 1: fast solver ───────────────────────────────────────────────
        solutions = try_method_with_guesses('hybr', initial_guesses, tolerance=1)
        best = select_best_solution(solutions)
        if best:
            return best

        # ── Step 2: robust solver ─────────────────────────────────────────────
        solutions = try_method_with_guesses('lm', initial_guesses, tolerance=1)
        best = select_best_solution(solutions)
        if best:
            return best

        # ── Step 3: safeguard solver ──────────────────────────────────────────
        solutions = try_method_with_guesses('df-sane', initial_guesses, tolerance=1)
        best = select_best_solution(solutions)
        if best:
            return best

        # ── Step 4: dense grid search (last resort) ───────────────────────────
        f_alpha_guesses = [0.001, 0.01, 0.1, 0.3, 0.5]
        Q_guesses       = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]
        grid_guesses    = [[fa, Qg] for fa in f_alpha_guesses for Qg in Q_guesses]

        solutions = try_method_with_guesses('df-sane', grid_guesses, tolerance=1)
        best = select_best_solution(solutions)
        if best:
            return best

        return np.nan, np.nan   # Convergence failure

    f_alpha_solution, Q_solution = solve_f_alpha_Q()

    # ── Return nan tuple on convergence failure ───────────────────────────────
    _nan = np.nan
    if np.isnan(f_alpha_solution) or np.isnan(Q_solution):
        return (
            _nan, _nan, _nan,                  # B0, B_CS, B_pol
            _nan, _nan,                          # tauE, W_th
            _nan, _nan, _nan,                  # Q, Volume, Surface
            _nan, _nan, _nan, _nan,            # Ip, Ib, I_CD, I_Ohm
            _nan, _nan, _nan,                  # nbar, nG, pbar
            _nan, _nan, _nan,                  # betaN, betaT, betaP
            _nan, _nan,                          # qstar, q95
            _nan, _nan, _nan, _nan, _nan,      # P_CD, P_sep, P_Thresh, eta_CD, P_elec
            _nan, _nan, _nan,                  # cost, P_Brem, P_syn
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
        )

    # =========================================================================
    #    POST-CONVERGENCE OUTPUTS
    #    Full calculation of all derived quantities after (f_alpha, Q) is known.
    # =========================================================================

    # Plasma thermodynamics
    nbar_solution   = f_nbar(P_fus, nu_n, nu_T, f_alpha_solution, Tbar, R0, a, κ,
                                  rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    pbar_solution   = f_pbar(nu_n, nu_T, nbar_solution, Tbar,
                                  rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    W_th_solution   = f_W_th(nbar_solution, Tbar, Volume_solution)

    # Radiative losses
    P_Brem_solution = f_P_bremsstrahlung(Volume_solution, nbar_solution, Tbar, Zeff, R0, a)
    P_syn_solution  = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_solution,
                                      κ, nu_n, nu_T, r_synch,
                                      rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    P_rad_solution  = P_Brem_solution + P_syn_solution

    # Heating power partition
    if Operation_mode == 'Steady-State':
        P_Aux_solution = P_fus / Q_solution
        P_Ohm_solution = 0
    elif Operation_mode == 'Pulsed':
        P_Aux_solution = P_aux_input
        P_Ohm_solution = P_fus / Q_solution - P_Aux_solution
    else:
        print("Choose a valid operation mode")

    # Energy confinement and alpha-particle confinement times
    tauE_solution = f_tauE(pbar_solution, Volume_solution, P_Alpha,
                           P_Aux_solution, P_Ohm_solution, P_rad_solution)
    tau_alpha = f_tau_alpha(nbar_solution, Tbar, tauE_solution, C_Alpha, nu_T,
                            rho_ped=rho_ped, T_ped_frac=T_ped_frac)

    # Plasma current
    Ip_solution = f_Ip(tauE_solution, R0, a, κ, δ, nbar_solution, B0_solution, Atomic_mass,
                       P_Alpha, P_Ohm_solution, P_Aux_solution, P_rad_solution, H, C_SL,
                       alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
                       alpha_R, alpha_B, alpha_n, alpha_I, alpha_P)

    # --- Compute MHD quantities needed by neoclassical bootstrap models ---
    qstar_solution  = f_qstar(a, B0_solution, R0, Ip_solution, κ)
    q95_solution    = f_q95(B0_solution, Ip_solution, R0, a, κ_95, δ_95)
    
    # Bootstrap current
    if Bootstrap_choice == 'Freidberg':
        Ib_solution = f_Freidberg_Ib(R0, a, κ, pbar_solution, Ip_solution)
    
    elif Bootstrap_choice == 'Segal':
        Ib_solution = f_Segal_Ib(nu_n, nu_T, a / R0, κ, nbar_solution, Tbar, R0, Ip_solution,
                                     rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    
    elif Bootstrap_choice == 'Sauter':
        Ib_solution = f_Sauter_Ib(R0, a, κ, B0_solution, nbar_solution, Tbar,
                                   q95_solution, Zeff, nu_n, nu_T,
                                   rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    
    elif Bootstrap_choice == 'Redl':
        Ib_solution = f_Redl_Ib(R0, a, κ, B0_solution, nbar_solution, Tbar,
                                 q95_solution, Zeff, nu_n, nu_T,
                                 rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    
    else:
        raise ValueError(f"Unknown Bootstrap_choice: '{Bootstrap_choice}'. "
                         "Valid options: 'Freidberg', 'Segal', 'Sauter', 'Redl'.")
    
    # --- Remaining MHD quantities (qstar/q95 already computed above) ---
    B_pol_solution  = f_Bpol(q95_solution, B0_solution, a, R0)
    betaT_solution  = f_beta_T(pbar_solution, B0_solution)
    B_pol_solution  = f_Bpol(q95_solution, B0_solution, a, R0)
    betaT_solution  = f_beta_T(pbar_solution, B0_solution)
    betaP_solution  = f_beta_P(a, κ, pbar_solution, Ip_solution)
    beta_solution   = f_beta(betaP_solution, betaT_solution)
    betaN_solution  = f_beta_N(betaT_solution, B0_solution, a, Ip_solution)
    nG_solution     = f_nG(Ip_solution, a) * Greenwald_limit

    # Current drive and power balance
    eta_CD_solution = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T,
                              rho_ped=rho_ped, n_ped_frac=n_ped_frac)

    if Operation_mode == 'Steady-State':
        I_Ohm_solution  = 0
        P_Ohm_solution  = 0
        I_CD_solution   = f_ICD(Ip_solution, Ib_solution, I_Ohm_solution)
        P_CD_solution   = f_PCD(R0, nbar_solution, I_CD_solution, eta_CD_solution)

    elif Operation_mode == 'Pulsed':
        P_CD_solution   = P_aux_input
        I_CD_solution   = f_I_CD(R0, nbar_solution, eta_CD_solution, P_CD_solution)
        I_Ohm_solution  = f_I_Ohm(Ip_solution, Ib_solution, I_CD_solution)
        P_Ohm_solution  = f_P_Ohm(I_Ohm_solution, Tbar, R0, a, κ)

    else:
        print("Choose a valid operation mode")

    # Separatrix and wall power loads
    P_sep_solution      = f_P_sep(P_fus, P_CD_solution)
    Gamma_n_solution    = f_Gamma_n(a, P_fus, R0, κ)
    heat_D0FUS_solution = f_heat_D0FUS(R0, P_sep_solution)
    heat_par_solution   = f_heat_par(R0, B0_solution, P_sep_solution)
    heat_pol_solution   = f_heat_pol(R0, B0_solution, P_sep_solution, a, q95_solution)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(
        P_sep_solution, B_pol_solution, R0, a / R0, theta_deg)
    P_1rst_wall_Hmod = f_P_1rst_wall_Hmod(P_sep_solution, P_CD_solution, Surface_solution)
    P_1rst_wall_Lmod = f_P_1rst_wall_Lmod(P_sep_solution, Surface_solution)
    P_elec_solution  = f_P_elec(P_fus, P_CD_solution, eta_T, eta_RF)
    li_solution = f_li(
        nu_T, nu_n, Tbar, nbar_solution,
        Zeff, a, R0, q95_solution,
        eta_model=eta_model,
        rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac
    )

    # L-H power threshold
    if L_H_Scaling_choice == 'Martin':
        P_Thresh = P_Thresh_Martin(nbar_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_S':
        P_Thresh = P_Thresh_New_S(nbar_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_Ip':
        P_Thresh = P_Thresh_New_Ip(nbar_solution, B0_solution, a, R0, κ, Ip_solution, Atomic_mass)
    else:
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
            nbar_solution,  nG_solution,  pbar_solution,
            betaN_solution,  betaT_solution,  betaP_solution,
            qstar_solution,  q95_solution,
            P_CD_solution,  P_sep_solution,  P_Thresh,  eta_CD_solution,  P_elec_solution,
            cost_solution,  P_Brem_solution,  P_syn_solution,
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
            ΨPI,  ΨRampUp,  Ψplateau,  ΨPF,  ΨCS)


#%% Output writer
# ---------------------------------------------------------------------------

def save_run_output(config: GlobalConfig,
                    results: tuple,
                    output_dir: str,
                    input_file_path: str = None) -> str:
    """
    Write a timestamped output directory containing:
      - a copy (or reconstruction) of the input parameters
      - a complete human-readable results report

    Parameters
    ----------
    config         : GlobalConfig — configuration used for this run
    results        : tuple        — return value of run()
    output_dir     : str          — base output directory
    input_file_path: str, optional — original input file to copy verbatim

    Returns
    -------
    str : absolute path to the created output directory
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
     nbar, nG, pbar,
     betaN, betaT, betaP,
     qstar, q95,
     P_CD, P_sep, P_Thresh, eta_CD, P_elec,
     cost, P_Brem, P_syn,
     heat, heat_par, heat_pol, lambda_q, q_target,
     P_wall_H, P_wall_L,
     Gamma_n,
     f_alpha, tau_alpha,
     J_TF, J_CS,
     c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
     d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS,
     r_minor, r_sep, r_c, r_d,
     κ, κ_95, δ, δ_95,
     ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS_Total) = results

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
        print(f"[O] P_syn  (Synchrotron radiation power)            : {P_syn:.3f} [MW]",          file=out)
        print(f"[O] P_Brem (Bremsstrahlung power)                   : {P_Brem:.3f} [MW]",         file=out)
        print(f"[O] eta_CD (Current drive efficiency)               : {eta_CD:.3f} [MA/MW·m²]",   file=out)
        print(f"[O] Q      (Energy gain factor)                     : {Q:.3f}",                   file=out)
        print(f"[O] P_elec (Net electrical power)                   : {P_elec:.3f} [MW]",         file=out)
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
        print(f"[O] nG    (Greenwald density limit)                 : {nG:.3f} [10²⁰ m⁻³]",       file=out)
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

    print(f"\n✓ Results saved to: {output_path}\n")
    return output_path


#%% Main entry point
# ---------------------------------------------------------------------------

def main(input_file: str = None) -> tuple:
    """
    Load a configuration, run the calculation, and save the output report.

    Parameters
    ----------
    input_file : str, optional
        Path to a plain-text input file.  If None, the module looks for
        ``D0FUS_INPUTS/default_input.txt`` relative to the package root.
        Missing files trigger a warning and the code runs with DEFAULT_CONFIG.

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
        print(f"\nLoading parameters from: {input_file}")
        config = load_config_from_file(input_file)
    else:
        print("\nWarning: input file not found — running with DEFAULT_CONFIG.")
        config = DEFAULT_CONFIG
        input_file_path = None

    # ── Launch calculation ────────────────────────────────────────────────────
    print("\n" + "=" * 73)
    print("Starting D0FUS calculation...")
    print("=" * 73 + "\n")

    try:
        results    = run(config)
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        save_run_output(config, results, output_dir, input_file_path)
        return results

    except Exception as e:
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