"""
D0FUS Parameterization Module
==============================
Physical constants, material properties, and default parameters for tokamak design.

Created: December 2023
Author: Auclair Timothé
"""

#%% Imports

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *

#%% D0FUS Physical Constants
"""
Fundamental physical constants. These values are fixed by nature
and must never be modified by the user.
"""

# ── Fundamental constants ──────────────────────────────────────────────────────
E_ELEM  = 1.602176634e-19    # Elementary charge [C]         (CODATA 2018)
M_E     = 9.10938370e-31     # Electron mass [kg]            (CODATA 2018)
M_I     = 2 * 1.6726e-27     # Ion mass (deuterium) [kg]
μ0      = 4.0 * np.pi * 1e-7 # Vacuum permeability [H/m]
EPS_0   = 8.854187817e-12    # Vacuum permittivity [F/m]     (CODATA 2018)
C_LIGHT = 2.99792458e8       # Speed of light [m/s]          (CODATA 2018)

# ── Fusion reaction energetics ─────────────────────────────────────────────────
E_ALPHA = 3.5168e6  * E_ELEM   # Alpha particle energy [J]
E_N     = 14.0671e6 * E_ELEM   # Neutron energy [J]

#%% D0FUS Global Configuration
"""
Centralised repository of all user-adjustable design parameters,
including primary scan variables (geometry, field, power) and
physical/engineering assumptions.
"""

@dataclass
class GlobalConfig:
    """
    Complete configuration object for a D0FUS design point.

    All fields carry physically motivated default values corresponding
    to a conservative DEMO-class power plant baseline.

    Usage
    -----
    Full default run:
        cfg = GlobalConfig()

    Override selected parameters only (all others stay at default):
        cfg = GlobalConfig(R0=8.0, Bmax=13.0, Supra_choice='REBCO')

    Create a variant from an existing config (useful for scans):
        from dataclasses import replace
        cfg2 = replace(cfg, R0=9.0)

    Notes
    -----
    Parameters are grouped into:
      1. Primary scan variables     – geometry, field, power, confinement
      2. Physics & operation mode   – profiles, bootstrap, scaling laws
      2b. Plasma geometry model     – Academic vs Miller (D0FUS) volume integrals
      3. Plasma stability limits    – beta, q, Greenwald
      4. Plasma composition         – Zeff, mass, radiation, impurities
      5. Magnetic flux model        – Ejima constant, resistivity
      6. Structural materials       – steel properties, fatigue
      7. TF coil engineering        – stress partition, backplate
      8. Central Solenoid           – clearance, sub-modules
      9. Superconductor conditions  – temperatures, fractions, quench
     10. Power conversion           – thermal efficiency, blanket multiplication
     11. Multi-source current drive – source powers, fractions, deposition radii
     12. Plasma-facing components   – divertor geometry
     13. Maintenance constraints    – ripple, port access
     14. Disruption RE diagnostic   – thermal quench parameters (indicative)
     15. Techno-economic cost model – Sheffield (2016), post-convergence
    """

    # ── 1. Primary scan variables ────────────────────────────────────────────
    R0    : float = 9.0            # Major plasma radius [m]
    a     : float = 3.0            # Minor plasma radius [m]
    b     : float = 1.2            # Breeding blanket + neutron shield radial thickness [m]
    Bmax_TF  : float = 12.0        # Peak magnetic field on TF conductor [T]
    Bmax_CS_adm  : float = 25.0    # Peak magnetic field allowed on the CS [T]
    P_fus : float = 2000.0         # Total fusion power [MW]
    Tbar  : float = 14.0           # Volume-averaged electron temperature T_e [keV]
    Tbar_mode : str = 'manual'
    # Temperature prescription mode:
    #   'manual'    : Tbar above is used directly (historical behaviour).
    #   'greenwald' : Tbar is SOLVED by scalar root-finding (brentq) so that
    #                 the converged operating point sits at the requested
    #                 Greenwald fraction f_GW_target = nbar_line / (Ip/pi a^2).
    #                 The search bracket is [Tbar_min, Tbar_max]; the solve
    #                 wraps the full run() (one design solve per iteration).
    f_GW_target : float = 0.85   # Target Greenwald fraction (Tbar_mode='greenwald') [-]
    Tbar_min    : float = 4.0    # Lower Tbar bracket for the f_GW solve [keV]
    Tbar_max    : float = 30.0   # Upper Tbar bracket for the f_GW solve [keV]
    tau_i_e : float = 1.0          # Ion-to-electron temperature ratio T_i/T_e [-]
                                   # 1.0 -> single-temperature plasma (T_i = T_e).
                                   # Prescribed: T_i(rho) = tau_i_e * T_e(rho),
    H     : float = 1.0            # Confinement enhancement factor (H-factor) [-]

    # ── 2. Physics and operation ─────────────────────────────────────────────
    Operation_mode     : str   = 'Pulsed'       # 'Steady-State' or 'Pulsed'
    Temps_Plateau_input: float = 3600.0         # Flat-top duration (pulsed only) [s]
    P_aux_input        : float = 50.0           # Auxiliary heating power (pulsed only) [MW]

    Plasma_profiles    : str   = 'H'            # Profile peaking: 'L', 'H', 'Advanced', 'Manual'
    nu_n_manual        : float = 0.1            # Density peaking factor (Manual mode only) [-]
    nu_T_manual        : float = 1.0            # Temperature peaking factor (Manual mode only) [-]
    rho_ped    : float = 1.0                    # Normalised pedestal radius (1.0 = no pedestal) [-]
    n_ped_frac : float = 0.0                    # n_ped / nbar [-]
    T_ped_frac : float = 0.0                    # T_ped / Tbar [-]

    Scaling_Law        : str   = 'IPB98(y,2)'   # Energy confinement scaling law
    L_H_Scaling_choice : str   = 'New_Ip'       # L-H threshold scaling: 'Martin', 'New_S', 'New_Ip'
    Bootstrap_choice    : str   = 'Sauter-Redl' # Bootstrap current model 'Segal' or 'Sauter-Redl'
    trapped_fraction_model : str = 'Sauter2002' # 'Sauter2002' (standard, NEOS/JINTRAC)
                                                # 'ASTRA' (Fable, used by PROCESS/ASTRA)

    # q₉₅ formula selector:
    #   'Sauter'    — uses LCFS values (κ_edge, δ_edge). Sauter, FED 112 (2016) Eq. 30.
    #   'ITER_1989' — uses ψ_N = 0.95 values (κ₉₅, δ₉₅). Uckan (1989), also Johner (2011).
    Option_q95         : str = 'Sauter'         # q₉₅ formula: 'Sauter' (default) or 'ITER_1989'
    Option_Kappa       : str = 'Wenninger'      # Elongation model: 'Wenninger', 'Stambaugh', 'Freidberg', 'Manual'
    κ_manual           : float = 1.9            # Elongation (Manual mode only) [-]

    # ── 2a. Safety factor and current density profiles ────────────────────────
    # Two strictly distinct philosophies for q(ρ) and j(ρ):
    #
    #   'academic' : Parametric ansatz j(ρ) ∝ (1 − ρ²)^alpha_J (PROCESS/Uckan).
    #                The user prescribes alpha_J as a pedagogical input.
    #                Cylindrical Ampère then yields q(ρ) analytically with
    #                q(ρ_95) = q95 imposed as edge normalisation.
    #                No coupling to bootstrap, CD, or Ohmic physics.
    #                Reference: Uckan IPDG89, Kovari Fus.Eng.Des. 89 (2014).
    #
    #   'refined'  : Self-consistent Picard iteration on q(ρ) itself.
    #                The composite j_total = j_Ohm(σ_neo, T) + j_CD(deposition)
    #                + j_bs(Sauter-Redl) is integrated through Ampère to yield
    #                q(ρ) at every iteration; q(ρ) feeds back into σ_neo and
    #                Sauter-Redl coefficients until convergence.
    #                No imposed shape, no parametric exponent.  Reversed shear
    #                allowed; q(ρ_95) ≠ q95 from f_q95 in general (the latter
    #                is an MHD-stability scaling, not a profile constraint).
    #
    # alpha_J is consumed only by the 'academic' branch.  Default 1.5 follows
    # the IPDG89 / PROCESS convention (Hender 1992) and gives l_i(3) ≈ 1.08.
    q_profile_mode : str   = 'refined'   # 'academic' or 'refined'
    alpha_J        : float = 1.5         # Current peaking exponent (academic mode)

    # ── 2b. Plasma geometry model ────────────────────────────────────────────
    # Controls volume integrals and cross-section area elements.
    # 'Academic' : Cylindrical-torus approx, V = 2π²R₀κa²
    # 'refined'  : Full Miller flux-surface parameterisation (Miller 1998).
    #              Numerically computed V'(ρ) with radial κ(ρ), δ(ρ) profiles.
    Plasma_geometry : str = 'refined'           # 'Academic' or 'refined'

    # ── 3. Plasma stability limits ───────────────────────────────────────────
    betaN_limit     : float = 2.8  # Troyon normalised beta limit [% m T/MA]
    # Kink safety factor limit.
    # Parameter used for the kink stability constraint.
    #   'q_star' — cylindrical safety factor q*.
    #   'q95'    — safety factor at 95% poloidal flux.
    kink_parameter  : str   = 'q95'  # 'q_star' or 'q95'
    # The constraint enforces q_kink > q_limit, where q_kink is either q* or
    # q95 depending on kink_parameter.
    #   kink_parameter='q_star' → q_limit ≈ 2.0–2.5  (Freidberg 2015: q*>2)
    #   kink_parameter='q95'    → q_limit ≈ 3.0–3.5  (ITER/EU-DEMO practice)
    q_limit         : float = 3.0  # Kink safety factor threshold [-]
    Greenwald_limit : float = 1.0  # Density-limit fraction (margin) applied to the
                                   # selected density-limit model [-]
    density_limit_model : str = 'greenwald'
    # Density-limit model used by the feasibility constraint (nbar_line < limit):
    #   'greenwald' : n_GW = Ip/(pi a^2)            [Greenwald, PPCF 44 (2002) R27]
    #   'giacomin'  : power-dependent edge limit    [Giacomin et al., PRL 128 (2022)
    #                 185003, Eq. 12], converted to a line-averaged cap through
    #                 n_sep = f_n_sep_line * nbar_line.
    #   'zanca'     : power-balance line-avg limit  [Zanca et al., NF 59 (2019)
    #                 126011, Eq. 20].
    alpha_giacomin : float = 3.3   # Giacomin fitted prefactor (3.3 +/- 0.3)
    f0_zanca       : float = 0.5   # Zanca effective neutral concentration [%]
    Ip_limit        : float = None # Upper bound on plasma current [MA]; None = no ceiling (GA)
    ms              : float = 0.3  # Vertical stability margin parameter [-]

    # ── 4. Plasma composition ────────────────────────────────────────────────
    Atomic_mass : float = 2.5   # Volume-averaged ionic mass [AMU]  (D-T: 2.5)
    Zeff        : float = None   # Effective plasma charge [-].
                                 # None  -> computed self-consistently from the impurity
                                 #          inventory (impurity_species + f_imp_core) and the
                                 #          helium ash fraction (see _compute_Zeff_effective):
                                 #              Z_eff = 1 + 2 f_He - sum_j <Z_j> c_j
                                 #                                  + sum_j <Z_j>^2 c_j
                                 #          An empty impurity inventory therefore yields a clean
                                 #          D-T+He plasma with Z_eff ~ 1 + 2 f_He (~1.1).
                                 # float -> manual override (legacy behaviour, e.g. Zeff = 2.0).
    r_synch     : float = 0.5   # Synchrotron radiation wall reflectivity [-]
    C_Alpha     : float = 7.0   # Helium ash dilution tuning parameter [-] (default taken from PROCESS)
    # Impurity line radiation (0D bulk-plasma estimate).
    # Comma-separated species ('W', 'Ar', 'Ne', 'C', 'N', 'Kr') with matching
    # concentrations n_imp/n_e. Empty string = disabled (pure D-T).
    # Typical: W 1e-5–5e-4, Ar/Ne 1e-3–5e-3.
    impurity_species : str = ''     # Comma-separated species: 'W', 'W, Ne', '' = none
    detachment_impurity : str = 'Ne'
    # Seeding species used by the Lengyel detachment diagnostic ('N', 'Ne'
    # or 'Ar'; '' disables the diagnostic). The required SOL concentration
    # to radiate the two-point-model power-loss fraction f_pwr_loss_req is
    # reported in the run output (diagnostic only, no constraint).
    lengyel_T_target_eV : float = 25.0
    # Desired post-seeding target electron temperature [eV] for the Lengyel
    # integral lower bound (cfspopcon SPARC PRD convention: 25 eV). This is
    # the DESIRED detached/low-recycling target condition, deliberately
    # decoupled from the two-point-model operating T_et (which can be
    # sheath-limited at T_u when the operating point is unmitigated).
    f_imp_core       : str = ''     # Matching concentrations: '5e-5', '1e-5, 3e-3'
    # Core/edge radiation boundary for τ_E and P_sep convention.
    # ρ < rho_rad_core → subtracted from P_heat (core). ρ > → edge (divertor load).
    # Set to 1.0 to recover legacy behaviour (all radiation subtracted).
    rho_rad_core : float = 0.75  # Core/edge radiation boundary (normalised radius) [-]
    # Fraction of computed core radiation (ρ < rho_rad_core) subtracted from
    # P_loss in the confinement power balance (τ_E, scaling-law inversion):
    # P_loss = P_α + P_aux + P_Ohm − coreradiationfraction × P_rad_core.
    # Default: 1.0 (conservative). PROCESS convention: 0.6 (Kovari 2014).
    coreradiationfraction : float = 1.0

    # ── 5. Magnetic flux model ───────────────────────────────────────────────
    Ce        : float = 0.30     # Ejima constant (resistive ramp-up flux) [-]
                                 # Ψ_res = Ce * μ0 * R0 * Ip
                                 # Typical: ITER 0.45, EU-DEMO 0.30, CFETR 0.30
    f_heat_ramp : float = 0.0    # Fraction of current ramp-up driven non-inductively
                                 # by auxiliary heating/CD systems [-].
                                 # Reduces the CS volt-second requirement:
                                 # Ψ_RampUp_eff = (1 - f_heat_ramp) × Ψ_RampUp
    eta_model : str   = 'redl'   # Plasma resistivity model for R_eff and li
                                 # 'old' | 'spitzer' | 'sauter' | 'redl' (recommended)
                                 # Plasma initiation flux: Ψ_PI = 2π * R0 * E_BD
    E_BD      : float = 0.25     # Breakdown calibration parameter [V.s/m]
                                 # Product of E_phi [V/m] and t_BD [s].
                                 # Calibrated on ITER: Ψ_PI ~ 10 Wb => E_BD ~ 0.26.
                                 # Ref: Lloyd et al., PPCF 33(11), 1991.
                                 
    # ── 6. Structural materials ──────────────────────────────────────────────
    Chosen_Steel         : str   = '316L'   # Structural steel grade '316L' , 'N50H', 'Manual'
    σ_manual             : float = 1500.0   # Manual steel yield strength [MPa] (used only if Chosen_Steel='Manual')
    nu_Steel             : float = 0.29     # Steel Poisson's ratio [-] (only used in CIRCE model)
    Young_modul_Steel    : float = 200e9    # Steel Young's modulus [Pa] (only used in CIRCE model)
    Young_modul_GF       : float = 90e9     # S-glass fiber Young's modulus [Pa] (only used in CIRCE model)
    fatigue_CS           : float = 2.0      # CS fatigue knockdown factor (pulsed & wedging only) [-]
    # ── Steel allowable safety factors ───────────────────────────────────────
    # Dimensionless knockdown applied to the steel mechanical allowable inside
    # the TF and CS thickness solvers: σ_eff = σ_allowable / SF.
    # Captures effects not represented by the idealised CICC area model:
    #   – Realistic winding-pack filling factor: round/square cables inside a
    #     rectangular envelope leave 15–30% of the WP cross-section occupied
    #     by ground insulation, inter-pancake plates, helium manifolds,
    #     terminations and assembly clearances. The structural jacket carries
    #     the electromagnetic load through a section reduced by this factor.
    #   – Stress concentrations at jacket corners and transitions (K_t ≈ 1.2–1.5).
    #   – Weld efficiency on jacket seams (η_w ≈ 0.85–0.90).
    #   – Manufacturing thickness tolerances (≈ ±5%).
    # Default 1.0 preserves backward compatibility (raw material allowable used
    # directly). Realistic engineering value for both TF and CS large-magnet
    # CICC designs is ≈ 1.5 (composition of the four contributions above).
    # The CS fatigue knockdown (fatigue_CS) multiplies on top of SF_CS, since
    # the two factors address disjoint phenomena: primary static stress (SF)
    # versus cyclic damage accumulation (fatigue_CS).
    SF_TF                : float = 1.0      # Safety factor on TF steel allowable [-] (suggested ≈ 1.5 for realistic CICC packing)
    SF_CS                : float = 1.0      # Safety factor on CS steel allowable [-] (suggested ≈ 1.5 for realistic CICC packing)

    # ── 7. TF coil engineering ───────────────────────────────────────────────
    Radial_build_model   : str   = 'refined' # Stress model: 'academic', 'refined', 'CIRCE'
    Choice_Buck_Wedg     : str   = 'Wedging' # Mechanical config: 'Wedging', 'Bucking', 'Plug'
    Supra_choice         : str   = 'Nb3Sn'   # Superconductor: 'NbTi', 'Nb3Sn', 'REBCO', 'Manual'
    J_wost_Manual        : float = 100e6     # Engineering current density (Manual SC only) [A/m²]

    coef_inboard_tension : float = 0.5      # Inboard/outboard vertical stress ratio [-]
    F_CClamp             : float = 0.0      # C-clamp structural limit [N]
                                            # Typical: 30e6 (DDD ITER) to 60e6 N (Bachmann 2023, FED)
    n_shape_TF           : float = 1.0      # TF conductor shape factor (1 = square, 0 = optimal) [-]
    c_BP                 : float = 0.07     # Backplate thickness [m]
    TF_grading           : bool  = False    # TF WP conductor grading: α(R) varies to saturate Tresca [-]
    f_TF_steel_mass      : float = 2.0      # Multiplicative factor on total TF steel mass to account
                                            # for geometry approximations, gravitational supports, and
                                            # inter-coil structures [-]. Default = 2.0 (benchmarked
                                            # against ITER TF coil set total steel mass).

    # ── 8. Central Solenoid ──────────────────────────────────────────────────
    Gap      : float = 0.10      # CS–TF mechanical clearance [m]
    n_shape_CS   : float = 1.0   # CS conductor shape factor (1 = square, 0 = optimal) [-]
    N_sub_CS : int   = 6         # Number of CS sub-modules [-]
    H_CS : float = None          # CS total height [m]. None = 2(κa + b + 1) (default formula)
    f_swing_usable : float = 0.75   # Fraction of the CS bipolar swing usable for the
                                    # inductive flux budget [-]. The CS hardware swings
                                    # over the full bipolar range (-I_max → +I_max) and
                                    # delivers a total flux capacity ΨCS^capacity. A
                                    # fraction (1 - f_swing_usable) of this capacity is
                                    # reserved for plasma control during the discharge
                                    # (vertical stabilization, error-field correction,
                                    # shape feedback). The plasma inductive budget is
                                    # therefore ΨCS^plasma = f_swing_usable × ΨCS^capacity,
                                    # equivalently the hardware capacity demand is
                                    #   ΨCS^capacity = ΨCS^plasma / f_swing_usable.
                                    # Default 0.75 = 25% reserve.
                                    # Set to 1.0 to reproduce existing-machine benchmarks
                                    # where published flux values already correspond to
                                    # the hardware full-swing capacity.

    # ── 9. Superconductor operating conditions ───────────────────────────────
    T_helium  : float = 4.2     # Liquid helium bath temperature [K]
    Marge_T_He: float = 0.3     # Temperature margin from 10-bar He operation [K]
    f_He_pipe : float = 0.10    # Helium cooling pipe/channel fraction in wost [-]
    f_void    : float = None    # Interstitial void fraction in strand bundle [-]
                                # None = auto: 0.33 (LTS) / 0.00 (REBCO).
                                # Set explicitly (e.g. 0.30) to override.
    f_In      : float = 0.05    # Insulation area fraction [-]
    # Winding-pack overhead fraction in wost, treated exactly like the
    # helium fraction: ground and inter-pancake insulation plus assembly
    # clearances that occupy volume but carry neither current nor load.
    f_gap     : float = 0.15    # WP insulation + clearance fraction in wost [-]

    # Temperature margins above T_helium defining T_operating [K]
    # Conservative baseline: Corato et al., "Common operating values for DEMO…" (2016)
    Marge_T_Nb3Sn : float = 1.5    # Nb₃Sn temperature margin [K]
    Marge_T_NbTi  : float = 1.5    # NbTi temperature margin [K]
    Marge_T_REBCO : float = 5.0    # REBCO temperature margin [K]

    # Strand / tape operating parameters
    Eps : float = -6e-3  # Effective axial strain for Nb₃Sn [-]
                         # Ref: Corato et al. (2016)
    Tet : float = 0.0    # REBCO tape field angle [rad]  (0 = B⊥tape, π/2 = B∥ab-plane)

    # Quench protection
    I_cond                   : float = 50e3   # Nominal conductor current [A]
    V_max                    : float = 10e3   # Maximum terminal voltage (= I × R_dump) [V]
    tau_h_LTS                : float = 3.0    # Detection + hold time, LTS magnets [s]
    tau_h_HTS                : float = 10.0   # Detection + hold time, HTS magnets [s]
    T_hotspot                : float = 250.0  # Maximum hot-spot temperature [K]
                                              # It is taken as an equivalent of 150 K real hotspot + external contributions
    RRR                      : float = 100.0  # Copper residual resistivity ratio [-] (pessimistic)
    Dump_resistor_subdivision: int   = 2      # TF coils per dump resistor [-] (ITER reference)

    # ── 10. Power conversion ─────────────────────────────────────────────────
    eta_T     : float = 0.40   # Thermal-to-electric conversion efficiency [-] (0.35–0.45)
    M_blanket : float = 1.0    # Blanket energy multiplication factor [-] (1.0–1.3)
    eta_RF    : float = 0.4    # Wall-plug efficiency for Multi CD modes [-] (reserved)

    # ── 11. Auxiliary heating and current drive ──────────────────────────────
    # Technology-specific models ('LHCD', 'ECCD', 'NBCD', 'Multi') are under development
    CD_source   : str   = 'Academic'  # CD model: 'Academic' | 'LHCD' | 'ECCD' | 'NBCD' | 'Multi'
    # ── Academic (technology-agnostic) heating/CD ────────────────────────────
    # I_CD = γ_CD_acad × P_aux / (R₀ × n̄_e);  P_wallplug = P_aux / η_WP_acad
    gamma_CD_acad : float = 0.20   # CD figure of merit [MA/(MW·m²)] (0.15–0.40 typical)
    eta_WP_acad   : float = 0.40   # Wall-plug efficiency [-] (0.30–0.60 typical)
    # --- Pulsed mode: fixed plasma power per source [MW] ---
    # Used when Operation_mode = 'Pulsed' and CD_source = 'Multi'.
    # P_CD_total = P_LH + P_ECRH + P_NBI + P_ICRH  (P_aux_input ignored).
    P_LH   : float = 0.0      # LHCD plasma power [MW]
    P_ECRH : float = 0.0      # ECRH plasma power [MW]
    P_NBI  : float = 0.0      # NBI injected power[MW]
    P_ICRH : float = 0.0      # ICRH plasma power [MW]   (heating only, no current drive)
    # --- Steady-State mode: power fractions per source [-] ---
    # Used when Operation_mode = 'Steady-State' and CD_source = 'Multi'
    # The solver determines P_CD_total from the current-drive requirement
    # individual powers are then P_i = (f_heat_i / Σf) * P_CD_total
    # ICRH fraction contributes to heating but not to current drive (γ_ICR = 0)
    f_heat_LH  : float = 0.25   # Heating fraction allocated to LHCD [-]
    f_heat_EC  : float = 0.25   # Heating fraction allocated to ECCD [-]
    f_heat_NBI : float = 0.25   # Heating fraction allocated to NBCD [-]
    f_heat_ICR : float = 0.25   # Heating fraction allocated to ICRH [-]
    # ECCD deposition and injection parameters
    rho_EC : float = 0.3      # Normalised deposition radius [-]
    theta_EC_pol_deg : float = 0.0  # Poloidal angle of EC deposition [deg]
                                    # 0 = outboard midplane (LFS, conservative)
                                    # 90 = top of plasma
                                    # 180 = inboard midplane (HFS, best for CD)
    # NBCD deposition and beam parameters
    rho_NBI    : float = 0.3      # Normalised deposition radius [-]
    A_beam     : int   = 2        # Beam ion mass number: 1 = H, 2 = D, 3 = T
    E_beam_keV : float = 500.0    # Beam injection energy [keV]
    angle_NBI_deg : float = 20.0  # NBI injection angle from tangential [deg]
                                  # 0 = fully tangential, 90 = perpendicular
                                  # ITER reference ~ 20 deg

    # ── 12. Plasma-facing components ─────────────────────────────────────────
    theta_deg : float = 2.7      # Divertor strike-point grazing angle [deg]

    # Refined divertor exhaust (two-point model, Stangeby 2018). All used only
    # in the post-convergence f_heat_two_point evaluation; they do not feed the
    # core power balance.
    f_n_sep       : float = 0.20  # n_sep / volume-average density n̄ [-].
                                  # Groth value taken from https://arxiv.org/pdf/2406.15693
    f_cooling_div : float = 0.0   # SOL volumetric power-loss fraction [-] (2PM input)
    f_mom_div     : float = 0.0   # SOL volumetric momentum-loss fraction [-] (2PM input)
    q_dep_limit   : float = 5.0   # Tolerable deposited target heat flux [MW/m²]
    flux_expansion: float = 1.0   # Target/upstream flux expansion R_t/R_u [-]

    # ── 13. Maintenance constraints ──────────────────────────────────────────
    ripple_adm : float = 0.01    # Admissible toroidal field ripple [-]  (default 1%)
    L_min      : float = 3.00    # Minimum toroidal maintenance access width [m]

    # ── 14. Disruption RE diagnostic (indicative, post-convergence) ──────────
    tau_TQ       : float = 1e-3  # Thermal quench e-folding time [s] (0.1–3 ms typical)
    Te_final_eV  : float = 5.0   # Post-TQ residual Te [eV] (2–20 eV typical)
    pellet_dilution : float = 10.0   # SPI density multiplication [-] (1=unmitigated, 10–30=effective)
    pellet_dilution_cools : bool = False  # True: isobaric cooling before TQ (T ∝ 1/dilution)

    # ── 15. Techno-economic cost model (indicative, post-convergence) ────────
    # Model: Sheffield & Milora, Fus. Sci. Technol. 70 (2016).
    cost_model          : str   = 'Sheffield'  # 'Sheffield' or 'None'
    # Financial parameters (Ref: Sheffield 2016, Section III.D)
    discount_rate       : float = 0.07   # Real discount rate [-]
    T_life              : int   = 40     # Plant operational lifetime [yr]
    T_build             : int   = 10     # Construction phase duration [yr]
    contingency         : float = 0.15   # Contingency fraction (owner's cost, risk) [-]
    # Availability & capacity factor
    Util_factor         : float = 0.85   # Utilisation factor [-]
    Dwell_factor        : float = 1.0    # Dwell factor (1.0 = SS, <1 for pulsed) [-]
    # Superconductor cost multiplier (1.5 Nb3Sn mature — 3.0 REBCO FOAK)
    Supra_cost_factor   : float = 2.0    # SC coil cost multiplier vs Cu [-]
    # Budget constraint (genetic algorithm)
    # Designs exceeding C_invest_max are penalised. Set to 1e6 to disable.
    C_invest_max        : float = 25e3   # Capital cost ceiling [M EUR]

    # ── 17. Radial build sublayer widths ─────────────────────────────────────
    # b = total plasma→TF radial gap (drives all existing machinery).
    # Per-concept layer widths (SOL, FW, breeder, structure, shields, VV, gaps)
    # are defined in BLANKET_CONCEPTS[Blanket_choice]['radial_layers']; only the
    # SOL/FW elongation factor remains a global geometric parameter.
    f_kappa_SOL  : float = 0.25   # Elongation increase factor for SOL/FW shapes [-]

    # ── Blanket concept selection ─────────────────────────────────────────────
    # Selects the breeder/coolant/structure/multiplier combination and the
    # breeding-blanket (BB) sub-layer breakdown from BLANKET_CONCEPTS below.
    # Options: 'HCPB', 'HCLL', 'DCLL', 'SCLL', 'SCLV', 'F-LIB'
    # Ref: "Infinity Two pilot plant blanket trade study",
    #      J. Plasma Phys. 91, E79 (2025), doi:10.1017/S002237782500039X
    Blanket_choice : str = 'HCPB'

    # ── Effective volumetric densities for radial-build mass estimates ────────
    # Plasma->TF radial_layers densities are derived per layer from their
    # 'composition' (volume fractions) and BLANKET_MATERIAL_DENSITIES; see
    # BLANKET_CONCEPTS[Blanket_choice]['radial_layers'].
    rho_divertor : float = 13000.0   # Divertor             [kg/m³]

    # ── 16. Component lifetime & availability model ───────────────────────────
    # Blanket lifetime: t_bl [fpy] = dpa_lim * A_FW / (0.8 * C_dpa * P_fus)
    # Ref: Gilbert et al. (2013); EUROfusion (2015)
    dpa_lim             : float = 70.0   # Blanket structural damage limit [dpa]
    C_dpa               : float = 12.0   # dpa rate per neutron wall load [dpa fpy⁻¹ / (MW m⁻²)]
    # Divertor lifetime: t_div [fpy] = epsilon_div * A_div / (f_peak * P_sep)
    # A_div = f_div_area_fraction * A_FW (first-wall area)
    # Ref: ITER Organization (2025); CEA IRFM (2017)
    epsilon_div         : float = 30.0   # Divertor integrated heat limit [MW yr / m²]
    f_peak              : float = 3.0    # Divertor heat flux peaking factor [-]
    f_div_area_fraction : float = 0.10   # A_div / A_FW ratio [-]
    # Replacement downtimes (performed in parallel if scheduled together)
    dt_rep_bl           : float = 0.5    # Blanket replacement downtime [yr]
    dt_rep_div          : float = 0.4    # Divertor replacement downtime [yr]

# Module-level default instance — import and reuse rather than reinstantiating
DEFAULT_CONFIG = GlobalConfig()

# =============================================================================
#  Shared input-deck value coercion
# =============================================================================
# Single source of truth for converting a raw "key = value" token read from a
# text input deck into the correctly typed Python value for the matching
# GlobalConfig field. The three loaders (load_config_from_file in D0FUS_run,
# load_input_file in D0FUS_genetic, load_scan_parameters in D0FUS_scan) all call
# this helper, so they can never again disagree on how a token is interpreted.
#
# The GlobalConfig field default acts as the type oracle:
#   - boolean field  : "True"/"False" (any case, also 1/0/yes/no/on/off) -> bool
#   - optional field : "None"/"null" -> None, otherwise a number
#   - string field   : kept verbatim, so a string sentinel such as
#                       cost_model = "None" is preserved instead of being
#                       turned into the Python None
#   - numeric field  : float, narrowed to int when integral
# Keys that are not GlobalConfig fields (e.g. GA or scan hyperparameters) keep
# the historical float-or-verbatim-string behaviour so their dedicated
# downstream handling is left untouched.

_CONFIG_FIELD_KINDS = None


def _config_field_kinds():
    """Build and cache a {field_name: kind} map from the GlobalConfig defaults."""
    global _CONFIG_FIELD_KINDS
    if _CONFIG_FIELD_KINDS is None:
        from dataclasses import fields as _dc_fields
        kinds = {}
        for _f in _dc_fields(GlobalConfig):
            _d = _f.default
            if isinstance(_d, bool):
                kinds[_f.name] = 'bool'
            elif _d is None:
                kinds[_f.name] = 'optional'
            elif isinstance(_d, str):
                kinds[_f.name] = 'str'
            else:
                kinds[_f.name] = 'numeric'
        _CONFIG_FIELD_KINDS = kinds
    return _CONFIG_FIELD_KINDS


def coerce_input_value(key, raw_value):
    """Coerce a raw input-deck token to the right type for GlobalConfig[key].

    See the module comment above for the coercion rules. Already-typed (non
    string) inputs are returned unchanged, so the function is safe to call on
    values that have already been parsed.
    """
    if not isinstance(raw_value, str):
        return raw_value

    raw = raw_value.strip()
    low = raw.lower()
    kind = _config_field_kinds().get(key, 'numeric')

    if kind == 'bool':
        if low in ('true', '1', 'yes', 'on'):
            return True
        if low in ('false', '0', 'no', 'off'):
            return False
        return raw  # malformed boolean: keep visible rather than guess

    if kind == 'optional':
        if low in ('none', 'null', ''):
            return None
        try:
            v = float(raw)
            return int(v) if v.is_integer() else v
        except ValueError:
            return raw

    if kind == 'str':
        return raw  # preserve string sentinels (e.g. "None", "Sheffield")

    # Numeric GlobalConfig field, or a non-config key (hyperparameter):
    # historical float-or-verbatim-string behaviour.
    try:
        v = float(raw)
        return int(v) if v.is_integer() else v
    except ValueError:
        return raw



# ─────────────────────────────────────────────────────────────────────────────
# Two named presets are exposed as top-level factories.  They each return a
# fresh GlobalConfig with a coherent set of sub-mode choices.  Users obtain
# the desired baseline with one line and may then surcharge any field by
# composing dataclasses.replace() over the result.
#
#     cfg = preset_refined()
#     cfg = replace(cfg, R0=9.0, a=2.5, Bmax_TF=12.0)
#
# The presets adjust only the sub-mode selectors that distinguish the two
# philosophies; all engineering, geometric and economic defaults are
# inherited from GlobalConfig and remain user-controllable.
# ─────────────────────────────────────────────────────────────────────────────

# Sub-modes overridden by each preset (single source of truth).
_PRESET_ACADEMIC_SUBMODES = dict(
    # Geometry: cylindrical-torus closed forms (no Miller, no triangularity).
    Plasma_geometry        = 'Academic',
    Radial_build_model     = 'academic',
    # Profiles: L-mode parabolic shapes, no pedestal.
    Plasma_profiles        = 'L',
    # Bootstrap: Segal-Cerfon-Freidberg analytical fit (Nucl. Fusion 2021).
    Bootstrap_choice       = 'Segal',
    # Trapped fraction: simple closed-form expression (Fable / ASTRA).
    trapped_fraction_model = 'ASTRA',
    # Plasma current scaling: legacy Uckan (1989) shape correction.
    Option_q95             = 'ITER_1989',
    # Resistivity: pure Spitzer, no neoclassical trapped-particle correction.
    eta_model              = 'spitzer',
    # Heating and current drive: technology-agnostic figure-of-merit gamma_CD.
    CD_source              = 'Academic',
    # q-profile: parametric ansatz j ∝ (1 − ρ²)^alpha_J with alpha_J prescribed.
    q_profile_mode         = 'academic',
    alpha_J                = 1.5,
)

_PRESET_REFINED_SUBMODES = dict(
    # Geometry: full Miller flux-surface parameterisation with smoothstep5 profiles.
    Plasma_geometry        = 'refined',
    Radial_build_model     = 'refined',
    # Profiles: H-mode pedestal-aware shapes.
    Plasma_profiles        = 'H',
    # Bootstrap: Sauter (1999/2002) structure with Redl (2021) refitted
    # coefficients — improved accuracy in the pedestal and with impurities.
    Bootstrap_choice       = 'Sauter-Redl',
    # Trapped fraction: Sauter (2002) numerical fit, valid for shaped flux
    # surfaces.
    trapped_fraction_model = 'Sauter2002',
    # Plasma current scaling: Sauter (2016) edge-shaping formulation.
    Option_q95             = 'Sauter',
    # Resistivity: Sauter (1999) + Redl (2021) refitted neoclassical fit.
    eta_model              = 'redl',
    # Heating and current drive: physics-based combined LH/EC/NB/IC; the user
    # selects the actual technology mix via P_LH, P_ECRH, P_NBI, P_ICRH (or
    # f_heat_* fractions in steady-state mode).  Setting any of those to 0
    # disables the corresponding source.
    CD_source              = 'Multi',
    # q-profile: Picard self-consistency on q(ρ) from j_Ohm + j_CD + j_bs.
    # alpha_J is unused in this mode (kept at default for cross-mode symmetry).
    q_profile_mode         = 'refined',
)


def preset_academic(**overrides) -> 'GlobalConfig':
    """
    Build a GlobalConfig configured for the pedagogical 'academic' mode.

    All sub-mode selectors are set to their analytical / closed-form
    counterparts; engineering, geometric and economic fields keep their
    GlobalConfig defaults.  Pass keyword arguments to surcharge any field
    after the preset has been applied.

    Returns
    -------
    GlobalConfig
        Fresh instance with academic sub-modes.

    Examples
    --------
    >>> cfg = preset_academic()
    >>> cfg = preset_academic(R0=6.2, a=2.0, Bmax_TF=11.8)
    >>> cfg = preset_academic(eta_model='redl')   # surcharge a sub-mode
    """
    # Merge: overrides take precedence over the preset defaults so the user
    # can change any single sub-mode without bypassing the preset entirely.
    return GlobalConfig(**{**_PRESET_ACADEMIC_SUBMODES, **overrides})


def preset_refined(**overrides) -> 'GlobalConfig':
    """
    Build a GlobalConfig configured for the physically detailed 'refined'
    mode.

    All sub-mode selectors are set to their best-validated physics-based
    counterparts; engineering, geometric and economic fields keep their
    GlobalConfig defaults.  Pass keyword arguments to surcharge any field
    after the preset has been applied.

    Returns
    -------
    GlobalConfig
        Fresh instance with refined sub-modes.

    Examples
    --------
    >>> cfg = preset_refined()
    >>> cfg = preset_refined(R0=9.0, a=2.5, Bmax_TF=12.0, P_NBI=50.0)
    >>> cfg = preset_refined(Bootstrap_choice='Segal')   # surcharge a sub-mode
    """
    # Merge: overrides take precedence over the preset defaults so the user
    # can change any single sub-mode without bypassing the preset entirely.
    return GlobalConfig(**{**_PRESET_REFINED_SUBMODES, **overrides})


# ─────────────────────────────────────────────────────────────────────────────
#  Coil material volumetric mass densities [kg/m³]
# ─────────────────────────────────────────────────────────────────────────────
#
# Structural steels
# -----------------
#   316LN  (ITER TF/CS casing, vacuum vessel) — austenitic stainless, cryogenic
#           ρ = 7930 kg/m³.
#           Ref: ITER DDD 1.6 (2013); Garland et al., Fusion Eng. Des. 2008.
#
#   N50H   (high-strength austenitic, ITER Nb3Sn conduit) — Incoloy-like
#           ρ = 8050 kg/m³.
#           Ref: Mitchell, Fusion Eng. Des. 75-79 (2005).
#
# Superconducting materials  (volume fraction in D0FUS = non-Cu SC material)
# ---------------------------------------------------------------------------
#   Nb3Sn  A15 compound (filaments without Cu matrix)
#           ρ = 8600 kg/m³.
#           Ref: Wilson, "Superconducting Magnets", Clarendon Press (1983);
#                Larbalestier, MRS Bull. 29 (2004).
#
#   NbTi   alloy Ti-46.5 wt% Nb (filaments without Cu matrix)
#           ρ = 6100 kg/m³.
#           Ref: Iwasa, "Case Studies in Superconducting Magnets", 2nd ed.
#                Springer (2009), Table A1.3.
#
#   REBCO  coated-conductor non-Cu stack (REBCO layer + buffer + Hastelloy
#           substrate — weighted average for Fujikura / SuperPower 12 mm tape)
#           ρ = 7800 kg/m³.
#           Ref: Senatore et al., Supercond. Sci. Technol. 37 (2024);
#                Fujikura HTS tape datasheet (2022).
#
# Copper stabiliser
# -----------------
#   OFHC copper (oxygen-free high-conductivity)
#           ρ = 8960 kg/m³.
#           Ref: ASM Handbook Vol. 2 (1990).
#
# Insulation
# ----------
#   Cryogenic glass-epoxy (GFRP, S-glass / CTD-101K binder)
#           ρ = 1900 kg/m³.
#           Ref: Weisend II (ed.), "Handbook of Cryogenic Engineering",
#                Taylor & Francis (1998), p. 386;
#                Bauer et al., Cryogenics 42 (2002).
#
# ─────────────────────────────────────────────────────────────────────────────

COIL_MATERIAL_DENSITIES = {
    # Structural steel — keyed by Chosen_Steel value
    'steel': {
        '316L':   7930.0,   # [kg/m³]  316LN austenitic stainless
        'N50H':   8050.0,   # [kg/m³]  N50H high-strength austenitic
        'Manual': 7930.0,   # [kg/m³]  fallback to 316LN
    },
    # Superconductor material (non-Cu fraction) — keyed by Supra_choice
    'SC': {
        'Nb3Sn':  8600.0,   # [kg/m³]  Nb3Sn A15 filament
        'NbTi':   6100.0,   # [kg/m³]  NbTi alloy
        'REBCO':  7800.0,   # [kg/m³]  REBCO tape non-Cu stack
        'Manual': 8600.0,   # [kg/m³]  fallback to Nb3Sn
    },
    # Copper stabiliser
    'Cu':         8960.0,   # [kg/m³]  OFHC copper
    # Insulation (cryogenic glass-epoxy)
    'insulation': 1900.0,   # [kg/m³]  GFRP CTD-101K
}

# Volumetric mass densities [kg/m³] of the constituent materials appearing in
# the per-layer 'composition' (volume-fraction) dicts of
# BLANKET_CONCEPTS[...]['radial_layers'], digitised from figures 10-15
# (Appendix A) of the reference study. f_radial_build_layers() combines these
# with each layer's 'composition' to obtain an effective smeared density
# rho = sum(frac_i * BLANKET_MATERIAL_DENSITIES[i]) and per-material
# component masses. Values are room-temperature/handbook densities unless
# noted; this is a 0D mass estimate, not a substitute for detailed material
# property sets.
#
# Citations:
#  - void, airSTP:  vacuum (0) and standard air at 15C/101.325 kPa (ISA),
#       rho_air = 1.225 kg/m3 (CRC Handbook of Chemistry and Physics).
#  - Water: liquid water at 4C reference, rho = 1000 kg/m3 (CRC Handbook).
#  - EUROFER97, MF82H, BMF82H: RAFM steels of the same Fe-9Cr-W-V-Ta family;
#       rho(EUROFER97) = 7750 kg/m3, e.g. Lindau et al., J. Nucl. Mater.
#       336 (2005) 81; same value adopted for F82H-class steels (MF82H,
#       boron-doped BMF82H).
#  - SS316L: 316L austenitic stainless steel, rho = 7990 kg/m3 (CRC Handbook /
#       standard mill data sheets).
#  - Inconel718: rho = 8190 kg/m3 (Special Metals / ESPI Metals datasheet,
#       https://www.espimetals.com/index.php/technical-data/91-inconel-718).
#  - W: tungsten, rho = 19250 kg/m3 (CRC Handbook).
#  - WC: tungsten carbide, rho = 15630 kg/m3 (CRC Handbook).
#  - SiC: silicon carbide, theoretical/fully-dense rho = 3210 kg/m3
#       (e.g. Schunk Technical Ceramics datasheet; ScienceDirect SiC overview).
#  - Li4SiO4: lithium orthosilicate breeder ceramic, theoretical rho =
#       2350 kg/m3 (Hernandez et al./KIT HCPB pebble fabrication studies,
#       ScienceDirect S1738573323002139).
#  - Li2TiO3: lithium titanate breeder ceramic, theoretical rho = 3430 kg/m3
#       (INL TPBAR/ITER data compilation, inldigitallibrary.inl.gov
#       Sort_146241.pdf).
#  - Be12Ti: beryllide neutron multiplier, rho = 2288 kg/m3
#       (Kawamura et al., J. Nucl. Mater.; ScienceDirect S0966979518301122,
#       S0920379611002791).
#  - Be: beryllium metal, rho = 1850 kg/m3 (CRC Handbook).
#  - CaO: calcium oxide (Li ceramic-breeder additive), rho = 3340 kg/m3
#       (CRC Handbook / standard chemistry references).
#  - PbLi: Pb-15.7Li eutectic breeder/coolant, rho ~ 9800 kg/m3 from the
#       correlation rho(kg/m3) = 10520.35 - 1.19051*T[K] at T ~ 600 K (327C)
#       (Mas de les Valls et al., J. Nucl. Mater. 376 (2008) 353, "Lead-lithium
#       eutectic material database for nuclear fusion technology").
#  - Li: pure liquid lithium breeder (SCLV), rho = 485 kg/m3 at 500C
#       (NASA thermophysical compilation of liquid lithium,
#       ntrs.nasa.gov/api/citations/19680018893).
#  - FLiBe: 2LiF-BeF2 molten-salt breeder, rho ~ 2020 kg/m3 from the
#       correlation rho(kg/m3) = 2413 - 0.488*T[K] at T ~ 800 K (527C)
#       (Romatoski & Forsberg review; Lee et al., J. Chem. Eng. Data 68
#       (2023), doi:10.1021/acs.jced.2c00212, PMC9743087).
#  - V4Cr4Ti: V-4Cr-4Ti vanadium structural alloy, rho = 6060 kg/m3, computed
#       via rule-of-mixtures from elemental densities (V 6110, Cr 7190,
#       Ti 4506 kg/m3) at the nominal 92/4/4 wt% composition (Smith et al.,
#       J. Nucl. Mater. 233-237 (1996) 356, "Reference vanadium alloy
#       V-4Cr-4Ti for fusion application").
#  - HeT410P80: helium coolant at T = 410 C, P = 8.0 MPa, rho = 5.64 kg/m3
#       from the ideal-gas law rho = P*M/(R*T) (M_He = 4.003 g/mol).
BLANKET_MATERIAL_DENSITIES = {
    'void':       0.0,       # vacuum / gap
    'airSTP':     1.225,     # air, 15 C / 101.325 kPa
    'Water':      1000.0,    # liquid water, 4 C reference
    'EUROFER97':  7750.0,    # RAFM steel
    'MF82H':      7750.0,    # RAFM steel (F82H-class)
    'BMF82H':     7750.0,    # RAFM steel (boron-doped F82H-class)
    'SS316L':     7990.0,    # 316L austenitic stainless steel
    'Inconel718': 8190.0,    # Ni-based superalloy
    'W':          19250.0,   # tungsten armour
    'WC':         15630.0,   # tungsten carbide (shield)
    'SiC':        3210.0,    # silicon carbide
    'Li4SiO4':    2350.0,    # lithium orthosilicate breeder ceramic
    'Li2TiO3':    3430.0,    # lithium titanate breeder ceramic
    'Be12Ti':     2288.0,    # beryllide neutron multiplier
    'Be':         1850.0,    # beryllium multiplier
    'CaO':        3340.0,    # calcium oxide additive
    'PbLi':       9800.0,    # Pb-15.7Li eutectic, ~327 C
    'Li':         485.0,     # pure liquid lithium, ~500 C
    'FLiBe':      2020.0,    # 2LiF-BeF2 molten salt, ~527 C
    'V4Cr4Ti':    6060.0,    # vanadium structural alloy
    'HeT410P80':  5.64,      # He coolant, 410 C / 8.0 MPa (ideal gas)
}

# ─────────────────────────────────────────────────────────────────────────────
# Breeding-blanket concepts
# ─────────────────────────────────────────────────────────────────────────────
# Six concepts compared in the "Infinity Two pilot plant blanket trade study"
# (J. Plasma Phys. 91, E79 (2025), doi:10.1017/S002237782500039X).
#
# Each entry defines a concept-specific radial-build layer stack 'radial_layers'
# (ordered plasma -> TF, widths in [m]), reflecting the inboard radial builds of
# figures 10-15 of the reference study.  Exactly one entry per concept has
# 'width': None and 'role': 'breeder' — its thickness is the residual of the
# top-level design variable b once every other (fixed-width) layer in the stack
# has been subtracted off (see radial_layers_effective()).  High/low-temperature
# shield layers ('shield_HT' / 'shield_LT') use the DEFAULT_SHIELD_HT_WIDTH /
# DEFAULT_SHIELD_LT_WIDTH defaults below where the source figures do not give a
# value.  Every layer carries a 'composition' dict (volume fractions, keys
# into BLANKET_MATERIAL_DENSITIES, summing to 1.0) digitised from figures
# 10-15 (Appendix A); f_radial_build_layers() derives each layer's effective
# density, per-material component masses and total mass from this
# composition.  Where a single radial_layers entry (e.g. the FW) aggregates
# several figure sub-layers, 'composition' is the volume-weighted average
# over those sub-layers.
#
# TBR_max and delta_BB_sat (breeder-zone thickness for ~95% of TBR_max) feed the
# saturation-curve model f_TBR():
#   TBR(delta_BZ) = TBR_max * (1 - exp(-delta_BZ / delta_e)),
#   delta_e = delta_BB_sat / ln(20)
# where delta_BZ = delta_BB_ib (breeder-layer IB thickness).
#
# All numeric values are illustrative, smeared 0D estimates consistent with
# the ranges reported in the reference study; they are not a substitute for
# detailed neutronics.
DEFAULT_SHIELD_HT_WIDTH = 0.20   # [m]  high-temperature shield default thickness
DEFAULT_SHIELD_LT_WIDTH = 0.10   # [m]  low-temperature shield default thickness

BLANKET_CONCEPTS = {
    'HCPB': {
        'label':       'Helium-Cooled Pebble Bed',
        'breeder':     'Li4SiO4 (pebble bed)',
        'multiplier':  'Be12Ti (pebble bed)',
        'coolant':     'He',
        'structure':   'EUROFER97',
        'TBR_max':       1.46,   # [-]   saturated TBR (parametric study)
        'delta_BB_sat':  0.50,   # [m]   breeder-zone thickness for ~saturated TBR
        'M_blanket':     1.35,   # [-]   energy multiplication factor
        'eta_T_range':  (0.30, 0.40),   # [-] net thermal efficiency range
        'Li6_enrichment': 'Not required (natural Li)',
        'shield_quality': 'Excellent (dense solid breeder)',
        'VV_feasible':    True,
        'sublayers': [
            {'name': 'Breeder/Multiplier Zone (Li4SiO4 + Be12Ti)', 'f_width': 0.65, 'rho': 2700.0},
            {'name': 'Manifold / Back Structure',                   'f_width': 0.35, 'rho': 7000.0},
        ],
        'radial_layers': [
            {'name': 'SOL',         'role': 'SOL',       'width': 0.05,
             'composition': {'void': 1.0}},
            {'name': 'First wall',  'role': 'FW',        'width': 0.04,
             # 0.2cm W armour + 3.8cm MF82H/HeT410P80 structural (fig. 10), volume-weighted
             'composition': {'W': 0.05, 'MF82H': 0.323, 'HeT410P80': 0.627}},
            {'name': 'Breeder/Multiplier Zone (Li4SiO4 + Be12Ti)', 'role': 'breeder', 'width': None,
             'composition': {'Li4SiO4': 0.063, 'Li2TiO3': 0.022, 'EUROFER97': 0.110,
                              'Be12Ti': 0.539, 'HeT410P80': 0.266}},
            {'name': 'Back Wall',   'role': 'structure', 'width': 0.10,
             'composition': {'Li4SiO4': 0.063, 'Li2TiO3': 0.022, 'EUROFER97': 0.459, 'HeT410P80': 0.455}},
            {'name': 'Manifold',    'role': 'structure', 'width': 0.15,
             'composition': {'EUROFER97': 0.890, 'HeT410P80': 0.110}},
            {'name': 'High-Temp Shield', 'role': 'shield_HT', 'width': DEFAULT_SHIELD_HT_WIDTH,
             'composition': {'MF82H': 0.28, 'WC': 0.52, 'HeT410P80': 0.20}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'void': 1.0}},
            {'name': 'Vacuum Vessel', 'role': 'VV',      'width': 0.10,
             'composition': {'SS316L': 0.76, 'HeT410P80': 0.24}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'airSTP': 1.0}},
            {'name': 'Low-Temp Shield', 'role': 'shield_LT', 'width': DEFAULT_SHIELD_LT_WIDTH,
             'composition': {'SS316L': 0.39, 'BMF82H': 0.29, 'Water': 0.32}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.10,
             'composition': {'airSTP': 1.0}},
        ],
    },
    'HCLL': {
        'label':       'Helium-Cooled Lithium-Lead',
        'breeder':     'PbLi (eutectic, inherent multiplier)',
        'multiplier':  'None (PbLi self-multiplies)',
        'coolant':     'He',
        'structure':   'EUROFER97',
        'TBR_max':       1.38,   # [-]   mid-range (1.35-1.40)
        'delta_BB_sat':  0.60,   # [m]
        'M_blanket':     1.32,   # [-]
        'eta_T_range':  (0.30, 0.35),
        'Li6_enrichment': '90% required',
        'shield_quality': 'Intermediate',
        'VV_feasible':    True,
        'sublayers': [
            {'name': 'Breeder Zone (PbLi + stiffening plates)', 'f_width': 0.75, 'rho': 9000.0},
            {'name': 'Manifold / Back Structure',               'f_width': 0.25, 'rho': 6500.0},
        ],
        'radial_layers': [
            {'name': 'SOL',         'role': 'SOL',       'width': 0.05,
             'composition': {'void': 1.0}},
            {'name': 'First wall',  'role': 'FW',        'width': 0.04,
             # 0.2cm W armour + 3.8cm MF82H/HeT410P80 structural (fig. 11), volume-weighted
             'composition': {'W': 0.05, 'MF82H': 0.323, 'HeT410P80': 0.627}},
            {'name': 'Breeder Zone (PbLi + stiffening plates)', 'role': 'breeder', 'width': None,
             'composition': {'PbLi': 0.79, 'EUROFER97': 0.158, 'HeT410P80': 0.052}},
            {'name': 'Back Wall',   'role': 'structure', 'width': 0.02,
             'composition': {'EUROFER97': 0.698, 'HeT410P80': 0.302}},
            {'name': 'Manifold',    'role': 'structure', 'width': 0.16,
             'composition': {'EUROFER97': 0.85, 'HeT410P80': 0.10, 'PbLi': 0.05}},
            {'name': 'High-Temp Shield', 'role': 'shield_HT', 'width': DEFAULT_SHIELD_HT_WIDTH,
             'composition': {'MF82H': 0.28, 'WC': 0.52, 'HeT410P80': 0.20}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'void': 1.0}},
            {'name': 'Vacuum Vessel', 'role': 'VV',      'width': 0.10,
             'composition': {'SS316L': 0.76, 'HeT410P80': 0.24}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'airSTP': 1.0}},
            {'name': 'Low-Temp Shield', 'role': 'shield_LT', 'width': DEFAULT_SHIELD_LT_WIDTH,
             'composition': {'SS316L': 0.39, 'BMF82H': 0.29, 'Water': 0.32}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.10,
             'composition': {'airSTP': 1.0}},
        ],
    },
    'DCLL': {
        'label':       'Dual-Coolant Lithium-Lead',
        'breeder':     'PbLi (eutectic, inherent multiplier)',
        'multiplier':  'None (PbLi self-multiplies); SiC flow-channel inserts',
        'coolant':     'He (primary) + PbLi (secondary, self-cooled)',
        'structure':   'EUROFER97 / F82H',
        'TBR_max':       1.42,   # [-]
        'delta_BB_sat':  0.70,   # [m]
        'M_blanket':     1.32,   # [-]
        'eta_T_range':  (0.30, 0.40),
        'Li6_enrichment': '90% required',
        'shield_quality': 'Lower than HCPB (thicker shielding needed)',
        'VV_feasible':    True,
        'sublayers': [
            {'name': 'Breeder Zone (PbLi + SiC FCIs)', 'f_width': 0.92, 'rho': 8200.0},
            {'name': 'Manifold / Back Structure',      'f_width': 0.08, 'rho': 6000.0},
        ],
        'radial_layers': [
            {'name': 'SOL',         'role': 'SOL',       'width': 0.05,
             'composition': {'void': 1.0}},
            {'name': 'First wall',  'role': 'FW',        'width': 0.04,
             # 0.2cm W armour + 3.8cm MF82H/HeT410P80 structural (fig. 12), volume-weighted
             'composition': {'W': 0.05, 'MF82H': 0.323, 'HeT410P80': 0.627}},
            {'name': 'Breeder Zone (PbLi + SiC FCIs)', 'role': 'breeder',   'width': None,
             'composition': {'PbLi': 0.77, 'SiC': 0.035, 'MF82H': 0.06, 'HeT410P80': 0.135}},
            {'name': 'Back Wall',   'role': 'structure', 'width': 0.02,
             'composition': {'MF82H': 0.80, 'HeT410P80': 0.20}},
            {'name': 'Manifold',    'role': 'structure', 'width': 0.06,
             'composition': {'MF82H': 0.30, 'HeT410P80': 0.70}},
            {'name': 'High-Temp Shield', 'role': 'shield_HT', 'width': DEFAULT_SHIELD_HT_WIDTH,
             'composition': {'MF82H': 0.28, 'WC': 0.52, 'HeT410P80': 0.20}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'void': 1.0}},
            {'name': 'Vacuum Vessel', 'role': 'VV',      'width': 0.10,
             'composition': {'SS316L': 0.76, 'HeT410P80': 0.24}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'airSTP': 1.0}},
            {'name': 'Low-Temp Shield', 'role': 'shield_LT', 'width': DEFAULT_SHIELD_LT_WIDTH,
             'composition': {'SS316L': 0.39, 'BMF82H': 0.29, 'Water': 0.32}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.10,
             'composition': {'airSTP': 1.0}},
        ],
    },
    'SCLL': {
        'label':       'Self-Cooled Lithium-Lead',
        'breeder':     'PbLi (eutectic, inherent multiplier)',
        'multiplier':  'Optional Be booster/reflector',
        'coolant':     'Self-cooled (PbLi flow)',
        'structure':   'SiC composite',
        'TBR_max':       1.47,   # [-]   highest among liquid concepts
        'delta_BB_sat':  0.45,   # [m]   most compact radial build
        'M_blanket':     1.30,   # [-]
        'eta_T_range':  (0.35, 0.45),
        'Li6_enrichment': '90% required',
        'shield_quality': 'Adequate (limited by liquid-metal moderation)',
        'VV_feasible':    True,
        'sublayers': [
            {'name': 'Breeder Zone (PbLi + SiC)',  'f_width': 0.88, 'rho': 8500.0},
            {'name': 'Manifold / Back Structure',  'f_width': 0.12, 'rho': 5500.0},
        ],
        'radial_layers': [
            {'name': 'SOL',         'role': 'SOL',       'width': 0.05,
             'composition': {'void': 1.0}},
            {'name': 'Breeder Zone (PbLi + SiC)', 'role': 'breeder',   'width': None,
             'composition': {'PbLi': 0.696, 'SiC': 0.222, 'void': 0.082}},
            {'name': 'High-Temp Shield', 'role': 'shield_HT', 'width': DEFAULT_SHIELD_HT_WIDTH,
             'composition': {'MF82H': 0.28, 'WC': 0.52, 'HeT410P80': 0.20}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'void': 1.0}},
            {'name': 'Vacuum Vessel', 'role': 'VV',      'width': 0.10,
             'composition': {'SS316L': 0.76, 'HeT410P80': 0.24}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.02,
             'composition': {'airSTP': 1.0}},
            {'name': 'Low-Temp Shield', 'role': 'shield_LT', 'width': DEFAULT_SHIELD_LT_WIDTH,
             'composition': {'SS316L': 0.39, 'BMF82H': 0.29, 'Water': 0.32}},
            {'name': 'Gap',         'role': 'gap',       'width': 0.10,
             'composition': {'airSTP': 1.0}},
        ],
    },
    'SCLV': {
        'label':       'Self-Cooled Lithium-Vanadium',
        'breeder':     'Li (pure liquid lithium, inherent multiplier)',
        'multiplier':  'Optional external multiplier/reflector',
        'coolant':     'Self-cooled (Li flow)',
        'structure':   'V-4Cr-4Ti (vanadium alloy)',
        'TBR_max':       1.47,   # [-]   equivalent to SCLL
        'delta_BB_sat':  0.55,   # [m]   mid-range
        'M_blanket':     1.35,   # [-]
        'eta_T_range':  (0.35, 0.40),
        'Li6_enrichment': 'Not required (natural Li)',
        'shield_quality': 'Lower than solid breeders (thicker build needed)',
        'VV_feasible':    False,   # oxidation/embrittlement concerns
        'sublayers': [
            {'name': 'Breeder Zone (Li + V-alloy structure)', 'f_width': 0.80, 'rho': 2200.0},
            {'name': 'Manifold / Back Structure',             'f_width': 0.20, 'rho': 5000.0},
        ],
        'radial_layers': [
            {'name': 'SOL',         'role': 'SOL',        'width': 0.05,
             'composition': {'void': 1.0}},
            {'name': 'First wall',  'role': 'FW',         'width': 0.032,
             # 0.2cm W armour + 1cm V4Cr4Ti structural + 2cm Li coolant channel (fig. 14), volume-weighted
             'composition': {'W': 0.0625, 'V4Cr4Ti': 0.3625, 'CaO': 0.0125, 'Li': 0.5625}},
            {'name': 'Gap',         'role': 'gap',        'width': 0.01,
             'composition': {'void': 1.0}},
            {'name': 'Breeder Zone (Li + V-alloy structure)', 'role': 'breeder',    'width': None,
             'composition': {'V4Cr4Ti': 0.08, 'CaO': 0.02, 'Li': 0.90}},
            {'name': 'Multiplier (Be)', 'role': 'multiplier', 'width': 0.01,
             'composition': {'Be': 1.0}},
            {'name': 'High-Temp Shield', 'role': 'shield_HT',  'width': DEFAULT_SHIELD_HT_WIDTH,
             'composition': {'MF82H': 0.28, 'WC': 0.52, 'HeT410P80': 0.20}},
            {'name': 'Vacuum Vessel', 'role': 'VV',         'width': 0.10,
             'composition': {'SS316L': 0.76, 'HeT410P80': 0.24}},
            {'name': 'Low-Temp Shield', 'role': 'shield_LT',  'width': DEFAULT_SHIELD_LT_WIDTH,
             'composition': {'SS316L': 0.39, 'BMF82H': 0.29, 'Water': 0.32}},
            {'name': 'Gap',         'role': 'gap',        'width': 0.10,
             'composition': {'airSTP': 1.0}},
        ],
    },
    'F-LIB': {
        'label':       'FLiBe Liquid-Immersion Blanket',
        'breeder':     'FLiBe (2:1 LiF-BeF2 molten salt)',
        'multiplier':  'Be (inherent + optional additions)',
        'coolant':     'Self-cooled (quasi-stagnant salt pool)',
        'structure':   'Inconel-718 (not low-activation)',
        'TBR_max':       1.09,   # [-]   lowest of all concepts (self-shielding)
        'delta_BB_sat':  0.35,   # [m]   smallest radial builds
        'M_blanket':     1.20,   # [-]
        'eta_T_range':  (0.35, 0.40),
        'Li6_enrichment': 'Optional (not yet optimised)',
        'shield_quality': 'Superior moderation/absorption, but self-shielding limits breeding',
        'VV_feasible':    False,   # VV not adequately shielded for 5-yr lifetime
        'sublayers': [
            {'name': 'Breeder Zone (FLiBe + Be)',  'f_width': 0.90, 'rho': 2200.0},
            {'name': 'Manifold / Back Structure',  'f_width': 0.10, 'rho': 7500.0},
        ],
        'radial_layers': [
            {'name': 'SOL',         'role': 'SOL',        'width': 0.05,
             'composition': {'void': 1.0}},
            {'name': 'First wall',  'role': 'FW',         'width': 0.032,
             # 0.2cm W armour + 1cm Inconel718 structural + 2cm FLiBe coolant channel (fig. 15), volume-weighted
             'composition': {'W': 0.0625, 'Inconel718': 0.3125, 'FLiBe': 0.625}},
            {'name': 'Multiplier (Be)', 'role': 'multiplier', 'width': 0.01,
             'composition': {'Be': 1.0}},
            {'name': 'Vacuum Vessel', 'role': 'VV',         'width': 0.03,
             'composition': {'Inconel718': 1.0}},
            {'name': 'Breeder Zone (FLiBe + Be)', 'role': 'breeder',    'width': None,
             'composition': {'FLiBe': 1.0}},
            {'name': 'Back Wall',   'role': 'structure',  'width': 0.06,
             'composition': {'Inconel718': 0.99, 'FLiBe': 0.01}},
            {'name': 'High-Temp Shield', 'role': 'shield_HT',  'width': DEFAULT_SHIELD_HT_WIDTH,
             'composition': {'MF82H': 0.28, 'WC': 0.52, 'HeT410P80': 0.20}},
            {'name': 'Gap',         'role': 'gap',        'width': 0.01,
             'composition': {'airSTP': 1.0}},
            {'name': 'Low-Temp Shield', 'role': 'shield_LT',  'width': DEFAULT_SHIELD_LT_WIDTH,
             'composition': {'SS316L': 0.39, 'BMF82H': 0.29, 'Water': 0.32}},
            {'name': 'Gap',         'role': 'gap',        'width': 0.02,
             'composition': {'airSTP': 1.0}},
        ],
    },
}


def material_rho(Chosen_Steel: str, Supra_choice: str) -> dict:
    """
    Return volumetric mass densities [kg/m³] for all coil materials.

    Parameters
    ----------
    Chosen_Steel : str  Steel grade key from GlobalConfig ('316L', 'N50H', 'Manual').
    Supra_choice : str  SC type from GlobalConfig ('Nb3Sn', 'NbTi', 'REBCO', 'Manual').

    Returns
    -------
    dict with keys: 'steel', 'SC', 'Cu', 'insulation'  — all in [kg/m³].
    """
    steel_key = Chosen_Steel if Chosen_Steel in COIL_MATERIAL_DENSITIES['steel'] else 'Manual'
    sc_key    = Supra_choice  if Supra_choice  in COIL_MATERIAL_DENSITIES['SC']    else 'Manual'
    return {
        'steel':       COIL_MATERIAL_DENSITIES['steel'][steel_key],
        'SC':          COIL_MATERIAL_DENSITIES['SC'][sc_key],
        'Cu':          COIL_MATERIAL_DENSITIES['Cu'],
        'insulation':  COIL_MATERIAL_DENSITIES['insulation'],
    }


def material_blanket(Blanket_choice: str) -> dict:
    """
    Return the BLANKET_CONCEPTS entry for the given concept choice.

    Parameters
    ----------
    Blanket_choice : str  Concept key ('HCPB', 'HCLL', 'DCLL', 'SCLL', 'SCLV', 'F-LIB').
                          Falls back to 'HCPB' if not recognised.

    Returns
    -------
    dict  Concept entry (label, breeder, coolant, structure, multiplier,
          TBR_max, delta_BB_sat, M_blanket, eta_T_range, Li6_enrichment,
          shield_quality, VV_feasible, sublayers).
    """
    return BLANKET_CONCEPTS.get(Blanket_choice, BLANKET_CONCEPTS['HCPB'])


def M_blanket_effective(Blanket_choice: str) -> float:
    """
    Blanket energy multiplication factor [-] for the selected concept.

    Returns BLANKET_CONCEPTS[Blanket_choice]['M_blanket'], used in place of
    the generic GlobalConfig.M_blanket default in the power balance.
    """
    return material_blanket(Blanket_choice)['M_blanket']


def eta_T_effective(Blanket_choice: str) -> float:
    """
    Thermal-to-electric conversion efficiency [-] for the selected concept.

    Returns the midpoint of BLANKET_CONCEPTS[Blanket_choice]['eta_T_range'],
    used in place of the generic GlobalConfig.eta_T default in the power
    balance.
    """
    eta_lo, eta_hi = material_blanket(Blanket_choice)['eta_T_range']
    return 0.5 * (eta_lo + eta_hi)


def radial_layers_effective(Blanket_choice: str, b: float, Delta_TF: float = 0.0) -> list:
    """
    Concept-specific radial-build layer stack with the breeder layer's width
    resolved to a value.

    Returns BLANKET_CONCEPTS[Blanket_choice]['radial_layers'] (ordered plasma ->
    TF) as a list of dicts, each augmented with 'delta_ib' and 'delta_ob' [m]
    (IB/OB thickness).  Every fixed-width layer has delta_ib == delta_ob ==
    'width'.  The single 'breeder' layer's thickness is the residual of b (IB)
    / b + Delta_TF (OB) once every other layer's width has been subtracted:

        delta_breeder_ib = b            - sum(fixed widths)
        delta_breeder_ob = b + Delta_TF - sum(fixed widths)

    clipped to >= 0.

    Parameters
    ----------
    Blanket_choice : str    Concept key (see BLANKET_CONCEPTS).
    b              : float  Breeding blanket + shield radial thickness [m].
    Delta_TF       : float  Outboard port-access radial clearance [m].

    Returns
    -------
    list of dict, one per layer, each with: name, role, width (None for
    breeder), rho (None if not concept-specific — falls back to GlobalConfig at
    run time), delta_ib, delta_ob.
    """
    concept    = material_blanket(Blanket_choice)
    fixed_sum  = sum(layer['width'] for layer in concept['radial_layers']
                     if layer['width'] is not None)
    delta_ib   = max(b - fixed_sum, 0.0)
    delta_ob   = max(b + Delta_TF - fixed_sum, 0.0)

    layers = []
    for layer in concept['radial_layers']:
        entry = dict(layer)
        if entry['width'] is None:
            entry['delta_ib'] = delta_ib
            entry['delta_ob'] = delta_ob
        else:
            entry['delta_ib'] = entry['width']
            entry['delta_ob'] = entry['width']
        layers.append(entry)
    return layers