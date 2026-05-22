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
    Tbar  : float = 14.0           # Volume-averaged ion temperature [keV]
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
    Option_Kappa       : str = 'Wenninger'    # Elongation model: 'Wenninger', 'Stambaugh', 'Freidberg', 'Manual'
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
    Greenwald_limit : float = 1.0  # Greenwald density fraction limit [-]
    ms              : float = 0.3  # Vertical stability margin parameter [-]

    # ── 4. Plasma composition ────────────────────────────────────────────────
    Atomic_mass : float = 2.5   # Volume-averaged ionic mass [AMU]  (D-T: 2.5)
    Zeff        : float = 2.0   # Effective plasma charge [-]
    r_synch     : float = 0.5   # Synchrotron radiation wall reflectivity [-]
    C_Alpha     : float = 7.0   # Helium ash dilution tuning parameter [-] (default taken from PROCESS)
    # Impurity line radiation (0D bulk-plasma estimate).
    # Comma-separated species ('W', 'Ar', 'Ne', 'C', 'N', 'Kr') with matching
    # concentrations n_imp/n_e. Empty string = disabled (pure D-T).
    # Typical: W 1e-5–5e-4, Ar/Ne 1e-3–5e-3.
    impurity_species : str = ''     # Comma-separated species: 'W', 'W, Ne', '' = none
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
    f_In      : float = 0.15    # Insulation area fraction [-]

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
    P_NBI  : float = 0.0      # NBI injected power [MW]  (5 % losses applied internally)
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
    # Single width per component (no IB/OB distinction except for the BB which
    # derives separately: δ_BB_ib = b − Σ(fixed), δ_BB_ob = b + Delta_TF − Σ(fixed)).
    # Layers in order from plasma outward: SOL, FW (Princeton-D), BB(derived), shield, VV, gap_TF
    delta_SOL    : float = 0.10   # SOL / far-SOL width at IB and OB midplane [m]
    f_kappa_SOL  : float = 0.25   # Elongation increase factor for SOL/FW shapes [-]
    delta_FW     : float = 0.05   # First wall width at IB and OB midplane [m]
    delta_shield     : float = 0.30   # Neutron shield [m]
    delta_VV         : float = 0.15   # Vacuum vessel [m]
    delta_gap_TF     : float = 0.05   # VV→TF gap [m]

    # ── Effective volumetric densities for radial-build mass estimates ────────
    # Smeared/homogenised values.  Defaults from BLANKET_MATERIAL_DENSITIES.
    rho_FW       : float = 7900.0    # First wall           [kg/m³]
    rho_BB       : float = 4500.0    # Breeding blanket     [kg/m³]
    rho_shield   : float = 5500.0    # Neutron shield       [kg/m³]
    rho_VV       : float = 7500.0    # Vacuum vessel        [kg/m³]
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

# Effective volumetric densities for radial-build components.
# These are smeared/homogenised values for 0D mass estimates.
BLANKET_MATERIAL_DENSITIES = {
    'FW':       7900.0,    # [kg/m³]  EUROFER / 316L steel (steel-dominated)
    'BB':       4500.0,    # [kg/m³]  effective (HCPB ~3500, WCLL ~6000; mid-range)
    'shield':   5500.0,    # [kg/m³]  steel + borated-water mix (~60/40 by vol.)
    'VV':       7500.0,    # [kg/m³]  double-wall 316L SS with water fill
    'divertor': 13000.0,   # [kg/m³]  W monoblocks + CuCrZr + steel structure
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