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
    Bootstrap_choice    : str   = 'Redl'        # Bootstrap current model 'Freidberg', 'Segal', 'Sauter', 'Redl'
    trapped_fraction_model : str = 'Sauter2002' # 'Sauter2002' (standard, NEOS/JINTRAC)
                                                # 'ASTRA' (Fable, used by PROCESS/ASTRA)

    # q₉₅ formula selector:
    #   'Sauter'    — uses LCFS values (κ_edge, δ_edge). Sauter, FED 112 (2016) Eq. 30.
    #   'ITER_1989' — uses ψ_N = 0.95 values (κ₉₅, δ₉₅). Uckan (1989), also Johner (2011).
    Option_q95         : str = 'Sauter'         # q₉₅ formula: 'Sauter' (default) or 'ITER_1989'
    Option_Kappa       : str   = 'Wenninger'    # Elongation model: 'Wenninger', 'Stambaugh', 'Freidberg', 'Manual'
    κ_manual           : float = 1.9            # Elongation (Manual mode only) [-]

    # ── 2b. Plasma geometry model ────────────────────────────────────────────
    # Controls volume integrals and cross-section area elements.
    # 'Academic' : Cylindrical-torus approx, V = 2π²R₀κa²
    # 'D0FUS'    : Full Miller flux-surface parameterisation (Miller 1998).
    #              Numerically computed V'(ρ) with radial κ(ρ), δ(ρ) profiles.
    Plasma_geometry : str = 'D0FUS'             # 'Academic' or 'D0FUS'

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
    Radial_build_model   : str   = 'D0FUS'   # Stress model: 'academic', 'D0FUS', 'CIRCE'
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
    dt_rep              : float = 1.5    # Scheduled replacement downtime [yr]
    # Superconductor cost multiplier (1.5 Nb3Sn mature — 3.0 REBCO FOAK)
    Supra_cost_factor   : float = 2.0    # SC coil cost multiplier vs Cu [-]
    # Budget constraint (genetic algorithm)
    # Designs exceeding C_invest_max are penalised. Set to 1e6 to disable.
    C_invest_max        : float = 25e3   # Capital cost ceiling [M EUR]

# Module-level default instance — import and reuse rather than reinstantiating
DEFAULT_CONFIG = GlobalConfig()