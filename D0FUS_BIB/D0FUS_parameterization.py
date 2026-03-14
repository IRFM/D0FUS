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
E_ELEM = 1.602176634e-19    # Elementary charge [C]
M_E    = 9.1094e-31         # Electron mass [kg]
M_I    = 2 * 1.6726e-27     # Ion mass (deuterium) [kg]
μ0     = 4.0 * np.pi * 1e-7 # Vacuum permeability [H/m]
EPS_0  = 8.8542e-12         # Vacuum permittivity [F/m]

# ── Fusion reaction energetics ─────────────────────────────────────────────────
E_ALPHA = 3.5e6  * E_ELEM   # Alpha particle energy [J]
E_N     = 14.1e6 * E_ELEM   # Neutron energy [J]
E_F     = 22.4e6 * E_ELEM   # Total fusion energy (Li blanket breeding assumed) [J]

#%% D0FUS Global Configuration
"""
Centralised repository of all user-adjustable design parameters,
including primary scan variables (geometry, field, power) and
physical/engineering assumptions.
"""

from dataclasses import dataclass, field
from typing import Optional

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
    """

    # ── 1. Primary scan variables ──────────────────────────────────────────────
    R0    : float = 9.0        # Major plasma radius [m]
    a     : float = 3.0        # Minor plasma radius [m]
    b     : float = 1.2        # Breeding blanket + neutron shield radial thickness [m]
    Bmax_TF  : float = 12.0    # Peak magnetic field on TF conductor [T]
    Bmax_CS  : float = 25.0    # Peak magnetic field allowed on the CS [T]
    P_fus : float = 2000.0     # Total fusion power [MW]
    Tbar  : float = 14.0       # Volume-averaged ion temperature [keV]
    H     : float = 1.0        # Confinement enhancement factor (H-factor) [-]

    # ── 2. Physics and operation ───────────────────────────────────────────────
    Operation_mode     : str   = 'Pulsed'       # 'Steady-State' or 'Pulsed'
    Temps_Plateau_input: float = 3600.0         # Flat-top duration (pulsed only) [s]
    P_aux_input        : float = 50.0           # Auxiliary heating power (pulsed only) [MW]

    Plasma_profiles    : str   = 'H'            # Profile peaking: 'L', 'H', 'Advanced', 'Manual'
    nu_n_manual        : float = 0.1            # Density peaking factor (Manual mode only) [-]
    nu_T_manual        : float = 1.0            # Temperature peaking factor (Manual mode only) [-]
    rho_ped    : float = 1.0                    # Normalised pedestal radius (1.0 = no pedestal) [-]
    n_ped_frac : float = 0.0                    # n_ped / nbar [-]
    T_ped_frac : float = 0.0                    # T_ped / Tbar [-]

    Scaling_Law        : str   = 'IPB98(y,2)'  # Energy confinement scaling law
    L_H_Scaling_choice : str   = 'New_Ip'      # L-H threshold scaling: 'Martin', 'New_S', 'New_Ip'
    Bootstrap_choice    : str   = 'Redl'       # Bootstrap current model ['Freidberg', 'Segal', 'Sauter', 'Redl']

    Option_Kappa       : str   = 'Wenninger'   # Elongation model: 'Wenninger', 'Stambaugh', 'Freidberg', 'Manual'
    κ_manual           : float = 1.9           # Elongation (Manual mode only) [-]

    # ── 2b. Plasma geometry model ──────────────────────────────────────────────
    # Controls how volume integrals and cross-section area elements are computed.
    # Affects: P_fus, <nT>, W_th, I_bs, P_rad, first-wall surface area.
    # Does NOT affect scaling-law inputs (nbar, Tbar remain volume/line averages).
    #
    # 'Academic' : Cylindrical-torus approximation, uniform kappa_edge.
    #              V'(rho) = 4π²R₀a²κ_edge·ρ   →   V = 2π²R₀κa²  (Wesson)
    #              Fast, analytical.  Consistent with PROCESS (Kovari 2014).
    #              Valid to ~10%: elongation penetrates efficiently to the core
    #              (κ_core ≈ κ₉₅ ≈ κ_edge/1.12) and δ enters only at 2nd order.
    #
    # 'D0FUS'    : Full Miller flux-surface parameterisation (Miller et al. 1998).
    #              Radial shape profiles built with PCHIP (C¹, monotone):
    #                κ(ρ): flat-core → κ₉₅ for ρ ≤ ρ₉₅, rising to κ_edge
    #                δ(ρ): zero on-axis (Grad-Shafranov constraint) → δ₉₅ → δ_edge
    #              V'(ρ) computed numerically from the Jacobian |∂(R,Z)/∂(ρ,θ)|.
    #              Physically: core flux surfaces are more circular than the LCFS
    #              (κ(0) ≈ 1, δ(0) = 0 exactly — Ball & Parra 2015).
    #
    # References:
    #   Miller et al., Phys. Plasmas 5, 973 (1998)  — parameterisation
    #   Ball & Parra, PPCF 57, 035006 (2015)        — κ/δ radial penetration
    #   Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980) — PCHIP
    #   Kovari et al., Fusion Eng. Des. 89, 3054 (2014) — PROCESS reference
    Plasma_geometry : str = 'Academic'  # 'Academic' or 'D0FUS'
    # Note: 'D0FUS' mode requires a call to precompute_Vprime() before every
    # volume integral.  This adds ~O(N_rho × N_theta) overhead per design point.
    # Use 'Academic' for large parameter scans; 'D0FUS' for single-point analysis
    # or when triangularity effects on volume (δ > 0.3) are physically relevant.

    # Grid resolution for Miller numerical integration (D0FUS mode only)
    N_rho_miller   : int = 200    # Radial grid points for V'(ρ) [-]
    N_theta_miller : int = 200    # Poloidal grid points for V'(ρ) [-]

    # ── 3. Plasma stability limits ─────────────────────────────────────────────
    betaN_limit     : float = 2.8  # Troyon normalised beta limit [% m T/MA]
    q_limit         : float = 2.5  # Kink safety factor limit (q* > 2.5 → ~q95 > 3) [-]
    Greenwald_limit : float = 1.0  # Greenwald density fraction limit [-]
    ms              : float = 0.3  # Vertical stability margin parameter [-]

    # ── 4. Plasma composition ──────────────────────────────────────────────────
    Atomic_mass : float = 2.5   # Volume-averaged ionic mass [AMU]  (D-T: 2.5)
    Zeff        : float = 2.0   # Effective plasma charge [-]
    r_synch     : float = 0.5   # Synchrotron radiation wall reflectivity [-]
    C_Alpha     : float = 5.0   # Helium ash dilution tuning parameter [-]

    # Impurity line radiation (0D bulk-plasma estimate)
    # impurity_species : None → line radiation disabled (pure D-T, no seeding).
    #                    Otherwise, one of: 'W', 'Ar', 'Ne', 'C', 'N', 'Kr'.
    # f_imp_core       : impurity concentration n_imp / n_e in the bulk plasma [-].
    #                    Typical ranges (reactor-grade):
    #                      W  : 1e-5 – 5e-4  (metals, erosion from PFCs)
    #                      Ar : 1e-3 – 5e-3  (noble gas seeding, divertor radiator)
    #                      Ne : 1e-3 – 5e-3  (noble gas seeding)
    # Caution: for high-Z impurities (W, Kr), the bulk of line radiation
    # originates in the cold SOL/divertor.  The 0D estimate should be
    # interpreted as an upper bound on the core contribution only.
    # References:
    #   Pütterich et al., Nucl. Fusion 50 (2010) 025012
    #   Mavrin, Rad. Eff. Def. Solids 173 (2018) 388
    impurity_species : str = ''     # Comma-separated species: 'W', 'W, Ne', '' = none
    f_imp_core       : str = ''     # Matching concentrations: '5e-5', '1e-5, 3e-3'

    # ── 5. Magnetic flux model ─────────────────────────────────────────────────
    Ce        : float = 0.30     # Ejima constant (resistive ramp-up flux) [-]
                                 # Ψ_res = Ce * μ0 * R0 * Ip
                                 # Typical: ITER 0.45, EU-DEMO 0.30, CFETR 0.30
    eta_model : str   = 'sauter'  # Plasma resistivity model for R_eff and li
                                 # 'old' | 'spitzer' | 'sauter' | 'redl' (recommended)
    # Plasma initiation flux: Ψ_PI = 2π * R0 * E_phi_BD * t_BD
    E_phi_BD  : float = 0.5     # Toroidal electric field at breakdown [V/m]
                                 # Ref: Lloyd et al., PPCF 33(11), 1991
                                 # Range: 0.3 V/m (aggressive) to 1.0 V/m (conservative)
    t_BD      : float = 0.5     # Breakdown duration [s]  → Ψ_PI ≈ 10 Wb (ITER-scale)
    # PF coil flux contribution
    C_PF      : float = 0.9     # Empirical PF coil scaling coefficient [-]
                                 # Ψ_PF = C_PF * μ0 * R0 * Ip
                                 # Calibrated: ITER ~0.95, CFETR ~0.82; uncertainty ±30%
                                 
    # ── 6. Structural materials ────────────────────────────────────────────────
    Chosen_Steel         : str   = '316L'   # Structural steel grade '316L' , 'N50H', 'Manual'
    σ_manual             : float = 1500.0    # Manual steel yield strength [MPa] (used only if Chosen_Steel='Manual')
    nu_Steel             : float = 0.29     # Steel Poisson's ratio [-]
    Young_modul_Steel    : float = 200e9    # Steel Young's modulus [Pa]
    Young_modul_GF       : float = 90e9     # S-glass fiber Young's modulus [Pa]
    fatigue_CS           : float = 2.0      # CS fatigue knockdown factor (pulsed & wedging only) [-]

    # ── 7. TF coil engineering ─────────────────────────────────────────────────
    Radial_build_model   : str   = 'D0FUS'  # Stress model: 'academic', 'D0FUS', 'CIRCE'
    Choice_Buck_Wedg     : str   = 'Wedging'# Mechanical config: 'Wedging', 'Bucking', 'Plug'
    Supra_choice         : str   = 'Nb3Sn'  # Superconductor: 'NbTi', 'Nb3Sn', 'REBCO', 'Manual'
    J_wost_Manual        : float = 50e6     # Engineering current density (Manual SC only) [A/m²]

    coef_inboard_tension : float = 0.5      # Inboard/outboard vertical stress ratio [-]
    F_CClamp             : float = 0.0      # C-clamp structural limit [N]
                                            # Typical: 30e6 (DDD) to 60e6 N (Bachmann 2023, FED)
    n_TF                 : float = 1.0      # TF conductor shape factor (1 = square, 0 = optimal) [-]
    c_BP                 : float = 0.07     # Backplate thickness [m]

    # ── 8. Central Solenoid ────────────────────────────────────────────────────
    Gap      : float = 0.10  # CS–TF mechanical clearance [m]
    n_CS     : float = 1.0   # CS conductor shape factor (1 = square, 0 = optimal) [-]
    N_sub_CS : int   = 6     # Number of CS sub-modules [-]
    cs_axial_stress : bool = False  # Include fringe-field axial stress in CS Tresca [bool]

    # ── 9. Superconductor operating conditions ─────────────────────────────────
    T_helium  : float = 4.2   # Liquid helium bath temperature [K]
    Marge_T_He: float = 0.3   # Temperature margin from 10-bar He operation [K]
    f_He      : float = 0.3   # Helium channel area fraction [-]
    f_In      : float = 0.15  # Insulation area fraction [-]

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
    RRR                      : float = 100.0  # Copper residual resistivity ratio [-] (pessimistic)
    Dump_resistor_subdivision: int   = 2      # TF coils per dump resistor [-]

    # ── 10. Power conversion ───────────────────────────────────────────────────
    eta_T     : float = 0.40   # Thermal-to-electric conversion efficiency [-]
                               # (steam turbine + generator, typical 0.35–0.45)
    M_blanket : float = 1.0    # Blanket energy multiplication factor [-]
                               # Net thermal power: P_th = M_blanket × P_fus
                               # Arises from exothermic ⁶Li + n reactions in the TBM.
                               # Typical: 1.0 (conservative/no credit) to 1.3 (HCPB blanket)
                               # Hernandez et al., Nucl. Fusion 57 (2017) 016011
    eta_RF    : float = 0.4    # Wall-plug efficiency of heating/CD systems [-]
                               # P_wallplug = P_CD / eta_RF
                               # Used in f_P_elec:
                               #   P_elec = η_T × M_blanket × P_fus − P_CD / η_RF
                               # Typical values by source:
                               #   LH  klystron  : 0.50–0.60
                               #   EC  gyrotron  : 0.40–0.55
                               #   NBI injector  : 0.25–0.40
                               #   ICR amplifier : 0.70–0.85
                               # Use the power-weighted average for a mixed heating scenario.
                               # Default 0.40: conservative estimate for an EC-dominated system.
                               # Gormezano et al. (ITER PIPB Ch. 6), NF 47 (2007) S285

    # ── 11. Multi-source current drive ────────────────────────────────────────
    CD_source   : str   = 'LHCD'   # Active CD model: 'LHCD' | 'ECCD' | 'NBCD' | 'Multi'

    # --- Pulsed mode: fixed plasma power per source [MW] ---
    # Used when Operation_mode = 'Pulsed' and CD_source = 'Multi'.
    # P_CD_total = P_LH + P_ECRH + P_NBI + P_ICRH  (P_aux_input ignored).
    P_LH   : float = 0.0      # LHCD plasma power [MW]
    P_ECRH : float = 0.0      # ECRH plasma power [MW]
    P_NBI  : float = 0.0      # NBI injected power [MW]  (5 % losses applied internally)
    P_ICRH : float = 0.0      # ICRH plasma power [MW]   (heating only, no current drive)

    # --- Steady-State mode: power fractions per source [-] ---
    # Used when Operation_mode = 'Steady-State' and CD_source = 'Multi'.
    # The solver determines P_CD_total from the current-drive requirement;
    # individual powers are then P_i = (f_heat_i / Σf) * P_CD_total.
    # Normalisation is applied internally — fractions need not sum to exactly 1.
    # ICRH fraction contributes to heating but not to current drive (γ_ICR = 0).
    f_heat_LH  : float = 0.25   # Heating fraction allocated to LHCD [-]
    f_heat_EC  : float = 0.25   # Heating fraction allocated to ECCD [-]
    f_heat_NBI : float = 0.25   # Heating fraction allocated to NBCD [-]
    f_heat_ICR : float = 0.25   # Heating fraction allocated to ICRH [-]

    # ECCD deposition and injection parameters
    rho_EC : float = 0.3      # Normalised deposition radius [-]
    C_EC   : float = 0.32     # Pre-factor (O-mode, tangential; calibrate vs TRAVIS/TORBEAM)

    # NBCD deposition and beam parameters
    rho_NBI    : float = 0.3      # Normalised deposition radius [-]
    A_beam     : int   = 2        # Beam ion mass number: 1 = H, 2 = D, 3 = T
    E_beam_keV : float = 500.0    # Beam injection energy [keV]
    C_NBI      : float = 0.19     # Pre-factor (tangential co-injection; Cordey 1982)

    # ── 12. Plasma-facing components ───────────────────────────────────────────
    theta_deg : float = 2.7      # Divertor strike-point grazing angle [deg]
    # Refs:
    #   Reiter, "Basic Fusion Boundary Plasma Physics," ITER School (2019)
    #   SOLPS-ITER simulations, J. Nucl. Mater. (2024)

    # ── 13. Maintenance constraints ────────────────────────────────────────────
    ripple_adm : float = 0.01    # Admissible toroidal field ripple [-]  (1%)
    L_min      : float = 3.6     # Minimum toroidal maintenance access width [m]

# Module-level default instance — import and reuse rather than reinstantiating
DEFAULT_CONFIG = GlobalConfig()