"""
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Import

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_parameterization import *

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_parameterization import *

#%% print

if __name__ == "__main__":
    print("##################################################### J Model ##########################################################")

#%% Steel

def Steel(Chosen_Steel):
    if Chosen_Steel == '316L':
        σ = 660*1e6        # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'N50H':
        σ = 1000*1e6       # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'Manual':
        σ = σ_manual*1e6   # Mechanical limit of the steel considered in [Pa]
    else : 
        print('Choose a valid steel')
    return(σ)

#%% Critical Current Density Scaling Laws for Superconductors
"""
Critical Current Density Scaling Laws for Superconductors
==========================================================

Scaling laws for Nb3Sn, NbTi, and REBCO superconductors used in fusion magnet design.

References
----------
[1] Corato, V. et al. (2016). "Common operating values for DEMO magnets design 
    for 2016". EUROfusion, IDM Ref. EFDA_D_2MMDTG.
    
[2] Fleiter, J. & Ballarino, A. (2014). "Parameterization of the critical surface 
    of REBCO conductors from Fujikura". CERN EDMS 1426239.
    
[3] Bajas, H. & Tommasini, D. (2022). "The SHiP spectrometer magnet – 
    Superconducting options". CERN-SHiP-NOTE-2022-001, EDMS 2440157.
    
[4] Tsuchiya, K. et al. (2017). "Critical current measurement of commercial 
    REBCO conductors at 4.2 K". Cryogenics 85, 1-7.
    
[5] Senatore, C. et al. (2024). "REBCO tapes for applications in ultra-high 
    fields: critical current surface and scaling relations". 
    Supercond. Sci. Technol. 37, 115013.

"""

def Jc_Nb3Sn(B, T, Eps):
    """
    Critical current density for Nb3Sn (EU-DEMO WST strand parametrization).
    
    Formula:
        Jc = (C/B) · s(ε) · (1-t^1.52) · (1-t²) · b^p · (1-b)^q
    
    Parameters
    ----------
    B : float or array
        Magnetic field [T]
    T : float or array
        Temperature [K]
    Eps : float
        Applied strain [-]
        
    Returns
    -------
    Jc : float or array
        Critical current density [A/m²] on superconducting (non-Cu) area
        
    Reference
    ---------
    Corato et al. (2016), Table 2.1 - EU-DEMO WST strand parameters
    """
    B = np.atleast_1d(np.array(B, dtype=float))
    T = np.atleast_1d(np.array(T, dtype=float))
    
    # EU-DEMO WST strand parameters [Ref. 1]
    Ca1, Ca2 = 50.06, 0.0
    Eps0a = 0.00312
    Bc2m, Tcm = 33.24, 16.34      # [T], [K]
    C = 83075                      # [AT/mm²] on SC area
    p, q = 0.593, 2.156
    
    # Strain function s(ε)
    s_eps = 1 + (Ca1 / (1 - Ca1 * Eps0a)) * (
        np.sqrt(Eps0a**2) - np.sqrt((Eps)**2 + Eps0a**2)
    )
    
    # Critical temperature and field
    Tc0_eps = Tcm * s_eps**(1/3)
    t = np.clip(T / Tc0_eps, 0, 1 - 1e-10)
    
    Bc2_T_eps = Bc2m * s_eps * (1 - t**1.52)
    b = np.clip(B / Bc2_T_eps, 0, 1 - 1e-10)
    
    # Jc formula
    Jc = (C / B) * s_eps * (1 - t**1.52) * (1 - t**2) * b**p * (1 - b)**q
    Jc = Jc * 1e6  # AT/mm² → A/m²
    Jc = np.where((t >= 1) | (b >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)

def Jc_NbTi(B, T):
    """
    Critical current density for NbTi (ITER/EU-DEMO parametrization).
    
    Formula:
        Jc = (C0/B) · (1 - t^1.7)^γ · b^α · (1-b)^β
    
    Parameters
    ----------
    B : float or array
        Magnetic field [T]
    T : float or array
        Temperature [K]
        
    Returns
    -------
    Jc : float or array
        Critical current density [A/m²] on NbTi (non-Cu) area
        
    Reference
    ---------
    Corato et al. (2016), Section 2.2 - ITER/EU-DEMO parameters
    """
    B = np.atleast_1d(np.array(B, dtype=float))
    T = np.atleast_1d(np.array(T, dtype=float))
    
    # ITER/EU-DEMO parameters [Ref. 1]
    Tc0, Bc20 = 9.03, 14.61       # [K], [T]
    C0 = 168512                    # [A/mm²] on NbTi area
    alpha, beta, gamma = 1.0, 1.54, 2.1
    
    t = np.clip(T / Tc0, 0, 1 - 1e-10)
    Bc2_T = Bc20 * (1 - t**1.7)
    b = np.clip(B / Bc2_T, 0, 1 - 1e-10)
    
    Jc = (C0 / B) * (1 - t**1.7)**gamma * b**alpha * (1 - b)**beta
    Jc = Jc * 1e6  # A/mm² → A/m²
    Jc = np.where((t >= 1) | (b >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)

def Jc_Rebco(B, T, Tet):
    """
    Critical current density for REBCO tape (Fleiter/CERN parametrization).
    
    Parameters
    ----------
    B : float or array
        Magnetic field [T]
    T : float or array
        Temperature [K]
    Tet : float
        Field angle [rad]: 0 = B⊥tape (B//c), π/2 = B//tape (B//ab)
        
    Returns
    -------
    Jc : float or array
        Engineering critical current density [A/m²] on tape cross-section
        
    Notes
    -----
    - Parameters calibrated on Fujikura 12mm tape (FESC series)
    - SC layer: ~2 µm REBCO
    - Total tape thickness: ~100 µm (including Hastelloy substrate, Cu, Ag)
    - α parameters give Ic when multiplied by tape width
    
    References
    ----------
    [2] Fleiter & Ballarino (2014), CERN EDMS 1426239
    [3] Bajas & Tommasini (2022), CERN-SHiP-NOTE-2022-001, Table 3
    """
    B = np.atleast_1d(np.array(B, dtype=float))
    T = np.atleast_1d(np.array(T, dtype=float))
    
    # Geometry (Fujikura 12mm tape)
    w_tape = 12e-3      # Tape width [m]
    t_tape = 100e-6     # Total tape thickness [m] (~100 µm)
    A_tape = w_tape * t_tape  # Cross-section [m²]
    
    # Fleiter 2014 parameters (Table 3, page 49 of Ref. [3])
    Tc0 = 93.0          # Critical temperature [K]
    n = 1.0
    
    # c-axis (B perpendicular to tape) parameters
    # Note: α scaled to match Tsuchiya (2017) Fujikura data at 12T, 4.2K
    Bi0c = 140.0        # Irreversibility field at T=0 [T]
    Alfc = 3.0e6 * w_tape    # [A·T] - calibrated to give Ic (~800-1000 A/mm² @ 12T)
    pc = 0.5
    qc = 2.5
    gamc = 2.44
    
    # ab-plane (B parallel to tape) parameters
    Bi0ab = 250.0       # [T]
    Alfab = 110e6 * w_tape   # [A·T] - scaled proportionally
    pab = 1.0
    qab = 5.0
    gamab = 1.63
    
    # Temperature exponents
    n1 = 1.4
    n2 = 4.45
    a = 0.1
    
    # Angular interpolation parameters
    Nu = 0.857
    g0, g1, g2, g3 = 0.03, 0.25, 0.06, 0.058
    
    # Reduced temperature
    tred = T / Tc0
    
    # Irreversibility fields
    Bic = Bi0c * (1 - tred**n)
    Biab = Bi0ab * ((1 - tred**n1)**n2 + a * (1 - tred**n))
    
    # Reduced fields (clipped to avoid singularities)
    bredc = np.clip(B / Bic, 1e-10, 1 - 1e-10)
    bredab = np.clip(B / Biab, 1e-10, 1 - 1e-10)
    
    # Critical currents [A] for each orientation
    Icc = (Alfc / B) * bredc**pc * (1 - bredc)**qc * (1 - tred**n)**gamc
    Icab = (Alfab / B) * bredab**pab * (1 - bredab)**qab * \
           ((1 - tred**n1)**n2 + a * (1 - tred**n))**gamab
    
    # Angular interpolation between c-axis and ab-plane
    g = g0 + g1 * np.exp(-g2 * np.exp(g3 * T) * B)
    Ic = Icc + (Icab - Icc) / (1 + (np.abs(Tet - np.pi/2) / g)**Nu)
    
    # Convert to engineering Jc on tape cross-section
    Jc = Ic / A_tape
    
    # Zero outside valid range
    Jc = np.where((tred >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)

def Jc(Supra_choice, B_supra, T_He, Jc_Manual):
    """
    Main interface: Critical current density with temperature margins.
    
    Parameters
    ----------
    Supra_choice : str
        Superconductor type: "Nb3Sn", "NbTi", or "Rebco"
    B_supra : float or array
        Magnetic field [T]
    T_He : float
        Helium bath temperature [K]
        
    Returns
    -------
    Jc : float or array
        Critical current density [A/m²]
        - Nb3Sn: on superconducting (non-Cu) area
        - NbTi: on non-Cu area  
        - REBCO: on tape cross-section (engineering Je)
    """
    if Supra_choice == "Nb3Sn":
        return Jc_Nb3Sn(B_supra, T_He + Marge_T_Helium + Marge_T_Nb3Sn, Eps)
    elif Supra_choice == "NbTi":
        return Jc_NbTi(B_supra, T_He + Marge_T_Helium + Marge_T_NbTi)
    elif Supra_choice == "Rebco":
        return Jc_Rebco(B_supra, T_He + Marge_T_Helium + Marge_T_Rebco, Tet)
    elif Supra_choice == 'Manual':
        return Jc_Manual
    else:
        raise ValueError(f"Unknown superconductor: {Supra_choice}")


#%% J Validation
if __name__ == "__main__":

    # ═══════════════════════════════════════════════════════════════════════════
    # BASIC CROSSCHECK – EU-DEMO / ITER SUPERCONDUCTOR SCALINGS
    # ═══════════════════════════════════════════════════════════════════════════
    print("="*70)
    print("SUPERCONDUCTOR CRITICAL CURRENT DENSITY – VALIDATION CROSSCHECK")
    print("="*70)
    
    # NbTi – ITER PF
    B, T = 5.0, 4.2
    Jc_calc = Jc_NbTi(B, T) / 1e6   # → MA/m² = A/mm²
    print(f"NbTi  (ITER PF coils) {B:.1f} T, {T:.1f} K")
    print(f"  Calculated Jc                  : {Jc_calc:7.0f} A/mm²")
    print(f"  Tested ITER PF reference value :    2900 A/mm²")
    print("="*70)
    
    # Nb3Sn – ITER TF
    B, T = 11.8, 4.2
    Jc_calc = Jc_Nb3Sn(B, T, Eps=-0.0035) / 1e6
    print(f"Nb3Sn (ITER TF coils) {B:.1f} T, {T:.1f} K")
    print(f"  Calculated Jc                   : {Jc_calc:7.0f} A/mm²")
    print(f"  Tested ITER TF reference value  :     900 A/mm²")
    print("="*70)
    
    # REBCO
    B, T = 18.0, 4.2
    Jc_calc_18T = Jc_Rebco(B, T, Tet=0) / 1e6
    print(f"REBCO tape (B//ab) {B:.1f} T, {T:.1f} K")
    print(f"  Calculated Jc             : {Jc_calc_18T:7.0f} A/mm²")
    print(f"  Tested SuperPower tape    :     650 A/mm²")
    print("="*70)

    # ─────────────────────────────────────────────────────────────
    # ITER TF coil - Current density cascade
    # ─────────────────────────────────────────────────────────────
    print("ITER TF COIL – Current Density Cascade\n")
    
    B, T = 11.8, 4.2
    Jc_nonCu = Jc("Nb3Sn", B, T, Jc_Manual)
    # Cascade
    J_strand = Jc_nonCu * f_Cu_Non_Cu
    J_all_strand = J_strand * f_Cu_Strand
    J_cable = J_all_strand * f_Cool
    J_no_steel = J_cable * f_In
    levels = ["Non-Cu", "Su Strand", "All Strands", "Cable", "Non-Steel"]
    Jvals = [Jc_nonCu, J_strand, J_all_strand, J_cable, J_no_steel]
    Jvals_MA = [j/1e6 for j in Jvals]
    
    print("Current densities [MA/m²]:")
    for name, val in zip(levels, Jvals_MA):
        print(f"  {name:12s}: {val:5.1f}")
    print("Reference value for ITER design:")
    # Source: Alexandre Torre lecture on current density
    # "Master on Fusion and Plasma science Superconductors for Fusion [SCF]"
    # TD 3
    print("Non-Cu        : 318")
    print("Su strand     : 159")
    print("All strands   : 106")
    print("Cable         : 71")
    print("Non-Steel     : 56")
    
    # Factors
    details = [
        "Non-Cu",
        f"+ Cu in strand",
        f"+ Cu strands",
        f"+ He",
        f"+Insulation",
    ]
    plt.figure(figsize=(6,4))
    bars = plt.bar(
        levels,
        Jvals_MA,
        width=0.3,
        color="black",
        edgecolor="black",
        linewidth=1.3
    )
    plt.ylabel("J [MA/m²]")
    plt.ylim(0, 500)
    plt.title(f"ITER TF Current Density Cascade\n(B = {B} T, T = {T} K)")
    plt.grid(axis="y", alpha=0.25)
    for bar, val, txt in zip(bars, Jvals_MA, details):
        plt.annotate(
            f"{val:.0f}\n({txt})",
            xy=(bar.get_x() + bar.get_width()/2, val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )
    plt.tight_layout()
    plt.show()
    
    # —––––––––––––––––––––––––––––––––––––
    # Maglab Crosscheck
    # —––––––––––––––––––––––––––––––––––––
    B_vals = np.linspace(0.5, 45, 100)
    B_mesh = np.meshgrid(B_vals)
    
    # —––––––––––––––––––––––––––––––––––––
    # 2D plot at 4.2 K
    T0 = 4.2
    J_NbTi_42 = Jc_NbTi(B_vals, T0)/1e6
    J_Nb3Sn_42 = Jc_Nb3Sn(B_vals, T0, Eps=-0.003)/1e6
    J_Rebco_42 = Jc_Rebco(B_vals, T0, Tet=0)/1e6
    Whole_wire = 0.5
    
    plt.figure(figsize=(6,4))
    plt.plot(B_vals, J_NbTi_42 * Whole_wire, label='NbTi @ 4.2 K', lw=2)
    plt.plot(B_vals, J_Nb3Sn_42 * Whole_wire, label='Nb3Sn @ 4.2 K', lw=2)
    plt.plot(B_vals, J_Rebco_42 , label='Rebco @ 4.2 K', lw=2)
    plt.xlabel('Magnetic field B (T)')
    plt.ylabel('Wire or Tape current density (MA/m²)')
    plt.title('MAGLAB benchmark at 4.2 K')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 45)
    plt.ylim(10, 1e4)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
    
#%% Print
        
if __name__ == "__main__":
    print("##################################################### CIRCE Model ##########################################################")
    
#%% CIRCE 0D module

"""
F_CIRCE0D - Analytical stress solver for multilayer thick cylinders
Developed by B.Boudes
Addapted by T.Auclair
====================================================================

This module implements the analytical solution for the elasticity problem of
concentric cylinder stacks subjected to:
- Internal and external pressures (Pi, Pe)
- Electromagnetic body loads (J × B) with linear radial profile

Theory
------
For each layer, the Lorentz body force is modeled as:
    f_r(r) = J(r) × B(r) ≈ K1·r + K2

where K1 and K2 depend on the field configuration (increasing/decreasing).

The axisymmetric equilibrium equation:
    dσr/dr + (σr - σθ)/r + f_r = 0

is solved analytically with Hooke's law, yielding expressions for σr, σθ, 
and ur as functions of integration constants.

Boundary conditions are:
- σr(ri) = -Pi  (internal pressure)
- σr(re) = -Pe  (external pressure)
- Continuity of ur at interfaces

The code handles multilayer structures where each layer can have different:
- Young's modulus E (e.g., conductor vs structural steel)
- Current density J (can be zero for passive layers)
- Magnetic field B
- Field profile configuration

References
----------
- Timoshenko & Goodier, "Theory of Elasticity"
- CIRCE B.Boudes (CEA)

"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class LayerResult:
    """Results for a single layer."""
    r: np.ndarray          # Radial positions [m]
    sigma_r: np.ndarray    # Radial stress [Pa]
    sigma_t: np.ndarray    # Hoop stress [Pa]
    u_r: np.ndarray        # Radial displacement [m]


def compute_body_load_coefficients(
    J: float, 
    B: float, 
    r_inner: float, 
    r_outer: float, 
    config: int
) -> Tuple[float, float]:
    """
    Compute K1 and K2 coefficients for the linear body load profile.
    
    The volumetric force profile is: f(r) = K1·r + K2
    
    Parameters
    ----------
    J : float
        Average current density in the layer [A/m²]
    B : float
        Characteristic magnetic field [T]
    r_inner : float
        Inner radius of the layer [m]
    r_outer : float
        Outer radius of the layer [m]
    config : int
        Field profile configuration:
        - 1 : Decreasing field (B max at r_inner) → typical for CS
        - 2 : Increasing field (B max at r_outer) → typical for TF
        
    Returns
    -------
    K1, K2 : float
        Linear profile coefficients for f(r) = K1·r + K2
        
    Notes
    -----
    For an infinite solenoid, B(r) is linear in r.
    - Config 1: B(r) = B_max·(r_outer - r)/(r_outer - r_inner) → decreasing
    - Config 2: B(r) = B_max·(r - r_inner)/(r_outer - r_inner) → increasing
    
    For passive layers (J=0 or B=0), both K1 and K2 are zero.
    """
    dr = r_outer - r_inner
    
    # Handle passive layers (no electromagnetic load)
    if J == 0 or B == 0 or dr == 0:
        return 0.0, 0.0
    
    K1 = J * B / dr
    
    if config == 1:
        # Field maximum at r_inner (decreasing outward)
        K2 = -J * B * r_inner / dr
    else:
        # Field maximum at r_outer (increasing outward)
        K2 = -J * B * r_outer / dr
    
    return K1, K2


def _build_continuity_rhs_first(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    Pi: float, config: List[int]
) -> float:
    """
    Build the right-hand side of the continuity equation for the first interface.
    
    Includes the internal pressure Pi contribution.
    """
    K1_prev, K2_prev = compute_body_load_coefficients(J_prev, B_prev, ri, r0, config[0])
    K1_curr, K2_curr = compute_body_load_coefficients(J_curr, B_curr, r0, re, config[1])
    
    # Contribution from outer layer (n)
    term_ext = (
        (1 + nu) / E_curr * re**2 * (K1_curr * (nu + 3) / 8 + K2_curr * (nu + 2) / (3 * (re + r0))) +
        (1 - nu) / E_curr * (
            (K2_curr * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) +
            (re**2 + r0**2) * K1_curr * (nu + 3) / 8
        )
    )
    
    # Contribution from inner layer (n-1)
    term_int = (
        (1 + nu) / E_prev * ri**2 * (K1_prev * (nu + 3) / 8 + K2_prev * (nu + 2) / (3 * (r0 + ri))) +
        (1 - nu) / E_prev * (
            (K2_prev * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) +
            (r0**2 + ri**2) * K1_prev * (nu + 3) / 8
        )
    )
    
    # Jump terms at interface (discontinuity in material properties)
    jump_term = (
        (1 - nu**2) / 8 * r0**2 * (K1_prev / E_prev - K1_curr / E_curr) +
        (1 - nu**2) / 3 * r0 * (K2_prev / E_prev - K2_curr / E_curr)
    )
    
    # Internal pressure contribution
    pressure_term = -Pi * (-2 * ri**2) / (E_prev * (r0**2 - ri**2))
    
    return term_ext - term_int + jump_term + pressure_term


def _build_continuity_rhs_last(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    Pe: float, config: List[int]
) -> float:
    """
    Build the right-hand side for the last interface.
    
    Includes the external pressure Pe contribution.
    """
    K1_prev, K2_prev = compute_body_load_coefficients(J_prev, B_prev, ri, r0, config[0])
    K1_curr, K2_curr = compute_body_load_coefficients(J_curr, B_curr, r0, re, config[1])
    
    # Same terms as internal interfaces
    term_ext = (
        (1 + nu) / E_curr * re**2 * (K1_curr * (nu + 3) / 8 + K2_curr * (nu + 2) / (3 * (re + r0))) +
        (1 - nu) / E_curr * (
            (K2_curr * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) +
            (re**2 + r0**2) * K1_curr * (nu + 3) / 8
        )
    )
    
    term_int = (
        (1 + nu) / E_prev * ri**2 * (K1_prev * (nu + 3) / 8 + K2_prev * (nu + 2) / (3 * (r0 + ri))) +
        (1 - nu) / E_prev * (
            (K2_prev * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) +
            (r0**2 + ri**2) * K1_prev * (nu + 3) / 8
        )
    )
    
    jump_term = (
        (1 - nu**2) / 8 * r0**2 * (K1_prev / E_prev - K1_curr / E_curr) +
        (1 - nu**2) / 3 * r0 * (K2_prev / E_prev - K2_curr / E_curr)
    )
    
    # External pressure contribution
    pressure_term = -Pe * (-2 * re**2) / (E_curr * (re**2 - r0**2))
    
    return term_ext - term_int + jump_term + pressure_term


def _build_continuity_rhs_middle(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    config: List[int]
) -> float:
    """
    Build the right-hand side for internal interfaces.
    
    No pressure terms (pressures are at boundaries only).
    """
    K1_prev, K2_prev = compute_body_load_coefficients(J_prev, B_prev, ri, r0, config[0])
    K1_curr, K2_curr = compute_body_load_coefficients(J_curr, B_curr, r0, re, config[1])
    
    term_ext = (
        (1 + nu) / E_curr * re**2 * (K1_curr * (nu + 3) / 8 + K2_curr * (nu + 2) / (3 * (re + r0))) +
        (1 - nu) / E_curr * (
            (K2_curr * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) +
            (re**2 + r0**2) * K1_curr * (nu + 3) / 8
        )
    )
    
    term_int = (
        (1 + nu) / E_prev * ri**2 * (K1_prev * (nu + 3) / 8 + K2_prev * (nu + 2) / (3 * (r0 + ri))) +
        (1 - nu) / E_prev * (
            (K2_prev * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) +
            (r0**2 + ri**2) * K1_prev * (nu + 3) / 8
        )
    )
    
    jump_term = (
        (1 - nu**2) / 8 * r0**2 * (K1_prev / E_prev - K1_curr / E_curr) +
        (1 - nu**2) / 3 * r0 * (K2_prev / E_prev - K2_curr / E_curr)
    )
    
    return term_ext - term_int + jump_term


def _build_stiffness_row_first(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float
) -> List[float]:
    """
    Build the first row of the stiffness matrix.
    
    Coefficients for [P1, P2] (P0 = Pi is not a variable).
    """
    # Diagonal coefficient (continuity at interface 1)
    diag = (
        ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
        ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
    )
    
    # Off-diagonal coefficient (coupling with P2)
    off_diag = (-2 * re**2) / (E_curr * (re**2 - r0**2))
    
    return [diag, off_diag]


def _build_stiffness_row_last(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float
) -> List[float]:
    """
    Build the last row of the stiffness matrix.
    
    Coefficients for [P_{n-2}, P_{n-1}] (P_n = Pe is not a variable).
    """
    # Off-diagonal coefficient (coupling with P_{n-2})
    off_diag = (-2 * ri**2) / (E_prev * (r0**2 - ri**2))
    
    # Diagonal coefficient
    diag = (
        ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
        ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
    )
    
    return [off_diag, diag]


def _build_stiffness_row_middle(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float
) -> List[float]:
    """
    Build an intermediate row of the stiffness matrix.
    
    Tridiagonal coefficients for [P_{i-1}, P_i, P_{i+1}].
    """
    # Sub-diagonal coefficient
    sub_diag = (-2 * ri**2) / (E_prev * (r0**2 - ri**2))
    
    # Diagonal coefficient
    diag = (
        ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
        ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
    )
    
    # Super-diagonal coefficient
    super_diag = (-2 * re**2) / (E_curr * (re**2 - r0**2))
    
    return [sub_diag, diag, super_diag]


def compute_layer_stresses(
    r: np.ndarray,
    ri: float, re: float,
    P_inner: float, P_outer: float,
    K1: float, K2: float,
    E: float, nu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stresses and displacements within a single layer.
    
    Parameters
    ----------
    r : np.ndarray
        Radial positions [m]
    ri, re : float
        Inner and outer radii of the layer [m]
    P_inner, P_outer : float
        Pressures at inner and outer interfaces [Pa]
    K1, K2 : float
        Body load profile coefficients
    E : float
        Young's modulus [Pa]
    nu : float
        Poisson's ratio [-]
        
    Returns
    -------
    sigma_r, sigma_t, u_r : np.ndarray
        Radial stress, hoop stress [Pa] and radial displacement [m]
        
    Notes
    -----
    Generalized Lamé solution with linear body load.
    For passive layers (K1=K2=0), this reduces to the classical Lamé solution.
    """
    # Integration constants (modified Lamé equations)
    C1 = (
        K1 * (nu + 3) / 8 + 
        K2 * (nu + 2) / (3 * (re + ri)) - 
        (P_inner - P_outer) / (re**2 - ri**2)
    )
    
    C2 = (
        (K2 * (nu + 2) / 3) * (re**2 + ri**2 + re * ri) / (re + ri) +
        (P_outer * re**2 - P_inner * ri**2) / (re**2 - ri**2) +
        (re**2 + ri**2) * K1 * (nu + 3) / 8
    )
    
    # Radial stress: σr = A/r² + B·r² + C·r + D
    sigma_r = (
        re**2 * ri**2 / r**2 * C1 + 
        K1 * (nu + 3) / 8 * r**2 + 
        K2 * (nu + 2) / 3 * r - 
        C2
    )
    
    # Hoop stress: σθ = -A/r² + B'·r² + C'·r + D
    sigma_t = (
        -re**2 * ri**2 / r**2 * C1 + 
        K1 * (3 * nu + 1) / 8 * r**2 + 
        K2 * (2 * nu + 1) / 3 * r - 
        C2
    )
    
    # Radial displacement (generalized plane strain Hooke's law)
    u_r = r / E * (
        -re**2 * ri**2 / r**2 * C1 * (1 + nu) + 
        (1 - nu**2) / 8 * K1 * r**2 +
        (1 - nu**2) / 3 * K2 * r - 
        C2 * (1 - nu)
    )
    
    return sigma_r, sigma_t, u_r


def F_CIRCE0D(
    n_points: int,
    R: List[float],
    J: List[float],
    B: List[float],
    Pi: float,
    Pe: float,
    E: List[float],
    nu: float,
    config: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main stress solver for multilayer cylinders.
    
    Solves the axisymmetric elasticity problem for a stack of n concentric
    cylinders subjected to pressures and electromagnetic body loads (J × B).
    
    This function supports heterogeneous structures where:
    - Active layers (conductors) have J > 0 and experience Lorentz forces
    - Passive layers (structural) have J = 0 and only transmit stresses
    
    Parameters
    ----------
    n_points : int
        Number of radial discretization points per layer
    R : List[float]
        Interface radii [m], length (n_layers + 1)
        R = [r0, r1, r2, ..., rn] where r0 is the inner radius
    J : List[float]
        Current densities per layer [A/m²]
        Use J=0 for passive/structural layers
    B : List[float]
        Characteristic magnetic fields per layer [T]
        For passive layers, this value is irrelevant (multiplied by J=0)
    Pi : float
        Internal pressure (at r = R[0]) [Pa]
    Pe : float
        External pressure (at r = R[-1]) [Pa]
    E : List[float]
        Young's moduli per layer [Pa]
        Typical values: ~50 GPa for smeared conductor, ~200 GPa for steel
    nu : float
        Poisson's ratio (assumed identical for all layers) [-]
    config : List[int]
        Field profile configuration per layer:
        - 1 : Decreasing field (max at r_inner)
        - 2 : Increasing field (max at r_outer)
        For passive layers, this value is irrelevant
        
    Returns
    -------
    sigma_r : np.ndarray
        Radial stress over the entire domain [Pa]
    sigma_t : np.ndarray
        Hoop stress over the entire domain [Pa]
    u_r : np.ndarray
        Radial displacement over the entire domain [m]
    r_vec : np.ndarray
        Corresponding radial positions [m]
    P : np.ndarray
        Pressures at interfaces (including Pi and Pe) [Pa]
        
    Raises
    ------
    ValueError
        If input dimensions are inconsistent
        
    Examples
    --------
    >>> # Two-layer case: conductor + steel jacket
    >>> R = [1.0, 1.5, 1.7]           # radii [m]
    >>> J = [50e6, 0.0]               # current density: conductor, passive steel
    >>> B = [13.0, 0.0]               # magnetic field [T]
    >>> E = [50e9, 200e9]             # Young's modulus: smeared conductor, steel
    >>> sigma_r, sigma_t, u_r, r, P = F_CIRCE0D(50, R, J, B, 0, 0, E, 0.3, [1, 1])
    """
    n_layers = len(E)
    
    # Input validation
    if len(R) != n_layers + 1:
        raise ValueError(f"R must have {n_layers + 1} elements, got {len(R)}")
    if len(J) != n_layers or len(B) != n_layers or len(config) != n_layers:
        raise ValueError("J, B, and config must have the same number of elements as E")
    
    # --- Solve for interface pressures ---
    
    if n_layers == 1:
        # Trivial case: single layer, no internal interfaces
        P = np.array([Pi, Pe])
        
    elif n_layers == 2:
        # Two-layer case: single unknown pressure P1
        # Scalar system: MG * P1 = MD
        
        ri, r0, re = R[0], R[1], R[2]
        E_prev, E_curr = E[0], E[1]
        
        # Stiffness matrix (scalar)
        MG = (
            ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
            ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
        )
        
        # Right-hand side (body loads + boundary pressures)
        MD = _build_continuity_rhs_first(
            ri, r0, re, E_prev, E_curr, nu,
            J[0], B[0], J[1], B[1], Pi, [config[0], config[1]]
        )
        MD += -Pe * (-2 * re**2) / (E_curr * (re**2 - r0**2))
        
        P1 = MD / MG
        P = np.array([Pi, P1, Pe])
        
    else:
        # General case: n_layers - 1 unknown pressures
        # Tridiagonal matrix system: MG @ P_vec = MD
        
        n_unknowns = n_layers - 1
        MG = np.zeros((n_unknowns, n_unknowns))
        MD = np.zeros(n_unknowns)
        
        for i in range(n_unknowns):
            # Interface i+1 is between layer i and layer i+1
            ri = R[i]
            r0 = R[i + 1]
            re = R[i + 2]
            E_prev = E[i]
            E_curr = E[i + 1]
            
            if i == 0:
                # First interface
                row = _build_stiffness_row_first(ri, r0, re, E_prev, E_curr, nu)
                MG[i, i:i+2] = row
                MD[i] = _build_continuity_rhs_first(
                    ri, r0, re, E_prev, E_curr, nu,
                    J[i], B[i], J[i+1], B[i+1], Pi,
                    [config[i], config[i+1]]
                )
                
            elif i == n_unknowns - 1:
                # Last interface
                row = _build_stiffness_row_last(ri, r0, re, E_prev, E_curr, nu)
                MG[i, i-1:i+1] = row
                MD[i] = _build_continuity_rhs_last(
                    ri, r0, re, E_prev, E_curr, nu,
                    J[i], B[i], J[i+1], B[i+1], Pe,
                    [config[i], config[i+1]]
                )
                
            else:
                # Intermediate interface
                row = _build_stiffness_row_middle(ri, r0, re, E_prev, E_curr, nu)
                MG[i, i-1:i+2] = row
                MD[i] = _build_continuity_rhs_middle(
                    ri, r0, re, E_prev, E_curr, nu,
                    J[i], B[i], J[i+1], B[i+1],
                    [config[i], config[i+1]]
                )
        
        # Solve linear system
        P_internal = np.linalg.solve(MG, MD)
        P = np.concatenate([[Pi], P_internal, [Pe]])
    
    # --- Compute stresses and displacements per layer ---
    
    sigma_r_list = []
    sigma_t_list = []
    u_r_list = []
    r_list = []
    
    for i in range(n_layers):
        ri = R[i]
        re = R[i + 1]
        
        K1, K2 = compute_body_load_coefficients(J[i], B[i], ri, re, config[i])
        
        r = np.linspace(ri, re, n_points)
        
        sigma_r, sigma_t, u_r = compute_layer_stresses(
            r, ri, re, P[i], P[i+1], K1, K2, E[i], nu
        )
        
        sigma_r_list.append(sigma_r)
        sigma_t_list.append(sigma_t)
        u_r_list.append(u_r)
        r_list.append(r)
    
    # Concatenate results
    sigma_r_total = np.concatenate(sigma_r_list)
    sigma_t_total = np.concatenate(sigma_t_list)
    u_r_total = np.concatenate(u_r_list)
    r_vec = np.concatenate(r_list)
    
    return sigma_r_total, sigma_t_total, u_r_total, r_vec, P


def compute_von_mises_stress(sigma_r: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    """
    Compute von Mises equivalent stress in axisymmetric conditions.
    
    For an axisymmetric stress state (σr, σθ, σz=0):
        σ_VM = sqrt(σr² + σθ² - σr·σθ)
    
    Parameters
    ----------
    sigma_r : np.ndarray
        Radial stress [Pa]
    sigma_t : np.ndarray
        Hoop stress [Pa]
        
    Returns
    -------
    sigma_vm : np.ndarray
        von Mises stress [Pa]
    """
    return np.sqrt(sigma_r**2 + sigma_t**2 - sigma_r * sigma_t)


def compute_tresca_stress(sigma_r: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    """
    Compute Tresca equivalent stress in axisymmetric conditions.
    
    Parameters
    ----------
    sigma_r : np.ndarray
        Radial stress [Pa]
    sigma_t : np.ndarray
        Hoop stress [Pa]
        
    Returns
    -------
    sigma_tresca : np.ndarray
        Tresca stress [Pa]
    """
    sigma_z = np.zeros_like(sigma_r)  # Zero axial stress
    
    # Principal stress differences
    diff1 = np.abs(sigma_r - sigma_t)
    diff2 = np.abs(sigma_t - sigma_z)
    diff3 = np.abs(sigma_z - sigma_r)
    
    return np.maximum(np.maximum(diff1, diff2), diff3)


#%% TEST CASES CIRCE

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("F_CIRCE0D - Validation Tests")
    print("=" * 70)
    
    nu = 0.3  # Poisson's ratio (common to all tests)
    
    # --- Test 1: Simple cylinder under pressure (Lamé validation) ---
    print("--- Test 1: Pressurized cylinder (Lamé solution) ---")
    
    R_lame = [1.0, 2.0]
    J_lame = [0.0]  # No current (passive)
    B_lame = [0.0]
    E_lame = [200e9]  # Steel
    Pi_lame = 100e6  # 100 MPa internal pressure
    Pe_lame = 0.0
    
    sigma_r, sigma_t, u_r, r, P = F_CIRCE0D(
        100, R_lame, J_lame, B_lame, Pi_lame, Pe_lame, E_lame, nu, [1]
    )
    
    # Analytical Lamé solution
    a, b = R_lame[0], R_lame[1]
    sigma_r_lame = Pi_lame * a**2 / (b**2 - a**2) * (1 - b**2 / r**2)
    sigma_t_lame = Pi_lame * a**2 / (b**2 - a**2) * (1 + b**2 / r**2)
    
    err_r = np.max(np.abs(sigma_r - sigma_r_lame)) / Pi_lame * 100
    err_t = np.max(np.abs(sigma_t - sigma_t_lame)) / Pi_lame * 100
    
    print(f"  Max error σr: {err_r:.2e} %")
    print(f"  Max error σθ: {err_t:.2e} %")
    print(f"  σr(ri) = {sigma_r[0]/1e6:.2f} MPa (expected: {-Pi_lame/1e6:.2f} MPa)")
    print(f"  σr(re) = {sigma_r[-1]/1e6:.2f} MPa (expected: 0 MPa)")
    
    print("="*70)
    
    # --- Test 2: CS with 3 active layers ---
    print("--- Test 2: Central Solenoid with 3 active layers ---")
    
    # ITER-like CS parameters
    R_cs = [1.3, 1.5, 1.7, 2.0]  # 3 layers
    J_cs = [40e6, 45e6, 50e6]    # Increasing current density
    B_cs = [13.0, 11.0, 8.0]    # Decreasing field
    E_cs = [50e9, 50e9, 50e9]   # Smeared conductor modulus
    Pi_cs = 0.0
    Pe_cs = 0.0
    config_cs = [1, 1, 1]  # All decreasing field profile
    
    sigma_r_cs, sigma_t_cs, u_r_cs, r_cs, P_cs = F_CIRCE0D(
        50, R_cs, J_cs, B_cs, Pi_cs, Pe_cs, E_cs, nu, config_cs
    )
    
    sigma_vm_cs = compute_von_mises_stress(sigma_r_cs, sigma_t_cs)
    
    print(f"  Max σθ: {np.max(np.abs(sigma_t_cs))/1e6:.1f} MPa")
    print(f"  Max σ_VM: {np.max(sigma_vm_cs)/1e6:.1f} MPa")
    print(f"  Max displacement: {np.max(np.abs(u_r_cs))*1000:.3f} mm")
    
    print("="*70)
    
    # --- Test 3: COMPOSITE STRUCTURE - Conductor + Steel jacket ---
    print("--- Test 3: Composite structure (Conductor + Steel jacket) ---")
    
    # Configuration: superconducting winding pack + external steel jacket
    # The conductor generates J×B forces, the steel jacket provides structural support
    
    R_composite = [1.0, 1.4, 1.5]  # Inner conductor (40 cm), steel jacket (10 cm)
    J_composite = [50e6, 0.0]      # Current only in conductor, J=0 in steel
    B_composite = [13.0, 0.0]      # Field in conductor region, irrelevant for steel
    E_composite = [50e9, 200e9]    # Smeared conductor (~50 GPa), Steel (~200 GPa)
    Pi_composite = 0.0
    Pe_composite = 0.0
    config_composite = [1, 1]      # Decreasing field in conductor
    
    sigma_r_comp, sigma_t_comp, u_r_comp, r_comp, P_comp = F_CIRCE0D(
        100, R_composite, J_composite, B_composite, 
        Pi_composite, Pe_composite, E_composite, nu, config_composite
    )
    
    sigma_vm_comp = compute_von_mises_stress(sigma_r_comp, sigma_t_comp)
    
    # Find interface index
    idx_interface = np.argmin(np.abs(r_comp - R_composite[1]))
    
    # Separate conductor and steel regions
    sigma_vm_cond = sigma_vm_comp[:idx_interface+1]
    sigma_vm_steel = sigma_vm_comp[idx_interface:]
    
    print(f"  Geometry:")
    print(f"    - Conductor: ri={R_composite[0]} m, re={R_composite[1]} m, E={E_composite[0]/1e9:.0f} GPa")
    print(f"    - Steel jacket: ri={R_composite[1]} m, re={R_composite[2]} m, E={E_composite[1]/1e9:.0f} GPa")
    print(f"  Results:")
    print(f"    Conductor - Max σ_VM: {np.max(sigma_vm_cond)/1e6:.1f} MPa")
    print(f"    Steel jacket - Max σ_VM: {np.max(sigma_vm_steel)/1e6:.1f} MPa")
    print(f"    Radial displacement at outer surface: {u_r_comp[-1]*1000:.3f} mm")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Test 1: Lamé validation
    ax1 = axes[0]
    ax1.plot(r, sigma_r/1e6, 'b-', linewidth=2, label='σr (CIRCE0D)')
    ax1.plot(r, sigma_t/1e6, 'r-', linewidth=2, label='σθ (CIRCE0D)')
    ax1.plot(r, sigma_r_lame/1e6, 'b--', linewidth=1.5, label='σr (Lamé)')
    ax1.plot(r, sigma_t_lame/1e6, 'r--', linewidth=1.5, label='σθ (Lamé)')
    ax1.set_xlabel('r [m]')
    ax1.set_ylabel('Stress [MPa]')
    ax1.set_title('Test 1: Pressurized cylinder (Lamé)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test 2: CS 3 layers
    ax2 = axes[1]
    ax2.plot(r_cs, sigma_r_cs/1e6, 'b-', linewidth=2, label='σr')
    ax2.plot(r_cs, sigma_t_cs/1e6, 'r-', linewidth=2, label='σθ')
    ax2.plot(r_cs, sigma_vm_cs/1e6, 'g--', linewidth=2, label='σ_VM')
    for ri in R_cs[1:-1]:
        ax2.axvline(ri, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('r [m]')
    ax2.set_ylabel('Stress [MPa]')
    ax2.set_title('Test 2: CS with 3 active layers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test 3: Composite (conductor + steel)
    ax3 = axes[2]
    ax3.plot(r_comp, sigma_r_comp/1e6, 'b-', linewidth=2, label='σr')
    ax3.plot(r_comp, sigma_t_comp/1e6, 'r-', linewidth=2, label='σθ')
    ax3.plot(r_comp, sigma_vm_comp/1e6, 'g--', linewidth=2, label='σ_VM')
    ax3.axvline(R_composite[1], color='orange', linestyle='-', linewidth=2)
    ax3.axvspan(R_composite[0], R_composite[1], alpha=0.2, color='blue', label='Conductor')
    ax3.axvspan(R_composite[1], R_composite[2], alpha=0.2, color='gray', label='Steel')
    ax3.set_xlabel('r [m]')
    ax3.set_ylabel('Stress [MPa]')
    ax3.set_title('Test 3: Conductor + Steel jacket')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#%% print

if __name__ == "__main__":
    print("##################################################### TF Model ##########################################################")

#%% Number of TF to satisfy ripple

def Number_TF_coils(R0, a, b, ripple_adm, L_min):
    """
    Find the minimum number of toroidal field (TF) coils required
    to keep the magnetic field ripple below a target value and satisfy
    a minimum toroidal access.

    Model (Wesson, 'Tokamaks', p.169):
        Ripple ≈ ((R0 - a - b)/(R0 + a))**N_TF + ((R0 + a)/(R0 + a + b + Delta))**N_TF
        L_access = 2 * pi * r2 / N_TF

    Parameters
    ----------
    R0 : float
        Major radius of the plasma [m]
    a : float
        Minor radius of the plasma [m]
    b : float
        Base radial distance between plasma edge and TF coil [m]
    ripple_adm : float
        Maximum admissible ripple (fraction, e.g. 0.01 for 1%)
    delta_max : float
        Maximum additional radial margin to scan [m]
    L_min : float
        Minimum toroidal access [m]

    Returns
    -------
    N_TF : int
        Minimum integer number of TF coils satisfying ripple <= ripple_adm
        and L_access >= L_min
    ripple_val : float
        Corresponding ripple value
    Delta : float
        Additional margin added to r2 to satisfy both constraints
    """

    N_min = 1
    N_max = 200
    delta_step = 0.01
    delta_max = 6

    if ripple_adm <= 0 or ripple_adm >= 1:
        raise ValueError("ripple_adm must be a fraction between 0 and 1.")
    if b <= 0:
        raise ValueError("b must be positive (coil must be outside the plasma).")
    if L_min <= 0:
        raise ValueError("L_min must be positive.")

    # Scan Delta from 0 to delta_max
    Delta = 0.0
    while Delta <= delta_max:
        r2 = R0 + a + b + Delta
        for N_TF in range(N_min, N_max + 1):
            ripple = ((R0 - a - b) / (R0 + a)) ** N_TF + ((R0 + a) / r2) ** N_TF
            L_access = 2 * math.pi * r2 / N_TF
            if ripple <= ripple_adm and L_access >= L_min:
                return N_TF, ripple, Delta
        Delta += delta_step

    raise ValueError(f"No N_TF and Delta combination found up to Delta_max={delta_max} m "
                     f"satisfying ripple ≤ {ripple_adm} and L_access ≥ {L_min} m")


if __name__ == "__main__":
    
    print("="*70)
    print("ITER prediction for the number of TF")
    print("Considering ripple and port minimal size")
    print("="*70)
    
    R0 = 6.2           # [m] major radius
    a = 2.0            # [m] minor radius
    b = 1.2            # [m] base radial distance
    ripple_adm = 0.01  # 1% ripple
    L_min = 3.5        # [m] minimum toroidal access

    N_TF, ripple, Delta = Number_TF_coils(R0, a, b, ripple_adm, L_min)
    r2 = R0 + a + b + Delta

    print(f"Minimum number of TF coils: {N_TF}")
    print(f"ITER TF coils: 18")
    print(f"Additional Delta = {Delta:.3f} m")
    print(f"Ripple = {ripple*100:.3f}%")
    print(f"L_access = {2*math.pi*r2/N_TF:.3f} m")
    
    print("="*70)

    
#%% Academic model

def f_TF_academic(a, b, R0, σ_TF, J_max_TF, B_max_TF, Choice_Buck_Wedg):
    """
    Calculate the thickness of the TF coil using a 2-layer thin cylinder model.

    Parameters:
    a : float
        Minor radius (m).
    b : float
        First Wall + Breeding Blanket + Neutron Shield + Gaps (m).
    R0 : float
        Major radius (m).
    σ_TF : float
        Yield strength of the TF steel (MPa).
    μ0 : float
        Magnetic permeability of free space.
    J_max_TF : float
        Maximum current density of the chosen Supra + Cu + He (A/m²).
    B_max_TF : float
        Maximum magnetic field (T).
    Choice_Buck_Wedg : str
        Mechanical option, either "Bucking" or "Wedging".

    Returns:
    c : float
        TF width (m).
    ratio_tension : float
        Ratio of axial to total stress.
    """
    
    def f_B0(Bmax, a, b, R0):
        """
        
        Estimate the magnetic field in the centre of the plasma
        
        Parameters
        ----------
        Bmax : The magnetic field at the inboard of the Toroidal Field (TF) coil [T]
        a : Minor radius [m]
        b : Thickness of the First Wall+ the Breeding Blanket+ The Neutron shield+ The Vacuum Vessel + Gaps [m]
        R0 : Major radius [m]

        Returns
        -------
        B0 : The estimated central magnetic field [T]
        
        """
        B0 = Bmax*(1-((a+b)/R0))
        return B0

    # 1. Calculate the central magnetic field B0 based on geometry and maximum field
    B0 = f_B0(B_max_TF, a, b, R0)

    # 2. Inner (inboard leg) and outer (outboard leg) radii
    R1_0 = R0 - a - b
    R2_0 = R0 + a + b

    # 3. Effective number of turns NI required to generate B0
    NI = 2 * np.pi * R0 * B0 / μ0

    # 4. Conductor cross-section required to provide the desired current
    S_cond = NI / J_max_TF

    # 5. Inner layer thickness c1 derived from the circular cross-section
    c_WP = R1_0 - np.sqrt(R1_0**2 - S_cond / np.pi)

    # 6. Calculate new radii after adding c1
    R1 = R1_0 - c_WP  # Effective inner radius
    R2 = R2_0 + c_WP  # Effective outer radius

    # 7. Calculate the tension T
    if (R2 > 0) and (R1 > 0) and (R2 / R1 > 0):
        # Tension calculation formula
        T = abs(((np.pi * B0 * 2 * R0**2) / μ0 * math.log(R2 / R1) - F_CClamp) * coef_inboard_tension)
    else:
        # Invalid geometric conditions
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 8. Radial pressure P due to the magnetic field B_max_TF
    P = B_max_TF**2 / (2 * μ0)

    # 9. Mechanical option choice: "bucking" or "wedging"
    if Choice_Buck_Wedg == "Bucking" or "Plug":
        # Thickness c2 for bucking, valid if R1 >> c
        c_Nose = (B0**2 * R0**2) * math.log(R2 / R1) / (2 * μ0 * 2 * R1 * (σ_TF - P))
        σ_r = P  # Radial stress
        σ_z = T / (2 * np.pi * R1 * c_Nose)  # Axial stress
        σ_theta = 0

    elif Choice_Buck_Wedg == "Wedging":
        # Thickness c2 for wedging, valid if R1 >> c
        c_Nose = (B0**2 * R0**2) / (2 * μ0 * R1 * σ_TF) * (1 + math.log(R2 / R1) / 2)
        σ_theta = P * R1 / c_Nose  # Circumferential stress
        σ_z = T / (2 * np.pi * R1 * c_Nose)  # Axial stress
        σ_r = 0
        
    else:
        raise ValueError("Choose 'Bucking' or 'Wedging' as the mechanical option.")

    # 10. Total thickness c (sum of the two layers)
    c = c_WP + c_Nose

    # Verify that c_WP is valid
    if c is None or np.isnan(c) or c < 0 or c > (c_WP + c_Nose) or c > R0 - a - b:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    Steel_fraction = c_Nose / c

    return c, c_WP, c_Nose, σ_z, σ_theta, σ_r, Steel_fraction

    
#%% D0FUS model

def Winding_Pack_D0FUS(R_0, a, b, sigma_max, J_max, B_max, omega, n):
    
    """
    Computes the winding pack thickness and stress ratio under Tresca criterion.
    
    Args:
        R_0: Reference radius [m]
        a, b: Geometric dimensions [m]
        sigma_max: Maximum allowable stress [Pa]
        J_max: Maximum engineering current density [A/m²]
        mu_0: Vacuum permeability [H/m]
        B_max: Peak magnetic field [T]
        omega: Scaling factor for axial load [dimensionless]
        n: Geometric factor [dimensionless]
        method: 'auto' for Brent, 'scan' for manual root search
    
    Returns:
        winding_pack_thickness: R_ext - R_sep [m]
        ratio_tension: σ_z / σ_Tresca
    """
    
    plot = False
    Choice_solving_TF_method = 'brentq'
    R_ext = R_0 - a - b

    if R_ext <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        # raise ValueError("R_ext must be positive. Check R_0, a, and b.")

    ln_term = np.log((R_0 + a + b) / (R_ext))
    if ln_term <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        # raise ValueError("Invalid logarithmic term: ensure R_0 + a + b > R_0 - a - b")

    def alpha(R_sep):
        denom = R_ext**2 - R_sep**2
        if denom <= 0:
            return np.nan
        val = (2 * B_max / (μ0 * J_max)) * (R_ext / denom)
        if np.iscomplex(val) or val < 0 or val > 1 :
            return np.nan
        return val

    def gamma(alpha_val, n_val):
        if alpha_val <= 0 or alpha_val >= 1:
            return np.nan
        A = 2 * np.pi + 4 * alpha_val * (n_val - 1)
        discriminant = A**2 - 4 * np.pi * (np.pi - 4 * alpha_val)
        if discriminant < 0:
            return np.nan
        val = (A - np.sqrt(discriminant)) / (2 * np.pi)
        if val < 0 or val > 1:
            return np.nan
        return val

    def tresca_residual(R_sep):
        a_val = alpha(R_sep)
        if np.isnan(a_val):
            return np.inf
        g_val = gamma(a_val, n)
        if np.isnan(g_val):
            return np.inf
        try:
            sigma_r = B_max**2 / (2 * μ0 * g_val)
            denom_z = R_ext**2 - R_sep**2
            if denom_z <= 0:
                return np.inf
            sigma_z = (omega / (1 - a_val)) * B_max**2 * R_ext**2 / (2 * μ0 * denom_z) * ln_term
            val = sigma_r + sigma_z - sigma_max
            return np.sign(val) * np.log1p(abs(val))
        except Exception:
            return np.inf

    # === Root search ===
    R_sep_solution = None
    residuals = []
    R_vals = np.linspace(0.001, R_ext * 0.999, 10000)

    if Choice_solving_TF_method == 'manual':
        residuals = [tresca_residual(R) for R in R_vals]
        for i in range(len(R_vals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                R_sep_solution = R_vals[i + 1]
                break
        if R_sep_solution is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    elif Choice_solving_TF_method == 'brentq':
        found = False
        R1 = R_ext * 0.999
        R_min = 0.001
        while R1 > R_min :
            R2 = R1 - 0.001
            try:
                if tresca_residual(R1) * tresca_residual(R2) < 0:
                    R_sep_solution = brentq(tresca_residual, R2, R1)
                    found = True
                    break
            except ValueError:
                pass
            R1 = R2
        if not found:
            return np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        raise ValueError("Invalid method Use 'manual' or 'brentq'")

    # === Final stress calculation ===
    a_val = alpha(R_sep_solution)
    g_val = gamma(a_val, n)

    if np.isnan(a_val) or np.isnan(g_val):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        sigma_r = B_max**2 / (2 * μ0 * g_val)
        sigma_z = (omega / (1 - a_val)) * B_max**2 * R_ext**2 / (2 * μ0 * (R_ext**2 - R_sep_solution**2)) * ln_term
        sigma_theta = 0
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    winding_pack_thickness = R_ext - R_sep_solution

    # === Optional plot ===
    if plot:
        residuals = [tresca_residual(R) for R in R_vals]
        residuals_abs = [np.abs(r) for r in residuals]
        plt.figure(figsize=(8, 5))
        plt.plot(R_vals, residuals_abs, label="|Tresca Residual|")
        if R_sep_solution:
            plt.axvline(R_sep_solution, color='red', linestyle='--', label=f"Solution: R_sep = {R_sep_solution:.4f}")
        plt.yscale('log')
        plt.xlabel("R_sep [m]")
        plt.ylabel("|σ_r + σ_z − σ_max| [Pa]")
        plt.title("Tresca Residual vs R_sep (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    Steel_fraction = (1-a_val)

    return winding_pack_thickness, sigma_r, sigma_z, sigma_theta, Steel_fraction


def Nose_D0FUS(R_ext_Nose, sigma_max, omega, B_max, R_0, a, b):
    """
    Compute the internal radius Ri based on analytical expressions.

    Parameters:
    - R_ext_Nose : float, external radius at the nose (R_ext^Nose)
    - sigma_max  : float, maximum admissible stress
    - beta       : float, dimensionless coefficient (0 ≤ beta ≤ 1)
    - B_max      : float, maximum magnetic field
    - mu_0       : float, magnetic permeability of vacuum
    - R_0, a, b  : floats, geometric parameters

    Returns:
    - Ri : float, internal radius
    """
    
    # Compute P_Nose
    P = (B_max**2) / (2 * μ0) * (R_0 - a - b) / R_ext_Nose
    
    # Compute the logarithmic term
    log_term = np.log((R_0 + a + b) / (R_0 - a - b))
    
    # Compute the full expression under the square root
    term_intermediate = (R_ext_Nose**2 / sigma_max) * (2 * P + (1 - omega) * (B_max**2 * coef_inboard_tension / μ0) * log_term)
    term = R_ext_Nose**2 - term_intermediate
    if term < 0:
        # raise ValueError("Negative value under square root. Check your input parameters.")
        return(np.nan)
    
    Ri = np.sqrt(term)
    
    return Ri

def f_TF_D0FUS(a, b, R0, σ_TF , J_max_TF, B_max_TF, Choice_Buck_Wedg, omega, n):
    
    """
    Calculate the thickness of the TF coil using a 2 layer thick cylinder model 

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A/m²)
    B_max_TF : Maximum magnetic field (T)

    Returns:
    c : TF width
    
    """
    
    debuging = 'Off'
    
    if Choice_Buck_Wedg == "Wedging":
        
        c_WP, σ_r, σ_z, σ_theta, Steel_fraction  = Winding_Pack_D0FUS( R0, a, b, σ_TF, J_max_TF, B_max_TF, omega, n)
        
        # Vérification que c_WP est valide
        if c_WP is None or np.isnan(c_WP) or c_WP < 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        c_Nose = R0 - a - b - c_WP - Nose_D0FUS(R0 - a - b - c_WP, σ_TF, omega, B_max_TF, R0, a, b)

        # Vérification que c_Nose est valide
        if c_Nose is None or np.isnan(c_Nose) or c_Nose < 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Vérification que la somme ne dépasse pas R0 - a - b
        if (c_WP + c_Nose) > (R0 - a - b):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Epaisseur totale de la bobine
        c  = c_WP + c_Nose + c_BP
        
        if __name__ == "__main__" and debuging == 'On':
            
            print(f'Winding pack width : {c_WP}')
            print(f'Nose width : {c_Nose}')
            print(f'Backplate width : {c_BP}')
    
    elif Choice_Buck_Wedg == "Bucking" or Choice_Buck_Wedg == "Plug":
        
        c_WP, σ_r, σ_z, σ_theta, Steel_fraction = Winding_Pack_D0FUS(R0, a, b, σ_TF, J_max_TF, B_max_TF, omega, n)
        
        c = c_WP
        c_Nose = 0
        
        # Vérification que c est valide
        if c is None or np.isnan(c) or c < 0 or c > R0 - a - b :
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    else : 
        print( "Choose a valid mechanical configuration" )
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return c, c_WP, c_Nose, σ_z, σ_theta, σ_r, Steel_fraction

#%% TF benchmark

if __name__ == "__main__":
    
    def get_machine_parameters_TF(machine_name):

        machines = {
            "SF": { "a_TF": 1.6, "b_TF": 1.2, "R0_TF": 6, "σ_TF_tf": 660e6, "T_supra": 4.2, "B_max_TF_TF": 20,
                   "n_TF": 1, "supra_TF" : "Rebco", "config" : "Wedging", "J" : 600e6},
            # Source : PEPR SupraFusion

            "ITER": { "a_TF": 2, "b_TF": 1.23, "R0_TF": 6.2, "σ_TF_tf": 660e6, "T_supra": 4.2, "B_max_TF_TF": 11.8,
                     "n_TF": 1, "supra_TF" : "Nb3Sn", "config" : "Wedging", "J" : 140e6},
            # Source : Sborchia, C., Fu, Y., Gallix, R., Jong, C., Knaster, J., & Mitchell, N. (2008). Design and specifications of the ITER TF coils. IEEE transactions on applied superconductivity, 18(2), 463-466.
            
            "DEMO": { "a_TF": 2.92, "b_TF": 1.9, "R0_TF": 9.07, "σ_TF_tf": 660e6, "T_supra": 4.2, "B_max_TF_TF": 13,
                     "n_TF": 1/2, "supra_TF" : "Nb3Sn", "config" : "Wedging", "J" : 150e6},
            # Source : Federici, G., Siccinio, M., Bachmann, C., Giannini, L., Luongo, C., & Lungaroni, M. (2024). Relationship between magnetic field and tokamak size—a system engineering perspective and implications to fusion development. Nuclear Fusion, 64(3), 036025.
            
            "CFETR": { "a_TF": 2.2, "b_TF": 1.52, "R0_TF": 7.2, "σ_TF_tf": 1000e6, "T_supra": 4.2, "B_max_TF_TF": 14,
                      "n_TF": 1, "supra_TF" : "Nb3Sn", "config" : "Wedging", "J" : 150e6},
            # Source : Wu, Y., Li, J., Shen, G., Zheng, J., Liu, X., Long, F., ... & Han, H. (2021). Preliminary design of CFETR TF prototype coil. Journal of Fusion Energy, 40, 1-14.
            
            "EAST": { "a_TF": 0.45, "b_TF": 0.45, "R0_TF": 1.85, "σ_TF_tf": 660e6, "T_supra": 4.2,
                     "B_max_TF_TF": 7.2, "n_TF": 1, "supra_TF" : "NbTi", "config" : "Wedging", "J" : 200e6},
            # Source : Chen, S. L., Villone, F., Xiao, B. J., Barbato, L., Luo, Z. P., Liu, L., ... & Xing, Z. (2016). 3D passive stabilization of n= 0 MHD modes in EAST tokamak. Scientific Reports, 6(1), 32440.
            # Source : Yi, S., Wu, Y., Liu, B., Long, F., & Hao, Q. W. (2014). Thermal analysis of toroidal field coil in EAST at 3.7 K. Fusion Engineering and Design, 89(4), 329-334.
            # Source : Chen, W., Pan, Y., Wu, S., Weng, P., Gao, D., Wei, J., ... & Chen, S. (2006). Fabrication of the toroidal field superconducting coils for the EAST device. IEEE transactions on applied superconductivity, 16(2), 902-905.
            
            "K-STAR": { "a_TF": 0.5, "b_TF": 0.35, "R0_TF": 1.8, "σ_TF_tf": 660e6, "T_supra": 4.2 , "B_max_TF_TF": 7.2,
                       "n_TF": 1, "supra_TF" : "Nb3Sn", "config" : "Wedging", "J" : 200e6},
            # Source :
            # Oh, Y. K., Choi, C. H., Sa, J. W., Ahn, H. J., Cho, K. J., Park, Y. M., ... & Lee, G. S. (2002, January). Design overview of the KSTAR magnet structures. In Proceedings of the 19th IEEE/IPSS Symposium on Fusion Engineering. 19th SOFE (Cat. No. 02CH37231) (pp. 400-403). IEEE.
            # Choi, C. H., Sa, J. W., Park, H. K., Hong, K. H., Shin, H., Kim, H. T., ... & Hong, C. D. (2005, January). Fabrication of the KSTAR toroidal field coil structure. In 20th IAEA fusion energy conference 2004. Conference proceedings (No. IAEA-CSP--25/CD, pp. 6-6).
            # Oh, Y. K., Choi, C. H., Sa, J. W., Lee, D. K., You, K. I., Jhang, H. G., ... & Lee, G. S. (2002). KSTAR magnet structure design. IEEE transactions on applied superconductivity, 11(1), 2066-2069.
    
            "ARC": { "a_TF": 1.07, "b_TF": 0.89, "R0_TF": 3.3, "σ_TF_tf": 1000e6, "T_supra": 20, "supra_TF" : "Rebco",
                    "B_max_TF_TF": 23, "n_TF": 1, "config" : "Plug", "J" : 600e6},
            # Source :
            # Hartwig, Z. S., Vieira, R. F., Sorbom, B. N., Badcock, R. A., Bajko, M., Beck, W. K., ... & Zhou, L. (2020). VIPER: an industrially scalable high-current high-temperature superconductor cable. Superconductor Science and Technology, 33(11), 11LT01.
            # Kuznetsov, S., Ames, N., Adams, J., Radovinsky, A., & Salazar, E. (2024). Analysis of Strains in SPARC CS PIT-VIPER Cables. IEEE Transactions on Applied Superconductivity.
            # Sanabria, C., Radovinsky, A., Craighill, C., Uppalapati, K., Warner, A., Colque, J., ... & Brunner, D. (2024). Development of a high current density, high temperature superconducting cable for pulsed magnets. Superconductor Science and Technology, 37(11), 115010.
            # Sorbom, B. N., Ball, J., Palmer, T. R., Mangiarotti, F. J., Sierchio, J. M., Bonoli, P., ... & Whyte, D. G. (2015). ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets. Fusion Engineering and Design, 100, 378-405.
            
            "SPARC": { "a_TF": 0.57, "b_TF": 0.18, "R0_TF": 1.85, "σ_TF_tf": 1000e6, "T_supra": 20,     "B_max_TF_TF": 20,
                      "n_TF": 1, "supra_TF" : "Rebco", "config" : "Bucking" , "J" : 600e6},
            # Source : Creely, A. J., Greenwald, M. J., Ballinger, S. B., Brunner, D., Canik, J., Doody, J., ... & Sparc Team. (2020). Overview of the SPARC tokamak. Journal of Plasma Physics, 86(5), 865860502.
            
            "JT60-SA": { "a_TF": 1.18, "b_TF": 0.27, "R0_TF": 2.96, "σ_TF_tf": 660e6, "T_supra": 4.2,
                        "B_max_TF_TF": 5.65, "n_TF": 1, "supra_TF" : "NbTi", "config" : "Wedging", "J" : 150e6/1.95}
            # Source :
            # 150 / 1.95 to take into account the ratio of Cu to Supra = 1.95 and not 1, see table 4 of
            # Obana, Tetsuhiro, et al. "Conductor and joint test results of JT-60SA CS and EF coils using the NIFS test facility." Cryogenics 73 (2016): 25-41.
            # Koide, Y., Yoshida, K., Wanner, M., Barabaschi, P., Cucchiaro, A., Davis, S., ... & Zani, L. (2015). JT-60SA superconducting magnet system. Nuclear Fusion, 55(8), 086001.
            # Polli, G. M., Cucchiaro, A., Cocilovo, V., Corato, V., Rossi, P., Drago, G., ... & Tomarchio, V. (2019). JT-60SA toroidal field coils procured by ENEA: A final review. Fusion Engineering and Design, 146, 2489-2493.
        }
    
        return machines.get(machine_name, None)
    
    # === BENCHMARK ===

    # === Machines to test ===
    machines = ["ITER", "DEMO", "JT60-SA", "EAST", "ARC", "SPARC"]
    
    # === Helper function for clean results ===
    def clean_result(val):
        """Return a clean float or NaN if invalid/complex."""
        # Handle tuples (extract first element)
        if isinstance(val, tuple):
            val = val[0]
        # Handle None, NaN, or complex
        if val is None or np.isnan(val) or np.iscomplex(val):
            return np.nan
        # Return rounded float
        return round(float(np.real(val)), 2)
    
    # === Accumulate results ===
    table = []
    
    for machine in machines:
        # Get machine parameters
        params = get_machine_parameters_TF(machine)
        if params is None:
            continue
        
        # Unpack input parameters
        a = params["a_TF"]
        b = params["b_TF"]
        R0 = params["R0_TF"]
        σ = params["σ_TF_tf"]
        T_supra = params["T_supra"]
        B_max_TF = params["B_max_TF_TF"]
        n = params["n_TF"]
        Supra_choice_TF = params["supra_TF"]
        config = params["config"]
        Jc_strand = params["J"]
        f_Cu_Non_Cu_benchmark = 0.5
        f_Cu_Strand_benchmark = 0.75
        f_Cool_benchmark = 0.7
        Jmax = Jc_strand * f_Cu_Non_Cu_benchmark * f_Cu_Strand_benchmark * f_Cool_benchmark
        
        # === Run D0FUS model for machine-specific configuration ===
        if config == "Wedging":
            thickness = f_TF_D0FUS(a, b, R0, σ, Jmax, B_max_TF, "Wedging", 0.5, n)
        else:  # Bucking
            thickness = f_TF_D0FUS(a, b, R0, σ, Jmax, B_max_TF, "Bucking", 1.0, n)
        
        # Store results
        table.append({
            "Machine": machine,
            "Config": config,
            "J [MA/m²]": clean_result(Jc_strand / 1e6),
            "σ [MPa]" : σ/1e6,
            "Thickness [m]": clean_result(thickness),
        })
    
    # === Display Table ===
    df = pd.DataFrame(table)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(df.columns)):
        tbl[(0, i)].set_facecolor('#4CAF50')
        tbl[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors for better readability
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                tbl[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title("TF Thickness Results", 
              fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    plt.show()

#%% TF plot
    
if __name__ == "__main__":

    # EU DEMO
    a_TF = 3
    b_TF = 1.7
    R0_TF = 9
    # Default values
    σ_TF_tf = 860e6
    n_TF = 1

    # === B_max_TF_TF RANGE DEFINITION ===
    B_max_TF_values = np.linspace(0, 25, 50)  # Magnetic field range (0 T to 25 T)

    # === INITIALIZE RESULT LISTS ===
    academic_w = []
    academic_b = []
    d0fus_w = []
    d0fus_b = []

    # === COMPUTATION LOOP ===
    for B_max_TF_TF in B_max_TF_values:
        
        T_supra = 20
        f_Cu_Strand_benchmark = 0.75
        f_Cu_Non_Cu_benchmark = 0.5
        f_Cool_benchmark = 0.7
        J_max_TF_tf = Jc("Rebco", B_max_TF_TF, T_supra, Jc_Manual) * f_Cu_Non_Cu_benchmark * f_Cu_Strand_benchmark * f_Cool_benchmark
        
        # Academic models
        res_acad_w = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, B_max_TF_TF, "Wedging")
        res_acad_b = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, B_max_TF_TF, "Bucking")

        # D0FUS models (γ = 0.5 for Wedging, γ = 1 for Bucking)
        res_d0fus_w = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, B_max_TF_TF, "Wedging", 0.5, n_TF)
        res_d0fus_b = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, B_max_TF_TF, "Bucking", 1, n_TF)

        # Store results (only first return value assumed to be thickness)
        academic_w.append(res_acad_w[0])
        academic_b.append(res_acad_b[0])
        d0fus_w.append(res_d0fus_w[0])
        d0fus_b.append(res_d0fus_b[0])
    
    # Couleurs par modèle : bleu, vert, rouge
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # MADE fit
    
    x = np.array([11.25, 13.25, 14.75, 16, 17, 18, 19, 19.75,20.5, 21.25, 22, 22.5])
    y = np.array([0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.65, 2.85])
    
    # --- FIGURE 1 : Wedging ---
    plt.figure(figsize=(5, 5))
    
    plt.plot(B_max_TF_values, academic_w, color=colors[0], linestyle='-', linewidth=2,
             label='Academic Wedging')
    plt.plot(B_max_TF_values, d0fus_w, color=colors[1], linestyle='-', linewidth=2,
             label='D0FUS Wedging')
    plt.scatter(x, y, color="black", marker = 'x', s=80, label="MADE")
    
    plt.xlabel('TF magnetic field (T)', fontsize=14)
    plt.ylabel('TF thickness [m]', fontsize=14)
    plt.title('Mechanical models comparison: Wedging', fontsize=16)
    plt.legend(fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # --- FIGURE 2 : Bucking ---
    plt.figure(figsize=(5, 5))
    
    plt.plot(B_max_TF_values, academic_b, color=colors[0], linestyle='-', linewidth=2,
             label='ACADEMIC Bucking')
    plt.plot(B_max_TF_values, d0fus_b, color=colors[1], linestyle='-', linewidth=2,
             label='D0FUS Bucking')
    
    plt.xlabel('TF magnetic field (T)', fontsize=14)
    plt.ylabel('TF thickness [m]', fontsize=14)
    plt.title('Mechanical models comparison: Bucking', fontsize=16)
    plt.legend(fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

#%% Print
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% Magnetic flux calculation

def Magnetic_flux(Ip, I_Ohm, B_max_TF, a, b, c, R0, κ, nbar, Tbar, Ce, Temps_Plateau, Li, Choice_Buck_Wedg):
    """
    Calculate the magnetic flux components for a tokamak plasma.

    Parameters:
    Ip : float
        Plasma current (MA).
    I_Ohm : float
        Ohmic current (MA).
    B_max_TF : float
        Maximum magnetic field (T).
    a : float
        Minor radius (m).
    b : float
        First Wall + Breeding Blanket + Neutron Shield + Gaps (m).
    c : float
        TF coil width (m).
    R0 : float
        Major radius (m).
    κ : float
        Elongation.
    nbar : float
        Mean electron density (1e20 particles/m³).
    Tbar : float
        Mean temperature (keV).
    Ce : float
        Ejima constant
    Temps_Plateau : float
        Plateau time (s).
    Li : float
        Internal inductance.

    Returns:
    ΨPI : float
        Flux needed for plasma initiation (Wb).
    ΨRampUp : float
        Total ramp-up flux (Wb).
    Ψplateau : float
        Flux related to the plateau (Wb).
    ΨPF : float
        Available flux from the PF system (Wb).
    """

    # Convert currents from MA to A
    Ip = Ip * 1e6
    I_Ohm = I_Ohm * 1e6
    
    def f_B0(Bmax, a, b, R0):
        """
        
        Estimate the magnetic field in the centre of the plasma
        
        Parameters
        ----------
        Bmax : The magnetic field at the inboard of the Toroidal Field (TF) coil [T]
        a : Minor radius [m]
        b : Thickness of the First Wall+ the Breeding Blanket+ The Neutron shield+ The Vacuum Vessel + Gaps [m]
        R0 : Major radius [m]

        Returns
        -------
        B0 : The estimated central magnetic field [T]
        
        """
        B0 = Bmax*(1-((a+b)/R0))
        return B0

    # Toroidal magnetic field
    B0 = f_B0(B_max_TF, a, b, R0)

    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))

    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS
    if Choice_Buck_Wedg == 'Bucking' or Choice_Buck_Wedg == 'Plug':
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        print("Choose between Wedging and Bucking")
        return (np.nan, np.nan, np.nan)

    #----------------------------------------------------------------------------------------------------------------------------------------
    #### Flux calculation ####

    # Flux needed for plasma initiation (ΨPI)
    ΨPI = ITERPI  # Initiation consumption from ITER 20 [Wb]

    # Flux needed for the inductive part (Ψind)
    if (8 * R0 / (a * math.sqrt(κ))) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        Lp = 1.07 * μ0 * R0 * (1 + 0.1 * βp) * (Li / 2 - 2 + math.log(8 * R0 / (a * math.sqrt(κ))))
        Ψind = Lp * Ip

    # Flux related to the resistive part (Ψres)
    Ψres = Ce * μ0 * R0 * Ip

    # Total ramp-up flux (ΨRampUp)
    ΨRampUp = Ψind + Ψres

    # Plateau flux calculation
    eta = 2.8e-8 / (Tbar**1.5) # Spitzer resistivity from Wesson
    R_eff = eta * (2 * R0) / (a**2 * κ) # Resistivity with A = pi a^2 kappa
    Vloop = R_eff * I_Ohm # Loop voltage
    Ψplateau = Vloop * Temps_Plateau  # Plateau flux
    
    # Available flux from PF system (ΨPF)
    if (8 * a / κ**(1 / 2)) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        ΨPF = μ0 * Ip / (4 * R0) * (βp + (Li - 3) / 2 + math.log(8 * a / κ**(1 / 2))) * (R0**2) # (R0**2 - RCS_ext**2) ?

    # Theoretical expression of CS flux
    # ΨCS = 2 * (math.pi * μ0 * J_max_CS * Alpha) / 3 * (RCS_ext**3 - RCS_int**3)

    return (ΨPI, ΨRampUp, Ψplateau, ΨPF)

#%% Flux generation test

if __name__ == "__main__":
    
    # ITER
    a_cs = 2
    b_cs = 1.25
    c_cs = 0.90
    R0_cs = 6.2
    B_max_TF_cs = 13
    configuration = "Wedging"
    T_CS = 4.2
    κ_cs = 1.7                    # Elongation
    Tbar_cs = 8                   # Average temperature [keV]
    nbar_cs = 1                   # Average density [10^20 m^-3]
    Ip_cs = 15                    # Plasma current [MA]
    I_Ohm_cs = 3                  # Ohmic current wanted [MA]
    Ce_cs = 0.45                  # Efficiency coefficient
    Temps_Plateau_cs = 10 * 60    # Plateau duration
    Li_cs = 0.8                   # Internal inductance
    p_bar = 0.2
    
    # Compute total magnetic flux contributions using provided function
    ΨPI, ΨRampUp, Ψplateau, ΨPF = Magnetic_flux(
        Ip_cs, I_Ohm_cs, B_max_TF_cs,
        a_cs, b_cs, c_cs, R0_cs,
        κ_cs, nbar_cs, Tbar_cs,
        Ce_cs, Temps_Plateau_cs, Li_cs, configuration 
    )
    
    # Print benchmark results in a clear format
    print("\n=== Magnetic Flux Benchmark ===")
    print(f"Machine                      ITER  D0FUS")
    print(f"Required Ψ initiation phase : 20  : {ΨPI:.2f} Wb")
    print(f"Required Ψ ramp-up phase    : 200 : {ΨRampUp:.2f} Wb")
    print(f"Required Ψ flat-top phase   : 36  : {Ψplateau:.2f} Wb")
    print(f"Provided Ψ PF               : 115  : {ΨPF:.2f} Wb")
    print(f"Needed Ψ CS                 : 137 : {ΨPI+ΨRampUp+Ψplateau-ΨPF:.2f} Wb")

#%% CS academic model

def f_CS_ACAD(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, f_Cu_Non_Cu , f_Cu_Strand , f_Cool , f_In, Choice_Buck_Wedg):
    """
    Calculate the Central Solenoid (CS) thickness using thin-layer approximation 
    and a 2-cylinder model (superconductor + steel structure).
    
    Parameters
    ----------
    ΨPI : float
        Plasma initiation flux (Wb)
    ΨRampUp : float
        Current ramp-up flux swing (Wb)
    Ψplateau : float
        Flat-top operational flux (Wb)
    ΨPF : float
        Poloidal field coil contribution (Wb)
    a : float
        Plasma minor radius (m)
    b : float
        Cumulative radial build: 1st wall + breeding blanket + neutron shield + gaps (m)
    c : float
        Toroidal field (TF) coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T) [currently unused in implementation]
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier for Jc calculation
    Jc_manual : float
        If needed, manual current density
    T_Helium : float
        Helium coolant temperature (K)
    f_Cu_non_Cu : float
        Copper fraction in the strands
    f_Cu : float
        Copper strands fraction in conductor
    f_Cool : float
        Cooling fraction
    Choice_Buck_Wedg : str
        Mechanical support configuration: 'Bucking', 'Wedging', or 'Plug'
    
    Returns
    -------
    tuple of float
        (d, alpha, B_CS, J_max_CS)
        - d : CS radial thickness (m)
        - alpha : Conductor volume fraction (dimensionless, 0 < alpha < 1)
        - B_CS : CS magnetic field (T)
        - J_max_CS : Maximum current density in conductor (A/m²)
        
        Returns (nan, nan, nan, nan) if no physically valid solution exists.
    
    Notes
    -----
    - Uses thin-wall cylinder approximation for initial B_CS estimation
    - B_CS, J_max_CS, and d_SU determined analytically (no iteration)
    - Only d_SS requires iterative solving via brentq
    - Mechanical models vary by support configuration:
        * Bucking: CS bears TF support loads, two limiting cases evaluated
        * Wedging: CS isolated from TF by gap, pure hoop stress
        * Plug: Central plug supports TF, combined pressure loading

    References
    ----------
    [Auclair et al. NF 2026]
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    
    # Debug option
    debug = False
    Tol_CS = 1e-3
    Gap = 0.05  # Wedging gap (m) - adjust if needed

    # --- Compute external CS radius depending on mechanical choice ---
    if Choice_Buck_Wedg in ('Bucking', 'Plug'):
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        if debug:
            print("[f_CS_ACAD] Invalid mechanical choice:", Choice_Buck_Wedg)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if RCS_ext <= 0.0:
        if debug:
            print("[f_CS_ACAD] Non-positive RCS_ext:", RCS_ext)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Total flux for CS
    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF
    
    # ------------------------------------------------------------------
    # STEP 1: Determine B_CS, J_max_CS, and d_SU analytically
    # ------------------------------------------------------------------
    
    # Thin cylinder approximation for initial estimate of B_CS
    # Φ = π * R² * B  →  B = Φ / (π * R²)
    B_CS_thin = ΨCS / (np.pi * RCS_ext**2)
    
    if debug:
        print(f"[STEP 1] Thin cylinder B_CS estimate: {B_CS_thin:.2f} T")
    
    # Compute maximum current density at this field
    # J depends on B through superconductor critical current properties
    if Supra_choice_CS == 'Manual':
        J_max_CS = Jc_manual * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
    elif Supra_choice_CS == 'Rebco':
        J_max_CS = Jc(Supra_choice_CS, B_CS_thin, T_Helium, Jc_Manual) * f_Cu_Strand * f_Cool * f_In
    else :
        J_max_CS = Jc(Supra_choice_CS, B_CS_thin, T_Helium, Jc_Manual) * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        
    if J_max_CS < Tol_CS:
        if debug:
            print(f"[STEP 1] Non-positive J_max_CS: {J_max_CS:.2e} A/m²")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    if debug:
        print(f"[STEP 1] J_max_CS at B={B_CS_thin:.2f} T: {J_max_CS:.2e} A/m²")
    
    """
    Compute separatrix radius (steel/superconductor interface)
    From thick solenoid flux formula and Ampere's law:
    Φ = (2π/3) * B * (R_ext³ + R_ext*R_sep² + R_sep³) / (R_ext + R_sep)
    With B = μ₀ * J * d_SU and some algebra:
    R_sep³ = R_ext³ - (3*Φ) / (2π * μ₀ * J)
    """
    
    RCS_sep_cubed = RCS_ext**3 - (3 * ΨCS) / (2 * np.pi * μ0 * J_max_CS)
    
    if RCS_sep_cubed <= 0:
        if debug:
            print(f"[STEP 1] Invalid RCS_sep³: {RCS_sep_cubed:.2e} (negative or zero)")
            print(f"  This means J_max_CS is too low to generate required flux")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    RCS_sep = RCS_sep_cubed**(1/3)
    
    # Superconductor thickness
    d_SU = RCS_ext - RCS_sep
    
    if d_SU <= Tol_CS or d_SU >= RCS_ext - Tol_CS:
        if debug:
            print(f"[STEP 1] Invalid d_SU: {d_SU:.4f} m (out of physical bounds)")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Recompute B_CS with thick solenoid formula for accuracy
    B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_sep + RCS_sep**2))
    
    if B_CS > B_max_CS:
        if debug:
            print(f"[STEP 1] Too high B_CS")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    if debug:
        print(f"[STEP 1] Superconductor sizing complete:")
        print(f"  d_SU = {d_SU:.4f} m")
        print(f"  RCS_sep = {RCS_sep:.4f} m")
        print(f"  B_CS (refined) = {B_CS:.2f} T")
    
    # ------------------------------------------------------------------
    # STEP 2: Find steel thickness d_SS using brentq
    # ------------------------------------------------------------------
    
    # Magnetic pressures (constant, computed once)
    P_CS = B_CS**2 / (2.0 * μ0)
    P_TF = B_max_TF**2 / (2.0 * μ0)
    
    # If plug model:
    if Choice_Buck_Wedg == 'Plug' and abs(P_CS - P_TF) <= abs(P_TF):
        # No steel, conductor directly compress the central plug
        d = d_SU
        alpha = 1
        σ_z = 0
        σ_theta = 0
        σ_r = 0
        Steel_fraction = 1 - alpha
        return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS
    
    def stress_residual(d_SS):
        """
        Residual function: Sigma_CS(d_SS) - σ_CS
        
        Parameters
        ----------
        d_SS : float
            Steel layer thickness [m]
        
        Returns
        -------
        float
            Stress residual [Pa], or nan if unphysical
        """
        
        # Geometric constraint check
        if d_SS <= Tol_CS:
            return np.nan
        
        d_total = d_SS + d_SU
        if d_total >= RCS_ext - Tol_CS:
            return np.nan
        
        # Compute stress in steel based on mechanical configuration:
        if Choice_Buck_Wedg == 'Bucking':
            # CS bears TF support loads, two limiting cases evaluated
            # σ_hoop = P * R / t, taking maximum of two loading scenarios
            Sigma_CS = abs(np.nanmax([P_TF, abs(P_CS - P_TF)])) * RCS_sep / d_SS
            
        elif Choice_Buck_Wedg == 'Wedging':
            # CS isolated from TF by gap, pure hoop stress from CS pressure
            Sigma_CS = abs(P_CS * RCS_sep / d_SS)
            
        elif Choice_Buck_Wedg == 'Plug':
            # If the CS pressure is dominant (if not, filtered before)
            if abs(P_CS - P_TF) > abs(P_TF):
                # classical bucking case
                Sigma_CS = abs(abs(P_CS - P_TF) * RCS_sep / d_SS)
            else:
                return np.nan
        else:
            return np.nan
        
        # Sanity check
        if Sigma_CS < Tol_CS:
            return np.nan
        
        # Return residual: we want Sigma_CS = σ_CS
        return Sigma_CS - σ_CS
    
    # Search for sign changes in stress residual
    d_SS_min = Tol_CS
    d_SS_max = RCS_ext - d_SU - Tol_CS
    
    # Sample the residual function to find sign changes
    d_SS_vals = np.linspace(d_SS_min, d_SS_max, 200)
    sign_changes = []
    
    for i in range(1, len(d_SS_vals)):
        y1 = stress_residual(d_SS_vals[i-1])
        y2 = stress_residual(d_SS_vals[i])
        
        # Check for sign change (both values must be finite)
        if np.isfinite(y1) and np.isfinite(y2) and y1 * y2 < 0:
            sign_changes.append((d_SS_vals[i-1], d_SS_vals[i]))
    
    if len(sign_changes) == 0:
        if debug:
            print("[STEP 2] No sign changes found in stress_residual")
            print(f"  d_SS range: [{d_SS_min:.4f}, {d_SS_max:.4f}] m")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Refine each sign change interval with brentq
    valid_solutions = []
    
    for interval in sign_changes:
        try:
            d_SS_sol = brentq(stress_residual, interval[0], interval[1], xtol=1e-9)
            
            # Verify the solution satisfies physical constraints
            Sigma_check = stress_residual(d_SS_sol) + σ_CS
            
            if Sigma_check > 0:
                valid_solutions.append(d_SS_sol)
                
                if debug:
                    print(f"[STEP 2] Valid d_SS found: {d_SS_sol:.4f} m")
                    print(f"  Sigma_CS = {Sigma_check/1e6:.1f} MPa")
                    
        except ValueError:
            continue
    
    if len(valid_solutions) == 0:
        if debug:
            print("[STEP 2] No valid d_SS solution found after refinement")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Select smallest steel thickness (most compact design)
    d_SS = min(valid_solutions)
    
    # ------------------------------------------------------------------
    # STEP 3: Compute final outputs
    # ------------------------------------------------------------------
    
    d_total = d_SS + d_SU
    alpha = d_SU / d_total
    # Compute stress in steel based on mechanical configuration:
    if Choice_Buck_Wedg == 'Bucking':
        σ_theta = abs(np.nanmax([P_TF, abs(P_CS - P_TF)])) * RCS_sep / d_SS
        σ_r = 0
    elif Choice_Buck_Wedg == 'Wedging':
        σ_theta = abs(P_CS * RCS_sep / d_SS)
        σ_r = 0
    elif Choice_Buck_Wedg == 'Plug':
        if abs(P_CS - P_TF) > abs(P_TF):
            σ_r = abs(abs(P_CS - P_TF) * RCS_sep / d_SS)
            σ_theta = 0
    
    # Final sanity checks
    if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
        if debug:
            print(f"[FINAL] Invalid alpha: {alpha:.4f}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    if debug:
        print(f"\n[FINAL SOLUTION]")
        print(f"  d_total = {d_total:.4f} m")
        print(f"  d_SS = {d_SS:.4f} m ({d_SS/d_total*100:.1f}%)")
        print(f"  d_SU = {d_SU:.4f} m ({d_SU/d_total*100:.1f}%)")
        print(f"  alpha = {alpha:.4f}")
        print(f"  B_CS = {B_CS:.2f} T")
        print(f"  J_max_CS = {J_max_CS:.2e} A/m²")
        
        # Verify flux
        flux_check = 2 * np.pi * B_CS * (RCS_ext**2 + RCS_ext * RCS_sep + RCS_sep**2) / 3
        print(f"  Flux check: {flux_check:.2f} Wb (target: {ΨCS:.2f} Wb)")
        print(f"  Flux error: {abs(flux_check - ΨCS)/ΨCS * 100:.2f}%")
        
    
    d = d_total
    Steel_fraction = 1 - alpha
    σ_z = 0
    
    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

    
#%% CS D0FUS model

def f_CS_D0FUS( ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
    Supra_choice_CS, Jc_manual, T_Helium, f_Cu_Non_Cu , f_Cu_Strand , f_Cool , f_In, Choice_Buck_Wedg):
    
    """
    Calculate the Central Solenoid (CS) thickness using thick-layer approximation
    
    The function solves for CS geometry by balancing electromagnetic stresses 
    with structural limits, accounting for different mechanical configurations.
    
    Parameters
    ----------
    ΨPI : float
        Plasma initiation flux (Wb)
    ΨRampUp : float
        Current ramp-up flux swing (Wb)
    Ψplateau : float
        Flat-top operational flux (Wb)
    ΨPF : float
        Poloidal field coil contribution (Wb)
    a : float
        Plasma minor radius (m)
    b : float
        Cumulative radial build: 1st wall + breeding blanket + neutron shield + gaps (m)
    c : float
        Toroidal field (TF) coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T) [currently unused in implementation]
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier for Jc calculation
    Jc_manual : float
        If needed, manual current density
    T_Helium : float
        Helium coolant temperature (K)
    f_Cu_Non_Cu : float
        Copper fraction in the strands
    f_Cu : float
        Copper strand fraction vs superconductive strand
    f_Cool : float
        Cooling fraction
    Choice_Buck_Wedg : str
        Mechanical support configuration: 'Bucking', 'Wedging', or 'Plug'
    
    Returns
    -------
    tuple of float
        (d, alpha, B_CS, J_max_CS)
        - d : CS radial thickness (m)
        - alpha : Conductor volume fraction (dimensionless, 0 < alpha < 1)
        - B_CS : CS magnetic field (T)
        - J_max_CS : Maximum current density in conductor (A/m²)
        
        Returns (nan, nan, nan, nan) if no physically valid solution exists.
    
    Notes
    -----
    - Uses thick-wall cylinder approximation for stress calculation
    - Mechanical models vary by support configuration:
        * Bucking: CS bears TF support loads, two limiting cases evaluated
        * Wedging: CS isolated from TF by gap, pure hoop stress
        * Plug: Central plug supports TF, combined pressure loading
    - Numerical tolerances (Tol_CS ~ 1e-3) critical for solution filtering

    References
    ----------
    [Auclair et al. NF 2026]
    """
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    
    # Debug option
    debug = False
    Tol_CS = 1e-3

    # --- compute external CS radius depending on mechanical choice ---
    if Choice_Buck_Wedg in ('Bucking', 'Plug'):
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        if debug:
            print("[f_CS_D0FUS] Invalid mechanical choice:", Choice_Buck_Wedg)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if RCS_ext <= 0.0:
        if debug:
            print("[f_CS_D0FUS] Non-positive RCS_ext:", RCS_ext)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # total flux for CS
    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF
    
    # ------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------

    def d_to_solve(d):
        
        # --- Sanity checks ---
        if d < Tol_CS or d > RCS_ext - Tol_CS:
            if debug:
                print("d problem")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # --- R_int ---
        RCS_int = RCS_ext - d
        
        # --- Compute B, J , alpha ---
        B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        if Supra_choice_CS == 'Manual':
            J_max_CS = Jc_manual * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        elif Supra_choice_CS == 'Rebco':
            J_max_CS = Jc(Supra_choice_CS, B_CS, T_Helium, Jc_Manual) * f_Cu_Strand * f_Cool * f_In
        else :
            J_max_CS = Jc(Supra_choice_CS, B_CS, T_Helium, Jc_Manual) * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        alpha = B_CS / (μ0 * J_max_CS * d)
        
        # --- Sanity checks ---
        if J_max_CS < Tol_CS:
            if debug:
                print(f"J problem: non-positive current density J_max_CS={J_max_CS:.2e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
            if debug:
                print(f"alpha problem: {alpha:.4f} outside valid range (0, 1)")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if B_CS < Tol_CS or B_CS > B_max_CS:
            if debug:
                print(f"B_CS problem: non-positive field B_CS={B_CS:.3f}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # --- Compute the stresses ---
        # Pressures
        P_CS = B_CS**2 / (2.0 * μ0)
        P_TF = B_max_TF**2 / (2.0 * μ0) * (R0 - a - b) / RCS_ext
        # denominateur commun
        denom_stress = RCS_ext**2 - RCS_int**2
        if abs(denom_stress) < 1e-30:
            if debug:
                print("denom_stress problem")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # mechanical models (thick cylinder approximations)
        if Choice_Buck_Wedg == 'Bucking':
            # Light bucking case (J_CS = max)
            Sigma_light = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) -
                           (2.0 * P_TF * RCS_ext**2 / denom_stress)) / (1.0 - alpha)
            # Strong bucking case (J_CS = 0)
            Sigma_strong = (2.0 * P_TF * RCS_ext**2 / denom_stress) / (1.0 - alpha)
            Sigma_CS = max(abs(Sigma_light), abs(Sigma_strong))
            σ_theta = Sigma_CS
            σ_r = 0
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = abs((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) / (1.0 - alpha))
            σ_theta = Sigma_CS
            σ_r = 0
        elif Choice_Buck_Wedg == 'Plug':
            
            def gamma(alpha_val, n_val):
                if alpha_val <= 0 or alpha_val >= 1:
                    return np.nan
                A = 2 * np.pi + 4 * alpha_val * (n_val - 1)
                discriminant = A**2 - 4 * np.pi * (np.pi - 4 * alpha_val)
                if discriminant < 0:
                    return np.nan
                val = (A - np.sqrt(discriminant)) / (2 * np.pi)
                if val < 0 or val > 1:
                    return np.nan
                return val

            # If the CS pressure is dominant:
            if abs(P_CS - P_TF) > abs(P_TF):
                # classical bucking case
                # Light bucking case (J_CS = max)
                Sigma_light = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) -
                               (2.0 * P_TF * RCS_ext**2 / denom_stress)) / (1.0 - alpha)
                # Strong bucking case (J_CS = 0)
                Sigma_strong = (2.0 * P_TF * RCS_ext**2 / denom_stress) / (1.0 - alpha)
                Sigma_CS = max(abs(Sigma_light), abs(Sigma_strong))
                σ_theta = Sigma_CS
                σ_r = 0
            # Else, plug case:
            elif abs(P_CS - P_TF) <= abs(P_TF):
                # Gamma computation
                gamma = gamma(alpha, n_CS)
                # sigma_r computation
                Sigma_CS = abs(abs(P_TF) / gamma)
                σ_r = Sigma_CS
                σ_theta = 0
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            if debug:
                print("Choice buck wedg problem")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # --- Sanity checks ---
        if Sigma_CS < Tol_CS:
            if debug:
                print(f"Sigma_CS problem: non-positive {Sigma_CS/1e6:.3f}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        σ_z = 0
        Steel_fraction = 1 - alpha

        return (float(Sigma_CS), float(σ_z),float(σ_theta),float(σ_r), float(Steel_fraction), float(B_CS), float(J_max_CS))
    
    # ------------------------------------------------------------------
    # Root function
    # ------------------------------------------------------------------
    
    # define helper function for root search
    def f_sigma_diff(d):
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d)
        if not np.isfinite(Sigma_CS):
            return np.nan
        val = Sigma_CS - σ_CS 
        return val  # we want this to be 0
    
    # ------------------------------------------------------------------
    # Plot option
    # ------------------------------------------------------------------
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = x_range
        y = [CS_to_solve(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='CS_to_solve(d)')
        plt.axhline(y=0, color='r', linestyle='--', label='y=0')
        plt.grid(True)
        plt.xlabel('d')
        plt.ylabel('Function Value')
        plt.title('Comportement de CS_to_solve')
        plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        return zero_crossings
    
    # --------------------------------------------------------------
    # Sign change detection
    # --------------------------------------------------------------
    def find_sign_changes(f, a, b, n):
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        sign_changes = []
        for i in range(1, len(x_vals)):
            if not (np.isfinite(y_vals[i-1]) and np.isfinite(y_vals[i])):
                continue
            if y_vals[i-1] * y_vals[i] < 0:
                sign_changes.append((x_vals[i-1], x_vals[i]))
        return sign_changes

    # --------------------------------------------------------------
    # Reffinement with BrentQ
    # --------------------------------------------------------------
    def refine_zeros(f, intervals):
        roots = []
        for a, b in intervals:
            try:
                root = brentq(f, a, b)
                roots.append(root)
            except ValueError:
                continue
        return roots

    # ------------------------------------------------------------------
    # Root-finding
    # ------------------------------------------------------------------
    
    def find_d_solution():
        """
        Trouve d tel que Sigma_CS = σ_CS en utilisant une détection de changement de signe
        puis un raffinement avec la méthode de Brent.
        Retourne (d_sol, alpha, B_CS, J_max_CS)
        """
    
        d_min = Tol_CS
        d_max = RCS_ext - Tol_CS
    
        if debug:
            plot_function_CS(f_sigma_diff, np.linspace(d_min, d_max, 200))
        
        # --------------------------------------------------------------
        # Recherche des intervalles puis des racines
        # --------------------------------------------------------------
        sign_intervals = find_sign_changes(f_sigma_diff, d_min, d_max, n=1500)
        roots = refine_zeros(f_sigma_diff, sign_intervals)
        
        if debug:
            print(f'Possible solutions : {len(roots)}')
            for root in roots:
                print(f'Solutions: {root}')
                print(d_to_solve(root))
            if len(roots) == 0:
                print("[f_CS_D0FUS] Aucun changement de signe détecté.")
            else:
                print(f"[f_CS_D0FUS] Racines candidates détectées : {roots}")
    
        # --------------------------------------------------------------
        # Filtrage des racines valides
        # --------------------------------------------------------------
        valid_solutions = []
        for d_sol in roots:
            if np.isnan(d_sol):
                continue
            try:
                Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d_sol)
                valid_solutions.append((d_sol, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS))
            except Exception:
                continue
    
        if len(valid_solutions) == 0:
            if debug:
                print("[f_CS_D0FUS] Aucune solution valide trouvée.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
        # --------------------------------------------------------------
        # Sélection de la meilleure racine
        # --------------------------------------------------------------
        valid_solutions.sort(key=lambda x: x[0])  # plus petite épaisseur
        d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = valid_solutions[0]
    
        return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS
    
    
    # --- Try to find a valid solution ---
    d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = find_d_solution()

    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

debug_CS = 0
if __name__ == "__main__" and debug_CS == 1:
    J_CS = Jc('Nb3Sn', 13, 4.2, Jc_Manual)
    print(f'Predicted current density ITER: {np.round(J_CS/1e6,2)}')
    print("ITER like")
    print(f_CS_D0FUS(0, 0, 230, 0, 2, 1.25, 0.9, 6.2, 11.8, 40, 400e6, 'Manual', J_CS, 4.2, 0.5, 0.7, 0.75, 'Wedging'))
    print("ARC like")
    print(f_CS_D0FUS(0, 0, 32, 0, 1.07, 0.89, 0.64, 3.3, 23, 40, 500e6, 'Manual', 800e6, 4.2, 0.5, 0.7, 0.75, 'Plug'))
    print("SPARC like")
    print(f_CS_D0FUS(0, 0, 42, 0, 0.57, 0.18, 0.35, 1.85, 20, 40, 500e6, 'Manual', 800e6, 4.2, 0.5, 0.7, 0.75, 'Bucking'))
    
#%% CS D0FUS & CIRCE

def f_CS_CIRCE( ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
    Supra_choice_CS, Jc_manual, T_Helium, f_Cu_Non_Cu, f_Cu_Strand, f_Cool, f_In, Choice_Buck_Wedg):
    
    """
    Calculate the Central Solenoid (CS) thickness using CIRCE:
    The function solves for CS geometry by balancing electromagnetic stresses 
    with structural limits, accounting for different mechanical configurations.
    
    Parameters
    ----------
    ΨPI : float
        Plasma initiation flux (Wb)
    ΨRampUp : float
        Current ramp-up flux swing (Wb)
    Ψplateau : float
        Flat-top operational flux (Wb)
    ΨPF : float
        Poloidal field coil contribution (Wb)
    a : float
        Plasma minor radius (m)
    b : float
        Cumulative radial build: 1st wall + breeding blanket + neutron shield + gaps (m)
    c : float
        Toroidal field (TF) coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T) [currently unused in implementation]
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier for Jc calculation
    Jc_manual : float
        If needed, manual current density
    T_Helium : float
        Helium coolant temperature (K)
    f_Cu_Non_Cu : float
        Copper fraction in the Su strand
    f_Cu : float
        n_Cu strands / n_total strands
    f_Cool : float
        Cooling fraction
    Choice_Buck_Wedg : str
        Mechanical support configuration: 'Bucking', 'Wedging', or 'Plug'
    
    Returns
    -------
    tuple of float
        (d, alpha, B_CS, J_max_CS)
        - d : CS radial thickness (m)
        - alpha : Conductor volume fraction (dimensionless, 0 < alpha < 1)
        - B_CS : CS magnetic field (T)
        - J_max_CS : Maximum current density in conductor (A/m²)
        
        Returns (nan, nan, nan, nan) if no physically valid solution exists.
    
    Notes
    -----
    - Uses CIRCE for mechanical calculations
    - Mechanical models vary by support configuration:
        * Bucking: CS bears TF support loads, two limiting cases evaluated
        * Wedging: CS isolated from TF by gap, pure hoop stress
        * Plug: Central plug supports TF, combined pressure loading
    - Numerical tolerances (Tol_CS ~ 1e-3) is critical for solution filtering

    References
    ----------
    [Auclair et al. NF 2026]
    """
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    
    # Debug option
    debug = False
    Tol_CS = 1e-3

    # --- compute external CS radius depending on mechanical choice ---
    if Choice_Buck_Wedg in ('Bucking', 'Plug'):
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        if debug:
            print("[f_CS_D0FUS] Invalid mechanical choice:", Choice_Buck_Wedg)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if RCS_ext <= 0.0:
        if debug:
            print("[f_CS_D0FUS] Non-positive RCS_ext:", RCS_ext)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # total flux for CS
    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF
    
    # ------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------

    def d_to_solve(d):
        
        # --- Sanity checks ---
        if d < 0.0 + Tol_CS or d > RCS_ext - Tol_CS:
            if debug:
                print("d problem")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # --- R_int ---
        RCS_int = RCS_ext - d
        
        # --- Compute B, J , alpha ---
        B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        if Supra_choice_CS == 'Manual':
            J_max_CS = Jc_manual * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        elif Supra_choice_CS == 'Rebco':
            J_max_CS = Jc(Supra_choice_CS, B_CS, T_Helium, Jc_Manual) * f_Cu_Strand * f_Cool * f_In
        else :
            J_max_CS = Jc(Supra_choice_CS, B_CS, T_Helium, Jc_Manual) * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        alpha = B_CS / (μ0 * J_max_CS * d)
        
        # --- Sanity checks ---
        if J_max_CS < Tol_CS:
            if debug:
                print(f"J problem: non-positive current density J_max_CS={J_max_CS:.2e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
            if debug:
                print(f"alpha problem: {alpha:.4f} outside valid range (0, 1)")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if B_CS < Tol_CS or B_CS > B_max_CS:
            if debug:
                print(f"B_CS problem: non-positive field B_CS={B_CS:.3f}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # --- Compute the stresses ---
        # Pressures
        P_CS = B_CS**2 / (2.0 * μ0)
        P_TF = B_max_TF**2 / (2.0 * μ0) * (R0 - a - b) / RCS_ext
        
        # --- CIRCE computation ---
        if Choice_Buck_Wedg == 'Bucking':
            
            # J_CS = Maximal: Light bucking
            disR = 20                                           # Pas de discrétisation
            R = np.array([RCS_int,RCS_ext])                     # Radii
            J = np.array([J_max_CS*alpha])                      # Current densities
            B = np.array([B_CS])                                # Magnetic fields
            Pi = 0                                              # Internal pressure
            Pe = P_TF                                           # External pressure
            E = np.array([Young_modul_Steel])                # Young's modul
            nu = nu_Steel                                       # Poisson's ratio
            config = np.array([0])                              # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS_light = max(np.abs(SigTtot)) * 1/(1-alpha)
            
            # J_CS = 0 : Strong Bucking
            disR = 20                                          # Pas de discrétisation
            R = np.array([RCS_int,RCS_ext])                    # Radii
            J = np.array([0])                                  # Current densities
            B = np.array([0])                                  # Magnetic fields
            Pi = 0                                             # Internal pressure
            Pe = P_TF                                          # External pressure
            E = np.array([Young_modul_Steel])                  # Young's modul
            nu = nu_Steel                                      # Poisson's ratio
            config = np.array([0])                             # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS_strong = max(np.abs(SigTtot)) * 1/(1-alpha)
            
            σ_theta = max(np.abs(SigTtot)) * 1/(1-alpha)
            σ_r = max(np.abs(SigRtot)) * 1/(1-alpha)
            
            # Final sigma
            Sigma_CS = max(Sigma_CS_light, Sigma_CS_strong)
            
        elif Choice_Buck_Wedg == 'Wedging':
            
            disR = 20                                           # Pas de discrétisation
            R = np.array([RCS_int,RCS_ext])                     # Radii
            J = np.array([J_max_CS*alpha])                      # Current densities
            B = np.array([B_CS])                                # Magnetic fields
            Pi = 0                                              # Internal pressure
            Pe = 0                                              # External pressure
            E = np.array([Young_modul_Steel])                   # Young's modul
            nu = nu_Steel#                                      # Poisson's ratio
            config = np.array([0])                              # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS = max(np.abs(SigTtot)) * 1/(1-alpha)
            
            σ_theta = max(np.abs(SigTtot)) * 1/(1-alpha)
            σ_r = max(np.abs(SigRtot)) * 1/(1-alpha)
            
        elif Choice_Buck_Wedg == 'Plug':
            
            def gamma(alpha_val, n_val):
                if alpha_val <= 0 or alpha_val >= 1:
                    return np.nan
                A = 2 * np.pi + 4 * alpha_val * (n_val - 1)
                discriminant = A**2 - 4 * np.pi * (np.pi - 4 * alpha_val)
                if discriminant < 0:
                    return np.nan
                val = (A - np.sqrt(discriminant)) / (2 * np.pi)
                if val < 0 or val > 1:
                    return np.nan
                return val
            
            # If the CS pressure is dominant:
            if abs(P_CS - P_TF) > abs(P_TF):
                
                # classical bucking:
                # J_CS = Maximal: Light bucking
                disR = 20                                           # Pas de discrétisation
                R = np.array([RCS_int,RCS_ext])                     # Radii
                J = np.array([J_max_CS*alpha])                      # Current densities
                B = np.array([B_CS])                                # Magnetic fields
                Pi = 0                                              # Internal pressure
                Pe = P_TF                                           # External pressure
                E = np.array([Young_modul_Steel])                   # Young's modul
                nu = nu_Steel                                       # Poisson's ratio
                config = np.array([0])                              # TF = 1 , CS = 0
                # Appeler la fonction principale
                SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
                Sigma_CS_light = max(np.abs(SigTtot)) * 1/(1-alpha)
                
                # J_CS = 0 : Strong Bucking
                disR = 20                                          # Pas de discrétisation
                R = np.array([RCS_int,RCS_ext])                    # Radii
                J = np.array([0])                                  # Current densities
                B = np.array([0])                                  # Magnetic fields
                Pi = 0                                             # Internal pressure
                Pe = P_TF                                          # External pressure
                E = np.array([Young_modul_Steel])                  # Young's modul
                nu = nu_Steel                                      # Poisson's ratio
                config = np.array([0])                             # TF = 1 , CS = 0
                # Appeler la fonction principale
                SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
                Sigma_CS_strong = max(np.abs(SigTtot)) * 1/(1-alpha)
                
                # Final sigma
                Sigma_CS = max(Sigma_CS_light, Sigma_CS_strong)
                
                σ_theta = max(np.abs(SigTtot)) * 1/(1-alpha)
                σ_r = max(np.abs(SigRtot)) * 1/(1-alpha)
            
            # Else, plug case:
            elif abs(P_CS - P_TF) <= abs(P_TF):
                # Gamma computation
                gamma = gamma(alpha, n_CS)
                # sigma_r computation
                disR = 20                               # Pas de discrétisation
                R = np.array([(RCS_ext-d),RCS_ext])     # Radii
                J = np.array([0])                       # Current densities
                B = np.array([0])                       # Magnetic fields
                Pi = P_TF                               # Internal pressure
                Pe = P_TF                               # External pressure
                E = np.array([Young_modul_Steel])       # Young's modul
                nu = nu_Steel                           # Poisson's ratio
                config = np.array([0])                  # TF = 1 , CS = 0
                # Appeler la fonction principale
                SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
                Sigma_CS = max(np.abs(SigTtot)) * 1/gamma
                σ_theta = max(np.abs(SigTtot)) * 1/gamma
                σ_r = max(np.abs(SigRtot)) * 1/gamma
                
            else:
                if debug:
                    print("Plug pressure problem")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        else:
            if debug:
                print("Choice buck wedg problem")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # --- Sanity checks ---
        if Sigma_CS < Tol_CS:
            if debug:
                print(f"Sigma_CS problem: non-positive {Sigma_CS/1e6:.3f}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        σ_z = 0
        Steel_fraction = 1 - alpha
        
        return (float(Sigma_CS), float(σ_z),float(σ_theta),float(σ_r), float(Steel_fraction), float(B_CS), float(J_max_CS))
    
    # ------------------------------------------------------------------
    # Root function
    # ------------------------------------------------------------------
    
    # define helper function for root search
    def f_sigma_diff(d):
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d)
        if not np.isfinite(Sigma_CS):
            return np.nan
        return Sigma_CS - σ_CS # we want this to be 0
    
    # --------------------------------------------------------------
    # Sign change detection
    # --------------------------------------------------------------
    def find_sign_changes(f, a, b, n):
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        sign_changes = []
        for i in range(1, len(x_vals)):
            if not (np.isfinite(y_vals[i-1]) and np.isfinite(y_vals[i])):
                continue
            if y_vals[i-1] * y_vals[i] < 0:
                sign_changes.append((x_vals[i-1], x_vals[i]))
        return sign_changes

    # --------------------------------------------------------------
    # Reffinement with BrentQ
    # --------------------------------------------------------------
    def refine_zeros(f, intervals):
        roots = []
        for a, b in intervals:
            try:
                root = brentq(f, a, b)
                roots.append(root)
            except ValueError:
                continue
        return roots

    # ------------------------------------------------------------------
    # Root-finding
    # ------------------------------------------------------------------
    
    def find_d_solution():
        """
        Trouve d tel que Sigma_CS = σ_CS en utilisant une détection de changement de signe
        puis un raffinement avec la méthode de Brent.
        Retourne (d_sol, alpha, B_CS, J_max_CS)
        """
    
        d_min = Tol_CS
        d_max = RCS_ext - Tol_CS
        
        # --------------------------------------------------------------
        # Recherche des intervalles puis des racines
        # --------------------------------------------------------------
        sign_intervals = find_sign_changes(f_sigma_diff, d_min, d_max, n=1500)
        roots = refine_zeros(f_sigma_diff, sign_intervals)
    
        if debug:
            print(f'Possible solutions : {len(roots)}')
            for root in roots:
                print(f'Solutions: {root}')
                print(d_to_solve(root))
            if len(roots) == 0:
                print("[f_CS_D0FUS] Aucun changement de signe détecté.")
            else:
                print(f"[f_CS_D0FUS] Racines candidates détectées : {roots}")
    
        # --------------------------------------------------------------
        # Filtrage des racines valides
        # --------------------------------------------------------------
        valid_solutions = []
        for d_sol in roots:
            if np.isnan(d_sol):
                continue
            try:
                Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d_sol)
                valid_solutions.append((d_sol, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS))
            except Exception:
                continue
    
        if len(valid_solutions) == 0:
            if debug:
                print("[f_CS_D0FUS] Aucune solution valide trouvée.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
        # --------------------------------------------------------------
        # Sélection de la meilleure racine
        # --------------------------------------------------------------
        valid_solutions.sort(key=lambda x: x[0])  # plus petite épaisseur
        d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = valid_solutions[0]
    
        return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS
    
    # --- Try to find a valid solution ---
    d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = find_d_solution()

    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

    
#%% CS Benchmark

if __name__ == "__main__":

    # === Machines definition with their CS parameters and target Ψplateau ===
    machines = {
        "ITER":    {"J_CS":150e6, "Ψplateau": 230, "a_cs": 2.00, "b_cs": 1.25, "c_cs": 0.90, "R0_cs": 6.20, "B_TF": 11.8, "B_cs": 13, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2},
        "EU-DEMO": {"J_CS":150e6, "Ψplateau": 600, "a_cs": 2.92, "b_cs": 1.80, "c_cs": 1.19, "R0_cs": 9.07, "B_TF": 13, "B_cs": 13.5, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2},
        "JT60-SA": {"J_CS":150e6, "Ψplateau":  40, "a_cs": 1.18, "b_cs": 0.27, "c_cs": 0.45, "R0_cs": 2.96, "B_TF": 5.65, "B_cs": 8.9, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2},
        "EAST":    {"J_CS":120e6*(2/3), "Ψplateau":  10, "a_cs": 0.45, "b_cs": 0.4, "c_cs": 0.25, "R0_cs": 1.85, "B_TF": 7.2, "B_cs": 4.7, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "NbTi", "T_CS": 4.2},
        # * 2/3 to account for the copper representation in EAST cables, see PF system:
        # Wu, Songtao, and EAST team. "An overview of the EAST project." Fusion Engineering and Design 82.5-14 (2007): 463-471.
        "ARC":     {"J_CS":600e6, "Ψplateau":  32, "a_cs": 1.07, "b_cs": 0.89, "c_cs": 0.64, "R0_cs": 3.30, "B_TF": 23, "B_cs": 12.9, "σ_CS": 1000e6, "config": "Plug", "SupraChoice": "Rebco", "T_CS": 20},
        "SPARC":   {"J_CS":600e6, "Ψplateau":  42, "a_cs": 0.57, "b_cs": 0.18, "c_cs": 0.35, "R0_cs": 1.85, "B_TF": 20, "B_cs": 25, "σ_CS": 1000e6, "config": "Bucking", "SupraChoice": "Rebco", "T_CS": 20},
    }   # No fatigue in bucking or plug

    # === Accumulate rows for DataFrame ===
    rows_Acad = []
    rows_D0FUS = []
    rows_CIRCE = []
    
    for name, p in machines.items():
        # Unpack inputs
        Ψplateau = p["Ψplateau"]
        a, b, c, R0 = p["a_cs"], p["b_cs"], p["c_cs"], p["R0_cs"]
        B_TF, B_cs, σ = p["B_TF"], p["B_cs"], p["σ_CS"]
        Supra_Choice, T_CS = p["SupraChoice"], p["T_CS"]
        J_strand_CS = p["J_CS"]
        config = p["config"]
        T_Helium = p["T_CS"]
        f_Cu_Strand_benchmark = 0.7
        f_Cool_benchmark = 0.75
        f_Cu_Non_Cu_benchmark = 0.5
        f_In_benchmark = 1
        B_max_CS = 50

        # Call the models with machine-specific configuration
        acad = f_CS_ACAD(0, 0, Ψplateau, 0, a, b, c, R0, B_TF, B_max_CS, σ, 
                         'Manual', J_strand_CS, T_Helium, f_Cu_Non_Cu_benchmark ,f_Cu_Strand_benchmark ,f_Cool_benchmark, f_In_benchmark, config)
        d0fus = f_CS_D0FUS(0, 0, Ψplateau, 0, a, b, c, R0, B_TF, B_max_CS, σ, 
                           'Manual', J_strand_CS, T_CS, f_Cu_Non_Cu_benchmark ,f_Cu_Strand_benchmark ,f_Cool_benchmark,f_In_benchmark, config)
        circe = f_CS_CIRCE(0, 0, Ψplateau, 0, a, b, c, R0, B_TF, B_max_CS, σ, 
                           'Manual', J_strand_CS, T_CS, f_Cu_Non_Cu_benchmark ,f_Cu_Strand_benchmark ,f_Cool_benchmark,f_In_benchmark, config)
        
        def clean_result(val):
            """Return a clean float or NaN if invalid/complex."""
            # Handle None
            if val is None:
                return np.nan
            # Convert to plain float (handles np.float64, etc.)
            val = np.real(val)
            # Check for complex or nan
            if np.iscomplexobj(val) or not np.isfinite(val):
                return np.nan
            # Return rounded float
            return round(float(val), 2)

        # Build one row combining inputs + outputs for each model
        rows_Acad.append({
            "Machine":        name,
            "Config":         config,
            "Ψ [Wb]":         Ψplateau,
            "σ_CS [MPa]":     σ / 1e6,
            "Width [m]":      clean_result(acad[0]),   # d
            "B_CS [T]":       clean_result(acad[5]),   # B_CS (index 5, pas 2!)
        })
        
        rows_D0FUS.append({
            "Machine":        name,
            "Config":         config,
            "Ψ [Wb]":         Ψplateau,
            "σ_CS [MPa]":     σ / 1e6,
            "Width [m]":      clean_result(d0fus[0]),
            "B_CS [T]":       clean_result(d0fus[5]),  # B_CS
        })
        
        rows_CIRCE.append({
            "Machine":        name,
            "Config":         config,
            "Ψ [Wb]":         Ψplateau,
            "σ_CS [MPa]":     σ / 1e6,
            "Width [m]":      clean_result(circe[0]),
            "B_CS [T]":       clean_result(circe[5]),  # B_CS
        })
    
    # === Print table D0FUS ===
    df_d0fus = pd.DataFrame(rows_D0FUS)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=df_d0fus.values,
        colLabels=df_d0fus.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(df_d0fus.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title("CS D0FUS Benchmark", 
              fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    plt.show()

#%% CS plot

if __name__ == "__main__":

    # === Input parameters ===
    a_cs = 3
    b_cs = 1.2
    c_cs = 2
    R0_cs = 9
    B_max_TF_cs = 13
    B_max_CS = 50
    T_He_CS = 4.75
    σ_CS_cs = 300e6 # Red curve on fig.2 of the Sarasola paper
    J_CS = 120e6 # Directly cited in Sarasola paper
    f_Cu_Strand_benchmark = 0.7
    f_Cool_benchmark = 0.75
    f_Cu_Non_Cu_benchmark = 0.5
    f_In_benchmark = 1
    
    # === Ψplateau scan range ===
    psi_values = np.linspace(0, 500, 30)

    # === Result storage ===
    results = {
        "Academic": {"Wedging": {"thickness": [], "B": []},
                     "Bucking": {"thickness": [], "B": []},
                     "Plug": {"thickness": [], "B": []}},
        "D0FUS": {"Wedging": {"thickness": [], "B": []},
                  "Bucking": {"thickness": [], "B": []},
                  "Plug": {"thickness": [], "B": []}},
        "CIRCE": {"Wedging": {"thickness": [], "B": []},
                  "Bucking": {"thickness": [], "B": []},
                  "Plug": {"thickness": [], "B": []}}
    }

    # === Reference data (external or experimental) ===
    ref_thickness = [1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # [m]
    ref_flux = [209*2, 209*2, 208*2, 205*2, 200*2, 192*2, 185*2, 177*2, 161*2]  # [Wb]

    # Dictionnaire pour stocker les durées
    timings = {"Academic": 0.0, "D0FUS": 0.0, "CIRCE": 0.0}
    
    # === Main loop over Ψplateau ===
    for psi in tqdm(psi_values,desc = 'Scanning Psi'):
        for config in ['Wedging', 'Bucking', 'Plug']:
            # --- Academic model ---
            start = time.time()
            res_acad = f_CS_ACAD(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs,
                      B_max_TF_cs, B_max_CS, σ_CS_cs, 'Manual', J_CS, T_Helium, f_Cu_Non_Cu_benchmark ,f_Cu_Strand_benchmark ,f_Cool_benchmark , f_In_benchmark, config)
            timings["Academic"] += time.time() - start
    
            # --- D0FUS model ---
            start = time.time()
            res_d0fus = f_CS_D0FUS(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs,
                      B_max_TF_cs, B_max_CS, σ_CS_cs, 'Manual', J_CS, T_Helium, f_Cu_Non_Cu_benchmark ,f_Cu_Strand_benchmark ,f_Cool_benchmark ,f_In_benchmark, config)
            timings["D0FUS"] += time.time() - start
    
            # --- CIRCE model ---
            start = time.time()
            res_CIRCE = f_CS_CIRCE(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs,
                      B_max_TF_cs, B_max_CS, σ_CS_cs, 'Manual', J_CS, T_Helium, f_Cu_Non_Cu_benchmark ,f_Cu_Strand_benchmark ,f_Cool_benchmark ,f_In_benchmark, config)
            timings["CIRCE"] += time.time() - start
    
            # --- Store results ---
            results["Academic"][config]["thickness"].append(res_acad[0])
            results["Academic"][config]["B"].append(res_acad[5])
    
            results["D0FUS"][config]["thickness"].append(res_d0fus[0])
            results["D0FUS"][config]["B"].append(res_d0fus[5])
    
            results["CIRCE"][config]["thickness"].append(res_CIRCE[0])
            results["CIRCE"][config]["B"].append(res_CIRCE[5])
    
    # === Fin des calculs ===
    print("\n=== Temps d'exécution total par modèle ===")
    for model, duration in timings.items():
        print(f"{model:>8s} : {duration:.3f} s")


    # === Colors for each model ===
    colors = {"Academic": "blue", "D0FUS": "green", "CIRCE": "red"}

    # === Plotting function ===
    def plot_config(config, quantity, ylabel, title_suffix, ref_data=False):
        plt.figure(figsize=(5, 5))
        for model in results.keys():
            plt.plot(psi_values,
                     results[model][config][quantity],
                     color=colors[model],
                     linestyle='-',
                     linewidth=2,
                     label=f"{model} {config}")

        if ref_data:
            plt.scatter(ref_flux, ref_thickness, color="black",
                        marker='x', s=50, label="MADE")

        plt.xlabel("Ψplateau (Wb)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{config} comparison ({title_suffix})", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.show()

    # === Generate all figures ===
    for config in ["Wedging", "Bucking", "Plug"]:
        # Thickness plots
        plot_config(config, "thickness", "CS thickness [m]", "Coil thickness", ref_data=(config == "Wedging"))

        # Magnetic field plots
        plot_config(config, "B", "B CS [T]", "Magnetic field")

#%% Note:
# CIRCE TF double cylindre en wedging ? voir multi cylindre pour grading ?
# Nécessite la résolution de R_int et R_sep en même temps
# Permettrait aussi de mettre la répartition en tension en rapport de surface
#%% Print

# print("D0FUS_radial_build_functions loaded")