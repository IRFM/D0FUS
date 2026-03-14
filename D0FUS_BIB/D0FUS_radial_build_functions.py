"""
Radial Build functions definition for the D0FUS - Design 0-dimensional for FUsion Systems project
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Import

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_parameterization import *
    from .D0FUS_physical_functions import f_nprof, f_Tprof

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_parameterization import *
    from D0FUS_BIB.D0FUS_physical_functions import f_nprof, f_Tprof
    
    # Initialisation of the default config paramters
    cfg = DEFAULT_CONFIG
    
    # Suppress numpy divide/invalid warnings — these are handled via NaN returns
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
#%% print

if __name__ == "__main__":
    print("##################################################### Steel Model ##########################################################")
    
#%% Steel

def Steel(Chosen_Steel, σ_manual):
    if Chosen_Steel == '316L':
        σ = 660*1e6        # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'N50H':
        σ = 1000*1e6       # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'Manual':
        σ = σ_manual*1e6   # Mechanical limit of the steel considered in [Pa]
    else : 
        print('Choose a valid steel')
    return(σ)

#%% print

if __name__ == "__main__":
    print("##################################################### TF coils number ##########################################################")

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
    
#%% TF coil number test

if __name__ == "__main__":
    
    print("="*70)
    print("ITER prediction for the number of TF")
    print("Considering ripple and port minimal size")
    print("="*70)
    
    R0 = 6.2           # [m] major radius
    a = 2.0            # [m] minor radius
    b = 1.2            # [m] base radial distance
    ripple_adm = 0.01  # 1% ripple
    L_min = 3.6        # [m] minimum toroidal access

    N_TF, ripple, Delta = Number_TF_coils(R0, a, b, ripple_adm, L_min)
    r2 = R0 + a + b + Delta

    print(f"Calculated number of TF coils: {N_TF}")
    print(f"ITER TF coils: 18")
    print(f"Additional Delta = {Delta:.3f} m")
    print(f"Ripple = {ripple*100:.3f}%")
    print(f"L_access = {2*math.pi*r2/N_TF:.3f} m")
    print("="*70)

#%% print

if __name__ == "__main__":
    print("##################################################### Jc Model ##########################################################")

#%% Critical Current Density Scaling Laws for Superconductors

"""
Critical Current Density Scaling Laws for Superconductors
==========================================================

The critical current density Jc​(B,T) of a superconductor decreases as the 
magnetic field and temperature approach the material's critical surface, 
bounded by the upper critical field Bc2​ and critical temperature Tc​.
Empirical scaling laws capture this behavior through fits to experimental
data on production-grade conductors.
Nb₃Sn scaling additionally includes strain dependence ε due to the brittleness of
its crystal structure.
REBCO scaling also accounts for the anisotropy of Jc​ with respect to field orientation.

Scaling laws for Nb3Sn, NbTi, and REBCO superconductors used in fusion magnet design:

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

def J_non_Cu_Nb3Sn(B, T, Eps=-0.003):
    """
    Critical current density for Nb3Sn on non-Cu cross-section.
    
    Formula:
        Jc = (C/B) · s(ε) · (1-t^1.52) · (1-t²) · b^p · (1-b)^q
    
    Parameters
    ----------
    B : float or array
        Magnetic field [T]
    T : float or array
        Temperature [K]
    Eps : float
        Applied strain [-] (default: -0.003, typical operating strain)
        
    Returns
    -------
    J_non_Cu : float or array
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


def J_non_Cu_NbTi(B, T):
    """
    Critical current density for NbTi on non-Cu cross-section.
    
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
    J_non_Cu : float or array
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


def J_non_Cu_REBCO(B, T, Tet=0):
    """
    Engineering current density for REBCO (non copper cross section considered).
    
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
    J_tape : float or array
        Engineering current density [A/m²] on tape cross-section
        
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
    t_tape = 100e-6     # Total tape thickness [m]
    # 50 µm Hasteloy, 2 µm Ag, 2 µm REBCO, 40 µm Cu  -> ~ 100 µm in total
    t_tape_wCu = 60e-6  # tape thickness without Copper
    A_tape_non_Cu = w_tape * t_tape_wCu  # Cross-section [m²]
    
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
    Jc = Ic / A_tape_non_Cu
    
    # Zero outside valid range
    Jc = np.where((tred >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)


#%% Current density validation

if __name__ == "__main__":
    # Superconductor Jc scaling laws: NbTi, Nb3Sn, REBCO — MAGLAB benchmark
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_Jc_scaling(T_op=4.2)

#%% Print
        
if __name__ == "__main__":
    print("##################################################### Cu Model ##########################################################")

#%% Copper Stabilizer Sizing for CICC Conductors
"""
Copper Stabilizer Sizing for CICC Conductors
=============================================

This module provides functions to size the copper stabilizer fraction in 
Cable-In-Conduit Conductors (CICC) based on quench protection requirements.

The approach follows the Maddock (adiabatic hot-spot) criterion: during a quench,
the conductor must survive the thermal transient without exceeding a maximum 
temperature. This sets a maximum allowable current density in the copper stabilizer,
which in turn determines the required Cu/SC ratio.

Physical basis:
    - During quench, current transfers from SC to Cu stabilizer
    - Adiabatic assumption: no heat removal during the fast transient
    - Energy balance: ∫ρJ²dt = ∫Cp·dT  →  J²·t_dump = Z(T_hotspot)
    - Joule integral: Z(T) = ∫(Cp/ρ)dT from T_op to T_hotspot

References
----------
[1] Maddock, B. J., et al. (1969). "Superconductive composites: heat transfer 
    and steady state stabilization". Cryogenics 9(4), 261-273.
[2] Mitchell, N., et al. (2008). "The ITER Magnet System". 
    IEEE Trans. Appl. Supercond. 18(2), 435-440.
[3] Torre, A. "Superconductors for Fusion". CEA Lecture Notes.
"""

# =============================================================================
# COPPER MATERIAL PROPERTIES
# =============================================================================

def get_copper_properties(T, B, RRR):
    """
    Copper resistivity and volumetric heat capacity.
    
    Parameters
    ----------
    T : float
        Temperature [K]
    B : float
        Magnetic field [T]
    RRR : float
        Residual Resistivity Ratio (default: 100)
        
    Returns
    -------
    rho : float
        Electrical resistivity [Ω·m]
    cp_vol : float
        Volumetric heat capacity [J/(m³·K)]
        
    Notes
    -----
    Source: CryoSoft/THEA library (rCopper.m, MAGRCU.m, cCopper.m)
    Valid range: 0.1-1000 K, 0-30 T, RRR 1.5-3000
    
    Includes magnetoresistance effect (Kohler's rule).
    """
    # Clamp inputs to valid range
    TT = max(0.1, min(T, 1000.0))
    R = max(1.5, min(RRR, 3000.0))
    BB = max(0.0, min(B, 30.0))
    
    # --- Resistivity ---
    RHO273 = 1.54e-8  # Resistivity at 273 K [Ω·m]
    P1, P2, P3, P4, P5, P6, P7 = 0.1171e-16, 4.49, 3.841e10, -1.14, 50.0, 6.428, 0.4531
    
    # Zero-temperature resistivity (impurity scattering)
    rhoZero = RHO273 / (R - 1.0)
    
    # Ideal resistivity (phonon scattering)
    arg = min((P5 / TT)**P6, 30.0)
    rhoI = P1 * TT**P2 / (1.0 + P1 * P3 * TT**(P2 + P4) * np.exp(-arg))
    
    # Matthiessen's rule with deviation term
    rhoI0 = P7 * rhoI * rhoZero / (rhoI + rhoZero)
    rho0 = rhoZero + rhoI + rhoI0
    
    # Magnetoresistance (Kohler's rule)
    RHORRR = 2.37e-8
    A1, A2, A3, A4 = 0.382806e-3, 1.32407, 0.167634e-2, 0.789953
    rhoIce = RHO273 + RHORRR / R
    brr = min(BB * rhoIce / rho0, 40.0e3)
    magR = A1 * brr**A2 / (1.0 + A3 * brr**A4) + 1.0 if brr > 1 else 1.0
    
    rho = magR * rho0
    
    # --- Heat capacity ---
    DENSITY = 8900.0  # kg/m³
    T0, T1 = 10.4529369, 48.26583891
    
    if TT <= T0:
        Cp = 0.01188007*TT - 0.00050323*TT**2 + 0.00075762*TT**3
    elif TT <= T1:
        Cp = -5.03283229 + 1.27426834*TT - 0.11610718*TT**2 + 0.00522402*TT**3 - 5.2996e-5*TT**4
    else:
        Cp = (-65.07570094*TT / (1.833505318 + TT)**0.518553624 +
              624.7552517*TT**3 / (16.55124429 + TT)**2.855560719 +
              0.529512119*TT**4 / (-0.000101401 + TT)**2.983928329)
    
    cp_vol = max(Cp, 0) * DENSITY
    
    return rho, cp_vol


def compute_quench_integral(T_op, T_hotspot, B, RRR, n_steps=200):
    """
    Joule integral Z(T) = ∫(Cp/ρ)dT for Maddock criterion.
    
    Parameters
    ----------
    T_op : float
        Operating temperature [K]
    T_hotspot : float
        Maximum allowable hot-spot temperature [K]
    B : float
        Magnetic field [T]
    RRR : float
        Residual Resistivity Ratio (default: 100)
    n_steps : int
        Number of integration steps (default: 200)
        
    Returns
    -------
    Z : float
        Joule integral [A²·s/m⁴]
        
    Reference
    ---------
    Maddock et al. (1969), Eq. 12
    """
    temp_array = np.linspace(T_op, T_hotspot, n_steps)
    dT = (T_hotspot - T_op) / n_steps
    
    Z = 0.0
    for T in temp_array:
        rho, cp = get_copper_properties(T, B, RRR)
        Z += (cp / rho) * dT
    
    return Z


# =============================================================================
# MAGNETIC ENERGY CALCULATIONS
# =============================================================================

def calculate_E_mag_TF(B_max, r_bore_in, r_bore_out, H_TF):
    """
    Magnetic energy stored in toroidal field.
    
    The toroidal field varies as B(r) = B_max × r_bore_in / r.
    Integration over the toroidal volume yields:
        E_mag = (B_max² × r_bore_in² / 2μ₀) × 2π × H_TF × ln(r_bore_out / r_bore_in)
    
    Parameters
    ----------
    B_max : float
        Peak toroidal field at inner bore radius [T]
    r_bore_in : float
        Inner radius of TF bore (where B = B_max) [m]
        Typically: R_0 - a - gaps, or inner radius of TF inner leg
    r_bore_out : float
        Outer radius of toroidal volume [m]
        Typically: R_0 + a + gaps, or outer radius of TF outer leg
    H_TF : float
        Effective height of toroidal field region [m]
        
    Returns
    -------
    E_mag : float
        Magnetic energy [J]
    """
    E_mag = (B_max**2 * r_bore_in**2 / (2 * μ0)) * 2 * np.pi * H_TF * np.log(r_bore_out / r_bore_in)
    return E_mag



def calculate_E_mag_CS(B_max, r_in_CS, r_out_CS, H_CS):
    """
    Magnetic energy stored in Central Solenoid.
    
    The CS is modeled as a thick solenoid with nearly uniform axial field.
    Energy: E_mag = (B_avg² / 2μ₀) × Volume
    
    For a well-designed CS, B_avg ≈ 0.95 × B_max.
    
    Parameters
    ----------
    B_max : float
        Peak magnetic field (at inner radius) [T]
    r_in_CS : float
        Inner radius of CS winding pack [m]
    r_out_CS : float
        Outer radius of CS winding pack [m]
    H_CS : float
        Total height of CS [m]
        Typical estimate: H_CS ≈ 2 × ((κ+1)×a + δ), similar to TF.
        
    Returns
    -------
    E_mag : float
        Magnetic energy [J]
        
    Reference
    ---------
    ITER Design Description Document - CS System.
    """
    V_CS = np.pi * (r_out_CS**2 - r_in_CS**2) * H_CS
    B_avg = 0.95 * B_max
    E_mag = (B_avg**2 / (2 * μ0)) * V_CS
    return E_mag


# =============================================================================
# QUENCH PROTECTION
# =============================================================================

def calculate_t_dump(E_mag, I_cond, V_max, N_sub, tau_h):
    """
    Effective discharge time for hot-spot criterion.
    
    The dump circuit is characterized by:
        - Dump resistor: R_dump = V_max / I_cond
        - Coil inductance: L = 2 × E_mag / I_cond²
        - Discharge time constant: τ_dis = L / (N_sub × R_dump)
    
    The effective time for Maddock criterion:
        t_dump = τ_h + τ_dis / 2
    
    Parameters
    ----------
    E_mag : float
        Total magnetic energy [J]
    I_cond : float
        Operating current per conductor [A]
    V_max : float
        Maximum voltage to ground [V]
        Typical: 5-10 kV
    N_sub : int or float
        Number of subdivisions in protection circuit.
        Each subdivision has its own dump resistor.
    tau_h : float
        Detection + holding time before discharge [s]
        Typical: 2-5 s (LTS), 10-20 s (HTS)
        
    Returns
    -------
    t_dump : float
        Effective discharge time for hot-spot calculation [s]
        
    Reference
    ---------
    Torre, A. CEA Lecture Notes
    
    Note
    ---------
    Voltage convention :
    V_max here refers to the terminal voltage across the dump resistor
    (i.e. V_terminal = I_cond × R_dump), NOT the voltage to ground.
    For a symmetrically grounded circuit, the voltage to ground is:
        V_to_ground = V_terminal / 2
    ITER TF:  V_terminal ≈ 10–12 kV  →  V_to_ground ≈ 5–6 kV  [Fink 2005]
    ITER CS:  V_terminal ≈  6.3 kV   →  V_to_ground ≈ 3.2 kV  [Duchateau 2009]
    EU DEMO:  V_terminal ≈ 10 kV     →  V_to_ground ≈ 5 kV    [Novello 2019]
    A conservative default of 10 kV (terminal) is chosen
    """
    tau_dis = 2 * E_mag / (I_cond * V_max * N_sub)
    t_dump = tau_h + tau_dis / 2
    return t_dump


# =============================================================================
# CABLE FRACTION SIZING
# =============================================================================

def size_cable_fractions(J_non_Cu, B_peak, T_op, t_dump, 
                         T_hotspot, f_He, f_In, RRR):
    """
    Calculate cable composition based on quench protection (Maddock criterion).
    
    The non-steel part of the conductor is composed of:
        - Superconductor (SC): carries current in normal operation
        - Copper (Cu): stabilizer, carries current during quench
        - Helium (He): coolant void fraction
        - Insulation (Ins): electrical insulation between turns
    
    The Cu fraction is sized so that the hot-spot temperature during quench
    does not exceed T_hotspot.
    
    Parameters
    ----------
    J_non_Cu : float
        Critical current density on non-Cu cross-section [A/m²]
        For REBCO, this should be converted from J_tape beforehand.
    B_peak : float
        Peak magnetic field at conductor [T]
    T_op : float
        Operating temperature [K]
    t_dump : float
        Effective discharge time [s] (from calculate_t_dump)
    T_hotspot : float
        Maximum allowable hot-spot temperature [K] (default: 250)
        Note: 250 K accounts for margin from steel heat capacity and He cooling.
        Conservative designs use 150-200 K.
    f_He : float
        Helium void fraction in cable (default: 0.30, typical range: 0.25-0.35)
    f_In : float
        Insulation fraction in non-steel area (default: 0.10)
    RRR : float
        Copper Residual Resistivity Ratio (default: 100)
        
    Returns
    -------
    dict
        f_sc : float   - Superconductor volume fraction (wost) [-]
        f_cu : float   - Copper volume fraction (wost) [-]
        f_He : float   - Helium volume fraction (wost) [-]
        f_In : float   - Insulation volume fraction (wost) [-]
        J_wost : float - Current density without steel [A/m²]
        
    Notes
    -----
    "wost" = Without Steel. All fractions are relative to the non-steel
    cross-section and sum to 1.0. The steel jacket is sized separately
    based on mechanical requirements.
        
    References
    ----------
    [1] Maddock et al. (1969) - Adiabatic hot-spot criterion
    [2] Mitchell et al. (2008) - ITER magnet design
    """
    # Joule integral from T_op to T_hotspot
    Z = compute_quench_integral(T_op, T_hotspot, B_peak, RRR)
    
    # Maximum Cu current density (adiabatic criterion)
    # For exponential decay: ∫J²dt = J₀² × τ/2, so J_max = √(2Z/t_dump)
    J_cu_max = np.sqrt(Z / t_dump)
    
    # Required Cu/SC ratio
    ratio_cu_sc = J_non_Cu / J_cu_max
    
    # Volume fractions in cable (SC + Cu + He)
    f_cable = 1 - f_In  # Cable fraction in non-steel area
    f_strand_in_cable = 1 - f_He  # Strand fraction in cable (SC + Cu)
    
    # SC and Cu fractions relative to cable
    f_sc_in_cable = f_strand_in_cable / (1 + ratio_cu_sc)
    f_cu_in_cable = f_strand_in_cable - f_sc_in_cable
    
    # Convert to fractions relative to non-steel area (wost)
    f_sc = f_sc_in_cable * f_cable
    f_cu = f_cu_in_cable * f_cable
    f_He = f_He * f_cable
    
    # Current density on non-steel area
    J_wost = J_non_Cu * f_sc
    
    return {
        "f_sc": f_sc,
        "f_cu": f_cu,
        "f_He": f_He,
        "f_In": f_In,
        "J_wost": J_wost,
    }

#%% E_mag and t_dump test
    
if __name__ == "__main__":
    """
    Validation of magnetic energy and quench discharge time calculations.
    
    References
    ----------
    [1] Mitchell, N. et al., "The ITER Magnet System", IEEE Trans. Appl. Supercond. 18(2), 2008.
    [2] Fink, S. et al., "Transient electrical behaviour of the ITER TF coils during fast discharge", 
        Fusion Eng. Des. 75-79, 2005.
    [3] Takahashi, Y. et al., "Quench Detection Using Pick-Up Coils for the ITER Central Solenoid",
        IEEE Trans. Appl. Supercond. 15(2), 2005.
    [4] Duchateau, J.-L. et al., "Detailed design of the ITER central solenoid", 
        Fusion Eng. Des. 84(2-6), 2009.
    """
    
    print("\n" + "="*65)
    print("MAGNETIC ENERGY & DISCHARGE TIME VALIDATION")
    print("="*65)
    
    # =========================================================================
    # ITER TF SYSTEM
    # =========================================================================
    # The TF system consists of 18 coils with 9 Fast Discharge Units (FDU).
    # Reference values [1,2]: E_mag = 41 GJ, tau_dis = 11 s, I = 68 kA
    print("\n--- ITER TF ---")
    # Geometry: r_bore_in/out are the inner/outer radii of the toroidal volume
    # where B_tor exists, i.e., the TF inner and outer leg positions.
    B_max_TF = 11.8       # Peak field at inner leg [T]
    r_bore_in_TF = 2.7    # Inner leg radius [m]
    r_bore_out_TF = 9.5   # Outer leg radius [m]
    H_TF = 12.0           # Effective height [m]
    E_TF = calculate_E_mag_TF(B_max_TF, r_bore_in_TF, r_bore_out_TF, H_TF)
    E_TF_ref = 41e9
    # Discharge parameters [2]: 9 FDU, V_max ~ 6 kV, tau_dis ~ 11 s
    I_TF = 68e3
    V_max_TF = 10e3
    N_FDU_TF = 9
    tau_h_TF = 2.0
    t_dump_TF = calculate_t_dump(E_TF, I_TF, V_max_TF, N_FDU_TF, tau_h_TF)
    tau_dis_TF = 2 * E_TF / (I_TF * V_max_TF * N_FDU_TF)
    print(f"  E_mag   = {E_TF/1e9:.1f} GJ   (ref: 41 GJ)")
    print(f"  tau_dis = {tau_dis_TF:.1f} s    (ref: ~11 s)")
    
    # =========================================================================
    # ITER CS SYSTEM
    # =========================================================================
    # The CS consists of 6 independent modules, each with its own FDU.
    # Reference values [3,4]: E_mag = 6.4 GJ, tau_dis = 7.5 s, I = 45 kA
    print("\n--- ITER CS ---")
    # Geometry: thick solenoid with nearly uniform axial field
    B_max_CS = 13.0       # Peak field [T]
    r_in_CS = 1.3         # Inner winding pack radius [m]
    r_out_CS = 2.1        # Outer winding pack radius [m]
    H_CS = 12.0           # Total height [m]
    E_CS = calculate_E_mag_CS(B_max_CS, r_in_CS, r_out_CS, H_CS)
    E_CS_ref = 6.4e9
    # Discharge parameters [3]: 6 modules, tau_dis = 7.5 s
    # V_max derived from: V = 2*E / (I * tau_dis * N) ≈ 6.3 kV
    I_CS = 45e3
    V_max_CS = 6.3e3
    N_FDU_CS = 6
    tau_h_CS = 2.0
    t_dump_CS = calculate_t_dump(E_CS, I_CS, V_max_CS, N_FDU_CS, tau_h_CS)
    tau_dis_CS = 2 * E_CS / (I_CS * V_max_CS * N_FDU_CS)
    print(f"  E_mag   = {E_CS/1e9:.1f} GJ   (ref: 6.4 GJ)")
    print(f"  tau_dis = {tau_dis_CS:.1f} s    (ref: 7.5 s)")
    
#%% Print
        
if __name__ == "__main__":
    print("##################################################### J_cable Model ##########################################################")

#%% Final cable current density

# =============================================================================
# CABLE CURRENT DENSITY
# =============================================================================

def _compute_cable_current_density_core(
    sc_type,
    B_peak,
    T_op,
    E_mag,
    I_cond,
    V_max,
    N_sub,
    tau_h,
    f_He,
    f_In,
    T_hotspot,
    RRR,
    Marge_T_He,
    Marge_T_Nb3Sn,
    Marge_T_NbTi,
    Marge_T_REBCO,
    Eps,
    Tet,
    J_wost_Manual=None,
):
    """
    Core computation function for cable current density calculations.
    
    This function contains the actual physics calculations for determining
    the overall current density in superconducting cables, including the
    fraction of superconductor, copper, helium, and insulation.
    
    Parameters
    ----------
    sc_type : str
        Superconductor type ('Nb3Sn', 'NbTi', 'REBCO', 'Manual')
    B_peak : float
        Peak magnetic field at conductor location [T]
    T_op : float
        Operating temperature [K]
    E_mag : float
        Magnetic energy stored in the coil system [J]
    I_cond : float
        Conductor current [A]
    V_max : float
        Maximum terminal voltage during quench [V]
    N_sub : int
        Number of subcables in the cable
    tau_h : float
         Detection + holding time before discharge [s]
    f_He : float
        Helium fraction in cable [-]
    f_In : float
        Insulation fraction in cable [-]
    T_hotspot : float
        Maximum allowable hotspot temperature [K]
    RRR : float
        Residual resistivity ratio of copper [-]
    Marge_T_He : float
        Temperature margin for helium cooling [K]
    Marge_T_Nb3Sn : float
        Temperature margin for Nb3Sn superconductor [K]
    Marge_T_NbTi : float
        Temperature margin for NbTi superconductor [K]
    Marge_T_REBCO : float
        Temperature margin for REBCO superconductor [K]
    Eps : float
        Strain for Nb3Sn superconductor [-]
    Tet : float
        Field angle for REBCO [rad]
    J_wost_Manual : float, optional
        Manual override for overall current density [A/m²]
        If provided, skips automatic calculation
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'J_non_Cu': Critical current density on SC area [A/m²]
        - 'J_wost': Overall current density in cable [A/m²]
        - 'f_sc': Superconductor fraction [-]
        - 'f_cu': Copper fraction [-]
        - 'f_He': Helium fraction [-]
        - 'f_In': Insulation fraction [-]
        - 't_dump': Effective discharge time [s]
    
    Notes
    -----
    This is the core computational function that should NOT be called directly.
    Use calculate_cable_current_density() instead, which handles caching.
    
    The function calculates the required current density based on:
    - Critical current density of the superconductor (field and temperature dependent)
    - Thermal stability requirements (hotspot temperature limit)
    - Quench protection constraints (maximum terminal voltage)
    
    "wost" = Without Steel. The steel jacket fraction is sized separately
    based on mechanical stress requirements (Lorentz forces, thermal stress).
    """
    
    # Guard against invalid inputs (NaN propagation prevention)
    if np.isnan(B_peak) or np.isnan(E_mag) or np.isnan(N_sub):
        return {
            'J_non_Cu': 0,
            'J_wost': 0,
            'f_sc': 0,
            'f_cu': 0,
            'f_He': f_He,
            'f_In': f_In,
            't_dump': np.nan,
        }
    
    # 1. Superconductor critical current density
    if sc_type == "Manual":
        return {
            'J_non_Cu': np.nan,
            'J_wost': J_wost_Manual,
            'f_sc': np.nan,
            'f_cu': np.nan,
            'f_He': np.nan,
            'f_In': np.nan,
            't_dump': np.nan,
        }
    elif sc_type == "Nb3Sn":
        T_eff = T_op + Marge_T_He + Marge_T_Nb3Sn
        J_non_Cu = J_non_Cu_Nb3Sn(B_peak, T_eff, Eps)
        
    elif sc_type == "NbTi":
        T_eff = T_op + Marge_T_He + Marge_T_NbTi
        J_non_Cu = J_non_Cu_NbTi(B_peak, T_eff)
        
    elif sc_type == "REBCO":
        T_eff = T_op + Marge_T_He + Marge_T_REBCO
        J_non_Cu = J_non_Cu_REBCO(B_peak, T_eff, Tet)
        
    else:
        raise ValueError(f"Unknown superconductor: {sc_type}")
    
    # Handle case where SC is beyond critical surface
    if J_non_Cu <= 0:
        return {
            'J_non_Cu': 0,
            'J_wost': 0,
            'f_sc': 0,
            'f_cu': 0,
            'f_He': f_He * (1 - f_In),
            'f_In': f_In,
            't_dump': np.inf,
        }
    
    # 2. Effective discharge time
    t_dump = calculate_t_dump(E_mag, I_cond, V_max, N_sub, tau_h)
    
    # 3. Size all non-steel fractions (SC, Cu, He, Ins)
    result = size_cable_fractions(
        J_non_Cu=J_non_Cu,
        B_peak=B_peak,
        T_op=T_op,
        t_dump=t_dump,
        T_hotspot=T_hotspot,
        f_He=f_He,
        f_In=f_In,
        RRR=RRR,
    )
    
    return {
        'J_non_Cu': J_non_Cu,
        'J_wost': result['J_wost'],
        'f_sc': result['f_sc'],
        'f_cu': result['f_cu'],
        'f_He': result['f_He'],
        'f_In': result['f_In'],
        't_dump': t_dump,
    }


@lru_cache(maxsize=2000)
def _cached_cable_current_density(
    sc_type: str,
    B_peak_rounded: float,
    T_op: float,
    E_mag_rounded: float,
    I_cond: float,
    V_max: float,
    N_sub: int,
    tau_h: float,
    f_He: float,
    f_In: float,
    T_hotspot: float,
    RRR: float,
    Marge_T_He: float,
    Marge_T_Nb3Sn: float,
    Marge_T_NbTi: float,
    Marge_T_REBCO: float,
    Eps: float,
    Tet: float,
    J_wost_Manual: float,
):
    """
    LRU-cached wrapper for cable current density calculations.
    
    This function provides memoization of expensive cable current density
    calculations. It uses rounded input parameters to improve cache hit rates
    while maintaining sufficient numerical accuracy for engineering design.
    
    Parameters
    ----------
    sc_type : str
        Superconductor type ('Nb3Sn', 'NbTi', 'REBCO', 'Manual')
    B_peak_rounded : float
        Peak magnetic field rounded to 0.01 T precision [T]
    T_op : float
        Operating temperature [K]
    E_mag_rounded : float
        Magnetic energy rounded to 100 kJ precision [J]
    I_cond : float
        Conductor current [A]
    V_max : float
        Maximum terminal voltage during quench [V]
    N_sub : int
        Number of subcables in the cable
    tau_h : float
        Hydraulic time constant [s]
    f_He : float
        Helium fraction in cable [-]
    f_In : float
        Insulation fraction in cable [-]
    T_hotspot : float
        Maximum allowable hotspot temperature [K]
    RRR : float
        Residual resistivity ratio of copper [-]
    Marge_T_He : float
        Temperature margin for helium cooling [K]
    Marge_T_Nb3Sn : float
        Temperature margin for Nb3Sn superconductor [K]
    Marge_T_NbTi : float
        Temperature margin for NbTi superconductor [K]
    Marge_T_REBCO : float
        Temperature margin for REBCO superconductor [K]
    Eps : float
        Strain for Nb3Sn superconductor [-]
    Tet : float
        Field angle for REBCO [rad]
    J_wost_Manual : float
        Manual override for current density [A/m²]
        Use -1.0 to indicate no manual override
    
    Returns
    -------
    dict
        Dictionary containing cable current density results
        (see _compute_cable_current_density_core for details)
    
    Notes
    -----
    This function should NOT be called directly by users. Use the public
    calculate_cable_current_density() function instead.
    
    The cache uses LRU (Least Recently Used) eviction policy with a maximum
    of 2000 entries. This provides significant speedup for iterative design
    optimization where similar cable configurations are evaluated repeatedly.
    
    Cache key considerations:
    - All parameters must be hashable (hence conversion of J_wost_Manual)
    - Rounding is applied to continuous variables to increase cache hits
    - String and integer parameters are used directly
    """
    return _compute_cable_current_density_core(
        sc_type=sc_type,
        B_peak=B_peak_rounded,
        T_op=T_op,
        E_mag=E_mag_rounded,
        I_cond=I_cond,
        V_max=V_max,
        N_sub=N_sub,
        tau_h=tau_h,
        f_He=f_He,
        f_In=f_In,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet,
        J_wost_Manual=J_wost_Manual if J_wost_Manual > 0 else None,
    )


def calculate_cable_current_density(
    sc_type,
    B_peak,
    T_op,
    E_mag,
    I_cond,
    V_max,
    N_sub,
    tau_h,
    f_He,
    f_In,
    T_hotspot,
    RRR,
    Marge_T_He,
    Marge_T_Nb3Sn,
    Marge_T_NbTi,
    Marge_T_REBCO,
    Eps,
    Tet,
    J_wost_Manual=None,
):
    """
    Calculate current density on the non-steel part of the conductor.
    
    This function integrates:
    1. Superconductor J_non_Cu calculation (with technology-specific temperature margins)
    2. Discharge time from quench protection circuit
    3. Copper fraction sizing (Maddock criterion)
    4. Final composition and current density (excluding steel jacket)
    
    Conductor hierarchy:
        - Strand = SC + Cu
        - Cable = Strands + He void
        - Non-steel (wost) = Cable + Insulation (sized here)
        - Conductor = Non-steel + Steel jacket (steel sized separately)
    
    Parameters
    ----------
    sc_type : str
        Superconductor type: "NbTi", "Nb3Sn", "REBCO", or "Manual"
    B_peak : float
        Peak magnetic field at conductor [T]
        Automatically rounded to 0.01 T for caching
    T_op : float
        Operating temperature [K]
    E_mag : float
        Total magnetic energy of the coil [J]
        Automatically rounded to 100 kJ for caching
    I_cond : float
        Operating current per conductor [A]
    V_max : float, optional
        Maximum voltage to ground during dump [V] (default: 10 kV)
    N_sub : int or float, optional
        Number of protection subdivisions (default: 6)
        Automatically converted to integer
    tau_h : float, optional
        Detection + delay time [s] (default: 2.0 s for LTS)
    f_He : float, optional
        Helium void fraction in cable (default: 0.30)
    f_In : float, optional
        Insulation fraction in non-steel area (default: 0.10)
    T_hotspot : float, optional
        Maximum hot-spot temperature [K] (default: 250 K)
    RRR : float, optional
        Copper residual resistivity ratio (default: 100)
    Marge_T_He : float, optional
        Helium temperature margin [K] (default: 0.0)
    Marge_T_Nb3Sn : float, optional
        Nb3Sn-specific temperature margin [K] (default: 1.5)
    Marge_T_NbTi : float, optional
        NbTi-specific temperature margin [K] (default: 1.0)
    Marge_T_REBCO : float, optional
        REBCO-specific temperature margin [K] (default: 2.0)
    Eps : float, optional
        Strain for Nb3Sn (default: -0.003)
    Tet : float, optional
        Field angle for REBCO [rad] (default: 0)
    J_wost_Manual : float, optional
        Manual override for overall current density [A/m²]
        If None, current density is calculated automatically
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'J_non_Cu': Critical current density on SC area [A/m²]
        - 'J_wost': Current density on non-steel area [A/m²]
        - 'f_sc': SC volume fraction (wost) [-]
        - 'f_cu': Cu volume fraction (wost) [-]
        - 'f_He': He volume fraction (wost) [-]
        - 'f_In': Insulation volume fraction (wost) [-]
        - 't_dump': Effective discharge time [s]
        
    Notes
    -----
    Parameter rounding strategy for caching:
    - B_peak: Rounded to 0.01 T (10 mT precision)
    - E_mag: Rounded to 100 kJ (0.1 MJ precision)
    - N_sub: Converted to integer
    
    These rounding levels balance cache efficiency with numerical accuracy
    requirements for tokamak magnet design.
    
    "wost" = Without Steel. The steel jacket fraction is sized separately
    based on mechanical stress requirements (Lorentz forces, thermal stress).
    
    The function returns zero current density if any critical input
    (B_peak, E_mag, N_sub) is NaN, preventing error propagation in
    iterative calculations.
    
    See Also
    --------
    clear_cable_cache : Clear the calculation cache
    get_cache_stats : Retrieve cache performance statistics
    """
    
    # Guard against invalid inputs (NaN propagation prevention)
    if np.isnan(B_peak) or np.isnan(E_mag) or np.isnan(N_sub):
        return {
            'J_non_Cu': 0,
            'J_wost': 0,
            'f_sc': 0,
            'f_cu': 0,
            'f_He': f_He,
            'f_In': f_In,
            't_dump': np.nan,
        }
    
    # Round continuous parameters for improved cache hit rate
    # B_peak rounded to 10 mT precision (sufficient for magnet design)
    B_peak_rounded = round(B_peak, 2)
    
    # E_mag rounded to 100 kJ precision (0.2% accuracy for typical tokamak energies)
    E_mag_rounded = round(E_mag / 1e5) * 1e5
    
    # Convert manual override to cache-compatible format
    # Use -1.0 as sentinel value for "no manual override"
    J_wost_Manual_val = J_wost_Manual if J_wost_Manual is not None else -1.0
    
    # Call cached computation with rounded parameters
    return _cached_cable_current_density(
        sc_type=sc_type,
        B_peak_rounded=B_peak_rounded,
        T_op=T_op,
        E_mag_rounded=E_mag_rounded,
        I_cond=I_cond,
        V_max=V_max,
        N_sub=int(N_sub),  # Ensure integer for cache key
        tau_h=tau_h,
        f_He=f_He,
        f_In=f_In,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet,
        J_wost_Manual=J_wost_Manual_val,
    )


def clear_cable_cache():
    """
    Clear the cable current density calculation cache.
    
    This function should be called when global parameters or material
    properties have been modified, ensuring that subsequent calculations
    use updated values rather than stale cached results.
    
    Examples
    --------
    >>> # After changing material properties or global constants
    >>> clear_cable_cache()
    >>> # Next calculation will use fresh values
    
    Notes
    -----
    Typical scenarios requiring cache clearing:
    - Modification of superconductor critical surface parameterization
    - Changes to material property databases (Cu resistivity, He properties)
    - Updates to quench protection algorithms
    - Switching between different design standards or safety factors
    
    The cache will automatically rebuild as new calculations are performed.
    """
    _cached_cable_current_density.cache_clear()


def get_cache_stats():
    """
    Retrieve cache performance statistics.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'hits': Number of cache hits (reused calculations)
        - 'misses': Number of cache misses (new calculations required)
        - 'hit_rate': Cache hit rate as a fraction [0, 1]
    
    Notes
    -----
    A high hit rate (>80%) indicates effective cache utilization and
    significant computational savings. Low hit rates may suggest:
    - Parameter ranges too broad (consider tighter rounding)
    - Sequential rather than iterative calculations
    - Cache size too small (increase maxsize in @lru_cache)
    
    For optimization studies with >10,000 evaluations, cache hit rates
    above 95% are typical and can reduce computation time by 10-30x.
    """
    info = _cached_cable_current_density.cache_info()
    total = info.hits + info.misses
    return {
        'hits': info.hits,
        'misses': info.misses,
        'hit_rate': info.hits / total if total > 0 else 0
    }

#%% Without Steel current density test

"""
Validation benchmark for cable current density calculations.

Definitions
---------------
J_wost : Current density in winding pack without steel [A/m²]
f_sc   : Fraction of non-copper material (SC filaments + matrix/substrate + solder)
f_cu   : Fraction of copper (stabilizer + matrix + structural Cu)
f_He   : Fraction of helium (void)
f_In   : Fraction of insulation/gaps (cable level only)

References
----------
[1] JT-60SA TF Conductor
    - Sborchia et al., "The toroidal field coils for the JT-60SA tokamak",
      Fusion Engineering and Design 86 (2011) 572-579
    - JT-60SA Technical Design Report (2009)
    
[2] ITER TF Conductor
    - Mitchell et al., "The ITER magnet system", 
      Supercond. Sci. Technol. 25 (2012) 095004
    - ITER Design Description Document (DDD 1.1 - Magnet)
    
[3] SPARC CSMC (PIT VIPER cable)
    - Hartwig et al., "The SPARC Toroidal Field Model Coil Program" (Context on VIPER),
      IEEE Trans. Appl. Supercond. (2024)
    - CFS Press Release, "CFS Second Breakthrough... PIT VIPER", Oct 2024
    - Salazar et al., "Fiber optic quench detection... on VIPER cable",
      Supercond. Sci. Technol. 34 (2021)
    - SPARC CSMC Test parameters (nominal design values)
"""

if __name__ == "__main__":
    
    results = []
    
    # -- Unpack superconductor parameters from global configuration --
    T_hotspot     = cfg.T_hotspot
    RRR           = cfg.RRR
    Marge_T_He    = cfg.Marge_T_He
    Marge_T_Nb3Sn = cfg.Marge_T_Nb3Sn
    Marge_T_NbTi  = cfg.Marge_T_NbTi
    Marge_T_REBCO = cfg.Marge_T_REBCO
    Eps           = cfg.Eps
    Tet           = cfg.Tet
    f_He          = cfg.f_He
    f_In          = cfg.f_In
    
    # =========================================================================
    # JT-60SA TF (NbTi CICC) - [1]
    # =========================================================================
    # Conductor geometry [Sborchia 2011, Table 1]:
    #   - Rectangular jacket: 22 × 26 mm² (outer dimensions)
    #   - Wall thickness: 2 mm (316LN stainless steel)
    #   - Cable space: 18 × 22 mm²
    jacket_w = 22.0              # mm [Sborchia 2011]
    jacket_h = 26.0              # mm [Sborchia 2011]
    wall = 2.0                   # mm [Sborchia 2011]
    cable_w = jacket_w - 2*wall  # 18 mm
    cable_h = jacket_h - 2*wall  # 22 mm
    A_cable = cable_w * cable_h  # 396 mm²
    # Strand composition [Sborchia 2011, Table 2]:
    #   - Total: 486 strands
    #   - 324 NbTi strands (2/3) + 162 pure Cu strands (1/3)
    #   - Strand diameter: 0.81 mm
    #   - Cu:non-Cu ratio in NbTi strand: ~2.0
    n_NbTi = 324                 # [Sborchia 2011]
    n_Cu = 162                   # [Sborchia 2011]
    d_strand = 0.81              # mm [Sborchia 2011]
    Cu_nonCu = 2.0               # Typical value
    A_strand = np.pi * (d_strand/2)**2  # 0.515 mm²
    A_NbTi = n_NbTi * A_strand          # 166.9 mm²
    A_Cu_pure = n_Cu * A_strand         # 83.4 mm²
    f_NbTi = 1/(1+Cu_nonCu)             # 33.3% (NbTi in SC strand)
    A_nonCu = A_NbTi * f_NbTi           # 55.6 mm² (NbTi filaments)
    A_Cu_matrix = A_NbTi * (1-f_NbTi)   # 111.3 mm² (Cu in SC strand)
    A_Cu_tot = A_Cu_matrix + A_Cu_pure  # 194.7 mm²
    # Void fraction [Sborchia 2011]:
    void = 0.33                  # [Sborchia 2011, measured]
    A_He = void * A_cable        # 130.7 mm²
    # Operating parameters:
    I_op = 25.7e3                # A [Sborchia 2011]
    B_peak = 5.65                # T [JT-60SA TDR]
    J_wost_ref  = I_op / (A_cable * 1e-6)  # 64.9 MA/m²
    
    # JT-60SA:
    calc = calculate_cable_current_density(
            sc_type="NbTi",
            B_peak=B_peak,
            T_op=4.2,
            E_mag=1.06e9,
            I_cond=I_op,
            V_max=5e3,
            N_sub=3,
            tau_h=1.0,
            f_He=0.33,
            f_In=0.05,
            T_hotspot=T_hotspot,
            RRR=RRR,
            Marge_T_He=Marge_T_He,
            Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi,
            Marge_T_REBCO=Marge_T_REBCO,
            Eps=Eps,
            Tet=Tet
            )

    
    results.append({
        "name": "JT-60SA TF", "ref": "[1]", "sc": "NbTi", "B": B_peak,
        "J_ref": J_wost_ref, "f_sc_ref": A_nonCu/A_cable, "f_cu_ref": A_Cu_tot/A_cable,
        "f_He_ref": void, "f_In_ref": 1 - A_nonCu/A_cable - A_Cu_tot/A_cable - void,
        "J_calc": calc['J_wost'], "f_sc_calc": calc['f_sc'], "f_cu_calc": calc['f_cu'],
        "f_He_calc": calc['f_He'], "f_In_calc": calc['f_In'],
    })
    
    # =========================================================================
    # ITER TF (Nb3Sn CICC) - [2]
    # =========================================================================
    # Conductor geometry [Mitchell 2012]:
    #   - Circular jacket: Ø 43.7 mm
    #   - Wall thickness: 2 mm
    #   - Central cooling spiral: Ø 10 mm
    D_jacket = 43.7              # mm [Mitchell 2012]
    wall = 2.0                   # mm [Mitchell 2012]
    D_spiral = 10.0              # mm [Mitchell 2012]
    D_int = D_jacket - 2*wall    # 39.7 mm
    A_jacket = np.pi * (D_int/2)**2      # 1237 mm²
    A_spiral = np.pi * (D_spiral/2)**2   # 78.5 mm²
    A_cable = A_jacket - A_spiral        # 1159 mm² (cable space)
    A_wost = A_jacket                    # 1237 mm²
    # Strand composition [Mitchell 2012]:
    #   - 900 Nb3Sn strands + 522 pure Cu strands
    #   - Strand diameter: 0.82 mm
    #   - Cu:non-Cu ratio: 1.0
    n_Nb3Sn = 900                # [Mitchell 2012]
    n_Cu = 522                   # [Mitchell 2012]
    d_strand = 0.82              # mm [Mitchell 2012]
    Cu_nonCu = 1.0               # [ITER DDD]
    A_strand = np.pi * (d_strand/2)**2  # 0.528 mm²
    A_Nb3Sn = n_Nb3Sn * A_strand        # 475.3 mm²
    A_Cu_pure = n_Cu * A_strand         # 275.7 mm²
    f_Nb3Sn = 1/(1+Cu_nonCu)            # 50%
    A_nonCu = A_Nb3Sn * f_Nb3Sn         # 237.6 mm²
    A_Cu_matrix = A_Nb3Sn * (1-f_Nb3Sn) # 237.6 mm²
    A_Cu_tot = A_Cu_matrix + A_Cu_pure  # 513.3 mm²
    # Void fraction:
    void = 0.33                  # [Mitchell 2012]
    A_He = void * A_cable        # 382.5 mm²
    # Operating parameters:
    I_op = 68.0e3                # A [Mitchell 2012]
    B_peak = 11.8                # T [ITER DDD]
    J_wost_ref  = I_op / (A_wost * 1e-6)  # 54.9 MA/m²
    
    # ITER TF:
    calc = calculate_cable_current_density(
        sc_type="Nb3Sn",
        B_peak=B_peak,
        T_op=4.2,
        E_mag=41e9,
        I_cond=I_op,
        V_max=10e3,
        N_sub=9,
        tau_h=5.0,
        f_He= 0.33,
        f_In=0.05,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet
    )
    
    results.append({
        "name": "ITER TF", "ref": "[2]", "sc": "Nb3Sn", "B": B_peak,
        "J_ref": J_wost_ref, "f_sc_ref": A_nonCu/A_cable, "f_cu_ref": A_Cu_tot/A_cable,
        "f_He_ref": void, "f_In_ref": 1 - A_nonCu/A_cable - A_Cu_tot/A_cable - void,
        "J_calc": calc['J_wost'], "f_sc_calc": calc['f_sc'], "f_cu_calc": calc['f_cu'],
        "f_He_calc": calc['f_He'], "f_In_calc": calc['f_In'],
    })
    
    # =========================================================================
    # SPARC CS (PIT VIPER REBCO) - [3]
    # =========================================================================
    # Geometry:
    #   - Copper Jacket OD: ~26 mm
    #   - Turn Insulation: Essential for CS stack. Assumed 0.5 mm (fiberglass tape).
    #   - NOTE: This creates a ~5-8% insulation fraction area.
    D_core_out = 26.0            # mm (Conductor OD)
    t_ins = 0.5                  # mm (Turn insulation)
    D_wost = D_core_out + 2*t_ins # 27.0 mm
    A_wost = np.pi * (D_wost/2)**2 # 572.5 mm²
    # Internal components (same as before)
    D_core = 22.0
    D_cool = 6.0
    A_core = np.pi * (D_core/2)**2
    A_cool = np.pi * (D_cool/2)**2
    A_jacket_cu = np.pi/4 * (D_core_out**2 - D_core**2)
    # Tape Stack (PIT VIPER 50kA Class)
    n_tapes = 120
    w_tape = 4.0
    t_tape = 0.100
    A_tapes = n_tapes * (w_tape * t_tape) # 48.0 mm²
    # Fractions within tape
    f_nonCu_tape = 0.55
    f_Cu_tape = 0.45
    A_nonCu_tape = A_tapes * f_nonCu_tape
    A_Cu_tape = A_tapes * f_Cu_tape
    # Channels & Solder
    A_channels = 160.0
    A_solder = A_channels - A_tapes
    # Aggregates
    A_Cu_core = A_core - A_cool - A_channels
    A_Cu_tot = A_Cu_core + A_jacket_cu + A_Cu_tape
    A_nonCu_tot = A_nonCu_tape + A_solder # Solder + SC/Substrate
    A_He = A_cool # Central channel
    # Operating Parameters (Design Values)
    # Using the FULL SPARC CS design requirement, not just the model coil limit.
    I_op = 50.0e3    # A
    B_peak = 24.0    # T [Design requirement for SPARC CS]
    J_wost_ref  = I_op / (A_wost * 1e-6) # 87.3 MA/m²
    
    # SPARC CS
    calc = calculate_cable_current_density(
        sc_type="REBCO",
        B_peak=B_peak,
        T_op=20.0,
        E_mag=0.5e9,
        I_cond=I_op,
        V_max=5e3,
        N_sub=6,
        tau_h=10,
        f_He= 0.05,
        f_In= 0.07,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet
    )
    
    results.append({
        "name": "SPARC CS", "ref": "[3]", "sc": "REBCO", "B": B_peak,
        "J_ref": J_wost_ref, "f_sc_ref": A_nonCu_tot/A_wost, "f_cu_ref": A_Cu_tot/A_wost,
        "f_He_ref": A_He/A_wost, "f_In_ref": 1 - (A_nonCu_tot + A_Cu_tot + A_He)/A_wost,
        "J_calc": calc['J_wost'], "f_sc_calc": calc['f_sc'], "f_cu_calc": calc['f_cu'],
        "f_He_calc": calc['f_He'], "f_In_calc": calc['f_In'],
    })
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*90)
    print("CABLE CURRENT DENSITY BENCHMARK")
    print("="*90)
    print(f"\n{'Machine':<14} {'Type':<6} {'SC':<7} {'B[T]':<6} {'J_wost[MA/m²]':<14} "
          f"{'f_sc[%]':<9} {'f_cu[%]':<9} {'f_He[%]':<9} {'f_In[%]':<9}")
    print("-"*90)
    
    for r in results:
        # Ligne référence
        print(f"{r['name']:<14} {'Ref':<6} {r['sc']:<7} {r['B']:<6.2f} "
              f"{r['J_ref']/1e6:<14.1f} "
              f"{r['f_sc_ref']*100:<9.1f} {r['f_cu_ref']*100:<9.1f} "
              f"{r['f_He_ref']*100:<9.1f} {r['f_In_ref']*100:<9.1f}")
        # Ligne calculée
        print(f"{'':<14} {'Calc':<6} {'':<7} {'':<6} "
              f"{r['J_calc']/1e6:<14.1f} "
              f"{r['f_sc_calc']*100:<9.1f} {r['f_cu_calc']*100:<9.1f} "
              f"{r['f_He_calc']*100:<9.1f} {r['f_In_calc']*100:<9.1f}")
        print("-"*90)

if __name__ == "__main__":
    # Worst-case cable current density vs field under quench protection
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_cable_current_density(
        E_mag=6e9, I_cond=45e3, V_max=10e3, N_sub=6, tau_h=0.5, cfg=cfg)

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
    # CIRCE0D stress solver validation: Lame, CS 3-layer, composite
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_CIRCE_stress_validation()

#%% print

if __name__ == "__main__":
    print("##################################################### TF Model ##########################################################")
    
#%% Academic model

def f_TF_academic(a, b, R0, σ_TF, J_max_TF, B_max_TF, Choice_Buck_Wedg,
                  coef_inboard_tension, F_CClamp):
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
        μ0: Vacuum permeability [H/m]
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

    # Validate J_max before proceeding
    if J_max is None or J_max <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

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


def Nose_D0FUS(R_ext_Nose, sigma_max, omega, B_max, R_0, a, b,
               coef_inboard_tension):
    """
    Compute the internal radius Ri based on analytical expressions.

    Parameters:
    - R_ext_Nose : float, external radius at the nose (R_ext^Nose)
    - sigma_max  : float, maximum admissible stress
    - beta       : float, dimensionless coefficient (0 ≤ beta ≤ 1)
    - B_max      : float, maximum magnetic field
    - μ0       : float, magnetic permeability of vacuum
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

def f_TF_D0FUS(a, b, R0, σ_TF, J_max_TF, B_max_TF, Choice_Buck_Wedg, omega, n,
               c_BP, coef_inboard_tension, F_CClamp):
    
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
        
        c_Nose = R0 - a - b - c_WP - Nose_D0FUS(R0 - a - b - c_WP, σ_TF, omega, B_max_TF, R0, a, b,
                                          coef_inboard_tension)

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


#%% TF Benchmark
if __name__ == "__main__":
    # TF coil benchmark table: D0FUS vs published references
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_TF_benchmark_table(cfg=cfg)

#%% TF plot
    
if __name__ == "__main__":
    # TF winding-pack thickness vs B_max — Academic vs D0FUS, EU-DEMO geometry
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_TF_thickness_vs_field(
        a=3.0, b=1.7, R0=9.0, sigma_TF=860e6, J_max_TF=50e6, cfg=cfg)

#%% Print
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% Resistivity

def eta_old(T_keV, ne, Z_eff=1.0):
    """
    Simple Spitzer resistivity from Wesson (no Z_eff, no ln(Lambda) dependence).
    """
    eta = 2.8e-8 / (T_keV**1.5)  # Spitzer resistivity from Wesson
    
    return eta

def eta_spitzer(T_keV, ne, Z_eff=1.0):
    """
    Classical Spitzer-Härm parallel resistivity [Ohm.m].
    
    Accounts for electron-electron collisions via the g(Z) factor from 
    Spitzer & Härm (1953). This correction reduces resistivity compared 
    to naive Z_eff scaling, especially at high Z_eff.
    
    Formula:
        eta = 1.65e-9 * ln(Lambda) * T_keV^(-1.5) * Z_eff * g(Z)/g(1)
        
    where g(Z) = (1 + 1.198*Z + 0.222*Z^2) / (1 + 2.966*Z + 0.753*Z^2)
    captures e-e collision effects. Tabulated Spitzer values: 
        g(1)=0.513, g(2)=0.438, g(4)=0.362
    
    Coulomb logarithm (NRL Formulary):
        T < 10 eV:  ln(L) = 23 - ln(ne_cm3^0.5 * T_eV^-1.5)
        T > 10 eV:  ln(L) = 24.15 - ln(ne_cm3^0.5 * T_eV^-1)
    
    References:
        [1] L. Spitzer & R. Härm, Phys. Rev. 89, 977 (1953)
        [2] NRL Plasma Formulary (2019), p.34
    """
    T_eV = T_keV * 1000.0
    ne_cm3 = ne * 1e-6
    
    # Coulomb logarithm with temperature-dependent formula
    if T_eV < 10:
        ln_lambda = 23.0 - np.log(ne_cm3**0.5 * T_eV**(-1.5))
    else:
        ln_lambda = 24.15 - np.log(ne_cm3**0.5 * T_eV**(-1.0))
    ln_lambda = max(ln_lambda, 5.0)
    
    # Spitzer-Härm g(Z) correction factor
    g_Z = (1 + 1.198*Z_eff + 0.222*Z_eff**2) / (1 + 2.966*Z_eff + 0.753*Z_eff**2)
    g_1 = (1 + 1.198 + 0.222) / (1 + 2.966 + 0.753)
    coef_Zeff = Z_eff * g_Z / g_1
    
    return 1.65e-9 * ln_lambda * T_keV**(-1.5) * coef_Zeff

def eta_sauter(T_keV, ne, Z_eff, epsilon, q=2.0, R0=6.2):
    """
    Neoclassical resistivity according to Sauter et al. (1999).
    
    Trapped electrons (f_t ~ 1.46*sqrt(eps)) reduce parallel conductivity.
    The effective trapped fraction interpolates between banana (low nu*) 
    and Pfirsch-Schlüter (high nu*) regimes.
    
    Key relations (Sauter Eqs. 16a-b):
        sigma_neo/sigma_Sp = 1 - (1+0.36/Z)*X + 0.59/Z*X^2 - 0.23/Z*X^3
        f_t_eff = f_t / [1 + 0.26*(1-f_t)*sqrt(nu*)*(1+0.18*(Z-1)^0.5)
                         + 0.18*(1-0.37*f_t)*nu*/sqrt(Z)]
    
    Collisionality (Eq. 18c):
        nu*_e = 0.012 * q*R*Z_eff*n_19*ln(L) / (eps^1.5 * T_keV^2)
    
    References:
        [1] O. Sauter et al., Phys. Plasmas 6, 2834 (1999)
        [2] O. Sauter et al., Erratum, Phys. Plasmas 9, 5140 (2002)
        [3] https://crppwww.epfl.ch/~sauter/neoclassical/
    """
    T_eV = T_keV * 1000.0
    ne_cm3 = ne * 1e-6
    
    if T_eV < 10:
        ln_lambda = 23.0 - np.log(ne_cm3**0.5 * T_eV**(-1.5))
    else:
        ln_lambda = 24.15 - np.log(ne_cm3**0.5 * T_eV**(-1.0))
    ln_lambda = max(ln_lambda, 5.0)
    
    # Geometric trapped fraction (Kim et al. PoF 1991)
    sqrt_eps = np.sqrt(epsilon)
    f_t = 1.46 * sqrt_eps * (1 - 0.54 * sqrt_eps)
    
    # Electron collisionality
    n_19 = ne / 1.0e19
    nu_star_e = 0.012 * q * R0 * Z_eff * n_19 * ln_lambda / (epsilon**1.5 * T_keV**2)
    
    # Effective trapped fraction (Eq. 16b)
    sqrt_nu = np.sqrt(nu_star_e)
    denom = (1.0 + 0.26*(1-f_t)*sqrt_nu*(1 + 0.18*(Z_eff-1)**0.5)
             + 0.18*(1-0.37*f_t)*nu_star_e/np.sqrt(Z_eff))
    f_t_eff = f_t / denom
    
    # Conductivity ratio (Eq. 16a)
    X = f_t_eff
    sigma_ratio = 1.0 - (1.0+0.36/Z_eff)*X + 0.59/Z_eff*X**2 - 0.23/Z_eff*X**3
    
    return eta_spitzer(T_keV, ne, Z_eff) / sigma_ratio

def eta_redl(T_keV, ne, Z_eff, epsilon, q=2.0, R0=6.2):
    """
    Neoclassical resistivity according to Redl et al. (2021).
    
    Improved Sauter formulae refitted against NEO drift-kinetic code.
    Better accuracy at high collisionality (edge pedestal) and with impurities.
    
    Key relations (Redl Eqs. 17-18):
        sigma_neo/sigma_Sp = 1 - (1+0.21/Z)*X + 0.54/Z*X^2 - 0.33/Z*X^3
        f_t_eff = f_t / [1 + 0.25*(1-0.7*f_t)*sqrt(nu*)*(1+0.45*(Z-1)^0.5)
                         + 0.61*(1-0.41*f_t)*nu*/sqrt(Z)]
    
    References:
        [1] A. Redl et al., Phys. Plasmas 28, 022502 (2021)
        [2] Data: https://doi.org/10.5281/zenodo.4072358
    """
    T_eV = T_keV * 1000.0
    ne_cm3 = ne * 1e-6
    
    if T_eV < 10:
        ln_lambda = 23.0 - np.log(ne_cm3**0.5 * T_eV**(-1.5))
    else:
        ln_lambda = 24.15 - np.log(ne_cm3**0.5 * T_eV**(-1.0))
    ln_lambda = max(ln_lambda, 5.0)
    
    # Geometric trapped fraction
    sqrt_eps = np.sqrt(epsilon)
    f_t = 1.46 * sqrt_eps * (1 - 0.54 * sqrt_eps)
    
    # Electron collisionality
    n_19 = ne / 1.0e19
    nu_star_e = 0.012 * q * R0 * Z_eff * n_19 * ln_lambda / (epsilon**1.5 * T_keV**2)
    
    # Effective trapped fraction (Eq. 18)
    sqrt_nu = np.sqrt(nu_star_e)
    denom = (1.0 + 0.25*(1-0.7*f_t)*sqrt_nu*(1 + 0.45*(Z_eff-1)**0.5)
             + 0.61*(1-0.41*f_t)*nu_star_e/np.sqrt(Z_eff))
    f_t_eff = f_t / denom
    
    # Conductivity ratio (Eq. 17)
    X = f_t_eff
    sigma_ratio = 1.0 - (1.0+0.21/Z_eff)*X + 0.54/Z_eff*X**2 - 0.33/Z_eff*X**3
    
    return eta_spitzer(T_keV, ne, Z_eff) / sigma_ratio

if __name__ == "__main__":
    # Plasma resistivity: Wesson / Spitzer / Sauter / Redl — ITER-like parameters
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_resistivity_models(ne=1e20, Z_eff=1.7, R0=6.2, a=2.0)

def f_Reff(a, kappa, R0, Tbar, nbar, Z_eff, q, nu_T, nu_n, eta_model='spitzer',
           rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0):
    """
    Compute effective plasma resistance by numerical integration.
    
    In stationary ohmic regime, the loop electric field E_phi is quasi-uniform 
    across the plasma cross-section. The current density follows:
        j(rho) = E_phi / eta(rho)
    
    Modeling the plasma as parallel resistances (one per flux surface),
    the effective resistance is:
        R_eff = 2*pi*R0 / integral(dA/eta)
    
    where the integral runs over the plasma cross-section.
    With elliptical cross-section: dA = 2*pi*a^2*kappa * rho * drho
    
    This formulation naturally accounts for:
        - Current peaking in the hot core (where eta is low)
        - The full temperature dependence of eta (including neoclassical
          effects where eta depends on collisionality nu* ~ n*T^-2)
    
    Parameters
    ----------
    a : float
        Minor radius [m]
    kappa : float
        Elongation
    R0 : float
        Major radius [m]
    Tbar : float
        Mean temperature [keV]
    nbar : float
        Mean density [1e20 m^-3]
    Z_eff : float
        Effective charge
    q : float
        Safety factor (used for neoclassical models)
    nu_T : float
        Temperature profile peaking exponent (core region).
    nu_n : float
        Density profile peaking exponent (core region).
    eta_model : str
        Resistivity model: 'old', 'spitzer', 'sauter', 'redl'
    rho_ped : float
        Normalised pedestal radius (1.0 = no pedestal, purely parabolic).
    n_ped_frac : float
        n_ped / nbar.  Ignored when rho_ped = 1.0.
    T_ped_frac : float
        T_ped / Tbar.  Ignored when rho_ped = 1.0.
    
    Returns
    -------
    R_eff : float
        Effective plasma resistance [Ohm]

    Notes
    -----
    Integration stops at rho_max = 0.98 to avoid edge singularities where
    T -> 0 and eta -> infinity.  With a pedestal profile, T drops abruptly
    near rho_ped, making this guard even more important.
    """
    from scipy import integrate
    import warnings
    from scipy.integrate import IntegrationWarning
    
    epsilon = a / R0
    
    def eta_local(rho):
        """Compute resistivity at normalized radius rho using profile-aware functions."""
        T_loc = max(float(f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac)), 0.1)
        n_loc = float(f_nprof(nbar, nu_n, rho, rho_ped, n_ped_frac))
        
        if eta_model == 'old':
            return eta_old(T_loc, n_loc, Z_eff)
        elif eta_model == 'spitzer':
            return eta_spitzer(T_loc, n_loc, Z_eff)
        elif eta_model == 'sauter':
            return eta_sauter(T_loc, n_loc, Z_eff, epsilon, q, R0)
        elif eta_model == 'redl':
            return eta_redl(T_loc, n_loc, Z_eff, epsilon, q, R0)
        else:
            raise ValueError(f"Unknown eta_model '{eta_model}'. Choose from: 'old', 'spitzer', 'sauter', 'redl'")
    
    def integrand(rho):
        """
        Integrand: rho / eta(rho)
        
        Regularized to avoid numerical issues when eta becomes very small
        (hot core) or very large (edge).
        """
        eta = eta_local(rho)
        eta_reg = max(eta, 1e-12)
        return rho / eta_reg
    
    # Suppress integration warnings (roundoff errors are acceptable for this application)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=IntegrationWarning)
        
        integral_result, error = integrate.quad(
            integrand, 
            0, 
            0.98,              # Avoid extreme edge (plasma becomes cold)
            limit=200,
            epsabs=1e-8,
            epsrel=1e-4
        )
        
        if error > 1e-3 * abs(integral_result):
            warnings.warn(
                f"Plasma resistance integration may have significant error: "
                f"relative error ~ {error/abs(integral_result):.2e}",
                RuntimeWarning
            )
    
    # R_eff = 2*pi*R0 / integral(dA/eta)
    #       = 2*pi*R0 / (2*pi*a^2*kappa * integral(rho/eta drho))
    R_eff = 2 * np.pi * R0 / (2 * np.pi * a**2 * kappa * integral_result)
    
    return R_eff

def f_li(nu_T, nu_n, Tbar, nbar, Z_eff, a, R0, q,
         eta_model='redl', n_pts=100,
         rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0):
    """
    Compute the normalized plasma internal inductance li by direct numerical
    integration of the current density profile.

    Physical basis
    --------------
    In ohmic quasi-steady state the loop electric field E_phi is uniform
    across the plasma cross-section (flux-surface averaged). The local current
    density is therefore:

        j(rho) = E_phi / eta(rho)   [A/m^2]

    where eta is evaluated with the prescribed T(rho) and n(rho) profiles and
    the chosen resistivity model.  The enclosed current fraction follows:

        I(rho)/Ip = integral_0^rho  j * 2*pi*a^2*kappa * rho' d_rho'
                   / integral_0^1  j * 2*pi*a^2*kappa * rho' d_rho'
                 = integral_0^rho  sigma(rho') * rho' d_rho'
                 / integral_0^1   sigma(rho') * rho' d_rho'

    where sigma = 1/eta and the a^2*kappa factor cancels.

    The cylindrical internal inductance is then:

        li = 2 * integral_0^1  [I(rho)/Ip]^2 / rho  d_rho

    Advantages over the analytic nu_J approach
    -------------------------------------------
    - Consistent with f_Reff (same eta models, same profile shapes).
    - Neoclassical corrections (Sauter, Redl) naturally included: trapped
      electrons reduce sigma near the edge, slightly flattening the effective
      current profile relative to pure Spitzer.
    - ln(Lambda) variation with rho accounted for.
    - No intermediate approximation j ~ T^(3/2) -> nu_J = 1.5*nu_T.

    Limitations
    -----------
    - Still assumes cylindrical geometry (ignores toroidal corrections ~a/R0).
    - Does not model inductive current diffusion during ramp-up (flat-top
      equilibrium assumed).  During ramp-up, li may be lower if the current
      diffusion timescale tau_R ~ μ0*a^2/eta exceeds the ramp duration.
    - Neoclassical models (Sauter, Redl) are calibrated for flat-top
      conditions; applying them to a ramp-up profile is an extrapolation.

    Parameters
    ----------
    nu_T : float
        Temperature profile peaking exponent (core region) [-].
    nu_n : float
        Density profile peaking exponent (core region) [-].
    Tbar : float
        Volume-averaged electron temperature [keV].
    nbar : float
        Volume-averaged electron density [1e20 m^-3].
    Z_eff : float
        Effective ionic charge [-].
    a : float
        Plasma minor radius [m].
    R0 : float
        Plasma major radius [m].
    q : float
        Safety factor (used by neoclassical resistivity models) [-].
    eta_model : str, optional
        Resistivity model: 'spitzer' | 'sauter' | 'redl' (default).
        'redl' is recommended for reactor-relevant conditions.
    n_pts : int, optional
        Number of radial points for trapezoidal integration (default 100).
        Convergence check: doubling n_pts should change li by < 0.1%.
    rho_ped : float, optional
        Normalised pedestal radius (1.0 = no pedestal, purely parabolic).
    n_ped_frac : float, optional
        n_ped / nbar.  Ignored when rho_ped = 1.0.
    T_ped_frac : float, optional
        T_ped / Tbar.  Ignored when rho_ped = 1.0.

    Returns
    -------
    li : float
        Normalized plasma internal inductance [-].
        Typical range: 0.5 (flat profile) to ~1.5 (very peaked profile).

    References
    ----------
    Freidberg, Plasma Physics and Fusion Energy, Cambridge 2007, ch. 3.
    Wesson, Tokamaks, 4th ed., Oxford 2011, ch. 3.
    Sauter et al., Phys. Plasmas 6, 2834 (1999).
    Redl et al., Phys. Plasmas 28, 022502 (2021).
    """
    from scipy.integrate import quad

    epsilon = a / R0

    # ── Radial conductivity profile sigma(rho) = 1/eta(rho) ─────────────────
    # Avoid the exact edge (rho=1) where T -> 0 and eta -> infinity.
    # Using rho_max = 0.98 is consistent with f_Reff.
    rho = np.linspace(0.0, 0.98, n_pts)
    sigma = np.empty(n_pts)

    for i, r in enumerate(rho):
        # Profile-aware evaluation: supports parabolic and pedestal models
        T_loc = max(float(f_Tprof(Tbar, nu_T, r, rho_ped, T_ped_frac)), 0.1)
        n_loc = float(f_nprof(nbar, nu_n, r, rho_ped, n_ped_frac))
        n_loc_m3 = n_loc * 1e20   # [1e20 m^-3] -> [m^-3]

        if eta_model == 'spitzer':
            eta_val = eta_spitzer(T_loc, n_loc_m3, Z_eff)
        elif eta_model == 'sauter':
            eta_val = eta_sauter(T_loc, n_loc_m3, Z_eff, epsilon, q, R0)
        elif eta_model == 'redl':
            eta_val = eta_redl(T_loc, n_loc_m3, Z_eff, epsilon, q, R0)
        else:
            raise ValueError(
                f"Unknown eta_model '{eta_model}'. "
                "Choose from: 'spitzer', 'sauter', 'redl'."
            )

        sigma[i] = 1.0 / max(eta_val, 1e-12)

    # ── Enclosed current fraction I(rho)/Ip ──────────────────────────────────
    # I(rho) / Ip = integral_0^rho sigma*rho' d_rho' / integral_0^1 sigma*rho' d_rho'
    # Trapezoidal rule on the rho grid (uniform spacing).
    integrand_sigma = sigma * rho
    I_cumulative    = np.cumsum(integrand_sigma) * (rho[1] - rho[0])
    I_total         = I_cumulative[-1]
    f_enclosed      = I_cumulative / I_total

    # ── Internal inductance li ────────────────────────────────────────────────
    # li = 2 * integral_0^1  f_enclosed(rho)^2 / rho  d_rho
    # Avoid the rho=0 singularity: f_enclosed ~ rho^2 near origin -> f^2/rho ~ rho^3 -> 0.
    rho_safe     = np.where(rho > 1e-8, rho, 1.0)
    li_integrand = np.where(rho > 1e-8, f_enclosed**2 / rho_safe, 0.0)
    li = 2.0 * np.trapezoid(li_integrand, rho)

    return li
    
#%% Magnetic flux calculation

def f_Psi_PI(R0, E_phi_BD=0.5, t_BD=0.5):
    """
    Estimate the magnetic flux required for plasma breakdown and initiation.

    Physical model: the flux swing needed to sustain a toroidal electric field
    E_phi_BD over a pre-ionization duration t_BD at major radius R0.

        Psi_PI = 2 * pi * R0 * E_phi_BD * t_BD

    Typical values:
        E_phi_BD ~ 0.3–0.5 V/m (Lloyd et al. 1991, Ejima et al. 1982)
        t_BD     ~ 0.3–0.5 s

    Cross-check: ITER (R0=6.2 m, E=0.5 V/m, t=0.5 s) -> ~9.7 Wb ≈ 10 Wb [OK]

    References
    ----------
    Lloyd et al., Plasma Phys. Control. Fusion 33(11), 1991.
    Ejima et al., Nucl. Fusion 22(10), 1982.

    Parameters
    ----------
    R0 : float
        Plasma major radius [m].
    E_phi_BD : float, optional
        Toroidal electric field threshold for breakdown [V/m]. Default: 0.5.
    t_BD : float, optional
        Pre-ionization / breakdown duration [s]. Default: 0.5.

    Returns
    -------
    Psi_PI : float
        Flux required for plasma initiation [Wb].
    """
    Psi_PI = 2.0 * np.pi * R0 * E_phi_BD * t_BD
    return Psi_PI

def f_Psi_PF(Ip, R0, C_PF=0.9):
    """
    Estimate the poloidal flux contribution from the PF coil system
    using an empirical scaling law analogous to the Ejima formula.

        Psi_PF = C_PF * μ0 * R0 * Ip

    This scaling is motivated by dimensional analysis and calibrated
    against ITER (C_PF ~ 0.98) and CFETR (C_PF ~ 0.82).
    The uncertainty on C_PF is estimated at ±30%.

    At 0D level, a reliable analytical estimate of Psi_PF is not
    achievable without knowledge of the PF coil layout. This empirical
    formula is the most honest approximation available.

    Cross-check:
        ITER  (R0=6.2 m,   Ip=15 MA):  Psi_PF = 110.9 Wb  [ref: ~115 Wb]
        CFETR (R0=7.2 m,   Ip=13 MA):  Psi_PF = 112.0 Wb  [ref:  ~97 Wb]
        JET   (R0=2.96 m,  Ip=4  MA):  Psi_PF =  14.1 Wb  [ref:  ~10 Wb]

    References
    ----------
    Ejima et al., Nucl. Fusion 22(10), 1982  (analogous formula for Psi_res).
    Kovari et al., Fusion Eng. Des. 89, 2014 (PROCESS: C_PF treated as input).

    Parameters
    ----------
    Ip    : float  Plasma current [A].
    R0    : float  Major radius [m].
    C_PF  : float  Empirical PF flux coefficient [-]. Default: 0.95.
                   Calibrated on ITER. Uncertainty: ±30%.

    Returns
    -------
    Psi_PF : float  Estimated PF flux contribution [Wb].
    """
    return C_PF * μ0 * R0 * Ip

def Magnetic_flux(Ip, I_Ohm, B_max_TF, a, b, c, R0, κ, nbar, Tbar,
                  Ce, Temps_Plateau, Li, Choice_Buck_Wedg, Gap,
                  Z_eff, q, nu_T, nu_n, eta_model,
                  E_phi_BD=0.5, t_BD=0.5,
                  C_PF=0.9,
                  rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0):
    """
    Calculate the four magnetic flux components for a tokamak plasma scenario.

    Flux budget decomposition
    -------------------------
    The total flux that the CS must provide is:

        Ψ_CS = Ψ_PI + Ψ_RampUp + Ψ_plateau - Ψ_PF

    where each term is detailed below.

    Parameters
    ----------
    Ip : float
        Plasma current [MA].
    I_Ohm : float
        Ohmic (inductive) plasma current fraction at flat-top [MA].
    B_max_TF : float
        Peak magnetic field at the inboard TF conductor [T].
        Used to derive B0 = B_max_TF * (1 - (a+b)/R0).
    a : float
        Plasma minor radius [m].
    b : float
        Radial thickness from plasma edge to TF coil inner face [m].
        Includes: first wall + breeding blanket + neutron shield +
        vacuum vessel + assembly gaps.
    c : float
        TF coil inboard leg radial thickness [m].
    R0 : float
        Plasma major radius [m].
    κ : float
        Plasma elongation (X-point value) [-].
        Used in the effective minor radius: a_eff = a * sqrt(κ).
    nbar : float
        Volume-averaged electron density [1e20 m^-3].
    Tbar : float
        Volume-averaged electron temperature [keV].
    Ce : float
        Ejima resistive startup coefficient [-].
        Resistive ramp-up flux: Ψ_res = Ce * μ0 * R0 * Ip.
        Typical values: ITER 0.45, EU-DEMO 0.30, CFETR 0.30.
        Reference: Ejima et al., Nucl. Fusion 22(10), 1982.
    Temps_Plateau : float
        Flat-top (burn) duration [s].
        Examples: ITER 600 s (10 min), EU-DEMO 7200 s (2 h),
                  CFETR 14400 s (4 h).
    Li : float
        Normalized plasma internal inductance [-].
        Recommended: compute via f_li(nu_T, nu_n, Tbar, nbar, Z_eff,
                                      a, R0, q, eta_model='redl')
        which integrates j(ρ) = 1/η(ρ) consistently with f_Reff.
        Typical range: 0.5 (flat profile) to ~1.5 (very peaked).
        Reference values: ITER ~0.80, EU-DEMO ~1.10, CFETR ~0.85.
    Choice_Buck_Wedg : str
        CS structural configuration: 'Bucking', 'Plug', or 'Wedging'.
        Determines the CS outer radius: R_CS_ext = R0 - a - b - c [-Gap].
    Gap : float
        Radial gap between TF inboard leg and CS outer face [m].
        Only used when Choice_Buck_Wedg = 'Wedging'.
    Z_eff : float
        Plasma effective ionic charge [-].
        Used in resistivity models (Spitzer, Sauter, Redl).
    q : float
        Safety factor at the 95% flux surface [-].
        Used by neoclassical resistivity models (Sauter, Redl) for the
        electron collisionality: ν*_e ∝ q*R0 / (ε^1.5 * T^2).
    nu_T : float
        Temperature profile peaking factor [-].
        Profile: T(ρ) = Tbar * (1 + nu_T) * (1 - ρ²)^nu_T.
        Typical: 0.8–1.5 for H-mode plasmas.
    nu_n : float
        Density profile peaking factor [-].
        Profile: n(ρ) = nbar * (1 + nu_n) * (1 - ρ²)^nu_n.
        Typical: 0.1–0.5 for H-mode plasmas.
    eta_model : str
        Plasma resistivity model for f_Reff and f_li:
        'spitzer'  — classical Spitzer-Härm with Z_eff and ln(Λ) correction.
        'sauter'   — neoclassical Sauter et al., Phys. Plasmas 6, 2834 (1999).
        'redl'     — neoclassical Redl et al., Phys. Plasmas 28, 022502 (2021).
                     Recommended: better accuracy at high collisionality.
    E_phi_BD : float, optional
        Toroidal electric field threshold for plasma breakdown [V/m].
        Default: 0.5 V/m.
        Reference: Lloyd et al., Plasma Phys. Control. Fusion 33(11), 1991.
    t_BD : float, optional
        Pre-ionization / breakdown duration [s]. Default: 0.5 s.
        Together: Ψ_PI = 2π * R0 * E_phi_BD * t_BD ≈ 10 Wb (ITER-scale).
    rho_ped : float, optional
        Normalised pedestal radius (1.0 = no pedestal, purely parabolic).
        Passed through to f_Reff, which uses it to evaluate the local
        resistivity via f_Tprof / f_nprof.
    n_ped_frac : float, optional
        n_ped / nbar.  Ignored when rho_ped = 1.0.
    T_ped_frac : float, optional
        T_ped / Tbar.  Ignored when rho_ped = 1.0.

    Returns
    -------
    ΨPI : float
        Plasma initiation (breakdown) flux [Wb].
        Ψ_PI = 2π * R0 * E_phi_BD * t_BD.
        Weakly depends on machine size; typically 8–15 Wb.
    ΨRampUp : float
        Total flux consumed during current ramp-up [Wb].
        Ψ_RampUp = Ψ_ind + Ψ_res
                 = μ0*R0*[ln(8R0/a_eff) + Li/2 - 2]*Ip
                   + Ce*μ0*R0*Ip.
    Ψplateau : float
        Flux consumed during flat-top operation [Wb].
        Ψ_plateau = R_eff * I_Ohm * Temps_Plateau,
        where R_eff is computed by f_Reff (numerical integration of
        j(ρ) = 1/η(ρ) over the plasma cross-section).
        Note: 0D R_eff tends to underestimate the true loop voltage by
        ~15–30% because current peaking in the hot core biases the
        conductance-weighted average. Treat as a lower bound.
    ΨPF : float
        Flux swing provided by the external PF coil system [Wb].
        Empirical scaling: Ψ_PF = C_PF * μ0 * R0 * Ip,
        analogous to the Ejima formula. Calibrated on ITER (C_PF ~ 0.95)
        and CFETR (C_PF ~ 0.82); default C_PF = 0.9.
        Uncertainty: ±30% (geometry-dependent).

    Notes
    -----
    Benchmark summary (ref vs D0FUS, eta_model='redl'):

        Machine    ΨPI      ΨRampUp   Ψplateau   ΨPF     ΨCS
        -------    ------   -------   --------   -----   -----
        ITER       10/~10   200/205   36/25*     115/105 137/134
        EU-DEMO    10/~10   359/375   304/333*   313/193 383/530
        CFETR      10/~11   223/208   250/295*   97/114  380/400
        (* Ψplateau underestimated due to 0D R_eff bias; see above)

    References
    ----------
    Lloyd et al., Plasma Phys. Control. Fusion 33(11), 1441 (1991).
        -> Breakdown flux (ΨPI).
    Mikkelsen, Nucl. Fusion 29(7), 1990 (1989).
        -> Plasma self-inductance formula (ΨRampUp).
    Freidberg, Plasma Physics and Fusion Energy, Cambridge (2007), ch. 11.
        -> Large-aspect-ratio inductance; cylindrical li.
    Wesson, Tokamaks, 4th ed., Oxford (2011), ch. 3.
        -> βp definition; li convention.
    Ejima et al., Nucl. Fusion 22(10), 1341 (1982).
        -> Resistive startup coefficient Ce (ΨRampUp).
    Sauter et al., Phys. Plasmas 6, 2834 (1999); Erratum 9, 5140 (2002).
        -> Neoclassical resistivity (Ψplateau).
    Redl et al., Phys. Plasmas 28, 022502 (2021).
        -> Improved neoclassical resistivity (Ψplateau, recommended).
    Shimada et al., Nucl. Fusion 47, S1 (2007).
        -> ITER reference flux budget.
    Kovari et al., Fusion Eng. Des. 89, 3054 (2014).
        -> PROCESS systems code; EU-DEMO flux budget (DEMO1 2017-03).
    Wan et al., Nucl. Fusion 57, 102009 (2017).
        -> CFETR reference flux budget.
    Cao et al., Nucl. Fusion 61, 046002 (2021).
        -> CFETR hybrid scenario ohmic fraction (f_ohm ~ 24%).
    """

    # --- Unit conversion ---
    Ip    = Ip    * 1e6   # [MA] -> [A]
    I_Ohm = I_Ohm * 1e6  # [MA] -> [A]

    # -----------------------------------------------------------------------
    # On-axis toroidal field (1/R scaling)
    # -----------------------------------------------------------------------
    B0 = B_max_TF * (1.0 - (a + b) / R0)

    # -----------------------------------------------------------------------
    # Geometrical and plasma physics quantities
    # -----------------------------------------------------------------------
    # Approximate poloidal perimeter of the last closed flux surface [m]
    L  = np.pi * np.sqrt(2.0 * (a**2 + (κ * a)**2))

    # Poloidal beta (Wesson, Tokamaks 4th ed., eq. 3.9.3)
    βp = (4.0 / μ0) * L**2 * (nbar * 1e20) * (E_ELEM * 1e3 * Tbar) / Ip**2

    # -----------------------------------------------------------------------
    # CS outer radius (geometry-dependent)
    # -----------------------------------------------------------------------
    if Choice_Buck_Wedg in ('Bucking', 'Plug'):
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        raise ValueError(
            f"Choice_Buck_Wedg must be 'Bucking', 'Plug', or 'Wedging'. "
            f"Got: '{Choice_Buck_Wedg}'."
        )

    # =======================================================================
    # FLUX COMPONENTS
    # =======================================================================

    # -----------------------------------------------------------------------
    # 1. Plasma initiation flux (ΨPI)
    #    Ψ_PI = 2π * R0 * E_phi_BD * t_BD
    #    Reference: Lloyd et al., PPCF 33(11), 1991.
    #    Cross-check: ITER -> ~9.7 Wb ≈ 10 Wb [OK]
    # -----------------------------------------------------------------------
    ΨPI = f_Psi_PI(R0, E_phi_BD=E_phi_BD, t_BD=t_BD)

    # -----------------------------------------------------------------------
    # 2. Ramp-up flux (ΨRampUp = Ψind + Ψres)
    #    Elongation-corrected minor radius: a_eff = a * sqrt(κ)
    #    Plasma self-inductance (large-aspect-ratio Neumann formula):
    #        Lp = μ0*R0 * [ln(8*R0/a_eff) + Li/2 - 2]
    #    References: Freidberg (2007) ch.11; Wesson (2011) §3.9;
    #                Mikkelsen NF 29(7), 1989.
    #    Resistive term (Ejima formula):
    #        Ψ_res = Ce * μ0 * R0 * Ip
    #    Reference: Ejima et al., Nucl. Fusion 22(10), 1982.
    # -----------------------------------------------------------------------
    a_eff = a * math.sqrt(κ)   # Elongation-corrected effective minor radius [m]
    if a_eff <= 0.0 or R0 <= 0.0:
        return (np.nan, np.nan, np.nan, np.nan)
    Lp      = μ0 * R0 * (math.log(8.0 * R0 / a_eff) + Li / 2.0 - 2.0)
    Ψind    = Lp * Ip
    Ψres    = Ce * μ0 * R0 * Ip
    ΨRampUp = Ψind + Ψres

    # -----------------------------------------------------------------------
    # 3. Flat-top (plateau) flux (Ψplateau)
    #    Ψ_plateau = R_eff * I_Ohm * Temps_Plateau
    #    R_eff computed by f_Reff: numerical integration of j(ρ) = 1/η(ρ)
    #    over the plasma cross-section (consistent with f_li).
    #    Known limitation: R_eff is biased ~15–30% low because the
    #    hot-core current peaking dominates the conductance-weighted average.
    # -----------------------------------------------------------------------
    R_eff    = f_Reff(a, κ, R0, Tbar, nbar, Z_eff, q, nu_T, nu_n, eta_model,
                      rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    Vloop    = R_eff * I_Ohm
    Ψplateau = Vloop * Temps_Plateau

    # -----------------------------------------------------------------------
    # 4. Flux provided by PF coil system (ΨPF)
    #    Empirical scaling: Ψ_PF = C_PF * μ0 * R0 * Ip
    #    Analogous to Ejima formula. C_PF ≈ 0.95 (ITER), 0.82 (CFETR).
    #    Default C_PF = 0.9 (conservative mid-range estimate).
    #    Uncertainty: ±30%.
    # -----------------------------------------------------------------------
    ΨPF = f_Psi_PF(Ip, R0, C_PF)

    return (ΨPI, ΨRampUp, Ψplateau, ΨPF)

#%% Flux generation benchmark — multi-machine
# ============================================================================
# Benchmark of Magnetic_flux() against published flux budgets for:
#   - ITER       (Shimada et al. Nucl. Fusion 47 S1 2007; ITER EDA NF 39 1999)
#   - EU-DEMO    (PROCESS v1.0.10, DEMO1 Reference Design 2017-03, Kovari FED 89 2014)
#   - CFETR      (Wan et al. Nucl. Fusion 57 102009 2017;
#                 Liu et al. Chinese Physics B 29(2) 2020)
#
# Reference values
# ----------------
# ITER :   ΨPI~10, ΨRampUp~200, Ψplateau~36, ΨPF~115, ΨCS~137 Wb
# EU-DEMO: ΨPI~10, ΨRampUp~285, Ψplateau~304, ΨPF~63(?), ΨCS~600 Wb
# CFETR :  ΨPI~10, ΨRampUp~223, Ψplateau~250, ΨPF~97, ΨCS~380 Wb
#          (total V·s initiation→flat-top ~233 Wb from TSC simulation;
#           plateau target ΔΦohm ≤ 250 Wb for 4 h hybrid scenario)
# ============================================================================

if __name__ == "__main__":

    # ── Machine parameter table ───────────────────────────────────────────────
    # Each entry holds all inputs to Magnetic_flux() plus published reference
    # flux values for benchmarking.
    #
    # Fields
    # ------
    # Geometry      : a, b, c, R0  [m]
    # Plasma        : Ip [MA], κ, nbar [10^20 m^-3], Tbar [keV]
    # Magnetics     : B_max_TF [T], Li, Z_eff, q95, nu_T, nu_n
    # Operation     : I_Ohm [MA], Ce (Ejima), Temps_Plateau [s], configuration
    # Reference flux: ref_PI, ref_RampUp, ref_plateau, ref_PF, ref_CS  [Wb]
    #                 (None = no reliable published value available)
    #
    # Sources
    # -------
    # ITER    : Shimada NF 47 S1 (2007); ITER EDA NF 39 (1999)
    # EU-DEMO : Federici FED 136 (2018); PROCESS output (Kovari FED 89 2014)
    # CFETR   : Wan NF 57 102009 (2017); Liu Chinese Phys. B 29(2) (2020);
    #           Zhuang NF 59 112010 (2019)

    MACHINES = {

        "ITER": dict(
            # Geometry
            a=2.00, b=1.25, c=0.90, R0=6.2,
            # Plasma
            Ip=15.0, κ=1.70, nbar=1.0, Tbar=8.0,
            # Magnetics
            B_max_TF=13.0, Li=0.80, Z_eff=1.7, q=3.0, nu_T=1.0, nu_n=0.1,
            # Operation
            I_Ohm=3.0, Ce=0.45, Temps_Plateau=10 * 60,
            configuration="Wedging",
            # Published reference flux values [Wb]
            # Source: ITER EDA NF 39 (1999); Shimada NF 47 S1 (2007)
            ref_PI=10, ref_RampUp=200, ref_plateau=36, ref_PF=115, ref_CS=137,
            ref_li=0.80,
        ),

        "EU-DEMO": dict(
            # Geometry
            # Source: PROCESS v1.0.10, EU DEMO1 baseline 2017-03
            #         (DEMO1_Reference_Design_2017_March_EU_2NDSKT_v1_0.DAT)
            # Optimised: R0=8.938 m, a=2.883 m; TF bore=2.365 m, CS=0.802 m
            a=2.883, b=1.40, c=0.962, R0=8.938,
            # Plasma
            # Ip=19.075 MA; kappa=1.848 (X-point), kappa95=1.650
            # nbar=0.791e20 m^-3 (volume-averaged); Tbar=12.818 keV
            Ip=19.075, κ=1.848, nbar=0.791, Tbar=12.818,
            # Magnetics
            # B_max_TF=10.61 T (Amperes law); bmaxtfrp=11.05 T with ripple
            # Li (rli in PROCESS) = 1.096  [PROCESS cylindrical li convention]
            B_max_TF=10.61, Li=1.096, Z_eff=2.179, q=3.0, nu_T=1.0, nu_n=0.1,
            # Operation
            # Ce (Ejima gamma) = 0.300; verified: Ce*μ0*R0*Ip = 64.27 Wb
            #                             matches PROCESS vsres = 64.28 Wb
            # tburn = 7200 s (2 h flat-top)
            I_Ohm=7.0, Ce=0.300, Temps_Plateau=7200,
            configuration="Wedging",
            # Reference flux values [Wb] -- directly from PROCESS output
            # vsres  (start-up resistive, Ejima) = 64.28 Wb  -> ref_PI
            # vsind  (inductive, ramp-up)        = 295.0  Wb -> ref_RampUp
            # vsbrn  (flat-top resistive)        = 304.3  Wb -> ref_plateau
            # PF coils total V-s (start-up+burn) = 312.8  Wb -> ref_PF
            # CS coil  total V-s                 = 382.3  Wb -> ref_CS
            # Total capability 695.1 Wb; margin = +31.5 Wb over requirement
            #
            # NOTE: ref_PF includes BOTH equilibrium-field flux AND CS stray-
            # field balancing swings of PF coils, hence C_PF = 312.8/214.2
            # = 1.46, significantly higher than ITER (0.98). This reflects the
            # large CS-compensation contribution in the EU-DEMO PF design.
            # The "63 Wb" figure previously used was erroneous (unrelated source).
            # ref_PI  = breakdown flux (~10 Wb), not included in PROCESS vsres
            # ref_RampUp = vsind(295.0) + vsres_ramp(64.3) = 359.3 Wb
            #   (PROCESS vsres=64.28 Wb is the TOTAL resistive ramp term,
            #    not a pre-ionisation flux; D0FUS ΨPI is the breakdown only)
            ref_PI=10.0, ref_RampUp=359.3, ref_plateau=304.3,
            ref_PF=312.8, ref_CS=382.3,
            ref_li=1.096,  # PROCESS rli = li(3), cylindrical convention
        ),

        "CFETR": dict(
            # Geometry
            # Source: Wan NF 57 102009 (2017); Zhuang NF 59 112010 (2019)
            a=2.20, b=1.40, c=1.20, R0=7.2,
            # Plasma
            # Source: Liu Chinese Phys. B 29(2) (2020) Table 1
            # Ip=14 MA (2019+ baseline); κ=2.0; nbar~1.3×10^20; Tbar~12 keV
            Ip=14.0, κ=2.00, nbar=1.3, Tbar=12.0,
            # Magnetics
            # B_max_TF=14.4 T (conductor design, Wan 2022); Li estimated
            B_max_TF=14.4, Li=0.85, Z_eff=2.0, q=3.5, nu_T=1.0, nu_n=0.1,
            # Operation
            # Ce: effective Ejima for CFETR ramp-up ~0.3 (with LHCD assist,
            #     Liu 2020 gives CE time-varying; 0.3 is a reasonable 0D estimate)
            # Temps_Plateau: 4 h hybrid scenario (Cao NF 61 046002 2021)
            # I_Ohm: ohmic fraction at flat-top from Cao NF 61 046002 (2021)
            #   METIS hybrid scenario: bootstrap~40%, LHCD~40%, ohmic~20%
            #   -> I_ohm = 0.2 * 14 MA = 2.8 MA
            I_Ohm=2.8, Ce=0.30, Temps_Plateau=4 * 3600,
            configuration="Wedging",
            # Published reference flux values [Wb]
            # ΨPI+ΨRampUp ~ 233 Wb total (Liu 2020 TSC); split estimated
            # Ψplateau ≤ 250 Wb for 4 h (Cao NF 61 046002 2021)
            # ΨPF ~ 97 Wb, ΨCS ~ 380 Wb (Wan NF 57 102009 2017)
            ref_PI=10, ref_RampUp=223, ref_plateau=250, ref_PF=97, ref_CS=380,
            ref_li=0.85,   # estimated
        ),
    }

    # ── Run Magnetic_flux for each machine and collect results ──────────────
    results = {}
    for machine_name, p in MACHINES.items():
        ΨPI, ΨRampUp, Ψplateau, ΨPF = Magnetic_flux(
            p["Ip"],    p["I_Ohm"],  p["B_max_TF"],
            p["a"],     p["b"],      p["c"],          p["R0"],
            p["κ"],     p["nbar"],   p["Tbar"],
            p["Ce"],    p["Temps_Plateau"],            p["Li"],
            p["configuration"],      0.1,
            p["Z_eff"], p["q"],      p["nu_T"],        p["nu_n"],
            eta_model='redl'
        )
        results[machine_name] = dict(
            ΨPI=ΨPI, ΨRampUp=ΨRampUp, Ψplateau=Ψplateau, ΨPF=ΨPF,
            ΨCS=ΨPI + ΨRampUp + Ψplateau - ΨPF,
            Ψtot=ΨPI + ΨRampUp + Ψplateau,
        )

    # ── Compute li for each machine (new physics-consistent method) ─────────
    # Uses f_li(nu_T, nu_n, Tbar, nbar, Z_eff, a, R0, q, eta_model) which
    # integrates j(rho) = 1/eta(rho) directly, consistent with f_Reff.
    for machine_name, p in MACHINES.items():
        results[machine_name]["li_calc"] = f_li(
            p["nu_T"], p["nu_n"], p["Tbar"], p["nbar"],
            p["Z_eff"], p["a"], p["R0"], p["q"],
            eta_model='redl'
        )

    # ── Single summary table ─────────────────────────────────────────────────
    # Format: "ref  calc" columns per machine (no error display)
    machines = list(MACHINES.keys())

    W  = 22   # label column width
    CW = 20   # width per machine pair (ref + calc)

    def cell(ref, val):
        """Format one cell as 'ref   calc'."""
        return f"  {ref:>7.1f}  {val:>7.1f}"

    sep        = "  " + "-" * W + ("  " + "-" * (CW - 2)) * len(machines)
    hdr_names  = f"  {'':{W}}" + "".join(f"  {m:>{CW-2}}" for m in machines)
    hdr_cols   = f"  {'':{W}}" + "".join(f"  {'ref':>7}  {'calc':>7}" for _ in machines)
    total_line = "  " + "=" * (W + CW * len(machines))

    print()
    print("  Magnetic Flux Benchmark  [Wb]")
    print(total_line)
    print(hdr_names)
    print(hdr_cols)
    print(sep)

    # li first — it feeds into Magnetic_flux via Li parameter
    print(f"  {'Internal inductance li':{W}}" + "".join(
        cell(MACHINES[m]["ref_li"], results[m]["li_calc"]) for m in machines
    ))
    print(sep)

    FLUX_ROWS = [
        ("Ψ initiation  (ΨPI)",     "ΨPI",     "ref_PI"),
        ("Ψ ramp-up     (ΨRampUp)", "ΨRampUp", "ref_RampUp"),
        ("Ψ flat-top    (Ψplateau)","Ψplateau","ref_plateau"),
    ]
    for label, key, ref_key in FLUX_ROWS:
        print(f"  {label:{W}}" + "".join(
            cell(MACHINES[m][ref_key], results[m][key]) for m in machines
        ))

    print(sep)

    ref_totals = {m: MACHINES[m]["ref_PI"] + MACHINES[m]["ref_RampUp"] + MACHINES[m]["ref_plateau"]
                  for m in machines}
    print(f"  {'Total requirement':{W}}" + "".join(
        cell(ref_totals[m], results[m]["Ψtot"]) for m in machines
    ))
    print(f"  {'Ψ PF coils    (ΨPF)':{W}}" + "".join(
        cell(MACHINES[m]["ref_PF"], results[m]["ΨPF"]) for m in machines
    ))
    print(f"  {'Ψ CS needed   (ΨCS)':{W}}" + "".join(
        cell(MACHINES[m]["ref_CS"], results[m]["ΨCS"]) for m in machines
    ))
    print(total_line)
        
#%% ===========================================================================
# CS ACADEMIC MODEL
# =============================================================================

def f_CS_ACAD(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
              config: GlobalConfig):
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
        Maximum CS magnetic field (T)
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier for Jc calculation
    Jc_manual : float
        If needed, manual current density
    T_Helium : float
        Helium coolant temperature (K)
    Choice_Buck_Wedg : str
        Mechanical support configuration: 'Bucking', 'Wedging', or 'Plug'
    κ : float
        Plasma elongation
    
    Returns
    -------
    tuple of float
        (d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS)
        
        Returns (nan, nan, nan, nan, nan, nan, nan) if no physically valid solution exists.

    References
    ----------
    [Auclair et al. NF 2026]

    config : GlobalConfig
        Global design configuration. Used to access: Gap, I_cond, V_max,
        f_He, f_In, T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn, Marge_T_NbTi,
        Marge_T_REBCO, Eps, Tet, n_CS.
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    
    Gap         = config.Gap
    I_cond      = config.I_cond
    V_max       = config.V_max
    f_He        = config.f_He
    f_In        = config.f_In
    T_hotspot   = config.T_hotspot
    RRR         = config.RRR
    Marge_T_He      = config.Marge_T_He
    Marge_T_Nb3Sn   = config.Marge_T_Nb3Sn
    Marge_T_NbTi    = config.Marge_T_NbTi
    Marge_T_REBCO   = config.Marge_T_REBCO
    Eps         = config.Eps
    Tet         = config.Tet
    n_CS        = config.n_CS
    debug = False
    Tol_CS = 1e-3

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
    # STEP 1: Determine B_CS, J_max_CS, and d_SU with self-consistent
    #         iteration on RCS_int (fixes the R_int = 0.5*R_ext assumption)
    # ------------------------------------------------------------------
    
    # Thin cylinder approximation for initial estimate of B_CS
    B_CS_thin = ΨCS / (np.pi * RCS_ext**2)
    
    if debug:
        print(f"[STEP 1] Thin cylinder B_CS estimate: {B_CS_thin:.2f} T")
    
    H_CS = 2 * (κ * a + b + 1)
    
    # --- Analytical initial estimate for RCS_int ---
    # Thin-shell: d ≈ Ψ / (2π R_ext B_CS)  →  R_int ≈ R_ext - d
    d_est = ΨCS / (2 * np.pi * RCS_ext * B_CS_thin)
    RCS_int = max(0.1 * RCS_ext, RCS_ext - d_est)
    
    # --- Iterative convergence loop ---
    max_iter_CS = 15
    converged = False
    
    for i_iter in range(max_iter_CS):
        # Magnetic energy with current R_int estimate
        E_mag_CS = calculate_E_mag_CS(B_CS_thin, RCS_int, RCS_ext, H_CS)
        
        # Cable current density (cached → cheap on repeated calls)
        result_J = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS_thin, T_op=T_Helium,
            E_mag=E_mag_CS, I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS,
            tau_h=tau_h, f_He=f_He, f_In=f_In, T_hotspot=T_hotspot,
            RRR=RRR, Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO,
            Eps=Eps, Tet=Tet,
            J_wost_Manual=Jc_manual if Supra_choice_CS == 'Manual' else None)
        J_max_CS = result_J['J_wost']
        
        if J_max_CS < Tol_CS:
            if debug:
                print(f"[STEP 1] iter {i_iter}: Non-positive J_max_CS: {J_max_CS:.2e} A/m²")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Thick solenoid flux inversion → new R_int
        RCS_sep_cubed = RCS_ext**3 - (3 * ΨCS) / (2 * np.pi * μ0 * J_max_CS)
        
        if RCS_sep_cubed <= 0:
            if debug:
                print(f"[STEP 1] iter {i_iter}: Invalid RCS_sep³: {RCS_sep_cubed:.2e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        RCS_int_new = RCS_sep_cubed**(1/3)
        
        if debug:
            print(f"[STEP 1] iter {i_iter}: RCS_int = {RCS_int:.4f} → {RCS_int_new:.4f} m  "
                  f"(ΔR = {abs(RCS_int_new - RCS_int)*1e3:.2f} mm, "
                  f"J_wost = {J_max_CS/1e6:.1f} MA/m²)")
        
        # Convergence check
        if abs(RCS_int_new - RCS_int) < Tol_CS:
            converged = True
            RCS_int = RCS_int_new
            break
        
        RCS_int = RCS_int_new
    
    if not converged and debug:
        print(f"[STEP 1] WARNING: R_int loop did not converge after {max_iter_CS} iterations "
              f"(last ΔR = {abs(RCS_int_new - RCS_int)*1e3:.2f} mm)")
    
    # --- Final values from converged R_int ---
    RCS_sep = RCS_int
    d_SU = RCS_ext - RCS_sep
    
    if d_SU <= Tol_CS or d_SU >= RCS_ext - Tol_CS:
        if debug:
            print(f"[STEP 1] Invalid d_SU: {d_SU:.4f} m")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Recompute B_CS with thick solenoid formula (consistent with converged geometry)
    B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_sep + RCS_sep**2))
    
    if B_CS > B_max_CS:
        if debug:
            print(f"[STEP 1] B_CS = {B_CS:.2f} T exceeds limit {B_max_CS:.2f} T")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # ------------------------------------------------------------------
    # STEP 2: Find steel thickness d_SS using brentq
    # ------------------------------------------------------------------
    
    P_CS = B_CS**2 / (2.0 * μ0)
    P_TF = B_max_TF**2 / (2.0 * μ0)
    
    # Special case: Plug with TF dominant
    if Choice_Buck_Wedg == 'Plug' and abs(P_CS - P_TF) <= abs(P_TF):
        d = d_SU
        return d, 0, 0, 0, 0, B_CS, J_max_CS
    
    def stress_residual(d_SS):
        if d_SS <= Tol_CS:
            return np.nan
        d_total = d_SS + d_SU
        if d_total >= RCS_ext - Tol_CS:
            return np.nan
        
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = abs(np.nanmax([P_TF, abs(P_CS - P_TF)])) * RCS_sep / d_SS
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = abs(P_CS * RCS_sep / d_SS)
        elif Choice_Buck_Wedg == 'Plug':
            if abs(P_CS - P_TF) > abs(P_TF):
                Sigma_CS = abs(abs(P_CS - P_TF) * RCS_sep / d_SS)
            else:
                return np.nan
        else:
            return np.nan
        
        if Sigma_CS < Tol_CS:
            return np.nan
        return Sigma_CS - σ_CS
    
    # ------------------------------------------------------------------
    # STEP 2 — Direct two-point brentq (no linear scan required)
    # ------------------------------------------------------------------
    # For all three mechanical configurations (Wedging, Bucking, Plug),
    # stress_residual(d_SS) = K / d_SS - σ_CS, where K is a positive
    # constant (K = P_CS * RCS_sep or similar). This function is strictly
    # monotone decreasing on the valid domain (Tol_CS, d_SS_max), so:
    #   • At most one root exists.
    #   • Evaluating at both endpoints is sufficient to determine whether
    #     a root is present; no scan is needed.
    #   • If residual(d_SS_min) > 0 and residual(d_SS_max) < 0 → root
    #     exists; direct brentq.
    #   • Otherwise → design is geometrically infeasible (steel thickness
    #     required to meet stress limit exceeds available radial space).
    # Total cost: ~2 (bound checks) + ~15 (brentq) = ~17 evaluations,
    # compared to 200 for the previous linear scan.
    # ------------------------------------------------------------------

    d_SS_min = Tol_CS
    d_SS_max = RCS_ext - d_SU - Tol_CS

    if d_SS_max <= d_SS_min:
        if debug:
            print("[STEP 2] No space left for steel: d_SS_max <= d_SS_min.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # stress_residual returns nan when d_SS <= Tol_CS (guard) or when
    # d_SS + d_SU >= RCS_ext - Tol_CS (guard). Evaluating at the exact
    # bound values would therefore return nan on both sides, incorrectly
    # indicating an infeasible design. A small inward offset eps_SS avoids
    # both guards while keeping the bracket physically valid.
    eps_SS   = Tol_CS * 1e-3   # 1 μm for Tol_CS = 1 mm; negligible vs solution
    d_lo_ss  = d_SS_min + eps_SS
    d_hi_ss  = d_SS_max - eps_SS

    if d_lo_ss >= d_hi_ss:
        if debug:
            print("[STEP 2] Effective bracket collapsed after eps offset.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    y_lo = stress_residual(d_lo_ss)
    y_hi = stress_residual(d_hi_ss)

    if debug:
        print(f"[STEP 2] stress_residual: "
              f"f({d_lo_ss:.6f}) = {y_lo:.3e},  "
              f"f({d_hi_ss:.4f}) = {y_hi:.3e}")

    if not (np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo * y_hi < 0):
        # No sign change: stress constraint cannot be met in available space.
        if debug:
            print("[STEP 2] No root in valid domain — design infeasible.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        d_SS = brentq(stress_residual, d_lo_ss, d_hi_ss, xtol=1e-9)
    except ValueError:
        if debug:
            print("[STEP 2] brentq failed on [d_lo_ss, d_hi_ss].")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # ------------------------------------------------------------------
    # STEP 3: Compute final outputs
    # ------------------------------------------------------------------
    
    d_total = d_SS + d_SU
    alpha = d_SU / d_total
    
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
    
    if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    d = d_total
    Steel_fraction = 1 - alpha
    σ_z = 0
    
    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

#%% ===========================================================================
# CS D0FUS MODEL
# =============================================================================


def f_sigma_z_CS_axial(J_smear, R_i, R_e, h):
    """
    Analytical axial (compressive) smeared stress at the CS midplane,
    induced by fringe-field Lorentz forces.

    Derivation: integrate the axial Lorentz body force J_smear x B_r from
    the free end (z=h, sigma_z=0) to the midplane (z=0).
    B_r is approximated from div.B=0 in the bore: B_r ~ -(r/2)*dBz/dz.

    Formula (Auclair 2026):
        sigma_z = -(mu0 * J_smear^2 * h * R_i / 2) * [L(h) - L(2h)]
        L(zeta) = ln((R_e + sqrt(R_e^2+zeta^2)) / (R_i + sqrt(R_i^2+zeta^2)))

    To convert smeared stress to steel stress:
        sigma_z_steel = sigma_z_smear / gamma
    where gamma = gamma_func(alpha, n_CS), n_CS being the conductor shape
    factor (1 = square jacket, 0 = optimal).

    Parameters
    ----------
    J_smear : float
        Smeared (homogenised) current density over the full CS cross-section
        [A/m^2]. Pass J_wost * alpha, where:
          alpha  = B_CS / (mu0 * J_wost * (R_e - R_i))  [-]
          J_wost = engineering current density without steel jacket [A/m^2]
        J_smear is the macroscopic J entering Maxwell: curl(B) = mu0 * J_smear.
    R_i : float
        CS inner winding radius [m]
    R_e : float
        CS outer winding radius [m]
    h : float
        CS half-height [m]  (h = H_CS / 2)

    Returns
    -------
    sigma_z : float
        Smeared axial stress at the midplane [Pa]. Negative = compressive.

    References
    ----------
    Auclair T. (2026), "Analytical axial stress at CS midplane", internal note.
    """
    def calL(zeta):
        # Logarithmic geometry factor for the on-axis field of a thick solenoid
        return np.log((R_e + np.sqrt(R_e**2 + zeta**2)) /
                      (R_i + np.sqrt(R_i**2 + zeta**2)))

    delta_L = calL(h) - calL(2.0 * h)   # > 0 since calL decreases with zeta

    sigma_z = -(μ0 * J_smear**2 * h * R_i / 2.0) * delta_L
    return float(sigma_z)


def f_CS_D0FUS(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
              config: GlobalConfig):
    """
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
        Cumulative radial build (m)
    c : float
        TF coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T)
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier
    Jc_manual : float
        Manual current density if needed
    T_Helium : float
        Helium coolant temperature (K)
    Choice_Buck_Wedg : str
        Mechanical configuration: 'Bucking', 'Wedging', or 'Plug'
    κ : float
        Plasma elongation
        
    Returns
    -------
    tuple of float
        (d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS)
        
        Returns (nan, nan, nan, nan, nan, nan, nan) if no solution exists.

    References
    ----------
    [Auclair et al. NF 2026]
    
    config : GlobalConfig
        Global design configuration. Used to access: Gap, I_cond, V_max,
        f_He, f_In, T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn, Marge_T_NbTi,
        Marge_T_REBCO, Eps, Tet, n_CS.
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    Gap         = config.Gap
    I_cond      = config.I_cond
    V_max       = config.V_max
    f_He        = config.f_He
    f_In        = config.f_In
    T_hotspot   = config.T_hotspot
    RRR         = config.RRR
    Marge_T_He      = config.Marge_T_He
    Marge_T_Nb3Sn   = config.Marge_T_Nb3Sn
    Marge_T_NbTi    = config.Marge_T_NbTi
    Marge_T_REBCO   = config.Marge_T_REBCO
    Eps         = config.Eps
    Tet         = config.Tet
    n_CS        = config.n_CS
    debug = False
    Tol_CS = 1e-3

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

    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF

    # ------------------------------------------------------------------
    # Main solving function (uses CACHED cable current density)
    # ------------------------------------------------------------------

    def d_to_solve(d):
        if d < Tol_CS or d > RCS_ext - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        RCS_int = RCS_ext - d
        B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        H_CS = 2 * (κ * a + b + 1)
        E_mag_CS = calculate_E_mag_CS(B_CS, RCS_int, RCS_ext, H_CS)
        
        # STRATEGY C: Use cached cable current density
        result_J = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS, T_op=T_Helium, E_mag=E_mag_CS,
            I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS, tau_h=tau_h, f_He=f_He, f_In=f_In,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=Jc_manual if Supra_choice_CS == 'Manual' else None)
        J_max_CS = result_J['J_wost']
        
        if J_max_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        alpha = B_CS / (μ0 * J_max_CS * d)
        
        if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if B_CS < Tol_CS or B_CS > B_max_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Compute principal stresses and apply 3D Tresca criterion.
        # Hoop and radial stresses in the structural steel are scaled by 1/(1-alpha).
        P_CS = B_CS**2 / (2.0 * μ0)
        P_TF = B_max_TF**2 / (2.0 * μ0) * (R0 - a - b) / RCS_ext
        denom_stress = RCS_ext**2 - RCS_int**2

        if abs(denom_stress) < 1e-30:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        def gamma_func(alpha_val, n_val):
            """
            Effective steel area fraction for transverse (axial-face) loading.
            Identical to the TF gamma function; derived from the square conductor
            geometry with n_val turns sharing the inter-turn load.
            """
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

        gamma_val = gamma_func(alpha, n_CS)
        if np.isnan(gamma_val):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Axial smeared stress at the midplane (compressive, < 0).
        # Controlled by config.cs_axial_stress (default False = disabled).
        # When disabled, sigma_z = 0 in all Tresca evaluations.
        _cs_axial_on = getattr(config, 'cs_axial_stress', False)
        if _cs_axial_on:
            J_smear   = J_max_CS * alpha
            σ_z_smear = f_sigma_z_CS_axial(J_smear, RCS_int, RCS_ext, H_CS / 2.0)
            σ_z_steel = σ_z_smear / gamma_val                       # Peak steel axial stress [Pa]
        else:
            σ_z_smear = 0.0
            σ_z_steel = 0.0

        def tresca(s_t, s_r, s_z):
            """Tresca equivalent stress: max principal stress difference [Pa]."""
            return max(abs(s_t - s_r), abs(s_r - s_z), abs(s_t - s_z))

        if Choice_Buck_Wedg == 'Bucking':
            # Light case (CS energized): tensile hoop − TF back-pressure, non-zero σ_z.
            # Strong case (J = 0, TF only): compressive hoop, σ_z = 0.
            sigma_theta_light  = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) -
                                  (2.0 * P_TF * RCS_ext**2 / denom_stress)) / (1.0 - alpha)
            sigma_theta_strong = -(2.0 * P_TF * RCS_ext**2 / denom_stress) / (1.0 - alpha)

            tresca_light  = tresca(sigma_theta_light,  0.0, σ_z_steel)
            tresca_strong = tresca(sigma_theta_strong, 0.0, 0.0)

            Sigma_CS = max(tresca_light, tresca_strong)
            σ_theta  = sigma_theta_light
            σ_r      = 0.0
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Wedging':
            sigma_theta = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress)
                           / (1.0 - alpha))
            Sigma_CS = tresca(sigma_theta, 0.0, σ_z_steel)
            σ_theta  = sigma_theta
            σ_r      = 0.0
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Plug':
            if abs(P_CS - P_TF) > abs(P_TF):
                # Bucking-dominated: two-case Tresca as above.
                sigma_theta_light  = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) -
                                      (2.0 * P_TF * RCS_ext**2 / denom_stress)) / (1.0 - alpha)
                sigma_theta_strong = -(2.0 * P_TF * RCS_ext**2 / denom_stress) / (1.0 - alpha)
                tresca_light  = tresca(sigma_theta_light,  0.0, σ_z_steel)
                tresca_strong = tresca(sigma_theta_strong, 0.0, 0.0)
                Sigma_CS = max(tresca_light, tresca_strong)
                σ_theta  = sigma_theta_light
                σ_r      = 0.0
                σ_z      = σ_z_steel
            elif abs(P_CS - P_TF) <= abs(P_TF):
                # Plug-dominated: J → 0, dominant stress is radial, σ_z = 0.
                sigma_r_signed = -(abs(P_TF) / gamma_val)
                Sigma_CS = tresca(0.0, sigma_r_signed, 0.0)
                σ_r      = sigma_r_signed
                σ_theta  = 0.0
                σ_z      = 0.0
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if Sigma_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        Steel_fraction = 1 - alpha

        return (float(Sigma_CS), float(σ_z), float(σ_theta), float(σ_r),
                float(Steel_fraction), float(B_CS), float(J_max_CS))
    
    # ------------------------------------------------------------------
    # Root function
    # ------------------------------------------------------------------
    
    def f_sigma_diff(d):
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d)
        if not np.isfinite(Sigma_CS):
            return np.nan
        return Sigma_CS - σ_CS
    
    # ------------------------------------------------------------------
    # Root-finding — monotone-aware adaptive solver
    # ------------------------------------------------------------------
    # f_sigma_diff(d) is strictly monotone decreasing on its valid domain
    # [d_min_valid, RCS_ext - Tol_CS], so at most one root exists.
    #
    # Physical basis: increasing d → lower B_CS → higher J_max (SC at
    # lower field) → lower alpha → lower Sigma_CS (see analysis notes).
    #
    # Two-pass adaptive strategy:
    #
    #   Pass 1 — coarse log-spaced probe (N_PROBE_1 = 15 pts, full range):
    #     Detects the nan→finite transition and/or a sign-change bracket.
    #     Log spacing gives simultaneous resolution near d→0 (compact HTS,
    #     d* ~ mm) and near d→RCS_ext (large LTS, d* ~ metres).
    #
    #   Pass 2 — adaptive refinement (triggered only if Pass 1 fails):
    #     a. Binary search on nan/finite boundary → pins d_min_valid (~20).
    #     b. Second log-spaced probe on [d_min_valid, RCS_ext - Tol_CS]
    #        with N_PROBE_2 = 20 pts → catches extreme cases where the root
    #        sits in a very narrow window just above d_min_valid (ultra-thin
    #        HTS CS) or just below d_hi (very large LTS CS, far from a
    #        solid cylinder). Direct bracket on the whole range gives brentq
    #        too wide an interval in these cases; the finer probe narrows it.
    #
    #   Pass 3 — brentq on the bracketed interval (~15 calls).
    #
    # Worst-case call budget: 15 + 20 + 20 + 15 = ~70 evaluations.
    # Typical (Pass 1 succeeds): ~30 evaluations.
    # ------------------------------------------------------------------

    N_PROBE_1 = 15   # Coarse probe: full domain [Tol_CS, RCS_ext - Tol_CS]
    N_PROBE_2 = 20   # Fine probe:   adaptive sub-domain [d_min_valid, d_hi]

    def _find_bracket(d_lo, d_hi, n_pts):
        """
        Log-spaced probe returning the first nan→finite and sign-change
        brackets found in [d_lo, d_hi].

        Parameters
        ----------
        d_lo, d_hi : float
            Search bounds (both > 0 required for log spacing).
        n_pts : int
            Number of probe points.

        Returns
        -------
        nan_to_finite : tuple or None
            (d_left, d_right) straddling the first nan→finite transition.
        sign_change : tuple or None
            (d_left, d_right) straddling the root (first sign change).
        """
        d_vals = np.logspace(np.log10(d_lo), np.log10(d_hi), n_pts)
        y_vals = np.array([f_sigma_diff(d) for d in d_vals])

        nan_to_finite = None
        sign_change   = None

        for i in range(1, n_pts):
            fp = y_vals[i - 1]
            fc = y_vals[i]
            finite_p = np.isfinite(fp)
            finite_c = np.isfinite(fc)

            if not finite_p and finite_c and nan_to_finite is None:
                nan_to_finite = (d_vals[i - 1], d_vals[i])

            if finite_p and finite_c and fp * fc < 0:
                sign_change = (d_vals[i - 1], d_vals[i])
                break   # Monotone: first change is unique

        return nan_to_finite, sign_change

    def _binary_search_valid_boundary(lo, hi, n_iter=25, tol=1e-6):
        """
        Bisection on the nan/finite boundary of f_sigma_diff.

        Converges to d_min_valid: the smallest d where f_sigma_diff is
        finite (valid domain onset). Requires f_sigma_diff(lo) = nan and
        f_sigma_diff(hi) = finite at entry.

        Parameters
        ----------
        lo, hi : float   Initial bracket straddling the nan/finite edge.
        n_iter  : int    Maximum bisection iterations.
        tol     : float  Convergence tolerance on d [m].

        Returns
        -------
        float   d_min_valid (hi side of converged bracket).
        """
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if np.isfinite(f_sigma_diff(mid)):
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return hi   # Conservative: first definitely-finite point

    def find_d_solution():

        d_lo = Tol_CS + 1e-6   # Strict lower bound (never a valid solenoid at d→0)
        d_hi = RCS_ext - Tol_CS

        if d_lo >= d_hi:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # ── Pass 1: coarse log-spaced probe ──────────────────────────────
        nan_to_finite, sign_change = _find_bracket(d_lo, d_hi, N_PROBE_1)

        if debug:
            print(f'[SEARCH P1] d in [{d_lo:.5f}, {d_hi:.4f}] m, '
                  f'{N_PROBE_1} pts')
            print(f'[SEARCH P1] nan→finite: {nan_to_finite}, '
                  f'sign-change: {sign_change}')

        # ── Pass 2: adaptive refinement (only if Pass 1 insufficient) ────
        if sign_change is None:

            if nan_to_finite is None:
                # f_sigma_diff is nan everywhere: domain is entirely infeasible
                # (B_CS > B_max_CS for all d, or no SC operating point).
                if debug:
                    print("[SEARCH P2] No finite region found — design infeasible.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Pin d_min_valid via bisection on the nan/finite boundary.
            d_min_valid = _binary_search_valid_boundary(
                nan_to_finite[0], nan_to_finite[1])

            if debug:
                print(f'[SEARCH P2] d_min_valid = {d_min_valid:.6f} m  '
                      f'(bisection from {nan_to_finite})')

            # Finer log-spaced probe on the valid sub-domain.
            # This handles the two extreme failure modes of Pass 1:
            #   (a) Root is in a very narrow window just above d_min_valid
            #       (ultra-thin HTS CS): log spacing from d_min_valid is
            #       much denser than from d_lo = Tol_CS.
            #   (b) Root is very close to d_hi (large LTS CS, nearly a
            #       solid cylinder is excluded geometrically): the valid
            #       domain's upper extent is well-sampled by log spacing
            #       anchored at d_min_valid.
            d_lo2 = d_min_valid
            d_hi2 = d_hi
            if d_lo2 >= d_hi2:
                if debug:
                    print("[SEARCH P2] d_min_valid >= d_hi: design infeasible.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            _, sign_change = _find_bracket(d_lo2, d_hi2, N_PROBE_2)

            if debug:
                print(f'[SEARCH P2] fine probe [{d_lo2:.6f}, {d_hi2:.4f}] m, '
                      f'{N_PROBE_2} pts → sign-change: {sign_change}')

            if sign_change is None:
                # No sign change after fine probe: stress constraint cannot
                # be satisfied within the geometrically valid domain.
                if debug:
                    print("[SEARCH P2] No root found — stress limit "
                          "not reachable in valid domain.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # ── Pass 3: brentq on the tight bracket ──────────────────────────
        try:
            d_sol = brentq(f_sigma_diff,
                           sign_change[0], sign_change[1],
                           xtol=1e-4, full_output=False)
        except ValueError:
            if debug:
                print(f"[SEARCH P3] brentq failed on bracket {sign_change}.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # ── Recover full output at solution ───────────────────────────────
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = \
            d_to_solve(d_sol)

        if not np.isfinite(Sigma_CS):
            if debug:
                print(f"[SEARCH P3] d_to_solve nan at d_sol={d_sol:.5f} m.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if debug:
            print(f"[SEARCH P3] Solution: d={d_sol:.5f} m, "
                  f"B_CS={B_CS:.2f} T, Sigma_CS={Sigma_CS/1e6:.1f} MPa")

        return d_sol, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

    # Execute search
    d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = find_d_solution()

    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

#%% ===========================================================================
# CS CIRCE MODEL
# =============================================================================

def f_CS_CIRCE(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
              config: GlobalConfig):
    """
    Calculate the Central Solenoid (CS) thickness using CIRCE.
    
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
        Cumulative radial build (m)
    c : float
        TF coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T)
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier
    Jc_manual : float
        Manual current density if needed
    T_Helium : float
        Helium coolant temperature (K)
    Choice_Buck_Wedg : str
        Mechanical configuration: 'Bucking', 'Wedging', or 'Plug'
    κ : float
        Plasma elongation
        
    Returns
    -------
    tuple of float
        (d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS)
        
        Returns (nan, nan, nan, nan, nan, nan, nan) if no solution exists.

    References
    ----------
    [Auclair et al. NF 2026]

    config : GlobalConfig
        Global design configuration. Used to access: Gap, I_cond, V_max,
        f_He, f_In, T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn, Marge_T_NbTi,
        Marge_T_REBCO, Eps, Tet, n_CS.
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    Gap         = config.Gap
    I_cond      = config.I_cond
    V_max       = config.V_max
    f_He        = config.f_He
    f_In        = config.f_In
    T_hotspot   = config.T_hotspot
    RRR         = config.RRR
    Marge_T_He      = config.Marge_T_He
    Marge_T_Nb3Sn   = config.Marge_T_Nb3Sn
    Marge_T_NbTi    = config.Marge_T_NbTi
    Marge_T_REBCO   = config.Marge_T_REBCO
    Eps         = config.Eps
    Tet         = config.Tet
    n_CS        = config.n_CS
    Young_modul_Steel = config.Young_modul_Steel
    nu_Steel          = config.nu_Steel
    debug = False
    Tol_CS = 1e-3

    if Choice_Buck_Wedg in ('Bucking', 'Plug'):
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        if debug:
            print("[f_CS_CIRCE] Invalid mechanical choice:", Choice_Buck_Wedg)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if RCS_ext <= 0.0:
        if debug:
            print("[f_CS_CIRCE] Non-positive RCS_ext:", RCS_ext)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF
    
    # ------------------------------------------------------------------
    # Main solving function (uses CACHED cable current density)
    # ------------------------------------------------------------------

    def d_to_solve(d):
        
        if d < Tol_CS or d > RCS_ext - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        RCS_int = RCS_ext - d
        B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        H_CS = 2 * (κ * a + b + 1)
        E_mag_CS = calculate_E_mag_CS(B_CS, RCS_int, RCS_ext, H_CS)
        
        # STRATEGY C: Use cached cable current density
        result_J = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS, T_op=T_Helium, E_mag=E_mag_CS,
            I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS, tau_h=tau_h, f_He=f_He, f_In=f_In,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=Jc_manual if Supra_choice_CS == 'Manual' else None)
        J_max_CS = result_J['J_wost']
        
        if J_max_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        alpha = B_CS / (μ0 * J_max_CS * d)
        
        if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if B_CS < Tol_CS or B_CS > B_max_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Compute principal stresses with CIRCE and apply 3D Tresca criterion.
        # Hoop and radial smeared stresses are converted to steel via 1/(1-alpha).
        P_CS = B_CS**2 / (2.0 * μ0)
        P_TF = B_max_TF**2 / (2.0 * μ0) * (R0 - a - b) / RCS_ext

        disR       = 20
        R_arr      = np.array([RCS_int, RCS_ext])
        E_arr      = np.array([Young_modul_Steel])
        config_arr = np.array([0])

        def gamma_func(alpha_val, n_val):
            """
            Effective steel area fraction for transverse (axial-face) loading.
            Identical to the TF gamma function; derived from the square conductor
            geometry with n_val turns sharing the inter-turn load.
            """
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

        gamma_val = gamma_func(alpha, n_CS)
        if np.isnan(gamma_val):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Axial smeared stress at the midplane (compressive, < 0).
        # Controlled by config.cs_axial_stress (default False = disabled).
        # When disabled, sigma_z = 0 in all Tresca evaluations.
        _cs_axial_on = getattr(config, 'cs_axial_stress', False)
        if _cs_axial_on:
            J_smear   = J_max_CS * alpha
            σ_z_smear = f_sigma_z_CS_axial(J_smear, RCS_int, RCS_ext, H_CS / 2.0)
            σ_z_steel = σ_z_smear / gamma_val                       # Peak steel axial stress [Pa]
        else:
            σ_z_smear = 0.0
            σ_z_steel = 0.0

        scale_tr = 1.0 / (1.0 - alpha)   # Hoop/radial smeared → steel (volume average)
        scale_z  = 1.0 / gamma_val        # Axial smeared → steel (minimum ligament)

        def tresca_radial(SigT, SigR, sz_smear, s_tr, s_z):
            """
            Point-wise Tresca criterion on the CIRCE radial stress profile.

            Parameters
            ----------
            SigT, SigR : ndarray
                Smeared hoop and radial stress profiles from F_CIRCE0D [Pa].
            sz_smear : float
                Smeared axial stress (scalar, uniform over profile) [Pa].
            s_tr : float
                Scale factor smeared → steel for hoop/radial: 1/(1-alpha)
                (volume-average section, body force in steel).
            s_z : float
                Scale factor smeared → steel for axial: 1/gamma_val
                (minimum ligament — cable→jacket wall transfer).

            Returns
            -------
            float
                Maximum Tresca equivalent stress in the steel [Pa].
            """
            T = SigT * s_tr
            R = SigR * s_tr
            Z = sz_smear * s_z
            return float(np.max(np.maximum(np.maximum(np.abs(T - R),
                                                      np.abs(R - Z)),
                                           np.abs(T - Z))))

        # --- CIRCE computation ---
        if Choice_Buck_Wedg == 'Bucking':

            # Light case: CS energized — non-zero σ_z.
            J_light = np.array([J_max_CS * alpha])
            B_light = np.array([B_CS])
            SigR_L, SigT_L, _, _, _ = F_CIRCE0D(
                disR, R_arr, J_light, B_light, 0, P_TF, E_arr, nu_Steel, config_arr)
            tresca_light = tresca_radial(SigT_L, SigR_L, σ_z_smear, scale_tr, scale_z)

            # Strong case: J = 0, σ_z = 0.
            J_strong = np.array([0.0])
            B_strong = np.array([0.0])
            SigR_S, SigT_S, _, _, _ = F_CIRCE0D(
                disR, R_arr, J_strong, B_strong, 0, P_TF, E_arr, nu_Steel, config_arr)
            tresca_strong = tresca_radial(SigT_S, SigR_S, 0.0, scale_tr, scale_z)

            Sigma_CS = max(tresca_light, tresca_strong)
            σ_theta  = float(np.max(np.abs(SigT_L))) * scale_tr
            σ_r      = float(np.max(np.abs(SigR_L))) * scale_tr
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Wedging':

            J_w = np.array([J_max_CS * alpha])
            B_w = np.array([B_CS])
            SigR_W, SigT_W, _, _, _ = F_CIRCE0D(
                disR, R_arr, J_w, B_w, 0, 0, E_arr, nu_Steel, config_arr)
            Sigma_CS = tresca_radial(SigT_W, SigR_W, σ_z_smear, scale_tr, scale_z)
            σ_theta  = float(np.max(np.abs(SigT_W))) * scale_tr
            σ_r      = float(np.max(np.abs(SigR_W))) * scale_tr
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Plug':

            if abs(P_CS - P_TF) > abs(P_TF):
                # Bucking-dominated: two-case Tresca as Bucking.
                J_light  = np.array([J_max_CS * alpha])
                B_light  = np.array([B_CS])
                SigR_L, SigT_L, _, _, _ = F_CIRCE0D(
                    disR, R_arr, J_light, B_light, 0, P_TF, E_arr, nu_Steel, config_arr)
                tresca_light = tresca_radial(SigT_L, SigR_L, σ_z_smear, scale_tr, scale_z)

                J_strong = np.array([0.0])
                B_strong = np.array([0.0])
                SigR_S, SigT_S, _, _, _ = F_CIRCE0D(
                    disR, R_arr, J_strong, B_strong, 0, P_TF, E_arr, nu_Steel, config_arr)
                tresca_strong = tresca_radial(SigT_S, SigR_S, 0.0, scale_tr, scale_z)

                Sigma_CS = max(tresca_light, tresca_strong)
                σ_theta  = float(np.max(np.abs(SigT_L))) * scale_tr
                σ_r      = float(np.max(np.abs(SigR_L))) * scale_tr
                σ_z      = σ_z_steel

            elif abs(P_CS - P_TF) <= abs(P_TF):
                # Plug-dominated: J → 0, dominant stress is radial, σ_z = 0.
                # Hoop/radial scaled by 1/gamma_val (TF plug geometry).
                gamma_val_plug = gamma_func(alpha, n_CS)
                J_plug = np.array([0.0])
                B_plug = np.array([0.0])
                SigR_P, SigT_P, _, _, _ = F_CIRCE0D(
                    disR, R_arr, J_plug, B_plug, P_TF, P_TF, E_arr, nu_Steel, config_arr)
                scale_plug = 1.0 / gamma_val_plug
                Sigma_CS = tresca_radial(SigT_P, SigR_P, 0.0, scale_plug, scale_plug)
                σ_theta  = float(np.max(np.abs(SigT_P))) * scale_plug
                σ_r      = float(np.max(np.abs(SigR_P))) * scale_plug
                σ_z      = 0.0
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if Sigma_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        Steel_fraction = 1 - alpha

        return (float(Sigma_CS), float(σ_z), float(σ_theta), float(σ_r),
                float(Steel_fraction), float(B_CS), float(J_max_CS))
    
    # ------------------------------------------------------------------
    # Root function
    # ------------------------------------------------------------------
    
    def f_sigma_diff(d):
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d)
        if not np.isfinite(Sigma_CS):
            return np.nan
        return Sigma_CS - σ_CS
    
    # ------------------------------------------------------------------
    # Root-finding — monotone-aware adaptive solver (see f_CS_D0FUS for
    # full rationale and algorithm description).
    # ------------------------------------------------------------------

    N_PROBE_1 = 15   # Coarse probe: full domain [Tol_CS, RCS_ext - Tol_CS]
    N_PROBE_2 = 20   # Fine probe:   adaptive sub-domain [d_min_valid, d_hi]

    def _find_bracket(d_lo, d_hi, n_pts):
        """Log-spaced probe → (nan_to_finite, sign_change) brackets."""
        d_vals = np.logspace(np.log10(d_lo), np.log10(d_hi), n_pts)
        y_vals = np.array([f_sigma_diff(d) for d in d_vals])
        nan_to_finite = None
        sign_change   = None
        for i in range(1, n_pts):
            fp, fc = y_vals[i - 1], y_vals[i]
            finite_p, finite_c = np.isfinite(fp), np.isfinite(fc)
            if not finite_p and finite_c and nan_to_finite is None:
                nan_to_finite = (d_vals[i - 1], d_vals[i])
            if finite_p and finite_c and fp * fc < 0:
                sign_change = (d_vals[i - 1], d_vals[i])
                break
        return nan_to_finite, sign_change

    def _binary_search_valid_boundary(lo, hi, n_iter=25, tol=1e-6):
        """Bisection on the nan/finite boundary → d_min_valid."""
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if np.isfinite(f_sigma_diff(mid)):
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return hi

    def find_d_solution():

        d_lo = Tol_CS + 1e-6
        d_hi = RCS_ext - Tol_CS

        if d_lo >= d_hi:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Pass 1: coarse log-spaced probe over full domain.
        nan_to_finite, sign_change = _find_bracket(d_lo, d_hi, N_PROBE_1)

        if debug:
            print(f'[SEARCH P1] d in [{d_lo:.5f}, {d_hi:.4f}] m, '
                  f'{N_PROBE_1} pts')
            print(f'[SEARCH P1] nan→finite: {nan_to_finite}, '
                  f'sign-change: {sign_change}')

        # Pass 2: adaptive refinement if Pass 1 did not bracket root.
        if sign_change is None:

            if nan_to_finite is None:
                if debug:
                    print("[SEARCH P2] No finite region — design infeasible.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Pin d_min_valid and re-probe on [d_min_valid, d_hi].
            d_min_valid = _binary_search_valid_boundary(
                nan_to_finite[0], nan_to_finite[1])

            if debug:
                print(f'[SEARCH P2] d_min_valid = {d_min_valid:.6f} m')

            d_lo2, d_hi2 = d_min_valid, d_hi
            if d_lo2 >= d_hi2:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            _, sign_change = _find_bracket(d_lo2, d_hi2, N_PROBE_2)

            if debug:
                print(f'[SEARCH P2] fine probe → sign-change: {sign_change}')

            if sign_change is None:
                if debug:
                    print("[SEARCH P2] No root — stress limit not reachable.")
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Pass 3: brentq on the tight bracket.
        try:
            d_sol = brentq(f_sigma_diff,
                           sign_change[0], sign_change[1],
                           xtol=1e-4, full_output=False)
        except ValueError:
            if debug:
                print(f"[SEARCH P3] brentq failed on bracket {sign_change}.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Recover full output at solution.
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = \
            d_to_solve(d_sol)

        if not np.isfinite(Sigma_CS):
            if debug:
                print(f"[SEARCH P3] d_to_solve nan at d_sol={d_sol:.5f} m.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if debug:
            print(f"[SEARCH P3] Solution: d={d_sol:.5f} m, "
                  f"B_CS={B_CS:.2f} T, Sigma_CS={Sigma_CS/1e6:.1f} MPa")

        return d_sol, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS
    
    # Execute search
    d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = find_d_solution()

    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

#%% CS Benchmark

if __name__ == "__main__":
    # CS coil benchmark table: D0FUS vs published references
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_CS_benchmark_table(cfg=cfg)

#%% CS plot

if __name__ == "__main__":
    # CS winding-pack thickness and B_CS vs volt-second — Academic/D0FUS/CIRCE
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_CS_thickness_vs_flux(
        a_cs=3.0, b_cs=1.2, c_cs=2.0, R0_cs=9.0,
        B_TF=13.0, sigma_CS=300e6, J_wost_CS=30e6, cfg=cfg)

#%% Note:
# CIRCE TF double cylindre en wedging ? multi cylindre for grading ?
# Nécessite la résolution de R_int et R_sep en même temps
# Permettrait aussi de mettre la répartition en tension en rapport de surface
#%% Print

# print("D0FUS_radial_build_functions loaded")