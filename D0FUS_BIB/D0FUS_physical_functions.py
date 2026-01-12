"""
Physical functions definition for the D0FUS - Design 0-dimensional for FUsion Systems project
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Imports

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

#%% Physical Functions

def f_Kappa(A, Option_Kappa, κ_manual, ms):
    """
    Estimate the maximum achievable plasma elongation as a function of aspect ratio.
    
    The elongation is limited by vertical stability considerations. Different empirical
    scalings are available based on tokamak databases and theoretical limits.
    
    Parameters
    ----------
    A : float
        Plasma aspect ratio (R₀/a)
    Option_Kappa : str
        Scaling law selection:
        - 'Stambaugh' : Empirical scaling from tokamak database (most optimistic)
        - 'Freidberg' : Theoretical MHD stability limit
        - 'Wenninger' : EU-DEMO scaling (most pessimistic)
        - 'Manual'    : User-defined value (uses κ_manual)
    κ_manual : float
        Manual elongation value (only used if Option_Kappa == 'Manual')
    
    Returns
    -------
    κ : float
        Maximum achievable elongation
        Returns np.nan if computed value is non-physical (κ ≤ 0)
    
    References
    ----------
    - Stambaugh:
    Stambaugh, R. D., L. L. Lao, and E. A. Lazarus.
    "Relation of vertical stability and aspect ratio in tokamaks." Nuclear fusion 32.9 (1992): 1642.
    
    - Freidberg:
    Freidberg, J. P., Cerfon, A., & Lee, J. P. (2015).
    "Tokamak elongation–how much is too much?" Part 1. Theory. Journal of Plasma Physics, 81(6), 515810607.
    +
    Lee, J. P., Cerfon, A., Freidberg, J. P., & Greenwald, M. (2015). 
    "Tokamak elongation–how much is too much?" Part 2. Numerical results. Journal of Plasma Physics, 81(6), 515810608.
    
    - Wenninger:
    Wenninger, R., Arbeiter, F., Aubert, J., Aho-Mantila, L., Albanese, R., Ambrosino, R., ... & Zohm, H. (2015).
    "Advances in the physics basis for the European DEMO design". Nuclear Fusion, 55(6), 063003.
    +
    Coleman, M., Zohm, H., Bourdelle, C., Maviglia, F., Pearce, A. J., Siccinio, M., ... & Wiesen, S. (2025).
    "Definition of an EU-DEMO design point robust to epistemic plasma physics uncertainties". Nuclear Fusion, 65(3), 036039.
    
    """
    
    if Option_Kappa == 'Stambaugh':
        # Empirical scaling with exponential rolloff at low aspect ratio
        κ = 0.95 * (2.4 + 65 * np.exp(-A / 0.376))
        
    elif Option_Kappa == 'Freidberg':
        # Theoretical MHD stability limit
        κ = 0.95 * (1.81153991 * A**0.009042 + 1.5205 * A**(-1.63))
        
    elif Option_Kappa == 'Wenninger':
        # EU-DEMO scaling with stability margin
        κ = 1.12 * ((18.84 - 0.87*A - np.sqrt(4.84*A**2 - 28.77*A + 52.52 + 14.74*ms)) / 7.37)
        
    elif Option_Kappa == 'Manual':
        # User-specified elongation
        κ = κ_manual
        
    else:
        raise ValueError(f"Unknown Option_Kappa: '{Option_Kappa}'. "
                        f"Valid options: 'Stambaugh', 'Freidberg', 'Wenninger', 'Manual'")
    
    # Physical validity check
    κ = np.where(np.asarray(κ) <= 0, np.nan, κ)
    
    return κ

if __name__ == "__main__":
    
    A = np.linspace(1.5, 5.0, 200)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.plot(A, f_Kappa(A, 'Stambaugh', κ_manual=1.7, ms=0.3), label='Stambaugh')
    ax.plot(A, f_Kappa(A, 'Freidberg', κ_manual=1.7, ms=0.3), label='Freidberg')
    ax.plot(A, f_Kappa(A, 'Wenninger', κ_manual=1.7, ms=0.3), label='Wenninger')
    
    ax.set_xlabel('$A$')
    ax.set_ylabel('$\\kappa$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def f_Kappa_95(kappa):
    """
    
    Estimate the elongation at 95% of the poloidal flux (kappa_95) 
    from the total elongation (kappa).

    The 95% elongation typically reflects the shape of the inner plasma
    and is slightly lower than the total elongation measured at the last 
    closed flux surface (LCFS).
    Scaling taken from 1989 ITER guidelines

    Parameters
    ----------
    kappa : Total elongation (dimensionless)

    Returns:
    -------
    kappa_95 : Estimated elongation at 95% poloidal flux (dimensionless)
    
    """
    kappa_95 = kappa / 1.12
    return kappa_95

if __name__ == "__main__":
    
    A = np.linspace(1.5, 5.0, 200)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.plot(A, f_Kappa_95(f_Kappa(A, 'Stambaugh', κ_manual=1.7, ms=0.3)), label='Stambaugh')
    ax.plot(A, f_Kappa_95(f_Kappa(A, 'Freidberg', κ_manual=1.7, ms=0.3)), label='Freidberg')
    ax.plot(A, f_Kappa_95(f_Kappa(A, 'Wenninger', κ_manual=1.7, ms=0.3)), label='Wenninger')
    
    ax.set_xlabel('$A$')
    ax.set_ylabel('$\\kappa_{95}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def f_Delta(kappa):
    """
    
    Estimate the maximum triangularity (delta) from the total elongation (kappa).

    This empirical relationship approximates the maximum triangularity 
    Scaling taken from TREND p 53

    Parameters:
    -------
    kappa : Total elongation (dimensionless)

    Returns:
    -------
    delta: Estimated maximum triangularity (delta, dimensionless)
    
    """
    delta = 0.6 * (kappa - 1)
    return delta


def f_Delta_95(delta):
    """
    
    Estimate the triangularity at 95% of the poloidal flux (delta_95) 
    from the maximum triangularity (delta).

    This is useful to characterize the inner shape of the plasma where 
    confinement and stability are more critical.
    Scaling taken from 1989 ITER guidelines

    Parameters:
    -------    
    delta : Maximum triangularity (dimensionless)

    Returns:
    -------
    delta_95: Estimated triangularity at 95% poloidal flux
    
    """
    delta_95 = delta / 1.5
    return delta_95

if __name__ == "__main__":
    
    # Plasma cross-section parameters
    a = 1.0         # Minor radius [m]
    kappa = 1.7     # Elongation (vertical stretching ratio)
    delta = 0.5     # Triangularity (top-bottom asymmetry)
    n_theta = 500   # Number of poloidal angle points
    
    # Poloidal angle array [0, 2π]
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    # --- Case 1: No triangularity (simple ellipse) ---
    x_ellipse = a * np.cos(theta)           # Horizontal coordinate [m]
    z_ellipse = kappa * a * np.sin(theta)   # Vertical coordinate [m]
    
    # --- Case 2: With triangularity (Miller parameterization) ---
    # The triangularity shifts the plasma cross-section horizontally
    # as a function of poloidal angle, creating a D-shaped profile
    x_tri = a * np.cos(theta + delta * np.sin(theta))  # Modified horizontal coordinate [m]
    z_tri = kappa * a * np.sin(theta)                  # Vertical coordinate [m]
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot elliptical cross-section (δ = 0)
    ax.plot(x_ellipse, z_ellipse, 'b--', linewidth=2, 
            label=f"Ellipse (δ = 0)")
    
    # Plot D-shaped cross-section (δ = {delta})
    ax.plot(x_tri, z_tri, 'r-', linewidth=2, 
            label=f"D-shape (δ = {delta})")
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel("R - R₀ [m] (radial direction)", fontsize=11)
    ax.set_ylabel("Z [m] (vertical direction)", fontsize=11)
    ax.set_title(f"Plasma Poloidal Cross-Section\n(a = {a} m, κ = {kappa})", 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)  # Midplane reference
    ax.axvline(0, color='k', linewidth=0.5)  # Magnetic axis reference
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    

def f_li(nu_n, nu_T):
    """
    
    Estimate the internal inductance (li) of a plasma using an empirical formula
    based on current profile shape parameters.

    The formula is derived from an empirical relationship of D3D founded in the Wesson.

    Parameters:
    -------
    nu_n : Density profile exponent (e.g., from n(r) ∝ (1 - r^2)^nu_n)
    nu_T : Temperature profile exponent (e.g., from T(r) ∝ (1 - r^2)^nu_T)

    Returns:
    -------
    li : Estimated internal inductance (dimensionless)

    Notes:
    The effective current profile exponent is approximated as:
    nu_J = 0.453 - 0.1 * (nu_p - 1.5)
    where nu_p = nu_n + nu_T
    Taken from Eq 36 from [Segal Pulsed vs Steady State]
    
    """
    
    nu_p = nu_n + nu_T
    nu_J = 0.453 - 0.1 * (nu_p - 1.5)
    li = np.log(1.65 + 0.89 * nu_J)
    
    return li

if __name__ == "__main__":
    """
    Test of internal inductance calculation function f_li.
    Typical ITER value: li ~ 0.8 for parabolic profiles (nu_n ≈ 0.1, nu_T ≈ 1)
    """
    
    # ITER-like test case with parabolic profiles
    nu_n = 0.1  # Density profile exponent: n(r) ∝ (1 - r²)^nu_n
    nu_T = 1.0  # Temperature profile exponent: T(r) ∝ (1 - r²)^nu_T
    
    li_result = f_li(nu_n, nu_T)
    
    print("="*60)
    print("Internal Inductance (li) Function Test")
    print("="*60)
    print(f"Input parameters:")
    print(f"  nu_n = {nu_n}  (density profile exponent)")
    print(f"  nu_T = {nu_T}  (temperature profile exponent)")
    print(f"\nProfile shapes:")
    print(f"  n(r) ∝ (1 - r²)^{nu_n}")
    print(f"  T(r) ∝ (1 - r²)^{nu_T}")
    print(f"\nResult:")
    print(f"  li = {li_result:.4f}")
    print(f"\nExpected range: [0.7, 1.0] (typical tokamak)")
    print(f"ITER reference:  ~0.8")
    print("="*60)

def f_plasma_volume(R0, a, kappa, delta):
    """
    
    Calculate the volume of an axisymmetric tokamak plasma 
    using elongation (κ) and triangularity (δ).
    Approximation from Miller coordinates at O(2)

    Parameters:
    -------
    R0 : Major radius [m]
    a : Minor radius [m]
    kappa : Elongation 
    delta : Triangularity

    Returns:
    -------
    float: Volume du plasma [m³]

    Notes:
    D-shape approximation, no squareness is taken into account
    
    """
    
    V = 2 * np.pi**2 * R0 * a**2 * kappa * (1 - (a * delta) / (4 * R0) - (delta**2) / 8)
    
    return V
    
if __name__ == "__main__":

    # -------------------------
    # Physical parameters
    # -------------------------
    R0 = 3.0      # Major radius [m]
    a = 1.0       # Minor radius [m]
    kappa = 1.7   # Elongation
    
    # Discretization for numerical integration
    n_theta = 5000
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    dtheta = theta[1] - theta[0]
    
    # -------------------------
    # Volume functions
    # -------------------------
    # Simpler expression [Wesson]
    def V_simple(R0, a, kappa):
        return 2 * np.pi**2 * R0 * a**2 * kappa
    # Often used in system code ex: PROCESS and mentionned as reference in [Martin]
    def V_process(R0, a, kappa, delta):
        return 2 * np.pi**2 * R0 * a**2 * kappa * (1 - (1 - 8/(3*np.pi)) * delta * a / R0)
    # 1rst order from Miller [Auclair]
    def V_rec_1(R0, a, kappa, delta):
        return 2 * np.pi**2 * R0 * a**2 * kappa * (1 - (delta * a) / (4 * R0))
    # second order from Miller [Auclair]
    def V_rec_2(R0, a, kappa, delta):
        return 2 * np.pi**2 * R0 * a**2 * kappa * (1 - (a * delta) / (4 * R0) - (delta**2) / 8)
    # Miller coordinates
    def V_miller(R0, a, kappa, delta):
        R_theta = R0 + a * np.cos(theta + delta * np.sin(theta))
        Z_theta = kappa * a * np.sin(theta)
        dZ = np.gradient(Z_theta, dtheta)
        integrand = R_theta**2 * dZ
        return np.pi * np.trapezoid(integrand, theta)
    
    # -------------------------
    # Sweep over triangularity
    # -------------------------
    deltas = np.linspace(0, 0.5, 30)
    
    V_s    = [V_simple(R0, a, kappa) for d in deltas]
    V_proc = [V_process(R0, a, kappa, d) for d in deltas]
    V_ord1  = [V_rec_1(R0, a, kappa, d) for d in deltas]
    V_ord2 = [V_rec_2(R0, a, kappa, d) for d in deltas]
    V_num  = [V_miller(R0, a, kappa, d) for d in deltas]
    
    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(9,6))
    plt.plot(deltas, V_s, 'k--', lw=2, label='V simple [Wesson]')
    plt.plot(deltas, V_proc, 'g-.', lw=2, label='V Process [Martin]')
    plt.plot(deltas, V_ord1, 'b-.', lw=2, label='V 1rst Order [Auclair]')
    plt.plot(deltas, V_ord2, 'm-.', lw=2, label='V 2d Order [Auclair]')
    plt.plot(deltas, V_num, 'ro-', markersize=5, label='Numerical [Miller]')
    
    plt.xlabel('Triangularity δ', fontsize=12)
    plt.ylabel('Plasma Volume [m³]', fontsize=12)
    plt.title('Comparison of Plasma Volumes', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


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

def f_Tprof(Tbar,nu_T,rho):
    """
    
    Estimate the temperature at rho
    Considering a specific temperature profile and axisymmetry
    No pedestal for now (to be implemented)

    Parameters
    ----------
    Tbar : The mean temperature of the plasma [keV]
    nu_T : Temperature profile parameter
    rho : Normalized minor radius = r/a
        
    Returns
    -------
    T : The estimated temperature at rho position
    
    """
    T = Tbar*(1+nu_T)*(1-rho**2)**nu_T
    return T

def f_nprof(nbar,nu_n,rho):
    """
    
    Estimate the density at rho
    Considering a specific density profile and axisymmetry
    No pedestal for now (to be implemented)

    Parameters
    ----------
    nbar : The mean electronic density of the plasma [1e20p/m^3]
    nu_n : Density profile parameter
    rho : Normalized minor radius = r/a
        
    Returns
    -------
    n : The estimated density at rho position
    
    """
    n = nbar*(1+nu_n)*(1-rho**2)**nu_n
    return n

def plot_profiles(Tbar, nu_T, nbar, nu_n, nrho=100):
    """
    Plot temperature and density profiles
    
    Parameters
    ----------
    Tbar : float - mean temperature [keV]
    nu_T : float - temperature profile parameter
    nbar : float - mean density [1e20 p/m^3]
    nu_n : float - density profile parameter
    nrho : int - number of points for rho
    """
    rho = np.linspace(0, 1, nrho)
    T = f_Tprof(Tbar, nu_T, rho)
    n = f_nprof(nbar, nu_n, rho)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Normalized minor radius (rho)")
    ax1.set_ylabel("Temperature [keV]", color="tab:red")
    ax1.plot(rho, T, color="tab:red", label="Temperature")
    ax1.tick_params(axis='y', labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Density [1e20 p/m^3]", color="tab:blue")
    ax2.plot(rho, n, color="tab:blue", linestyle="--", label="Density")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    plt.title("Plasma Temperature and Density Profiles")
    fig.tight_layout()
    
    # Grille activée
    ax1.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.show()

if __name__ == "__main__":
    """
    Test of radial profile visualization function.
    Plots density and temperature profiles with typical ITER-like parameters.
    """
    
    # ITER-like plasma parameters
    Tbar = 14        # Volume-averaged temperature [keV]
    nu_T = 1.0       # Temperature profile exponent: T(r) ∝ (1 - r²)^nu_T
    nbar = 1e20      # Volume-averaged density [m⁻³]
    nu_n = 0.1       # Density profile exponent: n(r) ∝ (1 - r²)^nu_n
    
    print("="*60)
    print("Radial Profile Visualization Test")
    print("="*60)
    print(f"Parameters:")
    print(f"  T̄     = {Tbar} keV")
    print(f"  nu_T  = {nu_T} (temperature peaking)")
    print(f"  n̄     = {nbar:.1e} m⁻³")
    print(f"  nu_n  = {nu_n} (density peaking)")
    print("="*60)
    
    # Generate profile plots
    plot_profiles(Tbar=Tbar, nu_T=nu_T, nbar=nbar, nu_n=nu_n)

def f_sigmav(T):
    """
    
    Allows the calculation of the cross section ⟨σv⟩ from the temperature.
    Here for the D-T reaction, possible to add other ones
    Source: Bosch and Hale, 1992, Nuclear Fusion.
    Range of validity: 0.2 to 100 keV with a maximum deviation of 0.35%.
    
    Parameters
    ----------
    T : The Temperature [keV]
        
    Returns
    -------
    sigma_v : The estimated cross section [m^3 s^-1]
    
    """
    Bg = 34.3827  # in (keV**(1/2))
    mc2 = 1124656  # in keV
    c1 = 1.17302e-9
    c2 = 1.51361e-2
    c3 = 7.51886e-2
    c4 = 4.60643e-3
    c5 = 1.35000e-2
    c6 = -1.06750e-4
    c7 = 1.36600e-5
    theta = T / (1 - (T * (c2 + T * (c4 + T * c6)) / (1 + T * (c3 + T * (c5 + T * c7)))))
    phi = (Bg**2/ (4 * theta))**(1/3)
    sigma_v = c1 * theta * (phi/(mc2*T**3))**(1/2) * np.exp(-3 * phi) * 1e-6
    
    return sigma_v

def f_nbar_advanced(P_fus, nu_n, nu_T, f_alpha, Tbar, V):
    """
    Compute the mean electron density required to reach 
    a given fusion power P_fus in a plasma of volume V.

    Parameters
    ----------
    P_fus : target fusion power [MW]
    nu_n  : density profile parameter
    nu_T  : temperature profile parameter
    f_alpha : relative fraction of alpha particles in the plasma
    Tbar  : average temperature [keV]
    V     : plasma volume [m^3]

    Returns
    -------
    n_bar : mean electron density [10^20 m^-3]
    """

    # --- Normalized integral for <σv>eff ---
    def integrand(rho):
        T_local = f_Tprof(Tbar, nu_T, rho)     # temperature profile T(ρ)
        sigmav  = f_sigmav(T_local)            # reactivity <σv>(T)
        return sigmav * (1 - rho**2)**(2*nu_n) * 2 *  rho

    sigma_v, _ = quad(integrand, 0, 1)

    # --- Solve for n (total fuel density D+T) ---
    P_watt = P_fus * 1e6
    n = 2 / (1 + nu_n) * np.sqrt(P_watt / (sigma_v * (E_ALPHA + E_N) * V))

    # --- Convert to electron density (including alpha dilution) ---
    # n_e = n_D + n_T + 2 * n_alpha = n + 2 * f_alpha * n_e
    n_e = n / (1 - 2*f_alpha)

    # Return in units of 1e20 m^-3
    return n_e / 1e20

def f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, kappa):
    """
    Compute the mean electron density required to reach 
    a given fusion power P_fus in a plasma of volume V.

    Parameters
    ----------
    P_fus : target fusion power [MW]
    nu_n  : density profile parameter
    nu_T  : temperature profile parameter
    f_alpha : relative fraction of alpha particles in the plasma
    Tbar  : average temperature [keV]
    V     : plasma volume [m^3]

    Returns
    -------
    n_bar : mean electron density [10^20 m^-3]
    """

    # --- Normalized integral for <σv>eff ---
    def integrand(rho):
        T_local = f_Tprof(Tbar, nu_T, rho)     # temperature profile T(ρ)
        sigmav  = f_sigmav(T_local)            # reactivity <σv>(T)
        return sigmav * (1 - rho**2)**(2*nu_n) * 2 *  rho

    I, _ = quad(integrand, 0, 1)

    # --- Solve for n (total fuel density D+T) ---
    P_watt = P_fus * 1e6
    V = 2 * np.pi**2 * R0 * kappa * a**2
    n = 2 / (1 + nu_n) * np.sqrt(P_watt / (I * (E_ALPHA + E_N) * V))

    # --- Convert to electron density (including alpha dilution) ---
    # n_e = n_D + n_T + 2 * n_alpha = n + 2 * f_alpha * n_e
    n_e = n / (1 - 2*f_alpha)

    # Return in units of 1e20 m^-3
    return n_e / 1e20

if __name__ == "__main__":
    """
    Test of density requirement function f_nbar.
    Sweeps major radius to visualize the density-geometry trade-off 
    for achieving a target fusion power.
    """
    
    # Target fusion power and plasma parameters
    P_fus = 2000        # Fusion power [MW] (ITER Q=10 scenario)
    nu_n = 0.1          # Density profile exponent
    nu_T = 1.0          # Temperature profile exponent
    f_alpha = 0.06      # Alpha particle confinement fraction
    Tbar = 14           # Volume-averaged temperature [keV]
    
    # Fixed geometry parameters
    aspect_ratio = 3.0  # A = R0/a
    kappa = 1.7         # Plasma elongation
    
    # Major radius sweep range
    R0_min = 3.0        # Minimum major radius [m]
    R0_max = 10.0       # Maximum major radius [m]
    n_points = 500      # Number of points in sweep
    
    print("="*60)
    print("Density-Geometry Trade-off Analysis")
    print("="*60)
    print(f"Target fusion power: P_fus = {P_fus} MW")
    print(f"Temperature:         T̄    = {Tbar} keV")
    print(f"Profile exponents:   nu_n = {nu_n}, nu_T = {nu_T}")
    print(f"Alpha fraction:      f_α  = {f_alpha}")
    print(f"Aspect ratio:        A    = {aspect_ratio}")
    print(f"Elongation:          κ    = {kappa}")
    print(f"Major radius range:  {R0_min} - {R0_max} m")
    print("="*60)
    
    # Compute required density and volume for each major radius
    R0_values = np.linspace(R0_min, R0_max, n_points)
    nbar_values = []
    V_values = []
    
    for R0 in R0_values:
        a = R0 / aspect_ratio  # Minor radius from aspect ratio
        
        # Calculate required density
        nbar = f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, kappa)
        nbar_values.append(nbar)
        
        # Calculate plasma volume for reference
        delta = f_Delta(kappa)  # Triangularity from elongation
        V = f_plasma_volume(R0, a, kappa, delta)
        V_values.append(V)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Density vs Major Radius
    ax1.plot(R0_values, nbar_values, 'b-', linewidth=2)
    ax1.set_xlabel("Major Radius $R_0$ [m]", fontsize=12)
    ax1.set_ylabel("Required Mean Density $\\bar{n}_e$ [$10^{20}$ m$^{-3}$]", 
                   fontsize=12)
    ax1.set_title("Density vs Major Radius", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Density vs Volume
    ax2.plot(V_values, nbar_values, 'r-', linewidth=2)
    ax2.set_xlabel("Plasma Volume $V$ [m³]", fontsize=12)
    ax2.set_ylabel("Required Mean Density $\\bar{n}_e$ [$10^{20}$ m$^{-3}$]", 
                   fontsize=12)
    ax2.set_title("Density vs Volume", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Add text with parameters
    textstr = f'$P_{{fus}}$ = {P_fus} MW\n$\\bar{{T}}$ = {Tbar} keV\n$A$ = {aspect_ratio}\n$\\kappa$ = {kappa}'
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print some reference values
    print("\nReference values:")
    print(f"  R0 = 6 m → V = {V_values[np.argmin(np.abs(np.array(R0_values) - 6.0))]:.1f} m³, "
          f"n̄ = {nbar_values[np.argmin(np.abs(np.array(R0_values) - 6.0))]:.2f} × 10²⁰ m⁻³")

def f_pbar(nu_n, nu_T, n_bar, Tbar, f_alpha):
    """
    Estimate the mean plasma pressure.

    Parameters
    ----------
    nu_n : density profile parameter
    nu_T : temperature profile parameter
    n_bar : mean electron density [1e20 m^-3]
    Tbar : mean temperature [keV]
    f_alpha : relative alpha particle fraction

    Returns
    -------
    p_bar : mean plasma pressure [MPa]
    """

    # --- Profile factor ---
    profile_factor = 2 * (1 + nu_T) * (1 + nu_n) / (1 + nu_T + nu_n)

    # --- Convert inputs to SI units ---
    # n_bar * 1e20 → electron density [m^-3]
    # Tbar * E_ELEM * 1e3 → temperature [J]
    # Divide by 1e6 → convert Pa to MPa
    p_bar = (
        profile_factor
        * (n_bar * 1e20)
        * (Tbar * E_ELEM * 1e3)
        / 1e6
    )

    return p_bar

def f_beta_T(pbar_MPa, B0):
    """
    
    Calculate the toroidal plasma beta.

    The normalized ratio of the plasma pressure and the toroidal magnetic pressure,
    representing the 'efficiency' of the toroidal confinement.

    Parameters
    ----------
    pbar_MPa : Volume‐averaged plasma pressure [MPa]
    B0 : Central toroidal magnetic field [T]

    Returns
    -------
    beta_T : Toroidal beta (dimensionless)
    
    """
    # Convert pressure from MPa to Pa
    pbar = pbar_MPa * 1e6
    
    beta_T = 2 * μ0 * pbar / B0**2
    
    return beta_T


def f_beta_P(a, κ, pbar_MPa, Ip_MA):
    """

    Calculates the poloidal beta from the volume-averaged plasma pressure (pbar).

    Parameters
    ----------
    a : Plasma minor radius [m]
    kappa : Plasma elongation
    pbar_MPa : Volume-averaged plasma pressure [MPa]
    Ip_MA : Plasma current [MA]

    Returns
    -------
    beta_P : Poloidal beta (dimensionless)

    Note
    ----
    The average poloidal magnetic field B_pol is not explicitly entered,
    but is estimated indirectly via Ampere's law: B_pol ≈ μ₀ * I_p / L
    where L is a characteristic length representing an effective perimeter
    of the plasma cross-section. This approximation allows relating 
    the magnetic confinement to the plasma current without solving 
    the MHD equilibrium.
    
    """

    # Unit conversions to SI
    pbar_SI = pbar_MPa * 1e6    # [MPa] → [Pa] (N/m²)
    Ip_SI = Ip_MA * 1e6         # [MA] → [A]
    
    # Characteristic poloidal circumference (ellipse approximation)
    # L ≈ π√(2(a² + (κa)²)) for an ellipse with semi-axes a and κa
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta formula
    beta_P = (2 * L**2 * pbar_SI) / (μ0 * Ip_SI**2)
    
    return beta_P

def f_beta(beta_P, beta_T):
    """
    Calculate the total-field plasma beta via harmonic mean.

    Only valid if B0^2 = B_T^2 + B_P^2, i.e., when combining orthogonal toroidal
    and poloidal field contributions to the total magnetic pressure.

    Parameters
    ----------
    beta_P : Poloidal beta (dimensionless).
    beta_T : Toroidal beta (dimensionless).

    Returns
    -------
    beta : Total-field beta (dimensionless)
        
    """
    beta = 1.0 / ((1.0 / beta_P) + (1.0 / beta_T))
    
    return beta


def f_beta_N(beta, a, B0, Ip_MA):
    """
    Calculate the normalized plasma beta.

    The beta normalized to plasma geometry and current, often quoted in percent.

    Parameters
    ----------
    beta : float
        Total-field beta (dimensionless).
    a : float
        Plasma minor radius in meters [m].
    B0 : float
        Reference magnetic field in tesla [T].
    Ip_MA : float
        Plasma current in mega‐amperes [MA].

    Returns
    -------
    beta_N : Normalized beta in percent [%]
        
    """
    
    beta_N = beta * a * B0 / Ip_MA * 100
    
    return beta_N


def f_Gamma_n(a, P_fus, R0, κ):
    """
    
    Estimate the neutron flux
    
    Parameters
    ----------
    a : Minor radius [m]
    P_fus : The Fusion power [MW]
    R0 : Major radius [m]
    κ : Elongation
        
    Returns
    -------
    Gamma_n : The neutron flux [MW/m²]
    
    """
    
    Gamma_n = (E_N * P_fus / (E_ALPHA + E_N)) / (4*np.pi**2*R0*a*np.sqrt((1 + κ**2)/2))
    
    return Gamma_n

def f_nG(Ip, a):
    """
    
    Calculation of the Greenwal density limit
    
    Parameters
    ----------
    a : Minor radius [m]
    Ip : Plasma current [MA]
        
    Returns
    -------
    nG : The Greenwald fraction [1e20p/m^3]
    
    """
    nG = Ip / (np.pi * a**2)
    
    return nG

def f_qstar(a, B0, R0, Ip, κ):
    """
    
    Calculation of qstar, the kink safety factor (see Freidberg et al. PoP 2015, eq. 30)
    
    Parameters
    ----------
    a  : Minor radius [m]
    B0 : Central magnetic field [T]
    R0 : Major radius [m]
    Ip : Plasma current [MA]
    κ  : Elongation of the LCFS
        
    Returns
    -------
    qstar
    
    """
    
    qstar = (np.pi * a**2 * B0 * (1 + κ**2)) / (μ0 * R0 * Ip*1e6)
    
    return qstar

def f_cost(a,b,c,d,R0,κ,P_fus):
    """
    
    Calculation of the 'cost' parameter
    For now it is just the sum of the volume of the Breeding Blanket, TF coil and CS coil divided by the gain factor Q
    To see as an indicator to compare designs, the value in itself does not mean so much
    
    Parameters
    ----------
    a : Minor radius [m]
    b : Thickness of the First Wall+ the Breeding Blanket+ The Neutron shield+ The Vacuum Vessel + Gaps [m]
    c : Thickness of the TF coil
    d : Thickness of the CS coil
    R0 : Major radius [m]
    κ : Elongation
    P_fus : Fusion Power [MW]
        
    Returns
    -------
    cost : Cost parameter [m^3]
    
    """
    V_BB = 2*(b*2*np.pi*((R0+a+b)**2-(R0-a-b)**2))+(4*κ*a*np.pi)*((R0-a)**2+(R0+a+b)**2-(R0-a-b)**2-(R0+a)**2) # Cylindrical BB model
    V_TF = 8*np.pi*(R0-a-b-(c/2))*c*((κ+1)*a+(2*b)+c) # Rectangular TF model coil
    V_CS = 2*np.pi*((R0-a-b-c)**2-(R0-a-b-c-d)**2)*(2*(a*κ+b+c)) # h approx to 2*(a*κ+b+c) and cylindrical model
    
    cost = (V_BB + V_TF + V_CS) / P_fus
    return cost

def f_P_sep(P_fus, P_CD):
    """
    Calculate the separatrix power (P_sep) based on the given fusion power (P_fus),
    current drive power (P_CD), alpha particle energy (E_ALPHA), and neutron energy (E_N).

    Parameters:
    P_fus (float): Fusion power in megawatts (MW)
    P_CD (float): Current drive power in megawatts (MW)
    E_ALPHA (float): Energy of alpha particles in megaelectronvolts (MeV)
    E_N (float): Energy of neutrons in megaelectronvolts (MeV)

    Returns:
    float: Separator power (P_sep) in megawatts (MW)
    
    """
    P_sep = P_CD + (P_fus * E_ALPHA / (E_ALPHA + E_N))
    
    return P_sep

def f_heat_D0FUS(R0, P_sep):
    """
    
    Calculation of the heat parameter (robust version)
    
    Parameters
    ----------
    B0 : The central magnetic field [T]
    R0 : Major radius [m]
    P_fus : The Fusion power [MW]
        
    Returns
    -------
    heat : heat parameter [MW/m]
    
    """
    heat = P_sep / R0
    
    return heat

def f_heat_par(R0, B0, P_sep):
    """
    
    Calculation of the parralel heat parameter as defined in Freidberg paper
    
    Parameters
    ----------
    B0 : The central magnetic field [T]
    R0 : Major radius [m]
    P_fus : The Fusion power [MW]
        
    Returns
    -------
    heat : heat parameter [MW/m]
    
    """
    heat =  P_sep * B0 / R0
    return heat

def f_heat_pol(R0, B0, P_sep, a, q95):
    """
    
    Calculation of the poloidal heat parameter as defined in Siccinion 2019
    
    Parameters
    ----------
    B0 : The central magnetic field [T]
    R0 : Major radius [m]
    P_fus : The Fusion power [MW]
        
    Returns
    -------
    heat : heat parameter [MW/m]
    
    """
    A = R0 / a
    heat =  (P_sep * B0) / (q95 * R0 * A * R0) 
    return heat

def f_Bpol(q95, B_tor, a, R0):
    """
    Calculate the poloidal magnetic field B_pol from the safety factor q_{95}

    Approximation taken from Wesson 2004:
      q95 = (a * B_tor) / (R * B_pol)
    Implies
      B_pol = (a * B_tor) / (R * q95)

    Parameters
    ----------
    q95   : Safety factor
    B_tor : Toroidal magnetic field on the axis (T)
    a     : Minor radius (m)
    R0     : Major radius (m)

    Returns
    -------
    B_pol : Poloidal magnetic field (T)
    
    """
    B_pol = (a * B_tor) / (R0 * q95)
    
    return B_pol

def f_heat_PFU_Eich(P_sol, B_pol, R, eps, theta_deg):
    """
    Calculate divertor heat flux using the Eich scaling law.
    
    Estimates the heat load on Plasma-Facing Units (PFU) in the divertor region
    based on empirical multi-machine scaling. This approximation provides quick
    estimates but should be validated with dedicated edge transport codes (SOLPS, UEDGE).
    
    Parameters
    ----------
    P_sol : float
        Power crossing the separatrix into the Scrape-Off Layer (SOL) [MW]
    B_pol : float
        Poloidal magnetic field at the outer midplane [T]
    R : float
        Major radius [m]
    eps : float
        Inverse aspect ratio (a/R)
    theta_deg : float
        Grazing angle of magnetic field lines at divertor target [degrees]
        Typical values: 1-5° for vertical targets
    
    Returns
    -------
    lambda_q_m : float
        SOL power decay length (e-folding length) [m]
    q_parallel0 : float
        Peak parallel heat flux at the separatrix [MW/m²]
    q_target : float
        Heat flux incident on the divertor target [MW/m²]
    
    Notes
    -----
    The Eich scaling law (2013) for the SOL width is:
        λ_q [mm] = 1.35 * R^0.04 * B_pol^(-0.92) * ε^0.42 * P_sol^(-0.02)
    
    The parallel heat flux is calculated assuming toroidal symmetry:
        q_∥0 = P_sol / (2πR * λ_q)
    
    The target heat flux accounts for the grazing angle:
        q_target = q_∥0 * sin(θ)
    
    **Warning**: This is a simplified 0D approximation. Actual divertor design
    requires detailed edge plasma modeling including:
    - Radial transport and recycling (SOLPS-ITER)
    - Impurity radiation
    - Detachment physics
    - 3D effects (ELMs, RMPs)
    
    References
    ----------
    T. Eich et al., "Scaling of the tokamak near the scrape-off layer H-mode 
    power width and implications for ITER," Nuclear Fusion 53 (2013) 093031
    
    """
    
    # Convert grazing angle to radians
    theta = np.deg2rad(theta_deg)
    
    # Eich scaling: SOL power decay length [mm]
    lambda_q_mm = 1.35 * R**0.04 * B_pol**(-0.92) * eps**0.42 * P_sol**(-0.02)
    
    # Convert to meters
    lambda_q_m = lambda_q_mm * 1e-3
    
    # Peak parallel heat flux at the separatrix [MW/m²]
    # Assumes power spreads over wetted area 2πR × λ_q
    q_parallel0 = P_sol / (2 * np.pi * R * lambda_q_m)
    
    # Heat flux incident on divertor target [MW/m²]
    # Projection factor accounts for grazing angle geometry
    q_target = q_parallel0 * np.sin(theta)
    
    return lambda_q_m, q_parallel0, q_target

def f_tauE(pbar, V, P_Alpha, P_Aux, P_Ohm, P_Rad):
    """
    
    Calculation of the confinement time from the power balance
    
    Parameters
    ----------
    pbar : The mean pressure [MPa]
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    P_Alpha : The Alpha power [MW]
    P_Aux : The Auxilary power [MW]
    P_Ohm : The Ohmic power [MW]
        
    Returns
    -------
    tauE : Confinement time [s]
    
    """
    
    # conversion en SI
    p_Pa = pbar * 1e6
    P_total_W = (P_Alpha + P_Aux + P_Ohm - P_Rad) * 1e6

    if P_total_W <= 0:
        return np.nan

    W_th_J = 3/2 * p_Pa * V

    tauE_s = W_th_J / P_total_W
    
    return tauE_s

def f_P_alpha(P_fus, E_ALPHA, E_N):
    """
    
    Calculation of the alpha power
    
    Parameters
    ----------
    P_fus : The Fusion power [MW]
    E_ALPHA : Alpha energy [J]
    E_N : Neutron energy [J]
        
    Returns
    -------
    P_Alpha : The Alpha power [MW]
    
    """
    
    P_Alpha = P_fus * E_ALPHA / (E_ALPHA + E_N)
    
    return P_Alpha
        
def f_Ip(tauE, R0, a, κ, δ, nbar, B0, Atomic_mass, P_Alpha, P_Ohm, P_Aux, P_rad, H, C_SL,
         alpha_delta,alpha_M,alpha_kappa,alpha_epsilon, alpha_R,alpha_B,alpha_n,alpha_I,alpha_P):
    """
    
    Calculation of the plasma current using a tau_E scaling law
    
    Parameters
    ----------
    tauE : Minor radius [m]
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    n_bar : The mean electronic density [1e20p/m^3]
    B0 : The central magnetic field [T]
    Atomic_mass : The mean atomic mass [AMU]
    P_Alpha : The Alpha power [MW]
    P_Aux : The Auxilary power [MW]
    P_Ohm : The Ohmic power [MW]
        
    Returns
    -------
    Ip : Plasma current [MA]
    
    """
    
    P = P_Alpha + P_Ohm + P_Aux - P_rad
    Epsilon = a/R0
    
    Denominateur = H* C_SL * R0**alpha_R * Epsilon**alpha_epsilon * κ**alpha_kappa * (nbar*10)**alpha_n * B0**alpha_B * Atomic_mass**alpha_M * P**alpha_P * (1 + δ)**alpha_delta
    inv_cI  = 1./alpha_I
    
    Ip = ((tauE/Denominateur)**inv_cI) # in MA
    
    return Ip

def f_Freidberg_Ib(R0, a, κ, pbar, Ip):
    """
    
    Calculation of the bootstrap current using the Freidberg calculations
    
    Parameters
    ----------
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    p_bar : The mean pressure [MPa]
    Ip : Plasma current [MA]
        
    Returns
    -------
    Ib : Bootstrap current [MA]
    
    """
    
    # fonction b_theta(rho)
    def f_btheta(rho):
        alpha = 2.53
        num = (1 + alpha - (alpha* rho**(9. / 4.))) * np.exp(alpha* rho**(9. / 4.)) - 1 - alpha
        denom = rho * (np.exp(alpha) - 1 - alpha)
        return num / denom

    # intégrande de l'intégrale sur rho
    def integrand(rho):
        b_theta = f_btheta(rho)
        return rho**(5. / 2.) * np.sqrt(1 - rho**2) / b_theta

    # Calcul l'intégrale de 0 à 1
    integral, error = quad(integrand, 0, 1)
    
    # Calcul du terme numérateur et dénominateur pour Ib
    num = 268 * a**(5. / 2.) * κ**(5. / 4.) * pbar * integral
    denom = μ0 * np.sqrt(R0) * Ip
    
    # Calcul de Ib
    Ib = num / denom / 1e6
    return Ib

if __name__ == "__main__":
    """
    Validation of Freidberg bootstrap current formula against reference cases.
    
    Tests the f_Freidberg_Ib function against two published tokamak designs:
    1. Freidberg textbook example
    2. ARC (Affordable Robust Compact) reactor design
    """
    
    print("="*70)
    print("Bootstrap Current Calculation - Validation Test")
    print("="*70)
    
    # Test Case 1: Freidberg textbook example
    # Reference: Freidberg, "Plasma Physics and Fusion Energy" (2007)
    print("\n[Test 1] Freidberg Textbook Case")
    print("-" * 70)
    
    # Freidberg parameters
    R_Fried = 5.34      # Major radius [m]
    a_Fried = 1.34      # Minor radius [m]
    κ_Fried = 1.7       # Elongation
    eps_Fried = 0.76    # Inverse aspect ratio
    Tbar_Fried = 14.3   # Volume-averaged temperature [keV]
    
    Ib_ref_Fried = 6.3  # Reference bootstrap current [MA]
    Ib_calc_Fried = f_Freidberg_Ib(R_Fried, a_Fried, κ_Fried, eps_Fried, Tbar_Fried)
    
    print(f"  Expected:   I_bs = {Ib_ref_Fried} MA")
    print(f"  Calculated: I_bs = {Ib_calc_Fried:.1f} MA")
    print(f"  Error:      {abs(Ib_calc_Fried - Ib_ref_Fried)/Ib_ref_Fried*100:.1f}%")
    
    print("="*70)

def calculate_CB(nu_J, nu_p):
    """
    
    Numerically calculates the coefficient C_B(nu_J, nu_p) according to the integral equation (35)
    from the following article:
    D.J. Segal, A.J. Cerfon, J.P. Freidberg, "Steady state versus pulsed tokamak reactors",
    Nuclear Fusion, 61(4), 045001, 2021.

    Parameters
    ----------
    nu_J : Current profile parameter
    nu_p : Pressure profile parameter

    Returns
    -------
    CB : Numerical value of the coefficient C_B
    
    """
    def integrand(x):
        """
        Integrand function of equation (35)
        """
        polynomial = (1 + (1 - 3 * nu_J) * x + nu_J * x**2)**2
        return x**(1/4) * (1 - x)**(nu_p - 1) * polynomial

    # Calculate the integral
    integral, _ = quad(integrand, 0, 1)

    # Final coefficient
    CB = integral / (1 - nu_J)**2
    return CB


def f_Segal_Ib(nu_n, nu_T, epsilon, kappa, n20, Tk, R0, I_M):
    """
    
    Source: Segal, D. J., Cerfon, A. J., & Freidberg, J. P. (2021).
    Steady state versus pulsed tokamak reactors. Nuclear Fusion, 61(4), 045001.

    Calculates the bootstrap current fraction f_B

    Parameters :
    ----------
    nu_n : Density profile parameter
    nu_T : Temperature profile parameter
    nu_J : Current profile parameter
    epsilon : Inverse aspect ratio (a/R0)
    kappa : Elongation
    n20 : Average density [10^20 m^-3]
    Tk : Average temperature [keV]
    R0 : Major radius [m]
    I_M : Plasma current [MA]

    Returns:
    ----------
    I_b : Bootstrap Current [MA]
    
    """
    nu_p = nu_n + nu_T
    nu_J = 0.453 - 0.1 * (nu_p - 1.5)  # Eq 36 Source

    # Calculate C_B
    CB = calculate_CB(nu_J, nu_p)

    # Calculate K_b (equation A15)
    K_b = 0.6099 * (1 + nu_n) * (1 + nu_T) * (nu_n + 0.054 * nu_T)
    K_b *= (epsilon ** 2.5) * (kappa ** 1.27) * CB

    # Calculate f_B (equation 34)
    numerator = K_b * n20 * Tk * R0**2
    denominator = I_M**2
    f_B = numerator / denominator

    # Bootstrap Current
    I_b = f_B * I_M

    return I_b

if __name__ == "__main__":
    """
    Validation of Segal bootstrap current formula against ARC reactor design.
    
    Tests the f_Segal_Ib function against the published ARC (Affordable Robust Compact)
    reactor design parameters.
    """
    
    print("="*70)
    print("Bootstrap Current Calculation - Segal Formula Validation")
    print("="*70)
    
    # Test Case: ARC reactor design
    # Reference: Sorbom et al., "ARC: A compact, high-field, fusion nuclear 
    #            science facility..." Fusion Eng. Design 100 (2015) 378-405
    print("\n[Test] ARC Reactor Design")
    print("-" * 70)
    
    # ARC parameters
    nu_n = 0.385        # Density profile exponent
    nu_T = 0.929        # Temperature profile exponent
    delta = 0.34        # Triangularity
    κ = 1.84            # Elongation
    Zeff = 1.3          # Effective charge
    nbar_1e20 = 14      # Volume-averaged density [10²⁰ m⁻³]
    R = 3.3             # Major radius [m]
    Tbar = 7.8          # Volume-averaged temperature [keV]
    
    # Expected value from reference
    Ib_ref_ARC = 5.0    # Reference bootstrap current [MA] (approximate from Segal)
    
    # Calculate bootstrap current
    Ib_calc_ARC = f_Segal_Ib(nu_n, nu_T, delta, κ, Zeff, nbar_1e20, R, Tbar)
    
    print(f"  Input parameters:")
    print(f"    nu_n  = {nu_n}")
    print(f"    nu_T  = {nu_T}")
    print(f"    δ     = {delta}")
    print(f"    κ     = {κ}")
    print(f"    Z_eff = {Zeff}")
    print(f"    n̄     = {nbar_1e20} × 10²⁰ m⁻³")
    print(f"    R     = {R} m")
    print(f"    T̄     = {Tbar} keV")
    print(f"\n  Result:")
    print(f"    I_bs (Segal) = {Ib_calc_ARC:.1f} MA")
    print(f"    Reference    ≈ {Ib_ref_ARC} MA")
    
    # Calculate relative difference
    error = abs(Ib_calc_ARC - Ib_ref_ARC) / Ib_ref_ARC * 100
    print(f"    Difference   = {error:.1f}%")
    
    # Validation status
    print("\n" + "="*70)
    if error < 10:
        print("✓ Validation PASSED: Formula reproduces ARC design within 10%")
    else:
        print("✗ Validation WARNING: Error exceeds 10% threshold")
    
    print("\nNote: Segal formula is more detailed than Freidberg, accounting for")
    print("      profile shapes (nu_n, nu_T) and plasma shaping (δ, κ).")
    print("="*70)

def f_etaCD(a, R0, B0, nbar, Tbar, nu_n, nu_T):
    """
    
    Compute the efficienty of LHCD
    
    Parameters
    ----------
    a : Minor radius [m]
    R0 : Major radius [m]
    B0 : The central magnetic field [T]
    n_bar : The mean electronic density [1e20p/m^3]
    T_bar : The mean temperature [keV]
    nu_n : Density profile parameter 
    nu_T : Temperature profile parameter
        
    Returns
    -------
    eta_CD : Current drive efficienty [MA/MW-m²]
    
    """
    rho_m = 0.8
    # Calcul de la température locale et de la densité locale
    n_loc = f_nprof(nbar, nu_n, rho_m)
    eps = a / R0
    B_loc = B0 / (1 + eps * rho_m)
    omega_ce = E_ELEM * B_loc / M_E # Cyclotron frequency
    omega_pe = E_ELEM * np.sqrt(n_loc*1e20 / (EPS_0 * M_E)) # Plasma frequency
    # Calcul de n_parallel
    n_parall = omega_pe / omega_ce + np.sqrt(1 + (omega_pe / omega_ce)**2) * np.sqrt(3. / 4.)

    # Calcul de eta_CD
    eta_CD = 1.2 / (n_parall**2)
    return eta_CD

def f_PCD(R0, nbar, I_CD, eta_CD):
    """
    
    Estimate the Currend Drive (CD) power needed
    
    Parameters
    ----------
    a : Minor radius [m]
    R0 : Major radius [m]
    n_bar : The mean electronic density [1e20p/m^3]
    I_CD : Current drive current [MA]
        
    Returns
    -------
    P_CD : Current drive power to inject [MW]
    
    """
    P_CD = R0 * nbar * I_CD / eta_CD
    return P_CD

def f_I_Ohm(Ip, Ib, I_CD):
    """
    
    Estimate the Ohmic current
    
    Parameters
    ----------
    Ip : Plasma current [MA]
    Ib : Bootstrap current [MA]
    I_CD : Current drive current [MA]
        
    Returns
    -------
    I_Ohm : Current drive power injected [MW]
    
    """
    I_Ohm = abs(Ip - Ib - I_CD)
    return I_Ohm

def f_ICD(Ip, Ib, I_Ohm):
    """
    
    Estimate the Current drive
    
    Parameters
    ----------
    Ip : Plasma current [MA]
    Ib : Bootstrap current [MA]
    I_Ohm : Ohmic current [MA]
        
    Returns
    -------
    I_CD : Current drive power injected [MW]
    
    """
    I_CD = abs(Ip - Ib - I_Ohm)
    return I_CD

def f_I_CD(R0, nbar, eta_CD, P_CD):
    """
    
    Estimate the Currend Drive (CD) current from the CD power
    
    Parameters
    ----------
    a : Minor radius [m]
    R0 : Major radius [m]
    n_bar : The mean electronic density [1e20p/m^3]
    P_CD : Current drive power injected [MW]
        
    Returns
    -------
    I_CD : Current drive current [MA]
    
    """
    I_CD = P_CD*eta_CD / (R0*nbar)
    return I_CD


def f_PLH(eta_RF, f_RP, P_CD):
    """
    
    Estimate the Lower Hybrid Electrical Power
    
    Parameters
    ----------
    eta_RF : conversion efficiency from wall power to klystron
    f_RP : fraction of klystron power absorbed by plasma
    P_CD : Current drive power injected [MW]
        
    Returns
    -------
    P_LH : Electrical Power estimated to drive such a current [MW]
    
    """
    P_LH = (1/eta_RF)*(1/f_RP)*P_CD
    return P_LH

def f_P_Ohm(I_Ohm, Tbar, R0, a, kappa):
    """
    Estimate the Ohmic heating power in a tokamak plasma.
    
    Ohmic heating results from the resistive dissipation of the plasma current.
    At high temperatures, the resistivity decreases as T^(-3/2) (Spitzer scaling),
    making Ohmic heating ineffective for reactor-grade plasmas.
    
    Parameters
    ----------
    I_Ohm : float
        Ohmic plasma current [MA]
    Tbar : float
        Volume-averaged electron temperature [keV]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    kappa : float
        Plasma elongation
        
    Returns
    -------
    P_Ohm : float
        Ohmic heating power [MW]
    
    Notes
    -----
    The calculation follows three steps:
    
    1. Spitzer resistivity (classical collisional transport):
       η [Ω·m] = 2.8×10⁻⁸ / T^(3/2)  [with T in keV]
    
    2. Effective plasma resistance (approximate):
       R_eff [Ω] = η * (2πR₀) / (πa²κ) = η * (2R₀) / (a²κ)
       
       This approximation assumes:
       - Toroidal current path length ≈ 2πR₀
       - Effective cross-sectional area ≈ πa²κ
    
    3. Ohmic power dissipation:
       P_Ohm = R_eff * I²
    
    **Important**: This is a simplified 0D estimate. Actual Ohmic power depends on:
    - Current density profile j(r)
    - Temperature profile T(r)
    - Neoclassical corrections (trapped particles)
    - Impurity content (Z_eff)
    
    References
    ----------
    Spitzer, L., & Härm, R. (1953). "Transport phenomena in a completely 
    ionized gas." Physical Review, 89(5), 977-981.
    """
    
    # Spitzer resistivity [Ω·m]
    # Classical collisional resistivity for a fully ionized plasma
    eta = 2.8e-8 / (Tbar**1.5)
    
    # Effective plasma resistance [Ω]
    # Approximates the plasma as a toroidal conductor with:
    #   - Current path length: 2πR₀
    #   - Cross-sectional area: πa²κ
    R_eff = eta * (2 * R0) / (a**2 * kappa)
    
    # Ohmic heating power [MW]
    # Convert current from MA to A, then power from W to MW
    I_Ohm_A = I_Ohm * 1e6           # [MA] → [A]
    P_Ohm_W = R_eff * I_Ohm_A**2    # [W]
    P_Ohm = P_Ohm_W * 1e-6          # [W] → [MW]
    
    return P_Ohm

def f_Q(P_fus,P_CD,P_Ohm):
    """
    
    Calculate the plasma amplification factor Q
    
    Parameters
    ----------
    P_fus = Fusion power [MW]
    P_CD = Current drive power [MW]
    P_Ohm = Ohmic power [MW]
        
    Returns
    -------
    Q : Plasma amplification factor
    
    """
    Q = P_fus/(P_CD + P_Ohm)
    return Q

def P_Thresh_Martin(n_bar, B0, a, R0, κ, Atomic_mass):
    """
    
    Calculate the L-H transition power from L to H mode
    Source : MARTIN, Y. R., TAKIZUKA, Tomonori, et al. Power requirement for accessing the H-mode in ITER. In : Journal of Physics: Conference Series. IOP Publishing, 2008. p. 012033.
    Database from 2008 , exponent on S free and exponent on M fixed
    Incertitudes titanesques, estimation pour ITER : 85MW nécessaire mais [45-160] pour être dans un interval de confiance à 95% : RMSE = 30%
    
    Parameters
    ----------
    n_bar : The mean electronic density [1e20p/m^3]
    B0 : The central magnetic field [T]
    a : Minor radius [m]
    R0 : Major radius [m]
    κ : Elongation
    Atomic_mass : Atomic mass [AMU]
    
    Returns
    -------
    P_Martin : L-H power threshold from Martin scaling [MW]
    
    """
    
    Torre_Surface = 4*np.pi**2*R0*a*((1+κ**2)/2)**(1/2)
    const_Martin = 0.0488
    exp_n = 0.717 # n in 1e20
    exp_B0 = 0.803
    exp_S = 0.941
    exp_M = 1
    
    P_Martin = const_Martin* (2/Atomic_mass**exp_M) * (n_bar ** exp_n) * (B0 ** exp_B0) * (Torre_Surface ** exp_S)
    
    return P_Martin


def P_Thresh_New_S(n_bar, B0, a, R0, κ, Atomic_mass):
    """
    
    Calculate the L-H transition power from L to H mode
    Source : E. Delabie, ITPA 2017, TC-26: L-H/H-L scaling in the presence of Metallic walls. (Not published)
    Database from 2017 and exponent on S fixed to 1 but exp_M free 
    Incertitudes comparables à Martin: RMSE = 26%
    
    Parameters
    ----------
    n_bar : The mean electronic density [1e20p/m^3]
    B0 : The central magnetic field [T]
    a : Minor radius [m]
    R0 : Major radius [m]
    κ : Elongation
    Atomic_mass : Atomic mass [AMU]
    
    Returns
    -------
    P_New_S : L-H power threshold from new scaling using S [MW]
    
    """
    
    Torre_Surface = 4*np.pi**2*R0*a*((1+κ**2)/2)**(1/2)
    const_New_S = 0.045
    exp_n = 1.08 # n in 1e20
    exp_B0 = 0.56
    exp_S = 1
    exp_M = 0.96
    
    # If one consider VT/corner configuration, to change for 1.93 
    Divertor_configuration = 1
    
    P_New_S = const_New_S*Divertor_configuration* (2/Atomic_mass**exp_M) * (n_bar ** exp_n) * (B0 ** exp_B0) * (Torre_Surface ** exp_S)
    
    return P_New_S

def P_Thresh_New_Ip(n_bar, B0, a, R0, κ, Ip, Atomic_mass):
    """
    
    Calculate the L-H transition power from L to H mode
    Source : E. Delabie, ITPA 2017, TC-26: L-H/H-L scaling in the presence of Metallic walls. (Not published)
    # Database from 2017 and exponent on S fixed to 1 but exp_M free (here =1)
    # New regression technique, trying to use Ip/a and showing lower incertitudes (still enormous) : RMSE = 21%
    
    Parameters
    ----------
    n_bar : The mean electronic density [1e20p/m^3]
    B0 : The central magnetic field [T]
    a : Minor radius [m]
    R0 : Major radius [m]
    κ : Elongation
    Ip : Plasma current [MA]
    Atomic_mass : Atomic mass [AMU]
    
    Returns
    -------
    P_New_Ip : L-H power threshold from new scaling using Ip/a [MW]
    
    """
    
    Torre_Surface = 4*np.pi**2*R0*a*((1+κ**2)/2)**(1/2)
    const_New_Ip = 0.049
    exp_n = 1.06 # n in 1e20
    exp_Ip_a = 0.65 # Ip in MA
    exp_S = 1
    exp_M = 1
    
    P_New_Ip = const_New_Ip* (2/Atomic_mass**exp_M) * (n_bar ** exp_n) * ((Ip/a) ** exp_Ip_a) * (Torre_Surface ** exp_S)
    
    return P_New_Ip

if __name__ == "__main__":
    """
    Validation of Delabie L-H transition threshold power scalings.
    
    Tests the P_Thresh_New_S and P_Thresh_New_Ip functions against reference tokamaks:
    1. ITER: Large superconducting tokamak (design phase)
    2. WEST: Medium-size superconducting tokamak (operational)
    
    Compares:
    - Martin scaling (2008) - ITER baseline
    - Delabie New_S scaling - Surface area based
    - Delabie New_Ip scaling - Plasma current based (refined)
    """
    
    print("="*70)
    print("L-H Transition Threshold Power - Delabie Scaling Validation")
    print("="*70)
    
    # Test Case 1: ITER
    # Reference: ITER Physics Basis, Nuclear Fusion 39 (1999)
    print("\n[Test 1] ITER Design Parameters")
    print("-" * 70)
    
    # ITER parameters
    nbar_ITER = 1.0             # Line-averaged density [10²⁰ m⁻³]
    B0_ITER = 5.3               # Central magnetic field [T]
    a_ITER = 2.0                # Minor radius [m]
    R0_ITER = 6.0               # Major radius [m]  
    kappa_ITER = 1.7            # Elongation
    Ip_ITER = 15.0              # Plasma current [MA]
    Atomic_mass_ITER = 2.5      # D-T mixture [AMU]
    
    P_thresh_Martin_ITER = P_Thresh_Martin(
        nbar_ITER, B0_ITER, a_ITER, R0_ITER, kappa_ITER, Atomic_mass_ITER
    )
    
    P_thresh_Delabie_S_ITER = P_Thresh_New_S(
        nbar_ITER, B0_ITER, a_ITER, R0_ITER, kappa_ITER, Atomic_mass_ITER
    )
    
    P_thresh_Delabie_Ip_ITER = P_Thresh_New_Ip(
        nbar_ITER, B0_ITER, a_ITER, R0_ITER, kappa_ITER, Ip_ITER, Atomic_mass_ITER
    )
    
    print(f"  Parameters:")
    print(f"    n̄_e   = {nbar_ITER} × 10²⁰ m⁻³")
    print(f"    B₀    = {B0_ITER} T")
    print(f"    a     = {a_ITER} m")
    print(f"    R₀    = {R0_ITER} m")
    print(f"    κ     = {kappa_ITER}")
    print(f"    I_p   = {Ip_ITER} MA")
    print(f"    M_ion = {Atomic_mass_ITER} AMU (D-T)")
    print(f"\n  Results:")
    print(f"    P_LH (Martin)         = {P_thresh_Martin_ITER:.2f} MW")
    print(f"    P_LH (Delabie New_S)  = {P_thresh_Delabie_S_ITER:.2f} MW")
    print(f"    P_LH (Delabie New_Ip) = {P_thresh_Delabie_Ip_ITER:.2f} MW (refined)")
    print(f"\n  Differences from Martin:")
    print(f"    New_S:  {abs(P_thresh_Delabie_S_ITER - P_thresh_Martin_ITER):.2f} MW "
          f"({abs(P_thresh_Delabie_S_ITER - P_thresh_Martin_ITER)/P_thresh_Martin_ITER*100:+.1f}%)")
    print(f"    New_Ip: {abs(P_thresh_Delabie_Ip_ITER - P_thresh_Martin_ITER):.2f} MW "
          f"({abs(P_thresh_Delabie_Ip_ITER - P_thresh_Martin_ITER)/P_thresh_Martin_ITER*100:+.1f}%)")
    
    # Test Case 2: WEST
    # Reference: Bucalossi et al., Fusion Eng. Design 89 (2014)
    print("\n[Test 2] WEST (Tungsten Environment Steady-State Tokamak)")
    print("-" * 70)
    
    # WEST parameters
    nbar_WEST = 0.6             # Line-averaged density [10²⁰ m⁻³]
    B0_WEST = 3.7               # Central magnetic field [T]
    a_WEST = 0.72               # Minor radius [m]
    R0_WEST = 2.4               # Major radius [m]
    kappa_WEST = 1.3            # Elongation
    Ip_WEST = 1.0               # Plasma current [MA] (typical)
    Atomic_mass_WEST = 2.0      # Deuterium [AMU]
    
    P_thresh_Martin_WEST = P_Thresh_Martin(
        nbar_WEST, B0_WEST, a_WEST, R0_WEST, kappa_WEST, Atomic_mass_WEST
    )
    
    P_thresh_Delabie_S_WEST = P_Thresh_New_S(
        nbar_WEST, B0_WEST, a_WEST, R0_WEST, kappa_WEST, Atomic_mass_WEST
    )
    
    P_thresh_Delabie_Ip_WEST = P_Thresh_New_Ip(
        nbar_WEST, B0_WEST, a_WEST, R0_WEST, kappa_WEST, Ip_WEST, Atomic_mass_WEST
    )
    
    print(f"  Parameters:")
    print(f"    n̄_e   = {nbar_WEST} × 10²⁰ m⁻³")
    print(f"    B₀    = {B0_WEST} T")
    print(f"    a     = {a_WEST} m")
    print(f"    R₀    = {R0_WEST} m")
    print(f"    κ     = {kappa_WEST}")
    print(f"    I_p   = {Ip_WEST} MA")
    print(f"    M_ion = {Atomic_mass_WEST} AMU (D)")
    print(f"\n  Results:")
    print(f"    P_LH (Martin)         = {P_thresh_Martin_WEST:.2f} MW")
    print(f"    P_LH (Delabie New_S)  = {P_thresh_Delabie_S_WEST:.2f} MW")
    print(f"    P_LH (Delabie New_Ip) = {P_thresh_Delabie_Ip_WEST:.2f} MW (refined)")
    print(f"\n  Differences from Martin:")
    print(f"    New_S:  {abs(P_thresh_Delabie_S_WEST - P_thresh_Martin_WEST):.2f} MW "
          f"({abs(P_thresh_Delabie_S_WEST - P_thresh_Martin_WEST)/P_thresh_Martin_WEST*100:+.1f}%)")
    print(f"    New_Ip: {abs(P_thresh_Delabie_Ip_WEST - P_thresh_Martin_WEST):.2f} MW "
          f"({abs(P_thresh_Delabie_Ip_WEST - P_thresh_Martin_WEST)/P_thresh_Martin_WEST*100:+.1f}%)")


def f_q95(B0, Ip, R0, a, kappa_95, delta_95):
    """
    Estimate the safety factor q at 95% normalized poloidal flux (q95).
    
    The safety factor q quantifies the helical pitch of magnetic field lines.
    q95 is a key parameter for:
    - MHD stability limits (Greenwald density, disruption avoidance)
    - Edge localized mode (ELM) behavior
    - Confinement scaling (H-mode pedestal)
    
    Parameters
    ----------
    B0 : float
        Central toroidal magnetic field (on axis) [T]
    Ip : float
        Plasma current [MA]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    kappa_95 : float
        Elongation at 95% of the Last Closed Flux Surface (LCFS)
    delta_95 : float
        Triangularity at 95% of the LCFS
    
    Returns
    -------
    q95 : float
        Safety factor at ψ_N = 0.95 (dimensionless)
    
    Notes
    -----
    This function uses the Sauter (2016) empirical formula, which accounts for:
    - Aspect ratio effects (A = R₀/a)
    - Plasma shaping (elongation κ, triangularity δ)
    - Assumes zero squareness (w07 = 1)
    
    The formula is:
        q95 = (4.1 a² B₀) / (R₀ I_p) × f_κ(κ) × f_δ(δ, A)
    
    where:
        f_κ(κ) = 1 + 1.2(κ-1) + 0.56(κ-1)²
        f_δ(δ, A) = (1 + 0.09δ + 0.16δ²)(1 + 0.45δ/A) / (1 - 0.74/A)
    
    **Alternative formulation (commented out):**
    Johner FST 2011 (used in HELIOS code):
        q95 = (2πa²B₀)/(μ₀I_pR₀) × [(1.17-0.65/A)/(1-1/A²)] × 
              [1 + κ²(1 + 2δ² - 1.2δ³)]/2
    
    The Sauter formula is preferred for its improved accuracy across a wider
    parameter range, particularly for highly shaped plasmas.
    
    References
    ----------
    O. Sauter et al., "Geometric formulas for system codes including the 
    effect of negative triangularity," Fusion Engineering and Design 112 
    (2016) 633-645.
    
    F. Johner, "HELIOS: A zero-dimensional tool for next step and reactor 
    studies," Fusion Science and Technology 59 (2011) 308-349.
    
    """
    
    # Calculate aspect ratio
    Aspect_ratio = R0 / a
    
    # Sauter (2016) formula - preferred formulation
    # Factor 4.1 includes geometric corrections and unit conversions
    q95 = (4.1 * a**2 * B0) / (R0 * Ip) * \
          (1 + 1.2*(kappa_95 - 1) + 0.56*(kappa_95 - 1)**2) * \
          (1 + 0.09*delta_95 + 0.16*delta_95**2) * \
          (1 + 0.45*delta_95 / Aspect_ratio) / \
          (1 - 0.74 / Aspect_ratio)
    
    # Alternative: Johner (2011) formula (HELIOS code)
    # Uncomment to use this formulation instead:
    # q95 = (2 * np.pi * a**2 * B0) / (μ0 * Ip*1e6 * R0) * \
    #       (1.17 - 0.65/Aspect_ratio) / (1 - 1/Aspect_ratio**2) * \
    #       (1 + kappa_95**2 * (1 + 2*delta_95**2 - 1.2*delta_95**3)) / 2
    
    return q95

if __name__ == "__main__":
    """
    Validation of q95 calculation against ITER baseline scenario.
    """
    
    print("="*70)
    print("Safety Factor q95 Calculation - Validation Test")
    print("="*70)
    
    # Test Case: ITER baseline H-mode
    print("\n[Test] ITER Baseline Scenario")
    print("-" * 70)
    
    # ITER parameters
    B0_ITER = 5.3           # Central field [T]
    Ip_ITER = 15.0          # Plasma current [MA]
    R0_ITER = 6.2           # Major radius [m]
    a_ITER = 2.0            # Minor radius [m]
    κ_ITER = 1.7            # Elongation
    δ_ITER = 0.33           # Triangularity
    
    q95_ITER = f_q95(B0_ITER, Ip_ITER, R0_ITER, a_ITER, κ_ITER, δ_ITER)
    q95_ref_ITER = 3.0      # ITER design reference
    
    print(f"  Parameters:")
    print(f"    B₀ = {B0_ITER} T")
    print(f"    I_p = {Ip_ITER} MA")
    print(f"    R₀ = {R0_ITER} m")
    print(f"    a = {a_ITER} m")
    print(f"    κ = {κ_ITER}")
    print(f"    δ = {δ_ITER}")
    print(f"\n  Result:")
    print(f"    q95 (calculated) = {q95_ITER:.2f}")
    print(f"    q95 (reference)  = {q95_ref_ITER:.2f}")
    print(f"    Error = {abs(q95_ITER - q95_ref_ITER)/q95_ref_ITER*100:.1f}%")


def f_He_fraction(n_bar, T_bar, tauE, C_Alpha, nu_T):
    """
    Estimate the helium ash (alpha particle) density fraction in the plasma.
    
    Alpha particles (He⁴⁺) are the fusion products that must be expelled to maintain
    fuel purity. Excessive helium accumulation dilutes the fuel and degrades fusion
    performance. This function calculates the equilibrium alpha fraction based on
    production (fusion reactions) and removal (confinement time) rates.
    
    Parameters
    ----------
    n_bar : float
        Volume-averaged electron density [10²⁰ m⁻³]
    T_bar : float
        Volume-averaged temperature [keV]
    tauE : float
        Energy confinement time [s]
    C_Alpha : float
        Alpha particle removal efficiency parameter (typical: 5)
        - Higher values → faster alpha removal → lower f_alpha
        - Related to pumping efficiency and divertor performance
    nu_T : float
        Temperature profile exponent: T(r) ∝ (1 - r²)^nu_T
    
    Returns
    -------
    f_alpha : float
        Alpha particle density fraction: n_α / n_e (dimensionless)
        Typical values: 0.05-0.15 (5-15%)
    
    Notes
    -----
    The calculation follows the steady-state particle balance:
    
    1. Alpha production rate ∝ n²⟨σv⟩ (fusion reactions)
    2. Alpha removal rate ∝ n_α/τ_α (particle confinement)
    3. At equilibrium: production = removal
    
    The model uses a quadratic equation (Appendix B, Sarazin):
        f_α = [C + 1 - √(2C + 1)] / (2C)
    where:
        C = n̄ ⟨σv⟩ C_α τ_E
    
    The integral ⟨σv⟩ accounts for the radial temperature profile:
        ⟨σv⟩ = ∫₀¹ σv[T(ρ)] dρ
    
    **Physical interpretation:**
    - f_α too high (>15%): Fuel dilution, Q degradation
    - f_α too low (<5%): Inefficient alpha heating
    - Optimal range: 8-12% for reactor operation
    
    References
    ----------
    Y. Sarazin et al., "Impact of scaling laws on tokamak reactor dimensioning,"
    Nuclear Fusion (year). See Appendix B for derivation.
    
    """
    
    # Integrate fusion reactivity over radial temperature profile
    # ⟨σv⟩ = ∫₀¹ σv[T(ρ)] dρ
    def integrand(rho):
        T_local = f_Tprof(T_bar, nu_T, rho)
        return f_sigmav(T_local)
    
    sigmav_avg, _ = quad(integrand, 0, 1)
    
    # Dimensionless parameter governing alpha accumulation
    # C = n̄ ⟨σv⟩ C_α τ_E
    C_equa_alpha = n_bar * 1e20 * sigmav_avg * C_Alpha * tauE
    
    # Solve quadratic equilibrium equation for alpha fraction
    # Derivation: balance production (∝ n²⟨σv⟩) with removal (∝ n_α/τ_α)
    f_alpha = (C_equa_alpha + 1 - np.sqrt(2 * C_equa_alpha + 1)) / (2 * C_equa_alpha)
    
    return f_alpha


def f_tau_alpha(n_bar, T_bar, tauE, C_Alpha, nu_T):
    """
    Estimate the alpha particle confinement time (tau_alpha).
    
    The alpha confinement time represents how long alpha particles remain
    confined before being exhausted through the divertor. It is derived
    consistently from the helium fraction equilibrium model.
    
    Parameters
    ----------
    n_bar : float
        Volume-averaged electron density [10²⁰ m⁻³]
    T_bar : float
        Volume-averaged temperature [keV]
    tauE : float
        Energy confinement time [s]
    C_Alpha : float
        Alpha particle removal efficiency parameter (typical: 5)
    nu_T : float
        Temperature profile exponent: T(r) ∝ (1 - r²)^nu_T
    
    Returns
    -------
    tau_alpha : float
        Alpha particle confinement time [s]
        Typical range: 1-10× τ_E (alphas confined longer than energy)
    
    Notes
    -----
    The alpha confinement time is related to the helium fraction by:
        τ_α = (f_α × τ_E) / C
    where:
        C = n̄ ⟨σv⟩ C_α τ_E
        f_α = alpha fraction from f_He_fraction()
    
    **Physical interpretation:**
    - τ_α ≫ τ_E: Alphas well-confined, risk of accumulation
    - τ_α ~ τ_E: Good balance, optimal helium removal
    - τ_α ≪ τ_E: Over-pumping, loss of alpha heating
    
    The relationship τ_α/τ_E is a key reactor design parameter.
    
    References
    ----------
    Derived from Y. Sarazin et al., "Impact of scaling laws on tokamak 
    reactor dimensioning," Appendix B.

    """
    
    # Integrate fusion reactivity over radial temperature profile
    def integrand(rho):
        T_local = f_Tprof(T_bar, nu_T, rho)
        return f_sigmav(T_local)
    
    sigmav_avg, _ = quad(integrand, 0, 1)
    
    # Dimensionless parameter (same as in f_He_fraction)
    C_equa_alpha = n_bar * 1e20 * sigmav_avg * C_Alpha * tauE
    
    # Alpha fraction (equilibrium solution)
    f_alpha = (C_equa_alpha + 1 - np.sqrt(2 * C_equa_alpha + 1)) / (2 * C_equa_alpha)
    
    # Alpha confinement time from particle balance
    # Relationship: n_α/τ_α = production rate → τ_α = (f_α × τ_E) / C
    tau_alpha = (f_alpha * tauE) / C_equa_alpha
    
    return tau_alpha


if __name__ == "__main__":
    """
    Validation of helium ash fraction predictions for ITER and EU-DEMO.
    """
    
    print("="*70)
    print("Helium Ash Fraction Calculation - Validation Test")
    print("="*70)
    
    # Common parameters
    C_Alpha = 5         # Standard removal efficiency parameter
    nu_T = 1.0          # Parabolic temperature profile
    
    # Test Case 1: ITER Q=10 scenario
    print("\n[Test 1] ITER Q=10 Baseline Scenario")
    print("-" * 70)
    
    n_bar_ITER = 1.0        # Density [10²⁰ m⁻³]
    T_bar_ITER = 9.0        # Temperature [keV]
    tauE_ITER = 3.1         # Confinement time [s]
    
    f_alpha_ITER = f_He_fraction(n_bar_ITER, T_bar_ITER, tauE_ITER, C_Alpha, nu_T)
    tau_alpha_ITER = f_tau_alpha(n_bar_ITER, T_bar_ITER, tauE_ITER, C_Alpha, nu_T)
    
    print(f"  Parameters:")
    print(f"    n̄     = {n_bar_ITER} × 10²⁰ m⁻³")
    print(f"    T̄     = {T_bar_ITER} keV")
    print(f"    τ_E   = {tauE_ITER} s")
    print(f"    C_α   = {C_Alpha}")
    print(f"\n  Results:")
    print(f"    f_α   = {f_alpha_ITER:.3f} ({f_alpha_ITER*100:.1f}%)")
    print(f"    τ_α   = {tau_alpha_ITER:.2f} s")
    print(f"    τ_α/τ_E = {tau_alpha_ITER/tauE_ITER:.2f}")
    
    # Test Case 2: EU-DEMO
    print("\n[Test 2] EU-DEMO Baseline Scenario")
    print("-" * 70)
    
    n_bar_DEMO = 1.2        # Density [10²⁰ m⁻³]
    T_bar_DEMO = 12.5       # Temperature [keV]
    tauE_DEMO = 4.6         # Confinement time [s]
    
    f_alpha_DEMO = f_He_fraction(n_bar_DEMO, T_bar_DEMO, tauE_DEMO, C_Alpha, nu_T)
    tau_alpha_DEMO = f_tau_alpha(n_bar_DEMO, T_bar_DEMO, tauE_DEMO, C_Alpha, nu_T)
    
    f_alpha_ref_DEMO = 0.16  # Reference: 16%
    
    print(f"  Parameters:")
    print(f"    n̄     = {n_bar_DEMO} × 10²⁰ m⁻³")
    print(f"    T̄     = {T_bar_DEMO} keV")
    print(f"    τ_E   = {tauE_DEMO} s")
    print(f"    C_α   = {C_Alpha}")
    print(f"\n  Results:")
    print(f"    f_α (calculated) = {f_alpha_DEMO:.3f} ({f_alpha_DEMO*100:.1f}%)")
    print(f"    f_α (reference)  = {f_alpha_ref_DEMO:.3f} ({f_alpha_ref_DEMO*100:.1f}%)")
    print(f"    Error = {abs(f_alpha_DEMO - f_alpha_ref_DEMO)/f_alpha_ref_DEMO*100:.1f}%")
    print(f"    τ_α   = {tau_alpha_DEMO:.2f} s")
    print(f"    τ_α/τ_E = {tau_alpha_DEMO/tauE_DEMO:.2f}")

def f_surface_premiere_paroi(kappa, R0, a):
    """
    
    Calculate the surface area of the first wall in a tokamak
    from the elongation (kappa), major radius (R0), and minor radius (a)

    Parameters
    ----------
    kappa : Elongation (dimensionless)
    R0 : Major radius [m]
    a : Minor radius [m]

    Returns
    -------
    S : Surface area [m²]
        
    """
    # Approximation of the ellipse perimeter (plasma cross-section) by Ramanujan
    Pe = math.pi * a * (3 * (1 + kappa) - math.sqrt((3 + kappa) * (1 + 3 * kappa)))
    # First wall surface area
    S = 2 * math.pi * R0 * Pe
    return S

def f_P_elec(P_fus, P_LH, eta_T, eta_RF):
    """
    
    Calculate the net electrical power P_elec
    
    Parameters
    ----------
    P_fus : Fusion power [MW]
    P_LH : LHCD power [MW]
    eta_T : Conversion efficienty from fusion power to electrical power
    eta_RF : Conversion efficienty from wall to klystron

    Returns
    -------
    P_elec : Net electrical power [MW]
    
    """
    P_th = P_fus * E_F / (E_ALPHA + E_N)
    P_elec = eta_T * P_th - P_LH / eta_RF
    return P_elec

def f_W_th(n_avg, T_avg, volume):
    """
    
    Calculate the total thermal energy W_th of a plasma assuming 
    n_i = n_e and T_i = T_e.

    Parameters
    ----------
    n_avg : Average density (electronic and ionic) [1e20 m⁻³]
    T_avg : Average temperature (electronic and ionic) [keV]
    volume : Plasma volume [m³]

    Returns
    -------
    W_th : Thermal energy W_th [Joules]
    
    """
    
    n_m3 = n_avg * 1e20  # Convert n to m⁻³
    T_eV = T_avg * 1e3   # Convert T to eV
    W_th = 3 * n_m3 * T_eV * volume * E_ELEM
    
    return W_th

def f_P_1rst_wall_Hmod(P_sep_solution, P_CD_solution, Surface_solution):
    """
    
    Calculate the power deposited on the first wall in H-mode

    Parameters
    ----------
    P_sep_solution : Power leaving the plasma [MW]
    P_CD_solution : Power injected for current drive [MW]
    Surface_solution : Surface area of the first wall [m²]

    Returns
    -------
    P_1rst_wall_Hmod : Surface power density on the first wall in H-mode [MW/m²]
    
    """
    
    P_1rst_wall_Hmod = (P_sep_solution - P_CD_solution) / Surface_solution
    
    return P_1rst_wall_Hmod

def f_P_1rst_wall_Lmod(P_sep_solution, Surface_solution):
    """
    
    Calculate the power deposited on the first wall in L-mode

    Parameters
    ----------
    P_sep_solution : Power leaving the plasma [MW]
    Surface_solution : Surface area of the first wall [m²]

    Returns
    -------
    P_1rst_wall_Lmod : Surface power density on the first wall in L-mode [MW/m²]
        
    """
    
    P_1rst_wall_Lmod = P_sep_solution / Surface_solution
    
    return P_1rst_wall_Lmod

def f_P_synchrotron(T0_keV, R, a, Bt, ne0, kappa, nu_n, nu_T, r):
    """
    Calculate the total synchrotron radiation power (in MW) using the
    improved formulation from Albajar et al. (2001).
    
    Parameters
    ----------
    T0_keV : float
        Central electron temperature [keV]
    R : float
        Major radius [m]
    a : float
        Minor radius [m]
    Bt : float
        Toroidal magnetic field [T]
    ne0 : float
        Central electron density [10^20 m⁻³]
    kappa : float
        Plasma vertical elongation
    nu_n : float
        Density profile exponent: n(rho) = n0 * (1 - rho^2)^nu_n
    nu_T : float
        Temperature profile exponent: T(rho) = T0 * (1 - rho^2)^nu_T
    r : float
        Wall reflection coefficient (typically 0.6-0.9)
        
    Returns
    -------
    P_syn : float
        Total synchrotron radiation power [MW]
        
    References
    ----------
    Albajar, F., Johner, J., & Granata, G. (2001). 
    Nuclear Fusion, 41(6), 665.
    """
    A = R / a
    
    # Opacity parameter (Eq. 7)
    pa0 = 6.04e3 * a * ne0 / Bt
    
    # Profile factor K (Eq. 13)
    K_numer = (nu_n + 3.87*nu_T + 1.46)**(-0.79) * (1.98 + nu_T)**1.36 * nu_T**2.14
    K_denom = (nu_T**1.53 + 1.87*nu_T - 0.16)**1.33
    K = K_numer / K_denom
    
    # Aspect ratio correction G (Eq. 15)
    G = 0.93 * (1 + 0.85 * math.exp(-0.82 * A))
    
    # Main expression (Eq. 16)
    term1 = 3.84e-8 * (1 - r)**0.5
    term2 = R * a**1.38 * kappa**0.79 * Bt**2.62 * ne0**0.38
    term3 = T0_keV * (16 + T0_keV)**2.61
    term4 = (1 + 0.12 * T0_keV / pa0**0.41)**(-1.51)
    
    return term1 * term2 * term3 * term4 * K * G

def f_P_bremsstrahlung(V, n_e, T_e, Z_eff, R, a):
    """
    Note : Under developement
    
    Calculate the total Bremsstrahlung power (in MW)

    Parameters
    -------
    n_e : Electron density [10^20 m⁻³]
    T_e : Electron temperature [keV]
    Z_eff : Effective charge
    V : Plasma volume [m³]
    
    Returns
    -------
    P_Brem : Bremsstrahlung power [MW]

    Assumptions:
        - Fully ionized plasma
        - Radial shape factor g_r ≈ 1 (flat profiles)

    Sources
    -------
    NRL Plasma Formulary, 2022 edition, section on bremsstrahlung radiation.
    Wesson, J., "Tokamaks", 3rd ed., Oxford University Press, p.228
    
    """
    
    P_Brem = 5.35e3 * Z_eff**2 * n_e**2 * T_e**(1/2) * V
    
    return P_Brem / 1e6

def f_P_line_radiation(V, n_e, T_e, f_imp, L_z, R, a):
    """
    
    Note : Under developement
    
    Calculate the line radiation power (in MW) due to a given impurity in a plasma

    Parameters
    -------
    n_e: Electron density [1e20 m⁻³]
    f_imp: Impurity fraction (n_imp / n_e)
    L_z: Radiative loss coefficient [W·m³] for the given impurity
    V : Plasma volume [m³]
    
    Returns
    -------
    P_line : Line radiation power [MW]

    Assumptions:
        - Uniform impurity concentration
        - Homogeneous plasma
        - Line radiation + radiative recombination included in L_z(T_e)

    Sources
    -------
    - H. Pütterich et al., "Radiative cooling rates of heavy elements for fusion plasmas", Nucl. Fusion 50 (2010) 025012.
    - Summers et al., Atomic Data and Analysis Structure (ADAS): http://adas.ac.uk
    - IAEA-INDC report on radiative losses, INDC(NDS)-457.

    Note
    ----
    This function can be adapted to any impurity by changing L_z 
    according to the species (W, C, N, etc.).
    
    """
    
    P_line = (n_e * 1e20)**2 * f_imp * L_z * V

    return P_line / 1e6

def get_Lz(impurity, Te_keV):
    """
    Return the line radiative loss coefficient Lz (W·m³) for a given 
    impurity and electron temperature.
    
    Parameters
    ----------
    impurity : str
        Impurity name or symbol ("W", "Ar", "Ne", "C", 
        or long forms "tungsten", "argon", "neon", "carbon").
    Te_keV : float or array-like
        Electron temperature in keV.
    
    Returns
    -------
    Lz : float or ndarray
        Line radiative cooling coefficient (W·m³).
    
    Notes
    -----
    - Values based on Mavrin (2018) polynomial fits and Pütterich (2010) for W
    - Log-log interpolation for numerical stability
    - Linear extrapolation outside bounds (with warning)
    
    References
    ----------
    [1] Mavrin A.A. (2018), Rad. Eff. Def. Solids 173:5-6, 388-398
    [2] Pütterich et al. (2010), Nucl. Fusion 50, 025012
    [3] Pütterich et al. (2019), Nucl. Fusion 59, 056013
    
    Examples
    --------
    >>> get_Lz("W", 10.0)   # Tungsten at 10 keV
    ~4e-32 W·m³
    """
    # Normalize impurity name
    imp_map = {
        "w": "W", "tungsten": "W",
        "ar": "Ar", "argon": "Ar",
        "ne": "Ne", "neon": "Ne",
        "c": "C", "carbon": "C",
        "n": "N", "nitrogen": "N",
        "kr": "Kr", "krypton": "Kr",
    }
    
    imp = impurity.strip().lower()
    if imp in imp_map:
        imp = imp_map[imp]
    else:
        imp = impurity.strip().upper()
    
    # =========================================================================
    # Lz data tables (W·m³) vs Te (keV)
    # CORRECTED values based on Pütterich and Mavrin
    # =========================================================================
    
    # Temperature grid (keV) - extended range
    Te_grid = np.array([
        0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 
        10.0, 20.0, 50.0, 100.0
    ])
    
    # -------------------------------------------------------------------------
    # Tungsten (W, Z=74)
    # Source: Pütterich et al. (2010), Nucl. Fusion 50, 025012
    # W has a radiation peak at ~0.1-1 keV, then DECREASES sharply
    # At high Te (>5 keV), W is highly ionized → strongly reduced line radiation
    # Values calibrated to give ~5-10 MW for 0.01% W in ITER
    # -------------------------------------------------------------------------
    Lz_W = np.array([
        1.0e-32,   # 0.01 keV - W weakly ionized, complex spectrum
        5.0e-32,   # 0.02 keV
        2.0e-31,   # 0.05 keV - rising toward peak
        4.0e-31,   # 0.1 keV  - near peak (W27+-W35+)
        5.0e-31,   # 0.2 keV  - PEAK region (strongest radiation)
        4.0e-31,   # 0.5 keV  - main peak (W35+-W45+)
        3.0e-31,   # 1.0 keV  - still high (W40+-W46+)
        1.0e-31,   # 2.0 keV  - W44+-W50+ dominant
        2.0e-32,   # 5.0 keV  - decreasing rapidly (W50+-W56+)
        8.0e-33,   # 10.0 keV - W56+-W64+ highly ionized
        4.0e-33,   # 20.0 keV - approaching fully ionized
        2.0e-33,   # 50.0 keV - very few bound electrons
        1.0e-33,   # 100.0 keV - near fully ionized
    ])
    
    # -------------------------------------------------------------------------
    # Argon (Ar, Z=18)
    # Ar is FULLY IONIZED above ~2-3 keV
    # At high Te, only recombination radiation (very small)
    # -------------------------------------------------------------------------
    Lz_Ar = np.array([
        2.0e-31,   # 0.01 keV - strong radiation (Li-like, Be-like)
        5.0e-31,   # 0.02 keV - near peak
        3.0e-31,   # 0.05 keV - He-like Ar dominant
        1.0e-31,   # 0.1 keV
        3.0e-32,   # 0.2 keV  - becoming H-like
        3.0e-33,   # 0.5 keV  - mostly H-like/bare
        5.0e-34,   # 1.0 keV  - fully ionized
        1.5e-34,   # 2.0 keV
        5.0e-35,   # 5.0 keV  - only recombination
        3.0e-35,   # 10.0 keV
        2.0e-35,   # 20.0 keV
        1.0e-35,   # 50.0 keV
        8.0e-36,   # 100.0 keV
    ])
    
    # -------------------------------------------------------------------------
    # Neon (Ne, Z=10)
    # Ne is FULLY IONIZED above ~0.5-1 keV
    # -------------------------------------------------------------------------
    Lz_Ne = np.array([
        8.0e-32,   # 0.01 keV
        2.0e-31,   # 0.02 keV - peak (Li-like, He-like)
        1.5e-31,   # 0.05 keV
        5.0e-32,   # 0.1 keV
        1.0e-32,   # 0.2 keV
        1.0e-33,   # 0.5 keV  - mostly ionized
        2.0e-34,   # 1.0 keV  - fully ionized
        8.0e-35,   # 2.0 keV
        3.0e-35,   # 5.0 keV
        1.5e-35,   # 10.0 keV
        1.0e-35,   # 20.0 keV
        5.0e-36,   # 50.0 keV
        3.0e-36,   # 100.0 keV
    ])
    
    # -------------------------------------------------------------------------
    # Carbon (C, Z=6)
    # C is FULLY IONIZED above ~0.2-0.3 keV
    # -------------------------------------------------------------------------
    Lz_C = np.array([
        3.0e-32,   # 0.01 keV
        1.0e-31,   # 0.02 keV - peak
        6.0e-32,   # 0.05 keV
        1.5e-32,   # 0.1 keV
        2.0e-33,   # 0.2 keV  - becoming fully ionized
        2.0e-34,   # 0.5 keV  - fully ionized
        5.0e-35,   # 1.0 keV
        2.0e-35,   # 2.0 keV
        8.0e-36,   # 5.0 keV
        5.0e-36,   # 10.0 keV
        3.0e-36,   # 20.0 keV
        1.5e-36,   # 50.0 keV
        1.0e-36,   # 100.0 keV
    ])
    
    # -------------------------------------------------------------------------
    # Nitrogen (N, Z=7)
    # -------------------------------------------------------------------------
    Lz_N = np.array([
        5.0e-32,   # 0.01 keV
        1.5e-31,   # 0.02 keV - peak
        1.0e-31,   # 0.05 keV
        3.0e-32,   # 0.1 keV
        5.0e-33,   # 0.2 keV
        4.0e-34,   # 0.5 keV
        1.0e-34,   # 1.0 keV
        4.0e-35,   # 2.0 keV
        1.5e-35,   # 5.0 keV
        1.0e-35,   # 10.0 keV
        6.0e-36,   # 20.0 keV
        3.0e-36,   # 50.0 keV
        2.0e-36,   # 100.0 keV
    ])
    
    # -------------------------------------------------------------------------
    # Krypton (Kr, Z=36)
    # Intermediate-Z, used for DEMO seeding
    # -------------------------------------------------------------------------
    Lz_Kr = np.array([
        5.0e-32,   # 0.01 keV
        2.0e-31,   # 0.02 keV
        5.0e-31,   # 0.05 keV - peak
        4.0e-31,   # 0.1 keV
        2.0e-31,   # 0.2 keV
        5.0e-32,   # 0.5 keV
        1.5e-32,   # 1.0 keV
        5.0e-33,   # 2.0 keV
        1.5e-33,   # 5.0 keV
        8.0e-34,   # 10.0 keV
        4.0e-34,   # 20.0 keV
        2.0e-34,   # 50.0 keV
        1.0e-34,   # 100.0 keV
    ])
    
    # Data dictionary
    tables = {
        "W": Lz_W,
        "Ar": Lz_Ar,
        "Ne": Lz_Ne,
        "C": Lz_C,
        "N": Lz_N,
        "Kr": Lz_Kr,
    }
    
    if imp not in tables:
        available = list(tables.keys())
        raise ValueError(
            f"Impurity '{impurity}' not supported. "
            f"Choose from: {available}"
        )
    
    Lz_table = tables[imp]
    
    # Log-log interpolation
    log_Te_grid = np.log10(Te_grid)
    log_Lz_table = np.log10(Lz_table)
    
    f_interp = interp1d(
        log_Te_grid, 
        log_Lz_table,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    # Compute Lz
    Te_keV = np.atleast_1d(Te_keV)
    log_Lz = f_interp(np.log10(Te_keV))
    Lz = 10.0 ** log_Lz
    
    # Warning if outside bounds
    if np.any(Te_keV < Te_grid[0]) or np.any(Te_keV > Te_grid[-1]):
        import warnings
        warnings.warn(
            f"Te outside validated range [{Te_grid[0]:.3f}, {Te_grid[-1]:.1f}] keV. "
            f"Extrapolation used.",
            UserWarning
        )
    
    if Lz.size == 1:
        return float(Lz[0])
    return Lz


def get_average_charge(impurity, Te_keV):
    """
    Return the average charge state of the impurity in coronal equilibrium.
    
    Parameters
    ----------
    impurity : str
        Impurity symbol ("W", "Ar", "Ne", "C").
    Te_keV : float
        Electron temperature in keV.
    
    Returns
    -------
    Z_avg : float
        Average ion charge state.
    """
    imp_map = {
        "w": "W", "tungsten": "W",
        "ar": "Ar", "argon": "Ar", 
        "ne": "Ne", "neon": "Ne",
        "c": "C", "carbon": "C",
        "n": "N", "nitrogen": "N",
        "kr": "Kr", "krypton": "Kr",
    }
    
    imp = impurity.strip().lower()
    imp = imp_map.get(imp, impurity.strip().upper())
    
    Z_max = {"W": 74, "Ar": 18, "Ne": 10, "C": 6, "N": 7, "Kr": 36}
    
    if imp not in Z_max:
        raise ValueError(f"Impurity '{impurity}' not supported.")
    
    # Temperature scale for full ionization (approximate)
    Te_ionization = {
        "W": 5.0,    # W approaches full ionization ~50 keV
        "Ar": 0.3,   # Ar fully ionized ~3 keV
        "Ne": 0.15,  # Ne fully ionized ~1.5 keV
        "C": 0.05,   # C fully ionized ~0.5 keV
        "N": 0.07,   # N fully ionized ~0.7 keV
        "Kr": 1.0,   # Kr fully ionized ~10 keV
    }
    
    Z = Z_max[imp]
    Te_ion = Te_ionization[imp]
    
    Z_avg = Z * (1.0 - np.exp(-Te_keV / Te_ion))
    
    return float(max(1.0, min(Z_avg, Z)))


if __name__ == "__main__":
    
    # Test the function
    print("=== Testing get_Lz ===")
    print(f"Lz(W, 10 keV)  = {get_Lz('W', 10.0):.2e} W·m³")
    print(f"Lz(Ar, 5 keV)  = {get_Lz('Ar', 5.0):.2e} W·m³")
    print(f"Lz(Ne, 1 keV)  = {get_Lz('Ne', 1.0):.2e} W·m³")
    print(f"Lz(C, 0.5 keV) = {get_Lz('C', 0.5):.2e} W·m³")
    
    # Plot Lz(Te) curves
    Te_plot = np.logspace(-2, 2, 200)  # 0.01 - 100 keV
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        "W": "red", "Ar": "blue", "Ne": "green", 
        "C": "orange", "N": "purple", "Kr": "cyan"
    }
    labels = {
        "W": "Tungsten (Z=74)", "Ar": "Argon (Z=18)", 
        "Ne": "Neon (Z=10)", "C": "Carbon (Z=6)",
        "N": "Nitrogen (Z=7)", "Kr": "Krypton (Z=36)"
    }
    
    for imp in ["W", "Ar", "Ne", "C", "N", "Kr"]:
        Lz = get_Lz(imp, Te_plot)
        ax.loglog(Te_plot, Lz, label=labels[imp], color=colors[imp], linewidth=2)
    
    ax.set_xlabel("Electron temperature Te (keV)", fontsize=12)
    ax.set_ylabel("Radiative cooling coefficient Lz (W·m³)", fontsize=12)
    ax.set_title("Coronal equilibrium radiative cooling coefficients\n"
                 "(Based on Mavrin 2018 / ADAS)", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(1e-36, 1e-30)
    
    # Annotate plasma regions
    ax.axvspan(0.01, 0.1, alpha=0.1, color='blue')
    ax.axvspan(1, 30, alpha=0.1, color='red')
    ax.text(0.03, 5e-31, "Edge/\nDivertor", fontsize=9, ha='center')
    ax.text(5, 5e-31, "Core", fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.show()

    """
    Test radiative power losses for ITER-like plasma parameters.
    Validates bremsstrahlung, synchrotron, and line radiation calculations.
    """
    
    # ITER-like plasma parameters
    Te_keV = 7.0      # Electron temperature [keV]
    ne = 1.0          # Electron density [10^20 m^-3]
    V = 830.0         # Plasma volume [m³]
    R, a = 6.2, 2.0   # Major and minor radius [m]
    Bt = 5.3          # Toroidal magnetic field [T]
    kappa = 1.7       # Plasma elongation
    Z_eff = 1.5       # Effective charge (more realistic than 1.0)
    
    # Synchrotron reflection coefficient
    r = r_synch       # Wall reflection coefficient (from parameterization)
    
    # Profile parameters (typical parabolic profiles)
    nu_n = 0.1     # Density profile exponent
    nu_T = 1.0     # Temperature profile exponent
    
    # Impurities
    impurities = ['W', 'Ar']
    fractions = [0.0001, 0.02]   # 0.01% W, 2% Ar
    
    print("="*70)
    print("Radiative Power Loss Analysis - ITER-like Parameters")
    print("="*70)
    print(f"Electron temperature: Te = {Te_keV} keV")
    print(f"Electron density:     ne = {ne} × 10²⁰ m⁻³")
    print(f"Plasma volume:        V  = {V} m³")
    print(f"Magnetic field:       Bt = {Bt} T")
    print(f"Effective charge:     Zeff = {Z_eff}")
    print("="*70)
    
    # Calculate bremsstrahlung
    P_brem = f_P_bremsstrahlung(V, ne, Te_keV, Z_eff, R, a)
    print(f"\n1. Bremsstrahlung power: {P_brem:.2f} MW")
    print(f"   (Expected ~10-15 MW for ITER)")
    
    # Calculate synchrotron using Albajar formula
    # NOTE: Removed beta_T from the call - it's not a parameter!
    P_syn = f_P_synchrotron(Te_keV, R, a, Bt, ne, kappa, nu_n, nu_T, r)
    print(f"\n2. Synchrotron power (Albajar): {P_syn:.2f} MW")
    print(f"   (Expected ~1-2 MW for ITER)")
    print(f"   Reflection coefficient r = {r}")
    
    # Line radiation from impurities
    print(f"\n3. Line radiation:")
    P_line_total = 0
    for imp, f_imp in zip(impurities, fractions):
        Lz = get_Lz(imp, Te_keV)
        P_line = f_P_line_radiation(V, ne, Te_keV, f_imp, Lz, R, a)
        P_line_total += P_line
        print(f"   {imp:2s} ({f_imp*100:.2f}%): {P_line:.2e} MW")
    
    print(f"   Total line radiation: {P_line_total:.2f} MW")
    
    # Total radiative losses
    P_rad_total = P_brem + P_syn + P_line_total
    print(f"\n" + "="*70)
    print(f"TOTAL RADIATIVE LOSSES: {P_rad_total:.2f} MW")
    print(f"="*70)
    
    # Comparison with typical ITER values
    print(f"\nComparison with ITER H-mode baseline:")
    print(f"  Fusion power:        ~500 MW")
    print(f"  Alpha power:         ~100 MW")
    print(f"  Radiated fraction:   ~{(P_rad_total/100)*100:.0f}% of alpha power")

def f_Get_parameter_scaling_law(Scaling_Law):
    
    # Considering :
        # B the toroidal magnetic field on R0 (T)
        # R0 the geometrcial majopr radius (m)
        # Kappa the elongation
        # M or A  the average atomic mass (AMU)
        # Epsilon the inverse aspect ratio
        # n the density (10**19/m cube)
        # I plasma current (MA)
        # P the absorbed power (MW)
        # H an amplification factor = Taue/Taue_Hmode
    
    # Definition des valeurs pour chaque loi
    param_values = {
        'IPB98(y,2)': {
            'C_SL': 0.0562,
            'alpha_(1+delta)': 0,
            'alpha_M': 0.19,
            'alpha_kappa': 0.78,
            'alpha_epsilon': 0.58,
            'alpha_R': 1.97,
            'alpha_B': 0.15,
            'alpha_n': 0.41,
            'alpha_I': 0.93,
            'alpha_P': -0.69
        },
        'ITPA20-IL': {
            'C_SL': 0.067,
            'alpha_(1+delta)': 0.56,
            'alpha_M': 0.3,
            'alpha_kappa': 0.67,
            'alpha_epsilon': 0,
            'alpha_R': 1.19,
            'alpha_B': -0.13,
            'alpha_n': 0.147,
            'alpha_I': 1.29,
            'alpha_P': -0.644
        },
        'ITPA20': {
            'C_SL': 0.053,
            'alpha_(1+delta)': 0.36,
            'alpha_M': 0.2,
            'alpha_kappa': 0.8,
            'alpha_epsilon': 0.35,
            'alpha_R': 1.71,
            'alpha_B': 0.22,
            'alpha_n': 0.24,
            'alpha_I': 0.98,
            'alpha_P': -0.669
        },
        'DS03': {
            'C_SL': 0.028,
            'alpha_(1+delta)': 0,
            'alpha_M': 0.14,
            'alpha_kappa': 0.75,
            'alpha_epsilon': 0.3,
            'alpha_R': 2.11,
            'alpha_B': 0.07,
            'alpha_n': 0.49,
            'alpha_I': 0.83,
            'alpha_P': -0.55
        },
        'L-mode': {
            'C_SL': 0.023,
            'alpha_(1+delta)': 0,
            'alpha_M': 0.2,
            'alpha_kappa': 0.64,
            'alpha_epsilon': -0.06,
            'alpha_R': 1.83,
            'alpha_B': 0.03,
            'alpha_n': 0.4,
            'alpha_I': 0.96,
            'alpha_P': -0.73
        },
        'L-mode OK': {
            'C_SL': 0.023,
            'alpha_(1+delta)': 0,
            'alpha_M': 0.2,
            'alpha_kappa': 0.64,
            'alpha_epsilon': -0.06,
            'alpha_R': 1.78,
            'alpha_B': 0.03,
            'alpha_n': 0.4,
            'alpha_I': 0.96,
            'alpha_P': -0.73
        },
        'ITER89-P': {
            'C_SL': 0.048,
            'alpha_(1+delta)': 0,
            'alpha_M': 0.5,
            'alpha_kappa': 0.5,
            'alpha_epsilon': 0.3,
            'alpha_R': 1.2,
            'alpha_B': 0.2,
            'alpha_n': 0.08,
            'alpha_I': 0.85,
            'alpha_P': -0.5
        }
    }
    
    if Scaling_Law in param_values:
        C_SL = param_values[Scaling_Law]['C_SL']
        alpha_delta = param_values[Scaling_Law]['alpha_(1+delta)']
        alpha_M = param_values[Scaling_Law]['alpha_M']
        alpha_kappa = param_values[Scaling_Law]['alpha_kappa']
        alpha_epsilon = param_values[Scaling_Law]['alpha_epsilon']
        alpha_R = param_values[Scaling_Law]['alpha_R']
        alpha_B = param_values[Scaling_Law]['alpha_B']
        alpha_n = param_values[Scaling_Law]['alpha_n']
        alpha_I = param_values[Scaling_Law]['alpha_I']
        alpha_P = param_values[Scaling_Law]['alpha_P']
        return C_SL,alpha_delta,alpha_M,alpha_kappa,alpha_epsilon,alpha_R,alpha_B,alpha_n,alpha_I,alpha_P
    else:
        raise ValueError(f"La loi {Scaling_Law} n'existe pas.")

#%%

# print("D0FUS_physical_functions loaded")
