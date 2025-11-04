# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:31:18 2025

@author: TA276941
"""

#%% Import

from D0FUS_parameterization import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))


#%% Physical Functions

def f_Kappa(A,Option_Kappa, κ_manual):
    """
    
    Estimate the maximum elongation as a function of the aspect ratio
    
    Parameters
    ----------
    A : The aspect ratio of the machine
    Option_Kappa : Option selection between the available scalings

    Returns
    -------
    κ : The estimated maximum elongation
    
    """
    if Option_Kappa == 'Stambaugh':
        κ = 0.95*(2.4+65*np.exp(-A/0.376))
    elif Option_Kappa == 'Freidberg':
        κ = 0.95*(1.81153991*A**0.009042+1.5205*A**-1.63)
    elif Option_Kappa == 'Wenninger':
        ms = 0.2
        κ = 1.12*((18.84-0.87*A-np.sqrt(4.84*A**2-28.77*A+52.52+14.74*ms))/7.37)
    if Option_Kappa == 'Manual' :
        κ = κ_manual  # Elongation
    if κ <=0 :
        κ = np.nan
    
    return(κ)

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
    
    # Paramètres
    a = 1.0       # rayon mineur [m]
    kappa = 1.7   # allongement
    delta = 0.5   # triangularité
    n_theta = 500
    theta = np.linspace(0, 2*np.pi, n_theta)

    # --- Cas sans triangularité : ellipse simple ---
    x_ellipse = a * np.cos(theta)
    z_ellipse = kappa * a * np.sin(theta)

    # --- Cas avec triangularité (paramétrisation Miller) ---
    x_tri = a * np.cos(theta + delta * np.sin(theta))
    z_tri = kappa * a * np.sin(theta)

    # --- Tracé ---
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(x_ellipse, z_ellipse, 'b--', label="Without triangularity")
    ax.plot(x_tri, z_tri, 'r-', label="With triangularity")

    ax.set_aspect('equal')
    ax.set_xlabel("x [m] (poloidal, horizontal)")
    ax.set_ylabel("z [m] (poloidal, vertical)")
    ax.set_title("Poloidal section of the plasma")
    ax.grid(True)
    ax.legend()

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

# Typical test (ITER ~ 0.8)
# print(f_li(0.1,1))

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
        return np.pi * np.trapz(integrand, theta)
    
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
    # Exemple d'utilisation
    plot_profiles(Tbar=14, nu_T=1, nbar=1e20, nu_n=0.1)

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
    Bg = 34.3827  # en (keV**(1/2))
    mc2 = 1124656  # en keV
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
    # Parameters
    P_fus   = 2000   # [MW]
    nu_n    = 0.1
    nu_T    = 1
    f_alpha = 0.06
    Tbar    = 14     # [keV]
    
    # Sweep over plasma volumes (10 → 1000 m³, 500 points)
    V_values = np.linspace(10, 1000, 500)
    nbar_values = [f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, V) for V in V_values]
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(V_values, nbar_values, 'b-', lw=2)
    
    plt.xlabel("Plasma volume V [m³]", fontsize=12)
    plt.ylabel("Mean electron density $\\bar{n}_e$ [$10^{20}$ m$^{-3}$]", fontsize=12)
    plt.title("Required electron density vs plasma volume", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

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

    # Conversion des unités
    pbar_SI = pbar_MPa * 1e6  # MPa → Pa (N/m²)
    Ip_SI = Ip_MA * 1e6       # MA → A

    # Longueur caractéristique du périmètre (approx. ellipse)
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))

    # Formule de beta poloidal
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
    # Convert plasma current from MA to A
    
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
    
    Calculation of the safety factor
    
    Parameters
    ----------
    a : Minor radius [m]
    B0 : The central magnetic field [T]
    R0 : Major radius [m]
    Ip : Plasma current [MA]
    κ : Elongation
        
    Returns
    -------
    qstar : The safety factor
    
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
    
    cost = (V_BB+V_TF+V_CS) / P_fus
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

def f_heat_PFU_Eich(
    P_sol,          # Puissance franchissant la séparatrice (en MW)
    B_pol,          # Champ poloidal à l’axe médian (en T)
    R,              # Rayon majeur (en m)
    eps,            # Aspect ratio inverse (a/R)
    theta_deg       # Angle d’incidence sur le PFU (en degrés)
):
    """
    
    PFU approximation (not benchmarked)
    
    Calculates from the Eich scaling:
      - lambda_q (m): decay length of the Scrape-Off Layer (SOL) power
      - q_parallel0 (MW/m²): peak parallel heat flux
      - q_target (MW/m²): heat flux on the Plasma-Facing Unit (PFU) at incidence angle theta

    Source : [Eich scaling 2013]
    lambda_q [mm] = 1.35 * R^0.04 * B_pol^(-0.92) * eps^0.42 * P_sol^(-0.02)

    Returns:
    lambda_q_m : Width of the Scrape Of Layer (SOL) [m]
    q_parallel0 : Peack heat flux on the PFU [MW/m²]
    q_target Heat flux on the PFU [MW/m²]
    
    """
    # conversion de l'angle en radians
    theta = np.deg2rad(theta_deg)

    # calcul de lambda_q en mm puis m
    lambda_q_mm = 1.35 * R**0.04 * B_pol**(-0.92) * eps**0.42 * P_sol**(-0.02)
    lambda_q_m = lambda_q_mm * 1e-3

    # pic de flux parallèle en MW/m²
    q_parallel0 = P_sol / (2 * np.pi * R * lambda_q_m)
    # flux sur la PFU en MW/m²
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
    
def f_Ip_Auclair(tauE, R0, a, kappa, delta, nbar, B0, Atomic_mass, P_Alpha, P_Ohm, P_Aux, P_rad):
    """
    Calculate the plasma current (Ip) using the Auclair scaling law.

    Parameters
    ----------
    tauE : Energy confinement time [s]
    R0 : Major radius [m]
    a : Minor radius [m]
    kappa : Elongation
    delta : Triangularity
    nbar : Mean electron density [1e19 m^-3]
    B0 : Central magnetic field [T]
    Atomic_mass : Mean ion mass [AMU]
    P_Alpha : Alpha power [MW]
    P_Aux : Auxiliary power [MW]
    P_Ohm : Ohmic power [MW]
    P_rad : Radiated power [MW]
    
    Returns
    -------
    Ip : Plasma current [MA]

    Scaling law (WLS multivariable, Auclair, STD5 database):
    Ip [MA] ≈ exp(1.119) * tauE^0.366 * B0^0.332 * ne^0.043 * P^0.252 *
              R0^0.161 * (1+delta)^-0.121 * kappa^0.448 * (a/R0)^1.047 * Atomic_mass^-0.028
    """
    
    one_plus_delta = 1 + delta
    epsilon = a / R0
    P = P_Alpha + P_Ohm + P_Aux - P_rad  # Total heating in MW

    Ip = np.exp(1.119) * tauE**0.366 * B0**0.332 * nbar**0.043 * \
         P**0.252 * R0**0.161 * one_plus_delta**-0.121 * \
         kappa**0.448 * epsilon**1.047 * Atomic_mass**-0.028

    return Ip  # MA



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

# Freidberg case : Ib = 6.3
# print('Ib Freidberg test case [MA] : 6.3')
# print(f'Ib original test case [MA] : {round(f_Freidberg_Ib(5.34, 1.34, 1.7, 0.76, 14.3),1)}')

# ARC case : Ib = 5.03
# print('Ib ARC [MA] : 5.03')
# print(f'Ib original ARC [MA] : {round(f_Freidberg_Ib(3.3, 1.1, 1.84, 0.58, 7.8),1)}')

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

# ARC case
# print(f'Ib Segal ARC [MA] : {round(f_Segal_Ib(0.385, 0.929, 0.34, 1.84, 1.3, 14, 3.3, 7.8),1)}')

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

def f_PCD(R0, nbar, Ip, Ib, eta_CD):
    """
    
    Estimate the Currend Drive (CD) power needed
    
    Parameters
    ----------
    a : Minor radius [m]
    R0 : Major radius [m]
    n_bar : The mean electronic density [1e20p/m^3]
    Ip : Plasma current [MA]
    Ib : Bootstrap current [MA]
        
    Returns
    -------
    P_CD : Current drive power injected [MW]
    
    """
    P_CD = R0*nbar*abs(Ip-Ib)/eta_CD
    return P_CD

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
    Estimate the Ohmic heating power in a tokamak
    
    Parameters
    ----------
    I_Ohm : Ohmic plasma current [MA]
    Tbar : Volume-averaged electron temperature [keV]
    R0 : Major radius [m]
    a : Minor radius [m]
    kappa : Plasma elongation
        
    Returns
    -------
    P_Ohm : Ohmic power [MW]
    """
    
    # Résistivité de Spitzer [Ohm·m]
    # Réf. : Spitzer, L., & Härm, R. (1953). Transport phenomena in a completely ionized gas. Phys. Rev., 89(5), 977.
    eta = 2.8e-8 / (Tbar**1.5)
    
    # Résistance plasma effective [Ohm]
    R_eff = eta * (2 * R0) / (a**2 * kappa)
    
    # Puissance ohmique en Watt
    P_Ohm = R_eff * (I_Ohm*1e6)**2 * 1e-6 # Current in A P in MW
    
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

# Test ITER :
# P_Thresh_Martin(1, 5.3, 2, 6, 1.7, 2.5)
# Test WEST :
# print(P_Thresh_Martin(0.6, 3.7, 0.72, 2.4, 1.3, 2))

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

# Test ITER :
# P_Thresh_New_S(1, 5.3, 2, 6, 1.7, 2.5)
# print(P_Thresh_New_S(0.6, 3.7, 0.72, 2.4, 1.3, 2))

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

# Test ITER :
# P_Thresh_New_Ip(1, 5.3, 2, 6, 1.7,15, 2.5)

def f_q95(B0, Ip, R0, a, κ, δ):
    """
    
    Estimate q95, the safety factor at psi_N=0.95
    
    Parameters
    ----------
    B0 : The central magnetic field [T]
    Ip : Plasma current [MA]
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation of the LCFS
    δ : Triangularity of the LCFS
    
    Returns
    -------
    q95
    
    """
    Aspect_ratio = R0/a
    
    # Formula from Johner FST 2011 (used in the HELIOS code)
    # q95 = (2 * np.pi * a**2 * B0) / (μ0  * Ip*1e6 * R0) * (1.17-0.65/Aspect_ratio)/(1-1/Aspect_ratio**2)*(1+κ**2*(1+2*δ**2-1.2*δ**3))/2
    
    # Formula from Sauter FED 2016 considering that w07 = 1, i.e. assuming no squareness
    q95 = (4.1 * a**2 * B0) / (R0 * Ip) * (1 + 1.2*(κ-1) + 0.56*(κ-1)**2) * (1 + 0.09*δ + 0.16*δ**2) * (1 + 0.45*δ/Aspect_ratio) / (1 - 0.74/Aspect_ratio)  
    
    return q95

def f_q_mhd(a, Bt, R, Ip, eps, kappa95, delta95):
    """
    
    Calculates the MHD safety factor q from the definition of the shaping factor S_k

    Parameters
    ----------
    a : Plasma minor radius [m].
    Bt : Toroidal magnetic field [T].
    R : Major radius of the tokamak [m].
    Ip : Plasma current [A].
    eps : Inverse aspect ratio (a/R).
    kappa95 : Elongation at the 95% flux surface.
    delta95 : Triangularity at the 95% flux surface.

    Returns
    -------
    q_MHD : MHD safety factor
    
    """
    # Calculate the shaping factor S_k
    S_k = (
        0.5 * (1.17 - 0.65 * eps) / (1.0 - eps**2)**2
        * (1.0 + kappa95**2 * (1.0 + 2.0 * delta95**2 - 1.2 * delta95**3))
    )

    # Calculate q
    q_MHD = 5 * a**2 * Bt * S_k / (R * Ip)

    return q_MHD

def f_He_fraction(n_bar, T_bar, tauE, C_Alpha, nu_T):
    """
    
    Estimate the fraction of Alpha particles
    Source : Appendix B, Y.Sarazin Impact of scaling laws on tokamak reactor dimensioning
    
    Parameters
    ----------
    n_bar : The mean electron density [1e20p/m^3]
    T_bar : The mean temperature [keV]
    tauE : Confinement time [s]
    C_Alpha : Tuning parameter
    
    Returns
    -------
    f_alpha : Alpha fraction (n_alpha/n_bar)
    
    """
    
    def integrand(rho):
        T_prof_value = f_Tprof(T_bar, nu_T, rho)
        return f_sigmav(T_prof_value)
    
    # Intégration de 0 à 1
    sigmav, _ = quad(integrand, 0, 1)

    C_equa_alpha = (n_bar*1e20*sigmav*C_Alpha*tauE)
    f_alpha = (C_equa_alpha + 1 - np.sqrt( 2 * C_equa_alpha + 1 )) / ( 2 * C_equa_alpha )
    
    return f_alpha

def f_tau_alpha(n_bar, T_bar, tauE, C_Alpha, nu_T):
    """
    Estimate the alpha particle confinement time Tau_alpha
    (derived consistently from f_He_fraction)

    Parameters
    ----------
    n_bar : The mean electron density [1e20p/m^3]
    T_bar : The mean temperature [keV]
    tauE : Confinement time [s]
    C_Alpha : Tuning parameter

    Returns
    -------
    tau_alpha : Alpha confinement time [s]
    """

    # On reprend les mêmes étapes que dans f_He_fraction
    def integrand(rho):
        T_prof_value = f_Tprof(T_bar, nu_T, rho)
        return f_sigmav(T_prof_value)

    sigmav, _ = quad(integrand, 0, 1)

    # Grandeur intermédiaire déjà utilisée
    C_equa_alpha = (n_bar*1e20 * sigmav * C_Alpha * tauE)

    # Fraction d’alphas
    f_alpha = (C_equa_alpha + 1 - np.sqrt(2*C_equa_alpha + 1)) / (2*C_equa_alpha)

    # Temps de confinement des alphas (relation directe)
    tau_alpha_value = f_alpha * tauE / C_equa_alpha

    return tau_alpha_value


# Test
# print(f"ITER Helium fraction: {round(f_He_fraction(1, 9, 3.1, 5),3)}") # ITER : ?%
# print(f"EU-DEMO Helium fraction: {round(f_He_fraction(1.2, 12.5, 4.6, 5),3)}") # DEMO : 16%

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

def f_P_synchrotron(T0_keV, R, a, Bt, ne0, kappa, alpha_n, alpha_T, beta_T, r):
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
    alpha_n : float
        Density profile peaking parameter
    alpha_T : float
        Temperature profile peaking parameter
    beta_T : float
        Temperature shape factor
    r : float, optional
        Wall reflection coefficient (default=0.9)

    Returns
    -------
    P_syn : float
        Total synchrotron radiation power [MW]

    References
    ----------
    Albajar, F., Johner, J., & Granata, G. (2001). 
    Improved calculation of synchrotron radiation losses in realistic tokamak plasmas. 
    Nuclear Fusion, 41(6), 665.
    """
    # Aspect ratio
    A = R / a
    
    # Calculate opacity parameter (Eq. 7)
    pa0 = 6.04e3 * a * ne0 / Bt  # Dimensionless

    # Calculate profile factor K (Eq. 13)
    K_numer = (alpha_n + 3.87*alpha_T + 1.46)**(-0.79) * (1.98 + alpha_T)**1.36 * beta_T**2.14
    K_denom = (beta_T**1.53 + 1.87*alpha_T - 0.16)**1.33
    K = K_numer / K_denom

    # Calculate aspect ratio correction factor G (Eq. 15)
    G = 0.93 * (1 + 0.85 * math.exp(-0.82 * A))

    # Calculate the main expression (Eq. 16)
    term1 = 3.84e-8 * (1 - r)**0.5
    term2 = R * a**1.38 * kappa**0.79 * Bt**2.62 * ne0**0.38
    term3 = T0_keV * (16 + T0_keV)**2.61
    term4 = (1 + 0.12 * T0_keV / pa0**0.41)**(-1.51)
    
    P_syn = term1 * term2 * term3 * term4 * K * G

    return P_syn

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
    Retourne le coefficient de pertes radiatives de raies Lz (W·m³)
    pour une impureté donnée et une température électronique.

    Paramètres
    ----------
    impurity : str
        Nom de l'impureté ("W", "Ar", "Ne", "C").
    Te_keV : float
        Température électronique en keV (valide pour 1–30 keV).

    Retour
    ------
    Lz : float
        Coefficient radiatif de raies (W·m³).

    Notes
    -----
    - Tables issues de :
      * Pütterich et al. (2010) pour W,
      * Fichiers ADF11/PLT (ADAS) pour Ar, Ne, C.
    - Interpolation en log-log pour plus de stabilité.
    """

    # Grille commune en keV
    Te_grid = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30])

    # Tables Lz (W·m³)
    tables = {
        "W":  np.array([4.21e-31, 4.38e-31, 2.28e-31, 1.87e-31,
                        1.33e-31, 9.47e-32, 7.30e-32, 6.11e-32, 5.47e-32]),
        "Ar": np.array([1.137e-29]*9),
        "Ne": np.array([3.425e-23, 6.664e-16, 3.426e-16, 1.712e-16,
                        3.425e-17, 1.712e-17, 1.047e-17, 7.150e-18, 5.235e-18]),
        "C":  np.array([1.428e-26, 8.575e-23, 4.593e-23, 2.330e-23,
                        4.545e-24, 2.277e-24, 1.394e-24, 9.531e-25, 6.985e-25])
    }

    imp = impurity.strip().capitalize()  # normalisation simple
    if imp not in tables:
        raise ValueError(f"Impureté '{impurity}' non supportée. Choisir parmi {list(tables.keys())}")

    # Interpolation log-log
    Lz_table = tables[imp]
    f = interp1d(np.log10(Te_grid), np.log10(Lz_table),
                 kind="linear", bounds_error=False, fill_value="extrapolate")
    return float(10**f(np.log10(Te_keV)))


if __name__ == "__main__":
    # ITER-like plasma parameters
    Te_keV = 7.0
    ne = 1.0          # in 1e20 m^-3
    V = 830.0         # m^3
    R, a = 6.2, 2.0
    Bt = 5.3
    kappa = 1.7       # Plasma elongation (typical for ITER)
    Z_eff = 1         # approximate effective charge
    r = r_synch
    beta_T = 2        # Beta_T taken from [J.Johner Helios]

    # Profile parameters (typical parabolic profiles)
    alpha_n = 0.1     # Density profile parameter
    alpha_T = 1.0     # Temperature profile parameter

    # Impurities
    impurities = ['W', 'Ar']
    fractions = [0.0001, 0.02]   # 0.01% W, 2% Ar

    # Calculate bremsstrahlung
    P_brem = f_P_bremsstrahlung(V, ne, Te_keV, Z_eff, R, a)
    print(f"Bremsstrahlung power: {P_brem:.2f} MW (expected ~10 MW for ITER)")

    # Calculate synchrotron using Albajar formula
    P_syn = f_P_synchrotron(Te_keV, R, a, Bt, ne, kappa, alpha_n, alpha_T, beta_T, r) 
    print(f"Synchrotron power (Albajar): {P_syn:.2f} MW (expected ~1 MW for ITER)")

    # Line radiation
    for imp, f_imp in zip(impurities, fractions):
        Lz = get_Lz(imp, Te_keV)
        P_line = f_P_line_radiation(V, ne, Te_keV, f_imp, Lz, R, a)
        print(f"Line radiation ({imp}): {P_line:.2e} MW (using ADAS table)")

    # Total
    P_line_total = sum(
        f_P_line_radiation(V, ne, Te_keV, f, get_Lz(imp, Te_keV), R, a)
        for imp, f in zip(impurities, fractions)
    )
    print(f"Total line radiation: {P_line_total:.2f} MW")

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

print("D0FUS_physical_functions loaded")