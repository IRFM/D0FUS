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

# Inputs : Pfus[MW] Bmax[T] R0[m] a[m]

def f_Kappa(A,Option_Kappa):
    """
    
    Estimate the maximum elongation as a function of the aspect ratio
    None of these scalings are satisfactory during large parameter scans
    In practice, elongation is often manually fixed
    
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
    return(κ)


def f_B0(Bmax, a, b, R0):
    """
    
    Estimate the magnetif field in the centre of the plasma
    
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

def f_nbar(P_fus,R0,a,κ,nu_n,nu_T,f_alpha,Tbar):
    """
    
    Allows for the calculation of the mean electronic density needed for the fusion power chosen by the user
    
    Parameters
    ----------
    P_fus : The Fusion power [MW]
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    nu_n : Density profile parameter 
    nu_T : Temperature profile parameter
        
    Returns
    -------
    n_bar : The mean electronic density [1e20p/m^3]
    
    """
    # Définition de l'intégrande
    def integrand(rho):
        Tprof_value = f_Tprof(Tbar, nu_T, rho)  # Calcul de f_Tprof(Tbar, nu_T, rho)
        sigmav_value = f_sigmav(Tprof_value)    # Calcul de f_sigmav(Tprof_value)
        return sigmav_value * rho * (1 - rho**2)**(2 * nu_n)

    # Effectuer l'intégration de 0 à 1
    result_integration, error = quad(integrand, 0, 1)
    
    n_DT = np.sqrt(P_fus*1e6/((E_N+E_ALPHA)*np.pi**2*R0*κ*a**2*(1+nu_n)**2*result_integration))/(1e20*2)
    
    # Taking into account the alpha dilution
    n_bar = n_DT* (2 + (4*f_alpha)/(1-2*f_alpha))
    
    return(n_bar)

def f_pbar(nu_n,nu_T,n_bar,Tbar,f_alpha):
    """
    
    Estimate the plasma pressure
    
    Parameters
    ----------
    Tbar : Mean temperature [keV]
    nbar : mean electronic density [1e20p/m^3]
    nu_n : Density profile parameter 
    nu_T : Temperature profile parameter
        
    Returns
    -------
    p_bar : The mean pressure [MPa]
    
    """
    p_bar = (2-f_alpha)*((1+nu_T)*(1+nu_n)/(1+nu_n+nu_T))*(n_bar*1e20)*(Tbar*E_ELEM*1e3)/1e6
    return p_bar

def f_beta(pbar,B0,a,Ip):
    """
    
    Calculation of the normalized plasma beta
    Normalized ratio of the plasma pressure and the magnetic pressure
    Represent the 'efficiency' of the confinement
    
    Parameters
    ----------
    pbar : Mean pressure [MPa]
    B0 : The central magnetic field [T]
    a : Minor radius [m]
    Ip : Plasma current [MA]
        
    Returns
    -------
    beta : The plasma beta
    
    """
    beta = 2*μ0*pbar*a/(B0*Ip/1e6) * 100. # in %
    return beta

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

def f_nG(Ip,a):
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
    nG = Ip/(np.pi*a**2)
    return nG

def f_qstar(a,B0,R0,Ip,κ):
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
    qstar = (np.pi*a**2*B0*(1+κ**2))/(μ0*R0*Ip*1e6)
    return qstar

def f_cost(a,b,c,d,R0,κ,Q):
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
    Q : Gain factor
        
    Returns
    -------
    cost : Cost parameter [m^3]
    
    """
    V_BB = 2*(b*2*np.pi*((R0+a+b)**2-(R0-a-b)**2))+(4*κ*a*np.pi)*((R0-a)**2+(R0+a+b)**2-(R0-a-b)**2-(R0+a)**2) # Cylindrical BB model
    V_TF = 8*np.pi*(R0-a-b-(c/2))*c*((κ+1)*a+(2*b)+c) # Rectangular TF model coil
    V_CS = 2*np.pi*((R0-a-b-c)**2-(R0-a-b-c-d)**2)*(2*(a*κ+b+c)) # h approx to 2*(a*κ+b+c) and cylindrical model
    cost = (V_BB+V_TF+V_CS)/Q
    return cost

def f_heat(B0,R0,P_fus):
    """
    
    Calculation of the heat parameter
    For now it is just the Alpha power divided by the length of the separatrix 
    One can also choos to pultiply by lambda_q to approximate the surface deposition
    But current scallings as the Martin one seems inaccurate on vast scans
    To see as an indicator to compare designs
    
    Parameters
    ----------
    B0 : The central magnetic field [T]
    R0 : Major radius [m]
    P_fus : The Fusion power [MW]
        
    Returns
    -------
    heat : heat parameter [MW/m]
    
    """
    heat = (E_ALPHA*P_fus)/(E_ALPHA+E_N)/2*np.pi*R0
    return heat

def f_tauE(pbar,R0,a,κ,P_fus,Q):
    """
    
    Calculation of the confinement time from the power balance
    
    Parameters
    ----------
    pbar : The mean pressure [MPa]
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    P_fus : The Fusion power [MW]P_fus : The Fusion power [MW]
    Q : Gain factor
        
    Returns
    -------
    tauE : Confinement time [s]
    
    """
    tauE = pbar*1e6*3.*np.pi**2*R0*a**2*κ*(E_N+E_ALPHA)/(E_ALPHA*P_fus*1e6) *(1/(1+5/Q))
    return tauE
        
def f_Ip(tauE,R0,a,κ,nbar,B0,Atomic_mass,P_fus,Q):
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
    P_fus : The Fusion power [MW]
    Q : Gain factor
        
    Returns
    -------
    Ip : Plasma current [MA]
    
    """
    
    P = (E_ALPHA*P_fus)/(E_ALPHA+E_N) * (1+5/Q)
    Epsilon = a/R0
    
    # A creuser
    Suspect = B0**alpha_B
    partie_reelle = Suspect.real
    
    Denominateur = H* C_SL * R0**alpha_R * Epsilon**alpha_epsilon * κ**alpha_kappa * (nbar*10)**alpha_n * partie_reelle * Atomic_mass**alpha_M * P**alpha_P * (1 + δ)**alpha_delta
    inv_cI  = 1./alpha_I
    
    Ip = ((tauE/Denominateur)**inv_cI) # in MA
    
    return Ip

def f_Ib(R0, a, κ, pbar, Ip):
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
    eta_CD : Current drive efficienty [MA]
    
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

def f_PCD(R0,nbar,Ip,Ib,eta_CD):
    """
    
    Estimate the Currend Drive (CD) power needed
    
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
    P_CD : Current drive power injected [MW]
    
    """
    P_CD = R0*nbar*abs(Ip-Ib)/eta_CD
    return P_CD

def f_PLH(eta_RF,f_RP,P_CD):
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

def Q(P_fus,P_CD):
    """
    
    Calculate the plasma amplification factor Q
    
    Parameters
    ----------
    P_fus = Fusion power [MW]
    P_CD = Current drive power [MW]
        
    Returns
    -------
    Q : Plasma amplification factor
    
    """
    Q = P_fus/P_CD
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

def f_q95(B0,Ip,R0,a,κ,δ):
    """
    
    Estimate the safety factor at 95% of the separatrix q_95
    Source : ITER Chapter  1: Overview and summary 1999 and HELIOS code J.Johner
    
    Parameters
    ----------
    B0 : The central magnetic field [T]
    Ip : Plasma current [MA]
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    δ : triangularity
    
    Returns
    -------
    q95 : safety factor at 95% of the separatrix
    
    """
    Aspect_ratio = R0 / a
    # Calcul de q95
    q95 = (2 * np.pi * a**2 * B0) / (μ0  * Ip*1e6 * R0) * (1.17-0.65/Aspect_ratio)/(1-1/Aspect_ratio**2)*(1+κ**2*(1+2*δ**2-1.2*δ**3))/2
    return q95

def f_He_fraction(n_bar,T_bar,tauE,C_Alpha):
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
    # f_alpha = (1.18*1e-4*C_Alpha)/4 * n_bar * T_bar**2 * tauE 
    
    def integrand(rho):
        T_prof_value = f_Tprof(T_bar, nu_T, rho)
        return f_sigmav(T_prof_value)
    
    # Intégration de 0 à 1
    sigmav, _ = quad(integrand, 0, 1)

    C_equa_alpha = (n_bar*1e20*sigmav*C_Alpha*tauE)
    f_alpha = (C_equa_alpha+1-np.sqrt(2*C_equa_alpha+1))/(2*C_equa_alpha)
    
    return f_alpha

# Test
# print(f"ITER Helium fraction: {round(f_He_fraction(1, 9, 3.1, 5),3)}") # ITER : ?%
# print(f"EU-DEMO Helium fraction: {round(f_He_fraction(1.2, 12.5, 4.6, 5),3)}") # DEMO : 16%

#%%

print("D0FUS_physical_functions loaded")