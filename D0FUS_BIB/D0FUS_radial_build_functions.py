# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:32:03 2025

@author: TA276941
"""

#%% Import

from D0FUS_physical_functions import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%%

##################################################### TF Model ##########################################################

if __name__ == "__main__":
    print("##################################################### TF Model ##########################################################")

#%% Modèles historiques Freidberg

def f_TF_coil_wedging_freidberg(a, b, R0, B0, σ_TF, μ0, J_max_TF):
    """
    Calculate the thickness and stress ratio for the TF coil using the historical 
    Freidberg model, in wedging

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)

    Returns:
    tuple: A tuple containing the calculated values :
    (c, TF_ratio_bucking)
    With
    c : TF width
    TF_ratio_wedging : % of centering stress in the tresca
    
    """

    # Calculate the strain parameter eps_B
    eps_B = (a + b) / R0

    # Calculate the magnetic stress parameter alpha_M
    alpha_M = (B0**2 / (μ0 * σ_TF)) * (
        (2 * eps_B) / (1 + eps_B) + 0.5 * np.log((1 + eps_B) / (1 - eps_B))
    )

    # Calculate the current density stress parameter alpha_J
    alpha_J = (2 * B0) / (μ0 * R0 * J_max_TF)

    # Calculate the thickness of the magnetic layer cM
    cM = R0 * (1 - eps_B - np.sqrt((1 - eps_B)**2 - alpha_M))

    # Calculate the thickness of the current density layer cJ
    cJ = R0 * (1 - eps_B - np.sqrt((1 - eps_B)**2 - alpha_J))

    # Calculate the total thickness c
    c = cM + cJ

    # Calculate the strain parameter eps_M
    eps_M = cM / R0

    # Calculate the tangential stress Sigma_T
    Sigma_T = (B0**2 / (2 * μ0)) * np.log((1 + eps_B) / (1 - eps_B)) / (eps_M * (2 - 2 * eps_B - eps_M))

    # Calculate the compressive stress Sigma_C
    Sigma_C = (B0**2 / (μ0 * eps_M)) * (2 / (2 - 2 * eps_B - eps_M)) * (eps_B / (1 + eps_B))

    # Calculate the TF ratio for wedging
    TF_ratio_wedging = Sigma_T / (Sigma_C + Sigma_T)

    return (c, TF_ratio_wedging)

if __name__ == "__main__":
    
    # Test parameters for ITER case
    a_freidberg = 2
    b_freidberg = 1.45
    R0_freidberg = 6.2
    B0_freidberg = 5.3
    σ_TF_freidberg = 660e6
    μ0_freidberg = 4 * np.pi * 1e-7
    J_max_freidberg = 20 * 1e6
    Bmax_freidberg = 12
    
    result_freidberg_wedging = f_TF_coil_wedging_freidberg(a_freidberg, b_freidberg, R0_freidberg, B0_freidberg, σ_TF_freidberg, μ0_freidberg, J_max_freidberg)
    print(f"TF freidberg model wedging : {result_freidberg_wedging}")
    
    
#%% Modèles CEA Torre

def f_TF_Torre_wedging(a, b, R0, B0, σ_TF, μ0, J_max_TF, Bmax):
    
    """
    Calculate the thickness of the TF coil using the Torre model in wedging

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)
    Bmax : Maximum magnetic field (T)

    Returns:
    c : TF width
    
    """
    
    R1_0 = R0 - a - b # Jambe interne
    R2 = R0 + a + b # Jambe externe
    
    NI = 2*np.pi*R0*B0/μ0
    
    S_cond = NI/J_max_TF
    
    c1 = R1_0-np.sqrt(R1_0**2-S_cond/np.pi)
    
    R1 = R1_0 - c1
    
    T = (np.pi*B0**2*R0**2)/(2*μ0)*math.log(R2/R1)
    P = (B0**2*R0**2)/(2*μ0*R1**2)
    
    c2 = (B0**2*R0**2)/(2*μ0*R1*σ_TF)*(1+math.log(R2/R1)/2) # Valable si R1>>c
    
    σ_theta = P*R1/c2
    # print(f"σ_theta Alex : {σ_theta/1e6}")
    # r1 = R1
    # r2 = R1-c2
    # σ_theta_epais = 2*P*r1**2/(r1**2-r2**2)
    # print(f"σ_theta épais : {σ_theta_epais/1e6}")
    # σ_theta_feidberg = (B0**2 * R0) / (μ0 * (r1-r2)) * (
    #     (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
    #     (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
    # )
    # print(f"σ_theta Freidberg : {σ_theta_feidberg/1e6}")
    # σ_theta_feidberg_in = (B0**2 * R0) / (μ0 * (r1-r2)) * (
    #     (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1
    # )
    # print(f"σ_theta Freidberg without F_out : {σ_theta_feidberg_in/1e6}")
        
    σ_z = T/(2*np.pi*R1*c2)
    
    ratio_tension = σ_z/(σ_theta+σ_z)
    
    c  = c1 + c2

    return(c, ratio_tension, σ_theta, σ_z)

def f_TF_Torre_bucking(a, b, R0, B0, σ_TF, μ0, J_max_TF, Bmax):
    
    """
    Calculate the thickness of the TF coil using the Torre model in bucking

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)
    Bmax : Maximum magnetic field (T)

    Returns:
    c : TF width
    
    """
    
    R1_0 = R0 - a - b # Jambe interne
    R2 = R0 + a + b # Jambe externe
    
    NI = 2*np.pi*R0*B0/μ0
    
    S_cond = NI/J_max_TF
    
    c1 = R1_0-np.sqrt(R1_0**2-S_cond/np.pi)
    
    R1 = R1_0 - c1
    
    T = (np.pi*B0**2*R0**2)/(2*μ0)*math.log(R2/R1)
    P = (B0**2*R0**2)/(2*μ0*R1**2)
    
    c2 = (B0**2*R0**2)*math.log(R2/R1)/(2*μ0*2*R1*(σ_TF-P)) # Valable si R1>>c
    
    σ_r = P
    σ_z = T/(2*np.pi*R1*c2)
    
    ratio_tension = σ_z / (σ_z + σ_r)

    c  = c1 + c2

    return(c, ratio_tension, σ_r, σ_z)

def f_TF_Torre_dilution_wedging(a, b, R0, B0, σ_TF, μ0, J_max_TF, Bmax):
    
    R1 = R0 - a - b # Jambe interne
    R2 = R0 + a + b # Jambe externe
    
    NI = 2*np.pi*R0*B0/μ0
    S_cond = NI/J_max_TF
    
    T = (np.pi*B0**2*R0**2)/(2*μ0)*math.log(R2/R1)
    P = (B0**2*R0**2)/(2*μ0*R1**2)
    
    fb = (2*np.sqrt(S_cond)-P*R1)/σ_TF
    fc = (-T/σ_TF-2*P*R1*np.sqrt(S_cond))/σ_TF
    Delta = fb**2 - 4*fc
    
    c1 = (-fb+np.sqrt(Delta))/2 + np.sqrt(S_cond)
    c2 = (-fb-np.sqrt(Delta))/2 + np.sqrt(S_cond)
    
    c = min(c1,c2)

    return(c)

def f_TF_Torre_dilution_bucking(a, b, R0, B0, σ_TF, μ0, J_max_TF, Bmax):
    
    R1 = R0 - a - b # Jambe interne
    R2 = R0 + a + b # Jambe externe
    
    NI = 2*np.pi*R0*B0/μ0
    S_cond = NI/J_max_TF
    
    T = (np.pi*B0**2*R0**2)/(2*μ0)*math.log(R2/R1)
    P = (B0**2*R0**2)/(2*μ0*R1**2)
    
    fb = (P*3*np.sqrt(S_cond)-σ_TF*2*np.sqrt(S_cond))/P-σ_TF
    fc = (R1+2*P*np.sqrt(S_cond)**2)/P-σ_TF
    Delta = fb**2 - 4*fc
    
    c1 = (-fb+np.sqrt(Delta))/2 + np.sqrt(S_cond)
    c2 = (-fb-np.sqrt(Delta))/2 + np.sqrt(S_cond)
    
    c = min(c1,c2)

    return(c)

if __name__ == "__main__":
    
    # ITER test parameters
    a_Torre = 2
    b_Torre = 1.45
    R0_Torre = 6.2
    B0_Torre = 5.3
    σ_TF_Torre = 660e6
    μ0_Torre = 4 * np.pi * 1e-7
    J_max_TF_Torre = 50 * 1e6
    Bmax_Torre = 12

    # Test the function
    result_Torre_wedging = f_TF_Torre_wedging(a_Torre, b_Torre, R0_Torre, B0_Torre, σ_TF_Torre, μ0_Torre, J_max_TF_Torre, Bmax_Torre)
    print(f"TF Torre wedging : {result_Torre_wedging}")
    result_Torre_bucking = f_TF_Torre_bucking(a_Torre, b_Torre, R0_Torre, B0_Torre, σ_TF_Torre, μ0_Torre, J_max_TF_Torre, Bmax_Torre)
    print(f"TF Torre bucking : {result_Torre_bucking}")
    result_Torre_wedging_dilution = f_TF_Torre_dilution_wedging(a_Torre, b_Torre, R0_Torre, B0_Torre, σ_TF_Torre, μ0_Torre, J_max_TF_Torre, Bmax_Torre)
    print(f"TF Torre wedging dilution : {result_Torre_wedging_dilution}")
    result_Torre_bucking_dilution = f_TF_Torre_dilution_bucking(a_Torre, b_Torre, R0_Torre, B0_Torre, σ_TF_Torre, μ0_Torre, J_max_TF_Torre, Bmax_Torre)
    print(f"TF Torre bucking dilution : {result_Torre_bucking_dilution}")
    
#%% Modèles D0FUS cylindres épais

def f_TF_epais_dilution_wedging(a, b, R0, B0, σ_TF, μ0, J_max_TF, Bmax):
    
    re = R0 - a - b # Jambe interne
    re2 = R0 + a + b  # Jambe externe
    
    ri_initial_guess = re - 0.5
    
    def TF_to_solve(ri):
        
        # Winding pack
        
        # Aire totale de la bobine
        S_total = np.pi * (re**2 - ri**2)
        
        # Nombre d'ampères-tours nécessaire
        NI = 2 * np.pi * R0 * B0 / μ0
        
        # Aire de supraconducteur nécessaire
        S_cond = NI / J_max_TF
        ri_cond = np.sqrt(re**2 - S_cond / np.pi)
        
        # Dilution test
        alpha = S_cond / S_total
        if alpha > 1 or alpha < 0 :
            return np.nan
        
        # Calcul des contraintes
        S_steel = S_total - S_cond
        
        T = (np.pi * B0**2 * R0**2) / (2 * μ0) * math.log(re2 / re)
        P = (B0**2 * R0**2) / (2 * μ0 * ri_cond**2)
        
        σ_theta = (2 * P * ri_cond**2) / (ri_cond**2 - ri**2)
        σ_z = T / S_steel
        σ_r = P
        
        Tresca = σ_theta + σ_z + σ_r
        Diff = Tresca - σ_TF
        
        return Diff
    
    try:
        ri_solution, info, ier, msg = fsolve(TF_to_solve, ri_initial_guess, full_output=True)
        
        # Check convergence
        if ier != 1:
            return (np.nan)
        
        ri = ri_solution[0]  # fsolve returns an array, we need the single value
        
        P = (B0**2 * R0**2) / (2 * μ0 * ri_solution**2)
        σ_theta = (2 * P * ri_cond**2) / (ri_solution**2 - ri**2)
        print(f"σ_theta cylindre épais : {σ_theta/1e6}")
        
        return(re-ri)
    
    except Exception as e:
        return (np.nan)
    
def f_TF_epais_dilution_wedging_test(a, b, R0, B0, σ_TF, μ0, J_max_TF, Bmax):
    
    # Calculer les valeurs intermédiaires
    re = R0 - a - b  # Jambe interne
    re2 = R0 + a + b  # Jambe externe

    # Nombre d'ampères-tours nécessaire
    NI = 2 * np.pi * R0 * B0 / μ0

    # Aire de supraconducteur nécessaire
    S_cond = NI / J_max_TF
    
    # epaisseur de supraconducteur nécessaire
    d_c = np.sqrt(S_cond)

    # Calculer T et P
    T = (np.pi * B0**2 * R0**2) / (2 * μ0) * math.log(re2 / re)
    P = (B0**2 * R0**2) / (2 * μ0 * (re - d_c) **2)

    # Coefficients du polynôme
    cste_a = σ_TF
    cste_b = 4 * d_c * σ_TF - 2 * re * σ_TF
    cste_c = 2 * P * d_c**2 - 4 * P * d_c * re + 2 * P * re**2 + 4 * d_c**2 * σ_TF - 4 * d_c * re * σ_TF
    cste_d = 4 * P * d_c**3 - 8 * P * d_c**2 * re + 4 * P * d_c * re**2 + T

    coefficients = [cste_a, cste_b, cste_c, cste_d]

    # Vérification des coefficients
    if np.isnan(coefficients).any() or np.isinf(coefficients).any():
        return (np.nan)

    # Résolution du polynôme
    d_ss_solutions = np.roots(coefficients)
    print(d_ss_solutions)

    # Sélectionner la plus grande racine réelle inférieure à r1 et supérieure à 0
    largest_real_root_under_re = max(
        [sol.real for sol in d_ss_solutions if np.isclose(sol.imag, 0) and sol.real < re - 0.01 - d_c and sol.real > 0],
        default=np.nan
    )

    return largest_real_root_under_re + d_c
    
if __name__ == "__main__":
    
    # ITER test parameters
    a_epais = 2
    b_epais = 1.45
    R0_epais = 6.2
    B0_epais = 5.3
    σ_TF_epais = 660e6
    μ0_epais = 4 * np.pi * 1e-7
    J_max_TF_epais = 50 * 1e6
    Bmax_epais = 12

    # Test the function
    result_epais_wedging = f_TF_epais_dilution_wedging(a_epais, b_epais, R0_epais, B0_epais, σ_TF_epais, μ0_epais, J_max_TF_epais, Bmax_epais)
    print(f"TF epais wedging : {result_epais_wedging}")
    result_epais_wedging_test = f_TF_epais_dilution_wedging_test(a_epais, b_epais, R0_epais, B0_epais, σ_TF_epais, μ0_epais, J_max_TF_epais, Bmax_epais)
    print(f"TF epais wedging final : {result_epais_wedging_test}")


#%% Modèles D0FUS Auclair


def f_TF_winding_pack_bucking_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
    """
    Calculate the thickness of the TF winding pack
    with it's associated dilution factor
    and % of the centering stress in the TRESCA

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)
    F_CClamp : Clamping force (N)
    Bmax : Maximum magnetic field in the TF
    Choice_Buck_Wedg : Mechanical configuration

    Returns:
    tuple: A tuple containing the calculated values :
    (c, Dilution_solution, TF_ratio_bucking)
    With
    c_winding_pack : Winding pack width
    Dilution_solution : % of conductor
    TF_ratio_bucking : % of centering stress in the tresca
    
    """

    # Preliminary results
    Dilution_initial_guess = 0.5
    r1 = R0 - a - b

    def TF_to_solve(Dilution):
        """
        Function to solve for the determination of the dilution factor

        Parameters:
        Dilution (float): Dilution factor.

        Returns:
        float: The difference between the Tresca stress and the yield strength.
        
        """
        # r2 to generate the magnetic field
        r2 = r1 - (Bmax / (μ0 * J_max_TF * Dilution))
        
        # Sigma_T
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))

        # Sigma_C
        Sigma_C = ( B0**2 * R0) / (μ0 * r2) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        
        # Tresca stress
        Tresca = Sigma_T + Sigma_C

        return Tresca - (σ_TF * (1 - Dilution))
    
    try:
        # Find the dilution solution using fsolve
        Dilution_solution, info, ier, msg = fsolve(TF_to_solve, Dilution_initial_guess, full_output=True)
        # Check convergence
        if ier != 1:
            # print("No convergence")
            return (np.nan, np.nan, np.nan)
        Dilution_solution = Dilution_solution[0]  # fsolve returns an array, we need the single value
        
        # Calculate r2 to generate the magnetic field
        r2 = r1 - (Bmax / (μ0 * J_max_TF * Dilution_solution))
        
        # Calculate Sigma ratio
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = ( B0**2 * R0) / (μ0 * r2) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_bucking = Sigma_T / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_bucking)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
def f_TF_winding_pack_bucking_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
    """
    Calculate the thickness of the TF winding pack
    with it's associated dilution factor
    and % of the centering stress in the TRESCA

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)
    F_CClamp : Clamping force (N)
    Bmax : Maximum magnetic field in the TF

    Returns:
    tuple: A tuple containing the calculated values :
    (c, Dilution_solution, TF_ratio_bucking)
    With
    c_winding_pack : Winding pack width
    Dilution_solution : % of conductor
    TF_ratio_bucking : % of centering stress in the tresca
    
    """

    # Preliminary results
    r1 = R0 - a - b
    F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
    F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
    epsilon_B = (a+b)/R0
    
    try:
        
        # Coefficients du polynôme
        # r2**5
        cste_a = -np.pi * σ_TF * μ0 / R0**2
        
        # r2**4
        cste_b = (
            -np.pi * Bmax * σ_TF
            + 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0
            + 2 * np.pi * J_max_TF * σ_TF * μ0 * r1
        ) / (J_max_TF * R0**2)
        
        # r2**3
        cste_c = (
            2 * np.pi * B0**2 * J_max_TF * R0**2
            + 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B
            + np.pi * Bmax * σ_TF * r1
            - F_T * J_max_TF * μ0
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0
            - 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1
        ) / (J_max_TF * R0**2)
        
        # r2**2
        cste_d = (
            -4 * np.pi * B0**2 * J_max_TF * R0**3 * epsilon_B
            - 2 * np.pi * B0**2 * J_max_TF * R0**2 * r1
            - 4 * np.pi * Bmax * R0**2 * σ_TF * epsilon_B**2
            + 4 * np.pi * Bmax * R0**2 * σ_TF
            + np.pi * Bmax * σ_TF * r1**2
            + 4 * F_T * J_max_TF * R0 * epsilon_B * μ0
            + 2 * F_T * J_max_TF * μ0 * r1
            - 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1**2
            - 2 * np.pi * J_max_TF * σ_TF * μ0 * r1**3
        ) / (J_max_TF * R0**2)
        
        # r2**1
        cste_e = (
            -2 * np.pi * B0**2 * J_max_TF * R0**2 * r1**2
            - 4 * np.pi * Bmax * R0**2 * σ_TF * epsilon_B**2 * r1
            + 4 * np.pi * Bmax * R0**2 * σ_TF * r1
            - 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B * r1**2
            - np.pi * Bmax * σ_TF * r1**3
            - 4 * F_T * J_max_TF * R0**2 * epsilon_B**2 * μ0
            + 4 * F_T * J_max_TF * R0**2 * μ0
            - 4 * F_T * J_max_TF * R0 * epsilon_B * μ0 * r1
            - F_T * J_max_TF * μ0 * r1**2
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0 * r1**2
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0 * r1**2
            + 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1**3
            + np.pi * J_max_TF * σ_TF * μ0 * r1**4
        ) / (J_max_TF * R0**2)
        
        # r2**0
        cste_f = 4 * np.pi * B0**2 * R0 * epsilon_B * r1**2 + 2 * np.pi * B0**2 * r1**3
        # Coefficients du polynôme
        coefficients = [cste_a, cste_b, cste_c, cste_d, cste_e, cste_f]

        # Vérification des coefficients
        if np.isnan(coefficients).any() or np.isinf(coefficients).any():
            return (np.nan,np.nan,np.nan)

        # Résolution du polynôme
        solutions_bucking = np.roots(coefficients)
        
        # Sélectionner la plus grande racine réelle inférieure à r1 et supérieure à 0
        largest_real_root_under_r1 = max(
            [sol.real for sol in solutions_bucking if np.isclose(sol.imag, 0) and sol.real < r1-0.01  and sol.real > 0], 
            default=np.nan
        )
        
        if np.isnan(largest_real_root_under_r1).any() :
            return (np.nan,np.nan,np.nan)
        
        r2 = largest_real_root_under_r1
        
        Dilution_solution = Bmax / (μ0 * J_max_TF * (r1 - r2))

        # Calculate Sigma ratio
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (2 * B0**2 * R0) / (μ0 * r2) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_bucking = Sigma_T / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_bucking)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
def f_TF_winding_pack_wedging_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
    
    """
    Calculate the thickness of the TF winding pack
    with it's associated dilution factor
    and % of the centering stress in the TRESCA

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)
    F_CClamp : Clamping force (N)
    Bmax : Maximum magnetic field in the TF
    Choice_Buck_Wedg : Mechanical configuration

    Returns:
    tuple: A tuple containing the calculated values :
    (c, Dilution_solution, TF_ratio_bucking)
    With :

    c_winding_pack : Winding pack thickness
    Dilution_solution : % of conductor
    TF_ratio_wedging : % of centering stress in the tresca
    
    """

    # Preliminary results
    Dilution_initial_guess = 0.5
    r1 = R0 - a - b

    def TF_to_solve(Dilution):
        """
        Function to solve for the determination of the dilution factor

        Parameters:
        Dilution (float): Dilution factor.

        Returns:
        float: The difference between the Tresca stress and the yield strength.
        
        """
        # r2 to generate the magnetic field
        r2 = r1 - (Bmax / (μ0 * J_max_TF * Dilution))
        
        # Sigma_T
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))

        # Sigma_C
        Sigma_C = (B0**2 * R0) / (μ0 * (r1-r2)) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        
        # Tresca stress
        Tresca = Sigma_T + Sigma_C

        return Tresca - (σ_TF * (1 - Dilution))
    

    try:
        # Find the dilution solution using fsolve
        Dilution_solution, info, ier, msg = fsolve(TF_to_solve, Dilution_initial_guess, full_output=True)
        # Check convergence
        if ier != 1:
            # print("No convergence")
            return (np.nan, np.nan, np.nan)
        Dilution_solution = Dilution_solution[0]  # fsolve returns an array, we need the single value

        # Calculate r2 to generate the magnetic field
        r2 = r1 - (Bmax / (μ0 * J_max_TF * Dilution_solution))
        
        # Calculate Sigma ratio
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (B0**2 * R0) / (μ0 * (r1-r2)) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_wedging = Sigma_T / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_wedging)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
def f_TF_winding_pack_wedging_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
    """
    Calculate the thickness of the TF winding pack
    with it's associated dilution factor
    and % of the centering stress in the TRESCA

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)
    F_CClamp : Clamping force (N)
    Bmax : Maximum magnetic field in the TF

    Returns:
    tuple: A tuple containing the calculated values :
    (c, Dilution_solution, TF_ratio_bucking)
    With

    c_winding_pack : Winding pack width [m]
    Dilution_solution : % of conductor
    TF_ratio_wedging : % of centering stress in the tresca
    
    """

    # Preliminary results
    r1 = R0 - a - b
    F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
    F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
    epsilon_B = (a+b)/R0
    
    try:
        
        # Coefficients du polynôme
        # r2**5
        cste_a = np.pi * σ_TF * μ0 / R0**2
        
        # r2**4
        cste_b = (
            np.pi * Bmax * σ_TF
            - 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0
            - 3 * np.pi * J_max_TF * σ_TF * μ0 * r1
        ) / (J_max_TF * R0**2)
        
        # r2**3
        cste_c = (
            2 * np.pi * B0**2 * J_max_TF * R0**2
            - 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B
            - 2 * np.pi * Bmax * σ_TF * r1
            + F_T * J_max_TF * μ0
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0
            + 8 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1
            + 2 * np.pi * J_max_TF * σ_TF * μ0 * r1**2
        ) / (J_max_TF * R0**2)
        
        # r2**2
        cste_d = (
            -4 * np.pi * B0**2 * J_max_TF * R0**3 * epsilon_B
            - 2 * np.pi * B0**2 * J_max_TF * R0**2 * r1
            + 4 * np.pi * Bmax * R0**2 * σ_TF * epsilon_B**2
            - 4 * np.pi * Bmax * R0**2 * σ_TF
            + 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B * r1
            - 4 * F_T * J_max_TF * R0 * epsilon_B * μ0
            - 3 * F_T * J_max_TF * μ0 * r1
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0 * r1
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0 * r1
            + 2 * np.pi * J_max_TF * σ_TF * μ0 * r1**3
        ) / (J_max_TF * R0**2)
        
        # r2**1
        cste_e = (
            -2 * np.pi * B0**2 * J_max_TF * R0**2 * r1**2
            + 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B * r1**2
            + 2 * np.pi * Bmax * σ_TF * r1**3
            + 4 * F_T * J_max_TF * R0**2 * epsilon_B**2 * μ0
            - 4 * F_T * J_max_TF * R0**2 * μ0
            + 8 * F_T * J_max_TF * R0 * epsilon_B * μ0 * r1
            + 3 * F_T * J_max_TF * μ0 * r1**2
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0 * r1**2
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0 * r1**2
            - 8 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1**3
            - 3 * np.pi * J_max_TF * σ_TF * μ0 * r1**4
        ) / (J_max_TF * R0**2)
        
        # r2**0
        cste_f = (
            4 * np.pi * B0**2 * J_max_TF * R0**3 * epsilon_B * r1**2
            + 2 * np.pi * B0**2 * J_max_TF * R0**2 * r1**3
            - 4 * np.pi * Bmax * R0**2 * σ_TF * epsilon_B**2 * r1**2
            + 4 * np.pi * Bmax * R0**2 * σ_TF * r1**2
            - 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B * r1**3
            - np.pi * Bmax * σ_TF * r1**4
            - 4 * F_T * J_max_TF * R0**2 * epsilon_B**2 * μ0 * r1
            + 4 * F_T * J_max_TF * R0**2 * μ0 * r1
            - 4 * F_T * J_max_TF * R0 * epsilon_B * μ0 * r1**2
            - F_T * J_max_TF * μ0 * r1**3
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0 * r1**3
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0 * r1**3
            + 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1**4
            + np.pi * J_max_TF * σ_TF * μ0 * r1**5
        ) / (J_max_TF * R0**2)
        
        # Coefficients du polynôme
        coefficients = [cste_a, cste_b, cste_c, cste_d, cste_e, cste_f]

        # Vérification des coefficients
        if np.isnan(coefficients).any() or np.isinf(coefficients).any():
            return (np.nan,np.nan,np.nan)

        # Résolution du polynôme
        solutions = np.roots(coefficients)
        
        # Sélectionner la plus grande racine réelle inférieure à r1 et supérieure à 0
        largest_real_root_under_r1 = max(
            [r.real for r in solutions if np.isclose(r.imag, 0) and r.real > 0 and r.real < r1-0.01], 
            default=np.nan
        )
        
        if np.isnan(largest_real_root_under_r1) :
            return (np.nan,np.nan,np.nan)
        
        r2 = largest_real_root_under_r1
        
        Dilution_solution = Bmax / (μ0 * J_max_TF * (r1 - r2))

        # Calculate Sigma ratio
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = np.nanmax([abs(F_z - F_CClamp), abs(F_CClamp)]) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        
        Sigma_C = (B0**2 * R0) / (μ0 * (r1-r2)) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        
        TF_ratio_wedging = Sigma_T / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_wedging)

    except Exception as e:
        return (np.nan, np.nan, np.nan)


if __name__ == "__main__":
    
    # ITER test parameters
    a_D0FUS = 2
    b_D0FUS = 1.45
    R0_D0FUS = 6.2
    B0_D0FUS = 5.3
    σ_TF_D0FUS = 660e6
    μ0_D0FUS = 4 * np.pi * 1e-7
    J_max_TF_D0FUS = 50 * 1e6
    F_CClamp_D0FUS = 0
    Bmax_D0FUS = 12

    # Test the function
    result_D0FUS_1 = f_TF_winding_pack_bucking_convergence(a_D0FUS, b_D0FUS, R0_D0FUS, B0_D0FUS, σ_TF_D0FUS, μ0_D0FUS, J_max_TF_D0FUS, F_CClamp_D0FUS, Bmax_D0FUS)
    print(f"TF D0FUS bucking winding pack convergence : {result_D0FUS_1}")
    result_D0FUS_2 = f_TF_winding_pack_bucking_polynomiale(a_D0FUS, b_D0FUS, R0_D0FUS, B0_D0FUS, σ_TF_D0FUS, μ0_D0FUS, J_max_TF_D0FUS, F_CClamp_D0FUS, Bmax_D0FUS)
    print(f"TF D0FUS bucking winding pack polynomial : {result_D0FUS_2}")
    result_D0FUS_3 = f_TF_winding_pack_wedging_convergence(a_D0FUS, b_D0FUS, R0_D0FUS, B0_D0FUS, σ_TF_D0FUS, μ0_D0FUS, J_max_TF_D0FUS, F_CClamp_D0FUS, Bmax_D0FUS)
    print(f"TF D0FUS wedging winding pack convergence : {result_D0FUS_3}")
    result_D0FUS_4 = f_TF_winding_pack_wedging_polynomiale(a_D0FUS, b_D0FUS, R0_D0FUS, B0_D0FUS, σ_TF_D0FUS, μ0_D0FUS, J_max_TF_D0FUS, F_CClamp_D0FUS, Bmax_D0FUS)
    print(f"TF D0FUS wedging winding pack polynomiale : {result_D0FUS_4}")

    
def f_TF_DOFUS(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax, Choice_Buck_Wedg):
    """
    Calculate the thickness, dilution and stress ratio 
    for the TF coil using D0FUS model

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A)

    Returns:
    tuple: A tuple containing the calculated values :
    (c, TF_ratio_bucking)
    With
    c : TF width [m]
    Winding_pack_thickness : Winding pack thickness (can be the whole coil thickness) [m]
    Winding_pack_dilution : dilution factor between supra and steel [%]
    Winding_pack_centering_ratio : % of centering stress in the tresca [%]
    Nose_thickness : Nose thickness (can be 0) [m]
    
    """
    
    r1 = R0 - a - b

    if Choice_Buck_Wedg == 'Bucking':
        
        winding_pack = f_TF_winding_pack_bucking_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
        
        # In bucking, only winding pack is considered

        Winding_pack_thickness = winding_pack[0]
        if np.isnan(Winding_pack_thickness) or Winding_pack_thickness <= 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        Winding_pack_dilution = winding_pack[1]
        Winding_pack_centering_ratio = winding_pack[2]
        Nose_thickness = np.nan
        c = Winding_pack_thickness # TF thickness
        
        return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_centering_ratio, Nose_thickness)
        
    elif Choice_Buck_Wedg == 'Wedging':
        
        if Mechanical_model == 'Winding_pack_and_Nose': # Full_winding_pack or Winding_pack_and_Nose
        
            # In wedging, one need to add the steel nose to the bucking winding pack
            winding_pack = f_TF_winding_pack_bucking_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
            
            # Extraction des résultats du winding pack
            Winding_pack_thickness = winding_pack[0]
            if Winding_pack_thickness <= 0:
                    return(np.nan,np.nan,np.nan,np.nan)
            Winding_pack_dilution = winding_pack[1]
            Winding_pack_ratio = winding_pack[2]
    
            # Calculate the strain parameter eps_B
            eps_B = (a + b) / R0
            eps_C = Winding_pack_thickness / R0
        
            # Calculate the thickness of the steel nose
            Nose_thickness = (B0**2 * R0 / (μ0 * σ_TF)) * ((1 / (2 - 2 * eps_B - eps_C)) - (1 / (2 + 2 * eps_B + eps_C)))
            
            c = Winding_pack_thickness + Nose_thickness # TF thickness
            
            return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_ratio, Nose_thickness)
        
        elif Mechanical_model == 'Full_winding_pack': # Full_winding_pack or Winding_pack_and_Nose
        
            # A single winding pack is dimensionned taking into account the vault effect
            # winding_pack = f_TF_winding_pack_wedging_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
            winding_pack = f_TF_winding_pack_wedging_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)

            Winding_pack_thickness = winding_pack[0]
            if Winding_pack_thickness == 0:
                    return(np.nan,np.nan,np.nan,np.nan)
            Winding_pack_dilution = winding_pack[1]
            Winding_pack_ratio = winding_pack[2]
            Nose_thickness = np.nan
            c = Winding_pack_thickness # TF thickness
            
            return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_ratio, Nose_thickness)
        
        else:
            print("Please provide a valid argument : Full_winding_pack or Winding_pack_and_Nose")

    else:
        print("Please provide a valid argument : Wedging or Bucking")
        

if __name__ == "__main__":
    
    # ITER test parameters
    a_D0FUS = 2
    b_D0FUS = 1.45
    R0_D0FUS = 6.2
    B0_D0FUS = 5.3
    σ_TF_D0FUS = 660e6
    μ0_D0FUS = 4 * np.pi * 1e-7
    J_max_TF_D0FUS = 50 * 1e6
    F_CClamp_D0FUS = 0
    Bmax_D0FUS = 12
    Mechanical_model = 'Winding_pack_and_Nose'

    result_D0FUS_10 = f_TF_DOFUS(a_D0FUS, b_D0FUS, R0_D0FUS, B0_D0FUS, σ_TF_D0FUS, μ0_D0FUS, J_max_TF_D0FUS, F_CClamp_D0FUS, Bmax_D0FUS, "Wedging")
    print(f"TF D0FUS wedging 2 layers convergence : {result_D0FUS_10}")
    
    Mechanical_model = 'Full_winding_pack'


#%%

##################################################### CS Model ##########################################################
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% CS D0FUS model 

    
def f_CS_DOFUS_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
    """
    Calculate the CS thickness 
    with it's associated dilution and centering ratio

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    c : TF thickness (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_CS : Yield strength of the CS steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_CS : Maximum current density of the chosen Supra + Cu + He (A)
    Choice_Buck_Wedg : Mechanical configuration
    T_bar : The mean temperature [keV]
    n_bar : The mean electron density [1e20p/m^3]
    Ip : Plasma current [MA]
    Ib : Bootstrap current [MA]

    Returns:
    tuple: A tuple containing the calculated values :
    (d, Alpha, B_CS)
    With
    d : CS width
    Alpha : % of conductor
    B_CS : CS magnetic field
    
    """
#----------------------------------------------------------------------------------------------------------------------------------------    
    
    #### Preliminary results ####
    
    # Convert currents from MA to A
    Ip = Ip * 1e6
    Ib = Ib * 1e6

    # Calculate the maximum magnetic field considering B_CS_max limit
    Bmax = B0 / (1 - ((a + b) / R0))
    
    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS calculation
    if Choice_Buck_Wedg == 'Bucking':
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
    # Total ramp up flux (ΨRampUp)
    ΨRampUp = Ψind + Ψres

    # Flux related to the plateau
    res = 2.8 * 10**-8 / Tbar**(3 / 2)  # Plasma resistivity [Ohm*m]
    σp = 1 / res  # Plasma conductivity [S/m]
    Vloop = abs(Ip - Ib) / σp * 2 * math.pi * R0 / (κ * a**2)
    Ψplateau = Vloop * Temps_Plateau

    # Available flux from PF system (ΨPF)
    if (8 * a / κ**(1 / 2)) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        ΨPF = μ0 * Ip / (4 * R0) * (βp + (Li - 3) / 2 + math.log(8 * a / κ**(1 / 2))) * (R0**2 - RCS_ext**2)

    # Theoretical expression of CS flux
    # ΨCS = (math.pi * μ0 * J_max_CS * Alpha) / 3 * (RCS_ext**3 - RCS_int**3)
    
#---------------------------------------------------------------------------------------------------------------------------------------- 

    #### Solving RCS_int ####
    
    def CS_to_solve(RCS_int):
        
        # Dilution factor
        Alpha = (3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (2 * np.pi * μ0 * J_max_CS * Flux_CS_Utile * (RCS_ext**3 - RCS_int**3))

        # Mechanical calculations
        # J cross B
        Sigma_JB = μ0 * (J_max_CS * Alpha)**2 * (RCS_ext - RCS_int) * RCS_int

        # Centering
        Sigma_centering = (B0**2 * R0) / (μ0 * (RCS_ext - RCS_int)) * (
            1/(2 - 2 * ((a + b) / R0) - (c / R0)) -
            1/(2 + 2 * ((a + b) / R0) + (c / R0))
        )

        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = np.nanmax([Sigma_centering, abs(Sigma_JB - Sigma_centering)])
            
            # Test parameter to check the mechanical configuration
            # if Sigma_centering > abs(Sigma_JB - Sigma_centering):
            #     CS_config_meca = 'Centering'
            # elif Sigma_centering <= abs(Sigma_JB - Sigma_centering):
            #     CS_config_meca = 'JB-Centering'
            # else :
            #     CS_config_meca = np.nan
            # print(CS_config_meca)
            
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = Sigma_JB
        else:
            print("Choose between Wedging and Bucking")

        return (Sigma_CS - (σ_CS * (1 - Alpha)))

    try:
        # Find the root of the equation
        RCS_initial_guess = RCS_ext - 0.5
        RCS_int_solution, info, ier, msg = fsolve(CS_to_solve, RCS_initial_guess, full_output=True)

        if ier != 1:
            if __name__ == "__main__":
                raise ValueError(f"CS did not converge: {msg}")
            return (np.nan, np.nan, np.nan)

        RCS_int_solution = RCS_int_solution[0]  # fsolve returns an array, we need the single value
        Alpha = (3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (Flux_CS_Utile * 2 * math.pi * μ0 * J_max_CS * (RCS_ext**3 - RCS_int_solution**3))
        B_CS = μ0 * (J_max_CS * Alpha) * (RCS_ext - RCS_int_solution)

#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results filtering ####
        
        if Alpha > 1 or Alpha < 0:
            return (np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        if RCS_int_solution < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, Alpha, B_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
def f_CS_2_layers_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
    
    """
    Calculate the CS thickness 
    with it's associated dilution and centering ratio

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    c : TF thickness (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_CS : Yield strength of the CS steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_CS : Maximum current density of the chosen Supra + Cu + He (A)
    Choice_Buck_Wedg : Mechanical configuration
    T_bar : The mean temperature [keV]
    n_bar : The mean electron density [1e20p/m^3]
    Ip : Plasma current [MA]
    Ib : Bootstrap current [MA]

    Returns:
    tuple: A tuple containing the calculated values :
    (d, Alpha, B_CS)
    With
    d : CS width
    Alpha : % of conductor
    B_CS : CS magnetic field
    
    """
#----------------------------------------------------------------------------------------------------------------------------------------    
    
    #### Preliminary results ####
    
    # Convert currents from MA to A
    Ip = Ip * 1e6
    Ib = Ib * 1e6

    # Calculate the maximum magnetic field considering B_CS_max limit
    Bmax = B0 / (1 - ((a + b) / R0))
    
    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS calculation
    if Choice_Buck_Wedg == 'Bucking':
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
    # Total ramp up flux (ΨRampUp)
    ΨRampUp = Ψind + Ψres

    # Flux related to the plateau
    res = 2.8 * 10**-8 / Tbar**(3 / 2)  # Plasma resistivity [Ohm*m]
    σp = 1 / res  # Plasma conductivity [S/m]
    Vloop = abs(Ip - Ib) / σp * 2 * math.pi * R0 / (κ * a**2)
    Ψplateau = Vloop * Temps_Plateau

    # Available flux from PF system (ΨPF)
    if (8 * a / κ**(1 / 2)) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        ΨPF = μ0 * Ip / (4 * R0) * (βp + (Li - 3) / 2 + math.log(8 * a / κ**(1 / 2))) * (R0**2 - RCS_ext**2)

    # Theoretical expression of CS flux
    # ΨCS = 2 (math.pi * μ0 * J_max_CS * Alpha) / 3 * (RCS_ext**3 - RCS_int**3)
    
    ri_c = np.cbrt(RCS_ext**3 - (( 3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (2 * np.pi * μ0 * J_max_CS * Flux_CS_Utile)))
    B_CS = μ0 * (J_max_CS) * (RCS_ext - ri_c)
#---------------------------------------------------------------------------------------------------------------------------------------- 
    #### Solving RCS_int ####
    
    def CS_to_solve(d_SS):
        
        RCS_int = ri_c - d_SS
        
        # Mechanical calculations
        # J cross B
        Sigma_JB = μ0 * J_max_CS**2 * (ri_c - RCS_int) * RCS_int

        # Centering
        P = Bmax**2/(2*μ0)
        Sigma_centering = P * RCS_int / d_SS

        if Choice_Buck_Wedg == 'Bucking':
            
            Sigma_CS = np.nanmax([Sigma_centering, abs(Sigma_JB - Sigma_centering)])
            
        elif Choice_Buck_Wedg == 'Wedging':
            
            Sigma_CS = Sigma_JB
            
        else:
            print("Choose between Wedging and Bucking")

        return (Sigma_CS - σ_CS)

    try:
        
        if Choice_Buck_Wedg == 'Bucking':
            
            d_SS_initial_guess = 0.5
            # Appliquer la méthode de résolution
            result = root(CS_to_solve, d_SS_initial_guess, method='lm') # Levenberg-Marquardt method
            # Récupérer la solution
            d_SS_solution = result.x[0]
            
        elif Choice_Buck_Wedg == 'Wedging':
            
            # Find the root of the equation
            d_SS_initial_guess = 0.5
            d_SS, info, ier, msg = fsolve(CS_to_solve, d_SS_initial_guess, full_output=True)
            d_SS_solution = d_SS[0]
            
        else:
            print("Choose between Wedging and Bucking")
        
        RCS_int_solution = ri_c - d_SS_solution
        alpha = (np.pi * ( RCS_ext**2 - ri_c**2 )) / (np.pi * (RCS_ext**2 - (ri_c - d_SS_solution)**2))

#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results filtering ####
        
        if d_SS_solution < 0 :
            return(np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        if RCS_int_solution < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, alpha, B_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan)

def f_CS_DOFUS_polynomiale(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
    """
    Calculate the CS thickness 
    with it's associated dilution and centering ratio

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    c : TF thickness (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_CS : Yield strength of the CS steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_CS : Maximum current density of the chosen Supra + Cu + He (A)
    Choice_Buck_Wedg : Mechanical configuration
    T_bar : The mean temperature [keV]
    n_bar : The mean electron density [1e20p/m^3]
    Ip : Plasma current [MA]
    Ib : Bootstrap current [MA]

    Returns:
    tuple: A tuple containing the calculated values :
    (d, Alpha, B_CS)
    With
    d : CS width
    Alpha : % of conductor
    B_CS : CS magnetic field
    
    """
#----------------------------------------------------------------------------------------------------------------------------------------    
    
    #### Preliminary results ####
    
    # Convert currents from MA to A
    Ip = Ip * 1e6
    Ib = Ib * 1e6

    # Calculate the maximum magnetic field considering B_CS_max limit
    Bmax = B0 / (1 - ((a + b) / R0))
    P = (
        1/(2 - 2 * ((a + b) / R0) - (c / R0)) -
        1/(2 + 2 * ((a + b) / R0) + (c / R0))
    )
    
    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]
    
    # Force due to the magnetic field
    F_z_TF = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))

    # External radius of the CS calculation
    if Choice_Buck_Wedg == 'Bucking':
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        Gap = 0.1
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
    # Total ramp up flux (ΨRampUp)
    ΨRampUp = Ψind + Ψres

    # Flux related to the plateau
    res = 2.8 * 10**-8 / Tbar**(3 / 2)  # Plasma resistivity [Ohm*m]
    σp = 1 / res  # Plasma conductivity [S/m]
    Vloop = abs(Ip - Ib) / σp * 2 * math.pi * R0 / (κ * a**2)
    Ψplateau = Vloop * Temps_Plateau

    # Available flux from PF system (ΨPF)
    if (8 * a / κ**(1 / 2)) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        ΨPF = μ0 * Ip / (4 * R0) * (βp + (Li - 3) / 2 + math.log(8 * a / κ**(1 / 2))) * (R0**2 - RCS_ext**2)

    # Theoretical expression of CS flux
    # ΨCS = (math.pi * μ0 * J_max_CS * Alpha) / 3 * (RCS_ext**3 - RCS_int**3)
    
#---------------------------------------------------------------------------------------------------------------------------------------- 
    # Polynomial solving in bucking
    
    if Choice_Buck_Wedg == 'Bucking':
        
        # Cste definition
        Ψ = abs((ΨPI + ΨRampUp + Ψplateau - ΨPF) / Flux_CS_Utile)
        
        def limiting_force(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, RCS_int):
            
            # Dilution factor
            Alpha = 3 * Ψ / (2 * np.pi * μ0 * J_max_CS * (RCS_ext**3 - RCS_int**3))

            # Mechanical calculations
            # J cross B
            Sigma_JB = μ0 * (J_max_CS * Alpha)**2 * (RCS_ext - RCS_int) * RCS_int
            # Centering
            Sigma_centering = (B0**2 * R0) / (μ0 * (RCS_ext - RCS_int)) * (
                1/(2 - 2 * ((a + b) / R0) - (c / R0)) -
                1/(2 + 2 * ((a + b) / R0) + (c / R0))
            )

            if Sigma_centering > abs(Sigma_JB - Sigma_centering) :
                return('Centering')
            elif Sigma_centering <= abs(Sigma_JB - Sigma_centering) :
                return('JB-Centering')
            else :
                return(np.nan)
        
        def calculate_rCS_int_solutions_Sigma_C(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, P):
            
            coef_3 = -2 * np.pi * J_max_CS * μ0 * σ_CS
            coef_2 = -2 * np.pi * B0**2 * J_max_CS * P * R0
            coef_1 = -2 * np.pi * B0**2 * J_max_CS * P * R0 * RCS_ext
            coef_0 = -2 * np.pi * B0**2 * J_max_CS * P * R0 * RCS_ext**2 + 2 * np.pi* J_max_CS * RCS_ext**3 * μ0 * σ_CS - 3 * Ψ * σ_CS
                  
            # Coefficients du polynôme
            coefficients = [coef_3, coef_2, coef_1, coef_0]
            
            # Use numpy.roots to calculate the roots
            if np.isnan(coefficients).any() or np.isinf(coefficients).any(): # Nan or infinit check
                return (np.nan)
            
            solutions = np.roots(coefficients)
            
            # Filtrer pour obtenir uniquement les racines réelles, supérieures à 0 et inférieures à RCS_ext
            filtred_solution = [r.real for r in solutions if np.isclose(r.imag, 0) and r.real > 0 and r.real < RCS_ext-0.01 and limiting_force(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, r.real)=='Centering']
            
            # Select the biggest real root
            RCS_int_solution = max(filtred_solution) if filtred_solution else np.nan
        
            return (RCS_int_solution)
        
        def calculate_rCS_int_solution_Sigma_JB_Sigma_C(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, P):
        
            # Définition des coefficients
            coef_6 = 4 * np.pi**2 * J_max_CS**2 * μ0**2 * σ_CS
            coef_5 = -4 * np.pi**2 * B0**2 * J_max_CS**2 * P * R0 * μ0
            coef_4 = -4 * np.pi**2 * B0**2 * J_max_CS**2 * P * R0 * RCS_ext * μ0
            coef_3 = -4 * np.pi**2 * B0**2 * J_max_CS**2 * P * R0 * RCS_ext**2 * μ0 - 8 * np.pi**2 * J_max_CS**2 * RCS_ext**3 * μ0**2 * σ_CS + 6 * np.pi * J_max_CS * Ψ * μ0 * σ_CS
            coef_2 = 4 * np.pi**2 * B0**2 * J_max_CS**2 * P * R0 * RCS_ext**3 * μ0 + 9 * J_max_CS**2 * Ψ**2 * μ0
            coef_1 = 4 * np.pi**2 * B0**2 * J_max_CS**2 * P * R0 * RCS_ext**4 * μ0 - 9 * J_max_CS**2 * RCS_ext * Ψ**2 * μ0
            coef_0 = 4 * np.pi**2 * B0**2 * J_max_CS**2 * P * R0 * RCS_ext**5 * μ0 + 4 * np.pi**2 * J_max_CS**2 * RCS_ext**6 * μ0**2 * σ_CS - 6 * np.pi * J_max_CS * RCS_ext**3 * Ψ * μ0 * σ_CS
            
            # Coefficients du polynôme
            coefficients = [coef_6,coef_5,coef_4, coef_3, coef_2, coef_1, coef_0]
            
            # Use numpy.roots to calculate the roots
            if np.isnan(coefficients).any() or np.isinf(coefficients).any(): # Nan or infinit check
                return (np.nan)
            
            solutions = np.roots(coefficients)
            
            # Filtrer pour obtenir uniquement les racines réelles, supérieures à 0 et inférieures à RCS_ext
            filtred_solution = [r.real for r in solutions if np.isclose(r.imag, 0) and r.real > 0 and r.real < RCS_ext-0.01 and limiting_force(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, r.real)=='JB-Centering']
            
            # Select the biggest real root
            RCS_int_solution = max(filtred_solution) if filtred_solution else np.nan
            
            return(RCS_int_solution)
        
        solution_sigma_C = calculate_rCS_int_solutions_Sigma_C(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, P)
        solution_sigma_JB_sigma_C = calculate_rCS_int_solution_Sigma_JB_Sigma_C(B0, R0, μ0, σ_CS, RCS_ext, J_max_CS, Ψ, P)
        
        RCS_int_solution = np.nanmax([solution_sigma_C,solution_sigma_JB_sigma_C])
        
        if RCS_int_solution is np.nan or None:
            return (np.nan, np.nan, np.nan)
        
        Alpha = (3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (Flux_CS_Utile * 2 * math.pi * μ0 * J_max_CS * (RCS_ext**3 - RCS_int_solution**3))
        B_CS = μ0 * (J_max_CS * Alpha) * (RCS_ext - RCS_int_solution)
        
        if Alpha > 1 or Alpha < 0:
            return (np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, Alpha, B_CS)
        
#---------------------------------------------------------------------------------------------------------------------------------------- 
    # Polynomial solving in wedging
        
    elif Choice_Buck_Wedg == 'Wedging':
        
        # Cste definition
        Ψ = abs((ΨPI + ΨRampUp + Ψplateau - ΨPF) / Flux_CS_Utile)

        # Définition des coefficients
        coef_5 = -4 * np.pi**2 * J_max_CS * σ_CS * μ0 / 9
        coef_4 = -4 * np.pi**2 * J_max_CS * σ_CS * μ0 * RCS_ext / 9
        coef_3 = -4 * np.pi**2 * J_max_CS * σ_CS * μ0 * RCS_ext**2 / 9
        coef_2 = 4 * np.pi**2 * J_max_CS * σ_CS * μ0 * RCS_ext**3 / 9 - 2 * np.pi * Ψ * σ_CS / 3
        coef_1 = -J_max_CS * Ψ**2 + 4 * np.pi**2 * J_max_CS * σ_CS * μ0 * RCS_ext**4 / 9 - 2 * np.pi * Ψ * σ_CS * RCS_ext / 3
        coef_0 = 4 * np.pi**2 * J_max_CS * σ_CS * μ0 * RCS_ext**5 / 9 - 2 * np.pi * Ψ * σ_CS * RCS_ext**2 / 3
        
        # Coefficients du polynôme
        coefficients = [coef_5, coef_4, coef_3, coef_2, coef_1, coef_0]

        # Use numpy.roots to calculate the roots
        if np.isnan(coefficients).any() or np.isinf(coefficients).any():
            return (np.nan, np.nan, np.nan)

        solutions = np.roots(coefficients)

        # Filter to get only the real roots
        filtred_solution = [r.real for r in solutions if np.isclose(r.imag, 0) and r.real > 0 and r.real < RCS_ext-0.01]

        # Select the biggest real root
        RCS_int_solution = max(filtred_solution) if filtred_solution else np.nan

        Alpha = (3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (Flux_CS_Utile * 2 * math.pi * μ0 * J_max_CS * (RCS_ext**3 - RCS_int_solution**3))
        B_CS = μ0 * (J_max_CS * Alpha) * (RCS_ext - RCS_int_solution)
        
        if Alpha > 1 or Alpha < 0:
            return (np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, Alpha, B_CS)
    else:
        print("Choose between Wedging and Bucking")
        return (np.nan, np.nan, np.nan)


if __name__ == "__main__":
    
    # # Test parameters JB-Centering
    # a = 0.5
    # b = 0.2
    # c = 0.2
    # R0 = 4
    # B0 = 2
    # σ_CS = 600e6
    # μ0 = 4 * np.pi * 1e-7
    # J_max_CS = 50e6
    # Tbar = 14
    # nbar = 1
    # Ip = 6
    # Ib = 3
    
    # Test parameters Centering (ITER)
    a = 2
    b = 1.45
    c = 0.9
    R0 = 6.2
    B0 = 5.3
    σ_CS = 660e6
    μ0 = 4 * np.pi * 1e-7
    J_max_CS = 50e6
    Tbar = 14
    nbar = 1
    Ip = 12
    Ib = 8

    # Test the function
    result_CS1 = f_CS_DOFUS_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Wedging', Tbar, nbar, Ip, Ib)
    print(f"CS test convergence Wedging : {result_CS1}")
    result_CS2 = f_CS_DOFUS_polynomiale(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Wedging', Tbar, nbar, Ip, Ib)
    print(f"CS test polynomiale Wedging : {result_CS2}")
    result_CS5 = f_CS_2_layers_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Wedging', Tbar, nbar, Ip, Ib)
    print(f"CS simple model Wedging : {result_CS5}")
    result_CS3 = f_CS_DOFUS_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Bucking', Tbar, nbar, Ip, Ib)
    print(f"CS test convergence Bucking : {result_CS3}")
    result_CS4 = f_CS_DOFUS_polynomiale(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Bucking', Tbar, nbar, Ip, Ib)
    print(f"CS test polynomiale Bucking : {result_CS4}")
    result_CS6 = f_CS_2_layers_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Bucking', Tbar, nbar, Ip, Ib)
    print(f"CS simple model Bucking : {result_CS6}")

#%% CIRCEE 0D

def MD1v2(ri, r0, re, Enmoins1, En, nu, J0, B0, J1, B1, Pi, config):
    """
    Calcul des contraintes et déplacements pour un cylindre épais avec body load en multi-couches.
    """
    K1nmoins1 = J0 * B0 / (r0 - ri)
    K1n = J1 * B1 / (re - r0)
    
    if config[0] == 1:
        K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
    else:
        K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
    
    if config[1] == 1:
        K2n = -J1 * B1 * r0 / (re - r0)
    else:
        K2n = -J1 * B1 * re / (re - r0)
    
    MD1 = (1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
      (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
      (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
      (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
      (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
      (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
      (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
      (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En) - \
          Pi * (-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2))

    return MD1

def MDendv2(ri, r0, re, Enmoins1, En, nu, J0, B0, J1, B1, Pe, config):
    """
    Calcul des contraintes et déplacements pour un cylindre épais avec body load en multi-couches.
    """
    K1nmoins1 = J0 * B0 / (r0 - ri)
    K1n = J1 * B1 / (re - r0)
    
    if config[0] == 1:
        K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
    else:
        K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
    
    if config[1] == 1:
        K2n = -J1 * B1 * r0 / (re - r0)
    else:
        K2n = -J1 * B1 * re / (re - r0)
    
    MDend = (1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
        (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
        (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
        (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
        (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
        (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
        (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
        (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En) - \
        Pe * (-2 * re**2) / (En * (re**2 - r0**2))

    return MDend

def MDv2(ri, r0, re, Enmoins1, En, nu, J0, B0, J1, B1, config):
    """
    Calcul des contraintes et déplacements pour un cylindre épais avec body load en multi-couches.
    """
    K1nmoins1 = J0 * B0 / (r0 - ri)
    K1n = J1 * B1 / (re - r0)
    
    if config[0] == 1:
        K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
    else:
        K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
    
    if config[1] == 1:
        K2n = -J1 * B1 * r0 / (re - r0)
    else:
        K2n = -J1 * B1 * re / (re - r0)
    
    MD = (1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
     (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
     (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
     (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
     (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
     (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
     (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
     (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En)

    return MD

def MG1v2(ri, r0, re, Enmoins1, En, nu):
    """
    Calcul des coefficients de la matrice de rigidité pour un cylindre épais avec body load en multi-couches.
    """
    MG1 = [(((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2))) + \
       (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2))),
       (-2*re**2)/(En*(re**2-r0**2)) ]
    
    return MG1
    
def MGendv2(ri, r0, re, Enmoins1, En, nu):
    """
    Calcul des coefficients de la matrice de rigidité pour un cylindre épais avec body load en multi-couches.
    """
    MGendv2 = [(-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2)), \
         (((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2))) + \
         (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2)))]

    return MGendv2

def MGv2(ri, r0, re, Enmoins1, En, nu):
    """
    Calcul des coefficients de la matrice de rigidité pour un cylindre épais avec body load en multi-couches.
    """
    MG = [(-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2)), \
      (((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2))) + \
      (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2))),\
         (-2*re**2)/(En*(re**2-r0**2)) ]

    return MG
   
    

def F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config):
    """
    F_CIRCE0D Summary of this function goes here
    Essai d'implémentation du calcul de cylindre épais avec body load en
    multi-couches
    """
    
   
    nlayer = len(E)

    if nlayer == 1:        

        SigRtot = []
        SigTtot = []
        urtot = []
        P = [Pi, Pe]
        Rvec = []
        
    elif nlayer == 2:
        ri = R[0]
        r0 = R[1]
        re = R[2]
        Enmoins1 = E[0]
        En = E[1]
        J0 = J[0]
        J1 = J[1]
        B0 = B[0]
        B1 = B[1]
        K1nmoins1 = J[0] * B[0] / (R[1] -R[0])
        K1n = J1 * B1 / (re - r0)
   
        if config[0] == 1:
            K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
        else:
            K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
        
        if config[1] == 1:
              K2n = -J1 * B1 * r0 / (re - r0)
        else:
            K2n = -J1 * B1 * re / (re - r0)
        
        MGtot = np.array([[((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2)) + \
                           (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2)))]])
        
        MDtot = np.array([[(1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
                           (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
                           (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
                       (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
                       (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
                       (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
                       (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
                       (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En) - \
                       Pi * (-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2)) - \
                       Pe * (-2 * re**2) / (En * (re**2 - r0**2))]])
    
        P = np.linalg.inv(MGtot) @ MDtot
        SigRtot = []
        SigTtot = []
        urtot = []
        P = [Pi, P[0], Pe]
        Rvec = []
        

    else:
        MDtot = np.zeros((nlayer - 1, 1))
        MGtot = np.zeros((nlayer - 1, nlayer - 1))
    
        for ilayer in range(2, nlayer+1):
            if ilayer == 2:
                MGtot[ilayer - 2, ilayer - 2:ilayer ] = MG1v2(R[ilayer - 2], R[ilayer - 1], R[ilayer],
                                                               E[ilayer - 2], E[ilayer - 1], nu)
                MDtot[ilayer - 2] = MD1v2(R[ilayer - 2], R[ilayer - 1], R[ilayer], E[ilayer - 2], E[ilayer - 1], nu,
                                         J[ilayer - 2], B[ilayer - 2], J[ilayer - 1], B[ilayer - 1], Pi,
                                         [config[ilayer - 2], config[ilayer - 1]])
            elif ilayer == nlayer:
                MGtot[ilayer - 2, ilayer - 3:ilayer - 1] = MGendv2(R[ilayer - 2], R[ilayer - 1], R[ilayer],
                                                                 E[ilayer - 2], E[ilayer - 1], nu)
                MDtot[ilayer - 2] = MDendv2(R[ilayer - 2], R[ilayer - 1], R[ilayer], E[ilayer - 2], E[ilayer - 1], nu,
                                            J[ilayer - 2], B[ilayer - 2], J[ilayer - 1], B[ilayer - 1], Pe,
                                            [config[ilayer - 2], config[ilayer - 1]])
            else:
                MGtot[ilayer - 2, ilayer - 3 : ilayer] = MGv2(R[ilayer - 2], R[ilayer - 1], R[ilayer],
                                                               E[ilayer - 2], E[ilayer - 1], nu)
                MDtot[ilayer - 2] = MDv2(R[ilayer - 2], R[ilayer - 1], R[ilayer], E[ilayer - 2], E[ilayer - 1], nu,
                                         J[ilayer - 2], B[ilayer - 2], J[ilayer - 1], B[ilayer - 1],
                                         [config[ilayer - 2], config[ilayer - 1]])
    
        print(MGtot)
        print(MDtot)

        P = np.linalg.inv(MGtot) @ MDtot
        SigRtot = []
        SigTtot = []
        urtot = []
        P = [[Pi], P,[ Pe]]
        Rvec = []
        P = np.vstack(P)
        print(P)

    for ilayer in range(nlayer):
        K1 = J[ilayer] * B[ilayer] / (R[ilayer + 1] - R[ilayer])
        if config[ilayer] == 1:
            K2 = -J[ilayer] * B[ilayer] * R[ilayer] / (R[ilayer + 1] - R[ilayer])
        else:
            K2 = -J[ilayer] * B[ilayer] * R[ilayer + 1] / (R[ilayer + 1] - R[ilayer])

        re = R[ilayer + 1]
        ri = R[ilayer]
        r = np.linspace(R[ilayer], R[ilayer + 1],  disR)
        
        C1 = K1 * (nu + 3) / 8 + K2 * (nu + 2) / (3 * (re + ri)) - (P[ilayer] - P[ilayer + 1]) / (re**2 - ri**2)
        C2 = (K2 * (nu + 2) / 3) * ((re**2 + ri**2 + re * ri) / (re + ri)) + \
             (P[ilayer + 1] * re**2 - P[ilayer] * ri**2) / (re**2 - ri**2) + \
             (re**2 + ri**2) * K1 * (nu + 3) / 8
        
        SigR = re**2 * ri**2 / r**2 * C1 + K1 * (nu + 3) / 8 * r**2 + K2 * (nu + 2) / 3 * r - C2
        SigT = -re**2 * ri**2 / r**2 * C1 + K1 * (3 * nu + 1) / 8 * r**2 + K2 * (2 * nu + 1) / 3 * r - C2
        ur = r / E[ilayer] * (-re**2 * ri**2 / r**2 * C1 * (1 + nu) + (1 - nu**2) / 8 * K1 * r**2 + \
                               (1 - nu**2) / 3 * K2 * r - C2 * (1 - nu))
        
        SigRtot.append(SigR)
        SigTtot.append(SigT)
        urtot.append(ur)
        Rvec.append(r)
      
    SigRtot = np.concatenate(SigRtot)
    SigTtot = np.concatenate(SigTtot)
    urtot = np.concatenate(urtot)
    Rvec = np.concatenate(Rvec)  

    return SigRtot, SigTtot, urtot, Rvec, P

#  Définir les valeurs d'entrée pour le cas 'ITER'

if __name__ == "__main__":
    
    disR = 100  # Pas de discrétisation
    R = np.array([1.317,2.057])  # Radii
    J = np.array([6*554*40000/((R[1]-R[0])*12.6)])  # Current densities
    B = np.array([13])  # Magnetic fields
    Pi = 0  # Internal pressure
    Pe = 50e6  # External pressure
    E = np.array([120e9])  # Young's moduli
    nu = 0.29  # Poisson's ratio
    config = np.array([0])  # Loading configs
    # Appeler la fonction principale
    SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
    # Tracer les résultats le long des valeurs de grand rayon R
    plt.figure(figsize=(12, 6))
    # Tracer les contraintes radiales
    plt.plot(Rvec, np.abs(SigRtot/1e6), label='Radial Stresses (SigRtot)')
    plt.plot(Rvec, np.abs( SigTtot/1e6), label='Tangential Stresses (SigTtot)')
    plt.plot(Rvec, np.abs(SigRtot/1e6)+np.abs(SigTtot/1e6), label='Tresca')
    plt.xlabel('Radius')
    plt.ylabel('Radial Stress (MPa)')
    plt.title('Radial Stresses vs Radius')
    plt.legend()
    plt.grid(True)
    
    # Tracer les déplacements radiaux
    plt.figure(figsize=(12, 6))
    plt.plot(Rvec, urtot, label='Radial Displacements (urtot)')
    plt.xlabel('Radius')
    plt.ylabel('Radial Displacement (m)')
    plt.title('Radial Displacements vs Radius')
    plt.legend()
    plt.grid(True)
    # Afficher les plots
    plt.show()

#%% Print

print("D0FUS_radial_build_functions loaded")