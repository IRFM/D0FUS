# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:32:03 2025

@author: TA276941
"""

#%% Import

from D0FUS_physical_functions import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% Radial Build functions

def f_TF_winding_pack(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
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
    c : TF width
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
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))

        # Sigma_C
        Sigma_C = (2 * B0**2 * R0) / (μ0 * r2) * (
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
        
        print(Dilution_solution)
        print(TF_to_solve(Dilution_solution))
        
        # Calculate r2 to generate the magnetic field
        r2 = r1 - (Bmax / (μ0 * J_max_TF * Dilution_solution))
        
        # Calculate Sigma ratio
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (2 * B0**2 * R0) / (μ0 * r2) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_bucking = Sigma_C / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_bucking)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
def f_TF_winding_pack_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
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
    c : TF width
    Dilution_solution : % of conductor
    TF_ratio_bucking : % of centering stress in the tresca
    
    """

    # Preliminary results
    r1 = R0 - a - b
    F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
    F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
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
            4 * np.pi * B0**2 * J_max_TF * R0**2
            + 4 * np.pi * Bmax * R0 * σ_TF * epsilon_B
            + np.pi * Bmax * σ_TF * r1
            - F_T * J_max_TF * μ0
            - 4 * np.pi * J_max_TF * R0**2 * σ_TF * epsilon_B**2 * μ0
            + 4 * np.pi * J_max_TF * R0**2 * σ_TF * μ0
            - 4 * np.pi * J_max_TF * R0 * σ_TF * epsilon_B * μ0 * r1
        ) / (J_max_TF * R0**2)
        
        # r2**2
        cste_d = (
            -8 * np.pi * B0**2 * J_max_TF * R0**3 * epsilon_B
            - 4 * np.pi * B0**2 * J_max_TF * R0**2 * r1
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
            -4 * np.pi * B0**2 * J_max_TF * R0**2 * r1**2
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
        cste_f = 8 * np.pi * B0**2 * R0 * epsilon_B * r1**2 + 4 * np.pi * B0**2 * r1**3
        # Coefficients du polynôme
        coefficients = [cste_a, cste_b, cste_c, cste_d, cste_e, cste_f]

        # Vérification des coefficients
        if np.isnan(coefficients).any() or np.isinf(coefficients).any():
            return None

        # Résolution du polynôme
        solutions = np.roots(coefficients)
        print(solutions)
        
        # Sélectionner la plus grande racine réelle inférieure à r1 et supérieure à 0
        largest_real_root_under_r1 = max(
            [sol.real for sol in solutions if np.isclose(sol.imag, 0) and sol.real < r1  and sol.real > 0], 
            default=None
        )
        
        r2 = largest_real_root_under_r1
        
        Dilution_solution = Bmax / (μ0 * J_max_TF * (r1 - r2))

        # Calculate Sigma ratio
        F_z = (np.pi / μ0) * B0**2 * R0**2 * np.log((1 + (a + b) / R0) / (1 - (a + b) / R0))
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (2 * B0**2 * R0) / (μ0 * r2) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_bucking = Sigma_C / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_bucking)

    except Exception as e:
        return (np.nan, np.nan, np.nan)

if __name__ == "__main__":
    # ITER test parameters
    a = 2
    b = 1.45
    R0 = 6.2
    B0 = 5.3
    σ_TF = 600e6
    μ0 = 4 * np.pi * 1e-7
    J_max_TF = 50 * 1e6
    F_CClamp = 0
    Bmax = 12

    # Test the function
    result = f_TF_winding_pack(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS winding pack test : {result}")
    result2 = f_TF_winding_pack_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS winding pack polynomial test : {result2}")
    
    
def f_TF(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax, Choice_Buck_Wedg):
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
    c : TF width
    TF_ratio_wedging : % of centering stress in the tresca
    
    """
    
    r1 = R0 - a - b
    winding_pack = f_TF_winding_pack(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)

    if Choice_Buck_Wedg == 'Bucking':
        
        # In bucking, only winding pack is considered

        Winding_pack_thickness = winding_pack[0]
        Winding_pack_dilution = winding_pack[1]
        Winding_pack_centering_ratio = winding_pack[2]
        Nose_thickness = None
        c = Winding_pack_thickness # TF thickness
        
        return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_centering_ratio, Nose_thickness)
        
    elif Choice_Buck_Wedg == 'Wedging':
        
        # In wedging, one need to add the steel nose to the winding pack
        
        # Extraction des résultats du winding pack
        Winding_pack_thickness = winding_pack[0]
        Winding_pack_dilution = winding_pack[1]
        Winding_pack_centering_ratio = winding_pack[2]

        # Calculate the strain parameter eps_B
        eps_B = (a + b) / R0
        eps_C = Winding_pack_thickness / R0
    
        # Calculate the thickness of the steel nose
        Nose_thickness = (B0**2 * R0 / (μ0 * σ_TF)) * ((1 / (2 - 2 * eps_B - eps_C)) - (1 / (2 + 2 * eps_B + eps_C)))
        
        c = Winding_pack_thickness + Nose_thickness # TF thickness
        
        return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_centering_ratio, Nose_thickness)

    else:
        print("Please provide a valid argument : Wedging or Bucking")
        
if __name__ == "__main__":
    # Test parameters for ITER case
    a = 2
    b = 1.45
    R0 = 6.2
    B0 = 5.3
    σ_TF = 600e6
    μ0 = 4 * np.pi * 1e-7
    J_max_TF = 50 * 1e6
    Bmax = 12
    F_CClamp = 0
    Choice_Buck_Wedg = "Wedging"

    # Test the function
    result = f_TF(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax, Choice_Buck_Wedg)
    print(f"TF D0FUS test : {result}")

def f_TF_winding_pack_wedging_test(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax):
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
    c : TF width
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
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))

        # Sigma_C
        Sigma_C = (2 * B0**2 * R0) / (μ0 * 2 * (r1-r2)) * (
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
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (2 * B0**2 * R0) / (μ0 * 2 * (r1-r2)) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_bucking = Sigma_C / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_bucking)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
if __name__ == "__main__":
    # ITER test parameters
    a = 2
    b = 1.45
    R0 = 6.2
    B0 = 5.3
    σ_TF = 600e6
    μ0 = 4 * np.pi * 1e-7
    J_max_TF = 50 * 1e6
    F_CClamp = 0
    Bmax = 12

    # Test the function
    result = f_TF_winding_pack_wedging_test(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS wedging full winding pack test : {result}")

def f_TF_coil_freidberg(a, b, R0, B0, σ_TF, μ0, J_max_TF):
    """
    Calculate the thickness and stress ratio for the TF coil using the Freidberg model

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
    TF_ratio_wedging = Sigma_C / (Sigma_C + Sigma_T)

    return (c, TF_ratio_wedging)

if __name__ == "__main__":
    # ITER Test parameters
    a = 2
    b = 1.45
    R0 = 6.2
    B0 = 5.3
    σ_TF = 600e6
    μ0 = 4 * np.pi * 1e-7
    J_max_TF = 20 * 1e6

    # Test the function
    result = f_TF_coil_freidberg(a, b, R0, B0, σ_TF, μ0, J_max_TF)
    print(f"TF freidberg test : {result}")

def f_CS_coil(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
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
    c : TF width
    Alpha : % of conductor
    B_CS : CS magnetic field
    
    """

    # Convert currents from MA to A
    Ip = Ip * 1e6
    Ib = Ib * 1e6

    # Calculate the maximum magnetic field considering B_CS_max limit
    Bmax = B0 / (1 - ((a + b) / R0))

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

    # Flux calculation
    # ----------------------------------------------------Calculate flux needed for plasma initiation (ΨPI)
    ΨPI = ITERPI  # Initiation consumption from ITER 20 [Wb]

    # ----------------------------------------------------Calculate flux needed for the inductive part (Ψind)
    if (8 * R0 / (a * math.sqrt(κ))) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        Lp = 1.07 * μ0 * R0 * (1 + 0.1 * βp) * (Li / 2 - 2 + math.log(8 * R0 / (a * math.sqrt(κ))))
        Ψind = Lp * Ip

    # ---------------------------------------------------Calculate flux related to the resistive part (Ψres)
    Ψres = Ce * μ0 * R0 * Ip

    # ---------------------------------------------------Calculate total ramp up flux (ΨRampUp)
    ΨRampUp = Ψind + Ψres

    # ---------------------------------------------------Calculate flux related to the plateau
    Temps_Plateau = 0
    res = 2.8 * 10**-8 / Tbar**(3 / 2)  # Plasma resistivity [Ohm*m]
    σp = 1 / res  # Plasma conductivity [S/m]
    Vloop = abs(Ip - Ib) / σp * 2 * math.pi * R0 / (κ * a**2)
    Ψplateau = Vloop * Temps_Plateau

    # ---------------------------------------------------Calculate available flux from PF system (ΨPF)
    if (8 * a / κ**(1 / 2)) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        ΨPF = μ0 * Ip / (4 * R0) * (βp + (Li - 3) / 2 + math.log(8 * a / κ**(1 / 2))) * (R0**2 - RCS_ext**2)

    # Theoretical expression of CS flux
    # ΨCS = (math.pi * μ0 * J_max_CS * Alpha) / 3 * (RCS_ext**3 - RCS_int**3)

    CS_solving = "Polynomial"  # Convergence or Polynomial

    # Convergence loop
    if CS_solving == "Convergence":
        # Re-injecting the dilution factor Alpha, calculated from flux consumption onto the mechanical equation
        def CS_to_solve(RCS_int):
            # Dilution factor
            Alpha = (3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (Flux_CS_Utile * 2 * math.pi * μ0 * J_max_CS * (RCS_ext**3 - RCS_int**3))

            # Mechanical calculations
            # J cross B
            Sigma_JB = μ0 * (J_max_CS * Alpha)**2 * (RCS_ext - RCS_int) * RCS_int

            # Centering
            Sigma_centering = 2 * B0 * R0 / μ0 * (
                (2 - 2 * ((a + b) / R0) - ((RCS_ext - RCS_int) / R0))**(-1) -
                (2 + 2 * ((a + b) / R0) + ((RCS_ext - RCS_int) / R0))**(-1)
            )

            if Choice_Buck_Wedg == 'Bucking':
                Sigma_CS = max(Sigma_centering, (Sigma_JB - Sigma_centering))
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

            # J cross B
            Sigma_JB = μ0 * (J_max_CS * Alpha)**2 * (RCS_ext - RCS_int_solution) * RCS_int_solution

            # Centering due to bucking
            Sigma_centering = 2 * B0 * R0 / μ0 * (
                (2 - 2 * ((a + b) / R0) - ((RCS_ext - RCS_int_solution) / R0))**(-1) -
                (2 + 2 * ((a + b) / R0) + ((RCS_ext - RCS_int_solution) / R0))**(-1)
            )

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

    elif CS_solving == "Polynomial":
        Ψ = abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)
        cste_a = σ_CS * 4 * np.pi**2 * μ0 * J_max_CS / 9
        cste_b = σ_CS * Ψ * 2 * np.pi / 3
        cste_c = J_max_CS * Ψ**2

        # Define the coefficients of the polynomial
        coefficients = [
            -cste_a,                                            # Coefficient of RCS_int^5
            -cste_a * RCS_ext,                                  # Coefficient of RCS_int^4
            -cste_a * RCS_ext**2,                               # Coefficient of RCS_int^3
            (cste_a * RCS_ext**3 - cste_b),                     # Coefficient of RCS_int^2
            (cste_a * RCS_ext**4 - cste_b * RCS_ext - cste_c),  # Coefficient of RCS_int^1
            (cste_a * RCS_ext**5 - cste_b * RCS_ext**2)         # Coefficient of RCS_int^0
        ]

        # Use numpy.roots to calculate the roots
        if np.isnan(coefficients).any() or np.isinf(coefficients).any():
            return (np.nan, np.nan, np.nan)

        solutions = np.roots(coefficients)

        # Filter to get only the real roots
        racines_reelles = [r.real for r in solutions if np.isclose(r.imag, 0)]

        # Select the smallest real root
        RCS_int_solution = max(racines_reelles) if racines_reelles else None

        Alpha = (3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (Flux_CS_Utile * 2 * math.pi * μ0 * J_max_CS * (RCS_ext**3 - RCS_int_solution**3))
        B_CS = μ0 * (J_max_CS * Alpha) * (RCS_ext - RCS_int_solution)
        if Alpha > 1 or Alpha < 0:
            return (np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        if RCS_int_solution < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, Alpha, B_CS)

if __name__ == "__main__":
    # Test parameters
    a = 3
    b = 2
    c = 2
    R0 = 9
    B0 = 5.5
    σ_CS = 600e6
    μ0 = 4 * np.pi * 1e-7
    J_max_CS = 50e6
    Choice_Buck_Wedg = 'Bucking'
    Tbar = 14
    nbar = 1
    Ip = 15
    Ib = 8

    # Test the function
    result = f_CS_coil(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib)
    print(f"CS test : {result}")
    
#%%

print("D0FUS_radial_build_functions loaded")