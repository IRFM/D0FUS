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
            return (np.nan,np.nan,np.nan)

        # Résolution du polynôme
        solutions = np.roots(coefficients)
        
        # Sélectionner la plus grande racine réelle inférieure à r1 et supérieure à 0
        largest_real_root_under_r1 = max(
            [sol.real for sol in solutions if np.isclose(sol.imag, 0) and sol.real < r1-0.01  and sol.real > 0], 
            default=np.nan
        )
        
        if np.isnan(largest_real_root_under_r1).any() :
            return (np.nan,np.nan,np.nan)
        
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
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (B0**2 * R0) / (μ0 * (r1-r2)) * (
            (2 - 2 * (a + b) / R0 - (r1 - r2) / R0)**-1 -
            (2 + 2 * (a + b) / R0 + (r1 - r2) / R0)**-1
        )
        TF_ratio_bucking = Sigma_C / (Sigma_C + Sigma_T)
        
        # Calculate associated thickness
        c_winding_pack = r1 - r2

        return (c_winding_pack, Dilution_solution,TF_ratio_bucking)

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
        F_T = max(abs(F_z - F_CClamp), abs(F_CClamp)) / 2
        Sigma_T = F_T / (np.pi * (r1**2 - r2**2))
        Sigma_C = (B0**2 * R0) / (μ0 * (r1-r2)) * (
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
    result = f_TF_winding_pack_bucking_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS bucking winding pack convergence test : {result}")
    result2 = f_TF_winding_pack_bucking_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS bucking winding pack polynomial test : {result2}")
    result3 = f_TF_winding_pack_wedging_convergence(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS wedging winding pack convergence : {result3}")
    result6 = f_TF_winding_pack_wedging_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
    print(f"TF D0FUS wedging winding pack polynomiale : {result6}")
    
    
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
    c : TF width
    TF_ratio_wedging : % of centering stress in the tresca
    
    """
    
    r1 = R0 - a - b

    if Choice_Buck_Wedg == 'Bucking':
        
        winding_pack = f_TF_winding_pack_bucking_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
        
        # In bucking, only winding pack is considered

        Winding_pack_thickness = winding_pack[0]
        if Winding_pack_thickness == 0:
                return(np.nan,np.nan,np.nan,np.nan)
        Winding_pack_dilution = winding_pack[1]
        Winding_pack_centering_ratio = winding_pack[2]
        Nose_thickness = np.nan
        c = Winding_pack_thickness # TF thickness
        
        return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_centering_ratio, Nose_thickness)
        
    elif Choice_Buck_Wedg == 'Wedging':
        
        if Mechanical_model == 'Winding_pack_and_Nose': # Full_winding_pack or Winding_pack_and_Nose
        
            # In wedging, one need to add the steel nose to the bucking winding pack
            winding_pack = f_TF_winding_pack_bucking_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
            
            # Extraction des résultats du winding pack
            Winding_pack_thickness = winding_pack[0]
            if Winding_pack_thickness == 0:
                    return(np.nan,np.nan,np.nan,np.nan)
            Winding_pack_dilution = winding_pack[1]
            Winding_pack_centering_ratio = winding_pack[2]
    
            # Calculate the strain parameter eps_B
            eps_B = (a + b) / R0
            eps_C = Winding_pack_thickness / R0
        
            # Calculate the thickness of the steel nose
            Nose_thickness = (B0**2 * R0 / (μ0 * σ_TF)) * ((1 / (2 - 2 * eps_B - eps_C)) - (1 / (2 + 2 * eps_B + eps_C)))
            
            c = Winding_pack_thickness + Nose_thickness # TF thickness
            
            return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_centering_ratio, Nose_thickness)
        
        elif Mechanical_model == 'Full_winding_pack': # Full_winding_pack or Winding_pack_and_Nose
        
            # A single winding pack is dimensionned taking into account the vault effect
            # winding_pack = f_TF_winding_pack_wedging_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)
            winding_pack = f_TF_winding_pack_wedging_polynomiale(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax)

            Winding_pack_thickness = winding_pack[0]
            if Winding_pack_thickness == 0:
                    return(np.nan,np.nan,np.nan,np.nan)
            Winding_pack_dilution = winding_pack[1]
            Winding_pack_centering_ratio = winding_pack[2]
            Nose_thickness = np.nan
            c = Winding_pack_thickness # TF thickness
            
            return(c, Winding_pack_thickness, Winding_pack_dilution, Winding_pack_centering_ratio, Nose_thickness)
        
        else:
            print("Please provide a valid argument : Full_winding_pack or Winding_pack_and_Nose")

    else:
        print("Please provide a valid argument : Wedging or Bucking")

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
    result4 = f_TF_DOFUS(a, b, R0, B0, σ_TF, μ0, J_max_TF, F_CClamp, Bmax, Choice_Buck_Wedg)
    print(f"TF D0FUS test : {result4}")
    result5 = f_TF_coil_freidberg(a, b, R0, B0, σ_TF, μ0, 20e6)
    print(f"TF freidberg test : {result5}")
    
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
    c : TF width
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
            Sigma_CS = max(Sigma_centering, abs(Sigma_JB - Sigma_centering))
            
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
    c : TF width
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
            coef_7 = -8 * np.pi**3 * B0**2 * J_max_CS * P * R0 * μ0
            coef_6 = -16 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext * μ0
            coef_5 = -24 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext**2 * μ0
            coef_4 = -8 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext**3 * μ0 + 18 * np.pi * J_max_CS * Ψ**2 * μ0
            coef_3 = 8 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext**4 * μ0 - 2 * np.pi * J_max_CS * μ0 * σ_CS
            coef_2 = 24 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext**5 * μ0
            coef_1 = 16 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext**6 * μ0 - 18 * np.pi * J_max_CS * Ψ**2 * RCS_ext**3 * μ0
            coef_0 = 8 * np.pi**3 * B0**2 * J_max_CS * P * R0 * RCS_ext**7 * μ0 + 2 * np.pi * J_max_CS * RCS_ext**3 * μ0 * σ_CS - 3 * Ψ * σ_CS
                  
            # Coefficients du polynôme
            coefficients = [coef_7,coef_6,coef_5,coef_4, coef_3, coef_2, coef_1, coef_0]
            
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
        
        RCS_int_solution = max(solution_sigma_C,solution_sigma_JB_sigma_C)
        
        if RCS_int_solution is np.nan or None:
            return (np.nan, np.nan, np.nan)
        
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
        if RCS_int_solution < 0:
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
    
    # Test parameters Centering
    a = 2
    b = 1.45
    c = 0.9
    R0 = 6.2
    B0 = 5.3
    σ_CS = 600e6
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
    result_CS3 = f_CS_DOFUS_convergence(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Bucking', Tbar, nbar, Ip, Ib)
    print(f"CS test convergence Bucking : {result_CS3}")
    result_CS4 = f_CS_DOFUS_polynomiale(a, b, c, R0, B0, σ_CS, μ0, J_max_CS, 'Bucking', Tbar, nbar, Ip, Ib)
    print(f"CS test polynomiale Bucking : {result_CS4}")
    
#%% Print

print("D0FUS_radial_build_functions loaded")