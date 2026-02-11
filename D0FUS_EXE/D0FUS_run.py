"""
D0FUS Run 5 Module - Complete Version
Generates a single design point with full output
Author: Auclair Timothe
"""

#%% Imports standards

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_BIB.D0FUS_cost import *
from D0FUS_BIB.D0FUS_cost_data import *

#%% Code

class CostParameters:
    def __init__(self, model):
        
        self.Util_factor = 0.9
        self.Dwell_factor = 0.95
        self.dt_rep = 0.1
        self.discount_rate = 0.05
        self.T_life = 30
        if model == 'Whyte':
            self.F_dpa = 10
            self.L_dpa = 25
        elif model == 'Sheffield':
            self.contingency = 0.15
            self.T_build = 5
        else:
            raise ValueError(f"\n ERROR : !!! Unknown cost model: {model} !!! \n")
            
    def to_dict(self):
        return {
            k: v for k, v in vars(self).items()
            if not k.startswith("_")
        }

class Parameters:
    """Main input container"""

    def __init__(self):
        self.P_fus = 2000
        self.R0 = 9
        self.a = 3
        self.Bmax = 12
        self.b = 1.2

        self.Option_Kappa = 'Wenninger'
        self.kappa_manual = 1.7

        self.Supra_choice = 'Nb3Sn'
        self.Radial_build_model = 'D0FUS'
        self.Choice_Buck_Wedg = 'Wedging'
        self.Chosen_Steel = '316L'

        self.Scaling_Law = 'IPB98(y,2)'
        self.L_H_Scaling_choice = 'New_Ip'
        self.Bootstrap_choice = 'Freidberg'

        self.H = 1
        self.Tbar = 14
        self.nu_T = 1
        self.nu_n = 0.1

        self.Operation_mode = 'Steady-State'

        if self.Operation_mode == 'Pulsed':
            self.Temps_Plateau_input = 120 * 60
            self.P_aux_input = 100
            self.fatigue = 2 if self.Choice_Buck_Wedg == 'Wedging' else 1
        else:
            self.Temps_Plateau_input = 0
            self.P_aux_input = 0
            self.fatigue = 1
            
        self.cost_model = 'Whyte'
        self.cost = CostParameters(self.cost_model)
    
    
    def parse_assignments(self, lines):
        """Parse input file lines, skipping scan parameters (those with brackets)"""
        result = []
        cleaned_lines = []
        for line in lines:
            line = line.split('#')[0].strip()
            if line:
                cleaned_lines.append(line)
        
        for line in cleaned_lines:
            if '=' not in line:
                continue
            
            var, val = line.split("=", 1)
            var = var.strip()
            val = val.strip()
            
            # Skip scan parameters (those with brackets [min, max, n])
            if val.startswith('[') and val.endswith(']'):
                print(f"Skipping scan parameter: {var} = {val}")
                continue
            
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except ValueError:
                pass
            result.append([var, val])
        return result
    
    def open_input(self, filepath):
        """Load parameters from input file"""
        with open(filepath, "r", encoding='utf-8') as f:
            read = f.read()
        
        lines = read.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        
        inputs = self.parse_assignments(lines)
            
        for var, val in inputs:
            if var == 'cost_model':
                self.cost_model = val
                self.cost = CostParameters(self.cost_model)
                print(f"Input change : self.cost_model = {val}")
                print(f"Rebuilding cost model with '{val}' parameters")
            elif hasattr(self.cost, var):
                setattr(self.cost, var, val)
                print(f"Input change : self.cost.{var} = {val}")
            else:
                setattr(self, var, val)
                print(f"Input change : self.{var} = {val}")


class Results(dict):
    """Dictionary-like container for run outputs"""
    pass


def run(p: Parameters):
    """
    Main calculation function
    Returns all output parameters
    """
    # Unpack dimensioning parameters
    a = p.a
    R0 = p.R0
    Bmax = p.Bmax
    P_fus = p.P_fus
    Tbar = p.Tbar
    H = p.H
    Temps_Plateau_input = p.Temps_Plateau_input 
    b = p.b
    nu_n = p.nu_n
    nu_T = p.nu_T
    Supra_choice = p.Supra_choice
    Chosen_Steel = p.Chosen_Steel 
    Radial_build_model = p.Radial_build_model 
    Choice_Buck_Wedg = p.Choice_Buck_Wedg
    Option_Kappa = p.Option_Kappa 
    kappa_manual = p.kappa_manual 
    L_H_Scaling_choice = p.L_H_Scaling_choice 
    Scaling_Law = p.Scaling_Law 
    Bootstrap_choice = p.Bootstrap_choice 
    Operation_mode = p.Operation_mode 
    fatigue = p.fatigue 
    P_aux_input = p.P_aux_input
    
    # Unpack cost model
    cost_model = p.cost_model
    
    
    # Stress limits in steel
    σ_TF = Steel(Chosen_Steel)
    σ_CS = Steel(Chosen_Steel) / fatigue
    
    # Current densities in coils
    if Supra_choice == 'Rebco':
        J_max_TF_conducteur = Jc(Supra_choice, Bmax, T_helium) * f_Cu_Strand * f_Cool * f_In
        J_max_CS_conducteur = Jc(Supra_choice, Bmax, T_helium) * f_Cu_Strand * f_Cool * f_In
    else:
        J_max_TF_conducteur = Jc(Supra_choice, Bmax, T_helium) * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        J_max_CS_conducteur = Jc(Supra_choice, Bmax, T_helium) * f_Cu_Non_Cu * f_Cu_Strand * f_Cool * f_In
        
    # Fraction of vertical tension allocated to the winding pack of the TF coils
    if Choice_Buck_Wedg == "Wedging":
        omega_TF = 1/2
    elif Choice_Buck_Wedg == "Bucking" or Choice_Buck_Wedg == "Plug":
        omega_TF = 1
    else: 
        print('Choose a valid mechanical configuration')
        
    # Confinement time scaling law parameters
    (C_SL, alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
     alpha_R, alpha_B, alpha_n, alpha_I, alpha_P) = f_Get_parameter_scaling_law(Scaling_Law)

    # Plasma geometry
    kappa = f_Kappa(R0/a, Option_Kappa, kappa_manual, ms)
    kappa_95 = f_Kappa_95(kappa)
    δ = f_Delta(kappa)
    δ_95 = f_Delta_95(δ)
    Volume_solution = f_plasma_volume(R0, a, kappa, δ)
    Surface_solution = f_surface_premiere_paroi(kappa, R0, a)
    
    # Central magnetic field
    B0_solution = f_B0(Bmax, a, b, R0)
    # Alpha power
    P_Alpha = f_P_alpha(P_fus, E_ALPHA, E_N)
    
    # Function to solve both the alpha particle fraction f_alpha and Q
    def to_solve_f_alpha_and_Q(vars):
        
        f_alpha, Q = vars
        
        # Sanity check
        if f_alpha > 1 or f_alpha < 0:
            return[1e10,1e10]
        if Q < 0:
            return[1e10,1e10]
        
        # Density and pressure
        nbar_alpha = f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, kappa)
        pbar_alpha = f_pbar(nu_n, nu_T, nbar_alpha, Tbar, f_alpha)
        # Radiative losses
        P_Brem_alpha = f_P_bremsstrahlung(Volume_solution, nbar_alpha, Tbar, Zeff, R0, a)
        P_syn_alpha = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_alpha, kappa, nu_n, nu_T, r_synch)
        P_rad_alpha = P_Brem_alpha + P_syn_alpha

        # Initialize power inputs based on operation mode
        if Operation_mode == 'Steady-State':
            P_Aux_alpha_init = abs(P_fus / Q)
            P_Ohm_alpha_init = 0
        elif Operation_mode == 'Pulsed':
            P_Aux_alpha_init = abs(P_aux_input)
            P_Ohm_alpha_init = abs(P_fus / Q - P_Aux_alpha_init)
        else:
            print("Choose a valid operation mode")
        # Confinement time calculation
        tau_E_alpha = f_tauE(pbar_alpha, Volume_solution, P_Alpha, P_Aux_alpha_init, P_Ohm_alpha_init, P_rad_alpha)
        # Associated Plasma Current
        Ip_alpha = f_Ip(tau_E_alpha, R0, a, kappa, δ, nbar_alpha, B0_solution, Atomic_mass,
                        P_Alpha, P_Ohm_alpha_init, P_Aux_alpha_init, P_rad_alpha,
                        H, C_SL,
                        alpha_delta, alpha_M, alpha_kappa, alpha_epsilon, alpha_R, alpha_B, alpha_n, alpha_I, alpha_P)
        # Associated bootstrap current
        if Bootstrap_choice == 'Freidberg':
            Ib_alpha = f_Freidberg_Ib(R0, a, kappa, pbar_alpha, Ip_alpha)
        elif Bootstrap_choice == 'Segal':
            Ib_alpha = f_Segal_Ib(nu_n, nu_T, a/R0, kappa, nbar_alpha, Tbar, R0, Ip_alpha)
        else:
            print("Choose a valid bootstrap model")
            
        if Operation_mode == 'Steady-State':
            # Current drive efficienty (complex calculation)
            eta_CD_alpha = f_etaCD(a, R0, B0_solution, nbar_alpha, Tbar, nu_n, nu_T)
            # I_Ohm = 0 by definition
            I_Ohm_alpha = 0
            # And associated power
            P_Ohm_alpha = 0
            # Current balance: Ip = Ib + I_CD + I_Ohm
            I_CD_alpha = f_ICD(Ip_alpha, Ib_alpha, I_Ohm_alpha)
            # Power required to drive I_CD
            P_CD_alpha = f_PCD(R0, nbar_alpha, I_CD_alpha, eta_CD_alpha)
            # Q calculation
            Q_alpha = f_Q(P_fus, P_CD_alpha, P_Ohm_alpha)
            
        elif Operation_mode == 'Pulsed':
        
            # Current drive efficienty (complex calculation)
            eta_CD_alpha = f_etaCD(a, R0, B0_solution, nbar_alpha, Tbar, nu_n, nu_T)
            # Pulsed mode: P_CD is a fixed INPUT from user
            P_CD_alpha = P_aux_input
            # Calculate I_CD from power and efficiency
            I_CD_alpha = f_I_CD(R0, nbar_alpha, eta_CD_alpha, P_CD_alpha)
            # Ohmic current from current balance: Ip = Ib + I_CD + I_Ohm
            I_Ohm_alpha = f_I_Ohm(Ip_alpha, Ib_alpha, I_CD_alpha)
            # Ohmic power calculation
            P_Ohm_alpha = f_P_Ohm(I_Ohm_alpha, Tbar, R0, a, kappa)
            # Q factor including both CD and Ohmic power
            Q_alpha = f_Q(P_fus, P_CD_alpha, P_Ohm_alpha)
           
        else:
            print("Choose a valid operation mode")
        # Helium fraction
        new_f_alpha = f_He_fraction(nbar_alpha, Tbar, tau_E_alpha, C_Alpha, nu_T)
        
        # To avoid division by 0
        epsilon = 1e-10
        
        # Residuals
        f_alpha_residual = abs(new_f_alpha - f_alpha) / (abs(new_f_alpha) + epsilon) * 100
        Q_residual = abs(Q - Q_alpha) / (abs(Q_alpha) + epsilon) * 100

        return [f_alpha_residual, Q_residual]
    
    def solve_f_alpha_Q():
        """
        Solve for f_alpha and Q with progressive robustness strategy:
        1. Fast search with 'hybr'
        2. Robust search with 'lm'
        3. Very robust search with 'df-sane'
        4. Grid search with 'lm' as last resort
        """
        
        def verify_solution(f_alpha, Q, residuals, tolerance=1e-2):
            """
            Verify if the solution is physically valid and numerically converged.
            
            Args:
                f_alpha, Q: Solution candidates
                residuals: Residuals from to_solve_f_alpha_and_Q
                tolerance: Maximum acceptable residual (default 0.1%)
            
            Returns:
                bool: True if solution is valid
            """
            
            # Check physical bounds
            if not (0 <= f_alpha <= 1 and Q >= 0):
                return False
            
            # Check numerical convergence
            if abs(residuals[0]) > tolerance or abs(residuals[1]) > tolerance:
                return False
                
            return True
        
        def is_duplicate_solution(sol, solutions_list, tol_f_alpha=1e-2, tol_Q=1.0):
            """
            Check if solution is a duplicate of existing solutions.
            """
            
            for existing_sol in solutions_list:
                if (abs(existing_sol['f_alpha'] - sol['f_alpha']) < tol_f_alpha and 
                    abs(existing_sol['Q'] - sol['Q']) < tol_Q):
                    return True
            return False
        
        def select_best_solution(valid_solutions):
            """
            Select solution with highest Q value
            
            Args:
                valid_solutions: List of valid solution dictionaries
            
            Returns:
                tuple: (f_alpha, Q) of best solution or None if no solutions
            """
            
            if not valid_solutions:
                return None
            
            # Maximize Q
            best = max(valid_solutions, key=lambda x: x['Q'])
            return (best['f_alpha'], best['Q'])
        
        def try_method_with_guesses(method_name, initial_guesses, tolerance=1):
            """
            Try solving with a specific method and multiple initial guesses.
            
            Args:
                method_name: Name of scipy.optimize.root method
                initial_guesses: List of [f_alpha, Q] initial guesses
                tolerance: Convergence tolerance
            
            Returns:
                List of valid solutions found
            """
            
            valid_solutions = []
            
            for guess in initial_guesses:
                try:
                    result = root(
                        to_solve_f_alpha_and_Q,
                        guess,
                        method=method_name,
                        tol=1e-8
                    )
                    
                    if result.success:
                        f_alpha, Q = result.x
                        
                        # Re-evaluate residuals to verify convergence
                        residuals = to_solve_f_alpha_and_Q([f_alpha, Q])
                        
                        # Verify this is a true convergence
                        if verify_solution(f_alpha, Q, residuals, tolerance=tolerance):
                            sol_dict = {
                                'f_alpha': f_alpha,
                                'Q': Q,
                                'residuals': residuals,
                                'residual_norm': np.sqrt(residuals[0]**2 + residuals[1]**2),
                                'method': method_name,
                                'guess': guess
                            }
                            
                            # Only add if not duplicate
                            if not is_duplicate_solution(sol_dict, valid_solutions):
                                valid_solutions.append(sol_dict)
                        
                except Exception as e:
                    continue
            
            return valid_solutions
        
        # Define initial guesses for all methods
        initial_guesses = [
            [0.05, 50],
            [0.05, 1000],
            [0.05, 1],
            [0.05, 5000],
        ]
        
        # ========================================================================
        # STEP 1: Fast search with 'hybr'
        # ========================================================================
        valid_solutions = try_method_with_guesses('hybr', initial_guesses, tolerance=1)
        
        if valid_solutions:
            best_solution = select_best_solution(valid_solutions)
            if best_solution:
                return best_solution
        
        # ========================================================================
        # STEP 2: Robust search with 'lm'
        # ========================================================================
        valid_solutions = try_method_with_guesses('lm', initial_guesses, tolerance=1)
        
        if valid_solutions:
            best_solution = select_best_solution(valid_solutions)
            if best_solution:
                return best_solution
        
        # ========================================================================
        # STEP 3: Very robust search with 'df-sane'
        # ========================================================================
        valid_solutions = try_method_with_guesses('df-sane', initial_guesses, tolerance=1)
        
        if valid_solutions:
            best_solution = select_best_solution(valid_solutions)
            if best_solution:
                return best_solution
        
        # ========================================================================
        # STEP 4: Grid search with 'df-sane' as last resort
        # ========================================================================
        f_alpha_guesses = [0.001, 0.01, 0.1, 0.3, 0.5]
        Q_guesses = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]
        
        grid_guesses = [[f_a, Q_g] for f_a in f_alpha_guesses for Q_g in Q_guesses]
        
        # Use only 'lm' for grid search
        valid_solutions = try_method_with_guesses('df-sane', grid_guesses, tolerance=1)
        
        if valid_solutions:
            
            best_solution = select_best_solution(valid_solutions)
            if best_solution:
                return best_solution
        
        # ========================================================================
        # No solution found
        # ========================================================================
        # print("No valid solution found after all attempts")
        return np.nan, np.nan

    f_alpha_solution, Q_solution = solve_f_alpha_Q()
    
    # Check if convergence succeeded
    if np.isnan(f_alpha_solution) or np.isnan(Q_solution):
        return (np.nan, np.nan, np.nan,                  # B0, B_CS, B_pol
                np.nan, np.nan,                          # tauE, W_th
                np.nan, np.nan, np.nan,                  # Q, Volume, Surface
                np.nan, np.nan, np.nan, np.nan,          # Ip, Ib, I_CD, I_Ohm
                np.nan, np.nan, np.nan,                  # nbar, nG, pbar
                np.nan, np.nan, np.nan,                  # betaN, betaT, betaP
                np.nan, np.nan,                          # qstar, q95
                np.nan, np.nan, np.nan, np.nan, np.nan,  # P_CD, P_sep, P_Thresh, eta_CD, P_elec
                np.nan, np.nan, np.nan,                  # cost, P_Brem, P_syn
                np.nan, np.nan, np.nan, np.nan, np.nan,  # heat, heat_par, heat_pol, lambda_q, q_target
                np.nan, np.nan,                          # P_wall_H, P_wall_L
                np.nan, np.nan,                          # T_op_limit, C_invest
                np.nan, np.nan,                          # CF, COE
                np.nan,                                  # Gamma_n
                np.nan, np.nan,                          # f_alpha, tau_alpha
                np.nan, np.nan,                          # J_TF, J_CS
                np.nan,                                  # TF_ratio
                np.nan, np.nan, np.nan, np.nan,          # r_minor, r_sep, r_c, r_d
                np.nan, np.nan, np.nan, np.nan)          # kappa, kappa_95, δ, δ_95
    
    # Once convergence is achieved, calculate all other parameters:
    nbar_solution = f_nbar(P_fus, nu_n, nu_T, f_alpha_solution, Tbar, R0, a, kappa)
    pbar_solution = f_pbar(nu_n, nu_T, nbar_solution, Tbar, f_alpha_solution)
    W_th_solution = f_W_th(nbar_solution, Tbar, Volume_solution)
    # Radiative losses
    P_Brem_solution = f_P_bremsstrahlung(Volume_solution, nbar_solution, Tbar, Zeff, R0, a)
    P_syn_solution = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_solution, kappa, nu_n, nu_T, r_synch)
    P_rad_solution = P_Brem_solution + P_syn_solution
    
    # Preliminary solving for confinement time calculation
    if Operation_mode == 'Steady-State':
        P_Aux_solution = P_fus / Q_solution
        P_Ohm_solution = 0
    elif Operation_mode == 'Pulsed':
        P_Aux_solution = P_aux_input
        P_Ohm_solution = P_fus / Q_solution - P_Aux_solution
    else:
        print("Choose a valid operation mode")
    # Confinement time
    tauE_solution = f_tauE(pbar_solution, Volume_solution, P_Alpha, P_Aux_solution, P_Ohm_solution, P_rad_solution)
    # alpha particles confinement time
    tau_alpha = f_tau_alpha(nbar_solution, Tbar, tauE_solution, C_Alpha, nu_T)
    # Plasma current needed
    Ip_solution = f_Ip(tauE_solution, R0, a, kappa, δ, nbar_solution, B0_solution, Atomic_mass, 
                       P_Alpha, P_Ohm_solution, P_Aux_solution, P_rad_solution, H, C_SL,
                       alpha_delta, alpha_M, alpha_kappa, alpha_epsilon, alpha_R, alpha_B, alpha_n, alpha_I, alpha_P)
    # Calculate the bootstrap current
    if Bootstrap_choice == 'Freidberg':
        Ib_solution = f_Freidberg_Ib(R0, a, kappa, pbar_solution, Ip_solution)
    elif Bootstrap_choice == 'Segal':
        Ib_solution = f_Segal_Ib(nu_n, nu_T, a/R0, kappa, nbar_solution, Tbar, R0, Ip_solution)
    else:
        print("Choose a valid bootstrap model")
    # Calculate derived quantities
    qstar_solution = f_qstar(a, B0_solution, R0, Ip_solution, kappa)
    q95_solution = f_q95(B0_solution, Ip_solution, R0, a, kappa, δ)
    B_pol_solution = f_Bpol(q95_solution, B0_solution, a, R0)
    betaT_solution = f_beta_T(pbar_solution, B0_solution)
    betaP_solution = f_beta_P(a, kappa, pbar_solution, Ip_solution)
    beta_solution = f_beta(betaP_solution, betaT_solution)
    betaN_solution = f_beta_N(betaT_solution, B0_solution, a, Ip_solution)
    nG_solution = f_nG(Ip_solution, a)
    
    # Recalculate currents and powers using same logic as convergence loop
    if Operation_mode == 'Steady-State':
        # Current drive efficiency
        eta_CD_solution = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
        # I_Ohm = 0 by definition
        I_Ohm_solution = 0
        # And associated power
        P_Ohm_solution = 0
        # Current balance: Ip = Ib + I_CD + I_Ohm
        I_CD_solution = f_ICD(Ip_solution, Ib_solution, I_Ohm_solution)
        # Power required to drive I_CD
        P_CD_solution = f_PCD(R0, nbar_solution, I_CD_solution, eta_CD_solution)
        
    elif Operation_mode == 'Pulsed':
        # Current drive efficiency
        eta_CD_solution = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
        # Pulsed mode: P_CD is a fixed INPUT from user
        P_CD_solution = P_aux_input
        # Calculate I_CD from power and efficiency
        I_CD_solution = f_I_CD(R0, nbar_solution, eta_CD_solution, P_CD_solution)
        # Ohmic current from current balance: Ip = Ib + I_CD + I_Ohm
        I_Ohm_solution = f_I_Ohm(Ip_solution, Ib_solution, I_CD_solution)
        # Ohmic power calculation
        P_Ohm_solution = f_P_Ohm(I_Ohm_solution, Tbar, R0, a, kappa)
        
    else:
        print("Choose a valid operation mode")
        
    P_sep_solution = f_P_sep(P_fus, P_CD_solution)
    Gamma_n_solution = f_Gamma_n(a, P_fus, R0, kappa)
    heat_D0FUS_solution = f_heat_D0FUS(R0, P_sep_solution)
    heat_par_solution = f_heat_par(R0, B0_solution, P_sep_solution)
    heat_pol_solution = f_heat_pol(R0, B0_solution, P_sep_solution, a, q95_solution)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(P_sep_solution, B_pol_solution, R0, a/R0, theta_deg)
    P_1rst_wall_Hmod = f_P_1rst_wall_Hmod(P_sep_solution, P_CD_solution, Surface_solution)
    P_1rst_wall_Lmod = f_P_1rst_wall_Lmod(P_sep_solution, Surface_solution)
    P_elec_solution = f_P_elec(P_fus, P_CD_solution, eta_T, eta_RF)
    li_solution = f_li(nu_n, nu_T)
    
    # Calculate the L-H threshold power
    if L_H_Scaling_choice == 'Martin':
        P_Thresh = P_Thresh_Martin(nbar_solution, B0_solution, a, R0, kappa, Atomic_mass)
    elif L_H_Scaling_choice == 'New_S':
        P_Thresh = P_Thresh_New_S(nbar_solution, B0_solution, a, R0, kappa, Atomic_mass)
    elif L_H_Scaling_choice == 'New_Ip':
        P_Thresh = P_Thresh_New_Ip(nbar_solution, B0_solution, a, R0, kappa, Ip_solution, Atomic_mass)
    else:
        print('Choose a valid Scaling for L-H transition')
    
    # Calculate the radial build
    if Radial_build_model == "academic":
        (c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_academic(a, b, R0, σ_TF, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg)
        (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(Ip_solution, I_Ohm_solution, Bmax, a, b, c, R0, kappa, nbar_solution, Tbar, Ce, Temps_Plateau_input, li_solution, Choice_Buck_Wedg)
        (d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS) = f_CS_ACAD(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, Bmax, σ_CS,
                           Supra_choice, J_max_CS_conducteur, T_helium,f_Cu_Non_Cu , f_Cu_Strand , f_Cool , f_In, Choice_Buck_Wedg)
    elif Radial_build_model == "D0FUS":
        (c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_D0FUS(a, b, R0, σ_TF, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg, omega_TF, n_TF)
        (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(Ip_solution, I_Ohm_solution, Bmax, a, b, c, R0, kappa, nbar_solution, Tbar, Ce, Temps_Plateau_input, li_solution, Choice_Buck_Wedg)
        (d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS) = f_CS_D0FUS(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, Bmax, σ_CS,
                           Supra_choice, J_max_CS_conducteur, T_helium, f_Cu_Non_Cu , f_Cu_Strand , f_Cool , f_In, Choice_Buck_Wedg)
    elif Radial_build_model == "CIRCEE":
        (c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF) = f_TF_D0FUS(a, b, R0, σ_TF, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg, omega_TF, n_TF)
        (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(Ip_solution, I_Ohm_solution, Bmax, a, b, c, R0, kappa, nbar_solution, Tbar, Ce, Temps_Plateau_input, li_solution, Choice_Buck_Wedg)
        (d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS) = f_CS_CIRCEE(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, Bmax, σ_CS,
                           Supra_choice, J_max_CS_conducteur, T_helium, f_Cu_Non_Cu , f_Cu_Strand , f_Cool , f_In, Choice_Buck_Wedg)
    else:
        print('Choose a valid mechanical model')
        
    # Estimate volume of blanket, TF coil and CS
    (V_BB, V_TF, V_CS, V_FI) = f_volume(a, b, c, d, R0, kappa)
    
    # Calculate a proxy for the machine cost
    cost_solution = f_size_per_power(V_BB, V_TF, V_CS, P_fus)
    
    #%% Cost-related computations (under development by Matteo Fletcher)
    
    if Operation_mode == 'Steady-State':
        Dwell_factor = 1
    elif Operation_mode == 'Pulsed':
        Dwell_factor =  p.cost.Dwell_factor
    else:
        print("Choose a valid operation mode")
        
    if Supra_choice == 'Rebco':
        Supra_cost_factor = 2
    else:
        Supra_cost_factor = 1
        
    P_th_solution = P_fus * E_F / (E_ALPHA + E_N)
    
    if cost_model == 'Whyte':
        (T_op_limit, CF, C_invest, COE) = f_costs_Whyte(p.cost.F_dpa, p.cost.L_dpa,
                            Surface_solution, p.cost.Util_factor, Dwell_factor,
                            p.cost.dt_rep, p.cost.discount_rate, p.cost.T_life, 
                            Gamma_n_solution, P_elec_solution)
    elif cost_model == 'Sheffield':
        (T_op_limit, CF, C_invest, COE) = f_costs_Sheffield(p.cost.discount_rate,
                            p.cost.contingency, p.cost.T_life, p.cost.T_build,
                            P_th_solution, P_elec_solution, P_CD_solution,
                            Gamma_n_solution, p.cost.Util_factor, Dwell_factor,
                            p.cost.dt_rep, V_FI, V_TF + V_CS, 1/6 * V_BB, 
                            5/6 * V_BB, 0.1 * Surface_solution, Supra_cost_factor)
    else:
        print("Choose a valid cost model")
        
    C_invest = C_invest / 1000 # M€ -> B€
        
    #%% End of cost-related computations

    return (B0_solution, B_CS, B_pol_solution,
            tauE_solution, W_th_solution,
            Q_solution, Volume_solution, Surface_solution,
            Ip_solution, Ib_solution, I_CD_solution, I_Ohm_solution,
            nbar_solution, nG_solution, pbar_solution,
            betaN_solution, betaT_solution, betaP_solution,
            qstar_solution, q95_solution,
            P_CD_solution, P_sep_solution, P_Thresh, eta_CD_solution, P_elec_solution,
            cost_solution, P_Brem_solution, P_syn_solution,
            heat_D0FUS_solution, heat_par_solution, heat_pol_solution, lambda_q_Eich_m, q_target_Eich,
            P_1rst_wall_Hmod, P_1rst_wall_Lmod,
            T_op_limit, C_invest,
            CF, COE,
            Gamma_n_solution,
            f_alpha_solution, tau_alpha,
            J_max_TF_conducteur, J_max_CS_conducteur,
            c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
            d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS,
            R0-a, R0-a-b, R0-a-b-c, R0-a-b-c-d,
            kappa, kappa_95, δ, δ_95)


def save_run_output(params, results, output_dir, input_file_path=None):
    """Save run results to timestamped directory with complete output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"Run_D0FUS_{timestamp}"
    output_path = os.path.join(output_dir,'run', output_name)
    
    # Create directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy original input file if provided
    if input_file_path and os.path.exists(input_file_path):
        input_copy = os.path.join(output_path, "input_parameters.txt")
        shutil.copy2(input_file_path, input_copy)
    else:
        # Generate input file from parameters
        input_copy = os.path.join(output_path, "input_parameters.txt")
        with open(input_copy, "w", encoding='utf-8') as f:
            f.write("# D0FUS Input Parameters\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write cost model first
            f.write(f"cost_model = {params.cost_model}\n")
            f.write("\n")
            
            # Write regular parameters (skip 'cost' and 'cost_model')
            for key, value in vars(params).items():
                if key in ['cost', 'cost_model']:
                    continue
                f.write(f"{key} = {value}\n")
            
            # Write cost parameters section
            f.write("\n# Cost Parameters\n")
            for ck, cv in params.cost.to_dict().items():
                f.write(f"{ck} = {cv}\n")

    
    # Unpack ALL results
    (B0, B_CS, B_pol,
     tauE, W_th,
     Q, Volume, Surface,
     Ip, Ib, I_CD, I_Ohm,
     nbar, nG, pbar,
     betaN, betaT, betaP,
     qstar, q95,
     P_CD, P_sep, P_Thresh, eta_CD, P_elec,
     cost, P_Brem, P_syn,
     heat, heat_par, heat_pol, lambda_q, q_target,
     P_wall_H, P_wall_L,
     T_op_limit, C_invest,
     CF, COE,
     Gamma_n,
     f_alpha, tau_alpha,
     J_TF, J_CS,
     c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
     d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS,
     r_minor, r_sep, r_c, r_d,
     kappa, kappa_95, δ, δ_95) = results
    
    # Save complete output
    output_file = os.path.join(output_path, "output_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        # Write to both file and console
        class DualWriter:
            def __init__(self, *files):
                self._files = files
            def write(self, data):
                for ff in self._files:
                    ff.write(data)
            def flush(self):
                for ff in self._files:
                    ff.flush()

        dual_output = DualWriter(sys.stdout, f)
        
        # Complete results display (matching original format)
        print("=========================================================================", file=dual_output)
        print("=== Calculation Results ===", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[I] R0 (Major Radius)                               : {params.R0:.3f} [m]", file=dual_output)
        print(f"[I] a (Minor Radius)                                : {params.a:.3f} [m]", file=dual_output)
        print(f"[I] b (BB & Nshield thickness)                      : {r_minor-r_sep:.3f} [m]", file=dual_output)
        print(f"[O] c (TF coil thickness)                           : {r_sep-r_c:.3f} [m]", file=dual_output)
        print(f"[O] d (CS thickness)                                : {r_c-r_d:.3f} [m]", file=dual_output)
        print(f"[O] R0-a-b-c-d                                      : {r_d:.3f} [m]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[O] Kappa (Elongation)                              : {kappa:.3f} ", file=dual_output)
        print(f"[O] Kappa_95 (Elongation at 95%)                    : {kappa_95:.3f} ", file=dual_output)
        print(f"[O] Delta (Triangularity)                           : {δ:.3f} ", file=dual_output)
        print(f"[O] Delta_95 (Triangularity at 95%)                 : {δ_95:.3f} ", file=dual_output)
        print(f"[O] Volume (Plasma)                                 : {Volume:.3f} [m^3]", file=dual_output)
        print(f"[O] Surface (1rst Wall)                             : {Surface:.3f} [m²]", file=dual_output)
        print(f"[I] Mechanical configuration                        : {params.Choice_Buck_Wedg} ", file=dual_output)
        print(f"[I] Superconductor technology                       : {params.Supra_choice} ", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[I] Bmax (Maximum Magnetic Field - TF)              : {params.Bmax:.3f} [T]", file=dual_output)
        print(f"[O] B0 (Central Magnetic Field)                     : {B0:.3f} [T]", file=dual_output)
        print(f"[O] BCS (Magnetic Field CS)                         : {B_CS:.3f} [T]", file=dual_output)
        print(f"[O] J_E-TF (Engineering current density TF)         : {J_TF/1e6:.3f} [MA/m²]", file=dual_output)
        print(f"[O] J_E-CS (Engineering current density CS)         : {J_CS/1e6:.3f} [MA/m²]", file=dual_output)
        print(f"[O] Steel ratio TF                                  : {Steel_fraction_TF*100:.3f} [%]", file=dual_output)
        print(f"[O] Steel ratio CS                                  : {Steel_fraction_CS*100:.3f} [%]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[I] P_fus (Fusion Power)                            : {params.P_fus:.3f} [MW]", file=dual_output)
        print(f"[O] P_CD (CD Power)                                 : {P_CD:.3f} [MW]", file=dual_output)
        print(f"[O] P_S (Synchrotron Power)                         : {P_syn:.3f} [MW]", file=dual_output)
        print(f"[O] P_B (Bremsstrahlung Power)                      : {P_Brem:.3f} [MW]", file=dual_output)
        print(f"[O] eta_CD (CD Efficiency)                          : {eta_CD:.3f} [MA/MW-m²]", file=dual_output)
        print(f"[O] Q (Energy Gain Factor)                          : {Q:.3f}", file=dual_output)
        print(f"[O] P_elec-net (Net Electrical Power)               : {P_elec:.3f} [MW]", file=dual_output)
        print(f"[O] Cost ((V_BB+V_TF+V_CS)/P_fus)                   : {cost:.3f} [m^3/MW]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[I] H (Scaling Law factor)                          : {params.H:.3f} ", file=dual_output)
        print(f"[I] Operation (Pulsed / Steady)                     : {params.Operation_mode} ", file=dual_output)
        print(f"[I] t (Plateau Time)                                : {params.Temps_Plateau_input:.3f} ", file=dual_output)
        print(f"[O] tau_E (Confinement Time)                        : {tauE:.3f} [s]", file=dual_output)
        print(f"[O] Ip (Plasma Current)                             : {Ip:.3f} [MA]", file=dual_output)
        print(f"[O] Ib (Bootstrap Current)                          : {Ib:.3f} [MA]", file=dual_output)
        print(f"[O] ICD (Current Drive)                             : {I_CD:.3f} [MA]", file=dual_output)
        print(f"[O] IOhm (Ohmic Current)                            : {I_Ohm:.3f} [MA]", file=dual_output)
        print(f"[O] f_b (Bootstrap Fraction)                        : {(Ib/Ip)*1e2:.3f} [%]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[I] Tbar (Mean Temperature)                         : {params.Tbar:.3f} [keV]", file=dual_output)
        print(f"[O] nbar (Average Density)                          : {nbar:.3f} [10^20 m^-3]", file=dual_output)
        print(f"[O] nG (Greenwald Density)                          : {nG:.3f} [10^20 m^-3]", file=dual_output)
        print(f"[O] pbar (Average Pressure)                         : {pbar:.3f} [MPa]", file=dual_output)
        print(f"[O] Alpha Fraction                                  : {f_alpha*1e2:.3f} [%]", file=dual_output)
        print(f"[O] Alpha Confinement Time                          : {tau_alpha:.3f} [s]", file=dual_output)
        print(f"[O] Thermal Energy Content                          : {W_th/1e6:.3f} [MJ]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[O] Beta_T (Toroidal Beta)                          : {betaT*1e2:.3f} [%]", file=dual_output)
        print(f"[O] Beta_P (Poloidal Beta)                          : {betaP:.3f}", file=dual_output)
        print(f"[O] Beta_N (Normalized Beta)                        : {betaN:.3f} [%]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[O] q* (Kink Safety Factor)                         : {qstar:.3f}", file=dual_output)
        print(f"[O] q95 (Safety Factor at 95%)                      : {q95:.3f}", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[O] P_sep (Separatrix Power)                        : {P_sep:.3f} [MW]", file=dual_output)
        print(f"[O] P_Thresh (L-H Power Threshold)                  : {P_Thresh:.3f} [MW]", file=dual_output)
        print(f"[O] (P_sep - P_thresh) / S                          : {P_wall_H:.3f} [MW/m²]", file=dual_output)
        print(f"[O] P_sep / S                                       : {P_wall_L:.3f} [MW/m²]", file=dual_output)
        print(f"[O] Heat scaling (P_sep / R0)                       : {heat:.3f} [MW/m]", file=dual_output)
        print(f"[O] Parallel Heat Flux (P_sep*B0 / R0)              : {heat_par:.3f} [MW-T/m]", file=dual_output)
        print(f"[O] Poloidal Heat Flux (P_sep*B0) / (q95*R0*A)      : {heat_pol:.3f} [MW-T/m]", file=dual_output)
        print(f"[O] Gamma_n (Neutron Flux)                          : {Gamma_n:.3f} [MW/m²]", file=dual_output)
        print("-------------------------------------------------------------------------", file=dual_output)
        print(f"[I] T_life (plant lifetime)                         : {params.cost.T_life:.3f} [yr]", file=dual_output)
        print(f"[I] Util_factor (utilization factor)                : {params.cost.Util_factor:.3f}", file=dual_output)
        print(f"[I] dt_rep (replacement time)                       : {params.cost.dt_rep:.3f} [yr]", file=dual_output)
        print(f"[O] T_op_limit (time before replacement)            : {T_op_limit:.3f} [yr]", file=dual_output)
        print(f"[O] C_invest (total invest cost)                    : {C_invest:.3f} [B€]", file=dual_output)
        print(f"[O] CF (power plant capacity factor)                : {CF:.3f}", file=dual_output)
        print(f"[O] COE (cost of electricity)                       : {COE:.3f} [€/MWh]", file=dual_output)
        print("=========================================================================", file=dual_output)
    
    print(f"\n✓ Results saved to: {output_path}\n")
    return output_path


def main(input_file=None):
    """Main execution function"""
    # Load parameters
    p = Parameters()
    
    # Store input file path for copying later
    input_file_path = input_file
    
    if input_file is None:
        default_input = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_INPUTS', 'default_input.txt')
        if os.path.exists(default_input):
            input_file = default_input
    
    if input_file and os.path.exists(input_file):
        print(f"\nLoading parameters from: {input_file}")
        p.open_input(input_file)
    else:
        print(f"\nWarning: Input file not found. Using default parameters.")
        input_file_path = None
    
    # Run calculation
    print("\n" + "="*73)
    print("Starting D0FUS calculation...")
    print("="*73 + "\n")
    
    try:
        results = run(p)
        
        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        save_run_output(p, results, output_dir, input_file_path)
        
        return results
    
    except Exception as e:
        print(f"\n!!! ERROR during calculation !!!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = None
    
    main(input_file)
    
    print("\nD0FUS_run completed successfully!")