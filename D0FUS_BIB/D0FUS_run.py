# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:33:47 2025

@author: TA276941
"""

#%% Import

from D0FUS_radial_build_functions import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% Main Function

def calcul(a, R0, Bmax, P_fus, Tbar):
    
    # Elongation
    κ = f_Kappa(R0/a,Option_Kappa)
    
    # Central magnetic field
    B0_solution = f_B0(Bmax,a,b,R0)
    
    # Function to solve for both f_alpha and Q
    def to_solve_f_alpha_and_Q(vars):
        
        f_alpha, Q = vars
    
        # Calculate intermediate values
        nbar_alpha = f_nbar(P_fus, R0, a, κ, nu_n, nu_T, f_alpha, Tbar)
        pbar_alpha = f_pbar(nu_n, nu_T, nbar_alpha, Tbar, f_alpha)
        tau_E_alpha = f_tauE(pbar_alpha, R0, a, κ, P_fus, Q)
        Ip_alpha = f_Ip(tau_E_alpha, R0, a, κ, nbar_alpha, B0_solution, Atomic_mass, P_fus, Q)
        Ib_alpha = f_Ib(R0, a, κ, pbar_alpha, Ip_alpha)
        eta_CD_alpha = f_etaCD(a, R0, B0_solution, nbar_alpha, Tbar, nu_n, nu_T)
        P_CD_alpha = f_PCD(R0, nbar_alpha, Ip_alpha, Ib_alpha, eta_CD_alpha)
    
        # Calculate new f_alpha
        new_f_alpha = f_He_fraction(nbar_alpha, Tbar, tau_E_alpha, C_Alpha)
    
        # Calculate the residuals
        f_alpha_residual = new_f_alpha - f_alpha
        Q_residual = (Q - P_fus / P_CD_alpha) * 100 / (P_fus / P_CD_alpha)
    
        return [f_alpha_residual, Q_residual]
    
    # Initial guesses for f_alpha and Q
    initial_guess = [0.1, 1000]
    
    # Solve the system of equations
    try:
        solution, info, ier, msg = fsolve(to_solve_f_alpha_and_Q, initial_guess, xtol=1e-3, full_output=True)
        if ier != 1:
            f_alpha_solution = np.nan
            Q_solution = np.nan
        else:
            f_alpha_solution, Q_solution = solution
            if f_alpha_solution > 1 or f_alpha_solution < 0:
                f_alpha_solution = np.nan
            if Q_solution < 0:
                Q_solution = np.nan
    except ValueError as e:
        f_alpha_solution, Q_solution = np.nan, np.nan

    # Une fois Q déterminé, calcul des autres paramètres
    nbar_solution = f_nbar(P_fus,R0,a,κ,nu_n,nu_T,f_alpha_solution,Tbar)
    pbar_solution = f_pbar(nu_n,nu_T,nbar_solution,Tbar,f_alpha_solution)
    eta_CD = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
    tauE_solution = f_tauE(pbar_solution,R0,a,κ,P_fus,Q_solution)
    Ip_solution = f_Ip(tauE_solution,R0,a,κ,nbar_solution,B0_solution,Atomic_mass,P_fus,Q_solution)
    Ib_solution = f_Ib(R0, a, κ, pbar_solution, Ip_solution)
    qstar_solution = f_qstar(a, B0_solution, R0, Ip_solution, κ)
    q95_solution = f_q95(B0_solution,Ip_solution,R0,a,κ,δ)
    B_pol_solution = f_Bpol(q95_solution, B0_solution, a, R0)
    betaT_solution = f_beta_T(pbar_solution,B0_solution)
    betaP_solution = f_beta_P(a, κ, pbar_solution, Ip_solution)
    beta_solution = f_beta(betaP_solution, betaT_solution)
    betaN_solution = f_beta_N(betaT_solution, B0_solution, a, Ip_solution)
    nG_solution = f_nG(Ip_solution, a)
    eta_CD_solution = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
    P_CD_solution = f_PCD(R0,nbar_solution,Ip_solution,Ib_solution,eta_CD_solution)
    P_sep_solution = f_P_sep(P_fus, P_CD_solution)
    Gamma_n_solution = f_Gamma_n(a, P_fus, R0, κ)
    heat_D0FUS_solution = f_heat_D0FUS(R0,P_sep_solution)
    heat_par_solution = f_heat_par(R0,B0_solution,P_sep_solution)
    heat_pol_solution = f_heat_pol(R0,B0_solution,P_sep_solution,a,q95_solution)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(P_sep_solution, B_pol_solution, R0, a/R0, theta_deg)
    surface_1rst_wall_solution = f_surface_premiere_paroi(κ, R0, a)
    P_1rst_wall_Hmod = (P_sep_solution-P_CD_solution)/surface_1rst_wall_solution
    P_1rst_wall_Lmod = P_sep_solution/surface_1rst_wall_solution
    P_elec_solution = f_P_elec(P_fus, P_CD_solution, eta_T, eta_RF)

    # L-H scaling :
    if L_H_Scaling_choice == 'Martin':
        P_Thresh = P_Thresh_Martin(nbar_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_S':
        P_Thresh = P_Thresh_New_S(nbar_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_Ip':
        P_Thresh = P_Thresh_New_Ip(nbar_solution, B0_solution, a, R0, κ, Ip_solution, Atomic_mass)
    else :
        print('Choose a valid Scaling for L-H transition')
    
    # Radial Build
    #J_e scaling
    if Supra_choice == "LTS"  :
        J_max_TF_conducteur = Jc_LTS(Bmax, T_helium + Marge_T_LTS + Marge_T_Helium)
        J_max_CS_conducteur = Jc_LTS(Bmax, T_helium + Marge_T_LTS + Marge_T_Helium)
    elif Supra_choice == "HTS"  :
        J_max_TF_conducteur = Jc_HTS(Bmax, T_helium + Marge_T_HTS + Marge_T_Helium)
        J_max_CS_conducteur = Jc_HTS(Bmax, T_helium + Marge_T_HTS + Marge_T_Helium)
    elif Supra_choice == "Manual"  :
        J_max_TF_conducteur = J_max_TF_conducteur_manual
        J_max_CS_conducteur = J_max_CS_conducteur_manual
    else :
        print('Choose a valid supra model')
    
    # Tf and Cs width
    if Radial_build_model == "academic" :
        (c, Winding_pack_tension_ratio) = f_TF_academic(a, b, R0, σ_TF, μ0, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg)
        (d,Alpha,B_CS) = f_CS_academic(a, b, c, R0, κ, Bmax, σ_CS, μ0, J_max_CS_conducteur, Choice_Buck_Wedg, Tbar, nbar_solution, Ip_solution, Ib_solution)
    elif Radial_build_model == "D0FUS" :
        (c, Winding_pack_tension_ratio) = f_TF_D0FUS(a, b, R0, σ_TF, μ0, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg, gamma_TF, beta_TF)
        (d,Alpha,B_CS, J_CS) = f_CS_D0FUS(a, b, c, R0, κ, Bmax, σ_CS, μ0, J_max_CS_conducteur, Choice_Buck_Wedg, Tbar, nbar_solution, Ip_solution, Ib_solution, gamma_CS)
    else :
        print('Choose a valid mechanical model')
    
    cost_solution = f_cost(a,b,c,d,R0,κ,Q_solution)

    return (B0_solution, B_CS, B_pol_solution,
            tauE_solution,
            Q_solution,
            Ip_solution, Ib_solution,
            nbar_solution, nG_solution,
            betaN_solution, betaT_solution, betaP_solution,
            qstar_solution, q95_solution,
            P_CD_solution, P_sep_solution, P_Thresh, eta_CD, P_elec_solution,
            cost_solution,
            heat_D0FUS_solution, heat_par_solution, heat_pol_solution, lambda_q_Eich_m, q_target_Eich,
            P_1rst_wall_Hmod, P_1rst_wall_Lmod,
            Gamma_n_solution,
            f_alpha_solution,
            J_max_TF_conducteur, J_max_CS_conducteur,
            Winding_pack_tension_ratio, R0-a, R0-a-b, R0-a-b-c, R0-a-b-c-d, κ)

#%% Benchmark

if __name__ == "__main__":

    # Sf Plant
    R0 = 6
    a = 1.6
    Pfus = 2000
    Bmax = 20
    b = 1.2
    Tbar = 14
    
    # Aries 1
    # R0 = 6.75
    # a = 1.5
    # Pfus = 2000
    # Bmax = 19
    # b = 1.2
    # Tbar = 20
    
    # End Benchmark parameters
    (B0_solution, B_CS, B_pol_solution,
    tauE_solution,
    Q_solution,
    Ip_solution, Ib_solution, 
    nbar_solution, nG_solution,
    betaN_solution, betaT_solution, betaP_solution,
    qstar_solution, q95_solution,
    P_CD, P_sep, P_Thresh, eta_CD, P_elec_solution,
    cost,
    heat_D0FUS_solution, heat_par_solution, heat_pol_solution, lambda_q_Eich_m, q_target_Eich,
    P_1rst_wall_Hmod, P_1rst_wall_Lmod,
    Gamma_n,
    f_alpha_solution,
    J_max_TF_conducteur, J_max_CS_conducteur,
    TF_ratio, r_minor, r_sep, r_c, r_d , κ ) = calcul(a, R0, Bmax, Pfus, Tbar)

    # Clean display of results
    print("=========================================================================")
    print("=== Calculation Results ===")
    print("-------------------------------------------------------------------------")
    print(f"[I] R0 (Major Radius)                               : {R0:.3f} [m]")
    print(f"[I] a (Minor Radius)                                : {a:.3f} [m]")
    print(f"[I] b (BB & Nshield thickness)                      : {r_minor-r_sep:.3f} [m]")
    print(f"[O] c (TF coil thickness)                           : {r_sep-r_c:.3f} [m]")
    print(f"[O] d (CS thickness)                                : {r_c-r_d:.3f} [m]")
    print(f"[O] R0-a-b-c-d                                      : {r_d:.3f} [m]")
    print(f"[O] Kappa (Elongation)                              : {κ:.3f} ")
    print(f"[I] Delta (Triangularity)                           : {δ:.3f} ")
    print(f"[I] Mechanical configuration                        : {Choice_Buck_Wedg} ")
    print(f"[I] Superconductor technology                       : {Supra_choice} ")
    print("-------------------------------------------------------------------------")
    print(f"[I] Bmax (Maximum Magnetic Field - TF)              : {Bmax:.3f} [T]")
    print(f"[O] B0 (Central Magnetic Field)                     : {B0_solution:.3f} [T]")
    print(f"[O] BCS (Magnetic Field CS)                         : {B_CS:.3f} [T]")
    print("-------------------------------------------------------------------------")
    print(f"[I] H (Scaling Law factor)                          : {H:.3f} ")
    print(f"[O] tau_E (Confinement Time)                        : {tauE_solution:.3f} [s]")
    print(f"[O] Ip (Plasma Current)                             : {Ip_solution:.3f} [MA]")
    print(f"[O] Ib (Bootstrap Current)                          : {Ib_solution:.3f} [MA]")
    print(f"[O] f_b (Bootstrap Fraction)                        : {(Ib_solution/Ip_solution)*1e2:.3f} [%]")
    print("-------------------------------------------------------------------------")
    print(f"[I] Tbar (Mean Temperature)                         : {Tbar:.3f} [keV]")
    print(f"[O] nbar (Average Density)                          : {nbar_solution:.3f} [10^20 m^-3]")
    print(f"[O] nG (Greenwald Density)                          : {nG_solution:.3f} [10^20 m^-3]")
    print(f"[O] Alpha Fraction                                  : {f_alpha_solution*1e2:.3f} [%]")
    print("-------------------------------------------------------------------------")
    print(f"[O] Beta_T (Toroidal Beta)                          : {betaT_solution*1e2:.3f} [%]")
    print(f"[O] Beta_P (Poloidal Beta)                          : {betaP_solution:.3f}")
    print(f"[O] Beta_N (Normalized Beta)                        : {betaN_solution:.3f} [%]")
    print("-------------------------------------------------------------------------")
    print(f"[O] q* (Kink Safety Factor)                         : {qstar_solution:.3f}")
    print(f"[O] q95 (Safety Factor at 95%)                      : {q95_solution:.3f}")
    print("-------------------------------------------------------------------------")
    print(f"[I] P_fus (Fusion Power)                            : {Pfus:.3f} [MW]")
    print(f"[O] P_CD (CD Power)                                 : {P_CD:.3f} [MW]")
    print(f"[O] eta_CD (CD Efficiency)                          : {eta_CD:.3f} [MA/MW-m²]")
    print(f"[O] Q (Energy Gain Factor)                          : {Q_solution:.3f}")
    print(f"[O] P_elec (Electrical Power)                       : {P_elec_solution:.3f} [MW]")
    print(f"[O] Cost ((V_BB+V_TF+V_CS)/Q)                       : {cost:.3f} [m^3]")
    print("-------------------------------------------------------------------------")
    print(f"[O] P_sep (Separatrix Power)                        : {P_sep:.3f} [MW]")
    print(f"[O] P_Thresh (L-H Power Threshold)                  : {P_Thresh:.3f} [MW]")
    print(f"[O] (P_sep - P_thresh) / S                          : {P_1rst_wall_Hmod:.3f} [MW/m²]")
    print(f"[O] P_sep / S                                       : {P_1rst_wall_Lmod:.3f} [MW/m²]")
    print(f"[O] Heat scaling (P_sep / R0)                       : {heat_D0FUS_solution:.3f} [MW/m]")
    print(f"[O] Parallel Heat Flux (P_sep*B0 / R0)              : {heat_par_solution:.3f} [MW-T/m]")
    print(f"[O] Poloidal Heat Flux (P_sep*B0) / (q95*R0*A*R0)   : {heat_pol_solution:.3f} [MW-T/m]")
    print(f"[O] Gamma_n (Neutron Flux)                          : {Gamma_n:.3f} [MW/m²]")
    print("=========================================================================")
    
    
#%%

print("D0FUS_run loaded")