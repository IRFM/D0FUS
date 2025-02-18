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
    
    # κ = f_Kappa(R0/a,Option_Kappa)
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
    beta_solution = f_beta(pbar_solution, B0_solution, a, Ip_solution)
    qstar_solution = f_qstar(a, B0_solution, R0, Ip_solution, κ)
    q95_solution = f_q95(B0_solution,Ip_solution,R0,a,κ,δ)
    nG_solution = f_nG(Ip_solution, a)
    eta_CD_solution = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
    P_CD = f_PCD(R0,nbar_solution,Ip_solution,Ib_solution,eta_CD_solution)
    Gamma_n = f_Gamma_n(a, P_fus, R0, κ)
    P_sep = P_CD+(P_fus*E_ALPHA/(E_ALPHA+E_N))
    heat = f_heat(B0_solution,R0,P_fus)
    
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
    
    if Radial_build_model == "academic" :
        
        (c, Winding_pack_tension_ratio) = f_TF_academic(a, b, R0, B0_solution, σ_TF, μ0, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg)
        (d,Alpha,B_CS) = f_CS_academic_convergence(a, b, c, R0, B0_solution, σ_CS, μ0, J_max_CS_conducteur, Choice_Buck_Wedg, Tbar, nbar_solution, Ip_solution, Ib_solution)
            
    elif Radial_build_model == "simple" :
        
        (c, Winding_pack_tension_ratio) = f_TF_simple(a, b, R0, B0_solution, σ_TF, μ0, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg)
        (d,Alpha,B_CS) = f_CS_simple_convergence(a, b, c, R0, B0_solution, σ_CS, μ0, J_max_CS_conducteur, Choice_Buck_Wedg, Tbar, nbar_solution, Ip_solution, Ib_solution)
    
    #elif Radial_build_model == "realistic" :
    else :
        print('Choose a valid mechanical model')

    cost = f_cost(a,b,c,d,R0,κ,Q_solution)

    return (B0_solution, B_CS, tauE_solution, Q_solution, Ip_solution, nbar_solution,
            beta_solution, qstar_solution, q95_solution, nG_solution, 
            P_CD, P_sep, P_Thresh, cost, heat, Gamma_n, f_alpha_solution, Winding_pack_tension_ratio,
            R0-a, R0-a-b, R0-a-b-c, R0-a-b-c-d)

if __name__ == "__main__":
    # Appeler la fonction de calcul
    # Benchmark
    R0 = 6
    a = 1.5
    Pfus = 2000
    Bmax = 20
    b = 1.2
    Tbar = 14
    # End Benchmark parameters
    B0_solution, B_CS, tauE_solution, Q_solution, Ip_solution, nbar_solution, \
    beta_solution, qstar_solution, q95_solution, nG_solution, \
    P_CD, P_sep, P_Thresh, cost, heat, Gamma_n, f_alpha_solution, TF_ratio, \
    r_minor, r_sep, r_c, r_d = calcul(a, R0, Bmax, Pfus, Tbar)

    # Affichage propre des résultats
    print("============================================================")
    print("=== Résultats du Calcul ===")
    print("-------------------------------------------------------------")
    print(f"Bmax (Champ magnétique TF)            : {Bmax:.3f} T")
    print(f"B0 (Champ magnétique central)         : {B0_solution:.3f} T")
    print(f"BCS (Champ magnétique CS)             : {B_CS:.3f} T")
    print("-------------------------------------------------------------")
    print(f"tau_E (Temps de confinement)          : {tauE_solution:.3f} s")
    print(f"Q (Facteur de gain énergétique)       : {Q_solution:.3f}")
    print(f"Ip (Courant plasma)                   : {Ip_solution:.3e} MA")
    print("-------------------------------------------------------------")
    print(f"nbar (Densité moyenne)                : {nbar_solution:.3f} 10^20 m^-3")
    print(f"nG (Densité de Greenwald)             : {nG_solution:.3f} 10^20 m^-3")
    print(f"Beta (Pression plasma/Pression mag)   : {beta_solution:.3f}")
    print(f"q* (Facteur de sécurité)              : {qstar_solution:.3f}")
    print(f"q95 (Facteur de sécurité à 95%)       : {q95_solution:.3f}")
    print(f"Alpha fraction                        : {f_alpha_solution:.3f}")
    print("-------------------------------------------------------------")
    print(f"P_fus (Puissance Fusion)              : {Pfus:.3f} MW")
    print(f"P_CD (Puissance CD)                   : {P_CD:.3f} MW")
    print(f"P_sep (Puissance separatrice)         : {P_sep:.3f} MW")
    print(f"P_Thresh (Seuil de puissance L-H)     : {P_Thresh:.3f} MW")
    print("-------------------------------------------------------------")
    print(f"Coût                                  : {cost:.3f}")
    print(f"Flux de chaleur                       : {heat:.3f} MW/m")
    print(f"Gamma_n (Flux neutronique)            : {Gamma_n:.3f} MW/m²")
    print("-------------------------------------------------------------")
    print(f"R0                                    : {R0:.3f} m")
    print(f"a                                     : {a:.3f} m")
    print(f"b                                     : {r_minor-r_sep:.3f} m")
    print(f"c                                     : {r_sep-r_c:.3f} m")
    print(f"d                                     : {r_c-r_d:.3f} m")
    print(f"R0-a-b-c-d                            : {r_d:.3f} m")
    print(f"Kappa (Elongation)                    : {κ:.3f} ")
    print(f"Delta (Triangularity)                 : {δ:.3f} ")
    print("============================================================")
    
    
#%%

print("D0FUS_run loaded")