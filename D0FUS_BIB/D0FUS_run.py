# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:50:41 2025

@author: TA276941
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:33:47 2025

@author: TA276941
"""

#%% Import

from D0FUS_radial_build_functions import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))


#%% Parameters
class Parameters:
    
    def __init__(self):
        
        self.P_fus = 2000                        # Fusion power [MW]
        self.R0 = 9                              # Major radius [m]
        self.a = 3                               # Minor radius [m]        
        self.Option_Kappa = 'Wenninger'          # Plasma elongation model: 'Stambaugh' , 'Freidberg' , 'Wenninger' or 'Manual'
        self.κ_manual = 1.7                      # Plasma vertical elongation when using 'Manual' model
        self.Bmax = 12                           # Toroidal field on the inner leg of the TF coils [T]
        self.Supra_choice = 'Nb3Sn'              # 'Nb3Sn' , 'Rebco', 'NbTi' or 'Manual'
        self.Radial_build_model = 'D0FUS'        # "academic" , "D0FUS" , "CIRCEE"
        self.b = 1.2                             # First wall + blanket + neutron shield + gaps thickness [m]
        self.Choice_Buck_Wedg = 'Wedging'        # 'Wedging' or 'Bucking'
        self.Chosen_Steel = '316L'               # '316L' , 'N50H' or 'Manual'
        self.Scaling_Law = 'IPB98(y,2)'          # Confinement time scaling law: 'IPB98(y,2)', 'ITPA20-IL', 'ITPA20', 'DS03', 'L-mode', 'L-mode OK', 'ITER89-P'
        self.H = 1                               # H factor for the energy confinement time
        self.Tbar  = 14                          # Volume-averaged temperature [keV]
        self.nu_T  = 1                           # Temperature profile peaking parameter
        self.nu_n  = 0.1                         # Density profile peaking parameter
        self.L_H_Scaling_choice = 'New_Ip'       # Scaling law for the L-H threshold power: 'Martin' , 'New_S', 'New_Ip'
        self.Bootstrap_choice = 'Freidberg'      # Bootstrap current model: 'Freidberg' or 'Segal'
        self.Operation_mode = 'Steady-State'     # 'Steady-State' or 'Pulsed'
        if self.Operation_mode == 'Pulsed':
            self.Temps_Plateau_input = 120 * 60  # Ip plateau duration [s]
            self.P_aux_input = 120               # Auxiliary power [MW]
            self.fatigue = 2                     # Fatigue parameter for the steel 
        else:
            self.Temps_Plateau_input = 0
            self.P_aux_input = 0
            self.fatigue = 1
        
    def parse_assignments(self, lines):
        result = []
        cleaned_lines = []
        for line in lines:

            line = line.split('#')[0].strip()
            if line: 
                cleaned_lines.append(line)                
        for line in cleaned_lines:
            line = line.strip()
            var, val = line.split("=")
            var = var.strip() # remove space and \n (only at the beggining or end of the str)
            val = val.strip()
            try:
                # try to convert into float
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except ValueError:
                pass  # keep in str if it's not a number
            result.append([var, val])
        return result
        
    def open_input(self, name):
        
        fic = open(name, "r")
        read = fic.read()
        fic.close
        
        read = read.split("\n")
      
        if read[-1] == "":
            read.pop() # remove the last element of the list 
        inputs = self.parse_assignments(read)
        for var, val in inputs:
            setattr(self, var, val)  
            print(f"Input change : self.{var} = {val}")
            

#%% Main Function

def run(a, R0, Bmax, P_fus, Tbar, H, Temps_Plateau_input, b, nu_n, nu_T,
        Supra_choice, Chosen_Steel, Radial_build_model, Choice_Buck_Wedg, 
        Option_Kappa, κ_manual, L_H_Scaling_choice, Scaling_Law, Bootstrap_choice, 
        Operation_mode, fatigue, P_aux_input):
    
    # Stress limits in steel
    σ_TF = Steel(Chosen_Steel)
    σ_CS = Steel(Chosen_Steel) / fatigue
    
    # Current densities in coils
    J_max_TF_conducteur = Jc(Supra_choice, Bmax, T_helium) * f_Cu * f_Cool * f_In
    J_max_CS_conducteur = Jc(Supra_choice, Bmax, T_helium + Marge_CS) * f_Cu * f_Cool * f_In
    
    # Fraction of vertical tension allocated to the winding pack of the TF coils (the rest being allocated to the nose)
    if Choice_Buck_Wedg == "Wedging":
        omega_TF = 1/2        # From ITER DDD TF p.97
    elif Choice_Buck_Wedg == "Bucking":
        omega_TF = 1          
    else: 
        print('Choose a valid mechanical configuration')
        
    # Confinement time scaling law
    (C_SL,alpha_delta,alpha_M,alpha_kappa,alpha_epsilon,
     alpha_R,alpha_B,alpha_n,alpha_I,alpha_P) = f_Get_parameter_scaling_law(Scaling_Law)

    # Plasma geometry
    κ    = f_Kappa(R0/a, Option_Kappa, κ_manual)
    κ_95 = f_Kappa_95(κ)
    δ    = f_Delta(κ)
    δ_95 = f_Delta_95(δ)
    
    # Plasma volume
    Volume_solution = f_plasma_volume(R0, a, κ, δ)
    Surface_solution = f_surface_premiere_paroi(κ, R0, a)
    
    # Central magnetic field
    B0_solution = f_B0(Bmax, a, b, R0)
    
    # Alpha power
    P_Alpha = f_P_alpha(P_fus, E_ALPHA, E_N)
    
    # Function to solve for both the alpha particle fraction f_alpha and Q
    def to_solve_f_alpha_and_Q(vars):
        
        f_alpha, Q = vars
    
        # Calculate intermediate values
        nbar_alpha   = f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, κ) 
        # Alternative with a different calculation of the volume:
        # nbar_alpha   = f_nbar_advanced(P_fus, nu_n, nu_T, f_alpha, Tbar, Volume_solution)
        pbar_alpha   = f_pbar(nu_n, nu_T, nbar_alpha, Tbar, f_alpha)
        
        # Radiative losses
        P_Brem_alpha = f_P_bremsstrahlung(Volume_solution, nbar_alpha, Tbar, Zeff, R0, a)
        beta_T = 2 # Beta_T taken from [J.Johner Helios]
        P_syn_alpha = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_alpha, κ, nu_n, nu_T, beta_T, r_synch)
        P_rad_alpha = P_Brem_alpha + P_syn_alpha
        
        # By taking the previous Q we provide a first approximation of the Ohmic and Auxilary power
        # By the Q convergence, this values will be coherent 
        # Proof : by definition of the convergence Q(n-1) = Q(n)
        if Operation_mode == 'Steady-State' :
            P_Ohm_alpha_init = 0
            P_Aux_alpha_init = P_fus / Q
        elif Operation_mode == 'Pulsed':
            P_Aux_alpha_init = P_aux_input
            P_Ohm_alpha_init = P_fus / Q - P_Aux_alpha_init
        else:
            print("Choose a valid operation mode ")
            
        tau_E_alpha  = f_tauE(pbar_alpha, Volume_solution, P_Alpha, P_Aux_alpha_init, P_Ohm_alpha_init, P_rad_alpha)
        Ip_alpha     = f_Ip(tau_E_alpha, R0, a, κ, δ, nbar_alpha, B0_solution, Atomic_mass,
                            P_Alpha, P_Ohm_alpha_init, P_Aux_alpha_init, P_rad_alpha,
                            H, C_SL,
                            alpha_delta,alpha_M,alpha_kappa,alpha_epsilon, alpha_R,alpha_B,alpha_n,alpha_I,alpha_P)
        if Bootstrap_choice == 'Freidberg' :
            Ib_alpha = f_Freidberg_Ib(R0, a, κ, pbar_alpha, Ip_alpha)
        elif Bootstrap_choice == 'Segal' :
            Ib_alpha = f_Segal_Ib(nu_n, nu_T, a/R0, κ, nbar_alpha, Tbar, R0, Ip_alpha)
        else :
            print("Choose a valid Bootstrap model")
            
        eta_CD_alpha = f_etaCD(a, R0, B0_solution, nbar_alpha, Tbar, nu_n, nu_T)
        
        # Real Current drive and Ohmic balance needed allowing the convergence on Q
        if Operation_mode == 'Steady-State' :
            P_CD_alpha    = f_PCD(R0, nbar_alpha, Ip_alpha, Ib_alpha, eta_CD_alpha)
            I_CD_alpha    = f_I_CD(R0, nbar_alpha, eta_CD_alpha, P_CD_alpha)
            P_Ohm_alpha   = 0
            I_Ohm_alpha   = 0
            Q_alpha       = f_Q(P_fus,P_CD_alpha,P_Ohm_alpha)
        elif Operation_mode == 'Pulsed':
            P_CD_alpha    = P_aux_input
            I_CD_alpha    = f_I_CD(R0, nbar_alpha, eta_CD_alpha, P_CD_alpha)
            I_Ohm_alpha   = f_I_Ohm(Ip_alpha, Ib_alpha, I_CD_alpha)
            P_Ohm_alpha   = f_P_Ohm(I_Ohm_alpha, Tbar, R0, a, κ)
            Q_alpha       = f_Q(P_fus,P_CD_alpha,P_Ohm_alpha)
        else:
            print("Choose a valid operation mode ")
        
        # Calculate new f_alpha
        new_f_alpha  = f_He_fraction(nbar_alpha, Tbar, tau_E_alpha, C_Alpha, nu_T)
    
        # Calculate the residuals
        f_alpha_residual = (new_f_alpha - f_alpha) * 100 / new_f_alpha    # In % to ease the convergence
        Q_residual       = (Q - Q_alpha) * 100 / Q_alpha                  # In % to ease the convergence

        return [f_alpha_residual, Q_residual]
        
    def solve_f_alpha_Q():
        # Version principale avec grid search
        f_alpha_guesses = np.concatenate([
            np.linspace(0.001, 0.01, 5),
            np.linspace(0.01, 0.1, 5),
            np.linspace(0.1, 1.0, 5)
        ])
        Q_guesses = np.logspace(1, 5, 8)
        
        initial_guesses = [[f_a, Q] for f_a in f_alpha_guesses for Q in Q_guesses]
        
        for method in ['lm', 'hybr', 'df-sane']:  # Plusieurs méthodes
            for guess in initial_guesses:
                try:
                    result = root(to_solve_f_alpha_and_Q, guess, method=method, tol=1e-8)
                    if result.success:
                        f_alpha, Q = result.x
                        if 0 <= f_alpha <= 1 and Q >= 0:
                            return (f_alpha, Q)
                except Exception:
                    continue
                    
        return np.nan, np.nan

    f_alpha_solution, Q_solution = solve_f_alpha_Q()
        
    # Once the convergence loop passed, every other parameters are calculated
    nbar_solution         = f_nbar(P_fus, nu_n, nu_T, f_alpha_solution, Tbar, R0, a, κ) 
    # alternative taking into account a reffined version of the volume :
    # nbar_solution         = f_nbar_advanced(P_fus, nu_n, nu_T, f_alpha_solution, Tbar, Volume_solution)
    pbar_solution         = f_pbar(nu_n,nu_T,nbar_solution,Tbar,f_alpha_solution)
    W_th_solution         = f_W_th(nbar_solution, Tbar, Volume_solution)
    # Radiative loss
    P_Brem_solution = f_P_bremsstrahlung(Volume_solution, nbar_solution, Tbar, Zeff, R0, a)
    beta_T = 2 # Beta_T taken from [J.Johner Helios]
    P_syn_solution = f_P_synchrotron(Tbar, R0, a, B0_solution, nbar_solution, κ, nu_n, nu_T, beta_T, r_synch)
    P_rad_solution = P_Brem_solution + P_syn_solution
    # Current drive efficiency
    eta_CD                = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
    if Operation_mode == 'Steady-State':
        P_Aux_solution = P_fus / Q_solution
        P_Ohm_solution = 0
    elif Operation_mode == 'Pulsed':
        P_Aux_solution = P_aux_input        
        P_Ohm_solution = P_fus / Q_solution - P_Aux_solution
    else:
        print("Choose a valid operation mode ")
        
    # Solve for energy confinement time
    tauE_solution = f_tauE(pbar_solution, Volume_solution, P_Alpha, P_Aux_solution, P_Ohm_solution, P_rad_solution)

    # Solve for alpha particles confinement time
    tau_alpha     = f_tau_alpha(nbar_solution, Tbar, tauE_solution, C_Alpha, nu_T)

    # Solve for Ip
    Ip_solution   = f_Ip(tauE_solution, R0, a, κ, δ, nbar_solution, B0_solution, Atomic_mass, 
                          P_Alpha, P_Ohm_solution, P_Aux_solution, P_rad_solution, H, C_SL,
                          alpha_delta,alpha_M,alpha_kappa,alpha_epsilon, alpha_R,alpha_B,alpha_n,alpha_I,alpha_P)

    # Calculate the bootstrap current
    if Bootstrap_choice == 'Freidberg' :
        Ib_solution = f_Freidberg_Ib(R0, a, κ, pbar_solution, Ip_solution)
    elif Bootstrap_choice == 'Segal' :
        Ib_solution = f_Segal_Ib(nu_n, nu_T, a/R0, κ, nbar_solution, Tbar, R0, Ip_solution)
    else :
        print("Choose a valid Bootstrap model")

    # Calculate the derived quantities from the solution found above
    qstar_solution        = f_qstar(a, B0_solution, R0, Ip_solution, κ)
    q95_solution          = f_q95(B0_solution,Ip_solution,R0,a,κ,δ)
    q_mhd_solution        = f_q_mhd(a, B0_solution, R0, Ip_solution, a/R0, κ, δ)
    B_pol_solution        = f_Bpol(q95_solution, B0_solution, a, R0)
    betaT_solution        = f_beta_T(pbar_solution,B0_solution)
    betaP_solution        = f_beta_P(a, κ, pbar_solution, Ip_solution)
    beta_solution         = f_beta(betaP_solution, betaT_solution)
    betaN_solution        = f_beta_N(betaT_solution, B0_solution, a, Ip_solution)
    nG_solution           = f_nG(Ip_solution, a)
    eta_CD_solution       = f_etaCD(a, R0, B0_solution, nbar_solution, Tbar, nu_n, nu_T)
    if Operation_mode == 'Steady-State':
        Temps_Plateau     = 0
        P_CD_solution     = f_PCD(R0,nbar_solution,Ip_solution,Ib_solution,eta_CD_solution)
        I_Ohm_solution    = 0
        I_CD_solution     = f_ICD(Ip_solution,Ib_solution, I_Ohm_solution)
        P_Ohm_solution    = 0
        Q_solution        = f_Q(P_fus,P_CD_solution,P_Ohm_solution)
    elif Operation_mode == 'Pulsed':
        P_CD_solution     = P_aux_input
        I_CD_solution     = f_I_CD(R0, nbar_solution, eta_CD_solution, P_CD_solution)
        I_Ohm_solution    = f_I_Ohm(Ip_solution, Ib_solution, I_CD_solution)
        P_Ohm_solution    = f_P_Ohm(I_Ohm_solution, Tbar, R0, a, κ)
        Q_solution        = f_Q(P_fus,P_CD_solution,P_Ohm_solution)
    else:
        print("Choose a valid operation mode ")
    P_sep_solution        = f_P_sep(P_fus, P_CD_solution)
    Gamma_n_solution      = f_Gamma_n(a, P_fus, R0, κ)
    heat_D0FUS_solution   = f_heat_D0FUS(R0,P_sep_solution)
    heat_par_solution     = f_heat_par(R0,B0_solution,P_sep_solution)
    heat_pol_solution     = f_heat_pol(R0,B0_solution,P_sep_solution,a,q95_solution)
    lambda_q_Eich_m, q_parallel0_Eich, q_target_Eich = f_heat_PFU_Eich(P_sep_solution, B_pol_solution, R0, a/R0, theta_deg)
    P_1rst_wall_Hmod      = f_P_1rst_wall_Hmod(P_sep_solution, P_CD_solution, Surface_solution)
    P_1rst_wall_Lmod      = f_P_1rst_wall_Lmod(P_sep_solution, Surface_solution)
    P_elec_solution       = f_P_elec(P_fus, P_CD_solution, eta_T, eta_RF)
    li_solution           = f_li(nu_n, nu_T)
    
    # Calculate the L-H threshold power:
    if L_H_Scaling_choice == 'Martin':
        P_Thresh = P_Thresh_Martin(nbar_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_S':
        P_Thresh = P_Thresh_New_S(nbar_solution, B0_solution, a, R0, κ, Atomic_mass)
    elif L_H_Scaling_choice == 'New_Ip':
        P_Thresh = P_Thresh_New_Ip(nbar_solution, B0_solution, a, R0, κ, Ip_solution, Atomic_mass)
    else:
        print('Choose a valid Scaling for L-H transition')
    
    # Calculate the radial build
    if Radial_build_model == "academic" :
        (c, Winding_pack_tension_ratio) = f_TF_academic(a, b, R0, σ_TF, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg)
        (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(Ip_solution, I_Ohm_solution, Bmax ,a ,b ,c , R0 ,κ ,nbar_solution, Tbar ,Ce ,Temps_Plateau_input , li_solution, Choice_Buck_Wedg)
        (d,Alpha,B_CS) = f_CS_academic(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, σ_CS, J_max_CS_conducteur, Choice_Buck_Wedg)
    elif Radial_build_model == "D0FUS" :
        (c, Winding_pack_tension_ratio) = f_TF_D0FUS(a, b, R0, σ_TF, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg, omega_TF, n_TF)
        (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(Ip_solution, I_Ohm_solution, Bmax ,a ,b ,c , R0 ,κ ,nbar_solution, Tbar ,Ce ,Temps_Plateau_input , li_solution, Choice_Buck_Wedg)
        (d,Alpha,B_CS, J_CS) = f_CS_D0FUS(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, σ_CS, J_max_CS_conducteur , Choice_Buck_Wedg)
    elif Radial_build_model == "CIRCEE" :
        (c, Winding_pack_tension_ratio) = f_TF_CIRCEE(a, b, R0, σ_TF, J_max_TF_conducteur, Bmax, Choice_Buck_Wedg, omega_TF, n_TF)
        (ΨPI, ΨRampUp, Ψplateau, ΨPF) = Magnetic_flux(Ip_solution, I_Ohm_solution, Bmax ,a ,b ,c , R0 ,κ ,nbar_solution, Tbar ,Ce ,Temps_Plateau_input , li_solution, Choice_Buck_Wedg)
        (d,Alpha,B_CS, J_CS) = f_CS_CIRCEE(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, σ_CS, J_max_CS_conducteur , Choice_Buck_Wedg)
    else :
        print('Choose a valid mechanical model')
    
    # Calculate a proxy for the machine cost   
    cost_solution = f_cost(a,b,c,d,R0,κ,P_fus)

    return (B0_solution, B_CS, B_pol_solution,
            tauE_solution, W_th_solution,
            Q_solution, Volume_solution, Surface_solution,
            Ip_solution, Ib_solution, I_CD_solution, I_Ohm_solution,
            nbar_solution, nG_solution, pbar_solution,
            betaN_solution, betaT_solution, betaP_solution,
            qstar_solution, q95_solution, q_mhd_solution,
            P_CD_solution, P_sep_solution, P_Thresh, eta_CD, P_elec_solution,
            cost_solution, P_Brem_solution, P_syn_solution,
            heat_D0FUS_solution, heat_par_solution, heat_pol_solution, lambda_q_Eich_m, q_target_Eich,
            P_1rst_wall_Hmod, P_1rst_wall_Lmod,
            Gamma_n_solution,
            f_alpha_solution, tau_alpha,
            J_max_TF_conducteur, J_max_CS_conducteur,
            Winding_pack_tension_ratio, R0-a, R0-a-b, R0-a-b-c, R0-a-b-c-d,
            κ, κ_95, δ, δ_95)


#%% 

if __name__ == "__main__":
    
    p = Parameters()
    
    if len(sys.argv) == 3:
        name_input_file = sys.argv[1]
        name_output_file = sys.argv[2] 
    else:
        name_input_file = "../D0FUS_run_input.txt"
        name_output_file = "../D0FUS_run_output.txt"
        
    p.open_input(name_input_file)

    (
        B0_solution, B_CS, B_pol_solution,
        tauE_solution, W_th_solution,
        Q_solution, Volume_solution, Surface_solution,
        Ip_solution, Ib_solution, I_CD_solution, I_Ohm_solution,
        nbar_solution, nG_solution, pbar_solution,
        betaN_solution, betaT_solution, betaP_solution,
        qstar_solution, q95_solution, q_mhd_solution,
        P_CD, P_sep, P_Thresh, eta_CD, P_elec_solution,
        cost, P_Brem_solution, P_syn_solution,
        heat_D0FUS_solution, heat_par_solution, heat_pol_solution,
        lambda_q_Eich_m, q_target_Eich,
        P_1rst_wall_Hmod, P_1rst_wall_Lmod,
        Gamma_n,
        f_alpha_solution, tau_alpha,
        J_max_TF_conducteur, J_max_CS_conducteur,
        TF_ratio, r_minor, r_sep, r_c, r_d,
        κ, κ_95, δ, δ_95
    ) = run(
    p.a, p.R0, p.Bmax, p.P_fus,
    p.Tbar, p.H, p.Temps_Plateau_input, p.b, p.nu_n, p.nu_T,
    p.Supra_choice, p.Chosen_Steel, p.Radial_build_model, p.Choice_Buck_Wedg,
    p.Option_Kappa, p.κ_manual, p.L_H_Scaling_choice, p.Scaling_Law,
    p.Bootstrap_choice, p.Operation_mode, p.fatigue, p.P_aux_input
)
    with open(name_output_file, "w", encoding="utf-8") as _f:
        class DualWriter:
            def __init__(self, *files):
                self._files = files
            def write(self, data):
                for ff in self._files:
                    ff.write(data)
            

        # Allows writing to the terminal and output file
        f = DualWriter(sys.stdout, _f)
        
        # Clean display of results
        print("=========================================================================", file=f)
        print("=== Calculation Results ===", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[I] R0 (Major Radius)                               : {p.R0:.3f} [m]", file=f)
        print(f"[I] a (Minor Radius)                                : {p.a:.3f} [m]", file=f)
        print(f"[I] b (BB & Nshield thickness)                      : {r_minor-r_sep:.3f} [m]", file=f)
        print(f"[O] c (TF coil thickness)                           : {r_sep-r_c:.3f} [m]", file=f)
        print(f"[O] d (CS thickness)                                : {r_c-r_d:.3f} [m]", file=f)
        print(f"[O] R0-a-b-c-d                                      : {r_d:.3f} [m]", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[O] Kappa (Elongation)                              : {κ:.3f} ", file=f)
        print(f"[O] Kappa_95 (Elongation at 95%)                    : {κ_95:.3f} ", file=f)
        print(f"[O] Delta (Triangularity)                           : {δ:.3f} ", file=f)
        print(f"[O] Delta_95 (Triangularity at 95%)                 : {δ_95:.3f} ", file=f)
        print(f"[O] Volume (Plasma)                                 : {Volume_solution:.3f} [m^3]", file=f)
        print(f"[O] Surface (1rst Wall)                             : {Surface_solution:.3f} [m²]", file=f)
        print(f"[I] Mechanical configuration                        : {p.Choice_Buck_Wedg} ", file=f)
        print(f"[I] Superconductor technology                       : {p.Supra_choice} ", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[I] Bmax (Maximum Magnetic Field - TF)              : {p.Bmax:.3f} [T]", file=f)
        print(f"[O] B0 (Central Magnetic Field)                     : {B0_solution:.3f} [T]", file=f)
        print(f"[O] BCS (Magnetic Field CS)                         : {B_CS:.3f} [T]", file=f)
        print(f"[O] J_E-TF (Enginnering current density TF)         : {J_max_TF_conducteur/1e6:.3f} [MA/m²]", file=f)
        print(f"[O] J_E-CS (Enginnering current density CS)         : {J_max_CS_conducteur/1e6:.3f} [MA/m²]", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[I] P_fus (Fusion Power)                            : {p.P_fus:.3f} [MW]", file=f)
        print(f"[O] P_CD (CD Power)                                 : {P_CD:.3f} [MW]", file=f)
        print(f"[O] P_S (Synchrotron Power)                         : {P_syn_solution:.3f} [MW]", file=f)
        print(f"[O] P_B (Bremsstrahlung Power)                      : {P_Brem_solution:.3f} [MW]", file=f)
        print(f"[O] eta_CD (CD Efficiency)                          : {eta_CD:.3f} [MA/MW-m²]", file=f)
        print(f"[O] Q (Energy Gain Factor)                          : {Q_solution:.3f}", file=f)
        print(f"[O] P_elec-net (Net Electrical Power)               : {P_elec_solution:.3f} [MW]", file=f)
        print(f"[O] Cost ((V_BB+V_TF+V_CS)/P_fus)                   : {cost:.3f} [m^3]", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[I] H (Scaling Law factor)                          : {p.H:.3f} ", file=f)
        print(f"[I] Operation (Pulsed / Steady)                     : {p.Operation_mode} ", file=f)
        print(f"[I] t (Plateau Time)                                : {p.Temps_Plateau_input:.3f} ", file=f)
        print(f"[O] tau_E (Confinement Time)                        : {tauE_solution:.3f} [s]", file=f)
        print(f"[O] Ip (Plasma Current)                             : {Ip_solution:.3f} [MA]", file=f)
        print(f"[O] Ib (Bootstrap Current)                          : {Ib_solution:.3f} [MA]", file=f)
        print(f"[O] ICD (Current Drive)                             : {I_CD_solution:.3f} [MA]", file=f)
        print(f"[O] IOhm (Ohmic Current)                            : {I_Ohm_solution:.3f} [MA]", file=f)
        print(f"[O] f_b (Bootstrap Fraction)                        : {(Ib_solution/Ip_solution)*1e2:.3f} [%]", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[I] Tbar (Mean Temperature)                         : {p.Tbar:.3f} [keV]", file=f)
        print(f"[O] nbar (Average Density)                          : {nbar_solution:.3f} [10^20 m^-3]", file=f)
        print(f"[O] nG (Greenwald Density)                          : {nG_solution:.3f} [10^20 m^-3]", file=f)
        print(f"[O] pbar (Average Pressure)                         : {pbar_solution:.3f} [MPa]", file=f)
        print(f"[O] Alpha Fraction                                  : {f_alpha_solution*1e2:.3f} [%]", file=f)
        print(f"[O] Alpha Confinement Time                          : {tau_alpha:.3f} [s]", file=f)
        print(f"[O] Thermal Energy Content                          : {W_th_solution/1e6:.3f} [MJ]", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[O] Beta_T (Toroidal Beta)                          : {betaT_solution*1e2:.3f} [%]", file=f)
        print(f"[O] Beta_P (Poloidal Beta)                          : {betaP_solution:.3f}", file=f)
        print(f"[O] Beta_N (Normalized Beta)                        : {betaN_solution:.3f} [%]", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[O] q* (Kink Safety Factor)                         : {qstar_solution:.3f}", file=f)
        print(f"[O] q95 (Safety Factor at 95%)                      : {q95_solution:.3f}", file=f)
        print(f"[O] q_MHD (MHD Safety Factor)                       : {q_mhd_solution:.3f}", file=f)
        print("-------------------------------------------------------------------------", file=f)
        print(f"[O] P_sep (Separatrix Power)                        : {P_sep:.3f} [MW]", file=f)
        print(f"[O] P_Thresh (L-H Power Threshold)                  : {P_Thresh:.3f} [MW]", file=f)
        print(f"[O] (P_sep - P_thresh) / S                          : {P_1rst_wall_Hmod:.3f} [MW/m²]", file=f)
        print(f"[O] P_sep / S                                       : {P_1rst_wall_Lmod:.3f} [MW/m²]", file=f)
        print(f"[O] Heat scaling (P_sep / R0)                       : {heat_D0FUS_solution:.3f} [MW/m]", file=f)
        print(f"[O] Parallel Heat Flux (P_sep*B0 / R0)              : {heat_par_solution:.3f} [MW-T/m]", file=f)
        print(f"[O] Poloidal Heat Flux (P_sep*B0) / (q95*R0*A)      : {heat_pol_solution:.3f} [MW-T/m]", file=f)
        print(f"[O] Gamma_n (Neutron Flux)                          : {Gamma_n:.3f} [MW/m²]", file=f)
        print("=========================================================================", file=f)
        
        sys.stdout = sys.__stdout__  
        
    
#%%

print("D0FUS_run loaded")