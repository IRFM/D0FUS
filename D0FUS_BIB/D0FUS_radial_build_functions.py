# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:32:03 2025

@author: TA276941
"""

#%% Import

from D0FUS_physical_functions import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% print

if __name__ == "__main__":
    print("##################################################### TF Model ##########################################################")
    
    
#%% Academic model

def f_TF_academic(a, b, R0, σ_TF, μ0, J_max_TF, Bmax, Choice_Buck_Wedg):
    
    """
    Calculate the thickness of the TF coil using a 2 layer thin cylinder model 

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
    
    B0 = f_B0(Bmax,a,b,R0)
    
    R1_0 = R0 - a - b # Jambe interne
    R2_0 = R0 + a + b # Jambe externe
    
    NI = 2*np.pi*R0*B0/μ0
    
    S_cond = NI/J_max_TF
    
    c1 = R1_0-np.sqrt(R1_0**2-S_cond/np.pi)
    
    R1 = R1_0 - c1
    R2 = R2_0 + c1
    
    # Vérifiez que R2_1 et R1_1 sont définis et que R2_1 / R1_1 est strictement positif
    Repartition_outboard_inboard_tension = 2
    if R2 > 0 and R1 > 0 and R2 / R1 > 0:
        T = ((np.pi * B0**2 * R0**2) / μ0 * math.log(R2 / R1) - F_CClamp )/ Repartition_outboard_inboard_tension 
        if T < 0:
            T = 0
    else:
        # raise ValueError("Les valeurs de R2_1 et R1_1 doivent être strictement positives et R2_1 / R1_1 doit être strictement positif.")
        return(np.nan, np.nan)
    P = Bmax**2 / (2*μ0)
    
    if Choice_Buck_Wedg == "Bucking" :
        c2 = (B0**2*R0**2)*math.log(R2/R1)/(2*μ0*2*R1*(σ_TF-P)) # Valable si R1>>c
        σ_r = P
        σ_z = T / (2*np.pi*R1*c2)
        ratio_tension = σ_z / (σ_z + σ_r)
        
    elif Choice_Buck_Wedg == "Wedging":
        c2 = (B0**2*R0**2)/(2*μ0*R1*σ_TF)*(1+math.log(R2/R1)/2) # Valable si R1>>c
        σ_theta = P*R1/c2
        σ_z = T / (2*np.pi*R1*c2)
        ratio_tension = σ_z/(σ_theta+σ_z)
    else :
        print('Choose a valid mechanical option')
        
    c  = c1 + c2

    return(c, ratio_tension)
    
#%% Modèles simple cylindres épais

def f_TF_simple(a, b, R0, σ_TF, μ0, J_max_TF, Bmax, Choice_Buck_Wedg):
    
    """
    Calculate the thickness of the TF coil using a 2 layer thick cylinder model 

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
    
    B0 = f_B0(Bmax,a,b,R0)
    
    R1_0 = R0 - a - b # Jambe interne bobine côté plasma
    R2_0 = R0 + a + b # Jambe externe bobine côté plasma
    
    # calcul de c1 pour générer le champ magnétique voulu
    c1 = Bmax / (μ0 * J_max_TF)
    
    R1_1 = R1_0 - c1 # Jambe interne séparation conducteur / acier
    R2_1 = R2_0 + c1 # Jambe externe séparation conducteur / acier
    
    Repartition_outboard_inboard_tension = 2
    # Vérifiez que R2_1 et R1_1 sont définis et que R2_1 / R1_1 est strictement positif
    if R2_1 > 0 and R1_1 > 0 and R2_1 / R1_1 > 0:
        T = ((np.pi * B0**2 * R0**2) / μ0 * math.log(R2_1 / R1_1) - F_CClamp )/ Repartition_outboard_inboard_tension
        if T < 0:
            T = 0
    else:
        # raise ValueError("Les valeurs de R2_1 et R1_1 doivent être strictement positives et R2_1 / R1_1 doit être strictement positif.")
        return(np.nan, np.nan)
    P = Bmax**2 / (2 * μ0)  # Pression magnétique
    
    
    # Calcul épaisseur d'acier c2
    if Choice_Buck_Wedg == "Wedging":
        
        # Calcul de c2 :
        discriminant = R1_1**2 * σ_TF**2 - 2 * σ_TF * P * R1_1**2 - σ_TF * (T / np.pi)
        # Vérification si la racine est positive
        if discriminant < 0:
            # raise ValueError("Le discriminant est négatif, pas de solution réelle pour c2.")
            return(np.nan, np.nan)
        
        # Calcul des deux solutions possibles
        sqrt_term = np.sqrt(discriminant)
        c2_sol1 = R1_1 + sqrt_term / σ_TF
        c2_sol2 = R1_1 - sqrt_term / σ_TF
        # Retourner uniquement la solution physique (c2 doit être positif et < R1_1)
        c2_solutions = [c2 for c2 in (c2_sol1, c2_sol2) if 0 < c2 < R1_1]
        if not c2_solutions:
            # raise ValueError("Aucune solution physique pour c2.")
            return(np.nan, np.nan)
        c2 = min(c2_solutions)  # On prend la plus petite valeur admissible
        
        # Calcul des efforts
        σ_theta = 2 * P * R1_1**2 / (R1_1**2 - (R1_1-c2)**2)
        σ_z = T / (np.pi * (R1_1**2 - (R1_1-c2)**2))
        
        # Calcul du ratio dans la TF entre tension et Tresca
        ratio_tension = σ_z / (σ_theta + σ_z)
        
        # Epaisseur totale de la bobine
        c  = c1 + c2
    
    elif Choice_Buck_Wedg == "Bucking":
        
        # Vérification de la validité de l'entrée
        if σ_TF <= P:
            # raise ValueError("sigma_TRESCA doit être strictement supérieur à P pour éviter une division par zéro.")
            return(np.nan, np.nan)
    
        # Calcul du discriminant sous la racine
        discriminant = R1_1**2 - (T / (np.pi * (σ_TF - P)))
        
        # Vérification si la racine est positive
        if discriminant < 0:
            # raise ValueError("Le discriminant est négatif, pas de solution réelle pour c2.")
            return(np.nan, np.nan)
        
        # Calcul des deux solutions possibles
        sqrt_term = np.sqrt(discriminant)
        c2_sol1 = R1_1 + sqrt_term
        c2_sol2 = R1_1 - sqrt_term
        
        # Retourner uniquement la solution physique (c2 doit être positif et < R1_1)
        c2_solutions = [c2 for c2 in (c2_sol1, c2_sol2) if 0 < c2 < R1_1]
        
        if not c2_solutions:
            # raise ValueError("Aucune solution physique pour c2.")
            return(np.nan, np.nan)
        
        c2 =  min(c2_solutions)  # On prend la plus petite valeur admissible
        
        # Calcul des efforts
        σ_r = P
        σ_z = T / (np.pi * (R1_1**2 - (R1_1-c2)**2))
        
        # Calcul du ratio dans la TF entre tension et Tresca
        ratio_tension = σ_z / (σ_r + σ_z)
        
        # Epaisseur totale de la bobine
        c  = c1 + c2
        
    else : 
        print( "Choose a valid mechanical configuration" )
        return(np.nan, np.nan)

    return(c, ratio_tension)
    
#%% Modèle TF avec cylindre épais + dilution de manière analytique

def Winding_Pack_complex(R_0, a, b, sigma_max, J_max, mu_0, B_max, gamma, beta):
    """
    Computes the winding pack thickness according to the analytical formula.
    
    Args:
        R_e: Outer radius [m]
        R_0: Reference radius [m]
        a, b: Geometric dimensions [m]
        sigma_max: Maximum stress [Pa]
        J_max: Maximum current density [A/m²]
        mu_0: Vacuum permeability [H/m]
        B_max: Maximum magnetic field [T]
        gamma: Geometric factor [dimensionless]
        beta: Additional scaling factor [dimensionless]
        
    Returns:
        R_i: Computed inner radius [m]
        
    Raises:
        ValueError: For physically invalid conditions
    """
    R_e = R_0 - a - b
    ln_term = np.log((R_0 + a + b) / (R_0 - a - b))
    
    # Numerator calculation (modified terms)
    term1 = (2 * sigma_max * B_max * R_e) / (mu_0 * J_max)  # Removed gamma division
    term2 = (beta * B_max**2 * R_e**2 * ln_term) / mu_0     # Added beta factor
    numerator = term1 + term2
    
    # Denominator calculation (modified form)
    denominator = sigma_max - (B_max**2 / (2 * mu_0 * gamma))  # Gamma moved to denominator
    
    if abs(denominator) < 1e-12:
        # raise ValueError("Invalid physical condition: zero denominator")
        return(np.nan, np.nan)
    
    R_i_squared = R_e**2 - (numerator / denominator)
    
    if R_i_squared < 0:
        # raise ValueError("Non-physical solution: negative R_i²")
        return(np.nan, np.nan)
        
    R_i =  np.sqrt(R_i_squared)
    
    
    # Calculs supplémentaires pour ratio_tension
    # Verification de la singularité en Re = Ri
    D = R_e**2 - R_i**2
    if D == 0:
        return np.inf

    log_term = np.log((R_0 + a + b) / (R_0 - a - b))
    P = B_max**2 / (2 * mu_0)
    
    alpha = (2 * B_max * R_e) / (mu_0 * J_max * D)
    sigma_r = P
    sigma_z = (B_max**2 * R_e**2 * log_term) / (mu_0 * D)
    
    ratio_tension = (sigma_z * beta) / ((sigma_r / gamma) + (sigma_z * beta))
    
    return (R_e - R_i, ratio_tension)

def Nose_complex(R_m, B_max, mu_0, sigma_max, beta, R_0, a, b):
    """
    Calcule Nose Thickness à partir des paramètres donnés.
    
    Paramètres :
    - R_e : Rayon extérieur (m)
    - B_max : Champ magnétique maximal (T)
    - mu_0 : Perméabilité magnétique du vide (H/m)
    - sigma_max : Contrainte maximale admissible (Pa)
    - theta : Paramètre theta (sans dimension)
    - R_0 : Paramètre R_0 (m)
    - a : Paramètre a (m)
    - b : Paramètre b (m)
    
    Retourne :
    - R_i : Nose interior radius (m)
    """
    
    log_term = np.log((R_0 + a + b) / (R_0 - a - b))
    term = 1 + (1 - beta) * log_term
    R_i_squared = R_m**2 * (1 - (B_max**2 / (mu_0 * sigma_max)) * term)
    
    if R_i_squared < 0:
        # raise ValueError("Le calcul donne un R_i^2 négatif. Vérifiez les paramètres.")
        return(np.nan)
    
    R_i = np.sqrt(R_i_squared)
    
    return (R_m - R_i)

def f_TF_complex(a, b, R0, σ_TF, μ0, J_max_TF, Bmax, Choice_Buck_Wedg, gamma, beta):
    
    """
    Calculate the thickness of the TF coil using a 2 layer thick cylinder model 

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
    
    if Choice_Buck_Wedg == "Wedging":
        
        (c_WP, ratio_tension) = Winding_Pack_complex( R0, a, b, σ_TF, J_max_TF, μ0, Bmax, gamma, beta)
        
        # Vérification que c_WP est valide
        if c_WP is None or np.isnan(c_WP) or c_WP < 0:
            if __name__ == "__main__":
                raise ValueError("c_WP invalide : négatif ou NaN")
            return(np.nan, np.nan)
        
        c_Nose = Nose_complex(R0 - a - b - c_WP , Bmax, μ0, σ_TF, beta, R0, a, b)
        
        # Vérification que c_Nose est valide
        if c_Nose is None or np.isnan(c_Nose) or c_Nose < 0:
            if __name__ == "__main__":
                raise ValueError("c_Nose invalide : négatif ou NaN")
            return(np.nan, np.nan)
        
        # Vérification que la somme ne dépasse pas R0 - a - b
        if (c_WP + c_Nose) > (R0 - a - b):
            if __name__ == "__main__":
                raise ValueError("La somme de c_WP et c_Nose dépasse R0 - a - b")
            return(np.nan, np.nan)
        
        # Epaisseur totale de la bobine
        c  = c_WP + c_Nose
    
    elif Choice_Buck_Wedg == "Bucking":
        
        (c, ratio_tension) = Winding_Pack_complex(R0, a, b, σ_TF, J_max_TF, μ0, Bmax, gamma, beta)
        
    else : 
        print( "Choose a valid mechanical configuration" )
        return(np.nan, np.nan)

    return(c, ratio_tension)

#%% TF benchmark

if __name__ == "__main__":
    
    # ITER test parameters
    a_TF = 2.2
    b_TF = 1.45
    R0_TF = 6.2
    σ_TF_tf = 660e6
    μ0_TF = 4 * np.pi * 1e-7
    J_max_TF_tf = 50 * 1e6
    Bmax_TF = 12.7

    # Test the function
    result_academic_wedging = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging")
    print(f"TF academic wedging : {result_academic_wedging}")
    result_academic_bucking = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking")
    print(f"TF academic bucking : {result_academic_bucking}")

    # Test the function
    result_epais_wedging = f_TF_simple(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging")
    print(f"TF simple wedging : {result_epais_wedging}")
    result_epais_bucking = f_TF_simple(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking")
    print(f"TF simple bucking  : {result_epais_bucking}")
    
    result_complex_wedging = f_TF_complex(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging", gamma_TF, beta_TF)
    print(f"TF complex wedging : {result_complex_wedging}")
    result_complex_bucking = f_TF_complex(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking", gamma_TF, beta_TF)
    print(f"TF complex bucking  : {result_complex_bucking}")
    
#%% Cylindre épais + dilution


#%% Print
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% CS thin layer model

def f_CS_academic_convergence(a, b, c, R0, Bmax, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
    
    """
    Calculate the CS thickness considering a thin layer approximation and a 2 cylinder (supra + steel) approach

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
    B0 = f_B0(Bmax,a,b,R0)
    
    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS
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
    
    # Fonction cible
    def CS_to_solve(d_SS):
        """
        Calcule la contrainte mécanique et retourne la différence avec la contrainte cible σ_CS.
        """
        # J cross B magnetic pressure
        P_CS = B_CS**2 / (2 * μ0)
        #  magnetic pressure
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / (RCS_ext)
    
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = np.nanmax( [P_TF , abs(P_CS - P_TF) ] ) * ri_c / d_SS
            
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = P_CS * ri_c / d_SS
            
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
    
        return Sigma_CS - σ_CS
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='CS_to_solve(d_SS)')
        plt.axhline(y=0, color='r', linestyle='--', label='y=0')
        plt.grid(True)
        plt.xlabel('d_SS')
        plt.ylabel('Function Value')
        plt.title('Comportement de CS_to_solve')
        plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        
        return zero_crossings
    
    # Détection des changements de signe
    def find_sign_changes(f, a, b, n=1000):
        """ 
        Détecte les intervalles où la fonction change de signe.
        """
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        
        sign_changes = []
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Changement de signe détecté
                sign_changes.append((x_vals[i-1], x_vals[i]))
    
        return sign_changes
    
    # Raffinement des passages par zéro avec brentq
    def refine_zeros(f, sign_change_intervals):
        """ 
        Utilise la méthode de Brent pour affiner les passages par zéro.
        """
        roots = []
        for a, b in sign_change_intervals:
            try:
                root = brentq(f, a, b)  # Brent garantit une convergence si un changement de signe est détecté
                roots.append(root)
            except ValueError:
                pass  # Si Brent échoue, on ignore l'intervalle
    
        return roots if roots else [np.nan]  # Si aucune racine n'est trouvée, retourne np.nan
    
    # Recherche des solutions en utilisant la méthode de Brent
    def find_d_SS_solution_brentq(a, b, n=1000):
        """
        Trouve les passages par zéro de CS_to_solve entre a et b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
    
        return roots
    
    # Vérification que la solution est bien un passage par zéro
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-6):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        - Vérifie que |f(root)| est proche de 0 (précision tol)
        - Vérifie un changement de signe autour de root (epsilon)
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
    
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
    
        if np.sign(f_left) != np.sign(f_right):  # Changement de signe autour de root
            return True
        else:
            return False  # Probablement un minimum local
    
    # Trouver la plus petite solution valide avec fsolve
    def find_d_SS_solution_fsolve(initial_guesses):
        """
        Cherche une solution avec fsolve en s'assurant qu'il s'agit bien d'un passage par zéro.
        - initial_guesses : Liste de points de départ pour fsolve
        - Retourne la plus petite solution valide ou np.nan si aucune racine correcte n'est trouvée
        """
        valid_solutions = []
    
        for guess in initial_guesses:
            root_candidate, info, ier, msg = fsolve(CS_to_solve, guess, full_output=True)
    
            if ier == 1:  # Vérifier si la convergence a réussi
                root = root_candidate[0]
                if is_valid_root(CS_to_solve, root):  # Vérifier que c'est bien un passage par zéro
                    valid_solutions.append(root)
    
        return min(valid_solutions) if valid_solutions else np.nan  # Prendre la plus petite solution
    
    def find_d_SS_solution_root(a, b):
        """
        Trouve les passages par zéro de CS_to_solve en utilisant scipy.optimize.root.
        """
        initial_guesses = np.linspace(a, b, 10)
        valid_solutions = []
    
        for guess in initial_guesses:
            sol = root(CS_to_solve, guess, method='hybr')
            if sol.success and is_valid_root(CS_to_solve, sol.x[0]):
                valid_solutions.append(sol.x[0])
    
        return min(valid_solutions) if valid_solutions else np.nan
    
    def is_valid_root(f, root, epsilon=0.01, tol=1e6):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
        
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
        
        return np.sign(f_left) != np.sign(f_right)  # Changement de signe autour de root
    
    try:
        
        """
        ### `fsolve`  
        Basé sur la méthode de Newton-Raphson (et ses variantes quasi-Newtoniennes)
        Utilise la dérivée locale pour ajuster l'approximation de la racine  
        Problème : Peut converger vers un **minimum local** au lieu d'un vrai passage par zéro si la dérivée est faible  
        
        ### `brentq`  
        Basé sur la méthode de Brent (hybride entre la bissection et la sécante)
        Nécessite un intervalle `[a, b]` avec un changement de signe
        Avantage : Garantie de trouver un passage exact par zéro si un changement de signe est détecté
        
        ### `root`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Permet d'utiliser plusieurs algorithmes (`hybr`, `lm`, `broyden1`, `broyden2`, etc.)  
        Par défaut (`method='hybr'`), il repose sur une approche dérivée de **l’algorithme de Powell** (similaire à Newton-Raphson)  
        Problème : Comme `fsolve`, il peut converger vers un minimum local au lieu d’un vrai passage par zéro si la dérivée est faible
        
        ### `manual`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Méthode maison parcourant toute les solutions et cherchant les changement de signe
        Problème : peu précise et couteuse en temps de calcul
        Avantage : Impossible de faire plus robuste et simple
        """
        
        if Choice_solving_CS_method == "fsolve":
        
            # Liste d'estimations initiales pour couvrir plusieurs racines possibles
            initial_guesses = np.linspace(1e-3, ri_c, 10)  # 10 points entre 1e-3 et ri_c
            # Trouver la solution avec fsolve
            d_SS_solution = find_d_SS_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            
        elif Choice_solving_CS_method == "brentq":
        
            # Trouver la solution avec brentq
            d_SS_solutions = find_d_SS_solution_brentq(0.01, ri_c)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            d_SS_solution = find_d_SS_solution_root(0.01, ri_c)
            # print(d_SS_solution)
            
        elif Choice_solving_CS_method == "manual":

            d_SS_solutions = plot_function_CS(CS_to_solve, [0.01, ri_c])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        else:
            print("Choisissez une méthode valide pour le CS")
            return (np.nan, np.nan, np.nan)
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results filtering ####
        
        RCS_int_solution = ri_c - d_SS_solution
        Dilution_conductor = np.pi*(RCS_ext**2 - ri_c**2) / ( np.pi*(RCS_ext**2 - ri_c**2) + np.pi*(ri_c**2**2 - RCS_int_solution**2))
        
        if d_SS_solution < 0 :
            return(np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        if RCS_int_solution < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, Dilution_conductor, B_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan)


#%% CS thick cylinder model

def f_CS_simple_convergence(a, b, c, R0, Bmax, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
    
    """
    Calculate the CS thickness considering a thick layer approximation and a 2 cylinder (supra + steel) approach

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
    B0 = f_B0(Bmax,a,b,R0)
    
    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS
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
    
    # Fonction cible
    def CS_to_solve(d_SS):
        """
        Calcule la contrainte mécanique et retourne la différence avec la contrainte cible σ_CS.
        """
        # J cross B magnetic pressure
        P_CS = B_CS**2 / (2 * μ0)
        #  magnetic pressure
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / (RCS_ext)
    
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = np.nanmax( [P_TF , abs(P_CS - P_TF) ] ) * 2 * ri_c**2 / (ri_c**2 - (ri_c - d_SS)**2)
            
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = (P_CS * (ri_c - d_SS)**2) / (ri_c**2 - (ri_c - d_SS)**2) * ((ri_c**2 / (ri_c - d_SS)**2) + 1)
            
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
    
        return Sigma_CS - σ_CS
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='CS_to_solve(d_SS)')
        plt.axhline(y=0, color='r', linestyle='--', label='y=0')
        plt.grid(True)
        plt.xlabel('d_SS')
        plt.ylabel('Function Value')
        plt.title('Comportement de CS_to_solve')
        plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        
        return zero_crossings
    
    # Détection des changements de signe
    def find_sign_changes(f, a, b, n=1000):
        """ 
        Détecte les intervalles où la fonction change de signe.
        """
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        
        sign_changes = []
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Changement de signe détecté
                sign_changes.append((x_vals[i-1], x_vals[i]))
    
        return sign_changes
    
    # Raffinement des passages par zéro avec brentq
    def refine_zeros(f, sign_change_intervals):
        """ 
        Utilise la méthode de Brent pour affiner les passages par zéro.
        """
        roots = []
        for a, b in sign_change_intervals:
            try:
                root = brentq(f, a, b)  # Brent garantit une convergence si un changement de signe est détecté
                roots.append(root)
            except ValueError:
                pass  # Si Brent échoue, on ignore l'intervalle
    
        return roots if roots else [np.nan]  # Si aucune racine n'est trouvée, retourne np.nan
    
    # Recherche des solutions en utilisant la méthode de Brent
    def find_d_SS_solution_brentq(a, b, n=1000):
        """
        Trouve les passages par zéro de CS_to_solve entre a et b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
    
        return roots
    
    # Vérification que la solution est bien un passage par zéro
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-6):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        - Vérifie que |f(root)| est proche de 0 (précision tol)
        - Vérifie un changement de signe autour de root (epsilon)
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
    
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
    
        if np.sign(f_left) != np.sign(f_right):  # Changement de signe autour de root
            return True
        else:
            return False  # Probablement un minimum local
    
    # Trouver la plus petite solution valide avec fsolve
    def find_d_SS_solution_fsolve(initial_guesses):
        """
        Cherche une solution avec fsolve en s'assurant qu'il s'agit bien d'un passage par zéro.
        - initial_guesses : Liste de points de départ pour fsolve
        - Retourne la plus petite solution valide ou np.nan si aucune racine correcte n'est trouvée
        """
        valid_solutions = []
    
        for guess in initial_guesses:
            root_candidate, info, ier, msg = fsolve(CS_to_solve, guess, full_output=True)
    
            if ier == 1:  # Vérifier si la convergence a réussi
                root = root_candidate[0]
                if is_valid_root(CS_to_solve, root):  # Vérifier que c'est bien un passage par zéro
                    valid_solutions.append(root)
    
        return min(valid_solutions) if valid_solutions else np.nan  # Prendre la plus petite solution
    
    def find_d_SS_solution_root(a, b):
        """
        Trouve les passages par zéro de CS_to_solve en utilisant scipy.optimize.root.
        """
        initial_guesses = np.linspace(a, b, 10)
        valid_solutions = []
    
        for guess in initial_guesses:
            sol = root(CS_to_solve, guess, method='hybr')
            if sol.success and is_valid_root(CS_to_solve, sol.x[0]):
                valid_solutions.append(sol.x[0])
    
        return min(valid_solutions) if valid_solutions else np.nan
    
    def is_valid_root(f, root, epsilon=0.01, tol=1e6):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
        
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
        
        return np.sign(f_left) != np.sign(f_right)  # Changement de signe autour de root
    
    try:
        
        """
        ### `fsolve`  
        Basé sur la méthode de Newton-Raphson (et ses variantes quasi-Newtoniennes)
        Utilise la dérivée locale pour ajuster l'approximation de la racine  
        Problème : Peut converger vers un **minimum local** au lieu d'un vrai passage par zéro si la dérivée est faible  
        
        ### `brentq`  
        Basé sur la méthode de Brent (hybride entre la bissection et la sécante)
        Nécessite un intervalle `[a, b]` avec un changement de signe
        Avantage : Garantie de trouver un passage exact par zéro si un changement de signe est détecté
        
        ### `root`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Permet d'utiliser plusieurs algorithmes (`hybr`, `lm`, `broyden1`, `broyden2`, etc.)  
        Par défaut (`method='hybr'`), il repose sur une approche dérivée de **l’algorithme de Powell** (similaire à Newton-Raphson)  
        Problème : Comme `fsolve`, il peut converger vers un minimum local au lieu d’un vrai passage par zéro si la dérivée est faible
        
        ### `manual`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Méthode maison parcourant toute les solutions et cherchant les changement de signe
        Problème : peu précise et couteuse en temps de calcul
        Avantage : Impossible de faire plus robuste et simple
        """
        
        if Choice_solving_CS_method == "fsolve":
        
            # Liste d'estimations initiales pour couvrir plusieurs racines possibles
            initial_guesses = np.linspace(1e-3, ri_c, 10)  # 10 points entre 1e-3 et ri_c
            # Trouver la solution avec fsolve
            d_SS_solution = find_d_SS_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            
        elif Choice_solving_CS_method == "brentq":
        
            # Trouver la solution avec brentq
            d_SS_solutions = find_d_SS_solution_brentq(0.01, ri_c)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            d_SS_solution = find_d_SS_solution_root(0.01, ri_c)
            # print(d_SS_solution)
            
        elif Choice_solving_CS_method == "manual":

            d_SS_solutions = plot_function_CS(CS_to_solve, [0.01, ri_c])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        else:
            print("Choisissez une méthode valide pour le CS")
            return (np.nan, np.nan, np.nan)
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results filtering ####
        
        RCS_int_solution = ri_c - d_SS_solution
        Dilution_conductor = np.pi*(RCS_ext**2 - ri_c**2) / ( np.pi*(RCS_ext**2 - ri_c**2) + np.pi*(ri_c**2**2 - RCS_int_solution**2))
        
        if d_SS_solution < 0 :
            return(np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        if RCS_int_solution < 0:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, Dilution_conductor, B_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
#%% CS complexe

def f_CS_complex_convergence(a, b, c, R0, Bmax, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib, gamma):
    
    """
    Calculate the CS thickness considering a thick layer approximation and a 2 cylinder (supra + steel) approach

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
    B0 = f_B0(Bmax,a,b,R0)
    
    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))
    
    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS
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
    
#---------------------------------------------------------------------------------------------------------------------------------------- 
    #### Solving RCS_int ####
    
    # Fonction cible
    def CS_to_solve(d):
        """
        Calcule la contrainte mécanique et retourne la différence avec la contrainte cible σ_CS.
        """
        
        # Preliminary calculations
        RCS_int = RCS_ext - d
        # Dilution coeficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        # magnetic pressure to support from the CS
        P_CS = (μ0 * J_max_CS**2 * alpha**2 / 2) * (RCS_ext - RCS_int)**2
        # magnetic pressure to support from the TF
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / (RCS_ext)
    
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_light = ((P_CS-P_TF) * (RCS_ext**2 + RCS_int**2)) / (RCS_ext**2 - RCS_int**2)
            Sigma_strong = (P_TF * RCS_ext**2) / (RCS_ext**2 - RCS_int**2)
            Sigma_CS =  max(abs(Sigma_light),abs(Sigma_strong)) / ((1 - alpha) * gamma)
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = (1 / ((1 - alpha) * gamma)) * (P_CS * ((RCS_ext**2 + RCS_int**2) / (RCS_ext**2 - RCS_int**2)))
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
            
        # Results filtering
        if d < 0  or d > RCS_ext :
            return(np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan)
        if alpha < 0 or alpha > 1 :
            return (np.nan)
        else:
            return Sigma_CS - σ_CS
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='CS_to_solve(d)')
        plt.axhline(y=0, color='r', linestyle='--', label='y=0')
        plt.grid(True)
        plt.xlabel('d')
        plt.ylabel('Function Value')
        plt.title('Comportement de CS_to_solve')
        plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        
        return zero_crossings
    
    # Détection des changements de signe
    def find_sign_changes(f, a, b, n=1000):
        """ 
        Détecte les intervalles où la fonction change de signe.
        """
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        
        sign_changes = []
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Changement de signe détecté
                sign_changes.append((x_vals[i-1], x_vals[i]))
    
        return sign_changes
    
    # Raffinement des passages par zéro avec brentq
    def refine_zeros(f, sign_change_intervals):
        """ 
        Utilise la méthode de Brent pour affiner les passages par zéro.
        """
        roots = []
        for a, b in sign_change_intervals:
            try:
                root = brentq(f, a, b)  # Brent garantit une convergence si un changement de signe est détecté
                roots.append(root)
            except ValueError:
                pass  # Si Brent échoue, on ignore l'intervalle
    
        return roots if roots else [np.nan]  # Si aucune racine n'est trouvée, retourne np.nan
    
    # Recherche des solutions en utilisant la méthode de Brent
    def find_d_SS_solution_brentq(a, b, n=1000):
        """
        Trouve les passages par zéro de CS_to_solve entre a et b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
    
        return roots
    
    # Vérification que la solution est bien un passage par zéro
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-6):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        - Vérifie que |f(root)| est proche de 0 (précision tol)
        - Vérifie un changement de signe autour de root (epsilon)
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
    
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
    
        if np.sign(f_left) != np.sign(f_right):  # Changement de signe autour de root
            return True
        else:
            return False  # Probablement un minimum local
    
    # Trouver la plus petite solution valide avec fsolve
    def find_d_SS_solution_fsolve(initial_guesses):
        """
        Cherche une solution avec fsolve en s'assurant qu'il s'agit bien d'un passage par zéro.
        - initial_guesses : Liste de points de départ pour fsolve
        - Retourne la plus petite solution valide ou np.nan si aucune racine correcte n'est trouvée
        """
        valid_solutions = []
    
        for guess in initial_guesses:
            root_candidate, info, ier, msg = fsolve(CS_to_solve, guess, full_output=True)
    
            if ier == 1:  # Vérifier si la convergence a réussi
                root = root_candidate[0]
                if is_valid_root(CS_to_solve, root):  # Vérifier que c'est bien un passage par zéro
                    valid_solutions.append(root)
    
        return min(valid_solutions) if valid_solutions else np.nan  # Prendre la plus petite solution
    
    def find_d_SS_solution_root(a, b):
        """
        Trouve les passages par zéro de CS_to_solve en utilisant scipy.optimize.root.
        """
        initial_guesses = np.linspace(a, b, 10)
        valid_solutions = []
    
        for guess in initial_guesses:
            sol = root(CS_to_solve, guess, method='hybr')
            if sol.success and is_valid_root(CS_to_solve, sol.x[0]):
                valid_solutions.append(sol.x[0])
    
        return min(valid_solutions) if valid_solutions else np.nan
    
    def is_valid_root(f, root, epsilon=0.01, tol=1e6):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
        
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
        
        return np.sign(f_left) != np.sign(f_right)  # Changement de signe autour de root
    
    try:
        
        """
        ### `fsolve`  
        Basé sur la méthode de Newton-Raphson (et ses variantes quasi-Newtoniennes)
        Utilise la dérivée locale pour ajuster l'approximation de la racine  
        Problème : Peut converger vers un **minimum local** au lieu d'un vrai passage par zéro si la dérivée est faible  
        
        ### `brentq`  
        Basé sur la méthode de Brent (hybride entre la bissection et la sécante)
        Nécessite un intervalle `[a, b]` avec un changement de signe
        Avantage : Garantie de trouver un passage exact par zéro si un changement de signe est détecté
        
        ### `root`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Permet d'utiliser plusieurs algorithmes (`hybr`, `lm`, `broyden1`, `broyden2`, etc.)  
        Par défaut (`method='hybr'`), il repose sur une approche dérivée de **l’algorithme de Powell** (similaire à Newton-Raphson)  
        Problème : Comme `fsolve`, il peut converger vers un minimum local au lieu d’un vrai passage par zéro si la dérivée est faible
        
        ### `manual`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Méthode maison parcourant toute les solutions et cherchant les changement de signe
        Problème : peu précise et couteuse en temps de calcul
        Avantage : Impossible de faire plus robuste et simple
        """
        
        if Choice_solving_CS_method == "fsolve":
            
            # Liste d'estimations initiales pour couvrir plusieurs racines possibles
            initial_guesses = np.linspace(0.01, RCS_ext, 10)  # 10 points entre 1e-3 et ri_c
            # Trouver la solution avec fsolve
            d_SS_solution = find_d_SS_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            
        elif Choice_solving_CS_method == "brentq":
        
            # Trouver la solution avec brentq
            d_SS_solutions = find_d_SS_solution_brentq(0.01, RCS_ext)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            d_SS_solution = find_d_SS_solution_root(0.01, RCS_ext)
            # print(d_SS_solution)
            
        elif Choice_solving_CS_method == "manual":

            d_solutions = plot_function_CS(CS_to_solve, [0.01, RCS_ext])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        else:
            print("Choisissez une méthode valide pour le CS")
            return (np.nan, np.nan, np.nan)
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results filtering ####
        
        RCS_int_solution = RCS_ext - d_SS_solution
        Dilution_conductor = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int_solution) * (RCS_ext**2 + RCS_ext * RCS_int_solution + RCS_int_solution**2))
        B_CS = μ0 * J_max_CS * Dilution_conductor * (RCS_ext - RCS_int_solution)
        
        if d_SS_solution < 0 or d_SS_solution > RCS_ext:
            return(np.nan, np.nan, np.nan)
        if B_CS > Bmax or B_CS < 0:
            return (np.nan, np.nan, np.nan)
        if Dilution_conductor < 0 or Dilution_conductor > 1:
            return (np.nan, np.nan, np.nan)
        else:
            d = d_SS_solution
            return (d, Dilution_conductor, B_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan)
    
#%% CS Test

if __name__ == "__main__":
    
    # Test parameters Centering (ITER)
    a_cs = 2
    b_cs = 1.45
    c_cs = 0.9
    R0_cs = 6.2
    Bmax_cs = 12.7
    σ_CS_cs = 660e6
    μ0_cs = 4 * np.pi * 1e-7
    J_max_CS_cs = 50e6
    Tbar_cs = 14
    nbar_cs = 1
    Ip_cs = 12
    Ib_cs = 8

    result_CS5 = f_CS_academic_convergence(a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    print(f"CS academic model Wedging : {result_CS5}")
    result_CS6 = f_CS_academic_convergence(a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    print(f"CS academic model Bucking : {result_CS6}")
    result_CS1 = f_CS_simple_convergence(a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    print(f"CS simple model Wedging : {result_CS1}")
    result_CS2 = f_CS_simple_convergence(a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    print(f"CS simple model Bucking : {result_CS2}")
    result_CS7 = f_CS_complex_convergence(a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
    print(f"CS complex model Wedging : {result_CS7}")
    result_CS8 = f_CS_complex_convergence(a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
    print(f"CS complex model Bucking : {result_CS8}")

#%% Print
        
if __name__ == "__main__":
    print("##################################################### CIRCEE Model ##########################################################")

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
    R = np.array([2.13,2.56,2.75])  # Radii
    J = np.array([0,50e6])  # Current densities
    B = np.array([0,12])  # Magnetic fields
    Pi = 0  # Internal pressure
    Pe = 0  # External pressure
    E = np.array([200e9,20e9])  # Young's moduli
    nu = 0.29  # Poisson's ratio
    config = np.array([1,1])  # Wedging or bucking
    # Appeler la fonction principale
    SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
    print(f'Maximum tengential stress : {max(np.abs(SigTtot/1e6))}')
    print(f'Maximum radial stress : {max(np.abs(SigRtot/1e6))}')
    
    plot_option_circee = 'False'
    
    if plot_option_circee == 'True':
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