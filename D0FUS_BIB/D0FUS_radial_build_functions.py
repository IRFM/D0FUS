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
    print("##################################################### J Model ##########################################################")

#%% Current density

def Jc_LTS(B, T):
    """
    Calcule la densité de courant critique Jc pour un fil Nb3Sn TFEU4 donnée
    en fonction du champ magnétique B (en T), de la température T (en K)
    et de la déformation Epsth.

    Arguments:
        B : scalaire ou array-like
            Champ magnétique en tesla.
        T : scalaire ou array-like
            Température en kelvin.
        Epsth : scalaire ou array-like
            Déformation à l'état thixotrope
        params : dict, optionnel
            Dictionnaire des paramètres si souhaité personnaliser.

    Retourne:
        Jc : array-like
            Densité de courant critique en A/m^2
    """
    # Convertir en array numpy
    B = np.array(B, dtype=float)
    T = np.array(T, dtype=float)

    # Charger les paramètres
    p = Nb3Sn_PARAMS.copy()

    Ca1 = p['Ca1']
    Ca2 = p['Ca2']
    Eps0a = p['Eps0a']
    Bc2m = p['Bc2m']
    Tcm = p['Tcm']
    C1 = p['C1']
    exponent_p = p['p']
    exponent_q = p['q']
    dbrin = p['dbrin']

    # Calcul intermédiaires
    Epssh = Ca2 * Eps0a / np.sqrt(Ca1**2 - Ca2**2)
    s2Eps = 1 + (1.0 / (1 - Ca1 * Eps0a)) * (
        Ca1 * (np.sqrt(Epssh**2 + Eps0a**2) - np.sqrt((Eps - Epssh)**2 + Eps0a**2))
        - Ca2 * Eps
    )
    Tc0 = Tcm * s2Eps**(1/3)
    # b0 inutilisé dans formule finale, mais calculable si besoin
    # b0 = Bc2m * s2Eps

    t = T / Tc0
    one_minus_t152 = (1 - t**1.52)
    b = B / (Bc2m * s2Eps * one_minus_t152)
    # Tc variable non utilisée dans Jc final mais peut être retournée
    # Tc = Tcm * s2Eps * one_minus_t152

    # Formule de Jc avant géométrie
    Jc_raw = (C1 / B) * s2Eps * one_minus_t152 * (1 - t**2) * b**exponent_p * (1 - b)**exponent_q

    # Division par surface du fil (section circulaire)
    section = np.pi * (dbrin**2) / 4
    Jc = Jc_raw / section

    return Jc

def Jc_HTS(B, T):
    """
    Calcul de la densité de courant critique Jc pour un ruban REBCO selon le scaling CERN2014.

    Entrées:
        B   : champ magnétique (T), scalaire ou array-like
        T   : température (K), scalaire ou array-like
        Tet : angle (rad), scalaire ou array-like (0 = mauvaise orientation)
        params : dict optionnel pour surcharger DEFAULT_PARAMS_REBCO

    Retourne:
        Jc : densité de courant critique en A/m^2
    """
    
    # Ruban width
    Ruban_width = 1e-4
    
    # Conversion en array numpy
    B = np.array(B, dtype=float)
    T = np.array(T, dtype=float)

    # Récupérer paramètres
    p = REBCO_PARAMS.copy()

    # Extraire
    Tc0 = p['Tc0']; n = p['n']
    Bi0c = p['Bi0c']; Alfc = p['Alfc']; pc = p['pc']; qc = p['qc']; gamc = p['gamc']
    Bi0ab = p['Bi0ab']; Alfab = p['Alfab']; pab = p['pab']; qab = p['qab']; gamab = p['gamab']
    n1 = p['n1']; n2 = p['n2']; a = p['a']
    Nu = p['Nu']; g0 = p['g0']; g1 = p['g1']; g2 = p['g2']; g3 = p['g3']
    Trebco = p['trebco']

    # Échelle réduite de température
    tred = T / Tc0

    # Champs d'échelle pour C et AB
    Bic = Bi0c * (1 - tred**n)
    Biab = Bi0ab * ((1 - tred**n1)**n2 + a * (1 - tred**n))

    # Champs réduits
    bredc = B / Bic
    bredab = B / Biab

    # Densité critique composante C
    Jcc = (Alfc / B) * bredc**pc * (1 - bredc)**qc * (1 - tred**n)**gamc
    # composante AB
    Jcab = (Alfab / B) * bredab**pab * (1 - bredab)**qab * ((1 - tred**n1)**n2 + a * (1 - tred**n))**gamab

    # Largeur angulaire g
    g = g0 + g1 * np.exp(-g2 * np.exp(g3 * T) * B)

    # Combinaison des deux composantes selon l'angle
    Jc = Jcc + (Jcab - Jcc) / (1 + (np.abs(Tet - np.pi/2) / g)**Nu)
    
    Jc = Jc * (Trebco / Ruban_width)

    return Jc

#%% Jc test

if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    
    # Nb3Sn TFEU4 parameters LTS
    Nb3Sn_PARAMS = {
        'Ca1': 44.48,
        'Ca2': 0.0,
        'Eps0a': 0.00256,
        'Epsm': -0.00049,
        'Bc2m': 32.97,
        'Tcm': 16.06,
        'C1': 19922.0,
        'p': 0.63,
        'q': 2.1,
        'dbrin': 0.82e-3,
        'CuNCu': 1.01,
    }
    
    # tipical Déformation
    Eps = -0.6 / 100
    Marge_T_LTS = 1 #K
    
    # Paramètres par défaut pour REBCO (CERN 2014)
    REBCO_PARAMS = {
        'trebco': 1.5e-6,   # épaisseur de la couche supraconductrice [m]
        'w': 4e-3,          # largeur du ruban [m]
        'Tc0': 93.0,        # température critique zéro champ [K]
        'n': 1.0,
        'Bi0c': 140.0,      # [T]
        'Alfc': 1.41e12,    # [A/m^2]
        'pc': 0.313,
        'qc': 0.867,
        'gamc': 3.09,
        'Bi0ab': 250.0,     # [T]
        'Alfab': 83.8e12,   # [A/m^2]
        'pab': 1.023,
        'qab': 4.45,
        'gamab': 4.73,
        'n1': 1.77,
        'n2': 4.1,
        'a': 0.1,
        'Nu': 0.857,
        'g0': -0.0056,
        'g1': 0.0944,
        'g2': -0.0008,
        'g3': 0.00388,
    }
    
    # HTS orientation (pessimistic one)
    Tet = 0
    Marge_T_HTS = 5 #K
    
    # —––––––––––––––––––––––––––––––––––––
    # 1) Plages et calculs
    B_vals = np.linspace(5, 30, 100)
    T_vals = np.linspace(2, 30, 100)
    B_mesh, T_mesh = np.meshgrid(B_vals, T_vals)
    
    # Calcul des densités critiques
    J_LTS = Jc_LTS(B_mesh, T_mesh + Marge_T_LTS + Marge_T_Helium)/1e6
    J_HTS = Jc_HTS(B_mesh, T_mesh + Marge_T_LTS + Marge_T_Helium)/1e6
    
    # Scan simple à T=4.2 K
    T0 = 4.2
    J_LTS_42 = Jc_LTS(B_vals, T0 + Marge_T_LTS + Marge_T_Helium)/1e6
    J_HTS_42 = Jc_HTS(B_vals, T0 + Marge_T_LTS + Marge_T_Helium)/1e6
    
    # —––––––––––––––––––––––––––––––––––––
    # 2) Plot 2D à 4.2 K
    plt.figure(figsize=(6,4))
    plt.plot(B_vals, J_LTS_42, label='LTS @ 4.2 K', lw=2)
    plt.plot(B_vals, J_HTS_42, label='HTS @ 4.2 K', lw=2)
    plt.xlabel('Champ magnétique B (T)')
    plt.ylabel('Jc (MA/m²)')
    plt.title('Comparaison Jc_HTS vs Jc_LTS à 4.2 K')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # —––––––––––––––––––––––––––––––––––––
    # 3) Surfaces 3D plus esthétiques
    def plot_surface(ax, B, T, J, title):
        surf = ax.plot_surface(
            B, T, J,
            cmap='viridis',        # colormap agréable
            edgecolor='none',      # pas de maillage
            antialiased=True,      # lissage
            rcount=100, ccount=100 # résolution
        )
        ax.set_xlabel('B (T)')
        ax.set_ylabel('T (K)')
        ax.set_zlabel('Jc (MA/m²)')
        ax.set_title(title)
        # colorbar intégrée
        plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    
    fig = plt.figure(figsize=(12,5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_surface(ax1, B_mesh, T_mesh, J_LTS, 'Jc_LTS(B, T)')
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_surface(ax2, B_mesh, T_mesh, J_HTS, 'Jc_HTS(B, T)')
    
    # Ajuster l'angle de vue
    for ax in (ax1, ax2):
        ax.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    plt.show()
    
    # Points de comparaison
    B_points = [12.0, 20.0]  # en tesla
    T0 = 4.2                 # en kelvin
    
    # Facteurs de correction
    factor_cu = 0.5          # division par 2 (cuivre)
    factor_cool = 0.7        # 1 - 0.30 (perte 30%) du au refroidissement
    insulation_factor = 0.75  # perte de 25% de surface pour l'isolation
    
    print(f"Comparaison Jc après cuivre & refroid. à T = {T0} K\n")
    
    for B0 in B_points:
        # Densités critiques brutes
        J_lts_raw = round(Jc_LTS(B0, T0 + Marge_T_LTS + Marge_T_Helium)/1e6)
        J_hts_raw = round(Jc_HTS(B0, T0 + Marge_T_HTS + Marge_T_Helium)/1e6)
    
        # Application des facteurs
        J_lts_eff = round(J_lts_raw * factor_cu * factor_cool * insulation_factor)
        J_hts_eff = round(J_hts_raw * factor_cu * factor_cool * insulation_factor)
    
        # Affichage
        print(f"——— À B = {B0:.1f} T ———")
        print(f"  Jc_LTS brin  = {J_lts_raw} MA/m²")
        print(f"  Jc_LTS effective = {J_lts_eff} MA/m²")
        print(f"  Jc_HTS ruban  = {J_hts_raw} MA/m²")
        print(f"  Jc_HTS effective = {J_hts_eff} MA/m²\n")

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
    
    # 1. Calcul du champ central B0 en fonction de la géométrie et du champ maximal
    B0 = f_B0(Bmax, a, b, R0)

    # 2. Rayon interne (jambe interne) et externe (jambe externe)
    R1_0 = R0 - a - b  # rayon intérieur brut avant ajout d'épaisseur
    R2_0 = R0 + a + b  # rayon extérieur brut avant ajout d'épaisseur

    # 3. Nombre de spires effectives NI nécessaires pour engendrer B0
    #    NI = 2π * R0 * B0 / μ0
    NI = 2 * np.pi * R0 * B0 / μ0

    # 4. Section de conducteur nécessaire pour supporter la densité de courant J_max_TF
    S_cond = NI / J_max_TF

    # 5. Épaisseur de la couche interne c1 déduite de la section circulaire
    #    S_cond = π * (R1_0^2 - (R1_0 - c1)^2) => résolution pour c1
    c_WP = R1_0 - np.sqrt(R1_0**2 - S_cond / np.pi)

    # 6. Calcul des nouveaux rayons après ajout de c1
    R1 = R1_0 - c_WP # rayon intérieur effectif
    R2 = R2_0 + c_WP  # rayon extérieur effectif

    # 7. Calcul du moment de torsion T sur la bobine
    if (R2 > 0) and (R1 > 0) and (R2 / R1 > 0):
        # Formule de calcul du moment T (intégrale des forces magnétiques)
        T = abs(((np.pi * B0**2 * R0**2) / μ0 * math.log(R2 / R1) - F_CClamp) * coef_inboard_tension)

    else:
        # Conditions géométriques invalides
        return np.nan, np.nan

    # 8. Pression radiale P due au champ magnétique Bmax
    P = Bmax**2 / (2 * μ0)

    # 9. Choix du mode mécanique : mise en contrainte par "bucking" ou par "wedging"
    if Choice_Buck_Wedg == "Bucking":
        # Épaisseur c2 pour le bucking, valable si R1 >> c
        c_Nose = (B0**2 * R0**2) * math.log(R2 / R1) / (2 * μ0 * 2 * R1 * (σ_TF - P))
        σ_r = P                            # contrainte radiale
        σ_z = T / (2 * np.pi * R1 * c_Nose)   # contrainte axiale
        ratio_tension = σ_z / (σ_z + σ_r)

    elif Choice_Buck_Wedg == "Wedging":
        # Épaisseur c2 pour le wedging, valable si R1 >> c
        c_Nose = (B0**2 * R0**2) / (2 * μ0 * R1 * σ_TF) * (1 + math.log(R2 / R1) / 2)
        σ_theta = P * R1 / c_Nose             # contrainte circonférentielle
        σ_z = T / (2 * np.pi * R1 * c_Nose)   # contrainte axiale
        ratio_tension = σ_z / (σ_theta + σ_z)

    else:
        raise ValueError("Choisir 'Bucking' ou 'Wedging' comme option mécanique.")

    # 10. Épaisseur totale c (somme des deux couches)
    c = c_WP + c_Nose
    
    # Vérification que c_WP est valide
    if c is None or np.isnan(c) or c < 0 or c > (c_WP + c_Nose) or c > R0 - a - b:
        return(np.nan, np.nan)

    return c, ratio_tension

    
#%% Dilution and Thick cylinder model

def Winding_Pack_D0FUS(R_0, a, b, sigma_max, J_max, mu_0, B_max, gamma, beta):
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
    
    # Numerator calculation
    term1 = (2 * sigma_max * B_max * R_e) / (mu_0 * J_max)  
    term2 = (beta * B_max**2 * R_e**2 * ln_term * coef_inboard_tension) / ( mu_0)     
    numerator = term1 + term2
    
    # Denominator calculation
    denominator = sigma_max - (B_max**2 / (2 * mu_0 * gamma))
    
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
        return (np.nan,np.nan)

    log_term = np.log((R_0 + a + b) / (R_0 - a - b))
    P = B_max**2 / (2 * mu_0)
    
    alpha = (2 * B_max * R_e) / (mu_0 * J_max * D)
    sigma_r = P
    sigma_z = (B_max**2 * R_e**2 * log_term * coef_inboard_tension) / (mu_0 * D)
    
    ratio_tension = (sigma_z * beta) / ((sigma_r / gamma) + (sigma_z * beta))
    
    return (R_e - R_i, ratio_tension)


def Nose_D0FUS(R_ext_Nose, sigma_max, beta, B_max, mu_0, R_0, a, b):
    """
    Compute the internal radius Ri based on analytical expressions.

    Parameters:
    - R_ext_Nose : float, external radius at the nose (R_ext^Nose)
    - sigma_max  : float, maximum admissible stress
    - beta       : float, dimensionless coefficient (0 ≤ beta ≤ 1)
    - B_max      : float, maximum magnetic field
    - mu_0       : float, magnetic permeability of vacuum
    - R_0, a, b  : floats, geometric parameters

    Returns:
    - Ri : float, internal radius
    """
    
    # Calculate P_Nose
    P = (B_max**2) / (2 * mu_0) * (R_0 - a - b) / R_ext_Nose
    
    # Compute the logarithmic term
    log_term = np.log((R_0 + a + b) / (R_0 - a - b))
    
    # Compute the full expression under the square root
    term = (R_ext_Nose**2 / sigma_max) * (2 * P + (1 - beta) * (B_max**2 * coef_inboard_tension / mu_0) * log_term)
    
    # Compute Ri
    Ri_squared = R_ext_Nose**2 - term

    if Ri_squared < 0:
        # raise ValueError("Negative value under square root. Check your input parameters.")
        return(np.nan)
    
    Ri = np.sqrt(Ri_squared)
    return Ri

def f_TF_D0FUS(a, b, R0, σ_TF, μ0, J_max_TF, Bmax, Choice_Buck_Wedg, gamma, beta):
    
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
        
        (c_WP, ratio_tension) = Winding_Pack_D0FUS( R0, a, b, σ_TF, J_max_TF, μ0, Bmax, gamma, beta)
        #print(c_WP)
        
        # Vérification que c_WP est valide
        if c_WP is None or np.isnan(c_WP) or c_WP < 0:
            return(np.nan, np.nan)
        c_Nose = R0 - a - b - c_WP - Nose_D0FUS(R0 - a - b - c_WP, σ_TF, beta, Bmax, μ0, R0, a, b)
        #print(c_Nose)
        # Vérification que c_Nose est valide
        if c_Nose is None or np.isnan(c_Nose) or c_Nose < 0:
            return(np.nan, np.nan)
        
        # Vérification que la somme ne dépasse pas R0 - a - b
        if (c_WP + c_Nose) > (R0 - a - b):
            return(np.nan, np.nan)
        
        # Epaisseur totale de la bobine
        c  = c_WP + c_Nose
    
    elif Choice_Buck_Wedg == "Bucking":
        
        (c, ratio_tension) = Winding_Pack_D0FUS(R0, a, b, σ_TF, J_max_TF, μ0, Bmax, gamma, beta)
        
        # Vérification que c_WP est valide
        if c is None or np.isnan(c) or c < 0 or c > R0 - a - b :
            return(np.nan, np.nan)
        
    else : 
        print( "Choose a valid mechanical configuration" )
        return(np.nan, np.nan)

    return(c, ratio_tension)

#%% TF benchmark

if __name__ == "__main__":
    
    # ITER test parameters
    a_TF = 2
    b_TF = 1.45
    R0_TF = 6.2
    σ_TF_tf = 660e6
    μ0_TF = 4 * np.pi * 1e-7
    J_max_TF_tf = 35 * 1e6
    Bmax_TF = 11.8
    
    # Test the function
    result_academic_wedging = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging")
    print(f"TF Academic wedging : {result_academic_wedging}")
    result_academic_bucking = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking")
    print(f"TF Academic bucking : {result_academic_bucking}")
    
    result_complex_wedging = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging", gamma_TF, 1/2)
    print(f"TF D0FUS wedging : {result_complex_wedging}")
    result_complex_bucking = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF_tf, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking", gamma_TF, 1)
    print(f"TF D0FUS bucking  : {result_complex_bucking}")
    
    plot_option_TF_sigma = 0
    plot_option_TF_B = 1
    
    if plot_option_TF_sigma == 1:
    
        # Plage de σ_TF (10 MPa à 1500 MPa)
        sigma_values = np.linspace(200e6, 1500e6, 100)  # 100 points
        
        # Initialisation des listes de résultats
        academic_w = []
        academic_b = []
        d0fus_w = []
        d0fus_b = []
        
        for σ_TF in sigma_values:
            # Modèles académiques
            res_acad_w = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging")
            res_acad_b = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking")
            
            # Modèles D0FUS (avec gamma_TF et paramètres supplémentaires)
            res_d0fus_w = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging", gamma_TF, 0.5)
            res_d0fus_b = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking", gamma_TF, 1)
            
            # Stockage des résultats (en supposant que ce sont des tuples/arrays)
            academic_w.append(res_acad_w[0])  # Prend le premier élément du résultat
            academic_b.append(res_acad_b[0])
            d0fus_w.append(res_d0fus_w[0])
            d0fus_b.append(res_d0fus_b[0])
        
        # Création du graphique
        plt.figure(figsize=(12, 7))
        plt.plot(sigma_values/1e6, academic_w, 'b--', label='Academic Wedging', linewidth=2)
        plt.plot(sigma_values/1e6, academic_b, 'r--', label='Academic Bucking', linewidth=2)
        plt.plot(sigma_values/1e6, d0fus_w, 'b-', label='D0FUS Wedging', linewidth=2)
        plt.plot(sigma_values/1e6, d0fus_b, 'r-', label='D0FUS Bucking', linewidth=2)
        
        # Mise en forme
        plt.xlabel('Contrainte σ_TF (MPa)', fontsize=12)
        plt.ylabel('Epaisseur TF [m]', fontsize=12)
        plt.title('Scan de σ_TF - Comparaison des modèles', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Affichage
        plt.tight_layout()
        plt.show()
    
    if plot_option_TF_B == 1:
        
        # Plage de Bmax_TF (5 T à 25 T)
        Bmax_values = np.linspace(0, 25, 100)  # 100 points
        
        # Initialisation des listes de résultats
        academic_w = []
        academic_b = []
        d0fus_w = []
        d0fus_b = []
        d0fus_w_gamma1 = []  # Nouveaux résultats pour gamma=1 (Wedging)
        d0fus_b_gamma1 = []  # Nouveaux résultats pour gamma=1 (Bucking)
        
        for Bmax_TF in Bmax_values:
            # Modèles académiques
            res_acad_w = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging")
            res_acad_b = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking")
            
            # Modèles D0FUS avec gamma_TF = 0.5 (Wedging) et 1 (Bucking) - version originale
            res_d0fus_w = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging", gamma_TF, 0.5)
            res_d0fus_b = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking", gamma_TF, 1)
            
            # Nouveaux modèles D0FUS avec gamma_TF = 1 pour les deux modes
            res_d0fus_w_gamma1 = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Wedging", 1, 0.5)
            res_d0fus_b_gamma1 = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF, μ0_TF, J_max_TF_tf, Bmax_TF, "Bucking", 1, 1)
            
            # Stockage des résultats
            academic_w.append(res_acad_w[0])
            academic_b.append(res_acad_b[0])
            d0fus_w.append(res_d0fus_w[0])
            d0fus_b.append(res_d0fus_b[0])
            d0fus_w_gamma1.append(res_d0fus_w_gamma1[0])
            d0fus_b_gamma1.append(res_d0fus_b_gamma1[0])
        
        # Création du graphique
        plt.figure(figsize=(12, 7))
        # Courbes originales
        plt.plot(Bmax_values, academic_w, 'b--', label='Academic Wedging', linewidth=2)
        plt.plot(Bmax_values, academic_b, 'r--', label='Academic Bucking', linewidth=2)
        plt.plot(Bmax_values, d0fus_w, 'b-', label='D0FUS Wedging', linewidth=2)
        plt.plot(Bmax_values, d0fus_b, 'r-', label='D0FUS Bucking', linewidth=2)
        # Nouvelles courbes avec gamma=1
        plt.plot(Bmax_values, d0fus_w_gamma1, 'b*', label='D0FUS Wedging (γ=1)', linewidth=3)
        plt.plot(Bmax_values, d0fus_b_gamma1, 'r*', label='D0FUS Bucking (γ=1)', linewidth=3)
        
        # Mise en forme
        plt.xlabel('Champ magnétique maximum Bmax_TF (T)', fontsize=12)
        plt.ylabel('Epaisseur TF [m]', fontsize=12)
        plt.title('Scan de Bmax_TF - Comparaison des modèles', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Affichage
        plt.tight_layout()
        plt.show()
        
    
#%% Cylindre épais + dilution


#%% Print
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% CS thin layer model

def f_CS_academic(a, b, c, R0, κ, Bmax, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib):
    
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
    
    # Toroidal magnetic field
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
    
    ri_c = np.cbrt(RCS_ext**3 - (( 3 * abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)) / (2 * np.pi * μ0 * J_max_CS)))
    B_CS = μ0 * (J_max_CS) * (RCS_ext - ri_c)
    
    # Results filtering
    if B_CS > Bmax or B_CS < 0:
        return (np.nan, np.nan, np.nan)
    
    # Test CS
    
    # def compute_constrained_CS(RCS_ext, ΨPI, ΨRampUp, Ψplateau, ΨPF, μ0, Bmax, J_max_CS_guess):
    #     psi_total = abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)
    
    #     def ri_c(J):
    #         denom = 2 * np.pi * μ0 * J
    #         volume_term = RCS_ext**3 - (3 * psi_total) / denom
    #         if volume_term <= 0:
    #             return np.nan  # Rayon non physique
    #         return np.cbrt(volume_term)
    
    #     def B_CS(J):
    #         ri = ri_c(J)
    #         if np.isnan(ri) or ri >= RCS_ext:
    #             return np.nan  # Champ non physique
    #         return μ0 * J * (RCS_ext - ri)
    
    #     def objective(J):
    #         B = B_CS(J)
    #         return np.nan if np.isnan(B) else B - Bmax
    
    #     # Vérifie si la valeur initiale est acceptable
    #     B_init = B_CS(J_max_CS_guess)
    #     if not np.isnan(B_init) and B_init <= Bmax:
    #         return ri_c(J_max_CS_guess), J_max_CS_guess, B_init
    
    #     # Recherche par dichotomie entre J_min et J_max
    #     try:
    #         J_min = 1e5   # valeur basse raisonnable
    #         J_max = J_max_CS_guess
    #         J_solution = brentq(objective, J_min, J_max, xtol=1e-6)
    #         ri = ri_c(J_solution)
    #         B = B_CS(J_solution)
    #         return ri, J_solution, B
    #     except (ValueError, RuntimeError):
    #         return np.nan, np.nan, np.nan

    # # Fonction de calcul
    # def ri_c(J):
    #     denom = 2 * np.pi * μ0 * J
    #     volume_term = RCS_ext**3 - (3 * psi_total) / denom
    #     if volume_term <= 0:
    #         return np.nan
    #     return np.cbrt(volume_term)
    
    # def B_CS(J):
    #     ri = ri_c(J)
    #     if np.isnan(ri) or ri >= RCS_ext:
    #         return np.nan
    #     return μ0 * J * (RCS_ext - ri)
    
    # # Plage de J_max_CS
    # psi_total = abs(ΨPI + ΨRampUp + Ψplateau - ΨPF)
    # J_values = np.logspace(5, 9, 500)  # de 1e5 à 1e9 A/m²
    # ri_values = []
    # B_values = []
    
    # for J in J_values:
    #     ri_values.append(ri_c(J))
    #     B_values.append(B_CS(J))
    
    # # Conversion en array
    # ri_values = np.array(ri_values)
    # B_values = np.array(B_values)
    
    # # Plot
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # # B_CS plot
    # ax1.set_xlabel('J_max_CS (A/m²)', fontsize=12)
    # ax1.set_ylabel('B_CS (T)', color='tab:red', fontsize=12)
    # ax1.plot(J_values, B_values, 'r-', label='B_CS(J)')
    # ax1.axhline(Bmax, color='tab:red', linestyle='--', label='Bmax')
    # ax1.set_xscale('log')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # # ri plot on second axis
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('ri (m)', color='tab:blue', fontsize=12)
    # ax2.plot(J_values, ri_values, 'b-', label='ri(J)')
    # ax2.axhline(RCS_ext, color='tab:blue', linestyle='--', label='R_ext')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # # Lignes de légende combinées
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=10)
    
    # plt.title("Évolution de B_CS et ri en fonction de J_max_CS", fontsize=14)
    # plt.grid(True, which='both', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()
        
    # ri_c, J_max_CS, B_CS = compute_constrained_CS(RCS_ext, ΨPI, ΨRampUp, Ψplateau, ΨPF, μ0, Bmax, J_max_CS)
    # # print(ri_c,J_max_CS,B_CS)
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
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / RCS_ext
    
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = np.nanmax( [P_TF , abs(P_CS - P_TF) ] ) * ri_c / d_SS
            
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = P_CS * ri_c / d_SS
            
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
        
        val = Sigma_CS - σ_CS
        # passage en log afin de lisser les sauts abruptes qui peuvent arriver et faciliter la recherche de racines
        return np.sign(val) * np.log1p(abs(val))
    
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
    def find_d_SS_solution_brentq(a, b, n=100):
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
    
    def CS_filters(d_SS_solution):
        
        tol=1e-6
        
        # Preliminary calculations
        RCS_int_solution = ri_c - d_SS_solution
        alpha = np.pi*(RCS_ext**2 - ri_c**2) / ( np.pi*(RCS_ext**2 - ri_c**2) + np.pi*(ri_c**2**2 - RCS_int_solution**2))
    
        # Apply robust constraints with tolerances
        if d_SS_solution < tol or d_SS_solution > ri_c - tol:
            return False
        if RCS_int_solution < tol:
            return False
        if alpha < tol or alpha > 1 - tol:
            return False
        
        return True
    
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
            initial_guesses = np.linspace(0, ri_c, 10)  # 10 points entre 1e-3 et ri_c
            # Trouver la solution avec fsolve
            d_SS_solution = find_d_SS_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "brentq":
        
            # Trouver la solution avec brentq
            d_SS_solutions = find_d_SS_solution_brentq(0, ri_c)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            
            d_SS_solutions = find_d_SS_solution_root(0, ri_c)
            # print(d_SS_solution)
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "manual":

            d_SS_solutions = plot_function_CS(CS_to_solve, [0, ri_c])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            
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
        
        tol=1e-6
        
        # Preliminary calculations
        RCS_int_solution = ri_c - d_SS_solution
        alpha = np.pi*(RCS_ext**2 - ri_c**2) / ( np.pi*(RCS_ext**2 - ri_c**2) + np.pi*(ri_c**2**2 - RCS_int_solution**2))
    
        # Apply robust constraints with tolerances
        if d_SS_solution < tol or d_SS_solution > ri_c - tol:
            return (np.nan, np.nan, np.nan)
        elif alpha <  tol or alpha > 1 - tol:
            return (np.nan, np.nan, np.nan)
        elif RCS_int_solution < tol :
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, alpha, B_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan)

    
#%% Thcik cylinder and dilution model

def f_CS_D0FUS(a, b, c, R0,κ , Bmax, σ_CS, μ0, J_max_CS, Choice_Buck_Wedg, Tbar, nbar, Ip, Ib, gamma):
    
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
    
    Always pay attention : the calculations are not exact, so for filtering, 
    and manipulation of the solutions, need to take a margin ~ 1e-4
    (has been the cause of several bugs especially in the research of the best solution)
    
    """
#----------------------------------------------------------------------------------------------------------------------------------------    
    
    #### Preliminary results ####
    
    # Convert currents from MA to A
    Ip = Ip * 1e6
    Ib = Ib * 1e6

    # Calculate the central magnetic field
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
    
    # print(f'flux[Wb] : {ΨPI + ΨRampUp + Ψplateau - ΨPF}')
    
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
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - (RCS_int**2/RCS_ext)) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = (μ0 * J_max_CS * alpha) / 2 * (RCS_ext - (RCS_int**2/RCS_ext))
        
        # magnetic pressure to support from the CS
        P_CS = (μ0 * J_max_CS**2 * alpha**2 / 8) * (RCS_ext - (RCS_int**2/RCS_ext))**2
        # magnetic pressure to support from the TF
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / (RCS_ext)
    
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_light = ((P_CS-P_TF) * (RCS_ext**2 + RCS_int**2)) / (RCS_ext**2 - RCS_int**2)
            Sigma_strong = (P_TF * RCS_ext**2) / (RCS_ext**2 - RCS_int**2)
            Sigma_CS =  np.nanmax( [abs(Sigma_light ), abs(Sigma_strong) ] ) / ((1 - alpha) * gamma)
            
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = (1 / ((1 - alpha) * gamma)) * (P_CS * ((RCS_ext**2 + RCS_int**2) / (RCS_ext**2 - RCS_int**2)))
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
            
        val = Sigma_CS - σ_CS
        # passage en log afin de lisser les sauts abruptes qui peuvent arriver et faciliter la recherche de racines
        return np.sign(val) * np.log1p(abs(val))
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        
        plot_option_function = 0
        
        if plot_option_function ==1:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', label='CS_to_solve(d)')
            plt.axhline(y=0, color='r', linestyle='--', label='y=0')
            plt.grid(True)
            plt.xlabel('d')
            plt.ylabel('Function Value')
            plt.ylim(-1e10,1e10)
            plt.title('Comportement de CS_to_solve')
            plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        return zero_crossings
    
    # Détection des changements de signe
    def find_sign_changes(f, a, b, n):
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
    def find_d_solution_brentq(a, b, n = 1000):
        """
        Trouve les passages par zéro de CS_to_solve entre a et b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
    
        return roots
    
    # Vérification que la solution est bien un passage par zéro
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-4):
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
    def find_d_solution_fsolve(initial_guesses):
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
    
    def find_d_solution_root(a, b):
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

    def CS_filters(d_solution):
        
        tol=1e-6
        
        # Preliminary calculations
        RCS_int = RCS_ext - d_solution
        # Dilution coefficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - (RCS_int**2/RCS_ext)) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = (μ0 * J_max_CS * alpha) / 2 * (RCS_ext - (RCS_int**2/RCS_ext))
        # Apply robust constraints with tolerances
        if d_solution < tol or d_solution > RCS_ext - tol:
            return False
        if B_CS < tol or B_CS > Bmax - tol:
            return False
        if alpha <  tol or alpha > 1 - tol:
            return False
        
        return True
    
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
            initial_guesses = np.linspace(0, RCS_ext, 10)  # 10 points entre 1e-3 et ri_c
            # Trouver la solution avec fsolve
            d_solutions = find_d_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "brentq":
            # Trouver la solution avec brentq
            d_solutions = find_d_solution_brentq(0, RCS_ext)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            
            d_solutions = find_d_solution_root(0, RCS_ext)
            # print(d_SS_solution)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "manual":

            d_solutions = plot_function_CS(CS_to_solve, [0, RCS_ext])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        else:
            print("Choisissez une méthode valide pour le CS")
            return (np.nan, np.nan, np.nan, np.nan)
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results ####
        tol=1e-6
        # Preliminary calculations
        RCS_int = RCS_ext - d_solution
        # Dilution coefficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - (RCS_int**2/RCS_ext)) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = (μ0 * J_max_CS * alpha) / 2 * (RCS_ext - (RCS_int**2/RCS_ext))
    
        # Apply robust constraints with tolerances
        if d_solution < tol or d_solution > RCS_ext - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        if B_CS < tol or B_CS > Bmax - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        if alpha <  tol or alpha > 1 - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        
        return (d_solution, alpha, B_CS, J_max_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan, np.nan)
    
#%% CS Test

if __name__ == "__main__":
    
    # Test parameters Centering (ITER)
    a_cs = 2
    b_cs = 1.45
    c_cs = 0.9
    R0_cs = 6.2
    Bmax_cs = 11.8
    σ_CS_cs = 660e6
    μ0_cs = 4 * np.pi * 1e-7
    J_max_CS_cs = 50e6
    Tbar_cs = 14
    nbar_cs = 1
    Ip_cs = 12
    Ib_cs = 8
    κ_CS = 1.7
    
    # Test parameters Centering (diagnostic)
    a_cs = 2.5
    b_cs = 1.2
    c_cs = 0.6
    R0_cs = 6
    Bmax_cs = 12
    σ_CS_cs = 660e6
    μ0_cs = 4 * np.pi * 1e-7
    J_max_CS_cs = 50e6
    Tbar_cs = 14
    nbar_cs = 0.8
    Ip_cs = 21.8
    Ib_cs = 12.8
    κ_CS = 2.1

    result_CS1 = f_CS_academic(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    print(f"CS academic model Wedging : {result_CS1}")
    result_CS2 = f_CS_academic(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    print(f"CS academic model Bucking : {result_CS2}")
    
    result_CS3 = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
    print(f"CS D0FUS model Wedging : {result_CS3}")
    result_CS4 = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
    print(f"CS D0FUS model Bucking : {result_CS4}")
    
    Numerical_alpha = result_CS4[1]

    plot_J = 0 
    plot_sigma = 0 
    plot_CICC = 0
    plot_option_CS_B = 0
    
    if plot_CICC == 1:

        # Paramètres du conducteur CICC
        S_tot = 50e-3 * 50e-3  # Surface totale du conducteur (50 mm x 50 mm)
        S_cable = Numerical_alpha * S_tot  # Surface du câble central
        r_cable = np.sqrt(S_cable / np.pi)  # Rayon du câble central
        
        # Tracé du conducteur CICC
        fig, ax = plt.subplots(figsize=(6, 6))
        square = plt.Rectangle((-25e-3, -25e-3), 50e-3, 50e-3, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(square)
        circle = plt.Circle((0, 0), r_cable, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-30e-3, 30e-3)
        ax.set_ylim(-30e-3, 30e-3)
        ax.set_aspect('equal')
        ax.set_xlabel("Largeur (mm)")
        ax.set_ylabel("Hauteur (mm)")
        ax.set_title(f"Conducteur CICC typique")
        plt.grid()
        plt.show()
        
    elif plot_sigma == 1:
    
        # Plage de σ_CS (10 MPa à 1500 MPa)
        sigma_values = np.linspace(200e6, 1500e6, 100)  # 100 points linéairement espacés
        
        # Initialisation des listes pour stocker les résultats
        academic_wedging = []
        academic_bucking = []
        d0fus_wedging = []
        d0fus_bucking = []
        
        for σ_CS in sigma_values:
            # Calcul des résultats
            result_CS1 = f_CS_academic(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
            result_CS2 = f_CS_academic(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
            result_CS3 = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
            result_CS4 = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
            
            # Stockage de result_CS[0]
            academic_wedging.append(result_CS1[0])
            academic_bucking.append(result_CS2[0])
            d0fus_wedging.append(result_CS3[0])
            d0fus_bucking.append(result_CS4[0])
        
        # Tracé des courbes
        # plt.figure(figsize=(12, 7))
        # plt.plot(sigma_values / 1e6, academic_wedging, label='Academic Wedging', 'b--', linewidth=2)
        # plt.plot(sigma_values / 1e6, academic_bucking, label='Academic Bucking', 'r--', linewidth=2)
        # plt.plot(sigma_values / 1e6, d0fus_wedging, label='D0FUS Wedging', 'b-', linewidth=2)
        # plt.plot(sigma_values / 1e6, d0fus_bucking, label='D0FUS Bucking', 'r-', linewidth=2)
        
        plt.figure(figsize=(12, 7))
        plt.plot(sigma_values / 1e6, academic_wedging,'b--', label='Academic Wedging', linewidth=2)
        plt.plot(sigma_values / 1e6, academic_bucking,'r--', label='Academic Bucking', linewidth=2)
        plt.plot(sigma_values / 1e6, d0fus_wedging,'b-', label='D0FUS Wedging', linewidth=2)
        plt.plot(sigma_values / 1e6, d0fus_bucking,'r-', label='D0FUS Bucking', linewidth=2)
        
        # Mise en forme du graphique
        plt.xlabel('σ_CS (MPa)', fontsize=12)
        plt.ylabel('Epaisseur du CS [m]', fontsize=12)
        plt.title('Scan de σ_CS - Comparaison des modèles', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Affichage
        plt.show()
    
    elif plot_J == 1:
    
        # Plage de σ_CS (10 MPa à 1500 MPa)
        J_values = np.linspace(10e6, 200e6, 100)  # 100 points linéairement espacés
        
        # Initialisation des listes pour stocker les résultats
        academic_wedging = []
        academic_bucking = []
        d0fus_wedging = []
        d0fus_bucking = []
        
        for J_max_CS_cs in J_values:
            # Calcul des résultats
            result_CS1 = f_CS_academic(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
            result_CS2 = f_CS_academic(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
            result_CS3 = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
            result_CS4 = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs,κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
            
            # Stockage de result_CS[0]
            academic_wedging.append(result_CS1[0])
            academic_bucking.append(result_CS2[0])
            d0fus_wedging.append(result_CS3[0])
            d0fus_bucking.append(result_CS4[0])
        
        # Tracé des courbes
        plt.figure(figsize=(12, 7))
        plt.plot(J_values / 1e6, academic_wedging,'b--', label='Academic Wedging', linewidth=2)
        plt.plot(J_values / 1e6, academic_bucking,'r--', label='Academic Bucking', linewidth=2)
        plt.plot(J_values / 1e6, d0fus_wedging,'b-', label='D0FUS Wedging', linewidth=2)
        plt.plot(J_values / 1e6, d0fus_bucking,'r-', label='D0FUS Bucking', linewidth=2)
        
        # Mise en forme du graphique
        plt.xlabel('J (MA)', fontsize=12)
        plt.ylabel('Epaisseur du CS [m]', fontsize=12)
        plt.title('Scan de J - Comparaison des modèles', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Affichage
        plt.show()
        
    elif plot_option_CS_B == 1:
        # Plage de Bmax_cs (par exemple de 0 à 25 T)
        Bmax_values = np.linspace(0, 25, 100)  # 100 points
        
        # Initialisation des listes de résultats
        academic_cs_w = []
        academic_cs_b = []
        d0fus_cs_w = []
        d0fus_cs_b = []
    
        for Bmax_cs in Bmax_values:
            # Modèle académique (Wedging & Bucking)
            res_cs_acad_w = f_CS_academic(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
            res_cs_acad_b = f_CS_academic(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs)
    
            # Modèle D0FUS (Wedging & Bucking)
            res_cs_d0fus_w = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Wedging', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
            res_cs_d0fus_b = f_CS_D0FUS(a_cs, b_cs, c_cs, R0_cs, κ_CS, Bmax_cs, σ_CS_cs, μ0_cs, J_max_CS_cs, 'Bucking', Tbar_cs, nbar_cs, Ip_cs, Ib_cs, gamma_CS)
    
            # Stockage des résultats
            academic_cs_w.append(res_cs_acad_w[0])
            academic_cs_b.append(res_cs_acad_b[0])
            d0fus_cs_w.append(res_cs_d0fus_w[0])
            d0fus_cs_b.append(res_cs_d0fus_b[0])
        
        # Création du graphique
        plt.figure(figsize=(12, 7))
        plt.plot(Bmax_values, academic_cs_w, 'r--', label='Academic CS Wedging', linewidth=2)
        plt.plot(Bmax_values, academic_cs_b, 'b--', label='Academic CS Bucking', linewidth=2)
        plt.plot(Bmax_values, d0fus_cs_w, 'r-', label='D0FUS CS Wedging', linewidth=2)
        plt.plot(Bmax_values, d0fus_cs_b, 'b-', label='D0FUS CS Bucking', linewidth=2)
    
        # Mise en forme
        plt.xlabel('Champ magnétique maximum Bmax_CS (T)', fontsize=12)
        plt.ylabel('Épaisseur CS [m]', fontsize=12)
        plt.title('Scan de Bmax_CS - Comparaison des modèles CS', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

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