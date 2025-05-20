# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:31:26 2024

@author: TA276941
"""
#%% Import

from D0FUS_import import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% Definition of constants

# Parameterization
Choice_Buck_Wedg = 'Bucking'        # Wedging or Bucking
Choice_solving_CS_method = "brentq" # CS solving method : default = "brentq" or "manual" for debuging
Option_Kappa = 'Wenninger'          # Choose between Stambaugh, Freidberg , Wenninger or Manual
L_H_Scaling_choice = 'New_Ip'       # 'Martin' , 'New_S', 'New_Ip'
Radial_build_model = 'D0FUS'        # Choose between "academic" , "D0FUS" , "CIRCEE"
Supra_choice = 'HTS'             # Choose between 'Manual' , 'HTS' or 'LTS'

# Physical constants
E_ELEM  = 1.6e-19           # Electron charge [Coulomb]
M_E     = 9.1094E-31        # kg
M_I     = 2 * 1.6726E-27    # kg
μ0    = 4.*np.pi*1.E-7      # Henri/m
EPS_0   = 8.8542E-12        # Farad/m

# Fusion constants
E_ALPHA = 3.5 *1.E6*E_ELEM  # Joule
E_N     = 14.1*1.E6*E_ELEM  # Joule
E_F     = 22.4*1.E6*E_ELEM  # Joule (considering all N react with Li)
Atomic_mass  = 2.5          # average atomic mass in AMU

# Plasma stability
betaN = 2.8  # Beta Tryon limit
q = 2.5      # Security factor limit
H = 1        # H factor

# Density and Temperature parameters
Tbar_init  = 14 # Mean Temperature keV
nu_n  = 0.1     # Density profile parameter
nu_T  = 1       # Temperature profile parameter
C_Alpha = 5     # Helium dilution tuning parameter

# Flux consumptions assumtions
Ce = 0.45            # Ejima constants
Temps_Plateau = 0    # Plateau time
Li = 0.85            # Internal inductance of the plasma -> To be replaced with proper model
ITERPI = 20          # ITER plasma induction current [Wb]

# Geometric parameters
δ = 0.5  # Triangularity
b = 1.2  # BB+1rst Wall+N shields+ Gaps

# Engineer constraints TF
σ_TF = 660.E6        # Mechanical limit of the steel considered in [Pa]
F_CClamp = 0e6       # C-Clamp limit in N , max order of magnitude from DDD : 30e6 N and of 60e6 N from [Bachmann (2023) FED]
gamma_TF = 1/2       # fraction d'acier pouvant soutenir les efforts suivant l'axe considéré (r ou theta) = facteur de concentration de contrainte dépendant de la géométrie du CICC
if Choice_Buck_Wedg == "Wedging" :
    beta_TF = 1/2        # fraction de la tension aloué au WP (valeur fixe prise à partir de ITER: DDD TF p.97)
elif Choice_Buck_Wedg == "Bucking" :
    beta_TF = 1          # fraction de la tension aloué au WP , en bucking = 1
coef_inboard_tension = 1/2 # Facteur de répartition de la tension entre jambe interne et externe

# Engineer constraints CS
Gap = 0.1            # Gap between wedging and bucking CS and TF [m]
σ_CS = 660.E6        # CS machanical limit [Pa]
gamma_CS = 1/2       # fraction d'acier pouvant soutenir les efforts suivant l'axe considéré (r ou theta) = facteur de concentration de contrainte dépendant de la géométrie du CICC

# Engineer constraints
eta_RF = 0.5         # conversion efficiency from wall power to klystron
f_RP   = 0.8         # fraction of klystron power absorbed by plasma
eta_T = 0.4          # Ratio between thermal and electrical power
theta_deg = 2.7      # Angle d'incidence sur les PFU pour calcul du flux de chaleur
# Source 1: T. R. Reiter, “Basic Fusion Boundary Plasma Physics,” ITER School Lecture Notes, Jan. 21 2019
# Source 2: “SOLPS-ITER simulations of the ITER divertor with improved plasma conditions,” Journal of Nuclear Materials (2024)

#%% Current density scaling

T_helium = 4.2 #○ K temeprature de l'helium considéré
Marge_T_Helium = 0.3 # Puisque l'helium est à 10 barre, il est poussé à 0.3 K au dessus de la consigne

if Supra_choice == "LTS"  :

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
    
elif Supra_choice == "HTS"  :

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
    
elif Supra_choice == "Manual"  :
    
    J_max_CS_conducteur_manual = 50.E6       # A/m² from ITER values
    
    J_max_TF_conducteur_manual = 50.E6       # A/m² from ITER values
    
else :
    print("Please choose a proper superconductor")
    

#%% Benchmark

#%% Scaling Law

# Considering :
    # B the toroidal magnetic field on R0 (T)
    # R0 the geometrcial majopr radius (m)
    # Kappa the elongation
    # M or A  the average atomic mass (AMU)
    # Epsilon the inverse aspect ratio
    # n the density (10**19/m cube)
    # I plasma current (MA)
    # P the absorbed power (MW)
    # H an amplification factor = Taue/Taue_Hmode

# Definition des valeurs pour chaque loi
param_values = {
    'IPB98(y,2)': {
        'C_SL': 0.0562,
        'alpha_(1+delta)': 0,
        'alpha_M': 0.19,
        'alpha_kappa': 0.78,
        'alpha_epsilon': 0.58,
        'alpha_R': 1.97,
        'alpha_B': 0.15,
        'alpha_n': 0.41,
        'alpha_I': 0.93,
        'alpha_P': -0.69
    },
    'ITPA20-IL': {
        'C_SL': 0.067,
        'alpha_(1+delta)': 0.56,
        'alpha_M': 0.3,
        'alpha_kappa': 0.67,
        'alpha_epsilon': 0,
        'alpha_R': 1.19,
        'alpha_B': -0.13,
        'alpha_n': 0.147,
        'alpha_I': 1.29,
        'alpha_P': -0.644
    },
    'ITPA20': {
        'C_SL': 0.053,
        'alpha_(1+delta)': 0.36,
        'alpha_M': 0.2,
        'alpha_kappa': 0.8,
        'alpha_epsilon': 0.35,
        'alpha_R': 1.71,
        'alpha_B': 0.22,
        'alpha_n': 0.24,
        'alpha_I': 0.98,
        'alpha_P': -0.669
    },
    'DS03': {
        'C_SL': 0.028,
        'alpha_(1+delta)': 0,
        'alpha_M': 0.14,
        'alpha_kappa': 0.75,
        'alpha_epsilon': 0.3,
        'alpha_R': 2.11,
        'alpha_B': 0.07,
        'alpha_n': 0.49,
        'alpha_I': 0.83,
        'alpha_P': -0.55
    },
    'L-mode': {
        'C_SL': 0.023,
        'alpha_(1+delta)': 0,
        'alpha_M': 0.2,
        'alpha_kappa': 0.64,
        'alpha_epsilon': -0.06,
        'alpha_R': 1.83,
        'alpha_B': 0.03,
        'alpha_n': 0.4,
        'alpha_I': 0.96,
        'alpha_P': -0.73
    },
    'L-mode OK': {
        'C_SL': 0.023,
        'alpha_(1+delta)': 0,
        'alpha_M': 0.2,
        'alpha_kappa': 0.64,
        'alpha_epsilon': -0.06,
        'alpha_R': 1.78,
        'alpha_B': 0.03,
        'alpha_n': 0.4,
        'alpha_I': 0.96,
        'alpha_P': -0.73
    },
    'ITER89-P': {
        'C_SL': 0.048,
        'alpha_(1+delta)': 0,
        'alpha_M': 0.5,
        'alpha_kappa': 0.5,
        'alpha_epsilon': 0.3,
        'alpha_R': 1.2,
        'alpha_B': 0.2,
        'alpha_n': 0.08,
        'alpha_I': 0.85,
        'alpha_P': -0.5
    }
}

# Fonction pour recuperer les valeurs en fonction de la loi et du paramètre
def get_parameter_value_scaling_law(law):
    if law in param_values:
        C_SL = param_values[law]['C_SL']
        alpha_delta = param_values[law]['alpha_(1+delta)']
        alpha_M = param_values[law]['alpha_M']
        alpha_kappa = param_values[law]['alpha_kappa']
        alpha_epsilon = param_values[law]['alpha_epsilon']
        alpha_R = param_values[law]['alpha_R']
        alpha_B = param_values[law]['alpha_B']
        alpha_n = param_values[law]['alpha_n']
        alpha_I = param_values[law]['alpha_I']
        alpha_P = param_values[law]['alpha_P']
        return C_SL,alpha_delta,alpha_M,alpha_kappa,alpha_epsilon,alpha_R,alpha_B,alpha_n,alpha_I,alpha_P
    else:
        raise ValueError(f"La loi {law} n'existe pas.")

# Demander à l'utilisateur de choisir une sclaing law
# law = input("Entrez la scaling law à etudier : (IPB98(y,2),ITPA20-IL,ITPA20,DS03,L-mode,L-mode OK,ITER89-P) :")
law = 'IPB98(y,2)'

# Initialisation
(C_SL,alpha_delta,alpha_M,alpha_kappa,alpha_epsilon,alpha_R,alpha_B,alpha_n,alpha_I,alpha_P) = get_parameter_value_scaling_law(law)


#%% Numerical initialisation

# Hide runtime-related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create a directory to save images if it doesn't exist
# save_directory = "Graphs"
save_directory = "C:/Users/TA276941/Desktop/D0Fus/Graphs/Données brutes/"
os.makedirs(save_directory, exist_ok=True)

#%%

print("D0FUS_parameterization loaded")