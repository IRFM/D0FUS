# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:31:26 2024

@author: TA276941
"""
#%% Import

from D0FUS_import import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% Panel control

# Parameterization
Supra_choice = 'HTS'                # 'Manual' , 'HTS' or 'LTS'
Chosen_Steel = '316L'               # '316L' , 'NH50' or 'Manual'
Radial_build_model = 'D0FUS'        # "academic" , "D0FUS" , "CIRCEE"
Choice_Buck_Wedg = 'Wedging'        # 'Wedging' or 'Bucking'
Option_Kappa = 'Wenninger'          # 'Stambaugh' , 'Freidberg' , 'Wenninger' or 'Manual'
L_H_Scaling_choice = 'New_Ip'       # 'Martin' , 'New_S', 'New_Ip'
Scaling_Law = 'IPB98(y,2)'          # 'IPB98(y,2)', 'ITPA20-IL', 'ITPA20', 'DS03', 'L-mode', 'L-mode OK', 'ITER89-P'

# Inputs
H = 1                               # H factor
Tbar  = 14                          # Mean Temperature keV
Temps_Plateau = 0                   # Plateau time
δ = 0.5                             # Triangularity
b = 1.2                             # BB + 1rst Wall + N shields + Gaps

#%% Constants initialisation

Choice_solving_CS_method = "brentq" # "brentq" or "manual" for debuging

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

# Density and Temperature parameters
nu_n  = 0.1     # Density profile parameter
nu_T  = 1       # Temperature profile parameter
C_Alpha = 5     # Helium dilution tuning parameter

# Flux
Ce = 0.45            # Ejima constants
Li = 0.85            # Internal inductance of the plasma -> To be replaced with proper model
ITERPI = 20          # ITER plasma induction current [Wb]

# TF
gamma_TF = 1/2       # fraction d'acier pouvant soutenir les efforts suivant l'axe considéré (r ou theta) = facteur de concentration de contrainte dépendant de la géométrie du CICC
coef_inboard_tension = 1/2 # Facteur de répartition de la tension entre jambe interne et externe
F_CClamp = 0e6       # C-Clamp limit in N , max order of magnitude from DDD : 30e6 N and of 60e6 N from [Bachmann (2023) FED]

# CS
Gap = 0.1            # Gap between wedging and bucking CS and TF [m]
gamma_CS = 1/2       # fraction d'acier pouvant soutenir les efforts suivant l'axe considéré (r ou theta) = facteur de concentration de contrainte dépendant de la géométrie du CICC

# Efficienty      
eta_T = 0.4          # Ratio between thermal and electrical power
eta_RF = 0.8 * 0.5   # fraction of klystron power absorbed by plasma * conversion efficiency from wall power to klystron

# PFU
theta_deg = 2.7      # Angle d'incidence sur les PFU pour calcul du flux de chaleur
# Source 1: T. R. Reiter, “Basic Fusion Boundary Plasma Physics,” ITER School Lecture Notes, Jan. 21 2019
# Source 2: “SOLPS-ITER simulations of the ITER divertor with improved plasma conditions,” Journal of Nuclear Materials (2024) )

# Current density scaling
T_helium = 4.2 #○ K temeprature de l'helium considéré
Marge_T_Helium = 0.3 # Puisque l'helium est à 10 barre, il est poussé à 0.3 K au dessus de la consigne


#%% Numerical initialisation

# Hide runtime-related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create a directory to save images if it doesn't exist
# save_directory = "Graphs"
save_directory = os.path.join(os.getcwd(), "Graphs", "Données brutes")

os.makedirs(save_directory, exist_ok=True)

#%%

print("D0FUS_parameterization loaded")