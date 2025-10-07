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
Operation_mode = 'Pulsed'     # 'Steady-State' or 'Pulsed'
Supra_choice = 'Rebco'              # 'Nb3Sn' , 'Rebco', 'NbTi' or Manual'
Chosen_Steel = '316L'               # '316L' , 'N50H' or 'Manual'
Radial_build_model = 'D0FUS'        # "academic" , "D0FUS" , "CIRCEE"
Choice_Buck_Wedg = 'Bucking'        # 'Wedging' or 'Bucking'
Option_Kappa = 'Wenninger'          # 'Stambaugh' , 'Freidberg' , 'Wenninger' or 'Manual'
L_H_Scaling_choice = 'New_Ip'       # 'Martin' , 'New_S', 'New_Ip'
Scaling_Law = 'IPB98(y,2)'          # 'IPB98(y,2)', 'ITPA20-IL', 'ITPA20', 'DS03', 'L-mode', 'L-mode OK', 'ITER89-P'
Bootstrap_choice = 'Segal'          # 'Freidberg' or 'Segal'

# Inputs
H = 1.1                               # H factor
Tbar  = 17.7                        # Mean Temperature keV
b = 1.2                             # BB + 1rst Wall + N shields + Gaps

# If Pulsed operation:
if Operation_mode == 'Pulsed' :
    Temps_Plateau_input = 20 * 60       # Plateau time [s]
    P_aux_input = 140                    # P_aux
else :
    Temps_Plateau_input = 0            # Plateau time [min]
    P_aux_input = 0                    # P_aux

#%% Constants initialisation

# If Pulsed operation:
if Operation_mode == 'Pulsed' :
    fatigue = 2                         # Fatigue parameter for the steel 
else :
    fatigue = 1                         # Fatigue parameter for the steel 

Choice_solving_CS_method = "brentq" # "brentq" or "manual" for debuging
Choice_solving_TF_method = "brentq" # "brentq" or "manual" for debuging

# Physical constants
E_ELEM  = 1.6e-19           # Electron charge [Coulomb]
M_E     = 9.1094E-31        # kg
M_I     = 2 * 1.6726E-27    # kg
μ0      = 4.*np.pi*1.E-7    # Henri/m
EPS_0   = 8.8542E-12        # Farad/m

# Fusion constants
E_ALPHA = 3.5 *1.E6*E_ELEM  # Joule
E_N     = 14.1*1.E6*E_ELEM  # Joule
E_F     = 22.4*1.E6*E_ELEM  # Joule (considering all N react with Li)
Atomic_mass  = 2.5          # average atomic mass in AMU
Zeff = 1                    # Zeff du plasma (default :1)
r_synch = 0.5               # Synchrotron reflection coefficient of the wall

# Plasma stability
betaN = 2.8  # Beta Tryon limit
q = 2.5      # Safety factor limit

# Radiative Power
beta_T = 2 # Beta_T taken from [J.Johner Helios]

# Elongation
κ_manual = 1.7    # Elongation if chosen to be manual

# Steel
σ_manual = 1500         # MPa
nu_Steel = 0.29         # Poisson's ratio [CIRCEE model]
Young_modul_Steel = 200e9   # Young modul steel [CIRCEE model]
Young_modul_Glass_Fiber = 90e9  # Young modul glass fiber [CIRCEE model]
# S glass founded in
# https://www.engineeringtoolbox.com/polymer-composite-fibers-d_1226.html

# Density and Temperature parameters
nu_n  = 0.1     # Density profile parameter
nu_T  = 1       # Temperature profile parameter
C_Alpha = 6     # Helium dilution tuning parameter (default : 5)

# Flux
Ce = 0.45            # Ejima constants
ITERPI = 20          # ITER plasma induction current [Wb]

# TF
coef_inboard_tension = 1/2 # Paramètre représentant la répartition jambe interne / externe de la tension
F_CClamp = 0e6       # C-Clamp limit in N , max order of magnitude from DDD : 30e6 N and of 60e6 N from [Bachmann (2023) FED]
n_TF = 1             # Asymetry conductor parameter 
c_BP = 0.07          # Backplate thickness

# CS
Gap = 0.1            # Gap between wedging and bucking CS and TF [m]

# Efficienty      
eta_T = 0.4          # Ratio between thermal and electrical power
eta_RF = 0.8 * 0.5   # fraction of klystron power absorbed by plasma * conversion efficiency from wall power to klystron

# PFU
theta_deg = 2.7      # Angle d'incidence sur les PFU pour calcul du flux de chaleur
# Source 1: T. R. Reiter, “Basic Fusion Boundary Plasma Physics,” ITER School Lecture Notes, Jan. 21 2019
# Source 2: “SOLPS-ITER simulations of the ITER divertor with improved plasma conditions,” Journal of Nuclear Materials (2024) )

# Current density scaling
T_helium = 4.2           # K temeprature de l'helium considéré
Marge_T_Helium = 0.3     # Puisque l'helium est à 10 barre, il est poussé à 0.3 K au dessus de la consigne à 1 Bar
Marge_CS = -1            # Puisque le CS n'est pas soumis à un flux neutronique, les marges sont plus souples
f_Cu = 0.5               # Copper fraction
f_Cool = 0.7             # Cooling fraction
f_In = 0.75              # Insulation fraction
# Rebco
Tet = 0 # Orientation pessimiste
Marge_T_Rebco = 5 #K
# Nb3Sn
Eps = -0.6 / 100 # Deformation criteria
Marge_T_Nb3Sn = 2 #K
# NbTi
Marge_T_NbTi = 1.7 #K

#%% Numerical initialisation

# Hide runtime-related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create a directory to save images if it doesn't exist
# save_directory = "Graphs"
save_directory = os.path.join(os.getcwd(), "Graphs", "Données brutes")

os.makedirs(save_directory, exist_ok=True)

#%%

print("D0FUS_parameterization loaded")