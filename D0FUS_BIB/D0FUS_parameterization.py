"""
Created on: Dec 2023
Author: Auclair Timothe
"""
#%% Import

from .D0FUS_import import *

#%% Code

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

# Plasma stability limits
betaN_limit = 2.8  # Beta Troyon limit
q_limit = 2.5      # Safety factor limit

# Steel
σ_manual = 1500         # MPa
nu_Steel = 0.29         # Poisson's ratio [CIRCEE model]
Young_modul_Steel = 200e9   # Young modulus for steel [CIRCEE model]
Young_modul_Glass_Fiber = 90e9  # Young modulus for glass fiber [CIRCEE model]
# S glass found in https://www.engineeringtoolbox.com/polymer-composite-fibers-d_1226.html

C_Alpha = 5     # Helium dilution tuning parameter

# Flux
Ce = 0.45            # Ejima constant
ITERPI = 20          # ITER plasma induction current [Wb]

# TF
coef_inboard_tension = 1/2    # Paramètre représentant la répartition jambe interne / externe de la tension
F_CClamp = 0e6                # C-Clamp limit in N , max order of magnitude from DDD : 30e6 N and of 60e6 N from [Bachmann (2023) FED]
n_TF = 1                      # Asymetry conductor parameter 
c_BP = 0.07                   # Backplate thickness [m]

# CS
Gap = 0.1            # Gap between wedging and bucking CS and TF [m]
n_CS = 1

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

# Conversion efficiencies      
eta_T = 0.4          # Ratio between thermal and electrical power
eta_RF = 0.8 * 0.5   # fraction of klystron power absorbed by plasma * conversion efficiency from wall power to klystron

# PFU
theta_deg = 2.7      # Angle d'incidence sur les PFU pour calcul du flux de chaleur
# Source 1: T. R. Reiter, “Basic Fusion Boundary Plasma Physics,” ITER School Lecture Notes, Jan. 21 2019
# Source 2: “SOLPS-ITER simulations of the ITER divertor with improved plasma conditions,” Journal of Nuclear Materials (2024) )


#%% Numerical initialisation

# Hide runtime-related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%%

# print("D0FUS_parameterization loaded")
