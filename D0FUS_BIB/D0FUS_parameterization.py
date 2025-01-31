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

# Physical constants
E_ELEM  = 1.6e-19           # Electron charge [Coulomb]
M_E     = 9.1094E-31        # kg
M_I     = 2 * 1.6726E-27    # kg
μ0    = 4.*np.pi*1.E-7    # Henri/m
EPS_0   = 8.8542E-12        # Farad/m

# Fusion constants
E_ALPHA = 3.5 *1.E6*E_ELEM  # Joule
E_N     = 14.1*1.E6*E_ELEM  # Joule
E_F     = 22.4*1.E6*E_ELEM  # Joule (considering all N react with Li)
Atomic_mass  = 2.5  # average atomic mass in AMU

# Plasma stability
betaN = 2.8  # in % (Beta Tryon limit)
q = 2.5 # Security factor limit

# Density and Temperature parameters
Tbar  = 14 # keV Mean Temperature
nu_n  = 0.1  # Density profile parameter
nu_T  = 1. # Temperature profile parameter
C_Alpha = 5 # Helium dilution tuning parameter

# Flux consumptions assumtions
Gap = 0.1 # Gap between wedging and bucking CS and TF
Ce = 0.45 # Ejima constants
Temps_Plateau = 0 # Plateautime [?]
Li = 0.85 # Internal inductance of the plasma
ITERPI = 20 # ITER plasma induction current [Wb]
Flux_CS_Utile = 0.95 #0.85 # pourcentage disponible, laissant une partie libre pour le controle du plasma , arbitrairement à 0.85, la litterature propose plusieurs values allant de 0.85 à 1

# Geometric parameters
κ = 2.1  # elongation
δ = 0.5 # Triangularity
b = 1.2 # BB+1rst Wall+N shields+ Gaps

# Engineer constraints
σ_TF = 660.E6   # Mechanical limit of the steel considered in [Pa]
σ_CS = 660.E6   # CS machanical limit [Pa] 
J_max_TF_conducteur = 50.E6       # A/m² from ITER values
J_max_TF_cable = 20e6  # Taken from ITER
J_max_CS_conducteur = 50.E6       # A/m² from ITER values
eta_RF = 0.5  # conversion efficiency from wall power to klystron
f_RP   = 0.8  # fraction of klystron power absorbed by plasma
eta_T = 0.4    # Ratio between thermal and electrical power
F_CClamp = 0 # Typical value if considered, of 60e6 from [Bachmann (2023) FED]

# Parameterization
H = 1.0 # H factor
Choice_Buck_Wedg = 'Wedging' # Wedging or Bucking
Mechanical_model = 'Full_winding_pack' # Full_winding_pack or Winding_pack_and_Nose
Option_Kappa = 'Wenninger' # Choose between Stambaugh, Freidberg or Wenninger
L_H_Scaling_choice = 'New_Ip' # 'Martin' , 'New_S', 'New_Ip'
Q_convergence_choice = 'Fast' #'Fast' or 'Slow'

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

def initialize_lists():
    # Initialize lists to store the results
    a_solutions = []
    nG_solutions = []
    n_solutions = []
    beta_solutions = []
    qstar_solutions = []
    fB_solutions = []
    fNC_solutions = []
    cost_solutions = []
    heat_solutions = []
    c_solutions = []
    sol_solutions = []
    R0_a_solutions = []
    R0_a_b_solutions = []
    R0_a_b_c_solutions = []
    R0_a_b_c_CS_solutions = []
    required_BCSs = []
    R0_solutions = []
    Ip_solutions = []
    fRF_solutions = []
    P_W_solutions = []
    
    return (
        a_solutions, 
        nG_solutions, 
        n_solutions, 
        beta_solutions, 
        qstar_solutions, 
        fB_solutions, 
        fNC_solutions, 
        cost_solutions, 
        heat_solutions, 
        c_solutions, 
        sol_solutions,
        R0_a_solutions,
        R0_a_b_solutions,
        R0_a_b_c_solutions,
        R0_a_b_c_CS_solutions,
        required_BCSs,
        R0_solutions,
        Ip_solutions,
        fRF_solutions,
        P_W_solutions
    )

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions,P_W_solutions) = initialize_lists()

#%%

print("D0FUS_parameterization loaded")