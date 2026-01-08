"""
D0FUS Parameterization Module
==============================
Physical constants, material properties, and default parameters for tokamak design.

Created: December 2023
Author: Auclair Timothé
"""

#%% Imports

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *

#%% Physical Constants

# Fundamental constants
E_ELEM = 1.6e-19            # Elementary charge [C]
M_E = 9.1094e-31            # Electron mass [kg]
M_I = 2 * 1.6726e-27        # Ion mass (deuterium) [kg]
μ0 = 4.0 * np.pi * 1.0e-7   # Vacuum permeability [H/m]
EPS_0 = 8.8542e-12          # Vacuum permittivity [F/m]

#%% Fusion Reaction Parameters

# Energy release per reaction
E_ALPHA = 3.5 * 1.0e6 * E_ELEM   # Alpha particle energy [J]
E_N = 14.1 * 1.0e6 * E_ELEM      # Neutron energy [J]
E_F = 22.4 * 1.0e6 * E_ELEM      # Total fusion energy (assuming all neutrons react with Li) [J]

# Plasma composition
Atomic_mass = 2.5       # Average atomic mass [AMU]
Zeff = 1                # Effective charge (default: 1)
r_synch = 0.5           # Synchrotron radiation reflection coefficient

#%% Plasma Stability Limits

betaN_limit = 2.8       # Troyon beta limit [% m T/MA]
q_limit = 2.5           # Minimum safety factor (q95 or q*)
ms = 0.3                # Vertical stability margin parameter

#%% Material Properties

# Structural steel
σ_manual = 1500                 # Manual stress limit [MPa]
nu_Steel = 0.29                 # Poisson's ratio (CIRCEE model)
Young_modul_Steel = 200e9       # Young's modulus [Pa] (CIRCEE model)
Young_modul_Glass_Fiber = 90e9  # Young's modulus for S-glass fiber [Pa]
# Reference: https://www.engineeringtoolbox.com/polymer-composite-fibers-d_1226.html

#%% Plasma Performance Parameters

C_Alpha = 5             # Helium ash dilution tuning parameter

#%% Magnetic Flux Parameters

Ce = 0.45               # Ejima constant (flux consumption)
ITERPI = 20             # ITER plasma induction flux [Wb]

#%% Toroidal Field (TF) Coil Parameters

coef_inboard_tension = 1/2      # Stress distribution ratio (inboard/outboard leg)
F_CClamp = 0e6                  # C-Clamp structural limit [N]
                                # Typical range: 30e6 N (DDD) to 60e6 N (Bachmann 2023, FED)
n_TF = 1                        # Conductor asymmetry parameter
c_BP = 0.07                     # Backplate thickness [m]

#%% Central Solenoid (CS) Parameters

Gap = 0.1               # Clearance between CS wedging/bucking and TF [m]
n_CS = 1                # CS conductor parameter

#%% Superconductor Operating Conditions

# If manual option: current density fixed
Jc_Manual = 100e6        # MA/m²

# Helium cooling
T_helium = 4.2          # Liquid helium temperature [K]
Marge_T_Helium = 0.3    # Temperature margin linked to 10 bar operation [K]

# Area fractions
f_Cu_Non_Cu = 1 - 0.5       # Copper fraction in the strand
f_Cu_Strand = 1 - 0.3       # Copper stabilizer fraction (n_Cu/(n_Cu+n_Su))
f_Cool = 1 - 0.3            # Cooling channel fraction
f_In = 1 - 0.2              # Insulation fraction

# Temperature margins [K]
# Added to T_He to obtain T_operating (conservative design approach)
Marge_T_Nb3Sn  = 2.0    # Safety margin for Nb3Sn
Marge_T_NbTi   = 1.7    # Safety margin for NbTi
Marge_T_Rebco  = 5.0    # Safety margin for REBCO

# Default operating parameters
Eps = -0.6/100           # Effective strain for Nb3Sn [-] (EU-DEMO TF: -0.35%)
Tet = 0.0               # REBCO tape angle [rad] (0 = B⊥, π/2 = B//ab)

#%% Power Conversion Efficiencies

eta_T = 0.4             # Thermal-to-electric conversion efficiency
eta_RF = 0.8 * 0.5      # RF heating efficiency (klystron efficiency × plasma absorption)

#%% Plasma-Facing Components (PFU)

theta_deg = 2.7         # Grazing angle at divertor strike point [deg]
# References:
# - T. R. Reiter, "Basic Fusion Boundary Plasma Physics," ITER School Lecture Notes (2019)
# - "SOLPS-ITER simulations of the ITER divertor with improved plasma conditions," 
#   Journal of Nuclear Materials (2024)

#%% Numerical Configuration

# Suppress runtime warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%%

# print("D0FUS_parameterization loaded")