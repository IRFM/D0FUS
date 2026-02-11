"""
D0FUS cost data Module
==============================
Cost-related data for tokamak design.

Created: Jan 2026
Author: Matteo FLETCHER
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
    
#%% Currency conversions

c_2025USD_2010USD = 1.4723
c_2025EUR_2025USD = 0.8862
c_2025EUR_2010USD = c_2025EUR_2025USD * c_2025USD_2010USD
    
#%% Denis Whyte model inputs (modified to ITER 2025 data)

C_invest_S_wht = 35.7 # 2025 ITER cost (~ 25 B$) / first wall surface (~ 700 m²)
C_blanket_S_wht = 0.07 * C_invest_S_wht # blanlet cost assumed 7% of total cost

#%% John Sheffield 2016 model inputs (2010 USD)

# Unit cost scalings from ITER 2008 cost estimates

c_pc_sfd  = 1.66  # $M / m^3
c_sg_sfd  = 0.29  # $M / m^3
c_aux_sfd = 5.3   # $M / W
c_bl_sfd  = 0.75  # $M / m^3
c_tt_sfd  = 0.114 # $M / m²

# External costs (no explicit reference)

C_waste_sfd = 0.5 # mill / kWh

# Annual fuel costs

C_fa_sfd = 7.5 # $M / yr

# Blanket, first wall and target lifetimes and loads

T_load_bl_sfd = 15 # MW yr / m²
T_load_tt_sfd = 10 # MW yr / m²
load_factor_tt_sfd = 10 / 3 # conversion factor : target thermal flux / neutron flux 

# Heat transfer system cost scaling

C_heat_sfd = 221  # $M
P_heat_sfd = 4150 # MWth
x_heat_sfd = 0.6

# BoP system + turbine cost scaling 

C_BoP_sfd = 900  # $M
C_tur_sfd = 900  # $M
P_tur_sfd = 1200 # MWe
P_BoP_sfd = 4150 # MWth
x_BoP_sfd = 0.6

# Buildings cost scaling

C_bld_sfd = 839  # $M
V_bld_sfd = 5100 # m^3
x_bld_sfd = 0.67

# Annual O&M costs scaling

C_OM_sfd = 108  # $M / yr
P_OM_sfd = 1200 # MWe
x_OM_sfd = 0.5

