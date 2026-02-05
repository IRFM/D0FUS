"""
Cost functions definition for the D0FUS - Design 0-dimensional for FUsion Systems project
Created on: Jan 2026
Author: Matteo FLETCHER
"""

#%% Imports

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_parameterization import *
    from .D0FUS_cost_data import *

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_parameterization import *
    from D0FUS_BIB.D0FUS_cost_data import *
    
#%% Financial & Economics related functions

def f_capital_recovery_factor(discount_rate, T_life):
    """
    Under development by Matteo Fletcher
    
    Computes capital recovery factor (CRF) = fixed charged rate (FCR) from
    discount rate and plant lifetime

    Parameters
    ----------
    discount_rate : discount rate
    T_life : plant lifetime (yr)

    Returns
    -------
    CRF : capital recovery factor

    """
    CRF = (discount_rate * (1+discount_rate)**T_life) / ((1+discount_rate)**T_life - 1)
    return CRF

def f_unit_cost_scaling(quantity, unit_cost):
    """
    Under development by M. Fletcher
    
    Computes total cost using unit cost & given quantity, caution with units
    

    Parameters
    ----------
    quantity :
    unit_cost :

    Returns
    -------
    cost : total cost 

    """
    cost = quantity * unit_cost
    return cost

#%% Technology & Operation related general functions

def f_pp_availability(T_op_limit, dt_rep):
    """
    
    Under development by M. Fletcher
    
    computes power plant availibility (Av) from : maximum operation time 
    (T_op_limit) before replacement, and replacement time (dt_rep)

    Parameters
    ----------
    T_op_limit : operation time before neutron load limit [yr]
    dt_rep : replacement time [yr]

    Returns
    -------
    Av : power plant availability

    """
    Av = T_op_limit / (T_op_limit + dt_rep)
    return Av

def f_pp_capacity_factor(Av, Util_factor, Dwell_factor):
    """
    Under development by M. Fletcher
    
    computes power plant capacity factor (CF) from : power plant availability
    (Av), utilization factor (Util_factor), and dwell factor (Dwell_factor)

    Parameters
    ----------
    Av : power plant availability
    Util_factor : utilization factor
    Dwell_factor : dwell factor

    Returns
    -------
    CF : power plant capacity factor

    """
    CF = Av * Util_factor * Dwell_factor
    return CF

def f_critical_neutron_load(F_dpa, L_dpa, S):
    """
    
    Under development by M. Fletcher
    
    Computes the critical neutron load - relative to dpa - in [MW . yr] of a 
    component X from : neutron flux to dpa conversion factor (F_dpa), dpa limit
    (L_dpa), and surface exposed to neutron flux (S)

    Parameters
    ----------
    F_dpa : conversion factor between neutron flux and displacement per atom 
            (dpa) which quantifies wall damage [dpa . m² / (MW_neutron . yr)]
    L_dpa : dpa limit of the material [dpa]
    S : surface of material exposed to neutron flux
    
    Returns
    -------
    X_crit_load : critical neutron load of component X [MW . yr]

    """
    # Approximation of critical neutron load by D. Whyte (Seminar JPP 2024)
    X_crit_load = (L_dpa * S) / F_dpa
    return X_crit_load

def f_op_time_before_load_limit(X_crit_load, Util_factor, Dwell_factor, Load_nn):
    """
    Under development by M. Fletcher
    
    Computes the operation time [yr] before reaching neutron load limit of a 
    component X from : critical neutron load of component X (X_crit_load), 
    utilization factor (Util_factor), dwell factor (Dwell_factor), and nominal
    neutron load (Load_nn)
    
    Parameters
    ----------
    X_crit_load : critical neutron load of component X [MW . yr]
    Util_factor : utilization factor / equivalent nominal reactor usage 
                  fraction when available (no dimension)
    Dwell_factor : dwell factor / = 1 for steady-state operation, < 1 for
                   pulsed operation taking into account dwell times
    Load_nn : nominal neutron load on component X [MW]

    Returns
    -------
    T_op_lim : operation time before neutron load limit for component X [yr]

    """
    # Adapted from D. Whyte (Seminar JPP 2024)
    T_op_lim = X_crit_load / (Load_nn * Util_factor * Dwell_factor)
    return T_op_lim

#%% Cost functions - Dennis Whyte 2024 adapted model 

def f_costs_Whyte(F_dpa, L_dpa, S, Util_factor, Dwell_factor, dt_rep, 
                  discount_rate, T_life, Gamma_n, P_elec):
    """
    
    Under development by Matteo Fletcher
    
    Computes techno-economic parameters using an adaptation of D. Whyte 2024 model

    Parameters
    ----------
    F_dpa : conversion factor between neutron flux and displacement per atom 
            (dpa) which quantifies wall damage [dpa . m² / (MW_neutron . yr)]
    L_dpa : dpa limit of the material [dpa]
    S : first wall surface (m²)
    Util_factor : utilization factor
    Dwell_factor : dwell factor
    P_fus : Fusion power [MW]
    dt_rep : replacement time [yr]
    discount_rate : discount rate
    T_life : power plant lifetime (years)
    Gamma_n : averaged neutron flux (MW_n / m²)
    P_elec : nominal net electrical power output (MW)

    Returns
    -------
    BB_crit_load : critical neutron load of blanket [MW . yr]
    CF : capacity factor of power plant
    C_overnight : total power plant overnight cost (M€)
    COE : annual averaged cost of electricity for breakeven (€ / MWh)

    """
    # currency conversion
    cur_conv = c_2025EUR_2025USD
    
    # Breeding blanket critical neutron load (D. Whyte)
    BB_crit_load = f_critical_neutron_load(F_dpa, L_dpa, S)
    
    # Maximum operation time before reaching critical neutron load (adapted from D. Whyte)
    T_op_limit = f_op_time_before_load_limit(BB_crit_load, Util_factor, 
                                             Dwell_factor, S * Gamma_n)
        
    # Power Plant Availability & Capacity Factor
    Av = f_pp_availability(T_op_limit, dt_rep)
    CF = f_pp_capacity_factor(Av, Util_factor, Dwell_factor)
        
    # Power plant total overnight cost and blanket cost (D. Whyte)
    C_overnight = f_unit_cost_scaling(S, C_overnight_S_wht) * cur_conv
    C_blanket = f_unit_cost_scaling(S, C_blanket_S_wht) * cur_conv

    # Annual-amortized fixed cost (D. Whyte)
    C_fixed = f_annual_fixed_cost_Whyte(C_overnight, discount_rate, T_life)
    
    # Annual-averaged blanket replacement cost (adapted from D. Whyte)
    C_blanket_rep = f_annual_element_cost_Whyte(C_blanket, CF, Gamma_n, F_dpa, L_dpa)
    
    # Annual-averaged cost of electricity (COE) (adapted from D. Whyte)
    COE = f_COE_computation_Whyte(C_fixed, C_blanket_rep, CF, P_elec)
    
    return (T_op_limit, CF, C_overnight, COE)

def f_annual_fixed_cost_Whyte(C_overnight, d, T_life):
    """
    Under development by M. Fletcher
    
    Computes amortized annual fixed cost of the fusion power plant from total
    overnight cost (adapted from D. Whyte 2024)
    

    Parameters
    ----------
    C_overnight : total power plant overnight cost (M€)
    d : discount rate
    T_life : power plant lifetime (years)

    Returns
    -------
    C_fixed : amortized annual fixed cost (M€ / yr)

    """
    C_fixed = C_overnight * f_capital_recovery_factor(d, T_life)
    return C_fixed

def f_annual_element_cost_Whyte(C_element, CF, Flux_n, F_dpa, L_dpa):
    """
    Under development by M. Fletcher
    
    Computes annual element replacement cost from its unit cost and averaged 
    replacement frequency (from averaged neutron flux) 
    (adapted from D. Whyte 2024)
    

    Parameters
    ----------
    C_element : element unit cost (M€)
    CF : power plant capacity factor
    Flux_n : averaged neutron flux (MW_n / m²)
    F_dpa : neutron fluence to dpa conversion factor (dpa m² / (MW_n yr))
    L_dpa : system element dpa limit (dpa)

    Returns
    -------
    C_element_rep : averaged annual element replacement cost (M€ / yr)

    """
    C_element_rep = C_element * CF * Flux_n * F_dpa / L_dpa
    return C_element_rep

def f_COE_computation_Whyte(C_fixed, C_elements_rep, CF, P_elec):
    """
    Under development by M. Fletcher
    
    Computes cost of electricity (COE) from annual averaged fixed and 
    replacement costs + electrical net output (adapted from D. Whyte 2024)

    Parameters
    ----------
    C_fixed : amortized annual fixed cost (M€ / yr)
    C_elements_rep : averaged annual element replacement cost (M€ / yr)
    CF : power plant capacity factor
    P_elec : nominal net electrical power output (MW)

    Returns
    -------
    COE : annual averaged cost of electricity for breakeven (€ / MWh)

    """
    COE = 1e6 * (C_fixed + C_elements_rep) / (8760 * CF * P_elec)
    return COE

#%% Cost functions - Sheffield 2016 model (2010 US$)

def f_costs_Sheffield(discount_rate, contingency, T_life, T_build, 
                      P_t, P_e, P_aux, Gamma_n,
                      Util_factor, Dwell_factor, dt_rep,
                      R_c, a_c, delta_c, kappa,
                      V_pc, V_sg, V_bl, S_tt):
    """
    Under development by Matteo Fletcher
    
    main function for cost assessment using J. Sheffield 2016 model
    

    Parameters
    ----------
    discount_rate : expected fraction, not %
    contingency : expected fraction, not %
    T_life : power plant lifetime (from begining of operation to decomissioning) [yr]
    T_build : power plant construction time [yr]
    P_t : nominal power plant thermal power output [MWth]
    P_e : nominal power plant electrical power output [MWe]
    P_aux : electrical power to the auxiliary systems (heating & CD) [MWe]
    Gamma_n : neutron flux [MW/m²]
    Util_factor : expected fraction, not %
    Dwell_factor : expected fraction, not %
    dt_rep : replacement/maintenance time [yr]
    R_c : axisymmetry center to blanket outer edge distance [m]
    a_c : plasma core to coil outer edge distance [m]
    delta_c : coil width [m]
    kappa : plasma elongation
    V_pc : primary coil volume [m^3]
    V_sg : shield & gaps volume [m^3]
    V_bl : blanket volume [m^3]
    S_tt : target surface [m²]

    Returns
    -------
    T_op_limit : maximum operation time before requirred replacement [yr]
    CF : power plant capacity factor
    C_C0 : total capital (or overnight, or investment) cost [M chosen currency]
    COE : cost of electricity [chosen currency / MWh]

    """
    
    F_CRO = f_capital_recovery_factor(discount_rate, T_life)
    # currency conversion
    cur_conv = c_2025EUR_2010USD
    
    # total component costs from unit costs
    C_pc = f_unit_cost_scaling(V_pc, c_pc_sfd)
    C_sg = f_unit_cost_scaling(V_sg, c_sg_sfd)
    C_aux = f_unit_cost_scaling(P_aux, c_aux_sfd)
    C_bl = f_unit_cost_scaling(V_bl, c_bl_sfd)
    C_tt = f_unit_cost_scaling(S_tt, c_tt_sfd)
    
    # CapEx 
    C_FI = f_fusion_island_cost_Sheffield(C_heat_sfd, P_t, P_heat_sfd, 
                                          x_heat_sfd, 0.5, C_pc, 0.25,
                                          C_sg, 0.1, C_aux)
    V_FI = f_volume_fusion_island_Sheffield(R_c, a_c, delta_c, kappa)
    C_D = f_direct_cost_Sheffield(C_BoP_sfd, C_tur_sfd, P_tur_sfd, P_e, P_t, 
                                  P_BoP_sfd, x_BoP_sfd, C_bld_sfd, 
                                  V_FI, V_bld_sfd, x_bld_sfd, C_FI)
    C_CO = f_total_capital_cost_Sheffield(C_D, T_build, contingency)
    C_CO = C_CO * cur_conv
    
    # Thermal/neutron flux and lifetimes
    T_load_bl = T_load_bl_sfd
    T_load_tt = T_load_tt_sfd
    Load_bl = Gamma_n                       # nominal load
    Load_tt = load_factor_tt_sfd * Load_bl  # nominal load
    
    # Maximum operation time
    T_op_limit_bl = f_op_time_before_load_limit(T_load_bl, Util_factor, 
                                                Dwell_factor, Load_bl)
    T_op_limit_tt = f_op_time_before_load_limit(T_load_tt, Util_factor, 
                                                Dwell_factor, Load_tt)
    T_op_limit = min(T_op_limit_bl, T_op_limit_tt)
    
    # Availability and capacity factor
    Av = f_pp_availability(T_op_limit, dt_rep)
    CF = f_pp_capacity_factor(Av, Util_factor, Dwell_factor)
    
    # OpEx
    C_F = f_annual_consumables_costs_Sheffield(F_CRO, CF, T_life, 0.1, 0.1, 
                                               C_bl, Load_bl, T_load_bl, 0.1, 
                                               0.2, C_tt, Load_tt, T_load_tt, 
                                               0.1, C_aux, C_fa_sfd)
    C_F = C_F * cur_conv
    C_OM = f_annual_OandM_costs_Sheffield(C_OM_sfd, P_e, P_OM_sfd, x_OM_sfd)
    C_OM = C_OM * cur_conv
    
    # Cost of electricity
    COE = 1e6 * (C_CO * F_CRO + C_F + C_OM) / (8760 * CF * P_e) + C_waste_sfd * cur_conv
    
    return (T_op_limit, CF, C_CO, COE)

def f_direct_cost_Sheffield(c_BOPe_cst, c_BOPe_scale, P_BOPe_scale, P_e, 
                            P_t, P_BOPth_scale, x_BOPth_scale, 
                            c_syst_scale, V_syst_scale, x_syst_scale,
                            V_FI, C_FI):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    c_BOPe_cst : BOP cost scaling variable [2010 M$]
    c_BOPe_scale : BOP cost scaling variable [2010 M$]
    P_BOPe_scale : BOP cost scaling variable [MWe]
    P_e : power plant electrical output [MWe]
    P_t : power plant thermal output [MWth]
    P_BOPth_scale : BOP cost scaling variable [MWth]
    x_BOPth_scale : BOP cost scaling variable
    c_syst_scale : auxiliary systems cost scaling variable [2010 M$]
    V_syst_scale : auxiliary systems cost scaling variable [m^3]
    x_syst_scale : auxiliary systems cost scaling variable
    V_FI : fusion island total volume [m^3]
    C_FI : fusion island total cost [2010 M$]

    Returns
    -------
    C_D : total capital direct cost [2010 M$]

    """
    C_BOP_e = (c_BOPe_cst + c_BOPe_scale * P_e / P_BOPe_scale)
    f_BOP_th = (P_t / P_BOPth_scale)**x_BOPth_scale
    C_syst = c_syst_scale * (V_FI / V_syst_scale)**x_syst_scale
    C_D = C_BOP_e * f_BOP_th + C_syst + C_FI
    return C_D

def f_fusion_island_cost_Sheffield(c_scale_heat, P_t, P_scale_heat, x_scale_heat,
                                   red_sc, C_pc, ducts, C_sg, spare_aux, C_aux):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    c_scale_heat : heating system cost scaling variable [2010 M$]
    P_t : power plant thermal output [MWth]
    P_scale_heat : heating system cost scaling variable [MWth]
    x_scale_heat : heating system cost scaling variable []
    red_sc : redundancy for superconducting coils []
    C_pc : primary coil set cost [2010 M$]
    ducts : added fraction to shield for ducts []
    C_sg : shield and gaps cost [2010 M$]
    spare_aux : added fraction for spares in auxiliary heating system []
    C_aux : auxiliary heating system cost [2010 M$]

    Returns
    -------
    C_FI : fusion island cost [2010 M$]

    """
    C_heat = c_scale_heat * (P_t / P_scale_heat)**x_scale_heat
    C_coils = (1 + red_sc) * C_pc
    C_shield = (1 + ducts) * C_sg
    C_FI = C_heat + C_coils + C_shield + (1 + spare_aux) * C_aux
    return C_FI

def f_volume_fusion_island_Sheffield(R_c, a_c, delta_c, kappa):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    R_c : axisymmetry center to blanket outer edge distance [m]
    a_c : plasma core to coil outer edge distance [m]
    delta_c : coil width [m]
    kappa : plasma elongation

    Returns
    -------
    V_FI : fusion island volume rough estimate [m^3]

    """
    V_FI = 2 * np.pi * (R_c + a_c + 0.5 * delta_c)**2 * (kappa * a_c + 0.5 * delta_c)
    return V_FI

def f_annual_OandM_costs_Sheffield(c_scale_OM, P_e, P_scale_OM, x_scale_OM):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    c_scale_OM : operation & maintenance cost scaling variable [2010 M$]
    P_e : P_e : power plant electrical output [MWe]
    P_scale_OM : operation & maintenance cost scaling variable [MWe]
    x_scale_OM : operation & maintenance cost scaling variable []

    Returns
    -------
    C_OM : annual operation & maintenance cost [2010 M$]

    """
    C_OM = c_scale_OM * (P_e / P_scale_OM)**x_scale_OM
    return C_OM

def f_annual_consumables_costs_Sheffield(F_CRO, CF, T_life, spare_bl, failure_bl, 
                                         C_bl, Load_bl, T_load_bl, spare_tt, 
                                         failure_tt, C_tt, Load_tt, T_load_tt, 
                                         f_aaux, C_aux, C_fa):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    F_CRO : capital recovery factor []
    CF : power plant capacity factor []
    T_life : power plant lifetime (operational time) [yr]
    spare_bl : added fraction for blanket spares []
    failure_bl : added fraction in case of blanket failure []
    C_bl : blanket cost [2010 M$]
    Load_bl : neutron flux to blanket [MW/m²]
    T_load_bl : maximum blanket neutron fluence [MW yr / m²]
    spare_tt : added fraction for targets spares []
    failure_tt : added fraction in case of targets failure []
    C_tt : targets cost [2010 M$]
    Load_tt : heat flux to targets [MW/m²]
    T_load_tt : maximum targets heat fluence [MW yr / m²]
    f_aaux : annual fraction of auxiliary heating system cost []
    C_aux : auxiliary heating system cost [2010 M$]
    C_fa : fuel cost [2010 M$]

    Returns
    -------
    C_F : annual "consumable" costs [2010 M$]

    """
    C_ba = f_annual_rep_cost_Sheffield(spare_bl, failure_bl, C_bl, F_CRO, CF, 
                                       T_life, Load_bl, T_load_bl)
    C_ta = f_annual_rep_cost_Sheffield(spare_tt, failure_tt, C_tt, F_CRO, CF, 
                                       T_life, Load_tt, T_load_tt)
    C_F = C_ba + C_ta + f_aaux * C_aux + C_fa
    return C_F

def f_total_capital_cost_Sheffield(C_D, T_build, contingency):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    C_D : power plant direct costs [2010 M$]
    T_build : power plant construction time [yr]
    contingency : contingency (owner's cost, risk assessment, ...) []

    Returns
    -------
    C_CO : power plant total overnight cost [2010 M$]

    """
    f_IND = f_indirect_charge_capital_Sheffield(T_build)
    f_CAPO = f_interest_charge_construction_Sheffield(T_build)
    C_CO = C_D * f_IND * f_CAPO * (1 + contingency)
    return C_CO

def f_indirect_charge_capital_Sheffield(T_build):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    T_build : power plant construction time [yr]

    Returns
    -------
    f_IND : indirect charge capital factor for construction (design, engineering, 
                                                             management, ...) []

    """
    f_IND = 1 + 0.5 * T_build / 8
    return f_IND

def f_interest_charge_construction_Sheffield(T_build):
    """
    Under development by Matteo Fletcher

    Parameters
    ----------
    T_build : power plant construction time [yr]

    Returns
    -------
    f_CAPO : interest charge on construction factor []

    """
    f_CAPO = 1.011**(T_build + 0.61)
    return f_CAPO

def f_annual_rep_cost_Sheffield(spare, failure, C_init, F_CRO, CF, T_life, Load_nom, T_load):
    """
    Under development by Matteo Fletcher
    

    Parameters
    ----------
    spare : added fraction for spare []
    failure : added fraction for failure []
    C_init : system element cost [2010 M$]
    F_CRO : capital recovery factor []
    CF : power plant capacity factor []
    T_life : power plant lifetime [yr]
    Load_nom : flux to the system element [MW/m²]
    T_load : maximum tolerable fluence of system element [MW yr / m²]

    Returns
    -------
    C_a : annual replacement cost of system element [2010 M$]

    """
    C_first = (1 + spare) * C_init * F_CRO
    C_others = (CF * T_life * Load_nom / T_load - 1) * C_init / T_life
    C_a = (1 + failure) * (C_first + C_others)
    return C_a
    

#%% Cost functions - Fletcher 2026 model