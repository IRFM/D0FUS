"""
D0FUS cost functions Module
==============================
Cost models for tokamak fusion power plant techno-economic assessment.

Models implemented:
    1. Whyte (2024 adapted)  — simplified investment scaling from FW surface.
    2. Sheffield (2016)      — generic magnetic fusion reactor cost model.

Architecture:
    Generic utilities (CRF, availability, capacity factor, neutron load)
    are shared by both models and defined in Section 1-2.
    Model-specific functions follow in Sections 3-4.

References
----------
[Whyte 2024]     D. Whyte, "Fusion economy", Seminar JPP, CEA Cadarache (2024).
[Sheffield 2016] J. Sheffield, S. L. Milora, "Generic Magnetic Fusion Reactor
                 Revisited", Fus. Sci. Technol. 70, 14-35 (2016).
                 doi:10.13182/FST15-157
[Sheffield 1986] J. Sheffield et al., "Cost Assessment of a Generic Magnetic
                 Fusion Reactor", Fusion Technol. 9, 199-249 (1986).
[Jo 2021]        G. Jo et al., "Cost Assessment of a Tokamak Fusion Reactor
                 with an Inventive Method for Optimum Build Determination",
                 Energies 14, 6817 (2021).  doi:10.3390/en14206817
                 — Benchmark: COE 109-140 mills/kWh for R0 ~ 6-7 m.

Created on: Jan 2026
Author: Matteo FLETCHER
"""

#%% Imports

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    try:
        from .D0FUS_import import *
        from .D0FUS_parameterization import *
        from .D0FUS_cost_data import *
    except ImportError:
        import numpy as np
        from D0FUS_cost_data import *

# When executed directly (for testing and development)
else:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    try:
        from D0FUS_BIB.D0FUS_import import *
        from D0FUS_BIB.D0FUS_parameterization import *
        from D0FUS_BIB.D0FUS_cost_data import *
    except ModuleNotFoundError:
        import numpy as np
        from D0FUS_cost_data import *

#%% 1. Financial & Economics related functions

def f_capital_recovery_factor(discount_rate, T_life):
    """
    Capital recovery factor (CRF), also called fixed charge rate (FCR).

    Annualises a lump-sum capital cost over the plant lifetime at a given
    discount rate.  Standard financial formula used identically in
    Sheffield (2016) Eq. 15 and Entler (2018) Section 6.

    Parameters
    ----------
    discount_rate : float  Real discount rate [-] (e.g. 0.07 for 7%).
    T_life        : float  Plant operational lifetime [yr].

    Returns
    -------
    CRF : float  Capital recovery factor [yr^-1].

    Notes
    -----
    CRF = r(1+r)^T / ((1+r)^T - 1),  with r = discount_rate, T = T_life.
    For r = 0.07, T = 40 yr: CRF ~ 0.0750.
    """
    CRF = (discount_rate * (1+discount_rate)**T_life) / ((1+discount_rate)**T_life - 1)
    return CRF

def f_unit_cost_scaling(quantity, unit_cost):
    """
    Linear cost scaling: C = quantity * unit_cost.

    Utility wrapper for readability.  Caller is responsible for unit
    consistency (e.g. m^3 * M$/m^3 -> M$).

    Parameters
    ----------
    quantity  : float  Physical quantity (volume, surface, power, ...).
    unit_cost : float  Cost per unit of quantity.

    Returns
    -------
    cost : float  Total cost [same currency as unit_cost].
    """
    cost = quantity * unit_cost
    return cost

#%% 2. Technology & Operation related general functions

def f_pp_availability(T_op_limit, dt_rep):
    """
    Power plant availability fraction.

    Simple model: the plant operates for T_op_limit years, then is shut
    down for dt_rep years for scheduled component replacement.
    Refs: Sheffield (2016) Section III.B; Entler (2018) Section 5.

    Parameters
    ----------
    T_op_limit : float  Maximum operation time before replacement [yr].
    dt_rep     : float  Scheduled replacement/maintenance downtime [yr].

    Returns
    -------
    Av : float  Plant availability [-], in (0, 1).
    """
    Av = T_op_limit / (T_op_limit + dt_rep)
    return Av

def f_pp_capacity_factor(Av, Util_factor, Dwell_factor):
    """
    Power plant capacity factor.

    CF = Av * Util_factor * Dwell_factor.
    - Av: hardware availability (replacement cycle).
    - Util_factor: fraction of available time at nominal power.
    - Dwell_factor: duty cycle correction for pulsed operation
      (1.0 for steady-state, < 1.0 for pulsed).

    Parameters
    ----------
    Av           : float  Plant availability [-].
    Util_factor  : float  Utilisation factor [-].
    Dwell_factor : float  Dwell time factor [-] (1.0 = steady-state).

    Returns
    -------
    CF : float  Capacity factor [-], in (0, 1).
    """
    CF = Av * Util_factor * Dwell_factor
    return CF

def f_critical_neutron_load(F_dpa, L_dpa, S):
    """
    Critical neutron load of a plasma-facing component [MW yr].

    The component reaches its end-of-life when the accumulated fluence
    (Gamma_n * t) produces L_dpa displacements per atom.

    Approximation from D. Whyte (Seminar JPP 2024).

    Parameters
    ----------
    F_dpa : float  Neutron-to-dpa conversion factor [dpa m^2 / (MW yr)].
                   Typical: ~10 for EUROFER97 at 14 MeV.
                   Ref: Gilbert et al., NF 57 (2017) 046015.
    L_dpa : float  Material dpa limit [dpa].
                   EUROFER97: ~50 dpa; ODS steels: potentially higher.
                   Ref: Zinkle & Snead, Annu. Rev. Mater. Res. 44 (2014) 241.
    S     : float  Surface exposed to neutron flux [m^2].

    Returns
    -------
    X_crit_load : float  Critical neutron load [MW yr].
    """
    X_crit_load = (L_dpa * S) / F_dpa
    return X_crit_load

def f_op_time_before_load_limit(X_crit_load, Util_factor, Dwell_factor, Load_nn):
    """
    Maximum operation time before reaching the neutron load limit [yr].

    T_op = X_crit_load / (Load_nn * Util_factor * Dwell_factor).
    Adapted from D. Whyte (Seminar JPP 2024).

    Parameters
    ----------
    X_crit_load  : float  Critical neutron load of the component [MW yr].
    Util_factor  : float  Utilisation factor [-].
    Dwell_factor : float  Dwell time factor [-].
    Load_nn      : float  Nominal neutron/heat load on the component [MW].

    Returns
    -------
    T_op_lim : float  Operation time before load limit [yr].
    """
    T_op_lim = X_crit_load / (Load_nn * Util_factor * Dwell_factor)
    return T_op_lim

#%% 3. Cost functions - Dennis Whyte 2024 adapted model

def f_costs_Whyte(F_dpa, L_dpa, S, Util_factor, Dwell_factor, dt_rep,
                  discount_rate, T_life, Gamma_n, P_elec):
    """
    Techno-economic assessment using an adaptation of D. Whyte 2024 model.

    Investment is proportional to first-wall surface area (scaled from ITER).
    Blanket replacement is the dominant OpEx.  Fuel costs are negligible.

    Parameters
    ----------
    F_dpa         : float  Neutron-to-dpa factor [dpa m^2 / (MW yr)].
    L_dpa         : float  Material dpa limit [dpa].
    S             : float  First wall surface [m^2].
    Util_factor   : float  Utilisation factor [-].
    Dwell_factor  : float  Dwell time factor [-].
    dt_rep        : float  Replacement downtime [yr].
    discount_rate : float  Real discount rate [-].
    T_life        : float  Plant lifetime [yr].
    Gamma_n       : float  Average neutron wall load [MW/m^2].
    P_elec        : float  Net electric power output [MWe].

    Returns
    -------
    tuple of 16 floats (same shape as Sheffield for interface compatibility):
        [0] T_op_limit : float  Time before blanket replacement [yr].
        [1] CF         : float  Capacity factor [-].
        [2] C_invest   : float  Total investment cost [M, local currency].
        [3] COE        : float  Cost of electricity [local currency / MWh].
        [4:] zeros (padding for interface compatibility with Sheffield).
    """
    # Currency conversion: 2025 USD -> 2025 EUR
    cur_conv = c_2025EUR_2025USD

    # Breeding blanket critical neutron load
    BB_crit_load = f_critical_neutron_load(F_dpa, L_dpa, S)

    # Maximum operation time before reaching critical neutron load
    T_op_limit = f_op_time_before_load_limit(BB_crit_load, Util_factor,
                                             Dwell_factor, S * Gamma_n)

    # Power Plant Availability & Capacity Factor
    Av = f_pp_availability(T_op_limit, dt_rep)
    CF = f_pp_capacity_factor(Av, Util_factor, Dwell_factor)

    # Investment cost and blanket cost
    C_invest  = f_unit_cost_scaling(S, C_invest_S_wht) * cur_conv
    C_blanket = f_unit_cost_scaling(S, C_blanket_S_wht) * cur_conv

    # Annual-amortized fixed cost
    C_fixed = f_annual_fixed_cost_Whyte(C_invest, discount_rate, T_life)

    # Annual-averaged blanket replacement cost
    C_blanket_rep = f_annual_element_cost_Whyte(C_blanket, CF, Gamma_n, F_dpa, L_dpa)

    # Cost of electricity
    COE = f_COE_computation_Whyte(C_fixed, C_blanket_rep, CF, P_elec)

    return (T_op_limit, CF, C_invest, COE, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0)

def f_annual_fixed_cost_Whyte(C_invest, d, T_life):
    """
    Amortized annual fixed cost = C_invest * CRF.

    Parameters
    ----------
    C_invest : float  Total investment cost [M, any currency].
    d        : float  Discount rate [-].
    T_life   : float  Plant lifetime [yr].

    Returns
    -------
    C_fixed : float  Annual fixed cost [M/yr].
    """
    C_fixed = C_invest * f_capital_recovery_factor(d, T_life)
    return C_fixed

def f_annual_element_cost_Whyte(C_element, CF, Flux_n, F_dpa, L_dpa):
    """
    Annual replacement cost of a nuclear component, averaged over its
    wear-out cycle.  Adapted from D. Whyte (2024).

    C_rep = C_element * CF * Gamma_n * F_dpa / L_dpa.

    Parameters
    ----------
    C_element : float  Component unit cost [M, any currency].
    CF        : float  Capacity factor [-].
    Flux_n    : float  Average neutron flux [MW/m^2].
    F_dpa     : float  Neutron-to-dpa factor [dpa m^2 / (MW yr)].
    L_dpa     : float  Material dpa limit [dpa].

    Returns
    -------
    C_element_rep : float  Annualised replacement cost [M/yr].
    """
    C_element_rep = C_element * CF * Flux_n * F_dpa / L_dpa
    return C_element_rep

def f_COE_computation_Whyte(C_fixed, C_elements_rep, CF, P_elec):
    """
    Cost of electricity (COE) from annualised costs.  Adapted from Whyte (2024).

    COE = 10^6 * (C_fixed + C_rep) / (8760 * CF * P_elec).

    Parameters
    ----------
    C_fixed        : float  Annual fixed cost [M/yr].
    C_elements_rep : float  Annual replacement cost [M/yr].
    CF             : float  Capacity factor [-].
    P_elec         : float  Net electric power [MWe].

    Returns
    -------
    COE : float  Cost of electricity [currency / MWh].
    """
    COE = 1e6 * (C_fixed + C_elements_rep) / (8760 * CF * P_elec)
    return COE

#%% 4. Cost functions - Sheffield 2016 model (2010 US$)

def f_costs_Sheffield(discount_rate, contingency, T_life, T_build,
                      P_t, P_e, P_aux, Gamma_n,
                      Util_factor, Dwell_factor, dt_rep,
                      V_FI, V_pc, V_sg, V_bl, S_tt,
                      Supra_cost_factor):
    """
    Full cost assessment using Sheffield & Milora (2016) model.

    Computes CapEx (fusion island + BoP + buildings + indirects), OpEx
    (O&M + replacements + fuel + waste), capacity factor, and COE.
    All internal calculations in 2010 USD, converted to 2025 EUR on output.

    Ref: Sheffield & Milora, Fus. Sci. Technol. 70, 14-35 (2016).
    Benchmark: Jo et al., Energies 14, 6817 (2021) — COE 109-140 mills/kWh.

    Parameters
    ----------
    discount_rate     : float  Real discount rate [-].
    contingency       : float  Contingency fraction [-].
    T_life            : int    Plant operational lifetime [yr].
    T_build           : int    Construction time [yr].
    P_t               : float  Thermal power output [MWth].
    P_e               : float  Net electric power output [MWe].
    P_aux             : float  Auxiliary heating power to plasma [MW].
    Gamma_n           : float  Neutron wall load [MW/m^2].
    Util_factor       : float  Utilisation factor [-].
    Dwell_factor      : float  Dwell factor [-] (1.0 for steady-state).
    dt_rep            : float  Replacement downtime [yr].
    V_FI              : float  Fusion island bounding volume [m^3].
    V_pc              : float  Primary coil volume (TF + CS) [m^3].
    V_sg              : float  Shield and gaps volume [m^3].
    V_bl              : float  Blanket volume [m^3].
    S_tt              : float  Divertor target surface [m^2].
    Supra_cost_factor : float  SC coil cost multiplier vs Cu [-].

    Returns
    -------
    tuple of 16 floats:
        (T_op_limit, CF, C_CO, COE,
         C_ind, C_Op_waste, C_Op_OM, C_Op_F,
         C_syst_other, C_syst_BOP, C_syst_heat, C_syst_aux,
         C_reac_tt, C_reac_bl, C_reac_sg, C_reac_pc)
    All costs in [2025 M EUR] except COE in [2025 EUR/MWh].
    """
    # Economic parameters
    F_CRO = f_capital_recovery_factor(discount_rate, T_life)
    cur_conv = c_2025EUR_2010USD

    # Tokamak component costs from unit costs [2010 M$]
    C_pc  = Supra_cost_factor * f_unit_cost_scaling(V_pc, c_pc_sfd)
    C_sg  = f_unit_cost_scaling(V_sg, c_sg_sfd)
    C_aux = f_unit_cost_scaling(P_aux, c_aux_sfd)
    C_bl  = f_unit_cost_scaling(V_bl, c_bl_sfd)
    C_tt  = f_unit_cost_scaling(S_tt, c_tt_sfd)

    # CapEx: fusion island -> direct cost -> total capital cost
    (C_FI, C_heat) = f_fusion_island_cost_Sheffield(C_heat_sfd, P_t, P_heat_sfd,
                                                    x_heat_sfd, red_pc_sfd, C_pc,
                                                    ducts_sg_sfd, C_sg, spare_bl_sfd,
                                                    C_bl, spare_tt_sfd, C_tt,
                                                    spare_aux_sfd, C_aux)
    (C_D, C_BOP_tot, C_syst) = f_direct_cost_Sheffield(C_BoP_sfd, C_tur_sfd,
                                                       P_tur_sfd, P_e, P_t,
                                                       P_BoP_sfd, x_BoP_sfd,
                                                       C_bld_sfd, V_bld_sfd,
                                                       x_bld_sfd, V_FI, C_FI)
    (C_CO, C_ind) = f_total_capital_cost_Sheffield(C_D, T_build, contingency)

    # Thermal/neutron flux and component lifetimes
    T_load_bl = T_load_bl_sfd
    Load_bl = Gamma_n
    ###########################################################################
    #### DIVERTOR TARGETS REPLACEMENTS NEGLECTED HERE BECAUSE OF ABSENCE OF ###
    ## MODEL TO COMPUTE FLUX TO DIVERTOR AND ITS LIFETIME : FOR FUTURE WORK ? #
    ###########################################################################
    T_load_tt = 1e20     # effectively infinite -> no divertor replacement
    Load_tt = load_factor_tt_sfd * Load_bl

    # Maximum operation time
    T_op_limit_bl = f_op_time_before_load_limit(T_load_bl, Util_factor,
                                                Dwell_factor, Load_bl)
    T_op_limit_tt = f_op_time_before_load_limit(T_load_tt, Util_factor,
                                                Dwell_factor, Load_tt)
    T_op_limit = min(T_op_limit_bl, T_op_limit_tt)

    # Availability and capacity factor
    Av = f_pp_availability(T_op_limit, dt_rep)
    CF = f_pp_capacity_factor(Av, Util_factor, Dwell_factor)

    # OpEx [2010 M$/yr]
    C_F = f_annual_consumables_costs_Sheffield(CF, T_life, 0.1, C_bl, Load_bl,
                                               T_load_bl, 0.2, C_tt, Load_tt,
                                               T_load_tt, 0.1, C_aux, C_fa_sfd)
    C_OM = f_annual_OandM_costs_Sheffield(C_OM_sfd, P_e, P_OM_sfd, x_OM_sfd)

    # Cost of electricity [2010 $/MWh]
    COE = 1e6 * (C_CO * F_CRO + C_F + C_OM) / (8760 * CF * P_e) + C_waste_sfd

    # Convert all outputs to 2025 EUR
    C_reac_pc    = C_pc  * (1+red_pc_sfd)    * cur_conv
    C_reac_sg    = C_sg  * (1+ducts_sg_sfd)  * cur_conv
    C_reac_bl    = C_bl  * (1+spare_bl_sfd)  * cur_conv
    C_reac_tt    = C_tt  * (1+spare_tt_sfd)  * cur_conv
    C_syst_aux   = C_aux * (1+spare_aux_sfd) * cur_conv
    C_syst_heat  = C_heat  * cur_conv
    C_syst_BOP   = C_BOP_tot * cur_conv
    C_syst_other = C_syst  * cur_conv
    C_Op_F       = C_F   * cur_conv
    C_Op_OM      = C_OM  * cur_conv
    C_Op_waste   = C_waste_sfd * cur_conv
    C_ind        = C_ind * cur_conv
    C_CO         = C_CO  * cur_conv
    COE          = COE   * cur_conv

    return (T_op_limit, CF, C_CO, COE, C_ind, C_Op_waste, C_Op_OM, C_Op_F,
            C_syst_other, C_syst_BOP, C_syst_heat, C_syst_aux, C_reac_tt,
            C_reac_bl, C_reac_sg, C_reac_pc)

def f_direct_cost_Sheffield(c_BOPe_cst, c_BOPe_scale, P_BOPe_scale, P_e,
                            P_t, P_BOPth_scale, x_BOPth_scale,
                            c_syst_scale, V_syst_scale, x_syst_scale,
                            V_FI, C_FI):
    """
    Direct capital cost: fusion island + BoP + buildings/auxiliaries.
    Sheffield (2016) Table IV.

    Parameters
    ----------
    c_BOPe_cst, c_BOPe_scale, P_BOPe_scale : BoP electric cost scaling.
    P_e           : float  Net electric power [MWe].
    P_t           : float  Thermal power [MWth].
    P_BOPth_scale : float  BoP thermal scaling reference [MWth].
    x_BOPth_scale : float  BoP thermal economy-of-scale exponent [-].
    c_syst_scale  : float  Auxiliary systems reference cost [2010 M$].
    V_syst_scale  : float  Auxiliary systems reference volume [m^3].
    x_syst_scale  : float  Auxiliary systems exponent [-].
    V_FI          : float  Fusion island volume [m^3].
    C_FI          : float  Fusion island cost [2010 M$].

    Returns
    -------
    C_D       : float  Total direct cost [2010 M$].
    C_BOP_tot : float  BoP + turbine cost [2010 M$].
    C_syst    : float  Buildings & auxiliary systems cost [2010 M$].
    """
    C_BOP_e   = (c_BOPe_cst + c_BOPe_scale * P_e / P_BOPe_scale)
    f_BOP_th  = (P_t / P_BOPth_scale)**x_BOPth_scale
    C_BOP_tot = C_BOP_e * f_BOP_th
    C_syst    = c_syst_scale * (V_FI / V_syst_scale)**x_syst_scale
    C_D       = C_BOP_tot + C_syst + C_FI
    return (C_D, C_BOP_tot, C_syst)

def f_fusion_island_cost_Sheffield(c_scale_heat, P_t, P_scale_heat, x_scale_heat,
                                   red_sc, C_pc, ducts, C_sg, spare_bl, C_bl,
                                   spare_tt, C_tt, spare_aux, C_aux):
    """
    Fusion island cost: heat transfer + coils + shield + blanket + targets + aux.
    Sheffield (2016) Table IV.

    Parameters
    ----------
    c_scale_heat, P_t, P_scale_heat, x_scale_heat : heat system scaling.
    red_sc   : float  SC coil redundancy fraction [-].
    C_pc     : float  Primary coil cost [2010 M$].
    ducts    : float  Shield duct surcharge fraction [-].
    C_sg     : float  Shield & gaps cost [2010 M$].
    spare_bl : float  Blanket spares fraction [-].
    C_bl     : float  Blanket cost [2010 M$].
    spare_tt : float  Target spares fraction [-].
    C_tt     : float  Target cost [2010 M$].
    spare_aux: float  Aux heating spares fraction [-].
    C_aux    : float  Aux heating cost [2010 M$].

    Returns
    -------
    C_FI   : float  Fusion island total cost [2010 M$].
    C_heat : float  Heat transfer system cost [2010 M$].
    """
    C_heat      = c_scale_heat * (P_t / P_scale_heat)**x_scale_heat
    C_coils     = (1 + red_sc)   * C_pc
    C_shield    = (1 + ducts)    * C_sg
    C_blanket   = (1 + spare_bl) * C_bl
    C_target    = (1 + spare_tt) * C_tt
    C_auxiliary = (1 + spare_aux)* C_aux
    C_FI = C_heat + C_coils + C_shield + C_blanket + C_target + C_auxiliary
    return (C_FI, C_heat)

def f_annual_OandM_costs_Sheffield(c_scale_OM, P_e, P_scale_OM, x_scale_OM):
    """
    Annual operation & maintenance cost.  Sheffield (2016) Table IV.
    Power-law scaling: C_OM = C_ref * (Pe / P_ref)^x.

    Parameters
    ----------
    c_scale_OM  : float  Reference O&M cost [2010 M$/yr].
    P_e         : float  Net electric power [MWe].
    P_scale_OM  : float  Reference electric power [MWe].
    x_scale_OM  : float  Economy-of-scale exponent [-].

    Returns
    -------
    C_OM : float  Annual O&M cost [2010 M$/yr].
    """
    C_OM = c_scale_OM * (P_e / P_scale_OM)**x_scale_OM
    return C_OM

def f_annual_consumables_costs_Sheffield(CF, T_life, failure_bl, C_bl, Load_bl,
                                         T_load_bl, failure_tt, C_tt, Load_tt,
                                         T_load_tt, f_aaux, C_aux, C_fa):
    """
    Annual consumable costs: blanket + target replacements + aux fraction + fuel.
    Sheffield (2016) Section III.C.

    Parameters
    ----------
    CF         : float  Capacity factor [-].
    T_life     : float  Plant lifetime [yr].
    failure_bl : float  Blanket failure surcharge fraction [-].
    C_bl       : float  Blanket cost [2010 M$].
    Load_bl    : float  Neutron load on blanket [MW/m^2].
    T_load_bl  : float  Blanket max fluence [MW yr/m^2].
    failure_tt : float  Target failure surcharge fraction [-].
    C_tt       : float  Target cost [2010 M$].
    Load_tt    : float  Heat load on targets [MW/m^2].
    T_load_tt  : float  Target max fluence [MW yr/m^2].
    f_aaux     : float  Annual fraction of aux system cost [-].
    C_aux      : float  Aux heating system cost [2010 M$].
    C_fa       : float  Annual fuel cost [2010 M$/yr].

    Returns
    -------
    C_F : float  Annual consumable cost [2010 M$/yr].
    """
    C_ba = f_annual_rep_cost_Sheffield(failure_bl, C_bl, CF,
                                       T_life, Load_bl, T_load_bl)
    C_ta = f_annual_rep_cost_Sheffield(failure_tt, C_tt, CF,
                                       T_life, Load_tt, T_load_tt)
    C_F = C_ba + C_ta + f_aaux * C_aux + C_fa
    return C_F

def f_total_capital_cost_Sheffield(C_D, T_build, contingency):
    """
    Total capital cost including indirect charges and contingency.
    Sheffield (2016) Section III.D.

    C_CO = C_D * f_IND * f_CAPO * (1 + contingency).

    Parameters
    ----------
    C_D         : float  Direct cost [2010 M$].
    T_build     : float  Construction time [yr].
    contingency : float  Contingency fraction [-].

    Returns
    -------
    C_CO  : float  Total capital cost [2010 M$].
    C_ind : float  Indirect charges (C_CO - C_D) [2010 M$].
    """
    f_IND  = f_indirect_charge_capital_Sheffield(T_build)
    f_CAPO = f_interest_charge_construction_Sheffield(T_build)
    C_CO   = C_D * f_IND * f_CAPO * (1 + contingency)
    C_ind  = C_CO - C_D
    return (C_CO, C_ind)

def f_indirect_charge_capital_Sheffield(T_build):
    """
    Indirect cost multiplier: design, engineering, management, ...
    Sheffield (2016) Eq. 16.

    f_IND = 1 + 0.5 * T_build / 8.

    Parameters
    ----------
    T_build : float  Construction time [yr].

    Returns
    -------
    f_IND : float  Indirect charge factor [-].
    """
    f_IND = 1 + 0.5 * T_build / 8
    return f_IND

def f_interest_charge_construction_Sheffield(T_build):
    """
    Interest during construction (IDC) factor.
    Sheffield (2016) Eq. 17.

    f_CAPO = 1.011^(T_build + 0.61).

    Parameters
    ----------
    T_build : float  Construction time [yr].

    Returns
    -------
    f_CAPO : float  IDC factor [-].
    """
    f_CAPO = 1.011**(T_build + 0.61)
    return f_CAPO

def f_annual_rep_cost_Sheffield(failure, C_init, CF, T_life, Load_nom, T_load):
    """
    Annual replacement cost of a single component over the plant lifetime.
    Sheffield (2016) Section III.C.

    Number of replacements = CF * T_life * Load_nom / T_load - 1
    (subtracting the initial set included in CapEx).

    Parameters
    ----------
    failure  : float  Failure surcharge fraction [-].
    C_init   : float  Component initial cost [2010 M$].
    CF       : float  Capacity factor [-].
    T_life   : float  Plant lifetime [yr].
    Load_nom : float  Nominal flux on component [MW/m^2].
    T_load   : float  Maximum tolerable fluence [MW yr/m^2].

    Returns
    -------
    C_a : float  Annualised replacement cost [2010 M$/yr].
    """
    C_others = (CF * T_life * Load_nom / T_load - 1) * C_init / T_life
    C_a = (1 + failure) * C_others
    return C_a


# ══════════════════════════════════════════════════════════════════════════════
#                              SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    SEP  = "─" * 72
    PASS = "  ✓ PASS"
    FAIL = "  ✗ FAIL"

    def _check(label, computed, reference, tol_pct):
        err = abs(computed - reference) / max(abs(reference), 1e-30) * 100
        tag = PASS if err < tol_pct else FAIL
        print(f"  {label:50s} = {computed:12.4f}  "
              f"(ref {reference:12.4f}, err {err:.2f}%){tag}")
        return err < tol_pct

    print("=" * 72)
    print("  D0FUS_cost_functions.py — standalone validation")
    print("=" * 72)

    # ── 1. Capital Recovery Factor ───────────────────────────────────────────
    #   Standard financial formula. Textbook values:
    #   CRF(7%, 40yr) = 0.07501,  CRF(10%, 30yr) = 0.10608.
    print(f"\n{SEP}")
    print("1. Capital Recovery Factor (CRF)")
    print(SEP)
    _check("CRF(r=7%, T=40yr)", f_capital_recovery_factor(0.07, 40), 0.07501, 0.1)
    _check("CRF(r=10%, T=30yr)", f_capital_recovery_factor(0.10, 30), 0.10608, 0.1)

    # ── 2. Availability & Capacity Factor ────────────────────────────────────
    #   Av(8yr op, 2yr rep) = 8/10 = 0.80
    #   CF(Av=0.8, Util=0.9, SS) = 0.72
    print(f"\n{SEP}")
    print("2. Availability & Capacity Factor")
    print(SEP)
    _check("Av(T_op=8, dt_rep=2)", f_pp_availability(8.0, 2.0), 0.80, 0.01)
    _check("CF(Av=0.8, Util=0.9, Dwell=1.0)",
           f_pp_capacity_factor(0.8, 0.9, 1.0), 0.72, 0.01)

    # ── 3. Sheffield model — DEMO-class inputs ───────────────────────────────
    #   Ref: Jo et al., Energies 14 (2021) 6817, Table 3:
    #     R0 ~ 6.2 m, A ~ 3.1, P_fus ~ 2500 MW, Pe ~ 1200 MWe
    #     COE in range 109-140 mills/kWh = 109-140 2010 $/MWh
    #     Capital cost ~ 5000-6000 M$ (2010)
    #
    #   We test with representative DEMO-class parameters.
    #   Expected: COE ~ 100-200 EUR/MWh (after 2010$->2025EUR conversion).
    print(f"\n{SEP}")
    print("3. Sheffield (2016) — DEMO-class smoke test")
    print(f"   Ref: Jo et al., Energies 14 (2021) 6817")
    print(SEP)
    res_shf = f_costs_Sheffield(
        discount_rate     = 0.07,
        contingency       = 0.15,
        T_life            = 40,
        T_build           = 8,
        P_t               = 4150.0,
        P_e               = 1200.0,
        P_aux             = 200.0,
        Gamma_n           = 1.0,
        Util_factor       = 0.85,
        Dwell_factor      = 1.0,
        dt_rep            = 1.5,
        V_FI              = 5000.0,
        V_pc              = 800.0,
        V_sg              = 600.0,
        V_bl              = 400.0,
        S_tt              = 50.0,
        Supra_cost_factor = 2.0,
    )
    T_op, CF, C_CO, COE = res_shf[0], res_shf[1], res_shf[2], res_shf[3]
    print(f"  T_op_limit  = {T_op:.1f} yr")
    print(f"  CF          = {CF:.3f}")
    print(f"  C_invest    = {C_CO:.0f} M EUR  ({C_CO*1e-3:.2f} B EUR)")
    print(f"  COE         = {COE:.1f} EUR/MWh")

    ok_shf = T_op > 0 and 0 < CF < 1 and C_CO > 0 and COE > 0
    # COE should be in 100-300 EUR/MWh range for DEMO-class
    ok_shf &= 50 < COE < 500
    print(f"  COE in [50, 500] EUR/MWh range{PASS if ok_shf else FAIL}")

    # ── 4. Sheffield — indirect cost factors ─────────────────────────────────
    #   f_IND(T_build=8) = 1 + 0.5*8/8 = 1.5
    #   f_CAPO(T_build=8) = 1.011^(8.61) ~ 1.099
    print(f"\n{SEP}")
    print("4. Sheffield — indirect cost factors")
    print(SEP)
    _check("f_IND(T_build=8)", f_indirect_charge_capital_Sheffield(8), 1.50, 0.01)
    _check("f_CAPO(T_build=8)", f_interest_charge_construction_Sheffield(8), 1.099, 1.0)

    # ── 5. Whyte model — ITER-scale smoke test ───────────────────────────────
    #   ITER: S_FW ~ 700 m^2, Gamma_n ~ 0.57 MW/m^2, P_elec ~ 0 (no BoP)
    #   For a power plant extrapolation: S ~ 700 m^2, P_elec = 1000 MWe.
    print(f"\n{SEP}")
    print("5. Whyte (2024) — ITER-scale extrapolation smoke test")
    print(SEP)
    res_wht = f_costs_Whyte(
        F_dpa         = 10.0,
        L_dpa         = 50.0,
        S             = 700.0,
        Util_factor   = 0.85,
        Dwell_factor  = 1.0,
        dt_rep        = 1.5,
        discount_rate = 0.07,
        T_life        = 40,
        Gamma_n       = 1.0,
        P_elec        = 1000.0,
    )
    T_op_w, CF_w, C_inv_w, COE_w = res_wht[0], res_wht[1], res_wht[2], res_wht[3]
    print(f"  T_op_limit  = {T_op_w:.1f} yr")
    print(f"  CF          = {CF_w:.3f}")
    print(f"  C_invest    = {C_inv_w:.0f} M EUR  ({C_inv_w*1e-3:.2f} B EUR)")
    print(f"  COE         = {COE_w:.1f} EUR/MWh")

    ok_wht = T_op_w > 0 and 0 < CF_w < 1 and C_inv_w > 0 and COE_w > 0
    print(f"  All outputs positive{PASS if ok_wht else FAIL}")

    # ── 6. Cross-model consistency ───────────────────────────────────────────
    #   Both models at similar scale should give COE in the same order of
    #   magnitude (100-500 EUR/MWh for FOAK fusion).
    #   Ref: Entler et al. (2018): LCOE ~ 160 $/MWh for EU-DEMO2.
    #        Lindley et al. (2023): LCOE > 150 $/MWh for early designs.
    print(f"\n{SEP}")
    print("6. Cross-model order-of-magnitude consistency")
    print(f"   Ref: Entler et al. (2018); Lindley et al. (2023)")
    print(SEP)
    ratio = COE / COE_w
    print(f"  COE_Sheffield / COE_Whyte = {ratio:.2f}  [expect O(1)]")
    ok_cross = 0.1 < ratio < 10
    print(f"  Ratio in [0.1, 10]{PASS if ok_cross else FAIL}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  All tests completed.")
    print("=" * 72)
