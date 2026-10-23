"""
D0FUS cost functions Module
==============================
Cost models for tokamak fusion power plant techno-economic assessment.

Models implemented (GlobalConfig.cost_model):
    'Sheffield'  Sheffield & Milora (2016) generic magnetic fusion reactor
                 cost model, the reference model.
    'Whyte'      Surface-proportional model adapted from Whyte (2024), kept
                 as a lightweight benchmark.

Both models share the plant quantities computed by D0FUS_run.run(): the
component lifetimes and the availability (f_availability_schedule), and the
annual net electricity E_net (f_net_electric_energy). The cost of
electricity is the annual cost divided by E_net:

    COE = 10^6 (annualised capital + annual operating cost) / E_net

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
                 Restates the Sheffield model (Eqs. 8-14).
                 Benchmark: COE 109-140 mills/kWh for R0 ~ 6-7 m.

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

#%% 1. Financial utilities

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

def f_COE(C_annual, E_net):
    """
    Cost of electricity [currency/MWh] = 10^6 C_annual / E_net.

    C_annual : float  Annual cost [M currency/yr].
    E_net    : float  Annual net electricity [MWh/yr].
    NaN when E_net <= 0 (no net electric output over the year).
    """
    if not (E_net > 0.0):
        return np.nan
    return 1e6 * C_annual / E_net

#%% 2. Whyte (2024) adapted model

def f_costs_Whyte(discount_rate, T_life, S_FW, t_life_bl_fpy, f_burn, E_net):
    """
    Techno-economic assessment with an adaptation of the D. Whyte (2024)
    model, kept as a lightweight benchmark of the Sheffield model.

    The investment is proportional to the first-wall surface, scaled on the
    ITER cost. The blanket replacement is the only operating expense. Fuel
    and O&M are neglected.

    The blanket lifetime and the availability are those of D0FUS (dpa model
    and replacement schedule), so that both cost models share the same plant
    operation. With t_bl = L_dpa / (F_dpa Gamma_n) [fpy], the Whyte annual
    replacement cost C_bl CF Gamma_n F_dpa / L_dpa reads C_bl f_burn / t_bl,
    where f_burn = Av x Util x D is the burn fraction of the year (full-power
    years per calendar year).

    Parameters
    ----------
    discount_rate : float  Real discount rate [-].
    T_life        : float  Plant lifetime [yr].
    S_FW          : float  First-wall surface [m^2].
    t_life_bl_fpy : float  Blanket lifetime [fpy].
    f_burn        : float  Burn fraction of the year, Av x Util x D [-].
    E_net         : float  Annual net electricity [MWh/yr].

    Returns
    -------
    dict  [2025 M EUR] C_invest, C_blanket, C_Op_fixed (annualised capital,
          M EUR/yr), C_Op_bl (annual blanket replacement, M EUR/yr),
          and COE [2025 EUR/MWh].
    """
    cur_conv = c_2025EUR_2025USD
    C_invest  = f_unit_cost_scaling(S_FW, C_invest_S_wht) * cur_conv
    C_blanket = f_unit_cost_scaling(S_FW, C_blanket_S_wht) * cur_conv
    C_fixed   = C_invest * f_capital_recovery_factor(discount_rate, T_life)
    C_bl_rep  = C_blanket * f_burn / t_life_bl_fpy
    return dict(C_invest=C_invest, C_blanket=C_blanket,
                C_Op_fixed=C_fixed, C_Op_bl=C_bl_rep,
                COE=f_COE(C_fixed + C_bl_rep, E_net))

#%% 3. Sheffield (2016) model (2010 US$)

def f_costs_Sheffield(discount_rate, contingency, T_life, T_build,
                      P_t, P_t_smoothed, P_e, E_net, P_aux, E_store_th,
                      Av, t_rep_bl_yr, t_rep_div_yr,
                      V_FI, V_pc, V_sg, V_bl, S_tt,
                      Supra_cost_factor):
    """
    Full cost assessment using Sheffield & Milora (2016) model.

    Computes CapEx (fusion island + BoP + thermal store + buildings +
    indirects), OpEx (O&M + replacements + fuel + waste) and COE.
    All internal calculations in 2010 USD, converted to 2025 EUR on output.

    Ref: Sheffield & Milora, Fus. Sci. Technol. 70, 14-35 (2016).
    Benchmark: Jo et al., Energies 14, 6817 (2021), COE 109-140 mills/kWh.

    Deviations from Sheffield & Milora, kept on purpose (their model as
    restated by Jo et al. 2021, Eqs. 8 and 12-14):
      - The first blanket and divertor are in the fusion-island direct
        cost, so they carry the indirect charges, the interest during
        construction and the contingency. Sheffield & Milora charge them
        every year through the fixed charge rate, in the fuel cycle cost.
      - C_waste_sfd = 5 $/MWh is added to the COE. Sheffield & Milora add
        0.5 mill/kWh for waste and 1.0 mill/kWh for decommissioning, that
        is 1.5 $/MWh.
      - Supra_cost_factor multiplies c_pc_sfd, which is already the unit
        cost of a superconducting coil set.
      - The divertor target area S_tt is an input (evaluate_costs takes
        f_div_area_fraction x plasma surface).
      - The COE divides by the annual net electricity E_net, which counts
        the fixed loads (cryoplant, house load) over the whole year,
        outages included.
      - Pulsed plant: the BoP thermal scaling uses the turbine-side thermal
        power P_t_smoothed, and the thermal store is costed (c_store_sfd).
        The turbine term keeps the net P_e, on which Sheffield & Milora
        calibrated it (900 M$ at 1200 MWe).

    Parameters
    ----------
    discount_rate  : float  Real discount rate [-].
    contingency    : float  Contingency fraction [-].
    T_life         : int    Plant operational lifetime [yr].
    T_build        : int    Construction time [yr].
    P_t            : float  Thermal power during the burn [MWth]
                            (heat transfer system).
    P_t_smoothed   : float  Thermal power to the turbine, cycle average
                            through the thermal store [MWth]. P_t in steady
                            state.
    P_e            : float  Net electric power, cycle average during
                            operation [MWe], >= 0 (BoP and O&M scalings).
    E_net          : float  Annual net electricity [MWh/yr]. The COE is NaN
                            when E_net <= 0.
    P_aux          : float  Auxiliary heating power to plasma [MW].
    E_store_th     : float  Thermal store capacity [MWh_th], 0 in steady state.
    Av             : float  Plant availability [-].
    t_rep_bl_yr    : float  Operating calendar time between two blanket
                            replacements [yr] (f_replacement_intervals).
    t_rep_div_yr   : float  Same for the divertor [yr].
    V_FI           : float  Fusion island bounding volume [m^3].
    V_pc           : float  Primary coil volume (TF + CS) [m^3].
    V_sg           : float  Shield and gaps volume [m^3].
    V_bl           : float  Blanket volume [m^3].
    S_tt           : float  Divertor target surface [m^2].
    Supra_cost_factor : float  SC coil cost multiplier vs Cu [-].

    Returns
    -------
    dict  C_invest, C_ind, component costs (C_reac_pc, C_reac_bl, C_reac_sg,
          C_reac_tt, C_syst_heat, C_syst_aux, C_syst_BOP, C_syst_store,
          C_syst_other) [2025 M EUR], annual costs C_Op_OM, C_Op_F
          [2025 M EUR/yr], C_Op_waste and COE [2025 EUR/MWh].
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
    C_st  = f_unit_cost_scaling(E_store_th, c_store_sfd)

    # CapEx: fusion island -> direct cost -> total capital cost
    (C_FI, C_heat) = f_fusion_island_cost_Sheffield(C_heat_sfd, P_t, P_heat_sfd,
                                                    x_heat_sfd, red_pc_sfd, C_pc,
                                                    ducts_sg_sfd, C_sg, spare_bl_sfd,
                                                    C_bl, spare_tt_sfd, C_tt,
                                                    spare_aux_sfd, C_aux)
    (C_D, C_BOP_tot, C_syst) = f_direct_cost_Sheffield(C_BoP_sfd, C_tur_sfd,
                                                       P_tur_sfd, P_e, P_t_smoothed,
                                                       P_BoP_sfd, x_BoP_sfd,
                                                       C_bld_sfd, V_bld_sfd,
                                                       x_bld_sfd, V_FI, C_FI, C_st)
    (C_CO, C_ind) = f_total_capital_cost_Sheffield(C_D, T_build, contingency)

    # OpEx [2010 M$/yr]
    C_F = f_annual_consumables_costs_Sheffield(T_life, Av,
                                               0.1, C_bl, t_rep_bl_yr,
                                               0.2, C_tt, t_rep_div_yr,
                                               0.1, C_aux, C_fa_sfd)
    C_OM = f_annual_OandM_costs_Sheffield(C_OM_sfd, P_e, P_OM_sfd, x_OM_sfd)

    # Cost of electricity [2010 $/MWh], NaN without net electricity
    COE = f_COE(C_CO * F_CRO + C_F + C_OM, E_net) + C_waste_sfd

    # All outputs in 2025 EUR
    return dict(
        C_invest     = C_CO * cur_conv,
        C_ind        = C_ind * cur_conv,
        C_reac_pc    = C_pc  * (1+red_pc_sfd)    * cur_conv,
        C_reac_sg    = C_sg  * (1+ducts_sg_sfd)  * cur_conv,
        C_reac_bl    = C_bl  * (1+spare_bl_sfd)  * cur_conv,
        C_reac_tt    = C_tt  * (1+spare_tt_sfd)  * cur_conv,
        C_syst_aux   = C_aux * (1+spare_aux_sfd) * cur_conv,
        C_syst_heat  = C_heat    * cur_conv,
        C_syst_BOP   = C_BOP_tot * cur_conv,
        C_syst_store = C_st      * cur_conv,
        C_syst_other = C_syst    * cur_conv,
        C_Op_F       = C_F   * cur_conv,
        C_Op_OM      = C_OM  * cur_conv,
        C_Op_waste   = C_waste_sfd * cur_conv,
        COE          = COE   * cur_conv)

def f_direct_cost_Sheffield(c_BOPe_cst, c_BOPe_scale, P_BOPe_scale, P_e,
                            P_t, P_BOPth_scale, x_BOPth_scale,
                            c_syst_scale, V_syst_scale, x_syst_scale,
                            V_FI, C_FI, C_store=0.0):
    """
    Direct capital cost: fusion island + BoP + thermal store + buildings/
    auxiliaries. Sheffield (2016) Table IV, plus the thermal store.

    Parameters
    ----------
    c_BOPe_cst, c_BOPe_scale, P_BOPe_scale : BoP electric cost scaling.
    P_e           : float  Net electric power [MWe].
    P_t           : float  Thermal power to the turbine [MWth].
    P_BOPth_scale : float  BoP thermal scaling reference [MWth].
    x_BOPth_scale : float  BoP thermal economy-of-scale exponent [-].
    c_syst_scale  : float  Auxiliary systems reference cost [2010 M$].
    V_syst_scale  : float  Auxiliary systems reference volume [m^3].
    x_syst_scale  : float  Auxiliary systems exponent [-].
    V_FI          : float  Fusion island volume [m^3].
    C_FI          : float  Fusion island cost [2010 M$].
    C_store       : float  Thermal store cost [2010 M$].

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
    C_D       = C_BOP_tot + C_store + C_syst + C_FI
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

def f_annual_consumables_costs_Sheffield(T_life, Av, failure_bl, C_bl, t_rep_bl_yr,
                                         failure_tt, C_tt, t_rep_div_yr,
                                         f_aaux, C_aux, C_fa):
    """
    Annual consumable costs: blanket + divertor replacements + aux fraction + fuel.
    Sheffield (2016) Section III.C.

    Parameters
    ----------
    T_life        : float  Plant lifetime [yr].
    Av            : float  Plant availability [-].
    failure_bl    : float  Blanket failure surcharge fraction [-].
    C_bl          : float  Blanket cost [2010 M$].
    t_rep_bl_yr   : float  Operating calendar time between blanket replacements [yr].
    failure_tt    : float  Divertor failure surcharge fraction [-].
    C_tt          : float  Divertor target cost [2010 M$].
    t_rep_div_yr  : float  Operating calendar time between divertor replacements [yr].
    f_aaux        : float  Annual fraction of aux system cost [-].
    C_aux         : float  Aux heating system cost [2010 M$].
    C_fa          : float  Annual fuel cost [2010 M$/yr].

    Returns
    -------
    C_F : float  Annual consumable cost [2010 M$/yr].
    """
    C_ba = f_annual_rep_cost_Sheffield(failure_bl, C_bl, T_life, t_rep_bl_yr, Av)
    C_ta = f_annual_rep_cost_Sheffield(failure_tt, C_tt, T_life, t_rep_div_yr, Av)
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

def f_annual_rep_cost_Sheffield(failure, C_init, T_life, t_rep_yr, Av=1.0):
    """
    Annual replacement cost of a single component over the plant lifetime.
    Sheffield (2016) Section III.C.

    The component is replaced after t_rep_yr calendar years of operation,
    and the plant operates a fraction Av of the time, so that
    N_rep = Av T_life / t_rep_yr sets are used over the plant life, the
    first one included (Sheffield: f_av N Gamma_n / F_blk, Jo et al. 2021
    Eq. 13). The first set is in the capital cost:

        C_a = (1 + failure) * C_init * max(N_rep - 1, 0) / T_life

    Parameters
    ----------
    failure  : float  Failure surcharge fraction [-].
    C_init   : float  Component initial cost [2010 M$].
    T_life   : float  Plant lifetime [yr].
    t_rep_yr : float  Operating calendar time between replacements [yr]
                      (np.inf: never replaced).
    Av       : float  Plant availability [-].

    Returns
    -------
    C_a : float  Annualised replacement cost [2010 M$/yr].
    """
    N_rep = Av * T_life / t_rep_yr
    C_a = (1 + failure) * max(N_rep - 1.0, 0.0) * C_init / T_life
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

    # ── 2. Replacement count with availability ───────────────────────────────
    #   t_rep = 5 yr of operation, Av = 0.8, T_life = 40 yr:
    #   N_rep = 0.8 * 40 / 5 = 6.4 sets, 5.4 bought after the first one.
    #   C_a = 1.1 * 5.4 * 100 / 40 = 14.85 M$/yr.
    print(f"\n{SEP}")
    print("2. Replacement count with availability")
    print(SEP)
    _check("C_a(f=0.1, C=100, T=40, t_rep=5, Av=0.8)",
           f_annual_rep_cost_Sheffield(0.1, 100.0, 40.0, 5.0, 0.8), 14.85, 0.01)
    _check("C_a(never replaced)",
           f_annual_rep_cost_Sheffield(0.1, 100.0, 40.0, np.inf, 0.8) + 1.0, 1.0, 0.01)

    # ── 3. Sheffield model — DEMO-class inputs ───────────────────────────────
    #   Ref: Jo et al., Energies 14 (2021) 6817:
    #     R0 ~ 6.2 m, A ~ 3.1, P_fus ~ 2500 MW, Pe ~ 1200 MWe
    #     COE in range 109-140 mills/kWh = 109-140 2010 $/MWh
    #   Steady state, operating fraction 0.7, fixed loads 100 MWe all year.
    #   Expected: COE ~ 100-200 EUR/MWh (after 2010$->2025EUR conversion).
    print(f"\n{SEP}")
    print("3. Sheffield (2016) — DEMO-class smoke test")
    print(f"   Ref: Jo et al., Energies 14 (2021) 6817")
    print(SEP)
    _f_op, _P_e, _P_fix = 0.70, 1200.0, 100.0
    _E_net = 8760.0 * (_f_op * _P_e - (1.0 - _f_op) * _P_fix)
    res_shf = f_costs_Sheffield(
        discount_rate  = 0.07,
        contingency    = 0.15,
        T_life         = 40,
        T_build        = 8,
        P_t            = 4150.0,
        P_t_smoothed   = 4150.0,
        P_e            = _P_e,
        E_net          = _E_net,
        P_aux          = 200.0,
        E_store_th     = 0.0,
        Av             = 0.80,
        t_rep_bl_yr    = 5.0,
        t_rep_div_yr   = 2.5,
        V_FI           = 5000.0,
        V_pc           = 800.0,
        V_sg           = 600.0,
        V_bl           = 400.0,
        S_tt           = 50.0,
        Supra_cost_factor = 2.0,
    )
    COE, C_CO = res_shf['COE'], res_shf['C_invest']
    print(f"  E_net       = {_E_net*1e-6:.3f} TWh/yr")
    print(f"  C_invest    = {C_CO:.0f} M EUR  ({C_CO*1e-3:.2f} B EUR)")
    print(f"  COE         = {COE:.1f} EUR/MWh")
    ok_shf = C_CO > 0 and 50 < COE < 500
    print(f"  COE in [50, 500] EUR/MWh range{PASS if ok_shf else FAIL}")
    print(f"  COE undefined without net electricity"
          f"{PASS if np.isnan(f_COE(1.0, -1.0)) else FAIL}")

    # ── 4. Sheffield — indirect cost factors ─────────────────────────────────
    #   f_IND(T_build=8) = 1 + 0.5*8/8 = 1.5
    #   f_CAPO(T_build=8) = 1.011^(8.61) ~ 1.099
    print(f"\n{SEP}")
    print("4. Sheffield — indirect cost factors")
    print(SEP)
    _check("f_IND(T_build=8)", f_indirect_charge_capital_Sheffield(8), 1.50, 0.01)
    _check("f_CAPO(T_build=8)", f_interest_charge_construction_Sheffield(8), 1.099, 1.0)

    # ── 5. Whyte model — same plant ──────────────────────────────────────────
    #   S ~ 700 m^2 (ITER-scale first wall), blanket 5 fpy, f_burn = 0.7.
    #   Annual blanket replacement = C_blanket * 0.7 / 5.
    print(f"\n{SEP}")
    print("5. Whyte (2024) — same plant")
    print(SEP)
    res_wht = f_costs_Whyte(discount_rate=0.07, T_life=40, S_FW=700.0,
                            t_life_bl_fpy=5.0, f_burn=0.7, E_net=_E_net)
    _check("C_Op_bl = C_blanket * f_burn / t_bl",
           res_wht['C_Op_bl'], res_wht['C_blanket'] * 0.7 / 5.0, 1e-6)
    print(f"  C_invest    = {res_wht['C_invest']:.0f} M EUR")
    print(f"  COE         = {res_wht['COE']:.1f} EUR/MWh")

    # ── 6. Cross-model consistency ───────────────────────────────────────────
    #   Both models at similar scale should give COE in the same order of
    #   magnitude (100-500 EUR/MWh for FOAK fusion).
    #   Ref: Entler et al. (2018): LCOE ~ 160 $/MWh for EU-DEMO2.
    #        Lindley et al. (2023): LCOE > 150 $/MWh for early designs.
    print(f"\n{SEP}")
    print("6. Cross-model order-of-magnitude consistency")
    print(f"   Ref: Entler et al. (2018); Lindley et al. (2023)")
    print(SEP)
    ratio = COE / res_wht['COE']
    print(f"  COE_Sheffield / COE_Whyte = {ratio:.2f}  [expect O(1)]")
    ok_cross = 0.1 < ratio < 10
    print(f"  Ratio in [0.1, 10]{PASS if ok_cross else FAIL}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  All tests completed.")
    print("=" * 72)
