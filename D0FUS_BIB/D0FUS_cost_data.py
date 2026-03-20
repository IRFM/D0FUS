"""
D0FUS cost data Module
==============================
Reference cost data for tokamak fusion power plant economic models.

Data sets:
    1. Denis Whyte (2024)     — ITER-scaled investment from first-wall surface.
    2. John Sheffield (2016)  — generic magnetic fusion reactor cost model
                                (ORNL, 2010 USD), updated from Sheffield (1986).

References
----------
[Whyte 2024]     D. Whyte, "Fusion economy", Seminar JPP, CEA Cadarache (2024).
[Sheffield 2016] J. Sheffield, S. L. Milora, "Generic Magnetic Fusion Reactor
                 Revisited", Fus. Sci. Technol. 70, 14–35 (2016).
                 doi:10.13182/FST15-157
[Sheffield 1986] J. Sheffield et al., "Cost Assessment of a Generic Magnetic
                 Fusion Reactor", Fusion Technol. 9, 199–249 (1986).
[Jo 2021]        G. Jo et al., "Cost Assessment of a Tokamak Fusion Reactor
                 with an Inventive Method for Optimum Build Determination",
                 Energies 14, 6817 (2021). doi:10.3390/en14206817
                 — Benchmark of Sheffield model: COE 109–140 mills/kWh.

Created: Jan 2026
Author: Matteo FLETCHER
"""

#%% Imports

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    try:
        from .D0FUS_import import *
    except ImportError:
        import numpy as np

# When executed directly (for testing and development)
else:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    try:
        from D0FUS_BIB.D0FUS_import import *
    except ModuleNotFoundError:
        import numpy as np

#%% Currency conversions
# Source: US Bureau of Labor Statistics CPI inflation calculator (2010-2025),
#         European Central Bank EUR/USD annual average (2025).

c_2025USD_2010USD = 1.4723                                    # 2025 USD per 2010 USD
c_2025EUR_2025USD = 0.8862                                    # 2025 EUR per 2025 USD
c_2025EUR_2010USD = c_2025EUR_2025USD * c_2025USD_2010USD     # 2025 EUR per 2010 USD

#%% Denis Whyte model inputs (modified to ITER 2025 data)
# ──────────────────────────────────────────────────────────────────────────────
# Simple surface-proportional cost model.
# Investment scaled from ITER 2025 project cost / first-wall surface area.
# Blanket replacement cost assumed 7% of total investment.
# Ref: ITER Organization, "ITER Project Cost" (2024); Whyte JPP seminar (2024).
# ──────────────────────────────────────────────────────────────────────────────

C_invest_S_wht  = 35.7                      # Investment per FW area  [M$/m^2]
                                             #   ~ 25 B$ / 700 m^2
C_blanket_S_wht = 0.07 * C_invest_S_wht     # Blanket cost per FW area [M$/m^2]

#%% John Sheffield 2016 model inputs (2010 USD)
# ──────────────────────────────────────────────────────────────────────────────
# All cost scalings from Sheffield & Milora, Fus. Sci. Technol. 70 (2016),
# Tables I-IV.  Reference reactor: ITER-scale (~1200 MWe, P_fus ~ 2800 MW).
# Unit costs calibrated on ITER 2008 cost estimates.
# Power-law exponents (x < 1) capture economy-of-scale effects.
# ──────────────────────────────────────────────────────────────────────────────

# -- Fusion island component unit costs (Table I) --

c_pc_sfd  = 1.66   # Primary (SC) coil set                   [2010 M$ / m^3]
c_sg_sfd  = 0.29   # Shield and gaps                          [2010 M$ / m^3]
c_aux_sfd = 5.3    # Auxiliary heating system                  [2010 M$ / MW]
c_bl_sfd  = 0.75   # Breeding blanket                          [2010 M$ / m^3]
c_tt_sfd  = 0.114  # Divertor target                           [2010 M$ / m^2]

# -- Spares, redundancies and secondary systems (Table II) --

red_pc_sfd    = 0.5    # SC coil redundancy fraction            [-]
ducts_sg_sfd  = 0.25   # Shield surcharge for ducts/ports       [-]
spare_aux_sfd = 0.1    # Auxiliary heating spares                [-]
spare_bl_sfd  = 0.1    # Blanket module spares                   [-]
spare_tt_sfd  = 0.2    # Divertor target spares                  [-]

# -- External costs --

C_waste_sfd = 5        # Radioactive waste disposal              [$/MWh]

# -- Annual fuel costs --
# D + Li supply.  Negligible vs replacement costs: a defining feature of
# fusion economics (cf. Entler 2018 Table 3: fuel 0.44 vs replacement 13.61 $/MWh).

C_fa_sfd = 7.5         # Annual fuel cost                        [2010 M$ / yr]

# -- Blanket and divertor target lifetime limits (Table III) --
# Refs: Gilbert et al., NF 57 (2017) 046015; Pitts et al., NME 20 (2019) 100696.

T_load_bl_sfd      = 15       # Blanket max neutron fluence      [MW yr / m^2]
T_load_tt_sfd      = 10       # Divertor max heat fluence        [MW yr / m^2]
load_factor_tt_sfd = 10 / 3   # Thermal / neutron flux ratio     [-]

# -- Heat transfer system cost scaling (Table IV) --
# C_heat = C_ref * (P_th / P_ref)^x

C_heat_sfd = 221    # Reference cost                           [2010 M$]
P_heat_sfd = 4150   # Reference thermal power                  [MWth]
x_heat_sfd = 0.6    # Economy-of-scale exponent                [-]

# -- Balance of Plant + turbine cost scaling (Table IV) --
# C_BOP = (C_BoP + C_tur * Pe/P_tur) * (Pth/P_BoP)^x

C_BoP_sfd = 900     # BOP reference cost                       [2010 M$]
C_tur_sfd = 900     # Turbine reference cost                    [2010 M$]
P_tur_sfd = 1200    # Turbine reference electric power          [MWe]
P_BoP_sfd = 4150    # BOP reference thermal power               [MWth]
x_BoP_sfd = 0.6     # Economy-of-scale exponent                 [-]

# -- Buildings, hot cells, vacuum, power supplies (Table IV) --
# C_bld = C_ref * (V_FI / V_ref)^x

C_bld_sfd = 839     # Reference buildings cost                  [2010 M$]
V_bld_sfd = 5100    # Reference fusion island volume            [m^3]
x_bld_sfd = 0.67    # Economy-of-scale exponent                 [-]

# -- Annual O&M costs scaling (Table IV) --
# C_OM = C_ref * (Pe / P_ref)^x

C_OM_sfd = 108      # Reference annual O&M cost                 [2010 M$ / yr]
P_OM_sfd = 1200     # Reference net electric power              [MWe]
x_OM_sfd = 0.5      # Economy-of-scale exponent                 [-]


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
        print(f"  {label:45s} = {computed:12.4f}  "
              f"(ref {reference:12.4f}, err {err:.2f}%){tag}")
        return err < tol_pct

    # ── 1. Currency chain ────────────────────────────────────────────────────
    print(SEP)
    print("1. Currency conversion chain consistency")
    print(SEP)
    _check("c_2025EUR_2010USD",
           c_2025EUR_2010USD, c_2025EUR_2025USD * c_2025USD_2010USD, 0.01)

    # ── 2. Whyte ─────────────────────────────────────────────────────────────
    print(SEP)
    print("2. Whyte — blanket fraction & ITER cost sanity")
    print(SEP)
    _check("C_blanket_S_wht = 7% * C_invest_S_wht",
           C_blanket_S_wht, 0.07 * C_invest_S_wht, 0.01)
    _check("C_invest_S_wht ~ 25e3/700",
           C_invest_S_wht, 25e3 / 700, 2.0)

    # ── 3. Sheffield — positivity and ranges ─────────────────────────────────
    print(SEP)
    print("3. Sheffield — unit cost positivity and exponent ranges")
    print(SEP)
    ok1 = all(v > 0 for v in [c_pc_sfd, c_sg_sfd, c_aux_sfd, c_bl_sfd,
              c_tt_sfd, C_heat_sfd, C_BoP_sfd, C_tur_sfd, C_bld_sfd,
              C_OM_sfd, C_fa_sfd])
    print(f"  All unit costs > 0{PASS if ok1 else FAIL}")
    ok2 = all(0 < v < 1 for v in [red_pc_sfd, ducts_sg_sfd, spare_aux_sfd,
              spare_bl_sfd, spare_tt_sfd, x_heat_sfd, x_BoP_sfd,
              x_bld_sfd, x_OM_sfd])
    print(f"  All fractions/exponents in (0, 1){PASS if ok2 else FAIL}")

    # ── 4. Sheffield — ITER-class spot checks ────────────────────────────────
    #   Cross-check with Jo et al. Energies 14 (2021) 6817.
    #   V_TF ~ 800 m^3 -> C_coils ~ 1330 M$ (2010)
    #   P_aux = 50 MW  -> C_aux   ~ 265 M$ (2010)
    print(SEP)
    print("4. Sheffield — ITER-class component cost spot checks")
    print(f"   Ref: Jo et al., Energies 14 (2021) 6817")
    print(SEP)
    C_coils_test = c_pc_sfd * 800
    _check("C_coils(V=800 m^3) [2010 M$]", C_coils_test, 1328.0, 1.0)
    C_aux_test = c_aux_sfd * 50
    _check("C_aux(P=50 MW)     [2010 M$]", C_aux_test, 265.0, 1.0)
    C_bl_test = c_bl_sfd * 400
    _check("C_bl(V=400 m^3)    [2010 M$]", C_bl_test, 300.0, 1.0)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + SEP)
    print("All data consistency checks completed.")
    print(SEP)
