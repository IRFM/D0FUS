"""Tests ported from the ``__main__`` self-check blocks of
``D0FUS_BIB/D0FUS_physical_functions.py`` that contain ``assert`` statements.

Source block: the confinement-scaling registry anchors, a former
``if __name__ == "__main__"`` block removed from the source once ported.
Its ``assert`` statements are ported verbatim; its ``_bench`` rows that
carry a tolerance (which make the self-check fail on violation) are ported
as relative-tolerance asserts.
"""
import numpy as np
import pytest

from D0FUS_BIB.D0FUS_physical_functions import (
    f_Get_fGW_correction,
    f_Get_kappa_convention,
    f_Get_parameter_scaling_law,
)


def rel_err(value, reference):
    return abs(value / reference - 1)


# ── Registry: published exponents stored exactly ─────────────────────────────
# IPB98(y,2): ITER Physics Basis, NF 39 (1999) 2175 / Doyle, NF 47 (2007) S18.
# ITPA20 / ITPA20-IL: Verdoolaege, NF 61 (2021) 076006.
@pytest.mark.parametrize(
    "law, expected",
    [
        ("IPB98(y,2)", (0.0562, 0, 0.19, 0.78, 0.58, 1.97, 0.15, 0.41, 0.93, -0.69)),
        ("ITPA20", (0.0534, 0.362, 0.195, 0.799, 0.354, 1.710, 0.218, 0.2442, 0.976, -0.6687)),
        ("ITPA20-IL", (0.0670, 0.560, 0.302, 0.673, 0, 1.194, -0.134, 0.1473, 1.291, -0.6442)),
    ],
)
def test_scaling_law_registry_exact(law, expected):
    assert f_Get_parameter_scaling_law(law) == expected


def test_kappa_convention():
    laws = ("IPB98(y,2)", "ITPA20", "ITPA20-IL", "ITER89-P")
    assert [f_Get_kappa_convention(law) for law in laws] == ["area", "area", "area", "edge"]


# ── Published anchors (ported from the _bench tables) ────────────────────────
def test_iter89p_prefactor():
    # Yushmanov, NF 30 (1990) 1999, converted to the n19 convention.
    assert rel_err(f_Get_parameter_scaling_law("ITER89-P")[0], 0.048 * 10**-0.1) < 5e-3


def test_tau_ipb98_iter_published_convention():
    # ITER Q=10 point, P = 87 MW total heating, kappa_x = 1.70, n19 = 10.1:
    # 3.7 s [IPB 1999], 8 % tolerance (kappa-convention / P_loss spread).
    C, _ad, aM, ak, ae, aR, aB, an, aI, aP = f_Get_parameter_scaling_law("IPB98(y,2)")
    tau98 = (C * 15**aI * 5.3**aB * 10.1**an * 2.5**aM * 6.2**aR
             * (2 / 6.2)**ae * 1.70**ak * 87.0**aP)
    assert rel_err(tau98, 3.7) < 0.08


def _tau_itpa_iter_point(law):
    # ITER point of Verdoolaege et al. (2021), section 3.2.2.
    C, ad, aM, ak, ae, aR, aB, an, aI, aP = f_Get_parameter_scaling_law(law)
    return (C * 15**aI * 5.3**aB * 10.3**an * 87.0**aP * 6.2**aR
            * 1.48**ad * 1.7**ak * 0.32**ae * 2.5**aM)


@pytest.mark.parametrize(
    "law, published",
    [("ITPA20", 3.067),      # Verdoolaege 2021, Table 16
     ("ITPA20-IL", 2.902)],  # Verdoolaege 2021, Table 10
)
def test_tau_itpa20_iter_point(law, published):
    assert rel_err(_tau_itpa_iter_point(law), published) < 5e-3


def test_tau_auclair2026_iter_baseline():
    # ITER baseline of paper 1 (Table 3): 2.96 s.
    C, _ad, _aM, ak, ae, aR, aB, an, aI, aP = f_Get_parameter_scaling_law("Auclair2026")
    tauA = (C * 15**aI * 5.3**aB * 10.3**an * 6.2**aR
            * 0.32**ae * 1.70**ak * 87.0**aP)
    assert rel_err(tauA, 2.96) < 0.01


def test_auclair2026_fgw_correction():
    # Paper 2: correction at f_GW = 0.85 is exp(-0.49 x 0.33).
    sf, fref = f_Get_fGW_correction("Auclair2026-fGW")
    assert rel_err(np.exp(sf * (0.85 - fref)), np.exp(-0.49 * 0.33)) < 1e-3
