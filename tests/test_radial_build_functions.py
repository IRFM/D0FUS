"""Tests ported from the ``__main__`` self-check blocks of
``D0FUS_BIB/D0FUS_radial_build_functions.py`` that contain ``assert`` statements.

The blocks were removed from the source once ported. One section per former
block. Checks and tolerances are unchanged from the source.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from D0FUS_BIB.D0FUS_parameterization import μ0
from D0FUS_BIB.D0FUS_radial_build_functions import (
    F_CIRCE0D,
    J_non_Cu_Nb3Sn,
    J_non_Cu_NbTi,
    J_non_Cu_REBCO,
    Number_TF_coils,
    _bore_log_integral,
    calculate_E_mag_CS,
    calculate_E_mag_TF,
    f_J_coil_CS,
    f_J_coil_TF,
    get_copper_properties,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel_err(value, reference):
    return abs(value / reference - 1)


# ── TF coil count at the ITER geometry ───────────────────────────────────────
def test_number_tf_coils_iter():
    # D0FUS selects n = 19 against ITER's 18 (port segmentation, Mitchell
    # IEEE TAS 22 (2012)); the achieved ripple must respect δ_adm = 0.5 %.
    n, ripple, _L = Number_TF_coils(6.2, 2.0, 1.4, 0.005, 3.6)
    assert 16 <= n <= 20, n
    assert ripple <= 0.005 + 1e-12, ripple


# ── Superconductor Jc anchors ────────────────────────────────────────────────
def test_jc_nb3sn_iter_strand_class():
    # ITER TF strand spec ≥ 800 A/mm² at 12 T, 4.22 K [Devred SuST 27 (2014)].
    J = float(J_non_Cu_Nb3Sn(12., 4.2, Eps=-0.003)) / 1e6
    assert 700. < J < 1200., J


def test_jc_nbti_textbook():
    # Bottura fit, IEEE TAS 10 (2000) 1054: ≈ 3000 A/mm² at 5 T, 4.2 K.
    J = float(J_non_Cu_NbTi(5., 4.2)) / 1e6
    assert 2600. < J < 3400., J


def test_jc_rebco_senatore_anchor():
    # Fujikura_2019 anchored at Jc(19 T, 4.2 K) = 2000 A/mm² [Senatore 2024].
    J = float(J_non_Cu_REBCO(19., 4.2, dataset="Fujikura_2019")) / 1e6
    assert rel_err(J, 2000.) < 1e-3, J


@pytest.mark.parametrize(
    "jc",
    [J_non_Cu_Nb3Sn,
     J_non_Cu_NbTi,
     lambda B, T: J_non_Cu_REBCO(B, T, dataset="Fujikura_2019")],
    ids=["Nb3Sn", "NbTi", "REBCO"],
)
def test_jc_decreasing_in_B_and_T(jc):
    assert float(jc(10., 4.2)) > float(jc(14., 4.2))
    assert float(jc(8., 4.2)) > float(jc(8., 6.5))


# ── Copper property anchors ──────────────────────────────────────────────────
def test_copper_properties():
    rho273, cv273 = get_copper_properties(273., 0., 100.)
    # ρ_Cu(273 K) = 1.543e-8 Ω·m [NIST Monograph 177 (1992)], 3 %.
    assert rel_err(rho273, 1.543e-8) < 0.03, rho273
    # RRR identity ρ(273 K)/ρ(4 K) = RRR, 2 %.
    rho4, _ = get_copper_properties(4.5, 0., 100.)
    assert abs(rho273 / rho4 / 100. - 1) < 0.02
    # Volumetric heat capacity ≈ 3.45e6 J/(m³·K) [CRC Handbook], 5 %.
    assert rel_err(cv273, 3.45e6) < 0.05, cv273
    # Kohler magnetoresistance.
    assert get_copper_properties(4.5, 12., 100.)[0] > rho4


# ── J_coil identities ────────────────────────────────────────────────────────
def test_j_coil_alpha_identities():
    rng = np.random.default_rng(0)
    err_TF, err_CS = 0.0, 0.0
    for _ in range(500):
        R_ext = rng.uniform(1.0, 6.0)
        c_WP = rng.uniform(0.02, 0.6) * R_ext
        B = rng.uniform(4.0, 25.0)
        J_w = rng.uniform(20e6, 300e6)
        R_sep = R_ext - c_WP
        alpha = 2 * B * R_ext / (μ0 * J_w * (R_ext**2 - R_sep**2))
        err_TF = max(err_TF, abs(f_J_coil_TF(B, R_ext, c_WP) / (alpha * J_w) - 1.0))
        d = rng.uniform(0.05, 1.5)
        B_CS = rng.uniform(4.0, 25.0)
        alpha = B_CS / (μ0 * J_w * d)
        err_CS = max(err_CS, abs(f_J_coil_CS(B_CS, d) / (alpha * J_w) - 1.0))
    assert err_TF < 1e-12, err_TF
    assert err_CS < 1e-12, err_CS


def test_j_coil_degenerate_geometry_returns_nan():
    assert np.isnan(f_J_coil_TF(10.0, 3.0, np.nan))
    assert np.isnan(f_J_coil_TF(10.0, 3.0, 0.0))
    assert np.isnan(f_J_coil_CS(10.0, 0.0))
    assert np.isnan(f_J_coil_CS(np.nan, 0.5))


# ── CIRCE0D vs Lamé thick cylinder ───────────────────────────────────────────
def test_circe0d_lame_identity():
    sr, st, _, r, _ = F_CIRCE0D(100, [1., 2.], [0.], [0.], 100e6, 0., [200e9], 0.3, [1])
    sr_lame = 100e6 * 1. / (4. - 1.) * (1 - 4. / r**2)
    st_lame = 100e6 * 1. / (4. - 1.) * (1 + 4. / r**2)
    assert np.max(np.abs(sr - sr_lame)) / 100e6 < 1e-10
    assert np.max(np.abs(st - st_lame)) / np.max(np.abs(st_lame)) < 1e-10


# ── Stored magnetic energy anchors ───────────────────────────────────────────
def test_e_mag_cs_iter():
    # ITER CS: 6.4 GJ at 13 T [Schultz, 21st SOFE (2005)], 6 %.
    E_CS = calculate_E_mag_CS(13., 1.32, 2.07, 12.)
    assert rel_err(E_CS, 6.4e9) < 0.06, E_CS / 1e9


def test_bore_log_integral_rectangle_exact():
    R = np.array([2.6, 11.5, 11.5, 2.6])
    Z = np.array([-7., -7., 7., 7.])
    assert rel_err(_bore_log_integral(R, Z), 14. * np.log(11.5 / 2.6)) < 1e-12


def test_e_mag_tf_iter():
    # ITER TF: 41 GJ [Mitchell IEEE TAS 18 (2008)], 10 %.
    E_TF = calculate_E_mag_TF(5.3 * 6.2 / 3.1, 2.0, 1.1, 6.2, 0.9, 1.59)
    assert rel_err(E_TF, 41e9) < 0.10, E_TF / 1e9


# ── ITER deck frozen radial-build anchors ────────────────────────────────────
ITER_DECK = os.path.join(REPO_ROOT, "D0FUS_INPUTS", "1_run_ITER.txt")


@pytest.mark.skipif(not os.path.exists(ITER_DECK), reason="ITER reference deck not present")
def test_iter_deck_frozen_radial_build():
    # Anti-drift guard on the frozen 2026-10 coil quantities (< 0.5 %).
    # Run in a fresh subprocess, as in the source block: in-process module
    # state can shift the CS sizing (see the note in the source).
    code = textwrap.dedent("""
        import sys; sys.path.insert(0, %r)
        from D0FUS_EXE.D0FUS_run import load_config_from_file, run
        res = run(load_config_from_file(%r), verbose=0)
        frozen = {44: 0.9385, 45: 0.5029, 46: 0.3656, 51: 0.7254, 1: 12.009}
        names = {44: 'c_TF', 45: 'c_WP', 46: 'c_Nose', 51: 'd_CS', 1: 'B_CS'}
        for i, v in frozen.items():
            assert abs(float(res[i])/v - 1) < 5e-3, (names[i], float(res[i]), v)
    """) % (REPO_ROOT, ITER_DECK)
    env = dict(os.environ, MPLBACKEND="Agg", D0FUS_NO_BANNER="1")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-400:]
