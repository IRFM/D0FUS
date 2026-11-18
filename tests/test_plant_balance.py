"""Plant electrical balance: pulsed operation with electrical smoothing
(``f_pulsed_electrical_quantities``, M. Fletcher 2026), tritium burn and the
absolute result indices read by the cost evaluator.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from D0FUS_BIB.D0FUS_physical_functions import (
    f_pulsed_electrical_quantities,
    f_tritium_consumption,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITER_DECK = os.path.join(REPO_ROOT, "D0FUS_INPUTS", "1_run_ITER.txt")

ETA, ETA_STORE, T_PLATEAU = 0.33, 0.90, 7200.0


def _balance(P_th, P_var, P_fix, P_cs, D):
    """Smoothing output and the burn and dwell net powers rebuilt from x."""
    out = f_pulsed_electrical_quantities(P_th, T_PLATEAU, D, ETA, P_var,
                                         P_fix, P_cs, ETA_STORE)
    t_dwell = T_PLATEAU * (1 - D) / D
    x = out['x_storage_split']
    burn = ETA * x * P_th - P_var - P_fix
    dwell = ETA * ETA_STORE * (1 - x) * P_th * T_PLATEAU / t_dwell - P_cs - P_fix
    return out, burn, dwell


# ── Electrical smoothing ─────────────────────────────────────────────────────
@pytest.mark.parametrize("P_cs", [0.0, 20.0, 400.0])
def test_smoothing_equalises_burn_and_dwell(P_cs):
    # Same net power in both phases, CS recharge drawn during the dwell
    out, burn, dwell = _balance(2000.0, 150.0, 90.0, P_cs, 0.9)
    assert out['smoothing_feasible']
    np.testing.assert_allclose(dwell, burn, rtol=1e-12)
    np.testing.assert_allclose([out['P_e_net_pulse'], out['P_e_net_dwell'],
                                out['P_e_net_cycle_average']], burn, rtol=1e-12)


def test_turbine_design_power_is_phase_maximum():
    out, _, _ = _balance(2000.0, 150.0, 90.0, 400.0, 0.9)
    assert out['P_th_turbine'] == max(out['P_th_reactor_to_turbine_pulse'],
                                      out['P_th_storage_to_turbine_dwell'])


def test_steady_state_has_no_storage():
    out = f_pulsed_electrical_quantities(2000.0, T_PLATEAU, 1.0, ETA, 150.0,
                                         90.0, 0.0, ETA_STORE)
    assert out['x_storage_split'] == 1.0
    assert out['E_store_th'] == 0.0
    np.testing.assert_allclose(out['P_e_net_cycle_average'], ETA * 2000.0 - 240.0)


def test_unbalanced_loads_fall_back_to_no_storage():
    # Burn loads above the gross burn power: x = 1, cycle average reported
    out = f_pulsed_electrical_quantities(500.0, T_PLATEAU, 0.8, ETA, 300.0,
                                         50.0, 0.0, ETA_STORE)
    assert not out['smoothing_feasible']
    assert out['x_storage_split'] == 1.0
    assert np.isnan(out['P_e_net_smoothed'])
    np.testing.assert_allclose(out['P_e_net_cycle_average'],
                               0.8 * (ETA * 500.0 - 350.0) + 0.2 * (-50.0))


# ── Tritium burn ─────────────────────────────────────────────────────────────
def test_tritium_burn_per_full_power_year():
    # 1 GW.yr / 17.59 MeV = 1.12e28 reactions, x 3.016 u = 56.1 kg
    m_fpy, m_yr = f_tritium_consumption(1000.0, 0.5)
    np.testing.assert_allclose(m_fpy, 56.1)
    np.testing.assert_allclose(m_yr, 28.05)


# ── Result indices read by the cost evaluator ────────────────────────────────
@pytest.mark.skipif(not os.path.exists(ITER_DECK), reason="ITER reference deck not present")
def test_cost_indices_match_named_plant_outputs():
    # evaluate_costs reads run() results by absolute index (_RI). Each plant
    # index must point at the output of the same name in the run dict.
    code = textwrap.dedent("""
        import sys; sys.path.insert(0, %r)
        import numpy as np
        from D0FUS_EXE.D0FUS_run import (load_config_from_file, run, _RI,
                                         _build_run_dict)
        cfg = load_config_from_file(%r)
        res = run(cfg, verbose=0)
        rd = _build_run_dict(cfg, res)
        plant = [k for k, i in _RI.items() if i >= 150]
        assert {'P_th', 'P_th_turbine', 'E_store_th', 'P_recirc_fix'} <= set(plant)
        for k in plant:
            np.testing.assert_allclose(float(res[_RI[k]]), float(rd[k]),
                                       rtol=1e-12, err_msg=k)
    """) % (REPO_ROOT, ITER_DECK)
    env = dict(os.environ, MPLBACKEND="Agg", D0FUS_NO_BANNER="1")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-400:]
