"""Midplane radial build drawn by ``plot_radial_build``: the segments of
``_midplane_build`` tile the midplane from the machine axis to the outer face of
the TF outboard leg. Structural checks only, no frozen values.
"""
import pytest

from D0FUS_BIB.D0FUS_figures import _midplane_build, demo_run_dict, plot_radial_build


@pytest.mark.parametrize("variant", [
    {},                                       # ITER demo, wedging with a TF-CS gap
    {"Choice_Buck_Wedg": "Bucking"},          # no TF-CS gap
    {"Delta_TF": 0.8},                        # outboard clearance for port access
    {"b": 0.15},                              # compressed mandatory layers
    {"b": 0.0},                               # no radial space for the layers
])
def test_midplane_build_tiles_the_midplane(variant):
    run = dict(demo_run_dict(), **variant)
    segs = _midplane_build(run)
    R0, a, b, c_TF = run["R0"], run["a"], run["b"], run["c_TF"]
    Delta_TF = run.get("Delta_TF", 0.0)
    assert segs[0]["R_in"] == 0.0
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt["R_in"] == pytest.approx(prev["R_out"], abs=1e-9)
    assert all(s["R_out"] > s["R_in"] for s in segs)
    plasma = next(s for s in segs if s["role"] == "plasma")
    assert (plasma["R_in"], plasma["R_out"]) == pytest.approx((R0 - a, R0 + a))
    assert segs[-1]["R_out"] == pytest.approx(R0 + a + b + Delta_TF + c_TF)


def test_plot_radial_build_writes_png(tmp_path):
    plot_radial_build(demo_run_dict(), save_dir=str(tmp_path))
    assert (tmp_path / "run_radial_build.png").is_file()
