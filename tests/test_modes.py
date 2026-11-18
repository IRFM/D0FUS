"""Smoke tests of the five execution modes through the command line.

Each test runs a reference ITER deck shrunk to a few evaluations (deck keys
rewritten, values free to change) and checks only that the mode completes
and writes its outputs. No result value is frozen.
"""
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(REPO_ROOT, "D0FUS_INPUTS")

# deck, output folder, {key: shrunk value}
CASES = [
    ("1_run_ITER.txt", "run", {}),
    ("2_scan_ITER.txt", "scan", {"a": "[1.8, 2.2, 3]", "R0": "[5.8, 6.6, 3]"}),
    ("3_genetic_ITER.txt", "genetic", {"population_size": "4", "generations": "2",
                                       "n_workers": "1", "make_gif": "False",
                                       "local_refine_method": "none"}),
    ("4_uncertainty_ITER.txt", "uncertainty", {"n_samples": "4"}),
    ("5_popcon_ITER.txt", "popcon", {"nbar_line": "[0.6, 1.2, 4]", "Tbar": "[6.0, 12.0, 4]"}),
]


def _shrink(text, values):
    for key, value in values.items():
        # A range replaces a range only: POPCON decks also hold the scalar Tbar
        old = r"\[[^\]\n]*\]" if value.startswith("[") else r"[^#\n]*"
        text, n = re.subn(rf"^{key}\s*=\s*{old}", f"{key} = {value}  ", text, flags=re.M)
        assert n == 1, f"key {key} not found once in the deck"
    return text


@pytest.mark.parametrize("deck, folder, values", CASES, ids=[c[1] for c in CASES])
def test_mode_runs(tmp_path, deck, folder, values):
    path = os.path.join(INPUTS, deck)
    if not os.path.exists(path):
        pytest.skip(f"{deck} not present")
    with open(path, encoding="utf-8") as f:
        text = _shrink(f.read(), values)
    small = tmp_path / deck
    small.write_text(text, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg", D0FUS_NO_BANNER="1")
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "D0FUS.py"), str(small),
         "--no-figures", "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=900, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
    produced = [f for _, _, files in os.walk(tmp_path / "out" / folder) for f in files]
    assert produced, f"no output written under {folder}/"
