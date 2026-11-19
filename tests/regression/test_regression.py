"""Regression tests: each stored deck must reproduce its stored outputs.

Every value of every output file is compared: numbers to an absolute
tolerance (``DEFAULT_ATOL`` = 1e-8 unless the case sets its own), text
exactly. Lines and keys that record the time, machine or paths of a run are
skipped. Text reports hold their numbers to the printed precision only.

Every comparison is kept, passed or failed, and summarised per file: the
number of comparisons and failures, the largest numeric difference, and each
failed comparison. A failing test shows the summary; ``pytest -rP`` shows it
for passing tests too.

After an intended change of the results, regenerate the references with
``python tests/regression/generate_references.py`` and review the diff.
The tests only read the stored decks and references: each run writes to
pytest's tmp_path, and the test fails if a stored file was changed.
"""
import hashlib

import pytest

import regression_utils as ru

pytestmark = pytest.mark.regression


def _digests(paths):
    digests = {}
    for path in paths:
        with open(path, "rb") as f:
            digests[path] = hashlib.sha256(f.read()).hexdigest()
    return digests


@pytest.mark.parametrize("name", list(ru.CASES))
def test_regression(name, tmp_path):
    expected = ru.reference_files(name)
    if not expected:
        pytest.skip(f"no references for {name}: run generate_references.py")
    stored = [ru.stored_deck(name), *expected.values()]
    before = _digests(stored)

    ru.run_case(name, ru.stored_deck(name), str(tmp_path))
    produced = ru.collect_outputs(str(tmp_path))

    atol = ru.CASES[name]["atol"]
    comparisons = ru.compare_outputs(expected, produced, atol)
    summary = ru.summarize(comparisons, atol)
    print(summary)  # shown with pytest -rP (passed) or -rA (all)

    assert _digests(stored) == before, "the run modified the stored deck or references"
    assert ru.reference_files(name) == expected, "the run added or removed reference files"
    n_failed = sum(not c.passed for c in comparisons)
    assert n_failed == 0, (f"{name}: {n_failed} of {len(comparisons)} comparisons "
                           f"failed\n{summary}")
