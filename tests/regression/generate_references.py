"""Regenerate the regression references of tests/regression.

Copies each deck of D0FUS_INPUTS into <case>/input, runs D0FUS on that copy
and stores its outputs in <case>/output. Run it only when a change of the
results is intended, then review the diff of the reference files.

    python tests/regression/generate_references.py            # every case
    python tests/regression/generate_references.py run scan   # some cases
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression_utils as ru  # noqa: E402


def generate(name):
    deck = ru.stored_deck(name)
    os.makedirs(os.path.dirname(deck), exist_ok=True)
    with open(os.path.join(ru.INPUTS, ru.CASES[name]["deck"]), encoding="utf-8") as f:
        text = ru.apply_overrides(f.read(), ru.CASES[name]["overrides"])
    with open(deck, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with tempfile.TemporaryDirectory() as tmp:
        ru.run_case(name, deck, tmp)
        ru.store_outputs(name, tmp)
    print(f"{name}: {len(ru.reference_files(name))} reference files written")


if __name__ == "__main__":
    names = sys.argv[1:] or list(ru.CASES)
    unknown = [n for n in names if n not in ru.CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; choose from {list(ru.CASES)}")
    for name in names:
        generate(name)
