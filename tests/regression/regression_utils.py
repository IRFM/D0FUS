"""Shared helpers of the regression tests: run a stored deck, compare outputs.

Layout of each case folder under ``tests/regression``::

    <case>/input/<deck>.txt     the input deck, as run
    <case>/output/...           the reference outputs it produced

The timestamped result folder a mode writes (``Run_D0FUS_<stamp>`` and so on)
is dropped from the stored paths, so references and fresh runs line up file
by file. ``generate_references.py`` rewrites the references.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

REGRESSION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(REGRESSION_DIR))
INPUTS = os.path.join(REPO_ROOT, "D0FUS_INPUTS")

DEFAULT_ATOL = 1e-7

# name: deck in D0FUS_INPUTS, deck keys rewritten in the stored copy (to keep
# the run short), extra command line arguments, absolute tolerance
CASES = {
    "run": dict(deck="1_run_ITER.txt", overrides={}, args=[], atol=DEFAULT_ATOL),
    "scan": dict(deck="2_scan_ITER.txt", overrides={}, args=[], atol=DEFAULT_ATOL),
    # The genetic deck draws its seed at random: fix it for a reproducible run
    "genetic": dict(deck="3_genetic_ITER.txt",
                    overrides={"population_size": "16", "generations": "3",
                               "n_workers": "1", "make_gif": "False",
                               "local_refine_top_k": "1",
                               "local_refine_max_iters": "20", "seed": "1"},
                    args=[], atol=DEFAULT_ATOL),
    # 12 model combinations x n_samples draws
    "uncertainty": dict(deck="4_uncertainty_ITER.txt", overrides={"n_samples": "20"},
                        args=[], atol=DEFAULT_ATOL),
    "popcon": dict(deck="5_popcon_ITER.txt", overrides={}, args=[], atol=DEFAULT_ATOL),
}


def apply_overrides(text, overrides):
    """Rewrite the scalar deck keys of ``overrides``, appending absent ones."""
    for key, value in overrides.items():
        new = f"{key} = {value}  # regression override"
        text, n = re.subn(rf"^{key}\s*=\s*[^#\n]*(#.*)?$", new, text, flags=re.M)
        assert n <= 1, f"key {key} found {n} times in the deck"
        if n == 0:
            text = text.rstrip("\n") + f"\n{new}\n"
    return text


# Folder names that carry a run timestamp, e.g. Run_D0FUS_20261119_160839
_STAMPED = re.compile(r"^[A-Za-z]+_D0FUS_\d{8}_\d{6}$")

# Input decks that each mode writes beside its results (a copy of the deck
# run, or the deck of the optimised design in the run/ folder of a genetic
# study): left out of the references at any depth.
_DECK_COPIES = {"input_parameters.txt", "scan_parameters.txt", "deck_copy.txt"}

# Report lines that change from one run to the next (time, machine, paths)
_VOLATILE_LINE = re.compile(
    r"^[#\s]*(Code version|Date / time|Git commit|Python|Platform|Input deck|"
    r"Output folder|Run timestamp|Input file)\s*:|^[#\s]*Generated on\b", re.I)

# JSON keys that change from one run to the next
_VOLATILE_KEYS = {"timestamp", "input_file", "output_dir", "output_folder",
                  "code_version", "git_commit", "python", "platform"}

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def case_dir(name):
    return os.path.join(REGRESSION_DIR, name)


def stored_deck(name):
    return os.path.join(case_dir(name), "input", CASES[name]["deck"])


def run_case(name, deck, out_dir):
    """Run D0FUS on ``deck`` without figures, writing under ``out_dir``.

    Returns the folder holding the mode's outputs, with every timestamped
    level removed (see ``collect_outputs``).
    """
    env = dict(os.environ, MPLBACKEND="Agg", D0FUS_NO_BANNER="1")
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "D0FUS.py"), deck,
         "--no-figures", "--output-dir", out_dir, *CASES[name]["args"]],
        capture_output=True, text=True, timeout=1800, env=env, cwd=REPO_ROOT)
    if result.returncode != 0:
        lines = (result.stdout + result.stderr).splitlines()
        raise RuntimeError(f"D0FUS failed on {name}:\n" + "\n".join(lines[-30:]))
    return out_dir


def collect_outputs(out_dir):
    """Map each output file to its path with the timestamped folders dropped.

    Returns {relative path: absolute path}. The top level is the mode folder
    (run/, scan/, ...) written by D0FUS, which is dropped as well. Copies of
    the input decks are left out.
    """
    files = {}
    for root, _, names in os.walk(out_dir):
        for fname in names:
            full = os.path.join(root, fname)
            parts = os.path.relpath(full, out_dir).split(os.sep)[1:]
            rel = "/".join(p for p in parts if not _STAMPED.match(p))
            if fname in _DECK_COPIES:
                continue
            assert rel not in files, f"two outputs map to {rel}"
            files[rel] = full
    return files


def store_outputs(name, out_dir):
    """Replace the stored references of ``name`` by the files under ``out_dir``."""
    ref = os.path.join(case_dir(name), "output")
    shutil.rmtree(ref, ignore_errors=True)
    for rel, src in collect_outputs(out_dir).items():
        dst = os.path.join(ref, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def reference_files(name):
    ref = os.path.join(case_dir(name), "output")
    files = {}
    for root, _, names in os.walk(ref):
        for fname in names:
            full = os.path.join(root, fname)
            files[os.path.relpath(full, ref).replace(os.sep, "/")] = full
    return files


# ── comparison ──────────────────────────────────────────────────────────────

@dataclass
class Comparison:
    """One compared value: a number, a piece of text, or a structural check.

    ``kind`` is "number" (passed when |new - ref| <= atol, or both NaN),
    "text" (passed when equal) or "structure" (an array, key, line or file
    present on one side only, or a shape mismatch: always failed).
    """
    file: str
    location: str
    kind: str
    ref: object
    new: object
    passed: bool
    abs_diff: float = None


def _number(file, location, ref, new, atol):
    ref, new = float(ref), float(new)
    if math.isnan(ref) or math.isnan(new):
        passed = math.isnan(ref) and math.isnan(new)
        diff = 0.0 if passed else math.nan
    elif ref == new:  # also equal infinities
        passed, diff = True, 0.0
    else:
        diff = abs(new - ref)
        passed = diff <= atol
    return Comparison(file, location, "number", ref, new, passed, diff)


def _text(file, location, ref, new):
    return Comparison(file, location, "text", ref, new, ref == new)


def _structure(file, location, ref, new):
    return Comparison(file, location, "structure", ref, new, False)


def compare_outputs(expected, produced, atol):
    """Compare every reference file with the file of the new run.

    ``expected`` and ``produced`` map relative paths to absolute paths.
    Returns the list of every Comparison made, passed or failed.
    """
    comparisons = []
    for rel in sorted(set(expected) | set(produced)):
        if rel not in produced:
            comparisons.append(_structure(rel, "file", "present", "missing"))
        elif rel not in expected:
            comparisons.append(_structure(rel, "file", "missing", "present"))
        else:
            comparisons += compare_file(rel, expected[rel], produced[rel], atol)
    return comparisons


def compare_file(rel, ref_path, new_path, atol):
    """Return every Comparison between a reference file and a new one."""
    ext = os.path.splitext(ref_path)[1].lower()
    if ext == ".npz":
        return _compare_npz(rel, ref_path, new_path, atol)
    if ext == ".json":
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        with open(new_path, encoding="utf-8") as f:
            new = json.load(f)
        return _compare_json(rel, ref, new, atol, "$")
    return _compare_text(rel, ref_path, new_path, atol)


def _compare_npz(rel, ref_path, new_path, atol):
    ref = np.load(ref_path, allow_pickle=False)
    new = np.load(new_path, allow_pickle=False)
    out = []
    for key in sorted(set(ref.files) | set(new.files)):
        if key not in new.files:
            out.append(_structure(rel, key, "present", "missing"))
            continue
        if key not in ref.files:
            out.append(_structure(rel, key, "missing", "present"))
            continue
        a, b = ref[key], new[key]
        if a.shape != b.shape:
            out.append(_structure(rel, f"{key} shape", a.shape, b.shape))
            continue
        numeric = a.dtype.kind in "fiub" and b.dtype.kind in "fiub"
        for idx in np.ndindex(a.shape):
            where = f"{key}[{','.join(map(str, idx))}]" if idx else key
            if numeric:
                out.append(_number(rel, where, a[idx], b[idx], atol))
            else:
                out.append(_text(rel, where, str(a[idx]), str(b[idx])))
    return out


def _compare_json(rel, ref, new, atol, where):
    if isinstance(ref, dict) and isinstance(new, dict):
        out = []
        for key in sorted((set(ref) | set(new)) - _VOLATILE_KEYS, key=str):
            if key not in new:
                out.append(_structure(rel, f"{where}.{key}", "present", "missing"))
            elif key not in ref:
                out.append(_structure(rel, f"{where}.{key}", "missing", "present"))
            else:
                out += _compare_json(rel, ref[key], new[key], atol, f"{where}.{key}")
        return out
    if isinstance(ref, list) and isinstance(new, list):
        out = [_compare_json(rel, a, b, atol, f"{where}[{i}]")
               for i, (a, b) in enumerate(zip(ref, new))]
        out = [c for group in out for c in group]
        for i in range(len(new), len(ref)):
            out.append(_structure(rel, f"{where}[{i}]", "present", "missing"))
        for i in range(len(ref), len(new)):
            out.append(_structure(rel, f"{where}[{i}]", "missing", "present"))
        return out
    numeric = (int, float)
    if (isinstance(ref, numeric) and isinstance(new, numeric)
            and not isinstance(ref, bool) and not isinstance(new, bool)):
        return [_number(rel, where, ref, new, atol)]
    return [_text(rel, where, ref, new)]


def _tokens(line):
    """Split a line into its text skeleton and its numbers."""
    numbers = [float(m) for m in _NUMBER.findall(line)]
    return _NUMBER.sub("#", line).rstrip(), numbers


def _significant_lines(path):
    with open(path, encoding="utf-8") as f:
        return [(i + 1, line.rstrip("\n")) for i, line in enumerate(f)
                if not _VOLATILE_LINE.match(line)]


def _split_label(line):
    """Split a report line 'label : value' into its parameter name and value.

    Tags such as [I]/[O] and tree characters are removed from the name.
    A line without a colon has no name: ('', the whole line).
    """
    if ":" not in line:
        return "", line.strip()
    label, value = line.split(":", 1)
    label = re.sub(r"^[\s#]*(\[[A-Z]\])?[\s├└│─]*", "", label)
    return " ".join(label.split()), value.strip()


def _compare_text(rel, ref_path, new_path, atol):
    """Compare line by line: the text of each line, then each of its numbers.

    Lines are paired in order (volatile lines left out); line numbers given
    are those of the reference file.
    """
    ref = _significant_lines(ref_path)
    new = _significant_lines(new_path)
    out = []
    for (i, a), (_, b) in zip(ref, new):
        text_a, num_a = _tokens(a)
        text_b, num_b = _tokens(b)
        label, value_a = _split_label(a)
        _, value_b = _split_label(b)
        name = label or f"line {i}"
        # Passes when the lines match with their numbers masked
        out.append(Comparison(rel, name, "text", value_a, value_b, text_a == text_b))
        if len(num_a) != len(num_b):
            out.append(_structure(rel, f"{name} (number count)", len(num_a), len(num_b)))
            continue
        # Numbers of the label (B0, P5, ...) come first, then those of the value
        n_label = len(_NUMBER.findall(label))
        n_value = len(num_a) - n_label
        for k, (x, y) in enumerate(zip(num_a, num_b)):
            if k < n_label:
                where = f"{name} (label number {k + 1})"
            elif n_value > 1:
                where = f"{name} [{k - n_label + 1}]"
            else:
                where = name
            out.append(_number(rel, where, x, y, atol))
    for i, a in ref[len(new):]:
        out.append(_structure(rel, f"line {i}", a.strip(), "missing"))
    for i, b in new[len(ref):]:
        out.append(_structure(rel, f"new line {i}", "missing", b.strip()))
    return out


# ── summary ─────────────────────────────────────────────────────────────────

def _rel_diff(c):
    """|new - ref| / |ref| of a number comparison as text ('-' for others)."""
    if c.kind != "number":
        return "-"
    if c.abs_diff != c.abs_diff:  # one side NaN
        return "nan"
    if c.ref == 0:
        return "0" if c.abs_diff == 0 else "inf"
    return f"{c.abs_diff / abs(c.ref):.3e}"


def _abs_diff(c):
    """|new - ref| of a number comparison as text ('-' for others)."""
    if c.kind != "number":
        return "-"
    if c.abs_diff != c.abs_diff:  # one side NaN
        return "nan"
    return f"{c.abs_diff:.3e}"


def _table(header, rows, indent="    "):
    """Lines of a plain text table: header, rule, then one line per row."""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

    def line(cells):
        return (indent + " | ".join(c.ljust(w) for c, w in zip(cells, widths))).rstrip()

    rule = indent + "-+-".join("-" * w for w in widths)
    return [line(header), rule] + [line(r) for r in rows]


def summarize(comparisons, atol):
    """Per-file summary of the comparisons, listing every failed one."""
    by_file = {}
    for c in comparisons:
        by_file.setdefault(c.file, []).append(c)
    blocks = []
    for rel, items in by_file.items():
        failed = [c for c in items if not c.passed]
        status = "FAIL" if failed else "ok"
        head = (f"[{status}] {rel}: {len(failed)} of {len(items)} comparisons failed ")
        lines = [head]
        if failed:
            lines += _table(["Parameter", "Reference", "New", "Abs. diff", "Rel. diff"],
                            [[c.location, c.ref, c.new, _abs_diff(c), _rel_diff(c)]
                             for c in failed])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
