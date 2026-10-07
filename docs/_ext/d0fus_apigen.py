"""
Generate the API reference pages of the D0FUS documentation from the source tree.

For every documented module, the top-level public functions and classes are
grouped under the section markers already present in the code:

    #%% Title                    (Spyder cell)
    # ── Title ──────────          (inline banner)
    # =========                   (block banner, title on the line(s) between)
    # Title
    # =========

Each group becomes a section of the module page, with an autosummary table
whose stubs (one page per object) are produced by sphinx.ext.autosummary.
The pages are rewritten at every build, so the reference follows the code
without any manual upkeep. Output files are build products (git-ignored).
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

# (module, file, page title, one-line role). Order = order in the API index.
MODULES = [
    ("D0FUS", "D0FUS.py", "D0FUS (entry point)",
     "Command-line entry point. Reads an input deck and dispatches it to the execution mode it describes."),
    ("D0FUS_EXE.D0FUS_run", "D0FUS_EXE/D0FUS_run.py", "D0FUS_run",
     "RUN mode: solve a single design point and write the full report."),
    ("D0FUS_EXE.D0FUS_scan", "D0FUS_EXE/D0FUS_scan.py", "D0FUS_scan",
     "SCAN mode: two-dimensional maps of the design space."),
    ("D0FUS_EXE.D0FUS_genetic", "D0FUS_EXE/D0FUS_genetic.py", "D0FUS_genetic",
     "OPTIMIZATION mode: genetic (memetic) search for the optimal design."),
    ("D0FUS_EXE.D0FUS_popcon", "D0FUS_EXE/D0FUS_popcon.py", "D0FUS_popcon",
     "POPCON mode: operating map in the (density, temperature) plane at fixed machine."),
    ("D0FUS_EXE.D0FUS_uncertainty", "D0FUS_EXE/D0FUS_uncertainty.py", "D0FUS_uncertainty",
     "UNCERTAINTY mode: Monte Carlo propagation of input and model-form uncertainties."),
    ("D0FUS_BIB.D0FUS_parameterization", "D0FUS_BIB/D0FUS_parameterization.py", "D0FUS_parameterization",
     "Physical constants and the GlobalConfig dataclass holding every input."),
    ("D0FUS_BIB.D0FUS_physical_functions", "D0FUS_BIB/D0FUS_physical_functions.py", "D0FUS_physical_functions",
     "Plasma physics: geometry, profiles, power balance, current, limits, exhaust, runaway electrons."),
    ("D0FUS_BIB.D0FUS_radial_build_functions", "D0FUS_BIB/D0FUS_radial_build_functions.py", "D0FUS_radial_build_functions",
     "Magnets and radial build: TF and CS sizing, conductors, quench protection, ripple."),
    ("D0FUS_BIB.D0FUS_cost_functions", "D0FUS_BIB/D0FUS_cost_functions.py", "D0FUS_cost_functions",
     "Techno-economic models (Sheffield, Whyte)."),
    ("D0FUS_BIB.D0FUS_cost_data", "D0FUS_BIB/D0FUS_cost_data.py", "D0FUS_cost_data",
     "Reference cost data used by the cost models."),
    ("D0FUS_BIB.D0FUS_figures", "D0FUS_BIB/D0FUS_figures.py", "D0FUS_figures",
     "Figure catalogue: render wrappers around the physics and radial-build functions."),
    ("D0FUS_BIB.D0FUS_machine_presets", "D0FUS_BIB/D0FUS_machine_presets.py", "D0FUS_machine_presets",
     "Curated 3D presets of existing and planned machines for the concept-view renderer."),
]

_CELL = re.compile(r"^#%%\s*(.*\S)?\s*$")
_INLINE = re.compile(r"^#\s*[─\-=]{2,}\s+(.+?)\s+[─\-=]{2,}\s*$")
_RULE = re.compile(r"^#\s*[─=\-]{10,}\s*$")
_IGNORED_TITLES = {"import", "imports", "main", "validation", "tests", "test"}


def _clean(title: str) -> str:
    """Normalise a marker title (strip decoration, trailing colons)."""
    title = re.sub(r"[─=]+", "", title).strip(" :-#")
    title = re.sub(r"\s{2,}", " ", title)
    if title.isupper():
        # "COPPER MATERIAL PROPERTIES" -> "Copper material properties"
        title = title.capitalize()
    return title


def _section_markers(lines: list[str]) -> list[tuple[int, str]]:
    """Return (line_number, title) for every section marker, 1-based lines."""
    marks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        m = _CELL.match(line)
        if m:
            if m.group(1):
                marks.append((i + 1, _clean(m.group(1))))
            i += 1
            continue
        m = _INLINE.match(line)
        if m:
            marks.append((i + 1, _clean(m.group(1))))
            i += 1
            continue
        if _RULE.match(line):
            # Block banner: rule, one or more comment lines, rule.
            j = i + 1
            text = []
            while j < n and lines[j].startswith("#") and not _RULE.match(lines[j].rstrip()):
                text.append(lines[j].lstrip("#").strip())
                j += 1
            if j < n and _RULE.match(lines[j].rstrip()) and 0 < len(text) <= 3 and text[0]:
                marks.append((i + 1, _clean(text[0])))
                i = j + 1
                continue
        i += 1
    return [(ln, t) for ln, t in marks if t and t.lower() not in _IGNORED_TITLES]


def _public_objects(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Top-level public functions and classes: (line, name, kind)."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                first = min([d.lineno for d in node.decorator_list] + [node.lineno])
                out.append((first, node.name, kind))
    return out


def group_module(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    """Group the public objects of one module under its section markers."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    marks = _section_markers(lines)
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for line, name, kind in _public_objects(ast.parse(src)):
        title = "General"
        for ln, t in marks:
            if ln < line:
                title = t
            else:
                break
        if title not in groups:
            groups[title] = []
            order.append(title)
        groups[title].append((name, kind))
    return [(t, groups[t]) for t in order]


def _underline(text: str, char: str) -> str:
    return f"{text}\n{char * max(len(text), 4)}\n"


def generate(repo_root: Path, out_dir: Path, git_ref: str = "main") -> None:
    """Write one reST page per module plus the stub folder layout."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    index = [
        ".. This file is generated by docs/_ext/d0fus_apigen.py. Do not edit.\n",
        ".. toctree::\n   :maxdepth: 1\n\n",
    ]
    for modname, rel, title, role in MODULES:
        path = repo_root / rel
        if not path.exists():
            continue
        groups = group_module(path)
        n_obj = sum(len(g) for _, g in groups)
        page = [
            ".. This file is generated by docs/_ext/d0fus_apigen.py. Do not edit.\n\n",
            ".. rst-class:: d0-api\n\n",          # code-name title (theme CSS)
            _underline(title, "="),
            f"\n{role}\n\n",
            f":Module: ``{modname}``\n",
            f":Source: `{rel} <https://github.com/IRFM/D0FUS/blob/{git_ref}/{rel}>`__\n",
            f":Public objects: {n_obj}\n\n",
            f".. automodule:: {modname}\n   :no-members:\n\n",
            f".. currentmodule:: {modname}\n\n",
        ]
        if len(groups) > 1 or (groups and groups[0][0] != "General"):
            for gtitle, objs in groups:
                page.append("\n" + _underline(gtitle, "-") + "\n")
                page.append(".. autosummary::\n   :toctree: objects\n   :nosignatures:\n\n")
                page.extend(f"   {name}\n" for name, _ in objs)
        elif groups:
            page.append(".. autosummary::\n   :toctree: objects\n   :nosignatures:\n\n")
            page.extend(f"   {name}\n" for name, _ in groups[0][1])
        (out_dir / f"{modname}.rst").write_text("".join(page), encoding="utf-8")
        index.append(f"   {modname}\n")


if __name__ == "__main__":
    # Quick inspection of the grouping, without Sphinx.
    root = Path(__file__).resolve().parents[2]
    for modname, rel, *_ in MODULES:
        print(f"\n== {modname}")
        for t, objs in group_module(root / rel):
            print(f"  [{len(objs):3d}] {t}")


# =============================================================================
# GlobalConfig reference table, read from the dataclass source
# =============================================================================

_GROUP = re.compile(r"^\s*#\s*(?:──|---)\s*(.+?)\s*(?:─+|-{3,})\s*$")


def _rst_escape(text: str) -> str:
    """Escape inline reST markup in a plain-text cell."""
    text = text.replace("\\", "\\\\")
    for ch in "*`|_[]<>":
        text = text.replace(ch, "\\" + ch)
    return text


def _config_fields(path: Path):
    """Parse GlobalConfig: list of (group, name, type, default, inline, notes)."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlobalConfig")
    fields = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields[node.lineno] = node

    rows = []
    group = "General"
    last = None              # last field row (dict)
    blank_since_last = False
    pending: list[str] = []
    field_indent = None
    start = cls.body[0].end_lineno + 1 if isinstance(cls.body[0], ast.Expr) else cls.lineno + 1

    for ln in range(start, cls.end_lineno + 1):
        line = lines[ln - 1]
        stripped = line.strip()
        if ln in fields:
            node = fields[ln]
            seg = ast.get_source_segment(src, node.value) if node.value is not None else ""
            ann = ast.get_source_segment(src, node.annotation)
            tail = line.split(seg, 1)[1] if seg and seg in line else ""
            inline = tail.split("#", 1)[1].strip() if "#" in tail else ""
            field_indent = len(line) - len(line.lstrip())
            last = {"group": group, "name": node.target.id, "type": ann,
                    "default": seg, "inline": inline, "notes": list(pending)}
            pending = []
            rows.append(last)
            blank_since_last = False
            continue
        if not stripped:
            blank_since_last = True
            continue
        m = _GROUP.match(line)
        if m:
            group = re.sub(r"^\d+[a-z]?\.\s*", "", m.group(1)).strip()
            group = group[0].upper() + group[1:]
            pending = []
            last = None
            continue
        if stripped.startswith("#"):
            text = stripped[1:].rstrip()
            text = text[1:] if text.startswith(" ") else text
            indent = len(line) - len(line.lstrip())
            if last is not None and field_indent is not None and indent > field_indent \
                    and not blank_since_last:
                last["notes"].append(text)          # aligned continuation
            elif last is not None and not last["inline"] and not blank_since_last \
                    and not pending:
                last["notes"].append(text)          # block describing the field above
            else:
                pending.append(text)                # block describing the next field
    return rows


def generate_config_reference(repo_root: Path, out_file: Path) -> None:
    rows = _config_fields(repo_root / "D0FUS_BIB" / "D0FUS_parameterization.py")
    groups: list[str] = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    out = [
        ".. This file is generated by docs/_ext/d0fus_apigen.py. Do not edit.\n\n",
        ".. _chap:globalconfig:\n\n",
        _underline("GlobalConfig fields", "="),
        "\nEvery input of D0FUS is a field of the ``GlobalConfig`` dataclass "
        "(``D0FUS_BIB/D0FUS_parameterization.py``). This page is generated from "
        "that file at each documentation build: names, types, defaults and "
        "descriptions are the ones in the code.\n\n",
        f"{len(rows)} fields in {len(groups)} groups. Any field can be set in an "
        "input deck (``name = value``) or through ``dataclasses.replace``.\n\n",
    ]
    for g in groups:
        out.append("\n" + _underline(g, "-") + "\n")
        out.append(".. list-table::\n   :header-rows: 1\n   :widths: 22 10 14 54\n"
                   "   :class: d0fus-config\n\n")
        out.append("   * - Field\n     - Type\n     - Default\n     - Description\n")
        for r in (r for r in rows if r["group"] == g):
            default = r["default"].replace("``", "")
            desc = _rst_escape(r["inline"]) if r["inline"] else ""
            notes = [n for n in r["notes"] if n.strip()]
            out.append(f"   * - ``{r['name']}``\n     - ``{r['type']}``\n"
                       f"     - ``{default}``\n")
            if not desc and not notes:
                out.append("     -\n")
                continue
            out.append(f"     - {desc}\n" if desc else "     -\n")
            if notes:
                pad = "       " if desc else "       "
                out.append("\n" if desc else "")
                for n in notes:
                    out.append(f"{pad}| {_rst_escape(n)}\n")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(out), encoding="utf-8")
