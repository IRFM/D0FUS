"""
Tolerant rendering of the D0FUS docstrings.

The code mixes NumPy-style docstrings (Parameters / Returns / References) with
free-form ones written for reading in an editor: aligned ASCII formulas,
indented tables, one-off section headers. This extension rewrites every
docstring in memory before Napoleon parses it, so that both styles render
cleanly. The source files are never modified.

Rules
-----
* A leading module title ("Title" over "=====") is dropped: the page has one.
* Parameter-type sections (Parameters, Returns, Raises, ...) are left to
  Napoleon unchanged.
* "References" entries are kept one per line (continuation lines merged).
* Any other header becomes a rubric. Indented runs of lines in free text
  (ASCII formulas, aligned tables, code) become preformatted blocks, so the
  alignment chosen by the author is preserved.
* Free-form "name : type  description" lines become a field list.
* Plain-text characters that reST would parse as markup are escaped.
"""

from __future__ import annotations

import re

_UNDERLINE = re.compile(r"^\s*([=\-~^])\1{2,}\s*$")

# Sections parsed by Napoleon itself (indentation is meaningful there).
_PARAM_SECTIONS = {
    "args", "arguments", "attributes", "keyword args", "keyword arguments",
    "other parameters", "parameters", "params", "receive", "receives",
    "return", "returns", "raise", "raises", "warns", "yield", "yields",
    "methods",
}
_REF_SECTIONS = {"references", "reference"}
# Sections Napoleon renders as admonitions or rubrics; content is free text.
_TEXT_SECTIONS = {"notes", "note", "examples", "example", "see also", "warning",
                  "warnings", "todo"}

# One-line NumPy entry: "name : type", two spaces or more, then the description.
_ONE_LINE_ENTRY = re.compile(
    r"^([A-Za-z_*\u0370-\u03ff][\w\u0370-\u03ff, *]*?)\s+:\s+(\S+(?: \S+)*?)\s{2,}(\S.*)$")

# "name : type   description" (free-form parameter line).
_FREEFORM_PARAM = re.compile(
    r"^(\s*)([A-Za-z_Ͱ-Ͽ][\wͰ-Ͽ]*)\s+:\s+(\S.*)$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _escape_text(line: str) -> str:
    """Escape characters that reST reads as markup in plain physics text."""
    st = line.strip()
    if st.startswith((".. ", ":")):
        return line
    if (st.startswith("+-") or st.startswith("+=") or st.startswith("|")) and st.endswith(("|", "+")):
        return line                     # grid-table row: leave the table intact
    line = line.replace("|", r"\|")
    if line.count("**") % 2:
        line = line.replace("**", r"\*\*")
    singles = re.sub(r"\*\*", "", line).count("*")
    if singles % 2:
        line = re.sub(r"(?<![\\*])\*(?!\*)", r"\\*", line)
    # word_ followed by space or punctuation is a hyperlink reference in reST.
    line = re.sub(r"(\w)_(?=[\s,.;:)\]*\\]|$)", r"\1\\_", line)
    return line


def _split_sections(lines):
    """Yield (header or None, body lines). Headers are 'Title' + underline."""
    sections = []
    header = None
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _UNDERLINE.match(nxt) and not _UNDERLINE.match(line):
            sections.append((header, body))
            header = (line.strip(), nxt.strip()[0])
            body = []
            i += 2
            continue
        body.append(line)
        i += 1
    sections.append((header, body))
    return sections


def _free_text(body: list[str], header_indent: int | None = None) -> list[str]:
    """Free text: indented runs become literal blocks, the rest is escaped."""
    nonblank = [l for l in body if l.strip()]
    if not nonblank:
        return body
    base = min(_indent(l) for l in nonblank)
    if header_indent is not None and base > header_indent:
        # Whole section indented under its header: an aligned block.
        while body and not body[0].strip():
            body = body[1:]
        return ["", " " * header_indent + "::", ""] + body + [""]
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        line = body[i]
        if line.strip() and _indent(line) > base:
            # Collect the indented run (blank lines inside it are kept).
            j = i
            run = []
            while j < n and (not body[j].strip() or _indent(body[j]) > base):
                run.append(body[j])
                j += 1
            while run and not run[-1].strip():
                run.pop()
                j -= 1
            prev = next((l for l in reversed(out) if l.strip()), "")
            if out and out[-1].strip() and prev.lstrip().startswith(":") \
                    and _indent(prev) == base:
                # Continuation of a field-list entry: one uniform indent.
                out.extend(" " * (base + 4) + l.strip() for l in run if l.strip())
                out.append("")
                i = j
                continue
            if prev.rstrip().endswith("::"):
                if out and out[-1].strip():
                    out.append("")
            else:
                if out and out[-1].strip():
                    out.append("")
                out.extend([" " * base + "::", ""])
            out.extend(run)
            out.append("")
            i = j
            continue
        out.append(_escape_text(line) if line.strip() else line)
        i += 1
    return out


def _references(body: list[str]) -> list[str]:
    """One reference per line: merge indented continuations, use a line block."""
    entries: list[str] = []
    base = min((_indent(l) for l in body if l.strip()), default=0)
    for line in body:
        if not line.strip():
            continue
        if entries and _indent(line) > base:
            entries[-1] += " " + line.strip()
        else:
            entries.append(line.strip())
    out = [""]
    for e in entries:
        e = e.replace("|", r"\|").replace("*", r"\*")
        e = re.sub(r"(\w)_(?=[\s,.;:)\]*\\]|$)", r"\1\\_", e)
        out.append("| " + e)
    out.append("")
    return out


def _param_section(body: list[str]) -> list[str]:
    """Napoleon parameter section: clean each entry's description block."""
    nonblank = [l for l in body if l.strip()]
    if not nonblank:
        return body
    base = min(_indent(l) for l in nonblank)
    out: list[str] = []
    desc: list[str] = []
    one_line = False

    def flush():
        if desc:
            out.extend(_free_text(desc))
            desc.clear()

    for line in body:
        if line.strip() and _indent(line) == base:
            flush()
            m = _ONE_LINE_ENTRY.match(line.strip())
            if m:
                # "name : type   description" written on one line: split it so
                # that Napoleon sees the type and the description separately.
                names, typ, text = m.groups()
                out.append(" " * base + f"{names} : {typ}")
                desc.append(" " * (base + 4) + text)
                one_line = True
            else:
                out.append(line)
                one_line = False
        elif one_line and line.strip():
            desc.append(" " * (base + 4) + line.strip())   # aligned continuation
        else:
            desc.append(line)
    flush()
    return out


def _fieldlist(body: list[str]) -> list[str]:
    """Free-form 'name : type  text' lines into a reST field list."""
    out: list[str] = []
    field_indent = None
    for line in body:
        m = _FREEFORM_PARAM.match(line)
        if m and not line.lstrip().startswith(".."):
            ind, pname, rest = m.groups()
            if field_indent is None and out and out[-1].strip():
                out.append("")
            out.append(f"{ind}:{pname}: {_escape_text(rest)}")
            field_indent = len(ind)
            continue
        if field_indent is not None and line.strip() and _indent(line) <= field_indent:
            out.append("")
            field_indent = None
        elif not line.strip():
            field_indent = None
        out.append(line)
    return out


def process(app, what, name, obj, options, lines):
    if not lines:
        return
    if what == "module":
        # Authorship metadata lines are not documentation.
        lines[:] = [l for l in lines
                    if not re.match(r"^\s*(Created( on)?|Author|Date)\s*:", l)
                    and "Design 0-dimensional for FUsion Systems project" not in l]
    sections = _split_sections(list(lines))
    has_numpy = any(h and h[0].lower() in _PARAM_SECTIONS for h, _ in sections)
    out: list[str] = []
    for k, (header, body) in enumerate(sections):
        if header is None:
            if not has_numpy:
                body = _fieldlist(body)
            out.extend(_free_text(body))
            continue
        title, char = header
        key = title.lower()
        if k == 1 and char == "=" and what == "module" and not any(
                l.strip() for l in sections[0][1]):
            # Leading module title: redundant with the page title.
            out.extend(_free_text(body))
            continue
        if key in _PARAM_SECTIONS and char == "-":
            out.extend(["", title, "-" * len(title)])
            out.extend(_param_section(body))
            continue
        if key in _REF_SECTIONS:
            out.extend(["", ".. rubric:: References"])
            out.extend(_references(body))
            continue
        if key in _TEXT_SECTIONS and char == "-":
            out.extend(["", title, "-" * len(title)])
            out.extend(_free_text(body, header_indent=0))
            continue
        out.extend(["", f".. rubric:: {title}", ""])
        out.extend(_free_text(body, header_indent=0))
    lines[:] = out


def setup(app):
    # Priority < 500 so this runs before Napoleon's own docstring processor.
    app.connect("autodoc-process-docstring", process, priority=400)
    return {"version": "1.1", "parallel_read_safe": True, "parallel_write_safe": True}
