"""
Theme helpers for the D0FUS documentation (pixel style).

The page header shows the location of the page as a terminal path,
``~/d0fus › models › plasma-model › geometry``, built from the parents of the
page in the table of contents. Each element links to its page.

Labels are slugs of the page titles. Titles that are already identifiers
(API objects such as ``f_q95`` or ``GlobalConfig``) are kept as they are, so
that the path stays readable for code.
"""

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_NOT_SLUG = re.compile(r"[^a-z0-9]+")
_IDENTIFIER = re.compile(r"^[A-Za-z_][\w.]*$")
_SECTION_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s+")
_PACKAGES = ("D0FUS_BIB.", "D0FUS_EXE.")


def _is_code_name(text):
    """True for identifiers that must keep their case (f_q95, GlobalConfig, D0FUS)."""
    return bool(_IDENTIFIER.match(text)) and (
        "_" in text or "." in text or re.search(r"[a-z][A-Z]|\d", text) is not None)


def slug(title):
    """Terminal-style label of a page title."""
    text = html.unescape(_TAG.sub("", title or "")).strip()
    text = _SECTION_NUMBER.sub("", text)            # "2.2. Geometry" -> "Geometry"
    if _is_code_name(text):
        for prefix in _PACKAGES:
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _NOT_SLUG.sub("-", text.lower()).strip("-")


def add_terminal_path(app, pagename, templatename, context, doctree):
    """Store the (label, link) pairs of the terminal path in the page context."""
    parents = context.get("parents") or []
    path = [(slug(p.get("title", "")), p.get("link")) for p in parents]
    path.append((slug(context.get("title", "")), None))
    context["d0fus_path"] = [(label, link) for label, link in path if label]


def setup(app):
    app.connect("html-page-context", add_terminal_path)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
