"""
Sphinx configuration for the D0FUS documentation.

Build locally (from the repository root):
    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

The API reference is regenerated from the source code at every build
(docs/_ext/d0fus_apigen.py), so it always matches the checked-out version.
"""

import os
import re
import sys
from datetime import date
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

# Make the D0FUS packages importable by autodoc, and the local extensions visible.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# Non-interactive Matplotlib backend: autodoc imports D0FUS_figures.
os.environ.setdefault("MPLBACKEND", "Agg")

# -- Project information -------------------------------------------------------
# Version read from pyproject.toml, the single source of truth.
_match = re.search(r'^version\s*=\s*"([^"]+)"',
                   (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)

project = "D0FUS"
author = "Timothé Auclair and the D0FUS contributors"
copyright = f"2023-{date.today().year}, CEA-IRFM"
release = _match.group(1) if _match else "unknown"
version = ".".join(release.split(".")[:2])

# Git reference used for the links to the source code on GitHub. Read the Docs
# sets the commit hash of the version being built. Local builds point to main.
GIT_REF = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main")

# -- API pages generated from the source tree ----------------------------------
import d0fus_apigen  # noqa: E402

d0fus_apigen.generate(REPO_ROOT, DOCS_DIR / "api" / "generated", git_ref=GIT_REF)
d0fus_apigen.generate_config_reference(
    REPO_ROOT, DOCS_DIR / "api" / "generated" / "globalconfig_fields.rst")

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "d0fus_docstrings",     # tolerant rendering of the free-form docstrings
    "d0fus_theme",          # terminal path above each page (pixel theme)
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# -- Autodoc / autosummary -----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"
autodoc_preserve_defaults = True       # show "R0=9.0", not the evaluated object
add_module_names = False
toc_object_entries_show_parents = "hide"

# -- Links from the API pages to the source on GitHub --------------------------
import importlib  # noqa: E402
import inspect  # noqa: E402


def linkcode_resolve(domain, info):
    """Return the GitHub URL of the source lines of a documented object."""
    if domain != "py" or not info.get("module"):
        return None
    try:
        obj = importlib.import_module(info["module"])
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        path = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    rel = Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    return (f"https://github.com/IRFM/D0FUS/blob/{GIT_REF}/{rel}"
            f"#L{start}-L{start + len(lines) - 1}")


# -- Napoleon (NumPy-style docstrings) -----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_preprocess_types = False
# Non-standard section headers are turned into rubrics by d0fus_docstrings.

# -- MyST (Markdown pages) -----------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "attrs_inline",
    "attrs_block",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3
myst_substitutions = {"release": release}

# -- Math ----------------------------------------------------------------------
math_numfig = True
numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}
math_eqref_format = "({number})"

# -- Bibliography (thesis biblio.bib) ------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"D0FUS {release}"
html_logo = "_static/d0fus_logo_256.png"
html_favicon = "_static/d0fus_logo_256.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_copy_source = False

html_theme_options = {
    "logo": {"text": "D0FUS", "alt_text": "D0FUS"},
    "github_url": "https://github.com/IRFM/D0FUS",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/d0fus/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "article_header_start": ["terminal-path"],
    # No right-hand column on the landing page: the hero takes the width.
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
        "index": [],
    },
    "navigation_with_keys": False,
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    # Base styles. The token colours are overridden in custom.css (Tokyo Night).
    "pygments_light_style": "github-light",
    "pygments_dark_style": "one-dark",
}
html_context = {
    "default_mode": "dark",
    "github_user": "IRFM",
    "github_repo": "D0FUS",
    "github_version": "main",
    "doc_path": "docs",
    "edit_page_provider_name": "GitHub",
    # Generated API pages have no source file in git: send them to the code.
    "edit_page_url_template": (
        "{% if file_name.startswith('api/generated') %}"
        "https://github.com/IRFM/D0FUS/tree/main/D0FUS_BIB?f={{ file_name }}"
        "{% else %}https://github.com/IRFM/D0FUS/edit/main/docs/{{ file_name }}{% endif %}"
    ),
}

# Ignore the warnings that come from third-party type hints only.
nitpicky = False
suppress_warnings = ["myst.header"]
