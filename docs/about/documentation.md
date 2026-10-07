# About this documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org) and hosted on
Read the Docs. Its sources live in the `docs/` folder of the repository. Three
parts are written by hand. Two parts are generated from the code at every build.

## What is generated from the code

**API reference.** `docs/_ext/d0fus_apigen.py` reads each module with Python's
`ast` module. It groups the public functions under the section markers already
present in the source:

```python
#%% Geometry formulas                 # Spyder cell

# ── Helium ash accumulation model ──   # inline banner

# =============================================================================
# Plasma volume                         # block banner
# =============================================================================
```

Each group becomes a section of the module page. Each function gets its own page,
built by `sphinx.ext.autodoc` from its docstring and signature, with a link to the
highlighted source.

**GlobalConfig fields.** The same script reads the `GlobalConfig` dataclass and
writes one table per `# ── N. Title ──` banner, with the name, type, default
value and comments of every field.

Neither file needs to be edited when the code changes. A new function, a new
field or a corrected docstring appears at the next build.

## Docstring conventions

The NumPy format renders best:

```python
def f_Kappa(A, Option_Kappa, κ_manual, ms):
    """
    Maximum achievable plasma elongation vs aspect ratio.

    Parameters
    ----------
    A : float or ndarray
        Plasma aspect ratio R₀/a [-].

    Returns
    -------
    κ : float or ndarray

    References
    ----------
    Stambaugh et al., Fusion Sci. Technol. 59, 279 (2011).
    """
```

Free-form docstrings are also accepted. The local extension
`docs/_ext/d0fus_docstrings.py` keeps their layout in memory at build time,
without touching the source: aligned formulas and tables stay preformatted,
`name : type  description` lines become a field list, and non-standard headers
("Physical context", "Algorithm", …) become sub-headings.

## What is written by hand

- `index.md`, `getting_started/` and the first pages of `user_guide/`, adapted from
  the README.
- `theory/`, `validation/` and `user_guide/reference/`, adapted from the thesis
  (Chapters 1, 2, 4 and Appendices B to D). Citations use the thesis
  bibliography, `docs/references.bib`, through `sphinxcontrib-bibtex`.

## Visual style

The pages use `pydata-sphinx-theme` with a pixel-art skin: Tokyo Night colours,
Pixelify Sans for titles, Silkscreen for labels and JetBrains Mono for text. The
skin is in `docs/_static/custom.css`. The fonts are in `docs/_static/fonts/`,
under the SIL Open Font License. The path shown above each page comes from
`docs/_ext/d0fus_theme.py`. Dark mode is the default. The button at the top right
switches to light mode.

## Building locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. A full build takes about one
minute. The `docs/_build/` and `docs/api/generated/` folders are build products
and are not versioned.

## Publication

Read the Docs rebuilds the documentation at every push on `main` and keeps one
version per git tag, as configured in `.readthedocs.yaml` at the repository root.
