# D0FUS

{.d0-label}
Design 0-dimensional for Fusion Systems · CEA-IRFM · release {{ release }}

{.d0-tagline}
An open-source Python system code for tokamak power plant design.

D0FUS covers plasma physics, superconducting magnet engineering and
techno-economics, and evaluates a complete design point in a fraction of a
second. It is developed at CEA-IRFM.

```{image} _static/d0fus_banner_web.png
:alt: D0FUS capabilities
:width: 100%
:class: banner
```

Every model is a closed-form or semi-analytical expression that can be read,
called on its own and replaced. On this elementary brick, D0FUS builds five
execution modes: the run of a single design point, two-dimensional scans, a
genetic optimisation, POPCON operating maps and the Monte Carlo propagation of
uncertainties.

```bash
pip install d0fus
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install D0FUS and run the ITER example deck.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

Execution modes, input decks, outputs and every input field.
:::

:::{grid-item-card} Models
:link: theory/index
:link-type: doc

Plasma, radial build, exhaust and cost models, with their derivations.
:::

:::{grid-item-card} Validation
:link: validation/index
:link-type: doc

Module-level checks and full-device benchmarks on EU-DEMO and ITER.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every function, generated from the source code at each build.
:::

:::{grid-item-card} Citing D0FUS
:link: about/citing
:link-type: doc

Reference publications and how to cite the code.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
user_guide/index
theory/index
validation/index
api/index
about/index
```
