# Input decks

All inputs live in one typed dataclass, `GlobalConfig`, and each field carries a
physically motivated default. A deck overrides only the fields it names. Every
other field keeps its default, so a complete machine fits in a few lines:

```ini
R0 = 7
Bmax_TF = 14
Supra_choice = REBCO
```

The complete list of fields, with types, defaults and the comments of the source
code, is on the {doc}`/api/generated/globalconfig_fields` page. That page is
regenerated from `D0FUS_parameterization.py` at each documentation build. Every
run also writes all inputs, defaults included, to `output_detailed.txt`.

## Deck syntax by mode

```ini
# RUN: fixed values                    # SCAN: exactly two brackets
R0 = 9                                 R0 = [3, 9, 25]
Bmax_TF = 13                           a  = [1, 3, 25]

# OPTIMIZATION: 2+ ranges              # POPCON: full RUN deck, then
R0 = [3, 9]                            [POPCON]
Bmax_TF = [10, 16]                     nbar_line = [0.35, 1.45, 45]
fitness_objective = COE                Tbar      = [3.5, 24.0, 36]

# UNCERTAINTY: full RUN deck, then
[UNCERTAINTY]
H           = norm(0.75, 1.50)                # truncated normal around the deck value
Scaling_Law = envelope(IPB98(y,2) | ITPA20)   # model switch (pipe-separated)
[CONTROLS]
n_samples = 1500
```

Continuous distributions: `norm(sigma)`, `norm(lo, hi)`, `norm(lo, centre, hi)`,
`tri(...)`, `unif(lo, hi)`. One worked example per mode ships in `D0FUS_INPUTS/`.
The full syntax, including the genetic optimiser controls, is given in
{doc}`reference/syntax` and {doc}`reference/uncertainty_syntax`.

## Programmatic inputs

In library mode, a configuration is built with `dataclasses.replace`:

```python
from dataclasses import replace
from D0FUS_BIB.D0FUS_parameterization import DEFAULT_CONFIG

cfg = replace(DEFAULT_CONFIG, R0=8.0, Bmax_TF=13.0, Supra_choice="REBCO")
```

## Fidelity levels

Each model carries its own selector in `GlobalConfig`. The factories
`preset_academic()` and `preset_refined()` set a coherent bundle. Both levels share
the same interface and solver, with comparable run times. See
{doc}`reference/fidelity`.
