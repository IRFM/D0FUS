# Quick start

## Run the ITER example

From the repository root:

```bash
python D0FUS.py D0FUS_INPUTS/1_run_ITER.txt
```

The execution mode is detected from the syntax of the input deck. This deck holds
fixed values only, so D0FUS solves a single design point (RUN mode). It writes a
timestamped folder under `D0FUS_OUTPUTS/run/` with the report files and the
figures. The {doc}`/user_guide/outputs` page lists them.

The other example decks run the other modes:

| Deck | Mode |
|------|------|
| `1_run_ITER.txt` | RUN, single design point |
| `2_scan_ITER.txt` | SCAN, two-dimensional map |
| `3_genetic_ITER.txt` | OPTIMIZATION, genetic search |
| `4_uncertainty_ITER.txt` | UNCERTAINTY, Monte Carlo propagation |
| `5_popcon_ITER.txt` | POPCON, operating map at fixed machine |

## Run from a script

Deck-driven run, with report files and figures:

```python
from D0FUS_EXE import D0FUS_run
D0FUS_run.main("D0FUS_INPUTS/1_run_ITER.txt")   # full run and report files
```

Programmatic single point, with no file involved:

```python
from dataclasses import replace
from D0FUS_BIB.D0FUS_parameterization import DEFAULT_CONFIG
from D0FUS_EXE import D0FUS_run

cfg = replace(DEFAULT_CONFIG, R0=8.0, Bmax_TF=13.0, Supra_choice="REBCO")
results = D0FUS_run.run(cfg)   # tuple of scalar outputs (np.nan if no solution)
```

## Call a single model

Every function of the library can be called on its own, outside the solver loop.
For instance, the maximum elongation given by each closure at aspect ratio 3:

```python
from D0FUS_BIB.D0FUS_physical_functions import f_Kappa

for law in ("Stambaugh", "Freidberg", "Wenninger", "Blend"):
    print(law, f_Kappa(3.0, law, None, 0.3))
```

The {doc}`/api/index` documents every function, with its assumptions and
references.
