# API reference

These pages are generated from the source code at every build. Each module page
follows the sections of its source file, and each function page shows its
docstring, its signature with default values, and a link to the highlighted
source.

The core library (`D0FUS_BIB`) holds the models. Every function can be called
on its own, outside the solver loop. The execution modes (`D0FUS_EXE`) chain
these functions into a run, a scan, an optimisation, a POPCON map or an
uncertainty study.

```{toctree}
:maxdepth: 1
:caption: Execution modes

generated/D0FUS
generated/D0FUS_EXE.D0FUS_run
generated/D0FUS_EXE.D0FUS_scan
generated/D0FUS_EXE.D0FUS_genetic
generated/D0FUS_EXE.D0FUS_popcon
generated/D0FUS_EXE.D0FUS_uncertainty
```

```{toctree}
:maxdepth: 1
:caption: Core library

generated/D0FUS_BIB.D0FUS_parameterization
generated/D0FUS_BIB.D0FUS_physical_functions
generated/D0FUS_BIB.D0FUS_radial_build_functions
generated/D0FUS_BIB.D0FUS_cost_functions
generated/D0FUS_BIB.D0FUS_cost_data
generated/D0FUS_BIB.D0FUS_figures
generated/D0FUS_BIB.D0FUS_machine_presets
```
