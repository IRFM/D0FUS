(sec:chap1_diagnostics)=
(ssec:chap1_operational)=

# Heat exhaust, second-order limits and performance

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.4. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The two hard viability tests of Sections {ref}`Plasma operational limits <sec:chap1_plasma>` and {ref}`Radial build limits <sec:chap1_magnets>` (plasma stability and radial build) are necessary but not sufficient to qualify a candidate machine. This last section gathers the remaining evaluations, in the hierarchy set out in the overview of the models: first the heat exhaust, the third first-order limit, which D0FUS deliberately treats through proxies rather than through a hard limit, then the second-order operability limits (H-mode access, neutron wall load, runaway electrons), and finally the performance of the converged design point: its plant-level electrical balance and its cost. All are post-convergence evaluations: they do not feed back into the solver, but they weigh heavily when candidate designs are compared in Chapter 3 of the thesis.

```{toctree}
:maxdepth: 1

heat_exhaust
second_order
performance
```
