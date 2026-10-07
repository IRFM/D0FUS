(sec:chap2_device)=

# Full-device benchmark

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 2.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The module-level agreement of Section {ref}`Module-level validation <sec:chap2_module>` does not by itself guarantee that D0FUS is globally correct, since errors could exist for example in the self-consistency loops coupling the different modules, or in aspects that have not been tested above, such as the power balance or the plasma stability limits. This section therefore benchmarks complete D0FUS design points against two documented references: the EU-DEMO 2017 baseline and the ITER Q = 10 scenario. In both cases D0FUS receives only the primary design inputs of the reference, and every other quantity is solved for.

```{toctree}
:maxdepth: 1

eudemo
iter
iter_inputs
```
