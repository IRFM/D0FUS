(chap:validation)=
# Validation and benchmarks

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Chapter 2 (introduction). The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The conclusions of the thesis rest on the predictions of a systems code, so those predictions must be established as reliable first.[^1] Chapter 3 of the thesis relies on this code to draw design conclusions with direct consequences on the size and cost of a fusion power plant. These pages set out to do so in two stages of increasing integration: Section {ref}`Module-level validation <sec:chap2_module>` validates each module in isolation against reference values, and Section {ref}`Full-device benchmark <sec:chap2_device>` benchmarks complete, self-consistent design points against the EU-DEMO 2017 baseline and the ITER Q = 10 scenario.

[^1]: Nobody believes a simulation except the one who ran it, goes the saying. Systems codes achieve something rarer: they are doubted even by their own authors.

```{toctree}
:maxdepth: 1
:numbered:

module_level
full_device
```
