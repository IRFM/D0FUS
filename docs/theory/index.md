(chap:d0fus)=
# Models

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Chapter 1 (introduction). The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


High-temperature superconductors bring within reach toroidal fields well beyond those of the present generation of tokamaks, and with them a simple promise: at higher field, the same fusion power should fit in a smaller, and hopefully cheaper, machine. Whether this promise survives the full set of physics and engineering constraints is the question driving the thesis, as set out in its introduction, and answering it requires evaluating many candidate machines quickly. This is the role of a system code: a deliberately reduced model of the entire plant, fast enough for whole design spaces to be scanned rather than single points analysed.

The reasons for developing a new code rather than adopting an existing one are given in the thesis introduction. The code built here, D0FUS, draws on the pedagogical approach of Freidberg *et al.* {footcite:p}`freidberg2015designing`, the legacy of HELIOS (the ancestor of SYCOMORE), and inspiring models from METIS {footcite:p}`artaud2018metis` and PROCESS {footcite:p}`kovari2014process`. A design point, in this work, is defined by its main inputs: the technological choices (steel, superconductor, etc.), the machine size ($R_0$,$a$), the target fusion power, the auxiliary power, and the desired plasma pulse duration. Given such a point, D0FUS answers two questions: is this machine viable? And what performance (e.g. cost metrics) can be expected from it?

Viability is judged against the operational limits of a tokamak, three of which stand at the first order.

- The plasma must be stable. An operating point has to exist within the density, pressure and current limits, and D0FUS enforces these as hard limits.

- The coils must be feasible. The toroidal field system and the central solenoid have to generate the requested field and flux, fit within the machine bore, survive their electromagnetic loads and protect their conductor. These too are hard limits.

- The heat exhaust is every bit as critical, but in the author’s view it cannot be reduced to a hard limit at the system-code level: whether an exhaust power is acceptable depends on the divertor geometry and the mitigation strategy, design studies in their own right that come downstream of a system-code study. D0FUS therefore evaluates the exhaust through proxies, which do not veto a design but quantify how hard its exhaust problem will be and rank it against machines whose solutions are established.

Beyond these three first-order limits, the code evaluates second-order operability limits (access to the H-mode confinement regime, the neutron wall load, and the runaway-electron risk during disruptions) and, last, the performance of the converged design point, essentially its cost.

The chapter follows this hierarchy. Section {ref}`General architecture of D0FUS <sec:chap1_architecture>` presents the tool itself: its architecture, its two fidelity levels and its execution modes. Section {ref}`Plasma operational limits <sec:chap1_plasma>` addresses plasma stability, the first of the first-order limits, opening with the three limits themselves before unrolling the physics chain required to evaluate them. Section {ref}`Radial build limits <sec:chap1_magnets>` addresses the second, the feasibility of the coils, through the radial build models. Section {ref}`Heat exhaust, second-order limits and performance <sec:chap1_diagnostics>` gathers the heat-exhaust proxies, the second-order operability indicators and the performance figures of merit. Throughout, the body of the chapter keeps to the logic of each model and to the reasons for including each phenomenon while the complete formulations are collected in {ref}`Reference material <chap:app_d0fus>` (code reference), {ref}`Plasma physics <chap:app_plasma>` (plasma) and {ref}`Coil sizing <chap:app_magnets>` (coils).

```{toctree}
:maxdepth: 1
:numbered:

architecture
plasma/index
radial_build/index
exhaust/index
limitations
model_uncertainties
derivations/index
```

```{rubric} References
```

```{footbibliography}
```
