(sssec:chap1_param)=
(app:d0fus_modules)=

# D0FUS module organisation

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page lists the six modules of the library and the domain each one covers, in the order in which a design-point evaluation traverses them.

:::{table} D0FUS modules (v2.1). Line counts are rounded. Function counts include all top-level callable routines (public and internal helpers). All files are prefixed with `D0FUS_`.
:name: tab:module_summary
:align: center

| Module                     | Lines | Functions | Role                |
|:---------------------------|:------|:----------|:--------------------|
| *Library (`D0FUS_BIB/`)*   |       |           |                     |
| `parameterization`         | 500   | —         | Configuration       |
| `physical_functions`       | 7 500 | 121       | Plasma functions    |
| `radial_build_functions`   | 5 000 | 52        | Coil functions      |
| `cost_functions`           | 780   | 19        | Economic functions  |
| `figures`                  | 4 000 | 42        | Plot functions      |
| *Execution (`D0FUS_EXE/`)* |       |           |                     |
| {py:obj}`run <D0FUS_EXE.D0FUS_run.run>`                      | 2 800 | —         | Single design point |
| `scan`                     | 2 300 | —         | 2D parameter sweep  |
| `genetic`                  | 1 300 | —         | Design optimisation |
:::

All third-party dependencies are centralised in a single file (`D0FUS_import.py`), which every other module imports. The library relies on a deliberately limited set of packages from the standard scientific Python ecosystem: NumPy and SciPy for numerical computation (integration, interpolation, root-finding, optimisation), Matplotlib for visualisation, Pandas for tabular data handling, and DEAP for the genetic algorithm. No domain-specific or in-house framework is required, so that the code can be installed and run on any machine with a standard scientific Python distribution (e.g. Anaconda). Python 3.10 or later is required.

The parameterization module defines the interface between the user and the code. Every user-adjustable parameter in D0FUS is gathered into a single Python object called {py:obj}`GlobalConfig <D0FUS_BIB.D0FUS_parameterization.GlobalConfig>`, which contains 145 typed fields grouped into 15 categories. The complete list of fields, with their physical meaning, units and default values, is given in {ref}`GlobalConfig fields <chap:globalconfig>`. Every field carries a physically motivated default value inspired by ITER and EU-DEMO, so that a user who wants to evaluate, say, an ITER-like configuration only needs to specify the handful of parameters that differ from the defaults:

        R0 = 6.2
        a  = 2.0
        Bmax_TF = 11.8
        P_fus = 500
        Tbar = 8.9

All unspecified parameters (confinement scaling law, bootstrap model, steel grade, superconductor type, etc.) silently take their default values. A complete tokamak calculation can thus be set up in a few lines, even by a user encountering the code for the first time.

The plasma physics module gathers all the plasma models needed for a D0FUS run. It covers flux-surface geometry (Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`), radial profiles and volume integrals (Section {ref}`Radial profiles <ssec:chap1_profiles>`), fusion power, density and pressure (Section {ref}`Fusion power, density and pressure <ssec:chap1_fusion>`), magnetic field and plasma beta (Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>`), energy confinement scaling laws (Section {ref}`Power balance and energy confinement time <ssec:chap1_confinement>`), radiation losses (Section {ref}`Radiation losses <ssec:chap1_radiation>`), power balance (Section {ref}`Power balance and energy confinement time <ssec:chap1_power_balance>`), auxiliary heating and current drive (Section {ref}`Plasma current and scaling law <ssec:chap1_heating_cd>`), plasma currents and resistivity (Section {ref}`Plasma current and scaling law <ssec:chap1_currents>`), stability limits (Section {ref}`Stability limits <ssec:chap1_stability>`), and operational constraints and diagnostics (Section {ref}`Heat exhaust, second-order limits and performance <ssec:chap1_operational>`).

The radial build module handles the engineering sizing of the superconducting magnets. It covers the computation of TF coil number and toroidal field ripple (Section {ref}`TF coil number and toroidal field ripple <ssec:chap1_ripple>`), superconductor critical current density scalings and cable current density (Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`), and the mechanical sizing of the TF coil and central solenoid, including the magnetic flux balance (Sections {ref}`Academic model <ssec:chap1_academic>` and {ref}`Refined model <ssec:chap1_refined>`).

The cost module provides the techno-economic evaluation of a converged design point (Section {ref}`Performance: net electric power and cost <ssec:chap1_cost>`). Developed by Mattéo Fletcher (GeePs, CentraleSupélec, private communication, 2026), it implements a classical Sheffield model  {footcite:p}`sheffield2016generic` and a simplified surface-proportional model adapted from Whyte {footcite:p}`whyte2024fusion`. The cost calculation is performed after convergence and does not feed back into the physics.

The figures module provides the graphical output of the code. It produces two types of figures: ten diagnostic plots generated automatically after each single-point calculation (plasma cross-section, flux surfaces, kinetic profiles, $q(\rho)$, radiation profiles, TF and CS engineering drawings), and a library of standalone parametric plots for model comparisons and benchmark tables.

```{rubric} References
```

```{footbibliography}
```
