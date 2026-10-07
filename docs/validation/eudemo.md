(ssec:chap2_eudemo)=

# Benchmark on EU-DEMO 2017

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 2.2.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The first full-device exercise targets the EU-DEMO1 2017 baseline established by PROCESS {footcite:p}`eurodemo2017process`. D0FUS receives the same input parameters and the same assumptions as the original run, its Academic options being precisely the ones whose conventions match those of PROCESS: cylindrical volume weighting, ITER-1989 $q_{95}$ definition (Eq. {eq}`eq:q95_iter89`), IPB98(y,2) confinement scaling, Academic radial build and Sauter bootstrap model, with the trapped fraction, the core radiation radius and the radiated fraction subtracted from the loss power aligned likewise. {numref}`Table %s <tab:chap2_eudemo_benchmark>` summarises the comparison. Beyond its validation value, this reproduced baseline is the point the rest of the thesis builds on: the high-field exploration of Chapter 3 of the thesis departs from it.

:::{table} Key D0FUS vs PROCESS outputs for the EU-DEMO1 2017 baseline {footcite:p}`eurodemo2017process`, D0FUS running in the matching Academic options of Section {ref}`General architecture of D0FUS <ssec:chap1_fidelity>`.
:name: tab:chap2_eudemo_benchmark
:align: center

| **Parameter**                            | **Symbol**        | **PROCESS** | **D0FUS** |
|:-----------------------------------------|:------------------|:------------|:----------|
| *Plasma physics*                         |                   |             |           |
| Plasma volume \[m$^3$\]                  | $V_p$             | 2466        | 2543      |
| Plasma current \[MA\]                    | $I_p$             | 19.08       | 19.12     |
| On-axis magnetic field \[T\]             | $B_0$             | 4.89        | 4.89      |
| Safety factor                            | $q_{95}$          | 3.000       | 3.11      |
| Normalised beta (total)                  | $\beta_N$         | 2.89        | 2.69      |
| Poloidal beta (total)                    | $\beta_p$         | 1.14        | 1.02      |
| Confinement time \[s\]                   | $\tau_E$          | 3.878       | 3.49      |
| Thermal energy \[MJ\]                    | $W_\mathrm{th}$   | 1251        | 1250      |
| Vol. avg. density \[$10^{20}$ m$^{-3}$\] | $\bar{n}_e$       | 0.791       | 0.698     |
| Bootstrap fraction                       | $f_\mathrm{bs}$   | 0.387       | 0.509     |
| Energy gain                              | $Q$               | 39.3        | 39.9      |
| Loop voltage \[mV\]                      | $V_\mathrm{loop}$ | 42.2        | 14.1      |
| Total radiated power \[MW\]              | $P_\mathrm{rad}$  | 275         | 223       |
| Power across separatrix \[MW\]           | $P_\mathrm{sep}$  | 156         | 227       |
| Neutron wall load \[MW/m$^2$\]           | $\Gamma_n$        | 1.036       | 1.066     |
| *Engineering (Academic radial build)*    |                   |             |           |
| TF inboard leg thickness \[m\]           | $c$               | 0.96        | 0.92      |
| CS radial thickness \[m\]                | $d$               | 0.80        | 0.25      |
:::

D0FUS lands essentially on the same design point: the plasma current, the field, the stored energy, the gain and the neutron wall load all agree within a few percent, and the inboard TF leg is reproduced to within $4\,\%$ ($0.92$ against $0.96$ m). Unlike the coil benchmarks of the previous section, where the current density and the structural allowable were imposed, only the steel grade and the superconductor are prescribed here: the sizing itself, current densities included, is produced end to end by D0FUS. One quantity disperses, the loop voltage, and with it the solenoid thickness ($0.25$ against $0.80$ m), which follows from it through the flux budget. It stems from an identified model difference: D0FUS integrates the neoclassical conductivity over the profiles and the H-mode pedestal, where PROCESS evaluates a zero-dimensional formula from volume-averaged quantities. Which is closer to the truth cannot be settled between two systems codes, but this baseline happens to have been simulated with integrated modelling: Franza *et al.* {footcite:p}`franza2022mira` apply the MIRA suite (a transport solver inherited from ASTRA coupled to a free-boundary equilibrium) to the very same point. The arbitration is favourable: the dispersion is carried entirely by the plasma resistance, $2.01$ n$\Omega$ here against $4.41$ for PROCESS, and MIRA returns $1.83$: the profile-integrated conductivity is therefore within $10\,\%$ of integrated modelling, where the zero-dimensional formula is more than twice too resistive.

This CS residual deserves a word here, because it does not stay confined to the present benchmark. Chapter 3 of the thesis uses the closure of the inboard build, that is the room left for the central solenoid and the TF inner leg inside $R_0 - a - \Delta_B$, as its main feasibility criterion. A solenoid predicted too thin shifts that criterion in the optimistic direction, and the absolute major radii quoted there should be read with this bias in mind. The comparisons drawn in that chapter are, however, differential: each design lever is evaluated against the same reference protocol and with the same CS model, so the ranking of the levers is far less sensitive to this residual than the absolute figures are. These residuals are attributed to model differences rather than to implementation errors, the bootstrap coefficients having been verified one by one against an independent implementation in Section {ref}`Plasma module <ssec:chap2_plasma_valid>`. Settling the question would however require a reference richer than a single systems-code run: this is what motivates the second and more thorough benchmark, on ITER, for which the literature provides not only systems-code outputs but also, and mainly, more precise modelling works.

```{rubric} References
```

```{footbibliography}
```
