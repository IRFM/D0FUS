(appendixinput)=

# D0FUS reference inputs

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.8. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


{numref}`Table %s <tab:input_parameters>` summarises the reference input parameters used throughout this study, calibrated on the EU-DEMO1 2017 baseline (PROCESS v1.0.10 {footcite:p}`kovari2016process`, run EU 2NDSKT v1.0 {footcite:p}`eurodemo2017process`) after aligning all input conventions between the two codes. This single parameter set defines the starting point from which all subsequent scans are derived. Parameters marked with $\dagger$ vary across configurations in Chapter 3 of the thesis. All others remain fixed.

:::{table} Reference input parameters calibrated on the EU-DEMO1 2017 baseline {footcite:p}`eurodemo2017process`. $\dagger$: varied across configurations in Chapter 3 of the thesis.
:name: tab:input_parameters
:align: center

| **Parameter**                        | **Sym.**                             | **Value**                           | **Unit**                     |
|:-------------------------------------|:-------------------------------------|:------------------------------------|:-----------------------------|
| *Geometry and power*                 |                                      |                                     |                              |
| Fusion power                         | $P_\mathrm{fus}$                     | 1998                                | MW                           |
| Major radius                         | $R_0$                                | 8.9                                 | m                            |
| Minor radius                         | $a$                                  | 2.8                                 | m                            |
| Inboard radial build                 | $\Delta_B$                           | 1.40                                | m                            |
| Elongation model                     | —                                    | Wenninger *et al.* {footcite:p}`wenninger2015advances`     | —                            |
| Plasma geometry                      | —                                    | Elliptical                          | —                            |
| *Confinement and transport*          |                                      |                                     |                              |
| Energy conf. scaling                 | —                                    | IPB98(y,2) {footcite:p}`iterphysicsbasis1999`             | —                            |
| H-factor                             | $H$                                  | 1.1                                 | —                            |
| $q_{95}$ formula                     | —                                    | ITER_1989 {footcite:p}`uckan1990iter`              | —                            |
| Bootstrap model                      | —                                    | Sauter *et al.* {footcite:p}`sauter1999neoclassical`        | —                            |
| Trapped fraction                     | —                                    | ASTRA {footcite:p}`fable_astra`                  | —                            |
| *Plasma profiles*                    |                                      |                                     |                              |
| Average temperature                  | $\bar{T}$                            | 12.82                               | keV                          |
| Density peaking                      | $\nu_n$                              | 1.0                                 | —                            |
| Temperature peaking                  | $\nu_T$                              | 1.45                                | —                            |
| Pedestal radius                      | $\rho_\mathrm{ped}$                  | 0.94                                | —                            |
| Ped. density fraction                | $n_\mathrm{ped}/\bar{n}$             | 0.78                                | —                            |
| Ped. temperature fraction            | $T_\mathrm{ped}/\bar{T}$             | 0.43                                | —                            |
| *Plasma composition and radiation*   |                                      |                                     |                              |
| Effective charge                     | $Z_\mathrm{eff}$                     | 2.18                                | —                            |
| Impurity species                     | —                                    | Xe, W                               | —                            |
| Impurity fractions                   | $f_\mathrm{Xe},f_\mathrm{W}$         | $3.5\times10^{-4},\,5\times10^{-5}$ | —                            |
| Core rad. boundary                   | $\rho_\mathrm{rad,core}$             | 0.75                                | —                            |
| Core rad. fraction                   | —                                    | 0.6                                 | —                            |
| Synchrotron reflection               | $r_\mathrm{syn}$                     | 0.6                                 | —                            |
| $\alpha$ conf. factor                | $C_\alpha$                           | 6.8                                 | —                            |
| *Magnets$^\dagger$*                  |                                      |                                     |                              |
| Peak TF field                        | $B_\mathrm{max}$                     | 10.5                                | T                            |
| Superconductor                       | —                                    | Nb$_3$Sn                            | —                            |
| Helium temperature                   | $T_\mathrm{He}$                      | 4.75                                | K                            |
| Temperature margin                   | $\Delta T_\mathrm{Nb_3Sn}$           | 1.5                                 | K                            |
| Conductor current                    | $I_\mathrm{cond}$                    | 90                                  | kA                           |
| Strain on SC                         | $\varepsilon$                        | $-6.6\times10^{-3}$                 | —                            |
| *CICC helium fractions*              |                                      |                                     |                              |
| Cooling channel fraction             | $f_\mathrm{CC}$                      | 0.10                                | —                            |
| Void fraction                        | $f_\mathrm{void}$                    | 0.30                                | —                            |
| Insulation fraction                  | $f_\mathrm{In}$                      | 0.15                                | —                            |
| *Mechanical configuration$^\dagger$* |                                      |                                     |                              |
| Radial build model                   | —                                    | Refined                             | —                            |
| Type of architecture                 | —                                    | Wedging                             | —                            |
| Steel grade                          | —                                    | 316L                                | —                            |
| CS fatigue knockdown                 | —                                    | 2.0                                 | —                            |
| *Operation and current drive*        |                                      |                                     |                              |
| Operation mode                       | —                                    | Pulsed                              | —                            |
| Plateau duration                     | $t_\mathrm{plateau}$                 | 7200                                | s                            |
| Auxiliary heating                    | $P_\mathrm{aux}$                     | 50                                  | MW                           |
| CD efficiency                        | $\gamma_\mathrm{CD}$                 | 0.30                                | $10^{20}$ A W$^{-1}$m$^{-2}$ |
| Wall-plug efficiency                 | $\eta_\mathrm{WP}$                   | 0.40                                | —                            |
| Ramp-up CD fraction                  | $f_h$                                | 0                                   | —                            |
| CS swing usable fraction             | $f_\mathrm{swing}^{\mathrm{usable}}$ | 0.75                                | —                            |
| *Stability limits*                   |                                      |                                     |                              |
| $\beta_N$ limit                      | —                                    | 2.9                                 | —                            |
| $q_{95}$ limit                       | —                                    | 3.0                                 | —                            |
| Greenwald limit                      | $f_\mathrm{GW,lim}$                  | 1.2                                 | —                            |
:::

```{rubric} References
```

```{footbibliography}
```
