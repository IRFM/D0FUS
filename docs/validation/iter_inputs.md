(appendixinput_iter)=

# D0FUS ITER benchmark inputs

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.9. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


{numref}`Table %s <tab:input_parameters_iter>` lists the full parameter set of the ITER Q = 10 inductive benchmark of Section {ref}`Benchmark on ITER <ssec:chap2_iter>`, run from the deck `1_run_ITER.txt`. Every entry is supplied to D0FUS as a primary input or a modelling convention. All the figures of merit reported in the benchmark are solved from this set alone.

:::{table} Input parameters of the ITER Q = 10 inductive benchmark of Section {ref}`Benchmark on ITER <ssec:chap2_iter>`.
:name: tab:input_parameters_iter
:align: center

| **Parameter**                                        | **Sym.**                             | **Value**                         | **Unit** |
|:-----------------------------------------------------|:-------------------------------------|:----------------------------------|:---------|
| *Geometry and power*                                 |                                      |                                   |          |
| Fusion power                                         | $P_\mathrm{fus}$                     | 500                               | MW       |
| Major radius                                         | $R_0$                                | 6.2                               | m        |
| Minor radius                                         | $a$                                  | 2.0                               | m        |
| Inboard radial build                                 | $\Delta_B$                           | 1.10                              | m        |
| Elongation model                                     | —                                    | Wenninger *et al.* {footcite:p}`wenninger2015advances`   | —        |
| Plasma geometry                                      | —                                    | Miller (Refined)                  | —        |
| *Confinement and transport*                          |                                      |                                   |          |
| Energy conf. scaling                                 | —                                    | IPB98(y,2) {footcite:p}`iterphysicsbasis1999`           | —        |
| H-factor                                             | $H$                                  | 1.0                               | —        |
| $q_{95}$ formula                                     | —                                    | ITER_1989 {footcite:p}`uckan1990iter`            | —        |
| Bootstrap model                                      | —                                    | Sauter-Redl {footcite:p}`sauter1999neoclassical`          | —        |
| $q$-profile                                          | —                                    | Refined (self-consistent)         | —        |
| *Plasma profiles*                                    |                                      |                                   |          |
| Greenwald fraction                                   | $\bar n/n_\mathrm{GW}$               | 0.85                              | —        |
| Density peaking                                      | $\nu_n$                              | 0.01                              | —        |
| Temperature peaking                                  | $\nu_T$                              | 2.80                              | —        |
| Pedestal radius                                      | $\rho_\mathrm{ped}$                  | 0.95                              | —        |
| Ped. density fraction                                | $n_\mathrm{ped}/\bar{n}$             | 0.99                              | —        |
| Ped. temperature fraction                            | $T_\mathrm{ped}/\bar{T}$             | 0.55                              | —        |
| *Plasma composition and radiation*                   |                                      |                                   |          |
| Effective charge                                     | $Z_\mathrm{eff}$                     | 1.65                              | —        |
| Impurity species                                     | —                                    | W, Ne                             | —        |
| Impurity fractions                                   | $f_\mathrm{W},f_\mathrm{Ne}$         | $2\times10^{-5},\,7\times10^{-3}$ | —        |
| Core/edge rad. boundary                              | $\rho_\mathrm{rad}$                  | 0.85                              | —        |
| Synchrotron reflection                               | $r_\mathrm{syn}$                     | 0.6                               | —        |
| $\alpha$ conf. ratio ($\tau_\mathrm{He}^{*}/\tau_E$) | $C_\alpha$                           | 5.7                               | —        |
| *Magnets*                                            |                                      |                                   |          |
| Peak TF field (conductor)                            | $B_\mathrm{max}$                     | 11.5                              | T        |
| Backplate cover                                      | $c_\mathrm{BP}$                      | 0.07                              | m        |
| Superconductor                                       | —                                    | Nb$_3$Sn                          | —        |
| Helium temperature                                   | $T_\mathrm{He}$                      | 4.5                               | K        |
| Temperature margin                                   | $\Delta T_\mathrm{Nb_3Sn}$           | 1.5                               | K        |
| Conductor current                                    | $I_\mathrm{cond}$                    | 68                                | kA       |
| Strain on SC                                         | $\varepsilon$                        | $-6\times10^{-3}$                 | —        |
| *CICC helium fractions*                              |                                      |                                   |          |
| Cooling channel fraction                             | $f_\mathrm{CC}$                      | 0.10                              | —        |
| Void fraction                                        | $f_\mathrm{void}$                    | 0.30                              | —        |
| Insulation fraction                                  | $f_\mathrm{In}$                      | 0.15                              | —        |
| *Mechanical configuration*                           |                                      |                                   |          |
| Radial build model                                   | —                                    | Refined                           | —        |
| Type of architecture                                 | —                                    | Wedging                           | —        |
| Steel grade                                          | —                                    | 316L                              | —        |
| CS fatigue knockdown                                 | —                                    | 2.0                               | —        |
| *Operation and current drive*                        |                                      |                                   |          |
| Operation mode                                       | —                                    | Pulsed                            | —        |
| Plateau duration                                     | $t_\mathrm{plateau}$                 | 450                               | s        |
| Auxiliary heating                                    | $P_\mathrm{aux}$                     | 50                                | MW       |
| NBI / EC / IC                                        | —                                    | $33 / 6.7 / 10$                   | MW       |
| Ejima coefficient                                    | $C_e$                                | 0.45                              | —        |
| Wall-plug efficiency                                 | $\eta_\mathrm{WP}$                   | 0.40                              | —        |
| Ramp-up CD fraction                                  | $f_h$                                | 0                                 | —        |
| CS swing usable fraction                             | $f_\mathrm{swing}^{\mathrm{usable}}$ | 0.75                              | —        |
| *Stability limits*                                   |                                      |                                   |          |
| $\beta_N$ limit                                      | —                                    | 2.8                               | —        |
| Kink $q$ limit                                       | —                                    | 2.5                               | —        |
| Greenwald limit                                      | $f_\mathrm{GW,lim}$                  | 1.0                               | —        |
:::

```{rubric} References
```

```{footbibliography}
```
