(sec:chap4_uq)=

(ssec:chap4_uqmode)=

# The D0FUS uncertainty mode

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 4.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


Many of the D0FUS input parameters are in fact model calibration parameters, whose values are uncertain: most are estimates or extrapolations rather than measured values. One of the most natural ways to take this into account is to describe them as distribution functions rather than as single numbers. This is what the uncertainty mode of D0FUS, introduced with the execution modes in {ref}`Models <chap:d0fus>`, does: each uncertain input receives a probability distribution instead of a value, the whole design chain is solved for a number of draws in the combined distributions, and the outputs are returned as distributions, together with the fraction of draws for which the design remains feasible and the reason the others fail.

Four distributions are available, illustrated in {numref}`Figure %s <fig:chap4_uq_families>`. A truncated normal, `norm(...)`, describes a quantity with a documented statistical scatter. A uniform law, `unif(lo, hi)`, describes a quantity known only to lie in an interval. A triangular law, `tri(...)`, describes a bracket with a preferred value. Finally `envelope(A `$|$` B `$|$` C)` is not a probability law at all but a discrete model switch, evaluated as a full factorial: it is how a choice between competing closures, such as the three unsettled models of Section {ref}`Model uncertainties <sec:chap4_modelform>`, is carried without pretending that the choice has a mean. The truncated normal accepts two forms, which explains why `norm` entries carry two or three arguments in {numref}`Table %s <tab:uq_presets_summary>`. The form `norm(lo, hi)` centres the law on the design value of the deck, sets the standard deviation to $\sigma = (hi - lo)/4$ and truncates the law at the bounds `lo` and `hi`, which therefore sit at about two standard deviations from a mid-band centre. The form `norm(lo, centre, hi)` sets the centre explicitly, allowing an off-centre or asymmetric law.

:::{figure} /figures/thesis/uq_law_families.png
:name: fig:chap4_uq_families
:width: 95%
:align: center

The four input-law families of the uncertainty mode. From left to right: the truncated normal ($\sigma = (hi-lo)/4$, centred on the deck value), the uniform law, the triangular law, and the discrete envelope, enumerated as a full factorial rather than sampled. The red dashed line marks the design value of the deck.
:::

The sampling follows a Latin hypercube scheme, a stratified variant of Monte-Carlo sampling. The technical details are given in {ref}`Uncertainty-mode input syntax and the default presets <app:uq_presets>`. Each draw is then pushed through the whole design chain and classified. This classification is how the robustness figures of Section 4.4 of the thesis (Figures 4.3 and 4.4) are to be read. In dark green, the design works exactly as proposed. In light green, an operating point still exists but the operator must adapt it: at fixed machine and fusion power, the volume-averaged temperature is moved within the admissible operating window, the density following through the power balance, with no change to the hardware (the mechanism and the window are detailed with the ITER study, Section 4.4.1 of the thesis). In warm colours, red and orange, the draw is blocked by one of the plasma stability limits. In shades of blue, the draw is blocked by the radial build, the gradation showing how deep a reduction of the central-solenoid flux demand, through a shorter flat-top or ramp-up assistance from the heating and current-drive systems for example, would release the design. In grey, finally, no operating point converged at all (a concrete example is given and discussed in the next section).

(ssec:chap4_presets)=
The propagation is only as meaningful as the distributions placed on its inputs, and a word of caution is due before stating them: these distributions are a first attempt, not a settled reference. Where the literature offered usable material, a regression RMSE, a database spread, a documented operational range, the adopted laws are inspired by it. But liberties were taken, and the balance between the optimistic and pessimistic sides of each law ultimately reflects the author’s judgement. The adopted default set is summarised in {numref}`Table %s <tab:uq_presets_summary>`. The full tables with their physical anchors, the sampled marginals, the justification of each entry and the comparison of the adopted widths with the uncertainty studies published around the PROCESS code are collected in {ref}`Uncertainty-mode input syntax and the default presets <app:uq_presets>`.

:::{table} Summary of the adopted default distributions.
:name: tab:uq_presets_summary
:align: center

| Input                              | Distribution                                                | Source       |
|:-----------------------------------|:------------------------------------------------------------|:-------------|
| *Confinement*                      |                                                             |              |
| $H_{98}$                           | norm(0.80, 1.60)                                            | {footcite:p}`iterphysicsbasis1999,doyle2007plasma` |
| scaling law                        | env(IPB98 $|$ ITPA20 $|$ ITPA20-IL)                         | {footcite:p}`verdoolaege2021itpa` |
| *Profiles and pedestal*            |                                                             |              |
| $\nu_n$ (density peaking)          | norm(0.00, 0.40)                                            | {footcite:p}`kim2018iter,angioni2007jetpeaking` |
| $\nu_T$ (temp. peaking)            | norm(2.20, 3.40)                                            | {footcite:p}`kessel2009development,rodriguez2020predictions` |
| $\rho_{ped}$                       | norm(0.92, 0.98)                                            | {footcite:p}`snyder2009pedestal,snyder2011eped` |
| $n_{ped}/\langle n\rangle$         | norm(0.75, 1.00)                                            | {footcite:p}`polevoi2005pellet,kim2018iter` |
| $T_{ped}/\langle T\rangle$         | norm(0.42, 0.68)                                            | {footcite:p}`snyder2011eped` |
| *Helium, impurities and radiation* |                                                             |              |
| $C_\alpha=\tau_{He}^*/\tau_E$      | norm(3.0, 5.7, 10.6)                                        | {footcite:p}`polevoi2005pellet,wade1995helium` |
| core W fraction                    | norm($10^{-5}$, $2\!\times\!10^{-5}$, $8\!\times\!10^{-5}$) | {footcite:p}`kim2018iter` |
| $r_\mathrm{synch}$                 | norm(0.50, 0.80)                                            | {footcite:p}`albajar2001synchrotron` |
| $\rho_\mathrm{rad,core}$           | norm(0.60, 0.85, 1.00)                                      | {footcite:p}`kovari2014process` |
| *Stability limits*                 |                                                             |              |
| $\beta_N$ limit                    | norm(2.60, 3.50)                                            | {footcite:p}`troyon1984mhd,hender2007mhd,wenninger2015advances` |
| $q_{95}$ limit                     | norm(2.00, 2.50, 3.00)                                      | {footcite:p}`devries2009statistical,wenninger2017physics` |
| Greenwald limit                    | norm(0.80, 1.50)                                            | {footcite:p}`greenwald2002density,lang2012highdensity,ding2024high` |
| achievable elongation              | env(Wenn. $|$ Freid. $|$ Blend $|$ Stamb.)                  | {footcite:p}`wenninger2015advances,freidberg2015tokamak,stambaugh1992relation` |
| *Flux budget and current drive*    |                                                             |              |
| $C_e$ (Ejima)                      | norm(0.25, 0.45, 0.58)                                      | {footcite:p}`ejima1982volt,wakatsuki2019safety` |
| $\eta_\mathrm{WP}$                 | norm(0.05, 0.30, 0.45)                                      | {footcite:p}`franke2017heating,fantz2018towards` |
| $\gamma_\mathrm{CD}$               | norm(0.10, 0.35)                                            | {footcite:p}`poli2013eccd,mikkelsen2018survey` |
| *Techno-economics*                 |                                                             |              |
| SC cost factor                     | norm(1.20, 3.00)                                            | {footcite:p}`sheffield2016generic,molodyk2021production` |
| discount rate                      | norm(0.04, 0.10)                                            | {footcite:p}`entler2018approximation,sheffield2016generic` |
:::

```{rubric} References
```

```{footbibliography}
```
