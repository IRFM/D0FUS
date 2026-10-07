(ssec:chap2_build_valid)=

# Radial build module

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 2.1.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The refined radial build models of D0FUS (see Section {ref}`Refined model <ssec:chap1_refined>`) are validated against published values, a dedicated magnet design code, and six constructed machines or highly refined designs, following the methodology published in {footcite:p}`auclair2026mechanical`. Parts of this section are reproduced from that publication with minor adaptations.

(ssec:chap2_Jc_bench)=

## Superconductor critical-current benchmark

The engineering current density that feeds the TF and CS sizings derives from the three $J_{\text{non-Cu}}$ scaling laws of Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`. Evaluated at 4.2 K, they cover complementary field ranges ({numref}`Fig. %s <fig:Jc_scaling>`): NbTi up to about 9 T, Nb$_3$Sn up to about 14 T, and REBCO well beyond 20 T. The NHMFL maglab data {footcite:p}`maglab2024` overlaid in {numref}`Fig. %s <fig:Jc_scaling>` agree quantitatively with the NbTi and Nb$_3$Sn scalings. For REBCO, the D0FUS curve sits about a factor of two above the older NHMFL SuperPower SP26 reference. The D0FUS scaling is anchored on tapes measured in 2019 {footcite:p}`senatore2024rebco`, and the offset simply reflects the progress of the technology in the intervening years. Nevertheless, the field-dependence trend agrees with the maglab data.

:::{figure} /figures/thesis/J_c_Maglab.png
:name: fig:Jc_scaling
:width: 85%
:align: center

Strand and tape current density of the three superconductor scalings implemented in D0FUS, evaluated at 4.2 K. NHMFL maglab data {footcite:p}`maglab2024` (transparent, 2011 vintage) is overlaid as an experimental reference. The REBCO scaling itself is not anchored on that series but on two more recent tapes measured at Geneva {footcite:p}`senatore2024rebco`, a Fujikura EuBCO tape (sample 19-0008) and a SuperOx YBCO tape (sample \#337-R), both received in 2019, which reach $1220$ A cm$^{-1}$ of critical current per unit width at $19$ T and $4.2$ K.
:::

{numref}`Table %s <tab:Jc_validation>` summarises a benchmark against ITER conductor specifications and commercial tape data: the agreement is within a few percent on each reference point.

:::{table} Benchmark of the $J_{\text{non-Cu}}$ scaling laws implemented in D0FUS against reference specifications.
:name: tab:Jc_validation
:align: center

| **Superconductor**    | **Conditions**                          | **D0FUS**     | **Reference**          | **Source**   |
|:----------------------|:----------------------------------------|:--------------|:-----------------------|:-------------|
| NbTi (ITER PF)        | 5 T, 4.2 K                              | 3057 A/mm$^2$ | $\approx$2900 A/mm$^2$ | {footcite:p}`devred2012nbti` |
| Nb$_3$Sn (ITER TF)    | 11.8 T, 4.2 K, $\varepsilon = -0.3\,\%$ | 929 A/mm$^2$  | $\approx$900 A/mm$^2$  | {footcite:p}`devred2012nb3sn` |
| REBCO (Fujikura 2019) | 19 T, 4.2 K, $B \perp$                  | 2000 A/mm$^2$ | $\approx$2000 A/mm$^2$ | {footcite:p}`senatore2024rebco` |
:::

## TF coil comparison with MADE

MADE (MAgnet Design Explorer) is a parametric optimisation tool developed for designing magnet systems for tokamaks {footcite:p}`giannini2023magnet`. It accounts for electromagnetic, structural, and superconducting constraints. MADE has been cross-checked against other pre-sizing tools such as the CEA magnet design code MADMACS {footcite:p}`torre2016tools,sutcliffe2025magnet`, and its outputs have been found consistent with detailed finite element analyses carried out downstream on specific design points, notably for the EU-DEMO TF and CS coils.

The D0FUS TF coil models are benchmarked against Fig. 19 of Ref. {footcite:p}`giannini2023magnet`. The same magnetic field scan is performed, considering an outer radius of the TF coil inner leg of 4.3 m, a wedging architecture, a round-in-square CICC with HTS superconductor (T = 20 K), a hot spot temperature of $250$ K and an austenitic steel ($\sigma_{\text{lim}} = 867$ MPa). The resulting thicknesses are compared in {numref}`Fig. %s(a) <fig:chap2_MADE_benchmark>`.

:::{figure} /figures/thesis/fig_chap2_MADE_benchmark_composite.png
:name: fig:chap2_MADE_benchmark
:width: 95%
:align: center

Comparison of the predicted thicknesses from the D0FUS Academic and Refined models with the MADE code in wedging configuration, using reference data from Ref. {footcite:p}`giannini2023magnet` fig. 19 for the TF coils and Ref. {footcite:p}`sarasola2020progress` fig. 2 for the CS. (a) TF coil: scan in maximum field on the inner leg; (b) CS: scan in magnetic flux $\Psi_{\rm CS}$.
:::

One can observe a quasi-perfect agreement between the Refined model and MADE. On the other hand, the Academic model reproduces the qualitative trend well but is not quantitatively accurate at high magnetic field, as expected given its simplified assumptions, notably the lack of steel in the winding pack.

## CS comparison with MADE

The D0FUS CS models are benchmarked against Fig. 2 of Ref. {footcite:p}`sarasola2020progress`. Note that the original figure is expressed in terms of the half-swing flux, whereas D0FUS uses the full-swing convention (Eq. {eq}`eq:Psi_CS`). The abscissa values of {numref}`Fig. %s(b) <fig:chap2_MADE_benchmark>` are therefore twice those read from the original reference.

The same magnetic flux scan is conducted, with every characteristic taken from the reference paper {footcite:p}`sarasola2020progress`: an outer radius of the CS of 2.7 m, a CS height $H_{\rm CS} = 17.92$ m (EU-DEMO baseline 2018 allocation), a wedging architecture, an HTS superconductor, and austenitic steel with a fatigue-reduced allowable stress $\sigma_{\text{eff}} = 300$ MPa. The resulting thicknesses are presented in {numref}`Fig. %s(b) <fig:chap2_MADE_benchmark>`.

One can observe a good match between the Refined model and MADE, although the Refined model is slightly more optimistic. On the other hand, the Academic model reproduces the qualitative trends well but is not quantitatively accurate, especially at high magnetic flux. The asymptotic behaviour visible at high flux in {numref}`Fig. %s(b) <fig:chap2_MADE_benchmark>` reflects the nonlinear feedback loop discussed in the last paragraph of Section {ref}`CS comparison with reference designs <ssec:chap2_CS_bench>`.

(ssec:chap2_TF_bench)=

## TF coil comparison with reference designs

The Refined model predictions are now compared against built machines and highly refined designs spanning a range of sizes, fields and mechanical architectures. The inputs taken from the literature are, for each machine, the outer radius of the TF inner leg $(R_0 - a - \Delta_{B})$, the steel grade and its allowable, the cable current density $J_{\rm TF}^{\rm wost}$ and the target peak field. Values that could not directly be found are estimated and marked $^{\ddagger}$. The current densities are frozen at the design values of each reference, with no correction for the field dependence of the critical current: they are inputs, not predictions. This is deliberate. These coils were procured decades apart, from different suppliers and different strand generations, so predicting each one would amount to modelling the state of the superconducting industry at the date of its order rather than the coil itself.

:::{table} TF coil benchmark. References: ITER {footcite:p}`sborchia2008design,mitchell2011iter`, EU-DEMO {footcite:p}`federici2024relationship,eurodemo2017process`, JT60-SA {footcite:p}`nannini2010mechanical,yoshida2010design,tsuchiya2008design`, EAST {footcite:p}`chen2008design,wei2010east`, ARC {footcite:p}`sorbom2015arc,Sanabria2024PITVIPER`, SPARC {footcite:p}`creely2020overview,hartwig2023sparc,Sanabria2024PITVIPER`.
:name: tab:chap2_tf_coil_benchmark
:align: center

|                                    | **ITER**        | **EU-DEMO**     | **JT60-SA**     | **EAST**         | **ARC V1**        | **SPARC**         |
|:-----------------------------------|:----------------|:----------------|:----------------|:-----------------|:------------------|:------------------|
| Inputs                             |                 |                 |                 |                  |                   |                   |
| Configuration                      | Wedging         | Wedging         | Wedging         | Wedging          | Plug              | Bucking           |
| $R_0$ (m)                          | 6.20            | 8.94            | 2.96            | 1.85             | 3.30              | 1.85              |
| $a$ (m)                            | 2.00            | 2.88            | 1.18            | 0.45             | 1.10              | 0.57              |
| $\Delta_{B}$ (m)                   | 1.10            | 1.82            | 0.46            | 0.15             | 0.86              | 0.26              |
| Superconductor                     | Nb$_3$Sn        | Nb$_3$Sn        | NbTi            | NbTi             | REBCO             | REBCO             |
| $J_{\rm TF}^{\rm wost}$ (MA/m$^2$) | 35$^{\ddagger}$ | 30$^{\ddagger}$ | 51$^{\ddagger}$ | 36$^{\ddagger}$  | 216$^{\ddagger}$  |                   |
| Steel                              | 316LN           | 316L            | 316LN           | 316LN            | N50H$^{\ddagger}$ |                   |
| $\sigma_{\rm lim}$ (MPa)           | 660             | 600             | 547             | 547$^{\ddagger}$ | 1000$^{\ddagger}$ | 1000$^{\ddagger}$ |
| $B_{\text{max}}$ (T)               | 11.8            | 10.6            | 5.65            | 5.8              | 23                | 20                |
| TF coil thickness (m)              |                 |                 |                 |                  |                   |                   |
| D0FUS                              | 0.94            | 1.00            | 0.24            | 0.30             | 0.50$^{\ddagger}$ | 0.34$^{\ddagger}$ |
| Published                          | 0.91            | 0.96            | 0.25            | 0.35             | 0.64              | 0.33              |
| $\Delta$ (m)                       | $+0.03$         | $+0.04$         | $-0.01$         | $-0.05$          | —                 | —                 |
| $\Delta$ (%)                       | $+3$            | $+4$            | $-4$            | $-14$            | $-23$             | $+6$              |
:::

Note that EAST publishes its steel grade but no allowable, so the $547$ MPa of JT60-SA, whose grade is the same and whose value is published as such {footcite:p}`tsuchiya2008design`, is carried over.

Also note that the two CFS columns are only estimates, since neither publishes its winding pack, and they are kept because they are the only refined designs available in bucking and plug configuration and at high field with REBCO. The PIT-VIPER conductor is adopted {footcite:p}`Sanabria2024PITVIPER`, the only REBCO cable for compact tokamaks whose operating parameters are published: an $18$ mm copper former in a $23$ mm square steel jacket with $0.5$ mm of turn insulation, which at the quoted $113$ A/mm$^2$ on the total section gives $J^{\rm wost} = 216$ A/mm$^2$ once the jacket is excluded, as the D0FUS convention requires. The same value is used for the solenoid. The stacked soldered tapes leave essentially no interstrand void, so the dilution chain of Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>` is evaluated at $f_\mathrm{void} \approx 0$ rather than at the $\approx 0.33$ of round LTS strands. Finally, both CFS coils are wound from stacked tapes, an architecture in which the conductor itself almost certainly carries part of the mechanical load, whereas D0FUS assigns all of it to the steel. The effect is ignored here for want of a published basis, and the consequence is accepted: what these two columns are asked to establish is an order of magnitude, which is what a global sizing needs, the optimisation of the winding pack coming afterwards.

The four wedged machines are reproduced within $15\,\%$ on the complete inboard leg, and three of them within $4\,\%$. The two compact high-field columns also close, over a wider spread, at $-23\,\%$ for ARC and $+6\,\%$ for SPARC. Given the range of configurations, steels and sizes this benchmark spans, that accuracy is, in the author’s view, adequate for a pre-design tool.

(ssec:chap2_CS_bench)=

## CS comparison with reference designs

The same methodology is applied to the solenoid. The inputs taken from the literature are, for each machine, the flux $\Psi_{\rm CS}$, the outer radius of the solenoid, its height, the mechanical architecture, the steel grade and its allowable, with the same convention on the current densities as above. The fatigue knockdown defined in Section {ref}`Refined model <ssec:chap1_refined>` is applied in the wedging and light bucking cases only.

:::{table} CS benchmark. References: ITER {footcite:p}`libeyre2009detailed`, EU-DEMO {footcite:p}`federici2024relationship,sarasola2023parametric,eurodemo2017process`, JT60-SA {footcite:p}`nannini2010mechanical,yoshida2010design,tsuchiya2008design`, EAST {footcite:p}`wu2003east,wu2008recent,guo2016divertor`, ARC {footcite:p}`sorbom2015arc,Sanabria2024PITVIPER`, SPARC {footcite:p}`creely2020overview,Sanabria2024PITVIPER,wang2024structure,diazpacheco2025electromechanical`.
:name: tab:chap2_cs_coil_benchmark
:align: center

|                                    | **ITER**        | **EU-DEMO**     | **JT60-SA**     | **EAST**         | **ARC V1**       | **SPARC**           |
|:-----------------------------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:--------------------|
| Inputs                             |                 |                 |                 |                  |                  |                     |
| Configuration                      | Wedging         | Wedging         | Wedging         | Wedging          | Plug             | Bucking             |
| $R_0$ (m)                          | 6.20            | 8.94            | 2.96            | 1.85             | 3.30             | 1.85                |
| $a$ (m)                            | 2.00            | 2.88            | 1.18            | 0.45             | 1.10             | 0.57                |
| $\Delta_{B}$ (m)                   | 1.10            | 1.82            | 0.46            | 0.15             | 0.86             | 0.26                |
| $\Delta_{TF}$ (m)                  | 0.91            | 0.96            | 0.26            | 0.35             | 0.64             | 0.33                |
| Steel                              | JK2LB           | 316L            | 316LN           | 316LN            | 316LN            | CHSN01$^{\ddagger}$ |
| $\sigma_{\rm lim}$ (MPa)           | 667             | 600             | 547             | 547$^{\ddagger}$ | 700              | 1000$^{\ddagger}$   |
| Superconductor                     | Nb$_3$Sn        | Nb$_3$Sn        | Nb$_3$Sn        | NbTi             | REBCO            | REBCO               |
| $J_{\rm CS}^{\rm wost}$ (MA/m$^2$) | 45$^{\ddagger}$ | 60$^{\ddagger}$ | 45$^{\ddagger}$ | 36$^{\ddagger}$  | 216$^{\ddagger}$ | 216$^{\ddagger}$    |
| $\Psi_{CS}$ (Wb)                   | 233             | 500             | 40              | 10               | 32               | 42                  |
| CS thickness                       |                 |                 |                 |                  |                  |                     |
| D0FUS (m)                          | 0.75            | 0.76            | 0.37            | 0.12             | 0.14             | 0.28                |
| Published (m)                      | 0.75            | 0.81            | 0.34            | 0.16             | 0.25             | 0.25                |
| $\Delta$ (m)                       | 0.00            | $-0.05$         | $+0.03$         | $-0.04$          | $-0.11$          | $+0.04$             |
| $\Delta$ (%)                       | $0$             | $-6$            | $+9$            | $-24$            | $-43$            | $+13$               |
| $B_{CS}$ results                   |                 |                 |                 |                  |                  |                     |
| D0FUS (T)                          | 12.4            | 10.2            | 9.6             | 3.8              | 12.8             | 21.3                |
| Published (T)                      | 13              | 11.4            | 8.9             | 4.5              | 12.9             | 25                  |
| $\Delta$ (T)                       | $-0.6$          | $-1.2$          | $+0.7$          | $-0.7$           | $-0.1$           | $-3.7$              |
| $\Delta$ (%)                       | $-5$            | $-10$           | $+8$            | $-16$            | $-1$             | $-15$               |
:::

Five of the six solenoids are reproduced within $25\,\%$ on thickness and all six within $20\,\%$ on peak field, across windings from $0.16$ to $0.81$ m and fields from $4.5$ to $25$ T, ITER, EU-DEMO and JT60-SA agreeing within $10\,\%$ on both. ARC is the exception on thickness, at $-43\,\%$, which the assumptions its column carries and the sensitivity discussed below would lead one to expect. Nevertheless, here again, the orders of magnitude are recovered over the whole range, which is what this benchmark set out to establish.

One remark is worth making. In the author’s experience the result is markedly sensitive to its inputs, because the solenoid feeds back on itself: gaining flux calls for a higher field, hence more structure and more superconductor, both of which thicken the winding and eat into the bore that generates the flux. This loop, visible as the asymptotic behaviour of {numref}`Fig. %s(b) <fig:chap2_MADE_benchmark>`, is part of the reason why the two least documented columns scatter most.

```{rubric} References
```

```{footbibliography}
```
