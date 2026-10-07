(ssec:chap2_plasma_valid)=

# Plasma module

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 2.1.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


## Flux-surface geometry.

The plasma volume is computed at the published ITER separatrix shaping {footcite:p}`shimada2007progress` in both geometry modes presented in Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`: the Academic shaped-torus formula and the Refined Miller integration both land within $2\,\%$ of the design value, and within $0.1\,\%$ of each other, so the cheap Academic formula loses essentially nothing on this quantity. The three elongation scalings presented in Section {ref}`Flux-surface geometry <ssec:chap1_geometry>` are evaluated at the ITER aspect ratio ($A=3.1$): the default Wenninger scaling {footcite:p}`wenninger2015advances` lands within $2\,\%$ of the design value, the Freidberg scaling {footcite:p}`freidberg2015tokamak,lee2015vertical` overshoots by some $7\,\%$, and the Stambaugh fit {footcite:p}`stambaugh1992relation` sits above both. The triangularity follows the single implemented model of the TREND code {footcite:p}`hartmann2013development` (Eq. {eq}`eq:delta_from_kappa`), landing within $10\,\%$ of the ITER value: the least accurate prediction of this module, but an acceptable one for a quantity whose design impact is second order in D0FUS.

:::{table} Flux-surface geometry of D0FUS against the published ITER separatrix.
:name: tab:chap2_geometry_check
:align: center

| Quantity                                             | D0FUS | ITER  | Source       |
|:-----------------------------------------------------|:------|:------|:-------------|
| $V_\mathrm{plasma}$, Academic mode \[m$^3$\]         | 844   | 831   | {footcite:p}`shimada2007progress` |
| $V_\mathrm{plasma}$, Refined mode \[m$^3$\]          | 843   | 831   | {footcite:p}`shimada2007progress` |
| $\kappa_\mathrm{sep}$, Wenninger ($m_s = 0.3$) \[-\] | 1.88  | 1.85  | {footcite:p}`shimada2007progress` |
| $\kappa_\mathrm{sep}$, Freidberg \[-\]               | 1.97  | 1.85  | {footcite:p}`shimada2007progress` |
| $\kappa_\mathrm{sep}$, Stambaugh \[-\]               | 2.30  | 1.85  | {footcite:p}`shimada2007progress` |
| $\delta_\mathrm{sep}$ \[-\]                          | 0.53  | 0.485 | {footcite:p}`shimada2007progress` |
:::

## Fusion reactivity.

The D-T reactivity implements the Bosch-Hale parameterisation {footcite:p}`bosch1992improved` presented in Section {ref}`Fusion power, density and pressure <ssec:chap1_fusion>`, tested against the tabulated values of the same paper (Table VIII) and against the independent `cfspopcon` implementation {footcite:p}`body2024detachment`. Both sit on the D0FUS curve over the full $1$ to $100$ keV range, five decades in reactivity ({numref}`Fig. %s <fig:chap2_sigmav>`).

:::{figure} /figures/thesis/chap2_sigmav_validation.png
:name: fig:chap2_sigmav
:width: 100%
:align: center

D-T reactivity implemented in D0FUS compared to the tabulated values of Bosch and Hale {footcite:p}`bosch1992improved` (Table VIII) and independent `cfspopcon` implementation {footcite:p}`body2024detachment`.
:::

## Radiation and atomic data.

The radiative cooling rates $L_z(T_e)$ and the mean charge state $\langle Z\rangle(T_e)$, implemented from the Mavrin fits {footcite:p}`mavrin2018improved` presented in Section {ref}`Radiation losses <ssec:chap1_radiation>`, are compared in {numref}`Fig. %s <fig:chap2_radiation>` to OpenADAS coronal equilibrium values {footcite:p}`summers1994adas` and to the `TORAX` implementation {footcite:p}`citrin2024torax`, covering species from helium to krypton. To test the implementation, D0FUS is compared to `TORAX`, which codes the same Mavrin fits: the two agree to a fraction of a percent. To test the physics, it is compared to OpenADAS, whose coronal equilibrium is solved independently. Departures exist, but remain small: within $6\,\%$ over the core-relevant window, larger only at the two ends of the range, which concern the scrape-off layer or lie beyond the validity of the reference tables.

:::{figure} /figures/thesis/chap2_radiation_validation.png
:name: fig:chap2_radiation
:width: 100%
:align: center

Atomic data implemented in D0FUS from the Mavrin fits {footcite:p}`mavrin2018improved` (lines), for species from helium to krypton: coronal radiative cooling rate $L_z(T_e)$ (a) and mean charge state $\langle Z\rangle(T_e)$ (b), each compared to OpenADAS coronal equilibrium values {footcite:p}`summers1994adas` and to the `TORAX` implementation {footcite:p}`citrin2024torax`. OpenADAS is the physical reference. `TORAX` shares the same fits, so its curves check the implementation.
:::

## Current drive.

The LHCD, ECCD and NBCD efficiencies presented in Section {ref}`Plasma current and scaling law <ssec:chap1_heating_cd>` are benchmarked against METIS {footcite:p}`artaud2018metis`, whose current-drive module was run alone at ITER-like parameters to produce the reference values collected below. The LH and EC efficiencies reproduce the METIS values to the digits shown, and the NBI efficiency lands within $2\,\%$. These three lines do not all carry the same weight: the LHCD model implemented in D0FUS is the default METIS model itself (Section {ref}`Plasma current and scaling law <ssec:chap1_heating_cd>`), so the LH line checks the implementation rather than the physics, whereas the NBCD efficiency is rebuilt from an independent slowing-down chain, which makes its $2\,\%$ residual a model-to-model comparison.

:::{table} Normalised current-drive efficiencies of D0FUS against METIS, run alone at ITER $Q=10$ parameters.
:name: tab:chap2_cd_check
:align: center

| Quantity                                                | D0FUS | METIS (ITER $Q=10$) |
|:--------------------------------------------------------|:------|:--------------------|
| $\gamma_\mathrm{LH}$ \[$10^{20}$ A W$^{-1}$ m$^{-2}$\]  | 0.33  | 0.33                |
| $\gamma_\mathrm{EC}$ \[$10^{20}$ A W$^{-1}$ m$^{-2}$\]  | 0.167 | 0.167               |
| $\gamma_\mathrm{NBI}$ \[$10^{20}$ A W$^{-1}$ m$^{-2}$\] | 0.340 | 0.334               |
:::

## Bootstrap.

The Sauter-Redl bootstrap model {footcite:p}`sauter1999neoclassical,redl2021bootstrap` implemented in D0FUS (Section {ref}`Plasma current and scaling law <ssec:chap1_currents>`) is benchmarked against the independent `TORAX` implementation {footcite:p}`citrin2024torax`, as shown in {numref}`Fig. %s <fig:chap2_bootstrap>`. The reference points sit on the D0FUS curves for all four coefficients, over five decades of collisionality and for the three trapped fractions, including the sign change and peak of $L_{32}$ and the steep collisional rise of $\alpha$: the implementation reproduces the reference to plotting accuracy, with no region of disagreement.

:::{figure} /figures/thesis/chap2_bootstrap_validation.png
:name: fig:chap2_bootstrap
:width: 100%
:align: center

Sauter bootstrap-current coefficients $L_{31}, L_{32}, L_{34}$ and $\alpha$ implemented in D0FUS {footcite:p}`sauter1999neoclassical` (lines), versus collisionality for three trapped fractions $f_t$ at $Z_\mathrm{eff}=2$, compared to the independent `TORAX` implementation {footcite:p}`citrin2024torax` (markers).
:::

## Bootstrap fraction, integral comparison.

Beyond the coefficients, the integral quantity can be compared against integrated modelling wherever profiles are documented. At the ITER reference point, D0FUS returns a bootstrap fraction of $29.7\,\%$, that is $4.67$ MA of the $15.70$ MA of plasma current, against the $20\,\%$ of the JINTRAC simulations of Ref. {footcite:p}`kim2018iter` and the $15$ to $25\,\%$ band of Ref. {footcite:p}`shimada2007progress`. The order of magnitude is recovered, but the residual remains to be explained.

## L-H threshold.

D0FUS implements three L-H threshold scalings (Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>`). The run benchmarked here uses the default one, the Martin scaling {footcite:p}`martin2008power`, and takes as reference the ITER predictions of the ITPA TC-26 campaign {footcite:p}`delabie2026metalwall`. D0FUS sits about $5\,\%$ above the reference at both densities, a small and systematic offset next to the intrinsic scatter of the threshold scaling itself.

:::{table} L-H power threshold of D0FUS, Martin scaling, against the ITER predictions of the ITPA TC-26 campaign.
:name: tab:chap2_lh_check
:align: center

| Quantity                                                            | D0FUS | Reference | Source       |
|:--------------------------------------------------------------------|:------|:----------|:-------------|
| $P_{\text{L-H}}$, ITER, $\bar n = 0.5\times10^{20}$ m$^{-3}$ \[MW\] | 54.8  | 52.3      | {footcite:p}`delabie2026metalwall` |
| $P_{\text{L-H}}$, ITER, $\bar n = 1.0\times10^{20}$ m$^{-3}$ \[MW\] | 90.1  | 86.0      | {footcite:p}`delabie2026metalwall` |
:::

## Scrape-off layer and detachment.

The heat-flux width implements regression \#15 of Eich *et al.* {footcite:p}`eich2013scaling` presented in Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>`, tested on the ITER evaluation provided by the same paper {footcite:p}`eich2013scaling`: D0FUS returns $0.70$ mm against the published $0.73$ mm, a $4\,\%$ agreement that validates the implementation of the regression.

## Safety factor.

The two $q_{95}$ closed forms (Eqs. {eq}`eq:q95_sauter` and {eq}`eq:q95_iter89`) are checked against published equilibrium values on ITER and SPARC ({numref}`Table %s <tab:chap2_q95_check>`). ITER-89 reproduces both within $7\,\%$. Sauter overshoots both, by $15$ to $20\,\%$. ITER-89 is therefore kept as the D0FUS default.

:::{table} The two $q_{95}$ closed forms of D0FUS against published equilibrium values, evaluated at the published field, current and shaping of each machine. References: ITER {footcite:p}`shimada2007progress`, SPARC {footcite:p}`rodriguez2020predictions,creely2020overview`.
:name: tab:chap2_q95_check
:align: center

| Machine | Published $q_{95}$ | ITER-89 (Eq. {eq}`eq:q95_iter89`) | Sauter (Eq. {eq}`eq:q95_sauter`) |
|:--------|:-------------------|:-----------------------|:----------------------|
| ITER    | $3.0$              | $3.00$                 | $3.44$                |
| SPARC   | $3.4$              | $3.62$                 | $4.14$                |
:::

```{rubric} References
```

```{footbibliography}
```
