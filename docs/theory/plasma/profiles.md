(ssec:chap1_profiles)=

# Radial profiles

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The fusion power, the volume-averaged pressure, the radiated power, and the bootstrap current all require integrating products of $n(\rho)$ and $T(\rho)$ over the plasma volume. A self-consistent determination of these profiles would couple them to neoclassical and turbulent transport, which is the role of integrated solvers like ASTRA {footcite:p}`pereverzev2002astra` or JINTRAC {footcite:p}`romanelli2014jintrac` and is out of scope for a 0D system code. D0FUS instead parameterises $n(\rho)$ and $T(\rho)$ with a closed-form ansatz that captures the essential features of L-mode and H-mode operation[^1]: $\bar{n}$ and $\bar{T}$ remain the only free variables, while the shape parameters (peaking exponents, pedestal location and height) are user inputs.

All radial quantities are built on a single five-parameter parameterisation: the volume-averaged value $\bar{X}$, a core peaking exponent $\nu$, a normalised pedestal radius $\rho_\mathrm{ped}$, a pedestal fraction $f_\mathrm{ped} = X_\mathrm{ped}/\bar{X}$ (with $X_\mathrm{ped} = X(\rho_\mathrm{ped})$ the pedestal-top value), and the geometry weight inherited from Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`, the same purely geometric quantity for all profiles, namely the normalised volume element

$$
w(\rho) = \frac{V'(\rho)}{V}
$$ (eq:w_rho)

with which every volume average of these pages is taken, $\bar{X} = \int_0^1 X(\rho)\, w(\rho)\, \mathrm{d}\rho$, and which reduces to $2\rho$ in Academic mode. With no pedestal, the profile reduces to the classical parabolic ansatz

$$
X(\rho) = \bar{X}\,(1+\nu)\,(1-\rho^2)^\nu
$$ (eq:parabolic_ansatz)

used in several power plant studies {footcite:p}`lackner1990comments,freidberg2015designing`. With the addition of a pedestal, a core parabola is multiplied by a $\tanh$ envelope that ensures the continuity of the profile between the pedestal and the core region. In every case the on-axis value is not a free parameter: it is fixed through the volume average of the profile, taken with the mode-consistent geometry weight. The functional forms, the analytical inversion for the core peak, and the quantitative mode comparison are given in {ref}`Radial profile models <app:profile_models>`.

Three production presets are shown in {numref}`Fig. %s <fig:chap1_nT_profiles>`, with the corresponding parameters given in {numref}`Table %s <tab:profile_presets>`. The L-mode preset reproduces a standard parabolic profile as used in studies by Freidberg et al. {footcite:p}`freidberg2015designing`. The H-mode preset, the default in D0FUS, combines a nearly flat core density with a strongly peaked core temperature above a pedestal at $\rho_\mathrm{ped} = 0.95$, approximating the CORSICA H-mode profiles of the 15 MA ITER baseline scenario {footcite:p}`kim2018iter`. The flat-density assumption may prove pessimistic, since H-mode peaking factors up to $\sim 1.5$ are reported in ITER-like conditions {footcite:p}`angioni2007jetpeaking`. An Advanced preset (more peaked core profiles above a similar pedestal, {numref}`Table %s <tab:profile_presets>`) mimics scenarios with this type of profiles. A fully Manual mode completes the set.

:::{table} Profile presets in D0FUS. The “Manual” mode reads all five parameters from the input file.
:name: tab:profile_presets
:align: center

| Mode     | $\nu_n$        | $\nu_T$ | $\rho_\mathrm{ped}$ | $n_\mathrm{ped}/\bar{n}$ | $T_\mathrm{ped}/\bar{T}$ |
|:---------|:---------------|:--------|:--------------------|:-------------------------|:-------------------------|
| L        | 0.50           | 1.00    | 1.00                | 0                        | 0                        |
| H        | 0.01           | 2.80    | 0.95                | 0.99                     | 0.55                     |
| Advanced | 1.50           | 2.00    | 0.96                | 0.95                     | 0.55                     |
| Manual   | user-specified |         |                     |                          |                          |
:::

:::{figure} /figures/thesis/d0fus_nT_profiles.png
:name: fig:chap1_nT_profiles
:width: 100%
:align: center

Normalised radial profiles $\hat{n}(\rho)$, $\hat{T}(\rho)$ and $\hat{p}(\rho) = \hat{n}\,\hat{T}$ for the three profile presets of D0FUS, with the parameters of {numref}`Table %s <tab:profile_presets>`: L-mode (parabolic, no pedestal), H-mode (nearly flat density, strongly peaked temperature above a pedestal at $\rho_\mathrm{ped} = 0.95$) and Advanced (more peaked core profiles).
:::

One conversion deserves emphasis before moving on. The solver primarily calculates the volume-averaged density, but the three most consequential empirical relations of the chain (the Greenwald limit, most confinement time scalings, and most L-H threshold scalings) use the line-averaged density measured by interferometry along a midplane chord. Because the volume weight vanishes on axis while the chord samples all surfaces evenly, the line average exceeds the volume average for any peaked profile, by 18 % for the L preset and 3 % for the H preset at ITER shaping. D0FUS performs the conversion from volume-averaged to line-averaged density explicitly ({ref}`Radial profile models <app:profile_models>`).

{numref}`Table %s <tab:profile_usage>` summarises where the profiles enter the chain: every integrand of the solver combines $n(\rho)$ and/or $T(\rho)$ with the geometry weight of Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`, so the prescribed peaking directly shapes the fusion power, the radiation losses and the bootstrap current. The sensitivity of each integral to the peaking, and the poloidal maps obtained by projecting the profiles onto the flux surfaces, are shown in {ref}`Radial profile models <app:profile_models>`.

:::{table} Usage of radial profiles in D0FUS. Each integrand combines $n(\rho)$ and/or $T(\rho)$ with the appropriate volume weight.
:name: tab:profile_usage
:align: center

| Quantity                           | Integrand                       |
|:-----------------------------------|:--------------------------------|
| Fusion power ($\bar{n}_e$)         | $n^2 \langle\sigma v\rangle(T)$ |
| Pressure ($\bar{p}$)               | $n\,T$                          |
| Bremsstrahlung ($P_\mathrm{brem}$) | $n^2 T^{1/2}$                   |
| Line radiation ($P_\mathrm{line}$) | $n^2 L_z(T)$                    |
| Bootstrap current ($I_b$)          | $\nabla \ln n$, $\nabla \ln T$  |
:::

[^1]: In H-mode, a transport barrier forms near the plasma edge and sustains steep local gradients: the profiles drop from a high value, the pedestal, to the separatrix over a narrow layer. The presets below parameterise this pedestal by its normalised position $\rho_\mathrm{ped}$ and its height.

```{rubric} References
```

```{footbibliography}
```
