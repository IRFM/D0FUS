(app:profile_models)=

# Radial profile models

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.2, which describes D0FUS v2.7.0, and updated to the code of October 2026. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section gives the functional forms of the profile parameterisation of Section {ref}`Radial profiles <ssec:chap1_profiles>`, the inversion fixing the core peak, the preset values, and the conversion between line-averaged and volume-averaged densities.

## Profile model

All radial quantities (density $n$, temperature $T$) are built on the same parameterisation, controlled by five parameters: the volume-averaged value $\bar{X}$, a core peaking exponent $\nu$, a normalised pedestal radius $\rho_\mathrm{ped}$, a pedestal fraction $f_\mathrm{ped} = X_\mathrm{ped}/\bar{X}$, and the radial coordinate, the volume label $\rho_V = \sqrt{V(\rho)/V}$ of Eq. {eq}`eq:rho_V`. The profiles below are written in $\rho_V$. The deck gives $\rho_\mathrm{ped}$ in the toroidal-flux label $\rho_\mathrm{tor}$, converted to $\rho_V$ before use (Section {ref}`Radial profiles <ssec:chap1_profiles>`). The model is inspired by the parabolic profile ansatz of Lackner {footcite:p}`lackner1990comments` and the pedestal parameterisation used in the ITER Physics Basis {footcite:p}`iterphysicsbasis1999`. Four presets (L-Mode, H-mode, Advanced confinement mode and an EU-DEMO 2017 profile set) are exposed with H-mode like profile as default. The Advanced preset ($\nu_n = 1.5$, $\nu_T = 2.00$, $\rho_\mathrm{ped} = 0.96$) mimics improved-confinement scenarios, with markedly more peaked profiles and a deeper pedestal than the standard H preset.

Two parameterisations are implemented. When $\rho_\mathrm{ped} = 1$ and $f_\mathrm{ped} = 0$, the profile reduces to the purely parabolic L-mode form,

$$
X(\rho_V) = \bar{X}\,(1 + \nu)\,(1 - \rho_V^2)^\nu
$$ (eq:profile_parabolic)

where the factor $(1 + \nu)$ normalises the volume average to $\bar{X}$ under the volume element $2\rho_V\, \mathrm{d}\rho_V$, in both geometries. The on-axis value $X_0/\bar{X} = 1 + \nu$ provides a closed-form check on the numerical inversion used in the general case.

The H-mode profile is built as the product of a core parabola, defined on $[0, \rho_\mathrm{ped}]$ and continued by zero beyond,

$$
X_\mathrm{core}(\rho_V) = X_\mathrm{ped} + (X_0 - X_\mathrm{ped})\left(1 - \frac{\rho_V^2}{\rho_\mathrm{ped}^2}\right)^\nu
$$ (eq:profile_core)

a tanh envelope smoothes the transition from the pedestal top to the separatrix:

$$
h(\rho_V) = \frac{1}{2}\left(1 + \tanh\frac{\rho_\mathrm{mid} - \rho_V}{w}\right)
$$ (eq:tanh_envelope)

with $\rho_\mathrm{mid} = (1 + \rho_\mathrm{ped})/2$ and $w = (1 - \rho_\mathrm{ped})/4$. The full profile is the product

$$
X(\rho_V) = X_\mathrm{core}(\rho_V)\, h(\rho_V)
$$ (eq:profile_full)

The multiplicative composition preserves the core peaking (the envelope $h$ stays essentially equal to one well inside $\rho_\mathrm{ped}$, leaving the core parabola unchanged there), restores a continuous gradient at $\rho_\mathrm{ped}$, and avoids the unphysical bump that an additive coupling would produce around the pedestal top. The functional form is a symmetric counterpart to the experimental mtanh fit of Groebner {footcite:p}`groebner1998hmode`, without the scrape-off-layer (SOL) ramp that a 0D code cannot model.

## Core peak from the volume-average constraint

The on-axis value $X_0$ is not a free parameter: it is fixed by the requirement that the volume average of the profile equals the prescribed mean,

$$
\langle X \rangle_\mathrm{vol} = \int_0^1 X(\rho_V)\, 2\rho_V\, \mathrm{d}\rho_V = \bar{X}
$$ (eq:volume_average_constraint)

The volume element $2\rho_V\, \mathrm{d}\rho_V$ is exact in both geometries. On the label $\rho$ of the geometry, the same element reads

$$
w(\rho)\, \mathrm{d}\rho = \begin{cases}
		2\rho\, \mathrm{d}\rho & \text{Academic (cylindrical torus, } \rho_V = \rho\text{)}, \\[4pt]
		\displaystyle\frac{V'(\rho)}{V}\, \mathrm{d}\rho & \text{Refined (Miller Jacobian)}
	\end{cases}
$$ (eq:weight_function)

Since $X(\rho_V)$ is linear in $X_0$, the constraint inverts analytically:

$$
\frac{X_0}{\bar{X}} = f_\mathrm{ped} + \frac{1 - f_\mathrm{ped}\, I_h}{I_{gh}}
$$ (eq:X0_inversion)

with $g(\rho_V) = \bigl(1 - (\rho_V/\rho_\mathrm{ped})^2\bigr)^\nu$, $I_h = \int_0^1 h\, 2\rho_V\, \mathrm{d}\rho_V$ and $I_{gh} = \int_0^1 g\, h\, 2\rho_V\, \mathrm{d}\rho_V$ evaluated numerically. In the purely parabolic limit ($\rho_\mathrm{ped} = 1$, $f_\mathrm{ped} = 0$), the result collapses to the closed form $X_0/\bar{X} = 1 + \nu$ already mentioned, which D0FUS returns directly without entering the numerical branch.

The geometry enters Eq. {eq}`eq:X0_inversion` only through $\rho_\mathrm{ped}$, converted from $\rho_\mathrm{tor}$ to $\rho_V$. {numref}`Table %s <tab:profile_X0>` gives the on-axis peak of the four presets in Academic geometry, where $\rho_V = \rho_\mathrm{tor}$. At ITER shaping, $\rho_\mathrm{ped} = 0.95$ places the pedestal at $\rho_V = 0.959$, and the peaks of the pedestal presets fall by at most 2 % (H preset: $n_0/\bar{n} = 1.045$, $T_0/\bar{T} = 2.503$).

:::{table} On-axis peak $X_0/\bar{X}$ of the density and temperature profiles of the four presets, in Academic geometry.
:name: tab:profile_X0
:align: center

| Mode     | $n_0/\bar{n}$ | $T_0/\bar{T}$ |
|:---------|:--------------|:--------------|
| L        | 1.500         | 2.000         |
| H        | 1.056         | 2.559         |
| Advanced | 1.188         | 2.086         |
| EU-DEMO  | 1.353         | 2.035         |
:::

## Profile presets

The L-mode preset uses purely parabolic profiles with no pedestal ($\rho_\mathrm{ped} = 1$). The values $\nu_n = 0.50$ and $\nu_T = 1.00$ reproduce the standard reactor profile ansatz of Freidberg {footcite:p}`freidberg2015designing`, yielding $n_0/\bar{n} = 1.50$ and $T_0/\bar{T} = 2.00$.

The H-mode preset is the default in D0FUS, featuring a nearly flat core density ($\nu_n = 0.01$, $n_\mathrm{ped}/\bar{n} = 0.99$) combined with a strongly peaked core temperature ($\nu_T = 2.80$, $T_\mathrm{ped}/\bar{T} = 0.55$) at a pedestal radius $\rho_\mathrm{ped} = 0.95$. These values reproduce the H-mode profiles of the CORSICA simulations of the 15 MA ITER baseline scenario by Kim {footcite:p}`kim2018iter` ($n_{e0}/\langle n_e\rangle \approx 1.04$, $\tanh$ pedestal near $\rho_\mathrm{tor} \approx 0.94$). The flat-density assumption may be pessimistic, since Angioni {footcite:p}`angioni2007jetpeaking` reports H-mode peaking factors rising up to $\sim 1.5$ in ITER-like conditions, which would translate into a higher fusion power at fixed $\bar{n}$ and $\beta$.

## Line-averaged versus volume-averaged density

The volume-averaged density $\bar{n}$ carried by the solver is not the quantity that experiments report. Interferometers measure the line-averaged density $\bar{n}_\mathrm{line}$ along a horizontal midplane chord, and three of the most consequential empirical relations encountered later (the Greenwald density limit, most confinement scalings, and the L-H threshold scalings) all consume the line average. D0FUS therefore performs the conversion explicitly.

The midplane chord at $Z = 0$ crosses every flux surface, weighting each one by the radial spacing $\mathrm{d}R$ between two neighbouring surfaces along that chord. The Miller surfaces cross the midplane at $R = R_0 \pm \rho a$ for any $\kappa(\rho)$ and $\delta(\rho)$, so this spacing is uniform in the label $\rho$ ($\mathrm{d}R = a\, \mathrm{d}\rho$). The chord average therefore reduces to a flat integral over $\rho$, in contrast with the volume average where each surface is weighted by the volume it encloses:

$$
\bar{n}_\mathrm{line} = \int_0^1 n\bigl(\rho_V(\rho)\bigr)\, \mathrm{d}\rho, \qquad
	\bar{n}_\mathrm{vol}  = \int_0^1 n(\rho_V)\, 2\rho_V\, \mathrm{d}\rho_V
$$ (eq:nline_nvol)

The volume weight vanishes on axis: the central flux surfaces enclose almost no volume, so the volume average sees the dense core much less than the chord does. The line average therefore exceeds the volume average for any peaked profile. The ratio $\bar{n}_\mathrm{line}/\bar{n}_\mathrm{vol}$ is $1.18$ for the L preset and $1.03$ for the H preset in Academic geometry. At ITER shaping it rises to $1.20$ and $1.03$, the elongation increasing towards the edge putting more volume in the outer surfaces.

:::{figure} /figures/thesis/d0fus_density_line_vol.png
:name: fig:chap1_density_line_vol
:width: 65%
:align: center

Relative difference $(\bar{n}_\mathrm{line} - \bar{n}_\mathrm{vol})/\bar{n}_\mathrm{vol}$ as a function of the density peaking exponent $\nu_n$, in the Academic geometry and L-Mode profile.
:::

```{rubric} References
```

```{footbibliography}
```
