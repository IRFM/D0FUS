(app:profile_models)=

# Radial profile models

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section gives the functional forms of the profile parameterisation of Section {ref}`Radial profiles <ssec:chap1_profiles>`, the inversion fixing the core peak, the preset values, and the conversion between line-averaged and volume-averaged densities.

## Profile model

All radial quantities (density $n$, temperature $T$) are built on the same parameterisation, controlled by five parameters: the volume-averaged value $\bar{X}$, a core peaking exponent $\nu$, a normalised pedestal radius $\rho_\mathrm{ped}$, a pedestal fraction $f_\mathrm{ped} = X_\mathrm{ped}/\bar{X}$, and the geometry weight inherited from Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`. The model is inspired by the parabolic profile ansatz of Lackner {footcite:p}`lackner1990comments` and the pedestal parameterisation used in the ITER Physics Basis {footcite:p}`iterphysicsbasis1999`. Four presets (L-Mode, H-mode, Advanced confinement mode and an EU-DEMO 2017 profile set) are exposed with H-mode like profile as default. The Advanced preset ($\nu_n = 1.5$, $\nu_T = 2.00$, $\rho_\mathrm{ped} = 0.96$) mimics improved-confinement scenarios, with markedly more peaked profiles and a deeper pedestal than the standard H preset.

Two parameterisations are implemented. When $\rho_\mathrm{ped} = 1$ and $f_\mathrm{ped} = 0$, the profile reduces to the purely parabolic L-mode form,

$$
X(\rho) = \bar{X}\,(1 + \nu)\,(1 - \rho^2)^\nu
$$ (eq:profile_parabolic)

where the factor $(1 + \nu)$ normalises the volume average to $\bar{X}$ under the cylindrical weight $w(\rho) = 2\rho$. The on-axis value $X_0/\bar{X} = 1 + \nu$ provides a closed-form check on the numerical inversion used in the general case.

The H-mode profile is built as the product of a core parabola, defined on $[0, \rho_\mathrm{ped}]$ and continued by zero beyond,

$$
X_\mathrm{core}(\rho) = X_\mathrm{ped} + (X_0 - X_\mathrm{ped})\left(1 - \frac{\rho^2}{\rho_\mathrm{ped}^2}\right)^\nu
$$ (eq:profile_core)

a tanh envelope smoothes the transition from the pedestal top to the separatrix:

$$
h(\rho) = \frac{1}{2}\left(1 + \tanh\frac{\rho_\mathrm{mid} - \rho}{w}\right)
$$ (eq:tanh_envelope)

with $\rho_\mathrm{mid} = (1 + \rho_\mathrm{ped})/2$ and $w = (1 - \rho_\mathrm{ped})/4$. The full profile is the product

$$
X(\rho) = X_\mathrm{core}(\rho)\, h(\rho)
$$ (eq:profile_full)

The multiplicative composition preserves the core peaking (the envelope $h$ stays essentially equal to one well inside $\rho_\mathrm{ped}$, leaving the core parabola unchanged there), restores a continuous gradient at $\rho_\mathrm{ped}$, and avoids the unphysical bump that an additive coupling would produce around the pedestal top. The functional form is a symmetric counterpart to the experimental mtanh fit of Groebner {footcite:p}`groebner1998hmode`, without the scrape-off-layer (SOL) ramp that a 0D code cannot model.

## Core peak from the volume-average constraint

The on-axis value $X_0$ is not a free parameter: it is fixed by the requirement that the volume average of the profile equals the prescribed mean,

$$
\langle X \rangle_\mathrm{vol} = \int_0^1 X(\rho)\, w(\rho)\, \mathrm{d}\rho = \bar{X}
$$ (eq:volume_average_constraint)

where the volume weight $w(\rho)$ is the only place the geometry enters (the radial shape of $X(\rho)$ itself is identical in both modes):

$$
w(\rho) = \begin{cases}
		2\rho & \text{Academic (cylindrical torus)}, \\[4pt]
		\displaystyle\frac{V'(\rho)}{V} & \text{Refined (Miller Jacobian)}
	\end{cases}
$$ (eq:weight_function)

Since $X(\rho)$ is linear in $X_0$, the constraint inverts analytically:

$$
\frac{X_0}{\bar{X}} = f_\mathrm{ped} + \frac{1 - f_\mathrm{ped}\, I_h}{I_{gh}}
$$ (eq:X0_inversion)

with $g(\rho) = \bigl(1 - (\rho/\rho_\mathrm{ped})^2\bigr)^\nu$, $I_h = \int_0^1 h\, w\, \mathrm{d}\rho$ and $I_{gh} = \int_0^1 g\, h\, w\, \mathrm{d}\rho$ evaluated numerically. In the purely parabolic limit ($\rho_\mathrm{ped} = 1$, $f_\mathrm{ped} = 0$, cylindrical weight), the result collapses to the closed form $X_0/\bar{X} = 1 + \nu$ already mentioned, which D0FUS returns directly without entering the numerical branch.

Switching from Academic to Refined geometry raises the on-axis peak $X_0/\bar{X}$ by $4$ to $8\,\%$ on the three presets. This shift is not a fixed geometric factor: it grows with the profile peaking (hence with $\nu$ and the pedestal depth), which is why it differs from one preset to the next, with the detailed values reported in {numref}`Table %s <tab:profile_X0_shift>`.

:::{table} Relative increase of the on-axis peak $X_0/\bar{X}$ when switching from Academic to Refined geometry, for the three preset configurations at ITER shaping ($R_0 = 6.2$ m, $a = 2.0$ m, $\kappa = 1.85$, $\delta = 0.50$). Both density and temperature profiles are shown.
:name: tab:profile_X0_shift
:align: center

| Mode     | Quantity | $X_0/\bar{X}$ Academic | $X_0/\bar{X}$ Refined | Increase |
|:---------|:---------|:-----------------------|:----------------------|:---------|
| L        | $n$      | 1.500                  | 1.574                 | +5.0 %   |
| L        | $T$      | 2.750                  | 2.932                 | +6.6 %   |
| H        | $n$      | 1.056                  | 1.102                 | +4.4 %   |
| H        | $T$      | 2.559                  | 2.762                 | +7.9 %   |
| Advanced | $n$      | 1.353                  | 1.457                 | +7.7 %   |
| Advanced | $T$      | 2.035                  | 2.186                 | +7.4 %   |
:::

## Profile presets

The L-mode preset uses purely parabolic profiles with no pedestal ($\rho_\mathrm{ped} = 1$). The values $\nu_n = 0.50$ and $\nu_T = 1.00$ reproduce the standard reactor profile ansatz of Freidberg {footcite:p}`freidberg2015designing`, yielding $n_0/\bar{n} = 1.50$ and $T_0/\bar{T} = 2.00$.

The H-mode preset is the default in D0FUS, featuring a nearly flat core density ($\nu_n = 0.01$, $n_\mathrm{ped}/\bar{n} = 0.99$) combined with a strongly peaked core temperature ($\nu_T = 2.80$, $T_\mathrm{ped}/\bar{T} = 0.55$) at a pedestal radius $\rho_\mathrm{ped} = 0.95$. These values reproduce the H-mode profiles of the CORSICA simulations of the 15 MA ITER baseline scenario by Kim {footcite:p}`kim2018iter` ($n_{e0}/\langle n_e\rangle \approx 1.04$, $\tanh$ pedestal near $\rho_\mathrm{tor} \approx 0.94$). The flat-density assumption may be pessimistic, since Angioni {footcite:p}`angioni2007jetpeaking` reports H-mode peaking factors rising up to $\sim 1.5$ in ITER-like conditions, which would translate into a higher fusion power at fixed $\bar{n}$ and $\beta$.

## Line-averaged versus volume-averaged density

The volume-averaged density $\bar{n}$ carried by the solver is not the quantity that experiments report. Interferometers measure the line-averaged density $\bar{n}_\mathrm{line}$ along a horizontal midplane chord, and three of the most consequential empirical relations encountered later (the Greenwald density limit, most confinement scalings, and the Martin L-H threshold) all consume the line average. D0FUS therefore performs the conversion explicitly.

The midplane chord at $Z = 0$ crosses every flux surface, weighting each one by the radial spacing $\mathrm{d}R$ between two neighbouring surfaces along that chord. In the $\delta = 0$ limit this spacing is uniform ($\mathrm{d}R = a\, \mathrm{d}\rho$), so the chord average reduces to a flat radial integral, in contrast with the volume average where each surface is weighted by the volume it encloses:

$$
\bar{n}_\mathrm{line} = \int_0^1 n(\rho)\, \mathrm{d}\rho, \qquad
	\bar{n}_\mathrm{vol}  = \int_0^1 n(\rho)\, w(\rho)\, \mathrm{d}\rho, \qquad w(\rho) = 2\rho \;\;\text{(Academic)}
$$ (eq:nline_nvol)

The volume weight (Eq. {eq}`eq:weight_function`, reducing to $2\rho$ in Academic mode) vanishes on axis: the central flux surfaces enclose almost no volume, so the volume average sees the dense core much less than the chord does. The line average therefore exceeds the volume average for any peaked profile, and at ITER shaping the ratio $\bar{n}_\mathrm{line}/\bar{n}_\mathrm{vol}$ is $1.18$ for the L preset and $1.03$ for the H preset.

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
