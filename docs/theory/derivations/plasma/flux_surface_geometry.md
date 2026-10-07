(app:geometry_models)=

# Flux-surface geometry models

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section details the geometry machinery summarised in Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`: the Miller parameterisation of the interior flux surfaces, the construction of the radial shaping profiles, and the volume and surface integrals derived from them.

## Miller flux-surface parameterisation

Once the edge shape is fixed, one needs a way to describe every interior flux surface. The simplest approach, used in Academic mode, is to assume that all surfaces are concentric ellipses sharing the same elongation $\kappa$ and no triangularity. Refined mode follows a more physically grounded path: it builds each flux surface from the Miller parameterisation {footcite:p}`miller1998noncircular`.

For a surface labelled by the normalised radial coordinate $\rho = r/a \in [0,1]$ and the poloidal angle $\theta \in [0, 2\pi)$, the coordinates in the $(R, Z)$ plane are

$$
\begin{aligned}
R(\rho, \theta) &= R_0 + \rho\, a \cos\!\bigl[\theta + \arcsin\!\bigl(\delta(\rho)\bigr) \sin\theta\bigr]
	\\
	Z(\rho, \theta) &= \kappa(\rho)\, \rho\, a \sin\theta
\end{aligned}
$$ (eq:miller_R)

where $\kappa(\rho)$ and $\delta(\rho)$ are radial profiles described below. The Shafranov shift (the outward radial displacement of the centres of the nested flux surfaces with increasing plasma pressure) is neglected: at percent-level accuracy {footcite:p}`lao2005equilibrium` it sits well within the 0D uncertainty budget, and its self-consistent treatment would require a 1.5D Grad-Shafranov solver as in METIS {footcite:p}`artaud2018metis`, ASTRA {footcite:p}`pereverzev2002astra` or JINTRAC {footcite:p}`romanelli2014jintrac`.

## Radial shaping profiles

The Miller parameterisation (Eqs. {eq}`eq:miller_R`-{eq}`eq:miller_R`) requires $\kappa(\rho)$ and $\delta(\rho)$ as inputs. Two analytical results constrain the on-axis behaviour. Greene and Johnson {footcite:p}`greene1961determination` showed by Taylor expansion of the poloidal flux that flux surfaces close to the axis are necessarily elliptical: only the $m=2$ harmonic (elongation) survives, while triangularity and all higher-order harmonics vanish on axis. Ball and Parra {footcite:p}`ball2015intuition` later clarified the radial dependence: elongation alone can be preserved unchanged from the boundary to the core, whereas triangularity decreases monotonically from the LCFS down to $\delta(0) = 0$.

These results fix the asymptotic behaviour at the axis but leave the full radial profile underdetermined. A consistent prescription would couple $\kappa(\rho)$ and $\delta(\rho)$ to the toroidal current profile through Grad-Shafranov, which is the role of equilibrium solvers like CHEASE {footcite:p}`lutjens1996chease` but is out of scope for a 0D code. D0FUS instead interpolates between three control points fixed by the on-axis constraint, the $95\%$-flux surface (Eq. {eq}`eq:shaping_95`), and the prescribed LCFS shape.

The interpolation is carried out by the PCHIP scheme of Fritsch and Carlson {footcite:p}`fritsch1980monotone`, allowing $C^1$ derivability and preserving monotonicity between control points. The elongation profile takes nodes $(0, \kappa_{95})$, $(\rho_{95}, \kappa_{95})$, and $(1, \kappa_\mathrm{edge})$, which gives a strictly flat core $\kappa(\rho) = \kappa_{95}$ for $\rho \leq \rho_{95}$ and a smooth monotone rise to $\kappa_\mathrm{edge}$ over the edge layer. The triangularity profile uses nodes $(0, 0)$, $(\rho_{95}, \delta_{95})$, and $(1, \delta_\mathrm{edge})$, enforcing $\delta(0) = 0$ exactly. Negative-triangularity configurations ($\delta_\mathrm{edge} < 0$) are handled identically and produce a monotonically decreasing profile.

:::{figure} /figures/thesis/d0fus_shaping_profiles.png
:name: fig:chap1_shaping_profiles
:width: 100%
:align: center

Radial elongation $\kappa(\rho)$ and triangularity $\delta(\rho)$ profiles in the two D0FUS modes. Left: Academic (constant $\kappa$, $\delta = 0$). Right: Refined PCHIP interpolation (Fritsch & Carlson 1980) of $\kappa(\rho)$ and $\delta(\rho)$ between the magnetic axis and the LCFS.
:::

## Volume element and plasma volume

The volume element $V'(\rho) = \mathrm{d}V/\mathrm{d}\rho$ is the central geometric quantity used by the radial integrals scattered throughout the physics chain of {ref}`Models <chap:d0fus>`. Its computation differs between the two geometry modes.

In Academic mode, the plasma is treated as an elliptical torus with constant $\kappa$ and no triangularity. The volume element is analytical,

$$
V'(\rho) = 4\pi^2 R_0\, a^2\, \kappa\, \rho
$$ (eq:Vprime_acad)

and the total plasma volume reduces to the well-known textbook expression, hereafter the Wesson formula {footcite:p}`wesson2011tokamaks`,

$$
V_\mathrm{Wesson} = 2\pi^2 R_0\, \kappa\, a^2
$$ (eq:V_acad)

In Refined mode, the volume element is computed numerically from the Jacobian of the Miller coordinate transformation. Defining $J_\mathrm{2D}(\rho,\theta) = \partial_\rho R\,\partial_\theta Z - \partial_\theta R\,\partial_\rho Z$, the toroidal-revolution volume element reads

$$
V'(\rho) = 2\pi \int_0^{2\pi} R(\rho, \theta)\, |J_\mathrm{2D}(\rho,\theta)|\, \mathrm{d}\theta
$$ (eq:Vprime_miller)

The integral is evaluated on an $(N_\rho \times N_\theta)$ grid (production grid $500 \times 200$), and the total volume follows as $V = \int_0^1 V'(\rho)\, \mathrm{d}\rho$.

The poloidal arc length $L_\theta(\rho)$ is obtained from the same partial derivatives that fed the Jacobian,

$$
L_\theta(\rho) = \int_0^{2\pi} \sqrt{(\partial_\theta R)^2 + (\partial_\theta Z)^2}\, \mathrm{d}\theta
$$ (eq:Lp_miller)

In Academic mode, $L_\theta(\rho)$ is computed analytically as the Ramanujan perimeter[^1] of the elliptical flux surface with semi-axes $\rho a$ and $\kappa \rho a$,

$$
L_\theta(\rho) = \pi \rho a \left[ 3(1 + \kappa) - \sqrt{(3 + \kappa)(1 + 3\kappa)} \right]
$$ (eq:ramanujan_perimeter)

D0FUS also exposes an analytical formula for the plasma volume that retains the leading-order triangularity correction,

$$
V_\mathrm{O(\delta^2)} = 2\pi^2 R_0\, \kappa\, a^2 \left(1 - \frac{a \delta}{4 R_0} - \frac{\delta^2}{8}\right)
$$ (eq:V_delta2)

derived in {ref}`Derivation of the O(δ²) plasma volume <app:volume_delta2>` from the Miller Jacobian by expanding to second order in $\delta$. This expression provides a default whenever the precomputed Miller grid is not available, and it serves as a useful sanity check against the numerical integration. For the ITER baseline, the three formulas give

$$
\begin{aligned}
V_\mathrm{Wesson} &= 905.6\,\mathrm{m^3} \\
	V_\mathrm{O(\delta^2)} &= 840.8\,\mathrm{m^3} \\
	V_\mathrm{Miller} &= 839.9\,\mathrm{m^3}
\end{aligned}
$$

so the simple Wesson formula overestimates the true Miller volume by approximately $7.8\%$, while the analytical $\mathrm{O}(\delta^2)$ correction recovers the numerical result to within $0.1\%$. The triangularity term in Eq. {eq}`eq:V_delta2` is therefore not a cosmetic refinement: at ITER shaping it captures essentially the full deviation from the cylindrical-torus approximation.

:::{figure} /figures/thesis/d0fus_volume_comparison.png
:name: fig:chap1_volume_comparison
:width: 100%
:align: center

Plasma volume $V$ at $R_0 = 6.2$ m, $a = 2.0$ m as a function of the shaping parameters, comparing the analytical Wesson formula (Eq. {eq}`eq:V_acad`), the $\mathcal{O}(\delta^2)$ correction (Eq. {eq}`eq:V_delta2`), and the numerical Miller integration (Eq. {eq}`eq:Vprime_miller`). ITER and EU-DEMO design points are highlighted.
:::

## First-wall surface area

D0FUS assumes that the first wall is conformal to the LCFS poloidal contour.

In Academic mode, the LCFS poloidal perimeter is the Ramanujan formula of Eq. {eq}`eq:ramanujan_perimeter` evaluated at the LCFS, $L_\theta(1)$, and the surface area follows from the toroidal revolution, $S_\mathrm{FW} = 2\pi R_0\, L_\theta(1)$.

In Refined mode, the LCFS Miller contour at $\rho = 1$ is revolved toroidally,

$$
S_\mathrm{FW} = 2\pi \int_0^{2\pi} R(1, \theta)\, \sqrt{(\partial_\theta R)^2 + (\partial_\theta Z)^2}\, \mathrm{d}\theta
$$ (eq:SFW_miller)

with the contour evaluated using $\delta_\mathrm{edge}$ and $\kappa_\mathrm{edge}$ directly (no profile interpolation needed since one is already at the LCFS). For the ITER baseline, the two estimates give

$$
\begin{aligned}
S_\mathrm{FW}^\mathrm{Academic} &= 713.2\,\mathrm{m^2} \\
	S_\mathrm{FW}^\mathrm{Refined}  &= 681.5\,\mathrm{m^2}
\end{aligned}
$$

so the Refined estimate is approximately $4.4\%$ *smaller* than the Academic one. This is consistent with the volume reduction discussed above: positive triangularity slightly contracts the LCFS contour relative to the equivalent ellipse with semi-axes $(a, \kappa a)$, and the smaller poloidal perimeter feeds directly into a smaller toroidal surface. For negative triangularity, the inequality reverses and the Refined surface is slightly larger than the Academic one.

:::{figure} /figures/thesis/d0fus_first_wall_surface.png
:name: fig:chap1_first_wall_surface
:width: 100%
:align: center

First-wall surface area $S_\mathrm{FW}$ as a function of triangularity $\delta$ at fixed $\kappa$, comparing the Academic Ramanujan ellipse (Eq. {eq}`eq:ramanujan_perimeter`) and the Refined Miller LCFS integration (Eq. {eq}`eq:SFW_miller`). The Refined estimate is smaller than the Academic one for $\delta > 0$ and larger for $\delta < 0$.
:::

## Summary of geometric outputs

Six quantities are then available to every downstream physics module: the plasma volume $V$, the volume element $V'(\rho)$, the poloidal cross-section area derivative $\mathrm{d}A_\mathrm{pol}/\mathrm{d}\rho$, the poloidal arc length $L_\theta(\rho)$, the perimeter-averaged $\langle 1/R^2\rangle(\rho)$, and the first-wall surface area $S_\mathrm{FW}$. In Academic mode, all six are closed-form expressions in $(R_0, a, \kappa)$. In Refined mode, all six come from a single Miller Jacobian precomputation, run once per design point on a $500 \times 200$ grid in $(\rho, \theta)$. The computational overhead, of the order of a few tens of milliseconds, is negligible compared to the cost of the self-consistent solver itself, and it earns the code an honest treatment of shaping that propagates cleanly into every integral encountered in the physics chain.

[^1]: The first of two approximations that Ramanujan published in his 1914 paper on modular equations and approximations to $\pi$ {footcite:p}`ramanujan1914modular`. He famously claimed that many of his formulas had been revealed to him in dreams by the family deity Namagiri, and would simply transcribe them upon waking: whether this one was among them is impossible to know. Whatever the route, its relative error stays below $0.04\,\%$ for any eccentricity, which makes it more accurate than any closed-form approximation derived in the following century.

```{rubric} References
```

```{footbibliography}
```
