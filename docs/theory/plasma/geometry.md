(ssec:chap1_geometry)=

# Flux-surface geometry

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The flux-surface geometry comes first, since its integrals weight everything that follows.

The poloidal cross-section of a tokamak plasma is usually not circular: elongation and triangularity are deliberately introduced to improve MHD stability and energy confinement {footcite:p}`troyon1984mhd,iterphysicsbasis1999`. Ignoring them would bias every geometric quantity that follows: the volume $V$ sets the fusion power, the first-wall area $S_\mathrm{FW}$ sets the neutron wall load, the poloidal arc length $L_\theta(\rho)$ ties the poloidal field, the safety factor and the internal inductance to the enclosed current, and the volume element $V'(\rho)$ weights every radial integral of these pages, with $\rho \in [0,1]$ the normalised radial coordinate labelling the flux surfaces ($\rho = 1$ at the LCFS).

D0FUS supports two geometry levels of fidelity: an Academic mode that treats the plasma as an elliptical torus with constant elongation and no triangularity, and a Refined mode that uses nested Miller flux surfaces with prescribed radial profiles of $\kappa$ and $\delta$.

## Edge shaping: elongation and triangularity

The plasma shape is characterised at the last closed flux surface (LCFS, $\rho = 1$) by two dimensionless parameters. The elongation $\kappa$ is the ratio of the plasma vertical half-height to the horizontal minor radius $a$. The triangularity $\delta$ is defined as the horizontal shift of the highest point of the surface with respect to its centre, normalised to $a$. Both definitions, together with the major radius $R_0$ and the minor radius $a$, are illustrated in {numref}`Fig. %s <fig:chap1_shape_def>`.

:::{figure} /figures/thesis/kappa_delta_definition.png
:name: fig:chap1_shape_def
:width: 100%
:align: center

Definition of the edge shaping parameters on a plasma cross-section (drawn for $\kappa = 1.8$, $\delta = 0.45$, $A = R_0/a = 3$). The elongation $\kappa$ is the vertical half-height of the last closed flux surface normalised to the minor radius $a$. The triangularity $\delta$ is the inward shift of its highest point with respect to the geometric centre, normalised to $a$.
:::

Elongation is desirable in the first place: it enlarges the plasma volume and cross-section at fixed minor radius, and it improves the achievable confinement and stability limits, so designers generally push it as high as possible {footcite:p}`freidberg2015designing`. It is not free, however. A vertically elongated equilibrium is produced by a poloidal field whose field-line curvature is such that a small vertical displacement of the current-carrying column is amplified rather than restored: the plasma is then vertically unstable, on an axisymmetric mode whose growth rate increases with the elongation. The more the plasma is elongated, the faster it runs away vertically, until the displacement can no longer be caught by the feedback control system: the event, known as a vertical displacement event (VDE), then ends in a disruption {footcite:p}`hender2007mhd`. The elongation a machine can actually operate is therefore set not by the equilibrium alone, but by how well this instability can be counteracted.

Two mechanisms are combined for that purpose. Passive stabilisation relies on the conducting structures surrounding the plasma, the vacuum vessel and, where they exist, dedicated stabilising plates. A vertical motion of the plasma changes the magnetic flux through these structures and induces currents in them which, by Lenz’s law, oppose the change: the induced toroidal current flows opposite to the plasma current on the side the plasma moves toward, and two conductors carrying opposite currents repel each other, so the column is pushed back. This effect only lasts as long as the induced currents do, that is over the resistive time of the structure, but that is enough to slow the instability down from the Alfvén time scale to a time scale on which a feedback system can act {footcite:p}`freidberg2015tokamak`. Active stabilisation then takes over: coils driven in feedback on the measured vertical position carry the currents needed to hold the column in place. The closer and the better conducting the passive structure, the larger the elongation that can be controlled, which is why the achievable elongation is a property of the plasma and of its surroundings together, and why one of the closures below carries a stabilisation margin alongside the aspect ratio.

In D0FUS, $\kappa$ can be prescribed manually or computed from the aspect ratio $A = R_0/a$ via one of three closures, or a blend of them derived for D0FUS. Their origins differ (as detailed below), but none predicts the operational threshold from plasma parameters alone: that threshold also depends on the proximity and conductivity of the passive structures and on the strength of the feedback system.

The Stambaugh scaling is an exponential fit, given in Ref. {footcite:p}`stambaugh2011fusion`, to the maximum stable elongation computed by Stambaugh et al. {footcite:p}`stambaugh1992relation` from free-boundary equilibria with a conformal ideal wall. It reads

$$
\kappa = 0.95 \left( 2.4 + 65\, e^{-A/0.376} \right)
$$ (eq:kappa_stambaugh)

where the 0.95 factor is the operating fraction recommended in Ref. {footcite:p}`stambaugh2011fusion`: with adequate feedback control, a tokamak can operate within a few percent of the ideal limit {footcite:p}`stambaugh1992relation`.

The Freidberg scaling (named after the first author of Refs. {footcite:p}`freidberg2015tokamak,lee2015vertical`) is a fit performed during the development of D0FUS to the numerical maximum-elongation results of Lee et al. {footcite:p}`lee2015vertical`, specifically the reference case of their study ($\beta_p = 1$, wall radius $b/a = 1.1$, feedback parameter $\gamma\tau_w = 1.5$), obtained from ideal-MHD vertical-stability analysis with a conformal wall and a finite feedback capability. The fit reproduces the published reference-case curve to better than $2\,\%$ over $\varepsilon = 0.1$ to $0.8$. With the same 0.95 operating fraction as Eq. {eq}`eq:kappa_stambaugh`, it gives

$$
\kappa = 0.95 \left( 1.81\, A^{0.009} + 1.52\, A^{-1.63} \right)
$$ (eq:kappa_freidberg)

The Wenninger scaling {footcite:p}`wenninger2015advances`, whose explicit form is given by Coleman et al. {footcite:p}`coleman2025definition` (their Eq. (22), written for $\kappa_{95}$, whence the $1.12$ conversion factor of Eq. {eq}`eq:shaping_95`), used in EU-DEMO design studies, additionally depends on a vertical stability margin $m_s$ that quantifies the passive stabilisation provided by the surrounding conducting structures against vertical displacement events,

$$
\kappa = \frac{1.12}{7.37}\left[ 18.84 - 0.87\,A
	- \sqrt{4.84\,A^2 - 28.77\,A + 52.52 + 14.74\,m_s} \right]
$$ (eq:kappa_wenninger)

The default value $m_s = 0.3$ is conservative and consistent with EU-DEMO design practice.

All three scalings encode similar behavior: at low aspect ratio ($A \lesssim 3$), the plasma can tolerate high elongation because the stronger curvature provides a natural stabilising effect, whereas at high aspect ratio ($A \gtrsim 4$), elongation must be reduced to maintain vertical stability. But note that the Wenninger scaling, although arguably the most refined of the three, displays a pathological behaviour at very low aspect ratio (spherical tokamaks), where it predicts unexpectedly low elongations.

This is why a fourth option, `Blend`, the default in D0FUS, was added after confronting these closures with the built machines. The Wenninger fit, derived for conventional aspect ratios, turns over below $A \approx 2.2$ and would unphysically re-decrease the elongation toward the spherical limit, while the Freidberg closure is the only one that remains consistent with the elongations achieved by spherical tokamaks. The blended model therefore follows Freidberg toward low aspect ratio and Wenninger toward conventional and high aspect ratio, through a smooth logistic crossover centred on the Wenninger turning point,

$$
\kappa_\mathrm{Blend} = (1 - w)\,\kappa_\mathrm{Freidberg} + w\,\kappa_\mathrm{Wenninger},
	\qquad w(A) = \frac{1}{1 + e^{-(A - A_t)/\Delta A}},
$$ (eq:kappa_blend_expr)

with $A_t = 2.235$ (the turning point at $m_s = 0.3$) and $\Delta A = 0.35$. {numref}`Fig. %s <fig:chap1_kappa_blend>` compares the four options with representative edge elongations of reference machines across the aspect-ratio range, where the Wenninger turnover is directly visible. The Stambaugh fit is not used in the blend, being markedly more optimistic than the other two over the whole range.

It must be stressed that these closures only provide an approximate estimate of the maximum allowed elongation, in the author’s view with a substantial uncertainty. It is no substitute for dedicated vertical-stability studies once a design point is chosen.

:::{figure} /figures/thesis/kappa_blend.png
:name: fig:chap1_kappa_blend
:width: 100%
:align: center

Elongation models $\kappa(A)$ implemented in D0FUS (Stambaugh, Freidberg, Wenninger and their blend) against representative edge elongations of reference tokamaks. The Blend option follows the Freidberg fit toward the spherical limit and the Wenninger fit at conventional and high aspect ratio, joined by a logistic crossover at the Wenninger turning point, below which the latter fit ceases to be physical.
:::

The triangularity is estimated from the elongation by the linear relation

$$
\delta = 0.6\,(\kappa - 1)
$$ (eq:delta_from_kappa)

introduced in the TREND system-code framework {footcite:p}`hartmann2013development` and consistent with the trend[^1] observed across existing tokamak designs, as illustrated in {numref}`Fig. %s <fig:chap1_delta_trend>`. Unlike the elongation, which is always pushed as high as the vertical stability allows, the triangularity is in practice a freer design choice. Eq. {eq}`eq:delta_from_kappa` is the simple empirical prescription adopted by TREND to encode the tendency of $\kappa$ and $\delta$ to increase together, and D0FUS retains it for lack of a better closure. In both {numref}`Fig. %s <fig:chap1_kappa_blend>` and {numref}`Fig. %s <fig:chap1_delta_trend>`, the machine points are single operational values found for reference discharges or design documents. In practice, every machine operates over a whole spectrum of elongations and triangularities, so these markers should be read as representative anchors rather than fixed characteristics.

:::{figure} /figures/thesis/delta_trend.png
:name: fig:chap1_delta_trend
:width: 100%
:align: center

Triangularity estimate $\delta = 0.6\,(\kappa - 1)$, composed with the four elongation models of {numref}`Fig. %s <fig:chap1_kappa_blend>` (same colour code), against the boundary triangularity of reference machines. The $95\,\%$ values, smaller by roughly a factor $1.5$, are deliberately not shown. No published value was found for ARC, NSTX-U and EU-DEMO, which are therefore absent. The $\delta = 0.6\,(\kappa - 1)$ estimate follows the general trend of the machines at conventional aspect ratio, with a substantial spread, and over-predicts toward the spherical limit.
:::

The values at the $\psi_N = 0.95$ flux surface are the ones conventionally used in the community, in particular because they are less sensitive to the presence of the X-point than the LCFS values. They follow the original ITER Physics Design Guidelines {footcite:p}`uckan1990iter`,

$$
\kappa_{95} = \kappa / 1.12, \qquad \delta_{95} = \delta / 1.5
$$ (eq:shaping_95)

## Flux surfaces and geometry

Once the edge shape is fixed, every interior flux surface must be described, since the volume element $V'(\rho) = \mathrm{d}V/\mathrm{d}\rho$ weights all the radial integrals of the chain. In Academic mode, all surfaces are concentric ellipses sharing the same elongation and no triangularity: the volume element is analytical,

$$
V'(\rho) = 4\pi^2 R_0 a^2 \kappa \rho \, ,
$$ (eq:Vprime_ellipse)

the total volume reduces to the textbook expression $V = 2\pi^2 R_0 \kappa a^2$ {footcite:p}`wesson2011tokamaks`, and the poloidal perimeter follows from the Ramanujan approximation for the ellipse (Eq. {eq}`eq:ramanujan_perimeter`, {ref}`Flux-surface geometry models <app:geometry_models>`)[^2]. In Refined mode, each surface is built from the Miller parameterisation {footcite:p}`miller1998noncircular`, with the shaping profiles $\kappa(\rho)$ and $\delta(\rho)$ anchored at three points and smoothly interpolated between them: the magnetic axis, where flux surfaces are elliptical with vanishing triangularity {footcite:p}`greene1961determination,ball2015intuition`, the 95 %-flux surface, and the prescribed LCFS shape. All geometric quantities then follow from a single numerical Jacobian precomputation, at the negligible cost of a few tens of milliseconds per design point. The Shafranov shift is neglected, a percent-level approximation on the geometric quantities, consistent with the module-level geometry validation of {ref}`Validation and benchmarks <chap:validation>`, whose self-consistent treatment may require a Grad-Shafranov solver and is left for future developments. The complete parameterisation, the shaping-profile construction and the volume and surface integrals are given in {ref}`Flux-surface geometry models <app:geometry_models>`.

Ignoring the triangularity, as done in the Academic mode, is not a cosmetic simplification. For the ITER baseline, the elliptical-torus formula overestimates the Miller plasma volume by about $7.8\,\%$, an error that propagates directly into the density required to hold a prescribed fusion power. An analytical correction derived for D0FUS and retaining the leading-order triangularity terms (Eq. {eq}`eq:V_delta2`, derived in {ref}`Derivation of the O(δ²) plasma volume <app:volume_delta2>`) recovers the numerical result to within $0.1\,\%$. The first-wall surface, which impacts the neutron wall load, moves the other way: positive triangularity contracts the LCFS contour, and the Refined estimate is about $4.4\,\%$ smaller than the elliptical one, the inequality reversing at negative triangularity. {numref}`Fig. %s <fig:chap1_miller_surfaces>` illustrates the resulting geometry in the two modes.

In both modes, six quantities are then made available to every downstream physics module: the plasma volume $V$, the volume element $V'(\rho)$, the poloidal cross-section area derivative, the poloidal arc length $L_\theta(\rho)$, the perimeter-averaged $\langle 1/R^2\rangle(\rho)$, and the first-wall surface area $S_\mathrm{FW}$.

:::{figure} /figures/thesis/d0fus_miller_surfaces.png
:name: fig:chap1_miller_surfaces
:width: 100%
:align: center

Miller flux surfaces for three geometry configurations: (a) Refined Miller with positive triangularity ($\delta > 0$, conventional D-shape), (b) Academic ellipse with constant $\kappa$ and $\delta = 0$, (c) Refined Miller with negative triangularity ($\delta < 0$).
:::

[^1]: Pun intended.

[^2]: Ramanujan famously attributed many of his formulas to visions received in dreams from the goddess Namagiri. Whether this particular approximation arrived that way is not documented, but the author likes to think so.

```{rubric} References
```

```{footbibliography}
```
