(ssec:chap1_stability)=

# Stability limits

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The three limits are stated in the order in which the chain assembles their ingredients: the density, then the pressure, then the current.

## Greenwald limit

The plasma density in a tokamak is subject to an empirical upper limit, the Greenwald density {footcite:p}`greenwald1988density,greenwald2002density`,

$$
n_G = \frac{I_p}{\pi a^2} \quad [10^{20}\,\mathrm{m}^{-3}]
$$ (eq:greenwald)

where $I_p$ is the plasma current in MA and $a$ the minor radius in metres. Its physical origin is, to the author’s knowledge, still debated, but converging evidence points to an edge phenomenon {footcite:p}`greenwald2002density,giacomin2022density`: the limit constrains the edge density rather than the core one. This reading also explains the routes around the limit, since any technique that decouples the core density from the edge allows stable operation above the nominal value. ASDEX Upgrade sustains H-modes at line-averaged densities up to $1.5\,n_G$ with pellet fuelling that feeds the core directly {footcite:p}`lang2012highdensity`, DIII-D reaches Greenwald fractions close to 2 in negative triangularity discharges {footcite:p}`sauter2025greenwald`, and EAST recently reported stable operation at $(1.3$ to $1.65)\,n_G$ with ECRH-assisted start-up {footcite:p}`liu2026east`. Without such decoupling, crossing the limit is empirically associated with edge radiative collapse and a sharply rising disruptivity.

The viability check enforces $f_\mathrm{GW} = \bar{n}_\mathrm{line}/n_G < f_\mathrm{GW,limit}$, with a default threshold $f_\mathrm{GW,limit} = 1.0$. The check thus makes use of two products of the chain: the line-averaged density, which the fusion power will demand (Sections {ref}`Radial profiles <ssec:chap1_profiles>` and {ref}`Fusion power, density and pressure <ssec:chap1_fusion>`), and the plasma current. Beyond the Greenwald form, D0FUS also implements two physics-based density-limit models as alternative checks, both converted to an equivalent cap on the line-averaged density: the edge-turbulence limit of Giacomin et al. {footcite:p}`giacomin2022density` and the power-balance limit of Zanca et al. {footcite:p}`zanca2019power`.

## Troyon limit

The Troyon limit {footcite:p}`troyon1984mhd` sets an upper bound on the plasma pressure the magnetic field can stably hold, conventionally written on the normalised beta $\beta_N$ (formally defined in Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>`, Eq. {eq}`eq:betaN`), $\beta_N < \beta_{N,\mathrm{limit}}$, with a default value of $2.8$, consistent with the experimentally observed limits {footcite:p}`strait1994stability`: beyond it, pressure-gradient-driven MHD modes become unstable {footcite:p}`wesson2011tokamaks,freidberg2007plasma`. The check is performed on the total normalised beta, $\beta_{N,\mathrm{tot}} = \beta_{N,\mathrm{th}} + \beta_{N,\mathrm{fast}}$, thermal and fast-alpha contributions included (Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>`). Indeed, to the author’s knowledge, ideal MHD responds to the total pressure gradient, fast particles included, and the experimental databases behind the Troyon coefficient contain the fast-ion content of their beam-heated discharges. This check makes use of the pressure, the field, and, once again, the plasma current.

## Kink limit

The plasma current must not be too high relative to the toroidal field, or it triggers the current-driven external kink mode and with it a disruption {footcite:p}`wesson1978hydromagnetic,shafranov1970hydromagnetic`. The criterion can be cast in either of two expressions, in terms of the edge safety factor $q_{95}$ or of the cylindrical (Kruskal-Shafranov) safety factor $q^*$ (both defined below), each with its own limiting value. The two are related but not identical measures of the same kink stability. The edge safety factor $q_{95}$ is defined at the normalised poloidal flux surface $\hat{\psi} = 0.95$. D0FUS computes $q_{95}$ from a closed-form expression rather than from a flux-surface average of an equilibrium reconstruction, and offers two expressions selectable by the user.

The two expressions given below share the same structure, discussed along the same lines in Ref. {footcite:p}`sarazin2020scaling`. The safety factor counts the toroidal turns a field line makes per poloidal turn, so it is the ratio of the toroidal to the poloidal field, weighted by the geometry of the surface on which the line winds. In the straight-cylinder limit the boundary poloidal field follows from Ampère’s law as $\mu_0 I_p / (2\pi a)$, while the toroidal field is $B_0$: the ratio, folded with the aspect ratio, produces the prefactor $a^2 B_0/(R_0 I_p)$ common to Eqs. {eq}`eq:q95_sauter` and {eq}`eq:q95_iter89`. The remaining factors correct this cylindrical estimate for the shape of the surface {footcite:p}`sarazin2020scaling`.

The default expression follows Ref. {footcite:p}`sauter2016q95`, which presents a fit of the safety factor at $\hat{\psi}=0.95$ on a database of CHEASE equilibria covering negative triangularity ($-0.6 < \delta < 0.8$) using the LCFS shaping parameters as the reference variables,

$$
q_{95}^\mathrm{Sauter} = \frac{4.1\, a^2\, B_0}{R_0\, I_p}\, f_\kappa(\kappa)\, f_\delta(\delta, \varepsilon)
$$ (eq:q95_sauter)

with $f_\kappa = 1 + 1.2(\kappa - 1) + 0.56(\kappa - 1)^2$, $f_\delta = (1 + 0.09\delta + 0.16\delta^2)(1 + 0.45\delta\varepsilon)/(1 - 0.74\varepsilon)$, $\varepsilon = a/R_0$, and $\kappa$, $\delta$ taken at the LCFS (how these shaping parameters are prescribed or computed is explained in Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`). The squareness factor of the original paper is set to unity in D0FUS, since the 0D context does not resolve squareness independently.

The alternative expression follows the original ITER Physics Design Guidelines {footcite:p}`uckan1990iter,johner2011helios`,

$$
q_{95}^\mathrm{ITER89} = \frac{5\, a^2\, B_0}{R_0\, I_p}\, F_\mathrm{shape}(\kappa_{95}, \delta_{95})\, F_\varepsilon(\varepsilon)
$$ (eq:q95_iter89)

with $F_\mathrm{shape} = [1 + \kappa_{95}^2(1 + 2\delta_{95}^2 - 1.2\delta_{95}^3)]/2$ and $F_\varepsilon = (1.17 - 0.65\varepsilon)/(1 - \varepsilon^2)^2$, evaluated using the shaping parameters at the 95 % flux surface.

The cylindrical $q^*$ follows from the closed-form expression of Freidberg et al. {footcite:p}`freidberg2015designing`,

$$
q^* = \frac{\pi\, a^2\, B_0\, (1 + \kappa^2)}{\mu_0\, R_0\, I_p}
$$ (eq:qstar)

The viability check enforces $q_\mathrm{edge} > q_\mathrm{limit,kink}$, where $q_\mathrm{edge}$ is either $q_{95}$ or $q^*$ depending on the user’s selection. Default thresholds are $q_\mathrm{limit,kink} = 3.0$ to 3.5 for $q_{95}$ (standard ITER and EU-DEMO practice {footcite:p}`iterphysicsbasis1999,coleman2025definition`) and $q_\mathrm{limit,kink} \approx 2.0$ to $2.5$ for $q^*$ {footcite:p}`freidberg2015designing`. This check makes use of the current, the field and the flux-surface geometry.

At this stage, several parameters entering these checks are still unknown: the density, the magnetic field and the plasma current. The flux-surface geometry, on the other hand, is already available, since the major and minor radii $R_0$ and $a$ are direct inputs of D0FUS and the shaping parameters follow from them, as described in the next section.

```{rubric} References
```

```{footbibliography}
```
