(ssec:chap1_field_beta)=

# Magnetic field and plasma beta

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.5. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


## On-axis toroidal field

The toroidal magnetic field of a tokamak is produced by the TF coil set and falls as $1/R$ across the major radius. D0FUS takes the peak field $B_\mathrm{max}$ at the inboard winding pack as an input parameter (it is usually linked to the superconductor operating limits, see Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`) and derives the on-axis field $B_0$ from the simple geometric relation

$$
B_0 = B_\mathrm{max}\left(1 - \frac{a + \Delta_B}{R_0}\right)
$$ (eq:B0)

where $\Delta_B$ is the total inboard radial distance between the plasma boundary and the TF coil inner face, i.e. the sum of the radial build contributions from the first wall, breeding blanket, neutron shield, vacuum vessel, and inter-component gaps, which are all defined on the radial build schematic of {numref}`Fig. %s <fig:radial_build_clean>`.

## Poloidal field

The poloidal field is not directly an input parameter: it is derived from the plasma current through Ampère’s law. In its integral form, and neglecting the displacement current at the time scales of interest here, the latter states that the circulation of the magnetic field along a closed contour equals $\mu_0$ times the current the contour encloses, $\oint \vec{B}\cdot\mathrm{d}\vec{l} = \mu_0 I$. Applied to the LCFS itself, on which only the poloidal component contributes and which encloses the whole plasma current, it gives the perimeter-averaged field

$$
\langle B_\mathrm{pol} \rangle = \frac{\mu_0\, I_p}{L_\theta}
$$ (eq:Bpol)

with $L_\theta$ the LCFS poloidal perimeter (Eq. {eq}`eq:ramanujan_perimeter` in Academic mode, Eq. {eq}`eq:Lp_miller` in Refined mode).

## Plasma beta

The plasma beta measures the ratio of the plasma kinetic pressure to the magnetic field pressure. Three different normalisations are used in tokamak physics, corresponding to the three relevant magnetic-field components. The toroidal beta is the ratio of the volume-averaged kinetic pressure to the toroidal magnetic pressure on axis,

$$
\beta_T = \frac{2\mu_0\, \bar{p}}{B_0^2}
$$ (eq:betaT)

The poloidal beta uses the average poloidal field of Eq. {eq}`eq:Bpol` as the reference,

$$
\beta_P = \frac{2\mu_0\, \bar{p}}{\langle B_\mathrm{pol}\rangle^2}
$$ (eq:betaP)

The total magnetic-field beta combines the two contributions through the harmonic relation

$$
\beta = \frac{\beta_T\, \beta_P}{\beta_T + \beta_P}
$$ (eq:beta_total)

which follows directly from the definitions: with $\beta = 2\mu_0 \bar{p}/B^2$ and $B^2 = B_0^2 + \langle B_\mathrm{pol}\rangle^2$ in the large-aspect-ratio limit, summing $1/\beta_T$ and $1/\beta_P$ gives $1/\beta$.

The normalised beta is defined as

$$
\beta_N = \frac{\beta\,[\%]\, a\, B_0}{I_p} \quad [\%\,\mathrm{m\,T\,MA^{-1}}]
$$ (eq:betaN)

with $\beta$ in percent, $a$ in metres, $B_0$ in Tesla, and $I_p$ in MA. Following the original Troyon convention, retained by the experimental databases and by reactor design analyses {footcite:p}`troyon1984mhd,strait1994stability,freidberg2015designing`, the $\beta$ entering this definition is the toroidal beta of Eq. {eq}`eq:betaT`, built on the vacuum toroidal field. It is this $\beta_N$ that the Troyon check of Section {ref}`Stability limits <ssec:chap1_stability>` uses, thermal and fast-alpha contributions included. The poloidal beta and $\langle B_\mathrm{pol}\rangle$, although they do not enter $\beta_N$, are kept in the chain: they enter the vertical-field flux contribution of Section {ref}`Academic model <ssec:chap1_academic>` and the scrape-off-layer width scaling of Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>`.

## Fast-alpha pressure contribution

In a burning plasma, the alphas born at $3.5\,\mathrm{MeV}$ carry an additional pressure component during their slowing down. It is not part of the thermal energy entering the confinement scalings, but it contributes to driving MHD instabilities {footcite:p}`wesson2011tokamaks`, so it must be counted against the Troyon limit {footcite:p}`troyon1984mhd,iterphysicsbasis1999`. D0FUS follows the textbook isotropic slowing-down treatment originally developed for neutral-beam ions {footcite:p}`stix1972heating`, whose equations and orders of magnitude are collected in {ref}`Fast-alpha pressure model <app:fast_alpha_model>` and {ref}`Derivation of the fast-alpha storage factor G_(eff) <app:fast_alpha_integral>`. The resulting fast beta adds to the thermal one, and it is the total $\beta_{N,\mathrm{tot}} = \beta_{N,\mathrm{th}} + \beta_{N,\mathrm{fast}}$ that is compared to the Troyon limit in the viability check of Section {ref}`Stability limits <ssec:chap1_stability>`, both contributions being reported separately in the outputs.

The Troyon check, like the Greenwald one, still requires the plasma current. That current is obtained in two steps. The power balance between the heating and loss channels determines the energy confinement time the plasma must achieve. An empirical scaling law, inverted, converts this confinement time into the required current.

```{rubric} References
```

```{footbibliography}
```
