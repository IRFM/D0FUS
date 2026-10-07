(app:blanket_freidberg)=

# The inboard blanket thickness estimate of Freidberg et al.

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page reproduces the calculation from which the default inboard thickness $\Delta_B = 1.2$ m of Section {ref}`Radial build definition and geometry <ssec:chap1_radial_build>` is traced, due to Freidberg *et al.* {footcite:p}`freidberg2015designing`. The blanket is idealised as a one-dimensional slab of pure natural lithium, entered at $x = 0$ by the flux $\Gamma_0$ of $14.1$ MeV fusion neutrons. Two processes compete inside the slab. The fast neutrons are slowed down by elastic, hard-sphere collisions on Li-7, with a constant cross-section $\sigma_S \approx 2$ barn giving a mean free path $\lambda_S = 1/(N_7 \sigma_S) \approx 0.1$ m. The slowed neutrons are then absorbed by the $7.5\,\%$ of Li-6 contained in natural lithium, through the breeding reaction of Eq. (2) of the thesis introduction, whose cross-section follows the $1/v$ law, $\sigma_\mathrm{BR}(E) = \sigma_B\,(E_T/E)^{1/2}$ with $\sigma_B = 960$ barn at $E_T = 0.025$ eV, giving a thermal absorption length $\lambda_B = 1/(N_6 \sigma_B) \approx 3$ mm.

Writing $E(x)$ for the local neutron energy and $\Gamma(x) = nv$ for the neutron flux, the slab is governed by the energy and mass balance equations of Ref. {footcite:p}`freidberg2015designing` (their Eq. 2),

$$
\frac{\mathrm{d}E}{\mathrm{d}x} = -\frac{E}{\lambda_S},
	\qquad
	\frac{\mathrm{d}\Gamma}{\mathrm{d}x} = -\frac{\Gamma}{\lambda_\mathrm{BR}(E)},
	\qquad
	\lambda_\mathrm{BR}(E) = \lambda_B \left(\frac{E}{E_T}\right)^{1/2},
$$ (eq:freidberg_blanket_balance)

with $E(0) = 14.1$ MeV and $\Gamma(0) = \Gamma_0$: the energy decays geometrically with the collision depth, while the neutrons are lost to breeding at the energy-dependent absorption length. Solving the first equation for $E(x)$, substituting into the second and inverting $\Gamma(x)$ yields the thickness needed to bring the input flux down to a residual level $\Gamma_b$,

$$
\Delta x = \lambda_S \ln\!\left[ 1 + \alpha_B \ln\!\left( \frac{\Gamma_0}{\Gamma_b} \right) \right],
	\qquad
	\alpha_B = \frac{\lambda_B}{\lambda_S}\left(\frac{E_F}{E_T}\right)^{1/2} \simeq 710 ,
$$ (eq:freidberg_blanket)

which, for the essentially complete burn-up required by the authors, $\Gamma_b/\Gamma_0 = 10^{-5}$, gives $\Delta x \simeq 0.9$ m. Adding the first wall, the neutron multiplication region, the shield and the vacuum vessel, and calibrating the sum against detailed blanket studies, Freidberg *et al.* retain $b \approx 1.2$ m {footcite:p}`freidberg2015designing`.

Two features of Eq. {eq}`eq:freidberg_blanket` justify treating $\Delta_B$ as a design-independent constant. The attenuation requirement enters through a double logarithm, so it is almost inert: sweeping $\Gamma_b/\Gamma_0$ over four decades, from $10^{-3}$ to $10^{-7}$, moves $\Delta x$ only from $0.85$ to $0.93$ m, and the absolute neutron flux, hence the design point, does not enter at all. Conversely the neutron energy enters at first order, through $\lambda_S$ and through $\alpha_B \propto E_F^{1/2}$. this is a slowing-down and breeding calculation rather than a shielding one: no dose or nuclear-heating criterion on the magnet appears in the derivation, the shield being folded into the calibrated step from $0.9$ to $1.2$ m, and no target tritium breeding ratio is quoted, the criterion being the residual flux alone.

```{rubric} References
```

```{footbibliography}
```
