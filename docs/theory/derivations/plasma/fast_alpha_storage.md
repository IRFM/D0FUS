(app:fast_alpha_integral)=

# Derivation of the fast-alpha storage factor $G_\mathrm{eff}$

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.6. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page derives the closed-form expression of the dimensionless storage factor $G_\mathrm{eff}(E_\alpha/E_c)$ used in Eq. {eq}`eq:Wfast` of the main text. The derivation follows the textbook treatment of Stix {footcite:p}`stix1972heating` and Wesson {footcite:p}`wesson2011tokamaks`, specialised here to a monoenergetic source of alpha particles born at $E_\alpha = 3.52\,\mathrm{MeV}$.

## Slowing-down equation in velocity space

Under the assumption $v_i \ll v \ll v_e$, where $v$ is the alpha velocity and $v_{i,e}$ are the thermal velocities of the bulk ions and electrons, Coulomb collisions cause the alpha particle to lose energy at the rate {footcite:p}`stix1972heating`

$$
\frac{dv}{dt} = -\frac{v}{\tau_\mathrm{se}}\left[1 + \left(\frac{v_c}{v}\right)^{3}\right]
$$ (eq:dvdt)

where $\tau_\mathrm{se}$ is the Spitzer electron drag time and $v_c$ is the critical velocity at which electron and ion drag contribute equally. The corresponding critical energy is $E_c = m_\alpha v_c^2/2$, given in practical units in the main text.

## Stationary isotropic distribution

In steady state, with an isotropic monoenergetic source of strength $S$ (alphas per unit volume per unit time) injected at velocity $v_\alpha$, conservation of particles in velocity space requires

$$
\frac{d}{dv}\!\left[v^{2}\, f(v)\, \frac{dv}{dt}\right] = -\frac{S}{4\pi}\, \delta(v - v_\alpha)
$$

Integrating from $v$ to $v_\alpha$ and using Eq. {eq}`eq:dvdt` yields the canonical isotropic slowing-down distribution

$$
f(v) = \frac{S\, \tau_\mathrm{se}}{4\pi}\, \frac{1}{v^{3} + v_c^{3}}\,\mathcal{H}(v_\alpha - v)
$$ (eq:f_slowing_down)

where $\mathcal{H}$ is the Heaviside step function. This is the Wesson Eq. (5.4.13) form transposed to fusion-born alphas.

## Fast-particle stored energy

The volume-averaged kinetic energy density carried by the slowing-down population follows from

$$
\frac{W_\mathrm{fast}}{V} = \int_0^{v_\alpha}\! \frac{1}{2}m_\alpha v^{2}\, f(v)\, 4\pi v^{2}\, dv = \frac{m_\alpha\, S\, \tau_\mathrm{se}}{2}\int_0^{v_\alpha}\! \frac{v^{4}}{v^{3} + v_c^{3}}\, dv
$$

Changing variable to $u = v/v_c$ and writing the alpha power source as $P_\alpha = S\, E_\alpha = (1/2)\, S\, m_\alpha v_\alpha^{2}$, this rearranges to

$$
W_\mathrm{fast} = P_\alpha\, \tau_\mathrm{se}\, G_\mathrm{eff}\!\left(\frac{E_\alpha}{E_c}\right)
	\qquad
	G_\mathrm{eff}\!\left(\frac{E_\alpha}{E_c}\right) = \frac{E_c}{E_\alpha}\int_0^{u_0}\! \frac{u^{4}}{u^{3} + 1}\, du
$$

with $u_0 = v_\alpha/v_c = \sqrt{E_\alpha/E_c}$. The factor $G_\mathrm{eff}$ contains all the dependence of the stored energy on the operating point.

## Closed-form expression

The integrand is split using $u^{4}/(u^{3}+1) = u - u/(u^{3}+1)$. Partial-fraction decomposition of the second term gives

$$
\frac{u}{u^{3}+1} = \frac{1}{3}\, \frac{u+1}{u^{2}-u+1} - \frac{1}{3}\, \frac{1}{u+1}
$$

and integration of each piece, combined with the boundary value at $u = 0$, yields

$$
\int_0^{u_0}\! \frac{u^{4}}{u^{3} + 1}\, du = \frac{u_0^{2}}{2} + \frac{1}{3}\ln(u_0 + 1) - \frac{1}{6}\ln(u_0^{2} - u_0 + 1) - \frac{1}{\sqrt{3}}\!\left[\arctan\!\frac{2u_0 - 1}{\sqrt{3}} + \frac{\pi}{6}\right]
$$ (eq:closed_form)

This closed form is exact and is evaluated to machine precision in D0FUS, removing the need for numerical quadrature. It belongs to the same analytical class as Wesson’s energy partition function $\Phi(\mathcal{E}_{b0}/\mathcal{E}_c)$ {footcite:p}`wesson2011tokamaks`, which arises from the same partial-fraction decomposition of $1/(t^{3}+1)$ in a related slowing-down integral.

## Asymptotic limits and operating point

Two limits are physically transparent:

- In the limit $E_\alpha \ll E_c$, the alphas are rapidly slowed by ion drag immediately after birth. The stored energy vanishes and $G_\mathrm{eff} \to 0$.

- In the limit $E_\alpha \gg E_c$, the alphas spend essentially their entire slowing-down time in the electron-drag regime. The leading-order behaviour of Eq. {eq}`eq:closed_form` is $u_0^{2}/2$, so $G_\mathrm{eff} \to 1/2$. This asymptote reflects the fact that the time-averaged energy carried by a slowing-down alpha is half its birth energy.

For ITER- and EU-DEMO-class plasmas, $E_\alpha/E_c \sim 7$ to $12$, which places the operating point in the rising part of the curve, with $G_\mathrm{eff}$ taking values around $0.4$.

```{rubric} References
```

```{footbibliography}
```
