(Dependence_sigma_z)=

# Geometrical dependence of $\Delta_{\rm TF}$ in bucking

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.9. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page uses the Academic model assumptions. In bucking, the radial stress $\sigma_r = B_{\max}^2/(2\mu_0)$ is set by the magnetic pressure alone and does not depend on $R_0$ or $a$. Saturating the Tresca criterion $\sigma_z + |\sigma_r| = \sigma_{\rm lim}$ therefore fixes $\sigma_z$ independently of the geometry,

$$
\sigma_z \;=\; \sigma_{\rm lim} - \frac{B_{\max}^2}{2\mu_0}
$$ (eq:sigmaz_allowable)

Using the notation $R_{\rm in} = R_0 - a - \Delta_B$ and $R_{\rm out} = R_0 + a + \Delta_B + \Delta_\mathrm{ext}$ introduced in {ref}`Determination of F_(z) <app:Fz>`, the thin-cylinder approximation $B_0 R_0 \approx B_{\max}\,R_{\rm in}$ recasts Eq. {eq}`eq:Fz_full` as

$$
F_z \;=\; \frac{\pi B_{\max}^2}{\mu_0\,N_{\rm coil}}\,R_{\rm in}^2\,\ln(R_{\rm out}/R_{\rm in})
$$ (eq:Fz_compact)

In the thin-wall limit $\Delta_{\rm TF} \ll R_{\rm in}$, the steel cross-section of the inboard leg that carries the tension is

$$
S \;=\; \pi\bigl(R_{\rm in}^2 - (R_{\rm in}-\Delta_{\rm TF})^2\bigr) \;\approx\; 2\pi\,R_{\rm in}\,\Delta_{\rm TF}
$$ (eq:area_thin)

Substituting Eqs. {eq}`eq:Fz_compact` and {eq}`eq:area_thin` yields

$$
\sigma_z \;=\; \frac{N_{\rm coil}\,F_z}{2\,S} \;\approx\; \frac{B_{\max}^2}{4\mu_0}\,\frac{R_{\rm in}\,\ln(R_{\rm out}/R_{\rm in})}{\Delta_{\rm TF}}
$$ (eq:sigmaz_thin)

Combining Eqs. {eq}`eq:sigmaz_allowable` and {eq}`eq:sigmaz_thin` and solving for $\Delta_{\rm TF}$,

$$
\boxed{\,\Delta_{\rm TF} \;=\; \frac{B_{\max}^2}{4\mu_0\,\sigma_z}\,R_{\rm in}\,\ln(R_{\rm out}/R_{\rm in})}
$$ (eq:DTF_final)

The geometrical factor $R_{\rm in}\,\ln(R_{\rm out}/R_{\rm in})$ is strictly increasing with $R_0$ at fixed $a$.
