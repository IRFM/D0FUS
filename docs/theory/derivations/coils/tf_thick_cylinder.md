(app:B_thick)=

# Thick-cylinder field and smeared radial stress in the TF winding pack

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page establishes the thick-cylinder toroidal field used in Eq. {eq}`eq:fc`, following the classical treatment of thick-walled cylinders under distributed body forces {footcite:p}`timoshenko1970elasticity`, and derives the geometric concentration factor $f_\mathrm{L}$ appearing in the refined radial stress of Eq. {eq}`eq:sigma_r_refined`. The winding pack of the TF inboard leg is modelled as a thick cylindrical shell occupying $R \in [R_\mathrm{TF}^\mathrm{sep}, R_\mathrm{TF}^\mathrm{ext}]$ and carrying a uniform smeared axial current density $j = f_c\, J^\mathrm{wost}$. To lighten the notation, let $R_i \equiv R_\mathrm{TF}^\mathrm{sep}$ and $R_e \equiv R_\mathrm{TF}^\mathrm{ext}$.

Ampère’s law applied to a circle of radius $R$ inside the winding pack, with no current enclosed below $R_i$, gives the enclosed current $N I(R) = j\,\pi(R^2 - R_i^2)$ and therefore

$$
B(R) = \frac{\mu_0\, N I(R)}{2\pi R} = \frac{\mu_0\, j}{2}\left(R - \frac{R_i^2}{R}\right), \qquad R_i \leq R \leq R_e
$$ (eq:app_B_thick)

which vanishes at the bore $R = R_i$ and reaches its maximum $B_\mathrm{max} = B(R_e) = (\mu_0 j/2)(R_e^2 - R_i^2)/R_e$ at the plasma-facing face. Inverting this relation gives the conductor fraction of Eq. {eq}`eq:fc`, $j = f_c\, J^\mathrm{wost} = 2 B_\mathrm{max} R_e / [\mu_0 (R_e^2 - R_i^2)]$.

The smeared current crosses the toroidal field and experiences a centripetal Lorentz body force $j\,B(R)$ per unit volume, directed inward. The total inward force carried by the annular region $[R, R_e]$, per unit height, is the integral of this body force over the shell. It is transmitted as a radial compression across the cylindrical surface at radius $R$, of area $2\pi R$ per unit height, so that the smeared radial stress is

$$
\sigma_r(R) = \frac{1}{R}\int_R^{R_e} j\, B(R')\, R'\, \mathrm{d}R'
$$ (eq:app_sigma_r_def)

By construction $\sigma_r(R_e) = 0$ at the free plasma-facing surface, and $|\sigma_r|$ grows monotonically inward as the load accumulates.

Substituting Eq. {eq}`eq:app_B_thick` into Eq. {eq}`eq:app_sigma_r_def` and evaluating at the bore $R = R_i$,

$$
\sigma_r(R_i) = \frac{1}{R_i}\int_{R_i}^{R_e} \frac{\mu_0 j^2}{2}\left(R'^2 - R_i^2\right) \mathrm{d}R' = \frac{\mu_0 j^2}{6\, R_i}\,(R_e - R_i)^2\,(R_e + 2 R_i)
$$

where the cubic $R_e^3 - 3 R_i^2 R_e + 2 R_i^3$ has been factorised as $(R_e - R_i)^2 (R_e + 2 R_i)$. Factoring out the magnetic pressure $P_\mathrm{TF} = B_\mathrm{max}^2/(2\mu_0)$ at the high-field face, and using $B_\mathrm{max} = (\mu_0 j/2)(R_e^2 - R_i^2)/R_e$, yields

$$
\sigma_r(R_i) = f_\mathrm{L}\, P_\mathrm{TF}, \qquad f_\mathrm{L} = \frac{4\, R_e^2\,(R_e + 2 R_i)}{3\, R_i\,(R_e + R_i)^2}
$$ (eq:app_fL)

which is the factor used in Eq. {eq}`eq:sigma_r_refined` with $R_e = R_\mathrm{TF}^\mathrm{ext}$ and $R_i = R_\mathrm{TF}^\mathrm{sep}$. In the thin-shell limit $R_i \to R_e$ one finds $f_\mathrm{L} \to 1$, recovering the Maxwell magnetic pressure $\sigma_r \to P_\mathrm{TF}$, while $f_\mathrm{L}$ grows as the winding pack thickens (for instance $f_\mathrm{L} \approx 1.27$ at $R_i/R_e = 0.83$ and $f_\mathrm{L} \approx 1.68$ at $R_i/R_e = 0.67$), reflecting the volumetric body force being reacted by a bore surface of shrinking area. The peak stress in the structural steel follows by dividing by the useful steel fraction, $\sigma_r^\mathrm{steel} = \sigma_r(R_i)/f_u$.

When the conductor fraction varies with radius, $f_c = f_c(R)$, the field is obtained by integrating $\mathrm{d}(N I)/\mathrm{d}R = f_c(R)\, J^\mathrm{wost}\, 2\pi R$ inward from $N I(R_e) = B_\mathrm{max}\, 2\pi R_e/\mu_0$, and the smeared radial stress retains the moment form

$$
\sigma_r(R) = \frac{1}{R}\int_R^{R_e} f_c(R')\, J^\mathrm{wost}\, B(R')\, R'\, \mathrm{d}R'
$$

which has no closed form in general but reduces exactly to Eq. {eq}`eq:app_fL` for a uniform $f_c$. This is the expression integrated numerically in the graded model of {ref}`Radially graded conductor fraction <appendix_grading>` (Eq. {eq}`eq:dsigma_r_graded`).

```{rubric} References
```

```{footbibliography}
```
