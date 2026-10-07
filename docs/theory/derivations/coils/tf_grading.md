(appendix_grading)=

# Radially graded conductor fraction

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.14. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


In the Refined model (Section {ref}`Refined model <ssec:chap1_refined>`), the conductor fraction $f_c$ is uniform across the winding pack. The Tresca criterion is then saturated only at the most loaded radius $R_\mathrm{TF}^\mathrm{int}$, while regions closer to the outer surface operate below the stress limit. Steel is therefore over-provisioned everywhere except at $R_\mathrm{TF}^\mathrm{int}$.

A graded model relaxes this assumption by allowing $f_c$ to vary with radius. Recalling from Eq. {eq}`eq:sigma_r_refined` that $\sigma_z$ already denotes the axial stress in the *steel* of the WP, the graded extension simply replaces $(1-f_c)$ by the area-weighted average steel fraction $\langle 1-f_c \rangle$ (defined in Eq. {eq}`eq:f_steel_graded`):

$$
\sigma_z = \frac{f_{z,\mathrm{WP}}\,F_z}{\langle 1-f_c \rangle\,S_\mathrm{tot}}
$$ (eq:sigma_z_graded_app)

With this approximation of a single value of $\sigma_z$ shared by all radii, the local Tresca criterion in the WP reads

$$
\frac{\sigma_r(R)}{f_u} + \sigma_z = \sigma_\mathrm{lim}
$$ (eq:tresca_graded)

and can be saturated everywhere in the winding pack by tuning $f_c(R)$ accordingly. At each radius $R$, the local conductor fraction $f_c(R)$ is determined by imposing

$$
f_u = \frac{\sigma_r(R)}{\sigma_\mathrm{lim} - \sigma_z}
$$ (eq:gamma_graded)

which is inverted numerically using the $f_u(f_c,n)$ relation derived in {ref}`Determination of f_(u)(f_(c);n) <app:gamma>`. Since $f_u$ is a decreasing function of $f_c$ (a higher conductor fraction means more stress concentration), and $\sigma_r$ increases monotonically from zero at $R = R_\mathrm{TF}^\mathrm{ext}$ to its peak at $R_\mathrm{TF}^\mathrm{int}$, the model naturally assigns more steel (lower $f_c$) to inner regions where stresses are highest, and more conductor (higher $f_c$) to outer regions.

The radial stress $\sigma_r(R)$ and the toroidal field $B(R)$ within the winding pack are coupled through the electromagnetic body force. The integration runs from the outer surface $R_\mathrm{TF}^\mathrm{ext}$ inward. At radius $R$, the enclosed ampere-turns $NI(R)$ and the field $B(R)$ follow from Ampère’s law:

$$
B(R) = \frac{\mu_0\, NI(R)}{2\pi R}, \qquad \frac{dNI}{dR} = -f_c(R)\,J_\mathrm{TF}^\mathrm{wost}\,2\pi R
$$ (eq:ampere_graded)

with $NI(R_\mathrm{TF}^\mathrm{ext}) = B_\mathrm{max}\,2\pi\,R_\mathrm{TF}^\mathrm{ext}/\mu_0$. The smeared radial stress accumulates as

$$
\frac{d\sigma_r}{dR} = -f_c(R)\,J_\mathrm{TF}^\mathrm{wost}\,B(R)
$$ (eq:dsigma_r_graded)

with $\sigma_r(R_\mathrm{TF}^\mathrm{ext}) = 0$ (free surface). Integration stops at $NI = 0$, which defines $R_\mathrm{TF}^\mathrm{sep}$ and thus $\Delta_\mathrm{WP} = R_\mathrm{TF}^\mathrm{ext} - R_\mathrm{TF}^\mathrm{sep}$.

The vertical stress $\sigma_z$ depends on the geometry through the same expression as in the uniform model (Section {ref}`Refined model <ssec:chap1_refined>`), but with $(1-f_c)$ replaced by the area-weighted average steel fraction:

$$
\langle 1 - f_c \rangle = \frac{\displaystyle\int_{R_\mathrm{TF}^\mathrm{sep}}^{R_\mathrm{TF}^\mathrm{ext}} \bigl(1-f_c(R)\bigr)\,R\,dR}{\displaystyle\int_{R_\mathrm{TF}^\mathrm{sep}}^{R_\mathrm{TF}^\mathrm{ext}} R\,dR}
$$ (eq:f_steel_graded)

Since $\sigma_z$ itself enters the Tresca budget (Eq. {eq}`eq:gamma_graded`) and therefore affects $f_c(R)$, the system is solved iteratively: $\sigma_z$ is initialized from a first guess, the inward integration is performed, the resulting $\langle 1-f_c \rangle$ and $R_\mathrm{TF}^\mathrm{sep}$ are used to update $\sigma_z$, and the process is repeated until convergence (Picard iteration, typically fewer than 10 iterations, with under-relaxation factor 0.2).

The graded model removes structural steel where it is not needed, packing more current density in the outer portion of the winding pack and reducing the total thickness required to carry the same ampere-turns.

{numref}`Figure %s <fig:grading>` compares the winding pack thickness $c_\mathrm{WP}$ obtained with and without radial grading as a function of $B_\mathrm{max}$. The relative gain is visible across the entire field range and is largest at moderate fields where $\sigma_r$ dominates the Tresca budget.

:::{figure} /figures/thesis/TF_grading.png
:name: fig:grading
:width: 75%
:align: center

TF winding pack thickness $c_\mathrm{WP}$ as a function of $B_\mathrm{max}$ for the ungraded and radially graded conductor fraction models ($R_0 = 9$ m, $a = 3$ m, $\Delta_B = 1.7$ m, $\sigma_\mathrm{lim} = 660$ MPa, $J_\mathrm{TF}^\mathrm{wost} = 50$ A/mm$^2$).
:::

This grading model is compatible with all three mechanical configurations (wedging, bucking, plug) and all superconductor scalings. It does not account for manufacturing constraints that would limit the number of distinct conductor grades in practice (multiple conductor production lines, and local structural inhomogeneity of the winding pack that can introduce 3D stress concentrations). A concrete example is the CFETR TF coil, designed with three graded sub-winding-packs (high-Jc Nb$_3$Sn, ITER-like Nb$_3$Sn, NbTi) {footcite:p}`hao2022conductor` and currently prototyped under the CRAFT project {footcite:p}`wu2021preliminary`, which will provide a first fusion-scale demonstration that even a small number of grades captures most of the theoretical gain.

```{rubric} References
```

```{footbibliography}
```
