(ssec:chap1_ripple)=

# TF coil number and toroidal field ripple

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.3.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


Whichever radial build model is retained, the number of TF coils $N_\mathrm{coil}$ is a quantity worth computing, since it conditions the ripple, the port access and the per-coil loads. It is determined by two simultaneous geometric constraints applied at the outboard side of the machine: the toroidal field ripple must remain low enough for plasma confinement reasons, and the toroidal extent left between coils must be wide enough for remote handling and port access.

The toroidal field $B_\varphi$ produced by a finite number of coils oscillates periodically in the toroidal angle $\varphi$ with a relative amplitude

$$
\delta_\mathrm{ripple} = \frac{B_\varphi^\mathrm{max}(\varphi) - B_\varphi^\mathrm{min}(\varphi)}{B_\varphi^\mathrm{max}(\varphi) + B_\varphi^\mathrm{min}(\varphi)}
$$ (eq:ripple_def)

evaluated at the outboard plasma edge $R = R_0 + a$. This ripple must remain below an admissible threshold $\delta_\mathrm{adm}$ {footcite:p}`wesson2011tokamaks`. The default value adopted in D0FUS is $\delta_\mathrm{adm} = 1\,\%$, following modern superconducting tokamak designs {footcite:p}`mitchell2008iter,kamada2013jt60sa,song2014east,vorpahl2019eudemo`.

By modelling each coil as an infinite straight wire (one for the inner leg at radius $R_\mathrm{TF}^\mathrm{ext} = R_0 - a - \Delta_B$ and one for the outer leg at radius $R_\mathrm{out} = R_0 + a + \Delta_B + \Delta_\mathrm{ext}$, with $\Delta_\mathrm{ext}$ an outboard radial standoff introduced below), a complex-potential analysis detailed in {ref}`Derivation of the toroidal field ripple <app:ripple_derivation>`, in line with the textbook estimations {footcite:p}`wesson2011tokamaks` yields the analytical estimate

$$
\delta_\mathrm{ripple} \approx \left(\frac{R_\mathrm{TF}^\mathrm{ext}}{R_0 + a}\right)^{N_\mathrm{coil}} + \left(\frac{R_0 + a}{R_\mathrm{out}}\right)^{N_\mathrm{coil}}
$$ (eq:ripple_formula)

The inner leg enters through $R_\mathrm{TF}^\mathrm{ext}$ rather than through the inner bore $R_\mathrm{TF}^\mathrm{int}$ because $N_\mathrm{coil}$ is computed before the TF thickness $\Delta_\mathrm{TF}$ is known. The inner-leg term is in any case much smaller than the outer-leg one. The ripple decays as $N_\mathrm{coil}$ increases. The resulting field pattern is shown from above in {numref}`Fig. %s <fig:tf_ripple>`, where the ripple can be observed through the iso-field contours toward each coil.

:::{figure} /figures/thesis/tf_ripple.png
:name: fig:tf_ripple
:width: 75%
:align: center

Top view of the toroidal field magnitude produced by the $N_\mathrm{coil} = 18$ TF coils of the ITER reproduction, computed from the filamentary model used to derive Eq. {eq}`eq:ripple_formula`. The field is shown in arbitrary units so that the ripple is visible. The dashed circle marks the outboard plasma edge $R_0 + a$ where the ripple is evaluated, and the grey disk is the central column (first wall, blanket, shielding, inboard TF, CS).
:::

The toroidal extent available to each coil at the outboard midplane must be wide enough to accommodate the equatorial ports used for remote handling, heating and diagnostics. D0FUS imposes a minimum toroidal access arc $L_\mathrm{min}$ per coil, evaluated at the outer-leg radius:

$$
L_\mathrm{access} = \frac{2\pi R_\mathrm{out}}{N_\mathrm{coil}} \;\geq\; L_\mathrm{min}
$$ (eq:Laccess)

The coil case width is not subtracted explicitly: $L_\mathrm{min}$ is defined as the minimum toroidal length per coil that leaves sufficient room for the port and the surrounding structure.

D0FUS scans $\Delta_\mathrm{ext}$ from zero upward and, at each value, searches for the smallest integer $N_\mathrm{coil}$ that simultaneously satisfies $\delta_\mathrm{ripple} \leq \delta_\mathrm{adm}$ and $L_\mathrm{access} \geq L_\mathrm{min}$. The first $(N_\mathrm{coil}, \Delta_\mathrm{ext})$ pair satisfying both constraints is returned. Applied to the ITER geometry ($R_0 = 6.2$ m, $a = 2.0$ m, $\Delta_B = 1.2$ m, $\delta_\mathrm{adm} = 1\,\%$), with the access margin raised to $L_\mathrm{min} = 3.6$ m to match the ITER equatorial port layout, the code predicts $N_\mathrm{coil} = 18$ with $\delta_\mathrm{ripple} \approx 0.98\,\%$ and $\Delta_\mathrm{ext} \approx 1.2$ m, in agreement with the actual ITER design of 18 D-shaped coils {footcite:p}`mitchell2008iter`.

```{rubric} References
```

```{footbibliography}
```
