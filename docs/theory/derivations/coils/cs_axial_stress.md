(app:sigma_z_CS)=

# Axial stress at the CS midplane

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page derives the compressive axial stress of Eq. {eq}`eq:sigma_z_CS_smear`, produced at the midplane of the central solenoid by the fringe field at its extremities. The CS is modelled as a thick solenoid occupying $R \in [R_i, R_e]$ (with $R_i \equiv R_\mathrm{CS}^\mathrm{int}$ and $R_e \equiv R_\mathrm{CS}^\mathrm{ext}$) and $z \in [-h, +h]$ with $h = H_\mathrm{CS}/2$, carrying a uniform smeared azimuthal current density $J_\mathrm{smear} = f_c\, J^\mathrm{wost,CS}$.

Integrating the on-axis contribution $\mathrm{d}B_z = \mu_0 J_\mathrm{smear} R^2 / [2 (R^2 + (z - z')^2)^{3/2}]\, \mathrm{d}R\, \mathrm{d}z'$ of each current ring over the winding ($R$ from $R_i$ to $R_e$, $z'$ from $-h$ to $+h$) gives the axial field at the axial position $z$ {footcite:p}`montgomery1969solenoid`,

$$
B_z(z) = \frac{\mu_0 J_\mathrm{smear}}{2}\Bigl[(h + z)\,\mathcal{L}(h + z) + (h - z)\,\mathcal{L}(h - z)\Bigr]
$$ (eq:app_Bz_solenoid)

where $\mathcal{L}$ is the geometric function introduced in Eq. {eq}`eq:L_function`,

$$
\mathcal{L}(\zeta) = \ln\!\frac{R_e + \sqrt{R_e^2 + \zeta^2}}{R_i + \sqrt{R_i^2 + \zeta^2}}
$$

At the midplane and at the free end this reduces to $B_z(0) = \mu_0 J_\mathrm{smear}\, h\, \mathcal{L}(h)$ and $B_z(h) = \mu_0 J_\mathrm{smear}\, h\, \mathcal{L}(2h)$ respectively, with $B_z(0) > B_z(h)$ since $\mathcal{L}$ is a decreasing function of $\zeta$.

Away from the infinite-solenoid idealisation, $\nabla \cdot \vec{B} = 0$ forces a radial component near the ends. In the paraxial (near-bore) approximation {footcite:p}`humphries1990charged`,

$$
B_r(r, z) \simeq -\frac{r}{2}\,\frac{\partial B_z}{\partial z}
$$

evaluated at the inner winding radius $r = R_i$, where the stress is largest. The azimuthal current crossing this radial field produces an axial body force $f_z = -J_\mathrm{smear}\, B_r$ directed toward the midplane, since the fringe field pulls the winding inward along $z$.

Axial equilibrium $\mathrm{d}\sigma_z/\mathrm{d}z = -f_z = J_\mathrm{smear} B_r$, integrated from the free end ($z = h$, $\sigma_z = 0$) to the midplane, gives

$$
\sigma_z(0) = \frac{J_\mathrm{smear} R_i}{2}\bigl[B_z(h) - B_z(0)\bigr] = -\frac{\mu_0 J_\mathrm{smear}^2\, h\, R_i}{2}\bigl[\mathcal{L}(h) - \mathcal{L}(2h)\bigr]
$$ (eq:app_sigma_z_CS)

which is Eq. {eq}`eq:sigma_z_CS_smear`. The bracket is positive (since $\mathcal{L}(h) > \mathcal{L}(2h)$), so the stress is compressive, as expected for a winding squeezed axially by its own fringe field. The peak value in the steel follows by dividing by the useful steel fraction, $\sigma_z^\mathrm{steel} = \sigma_z^\mathrm{smear}/f_u$. The monolithic, uniform-current idealisation is the main limitation of this estimate: a modular CS with independently powered modules develops a more intricate axial stress pattern, but the present expression reproduces the peak magnitude to within about 20 % of the detailed treatment of Iwasa {footcite:p}`iwasa2009casestudies`.

```{rubric} References
```

```{footbibliography}
```
