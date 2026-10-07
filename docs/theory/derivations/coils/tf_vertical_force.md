(app:Fz)=

# Determination of $F_z$

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.6. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The TF coils are themselves the source of the toroidal field $\vec B_\varphi = B_\varphi(R)\,\hat e_\varphi$ with $B_\varphi(R) = B_0 R_0 / R$. In the limit of a large number of coils, each inner leg can be idealised as a thin azimuthal current sheet across which $B_\varphi$ is discontinuous: $B_\varphi(R)$ on the plasma side and zero just beyond. The force per unit length on such a current sheet is given by the cross product of the current with the average of the fields on either side {footcite:p}`jackson1998classical`, so that

$$
d\vec F = I\,d\vec\ell \times \frac{\vec B_{\text{in}} + \vec B_{\text{out}}}{2} = \frac{1}{2}\,I\,d\vec\ell \times \vec B_\varphi,
$$

where $\vec B_\varphi$ is evaluated on the plasma-facing side and $I$ is the current per coil.

Consider a planar coil of arbitrary shape lying in the $(R,z)$ plane, with inner leg at $R_{\text{in}} = R_0 - a - \Delta_B$ and outer leg at $R_{\text{out}} = R_0 + a + \Delta_B + \Delta_\mathrm{ext}$. An element of its contour has components $d\vec\ell = dR\,\hat e_R + dz\,\hat e_z$, and

$$
d\vec\ell \times \vec B_\varphi = B_\varphi(R)\,(dR\,\hat e_z - dz\,\hat e_R)
$$

Integrating the vertical component along the upper half of the contour (from $R_{\text{in}}$ at $z = 0$ up to the top of the coil and back down to $R_{\text{out}}$ at $z = 0$) yields

$$
F_z = \frac{I}{2} \int_{R_{\text{in}}}^{R_{\text{out}}} \frac{B_0 R_0}{R}\,dR = \frac{I\,B_0 R_0}{2}\,\ln\!\left(\frac{R_{\text{out}}}{R_{\text{in}}}\right)
$$

The result depends only on the extreme radii $R_{\text{in}}$ and $R_{\text{out}}$, not on the specific shape of the contour in between, confirming the shape-independence noted in {footcite:p}`freidberg2015designing`.

Applying Ampère’s theorem to the toroidal solenoid formed by the $N_{\text{coil}}$ coils gives $B_0 = \mu_0 N_{\text{coil}} I / (2\pi R_0)$, so that $I = 2\pi R_0 B_0 / (\mu_0 N_{\text{coil}})$. Substituting yields the vertical force per coil,

$$
\boxed{
		F_z = \frac{\pi}{\mu_0\,N_{\text{coil}}}\,B_0^2\,R_0^2\,\ln\!\left(\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}\right)
	}
$$ (eq:Fz_full)

An equivalent derivation, reaching the same logarithmic dependence on the inner and outer radii, is given in Ref. {footcite:p}`thome1982mhd` (Page 109, Eq.3.2).

```{rubric} References
```

```{footbibliography}
```
