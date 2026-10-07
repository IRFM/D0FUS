(app:Psi_CS)=

# Determination of $\Psi_{\text{CS}}$

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.8. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The flux through the CS cross-section is computed assuming an infinitely long solenoid and a uniformly distributed current density, so that $B_z(r) = B_{CS}$ for $r \le R_{\text{CS}}^{\text{int}}$ and decreases linearly from $B_{CS}$ to zero across the winding pack, i.e. $B_z(r) = B_{CS}\,(R_{\text{CS}}^{\text{ext}} - r) / (R_{\text{CS}}^{\text{ext}} - R_{\text{CS}}^{\text{int}})$ for $R_{\text{CS}}^{\text{int}} \le r \le R_{\text{CS}}^{\text{ext}}$. Integrating,

$$
\begin{aligned}
		\Psi_{CS} &= 2\pi \int_0^{R_{\text{CS}}^{\text{ext}}} B_z(r)\,r\,dr \\
		&= 2\pi B_{CS} \left[ \frac{(R_{\text{CS}}^{\text{int}})^2}{2} + \frac{1}{R_{\text{CS}}^{\text{ext}} - R_{\text{CS}}^{\text{int}}} \int_{R_{\text{CS}}^{\text{int}}}^{R_{\text{CS}}^{\text{ext}}} (R_{\text{CS}}^{\text{ext}} - r)\,r\,dr \right] \\
		&= \frac{\pi B_{CS}}{3} \left[ (R_{\text{CS}}^{\text{ext}})^2 + R_{\text{CS}}^{\text{ext}} R_{\text{CS}}^{\text{int}} + (R_{\text{CS}}^{\text{int}})^2 \right]
	\end{aligned}
$$

A full swing of the CS provides a flux

$$
\boxed{
	\Psi'_{CS} = \frac{2\pi B_{CS}}{3} \left[ (R_{\text{CS}}^{\text{ext}})^2 + R_{\text{CS}}^{\text{ext}} R_{\text{CS}}^{\text{int}} + (R_{\text{CS}}^{\text{int}})^2 \right]
}
$$
