(app:thin_wall_stress)=

# Determination of $\sigma_\theta$ in the TF coil

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.7. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The notation follows that introduced in Section 2.1.

## Thick-wall solution

The Lamé-Clapeyron solution for stresses in a thick-walled cylinder under axisymmetric loading {footcite:p}`Clapeyron1829,timoshenko1970elasticity` writes

$$
\begin{cases}
	\sigma_r(r) = A - \dfrac{B}{r^2}, \\
	\sigma_\theta(r) = A + \dfrac{B}{r^2},
\end{cases}
$$

where $A$ and $B$ are integration constants fixed by the boundary conditions. For a TF coil inner leg in wedging, these are zero internal pressure $\sigma_r(R_{\text{TF}}^{\text{int}}) = 0$ and external pressure $\sigma_r(R_{\text{TF}}^{\text{ext}}) = -P_{\text{TF}}$. Solving the linear system yields

$$
A = -P_{\text{TF}}\,\frac{(R_{\text{TF}}^{\text{ext}})^2}{(R_{\text{TF}}^{\text{ext}})^2 - (R_{\text{TF}}^{\text{int}})^2}, \qquad
B = -P_{\text{TF}}\,\frac{(R_{\text{TF}}^{\text{ext}})^2\,(R_{\text{TF}}^{\text{int}})^2}{(R_{\text{TF}}^{\text{ext}})^2 - (R_{\text{TF}}^{\text{int}})^2}
$$

The hoop stress reaches its maximum at the inner face ($r = R_{\text{TF}}^{\text{int}}$), giving

$$
\boxed{
		|\max(\sigma_\theta^{\text{thick}})|
		= \left|A + \frac{B}{(R_{\text{TF}}^{\text{int}})^2}\right|
		= P_{\text{TF}}\,\frac{2\,(R_{\text{TF}}^{\text{ext}})^2}{(R_{\text{TF}}^{\text{ext}})^2 - (R_{\text{TF}}^{\text{int}})^2}
	}
$$ (eq:hoop_stress_thick)

## Thin-wall approximation

In the thin-wall limit $\Delta_R \equiv R_{\text{TF}}^{\text{ext}} - R_{\text{TF}}^{\text{int}} \ll R_{\text{TF}}^{\text{int}}$, factorising gives $(R_{\text{TF}}^{\text{ext}})^2 - (R_{\text{TF}}^{\text{int}})^2 \simeq 2\,\Delta_R\,R_{\text{TF}}^{\text{ext}}$, so that Eq. {eq}`eq:hoop_stress_thick` reduces to

$$
\boxed{
		\sigma_\theta^{\text{thin}} \approx \frac{P_{\text{TF}}\,R_{\text{TF}}^{\text{ext}}}{R_{\text{TF}}^{\text{ext}} - R_{\text{TF}}^{\text{int}}}
	}
$$ (eq:hoop_stress_thin)

```{rubric} References
```

```{footbibliography}
```
