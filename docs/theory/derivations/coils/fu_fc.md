(app:gamma)=

# Determination of $f_u(f_c;n)$

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.10. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


Consider a cable of circular cross-section of radius $r_c$ inside a conductor of rectangular cross-section ({numref}`Fig. %s <fig:surfacedilutionconductor>`), of horizontal width $2(r_c + \delta_{S_2})$ and vertical height $2(r_c + \delta_{S_1})$, with $\delta_{S_1} = n\,\delta_{S_2}$ and $0 \leq n \leq 1$.

The circular and total areas are:

$$
S_{\mathrm{circle}} = \pi\,r_c^2,
\qquad
S_{\mathrm{total}} = 4(r_c + \delta_{S_2})(r_c + \delta_{S_1})
$$

From the definition of $f_u$, considering a force acting in the vertical direction and the associated load-bearing section in the horizontal direction:

$$
f_u = \frac{\delta_{S_2}}{r_c + \delta_{S_2}} \;\Longrightarrow\; \delta_{S_2} = \frac{f_u\,r_c}{1 - f_u}
$$

Substituting into $\delta_{S_1} = n\,\delta_{S_2}$ gives $\delta_{S_1} = n\, f_u\, r_c/(1-f_u)$. Inserting both into $S_{\mathrm{total}}$ leads to:

$$
S_{\mathrm{total}} = \frac{4r_c^2}{(1 - f_u)^2}\bigl(1 + f_u (n - 1)\bigr)
$$

Therefore,

$$
\boxed{
	f_c(f_u, n) = \frac{\pi(1 - f_u)^2}{4(1 + f_u (n - 1))}
}
$$

This expression can be inverted to obtain $f_u$ as a function of $f_c$:

$$
\boxed{
	\begin{gathered}
		f_u(f_c,n) = \\[6pt]
		\frac{2\pi + 4 f_c (n-1) \pm \sqrt{\bigl(2\pi + 4 f_c (n-1)\bigr)^2 - 4\pi(\pi - 4 f_c)}}{2\pi}
	\end{gathered}
}
$$

with the physically meaningful root corresponding to the “$-$” branch.
