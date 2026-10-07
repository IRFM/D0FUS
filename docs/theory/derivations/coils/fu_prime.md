(Appendix_little_demo)=

# Determination of $f_u'(f_u ; n)$

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.11. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The stress concentration factor in the horizontal direction, denoted $f_u'$, is given by:

$$
f_u' = \frac{\delta_{S_1}}{r_c + \delta_{S_1}}
$$

Combining this with the vertical-direction expression $f_u = \delta_{S_2}/(r_c + \delta_{S_2})$ and the relation $\delta_{S_1} = n\,\delta_{S_2}$, one obtains:

$$
\boxed{
	f_u' = \frac{n\, f_u}{1 - f_u + n\, f_u}
}
$$
