(Annexe_B_thinlayer)=

# Determination of $B_{TF}$

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.5. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


Denote by $R_{\rm TF}^{\rm sep}$ the inner boundary of the current-carrying annulus, i.e. the interface between the winding pack and the steel nose.

Applying Ampère’s theorem with a uniform current density $J_{\rm TF}^{\rm wost}$,

$$
\oint \mathbf{B}\cdot d\mathbf{l} = \mu_0 \int_0^{2\pi} d\varphi \int_{R_{\rm TF}^{\rm sep}}^{R} r\, dr\, J_{\rm TF}^{\rm wost},
$$

and solving for $B(R)$ yields

$$
B(R) = \frac{\mu_0 J_{\rm TF}^{\rm wost}}{2}\left(R - \frac{(R_{\rm TF}^{\rm sep})^2}{R}\right)
$$

Evaluating at $R = R_{\rm TF}^{\rm ext}$ gives the thick-cylinder TF coil peak field,

$$
\boxed{
	B_{\max} = B(R_{\rm TF}^{\rm ext}) = \frac{\mu_0 J_{\rm TF}^{\rm wost}}{2}\left(R_{\rm TF}^{\rm ext} - \frac{(R_{\rm TF}^{\rm sep})^2}{R_{\rm TF}^{\rm ext}}\right)
}
$$

In the thin-shell limit $\Delta R = R_{\rm TF}^{\rm ext} - R_{\rm TF}^{\rm sep} \ll R_{\rm TF}^{\rm ext}$, the right-hand side can be factorised as

$$
B_{\max} = \frac{\mu_0 J_{\rm TF}^{\rm wost}}{2 R_{\rm TF}^{\rm ext}}\left(R_{\rm TF}^{\rm ext} - R_{\rm TF}^{\rm sep}\right)\left(R_{\rm TF}^{\rm ext} + R_{\rm TF}^{\rm sep}\right),
$$

and using $R_{\rm TF}^{\rm ext} + R_{\rm TF}^{\rm sep} \simeq 2\,R_{\rm TF}^{\rm ext}$, one obtains

$$
\boxed{
	B_{\max} \approx \mu_0 J_{\rm TF}^{\rm wost}\,(R_{\rm TF}^{\rm ext} - R_{\rm TF}^{\rm sep})
}
$$
