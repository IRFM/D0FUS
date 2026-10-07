(Annexe_A_thinlayer)=

# $R_{\text{int}}^{TF}$ thin cylinder expression

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.4. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page derives the thin-cylinder expression of the TF inner-leg interface radius used by the Academic radial build model (Section {ref}`Academic model <ssec:chap1_academic>`), for the wedging and bucking configurations in turn.

(app:wedging_academic)=

## Wedging 1st order

Starting from the Tresca expression,

$$
\sigma_{\text{Tresca}} = |\sigma_\theta - \sigma_z|,
$$

and inserting the $\sigma$ expressions, one obtains

$$
\sigma_{\text{Tresca}} = \frac{B_{\max}^2}{2 \mu_0 } \frac{R_{\text{TF}}^{\text{sep}}}{R_{\text{TF}}^{\text{sep}}-R_{\text{TF}}^{\text{int}}} + \frac{B_0^2 R_0^2 \ln \left( \frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B} \right)}{2 \mu_0 \left((R_{\text{TF}}^{\text{sep}})^2 - (R_{\text{TF}}^{\text{int}})^2\right)}
$$

Using $B_0 R_0 \approx B_{\max} R_{\text{TF}}^{\text{sep}}$ (thin cylinder approximation) and simplifying leads to

$$
\sigma_{\text{Tresca}} =
$$

$$
\frac{B_{\max}^2}{2\mu_0}
\left[
\frac{R_{\text{TF}}^{\text{sep}}}{R_{\text{TF}}^{\text{sep}}-R_{\text{TF}}^{\text{int}}}
+ \frac{\left(R_{\text{TF}}^{\text{sep}}\right)^2}{\left(R_{\text{TF}}^{\text{sep}}\right)^2-\left(R_{\text{TF}}^{\text{int}}\right)^2}
\ln \left( \frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B} \right)
\right]
$$ (eq:sigma)

Defining $\delta = R_{\text{TF}}^{\text{sep}} - R_{\text{TF}}^{\text{int}}$ and assuming $\delta \ll R_{\text{TF}}^{\text{sep}}$, this gives

$$
\left(R_{\text{TF}}^{\text{sep}}\right)^2 - \left(R_{\text{TF}}^{\text{int}}\right)^2
= \left(R_{\text{TF}}^{\text{sep}}-R_{\text{TF}}^{\text{int}}\right)\left(R_{\text{TF}}^{\text{sep}}+R_{\text{TF}}^{\text{int}}\right)
\simeq 2\,\delta\, R_{\text{TF}}^{\text{sep}},
$$

so that the Tresca stress reduces to

$$
\sigma_{\text{Tresca}}
\simeq \frac{B_{\max}^2 R_{\text{TF}}^{\text{sep}}}{2\mu_0 \delta}
\left(1+\frac{1}{2}\ln \left( \frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B} \right)\right)
$$

and therefore

$$
\boxed{
	R_{\text{TF}}^{\text{int}}
	\simeq R_{\text{TF}}^{\text{sep}}
	- \frac{B_{\max}^2 R_{\text{TF}}^{\text{sep}}}{2\mu_0 \sigma_{\text{Tresca}}}
	\left(1+\frac{1}{2}\ln \left( \frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}\right)\right)
}
$$

Recall that this expression is valid as a first-order approximation in the thin-wall limit $\delta \ll R_{\text{TF}}^{\text{sep}}$.

(app:bucking_academic)=

## Bucking 1st order

Starting from

$$
\sigma_{\text{Tresca}}
= \frac{B_{\max}^2}{2\mu_0}
+ \frac{B_{\max}^2 (R_{\text{TF}}^{\text{sep}})^2}{2\mu_0\bigl((R_{\text{TF}}^{\text{sep}})^2-(R_{\text{TF}}^{\text{int}})^2\bigr)}
\ln\!\left(\frac{R_0+a+\Delta_B+\Delta_\mathrm{ext}}{R_0-a-\Delta_B}\right),
$$

and applying the same thin-wall approximation as in {ref}`Wedging 1st order <app:wedging_academic>` (i.e. $(R_{\text{TF}}^{\text{sep}})^2 - (R_{\text{TF}}^{\text{int}})^2 \simeq 2\,\delta\,R_{\text{TF}}^{\text{sep}}$ with $\delta = R_{\text{TF}}^{\text{sep}} - R_{\text{TF}}^{\text{int}}$), the Tresca stress reduces to

$$
\sigma_{\text{Tresca}} \simeq \frac{B_{\max}^2}{2\mu_0}\left[1 + \frac{R_{\text{TF}}^{\text{sep}}}{2\,\delta}\ln\!\left(\frac{R_0+a+\Delta_B+\Delta_\mathrm{ext}}{R_0-a-\Delta_B}\right)\right]
$$

The internal radius is therefore

$$
\boxed{\,R_{\text{TF}}^{\text{int}}
	= R_{\text{TF}}^{\text{sep}}\left[1 - \frac{B_{\max}^2}{4\mu_0\left(\sigma_{\text{Tresca}}-\frac{B_{\max}^2}{2\mu_0}\right)}\ln\!\left(\frac{R_0+a+\Delta_B+\Delta_\mathrm{ext}}{R_0-a-\Delta_B}\right)\right]}
$$
