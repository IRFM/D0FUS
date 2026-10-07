(app:ripple_derivation)=

# Derivation of the toroidal field ripple

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.15. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page derives the toroidal field ripple produced by the finite number of TF coils, the constraint that sets the minimum coil count of the radial build and a standard element of tokamak design {footcite:p}`wesson2011tokamaks`.

## Setup

The aim is to estimate the magnetic field ripple produced by $N$ discrete TF coils in a tokamak. On the midplane, each coil can be approximated by two infinite straight conductors: one at the inner leg radius $R_{\rm in}$ and one at the outer leg radius $R_{\rm out}$. The ripple from $N$ conductors on a single circle is derived first, then apply it to both circles.

## Vector potential of a single infinite wire

An infinite wire carrying a current $I$ along the vertical axis at a distance $d$ from the observation point produces a magnetic field $B = \mu_0 I / (2\pi d)$ circling around the wire. This field derives from a vector potential directed along the wire axis:

$$
A = -\frac{\mu_0 I}{2\pi} \ln d
$$

The toroidal magnetic field is recovered by $B_\varphi = -\partial A / \partial R$, which gives back $\mu_0 I/(2\pi d)$.

## $N$ wires on a circle: complex notation

Place $N$ wires equally spaced on a circle of radius $R_c$. To sum their contributions, positions in the $(x,y)$ midplane are written as the complex number $w = x + iy = R\,e^{i\varphi}$, where $R$ is the distance to the machine axis and $\varphi$ the toroidal angle. The $k$-th wire is at:

$$
w_k = R_c\,e^{i\,2\pi k/N}, \qquad k = 0, 1, \ldots, N-1
$$

The distance from the observation point $w$ to wire $k$ is $|w - w_k|$, so the total vector potential is:

$$
A = -\frac{\mu_0 I}{2\pi} \sum_{k=0}^{N-1} \ln|w - w_k|
	= -\frac{\mu_0 I}{2\pi} \ln \left|\prod_{k=0}^{N-1} (w - w_k)\right|
$$ (eq:A_sum)

where $\sum \ln = \ln \prod$ and $\prod |a_k| = |\prod a_k|$ have been used.

## The key algebraic identity

The wire positions $w_k = R_c\,e^{i\,2\pi k/N}$ are the $N$ roots of the polynomial equation $X^N = R_c^N$. Indeed, $w_k^N = R_c^N\,e^{i\,2\pi k} = R_c^N$ for any integer $k$. Since a degree-$N$ polynomial with leading coefficient 1 is entirely determined by its $N$ roots:

$$
X^N - R_c^N = \prod_{k=0}^{N-1}(X - w_k)
$$

Setting $X = w$ (the observation point):

$$
\boxed{\prod_{k=0}^{N-1}(w - w_k) = w^N - R_c^N}
$$

Inserting into Eq. ({eq}`eq:A_sum`):

$$
A = -\frac{\mu_0 I}{2\pi} \ln\left|w^N - R_c^N\right|
$$ (eq:A_total)

## Separating the axisymmetric field from the ripple

To identify the ripple, split $A$ into an axisymmetric part (independent of $\varphi$) and an angular modulation:

$$
w^N - R_c^N = w^N\!\left[1 - \left(\frac{R_c}{w}\right)^{\!N}\right]
$$

so that, using $|w^N| = R^N$:

$$
\ln\left|w^N - R_c^N\right| = \underbrace{N\ln R}_{\text{axisymmetric}} \;+\; \underbrace{\ln\left|1 - \left(\frac{R_c}{w}\right)^{\!N}\right|}_{\text{contains the ripple}}
$$ (eq:split)

**Axisymmetric part.** The first term gives the axisymmetric potential:

$$
A^{(0)} = -\frac{\mu_0 NI}{2\pi}\ln R
$$

Taking the radial derivative:

$$
B_\varphi^{(0)} = -\frac{\partial A^{(0)}}{\partial R} = \frac{\mu_0 NI}{2\pi R}
$$

This is the Ampère law result for a total current $NI$: the toroidal field of an ideal solenoid.

**Ripple for $R > R_c$.** The second term in Eq. ({eq}`eq:split`) is easy to expand when $R > R_c$, because $(R_c/R)^N \ll 1$. Writing $w = Re^{i\varphi}$:

$$
\left(\frac{R_c}{w}\right)^{\!N} = \left(\frac{R_c}{R}\right)^{\!N} e^{-iN\varphi}
$$

Using $\ln|1 - \epsilon\, e^{i\theta}| \approx -\epsilon\cos\theta$ for $\epsilon \ll 1$ (see Eq. ({eq}`eq:modulus`) for the detailed calculation):

$$
\ln\left|1 - \left(\frac{R_c}{w}\right)^{\!N}\right| \approx -\left(\frac{R_c}{R}\right)^{\!N} \cos N\varphi
$$

The ripple amplitude relative to $B_\varphi^{(0)}$ is therefore:

$$
\boxed{\delta(R) \sim \left(\frac{R_c}{R}\right)^{N} \quad \text{for } R > R_c}
$$

**Ripple for $R < R_c$.** For $R < R_c$, the ratio $(R_c/R)^N \gg 1$ and the expansion of Eq. ({eq}`eq:split`) does not converge. Going back to Eq. ({eq}`eq:A_total`) and factoring differently:

$$
w^N - R_c^N = -R_c^N\left[1 - \left(\frac{R}{R_c}\right)^{\!N} e^{iN\varphi}\right]
$$

Now $(R/R_c)^N \ll 1$, so the same small-parameter expansion applies (Eq. ({eq}`eq:modulus`)):

$$
\ln\left|w^N - R_c^N\right| = N\ln R_c \;-\; \left(\frac{R}{R_c}\right)^{\!N} \cos N\varphi
$$

The first term is a constant (no $R$-dependence, no contribution to $B_\varphi$). The angular modulation gives:

$$
\boxed{\delta(R) \sim \left(\frac{R}{R_c}\right)^{N} \quad \text{for } R < R_c}
$$

In both cases, the ripple is a $\cos(N\varphi)$ modulation whose amplitude decays as the $N$-th power of the ratio between the smaller and the larger of $R$ and $R_c$.

**Modulus of $1 - \epsilon\,e^{i\theta}$.** For completeness, here is the expansion used above. For $\epsilon \ll 1$:

$$
\begin{aligned}
\left|1 - \epsilon\,e^{i\theta}\right|^2
	&= (1 - \epsilon\cos\theta)^2 + (\epsilon\sin\theta)^2 \\
	&= 1 - 2\epsilon\cos\theta + \epsilon^2
\end{aligned}
$$

Therefore:

$$
\ln\left|1 - \epsilon\,e^{i\theta}\right|
	= \frac{1}{2}\ln\left(1 - 2\epsilon\cos\theta + \epsilon^2\right)
	\approx -\epsilon\cos\theta
$$ (eq:modulus)

where $\ln(1+x) \approx x$ has been used with $x = -2\epsilon\cos\theta + \epsilon^2 \approx -2\epsilon\cos\theta$.

## Application to the tokamak

In a tokamak, the TF coils have two sets of conductors: inner legs on a circle of radius $R_{\rm in}$ and outer legs on a circle of radius $R_{\rm out}$. The ripple at any point is the sum of both contributions.

At the outboard plasma edge $R = R_0 + a$, one has $R > R_{\rm in}$ and $R < R_{\rm out}$, so:

$$
\boxed{\delta_{\rm ripple} \approx \left(\frac{R_{\rm in}}{R_0+a}\right)^N + \left(\frac{R_0+a}{R_{\rm out}}\right)^N}
$$

```{rubric} References
```

```{footbibliography}
```
