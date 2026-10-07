(app:currents_models)=

# Bootstrap current, resistivity and safety-factor profile

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.10. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section gives the two bootstrap formulations, the resistivity models and the safety-factor profile construction summarised in Section {ref}`Plasma current and scaling law <ssec:chap1_currents>`.

## Bootstrap current models

The Academic option is the closed-form analytical model of Segal, Cerfon, and Freidberg {footcite:p}`segal2021bootstrap`, which expresses the bootstrap fraction $f_b \equiv I_b / I_p$ directly as a function of the volume-averaged plasma quantities and the profile peaking exponents,

$$
f_b^\mathrm{Segal} = \frac{K_b\, \bar{n}\, \bar{T}\, R_0^2}{I_p^2}
$$

with the dimensionless coefficient $K_b$ given in closed form by

$$
K_b = 0.6099\; \varepsilon^{2.5}\, \kappa^{1.27}\; (1+\nu_n)(1+\nu_T)(\nu_n + 0.054\, \nu_T)\; C_B(\nu_J, \nu_p)
$$

where the coupling integral $C_B(\nu_J, \nu_p)$ reads

$$
C_B(\nu_J, \nu_p) = \frac{1}{(1-\nu_J)^2}\int_0^1 x^{1/4}\, (1-x)^{\nu_p - 1} \left[1 + (1 - 3\nu_J)\, x + \nu_J\, x^2\right]^2 \mathrm{d}x
$$

with pressure exponent $\nu_p = \nu_n + \nu_T$ and current exponent fixed by Segal’s empirical closure $\nu_J = 0.453 - 0.1\,(\nu_p - 1.5)$. The integral has no closed form and is evaluated numericaly.

For pedestal profiles, D0FUS extends the original parabolic formula by an equivalent-parabola substitution: effective exponents $\nu_n^\mathrm{eff} = \hat{n}(0) - 1$ and $\nu_T^\mathrm{eff} = \hat{T}(0) - 1$ are derived from the actual peak-to-average ratios of the prescribed profiles, then plugged into the Segal expression. This route is fast and avoids any radial integration, but it ignores the collisionality dependence of the neoclassical transport coefficients. For strongly shaped pedestals or high-collisionality scenarios, the Refined model is preferred.

The Refined default is the neoclassical model of Redl et al. {footcite:p}`redl2021bootstrap`, which evaluates the bootstrap current density at each radial grid point from the polynomial transport coefficients $L_{31}$, $L_{32}$, $L_{34}$, $\alpha$, which depend on the trapped-particle fraction $f_t$, the electron collisionality $\nu_e^*$, and $Z_\mathrm{eff}$,

$$
\langle \mathbf{j}_\mathrm{bs} \cdot \mathbf{B}\rangle = - p_e\left[L_{31}\, \frac{\mathrm{d}\ln n_e}{\mathrm{d}\hat{\psi}} + L_{32}\, \frac{\mathrm{d}\ln T_e}{\mathrm{d}\hat{\psi}} + L_{34}\, \alpha\, \frac{\mathrm{d}\ln T_i}{\mathrm{d}\hat{\psi}}\right]
$$ (eq:jbs_redl)

where $\hat{\psi}$ is the normalised poloidal flux. The coefficients are refits from the Sauter (1999) form against the modern drift-kinetic solver NEO {footcite:p}`sauter1999neoclassical,sauter2002erratum,redl2021bootstrap`, providing improved accuracy at high collisionality (relevant for the pedestal layer) and for plasmas with significant impurity content.

Since D0FUS carries the normalised radius $\rho$ rather than the poloidal flux as its radial variable, the logarithmic gradients with respect to the normalised poloidal flux $\hat{\psi}$ in Eq. {eq}`eq:jbs_redl` are evaluated on the $\rho$-grid by the chain rule, $\mathrm{d}/\mathrm{d}\hat{\psi} = (\mathrm{d}\rho/\mathrm{d}\hat{\psi})\,\mathrm{d}/\mathrm{d}\rho$. The poloidal-flux derivative is not prescribed independently: it follows from the self-consistent safety-factor profile through the definition $q = \mathrm{d}\Phi_\mathrm{tor}/\mathrm{d}\Phi_\mathrm{pol}$, so that $\mathrm{d}\psi_\mathrm{pol}/\mathrm{d}\rho = q^{-1}(\rho)\,\mathrm{d}\Phi_\mathrm{tor}/\mathrm{d}\rho$, with the toroidal flux $\Phi_\mathrm{tor}(\rho)$ obtained from the enclosed cross-section and the $1/R$ toroidal field of {ref}`Flux-surface geometry models <app:geometry_models>`, and the normalisation $\hat{\psi} = \psi_\mathrm{pol}/\psi_\mathrm{pol}(1)$ closing the mapping.

A three-way benchmark between D0FUS, the NEOS neoclassical solver, and PROCESS showed that D0FUS and NEOS agree to machine precision on all coefficients $L_{31}$, $L_{32}$, $L_{34}$, $\alpha$. A systematic 40 % difference was identified with PROCESS, traced to an unpublished reformulation by Fable that modifies the assembly of Eq. {eq}`eq:jbs_redl`.

## Plasma resistivity and ohmic power

The ohmic current $I_\Omega$ dissipates power $P_\Omega = \mathcal{R}_p\, I_\Omega^2$, where $\mathcal{R}_p$ is the effective plasma resistance computed by treating the flux surfaces as parallel resistors. D0FUS supports four resistivity models: a simplified formula taken from Wesson {footcite:p}`wesson2011tokamaks` , the classical Spitzer expression {footcite:p}`spitzer1953transport`, the Sauter neoclassical resistivity {footcite:p}`sauter1999neoclassical`, and the updated Redl model {footcite:p}`redl2021bootstrap` (default), which uses the same structure as Sauter with refitted coefficients.

The effective resistance integrates the local conductivity over the plasma volume,

$$
\mathcal{R}_p = \frac{(2\pi R_0)^2}{\displaystyle \int_0^1 \langle R_0/R\rangle_\rho\, \frac{V'(\rho)}{\eta_\mathrm{neo}(\rho)}\, \mathrm{d}\rho}
$$ (eq:Reff)

where $\eta_\mathrm{neo}(\rho) = 1/\sigma_\mathrm{neo}(\rho)$ is the local neoclassical resistivity, with $\sigma_\mathrm{neo}$ the neoclassical electrical conductivity evaluated from the same Sauter or Redl fits as the plasma resistivity option, and $\langle R_0/R\rangle_\rho = 1/\sqrt{1 - (\rho a/R_0)^2}$ is the flux-surface average of $R_0/R$ for concentric circular surfaces.

:::{figure} /figures/thesis/d0fus_resistivity_models.png
:name: fig:chap1_resistivity_models
:width: 55%
:align: center

Plasma resistivity $\eta$ as a function of the electron temperature $T_e$ for the four models supported by D0FUS
:::

## Self-consistent safety factor profile $q(\rho)$

The safety factor profile enters the bootstrap current calculation through the local collisionality $\nu_e^*(\rho) \propto q(\rho)$ and the local trapped fraction $f_t(\rho)$. Using a fixed $q_{95}$ everywhere instead of an actual profile overestimates $\nu_e^*$ at mid-radius, which reduces the predicted $I_b$. D0FUS therefore computes $q(\rho)$ in two complementary ways.

In Academic mode, $q(\rho)$ is given by an analytical formula consistent with the prescribed parametric current profile $j(\rho) \propto (1 - \rho^2)^{\alpha_J}$, with $\alpha_J = 1.5$ as default. The current profile $j(\rho)$ is first rescaled so that its surface integral over the poloidal cross-section matches the prescribed plasma current, $\int j\, \mathrm{d}A = I_p$. In the cylindrical approximation, Ampère’s law gives $B_\theta(\rho) \propto I_\mathrm{enc}(\rho)/\rho$, and the safety factor $q \propto \rho B_T/(R B_\theta)$ therefore reduces to the simple proportionality $q(\rho) \propto \rho^2/I_\mathrm{enc}(\rho)$, with the overall prefactor fixed by the boundary condition $q(\rho_{95}) = q_{95}$. The branch then returns $q$, $j$, $I_\mathrm{enc}$, and the internal inductance $l_i(3)$ (see below) in closed form, without iteration.

In Refined mode, $q(\rho)$ is solved by numerical iteration on the full current decomposition. Starting from the analytical $q(\rho)$ of the Academic branch as initial guess, each Picard step assembles $j_\Omega(\rho) + j_\mathrm{bs}(\rho) + j_\mathrm{CD}(\rho)$ at the current $q(\rho)$. The ohmic current density is distributed as $j_\Omega \propto \sigma_\mathrm{neo}(\rho)$, the bootstrap density follows the Sauter-Redl integrand of Eq. {eq}`eq:jbs_redl`, and the CD density is taken either from the source-specific deposition profile when available or from a fallback Gaussian centred at the user-prescribed $\rho_\mathrm{CD}$ in Academic CD mode. The enclosed current $I_\mathrm{enc}(\rho)$ is updated by the cumulative Ampère integral, and the new $q(\rho) = F\, L_p^2(\rho)\, \langle R^{-2}\rangle(\rho) / (2\pi \mu_0 I_\mathrm{enc}(\rho))$, whose cylindrical limit is the $\rho^2/I_\mathrm{enc}(\rho)$ form above, is mixed with the previous one through a damped relaxation $q^{(k+1)} = \omega\, q_\mathrm{new} + (1-\omega)\, q^{(k)}$ with $\omega = 0.5$ as default damping. The convergence is typically reached in 5 to 10 iterations.

:::{figure} /figures/thesis/d0fus_q_profile.png
:name: fig:chap1_q_profile
:width: 75%
:align: center

Safety factor profile $q(\rho)$ for an ITER-class run, drawn here in academic mode where the analytical form $q(\rho) \propto \rho^2/I_\mathrm{enc}(\rho)$ with $\alpha_J = 1.5$ closes the cylindrical Ampère relation at the imposed $q_{95} = 3$.
:::

```{rubric} References
```

```{footbibliography}
```
