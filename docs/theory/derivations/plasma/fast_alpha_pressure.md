(app:fast_alpha_model)=

# Fast-alpha pressure model

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.5. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section gives the isotropic slowing-down model behind the fast-alpha beta of Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>`. The storage factor $G_\mathrm{eff}$ is derived in Section {ref}`Derivation of the fast-alpha storage factor G_(eff) <app:fast_alpha_integral>`.

The isotropic slowing-down framework used here was originally developed in the context of neutral beam injection, where the suprathermal pressure carried by energetic injected ions during their thermalisation plays a significant role in the plasma pressure balance, and was directly inferred from experiments {footcite:p}`stix1972heating,wesson2011tokamaks`. Fusion-born alphas constitute a similar suprathermal population, sustained by a continuous internal source rather than an external injector, and the same formalism applies. D0FUS follows this textbook treatment.

Alpha particles born at $E_\alpha = 3.52\,\mathrm{MeV}$ lose energy to the bulk plasma over a Spitzer electron drag time

$$
\tau_\mathrm{se} = 6.27\times 10^{14}\, \frac{A_\alpha}{Z_\alpha^2}\, \frac{T_e^{3/2}\,[\mathrm{eV}]}{n_e\,[\mathrm{m}^{-3}]\, \ln\Lambda} \quad [\mathrm{s}]
$$ (eq:tau_se)

with $A_\alpha = 4$ and $Z_\alpha = 2$ the atomic mass number and the charge state of the alpha particles, and $\ln\Lambda$ the Coulomb logarithm. Below the critical energy

$$
E_c = 14.8\, A_\alpha\, T_e\, \left(\sum_j \frac{n_j Z_j^2}{n_e A_j}\right)^{2/3} \quad [\mathrm{keV}]
$$ (eq:Ec_stix)

where the sum runs over the background ion species, $n_j$, $Z_j$ and $A_j$ being their density, charge state and atomic mass number, the alpha particles transfer their remaining energy preferentially to the bulk ions rather than the electrons. The reason is that $E_c$ is, by construction, the energy at which the drag on electrons and the drag on ions contribute equally: above it the electron drag dominates, below it the collisional energy relaxation time between the alphas and the bulk ions falls below the one with the electrons, so the residual energy is delivered to the ions {footcite:p}`stix1972heating,wesson2011tokamaks`.

Integrating the isotropic slowing-down distribution over velocity space yields the steady-state stored fast-particle energy

$$
W_\mathrm{fast} = P_\alpha\, \tau_\mathrm{se}\, G_\mathrm{eff}\!\left(\frac{E_\alpha}{E_c}\right)
$$ (eq:Wfast)

where $G_\mathrm{eff}$ is the dimensionless energy storage factor taken from $\Phi(\mathcal{E}_{b0}/\mathcal{E}_c)$ given by Wesson, Eq. (5.4.12). It rises monotonically from zero to the asymptotic value $1/2$ and takes values around $0.4$ in ITER- and EU-DEMO-class plasmas. Its closed-form expression is given in {ref}`Derivation of the fast-alpha storage factor G_(eff) <app:fast_alpha_integral>`.

The fast-alpha pressure then follows from the isotropic relation {footcite:p}`goldston1981isotropic` $p_\mathrm{fast} = (2/3)\, W_\mathrm{fast}/V$, and its contribution to the toroidal beta reads

$$
\beta_\mathrm{fast,\alpha} = \frac{2\mu_0\, p_\mathrm{fast}}{B_0^2} = \frac{4\mu_0}{3}\, \frac{W_\mathrm{fast}}{V\, B_0^2}
$$ (eq:beta_fast)

The total normalised beta is the sum of the thermal and fast contributions, $\beta_{N,\mathrm{tot}} = \beta_{N,\mathrm{th}} + \beta_{N,\mathrm{fast}}$, and it is this total that must be compared against the Troyon limit.

```{rubric} References
```

```{footbibliography}
```
