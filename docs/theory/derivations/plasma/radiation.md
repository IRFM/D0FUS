(app:radiation_models)=

# Radiation models

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.7. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page details the effective charge and the three radiation channels entering the power balance of Section {ref}`Power balance and energy confinement time <ssec:chap1_power_balance>`: their emissivities, their validity domains and the atomic data they rely on.

## Effective charge: derivation of the grouped form

The effective charge is defined as $Z_\mathrm{eff} \equiv \sum_{s\,\in\,\mathrm{ions}} n_s Z_s^2 / n_e$, where the sum runs over all the ion species present in the plasma and only over them, the electrons being carried by the normalisation $n_e$. It is the charge of the fictitious pure plasma that would produce the same electron-ion Coulomb collisionality, since the scattering rate of an electron off an ion of charge $Z$ scales as $Z^2$. Writing the sum over the hydrogenic fuel, the helium ash and the impurity species, and eliminating the fuel fraction through quasi-neutrality ($f_\mathrm{fuel} = 1 - 2 f_\alpha - \sum_{j\,\neq\,\mathrm{He}} \langle Z_j\rangle f_j$, where this second sum runs over all the impurities other than helium, whose contribution is already carried by $f_\alpha$), the definition collapses to the grouped form used in the chapter: the leading 1 is the fuel, the ash contributes $2 f_\alpha$ (from $Z_\mathrm{He}^2 f_\alpha - 2 f_\alpha$ after the quasi-neutrality substitution), and the impurity sum combines the linear dilution correction with the $\langle Z_j\rangle^2$ moment. The mean charge states $\langle Z_j\rangle(T_e)$ are the coronal values of the Mavrin fits, so a clean fuel-plus-ash plasma recovers $Z_\mathrm{eff} = 1 + 2 f_\alpha$.

This section gives the three radiation channels summarised in Section {ref}`Radiation losses <ssec:chap1_radiation>`: Bremsstrahlung, synchrotron emission, and impurity line radiation.

### Bremsstrahlung

The volume-integrated Bremsstrahlung power follows the textbook scaling $\propto n_e^2\, Z_\mathrm{eff}\, T_e^{1/2}$ {footcite:p}`wesson2011tokamaks` and reads

$$
P_\mathrm{brem} = C_B\, Z_\mathrm{eff}^\mathrm{fuel}\, \bar{n}_e^2\, \bar{T}^{1/2}\, V\, f_\mathrm{peak}
$$ (eq:Pbrem)

with the practical-units coefficient $C_B = 5.35\times 10^3\, \mathrm{W\,m^3\,keV^{-1/2}}$ (densities expressed in $10^{20}\,\mathrm{m}^{-3}$). The effective charge entering Eq. {eq}`eq:Pbrem` is the fuel one, built on deuterium, tritium and helium ash only: the free-free emission of the intrinsic and seeded impurities is already included in the Mavrin cooling rates of the line-radiation channel below, and counting it here as well would count it twice. The dimensionless profile factor $f_\mathrm{peak} = \langle\hat{n}^2 \hat{T}^{1/2}\rangle_\mathrm{vol}$ captures the impact of the radial profile on the volume integral, weighted by the appropriate volume element. For purely parabolic profiles (Eq. {eq}`eq:profile_parabolic`) in cylindrical geometry, the integral admits the closed form

$$
f_\mathrm{peak} = \frac{(1+\nu_n)^2 (1+\nu_T)^{1/2}}{1 + 2\nu_n + \nu_T/2}
$$

which is the analytical fast path used in Academic mode {footcite:p}`sheffield1994`. When a pedestal is prescribed ($\rho_\mathrm{ped} < 1$) or Refined geometry is selected, $f_\mathrm{peak}$ is evaluated numerically by trapezoidal integration on the same radial grid as the profiles, with the cylindrical weight $2\rho$ in Academic mode and the Miller Jacobian $V'(\rho)/V$ in Refined mode. Bremsstrahlung peaks on axis where $n_e^2$ is largest, so D0FUS treats it as a purely core contribution.

### Synchrotron radiation

Unlike Bremsstrahlung, synchrotron emission is partially reabsorbed by the plasma at the lower harmonics, and the unabsorbed fraction reflects multiple times off the metallic first wall before either escaping or being absorbed back. The net synchrotron loss is therefore sensitive to the wall reflectivity, the plasma opacity, and the on-axis temperature, in a way that no simple one-line scaling can capture. D0FUS uses the Albajar parameterisation {footcite:p}`albajar2001synchrotron` corrected by the Fidone wall-reflectivity treatment {footcite:p}`fidone2001synchrotron`, expressed in practical units as

$$
P_\mathrm{syn} = 3.84\times 10^{-8}\, (1 - R_w)^{0.62}\, R_0\, a^{1.38}\, \kappa^{0.79}\, B_0^{2.62}\, n_{e,0}^{0.38}\, T_0\, (16 + T_0)^{2.61}\, \mathcal{F}\, K\, G
$$ (eq:Psyn)

where $T_0$ and $n_{e,0}$ are the on-axis temperature and density (in keV and $10^{20}\,\mathrm{m}^{-3}$), $R_w$ is the first-wall reflectivity for synchrotron photons, and the three dimensionless factors capture the geometry, the profile, and the opacity. The aspect-ratio factor (Albajar Eq. 15) reads $G = 0.93\, [1 + 0.85\, e^{-0.82 R_0/a}]$, the profile factor (Albajar Eq. 13) reads

$$
K = \frac{(1.98 + \nu_T)^{1.36}\, m_T^{2.14}}{\bigl(\nu_n + 3.87 \nu_T + 1.46\bigr)^{0.79}\, \bigl(m_T^{1.53} + 1.87 \nu_T - 0.16\bigr)^{1.33}}
$$

and the Fidone opacity correction reads $\mathcal{F} = [1 + 0.12 (T_0/p_{a0}^{0.41})(1-R_w)^{0.41}]^{-1.51}$ with the opacity parameter $p_{a0} = 6.04\times 10^3\, a\, n_{e,0}/B_0$. The inner temperature exponent $m_T$ of the Albajar profile family $T \propto (1 - \rho^{m_T})^{\nu_T}$ (written $\beta_T$ in the original paper, renamed here to avoid confusion with the toroidal beta) defaults to 2, the parabolic shape. Note that the on-axis values $T_0$ and $n_{e,0}$ entering Eq. {eq}`eq:Psyn` are evaluated on the actual D0FUS profiles, pedestal included. Only the profile factor $K$ retains Albajar’s parabolic parameterisation with the core exponents, an approximation when the H-mode pedestal profiles of Eq. {eq}`eq:profile_core` are used.

Two features of Eq. {eq}`eq:Psyn` are worth highlighting. First, the wall-reflectivity prefactor $(1 - R_w)^{0.62}$ is monotonically decreasing in $R_w$, as physically required: a more reflective wall returns more of the emitted photons to the plasma where they are reabsorbed, reducing the net loss. Second, the temperature dependence $T_0\, (16 + T_0)^{2.61}$ saturates at low temperature and grows steeply above $T_0 \approx 16\,\mathrm{keV}$, which is why synchrotron losses may only become significant for high-temperature and/or high-field designs. In this high-temperature regime the loss scales approximately as $T_0^{3.61}$ (the factor $T_0\,(16+T_0)^{2.61}$ tends to $T_0^{3.61}$ for $T_0 \gg 16\,\mathrm{keV}$), remarkably close to the $T_0^4$ dependence of black-body thermal radiation predicted by the Stefan-Boltzmann law.

### Impurity line radiation

The third radiation channel comes from partially ionised impurity ions, which lose energy through line excitation, radiative recombination, dielectronic recombination and free-free emission. All these contributions are bundled into the radiative cooling coefficient $L_z(T_e)$, with units of $\mathrm{W\,m^3}$. The local emissivity per impurity species is

$$
p_\mathrm{line}(\rho) = n_e^2(\rho)\, f_j\, L_{z,j}\!\bigl(T_e(\rho)\bigr)
$$ (eq:pline_local)

with $f_j = n_j/n_e$ the species concentration relative to electrons. The total power radiated by species $j$ follows from the volume integral

$$
P_{\mathrm{line},j} = f_j \int_0^1 n_e^2(\rho)\, L_{z,j}\!\bigl(T_e(\rho)\bigr)\, V'(\rho)\, \mathrm{d}\rho
$$ (eq:Pline_integral)

and the total impurity radiation is the sum $P_\mathrm{line} = \sum_j P_{\mathrm{line},j}$ over the user-prescribed species list. Multiple impurity species can be combined freely.

D0FUS uses the polynomial fits of Mavrin {footcite:p}`mavrin2018improved` to evaluate $L_z(T_e)$, validated against the ADAS atomic database {footcite:p}`summers1994adas`. The fits cover eleven species and are piecewise in $\log T_e$, with the segmentation listed in {numref}`Table %s <tab:Lz_species>`. To avoid recomputing the polynomials at every solver iteration, a log-log lookup table is precomputed at module load, sampling each species on a uniform grid covering $[0.01, 100]\,\mathrm{keV}$, accurate to better than $0.5\,\%$ relative to the original polynomials.

:::{table} Impurity species supported by D0FUS through the Mavrin {footcite:p}`mavrin2018improved` radiative-cooling fits. The peak temperature $T_\mathrm{peak}$ indicates where $L_z$ reaches its maximum value (qualitative).
:name: tab:Lz_species
:align: center

| Species | $Z$ | $T_\mathrm{peak}$ region             |
|:--------|:----|:-------------------------------------|
| He      | 2   | monotonic decrease                   |
| Li      | 3   | monotonic decrease                   |
| Be      | 4   | monotonic decrease                   |
| C       | 6   | SOL ($\sim 0.05\,\mathrm{keV}$)      |
| N       | 7   | SOL/edge ($\sim 0.1\,\mathrm{keV}$)  |
| O       | 8   | edge ($\sim 0.2\,\mathrm{keV}$)      |
| Ne      | 10  | edge ($\sim 0.5\,\mathrm{keV}$)      |
| Ar      | 18  | edge ($\sim 1\,\mathrm{keV}$)        |
| Kr      | 36  | edge ($\sim 1$ to $2\,\mathrm{keV}$) |
| Xe      | 54  | edge ($\sim 2\,\mathrm{keV}$)        |
| W       | 74  | edge ($\sim 1\,\mathrm{keV}$)        |
:::

```{rubric} References
```

```{footbibliography}
```
