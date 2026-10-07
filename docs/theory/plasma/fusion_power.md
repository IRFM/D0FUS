(ssec:chap1_fusion)=

# Fusion power, density and pressure

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.4. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The prescribed fusion power fixes the density the plasma must sustain, which is limited by the Greenwald limit.

In D0FUS, the fusion power $P_\mathrm{fus}$ is a user-prescribed input. The code determines the volume-averaged electron density $\bar{n}_e$ required to produce $P_\mathrm{fus}$ at the prescribed temperature $\bar{T}$, accounting for fuel dilution by helium ash and seeded impurities.

The volumetric power density is governed by the Maxwellian reactivity $\langle\sigma v\rangle(T)$, the thermal average of the product of the fusion cross-section and the relative ion velocity. The reactivity module of D0FUS is organised around a uniform interface so that any fusion reaction can in principle be plugged in (D-T, D-D, D-$^3$He, T-T, p-$^{11}$B...) by substituting the Maxwellian coefficients alone. The default and currently only implemented reaction is D-T, parameterised by the fit of Bosch and Hale {footcite:p}`bosch1992improved`. The fit is valid for ion temperatures between 0.2 and 100 keV with a maximum deviation below 0.35 %. As shown in {numref}`Fig. %s <fig:DT_reactivity>`, $\langle\sigma v\rangle$ rises steeply with $T$ in the power plant operating window (10 to 25 keV) and reaches its maximum around 65 keV. This steep dependence explains the strong sensitivity of $P_\mathrm{fus}$ to the on-axis temperature.

In a real tokamak, however, the temperature cannot be chosen so as to maximise the reactivity alone: as introduced in Section {ref}`Stability limits <ssec:chap1_stability>`, the plasma pressure is bounded from above by the Troyon limit. A device pushed against that limit operates at fixed pressure, so any increase in temperature must be compensated by a proportional decrease in density, $n \propto 1/T$. At fixed pressure, the volumetric fusion power then scales as

$$
p_\mathrm{fus} \propto n^2 \langle\sigma v\rangle = \frac{p^2}{4}\, \frac{\langle\sigma v\rangle}{T^2} \propto \frac{\langle\sigma v\rangle}{T^2}
$$ (eq:pfus_beta_limited)

shown in the right panel of {numref}`Fig. %s <fig:DT_reactivity>`. This figure of merit reaches its maximum around $13.5$ keV, which defines the natural operating temperature of a $\beta$-limited tokamak {footcite:p}`wesson2011tokamaks,freidberg2007plasma,freidberg2015designing`.

:::{figure} /figures/thesis/DT_reactivity.png
:name: fig:DT_reactivity
:width: 100%
:align: center

D-T Maxwellian reactivity $\langle\sigma v\rangle(T)$ from the Bosch-Hale fit {footcite:p}`bosch1992improved` (left) and the corresponding pressure-limited figure of merit $\langle\sigma v\rangle/T^2$ (right). The shaded band marks the 10 to 25 keV operating window.
:::

The total fusion power is the volume integral of the local power density:

$$
P_\mathrm{fus} = \int_0^1 \frac{1}{4}\, n_\mathrm{fuel}^2(\rho)\,
	\langle\sigma v\rangle\!\bigl(T(\rho)\bigr)\,
	(E_\alpha + E_n)\, V'(\rho)\, \mathrm{d}\rho
$$ (eq:Pfus_integral)

where $E_\alpha = 3.5168\,\mathrm{MeV}$ and $E_n = 14.0671\,\mathrm{MeV}$ are the CODATA alpha and neutron birth energies, $n_\mathrm{fuel} = n_D + n_T$ is the total fuel ion density (assumed equally split: $n_D = n_T = n_\mathrm{fuel}/2$, whence the $1/4$ prefactor) and the volume element $V'(\rho)$ follows either Eq. {eq}`eq:Vprime_acad` in Academic mode or the precomputed Miller Jacobian Eq. {eq}`eq:Vprime_miller` in Refined mode (Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`). The integrand is numerically evaluated on the normalised density profile $\hat{n}(\rho) = n(\rho)/\bar{n}$.

Since $P_\mathrm{fus}$ is prescribed, Eq. {eq}`eq:Pfus_integral` is inverted for the fuel density,

$$
\bar{n}_\mathrm{fuel} = 2\sqrt{\frac{P_\mathrm{fus}}{I_\mathrm{fus}\,
			(E_\alpha + E_n)\, V}}
$$ (eq:nfuel_inversion)

with $I_\mathrm{fus} = \int_0^1 \langle\sigma v\rangle\bigl(T(\rho)\bigr)\, \hat{n}^2(\rho)\, w(\rho)\, \mathrm{d}\rho$ the normalised reactivity integral and $w(\rho)$ the volume weight of Eq. {eq}`eq:w_rho`. Quasi-neutrality then closes the system:

$$
\bar{n}_e = \frac{\bar{n}_\mathrm{fuel}}{1 - 2\, f_\alpha - f_\mathrm{imp}}
$$ (eq:ne_from_nfuel)

where the denominator is the dimensionless fuel dilution factor. Helium ash contributes two electrons per alpha through $f_\alpha = n_\alpha/n_e$, and seeded or intrinsic impurities contribute $f_\mathrm{imp} = \sum_j Z_j n_j / n_e$, the two corrections being independent.

## Helium ash and fuel dilution

In a burning plasma, the alpha particles produced by fusion thermalise on the bulk and accumulate as helium ash until transport and divertor pumping remove them. This matters for the design point because every ash nucleus displaces fuel at fixed electron density: through the quasi-neutrality relation of Eq. {eq}`eq:ne_from_nfuel`, dilution raises the electron density required to hold the prescribed fusion power, and with it the pressure that the Troyon limit will judge. D0FUS models the ash balance with a simple reservoir model (following Ref. {footcite:p}`sarazin2020scaling`), in which the alpha production rate, proportional to $n^2 \langle\sigma v\rangle$, is balanced by a removal rate $n_\alpha/\tau_\alpha^*$, where $\tau_\alpha^* = C_\alpha\, \tau_E$ is the effective alpha confinement time. The single dimensionless parameter $C_\alpha$ collapses the combined effect of helium transport in the bulk and wall recycling. The resulting equilibrium ash fraction admits a closed form, derived in {ref}`Helium ash balance <app:helium_ash>`. $C_\alpha$ is not measured but calibrated, so that the predicted ash fraction matches the one the reference design projects: on the ITER deck, the one case where the range to expect is known with confidence, the D0FUS default value $C_\alpha = 5.7$ returns $4.3\,\%$, inside the $4$ to $6\,\%$ projected for the $Q = 10$ inductive scenario {footcite:p}`shimada2007progress`. This value is also consistent with the global burn criterion of Reiter et al. {footcite:p}`reiter1990burn`, which requires $\tau_\alpha^*/\tau_E$ to remain below about 10 to 15 for a stationary D-T burn, helium concentrations of 5 to 10 % being considered tolerable.

Beyond helium ash, intrinsic and seeded impurities further dilute the fuel through the charge-weighted factor $f_\mathrm{imp} = \sum_j Z_j\, f_j$ entering Eq. {eq}`eq:ne_from_nfuel`, with $f_j = n_j / n_e$ the species concentration and $Z_j$ its mean charge state in coronal equilibrium.[^1] Eleven impurity species are supported (He, Li, Be, C, N, O, Ne, Ar, Kr, Xe and W). For each of them, the coronal-equilibrium mean charge state $Z_j(T_e)$ and the radiative cooling coefficient $L_z(T_e)$ are evaluated from the polynomial fits of Mavrin {footcite:p}`mavrin2018improved`. The latter also feed the radiation calculation of Section {ref}`Radiation losses <ssec:chap1_radiation>`.

The same impurity inventory that sets the dilution also fixes the effective charge of the plasma. By default D0FUS reconstructs $Z_\mathrm{eff}$ self-consistently from the prescribed concentrations rather than treating it as a free input,

$$
Z_\mathrm{eff} = 1 + 2\, f_\alpha - \sum_j \langle Z_j\rangle\, f_j + \sum_j \langle Z_j\rangle^2\, f_j
$$ (eq:Zeff)

The squared charge reflects the physics being averaged: the electron-ion Coulomb collision rate scales as $Z^2$, so $Z_\mathrm{eff}$ is the charge of the fictitious pure plasma with the same collisionality. The derivation of Eq. {eq}`eq:Zeff` from this definition and quasi-neutrality is given in {ref}`Radiation models <app:radiation_models>`. This single number propagates consistently into the Bremsstrahlung emissivity, the neoclassical resistivity and the current-drive efficiencies, closing the loop between the assumed impurity content and its radiative and resistive consequences. A user-prescribed $Z_\mathrm{eff}$ remains available as an override.

The reactivity and the pressure, finally, are governed by the ion temperature, which need not equal the electron temperature. D0FUS carries an ion-to-electron temperature ratio $\tau_{ie} = T_i/T_e$, prescribed as a profile-independent constant ($\tau_{ie} = 1$ by default, the single-temperature plasma). It enters in two places: the D-T reactivity is evaluated at $T_i = \tau_{ie}\, T_e$ rather than at $T_e$, and the total pressure becomes $\bar{p} = (1 + \tau_{ie})\, \langle n_e T_e\rangle$, the single-temperature prefactor $2$ generalising to $(1 + \tau_{ie})$ so that $\tau_{ie} = 1$ recovers $\bar{p} = 2\langle n_e T_e\rangle$.

## Volume-averaged pressure and thermal energy

The volume-averaged total plasma pressure, with equal ion and electron temperatures, reads

$$
\bar{p} = 2\, \langle n\, T \rangle_\mathrm{vol} \quad [\mathrm{Pa}]
$$ (eq:pbar)

where the factor 2 accounts for the equal contributions of the ion and electron pressures in the single-temperature case, with $n$ in $\mathrm{m^{-3}}$ and $T$ expressed in energy units (the Boltzmann constant is absorbed into $T$). When $T_i \neq T_e$, this prefactor generalises to $(1 + \tau_{ie})$ as introduced above. All along the chain, $n_i \simeq n_e$ is assumed. This slightly overestimates the ion pressure, since dilution makes the true ion count lower than $n_e$ (by around 5 % at ITER-like helium and impurity content, hence a few percent on the total pressure). The correction is left for future work, the fast-alpha pressure being in any case counted separately (Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>`).

The volume average uses the same weight function $w(\rho)$ as the profiles introduced in Section {ref}`Radial profiles <ssec:chap1_profiles>`: cylindrical $2\rho$ in Academic mode, Miller $V'(\rho)/V$ in Refined mode. In the purely parabolic Academic limit (Eq. {eq}`eq:parabolic_ansatz`), and with $\hat{X} = X/\bar{X}$ denoting the profiles normalised to their volume average (Section {ref}`Radial profiles <ssec:chap1_profiles>`), the integral admits the closed form,

$$
\langle\hat{n}\hat{T}\rangle_\mathrm{vol} = \frac{(1+\nu_n)(1+\nu_T)}{1+\nu_n+\nu_T}
$$

which D0FUS uses directly as a fast path. For pedestal profiles or Refined geometry, the integral is evaluated numerically. The total thermal energy stored in the plasma follows as:

$$
W_\mathrm{th} = \frac{3}{2}\, \bar{p}\, V \quad [\mathrm{J}]
$$ (eq:Wth)

The density and the pressure demanded by the prescribed fusion power are now known. Neither can be compared to its respective stability limit yet. Both the Greenwald limit and the Troyon limit indeed scale with the still-unknown plasma current, and the Troyon limit furthermore scales with the magnetic field.

[^1]: The ionisation balance of a stationary, optically thin plasma, in which electron-impact ionisation balances recombination: the charge-state distribution then depends on the local electron temperature only. The assumption holds in the hot core but fails near the edge, where the residence time is short. The non-coronal cooling curves of Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>` are the correction to it.

```{rubric} References
```

```{footbibliography}
```
