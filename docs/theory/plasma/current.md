(ssec:chap1_currents)=
(ssec:chap1_heating_cd)=

# Plasma current and scaling law

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.7. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The confinement quality a tokamak can deliver can be estimated through empirical scaling laws fitted on multi-machine databases. All scaling laws supported by D0FUS share a common multiplicative form, conventionally written in the ITER Engineering Parameters notation as

$$
\tau_E = H\, C_\mathrm{SL}\, R_0^{\alpha_R}\, \epsilon^{\alpha_\epsilon}\, \kappa_x^{\alpha_\kappa}\, (1+\delta)^{\alpha_\delta}\, (10\, \bar{n}_\mathrm{line})^{\alpha_n}\, B_0^{\alpha_B}\, M^{\alpha_M}\, P_\mathrm{loss}^{\alpha_P}\, I_p^{\alpha_I}
$$ (eq:scaling_law)

where $\epsilon = a/R_0$ is the inverse aspect ratio, $\kappa_x$ the cross-section elongation of Eq. {eq}`eq:kappa_x`, $M$ the effective ion mass in atomic mass units (2.5 for a 50/50 DT plasma), $H$ the dimensionless confinement enhancement factor, $\bar{n}_\mathrm{line}$ the line-averaged electron density of Section {ref}`Radial profiles <ssec:chap1_profiles>` in $10^{20}\,\mathrm{m}^{-3}$ (the factor 10 converts the density to the $10^{19}\,\mathrm{m}^{-3}$ units used in the original fit), $I_p$ the plasma current in MA, and $P_\mathrm{loss}$ the loss power of Eq. {eq}`eq:Ploss`, in MW. The prefactor $C_\mathrm{SL}$ and the nine exponents $\alpha_i$ are stored in an internal registry: {numref}`Table %s <tab:scaling_laws>` lists the values for the different scaling laws presently available in D0FUS. The default is the H-mode scaling ITPA20 {footcite:p}`verdoolaege2021itpa`.

:::{table} Confinement scaling laws available in D0FUS, with the prefactor $C_\mathrm{SL}$ and the dimensional exponents in the convention of Eq. {eq}`eq:scaling_law`. All scaling laws use the cross-section elongation $\kappa_x = S_0/(\pi a^2)$ and the line-averaged density $\bar{n}_\mathrm{line}$. References: IPB98(y,2) {footcite:p}`iterphysicsbasis1999`, ITPA20 and ITPA20-IL {footcite:p}`verdoolaege2021itpa`, DS03 {footcite:p}`doyle2007plasma`, L-mode {footcite:p}`kaye1997iter`, ITER89-P {footcite:p}`yushmanov1990scalings`.
:name: tab:scaling_laws
:align: center

| Scaling law | $C_\mathrm{SL}$ | $\alpha_\delta$ | $\alpha_M$ | $\alpha_\kappa$ | $\alpha_\epsilon$ | $\alpha_R$ | $\alpha_B$ | $\alpha_n$ | $\alpha_I$ | $\alpha_P$ |
|:------------|:----------------|:----------------|:-----------|:----------------|:------------------|:-----------|:-----------|:-----------|:-----------|:-----------|
| IPB98(y,2)  | 0.0562          | 0               | 0.19       | 0.78            | 0.58              | 1.97       | 0.15       | 0.41       | 0.93       | $-0.69$    |
| ITPA20      | 0.053           | 0.36            | 0.20       | 0.80            | 0.35              | 1.71       | 0.22       | 0.24       | 0.98       | $-0.669$   |
| ITPA20-IL   | 0.067           | 0.56            | 0.30       | 0.67            | 0                 | 1.19       | $-0.13$    | 0.147      | 1.29       | $-0.644$   |
| DS03        | 0.028           | 0               | 0.14       | 0.75            | 0.30              | 2.11       | 0.07       | 0.49       | 0.83       | $-0.55$    |
| L-mode      | 0.023           | 0               | 0.20       | 0.64            | $-0.06$           | 1.83       | 0.03       | 0.40       | 0.96       | $-0.73$    |
| ITER89-P    | 0.0381          | 0               | 0.50       | 0.50            | 0.30              | 1.50       | 0.20       | 0.10       | 0.85       | $-0.50$    |
:::

Two conventions matter when evaluating these fits for a design point. The first concerns the elongation: the one that enters Eq. {eq}`eq:scaling_law` is not the LCFS elongation $\kappa_\mathrm{LCFS}$ used in the geometric module of Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`, but the cross-section elongation

$$
\kappa_x = \frac{S_0}{\pi a^2}
$$ (eq:kappa_x)

where $S_0$ is the area of the poloidal cross-section bounded by the LCFS. For an unshaped ellipse with $\delta = 0$, the three quantities collapse to the same value $\kappa_\mathrm{LCFS}$. Positive triangularity, however, removes area from the cross-section and reduces $\kappa_x$ below $\kappa_\mathrm{LCFS}$. For ITER ($\kappa_\mathrm{LCFS} = 1.85$, $\delta = 0.48$, $S_0 = 22\,\mathrm{m}^2$, $a = 2\,\mathrm{m}$), Eq. {eq}`eq:kappa_x` gives $\kappa_x \approx 1.75$ {footcite:p}`shimada2007progress`. Passing $\kappa_\mathrm{LCFS}$ in place of $\kappa_x$ to a scaling law with $\alpha_\kappa = 0.78$ (IPB98) overestimates $\tau_E$ by about $(1.85/1.75)^{0.78} \approx 4.6\,\%$, which propagates into the inverted $I_p$ as an underestimate of comparable magnitude. The cross-section elongation is computed from the prescribed shape, with $S_0$ obtained analytically in Academic mode and from the Miller contour integral in Refined mode (Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`).

The second convention concerns the density: the one appearing in Eq. {eq}`eq:scaling_law` is the line-averaged electron density $\bar{n}_\mathrm{line}$, not the volume-averaged $\bar{n}_\mathrm{vol}$. This convention reflects the experimental practice of measuring densities by interferometry along a horizontal chord through the plasma centre, which is what the databases underlying IPB98(y,2) and ITPA20 record. For peaked profiles the two quantities can differ by up to 30 % (Section {ref}`Radial profiles <ssec:chap1_profiles>`), so the conversion matters.

One dimensionless degree of freedom remains. The multiplier $H$ in Eq. {eq}`eq:scaling_law` captures the deviation of the actual confinement quality from the scaling-law prediction. By construction $H = 1$ corresponds to the nominal scaling. ITER is designed at $H_{98} = 1.0$, i.e. at the IPB98(y,2) prediction. EU-DEMO design studies typically assume a slightly optimistic $H_{98} = 1.1$, and ARC V1 {footcite:p}`sorbom2015arc` goes further, adopting $H_{98} = 1.8$ at the design point. Values of this order have already been reached, for example in DIII-D plasmas {footcite:p}`petty2017hybrid`, but they remain confined to narrow operational windows. Their extrapolation to power-plant scale remains, in the author’s view, uncertain, justifying the D0FUS default value of 1.

In the D0FUS solver the plasma current is not a free input but a derived quantity. At each iteration of the self-consistency loop (Section {ref}`Solver <ssec:chap1_solver>`), the converged thermal pressure and power balance fix $\tau_E$ through Eq. {eq}`eq:tauE_def`. Eq. {eq}`eq:scaling_law` is then inverted algebraically for $I_p$,

$$
I_p = \left(\frac{\tau_E}{H\, C_\mathrm{SL}\, R_0^{\alpha_R}\, \epsilon^{\alpha_\epsilon}\, \kappa_x^{\alpha_\kappa}\, (1+\delta)^{\alpha_\delta}\, (10\, \bar{n}_\mathrm{line})^{\alpha_n}\, B_0^{\alpha_B}\, M^{\alpha_M}\, P_\mathrm{loss}^{\alpha_P}}\right)^{1/\alpha_I}
$$ (eq:Ip_inversion)

The current being determined, the three viability limits of Section {ref}`Stability limits <ssec:chap1_stability>` can now be computed.

The total current then has to be split into the part the plasma generates by itself, the part driven externally, and, in pulsed operation only, the part the central solenoid supplies inductively (Section {ref}`Power balance and energy confinement time <ssec:chap1_power_balance>`). This split sizes the flux demand of Section {ref}`Radial build limits <sec:chap1_magnets>`.

The plasma current is then decomposed into three contributions:

$$
I_p = I_b + I_\mathrm{CD} + I_\Omega
$$ (eq:Ip_decomposition)

where $I_b$ is the bootstrap current self-generated by the radial pressure gradient, $I_\mathrm{CD}$ is the externally driven non-inductive current detailed below, and $I_\Omega$ is the inductively driven (ohmic) current. In steady-state operation, $I_\Omega = 0$ by construction, so the entire current must be supplied by the combination of bootstrap and external current drive. The total $I_p$ itself is the one just determined by the scaling-law inversion. The paragraphs below describe how its different components are calculated: $I_b$ and $I_\mathrm{CD}$ are computed self-consistently, and the budget is closed by setting $I_\Omega = I_p - I_b - I_\mathrm{CD}$ in pulsed mode (or, equivalently, by solving for the required $I_\mathrm{CD}$ given $I_b$ and $I_\Omega = 0$ in steady-state mode).

The inductive share $I_\Omega$ is what the central solenoid has to drive, and therefore what sizes its flux reserve in Section {ref}`Radial build limits <sec:chap1_magnets>`.

## Bootstrap current

The origin of the bootstrap current is neoclassical, that is, it follows from the toroidal geometry alone. Since the toroidal field varies as $1/R$, a particle moving toward the inboard side sees an increasing field and is magnetically mirrored back if its parallel velocity is small enough: it remains trapped on the low-field side, and its guiding centre describes in the poloidal plane a banana-shaped orbit whose radial width is much larger than the Larmor radius. Consider now two such banana orbits tangent to a given flux surface, one on each side of it. On that surface the two orbits carry particles in opposite parallel directions, but each orbit is populated according to the flux surface it belongs to, so a radial density gradient leaves the inner one more populated than the outer one. The two opposite flows no longer cancel, and a net parallel flow of trapped particles appears. Collisions then transfer this net parallel momentum to the passing particles, which end up carrying most of the resulting current. The current is thus driven by the pressure gradient itself, and requires nothing external to the plasma. Both ingredients of this picture, the banana orbit and the trapped-passing boundary, are illustrated in {numref}`Fig. %s <fig:chap1_banana_cone>`.

:::{figure} /figures/thesis/fig_chap1_banana_cone_composite.png
:name: fig:chap1_banana_cone
:width: 95%
:align: center

The two building blocks of the bootstrap-current picture: the banana orbit (a, b) and the trapped-passing boundary (c). (a) Guiding-centre trajectory of a trapped particle, mirrored back by the stronger inboard field and confined to the low-field side. (b) The same trajectory projected onto the poloidal plane, where it traces the banana-shaped orbit about its flux surface. (c) Cut through velocity space at the outboard midplane: particles inside the two cones circulate freely, those outside are trapped. (a) Banana orbit, 3D view; (b) The same orbit in the poloidal plane; (c) Trapped-passing boundary in velocity space.
:::

Following the two-fidelity philosophy of D0FUS, two bootstrap models are available[^1]. Their complete formulations are given in {ref}`Bootstrap current, resistivity and safety-factor profile <app:currents_models>`. The Academic option is the closed-form model of Segal, Cerfon and Freidberg {footcite:p}`segal2021bootstrap`, which expresses the bootstrap fraction directly from the volume-averaged quantities and the profile peaking exponents: it is fast and integral-free, at the price of ignoring the collisionality dependence of the neoclassical transport coefficients. The Refined option, the default, evaluates the local bootstrap current density from the neoclassical model of Redl et al. {footcite:p}`redl2021bootstrap`, a refit of the Sauter coefficients {footcite:p}`sauter1999neoclassical,sauter2002erratum` against the modern drift-kinetic solver NEO, and integrates it radially on the prescribed profiles.

One refinement matters for the accuracy of the bootstrap term. The safety factor profile enters the bootstrap calculation through the local collisionality $\nu_e^*(\rho) \propto q(\rho)$ and the local trapped fraction: using a flat $q_{95}$ everywhere overestimates the mid-radius collisionality and therefore underestimates the predicted bootstrap current. D0FUS computes $q(\rho)$ either analytically, from a prescribed parametric current profile closed by the cylindrical Ampère relation (Academic mode), or by a damped Picard iteration that reassembles the ohmic, bootstrap and driven current densities at each step until the profile and the current decomposition are mutually consistent (Refined mode). Both branches are detailed in {ref}`Bootstrap current, resistivity and safety-factor profile <app:currents_models>`.

## Auxiliary current

The externally driven current is where the heating and current drive technology enters the design point. In a tokamak power plant, the same external systems typically provide both plasma heating and current drive: the injected power raises the plasma temperature and drives a non-inductive toroidal current that extends (or replaces) the inductively driven current. D0FUS models the current drive efficiency of four standard technologies and provides both a simplified (Academic) and a technology-resolved (Refined, the `Multi` option of the code) mode.

The standard figure of merit for non-inductive current drive is the normalised efficiency {footcite:p}`fisch1987current`:

$$
\gamma_\mathrm{CD} = \frac{\bar{n}_e\, R_0\, I_\mathrm{CD}}{P_\mathrm{aux}} \quad [\mathrm{10^{20}\,A\,W^{-1}\,m^{-2}}]
$$ (eq:gamma_CD)

where $I_\mathrm{CD}$ is the driven current in MA, $P_\mathrm{aux}$ the injected power in MW, $\bar{n}_e$ in $10^{20}\,\mathrm{m}^{-3}$, and $R_0$ in m. The notation $P_\mathrm{aux}$ is kept (rather than a dedicated $P_\mathrm{CD}$) because the same injected power heats the plasma and drives the current.

Two levels of description are available, in line with the two-layer philosophy of the code. In Academic mode, a single technology-agnostic efficiency $\gamma_\mathrm{CD,acad}$ (by default equal to 0.20, a mid-range value typical of the demonstrated ITER-class NBI figure of merit) and a single wall-plug efficiency $\eta_\mathrm{WP,acad}$, the ratio of the plasma-absorbed power to the electric power drawn from the grid (by default equal to 0.40, typical of gyrotron-based systems), are used. This mode is sufficient for exploratory scans where the current drive technology has not yet been selected. The driven current is simply:

$$
I_\mathrm{CD} = \frac{\gamma_\mathrm{CD,acad}\, P_\mathrm{aux}}{\bar{n}_e\, R_0}
$$ (eq:ICD_acad)

In Multi-source mode, the user specifies the power injected by each source independently, for the lower hybrid (LH), electron cyclotron (EC), neutral beam injection (NBI) and ion cyclotron (IC) systems ($P_\mathrm{LH}$, $P_\mathrm{EC}$, $P_\mathrm{NBI}$, $P_\mathrm{IC}$, all in MW). The total auxiliary power is their sum: $P_\mathrm{aux} = P_\mathrm{LH} + P_\mathrm{EC} + P_\mathrm{NBI} + P_\mathrm{IC}$. The total driven current is the sum of the per-source contributions:

$$
I_\mathrm{CD} = \sum_{s \in \{\mathrm{LH, EC, NBI}\}} \frac{\gamma_s\, P_s}{\bar{n}_e\, R_0}
$$ (eq:ICD_multi)

Each source also has its own wall-plug efficiency, so the total wall-plug power is $P_\mathrm{WP} = \sum_s P_s / \eta_{\mathrm{WP},s}$.

Each source carries a physics-based efficiency model, detailed in {ref}`Current drive efficiency models <app:cd_models>`. Lower hybrid current drive follows the default METIS model {footcite:p}`artaud2018metis` and remains the most efficient option per injected watt. Electron cyclotron current drive follows the Giruzzi trapped-electron model with the Lin-Liu $Z_\mathrm{eff}$ correction {footcite:p}`giruzzi1987eccd,linliu2003eccd`, its efficiency depending strongly on the deposition location through trapped-electron effects. Neutral beam current drive is built on the same isotropic slowing-down distribution as the fusion alphas ({ref}`Fast-alpha pressure model <app:fast_alpha_model>`), combining the Cordey velocity-space integral, the Lin-Liu electron return current and an orbit-trapping factor for the beam ions {footcite:p}`cordey1979nbcd,linliu1997trapping`. The assembled chain matches METIS to within $1.5\,\%$ on three machine cases (ITER, ARC and EU-DEMO beams). Ion cyclotron heating is treated as pure heating ($\gamma_\mathrm{IC} = 0$), fast-wave current drive with asymmetric antenna phasing being left to future work.

## Inductive current

The ohmic current $I_\Omega$ dissipates the power $P_\Omega = \mathcal{R}_p\, I_\Omega^2$ and consumes flux through the loop voltage $V_\mathrm{loop} = \mathcal{R}_p\, I_\Omega$, which is why the plasma resistance matters twice: in the power balance, and in the central-solenoid flux budget of Section {ref}`Radial build limits <sec:chap1_magnets>`. The effective resistance $\mathcal{R}_p$ is obtained by treating the flux surfaces as parallel resistors and integrating the local conductivity over the volume with an explicit toroidal correction (Eq. {eq}`eq:Reff` in {ref}`Bootstrap current, resistivity and safety-factor profile <app:currents_models>`). Four resistivity models are supported, from the classical Spitzer expression {footcite:p}`spitzer1953transport` to the neoclassical Sauter and Redl fits {footcite:p}`sauter1999neoclassical,redl2021bootstrap`, the latter being the default: neglecting the neoclassical trapped-particle correction underestimates the resistivity, hence the loop voltage and the flux consumption, which can have substantial consequences on the CS sizing.

An additional moment of the current profile is extracted for later use. The internal inductance quantifies the peaking of the current density profile and enters the magnetic flux balance for CS sizing (Section {ref}`Academic model <ssec:chap1_academic>`). D0FUS uses the ITER/EFIT convention {footcite:p}`luce2014inductance`, the index $(3)$ labelling the third of the normalisations of the internal inductance in use in the literature {footcite:p}`freidberg2007plasma,jackson2008ITER`, the one that normalises the volume-integrated poloidal field energy with the geometric major radius $R_0$, the alternative conventions $l_i(1)$ and $l_i(2)$ differing by the radius entering that normalisation,

$$
l_i(3) = \frac{2}{R_0}\, \int_0^1 \left(\frac{I_\mathrm{enc}(\rho)}{I_p}\right)^2\, \frac{V'(\rho)}{L_\theta^2(\rho)}\, \mathrm{d}\rho
$$ (eq:li3)

where $I_\mathrm{enc}(\rho)$ is the cumulative enclosed current and $L_\theta(\rho)$ is the poloidal arc length of the flux surface at $\rho$. In Academic mode, $L_\theta$ uses the Ramanujan ellipse formula of Eq. {eq}`eq:ramanujan_perimeter` applied at each radial position. In Refined mode, the true Miller arc length precomputed alongside $V'(\rho)$ in Section {ref}`Flux-surface geometry <ssec:chap1_geometry>` is used directly.

Each of the three currents, bootstrap, driven and inductive, is evaluated from a plasma state, the density, temperature and current profiles, that their own sum fixes. None of them can therefore be computed before the other two. The self-consistent solver enforces their consistency, and at the same time deals with the ash-fraction and power couplings met along the chain.

[^1]: The name comes from the English idiom *to pull oneself up by one’s bootstraps*, which evokes the absurd image of lifting oneself by pulling on one’s own boots. The term entered tokamak physics with the diffusion-driven current predicted by Bickerton, Connor and Taylor {footcite:p}`bickerton1971bootstrap`.

```{rubric} References
```

```{footbibliography}
```
