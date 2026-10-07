(ssec:chap1_power_balance)=
(ssec:chap1_confinement)=

# Power balance and energy confinement time

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.6. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The energy confinement time the plasma must achieve follows from the power balance between the heating and loss channels.

The steady-state thermal energy balance of the confined plasma reads

$$
P_\alpha + P_\mathrm{aux} + P_\Omega = P_\mathrm{rad,core} + P_\mathrm{loss}
$$ (eq:power_balance_internal)

where the left-hand side gathers the heating channels, $P_\mathrm{rad,core}$ is the power radiated from the confined core (introduced below), and $P_\mathrm{loss}$ is the power lost through turbulent transport across the confined region.

The energy confinement time is defined as the ratio of the thermal stored energy to this net heating power crossing the confined region,

$$
\tau_E = \frac{W_\mathrm{th}}{P_\mathrm{loss}}
$$ (eq:tauE_def)

where $W_\mathrm{th} = (3/2)\, \bar{p}\, V$ is the volume-integrated thermal energy already introduced in Section {ref}`Fusion power, density and pressure <ssec:chap1_fusion>` and, rearranging the balance above,

$$
P_\mathrm{loss} = P_\alpha + P_\mathrm{aux} + P_\Omega - P_\mathrm{rad,core}
$$ (eq:Ploss)

Subtracting the core radiated power is a convention as much as a physical statement: this is the convention used to derive the empirical $\tau_E$ scalings presented below, and the share of the heating power radiated from the confined region, modest in those discharges, is far larger in a reactor-grade, impurity-seeded plasma. It is also physically reasonable, since $\tau_E$ is meant to characterise the turbulent transport across the confined region, from which the power radiated in the core must be deducted. The power radiated near the edge, by contrast, barely affects that transport, which is why it is not subtracted. The core/edge split introduced below is the lever through which this choice can be moved.

The auxiliary power $P_\mathrm{aux}$ is a user input in pulsed operation. In steady-state operation it is instead one of the unknowns of the solver, set by the current-drive requirement (Section {ref}`Solver <ssec:chap1_solver>`).

The alpha heating power follows from the partitioning of the DT reaction energy between the alpha particle ($E_\alpha = 3.5168\,\mathrm{MeV}$, charged, confined) and the neutron ($E_n = 14.0671\,\mathrm{MeV}$, escaping to the blanket),

$$
P_\alpha = \frac{E_\alpha}{E_\alpha + E_n}\, P_\mathrm{fus} \approx 0.2\, P_\mathrm{fus}
$$ (eq:Palpha)

implemented with the exact CODATA energies of Eq. {eq}`eq:Pfus_integral`.

The ohmic term is the Joule dissipation of the inductive current, $P_\Omega = \mathcal{R}_p\, I_\Omega^2$. The effective plasma resistance $\mathcal{R}_p$ is obtained by integrating the conductivity profile over the flux surfaces: in flat-top the toroidal electric field scales as $1/R$, so the ohmic current density follows $j_\Omega(\rho) \propto \sigma(\rho)/R$, and the volume integration with the $V'(\rho)$ weight of Section {ref}`Flux-surface geometry <ssec:chap1_geometry>` yields $\mathcal{R}_p$ ({ref}`Bootstrap current, resistivity and safety-factor profile <app:currents_models>`). The conductivity is the inverse of the resistivity model selected in Section {ref}`Plasma current and scaling law <ssec:chap1_currents>` (classical Spitzer, or neoclassical Sauter or Redl). Note that this term is small at power plant-grade temperatures: at the ITER reference point it amounts to less than one percent of $P_\alpha + P_\mathrm{aux}$, the loop voltage of $51$ mV over a plasma current of $15.7$ MA giving about $0.8$ MW against the $150$ MW of alpha and auxiliary power combined.

(ssec:chap1_radiation)=

## Radiation losses

The last term corresponds to the radiated power, a non-negligible loss channel of the plasma. These radiated photons carry a heat flux deposited over the whole first wall rather than focused on the divertor targets. Thus, two distinct facets of this loss matter for the design point: the radiation emitted from the confined core reduces the heating power available to drive transport, and is the term subtracted from the loss power in Eq. {eq}`eq:Ploss` above, while the radiation emitted at the plasma edge lowers the divertor heat load. D0FUS computes three independent radiation mechanisms (Bremsstrahlung, synchrotron, and impurity line radiation) and applies the corresponding core/edge split at a user-prescribed normalised radius.

The three mechanisms differ by what accelerates the electrons, which dominate all three channels by virtue of their small mass. Bremsstrahlung comes from their Coulomb deflection, mainly in the field of the ions, so it grows with the ion charge and with the product of the colliding densities. Synchrotron emission comes from the cyclotron motion itself, so it is governed by the magnetic field and the electron energy rather than by collisions. Line radiation, finally, requires bound electrons: a light impurity stops radiating once the core temperature strips it completely, whereas a heavy one, never fully stripped, keeps radiating everywhere.

The complete formulations of these three mechanisms are given in {ref}`Radiation models <app:radiation_models>`. Bremsstrahlung, scaling as $n_e^2\, Z_\mathrm{eff}\, T_e^{1/2}$ {footcite:p}`wesson2011tokamaks` peaks on axis where $n_e^2$ is largest, and D0FUS treats it as a purely core loss. Synchrotron emission is partially reabsorbed by the plasma at the lower harmonics and reflected by the metallic first wall, so its net loss depends on the wall reflectivity, the plasma opacity and the on-axis temperature in a way that no simple scaling can capture. D0FUS therefore uses the Albajar parameterisation corrected by the Fidone wall-reflectivity treatment {footcite:p}`albajar2001synchrotron,fidone2001synchrotron`. With a steep temperature dependence, this channel only becomes significant for high-temperature operation points. Impurity line radiation, finally, is the channel a designer actually controls: partially ionised impurities radiate through the cooling coefficient $L_z(T_e)$ shown in {numref}`Fig. %s <fig:chap1_Lz_cooling>`, evaluated from the polynomial fits of Mavrin {footcite:p}`mavrin2018improved` to the ADAS atomic database {footcite:p}`summers1994adas` for eleven species, so that intrinsic contamination and deliberate seeding, the main lever on the exhaust problem of Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>`, can be combined freely. Physically, $L_z(T_e)$ is proportional to the radiated power per electron and per impurity ion: at each temperature it sums the line emission of the charge states present in coronal equilibrium, which is why it peaks where partially ionised states abound and drops once the species is fully stripped.

:::{figure} /figures/thesis/d0fus_Lz_cooling.png
:name: fig:chap1_Lz_cooling
:width: 100%
:align: center

Coronal radiative cooling coefficient $L_z(T_e)$ for six representative impurity species (W, Kr, Ar, Ne, N, C) selected among the eleven supported by D0FUS, evaluated from the Mavrin {footcite:p}`mavrin2018improved` polynomial fits to the ADAS database. The three coloured bands in the background indicate the typical edge, pedestal, and core temperature ranges.
:::

Bremsstrahlung and synchrotron emissions are assumed to originate entirely from the core region in D0FUS, and no edge contribution is considered for these channels. Impurity line radiation is instead split into a core contribution emitted at $\rho < \rho_\mathrm{rad,core}$ and an edge contribution emitted at $\rho \ge \rho_\mathrm{rad,core}$, with two canonical presets for $\rho_\mathrm{rad,core}$: $0.6$, aligned with the PROCESS {footcite:p}`kovari2014process` core-region radius, and $1$, the conservative and default choice that therefore subtracts all the radiated power from $P_\mathrm{loss}$. The choice is not cosmetic: at the ITER reference point, moving the boundary from $0.6$ to $1.0$ raises the required $I_p$ by about $10\,\%$. This treatment remains a simplified approximation {footcite:p}`lux2016radiation`. In practice, a more realistic description would require impurity profile modelling, or even a dedicated scrape-off-layer (SOL) radiation model, in order to capture the spatial distribution of radiative losses more accurately. But such a level of detail is beyond the scope of the present system-level description.

## Closing the balance

At convergence every term of Eq. {eq}`eq:power_balance_internal` is known, so the balance yields the required confinement time exactly.

One global figure of merit closes the balance. The fusion gain factor quantifies the amplification effect provided by the plasma,

$$
Q = \frac{P_\mathrm{fus}}{P_\mathrm{aux} + P_\Omega}
$$ (eq:Q)

and is the figure of merit usually quoted to characterise a tokamak operating point.[^1] The two modes of operation opposed throughout these pages deserve a precise definition at this point. In pulsed operation, part of the plasma current is driven inductively by the central solenoid: the discharge lasts only as long as the solenoid flux swing allows, and the plant delivers its power in pulses separated by the dwell time needed to re-magnetise the solenoid. In steady-state operation, the current is sustained entirely by the bootstrap and externally driven contributions, so that the flux consumption is no longer that of the flat-top but only that of the transients, plasma initiation and current ramp-up, and the duration of the discharge is set by the technology rather than by the flux budget. In pulsed operation, $P_\mathrm{aux}$ is a user input and $Q$ is a derived output of the converged solver state. In steady-state operation, the inductive component vanishes ($I_\Omega = 0$, $P_\Omega = 0$) and the auxiliary power is itself constrained by the current-drive requirement. The gain then becomes a free variable (Section {ref}`Solver <ssec:chap1_solver>`).

Now that the required $\tau_E$ has been expressed through the power balance, one can deduce how much plasma current is required to obtain this confinement time.

[^1]: As a reminder, ITER targets $Q = 10$ in its baseline 15 MA inductive scenario, whereas DEMO-class machines typically aim for $Q \gtrsim 30$.

```{rubric} References
```

```{footbibliography}
```
