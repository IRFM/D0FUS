(ssec:chap1_solver)=

# Solver

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2.8. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The chain just described cannot be evaluated in a single pass, because several of its quantities are mutually coupled: the helium ash fraction depends on the confinement time, which depends on the density and the current, which depend on the ash fraction through the fuel dilution. A convergence loop is therefore unavoidable, and is kept as lightweight as possible (no heavy numerical libraries, and convergence loops only where they are necessary). Starting from an initial guess for the helium ash fraction $f_\alpha$ (and the auxiliary power $P_\mathrm{aux}$ in Steady-State operation), the solver iterates the physics chain shown in {numref}`Fig. %s <fig:physics_chain>` until self-consistency is reached.

:::{figure} /figures/thesis/physics_chain_flowchart_parallel.png
:name: fig:physics_chain
:width: 100%
:align: center

Self-consistent solver architecture. The physics chain (teal boxes) is iterated until convergence on $f_\alpha$ (pulsed) or $(f_\alpha, Q)$ (steady-state). Branches drawn side by side are independent of one another and are evaluated in any order.
:::

The power balance is what makes the D0FUS solver a fixed-point problem rather than a sequence of independent calculations. At each iteration of the loop, the balance of Eq. {eq}`eq:power_balance_internal` fixes $\tau_E$, which through Eq. {eq}`eq:scaling_law` fixes the plasma current $I_p$, which through the bootstrap and current-drive models of Section {ref}`Plasma current and scaling law <ssec:chap1_currents>` fixes the inductive current $I_\Omega$ and hence $P_\Omega$, which feeds back into the balance. In pulsed operation, the user fixes $P_\mathrm{aux}$ and the solver converges $f_\alpha$ through a one-dimensional root finding, inside which an inner loop ensures the consistency of $P_\Omega$ (two nested loop levels in total). In steady-state operation, $P_\Omega = 0$ by construction and the auxiliary power becomes $P_\mathrm{aux} = P_\mathrm{fus}/Q$: the gain factor $Q$ becomes an a priori undetermined quantity, and the solver converges the pair $(f_\alpha, Q)$ simultaneously.

Note that the ash fraction, although it depends on $\tau_E$ alone through $\tau_\alpha^* = C_\alpha\, \tau_E$, is only updated at the end of each iteration of the outer loop: the balance that returns $\tau_E$ contains the ohmic power, which is known once the current has been split into its bootstrap, driven and inductive shares.

The solver architecture follows: a one-dimensional root-finding in pulsed operation, a two-dimensional solve in steady state. The corresponding numerical strategies (methods, initial guesses and fallbacks) are detailed in {ref}`Execution-mode input syntax and the genetic optimiser <app:d0fus_modes_syntax>`. In practice, convergence is reached for the vast majority of design points in a few iterations. When non-convergence occurs, it is usually due to a physical rather than numerical reason: for example, a too large $C_\alpha$ (insufficient helium pumping) makes the ash fraction and the required electron density rise together until radiation overwhelms the heating and no stationary burn exists at the requested fusion power.

Once the plasma state is converged, a series of post-convergence calculations complete the design evaluation: the TF coil mechanical sizing and the flux budget of the central solenoid (Section {ref}`Radial build limits <sec:chap1_magnets>`), then the heat-exhaust proxies, the operability indicators and the performance figures of merit (Section {ref}`Heat exhaust, second-order limits and performance <sec:chap1_diagnostics>`).
