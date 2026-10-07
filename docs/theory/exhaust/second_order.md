(ssec:chap1_second_order)=

# Second-order operability limits

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.4.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


Three further indicators qualify the operability of the converged point without constituting first-order vetoes at the system-code level.

## L-H power threshold

Access to the H-mode confinement regime requires the power crossing the separatrix to exceed an empirical threshold $P_{\text{L-H}}$ (the notation avoids confusion with the lower hybrid power $P_\mathrm{LH}$ of Section {ref}`Plasma current and scaling law <ssec:chap1_heating_cd>`) that depends on the magnetic field, the density, and the plasma surface. D0FUS supports three scalings. The default is the Martin (2008) multi-machine scaling {footcite:p}`martin2008power`, $P_{\text{L-H}} \propto \bar{n}_\mathrm{line}^{0.72}\, B_0^{0.80}\, S^{0.94}$, where $S$ is the plasma surface area. This is the scaling adopted by the ITER Physics Basis and is the reference for system-code design studies, with a residual scatter of about $30\,\%$ on the original database. The two alternatives are due to Delabie {footcite:p}`delabie2017lh` and were derived as part of the ITPA TC-26 effort to refine the L-H prediction on a more recent multi-machine dataset (not yet published). The first variant keeps the explicit surface-area dependence, with the surface exponent fixed to unity and the ion-mass exponent refitted, and reduces the residual scatter to about $26\,\%$. The second keeps the surface-area dependence but replaces the magnetic field with an $I_p/a$ dependence, and reaches the lowest scatter at about $21\,\%$. All three scalings use the line-averaged density rather than the volume-averaged value, consistently with the experimental diagnostic from which the original database was assembled.

The H-mode access margin is quantified by the ratio $P_\mathrm{sep}/P_{\text{L-H}}$, which must exceed unity at the design point. In D0FUS this ratio is reported as an output at every design point and drawn as a boundary on the POPCON maps (Section {ref}`Execution modes <sssec:chap1_modes>`), but it is not enforced as a constraint: checking it is left to the user, and turning it into an explicit warning or an optional constraint is left for future work.

## Neutron and radiative wall loads

The radiative load on the first wall is given by $P_\mathrm{rad,total}/S_\mathrm{FW}$ with $S_\mathrm{FW}$ the first-wall surface area, and the neutron wall load by

$$
\Gamma_n = \frac{E_n}{E_\alpha + E_n}\, \frac{P_\mathrm{fus}}{S_\mathrm{FW}} \approx 0.80\, \frac{P_\mathrm{fus}}{S_\mathrm{FW}} \quad [\mathrm{MW/m^2}]
$$ (eq:Gamma_n)

where the prefactor is the neutron energy fraction of the DT reaction (Section {ref}`Fusion power, density and pressure <ssec:chap1_fusion>`). The neutron wall load is the primary driver of structural-material damage through displacement-per-atom (dpa) accumulation, and it sets the blanket and first-wall replacement schedule that enters the capacity factor and the cost model of Section {ref}`Performance: net electric power and cost <ssec:chap1_cost>`. The wall surface defaults to the Ramanujan elliptical approximation in Academic mode and to the Miller-integrated surface in Refined geometry mode (Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`).

## Runaway electron diagnostic

D0FUS includes a post-convergence runaway electron (RE) module, developed by Louis Puel (private communication), that estimates the RE current generated during a disruption mitigated by Massive Material Injection (MMI). The generation is evaluated in two stages. First, the primary generation is assumed to be dominated by the hot-tail mechanism (thus neglecting Dreicer and nuclear mechanisms, which are left for future implementation), occurring during the Thermal Quench (TQ) phase when the plasma rapidly cools. The hot-tail current is estimated using the analytical isotropic Smith model (Eq. 19 in {footcite:p}`smith2008hottail`), based on a set of ad hoc parameters defined by the user according to the considered TQ: the post-dilution electron density, the pre- and post-disruption electron temperatures, and the characteristic TQ duration. Second, the secondary generation via the avalanche mechanism, which occurs during the Current Quench (CQ) as the plasma current decreases, is modelled using Eq. (99) in {footcite:p}`breizman2019physics`. Note that this estimate is a first approximation, and that an accurate modelling of the phenomenon can hardly be reduced to such a simplified approach, given the strong sensitivity of RE generation to disruption parameters. As such, the results should rather be interpreted as “risk scores” for potential RE generation, enabling comparisons between different designs and reflecting the associated challenges in achieving an effective Disruption Mitigation System (DMS).

```{rubric} References
```

```{footbibliography}
```
