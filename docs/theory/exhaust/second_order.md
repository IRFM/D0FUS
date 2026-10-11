(ssec:chap1_second_order)=

# Second-order operability limits

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.4.2, which describes D0FUS v2.7.0, and updated to the code of October 2026. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


Three further indicators qualify the operability of the converged point without constituting first-order vetoes at the system-code level.

## L-H power threshold

Access to the H-mode confinement regime requires the power crossing the separatrix to exceed an empirical threshold $P_{\text{L-H}}$ (the notation avoids confusion with the lower hybrid power $P_\mathrm{LH}$ of Section {ref}`Plasma current and scaling law <ssec:chap1_heating_cd>`) that depends on the magnetic field, the density, and the plasma surface. D0FUS supports three scalings. The default is the metal-wall scaling of the ITPA TC-26 group, published by Delabie *et al.* {footcite:p}`delabie2026metalwall`,

$$
P_{\text{L-H}} = 0.0441\, S\, B_0^{0.580}\, \bar{n}_\mathrm{line}^{1.08} \left(\frac{2}{M}\right)^{0.975} D
$$ (eq:PLH_TC26)

where $S$ is the plasma surface area in m², $\bar{n}_\mathrm{line}$ is in $10^{20}\,\mathrm{m}^{-3}$ and $M$ is the ion mass in atomic mass units. It is the baseline scaling of the paper, with a root-mean-square error of 0.24. The factor $D$ accounts for the divertor configuration. $D = 1$ holds for a divertor with horizontal-like targets, and the report also gives the threshold at $D = 1.93$, the upper bound fitted on vertical-target configurations. The second option of the same paper replaces the toroidal field by the separatrix poloidal field, $P_{\text{L-H}} = 0.166\, S\, B_\mathrm{pol}^{0.631}\, \bar{n}_\mathrm{line}^{1.07}\, (2/M)^{1.01}\, D$ with $B_\mathrm{pol} = 2\pi\mu_0 I_p R_0/S$. It has the lowest error, 0.19, but the authors do not recommend it for extrapolation, since the current and surface of ITER lie outside the database. Both TC-26 scalings hold on the high-density branch only, above the density of minimum threshold. The third option is the Martin (2008) scaling {footcite:p}`martin2008power`, $P_{\text{L-H}} \propto \bar{n}_\mathrm{line}^{0.72}\, B_0^{0.80}\, S^{0.94}$, adopted by the ITER Physics Basis, with a residual scatter of about $30\,\%$ on the original database. All three scalings use the line-averaged density rather than the volume-averaged value, consistently with the experimental diagnostic from which the databases were assembled. The surface $S$ is the plasma surface of Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`.

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
