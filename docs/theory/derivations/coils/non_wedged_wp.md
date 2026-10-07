(app:wedg_approx)=

# Non-wedged winding-pack hypothesis

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.12. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The D0FUS Refined model assumes that the toroidal centring load is reacted by the steel nose alone, the WP transmitting only radial stress ($\sigma_\theta^\mathrm{WP} \approx 0$). This page tests that simplification on an ITER-like reference case by sweeping the share of the load reacted by the WP.

Define a wedging fraction $\alpha_\mathrm{WP} \in [0, 1]$ as the share of the tangential reaction moment absorbed by the WP, the remainder being reacted by the nose:

$$
\alpha_\mathrm{WP} \;=\; \frac{\langle \sigma_\theta \rangle_\mathrm{WP} \, \Delta_\mathrm{WP}}{\langle \sigma_\theta \rangle_\mathrm{WP} \, \Delta_\mathrm{WP} + \langle \sigma_\theta \rangle_\mathrm{nose} \, \Delta_\mathrm{nose}}
$$ (eq:alpha_WP_def)

with $\langle \cdot \rangle$ a layer-averaged value. The limit $\alpha_\mathrm{WP} = 0$ corresponds to the D0FUS hypothesis (the nose reacts the entire load) and $\alpha_\mathrm{WP} = 1$ to the opposite limit (the WP reacts the entire load).

The radial elastic problem is solved with the analytical multilayer thick-cylinder solver CIRCE {footcite:p}`boudes2025circe`, which provides closed-form expressions for the displacement and stress fields in concentric layers under combined boundary loading and a Lorentz body force $f_r(r) = J_\theta(r)\, B_z(r)$.

The validation strategy is to sweep $\alpha_\mathrm{WP}$ across the full $[0, 1]$ interval and, at each value, use CIRCE to compute the stress field that is consistent with the prescribed level. The geometry $(R_\mathrm{sep}, \Delta_\mathrm{nose})$ is then optimised to minimise the total radial extent $\Delta_\mathrm{nose} + \Delta_\mathrm{WP}$ subject to the Tresca criterion saturated in both the nose steel and the WP steel. The procedure therefore produces, at every $\alpha_\mathrm{WP}$, a fully self-consistent design where mechanical equilibrium, the Tresca limit, and the ampere-turn constraint are simultaneously satisfied.

:::{figure} /figures/thesis/alpha_wedge_circe_refined.png
:name: fig:wedging_alpha
:width: 75%
:align: center

Optimal radial build of the inboard TF leg as a function of $\alpha_\mathrm{WP}$, computed with the CIRCE multilayer solver on ITER Q=10 reference parameters. The WP steel is split visually into a transmissive part ($\propto 1-\alpha_\mathrm{WP}$) and a vault part ($\propto \alpha_\mathrm{WP}$). This decomposition is conceptual since a single piece of steel reacts $\sigma_r$, $\sigma_\theta$ and $\sigma_z$ simultaneously.
:::

{numref}`Figure %s <fig:wedging_alpha>` shows the optimised radial build for the ITER Q=10 parameter set of {numref}`Table %s <tab:chap2_tf_coil_benchmark>`. The total thickness $\Delta_\mathrm{tot}$ is essentially flat, varying by less than 5% over the full $\alpha_\mathrm{WP}$ interval.

Treating the WP as non-wedged ($\sigma_\theta^\mathrm{WP} \approx 0$) is therefore a reasonable approximation for sizing the total radial extent of the inboard TF leg. This justifies the modelling choice of Section {ref}`Refined model <ssec:chap1_refined>` and removes the need to introduce $\alpha_\mathrm{WP}$ as an additional design degree of freedom in D0FUS.

```{rubric} References
```

```{footbibliography}
```
