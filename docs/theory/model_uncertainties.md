(sec:chap4_modelform)=

# Model uncertainties

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 4.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


For some models of the design chain, several formulations coexist in the literature, each resting on different assumptions or on different experimental datasets, and the design that comes out depends on which one is adopted. Three of them are examined here because of their notable impact on the design: the operational stability limits, the maximum achievable elongation and the energy confinement scaling law. To measure the impact of these choices on the machine design space, the $(R_0, a)$ maps of Section 3.1.3 of the thesis are repeated on the EU-DEMO 2017 reference with one model changed at a time. Note that the whole chain now runs with its Refined models, hence the slight differences between the base map and its counterpart in Chapter 3 of the thesis (Figure 3.2(a)). One methodological point should be kept in mind. For the kink criterion, the maximal elongation and the $\tau_E$ scaling law, the whole map is recomputed with the alternative model. For the density limit, nothing is recomputed: the published EU-DEMO density is simply compared to the limit each alternative model predicts at the reference point. The four maps are gathered in {numref}`Figure %s <fig:chap4_eudemo_maps>`, where it can be seen that, under models that are all defensible, the plasma stability domain (colour background) and the machine feasibility domain change significantly. These changes are discussed in the next subsections.

:::{figure} /figures/thesis/fig_chap4_eudemo_maps_composite.png
:name: fig:chap4_eudemo_maps
:width: 95%
:align: center

Feasible design space of the EU-DEMO 2017 reference in the $(R_0, a)$ plane, with the conventions of the maps of Section 3.1.3 of the thesis: each cell is coloured by its most binding plasma limit (green: kink, red: Troyon, blue: density), the white curve is the plasma stability boundary and the black curve the radial build limit. Panel (a) is the reference, and each other panel changes exactly one model with respect to it. (a) Base assumptions: Wenninger elongation, IPB98(y,2) scaling, kink limit on $q_{95}$; (b) Freidberg elongation closure in place of Wenninger; (c) Kink criterion $q_\ast \geq 2$ in place of $q_{95} \geq 3.5$; (d) ITPA20-IL confinement scaling in place of IPB98(y,2).
:::

(ssec:chap4_limits)=

## The operational stability limits

System codes apply the operational limits as firm numbers. However, the quantity on which these limits should hold, as well as the limit value, can be subject to discussion. The kink limit may be placed on $q_{95}$ (Eqs. {eq}`eq:q95_sauter` and {eq}`eq:q95_iter89`), with thresholds typically at 3 to 3.5, or on the cylindrical $q_\ast$ (Eq. {eq}`eq:qstar`), with thresholds rather at 2 to 2.5 {footcite:p}`freidberg2015designing,coleman2025definition`. The $\beta$ limit depends on which $\beta$ is normalised, thermal or including the fast-particle pressure, and on where the threshold is set, the Troyon coefficient spanning roughly 2.8 to 3.5 across studies (Section {ref}`Stability limits <ssec:chap1_stability>`). Finally, several substantially different models are candidates to describe the density limit: the Greenwald form (Eq. {eq}`eq:greenwald`), the power-balance limit of Zanca and the edge-turbulence limit of Giacomin (Section {ref}`Stability limits <ssec:chap1_stability>`). They lie on different bases. The Greenwald limit rests on a large experimental base and reproduces the limit observed on many machines reasonably well {footcite:p}`greenwald2002density`. The theoretical and numerical limits of Zanca {footcite:p}`zanca2019power` and Giacomin {footcite:p}`giacomin2022density` are more recent and less consensual. The density limit from each model, and the fraction of this limit at which the machine operates, are given for the reference design in {numref}`Table %s <tab:density_models>`. The operating fraction is the ratio of the published EU-DEMO density to the limit predicted by each model, both expressed in the convention proper to that model: the line-averaged density for Greenwald and Zanca, the near-separatrix density for Giacomin. The three models disagree quite strongly. The Greenwald limit is the only one that is overcome, EU-DEMO operating at $1.1$ times the Greenwald density, while the Zanca limit leaves $40\%$ of headroom and the Giacomin power-dependent edge limit is effectively non-binding at power plant scale. Note that the Giacomin fraction is not directly comparable to the other two: the model bounds the near-separatrix density. Opting for a different maximum-density model could then have a major impact on the design, the accessible density being one of the main levers on the fusion power at fixed volume, a point also raised by a recent work by Angioni et al. {footcite:p}`angioni2026density`.

The choice of the kink limit model is just as consequential. On the base map ({numref}`Figure %s(a) <fig:chap4_eudemo_maps>`) the feasible region is kink-bounded from below: $q_{95}$ reduces as the plasma shrinks, and against the EU-DEMO design limit $q_{95} \geq 3.0$, the floor is crossed near $R_0 \approx 9$ m. Replacing the criterion by the Freidberg convention $q_\ast \geq 2$ {footcite:p}`freidberg2015designing` ({numref}`Figure %s(c) <fig:chap4_eudemo_maps>`) reveals that $q_\ast > 2$ is satisfied over a much broader region, expanding the feasible region significantly and leaving room for designs below $2.5$ m of minor radius.

:::{table} Density limit predicted by three models at the EU-DEMO 2017 reference point. The operating fraction is evaluated in each model’s own convention: on the line-averaged density ($\bar n = 0.79 \times 10^{20}\,\mathrm{m^{-3}}$) for Greenwald and Zanca, on the near-separatrix density ($n(\rho = 0.9) \approx 0.58 \times 10^{20}\,\mathrm{m^{-3}}$) for Giacomin.
:name: tab:density_models
:align: center

| Density-limit model     | Convention      | Limit $[10^{20}\,\mathrm{m^{-3}}]$ | Operating fraction |
|:------------------------|:----------------|:-----------------------------------|:-------------------|
| Greenwald $I_p/\pi a^2$ | line-averaged   | $0.73$                             | $1.08$             |
| Zanca (2019)            | line-averaged   | $1.31$                             | $0.60$             |
| Giacomin (2022)         | near-separatrix | $3.60$                             | $0.16$             |
:::

(ssec:chap4_elongation)=

## The maximum achievable elongation

The choice of the model for the maximum achievable elongation (Section {ref}`Flux-surface geometry <ssec:chap1_geometry>`) also has an important impact on the design, through two distinct channels. First, at a fixed safety-factor limit, a higher elongation allows a higher plasma current, which improves the confinement time (the IPB98(y,2) scaling carries $\tau_E \propto I_p^{0.93}\,\kappa^{0.78}$) and, through $n_{GW} = I_p/\pi a^2$ (Eq. {eq}`eq:greenwald`), raises the Greenwald ceiling at fixed minor radius. Second, at fixed $R_0$, $a$ and $I_p$, the plasma volume grows with $\kappa$ while $n_{GW}$ is unchanged: the fusion power can then rise at a constant Greenwald fraction. Elongation is indeed expected, in the literature, to be one of the parameters with the largest impact on a design {footcite:p}`kahn2020sensitivity,kemp2017dealing,pearce2022sensitivity`. For example, at the EU-DEMO aspect ratio, the Freidberg model gives $\kappa = 1.97$ while the Wenninger model gives $1.88$, shifting the smallest feasible major radius from about $9$ m to about $8$ m ({numref}`Figure %s(b) <fig:chap4_eudemo_maps>`).

(ssec:chap4_scaling)=

## The energy confinement scaling law

Finally, replacing the IPB98(y,2) scaling with the more recent ITPA20-IL scaling {footcite:p}`verdoolaege2021itpa` ({numref}`Figure %s(d) <fig:chap4_eudemo_maps>`) contracts the feasible region and pushes it towards larger machines, no design remaining available below a minor radius of $3.6$ m. The reason is that ITPA20-IL is less optimistic, so that, at fixed size, the plasma current has to be increased to keep the same confinement, which reduces $q_{95}$. This law should definitely not be treated as settled. Indeed, its dependence on the major radius, $R^{1.19}$, is markedly weaker than that of the standard ITPA20 fit, $R^{1.71}$, and weaker still than the $R^{1.97}$ of earlier work, which may raise doubt. The 2025 ITPA review recommends using it nonetheless, while stating that the reason for this reduced size dependence is currently under investigation {footcite:p}`yoshida2025transport,hall2023confinement`. No revised law has been published to date.

## Model uncertainties synthesis

{numref}`Figure %s <fig:chap4_eudemo_maps>` summarises the situation: the same machine, under different model choices that are all defensible, yields feasible regions that significantly vary from one another. Note that these maps only vary the model choices one at a time: taken in combination, their impact may be more marked still. These maps also temper a message of Chapter 3 of the thesis: the statement that the radial build is the dominant constraint at high field holds under the adopted plasma conventions, and an equally defensible set of conventions could hand the leading role back to the plasma limits. This model-form uncertainty is, in principle, reducible: once the community settles a question, the most refined formulation can simply be adopted, as is already the case in D0FUS for the bootstrap current.

A second source of uncertainty is of a different kind: for a fixed choice of model variants, the values of the input and calibration parameters are themselves uncertain.

```{rubric} References
```

```{footbibliography}
```
