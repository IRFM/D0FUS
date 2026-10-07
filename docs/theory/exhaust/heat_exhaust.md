(ssec:chap1_exhaust)=

# Heat exhaust: the third first-order limit

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.4.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The heat flux reaching the divertor target plates is one of the most critical aspects of a power plant design. Unlike the plasma stability and the coil feasibility, however, it cannot be turned into a hard limit at the system-code level: whether a given exhaust power is acceptable depends on the divertor geometry, on the target technology and on the mitigation strategy (impurity seeding, detachment control, advanced magnetic configurations), all of which belong to dedicated design studies conducted once a candidate machine exists, and outside the scope of a system code. D0FUS therefore evaluates the exhaust through proxies with various levels of physical content, which do not veto a design but quantify how hard its exhaust problem will be, and allow comparisons against machines whose exhaust solutions are better established.

The exhaust chain is anchored on two powers inherited from the balance of Section {ref}`Power balance and energy confinement time <ssec:chap1_power_balance>`. The power crossing the separatrix (i.e. the LCFS), used for comparison against the L-H threshold, reads

$$
P_\mathrm{sep} = P_\alpha + P_\mathrm{aux} - P_\mathrm{rad,total}
$$ (eq:Psep)

with $P_\mathrm{aux}$ the total plasma-absorbed auxiliary power (in steady state, itself set by the current-drive requirement of Section {ref}`Plasma current and scaling law <ssec:chap1_currents>`) and $P_\mathrm{rad,total}$ the total power radiated from the confined plasma (Section {ref}`Power balance and energy confinement time <ssec:chap1_power_balance>`). The whole of the radiated power is subtracted here: wherever inside the LCFS the photons are emitted, they land on the first wall directly and do not cross the separatrix as conducted or convected heat. The ohmic contribution is omitted for consistency with the L-H threshold scalings {footcite:p}`martin2008power`, where $P_\Omega$ is neglected. The divertor heat flux input, the reference for the scalings of the power decay length $\lambda_q$ (the e-folding width, at the outer midplane, of the heat flux profile flowing in the scrape-off layer introduced with the exhaust problem in the thesis introduction, which together with the field-line pitch sets the surface actually wetted on the targets) and for the target heat flux estimate, only restores the small ohmic term,

$$
P_\mathrm{div} = P_\alpha + P_\mathrm{aux} + P_\Omega - P_\mathrm{rad,total} = P_\mathrm{sep} + P_\Omega
$$ (eq:Pdiv)

so that $P_\mathrm{div} \approx P_\mathrm{sep}$ at this level of description: no radiation is emitted in the scrape-off layer itself at this stage, the dissipation inside it being precisely what the third-layer model below estimates. Deliberate impurity seeding with neon or argon raises $P_\mathrm{rad,total}$ and thereby lowers both quantities, while only weakly affecting the core power balance, since, as discussed in Section {ref}`Power balance and energy confinement time <ssec:chap1_power_balance>`, the power radiated near the edge barely alters the turbulent transport that the confinement scalings describe.

The power heading for the targets is only one of the two terms of the problem. The other is the surface over which it deposits, which D0FUS addresses in three layers of increasing physical content.

The first layer is a set of closed-form proxies built directly from global quantities, simple but useful for ranking the exhaust difficulty of a design. All descend from the same empirical fact: the scrape-off-layer width scales as $\lambda_q \propto 1/B_\mathrm{pol}$ {footcite:p}`eich2013scaling`. The crudest proxy, $P_\mathrm{sep}/R_0$, amounts to treating $\lambda_q$ as a constant and keeps only the scaling with machine size, giving a first proxy for the heat flux density striking the targets. Combining $\lambda_q \propto 1/B_\mathrm{pol}$ with the field-line pitch makes the poloidal field cancel exactly, leaving $P_\mathrm{sep}\, B_0/R_0$ as a direct measure of the parallel heat flux to dissipate, a figure of merit in common use in systems studies and spelled out for example in Ref. {footcite:p}`freidberg2015designing`. That of Siccinio et al. {footcite:p}`siccinio2019heat`, $P_\mathrm{sep}\, B_0/(q_{95}\, A\, R_0)$ with $A = R_0/a$, keeps the safety factor and aspect ratio explicit, and is proportional to $P_\mathrm{sep}\, B_\mathrm{pol}/R_0$, the poloidal projection of the same flux, hence again a proxy for the heat flux density striking the targets.

The second layer refines the estimate of the deposition surface. The parallel heat flux that flows in the scrape-off layer is set by the power decay length $\lambda_q$ at the outer midplane. Following the multi-machine analysis of Eich et al. {footcite:p}`eich2013scaling`,

$$
\lambda_q = 1.35\, R_0^{0.04}\, B_\mathrm{pol}^{-0.92}\, \varepsilon^{0.42}\, P_\mathrm{sol}^{-0.02} \quad [\mathrm{mm}]
$$ (eq:lambda_q)

where $B_\mathrm{pol}$ is the outer-midplane poloidal field, $\varepsilon = a/R_0$, and the scrape-off-layer power $P_\mathrm{sol}$ is taken equal to $P_\mathrm{sep}$, the quantity on which the regression itself was fitted. The numerical coefficients in Eq. {eq}`eq:lambda_q` are those of regression number 15 of Eich et al. The SOL power flows along the field lines through the annular cross-section $2\pi R_0\, \lambda_q\, (B_\theta/B)$, so the peak parallel heat flux at the outer midplane reads $q_{\|u} = f_\mathrm{out}\, P_\mathrm{sol}\, (B/B_\theta)/(2\pi R_0\, \lambda_q)$, where $B/B_\theta$ is the outer-midplane field-line pitch (roughly $q_{95}\,A/\kappa$, a factor $3$ to $6$ in conventional tokamaks) and $f_\mathrm{out} = 0.65$ the fraction of $P_\mathrm{sol}$ carried to the outer target. The flux deposited on the target at the total field-line incidence angle $\theta$ is $q_\mathrm{target} = q_{\|u}\, \sin\theta$, with the default $\theta = 2.7^\circ$ calibrated on ITER {footcite:p}`gunn2017surface`. This attached, fully conducting estimate takes the target-to-upstream flux expansion $R_t/R_u$ (the ratio of the strike-point to outer-midplane major radii, which rescales the wetted area) equal to unity and therefore represents an upper bound.

The Eich regression is dominated by its poloidal-field dependence, $\lambda_q \propto B_\mathrm{pol}^{-0.92} \approx B_\mathrm{pol}^{-1}$, so the parallel flux scales as $q_{\|u} \propto P_\mathrm{sep}\, B_\mathrm{pol}/R_0$, closing the loop with the first-layer proxies.

The third layer, finally, models the scrape-off-layer state explicitly. It stands apart from the two previous ones: where they rest on multi-machine scaling laws, it rests on a reduced model whose assumptions weigh as much as its output, and it is therefore only summarised here. Its purpose is to estimate how much impurity seeding is required to dissipate the exhaust power before it reaches the targets. D0FUS implements a two-point model in the Stangeby formulation {footcite:p}`stangeby2018detachment,kotov2009numerical`, complemented by the Lengyel integral {footcite:p}`lengyel1981analysis` for the impurity concentration required to radiate the power to be dissipated, and fed with non-coronal cooling curves appropriate to the finite residence time of impurities in the scrape-off layer. It returns two outputs: the target electron temperature, which places the design on the attached-to-detached scale, and the fraction of the exhaust power that must be radiated away, through the seeding concentration computed with the Lengyel integral, for the deposited flux to stay below a prescribed engineering limit. The model, its assumptions, the temperature thresholds against which its output is read and the non-coronal cooling data are described in {ref}`Two-point divertor model and the Lengyel integral <app:two_point_model>`.

```{rubric} References
```

```{footbibliography}
```
