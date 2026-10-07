(app:refined_model)=

# Refined radial build model

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.17. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section gives the complete Refined radial build model summarised in Section {ref}`Refined model <ssec:chap1_refined>`. The notation is that of Section {ref}`Radial build definition and geometry <ssec:chap1_radial_build>`, and the sizing criteria are those of {ref}`Models <chap:d0fus>`.

The Refined model provides more realistic thickness predictions at the cost of moderately increased complexity. It draws on classical analytical treatments of stress distributions in fusion magnets {footcite:p}`thome1982mhd,wilson1983superconducting,burkhard1975magnetic,gray1977electromechanical,johnson1977stress,swanson2022validation,boudes2025circe` and on their integration into system codes {footcite:p}`morris2015implications,morris2021preparing,reux2018demo,duchateau2014conceptual,giannini2023magnet`. Two key changes are introduced with respect to the Academic model of Section {ref}`Academic model <ssec:chap1_academic>`: thin-cylinder stress theory is replaced by thick-cylinder (Lamé-Clapeyron {footcite:p}`LameClapeyron1833`) theory, and the idealised pure-conductor layer is replaced by a composite winding pack in which the steel jacket of the conductor coexists with the cable in every cross-section. Both the TF and CS coils are modelled as cable-in-conduit conductors (CICC). In reality the TF coil can also be composed of radial plates in which the cables are embedded, but the sizing of the different thicknesses would not change. The notation is that of Section {ref}`Radial build definition and geometry <ssec:chap1_radial_build>`.

## TF winding pack

Three dimensionless fractions characterise the composite winding pack. They are summarised in {numref}`Table %s <tab:fractions_TF>`.

:::{table} Geometric and load-distribution fractions used throughout the Refined model.
:name: tab:fractions_TF
:align: center

| **Symbol**          | **Definition**                                                             |
|:--------------------|:---------------------------------------------------------------------------|
| $f_c$               | Cable (non-steel) fraction of the WP cross-section: $S_C / (S_C + S_S)$    |
| $f_u$               | Useful steel fraction for the radial stress: $S_U / S_R$                   |
| $f_{z,\mathrm{WP}}$ | Share of the vertical tension carried by the WP: $F_{z,\mathrm{WP}} / F_z$ |
:::

The cable fraction $f_c = S_C/(S_C + S_S)$ compares the non-steel conductor surface $S_C$ (superconductor, copper, helium, insulation, the surface on which $J^\mathrm{wost}$ is defined) to the steel jacket surface $S_S$ in a given CICC ({numref}`Fig. %s <fig:wpmodelcicc_eclate>`). When the whole cross-section is filled with identical CICC, $f_c$ does not depend on the radial position and is sized to handle the worst loading case. The radially varying case (grading) is treated in {ref}`Radially graded conductor fraction <appendix_grading>`.

:::{figure} /figures/thesis/WP_model_CICC_eclate.png
:name: fig:wpmodelcicc_eclate
:width: 65%
:align: center

CICC illustration and surface definition: in red the non-steel cross-section on which $J^\mathrm{wost}$ is defined ($S_C$), and in blue the steel jacket sized for mechanical purposes ($S_S$).
:::

The tension-sharing fraction $f_{z,\mathrm{WP}} = F_{z,\mathrm{WP}}/F_z$ ranges from 1 (all the tension held by the winding pack) to 0 (all the tension held by the nose). It affects the distribution of steel between the winding pack and the nose but has little impact on the total inboard leg thickness, as shown in {ref}`Sensitivity to the WP/nose tension partition <app:omega_sensitivity>`. It would have been possible to determine it self-consistently by comparing the steel surfaces of the two components and iterating until convergence, but the weak sensitivity of the final result does not justify such an additional convergence loop. In wedging the default $f_{z,\mathrm{WP}} = 1/2$ is adopted, consistent with the ITER design {footcite:p}`federici2026iter`. In bucking it is logically set to $f_{z,\mathrm{WP}} = 1$ since there is no nose.

The useful steel fraction $f_u = S_U/S_R$ compares the steel surface $S_U$ that actually withstands the loads in the $r$ direction to the total cylindrical surface $S_R$ formed by the coils ({numref}`Fig. %s <fig:surfacedilution>`). Considering that the cable in the CICC cannot support any load (which is true for low-temperature superconductors, namely Nb$_3$Sn and NbTi, but more debatable for high-temperature superconductors, namely REBCO {footcite:p}`Godeke2006_Nb3Sn_review,Zhou2023_REBCO_mech,scanlan1980mechanical,barth2015electro`), the cables are mechanically equivalent to holes in the winding pack, leading to a radial stress concentration factor.

:::{figure} /figures/thesis/fig_surfacedilution_composite.png
:name: fig:surfacedilution
:width: 95%
:align: center

Cylinder with drilled hole approximation of the coil (a and b) and definition of the surfaces $S_R$ and $S_U$ used in the stress calculations (c and d). (a) Vault created by the coils; (b) Cylindrical approximation of the coils; (c) Complete cylindrical surface $S_R$; (d) Useful cylindrical surface $S_U$.
:::

Following the field generation criterion of Eq. {eq}`eq:B_thin_acad` but without the thin-cylinder approximation ({ref}`Thick-cylinder field and smeared radial stress in the TF winding pack <app:B_thick>`), Ampère’s law on a thick cylinder gives $B_\mathrm{max} = (\mu_0 f_c J^\mathrm{wost}/2)\,[R_\mathrm{TF}^\mathrm{ext} - (R_\mathrm{TF}^\mathrm{sep})^2/R_\mathrm{TF}^\mathrm{ext}]$, hence

$$
f_c = \frac{2\, B_\mathrm{max}}{\mu_0\, J^\mathrm{wost}}\,\frac{R_\mathrm{TF}^\mathrm{ext}}{(R_\mathrm{TF}^\mathrm{ext})^2 - (R_\mathrm{TF}^\mathrm{sep})^2}
$$ (eq:fc)

For a uniform $f_c$ this fixes the relation between $f_c$ and $R_\mathrm{TF}^\mathrm{sep}$.

Parametrising the CICC steel jacket by an asymmetry parameter $n = \delta_{S_1}/\delta_{S_2}$ (ratio of radial to toroidal jacket thicknesses, {ref}`Determination of f_(u)(f_(c);n) <app:gamma>`, {numref}`Fig. %s <fig:surfacedilutionconductor>`), one obtains

$$
f_u(f_c,\,n) = \frac{2\pi + 4\,f_c\,(n-1) - \sqrt{\bigl[2\pi + 4\,f_c\,(n-1)\bigr]^2 - 4\pi(\pi - 4\,f_c)}}{2\pi}
$$ (eq:fu)

The asymmetry parameter ranges from $n = 1$ (a square jacket with equal thickness in both directions, the standard round-in-square CICC geometry and the D0FUS default) down to $n = 0$ (no steel in the radial direction, maximising $f_u$ for a given $f_c$ but geometrically unrealistic). The default $n = 1$ yields the lowest $f_u$ for a given $f_c$, hence the most conservative (thickest) coil predictions. Intermediate values, achievable by elongating the jacket toroidally, are an optimisation lever exploited in Chapter 3 of the thesis. The resulting $f_u(f_c)$ relation is plotted in {numref}`Fig. %s <fig:fu_vs_fc>` for several values of $n$.

:::{figure} /figures/thesis/Surface_dilution_conductor.png
:name: fig:surfacedilutionconductor
:width: 75%
:align: center

Conductor with parametrisation of its characteristic lengths in a generic, $n = 0$ and finally $n = 1$ cases.
:::

:::{figure} /figures/thesis/fu_vs_fc.png
:name: fig:fu_vs_fc
:width: 65%
:align: center

Useful steel fraction $f_u(f_c)$ for different values of the asymmetry parameter $n$, from Eq. {eq}`eq:fu`.
:::

The radial stress in the steel at the WP bore is the smeared magnetic load concentrated on the useful steel surface fraction $f_u$. Departing from the thin-cylinder limit, the distributed Lorentz body force is integrated through the thick cylinder from the outer surface inward (with $B(R)$ from {ref}`Thick-cylinder field and smeared radial stress in the TF winding pack <app:B_thick>`), which gives

$$
\begin{aligned}
\sigma_r &= -\frac{f_\mathrm{L}(R_\mathrm{TF}^\mathrm{ext}, R_\mathrm{TF}^\mathrm{sep})}{f_u}\,P_\mathrm{TF}, \qquad f_\mathrm{L} = \frac{4\,(R_\mathrm{TF}^\mathrm{ext})^2\,(R_\mathrm{TF}^\mathrm{ext} + 2 R_\mathrm{TF}^\mathrm{sep})}{3\,R_\mathrm{TF}^\mathrm{sep}\,(R_\mathrm{TF}^\mathrm{ext} + R_\mathrm{TF}^\mathrm{sep})^2} \\[6pt]
	\sigma_z &= \frac{f_{z,\mathrm{WP}}}{(1 - f_c)}\,\frac{B_\mathrm{max}^2\,(R_\mathrm{TF}^\mathrm{ext})^2}{\bigl[(R_\mathrm{TF}^\mathrm{ext})^2 - (R_\mathrm{TF}^\mathrm{sep})^2\bigr]\,2\mu_0}\,\ln\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}
\end{aligned}
$$ (eq:sigma_r_refined)

with $P_\mathrm{TF}$ the magnetic pressure of Eq. {eq}`eq:P_TF`. The geometric factor $f_\mathrm{L}$ accounts for the thick-cylinder accumulation of the distributed Lorentz body force: it tends to unity in the thin-shell limit, where the magnetic pressure $P_\mathrm{TF}$ is recovered (the Academic surface-pressure result), and grows monotonically as the winding pack thickens, reaching $\approx 1.3$ for a moderate inner leg and $\approx 2$ for the thickest high-field coils. The vertical stress $\sigma_z$ distributes the tension over the steel area $(1 - f_c)\, S_\mathrm{tot}$ in the plane perpendicular to $z$.

For completeness, departing from the thin-cylinder limit, Lamé-Clapeyron theory gives the winding-pack hoop stress ({ref}`Determination of f_(u)′(f_(u);n) <Appendix_little_demo>`)

$$
\sigma_\theta = \frac{2\, P_\mathrm{TF}\, (R_\mathrm{TF}^\mathrm{ext})^2}{(R_\mathrm{TF}^\mathrm{ext})^2 - (R_\mathrm{TF}^\mathrm{sep})^2}\,\frac{1 - f_u + n\, f_u}{n\, f_u}
$$ (eq:sigma_theta_WP_refined)

the factor $(1 - f_u + n\, f_u)/(n\, f_u)$ accounting for the useful steel surface in the $\theta$ direction. This expression is reported for completeness. As detailed below, $\sigma_\theta$ is not retained in the winding pack sizing.

In both configurations (wedging and bucking), the hoop stress $\sigma_\theta$ in the winding pack is neglected: it is assumed to be predominantly recovered by the nose. In reality $\sigma_\theta$ does contribute to the winding pack stress (in ITER about 30 % of the vault effect is estimated to be taken by the WP {footcite:p}`wilson1983superconducting`), but as quantified in {ref}`Non-wedged winding-pack hypothesis <app:wedg_approx>` with the multi-layer thick-cylinder solver CIRCE {footcite:p}`boudes2025circe`, this mainly affects the steel distribution between the nose and the winding pack, not the total coil thickness. With $\sigma_\theta \approx 0$ in the winding pack, and noting that $\sigma_r < 0$ (compression) while $\sigma_z > 0$ (tension), the Tresca criterion (Eq. {eq}`eq:Tresca`) reduces to

$$
|\sigma_r| + \sigma_z \;\leq\; \sigma_\mathrm{lim}
$$ (eq:Tresca_WP_simplified)

which holds for both mechanical architectures. Setting this to its limit and solving for $R_\mathrm{TF}^\mathrm{sep}$ yields an implicit equation, since $f_c$ depends on $R_\mathrm{TF}^\mathrm{sep}$ (Eq. {eq}`eq:fc`), $f_u$ on $f_c$ (Eq. {eq}`eq:fu`), and $f_\mathrm{L}$ on $R_\mathrm{TF}^\mathrm{sep}$ (Eq. {eq}`eq:sigma_r_refined`). The solution is found numerically by iterating inward from $R_\mathrm{TF}^\mathrm{ext}$. The conductor fraction $f_c$ has so far been taken uniform across the winding pack. A radially graded variant, which redistributes the steel to saturate the Tresca criterion at every radius and reduce the total thickness, is derived in {ref}`Radially graded conductor fraction <appendix_grading>` and its impact on the radial build assessed in Chapter 3 of the thesis.

A cover of thickness $\Delta_\mathrm{cover} = 7$ cm is added on top of the winding pack for reinforcement and high-voltage impregnation (see {numref}`Fig. %s <fig:ITER>`), following the ITER design {footcite:p}`sborchia2008design`. This thickness is relatively consistent across the designs studied, ranging from 5 to 10 cm.

:::{figure} /figures/thesis/ITER_TF.png
:name: fig:ITER
:width: 60%
:align: center

ITER TF coil cross-section, showing the winding pack, the steel case and the outer cover, taken from {footcite:p}`sborchia2008design`.
:::

## TF coil nose (wedging only)

In the wedging configuration (the only one where a steel nose is considered), the centering force is conserved but applied on a smaller radius, so that the effective pressure at $R_\mathrm{TF}^\mathrm{sep}$ is

$$
P'_\mathrm{TF} = P_\mathrm{TF}\,\frac{R_\mathrm{TF}^\mathrm{ext}}{R_\mathrm{TF}^\mathrm{sep}} = \frac{B_\mathrm{max}^2}{2\mu_0}\,\frac{R_0 - a - \Delta_B}{R_\mathrm{TF}^\mathrm{sep}}
$$

The nose is in azimuthal compression ($\sigma_\theta < 0$) and vertical tension ($\sigma_z > 0$), so the Tresca criterion reads $\sigma_\mathrm{Tresca} = \sigma_z - \sigma_\theta \leq \sigma_\mathrm{lim}$, with

$$
\sigma_\theta = -\frac{2\, P'_\mathrm{TF}\,(R_\mathrm{TF}^\mathrm{sep})^2}{(R_\mathrm{TF}^\mathrm{sep})^2 - (R_\mathrm{TF}^\mathrm{int})^2}, \qquad
	\sigma_z = (1 - f_{z,\mathrm{WP}})\,\frac{B_\mathrm{max}^2\,(R_\mathrm{TF}^\mathrm{sep})^2}{\bigl[(R_\mathrm{TF}^\mathrm{sep})^2 - (R_\mathrm{TF}^\mathrm{int})^2\bigr]\,2\mu_0}\,\ln\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}
$$

Setting $\sigma_\mathrm{Tresca} = \sigma_\mathrm{lim}$ leads to the nose inner radius

$$
R_\mathrm{TF}^\mathrm{int} = \sqrt{(R_\mathrm{TF}^\mathrm{sep})^2 - \frac{(R_\mathrm{TF}^\mathrm{sep})^2}{\sigma_\mathrm{lim}}\left[2\,P'_\mathrm{TF} + (1 - f_{z,\mathrm{WP}})\,\frac{B_\mathrm{max}^2}{2\mu_0}\,\ln\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}\right]}
$$ (eq:nose)

## CS winding pack

The CS winding pack is modelled analogously to the TF coil winding pack, using the same fractions $f_c$ and $f_u$ introduced above ({numref}`Table %s <tab:fractions_TF>`). Since no CS nose is considered in this model, the parameter $f_{z,\mathrm{WP}}$ is not used.

The CS must deliver to the plasma the flux $\Psi_\mathrm{CS} = \Psi_\mathrm{Init} + \Psi_\mathrm{Ramp\text{-}Up} + \Psi_\mathrm{plateau} - \Psi_\mathrm{PF}$ from the balance of Eq. {eq}`eq:CS_flux_balance`, and the hardware capacity it is sized for is $\Psi_\mathrm{CS}^\mathrm{cap} = \Psi_\mathrm{CS}/f_\mathrm{swing}^\mathrm{usable}$, with the usable-swing fraction $f_\mathrm{swing}^\mathrm{usable} = 0.75$ of Section {ref}`Academic model <ssec:chap1_academic>`. Combining this capacity with the full-swing flux expression and Ampère’s theorem for the composite solenoid gives

$$
\Psi_\mathrm{CS}^\mathrm{cap} = \frac{2\pi B_\mathrm{CS}}{3}\,\Bigl[(R_\mathrm{CS}^\mathrm{ext})^2 + R_\mathrm{CS}^\mathrm{ext}\, R_\mathrm{CS}^\mathrm{int} + (R_\mathrm{CS}^\mathrm{int})^2\Bigr], \qquad B_\mathrm{CS} = \mu_0\, f_c\, J^\mathrm{wost,CS}\, (R_\mathrm{CS}^\mathrm{ext} - R_\mathrm{CS}^\mathrm{int})
$$ (eq:Psi_CS)

the second relation being Ampère’s law for the composite solenoid. The only unknown is $R_\mathrm{CS}^\mathrm{int}$, but a thicker winding pack delivers more flux while the field $B_\mathrm{CS}$ that the conductor can sustain decreases with the operating field through $J^\mathrm{wost,CS}(B_\mathrm{CS})$, so a self-consistent iteration on $B_\mathrm{CS}$ is required, initialised by $B_\mathrm{CS} \approx \Psi_\mathrm{CS}^\mathrm{cap}/(\pi (R_\mathrm{CS}^\mathrm{ext})^2)$.

Unlike the Academic model, the Refined model accounts for the axial compressive stress induced by the radial component of the fringe field at the CS ends. In a finite-length solenoid, $\nabla \cdot \vec{B} = 0$ implies a non-zero $B_r$ near the coil extremities, and the resulting $J_\theta\, B_r$ force pushes the winding pack toward the midplane. Integrating from the free end ($z = h$, $\sigma_z = 0$) to the midplane ($z = 0$) yields (derivation in {ref}`Axial stress at the CS midplane <app:sigma_z_CS>`)

$$
\sigma_z^\mathrm{smear} = -\frac{\mu_0\, J_\mathrm{smear}^2\, h\, R_\mathrm{CS}^\mathrm{int}}{2}\,\bigl[\mathcal{L}(h) - \mathcal{L}(2h)\bigr]
$$ (eq:sigma_z_CS_smear)

with $J_\mathrm{smear} = f_c\, J^\mathrm{wost,CS}$ the homogenised current density, $h = H_\mathrm{CS}/2$ the CS half-height ($H_\mathrm{CS} = 2(\kappa a + \Delta_B + 1)$ by default), and

$$
\mathcal{L}(\zeta) = \ln\!\frac{R_\mathrm{CS}^\mathrm{ext} + \sqrt{(R_\mathrm{CS}^\mathrm{ext})^2 + \zeta^2}}{R_\mathrm{CS}^\mathrm{int} + \sqrt{(R_\mathrm{CS}^\mathrm{int})^2 + \zeta^2}}
$$ (eq:L_function)

a decreasing function of $\zeta$, so the bracketed term is positive and the stress compressive. The peak steel stress is $\sigma_z^\mathrm{steel} = \sigma_z^\mathrm{smear}/f_u$. This stress is compressive and typically one order of magnitude smaller than the hoop stress. It is computed using the CS current at the most critical instant identified in Section {ref}`Academic model <ssec:chap1_academic>`: in the wedging configuration and in light bucking this corresponds to $I_\mathrm{CS} = I_\mathrm{CS,max}$, while in strong bucking ($I_\mathrm{CS} = 0$) it vanishes. A real modular CS, whose modules carry different and sometimes reversed currents, gives a more complex distribution. The monolithic-solenoid approximation nonetheless provides a reasonable first estimate of the relevant magnitudes.

In the wedging configuration, the most critical hoop stress occurs at the bore $R_\mathrm{CS}^\mathrm{int}$ (where $\sigma_r = 0$). The CICC is oriented in the $\theta$ direction, so the useful steel surface factor is $f_c$ (rather than $f_u$). Applying thick-cylinder Lamé theory under the internal magnetic pressure $P_\mathrm{CS} = B_\mathrm{CS}^2/(2\mu_0)$,

$$
\sigma_\theta^\mathrm{max} = \frac{1}{(1 - f_c)}\,\frac{P_\mathrm{CS}\,\bigl[(R_\mathrm{CS}^\mathrm{ext})^2 + (R_\mathrm{CS}^\mathrm{int})^2\bigr]}{(R_\mathrm{CS}^\mathrm{ext})^2 - (R_\mathrm{CS}^\mathrm{int})^2}
$$ (eq:sigma_theta_CS_wedg_refined)

Since $|\sigma_z^\mathrm{steel}| \ll |\sigma_\theta^\mathrm{max}|$, the Tresca criterion is dominated by the hoop stress. Setting $|\sigma_\theta^\mathrm{max} - \sigma_z^\mathrm{steel}| = \sigma_\mathrm{lim}$ yields a polynomial equation of degree 7 in $R_\mathrm{CS}^\mathrm{int}$, solved numerically.

(sec:CS_fatigue)=
**Fatigue.** In wedging and light bucking, the CS breathes at every plasma pulse, the hoop stress cycling between full tension at $I_\mathrm{CS} = \pm I_\mathrm{CS,max}$ and rest at $I_\mathrm{CS} = 0$. This cyclic loading induces fatigue, whose actual impact depends on the steel microstructure and the number of cycles over the plant lifetime {footcite:p}`jong2007iter,sarasola2020progress,sutcliffe2025magnet`. As a first approximation, the steel allowable entering the CS Tresca criterion is divided by a configurable fatigue knockdown factor (default 2), consistent with standard practice in preliminary design studies. In strong bucking and plug configurations, or in steady-state operation, the CS remains in compression throughout the cycle (which tends to close rather than propagate cracks {footcite:p}`elber1971significance,pippan2017fatigue,newman1981crack,shih1974study`) or does not cycle, so the fatigue knockdown is not applied.

In the bucking configuration, as established in Section {ref}`Academic model <ssec:chap1_academic>`, the regime expected for tokamak designs is strong bucking, where the critical instant is $I_\mathrm{CS} = 0$ and only the TF pressure acts on the CS, transported by cylindrical force balance to its outer surface,

$$
P''_\mathrm{TF} = \frac{B_\mathrm{max}^2}{2\mu_0}\,\frac{R_0 - a - \Delta_B}{R_\mathrm{CS}^\mathrm{ext}}
$$ (eq:Ppp_TF)

The hoop stress is then

$$
|\sigma_\theta^\mathrm{max}| = \frac{1}{(1 - f_c)}\,\frac{2\, P''_\mathrm{TF}\,(R_\mathrm{CS}^\mathrm{ext})^2}{(R_\mathrm{CS}^\mathrm{ext})^2 - (R_\mathrm{CS}^\mathrm{int})^2}
$$ (eq:sigma_theta_CS_buck)

Since the CS current vanishes at the dimensioning instant, the axial stress is zero and the Tresca criterion reduces to $|\sigma_\theta^\mathrm{max}| \leq \sigma_\mathrm{lim}$, solved numerically for $R_\mathrm{CS}^\mathrm{int}$. The light bucking branch (where the axial stress is retained, giving $|\sigma_\theta^\mathrm{max} - \sigma_z^\mathrm{steel}| \leq \sigma_\mathrm{lim}$) is implemented for completeness through a $\max(\cdot)$ comparison but is almost never selected in tokamak designs.

In the plug configuration, when the TF pressure dominates over the CS own pressure (the usual case, Section {ref}`Academic model <ssec:chap1_academic>`), the critical instant is $I_\mathrm{CS} = 0$ and the CS simply transmits the radial pressure to the plug. The dominant stress is the radial compression $\sigma_r = P''_\mathrm{TF}/f_u$, and the sizing equation is solved numerically as in the previous configurations. In the rare opposite case, the bucking model applies. To first order the plug relaxes the CS mechanical constraint to $P''_\mathrm{TF} < \sigma_\mathrm{lim}$, easily satisfied at all realistic field levels, so the CS sizing is then governed essentially by the flux requirement. This is the key structural advantage of the plug at high field, confirmed by the COMSOL study of Section {ref}`Academic model <ssec:chap1_academic>` ({numref}`Fig. %s <fig:comsolplug>`).

The thickness $\Delta_\mathrm{CS} = R_\mathrm{CS}^\mathrm{ext} - R_\mathrm{CS}^\mathrm{int}$ is obtained by the same inward root-finding on the winding-pack thickness as for the TF coil, and converges in a handful of iterations for all realistic design points.

```{rubric} References
```

```{footbibliography}
```
