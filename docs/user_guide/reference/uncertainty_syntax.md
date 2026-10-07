(app:uq_presets)=

# Uncertainty-mode input syntax and the default presets

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.5. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page collects the input syntax of the Uncertainty mode introduced in Section {ref}`Execution modes <sssec:chap1_modes>` and derives every default distribution of {numref}`Tables %s <tab:uq_presets>` and {numref}`%s <tab:uq_presets_eng>` from its documented anchor. The values quoted are those adopted for the ITER reference study. Where the power-plant context (a single, repeated, performant scenario) calls for a recentred variant, it is noted in passing and detailed with the plant study. The aim throughout is a spread centred on the design deck and bounded by documented anchors on both sides.

## Uncertainty-mode input file and controls

The Uncertainty mode is detected from the presence of an `[UNCERTAINTY]` section appended to a complete Run deck. The deck supplies the design point, and therefore the central value of every law declared below, so that a study re-centres itself automatically when the design changes. A second section, `[CONTROLS]`, governs the propagation.

Under `[UNCERTAINTY]`, a parameter is made uncertain by giving it a distribution in place of a value, the parameters left out keeping their deck value. Three continuous families are available. The triangular law is written `tri(lo, hi)`, whose mode is then the design value, or `tri(lo, mode, hi)` with the mode set explicitly. The uniform law `unif(lo, hi)` is reserved for conventions with no preferred value. The normal law is truncated and accepts one to five arguments: `norm(sigma)` centres on the design value without bounds. `norm(lo, hi)` centres on the design value and sets $\sigma = (hi-lo)/4$, so that the bounds sit at about the mean plus or minus two standard deviations, exactly so when the design value is their midpoint. `norm(lo, centre, hi)` sets the centre explicitly, which is used whenever the design value is not the best estimate or the uncertainty is asymmetric. `norm(lo, centre, hi, sigma)` decouples the width from the bounds, for a narrow peak with a far-reaching but rare tail, and `norm(lo, centre, hi, s_lo, s_hi)` gives a split, two-piece normal with a sharp side and a long tail. A centre lying outside its own bounds is clamped, with a warning.

Model-form choices are declared instead as `envelope(A | B | C)`, a discrete list of options for a configuration switch such as the confinement scaling law or the elongation closure. Envelopes are not sampled: every combination of every declared envelope is enumerated as a full factorial and each combination receives the whole sample budget, so that $n$ envelopes of $m_i$ members produce $\prod_i m_i$ independent studies that can be compared side by side. The same Latin-hypercube draw matrix is reused for every combination, so the envelope members are compared on paired samples.

Under `[CONTROLS]`, `n_samples` is the number of draws per envelope combination and `seed` fixes the pseudo-random stream, so that a run is exactly reproducible. The draws are taken by Latin-hypercube sampling of the declared marginals, which for a few hundred draws covers the input space far more evenly than plain Monte-Carlo. Two optional operator levers may then be activated. Setting `retune = Tbar` allows a draw that fails at the deck temperature to be re-evaluated along a short temperature ladder inside the window given by `Tbar_window`, at the resolution set by `retune_points`. The direction of the search follows the failing constraint, since the density and beta limits call for a higher temperature and the kink limit and flux closure for a lower one. Setting `cs_relief = 0.25, 0.5, 0.75` allows a draw that closes structurally but exhausts the central-solenoid inductive flux to shed the listed fractions of that flux in turn, through a heated current ramp-up or a shorter flat-top, the smallest relief that closes being the one reported. The related `pulse_retune`, `pulse_fractions` and `pulse_floor` keys expose the flat-top shortening directly.

Every draw is evaluated through the same feasibility test as the Scan and Optimization modes, namely the Greenwald, Troyon and kink limits together with radial-build closure, so that “feasible” carries exactly the meaning it has for the optimiser. A draw that is not feasible is classified into one of four families: `radial_build` when the build or the central solenoid does not close, `stability` when a plasma limit binds, `no_operating_point` when the inverse plasma solve finds no root anywhere in the admissible temperature window, and `crash` when the evaluation raises. The last two are non-converged outcomes and are reported separately rather than folded into the infeasible count, since a draw with no operating point is a different statement from a design that violates a limit.

## Conventions and calibration

Three kinds of uncertain input are distinguished, and the distribution family follows the kind. Physics quantities with a documented statistical scatter take a truncated normal whose width is tied to the published RMSE or database spread. Operational limits whose position is empirically bracketed take a truncated normal between the pessimistic and optimistic experimental anchors. Modelling conventions with no measurable true value take a uniform interval or a discrete envelope, because sampling a convention as a normal would assert a knowledge that does not exist. The width scale is informed by the PROCESS uncertainty studies {footcite:p}`lux2017uncertainties,kemp2017dealing` and by the EU-DEMO robust-design work {footcite:p}`coleman2025definition`. The PROCESS studies sample widths of the same order as the ones adopted here, for instance a confinement multiplier of $\sigma \approx 0.1$, a density-limit factor of $\sigma \approx 0.1$ and a radiation core radius of $0.6 \pm 0.15$, and they describe their own distributions as “mainly educated guesses” {footcite:p}`kemp2017dealing`. The same reading applies below: each width rests on a documented anchor where one exists, and is an explicit assumption where none does. One stance is adopted throughout: the study evaluates an optimised, repeated scenario, not a blind draw from the multi-machine databases. The laws on the scenario-quality inputs ($H_{98}$, Greenwald fraction, $C_e$) are therefore mildly skewed towards the optimised side.

## Adopted default distributions

The full adopted set for the ITER reference study is collected in {numref}`Tables %s <tab:uq_presets>` and {numref}`%s <tab:uq_presets_eng>`, each entry carrying its distribution, its physical anchor and its sources. The sampled marginals are drawn in {numref}`Figure %s <fig:chap4_uq_inputs>`. The condensed summary is {numref}`Table %s <tab:uq_presets_summary>` of Section {ref}`The D0FUS uncertainty mode <ssec:chap4_presets>`. The derivation of every entry follows, family by family.

:::{table} Adopted default distributions of the uncertain plasma-physics inputs for the ITER reference study. Central values are the design-deck values unless the three-argument form sets them explicitly. The limits, flux budget and techno-economic inputs are continued in {numref}`Table %s <tab:uq_presets_eng>`.
:name: tab:uq_presets
:align: center

| Input                         | Distribution                                                | Physical anchor                                                                | Source       |
|:------------------------------|:------------------------------------------------------------|:-------------------------------------------------------------------------------|:-------------|
| *Confinement*                 |                                                             |                                                                                |              |
| $H_{98}$                      | norm(0.80, 1.60)                                            | IPB98 RMSE $\sim$15%, degraded to improved H-mode                              | {footcite:p}`iterphysicsbasis1999,doyle2007plasma` |
| scaling law                   | env(IPB98 $|$ ITPA20 $|$ ITPA20-IL)                         | $\tau_E$ ratios 1.00 / 0.85 / 0.80 at the ITER point                           | {footcite:p}`verdoolaege2021itpa` |
| *Profiles and pedestal*       |                                                             |                                                                                |              |
| $\nu_n$ (density peaking)     | norm(0.00, 0.40)                                            | flat baseline ($\hat n_0 = 1.04$) to moderate peaking ($\hat n_0 \approx 1.2$) | {footcite:p}`kim2018iter,angioni2007jetpeaking` |
| $\nu_T$ (temp. peaking)       | norm(2.20, 3.40)                                            | stiff profiles, $T_{e0}/\langle T_e\rangle$ 2.3 to 2.9                         | {footcite:p}`kessel2009development,rodriguez2020predictions` |
| $\rho_{ped}$                  | norm(0.92, 0.98)                                            | EPED width, ITER $\psi_{N,ped}$ 0.95 to 0.96                                   | {footcite:p}`snyder2009pedestal,snyder2011eped` |
| $n_{ped}/\langle n\rangle$    | norm(0.75, 1.00)                                            | peaked (0.85) to flat (0.99) pedestal density                                  | {footcite:p}`polevoi2005pellet,kim2018iter` |
| $T_{ped}/\langle T\rangle$    | norm(0.42, 0.68)                                            | EPED $T_{ped}$ 4 to 5.5 keV, widened by 15 to 20%                              | {footcite:p}`snyder2011eped` |
| *Helium and composition*      |                                                             |                                                                                |              |
| $C_\alpha=\tau_{He}^*/\tau_E$ | norm(3.0, 5.7, 10.6)                                        | deck nominal 5.7, ITER design assumption 5, DIII-D measured 10 to 20           | {footcite:p}`kovari2014process,wade1995helium,lux2017uncertainties` |
| core W fraction               | norm($10^{-5}$, $2\!\times\!10^{-5}$, $8\!\times\!10^{-5}$) | tolerable W, H-mode lost above $\sim\!5\!\times\!10^{-5}$                      | {footcite:p}`kim2018iter` |
| $r_\mathrm{synch}$            | norm(0.50, 0.80)                                            | metallic-wall band, Albajar-Fidone $(1-r)^{0.62}$ as implemented               | {footcite:p}`albajar2001synchrotron` |
| $\rho_\mathrm{rad,core}$      | norm(0.60, 0.85, 1.00)                                      | ITER-calibrated 0.85, PROCESS 0.6 to conservative 1.0                          | {footcite:p}`kovari2014process` |
:::

:::{table} Adopted default distributions of the stability limits, flux-budget and techno-economic inputs for the ITER reference study, continuing {numref}`Table %s <tab:uq_presets>`. Model-form choices are sampled as discrete envelopes rather than as probability laws.
:name: tab:uq_presets_eng
:align: center

| Input                           | Distribution                               | Physical anchor                                                                                          | Source       |
|:--------------------------------|:-------------------------------------------|:---------------------------------------------------------------------------------------------------------|:-------------|
| *Stability limits*              |                                            |                                                                                                          |              |
| $\beta_N$ limit                 | norm(2.60, 3.50)                           | Troyon 2.8, operational $\sim$3.5, no-wall $n{=}1$ limit 3.1 for the reactor shape                       | {footcite:p}`troyon1984mhd,hender2007mhd,wenninger2015advances` |
| $q_{95}$ limit                  | norm(2.00, 2.50, 3.00)                     | hard kink 2, JET disruptivity boundary $\sim$2.5, ITER design 3                                          | {footcite:p}`devries2009statistical,wenninger2017physics` |
| Greenwald limit                 | norm(0.80, 1.50)                           | H-mode 0.8 to 1.0, peaked/pellet fuelling beyond                                                         | {footcite:p}`greenwald2002density,lang2012highdensity,ding2024high` |
| elongation closure              | env(Wenn. $|$ Freid. $|$ Blend $|$ Stamb.) | $\kappa_\mathrm{sep}(A\!=\!3.1)$ 1.88 to 2.30, same four members for ITER and the harmonised plant study | {footcite:p}`wenninger2015advances,freidberg2015tokamak,stambaugh1992relation` |
| *Flux budget and current drive* |                                            |                                                                                                          |              |
| $C_e$ (Ejima)                   | norm(0.25, 0.45, 0.58)                     | ITER design 0.45, heated ramp-up below 0.27                                                              | {footcite:p}`ejima1982volt,wakatsuki2019safety` |
| $\eta_\mathrm{WP}$              | norm(0.05, 0.30, 0.45)                     | NBI 0.25 to 0.28, EC $\sim$0.35, mix $\sim$0.29                                                          | {footcite:p}`franke2017heating,fantz2018towards` |
| $\gamma_\mathrm{CD}$            | norm(0.10, 0.35)                           | ITER EC $\sim$0.2, NB $\sim$0.3, survey band                                                             | {footcite:p}`poli2013eccd,mikkelsen2018survey` |
| *Techno-economics*              |                                            |                                                                                                          |              |
| SC cost factor                  | norm(1.20, 3.00)                           | assumption, informed by unit-cost literature                                                             | {footcite:p}`sheffield2016generic,molodyk2021production` |
| discount rate                   | norm(0.04, 0.10)                           | 7% real, fixed charge rate 0.078                                                                         | {footcite:p}`entler2018approximation,sheffield2016generic` |
:::

:::{figure} /figures/thesis/uq_input_distributions.png
:name: fig:chap4_uq_inputs
:width: 92%
:align: center

Sampled marginals of the uncertain inputs for the ITER reference study, grouped by family. Each panel shows the drawn histogram, the analytic marginal (black) and the design value (red dashed). The two model-form switches (confinement scaling and elongation closure) are equal-weight categorical panels. The distributions realise {numref}`Tables %s <tab:uq_presets>` and {numref}`%s <tab:uq_presets_eng>`.
:::

## Confinement

### Confinement multiplier $H_{98}$, norm(0.80, 1.60).

The IPB98(y,2) regression carries an RMSE near $15\,\%$ {footcite:p}`iterphysicsbasis1999`, and the 2021 ITPA update gives a $95\,\%$ interval of $-20$ to $+25\,\%$ on the ITER prediction {footcite:p}`verdoolaege2021itpa`. The law is centred on $H_{98} = 1$, with a lower anchor at 0.80 for a degraded excursion and an upper anchor at 1.60 for the improved-confinement regimes reached on several devices {footcite:p}`doyle2007plasma`. The law is asymmetric: with $\sigma = 0.20$, the lower anchor sits one standard deviation below the mode and the upper anchor three above it. The model-form part of this uncertainty is carried separately by the scaling-law envelope.

### Scaling-law envelope, IPB98(y,2) $|$ ITPA20 $|$ ITPA20-IL.

At the ITER reference point, the published exponents give 3.62, 3.07 and 2.90 s respectively {footcite:p}`verdoolaege2021itpa`. This is a discrete choice rather than a samplable scatter, so it is enumerated as a full factorial. The ITER-like variant is kept because it is the one built to resemble ITER and gives the most pessimistic prediction.

### Operating temperature.

The volume-averaged temperature is an operating choice, not a physics uncertainty, so it is not sampled. At fixed fusion power it is set through density control, and its residual spread is handled by the operator-retuning search of the controls block.

## Profiles and pedestal

### Density peaking $\nu_n$, norm(0.00, 0.40).

The ITER baseline modelling uses a nearly flat density, $n_{e0}/\langle n_e\rangle = 1.04$, and scans peaking up to 1.31 {footcite:p}`kim2018iter`. The collisionality scaling predicts about 1.45 at ITER parameters {footcite:p}`angioni2007jetpeaking`. The law is centred on the deck value 0.01. With the pedestal overlay at its deck values, the sampled band spans $n_{e0}/\langle n_e\rangle \approx 1.05$ to 1.08. Jointly with the sampled pedestal-density fraction, it reaches about 1.20. The strongest predicted peaking therefore lies above the sampled range. The band is a flat-to-moderately-peaked assumption, matching the gas-fuelled ITER baseline rather than the collisionality prediction considered by the author as too optimistic.

### Temperature peaking $\nu_T$, norm(2.20, 3.40).

Temperature profiles are stiff, so the peaking varies little across H-mode databases. The ITER anchor is $T_{e0}/\langle T_e\rangle \approx 2.6$ {footcite:p}`kessel2009development,kim2018iter` and burning-plasma predictions use 2.5 to 2.7 {footcite:p}`rodriguez2020predictions`. The law is centred on 2.8 with $\sigma = 0.3$. With the deck pedestal, the band maps to $T_{e0}/\langle T_e\rangle = 2.24$ to 2.88, and the centre reproduces the ITER anchor ($T_{e0}/\langle T_e\rangle = 2.56$).

### Pedestal position $\rho_{ped}$, norm(0.92, 0.98).

The ITER scenario modelling places the pedestal top at $\rho \approx 0.95$ {footcite:p}`kim2018iter,polevoi2005pellet`, and the EPED width scaling between $\psi_N = 0.95$ and 0.96 {footcite:p}`snyder2009pedestal,snyder2011eped`. The band brackets the deck value 0.95, its lower bound admitting the wider pedestals obtained when that scaling is degraded.

### Pedestal density fraction, norm(0.75, 1.00).

ITER modelling places $n_{ped}$ at 0.85 to 0.90 of the volume average for the pellet-fuelled cases {footcite:p}`polevoi2005pellet` and at 0.99 for the flat-density baseline {footcite:p}`kim2018iter`. The truncation at unity excludes a hollow core.

### Pedestal temperature fraction, norm(0.42, 0.68).

EPED-class predictions span 4 to 5.5 keV against $\langle T\rangle \approx 8.9$ keV, that is 0.45 to 0.62, with a reported accuracy of 15 to $20\,\%$ {footcite:p}`snyder2011eped`. The band widens that span to 0.42 to 0.68, a margin of 7 to 10 % on each side, within the reported accuracy.

## Helium and composition

### Helium-ash ratio $C_\alpha = \tau_{He}^*/\tau_E$, norm(3.0, 5.7, 10.6).

The nominal 5.7 is the value reproducing the ITER helium projection under the impurity-diluted ash balance, close to the ITER design assumption of 5 {footcite:p}`kovari2014process`. The law is right-skewed, a longer ash residence being the pessimistic direction: its upper reach approaches the 10 to 20 $\tau_E$ measured in DIII-D H-modes with argon-frosted cryopumping {footcite:p}`wade1995helium`. The EU-DEMO uncertainty studies span 6.5 to 12.6 in the same ratio, overlapping the upper half of this band {footcite:p}`lux2017uncertainties`. The sensitivity of a burning plasma to this ratio is discussed by Reiter {footcite:p}`reiter1990burn`.

### Core tungsten fraction, norm($10^{-5}$, $2\times10^{-5}$, $8\times10^{-5}$).

Reliable H-mode access requires a concentration near $10^{-5}$ and becomes marginal at 2 to $3\times10^{-5}$ in the ITER integrated modelling {footcite:p}`kim2018iter`, in line with the tolerable-concentration surveys {footcite:p}`putterich2019tolerable`. Integrated modelling further reports H-mode lost above about $5\times10^{-5}$. For scale, the EU-DEMO studies sample $10^{-4} \pm 5\times10^{-5}$ for a full-tungsten reactor {footcite:p}`kemp2017dealing`. The upper tail therefore probes genuine risk. Sampling the inventory rather than $Z_\mathrm{eff}$ keeps dilution, radiation and effective charge mutually consistent.

### Wall reflectivity $r_\mathrm{synch}$, norm(0.50, 0.80).

Metallic-wall reflectivity spans roughly 0.5 to 0.8, and the correction enters as $(1-r)^{0.62}$ in the D0FUS implementation {footcite:p}`albajar2001synchrotron`. PROCESS leaves it as a user input, 0.6 being common practice {footcite:p}`kovari2014process`. The law is centred on the deck value 0.6. Minor at ITER temperatures, this input grows relevant at reactor temperatures.

### Core/edge radiation split $\rho_\mathrm{rad,core}$, norm(0.60, 0.85, 1.00).

This is a bookkeeping convention. PROCESS counts radiation inside the normalised radius 0.6 against the confined-power balance, the conservative reading subtracts everything up to 1.0 {footcite:p}`kovari2014process`, and the centre 0.85 is the value retained by the calibrated ITER deck ({ref}`D0FUS ITER benchmark inputs <appendixinput_iter>`). The PROCESS uncertainty studies vary the same radius as $0.6 \pm 0.15$ {footcite:p}`lux2017uncertainties`. On the ITER deck, moving the split from 0.6 to 0.85 raises the required plasma current by 6 %.

## Stability limits

### Normalised beta limit, norm(2.60, 3.50).

The Troyon coefficient is 2.8 {footcite:p}`troyon1984mhd`. Shaped plasmas reach $\beta_N \approx 3.5$ to 4 {footcite:p}`hender2007mhd`, and the computed no-wall $n=1$ limit for the reactor shape is 3.1 {footcite:p}`wenninger2015advances`. The lower bound is kept tight at 2.60 because operating below the Troyon coefficient is improbable for an optimised scenario, so most of the mass sits at 2.8 and above. The band stops below 4, which would require active resistive-wall-mode control.

### Kink limit on $q_{95}$, norm(2.00, 2.50, 3.00).

The hard limit is $q_{95} = 2$, the $m=2/n=1$ kink. JET statistics show the disruptivity rising below 2.5 and flat above 3.5 {footcite:p}`devries2009statistical`, and ITER and EU-DEMO adopt 3.0 to 3.2 as design margins {footcite:p}`wenninger2017physics`. The law is centred on 2.5, between the two. It is applied verbatim to the power-plant study, so that both are held to the same kink standard (Section 4.4.2 of the thesis).

### Greenwald limit, norm(0.80, 1.50).

H-mode density limits with flat gas fuelling cluster at 0.8 to 1.0 of the Greenwald fraction {footcite:p}`greenwald2002density`. Pellet-fuelled peaked profiles reach up to 1.5, at the price of degraded confinement {footcite:p}`lang2012highdensity`. The high-poloidal-beta path sustains 1.2 at full confinement {footcite:p}`ding2024high`. The band is centred on the deck value 1.0.

### Elongation closure, Wenninger $|$ Freidberg $|$ Blend $|$ Stambaugh.

The maximum achievable elongation is design-shaping but uncertain, so it is treated as a model-form choice rather than a marginal. At $A = 3.1$ the Wenninger, Blend and Freidberg closures give 1.88, 1.89 and 1.97, bracketing the built ITER value of 1.85, while Stambaugh gives 2.30 {footcite:p}`wenninger2015advances,freidberg2015tokamak,stambaugh1992relation`. The Stambaugh member extrapolates machines whose passive structure sits close to the plasma, so it is optimistic for a blanket-separated power plant. The vertical-stability margin $m_s$ is not sampled separately, to avoid counting the same uncertainty twice.

## Flux budget and current drive

### Ejima coefficient $C_e$, norm(0.25, 0.45, 0.58).

The value 0.45 is the ITER design assumption {footcite:p}`iterphysicsbasis1999ch8`. It follows from the $2.0 \pm 0.2$ V s/MA reported at flat-top with about $40\,\%$ consumed resistively, that is $0.8/(\mu_0 R_0) \approx 0.45$ for the Doublet III major radius $R_0 = 1.43$ m {footcite:p}`ejima1982volt`. PROCESS defaults to 0.4 {footcite:p}`kovari2014process`, and heated ramp-up simulations reach below 0.27 {footcite:p}`wakatsuki2019safety`. The law is centred on 0.45, its lower reach crediting the heating assist. The upper reach 0.58 in an author arbitrary pessimistic value.

### Wall-plug efficiency $\eta_\mathrm{WP}$, norm(0.05, 0.30, 0.45).

ITER-generation systems give 0.25 to 0.28 for negative-ion neutral beams and about 0.35 for electron cyclotron, a power-weighted mix near 0.29 {footcite:p}`franke2017heating,fantz2018towards`, against 0.4 for DEMO {footcite:p}`federici2019demo`. The long lower tail down to 0.05 is a deliberately pessimistic floor for a poorly converting heating mix. It is an assumption. This input affects the recirculating power, not the plasma-limit feasibility.

### Current-drive figure of merit $\gamma_\mathrm{CD}$, norm(0.10, 0.35).

ITER figures of merit are near 0.2 for electron cyclotron and 0.3 for neutral beams {footcite:p}`poli2013eccd`, and reactor surveys span 0.15 to 0.55 across technologies {footcite:p}`mikkelsen2018survey`. The law is centred on 0.20. Its upper bound stops below the most favourable survey entries, which rely on technologies not modelled here.

## Techno-economics and power conversion

### Superconductor cost factor, norm(1.20, 3.00).

No published copper-to-superconductor multiplier exists. The preset is a modelling assumption, informed by generic unit costs {footcite:p}`sheffield2016generic`, first-of-a-kind REBCO tape quotes {footcite:p}`sorbom2015arc` and the tape learning curve towards tens of dollars per kA m {footcite:p}`molodyk2021production`. It is centred on 2.0 and should be read as an assumption rather than as a literature value.

### Discount rate, norm(0.04, 0.10).

Fusion cost studies use a $7\,\%$ real rate {footcite:p}`entler2018approximation` and the Sheffield model a fixed charge rate of 0.078 {footcite:p}`sheffield2016generic`. The law is centred on 0.07.

The thermal-to-electric efficiency and the blanket multiplication become uncertain as well in a power-plant context, spanning 0.31 to 0.45 depending on the coolant {footcite:p}`kovari2016process` and 1.2 to 1.3 respectively {footcite:p}`federici2019demo`. Both are held fixed in the ITER study, which is not a power plant.

## Inputs held fixed

Several candidates were considered and set aside. An L-H threshold multiplier is not sampled because H-mode access does not enter the feasibility verdict, which rests on the Greenwald, Troyon, kink and radial-build constraints. It is monitored as a reported margin instead, the Martin threshold carrying a scatter of about $31\,\%$ {footcite:p}`martin2008power`. The usable flux-swing fraction is an ITER-practice allocation with no citable uncertainty statement. The structural safety factors are calibrated against the ITER benchmark, and sampling them would break that anchoring. The kink parameter, $q_{95}$ against $q^*$, is a convention: changing it mid-study would change the meaning of the threshold.

```{rubric} References
```

```{footbibliography}
```
