(ssec:chap1_supraconductors)=

# Superconductors and engineering current density

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.3.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The radial build models of the following sections take as input the engineering current density $J^\mathrm{wost}$ (“without steel”), defined on the non-steel cross-section of the conductor. It derives from the superconductor critical current density, which falls as the field $B$ in which the conductor sits increases and as its temperature $T$ rises, combined with the conductor composition (copper, helium and insulation fractions) detailed below. The relevant quantity is thus $J^\mathrm{wost}(B,T)$, used as such in the coil sizing discussed below.

## Hierarchy of current density definitions

The current density in a superconducting magnet can be defined at several scales and is a frequent source of confusion in the literature. Going from the smallest to the largest reference cross-section, illustrated in {numref}`Fig. %s <fig:Jhierarchy>`:

- $J_\mathrm{SC}$: intrinsic critical current density of the superconducting material, defined on the superconducting cross-section alone (the filaments for LTS, the REBCO layer for HTS).

- $J_{\text{non-Cu}}$: critical current density on the non-copper cross-section, which adds the strand matrix for Nb$_3$Sn or the substrate and buffer layers for REBCO. This is the quantity returned by the scaling laws and the one most commonly tabulated against $B$ and $T$ by manufacturers.

- $J_\mathrm{strand}$ (or $J_\mathrm{tape}$): strand current density, which adds the copper stabiliser carried within the strand (or tape).

- $J^\mathrm{wost}$: engineering current density “without steel”, defined on the full CICC cross-section minus the steel jacket. It includes the strands, the interstrand helium void, the dedicated helium cooling channel, and the electrical insulation.

- $J_\mathrm{coil}$: overall coil current density, defined on the full coil cross-section including the steel jacket, the structural case and potentially extra insulation.

:::{figure} /figures/thesis/fig_Jhierarchy_composite.png
:name: fig:Jhierarchy
:width: 95%
:align: center

Hierarchy of current density definitions in a low-temperature superconducting CICC, from the strand-level superconducting filaments (a) up to the full TF coil inboard leg (c). At each level, a fraction of the cross-section is dedicated to non-current-carrying material (copper stabiliser, helium, insulation, structural steel), so that the current density progressively decreases as the reference surface grows. The quantity used in D0FUS is $J^\mathrm{wost}$, defined on the CICC non-steel cross-section (everything inside the steel jacket of (b)). (a) SC strand: superconducting filaments (green) embedded in a copper matrix (orange). $J_\mathrm{SC}$ is defined on the filament cross-section only; (b) CICC: SC and Cu strands cabled together around a central He pipe, jacketed in steel. $J^\mathrm{wost}$ is defined on everything inside the steel jacket; (c) TF coil inboard leg: a winding pack of CICC turns held by a thick steel case. $J_\mathrm{coil}$ is defined on the full cross-section, steel case included.
:::

The scaling laws used for $J_{\text{non-Cu}}$ differ between conductor technologies. D0FUS implements the ITER/EU-DEMO NbTi parameterisation {footcite:p}`corato2016common`, the EU-DEMO WST Nb$_3$Sn scaling {footcite:p}`bottura2009jc,corato2016common`, and for REBCO the Senatore *et al.* (2024) pinning-force scaling {footcite:p}`senatore2024rebco` calibrated on modern tapes, with the older Fleiter/CERN (2014) parameterisation {footcite:p}`fleiter2014rebco,bajas2022ship` also available.

Evaluated at 4.2 K, the three scalings cover complementary field ranges: NbTi up to about 9 T, Nb$_3$Sn up to about 14 T, and REBCO well beyond 20 T, as illustrated by the measured-data comparison of {numref}`Fig. %s <fig:Jc_scaling>` in the validation chapter. The parameterisations and their coefficient tables are given in {ref}`Superconductor critical-current scalings and quench protection <app:jc_scalings>`. Their benchmark against experimental reference data and ITER conductor specifications is reported in the validation chapter (Section {ref}`Superconductor critical-current benchmark <ssec:chap2_Jc_bench>`).

## Cable-space dilution and $J^\mathrm{wost}$

The non-steel cross-section of a CICC is decomposed hierarchically. A fraction $f_\mathrm{In}$ is occupied by the electrical insulation and $f_\mathrm{CC}$ by a dedicated helium cooling channel running along the conductor centre, the remainder forming the strand (or tape) bundle. Inside the bundle, the strands (or tapes) are packed with an interstitial helium void fraction $f_\mathrm{void}$, typically $\approx 0.33$ for round LTS strands and $\approx 0$ for stacked HTS tapes following the PIT-VIPER design {footcite:p}`Sanabria2024PITVIPER`. The copper stabiliser can be present both inside the strands (the matrix surrounding the filaments) and as separate pure-copper strands. All of this stabiliser is accounted for through the copper-to-non-copper ratio $r_{\text{Cu/non-Cu}} = S_\mathrm{Cu}/S_{\text{non-Cu}}$. The engineering current density is then obtained by combining the successive dilution factors:

$$
J^\mathrm{wost} = J_{\text{non-Cu}} \times f_{\text{non-Cu}} \times (1 - f_\mathrm{void}) \times (1 - f_\mathrm{In} - f_\mathrm{CC})
$$ (eq:Jwost)

where $f_{\text{non-Cu}} = 1/(1 + r_{\text{Cu/non-Cu}})$ is the non-copper fraction of a strand. For HTS windings, the same decomposition applies unchanged, reading tape wherever strand is written.

(sssec:chap1_quench)=

## Quench protection and copper fraction

The copper-to-non-copper ratio $r_{\text{Cu/non-Cu}}$ entering Eq. {eq}`eq:Jwost` is not a free parameter: it is fixed by quench protection. When the superconductor quenches, its current transfers to the copper stabiliser, which must carry it resistively for the few seconds needed to detect the event and discharge the coil, without the local hot spot exceeding an allowable temperature. D0FUS enforces this through the Maddock adiabatic hot-spot criterion {footcite:p}`maddock1969,wilson1983superconducting`, balancing the Joule heating in the copper against the enthalpy rise of the composite conductor, with the discharge time constant computed from the analytically estimated magnet stored energy and an ITER-like protection scheme (10 kV dump voltage, paired TF dump units, six CS modules) {footcite:p}`sborchia2008design`. The module is adapted from the CEA magnet design code MADMACS {footcite:p}`zani2019parametric,sutcliffe2025magnet`. Since the discharge time constant and the copper properties depend on the coil and on the design point, this ratio is recomputed for each configuration rather than kept constant. The criterion and the default protection parameters are detailed in {ref}`Superconductor critical-current scalings and quench protection <app:jc_scalings>`.

(sssec:chap1_margins)=

## Operating margins

The critical current density is not evaluated at the nominal helium temperature but at a conservative design temperature that builds in a stability margin. Three temperatures are distinguished:

- $T_\mathrm{He} = 4.2$ K, the nominal helium temperature at saturation (1 bar).

- $T_\mathrm{op}$, the actual operating temperature of the conductor. Because the CICC channels are fed with supercritical helium at $\sim10$ bar rather than at 1 bar, $T_\mathrm{op}$ sits a few tenths of a kelvin above $T_\mathrm{He}$ (typically 0.3 K).

- $T_\mathrm{calc} = T_\mathrm{op} + \Delta T_\mathrm{margin}$, the design temperature at which $J_{\text{non-Cu}}$ is evaluated for conductor sizing. Following EU-DEMO design rules {footcite:p}`corato2016common` and MADMACS conventions, the default values are $\Delta T_\mathrm{margin} = 1.7$ K for NbTi, 1.5 K for Nb$_3$Sn, and 5.0 K for REBCO.

```{rubric} References
```

```{footbibliography}
```
