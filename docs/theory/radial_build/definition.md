(ssec:chap1_radial_build)=

# Radial build definition and geometry

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.3.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The inboard radial build, illustrated in {numref}`Fig. %s <fig:radial_build_clean>`, is a schematic section of a tokamak at the midplane ($z = 0$). From the plasma edge inward, the components are:

1.  **Plasma**: minor radius $a$ \[m\], centred at the major radius $R_0$ (in red).

2.  **First wall, breeding blanket, neutron shield, vacuum vessel and assembly gaps**: their combined inboard radial thickness is denoted $\Delta_B$ \[m\], with a default total of 1.2 m considered reasonable for power-plant studies {footcite:p}`freidberg2015designing,kovari2016process`. Only this lumped value enters the radial build physics, its decomposition into sub-components is resolved by the breeding-blanket model described below, which sets the breeder-zone thickness, the energy multiplication and the tritium breeding ratio without changing $\Delta_B$ (this resolution serves to check the tritium self-sufficiency and to feed the plant power balance, as detailed below).

3.  **TF coil inner leg**: total radial thickness $\Delta_\mathrm{TF}$ \[m\], decomposed into a winding pack (WP) of thickness $\Delta_\mathrm{WP}$ and, in the wedging configuration, a steel nose of thickness $\Delta_\mathrm{nose}$. $\Delta_\mathrm{TF}$ is an output of the code, determined by the magnetic field generation and from considerations on the mechanical stress detailed in Sections {ref}`Academic model <ssec:chap1_academic>` and {ref}`Refined model <ssec:chap1_refined>`.

4.  **Gap** (wedging only): a clearance $g$ \[m\] between the TF inner bore and the CS outer face. The default value is $g = 0.10$ m. In the bucking and plug configurations, the TF coils and the CS are assumed to be in mechanical contact and $g = 0$ (no bucking cylinder is considered).

5.  **Central solenoid**: winding pack thickness $\Delta_\mathrm{CS}$ \[m\]. This is also an output of the code.

The three key radii of the TF inboard leg, used throughout the rest of the chapter, are defined as

$$
\begin{aligned}
R_\mathrm{TF}^\mathrm{ext} &= R_0 - a - \Delta_B          & &\text{(TF outer face)} \\
	R_\mathrm{TF}^\mathrm{sep} &= R_\mathrm{TF}^\mathrm{ext} - \Delta_\mathrm{WP} & &\text{(WP / nose interface)} \\
	R_\mathrm{TF}^\mathrm{int} &= R_\mathrm{TF}^\mathrm{ext} - \Delta_\mathrm{TF}  & &\text{(TF inner bore)}
\end{aligned}
$$ (eq:radii_TF)

The corresponding three radii of the CS are $R_\mathrm{CS}^\mathrm{ext}$ (CS outer face), $R_\mathrm{CS}^\mathrm{sep}$ (the radius separating the conductor-carrying layer from the structural layer in the simplified model of Section {ref}`Academic model <ssec:chap1_academic>`) and $R_\mathrm{CS}^\mathrm{int}$ (CS inner bore). They satisfy

$$
R_\mathrm{CS}^\mathrm{ext} = R_\mathrm{TF}^\mathrm{int} - g, \qquad R_\mathrm{CS}^\mathrm{int} = R_\mathrm{CS}^\mathrm{ext} - \Delta_\mathrm{CS}
$$ (eq:radii_CS)

The objective of the radial build criterion is to fit all these thicknesses into the most spatially constrained region of the tokamak: the high-field (inboard, see {numref}`Fig. %s <fig:radial_build_clean>`) side.

:::{figure} /figures/thesis/Radial_Build_Clean.png
:name: fig:radial_build_clean
:width: 85%
:align: center

Typical inboard radial build in D0FUS, here in the wedging configuration. The plasma minor radius $a$ and the aggregated inboard shield thickness $\Delta_B$ are user inputs, the TF coil thickness $\Delta_\mathrm{TF}$ and the central solenoid thickness $\Delta_\mathrm{CS}$ are determined self-consistently by D0FUS. The $\theta$ of the coordinate triad is the azimuthal direction about the machine axis, not the poloidal angle.
:::

:::{figure} /figures/thesis/Radial_Build_Assembly.png
:name: fig:radial_build_assembly
:width: 68%
:align: center

Full poloidal radial build assembled by D0FUS for a converged design point (here the ITER baseline), from the plasma boundary out to the TF coil. In contrast with the inboard schematic of {numref}`Fig. %s <fig:radial_build_clean>`, this figure is produced automatically by the code and stacks the breeding-blanket layers of the selected concept, the central solenoid and the TF inner leg at their computed thicknesses, providing an at-a-glance check that the high-field side closes.
:::

The separation between inputs and outputs is a deliberate choice. The plasma geometry ($R_0$, $a$) and the inboard shielding thickness ($\Delta_B$) are prescribed by the user, while the coil thicknesses ($\Delta_\mathrm{TF}$, $\Delta_\mathrm{CS}$) are computed by the engineering models. Treating $\Delta_B$ as an input is partly a matter of necessity: sizing the shield from first principles requires heavy neutron-transport calculations, beyond the scope of a 0D system code. It is also supported by published practice: power-plant studies converge on inboard stacks of about 1.2 m {footcite:p}`freidberg2015designing,kovari2016process`, and once the shielding and breeding technology is fixed this thickness is expected to vary little between design points. On the contrary, the coil thicknesses follow directly from the structural response to the electromagnetic loads and must be recomputed for each configuration.

The default value comes from a slab slowing-down and breeding calculation by Freidberg *et al.* {footcite:p}`freidberg2015designing`, reproduced and detailed in {ref}`The inboard blanket thickness estimate of Freidberg et al. <app:blanket_freidberg>`: about $0.9$ m of natural lithium suffices for an essentially complete burn-up of the fusion neutrons, and adding the first wall, multiplier, shield and vacuum vessel brings the total to $b \approx 1.2$ m, the default adopted here. The result supports treating $\Delta_B$ as a design-independent constant, since the required thickness depends on the attenuation target only through a double logarithm and on the neutron flux not at all, the neutron energy alone entering at first order. The slab picture is however geometrically optimistic, as it lets the neutrons slow down along straight lines.

## Breeding-blanket resolution

The breeding blanket, which occupies most of $\Delta_B$, fulfils four functions at once: shielding the vacuum vessel and the coils from the fusion neutrons (alongside the dedicated neutron shield), multiplying these neutrons, breeding from them the tritium that the plant burns, and extracting the energy they carry towards the power conversion cycle. While $\Delta_B$ is kept as the master radial build input, the way it is filled is therefore not arbitrary, and D0FUS resolves the inboard stack into the physical layers of a chosen breeding-blanket concept, in an implementation contributed by Mattéo Fletcher (private communication). The purpose of this resolution is twofold: estimating the tritium breeding ratio, and replacing the generic power-balance defaults by concept-consistent values, as detailed below. Six concepts are available (the helium-cooled pebble bed HCPB, taken as default, the helium- and dual-coolant lithium-lead blankets HCLL and DCLL, and three self-cooled variants), with their layer breakdowns and material compositions digitised from the Infinity Two pilot-plant blanket trade-off study {footcite:p}`clark2025blanket`. Each concept fixes the widths of the first wall, the back structure, the shields, the vacuum vessel and the assembly gaps, and the breeder zone takes up whatever inboard space remains, $\delta_\mathrm{BB}^\mathrm{ib} = \Delta_B - \sum \delta_\mathrm{fixed}$, so that the lumped $\Delta_B$ is preserved exactly: for the default HCPB concept at $\Delta_B = 1.2$ m this leaves a $0.32$ m inboard breeder zone.

The selected concept also carries the two quantities that enter the plant power balance, the energy multiplication $M_\mathrm{blanket}$ and the thermal-to-electric efficiency $\eta_T$, used in Eqs. {eq}`eq:Pth` and {eq}`eq:Pelec` in place of the generic defaults (HCPB: $M_\mathrm{blanket} = 1.35$, $\eta_T = 0.35$). Finally, the inboard breeder thickness feeds a saturating estimate of the tritium breeding ratio (TBR),

$$
\mathrm{TBR}(\delta_\mathrm{BB}) = \mathrm{TBR}_\mathrm{max}\left(1 - e^{-\delta_\mathrm{BB}/\delta_e}\right), \qquad \delta_e = \frac{\delta_\mathrm{BB}^\mathrm{sat}}{\ln 20}
$$ (eq:TBR)

where $\delta_e$ is calibrated so that the ratio reaches $95\,\%$ of its asymptote $\mathrm{TBR}_\mathrm{max}$ at the concept-specific saturation thickness $\delta_\mathrm{BB}^\mathrm{sat}$. This is a deliberately coarse closure, intended only to flag whether a candidate machine retains enough inboard breeding space to be self-sufficient, not to replace a dedicated neutronics calculation. The six concepts are compared in {numref}`Fig. %s <fig:blanket_concepts>`, which shows their TBR saturation curves alongside their asymptotic breeding ratio and energy multiplication.

:::{figure} /figures/thesis/blanket_concepts_comparison.png
:name: fig:blanket_concepts
:width: 100%
:align: center

Comparison of the six breeding-blanket concepts available in D0FUS, generated by the Run mode. Left: the tritium breeding ratio of Eq. {eq}`eq:TBR` as a function of the breeder-zone thickness, the dotted vertical lines marking each concept-specific saturation thickness $\delta_\mathrm{BB}^\mathrm{sat}$ and the dashed horizontal line the self-sufficiency threshold $\mathrm{TBR} = 1$. Right: the asymptotic breeding ratio $\mathrm{TBR}_\mathrm{max}$ and the energy multiplication factor $M_\mathrm{blanket}$ of each concept. The values are digitised from the Infinity Two pilot-plant blanket trade-off study {footcite:p}`clark2025blanket`.
:::

The radial build is determined by three requirements: (i) generating the peak field $B_\mathrm{max}$ on the inner leg of the TF coils (which sets the winding pack thickness $\Delta_\mathrm{WP}$ via Ampère’s law and the engineering current density $J^\mathrm{wost}$, see Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`), (ii) generating the magnetic flux $\Psi_\mathrm{CS}$ required from the CS to initiate, ramp up and maintain the plasma current, and (iii) withstanding the mechanical stresses induced by the associated Lorentz forces. These requirements are coupled, as will become apparent below. The models described in the following sections solve this coupled system either analytically (Academic) or by root finding (Refined).

:::{figure} /figures/thesis/fig_tf_forces_composite.png
:name: fig:tf_forces
:width: 95%
:align: center

The three families of Laplace forces acting on a toroidal field coil system. The plasma is shown in pale violet and the forces in red. (a) Poloidal section and in plane forces illustration; (b) Top view and in plane forces illustration; (c) Poloidal section and out of plane forces illustration.
:::

Following the two-fidelity approach of D0FUS (Section {ref}`General architecture of D0FUS <ssec:chap1_fidelity>`), two radial build models are available:

- The Academic model (Section {ref}`Academic model <ssec:chap1_academic>`) uses a two-layer idealisation for each coil: a pure conductor layer (superconductor, copper, helium, insulation) that generates the field but carries no mechanical stress, and a pure steel layer that handles all the loads. Thin-cylinder Lamé-Clapeyron theory {footcite:p}`LameClapeyron1833` provides closed-form expressions for the stress components ({ref}`Determination of σ_(θ) in the TF coil <app:thin_wall_stress>`). Its transparency makes it well suited for understanding the scaling of coil thickness with field, geometry and material properties.

- The Refined model (Section {ref}`Refined model <ssec:chap1_refined>`) introduces three improvements: firstly the winding pack is treated as a composite cable-in-conduit conductor (CICC) where steel and conductor coexist in each cross-section. Secondly, thick-cylinder Lamé theory replaces the thin-cylinder approximation, capturing the stress gradient across the coil. Finally, the CS axial stress induced by the fringe field at the solenoid extremities is included. Other system codes have pursued similar refinements {footcite:p}`swanson2022validation,morris2015implications`. The present model follows this trend while remaining fully analytical. It does not aim at the multiphysics detail of dedicated magnet design codes such as MADE {footcite:p}`giannini2023magnet` or MADMACS {footcite:p}`zani2019parametric`, but approaches their predictive capability for radial build sizing, as demonstrated by the benchmark in Chapter 3 of the thesis.

In addition to these two native models, D0FUS is also interfaced with the multi-layer thick-cylinder solver CIRCE {footcite:p}`boudes2025circe` developed at CEA-IRFM by B. Boudes, which provides closed-form stress and displacement fields in an arbitrary number of concentric layers and can be used as a higher-fidelity cross-check. The coupling is described in {ref}`Coupling to the CIRCE multi-layer solver <app:circe>`.

All radial build models in D0FUS operate in two dimensions (the poloidal plane) and assume axisymmetry. Out-of-plane forces (illustrated in {numref}`Fig. %s <fig:tf_forces>`), which are the toroidal components of the Lorentz load arising from the interaction of the TF current with the poloidal field, are inherently three-dimensional and are not evaluated within this framework. What sizes the structure against them, the distribution of the load along the coil, the bending and shear it induces in the winding pack, and the response of the inter-coil structures, calls for dedicated 3D structural analyses, outside the scope of a system code.

```{rubric} References
```

```{footbibliography}
```
