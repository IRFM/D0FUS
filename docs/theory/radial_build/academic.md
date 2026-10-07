(ssec:chap1_academic)=

# Academic model

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.3.4. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The Academic model determines the TF and CS thicknesses from the three sizing requirements of Section {ref}`Radial build definition and geometry <ssec:chap1_radial_build>`, under a deliberately simplified set of assumptions that prioritise transparency over quantitative accuracy. A simple two-layer coil model is adopted for both the CS and the TF coils: the first layer consists of a pure conductor (superconductor, copper, insulation and cooling) designed to generate $B_\mathrm{max}$ or $\Psi_\mathrm{CS}$, and is assumed not to handle any mechanical stress. The second layer is solely composed of steel to withstand all the mechanical stresses. This approach, although simplistic, makes it very easy to understand the underlying trends, and is typical of the magnet sizing modules found in other system codes {footcite:p}`panin2017mechanical,kovari2016process,duchateau2014conceptual`.

Two limitations must be noted here. First, any realistic winding pack includes a significant portion of steel for force handling, whereas the two-layer model concentrates all the steel in a dedicated layer, considered homogeneously loaded at the allowable limit $\sigma_\mathrm{lim}$ (which neglects stress concentration in the winding pack). Second, the thin-cylinder approximation ($\Delta R \ll R$) is used throughout, which allows simple mechanical expressions. These two limitations are alleviated in the Refined model of Section {ref}`Refined model <ssec:chap1_refined>`. Throughout, tensile stresses are counted as positive and compressive stresses as negative, and absolute values are used when comparing against $\sigma_\mathrm{lim}$.

## Toroidal field coil

The inner legs of the TF coils form a cylinder of outer radius $R_\mathrm{TF}^\mathrm{ext} = R_0 - a - \Delta_B$ and inner radius $R_\mathrm{TF}^\mathrm{int}$, with the conductor/steel interface at $R_\mathrm{TF}^\mathrm{sep}$ (Eq. {eq}`eq:radii_TF`). The aim is to determine $R_\mathrm{TF}^\mathrm{int}$ and the associated TF coil thickness $\Delta_\mathrm{TF}$ shown in {numref}`Fig. %s <fig:2layerstf>`.

:::{figure} /figures/thesis/2Layers__TF.png
:name: fig:2layerstf
:width: 55%
:align: center

Schematic horizontal cross-section of the high-field-side TF inner leg in the two-layer approximation, with the steel layer ($R_\mathrm{TF}^\mathrm{int}$ to $R_\mathrm{TF}^\mathrm{sep}$) inside the conductor layer ($R_\mathrm{TF}^\mathrm{sep}$ to $R_\mathrm{TF}^\mathrm{ext}$).
:::

### Magnetic field generation.

The first requirement is to carry enough current to generate the maximum magnetic field on the inner leg $B_\mathrm{max}$. The internal radius of the conductor layer $R_\mathrm{TF}^\mathrm{sep}$, illustrated in {numref}`Fig. %s <fig:2layerstf>`, follows directly in the thin-cylinder approximation ($\Delta R = R_\mathrm{TF}^\mathrm{ext} - R_\mathrm{TF}^\mathrm{sep} \ll R_\mathrm{TF}^\mathrm{ext}$) as detailed in {ref}`Determination of B_(TF) <Annexe_B_thinlayer>`:

$$
B_\mathrm{max} = \mu_0\, J^\mathrm{wost}\,(R_\mathrm{TF}^\mathrm{ext} - R_\mathrm{TF}^\mathrm{sep}) \quad\Longrightarrow\quad R_\mathrm{TF}^\mathrm{sep} = R_\mathrm{TF}^\mathrm{ext} - \frac{B_\mathrm{max}}{\mu_0\, J^\mathrm{wost}}
$$ (eq:B_thin_acad)

Here $J^\mathrm{wost}$ is the engineering current density defined on the non-steel cross-section of the conductor, whose determination as a function of field and temperature is detailed in Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`.

### Mechanical stress.

The electromagnetic forces acting on the TF coil case can be decomposed into two components: a tensile force along the $z$-axis and a centering force along the $r$-axis directed toward the machine axis.[^1] An important aspect of the resulting stress distribution is the so-called *vault effect*, illustrated in {numref}`Fig. %s <fig:topviewtf_thesis>`, which shows a sector of the tokamak with an inner TF coil leg and an associated section of the CS. This effect arises from the mechanical response of cylindrical shells to radial pressure. The coil structure must then be designed to withstand the resulting stress state.

Three mechanical architectures are considered throughout these pages, illustrated in {numref}`Fig. %s <fig:topviewtf_thesis>`. In the *wedging* configuration the inboard legs of the TF coils are in toroidal contact. In the *bucking* configuration they are separated and rest radially onto the central solenoid. The *plug* configuration is a bucking variant complemented by a stiff central insert inside the CS bore. The wedging and bucking architectures are detailed below. The plug configuration involves the CS specifically and is deferred to the central-solenoid part.

A common failure criterion is based on the Tresca stress, defined as the maximum absolute difference between principal stresses,

$$
\sigma_\mathrm{Tresca} = \max\!\left(|\sigma_r - \sigma_\theta|,\; |\sigma_r - \sigma_z|,\; |\sigma_\theta - \sigma_z|\right)
$$

The Tresca criterion is adopted as the stress acceptability condition: it requires that the largest difference between any two principal stresses remains below a prescribed allowable stress, here taken as $\sigma_\mathrm{lim} = \tfrac{2}{3}\sigma_\mathrm{yield}$ following the ITER magnet structural design criteria {footcite:p}`jong2007iter`, providing a conservative bound for components subjected to multiaxial loading. Two steel grades are available in D0FUS: 316L austenitic steel ($\sigma_\mathrm{yield} \approx 990$ MPa at 4.2 K, $\sigma_\mathrm{lim} = 660$ MPa) and CHSN01 high-strength steel ($\sigma_\mathrm{yield} \approx 1500$ MPa, $\sigma_\mathrm{lim} = 1000$ MPa) {footcite:p}`sutcliffe2025magnet`, also designated N50H in part of the qualification literature and in the code input decks. In practice, the dominant stresses are the tensile $\sigma_z$ and centering $\sigma_r$ stress in the bucking configuration, and the tensile $\sigma_z$ and vault $\sigma_\theta$ stress in the wedging configuration:

$$
\begin{aligned}
\text{Bucking case:} \quad &\sigma_\mathrm{Tresca} = |\sigma_r - \sigma_z| \leq \sigma_\mathrm{lim} \\
		\text{Wedging case:} \quad &\sigma_\mathrm{Tresca} = |\sigma_\theta - \sigma_z| \leq \sigma_\mathrm{lim}
\end{aligned}
$$ (eq:Tresca)

:::{figure} /figures/thesis/fig_topviewtf_thesis_composite.png
:name: fig:topviewtf_thesis
:width: 95%
:align: center

Illustration of the three mechanical configurations of the TF inner legs and CS: wedging (a), bucking (b) and plug (c). In wedging, the inboard TF legs form a closed vault that transforms the centering pressure into a hoop stress and the CS is mechanically independent. In bucking, the TF legs rest radially onto the CS, which then absorbs the centering load. In plug, a stiff central cylinder takes over the TF compression under a nearly hydrostatic stress state. (a) Wedging; (b) Bucking; (c) Plug.
:::

**Bucking.** The bucking configuration consists in supporting the inner leg of the TF coils on the CS to transfer the centering forces to it: the TF coils are assumed to have no contact between them. At the inner leg, where the toroidal field peaks at $B_\mathrm{max}$, the centering stress $\sigma_r$ can be approximated by the magnetic pressure

$$
\sigma_r = P_\mathrm{TF} = \frac{B_\mathrm{max}^2}{2\mu_0}
$$ (eq:P_TF)

which reaches $\approx 55$ MPa for ITER ($B_\mathrm{max} = 11.8$ T) and $\sim 160$ MPa at $B_\mathrm{max} = 20$ T. The vertical force $F_z$, defined as the upward force acting on the upper half of a coil assumed vertically symmetric, is independent of the coil shape {footcite:p}`freidberg2015designing`. Its expression per coil for a circular geometry (also valid for D-shaped coils) is derived in {ref}`Determination of F_(z) <app:Fz>`:

$$
F_z = \frac{\pi}{\mu_0 N_\mathrm{coil}}\, B_0^2\, R_0^2\, \ln\!\left(\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}\right)
$$ (eq:Fz)

with $B_0$ the central magnetic field, $N_\mathrm{coil}$ the number of TF coils (the total vertical force being $N_\mathrm{coil} F_z$), and the outboard leg located at $R_0 + a + \Delta_B + \Delta_\mathrm{ext}$, $\Delta_\mathrm{ext}$ being the outboard radial standoff introduced in Section {ref}`TF coil number and toroidal field ripple <ssec:chap1_ripple>`. The tensile stress $\sigma_z$ is then obtained by distributing a fraction $f_{z,\mathrm{WP}}$ of the total tension $N_\mathrm{coil} F_z$ over the inboard leg cross-section, an assumption which is exact for a Princeton D-shaped coil (neglecting inter-coil structures). With the default $f_{z,\mathrm{WP}} = 1/2$ (equal sharing between inboard and outboard legs {footcite:p}`federici2026iter`) and no external clamping, that is no external pre-compression structure offloading part of the vertical tension ($F_\mathrm{clamp} = 0$),

$$
\sigma_z = \frac{N_\mathrm{coil} F_z}{2\pi\left[(R_\mathrm{TF}^\mathrm{sep})^2 - (R_\mathrm{TF}^\mathrm{int})^2\right]}
$$ (eq:sigma_z_TF_acad)

In D0FUS, both $f_{z,\mathrm{WP}}$ and $F_\mathrm{clamp}$ (subtracted from $F_z$) are configurable parameters.

**Wedging.** For the wedging case, $\sigma_z$ is still given by Eq. {eq}`eq:sigma_z_TF_acad`. The centering force is now transmitted to the steel noses located at the inner radius of each TF coil. In contact with each other, they form a vault that transforms the radial stress into an azimuthal one $\sigma_\theta$. In the limit of a thin cylindrical shell subject to a sole external pressure, the Lamé-Clapeyron theory {footcite:p}`LameClapeyron1833` provides the simple expression detailed in {ref}`Determination of σ_(θ) in the TF coil <app:thin_wall_stress>`,

$$
\sigma_\theta = \frac{P_\mathrm{TF}\, R_\mathrm{TF}^\mathrm{sep}}{R_\mathrm{TF}^\mathrm{sep} - R_\mathrm{TF}^\mathrm{int}}
$$ (eq:sigma_theta_TF_acad)

**Solution.** The engineering current density $J^\mathrm{wost}$, obtained from the superconductor scalings of Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`, fixes the winding pack thickness and hence $R_\mathrm{TF}^\mathrm{sep}$ (Eq. {eq}`eq:B_thin_acad`). Setting $\sigma_\mathrm{Tresca}$ to its limit $\sigma_\mathrm{lim}$ then yields $R_\mathrm{TF}^\mathrm{int}$. Solving for wedging in the thin-cylinder approximation gives (full calculus in {ref}`Wedging 1st order <app:wedging_academic>`)

$$
R_\mathrm{TF}^\mathrm{int} = R_\mathrm{TF}^\mathrm{sep} - \frac{B_\mathrm{max}^2\, R_\mathrm{TF}^\mathrm{sep}}{2\mu_0\, \sigma_\mathrm{lim}}\left(1 + \frac{1}{2}\ln\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}\right)
$$ (eq:Rint_wedging_acad)

and similarly in the bucking configuration (full calculus in {ref}`Bucking 1st order <app:bucking_academic>`)

$$
R_\mathrm{TF}^\mathrm{int} = R_\mathrm{TF}^\mathrm{sep} - \frac{B_\mathrm{max}^2\, R_\mathrm{TF}^\mathrm{sep}}{4\mu_0\left(\sigma_\mathrm{lim} - B_\mathrm{max}^2/(2\mu_0)\right)}\,\ln\frac{R_0 + a + \Delta_B + \Delta_\mathrm{ext}}{R_0 - a - \Delta_B}
$$ (eq:Rint_bucking_acad)

## Central solenoid

The CS is treated with the same two-layer model, illustrated in {numref}`Fig. %s <fig:2layerscs>`: an outer conductor layer sized to generate the required flux $\Psi_\mathrm{CS}$, and an inner steel layer carrying the mechanical load. In the wedging configuration the CS can be designed as several modules stacked together, whose currents are varied independently to improve plasma shaping and vertical-stability control, as in ITER {footcite:p}`libeyre2009detailed`. In contrast, in the case of bucking, the CS contact surface with the TF coils must form a continuous interface along its entire length to ensure a uniform force distribution. Note that this requirement need not be met by the solenoid itself: in JET for example, the interface is provided by a solid cylinder inserted between the solenoid stack and the TF coils, which lets the solenoid remain modular {footcite:p}`rebut1981jet`. For simplicity, the CS is approximated as an infinite solenoid in this model and vertical stresses are neglected, as they are usually less critical than the hoop stress {footcite:p}`Fu2006,jong2009mechanical,nunio2019mechanical`. The Refined model of Section {ref}`Refined model <ssec:chap1_refined>` relaxes this assumption and accounts for the axial stress induced at the CS midplane by the radial fringe field at the solenoid ends.

:::{figure} /figures/thesis/2Layers_CS.png
:name: fig:2layerscs
:width: 55%
:align: center

Schematic cross-section of the CS in the two-layer approximation, with the steel layer ($R_\mathrm{CS}^\mathrm{int}$ to $R_\mathrm{CS}^\mathrm{sep}$) inside the conductor layer ($R_\mathrm{CS}^\mathrm{sep}$ to $R_\mathrm{CS}^\mathrm{ext}$).
:::

### Flux consumption criteria.

The magnetic flux requirement can be expressed as the balance

$$
\Psi_\mathrm{Init} + \Psi_\mathrm{Ramp\text{-}Up} + \Psi_\mathrm{plateau} = \Psi_\mathrm{CS} + \Psi_\mathrm{PF}
$$ (eq:CS_flux_balance)

whose terms are detailed below.

$\Psi_\mathrm{Init}$ is the flux required to initiate the plasma. Integrating Faraday’s law over the breakdown duration $t_\mathrm{BD}$ gives

$$
\Psi_\mathrm{Init} = 2\pi R_0\, E_\varphi\, t_\mathrm{BD} = 2\pi R_0\, \mathcal{E}_\mathrm{BD}
$$ (eq:Psi_init_acad)

where $\mathcal{E}_\mathrm{BD} = E_\varphi\, t_\mathrm{BD}$ (in V.s/m) lumps all breakdown physics (fill pressure, error fields, impurity burn-through) into a single calibration parameter. This formulation captures the machine-size scaling: a larger torus requires more flux for the same breakdown conditions. $\mathcal{E}_\mathrm{BD}$ is treated as a calibration constant. Inverting Eq. {eq}`eq:Psi_init_acad` on the ITER values ($R_0 = 6.2$ m, $\Psi_\mathrm{Init} \approx 10$ Wb {footcite:p}`shimada2007progress`) gives $\mathcal{E}_\mathrm{BD} \approx 0.25$ V.s/m {footcite:p}`lloyd1991plasma`. This ITER-calibrated value is retained as the default. In practice, $\Psi_\mathrm{Init}$ represents less than 5 % of the global CS flux budget, so the result is only weakly sensitive to $\mathcal{E}_\mathrm{BD}$.

The flux consumption to ramp up the plasma current is composed of an inductive and a resistive term,

$$
\Psi_\mathrm{Ramp\text{-}Up} = (1 - f_h)\left[\underbrace{L_p\, I_p}_{\Psi_\mathrm{Ind}} + \underbrace{C_\mathrm{Ejima}\, \mu_0 R_0\, I_p}_{\Psi_\mathrm{Res}}\right]
$$ (eq:Psi_rampup_acad)

In D0FUS, a fraction $f_h \in [0,1]$ of the ramp-up flux consumption is assumed to be saved by the non-inductive contributions of the heating and current-drive systems, both terms of Eq. {eq}`eq:Psi_rampup_acad` being reduced by the factor $(1 - f_h)$. The conservative default is $f_h = 0$. The plasma self-inductance $L_p = L_\mathrm{ext} + L_\mathrm{int}$ is decomposed into an external part (field energy outside the plasma) and an internal part $L_\mathrm{int} = \mu_0 R_0\, l_i/2$, with the normalised internal inductance $l_i$ computed as in Section {ref}`Plasma current and scaling law <ssec:chap1_currents>` (Eq. {eq}`eq:li3`). The external inductance, corresponding to the poloidal magnetic field energy stored in the vacuum around the plasma ring, is determined, at a given inverse aspect ratio $\varepsilon = a/R_0$, by the Hirshman and Neilson fit {footcite:p}`hirshman1986external`, built on numerical equilibria of a finite-aspect-ratio torus,

$$
L_\mathrm{ext} = \mu_0 R_0\, \frac{a_\varepsilon\,(1 - \varepsilon)}{1 - \varepsilon + b_\varepsilon\, \kappa}
$$

with $\kappa$ the elongation at the last closed flux surface and

$$
a_\varepsilon = (1 + 1.81\sqrt{\varepsilon} + 2.05\,\varepsilon)\ln\frac{8}{\varepsilon} - (2 + 9.25\sqrt{\varepsilon} - 1.21\,\varepsilon), \qquad b_\varepsilon = 0.73\sqrt{\varepsilon}\,(1 + 2\varepsilon^4 - 6\varepsilon^5 + 3.7\varepsilon^6)
$$

The resistive term follows the Ejima scaling {footcite:p}`ejima1982volt` with the default value $C_\mathrm{Ejima} = 0.45$, the practical estimate of the minimum resistive flux consumption recommended by the ITER Physics Basis {footcite:p}`iterphysicsbasis1999ch8`.

The plateau magnetic flux $\Psi_\mathrm{plateau}$ is estimated from the loop voltage $V_\mathrm{loop}$ and the plateau duration $t_\mathrm{plateau}$,

$$
\Psi_\mathrm{plateau} = V_\mathrm{loop}\, t_\mathrm{plateau}
$$ (eq:Psi_plateau_acad)

The decomposition of the plasma current $I_p = I_\Omega + I_b + I_\mathrm{CD}$ into ohmic, bootstrap and externally driven contributions, the bootstrap current $I_b$, and the effective plasma resistance $\mathcal{R}_p$ entering the loop voltage $V_\mathrm{loop} = \mathcal{R}_p\, I_\Omega$ are all developed in Section {ref}`Plasma current and scaling law <ssec:chap1_currents>` (Eqs. {eq}`eq:Ip_decomposition` and {eq}`eq:Reff`). In steady-state configuration the plasma current is entirely non-inductively driven, and $\Psi_\mathrm{plateau}$ is taken equal to 0.

The flux provided by an infinite solenoid over a full swing can be expressed as (details in {ref}`Determination of Ψ_(CS) <app:Psi_CS>`)

$$
\Psi_\mathrm{CS} = \frac{2\pi B_\mathrm{CS}}{3}\left[(R_\mathrm{CS}^\mathrm{ext})^2 + R_\mathrm{CS}^\mathrm{ext}\, R_\mathrm{CS}^\mathrm{sep} + (R_\mathrm{CS}^\mathrm{sep})^2\right]
$$ (eq:Psi_CS_acad)

where $B_\mathrm{CS}$ is the magnetic field inside the CS at the beginning of the pulse. Eq. {eq}`eq:Psi_CS_acad` expresses the flux that the CS hardware can deliver over a full swing, but only a fraction of this swing may actually be available to drive the plasma current, the rest being reserved for plasma control during the discharge. To account for this, D0FUS introduces a parameter $f_\mathrm{swing}^\mathrm{usable} \in (0,1]$ such that the inductive flux available to the plasma is $f_\mathrm{swing}^\mathrm{usable}\, \Psi_\mathrm{CS}$. A default value of $f_\mathrm{swing}^\mathrm{usable} = 0.75$ is adopted, in line with EU-DEMO and ITER design practice (F. Maviglia, private communication).

The contribution of the PF coils is approximated by computing the vertical magnetic field and the corresponding flux, as in Ref. {footcite:p}`duchateau2014conceptual`,

$$
B_\mathrm{vert} = \frac{\mu_0 I_p}{4\pi R_0}\left(\beta_P + \frac{l_i - 3}{2} + \log\frac{8 R_0}{a\sqrt{\kappa}}\right)
$$

$$
\Psi_\mathrm{PF} = B_\mathrm{vert}\, \pi R_0^2
$$ (eq:Psi_PF_acad)

with the poloidal beta $\beta_P$ defined in Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>` (Eq. {eq}`eq:betaP`). The integration of $B_\mathrm{vert}$ extends down to the magnetic axis because the PF coils enter the poloidal flux balance through two independent contributions, each associated with a different phase of the scenario. At breakdown, the PF coils cancel the CS stray field over a broad region around $R_0$, producing the null-field zone required for plasma initiation {footcite:p}`formisano2017analysis,devries2019breakdown`. This is what makes the infinite-solenoid hypothesis of Eq. {eq}`eq:Psi_CS_acad` applicable to a finite CS, since the radial leakage of the field outside the CS bore is then exactly compensated up to $R_0$. Once the plasma carries current, the PF coils must also generate the Shafranov vertical field $B_\mathrm{vert}$ that holds the column at $R_0$. Because $B_\mathrm{vert}$ is approximately uniform across the plasma cross-section, its flux is integrated over the full disk up to the magnetic axis.

Applying Ampère’s theorem to an infinite solenoid of winding pack thickness $R_\mathrm{CS}^\mathrm{ext} - R_\mathrm{CS}^\mathrm{sep}$ carrying a uniform current density $J^\mathrm{wost}$ gives

$$
B_\mathrm{CS} = \mu_0 J^\mathrm{wost}\,(R_\mathrm{CS}^\mathrm{ext} - R_\mathrm{CS}^\mathrm{sep})
$$ (eq:BCS_acad)

The only remaining unknown being $R_\mathrm{CS}^\mathrm{sep}$, it follows from combining Eqs. {eq}`eq:CS_flux_balance`, {eq}`eq:Psi_CS_acad` and {eq}`eq:BCS_acad`, the net plasma demand being supplied by the usable fraction $f_\mathrm{swing}^\mathrm{usable}$ of the full swing,

$$
R_\mathrm{CS}^\mathrm{sep} = \sqrt[3]{\,(R_\mathrm{CS}^\mathrm{ext})^3 - \frac{3\left|\Psi_\mathrm{Init} + \Psi_\mathrm{Ramp\text{-}Up} + \Psi_\mathrm{plateau} - \Psi_\mathrm{PF}\right|}{2\pi \mu_0 J^\mathrm{wost}\, f_\mathrm{swing}^\mathrm{usable}}\,}
$$

However, since $J^\mathrm{wost}$ depends on $B_\mathrm{CS}$ through the superconductor scaling laws (Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`), the system must be solved iteratively. Initialising $B_\mathrm{CS}$ with the thin-solenoid estimate $B_\mathrm{CS} \approx \Psi_\mathrm{CS}/(\pi (R_\mathrm{CS}^\mathrm{ext})^2)$ provides a good starting point, and convergence is reached within a few iterations.

### Mechanical stress.

A temporal analysis of the forces acting on the CS is necessary to identify the limiting cases for sizing it. {numref}`Fig. %s(a) <fig:CS_temporal_stress>` illustrates the typical evolution of the CS current during a pulse: a pre-magnetisation phase raises the current to $I_\mathrm{CS,max}$, followed by a discharge that passes through zero at $t_1$ (during the ramp-up, or possibly later in the flat-top) and reverses to enable further ramp-up of the plasma current. The current is then held during the plateau and returns to zero at ramp-down. The example assumes that $I_p$ is at least partly driven by the CS during the plateau. In a steady-state scenario the slope of $I_\mathrm{CS}$ vanishes there, but the overall conclusions on the mechanical stress remain unchanged.

:::{figure} /figures/thesis/fig_CS_temporal_stress_composite.png
:name: fig:CS_temporal_stress
:width: 95%
:align: center

Temporal evolution of the CS current during a typical discharge (a) and of the resulting CS stresses in the wedging (b), light bucking (c) and strong bucking (d) configurations. Sign convention: tensile stresses are positive, compressive stresses are negative. (a) Schematic temporal evolution of the CS current during a typical pulsed discharge; (b) CS stress evolution in wedging; (c) CS stress evolution in light bucking; (d) CS stress evolution in strong bucking.
:::

**Wedging.** In the wedging configuration the CS is free-standing: the TF coils are separated from it, so the CS must withstand its own $J \times B$ forces through hoop tension. As shown in {numref}`Fig. %s(b) <fig:CS_temporal_stress>`, the critical instants occur at $t_0$ and $t_2$, when $I_\mathrm{CS} = \pm I_\mathrm{CS,max}$. Within the thin-cylinder approximation,

$$
\sigma_\theta = \frac{P_\mathrm{CS}\, R_\mathrm{CS}^\mathrm{sep}}{R_\mathrm{CS}^\mathrm{sep} - R_\mathrm{CS}^\mathrm{int}}, \qquad P_\mathrm{CS} = \frac{B_\mathrm{CS}^2}{2\mu_0}
$$ (eq:PCS)

Setting $\sigma_\theta = \sigma_\mathrm{lim}$ gives directly

$$
R_\mathrm{CS}^\mathrm{int} = R_\mathrm{CS}^\mathrm{sep}\left(1 - \frac{P_\mathrm{CS}}{\sigma_\mathrm{lim}}\right)
$$ (eq:Rint_CS_wedg_acad)

In the configurations where the CS stress cycles at every pulse (wedging and light bucking), a fatigue knockdown factor further divides the steel allowable, in the Academic model as well as in the Refined one (Section {ref}`Refined model <ssec:chap1_refined>`).

**Bucking.** In this configuration the TF coils transfer their centering force to the CS, resulting in its compression. Two limiting instants could a priori dimension the CS. At $t_0$, the CS carries its maximum current: its own hoop stress $|\sigma_{\theta,\mathrm{CS}}|$ (driven by $P_\mathrm{CS}$) is partially relieved by the TF compression $|\sigma_{\theta,\mathrm{TF}}|$ (driven by $P_\mathrm{TF}$), giving a net stress $|\sigma_{\theta,\mathrm{CS}}| - |\sigma_{\theta,\mathrm{TF}}|$ ({numref}`Fig. %s(c) <fig:CS_temporal_stress>`). At $t_1$, the CS current vanishes and only the TF pre-compression remains, with a stress $|\sigma_{\theta,\mathrm{TF}}|$ ({numref}`Fig. %s(d) <fig:CS_temporal_stress>`). The dimensioning case is the larger of the two: the net stress scales as $P_\mathrm{CS} - P_\mathrm{TF}$ at $t_0$ and as $P_\mathrm{TF}$ at $t_1$, so the former dominates only when $P_\mathrm{CS} > 2 P_\mathrm{TF}$ (Eqs. {eq}`eq:PCS` and {eq}`eq:P_TF`). The CS own pressure thus dominates (“light” bucking) only when $B_\mathrm{CS} > \sqrt{2}\, B_\mathrm{max}$, a condition unlikely to be met in tokamaks since $B_\mathrm{CS}$ and $B_\mathrm{max}$ are expected to be of comparable magnitude. The TF pressure therefore dominates in essentially all relevant designs, a regime referred to as “strong” bucking. Both branches are retained in D0FUS through a $\max(\cdot)$ comparison, but only the strong bucking case is detailed here. The CS hoop stress then reads

$$
|\sigma_\theta| = \frac{P_\mathrm{TF}\, R_\mathrm{CS}^\mathrm{sep}}{R_\mathrm{CS}^\mathrm{sep} - R_\mathrm{CS}^\mathrm{int}}
$$ (eq:sigma_theta_CS_buck_acad)

and setting $|\sigma_\theta| = \sigma_\mathrm{lim}$,

$$
R_\mathrm{CS}^\mathrm{int} = R_\mathrm{CS}^\mathrm{sep}\left(1 - \frac{P_\mathrm{TF}}{\sigma_\mathrm{lim}}\right)
$$ (eq:Rint_CS_buck_acad)

**Plug.** The addition of a plug at the centre of the CS ({numref}`Fig. %s <fig:topviewtf_thesis>`) is an option of interest in the strong bucking case, considered for example in the V1 design of ARC {footcite:p}`sorbom2015arc,wade2021cost`. The main advantage of this technique is that the plug experiences a uniform radial pressure, producing a near-hydrostatic stress state ($\sigma_r \approx \sigma_\theta \approx -P$) in which the Tresca stress $|\sigma_r - \sigma_\theta| \approx 0$. The residual stresses are governed by local effects (material defects, geometric imperfections) and remain well below the allowable limits {footcite:p}`puthoff1969digital,yiannopoulos5stress,grigorenko2006solving`, so the plug can be assumed to withstand the applied loads without detailed dimensioning. Its usefulness logically depends on its stiffness, which determines the distribution between hoop and radial stress in the CS: a very soft plug will not significantly change the situation, the CS still reacting the TF pressure through the vault effect (large $\sigma_\theta$), whereas a very stiff plug leaves the CS to support only its $\sigma_r$, entirely transferred to the plug. To illustrate this and obtain relevant orders of magnitude, several COMSOL Multiphysics simulations were carried out, applying an external pressure of 100 MPa (corresponding to the load exerted by a TF coil system at $B_\mathrm{max} \approx 16$ T) to the outer radius of a typical ITER-like CS {footcite:p}`libeyre2009detailed` (outer radius 2 m, inner radius 1 m, 316L steel) with the plug Young’s modulus varied as a free parameter. As shown in {numref}`Fig. %s <fig:comsolplug>`, a very soft plug ($\approx 0$ GPa) recovers the bucking case ($\sigma_\mathrm{max} = \max(\sigma_\theta)$), a plug as stiff as steel ($200$ GPa) gives a uniform $\sigma_\mathrm{max} = \sigma_r$, and the reduction already reaches one third with a plug of low mechanical resistance ($30$ GPa), the approximation $\sigma_\mathrm{max} \approx \sigma_r$ becoming acceptable above $\approx 100$ GPa. Glass-fibre/epoxy composites (e.g. G10/G11) appear to be good candidates for a CS plug, combining very high electrical resistivity and low magnetic susceptibility (minimising eddy-current heating and magnetic interactions) with adequate mechanical strength and machinability {footcite:p}`kasen1980mechanical,li2019flashover,benzinger1980manufacturing`. Their typical Young’s modulus of 100 GPa is stiff enough relative to the CS (whose effective modulus, as a composite of steel and non-structural materials, is below that of pure 316L) to justify the approximation $\sigma_\mathrm{max} \approx \sigma_r$. Moreover, in the thin-cylinder approximation $\sigma_r$ is simply equal to $P_\mathrm{TF}$ and does not depend at all on the steel thickness, suggesting the latter could be very small. This illustrates that, to first order, the plug configuration relaxes the mechanical constraint to the simple condition $P_\mathrm{TF} < \sigma_\mathrm{lim}$, which is easily satisfied at all field levels considered here.

:::{figure} /figures/thesis/Comsol_Plug.png
:name: fig:comsolplug
:width: 100%
:align: center

Von Mises stress from COMSOL simulations of a CS under $100$ MPa external pressure, for varying plug Young’s modulus.
:::

[^1]: It helps to internalise just how enormous these electromagnetic forces are: on a single ITER TF coil, the radial (centering) component of the $\vec{J} \times \vec{B}$ load amounts to roughly $4 \times 10^8$ N, while the vertical force on the upper half of the coil reaches roughly $2 \times 10^8$ N {footcite:p}`titus2002provisions,mitchell2008iter`. To put this in everyday terms: each TF coil experiences a centering load equivalent to the weight of about four Eiffel Towers pressing it inward, and a vertical pull equivalent to two Eiffel Towers. ITER has eighteen such coils.

```{rubric} References
```

```{footbibliography}
```
