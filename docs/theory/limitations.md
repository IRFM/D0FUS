(sec:limitations)=

(chap:uncertainties)=
(chap:uncertainties_doc)=
# Intrinsic limitations of a zero-dimensional system code

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 4.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The analyses presented in Chapter 3 of the thesis are performed at the system-code level and carry inherent limitations.

Firstly, the models are purely in-plane and cannot address effects requiring 3D structural analysis. The bucking configuration in particular raises challenging out-of-plane force management issues. The toroidal component of the Lorentz force ($I_\mathrm{TF} \times B_\mathrm{pol}$) creates an overturning moment on the TF coil inner leg {footcite:p}`titus2002provisions`. In wedging, this force is partly balanced by toroidal friction between adjacent coils. In bucking, the coils are no longer in toroidal contact and the out-of-plane loads must be managed by dedicated structures. The present work assumes that these forces *can* be managed, and aims to help decide whether the resulting radial build gains are worth pursuing.

Secondly, beyond the management of out-of-plane loads, both bucking and plug configurations raise practical engineering challenges. A modular CS, with independent current control in each module, can be desirable for plasma shaping purposes {footcite:p}`huguet2001iter`, but is difficult to implement in these architectures. The central issue is the routing of helium cooling lines and electrical feeds. It is already constrained in bucking by the direct TF-CS mechanical contact, and potentially impractical in the plug configuration, where it must pass through a solid central cylinder. Moreover, this TF-CS interface, idealised as direct contact here, must accommodate a dedicated structural component (such as the JET sliding cylinder {footcite:p}`rebut1976jet`) with specific surface, friction, and load-transfer properties, which also govern the long-term resistance to scuffing and progressive CS erosion under the cyclic TF torsion.

Thirdly, this study has only addressed the mechanical aspects of the radial build. Other constraints, most notably divertor heat flux, neutron wall load and maintenance access, can become critical in compact machines. To illustrate the divertor side, {numref}`Table %s <tab:Psep_R0>` reports the proxy $P_\mathrm{sep}/R_0$ for three designs at $P_\mathrm{fus} \approx 2$ GW: the conventional EU-DEMO 2017 baseline and the two compact 20 T configurations of Table 3.2 of the thesis. This ratio enters the simplest scaling of the poloidal heat flux at the divertor entrance, $q_\mathrm{pol} \propto P_\mathrm{sep}/(R_0\,\lambda_q)$ (Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>`). Absolute values depend on the conventions adopted for $P_\mathrm{sep}$, but their ratios consistently rank design difficulty: relative to the EU-DEMO baseline, the two compact 20 T designs of {numref}`Table %s <tab:Psep_R0>` raise this proxy by a factor of $1.2$ and $1.7$ respectively, and the SF-Plant design point of Section 3.2 of the thesis by a factor of about $2$. A more refined analysis {footcite:p}`siccinio2019heat` indicates that such compact configurations lie at, or beyond, the lower edge of the viable divertor design space, and discusses possible mitigation strategies.

:::{table} Divertor heat-flux proxy $P_\mathrm{sep}/R_0$ for the EU-DEMO 2017 baseline and the two compact 20 T designs of Table 3.2 of the thesis (at $a = 2.5$ m).
:name: tab:Psep_R0
:align: center

| Design                      | $P_\mathrm{sep}/R_0$ \[MW/m\] |
|:----------------------------|:------------------------------|
| EU-DEMO 2017 baseline       | 29.5                          |
| First-order levers combined | 34.1                          |
| All levers combined         | 51.2                          |
:::

Lastly, as previously demonstrated, exploiting high fields in a compact machine requires a combination of innovations (REBCO, high-strength steels, alternative mechanical configurations, conductor optimisation), each carrying its own development risk, and at very different levels of maturity. Bucking, plug, and high-strength steels have been rarely studied at the power plant scale and even more rarely tested in actual tokamaks, making them inherently higher-risk options. In fact, wedging with 316L/JK2LB steel and no grading is the baseline for most existing superconducting tokamaks (ITER, JT-60SA, EAST, KSTAR, etc.), with the notable exceptions of JET {footcite:p}`rebut1976jet` and early ITER designs {footcite:p}`no1999final,mitchell1999iter,titus2002provisions,titus1998analysis,titus1995structural`, both in bucking configuration. Note that this situation should evolve in the near future: SPARC {footcite:p}`creely2020overview,diazpacheco2025electromechanical` is expected to provide the first modern bucking design, while BEST {footcite:p}`zhu2025electromagnetic,wang2023study,wang2024structure,wang2026mechanical,wang2026mass` plans to use CHSN01 steel in both its TF and CS magnets, offering the first validation of this steel grade in a fusion magnet system.

(sec:chap4_scope)=
Regarding core plasma physics, limitations also exist: the confinement is imposed through a global scaling multiplier and profile shape factors rather than computed from the underlying turbulent transport, which would require a reduced transport model, e.g. TGLF {footcite:p}`staebler2007tglf,artaud2010cronos`. The pedestal is parameterised through its position, height and width rather than obtained from a stability calculation based on peeling-ballooning and kinetic-ballooning modes, as in EPED {footcite:p}`snyder2011eped`. The stability limits are algebraic thresholds rather than MHD computations, etc. The density limit itself is debated: edge-turbulence analyses {footcite:p}`giacomin2022density` and recent experimental re-examinations {footcite:p}`angioni2026density` question the Greenwald form. The empirical laws and scalings used along the chain, in particular the $\tau_E$ scaling law, add a caveat of their own. Beyond their intrinsic error bars, they have been extracted from a limited domain of the plasma parameter space. Inside that domain they interpolate, but for a power plant study, some of them are used in extrapolation.

Note that this reduction, however, both for plasma physics and engineering, is also precisely what makes a system code so fast. Because a single evaluation is nearly free, tens of thousands of design points can be swept. A system-code study is in that sense a scouting instrument: its role is to rough out the design space in order to propose a design point, and to hand the dedicated codes and the experiments a short list of assumptions to examine.

The limitations above are of two kinds: effects that the description leaves out altogether, such as the out-of-plane loads, and models that are included but carry an uncertainty of their own. Among the latter, a handful exist in the literature in several formulations that are all widely used in the community yet lead to markedly different machines. These alternative model variants, and the consequences they have on a power plant design, are the subject of the next section.

```{rubric} References
```

```{footbibliography}
```
