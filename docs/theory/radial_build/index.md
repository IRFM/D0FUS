(sec:chap1_magnets)=

# Radial build limits

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The plasma models of the previous section fix what a candidate machine demands from its magnets: the on-axis field, the plasma current, and the flux consumption of the scenario. Whether these demands can be met is an engineering question. The central solenoid and the toroidal field coils must fit within the inboard bore left by the plasma and the blanket, carry the required current densities within the limits of the superconductor, and survive the resulting electromagnetic loads, all of which become harder as the field increases.

The radial build models of D0FUS address this question. They form the second family of first-order hard limits.

The performance and the constructability of a tokamak power plant are ultimately limited by the space available on the high-field side (inboard midplane), where the plasma, the breeding blanket, the TF coil inner leg and the central solenoid must all fit between the magnetic axis and the machine axis. The radial build model determines the thicknesses of these components self-consistently from magnetic, electrical and mechanical constraints. These models, and their benchmarking against reference machines and dedicated magnet design codes, are presented in full detail in the reference paper {footcite:p}`auclair2026mechanical`, on which this section is closely based.

The coil models below are written in the cylindrical basis $(r, \theta, z)$ attached to the machine axis, following the convention of the magnet community.[^1]

[^1]: $\sigma_\theta$ is thus the hoop stress, along the azimuthal direction around the vertical axis of the tokamak: this $\theta$ is not the poloidal angle of Section {ref}`Plasma operational limits <sec:chap1_plasma>` anymore, but now denotes the azimuthal (toroidal) angle.

```{toctree}
:maxdepth: 1

definition
ripple
superconductors
academic
refined
```

```{rubric} References
```

```{footbibliography}
```
