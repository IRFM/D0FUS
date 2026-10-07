(app:d0fus_fidelity)=

# Fidelity levels and model selectors

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section details the two fidelity levels summarised in Section {ref}`General architecture of D0FUS <ssec:chap1_fidelity>`.

Each model carries its own selector inside {py:obj}`GlobalConfig <D0FUS_BIB.D0FUS_parameterization.GlobalConfig>` (`Plasma_geometry`, `Radial_build_model`, `Bootstrap_choice`, `eta_model`, `q_profile_mode`, `CD_source`, etc.), so that any combination of academic and refined sub-models can be tested. Two factory functions, {py:obj}`preset_academic <D0FUS_BIB.D0FUS_parameterization.preset_academic>` and {py:obj}`preset_refined <D0FUS_BIB.D0FUS_parameterization.preset_refined>`, return a {py:obj}`GlobalConfig <D0FUS_BIB.D0FUS_parameterization.GlobalConfig>` with a coherent set of sub-mode choices in a single call, while leaving every individual selector accessible for surcharge.

The Academic level uses closed-form analytical expressions wherever possible: cylindrical-torus volume integrals ($V = 2\pi^2 R_0 \kappa a^2$), uniform elongation without triangularity, a two-layer coil model (pure conductor plus pure steel), thin-cylinder stress theory, and classical Spitzer resistivity. Its transparency makes it well suited for pedagogical use, for cross-verification against published analytical results, and as a reference against which the effect of more detailed modelling can be quantified.

The Refined level replaces some analytical approximations by numerical models: volume integrals over Miller flux surfaces with radially varying $\kappa(\rho)$ and $\delta(\rho)$ profiles, composite cable-in-conduit conductor (CICC) winding packs (the current-carrying core of the coil) with self-consistent conductor and useful-steel fractions $f_c(R)$ and $f_u(R)$, thick-cylinder stress theory with the Tresca criterion, neoclassical resistivity (Sauter {footcite:p}`sauter1999neoclassical` or Redl {footcite:p}`redl2021bootstrap`), and a self-consistent $q(\rho)$ profile.

Both levels share the same input/output interface, the same GlobalConfig object, the same solver infrastructure, and have comparable execution times. {numref}`Table %s <tab:fidelity_levels>` in the chapter body summarises the differences between the two levels.

```{rubric} References
```

```{footbibliography}
```
