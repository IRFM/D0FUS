(ssec:chap1_CIRCE)=
(app:circe)=

# Coupling to the CIRCE multi-layer solver

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.18. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The Academic and Refined models of Sections {ref}`Academic model <ssec:chap1_academic>` and {ref}`Refined model <ssec:chap1_refined>` both rely on a simplified representation of the inboard column (winding pack plus nose or plug, with a single set of mechanical properties per layer). This is sufficient for sizing the radial build of a system-code design point, but does not capture the full elastic interaction between the inboard components. For applications requiring a higher fidelity stress field, D0FUS is interfaced with the analytical multi-layer solver CIRCE {footcite:p}`boudes2025circe` developed at CEA-IRFM by B. Boudes.

CIRCE solves the elastic equilibrium of an arbitrary number of concentric cylindrical layers under combined boundary loading (an internal pressure $P_i$ and an external pressure $P_e$) and a Lorentz body force $f_r(R) = J_\theta(R)\, B_z(R)$. Each layer is characterised by its inner and outer radii, its Young’s modulus and its azimuthal current density (set to zero for a purely structural layer), the Poisson ratio being assumed identical across all layers. The solver returns closed-form expressions for the radial displacement $u_r$ and for the two in-plane stress components $(\sigma_r, \sigma_\theta)$ at every radial node, with continuity of $u_r$ and $\sigma_r$ enforced at each interface by solving for the interface pressures. The axial stress $\sigma_z$ (vertical tension in the TF, fringe-field compression in the CS) is not part of the in-plane Lamé solution, it is evaluated with the same analytical expressions as in the Refined model and added to the CIRCE stress field to form the Tresca criterion.

CIRCE can be used for cross-validation of the Refined model on selected reference designs.

```{rubric} References
```

```{footbibliography}
```
