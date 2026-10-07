(app:omega_sensitivity)=

# Sensitivity to the WP/nose tension partition

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.13. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The Refined model parameter $f_{z,\mathrm{WP}} \in [0,1]$ defined in Section {ref}`Refined model <ssec:chap1_refined>` sets the fraction of the total vertical tension $F_z$ borne by the winding pack, the complementary fraction $1 - f_{z,\mathrm{WP}}$ being reacted by the steel nose. This page quantifies the impact of $f_{z,\mathrm{WP}}$ on the predicted radial build.

{numref}`Figure %s <fig:omega_scan>` shows the result of an $f_{z,\mathrm{WP}}$ scan on the ITER reference parameter set of {numref}`Table %s <tab:chap2_tf_coil_benchmark>`. Over the realistic range $f_{z,\mathrm{WP}} \in [0.2, 0.8]$, the total thickness varies by less than 10%. This weak dependence has a simple origin: at fixed $F_z$ and $\sigma_\mathrm{lim}$, the total steel cross-section needed to react the axial tension is set by $\sigma_z = F_z / S_\mathrm{steel} \leq \sigma_\mathrm{lim}$. Varying $f_{z,\mathrm{WP}}$ only transfers steel from one region to the other.

:::{figure} /figures/thesis/omega_scan_iter.png
:name: fig:omega_scan
:width: 75%
:align: center

Sensitivity of the inboard leg thicknesses to $f_{z,\mathrm{WP}}$ on an ITER-like wedging/316L case.
:::

Two conclusions follow. First, the weak sensitivity of $c_\mathrm{total}$ on $f_{z,\mathrm{WP}}$ justifies treating it as a fixed input rather than introducing a dedicated convergence loop on the steel area distribution. Second, the default $f_{z,\mathrm{WP}} = 1/2$ is consistent with the ITER TF design ({numref}`Fig. %s <fig:ITER>`, Ref. {footcite:p}`federici2026iter,sborchia2008design`), which shows comparable amounts of structural steel in the WP and the nose.

```{rubric} References
```

```{footbibliography}
```
