(ssec:chap1_refined)=

# Refined model

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.3.5. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The Refined model provides more realistic thickness predictions at the cost of moderately increased complexity, drawing on classical analytical treatments of stress distributions in fusion magnets {footcite:p}`thome1982mhd,wilson1983superconducting,boudes2025circe` and on their integration into system codes {footcite:p}`morris2015implications,reux2018demo,duchateau2014conceptual,giannini2023magnet`. Two structural changes are introduced with respect to the Academic model. First, the idealised pure-conductor layer is replaced by a composite winding pack, in which the steel jacket of the CICC coexists with the cable: two fractions then govern the sizing, the cable fraction $f_c$ set by the field-generation requirement, and the useful steel fraction $f_u$ that concentrates the radial load onto the jacket walls actually aligned with it. Second, thin-cylinder stress theory is replaced by thick-cylinder Lamé theory {footcite:p}`LameClapeyron1833`: the distributed Lorentz body force is integrated through the winding pack, which amplifies the radial compression at the bore by a geometric factor growing from unity in the thin-shell limit to about 2 for the thickest high-field coils.

The CS additionally receives two effects the Academic picture cannot see. The radial fringe field at the solenoid ends induces an axial compressive stress at the midplane, typically one order of magnitude below the hoop stress but retained in the Tresca criterion. And in the configurations where the CS hoop stress cycles at every pulse (wedging and light bucking), a configurable fatigue knockdown factor divides the steel allowable stress, consistent with preliminary-design practice {footcite:p}`jong2007iter,sarasola2020progress,sutcliffe2025magnet`. This fatigue penalty is one of the arguments weighed in Chapter 3 of the thesis when comparing mechanical architectures. A reinforcement cover completes the winding pack, following the ITER design {footcite:p}`libeyre2009detailed`.

The solution procedure is unchanged in spirit: the Tresca criterion, now written on the steel stresses of the composite pack, is saturated and solved numerically for the coil radii by iterating inward. The complete model (TF winding pack, nose, CS in the three mechanical configurations, and the radially graded variant) is given in {ref}`Refined radial build model <app:refined_model>`, and its benchmark against the dedicated magnet design code MADE and six constructed machines is reported in Chapter 3 of the thesis. For applications requiring a full multi-layer stress field, D0FUS is also interfaced with the analytical solver CIRCE {footcite:p}`boudes2025circe` ({ref}`Coupling to the CIRCE multi-layer solver <app:circe>`).

```{rubric} References
```

```{footbibliography}
```
