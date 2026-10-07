(app:whyte_cost)=

# Whyte surface-proportional cost model

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.6. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page details the lightweight cost model summarised in Section {ref}`Performance: net electric power and cost <ssec:chap1_cost>`.

A complementary lightweight model is also implemented, adapted from D. Whyte {footcite:p}`whyte2024fusion`. The investment cost in this model is taken proportional to the first-wall surface area, $C_\mathrm{invest} \propto S_\mathrm{FW}$, with a unit-area coefficient calibrated against ITER reference cost data. The dominant operating expense is the periodic replacement of the breeding blanket, whose annualised cost is computed by spreading the blanket unit cost over its dpa-limited operating life,

$$
C_\mathrm{rep,annual} = C_\mathrm{blanket}\, \mathrm{CF}\, \Gamma_n\, \frac{F_\mathrm{dpa}}{L_\mathrm{dpa}}
$$

where $F_\mathrm{dpa}$ is the neutron-to-dpa conversion factor for the chosen structural material and $L_\mathrm{dpa}$ is its dpa endurance limit. Tritium fuel and consumables costs are neglected. The LCOE then takes the same generic form as Eq. {eq}`eq:COE_sheffield` but with the simplified annualised CapEx and OpEx terms.

```{rubric} References
```

```{footbibliography}
```
