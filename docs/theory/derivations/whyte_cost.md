(app:whyte_cost)=

# Whyte surface-proportional cost model

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.6. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This page details the lightweight cost model summarised in Section {ref}`Performance: net electric power and cost <ssec:chap1_cost>`.

A complementary lightweight model is also implemented, adapted from D. Whyte {footcite:p}`whyte2024fusion`, and selected by `cost_model = Whyte`. It serves as a benchmark of the Sheffield-Milora model. The investment cost in this model is taken proportional to the first-wall surface area, $C_\mathrm{invest} \propto S_\mathrm{FW}$, with a unit-area coefficient calibrated against ITER reference cost data. The dominant operating expense is the periodic replacement of the breeding blanket, whose annualised cost is computed by spreading the blanket unit cost over its dpa-limited operating life,

$$
C_\mathrm{rep,annual} = C_\mathrm{blanket}\, f_\mathrm{burn}\, \Gamma_n\, \frac{F_\mathrm{dpa}}{L_\mathrm{dpa}} = C_\mathrm{blanket}\, \frac{f_\mathrm{burn}}{t_\mathrm{bl}}
$$

where $F_\mathrm{dpa}$ is the neutron-to-dpa conversion factor for the chosen structural material, $L_\mathrm{dpa}$ is its dpa endurance limit and $t_\mathrm{bl} = L_\mathrm{dpa} / (F_\mathrm{dpa}\, \Gamma_n)$ the blanket lifetime in full-power years. $f_\mathrm{burn} = A_v\, U\, D$ is the burn fraction of the year. D0FUS takes $t_\mathrm{bl}$ and $A_v$ from its own lifetime and replacement-schedule model, so that both cost models share the same plant operation. Tritium fuel, consumables and O&M costs are neglected. The LCOE then takes the same generic form as Eq. {eq}`eq:COE_sheffield`,

$$
\mathrm{LCOE} = \frac{10^6\, (C_\mathrm{invest}\, F_\mathrm{CRO} + C_\mathrm{rep,annual})}{E_\mathrm{net}}
$$

with $E_\mathrm{net}$ the annual net electricity of Eq. {eq}`eq:Enet`.

```{rubric} References
```

```{footbibliography}
```
