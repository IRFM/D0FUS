(ssec:chap1_cost)=

# Performance: net electric power and cost

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.4.3. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The performance of the converged design point is summarised at two levels: the net electric power produced by the plant, and the cost of the electricity produced.

## Plant-level electrical balance

The conversion from fusion power to net electrical output proceeds in three stages. The thermal power collected by the coolant circuits is

$$
P_\mathrm{th} = M_\mathrm{blanket}\, P_n + P_\alpha + P_\mathrm{aux}
$$ (eq:Pth)

where the nuclear multiplication applies only to the neutron power $P_n \approx 0.8\, P_\mathrm{fus}$: the alpha power and the absorbed auxiliary power reach the coolants unmultiplied, through plasma transport to the divertor and radiation to the first wall. The factor $M_\mathrm{blanket}$ is the energy multiplication of the breeding blanket (typically $1.1$ to $1.3$, accounting for the exothermic $^6\mathrm{Li}(n, \alpha)\mathrm{T}$ reaction, with $4.8\,\mathrm{MeV}$ released per neutron capture). A resolved accounting of the separate divertor and first-wall cooling circuits is left for future work.

The auxiliary heating systems consume electrical power from the grid at the wall-plug efficiency $\eta_\mathrm{WP}$,

$$
P_\mathrm{WP} = \frac{P_\mathrm{aux}}{\eta_\mathrm{WP}}
$$ (eq:Pwallplug)

where $P_\mathrm{aux}$ is the total plasma-absorbed power across all heating and current-drive sources. The wall-plug efficiency $\eta_\mathrm{WP}$ varies considerably by source: lower hybrid klystrons reach $50$ to $60\,\%$, electron-cyclotron gyrotrons $40$ to $55\,\%$, neutral beam injectors $25$ to $40\,\%$ and ICRH amplifiers $70$ to $85\,\%$ {footcite:p}`gormezano2007icrh`. In Academic mode, D0FUS uses a single effective $\eta_\mathrm{WP,acad}$. In Multi-source mode, each source has its own efficiency and the wall-plug power is summed source by source. The net electrical output reads

$$
P_\mathrm{elec} = \eta_T\, P_\mathrm{th} - P_\mathrm{WP}
$$ (eq:Pelec)

with $\eta_T$ the thermal-to-electric conversion efficiency: the cycle fits of the PROCESS engineering paper give about $0.31$ to $0.34$ for a Rankine cycle at water-blanket coolant temperatures, and up to $0.45$ for helium-cooled designs at higher outlet temperature {footcite:p}`kovari2016process`. This expression neglects the balance-of-plant auxiliary loads: cryogenics, fuelling and vacuum pumping, coolant circulation and control systems. These are however not necessarily small: for EU-DEMO, Wenninger *et al.* {footcite:p}`wenninger2017physics` quote about $150$ MW of circulating power for a helium-cooled blanket and about $20$ MW for a water-cooled one, so the $Q_\mathrm{eng}$ reported here is an upper bound. A parametric balance-of-plant term is left for future work.

The engineering gain factor closes the plant-level balance,

$$
Q_\mathrm{eng} = \frac{P_\mathrm{elec}}{P_\mathrm{WP}}
$$ (eq:Q_eng)

and quantifies how much electricity the plant produces per unit of recirculating power. Net electricity production requires $Q_\mathrm{eng} > 1$ (the engineering counterpart of the plasma break-even condition $Q > 1$), but commercial viability requires $Q_\mathrm{eng}$ well above unity, so that the recirculating power fraction $1/Q_\mathrm{eng}$ stays modest, the precise threshold depending on the local electricity price and the capital cost amortisation schedule {footcite:p}`whyte2026criteria`.

## Cost models

D0FUS includes a techno-economic module, implemented by Mattéo Fletcher (GeePs, CentraleSupélec, private communication, 2026), that estimates a simplistic Levelised Cost Of Electricity (LCOE) of the converged design point. Two cost models are available: a Sheffield-Milora model, presented below, and a simplified surface-proportional model adapted from Whyte, deferred to {ref}`Whyte surface-proportional cost model <app:whyte_cost>`.

## Sheffield-Milora 2016

The reference cost model in D0FUS is the generic magnetic fusion power plant model of Sheffield and Milora {footcite:p}`sheffield2016generic`, which is itself a peer-reviewed update of the original Sheffield et al. (1986) framework {footcite:p}`sheffield1986cost`. The model decomposes the capital cost into the fusion island, the balance of plant, the buildings and auxiliary systems, and the indirect costs. The fusion island lumps the primary coils, the shield and gaps, the breeding blanket, the divertor, and the auxiliary heating and current drive systems, with each component scaled by its volume or power through the Sheffield 2016 unit-cost coefficients. The balance of plant aggregates the turbine and primary heat exchanger costs, the secondary balance-of-plant (electrical, water, instrumentation) costs scaling on $P_\mathrm{elec}$, and the buildings and site infrastructure scaling on the fusion island bounding volume $V_\mathrm{FI}$. The indirect costs include the engineering, project management, and the contingency reserve, applied as a fraction of the direct cost.

The annual operating cost is built up from the operations and maintenance contract, the consumables (tritium fuel, blanket and divertor target), and the radioactive waste disposal cost. The blanket replacement frequency is itself derived from the displacement-per-atom (dpa) limit of the structural material divided by the per-year dpa flux estimated from the neutron wall load $\Gamma_n$. The capacity factor is then computed from the time-on-load between two replacements, combined with the user-prescribed utilisation factor and dwell factor. The LCOE follows the closed-form expression

$$
\mathrm{LCOE} = \frac{10^6\, (C_\mathrm{CO}\, F_\mathrm{CRO} + C_F + C_\mathrm{OM})}{8760\, \mathrm{CF}\, P_\mathrm{elec}} + C_\mathrm{waste} \quad [\mathrm{EUR/MWh}]
$$ (eq:COE_sheffield)

where $C_\mathrm{CO}$ is the total capital cost (direct + indirect + contingency), $F_\mathrm{CRO}$ is the capital recovery factor for the user-prescribed real discount rate and plant lifetime, $C_F$ is the annual consumables cost, $C_\mathrm{OM}$ the annual operations and maintenance cost, $\mathrm{CF}$ the capacity factor, $P_\mathrm{elec}$ the net electric power of Eq. {eq}`eq:Pelec` (in MW), and $C_\mathrm{waste}$ the constant radioactive waste term. The factor $10^6$ converts the numerator from M\$/year to \$/year, while the factor $8760$ converts the denominator to MWh/year. The internal computation is in 2010 USD (consistent with the Sheffield 2016 unit costs), the final output is converted to 2025 EUR through a fixed exchange-and-inflation factor. The capital term $C_\mathrm{CO}$ aggregates the direct component costs evaluated from the Sheffield 2016 unit-cost tables (magnets, blanket and structure, heating systems, buildings and balance of plant), completed by the indirect-cost and contingency allowances of the same reference. The waste term is a flat per-MWh charge taken from the Sheffield model, $C_\mathrm{waste} = 5$ \$/MWh (2010 USD), covering the disposal of activated components: being a per-MWh charge, it adds directly to the LCOE, outside the capital annualisation.

A complementary lightweight model adapted from Whyte {footcite:p}`whyte2024fusion`, with an investment cost proportional to the first-wall area and the periodic blanket replacement as the dominant operating expense, is deferred to {ref}`Whyte surface-proportional cost model <app:whyte_cost>`.

```{rubric} References
```

```{footbibliography}
```
