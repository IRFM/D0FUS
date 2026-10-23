(ssec:chap1_cost)=

# Performance: net electric power and cost

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.4.3, which describes D0FUS v2.7.0, and updated to the code of October 2026. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
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

where $P_\mathrm{aux}$ is the total plasma-absorbed power across all heating and current-drive sources. The wall-plug efficiency $\eta_\mathrm{WP}$ varies considerably by source: lower hybrid klystrons reach $50$ to $60\,\%$, electron-cyclotron gyrotrons $40$ to $55\,\%$, neutral beam injectors $25$ to $40\,\%$ and ICRH amplifiers $70$ to $85\,\%$ {footcite:p}`gormezano2007icrh`. In Academic mode, D0FUS uses a single effective $\eta_\mathrm{WP,acad}$. In Multi-source mode, each source has its own efficiency and the wall-plug power is summed source by source. The net electrical output is the gross output of the thermal cycle minus the recirculated power,

$$
P_\mathrm{elec} = \eta_T\, \bar{P}_\mathrm{th} - \langle P_\mathrm{recirc} \rangle, \qquad
\langle P_\mathrm{recirc} \rangle = D\, P_\mathrm{var} + (1 - D)\, P_\mathrm{CS} + P_\mathrm{fix}
$$ (eq:Pelec)

$$
P_\mathrm{var} = \frac{P_\mathrm{aux}}{\eta_\mathrm{WP}} + f_\mathrm{pump}\, P_\mathrm{th}, \qquad
P_\mathrm{fix} = P_\mathrm{cryo} + P_\mathrm{house}
$$ (eq:Precirc)

with $\eta_T$ the gross thermal-to-electric efficiency and $f_\mathrm{pump}$ the coolant pumping load as a fraction of the thermal power. The values of $M_\mathrm{blanket}$, $\eta_T$ and $f_\mathrm{pump}$ are those of the blanket concept chosen, taken from the comparison of Clark *et al.* {footcite:p}`clark2025blanket`, unless the deck sets them. The concept value of $M_\mathrm{blanket}$ is reduced when the breeder is thinner than the concept reference.

In pulsed operation, a thermal store delivers a constant power $\bar{P}_\mathrm{th}$ over the whole cycle, flat-top and dwell, so that $P_\mathrm{elec}$ is a cycle average. Each load is given at its nominal value, which sizes the plant. The heating and the coolant pumping $P_\mathrm{var}$ run during the burn, $D$ being the flat-top fraction of the cycle (`Dwell_factor`). The central solenoid is recharged during the dwell, at $P_\mathrm{CS} = E_\mathrm{CS} / t_\mathrm{dwell}$. The cryoplant $P_\mathrm{cryo}$, scaled on the magnet cold mass from the EU-DEMO reference, and the house load $P_\mathrm{house}$ run all the time. The duty cycle enters only once, through the cycle average $\langle P_\mathrm{recirc} \rangle$. The house load follows the PROCESS model {footcite:p}`kovari2016process`: a fixed plant base, a term proportional to the building floor area, the tritium plant, the vacuum pumps and the TF power supplies. In steady state, $D = 1$ and $\bar{P}_\mathrm{th} = P_\mathrm{th}$. $P_\mathrm{elec}$ can be negative. The plant then produces no net electricity and the cost of electricity is undefined.

The engineering gain factor closes the plant-level balance,

$$
Q_\mathrm{eng} = \frac{\eta_T\, \bar{P}_\mathrm{th}}{\langle P_\mathrm{recirc} \rangle}
$$ (eq:Q_eng)

and quantifies how much electricity the plant produces per unit of recirculating power. Net electricity production requires $Q_\mathrm{eng} > 1$ (the engineering counterpart of the plasma break-even condition $Q > 1$), but commercial viability requires $Q_\mathrm{eng}$ well above unity, so that the recirculating power fraction $1/Q_\mathrm{eng}$ stays modest, the precise threshold depending on the local electricity price and the capital cost amortisation schedule {footcite:p}`whyte2026criteria`.

## Cost models

D0FUS includes a techno-economic module, implemented by Mattéo Fletcher (GeePs, CentraleSupélec, private communication, 2026), that estimates a simplistic Levelised Cost Of Electricity (LCOE) of the converged design point. Two cost models are available, selected by `cost_model`: a Sheffield-Milora model (`Sheffield`, the reference), presented below, and a simplified surface-proportional model adapted from Whyte (`Whyte`, a benchmark), deferred to {ref}`Whyte surface-proportional cost model <app:whyte_cost>`.

Both models divide an annual cost by the annual net electricity

$$
E_\mathrm{net} = 8760 \left[\mathrm{CF}\, P_\mathrm{elec} - (1 - \mathrm{CF})\, P_\mathrm{fix}\right] \quad [\mathrm{MWh/yr}]
$$ (eq:Enet)

where $\mathrm{CF} = A_v\, U$ is the operating fraction of the year, with $A_v$ the availability set by the replacement schedule and $U$ the utilisation factor. The fixed loads are drawn from the grid all year, outages included. CF does not contain the dwell, which is already in the cycle-averaged $P_\mathrm{elec}$. The burn fraction of the year, $A_v\, U\, D$, sets the fluence and the tritium burn.

## Sheffield-Milora 2016

The reference cost model in D0FUS is the generic magnetic fusion power plant model of Sheffield and Milora {footcite:p}`sheffield2016generic`, which is itself a peer-reviewed update of the original Sheffield et al. (1986) framework {footcite:p}`sheffield1986cost`. The model decomposes the capital cost into the fusion island, the balance of plant, the buildings and auxiliary systems, and the indirect costs. The fusion island lumps the primary coils, the shield and gaps, the breeding blanket, the divertor, and the auxiliary heating and current drive systems, with each component scaled by its volume or power through the Sheffield 2016 unit-cost coefficients. The balance of plant aggregates the turbine and primary heat exchanger costs, the secondary balance-of-plant (electrical, water, instrumentation) costs scaling on $P_\mathrm{elec}$, and the buildings and site infrastructure scaling on the fusion island bounding volume $V_\mathrm{FI}$. In pulsed operation, the thermal scaling of the balance of plant uses the turbine-side power $\bar{P}_\mathrm{th}$, and the thermal store is costed at 27 \$/kWh$_\mathrm{th}$, the two-tank molten-salt value of Turchi and Heath (NREL/TP-5500-57625, 2013). This value may be optimistic for a blanket coolant colder than solar salt. The indirect costs include the engineering, project management, and the contingency reserve, applied as a fraction of the direct cost.

The annual operating cost is built up from the operations and maintenance contract, the consumables (tritium fuel, blanket and divertor target), and the radioactive waste disposal cost. The blanket lifetime is derived from the displacement-per-atom (dpa) limit of the structural material divided by the per-year dpa flux estimated from the neutron wall load $\Gamma_n$. The divertor lifetime follows from its integrated heat load. The two components are replaced in parallel. The shorter-lived one is replaced after $t_B$ of operation, the other one with the $n$-th replacement, $n = \lfloor t_A / t_B \rfloor$. A component replaced every $t_\mathrm{rep}$ of operation uses $A_v\, T_\mathrm{life} / t_\mathrm{rep}$ sets over the plant life, the first one being in the capital cost. The LCOE follows the closed-form expression

$$
\mathrm{LCOE} = \frac{10^6\, (C_\mathrm{CO}\, F_\mathrm{CRO} + C_F + C_\mathrm{OM})}{E_\mathrm{net}} + C_\mathrm{waste} \quad [\mathrm{EUR/MWh}]
$$ (eq:COE_sheffield)

where $C_\mathrm{CO}$ is the total capital cost (direct + indirect + contingency), $F_\mathrm{CRO}$ is the capital recovery factor for the user-prescribed real discount rate and plant lifetime, $C_F$ is the annual consumables cost, $C_\mathrm{OM}$ the annual operations and maintenance cost, $E_\mathrm{net}$ the annual net electricity of Eq. {eq}`eq:Enet`, and $C_\mathrm{waste}$ the constant radioactive waste term. The factor $10^6$ converts the numerator from M\$/year to \$/year. The internal computation is in 2010 USD (consistent with the Sheffield 2016 unit costs), the final output is converted to 2025 EUR through a fixed exchange-and-inflation factor. The capital term $C_\mathrm{CO}$ aggregates the direct component costs evaluated from the Sheffield 2016 unit-cost tables (magnets, blanket and structure, heating systems, buildings and balance of plant), completed by the indirect-cost and contingency allowances of the same reference. The waste term is a flat per-MWh charge, $C_\mathrm{waste} = 5$ \$/MWh (2010 USD), covering the disposal of activated components. It is above the 1.5 \$/MWh of Sheffield and Milora (0.5 for waste, 1.0 for decommissioning). Being a per-MWh charge, it adds directly to the LCOE, outside the capital annualisation.

A complementary lightweight model adapted from Whyte {footcite:p}`whyte2024fusion`, with an investment cost proportional to the first-wall area and the periodic blanket replacement as the dominant operating expense, is deferred to {ref}`Whyte surface-proportional cost model <app:whyte_cost>`.

```{rubric} References
```

```{footbibliography}
```
