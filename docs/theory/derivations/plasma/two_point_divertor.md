(app:two_point_model)=

# Two-point divertor model and the Lengyel integral

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.11. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section details the two-point estimate of the divertor state and of the required dissipation summarised in Section {ref}`Heat exhaust: the third first-order limit <ssec:chap1_exhaust>`.

The dissipation required, and the divertor state that results, can be estimated with a two-point model following Stangeby {footcite:p}`stangeby2018detachment` in the formulation of Kotov and Reiter {footcite:p}`kotov2009numerical`, with an implementation inspired by {footcite:p}`body2025detachment,body2024detachment`. Two control parameters carry the volumetric losses along the tube: a power-loss fraction $f_\mathrm{cooling}$ and a momentum-loss fraction $f_\mathrm{mom}$. With the upstream temperature set by Spitzer parallel conduction over the connection length $L \simeq \pi R_0\, q_{95}$, namely $T_{e,u} \simeq (7\, q_{\|u} L / 2\kappa_{0e})^{2/7}$, the separatrix density anchored to the volume-average value ($n_\mathrm{sep} = f_{n_\mathrm{sep}}\, \bar{n}$ with $f_{n_\mathrm{sep}} \approx 0.2$, in the range reported in the ITPA multi-machine database {footcite:p}`kotschenreuther2024separatrix`), and the upstream pressure $p_u = 2\, n_\mathrm{sep}\, e\, T_{e,u}$, the target electron temperature reads

$$
T_{e,t} = \frac{8\, m_f}{e\, \gamma^2}\left(\frac{q_{\|u}}{p_u}\right)^{\!2} \left(\frac{1 - f_\mathrm{cooling}}{1 - f_\mathrm{mom}}\right)^{\!2}
$$ (eq:2pm_Tet)

with $\gamma \simeq 7$ the sheath heat transmission coefficient (the energy carried through the sheath per electron-ion pair, in units of $e\,T_{e,t}$) and $m_f$ the fuel ion mass. The elementary charge $e$ in the prefactor expresses $T_{e,t}$ in eV. This single number sets the divertor regime: detachment is conventionally placed at $T_{e,t} = 10\,\mathrm{eV}$, and gross tungsten sputtering is strongly suppressed below about $5\,\mathrm{eV}$. Requiring instead that the deposited flux stay below an engineering limit $q_\mathrm{dep}^{\,\mathrm{lim}}$ fixes the minimum dissipation the scrape-off layer must provide,

$$
f_\mathrm{diss} = 1 - \frac{q_\mathrm{dep}^{\,\mathrm{lim}}}{q_{\|u}\,\sin\theta}\, \frac{R_t}{R_u}
$$ (eq:fdiss)

with $R_t/R_u$ the target-to-upstream flux expansion.

The impurity concentration required to provide a given dissipation is estimated with the Lengyel integral {footcite:p}`lengyel1981analysis`, which balances Spitzer parallel conduction against impurity line radiation along the flux tube between the target and upstream temperatures, in the formulation of {footcite:p}`body2024detachment`. At scrape-off-layer temperatures the coronal radiances of Section {ref}`Radiation losses <ssec:chap1_radiation>` do not apply: the Mavrin fits are defined above $0.1$ keV, and at the finite residence time of an impurity transiting the scrape-off layer the ionisation balance lags behind equilibrium, so partially ionised, strongly line-radiating states survive at temperatures where coronal equilibrium is fully stripped. D0FUS therefore feeds the Lengyel integral with non-coronal cooling curves generated from the OpenADAS database {footcite:p}`summers1994adas` at $n_e\tau = 5\times10^{16}$ m$^{-3}\,$s, shown in {numref}`Fig. %s <fig:chap1_Lz_SOL>`. The enhancement over the coronal fits reaches a factor 50 for nitrogen at keV temperatures and is smallest for argon.

:::{figure} /figures/thesis/chap1_Lz_SOL.png
:name: fig:chap1_Lz_SOL
:width: 100%
:align: center

Non-coronal SOL cooling curves used by D0FUS (OpenADAS, $n_e\tau = 5\times10^{16}$ m$^{-3}\,$s) and coronal Mavrin fits {footcite:p}`mavrin2018improved` on their $0.1$ to $2$ keV overlap (the fits are not defined below $0.1$ keV). The bottom panels show the ratio of the two: at finite residence time the cooling is systematically enhanced, most strongly for nitrogen, fully stripped in coronal equilibrium above a few hundred eV.
:::

```{rubric} References
```

```{footbibliography}
```
