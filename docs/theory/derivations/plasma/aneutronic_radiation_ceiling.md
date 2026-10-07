(app:aneutronic_radiation)=

# The radiation ceiling of aneutronic fuels

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.8. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


The main text notes that the aneutronic fuels, and proton-boron-11 in particular, are held back by a radiation ceiling that the reference D-T reaction does not face. This page makes that statement quantitative, using the same bremsstrahlung and synchrotron models that D0FUS applies to a D-T plasma, Eqs. {eq}`eq:Pbrem` and {eq}`eq:Psyn`.

Two features set proton-boron-11 apart. Its reactivity peaks near an ion temperature of order two hundred keV, roughly an order of magnitude above the D-T optimum, and its fuel carries a boron nucleus of charge $Z = 5$, so that even a modest boron fraction lifts the effective charge $Z_\mathrm{eff}$ well above unity. Both facts work against the power balance: the fusion power density at the optimum is no larger than for D-T, whereas the radiated power, which grows with temperature and with $Z_\mathrm{eff}$, is far larger.

Bremsstrahlung is already close to prohibitive on its own. Since $P_\mathrm{brem} \propto Z_\mathrm{eff}\, n_e^2\, T^{1/2}$ from Eq. {eq}`eq:Pbrem` and rises only as $T^{1/2}$, it cannot be outrun by raising the temperature, and with the elevated $Z_\mathrm{eff}$ of a boron plasma it becomes, in thermal equilibrium, comparable to the proton-boron fusion power itself. Synchrotron emission then settles the question. Its steep temperature dependence, $P_\mathrm{syn} \propto T_0\,(16 + T_0)^{2.61}$, which tends to $T_0^{3.61}$ in the hot limit of Eq. {eq}`eq:Psyn`, means that moving from a D-T plasma near $20~\mathrm{keV}$ to a proton-boron plasma near $200~\mathrm{keV}$ multiplies the temperature factor of the synchrotron loss by about three orders of magnitude, while the fusion power does not follow. Left to itself the synchrotron loss then exceeds the fusion output.

The only relief is to return the radiation to the plasma. The wall reflectivity enters Eq. {eq}`eq:Psyn` through the factor $(1 - R_w)^{0.62}$, so that a highly reflecting wall reabsorbs most of the emitted photons. Requiring the reflected synchrotron loss to fall back below the fusion power, $(1 - R_w)^{0.62} \lesssim P_\mathrm{fus}/P_\mathrm{syn,0}$, gives

$$
1 - R_w \;\lesssim\; \left(\frac{P_\mathrm{fus}}{P_\mathrm{syn,0}}\right)^{1/0.62},
$$ (eq:reflectivity_requirement)

where $P_\mathrm{syn,0}$ is the loss that a perfectly absorbing wall would allow. Even in the most favourable reading, an unreflected loss a single order of magnitude above the fusion power, $P_\mathrm{syn,0}/P_\mathrm{fus} \sim 10$, Eq. {eq}`eq:reflectivity_requirement` requires $1 - R_w \lesssim 2\times 10^{-2}$ with the $0.62$ exponent, and $1 - R_w \lesssim 10^{-2}$ with the classical Trubnikov exponent of one half: a reflectivity of order $99\,\%$ is the entry point, and it is the floor quoted in the thesis introduction. For the two orders of magnitude that the temperature factor above actually suggests, $P_\mathrm{syn,0}/P_\mathrm{fus} \sim 10^{2}$, the demand tightens to $1 - R_w \lesssim 6\times 10^{-4}$ with the $0.62$ exponent, a reflectivity of $99.9\,\%$, and to $10^{-4}$, or $99.99\,\%$, with the Trubnikov one.

Such a reflectivity cannot survive contact with a real machine. The first wall is pierced by heating antennas, diagnostic windows and the divertor throat, and these openings return essentially none of the radiation that reaches them. If a fraction $f_\mathrm{open}$ of the surface is non-reflecting, the area-averaged reflectivity cannot exceed $1 - f_\mathrm{open}$, and because a synchrotron photon reflects many times before it is finally absorbed it samples those openings on nearly every bounce, so their effect is if anything worse than this simple bound. A few percent of openings, unavoidable in any device that has to be heated and diagnosed, therefore holds the effective reflectivity near $95$ to $98\,\%$, short even of the most favourable $99\,\%$ bound, let alone the $99.9\,\%$ of a realistic loss ratio. This is the sense in which the reflecting-wall remedy is illusory, and it is the quantitative basis for the statement, in the thesis introduction, that proton-boron-11 faces a radiation ceiling from which D-T is spared. The conclusion is not new: the classical assessments of the proton-boron fuel reached it from the same power-balance arguments {footcite:p}`moreau1977potentiality,nevins1998review`.

```{rubric} References
```

```{footbibliography}
```
