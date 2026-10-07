(app:helium_ash)=

# Helium ash balance

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.4. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section derives the closed-form equilibrium helium ash fraction used in Section {ref}`Fusion power, density and pressure <ssec:chap1_fusion>`.

In a burning plasma, the alpha particles produced by fusion reactions thermalise on the bulk plasma and accumulate as helium ash unless they are removed by transport and divertor pumping fast enough. D0FUS models the ash balance using a simple model of reservoirs (following Ref. {footcite:p}`sarazin2020scaling`), in which the alpha production rate $\propto n^2 \langle\sigma v\rangle$ is balanced by alpha removal $\propto n_\alpha / \tau_\alpha^*$, where $\tau_\alpha^* = C_\alpha\, \tau_E$ is the effective alpha confinement time. The dimensionless parameter $C_\alpha$ collapses into a single number the combined effect of alpha thermalisation, helium transport in the bulk plasma, and wall recycling.

The local particle balance, integrated over the plasma volume and combined with quasi-neutrality $n_\mathrm{fuel} = n_e\,(1 - 2 f_\alpha - f_\mathrm{imp})$, produces a quadratic equation for $f_\alpha$. Defining the dimensionless combination

$$
\mathcal{C} = \bar{n}_e\, \langle\sigma v\rangle_\mathrm{vol}\,
	C_\alpha\, \tau_E
$$ (eq:C_alpha_param)

where $\langle\sigma v\rangle_\mathrm{vol}$ is the volume-averaged DT reactivity, the equilibrium ash fraction admits the closed form

$$
f_\alpha = (1 - f_\mathrm{imp})\,
	\frac{\mathcal{C}_s + 1 - \sqrt{2\,\mathcal{C}_s + 1}}{2\,\mathcal{C}_s},
	\qquad \mathcal{C}_s = \mathcal{C}\,(1 - f_\mathrm{imp})
$$ (eq:f_alpha_sarazin)

For $f_\mathrm{imp} = 0$ this reduces to the familiar form $f_\alpha = (\mathcal{C} + 1 - \sqrt{2\,\mathcal{C} + 1})/(2\,\mathcal{C})$. At ITER-like impurity content ($f_\mathrm{imp} \approx 0.07$), the dilution lowers $f_\alpha$ by about $13\,\%$. In the weak-source limit $\mathcal{C} \ll 1$ the root behaves as $f_\alpha \simeq \mathcal{C}\,(1 - f_\mathrm{imp})^2/4$, the source-times-lifetime estimate, while for $\mathcal{C} \gg 1$ it saturates at $f_\alpha \to (1 - f_\mathrm{imp})/2$, complete burn-up of the fuel.

A word on how $C_\alpha$ is set, because the parameter invites a misunderstanding. It is not a quantity one reads off a reference design: PROCESS, for instance, takes the helium fraction as input and returns $\tau_\mathrm{He}^{*}/\tau_E$ as an output, its only input on the ratio being a lower bound whose default value is $5$ {footcite:p}`kovari2014process`. $C_\alpha$ is therefore calibrated here, deck by deck, so that the predicted ash fraction lands on the one the reference design projects. This gives $C_\alpha = 5.7$ for ITER, returning $4.3\,\%$ against the $4$ to $6\,\%$ of the $Q = 10$ inductive scenario {footcite:p}`shimada2007progress`, and $C_\alpha = 6.8$ for EU-DEMO 2017, returning $8.8\,\%$ against a European baseline that assumes $c_\mathrm{He} = 10\,\%$, itself associated with $\tau_\mathrm{He}^{*}/\tau_E = 6.5$ in the 2015 iteration {footcite:p}`wenninger2017physics`. The calibration is machine-specific because it absorbs the profiles, the impurity dilution and the flux-surface volume weight of each point, which is why the same $C_\alpha$ does not carry from one deck to the other. In practice, $C_\alpha$ is bounded from above by the Reiter-Wolf-Kever ignition criterion {footcite:p}`reiter1990burn`, which sets $C_\alpha \lesssim 10$ for plasmas with realistic impurity content.

```{rubric} References
```

```{footbibliography}
```
