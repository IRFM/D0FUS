(ssec:chap2_iter)=

# Benchmark on ITER

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 2.2.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


ITER is the natural pinnacle of this validation: it is the best documented tokamak design in existence {footcite:p}`shimada2007progress`. In contrast with the EU-DEMO exercise, where matching the 2017 PROCESS baseline required falling back on the Academic options, this benchmark runs D0FUS at its highest fidelity throughout: Miller flux-surface geometry, self-consistent $q(\rho)$ profile, Sauter-Redl bootstrap current, profile-integrated neoclassical resistivity, multi-source current drive, and the Refined radial build model. The corresponding input deck is listed in {ref}`D0FUS ITER benchmark inputs <appendixinput_iter>`.

D0FUS receives the quantities that the ITER design itself fixes: the geometry ($R_0 = 6.2$ m, $a = 2.0$ m), the inboard stack thickness $\Delta_B = 1.10$ m taken from the published radial build {footcite:p}`sborchia2008design,mitchell2011iter,libeyre2009detailed`, the fusion power $P_\mathrm{fus} = 500$ MW with 50 MW of auxiliary power (Q = 10), a 450 s burn, the IPB98(y,2) scaling at $H = 1$, the ITER pedestal width and height taken from CORSICA simulations {footcite:p}`kim2018iter` shown in {numref}`Fig. %s <fig:chap2_iter_profiles>`, and a tungsten plus neon impurity mix consistent with $Z_\mathrm{eff} = 1.65$. The helium confinement ratio $C_\alpha = 5.7$ is calibrated rather than predicted, so that the impurity-diluted ash balance returns the helium fraction of about $4.5\,\%$ expected for ITER {footcite:p}`shimada2007progress`.

:::{figure} /figures/thesis/chap2_iter_profiles.png
:name: fig:chap2_iter_profiles
:width: 100%
:align: center

Density and temperature D0FUS profiles for the ITER benchmark run.
:::

{numref}`Table %s <tab:chap2_iter_benchmark>` compares the solved design point to the published scenario. The confinement time, on-axis field, plasma current and line density, safety factor, stored energy and plasma volume all agree with the reference within a few percent.

:::{table} Full-device benchmark on the ITER Q = 10 inductive scenario. D0FUS receives the geometry, $\Delta_B$, $B_\mathrm{max}$ (conductor convention), $P_\mathrm{fus}$, $P_\mathrm{aux}$, the burn length and $\bar n/n_\mathrm{GW} = 0.85$. All listed quantities are solved for. The full input set is given in {ref}`D0FUS ITER benchmark inputs <appendixinput_iter>`. Reference values from {footcite:p}`shimada2007progress`.
:name: tab:chap2_iter_benchmark
:align: center

| Parameter                                | Symbol          | D0FUS | ITER | $\Delta$ (%) |
|:-----------------------------------------|:----------------|:------|:-----|:-------------|
| On-axis field \[T\]                      | $B_0$           | 5.62  | 5.3  | $+6.0$       |
| Plasma current \[MA\]                    | $I_p$           | 15.69 | 15.0 | $+4.6$       |
| Line-avg. density \[$10^{20}$ m$^{-3}$\] | $\bar n_l$      | 1.06  | 1.01 | $+5.1$       |
| Thermal energy \[MJ\]                    | $W_\mathrm{th}$ | 351   | 350  | $+0.3$       |
| Plasma volume \[m$^3$\]                  | $V_p$           | 848   | 831  | $+2.0$       |
| Normalised beta                          | $\beta_N$       | 1.52  | 1.8  | $-16$        |
| Confinement time \[s\]                   | $\tau_E$        | 3.71  | 3.7  | $+0.3$       |
| Safety factor                            | $q_{95}$        | 3.02  | 3.0  | $+0.5$       |
| Helium ash fraction \[%\]                | $f_\mathrm{He}$ | 4.29  | 4.4  | $-2.5$       |
:::

On the engineering side, the self-consistent chain reproduces the ITER coil set closely, as {numref}`Table %s <tab:chap2_iter_engineering>` shows.

:::{table} Coil quantities solved for by the Refined radial build for the ITER benchmark, against the published ITER values {footcite:p}`mitchell2011iter,libeyre2009detailed`.
:name: tab:chap2_iter_engineering
:align: center

| Quantity                       | Symbol               | D0FUS | ITER | $\Delta$ (%) |
|:-------------------------------|:---------------------|:------|:-----|:-------------|
| TF inboard leg thickness \[m\] | $c_\mathrm{TF}$      | 0.94  | 0.90 | $+4$         |
| CS radial thickness \[m\]      | $\Delta_\mathrm{CS}$ | 0.69  | 0.80 | $-14$        |
| CS peak field \[T\]            | $B_\mathrm{CS}$      | 11.5  | 13.0 | $-12$        |
:::

The inboard TF leg thickness lands within a few percent of the reference. The only larger residual is on the CS thickness and peak field. This is expected: the solenoid sits at the meeting point of the physics, through its flux demand, and the engineering, through its bore and its own electro-mechanical sizing.

Beyond these two single-reference exercises, D0FUS was also confronted with two other system codes, METIS-MADE and SARAS, on the pre-conceptual EUROfusion Pilot Plant B pilot studied in collaboration with the EUROfusion DEMO Central Team (Appendix E.1 of the thesis). That comparison is complementary in two respects: it confronts D0FUS with two systems codes at once, and it freezes the density and temperature profiles to the high-fidelity JETTO/HFPS predictions adopted for that study {footcite:p}`romanelli2014jintrac`, so that only the engineering models are exercised. D0FUS reproduces the plasma and magnet quantities of that reference well, the codes disperse only on the loop voltage and the internal inductance, and on both D0FUS stays the closest to the high-fidelity values, although one design point is too narrow a basis to generalise. The exercise reinforces the ITER and EU-DEMO conclusions.

```{rubric} References
```

```{footbibliography}
```
