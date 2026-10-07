(app:jc_scalings)=

# Superconductor critical-current scalings and quench protection

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix C.16. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section gives the critical-current parameterisations and the quench-protection criterion summarised in Section {ref}`Superconductors and engineering current density <ssec:chap1_supraconductors>`.

(sssec:chap1_NbTi)=

## NbTi scaling

The NbTi scaling follows the ITER/EU-DEMO parameterisation {footcite:p}`corato2016common`:

$$
J_{\text{non-Cu}}(B, T) = \frac{C_0}{B}\,(1 - t^{1.7})^\gamma\, b^\alpha\,(1 - b)^\beta \qquad [\mathrm{A/mm^2}]
$$ (eq:Jc_NbTi)

where $t = T/T_{c0}$ is the reduced temperature, $b = B/B_{c2}(T)$ the reduced field with $B_{c2}(T) = B_{c2,0}(1 - t^{1.7})$ the upper critical field beyond which superconductivity is lost. The exponents $\alpha$ and $\beta$ control the field dependence, while $\gamma$ governs the temperature roll-off near $T_{c0}$. The corresponding parameters are given in {numref}`Table %s <tab:NbTi_params>`.

:::{table} NbTi scaling parameters (ITER/EU-DEMO) {footcite:p}`corato2016common`
:name: tab:NbTi_params
:align: center

| **Parameter** | **Value** | **Unit**         |
|:--------------|:----------|:-----------------|
| $T_{c0}$      | 9.03      | K                |
| $B_{c2,0}$    | 14.61     | T                |
| $C_0$         | 168512    | A$\cdot$T/mm$^2$ |
| $\alpha$      | 1.0       | \-               |
| $\beta$       | 1.54      | \-               |
| $\gamma$      | 2.1       | \-               |
:::

(sssec:chap1_Nb3Sn)=

## Nb$_3$Sn scaling

The Nb$_3$Sn scaling follows the EU-DEMO WST (Western Superconducting Technologies) strand parameterisation {footcite:p}`bottura2009jc,corato2016common` and accounts for the sensitivity to mechanical strain $\varepsilon$, which is particularly important for this brittle A15 compound. A strain function $s(\varepsilon)$ modifies both the critical temperature and the upper critical field:

$$
s(\varepsilon) = 1 + \frac{C_{a1}}{1 - C_{a1}\,\varepsilon_{0a}} \left(\sqrt{\varepsilon_{0a}^2} - \sqrt{\varepsilon^2 + \varepsilon_{0a}^2}\right)
$$ (eq:strain_function)

with $T_{c0}^*(\varepsilon) = T_{cm}\, s(\varepsilon)^{1/3}$ and $B_{c2}^*(\varepsilon, T) = B_{c2m}\, s(\varepsilon)\,(1 - t^{1.52})$ where $t = T/T_{c0}^*$. The critical current density then reads:

$$
J_{\text{non-Cu}}(B,T,\varepsilon) = \frac{C}{B}\, s(\varepsilon)\,(1 - t^{1.52})(1 - t^2)\, b^p\,(1 - b)^q \qquad [\mathrm{A/mm^2}]
$$ (eq:Jc_Nb3Sn)

with $b = B/B_{c2}^*$. The exponents $p$ and $q$ govern the field dependence, while the temperature dependence is carried by the $(1 - t^{1.52})(1 - t^2)$ factor. The parameters for the EU-DEMO WST strand are given in {numref}`Table %s <tab:Nb3Sn_params>`. A default effective strain of $\varepsilon = -0.6$ % following MADMACS convention.

:::{table} Nb$_3$Sn scaling parameters (EU-DEMO WST strand) {footcite:p}`corato2016common`
:name: tab:Nb3Sn_params
:align: center

| **Parameter**      | **Value**           | **Unit**         |
|:-------------------|:--------------------|:-----------------|
| $T_{cm}$           | 16.34               | K                |
| $B_{c2m}$          | 33.24               | T                |
| $C$                | 83075               | A$\cdot$T/mm$^2$ |
| $C_{a1}$           | 50.06               | \-               |
| $\varepsilon_{0a}$ | $3.12\times10^{-3}$ | \-               |
| $p$                | 0.593               | \-               |
| $q$                | 2.156               | \-               |
:::

(sssec:chap1_REBCO)=

## REBCO scaling

The default REBCO model follows the Dew-Hughes pinning-force scaling of Senatore *et al.* {footcite:p}`senatore2024rebco`, calibrated on modern tapes (Fujikura FESC 2019 and SuperOx 2019). At a reference operating point $(B_\mathrm{ref}, T_\mathrm{ref})$ where the non-copper current density is known from transport measurements:

$$
J_{\text{non-Cu}}(B,T) = J_{\text{non-Cu},\mathrm{ref}}\,\exp\!\left(-\frac{T - T_\mathrm{ref}}{T^*}\right) \frac{f_p(B,T)}{f_p(B_\mathrm{ref},T_\mathrm{ref})}
$$ (eq:Jc_REBCO)

where the pinning-force shape function is $f_p(B,T) = b^{p-1}(1-b)^q$ with $b = B/B_\mathrm{irr}(T)$, and the irreversibility field decreases with temperature as $B_\mathrm{irr}(T) = B_{\mathrm{irr},0}\,(1 - (T/T_c)^{n_1})^{n_2}$. The parameters for Fujikura FESC 2019 tapes (used by default in D0FUS) are given in {numref}`Table %s <tab:REBCO_params_senatore>`. The worst-case perpendicular field orientation ($B \perp$ tape, $\theta = 0$) is assumed throughout, which is conservative since the parallel orientation yields significantly higher $J_{\text{non-Cu}}$. The earlier Fleiter/CERN parameterisation {footcite:p}`fleiter2014rebco,bajas2022ship` is also available as an option for cross-comparisons.

:::{table} REBCO scaling parameters (Senatore 2024, Fujikura FESC 2019 tape) {footcite:p}`senatore2024rebco`
:name: tab:REBCO_params_senatore
:align: center

| **Parameter**                       | **Value**     | **Unit** |
|:------------------------------------|:--------------|:---------|
| $T_{c}$                             | 93.0          | K        |
| $B_{\mathrm{irr},0}$                | 187           | T        |
| $n_1$, $n_2$                        | 0.40, 1.0     | \-       |
| $p$, $q$                            | 0.77, 4.5     | \-       |
| $T^*$                               | 22            | K        |
| $J_{\text{non-Cu},\mathrm{ref}}$    | 2000          | A/mm$^2$ |
| $(B_\mathrm{ref},\,T_\mathrm{ref})$ | (19 T, 4.2 K) | \-       |
:::

## Maddock hot-spot criterion and copper fraction

The copper-to-non-copper ratio $r_{\text{Cu/non-Cu}}$ is not a free parameter: it is determined by quench protection through the Maddock adiabatic hot-spot criterion {footcite:p}`maddock1969,wilson1983superconducting`. When a superconductor quenches, its current transfers very quickly to the copper stabiliser, which then carries the full operating current resistively for the few seconds it takes the protection system to detect the event and discharge the coil. During this transient, the local copper temperature can rise dramatically, the Maddock criterion expresses the energy balance between the Joule heating in the copper and the enthalpy rise of all the materials in the conductor, from the operating temperature up to a maximum allowable hot-spot temperature $T_\mathrm{hs}$:

$$
\int_{T_\mathrm{op}}^{T_\mathrm{hs}} \frac{\rho_\mathrm{mat}(T)\,c_{v,\mathrm{mat}}(T)}{\rho_\mathrm{Cu}(T,B,\mathrm{RRR})}\,\mathrm{d}T = J_\mathrm{Cu,0}^{\,2}\!\left(\tau_h + \frac{\tau_d}{2}\right)
$$ (eq:maddock)

where $\rho_\mathrm{mat}\, c_{v,\mathrm{mat}}$ is the volumetric heat capacity, evaluated with the copper stabiliser properties alone, a conservative simplification of the composite conductor (copper, superconductor, helium, jacket), $\rho_\mathrm{Cu}$ the copper magnetoresistivity, $\tau_h$ the quench detection and hold time, and $\tau_d = 2\, E_\mathrm{mag}/(N_\mathrm{sub}\, I_0\, V_\mathrm{max})$ the exponential decay time constant of the discharge circuit, set by the total magnet stored energy $E_\mathrm{mag}$, the operating current $I_0$, the maximum dump voltage $V_\mathrm{max}$ and the number of independently-protected subdivisions $N_\mathrm{sub}$ (each with its own dump resistor).

Solving Eq. {eq}`eq:maddock` for the copper current density $J_\mathrm{Cu,0}$ gives the minimum copper cross-section per unit non-copper area, which translates directly into $r_{\text{Cu/non-Cu}}$. Default D0FUS values are $T_\mathrm{hs} = 250$ K (which corresponds to a $\sim$150 K real hot-spot once the enthalpy of the structural material is accounted for), $\mathrm{RRR} = 100$, $V_\mathrm{max} = 10$ kV (consistent with the ITER fast-discharge design {footcite:p}`sborchia2008design`), and $\tau_h = 3$ s for LTS or 10 s for HTS (the longer HTS value reflects the slower quench propagation, which may makes quench detection harder). For the subdivision count, the TF system groups two coils per dump unit, so $N_\mathrm{sub} = N_\mathrm{coil}/2$ (i.e. 9 for an 18-coil machine), while the CS uses its $N_\mathrm{sub} = 6$ modules, following the ITER protection scheme. The magnet stored energies $E_\mathrm{mag,TF}$ and $E_\mathrm{mag,CS}$ are computed analytically from the coil geometry through a thick toroidal inductor formula for the TF system and a thick solenoid formula for the CS. The corresponding module in D0FUS is actually adapted from the CEA magnet design code MADMACS {footcite:p}`zani2019parametric,sutcliffe2025magnet`.

```{rubric} References
```

```{footbibliography}
```
