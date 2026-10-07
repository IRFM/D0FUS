(sec:chap1_plasma)=

# Plasma operational limits

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.2. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


A tokamak plasma cannot be operated at arbitrary density, pressure and current. Decades of experiments and theory have condensed these bounds into three operational limits, and a candidate design is only viable if it respects all three: an empirical limit on the density (Greenwald {footcite:p}`greenwald1988density,greenwald2002density`), an ideal-MHD limit on the pressure (Troyon {footcite:p}`troyon1984mhd`), and a lower bound on the edge safety factor against the external kink {footcite:p}`wesson2011tokamaks,freidberg2007plasma`.

None of the quantities entering these three limits is a direct input of the code, and producing them is the whole purpose of the plasma model chain. Following the logic popularised by Freidberg et al. {footcite:p}`freidberg2015designing` and adopted in the D0FUS reference paper {footcite:p}`auclair2025tokamak`, the prescribed inputs make demands on the plasma, and the chain unrolls these demands in causal order. The target fusion power, at fixed temperature, demands a density, and with it a pressure: the quantities the Greenwald and Troyon limits will judge. Sustaining that pressure against transport demands a confinement time, and the empirical confinement scalings convert this last demand into the needed plasma current. From that current, finally, the edge safety factor follows, to be compared against the kink limit.

The complete formulations are collected in {ref}`Plasma physics <chap:app_plasma>`.

```{toctree}
:maxdepth: 1

stability_limits
geometry
profiles
fusion_power
field_beta
power_balance
current
solver
```

```{rubric} References
```

```{footbibliography}
```
