(app:cd_models)=

# Current drive efficiency models

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix B.9. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section details the four source-specific current-drive efficiency models summarised in Section {ref}`Plasma current and scaling law <ssec:chap1_heating_cd>`.

Four current drive sources are available, each with a physics-based efficiency model.

For LHCD (lower hybrid current drive), the efficiency follows the default METIS model {footcite:p}`artaud2018metis`:

$$
\gamma_\mathrm{LH} = \frac{2.4}{5 + Z_\mathrm{eff}} \tanh\!\left(\frac{\bar{T}_e\,[\mathrm{keV}]}{6}\right)
$$ (eq:gamma_LH)

The $1/(5+Z_\mathrm{eff})$ prefactor is the Spitzer parallel conductivity correction for Landau-damped current drive {footcite:p}`fisch1987current`, and the numerical coefficients are calibrated against Tore Supra in line with the ITER Physics Basis {footcite:p}`gormezano2007icrh`.

For ECCD (electron cyclotron current drive), the efficiency follows the Giruzzi {footcite:p}`giruzzi1987eccd` trapped-electron model with the $Z_\mathrm{eff}$ correction of Lin-Liu et al. {footcite:p}`linliu2003eccd`:

$$
\gamma_\mathrm{EC} = \frac{T_{e,\mathrm{loc}}}{T_{e,\mathrm{loc}} + 100\,\mathrm{keV}}
	\times G_\mathrm{trap}(\varepsilon, \theta_p, Z_\mathrm{eff})
	\times \frac{6}{1 + 4 f_c + Z_\mathrm{eff}}
$$ (eq:gamma_EC)

where $T_{e,\mathrm{loc}}$ is the local electron temperature (in keV) at the EC deposition radius $\rho_\mathrm{EC}$, $\varepsilon = \rho_\mathrm{EC}\,a/R_0$ is the local inverse aspect ratio, $\theta_p$ is the poloidal angle of the deposition point, and $f_c = 1 - \sqrt{2\varepsilon/(1+\varepsilon)}$ is the passing-particle fraction. The trapped-particle correction reads:

$$
G_\mathrm{trap} = 1 - \left(1 + \frac{\hat{\rho}}{3}\right) \left(\sqrt{2}\,\mu_t\right)^{\hat{\rho}}
$$ (eq:Gtrap)

with $\hat{\rho} = (5+Z_\mathrm{eff})/(1+Z_\mathrm{eff})$ and $\mu_t = \sqrt{\varepsilon(1+\cos\theta_p)/(1+\varepsilon\cos\theta_p)}$ the pitch-angle boundary for trapped orbits at the poloidal deposition angle.

The poloidal angle $\theta_p$ is an input parameter: $\theta_p = 0$ corresponds to outboard midplane injection (LFS, conservative), and $\theta_p = 180^\circ$ to inboard midplane (HFS, optimal for current drive because $\mu_t = 0$ and $G_\mathrm{trap} = 1$).

The model evaluates the local $T_e$ at the user-specified deposition radius $\rho_\mathrm{EC}$, computes the trapped-particle factor from Eqs. {eq}`eq:Gtrap` and the surrounding definitions, and returns the assembled $\gamma_\mathrm{EC}$.

ICRH (ion cyclotron resonance heating) is treated as pure heating ($\gamma_\mathrm{ICR} = 0$): its power enters $P_\mathrm{aux}$ but not $I_\mathrm{CD}$. Fast wave current drive with asymmetric antenna phasing is left to future work.

For NBCD (neutral beam current drive), a beam ion injected at energy $E_b$ slows down on the same isotropic Stix distribution as the fusion alphas (Section {ref}`Magnetic field and plasma beta <ssec:chap1_field_beta>`), now evaluated at the deposition radius $\rho_\mathrm{NBI}$ for the beam ions ($A_b$, $Z_b = 1$). The Spitzer drag time $\tau_\mathrm{se}$ of Eq. {eq}`eq:tau_se` and the Stix critical energy $E_c$ of Eq. {eq}`eq:Ec_stix` carry over directly. The driven current, however, calls for a different moment of the same distribution: where $G_\mathrm{eff}$ in Eq. {eq}`eq:Wfast` weighted velocity space by stored energy, NBCD weights it by the carried current. Following METIS {footcite:p}`artaud2018metis`, three corrections then enter $\gamma_\mathrm{NBI}$.

*(i) Cordey velocity-space integral.* Above $v_c$, electron drag slows the beam ions without scattering their direction: below $v_c$, ion drag isotropises them on a comparable timescale and washes out the carried current. The effective current-carrying velocity is therefore the slowing-down average of $v$ between $v_b$ and $v_c$ {footcite:p}`cordey1979nbcd`:

$$
v_\mathrm{eff} = v_c \left(\frac{v_b^3 + v_c^3}{v_b^3}\right)^{\varepsilon_v - 1} \int_0^{v_b/v_c} x \left(\frac{x^3}{1+x^3}\right)^{\varepsilon_v}\!\mathrm{d}x
$$ (eq:cordey_integral)

with $v_b = \sqrt{2 E_b / (A_b m_p)}$ the beam injection velocity, $v_c = \sqrt{2 E_c / (A_b m_p)}$, and $\varepsilon_v = 1 + \tfrac{2}{3}(v_g/v_c)^3$, where $v_g = \sqrt{2 E_{c,\gamma} / (A_b m_p)}$ is the velocity associated with the CD-modified critical energy $E_{c,\gamma} = 14.8\,T_e\,(2\sqrt{A_b}\,Z_\mathrm{eff})^{2/3}$. The textbook scaling $v_\mathrm{eff} \propto \sqrt{E_b/A_b}$ is recovered in the limit $v_b \gg v_c$.

*(ii) Lin-Liu electron return current.* The bulk electrons partially short-circuit the beam ion flow through a parallel return current. In a cylindrical plasma this cancellation is maximal, set by the Spitzer conductivity: in a tokamak, trapped electrons cannot carry parallel current, so more of the beam-driven current survives. The Lin-Liu neoclassical fit {footcite:p}`linliu1997trapping` captures the residual:

$$
1 - \frac{1 - G(Z_\mathrm{eff}, f_\mathrm{trap})}{Z_\mathrm{eff}}
$$ (eq:linliu_factor)

with $G$ a rational function of $f_\mathrm{trap}/(1 - f_\mathrm{trap})$ and $Z_\mathrm{eff}$, going from $0$ in the cylindrical limit ($f_\mathrm{trap} \to 0$, residual $1 - 1/Z_\mathrm{eff}$) to $1$ in the strongly trapped limit (residual $\to 1$).

*(iii) Orbit trapping screening.* A beam ion launched with pitch $\mu = \cos\theta_\mathrm{NBI}$ below the local trapping cone $\mu_\mathrm{trap} = \sqrt{2\varepsilon / (1+\varepsilon)}$, with $\varepsilon = \rho_\mathrm{NBI}\,a/R_0$, is on a banana orbit from the start and carries no net current. A smoothed indicator $f_i$ suppresses $\gamma_\mathrm{NBI}$ when $\mu$ falls below this threshold.

Assembling the three corrections,

$$
\gamma_\mathrm{NBI} = \frac{e}{2\pi\,E_b}\,(\tau_\mathrm{se}\, n_e)\,|\mu|\,v_\mathrm{eff}\,\left(1 - \frac{1 - G}{Z_\mathrm{eff}}\right)\,f_i
$$ (eq:gamma_NBCD)

where $\tau_\mathrm{se}\, n_e$ is almost density-independent, so $\gamma_\mathrm{NBI}$ depends on $n_e$ only weakly through $\ln\Lambda$ and the ion composition. Engineering inputs are $A_b$, $E_b$, $\theta_\mathrm{NBI}$ (default $20^\circ$ from tangential), and the helium ash fraction $f_\alpha$. The chain matches METIS to within $1.5\,\%$ on three machines cases (ITER 1 MeV D, ARC 150 keV D, EU-DEMO 1 MeV D).

```{rubric} References
```

```{footbibliography}
```
