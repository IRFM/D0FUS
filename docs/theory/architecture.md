(sec:chap1_architecture)=

# General architecture of D0FUS

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Section 1.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


D0FUS (Design 0-Dimensional for Fusion Systems) is built with a clear goal: to provide a tool where every physical assumption is traceable, every design trade-off is interpretable, and where a new user can understand the full modelling chain in a matter of days rather than months. In practice, this translates into four concrete choices:

- D0FUS is distributed under the CeCILL-C licence[^1] and has been publicly available on GitHub[^2] since February 2026. Every equation, every approximation can be inspected, questioned, and if needed, corrected by anyone.

- The entire code is written in Python, with NumPy and SciPy as the only computational dependencies. There is no compilation step, no Makefile, no platform-specific configuration. A simple `pip install d0fus` installs the package from PyPI and provides a working installation on any machine with a standard Python distribution. The code runs equally well in a Jupyter notebook on a laptop, in an interactive session in Spyder, or as a batch job on a cluster. Three installation paths are documented in {ref}`Repository layout, installation and library usage <app:d0fus_repo>`.

- Rather than relying on complex numerical solvers, D0FUS implements closed-form or semi-analytical expressions for every physical and engineering quantity. Two levels of fidelity are available: a straightforward *Academic* level based on textbook formulas {footcite:p}`freidberg2007plasma,wesson2011tokamaks`, and a *Refined* level that incorporates more refined physical effects and an advanced treatment of technological constraints, while remaining fully traceable. Both levels share the same interface and wherever possible the same numerical solver, so that the impact of any modelling assumption can be isolated by changing a single parameter.

- The code is decomposed into small, self-contained functions, each implementing one well-defined physical or engineering model. Every function carries a docstring (the standard in-code documentation string of Python) that documents its assumptions, its validity domain, and the publication from which it was taken. Adding a new confinement scaling law, a new superconductor parameterisation, or a new cost model is a matter of writing one function and registering it in the appropriate lookup table. Testing a modification requires no recompilation and no restart: the new function can be called interactively, in isolation, from a notebook cell. At the time of writing, D0FUS consists of approximately 39 000 lines of Python distributed across five functional modules (plus a visualisation library), totalling 490 documented functions.

The supporting material (the organisation of the library into six domain modules, the repository layout, the installation paths, the dependencies and library-usage examples) is collected in {ref}`Repository layout, installation and library usage <app:d0fus_repo>` and {ref}`D0FUS module organisation <app:d0fus_modules>`.

(ssec:chap1_fidelity)=
For most of its models, D0FUS thus provides two layers. The Academic layer relies on textbook closed forms: it is the natural entry point to the code and a convenient way to introduce each model and its logic, which makes it particularly useful for pedagogy. The Refined layer implements, for each model, the most refined formulation identified in this work while remaining tractable and understandable at the system-code level. {numref}`Table %s <tab:fidelity_levels>` summarises the differences between the two layers. The selectors, the coherent presets and the remaining details are given in {ref}`Fidelity levels and model selectors <app:d0fus_fidelity>`.

:::{table} Comparison of the Academic and Refined fidelity levels in D0FUS.
:name: tab:fidelity_levels
:align: center

| Model             | Academic                          | Refined                                              |
|:------------------|:----------------------------------|:-----------------------------------------------------|
| Plasma geometry   | Elliptical torus, $\delta = 0$    | Miller flux surfaces, $\kappa(\rho)$, $\delta(\rho)$ |
| Volume element    | $V' = 4\pi^2 R_0 a^2 \kappa \rho$ | Numerical on $(N_\rho \times N_\theta)$ grid         |
| Bootstrap current | Segal-Cerfon-Freidberg            | Sauter-Redl (Sauter 1999/2002 + Redl 2021)           |
| $q(\rho)$ profile | Assumed parabolic                 | Self-consistent                                      |
| TF stress model   | Thin-cylinder, two-layer          | Thick-cylinder, composite CICC                       |
| CS stress model   | Hoop stress only                  | Hoop + axial fringe-field stress                     |
| Resistivity       | Spitzer (classical)               | Sauter/Redl (neoclassical)                           |
:::

Beyond the practical outcome of producing a working system code, starting from a blank page, with every model re-derived or deliberately adopted rather than inherited, revealed areas where existing modelling choices could not capture the specificities of HTS-based compact configurations. Addressing these gaps required dedicated models, notably modern REBCO critical current density scaling laws and flexible mechanical configuration options (bucking, wedging, plug), which are presented in these pages.

(sssec:chap1_modes)=

## Execution modes

D0FUS provides four execution modes, all built on the same elementary brick: the Run of a consistent single design point. Because a Run is fast (a fraction of a second, see the execution times below), a set of exploration tools has been developed around it: Scan evaluates a two-dimensional grid and returns either design-space maps or the POPCON operating maps {footcite:p}`houlberg1982contour` of a fixed machine, Optimization searches the design space adaptively, and Uncertainties propagates input distributions through the chain. The main entry point detects the execution mode automatically from the input file syntax ({ref}`Execution-mode input syntax and the genetic optimiser <app:d0fus_modes_syntax>`).

### Run mode

The Run of a single design point is the elementary building block of D0FUS. A Run is specified by a handful of primary inputs, the machine size $(R_0, a)$, the peak conductor field $B_\mathrm{max}$, the inboard stack thickness $\Delta_B$, the target fusion power, the auxiliary power (or the target gain in steady state) and the technology choices, and solves self-consistently for the plasma state and the coil sizes. It returns 167 scalar outputs, organised into thirteen families ({ref}`Run mode outputs <chap:run_outputs>`) and written to a text file accompanied by sixteen automatically generated diagnostic figures (kinetic profiles, radial build assembly, coil cross-sections, cost breakdown, and so on), several of which illustrate these pages. The self-consistent solver that ties it together will be presented below, once the physics chain has been introduced (Section {ref}`Solver <ssec:chap1_solver>`).

### Scan mode

The second execution mode evaluates a complete design point at every node of a two-dimensional grid: exactly two parameters are given as $[\mathrm{min}, \mathrm{max}, n_\mathrm{points}]$ brackets in the input file, all others taking either their file-specified value or their default value (the input syntax is illustrated in {ref}`Execution-mode input syntax and the genetic optimiser <app:d0fus_modes_syntax>`). As an illustrative example, consider a scan for a steady-state tokamak over the major radius $R_0$ and the minor radius $a$, with $50 \times 50 = 2500$ complete Run evaluations, one per grid point, parallelised across all available CPU cores.

The user can then overlay any registered output on the 2D map as colour background, iso-contours, or both. {numref}`Fig. %s <fig:scan_illustration>` shows the result of the example scan above. Each D0FUS map displays the plasma stability domain: the background colours encode which limit is most restrictive (blue: Greenwald, red: Troyon, green: kink), with darker shading indicating stronger violation. The solid white contour marks the plasma stability boundary, while the solid black contour indicates the radial build limit, beyond which the CS + TF inner leg do not fit. The intersection of the two encloses the viable design region, inside which dashed black contours show iso-values of a chosen parameter, here the plasma current $I_p$. [^3]

:::{figure} /figures/thesis/IllustrationSteadyState.png
:name: fig:scan_illustration
:width: 100%
:align: center

Illustrative output of D0FUS Scan mode for a steady-state $(R_0, a)$ scan with all other parameters at their default values, in particular the prescribed fusion power, the peak field on the TF conductor and the steady-state operation assumptions. Coloured backgrounds show the three normalised plasma limits ($n/n_G$, $\beta_N/\beta_{N,\mathrm{lim}}$, $q_\mathrm{lim}/q_{95}$). The white contour is the plasma stability boundary, the black contour is the radial build limit. Their intersection encloses the viable design region. Dashed lines are iso-contours of the plasma current $I_p$.
:::

These maps are a particularly powerful tool for visualising the design space at a glance, and for assessing how it evolves switching from one assumption to another.

### POPCON maps

The same scan machinery produces a second kind of map. The POPCON (Plasma OPerating CONtour) maps, introduced by Houlberg *et al.* {footcite:p}`houlberg1982contour` and widely used in American design studies, scan not the machine design space but the plasma operation space of one fixed machine. D0FUS evaluates a complete power balance at every node of a density-temperature grid, returning the fusion gain $Q$, the fusion and auxiliary powers, and the H-mode access margin $P_\mathrm{sep}/P_{\text{L-H}}$. The accessible operating window then emerges as the region bounded by the Greenwald density limit, the L-H power threshold, and a chosen target contour (iso-$Q$ or iso-$P_\mathrm{fus}$). The D0FUS implementation reproduces the logic of the open-source `cfspopcon` framework {footcite:p}`cfspopcon2024`. These maps are requested through a dedicated `[POPCON]` block in the input file, inside which the line-averaged density and the temperature grids are bracketed. An illustrative output for the ITER baseline is shown in {numref}`Fig. %s <fig:popcon_iter>`.

:::{figure} /figures/thesis/POPCON_ITER.png
:name: fig:popcon_iter
:width: 100%
:align: center

Illustrative POPCON output of D0FUS for the ITER baseline, in the line-averaged density / volume-averaged temperature plane. The colour map is the driven fusion gain $\log_{10} Q$. Solid blue and dashed green lines are iso-contours of the fusion and auxiliary powers, the red line marks the L-H access boundary $P_\mathrm{sep} = P_{\text{L-H}}$, the purple line the Greenwald density limit, and the dotted black lines the iso-$Q$ contours. The red star is the converged ITER design point on the $Q = 10$ contour (the plasma current quoted in the title is the value converged by D0FUS, discussed in {ref}`Validation and benchmarks <chap:validation>`).
:::

### Optimization mode

The Genetic mode actively searches for a viable design that minimises a chosen objective (the levelised cost of electricity by default, with capital cost, net electric power and a volume-based cost proxy also available) under the plasma stability and radial build constraints, using a genetic algorithm built on the DEAP library {footcite:p}`fortin2012deap`. Instead of evaluating a regular grid, it navigates the design space adaptively: a population of candidate designs is evaluated by D0FUS, the best are selected as parents, and crossover and mutation produce the next generation. Constraint violations degrade the fitness, which makes the search converge toward the feasible region. The run returns the best configuration evaluated in full Run mode, a convergence plot, and a Hall of Fame of the ten best individuals, whose spread gives a first indication of how flat the optimum is. The input syntax, the loop schematic and the implementation details are given in {ref}`Execution-mode input syntax and the genetic optimiser <app:d0fus_modes_syntax>`.

What motivates a genetic search here is not its speed but its robustness. {numref}`Fig. %s <fig:genetic_toy>` tests it on a synthetic cost landscape carrying several minima and infeasible regions, of the kind that feasibility limits create in the parameter plane of a system study. Run with the exact D0FUS operators and settings, the optimiser reaches the global feasible optimum, and does so on all thirty random instances of the problem.

:::{figure} /figures/thesis/genetic_toy_landscape.png
:name: fig:genetic_toy
:width: 100%
:align: center

Robustness test of the Optimization mode on a synthetic cost landscape, run with the D0FUS operators and default settings (population of 200 individuals, 50 generations): two broad local minima (crosses), one deeper but narrower global minimum (star), infeasible holes (white). The four frames show the population (red) from the initial draw to convergence. The layout shown is a median case among thirty random ones. On all thirty, the optimiser reaches the true feasible optimum.
:::

(par:chap1_uncertainties)=

### Uncertainties mode

A single Run evaluation returns one deterministic design point, but several of its inputs are actually known only within a range, and some of its physics models carry a substantial extrapolation uncertainty. The Uncertainties mode quantifies how this input spread propagates to the outputs by Monte Carlo sampling. The user assigns a probability distribution to any subset of inputs, and optionally a discrete set of alternative model choices. D0FUS then draws $N$ samples, runs a full Run evaluation for each, and returns the resulting distribution of every output, summarised by its median and user-selected confidence intervals. The samples being independent of one another, the sampling parallelises trivially and reuses the same `joblib`/`loky` backend as the Scan mode. This mode is only presented here as an execution mode of the code: the uncertainties themselves, their sources and their propagation through the design chain, are the subject of an entire chapter ({ref}`Limitations of a zero-dimensional system code <chap:uncertainties>`).

### Execution time.

On a modern 14-core workstation (the laptop used to develop D0FUS), a single Run-mode evaluation takes approximately 200 ms, a $50 \times 50$ Scan completes in about 25 s thanks to the parallel evaluation of independent grid points, and a Genetic optimisation with 50 individuals over 10 generations runs in roughly 2 minutes 30 seconds. These execution times make interactive design-space exploration entirely feasible without access to a computing cluster.

[^1]: Compatible with the GNU LGPL. See <https://cecill.info/licences/Licence_CeCILL-C_V1-en.html>.

[^2]: <https://github.com/IRFM/D0FUS>

[^3]: Note that on typical D0FUS maps, the diagonal of these $(R_0, a)$ scan is a line of constant aspect ratio, here chosen as $A = R_0/a = 3$, so that moving along it amounts to a pure size scan at fixed shape.

```{rubric} References
```

```{footbibliography}
```
