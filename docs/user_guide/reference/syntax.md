(app:d0fus_modes_syntax)=

# Execution-mode input syntax and the genetic optimiser

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.4, which describes D0FUS v2.7.0, and updated to the code of October 2026. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section collects the solver internals of the Run mode, the input-file syntax of the Scan and Optimization modes of Section {ref}`Execution modes <sssec:chap1_modes>`, and the detailed description of the genetic optimiser. The main entry point `D0FUS.py` detects the execution mode automatically from the input file syntax. When launched without arguments, it opens an interactive file picker that lists all available input files alongside their detected modes.

## Run-mode solver internals

In pulsed mode, the auxiliary power $P_\mathrm{aux}$ is fixed by the user, and the problem reduces to a scalar root-finding on the residual $r(f_\alpha) = f_\alpha^\mathrm{new}(f_\alpha) - f_\alpha$, solved by Brent’s method. The bracket is found by a scan of $[0, f_{\alpha,\max}]$, or near the previous root when the outer $q(\rho)$ loop solves again. Inside each evaluation, secant steps close the ohmic power to $10^{-6}$ MW. In steady-state mode, the ohmic current vanishes ($I_\Omega = 0$) and the auxiliary power becomes $P_\mathrm{aux} = P_\mathrm{fus}/Q$. The gain factor $Q$ becomes a free variable, and the system is two-dimensional. The primary solver is Powell’s hybrid method (`scipy.optimize.root`, `method=’hybr’`) with a finite-difference Jacobian, started from the previous solution when there is one and from four physically motivated initial guesses. If it fails, two fallbacks are tried in turn, a Levenberg-Marquardt step and an Anderson-accelerated two-dimensional fixed-point iteration. The first accepted solution is returned. A solution is accepted when the chain re-evaluated at $(f_\alpha, Q)$ closes both equations, to $10^{-7}$ on $f_\alpha$ and to $10^{-6}$ in relative terms on $Q$. In both modes, the $q(\rho)$ profile is held fixed during the solve and updated by the outer loop described in Section {ref}`Solver <ssec:chap1_solver>`.

## Scan mode input syntax

A Scan is triggered by bracketing exactly two parameters as $[\mathrm{min}, \mathrm{max}, n_\mathrm{points}]$. For the illustrative $(R_0, a)$ scan of {numref}`Fig. %s <fig:scan_illustration>`:

        a              = [1, 3, 50]      # scan: min, max, n_points
        R0             = [3, 9, 50]      # scan: min, max, n_points
        Operation_mode = Steady-State    # fixed override

## The genetic optimiser in detail

The Genetic mode of D0FUS is designed to actively search for a viable design that minimises (or maximises) a chosen cost function while satisfying a set of physical and engineering constraints. Instead of evaluating every point of a regular grid, it navigates the design space adaptively, concentrating evaluations in the most promising regions. The search is driven by an evolutionary genetic algorithm implemented with the DEAP library {footcite:p}`fortin2012deap`.

The input file specifies two or more parameters with their admissible range (two-element bracket, distinguishing this mode from Scan):

        R0      = [4.0, 10.0]        # optimise: min, max
        a       = [1.0, 4.0]         # optimise: min, max
        Bmax_TF = [10,  20]          # optimise: min, max
        Tbar    = [10,  20]          # optimise: min, max
        P_fus   = 2000               # fixed
        fitness_objective = COE      # objective function

Six fitness objectives are available: **COE** (minimise the levelised cost of electricity, default), **C_invest** (minimise capital cost), **P_elec** (maximise net electric power), **volume** (minimise the volume normalised to fusion power, a robust, geometry-based cost proxy), and **R0** (minimise the major radius, which gives the most compact feasible machine when $R_0$ is optimised). The sixth, **custom**, minimises an arithmetic expression given by `fitness_expression`. It may combine numbers, the numeric inputs, the outputs of the run (for instance `R0`, `a`, `B0`, `Ip`, `Q`, `P_fus`, `P_sep`, `Gamma_n`, `betaN_total`), the derived quantities `A`, `f_GW`, `P_aux`, `C_invest` (M EUR) and `COE`, the operators `+ - * / ** %` and the functions `min`, `max`, `abs`, `sqrt`, `log`, `log10`, `exp`. A lower value is better. A target can be imposed through a penalty, as in the smallest machine reaching $Q \geq 20$ with $P_{fus}$ and $P_{aux}$ optimised:

        fitness_objective  = custom
        fitness_expression = R0 + 10 * max(0, 20 - Q)

The expression is checked when the run starts, and `custom_fitness_names()` of `D0FUS_genetic` lists the names it may use. Square brackets and `#` cannot appear in it, since they mark ranges and comments in the input file.

A design is feasible under the same rule as in Run mode: the Greenwald density limit, the normalised beta limit, the kink safety factor, the plasma-current ceiling when one is set, and the closure of the radial build. The ranking puts feasibility first, following the rules of Deb: any feasible design beats any infeasible one, and infeasible designs are ranked by their total violation of the limits. A design that just fails a limit therefore ranks above one that fails it widely. This keeps the search moving toward the feasible boundary, where optimal designs typically sit. A design whose radial build does not close receives the worst score. An optional capital-cost ceiling adds a soft quadratic penalty.

The algorithm borrows its logic from biological evolution, sketched in {numref}`Fig. %s <fig:genetic_loop>`. The algorithm starts from an initial population of random tokamak designs, each one being a particular combination of the parameters to optimise. Every design is evaluated by D0FUS and assigned a fitness score, equal to the value of the cost function to minimise, penalised if any constraint is violated. A new generation is then built from the current one in three steps: the best designs are kept as parents, pairs of parents are mixed through crossover to produce children that inherit features from both, and each child undergoes a small random mutation. The new generation is evaluated, and the cycle repeats. To prevent the population from collapsing around a single design, the worst individuals are periodically replaced by fresh random candidates. The run stops automatically either when the best fitness has not improved for a configurable number of iterations, or when the maximum number of allowed generations is reached.

:::{figure} /figures/thesis/genetic_loop.png
:name: fig:genetic_loop
:width: 100%
:align: center

Schematic of the genetic algorithm loop used by D0FUS Genetic mode. An initial population of random designs feeds into a three-step evolutionary loop: evaluation by D0FUS, selection of the best as parents, and crossover and mutation to produce a new generation. The cycle repeats until the best fitness no longer improves, at which point the best design and the Hall of Fame are returned.
:::

At the end of the run, D0FUS returns the best configuration found, evaluated in full Run mode with its report files and its feasibility verdict, together with a convergence plot showing how the fitness evolved over the generations and a Hall of Fame containing the ten best individuals encountered along the way. The spread of these ten near-optimal designs also gives a first, rough indication of how flat the optimum is, and hence of the uncertainty on the optimal design. The file `optimization_summary.json` records the settings, the random seed and the Hall of Fame, and the deck is copied to the output folder, so that a run can be reproduced exactly.

```{rubric} References
```

```{footbibliography}
```
