(app:d0fus_modes_syntax)=

# Execution-mode input syntax and the genetic optimiser

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.4. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section collects the solver internals of the Run mode, the input-file syntax of the Scan and Optimization modes of Section {ref}`Execution modes <sssec:chap1_modes>`, and the detailed description of the genetic optimiser. The main entry point `D0FUS.py` detects the execution mode automatically from the input file syntax. When launched without arguments, it opens an interactive file picker that lists all available input files alongside their detected modes.

## Run-mode solver internals

In pulsed mode, the auxiliary power $P_\mathrm{aux}$ is fixed by the user, and the problem reduces to a scalar root-finding on the residual $r(f_\alpha) = f_\alpha^\mathrm{new}(f_\alpha) - f_\alpha$, solved in 8 to 12 evaluations by Brent’s method. In steady-state mode, the ohmic current vanishes ($I_\Omega = 0$) and the auxiliary power becomes $P_\mathrm{aux} = P_\mathrm{fus}/Q$. The gain factor $Q$ becomes a free variable, and the system is two-dimensional. The primary solver is Powell’s hybrid method (`scipy.optimize.root`, `method=’hybr’`) with an analytical Jacobian and four physically motivated initial guesses. If it fails, two fallbacks are tried in turn, a Levenberg-Marquardt step and an Anderson-accelerated two-dimensional fixed-point iteration. In practice, the primary solver succeeds for the vast majority of design points.

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

Four fitness objectives are available: **COE** (minimise the levelised cost of electricity, default), **C_invest** (minimise capital cost), **P_elec** (maximise net electric power), and **volume** (minimise the volume normalised to fusion power, a robust, geometry-based cost proxy).

Rather than discarding infeasible designs outright, the algorithm penalises them softly: each constraint violation degrades the fitness in proportion to its severity, so that a design that just barely fails one of the limits remains in the running. This is important because optimal designs typically sit right on the edge of the feasible region, and keeping near-feasible candidates in the population helps the search converge toward that boundary rather than fleeing it. Three plasma stability constraints are enforced by default (Greenwald density limit, normalised beta limit, kink safety factor), with an optional capital cost ceiling on top.

The algorithm borrows its logic from biological evolution, sketched in {numref}`Fig. %s <fig:genetic_loop>`. The algorithm starts from an initial population of random tokamak designs, each one being a particular combination of the parameters to optimise. Every design is evaluated by D0FUS and assigned a fitness score, equal to the value of the cost function to minimise, penalised if any constraint is violated. A new generation is then built from the current one in three steps: the best designs are kept as parents, pairs of parents are mixed through crossover to produce children that inherit features from both, and each child undergoes a small random mutation. The new generation is evaluated, and the cycle repeats. To prevent the population from collapsing around a single design, the worst individuals are periodically replaced by fresh random candidates. The run stops automatically either when the best fitness has not improved for a configurable number of iterations, or when the maximum number of allowed generations is reached.

:::{figure} /figures/thesis/genetic_loop.png
:name: fig:genetic_loop
:width: 100%
:align: center

Schematic of the genetic algorithm loop used by D0FUS Genetic mode. An initial population of random designs feeds into a three-step evolutionary loop: evaluation by D0FUS, selection of the best as parents, and crossover and mutation to produce a new generation. The cycle repeats until the best fitness no longer improves, at which point the best design and the Hall of Fame are returned.
:::

At the end of the run, D0FUS returns the best configuration found, evaluated in full Run mode with all the diagnostic figures, together with a convergence plot showing how the fitness evolved over the generations and a Hall of Fame containing the ten best individuals encountered along the way. The spread of these ten near-optimal designs also gives a first, rough indication of how flat the optimum is, and hence of the uncertainty on the optimal design.

```{rubric} References
```

```{footbibliography}
```
