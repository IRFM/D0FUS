# Execution modes

The mode is detected from the syntax of the input file.

| Mode | Purpose | Trigger in the deck |
|------|---------|---------------------|
| **RUN** | Single design point, report files and figures | Fixed values only (`R0 = 9`) |
| **SCAN** | 2D feasibility map over any registered output | Exactly 2 entries `[min, max, n_points]` |
| **OPTIMIZATION** | Genetic cost minimisation (DEAP) | 2+ entries `[min, max]` |
| **POPCON** | Operating map (n̄, T̄) at fixed machine | `[POPCON]` section |
| **UNCERTAINTY** | Monte Carlo propagation of input uncertainties | `[UNCERTAINTY]` section |

**RUN** solves the coupled design point. In pulsed mode it is a scalar
root-finding on the helium ash fraction (Brent). In steady state it is a 2D solve
on (f_α, Q) (Powell). After convergence, D0FUS sizes the TF and CS coils and
evaluates the flux budget, the divertor loads, the L-H margin, the costs and the
runaway-electron indicators.

**SCAN** evaluates an independent grid in parallel (`joblib`/`loky`). It overlays
the operational limits (Greenwald, Troyon, kink) and the radial-build closure on
any registered output.

**OPTIMIZATION** minimises `COE` (default), `C_invest` or `volume`, or maximises
`P_elec`. Soft penalties act on the plasma limits, with an optional capital-cost
ceiling, so the search converges toward the feasible boundary rather than away
from it.

**POPCON** freezes the converged machine of the deck and sweeps (n̄_line, T̄). It
returns Q, the powers and the H-mode access margin at every node, bounded by the
Greenwald and L-H limits. It follows the logic of the open-source `cfspopcon` and
reuses the D0FUS physics chain.

**UNCERTAINTY** assigns distributions (`norm()`, `tri()`, `unif()`) to any input,
and discrete model switches with `envelope(A | B)`. It draws Latin Hypercube
samples and returns the output distributions with their feasibility shares. Draws
without a solution count as failures. See {doc}`uncertainty_mode`.

## Execution time

Measured on a 14-core laptop.

| Mode | Configuration | Wall time |
|------|---------------|-----------|
| RUN | single point | ~200 ms |
| SCAN | 50 × 50 grid | ~25 s |
| OPTIMIZATION | 50 × 10 generations | ~2 min 30 |
| POPCON / UNCERTAINTY | N nodes or samples | ~N × 200 ms / n_cores |

The models behind each step of a run are described in {doc}`/theory/architecture`.
