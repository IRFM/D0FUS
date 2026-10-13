# Execution modes

The mode is detected from the content of the input file. Text after `#` is a
comment, and section titles such as `[POPCON]` are optional.

| Mode | Purpose | Trigger in the deck |
|------|---------|---------------------|
| **RUN** | Single design point, report files and figures | Fixed values only (`R0 = 9`) |
| **SCAN** | 2D feasibility map over any registered output | Exactly 2 entries `[min, max, n_points]`, other than the POPCON pair |
| **OPTIMIZATION** | Genetic cost minimisation (DEAP) | 2+ entries `[min, max]` |
| **POPCON** | Operating map (n̄, T̄) at fixed machine | `nbar_line` and `Tbar` written `[min, max, n_points]` |
| **UNCERTAINTY** | Monte Carlo propagation of input uncertainties | At least one `tri()`, `norm()`, `unif()` or `envelope()` entry |

**RUN** solves the coupled design point. In pulsed mode it is a scalar
root-finding on the helium ash fraction (Brent). In steady state it is a 2D solve
on (f_α, Q) (Powell). An outer loop updates the q(ρ) profile used by the
bootstrap model and solves again, usually once. After convergence, D0FUS sizes the TF and CS coils and
evaluates the flux budget, the divertor loads, the L-H margin, the costs and the
runaway-electron indicators.

**SCAN** evaluates an independent grid in parallel (`joblib`/`loky`). It overlays
the operational limits (Greenwald, Troyon, kink) and the radial-build closure on
any registered output.

**OPTIMIZATION** minimises `COE` (default), `C_invest`, `volume` or `R0`,
maximises `P_elec`, or minimises an expression of the inputs and outputs
written in the deck (`custom`, see {doc}`reference/syntax`). Feasibility follows the RUN rule and comes first in the
ranking: any feasible design beats any infeasible one, and infeasible designs are
ranked by their violation of the limits. An optional capital-cost ceiling adds a
soft penalty.

**POPCON** freezes the converged machine of the deck and sweeps (n̄_line, T̄). It
returns Q, the powers and the H-mode access margin at every node, bounded by the
Greenwald, Troyon and L-H limits. It follows the logic of the open-source `cfspopcon` and
reuses the D0FUS physics chain.

**UNCERTAINTY** assigns distributions (`norm()`, `tri()`, `unif()`) to any input,
and discrete model switches with `envelope(A | B)`. It draws Latin Hypercube
samples and returns the output distributions and one feasible share, with its 95 %
Wilson confidence interval. Draws without a solution count as failures. See {doc}`uncertainty_mode`.

## Command-line options

```bash
python D0FUS.py DECK [--output-dir DIR] [--no-figures] [--no-3d] [--seed N]
```

| Option | Effect |
|--------|--------|
| `--output-dir DIR` | Parent folder of the result folders, instead of `D0FUS_OUTPUTS` |
| `--no-figures` | Reports and data files only |
| `--no-3d` | RUN: skips the 3D machine view (PyVista), the slowest figure |
| `--seed N` | Random seed of OPTIMIZATION and UNCERTAINTY, overriding the deck |
| `-h` | Help, with the deck syntax of each mode |

Without a deck, `D0FUS.py` lists the files of `D0FUS_INPUTS` with their detected
mode.

## Execution time

Measured on a 14-core laptop.

| Mode | Configuration | Wall time |
|------|---------------|-----------|
| RUN | single point | ~200 ms |
| SCAN | 50 × 50 grid | ~25 s |
| OPTIMIZATION | 50 × 10 generations | ~2 min 30 |
| POPCON / UNCERTAINTY | N nodes or samples | ~N × 200 ms / n_cores |

The models behind each step of a run are described in {doc}`/theory/architecture`.
