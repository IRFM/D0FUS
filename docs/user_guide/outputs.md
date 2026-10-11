# Outputs

Every mode writes a folder named after the mode and the time, to the second,
under `D0FUS_OUTPUTS/<mode>/`. From a clone of the repository, `D0FUS_OUTPUTS` is
at the root of the repository, whatever the working directory. For an installed
package, it is in the working directory. The option `--output-dir` sets another
parent folder.

A RUN produces:

- `input_parameters.txt`, a copy of the deck;
- `output_highlight.txt`, the key results;
- `output_detailed.txt`, every input including defaults, and every output.
  Currents are in MA, densities in 10²⁰ m⁻³, pressures in MPa and energies in MJ;
- 16 figures: cross-section, Miller surfaces, profiles, q(ρ), radiation, divertor
  two-point model, radial build, coil and CICC drawings, cost breakdown,
  temperature and density maps, and the 3D machine view (PyVista, skipped with
  `--no-3d`).

A design without a solution gives NaN in the reports and the figures.

SCAN writes the 2D map. OPTIMIZATION writes the convergence plot,
`optimization_summary.json` (settings, seed, Hall of Fame), a copy of the deck and
the report files of the best design. POPCON writes the operating-contour map.
UNCERTAINTY writes a summary with the feasible share and its confidence interval,
the distribution of each output and its figures.

The quantities reported by a RUN are listed in {doc}`reference/run_outputs`.
