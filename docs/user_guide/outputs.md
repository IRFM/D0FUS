# Outputs

Every mode writes a timestamped folder under `D0FUS_OUTPUTS/`.

A RUN produces:

- `input_parameters.txt`, a copy of the deck;
- `output_highlight.txt`, the key results;
- `output_detailed.txt`, every input including defaults, and every output;
- the figures: cross-section, Miller surfaces, profiles, q(ρ), radiation, divertor
  two-point model, radial build, coil and CICC drawings, cost breakdown,
  temperature and density maps, and an optional PyVista 3D view.

SCAN writes the 2D map. OPTIMIZATION writes the convergence history and the Hall
of Fame. POPCON writes the operating-contour map. UNCERTAINTY writes the
distribution of each output, the feasibility verdict and its figures.

The quantities reported by a RUN are listed in {doc}`reference/run_outputs`.
