# Short Term
* Check the heating module with IRFM specialists -> Timothe
* Check pep8 conventions -> Arthur
* Look at the use of Greek letters in Python and decide whether to remove them -> Arthur

# Mid Term
* Document the code using Sphinx ? It produces HTML documentation via formatted comments in the code -> Arthur
* Creation of an __init__.py file ? That way, it as a pure library -> Arthur

# Long Term
* Incorporate the Finn fatigue model -> Timothe , Laura ?
* Add a test case BEST -> Timothe
* Out of Plane study : is it possible at the system code level to approximate them ? Overturning moment from I_TF x B_pol; analytical estimate would close the limitation flagged in Winding_Pack_refined (10-20% of sigma_z at A < 2.5) -> Timothe , Alex , Baptiste, Laura
* Spherical tokamak study -> Timothe
* Proper 2 point model for Divertor study -> Eva ? Timothe
* Recirculating power balance (PROCESS-style): cryogenic loads (nuclear heating in WP + AC losses during ramp), wall-plug efficiencies of H&CD actuators, pumping -> explicit Q_eng output. Requires careful curation of efficiency factors; postponed until a credible data set is assembled -> Timothe
* PF coil set sizing and positioning: lightweight filamentary solver (vertical field + field null) on top of the existing B_V / Psi_PF infrastructure, providing PF mass and cost -> Timothe
* Regression test suite (pytest) + CI (GitHub Actions): freeze the 6-machine TF/CS benchmark and the ITER reference run as automated regression tests. Planned as a second pass, after brick-by-brick validation of each module against the ITER design point -> Timothe

## Implemented (2026-06, phases 0-5 batch)

- Centralised imports + NumPy error policy replacing module-wide warning filters (phase 0).
- Model-selectable density limit: Greenwald / Giacomin PRL 2022 / Zanca NF 2019 (phase 1).
- Mavrin <Z>(Te) charge states; non-coronal SOL Lz tables (OpenADAS/radas, ne_tau = 5e16);
  Lengyel detachment diagnostic validated against cfspopcon to 0.3% (phase 2).
- POPCON execution mode (5th mode, [POPCON] deck section); design-point closure:
  P_fus and tau_E exact, P_sep < 1%, Q < 5% (phase 3).
- Sobol sensitivity analysis in the UNCERTAINTY mode (analysis = sobol); estimators
  validated on the analytic Ishigami benchmark; censoring warning + per-QoI std (phase 4).
- Figure registry + catalogue CLI (--list / --only) in D0FUS_figures standalone (phase 5).
- Tbar_mode = 'greenwald': Tbar solved by brentq to hit f_GW_target (phase 5 add-on).

- TODO: CS-chain in-module statefulness — running the CS demo blocks shifts a subsequent in-process run() d_CS (0.622->0.660); production path (fresh import) unaffected. Identify the mutated module-level state (suspects: parameterization defaults touched by figure helpers, CS caches).
