
[![License: CeCILL-C](https://img.shields.io/badge/License-CeCILL--C-blue.svg)](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html)
[![PyPI version](https://badge.fury.io/py/d0fus.svg)](https://pypi.org/project/d0fus/)

<a name="readme-top"></a>

# D0FUS

**D0FUS** (Design 0-dimensional for Fusion Systems) is a Python tokamak systems code for fast 0D/1D design-space exploration, covering plasma physics, superconducting magnet engineering, and techno-economic assessment. It is developed at CEA-IRFM.

---

## Installation

Three installation options are available. **Option A is recommended for most users** (no prior Python knowledge required).

---

### Option A: Spyder standalone (recommended)

This is the simplest path. Spyder comes with its own bundled Python, with no separate Python installation needed.

#### Step 1: Download and install Spyder

Go to [spyder-ide.org](https://www.spyder-ide.org/) and download the installer for your operating system (Windows, macOS, or Linux). Run the installer and follow the on-screen instructions. Default settings are fine.

#### Step 2: Download D0FUS

**If you have git installed**, open a terminal (macOS/Linux) or Command Prompt (Windows) and run:
```bash
git clone https://github.com/IRFM/D0FUS.git
```

**Otherwise**, go to [github.com/IRFM/D0FUS](https://github.com/IRFM/D0FUS), click **Code → Download ZIP**, and extract the archive somewhere on your computer (e.g. `C:\Users\you\D0FUS\` on Windows or `~/D0FUS/` on macOS/Linux).

#### Step 3: Open Spyder

Launch Spyder from your applications menu (or desktop shortcut). You should see a layout with a code editor on the left, a file explorer on the top right, and an **IPython console** on the bottom right.

#### Step 4: Open D0FUS.py in Spyder

Go to **File → Open…** and open `D0FUS.py` from the `D0FUS/` folder. Spyder will automatically set the working directory to `D0FUS/`. You can confirm this by checking the folder path displayed in the toolbar at the top.

#### Step 5: Install dependencies

In the **IPython console** (bottom-right panel), type the following and press **Enter**:

```
%pip install -r requirements.txt
```

The `%pip` magic command installs packages directly into Spyder's internal Python. Since the Spyder standalone bundles its own pip, no prior installation is needed. Since the working directory is already set to `D0FUS/`, no path specification is needed. Wait for all packages to finish installing.

> **If `%pip` fails**, try: `!pip install -r requirements.txt`. If pip itself is missing (unlikely with the standalone), reinstalling Spyder usually fixes it.

#### Step 6: Run D0FUS

Press **F5** (or click the green ▶ **Run file** button). A file-picker dialog will appear. Navigate to `D0FUS_INPUTS/` and select `1_run_ITER.txt`. D0FUS will run and print results to the IPython console. Figures will open automatically.

---

### Option B: Miniforge (for advanced users)

[Miniforge](https://github.com/conda-forge/miniforge) is a free, community-maintained conda distribution with no commercial restrictions. Recommended if you manage multiple Python projects.

**Install Miniforge**, then open a Miniforge Prompt (Windows) or terminal (macOS/Linux):

```bash
conda create -n d0fus python=3.11
conda activate d0fus
conda install pip
conda install spyder
git clone https://github.com/IRFM/D0FUS.git
cd D0FUS
pip install -r requirements.txt
spyder
```

Once Spyder is open, follow Steps 5–6 from Option A. Alternatively, run directly from the terminal:

```bash
python D0FUS.py
# When prompted, enter the path to the input file:
# D0FUS_INPUTS/1_run_ITER.txt
```

---

### Option C: pip install (headless / script integration)

For use without a GUI, or to import D0FUS as a library in your own scripts:

```bash
pip install d0fus
```

To run the ITER example case from a script:

```python
from D0FUS_EXE import D0FUS_run

results = D0FUS_run.run("D0FUS_INPUTS/1_run_ITER.txt")
print(f"Q = {results['Q']:.1f},  COE = {results['COE']:.0f} EUR/MWh")
```

> Note: input files (`D0FUS_INPUTS/`) are not included in the PyPI package. Clone the repository separately to access them.

---

## Project Structure

```
D0FUS/
├── D0FUS_BIB/                          # Core library modules
│   ├── D0FUS_import.py                     # Centralised imports
│   ├── D0FUS_parameterization.py           # Physical constants, GlobalConfig dataclass
│   ├── D0FUS_physical_functions.py         # Plasma physics (profiles, bootstrap, q, RE…)
│   ├── D0FUS_radial_build_functions.py     # Engineering (TF/CS/CICC/quench)
│   ├── D0FUS_cost_functions.py             # Techno-economic models (Sheffield, Whyte)
│   ├── D0FUS_cost_data.py                  # Reference cost data and currency conversions
│   └── D0FUS_figures.py                    # Full figure catalogue (27+ figures)
│
├── D0FUS_INPUTS/                       # Input parameter files
│   ├── 1_run_ITER.txt                      # ITER Q=10 reference case (RUN mode)
│   ├── 2_scan_ITER.txt                     # 2D scan around ITER point (SCAN mode)
│   └── 3_genetic_ITER.txt                  # Genetic optimisation around ITER (OPTIMIZATION mode)
│
├── D0FUS_OUTPUTS/                      # Generated outputs (auto-created)
│   ├── Run_D0FUS_YYYYMMDD_HHMMSS/         # Single run results + figures
│   ├── Scan_D0FUS_YYYYMMDD_HHMMSS/        # 2D scan maps
│   └── Genetic_D0FUS_YYYYMMDD_HHMMSS/     # Optimisation results
│
├── D0FUS_EXE/                          # Execution modules
│   ├── D0FUS_run.py                        # Single design point
│   ├── D0FUS_scan.py                       # 2D parameter scan
│   └── D0FUS_genetic.py                    # Genetic algorithm optimisation
│
├── D0FUS.py                            # Main entry point
├── requirements.txt
└── README.md
```

---

## Execution Modes

D0FUS detects the execution mode from the input file format:

| Mode | Purpose | Input format | Parameters |
|------|---------|--------------|------------|
| **RUN** | Single design point | `R0 = 9` | Fixed values only |
| **SCAN** | 2D parameter space | `R0 = [3, 9, 25]` | Exactly 2 parameters with `[min, max, n_points]` |
| **OPTIMIZATION** | Genetic algorithm cost minimisation | `R0 = [3, 9]` | 2+ parameters with `[min, max]` |

**RUN mode** evaluates a single tokamak configuration and outputs all plasma parameters, magnetic fields, power balance, radial build, cost of electricity, and RE indicators.

**SCAN mode** generates 2D maps over two parameters, visualising feasibility regions bounded by stability limits (Greenwald, Troyon, kink) and engineering constraints. Iso-contours of any output quantity registered in `OUTPUT_REGISTRY` can be overlaid.

**OPTIMIZATION mode** uses a DEAP genetic algorithm to find the configuration minimising cost (COE by default) while satisfying all physics and engineering constraints. The budget ceiling `C_invest_max` is enforced via an exponential penalty.

---

## Input

### Parameter Handling

All parameters have physically motivated default values (DEMO-class baseline). When an input file is provided, only the specified parameters are overwritten. This allows minimal input files:

```ini
R0 = 7
Bmax_TF = 14
Supra_choice = REBCO
```

### Parameter Reference

#### Geometry

| Parameter | Description | Unit | Default |
|-----------|-------------|------|---------|
| `P_fus` | Fusion power | MW | 2000 |
| `R0` | Major radius | m | 9.0 |
| `a` | Minor radius | m | 3.0 |
| `b` | Blanket + shield radial thickness | m | 1.2 |
| `Option_Kappa` | Elongation model | — | `Wenninger` |
| `κ_manual` | Manual elongation (if `Option_Kappa = Manual`) | — | 1.9 |

Options for `Option_Kappa`: `Wenninger`, `Stambaugh`, `Freidberg`, `Manual`.

#### Magnetic Field

| Parameter | Description | Unit | Default |
|-----------|-------------|------|---------|
| `Bmax_TF` | Peak field on TF conductor | T | 12.0 |
| `Bmax_CS_adm` | Admissible peak field on CS | T | 25.0 |

#### Technology

| Parameter | Description | Unit | Default | Options |
|-----------|-------------|------|---------|---------|
| `Supra_choice` | Superconductor material | — | `Nb3Sn` | `NbTi`, `Nb3Sn`, `REBCO` |
| `Radial_build_model` | Stress model | — | `D0FUS` | `academic`, `D0FUS`, `CIRCE` |
| `Choice_Buck_Wedg` | TF mechanical configuration | — | `Wedging` | `Plug`, `Bucking`, `Wedging` |
| `Chosen_Steel` | Structural steel grade | — | `316L` | `316L`, `N50H`, `JK2LB`, `Manual` |

#### Plasma Physics

| Parameter | Description | Unit | Default | Options |
|-----------|-------------|------|---------|---------|
| `Scaling_Law` | Energy confinement scaling law | — | `IPB98(y,2)` | `IPB98(y,2)`, `ITPA20`, `ITPA20-IL`, `DS03`, `L-mode`, `ITER89-P` |
| `H` | Confinement enhancement factor | — | 1.0 | |
| `Tbar` | Volume-averaged ion temperature | keV | 14.0 | |
| `Plasma_profiles` | Profile peaking preset | — | `H` | `L`, `H`, `Advanced`, `Manual` |
| `nu_n_manual` | Density peaking factor (Manual only) | — | 0.1 | |
| `nu_T_manual` | Temperature peaking factor (Manual only) | — | 1.0 | |
| `rho_ped` | Normalised pedestal radius | — | 1.0 | |
| `n_ped_frac` | Pedestal density fraction n_ped/n̄ | — | 0.0 | |
| `T_ped_frac` | Pedestal temperature fraction T_ped/T̄ | — | 0.0 | |
| `Bootstrap_choice` | Bootstrap current model | — | `Redl` | `Freidberg`, `Segal`, `Sauter`, `Redl` |
| `Option_q95` | q₉₅ formula | — | `Sauter` | `Sauter`, `ITER_1989` |
| `L_H_Scaling_choice` | L-H threshold scaling | — | `New_Ip` | `Martin`, `New_S`, `New_Ip` |
| `Plasma_geometry` | Volume integral geometry | — | `Academic` | `Academic`, `D0FUS` |
| `Zeff` | Effective plasma charge | — | 2.0 | |
| `impurity_species` | Impurity species (radiation) | — | `''` | `W`, `Ar`, `Ne`, `C`, `N`, `Kr` |
| `f_imp_core` | Impurity concentration n_imp/n_e | — | `''` | |
| `rho_rad_core` | Core/edge radiation boundary | — | 0.75 | |

#### Operation

| Parameter | Description | Unit | Default | Options |
|-----------|-------------|------|---------|---------|
| `Operation_mode` | Operating scenario | — | `Pulsed` | `Steady-State`, `Pulsed` |
| `Temps_Plateau_input`* | Flat-top burn duration | s | 3600 | |
| `P_aux_input`* | Auxiliary heating power | MW | 50 | |
| `CD_source` | Current drive model | — | `Academic` | `Academic` (recommended), `LHCD`, `ECCD`, `NBCD`, `Multi` (dev.) |
| `gamma_CD_acad` | CD figure of merit (Academic mode) | MA/(MW·m²) | 0.20 | |
| `eta_WP_acad` | Wall-plug efficiency (Academic mode) | — | 0.40 | |

*Only relevant when `Operation_mode = Pulsed`.

> **Note on CD models**: only `CD_source = Academic` is fully validated. Technology-specific models (`LHCD`, `ECCD`, `NBCD`, `Multi`) are under active development and should not be used for production runs.

#### Disruption / Runaway Electron Indicators

These parameters control the post-convergence RE indicator computation. They do **not** enter the main physics solver.

| Parameter | Description | Unit | Default |
|-----------|-------------|------|---------|
| `tau_TQ` | Thermal quench e-folding time | s | 1e-3 |
| `Te_final_eV` | Post-TQ residual electron temperature | eV | 5.0 |
| `pellet_dilution` | Density multiplication from SPI/MGI | — | 10.0 |

#### Techno-Economic Model

| Parameter | Description | Unit | Default |
|-----------|-------------|------|---------|
| `cost_model` | Cost model | — | `Sheffield` |
| `discount_rate` | Real discount rate | — | 0.07 |
| `T_life` | Plant operational lifetime | yr | 40 |
| `T_build` | Construction duration | yr | 10 |
| `contingency` | Contingency fraction | — | 0.15 |
| `Util_factor` | Utilisation factor | — | 0.85 |
| `Supra_cost_factor` | SC coil cost multiplier vs Cu | — | 2.0 |
| `C_invest_max` | Capital cost ceiling (genetic) | M EUR | 25 000 |

Set `cost_model = None` to skip cost computation entirely.

### Input File Format

**RUN mode**: fixed values only:
```ini
R0 = 9
Bmax_TF = 13
Supra_choice = Nb3Sn
```

See `D0FUS_INPUTS/1_run_ITER.txt` for a complete example.

**SCAN mode**: exactly 2 scan parameters with `[min, max, n_points]`:
```ini
R0 = [3, 9, 25]
a  = [1, 3, 25]
```

See `D0FUS_INPUTS/2_scan_ITER.txt` for a complete example.

**OPTIMIZATION mode**: 2+ parameters with `[min, max]`:
```ini
R0     = [3, 9]
a      = [1, 3]
Bmax_TF = [10, 16]
fitness_objective = COE
C_invest_max = 20000
```

See `D0FUS_INPUTS/3_genetic_ITER.txt` for a complete example.

### Genetic Algorithm Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `population_size` | Individuals per generation | 50 |
| `generations` | Maximum generations | 100 |
| `crossover_rate` | Crossover probability | 0.7 |
| `mutation_rate` | Mutation probability | 0.2 |
| `fitness_objective` | Quantity to minimise | `COE` |

---

## Output

### RUN Mode Output

```
D0FUS_OUTPUTS/Run_D0FUS_YYYYMMDD_HHMMSS/
├── input_parameters.txt        # Copy of input configuration
├── output_results.txt          # Complete calculation results
└── figures/                    # 10 run-specific PNG figures (150 dpi)
    ├── 01_cross_section.png
    ├── 02_miller_surfaces.png
    ├── 03_shaping_profiles.png
    ├── 04_kinetic_profiles.png
    ├── 05_q_profile.png
    ├── 06_radiation_profile.png
    ├── 07_TF_side_view.png
    ├── 08_CICC_TF.png
    ├── 09_CS_cross_section.png
    └── 10_CICC_CS.png
```

**Output quantities include:**
- Plasma parameters: Ip, n̄_e, T̄, β_N, Q, τ_E, q₉₅, f_bs
- Magnetic fields: B0, B_CS, B_pol
- Power balance: P_fus, P_CD, P_rad, P_sep, P_elec
- Radial build: TF coil thickness, CS inner/outer radius, first-wall area
- Current profile: j_Ohm(ρ), j_CD(ρ), j_bs(ρ), q(ρ) (self-consistent iteration)
- Techno-economics: COE [EUR/MWh], C_invest [M EUR]
- RE indicators: I_RE_seed, I_RE_aval, f_RE/Ip, E_RE_kin

### SCAN Mode Output

```
D0FUS_OUTPUTS/Scan_D0FUS_YYYYMMDD_HHMMSS/
├── scan_parameters.txt
└── scan_map_[iso]_[bg].png     # 2D map (300 dpi)
```

**Available scan output quantities** (`OUTPUT_REGISTRY`):

| Category | Quantities |
|----------|-----------|
| Performance | Q, P_fus, P_elec, COE, C_invest |
| Plasma | Ip, n̄, β_N, β_T, q₉₅, τ_E, f_bs, f_Greenwald |
| Fields | B0, B_CS, B_pol |
| Power | P_rad, P_sep, P_CD, P_Ohm |
| Engineering | d_TF, d_CS, S_FW |
| RE indicators | I_RE_seed, I_RE_aval, f_RE_Ip, f_RE_avg, f_RE_core, E_RE_kin |

### OPTIMIZATION Mode Output

```
D0FUS_OUTPUTS/Genetic_D0FUS_YYYYMMDD_HHMMSS/
├── optimization_config.txt     # Bounds, algorithm settings
├── optimization_results.txt    # Best solution, COE, C_invest
└── convergence_plot.png        # Fitness evolution over generations
```

---

## Figures Catalogue

`plot_run()` generates 10 run-specific figures. `plot_all()` generates the full 27-figure catalogue, including:

| Group | Figures |
|-------|---------|
| Geometry & shaping | κ scaling, LCFS comparison, Miller flux surfaces, κ(ρ)/δ(ρ) profiles |
| Kinetics | n(ρ), T(ρ), p(ρ) profiles; line- vs volume-averaged density |
| Radiation | Lz(T) cooling functions, P_rad(ρ) profile, helium-ash fraction |
| Current & q | q(ρ) with j_Ohm/j_CD/j_bs decomposition |
| Magnets | Jc(B,T) scaling (NbTi/Nb₃Sn/REBCO), TF thickness vs B_max, CS thickness vs flux swing |
| Engineering drawings | Princeton-D TF side view, CS cross-section, CICC cross-section (hex-packed strands) |
| Benchmarks | TF coil benchmark table, CS benchmark table, CIRCE0D stress validation |

---

## Physics Models

### Plasma Geometry

D0FUS supports two plasma geometry models, selected via `Plasma_geometry`:

| Model | Description | When to use |
|-------|-------------|-------------|
| `Academic` | Cylindrical-torus approximation, uniform κ and δ | Large parameter scans, fast runs |
| `D0FUS` | Full Miller flux-surface parameterisation (Miller et al. 1998) with PCHIP κ(ρ) and δ(ρ) profiles | Single-point analysis, triangularity effects (δ > 0.3) |

The Miller model computes V′(ρ) numerically from the Jacobian of the (R, Z) flux-surface coordinates, with κ(0) = 1 and δ(0) = 0 enforced on-axis (Ball & Parra 2015). Volume integrals (P_fus, ⟨nT⟩, W_th, P_rad, bootstrap current) are weighted by V′(ρ)/V when in `D0FUS` mode.

### Safety Factor Profile

Two q(ρ) models are available:

- **Analytical** (`f_q_profile`): parameterised as j(ρ) ∝ (1 − ρ²)^α_J, with α_J derived automatically from the resistive diffusion timescale τ_R.
- **Self-consistent** (`f_q_profile_selfconsistent`): Picard iteration on j_total(ρ) = j_Ohm(ρ) + j_CD(ρ) + j_bs(ρ). Produces a decomposed current profile visualisation.

The q₉₅ boundary value is computed via the `Option_q95` selector:

| Option | Formula | Shaping argument |
|--------|---------|-----------------|
| `Sauter` | Sauter, Fusion Eng. Des. 112 (2016), Eq. 30 | LCFS values (κ_edge, δ_edge) |
| `ITER_1989` | ITER Physics Design Guidelines, Uckan (1990) | ψ_N = 0.95 surface values (κ₉₅, δ₉₅) |

### Bootstrap Current

Four models are supported via `Bootstrap_choice`:

| Model | Reference | Recommended use |
|-------|-----------|-----------------|
| `Freidberg` | Freidberg (2007) | Quick estimates |
| `Segal` | Segal (1993) | Alternative analytical |
| `Sauter` | Sauter et al., PoP 6 (1999) | Full neoclassical, all collisionality regimes |
| `Redl` | Redl et al., PoP 28 (2021) | Default, improved high-ν* accuracy |

### Confinement Scaling Laws

`Scaling_Law` options: `IPB98(y,2)`, `ITPA20`, `ITPA20-IL`, `DS03`, `L-mode`, `L-mode OK`, `ITER89-P`.

### Impurity Radiation

Line radiation from seeded or intrinsic impurities is modelled via the Mavrin (2018) cooling function tabulation (validated against ADAS). Supported species: `W`, `Ar`, `Ne`, `C`, `N`, `Kr`. Core radiation (ρ < `rho_rad_core`) is subtracted from the power balance; edge radiation reduces the divertor load P_sep.

### Runaway Electron Diagnostic

Post-convergence RE indicators are computed by `compute_RE_indicators()` using:
- **Hot-tail seed**: Smith & Verwichte (2008), exponentially sensitive to τ_TQ
- **Avalanche amplification**: Breizman et al. (2019)
- **Coulomb logarithm**: single relativistic lnΛ ≈ 15.3 via `_coulomb_log_relativistic()`
- **Literature benchmarks**: Martín-Solís et al. (2017)

Outputs include I_RE_seed, I_RE_aval, f_RE/Ip, and kinetic energy E_RE_kin. These are **indicators for comparative design ranking**, not quantitative predictions. Configurable via `tau_TQ`, `Te_final_eV`, and `pellet_dilution`.

---

## Engineering Models

### Radial Build

The TF coil inboard leg is shaped as a Princeton-D contour (Gralnick & Tenney 1976). Conductor sizing follows a three-level helium fraction hierarchy distinguishing the cooling pipe (`f_He_pipe`), interstitial void (`f_void`, LTS only), and active SC fraction. Three structural models are available via `Radial_build_model`: a simplified analytical model (`academic`), the D0FUS stress model (`D0FUS`), and the multi-layer CIRCE0D solver (`CIRCE`).

### Superconductor

Critical current density scaling laws are implemented for NbTi, Nb₃Sn (ITER strand parameterisation), and REBCO (Senatore 2024 / Fujikura 2019 datasets). Quench protection is sized via the Maddock hot-spot criterion, with dump time computed from stored magnetic energy and conductor current.

### Techno-Economic Model

Capital investment and COE are computed using the Sheffield & Milora (2016) volume-based cost scaling (2010 USD, converted to 2025 EUR). Component costs cover the SC coil set, blanket, shield, auxiliary heating, heat transfer system, balance of plant, buildings, and annual O&M. A simplified surface-proportional model (Whyte 2024) is also available for cross-checks.

---

## Benchmarks and Validation

The primary benchmark case is ITER Q=10 (Ip = 15 MA, B0 = 5.3 T, R0 = 6.2 m, P_fus = 500 MW), provided as `D0FUS_INPUTS/1_run_ITER.txt`. The main plasma and engineering quantities are recovered to the correct order of magnitude; detailed quantitative agreement is still being refined.

---

## Contributing

Contributions are welcome. Please contact:

- Email: timothe.auclair@cea.fr

---

## License

This project is licensed under the [CeCILL-C License](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html), a French free software license compatible with the GNU LGPL.

See the [LICENSE](LICENSE) file for details.

© 2025 CEA/IRFM

<p align="right">(<a href="#readme-top">back to top</a>)</p>
