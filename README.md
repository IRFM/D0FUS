<a name="readme-top"></a>

# D0FUS

**D0FUS** (Design 0-dimensional for Fusion Systems) is a comprehensive Python-based tool for tokamak fusion power plant design and analysis. It enables rapid exploration of design space through 0D/1D physics models.

## Installation

### Requirements
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Recommended
- [Spyder IDE](https://www.spyder-ide.org/) for interactive use

### Quick Install

```bash
# Clone the repository
git clone http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS.git
```

## Project Structure

```
D0FUS/
├── D0FUS_BIB/                      # Core library modules
│   ├── __init__.py
│   ├── D0FUS_import.py                 # Common imports
│   ├── D0FUS_parameterization.py       # Physical constants and parameters
│   ├── D0FUS_physical_functions.py     # Plasma physics functions
│   └── D0FUS_radial_build_functions.py # Engineering and radial build
│
├── D0FUS_INPUTS/                   # Input parameter files
│   └── default_input.txt               # Default configuration
│
├── D0FUS_OUTPUTS/                  # Generated outputs (auto-created)
│   ├── Run_D0FUS_YYYYMMDD_HHMMSS/      # Single run results
│   ├── Scan_D0FUS_YYYYMMDD_HHMMSS/     # Scan results with figures
│   └── Genetic_D0FUS_YYYYMMDD_HHMMSS/  # Genetic research of an optimal design result
│
├── D0FUS_EXE/                      # Execution modules
│   ├── D0FUS_run.py                    # Single design point calculation
│   ├── D0FUS_scan.py                   # 2D parameter space scan
│   └── D0FUS_genetic.py                # Genetic algorithm for optimal reasearch
│
├── D0FUS.py                        # Main entry point
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## D0FUS startup

### Recommended execution

Launch Spyder and open D0FUS.py

```bash
python D0FUS.py
```

Execute, you'll then be prompted to select an input file

### Bash execution

You can also import and use D0FUS modules in your own scripts:

```python
from D0FUS_EXE import D0FUS_run

# Run a calculation
results = D0FUS_run.main("D0FUS_INPUTS/my_config.txt")

# Access results
B0, B_CS, Q, Ip, betaN, ... = results
```

## Execution Modes

D0FUS automatically detects the execution mode based on the input file format. Three modes are available:

| Mode | Purpose | Input format | Parameters |
|------|---------|--------------|------------|
| **RUN** | Single design point calculation | `R0 = 9` | Fixed values only |
| **SCAN** | 2D parameter space exploration | `R0 = [3, 9, 25]` | Exactly 2 parameters with `[min, max, n_points]` |
| **OPTIMIZATION** | Genetic algorithm cost minimization | `R0 = [3, 9]` | 2+ parameters with `[min, max]` |

**RUN mode** evaluates a single tokamak configuration and outputs all plasma parameters, magnetic fields, power balance, and radial build dimensions.

**SCAN mode** generates 2D maps over two parameters, visualizing feasibility regions bounded by plasma stability limits (Greenwald density, Troyon beta, kink safety factor) and engineering constraints.

**OPTIMIZATION mode** uses a genetic algorithm to find the reactor configuration minimizing cost while satisfying all physics and engineering constraints. The optimizer explores the multi-dimensional parameter space defined by the bounds and evolves the population toward optimal solutions.

## Input

Input files are simple text files with parameter definitions. The format of variable parameters determines which mode D0FUS will execute:

```ini
# Fixed parameter → RUN mode
R0 = 9
```
See `D0FUS_INPUTS/default_input.txt` for a complete run example with all available parameters.

```ini
# Scan parameter → SCAN mode
R0 = [3, 9, 25]                 # [min, max, n_points]
```
See `D0FUS_INPUTS/scan_input_example.txt` for a complete run example with all available parameters.

```ini
# Optimization parameter → OPTIMIZATION mode
R0 = [3, 9]                     # [min, max]
```
See `D0FUS_INPUTS/input_genetic_example.txt` for a complete run example with all available parameters.


## Output

### RUN Mode Output

Results are saved in timestamped directories:

```
D0FUS_OUTPUTS/Run_D0FUS_20251106_123456/
├── input_parameters.txt        # Copy of input configuration
└── output_results.txt          # Complete calculation results
```

**Output includes:**
- Plasma parameters (Ip, ne, Te, βN, Q, τE)
- Magnetic fields (B0, BCS, Bpol)
- Power balance (Pfus, PCD, Psep, Pthresh)
- Radial build dimensions (TF thickness, CS thickness)

### SCAN Mode Output

```
D0FUS_OUTPUTS/Scan_D0FUS_20251106_123456/
├── scan_parameters.txt         # Scan configuration
└── scan_map_[iso]_[bg].png     # High-resolution figure (300 dpi)
```

**Scan visualizations show:**
- Plasma stability boundaries (density, beta, kink safety factor)
- Radial build feasibility limits
- Iso-contours of key parameters (Ip, Q, B0, etc.)

### SCAN Mode Output

```
D0FUS_OUTPUTS/Scan_D0FUS_20251106_123456/
├── scan_parameters.txt         # Scan configuration
└── scan_map_[iso]_[bg].png     # High-resolution figure (300 dpi)
```

**Scan visualizations show:**
- Plasma stability boundaries (density, beta, kink safety factor)
- Radial build feasibility limits
- Iso-contours of key parameters (Ip, Q, B0, etc.)

### OPTIMIZATION Mode Output

```
D0FUS_OUTPUTS/Genetic_D0FUS_20251106_123456/
├── optimization_config.txt     # Optimization parameters and bounds
├── optimization_results.txt    # Best solution and convergence history
└── convergence_plot.png        # Fitness evolution over generations
```

**Optimization results include:**
- Best individual: optimal parameter values minimizing reactor cost
- Fitness evolution: cost reduction across generations
- Constraint satisfaction: plasma stability (Greenwald, Troyon, kink) and radial build feasibility
- Convergence diagnostics: population diversity and stagnation metrics

**Input file format for optimization:**
```ini
# Variable parameters (2+ required)
R0 = [3, 9]                     # Major radius bounds [m]
a = [1, 3]                      # Minor radius bounds [m]
Bmax = [10, 16]                 # Max TF field bounds [T]

# Optional genetic algorithm settings
population_size = 50            # Number of individuals per generation
generations = 100               # Maximum generations
crossover_rate = 0.7            # Crossover probability
mutation_rate = 0.2             # Mutation probability
```

## Contributing

Contributions are welcome! Please contact us:

- Email: timothe.auclair@cea.fr

<p align="right">(<a href="#readme-top">back to top</a>)</p>
