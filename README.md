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

## Quick Start

Launch Spyder and open D0FUS.py

```bash
python D0FUS.py
```

Execute, uou'll then be prompted to select an input file

You can also import and use D0FUS modules in your own scripts:

```python
from D0FUS_EXE import D0FUS_run

# Run a calculation
results = D0FUS_run.main("D0FUS_INPUTS/my_config.txt")

# Access results
B0, B_CS, Q, Ip, betaN, ... = results
```

## Input File Format

Input files are simple text files with parameter definitions:

```ini
# D0FUS Input Parameters
# Lines starting with # are comments

# Geometry
P_fus = 2000                    # Fusion power [MW]
R0 = 9                          # Major radius [m]
a = 3                           # Minor radius [m]
b = 1.2                         # Blanket + shield thickness [m]

# Magnetic Field
Bmax = 12                       # Max toroidal field on TF coils [T]

# Technology
Supra_choice = Nb3Sn            # Superconductor type
Choice_Buck_Wedg = Wedging      # Mechanical configuration
Chosen_Steel = 316L             # Structural material

# Plasma Physics
Scaling_Law = IPB98(y,2)        # Confinement scaling law
H = 1                           # H-factor
Tbar = 14                       # Average temperature [keV]

# Operation
Operation_mode = Steady-State   # Steady-State or Pulsed
```

See `D0FUS_INPUTS/default_input.txt` for a complete example with all available parameters.
If one want to scan some parameters, just put it under bracket with the precision desired:

```ini
# Geometry
R0 = [3,9,25]                          # Major radius [m]
a = [1,3,25]                           # Minor radius [m]
```

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
- Engineering limits and safety factors

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

## Contributing

Contributions are welcome! Please contact us:

**Author**: Timothe Auclair

- Email: timothe.auclair@cea.fr

<p align="right">(<a href="#readme-top">back to top</a>)</p>
