<a name="readme-top"></a>

# D0FUS

**D0FUS** (Design 0-Dimensional for Fusion Systems) is a comprehensive Python-based tool for tokamak fusion reactor design and analysis. It enables rapid exploration of design space through 0D physics models.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/D0FUS.git
cd D0FUS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
matplotlib >= 3.4.0
sympy >= 1.9
tqdm >= 4.62.0
```

## 🗂️ Project Structure

```
D0FUS/
├── D0FUS_BIB/                      # Core library modules
│   ├── __init__.py
│   ├── D0FUS_import.py             # Common imports
│   ├── D0FUS_parameterization.py   # Physical constants and parameters
│   ├── D0FUS_physical_functions.py # Plasma physics functions
│   └── D0FUS_radial_build_functions.py # Engineering and radial build
│
├── D0FUS_INPUTS/                   # Input parameter files
│   └── default_input.txt           # Default configuration
│
├── D0FUS_OUTPUTS/                  # Generated outputs (auto-created)
│   ├── Run_D0FUS_YYYYMMDD_HHMMSS/  # Single run results
│   └── Scan_D0FUS_YYYYMMDD_HHMMSS/ # Scan results with figures
│
├── D0FUS_EXE/                      # Execution modules
│   ├── D0FUS_run.py                # Single design point calculation
│   └── D0FUS_scan.py               # 2D parameter space scan
│
├── D0FUS.py                        # Main entry point
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Interactive Mode

Launch D0FUS in interactive mode to select input files and operation mode:

```bash
python D0FUS.py
```

You'll be prompted to:
1. Select an input file (or use defaults)
2. Choose operation mode (RUN or SCAN)

### Command Line Mode

#### Single Design Point

Calculate a single reactor design:

```bash
python D0FUS.py run [input_file]

# Example
python D0FUS.py run D0FUS_INPUTS/default_input.txt
```

#### Parameter Space Scan

Generate a 2D scan over major radius (R₀) and minor radius (a):

```bash
python D0FUS.py scan [input_file]

# Example
python D0FUS.py scan D0FUS_INPUTS/default_input.txt
```

### Python API

Import and use D0FUS modules in your own scripts:

```python
from D0FUS_EXE import D0FUS_run

# Run a calculation
results = D0FUS_run.main("D0FUS_INPUTS/my_config.txt")

# Access results
B0, B_CS, Q, Ip, betaN, ... = results
```

## 📝 Input File Format

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

## 📊 Output

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

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ✉️ Contact

**Author**: Timothe Auclair

- Email: timothe.auclair@cea.fr

## Getting Started

The code is currently presented as a Python library comprising a tree of .py scripts for multiple objectives:
* D0FUS_import: Import all the necessary libraries
* D0FUS_parametrization: Panel control of the code, with the definition of all the constants
* D0FUS_physical_functions: Definition of all the physical functions
* D0FUS_radial_build_functions: Definition of the functions allowing for the different width calculations
* D0FUS_run: Major function definition, allowing for the calculation of a complete design point
* D0FUS_plot: Aesthetic functions

Additionally, the 'main' files, utilizing the previously presented library, are representative of the possible and current uses of the code:
* D0FUS_Benchmark: Provides a benchmark with several machines
* D0FUS_Gradient: Enables the search for an optimized point using various gradient descent techniques (mainly genetic algorithms)
* D0FUS_Scan_1D: 1D scan (typically on major or minor radius)
* D0FUS_Scan_2D: 2D scan (typically on major and minor radius). Major use of the code and the most refined script. Allows for complex visualization, choosing between several parameter plots.
* D0FUS_Scan_3D: 3D scan (typically on major and minor radius + magnetic field)

Please note that currently, not all the 'main' files are fully operational.

Additionally, and for more details, useful papers and notes on the code are grouped in the 'Reference Papers' file:
* [Plasma Physics Paper](http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS/-/blob/main/Reference%20Papers/D0FUS_Auclair_Timothe_FED_Submission_29.11.24.pdf)
* [Radial Build Paper](http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS/-/blob/main/Reference%20Papers/D0FUS_Auclair_Timothe_Radial_Build_09.01.2024.pdf?ref_type=heads)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites

D0FUS is a code fully developed in Python.
To use it, the author recommends using Spyder, but any other Python executor will suffice.
The code uses the following librarys:

* os
* math
* time
* warnings
* sys
* importlib
* numpy
* pandas
* sympy
* scipy
* matplotlib
* mpl_toolkits
* tqdm

To do this, if necessary, execute in the console:

* CMD
  ```sh
  conda install library_name
  ```

If you want to install every librarys in one single command :

* CMD
  ```sh
  conda install os math time warnings sys importlib numpy pandas sympy scipy matplotlib tqdm

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Attntion : conda install might not work, then use pip install.

In practice, most of the library are pre-installed in most Python executor, at the exception of tqdm.

## Installation

**Classical :**

To create a local copy of this project to use it, here are the necessary steps:

1. Download GIT on you machine
2. Open GIT BASH where you want to clone the project
3. Identification :
  ```sh
  git config --global user.name USERNAME
  ```
  ```sh
  git config --global user.email EMAIL
  ```
4. Clone the project :
  ```sh
  git clone http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS.git
  ```
**Alternative :**

Download all the folders from the git repository

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Developers

**Classical :**

To update the main branch, here are the steps to follow:
1. Add the necessary files to the GIT cache :
  ```sh
  git add FICHIER.TYPE
  ```
2. Commit the necessary files to the local GIT repository :
  ```sh
  git commit -m 'Actualisation message'
  ```
3. Push the local repository to the server:
  ```sh
  git push HEAD:main
  ```
**Alternative :**

Modify the code in the GIT interface and save as a new commit with associated message.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Outside Contributers

**Classical :**

If you have a suggestion that would make this better, please fork the repository and create a pull request. To do so :

1. Fork the Project
2. Create your Feature Branch 
```sh
git checkout -b feature/AmazingFeature
```
3. Commit your Changes
```sh
git commit -m 'Add some AmazingFeature'
```
4. Push to the Branch
```sh
git push origin feature/AmazingFeature
```
5. Open a Pull Request

You can also open an issue with the tag "enhancement".

**Alternative :**

From the GIT user interface create a new Branch.
Modify the code in the GIT interface and save as a new commit with associated message.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Server

For users belonging to IRFM, execution on the IRFM JupyterLab server is also possible and convenient. To do so, go to the IRFM website, then navigate to the 'Pratique' tab, and then 'Python IRFM'.

## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Auclair Timothe - timothe.auclair@cea.fr

Project Link: [http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS/.git]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
