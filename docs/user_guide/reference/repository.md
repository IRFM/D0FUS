(app:d0fus_repo)=
(sssec:chap1_archi)=

# Repository layout, installation and library usage

:::{admonition} Source
:class: thesis-source

Adapted from T. Auclair, PhD thesis, Aix-Marseille Université (2026), Appendix D.1. The thesis describes D0FUS v2.7.0. Where the code has changed since, the {doc}`API reference </api/index>`, generated from the current sources, is authoritative.
:::


This section collects the practical material supporting Section {ref}`General architecture of D0FUS <sec:chap1_architecture>`: the design principles, the repository layout, the installation paths, the dependencies and a library-usage example.

D0FUS is organised around two guiding principles. The first is *one file, one domain*: each source file covers a single physical or engineering topic (plasma physics, magnet sizing, cost modelling, etc.), so that a reader looking for, say, the bootstrap current model only needs to open one file. The second is *every function is callable in isolation*: all physics and engineering models can be imported and evaluated independently, without running a full tokamak calculation. This makes the code equally useful as a library for standalone analyses and as an integrated systems code.

The code depends on standard scientific Python libraries: NumPy, SciPy (optimisation, integration, interpolation), Matplotlib, Pandas, and the DEAP evolutionary algorithm framework. All dependencies are centralised in a single import file (`D0FUS_import.py`), so that version requirements and platform-specific settings are handled in one place. Python 3.10 or later is required.

The repository is structured as follows:

        D0FUS/
        |-- D0FUS_BIB/          Core library (physics, magnets, cost, figures)
        |-- D0FUS_EXE/          Execution modules (run, scan, genetic)
        |-- D0FUS_INPUTS/       Input parameter files (ITER, EU-DEMO, ...)
        |-- D0FUS_OUTPUTS/      Outputs (one timestamped folder per run)
        |-- D0FUS.py            Main entry point (automatic mode detection)
        +-- requirements.txt

The `D0FUS_BIB/` directory contains the physics library, structured as a standalone Python package. The `D0FUS_EXE/` directory provides three execution modes described in Section {ref}`Execution modes <sssec:chap1_modes>`. The `D0FUS_INPUTS/` directory ships example input files for the main benchmark machines, and `D0FUS_OUTPUTS/` is populated automatically with timestamped result folders at each execution.

Three installation paths are documented: a self-contained Spyder bundle (requiring no prior Python knowledge), a Miniforge conda environment (for users who manage multiple projects), and a headless `pip install` (for scripting and library usage). The version documented and used throughout the thesis is release 2.7.0, archived as the annotated tag `v2.7.0` of the repository and published on PyPI (`pip install d0fus==2.7.0`).

An important consequence of the modular design is that any function can be called in isolation, outside the solver loop. A user who wants to compare two bootstrap current models, or evaluate the critical current density of REBCO at a specific field and temperature, can do so in a few lines of Python:

        from D0FUS_BIB.D0FUS_radial_build_functions import J_non_Cu_REBCO
        Jc = J_non_Cu_REBCO(B=20, T=4.2)  # A/m^2, at 20 T and 4.2 K

This capability is particularly valuable for three use cases. For teaching, students can easily extract and explore the sensitivity of any model or parameter. For *benchmarking*, comparing a D0FUS function against a published result or against another code requires only calling that function with the same inputs. Finally, for code extension: adding a new scaling law or a new superconductor amounts to writing one function with the standard signature and registering it in the appropriate dictionary.
