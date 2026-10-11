"""

D0FUS - Design 0-dimensional for FUsion Systems
Author: Auclair Timothe

This software is governed by the CeCILL-C license under French law.
See LICENSE file or https://cecill.info/licences/Licence_CeCILL-C_V1-en.html
Copyright holders : Commissariat à l’Energie Atomique et aux Energies Alternatives (CEA), France
The terms and conditions of the CeCILL-C license are deemed to be accepted 
upon downloading the software and/or exercising any of the rights granted under the CeCILL-C license.

"""
#%% Imports
import argparse
import os
import sys
from pathlib import Path

# Add the path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Silence tqdm's cosmetic notebook-widget probe ("IProgress not found")
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*IProgress not found.*")

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_EXE import D0FUS_scan, D0FUS_run, D0FUS_genetic, D0FUS_uncertainty, D0FUS_popcon

#%% Mode detection

def detect_mode_from_input(input_file):
    """
    Detect the execution mode of an input file from its content.

    Comments (text after #) are ignored and section titles such as [POPCON]
    are optional. The rule is D0FUS_run.detect_mode_from_lines, shared with
    the parsers of the modes:
      - a tri(), norm(), unif() or envelope() entry           -> 'uncertainty'
      - axes nbar_line and Tbar written [min, max, n_points]  -> 'popcon'
      - 2 other axes written [min, max, n_points]             -> 'scan'
      - 2 or more parameters written [min, max]               -> 'optimization'
      - none of these                                         -> 'run'

    Returns:
        tuple: (mode, details), see D0FUS_run.detect_mode_from_lines

    Raises:
        ValueError: if invalid configuration
        FileNotFoundError: if input file doesn't exist
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        return D0FUS_run.detect_mode_from_lines(f.read().splitlines())

#%% Main functions

def print_banner():
    """Display the D0FUS startup banner.

    Simple double-line framed banner with title, tagline, version, author,
    license and repository URL. Can be silenced with the environment variable
    ``D0FUS_NO_BANNER`` (useful for batch scans, HPC jobs, CI).
    """
    if os.environ.get("D0FUS_NO_BANNER", "").strip() not in ("", "0", "false", "False"):
        return

    # Resolve the version. A source checkout reads pyproject.toml next to this
    # file (single source of truth, same as the detailed run report); the
    # installed-package metadata is only a fallback, because it is frozen at
    # the last "pip install" (editable installs included) and goes stale.
    v = None
    pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml")
    try:
        for line in open(pp, encoding="utf-8"):
            t = line.strip()
            if t.startswith("version") and "=" in t:
                v = t.split("=", 1)[1].strip().strip('"').strip("'")
                break
    except OSError:
        pass
    if v is None:
        try:
            from importlib.metadata import version, PackageNotFoundError
            try:
                v = version("d0fus")
            except PackageNotFoundError:
                v = "dev"
        except ImportError:
            v = "dev"

    # The version is the only dynamic line; the others are hardcoded for clarity.
    # Inside width is 51 characters; the version is centered programmatically.
    version_line = f"Version {v}".center(51)

    banner = f"""
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║                       D0FUS                       ║
    ║      Design 0-dimensional for Fusion Systems      ║
    ║                                                   ║
    ╠═══════════════════════════════════════════════════╣
    ║{version_line}║
    ║         T. Auclair, CEA-IRFM  |  CeCILL-C         ║
    ║           https://github.com/IRFM/D0FUS           ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)

def _modes_text():
    """Description of the modes shown by --help. GA defaults from GA_SETTINGS."""
    ga = D0FUS_genetic.GA_SETTINGS
    return f"""
Modes, detected from the content of the input file (text after # is a
comment, section titles such as [POPCON] are optional):

  RUN          One design point. No bracketed value, no distribution.
                 R0 = 6.2
  SCAN         2D map. Exactly 2 parameters written [min, max, n_points].
                 R0 = [3, 9, 25]
                 a  = [1, 3, 25]
  GENETIC      Genetic-algorithm optimisation. 2 or more parameters
               written [min, max].
                 R0      = [3, 9]
                 a       = [1, 3]
                 Bmax_TF = [10, 16]
               GA settings are read from the file. Defaults:
                 population_size = {ga['population_size'][0]}, generations = {ga['generations'][0]},
                 crossover_rate = {ga['crossover_rate'][0]}, mutation_rate = {ga['mutation_rate'][0]}, patience = {ga['patience'][0]}
  UNCERTAINTY  Monte-Carlo robustness study of one design point. At
               least one tri(), norm(), unif() or envelope() entry.
                 H           = tri(0.75, 1.5)
                 betaN_limit = tri(2.8, 3.6)
                 Scaling_Law = envelope(IPB98(y,2) | ITPA20)
               Study settings (n_samples, seed, retune, ...) are read
               by name.
  POPCON       Operating-space map of one design point. The two axes
               are the density and the temperature.
                 nbar_line = [0.35, 1.45, 45]
                 Tbar      = [3.5, 24.0, 36]

A file cannot mix [min, max], [min, max, n_points] and uncertain entries.
Each run writes a timestamped folder under D0FUS_OUTPUTS, or under the
folder given by --output-dir.
"""

def build_parser():
    """Command line parser of D0FUS.py."""
    parser = argparse.ArgumentParser(
        prog="D0FUS.py",
        description="D0FUS: 0D design of tokamak fusion power plants.",
        epilog=_modes_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    parser.add_argument("input_file", nargs="?", default=None,
                        help="Input file. Without it, a menu lists the files "
                             "of D0FUS_INPUTS.")
    parser.add_argument("--output-dir", metavar="DIR", default=None,
                        help="Parent folder of the result folders. Default: "
                             "D0FUS_OUTPUTS of the source tree (of the current "
                             "folder for an installed package).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Write the reports and data files without any figure.")
    parser.add_argument("--no-3d", action="store_true",
                        help="RUN: skip the 3D machine view (PyVista), the "
                             "slowest figure.")
    parser.add_argument("--seed", type=int, metavar="N", default=None,
                        help="Random seed of GENETIC and UNCERTAINTY. "
                             "Overrides the input file.")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Show this help and exit.")
    return parser

def print_usage():
    """Print usage information"""
    build_parser().print_help()

def list_input_files():
    """List available input files in D0FUS_INPUTS directory"""
    input_dir = Path(__file__).parent / 'D0FUS_INPUTS'
    if not input_dir.exists():
        print(f"Warning: Input directory '{input_dir}' not found.")
        return []
    
    input_files = list(input_dir.glob('*.txt'))
    return sorted(input_files)

def select_input_file():
    """Interactive input file selection"""
    input_files = list_input_files()
    
    if not input_files:
        print("\n No input files found in D0FUS_INPUTS directory.")
        print("Using default parameters for RUN mode.")
        return None
    
    print("\n" + "="*60)
    print("Available input files:")
    print("="*60)
    for i, file in enumerate(input_files, 1):
        # Try to detect mode for each file
        try:
            mode, params = detect_mode_from_input(str(file))
            if mode == 'scan':
                param_names = [p[0] for p in params]
                mode_str = f"SCAN ({param_names[0]} × {param_names[1]})"
            elif mode == 'optimization':
                opt_params = params
                param_names = list(opt_params.keys())
                mode_str = f"GENETIC ({len(param_names)} params)"
            elif mode == 'uncertainty':
                mode_str = (f"UNCERTAINTY ({params} params)" if params
                            else "UNCERTAINTY (Monte-Carlo)")
            elif mode == 'popcon':
                mode_str = "POPCON"
            else:
                mode_str = "RUN"
            print(f"  {i}. {file.name:<30} [{mode_str}]")
        except (OSError, ValueError):
            print(f"  {i}. {file.name:<30} [Unknown]")
    print(f"  0. Use default parameters (RUN mode)")
    print("="*60)
    
    while True:
        try:
            choice = input("\nSelect input file (number): ").strip()
            choice = int(choice)
            
            if choice == 0:
                return None
            elif 1 <= choice <= len(input_files):
                return str(input_files[choice - 1])
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(input_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)

def execute_with_mode_detection(input_file, output_dir=None, save_figures=True,
                                seed=None, view_3d=True):
    """
    Execute D0FUS with automatic mode detection

    Args:
        input_file   : Path to input file (or None for defaults)
        output_dir   : Parent folder of the result folders (None: default
                       of each mode, D0FUS_run.default_output_root())
        save_figures : False writes no figure (reports and data only)
        seed         : Random seed of GENETIC and UNCERTAINTY (None: from
                       the input file)
        view_3d      : False skips the 3D machine view of a RUN
    """
    if input_file is None:
        # No input file → use defaults for RUN mode
        print("\n" + "="*60)
        print("Mode: RUN (default parameters)")
        print("="*60 + "\n")
        D0FUS_run.main(None, save_figures=save_figures, output_dir=output_dir,
                       view_3d=view_3d)
        return
    
    # Detect mode from input file
    try:
        mode, params = detect_mode_from_input(input_file)
        if seed is not None and mode in ('run', 'scan', 'popcon'):
            print(f"\n [note] --seed has no effect in {mode.upper()} mode.")
        
        if mode == 'run':
            # RUN mode detected
            print("\n" + "="*60)
            print("Mode: RUN (single point calculation)")
            print(f"Input: {os.path.basename(input_file)}")
            print("="*60 + "\n")
            D0FUS_run.main(input_file, save_figures=save_figures,
                           output_dir=output_dir, view_3d=view_3d)
        
        elif mode == 'popcon':
            print("\n" + "="*60)
            print("Mode: POPCON (operating-space contour map)")
            print(f"Input: {os.path.basename(input_file)}")
            print("="*60 + "\n")
            D0FUS_popcon.main(input_file, save_figures=save_figures,
                              output_dir=output_dir)
        elif mode == 'scan':
            # SCAN mode detected
            param_names = [p[0] for p in params]
            print("\n" + "="*60)
            print(f"Mode: SCAN (2D parameter space)")
            print(f"Scan parameters: {param_names[0]} × {param_names[1]}")
            print(f"Input: {os.path.basename(input_file)}")
            
            # Display scan ranges
            for param_name, min_val, max_val, n_points in params:
                print(f"  {param_name}: [{min_val}, {max_val}] with {n_points} points")
            print("="*60 + "\n")
            
            D0FUS_scan.main(input_file, output_dir=output_dir,
                            save_figures=save_figures)
        
        elif mode == 'optimization':
            # OPTIMIZATION mode detected
            opt_params = params
            param_names = list(opt_params.keys())
            print("\n" + "="*60)
            print(f"Mode: OPTIMIZATION (genetic algorithm)")
            print(f"Optimization parameters: {', '.join(param_names)}")
            print(f"Input: {os.path.basename(input_file)}")
            
            # Display optimization ranges
            for param_name, (min_val, max_val) in opt_params.items():
                print(f"  {param_name}: [{min_val}, {max_val}]")
            print("="*60 + "\n")
            
            # GA settings come from the deck, or else from the GA_SETTINGS
            # defaults. run_genetic_optimization prints them with their source.
            D0FUS_genetic.run_genetic_optimization(
                input_file, verbose=True, seed=seed, output_dir=output_dir,
                save_figures=save_figures)
        
        elif mode == 'uncertainty':
            # UNCERTAINTY mode detected (Monte-Carlo robustness study)
            print("\n" + "="*60)
            print("Mode: UNCERTAINTY (Monte-Carlo robustness study)")
            print(f"Input: {os.path.basename(input_file)}")
            print("="*60 + "\n")
            
            D0FUS_uncertainty.main(input_file, save_figures=save_figures,
                                   output_dir=output_dir, seed=seed)
    
    except ValueError as e:
        # Invalid number of brackets or parsing error
        print(str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def interactive_mode(**options):
    """Interactive mode: select file, auto-detect mode, execute"""
    print_banner()
    
    # Select input file
    input_file = select_input_file()
    
    # Execute with automatic mode detection
    execute_with_mode_detection(input_file, **options)

def command_line_mode(input_file, **options):
    """Command line mode: auto-detect mode from input file"""
    print_banner()
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"\n Error: Input file not found: {input_file}")
        print("\nAvailable files in D0FUS_INPUTS:")
        for f in list_input_files():
            print(f"  - {f.name}")
        sys.exit(1)
    
    # Execute with automatic mode detection
    execute_with_mode_detection(input_file, **options)

def main(argv=None):
    """Main entry point. argv: command line arguments (default: sys.argv[1:])."""
    # A console that cannot encode a character (cp1252 terminal, pipe from
    # Git Bash) shows a replacement mark instead of stopping the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    # Help flag ('help' as a file name kept for the former command line)
    if args.help or args.input_file == 'help':
        print_banner()
        parser.print_help()
        sys.exit(0)

    options = dict(output_dir=args.output_dir,
                   save_figures=not args.no_figures,
                   seed=args.seed,
                   view_3d=not args.no_3d)

    # Run in appropriate mode
    try:
        if args.input_file is None:
            # No input file → interactive mode
            interactive_mode(**options)
        else:
            # Input file → command line mode
            command_line_mode(args.input_file, **options)

    except KeyboardInterrupt:
        print("\n\n Operation cancelled by user")
        sys.exit(0)

if __name__ == "__main__":
    main()