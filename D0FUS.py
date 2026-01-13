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
import sys
import os

# Add the path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_EXE import D0FUS_scan, D0FUS_run, D0FUS_genetic

#%% Mode detection

def detect_mode_from_input(input_file):
    """
    Detect if input file is for RUN, SCAN, or OPTIMIZATION mode
    
    Returns:
        tuple: ('run', 'scan', or 'optimization', additional parameters)
        
    Raises:
        ValueError: if invalid configuration
        FileNotFoundError: if input file doesn't exist
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract genetic algorithm parameters if present
    genetic_params = {}
    genetic_keywords = {
        'population_size': ('population_size', int),
        'generations': ('generations', int),
        'crossover_rate': ('crossover_rate', float),
        'mutation_rate': ('mutation_rate', float)
    }
    
    for keyword, (param_name, param_type) in genetic_keywords.items():
        pattern = rf'^\s*{keyword}\s*[:=]\s*([0-9.]+)'
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                genetic_params[param_name] = param_type(match.group(1))
            except ValueError:
                raise ValueError(f"Invalid value for {keyword}: {match.group(1)}")
    
    # Find all bracket patterns: parameter = [values]
    bracket_pattern = r'^\s*(\w+)\s*[:=]\s*\[([^\]]+)\]'
    matches = re.findall(bracket_pattern, content, re.MULTILINE)
    
    if not matches:
        # No brackets → RUN mode
        return 'run', []
    
    # Classify each bracket parameter
    scan_params = []      # [min, max, n_points]
    opt_params = {}       # [min, max]
    
    for param_name, values_str in matches:
        values = re.split(r'[,;]', values_str)
        values = [v.strip() for v in values if v.strip()]
        
        if len(values) == 2:
            # 2 values → OPTIMIZATION parameter
            try:
                min_val = float(values[0])
                max_val = float(values[1])
                opt_params[param_name] = (min_val, max_val)
            except ValueError as e:
                raise ValueError(
                    f"\n Invalid values for parameter {param_name}: {values_str}\n"
                    f"Values must be numeric: [min, max]\n"
                    f"Error: {str(e)}"
                )
        elif len(values) == 3:
            # 3 values → SCAN parameter
            try:
                min_val = float(values[0])
                max_val = float(values[1])
                n_points = int(float(values[2]))
                scan_params.append((param_name, min_val, max_val, n_points))
            except ValueError as e:
                raise ValueError(
                    f"\n Invalid values for scan parameter {param_name}: {values_str}\n"
                    f"Values must be numeric: [min, max, n_points]\n"
                    f"Error: {str(e)}"
                )
        else:
            raise ValueError(
                f"\n Invalid bracket format for {param_name}: {values_str}\n"
                f"Expected:\n"
                f"  - [min, max] for optimization\n"
                f"  - [min, max, n_points] for scan\n"
            )
    
    # Determine mode based on what we found
    n_scan = len(scan_params)
    n_opt = len(opt_params)
    
    if n_opt > 0 and n_scan == 0:
        # Only optimization parameters → OPTIMIZATION mode
        if n_opt < 2:
            param_name = list(opt_params.keys())[0]
            raise ValueError(
                f"\n Invalid optimization: Found only 1 parameter ({param_name}).\n"
                f"\n"
                f"OPTIMIZATION mode requires at least 2 parameters with [min, max].\n"
                f"\n"
                f"Example:\n"
                f"  R0 = [3, 9]\n"
                f"  a = [1, 3]\n"
                f"  Bmax = [10, 16]\n"
            )
        return 'optimization', (opt_params, genetic_params)
    
    elif n_scan > 0 and n_opt == 0:
        # Only scan parameters → SCAN mode
        if n_scan == 1:
            param_name = scan_params[0][0]
            raise ValueError(
                f"\n Invalid scan: Found only 1 parameter ({param_name}).\n"
                f"\n"
                f"SCAN mode requires exactly 2 parameters with [min, max, n_points].\n"
                f"\n"
                f"Example:\n"
                f"  R0 = [3, 9, 25]\n"
                f"  a = [1, 3, 25]\n"
            )
        elif n_scan == 2:
            return 'scan', scan_params
        else:
            param_names = [p[0] for p in scan_params]
            raise ValueError(
                f"\n Invalid scan: Found {n_scan} parameters: {', '.join(param_names)}.\n"
                f"\n"
                f"SCAN mode requires exactly 2 parameters.\n"
            )
    
    elif n_opt > 0 and n_scan > 0:
        # Mixed parameters → ERROR
        raise ValueError(
            f"\n Invalid input file: Mixed parameter formats detected.\n"
            f"  - {n_opt} optimization parameter(s) with [min, max]\n"
            f"  - {n_scan} scan parameter(s) with [min, max, n_points]\n"
            f"\n"
            f"Please choose ONE mode:\n"
            f"  - OPTIMIZATION: Use [min, max] for all variable parameters\n"
            f"  - SCAN: Use [min, max, n_points] for exactly 2 parameters\n"
            f"  - RUN: Remove all brackets for fixed values\n"
        )
    
    # Should not reach here
    return 'run', []

#%% Main functions

def print_banner():
    """Display D0FUS banner"""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║                       D0FUS                       ║
    ║     Design 0-dimensional for Fusion Systems       ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)

def print_usage():
    """Print usage information"""
    usage = """
Usage:
    python D0FUS.py [input_file]
    
If no input file is provided, interactive mode will start.

Modes (detected automatically from input file format):
    
    RUN mode:          Single point calculation
                       No brackets in input file
                       Example: R0 = 9
    
    SCAN mode:         2D parameter space exploration
                       Exactly 2 parameters with [min, max, n_points]
                       Example: R0 = [3, 9, 25]
                                a = [1, 3, 25]
    
    OPTIMIZATION mode: Genetic algorithm optimization
                       2+ parameters with [min, max] (no n_points)
                       Example: R0 = [3, 9]
                                a = [1, 3]
                                Bmax = [10, 16]
                       
                       Optional genetic algorithm parameters:
                       population_size = 50     (default: 50)
                       generations = 100        (default: 100)
                       crossover_rate = 0.7     (default: 0.7)
                       mutation_rate = 0.2      (default: 0.2)

Detection rules:
    • [min, max] format (2 values) → OPTIMIZATION (need 2+ parameters)
    • [min, max, n] format (3 values) → SCAN (need exactly 2 parameters)
    • No brackets → RUN
    • Cannot mix formats in same file

For help:
    python D0FUS.py --help
    """
    print(usage)

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
                opt_params, genetic_params = params
                param_names = list(opt_params.keys())
                mode_str = f"GENETIC ({len(param_names)} params)"
            else:
                mode_str = "RUN"
            print(f"  {i}. {file.name:<30} [{mode_str}]")
        except:
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

def execute_with_mode_detection(input_file):
    """
    Execute D0FUS with automatic mode detection
    
    Args:
        input_file: Path to input file (or None for defaults)
    """
    if input_file is None:
        # No input file → use defaults for RUN mode
        print("\n" + "="*60)
        print("Mode: RUN (default parameters)")
        print("="*60 + "\n")
        D0FUS_run.main(None)
        return
    
    # Detect mode from input file
    try:
        mode, params = detect_mode_from_input(input_file)
        
        if mode == 'run':
            # RUN mode detected
            print("\n" + "="*60)
            print("Mode: RUN (single point calculation)")
            print(f"Input: {os.path.basename(input_file)}")
            print("="*60 + "\n")
            D0FUS_run.main(input_file)
        
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
            
            D0FUS_scan.main(input_file)
        
        elif mode == 'optimization':
            # OPTIMIZATION mode detected
            opt_params, genetic_params = params
            param_names = list(opt_params.keys())
            print("\n" + "="*60)
            print(f"Mode: OPTIMIZATION (genetic algorithm)")
            print(f"Optimization parameters: {', '.join(param_names)}")
            print(f"Input: {os.path.basename(input_file)}")
            
            # Display optimization ranges
            for param_name, (min_val, max_val) in opt_params.items():
                print(f"  {param_name}: [{min_val}, {max_val}]")
            
            # Display genetic algorithm parameters
            print("\nGenetic algorithm parameters:")
            default_params = {
                'population_size': 50,
                'generations': 100,
                'crossover_rate': 0.7,
                'mutation_rate': 0.2
            }
            for param_name, default_value in default_params.items():
                actual_value = genetic_params.get(param_name, default_value)
                status = "" if param_name in genetic_params else " (default)"
                print(f"  {param_name}: {actual_value}{status}")
            
            print("="*60 + "\n")
            
            # Prepare parameters for genetic optimization
            ga_params = {
                'population_size': genetic_params.get('population_size', 50),
                'generations': genetic_params.get('generations', 100),
                'crossover_rate': genetic_params.get('crossover_rate', 0.7),
                'mutation_rate': genetic_params.get('mutation_rate', 0.2),
                'verbose': True
            }
            
            # Run genetic optimization with specified or default parameters
            D0FUS_genetic.run_genetic_optimization(input_file, **ga_params)
    
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

def interactive_mode():
    """Interactive mode: select file, auto-detect mode, execute"""
    print_banner()
    
    # Select input file
    input_file = select_input_file()
    
    # Execute with automatic mode detection
    execute_with_mode_detection(input_file)

def command_line_mode(input_file):
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
    execute_with_mode_detection(input_file)

def main():
    """Main entry point"""
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_banner()
        print_usage()
        sys.exit(0)
    
    # Run in appropriate mode
    try:
        if len(sys.argv) == 1:
            # No arguments → interactive mode
            interactive_mode()
        elif len(sys.argv) == 2:
            # One argument → command line mode with input file
            input_file = sys.argv[1]
            command_line_mode(input_file)
        else:
            print("Error: Too many arguments.")
            print_usage()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n Operation cancelled by user")
        sys.exit(0)

if __name__ == "__main__":
    main()