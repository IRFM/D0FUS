"""
D0FUS - Design 0-dimensional for Fusion Systems
Author: Auclair Timothe
"""
#%% Imports
import sys
import os
import re
from pathlib import Path
try: get_ipython().magic('autoreload 2') # Activating auto-reload
except: pass

# Add D0FUS_EXE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'D0FUS_EXE'))

from D0FUS_EXE import D0FUS_run, D0FUS_scan

#%% Mode detection

def detect_mode_from_input(input_file):
    """
    Detect if input file is for RUN or SCAN mode
    
    Returns:
        tuple: ('run' or 'scan', list of scan parameter tuples)
        
    Raises:
        ValueError: if 1 or >2 brackets found
        FileNotFoundError: if input file doesn't exist
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all bracket patterns: parameter = [min, max, n_points]
    bracket_pattern = r'^\s*(\w+)\s*=\s*\[([^\]]+)\]'
    matches = re.findall(bracket_pattern, content, re.MULTILINE)
    
    n_brackets = len(matches)
    
    # Parse the bracket values for SCAN mode
    scan_params = []
    for param_name, values_str in matches:
        values = re.split(r'[,;]', values_str)
        if len(values) != 3:
            raise ValueError(
                f"\n Invalid scan parameter {param_name}: Expected 3 values [min, max, n_points], "
                f"got {len(values)} values.\n"
                f"Example: {param_name} = [min_value, max_value, number_of_points]"
            )
        try:
            min_val = float(values[0].strip())
            max_val = float(values[1].strip())
            n_points = int(float(values[2].strip()))
            scan_params.append((param_name, min_val, max_val, n_points))
        except ValueError as e:
            raise ValueError(
                f"\n Invalid values for scan parameter {param_name}: {values_str}\n"
                f"Values must be numeric: [min, max, n_points]\n"
                f"Error: {str(e)}"
            )
    
    if n_brackets == 0:
        # No brackets → RUN mode
        return 'run', []
    elif n_brackets == 2:
        # Exactly 2 brackets → SCAN mode
        return 'scan', scan_params
    elif n_brackets == 1:
        # Only 1 bracket → ERROR
        param_name = scan_params[0][0]
        raise ValueError(
            f"\n Invalid input file: Found only 1 scan parameter ({param_name}).\n"
            f"\n"
            f"SCAN mode requires exactly 2 parameters with brackets [min, max, n_points].\n"
            f"For RUN mode, remove all brackets from the input file.\n"
            f"\n"
            f"Example for SCAN:\n"
            f"  R0 = [3, 9, 25]\n"
            f"  a = [1, 3, 25]\n"
            f"\n"
            f"Example for RUN:\n"
            f"  R0 = 9\n"
            f"  a = 3\n"
        )
    else:
        # More than 2 brackets → ERROR
        param_names = [p[0] for p in scan_params]
        raise ValueError(
            f"\n Invalid input file: Found {n_brackets} scan parameters: {', '.join(param_names)}.\n"
            f"\n"
            f"SCAN mode requires exactly 2 parameters with brackets [min, max, n_points].\n"
            f"Please select only 2 parameters to scan.\n"
            f"\n"
            f"Example:\n"
            f"  R0 = [3, 9, 25]     ← scan parameter 1\n"
            f"  a = [1, 3, 25]      ← scan parameter 2\n"
            f"  Bmax = 12           ← fixed parameter\n"
            f"  P_fus = 2000        ← fixed parameter\n"
        )

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
            mode, scan_params = detect_mode_from_input(str(file))
            if mode == 'scan':
                param_names = [p[0] for p in scan_params]
                mode_str = f"SCAN ({param_names[0]} × {param_names[1]})"
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
        mode, scan_params = detect_mode_from_input(input_file)
        
        if mode == 'run':
            # RUN mode detected
            print("\n" + "="*60)
            print("Mode: RUN (single point calculation)")
            print(f"Input: {os.path.basename(input_file)}")
            print("="*60 + "\n")
            D0FUS_run.main(input_file)
        
        elif mode == 'scan':
            # SCAN mode detected
            param_names = [p[0] for p in scan_params]
            print("\n" + "="*60)
            print(f"Mode: SCAN (2D parameter space)")
            print(f"Scan parameters: {param_names[0]} × {param_names[1]}")
            print(f"Input: {os.path.basename(input_file)}")
            
            # Display scan ranges
            for param_name, min_val, max_val, n_points in scan_params:
                print(f"  {param_name}: [{min_val}, {max_val}] with {n_points} points")
            print("="*60 + "\n")
            
            D0FUS_scan.main(input_file)
    
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
        print("\n\n Operation cancelled by user. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
