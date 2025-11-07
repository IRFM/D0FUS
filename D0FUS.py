"""
D0FUS - Fusion Reactor Design Tool
Main entry point for running calculations and parameter scans

Created on: Dec 2023
Author: Auclair Timothe

Usage:
    python D0FUS.py                          # Interactive mode
    python D0FUS.py run <input_file>         # Run mode
    python D0FUS.py scan <input_file>        # Scan mode
"""

import sys
import os
from pathlib import Path

# Add D0FUS_EXE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'D0FUS_EXE'))

from D0FUS_EXE import D0FUS_run, D0FUS_scan


def print_banner():
    """Display D0FUS banner"""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║              D0FUS - Version 1.0                  ║
    ║      Fusion Reactor Design & Analysis Tool        ║
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
    return input_files


def select_input_file():
    """Interactive input file selection"""
    input_files = list_input_files()
    
    if not input_files:
        print("\nNo input files found in D0FUS_INPUTS directory.")
        print("Using default parameters.")
        return None
    
    print("\n" + "="*50)
    print("Available input files:")
    print("="*50)
    for i, file in enumerate(input_files, 1):
        print(f"  {i}. {file.name}")
    print(f"  0. Use default parameters")
    print("="*50)
    
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


def select_mode():
    """Interactive mode selection"""
    print("\n" + "="*50)
    print("Select operation mode:")
    print("="*50)
    print("  1. RUN  - Calculate a single design point")
    print("  2. SCAN - Generate 2D parameter space map")
    print("  0. EXIT")
    print("="*50)
    
    while True:
        try:
            choice = input("\nSelect mode (number): ").strip()
            choice = int(choice)
            
            if choice == 0:
                print("\nExiting D0FUS. Goodbye!")
                sys.exit(0)
            elif choice == 1:
                return 'run'
            elif choice == 2:
                return 'scan'
            else:
                print("Invalid choice. Please enter 1, 2, or 0.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)


def interactive_mode():
    """Interactive mode for D0FUS"""
    print_banner()
    
    # Select input file
    input_file = select_input_file()
    
    # Select mode
    mode = select_mode()
    
    # Execute
    print("\n" + "="*50)
    if mode == 'run':
        print("Starting RUN mode...")
        print("="*50 + "\n")
        D0FUS_run.main(input_file)
    elif mode == 'scan':
        print("Starting SCAN mode...")
        print("="*50 + "\n")
        D0FUS_scan.main(input_file)


def command_line_mode(args):
    """Command line mode for D0FUS"""
    if len(args) < 2:
        print("Error: Mode not specified.")
        print_usage()
        sys.exit(1)
    
    mode = args[1].lower()
    input_file = args[2] if len(args) > 2 else None
    
    # Validate input file if provided
    if input_file and not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    print_banner()
    
    if mode == 'run':
        print("\n" + "="*50)
        print("Starting RUN mode...")
        print("="*50 + "\n")
        D0FUS_run.main(input_file)
    
    elif mode == 'scan':
        print("\n" + "="*50)
        print("Starting SCAN mode...")
        print("="*50 + "\n")
        D0FUS_scan.main(input_file)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        print_usage()
        sys.exit(1)


def print_usage():
    """Print usage information"""
    usage = """
Usage:
    python D0FUS.py                          # Interactive mode
    python D0FUS.py run [input_file]         # Run mode with optional input file
    python D0FUS.py scan [input_file]        # Scan mode with optional input file
    python D0FUS.py --help                   # Show this help message

Modes:
    run   - Calculate a single design point
    scan  - Generate 2D parameter space map (R0 vs a)

Input files should be placed in the D0FUS_INPUTS directory.
Results are saved in the D0FUS_OUTPUTS directory with timestamps.
    """
    print(usage)


def main():
    """Main entry point"""
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_banner()
        print_usage()
        sys.exit(0)
    
    # Run in appropriate mode
    if len(sys.argv) == 1:
        # No arguments - interactive mode
        try:
            interactive_mode()
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Goodbye!")
            sys.exit(0)
    else:
        # Command line mode
        command_line_mode(sys.argv)


if __name__ == "__main__":
    main()



