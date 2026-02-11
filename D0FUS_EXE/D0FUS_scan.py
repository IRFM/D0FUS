"""
D0FUS Scan Module
Generates 2D parameter space maps with full visualization
Supports scanning any 2 parameters dynamically
Allows user to choose any two output parameters for iso-contours visualization

"""
#%% Imports
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Matteo Fletcher addition 20260106
from typing import Optional, Tuple, Dict, List

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_EXE.D0FUS_run import run, Parameters, CostParameters
from D0FUS_BIB.D0FUS_cost import *
from D0FUS_BIB.D0FUS_cost_data import *

#%% Output Parameter Registry

@dataclass
class OutputParameter:
    """
    Complete metadata for a scan output parameter.
    Centralizes all information needed for storage, plotting, and display.
    """
    name: str                           # Internal key name
    label: str                          # LaTeX label for plots
    unit: str = ""                      # Physical unit
    levels: Optional[Tuple] = None      # (start, stop, step) for contour levels
    fmt: str = "%.1f"                   # Format string for contour labels
    use_radial_mask: bool = True        # Apply radial build validity mask?
    category: str = "other"             # Category for organization
    description: str = ""               # Human-readable description
    
    def get_levels(self) -> Optional[np.ndarray]:
        """Generate contour levels array from (start, stop, step) tuple"""
        if self.levels is None:
            return None
        return np.arange(*self.levels)
    
    def get_label_with_unit(self) -> str:
        """Return label with unit for legend"""
        if self.unit:
            return f"{self.label} [{self.unit}]"
        return self.label


# =============================================================================
# OUTPUT PARAMETER REGISTRY - Organized by categories
# =============================================================================

OUTPUT_REGISTRY: Dict[str, OutputParameter] = {
    
    # -------------------------------------------------------------------------
    # PLASMA PERFORMANCE
    # -------------------------------------------------------------------------
    'Q': OutputParameter(
        name='Q',
        label='$Q$',
        unit='',
        levels=(0, 150, 10),
        fmt='%d',
        use_radial_mask=False,
        category='performance',
        description='Fusion gain factor'
    ),
    'P_fus': OutputParameter(
        name='P_fus',
        label='$P_{fus}$',
        unit='MW',
        levels=(0, 5000, 100),
        fmt='%d',
        use_radial_mask=False,
        category='performance',
        description='Fusion power'
    ),
    'P_elec': OutputParameter(
        name='P_elec',
        label='$P_{elec}$',
        unit='MW',
        levels=(300, 2000, 50),
        fmt='%d',
        use_radial_mask=False,
        category='performance',
        description='Electric power output'
    ),
    'Cost': OutputParameter(
        name='Cost',
        label='Cost',
        unit='m$^3$',
        levels=(0, 1000, 20),
        fmt='%d',
        use_radial_mask=False,
        category='performance',
        description='Reactor volume (cost proxy)'
    ),
    'T_op_limit': OutputParameter(
        name='T_op_limit',
        label='$\\Delta t_{av}$',
        unit='yr',
        levels=(0, 10, 0.2),
        fmt='%.1f',
        use_radial_mask=True,
        category='performance',
        description='Maximum operational time'
    ),
    'C_invest': OutputParameter(
        name='C_invest',
        label='$C_{invest}$',
        unit='B€',
        levels=(0, 50, 2),
        fmt='%d',
        use_radial_mask=True,
        category='performance',
        description='investement / capital cost'
    ),
    'CF': OutputParameter(
        name='CF',
        label='Capacity factor',
        unit='',
        levels=(0, 1, 0.02),
        fmt='%.2f',
        use_radial_mask=True,
        category='performance',
        description='Capacity factor'
    ),
    'COE': OutputParameter(
        name='COE',
        label='COE',
        unit='€/MWh',
        levels=(50, 700, 20),
        fmt='%d',
        use_radial_mask=True,
        category='performance',
        description='Cost of electricity'
    ),
    
    # -------------------------------------------------------------------------
    # PLASMA PARAMETERS
    # -------------------------------------------------------------------------
    'Ip': OutputParameter(
        name='Ip',
        label='$I_p$',
        unit='MA',
        levels=(1, 30, 1),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Plasma current'
    ),
    'n': OutputParameter(
        name='n',
        label='$\\bar{n}$',
        unit='$10^{20}$ m$^{-3}$',
        levels=(0.25, 5, 0.25),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='Line-averaged electron density'
    ),
    'beta_N': OutputParameter(
        name='beta_N',
        label='$\\beta_N$',
        unit='%',
        levels=(0.5, 5, 0.25),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='Normalized beta'
    ),
    'beta_T': OutputParameter(
        name='beta_T',
        label='$\\beta_T$',
        unit='%',
        levels=(0, 20, 1),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Toroidal beta'
    ),
    'beta_P': OutputParameter(
        name='beta_P',
        label='$\\beta_P$',
        unit='',
        levels=(0, 5, 0.25),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='Poloidal beta'
    ),
    'q95': OutputParameter(
        name='q95',
        label='$q_{95}$',
        unit='',
        levels=(2, 10, 0.5),
        fmt='%.1f',
        use_radial_mask=True,
        category='plasma',
        description='Safety factor at 95% flux'
    ),
    'qstar': OutputParameter(
        name='qstar',
        label='$q_*$',
        unit='',
        levels=(1, 8, 0.5),
        fmt='%.1f',
        use_radial_mask=True,
        category='plasma',
        description='Cylindrical safety factor'
    ),
    'tauE': OutputParameter(
        name='tauE',
        label='$\\tau_E$',
        unit='s',
        levels=(0, 10, 0.5),
        fmt='%.1f',
        use_radial_mask=True,
        category='plasma',
        description='Energy confinement time'
    ),
    'W_th': OutputParameter(
        name='W_th',
        label='$W_{th}$',
        unit='MJ',
        levels=(0, 2000, 100),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Thermal stored energy'
    ),
    'f_bs': OutputParameter(
        name='f_bs',
        label='$f_{BS}$',
        unit='%',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Bootstrap fraction'
    ),
    'f_alpha': OutputParameter(
        name='f_alpha',
        label='$f_\\alpha$',
        unit='%',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=False,
        category='plasma',
        description='Alpha heating fraction'
    ),
    
    # -------------------------------------------------------------------------
    # MAGNETIC FIELD
    # -------------------------------------------------------------------------
    'B0': OutputParameter(
        name='B0',
        label='$B_0$',
        unit='T',
        levels=(0, 20, 0.5),
        fmt='%.1f',
        use_radial_mask=True,
        category='magnetic',
        description='On-axis toroidal field'
    ),
    'BCS': OutputParameter(
        name='BCS',
        label='$B_{CS}$',
        unit='T',
        levels=(0, 25, 1),
        fmt='%d',
        use_radial_mask=True,
        category='magnetic',
        description='Central solenoid peak field'
    ),
    'B_pol': OutputParameter(
        name='B_pol',
        label='$B_{pol}$',
        unit='T',
        levels=(0, 3, 0.1),
        fmt='%.1f',
        use_radial_mask=True,
        category='magnetic',
        description='Poloidal field at edge'
    ),
    'J_TF': OutputParameter(
        name='J_TF',
        label='$J_{TF}$',
        unit='A/mm²',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=True,
        category='magnetic',
        description='TF coil current density'
    ),
    'J_CS': OutputParameter(
        name='J_CS',
        label='$J_{CS}$',
        unit='A/mm²',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=True,
        category='magnetic',
        description='CS coil current density'
    ),
    
    # -------------------------------------------------------------------------
    # POWER & HEAT EXHAUST
    # -------------------------------------------------------------------------
    'Heat': OutputParameter(
        name='Heat',
        label='$q_\\parallel B_T/B_P$',
        unit='MW·T/m',
        levels=(500, 15000, 500),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Parallel heat flux parameter'
    ),
    'Gamma_n': OutputParameter(
        name='Gamma_n',
        label='$\\Gamma_n$',
        unit='MW/m²',
        levels=(0, 6, 0.5),
        fmt='%.2f',
        use_radial_mask=False,
        category='power',
        description='Neutron wall loading'
    ),
    'P_CD': OutputParameter(
        name='P_CD',
        label='$P_{CD}$',
        unit='MW',
        levels=(0, 200, 10),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Current drive power'
    ),
    'P_sep': OutputParameter(
        name='P_sep',
        label='$P_{sep}$',
        unit='MW',
        levels=(0, 500, 25),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Power crossing separatrix'
    ),
    'P_Thresh': OutputParameter(
        name='P_Thresh',
        label='$P_{L-H}$',
        unit='MW',
        levels=(0, 200, 10),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='L-H transition threshold power'
    ),
    'L_H': OutputParameter(
        name='L_H',
        label='$P_{sep}/P_{L-H}$',
        unit='',
        levels=(0, 10, 0.5),
        fmt='%.1f',
        use_radial_mask=False,
        category='power',
        description='L-H margin ratio'
    ),
    'P_Brem': OutputParameter(
        name='P_Brem',
        label='$P_{Brem}$',
        unit='MW',
        levels=(0, 200, 10),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Bremsstrahlung power loss'
    ),
    'P_syn': OutputParameter(
        name='P_syn',
        label='$P_{syn}$',
        unit='MW',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Synchrotron radiation power'
    ),
    'q_target': OutputParameter(
        name='q_target',
        label='$q_{target}$',
        unit='MW/m²',
        levels=(0, 50, 2),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Peak divertor heat flux'
    ),
    'lambda_q': OutputParameter(
        name='lambda_q',
        label='$\\lambda_q$',
        unit='mm',
        levels=(0, 10, 0.5),
        fmt='%.1f',
        use_radial_mask=False,
        category='power',
        description='SOL power decay length'
    ),
    
    # -------------------------------------------------------------------------
    # GEOMETRY & RADIAL BUILD
    # -------------------------------------------------------------------------
    'c': OutputParameter(
        name='c',
        label='$\\Delta_{TF}$',
        unit='m',
        levels=(0, 3, 0.1),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='TF coil radial thickness'
    ),
    'd': OutputParameter(
        name='d',
        label='$\\Delta_{CS}$',
        unit='m',
        levels=(0, 2, 0.1),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='CS coil radial thickness'
    ),
    'c_d': OutputParameter(
        name='c_d',
        label='$\\Delta_{TF}+\\Delta_{CS}$',
        unit='m',
        levels=(0, 5, 0.2),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='Combined TF + CS thickness'
    ),
    'r_minor': OutputParameter(
        name='r_minor',
        label='$a$',
        unit='m',
        levels=(0, 4, 0.2),
        fmt='%.1f',
        use_radial_mask=True,
        category='geometry',
        description='Plasma minor radius'
    ),
    'kappa': OutputParameter(
        name='kappa',
        label='$\\kappa$',
        unit='',
        levels=(1, 3, 0.1),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='Plasma elongation'
    ),
    'kappa_95': OutputParameter(
        name='kappa_95',
        label='$\\kappa_{95}$',
        unit='',
        levels=(1, 2.5, 0.1),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='Elongation at 95% flux'
    ),
    'delta': OutputParameter(
        name='delta',
        label='$\\delta$',
        unit='',
        levels=(0, 1, 0.05),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='Plasma triangularity'
    ),
    'Volume': OutputParameter(
        name='Volume',
        label='$V_p$',
        unit='m³',
        levels=(0, 3000, 100),
        fmt='%d',
        use_radial_mask=True,
        category='geometry',
        description='Plasma volume'
    ),
    'Surface': OutputParameter(
        name='Surface',
        label='$S_p$',
        unit='m²',
        levels=(0, 2000, 100),
        fmt='%d',
        use_radial_mask=True,
        category='geometry',
        description='Plasma surface area'
    ),
    'A': OutputParameter(
        name='A',
        label='$A$',
        unit='',
        levels=(2, 6, 0.25),
        fmt='%.2f',
        use_radial_mask=True,
        category='geometry',
        description='Aspect ratio R0/a'
    ),
    
    # -------------------------------------------------------------------------
    # STRUCTURAL / MECHANICAL
    # -------------------------------------------------------------------------
    'sigma_TF': OutputParameter(
        name='sigma_TF',
        label='$\\sigma_{VM,TF}$',
        unit='MPa',
        levels=(0, 1000, 50),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='TF coil Von Mises stress'
    ),
    'sigma_CS': OutputParameter(
        name='sigma_CS',
        label='$\\sigma_{VM,CS}$',
        unit='MPa',
        levels=(0, 1000, 50),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='CS coil Von Mises stress'
    ),
    'Steel_fraction_TF': OutputParameter(
        name='Steel_fraction_TF',
        label='$f_{steel,TF}$',
        unit='%',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='TF steel fraction'
    ),
    'Steel_fraction_CS': OutputParameter(
        name='Steel_fraction_CS',
        label='$f_{steel,CS}$',
        unit='%',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='CS steel fraction'
    ),
    
    # -------------------------------------------------------------------------
    # PLASMA LIMITS (internal use, colored background)
    # -------------------------------------------------------------------------
    'limits': OutputParameter(
        name='limits',
        label='Max limit',
        unit='',
        levels=(0, 2, 0.1),
        fmt='%.1f',
        use_radial_mask=False,
        category='limits',
        description='Maximum of all plasma limits'
    ),
    'density_limit': OutputParameter(
        name='density_limit',
        label='$n/n_G$',
        unit='',
        levels=(0.5, 2, 0.1),
        fmt='%.2f',
        use_radial_mask=False,
        category='limits',
        description='Greenwald density fraction'
    ),
    'beta_limit': OutputParameter(
        name='beta_limit',
        label='$\\beta_N/\\beta_{N,lim}$',
        unit='',
        levels=(0.5, 2, 0.1),
        fmt='%.2f',
        use_radial_mask=False,
        category='limits',
        description='Beta limit fraction'
    ),
    'q_limit': OutputParameter(
        name='q_limit',
        label='$q_{lim}/q_*$',
        unit='',
        levels=(0.5, 2, 0.1),
        fmt='%.2f',
        use_radial_mask=False,
        category='limits',
        description='Safety factor limit fraction'
    ),
    'radial_build': OutputParameter(
        name='radial_build',
        label='Radial build',
        unit='',
        levels=None,
        fmt='%.1f',
        use_radial_mask=False,
        category='limits',
        description='Radial build validity flag'
    ),
}

# Category descriptions for display
CATEGORY_INFO = {
    'performance': ('Performance', 'Global reactor performance metrics'),
    'plasma': ('Plasma Parameters', 'Core plasma physics quantities'),
    'magnetic': ('Magnetic Field', 'Magnetic field and coil currents'),
    'power': ('Power & Heat', 'Power balance and heat exhaust'),
    'geometry': ('Geometry', 'Plasma and coil dimensions'),
    'structural': ('Structural', 'Mechanical stress and materials'),
    'limits': ('Limits', 'Operational limits (internal)'),
}


#%% ScanOutputs Class

class ScanOutputs:
    """
    Container class for all 2D scan output matrices.
    Provides convenient access, storage, and plotting utilities.
    """
    
    def __init__(self, shape: Tuple[int, int]):
        """
        Initialize all output matrices with given shape.
        
        Args:
            shape: (n_param1, n_param2) dimensions of the scan grid
        """
        self.shape = shape
        self._matrices: Dict[str, np.ndarray] = {}
        
        # Initialize all registered parameters
        for name in OUTPUT_REGISTRY:
            self._matrices[name] = np.full(shape, np.nan)
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get matrix by parameter name"""
        if key not in self._matrices:
            raise KeyError(f"Unknown output parameter: {key}. Available: {list(self._matrices.keys())}")
        return self._matrices[key]
    
    def __setitem__(self, key: str, value: np.ndarray):
        """Set entire matrix"""
        self._matrices[key] = value
    
    def set_point(self, y: int, x: int, **kwargs):
        """
        Set multiple parameter values at a single grid point.
        
        Args:
            y: Index along first scan parameter
            x: Index along second scan parameter
            **kwargs: Parameter name-value pairs
        """
        for key, value in kwargs.items():
            if key in self._matrices:
                self._matrices[key][y, x] = value
            # Silently ignore unknown keys (flexibility for future additions)
    
    def fill_nan(self, y: int, x: int):
        """Fill all matrices with NaN at given point (for error cases)"""
        for matrix in self._matrices.values():
            matrix[y, x] = np.nan
    
    def get_definition(self, key: str) -> OutputParameter:
        """Get the OutputParameter definition for a key"""
        return OUTPUT_REGISTRY.get(key)
    
    def get_masked(self, key: str, radial_build_matrix: np.ndarray = None) -> np.ndarray:
        """
        Get matrix with optional radial build mask applied.
        
        Args:
            key: Parameter name
            radial_build_matrix: If provided, apply NaN mask where radial build is invalid
        """
        matrix = self._matrices[key].copy()
        param = OUTPUT_REGISTRY.get(key)
        
        if param and param.use_radial_mask and radial_build_matrix is not None:
            mask = np.isnan(radial_build_matrix)
            matrix[mask] = np.nan
        
        return matrix
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Export all matrices as dictionary (for backward compatibility)"""
        return self._matrices.copy()
    
    @property
    def available_parameters(self) -> List[str]:
        """List all available parameter names"""
        return list(self._matrices.keys())
    
    @staticmethod
    def get_parameters_by_category(category: str) -> List[str]:
        """Get list of parameter names for a given category"""
        return [name for name, param in OUTPUT_REGISTRY.items() 
                if param.category == category]
    
    @staticmethod
    def list_plottable_parameters() -> Dict[str, List[str]]:
        """
        Return dictionary of plottable parameters organized by category.
        Excludes internal 'limits' category.
        """
        result = {}
        for cat_key, (cat_name, cat_desc) in CATEGORY_INFO.items():
            if cat_key == 'limits':
                continue
            params = [name for name, p in OUTPUT_REGISTRY.items() 
                     if p.category == cat_key and p.levels is not None]
            if params:
                result[cat_name] = params
        return result


#%% Input File Parsing

def parse_scan_parameter(line: str) -> Optional[Tuple[str, float, float, int]]:
    """
    Parse a scan parameter line with bracket syntax.
    Example: "R0 = [3, 9, 25]" -> ("R0", 3.0, 9.0, 25)
    
    Returns:
        tuple: (param_name, min_value, max_value, n_points) or None
    """
    match = re.match(r'^\s*(\w+)\s*=\s*\[([^\]]+)\]', line)
    if not match:
        return None
    
    param_name = match.group(1).strip()
    values_str = match.group(2)
    
    values = re.split(r'[,;]', values_str)
    if len(values) != 3:
        raise ValueError(f"Scan parameter {param_name} must have exactly 3 values: [min, max, n_points]")
    
    min_val = float(values[0].strip())
    max_val = float(values[1].strip())
    n_points = int(float(values[2].strip()))
    
    return (param_name, min_val, max_val, n_points)


def load_scan_parameters(input_file: str) -> Tuple[List, Dict]:
    """
    Load parameters from input file, identifying scan parameters.
    
    Returns:
        tuple: (scan_params, fixed_params)
            scan_params: list of (name, min, max, n_points)
            fixed_params: dict of fixed parameter values
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    scan_params = []
    fixed_params = {}
    scan_param_names = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            scan_param = parse_scan_parameter(line)
            if scan_param:
                scan_params.append(scan_param)
                scan_param_names.append(scan_param[0])
                continue
            
            if '=' in line:
                parts = line.split('=', 1)
                param_name = parts[0].strip()
                param_value = parts[1].strip()
                
                if param_name in scan_param_names:
                    continue
                
                try:
                    param_value = float(param_value)
                    if param_value.is_integer():
                        param_value = int(param_value)
                except ValueError:
                    pass
                
                fixed_params[param_name] = param_value
    
    if len(scan_params) != 2:
        raise ValueError(f"Expected exactly 2 scan parameters, found {len(scan_params)}")
    
    return scan_params, fixed_params


def get_input_parameter_unit(param_name: str) -> str:
    """Get the unit for an input parameter name"""
    units = {
        'R0': 'm', 'a': 'm', 'b': 'm',
        'P_fus': 'MW',
        'Bmax': 'T',
        'Tbar': 'keV',
        'H': '',
        'nu_n': '', 'nu_T': '',
    }
    return units.get(param_name, '')


#%% Core Scan Function

def generic_2D_scan(scan_params: List, fixed_params: Dict, params_obj) -> Tuple[ScanOutputs, np.ndarray, np.ndarray, str, str]:
    """
    Perform generic 2D scan over any two parameters.
    
    Args:
        scan_params: list of 2 tuples (name, min, max, n_points)
        fixed_params: dict of fixed parameter values
        params_obj: Parameters object to update
    
    Returns:
        outputs: ScanOutputs object containing all result matrices
        param1_values: array of first parameter values
        param2_values: array of second parameter values
        param1_name: name of first scan parameter
        param2_name: name of second scan parameter
    """
    
    # Extract scan parameters
    param1_name, param1_min, param1_max, param1_n = scan_params[0]
    param2_name, param2_min, param2_max, param2_n = scan_params[1]
    
    param1_values = np.linspace(param1_min, param1_max, param1_n)
    param2_values = np.linspace(param2_min, param2_max, param2_n)
    
    print(f"\nStarting 2D scan:")
    print(f"  {param1_name}: [{param1_min}, {param1_max}] with {param1_n} points")
    print(f"  {param2_name}: [{param2_min}, {param2_max}] with {param2_n} points")
    print(f"  Total calculations: {param1_n * param2_n}\n")
    
    # Initialize outputs container
    outputs = ScanOutputs(shape=(param1_n, param2_n))
    
    # Apply fixed parameters to params_obj
    for param_name, param_value in fixed_params.items():
        # Check if it's a cost model change
        if param_name == 'cost_model':
            params_obj.cost_model = param_value
            params_obj.cost = CostParameters(param_value)
            print(f"  Fixed: cost_model = {param_value}")
        # Check if it's a cost parameter
        elif hasattr(params_obj.cost, param_name):
            setattr(params_obj.cost, param_name, param_value)
            print(f"  Fixed: cost.{param_name} = {param_value}")
        # Otherwise it's a regular parameter
        elif hasattr(params_obj, param_name):
            setattr(params_obj, param_name, param_value)
            print(f"  Fixed: {param_name} = {param_value}")
        else:
            print(f"  WARNING: Unknown parameter '{param_name}' - skipping")
    
    # Scanning loop
    for y, param1_val in enumerate(tqdm(param1_values, desc=f'Scanning {param1_name}')):
        for x, param2_val in enumerate(param2_values):
            
            # Set scan parameter values
            # Handle param1
            if param1_name == 'cost_model':
                params_obj.cost_model = param1_val
                params_obj.cost = CostParameters(param1_val)
            elif hasattr(params_obj.cost, param1_name):
                setattr(params_obj.cost, param1_name, param1_val)
            else:
                setattr(params_obj, param1_name, param1_val)
            
            # Handle param2
            if param2_name == 'cost_model':
                params_obj.cost_model = param2_val
                params_obj.cost = CostParameters(param2_val)
            elif hasattr(params_obj.cost, param2_name):
                setattr(params_obj.cost, param2_name, param2_val)
            else:
                setattr(params_obj, param2_name, param2_val)
            
            try:
                # Run calculation
                results = run(params_obj)
                
                # Unpack results
                (B0, B_CS, B_pol,
                 tauE, W_th,
                 Q, Volume, Surface,
                 Ip, Ib, I_CD, I_Ohm,
                 nbar, nG, pbar,
                 betaN, betaT, betaP,
                 qstar, q95,
                 P_CD, P_sep, P_Thresh, eta_CD, P_elec,
                 cost, P_Brem, P_syn,
                 heat, heat_par, heat_pol, lambda_q, q_target,
                 P_wall_H, P_wall_L,
                 T_op_limit, C_invest,
                 CF, COE,
                 Gamma_n,
                 f_alpha, tau_alpha,
                 J_TF, J_CS,
                 c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
                 d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS_out, J_CS_out,
                 r_minor, r_sep, r_c, r_d,
                 kappa, kappa_95, δ, δ_95) = results

                
                # Calculate plasma limit conditions
                betaN_limit_value = 2.8
                q_limit_value = 2.5
                
                n_condition = nbar / nG
                beta_condition = betaN / betaN_limit_value
                q_condition = q_limit_value / qstar
                
                max_limit = max(n_condition, beta_condition, q_condition)
                
                # Store all results using set_point
                outputs.set_point(y, x,
                    # Performance
                    Q=Q,
                    P_fus=params_obj.P_fus,
                    P_elec=P_elec,
                    Cost=cost,
                    T_op_limit=T_op_limit,
                    C_invest=C_invest,
                    CF=CF,
                    COE=COE,
                    
                    # Plasma parameters
                    Ip=Ip,
                    n=nbar,
                    beta_N=betaN,
                    beta_T=betaT,
                    beta_P=betaP,
                    q95=q95,
                    qstar=qstar,
                    tauE=tauE,
                    W_th=W_th,
                    f_bs=Ib/Ip * 100 if Ip > 0 else 0,
                    f_alpha=f_alpha * 100,
                    
                    # Magnetic field
                    B0=B0,
                    BCS=B_CS,
                    B_pol=B_pol,
                    J_TF=J_TF,
                    J_CS=J_CS,
                    
                    # Power & heat
                    Heat=heat,
                    Gamma_n=Gamma_n,
                    P_CD=P_CD,
                    P_sep=P_sep,
                    P_Thresh=P_Thresh,
                    L_H=P_sep / P_Thresh if P_Thresh > 0 else np.nan,
                    P_Brem=P_Brem,
                    P_syn=P_syn,
                    q_target=q_target,
                    lambda_q=lambda_q * 1000 if lambda_q else np.nan,  # Convert to mm
                    
                    # Geometry
                    c=r_sep - r_c if not np.isnan(r_c) else np.nan,
                    d=r_c - r_d if not np.isnan(r_d) else np.nan,
                    r_minor=r_minor,
                    kappa=kappa,
                    kappa_95=kappa_95,
                    delta=δ,
                    Volume=Volume,
                    Surface=Surface,
                    A=params_obj.R0 / params_obj.a if params_obj.a > 0 else np.nan,
                    
                    # Structural
                    sigma_TF=max(abs(σ_z_TF), abs(σ_theta_TF), abs(σ_r_TF)) if σ_z_TF else np.nan,
                    sigma_CS=max(abs(σ_z_CS), abs(σ_theta_CS), abs(σ_r_CS)) if σ_z_CS else np.nan,
                    Steel_fraction_TF=Steel_fraction_TF * 100 if Steel_fraction_TF else np.nan,
                    Steel_fraction_CS=Steel_fraction_CS * 100 if Steel_fraction_CS else np.nan,
                    
                    # Limits
                    limits=max_limit,
                    density_limit=n_condition,
                    beta_limit=beta_condition,
                    q_limit=q_condition,
                )
                
                # Combined c+d
                c_val = r_sep - r_c if not np.isnan(r_c) else np.nan
                d_val = r_c - r_d if not np.isnan(r_d) else np.nan
                outputs['c_d'][y, x] = c_val + d_val if not (np.isnan(c_val) or np.isnan(d_val)) else np.nan
                
                # Check radial build validity
                if not np.isnan(r_d) and max_limit < 1 and r_d > 0:
                    outputs['radial_build'][y, x] = params_obj.R0
                else:
                    outputs['radial_build'][y, x] = np.nan
                
            except Exception as e:
                outputs.fill_nan(y, x)
                if y < 2 and x < 2:
                    print(f"\n  Debug: Error at {param1_name}={param1_val:.2f}, {param2_name}={param2_val:.2f}: {str(e)}")
                continue
    
    print("\n✓ Scan calculation completed!\n")
    return outputs, param1_values, param2_values, param1_name, param2_name


#%% Plotting Functions

def display_available_parameters():
    """Display available parameters for iso-contours, organized by category"""
    print("\n" + "="*70)
    print("AVAILABLE OUTPUT PARAMETERS FOR ISO-CONTOURS")
    print("="*70)
    
    params_by_cat = ScanOutputs.list_plottable_parameters()
    
    for cat_name, params in params_by_cat.items():
        print(f"\n  {cat_name}:")
        # Display in rows of 4
        for i in range(0, len(params), 4):
            row = params[i:i+4]
            print("    " + ", ".join(f"{p:15}" for p in row))
    
    print("\n" + "="*70)


def get_user_plot_choice(prompt: str, valid_options: List[str]) -> str:
    """
    Get user's choice for plotting parameter with validation.
    
    Args:
        prompt: Question to display
        valid_options: List of valid parameter names
    
    Returns:
        Selected parameter name
    """
    while True:
        choice = input(prompt).strip()
        if choice in valid_options:
            return choice
        print(f"  Invalid choice '{choice}'. Please choose from the available parameters.")
        print(f"  Hint: {', '.join(valid_options[:10])}...")


def plot_generic_contours(ax, matrix: np.ndarray, param_key: str, 
                          color: str = 'black', linestyle: str = 'dashed',
                          linewidth: float = 2.5, fontsize: int = 22):
    """
    Plot contours for any registered parameter.
    
    Args:
        ax: Matplotlib axes
        matrix: 2D data matrix (already inverted if needed)
        param_key: Parameter name from OUTPUT_REGISTRY
        color: Contour line color
        linestyle: Line style ('solid', 'dashed', etc.)
        linewidth: Line width
        fontsize: Font size for contour labels
    
    Returns:
        Line2D object for legend, or None if no contours plotted
    """
    param = OUTPUT_REGISTRY.get(param_key)
    if param is None:
        print(f"  Warning: Unknown parameter '{param_key}'")
        return None
    
    levels = param.get_levels()
    if levels is None or len(levels) == 0:
        return None
    
    # Filter levels to data range
    data_min, data_max = np.nanmin(matrix), np.nanmax(matrix)
    valid_levels = levels[(levels >= data_min) & (levels <= data_max)]
    
    if len(valid_levels) == 0:
        print(f"  Note: No contour levels in data range for {param_key} [{data_min:.2f}, {data_max:.2f}]")
        return None
    
    try:
        contour = ax.contour(matrix, levels=valid_levels, colors=color,
                            linestyles=linestyle, linewidths=linewidth)
        ax.clabel(contour, inline=True, fmt=param.fmt, fontsize=fontsize)
        
        # Create legend entry
        legend_line = mlines.Line2D([], [], color=color, linestyle=linestyle,
                                   linewidth=linewidth, label=param.get_label_with_unit())
        return legend_line
    
    except Exception as e:
        print(f"  Warning: Could not plot contours for {param_key}: {e}")
        return None


def plot_scan_results(outputs: ScanOutputs, param1_values: np.ndarray, param2_values: np.ndarray,
                      param1_name: str, param2_name: str, params, output_dir,
                      iso_param_1: str = None, iso_param_2: str = None):
    """
    Generate scan visualization with two user-selectable iso-contour parameters.
    
    Args:
        outputs: ScanOutputs object with all result matrices
        param1_values: Array of first scan parameter values (Y-axis)
        param2_values: Array of second scan parameter values (X-axis)
        param1_name: Name of first scan parameter
        param2_name: Name of second scan parameter
        params: Parameters object (for reference values)
        output_dir: Output directory (unused, kept for compatibility)
        iso_param_1: Pre-selected first iso-contour parameter (black lines, optional)
        iso_param_2: Pre-selected second iso-contour parameter (white lines, optional)
    
    Returns:
        fig, ax, iso_param_1, iso_param_2
    """
    
    # Get units for scan parameters
    unit_param1 = get_input_parameter_unit(param1_name)
    unit_param2 = get_input_parameter_unit(param2_name)
    
    # Get available plottable parameters
    all_plottable = []
    for params_list in ScanOutputs.list_plottable_parameters().values():
        all_plottable.extend(params_list)
    
    # User selection if not provided
    if iso_param_1 is None or iso_param_2 is None:
        display_available_parameters()
    
    if iso_param_1 is None:
        iso_param_1 = get_user_plot_choice(
            "\nChoose ISO-CONTOUR 1 (black dashed lines): ", 
            all_plottable
        )
    
    if iso_param_2 is None:
        iso_param_2 = get_user_plot_choice(
            "Choose ISO-CONTOUR 2 (white dashed lines): ", 
            all_plottable
        )
    
    print(f"\n  Plotting: iso_1={iso_param_1}, iso_2={iso_param_2}")
    
    # Font sizes
    font_iso_1 = 22      # Black iso-contours
    font_iso_2 = 22      # White iso-contours
    font_legend = 20
    font_other = 15
    plt.rcParams.update({'font.size': font_other})
    
    # Get matrices and invert for plotting (Y-axis convention)
    radial_build = outputs['radial_build'][::-1, :]
    
    # Get iso-contour matrices with fixed masking behavior:
    #   - iso_param_1 (black): ALWAYS masked to radial build valid region
    #   - iso_param_2 (white): ALWAYS on full figure (no mask)
    iso_matrix_1 = outputs.get_masked(iso_param_1, outputs['radial_build'])[::-1, :]
    iso_matrix_2 = outputs[iso_param_2][::-1, :]
    
    # Limit matrices for colored background
    inv_density = outputs['density_limit'][::-1, :].copy()
    inv_beta = outputs['beta_limit'][::-1, :].copy()
    inv_q = outputs['q_limit'][::-1, :].copy()
    inv_limits = outputs['limits'][::-1, :]
    
    # Set NaN where not the dominant limit
    conditions = np.array([inv_density, inv_beta, inv_q])
    idx_max = np.argmax(conditions, axis=0)
    
    inv_density_plot = np.where(idx_max == 0, inv_density, np.nan)
    inv_beta_plot = np.where(idx_max == 1, inv_beta, np.nan)
    inv_q_plot = np.where(idx_max == 2, inv_q, np.nan)
    
    # Only show where limits < 2
    mask_valid = inv_limits < 2
    inv_density_plot = np.where(mask_valid, inv_density_plot, np.nan)
    inv_beta_plot = np.where(mask_valid, inv_beta_plot, np.nan)
    inv_q_plot = np.where(mask_valid, inv_q_plot, np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 13))
    
    # Plot color maps for plasma limits
    min_val, max_val = 0.5, 2.0
    im_density = ax.imshow(inv_density_plot, cmap='Blues', aspect='auto',
                          interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    im_q = ax.imshow(inv_q_plot, cmap='Greens', aspect='auto',
                     interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    im_beta = ax.imshow(inv_beta_plot, cmap='Reds', aspect='auto',
                       interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    
    # Plasma stability boundary (limit = 1)
    linewidth = 2.5
    ax.contour(inv_limits, levels=[1.0], colors='white', linewidths=linewidth)
    white_boundary = mlines.Line2D([], [], color='white', linewidth=linewidth,
                                   label='Plasma stability boundary')
    
    # Radial build boundary
    filled_matrix = np.where(np.isnan(radial_build), -1, 1)
    ax.contour(filled_matrix, levels=[0], linewidths=linewidth, colors='black')
    black_boundary = mlines.Line2D([], [], color='black', linewidth=linewidth,
                                   label='Radial build limit')
    
    # Configure axes labels
    label_param2 = f"${param2_name}$" + (f" [{unit_param2}]" if unit_param2 else "")
    label_param1 = f"${param1_name}$" + (f" [{unit_param1}]" if unit_param1 else "")
    ax.set_xlabel(label_param2, fontsize=24)
    ax.set_ylabel(label_param1, fontsize=24)
    
    # Configure colorbars
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("bottom", size="5%", pad=1.3)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    cax3 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    
    cax1.annotate('$n/n_{G}$', xy=(-0.01, 0.5), xycoords='axes fraction',
                 ha='right', va='center', fontsize=font_other)
    cax2.annotate(r'$\beta_N/\beta_{lim}$', xy=(-0.01, 0.5), xycoords='axes fraction',
                 ha='right', va='center', fontsize=font_other)
    cax3.annotate('$q_{lim}/q_*$', xy=(-0.01, 0.5), xycoords='axes fraction',
                 ha='right', va='center', fontsize=font_other)
    
    cbar_density = plt.colorbar(im_density, cax=cax1, orientation='horizontal')
    if cbar_density.ax.xaxis.get_ticklabels():
        cbar_density.ax.xaxis.get_ticklabels()[-1].set_visible(False)
    
    cbar_beta = plt.colorbar(im_beta, cax=cax2, orientation='horizontal')
    if cbar_beta.ax.xaxis.get_ticklabels():
        cbar_beta.ax.xaxis.get_ticklabels()[-1].set_visible(False)
    
    cbar_q = plt.colorbar(im_q, cax=cax3, orientation='horizontal')
    
    for cax in [cax1, cax2, cax3]:
        cax.axvline(x=1, color='white', linewidth=2.5)
    
    # Configure Y-axis ticks (param1)
    approx_step_y = (param1_values[-1] - param1_values[0]) / 10
    real_step_y = (param1_values[-1] - param1_values[0]) / (len(param1_values) - 1)
    index_step_y = max(1, int(round(approx_step_y / real_step_y)))
    y_indices = np.arange(0, len(param1_values), index_step_y)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(np.round(param1_values[::-1][y_indices], 2), fontsize=font_legend)
    
    # Configure X-axis ticks (param2)
    approx_step_x = (param2_values[-1] - param2_values[0]) / 10
    real_step_x = (param2_values[-1] - param2_values[0]) / (len(param2_values) - 1)
    index_step_x = max(1, int(round(approx_step_x / real_step_x)))
    x_indices = np.arange(0, len(param2_values), index_step_x)
    ax.set_xticks(x_indices)
    x_labels = [round(param2_values[i], 2) for i in x_indices]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=font_legend)
    
    # Plot ISO-CONTOUR 1 (black dashed)
    iso_legend_1 = plot_generic_contours(ax, iso_matrix_1, iso_param_1,
                                         color='black', linestyle='dashed',
                                         linewidth=linewidth, fontsize=font_iso_1)
    
    # Plot ISO-CONTOUR 2 (white dashed)
    iso_legend_2 = plot_generic_contours(ax, iso_matrix_2, iso_param_2,
                                         color='white', linestyle='dashed',
                                         linewidth=linewidth, fontsize=font_iso_2)
    
    # Build legend
    legend_handles = [white_boundary, black_boundary]
    if iso_legend_2:
        legend_handles.append(iso_legend_2)
    if iso_legend_1:
        legend_handles.append(iso_legend_1)
    
    ax.legend(handles=legend_handles, loc='upper left', facecolor='lightgrey',
             fontsize=font_legend)
    
    return fig, ax, iso_param_1, iso_param_2


#%% Save Results

def save_scan_results(fig, outputs: ScanOutputs, param1_values: np.ndarray, param2_values: np.ndarray,
                     param1_name: str, param2_name: str, params, output_dir: str,
                     iso_param_1: str, iso_param_2: str, input_file_path: str = None) -> str:
    """
    Save scan results to timestamped directory.
    
    Returns:
        Path to output directory
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"Scan_D0FUS_{timestamp}"
    output_path = os.path.join(output_dir, 'scan', output_name)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Copy original input file if provided
    if input_file_path and os.path.exists(input_file_path):
        input_copy = os.path.join(output_path, "scan_parameters.txt")
        shutil.copy2(input_file_path, input_copy)
    else:
        # Generate input file from parameters
        input_copy = os.path.join(output_path, "scan_parameters.txt")
        with open(input_copy, "w", encoding='utf-8') as f:
            f.write("# D0FUS Scan Parameters\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("# Scan parameters:\n")
            f.write(f"{param1_name} = [{param1_values[0]:.2f}, {param1_values[-1]:.2f}, {len(param1_values)}]\n")
            f.write(f"{param2_name} = [{param2_values[0]:.2f}, {param2_values[-1]:.2f}, {len(param2_values)}]\n")
            f.write("\n# Fixed parameters:\n")
            for key, value in vars(params).items():
                if key not in [param1_name, param2_name]:
                    f.write(f"{key} = {value}\n")
            f.write(f"\n# Visualization:\n")
            f.write(f"iso_param_1 = {iso_param_1}  # Black dashed lines\n")
            f.write(f"iso_param_2 = {iso_param_2}  # White dashed lines\n")
    
    # Save figure
    fig_filename = f"scan_map_{param1_name}_{param2_name}_{iso_param_1}_{iso_param_2}.png"
    fig_path = os.path.join(output_path, fig_filename)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    # Save matrices as NPZ for post-processing
    matrices_path = os.path.join(output_path, "scan_matrices.npz")
    np.savez(matrices_path, 
             param1_values=param1_values,
             param2_values=param2_values,
             param1_name=param1_name,
             param2_name=param2_name,
             **outputs.to_dict())
    
    print(f"✓ All results saved to: {output_path}\n")
    
    plt.rcdefaults()
    
    return output_path


#%% Main Function

def main(input_file: str = None, auto_plot: bool = False, 
         iso_param_1: str = None, iso_param_2: str = None):
    """
    Main execution function for scans.
    
    Args:
        input_file: Path to input file (optional)
        auto_plot: If True, use provided iso_param_1 and iso_param_2 without asking
        iso_param_1: First iso-contour parameter - black lines (if auto_plot=True)
        iso_param_2: Second iso-contour parameter - white lines (if auto_plot=True)
    """
    
    # Load parameters
    p = Parameters()
    
    input_file_path = input_file
    
    if input_file is None:
        default_input = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_INPUTS', 'scan_R0_a_example.txt')
        if os.path.exists(default_input):
            input_file = default_input
        else:
            raise FileNotFoundError("No input file provided and default scan input not found")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"\nLoading parameters from: {input_file}")
    
    # Load scan and fixed parameters
    scan_params, fixed_params = load_scan_parameters(input_file)
    
    # Print scan configuration
    print("\n" + "="*73)
    print("Starting D0FUS 2D parameter scan...")
    print("="*73)
    print(f"\nScan parameters:")
    for param_name, min_val, max_val, n_points in scan_params:
        unit = get_input_parameter_unit(param_name)
        unit_str = f" [{unit}]" if unit else ""
        print(f"  {param_name}: [{min_val}, {max_val}]{unit_str} with {n_points} points")
    
    print(f"\nFixed parameters:")
    for key, value in list(fixed_params.items())[:6]:
        print(f"  {key} = {value}")
    if len(fixed_params) > 6:
        print(f"  ... and {len(fixed_params) - 6} more")
    
    try:
        # Perform scan
        outputs, param1_values, param2_values, param1_name, param2_name = generic_2D_scan(
            scan_params, fixed_params, p
        )
        
        # Plot results
        if auto_plot and iso_param_1 and iso_param_2:
            fig, ax, iso_used_1, iso_used_2 = plot_scan_results(
                outputs, param1_values, param2_values, param1_name, param2_name,
                p, None, iso_param_1, iso_param_2
            )
        else:
            fig, ax, iso_used_1, iso_used_2 = plot_scan_results(
                outputs, param1_values, param2_values, param1_name, param2_name,
                p, None
            )
        
        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        output_path = save_scan_results(
            fig, outputs, param1_values, param2_values, param1_name, param2_name,
            p, output_dir, iso_used_1, iso_used_2, input_file_path
        )
        
        plt.show()
        
        return outputs, param1_values, param2_values, output_path
    
    except Exception as e:
        print(f"\n!!! ERROR during scan !!!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = None
    
    main(input_file)
    
    print("\nD0FUS_scan completed successfully!")