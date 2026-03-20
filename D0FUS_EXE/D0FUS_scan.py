"""
D0FUS Scan Module
=================
Generates 2D parameter space maps with full visualization.
Supports scanning any 2 parameters dynamically.
Allows user to choose any two output parameters for iso-contours visualization.

Adapted to the GlobalConfig / dc_replace architecture of D0FUS_run (v2).
"""
#%% Imports
import sys
import os
from dataclasses import replace as dc_replace, asdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_BIB.D0FUS_cost_functions import f_costs_Sheffield
from D0FUS_BIB.D0FUS_cost_data import *
from D0FUS_EXE.D0FUS_run import run, load_config_from_file, _PROFILE_PRESETS
from D0FUS_BIB.D0FUS_parameterization import GlobalConfig, DEFAULT_CONFIG

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
    levels: tuple = None                # (start, stop, step) for contour levels
    fmt: str = "%.1f"                   # Format string for contour labels
    use_radial_mask: bool = True        # Apply radial build validity mask?
    category: str = "other"             # Category for organization
    description: str = ""               # Human-readable description
    
    def get_levels(self):
        """Generate contour levels array from (start, stop, step) tuple"""
        if self.levels is None:
            return None
        return np.arange(*self.levels)
    
    def get_label_with_unit(self):
        """Return label with unit for legend"""
        if self.unit:
            return f"{self.label} [{self.unit}]"
        return self.label


# =============================================================================
# OUTPUT PARAMETER REGISTRY - Organized by categories
# =============================================================================

OUTPUT_REGISTRY = {
    
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
        levels=(0, 2000, 50),
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
    'COE': OutputParameter(
        name='COE',
        label='COE',
        unit='EUR/MWh',
        levels=(50, 500, 25),
        fmt='%d',
        use_radial_mask=True,
        category='performance',
        description='Cost of electricity (Sheffield 2016)'
    ),
    'C_invest': OutputParameter(
        name='C_invest',
        label='$C_{invest}$',
        unit='B EUR',
        levels=(5, 50, 2),
        fmt='%.0f',
        use_radial_mask=True,
        category='performance',
        description='Total capital investment cost'
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
        levels=(0, 5, 0.25),
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
    # RADIATION (additional)
    # -------------------------------------------------------------------------
    'P_line': OutputParameter(
        name='P_line',
        label='$P_{line}$',
        unit='MW',
        levels=(0, 300, 15),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Total impurity line radiation'
    ),
    'P_wallplug': OutputParameter(
        name='P_wallplug',
        label='$P_{wallplug}$',
        unit='MW',
        levels=(0, 500, 25),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Wall-plug heating and CD power'
    ),

    # -------------------------------------------------------------------------
    # ADDITIONAL PLASMA QUANTITIES
    # -------------------------------------------------------------------------
    'pbar': OutputParameter(
        name='pbar',
        label='$\\bar{p}$',
        unit='MPa',
        levels=(0, 1.5, 0.05),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='Volume-averaged plasma pressure'
    ),
    'nbar_vol': OutputParameter(
        name='nbar_vol',
        label='$\\bar{n}_{vol}$',
        unit='$10^{20}$ m$^{-3}$',
        levels=(0.25, 5, 0.25),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='Volume-averaged electron density'
    ),
    'Ib': OutputParameter(
        name='Ib',
        label='$I_{BS}$',
        unit='MA',
        levels=(0, 25, 1),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Bootstrap current'
    ),
    'I_CD': OutputParameter(
        name='I_CD',
        label='$I_{CD}$',
        unit='MA',
        levels=(0, 15, 1),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Non-inductive driven current'
    ),
    'I_Ohm': OutputParameter(
        name='I_Ohm',
        label='$I_{Ohm}$',
        unit='MA',
        levels=(0, 20, 1),
        fmt='%d',
        use_radial_mask=True,
        category='plasma',
        description='Ohmic (inductive) current'
    ),
    'tau_alpha': OutputParameter(
        name='tau_alpha',
        label='$\\tau_\\alpha$',
        unit='s',
        levels=(0, 20, 1),
        fmt='%.1f',
        use_radial_mask=True,
        category='plasma',
        description='Alpha particle confinement time'
    ),

    # -------------------------------------------------------------------------
    # MAGNETIC FLUX
    # -------------------------------------------------------------------------
    'PsiCS': OutputParameter(
        name='PsiCS',
        label='$\\Psi_{CS}$',
        unit='Wb',
        levels=(0, 500, 25),
        fmt='%d',
        use_radial_mask=True,
        category='magnetic',
        description='CS total flux swing'
    ),

    # -------------------------------------------------------------------------
    # PER-SOURCE CURRENT DRIVE
    # -------------------------------------------------------------------------
    'eta_LH': OutputParameter(
        name='eta_LH',
        label='$\\gamma_{LH}$',
        unit='$10^{20}$ A·m$^{-2}$·W$^{-1}$',
        levels=(0, 0.6, 0.05),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='LHCD figure of merit'
    ),
    'eta_EC': OutputParameter(
        name='eta_EC',
        label='$\\gamma_{EC}$',
        unit='$10^{20}$ A·m$^{-2}$·W$^{-1}$',
        levels=(0, 0.4, 0.02),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='ECCD figure of merit'
    ),
    'eta_NBI': OutputParameter(
        name='eta_NBI',
        label='$\\gamma_{NBI}$',
        unit='$10^{20}$ A·m$^{-2}$·W$^{-1}$',
        levels=(0, 0.5, 0.05),
        fmt='%.2f',
        use_radial_mask=True,
        category='plasma',
        description='NBCD figure of merit'
    ),

    # -------------------------------------------------------------------------
    # CONDUCTOR FRACTIONS (TF & CS)
    # -------------------------------------------------------------------------
    'f_sc_TF': OutputParameter(
        name='f_sc_TF',
        label='$f_{SC,TF}$',
        unit='%',
        levels=(0, 60, 5),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='TF superconductor cross-section fraction'
    ),
    'f_He_TF': OutputParameter(
        name='f_He_TF',
        label='$f_{He,TF}$',
        unit='%',
        levels=(0, 50, 5),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='TF helium cross-section fraction'
    ),
    'f_sc_CS': OutputParameter(
        name='f_sc_CS',
        label='$f_{SC,CS}$',
        unit='%',
        levels=(0, 60, 5),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='CS superconductor cross-section fraction'
    ),
    'f_He_CS': OutputParameter(
        name='f_He_CS',
        label='$f_{He,CS}$',
        unit='%',
        levels=(0, 50, 5),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='CS helium cross-section fraction'
    ),

    # -------------------------------------------------------------------------
    # RUNAWAY ELECTRONS (post-convergence diagnostic)
    # -------------------------------------------------------------------------
    # All RE quantities are computed from the converged plasma state using
    # the hot-tail seed model (Smith 2008) + avalanche amplification
    # (Breizman 2019).  They depend on config.tau_TQ and config.Te_final_eV.
    # -------------------------------------------------------------------------
    'I_RE_seed': OutputParameter(
        name='I_RE_seed',
        label='$I_{RE,seed}$',
        unit='kA',
        levels=(0, 10000, 500),
        fmt='%.0f',
        use_radial_mask=True,
        category='runaway',
        description='Hot-tail RE seed current (Smith 2008) — unmitigated ~kA range; stored in kA'
    ),
    'I_RE_aval': OutputParameter(
        name='I_RE_aval',
        label='$I_{RE,aval}$',
        unit='MA',
        levels=(0, 15, 1),
        fmt='%.1f',
        use_radial_mask=True,
        category='runaway',
        description='Final RE current after avalanche amplification (Breizman 2019)'
    ),
    'f_RE_Ip': OutputParameter(
        name='f_RE_Ip',
        label='$I_{RE}/I_p$',
        unit='%',
        levels=(0, 100, 5),
        fmt='%d',
        use_radial_mask=True,
        category='runaway',
        description='Ratio of final RE current to plasma current'
    ),
    'f_RE_avg': OutputParameter(
        name='f_RE_avg',
        label='$\\langle f_{RE} \\rangle$',
        unit='',
        levels=(0, 0.01, 5e-4),
        fmt='%.3e',
        use_radial_mask=True,
        category='runaway',
        description='Volume-averaged hot-tail seed fraction'
    ),
    'f_RE_core': OutputParameter(
        name='f_RE_core',
        label='$f_{RE}(0)$',
        unit='',
        levels=(0, 0.05, 0.005),
        fmt='%.3e',
        use_radial_mask=True,
        category='runaway',
        description='On-axis hot-tail seed fraction'
    ),
    'E_RE_kin': OutputParameter(
        name='E_RE_kin',
        label='$E_{RE,kin}$',
        unit='MJ',
        levels=(0, 2000, 100),
        fmt='%d',
        use_radial_mask=True,
        category='runaway',
        description='Kinetic energy of RE beam (⟨γ⟩=10)'
    ),
    'W_mag_RE': OutputParameter(
        name='W_mag_RE',
        label='$W_{mag,RE}$',
        unit='MJ',
        levels=(0, 500, 25),
        fmt='%d',
        use_radial_mask=True,
        category='runaway',
        description='RE beam magnetic energy (½LI²_RE)'
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
    'runaway': ('Runaway Electrons', 'RE seed and avalanche indicators (post-convergence diagnostic)'),
    'limits': ('Limits', 'Operational limits (internal)'),
}


#%% ScanOutputs Class

class ScanOutputs:
    """
    Container class for all 2D scan output matrices.
    Provides convenient access, storage, and plotting utilities.
    """
    
    def __init__(self, shape):
        """
        Initialize all output matrices with given shape.
        
        Args:
            shape: (n_param1, n_param2) dimensions of the scan grid
        """
        self.shape = shape
        self._matrices = {}
        
        # Initialize all registered parameters
        for name in OUTPUT_REGISTRY:
            self._matrices[name] = np.full(shape, np.nan)
    
    def __getitem__(self, key):
        """Get matrix by parameter name"""
        if key not in self._matrices:
            raise KeyError(f"Unknown output parameter: {key}. Available: {list(self._matrices.keys())}")
        return self._matrices[key]
    
    def __setitem__(self, key, value):
        """Set entire matrix"""
        self._matrices[key] = value
    
    def set_point(self, y, x, **kwargs):
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
    
    def fill_nan(self, y, x):
        """Fill all matrices with NaN at given point (for error cases)"""
        for matrix in self._matrices.values():
            matrix[y, x] = np.nan
    
    def get_definition(self, key):
        """Get the OutputParameter definition for a key"""
        return OUTPUT_REGISTRY.get(key)
    
    def get_masked(self, key, radial_build_matrix=None):
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
    
    def to_dict(self):
        """Export all matrices as dictionary (for backward compatibility)"""
        return self._matrices.copy()
    
    @property
    def available_parameters(self):
        """List all available parameter names"""
        return list(self._matrices.keys())
    
    @staticmethod
    def get_parameters_by_category(category):
        """Get list of parameter names for a given category"""
        return [name for name, param in OUTPUT_REGISTRY.items() 
                if param.category == category]
    
    @staticmethod
    def list_plottable_parameters():
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

def parse_scan_parameter(line):
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


def load_scan_parameters(input_file):
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



# =============================================================================
# INPUT PARAMETER REGISTRY
# =============================================================================
# Complete catalogue of scannable GlobalConfig fields.
# Each entry gives a human-readable name, physical unit, and a suggested
# [min, max, n_points] range for typical 2D scan usage.
#
# Usage in input files (bracket syntax):
#   R0 = [4.0, 10.0, 20]     → scanned parameter
#   Bmax_TF = 13.0            → fixed parameter
# =============================================================================

@dataclass
class InputParameter:
    """Metadata for a scannable GlobalConfig input parameter."""
    name         : str    # GlobalConfig field name
    display_name : str    # Human-readable description
    unit         : str    # Physical unit (empty string if dimensionless)
    min_val      : float  # Physically motivated lower bound
    max_val      : float  # Physically motivated upper bound
    n_default    : int    # Suggested grid resolution for a 2D scan
    fmt          : str = None     # Display format override (None → inferred from tick_step)
    tick_step    : float = None   # Preferred axis tick spacing in physical units
                                  # (None -> auto-computed from range)

INPUT_PARAMETER_REGISTRY = {

    # ── 1. Primary geometry ───────────────────────────────────────────────────
    'R0': InputParameter(
        name='R0', display_name='Major plasma radius',
        unit='m', min_val=0.0, max_val=40.0, n_default=20, tick_step=1.0),
    'a': InputParameter(
        name='a', display_name='Minor plasma radius',
        unit='m', min_val=0.0, max_val=10.0, n_default=20, tick_step=0.5),
    'b': InputParameter(
        name='b', display_name='Blanket + shield radial thickness',
        unit='m', min_val=0.0, max_val=5.0, n_default=10, tick_step=0.2),

    # ── 2. Magnetic field ─────────────────────────────────────────────────────
    'Bmax_TF': InputParameter(
        name='Bmax_TF', display_name='Peak field on TF conductor',
        unit='T', min_val=8.0, max_val=20.0, n_default=20, tick_step=2.0),
    'Bmax_CS_adm': InputParameter(
        name='Bmax_CS_adm', display_name='Peak field on CS conductor',
        unit='T', min_val=12.0, max_val=25.0, n_default=14, tick_step=2.0),

    # ── 3. Fusion power ───────────────────────────────────────────────────────
    'P_fus': InputParameter(
        name='P_fus', display_name='Total fusion power',
        unit='MW', min_val=200.0, max_val=4000.0, n_default=15, fmt='%.0f', tick_step=500),

    # ── 4. Core plasma physics ────────────────────────────────────────────────
    'Tbar': InputParameter(
        name='Tbar', display_name='Volume-averaged plasma temperature',
        unit='keV', min_val=8.0, max_val=22.0, n_default=20, tick_step=2.0),
    'H': InputParameter(
        name='H', display_name='H-factor (confinement enhancement)',
        unit='', min_val=0.8, max_val=1.5, n_default=15, tick_step=0.1),
    'Zeff': InputParameter(
        name='Zeff', display_name='Effective plasma charge',
        unit='', min_val=1.0, max_val=3.0, n_default=10, tick_step=0.5),
    'betaN_limit': InputParameter(
        name='betaN_limit', display_name='Troyon normalized beta limit',
        unit='% m T / MA', min_val=2.0, max_val=4.0, n_default=10, tick_step=0.5),
    'C_Alpha': InputParameter(
        name='C_Alpha', display_name='Helium ash confinement factor',
        unit='', min_val=3.0, max_val=8.0, n_default=10, tick_step=1.0),
    'rho_rad_core': InputParameter(
        name='rho_rad_core', display_name='Core/edge radiation boundary radius',
        unit='', min_val=0.5, max_val=1.0, n_default=10, tick_step=0.1),

    # ── 5. Profile peaking (Manual mode) ─────────────────────────────────────
    'nu_n': InputParameter(
        name='nu_n', display_name='Density peaking exponent (Manual profiles)',
        unit='', min_val=0.0, max_val=1.5, n_default=10, tick_step=0.25),
    'nu_T': InputParameter(
        name='nu_T', display_name='Temperature peaking exponent (Manual profiles)',
        unit='', min_val=0.5, max_val=3.5, n_default=10, tick_step=0.5),

    # ── 6. Operation and heating ──────────────────────────────────────────────
    'P_aux_input': InputParameter(
        name='P_aux_input', display_name='Auxiliary heating power (Pulsed)',
        unit='MW', min_val=10.0, max_val=200.0, n_default=15, fmt='%.0f', tick_step=25),
    'Temps_Plateau_input': InputParameter(
        name='Temps_Plateau_input', display_name='Flat-top (burn) duration (Pulsed)',
        unit='s', min_val=300.0, max_val=14400.0, n_default=10, fmt='%.0f', tick_step=2000),

    # ── 7. MHD stability limits ───────────────────────────────────────────────
    'q_limit': InputParameter(
        name='q_limit', display_name='Kink safety factor lower limit (q* > q_limit)',
        unit='', min_val=2.0, max_val=3.5, n_default=10, tick_step=0.5),
    'Greenwald_limit': InputParameter(
        name='Greenwald_limit', display_name='Greenwald density fraction limit',
        unit='', min_val=0.6, max_val=1.2, n_default=10, tick_step=0.1),

    # ── 8. Structural / TF engineering ───────────────────────────────────────
    'fatigue_CS': InputParameter(
        name='fatigue_CS', display_name='CS fatigue knockdown factor (Pulsed + Wedging)',
        unit='', min_val=1.5, max_val=3.0, n_default=10, tick_step=0.5),
    'coef_inboard_tension': InputParameter(
        name='coef_inboard_tension', display_name='TF inboard/outboard vertical stress fraction',
        unit='', min_val=0.3, max_val=0.7, n_default=10, tick_step=0.1),
    'Gap': InputParameter(
        name='Gap', display_name='Mechanical clearance CS–TF',
        unit='m', min_val=0.04, max_val=0.20, n_default=10, tick_step=0.02),
    'c_BP': InputParameter(
        name='c_BP', display_name='TF back-plate radial thickness',
        unit='m', min_val=0.02, max_val=0.15, n_default=8, tick_step=0.02),

    # ── 9. Superconductor operating conditions ────────────────────────────────
    'T_helium': InputParameter(
        name='T_helium', display_name='Helium bath temperature',
        unit='K', min_val=3.5, max_val=5.5, n_default=10, tick_step=0.5),
    'Marge_T_He': InputParameter(
        name='Marge_T_He', display_name='Temperature margin from 10-bar He operation',
        unit='K', min_val=0.1, max_val=0.6, n_default=8, tick_step=0.1),
    'Marge_T_Nb3Sn': InputParameter(
        name='Marge_T_Nb3Sn', display_name='Nb₃Sn temperature margin above T_op',
        unit='K', min_val=0.5, max_val=3.0, n_default=10, tick_step=0.5),
    'Marge_T_NbTi': InputParameter(
        name='Marge_T_NbTi', display_name='NbTi temperature margin above T_op',
        unit='K', min_val=0.5, max_val=3.0, n_default=10, tick_step=0.5),
    'Marge_T_REBCO': InputParameter(
        name='Marge_T_REBCO', display_name='REBCO temperature margin above T_op',
        unit='K', min_val=2.0, max_val=10.0, n_default=10, tick_step=1.0),
    'f_He_pipe': InputParameter(
        name='f_He_pipe', display_name='Helium pipe area fraction in CICC (without steel)',
        unit='', min_val=0.05, max_val=0.20, n_default=8, tick_step=0.05),
    'f_void': InputParameter(
        name='f_void', display_name='Interstitial void fraction in CICC strand bundle',
        unit='', min_val=0.20, max_val=0.45, n_default=8, tick_step=0.05),
    'f_In': InputParameter(
        name='f_In', display_name='Insulation area fraction in CICC cross-section',
        unit='', min_val=0.08, max_val=0.25, n_default=8, tick_step=0.05),

    # ── 10. Quench protection ─────────────────────────────────────────────────
    'I_cond': InputParameter(
        name='I_cond', display_name='Nominal conductor current',
        unit='A', min_val=20e3, max_val=100e3, n_default=8, fmt='%.0f', tick_step=10000),
    'V_max': InputParameter(
        name='V_max', display_name='Maximum dump voltage',
        unit='V', min_val=5e3, max_val=20e3, n_default=8, fmt='%.0f', tick_step=5000),
    'T_hotspot': InputParameter(
        name='T_hotspot', display_name='Maximum hot-spot temperature during quench',
        unit='K', min_val=150.0, max_val=350.0, n_default=10, fmt='%.0f', tick_step=50),
    'RRR': InputParameter(
        name='RRR', display_name='Copper residual resistivity ratio',
        unit='', min_val=50.0, max_val=300.0, n_default=8, fmt='%.0f', tick_step=50),

    # ── 11. Power conversion ──────────────────────────────────────────────────
    'eta_T': InputParameter(
        name='eta_T', display_name='Thermal-to-electric conversion efficiency',
        unit='', min_val=0.30, max_val=0.50, n_default=10, tick_step=0.05),
    'M_blanket': InputParameter(
        name='M_blanket', display_name='Blanket energy multiplication factor',
        unit='', min_val=1.0, max_val=1.4, n_default=8, tick_step=0.1),
    'eta_RF': InputParameter(
        name='eta_RF', display_name='Heating and CD wall-plug efficiency',
        unit='', min_val=0.25, max_val=0.65, n_default=10, tick_step=0.1),

    # ── 12. Multi-source CD (Steady-State) ────────────────────────────────────
    'f_heat_LH': InputParameter(
        name='f_heat_LH', display_name='LHCD power fraction (Steady-State Multi)',
        unit='', min_val=0.0, max_val=1.0, n_default=10, tick_step=0.2),
    'f_heat_EC': InputParameter(
        name='f_heat_EC', display_name='ECCD power fraction (Steady-State Multi)',
        unit='', min_val=0.0, max_val=1.0, n_default=10, tick_step=0.2),
    'f_heat_NBI': InputParameter(
        name='f_heat_NBI', display_name='NBCD power fraction (Steady-State Multi)',
        unit='', min_val=0.0, max_val=1.0, n_default=10, tick_step=0.2),
    'rho_EC': InputParameter(
        name='rho_EC', display_name='ECCD normalised deposition radius',
        unit='', min_val=0.1, max_val=0.7, n_default=10, tick_step=0.1),
    'rho_NBI': InputParameter(
        name='rho_NBI', display_name='NBCD normalised deposition radius',
        unit='', min_val=0.1, max_val=0.6, n_default=10, tick_step=0.1),
    'E_beam_keV': InputParameter(
        name='E_beam_keV', display_name='NBI beam injection energy',
        unit='keV', min_val=100.0, max_val=1000.0, n_default=10, fmt='%.0f', tick_step=100),

    # ── 13. Plasma-facing components ─────────────────────────────────────────
    'theta_deg': InputParameter(
        name='theta_deg', display_name='Divertor strike-point grazing angle',
        unit='°', min_val=0.5, max_val=6.0, n_default=10, tick_step=1.0),

    # ── 14. Ejima / flux model ────────────────────────────────────────────────
    'Ce': InputParameter(
        name='Ce', display_name='Ejima resistive ramp-up coefficient',
        unit='', min_val=0.2, max_val=0.5, n_default=10, tick_step=0.1),
    'C_PF': InputParameter(
        name='C_PF', display_name='PF coil flux scaling coefficient',
        unit='', min_val=0.7, max_val=1.1, n_default=10, tick_step=0.1),
    'E_phi_BD': InputParameter(
        name='E_phi_BD', display_name='Toroidal electric field at plasma breakdown',
        unit='V/m', min_val=0.3, max_val=1.0, n_default=8, tick_step=0.1),
    't_BD': InputParameter(
        name='t_BD', display_name='Plasma breakdown duration',
        unit='s', min_val=0.2, max_val=1.5, n_default=8, tick_step=0.25),

    # ── 15. Disruption / runaway electron indicators ──────────────────────────
    # These parameters govern the post-convergence RE indicator computation.
    # They do NOT enter the physics convergence loop.
    # A scan over pellet_dilution maps RE risk vs mitigation efficiency.
    'tau_TQ': InputParameter(
        name='tau_TQ', display_name='Thermal quench e-folding time',
        unit='s', min_val=1e-4, max_val=3e-3, n_default=10, fmt='%.2e', tick_step=5e-4),
    'Te_final_eV': InputParameter(
        name='Te_final_eV', display_name='Post-TQ residual electron temperature',
        unit='eV', min_val=2.0, max_val=20.0, n_default=10, tick_step=2.0),
    'pellet_dilution': InputParameter(
        name='pellet_dilution',
        display_name='SPI/MGI density multiplication (disruption mitigation)',
        unit='', min_val=1.0, max_val=50.0, n_default=10, tick_step=10),

    # ── 16. Techno-economic (Sheffield 2016, post-convergence) ────────────
    'discount_rate': InputParameter(
        name='discount_rate', display_name='Real discount rate',
        unit='', min_val=0.03, max_val=0.12, n_default=10, tick_step=0.02),
    'T_life': InputParameter(
        name='T_life', display_name='Plant operational lifetime',
        unit='yr', min_val=20, max_val=60, n_default=10, fmt='%.0f', tick_step=10),
    'T_build': InputParameter(
        name='T_build', display_name='Construction duration',
        unit='yr', min_val=6, max_val=15, n_default=10, fmt='%.0f', tick_step=2),
    'contingency': InputParameter(
        name='contingency', display_name='Contingency fraction',
        unit='', min_val=0.05, max_val=0.30, n_default=10, tick_step=0.05),
    'Util_factor': InputParameter(
        name='Util_factor', display_name='Utilisation factor',
        unit='', min_val=0.50, max_val=0.95, n_default=10, tick_step=0.1),
    'Dwell_factor': InputParameter(
        name='Dwell_factor', display_name='Dwell factor (1.0 = SS)',
        unit='', min_val=0.5, max_val=1.0, n_default=10, tick_step=0.1),
    'dt_rep': InputParameter(
        name='dt_rep', display_name='Scheduled replacement downtime',
        unit='yr', min_val=0.5, max_val=3.0, n_default=10, tick_step=0.5),
    'Supra_cost_factor': InputParameter(
        name='Supra_cost_factor', display_name='SC coil cost multiplier',
        unit='', min_val=1.0, max_val=4.0, n_default=10, tick_step=0.5),
    'C_invest_max': InputParameter(
        name='C_invest_max', display_name='Capital cost budget ceiling',
        unit='M EUR', min_val=5e3, max_val=50e3, n_default=10, fmt='%.0f', tick_step=5000),
}


def get_input_parameter_unit(param_name):
    """Return physical unit string for a scannable input parameter."""
    entry = INPUT_PARAMETER_REGISTRY.get(param_name)
    return entry.unit if entry else ''


def get_input_parameter_range(param_name):
    """Return suggested (min, max, n_default) scan range for a parameter."""
    entry = INPUT_PARAMETER_REGISTRY.get(param_name)
    if entry:
        return (entry.min_val, entry.max_val, entry.n_default)
    return None


# ─── Nice tick candidates for automatic step selection ────────────────────────
_NICE_STEPS = np.array([
    0.001, 0.002, 0.005,
    0.01, 0.02, 0.05,
    0.1, 0.2, 0.25, 0.5,
    1, 2, 2.5, 5,
    10, 20, 25, 50,
    100, 200, 250, 500,
    1000, 2000, 2500, 5000,
    10000, 20000, 50000,
])


def _auto_nice_step(data_range, target_nticks=8):
    """
    Choose a 'nice' tick step yielding ~target_nticks ticks.

    Parameters
    ----------
    data_range : float
        Total span of the axis (max - min).
    target_nticks : int
        Desired number of ticks (default 8).

    Returns
    -------
    float
        Rounded tick step from the _NICE_STEPS table.
    """
    raw = data_range / max(target_nticks, 1)
    idx = np.argmin(np.abs(_NICE_STEPS - raw))
    return _NICE_STEPS[idx]


def compute_physical_ticks(param_values, param_name):
    """
    Compute axis tick positions and labels aligned to physically round values.

    If *param_name* has a registered ``tick_step`` in INPUT_PARAMETER_REGISTRY
    the ticks are placed at exact multiples of that step.  Otherwise an
    automatic nice step is chosen from the data range.

    Parameters
    ----------
    param_values : ndarray
        1-D array of scanned physical values (monotonically increasing).
    param_name : str
        Name of the scanned parameter (key in INPUT_PARAMETER_REGISTRY).

    Returns
    -------
    tick_indices : ndarray of int
        Array indices into *param_values* closest to the round tick values.
    tick_labels : list of str
        Formatted label strings for each tick.
    """
    v_min, v_max = param_values[0], param_values[-1]
    data_range = v_max - v_min
    if data_range == 0:
        return np.array([0]), [f"{v_min}"]

    # --- Determine physical tick step ----------------------------------------
    entry = INPUT_PARAMETER_REGISTRY.get(param_name)
    if entry is not None and entry.tick_step is not None:
        step = entry.tick_step
    else:
        step = _auto_nice_step(data_range)

    # --- Generate round tick values spanning the data range -------------------
    first_tick = np.ceil(v_min / step) * step
    tick_values = np.arange(first_tick, v_max + step * 0.01, step)
    # Keep only values within the data range (small tolerance)
    tick_values = tick_values[(tick_values >= v_min - step * 0.01) &
                             (tick_values <= v_max + step * 0.01)]

    if len(tick_values) == 0:
        tick_values = np.array([v_min, v_max])

    # Safety: if too many ticks, double the step until manageable
    while len(tick_values) > 15:
        step *= 2
        first_tick = np.ceil(v_min / step) * step
        tick_values = np.arange(first_tick, v_max + step * 0.01, step)
        tick_values = tick_values[(tick_values >= v_min - step * 0.01) &
                                 (tick_values <= v_max + step * 0.01)]

    # --- Map physical tick values to nearest array indices --------------------
    tick_indices = np.array([np.argmin(np.abs(param_values - tv))
                            for tv in tick_values])

    # --- Format labels --------------------------------------------------------
    if step >= 100:
        fmt = "%.0f"
    elif step >= 1:
        fmt = "%.1f"   # Always one decimal: e.g. 3.0 rather than 3
    elif step >= 0.1:
        fmt = "%.1f"
    elif step >= 0.01:
        fmt = "%.2f"
    elif step >= 0.001:
        fmt = "%.3f"
    else:
        fmt = "%.2e"

    # Override with InputParameter.fmt if available
    if entry is not None and entry.fmt is not None:
        fmt = entry.fmt

    tick_labels = [fmt % tv for tv in tick_values]

    return tick_indices, tick_labels


def display_input_parameters():
    """Print all scannable input parameters grouped by theme."""
    groups = [
        ('Primary geometry',         ['R0', 'a', 'b']),
        ('Magnetic field',           ['Bmax_TF', 'Bmax_CS_adm']),
        ('Fusion power',             ['P_fus']),
        ('Core plasma physics',      ['Tbar', 'H', 'Zeff', 'betaN_limit', 'C_Alpha', 'rho_rad_core']),
        ('Profile peaking (Manual)', ['nu_n', 'nu_T']),
        ('Operation & heating',      ['P_aux_input', 'Temps_Plateau_input']),
        ('MHD limits',               ['q_limit', 'Greenwald_limit']),
        ('TF/CS engineering',        ['fatigue_CS', 'coef_inboard_tension', 'Gap', 'c_BP']),
        ('Superconductor',           ['T_helium', 'Marge_T_He', 'Marge_T_Nb3Sn',
                                      'Marge_T_NbTi', 'Marge_T_REBCO',
                                      'f_He_pipe', 'f_void', 'f_In']),
        ('Quench protection',        ['I_cond', 'V_max', 'T_hotspot', 'RRR']),
        ('Power conversion',         ['eta_T', 'M_blanket', 'eta_RF']),
        ('Multi-source CD',          ['f_heat_LH', 'f_heat_EC', 'f_heat_NBI',
                                      'rho_EC', 'rho_NBI', 'E_beam_keV']),
        ('Plasma-facing',            ['theta_deg']),
        ('Flux model',               ['Ce', 'C_PF', 'E_phi_BD', 't_BD']),
        ('Disruption / RE indicators (post-convergence)',
                                     ['tau_TQ', 'Te_final_eV', 'pellet_dilution']),
        ('Techno-economics (Sheffield, post-convergence)',
                                     ['discount_rate', 'T_life', 'T_build', 'contingency',
                                      'Util_factor', 'Dwell_factor', 'dt_rep',
                                      'Supra_cost_factor', 'C_invest_max']),
    ]
    print("\n" + "=" * 78)
    print("SCANNABLE INPUT PARAMETERS  (use bracket syntax: key = [min, max, n])")
    print("=" * 78)
    for group_name, keys in groups:
        print(f"\n  ── {group_name}")
        for k in keys:
            p = INPUT_PARAMETER_REGISTRY[k]
            rng = f"[{p.min_val}, {p.max_val}]"
            unit_str = f" [{p.unit}]" if p.unit else ""
            print(f"     {k:<22s} {p.display_name:<46s}{unit_str}  suggested: {rng}")
    print("=" * 78)




#%% Core Scan Function

def generic_2D_scan(scan_params, fixed_params, base_config, compute_re=True):
    """
    Perform generic 2D scan over any two parameters.

    Uses ``dataclasses.replace`` to build an immutable GlobalConfig for each
    grid point, consistent with the new D0FUS_run architecture.

    Parameters
    ----------
    scan_params : list of tuple
        Two entries of (name, min, max, n_points).
    fixed_params : dict
        Non-scanned parameter overrides parsed from the input file.
    base_config : GlobalConfig
        Base configuration from which each grid point is derived via
        ``dc_replace``.
    compute_re : bool, optional
        If True (default), compute runaway electron indicators at each grid
        point after the main physics chain.  Uses reduced resolution
        (N_rho=30, n_times=100) to limit overhead.  Disable for large scans
        (>30×30 points) or when RE quantities are not needed.

    Returns
    -------
    outputs : ScanOutputs
        Container with all 2D result matrices.
    param1_values, param2_values : ndarray
        Scanned parameter grids.
    param1_name, param2_name : str
        Names of the two scanned parameters.
    """
    
    # Extract scan parameters
    param1_name, param1_min, param1_max, param1_n = scan_params[0]
    param2_name, param2_min, param2_max, param2_n = scan_params[1]
    
    param1_values = np.linspace(param1_min, param1_max, param1_n)
    param2_values = np.linspace(param2_min, param2_max, param2_n)
    
    print(f"\nStarting 2D scan:")
    print(f"  {param1_name}: [{param1_min}, {param1_max}] with {param1_n} points")
    print(f"  {param2_name}: [{param2_min}, {param2_max}] with {param2_n} points")
    print(f"  Total calculations: {param1_n * param2_n}")
    print(f"  Runaway electron indicators: {'enabled (N_rho=30, n_times=100)' if compute_re else 'disabled'}\n")
    
    # Initialize outputs container
    outputs = ScanOutputs(shape=(param1_n, param2_n))

    # ── Build a single base config with all fixed overrides applied once ──
    fixed_overrides = {k: v for k, v in fixed_params.items()
                       if k in GlobalConfig.__dataclass_fields__}
    base = dc_replace(base_config, **fixed_overrides) if fixed_overrides else base_config
    
    # Scanning loop
    for y, param1_val in enumerate(tqdm(param1_values, desc=f'Scanning {param1_name}')):
        for x, param2_val in enumerate(param2_values):

            # ── Build per-point config via dc_replace (immutable) ─────────
            point_overrides = {param1_name: param1_val,
                               param2_name: param2_val}
            config = dc_replace(base, **point_overrides)

            try:
                # Run calculation (silent: verbose=0 to avoid flooding output)
                results = run(config, verbose=0)
                
                # ===========================================================
                # Unpack the full run() return tuple (v3 — 93 outputs)
                #
                # Index  Variable                     Note
                # -----  --------                     ----
                #   0    B0                            On-axis toroidal field [T]
                #   1    B_CS                          Central solenoid peak field [T]
                #   2    B_pol                         Poloidal field at LCFS [T]
                #   3    tauE                          Energy confinement time [s]
                #   4    W_th                          Thermal stored energy [J]  ← raw J
                #   5    Q                             Fusion gain
                #   6    Volume                        Plasma volume [m³]
                #   7    Surface                       First wall surface [m²]
                #   8    Ip                            Plasma current [MA]
                #   9    Ib                            Bootstrap current [MA]
                #  10    I_CD                          Non-inductive driven current [MA]
                #  11    I_Ohm                         Ohmic current [MA]
                #  12    nbar                          Volume-averaged density [10²⁰ m⁻³]
                #  13    nbar_line                     Line-averaged density [10²⁰ m⁻³]
                #  14    nG                            Greenwald density limit [10²⁰ m⁻³]
                #  15    pbar                          Volume-averaged pressure [MPa]
                #  16    betaN                         Normalized beta [% m T / MA]
                #  17    betaT                         Toroidal beta (fraction)
                #  18    betaP                         Poloidal beta
                #  19    qstar                         Kink safety factor
                #  20    q95                           Safety factor at 95% flux
                #  21    P_CD                          CD + heating power [MW]
                #  22    P_sep                         Power across separatrix [MW]
                #  23    P_Thresh                      L-H threshold power [MW]
                #  24    eta_CD                        Effective CD efficiency
                #  25    P_elec                        Net electric power [MW]
                #  26    P_wallplug                    Wall-plug power [MW]
                #  27    cost                          Machine cost proxy [m³]
                #  28    P_Brem                        Bremsstrahlung [MW]
                #  29    P_syn                         Synchrotron [MW]
                #  30    P_line                        Total line radiation [MW]
                #  31    P_line_core                   Core line radiation [MW]
                #  32    heat                          P_sep/R0 [MW/m]
                #  33    heat_par                      P_sep·B0/R0 [MW·T/m]
                #  34    heat_pol                      P_sep·B0/(q95·R0·A) [MW·T/m]
                #  35    lambda_q                      Eich SOL width [m]
                #  36    q_target                      Peak divertor heat flux [MW/m²]
                #  37    P_wall_H                      First-wall load H-mode [MW/m²]
                #  38    P_wall_L                      First-wall load L-mode [MW/m²]
                #  39    Gamma_n                       Neutron wall loading [MW/m²]
                #  40    f_alpha                       Helium ash fraction
                #  41    tau_alpha                     Alpha confinement time [s]
                #  42    J_TF                          TF cable current density [A/m²]
                #  43    J_CS                          CS cable current density [A/m²]
                #  44–50 TF radial build + stresses
                #  51–57 CS radial build + stresses
                #  58–61 r_minor, r_sep, r_c, r_d
                #  62–65 κ, κ_95, δ, δ_95
                #  66–70 ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS
                #  71    Vloop (steady-state loop voltage [V])
                #  72    li    (internal inductance li(3) [-])
                #  73–75 eta_LH, eta_EC, eta_NBI
                #  76–79 P_LH, P_EC, P_NBI, P_ICR
                #  80–82 I_LH, I_EC, I_NBI
                #  83–88 f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF
                #  89–94 f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS
                # ===========================================================
                (B0, B_CS, B_pol,
                 tauE, W_th,
                 Q, Volume, Surface,
                 Ip, Ib, I_CD, I_Ohm,
                 nbar, nbar_line, nG, pbar,
                 betaN, betaT, betaP,
                 qstar, q95,
                 P_CD, P_sep, P_Thresh, eta_CD, P_elec, P_wallplug,
                 cost, P_Brem, P_syn, P_line, P_line_core,
                 heat, heat_par, heat_pol, lambda_q, q_target,
                 P_wall_H, P_wall_L,
                 Gamma_n,
                 f_alpha, tau_alpha,
                 J_TF, J_CS,
                 c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
                 d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS_out, J_CS_out,
                 r_minor, r_sep, r_c, r_d,
                 κ, κ_95, δ, δ_95,
                 ΨPI, ΨRampUp, Ψplateau, ΨPF, ΨCS, Vloop_sc, li_sc,
                 eta_LH, eta_EC, eta_NBI,
                 P_LH, P_EC, P_NBI, P_ICR,
                 I_LH, I_EC, I_NBI,
                 f_sc_TF, f_cu_TF, f_He_pipe_TF, f_void_TF, f_He_TF, f_In_TF,
                 f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS) = results
                
                # ── Plasma stability limits ──────────────────────────────
                betaN_limit_value = config.betaN_limit
                q_limit_value     = config.q_limit

                # Greenwald limit is defined in line-averaged density:
                # n_condition must compare nbar_line (not nbar_vol) to n_G.
                n_condition    = nbar_line / nG      if nG > 0       else np.nan
                beta_condition = betaN / betaN_limit_value
                q_condition    = q_limit_value / qstar
                
                max_limit = max(n_condition, beta_condition, q_condition)

                # ── Sheffield cost model (post-convergence) ───────────
                _COE_val = np.nan
                _C_invest_val = np.nan
                if config.cost_model != 'None' and np.isfinite(cost):
                    try:
                        P_th_scan = config.P_fus * config.M_blanket + P_CD
                        (V_BB_s, V_TF_s, V_CS_s, V_FI_s) = f_volume(
                            config.a, config.b, c, d, config.R0, κ)
                        _cres = f_costs_Sheffield(
                            discount_rate=config.discount_rate,
                            contingency=config.contingency,
                            T_life=config.T_life,
                            T_build=config.T_build,
                            P_t=P_th_scan,
                            P_e=max(P_elec, 1.0),
                            P_aux=P_CD,
                            Gamma_n=Gamma_n,
                            Util_factor=config.Util_factor,
                            Dwell_factor=config.Dwell_factor,
                            dt_rep=config.dt_rep,
                            V_FI=V_FI_s,
                            V_pc=V_TF_s + V_CS_s,
                            V_sg=V_BB_s,
                            V_bl=V_BB_s,
                            S_tt=0.1 * Surface,
                            Supra_cost_factor=config.Supra_cost_factor,
                        )
                        _COE_val = _cres[3]
                        _C_invest_val = _cres[2] * 1e-3  # M EUR -> B EUR
                    except Exception:
                        pass

                # Store all results using set_point
                outputs.set_point(y, x,
                    # Performance
                    Q=Q,
                    P_fus=config.P_fus,
                    P_elec=P_elec,
                    Cost=cost,
                    COE=_COE_val,
                    C_invest=_C_invest_val,
                    
                    # Plasma parameters
                    Ip=Ip,
                    Ib=Ib,
                    I_CD=I_CD,
                    I_Ohm=I_Ohm,
                    n=nbar_line,         # Line-averaged density (consistent with scaling law label)
                    nbar_vol=nbar,       # Volume-averaged density
                    pbar=pbar,
                    beta_N=betaN,
                    beta_T=betaT,
                    beta_P=betaP,
                    q95=q95,
                    qstar=qstar,
                    tauE=tauE,
                    W_th=W_th * 1e-6,   # [J] → [MJ]  (registry unit is MJ)
                    f_bs=Ib/Ip * 100 if Ip > 0 else 0,
                    f_alpha=f_alpha * 100,
                    tau_alpha=tau_alpha,
                    
                    # Magnetic field
                    B0=B0,
                    BCS=B_CS,
                    B_pol=B_pol,
                    J_TF=J_TF * 1e-6,   # [A/m²] → [A/mm²]
                    J_CS=J_CS * 1e-6,   # [A/m²] → [A/mm²]
                    PsiCS=ΨCS,
                    
                    # Power & heat
                    Heat=heat,
                    Gamma_n=Gamma_n,
                    P_CD=P_CD,
                    P_sep=P_sep,
                    P_Thresh=P_Thresh,
                    L_H=P_sep / P_Thresh if P_Thresh > 0 else np.nan,
                    P_Brem=P_Brem,
                    P_syn=P_syn,
                    P_line=P_line,
                    P_wallplug=P_wallplug,
                    q_target=q_target,
                    lambda_q=lambda_q * 1000 if lambda_q else np.nan,  # [m] → [mm]
                    
                    # Per-source CD
                    eta_LH=eta_LH,
                    eta_EC=eta_EC,
                    eta_NBI=eta_NBI,
                    
                    # Geometry
                    c=r_sep - r_c if not np.isnan(r_c) else np.nan,
                    d=r_c - r_d   if not np.isnan(r_d) else np.nan,
                    r_minor=r_minor,
                    kappa=κ,
                    kappa_95=κ_95,
                    delta=δ,
                    Volume=Volume,
                    Surface=Surface,
                    A=config.R0 / config.a if config.a > 0 else np.nan,
                    
                    # Structural
                    sigma_TF=max(abs(σ_z_TF), abs(σ_theta_TF), abs(σ_r_TF)) if σ_z_TF else np.nan,
                    sigma_CS=max(abs(σ_z_CS), abs(σ_theta_CS), abs(σ_r_CS)) if σ_z_CS else np.nan,
                    Steel_fraction_TF=Steel_fraction_TF * 100 if Steel_fraction_TF else np.nan,
                    Steel_fraction_CS=Steel_fraction_CS * 100 if Steel_fraction_CS else np.nan,
                    f_sc_TF=f_sc_TF * 100 if np.isfinite(f_sc_TF) else np.nan,
                    f_He_TF=f_He_TF * 100 if np.isfinite(f_He_TF) else np.nan,
                    f_sc_CS=f_sc_CS * 100 if np.isfinite(f_sc_CS) else np.nan,
                    f_He_CS=f_He_CS * 100 if np.isfinite(f_He_CS) else np.nan,
                    
                    # Limits
                    limits=max_limit,
                    density_limit=n_condition,
                    beta_limit=beta_condition,
                    q_limit=q_condition,
                )
                
                # Combined TF + CS thickness
                c_val = r_sep - r_c if not np.isnan(r_c) else np.nan
                d_val = r_c  - r_d  if not np.isnan(r_d) else np.nan
                outputs['c_d'][y, x] = (c_val + d_val
                                        if not (np.isnan(c_val) or np.isnan(d_val))
                                        else np.nan)
                
                # Check radial build validity
                if not np.isnan(r_d) and max_limit < 1 and r_d > 0:
                    outputs['radial_build'][y, x] = config.R0
                else:
                    outputs['radial_build'][y, x] = np.nan

                # ── Runaway electron indicators ──────────────────────────────
                # Computed from the converged plasma state using the hot-tail
                # seed model (Smith 2008) + avalanche amplification (Breizman
                # 2019). Uses li from the run() output tuple (self-consistent).
                # N_rho and n_times are reduced from defaults for scan performance.
                if compute_re:
                    # Resolve profile peaking factors — mirrors the _PROFILE_PRESETS
                    # logic in run() so that nu_T, nu_n, rho_ped, n_ped_frac, and
                    # T_ped_frac are defined for the RE indicator calls below.
                    # Without this block these names are undefined → NameError.
                    if config.Plasma_profiles == 'Manual':
                        nu_n       = config.nu_n_manual
                        nu_T       = config.nu_T_manual
                        rho_ped    = config.rho_ped
                        n_ped_frac = config.n_ped_frac
                        T_ped_frac = config.T_ped_frac
                    else:
                        _pp        = _PROFILE_PRESETS.get(config.Plasma_profiles,
                                                           _PROFILE_PRESETS['H'])
                        nu_n       = _pp['nu_n'];       nu_T       = _pp['nu_T']
                        rho_ped    = _pp['rho_ped'];    n_ped_frac = _pp['n_ped_frac']
                        T_ped_frac = _pp['T_ped_frac']

                    try:
                        # li is available directly from run() output tuple
                        # (self-consistent li(3) from f_q_profile_selfconsistent).
                        _re = compute_RE_indicators(
                            Ip=Ip, nbar=nbar, Tbar=config.Tbar,
                            a=config.a, R0=config.R0, κ=κ,
                            Z_eff=config.Zeff, li=li_sc,
                            nu_n=nu_n, nu_T=nu_T,
                            rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                            T_ped_frac=T_ped_frac,
                            Te_final_eV=config.Te_final_eV,
                            tau_TQ=config.tau_TQ,
                            V=Volume,
                            N_rho=30,    # Reduced from default 50 for scan performance
                            n_times=100, # Reduced from default 300 for scan performance
                            pellet_dilution=config.pellet_dilution,
                        )
                        outputs['I_RE_seed'][y, x]  = _re['I_RE_seed'] * 1e-3        # [A] → [kA]
                        outputs['I_RE_aval'][y, x]  = _re['I_RE_avalanche'] * 1e-6  # [A] → [MA]
                        outputs['f_RE_Ip'][y, x]    = _re['f_RE_to_Ip']     * 100   # [-] → [%]
                        outputs['f_RE_avg'][y, x]   = _re['f_RE_avg']
                        outputs['f_RE_core'][y, x]  = _re['f_RE_core']
                        outputs['E_RE_kin'][y, x]   = _re['E_RE_kin']               # [MJ]
                        outputs['W_mag_RE'][y, x]   = _re['W_mag_RE']               # [MJ]
                    except Exception:
                        # RE computation is non-critical — leave as NaN on failure
                        pass

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
        params_str = " | ".join(params)
        print(f"\n  {cat_name}: {params_str}")
    
    print("\n" + "="*70)


def get_user_plot_choice(prompt, valid_options):
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


def plot_generic_contours(ax, matrix, param_key, 
                          color='black', linestyle='dashed',
                          linewidth=2.5, fontsize=22):
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


def plot_scan_results(outputs, param1_values, param2_values,
                      param1_name, param2_name, config, output_dir,
                      iso_param_1=None, iso_param_2=None):
    """
    Generate scan visualization with two user-selectable iso-contour parameters.

    Parameters
    ----------
    outputs       : ScanOutputs   All result matrices.
    param1_values : ndarray       First scan parameter grid (Y-axis).
    param2_values : ndarray       Second scan parameter grid (X-axis).
    param1_name   : str           Name of first scan parameter.
    param2_name   : str           Name of second scan parameter.
    config        : GlobalConfig  Reference configuration (for default values).
    output_dir    : str or None   Unused, kept for API compatibility.
    iso_param_1   : str or None   Pre-selected first iso-contour parameter.
    iso_param_2   : str or None   Pre-selected second iso-contour parameter.

    Returns
    -------
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
    
    # Configure Y-axis ticks (param1) — physically round values
    y_tick_idx, y_tick_labels = compute_physical_ticks(param1_values, param1_name)
    # Invert indices for the [::-1] convention used by imshow
    y_tick_idx_inv = (len(param1_values) - 1) - y_tick_idx
    ax.set_yticks(y_tick_idx_inv)
    ax.set_yticklabels(y_tick_labels[::-1], fontsize=font_legend)
    
    # Configure X-axis ticks (param2) — physically round values
    x_tick_idx, x_tick_labels = compute_physical_ticks(param2_values, param2_name)
    ax.set_xticks(x_tick_idx)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=font_legend)
    
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

def save_scan_results(fig, outputs, param1_values, param2_values,
                     param1_name, param2_name, config, output_dir,
                     iso_param_1, iso_param_2, input_file_path=None):
    """
    Save scan results to timestamped directory.

    Parameters
    ----------
    config : GlobalConfig
        Reference configuration (used to dump fixed parameters).

    Returns
    -------
    str : Path to output directory.
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
        # Generate input file from configuration
        input_copy = os.path.join(output_path, "scan_parameters.txt")
        config_dict = asdict(config)
        with open(input_copy, "w", encoding='utf-8') as f:
            f.write("# D0FUS Scan Parameters\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("# Scan parameters:\n")
            f.write(f"{param1_name} = [{param1_values[0]:.2f}, {param1_values[-1]:.2f}, {len(param1_values)}]\n")
            f.write(f"{param2_name} = [{param2_values[0]:.2f}, {param2_values[-1]:.2f}, {len(param2_values)}]\n")
            f.write("\n# Fixed parameters:\n")
            for key, value in config_dict.items():
                if key not in (param1_name, param2_name):
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

def main(input_file=None, auto_plot=False,
         iso_param_1=None, iso_param_2=None,
         compute_re=True):
    """
    Main execution function for scans.

    Parameters
    ----------
    input_file   : str or None   Path to input file.
    auto_plot    : bool           If True, use provided iso parameters without prompting.
    iso_param_1  : str or None    First iso-contour parameter (black lines).
    iso_param_2  : str or None    Second iso-contour parameter (white lines).
    compute_re   : bool           If True (default), compute runaway electron indicators.
                                  Set to False to speed up large scans.
                                  Can also be overridden by ``compute_re = 0`` in the input file.
    """
    
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

    # ── Build base config from fixed parameters in the input file ────────
    # load_config_from_file skips bracketed scan declarations and returns a
    # GlobalConfig with only the scalar overrides applied.
    base_config = load_config_from_file(input_file, verbose=2)
    
    # Load scan and fixed parameters (for grid definition and display)
    scan_params, fixed_params = load_scan_parameters(input_file)

    # Allow input file to override compute_re flag (e.g. "compute_re = 0")
    _TRUTHY = {'1', 'true', 'yes', 'on'}
    _FALSY  = {'0', 'false', 'no', 'off'}
    try:
        with open(input_file, 'r', encoding='utf-8') as _fh:
            for _line in _fh:
                _line = _line.split('#')[0].strip()
                if not _line or '=' not in _line:
                    continue
                _k, _, _v = _line.partition('=')
                if _k.strip().lower() == 'compute_re':
                    _v = _v.strip().lower()
                    if _v in _TRUTHY:
                        compute_re = True
                    elif _v in _FALSY:
                        compute_re = False
    except OSError:
        pass
    
    # Print scan configuration
    print("\n" + "="*73)
    print("Starting D0FUS 2D parameter scan...")
    print("="*73)
    print(f"\nScan parameters:")
    for param_name, min_val, max_val, n_points in scan_params:
        entry = INPUT_PARAMETER_REGISTRY.get(param_name)
        unit_str = f" [{entry.unit}]" if (entry and entry.unit) else ""
        desc_str = f"  ({entry.display_name})" if entry else ""
        print(f"  {param_name}: [{min_val}, {max_val}]{unit_str} × {n_points} pts{desc_str}")
    
    print(f"\nFixed parameters:")
    for key, value in list(fixed_params.items())[:6]:
        entry = INPUT_PARAMETER_REGISTRY.get(key)
        unit_str = f" [{entry.unit}]" if (entry and entry.unit) else ""
        print(f"  {key} = {value}{unit_str}")
    if len(fixed_params) > 6:
        print(f"  ... and {len(fixed_params) - 6} more")
    
    try:
        # Perform scan
        outputs, param1_values, param2_values, param1_name, param2_name = generic_2D_scan(
            scan_params, fixed_params, base_config, compute_re=compute_re
        )
        
        # Plot results
        if auto_plot and iso_param_1 and iso_param_2:
            fig, ax, iso_used_1, iso_used_2 = plot_scan_results(
                outputs, param1_values, param2_values, param1_name, param2_name,
                base_config, None, iso_param_1, iso_param_2
            )
        else:
            fig, ax, iso_used_1, iso_used_2 = plot_scan_results(
                outputs, param1_values, param2_values, param1_name, param2_name,
                base_config, None
            )
        
        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        output_path = save_scan_results(
            fig, outputs, param1_values, param2_values, param1_name, param2_name,
            base_config, output_dir, iso_used_1, iso_used_2, input_file_path
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