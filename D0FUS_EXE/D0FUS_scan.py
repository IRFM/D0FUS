"""
D0FUS Scan Module
=================
Generates 2D parameter space maps with full visualization.
Supports scanning any 2 parameters dynamically.
Allows user to choose any two output parameters for iso-contours visualization.

Adapted to the GlobalConfig / dc_replace architecture of D0FUS_run (v2).
"""
#%% Imports
# Bootstrap imports only: sys/os are required to make the package importable
# before D0FUS_import.py (which centralises every other import) is reachable.
import sys
import os

# joblib is always installed in Spyder/scientific Python environments.
# It uses loky+cloudpickle internally, which serializes functions by bytecode
# rather than by module reference — making parallel execution robust to module
# reloads within the same session (e.g. repeated %runfile in Spyder/IPython).

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_BIB.D0FUS_cost_data import *
from D0FUS_EXE.D0FUS_run import (load_config_from_file, evaluate_costs,
                                 evaluate_feasibility, new_output_dir,
                                 default_output_root)
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from D0FUS_BIB.D0FUS_parameterization import (GlobalConfig, DEFAULT_CONFIG, coerce_input_value,
                                               resolve_deprecated_key, apply_legacy_split_keys)

# Deck keys read by the SCAN driver itself (not GlobalConfig fields).
#   iso_param_1, iso_param_2 : iso-contour layers of the map (registry key or none)
#   compute_re               : runaway-electron indicators at each point (bool)
#   clip_to_viable           : restrict iso-contours to the feasible region (bool)
#   n_workers                : worker processes (default: one per 8 grid
#                              points, at most one per CPU)
SCAN_SETTING_KEYS = ('iso_param_1', 'iso_param_2', 'compute_re',
                     'clip_to_viable', 'n_workers')

# Backwards-compatible alias for dataclasses.replace (from D0FUS_import).
dc_replace = replace

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
        label='$(V_{BB}+V_{TF}+V_{CS})/P_{fus}$',
        unit='m$^3$/MW',
        levels=(0, 20, 0.5),
        fmt='%.1f',
        use_radial_mask=False,
        category='performance',
        description='Geometric cost proxy, build volume per unit fusion power'
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
        unit='% m T/MA',
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
        levels=(0, 20, 1),
        fmt='%d',
        use_radial_mask=False,
        category='plasma',
        description='Helium ash fraction n_He/n_e'
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
        levels=(0, 30, 2),
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
        label='$P_{sep}/R_0$',
        unit='MW/m',
        levels=(0, 60, 2),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Divertor heat scaling P_sep/R0'
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
        description='Heating and current-drive power coupled to the plasma'
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
        levels=(0, 2000, 25),
        fmt='%d',
        use_radial_mask=False,
        category='power',
        description='Unmitigated attached peak target heat flux (Eich lambda_q)'
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
        label='$|\\sigma|_{max,TF}$',
        unit='MPa',
        levels=(0, 1000, 50),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='TF coil largest principal stress in absolute value'
    ),
    'sigma_CS': OutputParameter(
        name='sigma_CS',
        label='$|\\sigma|_{max,CS}$',
        unit='MPa',
        levels=(0, 1000, 50),
        fmt='%d',
        use_radial_mask=True,
        category='structural',
        description='CS coil largest principal stress in absolute value'
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
        description='Impurity radiation, Mavrin L_z (line + recombination + bremsstrahlung)'
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
        description='He ash particle confinement time tau*_He = C_Alpha tau_E'
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
        description='Hot-tail RE seed current at the post-pellet density (Smith 2008), stored in kA'
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
        label='$q_{lim}/q_{kink}$',
        unit='',
        levels=(0.5, 2, 0.1),
        fmt='%.2f',
        use_radial_mask=False,
        category='limits',
        description='Safety factor limit fraction (q* or q95)'
    ),
    'radial_build': OutputParameter(
        name='radial_build',
        label='Radial build',
        unit='',
        levels=None,
        fmt='%.1f',
        use_radial_mask=False,
        category='limits',
        description='Feasibility flag (R0 where the radial build closes and the plasma limits hold, NaN elsewhere)'
    ),

    # -------------------------------------------------------------------------
    # RADIAL BUILD COMPONENT VOLUMES
    # -------------------------------------------------------------------------
    'V_rb_SOL': OutputParameter(
        name='V_rb_SOL',
        label='$V_{\\mathrm{SOL}}$',
        unit='m³',
        levels=(0, 200, 20),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='SOL / far-SOL volume (all around)'
    ),
    'V_rb_FW': OutputParameter(
        name='V_rb_FW',
        label='$V_{\\mathrm{FW}}$',
        unit='m³',
        levels=(0, 500, 50),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='First wall volume (excl. divertor region)'
    ),
    'V_rb_BB': OutputParameter(
        name='V_rb_BB',
        label='$V_{\\mathrm{BB}}$',
        unit='m³',
        levels=(0, 3000, 200),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='Breeding blanket volume (excl. divertor region)'
    ),
    'V_rb_shield': OutputParameter(
        name='V_rb_shield',
        label='$V_{\\mathrm{shield}}$',
        unit='m³',
        levels=(0, 2000, 200),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='Neutron shield volume (all around)'
    ),
    'V_rb_VV': OutputParameter(
        name='V_rb_VV',
        label='$V_{\\mathrm{VV}}$',
        unit='m³',
        levels=(0, 1000, 100),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='Vacuum vessel volume (all around)'
    ),
    'V_rb_gap_TF': OutputParameter(
        name='V_rb_gap_TF',
        label='$V_{\\mathrm{gap,TF}}$',
        unit='m³',
        levels=(0, 500, 50),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='VV→TF gap volume (all around)'
    ),
    'V_rb_divertor': OutputParameter(
        name='V_rb_divertor',
        label='$V_{\\mathrm{div}}$',
        unit='m³',
        levels=(0, 200, 20),
        fmt='%.0f',
        use_radial_mask=True,
        category='radial_build',
        description='Divertor volume'
    ),

    # -------------------------------------------------------------------------
    # RADIAL BUILD COMPONENT MASSES
    # -------------------------------------------------------------------------
    'M_rb_FW': OutputParameter(
        name='M_rb_FW',
        label='$M_{\\mathrm{FW}}$',
        unit='t',
        levels=(0, 5000, 500),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='First wall mass (excl. divertor region)'
    ),
    'M_rb_BB': OutputParameter(
        name='M_rb_BB',
        label='$M_{\\mathrm{BB}}$',
        unit='t',
        levels=(0, 20000, 2000),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Breeding blanket mass (excl. divertor region)'
    ),
    'M_rb_shield': OutputParameter(
        name='M_rb_shield',
        label='$M_{\\mathrm{shield}}$',
        unit='t',
        levels=(0, 20000, 2000),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Neutron shield mass'
    ),
    'M_rb_VV': OutputParameter(
        name='M_rb_VV',
        label='$M_{\\mathrm{VV}}$',
        unit='t',
        levels=(0, 10000, 1000),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Vacuum vessel mass'
    ),
    'M_rb_divertor': OutputParameter(
        name='M_rb_divertor',
        label='$M_{\\mathrm{div}}$',
        unit='t',
        levels=(0, 2000, 200),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Divertor mass'
    ),
    'M_rb_total': OutputParameter(
        name='M_rb_total',
        label='$M_{\\mathrm{rb}}^{\\mathrm{tot}}$',
        unit='t',
        levels=(0, 50000, 5000),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Total radial build mass (FW + BB + shield + VV + divertor)'
    ),

    # -------------------------------------------------------------------------
    # BLANKET & COIL MASSES
    # -------------------------------------------------------------------------
    'V_blanket': OutputParameter(
        name='V_blanket',
        label='$V_{\\mathrm{bl}}$',
        unit='m³',
        levels=(0, 2000, 100),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Blanket volume (Miller LCFS → TF inner face)'
    ),
    'M_total_TF': OutputParameter(
        name='M_total_TF',
        label='$M_{\\mathrm{TF}}^{\\mathrm{tot}}$',
        unit='t',
        levels=(0, 5000, 250),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Total TF coil mass (all coils, all materials)'
    ),
    'M_sc_TF': OutputParameter(
        name='M_sc_TF',
        label='$M_{\\mathrm{TF}}^{\\mathrm{SC}}$',
        unit='t',
        levels=(0, 500, 25),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='TF superconductor mass (all coils)'
    ),
    'M_steel_TF': OutputParameter(
        name='M_steel_TF',
        label='$M_{\\mathrm{TF}}^{\\mathrm{steel}}$',
        unit='t',
        levels=(0, 3000, 200),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='TF structural steel mass (all coils)'
    ),
    'M_cu_TF': OutputParameter(
        name='M_cu_TF',
        label='$M_{\\mathrm{TF}}^{\\mathrm{Cu}}$',
        unit='t',
        levels=(0, 500, 25),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='TF copper stabiliser mass (all coils)'
    ),
    'M_total_CS': OutputParameter(
        name='M_total_CS',
        label='$M_{\\mathrm{CS}}^{\\mathrm{tot}}$',
        unit='t',
        levels=(0, 1000, 50),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='Total CS mass'
    ),
    'M_sc_CS': OutputParameter(
        name='M_sc_CS',
        label='$M_{\\mathrm{CS}}^{\\mathrm{SC}}$',
        unit='t',
        levels=(0, 200, 20),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='CS superconductor mass'
    ),
    'M_steel_CS': OutputParameter(
        name='M_steel_CS',
        label='$M_{\\mathrm{CS}}^{\\mathrm{steel}}$',
        unit='t',
        levels=(0, 600, 50),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='CS structural steel mass'
    ),
    'M_cu_CS': OutputParameter(
        name='M_cu_CS',
        label='$M_{\\mathrm{CS}}^{\\mathrm{Cu}}$',
        unit='t',
        levels=(0, 200, 20),
        fmt='%.0f',
        use_radial_mask=True,
        category='materials',
        description='CS copper stabiliser mass'
    ),

    # -------------------------------------------------------------------------
    # CONDUCTOR LENGTHS
    # -------------------------------------------------------------------------
    'L_cable_TF': OutputParameter(
        name='L_cable_TF',
        label='$L_{\\mathrm{cable}}^{\\mathrm{TF}}$',
        unit='km',
        levels=(0, 500, 50),
        fmt='%.0f',
        use_radial_mask=True,
        category='conductor',
        description='Total TF cable conductor length (all coils)'
    ),
    'L_cable_CS': OutputParameter(
        name='L_cable_CS',
        label='$L_{\\mathrm{cable}}^{\\mathrm{CS}}$',
        unit='km',
        levels=(0, 100, 10),
        fmt='%.0f',
        use_radial_mask=True,
        category='conductor',
        description='Total CS cable conductor length'
    ),
    'L_sc_strand_TF': OutputParameter(
        name='L_sc_strand_TF',
        label='$L_{\\mathrm{SC}}^{\\mathrm{TF}}$',
        unit='km',
        levels=(0, 2000, 200),
        fmt='%.0f',
        use_radial_mask=True,
        category='conductor',
        description='Total TF SC strand length (all coils)'
    ),
    'L_sc_strand_CS': OutputParameter(
        name='L_sc_strand_CS',
        label='$L_{\\mathrm{SC}}^{\\mathrm{CS}}$',
        unit='km',
        levels=(0, 500, 50),
        fmt='%.0f',
        use_radial_mask=True,
        category='conductor',
        description='Total CS SC strand length'
    ),

    # -------------------------------------------------------------------------
    # COMPONENT LIFETIMES & AVAILABILITY
    # -------------------------------------------------------------------------
    't_blanket_fpy': OutputParameter(
        name='t_blanket_fpy',
        label='$t_{\\mathrm{bl}}$',
        unit='fpy',
        levels=(0, 15, 1),
        fmt='%.1f',
        use_radial_mask=True,
        category='availability',
        description='Blanket structural lifetime (dpa model)'
    ),
    't_div_fpy': OutputParameter(
        name='t_div_fpy',
        label='$t_{\\mathrm{div}}$',
        unit='fpy',
        levels=(0, 10, 1),
        fmt='%.1f',
        use_radial_mask=True,
        category='availability',
        description='Divertor lifetime (integrated heat model)'
    ),
    't_blanket_yr': OutputParameter(
        name='t_blanket_yr',
        label='$t_{\\mathrm{bl}}$',
        unit='yr',
        levels=(0, 20, 2),
        fmt='%.1f',
        use_radial_mask=True,
        category='availability',
        description='Blanket calendar lifetime'
    ),
    't_div_yr': OutputParameter(
        name='t_div_yr',
        label='$t_{\\mathrm{div}}$',
        unit='yr',
        levels=(0, 15, 1),
        fmt='%.1f',
        use_radial_mask=True,
        category='availability',
        description='Divertor calendar lifetime'
    ),
    'T_op_limit': OutputParameter(
        name='T_op_limit',
        label='$T_{\\mathrm{op}}$',
        unit='yr',
        levels=(0, 15, 1),
        fmt='%.1f',
        use_radial_mask=True,
        category='availability',
        description='Calendar operation time per replacement cycle'
    ),
    'dt_rep_eff': OutputParameter(
        name='dt_rep_eff',
        label='$\\Delta t_{\\mathrm{rep}}$',
        unit='yr',
        levels=(0, 3, 0.2),
        fmt='%.2f',
        use_radial_mask=True,
        category='availability',
        description='Effective average replacement downtime per cycle'
    ),
    'Av': OutputParameter(
        name='Av',
        label='$A_v$',
        unit='',
        levels=(0.5, 1.0, 0.05),
        fmt='%.2f',
        use_radial_mask=True,
        category='availability',
        description='Plant availability'
    ),
    'CF': OutputParameter(
        name='CF',
        label='$\\mathrm{CF}$',
        unit='',
        levels=(0.3, 0.9, 0.05),
        fmt='%.2f',
        use_radial_mask=True,
        category='availability',
        description='Plant capacity factor'
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
    'availability': ('Availability', 'Component lifetimes, replacement schedule, availability and capacity factor'),
    'radial_build': ('Radial Build Volumes', 'Component volumes: plasma gap, FW, breeding blanket, shield, VV, divertor'),
    'materials': ('Masses & Volumes', 'Blanket volume and TF/CS coil masses per material'),
    'conductor': ('Conductor Lengths', 'TF and CS cable and SC strand total lengths'),
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
                name = resolve_deprecated_key(scan_param[0])
                if name not in GlobalConfig.__dataclass_fields__:
                    raise ValueError(f"Scan parameter '{name}' is not a "
                                     "GlobalConfig field.")
                scan_params.append((name,) + scan_param[1:])
                scan_param_names.append(name)
                continue
            
            if '=' in line:
                parts = line.split('=', 1)
                param_name = resolve_deprecated_key(parts[0].strip())
                param_value = parts[1].strip()
                
                if param_name in SCAN_SETTING_KEYS:
                    continue
                
                # Type-aware coercion shared with the run / genetic loaders.
                fixed_params[param_name] = coerce_input_value(
                    param_name, param_value)
    
    if len(scan_params) != 2:
        raise ValueError(f"Expected exactly 2 scan parameters, found {len(scan_params)}")
    if scan_param_names[0] == scan_param_names[1]:
        raise ValueError(f"Both scan axes use '{scan_param_names[0]}'.")
    
    # A scanned key also given as a scalar keeps its scan range
    for name in scan_param_names:
        fixed_params.pop(name, None)

    # Old single keys split per coil system (Chosen_Steel, σ_manual)
    fixed_params = apply_legacy_split_keys(fixed_params)

    return scan_params, fixed_params


def load_scan_settings(input_file):
    """
    SCAN driver settings written in the deck (see SCAN_SETTING_KEYS).

    Returns
    -------
    dict  Only the keys present in the deck. iso_param_* are kept as
          strings, compute_re and clip_to_viable as bool, n_workers as int.
    """
    settings = {}
    with open(input_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.split('#')[0].strip()
            if '=' not in line:
                continue
            key, _, val = (t.strip() for t in line.partition('='))
            if key not in SCAN_SETTING_KEYS:
                continue
            if key in ('compute_re', 'clip_to_viable'):
                low = val.lower()
                if low not in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
                    raise ValueError(f"{key} = {val}: expected true or false.")
                settings[key] = low in ('1', 'true', 'yes', 'on')
            elif key == 'n_workers':
                settings[key] = max(1, int(float(val)))
            else:
                settings[key] = val
    return settings



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
        name='Tbar', display_name='Volume-averaged electron temperature',
        unit='keV', min_val=8.0, max_val=22.0, n_default=20, tick_step=2.0),
    'tau_i_e': InputParameter(
        name='tau_i_e', display_name='Ion/electron temperature ratio Ti/Te',
        unit='', min_val=0.7, max_val=1.5, n_default=10, tick_step=0.1),
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
    'nu_n_manual': InputParameter(
        name='nu_n_manual', display_name='Density peaking exponent (Manual profiles)',
        unit='', min_val=0.0, max_val=1.5, n_default=10, tick_step=0.25),
    'nu_T_manual': InputParameter(
        name='nu_T_manual', display_name='Temperature peaking exponent (Manual profiles)',
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
        name='q_limit', display_name='Kink safety factor lower limit (q_kink > q_limit)',
        unit='', min_val=2.0, max_val=3.5, n_default=10, tick_step=0.5),
    'Greenwald_limit': InputParameter(
        name='Greenwald_limit', display_name='Greenwald density fraction limit',
        unit='', min_val=0.6, max_val=1.2, n_default=10, tick_step=0.1),

    # ── 8. Structural / TF engineering ───────────────────────────────────────
    'fatigue_CS': InputParameter(
        name='fatigue_CS', display_name='CS fatigue knockdown factor (Pulsed + Wedging)',
        unit='', min_val=1.5, max_val=3.0, n_default=10, tick_step=0.5),
    'SF_TF': InputParameter(
        name='SF_TF', display_name='TF steel allowable safety factor',
        unit='', min_val=1.0, max_val=2.0, n_default=6, tick_step=0.2),
    'SF_CS': InputParameter(
        name='SF_CS', display_name='CS steel allowable safety factor',
        unit='', min_val=1.0, max_val=2.0, n_default=6, tick_step=0.2),
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
    'f_In_cable': InputParameter(
        name='f_In_cable',
        display_name='Turn insulation area fraction in CICC cross-section',
        unit='', min_val=0.02, max_val=0.15, n_default=8, tick_step=0.02),
    'f_In_WP': InputParameter(
        name='f_In_WP',
        display_name='Winding-pack insulation and clearance fraction',
        unit='', min_val=0.05, max_val=0.30, n_default=8, tick_step=0.05),

    # ── 10. Quench protection ─────────────────────────────────────────────────
    'I_cond': InputParameter(
        name='I_cond', display_name='Nominal TF conductor current',
        unit='A', min_val=20e3, max_val=100e3, n_default=8, fmt='%.0f', tick_step=10000),
    'I_cond_CS': InputParameter(
        name='I_cond_CS', display_name='Nominal CS conductor current',
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
    'E_BD': InputParameter(
        name='E_BD', display_name='Breakdown calibration parameter',
        unit='V·s/m', min_val=0.10, max_val=0.50, n_default=10, tick_step=0.05),

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
    'dt_rep_bl': InputParameter(
        name='dt_rep_bl', display_name='Blanket replacement downtime',
        unit='yr', min_val=0.2, max_val=3.0, n_default=10, tick_step=0.5),
    'dt_rep_div': InputParameter(
        name='dt_rep_div', display_name='Divertor replacement downtime',
        unit='yr', min_val=0.2, max_val=3.0, n_default=10, tick_step=0.5),
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


def display_input_parameters():
    """Print all scannable input parameters grouped by theme."""
    groups = [
        ('Primary geometry',         ['R0', 'a', 'b']),
        ('Magnetic field',           ['Bmax_TF', 'Bmax_CS_adm']),
        ('Fusion power',             ['P_fus']),
        ('Core plasma physics',      ['Tbar', 'tau_i_e', 'H', 'Zeff', 'betaN_limit', 'C_Alpha', 'rho_rad_core']),
        ('Profile peaking (Manual)', ['nu_n_manual', 'nu_T_manual']),
        ('Operation & heating',      ['P_aux_input', 'Temps_Plateau_input']),
        ('MHD limits',               ['q_limit', 'Greenwald_limit']),
        ('TF/CS engineering',        ['fatigue_CS', 'SF_TF', 'SF_CS', 'coef_inboard_tension', 'Gap', 'c_BP']),
        ('Superconductor',           ['T_helium', 'Marge_T_He', 'Marge_T_Nb3Sn',
                                      'Marge_T_NbTi', 'Marge_T_REBCO',
                                      'f_He_pipe', 'f_void', 'f_In_cable', 'f_In_WP']),
        ('Quench protection',        ['I_cond', 'I_cond_CS', 'V_max', 'T_hotspot', 'RRR']),
        ('Power conversion',         ['eta_T', 'M_blanket', 'eta_RF']),
        ('Multi-source CD',          ['f_heat_LH', 'f_heat_EC', 'f_heat_NBI',
                                      'rho_EC', 'rho_NBI', 'E_beam_keV']),
        ('Plasma-facing',            ['theta_deg']),
        ('Flux model',               ['Ce', 'E_BD']),
        ('Disruption / RE indicators (post-convergence)',
                                     ['tau_TQ', 'Te_final_eV', 'pellet_dilution']),
        ('Techno-economics (Sheffield, post-convergence)',
                                     ['discount_rate', 'T_life', 'T_build', 'contingency',
                                      'Util_factor', 'Dwell_factor', 'dt_rep_bl', 'dt_rep_div',
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

def _run_scan_point(args):
    """
    Worker for parallel 2-D scan — all dependencies imported locally.

    Using only local imports prevents cloudpickle from serializing the scan
    module's global namespace.  When D0FUS.py is run via %runfile in Spyder,
    that namespace is polluted with scipy symbols (erfc, etc.) that are bound
    to __main__, causing AttributeError in loky worker processes whose __main__
    is the frozen bootstrapper.

    The config is received as a plain dict and reconstructed into a GlobalConfig
    inside the worker.  Plain dicts with scalar values are always picklable,
    regardless of how the parent process imported the GlobalConfig class.

    The Greenwald closure of the deck (Tbar_mode or P_fus_mode = 'greenwald')
    is solved here, before run(), and the resolved configuration is returned,
    so that the stored P_fus, the costs and the runaway indicators use the
    solved operating point.

    Parameters
    ----------
    args : tuple  (y, x, config_dict, compute_re)
        y, x        : int   Grid indices.
        config_dict : dict  GlobalConfig fields as a plain Python dict.
        compute_re  : bool  Whether to compute RE indicators.

    Returns
    -------
    tuple : (y, x, results, re_dict, error, config_dict_resolved)
        results is None and error holds the exception text when the point
        fails. re_dict is None when compute_re is False, and holds a single
        'error' key when the RE indicators fail at this point.
    """
    # Bootstrap imports only (see module header): everything else comes from
    # the module globals re-created when the worker re-imports D0FUS_scan.
    import os, sys
    from dataclasses import asdict as _asdict

    # Make the D0FUS package importable in the worker process.
    _parent = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    y, x, config_dict, compute_re = args

    # Reconstruct GlobalConfig from the plain dict.
    from D0FUS_BIB.D0FUS_parameterization import GlobalConfig
    from D0FUS_EXE.D0FUS_run import (run, resolve_operating_point,
                                     runaway_indicators)
    config = GlobalConfig(**config_dict)

    try:
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            config = resolve_operating_point(config)
            results = run(config, verbose=0)
    except Exception as exc:
        return (y, x, None, None, f"{type(exc).__name__}: {exc}", None)

    re_dict = None
    if compute_re and np.isfinite(float(np.real(results[5]))):   # converged Q
        try:
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                re_dict = runaway_indicators(config, results)
        except Exception as exc:
            re_dict = {'error': f"{type(exc).__name__}: {exc}"}

    return (y, x, results, re_dict, None, _asdict(config))


def _finite(v):
    """True if v is a finite real number (False for None, NaN, Inf, text)."""
    try:
        return bool(np.isfinite(float(np.real(v))))
    except (TypeError, ValueError):
        return False


def generic_2D_scan(scan_params, fixed_params, base_config, compute_re=True,
                    n_workers=None):
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
        If True (default), compute the runaway electron indicators at each
        grid point with the RUN settings (runaway_indicators in D0FUS_run),
        so that a scan cell and a RUN of the same inputs give the same
        values. The cost is a few milliseconds per point.
    n_workers : int or None, optional
        Number of parallel worker processes.
        None (default) → one per 8 grid points, at most one per CPU.
        1              → sequential execution.
        N              → use exactly N worker processes.
        joblib/loky (cloudpickle-based) runs on every platform, Windows
        included, without any special guard.

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
    n_points = param1_n * param2_n
    
    print(f"\nStarting 2D scan:")
    print(f"  {param1_name}: [{param1_min}, {param1_max}] with {param1_n} points")
    print(f"  {param2_name}: [{param2_min}, {param2_max}] with {param2_n} points")
    print(f"  Total calculations: {n_points}")

    # Initialize outputs container
    outputs = ScanOutputs(shape=(param1_n, param2_n))

    # ── Build a single base config with all fixed overrides applied once ──
    fixed_overrides = {k: v for k, v in fixed_params.items()
                       if k in GlobalConfig.__dataclass_fields__}
    base = dc_replace(base_config, **fixed_overrides) if fixed_overrides else base_config
    
    # ── Resolve number of workers ─────────────────────────────────────────────
    # Default: one worker per 8 points, at most one per CPU. Each worker
    # imports D0FUS once (about 2 s), so small grids run faster sequentially.
    _n_cpu = os.cpu_count() or 1
    if n_workers is None:
        n_workers = min(_n_cpu, n_points // 8)
    n_workers = max(1, int(n_workers))

    _mode = f"parallel — {n_workers}/{_n_cpu} CPUs (joblib/loky)" if n_workers > 1 else "sequential"
    print(f"\n  Workers: {n_workers} ({_mode})")

    # ── Build flat job list (one entry per grid point) ────────────────────────
    # Configs are converted to plain dicts (asdict) — scalar values only,
    # no GlobalConfig class reference needed in the pickled payload.
    jobs = []
    for y, p1 in enumerate(param1_values):
        for x, p2 in enumerate(param2_values):
            cfg = dc_replace(base, **{param1_name: p1, param2_name: p2})
            jobs.append((y, x, asdict(cfg), compute_re))

    # ── Execute jobs ──────────────────────────────────────────────────────────
    # The progress bar advances when a point is finished (not when it is
    # submitted). Results come back in submission order.
    _desc = f'Scanning {param1_name}/{param2_name}'
    if n_workers == 1:
        results_list = [_run_scan_point(job) for job in tqdm(jobs, desc=_desc)]
    else:
        # loky (via joblib) uses cloudpickle — robust to module reloads.
        # The dict-based cache in D0FUS_radial_build_functions is picklable,
        # unlike the former @lru_cache wrapper which failed after %runfile.
        _gen = Parallel(n_jobs=n_workers, backend='loky', verbose=0,
                        return_as='generator')(
            delayed(_run_scan_point)(job) for job in jobs)
        results_list = list(tqdm(_gen, total=len(jobs), desc=_desc))

    # ── Process results in order ──────────────────────────────────────────────
    def _sigma_max(*sig):
        # Largest principal stress in absolute value [Pa -> MPa]
        return (max(abs(s) for s in sig) * 1e-6
                if all(_finite(s) for s in sig) else np.nan)

    def _pct(v):
        # Fraction -> percent, NaN kept for failed points
        return v * 100 if _finite(v) else np.nan

    n_error, n_no_plasma, n_no_build, n_re_error = 0, 0, 0, 0
    error_examples = {}   # first grid point of each distinct error message
    for y, x, res, re_dict, exc, config_dict in results_list:
        if res is None:
            outputs.fill_nan(y, x)
            n_error += 1
            error_examples.setdefault(exc, (y, x))
            continue
        # Resolved configuration of this point (Greenwald closure solved)
        config = GlobalConfig(**config_dict)

        # ===========================================================
        # Unpack the full run() return tuple
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
         f_sc_CS, f_cu_CS, f_He_pipe_CS, f_void_CS, f_He_CS, f_In_CS,
         beta_fast_alpha, betaN_total, tau_sd_alpha, W_fast_alpha,
         *_coil_extra) = res

        # ── Feasibility and cost: shared RUN evaluators ──────────────
        # Same rule and same Sheffield geometry as the RUN report, so a scan
        # cell and a RUN of the same inputs give identical verdicts and costs.
        _fz = evaluate_feasibility(config, res)
        n_condition    = _fz['ratios']['density']
        beta_condition = _fz['ratios']['beta']
        q_condition    = _fz['ratios']['q']
        max_limit      = _fz['max_ratio']
        _cres = evaluate_costs(config, res)
        _COE_val      = _cres.get('COE', np.nan)
        _C_invest_val = _cres.get('C_invest', np.nan) * 1e-3 if _cres else np.nan  # M EUR -> B EUR
        if not _finite(Q):
            n_no_plasma += 1
        elif not _fz['build_ok']:
            n_no_build += 1

        # ── Unpack radial build component volumes (offsets 37–43) ────────────
        V_rb_SOL_s        = _coil_extra[37]
        V_rb_FW_s         = _coil_extra[38]
        V_rb_BB_s         = _coil_extra[39]
        V_rb_shield_s     = _coil_extra[40]
        V_rb_VV_s         = _coil_extra[41]
        V_rb_gap_TF_s     = _coil_extra[42]
        V_rb_divertor_s   = _coil_extra[43]
        # ── Radial build masses (offsets 44–49) ──────────────────────────────
        M_rb_FW_s         = _coil_extra[44] * 1e-3   # kg → t
        M_rb_BB_s         = _coil_extra[45] * 1e-3
        M_rb_shield_s     = _coil_extra[46] * 1e-3
        M_rb_VV_s         = _coil_extra[47] * 1e-3
        M_rb_divertor_s   = _coil_extra[48] * 1e-3
        M_rb_total_s      = _coil_extra[49] * 1e-3

        # ── Unpack masses, volumes and conductor lengths from _coil_extra ───
        L_cable_TF_s    = _coil_extra[12] * 1e-3   # m → km
        L_cable_CS_s    = _coil_extra[13] * 1e-3
        L_sc_TF_s       = _coil_extra[16] * 1e-3
        L_sc_CS_s       = _coil_extra[17] * 1e-3
        M_steel_TF_s    = _coil_extra[18] * 1e-3   # kg → t
        M_sc_TF_s       = _coil_extra[19] * 1e-3
        M_cu_TF_s       = _coil_extra[20] * 1e-3
        M_total_TF_s    = _coil_extra[22] * 1e-3
        M_steel_CS_s    = _coil_extra[23] * 1e-3
        M_sc_CS_s       = _coil_extra[24] * 1e-3
        M_cu_CS_s       = _coil_extra[25] * 1e-3
        M_total_CS_s    = _coil_extra[27] * 1e-3
        V_blanket_s     = _coil_extra[28]

        # ── Unpack lifetime/availability outputs (offsets 29–36) ─────────────
        (t_bl_fpy, t_div_fpy,
         t_bl_yr,  t_div_yr,
         T_op_s,   dt_eff_s,
         Av_s,     CF_s_out) = _coil_extra[29:37]

        # ── Store all results ────────────────────────────────────────
        outputs.set_point(y, x,
            Q=Q,
            P_fus=config.P_fus,
            P_elec=P_elec,
            Cost=cost,
            COE=_COE_val,
            C_invest=_C_invest_val,
            Ip=Ip,
            Ib=Ib,
            I_CD=I_CD,
            I_Ohm=I_Ohm,
            n=nbar_line,
            nbar_vol=nbar,
            pbar=pbar,
            beta_N=betaN,
            beta_T=_pct(betaT),
            beta_P=betaP,
            q95=q95,
            qstar=qstar,
            tauE=tauE,
            W_th=W_th * 1e-6,
            f_bs=Ib / Ip * 100 if (_finite(Ip) and Ip > 0) else np.nan,
            f_alpha=_pct(f_alpha),
            tau_alpha=tau_alpha,
            B0=B0,
            BCS=B_CS,
            B_pol=B_pol,
            J_TF=J_TF * 1e-6,
            J_CS=J_CS * 1e-6,
            PsiCS=ΨCS,
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
            lambda_q=lambda_q * 1000 if _finite(lambda_q) else np.nan,
            eta_LH=eta_LH,
            eta_EC=eta_EC,
            eta_NBI=eta_NBI,
            c=c,
            d=d,
            c_d=c + d,
            r_minor=config.a,
            kappa=κ,
            kappa_95=κ_95,
            delta=δ,
            Volume=Volume,
            Surface=Surface,
            A=config.R0 / config.a if config.a > 0 else np.nan,
            sigma_TF=_sigma_max(σ_z_TF, σ_theta_TF, σ_r_TF),
            sigma_CS=_sigma_max(σ_z_CS, σ_theta_CS, σ_r_CS),
            Steel_fraction_TF=_pct(Steel_fraction_TF),
            Steel_fraction_CS=_pct(Steel_fraction_CS),
            f_sc_TF=_pct(f_sc_TF),
            f_He_TF=_pct(f_He_TF),
            f_sc_CS=_pct(f_sc_CS),
            f_He_CS=_pct(f_He_CS),
            limits=max_limit,
            density_limit=n_condition,
            beta_limit=beta_condition,
            q_limit=q_condition,
            V_rb_SOL=V_rb_SOL_s,
            V_rb_FW=V_rb_FW_s,
            V_rb_BB=V_rb_BB_s,
            V_rb_shield=V_rb_shield_s,
            V_rb_VV=V_rb_VV_s,
            V_rb_gap_TF=V_rb_gap_TF_s,
            V_rb_divertor=V_rb_divertor_s,
            M_rb_FW=M_rb_FW_s,
            M_rb_BB=M_rb_BB_s,
            M_rb_shield=M_rb_shield_s,
            M_rb_VV=M_rb_VV_s,
            M_rb_divertor=M_rb_divertor_s,
            M_rb_total=M_rb_total_s,
            V_blanket=V_blanket_s,
            M_total_TF=M_total_TF_s,
            M_sc_TF=M_sc_TF_s,
            M_steel_TF=M_steel_TF_s,
            M_cu_TF=M_cu_TF_s,
            M_total_CS=M_total_CS_s,
            M_sc_CS=M_sc_CS_s,
            M_steel_CS=M_steel_CS_s,
            M_cu_CS=M_cu_CS_s,
            L_cable_TF=L_cable_TF_s,
            L_cable_CS=L_cable_CS_s,
            L_sc_strand_TF=L_sc_TF_s,
            L_sc_strand_CS=L_sc_CS_s,
            t_blanket_fpy=t_bl_fpy,
            t_div_fpy=t_div_fpy,
            t_blanket_yr=t_bl_yr,
            t_div_yr=t_div_yr,
            T_op_limit=T_op_s,
            dt_rep_eff=dt_eff_s,
            Av=Av_s,
            CF=CF_s_out,
        )

        # Global feasibility (RUN rule): the radial build closes AND every
        # plasma limit ratio is <= 1. Drawn as the black feasibility limit.
        outputs['radial_build'][y, x] = config.R0 if _fz['feasible'] else np.nan

        # ── Runaway electron indicators (computed in worker) ─────────
        # re_dict is None when compute_re=False. It holds an 'error' key
        # when the RE calculation failed at this point.
        if re_dict is not None and 'error' in re_dict:
            n_re_error += 1
            error_examples.setdefault(f"RE indicators: {re_dict['error']}", (y, x))
        elif re_dict is not None:
            outputs['I_RE_seed'][y, x]  = re_dict['I_RE_seed'] * 1e-3
            outputs['I_RE_aval'][y, x]  = re_dict['I_RE_avalanche'] * 1e-6
            outputs['f_RE_Ip'][y, x]    = re_dict['f_RE_to_Ip'] * 100
            outputs['f_RE_avg'][y, x]   = re_dict['f_RE_avg']
            outputs['f_RE_core'][y, x]  = re_dict['f_RE_core']
            outputs['E_RE_kin'][y, x]   = re_dict['E_RE_kin']
            outputs['W_mag_RE'][y, x]   = re_dict['W_mag_RE']

    # ── Summary of the scan ──────────────────────────────────────────────────
    n_ok = n_points - n_error - n_no_plasma - n_no_build
    print(f"\n✓ Scan calculation completed: {n_points} points")
    print(f"  {n_ok:5d} converged with a closed radial build")
    print(f"  {n_no_build:5d} converged plasma, radial build not closed")
    print(f"  {n_no_plasma:5d} no converged plasma (solver)")
    print(f"  {n_error:5d} raised an exception")
    if compute_re and n_re_error:
        print(f"  {n_re_error:5d} without runaway indicators (RE calculation failed)")
    if error_examples:
        print("  Distinct error messages (first grid point of each):")
        for _msg, (_y, _x) in list(error_examples.items())[:8]:
            print(f"    {param1_name} = {param1_values[_y]:.4g}, "
                  f"{param2_name} = {param2_values[_x]:.4g}: {_msg}")
        if len(error_examples) > 8:
            print(f"    ... {len(error_examples) - 8} more")
    if n_error == n_points:
        raise RuntimeError(
            f"All {n_points} scan points raised an exception, for example: "
            f"{next(iter(error_examples))}")
    print()
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

    Parameters
    ----------
    prompt : str
        Question string displayed to the user.
    valid_options : list of str
        Valid parameter names from the output registry.

    Returns
    -------
    str
        Selected parameter name, or "none" when the user types "none" to
        suppress the iso-contour layer, or when no answer can be read
        (closed input stream).
    """
    while True:
        try:
            choice = input(prompt).strip()
        except (EOFError, OSError, RuntimeError):
            # No readable console (closed stdin, notebook without stdin)
            print("  No console input available: layer left out.")
            return "none"
        # Allow explicit opt-out: typing "none" disables the iso-contour layer
        if choice.lower() == "none":
            return "none"
        if choice in valid_options:
            return choice
        print(f"  Invalid choice '{choice}'. "
              f"Please choose from the available parameters, or type 'none' to skip.")
        print(f"  Hint: {', '.join(valid_options[:10])}...")


def _console_attached():
    """
    True when input() can reach a user: a terminal, or an IPython kernel
    (Spyder, Jupyter). False in batch runs, CI and piped executions.
    """
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return True
    except (AttributeError, ValueError):
        pass
    return 'ipykernel' in sys.modules


def select_iso_params(iso_param_1=None, iso_param_2=None, interactive=None):
    """
    Iso-contour layers of the scan map.

    A layer given as None is asked on the console when one is attached
    (interactive=None detects it, True forces the prompt, False never
    prompts) and is left out otherwise, so that a batch scan never blocks
    on input(). The string 'none' leaves the layer out.

    Returns
    -------
    (str or None, str or None)
        Registry keys of the two layers, None for a layer left out.

    Raises
    ------
    ValueError  If a requested name is not a plottable registry key.
    """
    valid = [k for keys in ScanOutputs.list_plottable_parameters().values()
             for k in keys]
    if interactive is None:
        interactive = _console_attached()
    prompts = ("\nChoose ISO-CONTOUR 1 (black dashed lines): ",
               "Choose ISO-CONTOUR 2 (white dashed lines): ")
    if interactive and (iso_param_1 is None or iso_param_2 is None):
        display_available_parameters()
    chosen = []
    for value, prompt in zip((iso_param_1, iso_param_2), prompts):
        if value is None:
            value = get_user_plot_choice(prompt, valid) if interactive else 'none'
        if str(value).lower() == 'none':
            chosen.append(None)
        elif value in valid:
            chosen.append(value)
        else:
            raise ValueError(f"Iso-contour parameter '{value}' is not a "
                             f"plottable output. Valid names: "
                             f"{', '.join(valid)}, or none.")
    return tuple(chosen)


def plot_generic_contours(ax, matrix, param_key,
                          color='black', linestyle='dashed',
                          linewidth=2.5, fontsize=22,
                          X=None, Y=None):
    """
    Plot contours for any registered parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    matrix : ndarray (2-D)
        Data matrix (already row-inverted if needed).
    param_key : str
        Key in OUTPUT_REGISTRY.
    color, linestyle, linewidth, fontsize : plot styling.
    X, Y : ndarray or None
        Physical coordinate meshgrids (cell centres). When provided the
        contours are drawn in the scanned parameter coordinates.

    Returns
    -------
    Line2D or None
        Legend handle, or None if no contours could be drawn.
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
        # Use physical coordinates when available
        if X is not None and Y is not None:
            contour = ax.contour(X, Y, matrix, levels=valid_levels, colors=color,
                                linestyles=linestyle, linewidths=linewidth)
        else:
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


# Mathtext symbols of the most common scan axes. The other inputs are
# labelled with their deck key in plain text.
_AXIS_SYMBOLS = {
    'R0': '$R_0$', 'a': '$a$', 'b': '$b$',
    'Bmax_TF': '$B_{max,TF}$', 'Bmax_CS_adm': '$B_{max,CS}$',
    'P_fus': '$P_{fus}$', 'Tbar': r'$\bar{T}$', 'H': '$H$',
    'Zeff': '$Z_{eff}$', 'betaN_limit': r'$\beta_{N,lim}$',
    'q_limit': '$q_{lim}$',
}


def _axis_label(param_name):
    """Axis label of a scanned input: symbol (or deck key) and unit."""
    unit = get_input_parameter_unit(param_name)
    label = _AXIS_SYMBOLS.get(param_name, param_name)
    return f"{label} [{unit}]" if unit else label


def _tick_formatter(param_name):
    """
    Tick label format of a scanned input: the registry fmt, else as many
    decimals as its tick_step, else None (matplotlib default formatter).
    """
    entry = INPUT_PARAMETER_REGISTRY.get(param_name)
    if entry is None:
        return None
    if entry.fmt:
        return FormatStrFormatter(entry.fmt)
    if entry.tick_step:
        txt = f"{entry.tick_step:g}"
        if 'e' not in txt:
            n_dec = len(txt.split('.')[1]) if '.' in txt else 0
            return FormatStrFormatter(f"%.{n_dec}f")
    return None


def plot_scan_results(outputs, param1_values, param2_values,
                      param1_name, param2_name, config, output_dir,
                      iso_param_1=None, iso_param_2=None,
                      clip_to_viable=True):
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
    iso_param_1   : str or None   First iso-contour layer (black dashed),
                                  'none' to leave it out. None asks on the
                                  console when one is attached.
    iso_param_2   : str or None   Second iso-contour layer (white dashed).
    clip_to_viable : bool
        If True (default), iso-contour quantities are restricted to the
        viable design region where the radial build closes AND all plasma
        limits are satisfied. The two boundary curves (white: plasma
        limits, black: feasibility limit, i.e. radial build and plasma
        limits together) are always drawn on the full domain.
        If False, iso-contour 1 is masked to the feasible region, and
        iso-contour 2 is unmasked.

    Returns
    -------
    fig, ax, iso_param_1, iso_param_2
    """
    iso_param_1, iso_param_2 = select_iso_params(iso_param_1, iso_param_2)
    print(f"\n  Plotting: iso_1={iso_param_1 or 'none'}, iso_2={iso_param_2 or 'none'}")

    # Font sizes (set on each artist: no change of the global rcParams)
    font_iso_1 = 22      # Black iso-contours
    font_iso_2 = 22      # White iso-contours
    font_legend = 20
    font_other = 15
    
    # Matrices in natural order (row 0 = smallest param1 value, at the bottom)
    radial_build = outputs['radial_build']
    
    # Limit matrices for colored background
    density_lim = outputs['density_limit'].copy()
    beta_lim = outputs['beta_limit'].copy()
    q_lim = outputs['q_limit'].copy()
    limits_all = outputs['limits']
    
    # Get iso-contour matrices with masking behavior controlled by clip_to_viable:
    #   clip_to_viable=True  → both masked to full viable zone (RB valid AND plasma ≤ 1)
    #   clip_to_viable=False → iso_param_1 masked to RB valid, iso_param_2 unmasked
    # When a parameter is None the corresponding layer is suppressed entirely.
    if clip_to_viable:
        _viable = ~np.isnan(radial_build) & (limits_all <= 1.0)
        iso_matrix_1 = (np.where(_viable, outputs[iso_param_1], np.nan)
                        if iso_param_1 is not None else None)
        iso_matrix_2 = (np.where(_viable, outputs[iso_param_2], np.nan)
                        if iso_param_2 is not None else None)
    else:
        iso_matrix_1 = (outputs.get_masked(iso_param_1, outputs['radial_build'])
                        if iso_param_1 is not None else None)
        iso_matrix_2 = (outputs[iso_param_2]
                        if iso_param_2 is not None else None)
    
    # ── Dominant (most-violated) stability limit per cell ────────────────────
    # Each cell is coloured by whichever of the three normalised ratios
    # (n/n_G, beta_N/beta_lim, q_lim/q95) is the largest, i.e. the most
    # binding constraint. The selection must be NaN-safe: np.argmax treats the
    # first NaN it encounters as the running maximum (since every comparison
    # `x > NaN` is False), which would mis-assign a cell to a layer whose ratio
    # is actually undefined there. We therefore (i) compete only finite ratios
    # by replacing NaN with -inf, and (ii) blank cells where *all* three ratios
    # are NaN (radial-build failure), so they are never falsely coloured.
    conditions = np.array([density_lim, beta_lim, q_lim])
    finite_any = np.any(np.isfinite(conditions), axis=0)
    conditions_safe = np.where(np.isfinite(conditions), conditions, -np.inf)
    idx_max = np.argmax(conditions_safe, axis=0)

    density_plot = np.where(finite_any & (idx_max == 0), density_lim, np.nan)
    beta_plot    = np.where(finite_any & (idx_max == 1), beta_lim,    np.nan)
    q_plot       = np.where(finite_any & (idx_max == 2), q_lim,       np.nan)
    
    # Only show where limits < 2
    mask_valid = limits_all < 2
    density_plot = np.where(mask_valid, density_plot, np.nan)
    beta_plot = np.where(mask_valid, beta_plot, np.nan)
    q_plot = np.where(mask_valid, q_plot, np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 13))
    
    # ── Physical coordinate system ──────────────────────────────────────
    # Each colour cell is centred on its grid point (shading='nearest'), the
    # same coordinates as the contour lines drawn below.
    X, Y = np.meshgrid(param2_values, param1_values)
    
    # Plot color maps for plasma limits
    _norm = Normalize(vmin=0.5, vmax=2.0)
    im_density = ax.pcolormesh(X, Y, density_plot, cmap='Blues', norm=_norm,
                               shading='nearest')
    im_q = ax.pcolormesh(X, Y, q_plot, cmap='Greens', norm=_norm,
                         shading='nearest')
    im_beta = ax.pcolormesh(X, Y, beta_plot, cmap='Reds', norm=_norm,
                            shading='nearest')
    
    # Plasma stability boundary (limit = 1) — always drawn on full data
    linewidth = 2.5
    ax.contour(X, Y, limits_all, levels=[1.0], colors='white', linewidths=linewidth)
    white_boundary = mlines.Line2D([], [], color='white', linewidth=linewidth,
                                   label='Plasma limits boundary')
    
    # Feasibility limit (radial build and plasma limits) — full data
    filled_matrix = np.where(np.isnan(radial_build), -1, 1)
    ax.contour(X, Y, filled_matrix, levels=[0], linewidths=linewidth, colors='black')
    black_boundary = mlines.Line2D([], [], color='black', linewidth=linewidth,
                                   label='Feasibility limit')
    
    # Configure axes labels
    ax.set_xlabel(_axis_label(param2_name), fontsize=24)
    ax.set_ylabel(_axis_label(param1_name), fontsize=24)
    
    # ── Axis ticks — use registry tick_step when available ─────────────
    _entry_x = INPUT_PARAMETER_REGISTRY.get(param2_name)
    _entry_y = INPUT_PARAMETER_REGISTRY.get(param1_name)
    if _entry_x and _entry_x.tick_step:
        ax.xaxis.set_major_locator(MultipleLocator(_entry_x.tick_step))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    if _entry_y and _entry_y.tick_step:
        ax.yaxis.set_major_locator(MultipleLocator(_entry_y.tick_step))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
    for _axis, _name in ((ax.xaxis, param2_name), (ax.yaxis, param1_name)):
        _fmt = _tick_formatter(_name)
        if _fmt is not None:
            _axis.set_major_formatter(_fmt)
    ax.tick_params(labelsize=font_legend, axis='x', rotation=45)
    ax.tick_params(labelsize=font_legend, axis='y')
    
    # Configure colorbars
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("bottom", size="5%", pad=1.3)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    cax3 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    
    cax1.annotate('$n/n_{G}$', xy=(-0.01, 0.5), xycoords='axes fraction',
                 ha='right', va='center', fontsize=font_other)
    cax2.annotate(r'$\beta_N/\beta_{lim}$', xy=(-0.01, 0.5), xycoords='axes fraction',
                 ha='right', va='center', fontsize=font_other)
    _q_cbar_label = ('$q_{lim}/q_{95}$' if config.kink_parameter == 'q95'
                      else '$q_{lim}/q_*$')
    cax3.annotate(_q_cbar_label, xy=(-0.01, 0.5), xycoords='axes fraction',
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
        cax.tick_params(labelsize=font_other)
    
    # Plot ISO-CONTOUR 1 (black dashed) — skipped when iso_param_1 is None
    iso_legend_1 = (
        plot_generic_contours(ax, iso_matrix_1, iso_param_1,
                              color='black', linestyle='dashed',
                              linewidth=linewidth, fontsize=font_iso_1,
                              X=X, Y=Y)
        if iso_param_1 is not None else None
    )

    # Plot ISO-CONTOUR 2 (white dashed) — skipped when iso_param_2 is None
    iso_legend_2 = (
        plot_generic_contours(ax, iso_matrix_2, iso_param_2,
                              color='white', linestyle='dashed',
                              linewidth=linewidth, fontsize=font_iso_2,
                              X=X, Y=Y)
        if iso_param_2 is not None else None
    )
    
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

def save_scan_results(outputs, param1_values, param2_values,
                      param1_name, param2_name, config, output_dir,
                      input_file_path=None, iso_param_1=None, iso_param_2=None):
    """
    Save a finished scan in a new folder output_dir/scan/Scan_D0FUS_<time>:
    the result matrices (scan_matrices.npz) and the deck
    (scan_parameters.txt, copied, or rebuilt from config when no deck file
    is given). main() calls it before plotting, so that a plotting error
    never loses a finished scan.

    Parameters
    ----------
    config : GlobalConfig
        Reference configuration (used to rebuild the deck when no file).

    Returns
    -------
    str : Path to output directory.
    """
    output_path, _ = new_output_dir(os.path.join(output_dir, 'scan'),
                                    'Scan_D0FUS')
    
    input_copy = os.path.join(output_path, "scan_parameters.txt")
    if input_file_path and os.path.exists(input_file_path):
        # Copy of the original input file
        shutil.copy2(input_file_path, input_copy)
    else:
        # Generate input file from configuration
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
            f.write(f"iso_param_1 = {iso_param_1 or 'none'}  # Black dashed lines\n")
            f.write(f"iso_param_2 = {iso_param_2 or 'none'}  # White dashed lines\n")
    
    # Save matrices as NPZ for post-processing
    np.savez(os.path.join(output_path, "scan_matrices.npz"),
             param1_values=param1_values,
             param2_values=param2_values,
             param1_name=param1_name,
             param2_name=param2_name,
             **outputs.to_dict())
    
    print(f"✓ Scan data saved to: {output_path}")
    return output_path


def save_scan_figure(fig, output_path, param1_name, param2_name,
                     iso_param_1, iso_param_2):
    """Save the scan map (PNG, 300 dpi) in output_path. Returns its path."""
    # "none" is the filename token of a layer left out
    _iso1_tag = iso_param_1 if iso_param_1 is not None else "none"
    _iso2_tag = iso_param_2 if iso_param_2 is not None else "none"
    fig_path = os.path.join(
        output_path,
        f"scan_map_{param1_name}_{param2_name}_{_iso1_tag}_{_iso2_tag}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Scan map saved to: {fig_path}\n")
    return fig_path


#%% Main Function

def main(input_file=None, auto_plot=False,
         iso_param_1=None, iso_param_2=None,
         compute_re=True, clip_to_viable=True, n_workers=None,
         output_dir=None, save_figures=True):
    """
    Main execution function for scans.

    Parameters
    ----------
    input_file   : str or None   Path to input file (default: the ITER
                                  example D0FUS_INPUTS/2_scan_ITER.txt).
    auto_plot    : bool           If True, never prompt: a layer given
                                  neither here nor in the deck is left out.
                                  If False (default), such a layer is asked
                                  on the console when one is attached.
    iso_param_1  : str or None    First iso-contour parameter (black lines).
                                  Overrides the deck key iso_param_1.
    iso_param_2  : str or None    Second iso-contour parameter (white lines).
                                  Overrides the deck key iso_param_2.
    compute_re   : bool           If True (default), compute runaway electron indicators.
                                  Can also be overridden by ``compute_re = 0`` in the input file.
    clip_to_viable : bool         If True (default), iso-contours are restricted to
                                  the viable design region (radial build valid AND
                                  plasma stable).  Boundary curves are always shown.
                                  Set to False to revert to legacy masking behavior.
                                  Can also be overridden by ``clip_to_viable = 0`` in the
                                  input file.
    n_workers    : int or None    Worker processes. Overrides the deck key
                                  n_workers. Default: one per 8 grid
                                  points, at most one per CPU.
    output_dir   : str or None    Parent folder of the scan/ result folders.
                                  Default: default_output_root() of D0FUS_run.
    save_figures : bool           If False, the map is not drawn, saved or
                                  shown, and no layer is asked. The matrices
                                  (NPZ) and the deck are written anyway.
    """
    if input_file is None:
        input_file = os.path.join(os.path.dirname(__file__), '..',
                                  'D0FUS_INPUTS', '2_scan_ITER.txt')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"\nLoading parameters from: {input_file}")

    # ── Build base config from fixed parameters in the input file ────────
    # load_config_from_file skips bracketed scan declarations and the SCAN
    # settings, and returns a GlobalConfig with the scalar overrides applied.
    base_config = load_config_from_file(input_file, verbose=2,
                                        ignore_keys=SCAN_SETTING_KEYS)
    
    # Load scan and fixed parameters (for grid definition and display)
    scan_params, fixed_params = load_scan_parameters(input_file)

    # SCAN settings of the deck (iso-contours, RE indicators, masking, workers)
    settings = load_scan_settings(input_file)
    compute_re = settings.get('compute_re', compute_re)
    clip_to_viable = settings.get('clip_to_viable', clip_to_viable)
    if n_workers is None:
        n_workers = settings.get('n_workers')
    if iso_param_1 is None:
        iso_param_1 = settings.get('iso_param_1')
    if iso_param_2 is None:
        iso_param_2 = settings.get('iso_param_2')
    # Layers chosen before the scan: a wrong name fails now, and the
    # console question is not left waiting after a long computation.
    iso_param_1, iso_param_2 = select_iso_params(
        iso_param_1, iso_param_2,
        interactive=False if (auto_plot or not save_figures) else None)
    
    # Print scan configuration
    print("\n" + "="*73)
    print("Starting D0FUS 2D parameter scan...")
    print("="*73)
    print(f"\nScan parameters:")
    
    try:
        # Perform scan
        outputs, param1_values, param2_values, param1_name, param2_name = generic_2D_scan(
            scan_params, fixed_params, base_config, compute_re=compute_re,
            n_workers=n_workers
        )
        
        # Save the data first, then plot
        if output_dir is None:
            output_dir = default_output_root()
        os.makedirs(output_dir, exist_ok=True)
        output_path = save_scan_results(
            outputs, param1_values, param2_values, param1_name, param2_name,
            base_config, output_dir, input_file_path=input_file,
            iso_param_1=iso_param_1, iso_param_2=iso_param_2
        )
        if not save_figures:
            return outputs, param1_values, param2_values, output_path
        
        fig, ax, iso_used_1, iso_used_2 = plot_scan_results(
            outputs, param1_values, param2_values, param1_name, param2_name,
            base_config, None, iso_param_1 or 'none', iso_param_2 or 'none',
            clip_to_viable=clip_to_viable
        )
        save_scan_figure(fig, output_path, param1_name, param2_name,
                         iso_used_1, iso_used_2)
        
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
