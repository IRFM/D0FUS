

"""
D0FUS Scan Module - Complete Generic Version
Generates 2D parameter space maps with full visualization
Supports scanning any 2 parameters dynamically

Author: Auclair Timothe
"""
#%% Import

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_BIB.D0FUS_radial_build_functions import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_EXE.D0FUS_run import run, Parameters


#%% Code

def parse_scan_parameter(line):
    """
    Parse a scan parameter line with bracket syntax
    Example: "R0 = [3, 9, 25]" -> ("R0", 3.0, 9.0, 25)
    
    Returns:
        tuple: (param_name, min_value, max_value, n_points)
    """
    # Pattern: parameter = [min, max, n_points]
    match = re.match(r'^\s*(\w+)\s*=\s*\[([^\]]+)\]', line)
    if not match:
        return None
    
    param_name = match.group(1).strip()
    values_str = match.group(2)
    
    # Split by comma or semicolon
    values = re.split(r'[,;]', values_str)
    if len(values) != 3:
        raise ValueError(f"Scan parameter {param_name} must have exactly 3 values: [min, max, n_points]")
    
    min_val = float(values[0].strip())
    max_val = float(values[1].strip())
    n_points = int(float(values[2].strip()))
    
    return (param_name, min_val, max_val, n_points)


def load_scan_parameters(input_file):
    """
    Load parameters from input file, identifying scan parameters
    
    Returns:
        tuple: (scan_params, fixed_params)
            scan_params: list of (name, min, max, n_points)
            fixed_params: dict of fixed parameter values
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    scan_params = []
    fixed_params = {}
    scan_param_names = []  # Keep track of scan parameter names
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            # Try to parse as scan parameter
            scan_param = parse_scan_parameter(line)
            if scan_param:
                scan_params.append(scan_param)
                scan_param_names.append(scan_param[0])  # Store the parameter name
                continue
            
            # Parse as fixed parameter (only if not a scan parameter)
            if '=' in line:
                parts = line.split('=', 1)
                param_name = parts[0].strip()
                param_value = parts[1].strip()
                
                # Skip if this is a scan parameter (already processed)
                if param_name in scan_param_names:
                    continue
                
                # Try to convert to float
                try:
                    param_value = float(param_value)
                    if param_value.is_integer():
                        param_value = int(param_value)
                except ValueError:
                    pass  # Keep as string
                
                fixed_params[param_name] = param_value
    
    if len(scan_params) != 2:
        raise ValueError(f"Expected exactly 2 scan parameters, found {len(scan_params)}")
    
    return scan_params, fixed_params


def get_parameter_unit(param_name):
    """Get the unit for a parameter name"""
    units = {
        'R0': 'm', 'a': 'm', 'b': 'm',
        'P_fus': 'MW',
        'Bmax': 'T',
        'Tbar': 'keV',
        'H': '',
        'nu_n': '', 'nu_T': '',
    }
    return units.get(param_name, '')


def generic_2D_scan(scan_params, fixed_params, params_obj):
    """
    Perform generic 2D scan over any two parameters
    
    Args:
        scan_params: list of 2 tuples (name, min, max, n_points)
        fixed_params: dict of fixed parameter values
        params_obj: Parameters object to update
    
    Returns:
        matrices: dict of result matrices
        param1_values: array of first parameter values
        param2_values: array of second parameter values
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
    
    # Initialize all matrices
    matrices = {
        'density': np.zeros((len(param1_values), len(param2_values))),
        'security': np.zeros((len(param1_values), len(param2_values))),
        'beta': np.zeros((len(param1_values), len(param2_values))),
        'radial_build': np.zeros((len(param1_values), len(param2_values))),
        'limits': np.zeros((len(param1_values), len(param2_values))),
        'Heat': np.zeros((len(param1_values), len(param2_values))),
        'Cost': np.zeros((len(param1_values), len(param2_values))),
        'Q': np.zeros((len(param1_values), len(param2_values))),
        'P_CD': np.zeros((len(param1_values), len(param2_values))),
        'Gamma_n': np.zeros((len(param1_values), len(param2_values))),
        'L_H': np.zeros((len(param1_values), len(param2_values))),
        'f_alpha': np.zeros((len(param1_values), len(param2_values))),
        'Ip': np.zeros((len(param1_values), len(param2_values))),
        'n': np.zeros((len(param1_values), len(param2_values))),
        'beta_N': np.zeros((len(param1_values), len(param2_values))),
        'q95': np.zeros((len(param1_values), len(param2_values))),
        'B0': np.zeros((len(param1_values), len(param2_values))),
        'BCS': np.zeros((len(param1_values), len(param2_values))),
        'c': np.zeros((len(param1_values), len(param2_values))),
        'd': np.zeros((len(param1_values), len(param2_values)))
    }
    
    # Apply fixed parameters to params_obj
    for param_name, param_value in fixed_params.items():
        if hasattr(params_obj, param_name):
            setattr(params_obj, param_name, param_value)
    
    # Scanning loop
    for y, param1_val in enumerate(tqdm(param1_values, desc=f'Scanning {param1_name}')):
        for x, param2_val in enumerate(param2_values):
            
            # Set scan parameter values
            setattr(params_obj, param1_name, param1_val)
            setattr(params_obj, param2_name, param2_val)
            
            try:
                # Run calculation
                results = run(
                    params_obj.a, params_obj.R0, params_obj.Bmax, params_obj.P_fus, 
                    params_obj.Tbar, params_obj.H,
                    params_obj.Temps_Plateau_input, params_obj.b, params_obj.nu_n, params_obj.nu_T,
                    params_obj.Supra_choice, params_obj.Chosen_Steel, params_obj.Radial_build_model,
                    params_obj.Choice_Buck_Wedg, params_obj.Option_Kappa, params_obj.κ_manual,
                    params_obj.L_H_Scaling_choice, params_obj.Scaling_Law, params_obj.Bootstrap_choice,
                    params_obj.Operation_mode, params_obj.fatigue, params_obj.P_aux_input
                )
                
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
                 Gamma_n,
                 f_alpha, tau_alpha,
                 J_TF, J_CS,
                 c, c_WP_TF, c_Nose_TF, σ_z_TF, σ_theta_TF, σ_r_TF, Steel_fraction_TF,
                 d, σ_z_CS, σ_theta_CS, σ_r_CS, Steel_fraction_CS, B_CS, J_CS,
                 r_minor, r_sep, r_c, r_d,
                 κ, κ_95, δ, δ_95) = results
                
                # Calculate plasma limit conditions
                betaN_limit_value = 2.8
                q_limit_value = 2.5
                
                n_condition = nbar / nG
                beta_condition = betaN / betaN_limit_value
                q_condition = q_limit_value / qstar
                
                max_limit = max(n_condition, beta_condition, q_condition)
                
                # Store all results in matrices
                matrices['Q'][y, x] = Q
                matrices['Cost'][y, x] = cost
                matrices['Heat'][y, x] = heat
                matrices['P_CD'][y, x] = P_CD
                matrices['Gamma_n'][y, x] = Gamma_n
                matrices['L_H'][y, x] = P_sep / P_Thresh if P_Thresh > 0 else np.nan
                matrices['f_alpha'][y, x] = f_alpha * 100
                matrices['Ip'][y, x] = Ip
                matrices['n'][y, x] = nbar
                matrices['beta_N'][y, x] = betaN
                matrices['q95'][y, x] = q95
                matrices['B0'][y, x] = B0
                matrices['BCS'][y, x] = B_CS
                matrices['c'][y, x] = r_sep - r_c
                matrices['d'][y, x] = r_c - r_d
                
                matrices['limits'][y, x] = max_limit
                
                # Initialize all limit matrices to NaN
                matrices['density'][y, x] = np.nan
                matrices['security'][y, x] = np.nan
                matrices['beta'][y, x] = np.nan
                
                # Check radial build validity
                if not np.isnan(r_d) and max_limit < 1 and r_d > 0:
                    matrices['radial_build'][y, x] = params_obj.R0
                else:
                    matrices['radial_build'][y, x] = np.nan
                
                # Identify the most limiting constraint and store ONLY that one
                conditions = np.array([n_condition, beta_condition, q_condition])
                idx_max = np.argmax(conditions)
                
                if max_limit < 2:
                    if idx_max == 0:
                        matrices['density'][y, x] = n_condition
                    elif idx_max == 1:
                        matrices['beta'][y, x] = beta_condition
                    elif idx_max == 2:
                        matrices['security'][y, x] = q_condition
                
            except Exception as e:
                # Fill with NaN on error
                for key in matrices:
                    matrices[key][y, x] = np.nan
                if y < 2 and x < 2:
                    print(f"\n  Debug: Error at {param1_name}={param1_val:.2f}, {param2_name}={param2_val:.2f}: {str(e)}")
                continue
    
    print("\n✓ Scan calculation completed!\n")
    return matrices, param1_values, param2_values, param1_name, param2_name


def plot_scan_results(matrices, param1_values, param2_values, param1_name, param2_name,
                      params, output_dir, iso_param=None, bg_param=None):
    """Generate and save scan visualization plots"""
    
    # Get units
    unit_param1 = get_parameter_unit(param1_name)
    unit_param2 = get_parameter_unit(param2_name)
    
    # Ask user for plot preferences if not provided
    if iso_param is None:
        iso_param = input("Choose iso-contour parameter (Ip, n, beta, q95, B0, BCS, c, d, c&d): ").strip()
    if bg_param is None:
        bg_param = input("Choose background parameter (Heat, Cost, Q, Gamma_n, L_H, Alpha, B0, BCS): ").strip()
    
    # Font sizes
    font_topological = 22
    font_background = 22
    font_subtitle = 15
    font_legend = 20
    font_other = 15
    font_title = 30
    plt.rcParams.update({'font.size': font_other})
    
    # Invert matrices for plotting
    inv_matrices = {key: val[::-1, :] for key, val in matrices.items()}
    
    # Create masked versions for radial build
    inv_Ip_mask = inv_matrices['Ip'].copy()
    inv_n_mask = inv_matrices['n'].copy()
    inv_beta_mask = inv_matrices['beta_N'].copy()
    inv_q95_mask = inv_matrices['q95'].copy()
    inv_B0_mask = inv_matrices['B0'].copy()
    inv_BCS_mask = inv_matrices['BCS'].copy()
    inv_c_mask = inv_matrices['c'].copy()
    inv_d_mask = inv_matrices['d'].copy()
    
    # Apply radial build mask
    mask = np.isnan(inv_matrices['radial_build'])
    inv_Ip_mask[mask] = np.nan
    inv_n_mask[mask] = np.nan
    inv_beta_mask[mask] = np.nan
    inv_q95_mask[mask] = np.nan
    inv_B0_mask[mask] = np.nan
    inv_BCS_mask[mask] = np.nan
    inv_c_mask[mask] = np.nan
    inv_d_mask[mask] = np.nan
    inv_c_d_mask = inv_c_mask + inv_d_mask
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 13))
    
    # Plot color maps for plasma limits
    min_val, max_val = 0.5, 2.0
    im_density = ax.imshow(inv_matrices['density'], cmap='Blues', aspect='auto',
                          interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    im_security = ax.imshow(inv_matrices['security'], cmap='Greens', aspect='auto',
                           interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    im_beta = ax.imshow(inv_matrices['beta'], cmap='Reds', aspect='auto',
                       interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    
    # Contour for plasma stability boundary
    linewidth = 2.5
    ax.contour(inv_matrices['limits'], levels=[1.0], colors='white', linewidths=linewidth)
    white_dashed_line = mlines.Line2D([], [], color='white', linewidth=linewidth, 
                                     label='Plasma stability boundary')
    
    # Contour for radial build boundary
    filled_matrix = np.where(np.isnan(inv_matrices['radial_build']), -1, 1)
    ax.contour(filled_matrix, levels=[0], linewidths=linewidth, colors='black')
    black_line = mlines.Line2D([], [], color='black', linewidth=linewidth, 
                               label='Radial build limit')
    
    # Configure axes
    label_param2 = f"${param2_name}$" + (f" [{unit_param2}]" if unit_param2 else "")
    label_param1 = f"${param1_name}$" + (f" [{unit_param1}]" if unit_param1 else "")
    ax.set_xlabel(label_param2, fontsize=24)
    ax.set_ylabel(label_param1, fontsize=24)
    
    # Configure colorbars
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("bottom", size="5%", pad=1.3)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    cax3 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    
    cax1.annotate('n/$n_{\\mathrm{G}}$', xy=(-0.01, 0.5), xycoords='axes fraction', 
                 ha='right', va='center', fontsize=font_other)
    cax2.annotate(r'$\beta$/$\beta_{T}$', xy=(-0.01, 0.5), xycoords='axes fraction', 
                 ha='right', va='center', fontsize=font_other)
    cax3.annotate('$q_{\\mathrm{K}}$/$q_{*}$', xy=(-0.01, 0.5), xycoords='axes fraction', 
                 ha='right', va='center', fontsize=font_other)
    
    cbar_density = plt.colorbar(im_density, cax=cax1, orientation='horizontal')
    if cbar_density.ax.xaxis.get_ticklabels():
        cbar_density.ax.xaxis.get_ticklabels()[-1].set_visible(False)
    
    cbar_beta = plt.colorbar(im_beta, cax=cax2, orientation='horizontal')
    if cbar_beta.ax.xaxis.get_ticklabels():
        cbar_beta.ax.xaxis.get_ticklabels()[-1].set_visible(False)
    
    cbar_security = plt.colorbar(im_security, cax=cax3, orientation='horizontal')
    
    for cax in [cax1, cax2, cax3]:
        cax.axvline(x=1, color='white', linewidth=2.5)
    
    # Configure y-axis (param1)
    approx_step_y = (param1_values[-1] - param1_values[0]) / 10
    real_step_y = (param1_values[-1] - param1_values[0]) / (len(param1_values) - 1)
    index_step_y = max(1, int(round(approx_step_y / real_step_y)))
    y_indices = np.arange(0, len(param1_values), index_step_y)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(np.round(param1_values[::-1][y_indices], 2), fontsize=font_legend)
    
    # Configure x-axis (param2)
    approx_step_x = (param2_values[-1] - param2_values[0]) / 10
    real_step_x = (param2_values[-1] - param2_values[0]) / (len(param2_values) - 1)
    index_step_x = max(1, int(round(approx_step_x / real_step_x)))
    x_indices = np.arange(0, len(param2_values), index_step_x)
    ax.set_xticks(x_indices)
    x_labels = [round(param2_values[i], 2) for i in x_indices]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=font_legend)
    
    # Add iso-contours
    grey_line = None
    if iso_param == 'Ip':
        contour_lines = ax.contour(inv_Ip_mask, levels=np.arange(1, 25, 1), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%d', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='$I_p$ [MA]')
    elif iso_param == 'n':
        contour_lines = ax.contour(inv_n_mask, levels=np.arange(0.25, 5, 0.25), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='$n$ [10$^{20}$ m$^{-3}$]')
    elif iso_param == 'beta':
        contour_lines = ax.contour(inv_beta_mask, levels=np.arange(1, 10, 0.25), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='$\\beta_N$ [%]')
    elif iso_param == 'q95':
        contour_lines = ax.contour(inv_q95_mask, levels=np.arange(1, 10, 0.5), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='$q_{95}$')
    elif iso_param == 'B0':
        contour_lines = ax.contour(inv_B0_mask, levels=np.arange(1, 25, 0.5), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='$B_0$ [T]')
    elif iso_param == 'BCS':
        contour_lines = ax.contour(inv_BCS_mask, levels=np.arange(1, 100, 2), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='$B_{CS}$ [T]')
    elif iso_param == 'c':
        contour_lines = ax.contour(inv_c_mask, levels=np.arange(0, 10, 0.1), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='TF width [m]')
    elif iso_param == 'd':
        contour_lines = ax.contour(inv_d_mask, levels=np.arange(0, 10, 0.1), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='CS width [m]')
    elif iso_param == 'c&d':
        contour_lines = ax.contour(inv_c_d_mask, levels=np.arange(0, 10, 0.1), 
                                   colors='black', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=font_topological)
        grey_line = mlines.Line2D([], [], linewidth=linewidth, color='black', 
                                 linestyle='dashed', label='CS + TF width [m]')
    
    # Add background contours
    white_line = None
    if bg_param == 'Heat':
        contour_bg = ax.contour(inv_matrices['Heat'], levels=np.arange(1000, 10000, 500), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='Heat// [MW-T/m]')
    elif bg_param == 'Q':
        contour_bg = ax.contour(inv_matrices['Q'], levels=np.arange(0, 110, 10), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='Q')
    elif bg_param == 'Cost':
        contour_bg = ax.contour(inv_matrices['Cost'], levels=np.arange(0, 1000, 20), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='Cost [m$^3$]')
    elif bg_param == 'Gamma_n':
        contour_bg = ax.contour(inv_matrices['Gamma_n'], levels=np.arange(0, 20, 1), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='$\\Gamma_n$ [MW/m²]')
    elif bg_param == 'L_H':
        contour_bg = ax.contour(inv_matrices['L_H'], levels=np.arange(0, 10, 1), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='L-H Transition')
    elif bg_param == 'Alpha':
        contour_bg = ax.contour(inv_matrices['f_alpha'], levels=np.arange(0, 100, 1), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='Alpha fraction [%]')
    elif bg_param == 'B0':
        contour_bg = ax.contour(inv_B0_mask, levels=np.arange(0, 25, 0.5), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%.1f', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='$B_0$ [T]')
    elif bg_param == 'BCS':
        contour_bg = ax.contour(inv_BCS_mask, levels=np.arange(0, 100, 5), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%.1f', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='$B_{CS}$ [T]')
    
    # Add legend
    legend_handles = [white_dashed_line, black_line]
    if white_line:
        legend_handles.append(white_line)
    if grey_line:
        legend_handles.append(grey_line)
    
    ax.legend(handles=legend_handles, loc='upper left', facecolor='lightgrey', 
             fontsize=font_legend)
    
    return fig, ax, iso_param, bg_param


def save_scan_results(fig, matrices, param1_values, param2_values, param1_name, param2_name,
                     params, output_dir, iso_param, bg_param, input_file_path=None):
    """Save scan results to timestamped directory"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"Scan_D0FUS_{timestamp}"
    output_path = os.path.join(output_dir,'scan', output_name)
    
    # Create directory
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
            f.write(f"# Iso-contour: {iso_param}\n")
            f.write(f"# Background: {bg_param}\n")
    
    # Save figure
    fig_filename = f"scan_map_{param1_name}_{param2_name}_{iso_param}_{bg_param}.png"
    fig_path = os.path.join(output_path, fig_filename)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ All results saved to: {output_path}\n")
    
    # Reset matplotlib to defaults
    plt.rcdefaults()
    
    return output_path


def main(input_file=None, auto_plot=False, iso_param=None, bg_param=None):
    """
    Main execution function for scans
    
    Args:
        input_file: Path to input file (optional)
        auto_plot: If True, use provided iso_param and bg_param without asking
        iso_param: Iso-contour parameter (if auto_plot=True)
        bg_param: Background parameter (if auto_plot=True)
    """
    
    # Load parameters
    p = Parameters()
    
    # Store input file path for copying later
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
        unit = get_parameter_unit(param_name)
        unit_str = f" [{unit}]" if unit else ""
        print(f"  {param_name}: [{min_val}, {max_val}]{unit_str} with {n_points} points")
    
    print(f"\nFixed parameters:")
    for key, value in list(fixed_params.items())[:6]:  # Show first 6
        print(f"  {key} = {value}")
    if len(fixed_params) > 6:
        print(f"  ... and {len(fixed_params) - 6} more")
    
    try:
        # Perform scan
        matrices, param1_values, param2_values, param1_name, param2_name = generic_2D_scan(
            scan_params, fixed_params, p
        )
        
        # Plot results
        if auto_plot and iso_param and bg_param:
            # Automatic plotting with provided parameters
            fig, ax, iso_used, bg_used = plot_scan_results(
                matrices, param1_values, param2_values, param1_name, param2_name,
                p, None, iso_param, bg_param
            )
        else:
            # Interactive plotting - ask user for preferences
            fig, ax, iso_used, bg_used = plot_scan_results(
                matrices, param1_values, param2_values, param1_name, param2_name,
                p, None
            )
        
        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        output_path = save_scan_results(
            fig, matrices, param1_values, param2_values, param1_name, param2_name,
            p, output_dir, iso_used, bg_used, input_file_path
        )
        
        # Show plot
        plt.show()
        
        return matrices, param1_values, param2_values, output_path
    
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