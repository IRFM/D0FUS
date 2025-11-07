"""
D0FUS Scan Module
Generates 2D parameter space maps with full visualization

Created on: Dec 2023
Author: Auclair Timothe
"""

import sys
import os
from datetime import datetime
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all necessary modules
from D0FUS_BIB.D0FUS_parameterization import *
from D0FUS_BIB.D0FUS_radial_build_functions import *

# Try to import physical functions if they exist in a separate file
try:
    from D0FUS_BIB.D0FUS_physical_functions import *
except ImportError:
    pass  # Functions might be in radial_build_functions

from D0FUS_EXE.D0FUS_run import run, Parameters


def R0_a_scan(params, scan_config=None):
    """
    Perform 2D scan over R0 and a parameters
    
    Args:
        params: Parameters object with baseline configuration
        scan_config: Dict with scan parameters (optional)
    
    Returns:
        matrices: Dict containing all calculated matrices
        a_values: Array of minor radius values
        R0_values: Array of major radius values
    """
    
    # Default scan configuration
    if scan_config is None:
        scan_config = {
            'a_min': 1, 'a_max': 3, 'a_N': 25,
            'R0_min': 3, 'R0_max': 9, 'R0_N': 25
        }
    
    a_values = np.linspace(scan_config['a_min'], scan_config['a_max'], scan_config['a_N'])
    R0_values = np.linspace(scan_config['R0_min'], scan_config['R0_max'], scan_config['R0_N'])
    
    # Initialize all matrices
    matrices = {
        'density': np.zeros((len(a_values), len(R0_values))),
        'security': np.zeros((len(a_values), len(R0_values))),
        'beta': np.zeros((len(a_values), len(R0_values))),
        'radial_build': np.zeros((len(a_values), len(R0_values))),
        'limits': np.zeros((len(a_values), len(R0_values))),
        'Heat': np.zeros((len(a_values), len(R0_values))),
        'Cost': np.zeros((len(a_values), len(R0_values))),
        'Q': np.zeros((len(a_values), len(R0_values))),
        'P_CD': np.zeros((len(a_values), len(R0_values))),
        'Gamma_n': np.zeros((len(a_values), len(R0_values))),
        'L_H': np.zeros((len(a_values), len(R0_values))),
        'f_alpha': np.zeros((len(a_values), len(R0_values))),
        'TF_ratio': np.zeros((len(a_values), len(R0_values))),
        'Ip': np.zeros((len(a_values), len(R0_values))),
        'n': np.zeros((len(a_values), len(R0_values))),
        'beta_N': np.zeros((len(a_values), len(R0_values))),
        'q95': np.zeros((len(a_values), len(R0_values))),
        'B0': np.zeros((len(a_values), len(R0_values))),
        'BCS': np.zeros((len(a_values), len(R0_values))),
        'c': np.zeros((len(a_values), len(R0_values))),
        'd': np.zeros((len(a_values), len(R0_values)))
    }
    
    print(f"\nStarting 2D scan:")
    print(f"  a: [{scan_config['a_min']}, {scan_config['a_max']}] m with {scan_config['a_N']} points")
    print(f"  R0: [{scan_config['R0_min']}, {scan_config['R0_max']}] m with {scan_config['R0_N']} points")
    print(f"  Total calculations: {len(a_values) * len(R0_values)}\n")
    
    # Scanning loop
    for x, R0 in enumerate(tqdm(R0_values, desc='Scanning R0')):
        for y, a in enumerate(a_values):
            try:
                # Run calculation
                results = run(
                    a, R0, params.Bmax, params.P_fus, params.Tbar, params.H,
                    params.Temps_Plateau_input, params.b, params.nu_n, params.nu_T,
                    params.Supra_choice, params.Chosen_Steel, params.Radial_build_model,
                    params.Choice_Buck_Wedg, params.Option_Kappa, params.κ_manual,
                    params.L_H_Scaling_choice, params.Scaling_Law, params.Bootstrap_choice,
                    params.Operation_mode, params.fatigue, params.P_aux_input
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
                 TF_ratio,
                 r_minor, r_sep, r_c, r_d,
                 κ, κ_95, δ, δ_95) = results
                
                # Calculate plasma limit conditions
                # betaN_limit and q_limit are imported from D0FUS_parameterization
                # They are your renamed variables: betaN_limit = 2.8, q_limit = 2.5
                
                n_condition = nbar / nG                # Should be < 1
                beta_condition = betaN / betaN_limit   # Should be < 1
                q_condition = q_limit / qstar          # Should be < 1
                
                max_limit = max(n_condition, beta_condition, q_condition)
                
                # Store all results in matrices
                matrices['Q'][y, x] = Q
                matrices['Cost'][y, x] = cost
                matrices['Heat'][y, x] = heat
                matrices['P_CD'][y, x] = P_CD
                matrices['Gamma_n'][y, x] = Gamma_n
                matrices['L_H'][y, x] = P_sep / P_Thresh if P_Thresh > 0 else np.nan
                matrices['f_alpha'][y, x] = f_alpha * 100
                matrices['TF_ratio'][y, x] = TF_ratio * 100
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
                    matrices['radial_build'][y, x] = R0
                else:
                    matrices['radial_build'][y, x] = np.nan
                
                # Identify the most limiting constraint and store ONLY that one
                conditions = np.array([n_condition, beta_condition, q_condition])
                idx_max = np.argmax(conditions)
                
                if max_limit < 2:
                    if idx_max == 0:
                        # Density is most constraining
                        matrices['density'][y, x] = n_condition
                    elif idx_max == 1:
                        # Beta is most constraining
                        matrices['beta'][y, x] = beta_condition
                    elif idx_max == 2:
                        # Safety factor is most constraining
                        matrices['security'][y, x] = q_condition
                
            except Exception as e:
                # Fill with NaN on error
                for key in matrices:
                    matrices[key][y, x] = np.nan
                # Print first few errors for debugging
                if x < 3 and y < 3:
                    print(f"\n  Debug: Error at R0={R0:.2f}, a={a:.2f}: {str(e)}")
                continue
    
    print("\n✓ Scan calculation completed!\n")
    return matrices, a_values, R0_values


def plot_scan_results(matrices, a_values, R0_values, params, output_dir, 
                      iso_param=None, bg_param=None):
    """
    Generate and save scan visualization plots
    
    Args:
        matrices: Dict of calculated matrices
        a_values: Array of minor radius values
        R0_values: Array of major radius values
        params: Parameters object
        output_dir: Directory to save results
        iso_param: Iso-contour parameter (if None, will ask user)
        bg_param: Background parameter (if None, will ask user)
    """
    
    # Ask user for plot preferences if not provided
    if iso_param is None:
        iso_param = input("Choose iso-contour parameter (Ip, n, beta, q95, B0, BCS, c, d, c&d): ").strip()
    if bg_param is None:
        bg_param = input("Choose background parameter (Heat, Cost, Q, Gamma_n, L_H, Alpha, TF, B0, BCS): ").strip()
    
    # Font sizes
    font_topological = 20
    font_background = 28
    font_subtitle = 15
    font_legend = 20
    font_other = 15
    font_title = 22
    plt.rcParams.update({'font.size': font_other})
    
    # Invert matrices for plotting (flip vertically)
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
    
    # Title
    title_choice = 1
    if title_choice == 1:
        plt.suptitle(f"Parameter space: a, $\\mathbf{{R_0}}$", fontsize=font_title, y=0.94, fontweight='bold')
        plt.title(f"$B_{{\\mathrm{{max}}}}$ = {params.Bmax} [T], $P_{{\\mathrm{{fus}}}}$ = {params.P_fus} [MW], "
                 f"scaling law: {params.Scaling_Law}", fontsize=font_subtitle)
    
    # Plot color maps for plasma limits
    min_val, max_val = 0.5, 2.0
    im_density = ax.imshow(inv_matrices['density'], cmap='Blues', aspect='auto',
                          interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    im_security = ax.imshow(inv_matrices['security'], cmap='Greens', aspect='auto',
                           interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    im_beta = ax.imshow(inv_matrices['beta'], cmap='Reds', aspect='auto',
                       interpolation='nearest', norm=Normalize(vmin=min_val, vmax=max_val))
    
    # Contour for plasma stability boundary (limit = 1)
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
    ax.set_xlabel('$R_0$ [m]', fontsize=24)
    ax.set_ylabel('a [m]', fontsize=24)
    
    # Configure colorbars
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("bottom", size="5%", pad=1.3)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    cax3 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    
    # Colorbar annotations
    cax1.annotate('n/$n_{\\mathrm{G}}$', xy=(-0.01, 0.5), xycoords='axes fraction', 
                 ha='right', va='center', fontsize=font_other)
    cax2.annotate(r'$\beta$/$\beta_{T}$', xy=(-0.01, 0.5), xycoords='axes fraction', 
                 ha='right', va='center', fontsize=font_other)
    cax3.annotate('$q_{\\mathrm{K}}$/$q_{*}$', xy=(-0.01, 0.5), xycoords='axes fraction', 
                 ha='right', va='center', fontsize=font_other)
    
    # Create colorbars
    cbar_density = plt.colorbar(im_density, cax=cax1, orientation='horizontal')
    tick_labels = cbar_density.ax.xaxis.get_ticklabels()
    if tick_labels:
        tick_labels[-1].set_visible(False)
    
    cbar_beta = plt.colorbar(im_beta, cax=cax2, orientation='horizontal')
    tick_labels = cbar_beta.ax.xaxis.get_ticklabels()
    if tick_labels:
        tick_labels[-1].set_visible(False)
    
    cbar_security = plt.colorbar(im_security, cax=cax3, orientation='horizontal')
    
    # Add vertical lines at value 1 for each colorbar
    for cax in [cax1, cax2, cax3]:
        cax.axvline(x=1, color='white', linewidth=2.5)
    
    # Configure y-axis (a_values)
    a_min, a_max = a_values[0], a_values[-1]
    approx_step_y = 0.5
    real_step_y = (a_max - a_min) / (len(a_values) - 1)
    index_step_y = max(1, int(round(approx_step_y / real_step_y)))
    y_indices = np.arange(0, len(a_values), index_step_y)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(np.round((a_max + a_min) - a_values[y_indices], 2), fontsize=font_legend)
    
    # Configure x-axis (R0_values)
    R0_min, R0_max = R0_values[0], R0_values[-1]
    approx_step_x = 1.0
    real_step_x = (R0_max - R0_min) / (len(R0_values) - 1)
    index_step_x = max(1, int(round(approx_step_x / real_step_x)))
    x_indices = np.arange(0, len(R0_values), index_step_x)
    ax.set_xticks(x_indices)
    x_labels = [round(R0_values[i], 2) for i in x_indices]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=font_legend)
    
    # Add iso-contours based on user choice
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
    else:
        print(f'Warning: Unknown iso parameter "{iso_param}"')
    
    # Add background contours based on user choice
    white_line = None
    if bg_param == 'Heat':
        contour_bg = ax.contour(inv_matrices['Heat'], levels=np.arange(1000, 10000, 500), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='Heat// [MW-T/m]')
    elif bg_param == 'Q':
        contour_bg = ax.contour(inv_matrices['Q'], levels=np.arange(0, 60, 10), 
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
    elif bg_param == 'TF':
        contour_bg = ax.contour(inv_matrices['TF_ratio'], levels=np.arange(0, 100, 5), 
                               colors='white', linestyles='dashed', linewidths=linewidth)
        ax.clabel(contour_bg, inline=True, fmt='%d', fontsize=font_background)
        white_line = mlines.Line2D([], [], linewidth=linewidth, color='white', 
                                   linestyle='dashed', label='TF tension fraction [%]')
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
    else:
        print(f'Warning: Unknown background parameter "{bg_param}"')
    
    # Add legend
    legend_handles = [white_dashed_line, black_line]
    if white_line:
        legend_handles.append(white_line)
    if grey_line:
        legend_handles.append(grey_line)
    
    ax.legend(handles=legend_handles, loc='upper left', facecolor='lightgrey', 
             fontsize=font_legend)
    
    return fig, ax, iso_param, bg_param


def save_scan_results(fig, matrices, a_values, R0_values, params, output_dir, 
                     iso_param, bg_param, input_file_path=None):
    """Save scan results to timestamped directory"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"Scan_D0FUS_{timestamp}"
    output_path = os.path.join(output_dir, output_name)
    
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
            f.write("# Fixed parameters:\n")
            for key, value in vars(params).items():
                f.write(f"{key} = {value}\n")
            f.write("\n# Scan configuration:\n")
            f.write(f"# a: [{a_values[0]:.2f}, {a_values[-1]:.2f}] m with {len(a_values)} points\n")
            f.write(f"# R0: [{R0_values[0]:.2f}, {R0_values[-1]:.2f}] m with {len(R0_values)} points\n")
            f.write(f"# Iso-contour: {iso_param}\n")
            f.write(f"# Background: {bg_param}\n")
    
    # Save figure
    fig_filename = f"scan_map_{iso_param}_{bg_param}.png"
    fig_path = os.path.join(output_path, fig_filename)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Scan figure saved to: {fig_path}")
    print(f"✓ All results saved to: {output_path}\n")
    
    # Reset matplotlib to defaults
    plt.rcdefaults()
    
    return output_path


def main(input_file=None, scan_config=None, auto_plot=False, 
         iso_param=None, bg_param=None):
    """
    Main execution function for scans
    
    Args:
        input_file: Path to input file (optional)
        scan_config: Dict with scan configuration (optional)
        auto_plot: If True, use provided iso_param and bg_param without asking
        iso_param: Iso-contour parameter (if auto_plot=True)
        bg_param: Background parameter (if auto_plot=True)
    """
    
    # Load parameters
    p = Parameters()
    
    # Store input file path for copying later
    input_file_path = input_file
    
    if input_file is None:
        default_input = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_INPUTS', 'default_input.txt')
        if os.path.exists(default_input):
            input_file = default_input
    
    if input_file and os.path.exists(input_file):
        print(f"\nLoading parameters from: {input_file}")
        p.open_input(input_file)
    else:
        print(f"\nWarning: Input file not found. Using default parameters.")
        input_file_path = None
    
    # Print scan configuration
    print("\n" + "="*73)
    print("Starting D0FUS 2D parameter scan...")
    print("="*73)
    print(f"\nFixed parameters:")
    print(f"  Bmax = {p.Bmax} T")
    print(f"  P_fus = {p.P_fus} MW")
    print(f"  Scaling Law = {p.Scaling_Law}")
    print(f"  Operation mode = {p.Operation_mode}")
    print(f"  Superconductor = {p.Supra_choice}")
    print(f"  Mechanical config = {p.Choice_Buck_Wedg}")
    
    # Default scan configuration if not provided
    if scan_config is None:
        scan_config = {
            'a_min': 1, 'a_max': 3, 'a_N': 25,
            'R0_min': 3, 'R0_max': 9, 'R0_N': 25
        }
    
    try:
        # Perform scan
        matrices, a_values, R0_values = R0_a_scan(p, scan_config)
        
        # Plot results
        if auto_plot and iso_param and bg_param:
            # Automatic plotting with provided parameters
            fig, ax, iso_used, bg_used = plot_scan_results(
                matrices, a_values, R0_values, p, None, iso_param, bg_param
            )
        else:
            # Interactive plotting - ask user for preferences
            fig, ax, iso_used, bg_used = plot_scan_results(
                matrices, a_values, R0_values, p, None
            )
        
        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'D0FUS_OUTPUTS')
        os.makedirs(output_dir, exist_ok=True)
        output_path = save_scan_results(
            fig, matrices, a_values, R0_values, p, output_dir,
            iso_used, bg_used, input_file_path
        )
        
        # Show plot
        plt.show()
        
        return matrices, a_values, R0_values, output_path
    
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