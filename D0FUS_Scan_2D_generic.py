# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:08:01 2024

@author: TA276941
"""
#%% Import
import sys
import os
import re
from sympy import symbols, latex

# Répertoire des modules
base_dir = os.path.dirname(__file__)
dofus_dir = os.path.join(base_dir, 'D0FUS_BIB')

# Ajouter le répertoire au sys.path s’il n’y est pas déjà
if dofus_dir not in sys.path:
    sys.path.append(dofus_dir)

# Trouver tous les fichiers .py dans D0FUS_BIB (hors __init__.py)
module_files = [
    f for f in os.listdir(dofus_dir)
    if f.endswith('.py') and f != '__init__.py'
]

# Convertir les noms de fichiers en noms de modules
module_names = [os.path.splitext(f)[0] for f in module_files]

# Supprimer les modules du cache (sys.modules)
for mod in module_names:
    full_mod_name = mod  # Ex: 'D0FUS_parameterization'
    if full_mod_name in sys.modules:
        del sys.modules[full_mod_name]

# Recharger dynamiquement chaque module avec import *
for mod in module_names:
    exec(f'from {mod} import *')

print("----- D0FUS reloaded -----")


def D0fus_Scan_2D_generic(param_1, param_2, inputs):
    
    
    a_min, a_max, a_N = 1, 3, 25
    R0_min, R0_max, R0_N = 3, 9, 25   

    n_points = 25
    
    param_1_min, param_1_max = param_1[1] , param_1[2]
    param_1_name = param_1[0]
    unit_param_1 = "m"
    param_1_values = np.linspace(param_1_min, param_1_max, int(param_1[3]))
     
    param_2_min, param_2_max = param_2[1] , param_2[2]
    param_2_name = param_2[0]
    unit_param_2 = "m"
    param_2_values = np.linspace(param_2_min, param_2_max, int(param_2[3]) )
    

  
    # Matrix creations
    max_limits_density = np.zeros((len(param_1_values),len(param_2_values)))
    max_limits_security = np.zeros((len(param_1_values),len(param_2_values)))
    max_limits_beta = np.zeros((len(param_1_values),len(param_2_values)))
    radial_build_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    max_limits_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    
    Heat_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    Cost_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    Q_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    P_CD_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    Gamma_n_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    L_H_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    f_alpha_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    TF_ratio_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    c_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    d_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    
    Ip_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    n_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    beta_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    q95_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    B0_matrix = np.zeros((len(param_1_values),len(param_2_values)))
    BCS_matrix = np.zeros((len(param_1_values),len(param_2_values)))
 
    
    
    def set_module_var(name, value):
        globals()[name] = value

    # default value for a and R0 
    set_module_var("a", 1.6)
    set_module_var("R0", 6)
    
    for elem in inputs:
        set_module_var(elem[0], elem[1])
        
    print("a", a)
    # Initialisation
    x = 0
    y = 0
    
    # Scanning R0
    for param_2 in tqdm(param_2_values, desc='Scanning parameters'):
        
        # Increment after iterations
        y = 0
        
        # Scanning param_1
        for param_1 in param_1_values:
            
            set_module_var(param_1_name, param_1)
            set_module_var(param_2_name, param_2)
            
            
            # Main calcul
            (B0_solution, B_CS, B_pol_solution,
            tauE_solution, W_th_solution,
            Q_solution, Volume_solution, Surface_solution,
            Ip_solution, Ib_solution, I_CD_solution, I_Ohm_solution,
            n_solution, nG_solution, pbar_solution,
            betaN_solution, betaT_solution, betaP_solution,
            qstar_solution, q95_solution, q_mhd_solution,
            P_CD, P_sep, P_Thresh, eta_CD, P_elec_solution,
            cost, P_Brem_solution, P_syn_solution,
            heat, heat_par_solution, heat_pol_solution, lambda_q_Eich_m, q_target_Eich,
            P_1rst_wall_Hmod, P_1rst_wall_Lmod,
            Gamma_n,
            f_alpha_solution, tau_alpha,
            J_max_TF_conducteur, J_max_CS_conducteur,
            TF_ratio, R0_a_solution, R0_a_b_solution, R0_a_b_c_solution, R0_a_b_c_d_solution,
            κ, κ_95, δ, δ_95) = run( a, R0, Bmax, P_fus, 
            Tbar, H, Temps_Plateau_input, b , nu_n, nu_T,
            Supra_choice, Chosen_Steel , Radial_build_model , 
            Choice_Buck_Wedg , Option_Kappa , κ_manual,
            L_H_Scaling_choice, Scaling_Law, Bootstrap_choice, Operation_mode)
            
            # Verifier les conditions
            n_condition = n_solution / nG_solution
            beta_condition = betaN_solution / betaN
            q_condition = q / qstar_solution
            
            max_limit = max(n_condition, beta_condition, q_condition)
            
            # Initialisation
            radial_build = np.nan
            max_limit_density = np.nan
            max_limit_security = np.nan
            max_limit_beta = np.nan
            max_limit_power = np.nan
            
            if not np.isnan(R0_a_b_c_d_solution) and not np.isnan(max_limit) and max_limit<1 and R0_a_b_c_d_solution>0 :
                radial_build = R0
            
            # Création d'un tableau contenant les valeurs des conditions
            conditions = np.array([n_condition, beta_condition, q_condition])
            # Indice de la condition la plus contraignante
            indice_max_condition = np.argmax(conditions)
            
            if not np.isnan(max_limit) and max_limit < 2 :
                # Action en fonction de la limite la plus contraignante
                if indice_max_condition == 0:
                    # Action spécifique pour n_condition
                    max_limit_density = max_limit
                elif indice_max_condition == 1:
                    # Action spécifique pour beta_condition
                    max_limit_beta = max_limit
                elif indice_max_condition == 2:
                    # Action spécifique pour q_condition
                    max_limit_security = max_limit
                    
            # Store the value in the matrix
            # Radial Build
            radial_build_matrix[y,x] = radial_build       
            # Plasma limits
            max_limits_density[y, x] = max_limit_density
            max_limits_security[y, x] = max_limit_security
            max_limits_beta[y, x] = max_limit_beta
            max_limits_matrix[y, x] = max_limit
            # Details
            Ip_matrix[y,x] = Ip_solution
            n_matrix[y,x] = n_solution
            beta_matrix[y,x] = betaN_solution
            q95_matrix[y,x] = q95_solution
            B0_matrix[y,x] = B0_solution
            BCS_matrix[y,x] = B_CS
            # Figure of merits
            Heat_matrix[y,x] = heat
            Q_matrix[y,x] = Q_solution
            P_CD_matrix[y,x] = P_CD
            Gamma_n_matrix[y,x] = Gamma_n
            Cost_matrix[y,x] = cost
            L_H_matrix[y,x] = P_sep/P_Thresh
            f_alpha_matrix[y,x] = f_alpha_solution*100
            TF_ratio_matrix[y,x] = TF_ratio*100
            c_matrix[y,x] = R0_a_b_solution - R0_a_b_c_solution
            d_matrix[y,x] = R0_a_b_c_solution - R0_a_b_c_d_solution
            
            y = y+1
            
        x = x+1
        
    taille_police_topological_map = 20 # typical value of 15, for prensentation : 20
    taille_police_background_map = 20 # typical value of 15, for prensentation : 20
    taille_police_subtitle = 15
    taille_police_legende = 20 # typical value of 15, for prensentation : 20
    taille_police_other = 15
    taille_titre = 22
    plt.rcParams.update({'font.size': taille_police_other})

    # Inverser l'ordre des lignes des matrices
    inverted_matrix_density_limit = max_limits_density[::-1, :]
    inverted_matrix_security_limit = max_limits_security[::-1, :]
    inverted_matrix_beta_limit = max_limits_beta[::-1, :]
    inverted_matrix_radial_build = radial_build_matrix[::-1, :]
    inverted_matrix_plasma_limit = max_limits_matrix[::-1, :]

    inverted_Ip_matrix = Ip_matrix[::-1, :]
    inverted_n_matrix = n_matrix[::-1, :]
    inverted_beta_matrix = beta_matrix[::-1, :]
    inverted_q95_matrix = q95_matrix[::-1, :]
    inverted_B0_matrix = B0_matrix[::-1, :]
    inverted_BCS_matrix = BCS_matrix[::-1, :]
    inverted_c_matrix = c_matrix[::-1, :]
    inverted_d_matrix = d_matrix[::-1, :]
    
    inverted_Ip_matrix_mask = inverted_Ip_matrix
    inverted_n_matrix_mask = inverted_n_matrix
    inverted_beta_matrix_mask = inverted_beta_matrix
    inverted_q95_matrix_mask = inverted_q95_matrix
    inverted_B0_matrix_mask = inverted_B0_matrix
    inverted_BCS_matrix_mask = inverted_BCS_matrix
    inverted_c_matrix_mask = inverted_c_matrix
    inverted_d_matrix_mask = inverted_d_matrix
    # Créez un masque pour les valeurs NaN dans les matrices associées au Radial build
    mask = np.isnan(inverted_matrix_radial_build)
    # Appliquez ce masque à la nouvelle matrice
    inverted_Ip_matrix_mask[mask] = np.nan
    inverted_n_matrix_mask[mask] = np.nan
    inverted_beta_matrix_mask[mask] = np.nan
    inverted_q95_matrix_mask[mask] = np.nan
    inverted_B0_matrix_mask[mask] = np.nan
    inverted_BCS_matrix_mask[mask] = np.nan
    inverted_c_matrix_mask[mask] = np.nan
    inverted_d_matrix_mask[mask] = np.nan
    inverted_c_d_matrix_mask = inverted_c_matrix_mask + inverted_d_matrix_mask
   
    inverted_Heat_matrix = Heat_matrix[::-1, :]
    inverted_Cost_matrix = Cost_matrix[::-1, :]
    inverted_Q_matrix = Q_matrix[::-1, :]
    inverted_P_CD_matrix = P_CD_matrix[::-1, :]
    inverted_Gamma_n_matrix = Gamma_n_matrix[::-1, :]
    inverted_L_H_matrix = L_H_matrix[::-1, :]
    inverted_f_alpha_matrix = f_alpha_matrix[::-1, :]
    inverted_TF_ratio_matrix = TF_ratio_matrix[::-1, :]

    # Ask the user to choose the second topologic map :
    chosen_isocontour = input("Choose the Iso parameter (Ip, n, beta, q95, B0, BCS, c, d, c&d): ")
    chosen_topologic = input("Choose the background parameter (Heat, Cost, Q, Gamma_n, L_H, Alpha, TF): ")
    
    # Créer une figure et un axe principal
    fig, ax = plt.subplots(figsize=(10, 13))
    svg = f"Bmax={Bmax}_Pfus={P_fus}_scaling_law={Scaling_Law}_Triangularity={δ}_Hfactor={H}_Meca={Choice_Buck_Wedg}_"
    plt.title(f"$B_{{\mathrm{{max}}}}$ = {Bmax} [T], $P_{{\mathrm{{fus}}}}$ = {P_fus} [MW], scaling law : {Scaling_Law}",fontsize=taille_police_subtitle)
    

    title_parameter = "$" + latex(param_1_name) + "$"
    title_parameter_2 = "$" + latex(param_2_name) + "$"
 
    plt.suptitle(f"Parameter space : {title_parameter}, {title_parameter_2}", fontsize=taille_titre,y=0.94, fontweight='bold')

    # Calculer le minimum et le maximum des valeurs numériques
    min_matrix = 0.5
    max_matrix = 2
    # Choix des couleurs
    color_choice_density = 'Blues'
    color_choice_security = 'Greens'
    color_choice_beta = 'Reds'
    # Afficher les heatmap pour les matrices avec imshow
    im_density = ax.imshow(
    max_limits_density,  # <--- inverse les lignes
    cmap=color_choice_density,
    aspect='auto',
    interpolation='nearest',
    norm=Normalize(vmin=min_matrix, vmax=max_matrix),
    origin='lower'
)


    im_security = ax.imshow(inverted_matrix_security_limit, cmap=color_choice_security, aspect='auto', interpolation='nearest', norm=Normalize(vmin=min_matrix, vmax=max_matrix))
    im_beta = ax.imshow(inverted_matrix_beta_limit, cmap=color_choice_beta, aspect='auto', interpolation='nearest', norm=Normalize(vmin=min_matrix, vmax=max_matrix))

    # Coutour limite plasma = 1
    threshold = 1.0
    ax.contour(inverted_matrix_plasma_limit, levels=[threshold], colors='#555555', linestyles='dashed')
    grey_dashed_line = mlines.Line2D([], [], color='#555555', linestyle='dashed', label='Plasma stability boundary')

    # Personnaliser les axes et le titre
    plt.xlabel(title_parameter_2 +  "[" + unit_param_2 + "]", fontsize = taille_police_legende)
    plt.ylabel(title_parameter + "[" + unit_param_1 + "]", fontsize = taille_police_legende)

    ### Color Bars

    # Créer un axe supplémentaire en dessous de l'axe principal pour les colorbars
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("bottom", size="5%", pad=1)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    cax3 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
    
    # Ajouter des annotations textuelles à côté des colorbars
    cax1.annotate('n/$n_{\mathrm{G}}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_other)
    cax2.annotate(r'$\beta$/$\beta_{T}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_other)
    cax3.annotate('$q_{\mathrm{K}}$/$q_{\mathrm{*}}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_other)
    
    # Créer les colorbars avec les orientations désirées
    cbar_density = plt.colorbar(im_density, cax=cax1, orientation='horizontal')
    tick_labels = cbar_density.ax.xaxis.get_ticklabels()
    tick_labels[-1].set_visible(False)
    cbar_beta = plt.colorbar(im_beta, cax=cax2, orientation='horizontal')
    tick_labels = cbar_beta.ax.xaxis.get_ticklabels()
    tick_labels[-1].set_visible(False)
    cbar_security = plt.colorbar(im_security, cax=cax3, orientation='horizontal')
    tick_labels = cbar_security.ax.xaxis.get_ticklabels()
    
    # Ajouter les traits en pointillé à la valeur 1 pour chaque color bar
    for cax in [cax1, cax2, cax3]:
        cax.axvline(x=1, color='#333333', linestyle='--', linewidth=1, dashes=(5, 3))

    ### Definition des axes

    # X = param_1_values
    ax.set_xticks(np.arange(len(param_1_values)))
    ax.set_xticklabels(
    [f"{v:.2f}" for v in param_1_values],
    rotation=45,
    ha='right',
    fontsize=taille_police_legende
        )
    ax.set_xlabel(f"{param_1_name} [{unit_param_1}]", fontsize=taille_police_legende)

    # Y = param_2_values
   

    ax.set_yticks(np.arange(len(param_2_values)))
    ax.set_yticklabels(
        [f"{int(v)}" for v in param_2_values[::-1] ],
        fontsize=taille_police_legende
    )
    ax.set_ylabel(f"{param_2_name} [{unit_param_2}]", fontsize=taille_police_legende)

    ### Ajouter des contours

    # Remplacer NaN par -1 et les autres par 1 dans la matrice pour les contours du radial build
    filled_matrix = np.where(np.isnan(inverted_matrix_radial_build), -1, 1)
    # Définir le niveau de contour pour les valeurs de transition
    contour_level = [0]
    # Tracer les contours autour des valeurs de transition
    contour_radial_line = ax.contour(
    filled_matrix, 
    levels=contour_level,
    colors='black'   # Noir
    )
    # Création d'une ligne de légende personnalisée correspondant au style du contour
    black_line = mlines.Line2D([], [], color='black', label='Radial build limit')
    
    # Tracer les isocontours choisis précédemment
    if chosen_isocontour == 'Ip':
        contour_lines = ax.contour(inverted_Ip_matrix_mask, levels=np.arange(1, 25,1), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%d', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$I_p$ [MA]')
    elif chosen_isocontour == 'n':
        contour_lines = ax.contour(inverted_n_matrix_mask, levels=np.arange(0.25, 5,0.25), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$n$ [10e20]')
    elif chosen_isocontour == 'beta':
        contour_lines = ax.contour(inverted_beta_matrix_mask, levels=np.arange(1, 10,0.25), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$\\beta$ [%]')
    elif chosen_isocontour == 'q95':
        contour_lines = ax.contour(inverted_q95_matrix_mask, levels=np.arange(1, 10,0.5), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$q_{95}$ []')
    elif chosen_isocontour == 'B0':
        contour_lines = ax.contour(inverted_B0_matrix_mask, levels=np.arange(1, 25 ,0.5), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$B_{0}$ [T]')
    elif chosen_isocontour == 'BCS':
        contour_lines = ax.contour(inverted_BCS_matrix_mask, levels=np.arange(1, 100 ,2), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$B_{CS}$ [T]')
    elif chosen_isocontour == 'c':
        contour_lines = ax.contour(inverted_c_matrix_mask, levels=np.arange(0, 10 ,0.1), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$TF$ width [m]')
    elif chosen_isocontour == 'd':
        contour_lines = ax.contour(inverted_d_matrix_mask, levels=np.arange(0, 10 ,0.1), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='$CS$ width [m]')
    elif chosen_isocontour == 'c&d':
        contour_lines = ax.contour(inverted_c_d_matrix_mask, levels=np.arange(0, 10 ,0.1), colors='#555555')
        ax.clabel(contour_lines, inline=True, fmt='%.2f', fontsize=taille_police_topological_map)
        grey_line = mlines.Line2D([], [], color='#555555', label='CS + TF width [m]')
    else:
        print('Choose a relevant Iso parameter')

    if chosen_topologic == 'Heat':
        # Heat contours
        contour_lines_Heat = ax.contour(inverted_Heat_matrix, levels=np.arange(1000, 10000 ,500), colors='white')
        ax.clabel(contour_lines_Heat, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='Heat// [MW-T/m]')
    elif chosen_topologic == 'Q':
        # Q contours
        contour_lines_Q = ax.contour(inverted_Q_matrix, levels=np.arange(0, 60 ,10), colors='white')
        ax.clabel(contour_lines_Q, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='Q []')
    elif chosen_topologic == 'Cost' :
        # Cost contours
        contour_lines_cost = ax.contour(inverted_Cost_matrix, levels=np.arange(0, 1000,20), colors='white')
        ax.clabel(contour_lines_cost, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='$Cost$ [$m^3$]')
    elif chosen_topologic == 'Gamma_n' :
        # Gamma_n contours
        contour_lines_Gamma_n = ax.contour(inverted_Gamma_n_matrix, levels=np.arange(-1, 20,1), colors='white')
        ax.clabel(contour_lines_Gamma_n, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='$\\Gamma_n$ [MW/m²]')
    elif chosen_topologic == 'L_H' :
        # L-H transition contours
        contour_lines_LH = ax.contour(inverted_L_H_matrix, levels=np.arange(-1, 10,1), colors='white')
        ax.clabel(contour_lines_LH, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='L-H Transition []')
    elif chosen_topologic == 'Alpha' :
        # Alpha fraction contours
        contour_lines_Alpha = ax.contour(inverted_f_alpha_matrix, levels=np.arange(-1, 100,1), colors='white')
        ax.clabel(contour_lines_Alpha, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='Alpha fraction [%]')
    elif chosen_topologic == 'TF' :
        # TF ratio contours
        contour_lines_TF = ax.contour(inverted_TF_ratio_matrix, levels=np.arange(0, 100,5), colors='white')
        ax.clabel(contour_lines_TF, inline=True, fmt='%d', fontsize=taille_police_background_map)
        white_line = mlines.Line2D([], [], color='white', label='TF tension fraction [%]')
    else :
        print('Choose a relevant background parameter')
    
    # Légende
    ax.legend(handles=[grey_line,white_line, grey_dashed_line], loc='upper left', facecolor='lightgrey', fontsize=taille_police_legende)
    
    # Save the image
    # Remplacer les virgules par des points dans svg
    svg = svg.replace('.', ',')
    path_to_save = os.path.join(save_directory,f"{param_1_name}_and_{param_2_name}_scan_with_{svg}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    # Afficher la figure
    plt.show()

    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()
    
#%% a and R0 scan
# 2D matrix calculation


class inputs_management:
    
    def __init__(self, name_file):
        self.name_file = name_file
        
        
    def run(self):
        self.read_file()
        self.extract_inputs()
        
    def read_file(self):
        fic = open(self.name_file)
        self.inputs = fic.read()
        fic.close()
        print(self.inputs)
               
    def extract_input_scan(self, input):
        """
        exemple of input : R0 = [4.5 ; 5 ; 25]

        """
        params = input.split("=")
        param_name = params[0].strip() # strip remove space ("R0 " -> "R0") 
        
        params_value = params[1].strip().strip("[]") # " [4.5 ; 5 ; 25]" ->  "4.5 ; 5 ; 25"
        params_value = params_value.split(";")
        
        param_min = float(params_value[0].strip()) # str "4.5 " -> float 4.5 
        param_max = float(params_value[1].strip()) 
        param_n = float(params_value[2].strip()) 
        
        return [param_name, param_min, param_max, param_n] 
    
    def extract_input(self, input):
        split_ = input.split("=")
        input_name = split_[0].strip()
        input_value = split_[1].strip()
        
        try:
            input_value = float(input_value)
        except ValueError:
            pass # Error means input_value is a str (some inputs are str)

        return [input_name, input_value] 
        
        
    def extract_inputs(self):
        debug = False
        split_ = self.inputs.split("\n")
        
        if split_[-1] == "": # -1 check the last element 
            split_.pop() # remove the last element if the last element is empty  
            
        self.param_1 = self.extract_input_scan(split_[0])
        self.param_2 = self.extract_input_scan(split_[1])
      
        self.inputs = [] 
        for i in range(2, len(split_)):
            self.inputs.append(self.extract_input(split_[i]))

        if debug:
            print(self.param_1)
            print(self.param_2)
            print(self.inputs)
             

        
if __name__ == "__main__":
    
    file = "inputs_scan_2D.txt"
    manager = inputs_management(file)
    manager.run()
    
    D0fus_Scan_2D_generic(manager.param_1, manager.param_2, manager.inputs)
    
        
        
