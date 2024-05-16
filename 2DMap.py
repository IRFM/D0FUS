# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:08:01 2024

@author: TA276941
"""


#%% Variation of 1 chosen parameter and mapping the a space
# 2D matrix calculation


chosen_parameter = None
chosen_unity = None

# Ask the user to choose the parameter to vary
chosen_parameter = input("Choose the parameter to vary (H, Bmax, Pfus, Pw, fobj): ")

# Define ranges and steps for each parameter
parameter_ranges = {
    'H': np.arange(0.6, 2, 0.01),
    'Bmax': np.arange(5, 25, 0.2),
    'Pfus': np.arange(500.E6, 5000.E6, 100.E6),
    'Pw': np.arange(1.E6, 5.E6, 1.E5),
    'fobj' : np.arange(0.01,1.0,0.01)
}
unit_mapping = {'H': '', 'Bmax': 'T', 'Pfus': 'GW', 'Pw': 'MW/m²','fobj': ''}
chosen_unity = unit_mapping.get(chosen_parameter, '')
a_min_choice = {'H':0.5, 'Bmax':0.3, 'Pfus':0.5, 'Pw':0.5,'fobj':0.5}
a_min = a_min_choice.get(chosen_parameter,)
a_max_choice = {'H':2, 'Bmax':2.3, 'Pfus':3.5, 'Pw':4.7,'fobj':2}
a_max = a_max_choice.get(chosen_parameter,)
parameter_values = parameter_ranges.get(chosen_parameter)
a_vec = np.arange(a_min,a_max,0.05)
max_limits_density = np.zeros((len(a_vec),len(parameter_values)))
max_limits_security = np.zeros((len(a_vec),len(parameter_values)))
max_limits_beta = np.zeros((len(a_vec),len(parameter_values)))
max_limits_power = np.zeros((len(a_vec),len(parameter_values)))
radial_build_matrix = np.zeros((len(a_vec),len(parameter_values)))
max_limits_matrix = np.zeros((len(a_vec),len(parameter_values)))
Ip_matrix = np.zeros((len(a_vec),len(parameter_values)))
Q_matrix = np.zeros((len(a_vec),len(parameter_values)))
f_Pheating_matrix = np.zeros((len(a_vec),len(parameter_values)))

(H,P_fus,P_W,Bmax,κ)=init(CHOICE)

x = 0
y = 0

if parameter_values is not None:
    for parameter_value in tqdm(parameter_values, desc='Processing parameters'):
        
        # Update the chosen parameter
        if chosen_parameter == 'H':
            H = parameter_value
        elif chosen_parameter == 'Bmax':
            Bmax = parameter_value
        elif chosen_parameter == 'Pfus':
            P_fus = parameter_value
        elif chosen_parameter == 'Pw':
            P_W = parameter_value
        elif chosen_parameter == 'fobj':
            f_RF_objectif = parameter_value
        
        y = 0
        
        for a_solution in a_vec :
        
            # Calculate useful values
            (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution, H, Bmax, P_fus, P_W)
            
            # Verifier les conditions
            n_condition = n_vec_solution / nG_solution
            beta_condition = beta_solution / betaN
            q_condition = q / qstar_solution
            fRF_condition = fRF_solution / f_RF_objectif
            
            max_limit = max(n_condition, beta_condition, q_condition, fRF_condition)
            
            radial_build = np.nan
            max_limit_density = np.nan
            max_limit_security = np.nan
            max_limit_beta = np.nan
            max_limit_power = np.nan
            
            if not np.isnan(R0_a_b_c_CS_solution) and not np.isnan(max_limit) and max_limit<1 :
                radial_build = R0_solution
            
            # Création d'un tableau contenant les valeurs des conditions
            conditions = np.array([n_condition, beta_condition, q_condition, fRF_condition])
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
                elif indice_max_condition == 3:
                    # Action spécifique pour fRF_condition
                    max_limit_power = max_limit
                    
            # Store the value in the matrix
            radial_build_matrix[y,x] = radial_build        
            max_limits_density[y, x] = max_limit_density
            max_limits_power[y, x] = max_limit_power
            max_limits_security[y, x] = max_limit_security
            max_limits_beta[y, x] = max_limit_beta
            max_limits_matrix[y, x] = max_limit
            Ip_matrix[y,x] = Ip_solution
            Q_matrix[y,x] = heat
            f_Pheating_matrix[y,x] = fRF_solution
            
            y = y+1
            
        x = x+1
        

else:
    print("Invalid chosen parameter. Please choose from 'H', 'Bmax', 'Pfus', 'Pw'.")
    
#%% Plot 2D

taille_police_topological_map = 10
taille_police_legende = 15
taille_police_other = 15
taille_titre = 22
plt.rcParams.update({'font.size': taille_police_other})

# Inverser l'ordre des lignes de max_limits_matrix
inverted_matrix_density_limit = max_limits_density[::-1, :]
inverted_matrix_security_limit = max_limits_security[::-1, :]
inverted_matrix_beta_limit = max_limits_beta[::-1, :]
inverted_matrix_power_limit = max_limits_power[::-1, :]
# Inverser l'ordre des lignes de max_limits_matrix
inverted_matrix_radial_build = radial_build_matrix[::-1, :]
inverted_matrix_plasma_limit = max_limits_matrix[::-1, :]
inverted_Ip_matrix = Ip_matrix[::-1, :]
inverted_Ip_matrix = inverted_Ip_matrix/1e6 # MA
inverted_Q_matrix = Q_matrix[::-1, :]
inverted_f_Pheating_matrix = f_Pheating_matrix[::-1, :]
inverted_f_Pheating_matrix = inverted_f_Pheating_matrix*100 # Pourcentages

# Ask the user to choose the second topologic map :
chosen_topologic = input("Choose the second parameter (Ip, Q, fP): ")

# Créer une figure et un axe principal
fig, ax = plt.subplots(figsize=(10, 12))

if chosen_parameter == 'Pfus':
    svg = f"Bmax={Bmax}_Pw={P_W/1e6}_fP={f_RF_objectif*100}_scaling_law={law}"
    plt.title(f"$B_{{\mathrm{{max}}}}$ = {Bmax} [T] , $P_{{\mathrm{{W}}}}$ = {P_W/1e6} [MW/m²], $f_P$ = {f_RF_objectif*100} [%], scaling law : {law}",fontsize=taille_police_legende)
    chosen_parameter_to_plot = '$P_{fus}$'
    title_parameter = '$\mathbf{P_{fus}}$'
elif chosen_parameter == 'Pw':
    svg = f"Bmax={Bmax}_Pfus={P_fus/1e9}_fP={f_RF_objectif*100}_scaling_law={law}"
    plt.title(f"$B_{{\mathrm{{max}}}}$ = {Bmax} [T], $P_{{\mathrm{{fus}}}}$ = {P_fus/1e9} [GW], $f_P$ = {f_RF_objectif*100} [%], scaling law : {law}",fontsize=taille_police_legende)
    chosen_parameter_to_plot = '$P_W$'
    title_parameter = '$\mathbf{P_{W}}$'
elif chosen_parameter == 'Bmax':
    svg = f"Pw={P_W/1e6}_Pfus={P_fus/1e9}_fP={f_RF_objectif*100}_scaling_law={law}"
    plt.title(f"$P_{{\mathrm{{fus}}}}$ = {P_fus/1e9} [GW], $P_{{\mathrm{{W}}}}$ = {P_W/1e6} [MW/m²], $f_P$ = {f_RF_objectif*100} [%], scaling law : {law}",fontsize=taille_police_legende)
    chosen_parameter_to_plot = '$B_{max}$'
    title_parameter = '$\mathbf{B_{max}}$'
else :
    svg = f"Bmax={Bmax}_Pfus={P_fus/1e9}_Pw={P_W/1e6}_fP={f_RF_objectif*100}_scaling_law:{law}"
    plt.title(f"$B_{{\mathrm{{max}}}}$ = {Bmax} [T], $P_{{\mathrm{{fus}}}}$ = {P_fus/1e9} [GW], $P_{{\mathrm{{W}}}}$ = {P_W/1e6} [MW/m²], $f_P$ = {f_RF_objectif*100} [%], scaling law : {law}",fontsize=taille_police_legende)
    chosen_parameter_to_plot = 'Input parameter'
    
plt.suptitle(f"Parameter space : a , {title_parameter}", fontsize=taille_titre,y=0.94, fontweight='bold')

# Calculer le minimum et le maximum des valeurs numériques
min_matrix = 0.5
max_matrix = 2

# Choix des couleurs
color_choice_density = 'Blues'
color_choice_security = 'Greens'
color_choice_power = 'Greys'
color_choice_beta = 'Reds'

# Afficher les heatmap pour les matrices avec imshow
im_density = ax.imshow(inverted_matrix_density_limit, cmap=color_choice_density, aspect='auto', interpolation='nearest', norm=Normalize(vmin=min_matrix, vmax=max_matrix))
im_security = ax.imshow(inverted_matrix_security_limit, cmap=color_choice_security, aspect='auto', interpolation='nearest', norm=Normalize(vmin=min_matrix, vmax=max_matrix))
im_beta = ax.imshow(inverted_matrix_beta_limit, cmap=color_choice_beta, aspect='auto', interpolation='nearest', norm=Normalize(vmin=min_matrix, vmax=max_matrix))
im_power = ax.imshow(inverted_matrix_power_limit, cmap=color_choice_power, aspect='auto', interpolation='nearest', norm=Normalize(vmin=min_matrix, vmax=max_matrix))

# Coutour limite plasma = 1
threshold = 1.0
ax.contour(inverted_matrix_plasma_limit, levels=[threshold], colors='black', linestyles='dashed')

# Personnaliser les axes et le titre
plt.xlabel(f"{chosen_parameter_to_plot} [{chosen_unity}]")
plt.ylabel('a [m]')

### Color Bars

# Créer un axe supplémentaire en dessous de l'axe principal pour les colorbars
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("bottom", size="5%", pad=0.9)
cax2 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
cax3 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)
cax4 = divider.append_axes("bottom", size="5%", pad=0.1, sharex=cax1)

# Ajouter des annotations textuelles à côté des colorbars
cax1.annotate('n/$n_{\mathrm{G}}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
cax2.annotate(r'$\beta$/$\beta_{\text{T}}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
cax3.annotate('$q_{\mathrm{K}}$/$q_{\mathrm{*}}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
cax4.annotate('$f_{\mathrm{RF}}$/ $f_{\mathrm{obj}}$', xy=(-0.01, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)

# Créer les colorbars avec les orientations désirées
cbar_density = plt.colorbar(im_density, cax=cax1,orientation='horizontal')
tick_labels = cbar_density.ax.xaxis.get_ticklabels()
tick_labels[-1].set_visible(False)
cbar_beta = plt.colorbar(im_beta, cax=cax2, orientation='horizontal')
tick_labels = cbar_beta.ax.xaxis.get_ticklabels()
tick_labels[-1].set_visible(False)
cbar_security = plt.colorbar(im_security, cax=cax3, orientation='horizontal')
tick_labels = cbar_security.ax.xaxis.get_ticklabels()
tick_labels[-1].set_visible(False)
cbar_power = plt.colorbar(im_power, cax=cax4, orientation='horizontal')

### Definition des axes

# Axe y (a_vec)
y_indices = np.arange(0, len(a_vec), 4)  # Sélectionner les indices des ticks
ax.set_yticks(y_indices)  # Positions
ax.set_yticklabels(np.round((a_max+a_min)-a_vec[y_indices],2)) # Etiquettes

# Axe x (parameter_values)
x_indices = np.arange(0, len(parameter_values), 5)  # Sélectionner les indices
ax.set_xticks(x_indices)  # Positions
if chosen_parameter == 'Pfus':
    x_labels = [round(parameter_values[i])*1e-9 for i in x_indices]  # Arrondir chaque valeur à une decimale
elif chosen_parameter == 'Pw':
    x_labels = [round(parameter_values[i])*1e-6 for i in x_indices]  # Arrondir chaque valeur à une decimale
else :
    x_labels = [round(parameter_values[i]) for i in x_indices]  # Arrondir chaque valeur à une decimale
ax.set_xticklabels(x_labels, rotation=45, ha='right')  # Etiquettes

### Ajouter des contours

# Remplacer NaN par -1 et les autres par 1 dans la matrice pour les contours du radial build
filled_matrix = np.where(np.isnan(inverted_matrix_radial_build), -1, 1)
# Définir le niveau de contour pour les valeurs de transition
contour_level = [0]
# Tracer les contours autour des valeurs de transition
contour_radial_line = ax.contour(filled_matrix, levels=contour_level, colors='#555555')
# Tracer les isocontours de R0
contour_lines = ax.contour(inverted_matrix_radial_build, levels=np.arange(1, 25), colors='#555555')
ax.clabel(contour_lines, inline=True, fmt='%d', fontsize=taille_police_topological_map)
if chosen_topologic =='Ip':
    # Ip contours
    contour_lines_Ip = ax.contour(inverted_Ip_matrix, levels=np.arange(5, 45,5), colors='white')
    ax.clabel(contour_lines_Ip, inline=True, fmt='%d', fontsize=taille_police_topological_map)
    white_line = mlines.Line2D([], [], color='white', label='$I_p$ [MA]')
elif chosen_topologic == 'Q':
    # Q contours
    contour_lines_Q = ax.contour(inverted_Q_matrix, levels=np.arange(0, 1000 ,100), colors='white')
    ax.clabel(contour_lines_Q, inline=True, fmt='%d', fontsize=taille_police_topological_map)
    white_line = mlines.Line2D([], [], color='white', label='Q// [MW-T/m]')
elif chosen_topologic == 'fP' :
    # fPheating contours
    contour_lines_fP = ax.contour(inverted_f_Pheating_matrix, levels=np.arange(0, 800,10), colors='white')
    ax.clabel(contour_lines_fP, inline=True, fmt='%d', fontsize=taille_police_topological_map)
    white_line = mlines.Line2D([], [], color='white', label='$f_P$ [%]')
else :
    print('Wrong choice')
# Légende
grey_line = mlines.Line2D([], [], color='#555555', label='$R_0$ [m]')
black_dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='Plasma limits = 1')
ax.legend(handles=[grey_line,white_line, black_dashed_line], loc='upper right', facecolor='lightgrey', fontsize=taille_police_legende)

# Save the image
# Remplacer les virgules par des points dans chosen_parameter et svg
chosen_parameter = chosen_parameter.replace('.', ',')
svg = svg.replace('.', ',')
path_to_save = os.path.join(save_directory,f"a_and_{chosen_parameter}_scan_with_{svg}.png")
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
# Afficher la figure
plt.show()

# Reinitialisation des paramètres par defaut
plt.rcdefaults()
    
        
        
