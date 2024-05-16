# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:16:54 2024

@author: TA276941
"""

#%% Generation 3D

# Ask the user to choose the parameter to vary
chosen_parameter = input("Choose the parameter to slice (Pfus, Pw, a, Perso) : ")

(H,P_fus,P_W,Bmax,κ)=init(CHOICE)
Bmax = int(input("Choose the Magnetic field : "))

if chosen_parameter == 'Pw':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 100)
    P_fus_values = np.linspace(1e9, 4e9, 100)
    P_W_values = np.linspace(1e6, 4e6, 5)

elif chosen_parameter == 'Pfus':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 100)
    P_fus_values = np.linspace(1e9, 4e9, 5)
    P_W_values = np.linspace(1e6, 4e6, 100)

elif chosen_parameter == 'a':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 5)
    P_fus_values = np.linspace(1e9, 4e9, 100)
    P_W_values = np.linspace(1e6, 4e6, 100)
    
elif chosen_parameter == 'Perso':
    # Définir les plages des variables a, P_fus et P_W
    n_a = input("How many points for a : ")
    a_values = np.linspace(0.3, 4, n_a)
    n_Pfus = input("How many points for Pfus : ")
    P_fus_values = np.linspace(1e9, 4e9, n_Pfus)
    n_Pw = input("How many points for Pw : ")
    P_W_values = np.linspace(1e6, 4e6, n_Pw)
    
else :
    print("Wrong choice")

# Créer une grille 3D pour les valeurs de a, P_fus et P_W
a_grid, P_fus_grid, P_W_grid = np.meshgrid(a_values, P_fus_values, P_W_values, indexing='ij')

# Initialiser un tableau pour stocker les valeurs de max_limit
max_limits_density = np.zeros_like(a_grid)
max_limits_security = np.zeros_like(a_grid)
max_limits_beta = np.zeros_like(a_grid)
max_limits_power = np.zeros_like(a_grid)
radial_build_matrix = np.zeros_like(a_grid)
max_limits_matrix = np.zeros_like(a_grid)

# Calculer max_limit pour chaque combinaison de valeurs de a, P_fus et P_W
for i in tqdm(range(len(a_values)), desc='Processing parameters'):
    for j in range(len(P_fus_values)):
        for k in range(len(P_W_values)):
            (R0_solution, B0_solution, pbar_solution, beta_solution, nbar_solution,
             tauE_solution, Ip_solution, qstar_solution, nG_solution, eta_CD_solution,
             fB_solution, fNC_solution, fRF_solution, n_vec_solution, c, cost, heat,
             solenoid, R0_a_solution, R0_a_b_solution, R0_a_b_c_solution,
             R0_a_b_c_CS_solution, required_Bcs) = calcul(a_values[i], H, Bmax, P_fus_values[j], P_W_values[k])
            
            n_condition = n_vec_solution / nG_solution
            beta_condition = beta_solution / betaN
            q_condition = q / qstar_solution
            fRF_condition = fRF_solution / f_RF_objectif

            max_limit = max(n_condition, beta_condition, q_condition, fRF_condition)
            
            if max_limit > 1:
                max_limit = np.nan
            
            radial_build = np.nan
            max_limit_density = np.nan
            max_limit_security = np.nan
            max_limit_beta = np.nan
            max_limit_power = np.nan
            
            if not np.isnan(R0_a_b_c_CS_solution) and not np.isnan(max_limit) :
                radial_build = R0_solution
            
            # Création d'un tableau contenant les valeurs des conditions
            conditions = np.array([n_condition, beta_condition, q_condition, fRF_condition])
            # Indice de la condition la plus contraignante
            indice_max_condition = np.argmax(conditions)
            
            if not np.isnan(max_limit) and max_limit < 1 :
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
            radial_build_matrix[i,j,k] = radial_build        
            max_limits_density[i,j,k] = max_limit_density
            max_limits_power[i,j,k] = max_limit_power
            max_limits_security[i,j,k] = max_limit_security
            max_limits_beta[i,j,k] = max_limit_beta
            max_limits_matrix[i,j,k] = max_limit

# Chemin de sauvegarde spécifique
save_three_dimensions_path = r'C:\Users\TA276941\Desktop\D0FUS\Matrices\Scans 3D'
matrix_name = f'matrices_{chosen_parameter}_scaling_law={law}_Bmax={Bmax}.npz'
np.savez(os.path.join(save_three_dimensions_path, matrix_name), 
         radial_build_matrix=radial_build_matrix, 
         max_limits_density=max_limits_density, 
         max_limits_power=max_limits_power, 
         max_limits_security=max_limits_security, 
         max_limits_beta=max_limits_beta, 
         max_limits_matrix=max_limits_matrix)

#%% Affichage 3D complexe

taille_police_legende = 18
taille_police_other = 14
taille_titre = 24
plt.rcParams.update({'font.size': taille_police_other})

# Ask the user to choose the parameter to vary
chosen_parameter = input("Choose the parameter to slice (Pfus, Pw, a, Perso) : ")
Bmax = input("Choose the Magnetic field : ")
matrix_name = f'matrices_{chosen_parameter}_scaling_law={law}_Bmax={Bmax}.npz'
save_three_dimensions_path = r'C:\Users\TA276941\Desktop\D0FUS\Matrices\Scans 3D'

if chosen_parameter == 'Pw':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 100)
    P_fus_values = np.linspace(1e9, 4e9, 100)
    P_W_values = np.linspace(1e6, 4e6, 5)

elif chosen_parameter == 'Pfus':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 100)
    P_fus_values = np.linspace(1e9, 4e9, 5)
    P_W_values = np.linspace(1e6, 4e6, 100)

elif chosen_parameter == 'a':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 5)
    P_fus_values = np.linspace(1e9, 4e9, 100)
    P_W_values = np.linspace(1e6, 4e6, 100)
    
elif chosen_parameter == 'Perso':
    # Définir les plages des variables a, P_fus et P_W
    a_values = np.linspace(0.3, 4, 5)
    P_fus_values = np.linspace(1e9, 4e9, 100)
    P_W_values = np.linspace(1e6, 4e6, 100)
    
else :
    print("Wrong choice")

# Charger les matrices à partir du fichier
data = np.load(os.path.join(save_three_dimensions_path, matrix_name))
radial_build_matrix = data['radial_build_matrix']
max_limits_density = data['max_limits_density']
max_limits_power = data['max_limits_power']
max_limits_security = data['max_limits_security']
max_limits_beta = data['max_limits_beta']
max_limits_matrix = data['max_limits_matrix']

## Radial Build consideration

radial_build_filtred = np.zeros_like(radial_build_matrix)
# Parcourir radial_build_matrix
for i in range(radial_build_matrix.shape[0]):
    for j in range(radial_build_matrix.shape[1]):
        for k in range(radial_build_matrix.shape[2]):
            # Vérifier si le point contient une valeur non-NaN
            if not np.isnan(radial_build_matrix[i, j, k]):
                # Définir la distance de détection dans les directions i et k
                distance_detection_i = 1
                distance_detection_k = 1
                distance_detection_j = 1
                if chosen_parameter == 'Pw':
                    # Extraire le cube centré autour du point (i, j, k)
                    cube = radial_build_matrix[max(0, i-distance_detection_i):min(radial_build_matrix.shape[0], i+distance_detection_i+1),
                                               max(0, j-distance_detection_j):min(radial_build_matrix.shape[1], j+distance_detection_j+1),
                                               k]
                elif chosen_parameter == 'Pfus':
                    # Extraire le cube centré autour du point (i, j, k)
                    cube = radial_build_matrix[max(0, i-distance_detection_i):min(radial_build_matrix.shape[0], i+distance_detection_i+1),
                                               j,
                                               max(0, k-distance_detection_k):min(radial_build_matrix.shape[2], k+distance_detection_k+1)]
                elif chosen_parameter == 'a':
                    # Extraire le cube centré autour du point (i, j, k)
                    cube = radial_build_matrix[i,
                                               max(0, j-distance_detection_j):min(radial_build_matrix.shape[1], j+distance_detection_j+1),
                                               max(0, k-distance_detection_k):min(radial_build_matrix.shape[2], k+distance_detection_k+1)]
                else :
                    # Extraire le cube centré autour du point (i, j, k)
                    cube = radial_build_matrix[max(0, i-distance_detection_i):min(radial_build_matrix.shape[0], i+distance_detection_i+1),
                                               max(0, j-distance_detection_j):min(radial_build_matrix.shape[1], j+distance_detection_j+1),
                                               max(0, k-distance_detection_k):min(radial_build_matrix.shape[2], k+distance_detection_k+1)]
                # Compter le nombre de valeurs NaN dans le cube
                nan_neighbors = np.isnan(cube).sum()
                # Vérifier si au moins 3 voisins sont NaN
                if nan_neighbors >= 3:
                    # Assigner la valeur du point à radial_build_filtred
                    radial_build_filtred[i, j, k] = 0
                else:
                    radial_build_filtred[i, j, k] = np.nan
            else:
                radial_build_filtred[i, j, k] = np.nan


# Définir la tolérance autour des nombres entiers
# tolerance = 0.5
# # Créer une copie de radial_build_matrix
# radial_build_filtred = radial_build_matrix.copy()
# # Appliquer la condition pour remplacer les valeurs
# condition = np.abs(radial_build_matrix - np.round(radial_build_matrix)) > tolerance
# radial_build_filtred[condition] = np.nan
# Valeur arbitraire à utiliser pour écraser les valeurs

unbalanced_value = 0
# Créer des masques pour les valeurs nulles de radial_build_filtred
mask_zeros = (radial_build_filtred == 0)
# Écraser les valeurs correspondantes dans max_limits_density, max_limits_power, max_limits_security, et max_limits_beta
max_limits_density[mask_zeros] = unbalanced_value
max_limits_power[mask_zeros] = unbalanced_value
max_limits_security[mask_zeros] = unbalanced_value
max_limits_beta[mask_zeros] = unbalanced_value

## 3D core

# Créer une grille 3D pour les valeurs de a, P_fus et P_W
a_grid, P_fus_grid, P_W_grid = np.meshgrid(a_values, P_fus_values, P_W_values, indexing='ij')

# Choix des couleurs
color_choice_density = 'Blues'
color_choice_security = 'Greens'
color_choice_power = 'Greys'
color_choice_beta = 'Reds'
colors = [(1, 1, 1)]  # Blanc
color_radial_build = mcolors.ListedColormap(colors)

# Calculer le minimum et le maximum des valeurs numériques
min_colorbar = 0.5
max_colorbar = 1

# Créer une figure avec deux subplots, en spécifiant la taille relative des subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [10, 1]})

# Premier subplot : Affichage 3D

ax1.axis('off') # Supprimer les axes
ax1 = fig.add_subplot(211, projection='3d')
# Tracer les points en utilisant les valeurs calculées comme couleur
# scatter_radial = ax1.scatter(P_W_grid.flatten(), P_fus_grid_radial.flatten(), a_grid.flatten(), c=radial_build_filtred.flatten(), cmap=color_radial_build,alpha=0.7,zorder=0)  #,s=1 
scatter_density = ax1.scatter(P_W_grid.flatten()/1e6, P_fus_grid.flatten()/1e9, a_grid.flatten(), c=max_limits_density.flatten(), cmap=color_choice_density,vmin=min_colorbar,vmax=max_colorbar,alpha=1)
scatter_beta = ax1.scatter(P_W_grid.flatten()/1e6, P_fus_grid.flatten()/1e9, a_grid.flatten(), c=max_limits_beta.flatten(), cmap=color_choice_beta,vmin=min_colorbar,vmax=max_colorbar,alpha=1)
scatter_security = ax1.scatter(P_W_grid.flatten()/1e6, P_fus_grid.flatten()/1e9, a_grid.flatten(), c=max_limits_security.flatten(), cmap=color_choice_security,vmin=min_colorbar,vmax=max_colorbar,alpha=1)
scatter_power = ax1.scatter(P_W_grid.flatten()/1e6, P_fus_grid.flatten()/1e9, a_grid.flatten(), c=max_limits_power.flatten(), cmap=color_choice_power,vmin=min_colorbar,vmax=max_colorbar,alpha=1)

# Légende
white_dashed_line = mlines.Line2D([], [], color='white', linestyle='--', label='Radial Build')
ax1.legend([white_dashed_line], ['Radial Build'], loc='upper right', fontsize=taille_police_legende, facecolor='lightgrey', bbox_to_anchor=(1.1, 0.93))
ax1.set_xlabel('$P_W$ [MW/m²]', labelpad=10)
ax1.set_ylabel('$P_{fus}$ [GW]', labelpad=10)
ax1.set_zlabel('a [m]', labelpad=10)

# Deuxième subplot : Colorbars

ax2.axis('off')  # Supprimer les axes
# Créer un axe supplémentaire en dessous de l'axe principal pour les colorbars
divider = make_axes_locatable(ax2)
cax1 = divider.append_axes("bottom", size="5%", pad=6)
cax2 = divider.append_axes("bottom", size="5%", pad=0.9, sharex=cax1)
cax3 = divider.append_axes("bottom", size="5%", pad=0.9, sharex=cax1)
cax4 = divider.append_axes("bottom", size="5%", pad=0.9, sharex=cax1)
# Ajouter des annotations textuelles à côté des colorbars
cax1.annotate('n/$n_{\mathrm{G}}$', xy=(1.09, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
cax2.annotate(r'$\beta$/$\beta_{\text{T}}$', xy=(1.09, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
cax3.annotate('$q_{\mathrm{K}}$/$q_{\mathrm{*}}$', xy=(1.1, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
cax4.annotate('$f_{\mathrm{RF}}$/ $f_{\mathrm{obj}}$', xy=(1.12, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=taille_police_legende)
# Créer les colorbars avec les orientations désirées
power_color = 2
# Créer un dégradé de couleurs allant du noir au blanc
gradient_colors = mcolors.LinearSegmentedColormap.from_list('custom', ['black', 'white'])
cbar_density = plt.colorbar(scatter_density, cax=cax1, orientation='horizontal')
# Obtenir les étiquettes de la légende de la colorbar
tick_labels = cbar_density.ax.xaxis.get_ticklabels()
# Pour chaque étiquette de la légende, définir la couleur en fonction de la valeur
for label in tick_labels:
    value = float(label.get_text())  # Récupérer la valeur de l'étiquette
    normalized_value = ((value - min_colorbar) / (max_colorbar - min_colorbar))**power_color  # Normaliser la valeur entre 0 et 1
    color = gradient_colors(normalized_value)  # Obtenir la couleur correspondante
    label.set_color(color)  # Définir la couleur de l'étiquette
cbar_beta = plt.colorbar(scatter_beta, cax=cax2, orientation='horizontal')
# Obtenir les étiquettes de la légende de la colorbar
tick_labels = cbar_beta.ax.xaxis.get_ticklabels()
# Pour chaque étiquette de la légende, définir la couleur en fonction de la valeur
for label in tick_labels:
    value = float(label.get_text())  # Récupérer la valeur de l'étiquette
    normalized_value = ((value - min_colorbar) / (max_colorbar - min_colorbar))**power_color  # Normaliser la valeur entre 0 et 1
    color = gradient_colors(normalized_value)  # Obtenir la couleur correspondante
    label.set_color(color)  # Définir la couleur de l'étiquette
cbar_security = plt.colorbar(scatter_security, cax=cax3, orientation='horizontal')
# Obtenir les étiquettes de la légende de la colorbar
tick_labels = cbar_security.ax.xaxis.get_ticklabels()
# Pour chaque étiquette de la légende, définir la couleur en fonction de la valeur
for label in tick_labels:
    value = float(label.get_text())  # Récupérer la valeur de l'étiquette
    normalized_value = ((value - min_colorbar) / (max_colorbar - min_colorbar))**power_color  # Normaliser la valeur entre 0 et 1
    color = gradient_colors(normalized_value)  # Obtenir la couleur correspondante
    label.set_color(color)  # Définir la couleur de l'étiquette
cbar_power = plt.colorbar(scatter_power, cax=cax4, orientation='horizontal')
# Obtenir les étiquettes de la légende de la colorbar
tick_labels = cbar_power.ax.xaxis.get_ticklabels()
# Pour chaque étiquette de la légende, définir la couleur en fonction de la valeur
for label in tick_labels:
    value = float(label.get_text())  # Récupérer la valeur de l'étiquette
    normalized_value = ((value - min_colorbar) / (max_colorbar - min_colorbar))**power_color  # Normaliser la valeur entre 0 et 1
    color = gradient_colors(normalized_value)  # Obtenir la couleur correspondante
    label.set_color(color)  # Définir la couleur de l'étiquette
    
# Title
fig.text(0.55, 0.9, "Parameter space : a , $\mathbf{P_{fus}}$ , $\mathbf{P_{W}}$", fontsize=taille_titre, ha='center', va='center', fontweight='bold')
fig.text(0.55, 0.86, f"$B_{{\mathrm{{max}}}}$ = {Bmax} [T], $f_P$ = {f_RF_objectif*100} [%], scaling law : {law}", fontsize=taille_police_legende, ha='center', va='center')

# Save the image
path_to_save = os.path.join(save_directory,f'Pfus_Pw_a_scan_with_Bmax={int(Bmax)}_fp={f_RF_objectif*100}_scaling law={law}.png')
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')

plt.show()

# Reinitialisation des paramètres par defaut
plt.rcdefaults()

#%% Affichage 3D Radial build

color_choice_radial = 'coolwarm'

# Créer une figure 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# title
plt.suptitle(f"3D Map Radial Build : Bmax={Bmax}T", fontsize=18, fontweight='bold',y=0.9)

# Tracer les points en utilisant les valeurs calculées comme couleur
scatter = ax.scatter(P_W_grid.flatten(), P_fus_grid.flatten(), a_grid.flatten(), c=radial_build_matrix.flatten(), cmap=color_choice_radial,alpha=0.5)

# Ajouter une barre de couleur
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Radial Build')

# Définir les labels des axes
ax.set_xlabel('P_W')
ax.set_ylabel('P_fus')
ax.set_zlabel('a')

# Save the image
path_to_save = os.path.join(save_directory,f'3D_map_Radial_build')
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
# Afficher la figure
plt.show()

# Reinitialisation des paramètres par defaut
plt.rcdefaults()


#%% Affichage 3D simple

color_choice_plasma = 'coolwarm'

# Créer une figure 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

plt.suptitle(f"3D Map Plasma limits : Bmax={Bmax}T", fontsize=18, fontweight='bold',y=0.9)

# Tracer les points en utilisant les valeurs calculées comme couleur
scatter = ax.scatter(P_W_grid.flatten(), P_fus_grid.flatten(), a_grid.flatten(), c=max_limits_matrix.flatten(), cmap=color_choice_plasma,alpha=0.8)

# Ajouter une barre de couleur
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Valeur de max_limit')

# Définir les labels des axes
ax.set_xlabel('P_W')
ax.set_ylabel('P_fus')
ax.set_zlabel('a')

# Save the image
path_to_save = os.path.join(save_directory,f'3D_map')
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
# Afficher la figure
plt.show()

# Reinitialisation des paramètres par defaut
plt.rcdefaults()

#%% Affichage 3D filtré

# Trouver les indices où max_limit est le plus proche de 1 avec une tolérance de 0.05
indices = np.where(np.isclose(max_limit_grid, 1, atol=0.01))

# Extraire les coordonnées correspondantes de a, P_fus et P_W pour ces indices
a_points = a_grid[indices]
P_fus_points = P_fus_grid[indices]
P_W_points = P_W_grid[indices]

# Créer une figure 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracer les points où max_limit est le plus proche de 1
ax.scatter(P_W_points, P_fus_points, a_points, c='b', marker='o', label='max_limit ≈ 1')

# Définir les labels des axes
ax.set_xlabel('P_W')
ax.set_ylabel('P_fus')
ax.set_zlabel('a')

# Afficher la figure
plt.show()



