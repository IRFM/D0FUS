# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:57:08 2024

@author: TA276941
"""

#%% 2D scan function

def Two_Dimensions_Scan_Database(P,Bmax,chosen_parameter):
    
    (H,P_fus_test,P_W_test,Bmax_test,κ)=init(CHOICE)
    chosen_unity = None
    f_RF_objectif = 1

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
    
    if chosen_parameter == 'Pw':
        P_fus = P
    elif chosen_parameter == 'Pfus':
        P_W = P
    else :
        print('Error, wrong parameter')

    x = 0
    y = 0

    if parameter_values is not None:
        for parameter_value in parameter_values:
            
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
        
    plt.ioff()
    
    chosen_topologic_Database = ['Ip','Q','fP']
    for chosen_topologic in chosen_topologic_Database:
    
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
        path_to_save = os.path.join(new_directory_path,f"a_and_{chosen_parameter}_scan_with_{chosen_topologic}_map_and_parameters={svg}.png")
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    
        # Reinitialisation des paramètres par defaut
        plt.rcdefaults()
        
        # Fermer toutes les figures précédemment créées
        plt.close('all')

#%% 3D scan function

def Three_Dimensions_Scan_Database(Bmax):

    chosen_parameter_3D_database = ['Pfus','Pw']
    for chosen_parameter in chosen_parameter_3D_database:
        
        # Chemin de sauvegarde spécifique
        save_three_dimensions_path = r'C:\Users\TA276941\Desktop\D0FUS\Matrices\Scans 3D'
        matrix_name = f'matrices_{chosen_parameter}_scaling_law={law}_Bmax={Bmax}.npz'
        matrix_path = os.path.join(save_three_dimensions_path, matrix_name)
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(matrix_path):
            print("Le fichier matrice existe déjà")
            
            # Charger les matrices à partir du fichier
            data = np.load(matrix_path)
            radial_build_matrix = data['radial_build_matrix']
            max_limits_density = data['max_limits_density']
            max_limits_power = data['max_limits_power']
            max_limits_security = data['max_limits_security']
            max_limits_beta = data['max_limits_beta']
            max_limits_matrix = data['max_limits_matrix']
            
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
        
        else :

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
            for i in range(len(a_values)):
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
            np.savez(matrix_path, 
                     radial_build_matrix=radial_build_matrix, 
                     max_limits_density=max_limits_density, 
                     max_limits_power=max_limits_power, 
                     max_limits_security=max_limits_security, 
                     max_limits_beta=max_limits_beta, 
                     max_limits_matrix=max_limits_matrix)
        
        plt.ioff()
        
        taille_police_legende = 18
        taille_police_other = 14
        taille_titre = 24
        plt.rcParams.update({'font.size': taille_police_other})

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
        path_to_save = os.path.join(new_directory_path,f'Pfus_Pw_a_scan_{chosen_parameter}_slices_Bmax={int(Bmax)}_fp={f_RF_objectif*100}_scaling law={law}.png')
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')

        # Reinitialisation des paramètres par defaut
        plt.rcdefaults()
        
        # Fermer toutes les figures précédemment créées
        plt.close('all')
    
#%% Core

# Demander à l'utilisateur la version du fichier
chosen_number_folder_name = input("Version : ")

# Demander à l'utilisateur de choisir une sclaing law
law = input("Entrez la scaling law à etudier : (IPB98(y,2),ITPA20-IL,ITPA20,DS03,L-mode,L-mode OK,ITER89-P) :")

# Initialisation scaling law
(C_SL,alpha_delta,alpha_M,alpha_kappa,alpha_epsilon,alpha_R,alpha_B,alpha_n,alpha_I,alpha_P) = get_parameter_value_scaling_law(law)

Bmax_Database = [14,22]

# Répertoire parent où vous souhaitez créer le nouveau répertoire
parent_directory = r'C:\Users\TA276941\Desktop\D0FUS\Graphs\Base de donnée 3D'
# Nom du nouveau répertoire
new_directory_name = f'V{chosen_number_folder_name}'
# Chemin complet du nouveau répertoire
new_directory_path = os.path.join(parent_directory, new_directory_name)
# Créer le nouveau répertoire s'il n'existe pas déjà
if not os.path.exists(new_directory_path):
    os.makedirs(new_directory_path)
    print(f"Le répertoire {new_directory_name} a été créé avec succès.")
else:
    print(f"Le répertoire {new_directory_name} existe déjà.")

# Répertoire parent où vous souhaitez créer le nouveau répertoire
parent_directory = f'C:\\Users\\TA276941\\Desktop\\D0FUS\\Graphs\\Base de donnée 3D\\V{chosen_number_folder_name}'
# Nom du nouveau répertoire
new_directory_name = f'{law}'
# Chemin complet du nouveau répertoire
new_directory_path = os.path.join(parent_directory, new_directory_name)
# Créer le nouveau répertoire s'il n'existe pas déjà
if not os.path.exists(new_directory_path):
    os.makedirs(new_directory_path)
    print(f"Le répertoire {new_directory_name} a été créé avec succès.")
else:
    print(f"Le répertoire {new_directory_name} existe déjà.")

(H,P_fus_test,P_W_test,Bmax_test,κ)=init(CHOICE)

for Bmax in Bmax_Database :
    
    # Répertoire parent où vous souhaitez créer le nouveau répertoire
    parent_directory = f'C:\\Users\\TA276941\\Desktop\\D0FUS\\Graphs\\Base de donnée 3D\\V{chosen_number_folder_name}\\{law}'
    # Nom du nouveau répertoire
    new_directory_name = f'Bmax={Bmax}T'
    # Chemin complet du nouveau répertoire
    new_directory_path = os.path.join(parent_directory, new_directory_name)
    # Créer le nouveau répertoire s'il n'existe pas déjà
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
        print(f"Le répertoire {new_directory_name} a été créé avec succès.")
    else:
        print(f"Le répertoire {new_directory_name} existe déjà.")
        
    ### 3D scan
    
    Three_Dimensions_Scan_Database(Bmax)
    
    ### Pfus scan
    
    # Répertoire parent où vous souhaitez créer le nouveau répertoire
    parent_directory = f'C:\\Users\\TA276941\\Desktop\\D0FUS\\Graphs\\Base de donnée 3D\\V{chosen_number_folder_name}\\{law}\\Bmax={Bmax}T'
    # Nom du nouveau répertoire
    new_directory_name = 'Pfus_scan'
    # Chemin complet du nouveau répertoire
    new_directory_path = os.path.join(parent_directory, new_directory_name)
    # Créer le nouveau répertoire s'il n'existe pas déjà
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
        print(f"Le répertoire {new_directory_name} a été créé avec succès.")
    else:
        print(f"Le répertoire {new_directory_name} existe déjà.")
        
    Pw_Database =  [1e6,2e6,3e6,4e6,5e6]
    for P_W in Pw_Database:
        Two_Dimensions_Scan_Database(P_W,Bmax,'Pfus')
        
    ### Pw scan
    
    # Répertoire parent où vous souhaitez créer le nouveau répertoire
    parent_directory = f'C:\\Users\\TA276941\\Desktop\\D0FUS\\Graphs\\Base de donnée 3D\\V{chosen_number_folder_name}\\{law}\\Bmax={Bmax}T'
    # Nom du nouveau répertoire
    new_directory_name = 'Pw_scan'
    # Chemin complet du nouveau répertoire
    new_directory_path = os.path.join(parent_directory, new_directory_name)
    # Créer le nouveau répertoire s'il n'existe pas déjà
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
        print(f"Le répertoire {new_directory_name} a été créé avec succès.")
    else:
        print(f"Le répertoire {new_directory_name} existe déjà.")

    Pfus_Database = [0.5e9,1e9,2e9,3e9,4e9,5e9]
    for P_fus in Pfus_Database:
        Two_Dimensions_Scan_Database(P_fus,Bmax,'Pw')
        
    
    
    
    
    