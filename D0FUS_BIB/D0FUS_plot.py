# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:34:16 2024

@author: TA276941
"""
#%% Import

from D0FUS_run import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% Plot functions

# Hatch Function
def Plot_red_hatch_above_separator(ax, y_value=1.0):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ymax = 2
    ax.fill_betweenx([y_value, ymax], xmin, xmax, color='red', alpha=0.3, hatch='//')
    return(ymax)

def _invert(x, limits):
    #inverts a value x on a scale from limits[0] to limits[1]
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    # scales data[1:] to ranges[0], inverts if the scale is reversed
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                label="axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables)
        [txt.set_rotation(angle-90) for txt, angle in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x, 2)) for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
            gridlabel[0] = ""  # clean up origin
            ax.set_rgrids(grid[:-1], labels=gridlabel[:-1], angle=angles[i], color='grey')  # Changer la couleur ici
            ax.set_ylim(*ranges[i])
            # Changer la couleur des etiquettes de mark
            for label in ax.get_xticklabels():
                label.set_color('black')  # Changer la couleur ici
                label.set_fontsize(12)  # Changer la taille de la police ici
                label.set_zorder(10)  # Afficher les etiquettes au-dessus du graphique
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

def Plot_radial_build_aesthetic(lengths_upper, names_upper, lengths_lower, names_lower):
    # Create a figure and an axis
    fig, ax = plt.subplots()
    # Starting position for the dashes
    dash_position_upper = 0.0
    dash_position_lower = 0.0

    # Variable de commutation pour alterner les motifs de hachure
    hatch_switch = True
    # Iterate through the lengths and names for the upper line
    for length, name in zip(lengths_upper, names_upper):
        
        # Definir le motif de hachure en fonction de l'etat de la variable de commutation
        if hatch_switch:
            hatch = '//'
        else:
            hatch = '\\\\'  # Utilisez '\\\\' pour des hachures vers la droite
        if name !='':
            # Dashed rectangle avec le motif de hachure approprie
            ax.fill_betweenx([0.1, 0.3], dash_position_upper, dash_position_upper+length, alpha=0.3, hatch=hatch)
            # Plot the length with a different color
            ax.plot([dash_position_upper, dash_position_upper + length], [0.2, 0.2], label=name)
        else:
            ax.plot([dash_position_upper, dash_position_upper + length], [0.2, 0.2],color='black', label=name)
        # Inverser l'etat de la variable de commutation pour la prochaine iteration
        hatch_switch = not hatch_switch
        # Draw vertical dashes to separate the zones
        ax.vlines(dash_position_upper + length, ymin=0.1, ymax=0.3, colors='k')
        # Update the position for the dashes
        dash_position_upper += length
        # Position and display text
        text_position = dash_position_upper - length / 2.0
        ax.text(text_position, 0.3, name, ha='center', va='bottom', fontsize=12)

    # Iterate through the lengths and names for the inner line
    for length, name in zip(lengths_lower, names_lower):
        ax.plot([dash_position_lower, dash_position_lower + length], [0,0], label=name, color='black')
        ax.vlines(dash_position_lower + length, ymin=-0.1, ymax=0.3, colors='k', linestyles='dashed')
        dash_position_lower += length
        text_position = dash_position_lower - length / 2.0  # Position the text in the middle of each segment
        ax.text(text_position, -0.1, name, ha='center', va='bottom', fontsize=12)

    # Central Tick
    ax.vlines(0, ymin=-0.2, ymax=0.4, colors='k', linestyles='dashed')
    ax.text(0.2, 0.41, 'Central axis', ha='center', va='bottom', fontsize=12)
    # Modifier la taille de police de l'echelle sur les deux axes (x et y)
    plt.gca().tick_params(axis='both', labelsize=12)
    # Hide Y
    ax.yaxis.set_visible(False)
    # Show the graph
    plt.ylim(-0.2, 0.5)
    plt.xlim(-0.6, dash_position_upper+0.1)
    
    # Ajouter un titre à la figure
    plt.title("Radial Build", fontsize=14)
    # Save the image
    path_to_save = os.path.join(save_directory,"Radial_Build_Aesthetic.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()

# Testing
# lengths_upper = [1.2,0.5, 0.1, 1, 0.8, 2]
# names_upper = ['','CS','', 'TFC', 'Blanket', 'Plasma']
# lengths_lower = [4.6]
# names_lower = ['R0']
# Plot_radial_build_aesthetic(lengths_upper, names_upper, lengths_lower, names_lower)

def Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design):

    # Definir la taille de la police par defaut
    plt.rcParams.update({'font.size': 17})
    
    # Plot parameter evolution
    plt.figure(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Operational domain', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une decimale pour Bmax, Pfus, Pw et H, et à deux decimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'R0':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)

    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'R0':
        plt.xlabel(f"$R_{{\mathrm{{0}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
    plt.ylabel("Normalized Values")

    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle=':', label='First acceptable value')
        
    if chosen_design is not None:
        plt.axvline(x=chosen_design, color='red', linestyle=':', label='chosen design')
    plt.plot(parameter_values, n_solutions / nG_solutions, 'k-', label='n/$n_{\mathrm{G}}$')
    plt.plot(parameter_values, beta_solutions / betaN, 'r-', label= r'$\beta$/$\beta_{\text{T}}$')
    plt.plot(parameter_values, q / qstar_solutions, 'g-', label='$q_{\mathrm{K}}$ / $q_{\mathrm{*}}$')
    plt.plot(parameter_values, fRF_solutions/f_RF_objectif, color='blue', linestyle='--', label='$f_{\mathrm{RF}}$/ $f_{\mathrm{obj}}$')
    plt.xlim(min(parameter_values), max(parameter_values))
    # Hatch the red zone above the separator
    Plot_red_hatch_above_separator(plt.gca(), y_value=1.0)
    plt.ylim(0, 1.98)
    plt.legend()
    plt.grid()
    # Save the image
    path_to_save = os.path.join(save_directory,f"Operational_Domain_{chosen_parameter}_fRF={f_RF_objectif}_R0={R0}_Pfus={P_fus_rounded}_Bmax={Bmax_rounded}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()
    
def Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design):
    
    # Definir la taille de la police par defaut
    plt.rcParams.update({'font.size': 17})
    
    plt.figure(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Heat Parameter', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une decimale pour Bmax, Pfus, Pw et H, et à deux decimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'R0':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)

    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'R0':
        plt.xlabel(f"$R_{{\mathrm{{0}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
        
    plt.ylabel("Heat parameter [MW-T/m²]")

    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle=':', label='First acceptable value')
        
    if chosen_design is not None:
        plt.axvline(x=chosen_design, color='red', linestyle=':', label='chosen design')
    plt.plot(parameter_values, heat_solutions, 'k-')
    plt.xlim(min(parameter_values), max(parameter_values))
    plt.grid()
    # Save the image
    path_to_save = os.path.join(save_directory,f"Heat_Parameter_{chosen_parameter}_fRF={f_RF_objectif}_R0={R0}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()
    
def Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_list,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design):
    
    # Definir la taille de la police par defaut
    plt.rcParams.update({'font.size': 17})
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Radial Build', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une decimale pour Bmax, Pfus, Pw et H, et à deux decimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'R0':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
        
    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'R0':
        plt.xlabel(f"$R_{{\mathrm{{0}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
        
    plt.ylabel("Length [m]")  # Label pour l'axe y principal
    # Trace des donnees sur le premier axe y
    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle='--', label='First acceptable value')
    plt.plot(parameter_values, R0_list, color='green', label='$R_{\mathrm{0}}$')
    plt.plot(parameter_values, R0_a_solutions, color='blue', label='$R_{\mathrm{0}}$-a')
    plt.plot(parameter_values, R0_a_b_solutions, color='purple', label='$R_{\mathrm{0}}$-a-$\Delta_{blanket}$')
    plt.plot(parameter_values, R0_a_b_c_solutions, color='orange', label='$R_{\mathrm{0}}$-a-$\Delta_{blanket}$-$\Delta_{TFC}$')
    plt.plot(parameter_values, R0_a_b_c_CS_solutions, color='c', label='$R_{\mathrm{CSi}}$')
    plt.legend(loc='upper left', facecolor='lightgrey')
    # Ajouter un deuxième axe y pour Ip_solutions
    ax2 = ax1.twinx()
    ax2.set_ylabel('$I_{\mathrm{p}}$ [MA]', color='black')
    ax2.plot(parameter_values, Ip_solutions/1e6, color='red', linestyle='--' ,label='$I_{\mathrm{p}}$')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right', facecolor='lightgrey')
    
    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle=':', label='First acceptable value')
    if chosen_design is not None:
        plt.axvline(x=chosen_design, color='red', linestyle=':', label='chosen design')
    plt.xlim(min(parameter_values), max(parameter_values))
    # Enregistrer l'image
    path_to_save = os.path.join(save_directory, f"Radial_Build_{chosen_parameter}_fRF={f_RF_objectif}_R0={R0}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()
    
def Plot_cost_function(chosen_parameter,parameter_values,cost_solutions,first_acceptable_value,chosen_unity,chosen_design):
    
    # Definir la taille de la police par defaut
    plt.rcParams.update({'font.size': 17})
    
    # Plot cost function
    plt.figure(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Cost function', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une decimale pour Bmax, Pfus, Pw et H, et à deux decimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'R0':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}m Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
        
    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'R0':
        plt.xlabel(f"$R_{{\mathrm{{0}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
    plt.ylabel("$V_{\mathrm{T}}$/$P_{\mathrm{E}}$")
    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle=':', label='First acceptable value')
    if chosen_design is not None:
        plt.axvline(x=chosen_design, color='red', linestyle=':', label='chosen design')
    plt.plot(parameter_values, cost_solutions*1e6, 'k-')
    plt.xlim(min(parameter_values), max(parameter_values))
    plt.grid()
    # Save the image
    path_to_save = os.path.join(save_directory,f"Cost_Function_{chosen_parameter}_fRF={f_RF_objectif}_R0={R0}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()
    
def Plot_tableau_valeurs(H,P_fus,R0,Bmax,κ,chosen_design):

    a_solution = chosen_design
    # Calculate useful values
    (B0_solution, tauE_solution, Q_solution, Ip_solution, n_solution,
    beta_solution, qstar_solution, q95_solution, nG_solution, 
    P_CD, P_sep, P_Thresh, fRF_solution, cost, heat, Gamma_n,
    R0_a_solution, R0_a_b_solution, R0_a_b_c_solution, R0_a_b_c_d_solution) = calcul(a_solution, H, Bmax, P_fus, R0)
    # Donnees à afficher dans le tableau
    table_data = [
        ["$R_0$", R0, "m"],
        ["$a$", a_solution, "m"],
        ["$b$", b, "m"],
        ["$c$", c, "m"],
        ["$B_{tf}$", Bmax, "T"],
        ["$B_0$", B0_solution, "T"],
        ["$I_p$", Ip_solution/1e6, "MA"],
        ["cost", cost*1e6, "$hm^{3}/MW$"],
        ["Q//", heat, "MW-T/m"],
        ["$\\bar{p}$", round(pbar_solution/1e6,2), "MPa"],
        ["$\\bar{n}$", n_vec_solution, "$10^{20}/m^{3}$"],
        ["$n_G$", nG_solution, "$10^{20}/m^{3}$"],
        ["$\\bar{T}$", joules_to_kev(Tbar), "keV"],
        ["$\\tau_E$", tauE_solution, "s"],
        ["$\\beta$", beta_solution, "%"],
        ["$\\beta_N$", betaN, "%"],
        ["$f_{Pc}$", fRF_solution*100, "%"],
        ["$q_{*}$", qstar_solution, ""],
        ["$P_{W}$", P_W, "MW/m²"],
    ]
    # Affichage
    # Formater les valeurs numeriques avec un seul chiffre après la virgule
    for i in range(len(table_data)):
        if isinstance(table_data[i][1], float):
            table_data[i][1] = round(table_data[i][1], 2)
    # Creation d'un DataFrame Pandas
    df = pd.DataFrame(table_data, columns=["Variable", "Valeur", "Unite"])
    # Creer une liste des hauteurs de ligne avec une hauteur uniforme
    row_heights = [0.1] * len(df)
    # Create a figure
    fig, ax = plt.subplots(figsize=(4, 4.5))
    # Hide the axes
    ax.axis('off')
    # Create a table from the DataFrame with increased row heights
    mpl_table = table(ax, df, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
    # Format the table
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    # Center the text in each cell
    for key, cell in mpl_table._cells.items():
        cell.set_text_props(ha='center', va='center')
        
    taille_titre_principal = 11
    taille_sous_titre = 9
    plt.suptitle('Main parameters', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une decimale pour Bmax, Pfus, Pw et H, et à deux decimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $R_{{\mathrm{{0}}}}$={R0}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    # Save the image
    path_to_save = os.path.join(save_directory, f"Table for P_fus={int(P_fus/1e9)} f_obj={f_RF_objectif} R0={R0} Bmax={Bmax} H={H}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    # Display the figure
    plt.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()

# Test Tableau
# (H,P_fus,P_W,Bmax,κ)=init(CHOICE)
# Plot_tableau_valeurs(H,P_fus,P_W,Bmax,κ,1)

def Plot_radar_chart():

    # Utiliser le style "whitegrid" de Seaborn
    sns.set_style("dark")
    
    # Utiliser la palette de couleurs "husl"
    sns.set_palette("dark")
    
    # Vos donnees
    Initials = ['Pf', 'Pe', 'Q', 'R0', 'a', 'A', 'Kappa', 'B0', 'Bmax', 'Pw', 'H98y2', 'Ip']
    Unit = ['MW', 'MW', '', 'm', 'm', '', '', 'T', 'T', 'MW/m²', '', 'MA']
    data = [300, 250, 8, 10, 1.8, 5, 1.65, 5, 18, 0.4, 1.78, 22]  # Vos donnees reelles
    ITER = [500, 0, 10, 6.2, 2, 3.1, 1.78, 5.3, 12.5, 0.5, 1, 15]  # Les valeurs de reference
    
    # Creation des affichages
    mark = [f"{init}[{unit}]" if unit else init for init, unit in zip(Initials, Unit)]
    ranges = [(0, max(val, iter_val) * 1.5) for val, iter_val in zip(data, ITER)]
    
    # Plotting
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, mark, ranges)
    radar.plot(ITER, "-", lw=2, color="r", alpha=0.4, label="ITER")
    radar.fill(ITER, alpha=0.2)
    radar.plot(data, "-", lw=2, color="g", alpha=0.4, label="Data")
    radar.fill(data, alpha=0.2)
    radar.ax.legend()
    
    sns.set()
    
    # Save the image
    path_to_save = os.path.join(save_directory,"Radar.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()

def Plot_bar_chart():

    # Donnees reelles et valeurs de reference
    Initials = ['Pf', 'Pe', 'Q', 'R0', 'a', 'A', 'κ', 'B0', 'Bmax', 'Pw', 'H98y2', 'Ip']
    Unit = ['MW', 'MW', '', 'm', 'm', '', '', 'T', 'T', 'MW/m²', '', 'MA']
    data = [300, 250, 8, 5.5, 1.8, 3.5, 1.65, 5, 16, 0.4, 1.8, 13]  # Vos donnees reelles
    ITER = [500, 0, 10, 6.2, 2, 3.1, 1.78, 5.3, 12.5, 0.5, 1, 15]  # Les valeurs de reference
    
    # Creation des sous-plots pour chaque paramètre
    fig = make_subplots(rows=1, cols=len(Initials), subplot_titles=["" for _ in Initials])
    
    # Ajouter les barres pour chaque paramètre
    for i, param in enumerate(Initials):
        # Creation des echelles personnalisees
        y_range = [0, max(data[i], ITER[i])]
        
        fig.add_trace(go.Bar(
            x=['', ''],
            y=[data[i], ITER[i]],
            marker=dict(color=['rgba(0, 0, 0, 1)', 'rgba(173, 216, 230, 0)'], line=dict(color='rgba(0,0,0,0)')),  # Pas de contour
            legendgroup=param,
            showlegend=False,
            width=0.25,  # Largeur des barres
            offset=-0.2,  # Decalage pour centrer les barres
        ), row=1, col=i+1)
    
    
    
        # Ajouter le texte du nom du paramètre en dessous du graphique
        fig.add_annotation(
            x=0,
            y=-0.15,
            text=param,
            showarrow=False,
            font=dict(size=16, color='black'),  # Taille de la police et couleur bleu fonce
            xref=f"x{i+1}",
            yref="paper",
            align="center"
        )
        # Configuration de l'axe y pour chaque paramètre
        fig.update_yaxes(range=y_range, title_text="", showticklabels=True, row=1, col=i+1, color='black')  # Changer la couleur des traits de l'echelle à noir
        
    # Ajouter les barres pour chaque paramètre
    for i, param in enumerate(Initials):
        
        # Ajouter une bande rouge à l'extremite de la barre ITER
        fig.add_shape(type="rect",
                      xref=f'x{i+1}',
                      yref=f'y{i+1}',
                      x0=-0.2,
                      y0=ITER[i] - max(ITER[i], data[i])/200,  # Position de depart de la bande rouge
                      x1=0.05,
                      y1=ITER[i] + max(ITER[i], data[i])/200,  # Position de fin de la bande rouge
                      fillcolor='rgba(255, 0, 0, 1)',
                      layer="above",
                      line=dict(color="rgba(0,0,0,0)"),  # Pas de contour
                      legendgroup=param,
                      showlegend=False)
    
    # Configuration de la mise en page
    fig.update_layout(
        height=400,
        width=1200,
        title="Valeurs de donnees reelles et de reference pour chaque paramètre",
        barmode='group',
        showlegend=True,
        legend=dict(
            x=0,
            y=0,
            orientation="h",
            bgcolor='white'
        ),
        plot_bgcolor='white'  # Couleur de fond blanche
    )
    
    # Affichage de la figure
    fig.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()

def Plot_spider_chart():
    # Definition des fonctions
    functions = {
        'f_rho': (['nx'], ['rho', 'drho']),
        'f_Tprof': (['Tbar', 'nu_T', 'rho'], ['temp']),
        'f_nprof': (['nbar', 'nu_n', 'rho'], ['dens']),
        'f_power': (['P_fus'], ['P_E']),
        'f_pbar': (['Tbar', 'nu_n', 'nu_T', 'R0', 'a', 'κ', 'P_E', 'eta_T', 'nx'], ['pbar']),
        'f_beta': (['pbar', 'B0'], ['beta']),
        'f_betaT': (['Ip', 'a', 'B0'], ['betaT']),
        'f_nbar': (['pbar', 'Tbar', 'nu_n', 'nu_T'], ['nbar']),
        'f_sigmav': (['T'], ['sigmav']),
        'f_R0': (['a', 'P_fus', 'P_W', 'eta_T', 'κ'], ['R0']),
        'f_B0': (['Bmax', 'a', 'b', 'R0'], ['B0']),
        'f_nG': (['Ip', 'a'], ['nG']),
        'f_tauE': (['pbar', 'R0', 'a', 'κ', 'eta_T', 'P_E'], ['tauE']),
        'f_SL_param': (['SL_choice'], ['C_SL', 'alpha_I', 'alpha_R', 'alpha_a', 'alpha_kappa', 'alpha_n', 'alpha_B', 'alpha_M', 'alpha_P']),
        'f_Ip': (['SL_choice', 'H', 'tauE', 'R0', 'a', 'κ', 'nbar', 'B0', 'A', 'eta_T', 'P_E'], ['Ip']),
        'f_qstar': (['a', 'B0', 'R0', 'Ip', 'κ'], ['qstar']),
        'f_etaCD': (['a', 'R0', 'B0', 'nbar', 'Tbar', 'nu_n', 'nu_T', 'nx'], ['eta_CD']),
        'f_fB': (['eta_CD', 'R0', 'Ip', 'nbar', 'eta_RF', 'f_RP', 'f_RF', 'P_E'], ['f_B']),
        'f_fNC': (['a', 'κ', 'pbar', 'R0', 'Ip', 'nx'], ['f_NC']),
        'f_coil': (['a', 'b', 'R0', 'B0', 'Sigm_max', 'μ0', 'J_max'], ['c']),
        'f_cost_parameter': (['a', 'b', 'R0', 'c', 'κ', 'P_E'], ['VI_Pe']),
        'f_heat_parameter': (['B0', 'R0', 'P_E', 'eta_T'], ['Q']),
        'f_solenoide': (['R0', 'a', 'b', 'c'], ['sol']),
        'physical_constants': ([], ['E_ELEM', 'M_E', 'M_I', 'E_ALPHA', 'E_N', 'E_F', 'μ0', 'EPS_0']),
        'engineering_constants': ([], ['A', 'Tbar', 'κ', 'nu_n', 'nu_T', 'Sigm_max', 'J_max', 'eta_RF', 'f_RP', 'f_RF', 'eta_T', 'Slnd']),
        'input_variables': ([],['H', 'P_fus', 'P_W', 'Bmax']),
        'initialization_elements': ([], ['na', 'nx', 'a_min', 'a_max', 'a_pas', 'a_init', 'b', 'CHOICE', 'SL_choice'])
    }
    
    # Creation des listes de donnees pour les bulles
    nodes_x = []
    nodes_y = []
    node_texts = []
    
    # Creation des listes de donnees pour les liens entre fonctions
    edge_x = []
    edge_y = []
    edge_texts = []
    
    # Ajout des nœuds (fonctions) aux listes de donnees
    for function, (inputs, outputs) in functions.items():
        nodes_x.append(function)
        nodes_y.append(len(inputs))  # Utilisation de len(inputs) pour ajuster la hauteur de la bulle
        node_texts.append(f"Inputs: {', '.join(inputs)}<br>Outputs: {', '.join(outputs)}")
    
    # Ajout des arêtes (liens entre fonctions) aux listes de donnees
    for source_function, (source_inputs, source_outputs) in functions.items():
        for target_function, (target_inputs, _) in functions.items():
            common_vars = set(source_outputs) & set(target_inputs)
            if common_vars and source_function != target_function:
                for common_var in common_vars:
                    edge_x.append(source_function)
                    edge_x.append(target_function)
                    edge_x.append(None)
                    edge_y.append(len(source_inputs))
                    edge_y.append(len(target_inputs))
                    edge_y.append(None)
                    edge_texts.append(f"{source_function} -> {target_function}<br>Variable: {common_var}")
    
    # Creation du graphique avec bulles et liens
    fig = go.Figure()
    
    # Ajout des bulles
    fig.add_trace(go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers',
        marker=dict(
            size=15,
            color='black'
        ),
        text=node_texts,
        hoverinfo='text'
    ))
    
    # Ajout des liens
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1, color='black'),
        hoverinfo='text',
        text=edge_texts
    ))
    
    # Ajout des annotations pour afficher de manière permanente le nom de chaque fonction au centre, au-dessus du point
    for function, (inputs, outputs) in functions.items():
        annotation_y = len(inputs) + 0.2  # Ajustez la hauteur de l'annotation selon vos preferences
        fig.add_annotation(
            x=function,
            y=annotation_y,
            text=function,
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=8)
        )
    
    # Ajout du point representant les constantes physiques
    physical_constants = ['E_ELEM', 'M_E', 'M_I', 'E_ALPHA', 'E_N', 'E_F', 'μ0', 'EPS_0']
    fig.add_trace(go.Scatter(
        x=['physical_constants'] * len(physical_constants),
        y=[0] * len(physical_constants),
        mode='markers',
        marker=dict(
            size=15,
            color='red'
        ),
        text=physical_constants,
        hoverinfo='text'
    ))
    
    # Ajout du point representant les constantes ingenieurs
    engineering_constants = ['A', 'Tbar', 'κ', 'nu_n', 'nu_T', 'Sigm_max', 'J_max', 'eta_RF', 'f_RP', 'f_RF', 'eta_T', 'Slnd']
    fig.add_trace(go.Scatter(
        x=['engineering_constants'] * len(engineering_constants),
        y=[0] * len(engineering_constants),
        mode='markers',
        marker=dict(
            size=15,
            color='blue'
        ),
        text=engineering_constants,
        hoverinfo='text'
    ))
    
    # Ajout du point representant les entrees du code
    input_variables = ['H', 'P_fus', 'P_W', 'Bmax']
    fig.add_trace(go.Scatter(
        x=['input_variables'] * len(input_variables),
        y=[0] * len(input_variables),
        mode='markers',
        marker=dict(
            size=15,
            color='green'
        ),
        text=input_variables,
        hoverinfo='text'
    ))
    
    # Ajout du point representant les elements d'initialisation
    initialization_elements = ['na', 'nx', 'a_min', 'a_max', 'a_pas', 'a_init', 'b', 'CHOICE', 'SL_choice']
    fig.add_trace(go.Scatter(
        x=['initialization_elements'] * len(initialization_elements),
        y=[0] * len(initialization_elements),
        mode='markers',
        marker=dict(
            size=15,
            color='pink'
        ),
        text=initialization_elements,
        hoverinfo='text'
    ))
    
    # Personnalisation du layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title='Graph interactif des fonctions',
        width=800,  # Ajustez la largeur selon vos besoins
        height=800,  # Ajustez la hauteur selon vos besoins
        paper_bgcolor='white'  # Fond blanc
    )
    
    # Affichage du graphique
    fig.show()
    # Reinitialisation des paramètres par defaut
    plt.rcdefaults()
    
#%%

print("D0FUS_plot loaded")