# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:36:38 2024

@author: TA276941
"""

#%% Import
import sys
import os
import importlib

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

if "D0FUS_parameterization" in sys.modules:
    # Le module est déjà importé, le recharger
    importlib.reload(sys.modules['D0FUS_parameterization'])
    # Mettre à jour les variables globales
    globals().update({k: getattr(sys.modules['D0FUS_parameterization'], k) for k in dir(sys.modules['D0FUS_parameterization']) if not k.startswith('_')})
    importlib.reload(sys.modules['D0FUS_core'])
    importlib.reload(sys.modules['D0FUS_plot'])
    print("D0FUS Reloaded")
else:
    # Le module n'est pas encore importé, l'importer
    from D0FUS_initialisation import *
    print("D0FUS loaded")
    
# Attention, lors de toute modification de fichiers exterieurs, il est conseillé, malgré l'implémentation de la fonction reaload, de redémarrer le Kernel.
# Python est capricieux avec le stockage des variables et seul un redmérage du kernel puis import (de nouveau) de labibliothèque est sur à 100%
# Le rechargement d'un module (importlib.reload) ne met en effet pas forcément à jour les références existantes dans d'autres modules importés précédemment
# La fonction reload est cependant , du moins dans les versions python de 3.1 à 3.3 permettre de palier à ce problème

#%% Gradient Descent of Parameters to find the optimum of the Loss Function

Grad_choice = 'H_fixed'

def gradient_descent(min_a,max_a,min_H,max_H,min_B,max_B,min_P,max_P,min_R,max_R):
    
    # Utilisation de la fonction pour initialiser les listes
    (a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions,P_W_solutions) = initialize_lists()
    
    try:
        
        if Grad_choice == 'All_free':
            bounds = [(min_a, max_a),(min_H,max_H), (min_B, max_B),(min_P,max_P),(min_R,max_R)] # a,H,Bmax,Pfus,Pw
        elif Grad_choice == 'H_fixed':
            H = 1 # H fixed at 1 also in DàFUS core
            bounds = [(min_a, max_a), (min_B, max_B),(min_P,max_P),(min_R,max_R)] # a,H,Bmax,Pfus,Pw
        elif Grad_choice == 'H_P_fixed':
            H = 1 # H fixed at 1 also in DàFUS core
            P_fus = 2e9
            bounds = [(min_a, max_a), (min_B, max_B),(min_R,max_R)] # a,H,Bmax,Pfus,Pw
        else :
            print('Choose a gradient choice')
        
        # Cost function to minimize
        def objective_function(x):

            cost_fct = 0    

            if Grad_choice == 'All_free':
                # x[0] corresponds to a, x[1] to H, x[2] to Bmax, x[3] to Pfus, and x[4] to Pw
                a , H , Bmax , P_fus , R0 = x
            elif Grad_choice == 'H_fixed':
                # x[0] corresponds to a, x[1] to Bmax, x[2] to Pfus, and x[3] to Pw
                a , Bmax , P_fus , R0 = x
                H = 1
            elif Grad_choice == 'H_P_fixed':
                # x[0] corresponds to a, x[1] to Bmax, x[2] to Pfus, and x[3] to Pw
                a , Bmax , R0 = x
                H = 1
                P_fus = 2e9
                
            # Calculate useful values
            (P_W,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Q_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a, H, Bmax, P_fus, R0)

            # Consideration of limits
            if n_vec_solution / nG_solution > 1:
                cost_fct = cost_fct + (n_vec_solution / nG_solution)**10
            if beta_solution / betaN > 1:
                cost_fct = cost_fct + (beta_solution / betaN)**10
            if q / qstar_solution > 1:
                cost_fct = cost_fct + (q / qstar_solution)**10
            if fRF_solution / f_RF_objectif > 1:
                cost_fct = cost_fct + (fRF_solution / f_RF_objectif)**10
            # if solenoid < Slnd:
            #     cost_fct = cost_fct + ((solenoid*solenoid+1)**1000)
            # solenoid consideration
            if np.isnan(R0_a_b_c_CS_solution) or R0_a_b_c_CS_solution<0 :
                cost_fct = cost_fct + 10
                
            # Minimization of the cost
            cost_fct = (cost_fct + (cost*1e7))
            
            # naN verification
            if np.isnan(cost_fct)==True or cost_fct is None:
                cost_fct = 100
            
            return cost_fct
        
        result = differential_evolution(objective_function, bounds, maxiter=max_iterations)
        
        if not result.success:
                raise ValueError(f"Differential evolution did not converge: {result.message}")
        
        if Grad_choice == 'All_free':
            a_solution = result.x[0]
            H = result.x[1]
            Bmax = result.x[2]
            P_fus = result.x[3]
            R0 = result.x[4]
        elif Grad_choice == 'H_fixed':
            a_solution = result.x[0]
            H = 1
            Bmax = result.x[1]
            P_fus = result.x[2]
            R0 = result.x[3]
        elif Grad_choice == 'H_P_fixed':
            a_solution = result.x[0]
            H = 1
            Bmax = result.x[1]
            P_fus = 2e9
            R0 = result.x[2]
        else :
            print('Choose a gradient descent parameter')
        
        
        # Calculate useful values
        (P_W,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Q_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution,H,Bmax,P_fus,R0)
        
        print("With a cost =", cost * 10**6)
        # Call the a_variation function
        (a_solutions,nG_solutions,n_solutions,beta_solutions,qstar_solutions,fB_solutions,fNC_solutions,cost_solutions,heat_solutions,c_solutions,sol_solutions,P_W_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,Ip_solutions,fRF_solutions) = Variation_a(H, Bmax, P_fus, R0)
        
        # Plot with respect to 'a'
        chosen_parameter = 'a'
        parameter_values = a_solutions
        chosen_unity = 'm'
        chosen_design = a_solution
        first_acceptable_value = None
        
        Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design)
        
        Plot_cost_function(chosen_parameter,parameter_values,cost_solutions,first_acceptable_value,chosen_unity,chosen_design)
        
        Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design)
        
        R0_list = [R0] * len(R0_a_solutions)
        
        Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_list,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design)
        
        Plot_tableau_valeurs(H,P_fus,R0,Bmax,κ,chosen_design)
        
        # Plot Radial Build aesthetic
        lengths_upper = [R0_a_b_c_CS_solution,solenoid-R0_a_b_c_CS_solution, 0.1, c, b, 2*a_solution]
        names_upper = ['','CS','', 'TFC', 'Blanket', 'Plasma']
        lengths_lower = [R0]
        names_lower = ['R0']
        Plot_radial_build_aesthetic(lengths_upper, names_upper, lengths_lower, names_lower)
        
        if Grad_choice == 'All_free':
            # Pe = Pe Freidberg = très approximatif
            PERSO = [round(result.x[3]*10**-6, 1), round(f_power(result.x[3])*10**-6, 1), round(10, 1),
                     round(R0, 1), round(result.x[0], 1), round(R0/result.x[0], 1), 1.7,
                     round(B0_solution, 1), round(result.x[2], 1), round(result.x[4]*10**-6, 1), round(result.x[1], 1),
                     round(Ip_solution*10**-6, 1)]
            
        elif Grad_choice == 'H_fixed':
            # Pe = Pe Freidberg = très approximatif
            PERSO = [round(result.x[2]*10**-6, 1), round(f_power(result.x[2])*10**-6, 1), round(10, 1),
                     round(R0, 1), round(result.x[0], 1), round(R0/result.x[0], 1), 1.7,
                     round(B0_solution, 1), round(result.x[1], 1), round(result.x[3]*10**-6, 1), round(1, 1),
                     round(Ip_solution*10**-6, 1)]
            
        elif Grad_choice == 'H_P_fixed':
            # Pe = Pe Freidberg = très approximatif
            PERSO = [round(P_fus*10**-6, 1), round(f_power(P_fus)*10**-6, 1), round(10, 1),
                     round(R0, 1), round(result.x[0], 1), round(R0/result.x[0], 1), 1.7,
                     round(B0_solution, 1), round(result.x[1], 1), round(result.x[2]*10**-6, 1), round(1, 1),
                     round(Ip_solution*10**-6, 1)]
            
        # Create a DataFrame with pandas
        df = pd.DataFrame({
            'Initials': Initials,
            'Description': Description,
            'ITER': ITER,
            'ARC': ARC,
            'EUdemo A3': EU_DEMO_A3,
            'EUdemo A4': EU_DEMO_A4,
            'CFDTR': CFDTR_DEMO,
            'EUDEMO2': EU_DEMO2,
            'K DEMO': K_DEMO_DN,
            'Prediction': PERSO,
            'Unit': Unit
        })
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 4))
        # Hide the axes
        ax.axis('off')
        # Create a table from the DataFrame
        tbl = table(ax, df, loc='center', colWidths=[0.08, 0.2, 0.12, 0.12, 0.12, 0.12,0.12,0.12,0.12, 0.12, 0.12])
        # Format the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.2)
        # Center the text in each cell
        for key, cell in tbl.get_celld().items():
            cell.set_text_props(ha='center', va='center')
        # Save the image
        path_to_save = os.path.join(save_directory,"table_optimized.png")
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
        # Display the figure
        plt.show()
    
    except Exception as e:
            print(f"An error occurred on the gradient descent : {e}")


if __name__ == "__main__":
    Grad_choice = 'H_P_fixed'
    gradient_descent(0.6,4,0.5,1,10,24,0.2e9,2e9,2,10)
    
#%% Personnal Genetic algorythm

from deap import base, creator, tools, algorithms

def evaluate_physic(individual, H, Bmax, P_fus, P_W):
    a, Aspect_ratio = individual
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul_aspect_ratio(a, H, Bmax, P_fus, P_W, Aspect_ratio)
    
    if qstar_solution > 1:
        return R0_solution,
    else:
        return 1e6,  # Une pénalité élevée si la contrainte n'est pas respectée
    
def evaluate_heat(individual, H, Bmax, P_fus, P_W):
    a, Aspect_ratio = individual
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul_aspect_ratio(a, H, Bmax, P_fus, P_W, Aspect_ratio)
    
    if qstar_solution > 1 and heat < heat_flux_limit :
        return R0_solution,
    else:
        return 1e6,  # Une pénalité élevée si la contrainte n'est pas respectée
    
def evaluate_radial(individual, H, Bmax, P_fus, P_W):
    a, Aspect_ratio = individual
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul_aspect_ratio(a, H, Bmax, P_fus, P_W, Aspect_ratio)
    
    if qstar_solution > 1 and heat < heat_flux_limit and not np.isnan(R0_a_b_c_CS_solution):
        return R0_solution,
    else:
        return 1e6,  # Une pénalité élevée si la contrainte n'est pas respectée

def optimize_aspect_ratio_physic(H, Bmax, P_fus, P_W):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    def generate_aspect_ratio():
        return np.random.uniform(2.5, 4)
    
    toolbox.register("attr_float", generate_aspect_ratio)  # Générer des valeurs aléatoires entre 2.5 et 4 pour l'aspect ratio
    toolbox.register("attr_float_a", np.random.uniform, 1, 4)  # Générer des valeurs aléatoires entre 1 et 4 pour a
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float_a, toolbox.attr_float))  # Créer un individu avec deux paramètres
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Créer une population d'individus
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement Blend
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)  # Mutation gaussienne
    toolbox.register("select", tools.selTournament, tournsize=3)  # Sélection par tournoi
    toolbox.register("evaluate", evaluate_physic, H=H, Bmax=Bmax, P_fus=P_fus, P_W=P_W)  # Évaluation de la fonction objective

    pop = toolbox.population(n=200)  # Taille de la population
    hof = tools.HallOfFame(1)  # Meilleur individu
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # Statistiques
    stats.register("min", np.min)  # Minimum
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=False)  # Algorithme génétique
    
    best_individual = hof[0]
    best_a, best_Aspect_ratio = best_individual
    best_fitness = best_individual.fitness.values[0]
    
    return best_a, best_Aspect_ratio, best_fitness

def optimize_aspect_ratio_radial(H, Bmax, P_fus, P_W):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    def generate_aspect_ratio():
        return np.random.uniform(2.5, 4)
    
    toolbox.register("attr_float", generate_aspect_ratio)  # Générer des valeurs aléatoires entre 2.5 et 4 pour l'aspect ratio
    toolbox.register("attr_float_a", np.random.uniform, 1, 4)  # Générer des valeurs aléatoires entre 1 et 4 pour a
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float_a, toolbox.attr_float))  # Créer un individu avec deux paramètres
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Créer une population d'individus
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement Blend
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)  # Mutation gaussienne
    toolbox.register("select", tools.selTournament, tournsize=3)  # Sélection par tournoi
    toolbox.register("evaluate", evaluate_radial, H=H, Bmax=Bmax, P_fus=P_fus, P_W=P_W)  # Évaluation de la fonction objective

    pop = toolbox.population(n=200)  # Taille de la population
    hof = tools.HallOfFame(1)  # Meilleur individu
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # Statistiques
    stats.register("min", np.min)  # Minimum
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=False)  # Algorithme génétique
    
    best_individual = hof[0]
    best_a, best_Aspect_ratio = best_individual
    best_fitness = best_individual.fitness.values[0]
    
    return best_a, best_Aspect_ratio, best_fitness

def optimize_aspect_ratio_heat(H, Bmax, P_fus, P_W):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    def generate_aspect_ratio():
        return np.random.uniform(2.5, 4)
    
    toolbox.register("attr_float", generate_aspect_ratio)  # Générer des valeurs aléatoires entre 2.5 et 4 pour l'aspect ratio
    toolbox.register("attr_float_a", np.random.uniform, 1, 4)  # Générer des valeurs aléatoires entre 1 et 4 pour a
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float_a, toolbox.attr_float))  # Créer un individu avec deux paramètres
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Créer une population d'individus
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement Blend
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)  # Mutation gaussienne
    toolbox.register("select", tools.selTournament, tournsize=3)  # Sélection par tournoi
    toolbox.register("evaluate", evaluate_heat, H=H, Bmax=Bmax, P_fus=P_fus, P_W=P_W)  # Évaluation de la fonction objective

    pop = toolbox.population(n=200)  # Taille de la population
    hof = tools.HallOfFame(1)  # Meilleur individu
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # Statistiques
    stats.register("min", np.min)  # Minimum
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=False)  # Algorithme génétique
    
    best_individual = hof[0]
    best_a, best_Aspect_ratio = best_individual
    best_fitness = best_individual.fitness.values[0]
    
    return best_a, best_Aspect_ratio, best_fitness

# # Exemple d'utilisation
# H = 1.0  # Exemple, doit être ajusté selon votre contexte
# Bmax = 15.0  # Exemple, doit être ajusté selon votre contexte
# P_fus = 2e9  # Exemple, doit être ajusté selon votre contexte
# P_W = 4e6  # Exemple, doit être ajusté selon votre contexte

# best_a, best_Aspect_ratio, best_fitness = optimize_aspect_ratio_physic(H, Bmax, P_fus, P_W)

# print(f"Best a: {best_a}")
# print(f"Best Aspect_ratio: {best_Aspect_ratio}")
# print(f"Best fitness: {best_fitness}")