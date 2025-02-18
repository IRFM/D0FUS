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
    from D0FUS_initialisation import *
    print("-----D0FUS Reloaded-----")
else:
    # Le module n'est pas encore importé, l'importer
    from D0FUS_initialisation import *
    print("-----D0FUS loaded-----")
    
# Attention, lors de toute modification de fichiers exterieurs, il est conseillé, malgré l'implémentation de la fonction reaload, de redémarrer le Kernel.
# Python est capricieux avec le stockage des variables et seul un redémarage du kernel puis import des bibliothèques est sur à 100%
# Le rechargement d'un module (importlib.reload) ne met en effet pas forcément à jour les références existantes dans d'autres modules importés précédemment
# La fonction reload doit cependant , du moins dans les versions python de 3.1 à 3.3, permettre de palier à ce problème

#%% Algos

# Fonction de coût à minimiser
def fitness_function(individual):
    a, R0, Bmax, P_fus, T_bar = individual
    results = calcul(a, R0, Bmax, P_fus, T_bar)
    nbar_solution = results[5]
    nG_solution = results[9]
    beta_solution = results[6]
    qstar_solution = results[7]
    R0_abcd = results[21]
    cost = results[13]


    if cost < 0 or np.isnan(cost) or R0_abcd < 0 or np.isnan(R0_abcd) or qstar_solution < 0 or np.isnan(qstar_solution) or beta_solution < 0 or np.isnan(beta_solution) or nbar_solution < 0 or np.isnan(nbar_solution) :
        return (float('inf'),)  # Retourne un tuple
    
    if nbar_solution > nG_solution :
        return (cost+((nbar_solution / nG_solution ) + 1)**10,)  # Retourne un tuple

    if beta_solution > betaN :
        return (cost+((beta_solution / betaN ) + 1)**10,)  # Retourne un tuple

    if qstar_solution < q  :
        return (cost+((q / qstar_solution ) + 1)**10,)  # Retourne un tuple

    return (cost,)  # Retourne un tuple contenant le coût

Choice_research = "Genetic"

# Benchmark results :
# Genetic (most flexible and fast ~ 1 min)
# Differential (most precise but ~ 10 min)
# Annealing (failed)
# Powell (failed)
# NelderMead (failed)
# PSO (failed)

# Définir les bornes pour chaque variable
bounds = [(0, 2), (3, 7), (6, 20), (1000, 2000), (5, 20)] # a, R0, Bmax, Pfus, T
# Best individual from genetic algorithm: [1.0267922161988503, 5.646477064593668, 19.02073188421982, 2089.274846909817, 12.343712356747329]

if Choice_research == "Genetic":

    # Paramètres de l'algorithme génétique
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Initialiser les individus avec des valeurs dans les bornes spécifiées pour chaque paramètre
    toolbox.register("attr_a", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("attr_R0", np.random.uniform, bounds[1][0], bounds[1][1])
    toolbox.register("attr_Bmax", np.random.uniform, bounds[2][0], bounds[2][1])
    toolbox.register("attr_P_fus", np.random.uniform, bounds[3][0], bounds[3][1])
    toolbox.register("attr_T_bar", np.random.uniform, bounds[4][0], bounds[4][1])
    
    # Créer un individu en regroupant ces attributs
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_a, toolbox.attr_R0, toolbox.attr_Bmax, toolbox.attr_P_fus, toolbox.attr_T_bar), n=1)
    
    # Définir une population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness_function)
    
    def genetic_algorithm():
        pop = toolbox.population(n=200)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
        best_individual = hof[0]
        print("Best individual from genetic algorithm:", best_individual)
    
    genetic_algorithm()
        
elif Choice_research == "Differential":
    result = differential_evolution(fitness_function, bounds)
    print("Best result from differential evolution:", result)

elif Choice_research == "SimulatedAnnealing":
    result = minimize(fitness_function, x0=[1.5, 3.5, 9, 25, 13], bounds=bounds, method='anneal')
    print("Best result from Simulated Annealing:", result)

elif Choice_research == "Powell":
    result = minimize(fitness_function, x0=[1.5, 3.5, 9, 25, 13], bounds=bounds, method='Powell')
    print("Best result from Powell optimization:", result)

elif Choice_research == "NelderMead":
    result = minimize(fitness_function, x0=[1.5, 3.5, 9, 25, 13], bounds=bounds, method='Nelder-Mead')
    print("Best result from Nelder-Mead optimization:", result)

elif Choice_research == "PSO":
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    result, _ = pso(fitness_function, lb, ub)
    print("Best result from Particle Swarm Optimization:", result)
    
# if __name__ == "__main__":
#     methods = {
#         "Genetic": genetic_algorithm,
#         "Differential": lambda: differential_evolution(fitness_function, bounds),
#         "SimulatedAnnealing": lambda: minimize(fitness_function, x0=[1.5, 3.5, 9, 25, 13], bounds=bounds, method='anneal'),
#         "Powell": lambda: minimize(fitness_function, x0=[1.5, 3.5, 9, 25, 13], bounds=bounds, method='Powell'),
#         "NelderMead": lambda: minimize(fitness_function, x0=[1.5, 3.5, 9, 25, 13], bounds=bounds, method='Nelder-Mead'),
#         "PSO": lambda: pso(fitness_function, [b[0] for b in bounds], [b[1] for b in bounds])
#     }
    
#     results = {}
#     for method, func in methods.items():
#         start_time = time.time()
#         try:
#             result = func()
#             execution_time = time.time() - start_time
#             results[method] = (result, execution_time)
#         except Exception as e:
#             results[method] = (None, None, str(e))
    
#     for method, (result, exec_time, *error) in results.items():
#         if error:
#             print(f"{method} failed: {error[0]}")
#         else:
#             print(f"{method} completed in {exec_time:.4f} seconds. Best result: {result}")


