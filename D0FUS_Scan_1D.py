# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:35:42 2024

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

#%% Fixed Pfus, R0, H and Bmax while varying the radius 'a':
    
def One_D_scan(H,Bmax,P_fus,R0):

    # Utilisation de la fonction pour initialiser les listes
    (a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions,P_W_solutions) = initialize_lists()
    
    # Call the a_variation function
    (a_solutions,nG_solutions,n_solutions,beta_solutions,qstar_solutions,fB_solutions,fNC_solutions,cost_solutions,heat_solutions,c_solutions,sol_solutions,P_W_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,Ip_solutions,fRF_solutions) = Variation_a(H, Bmax, P_fus, R0)
    
    # Initialize a variable to store the first acceptable value
    first_acceptable_value = None
    # Iterate over the results
    for i in range(len(a_solutions)):
        # Check if all values are less than 1
        if (
            n_solutions[i] / nG_solutions[i] < 1 and
            beta_solutions[i] / betaN < 1 and
            q / qstar_solutions[i] < 1 and
            fRF_solutions[i]/f_RF_objectif < 1
        ):
            # Save the first acceptable value
            first_acceptable_value = a_solutions[i]
            break  # Exit the loop as soon as a value is found
    # Check if a value has been found
    if first_acceptable_value is not None:
        print("The first acceptable value is:", round(first_acceptable_value,2))
    else:
        print("No acceptable value was found.")
    
    # Plot with respect to 'a'
    chosen_parameter = 'a'
    parameter_values = a_solutions
    chosen_unity = 'm'
    chosen_design = None
    
    Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design)
    
    Plot_cost_function(chosen_parameter,parameter_values,cost_solutions,first_acceptable_value,chosen_unity,chosen_design)
    
    Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design)
    
    R0_list = [R0] * len(R0_a_solutions)
    
    Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_list,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design)

if __name__ == "__main__":
    (H,P_fus,R0,Bmax,κ)=init(CHOICE)
    One_D_scan(H,Bmax,P_fus,R0)

#%% Variation of a chosen parameter and avaluating the optimal (a)

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions,P_W_solutions) = initialize_lists()

chosen_parameter = None
chosen_unity = None

# Ask the user to choose the parameter to vary
chosen_parameter = input("Choose the parameter to vary (H, Bmax, Pfus, R0, fobj): ")

# Define ranges and steps for each parameter
parameter_ranges = {
    'H': np.arange(0.6, 2, 0.01),
    'Bmax': np.arange(10, 25, 0.1),
    'Pfus': np.arange(500.E6, 5000.E6, 100.E6),
    'R0': np.arange(2, 10, 0.1),
    'fobj' : np.arange(0.01,1.0,0.01)
}

unit_mapping = {'H': '', 'Bmax': 'T', 'Pfus': 'GW', 'R0': 'm','fobj': ''}
chosen_unity = unit_mapping.get(chosen_parameter, '')

# Select the appropriate range for the chosen parameter
parameter_values = parameter_ranges.get(chosen_parameter)

a_vec = np.arange(a_min, a_max, 1/na)

(H,P_fus,R0,Bmax,κ)=init(CHOICE)

if parameter_values is not None:
    for parameter_value in tqdm(parameter_values, desc='Processing parameters'):
        
        # Update the chosen parameter
        if chosen_parameter == 'H':
            H = parameter_value
        elif chosen_parameter == 'Bmax':
            Bmax = parameter_value
        elif chosen_parameter == 'Pfus':
            P_fus = parameter_value
        elif chosen_parameter == 'R0':
            R0 = parameter_value
        elif chosen_parameter == 'fobj':
            f_RF_objectif = parameter_value

        # Historical solver
        # a_solution = Solveur_historique(H,Bmax,P_fus,P_W,f_RF_objectif)
        
        # New solver
        a_solution = Solveur_raffine(H,Bmax,P_fus,R0,f_RF_objectif)
        
        a_solutions.append(a_solution)
        
        # Calculate useful values
        (P_W,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Q_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution, H, Bmax, P_fus, R0)
        
        # Store the results
        n_solutions.append(n_vec_solution)
        nG_solutions.append(nG_solution)
        beta_solutions.append(beta_solution)
        qstar_solutions.append(qstar_solution)
        fB_solutions.append(fB_solution)
        fNC_solutions.append(fNC_solution)
        cost_solutions.append(cost)
        heat_solutions.append(heat)
        c_solutions.append(c)
        sol_solutions.append(solenoid)
        R0_solutions.append(R0)
        R0_a_solutions.append(R0_a_solution)
        R0_a_b_solutions.append(R0_a_b_solution)
        R0_a_b_c_solutions.append(R0_a_b_c_solution)
        R0_a_b_c_CS_solutions.append(R0_a_b_c_CS_solution)
        required_BCSs.append(required_Bcs)
        Ip_solutions.append(Ip_solution)
        fRF_solutions.append(fRF_solution)

else:
    print("Invalid chosen parameter. Please choose from 'H', 'Bmax', 'Pfus', 'Pw'.")

# Convert lists to NumPy arrays
a_solutions = np.array(a_solutions)
nG_solutions = np.array(nG_solutions)
n_solutions = np.array(n_solutions)
beta_solutions = np.array(beta_solutions)
qstar_solutions = np.array(qstar_solutions)
fB_solutions = np.array(fB_solutions)
fNC_solutions = np.array(fNC_solutions)
cost_solutions = np.array(cost_solutions)
heat_solutions = np.array(heat_solutions)
c_solutions = np.array(c_solutions)
sol_solutions = np.array(sol_solutions)
R0_solutions = np.array(R0_solutions)
R0_a_solutions = np.array(R0_a_solutions)
R0_a_b_solutions = np.array(R0_a_b_solutions)
R0_a_b_c_solutions = np.array(R0_a_b_c_solutions)
R0_a_b_c_CS_solutions = np.array(R0_a_b_c_CS_solutions)
required_BCSs = np.array(required_BCSs)
Ip_solutions = np.array(Ip_solutions)
fRF_solutions = np.array(fRF_solutions)

# Ask the user to choose to plot or not the first acceptable value
Plot_choice = input("Do you want to plot the first acceptable value ? (Yes/No)")

if Plot_choice == 'Yes':
    if chosen_parameter == 'Pw':
        # Initialize a variable to store the last acceptable value
        first_acceptable_value = None
        # Iterate over the results
        for i, param_value in enumerate(parameter_values):
            # Check if all values are less than 1
            if (
                n_solutions[i] / nG_solutions[i] <= 1 and
                beta_solutions[i] / betaN <= 1 and
                q / qstar_solutions[i] <= 1 and
                fRF_solutions[i]/f_RF_objectif <= 1
            ):
                # Save the first acceptable value
                first_acceptable_value = param_value
        # Check if a value has been found
        if first_acceptable_value is not None:
            print("The last acceptable value is:", first_acceptable_value)
        else:
            print("No acceptable value was found.")
    else:
        # Initialize a variable to store the last acceptable value
        first_acceptable_value = None
        # Iterate over the results
        for i, param_value in enumerate(parameter_values):
            # Check if all values are less than 1
            if (
                n_solutions[i] / nG_solutions[i] < 1 and
                beta_solutions[i] / betaN < 1 and
                q / qstar_solutions[i] < 1 and
                fB_solutions[i]/fNC_solutions[i] < 1
            ):
                # Save the first acceptable value
                first_acceptable_value = param_value
                break
        # Check if a value has been found
        if first_acceptable_value is not None:
            print("The first acceptable value is:", first_acceptable_value)
        else:
            print("No acceptable value was found.")
else :
    first_acceptable_value = None
    
chosen_design = None

Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design)

Plot_cost_function(chosen_parameter,parameter_values,cost_solutions,first_acceptable_value,chosen_unity,chosen_design)

Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design)

Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design)