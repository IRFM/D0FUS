# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:34:07 2023

@author: Timothe
"""

#%% Import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import differential_evolution
import pandas as pd
from pandas.plotting import table
import warnings
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns  # improves plot aesthetics
from scipy.optimize import root_scalar
import math
from scipy.optimize import minimize_scalar
import matplotlib.cm as cm
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.lines as mlines

#%% Definition of constants

# Physical constants
E_ELEM  = 1.6E-19           # Coulomb
M_E     = 9.1094E-31        # kg
M_I     = 2 * 1.6726E-27    # kg
μ0    = 4.*np.pi*1.E-7    # Henri/m
EPS_0   = 8.8542E-12        # Farad/m
k = 1.6*10**-16 # Boltzman constant [J/keV]

# Fusion constants
E_ALPHA = 3.5 *1.E6*E_ELEM  # Joule
E_N     = 14.1*1.E6*E_ELEM  # Joule
E_F     = 22.4*1.E6*E_ELEM  # Joule
betaN = 2.8  # in % (Beta Tryon limit)
Ce = 0.45 # Ejima constants
A     = 2.5  # average atomic mass

# Classical parameters for a Tokamak
Li = 0.85 # Internal induction of the plasma
Tbar  = 14*1e3*E_ELEM  # J
κ = 1.7  # elongation
nu_n  = 0.5
nu_T  = 1.
Cw = 1 #(1+(Ti/Te))/2 # Temperature ratio, dilution should be to add
ITERPI = 20 # ITER plasma induction current [Wb]
Flux_CS_Utile = 0.85 # pourcentage disponible, laissant une partie libre pour le controle du plasma , arbitrairement à 0.95, la littérature propose plusieurs values allant de 0.85 à 1
q = 2
lower_bound_a = 0.3
upper_bound_a = 3

# Scaling law choice
SL_choice = 1
#HIPB98
#DS03
#LIPB98

# Engineer constraints
Sigm_max = 600.E6   # Mechanical limit of the steel considered in [Pa]
J_max = 20.E6       #A/m²
eta_RF = 0.5  # conversion efficiency from wall power to klystron
f_RP   = 0.8  # fraction of klystron power absorbed by plasma
f_RF_objectif   = 1  # RF recirculating power target
eta_T = 0.4    # Ratio between thermal and electrical power
AJ_AC = 0.35 #Ratio entre l'aire de Jacket / l'aire de conducteur dans le CS , ici = ITER value
σcs = 600*10**6  # CS machanical limit [Pa]

#%% Database creation

Initials = ['Pf', 'Pe', 'Q', 'R0', 'a', 'A', 'κ', 'B0', 'Bmax', 'Pw', 'H98y2', 'Ip']
Description = ['Fusion Power', 'Electrical Power', 'Pfus/Pheat',
               'Major Radius', 'Plasma radius', 'Aspect Ratio', 'Elongation', 'Plasma field',
               'Field Coil', 'Deposited Power', 'H factor ITER98', 'Plasma Current']
Unit = ['MW', 'MW', '', 'm', 'm', '', '', 'T', 'T', 'MW/m²', '', 'MA']

# General informations : DOI 10.1088/0029-5515/47/6/S01  (Nuclear Fusion : Chapter 1: Overview and summary)
# Pw : https://doi.org/10.1016/j.fusengdes.2007.02.022 Benchmarking of MCAM 4.0 with the ITER 3D model
ITER = ['500', '0', '10', '6.2', '2', '3.1', '1.78', '5.3', '12.5', '0.5', '1', '15']

# https://doi.org/10.1016/j.fusengdes.2015.07.008 : Sorbom, B. N., et al. "ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets." Fusion Engineering and Design (2015)
ARC = ['525', '283', '3', '3.3', '1.13', '2.9', '1.84', '9.2', '23', '2.5', '1.8', '7.8']

# https://doi.org/10.1016/j.fusengdes.2014.01.070 : Overview of EU DEMO design and R&D activities , Federici (2014)
EU_DEMO_A3 = ['1793', '500', '10', '9', '3', '3', '1.6', '5.2', '10.8', '0.9', '1', '20.3']
EU_DEMO_A4 = ['1807', '500', '10', '9', '2.25', '4', '1.6', '7.36', '13', '1.2', '1', '15.2']

# DOI:10.1088/0029-5515/56/2/026009 : Diagnostics and control for the steady state and pulsed tokamak DEMO (2016) with Federici
EU_DEMO2 =['2500', '/', '12', '8.5', '2.83', '3', '1.65', '5.8', '/', '1.4', '1.3', '23.3']

# DOI 10.1088/0029-5515/55/5/053027 : Kim, K., et al. "Design concept of K-DEMO for near-term implementation." Nuclear Fusion 55.5 (2015)
K_DEMO_DN = ['2500', '500', '25', '6.8', '2.1', '3.2', '2', '7.4', '16', '2.9', '1.3', '12']

# DOI 10.1088/1741-4326/ab0e27 : Zhuang, Ge, et al. "Progress of the CFETR design." Nuclear Fusion 59.11 (2019)
CFDTR_A3 = ['482', '30', '5.87', '7.2', '2.2', '3.2', '2', '6.5', '/', '0.49', '1.32', '12.92']
CFDTR_A4 = ['974', '232', '11.89', '7.2', '2.2', '3.2', '2', '6.5', '/', '0.99', '1.41', '13.78']
CFDTR_DEMO = ['2192', '738', '28.17', '7.2', '2.2', '3.2', '2', '6.5', '14', '2.23', '1.42', '13.78']

#%% Reactor parameters initialisation

CHOICE = 0
#0 = Freidberg
#1 = ITER
#2 = ARC
#3 = EUDEMO_A3
#4 = EUDEMO_A4
#5 = EUDEMO2
#6 = K_DEMO
#7 = CFDTR

def init(Choice):
    
    # Choix 0: Freidberg
    if Choice == 0:
        H = 1
        P_fus = 2000*10**6 # Calculated from the P_E freidberg
        P_W = 4*10**6
        Bmax = 13
        κ = 1.7
    
    # Choix 1: ITER
    if Choice == 1:
        H = float(ITER[-2])
        P_fus = float(ITER[0])*10**6
        P_W = float(ITER[-3])*10**6
        Bmax = float(ITER[-4])
        κ = float(ITER[6])
    
    # Choix 2: ARC
    elif Choice == 2:
        H = float(ARC[-2])
        P_fus = float(ARC[0])*10**6
        P_W = float(ARC[-3])*10**6
        Bmax = float(ARC[-4])
        κ = float(ARC[6])
    
    # Choix 3: EU_DEMO_A3
    elif Choice == 3:
        H = float(EU_DEMO_A3[-2])
        P_fus = float(EU_DEMO_A3[0])*10**6
        P_W = float(EU_DEMO_A3[-3])*10**6
        Bmax = float(EU_DEMO_A3[-4])
        κ = float(EU_DEMO_A3[6])
    
    # Choix 4: EU_DEMO_A4
    elif Choice == 4:
        H = float(EU_DEMO_A4[-2])
        P_fus = float(EU_DEMO_A4[0])*10**6
        P_W = float(EU_DEMO_A4[-3])*10**6
        Bmax = float(EU_DEMO_A4[-4])
        κ = float(EU_DEMO_A4[6])
    
    # Choix 5: EU_DEMO2
    elif Choice == 5:
        H = float(EU_DEMO2[-2])
        P_fus = float(EU_DEMO2[0])*10**6
        P_W = float(EU_DEMO2[-3])*10**6
        Bmax = float(EU_DEMO2[-4])
        κ = float(EU_DEMO2[6])
    
    # Choix 6: K_DEMO_DN
    elif Choice == 6:
        H = float(K_DEMO_DN[-2])
        P_fus = float(K_DEMO_DN[0])*10**6
        P_W = float(K_DEMO_DN[-3])*10**6
        Bmax = float(K_DEMO_DN[-4])
        κ = float(K_DEMO_DN[6])
    
    # Choix 7: CFDTR_DEMO
    elif Choice == 7:
        H = float(CFDTR_DEMO[-2])
        P_fus = float(CFDTR_DEMO[0])*10**6
        P_W = float(CFDTR_DEMO[-3])*10**6
        Bmax = float(CFDTR_DEMO[-4])
        κ = float(CFDTR_DEMO[6])
    
    return(H,P_fus,P_W,Bmax,κ)

(H,P_fus,P_W,Bmax,κ)=init(CHOICE)

b = 1.2 #A implémenter

#%% Numerical initialisation

# Hide runtime-related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create a directory to save images if it doesn't exist
# save_directory = "Graphs"
save_directory = "C:/Users/TA276941/Desktop/D0Fus/Graphs/Données brutes/"
os.makedirs(save_directory, exist_ok=True)
   
# Set parameters related to a (radius)
na = 1000
nx = 1800  # Number of points for approximations
a_min = 0.4 # range of a for a chosen scan
a_max = 2.6
a_pas = 0.01 # gap between 2 try
a_init = 1   # Initial value of a in the searches for the zero of the loss function

max_iterations = 10000 # Max iteration for minimum research

def initialize_lists():
    # Initialize lists to store the results
    a_solutions = []
    nG_solutions = []
    n_solutions = []
    beta_solutions = []
    qstar_solutions = []
    fB_solutions = []
    fNC_solutions = []
    cost_solutions = []
    heat_solutions = []
    c_solutions = []
    sol_solutions = []
    R0_a_solutions = []
    R0_a_b_solutions = []
    R0_a_b_c_solutions = []
    R0_a_b_c_CS_solutions = []
    required_BCSs = []
    R0_solutions = []
    Ip_solutions = []
    fRF_solutions = []
    
    return (
        a_solutions, 
        nG_solutions, 
        n_solutions, 
        beta_solutions, 
        qstar_solutions, 
        fB_solutions, 
        fNC_solutions, 
        cost_solutions, 
        heat_solutions, 
        c_solutions, 
        sol_solutions,
        R0_a_solutions,
        R0_a_b_solutions,
        R0_a_b_c_solutions,
        R0_a_b_c_CS_solutions,
        required_BCSs,
        R0_solutions,
        Ip_solutions,
        fRF_solutions
    )

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()

#%% Physical Functions

def conductivity(Te):
    res = 2.8*10**-8/Te**(3/2) # plasma resistivity [Ohm*m]
    σp = 1/res  # Plasma conductivity [S/m]
    return(σp)

def LCFS(a):
    L = np.pi * np.sqrt(2 * (a**2 + (κ*a)**2)) # LCFS length
    return L

def f_rho(nx):
    rhomin = 1.E-6
    rhomax = 1.-rhomin
    rho    = np.linspace(rhomin,rhomax,nx)
    drho   = rho[1]-rho[0]
    return rho, drho

def f_Tprof(Tbar,nu_T,rho):
    one_m_rho2 = 1.-rho*rho
    temp = Tbar*(1+nu_T)*one_m_rho2**nu_T
    return temp

def f_nprof(nbar,nu_n,rho):
    one_m_rho2 = 1.-rho**2
    dens = nbar*(1+nu_n)*one_m_rho2**nu_n
    return dens

def f_power(P_fus):
    P_E = P_fus*(22.4/17.6)*eta_T    # Conversion from Pfus(classical) to P_E(freidberg)
    return(P_E)

def f_pbar(Tbar,nu_n,nu_T,R0,a,κ,P_fus,eta_T,nx):
    P_E = f_power(P_fus)
    nu_p = nu_n+nu_T
    pbar = 2*Tbar*(1+nu_T)/(np.pi*(1+nu_p)) * np.sqrt(P_E/(eta_T*E_F*R0*a*a*κ))
    [rho,drho] = f_rho(nx)
    temp = f_Tprof(Tbar,nu_T,rho)
    one_m_rho2  = 1.-rho*rho
    sigmav_prof = np.linspace(0.,1.,nx)
    for ix in range(0,nx):
        T_loc           = temp[ix]
        sigmav_prof[ix] = f_sigmav(T_loc)
    integrand = sigmav_prof*rho*one_m_rho2**(2.*nu_n) * drho
    integral  = np.sum(integrand)-0.5*(integrand[0]+integrand[-1])
    pbar = pbar / np.sqrt(integral)
    return pbar

def f_beta(pbar,B0,a,Ip):
    beta = 2*μ0*pbar*a/(B0*Ip*1.E-6) * 100. # in %
    return beta

def f_nbar(pbar,Tbar,nu_n,nu_T):
    nu_p = nu_n+nu_T
    nbar = pbar/Tbar*0.5*(1+nu_p)/((1+nu_n)*(1+nu_T))
    return nbar

def f_sigmav(T):
    T_keV = T*1e-3/E_ELEM
    lnT  = np.log(T_keV)
    lnT2 = lnT*lnT
    k0   = -60.4593
    k1   =   6.1371
    k2   = - 0.8609
    k3   =   0.0356
    k4   = - 0.0045
    sigmav = np.exp(k0+k1*lnT+k2*lnT2+k3*lnT2*lnT+k4*lnT2*lnT2)
    return sigmav

def f_R0(a,P_fus,P_W,eta_T,κ):
    P_E = f_power(P_fus)        
    R0 = (E_N/E_F)*(P_E/(eta_T*P_W))*np.sqrt(2. / (1. + κ ** 2))* (1 / (4. * np.pi ** 2 * a))
    return R0

def f_B0(Bmax,a,b,R0):
    B0 = Bmax*(1-((a+b)/R0))
    return B0

def f_nG(I,a):
    nG = (I*1.E-6)/(np.pi*a*a)
    return nG

def f_tauE(pbar,R0,a,κ,eta_T,P_fus):
    P_E = f_power(P_fus)
    tauE = pbar*3.*np.pi*np.pi*R0*a*a*κ
    tauE = tauE * E_F*eta_T/(E_ALPHA*P_E)
    return tauE

def f_SL_param(SL_choice):
    if SL_choice==1:
        #IPB98
        c0      = 0.0562*2.58
        c_epsilon = 0.58
        c_I     = 0.93
        c_R     = 1.97-c_epsilon
        c_a     = c_epsilon
        c_κ = 0.78
        c_n     = 0.41
        c_B     = 0.15
        c_A     = 0.19
        c_P     = -0.69
        
    elif SL_choice==2:

        #DS03
        c0 = 0.028*2.58
        c_epsilon = 0.30
        c_A = 0.14
        c_κ = 0.75
        c_a = c_epsilon
        c_n = 0.49
        c_I = 0.83
        c_R = 2.11-c_epsilon
        c_B = 0.07
        c_P = -0.55
    
    elif SL_choice==3:

        #L-mode
        c0 = 0.023*2.58
        c_epsilon = -0.06
        c_A = 0.20
        c_κ = 0.64
        c_a = c_epsilon
        c_n = 0.40
        c_I = 0.96
        c_R = 1.78-c_epsilon
        c_B = 0.03
        c_P = -0.73
        
    return c0,c_I,c_R,c_a,c_κ,c_n,c_B,c_A,c_P

def f_Ip(SL_choice,H,tauE,R0,a,κ,nbar,B0,A,eta_T,P_fus):
    P_E = f_power(P_fus)
    P_alpha = E_ALPHA*P_E/(E_F*eta_T) * 1.E-6 # in MW
    
    [c0,c_I,c_R,c_a,c_κ,c_n,c_B,c_A,c_P] = f_SL_param(SL_choice)
    
    Suspect = B0**c_B
    # Verification des solutions
    partie_reelle = Suspect.real
    partie_imaginaire = Suspect.imag
    
    # Vérification de la condition
    if partie_reelle > 1000 * partie_imaginaire:
        # Affectation de la valeur réelle à la variable B0_p
        Suspect = partie_reelle
    
    Numerateur = tauE*P_alpha**(-c_P)
    Denominateur = H*c0*R0**c_R*a**c_a*κ**c_κ*(nbar*1.E-20)**c_n*Suspect*A**c_A
    inv_cI  = 1./c_I
    Ip = ((Numerateur/Denominateur)**inv_cI)*10**6 # in A
    return Ip

def f_qstar(a,B0,R0,Ip,κ):
    qstar = 2.*np.pi*a*a*B0*0.5*(1+κ*κ)/(μ0*R0*Ip)
    return qstar

def f_etaCD(a,R0,B0,nbar,Tbar,nu_n,nu_T,nx):
    [rho,drho] = f_rho(nx)
    rho_m  = 0.8
    var    = np.abs(rho-rho_m)
    irho_m = int( np.min(np.where(var==np.min(var))) )
    temp   = f_Tprof(Tbar,nu_T,rho)
    dens   = f_nprof(nbar,nu_n,rho)
    n_loc  = dens[irho_m]
    T_loc  = temp[irho_m]
    eps    = a/R0
    eps_B  = (a+b)/R0
    B_loc  = B0/(1+eps*rho_m)
    omega_ce = E_ELEM*B_loc/M_E
    omega_pe = E_ELEM*np.sqrt(n_loc/(EPS_0*M_E))
    n_parall = omega_pe/omega_ce + np.sqrt(1+(omega_pe/omega_ce)**2)*np.sqrt(3./4.)
    eta_CD = 1.2/(n_parall*n_parall) 
    return eta_CD

def f_fB(eta_CD,R0,Ip,nbar,eta_RF,f_RP,f_RF_objectif,P_fus):
    P_E = f_power(P_fus)
    P_CD = eta_RF*f_RP*f_RF_objectif * P_E
    f_B  = 1 - eta_CD*P_CD/(R0*nbar*1.E-20*Ip)
    return f_B

def f_fRF(f_NC,eta_CD,R0,Ip,nbar,eta_RF,f_RP,P_fus):
    P_E = f_power(P_fus)
    f_RF =  (1-f_NC)*(R0*nbar*1.E-20*Ip)/(eta_CD*eta_RF*f_RP*P_E)
    return(f_RF)

def f_fNC(a,κ,pbar,R0,Ip,nx):
    [rho,drho] = f_rho(nx)
    def f_btheta(rho):
        alpha  = 2.53
        alphax = alpha*rho**(9./4.)
        num    = (1+alpha-alphax)*np.exp(alphax)-1-alpha
        denom  = rho*(np.exp(alpha)-1-alpha)
        b_theta = num/denom
        return b_theta
    b_theta   = f_btheta(rho)
    integrand = rho**(5./2.)*np.sqrt(1-rho*rho)/b_theta * drho
    integral  = np.sum(integrand)-0.5*(integrand[0]+integrand[-1])
    num   = 268*a**(5./2.)*κ**(5./4.)*pbar * integral
    denom = μ0*np.sqrt(R0)*Ip*Ip
    f_NC  = num/denom
    return f_NC

def f_coil(a,b,R0,B0,Sigm_max,μ0,J_max):
    eps_B  = (a+b)/R0
    alpha_M = (B0**2/(μ0*Sigm_max))*(((2*eps_B)/(1+eps_B))+(0.5*np.log((1+eps_B)/(1-eps_B))))
    alpha_J = (2*B0)/(μ0*R0*J_max)
    c = R0*(2*(1-eps_B)-np.sqrt(((1-eps_B)**2)-alpha_M)-np.sqrt(((1-eps_B)**2)-alpha_J))
    return c

def f_cost(a,b,R0,c,κ,P_fus):
    P_E = f_power(P_fus)
    V_b = 2*np.pi**2*R0*((a+b)*(κ*a+b)-κ*a**2)
    V_tf = 4*np.pi*c*(2*R0-2*a-2*b-c)*((1+κ)*a+2*b+c)
    VI_Pe = (V_b+V_tf)/P_E
    if np.isnan(VI_Pe)==True or VI_Pe is None:
        VI_Pe = 10e-6
    return VI_Pe

def f_heat(B0,R0,P_fus,eta_T):
    P_E = f_power(P_fus)
    P_alpha = (E_ALPHA*P_E)/(E_F*eta_T) * 1.E-6 # in MW
    Q = (P_alpha*B0)/R0
    return Q

def f_slnd(R0,a,b,c):
    reCS = R0-a-b-c-0.1
    return reCS

def f_beta_p_th(Cw,ne,Te,Ip,a):
    L = 2 * np.pi * (a**2 + (κ*a)**2)**(1/2)
    βp = 4/μ0 * Cw * ne * (Te*1000) * k / Ip**2 * L**2
    return(βp)

# Function to calculate the plateau duration
def f_plateau_duration(Bcs,R0,reCS,riCS,ne,Te,Ip,IBS,a):
    
    L = LCFS(a) # Length of the last closed flux surface
    βp = 4/μ0 * Cw * L**2 * (ne*k*Te) / Ip**2 # 0.62 for ITER
    σp = conductivity(Te)
    
    # Calculate available flux from CS system (ΨCS)
    ΨCS = math.pi * Bcs/3 * (riCS**2 + (riCS * reCS) + reCS**2) * 2 * Flux_CS_Utile
    print("Flux CS :", ΨCS)
    
    # Calculate available flux from PF system (ΨPF)
    ΨPF = μ0 * Ip / (4*R0) * (βp + (Li - 3)/2 + math.log(8 * a / κ**(1/2) )) * (R0**2 - reCS**2)
    print("Flux PF :", ΨPF)
    
    # Calculate flux needed for plasma initiation (ΨPI)
    ΨPI = ITERPI * (2 * math.pi**2 * R0 * a**2 * κ) / 836  # Assuming a constant evolution for plasma initiation consumption from ITER [Wb] with the plasma volume
    print("Flux Init :", ΨPI)

    # Calculate flux needed for the inductive part (Ψind)
    Lp = 1.07 * μ0 * R0 * (1+0.1*βp) * (Li/2 - 2 + math.log(8 * R0/(a*math.sqrt(κ))))
    Ψind = Lp * Ip
    print("Flux Ind :", Ψind)

    # Calculate flux related to the resistive part (Ψres)
    Ψres = Ce * μ0 * R0 * Ip
    print("Flux Res :", Ψres)

    # Calculate flux related to the current plateau (Ψplateau)
    Ψplateau = ΨCS + ΨPF - ΨPI - Ψind - Ψres
    print("Flux plateau :", Ψplateau)

    #IBS = Ip / 5  # Bootstrap current
    Vloop = (Ip - IBS)/ σp * 2 * math.pi * R0 /( κ * a**2)
    
    # Calculate plateau duration (tplateau)
    tplateau = Ψplateau / Vloop
    return tplateau

#%% Other Functions

def joules_to_kev(energy_joules):
    # 1 Joule = 6.242e+15 kilo-électrovolts
    conversion_factor = 6.242e+15
    energy_kev = energy_joules * conversion_factor
    return energy_kev

# Function returning fb-fnc for a given a
def Solveur_fb_fnc_fonction_de_a(H,Bmax,P_fus,P_W):
    def To_solve(a):
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
        return fB_solution - fNC_solution
    a = fsolve(To_solve, a_init)[0]
    a = a if a is not None else np.nan
    return(a)

def Solveur_parrallel(H,Bmax,P_fus,P_W,f_RF_objectif):
    delta = 0.002 # Finesse de la détection (en mètres)
    a_start = 0.4 # Valeur initiale de a 
    a_limite = 4.01 # Valeur max de a autorisée
    def To_solve_raffiné(a):
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
        return max(fRF_solution / f_RF_objectif ,n_vec_solution / nG_solution,beta_solution / betaN,q / qstar_solution)
    # Définir les limites de la recherche pour 'a'
    bounds = (lower_bound_a, upper_bound_a)
    # Minimiser la valeur maximale en utilisant minimize_scalar
    result = minimize_scalar(To_solve_raffiné, bounds=bounds)
    # La valeur optimale de 'a' sera dans result.x
    optimal_a = result.x
    optimal_a_raffine = np.nan
    
    # Si nan : retour nan
    if To_solve_raffiné(optimal_a) is None or np.isnan(To_solve_raffiné(optimal_a)):
        return np.nan
    
    # Si L'une des 4 limites supérieur à 1 alors on retourne a minisant le max des limites
    elif To_solve_raffiné(optimal_a)>1:
            return optimal_a
        
    # Si il existe une solution :
    elif To_solve_raffiné(optimal_a)<=1:
        # Variables de cout
        min_cost_Radial = 100
        min_cost_Plasma = 100
        # Boucle parcourant les valeurs de a
        for a in np.arange(a_start,a_limite,delta) :
            # Evaluation des différentes valeurs utiles
            (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
            cost = cost*1e6
            # Vérifier si toutes les conditions sont satisfaites
            if (
                n_vec_solution / nG_solution < 1 and
                beta_solution / betaN < 1 and
                q / qstar_solution < 1 and
                fRF_solution / f_RF_objectif < 1
            ):
                # Si oui :
                # Minimisation d'une fonction de cout souhaité, ici R0
                if min_cost_Plasma>R0_solution:
                    optimal_a = a
                    min_cost_Plasma = R0_solution
                # Si en plus, le Radial build est possible :
                if not np.isnan(R0_a_b_c_CS_solution):
                    # Minimisation d'une fonction de cout souhaité, ici R0
                    if min_cost_Radial>R0_solution:
                        optimal_a_raffine = a
                        min_cost_Radial = R0_solution
        
        # Si une solution avec radial build est possible , alors à favoriser
        if optimal_a_raffine is not None and not np.isnan(optimal_a_raffine):
            optimal_a = optimal_a_raffine
        
        return (optimal_a)

def Solveur_raffiné(H,Bmax,P_fus,P_W,f_RF_objectif):

    def To_solve_raffiné(a):
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
        return max(fRF_solution / f_RF_objectif ,n_vec_solution / nG_solution,beta_solution / betaN,q / qstar_solution)
    
    # Définir la fonction objectif à minimiser
    def objective_function_optimiseur(a):
        
        cost_fct = 0
        
        # Calculer les résultats pour le paramètre 'a' donné
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
        
        # Consideration of limits
        if n_vec_solution / nG_solution > 1:
            cost_fct = cost_fct + (n_vec_solution / nG_solution)**100
        if beta_solution / betaN > 1:
            cost_fct = cost_fct + (beta_solution / betaN)**100
        if q / qstar_solution > 1:
            cost_fct = cost_fct + (q / qstar_solution)**100
        if fRF_solution / f_RF_objectif > 1:
            cost_fct = cost_fct + (fRF_solution / f_RF_objectif)**100
        # solenoid consideration
        if np.isnan(R0_a_b_c_CS_solution) or R0_a_b_c_CS_solution<0 :
            cost_fct = cost_fct + 100
            
        # Minimization of the cost
        cost_fct = cost_fct + cost*1e6 #(cost*1e6) or R0_solution
        
        # naN verification
        if np.isnan(cost_fct)==True or cost_fct is None:
            cost_fct = 100
            
        return(cost_fct)
    
    
    # Définir les limites de la recherche pour 'a'
    bounds = (lower_bound_a, upper_bound_a)
    # Minimiser la valeur maximale en utilisant minimize_scalar
    result_minim = minimize_scalar(To_solve_raffiné, bounds=bounds)
    
    # Vérifier si l'optimisation a réussi et obtenir la valeur optimale de 'a'
    if result_minim.success:
        optimal_a = result_minim.x
    else:
        return np.nan  # Retourner NaN si minimize_scalar échoue
    
    # Si nan : retour nan
    if To_solve_raffiné(optimal_a) is None or np.isnan(To_solve_raffiné(optimal_a)):
        return np.nan
    
    # Si L'une des 4 limites supérieur à 1 alors on retourne a minisant le max des limites
    elif To_solve_raffiné(optimal_a)>1:
            return optimal_a
        
    # Si il existe une solution :
    elif To_solve_raffiné(optimal_a)<=1:
        
        # Définir les bornes pour le paramètre 'a' seulement
        bounds_a = [(lower_bound_a, upper_bound_a)]  # Limite pour 'a'

        # Utiliser la méthode de Differential Evolution pour minimiser la fonction
        result_grad = differential_evolution(objective_function_optimiseur, bounds=bounds_a, maxiter=max_iterations, tol=1e-3)
        
            # Vérifier si l'optimisation avec differential_evolution a réussi et obtenir la valeur optimale de 'a'
        if result_grad.success:
            optimal_a = result_grad.x[0]
        else:
            return np.nan  # Retourner NaN si differential_evolution échoue
        
        return (optimal_a)

def Solveur_historique(H,Bmax,P_fus,P_W,f_RF_objectif):
    delta = 0.002 # Finesse de la détection (en mètres)
    def To_solve_raffiné(a):
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
        return max(fRF_solution / f_RF_objectif ,n_vec_solution / nG_solution,beta_solution / betaN,q / qstar_solution)
    # Définir les limites de la recherche pour 'a'
    bounds = (lower_bound_a, upper_bound_a)
    # Minimiser la valeur maximale en utilisant minimize_scalar
    result = minimize_scalar(To_solve_raffiné, bounds=bounds)
    # La valeur optimale de 'a' sera dans result.x
    optimal_a = result.x
    optimal_a_raffine = np.nan
    
    # Si nan : retour nan
    if To_solve_raffiné(optimal_a) is None or np.isnan(To_solve_raffiné(optimal_a)):
        return np.nan
    
    # Si L'une des 4 limites supérieur à 1 alors on retourne a minisant le max des limites
    elif To_solve_raffiné(optimal_a)>1:
            return optimal_a
    
    # Si il existe une solution :
    elif To_solve_raffiné(optimal_a)<=1:
        # Boucle parcourant les valeurs de a
        while optimal_a <= 4:
            # Évaluer les valeurs en fonction de 'a'
            (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(optimal_a, H, Bmax, P_fus, P_W)
            # Vérifier si toutes les conditions sont satisfaites
            if (
                n_vec_solution / nG_solution < 1 and
                beta_solution / betaN < 1 and
                q / qstar_solution < 1 and
                fRF_solution / f_RF_objectif < 1
            ):
                if not np.isnan(R0_a_b_c_CS_solution):
                    optimal_a_raffine = optimal_a
                    
                # Si toutes les conditions sont satisfaites, passer à la prochaine valeur de a
                optimal_a += delta
            else:
                # Si une des conditions n'est pas satisfaite, sortir de la boucle avec la dernière valeur satisfaisant les conditions
                break 
        if optimal_a_raffine is not None and not np.isnan(optimal_a_raffine):
            optimal_a = optimal_a_raffine
        return (optimal_a-delta)


def Solveur_archaique(H,Bmax,P_fus,P_W,f_RF_objectif):
    delta = 0.002 # Finesse de la détection (en mètres)
    def To_solve_archaique(a):
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a,H,Bmax,P_fus,P_W)
        return max(fRF_solution / f_RF_objectif ,n_vec_solution / nG_solution,beta_solution / betaN,q / qstar_solution)
    # Définir les limites de la recherche pour 'a'
    bounds = (lower_bound_a, upper_bound_a)
    # Minimiser la valeur maximale en utilisant minimize_scalar
    result = minimize_scalar(To_solve_archaique, bounds=bounds)
    # La valeur optimale de 'a' sera dans result.x
    optimal_a = result.x
    
    # Si nan : retour nan
    if To_solve_archaique(optimal_a) is None or np.isnan(To_solve_archaique(optimal_a)):
        return np.nan
    
    # Si L'une des 4 limites supérieur à 1 alors on retourne a minisant le max des limites
    elif To_solve_archaique(optimal_a)>1:
            return optimal_a
    
    # Si il existe une solution :
    elif To_solve_archaique(optimal_a)<=1:
        # Boucle parcourant les valeurs de a
        while optimal_a <= 4:
            # Évaluer les valeurs en fonction de 'a'
            (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(optimal_a, H, Bmax, P_fus, P_W)
            # Vérifier si toutes les conditions sont satisfaites
            if (
                n_vec_solution / nG_solution < 1 and
                beta_solution / betaN < 1 and
                q / qstar_solution < 1 and
                fRF_solution / f_RF_objectif < 1
            ):
                optimal_a += delta
            else:
                # Si une des conditions n'est pas satisfaite, sortir de la boucle avec la dernière valeur satisfaisant les conditions
                break 
        return (optimal_a-delta)
            
# Cost function to minimize
def objective_function(x):

    cost_fct = 0    

    # x[0] corresponds to a, x[1] to H, x[2] to Bmax, x[3] to Pfus, and x[4] to Pw
    a , H , Bmax , P_fus , P_W = x
    
    # Calculate useful values
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a, H, Bmax, P_fus, P_W)

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
    cost_fct = cost_fct + (cost*1e6)
    
    # naN verification
    if np.isnan(cost_fct)==True or cost_fct is None:
        cost_fct = 100
    
    return cost_fct

def calcul(a, H, Bmax, P_fus, P_W):
    # Calculate useful values
    R0_solution = f_R0(a, P_fus, P_W, eta_T, κ)
    B0_solution = f_B0(Bmax, a, b, R0_solution)
    pbar_solution = f_pbar(Tbar, nu_n, nu_T, R0_solution, a, κ, P_fus, eta_T, nx)
    nbar_solution = f_nbar(pbar_solution, Tbar, nu_n, nu_T)
    Ip_solution = f_Ip(SL_choice, H, f_tauE(pbar_solution, R0_solution, a, κ, eta_T, P_fus), R0_solution, a, κ, nbar_solution, B0_solution, A, eta_T, P_fus)
    beta_solution = f_beta(pbar_solution, B0_solution, a, Ip_solution)
    qstar_solution = f_qstar(a, B0_solution, R0_solution, Ip_solution, κ)
    nG_solution = f_nG(Ip_solution, a)
    eta_CD_solution = f_etaCD(a, R0_solution, B0_solution, nbar_solution, Tbar, nu_n, nu_T, nx)
    fNC_solution = f_fNC(a, κ, pbar_solution, R0_solution, Ip_solution, nx)
    fB_solution = f_fB(eta_CD_solution, R0_solution, Ip_solution, nbar_solution, eta_RF, f_RP, f_RF_objectif, P_fus)
    fRF_solution = f_fRF(fNC_solution, eta_CD_solution, R0_solution, Ip_solution, nbar_solution, eta_RF, f_RP, P_fus)
    n_vec_solution = nbar_solution * 1.E-20
    c = f_coil(a, b, R0_solution, B0_solution, Sigm_max, μ0, J_max)
    cost = f_cost(a, b, R0_solution, c, κ, P_fus)
    heat = f_heat(B0_solution, R0_solution, P_fus, eta_T)
    solenoid = f_slnd(R0_solution, a, b, c)
    required_Bcs = find_required_Bcs(solenoid, R0_solution, Ip_solution, nbar_solution, Tbar, Bmax, a)
    R0_a_b_c_CS_solution = calculate_riCS(required_Bcs, solenoid)
    
    return (R0_solution, B0_solution, pbar_solution, beta_solution, nbar_solution,
            f_tauE(pbar_solution, R0_solution, a, κ, eta_T, P_fus), Ip_solution,
            qstar_solution, nG_solution, eta_CD_solution, fB_solution, fNC_solution,
            fRF_solution, n_vec_solution, c, cost, heat, solenoid, R0_solution - a,
            R0_solution - a - b, R0_solution - a - b - c, R0_a_b_c_CS_solution, required_Bcs)


# Scan on a
def Variation_a(H,Bmax,P_fus,P_W):
    
    # Utilisation de la fonction pour initialiser les listes
    (a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()
    
    # Iterate through the values of a
    a_values = np.arange(a_min, a_max, a_pas)
    
    for a in a_values:
    
        # Calculate useful values
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a, H, Bmax, P_fus, P_W)
        
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
        R0_solutions.append(R0_solution)
        R0_a_solutions.append(R0_a_solution)
        R0_a_b_solutions.append(R0_a_b_solution)
        R0_a_b_c_solutions.append(R0_a_b_c_solution)
        R0_a_b_c_CS_solutions.append(R0_a_b_c_CS_solution)
        required_BCSs.append(required_Bcs)
        Ip_solutions.append(Ip_solution)
        a_solutions.append(a)
        fRF_solutions.append(fRF_solution)
    
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
    
    return(a_solutions,nG_solutions,n_solutions,beta_solutions,qstar_solutions,fB_solutions,fNC_solutions,cost_solutions,heat_solutions,c_solutions,sol_solutions,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,Ip_solutions,fRF_solutions)



# Function to calculate riCS for a given Bcs
def calculate_riCS(Bcs,reCS):
    # Calculate riCS based on Bcs
    riCS = (μ0 * AJ_AC * σcs * reCS) / (Bcs**2 + μ0 * AJ_AC * σcs)
    return riCS

def find_required_reCS(R0,Ip,ne,Te,Bmax,a):
    
    Te = joules_to_kev(Te)
    
    L = LCFS(a) # Length of the last closed flux surface
    βp = 4/μ0 * Cw * L**2 * (ne*k*Te) / Ip**2 # 0.62 for ITER
    
    # Calculate flux needed for plasma initiation (ΨPI)
    ΨPI = ITERPI * (2 * math.pi**2 * R0 * a**2 * κ) / 836  # Assuming a constant evolution for plasma initiation consumption from ITER [Wb] with the plasma volume
    
    # Calculate flux needed for the inductive part (Ψind)
    Lp = 1.07 * μ0 * R0 * (1+0.1*βp) * (Li/2 - 2 + math.log(8 * R0/(a*math.sqrt(κ))))
    Ψind = Lp * Ip
    
    # Calculate flux related to the resistive part (Ψres)
    Ψres = 0.45 * μ0 * R0 * Ip
    
    # Function to calculate the total flux
    def total_flux_re(reCS):
        # Calculate available flux from CS system (ΨCS)
        ΨCS = math.pi * Bmax/3 * reCS**2 * 2 * Flux_CS_Utile
        # Calculate available flux from PF system (ΨPF)
        ΨPF = μ0 * Ip / (4*R0) * (βp + (Li - 3)/2 + math.log(8 * a / κ**(1/2) )) * (R0**2 - reCS**2)
        return ΨCS + ΨPF - ΨPI - Ψind - Ψres
    
    try:
        # Utiliser root_scalar pour trouver la racine (zéro) de la fonction total_flux
        result_re = root_scalar(total_flux_re, bracket=[0, 10], method='bisect', xtol=1e-4)
        result_re = result_re.root
    except ValueError:
        # Si f(a) et f(b) n'ont pas le même signe, retourner nan
        result_re = np.nan
        
    return result_re

# Function to find the required Bcs for zero plateau duration
def find_required_Bcs(reCS,R0,Ip,ne,Te,Bmax,a):
    
    Te = joules_to_kev(Te)
    
    L = LCFS(a) # Length of the last closed flux surface
    βp = 4/μ0 * Cw * L**2 * (ne*k*Te) / Ip**2 # 0.62 for ITER
    
    # Calculate available flux from PF system (ΨPF)
    ΨPF = μ0 * Ip / (4*R0) * (βp + (Li - 3)/2 + math.log(8 * a / κ**(1/2) )) * (R0**2 - reCS**2)
    
    # Calculate flux needed for plasma initiation (ΨPI)
    ΨPI = ITERPI * (2 * math.pi**2 * R0 * a**2 * κ) / 836  # Assuming a constant evolution for plasma initiation consumption from ITER [Wb] with the plasma volume
    
    # Calculate flux needed for the inductive part (Ψind)
    Lp = 1.07 * μ0 * R0 * (1+0.1*βp) * (Li/2 - 2 + math.log(8 * R0/(a*math.sqrt(κ))))
    Ψind = Lp * Ip
    
    # Calculate flux related to the resistive part (Ψres)
    Ψres = 0.45 * μ0 * R0 * Ip

    # Function to calculate the total flux
    def total_flux_B(Bcs):
        riCS = calculate_riCS(Bcs,reCS)
        # Calculate available flux from CS system (ΨCS)
        ΨCS = math.pi * Bcs/3 * (riCS**2 + (riCS * reCS) + reCS**2) * 2 * Flux_CS_Utile
        return ΨCS + ΨPF - ΨPI - Ψind - Ψres
    
    try:
        # Utiliser root_scalar pour trouver la racine (zéro) de la fonction total_flux
        result_B = root_scalar(total_flux_B, bracket=[0, Bmax], method='bisect', xtol=1e-4)
        result_B = result_B.root
    except ValueError:
        # Si f(a) et f(b) n'ont pas le même signe, retourner nan
        result_B = np.nan

    return result_B


        
def Test_working_point(bounds):
    """
    Function to test if there exists a working point for given bounds.

    Parameters:
    bounds (list): List of tuples defining the bounds for parameters (H, P_fus, P_W, Bmax).

    Returns:
    bool: True if there exists a working point, False otherwise.
    """
    Working_point = False
    H = bounds[1][1]
    P_fus = bounds[3][1]
    P_W = bounds[4][0]
    for Bmax in range(int(bounds[2][0]), int(bounds[2][1])):
        # Solve for 'a' based on given parameters
        a_solution = Solveur_fb_fnc_fonction_de_a(H, Bmax, P_fus, P_W)
        # Calculate useful values
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution, H, Bmax, P_fus, P_W)
        
        # Check if conditions for a working point are met
        if (
            n_vec_solution / nG_solution < 1 and
            beta_solution / betaN < 1 and
            q / qstar_solution < 1 and
            fB_solution / fNC_solution < 1
        ):
            Working_point = True
    
    return Working_point

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
            # Changer la couleur des étiquettes de mark
            for label in ax.get_xticklabels():
                label.set_color('black')  # Changer la couleur ici
                label.set_fontsize(12)  # Changer la taille de la police ici
                label.set_zorder(10)  # Afficher les étiquettes au-dessus du graphique
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
        
        # Définir le motif de hachure en fonction de l'état de la variable de commutation
        if hatch_switch:
            hatch = '//'
        else:
            hatch = '\\\\'  # Utilisez '\\\\' pour des hachures vers la droite
        if name !='':
            # Dashed rectangle avec le motif de hachure approprié
            ax.fill_betweenx([0.1, 0.3], dash_position_upper, dash_position_upper+length, alpha=0.3, hatch=hatch)
            # Plot the length with a different color
            ax.plot([dash_position_upper, dash_position_upper + length], [0.2, 0.2], label=name)
        else:
            ax.plot([dash_position_upper, dash_position_upper + length], [0.2, 0.2],color='black', label=name)
        # Inverser l'état de la variable de commutation pour la prochaine itération
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
    # Modifier la taille de police de l'échelle sur les deux axes (x et y)
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
    
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()

# Testing
# lengths_upper = [1.2,0.5, 0.1, 1, 0.8, 2]
# names_upper = ['','CS','', 'TFC', 'Blanket', 'Plasma']
# lengths_lower = [4.6]
# names_lower = ['R0']
# Plot_radial_build_aesthetic(lengths_upper, names_upper, lengths_lower, names_lower)

def Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design):

    # Définir la taille de la police par défaut
    plt.rcParams.update({'font.size': 17})
    
    # Plot parameter evolution
    plt.figure(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Operational domain', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une décimale pour Bmax, Pfus, Pw et H, et à deux décimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    P_W_rounded = round(P_W / 1e6, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Pw':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)

    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'Pw':
        plt.xlabel(f"$P_{{\mathrm{{w}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
    plt.ylabel("Normalized Values")
    if chosen_parameter == 'Pw':
        if first_acceptable_value is not None:
            plt.axvline(x=first_acceptable_value, color='olive', linestyle=':', label='Last acceptable value')
    else:
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
    plt.ylim(0, 1.9)
    plt.legend()
    plt.grid()
    # Save the image
    path_to_save = os.path.join(save_directory,f"Operational_Domain_{chosen_parameter}_fRF={f_RF_objectif}_Pw={int(P_W/1e6)}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()
    
def Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design):
    
    # Définir la taille de la police par défaut
    plt.rcParams.update({'font.size': 17})
    
    plt.figure(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Heat Parameter', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une décimale pour Bmax, Pfus, Pw et H, et à deux décimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    P_W_rounded = round(P_W / 1e6, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Pw':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
        
    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'Pw':
        plt.xlabel(f"$P_{{\mathrm{{w}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
    plt.ylabel("Q//")
    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle=':', label='First acceptable value')
    if chosen_design is not None:
        plt.axvline(x=chosen_design, color='red', linestyle=':', label='chosen design')
    plt.plot(parameter_values, heat_solutions, 'k-')
    plt.xlim(min(parameter_values), max(parameter_values))
    plt.grid()
    # Save the image
    path_to_save = os.path.join(save_directory,f"Heat_Parameter_{chosen_parameter}_fRF={f_RF_objectif}_Pw={int(P_W/1e6)}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()
    
def Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design):
    
    # Définir la taille de la police par défaut
    plt.rcParams.update({'font.size': 17})
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Radial Build', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une décimale pour Bmax, Pfus, Pw et H, et à deux décimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    P_W_rounded = round(P_W / 1e6, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Pw':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
        
    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'Pw':
        plt.xlabel(f"$P_{{\mathrm{{w}}}}$ [{chosen_unity}]")
    else :
        plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
    plt.ylabel("Length [m]")  # Label pour l'axe y principal
    # Tracé des données sur le premier axe y
    if first_acceptable_value is not None:
        plt.axvline(x=first_acceptable_value, color='olive', linestyle='--', label='First acceptable value')
    plt.plot(parameter_values, R0_solutions, color='green', label='$R_{\mathrm{0}}$')
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
    path_to_save = os.path.join(save_directory, f"Radial_Build_{chosen_parameter}_fRF={f_RF_objectif}_Pw={int(P_W/1e6)}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()
    
def Plot_cost_function(chosen_parameter,parameter_values,cost_solutions,first_acceptable_value,chosen_unity,chosen_design):
    
    # Définir la taille de la police par défaut
    plt.rcParams.update({'font.size': 17})
    
    # Plot cost function
    plt.figure(figsize=(8, 6))
    taille_titre_principal = 16
    taille_sous_titre = 14
    plt.suptitle('Cost function', fontsize=taille_titre_principal, fontweight='bold')
    # Arrondir les valeurs à une décimale pour Bmax, Pfus, Pw et H, et à deux décimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    P_W_rounded = round(P_W / 1e6, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    
    # Construction du titre en fonction du paramètre choisi
    if chosen_parameter == 'H':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Bmax':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter =='Pfus':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² H={H_rounded}", fontsize=taille_sous_titre)
    elif chosen_parameter == 'Pw':
        plt.title(f"Bmax={Bmax_rounded}T $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW H={H_rounded})", fontsize=taille_sous_titre)
    elif chosen_parameter == 'fobj':
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    else:
        plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
        
    if chosen_parameter == 'Bmax':
        plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='Pfus':
        plt.xlabel(f"$P_{{\mathrm{{fus}}}}$ [{chosen_unity}]")
    elif chosen_parameter =='a':
        plt.xlabel(f"a [{chosen_unity}]")
    elif chosen_parameter =='fobj':
        plt.xlabel(f"$f_{{\mathrm{{obj}}}}$ [{chosen_unity}]")
    elif chosen_parameter == 'Pw':
        plt.xlabel(f"$P_{{\mathrm{{w}}}}$ [{chosen_unity}]")
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
    path_to_save = os.path.join(save_directory,f"Cost_Function_{chosen_parameter}_fRF={f_RF_objectif}_Pw={int(P_W/1e6)}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.show()
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()
    
def Plot_tableau_valeurs(H,P_fus,P_W,Bmax,κ,chosen_design):

    a_solution = chosen_design
    # Calculate useful values
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution, H, Bmax, P_fus, P_W)
    # Données à afficher dans le tableau
    table_data = [
        ["$R_0$", R0_solution, "m"],
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
    ]
    # Affichage
    # Formater les valeurs numériques avec un seul chiffre après la virgule
    for i in range(len(table_data)):
        if isinstance(table_data[i][1], float):
            table_data[i][1] = round(table_data[i][1], 2)
    # Création d'un DataFrame Pandas
    df = pd.DataFrame(table_data, columns=["Variable", "Valeur", "Unité"])
    # Créer une liste des hauteurs de ligne avec une hauteur uniforme
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
    # Arrondir les valeurs à une décimale pour Bmax, Pfus, Pw et H, et à deux décimales pour f_obj
    Bmax_rounded = round(Bmax, 1)
    P_fus_rounded = round(P_fus / 1e9, 1)
    P_W_rounded = round(P_W / 1e6, 1)
    H_rounded = round(H, 1)
    f_obj_rounded = round(f_RF_objectif, 2)
    plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus_rounded}GW $f_{{\mathrm{{obj}}}}$={f_obj_rounded} $P_{{\mathrm{{w}}}}$={P_W_rounded}MW/m² Bmax={Bmax_rounded}T H={H_rounded}", fontsize=taille_sous_titre)
    # Save the image
    path_to_save = os.path.join(save_directory, f"Table for P_fus={int(P_fus/1e9)} f_obj={f_RF_objectif} P_w={int(P_W/1e6)} Bmax={Bmax} H={H}.png")
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    # Display the figure
    plt.show()
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()

# Test Tableau
# (H,P_fus,P_W,Bmax,κ)=init(CHOICE)
# Plot_tableau_valeurs(H,P_fus,P_W,Bmax,κ)

def Plot_radar_chart():

    # Utiliser le style "whitegrid" de Seaborn
    sns.set_style("dark")
    
    # Utiliser la palette de couleurs "husl"
    sns.set_palette("dark")
    
    # Vos données
    Initials = ['Pf', 'Pe', 'Q', 'R0', 'a', 'A', 'Kappa', 'B0', 'Bmax', 'Pw', 'H98y2', 'Ip']
    Unit = ['MW', 'MW', '', 'm', 'm', '', '', 'T', 'T', 'MW/m²', '', 'MA']
    data = [300, 250, 8, 10, 1.8, 5, 1.65, 5, 18, 0.4, 1.78, 22]  # Vos données réelles
    ITER = [500, 0, 10, 6.2, 2, 3.1, 1.78, 5.3, 12.5, 0.5, 1, 15]  # Les valeurs de référence
    
    # Création des affichages
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
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()

def Plot_bar_chart():

    # Données réelles et valeurs de référence
    Initials = ['Pf', 'Pe', 'Q', 'R0', 'a', 'A', 'κ', 'B0', 'Bmax', 'Pw', 'H98y2', 'Ip']
    Unit = ['MW', 'MW', '', 'm', 'm', '', '', 'T', 'T', 'MW/m²', '', 'MA']
    data = [300, 250, 8, 5.5, 1.8, 3.5, 1.65, 5, 16, 0.4, 1.8, 13]  # Vos données réelles
    ITER = [500, 0, 10, 6.2, 2, 3.1, 1.78, 5.3, 12.5, 0.5, 1, 15]  # Les valeurs de référence
    
    # Création des sous-plots pour chaque paramètre
    fig = make_subplots(rows=1, cols=len(Initials), subplot_titles=["" for _ in Initials])
    
    # Ajouter les barres pour chaque paramètre
    for i, param in enumerate(Initials):
        # Création des échelles personnalisées
        y_range = [0, max(data[i], ITER[i])]
        
        fig.add_trace(go.Bar(
            x=['', ''],
            y=[data[i], ITER[i]],
            marker=dict(color=['rgba(0, 0, 0, 1)', 'rgba(173, 216, 230, 0)'], line=dict(color='rgba(0,0,0,0)')),  # Pas de contour
            legendgroup=param,
            showlegend=False,
            width=0.25,  # Largeur des barres
            offset=-0.2,  # Décalage pour centrer les barres
        ), row=1, col=i+1)
    
    
    
        # Ajouter le texte du nom du paramètre en dessous du graphique
        fig.add_annotation(
            x=0,
            y=-0.15,
            text=param,
            showarrow=False,
            font=dict(size=16, color='black'),  # Taille de la police et couleur bleu foncé
            xref=f"x{i+1}",
            yref="paper",
            align="center"
        )
        # Configuration de l'axe y pour chaque paramètre
        fig.update_yaxes(range=y_range, title_text="", showticklabels=True, row=1, col=i+1, color='black')  # Changer la couleur des traits de l'échelle à noir
        
    # Ajouter les barres pour chaque paramètre
    for i, param in enumerate(Initials):
        
        # Ajouter une bande rouge à l'extrémité de la barre ITER
        fig.add_shape(type="rect",
                      xref=f'x{i+1}',
                      yref=f'y{i+1}',
                      x0=-0.2,
                      y0=ITER[i] - max(ITER[i], data[i])/200,  # Position de départ de la bande rouge
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
        title="Valeurs de données réelles et de référence pour chaque paramètre",
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
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()

def Plot_spider_chart():
    # Définition des fonctions
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
        'f_SL_param': (['SL_choice'], ['c0', 'c_I', 'c_R', 'c_a', 'c_κ', 'c_n', 'c_B', 'c_A', 'c_P']),
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
    
    # Création des listes de données pour les bulles
    nodes_x = []
    nodes_y = []
    node_texts = []
    
    # Création des listes de données pour les liens entre fonctions
    edge_x = []
    edge_y = []
    edge_texts = []
    
    # Ajout des nœuds (fonctions) aux listes de données
    for function, (inputs, outputs) in functions.items():
        nodes_x.append(function)
        nodes_y.append(len(inputs))  # Utilisation de len(inputs) pour ajuster la hauteur de la bulle
        node_texts.append(f"Inputs: {', '.join(inputs)}<br>Outputs: {', '.join(outputs)}")
    
    # Ajout des arêtes (liens entre fonctions) aux listes de données
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
    
    # Création du graphique avec bulles et liens
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
        annotation_y = len(inputs) + 0.2  # Ajustez la hauteur de l'annotation selon vos préférences
        fig.add_annotation(
            x=function,
            y=annotation_y,
            text=function,
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=8)
        )
    
    # Ajout du point représentant les constantes physiques
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
    
    # Ajout du point représentant les constantes ingénieurs
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
    
    # Ajout du point représentant les entrées du code
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
    
    # Ajout du point représentant les éléments d'initialisation
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
    # Réinitialisation des paramètres par défaut
    plt.rcdefaults()
    
#%% Fixed Pfus, Pw, H and Bmax while varying the radius 'a':

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()
  
(H,P_fus,P_W,Bmax,κ)=init(CHOICE)

# Call the a_variation function
(a_solutions,nG_solutions,n_solutions,beta_solutions,qstar_solutions,fB_solutions,fNC_solutions,cost_solutions,heat_solutions,c_solutions,sol_solutions,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,Ip_solutions,fRF_solutions) = Variation_a(H, Bmax, P_fus, P_W)

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
    print("The first acceptable value is:", first_acceptable_value)
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

Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design)

#%% Variation of a chosen parameter and avaluating the optimal (a)

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()

chosen_parameter = None
chosen_unity = None

# Ask the user to choose the parameter to vary
chosen_parameter = input("Choose the parameter to vary (H, Bmax, Pfus, Pw, fobj): ")

# Define ranges and steps for each parameter
parameter_ranges = {
    'H': np.arange(0.6, 2, 0.01),
    'Bmax': np.arange(10, 25, 0.1),
    'Pfus': np.arange(500.E6, 5000.E6, 100.E6),
    'Pw': np.arange(1.E6, 5.E6, 1.E5),
    'fobj' : np.arange(0.01,1.0,0.01)
}

unit_mapping = {'H': '', 'Bmax': 'T', 'Pfus': 'GW', 'Pw': 'MW/m²','fobj': ''}
chosen_unity = unit_mapping.get(chosen_parameter, '')

# Select the appropriate range for the chosen parameter
parameter_values = parameter_ranges.get(chosen_parameter)

a_vec = np.arange(a_min, a_max, 1/na)

(H,P_fus,P_W,Bmax,κ)=init(CHOICE)

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

        # Historical solver
        # a_solution = Solveur_historique(H,Bmax,P_fus,P_W,f_RF_objectif)
        
        # New solver
        a_solution = Solveur_raffiné(H,Bmax,P_fus,P_W,f_RF_objectif)
        
        a_solutions.append(a_solution)
        
        # Calculate useful values
        (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution, H, Bmax, P_fus, P_W)
        
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
        R0_solutions.append(R0_solution)
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

#%% Gradient Descent of Parameters to find the optimum of the Loss Function

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()

# Differential Evolution Method
bounds = [(0.3, 3),(0.4,1.2), (10, 24),(1e9,2e9),(0.5e6,4e6)] # a,H,Bmax,Pfus,Pw

# Test if there is at least a working point
Working_Point = Test_working_point(bounds)

if Working_Point == True:
    
    print("There is at least a working point")
    result = differential_evolution(objective_function, bounds, maxiter=max_iterations)
    
    a_solution = result.x[0]
    H = result.x[1]
    Bmax = result.x[2]
    P_fus = result.x[3]
    P_W = result.x[4]
    
    # Calculate useful values
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution,H,Bmax,P_fus,P_W)

    print("With a cost =", cost * 10**6)
    
    # Call the a_variation function
    (a_solutions,nG_solutions,n_solutions,beta_solutions,qstar_solutions,fB_solutions,fNC_solutions,cost_solutions,heat_solutions,c_solutions,sol_solutions,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,Ip_solutions,fRF_solutions) = Variation_a(H,Bmax,P_fus,P_W)
    
    # Plot with respect to 'a'
    chosen_parameter = 'a'
    parameter_values = a_solutions
    chosen_unity = 'm'
    chosen_design = a_solution
    first_acceptable_value = None

    Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design)

    Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design)

    Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design)
    
    Plot_tableau_valeurs(H,P_fus,P_W,Bmax,κ,chosen_design)
    
    # Plot Radial Build aesthetic
    lengths_upper = [R0_a_b_c_CS_solution,solenoid-R0_a_b_c_CS_solution, 0.1, c, b, 2*a_solution]
    names_upper = ['','CS','', 'TFC', 'Blanket', 'Plasma']
    lengths_lower = [R0_solution]
    names_lower = ['R0']
    Plot_radial_build_aesthetic(lengths_upper, names_upper, lengths_lower, names_lower)
    
    # Pe = Pe Freidberg = très approximatif
    PERSO = [round(result.x[3]*10**-6, 1), round(f_power(result.x[3])*10**-6, 1), round(10, 1),
             round(R0_solution, 1), round(result.x[0], 1), round(R0_solution/result.x[0], 1), 1.7,
             round(B0_solution, 1), round(result.x[2], 1), round(result.x[4]*10**-6, 1), round(result.x[1], 1),
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
    
elif  Working_Point == False :
    print("There is no working point")
    
    
#%% Robustness test : Results fixing H,Bmax,P_W and Pfus

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()

COMP = []
PERSO = []

# Demander à l'utilisateur de choisir une machine, ou de saisir les valeurs de H, P_E, P_W, Bmax
Reactor = input("Entrez la machine que vous souhaitez étudier : (ITER,ARC,EUdemo_A3,EUdemo_A4,CFDTR,EUDEMO2,K_DEMO) or if you want to enter new value (New) :")
if Reactor == "New" :
    H = float(input("Entrez la valeur de H : "))
    P_F = float(input("Entrez la valeur de P_F : "))
    P_W = float(input("Entrez la valeur de P_W : "))
    Bmax = float(input("Entrez la valeur de Bmax : "))
    # Demander à l'utilisateur la valeur choisie pour le design
    chosen_design = float(input("Entrez la valeur choisie pour le design : ")) 
# Choix 1: ITER
elif Reactor == 'ITER':
    COMP = ITER
    (H,P_fus,P_W,Bmax,κ)=init(1)
    chosen_design = 2
# Choix 2: ARC
elif Reactor == 'ARC':
    COMP = ARC
    (H,P_fus,P_W,Bmax,κ)=init(2)
    chosen_design = 1.13
# Choix 3: EU_DEMO_A3
elif Reactor == 'EUdemo_A3':
    COMP = EU_DEMO_A3
    (H,P_fus,P_W,Bmax,κ)=init(3)
    chosen_design = 3
# Choix 4: EU_DEMO_A4
elif Reactor == 'EUdemo_A4':
    COMP = EU_DEMO_A4
    (H,P_fus,P_W,Bmax,κ)=init(4)
    chosen_design = 2.25
# Choix 5: EU_DEMO2
elif Reactor == 'EUDEMO2':
    COMP = EU_DEMO2
    (H,P_fus,P_W,Bmax,κ)=init(5)
    chosen_design = 2.83
# Choix 6: K_DEMO_DN
elif Reactor == 'K_DEMO':
    COMP = K_DEMO_DN
    (H,P_fus,P_W,Bmax,κ)=init(6)
    chosen_design = 2.1
# Choix 7: CFDTR_DEMO
elif Reactor == 'CFDTR':
    COMP = CFDTR_DEMO
    (H,P_fus,P_W,Bmax,κ)=init(7)
    chosen_design = 2.2
else:
    print('Wrong choice')

(a_solutions,nG_solutions,n_solutions,beta_solutions,qstar_solutions,fB_solutions,fNC_solutions,cost_solutions,heat_solutions,c_solutions,sol_solutions,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,Ip_solutions,fRF_solutions)=Variation_a(H,Bmax,P_fus,P_W)

first_acceptable_value = None
# Plot with respect to 'a'
chosen_parameter = 'a'
parameter_values = a_solutions
chosen_unity = 'm'

Plot_operational_domain(chosen_parameter,parameter_values,first_acceptable_value,n_solutions,nG_solutions,beta_solutions,qstar_solutions,fRF_solutions,chosen_unity,chosen_design)

Plot_cost_function(chosen_parameter,parameter_values,cost_solutions,first_acceptable_value,chosen_unity,chosen_design)

Plot_heat_parameter(chosen_parameter,parameter_values,first_acceptable_value,chosen_unity,heat_solutions,chosen_design)

Plot_radial_build(chosen_parameter,parameter_values,chosen_unity,R0_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,Ip_solutions,first_acceptable_value,chosen_design)

Plot_tableau_valeurs(H,P_fus,P_W,Bmax,κ,chosen_design)

# Calculate useful values
(R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(chosen_design,H,Bmax,P_fus,P_W)

PERSO = [round(P_fus*10**-6, 1), round(f_power(P_fus)*10**-6, 1), round(1/f_RF_objectif, 1),
         round(R0_solution, 1), round(chosen_design, 1), round(R0_solution/chosen_design, 1), 1.7,
         round(B0_solution, 1), round(Bmax, 1), round(P_W*10**-6, 1), round(H, 1),
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

# Plot Radial Build Aesthgetic
lengths_upper = [R0_a_b_c_CS_solution,R0_a_b_c_solution-R0_a_b_c_CS_solution, 0.1, c, b, 2*chosen_design]
names_upper = ['','CS','', 'c', 'b', 'Plasma']
lengths_lower = [R0_solution]
names_lower = ['R0']
Plot_radial_build_aesthetic(lengths_upper, names_upper, lengths_lower, names_lower)

# Vos données
Initials = ['Pf', 'Pe', 'Q', 'R0', 'a', 'A', 'κ', 'B0', 'Bmax', 'Pw', 'H98y2', 'Ip']
Unit = ['MW', 'MW', '', 'm', 'm', '', '', 'T', 'T', 'MW/m²', '', 'MA']
data = PERSO
COMP = [float(x) for x in COMP]

# Création des affichages
mark = [f"{init}[{unit}]" if unit else init for init, unit in zip(Initials, Unit)]
ranges = [(0, max(val, iter_val) * 1.5) for val, iter_val in zip(data, COMP)]

# Plotting
fig1 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig1, mark, ranges)
radar.plot(COMP, "-", lw=2, color="r", alpha=0.4, label="Model")
radar.fill(COMP, alpha=0.2)
radar.plot(data, "-", lw=2, color="g", alpha=0.4, label="Data")
radar.fill(data, alpha=0.2)
radar.ax.legend()

sns.set()

# Save the image
path_to_save = os.path.join(save_directory,"Radar.png")
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
plt.show()

# Réinitialisation des paramètres par défaut
plt.rcdefaults()

#%% Plot showing variation of R0 depending on B

# Utilisation de la fonction pour initialiser les listes
(a_solutions, nG_solutions, n_solutions, beta_solutions,  qstar_solutions, fB_solutions, fNC_solutions, cost_solutions, heat_solutions, c_solutions, sol_solutions,R0_a_solutions,R0_a_b_solutions,R0_a_b_c_solutions,R0_a_b_c_CS_solutions,required_BCSs,R0_solutions,Ip_solutions,fRF_solutions) = initialize_lists()

a_physic_solutions = []
R0_physics_solutions = []

# Define ranges and steps for B parameter
chosen_unity = 'T'
parameter_values = np.arange(10, 25, 0.1)

(H,P_fus,P_W,Bmax,κ)=init(CHOICE)
P_W = 0.5e6

for Bmax in parameter_values:
    
    # New solver
    a_solution = Solveur_raffiné(H,Bmax,P_fus,P_W,f_RF_objectif)
    # Calculate useful values
    (R0_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_solution, H, Bmax, P_fus, P_W)
    # Store archaique values
    a_solutions.append(a_solution)
    R0_solutions.append(R0_solution)
    
    # Simple solver
    a_physic_solution = Solveur_archaique(H,Bmax,P_fus,P_W,f_RF_objectif)
    # Calculate useful values
    (R0_physics_solution,B0_solution,pbar_solution,beta_solution,nbar_solution,tauE_solution,Ip_solution,qstar_solution,nG_solution,eta_CD_solution,fB_solution,fNC_solution,fRF_solution,n_vec_solution,c,cost,heat,solenoid,R0_a_solution,R0_a_b_solution,R0_a_b_c_solution,R0_a_b_c_CS_solution,required_Bcs) = calcul(a_physic_solution, H, Bmax, P_fus, P_W)
    # Store new values
    a_physic_solutions.append(a_physic_solution)
    R0_physics_solutions.append(R0_physics_solution)

# Convert lists to NumPy arrays
a_physic_solutions = np.array(a_physic_solutions)
a_solutions = np.array(a_solutions)
R0_solutions = np.array(R0_solutions)
R0_physics_solutions = np.array(R0_physics_solutions)

# Nice graphics
plt.rcParams.update({'font.size': 17})
plt.figure(figsize=(8, 6))
taille_titre_principal = 16
taille_sous_titre = 14
# Titles
plt.suptitle('R0 minimisation with Bmax', fontsize=taille_titre_principal, fontweight='bold')
plt.title(f"$P_{{\mathrm{{fus}}}}$={P_fus/1e9}GW $f_{{\mathrm{{obj}}}}$={f_RF_objectif} $P_{{\mathrm{{w}}}}$={P_W/1e6}MW/m² H={H}", fontsize=taille_sous_titre)
# Labels
plt.xlabel(f"$B_{{\mathrm{{max}}}}$ [{chosen_unity}]")
plt.ylabel("R0 [m]")
# Plotting
plt.plot(parameter_values, R0_physics_solutions,color='blue',label='Pure Physics')
plt.plot(parameter_values, R0_solutions, color='red', linestyle='--',label='Physics and Radial Build')
plt.legend(loc='upper right', facecolor='lightgrey')
# Limits
plt.xlim(min(parameter_values), max(parameter_values))
plt.grid()
# Save the image
path_to_save = os.path.join(save_directory,"R0_minimisation_with_Bmax.png")
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
plt.show()
# Réinitialisation des paramètres par défaut
plt.rcdefaults()


#%% Variation of 1 chosen parameter and mapping the a space

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
max_limits_matrix = np.zeros((len(a_vec),len(parameter_values)))
radial_build_matrix = np.zeros((len(a_vec),len(parameter_values)))

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
            
            # Vérifier les conditions
            n_condition = n_vec_solution / nG_solution
            beta_condition = beta_solution / betaN
            q_condition = q / qstar_solution
            fRF_condition = fRF_solution / f_RF_objectif
            
            max_limit = max(n_condition, beta_condition, q_condition, fRF_condition)
            
            if np.isnan(max_limit) or max_limit > 2 :
                max_limit = np.nan
            
            radial_build = np.nan
            if not np.isnan(R0_a_b_c_CS_solution) and not np.isnan(max_limit) and max_limit<1 :
                radial_build = R0_solution
            
            # Store the value in the matrix
            max_limits_matrix[y, x] = max_limit
            radial_build_matrix[y,x] = radial_build        
            
            y = y+1
            
        x = x+1
        

else:
    print("Invalid chosen parameter. Please choose from 'H', 'Bmax', 'Pfus', 'Pw'.")

# Inverser l'ordre des lignes de max_limits_matrix
inverted_matrix_plasma_limit = max_limits_matrix[::-1, :]
# Inverser l'ordre des lignes de max_limits_matrix
inverted_matrix_radial_build = radial_build_matrix[::-1, :]

# Créer une figure
plt.figure(figsize=(10, 6))

color_choice_plasma = 'turbo'
# Other good choices : 'coolwarm' , 'jet' , 'gist_rainbow'
color_choice_radial = 'bone'
# Other good choice : 'plasma' , 'winter'

# Afficher le heatmap pour inverted_matrix_plasma_limit avec imshow et stocker l'objet mappable retourné
im_plasma_limit = plt.imshow(inverted_matrix_plasma_limit, cmap=color_choice_plasma, aspect='auto', interpolation='nearest') #lorsque plus de points disponibles, 'bilinear' possible afin d'interpoler de manière plus esthétique
# Définir la valeur seuil pour distinguer les deux plages de valeurs
threshold = 1.0
# Ajouter des contours pour marquer la ligne de démarcation pour inverted_matrix_plasma_limit
plt.contour(inverted_matrix_plasma_limit, levels=[threshold], colors='black', linestyles='dashed')
# Ajouter une barre de couleur avec une étiquette pour inverted_matrix_plasma_limit
cbar_plasma_limit = plt.colorbar(im_plasma_limit, label='max_limit')

# Limiter le nombre de valeurs affichées sur l'axe y
max_display_y = 20  # Nombre maximal de valeurs à afficher sur l'axe y
y_indices = np.linspace(0, len(a_vec) - 1, max_display_y, dtype=int)
y_labels = [round(a_vec[len(a_vec) - 1 - i], 1) for i in y_indices]  # Arrondir chaque valeur à un décimale
plt.yticks(y_indices, y_labels)
# Limiter le nombre de valeurs affichées sur l'axe x
max_display_x = 15  # Nombre maximal de valeurs à afficher sur l'axe x
x_indices = np.linspace(0, len(parameter_values) - 1, max_display_x, dtype=int)
if chosen_parameter == 'Pfus':
    x_labels = [round(parameter_values[i]*1e-9, 1) for i in x_indices]  # Arrondir chaque valeur à une décimale
elif chosen_parameter == 'Pw':
    x_labels = [round(parameter_values[i]*1e-6, 1) for i in x_indices]  # Arrondir chaque valeur à une décimale
else :
    x_labels = [round(parameter_values[i], 1) for i in x_indices]  # Arrondir chaque valeur à une décimale
plt.xticks(x_indices, x_labels)

# # Afficher inverted_matrix_radial_build avec une colormap et stocker l'objet mappable retourné
# im_radial_build = plt.imshow(inverted_matrix_radial_build, cmap=color_choice_radial, alpha=0.8, aspect='auto', interpolation='nearest')
# # Ajouter une colorbar avec une étiquette pour inverted_matrix_radial_build
# cbar_radial_build = plt.colorbar(im_radial_build, label='R0')

# Créer une copie de la matrice pour définir les niveaux de contour
filled_matrix = np.where(np.isnan(inverted_matrix_radial_build), -1, 1)  # Remplacer NaN par -1 et les autres par 1
# Tracer les contours autour des valeurs de transition pour inverted_matrix_radial_build
contour_level = [0.9]  # Niveau pour les contours
plt.contour(filled_matrix, levels=contour_level, colors='white')
# Définir les niveaux et les couleurs pour les contours
levels = np.arange(1, 25)  # Définit les niveaux de 1 à 24
# Ajouter des contours pour chaque niveau avec une couleur différente
contour_lines = plt.contour(inverted_matrix_radial_build, levels=levels, colors='white')
# Filtrer les niveaux pour n'afficher que ceux inférieurs à 10
filtered_levels = [level for level in contour_lines.levels if level <= 10]
# Ajouter les valeurs des niveaux à côté des contours
plt.clabel(contour_lines, filtered_levels, inline=True, fmt='%d', fontsize=8)
# Créer une ligne blanche avec une légende
white_line = mlines.Line2D([], [], color='white', label='Topological Map of R0')
# Ajouter la ligne à la légende
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(white_line)
plt.legend(handles=handles, loc='upper right', facecolor='lightgrey', fontsize=10)

# Personnaliser les axes et le titre
plt.xlabel(f"{chosen_parameter} [{chosen_unity}]")
plt.ylabel('a')
plt.title('2D Color Map based on plasma limits')

# Save the image
path_to_save = os.path.join(save_directory,f'2D_map_{chosen_parameter}')
plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
# Afficher la figure
plt.show()

# Réinitialisation des paramètres par défaut
plt.rcdefaults()

