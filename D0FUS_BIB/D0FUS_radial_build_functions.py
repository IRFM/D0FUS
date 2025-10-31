# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:32:03 2025

@author: TA276941
"""

#%% Import

from D0FUS_physical_functions import *

# Ajouter le répertoire 'D0FUS_BIB' au chemin de recherche de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'D0FUS_BIB'))

#%% print

if __name__ == "__main__":
    print("##################################################### J Model ##########################################################")

#%% Steel

def Steel(Chosen_Steel):
    if Chosen_Steel == '316L':
        σ = 660.E6        # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'N50H':
        σ = 1000.E6        # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'Manual':
        σ = σ_manual*1.E6        # Mechanical limit of the steel considered in [Pa]
    else : 
        print('Choose a valid steel')
    return(σ)

#%% Current density

def Jc_Nb3Sn(B, T, Nb3Sn_PARAMS, Eps):
    """
    Calculate the critical current density Jc for a given Nb3Sn TFEU4 wire
    as a function of the magnetic field B (in T), temperature T (in K),
    and thixotropic strain Eps.

    Arguments:
        B : scalar or array-like
            Magnetic field in tesla.
        T : scalar or array-like
            Temperature in kelvin.
        Eps : scalar or array-like
            Thixotropic strain.
        params : dict, optional
            Dictionary of parameters if customization is desired.

    Returns:
        Jc : array-like
            Critical current density in A/m^2.
    """
    # Convert to numpy array
    B = np.array(B, dtype=float)
    T = np.array(T, dtype=float)
    
    # Load parameters
    p = Nb3Sn_PARAMS.copy()
    Ca1 = p['Ca1']
    Ca2 = p['Ca2']
    Eps0a = p['Eps0a']
    Bc2m = p['Bc2m']
    Tcm = p['Tcm']
    C1 = p['C1']
    exponent_p = p['p']
    exponent_q = p['q']
    dbrin = p['dbrin']
    
    # Intermediate calculations
    Epssh = Ca2 * Eps0a / np.sqrt(Ca1**2 - Ca2**2)
    s2Eps = 1 + (1.0 / (1 - Ca1 * Eps0a)) * (
        Ca1 * (np.sqrt(Epssh**2 + Eps0a**2) - np.sqrt((Eps - Epssh)**2 + Eps0a**2))
        - Ca2 * Eps
    )
    Tc0 = Tcm * s2Eps**(1/3)
    # b0 not used in final formula, but calculable if needed
    # b0 = Bc2m * s2Eps
    t = T / Tc0
    one_minus_t152 = (1 - t**1.52)
    b = B / (Bc2m * s2Eps * one_minus_t152)
    # Tc variable not used in final Jc but can be returned
    # Tc = Tcm * s2Eps * one_minus_t152
    
    # Jc formula before geometry
    Jc_raw = (C1 / B) * s2Eps * one_minus_t152 * (1 - t**2) * b**exponent_p * (1 - b)**exponent_q
    # Division by wire cross-sectional area (circular section)
    section = np.pi * (dbrin**2) / 4
    
    Jc = Jc_raw / section
    
    return Jc

def Jc_NbTi(B, T, params):
    """
    Computes the critical current density J(B, T) for NbTi,
    using a dictionary of material parameters.
    Taken from scaling law of the NbTi strands from JT60-SA
    Tests conducted by A.Torre and reported in DEL08-K006-01C

    Parameters:
    - B : Magnetic field (Tesla), scalar or numpy array
    - T : Temperature (Kelvin), scalar or numpy array
    - params : dict containing keys:
        - 'Tco'  : Critical temperature at zero field (K)
        - 'Bc2o' : Upper critical magnetic field at 0 K (T)
        - 'Co'   : Normalization constant
        - 'gamm' : Exponent gamma
        - 'alph' : Exponent alpha
        - 'bet'  : Exponent beta

    Returns:
    - Jc : Critical current density [A/m²]
    """

    # Unpack parameters
    p = params.copy()
    Tco = p['Tco']
    Bc2o = p['Bc2o']
    Co = p['Co']
    gamm = p['gamm']
    alph = p['alph']
    bet = p['bet']

    # Reduced temperature and critical field
    t = T / Tco
    Bc2 = Bc2o * (1 - t**1.7)
    b = B / Bc2
    Tc = Tco * (1 - B / Bc2o)**(1/1.7)

    # Compute Jc
    Ic = Co / B * (1 - t**1.7)**gamm * b**alph * (1 - b)**bet * 1e6
    Jc = Ic * 5 # cf note Alex
    
    if np.iscomplexobj(Jc):
        Jc = np.full_like(Jc, np.nan)
        
    return Jc

def Jc_Rebco(B, T, REBCO_PARAMS, Tet):
    """
    Calculate the critical current density Jc for a REBCO tape according to the CERN2014 scaling.
    Inputs:
        B   : Magnetic field (T), scalar or array-like.
        T   : Temperature (K), scalar or array-like.
        Tet : Angle (rad), scalar or array-like (0 = poor orientation).
        params : Optional dict to override DEFAULT_PARAMS_REBCO.

    Returns:
        Jc : Critical current density in A/m^2.
    """
    # Tape width
    Ruban_width = 1e-4
    # Convert to numpy array
    B = np.array(B, dtype=float)
    T = np.array(T, dtype=float)
    
    # Retrieve parameters
    p = REBCO_PARAMS.copy()
    # Extract
    Tc0 = p['Tc0']; n = p['n']
    Bi0c = p['Bi0c']; Alfc = p['Alfc']; pc = p['pc']; qc = p['qc']; gamc = p['gamc']
    Bi0ab = p['Bi0ab']; Alfab = p['Alfab']; pab = p['pab']; qab = p['qab']; gamab = p['gamab']
    n1 = p['n1']; n2 = p['n2']; a = p['a']
    Nu = p['Nu']; g0 = p['g0']; g1 = p['g1']; g2 = p['g2']; g3 = p['g3']
    Trebco = p['trebco']
    
    # Reduced temperature scale
    tred = T / Tc0
    # Scaling fields for C and AB
    Bic = Bi0c * (1 - tred**n)
    Biab = Bi0ab * ((1 - tred**n1)**n2 + a * (1 - tred**n))
    # Reduced fields
    bredc = B / Bic
    bredab = B / Biab
    # Critical density component C
    Jcc = (Alfc / B) * bredc**pc * (1 - bredc)**qc * (1 - tred**n)**gamc
    # Component AB
    Jcab = (Alfab / B) * bredab**pab * (1 - bredab)**qab * ((1 - tred**n1)**n2 + a * (1 - tred**n))**gamab
    # Angular width g
    g = g0 + g1 * np.exp(-g2 * np.exp(g3 * T) * B)
    
    # Combination of the two components according to the angle
    Jc = Jcc + (Jcab - Jcc) / (1 + (np.abs(Tet - np.pi/2) / g)**Nu)
    Jc = Jc * (Trebco / Ruban_width)
    
    return Jc

def Jc(Supra_choice, B_supra , T_He):
    
    # Current density
    if Supra_choice == "Nb3Sn"  :

        # Nb3Sn TFEU4 parameters LTS
        Nb3Sn_PARAMS = {
            'Ca1': 44.48,
            'Ca2': 0.0,
            'Eps0a': 0.00256,
            'Epsm': -0.00049,
            'Bc2m': 32.97,
            'Tcm': 16.06,
            'C1': 19922.0,
            'p': 0.63,
            'q': 2.1,
            'dbrin': 0.82e-3,
            'CuNCu': 1.01,
        }
        
        Jc = Jc_Nb3Sn(B_supra, T_He + Marge_T_Helium + Marge_T_Nb3Sn , Nb3Sn_PARAMS, Eps)
        
    elif Supra_choice == "Rebco"  :

        # Paramètres par défaut pour REBCO (CERN 2014)
        REBCO_PARAMS = {
            'trebco': 1.5e-6,   # épaisseur de la couche supraconductrice [m]
            'w': 4e-3,          # largeur du ruban [m]
            'Tc0': 93.0,        # température critique zéro champ [K]
            'n': 1.0,
            'Bi0c': 140.0,      # [T]
            'Alfc': 1.41e12,    # [A/m^2]
            'pc': 0.313,
            'qc': 0.867,
            'gamc': 3.09,
            'Bi0ab': 250.0,     # [T]
            'Alfab': 83.8e12,   # [A/m^2]
            'pab': 1.023,
            'qab': 4.45,
            'gamab': 4.73,
            'n1': 1.77,
            'n2': 4.1,
            'a': 0.1,
            'Nu': 0.857,
            'g0': -0.0056,
            'g1': 0.0944,
            'g2': -0.0008,
            'g3': 0.00388,
        }
        
        Jc = Jc_Rebco(B_supra, T_He + Marge_T_Helium + Marge_T_Rebco , REBCO_PARAMS, Tet)

    elif Supra_choice == "NbTi"  :

        NbTi_PARAMS = {
            'Tco': 8.97,
            'Bc2o': 14.51,
            'Co': 24049.90,
            'gamm': 2.00,
            'alph': 0.77,
            'bet': 1.19,
        }
        
        Jc = Jc_NbTi(B_supra, T_He + Marge_T_Helium + Marge_T_NbTi , NbTi_PARAMS)

    elif Supra_choice == "Manual"  :

        Jc = 50.E6
        
    else :
        print("Please choose a proper superconductor")
    
    return(Jc)


#%% Jc test

if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    
    # Test
    B_test = 8
    T_test = 4.2
    Jc_val_NbTi = Jc("NbTi", B_test , T_test)
    print(f"Jc NbTi (T = {T_test} K, B = {B_test} T) = {Jc_val_NbTi/1e6:.2f} MA/m²")
    Jc_val_Nb3Sn = Jc("Nb3Sn", B_test , T_test)
    print(f"Jc (Nb3Sn T = {T_test} K, B = {B_test} T) = {Jc_val_Nb3Sn/1e6:.2f} MA/m²")
    Jc_val_Rebco = Jc("Rebco", B_test , T_test)
    print(f"Jc (Rebco T = {T_test} K, B = {B_test} T) = {Jc_val_Rebco/1e6:.2f} MA/m²")

    # —––––––––––––––––––––––––––––––––––––
    # 1) Plages et calculs
    B_vals = np.linspace(0, 45, 100)
    T_vals = np.linspace(2, 30, 100)
    B_mesh, T_mesh = np.meshgrid(B_vals, T_vals)
    
    # —––––––––––––––––––––––––––––––––––––
    # Plot 2D à 4.2 K
    T0 = 4.2 - Marge_T_NbTi
    J_NbTi_42 = Jc("NbTi", B_vals, T0)/1e6
    J_Nb3Sn_42 = Jc("Nb3Sn", B_vals, T0)/1e6
    J_Rebco_42 = Jc("Rebco", B_vals, T0)/1e6
    
    plt.figure(figsize=(6,4))
    plt.plot(B_vals, J_NbTi_42, label='NbTi @ 4.2 K', lw=2)
    plt.plot(B_vals, J_Nb3Sn_42, label='Nb3Sn @ 4.2 K', lw=2)
    plt.plot(B_vals, J_Rebco_42, label='Rebco @ 4.2 K', lw=2)
    plt.xlabel('Magnetic field B (T)')
    plt.ylabel('Jc (MA/m²)')
    plt.title('Jc at 4.2 K')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 45)
    plt.ylim(10, 1e4)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
    
    # —––––––––––––––––––––––––––––––––––––
    # Surfaces 3D
    
    # Calcul des densités critiques
    J_NbTi = Jc("NbTi", B_mesh, T_mesh)/1e6
    J_Nb3Sn = Jc("Nb3Sn", B_mesh, T_mesh)/1e6
    J_Rebco = Jc("Rebco", B_mesh, T_mesh)/1e6
    def plot_surface(ax, B, T, J, title):
        surf = ax.plot_surface(
            B, T, J,
            cmap='viridis',        
            edgecolor='none',      
            antialiased=True,      
            rcount=100, ccount=100 # résolution
        )
        ax.set_xlabel('B (T)')
        ax.set_ylabel('T (K)')
        ax.set_zlabel('Jc (MA/m²)')
        ax.set_title(title)
        # colorbar intégrée
        plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    
    fig = plt.figure(figsize=(12,5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_surface(ax1, B_mesh, T_mesh, J_Nb3Sn, 'Jc_Nb3Sn(B, T)')
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_surface(ax2, B_mesh, T_mesh, J_Rebco, 'Jc_Rebco(B, T)')
    
    # Ajuster l'angle de vue
    for ax in (ax1, ax2):
        ax.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    plt.show()
    
#%% Print
        
if __name__ == "__main__":
    print("##################################################### CIRCE Model ##########################################################")
    
#%% CIRCE 0D module

def MD1v2(ri, r0, re, Enmoins1, En, nu, J0, B0, J1, B1, Pi, config):
    """
    Calcul des contraintes et déplacements pour un cylindre épais avec body load en multi-couches.
    """
    K1nmoins1 = J0 * B0 / (r0 - ri)
    K1n = J1 * B1 / (re - r0)
    
    if config[0] == 1:
        K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
    else:
        K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
    
    if config[1] == 1:
        K2n = -J1 * B1 * r0 / (re - r0)
    else:
        K2n = -J1 * B1 * re / (re - r0)
    
    MD1 = (1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
      (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
      (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
      (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
      (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
      (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
      (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
      (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En) - \
          Pi * (-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2))

    return MD1

def MDendv2(ri, r0, re, Enmoins1, En, nu, J0, B0, J1, B1, Pe, config):
    """
    Calcul des contraintes et déplacements pour un cylindre épais avec body load en multi-couches.
    """
    K1nmoins1 = J0 * B0 / (r0 - ri)
    K1n = J1 * B1 / (re - r0)
    
    if config[0] == 1:
        K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
    else:
        K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
    
    if config[1] == 1:
        K2n = -J1 * B1 * r0 / (re - r0)
    else:
        K2n = -J1 * B1 * re / (re - r0)
    
    MDend = (1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
        (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
        (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
        (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
        (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
        (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
        (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
        (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En) - \
        Pe * (-2 * re**2) / (En * (re**2 - r0**2))

    return MDend

def MDv2(ri, r0, re, Enmoins1, En, nu, J0, B0, J1, B1, config):
    """
    Calcul des contraintes et déplacements pour un cylindre épais avec body load en multi-couches.
    """
    K1nmoins1 = J0 * B0 / (r0 - ri)
    K1n = J1 * B1 / (re - r0)
    
    if config[0] == 1:
        K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
    else:
        K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
    
    if config[1] == 1:
        K2n = -J1 * B1 * r0 / (re - r0)
    else:
        K2n = -J1 * B1 * re / (re - r0)
    
    MD = (1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
     (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
     (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
     (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
     (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
     (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
     (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
     (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En)

    return MD

def MG1v2(ri, r0, re, Enmoins1, En, nu):
    """
    Calcul des coefficients de la matrice de rigidité pour un cylindre épais avec body load en multi-couches.
    """
    MG1 = [(((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2))) + \
       (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2))),
       (-2*re**2)/(En*(re**2-r0**2)) ]
    
    return MG1
    
def MGendv2(ri, r0, re, Enmoins1, En, nu):
    """
    Calcul des coefficients de la matrice de rigidité pour un cylindre épais avec body load en multi-couches.
    """
    MGendv2 = [(-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2)), \
         (((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2))) + \
         (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2)))]

    return MGendv2

def MGv2(ri, r0, re, Enmoins1, En, nu):
    """
    Calcul des coefficients de la matrice de rigidité pour un cylindre épais avec body load en multi-couches.
    """
    MG = [(-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2)), \
      (((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2))) + \
      (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2))),\
         (-2*re**2)/(En*(re**2-r0**2)) ]

    return MG
   
def F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config):
    """
    F_CIRCE0D main function
    """
    
   
    nlayer = len(E)

    if nlayer == 1:        

        SigRtot = []
        SigTtot = []
        urtot = []
        P = [Pi, Pe]
        Rvec = []
        
    elif nlayer == 2:
        ri = R[0]
        r0 = R[1]
        re = R[2]
        Enmoins1 = E[0]
        En = E[1]
        J0 = J[0]
        J1 = J[1]
        B0 = B[0]
        B1 = B[1]
        K1nmoins1 = J[0] * B[0] / (R[1] -R[0])
        K1n = J1 * B1 / (re - r0)
   
        if config[0] == 1:
            K2nmoins1 = -J0 * B0 * ri / (r0 - ri)
        else:
            K2nmoins1 = -J0 * B0 * r0 / (r0 - ri)
        
        if config[1] == 1:
              K2n = -J1 * B1 * r0 / (re - r0)
        else:
            K2n = -J1 * B1 * re / (re - r0)
        
        MGtot = np.array([[((1 + nu) * re**2 + (1 - nu) * r0**2) / (En * (re**2 - r0**2)) + \
                           (((1 + nu) * ri**2 + (1 - nu) * r0**2) / (Enmoins1 * (r0**2 - ri**2)))]])
        
        MDtot = np.array([[(1 + nu) / En * re**2 * (K1n * (nu + 3) / 8 + K2n * (nu + 2) / (3 * (re + r0))) + \
                           (1 - nu) / En * ((K2n * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) + \
                           (re**2 + r0**2) * K1n * (nu + 3) / 8) - \
                       (1 + nu) / Enmoins1 * ri**2 * (K1nmoins1 * (nu + 3) / 8 + K2nmoins1 * (nu + 2) / (3 * (r0 + ri))) - \
                       (1 - nu) / Enmoins1 * ((K2nmoins1 * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) + \
                       (r0**2 + ri**2) * K1nmoins1 * (nu + 3) / 8) + \
                       (1 - nu**2) / 8 * r0**2 * (K1nmoins1 / Enmoins1 - K1n / En) + \
                       (1 - nu**2) / 3 * r0 * (K2nmoins1 / Enmoins1 - K2n / En) - \
                       Pi * (-2 * ri**2) / (Enmoins1 * (r0**2 - ri**2)) - \
                       Pe * (-2 * re**2) / (En * (re**2 - r0**2))]])
    
        P = np.linalg.inv(MGtot) @ MDtot
        SigRtot = []
        SigTtot = []
        urtot = []
        P = [Pi, P[0], Pe]
        Rvec = []
        

    else:
        MDtot = np.zeros((nlayer - 1, 1))
        MGtot = np.zeros((nlayer - 1, nlayer - 1))
    
        for ilayer in range(2, nlayer+1):
            if ilayer == 2:
                MGtot[ilayer - 2, ilayer - 2:ilayer ] = MG1v2(R[ilayer - 2], R[ilayer - 1], R[ilayer],
                                                               E[ilayer - 2], E[ilayer - 1], nu)
                MDtot[ilayer - 2] = MD1v2(R[ilayer - 2], R[ilayer - 1], R[ilayer], E[ilayer - 2], E[ilayer - 1], nu,
                                         J[ilayer - 2], B[ilayer - 2], J[ilayer - 1], B[ilayer - 1], Pi,
                                         [config[ilayer - 2], config[ilayer - 1]])
            elif ilayer == nlayer:
                MGtot[ilayer - 2, ilayer - 3:ilayer - 1] = MGendv2(R[ilayer - 2], R[ilayer - 1], R[ilayer],
                                                                 E[ilayer - 2], E[ilayer - 1], nu)
                MDtot[ilayer - 2] = MDendv2(R[ilayer - 2], R[ilayer - 1], R[ilayer], E[ilayer - 2], E[ilayer - 1], nu,
                                            J[ilayer - 2], B[ilayer - 2], J[ilayer - 1], B[ilayer - 1], Pe,
                                            [config[ilayer - 2], config[ilayer - 1]])
            else:
                MGtot[ilayer - 2, ilayer - 3 : ilayer] = MGv2(R[ilayer - 2], R[ilayer - 1], R[ilayer],
                                                               E[ilayer - 2], E[ilayer - 1], nu)
                MDtot[ilayer - 2] = MDv2(R[ilayer - 2], R[ilayer - 1], R[ilayer], E[ilayer - 2], E[ilayer - 1], nu,
                                         J[ilayer - 2], B[ilayer - 2], J[ilayer - 1], B[ilayer - 1],
                                         [config[ilayer - 2], config[ilayer - 1]])
    

        P = np.linalg.inv(MGtot) @ MDtot
        SigRtot = []
        SigTtot = []
        urtot = []
        P = [[Pi], P,[ Pe]]
        Rvec = []
        P = np.vstack(P)

    for ilayer in range(nlayer):
        K1 = J[ilayer] * B[ilayer] / (R[ilayer + 1] - R[ilayer])
        if config[ilayer] == 1:
            K2 = -J[ilayer] * B[ilayer] * R[ilayer] / (R[ilayer + 1] - R[ilayer])
        else:
            K2 = -J[ilayer] * B[ilayer] * R[ilayer + 1] / (R[ilayer + 1] - R[ilayer])

        re = R[ilayer + 1]
        ri = R[ilayer]
        r = np.linspace(R[ilayer], R[ilayer + 1],  disR)
        
        C1 = K1 * (nu + 3) / 8 + K2 * (nu + 2) / (3 * (re + ri)) - (P[ilayer] - P[ilayer + 1]) / (re**2 - ri**2)
        C2 = (K2 * (nu + 2) / 3) * ((re**2 + ri**2 + re * ri) / (re + ri)) + \
             (P[ilayer + 1] * re**2 - P[ilayer] * ri**2) / (re**2 - ri**2) + \
             (re**2 + ri**2) * K1 * (nu + 3) / 8
        
        SigR = re**2 * ri**2 / r**2 * C1 + K1 * (nu + 3) / 8 * r**2 + K2 * (nu + 2) / 3 * r - C2
        SigT = -re**2 * ri**2 / r**2 * C1 + K1 * (3 * nu + 1) / 8 * r**2 + K2 * (2 * nu + 1) / 3 * r - C2
        ur = r / E[ilayer] * (-re**2 * ri**2 / r**2 * C1 * (1 + nu) + (1 - nu**2) / 8 * K1 * r**2 + \
                               (1 - nu**2) / 3 * K2 * r - C2 * (1 - nu))
        
        SigRtot.append(SigR)
        SigTtot.append(SigT)
        urtot.append(ur)
        Rvec.append(r)
      
    SigRtot = np.concatenate(SigRtot)
    SigTtot = np.concatenate(SigTtot)
    urtot = np.concatenate(urtot)
    Rvec = np.concatenate(Rvec)  

    return SigRtot, SigTtot, urtot, Rvec, P

#%% print

if __name__ == "__main__":
    print("##################################################### TF Model ##########################################################")
    
    
#%% Academic model

def f_TF_academic(a, b, R0, σ_TF, J_max_TF, Bmax, Choice_Buck_Wedg):
    """
    Calculate the thickness of the TF coil using a 2-layer thin cylinder model.

    Parameters:
    a : float
        Minor radius (m).
    b : float
        First Wall + Breeding Blanket + Neutron Shield + Gaps (m).
    R0 : float
        Major radius (m).
    σ_TF : float
        Yield strength of the TF steel (MPa).
    μ0 : float
        Magnetic permeability of free space.
    J_max_TF : float
        Maximum current density of the chosen Supra + Cu + He (A/m²).
    Bmax : float
        Maximum magnetic field (T).
    Choice_Buck_Wedg : str
        Mechanical option, either "Bucking" or "Wedging".

    Returns:
    c : float
        TF width (m).
    ratio_tension : float
        Ratio of axial to total stress.
    """

    # 1. Calculate the central magnetic field B0 based on geometry and maximum field
    B0 = f_B0(Bmax, a, b, R0)

    # 2. Inner (inboard leg) and outer (outboard leg) radii
    R1_0 = R0 - a - b
    R2_0 = R0 + a + b

    # 3. Effective number of turns NI required to generate B0
    NI = 2 * np.pi * R0 * B0 / μ0

    # 4. Conductor cross-section required to provide the desired current
    S_cond = NI / J_max_TF

    # 5. Inner layer thickness c1 derived from the circular cross-section
    c_WP = R1_0 - np.sqrt(R1_0**2 - S_cond / np.pi)

    # 6. Calculate new radii after adding c1
    R1 = R1_0 - c_WP  # Effective inner radius
    R2 = R2_0 + c_WP  # Effective outer radius

    # 7. Calculate the tension T
    if (R2 > 0) and (R1 > 0) and (R2 / R1 > 0):
        # Tension calculation formula
        T = abs(((np.pi * B0 * 2 * R0**2) / μ0 * math.log(R2 / R1) - F_CClamp) * coef_inboard_tension)
    else:
        # Invalid geometric conditions
        return np.nan, np.nan

    # 8. Radial pressure P due to the magnetic field Bmax
    P = Bmax**2 / (2 * μ0)

    # 9. Mechanical option choice: "bucking" or "wedging"
    if Choice_Buck_Wedg == "Bucking":
        # Thickness c2 for bucking, valid if R1 >> c
        c_Nose = (B0**2 * R0**2) * math.log(R2 / R1) / (2 * μ0 * 2 * R1 * (σ_TF - P))
        σ_r = P  # Radial stress
        σ_z = T / (2 * np.pi * R1 * c_Nose)  # Axial stress
        ratio_tension = σ_z / (σ_z + σ_r)
    elif Choice_Buck_Wedg == "Wedging":
        # Thickness c2 for wedging, valid if R1 >> c
        c_Nose = (B0**2 * R0**2) / (2 * μ0 * R1 * σ_TF) * (1 + math.log(R2 / R1) / 2)
        σ_theta = P * R1 / c_Nose  # Circumferential stress
        σ_z = T / (2 * np.pi * R1 * c_Nose)  # Axial stress
        ratio_tension = σ_z / (σ_theta + σ_z)
    else:
        raise ValueError("Choose 'Bucking' or 'Wedging' as the mechanical option.")

    # 10. Total thickness c (sum of the two layers)
    c = c_WP + c_Nose

    # Verify that c_WP is valid
    if c is None or np.isnan(c) or c < 0 or c > (c_WP + c_Nose) or c > R0 - a - b:
        return np.nan, np.nan

    return c, ratio_tension

    
#%% D0FUS model

def Winding_Pack_D0FUS(R_0, a, b, sigma_max, J_max, B_max, omega, n, Choice_solving_TF_method):
    
    """
    Computes the winding pack thickness and stress ratio under Tresca criterion.
    
    Args:
        R_0: Reference radius [m]
        a, b: Geometric dimensions [m]
        sigma_max: Maximum allowable stress [Pa]
        J_max: Maximum engineering current density [A/m²]
        mu_0: Vacuum permeability [H/m]
        B_max: Peak magnetic field [T]
        omega: Scaling factor for axial load [dimensionless]
        n: Geometric factor [dimensionless]
        method: 'auto' for Brent, 'scan' for manual root search
    
    Returns:
        winding_pack_thickness: R_ext - R_sep [m]
        ratio_tension: σ_z / σ_Tresca
    """
    
    plot = False
    R_ext = R_0 - a - b

    if R_ext <= 0:
        return(np.nan, np.nan)
        # raise ValueError("R_ext must be positive. Check R_0, a, and b.")

    ln_term = np.log((R_0 + a + b) / (R_ext))
    if ln_term <= 0:
        return(np.nan, np.nan)
        # raise ValueError("Invalid logarithmic term: ensure R_0 + a + b > R_0 - a - b")

    def alpha(R_sep):
        denom = R_ext**2 - R_sep**2
        if denom <= 0:
            return np.nan
        val = (2 * B_max / (μ0 * J_max)) * (R_ext / denom)
        if np.iscomplex(val) or val < 0 or val > 1 :
            return np.nan
        return val

    def gamma(alpha_val, n_val):
        if alpha_val <= 0 or alpha_val >= 1:
            return np.nan
        A = 2 * np.pi + 4 * alpha_val * (n_val - 1)
        discriminant = A**2 - 4 * np.pi * (np.pi - 4 * alpha_val)
        if discriminant < 0:
            return np.nan
        val = (A - np.sqrt(discriminant)) / (2 * np.pi)
        if val < 0 or val > 1:
            return np.nan
        return val

    def tresca_residual(R_sep):
        a_val = alpha(R_sep)
        if np.isnan(a_val):
            return np.inf
        g_val = gamma(a_val, n)
        if np.isnan(g_val):
            return np.inf
        try:
            sigma_r = B_max**2 / (2 * μ0 * g_val)
            denom_z = R_ext**2 - R_sep**2
            if denom_z <= 0:
                return np.inf
            sigma_z = (omega / (1 - a_val)) * B_max**2 * R_ext**2 / (2 * μ0 * denom_z) * ln_term
            val = sigma_r + sigma_z - sigma_max
            return np.sign(val) * np.log1p(abs(val))
        except Exception:
            return np.inf

    # === Root search ===
    R_sep_solution = None
    residuals = []
    R_vals = np.linspace(0.001, R_ext * 0.999, 10000)

    if Choice_solving_TF_method == 'manual':
        residuals = [tresca_residual(R) for R in R_vals]
        for i in range(len(R_vals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                R_sep_solution = R_vals[i + 1]
                break
        if R_sep_solution is None:
            return np.nan, np.nan

    elif Choice_solving_TF_method == 'brentq':
        found = False
        R1 = R_ext * 0.999
        R_min = 0.001
        while R1 > R_min :
            R2 = R1 - 0.001
            try:
                if tresca_residual(R1) * tresca_residual(R2) < 0:
                    R_sep_solution = brentq(tresca_residual, R2, R1)
                    found = True
                    break
            except ValueError:
                pass
            R1 = R2
        if not found:
            return np.nan, np.nan
    else:
        raise ValueError("Invalid method. Use 'scan' or 'auto'.")

    # === Final stress calculation ===
    a_val = alpha(R_sep_solution)
    g_val = gamma(a_val, n)

    if np.isnan(a_val) or np.isnan(g_val):
        return np.nan, np.nan

    try:
        sigma_r = B_max**2 / (2 * μ0 * g_val)
        sigma_z = (omega / (1 - a_val)) * B_max**2 * R_ext**2 / (2 * μ0 * (R_ext**2 - R_sep_solution**2)) * ln_term
    except Exception:
        return np.nan, np.nan

    winding_pack_thickness = R_ext - R_sep_solution
    ratio_tension = sigma_z / sigma_max

    # === Optional plot ===
    if plot:
        residuals = [tresca_residual(R) for R in R_vals]
        residuals_abs = [np.abs(r) for r in residuals]
        plt.figure(figsize=(8, 5))
        plt.plot(R_vals, residuals_abs, label="|Tresca Residual|")
        if R_sep_solution:
            plt.axvline(R_sep_solution, color='red', linestyle='--', label=f"Solution: R_sep = {R_sep_solution:.4f}")
        plt.yscale('log')
        plt.xlabel("R_sep [m]")
        plt.ylabel("|σ_r + σ_z − σ_max| [Pa]")
        plt.title("Tresca Residual vs R_sep (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return winding_pack_thickness, ratio_tension


def Nose_D0FUS(R_ext_Nose, sigma_max, omega, B_max, R_0, a, b):
    """
    Compute the internal radius Ri based on analytical expressions.

    Parameters:
    - R_ext_Nose : float, external radius at the nose (R_ext^Nose)
    - sigma_max  : float, maximum admissible stress
    - beta       : float, dimensionless coefficient (0 ≤ beta ≤ 1)
    - B_max      : float, maximum magnetic field
    - mu_0       : float, magnetic permeability of vacuum
    - R_0, a, b  : floats, geometric parameters

    Returns:
    - Ri : float, internal radius
    """
    
    # Compute P_Nose
    P = (B_max**2) / (2 * μ0) * (R_0 - a - b) / R_ext_Nose
    
    # Compute the logarithmic term
    log_term = np.log((R_0 + a + b) / (R_0 - a - b))
    
    # Compute the full expression under the square root
    term_intermediate = (R_ext_Nose**2 / sigma_max) * (2 * P + (1 - omega) * (B_max**2 * coef_inboard_tension / μ0) * log_term)
    term = R_ext_Nose**2 - term_intermediate
    if term < 0:
        # raise ValueError("Negative value under square root. Check your input parameters.")
        return(np.nan)
    
    Ri = np.sqrt(term)
    
    return Ri

def f_TF_D0FUS(a, b, R0, σ_TF , J_max_TF, Bmax, Choice_Buck_Wedg, omega, n):
    
    """
    Calculate the thickness of the TF coil using a 2 layer thick cylinder model 

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_TF : Yield strength of the TF steel (MPa)
    μ0 : Magnetic permeability of free space
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A/m²)
    Bmax : Maximum magnetic field (T)

    Returns:
    c : TF width
    
    """
    
    debuging = 'Off'
    
    if Choice_Buck_Wedg == "Wedging":
        
        (c_WP, ratio_tension) = Winding_Pack_D0FUS( R0, a, b, σ_TF, J_max_TF, Bmax, omega, n, Choice_solving_TF_method)
        
        # Vérification que c_WP est valide
        if c_WP is None or np.isnan(c_WP) or c_WP < 0:
            return(np.nan, np.nan)
        
        c_Nose = R0 - a - b - c_WP - Nose_D0FUS(R0 - a - b - c_WP, σ_TF, omega, Bmax, R0, a, b)

        # Vérification que c_Nose est valide
        if c_Nose is None or np.isnan(c_Nose) or c_Nose < 0:
            return(np.nan, np.nan)
        
        # Vérification que la somme ne dépasse pas R0 - a - b
        if (c_WP + c_Nose) > (R0 - a - b):
            return(np.nan, np.nan)
        
        # Epaisseur totale de la bobine
        c  = c_WP + c_Nose + c_BP
        
        if __name__ == "__main__" and debuging == 'On':
            
            print(f'Winding pack width : {c_WP}')
            print(f'Nose width : {c_Nose}')
            print(f'Backplate width : {c_BP}')
    
    elif Choice_Buck_Wedg == "Bucking":
        
        (c, ratio_tension) = Winding_Pack_D0FUS(R0, a, b, σ_TF, J_max_TF, Bmax, omega, n , Choice_solving_TF_method)
        
        # Vérification que c_WP est valide
        if c is None or np.isnan(c) or c < 0 or c > R0 - a - b :
            return(np.nan, np.nan)
        
    else : 
        print( "Choose a valid mechanical configuration" )
        return(np.nan, np.nan)

    return(c, ratio_tension)

#%% TF benchmark

if __name__ == "__main__":
    
    def get_machine_parameters_TF(machine_name):

        machines = {
            "SF": {
                "a_TF": 1.6,
                "b_TF": 1.2,
                "R0_TF": 6,
                "σ_TF_tf": 660e6,
                "T_supra": 4.2,
                "Bmax_TF": 20,
                "n_TF": 1,
                "supra_TF" : "Rebco",
            },
            # """
            # Source : PEPR SupraFusion
            # """
            
            "ITER": {
                "a_TF": 2,
                "b_TF": 1.23,
                "R0_TF": 6.2,
                "σ_TF_tf": 660e6,
                "T_supra": 4.2,
                "Bmax_TF": 11.8,
                "n_TF": 1,
                "supra_TF" : "Nb3Sn",
            },
            # """
            # Source : Sborchia, C., Fu, Y., Gallix, R., Jong, C., Knaster, J., & Mitchell, N. (2008). Design and specifications of the ITER TF coils. IEEE transactions on applied superconductivity, 18(2), 463-466.
            # """
            
            "DEMO": {
                "a_TF": 2.92,
                "b_TF": 1.9,
                "R0_TF": 9.07,
                "σ_TF_tf": 660e6,
                "T_supra": 4.2,
                "Bmax_TF": 13,
                "n_TF": 1/2, # Fig 2
                "supra_TF" : "Nb3Sn",
            },
            # """
            # Source : Federici, G., Siccinio, M., Bachmann, C., Giannini, L., Luongo, C., & Lungaroni, M. (2024). Relationship between magnetic field and tokamak size—a system engineering perspective and implications to fusion development. Nuclear Fusion, 64(3), 036025.
            # """
            
            "CFETR": {
                "a_TF": 2.2,
                "b_TF": 1.52,
                "R0_TF": 7.2,
                "σ_TF_tf": 1000e6,
                "T_supra": 4.2,
                "Bmax_TF": 14,
                "n_TF": 1,
                "supra_TF" : "Nb3Sn",
            },
            # """
            # Source : Wu, Y., Li, J., Shen, G., Zheng, J., Liu, X., Long, F., ... & Han, H. (2021). Preliminary design of CFETR TF prototype coil. Journal of Fusion Energy, 40, 1-14.
            # """
            
            "EAST": {
                "a_TF": 0.45,
                "b_TF": 0.45,
                "R0_TF": 1.85,
                "σ_TF_tf": 660e6,
                "T_supra": 3.7,
                "Bmax_TF": 7.2,
                "n_TF": 1,
                "supra_TF" : "NbTi",
            },
            # """
            # Source : Chen, S. L., Villone, F., Xiao, B. J., Barbato, L., Luo, Z. P., Liu, L., ... & Xing, Z. (2016). 3D passive stabilization of n= 0 MHD modes in EAST tokamak. Scientific Reports, 6(1), 32440.
            # Source : Yi, S., Wu, Y., Liu, B., Long, F., & Hao, Q. W. (2014). Thermal analysis of toroidal field coil in EAST at 3.7 K. Fusion Engineering and Design, 89(4), 329-334.
            # Source : Chen, W., Pan, Y., Wu, S., Weng, P., Gao, D., Wei, J., ... & Chen, S. (2006). Fabrication of the toroidal field superconducting coils for the EAST device. IEEE transactions on applied superconductivity, 16(2), 902-905.
            # """
            
            "K-STAR": {
                "a_TF": 0.5,
                "b_TF": 0.35,
                "R0_TF": 1.8,
                "σ_TF_tf": 660e6,
                "T_supra": 4.2 ,
                "Bmax_TF": 7.2,
                "n_TF": 1,
                "supra_TF" : "Nb3Sn",
            },
            # """
            # Source : Oh, Y. K., Choi, C. H., Sa, J. W., Ahn, H. J., Cho, K. J., Park, Y. M., ... & Lee, G. S. (2002, January). Design overview of the KSTAR magnet structures. In Proceedings of the 19th IEEE/IPSS Symposium on Fusion Engineering. 19th SOFE (Cat. No. 02CH37231) (pp. 400-403). IEEE.
            # Source : Choi, C. H., Sa, J. W., Park, H. K., Hong, K. H., Shin, H., Kim, H. T., ... & Hong, C. D. (2005, January). Fabrication of the KSTAR toroidal field coil structure. In 20th IAEA fusion energy conference 2004. Conference proceedings (No. IAEA-CSP--25/CD, pp. 6-6).
            # Source : Oh, Y. K., Choi, C. H., Sa, J. W., Lee, D. K., You, K. I., Jhang, H. G., ... & Lee, G. S. (2002). KSTAR magnet structure design. IEEE transactions on applied superconductivity, 11(1), 2066-2069.
            # Résultats au final difficilement comparable car tehcnologie de Nb3Sn ancienne et moins performante que celle considéré ici dans notre scaling pour J
            # Si on rentre a la main la densité de courant, on retrouve bien les bons ordres de grandeur
            # """
    
            "ARC": {
                "a_TF": 1.07,
                "b_TF": 0.89,
                "R0_TF": 3.3,
                "σ_TF_tf": 660e6,
                "T_supra": 20,
                "supra_TF" : "Rebco",
                # Source :
                # Hartwig, Z. S., Vieira, R. F., Sorbom, B. N., Badcock, R. A., Bajko, M., Beck, W. K., ... & Zhou, L. (2020). VIPER: an industrially scalable high-current high-temperature superconductor cable. Superconductor Science and Technology, 33(11), 11LT01.
                # Kuznetsov, S., Ames, N., Adams, J., Radovinsky, A., & Salazar, E. (2024). Analysis of Strains in SPARC CS PIT-VIPER Cables. IEEE Transactions on Applied Superconductivity.
                # Sanabria, C., Radovinsky, A., Craighill, C., Uppalapati, K., Warner, A., Colque, J., ... & Brunner, D. (2024). Development of a high current density, high temperature superconducting cable for pulsed magnets. Superconductor Science and Technology, 37(11), 115010.
                "Bmax_TF": 23,
                "n_TF": 1,
            },
            # """
            # Source : Sorbom, B. N., Ball, J., Palmer, T. R., Mangiarotti, F. J., Sierchio, J. M., Bonoli, P., ... & Whyte, D. G. (2015). ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets. Fusion Engineering and Design, 100, 378-405.
            # """
            
            "SPARC": {
                "a_TF": 0.57,
                "b_TF": 0.18,
                "R0_TF": 1.85,
                "σ_TF_tf": 660e6,
                "T_supra": 20,    
                "Bmax_TF": 20,
                "n_TF": 1,
                "supra_TF" : "Rebco",
            },
            # """
            # Source : Creely, A. J., Greenwald, M. J., Ballinger, S. B., Brunner, D., Canik, J., Doody, J., ... & Sparc Team. (2020). Overview of the SPARC tokamak. Journal of Plasma Physics, 86(5), 865860502.
            # """
            
            "JT60-SA": {
                "a_TF": 1.18,
                "b_TF": 0.27,
                "R0_TF": 2.96,
                "σ_TF_tf": 660e6,
                "T_supra": 4.6,
                # Source : Operating Parameters in 
                # Koide, Y., Yoshida, K., Wanner, M., Barabaschi, P., Cucchiaro, A., Davis, S., ... & Zani, L. (2015). JT-60SA superconducting magnet system. Nuclear Fusion, 55(8), 086001.
                "Bmax_TF": 5.65,
                "n_TF": 1,
                "supra_TF" : "NbTi",
            }
            # """
            # Source : Polli, G. M., Cucchiaro, A., Cocilovo, V., Corato, V., Rossi, P., Drago, G., ... & Tomarchio, V. (2019). JT-60SA toroidal field coils procured by ENEA: A final review. Fusion Engineering and Design, 146, 2489-2493.
            # """
        }
    
        return machines.get(machine_name, None)
    
    
    # === BENCHMARK ===

    # === Machines à tester ===
    machines = ["CFETR", "DEMO", "ITER", "ARC" , "SPARC", "JT60-SA", "EAST", "K-STAR"]
    
    # === Résultats stockés ===
    table = []
    
    for machine in machines:
        params = get_machine_parameters_TF(machine)
        if params is None:
            continue
        
        # Paramètres d'entrée
        a = params["a_TF"]
        b = params["b_TF"]
        R0 = params["R0_TF"]
        σ = params["σ_TF_tf"]
        T_supra = params["T_supra"]
        Bmax = params["Bmax_TF"]
        n = params["n_TF"]
        Supra_choice_TF = params["supra_TF"]
    
        Jmax = Jc(Supra_choice_TF, Bmax, T_supra) * f_Cu * f_Cool * f_In
        Jmax_MApm2 = Jmax / 1e6
    
        def clean_result(val):
            if isinstance(val, tuple):
                val = val[0]
            if np.iscomplex(val) or val is None or np.isnan(val):
                return np.nan
            return round(val, 2)
    
        # === CIRCE ===
        out_wedg_CIRCE = f_TF_D0FUS(a, b, R0, σ, Jmax, Bmax, "Wedging", omega=0.5, n=n)
        out_buck_CIRCE = f_TF_D0FUS(a, b, R0, σ, Jmax, Bmax, "Bucking", omega=1.0, n=n)
    
        # Ajout au tableau
        table.append({
            "Machine": machine,
            "J [MA/m²]" : clean_result(Jmax_MApm2),
            "Wedging (m)": clean_result(out_wedg_CIRCE),
            "Bucking (m)": clean_result(out_buck_CIRCE),
        })
    
    # === Conversion en DataFrame ===
    df = pd.DataFrame(table)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.title("TF thickness results with D0FUS model", fontsize=14, pad=20)
    plt.show()

#%% TF plot
    
if __name__ == "__main__":

    # Extract individual parameters from the dictionary
    # ARC like
    a_TF = 3
    b_TF = 1.7
    R0_TF = 9
    # Default values
    σ_TF_tf = 660e6
    n_TF = 1

    # === Bmax_TF RANGE DEFINITION ===
    Bmax_values = np.linspace(0, 25, 50)  # Magnetic field range (0 T to 25 T)

    # === INITIALIZE RESULT LISTS ===
    academic_w = []
    academic_b = []
    d0fus_w = []
    d0fus_b = []
    # CIRCE_w = []
    # CIRCE_b = []

    # === COMPUTATION LOOP ===
    for Bmax_TF in Bmax_values:
        
        T_supra = 4.2
        J_max_TF_tf = Jc("Rebco", Bmax_TF, T_supra) * f_Cu * f_Cool * f_In
        
        # Academic models
        res_acad_w = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, Bmax_TF, "Wedging")
        res_acad_b = f_TF_academic(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, Bmax_TF, "Bucking")

        # D0FUS models (γ = 0.5 for Wedging, γ = 1 for Bucking)
        res_d0fus_w = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, Bmax_TF, "Wedging", 0.5, n_TF)
        res_d0fus_b = f_TF_D0FUS(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, Bmax_TF, "Bucking", 1, n_TF)

        # D0FUS models (γ = 0.5 for Wedging, γ = 1 for Bucking)
        # res_CIRCE_w = f_TF_CIRCE(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, Bmax_TF, "Wedging", 0.5, n_TF)
        # res_CIRCE_b = f_TF_CIRCE(a_TF, b_TF, R0_TF, σ_TF_tf, J_max_TF_tf, Bmax_TF, "Bucking", 1, n_TF)

        # Store results (only first return value assumed to be thickness)
        academic_w.append(res_acad_w[0])
        academic_b.append(res_acad_b[0])
        d0fus_w.append(res_d0fus_w[0]-0.07)
        d0fus_b.append(res_d0fus_b[0])
        # CIRCE_w.append(res_CIRCE_w[0]-0.07)
        # CIRCE_b.append(res_CIRCE_b[0])
    
    # Couleurs par modèle : bleu, vert, rouge
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # MADE fit
    
    x = np.array([0, 11.25, 13.25, 14.75, 16, 17, 18, 19, 19.75,20.5, 21.25, 22])
    y = np.array([0, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.65])
    
    # --- FIGURE 1 : Wedging ---
    plt.figure(figsize=(7, 7))
    
    plt.plot(Bmax_values, academic_w, color=colors[0], linestyle='-', linewidth=2,
             marker='o', markersize=6, markevery=5, label='Academic Wedging')
    plt.plot(Bmax_values, d0fus_w, color=colors[1], linestyle='-', linewidth=2,
             marker='s', markersize=6, markevery=5, label='D0FUS Wedging')
    # plt.plot(Bmax_values, CIRCE_w, color=colors[2], linestyle='-', linewidth=2,
             # marker='^', markersize=6, markevery=5, label='CIRCE Wedging')
    plt.scatter(x, y, color="black", label="MADE")
    
    plt.xlabel('Maximum magnetic field Bmax (T)', fontsize=14)
    plt.ylabel('TF thickness [m]', fontsize=14)
    plt.title('Mechanical models comparison: Wedging', fontsize=16)
    plt.legend(fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # --- FIGURE 2 : Bucking ---
    plt.figure(figsize=(7, 7))
    
    plt.plot(Bmax_values, academic_b, color=colors[0], linestyle='-', linewidth=2,
             marker='o', markersize=6, markevery=5, label='ACADEMIC Bucking')
    plt.plot(Bmax_values, d0fus_b, color=colors[1], linestyle='-', linewidth=2,
             marker='s', markersize=6, markevery=5, label='D0FUS Bucking')
    # plt.plot(Bmax_values, CIRCE_b, color=colors[2], linestyle='-', linewidth=2,
             # marker='^', markersize=6, markevery=5, label='CIRCE Bucking')
    
    plt.xlabel('Maximum magnetic field Bmax (T)', fontsize=14)
    plt.ylabel('TF thickness [m]', fontsize=14)
    plt.title('Mechanical models comparison: Bucking', fontsize=16)
    plt.legend(fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

#%% Print
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% Magnetic flux calculation

def Magnetic_flux(Ip, I_Ohm, Bmax, a, b, c, R0, κ, nbar, Tbar, Ce, Temps_Plateau, Li, Choice_Buck_Wedg):
    """
    Calculate the magnetic flux components for a tokamak plasma.

    Parameters:
    Ip : float
        Plasma current (MA).
    I_Ohm : float
        Ohmic current (MA).
    Bmax : float
        Maximum magnetic field (T).
    a : float
        Minor radius (m).
    b : float
        First Wall + Breeding Blanket + Neutron Shield + Gaps (m).
    c : float
        TF coil width (m).
    R0 : float
        Major radius (m).
    κ : float
        Elongation.
    nbar : float
        Mean electron density (1e20 particles/m³).
    Tbar : float
        Mean temperature (keV).
    Ce : float
        Ejima constant
    Temps_Plateau : float
        Plateau time (s).
    Li : float
        Internal inductance.

    Returns:
    ΨPI : float
        Flux needed for plasma initiation (Wb).
    ΨRampUp : float
        Total ramp-up flux (Wb).
    Ψplateau : float
        Flux related to the plateau (Wb).
    ΨPF : float
        Available flux from the PF system (Wb).
    """

    # Convert currents from MA to A
    Ip = Ip * 1e6
    I_Ohm = I_Ohm * 1e6

    # Toroidal magnetic field
    B0 = f_B0(Bmax, a, b, R0)

    # Length of the last closed flux surface
    L = np.pi * np.sqrt(2 * (a**2 + (κ * a)**2))

    # Poloidal beta
    βp = 4 / μ0 * L**2 * nbar * 1e20 * E_ELEM * 1e3 * Tbar / Ip**2  # 0.62 for ITER # Boltzmann constant [J/keV]

    # External radius of the CS
    if Choice_Buck_Wedg == 'Bucking':
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        print("Choose between Wedging and Bucking")
        return (np.nan, np.nan, np.nan)

    #----------------------------------------------------------------------------------------------------------------------------------------
    #### Flux calculation ####

    # Flux needed for plasma initiation (ΨPI)
    ΨPI = ITERPI  # Initiation consumption from ITER 20 [Wb]

    # Flux needed for the inductive part (Ψind)
    if (8 * R0 / (a * math.sqrt(κ))) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        Lp = 1.07 * μ0 * R0 * (1 + 0.1 * βp) * (Li / 2 - 2 + math.log(8 * R0 / (a * math.sqrt(κ))))
        Ψind = Lp * Ip

    # Flux related to the resistive part (Ψres)
    Ψres = Ce * μ0 * R0 * Ip

    # Total ramp-up flux (ΨRampUp)
    ΨRampUp = Ψind + Ψres

    # Plateau flux calculation
    eta = 2.8e-8 / (Tbar**1.5) # Spitzer resistivity from Wesson
    R_eff = eta * (2 * R0) / (a**2 * κ) # Resistivity with A = pi a^2 kappa
    Vloop = R_eff * I_Ohm # Loop voltage
    Ψplateau = Vloop * Temps_Plateau  # Plateau flux
    
    # Available flux from PF system (ΨPF)
    if (8 * a / κ**(1 / 2)) <= 0:
        return (np.nan, np.nan, np.nan)
    else:
        ΨPF = μ0 * Ip / (4 * R0) * (βp + (Li - 3) / 2 + math.log(8 * a / κ**(1 / 2))) * (R0**2) # (R0**2 - RCS_ext**2) ?

    # Theoretical expression of CS flux
    # ΨCS = 2 * (math.pi * μ0 * J_max_CS * Alpha) / 3 * (RCS_ext**3 - RCS_int**3)

    return (ΨPI, ΨRampUp, Ψplateau, ΨPF)

#%% Flux generation test

if __name__ == "__main__":
    
    # ITER
    a_cs = 2
    b_cs = 1.25
    c_cs = 0.90
    R0_cs = 6.2
    Bmax_cs = 13
    configuration = "Wedging"
    T_CS = 4.2
    κ_cs = 1.7                    # Elongation
    Tbar_cs = 8                   # Average temperature [keV]
    nbar_cs = 1                   # Average density [10^20 m^-3]
    Ip_cs = 15                    # Plasma current [MA]
    I_Ohm_cs = 3                  # Ohmic current wanted [MA]
    Ce_cs = 0.45                  # Efficiency coefficient
    Temps_Plateau_cs = 10 * 60    # Plateau duration
    Li_cs = 0.8                   # Internal inductance
    p_bar = 0.2
    
    # Compute total magnetic flux contributions using provided function
    ΨPI, ΨRampUp, Ψplateau, ΨPF = Magnetic_flux(
        Ip_cs, I_Ohm_cs, Bmax_cs,
        a_cs, b_cs, c_cs, R0_cs,
        κ_cs, nbar_cs, Tbar_cs,
        Ce_cs, Temps_Plateau_cs, Li_cs, Choice_Buck_Wedg
    )
    
    # Print benchmark results in a clear format
    print("\n=== Magnetic Flux Benchmark ===")
    print(f"Machine                      ITER  D0FUS")
    print(f"Required Ψ initiation phase : 20  : {ΨPI:.2f} Wb")
    print(f"Required Ψ ramp-up phase    : 200 : {ΨRampUp:.2f} Wb")
    print(f"Required Ψ flat-top phase   : 36  : {Ψplateau:.2f} Wb")
    print(f"Provided Ψ PF               : 115  : {ΨPF:.2f} Wb")
    print(f"Needed Ψ CS                 : 137 : {ΨPI+ΨRampUp+Ψplateau-ΨPF:.2f} Wb")

#%% CS academic model

def f_CS_academic(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, Bmax, σ_CS, J_max_CS, Choice_Buck_Wedg):
    """
    Calculate the CS thickness considering a thin layer approximation and a 2-cylinder (superconductor + steel) approach.

    Parameters:
    ΨPI : float
        Flux needed for plasma initiation (Wb).
    ΨRampUp : float
        Total ramp-up flux (Wb).
    Ψplateau : float
        Flux related to the plateau (Wb).
    ΨPF : float
        Available flux from the PF system (Wb).
    a : float
        Minor radius (m).
    b : float
        First Wall + Breeding Blanket + Neutron Shield + Gaps (m).
    c : float
        TF coil thickness (m).
    R0 : float
        Major radius (m).
    Bmax : float
        Maximum magnetic field (T).
    σ_CS : float
        Yield strength of the CS steel (MPa).
    μ0 : float
        Magnetic permeability of free space.
    J_max_CS : float
        Maximum current density of the chosen superconductor + Cu + He (A/m²).
    Choice_Buck_Wedg : str
        Mechanical configuration ('Bucking' or 'Wedging').

    Returns:
    tuple: A tuple containing the calculated values:
        d : CS width (m).
        alpha : Percentage of conductor.
        B_CS : CS magnetic field (T).
    """

    # External radius of the CS
    if Choice_Buck_Wedg == 'Bucking':
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        print("Choose between Wedging and Bucking")
        return (np.nan, np.nan, np.nan)

    # First layer solely composed of conductor
    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF
    RCS_sep = (RCS_ext**3 - (3 * ΨCS) / (2 * np.pi * μ0 * J_max_CS))**(1/3)
    B_CS = μ0 * J_max_CS * (RCS_ext - RCS_sep)
    
    if B_CS > Bmax :
        return (np.nan, np.nan, np.nan)

    # Second layer solely composed of steel
    def CS_to_solve(d_SS):
        """
        Calculate the mechanical stress and return the difference with the target stress σ_CS.
        """
        # J cross B magnetic pressure
        P_CS = B_CS**2 / (2 * μ0)
        # Magnetic pressure
        P_TF = Bmax**2 / (2 * μ0)
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = np.nanmax([P_TF, abs(P_CS - P_TF)]) * RCS_sep / d_SS
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = P_CS * RCS_sep / d_SS
        elif Choice_Buck_Wedg == 'Plug':
            Sigma_CS = np.nanmax([(P_CS-P_TF),P_TF])
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
        val = Sigma_CS - σ_CS
        # Convert to log to smooth abrupt jumps and facilitate root finding
        return np.sign(val) * np.log1p(abs(val))

    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualize the function over a given range to understand its behavior.
        """
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='CS_to_solve(d_SS)')
        plt.axhline(y=0, color='r', linestyle='--', label='y=0')
        plt.grid(True)
        plt.xlabel('d_SS')
        plt.ylabel('Function Value')
        plt.title('Behavior of CS_to_solve')
        plt.legend()
        # Identify points where the function changes sign
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        return zero_crossings

    # Detect sign changes
    def find_sign_changes(f, a, b, n=1000):
        """
        Detect intervals where the function changes sign.
        """
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        sign_changes = []
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Sign change detected
                sign_changes.append((x_vals[i-1], x_vals[i]))
        return sign_changes

    # Refine zero crossings with brentq
    def refine_zeros(f, sign_change_intervals):
        """
        Use Brent's method to refine zero crossings.
        """
        roots = []
        for a, b in sign_change_intervals:
            try:
                root = brentq(f, a, b)  # Brent guarantees convergence if a sign change is detected
                roots.append(root)
            except ValueError:
                pass  # If Brent fails, ignore the interval
        return roots if roots else [np.nan]  # If no roots are found, return np.nan

    # Find solutions using Brent's method
    def find_d_SS_solution_brentq(a, b, n=100):
        """
        Find zero crossings of CS_to_solve between a and b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
        return roots

    # Verify that the solution is a valid zero crossing
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-6):
        """
        Verify if root is a true zero crossing and not a local minimum.
        - Check that |f(root)| is close to 0 (precision tol)
        - Check for a sign change around root (epsilon)
        """
        if np.abs(f(root)) > tol:
            return False  # The function is not close to 0 -> failure
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
        if np.sign(f_left) != np.sign(f_right):  # Sign change around root
            return True
        else:
            return False  # Probably a local minimum

    # Find the smallest valid solution with fsolve
    def find_d_SS_solution_fsolve(initial_guesses):
        """
        Find a solution with fsolve ensuring it is a valid zero crossing.
        - initial_guesses: List of starting points for fsolve
        - Return the smallest valid solution or np.nan if no correct root is found
        """
        valid_solutions = []
        for guess in initial_guesses:
            root_candidate, info, ier, msg = fsolve(CS_to_solve, guess, full_output=True)
            if ier == 1:  # Check if convergence was successful
                root = root_candidate[0]
                if is_valid_root(CS_to_solve, root):  # Verify it is a valid zero crossing
                    valid_solutions.append(root)
        return min(valid_solutions) if valid_solutions else np.nan  # Take the smallest solution

    def find_d_SS_solution_root(a, b):
        """
        Find zero crossings of CS_to_solve using scipy.optimize.root.
        """
        initial_guesses = np.linspace(a, b, 10)
        valid_solutions = []
        for guess in initial_guesses:
            sol = root(CS_to_solve, guess, method='hybr')
            if sol.success and is_valid_root(CS_to_solve, sol.x[0]):
                valid_solutions.append(sol.x[0])
        return min(valid_solutions) if valid_solutions else np.nan

    def is_valid_root(f, root, epsilon=0.01, tol=1e6):
        """
        Verify if root is a true zero crossing and not a local minimum.
        """
        if np.abs(f(root)) > tol:
            return False  # The function is not close to 0 -> failure
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
        return np.sign(f_left) != np.sign(f_right)  # Sign change around root

    def CS_filters(d_SS_solution):
        tol = 1e-6
        # Preliminary calculations
        RCS_int_solution = RCS_sep - d_SS_solution
        alpha = (RCS_ext**2 - RCS_sep**2) / (RCS_ext**2 - RCS_int_solution**2)
        # Apply robust constraints with tolerances
        if d_SS_solution < tol or d_SS_solution > RCS_sep - tol:
            return False
        if RCS_int_solution < tol:
            return False
        if alpha < tol or alpha > 1 - tol:
            return False
        return True

    try:
        """
        ### `fsolve`
        Based on the Newton-Raphson method (and its quasi-Newton variants)
        Uses the local derivative to adjust the root approximation
        Problem: May converge to a local minimum instead of a true zero crossing if the derivative is weak
        ### `brentq`
        Based on Brent's method (hybrid between bisection and secant)
        Requires an interval `[a, b]` with a sign change
        Advantage: Guaranteed to find an exact zero crossing if a sign change is detected
        ### `root`
        General method to find the roots of a nonlinear function
        Allows using multiple algorithms (`hybr`, `lm`, `broyden1`, `broyden2`, etc.)
        By default (`method='hybr'`), it relies on an approach derived from Powell's algorithm (similar to Newton-Raphson)
        Problem: Like `fsolve`, it may converge to a local minimum instead of a true zero crossing if the derivative is weak
        ### `manual`
        General method to find the roots of a nonlinear function
        Custom method iterating over all solutions and looking for sign changes
        Problem: Less precise and computationally expensive
        Advantage: Impossible to be more robust and simple
        """
        if Choice_solving_CS_method == "fsolve":
            # List of initial estimates to cover multiple possible roots
            initial_guesses = np.linspace(0, RCS_sep, 10)  # 10 points between 0 and RCS_sep
            # Find the solution with fsolve
            d_SS_solution = find_d_SS_solution_fsolve(initial_guesses)
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Take the smallest solution
            else:
                return (np.nan, np.nan, np.nan)
        elif Choice_solving_CS_method == "brentq":
            # Find the solution with brentq
            d_SS_solutions = find_d_SS_solution_brentq(0, RCS_sep)
            # Filter valid solutions (exclude np.nan) and select the smallest
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Take the smallest solution
            else:
                return (np.nan, np.nan, np.nan)
        elif Choice_solving_CS_method == "root":
            d_SS_solutions = find_d_SS_solution_root(0, RCS_sep)
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Take the smallest solution
            else:
                return (np.nan, np.nan, np.nan)
        elif Choice_solving_CS_method == "manual":
            d_SS_solutions = plot_function_CS(CS_to_solve, [0, RCS_sep])
            # Filter valid solutions (exclude np.nan) and select the smallest
            valid_solutions = [sol for sol in d_SS_solutions if not np.isnan(sol) and CS_filters(sol)]
            if valid_solutions:
                d_SS_solution = min(valid_solutions)  # Take the smallest solution
            else:
                return (np.nan, np.nan, np.nan)
        else:
            print("Choose a valid method for the CS")
            return (np.nan, np.nan, np.nan)

        #### Results filtering ####
        tol = 1e-6
        # Preliminary calculations
        RCS_int_solution = RCS_sep - d_SS_solution
        alpha = (RCS_ext**2 - RCS_sep**2) / (RCS_ext**2 - RCS_int_solution**2)
        # Apply robust constraints with tolerances
        if d_SS_solution < tol or d_SS_solution > RCS_sep - tol:
            return (np.nan, np.nan, np.nan)
        elif alpha < tol or alpha > 1 - tol:
            return (np.nan, np.nan, np.nan)
        elif RCS_int_solution < tol:
            return (np.nan, np.nan, np.nan)
        else:
            d = RCS_ext - RCS_int_solution
            return (d, alpha, B_CS)
    except Exception as e:
        return (np.nan, np.nan, np.nan)

    
#%% D0FUS model

def f_CS_D0FUS(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0 , Bmax, σ_CS, J_max_CS, Choice_Buck_Wedg):
    
    """
    Calculate the CS thickness considering a thick layer approximation and a 2 cylinder (supra + steel) approach

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    c : TF thickness (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_CS : Yield strength of the CS steel (Pa)
    μ0 : Magnetic permeability of free space
    J_max_CS : Maximum current density of the chosen Supra + Cu + He (A/m²)
    Choice_Buck_Wedg : Mechanical configuration

    Returns:
    tuple: A tuple containing the calculated values :
    (d, Alpha, B_CS)
    With
    d : CS width
    Alpha : % of conductor
    B_CS : CS magnetic field
    
    Always pay attention : the calculations are not exact, so for filtering, 
    and manipulation of the solutions, need to take a margin ~ 1e-4
    (has been the cause of several bugs especially in the research of the best solution)
    
    """
    
    debuging_CS = 0
    
    # External radius of the CS
    if Choice_Buck_Wedg == 'Bucking':
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        print("Choose between Wedging and Bucking")
        return (np.nan, np.nan, np.nan)
    
    # Fonction cible
    def CS_to_solve(d):
        
        """
        Calcule la contrainte mécanique et retourne la différence avec la contrainte cible σ_CS.
        """
        
        # Preliminary calculations
        RCS_int = RCS_ext - d
        # Dilution coeficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        # magnetic pressure to support from the CS
        P_CS = B_CS**2 / (2 * μ0)
        # magnetic pressure to support from the TF
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / (RCS_ext)
    
        if Choice_Buck_Wedg == 'Bucking':
            # Light bucking
            Sigma_light = (P_CS * (RCS_ext**2 + RCS_int**2) / (RCS_ext**2 - RCS_int**2) - (2 * P_TF * RCS_ext**2) / (RCS_ext**2 - RCS_int**2))  * 1 / (1 - alpha)
            # Strong bucking
            Sigma_strong = (2 * P_TF * RCS_ext**2) / (RCS_ext**2 - RCS_int**2) * (1 / (1 - alpha))
            # Final sigma
            Sigma_CS = np.nanmax([abs(Sigma_light), abs(Sigma_strong)])
            
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = 1 / (1 - alpha) * (P_CS * ((RCS_ext**2 + RCS_int**2) / (RCS_ext**2 - RCS_int**2)))
            
        elif Choice_Buck_Wedg == 'Plug':
            Sigma_CS = 1 / (1 - alpha) * np.nanmax([(P_CS-P_TF),P_TF])
            
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
            
        val = Sigma_CS - σ_CS
        # passage en log afin de lisser les sauts abruptes qui peuvent arriver et faciliter la recherche de racines
        return np.sign(val) * np.log1p(abs(val))
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        
        plot_option_function = 0
        
        if plot_option_function ==1:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', label='CS_to_solve(d)')
            plt.axhline(y=0, color='r', linestyle='--', label='y=0')
            plt.grid(True)
            plt.xlabel('d')
            plt.ylabel('Function Value')
            plt.ylim(-1e10,1e10)
            plt.title('Comportement de CS_to_solve')
            plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        return zero_crossings
    
    # Détection des changements de signe
    def find_sign_changes(f, a, b, n):
        """ 
        Détecte les intervalles où la fonction change de signe.
        """
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        
        sign_changes = []
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Changement de signe détecté
                sign_changes.append((x_vals[i-1], x_vals[i]))
    
        return sign_changes
    
    # Raffinement des passages par zéro avec brentq
    def refine_zeros(f, sign_change_intervals):
        """ 
        Utilise la méthode de Brent pour affiner les passages par zéro.
        """
        roots = []
        for a, b in sign_change_intervals:
            try:
                root = brentq(f, a, b)  # Brent garantit une convergence si un changement de signe est détecté
                roots.append(root)
            except ValueError:
                pass  # Si Brent échoue, on ignore l'intervalle
    
        return roots if roots else [np.nan]  # Si aucune racine n'est trouvée, retourne np.nan
    
    # Recherche des solutions en utilisant la méthode de Brent
    def find_d_solution_brentq(a, b, n = 1000):
        """
        Trouve les passages par zéro de CS_to_solve entre a et b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
    
        return roots
    
    # Vérification que la solution est bien un passage par zéro
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-4):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        - Vérifie que |f(root)| est proche de 0 (précision tol)
        - Vérifie un changement de signe autour de root (epsilon)
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
    
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
    
        if np.sign(f_left) != np.sign(f_right):  # Changement de signe autour de root
            return True
        else:
            return False  # Probablement un minimum local
    
    # Trouver la plus petite solution valide avec fsolve
    def find_d_solution_fsolve(initial_guesses):
        """
        Cherche une solution avec fsolve en s'assurant qu'il s'agit bien d'un passage par zéro.
        - initial_guesses : Liste de points de départ pour fsolve
        - Retourne la plus petite solution valide ou np.nan si aucune racine correcte n'est trouvée
        """
        valid_solutions = []
    
        for guess in initial_guesses:
            root_candidate, info, ier, msg = fsolve(CS_to_solve, guess, full_output=True)
    
            if ier == 1:  # Vérifier si la convergence a réussi
                root = root_candidate[0]
                if is_valid_root(CS_to_solve, root):  # Vérifier que c'est bien un passage par zéro
                    valid_solutions.append(root)
    
        return min(valid_solutions) if valid_solutions else np.nan  # Prendre la plus petite solution
    
    def find_d_solution_root(a, b):
        """
        Trouve les passages par zéro de CS_to_solve en utilisant scipy.optimize.root.
        """
        initial_guesses = np.linspace(a, b, 10)
        valid_solutions = []
    
        for guess in initial_guesses:
            sol = root(CS_to_solve, guess, method='hybr')
            if sol.success and is_valid_root(CS_to_solve, sol.x[0]):
                valid_solutions.append(sol.x[0])
        return min(valid_solutions) if valid_solutions else np.nan

    def CS_filters(d_solution):
        
        tol=1e-6
        
        # Preliminary calculations
        RCS_int = RCS_ext - d_solution
        # Dilution coefficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        # Apply robust constraints with tolerances
        if d_solution < tol or d_solution > RCS_ext - tol:
            return False
        if B_CS < tol or B_CS > Bmax - tol:
            return False
        if alpha <  tol or alpha > 1 - tol:
            return False
        
        return True
    
    try:
        
        """
        ### `fsolve`  
        Basé sur la méthode de Newton-Raphson (et ses variantes quasi-Newtoniennes)
        Utilise la dérivée locale pour ajuster l'approximation de la racine  
        Problème : Peut converger vers un **minimum local** au lieu d'un vrai passage par zéro si la dérivée est faible  
        
        ### `brentq`  
        Basé sur la méthode de Brent (hybride entre la bissection et la sécante)
        Nécessite un intervalle `[a, b]` avec un changement de signe
        Avantage : Garantie de trouver un passage exact par zéro si un changement de signe est détecté
        
        ### `root`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Permet d'utiliser plusieurs algorithmes (`hybr`, `lm`, `broyden1`, `broyden2`, etc.)  
        Par défaut (`method='hybr'`), il repose sur une approche dérivée de **l’algorithme de Powell** (similaire à Newton-Raphson)  
        Problème : Comme `fsolve`, il peut converger vers un minimum local au lieu d’un vrai passage par zéro si la dérivée est faible
        
        ### `manual`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Méthode maison parcourant toute les solutions et cherchant les changement de signe
        Problème : peu précise et couteuse en temps de calcul
        Avantage : Impossible de faire plus robuste et simple
        """
        if Choice_solving_CS_method == "fsolve":
            
            # Liste d'estimations initiales pour couvrir plusieurs racines possibles
            initial_guesses = np.linspace(0, RCS_ext, 10)  # 10 points entre 1e-3 et RCS_sep
            # Trouver la solution avec fsolve
            d_solutions = find_d_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "brentq":
            
            # Trouver la solution avec brentq
            d_solutions = find_d_solution_brentq(0, RCS_ext)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if debuging_CS == 1:
                print(f'D0FUS CS solutions : {valid_solutions}')
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            
            d_solutions = find_d_solution_root(0, RCS_ext)
            # print(d_SS_solution)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "manual":

            d_solutions = plot_function_CS(CS_to_solve, [0, RCS_ext])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        else:
            print("Choisissez une méthode valide pour le CS")
            return (np.nan, np.nan, np.nan, np.nan)
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results ####
        tol=1e-6
        # Preliminary calculations
        RCS_int = RCS_ext - d_solution
        # Dilution coefficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        if __name__ == "__main__" and debuging_CS == 1:
            print(f'D0FUS CS width : {d_solution}')
            print(f'D0FUS CS alpha : {alpha}')
            print(f'D0FUS CS field : {B_CS}')
    
        # Apply robust constraints with tolerances
        if d_solution < tol or d_solution > RCS_ext - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        if B_CS < tol or B_CS > Bmax - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        if alpha <  tol or alpha > 1 - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        
        return (d_solution, alpha, B_CS, J_max_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan, np.nan)

#%% CS D0FUS & CIRCE

def f_CS_CIRCE(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0 , Bmax, σ_CS, J_max_CS, Choice_Buck_Wedg):
    
    """
    Calculate the CS thickness considering a thick layer approximation and a 2 cylinder (supra + steel) approach

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    c : TF thickness (m)
    R0 : Major radius (m)
    B0 : Central magnetic field (m)
    σ_CS : Yield strength of the CS steel (Pa)
    μ0 : Magnetic permeability of free space
    J_max_CS : Maximum current density of the chosen Supra + Cu + He (A/m²)
    Choice_Buck_Wedg : Mechanical configuration

    Returns:
    tuple: A tuple containing the calculated values :
    (d, Alpha, B_CS)
    With
    d : CS width
    Alpha : % of conductor
    B_CS : CS magnetic field
    
    Always pay attention : the calculations are not exact, so for filtering, 
    and manipulation of the solutions, need to take a margin ~ 1e-4
    (has been the cause of several bugs especially in the research of the best solution)
    
    """
    
    debuging_CS = 0
    
    # External radius of the CS
    if Choice_Buck_Wedg == 'Bucking':
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        print("Choose between Wedging and Bucking")
        return (np.nan, np.nan, np.nan)
    
    # Fonction cible
    def CS_to_solve(d):
        
        """
        Calcule la contrainte mécanique et retourne la différence avec la contrainte cible σ_CS.
        """
        
        # Preliminary calculations
        RCS_int = RCS_ext - d
        # Dilution coeficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        # magnetic pressure to support from the CS
        P_CS = B_CS**2 / (2 * μ0)
        # magnetic pressure to support from the TF
        P_TF = Bmax**2 / (2 * μ0) * (R0 - a - b) / (RCS_ext)
    
        if Choice_Buck_Wedg == 'Bucking':
            
            # Light bucking
            disR = 20                               # Pas de discrétisation
            R = np.array([(RCS_ext-d),RCS_ext])     # Radii
            J = np.array([J_max_CS*alpha])          # Current densities
            B = np.array([B_CS])                    # Magnetic fields
            Pi = 0                                  # Internal pressure
            Pe = P_TF                               # External pressure
            E = np.array([Young_modul_Steel*(1-alpha)])         # Young's modul
            nu = nu_Steel*(1-alpha)                                       # Poisson's ratio
            config = np.array([0])                  # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS_light = max(np.abs(SigTtot))
            
            # Strong bucking
            disR = 20                               # Pas de discrétisation
            R = np.array([(RCS_ext-d),RCS_ext])     # Radii
            J = np.array([0])                       # Current densities
            B = np.array([0])                       # Magnetic fields
            Pi = 0                                  # Internal pressure
            Pe = P_TF                               # External pressure
            E = np.array([Young_modul_Steel*(1-alpha)])         # Young's modul
            nu = nu_Steel*(1-alpha)                               # Poisson's ratio
            config = np.array([0])                  # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS_strong = max(np.abs(SigTtot))
            
            # Final sigma
            Sigma_CS = np.nanmax([abs(Sigma_CS_light), abs(Sigma_CS_strong)])
            
            val = Sigma_CS - σ_CS
            
        elif Choice_Buck_Wedg == 'Wedging':
            
            disR = 20                               # Pas de discrétisation
            R = np.array([(RCS_ext-d),RCS_ext])     # Radii
            J = np.array([J_max_CS*alpha])          # Current densities
            B = np.array([B_CS])                    # Magnetic fields
            Pi = 0                                  # Internal pressure
            Pe = 0                                  # External pressure
            E = np.array([Young_modul_Steel*(1-alpha)])         # Young's modul
            nu = nu_Steel*(1-alpha)                               # Poisson's ratio
            config = np.array([0])                  # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS = np.nanmax([np.abs(SigTtot)])
            
            val = Sigma_CS - σ_CS
            
        elif Choice_Buck_Wedg == 'Plug':
            
            # Light bucking
            disR = 20                               # Pas de discrétisation
            R = np.array([(RCS_ext-d),RCS_ext])     # Radii
            J = np.array([J_max_CS*alpha])          # Current densities
            B = np.array([B_CS])                    # Magnetic fields
            Pi = P_TF                                  # Internal pressure
            Pe = P_TF                               # External pressure
            E = np.array([Young_modul_Steel*(1-alpha)])         # Young's modul
            nu = nu_Steel*(1-alpha)                                       # Poisson's ratio
            config = np.array([0])                  # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS_light = max(np.abs(SigTtot))
            
            # Strong bucking
            disR = 20                               # Pas de discrétisation
            R = np.array([(RCS_ext-d),RCS_ext])     # Radii
            J = np.array([0])                       # Current densities
            B = np.array([0])                       # Magnetic fields
            Pi = P_TF                                  # Internal pressure
            Pe = P_TF                               # External pressure
            E = np.array([Young_modul_Steel*(1-alpha)])         # Young's modul
            nu = nu_Steel*(1-alpha)                               # Poisson's ratio
            config = np.array([0])                  # TF = 1 , CS = 0
            # Appeler la fonction principale
            SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
            Sigma_CS_strong = max(np.abs(SigTtot))
            
            # Final sigma
            Sigma_CS = np.nanmax([abs(Sigma_CS_light), abs(Sigma_CS_strong)])
            
            val = Sigma_CS - σ_CS
            
        else:
            raise ValueError("Choose between 'Wedging' and 'Bucking'")
        
        
        # passage en log afin de lisser les sauts abruptes qui peuvent arriver et faciliter la recherche de racines
        return np.sign(val) * np.log1p(abs(val))
    
    def plot_function_CS(CS_to_solve, x_range):
        """
        Visualise la fonction sur une plage donnée pour comprendre son comportement
        """
        
        x = np.linspace(x_range[0], x_range[1], 10000)
        y = [CS_to_solve(xi) for xi in x]
        
        plot_option_function = 0
        
        if plot_option_function ==1:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', label='CS_to_solve(d)')
            plt.axhline(y=0, color='r', linestyle='--', label='y=0')
            plt.grid(True)
            plt.xlabel('d')
            plt.ylabel('Function Value')
            plt.ylim(-1e10,1e10)
            plt.title('Comportement de CS_to_solve')
            plt.legend()
        
        # Identifier les points où la fonction change de signe
        zero_crossings = []
        for i in range(len(y)-1):
            if y[i] * y[i+1] <= 0:
                zero_crossings.append((x[i]))
        return zero_crossings
    
    # Détection des changements de signe
    def find_sign_changes(f, a, b, n):
        """ 
        Détecte les intervalles où la fonction change de signe.
        """
        x_vals = np.linspace(a, b, n)
        y_vals = np.array([f(x) for x in x_vals])
        
        sign_changes = []
        for i in range(1, len(x_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Changement de signe détecté
                sign_changes.append((x_vals[i-1], x_vals[i]))
    
        return sign_changes
    
    # Raffinement des passages par zéro avec brentq
    def refine_zeros(f, sign_change_intervals):
        """ 
        Utilise la méthode de Brent pour affiner les passages par zéro.
        """
        roots = []
        for a, b in sign_change_intervals:
            try:
                root = brentq(f, a, b)  # Brent garantit une convergence si un changement de signe est détecté
                roots.append(root)
            except ValueError:
                pass  # Si Brent échoue, on ignore l'intervalle
    
        return roots if roots else [np.nan]  # Si aucune racine n'est trouvée, retourne np.nan
    
    # Recherche des solutions en utilisant la méthode de Brent
    def find_d_solution_brentq(a, b, n = 1000):
        """
        Trouve les passages par zéro de CS_to_solve entre a et b.
        """
        sign_change_intervals = find_sign_changes(CS_to_solve, a, b, n)
        roots = refine_zeros(CS_to_solve, sign_change_intervals)
    
        return roots
    
    # Vérification que la solution est bien un passage par zéro
    def is_valid_root(f, root, epsilon=1e-4, tol=1e-4):
        """
        Vérifie si root est un vrai passage par zéro et non un minimum local.
        - Vérifie que |f(root)| est proche de 0 (précision tol)
        - Vérifie un changement de signe autour de root (epsilon)
        """
        if np.abs(f(root)) > tol:
            return False  # La fonction n'est pas proche de 0 -> échec
    
        f_left = f(root - epsilon)
        f_right = f(root + epsilon)
    
        if np.sign(f_left) != np.sign(f_right):  # Changement de signe autour de root
            return True
        else:
            return False  # Probablement un minimum local
    
    # Trouver la plus petite solution valide avec fsolve
    def find_d_solution_fsolve(initial_guesses):
        """
        Cherche une solution avec fsolve en s'assurant qu'il s'agit bien d'un passage par zéro.
        - initial_guesses : Liste de points de départ pour fsolve
        - Retourne la plus petite solution valide ou np.nan si aucune racine correcte n'est trouvée
        """
        valid_solutions = []
    
        for guess in initial_guesses:
            root_candidate, info, ier, msg = fsolve(CS_to_solve, guess, full_output=True)
    
            if ier == 1:  # Vérifier si la convergence a réussi
                root = root_candidate[0]
                if is_valid_root(CS_to_solve, root):  # Vérifier que c'est bien un passage par zéro
                    valid_solutions.append(root)
    
        return min(valid_solutions) if valid_solutions else np.nan  # Prendre la plus petite solution
    
    def find_d_solution_root(a, b):
        """
        Trouve les passages par zéro de CS_to_solve en utilisant scipy.optimize.root.
        """
        initial_guesses = np.linspace(a, b, 10)
        valid_solutions = []
    
        for guess in initial_guesses:
            sol = root(CS_to_solve, guess, method='hybr')
            if sol.success and is_valid_root(CS_to_solve, sol.x[0]):
                valid_solutions.append(sol.x[0])
        return min(valid_solutions) if valid_solutions else np.nan

    def CS_filters(d_solution):
        
        tol=1e-6
        
        # Preliminary calculations
        RCS_int = RCS_ext - d_solution
        # Dilution coefficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        # Apply robust constraints with tolerances
        if d_solution < tol or d_solution > RCS_ext - tol:
            return False
        if B_CS < tol or B_CS > Bmax - tol:
            return False
        if alpha <  tol or alpha > 1 - tol:
            return False
        
        return True
    
    try:
        
        """
        ### `fsolve`  
        Basé sur la méthode de Newton-Raphson (et ses variantes quasi-Newtoniennes)
        Utilise la dérivée locale pour ajuster l'approximation de la racine  
        Problème : Peut converger vers un **minimum local** au lieu d'un vrai passage par zéro si la dérivée est faible  
        
        ### `brentq`  
        Basé sur la méthode de Brent (hybride entre la bissection et la sécante)
        Nécessite un intervalle `[a, b]` avec un changement de signe
        Avantage : Garantie de trouver un passage exact par zéro si un changement de signe est détecté
        
        ### `root`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Permet d'utiliser plusieurs algorithmes (`hybr`, `lm`, `broyden1`, `broyden2`, etc.)  
        Par défaut (`method='hybr'`), il repose sur une approche dérivée de **l’algorithme de Powell** (similaire à Newton-Raphson)  
        Problème : Comme `fsolve`, il peut converger vers un minimum local au lieu d’un vrai passage par zéro si la dérivée est faible
        
        ### `manual`  
        Méthode générale pour trouver les racines d’une fonction non linéaire
        Méthode maison parcourant toute les solutions et cherchant les changement de signe
        Problème : peu précise et couteuse en temps de calcul
        Avantage : Impossible de faire plus robuste et simple
        """
        
        if Choice_solving_CS_method == "fsolve":
            
            # Liste d'estimations initiales pour couvrir plusieurs racines possibles
            initial_guesses = np.linspace(0, RCS_ext, 10)  # 10 points entre 1e-3 et RCS_sep
            # Trouver la solution avec fsolve
            d_solutions = find_d_solution_fsolve(initial_guesses)
            # print(d_SS_solution_fsolve)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "brentq":
            
            # Trouver la solution avec brentq
            d_solutions = find_d_solution_brentq(0, RCS_ext)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if debuging_CS == 1:
                print(f'D0FUS CS solutions : {valid_solutions}')
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "root":
            
            d_solutions = find_d_solution_root(0, RCS_ext)
            # print(d_SS_solution)
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        elif Choice_solving_CS_method == "manual":

            d_solutions = plot_function_CS(CS_to_solve, [0, RCS_ext])
            # Filtrer les solutions valides (exclure np.nan) et sélectionner la plus petite
            valid_solutions = [sol for sol in d_solutions if not np.isnan(sol) and CS_filters(sol)]
            
            if valid_solutions:
                d_solution = min(valid_solutions)  # Prendre la plus petite solution
                # print(d_SS_solution)
            else :
                return(np.nan, np.nan, np.nan, np.nan)
            
        else:
            print("Choisissez une méthode valide pour le CS")
            return (np.nan, np.nan, np.nan, np.nan)
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
        #### Results ####
        tol=1e-6
        # Preliminary calculations
        RCS_int = RCS_ext - d_solution
        # Dilution coefficient
        alpha = 3 * (ΨPI + ΨRampUp + Ψplateau - ΨPF) / (2 * np.pi * μ0 * J_max_CS * (RCS_ext - RCS_int) * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        # CS magnetic field
        B_CS = μ0 * J_max_CS * alpha * (RCS_ext - RCS_int)
        
        if __name__ == "__main__" and debuging_CS == 1:
            print(f'D0FUS CS width : {d_solution}')
            print(f'D0FUS CS alpha : {alpha}')
            print(f'D0FUS CS field : {B_CS}')
    
        # Apply robust constraints with tolerances
        if d_solution < tol or d_solution > RCS_ext - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        if B_CS < tol or B_CS > Bmax - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        if alpha <  tol or alpha > 1 - tol:
            return (np.nan, np.nan, np.nan, np.nan)
        
        return (d_solution, alpha, B_CS, J_max_CS)

    except Exception as e:
        return (np.nan, np.nan, np.nan, np.nan)
    
#%% CS Benchmark

if __name__ == "__main__":

    # === Machines definition with their CS parameters and target Ψplateau ===
    machines = {
        "ITER":    {"Ψplateau": 230, "a_cs": 2.00, "b_cs": 1.25, "c_cs": 0.90, "R0_cs": 6.20, "B_cs": 13, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2 - 1},
        "EU-DEMO": {"Ψplateau": 600, "a_cs": 2.92, "b_cs": 1.80, "c_cs": 1.19, "R0_cs": 9.07, "B_cs": 13.5, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2 - 1.5},
        "JT60-SA": {"Ψplateau":  40, "a_cs": 1.18, "b_cs": 0.27, "c_cs": 0.45, "R0_cs": 2.96, "B_cs": 8.9, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2 + 1},
        # Source : Operating Parameters in 
        # Koide, Y., Yoshida, K., Wanner, M., Barabaschi, P., Cucchiaro, A., Davis, S., ... & Zani, L. (2015). JT-60SA superconducting magnet system. Nuclear Fusion, 55(8), 086001.
        "EAST":    {"Ψplateau":  10, "a_cs": 0.45, "b_cs": 0.4, "c_cs": 0.25, "R0_cs": 1.85, "B_cs": 4.7, "σ_CS": 330e6, "config": "Wedging", "SupraChoice": "NbTi", "T_CS": 4.20 + 0.82},
        "ARC":     {"Ψplateau":  32, "a_cs": 1.07, "b_cs": 0.89, "c_cs": 0.64, "R0_cs": 3.30, "B_cs": 12.9, "σ_CS": 500e6, "config": "Bucking", "SupraChoice": "Rebco", "T_CS": 20},
        "SPARC":   {"Ψplateau":  30, "a_cs": 0.57, "b_cs": 0.18, "c_cs": 0.35, "R0_cs": 1.85, "B_cs": 25, "σ_CS": 500e6, "config": "Bucking", "SupraChoice": "Rebco", "T_CS": 20},
    }

    # === Accumulate rows for DataFrame ===
    rows = []
    for name, p in machines.items():
        # Unpack inputs
        Ψplateau = p["Ψplateau"]
        a, b, c, R0 = p["a_cs"], p["b_cs"], p["c_cs"], p["R0_cs"]
        Bcs, σ = p["B_cs"], p["σ_CS"]
        Supra_Choice, T_CS = p["SupraChoice"], p["T_CS"]
        Jmax = Jc(Supra_Choice, Bcs, T_CS) * f_Cu * f_Cool * f_In
        print(f"{name} : J = {np.round(Jmax/1e6,2)} MA/m² for Bcs = {Bcs} , T_CS = {T_CS} and {Supra_Choice}")
        Bmax = 50

        # Call the models
        acad_w = f_CS_academic(0, 0, Ψplateau, 0, a, b, c, R0, Bmax, σ, Jmax, "Wedging")
        acad_b = f_CS_academic(0, 0, Ψplateau, 0, a, b, c, R0, Bcs, σ, Jmax, "Bucking")
        d0fus_w = f_CS_D0FUS(0, 0, Ψplateau, 0, a, b, c, R0, Bmax, σ, Jmax, "Wedging")
        d0fus_b = f_CS_D0FUS(0, 0, Ψplateau, 0, a, b, c, R0, Bcs, σ, Jmax, "Bucking")
        
        def clean_result(val):
            # Élimine les valeurs complexes ou NaN
            if np.iscomplex(val) or val is None or np.isnan(val):
                return np.nan
            return round(val, 2)

        # Build one row combining inputs + outputs
        rows.append({
            "Machine":        name,
            "Ψ [Wb]":  Ψplateau,
            "σ_CS [MPa]":     σ / 1e6,
            "J [MA/m²]": clean_result(Jmax/ 1e6) ,
            "T [K]": clean_result(T_CS),
            "Width W [m]":  clean_result(d0fus_w[0]),
            "B_CS W [T]" : clean_result(d0fus_w[2]),
            "Width B [m]":  clean_result(d0fus_b[0]),
            "B_CS B [T]" : clean_result(d0fus_b[2]),
        })

    # === Print table ===
    df = pd.DataFrame(rows)
    
    # === Création d'une figure avec tableau ===
    fig, ax = plt.subplots(figsize=(9, 2))   # largeur/hauteur ajustables
    ax.axis("off")
    
    # Création du tableau
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    
    # Mise en forme
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)  # élargir pour lisibilité
    
    plt.title("CS Benchmark Summary", fontsize=14, pad=20)
    plt.show()
    
#%% CS plot

if __name__ == "__main__":
    
    # === Paramètres d'entrée ===
    a_cs = 2
    b_cs = 1.2
    c_cs = 0.9
    R0_cs = 6
    Bmax_cs = 13
    σ_CS_cs = 660e6
    J_max_CS_cs = 50e6
    
    # === Scan de Ψplateau ===
    psi_values = np.linspace(0, 400, 50)
    
    # === Listes de résultats ===
    acad_w, acad_b = [], []
    d0fus_w, d0fus_b = [], []
    CIRCE_w, CIRCE_b = [], []
    
    acad_B_w, acad_B_b = [], []
    d0fus_B_w, d0fus_B_b = [], []
    CIRCE_B_w, CIRCE_B_b = [], []
    
    # === Boucle sur Ψplateau ===
    for psi in psi_values:
        # Academic
        res_acad_w = f_CS_academic(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, J_max_CS_cs, "Wedging")
        res_acad_b = f_CS_academic(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, J_max_CS_cs, "Bucking")

        # D0FUS
        res_d0fus_w = f_CS_D0FUS(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, J_max_CS_cs, "Wedging")
        res_d0fus_b = f_CS_D0FUS(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, J_max_CS_cs, "Bucking")

        # CIRCE
        res_CIRCE_w = f_CS_CIRCE(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, J_max_CS_cs, "Wedging")
        res_CIRCE_b = f_CS_CIRCE(0, 0, psi, 0, a_cs, b_cs, c_cs, R0_cs, Bmax_cs, σ_CS_cs, J_max_CS_cs, "Bucking")
        
        # Stockage épaisseur
        acad_w.append(res_acad_w[0])
        acad_b.append(res_acad_b[0])
        d0fus_w.append(res_d0fus_w[0])
        d0fus_b.append(res_d0fus_b[0])
        CIRCE_w.append(res_CIRCE_w[0])
        CIRCE_b.append(res_CIRCE_b[0])
        
        # Stockage champ magnétique
        acad_B_w.append(res_acad_w[2])
        acad_B_b.append(res_acad_b[2])
        d0fus_B_w.append(res_d0fus_w[2])
        d0fus_B_b.append(res_d0fus_b[2])
        CIRCE_B_w.append(res_CIRCE_w[2])
        CIRCE_B_b.append(res_CIRCE_b[2])
    
    # === Couleurs par modèle ===
    colors = {
        "Academic": "blue",
        "D0FUS": "green",
        "CIRCE": "red"
    }
    
    # === FIGURE 1 : Wedging – Épaisseur ===
    plt.figure(figsize=(5, 5))
    plt.plot(psi_values, acad_w, color=colors["Academic"], linestyle='-', linewidth=2, label="Academic Wedging")
    plt.plot(psi_values, d0fus_w, color=colors["D0FUS"], linestyle='-', linewidth=2, label="D0FUS Wedging")
    plt.plot(psi_values, CIRCE_w, color=colors["CIRCE"], linestyle='-', linewidth=2, label="CIRCE Wedging")
    
    plt.xlabel("Ψplateau (Wb)", fontsize=12)
    plt.ylabel("CS thickness [m]", fontsize=12)
    plt.title("Ψplateau scan – Wedging comparison", fontsize=14)
    plt.ylim(bottom=0)   # <-- Force le départ à 0
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()
    
    # === FIGURE 2 : Bucking – Épaisseur ===
    plt.figure(figsize=(5, 5))
    plt.plot(psi_values, acad_b, color=colors["Academic"], linestyle='-', linewidth=2, label="Academic Bucking")
    plt.plot(psi_values, d0fus_b, color=colors["D0FUS"], linestyle='-', linewidth=2, label="D0FUS Bucking")
    plt.plot(psi_values, CIRCE_b, color=colors["CIRCE"], linestyle='-', linewidth=2, label="CIRCE Bucking")
    
    plt.xlabel("Ψplateau (Wb)", fontsize=12)
    plt.ylabel("CS thickness [m]", fontsize=12)
    plt.title("Ψplateau scan – Bucking comparison", fontsize=14)
    plt.ylim(bottom=0)   # <-- Force le départ à 0
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()
    
    # === FIGURE 3 : Wedging – Champ magnétique ===
    plt.figure(figsize=(5, 5))
    plt.plot(psi_values, acad_B_w, color=colors["Academic"], linestyle='-', linewidth=2, label="Academic Wedging")
    plt.plot(psi_values, d0fus_B_w, color=colors["D0FUS"], linestyle='-', linewidth=2, label="D0FUS Wedging")
    plt.plot(psi_values, CIRCE_B_w, color=colors["CIRCE"], linestyle='-', linewidth=2, label="CIRCE Wedging")
    
    plt.xlabel("Ψplateau (Wb)", fontsize=12)
    plt.ylabel("B CS [T]", fontsize=12)
    plt.title("Ψplateau scan – Wedging comparison (Magnetic field)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()
    
    # === FIGURE 4 : Bucking – Champ magnétique ===
    plt.figure(figsize=(5, 5))
    plt.plot(psi_values, acad_B_b, color=colors["Academic"], linestyle='-', linewidth=2, label="Academic Bucking")
    plt.plot(psi_values, d0fus_B_b, color=colors["D0FUS"], linestyle='-', linewidth=2, label="D0FUS Bucking")
    plt.plot(psi_values, CIRCE_B_b, color=colors["CIRCE"], linestyle='-', linewidth=2, label="CIRCE Bucking")
    
    plt.xlabel("Ψplateau (Wb)", fontsize=12)
    plt.ylabel("B CS [T]", fontsize=12)
    plt.title("Ψplateau scan – Bucking comparison (Magnetic field)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()
        
#%%  Solenoid comparison CIRCE vs D0FUS

if __name__ == "__main__":
    
    print('-------------------------------------------------------------------------------------------')
    
    # D0FUS model :
    R_ext_test = 2.1
    Flux_théorique = 132   # Rapport d'aspect fin
    Flux_théorique = 263.5   # rapport d'aspect épais
    Sigma_theorique = 300e6 # Steel
    J_CS_théorique = 50e6 # Conductor
    B_CS_theorique = 20 # Initialisation
    calcul_D0FUS = f_CS_D0FUS(0, Flux_théorique, 0, 0, 0, 0, 0, R_ext_test , B_CS_theorique, Sigma_theorique, J_CS_théorique, "Wedging")
    
    alpha_test = np.round(calcul_D0FUS[1],2)
    
    print(f'Epaisseur D0FUS prediction: {np.round(calcul_D0FUS[0],2)} m')
    print(f'Alpha D0FUS prediction: {np.round(calcul_D0FUS[1]*100,2)} %')
    print(f'B D0FUS prediction: {np.round(calcul_D0FUS[2],2)} T')
    print(f'J D0FUS prediction: {np.round(calcul_D0FUS[3]/1e6,2)} MA')
    print(f'Sigma D0FUS prediction: {np.round(Sigma_theorique*(1-calcul_D0FUS[1])/1e6,2)} MPA')
    
    print('-------------------------------------------------------------------------------------------')

    # Parameters :
    R_ext_test = 2.1 - 0.1
    R_int_test = R_ext_test - calcul_D0FUS[0]
    B_CS_test = calcul_D0FUS[2]
    J_CS_test = J_CS_théorique * alpha_test
    P_mag_test = B_CS_test**2 / (2*μ0)
    
    # D0FUS calculus :
    calcul_type = np.round(P_mag_test * (R_ext_test**2+R_int_test**2)/(R_ext_test**2-R_int_test**2)/1e6,2)
    print(f'Sigma_Steel manual surface crosscheck: {calcul_type} MPa')
    print('-------------------------------------------------------------------------------------------')
    
    # CIRCE modèle surfacique :
    disR = 100                      # Pas de discrétisation
    R = np.array([R_int_test,R_ext_test])       # Radii
    J = np.array([0])            # Current densities
    B = np.array([0])               # Magnetic fields
    Pi = P_mag_test             # Internal pressure
    Pe = 0                          # External pressure
    E = np.array([Young_modul_Steel])           # Young's moduli
    nu = nu_Steel                       # Poisson's ratio
    config = np.array([0])          # Wedging or bucking
    
    # Appeler la fonction principale
    SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
    Sig_Steel = np.round(max(np.abs(SigTtot/1e6)),2)
    print(f'Sigma_Steel CIRCE surface prediction: {Sig_Steel} MPa')
    print(f'Erreur D0FUS vs CIRCE: {np.round((calcul_type-Sig_Steel) / calcul_type * 100, 2)} %')
    print('-------------------------------------------------------------------------------------------')
    
    # CIRCE modèle volumique :
    disR = 100                      # Pas de discrétisation
    R = np.array([R_int_test,R_ext_test])       # Radii
    J = np.array([J_CS_test])            # Current densities
    B = np.array([B_CS_test])              # Magnetic fields
    Pi = 0                          # Internal pressure
    Pe = 0                          # External pressure
    E = np.array([Young_modul_Steel*(1-alpha_test)])           # Young's modul
    nu = nu_Steel                      # Poisson's ratio
    config = np.array([0])          # TF = 1 , CS = 0
    # Appeler la fonction principale
    SigRtot, SigTtot, urtot, Rvec, P = F_CIRCE0D(disR, R, J, B, Pi, Pe, E, nu, config)
    Sig_Steel = np.round(max(np.abs(SigTtot/1e6)),2)
    print(f'Sigma_Steel CIRCE volume prediction: {Sig_Steel} MPa')
    print(f'Erreur D0FUS vs CIRCE: {np.round((calcul_type-Sig_Steel) / max(calcul_type,Sig_Steel) * 100, 2)} %')

#%% Note:
# CIRCE TF double cylindre en wedging ? voir multi cylindre pour grading ?
# Nécessite la résolution de R_int et R_sep en même temps
# Permettrait aussi de mettre la répartition en tension en rapport de surface
#%% Print

print("D0FUS_radial_build_functions loaded")