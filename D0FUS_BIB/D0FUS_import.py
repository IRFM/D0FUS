# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:30:05 2024

@author: TA276941
"""

#%% Import

# Importations de bibliothèques standards
import os
import math
import time
import warnings
import sys
import importlib

# Importations de bibliothèques scientifiques et de calcul numérique
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import (
    fsolve,
    differential_evolution,
    root_scalar,
    minimize_scalar,
    minimize,
    bisect,
    root,
    basinhopping,
    brentq,
    bisect,
    root
)
from scipy.interpolate import griddata
from scipy.integrate import quad
from deap import base, creator, tools, algorithms
from pyswarm import pso  # Nécessite l'installation de pyswarm

# Importations de bibliothèques de visualisation
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns  # améliore l'esthétique des graphiques
from pandas.plotting import table
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importation pour le traitement des images
import imageio

#%%

print("D0FUS_import loaded")

