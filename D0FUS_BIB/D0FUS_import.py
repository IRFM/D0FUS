from __future__ import annotations

"""
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Import

# Importations de bibliothèques standards
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
import shutil
import math
import time
import warnings
import sys
import importlib
import re
from pathlib import Path

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
    least_squares,
    shgo
)
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.integrate import quad
from scipy.signal import find_peaks

# Importations de bibliothèques de visualisation
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from pandas.plotting import table
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Gentic research
import json
import random
from matplotlib.gridspec import GridSpec
from deap import base, creator, tools, algorithms

#%%

# print("D0FUS_import loaded")

