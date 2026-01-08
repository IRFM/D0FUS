from __future__ import annotations

"""
D0FUS Import Module
===================
Central import module for the D0FUS (Design 0-dimensional for FUsion Systems) project.

Created: December 2023
Author: Auclair Timothé
"""

#%% Environment Configuration

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#%% Standard Library Imports

import json
import math
import random
import re
import shutil
import sys
import time
import warnings
import importlib
from datetime import datetime
from pathlib import Path

#%% Scientific Computing Libraries

import numpy as np
import pandas as pd
import sympy as sp

#%% Scipy - Optimization and Numerical Methods

from scipy.integrate import quad
from scipy.interpolate import interp1d, griddata
from scipy.optimize import (
    basinhopping,
    bisect,
    brentq,
    differential_evolution,
    fsolve,
    least_squares,
    minimize,
    minimize_scalar,
    root,
    root_scalar,
    shgo
)
from scipy.signal import find_peaks

#%% Visualization Libraries

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import table
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

#%% Genetic Algorithm Libraries

from deap import algorithms, base, creator, tools

#%%

# print("D0FUS_import loaded")