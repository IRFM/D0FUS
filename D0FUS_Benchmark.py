# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:40:40 2024

@author: TA276941
"""

#%% CFETR

# Plasma
H = 1.42 # H factor
Tbar  = 14 # keV Mean Temperature

# Geometric parameters
κ = 2  # elongation
δ = 0.33 # Triangularity
b = 1.5 # BB+1rst Wall+N shields+ Gaps

# Mechanical
σ_TF = 730.E6   # Mechanical limit of the steel considered in [Pa]
σ_CS = 500.E6   # CS machanical limit [Pa] 
Choice_Buck_Wedg = 'Wedging' # Wedging or Bucking

# Core
R0 = 7.2
a = 2.2
Pfus = 2192
Bmax = 14

#%% K-DEMO

# Plasma
H = 1.17 # H factor
Tbar  = 17 # keV Mean Temperature

# Geometric parameters
κ = 2  # elongation
δ = 0.625 # Triangularity
b = 1.7 # BB+1rst Wall+N shields+ Gaps

# Mechanical
σ_TF = 860.E6   # Mechanical limit of the steel considered in [Pa]
σ_CS = 660.E6   # CS machanical limit [Pa] 
Choice_Buck_Wedg = 'Wedging' # Wedging or Bucking

# Core
R0 = 6.8
a = 2.1
Pfus = 3000
Bmax = 16

#%% J-DEMO

# Plasma
H = 1.31 # H factor
Tbar  = 16 # keV Mean Temperature

# Geometric parameters
κ = 1.85  # elongation
δ = 0.33 # Triangularity
b = 2 # BB+1rst Wall+N shields+ Gaps

# Mechanical
σ_TF = 800.E6   # Mechanical limit of the steel considered in [Pa]
σ_CS = 500.E6   # CS machanical limit [Pa] 
Choice_Buck_Wedg = 'Wedging' # Wedging or Bucking

# Core
R0 = 8.5
a = 2.42
Pfus = 1420
Bmax = 13.7

#%% ARC

# Plasma
H = 1.8 # H factor
Tbar  = 14 # keV Mean Temperature

# Geometric parameters
κ = 1.84  # elongation
δ = 0.33 # Triangularity
b = 0.86 # BB+1rst Wall+N shields+ Gaps

# Mechanical
σ_TF = 660.E6   # Mechanical limit of the steel considered in [Pa]
σ_CS = 660.E6   # CS machanical limit [Pa] 
Choice_Buck_Wedg = 'Bucking' # Wedging or Bucking

# Core
R0 = 3.3
a = 1.13
Pfus = 525
Bmax = 23


