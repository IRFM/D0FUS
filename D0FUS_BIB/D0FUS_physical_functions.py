"""
Physical functions definition for the D0FUS - Design 0-dimensional for FUsion Systems project
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Imports

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_parameterization import *

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_parameterization import *

#%% Geometry formulas

"""
Tokamak plasma cross-section geometry for D0FUS
=====================================================================

Definitions
-----------
The plasma poloidal cross-section is parameterised by three global shape
descriptors, all defined at the last closed flux surface (LCFS, ρ = 1):

    κ  (kappa)  Elongation: ratio of plasma half-height b to minor radius a.
                κ = b/a.  A circular plasma has κ = 1; modern tokamaks use
                κ ~ 1.6–1.9 to improve confinement and MHD stability.

    δ  (delta)  Triangularity: normalised horizontal displacement of the plasma
                extrema.  δ = (R₀ − R_top) / a where R_top is the major radius
                at the upper plasma tip.  δ > 0 gives a D-shaped cross-section.
                ITER operates at δ ≈ 0.5.

    ρ           Normalised radial label: ρ = r/a ∈ [0, 1].

Miller parameterisation of flux surfaces (Miller et al. 1998):
    R(ρ,θ) = R₀ + ρa cos(θ + arcsin(δ(ρ)) sinθ)
    Z(ρ,θ) = κ(ρ) ρa sinθ

Geometry models
---------------
'Academic'
    κ(ρ) = κ_edge = const,  δ(ρ) = 0.
    Cylindrical-torus approximation.  Consistent with PROCESS (Kovari 2014).
    Valid to ~10 % because elongation penetrates efficiently to the core
    (κ_core ≈ κ₉₅ ≈ κ_edge/1.12) and δ contributes only a 2nd-order
    correction to the volume.

'D0FUS'
    κ(ρ): flat-core PCHIP profile, κ ≈ κ₉₅ for ρ ≤ ρ₉₅, rising to κ_edge
          in the thin edge layer [ρ₉₅, 1].  Physically motivated by Ball &
          Parra (2015): for nearly flat current profiles dκ/dρ ≈ 0 in the bulk.
    δ(ρ): PCHIP profile through (0, 0), (ρ₉₅, δ₉₅), (1, δ_edge).
          δ(0) = 0 is an exact Grad-Shafranov constraint (m=3 harmonic
          vanishes on axis); Ball & Parra confirm poor radial penetration.
          PCHIP guarantees local monotonicity and C¹ continuity while
          passing exactly through the δ₉₅ constraint, unlike a power-law
          whose fixed analytical exponent (s_δ ≈ 8–9 for ITER 1989 scaling)
          cannot accommodate arbitrary δ₉₅ prescriptions from free-boundary
          codes such as CHEASE or EFIT.

ITER reference values used in all __main__ tests
-------------------------------------------------
    R₀ = 6.2 m,  a = 2.0 m,  κ_edge = 1.85,  δ_edge = 0.50

Physical justification
----------------------
The Grad-Shafranov Taylor expansion around the magnetic axis gives:
    ψ(R,Z) ≈ A(R−R₀)² + BZ²
The axis flux surfaces are ellipses with κ(0) = sqrt(B/A), determined
self-consistently by the equilibrium — NOT constrained to 1.  Only
higher-order harmonics (m ≥ 3) vanish on axis: δ(0) = 0 is exact.
Ball & Parra (2015) show that elongation (m=2) penetrates efficiently
(near-constant across ρ for flat current), while triangularity (m=3)
is confined to the plasma edge.  These two results together justify:
    — the flat-core PCHIP for κ,
    — the strongly edge-peaked PCHIP for δ,
    — and the Academic model as a sound first-order approximation.

References
----------
Miller et al., Phys. Plasmas 5, 973 (1998).
Ball & Parra, PPCF 57, 035006 (2015).
Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980).
Lao et al., Fusion Sci. Technol. 48, 968 (2005).
Kovari et al., Fusion Eng. Des. 89, 3054 (2014) — PROCESS.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import PchipInterpolator
from scipy.special import erfc
from scipy.optimize import root_scalar


# =============================================================================
# Edge shaping scalings
# =============================================================================

def f_Kappa(A, Option_Kappa, κ_manual, ms):
    """
    Maximum achievable plasma elongation vs aspect ratio.

    Parameters
    ----------
    A : float or ndarray
        Plasma aspect ratio R₀/a [-].
    Option_Kappa : str
        'Stambaugh' | 'Freidberg' | 'Wenninger' | 'Manual'
    κ_manual : float
        Used only when Option_Kappa == 'Manual'.
    ms : float
        Normalised wall-gap parameter (Wenninger scaling only).

    Returns
    -------
    κ : float or ndarray
        Returns np.nan where κ ≤ 0 (non-physical).

    References
    ----------
    Stambaugh et al., Nucl. Fusion 32, 1642 (1992).
    Freidberg et al., J. Plasma Phys. 81, 515810607 (2015).
    Lee et al.,       J. Plasma Phys. 81, 515810608 (2015).
    Wenninger et al., Nucl. Fusion 55, 063003 (2015).
    Coleman et al.,   Nucl. Fusion 65, 036039 (2025).
    """
    if Option_Kappa == 'Stambaugh':
        κ = 0.95 * (2.4 + 65 * np.exp(-A / 0.376))
    elif Option_Kappa == 'Freidberg':
        κ = 0.95 * (1.81153991 * A**0.009042 + 1.5205 * A**(-1.63))
    elif Option_Kappa == 'Wenninger':
        κ = 1.12 * ((18.84 - 0.87*A
                     - np.sqrt(4.84*A**2 - 28.77*A + 52.52 + 14.74*ms)) / 7.37)
    elif Option_Kappa == 'Manual':
        κ = κ_manual
    else:
        raise ValueError(f"Unknown Option_Kappa: '{Option_Kappa}'. "
                         "Valid: 'Stambaugh', 'Freidberg', 'Wenninger', 'Manual'")
    return np.where(np.asarray(κ) <= 0, np.nan, κ)


def f_Kappa_95(kappa):
    """
    κ₉₅ = κ_edge / 1.12   (ITER 1989 guideline).

    Parameters
    ----------
    kappa : float or ndarray

    Returns
    -------
    kappa_95 : float or ndarray
    """
    return kappa / 1.12


def f_Delta(kappa):
    """
    Triangularity estimate from elongation: δ = 0.6 (κ − 1).  [TREND p.53]

    Parameters
    ----------
    kappa : float or ndarray

    Returns
    -------
    delta : float or ndarray
    """
    return 0.6 * (kappa - 1)


def f_Delta_95(delta):
    """
    δ₉₅ = δ_edge / 1.5   (ITER 1989 guideline).

    Parameters
    ----------
    delta : float or ndarray

    Returns
    -------
    delta_95 : float or ndarray
    """
    return delta / 1.5


if __name__ == "__main__":
    # Elongation scaling laws: κ_edge(A) and κ_95(A)
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_kappa_scaling()

def kappa_profile(rho, kappa_edge, kappa_95, rho_95=0.95):
    """
    Elongation radial profile — flat core + edge rise (PCHIP).

    For a nearly flat toroidal current profile, Ball & Parra (2015, Figs. 3-4)
    show dκ/dρ ≈ 0 across the bulk plasma.  The GS expansion around the
    magnetic axis gives κ(0) = sqrt(B/A), set self-consistently — NOT forced
    to 1.  For typical flat-current equilibria κ(0) ≈ κ₉₅.

    Control points (PCHIP):
        (0,    κ₉₅)    — flat core; first two nodes identical → zero slope
        (ρ₉₅, κ₉₅)    — 95%-surface constraint
        (1,   κ_edge)  — LCFS; steep rise confined to edge layer [ρ₉₅, 1]

    Parameters
    ----------
    rho : float or ndarray  in [0, 1]
    kappa_edge : float  LCFS elongation
    kappa_95   : float  95%-surface elongation (use f_Kappa_95 for ITER 1989)
    rho_95     : float  default 0.95

    Returns
    -------
    kappa : float or ndarray

    References
    ----------
    Ball & Parra, PPCF 57, 035006 (2015).
    Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980).
    """
    rho = np.asarray(rho, dtype=float)
    interp = PchipInterpolator([0.0,      rho_95,   1.0       ],
                               [kappa_95, kappa_95, kappa_edge])
    return interp(rho)


def delta_profile(rho, delta_edge, delta_95, rho_95=0.95):
    """
    Triangularity radial profile — edge-peaked PCHIP interpolation.

    The Grad-Shafranov Taylor expansion around the magnetic axis shows
    that all Fourier harmonics of order m ≥ 3 vanish on axis, giving the
    exact constraint δ(0) = 0 (Freidberg 2014, §6).  Ball & Parra (2015,
    Fig. 4) further demonstrate that higher-order harmonics penetrate poorly
    inward for typical current profiles: triangularity is essentially confined
    to the edge pedestal layer.

    PCHIP over three control points encodes both constraints and the
    prescribed 95%-surface value in a single, physically consistent profile:

        Control points:
            (0,     0)          — exact GS on-axis constraint, δ(0) = 0
            (ρ₉₅,  δ₉₅)        — 95%-surface value from equilibrium / scaling
            (1,    δ_edge)      — LCFS triangularity

    PCHIP (Fritsch & Carlson 1980) is preferred over a simple power-law
    because:
      (i)  It is C¹-continuous at all nodes, unlike a ρ^s_δ law which has
           an infinite derivative at ρ = 0 for s_δ < 1;
      (ii) The exponent s_δ of a power-law is rigidly tied to a particular
           δ₉₅/δ_edge ratio (s_δ ≈ 7.9 for ITER 1989 scaling δ₉₅ = δ/1.5),
           making it unsuitable when δ₉₅ is prescribed independently, e.g.
           from a free-boundary solver (CHEASE, Lütjens et al. 1996; EFIT,
           Lao et al. 2005);
      (iii) PCHIP guarantees local monotonicity without any tuning,
           preventing spurious oscillations between sparse data points
           (property proved in Fritsch & Carlson 1980, Thm. 2.1).

    Parameters
    ----------
    rho : float or ndarray
        Normalised radial coordinate r/a in [0, 1].
    delta_edge : float
        Triangularity at the LCFS (ρ = 1).
    delta_95 : float
        Triangularity at ρ₉₅.  Use f_Delta_95(delta_edge) for ITER 1989
        scaling (δ_edge / 1.5).
    rho_95 : float, optional
        Normalised position of the 95% flux surface (default 0.95).

    Returns
    -------
    delta : float or ndarray
        Monotone profile from 0 to δ_edge, edge-peaked.
        Works for both positive (δ_edge > 0, D-shape) and negative
        (δ_edge < 0, reversed-D / negative triangularity) configurations.
        The only geometric constraint is |δ_edge| < 1, required by the
        arcsin in the Miller parameterisation; typical tokamak values
        (|δ| ≤ 0.7) satisfy this comfortably.
        For δ_edge < 0, PCHIP produces a monotonically decreasing profile
        0 → δ₉₅ → δ_edge with full monotonicity preservation (Fritsch &
        Carlson 1980, Thm. 2.1).

    References
    ----------
    Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980) — PCHIP algorithm
        and monotonicity preservation theorem.
    Ball & Parra, PPCF 57, 035006 (2015) — radial penetration of shaping,
        Figs. 3-4; triangularity confined to edge for flat current profiles.
    Freidberg, Ideal MHD, Cambridge University Press (2014) — GS axis
        constraint: m ≥ 3 harmonics vanish at the magnetic axis.
    Lütjens et al., Comput. Phys. Commun. 97, 219 (1996) — CHEASE
        free-boundary equilibrium code.
    Lao et al., Fusion Sci. Technol. 48, 968 (2005) — EFIT spline
        equilibrium reconstruction.
    """
    rho = np.asarray(rho, dtype=float)
    interp = PchipInterpolator([0.0, rho_95,   1.0       ],
                               [0.0, delta_95, delta_edge])
    return interp(rho)


if __name__ == "__main__":
    # Radial κ(ρ) and δ(ρ) profiles — ITER reference
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_shaping_profiles(kappa_edge=1.85, delta_edge=0.50)

def miller_RZ(rho, theta, R0, a, kappa_edge, delta_edge,
              kappa_95=None, delta_95=None, rho_95=0.95):
    """
    (R, Z) coordinates of Miller-parameterised flux surfaces.

    R(ρ,θ) = R₀ + ρa · cos(θ + arcsin(δ(ρ)) sinθ)
    Z(ρ,θ) = κ(ρ) · ρa · sinθ

    Shaping profiles:
      Academic : κ = κ_edge (const),  δ = 0
      D0FUS    : PCHIP κ (flat core) + PCHIP δ (edge-peaked)

    No Shafranov shift (outside scope of a 0D code).

    Parameters
    ----------
    rho   : float or ndarray
    theta : float or ndarray  [rad]
    R0, a : float  [m]
    kappa_edge, delta_edge : float
    kappa_95, delta_95 : float or None  (ITER 1989 defaults)
    rho_95 : float  default 0.95

    Returns
    -------
    R, Z : float or ndarray  [m]

    References
    ----------
    Miller et al., Phys. Plasmas 5, 973 (1998).
    """
    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa_edge)
    if delta_95 is None:
        delta_95 = f_Delta_95(delta_edge)

    rho   = np.asarray(rho,   dtype=float)
    theta = np.asarray(theta, dtype=float)
    k = kappa_profile(rho, kappa_edge, kappa_95, rho_95)
    d = delta_profile(rho, delta_edge, delta_95, rho_95)
    R = R0 + rho * a * np.cos(theta + np.arcsin(d) * np.sin(theta))
    Z = k * rho * a * np.sin(theta)
    return R, Z


if __name__ == "__main__":
    # Miller flux surfaces — Academic vs D0FUS +/- delta
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_miller_surfaces(R0=6.2, a=2.0, kappa_edge=1.85, delta_edge=0.50)

def precompute_Vprime(R0, a, kappa_edge, delta_edge,
                      geometry_model='D0FUS',
                      kappa_95=None, delta_95=None, rho_95=0.95,
                      N_rho=200, N_theta=200):
    """
    Precompute V'(ρ) = dV/dρ on a radial grid.

    Called once per design point; the returned tuple is passed to all
    functions that need volume integrals.

    Parameters
    ----------
    R0, a : float  [m]
    kappa_edge, delta_edge : float
    geometry_model : 'Academic' | 'D0FUS'
        'Academic' : V'(ρ) = 4π²R₀a²κ_edge·ρ
        'D0FUS'    : numerical Jacobian from Miller PCHIP profiles (κ and δ)
    kappa_95, delta_95 : float or None  (ITER 1989 defaults; D0FUS only)
    rho_95 : float  default 0.95
    N_rho, N_theta : int  grid resolution (D0FUS only)

    Returns
    -------
    rho_grid : ndarray  (N_rho,)
    Vprime   : ndarray  (N_rho,)  [m³]
    V_total  : float  [m³]
    """
    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa_edge)
    if delta_95 is None:
        delta_95 = f_Delta_95(delta_edge)

    rho_grid = np.linspace(1e-6, 1.0, N_rho)

    if geometry_model == 'Academic':
        Vprime  = 4.0 * np.pi**2 * R0 * a**2 * kappa_edge * rho_grid
        V_total = 2.0 * np.pi**2 * R0 * a**2 * kappa_edge
        return rho_grid, Vprime, V_total

    elif geometry_model == 'D0FUS':
        theta  = np.linspace(0.0, 2.0*np.pi, N_theta, endpoint=False)
        dtheta = theta[1] - theta[0]
        drho   = rho_grid[1] - rho_grid[0]

        RHO, THETA = np.meshgrid(rho_grid, theta, indexing='ij')
        R, Z = miller_RZ(RHO, THETA, R0, a, kappa_edge, delta_edge,
                         kappa_95, delta_95, rho_95)

        # Numerical Jacobian |∂(R,Z)/∂(ρ,θ)|
        dR_drho   = np.gradient(R, drho,   axis=0)
        dR_dtheta = np.gradient(R, dtheta, axis=1)
        dZ_drho   = np.gradient(Z, drho,   axis=0)
        dZ_dtheta = np.gradient(Z, dtheta, axis=1)
        jac = np.abs(dR_drho * dZ_dtheta - dR_dtheta * dZ_drho)

        Vprime  = np.sum(2.0 * np.pi * R * jac, axis=1) * dtheta
        V_total = float(np.trapezoid(Vprime, rho_grid))
        return rho_grid, Vprime, V_total

    else:
        raise ValueError(f"Unknown geometry_model '{geometry_model}'. "
                         "Valid: 'Academic', 'D0FUS'.")


def interpolate_Vprime(rho, rho_grid, Vprime):
    """
    Interpolate precomputed V'(ρ) at arbitrary radial positions.

    Parameters
    ----------
    rho : float or ndarray
    rho_grid, Vprime : ndarray from precompute_Vprime()

    Returns
    -------
    Vp : float or ndarray  [m³]
    """
    return np.interp(rho, rho_grid, Vprime)


# =============================================================================
# Plasma volume
# =============================================================================

def f_plasma_volume(R0, a, kappa, delta, Vprime_data=None):
    """
    Plasma volume [m³].

    Modes:
      Vprime_data = None        → O(δ²) analytical Miller formula
      Vprime_data = (ρ, V', V) → precomputed from precompute_Vprime()

    Parameters
    ----------
    R0, a : float  [m]
    kappa : float  edge elongation
    delta : float  edge triangularity
    Vprime_data : tuple or None

    Returns
    -------
    V : float  [m³]
    """
    if Vprime_data is not None:
        return Vprime_data[2]
    return 2*np.pi**2 * R0 * a**2 * kappa * (1 - (a*delta)/(4*R0) - delta**2/8)


if __name__ == "__main__":
    # Plasma volume formula comparison — scan over κ and δ
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_volume_comparison(R0=6.2, a=2.0)

def f_first_wall_surface(R0, a, kappa_edge, delta_edge=0.0,
                         geometry_model='Academic', N_theta=2000):
    """
    First wall surface area [m²], assuming the first wall follows the LCFS.

    Two models:
      'Academic' : Ramanujan ellipse perimeter × 2πR₀
                   P_e = πa · [3(1+κ) − √((3+κ)(1+3κ))]
                   S   = 2πR₀ · P_e

      'D0FUS'    : Numerical revolution of the LCFS Miller contour
                   S   = 2π ∫₀²π R(θ) · |dl/dθ| dθ
                   with |dl/dθ| = √((∂R/∂θ)² + (∂Z/∂θ)²)  at ρ = 1

    Parameters
    ----------
    R0, a : float  [m]
    kappa_edge : float  [-]
    delta_edge : float  [-]  (ignored for 'Academic')
    geometry_model : 'Academic' | 'D0FUS'
    N_theta : int  poloidal resolution (D0FUS)

    Returns
    -------
    S : float  [m²]

    References
    ----------
    Ramanujan (1914) — ellipse perimeter approximation.
    Miller et al., Phys. Plasmas 5, 973 (1998).
    """
    if geometry_model == 'Academic':
        Pe = np.pi * a * (3*(1 + kappa_edge)
                          - np.sqrt((3 + kappa_edge)*(1 + 3*kappa_edge)))
        return 2*np.pi * R0 * Pe

    elif geometry_model == 'D0FUS':
        theta  = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
        dtheta = theta[1] - theta[0]
        # LCFS Miller contour at ρ = 1 (use edge shaping values directly)
        R_t = R0 + a * np.cos(theta + np.arcsin(delta_edge) * np.sin(theta))
        Z_t = kappa_edge * a * np.sin(theta)
        dR  = np.gradient(R_t, dtheta)
        dZ  = np.gradient(Z_t, dtheta)
        dl  = np.sqrt(dR**2 + dZ**2)          # poloidal arc element [m/rad]
        return 2*np.pi * np.trapezoid(R_t * dl, theta)

    else:
        raise ValueError(f"Unknown geometry_model '{geometry_model}'. "
                         "Valid: 'Academic', 'D0FUS'.")


if __name__ == "__main__":
    # First wall surface area — Academic vs D0FUS Miller
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_first_wall_surface(R0=6.2, a=2.0)

#%% n, T, p and Pfus

"""
Radial plasma profiles and fusion power integrals for D0FUS
==================================================================================

Two geometry modes are supported throughout this module, controlled by the
Vprime_data argument:

  Academic  (Vprime_data = None)
      Volume element: dV = 4π²R₀a²κ · ρ dρ  (cylindrical torus, constant κ)
      Plasma volume:  V  = 2π²R₀κa²  [Wesson]
      Profile integrals use the cylindrical weight 2ρ dρ.
      Fast, analytical formulas available for parabolic profiles.

  D0FUS  (Vprime_data = (rho_grid, Vprime, V_total) from precompute_Vprime())
      Volume element: dV = V'_Miller(ρ) dρ  (full Miller geometry, PCHIP κ/δ)
      All integrals use the precomputed Jacobian; n(ρ) and T(ρ) are midplane
      radial profiles — the geometry enters only through the volume weight.

Profile model
-------------
Parabola-with-pedestal (Lackner 1990; used in ITER Physics Basis 1999):

  X(ρ) = X_ped + (X₀ − X_ped) · (1 − (ρ/ρ_ped)²)^ν     ρ ≤ ρ_ped
  X(ρ) = X_ped · (1 − ρ)/(1 − ρ_ped)                     ρ > ρ_ped  (X_sep = 0)

Special case ρ_ped = 1, f_ped = 0 → purely parabolic: X(ρ) = X̄(1+ν)(1−ρ²)^ν.

ITER reference values (used in __main__ tests)
-----------------------------------------------
  R₀ = 6.2 m,  a = 2.0 m,  κ = 1.85,  δ = 0.50,  Ip = 15 MA
  P_fus = 500 MW,  T̄ = 8.9 keV,  n̄ ≈ 1.01 × 10²⁰ m⁻³

References
----------
Lackner, Comments Plasma Phys. Control. Fusion 13, 163 (1990) — parabolic profile.
ITER Physics Basis, Nucl. Fusion 39, 2175 (1999) — profile parameterisation.
Bosch & Hale, Nucl. Fusion 32, 611 (1992) — DT reactivity fit.
Wesson, Tokamaks, 4th ed., Oxford (2011) — plasma volume and pressure.
Greenwald, PPCF 44, R27 (2002) — density limit.
Martin et al., JPCS 123, 012033 (2008) — L-H power threshold.
"""


# Module-level cache for _profile_core_peak (replaces @lru_cache to
# support unhashable Vprime_data arrays via id()-based keys).
_profile_core_peak_cache = {}


def _profile_core_peak(nu, rho_ped, f_ped, Vprime_data=None):
    """
    Core-peak normalised value X₀/X̄ for a parabola-with-pedestal profile.

    The volume-average constraint ⟨X⟩_vol = X̄ is enforced using:
      - cylindrical weight  w(ρ) = 2ρ          when Vprime_data is None
      - Miller weight  w(ρ) = V'(ρ)/V_total    when Vprime_data is provided

    This ensures that the profile normalisation is consistent with
    whichever volume element is used by downstream integrals.

    Parameters
    ----------
    nu      : float  Core peaking exponent (ν_n or ν_T).
    rho_ped : float  Normalised pedestal radius ∈ (0, 1].
    f_ped   : float  X_ped / X̄ (pedestal fraction of the volume average).
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → cylindrical weight (Academic mode).

    Returns
    -------
    X0_frac : float  X₀ / X̄
    """
    # ── Purely parabolic + cylindrical: exact analytical result ────────
    if rho_ped >= 1.0 and Vprime_data is None:
        return 1.0 + nu

    # ── Cache lookup (id of rho_grid array is stable per design point) ─
    geo_key = id(Vprime_data[0]) if Vprime_data is not None else None
    cache_key = (nu, rho_ped, f_ped, geo_key)
    cached = _profile_core_peak_cache.get(cache_key)
    if cached is not None:
        return cached

    # ── Build radial grid and normalised volume weight ─────────────────
    if Vprime_data is not None:
        rho_arr = Vprime_data[0]
        w_vol   = Vprime_data[1] / Vprime_data[2]   # V'(ρ)/V, ∫ w dρ = 1
    else:
        rho_arr = np.linspace(0, 1, 200)
        w_vol   = 2.0 * rho_arr                      # 2ρ,       ∫ w dρ = 1

    # ── Purely parabolic + Miller: no tanh, just X₀ from Miller weight ─
    if rho_ped >= 1.0:
        # X(ρ) = X₀ (1-ρ²)^ν,  ⟨X⟩ = X₀ ∫ (1-ρ²)^ν w dρ = 1
        g = np.maximum(1.0 - rho_arr**2, 0.0)**nu
        I_g = np.trapezoid(g * w_vol, rho_arr)
        result = 1.0 / I_g if I_g > 1e-15 else 1.0 + nu
        _profile_core_peak_cache[cache_key] = result
        return result

    # ── Tanh-envelope: solve ⟨X⟩ = X̄ via linearity in X₀ ─────────────
    # X(ρ) = [f_ped + (X0_frac - f_ped) · g(ρ)] × h(ρ)
    #   g(ρ) = (1 - (ρ/ρ_ped)²)^ν       core parabola
    #   h(ρ) = tanh envelope
    #
    # ⟨X⟩/X̄ = 1 = f_ped · I_h + (X0_frac - f_ped) · I_gh
    # ⟹ X0_frac = f_ped + (1 - f_ped · I_h) / I_gh
    #
    # where I_h  = ∫₀¹ h(ρ) · w(ρ) dρ
    #       I_gh = ∫₀¹ g(ρ) · h(ρ) · w(ρ) dρ
    delta   = (1.0 - rho_ped) / 2.0
    rho_mid = rho_ped + delta            # = (1 + rho_ped) / 2
    hw      = delta / 2.0               # half-width = (1 - rho_ped) / 4

    h = 0.5 * (1.0 + np.tanh((rho_mid - rho_arr) / hw))
    g = np.maximum(1.0 - (rho_arr / rho_ped)**2, 0.0)**nu

    I_h  = np.trapezoid(h * w_vol, rho_arr)
    I_gh = np.trapezoid(g * h * w_vol, rho_arr)

    if I_gh < 1e-15:
        result = 1.0 + nu     # degenerate fallback
    else:
        result = f_ped + (1.0 - f_ped * I_h) / I_gh

    _profile_core_peak_cache[cache_key] = result
    return result


def _tanh_envelope(rho, rho_ped):
    """
    Pedestal tanh envelope: ~1 in core, ~0 at separatrix.

    Centered at ρ_mid = (1 + ρ_ped)/2 (midpoint between pedestal and
    separatrix) with half-width w = (1 − ρ_ped)/4:

        h(ρ) = ½ (1 + tanh((ρ_mid − ρ) / w))

    Key values:
        h(ρ_ped) ≈ 0.98    (pedestal top preserved)
        h(ρ_mid) = 0.50     (transition midpoint)
        h(1)     ≈ 0.02     (separatrix ~ zero)
    """
    delta  = (1.0 - rho_ped) / 2.0
    rho_mid = rho_ped + delta          # = (1 + rho_ped) / 2
    w       = delta / 2.0             # half-width = (1 - rho_ped) / 4
    return 0.5 * (1.0 + np.tanh((rho_mid - rho) / w))


def _tanh_envelope_deriv(rho, rho_ped):
    """Derivative dh/dρ. Peaked at ρ_mid with amplitude −1/(2w)."""
    delta  = (1.0 - rho_ped) / 2.0
    rho_mid = rho_ped + delta
    w       = delta / 2.0
    arg     = (rho_mid - rho) / w
    sech2   = 1.0 / np.cosh(np.clip(arg, -30, 30))**2
    return -sech2 / (2.0 * w)


def f_Tprof(Tbar, nu_T, rho, rho_ped=1.0, T_ped_frac=0.0, Vprime_data=None):
    """
    Electron temperature radial profile T(ρ)  [keV].

    Two models selected by rho_ped:

    1. Purely parabolic (rho_ped = 1.0, default):
         T(ρ) = T̄(1 + ν_T)(1 − ρ²)^ν_T

    2. Parabola-with-tanh-pedestal (rho_ped < 1.0):
         T(ρ) = T_core(ρ) × h(ρ)

       where:
         T_core(ρ) = T_ped + (T₀ − T_ped)·(1 − (ρ/ρ_ped)²)^ν_T
         h(ρ)      = ½(1 + tanh((ρ_mid − ρ) / w))
         ρ_mid     = (1 + ρ_ped)/2,  w = (1 − ρ_ped)/4

       The tanh envelope replaces the linear SOL ramp, providing a smooth
       and physically motivated pedestal transition with continuous gradient.
       T₀ is determined from the volume-average constraint ⟨T⟩_vol = T̄.

    Profile normalisation:
      When Vprime_data is None (Academic), the volume average uses the
      cylindrical weight 2ρ dρ.  When Vprime_data is provided (D0FUS),
      the Miller weight V'(ρ)/V is used, ensuring consistency with
      downstream Miller-weighted integrals.

    Parameters
    ----------
    Tbar      : float  Volume-averaged electron temperature [keV].
    nu_T      : float  Temperature peaking exponent (core region).
    rho       : float or ndarray  Normalised minor radius r/a ∈ [0, 1].
    rho_ped   : float  Normalised pedestal radius (default 1.0 → parabolic).
    T_ped_frac: float  T_ped / T̄ (ignored when rho_ped = 1.0).
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → cylindrical normalisation (Academic mode).

    Returns
    -------
    T : float or ndarray  [keV]

    References
    ----------
    Lackner, Comments Plasma Phys. Control. Fusion 13, 163 (1990).
    ITER Physics Basis, Nucl. Fusion 39, 2175 (1999) — Sec. 2.
    Groebner & Osborne, Phys. Plasmas 5, 1800 (1998) — mtanh.
    """
    rho = np.asarray(rho, dtype=float)

    if rho_ped >= 1.0:
        T0 = _profile_core_peak(nu_T, rho_ped, 0.0, Vprime_data) * Tbar
        return T0 * np.maximum(1.0 - rho**2, 0.0)**nu_T

    T_ped = T_ped_frac * Tbar
    T0    = _profile_core_peak(nu_T, rho_ped, T_ped_frac, Vprime_data) * Tbar

    g = np.maximum(1.0 - (rho / rho_ped)**2, 0.0)**nu_T
    T_core = T_ped + (T0 - T_ped) * g
    h = _tanh_envelope(rho, rho_ped)

    return T_core * h


def f_nprof(nbar, nu_n, rho, rho_ped=1.0, n_ped_frac=0.0, Vprime_data=None):
    """
    Electron density radial profile n(ρ)  [10²⁰ m⁻³].

    Identical parameterisation to f_Tprof; see that docstring.

    Parameters
    ----------
    nbar      : float  Volume-averaged electron density [10²⁰ m⁻³].
    nu_n      : float  Density peaking exponent (core region).
    rho       : float or ndarray  r/a ∈ [0, 1].
    rho_ped   : float  Normalised pedestal radius (default 1.0 → parabolic).
    n_ped_frac: float  n_ped / n̄ (ignored when rho_ped = 1.0).
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → cylindrical normalisation (Academic mode).

    Returns
    -------
    n : float or ndarray  [10²⁰ m⁻³]
    """
    rho = np.asarray(rho, dtype=float)

    if rho_ped >= 1.0:
        n0 = _profile_core_peak(nu_n, rho_ped, 0.0, Vprime_data) * nbar
        return n0 * np.maximum(1.0 - rho**2, 0.0)**nu_n

    n_ped = n_ped_frac * nbar
    n0    = _profile_core_peak(nu_n, rho_ped, n_ped_frac, Vprime_data) * nbar

    g = np.maximum(1.0 - (rho / rho_ped)**2, 0.0)**nu_n
    n_core = n_ped + (n0 - n_ped) * g
    h = _tanh_envelope(rho, rho_ped)

    return n_core * h


if __name__ == "__main__":
    # Normalised density and temperature profiles — L/H/Advanced modes
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_nT_profiles()

def f_nbar_line(nbar_vol, nu_n, rho_ped=1.0, n_ped_frac=0.0, N=2000):
    """
    Convert volume-averaged electron density to line-averaged density.

    Interferometers measure n̄_line along a horizontal midplane chord (Z = 0).
    For the Academic geometry (δ = 0), the midplane flux-surface intersection
    gives R(ρ, θ=0) = R₀ + ρa exactly, so dR = a dρ and the chord maps ρ
    uniformly on [0, 1]:

        n̄_line = ∫₀¹ n(ρ) dρ          (exact for δ = 0)

    The volume average (cylindrical approximation) is:
        n̄_vol  = 2 ∫₀¹ n(ρ) ρ dρ

    The ratio n̄_line/n̄_vol > 1 for any peaked profile (the chord integral
    does not down-weight the dense core).  For a parabolic profile:
        ν_n = 0.5  →  ratio ≈ 1.18;   ν_n = 1.0  →  ratio ≈ 1.33.

    Limitation — triangularity correction (D0FUS mode)
    --------------------------------------------------
    In Miller geometry the outer midplane (θ = 0) gives:
        R(ρ, θ=0) = R₀ + ρa · cos(arcsin(δ(ρ)))
        dR/dρ     = a · cos(arcsin(δ(ρ)))    [≠ a when δ(ρ) ≠ 0]

    The exact chord integral is therefore:
        n̄_line = ∫₀¹ n(ρ) · cos(arcsin(δ(ρ))) dρ   (D0FUS mode)

    This function uses the δ = 0 approximation for both geometry modes.
    For ITER-class δ_edge ≈ 0.5, the PCHIP δ(ρ) profile is confined to the
    edge layer (δ ≲ 0.1 for ρ < 0.8), so the mean Jacobian correction is
    cos(arcsin(δ_edge)) × edge-layer fraction ≈ 0.87 × 0.05 ≲ 1 % error on
    n̄_line.  The correction grows to ~3 % only for very high triangularity
    (δ_edge ≈ 0.8) and is negligible in all ITER/EU-DEMO relevant cases.

    Note: elongation κ does NOT affect the midplane chord.  The horizontal
    line Z = 0 intersects the plasma at fixed ρ values independent of κ,
    since the Miller Z-coordinate scales as κ sin θ and vanishes at θ = 0.

    Parameters
    ----------
    nbar_vol   : float  Volume-averaged density [any unit].
    nu_n       : float  Density peaking exponent.
    rho_ped    : float  Normalised pedestal radius (default 1.0).
    n_ped_frac : float  n_ped / n̄.
    N          : int    Integration points (default 2000).

    Returns
    -------
    nbar_line : float  Line-averaged density [same unit as nbar_vol].

    Notes
    -----
    The IPB98(y,2), ITPA20, and ITER89-P confinement scalings all use
    line-averaged density (typically in units of 10¹⁹ m⁻³).  The Greenwald
    density limit and the Martin L-H threshold also use line-averaged density.

    References
    ----------
    ITER Physics Design Guidelines (1989) — line-average definition.
    ITER Physics Basis, Nucl. Fusion 39, 2175 (1999) — IPB98(y,2).
    Greenwald, PPCF 44, R27 (2002).
    Martin et al., JPCS 123, 012033 (2008).
    Miller et al., Phys. Plasmas 5, 973 (1998) — outer-midplane Jacobian.
    """
    rho_arr = np.linspace(0.0, 1.0, N)
    return float(np.trapezoid(f_nprof(nbar_vol, nu_n, rho_arr,
                                      rho_ped, n_ped_frac), rho_arr))


def f_nbar_vol_from_line(nbar_line, nu_n, rho_ped=1.0, n_ped_frac=0.0, N=2000):
    """
    Convert line-averaged electron density to volume-averaged density.

    Inverse of f_nbar_line(); uses the normalised profile ratio.

    Parameters
    ----------
    nbar_line  : float  Line-averaged density [any unit].
    nu_n       : float  Density peaking exponent.
    rho_ped, n_ped_frac : float  Pedestal parameters.
    N          : int    Integration points.

    Returns
    -------
    nbar_vol : float  Volume-averaged density [same unit as nbar_line].
    """
    ratio = f_nbar_line(1.0, nu_n, rho_ped, n_ped_frac, N)
    return nbar_line / ratio

if __name__ == "__main__":
    # Line vs volume-averaged density: relative difference vs nu_n
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_density_line_vol(nbar_vol=1.01)

def f_sigmav(T):
    """
    DT fusion reactivity ⟨σv⟩ as a function of ion temperature  [m³ s⁻¹].

    Parameterisation of Bosch & Hale (1992) for the D(d,n)⁴He reaction
    (equivalent: D+T → α + n).  Valid range: 0.2–100 keV; maximum relative
    deviation from tabulated data < 0.35 %.

    The Padé-exponential form is:
        θ  = T / [1 − T(c₂ + T(c₄ + Tc₆)) / (1 + T(c₃ + T(c₅ + Tc₇)))]
        ξ  = (B_G²/(4θ))^(1/3)
        σv = c₁ θ √(ξ/(m_c² T³)) exp(−3ξ)   [cm³ s⁻¹] → converted to m³ s⁻¹

    Parameters
    ----------
    T : float or ndarray  Ion temperature [keV].

    Returns
    -------
    sigmav : float or ndarray  ⟨σv⟩_DT  [m³ s⁻¹].

    References
    ----------
    Bosch & Hale, Nucl. Fusion 32, 611 (1992) — Table IV, DT branch.
    """
    # Bosch & Hale fit coefficients for DT (Table IV)
    Bg  = 34.3827           # Gamow constant  [keV^(1/2)]
    mc2 = 1124656.0         # Reduced mass energy  [keV]
    c1  =  1.17302e-9
    c2  =  1.51361e-2
    c3  =  7.51886e-2
    c4  =  4.60643e-3
    c5  =  1.35000e-2
    c6  = -1.06750e-4
    c7  =  1.36600e-5

    T = np.asarray(T, dtype=float)
    scalar = T.ndim == 0
    T = np.atleast_1d(T)

    # Suppress divide-by-zero and invalid warnings at T→0 (σv→0 analytically)
    with np.errstate(divide='ignore', invalid='ignore'):
        θ      = T / (1.0 - T*(c2 + T*(c4 + T*c6)) / (1.0 + T*(c3 + T*(c5 + T*c7))))
        ξ      = (Bg**2 / (4.0 * θ))**(1.0/3.0)
        sigmav = c1 * θ * np.sqrt(ξ / (mc2 * T**3)) * np.exp(-3.0 * ξ) * 1e-6

    # At T=0 the integrand is zero (no fusion reactivity)
    sigmav = np.where(np.isfinite(sigmav), sigmav, 0.0)
    return float(sigmav[0]) if scalar else sigmav


# =============================================================================
# Required electron density for a target fusion power
# =============================================================================

def f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, kappa,
           rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
           Vprime_data=None, f_imp=0.0):
    """
    Required volume-averaged electron density to achieve a target fusion power.

    The fusion power density is:
        p_fus(ρ) = (E_α + E_n)/4 · n_fuel²(ρ) · ⟨σv⟩(T(ρ))

    Integrating over the plasma volume and imposing P_fus = ∫ p_fus dV gives:

        n̄ = 2 √[P_fus / (I_fus · (E_α + E_n) · V)]

    where I_fus = ∫ ⟨σv⟩(T(ρ)) · n̂²(ρ) · w(ρ) dρ  is the normalised
    reactivity integral and w(ρ) is the volume weight (see below).

    Volume weights:
      Academic : w(ρ) = 2ρ   (cylindrical, V = 2π²R₀κa²)
      D0FUS    : w(ρ) = V'_Miller(ρ)/V_Miller  (full Miller geometry)

    The profiles n(ρ) and T(ρ) are midplane radial profiles; the geometry
    enters only through the volume weight.

    Particle balance (quasi-neutrality):
        n_e = n_fuel + 2 n_α + Σ_j Z_j n_j
    where Σ_j Z_j n_j / n_e = f_imp is the dimensionless impurity charge fraction.
    Solving for n_fuel:
        n_fuel = n_e · (1 − 2·f_α − f_imp)

    Parameters
    ----------
    P_fus      : float  Target fusion power [MW].
    nu_n       : float  Density peaking exponent.
    nu_T       : float  Temperature peaking exponent.
    f_alpha    : float  Helium-ash fraction n_α/n_e.
    Tbar       : float  Volume-averaged temperature [keV].
    R0, a      : float  Major and minor radii [m].
    kappa      : float  Plasma elongation (used for Academic volume only).
    rho_ped    : float  Normalised pedestal radius (default 1.0).
    n_ped_frac : float  n_ped / n̄.
    T_ped_frac : float  T_ped / T̄.
    Vprime_data : tuple or None
        Precomputed (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → Academic mode (cylindrical volume, quad integration).
    f_imp      : float, optional
        Dimensionless impurity charge fraction Σ_j Z_j n_j / n_e (default 0.0).
        Accounts for fuel dilution by metallic or gaseous impurities beyond
        helium ash.  For a pure D–T plasma, f_imp = 0.  For a W-seeded plasma
        with Z_eff ≈ 1.5 and low-Z puffing, f_imp ~ 0.02–0.10.
        Note: f_imp and f_alpha are independent corrections; the combined
        dilution factor is (1 − 2·f_alpha − f_imp).

    Returns
    -------
    n_e : float  Volume-averaged electron density [10²⁰ m⁻³].

    References
    ----------
    Wesson, Tokamaks, 4th ed., Oxford (2011) — fusion power integral.
    Bosch & Hale, Nucl. Fusion 32, 611 (1992) — ⟨σv⟩ parameterisation.
    Freidberg, Plasma Physics and Fusion Energy (2007) — dilution factor.
    """
    P_watt = P_fus * 1e6   # [W]

    if Vprime_data is not None:
        # D0FUS mode: Miller V'(ρ) integration
        rho_grid, Vprime, V_total = Vprime_data
        T_arr   = f_Tprof(Tbar, nu_T, rho_grid, rho_ped, T_ped_frac,
                          Vprime_data)
        n_hat   = f_nprof(1.0,  nu_n, rho_grid, rho_ped, n_ped_frac,
                          Vprime_data)
        sv_arr  = f_sigmav(T_arr)
        # Guard against NaN/inf at ρ→1 where T→0 and ⟨σv⟩→0 rapidly
        integrand = np.nan_to_num(sv_arr * n_hat**2 * Vprime,
                                   nan=0.0, posinf=0.0)
        # Normalised reactivity integral I_fus  [m³ s⁻¹]
        I_fus = np.trapezoid(integrand, rho_grid) / V_total
        V     = V_total
    else:
        # Academic mode: cylindrical weight 2ρ dρ
        def _integrand(rho):
            return (f_sigmav(f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac))
                    * f_nprof(1.0, nu_n, rho, rho_ped, n_ped_frac)**2
                    * 2.0 * rho)
        I_fus, _ = quad(_integrand, 0.0, 1.0, limit=200)
        V     = 2.0 * np.pi**2 * R0 * kappa * a**2   # Wesson volume

    # Fuel ion density from fusion power balance
    # Guard: I_fus ≤ 0 if T is too low for appreciable DT reactivity, or
    # if the profile integration fails numerically → sqrt would give inf or nan.
    if I_fus <= 0.0:
        raise ValueError(
            f"f_nbar: non-positive normalised reactivity integral "
            f"I_fus = {I_fus:.3e} m³ s⁻¹.  "
            f"Verify that Tbar = {Tbar:.2f} keV is above the DT ignition "
            "threshold and that the profile exponents produce a non-zero "
            "peak temperature (check nu_T, rho_ped, T_ped_frac)."
        )
    n_fuel = 2.0 * np.sqrt(P_watt / (I_fus * (E_ALPHA + E_N) * V))   # [m⁻³]

    # Electron density correcting for helium-ash and impurity dilution
    # n_fuel = n_e (1 - 2 f_alpha - f_imp)  →  n_e = n_fuel / dilution_factor
    dilution = 1.0 - 2.0 * f_alpha - f_imp
    if dilution <= 0.0:
        # Unphysical operating point: fuel fully diluted.
        # Return a very large density so the solver residuals blow up
        # naturally, rather than raising an exception that crashes the solver.
        return 1e6   # [10²⁰ m⁻³] — absurdly high, drives residual to reject this point
    n_e = n_fuel / dilution   # [m⁻³]

    return n_e / 1e20   # [10²⁰ m⁻³]


# =============================================================================
# Volume-averaged plasma pressure
# =============================================================================

def f_pbar(nu_n, nu_T, n_bar, Tbar,
           rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
           Vprime_data=None):
    """
    Volume-averaged plasma pressure  p̄ = 2⟨nT⟩_vol  [MPa].

    The factor 2 accounts for equal ion and electron pressure (T_i = T_e).
    The integral ⟨nT⟩_vol = ∫ n(ρ)·T(ρ)·w(ρ) dρ uses:

      Academic mode:
        Parabolic (analytical): ⟨n̂T̂⟩ = (1+ν_n)(1+ν_T)/(1+ν_n+ν_T)
        Pedestal  (numerical):  cylindrical weight 2ρ dρ
      D0FUS mode:
        Numerical with V'_Miller(ρ)/V weight for all profile shapes.

    Parameters
    ----------
    nu_n       : float  Density peaking exponent.
    nu_T       : float  Temperature peaking exponent.
    n_bar      : float  Volume-averaged electron density [10²⁰ m⁻³].
    Tbar       : float  Volume-averaged temperature [keV].
    rho_ped    : float  Normalised pedestal radius (default 1.0 → parabolic).
    n_ped_frac : float  n_ped / n̄.
    T_ped_frac : float  T_ped / T̄.
    Vprime_data : tuple or None  Precomputed Miller data (D0FUS mode).

    Returns
    -------
    p_bar : float  Mean plasma pressure [MPa].

    Notes
    -----
    The analytical parabolic result:
               = ∫₀¹ (1+ν_n)(1−ρ²)^ν_n · (1+ν_T)(1−ρ²)^ν_T · 2ρ dρ
               = (1+ν_n)(1+ν_T) · ∫₀¹ (1−ρ²)^(ν_n+ν_T) · 2ρ dρ
               = (1+ν_n)(1+ν_T) / (1+ν_n+ν_T)

    References
    ----------
    Wesson, Tokamaks, 4th ed., Oxford (2011) — beta and pressure.
    T.Auclair CEA internal note
    """
    if Vprime_data is not None:
        # D0FUS mode: Miller V'(ρ) integration
        rho_grid, Vprime, V_total = Vprime_data
        n_hat = f_nprof(1.0, nu_n, rho_grid, rho_ped, n_ped_frac,
                        Vprime_data)
        T_hat = f_Tprof(1.0, nu_T, rho_grid, rho_ped, T_ped_frac,
                        Vprime_data)
        C_vol = float(np.trapezoid(n_hat * T_hat * Vprime, rho_grid)) / V_total
        profile_factor = 2.0 * C_vol

    elif rho_ped >= 1.0:
        # Academic mode, parabolic: closed-form analytical result
        profile_factor = 2.0 * (1.0 + nu_n) * (1.0 + nu_T) / (1.0 + nu_n + nu_T)

    else:
        # Academic mode, pedestal: numerical with cylindrical weight 2ρ dρ
        # C_vol = <n̂ T̂>_vol = 2 ∫₀¹ n̂(ρ) T̂(ρ) ρ dρ   (cylindrical volume average)
        # profile_factor = 2 × C_vol   (factor 2 accounts for p = p_e + p_i = 2nT)
        rho_arr = np.linspace(0.0, 1.0, 2000)
        n_hat   = f_nprof(1.0, nu_n, rho_arr, rho_ped, n_ped_frac)
        T_hat   = f_Tprof(1.0, nu_T, rho_arr, rho_ped, T_ped_frac)
        C_vol = 2.0 * float(np.trapezoid(n_hat * T_hat * rho_arr, rho_arr))
        profile_factor = 2.0 * C_vol
        
    # Convert: n [10²⁰ m⁻³] × T [keV] → p [Pa] → [MPa]
    p_bar = profile_factor * (n_bar * 1e20) * (Tbar * E_ELEM * 1e3) / 1e6

    return p_bar


# =============================================================================
# Greenwald density limit
# =============================================================================

def f_nG(Ip, a):
    """
    Greenwald density limit  n_G  [10²⁰ m⁻³].

    Empirical limit observed across L- and H-mode tokamak experiments:
        n_G = Ip / (π a²)   [10²⁰ m⁻³]  with Ip in MA, a in m.

    The Greenwald limit uses the line-averaged density.  Exceeding n_G
    typically leads to a density-limit disruption.

    Parameters
    ----------
    Ip : float  Plasma current [MA].
    a  : float  Minor radius [m].

    Returns
    -------
    nG : float  Greenwald density [10²⁰ m⁻³].

    References
    ----------
    Greenwald, PPCF 44, R27 (2002) — review of density limit experiments.
    """
    return Ip / (np.pi * a**2)


if __name__ == "__main__":

    # ITER geometry and plasma parameters — Shimada et al., Nucl. Fusion 47 (2007) S1
    R0_IT  = 6.2;   a_IT   = 2.0
    kap_IT = 1.85;  del_IT = 0.50;  Ip_IT = 15.0   # MA
    Pfus   = 500.0  # MW
    Tbar   = 8.9    # keV
    falpha = 0.06   # He-ash fraction — ITER Physics Basis (1999), Sec. 2.4

    # H-mode pedestal profile parameters
    # Density: nearly flat core, high pedestal — Coleman et al., Nucl. Fusion 65 (2025) 036039
    # Temperature: moderate peaking — ITER Physics Basis (1999), Sec. 2.3
    nu_n       = 0.1    # Density peaking exponent    [-]
    nu_T       = 1.0    # Temperature peaking exponent [-]
    rho_ped    = 0.94   # Normalised pedestal radius   [-]
    n_ped_frac = 0.90   # n_ped / n_vol                [-]
    T_ped_frac = 0.40   # T_ped / T_vol                [-]

    # Academic mode
    n_ac = f_nbar(Pfus, nu_n, nu_T, falpha, Tbar, R0_IT, a_IT, kap_IT,
                  rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    p_ac = f_pbar(nu_n, nu_T, n_ac, Tbar,
                  rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac)
    nG   = f_nG(Ip_IT, a_IT)
    V_ac = 2.0 * np.pi**2 * R0_IT * kap_IT * a_IT**2

    # D0FUS (Miller) mode — geometry functions defined in §1 above
    k95 = f_Kappa_95(kap_IT)
    d95 = f_Delta_95(del_IT)
    Vpd = precompute_Vprime(R0_IT, a_IT, kap_IT, del_IT,
                            geometry_model='D0FUS',
                            kappa_95=k95, delta_95=d95)
    V_d0 = Vpd[2]
    n_d0 = f_nbar(Pfus, nu_n, nu_T, falpha, Tbar, R0_IT, a_IT, kap_IT,
                  rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                  Vprime_data=Vpd)
    p_d0 = f_pbar(nu_n, nu_T, n_d0, Tbar,
                  rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
                  Vprime_data=Vpd)

    # Console summary
    print("\n── ITER reference: Academic vs D0FUS — H-mode pedestal profile ──")
    print(f"  Profile: nu_n={nu_n}, nu_T={nu_T}, rho_ped={rho_ped}, "
          f"n_ped={n_ped_frac}, T_ped={T_ped_frac}")
    print(f"  {'Quantity':<30} {'Academic':>10} {'D0FUS':>10} {'ITER ref.':>10}")
    print("  " + "─"*62)
    print(f"  {'V  [m3]':<30} {V_ac:>10.1f} {V_d0:>10.1f} {'830':>10}")
    print(f"  {'n_e  [1e20 m-3]':<30} {n_ac:>10.3f} {n_d0:>10.3f} {'1.01':>10}")
    print(f"  {'p_bar  [MPa]':<30} {p_ac:>10.3f} {p_d0:>10.3f} {'0.28':>10}")
    print(f"  {'n_G  [1e20 m-3]':<30} {nG:>10.3f} {nG:>10.3f} {'1.19':>10}")
    print(f"  {'f_n = n_e/n_G  [-]':<30} {n_ac/nG:>10.3f} {n_d0/nG:>10.3f} {'0.85':>10}")

#%% B and beta

def f_B0(Bmax, a, b, R0):
    """
    Estimate the on-axis toroidal magnetic field B0 from the peak field at the
    TF coil inboard midplane, using the 1/R dependence of a toroidal solenoid.

    The TF field scales as B(R) = Bmax * R_min / R, where R_min = R0 - a - b
    is the radius at the inner face of the winding pack.  Evaluating at R = R0
    gives the simple geometric factor (1 - (a+b)/R0).

    Parameters
    ----------
    Bmax : float
        Peak magnetic field at the inboard TF coil winding pack [T].
        This is the superconductor operating point constraint.
    a : float
        Plasma minor radius [m].
    b : float
        Total inboard radial build between the plasma boundary and the TF coil
        inner face: first wall + breeding blanket + neutron shield +
        vacuum vessel + inter-component gaps [m].
    R0 : float
        Plasma major radius [m].

    Returns
    -------
    B0 : float
        On-axis toroidal magnetic field [T].

    References
    ----------
    Wesson, Tokamaks, 4th ed. (2011), §3.2.
    Freidberg, Plasma Physics and Fusion Energy (2007), §12.3.
    """
    B0 = Bmax * (1.0 - (a + b) / R0)
    return B0

def f_Bpol(q95, B_tor, a, R0, kappa=1.0):
    """
    Outer-midplane poloidal magnetic field estimate from q₉₅ and B_T.

    Two independent derivations yield the same correction factor √((1+κ²)/2):

    Route 1 — q-inversion (Freidberg et al. 2015, PoP eq. 30):
        The cylindrical safety factor for an elliptical cross-section is
            q* = π a² B₀ (1 + κ²) / (μ₀ R₀ I_p)
        Combining with the circular q-inversion q ≈ a B_T / (R₀ B_pol) gives
            B_pol = (a B_T / (R₀ q)) × (1+κ²)/2   [intermediate factor]
        which, after accounting for the perimeter correction, reduces to the
        form √((1+κ²)/2).

    Route 2 — Ampere's law (outer midplane):
        The poloidal perimeter of an ellipse with semi-axes (a, κa) is
            L_pol ≈ πa √(2(1+κ²))   [RMS approximation; Ramanujan 1914]
        Ampere's law: B_pol = μ₀ I_p / L_pol.
        Using the circular-plasma relation I_p = π a² B_T / (μ₀ R₀ q) to
        eliminate I_p gives exactly the same result as Route 1:
            B_pol = (a B_T / (R₀ q)) × √((1+κ²)/2)

    Both routes are self-consistent approximations; neither is exact because
    the exact B_pol requires a Grad-Shafranov equilibrium solution.  The
    estimate is adequate for downstream empirical scalings (Martin L-H
    threshold, Eich λ_q) that were fitted using outer-midplane B_pol data.

    For κ = 1 (circular): √((1+κ²)/2) = 1 → recovers the cylindrical limit.
    For κ ≈ 1.7 (ITER): correction ≈ 1.27.
    Note: ignoring the correction underestimates B_pol by ~25 % in well-shaped
    H-mode plasmas, propagating into β_P, λ_q, and L–H threshold estimates.

    Parameters
    ----------
    q95   : float  Safety factor at ψ_N = 0.95 [-]
    B_tor : float  On-axis toroidal field [T]
    a     : float  Minor radius [m]
    R0    : float  Major radius [m]
    kappa : float  Plasma elongation at the LCFS (default 1.0 → cylindrical limit)

    Returns
    -------
    B_pol : float  Outer-midplane poloidal field estimate [T]

    References
    ----------
    Freidberg et al., Phys. Plasmas 22 (2015) 070901 — q* for elliptical plasma.
    Sauter & Medvedev, Fusion Eng. Des. 112 (2016) 633 — shaped q₉₅ formula.
    Wesson, Tokamaks, 4th ed. (2011), §3.2 — Ampere's law and q.
    """
    # Elongation correction: common factor from both derivations above
    kappa_corr = np.sqrt((1.0 + kappa**2) / 2.0)
    B_pol = (a * B_tor * kappa_corr) / (R0 * q95)
    return B_pol


def f_beta_T(pbar_MPa, B0):
    """
    Compute the toroidal plasma beta β_T.

    β_T = 2μ₀ <p> / B₀²

    β_T quantifies the ratio of kinetic plasma pressure to toroidal magnetic
    pressure and is a key figure of merit for confinement efficiency.
    Typical tokamak operating values lie in the range 1–5 %.

    Parameters
    ----------
    pbar_MPa : float
        Volume-averaged plasma pressure [MPa].
    B0 : float
        On-axis toroidal magnetic field [T].

    Returns
    -------
    beta_T : float
        Toroidal beta, dimensionless.

    References
    ----------
    Freidberg (2007), §11.5.
    Wesson (2011), §3.5.
    """
    pbar = pbar_MPa * 1e6   # [MPa] → [Pa]
    beta_T = 2.0 * μ0 * pbar / B0**2
    return beta_T


def f_beta_P(a, κ, pbar_MPa, Ip_MA):
    """
    Compute the poloidal plasma beta β_P.

    β_P = 2μ₀ <p> / <B_pol>²

    The average poloidal field is not solved from the MHD equilibrium; instead
    it is estimated via Ampere's law as <B_pol> ≈ μ₀ Ip / L_pol, where L_pol
    is the effective poloidal circumference of the plasma cross-section.

    For an ellipse with semi-axes a and κa, the RMS perimeter approximation gives:
        L_pol ≈ π √(2 (a² + (κa)²))

    Note: this is the root-mean-square approximation to the ellipse perimeter,
    NOT Ramanujan's formula (which is used in f_first_wall_surface and is
    essentially exact).  The RMS form overestimates the perimeter by ~2% at
    κ = 1.85, leading to β_P values ~4% higher than the Ramanujan-based
    estimate.  The impact on β and β_N is negligible (< 0.5%) because
    β ≈ β_T when β_P ≫ β_T (the tokamak-relevant regime).

    Substituting yields:
        β_P = 2 L_pol² <p> / (μ₀ Ip²)

    β_P > 1 indicates a bootstrap-dominated, high-elongation regime; β_P ~ 0.5–1
    is typical for conventional H-mode scenarios.

    Parameters
    ----------
    a : float
        Plasma minor radius [m].
    κ : float
        Plasma elongation (edge value) [-].
    pbar_MPa : float
        Volume-averaged plasma pressure [MPa].
    Ip_MA : float
        Plasma current [MA].

    Returns
    -------
    beta_P : float
        Poloidal beta, dimensionless.

    References
    ----------
    Wesson (2011), §3.6.
    Freidberg (2007), §11.6.
    """
    pbar = pbar_MPa * 1e6   # [MPa] → [Pa]
    Ip   = Ip_MA   * 1e6   # [MA]  → [A]

    # Effective poloidal circumference — RMS ellipse perimeter approximation
    # (overestimates by ~2% at κ=1.85 vs exact Ramanujan/elliptic integral)
    L_pol = np.pi * np.sqrt(2.0 * (a**2 + (κ * a)**2))

    beta_P = 2.0 * L_pol**2 * pbar / (μ0 * Ip**2)
    return beta_P


def f_beta(beta_T, beta_P):
    """
    Compute the total-field plasma beta β via harmonic combination of β_T and β_P.

    The total magnetic pressure is B² = B_T² + B_P², so the total beta satisfies
    the harmonic relation:
        1/β = 1/β_T + 1/β_P

    This is exact under the assumption that the toroidal and poloidal field
    contributions to magnetic pressure are orthogonal and additive.

    Parameters
    ----------
    beta_T : float
        Toroidal beta, dimensionless.
    beta_P : float
        Poloidal beta, dimensionless.

    Returns
    -------
    beta : float
        Total-field beta, dimensionless.

    References
    ----------
    Wesson (2011), §3.5.
    """
    beta = 1.0 / (1.0 / beta_T + 1.0 / beta_P)
    return beta


def f_beta_N(beta, a, B0, Ip_MA):
    """
    Compute the normalised plasma beta β_N (Troyon normalisation).

    β_N = β [%] · a [m] · B0 [T] / Ip [MA]

    β_N captures the MHD stability limit independently of machine size and
    current.  The ideal Troyon limit corresponds to β_N ≲ β_N^max ≈ 2.8 %·m·T/MA
    for a conventional wall-stabilised configuration; advanced scenarios with
    resistive wall modes can reach β_N ~ 4–5 %.

    Parameters
    ----------
    beta : float
        Total-field beta, dimensionless.
    a : float
        Plasma minor radius [m].
    B0 : float
        On-axis toroidal magnetic field [T].
    Ip_MA : float
        Plasma current [MA].

    Returns
    -------
    beta_N : float
        Normalised beta [% m T MA⁻¹].

    References
    ----------
    Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    Wesson (2011), §6.7.
    """
    beta_N = beta * a * B0 / Ip_MA * 100.0
    return beta_N


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # ITER Q=10 reference — Shimada et al., Nucl. Fusion 47 (2007) S1
    # Beta functions only; p_bar taken as given design values.
    # ------------------------------------------------------------------
    R0_IT  = 6.2;  a_IT  = 2.0;  kap_IT = 1.85;  Ip_IT = 15.0   # MA
    Bmax   = 11.8  # Peak TF inboard field [T]
    B0_IT  = 5.3   # On-axis field [T] — Shimada 2007, Table 1
    b_IT   = R0_IT * (1.0 - B0_IT / Bmax) - a_IT   # Inboard radial build [m]

    # Pressure inputs — from §2 ITER console block (H-mode pedestal profile)
    p_ac = 0.260   # Academic mode [MPa]
    p_d0 = 0.262   # D0FUS mode    [MPa]

    B0   = f_B0(Bmax, a_IT, b_IT, R0_IT)
    res  = {}
    for label, p in [('Academic', p_ac), ('D0FUS', p_d0)]:
        bT = f_beta_T(p, B0)
        bP = f_beta_P(a_IT, kap_IT, p, Ip_IT)
        b  = f_beta(bT, bP)
        bN = f_beta_N(b, a_IT, B0, Ip_IT)
        res[label] = (bT, bP, b, bN)

    print(f"\n── ITER β  (B0={B0:.2f} T, b_inboard={b_IT:.3f} m) ────────────────────")
    print(f"  {'Quantity':<24} {'Academic':>10} {'D0FUS':>10} {'ITER ref.':>10}")
    print("  " + "─"*56)
    print(f"  {'beta_T  [%]':<24} {res['Academic'][0]*100:>10.3f} {res['D0FUS'][0]*100:>10.3f} {'2.50':>10}")
    print(f"  {'beta_P  [-]':<24} {res['Academic'][1]:>10.3f} {res['D0FUS'][1]:>10.3f} {'0.65':>10}")
    print(f"  {'beta  [%]':<24} {res['Academic'][2]*100:>10.3f} {res['D0FUS'][2]*100:>10.3f} {'2.42':>10}")
    print(f"  {'beta_N  [% m T/MA]':<24} {res['Academic'][3]:>10.3f} {res['D0FUS'][3]:>10.3f} {'1.77':>10}")
    print(f"  {'Troyon limit (< 2.8)':<24} "
          f"{'OK' if res['Academic'][3]<2.8 else 'WARN':>10} "
          f"{'OK' if res['D0FUS'][3]<2.8 else 'WARN':>10}")


#%% Power definitions

def f_P_alpha(P_fus):
    """
    Alpha-particle heating power from DT fusion.

    In a DT reaction, the total energy release Q_DT = 17.58 MeV is partitioned
    between the alpha particle (E_α = 3.52 MeV, charged, confined) and the
    neutron (E_n = 14.06 MeV, escaping to the blanket).  Only P_α is deposited
    in the plasma; the neutron power is recovered in the blanket for thermal
    conversion.

    Uses module-level constants E_ALPHA and E_N [J].

    Parameters
    ----------
    P_fus : float
        Total fusion power [MW].

    Returns
    -------
    float
        Alpha-particle heating power [MW].  P_α / P_fus ≈ 0.2 for DT.

    References
    ----------
    Wesson, Tokamaks, 4th ed. (2011), §1.8.
    """
    return P_fus * E_ALPHA / (E_ALPHA + E_N)


def f_P_Ohm(I_Ohm, Tbar, R0, a, kappa, Z_eff=1.0,
            nbar=1.0, eta_model='spitzer', q95=3.0):
    """
    Ohmic (resistive) heating power from 0D volume-averaged quantities.

    Ohmic heating arises from resistive dissipation of the inductive plasma
    current.  The Spitzer resistivity η ∝ Z_eff T_e^{-3/2} renders this term
    negligible at reactor-grade temperatures (T_e > 5 keV), but it contributes
    during the current ramp-up phase and in lower-temperature auxiliary-heated
    scenarios.

    Two-step 0D calculation:
      1. Resistivity η [Ω m] from the selected model (see eta_model), evaluated
         at volume-averaged T and n.  For neoclassical models (sauter, redl),
         the inverse-aspect-ratio is evaluated at a current-centroid estimate
         ρ_eff ≈ 0.5, i.e. ε_eff = 0.5 a / R₀.
      2. Effective toroidal resistance, approximating the plasma as a straight
         conductor of length 2πR₀ and cross-section πa²κ:
            R_eff = η × 2R₀ / (a² κ)   [Ω]
         Joule dissipation:
            P_Ohm = R_eff × I_Ohm²       [W → MW]

    For a profile-integrated estimate (radially resolved j(ρ), T(ρ)), use
    f_Reff × I_Ohm² instead.

    Parameters
    ----------
    I_Ohm : float
        Inductive (Ohmic) plasma current component [MA].
    Tbar : float
        Volume-averaged electron temperature [keV].
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    kappa : float
        Plasma elongation [-].
    Z_eff : float, optional
        Effective plasma ionic charge [-] (default 1.0).
    nbar : float, optional
        Volume-averaged electron density [10^20 m^-3] (default 1.0).
        Needed for Coulomb logarithm and collisionality in all models
        except 'old'.
    eta_model : str, optional
        Resistivity model selection (default 'spitzer'):
        - 'old'     : Wesson 2.8e-8 / T^1.5, no Z_eff, no lnΛ dependence.
        - 'spitzer' : Spitzer-Härm with proper lnΛ and g(Z) correction.
        - 'sauter'  : Neoclassical (Sauter 1999), trapped-particle correction.
        - 'redl'    : Neoclassical (Redl 2021), improved Sauter fit.
    q95 : float, optional
        Safety factor at 95%% flux surface [-] (default 3.0).
        Only used by neoclassical models ('sauter', 'redl') for
        collisionality evaluation.

    Returns
    -------
    float
        Ohmic heating power [MW].

    References
    ----------
    Wesson (2011), §14.1 — base Spitzer coefficient.
    Spitzer & Härm, Phys. Rev. 89 (1953) 977 — classical resistivity.
    Sauter O. et al., Phys. Plasmas 6 (1999) 2834 — neoclassical model.
    Redl A. et al., Phys. Plasmas 28 (2021) 022502 — improved neoclassical.
    NRL Plasma Formulary (2022), p. 34 — Spitzer resistivity overview.
    """
    ne = nbar * 1e20                              # [m^-3]

    # --- Resistivity dispatch (same models as f_Reff) ---
    if eta_model == 'old':
        eta = eta_old(Tbar, ne, Z_eff)
    elif eta_model == 'spitzer':
        eta = eta_spitzer(Tbar, ne, Z_eff)
    elif eta_model in ('sauter', 'redl'):
        # Current-centroid estimate: rho_eff ~ 0.5 for peaked j(rho)
        epsilon_eff = 0.5 * a / R0
        if eta_model == 'sauter':
            eta = eta_sauter(Tbar, ne, Z_eff, epsilon_eff, q95, R0)
        else:
            eta = eta_redl(Tbar, ne, Z_eff, epsilon_eff, q95, R0)
    else:
        raise ValueError(f"Unknown eta_model '{eta_model}'. "
                         f"Choose from: 'old', 'spitzer', 'sauter', 'redl'.")

    R_eff = eta * 2.0 * R0 / (a**2 * kappa)      # Effective toroidal resistance [Ω]
    return R_eff * (I_Ohm * 1e6)**2 * 1e-6        # [W] → [MW]


def f_P_elec(P_fus, P_CD, eta_T, M_blanket=1.0, eta_WP=1.0):
    """
    Net electrical output power — simplified thermodynamic model.

        P_elec = η_th × M_blanket × P_fus − P_CD / η_WP

    The recirculating power subtracted is the **wall-plug** power consumed by
    the heating and current-drive systems, not the plasma-absorbed power P_CD.
    Converting plasma power to wall-plug power requires the wall-plug
    efficiency η_WP (overall injector + transmission losses):

        P_wallplug = P_CD / η_WP

    Blanket energy multiplication M_blanket ~ 1.1–1.3 arises from exothermic
    reactions of 14 MeV neutrons with ⁶Li in the tritium-breeding blanket
    (⁶Li + n → α + T + 4.8 MeV), amplifying the thermal power entering the
    steam cycle relative to the bare DT fusion power.

    Parameters
    ----------
    P_fus : float
        Total DT fusion power [MW].
    P_CD : float
        Total plasma-absorbed auxiliary heating and current-drive power [MW].
    eta_T : float
        Thermal-to-electric conversion efficiency [-] (typically 0.35–0.45).
    M_blanket : float, optional
        Blanket energy multiplication factor [-] (default 1.0 → no credit).
        Typical range: 1.1–1.3 for a Li-bearing breeding blanket.
    eta_WP : float, optional
        Wall-plug efficiency of heating/CD systems [-] (default 1.0 → no loss).
        P_wallplug = P_CD / η_WP.
        Typical values (single effective η_WP for the heating mix):
          LH  klystron  : 0.50–0.60
          EC  gyrotron  : 0.40–0.55  (lower: gyrotron efficiency ~50 %)
          NBI injector  : 0.25–0.40  (neutraliser losses are large)
          ICR amplifier : 0.70–0.85  (solid-state, best coupling)

    Returns
    -------
    float
        Net electrical power [MW].  Negative → energy-negative operating point.

    Notes
    -----
    Full power balance:
        P_gross  = η_th × M_blanket × P_fus     [gross electrical output, MW]
        P_recirc = P_CD / η_WP                   [wall-plug CD/heating power, MW]
        P_elec   = P_gross − P_recirc

    Recirculating power fraction:
        f_recirc = P_recirc / P_gross
                 = P_CD / (η_WP × η_th × M_blanket × P_fus)
                 = 1 / (η_WP × Q × η_th × M_blanket)
    At DEMO Q=10, η_th=0.40, η_WP=0.40, M_blanket=1.15:
        f_recirc ≈ 54 %  — motivates steady-state scenarios and HTS to reduce P_CD.

    References
    ----------
    Freidberg, PoP 22 (2015) 070901.
    Kovari et al., Fusion Eng. Des. 89 (2014) 3054 — PROCESS model.
    Hernandez et al., Nucl. Fusion 57 (2017) 016011 — HCPB M_blanket values.
    Gormezano et al. (ITER PIPB Ch. 6), Nucl. Fusion 47 (2007) S285 — η_WP values.
    """
    if eta_WP <= 0.0:
        raise ValueError(f"f_P_elec: eta_WP must be > 0 (got {eta_WP}).")
    return eta_T * M_blanket * P_fus - P_CD / eta_WP


#%% Radiation losses

def f_P_synchrotron(Tbar, R0, a, B0, nbar, kappa, nu_n, nu_T, r_synch,
                    rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                    Vprime_data=None):
    """
    Total synchrotron (electron-cyclotron) radiation power — Albajar et al. (2001).

    Gyrating relativistic electrons emit cyclotron radiation at harmonics of the
    electron-cyclotron frequency.  At reactor temperatures (T_e > 5 keV), the
    relativistic correction and the plasma opacity become important.  The Albajar
    (2001) formula accounts for:

      - Partial opacity via the parameter p_{a0} (Eq. 7):
            p_{a0} = 6.04 × 10³ × a × n_{e0} / B₀
      - Profile shape via the factor K (Eq. 13), which depends on (ν_n, ν_T)
        and was derived for purely parabolic profiles; it remains a reasonable
        approximation when the core dominates emission.
      - Toroidal geometry via the aspect-ratio factor G (Eq. 15):
            G = 0.93 (1 + 0.85 exp(−0.82 A)),  A = R₀/a
      - Partial wall reflection by (1 − r_synch)^{0.5}; metallic walls with
        r_synch ~ 0.8–0.9 significantly reduce the net radiated power.

    Central on-axis values T₀ and n_{e0} are computed internally from the
    volume-averaged inputs via f_Tprof / f_nprof at ρ = 0, making the function
    compatible with both Academic (parabolic) and D0FUS (pedestal) profile modes.

    Parameters
    ----------
    Tbar : float
        Volume-averaged electron temperature [keV].
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    B0 : float
        On-axis toroidal magnetic field [T].
    nbar : float
        Volume-averaged electron density [10²⁰ m⁻³].
    kappa : float
        Plasma vertical elongation.
    nu_n, nu_T : float
        Density and temperature profile peaking exponents.
    r_synch : float
        First-wall reflectivity for synchrotron photons [0, 1).
        Matches ``GlobalConfig.r_synch``.
        Typical values: 0.5 (carbon/graphite wall), 0.8–0.9 (metallic wall).
    rho_ped, n_ped_frac, T_ped_frac : float
        Pedestal parameters forwarded to f_Tprof / f_nprof.
        Default values correspond to purely parabolic profiles.

    Returns
    -------
    float
        Total synchrotron radiation power [MW].

    References
    ----------
    Albajar et al., Nucl. Fusion 41 (2001) 665.
    Johner, CEA/IRFM internal report NT DSM/NTT-2011-3440 (2011).
    """
    import math
    T0  = float(f_Tprof(Tbar, nu_T, 0.0, rho_ped, T_ped_frac,
                        Vprime_data))                              # on-axis T [keV]
    ne0 = float(f_nprof(nbar,  nu_n,  0.0, rho_ped, n_ped_frac,
                        Vprime_data))                              # on-axis n [1e20 m-3]
    A   = R0 / a

    pa0 = 6.04e3 * a * ne0 / B0                                    # opacity parameter (Eq. 7)

    # Profile factor K (Albajar 2001, Eq. 13) — valid for ν_T ≥ 0.5.
    # The denominator (ν_T^1.53 + 1.87ν_T - 0.16) vanishes at ν_T ≈ 0.075
    # and becomes negative below, producing complex/NaN values.
    # Clamp to ν_T = 0.1 to avoid silent failures in design scans.
    nu_T_K = max(nu_T, 0.1)
    if nu_T < 0.1:
        warnings.warn(
            f"f_P_synchrotron: nu_T = {nu_T:.3f} < 0.1; clamped to 0.1 "
            "for the Albajar K-factor (valid for nu_T >= 0.5).",
            RuntimeWarning, stacklevel=2
        )

    K = ((nu_n + 3.87*nu_T_K + 1.46)**(-0.79)                      # profile factor (Eq. 13)
         * (1.98 + nu_T_K)**1.36 * nu_T_K**2.14
         / (nu_T_K**1.53 + 1.87*nu_T_K - 0.16)**1.33)

    G = 0.93 * (1.0 + 0.85 * math.exp(-0.82 * A))                  # geometry factor (Eq. 15)

    return (3.84e-8 * (1.0 - r_synch)**0.5                         # main expression (Eq. 16)
            * R0 * a**1.38 * kappa**0.79 * B0**2.62 * ne0**0.38
            * T0 * (16.0 + T0)**2.61
            * (1.0 + 0.12 * T0 / pa0**0.41)**(-1.51)
            * K * G)


def f_P_bremsstrahlung(nbar, Tbar, Z_eff, V, nu_n=0.0, nu_T=0.0,
                       rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                       Vprime_data=None):
    """
    Volume-integrated Bremsstrahlung (free-free) radiation power.

    Bremsstrahlung emission from electron-ion collisions scales as
    n_e² Z_eff T_e^{1/2}.  The 0D volume integral gives:

        P_brem = C_B × Z_eff × ⟨n²T^{1/2}⟩_vol × V

    where C_B = 5.35 × 10³ W m³ keV^{-1/2} per (10²⁰ m⁻³)².

    Two modes:
      Academic (Vprime_data = None):
        Analytical profile-peaking correction f_peak derived for
        purely parabolic profiles with cylindrical weight 2ρ dρ.
      D0FUS (Vprime_data provided):
        Numerical integration with Miller-normalised profiles and
        V'(ρ) weight, consistent with all other D0FUS integrals.

    Parameters
    ----------
    nbar : float
        Volume-averaged electron density [10²⁰ m⁻³].
    Tbar : float
        Volume-averaged electron temperature [keV].
    Z_eff : float
        Effective ion charge number.
    V : float
        Plasma volume [m³].  Pass V_ac (Academic) or V_d0 (D0FUS).
    nu_n : float, optional
        Density peaking exponent (default 0 → uniform).
    nu_T : float, optional
        Temperature peaking exponent (default 0 → uniform).
    rho_ped : float, optional
        Normalised pedestal radius (default 1.0 → parabolic).
    n_ped_frac : float, optional
        n_ped / n̄ (default 0.0).
    T_ped_frac : float, optional
        T_ped / T̄ (default 0.0).
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → Academic mode (analytical f_peak).

    Returns
    -------
    float
        Bremsstrahlung power [MW].

    Notes
    -----
    Analytical parabolic profile correction (cylindrical weight 2ρ dρ):
        ⟨n̂² T̂^{1/2}⟩ = ∫₀¹ (1+ν_n)²(1−ρ²)^{2ν_n} · (1+ν_T)^{1/2}(1−ρ²)^{ν_T/2} 2ρ dρ
                       = (1+ν_n)²(1+ν_T)^{1/2} / (1 + 2ν_n + ν_T/2)
        f_peak = ⟨n̂² T̂^{1/2}⟩   (normalised by definition since n̄=1, T̄=1)

    References
    ----------
    NRL Plasma Formulary (2022), p. 58.
    Wesson (2011), §3.3.
    """
    C_B = 5.35e3   # [W m³ keV^{-1/2} (10²⁰ m⁻³)⁻²]

    if Vprime_data is not None:
        # D0FUS mode: numerical ∫ n̂²(ρ) T̂^{1/2}(ρ) V'(ρ) dρ / V
        rho_grid, Vprime, V_total = Vprime_data
        n_hat = f_nprof(1.0, nu_n, rho_grid, rho_ped, n_ped_frac,
                        Vprime_data)
        T_hat = f_Tprof(1.0, nu_T, rho_grid, rho_ped, T_ped_frac,
                        Vprime_data)
        integrand = n_hat**2 * np.maximum(T_hat, 0.0)**0.5 * Vprime
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0)
        f_peak = float(np.trapezoid(integrand, rho_grid)) / V_total
    else:
        # Academic mode: analytical parabolic peaking factor
        # Reduces to 1 when nu_n = nu_T = 0 (uniform profile limit)
        f_peak = ((1.0 + nu_n)**2 * (1.0 + nu_T)**0.5
                  / (1.0 + 2.0*nu_n + 0.5*nu_T))

    return C_B * Z_eff * nbar**2 * Tbar**0.5 * V * f_peak * 1e-6


def f_P_line_radiation(nbar, f_imp, L_z, V):
    """
    Volume-integrated impurity line radiation power.

    Bound-bound (line) transitions and radiative recombination (free-bound)
    from a single impurity species are captured by the radiative cooling
    coefficient L_z(T_e), obtainable from `get_Lz`.  The 0D estimate is:

        P_line = n_e² × f_imp × L_z × V

    where f_imp = n_imp / n_e is the impurity concentration relative to electrons.

    The plasma volume V links this function to the geometry mode (same as
    f_P_bremsstrahlung).

    Caution: this is a 0D bulk-plasma estimate.  For high-Z impurities such as
    W, the bulk of line radiation originates in the SOL/divertor at low T_e; the
    result should be interpreted as an upper bound on the core contribution.

    Parameters
    ----------
    nbar : float
        Volume-averaged electron density [10²⁰ m⁻³].
    f_imp : float
        Impurity concentration n_imp / n_e [-].
    L_z : float
        Radiative cooling coefficient [W m³] at T̄_e (from `get_Lz`).
    V : float
        Plasma volume [m³].  Pass V_ac (Academic) or V_d0 (D0FUS).

    Returns
    -------
    float
        Line radiation power [MW].

    References
    ----------
    Pütterich et al., Nucl. Fusion 50 (2010) 025012.
    Summers et al., ADAS database (http://adas.ac.uk).
    """
    ne_SI = nbar * 1e20   # [10²⁰ m⁻³] → [m⁻³]
    return ne_SI**2 * f_imp * L_z * V * 1e-6


def f_P_line_radiation_profile(impurity, f_imp, nbar, Tbar, nu_n, nu_T, V,
                                rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                                Vprime_data=None, N=500, rho_core=None):
    """
    Profile-integrated impurity line radiation power [MW].

    Integrates the local emissivity ne²(ρ) × Lz(T(ρ)) over the full plasma
    volume, from the magnetic axis (ρ=0) to the separatrix (ρ=1).

    This resolves the strong temperature dependence of the radiative cooling
    coefficient Lz(T) across the radial profile, which is essential for
    medium-Z seeding species (Ne, Ar) whose Lz peaks at T ~ 0.1–2 keV
    (pedestal/edge region).

    The D0FUS profile model extends from ρ=0 (T=T0) to ρ=1 (T=0, n=0),
    with a linear SOL ramp between ρ_ped and 1.  The edge region captures
    the cold plasma layer where ne is still finite and Lz rises sharply.

    Core / edge split (rho_core parameter)
    ---------------------------------------
    When rho_core is provided, the integral is split at ρ = rho_core:

        P_line_core  = f_imp × ∫₀^{ρ_core} ne² × Lz × w dρ   (confined core)
        P_line_edge  = f_imp × ∫_{ρ_core}^1  ne² × Lz × w dρ   (edge/SOL)
        P_line_total = P_line_core + P_line_edge

    This distinction matters for the energy confinement time:
    - P_line_core should be subtracted from P_heat in P_loss = P_heat - P_rad
      (this power was radiated from the confined plasma and never crossed the
      separatrix by transport — it reduces τ_E).
    - P_line_edge was first transported across the confinement boundary, then
      radiated in the edge/SOL.  It does NOT affect τ_E but does reduce P_sep
      (power reaching the divertor).

    For Bremsstrahlung and synchrotron radiation (volumetric, ~ n²T^{1/2}
    and ~ nT^{3.5} respectively), the core contribution dominates (> 95%
    for typical reactor profiles), so the full integral is used as P_rad_core.

    Integral:
        P_line = f_imp × ∫₀¹ ne²(ρ) × Lz(T(ρ)) × w(ρ) dρ   [W]

    Volume weights:
      Academic : w(ρ) = V × 2ρ           (cylindrical element)
      D0FUS    : w(ρ) = V'_Miller(ρ)     (flux-surface Jacobian)

    Parameters
    ----------
    impurity    : str    Species symbol ('W', 'Ne', 'Ar', etc.)
    f_imp       : float  Impurity concentration n_imp / n_e [-]
    nbar        : float  Volume-averaged electron density [10²⁰ m⁻³]
    Tbar        : float  Volume-averaged electron temperature [keV]
    nu_n, nu_T  : float  Density and temperature peaking exponents
    V           : float  Plasma volume [m³] (Academic mode; ignored if Vprime_data)
    rho_ped     : float  Normalised pedestal radius (1.0 → parabolic)
    n_ped_frac  : float  n_ped / n̄ [-]
    T_ped_frac  : float  T_ped / T̄ [-]
    Vprime_data : tuple or None  Precomputed (rho_grid, Vprime, V_total)
    N           : int    Radial integration points (default 500)
    rho_core    : float or None
        Boundary between core and edge radiation regions.  When None
        (default), only the total is returned for backward compatibility.
        Typical value: 0.7 (inside pedestal top for ITER/DEMO-class shaping).

    Returns
    -------
    float or tuple
        If rho_core is None : P_line_total [MW]  (backward compatible).
        If rho_core is set  : (P_line_core, P_line_total) [MW, MW].
    """
    if f_imp <= 0.0:
        return (0.0, 0.0) if rho_core is not None else 0.0

    ne_SI = nbar * 1e20   # Volume-averaged density [m⁻³]

    if Vprime_data is not None:
        # D0FUS mode: Miller V'(ρ) integration
        rho_grid, Vprime, V_total = Vprime_data
        n_hat  = f_nprof(1.0,  nu_n, rho_grid, rho_ped, n_ped_frac,
                         Vprime_data)
        T_arr  = f_Tprof(Tbar, nu_T, rho_grid, rho_ped, T_ped_frac,
                         Vprime_data)
        T_clamped = np.clip(T_arr, 0.01, None)
        Lz_arr = get_Lz(impurity, T_clamped)
        integrand = ne_SI**2 * n_hat**2 * Lz_arr * Vprime
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0)
        P_total_W = f_imp * float(np.trapezoid(integrand, rho_grid))

        if rho_core is not None:
            # Core contribution: integrate from 0 to rho_core only
            mask_core = rho_grid <= rho_core
            if np.any(mask_core) and np.sum(mask_core) >= 2:
                P_core_W = f_imp * float(
                    np.trapezoid(integrand[mask_core], rho_grid[mask_core]))
            else:
                P_core_W = 0.0
    else:
        # Academic mode: cylindrical weight 2ρ dρ
        rho_arr = np.linspace(1e-6, 1.0, N)
        n_hat  = f_nprof(1.0,  nu_n, rho_arr, rho_ped, n_ped_frac)
        T_arr  = f_Tprof(Tbar, nu_T, rho_arr, rho_ped, T_ped_frac)
        T_clamped = np.clip(T_arr, 0.01, None)
        Lz_arr = get_Lz(impurity, T_clamped)
        integrand = ne_SI**2 * n_hat**2 * Lz_arr * 2.0 * rho_arr
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0)
        P_total_W = f_imp * V * float(np.trapezoid(integrand, rho_arr))

        if rho_core is not None:
            mask_core = rho_arr <= rho_core
            if np.any(mask_core) and np.sum(mask_core) >= 2:
                P_core_W = f_imp * V * float(
                    np.trapezoid(integrand[mask_core], rho_arr[mask_core]))
            else:
                P_core_W = 0.0

    if rho_core is not None:
        return P_core_W * 1e-6, P_total_W * 1e-6   # (P_core, P_total) [MW]
    return P_total_W * 1e-6   # P_total [MW]




# ══════════════════════════════════════════════════════════════════════════════
# Lz lookup tables — precomputed at module load from Mavrin (2018) polynomials
# ══════════════════════════════════════════════════════════════════════════════
#
# Each species has a pair (_LOG10_TE_LZ[sp], _LOG10_LZ[sp]) of 1-D arrays
# evaluated on a uniform log10(Te [keV]) grid covering [0.01, 100] keV.
# At runtime, get_Lz does a single vectorised np.interp in log-log space.
#
# Accuracy: linear interpolation on 300 log-spaced points gives < 0.5%
# relative error vs the original piecewise polynomials — negligible compared
# to the ~factor-2 uncertainty in the underlying ADAS atomic data.
# ══════════════════════════════════════════════════════════════════════════════

# Mavrin 2018 polynomial coefficients (frozen reference data)
_MAVRIN_LZ_COEFFS = {
    "He": [
        (0.1, 100, [-3.5551e+01,  3.1469e-01,  1.0156e-01, -9.3730e-02,  2.5020e-02]),
    ],
    "Li": [
        (0.1, 100, [-3.5115e+01,  1.9475e-01,  2.5082e-01, -1.6070e-01,  3.5190e-02]),
    ],
    "Be": [
        (0.1, 100, [-3.4765e+01,  3.7270e-02,  3.8363e-01, -2.1384e-01,  4.1690e-02]),
    ],
    "C": [
        (0.1, 0.5, [-3.4738e+01, -5.0085e+00, -1.2788e+01, -1.6637e+01, -7.2904e+00]),
        (0.5, 100, [-3.4174e+01, -3.6687e-01,  6.8856e-01, -2.9191e-01,  4.4470e-02]),
    ],
    "N": [
        (0.1, 0.5, [-3.4065e+01, -2.3614e+00, -6.0605e+00, -1.1570e+01, -6.9621e+00]),
        (0.5, 2.0, [-3.3899e+01, -5.9668e-01,  7.6272e-01, -1.7160e-01,  5.8770e-02]),
        (2.0, 100, [-3.3913e+01, -5.2628e-01,  7.0047e-01, -2.2790e-01,  2.8350e-02]),
    ],
    "O": [
        (0.1, 0.3, [-3.7257e+01, -1.5635e+01, -1.7141e+01, -5.3765e+00,  0.0000e+00]),
        (0.3, 100, [-3.3640e+01, -7.6211e-01,  7.9655e-01, -2.0850e-01,  1.4360e-02]),
    ],
    "Ne": [
        (0.1, 0.7, [-3.3132e+01,  1.7309e+00,  1.5230e+01,  2.8939e+01,  1.5648e+01]),
        (0.7, 5.0, [-3.3290e+01, -8.7750e-01,  8.6842e-01, -3.9544e-01,  1.7244e-01]),
        (5.0, 100, [-3.3410e+01, -4.5345e-01,  2.9731e-01,  4.3960e-02, -2.6930e-02]),
    ],
    "Ar": [
        (0.1, 0.6, [-3.2155e+01,  6.5221e+00,  3.0769e+01,  3.9161e+01,  1.5353e+01]),
        (0.6, 3.0, [-3.2530e+01,  5.4490e-01,  1.5389e+00, -7.6887e+00,  4.9806e+00]),
        (3.0, 100, [-3.1853e+01, -1.6674e+00,  6.1339e-01,  1.7480e-01, -8.2260e-02]),
    ],
    "Kr": [
        (0.1, 0.447,   [-3.4512e+01, -2.1484e+01, -4.4723e+01, -4.0133e+01, -1.3564e+01]),
        (0.447, 2.364, [-3.1399e+01, -5.0091e-01,  1.9148e+00, -2.5865e+00, -5.2704e+00]),
        (2.364, 100,   [-2.9954e+01, -6.3683e+00,  6.6831e+00, -2.9674e+00,  4.8356e-01]),
    ],
    "Xe": [
        (0.1, 0.5,  [-2.9303e+01,  1.4351e+01,  4.7081e+01,  5.9580e+01,  2.5615e+01]),
        (0.5, 2.5,  [-3.1113e+01,  5.9339e-01,  1.2808e+00, -1.1628e+01,  1.0748e+01]),
        (2.5, 10.0, [-2.5813e+01, -2.7526e+01,  4.8614e+01, -3.6885e+01,  1.0069e+01]),
        (10., 100,  [-2.2138e+01, -2.2592e+01,  1.9619e+01, -7.5181e+00,  1.0858e+00]),
    ],
    "W": [
        (0.1, 1.5, [-3.0374e+01,  3.8304e-01, -9.5126e-01, -1.0311e+00, -1.0103e-01]),
        (1.5, 4.0, [-3.0238e+01, -2.9208e+00,  2.2824e+01, -6.3303e+01,  5.1849e+01]),
        (4.0, 100, [-3.2153e+01,  5.2499e+00, -6.2740e+00,  2.6627e+00, -3.6759e-01]),
    ],
}

def _build_Lz_lookup(n_pts=300):
    """Build log-log lookup tables for all species at module load time.

    The Mavrin (2018) piecewise polynomials are valid for Te ∈ [0.1, 100] keV.
    Below 0.1 keV, the 4th-order polynomials diverge catastrophically
    (e.g. Lz(Ne, 0.01 keV) ≈ 10⁴³ instead of ~10⁻³²).
    The grid extends to 0.01 keV for robustness, but values below 0.1 keV
    are clamped to the Lz(0.1 keV) boundary value of each species.
    """
    log10_Te = np.linspace(np.log10(0.01), np.log10(100.0), n_pts)
    Te_arr   = 10.0 ** log10_Te
    tables = {}
    for sp, segments in _MAVRIN_LZ_COEFFS.items():
        log_Lz = np.empty(n_pts)
        for i, Te in enumerate(Te_arr):
            # Clamp to Mavrin validity lower bound (0.1 keV)
            Te_eval = max(Te, 0.1)
            X = np.log10(Te_eval)
            coeffs = segments[-1][2]
            for Tmin, Tmax, seg_coeffs in segments:
                if Te_eval <= Tmax:
                    coeffs = seg_coeffs
                    break
            A0, A1, A2, A3, A4 = coeffs
            log_Lz[i] = A0 + A1*X + A2*X**2 + A3*X**3 + A4*X**4
        tables[sp] = (log10_Te, log_Lz)
    return tables

_LZ_TABLES = _build_Lz_lookup()

# Species name mapping (built once)
_IMP_NAME_MAP = {
    "he": "He", "helium": "He", "li": "Li", "lithium": "Li",
    "be": "Be", "beryllium": "Be", "c": "C", "carbon": "C",
    "n": "N", "nitrogen": "N", "o": "O", "oxygen": "O",
    "ne": "Ne", "neon": "Ne", "ar": "Ar", "argon": "Ar",
    "kr": "Kr", "krypton": "Kr", "xe": "Xe", "xenon": "Xe",
    "w": "W", "tungsten": "W",
}


def get_Lz(impurity, Te_keV):
    """
    Radiative cooling coefficient Lz(Te) for common tokamak impurities.

    Fast lookup-table implementation: linear interpolation in log10-log10
    space on a precomputed 300-point grid.  Accuracy < 0.5% relative error
    vs the original Mavrin (2018) piecewise polynomials — negligible compared
    to the ~factor-2 uncertainty in the underlying ADAS atomic data.

    Parameters
    ----------
    impurity : str
        Impurity symbol or full name ('He', ..., 'W', 'neon', 'tungsten', ...).
    Te_keV : float or array-like
        Electron temperature [keV].

    Returns
    -------
    float or ndarray
        Radiative cooling coefficient [W m³].

    References
    ----------
    Mavrin, A.A., "Improved fits of coronal radiative cooling rates for
    high-temperature plasmas", Radiation Effects and Defects in Solids,
    vol. 173, no. 5-6, pp. 388-398 (2018).
    DOI: 10.1080/10420150.2018.1462361
    Pütterich et al., Nucl. Fusion 50 (2010) 025012.
    """
    imp = _IMP_NAME_MAP.get(impurity.strip().lower(), impurity.strip().upper())
    if imp not in _LZ_TABLES:
        raise ValueError(
            f"Impurity '{impurity}' not supported. "
            f"Available: {sorted(_LZ_TABLES.keys())}")

    log10_Te_grid, log10_Lz_grid = _LZ_TABLES[imp]

    Te_arr = np.atleast_1d(np.asarray(Te_keV, dtype=float))
    # Clamp to Mavrin polynomial validity range [0.1, 100] keV
    log10_Te = np.log10(np.clip(Te_arr, 0.1, 100.0))

    # Vectorised linear interpolation in log-log space (C-level, no Python loop)
    log_Lz = np.interp(log10_Te, log10_Te_grid, log10_Lz_grid)
    Lz = 10.0 ** log_Lz

    return float(Lz[0]) if Lz.size == 1 else Lz

if __name__ == "__main__":
    # Coronal radiative cooling coefficient L_z(T_e) for main impurities
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_Lz_cooling()

if __name__ == "__main__":
    # ITER Q=10 radiation budget — Shimada et al., NF 47 (2007) S1
    R0, a, kap, B0, nbar, Z_eff, r_synch = 6.2, 2.0, 1.85, 5.3, 1.01, 1.6, 0.4
    V = 2.0 * np.pi**2 * R0 * kap * a**2
    T = 8.9  # keV

    P_s  = f_P_synchrotron(T, R0, a, B0, nbar, kap, nu_n=0.1, nu_T=1.0, r_synch=r_synch)
    P_b  = f_P_bremsstrahlung(nbar, T, Z_eff, V)
    P_lW = f_P_line_radiation(nbar, 1e-5, get_Lz('W',  T), V)
    P_lAr = f_P_line_radiation(nbar, 1e-3, get_Lz('Ar', T), V)
    P_lNe = f_P_line_radiation(nbar, 3e-3, get_Lz('Ne', T), V)
    P_tot = P_s + P_b + P_lW

    print(f"\n── ITER radiation budget (academic, parabolic) {'─'*30}")
    print(f"  {'Channel':<32} {'[MW]':>7}  {'Ref.':>10}")
    print("  " + "─" * 53)
    print(f"  {'Bremsstrahlung (Z_eff=1.6)':<32} {P_b:>7.1f}  {'20–25':>10}")
    print(f"  {'Synchrotron (r_synch=0.4)':<32} {P_s:>7.1f}  {'3–6':>10}")
    print(f"  {'W line (f_W=1e-5)':<32} {P_lW:>7.1f}  {'2–4':>10}")
    print(f"  {'Ar line (f_Ar=1e-3)':<32} {P_lAr:>7.1f}  {'—':>10}")
    print(f"  {'Ne line (f_Ne=3e-3)':<32} {P_lNe:>7.1f}  {'—':>10}")
    print("  " + "─" * 53)
    print(f"  {'Total (brem+sync+W)':<32} {P_tot:>7.1f}  {'~30':>10}")


#%% Current drive
"""
Current Drive — non-inductive figure of merit γ [MA/(MW m²)].

Definition
----------
    γ = I_CD [MA] × R₀ [m] × n̄_e [10²⁰ m⁻³] / P_CD [MW]

equivalently η_CD [10²⁰ A m⁻² W⁻¹]  (Fisch 1987 convention).

LHCD models
-----------
Two models are implemented, accessible via f_etaCD_LH(model=...):

    model='fenstermacher'  (simple alternative)
      Empirical power law (Fenstermacher 1988), fit to multi-machine data.
      No free parameters; no Z_eff, β, or profile dependence.
      Equivalent to PROCESS iefrf=1 (AEA FUS 172 baseline).
      Scope: parameter scans where LHCD is not the main physics driver.

  model='ehst'  (default)
      Analytical formula from Ehst & Karney (1991), quasilinear theory.
      Uses volume-averaged T̄_e and n̄_e, no profile assumptions.
      Single calibration factor C_LH (see below).

On the calibration factor C_LH
-------------------------------
The Ehst-Karney formula was derived assuming a single n_∥ equal to the
cold-wave accessibility limit at the plasma edge.  Real LH launchers emit
a broad n_∥ spectrum; components below the local accessibility limit can
propagate deeper and resonate with faster electrons, driving more current.
This spectral effect is not capturable in a 0D formula without ray-tracing.

The calibration factor C_LH quantifies this systematic correction for a
given machine and launcher design.  It is NOT a free parameter — it is
derived from ray-tracing studies on a case-by-case basis:

  ITER 5 GHz system (Gormezano et al., NF 47 (2007) S285):
      γ_LH^raytracing ≈ 0.24 MA/(MW m²)
      γ_LH^Ehst-Karney bare ≈ 0.158 MA/(MW m²)   [at T̄_e=8.9 keV, n̄=10²⁰]
      C_LH = 0.24 / 0.158 = 1.52  (default)

  For other machines or launchers, C_LH must be re-derived from
  ray-tracing (GENRAY, TORAY, C3PO/RAYCON).  The absence of such a
  reference should prompt use of C_LH = 1.0 (conservative lower bound).

ECCD and NBCD models
--------------------
- ECCD :  Ehst & Karney (1991) quasilinear formula, local T_e at ρ_EC
          with f_pass correction (Kim 1991).  Preferred over PROCESS
          Cohen/IPDG89 model for T_e > 5 keV.
- NBCD :  Cordey, Jones & Start (1979) base + Start & Cordey (1980) trapped-
          electron shielding; √(E_b/A_b) intermediate-energy scaling from
          Mikkelsen & Singer (1983), valid for E_b/E_c ~ 2–10.

Trapped-particle correction (Kim et al. 1991):
    f_trap(ρ) = 1.46 √ε (1 − 0.54 √ε),   ε = ρ a / R₀.

Pre-factor calibration summary
-------------------------------
C_LH  = 1.52  (ITER 5 GHz, Gormezano et al. 2007)
C_EC  = 0.68  (O-mode tangential equatorial, Ehst & Karney 1991 Table 2)
C_NBI = 0.37  (JT-60U 360 keV H-beam, Urano et al. 2002; conservative)

Expected ordering at ITER Q=10:
    γ_LH ≈ 0.24 > γ_EC ≈ 0.20 > γ_NBI ≈ 0.16 MA/(MW m²)
ITER PIPB projections: γ_LH~0.24, γ_EC~0.20, γ_NBI~0.20–0.40 MA/(MW m²).

References
----------
Fisch, Rev. Mod. Phys. 59 (1987) 175 — theoretical basis.
Fisch & Boozer, Phys. Rev. Lett. 45 (1980) 720 — current drive mechanism.
Karney & Fisch, Phys. Fluids 28 (1985) 116 — quasilinear solutions.
Ehst & Karney, Nucl. Fusion 31 (1991) 1933 — LHCD and ECCD formulas.
Cordey, Jones & Start, Nucl. Fusion 19 (1979) 249 — base NBCD formula.
Start & Cordey, Phys. Fluids 23 (1980) 1477 — trapped-electron G(ε,Z) for NBCD.
Stix, Plasma Phys. 14 (1972) 367 — critical energy E_c.
Mikkelsen & Singer, Nucl. Tech.-Fusion 4 (1983) 237 — √E_b intermediate scaling.
Cordey, Nucl. Fusion 26 (1986) 123 — trapped-particle effects on NBCD.
Kim et al., Phys. Fluids B 3 (1991) 2050 — trapped particle correction.
Giruzzi, Nucl. Fusion 27 (1987) 1934 — trapped-electron ECCD correction.
Reid et al., ORNL/FEDC-87-7 (1988) — Fenstermacher fit origin.
Hender et al., AEA FUS 172, UKAEA (1992) — PROCESS LHCD and ECCD models.
Urano et al. (JT-60U N-NBI), IAEA FEC 2002, paper EX8/3 — C_NBI calibration.
Oikawa et al. (ITPA NBCD benchmark), IAEA FEC 2008, IT/P3-33
    — validation of f_pass vs Start-Cordey G(ε).
Gormezano et al. (ITER PIPB Ch. 6), Nucl. Fusion 47 (2007) S285.
"""


def _ln_Lambda_CD(Te_keV, ne_20):
    """
    Coulomb logarithm for electron-electron collisions — NRL Formulary (2022).

    Piecewise formula in D0FUS units (T in keV, n in 10²⁰ m⁻³):

        T_e < 10 eV :  lnΛ = 23.0  − ln(√n_e [cm⁻³] / T_e [eV]^{3/2})
        T_e ≥ 10 eV :  lnΛ = 24.15 − ln(√n_e [cm⁻³] / T_e [eV])

    Floored at 5 to prevent unphysical values in cold-edge regions.
    In reactor-grade core plasmas: lnΛ ~ 17–20.

    Parameters
    ----------
    Te_keV : float  Local electron temperature [keV].
    ne_20  : float  Local electron density [10²⁰ m⁻³].

    Returns
    -------
    float  Coulomb logarithm (dimensionless, ≥ 5).

    References
    ----------
    NRL Plasma Formulary (2022), p. 34.
    """
    Te_eV  = Te_keV * 1e3
    ne_cm3 = ne_20  * 1e14    # [10²⁰ m⁻³] → [cm⁻³]
    if Te_eV < 10.0:
        lnL = 23.0  - np.log(ne_cm3**0.5 * Te_eV**(-1.5))
    else:
        lnL = 24.15 - np.log(ne_cm3**0.5 * Te_eV**(-1.0))
    return max(lnL, 5.0)


def _f_trap_CD(rho, a, R0):
    """
    Geometric trapped-particle fraction — Kim et al. (1991).

        f_trap(ρ) = 1.46 √ε_loc (1 − 0.54 √ε_loc)

    where ε_loc = ρ a / R₀ is the local inverse aspect ratio.
    Consistent with the Sauter (1999) and Redl (2021) bootstrap models.

    Parameters
    ----------
    rho : float  Normalised minor radius [0, 1].
    a   : float  Minor radius [m].
    R0  : float  Major radius [m].

    Returns
    -------
    float  Trapped fraction ∈ [0, 1).

    References
    ----------
    Kim et al., Phys. Fluids B 3 (1991) 2050.
    """
    eps_loc  = rho * a / R0
    sqrt_eps = np.sqrt(max(eps_loc, 0.0))
    return 1.46 * sqrt_eps * (1.0 - 0.54 * sqrt_eps)


def f_etaCD_LH_simple(Tbar, nbar, R0):
    """
    LHCD figure of merit — Fenstermacher empirical formula (1988).

    Pure empirical fit to multi-machine LH current drive data:

        γ_LH = 0.36 × (1 + (T̄_e / 25)^1.16) / (R₀ × n̄_e)

    The (T/25)^1.16 term interpolates between the non-relativistic limit
    (T_e ≪ 25 keV, where γ ∝ T_e^1.16) and the weakly-relativistic limit
    (T_e ≫ 25 keV, where the T_e dependence saturates).  The 1/n̄_e R₀
    scaling follows directly from the Fisch (1987) current drive formula.

    No Z_eff, β, or radial profile dependence — the formula was fit to
    volume-averaged quantities.  Equivalent to PROCESS iefrf = 1.

    Parameters
    ----------
    Tbar : float  Volume-averaged electron temperature [keV].
    nbar : float  Volume-averaged electron density [10²⁰ m⁻³].
    R0   : float  Major radius [m].

    Returns
    -------
    float  γ_CD^LH  [MA / (MW m²)].

    References
    ----------
    Reid et al., ORNL/FEDC-87-7, Oak Ridge National Laboratory (1988)
        — original Fenstermacher fit in the Oak Ridge Systems Code.
    Hender et al., AEA FUS 172, UKAEA (1992)
        — adopted in the PROCESS/AEA baseline as iefrf=1.
    """
    return 0.36 * (1.0 + (Tbar / 25.0)**1.16) / (R0 * nbar)


def f_etaCD_LH(a, R0, B0, nbar, Tbar, nu_n, nu_T,
               Z_eff=1.6, model='ehst',
               C_LH=1.52, beta_T=None,
               rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0):
    """
    LHCD figure of merit — model dispatcher.

    Routes to one of two implementations via ``model``:

    model = 'ehst'  [default]
        Ehst & Karney (1991) quasilinear formula, identical to PROCESS iefrf=4:

        γ_LH = C_LH × T̄_e^0.77 × (0.034 + 0.196 β_T) × G(Z_eff) / lnΛ

    where  G(Z) = 32/(5+Z) + 2 + 12(6+Z)/[(5+Z)(3+Z)] + 14/(3+Z)
    is the Spitzer parallel conductivity correction (Ehst & Karney, eq. 9).

    The T̄_e^0.77 exponent (< 1 compared to the pure Fisch T_e^1 limit)
    reflects the fact that LHCD sits in an intermediate velocity regime
    (u_∥ = v_∥/v_te ≈ 2–4) between the plateau and the Landau-damping
    limit; the exponent is a fit to the full quasilinear solutions of
    Karney & Fisch (1985, Phys. Fluids 28, 116).

    The (0.034 + 0.196 β_T) factor accounts for the contribution of
    finite β to the wave-electron interaction (finite-Larmor-radius
    correction in the quasilinear diffusion tensor).

    C_LH = 1.52  (default, ITER 5 GHz system):
        The Ehst-Karney formula assumes a single n_∥ fixed at the cold-
        wave accessibility limit.  ITER's multi-junction launcher emits a
        broad n_∥ spectrum; components with n_∥ < n_∥,acc(edge) ≈ 1.3
        propagate further inward before damping and drive current more
        efficiently.  C_LH captures this spectral effect:
            γ_ray-tracing / γ_Ehst-Karney = 0.24 / 0.158 = 1.52
        Source: Gormezano et al. ITER PIPB Ch.6, NF 47 (2007) S285.
        For other launchers C_LH must be re-derived from ray-tracing;
        use C_LH=1.0 as a conservative lower bound in the absence of data.

    Parameters
    ----------
    a, R0, B0       : float  Minor/major radius [m] and on-axis field [T]
                             (B0, nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac
                              are accepted for API symmetry with ECCD/NBCD,
                              not used in either LHCD model).
    nbar, Tbar      : float  Volume-averaged density [10²⁰ m⁻³] and T_e [keV].
    nu_n, nu_T      : float  Profile exponents (unused, see note above).
    Z_eff           : float  Effective ion charge (used only in model='ehst').
    model = 'fenstermacher'
        Empirical Fenstermacher (1988) scaling — PROCESS iefrf=1.
        Uses only Tbar, nbar, R0.  No Z_eff, β or profile dependence.
        Underestimates by ×3 at ITER conditions; provided for comparison.

    model           : str    'ehst' (default) or 'fenstermacher'.
    C_LH            : float  Spectral correction factor for 'ehst' (default
                             1.52, calibrated to ITER 5 GHz ray-tracing).
    beta_T          : float or None  Toroidal beta (None → 0.025, ITER-like).
    rho_ped, n_ped_frac, T_ped_frac : float  Pedestal parameters (unused).

    Returns
    -------
    float  γ_CD^LH  [MA / (MW m²)].

    References
    ----------
    Fisch, Rev. Mod. Phys. 59 (1987) 175 — theoretical basis.
    Karney & Fisch, Phys. Fluids 28 (1985) 116 — quasilinear solutions.
    Ehst & Karney, Nucl. Fusion 31 (1991) 1933 — fitting formula.
    Gormezano et al. (ITER PIPB Ch.6), Nucl. Fusion 47 (2007) S285
        — C_LH=1.52 calibration source.
    Hender et al., AEA FUS 172, UKAEA (1992) — PROCESS iefrf=1,4.
    """
    if model == 'fenstermacher':
        return f_etaCD_LH_simple(Tbar, nbar, R0)

    # model='ehst': Ehst-Karney analytical formula
    # G(Z_eff): parallel Spitzer conductivity factor (Ehst & Karney 1991, eq. 9)
    G_Z  = (32.0 / (5.0 + Z_eff)
            + 2.0
            + 12.0 * (6.0 + Z_eff) / ((5.0 + Z_eff) * (3.0 + Z_eff))
            + 14.0 / (3.0 + Z_eff))
    beta = 0.025 if beta_T is None else float(beta_T)
    lnL  = _ln_Lambda_CD(Tbar, nbar)
    return C_LH * Tbar**0.77 * (0.034 + 0.196 * beta) * G_Z / lnL


def f_etaCD_EC(a, R0, Tbar, nbar, Z_eff, nu_T, nu_n, rho_EC,
               C_EC=0.68, rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
               Vprime_data=None):
    """
    ECCD figure of merit — Ehst & Karney (1991) quasilinear formula.

    Physical basis
    --------------
    EC waves undergo Landau/cyclotron damping on electrons at a resonant
    parallel velocity v_∥,res = (ω − Ω_ce) / k_∥.  For oblique O-mode or
    X-mode injection with n_∥ ≠ 0, resonant electrons carry a net parallel
    momentum asymmetry → net toroidal current (Fisch & Boozer 1980).

    The figure of merit is (Fisch 1987 convention):

        γ_EC = C_EC × T_{e,loc} × (1 − f_trap) / (lnΛ × (1 + Z_eff/2))

    Each factor has a direct physical origin:

    T_{e,loc} / lnΛ
        Fisch-Boozer scaling: γ ∝ v_res² / ν_ee ∝ T_e / (n_e lnΛ),
        because higher T_e reduces collisional momentum loss of the driven
        electrons (Fisch 1987, eq. 6).  Ehst & Karney (1991) replace the
        pure quasilinear T_e¹ scaling with a fit to full Fokker-Planck
        solutions of Karney & Fisch (1985): the exponent stays close to 1
        for ECCD in the relevant velocity range (u_res = v_res/v_te ~ 2–5).

    (1 − f_trap) = f_pass
        Trapped electrons cannot carry a net toroidal current (banana orbit
        cancellation).  f_trap from Kim et al. (1991) is consistent with
        the Start & Cordey (1980) neoclassical correction and is the form
        used in benchmark codes (OFMC, ACCOME; cf. Oikawa et al. 2008).
        Note: f_pass is correctly applied here — unlike LHCD where
        Ehst-Karney's exponent already encodes trapping implicitly.

    1 / (1 + Z_eff/2)
        Spitzer parallel conductivity correction: Ohm's law gives
        σ_∥ ∝ 1/(1 + Z_eff/2) for a Z_eff plasma in the weak-drive limit.
        This is eq. 9 of Ehst & Karney (1991), identical to the factor
        G(Z) used for LHCD.

    Comparison with PROCESS and METIS
    ----------------------------------
    PROCESS Culham model (eccdef, AEA FUS 172): uses the Cohen (IPDG89)
    weak-relativistic formula f ∝ T_e²/m_e²c⁴, averaged over 4 poloidal
    angles.  Less accurate than Ehst-Karney for T_e > 5 keV.
    METIS: uses Giruzzi (1987) trapped-electron correction combined with
    ray-tracing codes; consistent with the present formula at 0D level.
    D0FUS follows the Ehst-Karney analytical fit (PROCESS iefrf=4 spirit),
    which is the better-validated choice for systems-code work at reactor
    temperatures (T_e ~ 10–30 keV).

    Limitation: the formula does not resolve the launch geometry (poloidal
    angle, toroidal refractive index n_∥); all launch-angle effects are
    absorbed into C_EC.  For optimised launchers, C_EC must be re-derived.
    For T_e > 20 keV (EU-DEMO, compact HTS reactors), the fully relativistic
    Lin-Liu & Mau (2003) formula increases γ_EC by ~10–15%.

    Pre-factor C_EC = 0.68 (default)
    ---------------------------------
    Calibrated to Ehst & Karney (1991), Table 2, case: O-mode, tangential
    equatorial launch, ITER-like plasma (T_e,loc ≈ 15 keV, lnΛ ≈ 17.6,
    Z_eff = 1.6, ρ = 0.3, f_pass ≈ 0.62).  Gives γ_EC ≈ 0.20 MA/(MW m²),
    consistent with ITER PIPB Ch. 6 projections of ~0.6–1 MA per 20 MW
    (Gormezano et al. 2007).
    For X-mode top-launch (more efficient), C_EC should be increased.
    For ray-tracing codes (TRAVIS, TORBEAM, GRAY, GENRAY), C_EC is replaced
    by the code-integrated effective coefficient.

    Parameters
    ----------
    a, R0       : float  Minor/major radius [m].
    Tbar, nbar  : float  Volume-averaged T_e [keV] and n_e [10²⁰ m⁻³].
    Z_eff       : float  Effective ion charge.
    nu_T, nu_n  : float  Profile peaking exponents.
    rho_EC      : float  Normalised EC wave deposition radius.
    C_EC        : float  Pre-factor (default 0.68, O-mode tangential equatorial).
    rho_ped, n_ped_frac, T_ped_frac : float  Pedestal parameters.

    Returns
    -------
    float  γ_CD^EC  [MA / (MW m²)].

    References
    ----------
    Fisch & Boozer, Phys. Rev. Lett. 45 (1980) 720 — current drive mechanism.
    Fisch, Rev. Mod. Phys. 59 (1987) 175 — dimensional scaling, eq. 6.
    Karney & Fisch, Phys. Fluids 28 (1985) 116 — Fokker-Planck solutions.
    Ehst & Karney, Nucl. Fusion 31 (1991) 1933 — fitting formula, eq. 9 and Table 2.
    Kim et al., Phys. Fluids B 3 (1991) 2050 — trapped-particle fraction.
    Giruzzi, Nucl. Fusion 27 (1987) 1934 — trapped-electron ECCD correction.
    Oikawa et al. (ITPA benchmarking), IAEA FEC 2008, IT/P3-33
        — validation of f_pass against Start-Cordey G(ε).
    Gormezano et al. (ITER PIPB Ch. 6), Nucl. Fusion 47 (2007) S285 — ITER ref.
    Lin-Liu & Mau, Phys. Plasmas 10 (2003) 4054 — fully relativistic ECCD.
    Hender et al., AEA FUS 172 (1992) — PROCESS Culham model (Cohen/IPDG89).
    """
    Te = max(float(f_Tprof(Tbar, nu_T, rho_EC, rho_ped, T_ped_frac,
                           Vprime_data)), 0.1)
    ne = float(f_nprof(nbar, nu_n, rho_EC, rho_ped, n_ped_frac,
                       Vprime_data))
    lnL    = _ln_Lambda_CD(Te, ne)
    f_pass = 1.0 - _f_trap_CD(rho_EC, a, R0)
    return C_EC * Te * f_pass / (lnL * (1.0 + Z_eff / 2.0))


def f_etaCD_NBI(A_beam, E_beam_keV, a, R0, Tbar, nbar, Z_eff,
                nu_T, nu_n, rho_NBI, C_NBI=0.37,
                rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                Vprime_data=None):
    """
    NBCD figure of merit — Cordey-Mikkelsen intermediate-energy approximation.

    Physical basis
    --------------
    A fast beam injected tangentially is ionised and slows down on the
    background plasma.  The net driven current equals the fast-ion current
    minus the screening electron return current (Ohkawa effect).

    The Stix (1972) critical energy separates ion-friction and electron-
    friction regimes:

        E_c = 14.8 A_b^{2/3} T_{e,loc} [keV]

    At ITER Q=10 (E_b = 1 MeV D, T_{e,loc} = 9 keV): E_c ≈ 211 keV,
    E_b/E_c ≈ 4.7 — intermediate regime, not strictly E_b >> E_c.

    The formula:

        γ_NBI = C_NBI × √(E_b/A_b) × (1 − f_trap) / (lnΛ × (1 + Z_eff/2))

    uses the √(E_b/A_b) scaling, which is an approximation valid in the
    intermediate regime E_b/E_c ~ 2–10 (Mikkelsen & Singer 1983).
    The strict high-energy limit gives γ ∝ E_b/A_b; the true behaviour is:

        γ ∝ F(E_b/E_c) / A_b,  F = slowing-down-averaged Cordey integral.

    The √(E_b/A_b) approximation and C_NBI together absorb F(E_b/E_c)
    at the calibration point.  For E-scans spanning decades, implement F.

    Trapped-particle correction
    ---------------------------
    f_pass = 1 − f_trap approximates the Start & Cordey (1980) shielding
    factor G(ε, Z_eff).  Kim et al. (1991) agrees with Start-Cordey to
    within a few % at ε < 0.4 and Z_eff ~ 1–3 (Oikawa et al. 2008).

    Comparison with PROCESS
    -----------------------
    PROCESS Culham NBI model: Start & Cordey (1980) G(ε) table combined
    with Cordey, Jones & Start (1979) base formula — more accurate at large
    ε or extreme Z_eff.  Equivalent to D0FUS at ITER aspect ratio (ε~0.32)
    to within ~5%.

    Pre-factor C_NBI = 0.37 (default)
    -----------------------------------
    Calibrated to the JT-60U N-NBI record (Urano et al. 2002):
        η = 1.55 × 10¹⁹ A m⁻² W⁻¹ at E_b = 360 keV, A_b = 1 (H-beam),
        T_e(0) ≈ 13 keV, n̄ ≈ 0.0305 × 10²⁰ m⁻³, Z_eff ≈ 3, central dep.
    Inversion at this point → C_NBI,intrinsic = 0.417.  The adopted 0.37
    is conservative, accounting for off-axis deposition and first-orbit
    losses in reactor conditions.  Yields γ_NBI ≈ 0.163 MA/(MW m²) for
    1 MeV D at ITER Q=10 (lower end of PIPB 0.20–0.40 MA/(MW m²); upper
    end requires optimised injection geometry not captured at 0D).

    Parameters
    ----------
    A_beam      : int    Beam ion mass number (1=H, 2=D, 3=T).
    E_beam_keV  : float  Injection energy [keV].
    a, R0, Tbar, nbar, Z_eff, nu_T, nu_n : float  Plasma parameters.
    rho_NBI     : float  Normalised beam deposition radius.
    C_NBI       : float  Pre-factor (default 0.37, tangential co-injection).
    rho_ped, n_ped_frac, T_ped_frac : float  Pedestal parameters.

    Returns
    -------
    float  γ_CD^NBI  [MA / (MW m²)].

    References
    ----------
    Cordey, Jones & Start, Nucl. Fusion 19 (1979) 249 — base NBCD formula.
    Start & Cordey, Phys. Fluids 23 (1980) 1477 — trapped-electron G(ε,Z).
    Stix, Plasma Phys. 14 (1972) 367 — critical energy E_c.
    Mikkelsen & Singer, Nucl. Tech.-Fusion 4 (1983) 237
        — beam CD optimisation and √E_b intermediate-energy regime.
    Cordey, Nucl. Fusion 26 (1986) 123 — trapped-particle effects on NBCD.
    Kim et al., Phys. Fluids B 3 (1991) 2050 — f_trap analytical fit.
    Urano et al. (JT-60U N-NBI), IAEA FEC 2002, paper EX8/3 — C_NBI calibration.
    Oikawa et al. (ITPA NBCD benchmark), IAEA FEC 2008, IT/P3-33
        — validation of f_pass vs Start-Cordey G(ε).
    Gormezano et al. (ITER PIPB Ch. 6), Nucl. Fusion 47 (2007) S285 — ITER ref.
    """
    Te = max(float(f_Tprof(Tbar, nu_T, rho_NBI, rho_ped, T_ped_frac,
                           Vprime_data)), 0.1)
    ne = float(f_nprof(nbar, nu_n, rho_NBI, rho_ped, n_ped_frac,
                       Vprime_data))
    lnL    = _ln_Lambda_CD(Te, ne)
    f_pass = 1.0 - _f_trap_CD(rho_NBI, a, R0)
    return C_NBI * np.sqrt(E_beam_keV / A_beam) * f_pass / (lnL * (1.0 + Z_eff / 2.0))


def f_etaCD_effective(config, a, R0, B0, nbar, Tbar, nu_n, nu_T, Z_eff,
                      rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0):
    """
    Effective CD figure of merit γ_CD [MA/(MW·m²)] for the active heating mix.

    Routes to the appropriate CD model based on ``config.CD_source``:

    * ``'Academic'``: Fixed user-specified γ_CD (config.gamma_CD_acad).
                      No plasma-physics CD model is evaluated — simplest option.
    * ``'LHCD'``  : f_etaCD_LH  — Ehst-Karney quasilinear formula
    * ``'ECCD'``  : f_etaCD_EC  — Ehst-Karney with trapped-particle correction
    * ``'NBCD'``  : f_etaCD_NBI — Cordey-Mikkelsen intermediate-energy formula
    * ``'Multi'`` : power-weighted average of LH, EC, and NBI contributions.
                    ICRH (f_heat_ICR) heats but drives no current (γ_ICR = 0).

    .. warning:: **Development status (2026)**

       Only ``'Academic'`` is fully validated and recommended for production
       runs, parameter scans, and publications.

       The technology-specific branches (``'LHCD'``, ``'ECCD'``, ``'NBCD'``,
       ``'Multi'``) rely on physics models (Ehst-Karney, Cordey-Mikkelsen)
       that are implemented but **not yet validated** against experimental
       data or cross-checked with other systems codes (PROCESS, PLASMOD).
       Their quantitative output should be considered indicative only.

    For ``'Multi'``, the effective γ is:

        γ_eff = (f_LH·γ_LH + f_EC·γ_EC + f_NBI·γ_NBI) / (f_LH + f_EC + f_NBI + f_ICR)

    where the denominator is the sum of all heating fractions (i.e. P_CD_total
    is shared among all sources, including ICRH which contributes zero current).
    This correctly penalises γ_eff when ICRH carries a large fraction of power.

    Parameters
    ----------
    config  : GlobalConfig  Full configuration object (reads CD_source, f_heat_*,
                            rho_EC, C_EC, rho_NBI, C_NBI, A_beam, E_beam_keV).
    a, R0   : float  Minor and major radius [m].
    B0      : float  On-axis magnetic field [T].
    nbar    : float  Volume-averaged electron density [10²⁰ m⁻³].
    Tbar    : float  Volume-averaged electron temperature [keV].
    nu_n, nu_T : float  Density and temperature peaking exponents.
    Z_eff   : float  Effective plasma charge.
    rho_ped, n_ped_frac, T_ped_frac : float  Pedestal parameters.

    Returns
    -------
    float  Effective γ_CD [MA/(MW·m²)].

    References
    ----------
    Fisch, Rev. Mod. Phys. 59 (1987) 175.
    Ehst & Karney, Nucl. Fusion 31 (1991) 1933.
    """
    CD_source = config.CD_source

    if CD_source == 'Academic':
        # Technology-agnostic mode: return the user-specified fixed γ_CD.
        # No plasma-physics CD model is evaluated.
        return config.gamma_CD_acad

    elif CD_source == 'LHCD':
        return f_etaCD_LH(a, R0, B0, nbar, Tbar, nu_n, nu_T,
                          Z_eff=Z_eff,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac)

    elif CD_source == 'ECCD':
        return f_etaCD_EC(a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                          config.rho_EC, config.C_EC,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac)

    elif CD_source == 'NBCD':
        return f_etaCD_NBI(config.A_beam, config.E_beam_keV,
                           a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                           config.rho_NBI, config.C_NBI,
                           rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                           T_ped_frac=T_ped_frac)

    elif CD_source == 'Multi':
        # Individual efficiencies
        gamma_LH  = f_etaCD_LH(a, R0, B0, nbar, Tbar, nu_n, nu_T,
                                Z_eff=Z_eff,
                                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                T_ped_frac=T_ped_frac)
        gamma_EC  = f_etaCD_EC(a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                                config.rho_EC, config.C_EC,
                                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                T_ped_frac=T_ped_frac)
        gamma_NBI = f_etaCD_NBI(config.A_beam, config.E_beam_keV,
                                 a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                                 config.rho_NBI, config.C_NBI,
                                 rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                 T_ped_frac=T_ped_frac)

        # Power-weighted average: ICRH contributes heating but zero current drive
        f_LH  = config.f_heat_LH
        f_EC  = config.f_heat_EC
        f_NBI = config.f_heat_NBI
        f_ICR = config.f_heat_ICR
        f_total = f_LH + f_EC + f_NBI + f_ICR

        if f_total <= 0.0:
            raise ValueError(
                "f_etaCD_effective: all heating fractions are zero in Multi mode "
                f"(f_LH={f_LH}, f_EC={f_EC}, f_NBI={f_NBI}, f_ICR={f_ICR}). "
                "At least one fraction must be positive."
            )
        # γ_ICR = 0 → ICRH power enters denominator but not numerator
        return (f_LH * gamma_LH + f_EC * gamma_EC + f_NBI * gamma_NBI) / f_total

    else:
        raise ValueError(
            f"f_etaCD_effective: unknown CD_source '{CD_source}'. "
            "Valid options: 'Academic', 'LHCD', 'ECCD', 'NBCD', 'Multi'."
        )


def f_ICD(Ip, Ib, I_Ohm):
    """
    Required external current drive from the plasma current balance.

        I_CD = Ip − I_b − I_Ohm

    Alias for ``f_I_CD_from_balance``; provided for naming consistency with
    ``f_I_Ohm`` in run.py Steady-State and Pulsed current bookkeeping.

    Parameters
    ----------
    Ip    : float  Total plasma current [MA].
    Ib    : float  Bootstrap current [MA].
    I_Ohm : float  Inductive (Ohmic) current [MA]  (0 in steady state).

    Returns
    -------
    float  Required non-inductive driven current I_CD [MA].
    """
    return f_I_CD_from_balance(Ip, Ib, I_Ohm)


def f_CD_breakdown(config, P_CD_total, R0, nbar,
                   eta_LH, eta_EC, eta_NBI):
    """
    Decompose the total CD power into per-source powers and driven currents.

    For single-source modes (``CD_source`` ∈ {'LHCD', 'ECCD', 'NBCD'}):
        The full P_CD_total is assigned to that source; all others are zero.

    For ``CD_source = 'Multi'``:
        Powers are split in proportion to the heating fractions
        (f_heat_LH, f_heat_EC, f_heat_NBI, f_heat_ICR) after normalisation.
        ICRH receives its fraction of power but drives no current.

    Driven currents per source:
        I_i = γ_i × P_i / (R₀ × n̄_e)

    Parameters
    ----------
    config      : GlobalConfig
    P_CD_total  : float  Total absorbed CD/heating power [MW].
    R0          : float  Major radius [m].
    nbar        : float  Volume-averaged electron density [10²⁰ m⁻³].
    eta_LH      : float  LHCD figure of merit [MA/(MW·m²)].
    eta_EC      : float  ECCD figure of merit [MA/(MW·m²)].
    eta_NBI     : float  NBCD figure of merit [MA/(MW·m²)].

    Returns
    -------
    dict with keys:
        'P_LH', 'P_EC', 'P_NBI', 'P_ICR'  — power per source [MW]
        'I_LH', 'I_EC', 'I_NBI'            — driven current per source [MA]
        (ICRH drives no current: I_ICR is not returned)
    """
    CD_source = config.CD_source

    if CD_source == 'Academic':
        # Technology-agnostic: all power is generic auxiliary heating.
        # Mapped to P_LH slot for output-tuple compatibility; no physical
        # significance — per-source breakdown is meaningless in this mode.
        P_LH, P_EC, P_NBI, P_ICR = P_CD_total, 0.0, 0.0, 0.0

    elif CD_source == 'LHCD':
        P_LH, P_EC, P_NBI, P_ICR = P_CD_total, 0.0, 0.0, 0.0

    elif CD_source == 'ECCD':
        P_LH, P_EC, P_NBI, P_ICR = 0.0, P_CD_total, 0.0, 0.0

    elif CD_source == 'NBCD':
        P_LH, P_EC, P_NBI, P_ICR = 0.0, 0.0, P_CD_total, 0.0

    elif CD_source == 'Multi':
        f_LH  = config.f_heat_LH
        f_EC  = config.f_heat_EC
        f_NBI = config.f_heat_NBI
        f_ICR = config.f_heat_ICR
        f_total = f_LH + f_EC + f_NBI + f_ICR
        if f_total <= 0.0:
            f_LH = f_EC = f_NBI = f_ICR = 0.25   # equal split fallback
            f_total = 1.0
        P_LH  = P_CD_total * f_LH  / f_total
        P_EC  = P_CD_total * f_EC  / f_total
        P_NBI = P_CD_total * f_NBI / f_total
        P_ICR = P_CD_total * f_ICR / f_total

    else:
        raise ValueError(
            f"f_CD_breakdown: unknown CD_source '{CD_source}'. "
            "Valid: 'Academic', 'LHCD', 'ECCD', 'NBCD', 'Multi'."
        )

    # Non-inductive current per source:  I = γ × P / (R₀ × n̄)
    denom = R0 * nbar   # [m × 10²⁰ m⁻³]
    I_LH  = eta_LH  * P_LH  / denom if denom > 0 else 0.0
    I_EC  = eta_EC  * P_EC  / denom if denom > 0 else 0.0
    I_NBI = eta_NBI * P_NBI / denom if denom > 0 else 0.0

    return {
        'P_LH' : P_LH,  'P_EC' : P_EC,  'P_NBI' : P_NBI,  'P_ICR' : P_ICR,
        'I_LH' : I_LH,  'I_EC' : I_EC,  'I_NBI' : I_NBI,
    }


def f_I_CD(R0, nbar, eta_CD, P_CD):
    """
    Non-inductive current from a single CD source.

        I_CD = γ_CD × P_CD / (R₀ × n̄_e)

    Parameters
    ----------
    R0      : float  Major radius [m].
    nbar    : float  Volume-averaged density [10²⁰ m⁻³].
    eta_CD  : float  CD figure of merit γ [MA MW⁻¹ m⁻²].
    P_CD    : float  Absorbed CD power [MW].

    Returns
    -------
    float  Non-inductive driven current [MA].

    References
    ----------
    Fisch, Rev. Mod. Phys. 59 (1987) 175 — dimensional form.
    """
    return eta_CD * P_CD / (R0 * nbar)


def f_PCD(R0, nbar, I_CD, eta_CD):
    """
    Required CD power for a target non-inductive current.

        P_CD = R₀ × n̄_e × I_CD / γ_CD

    Inverse of f_I_CD; used in steady-state scenario optimisation.

    Parameters
    ----------
    R0, nbar  : float  Major radius [m] and density [10²⁰ m⁻³].
    I_CD      : float  Target non-inductive current [MA].
    eta_CD    : float  CD figure of merit [MA MW⁻¹ m⁻²].

    Returns
    -------
    float  Required CD power [MW].
    """
    return R0 * nbar * I_CD / eta_CD


def f_PLH(eta_RF, f_RP, P_CD):
    """
    Wall-plug electrical power for lower-hybrid current drive.

        P_wall = P_CD / (η_RF × f_RP)

    Parameters
    ----------
    eta_RF : float  Klystron wall-plug efficiency (typically 0.5–0.7).
    f_RP   : float  Fraction of klystron power absorbed by the plasma.
    P_CD   : float  CD power deposited in the plasma [MW].

    Returns
    -------
    float  Wall-plug electrical power demand [MW].
    """
    return P_CD / (eta_RF * f_RP)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # §6 Validation — CD figures of merit at ITER Q=10.
    #
    # Reference: Shimada et al., Nucl. Fusion 47 (2007) S1.
    # ITER PIPB Ch.6 projections (Gormezano et al., NF 47 (2007) S285):
    #   γ_LH ≈ 0.24,  γ_EC ≈ 0.20,  γ_NBI ≈ 0.15 MA/(MW m²)
    # ------------------------------------------------------------------

    kw = dict(a=2.0, R0=6.2, B0=5.3, nbar=1.01, Tbar=8.9,
              nu_n=0.1, nu_T=1.0, Z_eff=1.6, beta_T=0.025,
              rho_ped=0.94, n_ped_frac=0.80, T_ped_frac=0.40)

    # LHCD: Ehst-Karney with C_LH=1.52 (spectral correction, ITER calibrated)
    gLH = f_etaCD_LH(kw['a'], kw['R0'], kw['B0'], kw['nbar'], kw['Tbar'],
                     kw['nu_n'], kw['nu_T'], Z_eff=kw['Z_eff'],
                     model='ehst', C_LH=1.52, beta_T=kw['beta_T'])

    # ECCD and NBCD: local T_e at ρ_dep = 0.3, with trapped-particle correction
    gEC  = f_etaCD_EC(kw['a'], kw['R0'], kw['Tbar'], kw['nbar'], kw['Z_eff'],
                      kw['nu_T'], kw['nu_n'], rho_EC=0.3,
                      rho_ped=kw['rho_ped'], n_ped_frac=kw['n_ped_frac'],
                      T_ped_frac=kw['T_ped_frac'])
    
    # NBI reference: 0.20–0.40 (Gormezano 2007) spans all ITER scenarios.
    # For Scenario 2 (Q=10, n̄ ~ 1.01×10²⁰ m⁻³, 15 MA), the lower end
    # ~0.15–0.20 applies; the upper end (0.40) is Scenario 4 (Q=5,
    # n̄ ~ 0.67×10²⁰ m⁻³) where lower density greatly increases efficiency.
    # D0FUS result 0.163 is consistent with Q=10 conditions.
    gNBI = f_etaCD_NBI(2, 1000., kw['a'], kw['R0'], kw['Tbar'], kw['nbar'],
                       kw['Z_eff'], kw['nu_T'], kw['nu_n'], rho_NBI=0.3,
                       rho_ped=kw['rho_ped'], n_ped_frac=kw['n_ped_frac'],
                       T_ped_frac=kw['T_ped_frac'])

    P_EC, P_NBI = 20.0, 33.0   # absorbed CD power [MW]  — Shimada 2007
    I_EC  = f_I_CD(kw['R0'], kw['nbar'], gEC,  P_EC)
    I_NBI = f_I_CD(kw['R0'], kw['nbar'], gNBI, P_NBI * 0.95)  # 5% orbit loss

    print("\n── CD figures of merit — ITER Q=10 (H-mode pedestal) " + "─"*18)
    print(f"  {'Source':<18} {'γ [MA/(MW m²)]':>14}  {'I_CD [MA]':>10}"
          f"  {'P_inj [MW]':>11}  {'ITER ref.':>10}")
    print("  " + "─"*68)
    print(f"  {'LHCD':<18} {gLH:>14.4f}  {'—':>10}  {'—':>11}  {'~0.24':>10}")
    print(f"  {'ECCD (ρ=0.3)':<18} {gEC:>14.4f}  {I_EC:>10.3f}"
          f"  {P_EC:>11.1f}  {'~0.20':>10}")
    print(f"  {'NBCD D,1MeV':<18} {gNBI:>14.4f}  {I_NBI:>10.3f}"
          f"  {P_NBI:>11.1f}  {'~0.15':>10}")
    
    # Current budget check (Eriksson et al., Nucl. Fusion 64 (2024) 126033):
    #   Inductive ~ 10 MA, non-inductive ~ 5 MA (bootstrap dominant at ~3.5 MA),
    #   I_NBI + I_EC ~ 1-2 MA. D0FUS total: 0.634 + 0.814 = 1.45 MA. Consistent.


#%% L-H transition threshold

def f_P_sep(P_fus, P_CD, P_rad=0.0):
    """
    Net power exiting the confined plasma, with radiation subtracted.

    In steady-state power balance:

        P_out = P_α + P_CD − P_rad

    The physical meaning of P_out depends on *which* P_rad is passed:

    P_rad = P_rad_core (radiated inside ρ < ρ_core)
        → P_out = true separatrix power = power conducted + convected
          across the LCFS.  This is the quantity to compare against the
          L–H threshold (Martin 2008, Delabie 2017) and to use in the
          energy confinement time definition  τ_E = W_th / P_loss.

    P_rad = P_rad_total (core + edge radiation)
        → P_out = divertor power P_div = power actually reaching the
          divertor target plates.  Edge radiation (from seeded impurities
          in the SOL) reduces the divertor load without affecting τ_E.
          This is the correct input for Eich λ_q and heat flux estimates.

    In the D0FUS run module, this function is called with P_rad_total to
    compute P_div for downstream divertor metrics.  The L–H threshold
    comparison is done separately against P_Thresh.

    Parameters
    ----------
    P_fus : float
        Total fusion power [MW].
    P_CD : float
        Total auxiliary heating and current drive power injected [MW].
    P_rad : float, optional
        Radiated power [MW] (default 0 → upper-bound estimate).
        Pass P_rad_core for true P_sep; pass P_rad_total for P_div.

    Returns
    -------
    float
        Net power [MW] — P_sep or P_div depending on the P_rad input.

    References
    ----------
    Loarte et al., Nucl. Fusion 47 (2007) S203.
    Kallenbach et al., PPCF 55 (2013) 124041.
    Lux et al., PPCF 58 (2016) 075001 — core/edge radiation split.
    """
    return f_P_alpha(P_fus) + P_CD - P_rad


def f_P_wall(P_rad, S_wall):
    """
    Mean radiative power flux density on the first wall [MW m⁻²].

    The first wall receives power primarily from volumetric radiation
    (bremsstrahlung, synchrotron, impurity line emission) emitted
    isotropically from the confined plasma and the SOL.  Conducted heat
    from the SOL is channelled to the divertor strike points and does
    NOT contribute significantly to the main-chamber first-wall load
    in either H-mode or L-mode diverted configurations.

        q_FW = P_rad_total / S_wall

    The neutron wall load (Gamma_n) and ELM transient loads are computed
    separately.  Divertor target heat flux requires the Eich scaling
    (f_heat_PFU_Eich), not this function.

    Parameters
    ----------
    P_rad : float
        Total radiated power [MW] (bremsstrahlung + synchrotron + line).
    S_wall : float
        First-wall wetted surface area [m²].

    Returns
    -------
    float
        Mean first-wall radiative power flux density [MW m⁻²].

    References
    ----------
    Loarte et al., Nucl. Fusion 47 (2007) S203 — radiation balance.
    Wenninger et al., Nucl. Fusion 57 (2017) 046002 — EU-DEMO FW loads.
    """
    return P_rad / S_wall

def P_Thresh_Martin(nbar, B0, a, R0, kappa, M_ion):
    """
    L-H transition power threshold — Martin et al. (2008).

    Empirical multi-machine power-law regression from the ITPA H-mode database
    (2008 update), with the plasma surface area exponent free and the ion mass
    exponent fixed to 1:

        P_LH = 0.0488 × (2/M) × n̄^{0.717} × B₀^{0.803} × S^{0.941}

    where S = 4π²R₀ a √((1+κ²)/2) is the plasma surface area [m²].

    This is the ITER baseline L-H scaling.  Uncertainty: RMSE ~ 30 %.
    ITER Q=10 prediction: ~85 MW (95 % CI: 45–160 MW).

    Parameters
    ----------
    nbar : float  Line-averaged electron density [10²⁰ m⁻³].
    B0 : float  On-axis magnetic field [T].
    a, R0, kappa : float  Minor radius [m], major radius [m], elongation.
    M_ion : float  Main ion atomic mass [AMU] (DT: 2.5).

    Returns
    -------
    float  L-H power threshold [MW].

    References
    ----------
    Martin et al., J. Phys.: Conf. Ser. 123 (2008) 012033.
    """
    S = 4.0 * np.pi**2 * R0 * a * np.sqrt((1.0 + kappa**2) / 2.0)
    return 0.0488 * (2.0 / M_ion) * nbar**0.717 * B0**0.803 * S**0.941


def P_Thresh_New_S(nbar, B0, a, R0, kappa, M_ion):
    """
    L-H transition threshold — Delabie (ITPA 2017), surface-area regression.

    Updated regression on the 2017 ITPA database with S exponent fixed to 1
    and ion mass exponent free (fitted: 0.96):

        P_LH = 0.045 × C_div × (2/M)^{0.96} × n̄^{1.08} × B₀^{0.56} × S

    C_div = 1.0 for standard lower single-null divertor;
    C_div = 1.93 for upper-null or corner configurations.

    RMSE ~ 26 %.

    Parameters
    ----------
    nbar, B0, a, R0, kappa, M_ion : float  (see P_Thresh_Martin).

    Returns
    -------
    float  L-H power threshold [MW].

    References
    ----------
    Delabie, ITPA TC-26 (2017) — unpublished internal report.
    """
    S     = 4.0 * np.pi**2 * R0 * a * np.sqrt((1.0 + kappa**2) / 2.0)
    C_div = 1.0
    return 0.045 * C_div * (2.0 / M_ion)**0.96 * nbar**1.08 * B0**0.56 * S


def P_Thresh_New_Ip(nbar, B0, a, R0, kappa, Ip, M_ion):
    """
    L-H transition threshold — Delabie (ITPA 2017), Ip/a regression.

    Alternative regression replacing B₀ with Ip/a as the current-related
    variable, reducing scatter (RMSE ~ 21 %):

        P_LH = 0.049 × (2/M) × n̄^{1.06} × (Ip/a)^{0.65} × S

    Parameters
    ----------
    nbar, B0, a, R0, kappa, M_ion : float  (see P_Thresh_Martin).
    Ip : float  Plasma current [MA].

    Returns
    -------
    float  L-H power threshold [MW].

    References
    ----------
    Delabie, ITPA TC-26 (2017) — unpublished internal report.
    """
    S = 4.0 * np.pi**2 * R0 * a * np.sqrt((1.0 + kappa**2) / 2.0)
    return 0.049 * (2.0 / M_ion) * nbar**1.06 * (Ip / a)**0.65 * S


def f_P_thresh(nbar, B0, a, R0, kappa, M_ion, Ip=None,
               Option_PLH='Martin'):
    """
    L-H transition power threshold — unified dispatcher.

    Selects among the three available scaling laws via `Option_PLH`.
    This mirrors the `f_Get_parameter_scaling_law` pattern used for τ_E.

    Parameters
    ----------
    nbar  : float  Line-averaged electron density [10²⁰ m⁻³].
    B0    : float  On-axis toroidal field [T].
    a     : float  Minor radius [m].
    R0    : float  Major radius [m].
    kappa : float  Plasma elongation at the LCFS [-].
    M_ion : float  Main ion atomic mass [AMU] (D-T: 2.5).
    Ip    : float or None
        Plasma current [MA].  Required only for Option_PLH='New_Ip';
        a ValueError is raised if None is passed with that option.
    Option_PLH : str, optional
        Scaling law selector (default 'Martin'):
        'Martin'  — Martin et al. (2008).  ITER baseline, RMSE ~ 30 %.
        'New_S'   — Delabie ITPA (2017), surface-area regression, RMSE ~ 26 %.
        'New_Ip'  — Delabie ITPA (2017), Ip/a regression, RMSE ~ 21 %.

    Returns
    -------
    float  L-H transition power threshold [MW].

    References
    ----------
    Martin et al., J. Phys.: Conf. Ser. 123 (2008) 012033.
    Delabie, ITPA TC-26 (2017).
    """
    if Option_PLH == 'Martin':
        return P_Thresh_Martin(nbar, B0, a, R0, kappa, M_ion)
    elif Option_PLH == 'New_S':
        return P_Thresh_New_S(nbar, B0, a, R0, kappa, M_ion)
    elif Option_PLH == 'New_Ip':
        if Ip is None:
            raise ValueError(
                "f_P_thresh: Ip must be provided for Option_PLH='New_Ip'."
            )
        return P_Thresh_New_Ip(nbar, B0, a, R0, kappa, Ip, M_ion)
    else:
        raise ValueError(
            f"Unknown Option_PLH: '{Option_PLH}'. "
            "Valid options: 'Martin', 'New_S', 'New_Ip'."
        )


if __name__ == "__main__":

    # ITER Q=10 DT baseline — Shimada et al., Nucl. Fusion 47 (2007) S1
    # M_eff = 2.5 for DT already included; isotope factor P_LH ∝ 1/M applied.
    # Metal-wall correction (W/Be vs C): ×~0.70 [Ryter et al., NF 53 (2013) 113003;
    #   Maggi et al., NF 54 (2014) 023007] → best estimate ~50 MW.
    p_IT = dict(nbar=1.0, B0=5.3, a=2.0, R0=6.2, kappa=1.7, Ip=15.0, M=2.5)
    PM = P_Thresh_Martin(p_IT['nbar'], p_IT['B0'], p_IT['a'], p_IT['R0'],
                         p_IT['kappa'], p_IT['M'])
    PS = P_Thresh_New_S (p_IT['nbar'], p_IT['B0'], p_IT['a'], p_IT['R0'],
                         p_IT['kappa'], p_IT['M'])
    PI = P_Thresh_New_Ip(p_IT['nbar'], p_IT['B0'], p_IT['a'], p_IT['R0'],
                         p_IT['kappa'], p_IT['Ip'], p_IT['M'])

    print("\n── L-H power threshold — ITER Q=10 (DT, W/Be wall) ─────────────────────────")
    print(f"  {'Scaling':<10} {'P_LH [MW]':>10}  {'ITER ref. [MW]':>14}")
    print("  " + "─" * 72)
    print(f"  {'Martin':<10} {PM:>10.1f}  {'~50':>14}")
    print(f"  {'New_S':<10} {PS:>10.1f}  {'~50':>14}")
    print(f"  {'New_Ip':<10} {PI:>10.1f}  {'~50':>14}")


def f_P_LH_thresh(nbar, B0, a, R0, kappa, M_ion, Ip=None,
                  Option_PLH='Martin'):
    """Alias for f_P_thresh — kept for backward compatibility."""
    return f_P_thresh(nbar, B0, a, R0, kappa, M_ion, Ip, Option_PLH)


#%% Plasma resistivity
# ─────────────────────────────────────────────────────────────────────────────
# Classical and neoclassical parallel resistivity models.
# Used by f_Reff (loop voltage / flux consumption), f_q_profile_selfconsistent
# (Ohmic current density), and all functions requiring local η(ρ).
#
# Hierarchy:
#   eta_old      — simplified Spitzer (no Z_eff, no ln(Λ)), Wesson approx.
#   eta_spitzer  — classical Spitzer-Härm with g(Z) and Coulomb logarithm.
#   eta_sauter   — neoclassical, Sauter et al. PoP 6 (1999) 2834.
#   eta_redl     — neoclassical, Redl et al. PoP 28 (2021) 022502.
#
# The neoclassical models apply a trapped-particle correction to the Spitzer
# conductivity: eta_neo = eta_Spitzer / sigma_ratio(f_t_eff, Z_eff), where
# f_t_eff interpolates the banana–Pfirsch-Schlüter regimes.
# ─────────────────────────────────────────────────────────────────────────────


def _coulomb_logarithm(T_keV, ne):
    """
    Coulomb logarithm ln(Lambda) for electron-ion collisions.

    Temperature-dependent formula from the NRL Plasma Formulary (2019, p.34).
    Supports both scalar and array inputs (vectorised with np.where).

    Parameters
    ----------
    T_keV : float or ndarray
        Electron temperature [keV].
    ne : float or ndarray
        Electron density [m^-3].

    Returns
    -------
    ln_lambda : float or ndarray
        Coulomb logarithm [-], clamped to >= 5.
    """
    T_eV   = np.asarray(T_keV, dtype=float) * 1000.0
    ne_cm3 = np.maximum(np.asarray(ne, dtype=float) * 1e-6, 1.0)
    T_safe = np.maximum(T_eV, 0.1)
    ln_lambda = np.where(
        T_safe < 10.0,
        23.0  - np.log(ne_cm3**0.5 * T_safe**(-1.5)),
        24.15 - np.log(ne_cm3**0.5 * T_safe**(-1.0))
    )
    return np.maximum(ln_lambda, 5.0)


def eta_old(T_keV, ne, Z_eff=1.0):
    """
    Simple Spitzer resistivity from Wesson (no Z_eff, no ln(Lambda) dependence).
    """
    eta = 2.8e-8 / (T_keV**1.5)  # Spitzer resistivity from Wesson

    return eta


def eta_spitzer(T_keV, ne, Z_eff=1.0):
    """
    Classical Spitzer-Härm parallel resistivity [Ohm.m].

    Accounts for electron-electron collisions via the g(Z) factor from
    Spitzer & Härm (1953). This correction reduces resistivity compared
    to naive Z_eff scaling, especially at high Z_eff.
    Supports both scalar and array inputs (vectorised).

    Formula:
        eta = 1.65e-9 * ln(Lambda) * T_keV^(-1.5) * Z_eff * g(Z)/g(1)

    where g(Z) = (1 + 1.198*Z + 0.222*Z^2) / (1 + 2.966*Z + 0.753*Z^2)
    captures e-e collision effects. Tabulated Spitzer values:
        g(1)=0.513, g(2)=0.438, g(4)=0.362

    Coulomb logarithm (NRL Formulary):
        T < 10 eV:  ln(L) = 23 - ln(ne_cm3^0.5 * T_eV^-1.5)
        T > 10 eV:  ln(L) = 24.15 - ln(ne_cm3^0.5 * T_eV^-1)

    References:
        [1] L. Spitzer & R. Härm, Phys. Rev. 89, 977 (1953)
        [2] NRL Plasma Formulary (2019), p.34
    """
    ln_lambda = _coulomb_logarithm(T_keV, ne)

    # Spitzer-Härm g(Z) correction factor
    g_Z = (1 + 1.198*Z_eff + 0.222*Z_eff**2) / (1 + 2.966*Z_eff + 0.753*Z_eff**2)
    g_1 = (1 + 1.198 + 0.222) / (1 + 2.966 + 0.753)
    coef_Zeff = Z_eff * g_Z / g_1

    T_safe = np.maximum(np.asarray(T_keV, dtype=float), 1e-4)
    return 1.65e-9 * ln_lambda * T_safe**(-1.5) * coef_Zeff


def eta_sauter(T_keV, ne, Z_eff, epsilon, q=2.0, R0=6.2):
    """
    Neoclassical resistivity according to Sauter et al. (1999).

    Trapped electrons (f_t ~ 1.46*sqrt(eps)) reduce parallel conductivity.
    The effective trapped fraction interpolates between banana (low nu*) 
    and Pfirsch-Schlüter (high nu*) regimes.

    Key relations (Sauter Eqs. 16a-b):
        sigma_neo/sigma_Sp = 1 - (1+0.36/Z)*X + 0.59/Z*X^2 - 0.23/Z*X^3
        f_t_eff = f_t / [1 + 0.26*(1-f_t)*sqrt(nu*)*(1+0.18*(Z-1)^0.5)
                         + 0.18*(1-0.37*f_t)*nu*/sqrt(Z)]

    Collisionality (Eq. 18c):
        nu*_e = 0.012 * q*R*Z_eff*n_19*ln(L) / (eps^1.5 * T_keV^2)

    References:
        [1] O. Sauter et al., Phys. Plasmas 6, 2834 (1999)
        [2] O. Sauter et al., Erratum, Phys. Plasmas 9, 5140 (2002)
        [3] https://crppwww.epfl.ch/~sauter/neoclassical/
    """
    ln_lambda = _coulomb_logarithm(T_keV, ne)

    # Geometric trapped fraction (Kim et al. PoF 1991)
    # Regularize epsilon at magnetic axis: epsilon(rho=0) = 0 would cause
    # division by zero in nu_star. At epsilon -> 0: f_t -> 0 and the
    # neoclassical correction vanishes (sigma_neo -> sigma_Spitzer).
    eps_reg = max(epsilon, 1e-6)
    sqrt_eps = np.sqrt(eps_reg)
    f_t = 1.46 * sqrt_eps * (1 - 0.54 * sqrt_eps)

    # Electron collisionality
    n_19 = ne / 1.0e19
    nu_star_e = 0.012 * q * R0 * Z_eff * n_19 * ln_lambda / (eps_reg**1.5 * T_keV**2)

    # Effective trapped fraction (Eq. 16b)
    sqrt_nu = np.sqrt(nu_star_e)
    denom = (1.0 + 0.26*(1-f_t)*sqrt_nu*(1 + 0.18*(Z_eff-1)**0.5)
             + 0.18*(1-0.37*f_t)*nu_star_e/np.sqrt(Z_eff))
    f_t_eff = f_t / denom

    # Conductivity ratio (Eq. 16a)
    X = f_t_eff
    sigma_ratio = 1.0 - (1.0+0.36/Z_eff)*X + 0.59/Z_eff*X**2 - 0.23/Z_eff*X**3

    return eta_spitzer(T_keV, ne, Z_eff) / sigma_ratio


def eta_redl(T_keV, ne, Z_eff, epsilon, q=2.0, R0=6.2):
    """
    Neoclassical resistivity according to Redl et al. (2021).

    Improved Sauter formulae refitted against NEO drift-kinetic code.
    Better accuracy at high collisionality (edge pedestal) and with impurities.

    Key relations (Redl Eqs. 17-18):
        sigma_neo/sigma_Sp = 1 - (1+0.21/Z)*X + 0.54/Z*X^2 - 0.33/Z*X^3
        f_t_eff = f_t / [1 + 0.25*(1-0.7*f_t)*sqrt(nu*)*(1+0.45*(Z-1)^0.5)
                         + 0.61*(1-0.41*f_t)*nu*/sqrt(Z)]

    References:
        [1] A. Redl et al., Phys. Plasmas 28, 022502 (2021)
        [2] Data: https://doi.org/10.5281/zenodo.4072358
    """
    ln_lambda = _coulomb_logarithm(T_keV, ne)

    # Geometric trapped fraction
    # Regularize epsilon at magnetic axis (see eta_sauter for rationale)
    eps_reg = max(epsilon, 1e-6)
    sqrt_eps = np.sqrt(eps_reg)
    f_t = 1.46 * sqrt_eps * (1 - 0.54 * sqrt_eps)

    # Electron collisionality
    n_19 = ne / 1.0e19
    nu_star_e = 0.012 * q * R0 * Z_eff * n_19 * ln_lambda / (eps_reg**1.5 * T_keV**2)

    # Effective trapped fraction (Eq. 18)
    sqrt_nu = np.sqrt(nu_star_e)
    denom = (1.0 + 0.25*(1-0.7*f_t)*sqrt_nu*(1 + 0.45*(Z_eff-1)**0.5)
             + 0.61*(1-0.41*f_t)*nu_star_e/np.sqrt(Z_eff))
    f_t_eff = f_t / denom

    # Conductivity ratio (Eq. 17)
    X = f_t_eff
    sigma_ratio = 1.0 - (1.0+0.21/Z_eff)*X + 0.54/Z_eff*X**2 - 0.33/Z_eff*X**3

    return eta_spitzer(T_keV, ne, Z_eff) / sigma_ratio


#%% Loop voltage and effective resistance


def f_Reff(a, kappa, R0, Tbar, nbar, Z_eff, q95, nu_T, nu_n,
           eta_model='redl',
           rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
           Vprime_data=None):
    """
    Effective plasma resistance from neoclassical conductivity integration.

    Physical model
    --------------
    In steady-state flat-top, the loop electric field E_phi is uniform
    across the plasma cross-section.  The Ohmic current density follows:

        j_Ohm(rho) = E_phi / eta_neo(rho) = E_phi × sigma_neo(rho)

    The effective resistance is defined by V_loop = R_eff × I_Ohm:

        R_eff = (2 pi R0)^2 / ∫_0^1 V'(rho) / eta_neo(rho) drho

    where V'(rho) = dV/drho is the shaped volume derivative from Miller
    geometry (Vprime_data) when available, or the cylindrical approximation
    V' = 4 pi^2 R0 a^2 kappa rho otherwise.

    The current profile emerges self-consistently from T(rho) and n(rho)
    via the neoclassical conductivity — no prescribed alpha_J exponent.

    Parameters
    ----------
    a : float
        Minor radius [m].
    kappa : float
        Plasma elongation [-].
    R0 : float
        Major radius [m].
    Tbar : float
        Volume-averaged temperature [keV].
    nbar : float
        Volume-averaged density [1e20 m^-3].
    Z_eff : float
        Effective charge [-].
    q95 : float
        Safety factor at 95%% flux surface [-].
    nu_T : float
        Temperature profile peaking exponent [-].
    nu_n : float
        Density profile peaking exponent [-].
    eta_model : str
        Resistivity model: 'old', 'spitzer', 'sauter', 'redl' (default).
    rho_ped : float
        Normalised pedestal radius (1.0 = no pedestal).
    n_ped_frac : float
        n_ped / nbar.
    T_ped_frac : float
        T_ped / Tbar.
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        When None, uses cylindrical V' = 4 pi^2 R0 a^2 kappa rho.

    Returns
    -------
    R_eff : float
        Effective plasma resistance [Ohm].

    References
    ----------
    Sauter O. et al., Phys. Plasmas 6 (1999) 2834.
    Redl A. et al., Phys. Plasmas 28 (2021) 022502.
    """
    from scipy import integrate
    import warnings
    from scipy.integrate import IntegrationWarning

    def _eta_local(rho):
        """Neoclassical resistivity at normalised radius rho."""
        T_loc = max(float(f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac)), 0.1)
        n_loc = float(f_nprof(nbar, nu_n, rho, rho_ped, n_ped_frac))
        n_loc_m3 = n_loc * 1e20
        epsilon_loc = rho * a / R0

        if eta_model == 'old':
            return eta_old(T_loc, n_loc_m3, Z_eff)
        elif eta_model == 'spitzer':
            return eta_spitzer(T_loc, n_loc_m3, Z_eff)
        elif eta_model == 'sauter':
            return eta_sauter(T_loc, n_loc_m3, Z_eff, epsilon_loc, q95, R0)
        elif eta_model == 'redl':
            return eta_redl(T_loc, n_loc_m3, Z_eff, epsilon_loc, q95, R0)
        else:
            raise ValueError(f"Unknown eta_model '{eta_model}'.")

    def _Vprime_local(rho):
        """Volume derivative V'(rho) = dV/drho [m^3]."""
        if Vprime_data is not None:
            return float(interpolate_Vprime(rho, Vprime_data[0], Vprime_data[1]))
        return 4.0 * np.pi**2 * R0 * a**2 * kappa * rho

    def _integrand(rho):
        """V'(rho) / eta(rho) — shaped conductance integrand."""
        return _Vprime_local(rho) / max(_eta_local(rho), 1e-12)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=IntegrationWarning)
        sigma_integral, _ = integrate.quad(
            _integrand, 0, 1.0,
            limit=200, epsabs=1e-8, epsrel=1e-4)

    # R_eff = (2 pi R0)^2 / ∫ V'/eta drho
    return (2.0 * np.pi * R0)**2 / sigma_integral


def f_Vloop(I_Ohm, a, kappa, R0, Tbar, nbar, Z_eff, q95, nu_T, nu_n,
            eta_model='redl',
            rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
            Vprime_data=None):
    """
    Steady-state loop voltage: V_loop = R_eff × I_Ohm.

    Delegates to f_Reff() for the effective resistance computation.
    See f_Reff docstring for physics model and shaping treatment.

    Parameters
    ----------
    I_Ohm : float
        Ohmic (inductive) plasma current [MA].
    a, kappa, R0, Tbar, nbar, Z_eff, q95, nu_T, nu_n : see f_Reff.
    eta_model, rho_ped, n_ped_frac, T_ped_frac, Vprime_data : see f_Reff.

    Returns
    -------
    V_loop : float
        Steady-state loop voltage [V].
        Returns 0.0 if I_Ohm <= 0 (fully non-inductive scenario).
    """
    if I_Ohm <= 0:
        return 0.0
    R_eff = f_Reff(a, kappa, R0, Tbar, nbar, Z_eff, q95, nu_T, nu_n,
                   eta_model=eta_model,
                   rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                   T_ped_frac=T_ped_frac, Vprime_data=Vprime_data)
    return R_eff * I_Ohm * 1e6   # I_Ohm [MA] -> [A], R_eff [Ohm] -> V_loop [V]


#%% Currents

# NOTE: f_I_CD is defined in the CD module above (Fisch 1987 dimensional form).
# The duplicate definition that was previously here has been removed to avoid
# silent name shadowing in Python's module namespace.

def f_I_Ohm(Ip, Ib, I_CD):
    """
    Inductive (Ohmic) plasma current from the current balance.

    In steady state the total plasma current is the sum of inductive,
    bootstrap, and externally driven contributions:
        Ip = I_Ohm + I_b + I_CD
    Inverting gives I_Ohm as the remainder.  The absolute value guards
    against small numerical over-shoots that would produce a negative result.

    Parameters
    ----------
    Ip   : float  Total plasma current [MA].
    Ib   : float  Bootstrap current [MA].
    I_CD : float  Externally driven non-inductive current [MA].

    Returns
    -------
    I_Ohm : float  Inductive (Ohmic) current component [MA].
    """
    return abs(Ip - Ib - I_CD)


def f_I_CD_from_balance(Ip, Ib, I_Ohm):
    """
    Required external current drive from the current balance.

    Inverse of `f_I_Ohm`: given Ip, I_b, and I_Ohm, returns the
    non-inductive current that must be provided by CD systems:
        I_CD = Ip − I_b − I_Ohm

    Useful in steady-state scenario design where I_Ohm = 0 is imposed
    and the required CD current follows from the current budget.

    Note: this function returns the *required total non-inductive driven
    current*.  To obtain the required CD power use f_PCD with the
    appropriate figure-of-merit γ from f_etaCD_LH / f_etaCD_EC / f_etaCD_NBI.

    Parameters
    ----------
    Ip    : float  Total plasma current [MA].
    Ib    : float  Bootstrap current [MA].
    I_Ohm : float  Inductive (Ohmic) current [MA].

    Returns
    -------
    I_CD : float  Required non-inductive driven current [MA].
    """
    return abs(Ip - Ib - I_Ohm)



def f_Q_multiaux(P_fus, P_LH, P_ECRH, P_NBI, P_ICRH, P_Ohm):
    """
    Fusion gain factor with multiple auxiliary heating sources.

    Q = P_fus / (P_aux_total + P_Ohm)

    where P_aux_total = P_LH + P_ECRH + P_NBI + P_ICRH.
    Alpha power (P_fus / 5) is an internal source and is excluded.
    Ohmic power is an external inductive source and is included.

    Parameters
    ----------
    P_fus : float
        Total fusion power [MW].
    P_LH, P_ECRH, P_NBI, P_ICRH : float
        Plasma power from each auxiliary source [MW].
    P_Ohm : float
        Ohmic heating power [MW].

    Returns
    -------
    float
        Fusion gain Q (dimensionless).
    """
    P_aux = P_LH + P_ECRH + P_NBI + P_ICRH
    denom = P_aux + P_Ohm
    if denom <= 0.0:
        return np.inf
    return P_fus / denom


if __name__ == "__main__":
    # ITER Q=10 baseline: P_fus=500 MW, P_ECRH=20 MW, P_NBI=13 MW, P_Ohm~1.5 MW
    # Reference: Shimada et al., Nucl. Fusion 47 (2007) S1; PIPB Ch.6 (Gormezano 2007)
    Q_calc = f_Q_multiaux(P_fus=500.0, P_LH=20.0, P_ECRH=20.0,
                          P_NBI=13.0, P_ICRH=0.0, P_Ohm=1.5)

    print("\n── Fusion gain Q — ITER Q=10 baseline ─────────────────────────────────────")
    print(f"  {'Quantity':<30} {'D0FUS':>8}  {'ITER ref.':>10}")
    print("  " + "─" * 64)
    print(f"  {'Q  (500 MW / 54.5 MW ext)':<30} {Q_calc:>8.2f}  {'10':>10}")
    
# Bootstrap prediction
# Historical model extracted from
# Freidberg, J. P., F. J. Mangiarotti, and J. Minervini. 
# "Designing a tokamak fusion reactor—How does plasma physics fit in?."
# Physics of Plasmas 22.7 (2015).

def f_Freidberg_Ib(R0, a, κ, pbar, Ip):
    """
    
    Calculation of the bootstrap current using the Freidberg calculations
    
    Parameters
    ----------
    R0 : Major radius [m]
    a : Minor radius [m]
    κ : Elongation
    p_bar : The mean pressure [MPa]
    Ip : Plasma current [MA]
        
    Returns
    -------
    Ib : Bootstrap current [MA]
    
    """
    
    # Local poloidal field shape function b_theta(rho)
    def f_btheta(rho):
        alpha = 2.53
        num = (1 + alpha - (alpha* rho**(9. / 4.))) * np.exp(alpha* rho**(9. / 4.)) - 1 - alpha
        denom = rho * (np.exp(alpha) - 1 - alpha)
        return num / denom

    # Integrand of the radial bootstrap integral
    def integrand(rho):
        b_theta = f_btheta(rho)
        return rho**(5. / 2.) * np.sqrt(1 - rho**2) / b_theta

    # Numerical integration over normalised minor radius rho in [0, 1]
    integral, error = quad(integrand, 0, 1)

    # Numerator and denominator of the Freidberg bootstrap formula
    num = 268 * a**(5. / 2.) * κ**(5. / 4.) * pbar * integral
    denom = μ0 * np.sqrt(R0) * Ip

    # Bootstrap current [MA]
    Ib = num / denom / 1e6
    return Ib

if __name__ == "__main__":
    # ITER Q=10 baseline — Shimada et al., Nucl. Fusion 47 (2007) S1
    # p̄ = 2 n_e k_B T_e  (T_i = T_e assumed)
    _nbar_IT = 1.01;  _Tbar_IT = 8.9
    _pbar_IT = 2.0 * (_nbar_IT * 1e20) * (_Tbar_IT * 1e3 * E_ELEM) / 1e6
    Ib_Freidberg = f_Freidberg_Ib(R0=6.2, a=2.0, κ=1.75, pbar=_pbar_IT, Ip=15.0)
    
# Historical model extracted from
# D.J. Segal, A.J. Cerfon, J.P. Freidberg, "Steady state versus pulsed tokamak reactors",
# Nuclear Fusion, 61(4), 045001, 2021.

def calculate_CB(nu_J, nu_p):
    """
    
    Numerically calculates the coefficient C_B(nu_J, nu_p) according to the integral equation (35)
    from the following article:
    D.J. Segal, A.J. Cerfon, J.P. Freidberg, "Steady state versus pulsed tokamak reactors",
    Nuclear Fusion, 61(4), 045001, 2021.

    Parameters
    ----------
    nu_J : Current profile parameter
    nu_p : Pressure profile parameter

    Returns
    -------
    CB : Numerical value of the coefficient C_B
    
    """
    def integrand(x):
        """
        Integrand function of equation (35)
        """
        polynomial = (1 + (1 - 3 * nu_J) * x + nu_J * x**2)**2
        return x**(1/4) * (1 - x)**(nu_p - 1) * polynomial

    # Calculate the integral
    integral, _ = quad(integrand, 0, 1)

    # Final coefficient
    CB = integral / (1 - nu_J)**2
    return CB


def f_Segal_Ib(nu_n, nu_T, epsilon, kappa, n20, Tk, R0, I_M,
               rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
               Vprime_data=None):
    """
    Bootstrap current using the Segal-Cerfon-Freidberg analytical model.

    Source: Segal, D. J., Cerfon, A. J., & Freidberg, J. P. (2021).
    Steady state versus pulsed tokamak reactors. Nuclear Fusion, 61(4), 045001.

    Supports parabolic and parabola-with-pedestal profiles via the
    equivalent-parabola approach: effective exponents nu_n_eff, nu_T_eff are
    derived from the actual peak-to-average ratios of the profiles, then
    injected into the original Segal analytical formula.

    Parameters
    ----------
    nu_n      : Density peaking exponent (core region).
    nu_T      : Temperature peaking exponent (core region).
    epsilon   : Inverse aspect ratio a/R0.
    kappa     : Plasma elongation.
    n20       : Volume-averaged electron density [10^20 m^-3].
    Tk        : Volume-averaged temperature [keV].
    R0        : Major radius [m].
    I_M       : Plasma current [MA].
    rho_ped   : Normalised pedestal radius (1.0 = no pedestal, purely parabolic).
    n_ped_frac: n_ped / nbar.
    T_ped_frac: T_ped / Tbar.

    Returns
    -------
    I_b : Bootstrap current [MA].

    Notes
    -----
    The Segal formula (Eq. 34) reads:
        f_B = K_b * n̄ * T̄ * R₀² / I_p²
    where K_b encodes ε^2.5, κ^1.27, C_B (current/pressure profile integral)
    and the product (1+nu_n)*(1+nu_T)*(nu_n + 0.054*nu_T).

    For purely parabolic profiles (rho_ped = 1.0), the factors reduce to the
    original Segal expressions.  For pedestal profiles, equivalent exponents
    are derived from the normalised peak values:
        nu_n_eff = n(0)/nbar  - 1   [peak-to-average ratio minus one]
        nu_T_eff = T(0)/Tbar  - 1
    This is exact for parabolic profiles and provides the most natural
    generalisation for pedestal profiles at Segal's level of approximation
    (the formula itself is already a simplified fit).

    Limitation: the C_B integral still uses nu_p_eff = nu_n_eff + nu_T_eff.
    For strongly shaped pedestals, Sauter or Redl models are preferred.
    """

    if rho_ped >= 1.0:
        # Purely parabolic: use original exponents directly
        nu_n_eff = nu_n
        nu_T_eff = nu_T
    else:
        # Equivalent-parabola approach: derive effective exponents from
        # the actual normalised peak values of the profiles.
        # For a parabolic profile: X(0)/Xbar = 1 + nu_X  →  nu_X = X(0)/Xbar - 1
        n_hat0 = float(f_nprof(1.0, nu_n, 0.0, rho_ped, n_ped_frac,
                               Vprime_data))
        T_hat0 = float(f_Tprof(1.0, nu_T, 0.0, rho_ped, T_ped_frac,
                               Vprime_data))
        nu_n_eff = n_hat0 - 1.0
        nu_T_eff = T_hat0 - 1.0

    nu_p = nu_n_eff + nu_T_eff
    nu_J = 0.453 - 0.1 * (nu_p - 1.5)   # Current profile exponent [Segal Eq. 36]

    # Pressure-profile / current-profile coupling coefficient [Segal Eq. 35]
    CB = calculate_CB(nu_J, nu_p)

    # Geometric and profile coefficient K_b [Segal Eq. A15]
    K_b = 0.6099 * (1.0 + nu_n_eff) * (1.0 + nu_T_eff) * (nu_n_eff + 0.054 * nu_T_eff)
    K_b *= (epsilon**2.5) * (kappa**1.27) * CB

    # Bootstrap fraction and current [Segal Eq. 34]
    f_B = K_b * n20 * Tk * R0**2 / I_M**2
    I_b = f_B * I_M

    return I_b

if __name__ == "__main__":
    # ITER Q=10 baseline — Shimada et al., Nucl. Fusion 47 (2007) S1
    Ib_Segal = f_Segal_Ib(nu_n=0.5, nu_T=1.0, epsilon=2.0/6.2, kappa=1.75,
                          n20=1.01, Tk=8.9, R0=6.2, I_M=15.0)
    

"""
Neoclassical Bootstrap Current Model - Sauter et al. (1999)

This module implements the neoclassical bootstrap current formulas derived by
Sauter, Angioni, and Lin-Liu, valid for general axisymmetric equilibria and
arbitrary collisionality regimes.

References
----------
[1] O. Sauter, C. Angioni, Y.R. Lin-Liu,
    "Neoclassical conductivity and bootstrap current formulas for general
    axisymmetric equilibria and arbitrary collisionality regime",
    Physics of Plasmas 6(7), 2834-2839 (1999).
    DOI: 10.1063/1.873240

[2] O. Sauter, C. Angioni, Y.R. Lin-Liu,
    "Erratum: Neoclassical conductivity and bootstrap current formulas [...]",
    Physics of Plasmas 9(12), 5140 (2002).
    DOI: 10.1063/1.1517052
    CRITICAL CORRECTION: Sign of coefficient 0.315 in Eq. (17b) changed
    from negative to positive.

[3] O. Sauter,
    "Geometric formulas for system codes including the effect of negative
    triangularity",
    Fusion Engineering and Design 112, 633-645 (2016).
    DOI: 10.1016/j.fusengdes.2016.04.033
    (Trapped fraction approximation formulas)

[4] Y.R. Lin-Liu, R.L. Miller,
    "Upper and lower bounds of the effective trapped particle fraction
    in general tokamak equilibria",
    Physics of Plasmas 2(5), 1666-1668 (1995).
    DOI: 10.1063/1.871315

Physics Background
------------------
The bootstrap current arises from the collisional coupling between trapped
and passing particles in a toroidal plasma with pressure gradients. The
neoclassical theory provides expressions for the parallel current density
as a function of:
- Trapped particle fraction (geometry)
- Electron and ion collisionality (collision frequency / bounce frequency)
- Effective charge Zeff
- Pressure, density, and temperature gradients

The Sauter model parameterizes these effects through transport coefficients
L31, L32, L34, and alpha, which are fitted to Fokker-Planck code results
(CQL3D and CQLP) within 5% accuracy.

Implementation Notes
--------------------
This module provides:
1. Local coefficients (L31, L32, L34, alpha) as functions of trapped fraction
   and collisionality - for use in 1D transport codes
2. An integrated 0D bootstrap current estimate assuming parabolic profiles -
   suitable for system codes like D0FUS

"""

# ==============================================================================
# Internal Functions — Profile logarithmic gradients (used by Sauter and Redl)
# ==============================================================================
# Internal Functions (Sauter model)
# ==============================================================================

def _trapped_fraction(epsilon, kappa):
    """Trapped particle fraction with elongation correction."""
    eps_eff = epsilon / (1.0 + epsilon * kappa)
    sqrt_eps = np.sqrt(eps_eff)
    return 1.46 * sqrt_eps / (1.0 + 1.46 * sqrt_eps)


def _nu_e_star(n_e, T_e, q, R0, epsilon, Z_eff):
    """Electron collisionality [Eq. 18b]."""
    ln_Lambda = 31.3 - np.log(np.sqrt(n_e) / T_e)
    return 6.921e-18 * q * R0 * n_e * Z_eff * ln_Lambda / (T_e**2 * epsilon**1.5)


def _nu_i_star(n_i, T_i, q, R0, epsilon):
    """Ion collisionality [Eq. 18c]."""
    ln_Lambda = 30.0 - np.log(np.sqrt(n_i) / T_i**1.5)
    return 4.90e-18 * q * R0 * n_i * ln_Lambda / (T_i**2 * epsilon**1.5)


def _F31(X, Z):
    """Polynomial F31 [Eq. 14a]."""
    return ((1.0 + 1.4/(Z + 1.0)) * X 
            - (1.9/(Z + 1.0)) * X**2 
            + (0.3/(Z + 1.0)) * X**3 
            + (0.2/(Z + 1.0)) * X**4)


def _L31(f_t, nu_e, Z):
    """Coefficient L31 [Eq. 14]."""
    sqrt_nu = np.sqrt(nu_e)
    f_t_eff = f_t / (1.0 + (1.0 - 0.1*f_t)*sqrt_nu + 0.5*(1.0 - f_t)*nu_e/Z)
    return _F31(f_t_eff, Z)


def _L32(f_t, nu_e, Z):
    """Coefficient L32 = L32_ee + L32_ei [Eq. 15]."""
    sqrt_nu = np.sqrt(nu_e)
    sqrt_Z = np.sqrt(Z)
    
    # Electron-electron
    f_t_ee = f_t / (1.0 + 0.26*(1.0 - f_t)*sqrt_nu + 0.18*(1.0 - 0.37*f_t)*nu_e/sqrt_Z)
    X = f_t_ee
    F32_ee = ((0.05 + 0.62*Z)/(Z*(1.0 + 0.44*Z)) * (X - X**4) +
              1.0/(1.0 + 0.22*Z) * (X**2 - X**4 - 1.2*(X**3 - X**4)) +
              1.2/(1.0 + 0.5*Z) * X**4)
    
    # Electron-ion
    f_t_ei = f_t / (1.0 + (1.0 + 0.6*f_t)*sqrt_nu + 0.85*(1.0 - 0.37*f_t)*nu_e*(1.0 + Z))
    Y = f_t_ei
    F32_ei = (-(0.56 + 1.93*Z)/(Z*(1.0 + 0.44*Z)) * (Y - Y**4) +
              4.95/(1.0 + 2.48*Z) * (Y**2 - Y**4 - 0.55*(Y**3 - Y**4)) -
              1.2/(1.0 + 0.5*Z) * Y**4)
    
    return F32_ee + F32_ei


def _L34(f_t, nu_e, Z):
    """Coefficient L34 [Eq. 16]."""
    sqrt_nu = np.sqrt(nu_e)
    f_t_eff = f_t / (1.0 + (1.0 - 0.1*f_t)*sqrt_nu + 0.5*(1.0 - 0.5*f_t)*nu_e/Z)
    return _F31(f_t_eff, Z)


def _alpha(f_t, nu_i):
    """Ion flow coefficient [Eq. 17] WITH ERRATA +0.315."""
    alpha0 = -1.17 * (1.0 - f_t) / (1.0 - 0.22*f_t - 0.19*f_t**2)
    sqrt_nu = np.sqrt(nu_i)
    f_t_6 = f_t**6
    nu_sq = nu_i**2
    # NOTE: (alpha0 + 0.25*...*sqrt_nu) is the FULL numerator inside the
    # 1/(1+0.5*sqrt_nu) bracket — see NEOS neobscoeffmod.f90 lines 104-106.
    numer = (alpha0 + 0.25*(1.0 - f_t**2)*sqrt_nu) / (1.0 + 0.5*sqrt_nu) + 0.315*nu_sq*f_t_6
    denom = 1.0 + 0.15*nu_sq*f_t_6
    return numer / denom


# ── Safety factor radial profile ─────────────────────────────────────────────
# NOTE: moved here (before f_Sauter_Ib / f_Redl_Ib) so that the inline
# ``if __name__ == "__main__"`` test blocks can call these functions at
# module-level execution time.  Topically belongs with f_q95, but the
# forward dependency forces early placement.

def f_q_profile(rho, q95=3.0, rho95=0.95, alpha_J=1.5):
    """
    Safety-factor profile derived from the cylindrical Ampère relation
    with a prescribed current density j(ρ) ∝ (1 − ρ²)^αJ.

    Physical basis
    --------------
    Starting from the current density ansatz

        j(ρ) = j₀ (1 − ρ²)^αJ

    the enclosed current fraction follows by integration:

        I(ρ)/Ip = 1 − (1 − ρ²)^(αJ + 1)

    and the cylindrical safety factor is:

        q(ρ) = q_edge × ρ² / [1 − (1 − ρ²)^(αJ + 1)]

    where q_edge is determined by the constraint q(ρ₉₅) = q₉₅.

    On axis (L'Hôpital): q(0) = q_edge / (αJ + 1).

    This profile is self-consistent with the j-profile model used in
    f_Reff() for the loop voltage calculation and with f_li() for the
    internal inductance, ensuring coherent physics across the code.

    Choice of αJ
    -------------
    The current profile peaking exponent αJ is the solution of the
    resistive current relaxation equation in the limit of a single
    dominant diffusion eigenmode:

        αJ(t) = αJ_max × [1 − exp(−t_burn / τ_R)]

    where αJ_max = 3/2 × νT is the resistive equilibrium limit
    (j_eq ∝ σ ∝ T^{3/2}, with T ∝ (1−ρ²)^νT), and τ_R = μ₀ a² / η
    is the resistive diffusion time.  This is the exact solution of the
    0th-order ODE for the modal amplitude; higher spatial eigenmodes
    decay as τ_R/n² and are negligible after ~1 s.

    Typical values:
      αJ = 0.0 : flat current profile (start of burn, or full CD)
      αJ = 1.0 : moderately peaked (ITER Q=10, t_burn ≈ 10 min)
      αJ = 1.5 : default — intermediate between ITER and DEMO,
                  consistent with standard sawtoothing H-mode
                  (Uckan IPDG89, PROCESS systems code)
      αJ = 2.5 : strongly peaked (EU-DEMO, 2h burn, νT ≈ 2.8)

    The internal inductance l_i increases monotonically with αJ:
        l_i(0) = 0.50, l_i(1.5) ≈ 1.08, l_i(3) ≈ 1.55.
    Computed numerically by f_li().
    Note: the formula l_i = (αJ+1)/(2αJ+1) sometimes cited in textbooks
    gives the WRONG direction (decreasing with αJ).  Do not use it.

    Parameters
    ----------
    rho : float or ndarray
        Normalised radial coordinate ∈ [0, 1].
    q95 : float
        Safety factor at ρ = ρ₉₅ (default 3.0).
    rho95 : float
        Radial position of the 95%% flux surface (default 0.95).
    alpha_J : float
        Current density peaking exponent (default 1.5).

    Returns
    -------
    q : float or ndarray
        Safety factor profile q(ρ).

    References
    ----------
    Wesson, Tokamaks, 4th ed. (2011) ch. 3 — cylindrical j–q relation.
    Uckan et al., ITER Physics Design Guidelines, IAEA ITER Doc. Series
        No. 10 (1990) — reference j ∝ (1−ρ²)^αJ parameterisation.
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054 — PROCESS systems code
        (§4.1, §18): same current density ansatz and derived q-profile.
    Polevoi et al., Nucl. Fusion 55 (2015) 063019 — ITER V_loop
        validation of the prescribed j-profile model.
    """
    rho = np.asarray(rho, dtype=float)
    ap1 = alpha_J + 1.0

    # Normalisation: q_edge such that q(rho95) = q95
    I_frac_95 = 1.0 - (1.0 - rho95**2)**ap1
    q_edge = q95 * I_frac_95 / rho95**2

    # Enclosed current fraction: I(rho)/Ip = 1 - (1-rho^2)^(alpha_J+1)
    rho2 = rho**2
    I_frac = 1.0 - (np.maximum(1.0 - rho2, 0.0))**ap1

    # q(rho) = q_edge * rho^2 / I_frac, with L'Hôpital limit at rho=0
    q = np.where(I_frac > 1e-12, q_edge * rho2 / I_frac, q_edge / ap1)
    return q


# ==============================================================================
# Main Function
# ==============================================================================

def f_Sauter_Ib(R0, a, kappa, B0, nbar, Tbar, q95, Z_eff, nu_n, nu_T, n_rho=100,
                rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                Vprime_data=None, kappa_95=None, rho_95=0.95,
                return_profile=False):
    """
    Bootstrap current using Sauter neoclassical model (vectorised).

    Fully vectorised implementation — no Python loops.  All Sauter coefficient
    functions (_L31, _L32, _L34, _alpha) already accept numpy arrays.

    Parameters
    ----------
    R0, a, kappa, B0, nbar, Tbar, q95, Z_eff : float
        Plasma geometry, field, density, temperature, safety factor, Zeff.
    nu_n, nu_T : float
        Profile peaking exponents.
    n_rho : int
        Number of radial grid points in the core region (default 100).
    rho_ped, n_ped_frac, T_ped_frac : float
        Pedestal parameters.
    Vprime_data : tuple or None
        If provided, uses D0FUS mode with local kappa(rho).
    kappa_95 : float or None
        Elongation at 95% flux surface.
    rho_95 : float
        Position of the 95% flux surface (default 0.95).

    Returns
    -------
    I_bs : float
        Bootstrap current [MA]

    References
    ----------
    Sauter, Angioni & Lin-Liu, Phys. Plasmas 6 (1999) 2834.
    Sauter, Angioni & Lin-Liu, Phys. Plasmas 9 (2002) 5140 — errata.
    """
    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa)

    I_psi = R0 * B0

    # Radial grid — core + refined pedestal edge
    rho_core = np.linspace(0.05, rho_ped, n_rho, endpoint=False)
    rho_edge = np.linspace(rho_ped, 0.99, 3 * n_rho)
    rho_arr  = np.concatenate([rho_core, rho_edge])

    use_miller = (Vprime_data is not None)

    # ── Vectorised profile evaluation ─────────────────────────────────────
    T_arr = f_Tprof(Tbar, nu_T, rho_arr, rho_ped, T_ped_frac, Vprime_data)
    n_arr = f_nprof(nbar, nu_n, rho_arr, rho_ped, n_ped_frac, Vprime_data)

    # Numerical logarithmic gradients [m^-1]
    dT_drho = np.gradient(T_arr, rho_arr)
    dn_drho = np.gradient(n_arr, rho_arr)
    dln_T = np.where(T_arr > 0.01, dT_drho / (T_arr * a), 0.0)
    dln_n = np.where(n_arr > 1e-3, dn_drho / (n_arr * a), 0.0)

    # ── Vectorised local quantities ───────────────────────────────────────
    eps_arr = rho_arr * a / R0
    # Use q95 as representative safety factor for collisionality.
    # nu*_e ~ q*R/eps^{3/2}: the eps variation dominates over q(rho),
    # making the bootstrap integral insensitive to the q-profile shape.
    q_arr   = np.full_like(rho_arr, q95)

    # SI units
    n_e  = n_arr * 1e20           # [m^-3]
    T_eV = T_arr * 1e3            # [eV]
    n_i  = n_e / Z_eff

    # Pressure [Pa]
    p_e   = n_e * T_eV * E_ELEM
    p_i   = n_i * T_eV * E_ELEM
    p_tot = p_e + p_i
    R_pe  = np.where(p_tot > 0, p_e / p_tot, 0.5)

    # Local elongation: kappa(rho) in D0FUS mode, kappa_edge in Academic
    if use_miller:
        kappa_arr = kappa_profile(rho_arr, kappa, kappa_95, rho_95)
    else:
        kappa_arr = np.full_like(rho_arr, kappa)

    # Trapped fraction and collisionalities (vectorised)
    f_t  = _trapped_fraction(eps_arr, kappa_arr)
    nu_e = _nu_e_star(n_e, T_eV, q_arr, R0, eps_arr, Z_eff)
    nu_i_arr = _nu_i_star(n_i, T_eV, q_arr, R0, eps_arr)

    # Sauter coefficients (vectorised — all use only np operations)
    L31 = _L31(f_t, nu_e, Z_eff)
    L32 = _L32(f_t, nu_e, Z_eff)
    L34 = _L34(f_t, nu_e, Z_eff)
    alp = _alpha(f_t, nu_i_arr)

    # Logarithmic gradients
    dln_p  = dln_n + dln_T
    dln_Ti = dln_T   # Ti = Te assumed

    # Bootstrap coefficient [Sauter Eq. 5]
    C_bs = L31 * dln_p + L32 * R_pe * dln_T + L34 * alp * (1.0 - R_pe) * dln_Ti

    # Local j_bs [A/m^2]
    B_sq = B0**2 * (1.0 + eps_arr**2 / 2.0)
    j_bs = -I_psi * p_tot * C_bs / B_sq

    # Grid spacing for area-weighted integration
    drho = np.zeros_like(rho_arr)
    drho[0]    = rho_arr[1] - rho_arr[0]
    drho[-1]   = rho_arr[-1] - rho_arr[-2]
    drho[1:-1] = (rho_arr[2:] - rho_arr[:-2]) / 2.0

    # Area element: dA = 2π ρ a² κ_local dρ
    dA = 2.0 * np.pi * rho_arr * a**2 * kappa_arr * drho

    # Mask out unphysical points (eps too small, n or T too low)
    valid = (eps_arr >= 0.01) & (n_arr >= 1e-3) & (T_arr >= 0.1)
    j_bs = np.where(valid, j_bs, 0.0)

    I_bs = np.sum(j_bs * dA) / 1e6   # [MA]

    if return_profile:
        return {'I_bs': I_bs, 'rho': rho_arr, 'j_bs': j_bs,
                'dA': dA, 'kappa_arr': kappa_arr, 'q_arr': q_arr,
                'drho': drho}
    return I_bs


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    # ITER Q=10 baseline — Shimada et al., Nucl. Fusion 47 (2007) S1
    # H-mode pedestal profile consistent with blocks §2 / §6
    _bs_kw = dict(R0=6.2, a=2.0, kappa=1.75, B0=5.3,
                  nbar=1.01, Tbar=8.9, q95=3.0,
                  Z_eff=1.65, nu_n=0.1, nu_T=1.0,
                  rho_ped=0.94, n_ped_frac=0.90, T_ped_frac=0.40)
    Ib_Sauter = f_Sauter_Ib(**_bs_kw)


# ==============================================================================
# Sauter (1999) coefficient crosscheck vs NEOS reference implementation
# ==============================================================================
#
# Cross-verification of D0FUS _L31, _L32, _L34, _alpha against a line-by-line
# Python transliteration of Sauter's own Fortran code (neobscoeffmod.f90,
# subroutine neobscoeff_s, lines 54-109).
#
# Source: https://gitlab.epfl.ch/spc/public/NEOS  (Apache 2.0, © 2025 SPC-EPFL)
#         File: F90/neobscoeffmod.f90
#
# References
# ----------
# [1] Sauter, Angioni & Lin-Liu, Phys. Plasmas 6 (1999) 2834.
# [2] Sauter, Angioni & Lin-Liu, Phys. Plasmas 9 (2002) 5140 — errata.
# ==============================================================================

def _neos_ref(ft, ZZ, znuestar=0.0, znuistar=0.0):
    """
    NEOS reference: transliterated from neobscoeffmod.f90 lines 73-106.

    Variable names are kept identical to the Fortran source.
    Returns (L31, L32, L34, ALFA).
    """
    zsqnuest = np.sqrt(znuestar)
    # Effective trapped fractions (lines 78-86)
    zft31eff    = ft / (1 + (1 - 0.1*ft)*zsqnuest + 0.5*(1 - ft)*znuestar/ZZ)
    zft32ee_eff = ft / (1 + 0.26*(1 - ft)*zsqnuest + 0.18*(1 - 0.37*ft)*znuestar/np.sqrt(ZZ))
    zft32ei_eff = ft / (1 + (1 + 0.6*ft)*zsqnuest + 0.85*(1 - 0.37*ft)*znuestar*(1 + ZZ))
    zft34eff    = ft / (1 + (1 - 0.1*ft)*zsqnuest + 0.5*(1 - 0.5*ft)*znuestar/ZZ)
    zalfa0      = -1.17*(1 - ft) / (1 - 0.22*ft - 0.19*ft**2)
    # L31 (lines 91-92)
    zeffp1 = ZZ + 1
    L31 = zft31eff * ((1 + 1.4/zeffp1)
          - zft31eff*(1.9/zeffp1 - zft31eff*(0.3/zeffp1 + 0.2/zeffp1*zft31eff)))
    # L32 (lines 93-99)
    L32 = ((0.05 + 0.62*ZZ)/ZZ/(1 + 0.44*ZZ)*(zft32ee_eff - zft32ee_eff**4)
         + zft32ee_eff**2*(1 - 1.2*zft32ee_eff + 0.2*zft32ee_eff**2)/(1 + 0.22*ZZ)
         - (0.56 + 1.93*ZZ)/ZZ/(1 + 0.44*ZZ)*(zft32ei_eff - zft32ei_eff**4)
         + zft32ei_eff**2*(1 - 0.55*zft32ei_eff - 0.45*zft32ei_eff**2)*4.95/(1 + 2.48*ZZ)
         + 1.2/(1 + 0.5*ZZ)*(zft32ee_eff**4 - zft32ei_eff**4))
    # L34 (lines 100-101)
    L34 = zft34eff * ((1 + 1.4/zeffp1)
          - zft34eff*(1.9/zeffp1 - zft34eff*(0.3/zeffp1 + 0.2/zeffp1*zft34eff)))
    # ALFA (lines 104-106)
    zsqnui = np.sqrt(znuistar)
    znui2ft6 = znuistar**2 * ft**6
    ALFA = ((zalfa0 + 0.25*(1 - ft**2)*zsqnui)/(1 + 0.5*zsqnui) + 0.315*znui2ft6) \
         / (1 + 0.15*znui2ft6)
    return L31, L32, L34, ALFA


if __name__ == "__main__":
    
    DETAILED_DEBUG = False
    
    if DETAILED_DEBUG == True:

        print("\n" + "=" * 90)
        print("  D0FUS vs NEOS (Sauter, EPFL/SPC) — coefficient crosscheck")
        print("  Ref: https://gitlab.epfl.ch/spc/public/NEOS  F90/neobscoeffmod.f90")
        print("=" * 90)
    
        # Test cases: (ft, Zeff, nue*, nui*, description)
        _cases = [
            (0.50,  1.0,   0.0,    0.0,   "banana, Z=1"),
            (0.50,  2.0,   0.0,    0.0,   "banana, Z=2"),
            (0.30,  1.0,   0.0,    0.0,   "banana, low ft"),
            (0.65,  1.0,   0.0,    0.0,   "banana, high ft"),
            (0.50,  1.0,   0.1,    0.1,   "finite ν*"),
            (0.50,  1.0,   1.0,    1.0,   "plateau"),
            (0.50,  1.0,   10.0,   10.0,  "high ν*"),
            (0.65,  1.65,  0.01,   0.004, "ITER-like"),
        ]
    
        _n_ok = 0
        _n_tot = 0
    
        print(f"\n  {'Case':<20s}  {'L31':>9s}  {'L32':>9s}  {'L34':>9s}  {'α':>9s}  Status")
        print("  " + "─" * 76)
    
        for ft, Z, nue, nui, desc in _cases:
            nL31, nL32, nL34, nALF = _neos_ref(ft, Z, nue, nui)
            dL31 = _L31(ft, nue, Z)
            dL32 = _L32(ft, nue, Z)
            dL34 = _L34(ft, nue, Z)
            dALF = _alpha(ft, nui)
    
            # Max relative error across all 4 coefficients
            errs = []
            for dv, nv in [(dL31, nL31), (dL32, nL32), (dL34, nL34), (dALF, nALF)]:
                _n_tot += 1
                if abs(nv) > 1e-8:
                    errs.append(abs(dv - nv) / abs(nv) * 100)
                else:
                    errs.append(abs(dv - nv) * 1e6)
            all_ok = all(e < 0.01 for e in errs)
            _n_ok += 4 if all_ok else sum(1 for e in errs if e < 0.01)
            tag = "OK" if all_ok else f"FAIL (max Δ={max(errs):.2f}%)"
            print(f"  {desc:<20s}  {dL31:>+9.5f}  {dL32:>+9.5f}  {dL34:>+9.5f}  {dALF:>+9.5f}  {tag}")
    
        if _n_ok == _n_tot:
            print(f"  ALL {_n_tot} checks passed — D0FUS matches NEOS exactly.")
        else:
            print(f"\n  {_n_ok}/{_n_tot} passed, {_n_tot - _n_ok} FAILED.")
        print("=" * 90)


"""
Neoclassical Bootstrap Current Model - Redl et al. (2021)

This module implements the revised neoclassical bootstrap current formulas 
derived by Redl, Angioni, Belli, and Sauter, which improve upon the original 
Sauter model by fitting to the modern drift-kinetic solver NEO.

References
----------
[1] A. Redl, C. Angioni, E. Belli, O. Sauter, ASDEX Upgrade Team, EUROfusion MST1 Team,
    "A new set of analytical formulae for the computation of the bootstrap 
    current and the neoclassical conductivity in tokamaks",
    Physics of Plasmas 28(2), 022502 (2021).
    DOI: 10.1063/5.0012664
    
[2] E. Belli, J. Candy,
    "Kinetic calculation of neoclassical transport including self-consistent 
    electron and impurity dynamics",
    Plasma Physics and Controlled Fusion 54(1), 015015 (2012).
    DOI: 10.1088/0741-3335/54/1/015015
    (NEO code reference)

[3] O. Sauter, C. Angioni, Y.R. Lin-Liu,
    "Neoclassical conductivity and bootstrap current formulas for general
    axisymmetric equilibria and arbitrary collisionality regime",
    Physics of Plasmas 6(7), 2834-2839 (1999).
    DOI: 10.1063/1.873240
    (Original Sauter model - basis for Redl)

[4] M. Landreman, S. Buller, M. Drevlak,
    "Optimization of quasi-symmetric stellarators with self-consistent 
    bootstrap current and energetic particle confinement",
    Physics of Plasmas 29(8), 082501 (2022).
    DOI: 10.1063/5.0098166
    (Application and validation of Redl formula)

Physics Background
------------------
The Redl model is a revision of the Sauter model for computing the neoclassical
bootstrap current in tokamak plasmas. The key improvements are:

1. NEO Code Fitting: The coefficients are fitted to results from the NEO 
   drift-kinetic solver (Belli & Candy 2008, 2012), which is more accurate 
   than the older CQL3D and CQLP codes used by Sauter.

2. High Collisionality Accuracy: The Sauter model is known to be inaccurate 
   at electron collisionalities ν*_e > 1, which limits its applicability in 
   tokamak edge pedestals. The Redl model provides improved accuracy across 
   all collisionality regimes.

3. Impurity Treatment: Better handling of impurity effects, important for 
   high-Z wall materials (tungsten) in modern tokamaks like ASDEX Upgrade, 
   JET-ILW, and ITER.

4. Same Structure: The model retains the same analytical structure as 
   Sauter, using three key neoclassical parameters:
   - Trapped particle fraction f_t (geometry)
   - Collisionality ν* (collision frequency / bounce frequency)
   - Effective charge Z_eff

The bootstrap current density is given by the same master equation as Sauter:

    <j_∥ B> = σ_neo <E_∥ B> - I(ψ) p [L31 ∂ln(n_e)/∂ψ 
                                      + R_pe (L31 + L32) ∂ln(T_e)/∂ψ
                                      + (1-R_pe)(1 + L34/L31 · α) L31 ∂ln(T_i)/∂ψ]

where:
- I(ψ) = R·B_φ is the flux function
- R_pe = p_e/p is the electron pressure fraction
- L31, L32, L34 are transport coefficients (density/temperature gradients)
- α is the ion parallel flow coefficient

The Redl model provides new polynomial fits for L31, L32, L34, and α as 
functions of f_t, ν*_e, ν*_i, and Z_eff, with improved accuracy compared 
to the Sauter fits.

Validation
----------
The Redl model has been validated against:
- NEO drift-kinetic calculations (within ~2% in core, ~5% in pedestal)
- ASDEX Upgrade experimental profiles
- SFINCS calculations for quasi-symmetric stellarators

The model shows significant improvement over Sauter in:
- H-mode pedestal region (high collisionality)
- Plasmas with high Z_eff (impurity-rich)
- Low aspect ratio devices (spherical tokamaks)

Implementation Notes
--------------------

The numerical coefficients used here are derived from the polynomial fits 
presented in Redl et al. (2021), Tables I-III and Appendix equations.

"""

# ==============================================================================
# Internal Functions (Redl model — NEO-fitted coefficients)
#
# Redl, Angioni, Belli & Sauter, Phys. Plasmas 28, 022502 (2021).
# Equations (10)–(21).  Same analytical structure as Sauter (1999), but
# ALL numerical coefficients re-fitted to the drift-kinetic solver NEO.
# Key improvements: high-collisionality accuracy (pedestal), Z_eff
# dependence in α₀, and revised high-ν* limit α → 0.5 (not 2.1).
#
# _trapped_fraction, _nu_e_star, _nu_i_star are shared with Sauter.
# ==============================================================================


def _F31_Redl(X, Z):
    """Polynomial F31 for L31 [Redl Eq. 10].

    Replaces Sauter's 1/(Z+1) denominators with 1/(Z^1.2 − 0.71)
    and uses completely different numerical prefactors.
    """
    Zfac = Z**1.2 - 0.71
    return ((1.0 + 0.15 / Zfac) * X
            - (0.22 / Zfac) * X**2
            + (0.01 / Zfac) * X**3
            + (0.06 / Zfac) * X**4)


def _L31_Redl(f_t, nu_e, Z):
    """L31 coefficient [Redl Eqs. 10–11]."""
    sqrt_nu = np.sqrt(nu_e)
    # Effective trapped fraction [Eq. 11] — different structure from Sauter
    numer_term1 = 0.67 * (1.0 - 0.7 * f_t) * sqrt_nu / (0.56 + 0.44 * Z)
    numer_term2 = ((0.52 + 0.086 * sqrt_nu) * (1.0 + 0.87 * f_t) * nu_e
                   / (1.0 + 1.13 * np.sqrt(np.maximum(Z - 1.0, 0.0))))
    f_t_eff = f_t / (1.0 + numer_term1 + numer_term2)
    return _F31_Redl(f_t_eff, Z)


def _L32_Redl(f_t, nu_e, Z):
    """L32 = F32_ee + F32_ei [Redl Eqs. 12–16]."""
    sqrt_nu = np.sqrt(nu_e)

    # ── Electron-electron: F32_ee [Eqs. 13–14] ──
    # Effective trapped fraction [Eq. 14] — new Z^2 and f_t² terms
    Zm1 = np.maximum(Z - 1.0, 0.0)
    ee_t1 = 0.23 * (1.0 - 0.96 * f_t) * sqrt_nu / np.sqrt(Z)
    ee_t2 = (0.13 * (1.0 - 0.38 * f_t) * nu_e
             / (Z**2 * np.sqrt(1.0 + 2.0 * np.sqrt(Zm1))))
    ee_t3 = f_t**2 * np.sqrt((0.075 + 0.25 * Zm1**2) * nu_e)
    f_t_ee = f_t / (1.0 + ee_t1 + ee_t2 + ee_t3)

    X = f_t_ee
    # F32_ee polynomial [Eq. 13] — different denominators from Sauter
    F32_ee = ((0.1 + 0.6 * Z) / (Z * (0.77 + 0.63 * (1.0 + Zm1**1.1))) * (X - X**4)
              + 0.7 / (1.0 + 0.2 * Z) * (X**2 - X**4 - 1.2 * (X**3 - X**4))
              + 1.3 / (1.0 + 0.5 * Z) * X**4)

    # ── Electron-ion: F32_ei [Eqs. 15–16] ──
    # Effective trapped fraction [Eq. 16]
    ei_t1 = 0.87 * (1.0 + 0.39 * f_t) * sqrt_nu / (1.0 + 2.95 * Zm1**2)
    ei_t2 = 1.53 * (1.0 - 0.37 * f_t) * nu_e * (2.0 + 0.375 * Zm1)
    f_t_ei = f_t / (1.0 + ei_t1 + ei_t2)

    Y = f_t_ei
    # F32_ei polynomial [Eq. 15]
    F32_ei = (-(0.4 + 1.93 * Z) / (Z * (0.8 + 0.6 * Z)) * (Y - Y**4)
              + 5.5 / (1.5 + 2.0 * Z) * (Y**2 - Y**4 - 0.8 * (Y**3 - Y**4))
              - 1.3 / (1.0 + 0.5 * Z) * Y**4)

    return F32_ee + F32_ei


def _alpha_Redl(f_t, nu_i, Z):
    """Ion flow coefficient α [Redl Eqs. 20–21].

    Key differences from Sauter:
    - α₀ now depends on Z_eff (Eq. 20)
    - High-ν* limit revised: α → 0.5 (not 2.1) due to ion-electron coupling
    - Coefficient signs and magnitudes completely different
    """
    Zm1 = np.maximum(Z - 1.0, 0.0)

    # α₀ with Z_eff dependence [Eq. 20]
    # At Z=1: -(0.62/0.53) = -1.170 ≈ Sauter's -1.17 (recovers banana limit)
    alpha0 = (-(0.62 + 0.055 * Zm1) / (0.53 + 0.17 * Zm1)
              * (1.0 - f_t) / (1.0 - (0.31 - 0.065 * Zm1) * f_t - 0.25 * f_t**2))

    # Collisionality-dependent formula [Eq. 21]
    sqrt_nu = np.sqrt(nu_i)
    f_t_6 = f_t**6
    nu_sq = nu_i**2

    numer = ((alpha0 + 0.7 * Z * f_t**0.5 * sqrt_nu) / (1.0 + 0.18 * sqrt_nu)
             - 0.002 * nu_sq * f_t_6)
    denom = 1.0 + 0.004 * nu_sq * f_t_6

    return numer / denom


# ==============================================================================
# Main Function
# ==============================================================================

def f_Redl_Ib(R0, a, kappa, B0, nbar, Tbar, q95, Z_eff, nu_n, nu_T, n_rho=100,
              rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
              Vprime_data=None, kappa_95=None, rho_95=0.95,
              return_profile=False):
    """
    Bootstrap current using Redl neoclassical model (2021).

    Improved accuracy over Sauter model, especially at high collisionality
    (pedestal region) and for plasmas with impurities.
    Supports parabolic and parabola-with-pedestal profile models.

    Parameters
    ----------
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    kappa : float
        Plasma elongation at the LCFS (kappa_edge).
    B0 : float
        On-axis toroidal field [T]
    nbar : float
        Volume-averaged electron density [1e20 m^-3]
    Tbar : float
        Volume-averaged temperature [keV]
    q95 : float
        Safety factor at psi_N = 0.95
    Z_eff : float
        Effective ion charge
    nu_n : float
        Density peaking exponent (core)
    nu_T : float
        Temperature peaking exponent (core)
    n_rho : int
        Number of radial integration points (default 100)
    rho_ped    : float  Normalised pedestal radius (1.0 = no pedestal).
    n_ped_frac : float  n_ped / nbar.
    T_ped_frac : float  T_ped / Tbar.
    Vprime_data : tuple or None, optional
        If provided, activates D0FUS mode: area element dA and trapped
        fraction use the PCHIP local kappa(rho) instead of constant kappa_edge.
    kappa_95 : float or None, optional
        Elongation at the 95% flux surface.  Used only in D0FUS mode.
        Defaults to f_Kappa_95(kappa) (ITER 1989 guideline: kappa_edge / 1.12).
    rho_95 : float, optional
        Normalised position of the 95% flux surface (default 0.95).

    Returns
    -------
    I_bs : float
        Bootstrap current [MA]

    References
    ----------
    Redl et al., Phys. Plasmas 28, 022502 (2021).
    Ball & Parra, PPCF 57 (2015) 035006 — kappa radial penetration.
    """
    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa)

    I_psi = R0 * B0

    # Radial grid — core + refined pedestal edge
    rho_core = np.linspace(0.05, rho_ped, n_rho, endpoint=False)
    rho_edge = np.linspace(rho_ped, 0.99, 3 * n_rho)
    rho_arr  = np.concatenate([rho_core, rho_edge])

    use_miller = (Vprime_data is not None)

    # ── Vectorised profile evaluation ─────────────────────────────────────
    T_arr = f_Tprof(Tbar, nu_T, rho_arr, rho_ped, T_ped_frac, Vprime_data)
    n_arr = f_nprof(nbar, nu_n, rho_arr, rho_ped, n_ped_frac, Vprime_data)

    # Numerical logarithmic gradients [m^-1]
    dT_drho = np.gradient(T_arr, rho_arr)
    dn_drho = np.gradient(n_arr, rho_arr)
    dln_T = np.where(T_arr > 0.01, dT_drho / (T_arr * a), 0.0)
    dln_n = np.where(n_arr > 1e-3, dn_drho / (n_arr * a), 0.0)

    # ── Vectorised local quantities ───────────────────────────────────────
    eps_arr = rho_arr * a / R0
    # Use q95 as representative safety factor for collisionality.
    # See f_Sauter_Ib docstring for rationale.
    q_arr   = np.full_like(rho_arr, q95)

    # SI units
    n_e  = n_arr * 1e20           # [m^-3]
    T_eV = T_arr * 1e3            # [eV]
    n_i  = n_e / Z_eff

    # Pressure [Pa]
    p_e   = n_e * T_eV * E_ELEM
    p_i   = n_i * T_eV * E_ELEM
    p_tot = p_e + p_i
    R_pe  = np.where(p_tot > 0, p_e / p_tot, 0.5)

    # Local elongation: kappa(rho) in D0FUS mode, kappa_edge in Academic
    if use_miller:
        kappa_arr = kappa_profile(rho_arr, kappa, kappa_95, rho_95)
    else:
        kappa_arr = np.full_like(rho_arr, kappa)

    # Trapped fraction and collisionalities (vectorised)
    f_t  = _trapped_fraction(eps_arr, kappa_arr)
    nu_e = _nu_e_star(n_e, T_eV, q_arr, R0, eps_arr, Z_eff)
    nu_i_arr = _nu_i_star(n_i, T_eV, q_arr, R0, eps_arr)

    # Redl coefficients (vectorised — all use only np operations)
    # Redl Eq. 19: L34 = L31 (simplification validated by NEO)
    L31 = _L31_Redl(f_t, nu_e, Z_eff)
    L32 = _L32_Redl(f_t, nu_e, Z_eff)
    L34 = L31                                    # Redl Eq. 19
    alp = _alpha_Redl(f_t, nu_i_arr, Z_eff)     # Redl α depends on Z_eff

    # Logarithmic gradients
    dln_p  = dln_n + dln_T
    dln_Ti = dln_T   # Ti = Te assumed

    # Bootstrap coefficient [Redl Eq. 5]
    C_bs = L31 * dln_p + L32 * R_pe * dln_T + L34 * alp * (1.0 - R_pe) * dln_Ti

    # Local j_bs [A/m^2]
    B_sq = B0**2 * (1.0 + eps_arr**2 / 2.0)
    j_bs = -I_psi * p_tot * C_bs / B_sq

    # Grid spacing for area-weighted integration
    drho = np.zeros_like(rho_arr)
    drho[0]    = rho_arr[1] - rho_arr[0]
    drho[-1]   = rho_arr[-1] - rho_arr[-2]
    drho[1:-1] = (rho_arr[2:] - rho_arr[:-2]) / 2.0

    # Area element: dA = 2π ρ a² κ_local dρ
    dA = 2.0 * np.pi * rho_arr * a**2 * kappa_arr * drho

    # Mask out unphysical points (eps too small, n or T too low)
    valid = (eps_arr >= 0.01) & (n_arr >= 1e-3) & (T_arr >= 0.1)
    j_bs = np.where(valid, j_bs, 0.0)

    I_bs = np.sum(j_bs * dA) / 1e6   # [MA]

    if return_profile:
        return {'I_bs': I_bs, 'rho': rho_arr, 'j_bs': j_bs,
                'dA': dA, 'kappa_arr': kappa_arr, 'q_arr': q_arr,
                'drho': drho}
    return I_bs


# ==============================================================================
# Self-consistent q-profile from Ampère integration with neoclassical
# conductivity and bootstrap current
# ==============================================================================

def f_q_profile_selfconsistent(
        Ip, I_Ohm, I_CD, q95,
        R0, a, B0, kappa, nbar, Tbar, Z_eff,
        nu_n, nu_T,
        eta_model='redl', bootstrap_model='Redl',
        rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
        Vprime_data=None, kappa_95=None, rho_95=0.95,
        rho_CD=0.3, delta_CD=0.15,
        q_saw=1.0,
        n_rho=200, tol=1e-3, max_iter=50, damping=0.3):
    """
    Self-consistent safety-factor profile from Ampère's law.

    Computes q(rho) by integrating the total current density radially,
    decomposed into three physically distinct components:

        j_total(rho) = j_Ohm(rho) + j_CD(rho) + j_bs(rho)

    with a post-convergence sawtooth clamp to enforce q0 >= q_saw.

    Current density decomposition
    -----------------------------
    Ohmic current : j_Ohm(rho)
        In resistive equilibrium with uniform loop electric field E_phi:
            j_Ohm(rho) = E_phi * sigma_neo(rho) = E_phi / eta_neo(T, n, eps, q)
        Normalised so that integral(j_Ohm dA) = I_Ohm.

    Current drive : j_CD(rho)
        Parameterised as a Gaussian deposition profile:
            j_CD(rho) proportional to exp(-(rho - rho_CD)^2 / (2 delta_CD^2))
        Normalised so that integral(j_CD dA) = I_CD.
        The deposition centre rho_CD and width delta_CD depend on the CD
        source: LHCD is broad and off-axis (rho ~ 0.4-0.7, delta ~ 0.15),
        ECCD is narrow and tuneable (rho ~ 0.3, delta ~ 0.05), NBI is broad
        (rho ~ 0.3, delta ~ 0.15).

    Bootstrap current : j_bs(rho)
        Computed from the Sauter (1999) or Redl (2021) neoclassical
        coefficients, using the same functions as f_Sauter_Ib/f_Redl_Ib.
        Updated at each Picard iteration via q(rho) -> nu*_e, nu*_i.

    Sawtooth clamp
    --------------
    In standard H-mode scenarios, sawtooth oscillations (Kadomtsev
    reconnection) periodically flatten the current density inside the
    q = 1 surface, maintaining q0 ~ 1.  After the Picard iteration
    converges, if q(0) < q_saw:

      1. The mixing radius rho_mix where q(rho_mix) = q_saw is found.
      2. j_total is flattened to a constant value inside rho_mix,
         preserving the enclosed current at rho_mix.
      3. I_enc and q are recomputed from the clamped j profile.

    This is the standard approach in systems codes (PROCESS, SYCOMORE).

    Ampere integration
    ------------------
    The enclosed current and safety factor follow from:

        I_enc(rho) = integral_0_rho j_total(rho') * 2pi rho' a^2 kappa(rho') drho'
        q_raw(rho) = rho^2 / I_enc(rho)     (cylindrical Ampere)

    normalised so that q(rho_95) = q95 (MHD equilibrium constraint).

    Parameters
    ----------
    Ip : float
        Total plasma current [MA].
    I_Ohm : float
        Ohmic (inductive) current at flat-top [MA].
    I_CD : float
        Driven (non-inductive, non-bootstrap) current [MA].
    q95 : float
        Safety factor at rho = rho_95 (MHD constraint).
    R0, a, B0, kappa : float
        Major radius [m], minor radius [m], on-axis field [T],
        edge elongation [-].
    nbar, Tbar, Z_eff : float
        Volume-averaged density [1e20 m^-3], temperature [keV],
        effective charge [-].
    nu_n, nu_T : float
        Density and temperature peaking exponents.
    eta_model : str
        Resistivity model for sigma_neo: 'spitzer', 'sauter', or 'redl'.
    bootstrap_model : str
        Bootstrap coefficient set: 'Sauter' or 'Redl'.
    rho_ped, n_ped_frac, T_ped_frac : float
        Pedestal parameters (passed to f_Tprof / f_nprof).
    Vprime_data : tuple or None
        Miller geometry data (enables shaped area element and kappa(rho)).
    kappa_95 : float or None
        Elongation at rho_95.  Defaults to f_Kappa_95(kappa).
    rho_95 : float
        Position of the 95%% flux surface (default 0.95).
    rho_CD : float
        Normalised deposition radius for current drive (default 0.3).
    delta_CD : float
        Gaussian width of the CD deposition profile (default 0.15).
    q_saw : float
        Sawtooth floor for q0 (default 1.0).  Set to 0 to disable.
    n_rho : int
        Number of core radial grid points (default 200).
    tol : float
        Relative convergence tolerance on q(rho) (default 1e-3).
    max_iter : int
        Maximum Picard iterations (default 50).
    damping : float
        Mixing parameter for damped update (default 0.3).

    Returns
    -------
    dict with keys:
        q_arr      : ndarray  Converged safety-factor profile q(rho).
        rho        : ndarray  Radial grid.
        j_total    : ndarray  Total current density [A/m^2].
        j_Ohm      : ndarray  Ohmic current density [A/m^2].
        j_CD       : ndarray  CD current density [A/m^2].
        j_non_bs   : ndarray  j_Ohm + j_CD [A/m^2].
        j_bs       : ndarray  Bootstrap current density [A/m^2].
        I_enc      : ndarray  Enclosed current [A].
        li         : float    Normalised internal inductance li(3) [-].
        q0         : float    Central safety factor (after sawtooth clamp).
        n_iter     : int      Number of Picard iterations performed.
        converged  : bool     Whether tolerance was achieved.
        kappa_arr  : ndarray  Local elongation profile.

    References
    ----------
    Wesson, Tokamaks, 4th ed. (2011) ch. 3 -- cylindrical Ampere.
    Sauter et al., Phys. Plasmas 6 (1999) 2834 -- neoclassical sigma.
    Redl et al., Phys. Plasmas 28 (2021) 022502 -- improved coeffs.
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054 -- PROCESS S18.
    Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 389 -- sawtooth model.
    """
    # Resistivity models are defined in this module (D0FUS_physical_functions).

    mu0 = 4e-7 * np.pi

    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa)

    # == Radial grid ==========================================================
    # Fine grid starting near the axis (rho_min = 0.001) is essential for
    # accurate Ampere integration: the Ohmic current density peaks on axis
    # (j proportional to sigma_neo proportional to T^{3/2}) and missing the
    # rho < 0.05 region causes a large error on q(0).
    # The pedestal region (rho > rho_ped) is refined for bootstrap gradient
    # resolution.
    rho_inner = np.linspace(0.001, 0.05, 30, endpoint=False)
    rho_core  = np.linspace(0.05, rho_ped, n_rho, endpoint=False)
    rho_edge  = np.linspace(rho_ped, 0.99, 3 * n_rho)
    rho = np.concatenate([rho_inner, rho_core, rho_edge])
    n_pts = len(rho)

    # == Fixed profiles (independent of q) ====================================
    T_arr = f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac, Vprime_data)
    n_arr = f_nprof(nbar, nu_n, rho, rho_ped, n_ped_frac, Vprime_data)
    eps_arr = rho * a / R0

    use_miller = (Vprime_data is not None)
    if use_miller:
        kappa_arr = kappa_profile(rho, kappa, kappa_95, rho_95)
    else:
        kappa_arr = np.full_like(rho, kappa)

    # Grid spacing: midpoint rule with proper axis cell.
    # The first cell covers [0, (rho_0+rho_1)/2] so that the integral
    # starts at the magnetic axis.
    drho = np.zeros_like(rho)
    drho[0]    = (rho[0] + rho[1]) / 2.0
    drho[-1]   = rho[-1] - rho[-2]
    drho[1:-1] = (rho[2:] - rho[:-2]) / 2.0

    # Shaped area element: dA = 2 pi rho a^2 kappa(rho) drho
    dA = 2.0 * np.pi * rho * a**2 * kappa_arr * drho

    # SI units
    n_e  = n_arr * 1e20           # [m^-3]
    T_eV = T_arr * 1e3            # [eV]
    n_i  = n_e / Z_eff

    # Pressure [Pa]
    p_e   = n_e * T_eV * E_ELEM
    p_i   = n_i * T_eV * E_ELEM
    p_tot = p_e + p_i
    R_pe  = np.where(p_tot > 0, p_e / p_tot, 0.5)

    # Logarithmic gradients (fixed, independent of q)
    dT_drho = np.gradient(T_arr, rho)
    dn_drho = np.gradient(n_arr, rho)
    dln_T = np.where(T_arr > 0.01, dT_drho / (T_arr * a), 0.0)
    dln_n = np.where(n_arr > 1e-3, dn_drho / (n_arr * a), 0.0)
    dln_p  = dln_n + dln_T
    dln_Ti = dln_T   # Ti = Te assumed

    I_psi = R0 * B0
    B_sq  = B0**2 * (1.0 + eps_arr**2 / 2.0)

    # Current decomposition [A]
    I_Ohm_A = max(I_Ohm * 1e6, 0.0)
    I_CD_A  = max(I_CD  * 1e6, 0.0)

    # Validity mask for bootstrap (same as f_Sauter_Ib / f_Redl_Ib)
    valid = (eps_arr >= 0.01) & (n_arr >= 1e-3) & (T_arr >= 0.1)

    # == CD deposition profile (Gaussian, fixed) ==============================
    # j_CD(rho) proportional to exp(-(rho - rho_CD)^2 / (2 delta_CD^2)),
    # normalised to I_CD.
    j_CD_shape = np.exp(-0.5 * ((rho - rho_CD) / max(delta_CD, 0.01))**2)
    j_CD_shape_dA = np.sum(j_CD_shape * dA)
    if j_CD_shape_dA > 0 and I_CD_A > 0:
        j_CD_arr = I_CD_A * j_CD_shape / j_CD_shape_dA
    else:
        j_CD_arr = np.zeros(n_pts)

    # == Initial q(rho) guess for Picard iteration =============================
    # The analytical q-profile from f_q_profile (j ~ (1-rho^2)^alpha_J) is
    # used ONLY as an initial guess.  The Picard loop below replaces it with
    # the self-consistent q(rho) derived from j_Ohm + j_CD + j_bs at every
    # iteration.  The converged result is independent of this starting point;
    # alpha_J = 1.5 (IPDG89 standard) is a robust default that ensures
    # convergence in 5-8 iterations for all tested configurations.
    q_arr = f_q_profile(rho, q95=q95, rho95=rho_95, alpha_J=1.5)

    # == Picard iteration =====================================================
    rel_change = np.inf
    for _iter in range(max_iter):

        # 1. Neoclassical conductivity sigma(rho) = 1/eta_neo(T, n, eps, q)
        #    For eps < 0.02 (near axis), the trapped-particle fraction
        #    vanishes and neoclassical corrections are negligible;
        #    Spitzer resistivity is used directly.
        eta_arr = np.empty(n_pts)
        for i in range(n_pts):
            T_i   = max(T_arr[i], 0.1)
            eps_i = eps_arr[i]
            q_i   = max(q_arr[i], 0.5)
            if eps_i < 0.02 or eta_model == 'spitzer':
                eta_arr[i] = eta_spitzer(T_i, n_e[i], Z_eff)
            elif eta_model == 'redl':
                eta_arr[i] = eta_redl(T_i, n_e[i], Z_eff, eps_i, q_i, R0)
            elif eta_model == 'sauter':
                eta_arr[i] = eta_sauter(T_i, n_e[i], Z_eff, eps_i, q_i, R0)
            else:
                eta_arr[i] = eta_spitzer(T_i, n_e[i], Z_eff)

        sigma_arr = 1.0 / np.maximum(eta_arr, 1e-20)

        # 2. Ohmic current density: j_Ohm(rho) proportional to sigma_neo(rho)
        #    Normalised so that integral(j_Ohm dA) = I_Ohm.
        sigma_dA = np.sum(sigma_arr * dA)
        if sigma_dA > 0 and I_Ohm_A > 0:
            j_Ohm_arr = I_Ohm_A * sigma_arr / sigma_dA
        else:
            j_Ohm_arr = np.zeros(n_pts)

        # 3. Bootstrap current density from neoclassical coefficients
        #    Uses the current q(rho) for collisionality calculations.
        f_t      = _trapped_fraction(eps_arr, kappa_arr)
        nu_e_arr = _nu_e_star(n_e, T_eV, q_arr, R0, eps_arr, Z_eff)
        nu_i_loc = _nu_i_star(n_i, T_eV, q_arr, R0, eps_arr)

        if bootstrap_model == 'Redl':
            L31 = _L31_Redl(f_t, nu_e_arr, Z_eff)
            L32 = _L32_Redl(f_t, nu_e_arr, Z_eff)
            L34 = L31                                    # Redl Eq. 19
            alp = _alpha_Redl(f_t, nu_i_loc, Z_eff)
        else:
            L31 = _L31(f_t, nu_e_arr, Z_eff)
            L32 = _L32(f_t, nu_e_arr, Z_eff)
            L34 = _L34(f_t, nu_e_arr, Z_eff)
            alp = _alpha(f_t, nu_i_loc)

        C_bs  = L31 * dln_p + L32 * R_pe * dln_T + L34 * alp * (1.0 - R_pe) * dln_Ti
        j_bs  = -I_psi * p_tot * C_bs / B_sq
        j_bs  = np.where(valid, j_bs, 0.0)

        # 4. Total current density and enclosed current (shaped integration)
        j_total = j_Ohm_arr + j_CD_arr + j_bs
        I_enc   = np.cumsum(j_total * dA)              # [A]
        I_enc   = np.maximum(I_enc, 1e-3)              # guard division

        # 5. q(rho) from cylindrical Ampere: q proportional to rho^2 / I_enc
        #    Normalised to q(rho_95) = q95 (MHD equilibrium constraint).
        q_raw    = rho**2 / I_enc
        q_raw_95 = np.interp(rho_95, rho, q_raw)

        if q_raw_95 > 0:
            q_new = q95 * q_raw / q_raw_95
        else:
            q_new = q_arr                               # fallback

        # 6. Convergence check (relative L-infinity norm)
        rel_change = np.max(np.abs(q_new - q_arr) / np.maximum(q_arr, 0.5))

        # Damped Picard update
        q_arr = damping * q_new + (1.0 - damping) * q_arr

        if rel_change < tol:
            break

    # == Sawtooth clamp (Kadomtsev model) =====================================
    # If q(0) < q_saw, flatten j inside the mixing radius to restore q0.
    # Physically: sawtooth crashes redistribute core current outward,
    # maintaining q0 ~ 1 in standard H-mode scenarios.
    #
    # For q = const inside the mixing radius, cylindrical Ampere gives
    # I_enc ~ rho^2, hence j = const (flat current density).  This is
    # the standard 0D sawtooth model (PROCESS, SYCOMORE, METIS).
    if q_saw > 0 and q_arr[0] < q_saw:
        # Find the mixing radius: outermost point where q < q_saw
        below_saw = np.where(q_arr < q_saw)[0]
        if len(below_saw) > 0:
            i_mix = below_saw[-1]  # outermost index with q < q_saw

            # Flatten j_total inside rho_mix to a constant that
            # preserves the enclosed current at rho_mix.
            # q = const => j = const (cylindrical Ampere).
            I_at_mix  = I_enc[i_mix]
            dA_inside = np.sum(dA[:i_mix + 1])
            if dA_inside > 0:
                j_flat = I_at_mix / dA_inside
                j_total[:i_mix + 1] = j_flat

            # Recompute I_enc and q from the flattened j profile
            I_enc  = np.cumsum(j_total * dA)
            I_enc  = np.maximum(I_enc, 1e-3)
            q_raw  = rho**2 / I_enc
            q_raw_95 = np.interp(rho_95, rho, q_raw)
            if q_raw_95 > 0:
                q_arr = q95 * q_raw / q_raw_95
                # Re-enforce the clamp after renormalisation
                below_saw2 = np.where(q_arr < q_saw)[0]
                if len(below_saw2) > 0:
                    q_arr[below_saw2] = q_saw

    # == Derived quantities ===================================================

    # Non-bootstrap composite (for figures)
    j_non_bs = j_Ohm_arr + j_CD_arr

    # Internal inductance li(3) (ITER/EFIT convention):
    #
    #   li(3) = 4 Wp / (μ₀ R₀ Ip²)
    #
    # where Wp = ∫ Bp²/(2μ₀) dV is the poloidal magnetic energy, with:
    #   Bp(ρ)  = μ₀ I_enc(ρ) / Lp(ρ)     [Ampère's law on shaped surface]
    #   Lp(ρ)  ≈ 2πρa √((1+κ²)/2)        [ellipse perimeter approximation]
    #   dV     = V'(ρ) dρ                  [Miller Jacobian when available]
    #
    # Expanding: li(3) = (2/R₀) × ∫₀¹ (I_enc/Ip)² × V'(ρ)/Lp(ρ)² dρ
    #
    # In cylindrical circular geometry (κ=1, V'=4π²R₀a²ρ, Lp=2πρa),
    # this reduces to: li = 2 ∫ (I_enc/Ip)²/ρ dρ  (standard textbook form).
    # With shaping: li(3)_shaped = 2κ/(1+κ²) × li_cyl for constant κ,
    # which gives ~13% correction at κ = 1.7 (ITER).
    I_norm = I_enc / np.maximum(I_enc[-1], 1e-3)

    # Poloidal circumference of each flux surface [m]
    # Ellipse with semi-axes (ρa, ρaκ): Lp ≈ 2πρa √((1+κ²)/2)
    Lp_arr = 2.0 * np.pi * rho * a * np.sqrt((1.0 + kappa_arr**2) / 2.0)
    Lp_arr = np.where(rho > 1e-8, Lp_arr, 1.0)   # guard axis

    # Volume derivative V'(ρ) = dV/dρ [m³]
    if use_miller and Vprime_data is not None:
        Vprime_arr = interpolate_Vprime(rho, Vprime_data[0], Vprime_data[1])
    else:
        Vprime_arr = 4.0 * np.pi**2 * R0 * a**2 * kappa_arr * rho

    # li(3) = (2/R₀) × ∫ (I_enc/Ip)² × V'/Lp² dρ
    li_integrand = np.where(rho > 1e-8,
                            I_norm**2 * Vprime_arr / Lp_arr**2,
                            0.0)
    li = (2.0 / R0) * np.trapezoid(li_integrand, rho)

    return {
        'q_arr':       q_arr,
        'rho':         rho,
        'j_total':     j_total,
        'j_Ohm':       j_Ohm_arr,
        'j_CD':        j_CD_arr,
        'j_non_bs':    j_non_bs,
        'j_bs':        j_bs,
        'I_enc':       I_enc,
        'li':          li,
        'q0':          q_arr[0],
        'n_iter':      _iter + 1,
        'converged':   rel_change < tol,
        'kappa_arr':   kappa_arr,
    }



if __name__ == "__main__":
    # Complete bootstrap comparison table (all 4 models now defined)
    Ib_Redl = f_Redl_Ib(**_bs_kw)

    print("\n── Bootstrap current — ITER Q=10 baseline ───────────────────────────────────")
    print(f"  {'Model':<20} {'I_bs [MA]':>10}  {'ITER ref. [MA]':>14}")
    print("  " + "─" * 72)
    print(f"  {'Freidberg (2015)':<20} {Ib_Freidberg:>10.2f}  {'3.0':>14}")
    print(f"  {'Segal (2021)':<20} {Ib_Segal:>10.2f}  {'3.0':>14}")
    print(f"  {'Sauter (1999)':<20} {Ib_Sauter:>10.2f}  {'3.0':>14}")
    print(f"  {'Redl (2021)':<20} {Ib_Redl:>10.2f}  {'3.0':>14}")


#%% Other parameters
# ─────────────────────────────────────────────────────────────────────────────
# Engineering figures of merit: heat load proxies, plasma current, neutron
# wall loading, MHD safety factors, scaling-law registry, global energy
# descriptors, helium ash accumulation model, and cost proxy.
#
# Profile convention (applies to f_He_fraction, f_tau_alpha):
#   Academic  : rho_ped=1.0, T_ped_frac=0.0  (pure power-law, no pedestal)
#   D0FUS     : 0 < rho_ped < 1, T_ped_frac > 0  (H-mode pedestal profile)
#
# Geometry convention (applies to f_Gamma_n, f_tauE):
#   Academic  : S_wall=None → elliptical-torus approximation
#               V must be passed as 2π²R₀a²κ by the caller
#   D0FUS     : S_wall / V supplied by the Miller-geometry module
# ─────────────────────────────────────────────────────────────────────────────

def f_heat_D0FUS(R0, P_sep):
    """
    Evaluate the simple D0FUS heat load parameter H = P_sep / R₀.

    This is a robust proxy for the power flux across the separatrix that scales
    with machine size.  Used as a fast figure of merit in 0D reactor studies.

    Parameters
    ----------
    R0 : float
        Major radius [m]
    P_sep : float
        Power crossing the separatrix (= P_loss − P_rad,bulk) [MW]

    Returns
    -------
    heat : float
        Heat load parameter P_sep / R₀ [MW m⁻¹]
    """
    return P_sep / R0


def f_heat_par(R0, B0, P_sep):
    """
    Evaluate the parallel heat flux parameter H_∥ = P_sep B₀ / R₀.

    Introduced by Freidberg et al. (2015) as a dimensionally consistent measure
    of the power density carried along field lines in the SOL.  Scaling with B₀
    captures the field-line pitch effect on the wetted area.

    Parameters
    ----------
    R0 : float
        Major radius [m]
    B0 : float
        On-axis toroidal magnetic field [T]
    P_sep : float
        Power crossing the separatrix [MW]

    Returns
    -------
    heat : float
        Parallel heat load parameter P_sep B₀ / R₀ [MW T m⁻¹]

    References
    ----------
    J.P. Freidberg et al., Physics of Plasmas 22 (2015) 070901.
    """
    return P_sep * B0 / R0


def f_heat_pol(R0, B0, P_sep, a, q95):
    """
    Evaluate the poloidal heat flux parameter after Siccinio et al. (2019).

    Accounts for the safety factor q₉₅ and aspect ratio A = R₀/a, providing a
    geometry-sensitive measure of divertor loading relative to the poloidal
    field line length.

    Parameters
    ----------
    R0 : float
        Major radius [m]
    B0 : float
        On-axis toroidal magnetic field [T]
    P_sep : float
        Power crossing the separatrix [MW]
    a : float
        Minor radius [m]
    q95 : float
        Safety factor at ψ_N = 0.95 (dimensionless)

    Returns
    -------
    heat : float
        Poloidal heat load parameter [MW T m⁻²]

    References
    ----------
    M. Siccinio et al., Fusion Engineering and Design 176 (2019) 107523.
    """
    A = R0 / a
    return (P_sep * B0) / (q95 * A * R0**2)


def f_heat_PFU_Eich(P_sol, B_pol, R, eps, theta_deg):
    """
    Estimate the divertor peak heat flux using the Eich (2013) multi-machine
    SOL-width scaling law.

    Provides a fast 0D estimate of the power deposited on divertor plasma-facing
    units (PFUs).  For ITER/DEMO design validation, cross-check with dedicated
    edge transport codes (SOLPS-ITER, UEDGE).

    Parameters
    ----------
    P_sol : float
        Power entering the scrape-off layer (= P_sep) [MW]
    B_pol : float
        Poloidal magnetic field at the outer midplane [T]
    R : float
        Major radius [m]
    eps : float
        Inverse aspect ratio ε = a/R (dimensionless)
    theta_deg : float
        Grazing angle of field lines at the divertor target [°]
        Typical range: 1–5° for vertical-target configurations.

    Returns
    -------
    lambda_q : float
        SOL power e-folding decay length [m]
    q_par0 : float
        Peak parallel heat flux at the separatrix [MW m⁻²]
    q_target : float
        Peak perpendicular heat flux on the divertor target [MW m⁻²]

    Notes
    -----
    Eich (2013) regression (fit 7, Table 1):
        λ_q [mm] = 1.35 · R^{0.04} · B_pol^{-0.92} · ε^{0.42} · P_sol^{-0.02}
    Peak parallel flux (toroidal symmetry over the outer midplane):
        q_∥0 = P_sol / (2π R λ_q)
    Projection onto the target surface (grazing incidence):
        q_⊥ = q_∥0 · sin θ

    References
    ----------
    T. Eich et al., Nuclear Fusion 53 (2013) 093031.
    """
    θ = np.deg2rad(theta_deg)

    # SOL power decay length: empirical multi-machine regression [mm → m]
    lambda_q = 1.35 * R**0.04 * B_pol**(-0.92) * eps**0.42 * P_sol**(-0.02) * 1e-3

    # Peak parallel heat flux at the separatrix [MW m⁻²]
    q_par0 = P_sol / (2 * np.pi * R * lambda_q)

    # Perpendicular target heat flux accounting for grazing incidence [MW m⁻²]
    q_target = q_par0 * np.sin(θ)

    return lambda_q, q_par0, q_target


# ── Plasma current from τ_E scaling law ──────────────────────────────────────

def f_kappa_x(a, kappa_edge, delta_edge=0.0, geometry_model='Academic',
              S_cross=None):
    """
    Cross-section elongation κ_x = S₀ / (π a²) — the IPB98(y,2) convention.

    The IPB98(y,2) confinement scaling defines elongation as κ_x = S₀/(πa²)
    where S₀ is the plasma poloidal cross-section area (Shimada 2007 Table 2;
    Doyle 2007 PIPB Ch. 2 §5.3).

    For an ideal ellipse: S₀ = πa²κ → κ_x = κ_edge exactly.
    With triangularity: κ_x < κ_edge (D-shaping reduces cross-section area).
    ITER: κ_edge=1.85, δ=0.48, S₀=22m² → κ_x ≈ 1.75, not 1.85.

    Parameters
    ----------
    a, kappa_edge, delta_edge : float
    geometry_model : 'Academic' | 'D0FUS'
    S_cross : float or None  Externally provided cross-section area [m²].

    Returns
    -------
    kappa_x : float  Cross-section elongation [-].
    """
    if S_cross is not None:
        return S_cross / (np.pi * a**2)
    if geometry_model == 'Academic':
        return kappa_edge
    elif geometry_model == 'D0FUS':
        N = 2000
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        R_t = a * np.cos(theta + np.arcsin(delta_edge) * np.sin(theta))
        Z_t = kappa_edge * a * np.sin(theta)
        R_s = np.append(R_t, R_t[0])
        Z_s = np.append(Z_t, Z_t[0])
        S_cross = 0.5 * np.abs(np.sum(R_s[:-1] * Z_s[1:] - R_s[1:] * Z_s[:-1]))
        return S_cross / (np.pi * a**2)
    else:
        raise ValueError(f"Unknown geometry_model '{geometry_model}'.")

def f_Ip(tauE, R0, a, κ, δ, nbar, B0, Atomic_mass,
         P_Alpha, P_Ohm, P_Aux, P_rad,
         H, C_SL,
         alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
         alpha_R, alpha_B, alpha_n, alpha_I, alpha_P):
    """
    Invert a τ_E multi-machine scaling law to obtain the required plasma current.

    Given a target energy confinement time τ_E (derived from the power balance),
    solves for the plasma current I_p such that the chosen scaling law is
    exactly satisfied.

    .. warning:: Elongation convention
       The IPB98(y,2) scaling uses κ_x = S₀/(πa²), NOT κ_LCFS.
       For shaped plasmas (δ > 0), κ_x < κ_LCFS.
       Use f_kappa_x() to convert.  ITER: κ_LCFS=1.85 → κ_x ≈ 1.75.
       Passing κ_LCFS overestimates confinement by ~(κ_LCFS/κ_x)^0.78 ≈ 5%.
       Ref: Shimada et al., NF 47 (2007), Table 2 — S₀ = 22 m².

    .. warning:: Density convention
       The IPB98(y,2) and ITPA20 scalings were fitted using LINE-AVERAGED
       density (interferometer convention), NOT volume-averaged.
       Use f_nbar_line() to convert before calling.
       For peaked profiles the ratio n̄_line/n̄_vol > 1 (up to ~30%).
       Ref: ITER Physics Basis, NF 39 (1999) 2175, §6.4.

    Parameters
    ----------
    tauE : float
        Target energy confinement time [s]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    κ : float
        Plasma elongation — should be κ_x = S₀/(πa²) for IPB98(y,2).
        Use f_kappa_x() or κ_a = V/(2π²R₀a²) to compute.
    δ : float
        Plasma triangularity (LCFS) (dimensionless)
    nbar : float
        Electron density [10²⁰ m⁻³] — should be LINE-AVERAGED for
        IPB98(y,2), ITPA20, ITER89-P.  Use f_nbar_line() to convert.
    B0 : float
        On-axis toroidal field [T]
    Atomic_mass : float
        Effective ion mass [AMU]  (M = 2 for D, 2.5 for D–T)
    P_Alpha : float
        Alpha-particle heating power [MW]
    P_Ohm : float
        Ohmic heating power [MW]
    P_Aux : float
        Auxiliary heating power [MW]
    P_rad : float
        Core radiated power [MW] (P_rad_core = P_Brem + P_syn + P_line_core).
        Must use the same P_loss convention as f_tauE for consistency.
    H : float
        H-factor (confinement enhancement; H = 1 at the scaling-law value)
    C_SL, alpha_* : float
        Pre-factor and exponents from `f_Get_parameter_scaling_law`.

    Returns
    -------
    Ip : float
        Required plasma current [MA]

    Notes
    -----
    General multi-machine scaling (ITER Engineering Parameters convention):
        τ_E = H · C · R₀^α_R · ε^α_ε · κ^α_κ · (10 n̄)^α_n
                · B₀^α_B · M^α_M · P^α_P · (1+δ)^α_δ · I_p^α_I
    Inverting for I_p:
        I_p = (τ_E / denom)^{1/α_I}
    with P = P_α + P_Ohm + P_aux − P_rad.
    """
    P = P_Alpha + P_Ohm + P_Aux - P_rad
    ε = a / R0

    # Guard: all scaling laws have alpha_P < 0, so P ≤ 0 yields inf or complex.
    if P <= 0.0:
        warnings.warn(
            f"f_Ip: non-positive net heating power P = {P:.3f} MW "
            f"(P_α={P_Alpha:.2f}, P_Ohm={P_Ohm:.2f}, "
            f"P_aux={P_Aux:.2f}, P_rad={P_rad:.2f} MW). "
            "Scaling law requires P > 0 (alpha_P < 0 in all supported laws). "
            "Returning NaN.",
            RuntimeWarning, stacklevel=2
        )
        return np.nan

    denom = (H * C_SL
             * R0**alpha_R
             * ε**alpha_epsilon
             * κ**alpha_kappa
             * (nbar * 10)**alpha_n
             * B0**alpha_B
             * Atomic_mass**alpha_M
             * P**alpha_P
             * (1 + δ)**alpha_delta)

    return (tauE / denom) ** (1.0 / alpha_I)   # [MA]


# ── Neutron wall loading ──────────────────────────────────────────────────────

def f_Gamma_n(a, P_fus, R0, κ, S_wall=None):
    """
    Estimate the average neutron wall loading Γ_n at the first wall.

    The 14.1 MeV neutrons carry 80 % of the D–T fusion energy and are the
    primary driver of structural material damage (dpa) and tritium breeding.
    Two geometry approximations are supported:

    **Academic (default, S_wall = None)**
        Elliptical torus approximation:
            S_wall = 4π² R₀ a √[(1 + κ²)/2]
        Adequate for moderate shaping (κ ≲ 2, δ ≲ 0.4).

    **D0FUS / Miller (S_wall provided)**
        Pass the first-wall surface area pre-computed by the Miller geometry
        module (numerically integrated over actual flux-surface contours).
        Recommended for strongly shaped configurations.

    Parameters
    ----------
    a : float
        Minor radius [m]
    P_fus : float
        Total D–T fusion power [MW]
    R0 : float
        Major radius [m]
    κ : float
        Plasma elongation (LCFS) — only used when S_wall is None
    S_wall : float or None, optional
        Pre-computed first-wall surface area [m²].
        If None, the academic elliptical approximation is applied.

    Returns
    -------
    Γ_n : float
        Average neutron wall loading [MW m⁻²]

    Notes
    -----
    Neutron power fraction: f_n = E_n / (E_α + E_n) ≈ 0.80 for D–T.
    ITER design target: Γ_n ≈ 0.57 MW m⁻² at P_fus = 500 MW.
    """
    P_neutron = (E_N / (E_ALPHA + E_N)) * P_fus

    if S_wall is None:
        # Academic: elliptical torus first-wall area approximation
        S_wall = 4 * np.pi**2 * R0 * a * np.sqrt((1 + κ**2) / 2)

    return P_neutron / S_wall


# ── MHD safety factors ────────────────────────────────────────────────────────

def f_qstar(a, B0, R0, Ip, κ):
    """
    Compute the cylindrical (kink) safety factor q*.

    q* is the Kruskal–Shafranov stability parameter: the safety factor of the
    equivalent periodic cylinder carrying the same current.  Operational limits:
    q* > 2 (hard disruption boundary); q* > 3 recommended for margin.

    Parameters
    ----------
    a : float
        Minor radius [m]
    B0 : float
        On-axis toroidal field [T]
    R0 : float
        Major radius [m]
    Ip : float
        Plasma current [MA]
    κ : float
        Plasma elongation (LCFS) (dimensionless)

    Returns
    -------
    qstar : float
        Cylindrical safety factor (dimensionless)

    Notes
    -----
    Formula (Freidberg et al., PoP 2015, eq. 30):
        q* = π a² B₀ (1 + κ²) / (μ₀ R₀ I_p)

    References
    ----------
    J.P. Freidberg et al., Physics of Plasmas 22 (2015) 070901.
    """
    return (np.pi * a**2 * B0 * (1 + κ**2)) / (μ0 * R0 * Ip * 1e6)


def f_q95(B0, Ip, R0, a, kappa_edge, delta_edge,
          kappa_95=None, delta_95=None, Option_q95='Sauter'):
    """
    Estimate the edge safety factor q at ψ_N = 0.95.

    q₉₅ is the primary MHD stability parameter for H-mode scenario design
    (ELM behaviour, Greenwald limit, disruption avoidance).  Two analytical
    formulas are available via the `Option_q95` selector.

    .. important:: Shaping-parameter convention
       The two formulas use **different** shaping conventions:

       * **Sauter (2016)** — uses LCFS (edge) values κ_edge, δ_edge.
         Section 3 of the paper demonstrates that using ψ_N = 0.95 values
         introduces a systematic bias up to a factor 2 at negative δ (Fig. 2d).
       * **ITER_1989** — uses values at ψ_N = 0.95 (κ₉₅, δ₉₅), as in the
         original ITER Physics Design Guidelines (Uckan 1989/1991).

       The caller must supply both sets; if kappa_95 / delta_95 are omitted,
       they default to the ITER 1989 rule: κ₉₅ = κ_edge/1.12,
       δ₉₅ = δ_edge/1.5.

    Parameters
    ----------
    B0 : float
        On-axis toroidal field [T]
    Ip : float
        Plasma current [MA]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    kappa_edge : float
        Elongation at the LCFS (κ_edge).  Used by Sauter (2016).
    delta_edge : float
        Triangularity at the LCFS (δ_edge).  Used by Sauter (2016).
    kappa_95 : float or None, optional
        Elongation at ψ_N = 0.95.  Used by ITER_1989.
        Default: f_Kappa_95(kappa_edge) = kappa_edge / 1.12.
    delta_95 : float or None, optional
        Triangularity at ψ_N = 0.95.  Used by ITER_1989.
        Default: f_Delta_95(delta_edge) = delta_edge / 1.5.
    Option_q95 : str, optional
        Formula selector (default 'Sauter'):
        'Sauter'    — Sauter, Fusion Eng. Des. 112 (2016) 633, Eq. (30).
                      Fit on CHEASE equilibria using LCFS shaping parameters.
                      Recommended for 0D systems studies of shaped H-mode plasmas.
                      The squareness parameter is set to w₀₇ = 1 (no squareness
                      correction), so the [1 + 0.55(w₀₇ − 1)] factor in Eq. (30)
                      reduces to unity.
        'ITER_1989' — ITER Physics Design Guidelines, Uckan (1989/1991),
                      Eq. (4) in Sauter (2016).  Uses κ₉₅, δ₉₅.
                      Also adopted by Johner (2011) / HELIOS [ref. 5 in Sauter].
                      Use to cross-check against HELIOS outputs or CEA benchmarks.

    Returns
    -------
    q95 : float
        Safety factor at ψ_N = 0.95 (dimensionless)

    Notes
    -----
    Sauter (2016) Eq. (30), using LCFS shaping (w₀₇ = 1):
        q₉₅ = (4.1 a² B₀)/(R₀ I_p) · f_κ · f_δ · [1 + 0.55(w₀₇ − 1)]
        f_κ = 1 + 1.2(κ−1) + 0.56(κ−1)²              (κ = κ_edge)
        f_δ = (1 + 0.09δ + 0.16δ²)(1 + 0.45δε)       (δ = δ_edge)
              / (1 − 0.74ε)                            (ε = a/R₀)
        w₀₇ = 1  →  squareness factor = 1             (no correction)

    ITER_1989 / Uckan (1989), Eq. (4) in Sauter (2016), using 95% shaping:
        q₉₅ = (5 a² B₀)/(R₀ I_p) · F_shape · F_ε
        F_shape = [1 + κ₉₅²(1 + 2δ₉₅² − 1.2δ₉₅³)] / 2
        F_ε     = (1.17 − 0.65ε) / (1 − ε²)²

    References
    ----------
    [1] O. Sauter, Fusion Eng. Des. 112 (2016) 633–645.
        "Geometric formulas for system codes including the effect of
        negative triangularity."
    [2] N.A. Uckan & ITER Physics Group, IAEA/ITER/DS-10 (1990);
        Fusion Sci. Technol. 19 (1991) 1493.
    [3] J. Johner, Fusion Sci. Technol. 59 (2011) 308 (HELIOS).
    """
    A   = R0 / a
    eps = 1.0 / A                                  # inverse aspect ratio ε = a/R₀

    if Option_q95 == 'Sauter':
        # ── Sauter (2016) Eq. (30) — LCFS shaping ──────────────────────
        # Regression fit on 99 CHEASE equilibria; valid for −0.6 < δ < 0.8.
        # Squareness parameter fixed to w₀₇ = 1, i.e. [1 + 0.55(w₀₇ − 1)] = 1.
        f_kappa = 1 + 1.2 * (kappa_edge - 1) + 0.56 * (kappa_edge - 1)**2
        f_delta = ((1 + 0.09 * delta_edge + 0.16 * delta_edge**2)
                   * (1 + 0.45 * delta_edge * eps)
                   / (1 - 0.74 * eps))
        return (4.1 * a**2 * B0) / (R0 * Ip) * f_kappa * f_delta

    elif Option_q95 == 'ITER_1989':
        # ── Uckan (1989) / HELIOS — 95%-surface shaping ────────────────
        # Original ITER Physics Design Guidelines formula, Eq. (4) in
        # Sauter (2016).  Uses κ₉₅ and δ₉₅ by construction.
        if kappa_95 is None:
            kappa_95 = f_Kappa_95(kappa_edge)
        if delta_95 is None:
            delta_95 = f_Delta_95(delta_edge)
        return ((2 * np.pi * a**2 * B0) / (μ0 * Ip * 1e6 * R0)
                * (1.17 - 0.65 * eps) / (1 - eps**2)**2
                * (1 + kappa_95**2
                   * (1 + 2 * delta_95**2 - 1.2 * delta_95**3)) / 2)

    else:
        raise ValueError(f"Unknown Option_q95: '{Option_q95}'. "
                         "Valid options: 'Sauter', 'ITER_1989'.")


# ── Safety factor radial profile ─────────────────────────────────────────────
# f_q_profile() defined earlier (before f_Sauter_Ib) due to forward dependency.


# ── Scaling law coefficient registry ─────────────────────────────────────────

def f_Get_parameter_scaling_law(Scaling_Law):
    """
    Return the pre-factor and exponents of a global τ_E scaling law.

    The general form (ITER Engineering Parameters convention) is:

        τ_E = C · R₀^α_R · ε^α_ε · κ^α_κ · (1+δ)^α_δ
                · (10 n̄)^α_n · B₀^α_B · M^α_M · P^α_P · I_p^α_I

    with n̄ in 10²⁰ m⁻³ (factor 10 converts to 10¹⁹ m⁻³), I_p in MA, P in MW.

    Parameters
    ----------
    Scaling_Law : str
        Registry key.  Supported values:
        ``'IPB98(y,2)'``, ``'ITPA20'``, ``'ITPA20-IL'``,
        ``'DS03'``, ``'L-mode'``, ``'L-mode OK'``, ``'ITER89-P'``.

    Returns
    -------
    C_SL, alpha_delta, alpha_M, alpha_kappa, alpha_epsilon,
    alpha_R, alpha_B, alpha_n, alpha_I, alpha_P : float

    Raises
    ------
    ValueError
        If `Scaling_Law` is not found in the registry.

    References
    ----------
    IPB98(y,2) : ITER Physics Basis Expert Groups on Confinement & Transport,
        Nucl. Fusion 39 (1999) 2175, Table 5 (ELMy H-mode, type-I).
    ITPA20     : Verdoolaege, Kaye, Angioni, Kardaun, Maslov, Romanelli,
        Ryter & Thomsen, Nucl. Fusion 61 (2021) 076006, Table 4
        (full ELMy H-mode database DB5.2.3, log-linear regression).
    ITPA20-IL  : Same paper, Table 4 — ITER-Like subset (metal wall,
        single-null, type-I ELMs, n̄/nG < 1.2, q95 > 2.5).
    DS03       : Doyle et al. (PIPB Ch. 2), Nucl. Fusion 47 (2007) S18,
        Table III — dimensionless-variables-motivated fit (Petty 2008
        convention; named DS03 after the dataset used).
    L-mode     : Kaye et al., Nucl. Fusion 37 (1997) 1303, Table 2
        (ITER L-mode database, power-law fit).
    L-mode OK  : Variant of the L-mode scaling with reduced R exponent
        (α_R = 1.78 vs 1.83), from Goldston offset-linear regression
        (Lackner & Gottardi, Nucl. Fusion 30 (1990) 767).
    ITER89-P   : Yushmanov, Takizuka, Riedel, Kardaun, Cordey, Kaye &
        Post, Nucl. Fusion 30 (1990) 1999, Eq. 19 (ITER L-mode power-law
        scaling from the 1989 Confinement Workshop).
    """
    _registry = {
        # ITER Physics Basis (1999), NF 39, 2175 — Table 5, ELMy H-mode
        'IPB98(y,2)': dict(C_SL=0.0562, α_δ=0,    α_M=0.19, α_κ=0.78,
                           α_ε=0.58,  α_R=1.97,  α_B=0.15,
                           α_n=0.41,  α_I=0.93,  α_P=-0.69),
        # Verdoolaege et al. (2021), NF 61, 076006 — Table 4, ITER-Like subset
        'ITPA20-IL':  dict(C_SL=0.067,  α_δ=0.56, α_M=0.3,  α_κ=0.67,
                           α_ε=0,     α_R=1.19,  α_B=-0.13,
                           α_n=0.147, α_I=1.29,  α_P=-0.644),
        # Verdoolaege et al. (2021), NF 61, 076006 — Table 4, full H-mode DB
        'ITPA20':     dict(C_SL=0.053,  α_δ=0.36, α_M=0.2,  α_κ=0.8,
                           α_ε=0.35,  α_R=1.71,  α_B=0.22,
                           α_n=0.24,  α_I=0.98,  α_P=-0.669),
        # Doyle et al. (2007 PIPB Ch.2), NF 47, S18 — Table III
        'DS03':       dict(C_SL=0.028,  α_δ=0,    α_M=0.14, α_κ=0.75,
                           α_ε=0.3,   α_R=2.11,  α_B=0.07,
                           α_n=0.49,  α_I=0.83,  α_P=-0.55),
        # Kaye et al. (1997), NF 37, 1303 — Table 2, ITER L-mode DB
        'L-mode':     dict(C_SL=0.023,  α_δ=0,    α_M=0.2,  α_κ=0.64,
                           α_ε=-0.06, α_R=1.83,  α_B=0.03,
                           α_n=0.4,   α_I=0.96,  α_P=-0.73),
        # Variant with offset-linear R correction (Lackner & Gottardi 1990)
        'L-mode OK':  dict(C_SL=0.023,  α_δ=0,    α_M=0.2,  α_κ=0.64,
                           α_ε=-0.06, α_R=1.78,  α_B=0.03,
                           α_n=0.4,   α_I=0.96,  α_P=-0.73),
        # Yushmanov et al. (1990), NF 30, 1999 — Eq. 19
        'ITER89-P':   dict(C_SL=0.048,  α_δ=0,    α_M=0.5,  α_κ=0.5,
                           α_ε=0.3,   α_R=1.2,   α_B=0.2,
                           α_n=0.08,  α_I=0.85,  α_P=-0.5),
    }

    if Scaling_Law not in _registry:
        raise ValueError(
            f"Unknown scaling law '{Scaling_Law}'. "
            f"Available: {list(_registry.keys())}"
        )
    p = _registry[Scaling_Law]
    return (p['C_SL'], p['α_δ'], p['α_M'], p['α_κ'],
            p['α_ε'], p['α_R'], p['α_B'], p['α_n'], p['α_I'], p['α_P'])


# ── Global energy and confinement descriptors ─────────────────────────────────

def f_tauE(pbar, V, P_Alpha, P_Aux, P_Ohm, P_rad):
    """
    Compute the global energy confinement time from the power balance.

    Derived from τ_E ≡ W_th / P_loss, where the total thermal stored energy
    is W_th = (3/2) p̄ V and the net heating power is
    P_loss = P_α + P_Ohm + P_aux − P_rad.

    .. note:: P_loss convention debate
       The IPB98(y,2) scaling was originally fitted using
       P = P_heat = P_α + P_aux + P_Ohm (radiation NOT subtracted),
       because radiation is a loss mechanism that the scaling captures
       implicitly (Doyle et al. 2007, PIPB Ch. 2, §5.3).
       
       However, PROCESS (Kovari 2014) and most modern 0D codes subtract
       P_rad_core, arguing that core radiation escapes the confined region
       and should not count as heating.  This is the convention used here.
       
       The impact on Ip is ~(1/(1−f_rad))^(|α_P|/α_I):
         ITER (f_rad ≈ 0.16): ~6% on Ip.
         EU-DEMO (f_rad ≈ 0.5–0.7): 20–50% — much more significant.
       
       If strict IPB98(y,2) compliance is required, set P_rad = 0 in
       both this function and f_Ip, and account for radiation losses
       separately in the power balance.
       
       References:
         Doyle et al., NF 47 (2007) S18 — PIPB Chapter 2, §5.3.
         Kovari et al., Fusion Eng. Des. 89 (2014) 3054 — PROCESS.
         Zohm, Fusion Sci. Technol. 58 (2010) 613 — P_loss convention.

    The volume V should be provided by the caller using the geometry approach
    consistent with the rest of the calculation:
    - **Academic** : V = 2π² R₀ a² κ  (circular torus with elliptical section)
    - **D0FUS / Miller** : V numerically integrated from the Miller profile
      module over the actual shaped flux surfaces.

    Parameters
    ----------
    pbar : float
        Volume-averaged plasma thermal pressure [MPa]
        (n_e k_B T_e + n_i k_B T_i summed, with n_i = n_e, T_i = T_e)
    V : float
        Plasma volume [m³]
    P_Alpha : float
        Alpha-particle heating power [MW]
    P_Aux : float
        Auxiliary heating power [MW]
    P_Ohm : float
        Ohmic heating power [MW]
    P_rad : float
        Core radiated power [MW].  This should be P_rad_core (Bremsstrahlung
        + synchrotron + line radiation from ρ < ρ_rad_core), NOT P_rad_total.
        Edge/SOL radiation does not affect confinement — it was transported
        out of the confined region before being radiated.  See discussion in
        the note below.

    Returns
    -------
    tauE : float
        Energy confinement time [s].
        Returns np.nan when P_loss ≤ 0 (over-radiated or unphysical power
        balance).  A RuntimeWarning is issued so that the condition is
        visible during 2D scans without requiring the caller to check
        every return value manually.
    """
    p_Pa     = pbar * 1e6
    P_loss_W = (P_Alpha + P_Aux + P_Ohm - P_rad) * 1e6

    if P_loss_W <= 0:
        warnings.warn(
            f"f_tauE: non-positive net heating power "
            f"P_loss = {P_loss_W*1e-6:.3f} MW  "
            f"(P_α={P_Alpha:.2f}, P_aux={P_Aux:.2f}, "
            f"P_Ohm={P_Ohm:.2f}, P_rad={P_rad:.2f} MW). "
            "Returning NaN.",
            RuntimeWarning, stacklevel=2
        )
        return np.nan

    return 1.5 * p_Pa * V / P_loss_W


def f_W_th(pbar_MPa, volume):
    """
    Compute the total plasma thermal energy W_th.

    By definition of the volume-averaged total pressure p̄ = ⟨p_e + p_i⟩_vol,
    the stored thermal energy is exactly:

        W_th = (3/2) p̄ V = (3/2) ∫ (p_e + p_i) dV

    where f_pbar() already evaluates the profile-weighted integral
    ⟨n(ρ) T(ρ)⟩_V via numerical integration.

    Parameters
    ----------
    pbar_MPa : float
        Volume-averaged total plasma pressure (ions + electrons) [MPa],
        as returned by f_pbar().
    volume : float
        Plasma volume [m³].

    Returns
    -------
    W_th : float
        Total thermal energy [J].
    """
    return (3/2) * pbar_MPa * 1e6 * volume


def f_Q(P_fus, P_CD, P_Ohm):
    """
    Compute the plasma energy amplification factor Q.

    Q ≡ P_fus / P_heat = P_fus / (P_CD + P_Ohm) is the standard figure of
    merit for a fusion plasma.  ITER design target: Q = 10.
    Ignition corresponds to Q → ∞ (no net external power required).

    Parameters
    ----------
    P_fus : float
        Total D–T fusion power [MW]
    P_CD : float
        Auxiliary / current-drive heating power [MW]
    P_Ohm : float
        Ohmic heating power [MW]

    Returns
    -------
    Q : float
        Fusion energy amplification factor (dimensionless).
        Returns np.inf when P_CD + P_Ohm ≤ 0, corresponding to the ignition
        limit or an unphysical (negative-power) operating point.  A
        RuntimeWarning is emitted so that the condition is visible in scans.
    """
    P_heat = P_CD + P_Ohm
    if P_heat <= 0.0:
        warnings.warn(
            f"f_Q: non-positive external heating P_heat = {P_heat:.3f} MW "
            "(P_CD + P_Ohm ≤ 0 — plasma at or beyond ignition). "
            "Returning np.inf.",
            RuntimeWarning, stacklevel=2
        )
        return np.inf
    return P_fus / P_heat


# ── Helium ash accumulation model ─────────────────────────────────────────────

def _sigmav_vol(T_bar, nu_T, rho_ped=1.0, T_ped_frac=0.0, N=200,
                Vprime_data=None):
    """
    Volume-averaged DT reactivity ⟨σv⟩_vol = ∫₀¹ ⟨σv⟩[T(ρ)] · w(ρ) dρ.

    Shared helper for f_He_fraction and f_tau_alpha, avoiding redundant
    profile and reactivity evaluations when both are called on the same
    design point.

    Parameters
    ----------
    T_bar      : float  Volume-averaged electron temperature [keV].
    nu_T       : float  Temperature peaking exponent.
    rho_ped    : float  Normalised pedestal radius (1.0 → parabolic).
    T_ped_frac : float  T_ped / T̄.
    N          : int    Radial integration points (default 200).
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → cylindrical weight 2ρ (Academic mode).

    Returns
    -------
    float  Volume-averaged reactivity [m³ s⁻¹].
    """
    if Vprime_data is not None:
        # D0FUS mode: Miller weight V'(ρ)/V
        rho_grid, Vprime, V_total = Vprime_data
        T   = f_Tprof(T_bar, nu_T, rho_grid, rho_ped, T_ped_frac,
                       Vprime_data)
        sv  = f_sigmav(T)
        return float(np.trapezoid(sv * Vprime, rho_grid)) / V_total
    else:
        # Academic mode: cylindrical weight 2ρ dρ
        rho = np.linspace(0.0, 1.0, N)
        T   = f_Tprof(T_bar, nu_T, rho, rho_ped, T_ped_frac)
        sv  = f_sigmav(T)
        return float(np.trapezoid(sv * 2.0 * rho, rho))


def f_He_fraction(n_bar, T_bar, tauE, C_Alpha, nu_T,
                  rho_ped=1.0, T_ped_frac=0.0, Vprime_data=None):
    """
    Estimate the equilibrium helium ash fraction f_α = n_α / n_e.

    Alpha particles (⁴He²⁺) produced by D–T fusion must be expelled to
    prevent fuel dilution and Q degradation.  This function solves the
    steady-state particle balance between alpha production (∝ n² ⟨σv⟩)
    and alpha removal (∝ n_α / τ_α).

    Two temperature profile models are supported via `f_Tprof`:

    **Academic** : rho_ped = 1.0, T_ped_frac = 0.0  (default)
        Pure power-law core profile T(ρ) = T̄ (1 − ρ²)^ν_T; no pedestal.

    **D0FUS / H-mode** : 0 < rho_ped < 1, T_ped_frac > 0
        Profile with a flat pedestal for ρ > ρ_ped and a power-law core,
        consistent with the D0FUS plasma-profile module.

    Parameters
    ----------
    n_bar : float
        Volume-averaged electron density [10²⁰ m⁻³]
    T_bar : float
        Volume-averaged electron/ion temperature [keV]
    tauE : float
        Energy confinement time [s]
    C_Alpha : float
        Alpha-particle removal efficiency parameter (dimensionless).
        Defines τ_α = C_α τ_E; typical value C_α ≈ 5 for ITER.
    nu_T : float
        Temperature profile peaking exponent (core power-law)
    rho_ped : float, optional
        Normalised pedestal radius ρ_ped (default 1.0 → no pedestal)
    T_ped_frac : float, optional
        Pedestal temperature as fraction of T̄ (default 0.0)
    Vprime_data : tuple or None
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        None → cylindrical weight (Academic mode).

    Returns
    -------
    f_alpha : float
        Equilibrium helium fraction n_α / n_e (dimensionless).
        ITER operational target: 0.05–0.10.

    Notes
    -----
    Sarazin quadratic steady-state balance (Appendix B):
        C = n̄ ⟨σv⟩_vol · C_α · τ_E
        f_α = (C + 1 − √(2C + 1)) / (2C)
    where ⟨σv⟩_vol is the volume-averaged D–T reactivity using the
    volume weight consistent with the geometry mode (cylindrical 2ρ dρ
    in Academic mode, Miller V'(ρ)/V in D0FUS mode).

    References
    ----------
    Y. Sarazin et al., Nuclear Fusion (2021). Appendix B.
    """
    # Volume-averaged reactivity via shared helper
    sigmav_vol = _sigmav_vol(T_bar, nu_T, rho_ped, T_ped_frac,
                             Vprime_data=Vprime_data)
    C = n_bar * 1e20 * sigmav_vol * C_Alpha * tauE
    return (C + 1 - np.sqrt(2 * C + 1)) / (2 * C)


def f_tau_alpha(n_bar, T_bar, tauE, C_Alpha, nu_T,
                rho_ped=1.0, T_ped_frac=0.0, Vprime_data=None):
    """
    Effective alpha-particle confinement time τ*_α [s].

    In the Sarazin helium-ash model the effective confinement time
    (including wall recycling) is the constitutive assumption of the
    model, not an output:

        τ*_α  =  C_α · τ_E

    The equilibrium ash fraction f_α then follows from the quadratic
    balance solved in `f_He_fraction`.  Conversely, substituting f_α
    back into the particle balance yields the same identity:

        τ*_α  =  4 f_α / [ n_e (1 − 2f_α)² ⟨σv⟩_vol ]  =  C_α · τ_E

    so the two routes are algebraically equivalent.

    Parameters
    ----------
    n_bar, T_bar, tauE, C_Alpha, nu_T, rho_ped, T_ped_frac, Vprime_data :
        Identical to `f_He_fraction`.  Only *tauE* and *C_Alpha* are
        used; the remaining arguments are kept for call-site
        compatibility.

    Returns
    -------
    tau_star_alpha : float
        Effective alpha-particle confinement time τ*_α [s].
        Ratio τ*_α / τ_E = C_α by construction.

    References
    ----------
    Y. Sarazin et al., Nucl. Fusion 60 (2020) 016010, Appendix B,
    Eq. (B.1): τ*_α = τ_α / (1 − R_α) ∼ C_α τ_E.
    """
    return C_Alpha * tauE


# ── Component volumes ─────────────────────────────────────────────────────────

def f_volume(a, b, c, d, R0, κ):
    """
    Approximate volumes of the main reactor structural components.

    Under development by Matteo Fletcher.

    All toroidal shells (BB, TF) use Pappus' centroid theorem with rectangular
    cross-sections of half-height (κ a + thickness contributions).  The CS and
    fusion-island envelope are modelled as right cylinders.

    Parameters
    ----------
    a : float
        Plasma minor radius [m].
    b : float
        Combined radial thickness: first wall + breeding blanket + neutron
        shield + vacuum vessel + assembly gaps [m].
    c : float
        Radial thickness of the TF coil winding pack [m].
    d : float
        Radial thickness of the CS coil [m].
    R0 : float
        Major radius [m].
    κ : float
        Plasma elongation (LCFS) (dimensionless).

    Returns
    -------
    V_BB : float
        Volume of FW + BB + neutron shield + VV + gaps [m³].
    V_TF : float
        Volume of TF coil winding packs [m³].
    V_CS : float
        Volume of the central solenoid [m³].
    V_FI : float
        Fusion-island bounding-cylinder volume [m³].

    Notes
    -----
    Geometry assumptions (rectangular / cylindrical):

    - BB  : toroidal shell, cross-section 2b × 2(κa + b),
            V = 2π R₀ × A_cross  (Pappus).
    - TF  : toroidal shell, cross-section 2c × 2(κa + b + c),
            V = 2π R₀ × A_cross  (Pappus).
    - CS  : annular cylinder, height 2(κa + b + c),
            inner radius R₀ − a − b − c − d.
    - FI  : solid cylinder, height 2(κa + b + c),
            outer radius R₀ + a + b + c.
    """
    # Blanket shell (Pappus): A_cross = 4[(a+b)(κa+b) − a·κa] = 4b[a(1+κ)+b]
    V_BB = 8 * np.pi * b * R0 * (a * (1 + κ) + b)

    # TF shell (Pappus): A_cross = 4c[a(1+κ) + 2b + c]
    V_TF = 8 * np.pi * c * R0 * (a * (1 + κ) + 2 * b + c)

    # Central solenoid: π h (R_out² − R_in²), h = 2(κa + b + c)
    R_i = R0 - a - b - c
    V_CS = 2 * np.pi * (κ * a + b + c) * (2 * d * R_i - d**2)

    # Fusion-island bounding cylinder: π R² h
    V_FI = 2 * np.pi * (κ * a + b + c) * (R0 + a + b + c)**2

    return (V_BB, V_TF, V_CS, V_FI)


# ══════════════════════════════════════════════════════════════════════════════
# RUNAWAY ELECTRON INDICATORS (POST-DISRUPTION) developped by Puel Louis
# ══════════════════════════════════════════════════════════════════════════════
#
# Indicative assessment of runaway electron (RE) generation during a tokamak
# thermal quench (TQ) and subsequent current quench (CQ).
#
# Two mechanisms are modelled:
#   1. Hot-tail seed  — Smith, Phys. Plasmas 15, 072502 (2008)
#      Profile-integrated: the local RE fraction f_RE(ρ) depends nonlinearly
#      on Te(ρ) ne(ρ) and J(ρ), so ⟨f_RE(Te)⟩ ≠ f_RE(⟨Te⟩).  The code evaluates
#      the Smith model at each radial point and integrates over the plasma
#      volume, following the same methodology as f_P_line_radiation_profile.
#
#   2. Avalanche amplification — Breizman et al., Nucl. Fusion 59, 083001 (2019)
#      Inherently 0D (relates total I_p and I_RE through a transcendental
#      equation).  Uses volume-averaged parameters.
#
# These outputs are purely diagnostic (post-convergence), they do NOT enter
# the D0FUS self-consistent solver loop.
#
# ══════════════════════════════════════════════════════════════════════════════


# ── Physical constants (RE section) ──────────────────────────────────────────
# M_E, EPS_0, C_LIGHT, E_ELEM, μ0 are all imported from D0FUS_parameterization.


# ── Coulomb logarithm (NRL Plasma Formulary) ────────────────────────────────

def _coulomb_log_ee(ne, Te_eV):
    """
    Electron-electron Coulomb logarithm (NRL Plasma Formulary).

    Parameters
    ----------
    ne    : float or ndarray  Electron density [m^-3].
    Te_eV : float or ndarray  Electron temperature [eV].

    Returns
    -------
    float or ndarray  ln(Λ_ee)
    """
    Te_eV = np.asarray(Te_eV, dtype=float)
    ne    = np.asarray(ne, dtype=float)

    ne_cm3 = np.maximum(ne * 1e-6, 1.0)           # [cm^-3], floor for safety
    Te_safe = np.maximum(Te_eV, 0.1)               # [eV], floor to avoid log(0)

    log_ee = (
        23.5
        - np.log(ne_cm3**0.5 * Te_safe**(-1.25))
        - np.sqrt(1e-5 + (np.log(Te_safe) - 2.0)**2 / 16.0)
    )
    return np.maximum(log_ee, 2.0)


def _coulomb_log_ei(ne, Te_eV, Z=1):
    """
    Electron-ion Coulomb logarithm (NRL Plasma Formulary).

    Parameters
    ----------
    ne    : float or ndarray  Electron density [m^-3].
    Te_eV : float or ndarray  Electron temperature [eV].
    Z     : float             Effective ion charge.

    Returns
    -------
    float or ndarray  ln(Λ_ei)
    """
    Te_eV = np.asarray(Te_eV, dtype=float)
    ne    = np.asarray(ne, dtype=float)

    ne_cm3 = np.maximum(ne * 1e-6, 1.0)
    Te_safe = np.maximum(Te_eV, 0.1)

    # Temperature threshold for formula switch
    threshold = 10.0 * Z**2
    log_ei = np.where(
        Te_safe < threshold,
        23.0 - np.log(ne_cm3**0.5 * Z * Te_safe**(-1.5)),
        24.0 - np.log(ne_cm3**0.5 * Te_safe**(-1.0))
    )
    return np.maximum(log_ei, 2.0)


def _coulomb_log_relativistic(ne, Te_eV):
    """
    Coulomb logarithm for a relativistic test electron in thermal plasma.

        lnΛ = ln(λ_D / λ_C)

    where λ_D = sqrt(ε₀ T_e / (n_e e²)) is the Debye length of the
    thermal background and λ_C = ℏ/(m_e c) is the reduced Compton
    wavelength (quantum-mechanical minimum impact parameter for
    ultra-relativistic electrons).

    This is the standard Coulomb logarithm that enters the
    Rosenbluth-Putvinski avalanche growth rate (R&P 1997, Eq. 16)
    and its integrated form (Breizman 2019, Eq. 99).  It represents
    the ratio of small-angle to large-angle (knock-on) collision rates
    for relativistic electrons.  Typical values: 14–16 for post-TQ
    conditions (ne ~ 10²⁰ m⁻³, Te ~ 5–20 eV).

    Parameters
    ----------
    ne    : float or ndarray  Electron density [m^-3].
    Te_eV : float or ndarray  Electron temperature [eV] (thermal background).

    Returns
    -------
    float or ndarray  lnΛ for relativistic electrons.

    Notes
    -----
    The NRL Plasma Formulary ee/ei formulas (_coulomb_log_ee/ei) are
    designed for thermal test particles and give lnΛ ~ 9 at post-TQ
    conditions, which is NOT appropriate for the R&P avalanche context.
    The sum lnΛ_ee + lnΛ_ei ~ 17 is also incorrect (physically
    meaningless).  This function provides the single, well-defined
    classical Coulomb logarithm for a relativistic test electron:
    ln(λ_D/λ_C) ~ 15 for typical post-TQ conditions.

    References
    ----------
    Rosenbluth & Putvinski, Nucl. Fusion 37, 1355 (1997).
    Breizman et al., Nucl. Fusion 59, 083001 (2019), Eqs. 92–99.
    """
    _HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]

    Te_eV = np.maximum(np.asarray(Te_eV, dtype=float), 0.1)
    ne    = np.maximum(np.asarray(ne, dtype=float), 1e15)

    # Debye length: λ_D = sqrt(ε₀ Te / (ne e²))
    lambda_D = np.sqrt(EPS_0 * Te_eV * E_ELEM / (ne * E_ELEM**2))

    # Reduced Compton wavelength: λ_C = ℏ/(m_e c)
    lambda_C = _HBAR / (M_E * C_LIGHT)   # 3.86e-13 m

    return np.maximum(np.log(lambda_D / lambda_C), 2.0)


# ── Connor-Hastie critical electric field ────────────────────────────────────

def _E_critical_connor_hastie(ne, Te_eV):
    """
    Connor-Hastie critical electric field for runaway electron generation.

    Below this field, collisional friction exceeds the accelerating force
    and no runaway electrons can be produced.

        E_c = ne · e³ · lnΛ / (4π ε₀² · mₑ · c²)

    Parameters
    ----------
    ne    : float or ndarray  Electron density [m^-3].
    Te_eV : float or ndarray  Electron temperature [eV] (for Coulomb log).

    Returns
    -------
    float or ndarray  Critical electric field [V/m].

    References
    ----------
    Connor & Hastie, Nucl. Fusion 15, 415 (1975).
    """
    ln_coul = _coulomb_log_ee(ne, Te_eV)
    return (ne * E_ELEM**3 * ln_coul
            / (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2))


# ── Thermal collision time ───────────────────────────────────────────────────

def _tau_collision_thermal(ne, Te_eV):
    """
    Electron thermal collision time [s].

    τ_c = 4π ε₀² mₑ² v_th³ / (ne e⁴ lnΛ_ee)

    where v_th = sqrt(2 Te / mₑ).

    Parameters
    ----------
    ne    : float or ndarray  Electron density [m^-3].
    Te_eV : float or ndarray  Electron temperature [eV].

    Returns
    -------
    float or ndarray  Collision time [s].
    """
    Te_eV = np.maximum(np.asarray(Te_eV, dtype=float), 0.1)
    ne    = np.maximum(np.asarray(ne, dtype=float), 1e15)

    ln_coul = _coulomb_log_ee(ne, Te_eV)
    v_th = np.sqrt(2.0 * Te_eV * E_ELEM / M_E)

    return (4.0 * np.pi * EPS_0**2 * M_E**2 * v_th**3
            / (ne * E_ELEM**4 * ln_coul))


# ══════════════════════════════════════════════════════════════════════════════
# HOT-TAIL SEED — Smith, Phys. Plasmas 15, 072502 (2008)
# ══════════════════════════════════════════════════════════════════════════════

def _hot_tail_fraction_local(ne, Te0_eV, J_local,
                              Te_final_eV=5.0, tau_TQ=1e-3,
                              Z_eff=1.0, n_times=500):
    """
    Hot-tail runaway electron fraction at a single radial location.

    Implements Smith (2008) Eq. 19: during a fast thermal quench, electrons
    in the high-energy tail of the pre-disruption distribution outrun the
    collisional thermalisation and become runaways.

    Assumptions:
    - Exponential temperature decay: Te(t) = Te_f + (Te0 - Te_f) exp(-t/τ_TQ)
    - Current density J frozen during TQ (τ_R >> τ_TQ)
    - Spitzer E-field: E‖ = η_Sp(Te(t)) × J

    Parameters
    ----------
    ne          : float  Local electron density [m^-3].
    Te0_eV      : float  Pre-disruption local electron temperature [eV].
    J_local     : float  Local current density [A/m^2] (frozen during TQ).
    Te_final_eV : float  Post-TQ residual temperature [eV] (default 5).
    tau_TQ      : float  Thermal quench e-folding time [s] (default 1 ms).
    Z_eff       : float  Effective ionic charge.
    n_times     : int    Number of time-integration points.

    Returns
    -------
    f_RE : float
        Maximum runaway fraction n_RE / ne over the TQ duration.
        Range: [0, 1].  Returns 0 if unphysical inputs.
    """
    if Te0_eV < Te_final_eV + 1.0 or J_local < 1.0 or ne < 1e15:
        return 0.0

    # Time grid: extends well past TQ to capture full seed development
    time = np.linspace(0.0, 15.0 * tau_TQ, n_times)

    # Temperature evolution during thermal quench
    Te_t = Te_final_eV + (Te0_eV - Te_final_eV) * np.exp(-time / tau_TQ)

    # Parallel electric field: E = η_Spitzer(Te(t)) × J  (J frozen)
    eta_t = eta_spitzer(Te_t * 1e-3, ne, Z_eff)   # Te_t in keV
    E_par = np.abs(J_local) * eta_t

    # Critical electric field (Connor-Hastie)
    E_c = _E_critical_connor_hastie(ne, Te_t)

    # Critical velocity: v_c = c / sqrt(E/Ec - 1)
    ratio = E_par / np.maximum(E_c, 1e-30)
    x = ratio - 1.0
    v_c = np.full_like(x, 1e12)         # Large default (no runaways)
    mask = x > 0
    v_c[mask] = C_LIGHT / np.sqrt(x[mask])

    # Pre-disruption thermal velocity
    v_th0 = np.sqrt(2.0 * Te0_eV * E_ELEM / M_E)
    if v_th0 < 1e3:
        return 0.0

    # Collision frequency at t=0
    nu0 = 1.0 / _tau_collision_thermal(ne, Te_t[0])

    # Dimensionless time τ = ν₀ · (t - τ_TQ)
    #   (shifted so τ=0 at t=τ_TQ; negative before, captures pre-TQ tail)
    tau_dimless = nu0 * (time - tau_TQ)

    # Normalised critical momentum: u_c = (v_c³/v_th0³ + 3τ)^(1/3)
    #   Smith (2008) Eq. 17 — accounts for velocity-space diffusion
    arg = (v_c**3 / v_th0**3) + 3.0 * tau_dimless
    arg = np.maximum(arg, 0.0)
    u_c = arg**(1.0 / 3.0)

    # Smith (2008) Eq. 19: runaway density fraction
    #   n_RE/ne = (2/√π) · u_c · exp(-u_c²) + erfc(u_c)
    nRE_frac = (2.0 / np.sqrt(np.pi)) * u_c * np.exp(-u_c**2) + erfc(u_c)

    # The seed saturates after the TQ — take maximum
    return float(np.nanmax(nRE_frac))


def f_hot_tail_seed_profile(nbar, Tbar, Ip, a, R0, κ, Z_eff,
                             nu_n, nu_T,
                             rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                             Te_final_eV=5.0, tau_TQ=1e-3,
                             Vprime_data=None, V=None,
                             N_rho=50, n_times=300):
    """
    Profile-integrated hot-tail runaway electron seed.

    Evaluates the Smith (2008) hot-tail model at each radial location and
    integrates the local RE density over the plasma volume.  This captures
    the dominant contribution from the hot plasma core where f_RE is
    exponentially larger than at volume-averaged conditions.

    The current density profile J(ρ) is reconstructed from the Ohmic
    conductivity profile σ(ρ) = 1/η_Sp(T(ρ), n(ρ)), normalised so that
    ∫ J dA = Ip.  This is the same approach as used in f_li (see
    D0FUS_radial_build_functions.py).

    Volume integration follows the D0FUS convention:
      - If Vprime_data is provided: uses Miller V'(ρ) flux-surface Jacobian
      - Otherwise: cylindrical weight w(ρ) = V × 2ρ

    Parameters
    ----------
    nbar       : float  Volume-averaged electron density [10²⁰ m⁻³].
    Tbar       : float  Volume-averaged electron temperature [keV].
    Ip         : float  Plasma current [MA].
    a          : float  Minor radius [m].
    R0         : float  Major radius [m].
    κ          : float  Plasma elongation (LCFS).
    Z_eff      : float  Effective ionic charge.
    nu_n       : float  Density peaking exponent.
    nu_T       : float  Temperature peaking exponent.
    rho_ped    : float  Normalised pedestal radius (1.0 → parabolic).
    n_ped_frac : float  n_ped / n̄.
    T_ped_frac : float  T_ped / T̄.
    Te_final_eV: float  Post-TQ residual temperature [eV] (default 5).
    tau_TQ     : float  Thermal quench e-folding time [s] (default 1 ms).
    Vprime_data: tuple  Precomputed (rho_grid, Vprime, V_total) from
                        precompute_Vprime, or None for cylindrical mode.
    V          : float  Plasma volume [m³] (cylindrical mode; ignored if
                        Vprime_data is provided).
    N_rho      : int    Number of radial integration points (default 50).
    n_times    : int    Time points for each local hot-tail evaluation.

    Returns
    -------
    dict with keys:
        'N_RE'       : float  Total RE seed count [dimensionless].
        'I_RE_seed'  : float  RE seed current [A] (assuming v‖ ≈ c).
        'f_RE_avg'   : float  Volume-averaged RE fraction n_RE / ne.
        'f_RE_core'  : float  RE fraction at ρ=0 (peak).
        'f_RE_profile': ndarray  f_RE(ρ) array for diagnostics.
        'rho'        : ndarray  Radial grid used.

    Notes
    -----
    The RE seed current is estimated as I_RE = e × c × ∫ f_RE × ne × dA,
    where dA is the poloidal cross-section area element.  This assumes all
    seed runaways are fully relativistic (v‖ ≈ c), which overestimates the
    seed current for marginally runaway electrons.

    References
    ----------
    Smith, Phys. Plasmas 15, 072502 (2008).
    Stahl et al., Nucl. Fusion 56, 112009 (2016) — Fig. 2(b), isotropic runaway region.
    """
    Ip_A  = Ip * 1e6                    # [MA] -> [A]
    ne_SI = nbar * 1e20                  # [10²⁰ m⁻³] -> [m⁻³]

    # ── Build radial grid ────────────────────────────────────────────────────
    if Vprime_data is not None:
        rho_grid, Vprime, V_total = Vprime_data
        # Subsample if Vprime grid is finer than N_rho
        if len(rho_grid) > N_rho:
            idx = np.linspace(0, len(rho_grid) - 1, N_rho, dtype=int)
            rho   = rho_grid[idx]
            Vp    = Vprime[idx]
        else:
            rho = rho_grid.copy()
            Vp  = Vprime.copy()
        use_miller = True
    else:
        rho = np.linspace(1e-4, 0.98, N_rho)
        Vp  = None
        use_miller = False
        if V is None or V <= 0:
            V = 2.0 * np.pi**2 * R0 * a**2 * κ   # Approx. torus volume

    # ── Local profiles ───────────────────────────────────────────────────────
    n_hat  = f_nprof(1.0,  nu_n, rho, rho_ped, n_ped_frac,
                     Vprime_data)                                # n(ρ)/nbar
    T_arr  = f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac,
                     Vprime_data)                                # T(ρ) [keV]

    ne_rho = ne_SI * n_hat                                     # [m⁻³]
    Te_rho = np.maximum(T_arr * 1e3, 1.0)                     # [eV], floor 1 eV

    # ── Current density profile J(ρ) ─────────────────────────────────────────
    # σ(ρ) = 1 / η_Spitzer(T(ρ), ne(ρ)), then J ∝ σ normalised to Ip.
    # J frozen during TQ: τ_R = μ₀ a² / η >> τ_TQ ≈ 1 ms.
    T_keV_safe = np.maximum(T_arr, 1e-4)
    eta_rho = eta_spitzer(T_keV_safe, ne_rho, Z_eff)
    sigma_rho = 1.0 / np.maximum(eta_rho, 1e-15)

    # Cross-section area element dA = 2π a κ ρ dρ (cylindrical approximation)
    # ∫ J dA = Ip  →  J(ρ) = Ip × σ(ρ) / ∫ σ(ρ) 2πaκρ dρ
    sigma_area_int = np.trapezoid(sigma_rho * rho, rho)
    if sigma_area_int < 1e-20:
        # Degenerate case — uniform J fallback
        J_rho = Ip_A / (np.pi * a**2 * κ) * np.ones_like(rho)
    else:
        J_rho = Ip_A * sigma_rho / (2.0 * np.pi * a * κ * sigma_area_int)

    # ── Hot-tail evaluation — vectorised over radial grid (2D broadcasting) ──
    # Shape convention: ne_rho, Te_rho, J_rho are (N_rho,)
    #                   time grid is (n_times,)
    #                   2D arrays are (N_rho, n_times)

    # Validity mask: skip radial points where seed is trivially zero
    valid_ht = (Te_rho > Te_final_eV + 1.0) & (J_rho > 1.0) & (ne_rho > 1e15)
    f_RE = np.zeros_like(rho)

    if np.any(valid_ht):
        # Broadcast valid slices to 2D: axis 0 = radial, axis 1 = time
        ne_v  = ne_rho[valid_ht, np.newaxis]         # (N_valid, 1)
        Te0_v = Te_rho[valid_ht, np.newaxis]         # (N_valid, 1)
        J_v   = np.abs(J_rho[valid_ht, np.newaxis])  # (N_valid, 1)

        time = np.linspace(0.0, 15.0 * tau_TQ, n_times)[np.newaxis, :]  # (1, n_times)

        # Temperature evolution during thermal quench
        Te_t = Te_final_eV + (Te0_v - Te_final_eV) * np.exp(-time / tau_TQ)

        # Parallel electric field: E = η_Spitzer(Te(t)) × J  (J frozen)
        eta_t = eta_spitzer(Te_t * 1e-3, ne_v, Z_eff)
        E_par = J_v * eta_t

        # Connor-Hastie critical electric field
        E_c = _E_critical_connor_hastie(ne_v, Te_t)

        # Critical velocity: v_c = c / sqrt(E/Ec - 1)
        ratio = E_par / np.maximum(E_c, 1e-30)
        x = ratio - 1.0
        v_c = np.where(x > 0, C_LIGHT / np.sqrt(np.maximum(x, 1e-30)), 1e12)

        # Pre-disruption thermal velocity (per radial point)
        v_th0 = np.sqrt(2.0 * Te0_v * E_ELEM / M_E)  # (N_valid, 1)

        # Collision frequency at t=0: ν₀(ρ) = 1/τ_c(ne, Te0)
        nu0 = 1.0 / _tau_collision_thermal(ne_v[:, 0], Te0_v[:, 0])  # (N_valid,)
        nu0 = nu0[:, np.newaxis]  # (N_valid, 1) for broadcasting

        # Dimensionless time τ = ν₀ · (t - τ_TQ)
        tau_dimless = nu0 * (time - tau_TQ)

        # Normalised critical momentum: u_c = (v_c³/v_th0³ + 3τ)^(1/3)
        arg = (v_c**3 / v_th0**3) + 3.0 * tau_dimless
        arg = np.maximum(arg, 0.0)
        u_c = arg**(1.0 / 3.0)

        # Smith (2008) Eq. 19: runaway density fraction
        nRE_frac = (2.0 / np.sqrt(np.pi)) * u_c * np.exp(-u_c**2) + erfc(u_c)

        # Take maximum over time axis (seed saturates after TQ)
        # Mask out radial points with v_th0 too small
        v_th0_1d = v_th0[:, 0]
        max_frac = np.nanmax(nRE_frac, axis=1)
        max_frac = np.where(v_th0_1d > 1e3, max_frac, 0.0)

        f_RE[valid_ht] = max_frac

    # ── Volume integration of RE seed density ────────────────────────────────
    # N_RE  = ∫ f_RE(ρ) × ne(ρ) × V'(ρ) dρ          (total RE count)
    # I_RE  = e × c × ∫ f_RE(ρ) × ne(ρ) × dA(ρ)     (seed current)
    if use_miller:
        integrand_N = f_RE * ne_rho * Vp
        N_RE = float(np.trapezoid(integrand_N, rho))

        # For I_RE: approximate dA ≈ Vp / (2πR0)
        integrand_I = f_RE * ne_rho * Vp / (2.0 * np.pi * R0)
        I_RE = E_ELEM * C_LIGHT * float(np.trapezoid(integrand_I, rho))
    else:
        # Cylindrical: dV = V × 2ρ dρ,  dA = πa²κ × 2ρ dρ
        integrand_N = f_RE * ne_rho * 2.0 * rho
        N_RE = V * float(np.trapezoid(integrand_N, rho))

        area_CS = np.pi * a**2 * κ
        integrand_I = f_RE * ne_rho * 2.0 * rho
        I_RE = E_ELEM * C_LIGHT * area_CS * float(
            np.trapezoid(integrand_I, rho))

    # Volume-averaged RE fraction
    if use_miller:
        ne_total = float(np.trapezoid(ne_rho * Vp, rho))
    else:
        ne_total = V * float(np.trapezoid(ne_rho * 2.0 * rho, rho))
    f_RE_avg = N_RE / ne_total if ne_total > 0 else 0.0

    return {
        'N_RE':        N_RE,
        'I_RE_seed':   I_RE,           # [A]
        'f_RE_avg':    f_RE_avg,
        'f_RE_core':   f_RE[0],        # RE fraction at magnetic axis
        'f_RE_profile': f_RE,
        'rho':         rho,
        'J_profile':   J_rho,          # [A/m²] for diagnostics
    }


# ══════════════════════════════════════════════════════════════════════════════
# AVALANCHE AMPLIFICATION — Breizman et al., NF 59, 083001 (2019) Eq. 99
# ══════════════════════════════════════════════════════════════════════════════

def f_RE_avalanche(I0, Ire0, ne, Te_eV, li, Z_eff, IA=17e3):
    """
    Final runaway electron current after avalanche multiplication.

    Solves the transcendental equation (Breizman 2019, Eq. 99):

        ln(I_RE∞ / I_RE0) = [li / (√(Z+5) · lnΛ)] × (I₀ − I_RE∞) / I_A

    where I_A ≈ 17 kA is the Alfvén current.  The avalanche converts the
    remaining Ohmic current into RE current through knock-on collisions.

    Parameters
    ----------
    I0     : float  Initial total plasma current [A].
    Ire0   : float  Hot-tail RE seed current [A] (> 0).
    ne     : float  Volume-averaged electron density [m^-3].
    Te_eV  : float  Post-TQ electron temperature [eV] (typically 5–20 eV).
    li     : float  Normalised internal inductance.
    Z_eff  : float  Effective ionic charge.
    IA     : float  Alfvén current [A] (default 17 kA).

    Returns
    -------
    Ire_inf : float
        Final RE current after avalanche [A].
        Returns Ire0 if no physical solution exists (subcritical seed).

    Notes
    -----
    The Coulomb logarithm lnΛ in Eq. 99 is the standard single
    Coulomb logarithm for a relativistic test electron in thermal
    plasma: lnΛ = ln(λ_D / λ_C), where λ_D is the Debye length
    and λ_C = ℏ/(m_e c) is the reduced Compton wavelength.  This
    gives lnΛ ≈ 14–16 at typical post-TQ conditions (ne ~ 10²⁰ m⁻³,
    Te ~ 5–20 eV).  See _coulomb_log_relativistic for details.

    This is the same Coulomb logarithm that appears in the R&P (1997)
    avalanche growth rate (Eq. 92) and its strong-field limit (Eq. 93).
    It represents the ratio of small-angle to large-angle (knock-on)
    collision rates; it is NOT a sum of ee and ei contributions.

    At very low seed currents (Ire0 → 0), the avalanche can amplify by
    many orders of magnitude.  At Ire0 ≈ I0, the solution is trivially
    Ire∞ ≈ I0 (full conversion).

    References
    ----------
    Breizman et al., Nucl. Fusion 59, 083001 (2019), Eq. 99, Fig. 17.
    """
    if Ire0 <= 0 or I0 <= 0 or li <= 0:
        return max(Ire0, 0.0)

    # Coulomb logarithm for relativistic electrons (single, standard).
    # lnΛ = ln(λ_D / λ_C) with λ_C = ℏ/(mc), the quantum minimum
    # impact parameter for ultra-relativistic test particles.
    # This is the lnΛ that enters the R&P avalanche growth rate
    # (Eq. 92-93) and Breizman Eq. 99.  Typical value: 14-16.
    lnLambda = float(_coulomb_log_relativistic(ne, Te_eV))

    coef = li / (np.sqrt(Z_eff + 5.0) * lnLambda)

    def _equation(Ire_inf):
        """Residual: ln(Ire∞/Ire0) - coef × (I0 - Ire∞) / IA = 0"""
        if Ire_inf <= 0:
            return -1e30
        return np.log(Ire_inf / Ire0) - coef * (I0 - Ire_inf) / IA

    # Bracket: [Ire0, I0] — physical range (RE current cannot exceed Ip)
    try:
        # Check sign change
        f_lo = _equation(max(Ire0 * 0.99, 1e-20))
        f_hi = _equation(I0 * 0.999)

        if f_lo * f_hi > 0:
            # No sign change — subcritical: avalanche does not amplify
            return Ire0

        sol = root_scalar(
            _equation,
            bracket=[max(Ire0 * 0.99, 1e-20), I0 * 0.999],
            method='brentq',
            xtol=1e-3
        )
        return sol.root

    except (ValueError, RuntimeError):
        return Ire0


# ══════════════════════════════════════════════════════════════════════════════
# MASTER WRAPPER — Full RE assessment for a single design point
# ══════════════════════════════════════════════════════════════════════════════

def compute_RE_indicators(Ip, nbar, Tbar, a, R0, κ, Z_eff, li,
                           nu_n, nu_T,
                           rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                           Te_final_eV=5.0, tau_TQ=1e-3,
                           Vprime_data=None, V=None,
                           N_rho=50, n_times=300,
                           pellet_dilution=10.0):
    """
    Runaway electron (RE) risk indicators for a D0FUS design point.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  INDICATOR ONLY — NOT A PREDICTIVE TOOL                              ║
    ║                                                                      ║
    ║  The quantities returned by this function are ORDER-OF-MAGNITUDE     ║
    ║  estimates intended solely for COMPARATIVE RANKING of designs.       ║
    ║  They cannot and should not be used as absolute predictions of the   ║
    ║  RE current expected in a real disruption.                           ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Physical scenario
    -----------------
    This function models the safe-landing (mitigated disruption) scenario
    in which the disruption is *triggered* by the injection of a shattered
    pellet (SPI) or a massive gas injection (MGI).  The pellet raises n_e
    before the thermal quench onset, so the elevated density is seen by the
    hot-tail electrons during the quench itself.  Consequently:

      - Both the hot-tail seed and the avalanche amplification are computed
        at the post-pellet density  n_e,diluted = pellet_dilution × n_e,pre.
      - A higher n_e raises E_c ∝ n_e, suppresses hot-tail generation, and
        increases colisionality, all of which reduce I_RE.

    To model an unmitigated disruption (no SPI/MGI, pellet arrives after
    the TQ), set ``pellet_dilution = 1.0``.

    Physical upper bound
    --------------------
    Both I_RE_seed and I_RE_avalanche are capped at Ip (the total plasma
    current).  The RE current cannot physically exceed the pre-disruption
    plasma current because the available poloidal flux is fixed by L_p × Ip.
    This cap is most relevant for large-Ip, low-density designs where the
    avalanche model would otherwise return unphysical values > Ip.

    Calculation sequence
    --------------------
    Step 1 — Hot-tail seed (Smith 2008, Phys. Plasmas 15, 072502):
        Profile-integrated over T(ρ) and n_diluted(ρ) using post-pellet
        density.  Outputs I_RE_seed [A] (capped at Ip).

    Step 2 — Avalanche amplification (Breizman 2019, NF 59, 083001 Eq. 99):
        Uses n_e,diluted throughout.  Output I_RE_avalanche [A] (capped at Ip).

    Step 3 — Energy estimates:
        Kinetic energy: E_kin = N_RE × ⟨γ⟩ × m_e c²  (⟨γ⟩ = 10, indicative).
        Magnetic energy: W_mag = ½ L_i I_RE²  (dominates wall damage).

    Parameters
    ----------
    Ip         : float  Plasma current [MA].
    nbar       : float  Volume-averaged pre-pellet electron density [10²⁰ m⁻³].
                        Used to compute nbar_diluted = pellet_dilution × nbar.
    Tbar       : float  Volume-averaged electron temperature [keV].
    a          : float  Minor radius [m].
    R0         : float  Major radius [m].
    κ          : float  Plasma elongation (LCFS).
    Z_eff      : float  Effective ionic charge.
    li         : float  Normalised internal inductance.
    nu_n       : float  Density peaking exponent.
    nu_T       : float  Temperature peaking exponent.
    rho_ped    : float  Normalised pedestal radius (1.0 → parabolic).
    n_ped_frac : float  n_ped / n̄.
    T_ped_frac : float  T_ped / T̄.
    Te_final_eV: float  Post-TQ residual temperature [eV] (default 5).
    tau_TQ     : float  Thermal quench e-folding time [s] (default 1 ms).
    Vprime_data: tuple  Precomputed (rho_grid, Vprime, V_total) or None.
    V          : float  Plasma volume [m³] (cylindrical mode).
    N_rho      : int    Radial points for profile integration (default 50).
    n_times    : int    Time points per local hot-tail evaluation (default 300).
    pellet_dilution : float
        Density multiplication factor representing shattered-pellet or MGI
        assimilation into the background plasma before the current quench
        (default 10).  Applied to nbar for the avalanche step only.
        Set to 1.0 to disable pellet mitigation (unmitigated disruption).

    Returns
    -------
    dict with keys
        'I_RE_seed'      : float  Hot-tail seed current [A], at n_diluted,
                                  capped at Ip.
        'I_RE_avalanche' : float  Final RE current after avalanche [A],
                                  at n_diluted, capped at Ip.
        'f_RE_to_Ip'     : float  I_RE∞ / I_p  [-]  (∈ [0, 1] by construction).
        'N_RE_seed'      : float  Total seed electron count.
        'f_RE_avg'       : float  Volume-averaged hot-tail seed fraction.
        'f_RE_core'      : float  Seed fraction at magnetic axis (ρ=0).
        'E_RE_kin'       : float  RE beam kinetic energy [MJ]  (⟨γ⟩=10,
                                  indicative order of magnitude only).
        'W_mag_RE'       : float  RE beam magnetic energy [MJ]
                                  (½ L_internal × I_RE²; dominant damage
                                  channel during current quench).
        'hot_tail_detail': dict   Full hot-tail output from
                                  f_hot_tail_seed_profile.
        'tau_TQ'         : float  Thermal quench time used [s].
        'Te_final_eV'    : float  Post-TQ temperature used [eV].
        'nbar_diluted'   : float  Post-pellet density used [10²⁰ m⁻³].
        'pellet_dilution': float  Density multiplication factor used.

    References
    ----------
    Smith, Phys. Plasmas 15, 072502 (2008)        — hot-tail seed model.
    Breizman et al., NF 59, 083001 (2019)         — avalanche amplification.
    Lehnen et al., Nucl. Fusion 55, 123027 (2015) — ITER SPI specifications.
    Reux et al., Nucl. Fusion 61, 116054 (2021)   — DEMO disruption mitigation.
    Martin-Solis et al., PRL 105, 185002 (2010)   — Avalanche generation.
    """
    
    Ip_A  = Ip * 1e6                    # [MA] → [A]
    ne_SI = nbar * 1e20                 # [10²⁰ m⁻³] → [m⁻³]

    # ── Post-pellet density ───────────────────────────────────────────────────
    # In the safe-landing scenario modelled here, the disruption is itself
    # *triggered* by the pellet injection (SPI/MGI).  The pellet rapidly
    # raises n_e before the thermal quench onset, so both the hot-tail seed
    # and the avalanche are computed at the diluted density.
    # This differs from an unmitigated VDE (vertical displacement event) where
    # the pre-disruption density would apply to the seed.
    # Set pellet_dilution = 1.0 to recover the unmitigated-disruption scenario.
    nbar_diluted  = pellet_dilution * nbar          # [10²⁰ m⁻³]
    ne_SI_diluted = nbar_diluted * 1e20             # [m⁻³]

    # ── Step 1: Hot-tail seed — evaluated at post-pellet density ─────────────
    # Both n_e and the Coulomb logarithm in the Smith model use nbar_diluted.
    ht = f_hot_tail_seed_profile(
        nbar_diluted, Tbar, Ip, a, R0, κ, Z_eff,
        nu_n, nu_T,
        rho_ped=rho_ped, n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
        Te_final_eV=Te_final_eV, tau_TQ=tau_TQ,
        Vprime_data=Vprime_data, V=V,
        N_rho=N_rho, n_times=n_times
    )
    # Physical upper bound: seed current cannot exceed total plasma current
    I_RE_seed = min(ht['I_RE_seed'], Ip_A)

    # ── Step 2: Avalanche amplification — post-pellet density ─────────────────
    if I_RE_seed > 0:
        I_RE_final = f_RE_avalanche(
            Ip_A, I_RE_seed, ne_SI_diluted, Te_final_eV, li, Z_eff
        )
    else:
        I_RE_final = 0.0
    # Physical upper bound: RE current cannot exceed total plasma current
    I_RE_final = min(I_RE_final, Ip_A)

    # ── Step 3: Derived quantities ────────────────────────────────────────────
    f_RE_to_Ip = I_RE_final / Ip_A if Ip_A > 0 else 0.0

    # Kinetic energy: E_kin = N_RE × ⟨γ⟩ × m_e c²
    # ⟨γ⟩ = 10 is a conventional indicative value for disruption RE beams
    # (Hender et al. 2007, NF 47 S128).  Actual γ spans 2–100+ depending
    # on the electric field and current-quench duration.
    N_RE_final = I_RE_final / (E_ELEM * C_LIGHT)
    gamma_avg  = 10.0
    E_RE_kin   = N_RE_final * gamma_avg * M_E * C_LIGHT**2 * 1e-6  # [MJ]

    # Magnetic energy: W_mag = ½ L_internal × I_RE²
    # This is the principal energy channel for first-wall damage during the
    # CQ.  The internal inductance L_i = μ₀ R₀ li / 2 [H].
    L_internal = μ0 * R0 * li / 2.0
    W_mag      = 0.5 * L_internal * I_RE_final**2 * 1e-6  # [MJ]

    return {
        'I_RE_seed':       I_RE_seed,         # [A]   — pre-pellet hot-tail seed
        'I_RE_avalanche':  I_RE_final,         # [A]   — post-avalanche, post-pellet
        'f_RE_to_Ip':      f_RE_to_Ip,         # [-]
        'N_RE_seed':       ht['N_RE'],
        'f_RE_avg':        ht['f_RE_avg'],
        'f_RE_core':       ht['f_RE_core'],
        'E_RE_kin':        E_RE_kin,            # [MJ]  — indicative, ⟨γ⟩=10
        'W_mag_RE':        W_mag,               # [MJ]  — dominant damage channel
        'hot_tail_detail': ht,
        'tau_TQ':          tau_TQ,              # [s]
        'Te_final_eV':     Te_final_eV,         # [eV]
        'nbar_diluted':    nbar_diluted,        # [10²⁰ m⁻³] — post-pellet density
        'pellet_dilution': pellet_dilution,     # [-]   — assimilation factor used
    }


# ── Validation ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Helium ash fraction f_alpha vs C_alpha — ITER Q=10 validation
    # Console: q95, neutron wall load, f_alpha reference check
    # Figure:  f_alpha(C_alpha) sensitivity curve — academic vs H-mode pedestal

    # ITER Q=10 reference parameters (Shimada et al., NF 47 (2007) S1)
    # Shimada Table 1 reports ψ_N=0.95 values: κ₉₅=1.70, δ₉₅=0.33.
    # LCFS values from ITER baseline (single-null): κ_edge≈1.85, δ_edge≈0.49.
    _R0, _a               = 6.2, 2.0
    _κ_edge, _δ_edge      = 1.85, 0.49     # LCFS shaping (Sauter formula)
    _κ_95,   _δ_95        = 1.70, 0.33     # 95%-surface shaping (ITER_1989)
    _B0, _Ip              = 5.3, 15.0
    _nbar, _Tbar, _tauE   = 1.0, 8.9, 3.7
    _P_fus, _nu_T, _C_α   = 500.0, 1.0, 5.0

    _q95     = f_q95(_B0, _Ip, _R0, _a, _κ_edge, _δ_edge, _κ_95, _δ_95)
    _Gamma_n = f_Gamma_n(_a, _P_fus, _R0, _κ_edge)
    _f_alpha = f_He_fraction(_nbar, _Tbar, _tauE, _C_α, _nu_T)

    print("\n── Safety factor q₉₅ — ITER Q=10 ──────────────────────────────────────────")
    print(f"  {'Quantity':<20} {'D0FUS':>8}  {'ITER ref.':>10}")
    print("  " + "─" * 58)
    print(f"  {'q₉₅':<20} {_q95:>8.2f}  {'3.0':>10}")

    print("\n── Neutron wall load Γ_n — ITER Q=10 ──────────────────────────────────────")
    print(f"  {'Quantity':<20} {'D0FUS':>8}  {'ITER ref.':>10}")
    print("  " + "─" * 58)
    print(f"  {'Γ_n [MW/m²]':<20} {_Gamma_n:>8.3f}  {'0.57':>10}")

    print("\n── Helium ash fraction f_α — ITER Q=10 ────────────────────────────────────")
    print(f"  {'Quantity':<20} {'D0FUS':>8}  {'ITER ref.':>10}")
    print("  " + "─" * 58)
    print(f"  {f'f_α [%]  (C_α={_C_α:.0f})':<20} {_f_alpha*100:>8.1f}  {'5':>10}")

    # ── Runaway electrons — ITER Q=10 ────────────────────────────────────────
    # Hot-tail seed: Smith, Phys. Plasmas 15, 072502 (2008)
    # Avalanche:     Breizman et al., Nucl. Fusion 59, 083001 (2019) Eq. 99

    print("\n── Hot-tail seed — Stahl (2016) benchmark ──────────────────────────────────")
    print(f"  {'Quantity':<20} {'D0FUS':>12}  {'Reference':>12}  Source")
    print("  " + "─" * 58)
    _f_RE_stahl = _hot_tail_fraction_local(
        2.8e19, 3.1e3, 1.4e6, Te_final_eV=31.0, tau_TQ=0.3e-3, Z_eff=1.0)
    print(f"  {'f_RE (local)':<20} {_f_RE_stahl:>12.3e}  {'4-5e-4':>12}  Stahl (2016) Fig.2(b)")

    print("\n── Avalanche — Breizman (2019) Fig. 17 (li=1, Z=4, Ip=15 MA) ───────────────")
    print(f"  lnΛ = ln(λ_D/λ_C) = {_coulomb_log_relativistic(1e20, 5.0):.2f}  "
          f"(relativistic, ne=10²⁰, Te=5 eV)")
    print(f"  {'I_RE0 [A]':<20} {'D0FUS [MA]':>12}  {'Eq.99 [MA]':>12}  {'Fig.17':>12}")
    print("  " + "─" * 62)
    for _ire0, _ref99, _fig17 in [(1.0, 3.306, "~2-3"), (1e3, 7.999, "~7-9"), (1e6, 13.002, "~12-14")]:
        _ire_inf = f_RE_avalanche(15e6, _ire0, 1e20, 5.0, 1.0, 4)
        print(f"  {_ire0:<20.0e} {_ire_inf/1e6:>12.3f}  {_ref99:>12.3f}  {_fig17:>12}")

    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_He_fraction(C_Alpha=_C_α)

#%%

# print("D0FUS_physical_functions loaded")