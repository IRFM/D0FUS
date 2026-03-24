"""
D0FUS_figures.py
================
Centralised visualisation module for the D0FUS project.

All functions in this file are *pure render wrappers*: they contain no physics
calculations and depend exclusively on functions from D0FUS_physical_functions
and D0FUS_radial_build_functions.  The physics lives in those modules; this
module only decides how to display it.

Design conventions
------------------
* Every public function accepts an optional ``save_dir`` keyword argument.
  - ``save_dir=None``  (default) → ``plt.show()`` is called, figure is
    displayed interactively.
  - ``save_dir="path/to/dir"`` → the figure is saved as a PNG (150 dpi) and
    the axes object is closed immediately (non-interactive batch mode).
* Functions are named ``plot_<topic>`` and grouped by theme:
    1. Plasma shaping (κ, δ profiles, flux surfaces)
    2. Kinetic profiles (n, T, p)
    3. Nuclear / radiation physics (L_z, radiation channels)
    4. Superconductor / cable engineering (J_c scalings, CICC)
    5. Structural mechanics (CIRCE stress, TF/CS thickness)
    6. Resistivity models
* Default parameter values reproduce the ITER Q=10 reference geometry
  (Shimada et al., Nucl. Fusion 47, 2007, S1) wherever possible.

Author  : Auclair Timothe
Created : 2025
"""

# =============================================================================
# Imports
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

# Package-relative imports (production mode)
if __name__ != "__main__":
    from .D0FUS_physical_functions import (
        f_Kappa, f_Kappa_95, f_Delta_95,
        kappa_profile, delta_profile, miller_RZ,
        f_first_wall_surface,
        f_nprof, f_Tprof,
        f_nbar_line,
        get_Lz,
        f_He_fraction,
        f_q_profile,
    )
    from .D0FUS_radial_build_functions import (
        J_non_Cu_NbTi, J_non_Cu_Nb3Sn, J_non_Cu_REBCO,
        calculate_cable_current_density,
        eta_old, eta_spitzer, eta_sauter, eta_redl,
        f_TF_academic, f_TF_D0FUS,
        f_CS_ACAD, f_CS_D0FUS, f_CS_CIRCE,
        F_CIRCE0D, compute_von_mises_stress,
        calculate_E_mag_TF,
    )
    from .D0FUS_parameterization import DEFAULT_CONFIG, E_ELEM

# Standalone-execution imports (development / testing)
else:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from D0FUS_BIB.D0FUS_physical_functions import (
        f_Kappa, f_Kappa_95, f_Delta_95,
        kappa_profile, delta_profile, miller_RZ,
        f_first_wall_surface,
        f_nprof, f_Tprof,
        f_nbar_line,
        get_Lz,
        f_He_fraction,
        f_q_profile,
    )
    from D0FUS_BIB.D0FUS_radial_build_functions import (
        J_non_Cu_NbTi, J_non_Cu_Nb3Sn, J_non_Cu_REBCO,
        calculate_cable_current_density,
        eta_old, eta_spitzer, eta_sauter, eta_redl,
        f_TF_academic, f_TF_D0FUS,
        f_CS_ACAD, f_CS_D0FUS, f_CS_CIRCE,
        F_CIRCE0D, compute_von_mises_stress,
        calculate_E_mag_TF,
    )
    from D0FUS_BIB.D0FUS_parameterization import DEFAULT_CONFIG, E_ELEM


# =============================================================================
# Internal utilities
# =============================================================================

def _save_or_show(fig: plt.Figure, save_dir: str | None, fname: str) -> None:
    """
    Either save ``fig`` to ``save_dir/fname.png`` or call ``plt.show()``.

    Parameters
    ----------
    fig      : matplotlib Figure
    save_dir : str or None
        Target directory.  Created automatically if it does not exist.
        Pass ``None`` to display interactively.
    fname    : str
        File name *without* extension (PNG is always used).
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{fname}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# 1. Plasma shaping
# =============================================================================

def plot_kappa_scaling(
    A_min: float = 1.5,
    A_max: float = 5.0,
    kappa_manual: float = 1.7,
    ms: float = 0.3,
    save_dir: str | None = None,
) -> None:
    """
    Plot maximum achievable elongation κ_edge and κ_95 as a function of
    aspect ratio A = R₀/a for three empirical scaling laws.

    Produces two side-by-side figures:
      - Left  : κ_edge vs A  (Stambaugh, Freidberg, Wenninger)
      - Right : κ_95   vs A  (derived via f_Kappa_95)

    Parameters
    ----------
    A_min, A_max   : float  Aspect-ratio scan bounds [-].
    kappa_manual   : float  Value used when Option_Kappa='Manual' [-].
    ms             : float  Miller-stability margin parameter [-].
    save_dir       : str or None

    References
    ----------
    Stambaugh et al. (1992); Freidberg (2007); Wenninger et al. (2014).
    """
    A_arr = np.linspace(A_min, A_max, 300)
    styles = [
        ("Stambaugh", "tab:blue"),
        ("Freidberg", "tab:orange"),
        ("Wenninger", "tab:green"),
    ]

    # --- Figure 1a : κ_edge vs A ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    for label, color in styles:
        ax.plot(A_arr,
                f_Kappa(A_arr, label, κ_manual=kappa_manual, ms=ms),
                color=color, lw=2, label=label)
    ax.set_xlabel(r"$A = R_0/a$", fontsize=12)
    ax.set_ylabel(r"$\kappa_{\rm edge}$", fontsize=12)
    ax.set_title("Maximum achievable κ_edge vs aspect ratio", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Figure 1b : κ_95 vs A ---
    ax = axes[1]
    for label, color in styles:
        ax.plot(A_arr,
                f_Kappa_95(f_Kappa(A_arr, label, κ_manual=kappa_manual, ms=ms)),
                color=color, lw=2, label=label)
    ax.set_xlabel(r"$A = R_0/a$", fontsize=12)
    ax.set_ylabel(r"$\kappa_{95}$", fontsize=12)
    ax.set_title("Maximum achievable κ_95 vs aspect ratio", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Elongation scaling laws", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "kappa_scaling")


def plot_shaping_profiles(
    kappa_edge: float = 1.85,
    delta_edge: float = 0.50,
    rho_95: float = 0.95,
    save_dir: str | None = None,
) -> None:
    """
    Plot radial elongation κ(ρ) and triangularity δ(ρ) profiles.

    Both the 'Academic' (constant-κ, δ=0) and the 'D0FUS PCHIP' profiles are
    shown.  For δ, both positive and negative triangularity cases are
    superimposed.

    Parameters
    ----------
    kappa_edge : float  Edge elongation [-].
    delta_edge : float  Edge triangularity (positive D-shape value) [-].
    rho_95     : float  Normalised flux surface at 95 % of poloidal flux [-].
    save_dir   : str or None

    References
    ----------
    Christiansen et al., Nucl. Fusion 32, 291 (1992) — PCHIP parameterisation.
    Ball & Parra, PPCF 57, 045006 (2015) — κ core penetration.
    """
    rho = np.linspace(1e-4, 1.0, 500)
    kappa_95 = f_Kappa_95(kappa_edge)
    delta_95  = f_Delta_95(delta_edge)
    delta_edge_neg = -delta_edge
    delta_95_neg   = f_Delta_95(delta_edge_neg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- κ(ρ) panel ---
    ax = axes[0]
    ax.axhline(kappa_edge, color="tab:blue", lw=2, ls="-",
               label=f"Academic: κ = κ_edge = {kappa_edge}  (const)")
    ax.plot(rho, kappa_profile(rho, kappa_edge, kappa_95),
            color="tab:red", lw=2,
            label="D0FUS: flat core at κ₉₅, edge rise (PCHIP)")
    ax.axvline(rho_95,   color="gray", lw=1, ls="--", alpha=0.7)
    ax.axhline(kappa_95, color="gray", lw=1, ls="--", alpha=0.7)
    ax.plot(rho_95, kappa_95, "o", color="tab:red", ms=7, zorder=5,
            label=f"κ₉₅ = {kappa_95:.3f}  (ρ₉₅ = {rho_95})")
    ax.set_xlabel(r"$\rho = r/a$", fontsize=12)
    ax.set_ylabel(r"$\kappa(\rho)$", fontsize=12)
    ax.set_title("Elongation profile\n"
                 "κ penetrates efficiently to the core (Ball & Parra 2015)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(1.1, kappa_edge * 1.06)

    # --- δ(ρ) panel ---
    ax = axes[1]
    ax.axhline(0, color="tab:blue", lw=2, ls="-", label="Academic: δ = 0")
    ax.plot(rho, delta_profile(rho, delta_edge, delta_95),
            color="tab:red", lw=2,
            label=f"D0FUS PCHIP: δ_edge = +{delta_edge} (D-shape)")
    ax.plot(rho, delta_profile(rho, delta_edge_neg, delta_95_neg),
            color="tab:purple", lw=2,
            label=f"D0FUS PCHIP: δ_edge = {delta_edge_neg} (neg. triang.)")
    ax.axvline(rho_95,   color="gray",       lw=1, ls="--", alpha=0.7)
    ax.axhline(delta_95, color="tab:red",    lw=1, ls="--", alpha=0.6)
    ax.plot(rho_95, delta_95, "o", color="tab:red", ms=7, zorder=5,
            label=f"δ₉₅ = +{delta_95:.3f}  (ρ₉₅ = {rho_95})")
    ax.axhline(delta_95_neg, color="tab:purple", lw=1, ls="--", alpha=0.6)
    ax.plot(rho_95, delta_95_neg, "o", color="tab:purple", ms=7, zorder=5,
            label=f"δ₉₅ = {delta_95_neg:.3f}  (ρ₉₅ = {rho_95})")
    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlabel(r"$\rho = r/a$", fontsize=12)
    ax.set_ylabel(r"$\delta(\rho)$", fontsize=12)
    ax.set_title("Triangularity profile\n"
                 "δ confined to edge layer — positive & negative triangularity",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(delta_edge_neg * 1.08, delta_edge * 1.08)

    plt.suptitle(
        f"Radial shaping profiles — κ_edge = {kappa_edge}, δ_edge = ±{delta_edge}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "shaping_profiles")


def plot_miller_surfaces(
    R0: float = 6.2,
    a: float = 2.0,
    kappa_edge: float = 1.85,
    delta_edge: float = 0.50,
    n_levels: int = 10,
    save_dir: str | None = None,
) -> None:
    """
    Plot Miller flux surfaces for three geometry configurations side by side:
      1. D0FUS PCHIP — positive triangularity
      2. Academic      — circular (κ const, δ = 0)
      3. D0FUS PCHIP — negative triangularity

    Parameters
    ----------
    R0, a        : float  Major and minor radii [m].
    kappa_edge   : float  Edge elongation [-].
    delta_edge   : float  Edge triangularity (positive value) [-].
    n_levels     : int    Number of flux surface contours (default 10).
    save_dir     : str or None

    References
    ----------
    Miller et al., Phys. Plasmas 5, 973 (1998).
    """
    kappa_95       = f_Kappa_95(kappa_edge)
    delta_95       = f_Delta_95(delta_edge)
    delta_edge_neg = -delta_edge
    delta_95_neg   = f_Delta_95(delta_edge_neg)

    theta_plot = np.linspace(0, 2 * np.pi, 500)
    rho_levels = np.linspace(0.1, 1.0, n_levels)
    colors_fs  = cm.Blues(np.linspace(0.20, 0.95, n_levels))

    configs = [
        ("D0FUS",
         f"D0FUS PCHIP — positive δ\n"
         f"(κ₉₅ = {kappa_95:.3f}, δ_edge = +{delta_edge}, δ₉₅ = +{delta_95:.3f})",
         kappa_edge, delta_edge, kappa_95, delta_95),
        ("Academic",
         f"Academic  (κ = {kappa_edge} const, δ = 0)",
         kappa_edge, 0.0, kappa_95, 0.0),
        ("D0FUS",
         f"D0FUS PCHIP — negative δ\n"
         f"(κ₉₅ = {kappa_95:.3f}, δ_edge = {delta_edge_neg}, δ₉₅ = {delta_95_neg:.3f})",
         kappa_edge, delta_edge_neg, kappa_95, delta_95_neg),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for ax, (mode, title, ke, de, k95, d95) in zip(axes, configs):
        for rho_val, col in zip(rho_levels, colors_fs):
            if mode == "Academic":
                R_surf = R0 + rho_val * a * np.cos(theta_plot)
                Z_surf = ke * rho_val * a * np.sin(theta_plot)
            else:
                R_surf, Z_surf = miller_RZ(rho_val, theta_plot,
                                           R0, a, ke, de, k95, d95)
            lw    = 2.2 if np.isclose(rho_val, 1.0) else 0.8
            ls    = "-"  if np.isclose(rho_val, 1.0) else "--"
            label = (f"ρ = {rho_val:.1f}"
                     + (" (LCFS)" if np.isclose(rho_val, 1.0) else ""))
            ax.plot(R_surf, Z_surf, color=col, lw=lw, ls=ls, label=label)

        ax.plot(R0, 0, "k+", markersize=12, markeredgewidth=2,
                label="Magnetic axis")
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]", fontsize=12)
        ax.set_ylabel("Z [m]", fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Miller flux surfaces — Academic | D0FUS (+δ) | D0FUS (−δ)\n"
        f"(R₀ = {R0} m, a = {a} m, κ = {kappa_edge}, |δ| = {delta_edge})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "miller_surfaces")


def plot_volume_comparison(
    R0: float = 6.2,
    a: float = 2.0,
    delta_fix: float = 0.50,
    kappa_fix: float = 1.85,
    n_theta: int = 5000,
    save_dir: str | None = None,
) -> None:
    """
    Compare analytical and numerical plasma volume formulas.

    Left panel  : V vs κ  (δ fixed).
    Right panel : V vs δ  (κ fixed, including negative triangularity).

    Models compared:
      * Simple   : V = 2π²R₀a²κ  [Wesson]
      * PROCESS  : [Martin]
      * O(δ)     : First-order Miller expansion [Auclair]
      * O(δ²)    : Second-order Miller expansion [Auclair]
      * Numerical: Full LCFS Miller integration

    Parameters
    ----------
    R0         : float  Major radius [m].
    a          : float  Minor radius [m].
    delta_fix  : float  Fixed δ for the κ scan [-].
    kappa_fix  : float  Fixed κ for the δ scan [-].
    n_theta    : int    Poloidal resolution for numerical integration.
    save_dir   : str or None
    """
    theta_v  = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    dtheta_v = theta_v[1] - theta_v[0]

    def _V_simple(k, _d=None):
        return 2 * np.pi**2 * R0 * a**2 * k

    def _V_process(k, d):
        return 2 * np.pi**2 * R0 * a**2 * k * (
            1 - (1 - 8 / (3 * np.pi)) * d * a / R0)

    def _V_ord1(k, d):
        return 2 * np.pi**2 * R0 * a**2 * k * (1 - (a * d) / (4 * R0))

    def _V_ord2(k, d):
        return 2 * np.pi**2 * R0 * a**2 * k * (
            1 - (a * d) / (4 * R0) - d**2 / 8)

    def _V_num(k, d):
        R_t = R0 + a * np.cos(theta_v + d * np.sin(theta_v))
        Z_t = k * a * np.sin(theta_v)
        return np.pi * np.trapezoid(R_t**2 * np.gradient(Z_t, dtheta_v), theta_v)

    kappa_scan = np.linspace(1.1, 2.2, 40)
    delta_scan = np.linspace(-0.7, 0.7, 60)

    V_s_k = [_V_simple(k)              for k in kappa_scan]
    V_p_k = [_V_process(k, delta_fix)  for k in kappa_scan]
    V_1_k = [_V_ord1(k,   delta_fix)   for k in kappa_scan]
    V_2_k = [_V_ord2(k,   delta_fix)   for k in kappa_scan]
    V_n_k = [_V_num(k,    delta_fix)   for k in kappa_scan]

    V_s_d = [_V_simple(kappa_fix)      for _ in delta_scan]
    V_p_d = [_V_process(kappa_fix, d)  for d in delta_scan]
    V_1_d = [_V_ord1(kappa_fix,   d)   for d in delta_scan]
    V_2_d = [_V_ord2(kappa_fix,   d)   for d in delta_scan]
    V_n_d = [_V_num(kappa_fix,    d)   for d in delta_scan]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, xarr, ysets, xlabel, title in [
        (axes[0], kappa_scan,
         (V_s_k, V_p_k, V_1_k, V_2_k, V_n_k),
         r"$\kappa_{\rm edge}$ [-]",
         f"Plasma volume vs κ  (δ = {delta_fix})"),
        (axes[1], delta_scan,
         (V_s_d, V_p_d, V_1_d, V_2_d, V_n_d),
         r"$\delta_{\rm edge}$ [-]",
         f"Plasma volume vs δ  (κ = {kappa_fix})"),
    ]:
        ax.plot(xarr, ysets[0], "k--", lw=2,   label="Simple [Wesson]")
        ax.plot(xarr, ysets[1], "g-.", lw=2,   label="Process [Martin]")
        ax.plot(xarr, ysets[2], "b-.", lw=2,   label="O(δ)  [Auclair]")
        ax.plot(xarr, ysets[3], "m-",  lw=2,   label="O(δ²) [Auclair]")
        ax.plot(xarr, ysets[4], "ro-", ms=4,   label="Numerical [Miller]")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("V  [m³]", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Plasma volume formulas — R₀ = {R0} m, a = {a} m",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "volume_comparison")


def plot_first_wall_surface(
    R0: float = 6.2,
    a: float = 2.0,
    delta_fix: float = 0.50,
    kappa_fix: float = 1.85,
    save_dir: str | None = None,
) -> None:
    """
    Compare first-wall surface area from the Academic (Ramanujan ellipse)
    and D0FUS (Miller LCFS numerical) models.

    Left panel  : S vs κ  (δ fixed).
    Right panel : S vs δ  (κ fixed, including negative triangularity).

    Parameters
    ----------
    R0, a      : float  Geometry [m].
    delta_fix  : float  Fixed δ for the κ scan [-].
    kappa_fix  : float  Fixed κ for the δ scan [-].
    save_dir   : str or None

    References
    ----------
    Ramanujan (1914) — ellipse perimeter approximation.
    Miller et al., Phys. Plasmas 5, 973 (1998).
    """
    kappa_scan = np.linspace(1.1, 2.2, 40)
    delta_scan = np.linspace(-0.7, 0.7, 60)

    S_ac_k = [f_first_wall_surface(R0, a, k, delta_fix, "Academic")
              for k in kappa_scan]
    S_d0_k = [f_first_wall_surface(R0, a, k, delta_fix, "D0FUS")
              for k in kappa_scan]
    S_ac_d = [f_first_wall_surface(R0, a, kappa_fix, d, "Academic")
              for d in delta_scan]
    S_d0_d = [f_first_wall_surface(R0, a, kappa_fix, d, "D0FUS")
              for d in delta_scan]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, xarr, S_ac, S_d0, xlabel, title in [
        (axes[0], kappa_scan, S_ac_k, S_d0_k,
         r"$\kappa_{\rm edge}$ [-]",
         f"First wall surface vs κ  (δ = {delta_fix})"),
        (axes[1], delta_scan, S_ac_d, S_d0_d,
         r"$\delta_{\rm edge}$ [-]",
         f"First wall surface vs δ  (κ = {kappa_fix})"),
    ]:
        ax.plot(xarr, S_ac, "b-",  lw=2, label="Academic (Ramanujan ellipse)")
        ax.plot(xarr, S_d0, "r--", lw=2, label="D0FUS (Miller LCFS)")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("S  [m²]", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"First wall surface area — R₀ = {R0} m, a = {a} m",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "first_wall_surface")


# =============================================================================
# 2. Kinetic profiles  (n, T, p)
# =============================================================================

def plot_nT_profiles(
    rho_n: int = 500,
    save_dir: str | None = None,
) -> None:
    """
    Plot normalised radial density n̂(ρ), temperature T̂(ρ), and pressure
    p̂(ρ) = n̂·T̂ profiles for three plasma confinement modes:
      * L-mode     : parabolic profiles
      * H-mode     : with H-mode pedestal
      * Advanced   : highly peaked profiles with deep pedestal

    Parameters
    ----------
    rho_n    : int  Number of radial grid points.
    save_dir : str or None

    References
    ----------
    ITER Physics Basis, Nucl. Fusion 39, 2175 (1999).
    """
    # Confinement mode definitions — (nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac)
    MODES = {
        "L-mode": {
            "nu_n": 0.5, "nu_T": 1.75, "rho_ped": 1.00,
            "n_ped_frac": 0.00, "T_ped_frac": 0.00,
            "color": "#2166ac", "ls": "--",
        },
        "H-mode": {
            "nu_n": 1.0, "nu_T": 1.45, "rho_ped": 0.94,
            "n_ped_frac": 0.80, "T_ped_frac": 0.40,
            "color": "#d6604d", "ls": "-",
        },
        "Advanced": {
            "nu_n": 1.5, "nu_T": 2.00, "rho_ped": 0.96,
            "n_ped_frac": 0.95, "T_ped_frac": 0.55,
            "color": "#4dac26", "ls": "-.",
        },
    }
    rho = np.linspace(0.0, 1.0, rho_n)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for label, p in MODES.items():
        n_hat = f_nprof(1.0, p["nu_n"], rho, p["rho_ped"], p["n_ped_frac"])
        T_hat = f_Tprof(1.0, p["nu_T"], rho, p["rho_ped"], p["T_ped_frac"])
        p_hat = n_hat * T_hat           # Normalised pressure shape
        axes[0].plot(rho, n_hat, color=p["color"], ls=p["ls"], lw=2, label=label)
        axes[1].plot(rho, T_hat, color=p["color"], ls=p["ls"], lw=2, label=label)
        axes[2].plot(rho, p_hat, color=p["color"], ls=p["ls"], lw=2, label=label)

    for ax, sym, xbar in zip(
        axes,
        [r"$\hat{n} = n(\rho)\,/\,\bar{n}$",
         r"$\hat{T} = T(\rho)\,/\,\bar{T}$",
         r"$\hat{p} = \hat{n}\,\hat{T}$"],
        [r"$\bar{n}$", r"$\bar{T}$", r"$\bar{p}$"],
    ):
        ax.axhline(1.0, color="gray", lw=0.9, ls=":", alpha=0.6,
                   label=f"Volume average = {xbar}")
        ax.set_xlabel(r"$\rho = r/a$", fontsize=12)
        ax.set_ylabel(sym, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Density profile  n̂(ρ)", fontsize=12)
    axes[1].set_title("Temperature profile  T̂(ρ)", fontsize=12)
    axes[2].set_title("Pressure profile  p̂(ρ)", fontsize=12)

    plt.suptitle("Normalised radial profiles — L-mode vs H-mode vs Advanced",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "nTp_profiles")


def plot_density_line_vol(
    nbar_vol: float = 1.01,
    nu_max: float = 1.0,
    save_dir: str | None = None,
) -> None:
    """
    Plot the relative difference between line-averaged and volume-averaged
    electron density as a function of the density peaking exponent ν_n.

    The correction (n̄_line / n̄_vol − 1) is always positive for peaked
    profiles because the chord integral does not down-weight the dense core.

    Parameters
    ----------
    nbar_vol : float  Reference volume-averaged density [10²⁰ m⁻³].
    nu_max   : float  Maximum peaking exponent for the scan [-].
    save_dir : str or None

    References
    ----------
    ITER Physics Design Guidelines (1989) — line-average definition.
    Greenwald, PPCF 44, R27 (2002).
    """
    nu_arr  = np.linspace(0.0, nu_max, 200)
    err_arr = np.array([
        f_nbar_line(nbar_vol, nu) / nbar_vol - 1.0
        for nu in nu_arr
    ]) * 100.0  # [%]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(nu_arr, err_arr, "k-", lw=2)
    ax.fill_between(nu_arr, 0.0, err_arr, alpha=0.15, color="tab:red")
    ax.set_xlabel(r"$\nu_n$", fontsize=12)
    ax.set_ylabel(r"$(n_{\rm line} / n_{\rm vol} - 1)$   [%]", fontsize=12)
    ax.set_title("Line vs volume density: relative difference", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "density_line_vol")


# =============================================================================
# 3. Nuclear / radiation physics
# =============================================================================

def plot_Lz_cooling(
    Te_min_keV: float = 0.05,
    Te_max_keV: float = 100.0,
    n_points: int = 500,
    smoothing_width: int = 15,
    save_dir: str | None = None,
) -> None:
    """
    Plot the coronal radiative cooling coefficient L_z(T_e) for the main
    impurity species of interest in tokamak plasmas.

    The three plasma zones (edge, pedestal, core) are highlighted with a
    colour band in the background.

    Parameters
    ----------
    Te_min_keV, Te_max_keV : float  Electron temperature scan range [keV].
    n_points               : int    Number of temperature grid points.
    smoothing_width        : int    Convolution window for log-space smoothing.
    save_dir               : str or None

    References
    ----------
    Mavrin (2018); ADAS database; coronal equilibrium assumption.
    """
    Te_arr  = np.logspace(np.log10(Te_min_keV), np.log10(Te_max_keV), n_points)
    species = [
        ("W",  "W  (Z=74)",  "tab:red"),
        ("Kr", "Kr (Z=36)",  "tab:brown"),
        ("Ar", "Ar (Z=18)",  "tab:purple"),
        ("Ne", "Ne (Z=10)",  "tab:cyan"),
        ("N",  "N  (Z=7)",   "tab:green"),
        ("C",  "C  (Z=6)",   "tab:olive"),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))

    for imp, label, color in species:
        Lz     = np.array([get_Lz(imp, T) for T in Te_arr])
        Lz_sm  = np.exp(np.convolve(np.log(Lz),
                                     np.ones(smoothing_width) / smoothing_width,
                                     mode="same"))
        ax.plot(Te_arr, Lz_sm, color=color, lw=2.0, label=label)

    # Plasma zone shading
    ax.axvspan(0.1, 0.3,   alpha=0.07, color="steelblue")
    ax.axvspan(0.3, 1.0,   alpha=0.07, color="goldenrod")
    ax.axvspan(1.0, 100.0, alpha=0.07, color="tomato")
    
    ax.text(0.17, 2e-31, "Edge",     fontsize=10, color="steelblue", ha="center")
    ax.text(0.55, 2e-31, "Pedestal", fontsize=10, color="goldenrod", ha="center")
    ax.text(10,   2e-31, "Core",     fontsize=10, color="tomato", ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 50)
    ax.set_ylim(1e-34, 10e-31)
    ax.set_xlabel(r"$T_e$  [keV]", fontsize=12)
    ax.set_ylabel(r"$L_z(T_e)$  [W·m³]", fontsize=12)
    ax.set_title("Coronal radiative cooling coefficient\n"
                 r"(Mavrin 2018 / ADAS)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "Lz_cooling")


def plot_He_fraction(
    nbar: float = 1.0,
    Tbar: float = 8.9,
    tauE: float = 3.7,
    C_Alpha: float = 5.0,
    nu_T: float = 1.0,
    save_dir: str | None = None,
) -> None:
    """
    Plot helium ash fraction f_α as a function of the He removal efficiency
    C_α = τ_α / τ_E for ITER and EU-DEMO reference parameters, comparing
    academic (no pedestal) and H-mode pedestal profile assumptions.

    Parameters
    ----------
    nbar, Tbar, tauE : float  Reference plasma parameters [10²⁰ m⁻³, keV, s].
    C_Alpha          : float  Reference C_α value (drawn as vertical line) [-].
    nu_T             : float  Temperature peaking exponent [-].
    save_dir         : str or None

    References
    ----------
    ITER Physics Basis, Nucl. Fusion 39, §2.4 (1999).
    """
    C_arr = np.linspace(2, 15, 150)

    fa_ITER     = [f_He_fraction(1.0,  8.9,  3.7, C, nu_T) * 100 for C in C_arr]
    fa_ITER_ped = [f_He_fraction(1.0,  8.9,  3.7, C, nu_T,
                                 rho_ped=0.9, T_ped_frac=0.25) * 100
                   for C in C_arr]
    fa_DEMO     = [f_He_fraction(1.2, 12.5,  4.6, C, nu_T) * 100 for C in C_arr]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(C_arr, fa_ITER,     "b-",  lw=1.8, label="ITER — academic (no pedestal)")
    ax.plot(C_arr, fa_ITER_ped, "b--", lw=1.4, label="ITER — D0FUS H-mode pedestal")
    ax.plot(C_arr, fa_DEMO,     "r-",  lw=1.8, label="EU-DEMO — academic")
    ax.axvline(C_Alpha, color="k", lw=0.9, ls=":", label=f"$C_\\alpha$ = {C_Alpha:.0f}")
    ax.axhspan(5, 10, color="grey", alpha=0.12, label="ITER target 5–10 %")
    ax.set_xlabel(r"Removal efficiency $C_\alpha = \tau_\alpha / \tau_E$", fontsize=11)
    ax.set_ylabel(r"Helium ash fraction $f_\alpha$ [%]", fontsize=11)
    ax.set_title("He ash fraction — academic vs D0FUS H-mode pedestal", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(2, 15)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "He_fraction")


# =============================================================================
# 4. Superconductor / cable engineering
# =============================================================================

def plot_Jc_scaling(
    B_min: float = 0.5,
    B_max: float = 25.0,
    T_op: float = 4.2,
    f_non_Cu_LTS: float = 0.50,
    f_non_Cu_HTS: float = 0.60,
    save_dir: str | None = None,
) -> None:
    """
    Plot strand/tape engineering current density J_eng vs magnetic field for
    the three main superconductor families used in fusion magnets:
      * NbTi   (LTS, ITER TF reference)
      * Nb₃Sn  (LTS, ITER/EU-DEMO TF/CS)
      * REBCO  (HTS, ARC / SPARC)

    Parameters
    ----------
    B_min, B_max : float  Field scan range [T].
    T_op         : float  Operating temperature [K].
    f_non_Cu_LTS : float  Non-copper fraction for LTS strands [-].
    f_non_Cu_HTS : float  Non-copper fraction for HTS tapes [-].
    save_dir     : str or None

    References
    ----------
    ITER TF strand specifications; Nijhuis (2008); Fleiter & Ballarino (2014).
    """
    B_vals  = np.linspace(B_min, B_max, 300)
    J_NbTi  = J_non_Cu_NbTi(B_vals, T_op)  * f_non_Cu_LTS / 1e6   # [A/mm²]
    J_Nb3Sn = J_non_Cu_Nb3Sn(B_vals, T_op, Eps=-0.003) * f_non_Cu_LTS / 1e6
    J_REBCO = J_non_Cu_REBCO(B_vals, T_op, Tet=0) * f_non_Cu_HTS / 1e6

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(B_vals, J_NbTi,  lw=2, color="#A06AB4", label="NbTi strand")
    ax.plot(B_vals, J_Nb3Sn, lw=2, color="#E06C75", label="Nb₃Sn strand")
    ax.plot(B_vals, J_REBCO, lw=2, color="#D4B000", label="REBCO tape")
    ax.set_xlabel("Magnetic field B [T]", fontsize=12)
    ax.set_ylabel("Engineering current density J [A/mm²]", fontsize=12)
    ax.set_title(f"Superconductor Jc scalings @ {T_op} K", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, B_max)
    ax.set_ylim(10, 1e4)
    ax.set_yscale("log")

    plt.tight_layout()
    _save_or_show(fig, save_dir, "Jc_scaling")


def plot_cable_current_density(
    B_min: float = 4.0,
    B_max: float = 25.0,
    T_op: float = 4.2,
    E_mag: float = 6e9,
    I_cond: float = 45e3,
    V_max: float = 10e3,
    N_sub: float = 6,
    tau_h: float = 0.5,
    f_He_pipe: float = None,
    f_void: float = None,
    f_In: float = None,
    T_hotspot: float = None,
    RRR: float = None,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    Plot worst-case cable current density J_wost vs peak field for NbTi,
    Nb₃Sn, and REBCO conductors under quench protection constraints.

    The operating point is defined by a single-coil energy (E_mag), conductor
    current (I_cond), maximum voltage (V_max), number of subdivisions (N_sub),
    and detection hold time (tau_h).

    Temperature margins (Marge_T_*), strain (Eps), and tape angle (Tet) are
    read from ``cfg`` when provided; hard-coded defaults are used otherwise so
    the function remains callable without a config object.

    Parameters
    ----------
    B_min, B_max  : float  Field scan range [T].
    T_op          : float  Operating temperature [K].
    E_mag         : float  Coil magnetic energy [J].
    I_cond        : float  Conductor current [A].
    V_max         : float  Maximum dump voltage [V].
    N_sub         : float  Number of quench protection subdivisions [-].
    tau_h         : float  Detection + hold time [s].
    f_He_pipe     : float or None  He pipe fraction [-] (from cfg if None).
    f_void        : float or None  Interstitial void fraction [-] (from cfg if None).
    f_In          : float or None  Insulation fraction [-] (from cfg if None).
    T_hotspot     : float or None  Max hotspot temperature [K] (from cfg if None).
    RRR           : float or None  Copper residual resistivity ratio (from cfg if None).
    cfg           : config object  D0FUS global configuration (DEFAULT_CONFIG if None).
    save_dir      : str or None
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # Resolve parameters from cfg with explicit fallbacks
    f_He_pipe     = f_He_pipe     if f_He_pipe is not None else cfg.f_He_pipe
    f_void        = f_void        if f_void    is not None else cfg.f_void
    f_In          = f_In          if f_In      is not None else cfg.f_In
    T_hotspot     = T_hotspot     if T_hotspot is not None else cfg.T_hotspot
    RRR           = RRR           if RRR       is not None else cfg.RRR
    Marge_T_He    = cfg.Marge_T_He
    Marge_T_Nb3Sn = cfg.Marge_T_Nb3Sn
    Marge_T_NbTi  = cfg.Marge_T_NbTi
    Marge_T_REBCO = cfg.Marge_T_REBCO
    Eps           = cfg.Eps
    Tet           = cfg.Tet

    B_range = np.linspace(B_min, B_max, 30)
    coil_params = dict(
        E_mag=E_mag, I_cond=I_cond, V_max=V_max,
        N_sub=N_sub, tau_h=tau_h,
        f_He_pipe=f_He_pipe, f_void=f_void, f_In=f_In,
        T_hotspot=T_hotspot, RRR=RRR,
        Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps, Tet=Tet,
    )

    J_NbTi   = []
    J_Nb3Sn  = []
    J_REBCO  = []

    for B in B_range:
        for sc, store in [("NbTi",  J_NbTi),
                          ("Nb3Sn", J_Nb3Sn),
                          ("REBCO", J_REBCO)]:
            res = calculate_cable_current_density(
                sc_type=sc, B_peak=B, T_op=T_op, **coil_params)
            val = res["J_wost"]
            store.append(val / 1e6 if val > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(B_range, J_NbTi,  lw=2, color="#A06AB4", label="NbTi")
    ax.plot(B_range, J_Nb3Sn, lw=2, color="#E06C75", label="Nb₃Sn")
    ax.plot(B_range, J_REBCO, lw=2, color="#D4B000", label="REBCO")
    ax.set_xlabel(r"Peak field $B_{\rm peak}$ [T]", fontsize=12)
    ax.set_ylabel(r"Cable current density $J_{\rm wost}$ [MA/m²]", fontsize=12)
    ax.set_title("Cable current density vs field\n"
                 f"(E = {E_mag/1e9:.1f} GJ, I = {I_cond/1e3:.0f} kA, "
                 f"V = {V_max/1e3:.0f} kV, N = {int(N_sub)})", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(B_min, B_max)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "cable_current_density")


# =============================================================================
# 5. Structural mechanics
# =============================================================================

def plot_TF_thickness_vs_field(
    a: float = 3.0,
    b: float = 1.7,
    R0: float = 9.0,
    sigma_TF: float = 860e6,
    J_max_TF: float = 50e6,
    B_max: float = 25.0,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    Plot TF coil winding-pack thickness vs peak magnetic field for four
    mechanical models (Academic/D0FUS × Wedging/Bucking).

    An optional MADE benchmark scatter dataset is superimposed on the Wedging
    panel for EU-DEMO validation.

    Parameters
    ----------
    a, b, R0   : float  Minor radius, blanket+shield thickness, major radius [m].
    sigma_TF   : float  Allowable TF coil stress [Pa].
    J_max_TF   : float  Maximum coil current density [A/m²].
    B_max      : float  Maximum field at conductor [T].
    cfg        : config object (DEFAULT_CONFIG if None)
    save_dir   : str or None

    References
    ----------
    Giannini et al. (2023) — MADE EU-DEMO Bmax vs R0 scan.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    B_vals = np.linspace(0, B_max, 50)
    acad_w, acad_b, d0_w, d0_b = [], [], [], []

    for B in B_vals:
        acad_w.append(f_TF_academic(a, b, R0, sigma_TF, J_max_TF, B,
                                    "Wedging", cfg.coef_inboard_tension,
                                    cfg.F_CClamp)[0])
        acad_b.append(f_TF_academic(a, b, R0, sigma_TF, J_max_TF, B,
                                    "Bucking", cfg.coef_inboard_tension,
                                    cfg.F_CClamp)[0])
        d0_w.append(f_TF_D0FUS(a, b, R0, sigma_TF, J_max_TF, B,
                                "Wedging", 0.5, 1,
                                cfg.c_BP, cfg.coef_inboard_tension,
                                cfg.F_CClamp)[0])
        d0_b.append(f_TF_D0FUS(a, b, R0, sigma_TF, J_max_TF, B,
                                "Bucking", 1.0, 1,
                                cfg.c_BP, cfg.coef_inboard_tension,
                                cfg.F_CClamp)[0])

    # MADE EU-DEMO benchmark data (Giannini et al. 2023, Fig. 4i)
    x_made = np.array([11.25, 13.25, 14.75, 16, 17, 18, 19,
                        19.75, 20.5, 21.25, 22, 22.5])
    y_made = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
                        2.0, 2.2, 2.4, 2.65, 2.85])

    colors = ["#1f77b4", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Wedging panel
    ax = axes[0]
    ax.plot(B_vals, acad_w, color=colors[0], lw=2, label="Academic — Wedging")
    ax.plot(B_vals, d0_w,   color=colors[1], lw=2, label="D0FUS — Wedging")
    ax.scatter(x_made, y_made, color="k", marker="x", s=80, label="MADE")
    ax.set_xlabel("Peak field B_max [T]", fontsize=12)
    ax.set_ylabel("TF winding-pack thickness [m]", fontsize=12)
    ax.set_title("Wedging configuration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.6)

    # Bucking panel
    ax = axes[1]
    ax.plot(B_vals, acad_b, color=colors[0], lw=2, label="Academic — Bucking")
    ax.plot(B_vals, d0_b,   color=colors[1], lw=2, label="D0FUS — Bucking")
    ax.set_xlabel("Peak field B_max [T]", fontsize=12)
    ax.set_ylabel("TF winding-pack thickness [m]", fontsize=12)
    ax.set_title("Bucking configuration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.6)

    plt.suptitle(
        f"TF coil thickness vs peak field\n"
        f"(R₀ = {R0} m, a = {a} m, σ_max = {sigma_TF/1e6:.0f} MPa)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "TF_thickness_vs_field")


def plot_CS_thickness_vs_flux(
    a_cs: float = 3.0,
    b_cs: float = 1.2,
    c_cs: float = 2.0,
    R0_cs: float = 9.0,
    B_TF: float = 13.0,
    B_max_CS: float = 50.0,
    sigma_CS: float = 300e6,
    J_wost_CS: float = 30e6,
    T_He: float = 4.75,
    kappa_cs: float = 1.7,
    N_sub: int = 6,
    tau_h: float = 5.0,
    psi_max: float = 500.0,
    n_psi: int = 30,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    Plot CS coil winding-pack thickness and peak field vs volt-second budget
    for three mechanical models (Academic, D0FUS, CIRCE) and three
    configuration types (Wedging, Bucking, Plug).

    For the Wedging configuration, a MADE benchmark scatter dataset
    (Sarasola et al. 2023, EU-DEMO parametric study) is overlaid on the
    thickness panel.

    Default parameters correspond to the EU-DEMO simplified scan geometry
    (Sarasola et al. 2023):
      a_cs=3 m, b_cs=1.2 m, c_cs=2 m, R0=9 m, J=30 MA/m², σ=300 MPa.

    Parameters
    ----------
    a_cs, b_cs, c_cs : float  CS bore radius, TF inboard edge, plasma minor radius [m].
    R0_cs            : float  Major radius [m].
    B_TF, B_max_CS   : float  TF peak field and CS maximum allowable field [T].
    sigma_CS         : float  Allowable CS hoop stress [Pa].
    J_wost_CS        : float  CS conductor worst-case current density [A/m²].
    T_He             : float  Helium operating temperature [K].
    kappa_cs         : float  Plasma elongation [-].
    N_sub, tau_h     : int, float  Quench protection subdivisions and hold time.
    psi_max, n_psi   : float, int  Volt-second scan ceiling [Wb] and resolution.
    cfg              : config object (DEFAULT_CONFIG if None).
    save_dir         : str or None

    References
    ----------
    Sarasola et al., IEEE Trans. Appl. Supercond. 33, 1-5 (2023) — EU-DEMO CS
      parametric study; MADE code reference data.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # MADE reference data — Sarasola et al. (2023), extracted from Fig. 4
    # (CS winding-pack width vs total flux swing, Wedging configuration)
    made_thickness = np.array([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    made_flux      = np.array([209, 209, 208, 205, 200, 192, 185, 177, 161]) * 2

    psi_values  = np.linspace(0, psi_max, n_psi)
    mech_confs  = ["Wedging", "Bucking", "Plug"]
    model_funcs = {"Academic": f_CS_ACAD, "D0FUS": f_CS_D0FUS, "CIRCE": f_CS_CIRCE}
    colors      = {"Academic": "blue", "D0FUS": "green", "CIRCE": "red"}

    # Storage: results[model][config] = {"thickness": [], "B": []}
    results = {m: {c: {"thickness": [], "B": []} for c in mech_confs}
               for m in model_funcs}

    for psi in psi_values:
        for conf in mech_confs:
            for model, func in model_funcs.items():
                res = func(0, 0, psi, 0,
                           a_cs, b_cs, c_cs, R0_cs,
                           B_TF, B_max_CS, sigma_CS,
                           "Manual", J_wost_CS, T_He,
                           conf, kappa_cs, N_sub, tau_h, cfg)
                th = float(np.real(res[0])) if np.isfinite(np.real(res[0])) else np.nan
                Bc = float(np.real(res[5])) if np.isfinite(np.real(res[5])) else np.nan
                results[model][conf]["thickness"].append(th)
                results[model][conf]["B"].append(Bc)

    # One figure per configuration, two panels each (thickness + field)
    for conf in mech_confs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for quantity, ax, ylabel in [
            ("thickness", axes[0], "CS winding-pack thickness [m]"),
            ("B",         axes[1], "Peak field B_CS [T]"),
        ]:
            for model in model_funcs:
                ax.plot(psi_values, results[model][conf][quantity],
                        color=colors[model], lw=2, label=model)

            # MADE cross-check: Wedging thickness panel only
            if conf == "Wedging" and quantity == "thickness":
                ax.scatter(made_flux, made_thickness,
                           color="black", marker="x", s=60, zorder=5,
                           label="MADE (Sarasola 2023)")

            ax.set_xlabel(r"$\Psi_{\rm plateau}$ [Wb]", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{conf} — {ylabel.split('[')[0].strip()}", fontsize=11)
            ax.grid(True, ls="--", alpha=0.6)
            ax.legend(fontsize=10)

        plt.suptitle(
            f"CS coil sizing — {conf} configuration\n"
            f"(R₀ = {R0_cs} m, σ_max = {sigma_CS/1e6:.0f} MPa, "
            f"J = {J_wost_CS/1e6:.0f} MA/m²)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        _save_or_show(fig, save_dir, f"CS_thickness_{conf.lower()}")


# =============================================================================
# 6. Resistivity models
# =============================================================================

def plot_resistivity_models(
    ne: float = 1.0e20,
    Z_eff: float = 1.7,
    R0: float = 6.2,
    a: float = 2.0,
    q: float = 2.0,
    T_min: float = 1.0,
    T_max: float = 25.0,
    save_dir: str | None = None,
) -> None:
    """
    Plot plasma resistivity η vs electron temperature for four models:
      0. Old Wesson approximation
      1. Spitzer (classical)
      2. Sauter (neoclassical, 1999)
      3. Redl  (neoclassical, 2021)

    Parameters
    ----------
    ne     : float  Electron density [m⁻³].
    Z_eff  : float  Effective ion charge [-].
    R0, a  : float  Major and minor radii [m].
    q      : float  Safety factor (used for neoclassical corrections) [-].
    T_min, T_max : float  Temperature scan range [keV].
    save_dir     : str or None

    References
    ----------
    Spitzer & Härm, Phys. Rev. 89, 977 (1953).
    Sauter et al., Phys. Plasmas 6, 2834 (1999).
    Redl et al., Phys. Plasmas 28, 022502 (2021).
    """
    epsilon = a / R0
    T_arr   = np.linspace(T_min, T_max, 100)

    eta0 = np.array([eta_old(T, ne, Z_eff)                        for T in T_arr])
    eta1 = np.array([eta_spitzer(T, ne, Z_eff)                    for T in T_arr])
    eta2 = np.array([eta_sauter(T, ne, Z_eff, epsilon, q, R0)     for T in T_arr])
    eta3 = np.array([eta_redl(T,   ne, Z_eff, epsilon, q, R0)     for T in T_arr])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.semilogy(T_arr, eta0, "g--", lw=2, label="0. Old (Wesson)")
    ax.semilogy(T_arr, eta1, "k",   lw=2, label="1. Spitzer")
    ax.semilogy(T_arr, eta2, "b",   lw=2, label="2. Sauter (1999)")
    ax.semilogy(T_arr, eta3, "r",   lw=2, label="3. Redl (2021)")
    ax.set_xlabel("Temperature [keV]", fontsize=12)
    ax.set_ylabel(r"Resistivity $\eta$ [$\Omega \cdot$m]", fontsize=12)
    ax.set_title("Neoclassical resistivity vs temperature\n"
                 f"(nₑ = {ne:.1e} m⁻³, Z_eff = {Z_eff}, ε = {epsilon:.2f})",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(T_min, T_max)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "resistivity_models")


# =============================================================================
# 7. Run-output profiles  (Bloc A — requires a D0FUS run dict)
# =============================================================================
#
# Run-dict expected keys
# -----------------------
# Geometry
#   R0, a            : float   Major / minor radii [m]
#   kappa_edge       : float   Edge elongation κ_LCFS [-]
#   delta_edge       : float   Edge triangularity δ_LCFS [-]
#   kappa_95         : float   κ at the 95 % flux surface (derived if absent)
#   delta_95         : float   δ at the 95 % flux surface (derived if absent)
#   Vprime_data      : tuple or None   Precomputed (ρ, V', V) from precompute_Vprime
# Plasma state
#   B0, Ip           : float   On-axis field [T], plasma current [MA]
#   nbar, Tbar       : float   Volume-averaged density [10²⁰ m⁻³] and temperature [keV]
#   nu_n, nu_T       : float   Density / temperature peaking exponents [-]
#   rho_ped          : float   Normalised pedestal radius (1.0 = no pedestal)
#   n_ped_frac       : float   n_ped / nbar [-]
#   T_ped_frac       : float   T_ped / Tbar [-]
#   q95              : float   Safety factor at ρ₉₅ [-]
#   Z_eff            : float   Effective ion charge [-]
# Impurities (optional — absent or None → species not plotted)
#   f_W, f_Ne, f_Ar  : float   Species concentration n_imp / n_e [-]
# =============================================================================


# --- Internal helpers for run-profile functions ----------------------------

def _resolve_geometry(run: dict) -> tuple:
    """
    Extract and complete geometry parameters from run dict.

    Derives kappa_95 and delta_95 via ITER 1989 scaling if not provided.

    Returns
    -------
    R0, a, kappa_edge, delta_edge, kappa_95, delta_95 : float
    Vprime_data : tuple or None
    """
    R0          = float(run["R0"])
    a           = float(run["a"])
    kappa_edge  = float(run["kappa_edge"])
    delta_edge  = float(run["delta_edge"])
    kappa_95    = float(run.get("kappa_95", f_Kappa_95(kappa_edge)))
    delta_95    = float(run.get("delta_95", f_Delta_95(delta_edge)))
    Vprime_data = run.get("Vprime_data", None)
    return R0, a, kappa_edge, delta_edge, kappa_95, delta_95, Vprime_data


def _resolve_kinetics(run: dict) -> tuple:
    """
    Extract kinetic parameters from run dict with safe defaults.

    Returns
    -------
    nbar, Tbar, nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac : float
    """
    return (
        float(run["nbar"]),
        float(run["Tbar"]),
        float(run.get("nu_n",       1.0)),
        float(run.get("nu_T",       1.0)),
        float(run.get("rho_ped",    1.0)),
        float(run.get("n_ped_frac", 0.0)),
        float(run.get("T_ped_frac", 0.0)),
    )


def _radiation_emissivity(
    rho: np.ndarray,
    nbar: float,
    Tbar: float,
    nu_n: float,
    nu_T: float,
    rho_ped: float,
    n_ped_frac: float,
    T_ped_frac: float,
    impurities: dict,
) -> dict:
    """
    Compute local volumetric radiation emissivity profiles.

    Channels returned:
      ``P_brem``  : Bremsstrahlung shape ∝ n² √T  (peak-normalised)
      ``P_sync``  : Synchrotron shape ∝ n T^(3/2)  (peak-normalised)
      per impurity key : Line radiation emissivity [W/m³] (absolute)

    Parameters
    ----------
    rho              : ndarray  Normalised radial grid.
    nbar, Tbar       : float    Volume-averaged density [10²⁰ m⁻³] and temperature [keV].
    nu_n, nu_T       : float    Peaking exponents.
    rho_ped          : float    Pedestal radius.
    n_ped_frac       : float    n_ped / nbar.
    T_ped_frac       : float    T_ped / Tbar.
    impurities       : dict     {species: f_imp}, e.g. {'W': 5e-5, 'Ne': 3e-3}.

    Returns
    -------
    dict with keys 'rho', 'P_brem', 'P_sync', and one key per impurity species.
    """
    n_hat = f_nprof(1.0,  nu_n, rho, rho_ped, n_ped_frac)  # normalised density shape
    T_arr = f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac)  # [keV]

    n_e        = nbar * 1e20 * n_hat               # [m⁻³]
    T_eV       = np.clip(T_arr * 1e3, 10.0, None)  # [eV], floored for Lz interpolation

    # Bremsstrahlung emissivity shape [a.u.]
    brem_raw  = n_e**2 * np.sqrt(T_eV)
    brem_peak = brem_raw.max()
    P_brem    = brem_raw / brem_peak if brem_peak > 0 else brem_raw

    # Synchrotron emissivity shape [a.u.] — Albajar (2002) power law
    sync_raw  = n_e * T_eV**1.5
    sync_peak = sync_raw.max()
    P_sync    = sync_raw / sync_peak if sync_peak > 0 else sync_raw

    # Line radiation [W/m³] — coronal equilibrium (Mavrin 2018 / ADAS)
    line_out = {}
    for species, f_imp in impurities.items():
        if f_imp is None or f_imp <= 0.0:
            continue
        T_keV_clamped = np.clip(T_arr, 0.01, None)
        Lz_arr = np.array([get_Lz(species, T_keV_clamped[k])
                           for k in range(len(rho))])        # [W·m³]
        line_out[species] = f_imp * n_e**2 * Lz_arr         # [W/m³]

    return {"rho": rho, "P_brem": P_brem, "P_sync": P_sync, **line_out}


# ---------------------------------------------------------------------------
# A1 — Kinetic profiles: n(ρ), T(ρ), p(ρ)
# ---------------------------------------------------------------------------

def plot_run_nTp(
    run: dict,
    n_rho: int = 400,
    save_dir: str | None = None,
) -> None:
    """
    Plot absolute radial profiles n_e(ρ), T_e(ρ), p(ρ) from a D0FUS run.

    Horizontal three-panel figure:
      Left   : Electron density n_e(ρ)      [10²⁰ m⁻³]
      Centre : Electron temperature T_e(ρ)  [keV]
      Right  : Total pressure p(ρ)          [kPa]

    Volume-averaged values and on-axis peaks are annotated on each panel.

    Parameters
    ----------
    run      : dict   D0FUS run output (see section header for key list).
    n_rho    : int    Number of radial grid points.
    save_dir : str or None
    """
    nbar, Tbar, nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _resolve_kinetics(run)
    Z_eff = float(run.get("Z_eff", 1.7))
    rho   = np.linspace(0.0, 1.0, n_rho)

    n_prof = f_nprof(nbar, nu_n, rho, rho_ped, n_ped_frac)   # [10²⁰ m⁻³]
    T_prof = f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac)   # [keV]
    # Total pressure [kPa]: p = n_e (1 + 1/Z_eff) T_e
    p_prof = (n_prof * 1e20) * (1.0 + 1.0 / Z_eff) * (T_prof * 1e3) * E_ELEM / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _ped_vline(ax):
        """Draw pedestal location marker when a pedestal is present."""
        if rho_ped < 0.99:
            ax.axvline(rho_ped, color="gray", lw=0.9, ls=":", alpha=0.6,
                       label=f"ρ_ped = {rho_ped}")

    # --- n_e(ρ) ---
    ax = axes[0]
    ax.plot(rho, n_prof, "tab:blue", lw=2)
    ax.axhline(nbar, color="tab:blue", lw=1, ls="--", alpha=0.7,
               label=rf"$\bar{{n}}$ = {nbar:.2f}×10²⁰ m⁻³")
    _ped_vline(ax)
    ax.set_ylabel(r"$n_e(\rho)$  [10²⁰ m⁻³]", fontsize=12)
    ax.set_title("Electron density  n_e(ρ)", fontsize=11)
    ax.annotate(rf"$n_e(0) = {n_prof[0]:.2f}$", xy=(0.02, n_prof[0]),
                fontsize=10, color="tab:blue")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- T_e(ρ) ---
    ax = axes[1]
    ax.plot(rho, T_prof, "tab:red", lw=2)
    ax.axhline(Tbar, color="tab:red", lw=1, ls="--", alpha=0.7,
               label=rf"$\bar{{T}}$ = {Tbar:.1f} keV")
    _ped_vline(ax)
    ax.set_ylabel(r"$T_e(\rho)$  [keV]", fontsize=12)
    ax.set_title("Electron temperature  T_e(ρ)", fontsize=11)
    ax.annotate(rf"$T_e(0) = {T_prof[0]:.1f}$ keV", xy=(0.02, T_prof[0]),
                fontsize=10, color="tab:red")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- p(ρ) ---
    ax = axes[2]
    ax.plot(rho, p_prof, "tab:green", lw=2)
    _ped_vline(ax)
    ax.set_ylabel(r"$p(\rho)$  [kPa]", fontsize=12)
    ax.set_title("Total pressure  p(ρ)", fontsize=11)
    ax.annotate(rf"$p(0) = {p_prof[0]:.1f}$ kPa",
                xy=(0.02, p_prof[0] * 0.96), fontsize=10, color="tab:green")
    if rho_ped < 0.99:
        ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel(r"$\rho = r/a$", fontsize=12)
        ax.set_xlim(0, 1)

    plt.suptitle(
        rf"Kinetic profiles — R₀={run.get('R0','?')} m, a={run.get('a','?')} m"
        f"  ·  ν_n={nu_n}, ν_T={nu_T}, ρ_ped={rho_ped}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_profiles_nTp")


# ---------------------------------------------------------------------------
# A2 — Shaping profiles κ(ρ), δ(ρ)
# ---------------------------------------------------------------------------

def plot_shaping_run(
    run: dict,
    n_rho: int = 400,
    save_dir: str | None = None,
) -> None:
    """
    Plot radial elongation κ(ρ) and triangularity δ(ρ) profiles for the
    geometry used in a D0FUS run.

    In 'D0FUS' geometry mode, the Christiansen PCHIP parameterisation is
    used (radially varying κ and δ with core penetration).
    In 'Academic' mode, κ(ρ) = κ_edge = const and δ(ρ) = 0 everywhere,
    consistent with the cylindrical-torus approximation.

    Parameters
    ----------
    run      : dict   D0FUS run output.
    n_rho    : int    Number of radial grid points.
    save_dir : str or None

    References
    ----------
    Christiansen et al., Nucl. Fusion 32, 291 (1992).
    Ball & Parra, PPCF 57, 045006 (2015) — κ core penetration.
    """
    R0, a, kappa_edge, delta_edge, kappa_95, delta_95, _ = _resolve_geometry(run)
    geom_mode = run.get("Plasma_geometry", "D0FUS")
    rho = np.linspace(1e-4, 1.0, n_rho)

    if geom_mode == "Academic":
        # Academic: constant κ = κ_edge, δ = 0 everywhere
        kap_arr = np.full_like(rho, kappa_edge)
        del_arr = np.zeros_like(rho)
    else:
        # D0FUS PCHIP profiles with radial variation
        kap_arr = kappa_profile(rho, kappa_edge, kappa_95)
        del_arr = delta_profile(rho, delta_edge, delta_95)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── κ(ρ) panel ────────────────────────────────────────────────────
    ax = axes[0]
    if geom_mode == "Academic":
        ax.axhline(kappa_edge, color="tab:blue", lw=2.2,
                   label=f"Academic: κ = κ_edge = {kappa_edge:.3f} (const)")
    else:
        ax.plot(rho, kap_arr, "tab:blue", lw=2.2, label="D0FUS PCHIP")
        ax.axhline(kappa_edge, color="tab:orange", lw=1.4, ls="--",
                   label=f"κ_edge = {kappa_edge:.3f}")
        ax.axhline(kappa_95,   color="tab:gray",   lw=1.0, ls=":",
                   label=f"κ₉₅   = {kappa_95:.3f}")
        ax.axvline(0.95, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel(r"$\rho = r/a$", fontsize=12)
    ax.set_ylabel(r"$\kappa(\rho)$", fontsize=12)
    ax.set_title("Elongation profile κ(ρ)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # ── δ(ρ) panel ────────────────────────────────────────────────────
    ax = axes[1]
    if geom_mode == "Academic":
        ax.axhline(0, color="tab:blue", lw=2.2,
                   label="Academic: δ = 0 (const)")
    else:
        ax.plot(rho, del_arr, "tab:red", lw=2.2, label="D0FUS PCHIP")
        ax.axhline(delta_edge, color="tab:orange", lw=1.4, ls="--",
                   label=f"δ_edge = {delta_edge:.3f}")
        ax.axhline(delta_95,   color="tab:gray",   lw=1.0, ls=":",
                   label=f"δ₉₅   = {delta_95:.3f}")
        ax.axvline(0.95, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.6, alpha=0.35)
    ax.set_xlabel(r"$\rho = r/a$", fontsize=12)
    ax.set_ylabel(r"$\delta(\rho)$", fontsize=12)
    ax.set_title("Triangularity profile δ(ρ)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    _mode_str = "Academic" if geom_mode == "Academic" else "D0FUS PCHIP"
    plt.suptitle(
        f"Run shaping profiles ({_mode_str}) — R₀={R0} m, a={a} m\n"
        f"κ_edge={kappa_edge:.3f}, δ_edge={delta_edge:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_shaping_profiles")


# ---------------------------------------------------------------------------
# A3 — Miller flux surfaces from run geometry
# ---------------------------------------------------------------------------

def plot_flux_surfaces_run(
    run: dict,
    n_levels: int = 12,
    n_theta: int = 500,
    save_dir: str | None = None,
) -> None:
    """
    Plot nested flux surfaces for the actual geometry of a D0FUS run.

    When Plasma_geometry == 'D0FUS', surfaces follow the Miller PCHIP
    parameterisation with radially varying κ(ρ) and δ(ρ).
    When Plasma_geometry == 'Academic' (or unset), surfaces are concentric
    ellipses with constant κ = κ_edge and δ = 0, consistent with the
    cylindrical-torus approximation used in the Academic solver branch.

    Single-panel poloidal cross-section view.  Surfaces are coloured from
    dark blue (core, ρ ≈ 0.1) to light blue (LCFS, ρ = 1.0).

    Parameters
    ----------
    run      : dict   D0FUS run output.
    n_levels : int    Number of flux surface contours.
    n_theta  : int    Poloidal resolution per surface.
    save_dir : str or None

    References
    ----------
    Miller et al., Phys. Plasmas 5, 973 (1998).
    """
    R0, a, kappa_edge, delta_edge, kappa_95, delta_95, _ = _resolve_geometry(run)
    geom_mode = run.get("Plasma_geometry", "D0FUS")

    theta      = np.linspace(0, 2 * np.pi, n_theta)
    rho_levels = np.linspace(0.1, 1.0, n_levels)
    colors_fs  = cm.Blues(np.linspace(0.25, 0.92, n_levels))

    fig, ax = plt.subplots(figsize=(5.5, 8))

    for rho_val, col in zip(rho_levels, colors_fs):
        if geom_mode == "Academic":
            # Concentric ellipses: κ = κ_edge (const), δ = 0
            R_surf = R0 + rho_val * a * np.cos(theta)
            Z_surf = kappa_edge * rho_val * a * np.sin(theta)
        else:
            # D0FUS PCHIP Miller surfaces with radially varying κ(ρ), δ(ρ)
            R_surf, Z_surf = miller_RZ(rho_val, theta,
                                       R0, a, kappa_edge, delta_edge,
                                       kappa_95, delta_95)

        is_lcfs = np.isclose(rho_val, rho_levels[-1])
        ax.plot(R_surf, Z_surf,
                color=col,
                lw=2.2 if is_lcfs else 0.8,
                ls="-"  if is_lcfs else "--",
                label=f"LCFS  ρ={rho_val:.2f}" if is_lcfs else None)

    ax.plot(R0, 0, "k+", markersize=14, markeredgewidth=2,
            label=f"Magnetic axis  R₀={R0} m")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]", fontsize=12)
    ax.set_ylabel("Z [m]", fontsize=12)

    if geom_mode == "Academic":
        _geom_label = "Academic (elliptic, δ = 0)"
    else:
        _geom_label = f"D0FUS PCHIP (δ_edge = {delta_edge:.3f})"

    ax.set_title(
        f"Flux surfaces — {_geom_label}\n"
        f"R₀ = {R0} m, a = {a} m, κ_edge = {kappa_edge:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_flux_surfaces")


# ---------------------------------------------------------------------------
# A4 — Safety factor profile q(ρ)
# ---------------------------------------------------------------------------

def plot_q_profile(
    run: dict,
    n_rho: int = 400,
    save_dir: str | None = None,
) -> None:
    """
    Plot the safety factor, current density, and enclosed current profiles.

    Uses the self-consistent q-profile data from
    f_q_profile_selfconsistent() when available in run['_q_sc'].
    Falls back to the analytical f_q_profile() model otherwise.

    When self-consistent data is present, the current density panel
    shows the decomposition j_total = j_Ohm+CD + j_bootstrap.

    Three panels:
      (a) q(ρ) — safety factor profile with q₀ and q₉₅ markers.
      (b) j(ρ)/j(0) — normalised current density (total + decomposition).
      (c) I(ρ)/Ip — enclosed current fraction.

    Parameters
    ----------
    run      : dict   D0FUS run output (requires key 'q95').
    n_rho    : int    Number of radial grid points (used only for fallback).
    save_dir : str or None

    References
    ----------
    Wesson, Tokamaks, 4th ed., §3.5 — cylindrical j–q Ampère relation.
    Sauter et al., Phys. Plasmas 6 (1999) 2834 — neoclassical σ.
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054 — PROCESS §18.
    """
    q95     = float(run["q95"])
    Ip      = run.get("Ip", None)
    _q_sc   = run.get("_q_sc", None)

    # ── Data source: self-consistent or analytical fallback ────────────
    if _q_sc is not None and 'q_arr' in _q_sc:
        # Self-consistent path
        rho     = _q_sc['rho']
        q_arr   = _q_sc['q_arr']
        j_total = _q_sc['j_total']
        j_bs    = _q_sc['j_bs']
        j_Ohm   = _q_sc.get('j_Ohm', _q_sc['j_non_bs'])
        j_CD    = _q_sc.get('j_CD', np.zeros_like(rho))
        I_enc   = _q_sc['I_enc']
        li      = _q_sc['li']
        n_iter  = _q_sc['n_iter']
        has_decomposition = True

        # Normalise current profiles to j_total(0)
        j0 = j_total[0] if j_total[0] > 0 else 1.0
        j_norm      = j_total / j0
        j_Ohm_norm  = j_Ohm / j0
        j_CD_norm   = j_CD / j0
        j_bs_norm   = j_bs / j0

        # Normalise enclosed current to Ip
        I_norm = I_enc / np.maximum(I_enc[-1], 1e-3)

        suptitle_str = (
            r"Self-consistent $q(\rho)$ from Amp"
            + "\u00e8"
            + r"re integration"
            + rf"   —   $l_i(3) = {li:.2f}$,  {n_iter} iter."
        )
    else:
        # Analytical fallback (no self-consistent data available)
        rho     = np.linspace(0.0, 0.99, n_rho)
        alpha_J = 1.5   # IPDG89 reference for analytical fallback
        q_arr   = f_q_profile(rho, q95=q95, rho95=0.95, alpha_J=alpha_J)
        has_decomposition = False

        q_edge = q_arr[-1]
        I_norm = q_edge * rho**2 / np.maximum(q_arr, 1e-3)
        I_norm = I_norm / np.maximum(I_norm[-1], 1e-10)
        j_norm = np.maximum(1.0 - rho**2, 0.0)**alpha_J

        rho_s    = np.where(rho > 1e-8, rho, 1.0)
        li_integ = np.where(rho > 1e-8, I_norm**2 / rho_s, 0.0)
        li       = 2.0 * np.trapezoid(li_integ, rho)

        suptitle_str = (
            rf"$q(\rho)$ from $j \propto (1 - \rho^2)^{{\alpha_J}}$"
            rf"   —   analytical fallback ($\alpha_J = 1.5$)"
        )

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) q(ρ) ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(rho, q_arr, "tab:blue", lw=2.2)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(q95, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.95, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.plot(rho[0],  q_arr[0],  "o", color="tab:blue", ms=7, zorder=5)
    ax.plot(0.95, q95, "s", color="tab:red",  ms=7, zorder=5,
            label=rf"$q_{{95}} = {q95:.2f}$")
    ax.text(0.04, 0.92,
            rf"$q_0 = {q_arr[0]:.2f}$" + "\n"
            rf"$l_i(3) = {li:.2f}$" + "\n"
            rf"$q(1) = {q_arr[-1]:.2f}$",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))
    ax.set_xlabel(r"$\rho$", fontsize=12)
    ax.set_ylabel(r"$q(\rho)$", fontsize=12)
    ax.set_title("(a) Safety factor", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    # (b) j(ρ)/j(0) — current density decomposition ───────────────────
    ax = axes[1]
    if has_decomposition:
        ax.plot(rho, j_norm,     color="k",          lw=2.2,         label=r"$j_{total}$")
        ax.plot(rho, j_Ohm_norm, color="tab:blue",   lw=1.5, ls="-", label=r"$j_{Ohm}$")
        ax.plot(rho, j_CD_norm,  color="tab:red",    lw=1.5, ls="-", label=r"$j_{CD}$")
        ax.plot(rho, j_bs_norm,  color="tab:green",  lw=1.5, ls="-", label=r"$j_{bs}$")
        ax.legend(fontsize=8, loc="upper right")
        # Current fraction annotation
        _dA_fig = np.gradient(rho)
        _dA_fig[0] = _dA_fig[1]
        _norm = max(np.sum(j_norm * _dA_fig), 1e-10)
        _f_Ohm = np.sum(j_Ohm_norm * _dA_fig) / _norm
        _f_CD  = np.sum(j_CD_norm  * _dA_fig) / _norm
        _f_bs  = np.sum(j_bs_norm  * _dA_fig) / _norm
        ax.text(0.52, 0.92,
                rf"$I_{{Ohm}}$: {_f_Ohm:.0%}   "
                rf"$I_{{CD}}$: {_f_CD:.0%}   "
                rf"$I_{{bs}}$: {_f_bs:.0%}",
                transform=ax.transAxes, fontsize=8, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))
    else:
        ax.plot(rho, j_norm, "tab:orange", lw=2.2, label=r"$j_{total}$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$\rho$", fontsize=12)
    ax.set_ylabel(r"$j(\rho)\,/\,j(0)$", fontsize=12)
    ax.set_title("(b) Current density decomposition", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=min(0, np.min(j_norm) * 1.1))
    ax.grid(True, alpha=0.25)

    # (c) I(ρ)/Ip ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(rho, I_norm, "tab:green", lw=2.2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\rho$", fontsize=12)
    ax.set_ylabel(r"$I(\rho)\,/\,I_p$", fontsize=12)
    ax.set_title("(c) Enclosed current", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)

    li_text = rf"$l_i = {li:.3f}$"
    if Ip is not None:
        li_text += rf"  ($I_p = {Ip:.1f}$ MA)"
    ax.text(0.04, 0.92, li_text,
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

    fig.suptitle(suptitle_str, fontsize=12, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_q_profile")


# ---------------------------------------------------------------------------
# A6 — Radiation power density profiles
# ---------------------------------------------------------------------------

def plot_radiation_profile(
    run: dict,
    n_rho: int = 400,
    save_dir: str | None = None,
) -> None:
    """
    Plot volumetric radiation power density profiles as a function of ρ.

    Three radiation channels:
      * Bremsstrahlung  ε ∝ n² √T  (normalised shape, right y-axis)
      * Synchrotron     ε ∝ n T^(3/2)  (normalised shape, right y-axis)
      * Line radiation  ε = f_k n² L_{z,k}(T)  [MW/m³] (left y-axis)

    Absolute Bremsstrahlung and synchrotron magnitudes are not computed here
    (they require prefactors and Z_eff² terms); only their radial shapes are
    shown to indicate the dominant emission zone.

    Parameters
    ----------
    run      : dict   D0FUS run output. Impurity fractions read from keys
                      'f_W', 'f_Ne', 'f_Ar' (all optional).
    n_rho    : int    Number of radial grid points.
    save_dir : str or None

    References
    ----------
    Wesson, Tokamaks, 4th ed., §5.3 — Bremsstrahlung / synchrotron.
    Mavrin (2018) / ADAS — coronal cooling coefficients L_z(T).
    """
    nbar, Tbar, nu_n, nu_T, rho_ped, n_ped_frac, T_ped_frac = _resolve_kinetics(run)
    R0 = run.get("R0", "?")

    # Collect impurity concentrations
    imp_map = {k: run.get(k, None) for k in ("f_W", "f_Ne", "f_Ar")}
    # Key mapping: strip 'f_' prefix to get species symbol
    impurities = {k[2:]: v for k, v in imp_map.items()
                  if v is not None and v > 0.0}

    rho = np.linspace(0.0, 1.0, n_rho)
    rad = _radiation_emissivity(
        rho, nbar, Tbar, nu_n, nu_T,
        rho_ped, n_ped_frac, T_ped_frac,
        impurities,
    )

    line_colors = {"W": "tab:red",    "Ne": "tab:cyan",   "Ar": "tab:purple"}
    line_styles = {"W": "-",          "Ne": "-.",          "Ar": "--"}

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Left axis: line radiation [MW/m³]
    for species in impurities:
        if species in rad:
            ax1.plot(rho, rad[species] / 1e6,
                     color=line_colors.get(species, "k"),
                     ls=line_styles.get(species, "-"),
                     lw=2.0, label=f"Line  {species}  (f={impurities[species]:.1e})")

    ax1.set_xlabel(r"$\rho = r/a$", fontsize=12)
    ax1.set_ylabel(r"Line radiation $\varepsilon_{\rm line}$ [MW/m³]", fontsize=12)

    # Right axis: normalised shape profiles
    ax2 = ax1.twinx()
    ax2.plot(rho, rad["P_brem"], "tab:blue",  lw=1.8, ls="--",
             label="Bremsstrahlung (normalised)")
    ax2.plot(rho, rad["P_sync"],  "tab:green", lw=1.8, ls=":",
             label="Synchrotron (normalised)")
    ax2.set_ylabel("Normalised emissivity shape [−]", fontsize=10, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.30)

    # Pedestal marker
    if rho_ped < 0.99:
        ax1.axvline(rho_ped, color="gray", lw=0.9, ls=":", alpha=0.6,
                    label=f"ρ_ped = {rho_ped}")

    # Merged legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper center")

    ax1.set_xlim(0, 1)
    ax1.set_title(
        f"Radiation profiles — R₀={R0} m\n"
        f"n̄={nbar:.2f}×10²⁰ m⁻³,  T̄={Tbar:.1f} keV",
        fontsize=11,
    )
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_radiation_profile")


# ---------------------------------------------------------------------------
# A — Convenience wrapper
# ---------------------------------------------------------------------------

# plot_run_profiles and plot_run_geometry have been merged into plot_all().


# =============================================================================
# 8. Geometry views — cross-sections, coil layouts, machine comparison
# =============================================================================
#
# Additional run-dict keys consumed by geometry functions
# -------------------------------------------------------
# Radial build (all optional — ITER-like defaults applied if absent)
#   b        : float  Total inboard radial build, plasma edge → TF inner face [m]
#   c_TF     : float  TF coil inboard leg radial thickness (WP + nose) [m]
#   c_CS     : float  CS radial thickness [m]
#   N_TF     : int    Number of TF coils
#   Gap      : float  Gap between TF inboard face and CS outer face [m]
# =============================================================================

# Reference machine database — used by plot_cross_section_comparison()
_MACHINE_DB = {
    "JET":     {"R0": 2.96, "a": 1.25, "kappa": 1.70, "delta": 0.40, "color": "#4477AA"},
    "ITER":    {"R0": 6.20, "a": 2.00, "kappa": 1.85, "delta": 0.33, "color": "#EE6677"},
    "EU-DEMO": {"R0": 9.00, "a": 2.90, "kappa": 1.65, "delta": 0.33, "color": "#228833"},
    "STEP":    {"R0": 4.30, "a": 2.40, "kappa": 3.00, "delta": 0.55, "color": "#CCBB44"},
    "ARC":     {"R0": 3.30, "a": 1.13, "kappa": 1.84, "delta": 0.33, "color": "#AA3377"},
    "SPARC":   {"R0": 1.85, "a": 0.57, "kappa": 1.97, "delta": 0.54, "color": "#66CCEE"},
}

# Shading palette for radial build layers
_LAYER_COLORS = {
    "fw":      "#AABBCC",   # First wall        — light steel-blue
    "blanket": "#88BB88",   # Breeding blanket  — soft green
    "shield":  "#BBAA77",   # Shield + VV       — tan
    "tf":      "#6688AA",   # TF coil           — slate blue
    "cs":      "#AA7755",   # CS solenoid       — warm brown
    "plasma":  "#FFEECC",   # Plasma core fill  — pale amber
}


def _lcfs_RZ(R0: float, a: float, kappa: float, delta: float,
             n_theta: int = 500) -> tuple:
    """
    Compute LCFS contour using the Shafranov–Miller parametrisation.

    R(θ) = R₀ + a cos(θ + δ sin θ)
    Z(θ) = a κ sin θ

    This is the standard single-contour form used in Cross_Section.py and
    consistent with the θ = 0 convention of Miller et al. (1998).

    Parameters
    ----------
    R0, a   : float  Major / minor radii [m].
    kappa   : float  Elongation κ (typically evaluated at LCFS).
    delta   : float  Triangularity δ (LCFS value, arcsin form).
    n_theta : int    Number of poloidal angle points.

    Returns
    -------
    R, Z : ndarray  Poloidal contour arrays.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = a * kappa * np.sin(theta)
    return R, Z


def _resolve_build(run: dict) -> dict:
    """
    Extract radial build parameters from run dict, applying ITER-like defaults.

    Returns a flat dict with keys: b, c_TF, c_CS, N_TF, Gap, H_TF,
    R_TF_in, R_TF_out, R_CS_ext, R_CS_int,
    e_fw, e_blanket, e_shield, e_gap.
    """
    R0         = float(run["R0"])
    a          = float(run["a"])
    kappa_edge = float(run.get("kappa_edge", 1.85))
    B0         = float(run.get("B0",   5.3))
    B_max      = float(run.get("B_max", 11.8))

    # Inboard total build — derive from B0/B_max if not provided
    b_default  = R0 * (1.0 - B0 / B_max) - a
    b          = float(run.get("b",    max(b_default, 0.5)))
    c_TF       = float(run.get("c_TF", 0.56))
    c_CS       = float(run.get("c_CS", 0.70))
    N_TF       = int(  run.get("N_TF", 18))
    Gap              = float(run.get("Gap",  0.02))
    # Delta_TF: extra outboard radial clearance imposed by the port-access
    # constraint in Number_TF_coils() [m].  Default 0 for backward compatibility
    # with run dicts produced before this field was added.
    Delta_TF         = float(run.get("Delta_TF", 0.0))
    Choice_Buck_Wedg = run.get("Choice_Buck_Wedg", "Wedging")

    # Derived radii
    R_TF_in  = R0 - a - b                               # Plasma-side face of TF inboard leg [m]
    R_TF_out = R0 + a + b + Delta_TF + c_TF             # Outer face of TF outboard leg [m]
    # Delta_TF shifts the outboard leg outward to match the ripple model:
    # r2 = R0 + a + b + Delta_TF  (inner face of outboard leg, i.e. the plasma-facing face).
    # Effective TF-CS gap: non-zero only in Wedging; zero in Bucking/Plug.
    _gap_eff = Gap if Choice_Buck_Wedg == 'Wedging' else 0.0
    R_CS_ext = R_TF_in - c_TF - _gap_eff           # CS outer face radius [m]
    R_CS_int = max(R_CS_ext - c_CS, 0.05)          # CS inner bore radius [m]

    # TF coil full height — envelope of the D-shape (factor 1.15 above κ*(a+b))
    H_TF       = 2.0 * kappa_edge * (a + b) * 1.15  # Total height [m]

    # Optional inboard layer breakdown (fallback: equal split of b)
    e_fw       = float(run.get("e_fw",      0.05))
    e_blanket  = float(run.get("e_blanket", max(b * 0.38, 0.3)))
    e_shield   = float(run.get("e_shield",  max(b * 0.38, 0.3)))
    e_gap      = max(b - e_fw - e_blanket - e_shield, 0.0)

    return dict(
        b=b, c_TF=c_TF, c_CS=c_CS, N_TF=N_TF, Gap=Gap, Delta_TF=Delta_TF,
        H_TF=H_TF,
        R_TF_in=R_TF_in, R_TF_out=R_TF_out,
        R_CS_ext=R_CS_ext, R_CS_int=R_CS_int,
        e_fw=e_fw, e_blanket=e_blanket, e_shield=e_shield, e_gap=e_gap,
    )


# ---------------------------------------------------------------------------
# B1 — Reference machine comparison (poloidal cross-sections)
# ---------------------------------------------------------------------------

def plot_cross_section_comparison(
    run: dict | None = None,
    machines: list | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Overlay the LCFS contours of reference fusion devices on a single poloidal
    cross-section plot, with an optional D0FUS run design highlighted.

    Uses the Shafranov–Miller single-surface parametrisation.  No radial build
    is shown; only the last closed flux surface of each device is drawn.

    When ``run`` is provided, its geometry (R₀, a, κ, δ) is drawn as a thick
    red LCFS labelled "D0FUS".  When ``run`` is None a default test machine
    is used (R₀ = 6.0 m, a = 1.5 m, κ = 1.85, δ = 0.33).

    Built-in machine database (from published design values):

    +----------+------+------+-------+------+----------------------------------+
    | Machine  |  R₀  |  a   |   κ   |  δ   | Reference                        |
    +----------+------+------+-------+------+----------------------------------+
    | JET      | 2.96 | 1.25 | 1.70  | 0.40 | Wesson (2011), Table B.1         |
    | ITER     | 6.20 | 2.00 | 1.85  | 0.33 | Shimada et al., NF 47 S1 (2007)  |
    | EU-DEMO  | 9.00 | 2.90 | 1.65  | 0.33 | Federici et al., NF 58 (2018)    |
    | STEP     | 4.30 | 2.40 | 3.00  | 0.55 | Wilson et al., NF 60 (2020)      |
    | ARC      | 3.30 | 1.13 | 1.84  | 0.33 | Sorbom et al., FED 100 (2015)    |
    | SPARC    | 1.85 | 0.57 | 1.97  | 0.54 | Creely et al., JPP 86 (2020)     |
    +----------+------+------+-------+------+----------------------------------+

    Parameters
    ----------
    run       : dict or None
        D0FUS run dict.  Keys used: R0, a, kappa_edge, delta_edge.
        If None, a default test geometry is drawn.
    machines  : list of str or None
        Subset of machine keys to plot. None → all machines in _MACHINE_DB.
    save_dir  : str or None
    """
    subset = machines if machines is not None else list(_MACHINE_DB.keys())

    fig, ax = plt.subplots(figsize=(11, 9))

    # --- Reference machines (thin lines) ---
    for name in subset:
        if name not in _MACHINE_DB:
            continue
        p = _MACHINE_DB[name]
        R, Z = _lcfs_RZ(p["R0"], p["a"], p["kappa"], p["delta"])
        ax.plot(R,  Z, color=p["color"], lw=1.6, alpha=0.75, label=name)
        ax.plot(-R, Z, color=p["color"], lw=0.9, ls="--", alpha=0.40)
        ax.plot( p["R0"], 0, ".", color=p["color"], ms=5)
        ax.plot(-p["R0"], 0, ".", color=p["color"], ms=5)

    # --- D0FUS design (thick, highlighted) ---
    if run is not None:
        d_R0    = float(run["R0"])
        d_a     = float(run["a"])
        d_kappa = float(run.get("kappa_edge", 1.85))
        d_delta = float(run.get("delta_edge", 0.33))
    else:
        # Default test machine for standalone smoke test
        d_R0, d_a, d_kappa, d_delta = 6.0, 1.5, 1.85, 0.33

    R_d, Z_d = _lcfs_RZ(d_R0, d_a, d_kappa, d_delta)
    ax.plot(R_d,  Z_d, color="red", lw=3.0, zorder=5, label="D0FUS")
    ax.plot(-R_d, Z_d, color="red", lw=1.8, ls="--", alpha=0.55, zorder=5)
    ax.plot( d_R0, 0, "r.", ms=8, zorder=6)
    ax.plot(-d_R0, 0, "r.", ms=8, zorder=6)

    ax.axhline(0, color="k", lw=0.6, alpha=0.3, ls="--")
    ax.axvline(0, color="k", lw=0.6, alpha=0.3, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("R  [m]", fontsize=12)
    ax.set_ylabel("Z  [m]", fontsize=12)
    ax.set_title("Tokamak LCFS comparison", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "cross_section_comparison")


# ---------------------------------------------------------------------------
# B2 — TF coil Princeton-D view (poloidal plane)
# ---------------------------------------------------------------------------

def _princeton_D_contour(
    r1: float,
    r2: float,
    n_leg: int = 50,
) -> tuple:
    """
    Compute a Princeton-D constant-tension coil contour in the (R, Z) plane.

    The shape is obtained by numerical integration of the arc-length ODE
    for a filamentary conductor in a 1/R toroidal field:

        dR/ds     = cos(α)
        dZ/ds     = sin(α)
        dα/ds     = 1 / (k · R)

    where  k = ½ ln(r₂ / r₁)  is the shape parameter.  Integration starts
    at (r₂, 0) with α = π/2 (vertical upward) and proceeds until α = 3π/2
    (vertical downward at the inboard straight-leg junction).

    The full closed contour is assembled as:
      top constant-tension arc + straight inboard leg + mirrored bottom arc.

    Parameters
    ----------
    r1    : float  Inboard leg radial position [m].
    r2    : float  Outboard midplane radial position [m].
    n_leg : int    Number of discretisation points on the straight leg.

    Returns
    -------
    R, Z : ndarray  Closed contour arrays (ready for ax.fill / ax.plot).

    References
    ----------
    File, Stewart & Mills, IEEE TNS 18 (1971) — Princeton-D concept.
    Gralnick & Tenney, J. Appl. Phys. 47 (1976) — Analytical solution.
    """
    from scipy.integrate import solve_ivp

    k = 0.5 * np.log(r2 / r1)

    def _rhs(_s, y):
        """Arc-length ODE right-hand side."""
        _r, _z, _alpha = y
        return [np.cos(_alpha), np.sin(_alpha), 1.0 / (k * _r)]

    # Terminal event: stop when tangent angle reaches 3π/2 (vertical downward)
    def _evt_alpha(_s, y):
        return y[2] - 3.0 * np.pi / 2.0
    _evt_alpha.terminal = True

    y0 = [r2, 0.0, np.pi / 2.0]
    s_max = 30.0 * r2                  # Generous arc-length upper bound
    sol = solve_ivp(_rhs, [0.0, s_max], y0,
                    events=_evt_alpha,
                    max_step=0.01 * r2,
                    rtol=1e-10, atol=1e-12)

    r_top = sol.y[0]                   # Top half: (r2, 0) → (r1, z_leg)
    z_top = sol.y[1]
    z_leg = z_top[-1]
    r_leg = r_top[-1]                  # Should be ≈ r1

    # Straight inboard leg (top → bottom)
    z_leg_pts = np.linspace(z_leg, -z_leg, n_leg)
    r_leg_pts = np.full_like(z_leg_pts, r_leg)

    # Bottom half: time-reversed mirror of the top half
    r_bot = r_top[::-1]
    z_bot = -z_top[::-1]

    # Assemble closed contour
    R = np.concatenate([r_top, r_leg_pts[1:-1], r_bot])
    Z = np.concatenate([z_top, z_leg_pts[1:-1], z_bot])
    return R, Z


def _offset_contour(R: np.ndarray, Z: np.ndarray, d: float) -> tuple:
    """
    Compute an inward-offset (parallel) curve at constant distance *d*.

    The Princeton-D contour as assembled by ``_princeton_D_contour`` is
    wound **clockwise** in the (R, Z) plane.  For a CW contour the
    inward-pointing unit normal is  n = (−tZ, +tR)  where (tR, tZ) is
    the unit tangent.

    To avoid artefacts at the corners where the curved arcs meet the
    straight inboard leg, the contour is split into three segments
    (top arc, straight leg, bottom arc); each is offset independently
    and then stitched back together.

    Parameters
    ----------
    R, Z : ndarray  Original closed contour (clockwise winding).
    d    : float    Offset distance [m].  Positive = inward.

    Returns
    -------
    R_off, Z_off : ndarray  Offset contour arrays.
    """
    # --- Identify inboard straight leg (R ≈ R_min) ---
    R_min = R.min()
    tol = 0.01 * (R.max() - R_min)
    leg_idx = np.where(np.abs(R - R_min) < tol)[0]

    if len(leg_idx) < 2:
        # Fallback: simple normal offset everywhere (CW convention)
        dR = np.gradient(R); dZ = np.gradient(Z)
        ds = np.hypot(dR, dZ); ds[ds < 1e-15] = 1e-15
        return R - d * dZ / ds, Z + d * dR / ds

    i_s = leg_idx[0]                   # First index on the straight leg
    i_e = leg_idx[-1]                  # Last index on the straight leg

    # --- Helper: offset an open arc segment (CW inward normal) ---
    def _offset_arc(r_seg, z_seg):
        dr = np.gradient(r_seg)
        dz = np.gradient(z_seg)
        ds = np.hypot(dr, dz)
        ds[ds < 1e-15] = 1e-15
        # CW inward normal: n = (-dz, +dr) / ds
        return r_seg - d * dz / ds, z_seg + d * dr / ds

    # Top arc:  indices [0 .. i_s]  (outboard midplane → inboard junction)
    ro_top, zo_top = _offset_arc(R[:i_s + 1], Z[:i_s + 1])

    # Straight leg:  indices [i_s .. i_e]  — inward normal is simply (+d, 0)
    r_leg = R[i_s:i_e + 1] + d
    z_leg = Z[i_s:i_e + 1]

    # Bottom arc:  indices [i_e .. end]  (inboard junction → outboard midplane)
    ro_bot, zo_bot = _offset_arc(R[i_e:], Z[i_e:])

    # Stitch together (skip duplicate junction points)
    R_off = np.concatenate([ro_top[:-1], r_leg, ro_bot[1:]])
    Z_off = np.concatenate([zo_top[:-1], z_leg, zo_bot[1:]])

    return R_off, Z_off


def plot_TF_side_view(
    run: dict,
    save_dir: str | None = None,
) -> None:
    """
    Draw one TF coil in the poloidal (R-Z) plane as a Princeton-D shape.

    The coil contour follows the analytical constant-tension curve of
    Gralnick & Tenney (1976), computed by numerical integration of the
    arc-length ODE.  The inner (bore) boundary is obtained as a
    constant-distance inward offset (parallel curve) of the outer
    contour, ensuring uniform winding-pack thickness c_TF everywhere.
    The figure uses a technical-drawing black-and-white style.

    Only three dimension annotations are drawn:
      * c_TF  — coil radial thickness (constant around the D)
      * W_TF  — total coil radial width  (R_out − R_bore)
      * H_TF  — total coil vertical height (from the ODE solution)

    Parameters
    ----------
    run      : dict   D0FUS run output including radial build keys.
    save_dir : str or None

    References
    ----------
    File, Stewart & Mills, IEEE TNS 18 (1971) — Princeton-D concept.
    Gralnick & Tenney, J. Appl. Phys. 47 (1976) — Analytical solution.
    """
    # ── Extract geometry from run dict ──────────────────────────────
    R0         = float(run["R0"])
    a          = float(run["a"])
    kappa_edge = float(run.get("kappa_edge", 1.85))
    delta_edge = float(run.get("delta_edge", 0.33))
    bd         = _resolve_build(run)

    c_TF     = bd["c_TF"]
    R_TF_in  = bd["R_TF_in"]          # Inboard bore face (plasma side)
    R_TF_out = bd["R_TF_out"]         # Outboard outer face
    R_bore   = R_TF_in - c_TF         # Inboard outer face (machine-axis side)
    W_TF     = R_TF_out - R_bore      # Total radial width [m]

    # ── Build Princeton-D contours ──────────────────────────────────
    R_out, Z_out = _princeton_D_contour(R_bore, R_TF_out)

    # Inner contour: constant-thickness offset of the outer D
    R_in, Z_in = _offset_contour(R_out, Z_out, c_TF)

    # Actual coil height from the ODE solution
    H_TF = 2.0 * Z_out.max()
    h    = Z_out.max()

    # ── Black-and-white colour scheme ───────────────────────────────
    col_fill = "black"              # Light grey fill for coil body
    col_line = "black"                # All contour lines
    col_dim  = "#333333"              # Dimension annotations

    # ── Create figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 7))

    # 1. TF coil body — grey fill between outer and inner contour
    ax.fill(R_out, Z_out, fc=col_fill, ec="none", zorder=2)
    ax.fill(R_in,  Z_in,  fc="white", ec="none", zorder=3)

    # 2. Contour outlines
    ax.plot(R_out, Z_out, color=col_line, lw=1.8, zorder=4)
    ax.plot(R_in,  Z_in,  color=col_line, lw=1.0, zorder=4)

    # 3. Centre line — midplane axis
    ax.axhline(0, color=col_line, lw=0.4, ls="-.", alpha=0.4, zorder=1)

    # ── Three dimension annotations (engineering-drawing style) ─────
    arr_kw = dict(arrowstyle="<->", color=col_dim, lw=0.9,
                  mutation_scale=10)
    ext_kw = dict(color=col_dim, lw=0.4, ls="-", alpha=0.5)
    lbl_kw = dict(fontsize=9, color=col_dim, ha="center",
                  bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))

    z_bottom = -h

    # --- 1. c_TF — coil thickness (inboard leg, below coil) ---
    z_dim = z_bottom - 0.4
    ax.plot([R_bore,  R_bore],  [z_bottom, z_dim - 0.05], **ext_kw, zorder=9)
    ax.plot([R_TF_in, R_TF_in], [z_bottom, z_dim - 0.05], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_TF_in, z_dim), xytext=(R_bore, z_dim),
                arrowprops=arr_kw, zorder=9)
    ax.text((R_bore + R_TF_in) / 2, z_dim - 0.25,
            f"$c_{{TF}}$ = {c_TF:.2f} m", va="top", **lbl_kw)

    # --- 2. W_TF — total width (further below) ---
    z_dim2 = z_dim - 0.9
    ax.plot([R_bore,   R_bore],   [z_dim - 0.05, z_dim2 - 0.05], **ext_kw, zorder=9)
    ax.plot([R_TF_out, R_TF_out], [z_bottom,     z_dim2 - 0.05], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_TF_out, z_dim2), xytext=(R_bore, z_dim2),
                arrowprops=arr_kw, zorder=9)
    ax.text((R_bore + R_TF_out) / 2, z_dim2 - 0.25,
            f"$W_{{TF}}$ = {W_TF:.2f} m", va="top", **lbl_kw)

    # --- 3. H_TF — total height (left side) ---
    R_vann = R_bore - 0.35
    ax.plot([R_bore, R_vann - 0.03], [ h,  h], **ext_kw, zorder=9)
    ax.plot([R_bore, R_vann - 0.03], [-h, -h], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_vann, h), xytext=(R_vann, -h),
                arrowprops=arr_kw, zorder=9)
    ax.text(R_vann - 0.20, 0,
            f"$H_{{TF}}$ = {H_TF:.2f} m", rotation=90,
            fontsize=9, color=col_dim, ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))

    # ── Axes styling ────────────────────────────────────────────────
    ax.set_aspect("equal")
    ax.set_xlabel("$R$  [m]", fontsize=11)
    ax.set_ylabel("$Z$  [m]", fontsize=11)
    ax.set_title(
        "TF coil — Princeton-D  (poloidal cross-section)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(False)
    ax.tick_params(labelsize=10, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Axis limits — generous padding to avoid label clipping
    ax.set_xlim(R_bore - 1.5, R_TF_out + 0.8)
    ax.set_ylim(-h - 2.2, h + 0.8)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_TF_side_view")


# ---------------------------------------------------------------------------
# B3 — CS solenoid cross-section (R–Z plane)
# ---------------------------------------------------------------------------

def plot_CS_cross_section(
    run: dict,
    save_dir: str | None = None,
) -> None:
    """
    Draw a full poloidal cross-section of the central solenoid (R-Z plane).

    The view cuts through the machine axis, showing both the left (-R) and
    right (+R) halves of the CS as two black rectangular winding packs
    symmetric about the tokamak axis (R = 0).  A light grey background
    rectangle behind each half conveys the CS envelope.  The central bore
    is left white.

    The figure uses a technical-drawing black-and-white style consistent
    with ``plot_TF_side_view``.

    Parameters
    ----------
    run      : dict   D0FUS run output including radial build keys.
    save_dir : str or None

    References
    ----------
    ITER Technical Basis, IAEA NF 40 (2001), §2.2.
    Sarasola et al., IEEE Trans. Appl. Supercond. 33, 1-5 (2023).
    """
    from matplotlib.patches import Rectangle

    bd = _resolve_build(run)

    R_CS_int = bd["R_CS_int"]          # Inner radius of winding pack [m]
    R_CS_ext = bd["R_CS_ext"]          # Outer radius of winding pack [m]
    c_CS     = bd["c_CS"]              # Winding-pack radial thickness [m]
    H_TF     = bd["H_TF"]
    H_CS     = H_TF * 0.88            # CS slightly shorter than TF [m]
    h_cs     = H_CS / 2.0

    # ── Black-and-white colour scheme (matches TF side view) ────────
    col_wp   = "black"                 # Winding-pack fill
    col_bg   = "#D8D8D8"              # Light grey CS envelope background
    col_line = "black"                 # Contour lines
    col_dim  = "#333333"               # Dimension annotations

    # ── Create figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 6))

    # --- Continuous grey CS body visible behind the cut plane ---
    # One rectangle spanning from -R_CS_ext to +R_CS_ext mimics the
    # full solenoid cylinder seen in the background of the cross-section.
    ax.add_patch(Rectangle(
        (-R_CS_ext, -h_cs), 2 * R_CS_ext, H_CS,
        facecolor=col_bg, edgecolor="none", zorder=1,
    ))

    # --- Black winding-pack rectangles (left and right halves) ---
    for sign in (+1, -1):
        wp_x = R_CS_int if sign > 0 else -R_CS_ext
        ax.add_patch(Rectangle(
            (wp_x, -h_cs), c_CS, H_CS,
            facecolor=col_wp, edgecolor=col_line, lw=1.5, zorder=3,
        ))

    # --- Machine axis (vertical dash-dot line at R = 0) ---
    ax.axvline(0, color=col_line, lw=0.5, ls="-.", alpha=0.4, zorder=1)

    # --- Midplane axis (horizontal dash-dot line at Z = 0) ---
    ax.axhline(0, color=col_line, lw=0.4, ls="-.", alpha=0.4, zorder=1)

    # ── Dimension annotations (engineering-drawing style) ───────────
    arr_kw = dict(arrowstyle="<->", color=col_dim, lw=0.9,
                  mutation_scale=10)
    ext_kw = dict(color=col_dim, lw=0.4, ls="-", alpha=0.5)
    lbl_kw = dict(fontsize=9, color=col_dim, ha="center",
                  bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))

    # --- 1. c_CS — winding-pack radial thickness (below right half) ---
    z_dim1 = -h_cs - 0.35
    ax.plot([R_CS_int, R_CS_int], [-h_cs, z_dim1 - 0.05], **ext_kw, zorder=9)
    ax.plot([R_CS_ext, R_CS_ext], [-h_cs, z_dim1 - 0.05], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_CS_ext, z_dim1), xytext=(R_CS_int, z_dim1),
                arrowprops=arr_kw, zorder=9)
    ax.text((R_CS_int + R_CS_ext) / 2, z_dim1 - 0.22,
            f"$c_{{CS}}$ = {c_CS:.3f} m", va="top", **lbl_kw)

    # --- 2. 2·R_CS_ext — full outer diameter (further below) ---
    z_dim2 = z_dim1 - 0.8
    ax.plot([-R_CS_ext, -R_CS_ext], [z_dim1 - 0.05, z_dim2 - 0.05],
            **ext_kw, zorder=9)
    ax.plot([ R_CS_ext,  R_CS_ext], [z_dim1 - 0.05, z_dim2 - 0.05],
            **ext_kw, zorder=9)
    ax.annotate("", xy=(R_CS_ext, z_dim2), xytext=(-R_CS_ext, z_dim2),
                arrowprops=arr_kw, zorder=9)
    ax.text(0, z_dim2 - 0.22,
            f"$2\\,R_{{ext}}$ = {2 * R_CS_ext:.3f} m", va="top", **lbl_kw)

    # --- 3. H_CS — total height (left side) ---
    R_vann = -R_CS_ext - 0.35
    ax.plot([-R_CS_ext, R_vann - 0.03], [ h_cs,  h_cs], **ext_kw, zorder=9)
    ax.plot([-R_CS_ext, R_vann - 0.03], [-h_cs, -h_cs], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_vann, h_cs), xytext=(R_vann, -h_cs),
                arrowprops=arr_kw, zorder=9)
    ax.text(R_vann - 0.18, 0,
            f"$H_{{CS}}$ = {H_CS:.2f} m", rotation=90,
            fontsize=9, color=col_dim, ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))

    # --- 4. R_CS_int — bore radius (top, right side only) ---
    z_top = h_cs + 0.30
    ax.plot([0, 0],          [h_cs, z_top + 0.05], **ext_kw, zorder=9)
    ax.plot([R_CS_int, R_CS_int], [h_cs, z_top + 0.05], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_CS_int, z_top), xytext=(0, z_top),
                arrowprops=arr_kw, zorder=9)
    ax.text(R_CS_int / 2, z_top + 0.18,
            f"$R_{{int}}$ = {R_CS_int:.3f} m", va="bottom", **lbl_kw)

    # ── Axes styling (matches TF side view) ─────────────────────────
    ax.set_aspect("equal")
    ax.set_xlabel("$R$  [m]", fontsize=11)
    ax.set_ylabel("$Z$  [m]", fontsize=11)
    ax.set_title(
        "CS solenoid — poloidal cross-section",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(False)
    ax.tick_params(labelsize=10, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Axis limits — generous padding to avoid label clipping
    x_min = R_vann - 1.60             # Left of H_CS label
    x_max = R_CS_ext + 1.70           # Right margin
    y_min = z_dim2 - 1.15             # Below 2·R_ext label
    y_max = z_top + 1.15              # Above R_int label
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_CS_cross_section")


# ---------------------------------------------------------------------------
# B — Convenience wrapper
# ---------------------------------------------------------------------------


# =============================================================================
# Unified figure catalogue — plot_all
# =============================================================================

def plot_all(
    run: dict,
    save_dir: str | None = None,
    cfg=None,
) -> None:
    """
    Render the complete D0FUS figure catalogue in a single numbered sequence.

    Figures are grouped by theme:

      ── Plasma shaping           [ 1– 7]
      ── Kinetic profiles         [ 8–11]
      ── Transport & current      [12–14]
      ── Radiation & impurities   [15–17]
      ── Superconductor eng.      [18–20]
      ── Coil sizing & mechanics  [21–26]
      ── Machine comparison       [27]

    Parameters
    ----------
    run      : dict   D0FUS run output dict (ITER Q=10 reference by default).
    save_dir : str or None
        If provided, all figures are saved to that directory as PNG files.
        Pass ``None`` to display each figure interactively.
    cfg      : config object (DEFAULT_CONFIG if None).
    """
    N = 27

    def _p(i, label):
        print(f"  [{i:2d}/{N}] {label}")

    print("D0FUS_figures — generating full catalogue...")

    # ── Plasma shaping ────────────────────────────────────────────────
    _p(1, "κ scaling curves")
    plot_kappa_scaling(save_dir=save_dir)

    _p(2, "Shaping profiles κ(ρ), δ(ρ) — validation")
    plot_shaping_profiles(save_dir=save_dir)

    _p(3, "Run shaping profiles κ(ρ), δ(ρ)")
    plot_shaping_run(run, save_dir=save_dir)

    _p(4, "Miller flux surfaces — validation")
    plot_miller_surfaces(save_dir=save_dir)

    _p(5, "Run Miller flux surfaces")
    plot_flux_surfaces_run(run, save_dir=save_dir)

    _p(6, "Plasma volume comparison")
    plot_volume_comparison(save_dir=save_dir)

    _p(7, "First wall surface")
    plot_first_wall_surface(save_dir=save_dir)

    # ── Kinetic profiles ──────────────────────────────────────────────
    _p(8, "Normalised n̂, T̂, p̂ profiles")
    plot_nT_profiles(save_dir=save_dir)

    _p(9, "Run n(ρ), T(ρ), p(ρ)")
    plot_run_nTp(run, save_dir=save_dir)

    _p(10, "Line vs volume density")
    plot_density_line_vol(save_dir=save_dir)

    _p(11, "Safety factor q(ρ)")
    plot_q_profile(run, save_dir=save_dir)

    _p(12, "Resistivity models")
    plot_resistivity_models(save_dir=save_dir)

    _p(13, "Radiation profiles")
    plot_radiation_profile(run, save_dir=save_dir)

    # ── Radiation & impurities ────────────────────────────────────────
    _p(14, "Coronal cooling coefficient L_z(T)")
    plot_Lz_cooling(save_dir=save_dir)

    _p(16, "Helium ash fraction")
    plot_He_fraction(save_dir=save_dir)

    # ── Superconductor engineering ────────────────────────────────────
    _p(17, "Jc scalings (NbTi / Nb₃Sn / REBCO)")
    plot_Jc_scaling(save_dir=save_dir)

    _p(18, "Cable current density J_wost(B)")
    plot_cable_current_density(cfg=cfg, save_dir=save_dir)

    _p(19, "CICC equivalent conductor model (TF + CS)")
    fig_cicc, axes_cicc = plt.subplots(1, 2, figsize=(12, 7))
    fig_cicc.suptitle("CICC equivalent conductor model",
                      fontsize=13, fontweight="bold")
    conductor_TF = build_conductor_from_run(run, coil="TF")
    conductor_CS = build_conductor_from_run(run, coil="CS")
    plot_CICC_cross_section(conductor_TF, ax=axes_cicc[0])
    plot_CICC_cross_section(conductor_CS, ax=axes_cicc[1])
    plt.tight_layout()
    _save_or_show(fig_cicc, save_dir, "conductor_TF_CS")

    # ── Coil sizing & mechanics ───────────────────────────────────────
    _p(20, "TF thickness vs peak field")
    plot_TF_thickness_vs_field(cfg=cfg, save_dir=save_dir)

    _p(21, "CS thickness vs flux swing")
    plot_CS_thickness_vs_flux(cfg=cfg, save_dir=save_dir)

    _p(22, "CIRCE stress-model validation")
    plot_CIRCE_stress_validation(save_dir=save_dir)

    _p(23, "TF coil side view")
    plot_TF_side_view(run, save_dir=save_dir)

    _p(24, "CS cross-section")
    plot_CS_cross_section(run, save_dir=save_dir)

    # ── Benchmarks & machine comparison ───────────────────────────────
    _p(25, "TF benchmark table")
    plot_TF_benchmark_table(cfg=cfg, save_dir=save_dir)

    _p(26, "CS benchmark table")
    plot_CS_benchmark_table(cfg=cfg, save_dir=save_dir)

    _p(27, "Tokamak LCFS comparison")
    plot_cross_section_comparison(run=run, save_dir=save_dir)

    print("Done.")


def plot_run(
    run: dict,
    save_dir: str | None = None,
) -> None:
    """
    Render the run-specific figure set (10 figures).

    This is the subset called after each D0FUS run.  It contains only the
    figures that depend on the current run configuration and results —
    no validation curves, no benchmarks, no scaling-law surveys.

    Figures produced:
      [ 1/10]  Tokamak LCFS comparison (with D0FUS overlay)
      [ 2/10]  Miller flux surfaces (run geometry)
      [ 3/10]  Shaping profiles κ(ρ), δ(ρ)
      [ 4/10]  Kinetic profiles n(ρ), T(ρ), p(ρ)
      [ 5/10]  Safety factor q(ρ) and current decomposition
      [ 6/10]  Radiation profiles
      [ 7/10]  TF coil side view
      [ 8/10]  CICC TF conductor
      [ 9/10]  CS cross-section
      [10/10]  CICC CS conductor

    Parameters
    ----------
    run      : dict   D0FUS run output dict.
    save_dir : str or None
        If provided, figures are saved as PNG files.
        Pass ``None`` to display interactively.
    """
    N = 10

    def _p(i, label):
        print(f"  [{i:2d}/{N}] {label}")

    print("D0FUS_figures — generating run figures...")

    # ── Geometry ──────────────────────────────────────────────────────
    _p(1, "Tokamak LCFS comparison")
    plot_cross_section_comparison(run=run, save_dir=save_dir)

    _p(2, "Miller flux surfaces")
    plot_flux_surfaces_run(run, save_dir=save_dir)

    _p(3, "Shaping profiles κ(ρ), δ(ρ)")
    plot_shaping_run(run, save_dir=save_dir)

    # ── Kinetic profiles ──────────────────────────────────────────────
    _p(4, "Kinetic profiles n(ρ), T(ρ), p(ρ)")
    plot_run_nTp(run, save_dir=save_dir)

    _p(5, "Safety factor q(ρ)")
    plot_q_profile(run, save_dir=save_dir)

    # ── Transport & radiation ─────────────────────────────────────────
    _p(6, "Radiation profiles")
    plot_radiation_profile(run, save_dir=save_dir)

    # ── Coils & conductors ────────────────────────────────────────────
    _p(7, "TF coil side view")
    plot_TF_side_view(run, save_dir=save_dir)

    _p(8, "CICC TF conductor")
    plot_CICC_cross_section(build_conductor_from_run(run, coil="TF"), save_dir=save_dir)

    _p(9, "CS cross-section")
    plot_CS_cross_section(run, save_dir=save_dir)

    _p(10, "CICC CS conductor")
    plot_CICC_cross_section(build_conductor_from_run(run, coil="CS"), save_dir=save_dir)

    print("Done.")




# =============================================================================
# 7. CIRCE stress-model validation
# =============================================================================

def plot_CIRCE_stress_validation(
    nu: float = 0.3,
    save_dir=None,
) -> None:
    """
    Validate the F_CIRCE0D finite-element stress solver against three
    analytical reference cases:

      Test 1 : Pressurised thin cylinder — Lamé exact solution.
      Test 2 : Central solenoid with three active current-carrying layers.
      Test 3 : Composite conductor + steel jacket under J×B body load.

    Parameters
    ----------
    nu       : float  Poisson's ratio (applied to all tests) [-].
    save_dir : str or None

    References
    ----------
    Lamé G. (1852) — thick-cylinder pressure vessel solution.
    Sarasola et al., IEEE Trans. Appl. Supercond. 33, 1-5 (2023).
    """
    # --- Test 1: Lamé cylinder ---
    R_lame = [1.0, 2.0]
    Pi_lame = 100e6  # Internal pressure [Pa]
    sigma_r, sigma_t, _, r, _ = F_CIRCE0D(
        100, R_lame, [0.0], [0.0], Pi_lame, 0.0, [200e9], nu, [1])
    a_l, b_l = R_lame
    sigma_r_lame = Pi_lame * a_l**2 / (b_l**2 - a_l**2) * (1 - b_l**2 / r**2)
    sigma_t_lame = Pi_lame * a_l**2 / (b_l**2 - a_l**2) * (1 + b_l**2 / r**2)

    # --- Test 2: CS with 3 active layers ---
    R_cs = [1.3, 1.5, 1.7, 2.0]
    sigma_r_cs, sigma_t_cs, _, r_cs, _ = F_CIRCE0D(
        50, R_cs, [40e6, 45e6, 50e6], [13.0, 11.0, 8.0],
        0.0, 0.0, [50e9, 50e9, 50e9], nu, [1, 1, 1])
    sigma_vm_cs = compute_von_mises_stress(sigma_r_cs, sigma_t_cs)

    # --- Test 3: Composite conductor + steel ---
    R_comp = [1.0, 1.4, 1.5]
    sigma_r_comp, sigma_t_comp, _, r_comp, _ = F_CIRCE0D(
        100, R_comp, [50e6, 0.0], [13.0, 0.0],
        0.0, 0.0, [50e9, 200e9], nu, [1, 1])
    sigma_vm_comp = compute_von_mises_stress(sigma_r_comp, sigma_t_comp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Test 1 — Lamé comparison
    ax = axes[0]
    ax.plot(r, sigma_r       / 1e6, "b-",  lw=2.0, label=r"$\sigma_r$ (CIRCE0D)")
    ax.plot(r, sigma_t       / 1e6, "r-",  lw=2.0, label=r"$\sigma_\theta$ (CIRCE0D)")
    ax.plot(r, sigma_r_lame  / 1e6, "b--", lw=1.5, label=r"$\sigma_r$ (Lamé)")
    ax.plot(r, sigma_t_lame  / 1e6, "r--", lw=1.5, label=r"$\sigma_\theta$ (Lamé)")
    ax.set_xlabel("r [m]", fontsize=11)
    ax.set_ylabel("Stress [MPa]", fontsize=11)
    ax.set_title("Test 1: Pressurised cylinder\n(Lamé analytical validation)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 2 — CS three-layer
    ax = axes[1]
    ax.plot(r_cs, sigma_r_cs  / 1e6, "b-",  lw=2, label=r"$\sigma_r$")
    ax.plot(r_cs, sigma_t_cs  / 1e6, "r-",  lw=2, label=r"$\sigma_\theta$")
    ax.plot(r_cs, sigma_vm_cs / 1e6, "g--", lw=2, label=r"$\sigma_{\rm VM}$")
    for ri in R_cs[1:-1]:
        ax.axvline(ri, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("r [m]", fontsize=11)
    ax.set_ylabel("Stress [MPa]", fontsize=11)
    ax.set_title("Test 2: CS with 3 active layers", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 3 — Composite structure
    ax = axes[2]
    ax.plot(r_comp, sigma_r_comp  / 1e6, "b-",  lw=2, label=r"$\sigma_r$")
    ax.plot(r_comp, sigma_t_comp  / 1e6, "r-",  lw=2, label=r"$\sigma_\theta$")
    ax.plot(r_comp, sigma_vm_comp / 1e6, "g--", lw=2, label=r"$\sigma_{\rm VM}$")
    ax.axvline(R_comp[1], color="orange", lw=2, ls="-")
    ax.axvspan(R_comp[0], R_comp[1], alpha=0.15, color="blue",  label="Conductor")
    ax.axvspan(R_comp[1], R_comp[2], alpha=0.15, color="gray",  label="Steel jacket")
    ax.set_xlabel("r [m]", fontsize=11)
    ax.set_ylabel("Stress [MPa]", fontsize=11)
    ax.set_title("Test 3: Conductor + steel jacket", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("CIRCE0D stress-model validation suite",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "CIRCE_stress_validation")


# =============================================================================
# 8. TF coil benchmark table
# =============================================================================

def plot_TF_benchmark_table(cfg=None, save_dir=None) -> None:
    """
    Display a TF coil benchmark table comparing D0FUS thickness predictions
    against published reference values for ITER, EU-DEMO, JT-60SA, EAST,
    ARC, and SPARC.

    The table is rendered as a matplotlib figure (colour-coded header) so it
    can be saved to a file alongside the other run figures.

    Parameters
    ----------
    cfg      : config object (DEFAULT_CONFIG if None).
    save_dir : str or None

    References
    ----------
    Sborchia et al., IEEE Trans. Appl. Supercond. 18, 463 (2008) — ITER TF.
    Federici et al., Nucl. Fusion 64, 036025 (2024) — EU-DEMO.
    Creely et al., J. Plasma Phys. 86, 865860502 (2020) — SPARC.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    machines_TF = {
        "ITER":    {"a": 2.00, "b": 1.2,  "R0": 6.20, "σ": 660e6,  "T_op": 4.2,
                    "B_max": 11.8, "n_TF": 1,   "sc": "Nb3Sn", "config": "Wedging",
                    "κ": 1.7,  "I_cond": 68e3, "V_max": 10e3, "N_sub": 9,
                    "tau_h": 2.0,  "J_wost": 35e6},
        "EU-DEMO": {"a": 3.0,  "b": 1.80, "R0": 9.0,  "σ": 660e6,  "T_op": 4.2,
                    "B_max": 13.0, "n_TF": 0.5, "sc": "Nb3Sn", "config": "Wedging",
                    "κ": 1.7,  "I_cond": 80e3, "V_max": 10e3, "N_sub": 8,
                    "tau_h": 2.0,  "J_wost": 35e6},
        "JT60-SA": {"a": 1.18, "b": 0.3,  "R0": 2.96, "σ": 660e6,  "T_op": 4.2,
                    "B_max":  5.65, "n_TF": 1,   "sc": "NbTi",  "config": "Wedging",
                    "κ": 1.8,  "I_cond": 25.7e3,"V_max": 5e3,  "N_sub": 9,
                    "tau_h": 1.0,  "J_wost": 65e6},
        "EAST":    {"a": 0.45, "b": 0.4,  "R0": 1.85, "σ": 660e6,  "T_op": 3.7,
                    "B_max":  5.8,  "n_TF": 1,   "sc": "NbTi",  "config": "Wedging",
                    "κ": 1.9,  "I_cond": 14.5e3,"V_max": 5e3,  "N_sub": 8,
                    "tau_h": 1.0,  "J_wost": 50e6},
        "ARC":     {"a": 1.07, "b": 0.9,  "R0": 3.30, "σ": 1000e6, "T_op": 20.0,
                    "B_max": 23.0,  "n_TF": 1,   "sc": "REBCO", "config": "Plug",
                    "κ": 1.8,  "I_cond": 50e3, "V_max": 10e3, "N_sub": 9,
                    "tau_h": 20,   "J_wost": 150e6},
        "SPARC":   {"a": 0.57, "b": 0.18, "R0": 1.85, "σ": 1000e6, "T_op": 20.0,
                    "B_max": 20.0,  "n_TF": 1,   "sc": "REBCO", "config": "Bucking",
                    "κ": 1.75, "I_cond": 50e3, "V_max": 10e3, "N_sub": 9,
                    "tau_h": 20,   "J_wost": 150e6},
    }

    def _clean(val):
        """Return finite rounded float, else NaN."""
        if val is None:
            return np.nan
        if isinstance(val, tuple):
            val = val[0]
        val = np.real(val)
        return round(float(val), 2) if np.isfinite(val) else np.nan

    def _call_TF(model_label, p, cfg):
        """Dispatch TF thickness calculation to the appropriate model."""
        a, b, R0    = p["a"], p["b"], p["R0"]
        sigma       = p["σ"]
        J_wost      = p["J_wost"]
        B_max       = p["B_max"]
        conf        = p["config"]
        n_frac      = p.get("n_TF", 1)
        omega       = 0.5 if conf == "Wedging" else 1.0

        if model_label == "Academic":
            # f_TF_academic returns (c, c_WP, c_Nose, σ_z, σ_θ, σ_r, f_steel)
            return f_TF_academic(a, b, R0, sigma, J_wost, B_max,
                                 conf, cfg.coef_inboard_tension, cfg.F_CClamp)
        else:
            # f_TF_D0FUS returns same tuple layout
            return f_TF_D0FUS(a, b, R0, sigma, J_wost, B_max,
                              conf, omega, n_frac,
                              cfg.c_BP, cfg.coef_inboard_tension, cfg.F_CClamp)

    # Model definitions: (label, header colour)
    # Note: there is no separate f_TF_CIRCE yet; Academic / D0FUS are the two
    # available TF mechanical models.  A third "CIRCE" row is reserved for
    # future implementation and left blank (NaN) to keep the table layout
    # consistent with the CS benchmark.
    model_specs = [
        ("Academic", "#9C27B0"),
        ("D0FUS",    "#4CAF50"),
    ]

    import pandas as pd
    cols = ["Machine", "SC", "Config", "B_max [T]", "J [MA/m²]", "σ [MPa]", "c [m]"]

    for model_label, header_color in model_specs:
        rows = []
        for name, p in machines_TF.items():
            res = _call_TF(model_label, p, cfg)
            rows.append({
                "Machine":   name,
                "SC":        p["sc"],
                "Config":    p["config"],
                "B_max [T]": p["B_max"],
                "J [MA/m²]": round(p["J_wost"] / 1e6, 1),
                "σ [MPa]":   int(p["σ"] / 1e6),
                "c [m]":     _clean(res),
            })

        df = pd.DataFrame(rows, columns=cols)

        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.axis("off")
        tbl = ax.table(cellText=df.values, colLabels=df.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.6)
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor(header_color)
            tbl[(0, j)].set_text_props(weight="bold", color="white")
        for i in range(1, len(df) + 1):
            for j in range(len(cols)):
                if i % 2 == 0:
                    tbl[(i, j)].set_facecolor("#f0f0f0")
        plt.title(f"TF Coil Benchmark — {model_label} model",
                  fontsize=13, pad=18, weight="bold")
        plt.tight_layout()
        _save_or_show(fig, save_dir, f"TF_benchmark_{model_label.lower()}")


# =============================================================================
# 9. CS coil benchmark table
# =============================================================================

def plot_CS_benchmark_table(cfg=None, save_dir=None) -> None:
    """
    Display a CS coil benchmark table (Academic, D0FUS, CIRCE models) for
    ITER, EU-DEMO, JT-60SA, EAST, ARC, and SPARC.

    Parameters
    ----------
    cfg      : config object (DEFAULT_CONFIG if None).
    save_dir : str or None

    References
    ----------
    Shimada et al., Nucl. Fusion 47, S1 (2007) — ITER CS.
    Sarasola et al., IEEE Trans. Appl. Supercond. 33, 1-5 (2023) — EU-DEMO CS.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    machines = {
        "ITER":    {"Ψplateau": 230,  "a_cs": 2.00, "b_cs": 1.25, "c_cs": 0.90,
                    "R0_cs": 6.20, "B_TF": 11.8, "B_cs": 13,   "σ_CS": 330e6,
                    "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2,
                    "kappa": 1.7, "J_wost": 45e6},
        "EU-DEMO": {"Ψplateau": 600,  "a_cs": 2.92, "b_cs": 1.80, "c_cs": 1.19,
                    "R0_cs": 9.07, "B_TF": 13,   "B_cs": 13.5, "σ_CS": 330e6,
                    "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2,
                    "kappa": 1.65, "J_wost": 45e6},
        "JT60-SA": {"Ψplateau":  40,  "a_cs": 1.18, "b_cs": 0.27, "c_cs": 0.45,
                    "R0_cs": 2.96, "B_TF":  5.65, "B_cs":  8.9, "σ_CS": 330e6,
                    "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2,
                    "kappa": 1.87, "J_wost": 45e6},
        "EAST":    {"Ψplateau":  10,  "a_cs": 0.45, "b_cs": 0.4,  "c_cs": 0.25,
                    "R0_cs": 1.85, "B_TF":  7.2,  "B_cs":  4.7, "σ_CS": 330e6,
                    "config": "Wedging", "SupraChoice": "NbTi",  "T_CS": 4.2,
                    "kappa": 1.9,  "J_wost": 45e6},
        "ARC":     {"Ψplateau":  32,  "a_cs": 1.07, "b_cs": 0.89, "c_cs": 0.64,
                    "R0_cs": 3.30, "B_TF": 23,   "B_cs": 12.9, "σ_CS": 1000e6,
                    "config": "Plug",    "SupraChoice": "REBCO",  "T_CS": 20,
                    "kappa": 1.8,  "J_wost": 150e6},
        "SPARC":   {"Ψplateau":  42,  "a_cs": 0.57, "b_cs": 0.18, "c_cs": 0.35,
                    "R0_cs": 1.85, "B_TF": 20,   "B_cs": 25,   "σ_CS": 1000e6,
                    "config": "Bucking", "SupraChoice": "REBCO",  "T_CS": 20,
                    "kappa": 1.75, "J_wost": 150e6},
    }

    def _clean(val):
        """Return finite rounded float, else NaN."""
        if val is None:
            return np.nan
        val = np.real(val)
        return round(float(val), 2) if np.isfinite(val) else np.nan

    # Model definitions: (label, function, header colour)
    model_specs = [
        ("Academic", f_CS_ACAD,  "#9C27B0"),
        ("D0FUS",    f_CS_D0FUS, "#2196F3"),
        ("CIRCE",    f_CS_CIRCE, "#F44336"),
    ]

    import pandas as pd

    cols = ["Machine", "SC", "Config", "Ψ [Wb]", "σ [MPa]",
            "Width [m]", "B_CS [T]", "J [MA/m²]"]

    for model_label, model_func, header_color in model_specs:
        rows = []
        for name, p in machines.items():
            a, b, c, R0 = p["a_cs"], p["b_cs"], p["c_cs"], p["R0_cs"]
            psi   = p["Ψplateau"]
            sigma = p["σ_CS"]
            J_cs  = p["J_wost"]
            conf  = p["config"]
            kap   = p["kappa"]
            T_He  = p["T_CS"]
            B_TF  = p["B_TF"]
            # "Supra_Choice" key absent in dict → always "Manual" (consistent with original)
            Supra = p.get("Supra_Choice", "Manual")

            res = model_func(0, 0, psi, 0, a, b, c, R0, B_TF, 25, sigma,
                             Supra, J_cs, T_He, conf, kap, 6, 5, cfg)

            rows.append({
                "Machine":   name,
                "SC":        p["SupraChoice"],
                "Config":    conf,
                "Ψ [Wb]":    psi,
                "σ [MPa]":   int(sigma / 1e6),
                "Width [m]": _clean(res[0]),
                "B_CS [T]":  _clean(res[5]),
                "J [MA/m²]": round(J_cs / 1e6, 1),
            })

        df = pd.DataFrame(rows, columns=cols)

        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.axis("off")
        tbl = ax.table(cellText=df.values, colLabels=df.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.6)
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor(header_color)
            tbl[(0, j)].set_text_props(weight="bold", color="white")
        for i in range(1, len(df) + 1):
            for j in range(len(cols)):
                if i % 2 == 0:
                    tbl[(i, j)].set_facecolor("#f0f0f0")
        plt.title(f"CS Coil Benchmark — {model_label} model",
                  fontsize=13, pad=18, weight="bold")
        plt.tight_layout()
        _save_or_show(fig, save_dir, f"CS_benchmark_{model_label.lower()}")


# =============================================================================
# 9. Conductor cross-sections  (Bloc C — CICC geometry)
# =============================================================================
#
# Geometry (outside → inside):
#   1. Rectangular stainless-steel jacket  (aspect_ratio = h / w)
#   2. Circular insulation annulus (inscribed in steel, thin ring)
#   3. Circular cable space (He void background + strands)
#   4. Central circular He cooling pipe
#
# The absolute size is set by ``side`` (short side of the outer rectangle).
# All internal dimensions follow from area fractions at two levels:
#
#   Level 1 — total conductor rectangle (A_total = side^2 * AR):
#       f_steel + f_insulation + f_cable = 1    (f_cable derived)
#       Steel = rectangle minus circular hole of radius R_hole.
#       Insulation = annulus [R_cable, R_hole].
#       Cable = disc of radius R_cable.
#
#   Level 2 — cable-space circle:
#       f_He_pipe + f_SC + f_Cu + f_void = 1
#
# The conductor is described by a dict with the following keys:
#
#   name           : str    Display name (e.g. "ITER TF")
#   sc_type        : str    Superconductor material: 'NbTi', 'Nb3Sn', 'REBCO'
#   aspect_ratio   : float  h / w of the outer rectangle (>= 1 → portrait)
#   side           : float  Short side of the outer rectangle [mm]
#   f_steel        : float  Steel area / total rectangle area [-]
#   f_insulation   : float  Insulation annulus area / total rectangle area [-]
#   f_He_pipe      : float  He pipe area / cable-space area [-]
#   f_SC           : float  SC strand area / cable-space area [-]
#   f_Cu           : float  Cu strand area / cable-space area [-]
#   f_void         : float  Interstitial He void / cable-space area [-]
#   d_strand       : float  Individual strand diameter [mm]
#   Cu_nonCu       : float  Cu-to-nonCu ratio in SC strands [-]
#   I_op           : float  Operating current [A]  (optional, display only)
#   B_peak         : float  Peak field [T]         (optional, display only)
#   reference      : str    Literature citation (optional)
# =============================================================================

# ---------------------------------------------------------------------------
# Strand placement utility for circular cable space (annular region)
# ---------------------------------------------------------------------------

def _place_strands_circular(
    R_cable: float,
    d_strand: float, n_sc: int, n_cu: int,
    rng: np.random.Generator,
    exclude_radius: float = 0.0,
) -> tuple:
    """
    Place strands in a hexagonal packing within a circular cable space.

    Strands whose centres fall inside an exclusion circle of radius
    ``exclude_radius`` (central He cooling pipe) are rejected, as are
    strands extending beyond the cable-space boundary.

    Parameters
    ----------
    R_cable        : float  Radius of the circular cable space [mm].
    d_strand       : float  Strand diameter [mm].
    n_sc, n_cu     : int    Number of SC and Cu strands to place.
    rng            : Generator  Numpy random generator.
    exclude_radius : float  Radius of central exclusion zone [mm].

    Returns
    -------
    xs, ys   : ndarray  Strand centre coordinates [mm].
    is_sc    : ndarray  Boolean mask — True = SC strand.
    """
    r_s = d_strand / 2.0
    dx  = d_strand * 1.05                       # Horizontal lattice spacing
    dy  = d_strand * np.sqrt(3) / 2.0 * 1.05   # Vertical lattice spacing

    # Build hexagonal grid covering the bounding box, then clip to annulus
    candidates_x, candidates_y = [], []
    n_rows = int(2 * R_cable / dy) + 2
    n_cols = int(2 * R_cable / dx) + 2
    for row in range(-n_rows, n_rows + 1):
        for col in range(-n_cols, n_cols + 1):
            x = col * dx + (row % 2) * dx / 2.0
            y = row * dy
            r = np.sqrt(x**2 + y**2)
            if (r + r_s < R_cable
                    and r > exclude_radius + r_s):
                candidates_x.append(x)
                candidates_y.append(y)

    xs = np.array(candidates_x)
    ys = np.array(candidates_y)
    n_total = n_sc + n_cu

    if len(xs) == 0:
        return xs, ys, np.array([], dtype=bool)

    if len(xs) > n_total:
        idx = rng.choice(len(xs), size=n_total, replace=False)
        xs, ys = xs[idx], ys[idx]

    is_sc = np.zeros(len(xs), dtype=bool)
    sc_idx = rng.choice(len(xs), size=min(n_sc, len(xs)), replace=False)
    is_sc[sc_idx] = True
    return xs, ys, is_sc


# ---------------------------------------------------------------------------
# C1 — Single CICC conductor cross-section
# ---------------------------------------------------------------------------

# Color palette for conductor components
_CICC_COLORS = {
    "jacket_steel": "#7C8EA0",   # Stainless steel jacket
    "insulation":   "#E8D8B8",   # Insulation layer (glass-epoxy)
    "void_he":      "#C8E8F8",   # He void / coolant background
    "he_channel":   "#FFFFFF",   # Central He cooling channel (white)
    "strand_NbTi":  "#3A7FBD",   # NbTi SC strand
    "strand_Nb3Sn": "#2E8B57",   # Nb3Sn SC strand
    "strand_REBCO": "#CC3333",   # REBCO tape
    "strand_cu":    "#D4A017",   # Pure copper stabiliser strand
}

_SC_COLOR = {
    "NbTi":  _CICC_COLORS["strand_NbTi"],
    "Nb3Sn": _CICC_COLORS["strand_Nb3Sn"],
    "REBCO": _CICC_COLORS["strand_REBCO"],
}

# Default conductors — generic TF and CS, Nb3Sn CICC equivalent model
# Calibration visual: 100 mm side, 1/3-split cable fractions.
#
# Level 1:  f_steel + f_insulation + f_cable = 1   (f_cable derived)
# Level 2:  f_He_pipe + f_SC + f_Cu + f_void  = 1   (cable-space)


def _conductor_aspect_ratio(n: float, f_steel: float) -> float:
    """
    Compute the outer jacket aspect ratio h/w from the steel asymmetry
    parameter n = δ_S1/δ_S2 and the total steel fraction.

    Conductor geometry (see Surface_dilution_conductor figure):
        - Circular cable space of radius r_c inscribed in a rectangular
          steel jacket.
        - δ_S1 = radial steel thickness,  δ_S2 = toroidal steel thickness.
        - n = δ_S1 / δ_S2  (1 = square,  0 = no radial steel).
        - w = 2(r_c + δ_S2),  h = 2(r_c + n·δ_S2).

    From f_cable = π r_c² / (w·h) = 1 − f_steel, defining x = δ_S2/r_c:

        n x² + (1+n) x + (1 − C) = 0     with C = π / [4 (1−f_steel)]

    The positive root gives x, then AR = (1 + n x) / (1 + x).

    Parameters
    ----------
    n       : float  Steel asymmetry parameter [-] (0 ≤ n ≤ 1).
    f_steel : float  Total steel fraction [-] (0 < f_steel < 1).

    Returns
    -------
    float  Jacket aspect ratio h/w  (1.0 for n = 1, < 1 for n < 1).
    """
    if f_steel <= 0.0 or f_steel >= 1.0:
        return 1.0
    C = np.pi / (4.0 * (1.0 - f_steel))
    if C <= 1.0:
        # Cable circle cannot fit — degenerate case
        return 1.0

    if n < 1e-6:
        # n → 0 limit: linear equation  (1 + x) = C  →  x = C − 1
        x = C - 1.0
        return 1.0 / (1.0 + x)      # AR = 1 / (1 + x) since n·x ≈ 0

    # Quadratic: n x² + (1+n) x + (1−C) = 0
    a_coeff = n
    b_coeff = 1.0 + n
    c_coeff = 1.0 - C
    disc = b_coeff**2 - 4.0 * a_coeff * c_coeff
    if disc < 0:
        return 1.0
    x = (-b_coeff + np.sqrt(disc)) / (2.0 * a_coeff)
    if x < 0:
        return 1.0
    return (1.0 + n * x) / (1.0 + x)


def build_conductor_from_run(run: dict, coil: str = "TF") -> dict:
    """
    Build a CICC conductor dict from a D0FUS run output dict.

    Converts wost-level fractions (from calculate_cable_current_density) into
    the two-level format expected by plot_CICC_cross_section:
        Level 1 (total conductor): f_steel, f_insulation, f_cable
        Level 2 (cable-space):     f_He_pipe, f_SC, f_Cu, f_void

    Conductor geometry adapts to the run:
        aspect_ratio : derived from n (steel asymmetry δ_S1/δ_S2) and
                       f_steel via the inscribed-circle jacket model.
                       n = 1 → square;  n < 1 → wider than tall.
        side         : fixed at 50 mm for display (toroidal width)

    The visual SC/Cu strand ratio is determined by f_SC and f_Cu
    cable-space fractions (Cu_nonCu = 0, i.e. each strand is rendered
    as either pure SC or pure Cu).

    Parameters
    ----------
    run  : dict  D0FUS run output dict (from _build_run_dict).
    coil : str   'TF' or 'CS'.

    Returns
    -------
    dict  Conductor description for plot_CICC_cross_section.
          Returns the static fallback (_CONDUCTOR_TF) if fractions are unavailable.
    """
    f_steel = run.get(f"Steel_fraction_{coil}", np.nan)
    f_sc    = run.get(f"f_sc_{coil}", np.nan)
    f_cu    = run.get(f"f_cu_{coil}", np.nan)
    f_pipe  = run.get(f"f_He_pipe_{coil}", np.nan)
    f_void  = run.get(f"f_void_{coil}", np.nan)
    f_In    = run.get(f"f_In_{coil}", np.nan)
    sc_type = run.get("Supra_choice", "Nb3Sn")

    # Steel asymmetry parameter: n = δ_S1/δ_S2 (1 = square, 0 = optimal)
    n_cond = float(run.get(f"n_{coil}", 1.0))

    # Guard: fall back to static dict if any fraction is NaN or unphysical
    fallback = _CONDUCTOR_TF if coil == "TF" else _CONDUCTOR_CS
    fracs = [f_steel, f_sc, f_cu, f_pipe, f_void, f_In]
    if not all(np.isfinite(fracs)):
        return fallback
    if not all(0.0 <= f <= 1.0 for f in fracs):
        return fallback
    wost_sum = f_sc + f_cu + f_pipe + f_void + f_In
    if abs(wost_sum - 1.0) > 0.05:
        return fallback

    # ── Jacket aspect ratio from δ_S1/δ_S2 geometry ──
    aspect_ratio = _conductor_aspect_ratio(n_cond, f_steel)

    # ── Level 1: convert wost fractions to total-conductor fractions ──
    # wost fraction of total = (1 - f_steel)
    wost_frac = 1.0 - f_steel
    f_insulation_total = f_In * wost_frac   # insulation as fraction of total
    # f_cable_total = wost_frac - f_insulation_total  (derived)

    # ── Level 2: renormalise wost fractions to cable-space ──
    # cable-space = wost minus insulation → fraction of wost = (1 - f_In)
    f_cable_wost = 1.0 - f_In
    if f_cable_wost < 1e-6:
        return fallback

    f_SC_cable      = f_sc   / f_cable_wost
    f_Cu_cable      = f_cu   / f_cable_wost
    f_He_pipe_cable = f_pipe / f_cable_wost
    f_void_cable    = f_void / f_cable_wost

    # Cu_nonCu = 0: each strand is rendered as either pure SC or pure Cu.
    # The correct visual ratio is already ensured by f_SC_cable / f_Cu_cable
    # which determine the number of green vs gold strands placed.

    return {
        "name":          f"{coil} coil",
        "sc_type":       sc_type,
        "aspect_ratio":  aspect_ratio,
        "side":          50.0,           # Toroidal width for display [mm]
        "f_steel":       f_steel,
        "f_insulation":  f_insulation_total,
        "f_He_pipe":     f_He_pipe_cable,
        "f_SC":          f_SC_cable,
        "f_Cu":          f_Cu_cable,
        "f_void":        f_void_cable,
        "d_strand":      1.2,            # Cosmetic strand diameter [mm]
        "Cu_nonCu":      0.0,
    }


_CONDUCTOR_TF = {
    "name":          "TF coil",
    "sc_type":       "Nb3Sn",
    "aspect_ratio":  1.0,            # Square outer jacket (n_TF = 1)
    "side":          50.0,           # 50 mm display size
    "f_steel":       0.50,           # 50 % steel
    "f_insulation":  0.03,           # 3 % insulation (thin ring)
    # --- cable-space sub-fractions (sum = 1) ---
    "f_He_pipe":     0.05,           # 5 % He pipe (central spiral)
    "f_SC":          0.317,          # ~1/3 of remaining 95 %
    "f_Cu":          0.317,          # ~1/3
    "f_void":        0.317,          # ~1/3 interstitial He void (LTS)
    "d_strand":      1.2,            # Strand diameter [mm] (cosmetic)
    "Cu_nonCu":      0.0,            # Pure strand rendering (SC vs Cu by count)
}

_CONDUCTOR_CS = {
    "name":          "CS coil",
    "sc_type":       "Nb3Sn",
    "aspect_ratio":  1.0,            # Square jacket (n_CS = 1)
    "side":          50.0,           # 50 mm display size
    "f_steel":       0.50,           # 50 % steel
    "f_insulation":  0.03,           # 3 % insulation (thin ring)
    # --- cable-space sub-fractions (sum = 1) ---
    "f_He_pipe":     0.05,           # 5 % He pipe (central spiral)
    "f_SC":          0.317,          # ~1/3 of remaining 95 %
    "f_Cu":          0.317,          # ~1/3
    "f_void":        0.317,          # ~1/3 interstitial He void (LTS)
    "d_strand":      1.2,            # Strand diameter [mm] (cosmetic)
    "Cu_nonCu":      0.0,            # Pure strand rendering (SC vs Cu by count)
}


def plot_CICC_cross_section(
    conductor: dict,
    ax=None,
    seed: int = 42,
    save_dir: str | None = None,
) -> None:
    """
    Draw a schematic cross-section of a CICC conductor.

    From outside to inside:
      1. Rectangular stainless-steel jacket  (aspect_ratio = h / w)
      2. Circular insulation annulus inscribed in the steel rectangle
      3. Circular cable space (He void + strands in hexagonal packing)
      4. Central circular He cooling pipe

    All geometry is derived from ``side`` (short side of the outer
    rectangle) and two sets of area fractions:

    Level 1 — total conductor rectangle:
        f_steel + f_insulation + f_cable = 1   (f_cable derived)
        R_hole  = sqrt((1 - f_steel) * A_total / pi)
        R_cable = sqrt((1 - f_steel - f_insulation) * A_total / pi)
        t_ins   = R_hole - R_cable

    Level 2 — cable-space circle:
        f_He_pipe + f_SC + f_Cu + f_void = 1
        R_He = sqrt(f_He_pipe * A_cable / pi)

    Parameters
    ----------
    conductor : dict   Conductor description (see module-level docstring).
    ax        : matplotlib Axes or None
    seed      : int    Random seed for strand placement reproducibility.
    save_dir  : str or None  (only used when ``ax`` is None).
    """
    from matplotlib.patches import Circle, Rectangle, Patch

    rng        = np.random.default_rng(seed)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = ax.get_figure()

    # SC strand color: adapts to superconductor type if specified
    sc_type = conductor.get("sc_type", None)
    col_sc  = _SC_COLOR.get(sc_type, "#2E8B57") if sc_type else "#2E8B57"

    # ---- Geometry derivation ------------------------------------------------
    AR      = conductor["aspect_ratio"]
    side    = conductor["side"]                            # Short side [mm]
    f_steel = conductor["f_steel"]
    f_ins   = conductor.get("f_insulation", 0.03)
    f_cable = 1.0 - f_steel - f_ins                        # Derived

    # Outer rectangle
    w_j = side                                             # Width [mm]
    h_j = side * AR                                        # Height [mm]
    A_total = w_j * h_j                                    # Total area [mm2]

    # Circular hole in steel: area = (1 - f_steel) * A_total
    R_hole  = np.sqrt((1.0 - f_steel) * A_total / np.pi)  # Outer insulation radius
    R_cable = np.sqrt(f_cable * A_total / np.pi)           # Inner insulation / cable radius
    t_ins   = R_hole - R_cable                             # Insulation thickness [mm]

    # Clamp to inscribed circle if needed
    R_max = min(w_j, h_j) / 2.0
    if R_hole > R_max:
        # Scale both radii proportionally
        scale = R_max / R_hole
        R_hole  *= scale
        R_cable *= scale
        t_ins = R_hole - R_cable

    A_cable = np.pi * R_cable**2                           # Effective cable area [mm2]

    # He pipe
    f_He_pipe = conductor.get("f_He_pipe", 0.0)
    R_He = np.sqrt(f_He_pipe * A_cable / np.pi) if f_He_pipe > 0 else 0.0

    # ---- Layer 1: Steel rectangle (outermost) ------------------------------
    ax.add_patch(Rectangle((-w_j / 2, -h_j / 2), w_j, h_j,
                            fc=_CICC_COLORS["jacket_steel"], ec="k", lw=1.5,
                            zorder=1))

    # ---- Layer 2: Insulation circle (between steel and cable) --------------
    ax.add_patch(Circle((0, 0), R_hole,
                         fc=_CICC_COLORS["insulation"], ec="k", lw=0.8,
                         zorder=2))

    # ---- Layer 3: Cable space (circular, He void background) ---------------
    ax.add_patch(Circle((0, 0), R_cable,
                         fc=_CICC_COLORS["void_he"], ec="k", lw=0.6,
                         zorder=3))

    # ---- Layer 4: Central He cooling pipe ----------------------------------
    if R_He > 0:
        ax.add_patch(Circle((0, 0), R_He,
                             fc=_CICC_COLORS["he_channel"], ec="k", lw=0.8,
                             zorder=5))

    # ---- Strand placement (circular packing) --------------------------------
    d_s      = conductor["d_strand"]
    f_SC     = conductor["f_SC"]
    f_Cu     = conductor["f_Cu"]
    Cu_nonCu = conductor.get("Cu_nonCu", 1.0)

    A_strand = np.pi * (d_s / 2.0) ** 2
    A_sc_eff = f_SC * A_cable
    A_cu_eff = f_Cu * A_cable
    n_sc = max(1, int(round(A_sc_eff * (1.0 + Cu_nonCu) / A_strand)))
    n_cu = max(0, int(round((A_cu_eff - A_sc_eff * Cu_nonCu) / A_strand)))
    n_cu = max(n_cu, 0)

    xs, ys, is_sc = _place_strands_circular(
        R_cable, d_s, n_sc, n_cu, rng, exclude_radius=R_He)

    r_s = d_s / 2.0
    for x, y, sc in zip(xs, ys, is_sc):
        col = col_sc if sc else _CICC_COLORS["strand_cu"]
        ax.add_patch(Circle((x, y), r_s, fc=col, ec="none",
                             alpha=0.9, zorder=4))

    # ---- Dimension lines ----------------------------------------------------
    y_dim = -h_j / 2 - 3.0
    ax.annotate("", xy=(w_j / 2, y_dim), xytext=(-w_j / 2, y_dim),
                arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0))
    ax.text(0, y_dim - 1.5, f"{w_j:.1f} mm",
            ha="center", va="top", fontsize=8, color="dimgray")
    x_dim = -w_j / 2 - 3.0
    ax.annotate("", xy=(x_dim, h_j / 2), xytext=(x_dim, -h_j / 2),
                arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0))
    ax.text(x_dim - 2.5, 0, f"{h_j:.1f} mm",
            ha="right", va="center", fontsize=8, color="dimgray", rotation=90)

    # ---- Legend (color key only) ---------------------------------------------
    sc_label = f"SC ({sc_type})" if sc_type else "SC"
    legend_lines = [
        (_CICC_COLORS["jacket_steel"],   "Steel"),
        (_CICC_COLORS["insulation"],     "Insulation"),
        (col_sc,                         sc_label),
        (_CICC_COLORS["strand_cu"],      "Cu"),
        (_CICC_COLORS["he_channel"],     "He pipe"),
        (_CICC_COLORS["void_he"],        "He void"),
    ]
    handles = [Patch(fc=c, ec="k", lw=0.5, label=l) for c, l in legend_lines]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              framealpha=0.92, edgecolor="gray")

    lim = max(w_j, h_j) / 2 * 1.35
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 1.10, lim)
    ax.set_xlabel("x  [mm]", fontsize=10)
    ax.set_ylabel("y  [mm]", fontsize=10)

    # Retrieve void fraction for display
    f_void_val = conductor.get("f_void", 1.0 - f_He_pipe - f_SC - f_Cu)

    ax.set_title(
        f"{conductor['name']}  —  CICC equivalent model"
        + (f"  ({sc_type})" if sc_type else "") +
        f"\nSteel={f_steel*100:.0f}%   SC={f_SC*100:.1f}%   Cu={f_Cu*100:.1f}%   "
        f"He pipe={f_He_pipe*100:.1f}%   He void={f_void_val*100:.1f}%",
        fontsize=9, fontweight="bold",
    )
    ax.grid(False)

    if standalone:
        fname = "cicc_" + conductor["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        _save_or_show(fig, save_dir, fname)

# =============================================================================
# Stand-alone execution — smoke test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="D0FUS_figures.py — stand-alone smoke test.\n"
                    "Renders all figures interactively by default.\n"
                    "Pass --save-dir to write PNG files instead.")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save PNG figures (suppresses interactive display)")
    args = parser.parse_args()

    _out = args.save_dir
    if _out is not None:
        os.makedirs(_out, exist_ok=True)

    # ITER Q=10 reference case — Shimada et al., Nucl. Fusion 47 (2007) S1
    ITER_RUN = {
        # Plasma geometry
        "R0":               6.2,
        "a":                2.0,
        "Plasma_geometry": "D0FUS",
        "kappa_edge":       1.85,
        "delta_edge":       0.33,
        "Vprime_data":      None,
        # On-axis field and current
        "B0":          5.3,
        "B_max":       11.8,
        "Ip":          15.0,
        # Kinetics
        "nbar":        1.01,
        "Tbar":        8.9,
        "nu_n":        0.5,
        "nu_T":        1.0,
        "rho_ped":     0.94,
        "n_ped_frac":  0.80,
        "T_ped_frac":  0.35,
        # MHD
        "q95":         3.0,
        "Z_eff":       1.7,
        # Power balance
        "P_fus":       500.0,
        "P_aux":       50.0,
        "Q":           10.0,
        # Impurities
        "f_W":         5e-5,
        "f_Ne":        3e-3,
        "f_Ar":        None,
        # Radial build
        #   c_TF includes full inboard leg (WP + structural case)
        #   MHI/QST: ITER TF coil ≈ 16.5 m high (with casing), 9 m wide, 300 t
        "b":           1.42,
        "c_TF":        0.87,
        "c_WP":        0.36,
        "c_nose":      0.20,
        "c_CS":        0.70,
        "N_TF":        18,
        "Gap":         0.02,
        "e_fw":        0.05,
        "e_blanket":   0.45,
        "e_shield":    0.50,
    }

    plot_all(ITER_RUN, save_dir=_out)