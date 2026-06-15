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


import os   # needed before D0FUS_import for sys.path resolution in standalone mode

# Package-relative imports (production mode)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_physical_functions import (
        f_Kappa, f_Kappa_95, f_Delta_95,
        kappa_profile, delta_profile, miller_RZ,
        f_first_wall_surface,
        f_nprof, f_Tprof,
        f_nbar_line,
        get_Lz,
        f_He_fraction,
        f_q_profile,
        f_sigmav,
    )
    from .D0FUS_radial_build_functions import (
        J_non_Cu_NbTi, J_non_Cu_Nb3Sn, J_non_Cu_REBCO,
        calculate_cable_current_density,
        eta_old, eta_spitzer, eta_sauter, eta_redl,
        f_TF_academic, f_TF_refined,
        Winding_Pack_refined, gamma_func, _last_graded_profile,
        f_CS_ACAD, f_CS_refined, f_CS_CIRCE,
        F_CIRCE0D, compute_von_mises_stress,
        calculate_E_mag_TF,
        f_TF_cross_section,
        _sol_fw_miller_contours,
        _offset_contour,
        _radial_interp_contour,
        f_TBR,
    )
    from .D0FUS_parameterization import DEFAULT_CONFIG, E_ELEM, BLANKET_CONCEPTS

# Standalone-execution imports (development / testing)
else:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_physical_functions import (
        f_Kappa, f_Kappa_95, f_Delta_95,
        kappa_profile, delta_profile, miller_RZ,
        f_first_wall_surface,
        f_nprof, f_Tprof,
        f_nbar_line,
        get_Lz,
        f_He_fraction,
        f_q_profile,
        f_sigmav,
    )
    from D0FUS_BIB.D0FUS_radial_build_functions import (
        J_non_Cu_NbTi, J_non_Cu_Nb3Sn, J_non_Cu_REBCO,
        calculate_cable_current_density,
        eta_old, eta_spitzer, eta_sauter, eta_redl,
        f_TF_academic, f_TF_refined,
        Winding_Pack_refined, gamma_func, _last_graded_profile,
        f_CS_ACAD, f_CS_refined, f_CS_CIRCE,
        F_CIRCE0D, compute_von_mises_stress,
        calculate_E_mag_TF,
        f_TF_cross_section,
        _sol_fw_miller_contours,
        _offset_contour,
        _radial_interp_contour,
        f_TBR,
    )
    from D0FUS_BIB.D0FUS_parameterization import DEFAULT_CONFIG, E_ELEM, BLANKET_CONCEPTS


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

    Both the 'Academic' (constant-κ, δ=0) and the 'Refined PCHIP'
    profiles are shown.  For δ, both positive and negative triangularity
    cases are superimposed.

    Parameters
    ----------
    kappa_edge : float  Edge elongation [-].
    delta_edge : float  Edge triangularity (positive D-shape value) [-].
    rho_95     : float  Normalised flux surface at 95 % of poloidal flux [-].
    save_dir   : str or None

    References
    ----------
    Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980) — PCHIP scheme.
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
            label="Refined: flat core at κ₉₅, edge rise (PCHIP)")
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
            label=f"Refined PCHIP: δ_edge = +{delta_edge} (D-shape)")
    ax.plot(rho, delta_profile(rho, delta_edge_neg, delta_95_neg),
            color="tab:purple", lw=2,
            label=f"Refined PCHIP: δ_edge = {delta_edge_neg} (neg. triang.)")
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
      1. Refined PCHIP — positive triangularity
      2. Academic     — circular (κ const, δ = 0)
      3. Refined PCHIP — negative triangularity

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
        ("refined",
         f"Refined PCHIP — positive δ\n"
         f"(κ₉₅ = {kappa_95:.3f}, δ_edge = +{delta_edge}, δ₉₅ = +{delta_95:.3f})",
         kappa_edge, delta_edge, kappa_95, delta_95),
        ("Academic",
         f"Academic  (κ = {kappa_edge} const, δ = 0)",
         kappa_edge, 0.0, kappa_95, 0.0),
        ("refined",
         f"Refined PCHIP — negative δ\n"
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
        f"Miller flux surfaces — Academic | Refined (+δ) | Refined (−δ)\n"
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
    and refined (Miller LCFS numerical) models.

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
    S_d0_k = [f_first_wall_surface(R0, a, k, delta_fix, "refined")
              for k in kappa_scan]
    S_ac_d = [f_first_wall_surface(R0, a, kappa_fix, d, "Academic")
              for d in delta_scan]
    S_d0_d = [f_first_wall_surface(R0, a, kappa_fix, d, "refined")
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
        ax.plot(xarr, S_d0, "r--", lw=2, label="Refined (Miller LCFS)")
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

def plot_DT_reactivity(
    T_min_keV: float = 1.0,
    T_max_keV: float = 100.0,
    n_points: int = 600,
    T_op_min: float = 10.0,
    T_op_max: float = 25.0,
    save_dir: str | None = None,
) -> None:
    """
    Plot the D-T Maxwellian reactivity ⟨σv⟩(T) used by D0FUS, together with
    the pressure-limited fusion power density metric ⟨σv⟩/T².

    Left panel:
        ⟨σv⟩(T) computed from the Bosch & Hale (1992) parameterisation,
        with the power plant operating window [T_op_min, T_op_max] shaded
        and the reactivity maximum marked.

    Right panel:
        Pressure-limited figure of merit ⟨σv⟩/T² (arbitrary units).
        At fixed plasma pressure p = nT, the fuel ion density scales as
        n ∝ 1/T, so the volumetric fusion power density p_fus ∝ n² ⟨σv⟩
        is proportional to ⟨σv⟩/T². This identifies the optimal operating
        temperature for a β-limited tokamak near 14 keV.

    Parameters
    ----------
    T_min_keV, T_max_keV : float  Ion temperature scan range [keV].
    n_points             : int    Number of temperature grid points.
    T_op_min, T_op_max   : float  Power plant operating window bounds [keV].
    save_dir             : str or None

    References
    ----------
    Bosch & Hale, Nucl. Fusion 32, 611 (1992) — DT reactivity fit (Table IV).
    Freidberg, Plasma Physics and Fusion Energy (2007) — pressure-limited optimum.
    """
    # Temperature grid (linear, since the operating window lies in the rapid-rise zone)
    T_arr  = np.linspace(T_min_keV, T_max_keV, n_points)
    sv_arr = f_sigmav(T_arr)

    # Reactivity maximum
    i_peak           = int(np.argmax(sv_arr))
    T_peak, sv_peak  = T_arr[i_peak], sv_arr[i_peak]

    # Pressure-limited metric ⟨σv⟩/T² (units arbitrary, normalised below)
    metric           = sv_arr / T_arr**2
    i_opt            = int(np.argmax(metric))
    T_opt, m_opt     = T_arr[i_opt], metric[i_opt]

    # --- Figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: reactivity curve on log scale
    ax = axes[0]
    ax.semilogy(T_arr, sv_arr, color="tab:red", lw=2.0,
                label="Bosch & Hale (1992)")
    ax.axvspan(T_op_min, T_op_max, color="goldenrod", alpha=0.20,
               label=f"Operating window\n{T_op_min:.0f}–{T_op_max:.0f} keV")
    ax.axvline(T_peak, color="k", lw=1.0, ls="--",
               label=f"peak: T = {T_peak:.0f} keV")
    ax.set_xlabel(r"Ion temperature $T$  [keV]", fontsize=11)
    ax.set_ylabel(r"$\langle\sigma v\rangle_{DT}$  [m$^3$ s$^{-1}$]", fontsize=11)
    ax.set_title("D-T Maxwellian reactivity", fontsize=11)
    ax.set_xlim(T_min_keV, T_max_keV)
    ax.set_ylim(1e-25, 3e-21)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

    # Right panel: pressure-limited figure of merit
    ax = axes[1]
    ax.plot(T_arr, metric / m_opt, color="tab:blue", lw=2.0,
            label=r"$\langle\sigma v\rangle / T^2$  (normalised)")
    ax.axvspan(T_op_min, T_op_max, color="goldenrod", alpha=0.20,
               label=f"Operating window\n{T_op_min:.0f}–{T_op_max:.0f} keV")
    ax.axvline(T_opt, color="k", lw=1.0, ls="--",
               label=f"optimum: T = {T_opt:.1f} keV")
    ax.set_xlabel(r"Ion temperature $T$  [keV]", fontsize=11)
    ax.set_ylabel(r"$\langle\sigma v\rangle / T^2$  [normalised]", fontsize=11)
    ax.set_title(r"Pressure-limited fusion power figure of merit", fontsize=11)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

    plt.suptitle("Fusion reactivity and operating temperature window",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "DT_reactivity")


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
    C_Alpha_ITER: float = 5.0,
    C_Alpha_DEMO: float = 7.0,
    nu_T: float = 1.0,
    save_dir: str | None = None,
) -> None:
    """
    Plot helium ash fraction f_α as a function of the He removal efficiency
    C_α = τ_α / τ_E for ITER and EU-DEMO reference parameters, comparing
    academic (no pedestal) and H-mode pedestal profile assumptions.

    Two D0FUS default values for C_α are highlighted with vertical dotted
    lines: C_α = 5 for ITER (consistent with Progress in the ITER Physics
    Basis projections) and C_α = 7 for EU-DEMO 2017 (PROCESS reference run).

    Parameters
    ----------
    nbar, Tbar, tauE : float  Reference plasma parameters [10²⁰ m⁻³, keV, s].
    C_Alpha_ITER     : float  D0FUS default C_α for ITER (vertical line) [-].
    C_Alpha_DEMO     : float  D0FUS default C_α for EU-DEMO 2017 (vertical line) [-].
    nu_T             : float  Temperature peaking exponent [-].
    save_dir         : str or None

    References
    ----------
    ITER Physics Basis, Nucl. Fusion 39, §2.4 (1999).
    Shimada et al., Progress in the ITER Physics Basis, Ch. 1,
        Nucl. Fusion 47, S1 (2007).
    Kovari et al., Fus. Eng. Des. 89, 3054 (2014) — PROCESS systems code.
    Reiter, Wolf and Kever, Nucl. Fusion 30, 2141 (1990) — ignition bound on C_α.
    """
    C_arr = np.linspace(2, 15, 150)

    fa_ITER     = [f_He_fraction(1.0,  8.9,  3.7, C, nu_T) * 100 for C in C_arr]
    fa_ITER_ped = [f_He_fraction(1.0,  8.9,  3.7, C, nu_T,
                                 rho_ped=0.9, T_ped_frac=0.25) * 100
                   for C in C_arr]
    fa_DEMO     = [f_He_fraction(1.2, 12.5,  4.6, C, nu_T) * 100 for C in C_arr]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(C_arr, fa_ITER,     "b-",  lw=2.0, label="ITER — academic (no pedestal)")
    ax.plot(C_arr, fa_ITER_ped, "b--", lw=1.6, label="ITER — Refined H-mode pedestal")
    ax.plot(C_arr, fa_DEMO,     "r-",  lw=2.0, label="EU-DEMO — academic")
    # Two D0FUS default operating points: ITER and EU-DEMO 2017
    ax.axvline(C_Alpha_ITER, color="tab:blue", lw=1.6, ls=":",
               label=f"ITER default $C_\\alpha$ = {C_Alpha_ITER:.0f}")
    ax.axvline(C_Alpha_DEMO, color="tab:red",  lw=1.6, ls=":",
               label=f"EU-DEMO 2017 default $C_\\alpha$ = {C_Alpha_DEMO:.0f}")
    ax.axhspan(4, 6, color="grey", alpha=0.12, label="ITER target 4–6 %")
    ax.set_xlabel(r"Removal efficiency $C_\alpha = \tau_\alpha / \tau_E$", fontsize=14)
    ax.set_ylabel(r"Helium ash fraction $f_\alpha$  [%]", fontsize=14)
    ax.set_title("He ash fraction — academic vs refined H-mode pedestal",
                 fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(2, 15)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "He_fraction")


# =============================================================================
# 4. Superconductor / cable engineering
# =============================================================================

# -----------------------------------------------------------------------------
# NHMFL engineering current density reference data (T = 4.2 K)
# -----------------------------------------------------------------------------
# Source: National High Magnetic Field Laboratory (NHMFL / MagLab) "Engineering
# Current Density Plot", file Je_vs_B-041118a (updated 2021-01-04).  Values are
# whole-strand / whole-tape engineering current density Je [A/mm²] vs applied
# field B [T].  Only the three series directly comparable to the D0FUS strand
# scalings (NbTi, Nb3Sn bronze, REBCO worst-orientation) are kept here.
#
# Per-series provenance:
#   * REBCO B perp tape plane : SuperPower SP26, 50 µm substrate, 7.5%Zr,
#     measured at NHMFL (Braccini, Jaroszynski, Xu) - DOI 10.1088/0953-2048/24/3/035001
#   * Nb3Sn High Sn Bronze    : Miyazaki et al., MT-18 (IEEE TASC 14:2, 2004)
#     DOI 10.1109/TASC.2004.830344
#   * NbTi LHC 4.2 K          : Boutboul et al., MT-19 (IEEE TASC 16:2, 2006)
#     DOI 10.1109/TASC.2006.870777
_NHMFL_JE_REFERENCE = {
    "NbTi": {
        "label": "NbTi LHC strand",
        "B": np.array([0.61, 0.95, 1.34, 1.68, 2.23, 3.18, 4.15, 5.12, 6.10]),
        "Je": np.array([5106.88, 4054.15, 3180.60, 2710.23, 2217.46,
                        1724.69, 1411.11, 1187.12, 918.34]),
    },
    "Nb3Sn": {
        "label": "Nb₃Sn high-Sn bronze",
        "B": np.array([18.00, 19.01, 20.01, 21.01, 22.02, 23.00, 24.00, 25.00]),
        "Je": np.array([166.64, 137.81, 111.10, 84.38, 60.82, 40.08, 19.34, 8.09]),
    },
    "REBCO": {
        "label": "REBCO tape, B⊥ (worst case)",
        "B": np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                       10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0,
                       20.0, 22.0, 24.0, 25.0, 26.0, 28.0, 30.0, 31.0]),
        "Je": np.array([3665.00, 2920.00, 2380.00, 1796.15, 1447.69, 1188.46,
                        1034.37,  956.92,  851.83,  780.84,  720.00,  679.44,
                         626.09,  593.24,  563.48,  535.38,  516.52,  469.56,
                         433.08,  406.96,  391.30,  367.69,  360.00,  344.35,
                         328.70,  322.31]),
    },
}


def _overlay_nhmfl_reference(ax, color_map: dict, alpha: float = 0.30) -> None:
    """
    Overlay the NHMFL engineering current density reference (4.2 K) on ``ax``
    as transparent markers connected by a thin line, color-matched per material.

    Parameters
    ----------
    ax        : matplotlib Axes  Target axes (assumed log-y in A/mm²).
    color_map : dict             Mapping {material_key: hex_color} for NbTi,
                                 Nb3Sn, REBCO; the same colours used for the
                                 D0FUS curves so the visual pairing is direct.
    alpha     : float            Transparency for both markers and connecting
                                 line (default 0.30, "background" appearance).
    """
    for key, ref in _NHMFL_JE_REFERENCE.items():
        col = color_map.get(key, "#808080")
        ax.plot(
            ref["B"], ref["Je"],
            linestyle="-", linewidth=1.0,
            marker="o", markersize=4.5,
            color=col, alpha=alpha,
            zorder=1,
        )
    # Single neutral legend handle for the whole NHMFL reference set.
    ax.plot([], [], linestyle="-", linewidth=1.0, marker="o", markersize=4.5,
            color="#555555", alpha=alpha,
            label="NHMFL strand/tape data @ 4.2 K (2011 vintage, ref.)")


def plot_Jc_scaling(
    B_min: float = 0.5,
    B_max: float = 45.0,
    T_op: float = 4.2,
    f_non_Cu_LTS: float = 0.50,
    f_non_Cu_HTS: float = 0.60,
    show_nhmfl_ref: bool = True,
    save_dir: str | None = None,
) -> None:
    """
    Plot strand/tape engineering current density J_eng vs magnetic field for
    the three main superconductor families used in fusion magnets:
      * NbTi   (LTS, ITER TF reference)
      * Nb₃Sn  (LTS, ITER/EU-DEMO TF/CS)
      * REBCO  (HTS, ARC / SPARC)

    When ``show_nhmfl_ref`` is True, the NHMFL/MagLab experimental engineering
    current density data at 4.2 K is overlaid as a transparent background
    reference (color-matched markers + thin line per material).

    Parameters
    ----------
    B_min, B_max   : float  Field scan range [T].
    T_op           : float  Operating temperature [K].
    f_non_Cu_LTS   : float  Non-copper fraction for LTS strands [-].
    f_non_Cu_HTS   : float  Non-copper fraction for HTS tapes [-].
    show_nhmfl_ref : bool   Overlay NHMFL 4.2 K reference data (default True).
    save_dir       : str or None

    References
    ----------
    ITER TF strand specifications; Nijhuis (2008); Fleiter & Ballarino (2014).
    NHMFL/MagLab Engineering Current Density Plot, updated 2021-01-04.
    """
    B_vals  = np.linspace(B_min, B_max, 300)
    J_NbTi  = J_non_Cu_NbTi(B_vals, T_op)  * f_non_Cu_LTS / 1e6   # [A/mm²]
    J_Nb3Sn = J_non_Cu_Nb3Sn(B_vals, T_op, Eps=-0.003) * f_non_Cu_LTS / 1e6
    J_REBCO = J_non_Cu_REBCO(B_vals, T_op, Tet=0) * f_non_Cu_HTS / 1e6

    # Colour palette shared between D0FUS curves and NHMFL reference overlay.
    col_NbTi, col_Nb3Sn, col_REBCO = "#A06AB4", "#E06C75", "#D4B000"

    fig, ax = plt.subplots(figsize=(7, 5))

    # NHMFL background reference (drawn first → lower z-order).
    if show_nhmfl_ref:
        _overlay_nhmfl_reference(
            ax,
            color_map={"NbTi": col_NbTi, "Nb3Sn": col_Nb3Sn, "REBCO": col_REBCO},
            alpha=0.30,
        )

    ax.plot(B_vals, J_NbTi,  lw=2, color=col_NbTi,  label="NbTi strand",  zorder=3)
    ax.plot(B_vals, J_Nb3Sn, lw=2, color=col_Nb3Sn, label="Nb₃Sn strand", zorder=3)
    ax.plot(B_vals, J_REBCO, lw=2, color=col_REBCO, label="REBCO tape",   zorder=3)
    ax.set_xlabel("Magnetic field B [T]", fontsize=12)
    ax.set_ylabel("Strand/Tape current density [A/mm²]", fontsize=12)
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
    f_void_user   = f_void        if f_void    is not None else cfg.f_void
    f_In          = f_In          if f_In      is not None else cfg.f_In
    T_hotspot     = T_hotspot     if T_hotspot is not None else cfg.T_hotspot
    RRR           = RRR           if RRR       is not None else cfg.RRR
    Marge_T_He    = cfg.Marge_T_He
    Marge_T_Nb3Sn = cfg.Marge_T_Nb3Sn
    Marge_T_NbTi  = cfg.Marge_T_NbTi
    Marge_T_REBCO = cfg.Marge_T_REBCO
    Eps           = cfg.Eps
    Tet           = cfg.Tet

    # Auto-select f_void per SC type when the user/cfg did not fix it.
    # LTS (NbTi, Nb3Sn): round strands in CICC → ~33% interstitial void.
    # HTS (REBCO):       stacked tapes → no interstitial void.
    _f_void_LTS = f_void_user if f_void_user is not None else 0.33
    _f_void_HTS = f_void_user if f_void_user is not None else 0.00

    B_range = np.linspace(B_min, B_max, 30)
    coil_params_base = dict(
        E_mag=E_mag, I_cond=I_cond, V_max=V_max,
        N_sub=N_sub, tau_h=tau_h,
        f_He_pipe=f_He_pipe, f_In=f_In,
        T_hotspot=T_hotspot, RRR=RRR,
        Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps, Tet=Tet,
    )

    J_NbTi   = []
    J_Nb3Sn  = []
    J_REBCO  = []

    for B in B_range:
        for sc, store, fv in [("NbTi",  J_NbTi,  _f_void_LTS),
                              ("Nb3Sn", J_Nb3Sn, _f_void_LTS),
                              ("REBCO", J_REBCO, _f_void_HTS)]:
            res = calculate_cable_current_density(
                sc_type=sc, B_peak=B, T_op=T_op,
                f_void=fv, **coil_params_base)
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
    a: float = 2.0,
    b: float = 2.7,
    R0: float = 9.0,
    sigma_TF: float = 867e6,
    J_max_TF: float = 60e6,
    B_max: float = 25.0,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    Plot TF coil total inboard thickness vs peak magnetic field for four
    mechanical models (Academic/refined × Wedging/Bucking).

    An optional MADE benchmark scatter dataset is superimposed on the Wedging
    panel for EU-DEMO validation.

    Parameters
    ----------
    a, b, R0   : float  Minor radius, blanket+shield thickness, major radius [m].
                         Defaults give R_ext = R0 - a - b = 4.3 m (Giannini 2023).
    sigma_TF   : float  Allowable TF coil stress [Pa].
                         Default 867 MPa (SDC-IC Pm+Pb, austenitic steel at 4 K).
    J_max_TF   : float  Current density on the non-steel area [A/m²].
                         Default 60 MA/m². The refined cable model (Maddock
                         adiabatic hotspot) predicts ~40 MA/m² for HTS at
                         20 T,
                         From Giannini 2023 Fig. 20 (validated FEM
                         design at 20.3 T): 234 conductors × 107 kA over
                         a WP area of 0.67 m² gives J_WP = 38 MA/m²
                         (including steel). Correcting for a typical jacket
                         steel fraction of ~45% yields J_wost ≈ 55-65
                         MA/m², hence the 60 MA/m² default.
    B_max      : float  Upper bound of field scan [T].
    cfg        : config object (DEFAULT_CONFIG if None)
    save_dir   : str or None

    References
    ----------
    Giannini et al. (2023), FED 193, 113659.
        Fig. 19: TF inner-leg total radial build vs B_max (HTS, pancake
        wound, RIS cable, R_i = 4.3 m, σ = 867 MPa, 16 TF coils).
        DOI: 10.1016/j.fusengdes.2023.113659
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
        # c_BP = 0: MADE total radial build = WP + case vault (no backplate)
        d0_w.append(f_TF_refined(a, b, R0, sigma_TF, J_max_TF, B,
                                "Wedging", 0.5, 1,
                                0.0, cfg.coef_inboard_tension,
                                cfg.F_CClamp)[0])
        d0_b.append(f_TF_refined(a, b, R0, sigma_TF, J_max_TF, B,
                                "Bucking", 1.0, 1,
                                0.0, cfg.coef_inboard_tension,
                                cfg.F_CClamp)[0])

    # MADE benchmark data — Giannini et al. (2023), FED 193, 113659, Fig. 19.
    # Total TF inner-leg radial build vs B_max(TF).
    # HTS pancake wound, RIS cable, R_i = 4.3 m, σ = 867 MPa, 16 TF.
    x_made = np.array([11.25, 13.25, 14.75, 16, 17, 18, 19,
                        19.75, 20.5, 21.25, 22, 22.5])
    y_made = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
                        2.0, 2.2, 2.4, 2.65, 2.85])

    colors = ["#1f77b4", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Wedging panel
    ax = axes[0]
    ax.plot(B_vals, acad_w, color=colors[0], lw=2, label="Academic — Wedging")
    ax.plot(B_vals, d0_w,   color=colors[1], lw=2, label="Refined — Wedging")
    ax.scatter(x_made, y_made, color="k", marker="x", s=80, label="MADE")
    ax.set_xlabel("Peak field B_max [T]", fontsize=12)
    ax.set_ylabel("TF total inboard thickness [m]", fontsize=12)
    ax.set_title("Wedging configuration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.6)

    # Bucking panel
    ax = axes[1]
    ax.plot(B_vals, acad_b, color=colors[0], lw=2, label="Academic — Bucking")
    ax.plot(B_vals, d0_b,   color=colors[1], lw=2, label="Refined — Bucking")
    ax.set_xlabel("Peak field B_max [T]", fontsize=12)
    ax.set_ylabel("TF total inboard thickness [m]", fontsize=12)
    ax.set_title("Bucking configuration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.6)

    plt.suptitle(
        f"TF coil thickness vs peak field\n"
        f"(R_ext = {R0 - a - b:.1f} m, σ = {sigma_TF/1e6:.0f} MPa, "
        f"J = {J_max_TF/1e6:.0f} MA/m²)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save_or_show(fig, save_dir, "TF_thickness_vs_field")


# ── TF winding pack grading figures ──────────────────────────────────

def plot_TF_grading_thickness_vs_field(
    a: float = 3.0,
    b: float = 1.7,
    R0: float = 9.0,
    sigma_TF: float = 660e6,
    J_max_TF: float = 50e6,
    n: float = 6,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    WP thickness vs B_max comparing graded and ungraded designs.

    Parameters
    ----------
    a, b, R0    : float  Geometry [m].
    sigma_TF    : float  Allowable Tresca stress [Pa].
    J_max_TF    : float  Engineering current density (non-steel) [A/m²].
    n           : float  Conductor geometry factor for γ(α, n).
    cfg         : config object (DEFAULT_CONFIG if None).
    save_dir    : str or None.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    B_scan = np.arange(8, 21.5, 0.5)
    c_u = np.full_like(B_scan, np.nan)
    c_g = np.full_like(B_scan, np.nan)

    for i, Bm in enumerate(B_scan):
        ru = Winding_Pack_refined(R0, a, b, sigma_TF, J_max_TF, Bm,
                                0.5, n, grading=False)
        rg = Winding_Pack_refined(R0, a, b, sigma_TF, J_max_TF, Bm,
                                0.5, n, grading=True)
        if np.isfinite(ru[0]):
            c_u[i] = ru[0] * 100
        if np.isfinite(rg[0]):
            c_g[i] = rg[0] * 100

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(B_scan, c_u, 'r-o', ms=3, lw=1.8, label='Ungraded')
    ax.plot(B_scan, c_g, 'b-s', ms=3, lw=1.8, label='Graded')
    ax.set_xlabel('$B_{max}$ (T)')
    ax.set_ylabel('$c_{WP}$ (cm)')
    ax.set_title('Winding pack thickness vs peak field')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([B_scan[0], B_scan[-1]])
    ax.set_ylim(bottom=0)
    fig.suptitle(
        f'$R_0$={R0}, $a$={a}, $b$={b} m, '
        f'$\\sigma_{{lim}}$={sigma_TF/1e6:.0f} MPa, '
        f'$J_{{max}}$={J_max_TF/1e6:.0f} A/mm², $n$={n}',
        fontsize=9, y=0.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save_or_show(fig, save_dir, "TF_grading_thickness_vs_field")


def plot_TF_grading_reduction(
    a: float = 3.0,
    b: float = 1.7,
    R0: float = 9.0,
    sigma_TF: float = 660e6,
    J_max_TF: float = 50e6,
    n: float = 6,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    WP thickness reduction (%) from grading vs B_max.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    B_scan = np.arange(8, 21.5, 0.5)
    c_u = np.full_like(B_scan, np.nan)
    c_g = np.full_like(B_scan, np.nan)

    for i, Bm in enumerate(B_scan):
        ru = Winding_Pack_refined(R0, a, b, sigma_TF, J_max_TF, Bm,
                                0.5, n, grading=False)
        rg = Winding_Pack_refined(R0, a, b, sigma_TF, J_max_TF, Bm,
                                0.5, n, grading=True)
        if np.isfinite(ru[0]):
            c_u[i] = ru[0] * 100
        if np.isfinite(rg[0]):
            c_g[i] = rg[0] * 100

    reduction = (1 - c_g / c_u) * 100

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(B_scan, reduction, '-^', ms=4, lw=1.8, color='#2ca02c')
    ax.set_xlabel('$B_{max}$ (T)')
    ax.set_ylabel('Reduction (%)')
    ax.set_title('WP thickness reduction from grading')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([B_scan[0], B_scan[-1]])
    ax.set_ylim(bottom=0)
    fig.suptitle(
        f'$R_0$={R0}, $a$={a}, $b$={b} m, '
        f'$\\sigma_{{lim}}$={sigma_TF/1e6:.0f} MPa, '
        f'$J_{{max}}$={J_max_TF/1e6:.0f} A/mm², $n$={n}',
        fontsize=9, y=0.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save_or_show(fig, save_dir, "TF_grading_reduction")


def plot_TF_grading_alpha_profile(
    a: float = 3.0,
    b: float = 1.7,
    R0: float = 9.0,
    sigma_TF: float = 660e6,
    J_max_TF: float = 50e6,
    B_max: float = 13.0,
    n: float = 6,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    Conductor fraction α(R) profile for graded vs ungraded at a given B_max.

    Reads the profile stored in _last_graded_profile by _solve_graded_wp.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # Run graded (populates _last_graded_profile) then ungraded
    rg = Winding_Pack_refined(R0, a, b, sigma_TF, J_max_TF, B_max,
                            0.5, n, grading=True)
    ru = Winding_Pack_refined(R0, a, b, sigma_TF, J_max_TF, B_max,
                            0.5, n, grading=False)

    if not np.isfinite(rg[0]) or not np.isfinite(ru[0]):
        print("[plot_TF_grading_alpha_profile] Infeasible design point.")
        return

    prof = _last_graded_profile
    if not prof:
        print("[plot_TF_grading_alpha_profile] No graded profile available.")
        return

    alpha_unif = 1 - ru[4]
    n_plot = min(len(prof['R']), len(prof['alpha']))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prof['R'][:n_plot], prof['alpha'][:n_plot], 'b-', lw=2,
            label='Graded $\\alpha(R)$')
    ax.axhline(alpha_unif, color='r', ls='--', lw=1.5,
               label=f'Ungraded $\\alpha$ = {alpha_unif:.3f}')
    ax.set_xlabel('$R$ (m)')
    ax.set_ylabel('$\\alpha$ (conductor fraction)')
    ax.set_title(f'Conductor fraction profile — $B_{{max}}$ = {B_max:.0f} T')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f'$R_0$={R0}, $a$={a}, $b$={b} m, '
        f'$\\sigma_{{lim}}$={sigma_TF/1e6:.0f} MPa, '
        f'$J_{{max}}$={J_max_TF/1e6:.0f} A/mm², $n$={n}',
        fontsize=9, y=0.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save_or_show(fig, save_dir, "TF_grading_alpha_profile")


def plot_CS_thickness_vs_flux(
    a_cs: float = 3.0,
    b_cs: float = 1.2,
    c_cs: float = 2.0,
    R0_cs: float = 9.0,
    B_TF: float = 13.0,
    B_max_CS: float = 50.0,
    sigma_CS: float = 600e6,
    J_wost_CS: float = 85e6,
    T_He: float = 4.75,
    kappa_cs: float = 1.7,
    N_sub: int = 6,
    tau_h: float = 5.0,
    psi_max: float = 500.0,
    n_psi: int = 80,
    cfg=None,
    save_dir: str | None = None,
) -> None:
    """
    Plot CS coil winding-pack thickness and peak field vs volt-second budget
    for three mechanical models (Academic, refined, CIRCE) and three
    configuration types (Wedging, Bucking, Plug).

    For the Wedging configuration, a MADE benchmark scatter dataset
    (Sarasola et al. 2020, Fig. 2, HTS UCD at σ_h = 300 MPa) is overlaid
    on the thickness panel.

    Default parameters reproduce the Sarasola 2020 benchmark geometry
    (EU-DEMO baseline 2018, R_o = 2.7 m, H = 17.92 m):

      Geometry: a_cs=3 m, b_cs=1.2 m, c_cs=2 m, R0=9 m, Gap=0.1 m
        → R_CS_ext = 9 - 3 - 1.2 - 2 - 0.1 = 2.7 m

      Stress:  σ_CS = 600 MPa.  The refined model applies fatigue_CS = 2 internally
        → σ_eff = 300 MPa as the fig.2 source

      J_wost = 85 MA/m², derived from Sarasola 2020 Sec. II-B:
        j_Cu   = 120 A/mm² = 120 MA/m²
        f_void = 0.10                     (Just He pipe)
        f_In = 0.10                       (Insulation)
        f_Cu/NonCu = 0.85 ?               (Pit Viper like)
        → J_wost = j_Cu × f_void × f_In × f_Cu = 85 MA/m²

      H_CS = 17.92 m forced via cfg.H_CS override (baseline 2018 allocation,
        Sec. II-A).  The default formula H = 2(κa + b + 1) gives 14.6 m,
        which underestimates the actual CS height by 20%.

    Parameters
    ----------
    a_cs, b_cs, c_cs : float  Plasma minor radius, inboard build, TF thickness [m].
    R0_cs            : float  Major radius [m].
    B_TF, B_max_CS   : float  TF peak field and CS maximum allowable field [T].
    sigma_CS         : float  CS yield strength [Pa] (halved by fatigue_CS = 2).
    J_wost_CS        : float  CS cable-space current density [A/m²].
    T_He             : float  Helium operating temperature [K].
    kappa_cs         : float  Plasma elongation [-].
    N_sub, tau_h     : int, float  Quench protection subdivisions and hold time.
    psi_max, n_psi   : float, int  Volt-second scan ceiling [Wb] and resolution.
    cfg              : config object (DEFAULT_CONFIG if None).
    save_dir         : str or None

    References
    ----------
    Sarasola et al., IEEE Trans. Appl. Supercond. 30(4), 4200705 (2020)
      — Fig. 2: HTS UCD flux vs R_i at R_o = 2.7 m; MADE reference data.
    Sarasola et al., IEEE Trans. Appl. Supercond. 33(5), 4201205 (2023)
      — EU-DEMO CS parametric studies (extended analysis).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # Force CS height to baseline 2018 value (Sarasola 2020, Sec. II-A).
    # The default formula H = 2(κa + b + 1) gives 14.6 m, which is 19%
    # below the actual 17.92 m allocation and would bias the comparison.
    cfg.H_CS = 17.92

    # Benchmark convention: f_swing_usable = 1.0 (no control reserve
    # subtracted). The MADE reference data published by Sarasola et al.
    # corresponds to the CS hardware design output at full bipolar swing
    # (the curve is parameterised on the total volt-second capacity of
    # the coil, not on the plasma inductive demand). To compare the refined model to
    # MADE on the same footing, we feed the full-bipolar abscissa to the
    # solver and disable the additional 1/f_swing_usable rescaling. The
    # design-mode default (0.75 = 25% control reserve) is restored
    # automatically outside this benchmark function.
    cfg.f_swing_usable = 1.0

    # MADE reference data — Sarasola et al. (2020), IEEE TAS 30(4), Fig. 2
    # HTS uniform-current-density CS, R_o = 2.7 m, σ_h ≈ 300 MPa curve.
    # Sarasola publishes the premagnetization flux ψ_premag (one direction,
    # 0 → +I_max). D0FUS uses the full bipolar swing convention
    # (Eq. Flux_CS_eq, factor 2π/3), so the abscissa values are scaled
    # by ×2 to obtain the equivalent full-bipolar hardware capacity.
    made_thickness = np.array([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    made_flux      = np.array([209, 209, 208, 205, 200, 192, 185, 177, 161]) * 2

    psi_values  = np.linspace(0, psi_max, n_psi)
    mech_confs  = ["Wedging", "Bucking", "Plug"]
    model_funcs = {"Academic": f_CS_ACAD, "refined": f_CS_refined, "CIRCE": f_CS_CIRCE}
    colors      = {"Academic": "blue", "refined": "green", "CIRCE": "red"}

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
                           label="MADE (Sarasola 2020)")

            ax.set_xlabel(r"$\Psi_{\rm plateau}$ [Wb]", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{conf} — {ylabel.split('[')[0].strip()}", fontsize=11)
            ax.grid(True, ls="--", alpha=0.6)
            ax.legend(fontsize=10)

        plt.suptitle(
            f"CS coil sizing — {conf} configuration\n"
            f"(R₀ = {R0_cs} m, H = 17.92 m, "
            f"σ_eff = {sigma_CS/2e6:.0f} MPa, "
            f"J_wost = {J_wost_CS/1e6:.0f} MA/m²)",
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

    In 'refined' geometry mode, three-point PCHIP profiles are used for
    κ(ρ) and δ(ρ) (C¹ continuous globally, monotonicity-preserving,
    with radially varying core penetration).  In 'Academic' mode,
    κ(ρ) = κ_edge = const and δ(ρ) = 0 everywhere, consistent with the
    cylindrical-torus approximation.

    Parameters
    ----------
    run      : dict   D0FUS run output.
    n_rho    : int    Number of radial grid points.
    save_dir : str or None

    References
    ----------
    Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980) — PCHIP scheme.
    Ball & Parra, PPCF 57, 045006 (2015) — κ core penetration.
    """
    R0, a, kappa_edge, delta_edge, kappa_95, delta_95, _ = _resolve_geometry(run)
    geom_mode = run.get("Plasma_geometry", "refined")
    rho = np.linspace(1e-4, 1.0, n_rho)

    if geom_mode == "Academic":
        # Academic: constant κ = κ_edge, δ = 0 everywhere
        kap_arr = np.full_like(rho, kappa_edge)
        del_arr = np.zeros_like(rho)
    else:
        # Refined Miller PCHIP profiles with radial variation
        kap_arr = kappa_profile(rho, kappa_edge, kappa_95)
        del_arr = delta_profile(rho, delta_edge, delta_95)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── κ(ρ) panel ────────────────────────────────────────────────────
    ax = axes[0]
    if geom_mode == "Academic":
        ax.axhline(kappa_edge, color="tab:blue", lw=2.2,
                   label=f"Academic: κ = κ_edge = {kappa_edge:.3f} (const)")
    else:
        ax.plot(rho, kap_arr, "tab:blue", lw=2.2, label="Refined PCHIP")
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
        ax.plot(rho, del_arr, "tab:red", lw=2.2, label="Refined PCHIP")
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

    _mode_str = "Academic" if geom_mode == "Academic" else "Refined PCHIP"
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

    When Plasma_geometry == 'refined', surfaces follow the Miller
    parameterisation with radially varying κ(ρ) and δ(ρ) built from
    three-point PCHIP profiles (C¹ continuous, monotonicity-preserving).
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
    geom_mode = run.get("Plasma_geometry", "refined")

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
            # Refined Miller PCHIP surfaces with radially varying κ(ρ), δ(ρ)
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
        _geom_label = f"Refined PCHIP (δ_edge = {delta_edge:.3f})"

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
    Plot the safety factor profile q(rho).

    Picks data from run['_q_sc'], which is populated by the q,j dispatcher
    (f_q_profile_academic or f_q_profile_refined according to
    config.q_profile_mode).  When the dict is unavailable, falls back to
    the analytical parametric q(rho) from f_q_profile() with alpha_J = 1.5.

    The current density decomposition j_Ohm + j_CD + j_bs is meaningful
    only in 'refined' mode (in 'academic' mode those components are zero
    by construction).  The figure stays focused on q(rho) and reports the
    integral diagnostic l_i(3) along with q_0 and either q(0.95) or the
    imposed q95.

    Parameters
    ----------
    run      : dict   D0FUS run output (requires key 'q95').
    n_rho    : int    Number of radial grid points (used only for fallback).
    save_dir : str or None

    References
    ----------
    Wesson, Tokamaks, 4th ed., sec. 3.5 - cylindrical j-q Ampere relation.
    Uckan et al., ITER IPDG89 (1990) - j ~ (1-rho^2)^alpha_J reference.
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054 - PROCESS sec. 4.1.
    """
    q95    = float(run["q95"])
    _q_sc  = run.get("_q_sc", None)
    q_mode = str(run.get("q_profile_mode", "refined"))   # default for legacy runs

    # ── Data source: dispatcher solution or analytical fallback ──────────
    if _q_sc is not None and 'q_arr' in _q_sc:
        rho   = _q_sc['rho']
        q_arr = _q_sc['q_arr']
        li    = _q_sc['li']
        q_at_95 = float(_q_sc.get('q_at_95', np.interp(0.95, rho, q_arr)))
        suptitle_str = (rf"Safety factor profile $q(\rho)$  "
                        rf"({q_mode} mode)")
    else:
        # Analytical fallback (no run output available — used by tests).
        rho     = np.linspace(0.0, 0.99, n_rho)
        alpha_J = float(run.get("alpha_J", 1.5))
        q_arr   = f_q_profile(rho, q95=q95, rho95=0.95, alpha_J=alpha_J)

        # Cylindrical li estimate from the analytical form
        q_edge   = q_arr[-1]
        I_norm   = q_edge * rho**2 / np.maximum(q_arr, 1e-3)
        I_norm   = I_norm / np.maximum(I_norm[-1], 1e-10)
        rho_s    = np.where(rho > 1e-8, rho, 1.0)
        li_integ = np.where(rho > 1e-8, I_norm**2 / rho_s, 0.0)
        li       = 2.0 * np.trapezoid(li_integ, rho)

        q_at_95 = float(np.interp(0.95, rho, q_arr))
        suptitle_str = (rf"Safety factor profile $q(\rho)$  "
                        rf"(analytical fallback, $\alpha_J = {alpha_J:.2f}$)")

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(rho, q_arr, "tab:blue", lw=2.2)
    ax.axhline(1.0, color="gray", ls=":",  lw=0.8, alpha=0.5)
    ax.axhline(q95, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.95, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.plot(rho[0],  q_arr[0], "o", color="tab:blue", ms=7, zorder=5)
    # In academic mode, q(rho_95) = q95 by construction; in refined mode,
    # q(rho_95) generally differs and is a meaningful diagnostic.
    if q_mode == "refined":
        ax.plot(0.95, q_at_95, "o", color="tab:blue", ms=7, zorder=5)
        ax.plot(0.95, q95,     "s", color="tab:red",  ms=7, zorder=5,
                label=rf"$q_{{95}}^{{\rm scaling}} = {q95:.2f}$")
        info_text = (rf"$q_0 = {q_arr[0]:.2f}$" + "\n"
                     rf"$q(\rho_{{95}}) = {q_at_95:.2f}$" + "\n"
                     rf"$l_i(3) = {li:.2f}$" + "\n"
                     rf"$q(1) = {q_arr[-1]:.2f}$")
    else:  # academic mode: q95 is imposed
        ax.plot(0.95, q95, "s", color="tab:red", ms=7, zorder=5,
                label=rf"$q_{{95}} = {q95:.2f}$ (imposed)")
        info_text = (rf"$q_0 = {q_arr[0]:.2f}$" + "\n"
                     rf"$l_i(3) = {li:.2f}$" + "\n"
                     rf"$q(1) = {q_arr[-1]:.2f}$")
    ax.text(0.04, 0.95, info_text,
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))
    ax.set_xlabel(r"$\rho$", fontsize=12)
    ax.set_ylabel(r"$q(\rho)$", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    fig.suptitle(suptitle_str, fontsize=11, y=0.99)
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
    "ARC":     {"R0": 3.30, "a": 1.10, "kappa": 1.84, "delta": 0.33, "color": "#AA3377"},
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

    # Sublayer widths and SOL/FW shape parameters
    d_SOL       = float(run.get("delta_SOL", run.get("delta_gap_plasma", 0.10)))
    f_kappa_sol = float(run.get("f_kappa_SOL", 0.10))
    d_FW        = float(run.get("delta_FW",         0.05))
    d_sh   = float(run.get("delta_shield",     0.30))
    d_VV   = float(run.get("delta_VV",         0.15))
    d_gTF  = float(run.get("delta_gap_TF",     0.05))
    fixed  = d_SOL + d_FW + d_sh + d_VV + d_gTF
    d_BB_ib = max(b            - fixed, 0.01)
    d_BB_ob = max(b + Delta_TF - fixed, 0.01)

    return dict(
        b=b, c_TF=c_TF, c_CS=c_CS, N_TF=N_TF, Gap=Gap, Delta_TF=Delta_TF,
        H_TF=H_TF,
        R_TF_in=R_TF_in, R_TF_out=R_TF_out,
        R_CS_ext=R_CS_ext, R_CS_int=R_CS_int,
        e_fw=e_fw, e_blanket=e_blanket, e_shield=e_shield, e_gap=e_gap,
        d_SOL=d_SOL, f_kappa_sol=f_kappa_sol, d_FW=d_FW,
        d_sh=d_sh, d_VV=d_VV, d_gTF=d_gTF,
        d_BB_ib=d_BB_ib, d_BB_ob=d_BB_ob,
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
    | ARC      | 3.30 | 1.10 | 1.84  | 0.33 | Sorbom et al., FED 100 (2015)    |
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

# _princeton_D_contour and _offset_contour have been moved to
# D0FUS_radial_build_functions.py (functions of the same name).
# plot_TF_side_view calls f_TF_cross_section which encapsulates both.


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
    Delta_TF = bd["Delta_TF"]

    # ── Princeton-D cross-section (from radial build) ───────────────
    (R_bore, R_TF_out, H_TF, _A_cross, _L_turn,
     R_out, Z_out, R_in, Z_in) = f_TF_cross_section(a, bd["b"], R0, c_TF, Delta_TF)

    R_TF_in = R_bore + c_TF       # Plasma-facing inner face (= R0 − a − b)
    W_TF    = R_TF_out - R_bore   # Total radial width [m]
    h       = H_TF / 2.0          # Half-height [m]

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
# B2b — Radial build assembly (CS + TF + blanket, one side, R–Z plane)
# ---------------------------------------------------------------------------


def plot_assembly_side_view(
    run: dict,
    save_dir: str | None = None,
) -> None:
    """
    Combined poloidal (R-Z) side view: CS half + one TF coil + blanket.

    Shows the full radial build in true proportions on a single figure:
      - Right half of the CS solenoid (rectangle, inboard side).
      - One Princeton-D TF coil cross-section (winding-pack annulus).
      - Blanket (amber fill, bounded by Princeton-D outer / Miller LCFS inner).
      - Plasma interior (light blue, bounded by the Miller LCFS).

    Four dimension annotations are drawn below the figure, at a common
    horizontal level, using the same engineering-drawing style as the
    individual side-view figures:
      * c_CS   — CS winding-pack radial thickness
      * c_TF   — TF inboard leg radial thickness
      * b_in   — blanket inboard thickness (= b)
      * b_out  — blanket outboard thickness (= b + Delta_TF)

    The total TF height H_TF is annotated on the right side of the figure.

    Parameters
    ----------
    run      : dict   D0FUS run output including radial build keys.
    save_dir : str or None
    """
    # ── Geometry ────────────────────────────────────────────────────────
    R0         = float(run["R0"])
    a          = float(run["a"])
    kappa_edge = float(run.get("kappa_edge", 1.85))
    delta_edge = float(run.get("delta_edge", 0.33))
    bd         = _resolve_build(run)

    b        = bd["b"]
    c_TF     = bd["c_TF"]
    Delta_TF = bd["Delta_TF"]
    c_CS     = bd["c_CS"]
    R_CS_int = bd["R_CS_int"]
    R_CS_ext = bd["R_CS_ext"]

    # TF Princeton-D contours
    (R_bore, R_TF_out, H_TF, _A, _L,
     R_tf_out, Z_tf_out,
     R_tf_in,  Z_tf_in) = f_TF_cross_section(a, b, R0, c_TF, Delta_TF)
    h_tf = H_TF / 2.0

    # Miller LCFS
    theta  = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    R_lcfs = R0 + a * np.cos(theta + np.arcsin(delta_edge) * np.sin(theta))
    Z_lcfs = kappa_edge * a * np.sin(theta)

    # CS height
    H_CS = 2.0 * (kappa_edge * a + b + 1.0)
    h_cs = H_CS / 2.0

    # Key midplane radii for annotations
    R_tf_in_ib  = R0 - a - b          # TF inner face, inboard
    R_pl_ib     = R0 - a              # plasma inboard edge
    R_pl_ob     = R0 + a              # plasma outboard edge
    R_blkt_ob   = R0 + a + b + Delta_TF   # blanket outer, outboard
    b_out       = b + Delta_TF
    V_blkt      = float(run.get("V_blanket", float("nan")))

    # ── Sublayer thicknesses ─────────────────────────────────────────────
    d_SOL       = bd["d_SOL"];  f_kappa_sol = bd["f_kappa_sol"];  d_FW = bd["d_FW"]
    d_sh        = bd["d_sh"];   d_VV  = bd["d_VV"];  d_gTF = bd["d_gTF"]
    d_BB_ib     = bd["d_BB_ib"];  d_BB_ob = bd["d_BB_ob"]

    # ── Colour palette ───────────────────────────────────────────────────
    col_plasma  = "#FFD0DA"   # light pink — distinct from white gaps
    LAYER_COLORS = {
        "SOL":        "#ADD8E6",   # scrape-off layer — light blue
        "First wall": "#4A505A",
        "BB":         "#E07820",
        "Shield":     "#2E8B57",
        "VV":         "#6C7A89",
        "TF gap":     "white",
        "Divertor":   "#9B2335",
    }

    # ── SOL/FW outer: Miller ellipses from the physics source of truth ────────
    R_SOL_outer, Z_SOL_outer, R_FW_outer, Z_FW_outer = _sol_fw_miller_contours(
        R0, a, kappa_edge, delta_edge, d_SOL, f_kappa_sol, d_FW)
    # TF-side layers: inward offsets of the TF inner face (same shape family)
    R_VV_outer,  Z_VV_outer  = _offset_contour(R_tf_in, Z_tf_in, d_gTF)
    R_sh_outer,  Z_sh_outer  = _offset_contour(R_tf_in, Z_tf_in, d_gTF + d_VV)
    R_BB_outer,  Z_BB_outer  = _offset_contour(R_tf_in, Z_tf_in, d_gTF + d_VV + d_sh)

    # ── BB sub-layer breakdown (concept-specific) ─────────────────────────
    # Sub-layer boundary contours are obtained by radial (polar) interpolation
    # between the FW-outer (f=0) and BB-outer (f=1) contours at the cumulative
    # sub-layer width fractions.  Ordered FW-side -> shield-side.
    Blanket_choice = run.get("Blanket_choice", "HCPB")
    _bb_concept    = BLANKET_CONCEPTS.get(Blanket_choice, BLANKET_CONCEPTS["HCPB"])
    BB_sublayers   = _bb_concept["sublayers"]
    _BB_SUBLAYER_COLORS = ["#E07820", "#8B5A2B", "#C9A66B", "#5A4632"]
    _f_cum = np.cumsum([0.0] + [s["f_width"] for s in BB_sublayers])

    # ── Divertor polygon (shared helper) ─────────────────────────────────
    f_div = float(run.get("f_div_area_fraction", 0.08))

    def _bottom_arc(R, Z, frac):
        n = len(R); ds = np.hypot(np.diff(np.r_[R, R[0]]), np.diff(np.r_[Z, Z[0]]))
        tgt = frac * ds.sum() / 2.0; i0 = int(np.argmin(Z))
        idx = [i0]; af = ab = 0.0
        for s in range(1, n // 2 + 1):
            af += ds[(i0 + s - 1) % n]; ab += ds[(i0 - s) % n]
            idx.append((i0 + s) % n); idx.insert(0, (i0 - s) % n)
            if af >= tgt and ab >= tgt: break
        return np.array(sorted(set(idx)))

    _di_in  = _bottom_arc(R_SOL_outer, Z_SOL_outer, f_div)
    _di_out = _bottom_arc(R_BB_outer,  Z_BB_outer,  f_div)
    R_div_poly = np.concatenate([R_SOL_outer[_di_in],  R_BB_outer[_di_out][::-1]])
    Z_div_poly = np.concatenate([Z_SOL_outer[_di_in],  Z_BB_outer[_di_out][::-1]])

    # ── Colours ─────────────────────────────────────────────────────────
    col_cs_bg  = "#C8C8C8"
    col_cs_wp  = "#303030"
    col_tf     = "black"
    col_line   = "black"
    col_dim    = "#333333"

    # ── Figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1 — CS: grey envelope, bore white, WP black
    ax.add_patch(plt.Rectangle(
        (0, -h_cs), R_CS_ext, H_CS, fc=col_cs_bg, ec="none", zorder=2))
    ax.add_patch(plt.Rectangle(
        (0, -h_cs), R_CS_int, H_CS, fc="white",   ec="none", zorder=3))
    ax.add_patch(plt.Rectangle(
        (R_CS_int, -h_cs), c_CS, H_CS, fc=col_cs_wp, ec=col_line, lw=1.2, zorder=4))

    # 2 — TF winding pack body (outermost shape black)
    ax.fill(R_tf_out, Z_tf_out, fc=col_tf, ec="none", zorder=5)

    # 3 — Paint layers: overpaint from outside in.
    zbase = 6
    ax.fill(R_tf_in,      Z_tf_in,      fc=LAYER_COLORS["TF gap"],    ec="none", zorder=zbase)
    ax.fill(R_VV_outer,   Z_VV_outer,   fc=LAYER_COLORS["VV"],        ec="none", zorder=zbase+1)
    ax.fill(R_sh_outer,   Z_sh_outer,   fc=LAYER_COLORS["Shield"],    ec="none", zorder=zbase+2)
    # BB base fill: coloured as the back-most (shield-side) sub-layer.  Inner
    # sub-layers are overpainted on top, FW-side first, each as a smaller
    # "disk" bounded by the radially-interpolated sub-layer boundary contour
    # (so smaller fractions paint last / on top of larger ones).
    ax.fill(R_BB_outer, Z_BB_outer,
            fc=_BB_SUBLAYER_COLORS[(len(BB_sublayers) - 1) % len(_BB_SUBLAYER_COLORS)],
            ec="none", zorder=zbase+3)
    for _i in range(len(BB_sublayers) - 1):
        _R_sub, _Z_sub = _radial_interp_contour(
            R_FW_outer, Z_FW_outer, R_BB_outer, Z_BB_outer, R0, _f_cum[_i + 1])
        ax.fill(_R_sub, _Z_sub,
                fc=_BB_SUBLAYER_COLORS[_i % len(_BB_SUBLAYER_COLORS)],
                ec="none", zorder=zbase + 3 + (len(BB_sublayers) - _i) * 0.01)
    ax.fill(R_FW_outer,   Z_FW_outer,   fc=LAYER_COLORS["First wall"],ec="none", zorder=zbase+4)
    ax.fill(R_SOL_outer,  Z_SOL_outer,  fc=LAYER_COLORS["SOL"],       ec="none", zorder=zbase+5)
    ax.fill(R_lcfs,       Z_lcfs,       fc=col_plasma,                ec="none", zorder=zbase+6)
    # Divertor: bottom arc between SOL outer (FW inner) and BB outer (shield inner)
    ax.fill(R_div_poly, Z_div_poly, fc=LAYER_COLORS["Divertor"], ec="none", zorder=zbase+7)

    # 4 — Contour lines: TF outer face and TF inner face only (no LCFS contour)
    ax.plot(R_tf_out, Z_tf_out, color=col_line, lw=1.8, zorder=zbase+8)
    ax.plot(R_tf_in,  Z_tf_in,  color=col_line, lw=1.0, zorder=zbase+8)
    for R_edge in (R_CS_int, R_CS_ext):
        ax.plot([R_edge, R_edge], [-h_cs, h_cs], color=col_line, lw=0.8, zorder=zbase+8)
    ax.plot([0, R_CS_ext], [-h_cs, -h_cs], color=col_line, lw=0.8, zorder=zbase+8)
    ax.plot([0, R_CS_ext], [ h_cs,  h_cs], color=col_line, lw=0.8, zorder=zbase+8)

    # 5 — White gap strip between CS outer face and TF bore
    gap_width = R_bore - R_CS_ext
    if gap_width > 0:
        ax.add_patch(plt.Rectangle(
            (R_CS_ext, -h_cs), gap_width, H_CS,
            fc="white", ec="none", zorder=zbase+9))

    # 6 — Axes
    ax.axvline(0, color=col_line, lw=0.5, ls="-.", alpha=0.5, zorder=1)
    ax.axhline(0, color=col_line, lw=0.4, ls="-.", alpha=0.4, zorder=1)

    # ── Component labels ─────────────────────────────────────────────────
    lbl_z = zbase + 10
    lbl = dict(ha="center", va="center", fontweight="bold", zorder=lbl_z)
    ax.text((R_CS_int + R_CS_ext) / 2, h_cs * 0.55,
            "CS", color="white", fontsize=9, **lbl)
    ax.text((R_bore + R_tf_in_ib) / 2, 0,
            "TF", color="white", fontsize=8, rotation=90, **lbl)
    ax.text(R_TF_out - c_TF * 0.5, 0,
            "TF", color="white", fontsize=8, rotation=90, **lbl)
    ax.text(R0, 0, "Plasma", color="#8B0030", fontsize=10, **lbl)

    # ── Layer legend ─────────────────────────────────────────────────────
    # Fixed display order: plasma, SOL, first wall, divertor, BB sub-layers
    # (FW-side -> shield-side), shield, VV, TF gap.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(fc=col_plasma, ec="grey", lw=0.4, label="Plasma"),
        Patch(fc=LAYER_COLORS["SOL"], ec="grey", lw=0.4, label="SOL"),
        Patch(fc=LAYER_COLORS["First wall"], ec="grey", lw=0.4, label="First wall"),
        Patch(fc=LAYER_COLORS["Divertor"], ec="grey", lw=0.4, label="Divertor"),
    ]
    for _j, _sub in enumerate(BB_sublayers):
        legend_elements.append(Patch(
            fc=_BB_SUBLAYER_COLORS[_j % len(_BB_SUBLAYER_COLORS)],
            ec="grey", lw=0.4, label=_sub["name"]))
    legend_elements += [
        Patch(fc=LAYER_COLORS["Shield"], ec="grey", lw=0.4, label="Shield"),
        Patch(fc=LAYER_COLORS["VV"], ec="grey", lw=0.4, label="VV"),
        Patch(fc="white", ec="grey", lw=0.4, label="TF gap"),
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, framealpha=0.9, edgecolor="grey", handlelength=1.2,
              borderpad=0.6, labelspacing=0.3)

    # ── Dimension annotations ────────────────────────────────────────────
    arr_kw = dict(arrowstyle="<->", color=col_dim, lw=0.9, mutation_scale=10)
    ext_kw = dict(color=col_dim, lw=0.4, ls="-", alpha=0.5)
    lbl_kw = dict(fontsize=8.5, color=col_dim, ha="center",
                  bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))

    # Bottom annotations — staggered levels for the narrow inboard stack
    z_base = -max(h_tf, h_cs)
    z_a1   = z_base - 0.50   # c_CS  (level 1)
    z_a2   = z_base - 1.20   # b_in  (level 2)
    z_a3   = z_base - 0.50   # b_out (level 1, right side — no overlap)

    def _ann_bot(R_left, R_right, z_from, z_arr, label):
        """Extension lines downward from z_from to z_arr, then arrow + label."""
        ax.plot([R_left,  R_left],  [z_from, z_arr - 0.05], **ext_kw, zorder=9)
        ax.plot([R_right, R_right], [z_from, z_arr - 0.05], **ext_kw, zorder=9)
        ax.annotate("", xy=(R_right, z_arr), xytext=(R_left, z_arr),
                    arrowprops=arr_kw, zorder=9)
        ax.text((R_left + R_right) / 2, z_arr - 0.22, label, va="top", **lbl_kw)

    _ann_bot(R_CS_int,   R_CS_ext,  -h_cs, z_a1, f"$c_{{CS}}$ = {c_CS:.2f} m")
    _ann_bot(R_tf_in_ib, R_pl_ib,   -h_tf, z_a2, f"$b_{{in}}$ = {b:.2f} m")
    _ann_bot(R_pl_ob,    R_blkt_ob, -h_tf, z_a3, f"$b_{{out}}$ = {b_out:.2f} m")

    # c_TF — top annotation above the inboard TF leg
    z_top  = h_tf + 0.40
    ax.plot([R_bore,     R_bore],     [h_tf, z_top - 0.05], **ext_kw, zorder=9)
    ax.plot([R_tf_in_ib, R_tf_in_ib], [h_tf, z_top - 0.05], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_tf_in_ib, z_top), xytext=(R_bore, z_top),
                arrowprops=arr_kw, zorder=9)
    ax.text((R_bore + R_tf_in_ib) / 2, z_top + 0.10,
            f"$c_{{TF}}$ = {c_TF:.2f} m", va="bottom", **lbl_kw)

    # H_TF on the right side of the TF outer face
    R_vann = R_TF_out + 0.35
    ax.plot([R_TF_out, R_vann - 0.03], [ h_tf,  h_tf], **ext_kw, zorder=9)
    ax.plot([R_TF_out, R_vann - 0.03], [-h_tf, -h_tf], **ext_kw, zorder=9)
    ax.annotate("", xy=(R_vann, h_tf), xytext=(R_vann, -h_tf),
                arrowprops=arr_kw, zorder=9)
    ax.text(R_vann + 0.18, 0,
            f"$H_{{TF}}$ = {H_TF:.2f} m", rotation=90,
            fontsize=9, color=col_dim, ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))

    # ── Axes styling ─────────────────────────────────────────────────────
    ax.set_aspect("equal")
    ax.set_xlabel("$R$  [m]", fontsize=11)
    ax.set_ylabel("$Z$  [m]", fontsize=11)
    ax.set_title(
        "Radial build — poloidal cross-section  (CS  /  TF  /  blanket  /  plasma)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(False)
    ax.tick_params(labelsize=10, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.set_xlim(-0.15, R_TF_out + 1.0)
    ax.set_ylim(z_a2 - 0.55, z_top + 0.8)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "run_assembly_side_view")


# ---------------------------------------------------------------------------
# B2b — Breeding-blanket concept comparison
# ---------------------------------------------------------------------------

def plot_blanket_concepts_comparison(save_dir: str | None = None) -> None:
    """
    Compare the breeding-blanket concepts defined in BLANKET_CONCEPTS.

    Left panel : TBR(delta_BZ) saturation curves
                     TBR = TBR_max * (1 - exp(-delta_BZ / delta_e)),
                     delta_e = delta_BB_sat / ln(20)
                 for the breeder/multiplier-zone thickness delta_BZ, one
                 curve per concept (dotted vertical line at delta_BB_sat).
    Right panel: grouped bars comparing TBR_max and the blanket energy
                 multiplication factor M_blanket across concepts.

    This figure is independent of any specific run — it summarises the
    concept database only.

    Parameters
    ----------
    save_dir : str or None
    """
    concepts = list(BLANKET_CONCEPTS.keys())
    colors   = plt.cm.tab10(np.linspace(0.0, 1.0, len(concepts)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: TBR(delta_BZ) saturation curves ───────────────────────────
    delta_BZ = np.linspace(0.0, 1.2, 200)
    for c, col in zip(concepts, colors):
        concept   = BLANKET_CONCEPTS[c]
        delta_e   = concept["delta_BB_sat"] / np.log(20.0)
        TBR_curve = concept["TBR_max"] * (1.0 - np.exp(-delta_BZ / delta_e))
        ax1.plot(delta_BZ, TBR_curve, color=col, lw=2, label=c)
        ax1.axvline(concept["delta_BB_sat"], color=col, lw=0.8, ls=":", alpha=0.6)

    ax1.axhline(1.0, color="grey", lw=1.0, ls="--", alpha=0.7)
    ax1.set_xlabel(r"Breeder/multiplier-zone thickness  $\delta_{BZ}$  [m]", fontsize=11)
    ax1.set_ylabel("Tritium breeding ratio (TBR)", fontsize=11)
    ax1.set_title("TBR saturation curves", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 1.6)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    # ── Right: TBR_max and M_blanket comparison bars ────────────────────
    x     = np.arange(len(concepts))
    width = 0.38
    TBR_max_vals = [BLANKET_CONCEPTS[c]["TBR_max"]   for c in concepts]
    M_bl_vals    = [BLANKET_CONCEPTS[c]["M_blanket"] for c in concepts]

    bars1 = ax2.bar(x - width / 2, TBR_max_vals, width, color="#4477AA", label="TBR_max")
    ax2.axhline(1.0, color="grey", lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel("TBR$_{max}$  [-]", fontsize=11, color="#4477AA")
    ax2.set_ylim(0, 1.6)
    ax2.tick_params(axis="y", labelcolor="#4477AA")

    ax2b  = ax2.twinx()
    bars2 = ax2b.bar(x + width / 2, M_bl_vals, width, color="#CC6677", label="M_blanket")
    ax2b.set_ylabel("$M_{blanket}$  [-]", fontsize=11, color="#CC6677")
    ax2b.set_ylim(0, 1.6)
    ax2b.tick_params(axis="y", labelcolor="#CC6677")

    ax2.set_xticks(x)
    ax2.set_xticklabels(concepts, fontsize=10)
    ax2.set_title(r"TBR$_{max}$ and energy multiplication $M_{blanket}$", fontsize=12, fontweight="bold")
    ax2.legend([bars1, bars2], ["TBR_max", "M_blanket"], fontsize=9, loc="upper right")

    fig.suptitle("Breeding-blanket concept comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "blanket_concepts_comparison")


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

    a          = float(run["a"])
    kappa_edge = float(run.get("kappa_edge", 1.85))
    bd = _resolve_build(run)

    R_CS_int = bd["R_CS_int"]          # Inner radius of winding pack [m]
    R_CS_ext = bd["R_CS_ext"]          # Outer radius of winding pack [m]
    c_CS     = bd["c_CS"]              # Winding-pack radial thickness [m]
    H_CS     = 2.0 * (kappa_edge * a + bd["b"] + 1.0)
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
      ── Transport & current      [12–13]
      ── Radiation & impurities   [14–16]
      ── Superconductor eng.      [17–19]
      ── Coil sizing & mechanics  [20–29]
          · TF grading            [21–23]
          · CS / CIRCE / geometry [24–27]
          · Benchmarks            [28–29]
      ── Machine comparison       [30]

    Parameters
    ----------
    run      : dict   D0FUS run output dict (ITER Q=10 reference by default).
    save_dir : str or None
        If provided, all figures are saved to that directory as PNG files.
        Pass ``None`` to display each figure interactively.
    cfg      : config object (DEFAULT_CONFIG if None).
    """
    N = 31

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

    _p(15, "D-T reactivity ⟨σv⟩(T)")
    plot_DT_reactivity(save_dir=save_dir)

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

    # ── TF winding pack grading ───────────────────────────────────────
    _p(21, "TF grading — WP thickness vs B_max (graded vs ungraded)")
    plot_TF_grading_thickness_vs_field(cfg=cfg, save_dir=save_dir)

    _p(22, "TF grading — WP thickness reduction from grading (%)")
    plot_TF_grading_reduction(cfg=cfg, save_dir=save_dir)

    _p(23, "TF grading — conductor fraction α(R) profile")
    plot_TF_grading_alpha_profile(cfg=cfg, save_dir=save_dir)

    _p(24, "CS thickness vs flux swing")
    plot_CS_thickness_vs_flux(cfg=cfg, save_dir=save_dir)

    _p(25, "CIRCE stress-model validation")
    plot_CIRCE_stress_validation(save_dir=save_dir)

    _p(26, "Radial build assembly (CS / TF / blanket)")
    plot_assembly_side_view(run, save_dir=save_dir)

    _p(27, "TF coil side view")
    plot_TF_side_view(run, save_dir=save_dir)

    _p(28, "CS cross-section")
    plot_CS_cross_section(run, save_dir=save_dir)

    # ── Benchmarks & machine comparison ───────────────────────────────
    _p(29, "TF benchmark table")
    plot_TF_benchmark_table(cfg=cfg, save_dir=save_dir)

    _p(30, "CS benchmark table")
    plot_CS_benchmark_table(cfg=cfg, save_dir=save_dir)

    _p(31, "Tokamak LCFS comparison")
    plot_cross_section_comparison(run=run, save_dir=save_dir)

    print("Done.")


def plot_run(
    run: dict,
    save_dir: str | None = None,
) -> None:
    """
    Render the run-specific figure set (12 figures).

    This is the subset called after each D0FUS run.  It contains only the
    figures that depend on the current run configuration and results —
    no validation curves, no benchmarks, no scaling-law surveys.

    Figures produced:
      [ 1/12]  Tokamak LCFS comparison (with D0FUS overlay)
      [ 2/12]  Miller flux surfaces (run geometry)
      [ 3/12]  Shaping profiles κ(ρ), δ(ρ)
      [ 4/12]  Kinetic profiles n(ρ), T(ρ), p(ρ)
      [ 5/12]  Safety factor q(ρ) and current decomposition
      [ 6/12]  Radiation profiles
      [ 7/12]  Radial build assembly (CS / TF / blanket)
      [ 8/12]  Breeding-blanket concept comparison
      [ 9/12]  TF coil side view
      [10/12]  CICC TF conductor
      [11/12]  CS cross-section
      [12/12]  CICC CS conductor

    Parameters
    ----------
    run      : dict   D0FUS run output dict.
    save_dir : str or None
        If provided, figures are saved as PNG files.
        Pass ``None`` to display interactively.
    """
    N = 12

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
    _p(7, "Radial build assembly (CS / TF / blanket)")
    plot_assembly_side_view(run, save_dir=save_dir)

    _p(8, "Breeding-blanket concept comparison")
    plot_blanket_concepts_comparison(save_dir=save_dir)

    _p(9, "TF coil side view")
    plot_TF_side_view(run, save_dir=save_dir)

    _p(10, "CICC TF conductor")
    plot_CICC_cross_section(build_conductor_from_run(run, coil="TF"), save_dir=save_dir)

    _p(11, "CS cross-section")
    plot_CS_cross_section(run, save_dir=save_dir)

    _p(12, "CICC CS conductor")
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
    against published reference values for ITER, EU-DEMO 2017, JT-60SA,
    EAST, ARC, and SPARC.

    The table is rendered as a matplotlib figure (colour-coded header) so it
    can be saved to a file alongside the other run figures.

    Parameters
    ----------
    cfg      : config object (DEFAULT_CONFIG if None).
    save_dir : str or None

    References
    ----------
    ITER:
        Mitchell et al., IEEE TAS 18(2), 435 (2008).
        Sborchia et al., IEEE TAS 18(2), 463 (2008).
        ITER DDD 1.1 (Magnet).
    EU-DEMO 2017:
        PROCESS DEMO1_Reference_Design_2017, Kovari et al., FED 104, 9 (2016).
        Federici et al., NF 64, 036025 (2024).
    JT-60SA:
        Shirai et al., NF 57, 102002 (2017).
        Tsuchiya et al., IEEE TAS 18(2), 208 (2008).
        Muzzi et al., IEEE TAS 21(3), 1063 (2011).
        Di Pietro et al., FED 89, 2128 (2014).
    EAST:
        Weng et al., FED 81, 1589 (2007).
        Wan et al., IAEA FEC 2006, FT/P7-11.
        Wu, Y. et al., FED 65, 331 (2003).
    ARC:
        Sorbom et al., FED 100, 378 (2015).
    SPARC:
        Creely et al., J. Plasma Phys. 86(5), 865860502 (2020).
        Hartwig et al., IEEE TAS (arXiv:2308.12301, 2023).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # -----------------------------------------------------------------
    # Machine parameters — sourced March 2026
    #
    # ITER:
    #   Mitchell et al. (2008), IEEE TAS 18(2), 435 — Table I, VI.
    #   Sborchia et al. (2008), IEEE TAS 18(2), 463 — Table I, Fig. 2.
    #   Libeyre et al. (2009), FED 84, 1188 — CS outer radius.
    #   ITER DDD 1.1 (Magnet) for σ, c_BP, Gap.
    #   b = 1.104 m derived self-consistently from R_CS_out = 2.096 m
    #       (Libeyre 2009), Gap_CS-TF = 0.10 m (DDD), c_TF = 0.90 m (DDD):
    #       b = R0 - a - (R_CS_out + Gap + c) = 6.20 - 2.00 - 3.096 = 1.104.
    #
    # EU-DEMO 2017:
    #   PROCESS DEMO1_Reference_Design_2017_March_EU_2NDSKT_v1_0.
    #   bore = 2.365 m, ohcth = 0.802 m, R_CS_out = 3.167 m.
    #   B_max_CS = 11.35 T (bmaxoh0, BOP).
    #   σ_CS = 600 MPa in PROCESS (alstroh). NOTE: this PROCESS run
    #   did not account for fatigue; in a real design the allowable
    #   stress would be lower (~330 MPa with Paris law, as for ITER).
    #   We keep 600 MPa here to be consistent with the PROCESS run
    #   being benchmarked. J_wost = coheof/(1-oh_steel) ≈ 60 MA/m².
    #   Gap_eff = precomp + gapoh = 0.056 + 0.050 = 0.106 m.
    #   b = 1.820 m (self-consistent with PROCESS radial build).
    #   Ψ_CS = 360 Wb (BOBOZ, Auclair 2025), full swing at R0 = 8.938 m.
    #   Cross-checked against PROCESS: BOBOZ Ψ_bore = 376 Wb matches
    #   PROCESS CS BOP = 182 Wb × 2 = 364 Wb (3% agreement).
    # -----------------------------------------------------------------
    # JT-60SA:
    #   Yoshida et al. (2010), JPFR Series 9, 214 — Table 4 (final design):
    #       Rc = 0.824 m, dR = 0.340 m, dZ = 1.585 m, 549 turns/module,
    #       I = 20 kA, B_max = 8.9 T. R_CS_in = 0.654 m, R_CS_out = 0.994 m.
    #   Yoshida et al. (2008), IEEE TAS 18(2), 441 — Table III (pre re-baseline).
    #   Tsuchiya et al. (2008), IEEE TAS 18(2), 208 — σ_y(316LN, 4K) = 820 MPa,
    #       Sm = 2/3 × 820 = 547 MPa (ASME); gap CS-TF = 15 mm.
    #   Ψ_CS = 27.5 Wb, computed with BOBOZ (Auclair 2025) using the
    #       Yoshida 2010 CS geometry (4 modules, ~20 mm inter-module gap,
    #       all at +20 kA). Ψ is the full swing (2 × Ψ_one_dir) evaluated
    #       at R0 = 2.96 m, same method and convention as the ITER CS
    #       benchmark (BOBOZ ITER test case, Auclair 2025: 252 Wb total,
    #       111 Wb PF, 141 Wb CS).
    #       Published total swing (CS+EF) = 40 Wb (Yoshida 2008, 2010);
    #       EF contribution = 40 − 27.5 = 12.5 Wb (31%), consistent with
    #       ITER (~44%) and EU-DEMO (~30%).
    #   D0FUS result: d = 0.174 m, B = 5.3 T (−49% vs d_ref = 0.340 m).
    #   The underprediction is larger than for ITER/EU-DEMO because the
    #   JT-60SA CS is winding-limited (549 turns of 27.9 mm conductor
    #   set the radial thickness), not stress-limited. D0FUS optimises
    #   for stress and therefore undersizes. Same effect as EAST.
    #   The CS sizing remains extremely sensitive to flux: +5 Wb would
    #   bring Δ to ~−20%.
    #   b = 0.361 m, Gap = 0.015 m (Tsuchiya 2008, self-consistent).
    #
    # EAST:
    #   Wu Weiyue et al. (2003), IEEE 0-7803-7908-X — PF system design:
    #       6 CS coils (3 symmetric pairs), NbTi CICC 20.4×20.4 mm,
    #       7 radial × 20 axial = 140 turns/module, I = 14.5 kA.
    #       ID = 1.1 m → R_CS_in = 0.55 m, dR = 0.16 m → R_CS_out = 0.71 m.
    #       H_module = 0.45 m. B_max = 4.5 T. Peak stress ~300 MPa.
    #   Weng et al. (2001), FED 58-59, 827 — overview, B0 = 3.5 T.
    #   Wan et al. (2002), IAEA FT/P2-03 — R0 = 1.75 m, total ~10 V·s.
    #   Ψ_CS = 8.5 Wb, computed with BOBOZ (Auclair 2025) using the
    #       Wu 2003 CS geometry (6 modules, ~10 mm gaps, all at +14.5 kA).
    #       Ψ is the full swing evaluated at R0 = 1.85 m.
    #       Published total (CS+PF) ≈ 10 Wb; PF = 1.5 Wb (15%).
    #   Gap = 0.29 m: the CS outer radius (0.71 m) is much smaller than
    #       the TF bore (~1.0 m), leaving ~0.29 m of void. This is set
    #       as Gap in D0FUS to place the CS at the correct radius.
    #   D0FUS result: d = 0.089 m (−44% vs d_ref = 0.16 m). The CS is
    #       winding-limited (7 radial conductor turns set the thickness),
    #       not stress-limited. D0FUS optimises for stress and therefore
    #       undersizes. This is a known limitation for small, low-field CS.
    #
    # SPARC:
    #   Creely et al. (2020), J. Plasma Phys. 86(5), 865860502.
    #   Hartwig et al. (2023), IEEE TAS (arXiv:2308.12301) — TFMC
    #       20.1 T, 40.5 kA, T_op = 20 K.
    #
    # ARC:
    #   Sorbom et al. (2015), FED 100, 378 — Table 1, Table 3.
    # -----------------------------------------------------------------

    machines_TF = {
        "ITER":    {"a": 2.00, "b": 1.104,"R0": 6.20, "σ": 660e6,  "T_op": 4.2,
                    "B_max": 11.8, "n_TF": 1,   "sc": "Nb3Sn", "config": "Wedging",
                    "κ": 1.7,  "I_cond": 68e3, "V_max": 10e3, "N_sub": 9,
                    "tau_h": 2.0,  "J_wost": 35e6},
        "EU-DEMO": {"a": 2.883,"b": 1.821,"R0": 8.938,"σ": 600e6,  "T_op": 4.75,
                    "B_max": 10.61,"n_TF": 0.5, "sc": "Nb3Sn", "config": "Wedging",
                    "κ": 1.65, "I_cond": 90e3, "V_max": 8.6e3,"N_sub": 8,
                    "tau_h": 2.0,  "J_wost": 30e6},
        "JT60-SA": {"a": 1.18, "b": 0.36, "R0": 2.96, "σ": 547e6,  "T_op": 4.5,
                    "B_max":  5.65, "n_TF": 1,   "sc": "NbTi",  "config": "Wedging",
                    "κ": 1.95, "I_cond": 25.7e3,"V_max": 2.8e3,"N_sub": 3,
                    "tau_h": 1.0,  "J_wost": 20e6},
        "EAST":    {"a": 0.45, "b": 0.15, "R0": 1.85, "σ": 660e6,  "T_op": 4.5,
                    "B_max":  5.8,  "n_TF": 1,   "sc": "NbTi",  "config": "Wedging",
                    "κ": 1.9,  "I_cond": 14.3e3,"V_max": 5e3,  "N_sub": 4,
                    "tau_h": 1.0,  "J_wost": 30e6},
        "ARC":     {"a": 1.10, "b": 0.89, "R0": 3.30, "σ": 1000e6, "T_op": 20.0,
                    "B_max": 23.0,  "n_TF": 1,   "sc": "REBCO", "config": "Plug",
                    "κ": 1.84, "I_cond": 50e3, "V_max": 10e3, "N_sub": 6,
                    "tau_h": 20,   "J_wost": 120e6},
        "SPARC":   {"a": 0.57, "b": 0.18, "R0": 1.85, "σ": 1000e6, "T_op": 20.0,
                    "B_max": 20.0,  "n_TF": 1,   "sc": "REBCO", "config": "Bucking",
                    "κ": 1.75, "I_cond": 40.5e3,"V_max": 10e3, "N_sub": 4,
                    "tau_h": 20,   "J_wost": 120e6},
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
            # f_TF_refined returns the same tuple layout
            return f_TF_refined(a, b, R0, sigma, J_wost, B_max,
                              conf, omega, n_frac,
                              cfg.c_BP, cfg.coef_inboard_tension, cfg.F_CClamp)

    # Model definitions: (label, header colour)
    # Note: there is no separate f_TF_CIRCE yet; Academic / D0FUS are the two
    # available TF mechanical models.  A third "CIRCE" row is reserved for
    # future implementation and left blank (NaN) to keep the table layout
    # consistent with the CS benchmark.
    model_specs = [
        ("Academic", "#9C27B0"),
        ("refined",    "#4CAF50"),
    ]

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

    IMPORTANT: cfg.cs_axial_stress must be True for meaningful results.

    Parameters
    ----------
    cfg      : config object (DEFAULT_CONFIG if None).
    save_dir : str or None

    References per machine (consolidated)
    -------------------------------------
    ITER:
        Schultz et al., IEEE TAS 17(2), 1808 (2007) — total CS+PF swing
            of 277 Wb at full bipolar; peak field 13 T; stored energy 6.4 GJ.
        Polevoi et al., Nucl. Fusion 55, 063019 (2015) — total inductive
            consumption ≈ 240 Wb (PI + ramp-up + plateau).
        Duchateau et al., Fusion Eng. Des. 89, 2606 (2014) — PF coil
            equilibrium contribution ≈ 73 Wb.
        Libeyre et al., Fusion Eng. Des. 84, 1188 (2009) — Table 1
            geometry: R_in = 1.342 m, R_out = 2.096 m, height 12.96 m.
        Coatanéa-Gouachet et al., Fusion Eng. Des. 86, 1418 (2011) —
            quench protection, dump time 7.5 s.
    EU-DEMO 2017 baseline:
        PROCESS DEMO1_Reference_Design_2017_March (internal CEA / EUROfusion
            input) — bore = 2.365 m, ohcth = 0.802 m, peak field 11.35 T.
        Sarasola et al., IEEE TAS 30(4), 4200705 (2020) — methodology +
            premag target 250 Wb (= 500 Wb full bipolar) for the 2018
            baseline; Φ_CS ≈ 320 Wb earlier in the same year.
        Tomasek et al., Fusion Eng. Des. 178, 113114 (2022) — DEMO magnet
            system status, 250 Wb premag for 2018 baseline.
    JT-60SA:
        Yoshida et al., J. Plasma Fusion Res. SERIES 9, 214 (2010) —
            mechanical CS design, R_c = 0.824 m, dR = 0.34 m, H = 6.34 m.
        Tsuchiya et al., IEEE TAS 18(2), 208 (2008) — Nb3Sn conduit
            qualification, σ_y(316LN, 4 K) ≈ 820 MPa.
    EAST:
        Wu et al., Fusion Eng. Des. 65, 331 (2003) — initial design.
        Weng et al., Fusion Eng. Des. 81, 1589 (2007) — magnet system.
        Wan et al., Engineering 7, 1597 (2021) — operational status,
            CS modules at 4.5 T peak field.
    ARC:
        Sorbom et al., Fusion Eng. Des. 100, 378 (2015) — Table 1, Table 3,
            Fig. 2 (radial build), total VS budget 32 Wb (CS+PF). The
            CS-only share is taken at 60% (PF fraction ≈ 40% reflecting
            the small R_0 and high bootstrap fraction characteristic of
            the steady-state non-inductive scenario), giving Ψ_CS ≈
            19.2 Wb.
    SPARC:
        Creely et al., J. Plasma Phys. 86(5), 865860502 (2020) —
            Table 1, total VS budget 42 Wb (CS+PF). The CS-only share
            is taken at 60% (PF fraction ≈ 40%, same as ARC, reflecting
            the small R_0 characteristic of compact-HTS architectures),
            giving Ψ_CS ≈ 25.2 Wb. No public B_CS reference is
            available; D0FUS prediction is reported on its own.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # -----------------------------------------------------------------
    # CS machine parameters — sourced March 2026
    #
    # NOTE: cs_axial_stress MUST be True (cfg.cs_axial_stress = True)
    # for the CS benchmark to produce meaningful results.
    #
    # ── Flux convention ──
    # Ψplateau is fed to the solvers as the CS hardware capacity demand
    # at full bipolar swing (-I_max → +I_max), consistent with the
    # geometric formula
    #   ΨCS = (2π/3) B_CS (R_e² + R_e R_i + R_i²)
    # used internally by the solvers. The published flux values for the
    # six reference machines below correspond to the hardware full-swing
    # capacity (volt-second budget of the coil itself), as reported in
    # the design papers cited per machine.
    #
    # The benchmark function overrides cfg.f_swing_usable = 1.0 (set
    # below) because the published values already correspond to the
    # hardware capacity, with no additional 1/f_swing_usable rescaling.
    # The design-mode default (0.75 = 25% control reserve for vertical
    # stability, error-field correction and shape feedback) is restored
    # automatically outside this benchmark function.
    #
    # ── Stress convention ──
    # σ_CS = 2/3 × Sy (monotonic yield at 4K). D0FUS applies
    # fatigue_CS = 2 internally, giving σ_eff = σ_CS / 2 ≈ fatigue
    # allowable. Verified against Sarasola 2020 Fig. 3 for ITER
    # (JK2LB, 60k cycles: σ_fatigue ≈ 330 ≈ 667/2).
    #
    # ── H_CS override ──
    # The formula H = 2(κa + b + 1) is inaccurate for machines
    # where the CS extends beyond the plasma (ITER: -15%) or is
    # shorter than expected (EAST: +46%). H_CS override provides
    # the real CS height, which is critical for the magnetic energy
    # estimate consumed by the quench protection model.
    #
    # ── Per-machine details ──
    # ITER:
    #   Geometry: Libeyre et al. (2009), FED 84, 1188 — Table 1.
    #     R_CS_in = 1.342 m, R_CS_out = 2.096 m, H = 12.96 m.
    #     b = 1.104 m (from R_CS_out=2.096, Gap=0.10, c=0.90).
    #   Operation: B_max = 13 T (peak field at premagnetization).
    #   Quench: Coatanéa-Gouachet et al. (2011), FED 86, 1418 —
    #     stored energy 6.4 GJ, dump time 7.5 s.
    #   Stress: σ = 667 MPa = 2/3 × Sy(JK2LB, 4K) = 2/3 × 1000 MPa.
    #     Sy source: JAEA qualification (2015), IOP MSE 102, 012002.
    #   Conductor: J_wost = 45 MA/m² (JAEA conduit 51.3 mm, I = 45 kA,
    #     A_cable = 979 mm²).
    #   Flux: Ψ_CS = 233 Wb (CS hardware capacity, full bipolar).
    #     Cross-checks: Schultz et al. (2007), IEEE TAS 17(2), 1808
    #     cites a CS+PF total swing of 277 Wb; the CS share alone is
    #     consistent with 233 Wb after subtracting the PF equilibrium
    #     contribution (≈ 73 Wb, Duchateau et al., FED 89, 2606, 2014).
    #     Sborchia et al. (2008), IEEE TAS 18(2), 463, also report
    #     compatible numbers from the magnet system design study.
    #
    # EU-DEMO 2017 baseline:
    #   Geometry: PROCESS DEMO1_Reference_Design_2017_March (CEA /
    #     EUROfusion internal). bore = 2.365 m, ohcth = 0.802 m,
    #     R_CS_out = 3.167 m, H_CS = 15.15 m (formula exact).
    #     Gap_eff = precomp + gapoh = 0.056 + 0.050 = 0.106 m.
    #   Operation: B_max = 11.35 T (bmaxoh0, BOP).
    #   Stress: σ = 600 MPa (PROCESS alstroh ≈ 2/3 × Sy(316L, 4K)
    #     ≈ 2/3 × 900).
    #   Conductor: J_wost = 60 MA/m² (coheof/(1-oh_steel)).
    #   Flux: Ψ_CS = 500 Wb (CS hardware capacity, full bipolar).
    #     Source: Sarasola et al. (2020), IEEE TAS 30(4), 4200705 and
    #     Tomasek et al. (2022), FED 178, 113114, both citing a
    #     premagnetization target of 250 Wb (= 500 Wb full bipolar) for
    #     the EU-DEMO baseline. The earlier 2017 PROCESS run cited
    #     above gives a CS swing of 382 Wb (BOP +182 Wb to EOF -200 Wb,
    #     as read from the run's by-circuit volt-second table); this
    #     PROCESS swing assumes asymmetric currents between the five
    #     CS modules and is therefore lower than the full-bipolar
    #     hardware capacity of a uniform-J_smear single solenoid (the
    #     model used by D0FUS), for which 500 Wb is the appropriate
    #     equivalent input.
    #
    # JT-60SA:
    #   Geometry: Yoshida et al. (2010), JPFR SERIES 9, 214 — Rc = 0.824 m,
    #     dR = 0.340 m, H = 6.34 m, four CS modules.
    #     Confirmed in Murakami et al. (2021), IEEE TAS 31(5), 4200305.
    #   Operation: B_max = 8.9 T at 20 kA per module.
    #   Stress: Tsuchiya et al. (2008), IEEE TAS 18(2), 208 — σ_y(316L,
    #     4K) = 820 MPa, σ = 2/3 × 820 ≈ 547 MPa.
    #   Quench: Kizu et al. (2012), IEEE TAS 22(3), 4204004 — quench
    #     detection design for the CS conductor.
    #   Flux: Ψ_CS = 40 Wb (CS hardware capacity, full bipolar).
    #     Source: Di Pietro et al. (2014), FED 89, 2128 (CS module
    #     design and operating scenario).
    #
    # EAST:
    #   Geometry: Wu et al. (2003), IEEE 0-7803-7908-X — six CS coils,
    #     ID = 1.1 m → R_CS_in = 0.55 m, dR = 0.16 m → R_CS_out = 0.71 m,
    #     H_CS = 2.75 m. Gap = 0.29 m (CS far from TF inboard leg).
    #     Confirmed in Chen et al. (2016), Fusion Sci. Tech. 70, 533
    #     (3D field analysis) and Chen et al. (2006), IEEE TAS 16(2),
    #     780 (fabrication).
    #   Operation: B_max = 4.5 T peak field on the CS modules.
    #   Stress: σ = 547 MPa (316L, same convention as JT-60SA).
    #   Flux: Ψ_CS = 10 Wb (CS hardware capacity, full bipolar).
    #     Yi et al. (2014), Fusion Sci. Tech. 65, 244 also reports
    #     compatible values from the EAST experimental scenario
    #     studies.
    #
    # SPARC:
    #   Geometry: Creely et al. (2020), J. Plasma Phys. 86(5), 865860502,
    #     Table 1.
    #   Stress: σ = 1000 MPa = 2/3 × Sy(N50H, 4K) = 2/3 × 1500 MPa
    #     (Wang et al. (2024), Cryogenics 139, 103836).
    #   Conductor: J_wost = 120 MA/m² (PIT VIPER REBCO; Sanabria et al.
    #     (2024), SUST 37, 075003, Table 1: J_eng ≈ 113 MA/m² at peak
    #     field, 120 used here to account for the non-steel fraction).
    #   Flux: Ψ_CS = 25.2 Wb. Creely 2020 Table 1 cites 42 Wb as the
    #     total CS+PF volt-second budget. The PF coil contribution is
    #     estimated at 40% of the total, larger than the ITER-
    #     calibrated baseline (25%) because of the small R_0 = 1.85 m
    #     characteristic of compact-HTS architectures, which increases
    #     the relative PF/CS share. The same 40% fraction is applied
    #     to ARC for consistency. Applying 40% gives Ψ_CS = 0.60 × 42
    #     = 25.2 Wb.
    #   Operation: B_CS public reference value not disclosed at the
    #     time of writing. The CSMC test programme (Sanabria et al.
    #     2024) reported the cable-level performance of PIT VIPER but
    #     not the integrated machine peak field. The benchmark
    #     therefore reports the D0FUS prediction without a public
    #     comparison value.
    #     d_ref ≈ 0.25 m (Creely Fig. 2 radial build).
    #   Config: Bucking. T_op = 20 K.
    #
    # ARC:
    #   Geometry: Sorbom et al. (2015), FED 100, 378 — Table 1
    #     (R0 = 3.3 m, a = 1.1 m), Fig. 2 (radial build, d_ref = 0.30 m).
    #   Operation: B_CS = 13 T (REBCO peak field at the conductor).
    #   Stress: σ = 1000 MPa (N50H, same convention as SPARC).
    #   Conductor: J_wost = 120 MA/m² (VIPER REBCO, same as SPARC).
    #   Flux: Ψ_CS = 19.2 Wb. Sorbom 2015 Table 3 cites 32 Wb as the
    #     total CS+PF volt-second budget. The PF coil contribution is
    #     estimated at 40% of the total for ARC, larger than the ITER-
    #     calibrated baseline (25%) because of the smaller R_0 and
    #     the steady-state non-inductive scenario (f_BS = 0.63), both
    #     of which increase the relative PF/CS share. A BOBOZ
    #     Shafranov-equilibrium calculation supports a PF fraction in
    #     the 40-60% range for this compact-HTS architecture.
    #     Applying 40% gives Ψ_CS = 0.60 × 32 = 19.2 Wb.
    #   Config: Plug (demountable joint).
    # -----------------------------------------------------------------

    # Force benchmark convention (full hardware capacity at full bipolar
    # swing, no control reserve subtracted from the published values).
    cfg.f_swing_usable = 1.0

    machines = {
        "ITER":    {"Ψplateau": 233,  "a_cs": 2.00, "b_cs": 1.104,"c_cs": 0.90,
                    "R0_cs": 6.20, "B_TF": 11.8, "B_cs": 13,   "σ_CS": 667e6,
                    "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.2,
                    "kappa": 1.7, "J_wost": 45e6, "H_CS": 12.96},
        "EU-DEMO": {"Ψplateau": 500,  "a_cs": 2.883,"b_cs": 1.820,"c_cs": 0.962,
                    "R0_cs": 8.938,"B_TF": 10.61,"B_cs": 11.35,"σ_CS": 600e6,
                    "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.75,
                    "kappa": 1.65, "J_wost": 60e6, "Gap": 0.106, "H_CS": 15.15},
        "JT60-SA": {"Ψplateau":  40,  "a_cs": 1.18, "b_cs": 0.361,"c_cs": 0.410,
                    "R0_cs": 2.96, "B_TF":  5.65, "B_cs":  8.9, "σ_CS": 547e6,
                    "config": "Wedging", "SupraChoice": "Nb3Sn", "T_CS": 4.5,
                    "kappa": 1.95, "J_wost": 45e6, "Gap": 0.015, "H_CS": 6.34},
        "EAST":    {"Ψplateau":  10,  "a_cs": 0.45, "b_cs": 0.15, "c_cs": 0.25,
                    "R0_cs": 1.85, "B_TF":  5.8,  "B_cs":  4.5, "σ_CS": 547e6,
                    "config": "Wedging", "SupraChoice": "NbTi",  "T_CS": 4.5,
                    "kappa": 1.9,  "J_wost": 45e6, "Gap": 0.29, "H_CS": 2.75},
        "ARC":     {"Ψplateau": 19.2, "a_cs": 1.10, "b_cs": 0.89, "c_cs": 0.64,
                    "R0_cs": 3.30, "B_TF": 23,   "B_cs": 13,   "σ_CS": 1000e6,
                    "config": "Plug",    "SupraChoice": "REBCO",  "T_CS": 20,
                    "kappa": 1.84,  "J_wost": 120e6},
        "SPARC":   {"Ψplateau": 25.2, "a_cs": 0.57, "b_cs": 0.18, "c_cs": 0.35,
                    "R0_cs": 1.85, "B_TF": 20,   "B_cs": None, "σ_CS": 1000e6,
                    "config": "Bucking", "SupraChoice": "REBCO",  "T_CS": 20,
                    "kappa": 1.97, "J_wost": 120e6},
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
        ("refined", f_CS_refined, "#2196F3"),
        ("CIRCE",    f_CS_CIRCE, "#F44336"),
    ]

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

            # Override Gap and H_CS if specified per machine
            cfg.Gap = p.get("Gap", 0.10)
            cfg.H_CS = p.get("H_CS", None)

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
        "Plasma_geometry": "refined",
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