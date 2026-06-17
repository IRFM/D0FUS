"""
Physical functions definition for the D0FUS - Design 0-dimensional for FUsion Systems project
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Imports

# All standard / third-party imports are centralised in D0FUS_import.py,
# including the NumPy floating-point error policy that replaces the former
# module-wide RuntimeWarning filter (see the note there).

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

if __name__ == "__main__":
    # ════════════════════════════════════════════════════════════════════
    # Executable verification suite
    # ────────────────────────────────────────────────────────────────────
    # Active only under direct execution:
    #     python D0FUS_BIB/D0FUS_physical_functions.py
    #
    # Each function group below is followed by a __main__ block that
    # (i)  re-evaluates the published anchors of its functions, and/or
    # (ii) advances ONE step of a single coherent ITER chain, the same
    #      machine from A to Z, mirroring the production solver of
    #      D0FUS_EXE/D0FUS_run.py on the shipped deck
    #      D0FUS_INPUTS/1_run_ITER.txt.
    #
    # Every block ends with a uniform comparison table and a PASS/FAIL
    # verdict (_bench below). A global summary closes the file and exits
    # with a non-zero status if any check failed, so the suite can serve
    # as a regression guard.
    # ════════════════════════════════════════════════════════════════════

    _BENCH = {"checks": 0, "failed": [], "blocks": 0}

    def _fmt(v):
        """Compact numeric formatting for the verification tables."""
        if isinstance(v, str):
            return v
        if v is None:
            return "-"
        av = abs(v)
        if av != 0.0 and (av < 1e-3 or av >= 1e5):
            return f"{v:.3e}"
        return f"{v:.4g}"

    def _bench(title, rows, notes=()):
        """Print a uniform verification table with a PASS/FAIL verdict.

        Parameters
        ----------
        title : str
            Block title, printed as a section header.
        rows : list of tuple
            (quantity, value, reference, rel_tol, source) with:
            - rel_tol = float : checked row, |value/reference - 1| < rel_tol;
            - reference = (lo, hi) tuple : checked row, lo <= value <= hi;
            - rel_tol = None (and reference not a tuple) : informative row,
              printed but not counted.
        notes : iterable of str, optional
            Footnotes printed under the table.
        """
        _BENCH["blocks"] += 1
        print(f"\n-- {title} " + "-" * max(2, 92 - len(title)))
        print(f"  {'Quantity':<36} {'D0FUS':>11} {'Reference':>11} "
              f"{'D [%]':>8} {'tol':>6}  Source")
        print("  " + "-" * 94)
        n_pass, n_chk = 0, 0
        for qty, val, ref, tol, src in rows:
            if tol is None and not isinstance(ref, tuple):
                print(f"  {qty:<36} {_fmt(val):>11} {_fmt(ref):>11} "
                      f"{'-':>8} {'info':>6}  {src}")
                continue
            n_chk += 1
            _BENCH["checks"] += 1
            if isinstance(ref, tuple):
                lo, hi = ref
                ok = bool(lo <= val <= hi)
                dev_s, tol_s = ("in" if ok else "OUT"), "range"
                ref_s = f"{_fmt(lo)}..{_fmt(hi)}"
            else:
                dev = val / ref - 1.0
                ok = bool(abs(dev) < tol)
                dev_s = f"{dev * 100:+.2f}"
                tol_s = f"{tol * 100:g}%"
                ref_s = _fmt(ref)
            if ok:
                n_pass += 1
            else:
                _BENCH["failed"].append((title, qty))
            print(f"  {qty:<36} {_fmt(val):>11} {ref_s:>11} "
                  f"{dev_s:>8} {tol_s:>6}  {src:<22} "
                  f"{'ok' if ok else '** FAIL **'}")
        for note in notes:
            print(f"  . {note}")
        if n_chk:
            verdict = "PASSED" if n_pass == n_chk else "** FAILED **"
            print(f"  -> {verdict} ({n_pass}/{n_chk} checks within tolerance)")

    def _bench_summary():
        """Global verdict over all blocks; non-zero exit on any failure."""
        print("\n" + "=" * 96)
        if not _BENCH["failed"]:
            print(f"ALL TESTS PASSED - {_BENCH['checks']} checks "
                  f"in {_BENCH['blocks']} blocks")
        else:
            print(f"{len(_BENCH['failed'])} CHECK(S) FAILED "
                  f"out of {_BENCH['checks']}:")
            for blk, qty in _BENCH["failed"]:
                print(f"  - [{blk}] {qty}")
            raise SystemExit(1)

    # ────────────────────────────────────────────────────────────────────
    # Shared ITER Q=10 reference case - single source of truth for the
    # chain blocks below.
    #
    # ITER   : deck inputs, mirroring D0FUS_INPUTS/1_run_ITER.txt
    #          (benchmark conventions: peak TF field referenced at the
    #          winding-pack front face, published inboard stack
    #          b = 1.10 m, Tbar solved by resolve_Tbar for the Greenwald
    #          fraction f_GW = 0.85). The dict is progressively enriched
    #          with the chain results (V, nbar, pbar, P_rad, tau_E, Ip...)
    #          so that every block consumes the outputs of the previous
    #          ones, exactly like the production solver.
    # FROZEN : converged outputs of that deck (frozen 2026-06),
    #          re-asserted by the final full-deck regression block. Chain
    #          blocks read FROZEN only (i) to check consistency, and
    #          (ii) as forward references where the file order places a
    #          function after its first use (q95 for the bootstrap and
    #          SOL blocks, I_Ohm for the ohmic-power block, f_alpha and
    #          f_imp_dil for the density block); each forward reference
    #          is then independently re-derived and closed by the chain
    #          block that follows the corresponding definition.
    #
    # Published anchors: Shimada et al., Nucl. Fusion 47 (2007) S1
    # (machine values) and Kim et al., Nucl. Fusion 58 (2018) 056013
    # (flat-top heating mix and profiles), as documented in the deck.
    # ────────────────────────────────────────────────────────────────────
    ITER = dict(
        # Geometry and field (deck section 1)
        R0=6.2, a=2.0, b=1.10, Bmax_TF=10.60,
        # Power and operation
        P_fus=500.0,
        P_NBI=33.0, P_ECRH=6.7, P_ICRH=10.0, P_LH=0.0,   # flat-top mix [Kim 2018]
        P_aux=33.0 + 6.7 + 10.0,                          # = 49.7 MW
        Tbar=7.754,            # solved by resolve_Tbar for f_GW_target = 0.85
        H=1.0, M=2.5,
        # Profiles (deck section 2c)
        nu_n=0.01, nu_T=2.80,
        rho_ped=0.95, n_ped_frac=0.99, T_ped_frac=0.55,
        # Composition and radiation (deck section 4)
        Zeff=1.65, imp={'W': 2e-5, 'Ne': 7e-3}, r_synch=0.6,
        rho_rad_core=0.75, C_Alpha=5.0,
        # Current-drive deposition (deck section 11)
        A_beam=2, E_beam_keV=1000.0, rho_NBI=0.30, rho_EC=0.40,
        angle_NBI_deg=20.0,
    )

    FROZEN = dict(
        kappa=1.8792, kappa95=1.67785, delta=0.527517, delta95=0.351678,
        V=847.587, S=687.331,
        f_alpha=0.0297618, f_imp_dil=0.07103,
        nbar=0.98561, nbar_line=1.01245, pbar=0.267547, nG=1.19112,
        B0=5.300, W_th=340.154, betaT=0.023938, betaP=0.651514, betaN=1.63515,
        P_Ohm=0.3910, I_Ohm=8.52244,
        P_Brem=13.3005, P_syn=3.65678, P_line_core=24.9375, P_line=44.8641,
        tauE=3.14385, Ip=14.9681, q95=3.59843, Ib=4.81601,
        eta_LH=0.310283, eta_EC=0.0463759, eta_NBI=0.292349, I_CD=1.62962,
        Q=9.98184, P_sep=87.8792, P_LH_th=73.522,
        B_pol=0.715156, lambda_q_mm=1.12391, Gamma_n=0.58196,
        tau_alpha=15.7193,
    )

#%% Geometry formulas

# Explicit scipy import for the three-point shaping profiles below.
# Kept local to this section so the rest of the module is unaffected.
"""
Tokamak plasma cross-section geometry for D0FUS
=====================================================================

Definitions
-----------
The plasma poloidal cross-section is parameterised by three global shape
descriptors, all defined at the last closed flux surface (LCFS, ρ = 1):

    κ  (kappa)  Elongation: ratio of plasma half-height b to minor radius a.
                κ = b/a

    δ  (delta)  Triangularity: normalised horizontal displacement of the plasma
                extrema.  δ = (R₀ − R_top) / a where R_top is the major radius
                at the upper plasma tip.

    ρ           Normalised radial label: ρ = r/a ∈ [0, 1].

Miller parameterisation of flux surfaces (Miller et al. 1998):
    R(ρ,θ) = R₀ + ρa cos(θ + arcsin(δ(ρ)) sinθ)
    Z(ρ,θ) = κ(ρ) ρa sinθ

Geometry models
---------------
'Academic'
    κ(ρ) = κ_edge = const,  δ(ρ) = 0
    Cylindrical-torus approximation.  Consistent with PROCESS (Kovari 2014).

'refined'
    Three-point PCHIP interpolation (Fritsch & Carlson 1980) through the
    control nodes (ρ=0, ρ=ρ₉₅, ρ=1) for both κ(ρ) and δ(ρ):
      κ(ρ): nodes (0, κ₉₅), (ρ₉₅, κ₉₅), (1, κ_edge);
      δ(ρ): nodes (0, 0),    (ρ₉₅, δ₉₅), (1, δ_edge).
    PCHIP guarantees C¹ continuity globally and preserves monotonicity
    between control points (no overshoot).  The duplicated κ₉₅ value at
    the first two nodes forces a flat core kappa(ρ) ≡ κ₉₅ for ρ ≤ ρ₉₅
    (the Fritsch-Carlson rule sets the interior slope to zero whenever
    one of the neighbour secants vanishes).  The values at ρ₉₅ are
    taken from empirical scalings (ITER 1989) or from equilibrium
    reconstruction.

Physical grounding and limitations
----------------------------------
Two analytical results constrain the on-axis behaviour of the shaping
profiles:

  1. Greene & Johnson (1961) showed via Taylor expansion of ψ around the
     magnetic axis that flux surfaces near the axis are necessarily
     elliptical: only the m=2 harmonic survives at lowest order, while
     all higher-order harmonics (in particular triangularity) must vanish
     on axis.  The on-axis elongation κ(0) = sqrt(B/A) is set
     self-consistently by the equilibrium and is NOT constrained to 1.

  2. Ball & Parra (2015) extended this argument and clarified the
     dependence on the toroidal current profile.  Among all cylindrical
     harmonics, elongation alone can be preserved unchanged from the
     boundary to the core, exactly so for a flat current profile.
     Triangularity decreases monotonically from the LCFS down to
     δ(0) = 0, and its detailed radial profile depends on j_φ(ρ).

These two results fix the asymptotic behaviour at ρ = 0 but leave the
full radial profile underdetermined.  D0FUS therefore uses the PCHIP
interpolation as a pragmatic three-point convention, NOT as a rigorous
derivation from Ball & Parra.  Two limitations follow:

  (a) For strongly peaked or hollow current profiles, Ball & Parra
      report that κ(ρ) can vary by up to 25% between the axis and the
      LCFS — a variation the flat-core prescription does not capture.
  (b) The change of curvature of δ(ρ) at ρ₉₅ is an interpolation
      artefact rather than a feature of the underlying equilibrium.

A fully consistent treatment would couple κ(ρ) and δ(ρ) to j_φ(ρ)
through the Grad-Shafranov equation, which is the role of equilibrium
solvers (CHEASE, HELENA, EFIT) and is out of scope for a 0D systems
code.  The present convention is accepted in exchange for simplicity;
its impact on integrated quantities (plasma volume, V'(ρ)) remains
modest because both κ(ρ) and δ(ρ) enter radial integrals weighted by ρ,
which suppress the contribution of the inner region.

References
----------
Greene & Johnson, Phys. Fluids 4, 875 (1961).
Miller et al., Phys. Plasmas 5, 973 (1998).
Ball & Parra, PPCF 57, 035006 (2015).
Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980).
Lao et al., Fusion Sci. Technol. 48, 968 (2005).
Kovari et al., Fusion Eng. Des. 89, 3054 (2014) — PROCESS.
"""

# =============================================================================
# Ellipse perimeter — Ramanujan approximation (used throughout D0FUS)
# =============================================================================

def _ramanujan_perimeter(semi_a, semi_b):
    """
    Ellipse perimeter via Ramanujan's first approximation (1914).

    P ≈ π [3(a+b) - √((3a+b)(a+3b))]

    Accurate to better than 0.04% for any eccentricity. Used consistently
    in D0FUS for all Academic-mode poloidal perimeter calculations (β_P,
    l_i, first-wall surface, L-H threshold, neutron wall load).

    Parameters
    ----------
    semi_a, semi_b : float or ndarray
        Semi-axes of the ellipse [m].

    Returns
    -------
    P : float or ndarray
        Ellipse perimeter [m].

    References
    ----------
    Ramanujan, S. (1914). "Modular equations and approximations to π."
    Quarterly Journal of Mathematics 45, 350–372.
    """
    a = np.asarray(semi_a, dtype=float)
    b = np.asarray(semi_b, dtype=float)
    return np.pi * (3.0*(a + b) - np.sqrt((3.0*a + b)*(a + 3.0*b)))


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
        Vertical stability margin parameter (Wenninger scaling only).

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
    Elongation radial profile : three-point PCHIP interpolation.

    Control points
    --------------
        (0,      kappa_95)
        (rho_95, kappa_95)
        (1,      kappa_edge)

    The duplicated value at the first two nodes forces a flat core
    kappa(rho) ≡ kappa_95 for rho ≤ rho_95 (the Fritsch-Carlson rule
    sets the interior slope to zero whenever a neighbour secant
    vanishes), and a smooth monotone rise to kappa_edge over the edge
    layer [rho_95, 1].

    Physical context
    ----------------
    Ball & Parra (2015) show that elongation is the only cylindrical
    harmonic that can be preserved unchanged from the boundary to the
    core, exactly so for a flat toroidal current profile (their Sec. 3
    and Fig. 4(a-c)).  For strongly peaked or hollow current profiles,
    however, kappa(rho) varies monotonically across the full radius and
    can change by up to ~25% between the axis and the LCFS.  The
    flat-core prescription used here is therefore a convention valid in
    the limit of moderately peaked current profiles, NOT a rigorous
    consequence of Ball & Parra: it is an interpolation between two
    user-supplied control values (kappa_95 and kappa_edge), with rho_95
    acting as a third anchor rather than a physical boundary.

    Numerical properties
    --------------------
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is C^1
    continuous globally, including at the interior break point rho_95.
    The first derivative is therefore continuous through the Miller
    Jacobian, which is sufficient for a clean dA_per_drho and a smooth
    q(rho).  The second derivative is allowed to jump at rho_95 and
    rho = 1; this has no observable consequence on integrated 0D
    quantities (volume, V', poloidal arc length).

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
    Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980).
    Ball & Parra, PPCF 57, 035006 (2015) — radial penetration of shaping.
    """
    rho = np.asarray(rho, dtype=float)
    interp = PchipInterpolator(
        [0.0,      rho_95,   1.0       ],
        [kappa_95, kappa_95, kappa_edge],
    )
    return interp(rho)


def delta_profile(rho, delta_edge, delta_95, rho_95=0.95):
    """
    Triangularity radial profile : three-point PCHIP interpolation.

    Control points
    --------------
        (0,      0)         — exact on-axis constraint
        (rho_95, delta_95)  — empirical 95%-surface value (e.g. ITER 1989)
        (1,      delta_edge)— prescribed LCFS triangularity

    Physical context
    ----------------
    The on-axis constraint delta(0) = 0 is exact: Greene & Johnson
    (1961) and Ball & Parra (2015) show via Taylor expansion of psi
    around the magnetic axis that all Fourier harmonics of order
    m >= 3 must vanish on axis.  Ball & Parra (2015, Fig. 4(d-f))
    further show that the radial profile of triangularity decreases
    monotonically from the LCFS down to delta(0) = 0, with the
    detailed shape depending on the toroidal current profile.

    Convention used here
    --------------------
    A fully consistent profile would require coupling delta(rho) to
    j_phi(rho) through the Grad-Shafranov equation.  D0FUS instead
    uses a pragmatic three-point PCHIP interpolation.  The intermediate
    point delta_95 acts as an interpolation control rather than a
    physical milestone.  The change of curvature of delta(rho) at
    rho_95 is therefore an artefact of the three-point interpolation,
    not a feature of the underlying equilibrium.

    Numerical properties
    --------------------
    PCHIP guarantees C^1 continuity globally and preserves monotonicity
    between control points (no overshoot).  Negative-triangularity
    configurations (delta_edge < 0, delta_95 < 0) are handled
    identically: PCHIP simply returns a monotonically decreasing
    profile from delta(0) = 0 to delta_edge.

    Parameters
    ----------
    rho : float or ndarray
        Normalised radial coordinate r/a in [0, 1].
    delta_edge : float
        Triangularity at the LCFS (rho = 1).  Negative values are
        supported (negative-triangularity configurations).
    delta_95 : float
        Triangularity at rho_95.  Use f_Delta_95(delta_edge) for the
        ITER 1989 scaling.
    rho_95 : float  default 0.95

    Returns
    -------
    delta : float or ndarray

    References
    ----------
    Greene & Johnson, Phys. Fluids 4, 875 (1961).
    Ball & Parra, PPCF 57, 035006 (2015).
    Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980).
    """
    rho = np.asarray(rho, dtype=float)
    interp = PchipInterpolator(
        [0.0, rho_95,   1.0       ],
        [0.0, delta_95, delta_edge],
    )
    return interp(rho)


def miller_RZ(rho, theta, R0, a, kappa_edge, delta_edge,
              kappa_95=None, delta_95=None, rho_95=0.95):
    """
    (R, Z) coordinates of Miller-parameterised flux surfaces.

    R(ρ,θ) = R₀ + ρa · cos(θ + arcsin(δ(ρ)) sinθ)
    Z(ρ,θ) = κ(ρ) · ρa · sinθ

    Shaping profiles:
      Academic : κ = κ_edge (const),  δ = 0
      refined  : PCHIP κ (flat core) + PCHIP δ (edge-peaked)

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
    # Miller flux surfaces — Academic vs refined +/- delta
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_miller_surfaces(R0=6.2, a=2.0, kappa_edge=1.85, delta_edge=0.50)

def precompute_Vprime(R0, a, kappa_edge, delta_edge,
                      geometry_model='refined',
                      kappa_95=None, delta_95=None, rho_95=0.95,
                      N_rho=200, N_theta=200):
    """
    Precompute V'(rho), dA_pol/drho, L_p(rho), and <1/R^2>(rho).

    Called once per design point; the returned tuple is passed to all
    functions that need volume integrals, poloidal cross-section area
    weights, poloidal arc lengths, or the rigorous local q formula.  All
    four geometric quantities are derived consistently:
      'Academic' : closed-form ellipse at constant kappa, no triangularity.
      'refined'  : numerical Jacobian of the Miller parameterisation,
                   including the full effect of kappa(rho) and delta(rho).

    The flux-surface-perimeter average <1/R^2>(rho) is computed as

        <1/R^2>(rho) = (1/Lp) * oint dl_p / R^2

    and is used by f_q_profile_refined to evaluate the rigorous local q
    formula

        q(rho) = F * Lp^2 * <1/R^2> / (2 pi mu0 I_enc)

    derived from the axisymmetric definition q = (F/2pi) oint dl_p/(R^2 B_p)
    under the assumption B_p ~ mu0 I_enc / Lp constant around the surface.

    Parameters
    ----------
    R0, a : float  [m]
    kappa_edge, delta_edge : float
    geometry_model : 'Academic' | 'refined'
        'Academic' : ellipse at kappa = kappa_edge, no triangularity
        'refined'  : Miller PCHIP profiles kappa(rho), delta(rho); 2D Jacobian
    kappa_95, delta_95 : float or None  (ITER 1989 defaults; refined only)
    rho_95 : float  default 0.95
    N_rho, N_theta : int  grid resolution

    Returns
    -------
    rho_grid    : ndarray  (N_rho,)
    Vprime      : ndarray  (N_rho,)  [m^3]   dV/drho on rho_grid
    V_total     : float              [m^3]   total plasma volume
    dA_grid     : ndarray  (N_rho,)  [m^2]   dA_pol/drho, poloidal
                                             cross-section area element
                                             (includes kappa(rho) and delta(rho)
                                             in refined mode; exact ellipse in
                                             Academic mode).
    Lp_grid     : ndarray  (N_rho,)  [m]     true poloidal arc length
        In Academic mode, Lp_grid is the Ramanujan perimeter of the
        elliptical flux surface (exact at constant kappa, no delta).
        In refined mode, Lp_grid is the numerical contour integral
        including delta(rho).
    inv_R2_grid : ndarray  (N_rho,)  [m^-2]  <1/R^2>(rho), perimeter average
        Captures the finite aspect ratio (a/R0) and triangularity effect
        on the local safety factor.  At rho = 0, <1/R^2> = 1/R0^2 by
        continuity; at the LCFS, it differs from 1/R0^2 by ~10-20% at
        ITER aspect ratio.

    Notes
    -----
    The flux-surface-perimeter average <1/R^2>(rho) is the geometric
    average along the poloidal contour, not the Grad-Shafranov flux-
    surface average <X> = oint X dl_p/B_p / oint dl_p/B_p.  The two
    coincide when B_p is constant around the surface (low-beta, no
    Shafranov shift), which is the standard 0D approximation.  Higher-
    fidelity 1.5D codes (METIS, ASTRA) carry the full GS-weighted
    average; the residual discrepancy is typically a few percent for
    ITER and EU-DEMO baselines and is documented in the q-profile
    builder docstring.
    """
    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa_edge)
    if delta_95 is None:
        delta_95 = f_Delta_95(delta_edge)

    rho_grid = np.linspace(1e-6, 1.0, N_rho)

    if geometry_model == 'Academic':
        # Closed-form ellipse at constant kappa, no triangularity.
        # All weights are the exact analytical limits of the Miller
        # Jacobian formulation below when delta = 0 and kappa' = 0.
        Vprime  = 4.0 * np.pi**2 * R0 * a**2 * kappa_edge * rho_grid
        V_total = 2.0 * np.pi**2 * R0 * a**2 * kappa_edge
        # Poloidal area element: A_pol(rho) = pi(rho a)^2 kappa
        # => dA_pol/drho = 2 pi rho a^2 kappa
        dA_grid = 2.0 * np.pi * rho_grid * a**2 * kappa_edge
        # Ramanujan perimeter of each elliptical flux surface (rho a, kappa rho a)
        Lp_grid = _ramanujan_perimeter(rho_grid * a, kappa_edge * rho_grid * a)

        # Flux-surface-perimeter average <1/R^2>(rho), integrated numerically
        # on a fine theta grid for each rho.  Closed form for an ellipse
        # exists but is more cumbersome than the direct quadrature below.
        theta = np.linspace(0.0, 2.0*np.pi, max(N_theta, 200), endpoint=False)
        dtheta = theta[1] - theta[0]
        inv_R2_grid = np.zeros_like(rho_grid)
        inv_R2_grid[0] = 1.0 / R0**2     # axis limit
        for i in range(1, len(rho_grid)):
            r = rho_grid[i]
            R = R0 + r * a * np.cos(theta)
            Z = kappa_edge * r * a * np.sin(theta)
            dRdt = -r * a * np.sin(theta)
            dZdt =  kappa_edge * r * a * np.cos(theta)
            ds = np.sqrt(dRdt**2 + dZdt**2) * dtheta
            inv_R2_grid[i] = float(np.sum(ds / R**2)) / float(np.sum(ds))

        return (rho_grid, Vprime, V_total, dA_grid, Lp_grid, inv_R2_grid)

    elif geometry_model == 'refined':
        theta  = np.linspace(0.0, 2.0*np.pi, N_theta, endpoint=False)
        dtheta = theta[1] - theta[0]
        drho   = rho_grid[1] - rho_grid[0]

        RHO, THETA = np.meshgrid(rho_grid, theta, indexing='ij')
        R, Z = miller_RZ(RHO, THETA, R0, a, kappa_edge, delta_edge,
                         kappa_95, delta_95, rho_95)

        # Numerical 2D Jacobian |d(R,Z)/d(rho,theta)|, used for V' and dA.
        dR_drho   = np.gradient(R, drho,   axis=0)
        dR_dtheta = np.gradient(R, dtheta, axis=1)
        dZ_drho   = np.gradient(Z, drho,   axis=0)
        dZ_dtheta = np.gradient(Z, dtheta, axis=1)
        jac = np.abs(dR_drho * dZ_dtheta - dR_dtheta * dZ_drho)

        # Toroidal volume derivative: dV/drho = 2 pi oint R |J2D| dtheta
        Vprime  = np.sum(2.0 * np.pi * R * jac, axis=1) * dtheta
        V_total = float(np.trapezoid(Vprime, rho_grid))

        # Poloidal cross-section area element: dA_pol/drho = oint |J2D| dtheta
        dA_grid = np.sum(jac, axis=1) * dtheta

        # True poloidal arc length:
        #   L_p(rho) = oint sqrt((dR/dtheta)^2 + (dZ/dtheta)^2) dtheta
        arc_element = np.sqrt(dR_dtheta**2 + dZ_dtheta**2)
        Lp_grid = np.sum(arc_element, axis=1) * dtheta

        # Flux-surface-perimeter average <1/R^2>(rho):
        #   <1/R^2> = (oint dl_p / R^2) / Lp
        # Lp_grid > 0 except at rho = 0 where the contour degenerates;
        # set the axial limit by continuity to 1/R0^2.
        Lp_safe = np.where(Lp_grid > 1e-12, Lp_grid, 1.0)
        inv_R2_grid = np.sum(arc_element / R**2, axis=1) * dtheta / Lp_safe
        inv_R2_grid[Lp_grid <= 1e-12] = 1.0 / R0**2

        return (rho_grid, Vprime, V_total, dA_grid, Lp_grid, inv_R2_grid)

    else:
        raise ValueError(f"Unknown geometry_model '{geometry_model}'. "
                         "Valid: 'Academic', 'refined'.")



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


def interpolate_dA(rho, rho_grid, dA_grid):
    """
    Interpolate the precomputed poloidal cross-section area element
    dA_pol/dρ at arbitrary radial positions.

    In refined mode, dA_grid is the contour integral ∮|J2D|dθ of the
    Miller Jacobian, consistent with Vprime and including the full
    effect of κ(ρ) and δ(ρ).
    In Academic mode, dA_grid is the exact ellipse expression
    2πρa²κ_edge.

    Parameters
    ----------
    rho : float or ndarray
    rho_grid, dA_grid : ndarray from precompute_Vprime()

    Returns
    -------
    dA_per_drho : float or ndarray  [m²]
    """
    return np.interp(rho, rho_grid, dA_grid)


# =============================================================================
# Plasma volume
# =============================================================================

def f_plasma_volume(R0, a, kappa, delta, Vprime_data=None):
    """
    Plasma volume [m³].

    Modes:
      Vprime_data = None           → O(δ²) analytical Miller formula
      Vprime_data = (ρ, V', V, …) → precomputed from precompute_Vprime()

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

      'refined'  : Numerical revolution of the LCFS Miller contour
                   S   = 2π ∫₀²π R(θ) · |dl/dθ| dθ
                   with |dl/dθ| = √((∂R/∂θ)² + (∂Z/∂θ)²)  at ρ = 1

    Parameters
    ----------
    R0, a : float  [m]
    kappa_edge : float  [-]
    delta_edge : float  [-]  (ignored for 'Academic')
    geometry_model : 'Academic' | 'refined'
    N_theta : int  poloidal resolution (refined)

    Returns
    -------
    S : float  [m²]

    """
    if geometry_model == 'Academic':
        Pe = _ramanujan_perimeter(a, kappa_edge * a)
        return 2*np.pi * R0 * Pe

    elif geometry_model == 'refined':
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
                         "Valid: 'Academic', 'refined'.")


if __name__ == "__main__":
    # First wall surface area — Academic vs refined Miller
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_first_wall_surface(R0=6.2, a=2.0)

if __name__ == "__main__":
    # ── ITER chain (1/12) - shaping and geometry ─────────────────────────
    # Deck options: Option_Kappa = 'Wenninger' (shaping derived from the
    # aspect ratio, NOT imposed), refined Miller geometry on the
    # production grid (N_rho = 500, N_theta = 200).
    # Published anchor: V = 831 m3 inside the true separatrix (Shimada
    # 2007, Table 2); the analytic shaped torus carries a known +1.5 %
    # bias at the published shaping (kappa = 1.85, delta = 0.485),
    # tolerance 6 %.
    _k = f_Kappa(ITER['R0'] / ITER['a'], 'Wenninger', 1.85, 0.3)
    _k95 = f_Kappa_95(_k)
    _d = f_Delta(_k)
    _d95 = f_Delta_95(_d)
    ITER_Vpd = precompute_Vprime(ITER['R0'], ITER['a'], _k, _d,
                                 geometry_model='refined',
                                 kappa_95=_k95, delta_95=_d95,
                                 N_rho=500, N_theta=200)
    _V = f_plasma_volume(ITER['R0'], ITER['a'], _k, _d, Vprime_data=ITER_Vpd)
    _S = f_first_wall_surface(ITER['R0'], ITER['a'], _k, _d,
                              geometry_model='refined')
    _V_pub = f_plasma_volume(6.2, 2.0, 1.85, 0.485)   # analytic, published shaping
    _kappa_a = _V / (2.0 * np.pi**2 * ITER['R0'] * ITER['a']**2)
    ITER.update(kappa=_k, kappa95=_k95, delta=_d, delta95=_d95,
                V=_V, S=_S, kappa_a=_kappa_a)
    _bench("ITER chain 1/12 - shaping and geometry", [
        ("kappa_sep (Wenninger)", _k, FROZEN['kappa'], 1e-3, "deck frozen"),
        ("kappa_95", _k95, FROZEN['kappa95'], 1e-3, "deck frozen"),
        ("delta_sep", _d, FROZEN['delta'], 1e-3, "deck frozen"),
        ("delta_95", _d95, FROZEN['delta95'], 1e-3, "deck frozen"),
        ("V analytic, published shaping [m3]", _V_pub, 831.0, 0.06, "Shimada 2007"),
        ("V Miller, deck shaping [m3]", _V, FROZEN['V'], 5e-3, "deck frozen"),
        ("V Miller, deck shaping [m3]", _V, 831.0, 0.06, "Shimada 2007"),
        ("S first wall, Miller [m2]", _S, FROZEN['S'], 5e-3, "deck frozen"),
        ("kappa_area = V/(2 pi^2 R0 a^2)", _kappa_a, None, None, "IPB98 convention"),
    ], notes=[
        "Published separatrix shaping: kappa = 1.85, delta = 0.485 "
        "(Shimada 2007, Table 2); the Wenninger scaling at A = 3.1 "
        "returns kappa = 1.879, delta = 0.528.",
    ])


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

  refined  (Vprime_data = (rho_grid, Vprime, V_total) from precompute_Vprime())
      Volume element: dV = V'_Miller(ρ) dρ  (full Miller geometry, PCHIP κ/δ)
      All integrals use the precomputed Jacobian; n(ρ) and T(ρ) are midplane
      radial profiles — the geometry enters only through the volume weight.

Profile model
-------------
Parabola-with-pedestal (Lackner 1990, used in ITER Physics Basis 1999):

  X(ρ) = X_ped + (X₀ − X_ped) · (1 − (ρ/ρ_ped)²)^ν     ρ ≤ ρ_ped
  X(ρ) = X_ped · (1 − ρ)/(1 − ρ_ped)                   ρ > ρ_ped  (X_sep = 0)

Special case ρ_ped = 1, f_ped = 0 → purely parabolic: X(ρ) = X̄(1+ν)(1−ρ²)^ν.

ITER reference values (used in __main__ tests)
-----------------------------------------------
  R₀ = 6.2 m,  a = 2.0 m,  κ = 1.85,  δ = 0.50,  Ip = 15 MA
  P_fus = 500 MW,  T̄ = 8.9 keV,  n̄ ≈ 1.01 × 10²⁰ m⁻³

References
----------
Lackner, Comments Plasma Phys. Control. Fusion 13, 163 (1990)
ITER Physics Basis, Nucl. Fusion 39, 2175 (1999)
Bosch & Hale, Nucl. Fusion 32, 611 (1992)
Wesson, Tokamaks, 4th ed., Oxford (2011)
Greenwald, PPCF 44, R27 (2002)
Martin et al., JPCS 123, 012033 (2008)
"""


# Module-level cache for _profile_core_peak
_profile_core_peak_cache = {}


def _profile_core_peak(nu, rho_ped, f_ped, Vprime_data=None):
    """
    Core-peak normalised value X₀/X̄  for a parabola-with-pedestal profile.

    The volume-average constraint ⟨X⟩_vol = X̄  is enforced using:
      - cylindrical weight  w(ρ) = 2ρ          when Vprime_data is None
      - Miller weight  w(ρ) = V'(ρ)/V_total    when Vprime_data is provided

    This ensures that the profile normalisation is consistent with
    whichever volume element is used by downstream integrals.

    Parameters
    ----------
    nu      : float  Core peaking exponent (ν_n or ν_T).
    rho_ped : float  Normalised pedestal radius ∈ (0, 1].
    f_ped   : float  X_ped / X̄  (pedestal fraction of the volume average).
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
      cylindrical weight 2ρ dρ.  When Vprime_data is provided (refined),
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

    Limitation — triangularity correction (refined mode)
    --------------------------------------------------
    In Miller geometry the outer midplane (θ = 0) gives:
        R(ρ, θ=0) = R₀ + ρa · cos(arcsin(δ(ρ)))
        dR/dρ     = a · cos(arcsin(δ(ρ)))    [≠ a when δ(ρ) ≠ 0]

    The exact chord integral is therefore:
        n̄_line = ∫₀¹ n(ρ) · cos(arcsin(δ(ρ))) dρ   (refined mode)

    This function uses the δ = 0 approximation for both geometry modes.
    Direct evaluation of the chord integral on the PCHIP δ(ρ) profile
    gives a relative error on n̄_line below 1 % for ITER-class shaping
    (δ_edge ≈ 0.5) with moderately peaked density profiles (ν_n ≳ 0.5),
    and below 4 % for very high triangularity (δ_edge ≈ 0.8) combined
    with a flat density profile.  The correction is negligible in all
    ITER and EU-DEMO relevant operating points.

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

    Parameterisation of Bosch & Hale (1992) for the T(d,n)⁴He reaction
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
    Bosch & Hale, Nucl. Fusion 32, 611 (1992) — Table VII, DT branch.
    """
    # Bosch & Hale fit coefficients for DT (Table VII)
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


if __name__ == "__main__":
    # ── Published anchors - D-T reactivity (Bosch & Hale 1992) ──────────
    # Bosch & Hale, NF 32 (1992) 611, Table VII fit. Coefficients
    # cross-checked against cfspopcon 8.0.0 (machine precision); values
    # consistent with Table VIII (1.136e-16 cm3/s at 10 keV). The peak of
    # the reactivity sits near 64 keV (Wesson, Tokamaks).
    _sv_rows = [(f"<sigma v> at {_T:g} keV [m3/s]", f_sigmav(_T), _sv, 5e-4,
                 "B&H 1992 Tab. VIII")
                for _T, _sv in ((1.0, 6.8569e-27), (10.0, 1.1362e-22),
                                (20.0, 4.3302e-22))]
    _Tg = np.linspace(20., 150., 600)
    _sv_rows.append(("peak position [keV]",
                     float(_Tg[np.argmax(f_sigmav(_Tg))]),
                     (55.0, 75.0), 1, "Wesson 2004"))
    _bench("Published anchors - D-T reactivity (Bosch-Hale)", _sv_rows)

# =============================================================================
# Required electron density for a target fusion power
# =============================================================================

def f_nbar(P_fus, nu_n, nu_T, f_alpha, Tbar, R0, a, kappa,
           rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
           Vprime_data=None, f_imp=0.0, tau_i_e=1.0):
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
      refined    : w(ρ) = V'_Miller(ρ)/V_Miller  (full Miller geometry)

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
        # refined mode: Miller V'(ρ) integration
        rho_grid, Vprime, V_total = Vprime_data[:3]  # safe: 5-tuple (rho, V', V, dA, Lp)
        T_arr   = f_Tprof(Tbar, nu_T, rho_grid, rho_ped, T_ped_frac,
                          Vprime_data)
        n_hat   = f_nprof(1.0,  nu_n, rho_grid, rho_ped, n_ped_frac,
                          Vprime_data)
        # DT reactivity is governed by the ion temperature: T_i = tau_i_e * T_e.
        # T_arr is the electron profile; scale it to the ion profile here.
        sv_arr  = f_sigmav(tau_i_e * T_arr)
        # Guard against NaN/inf at ρ→1 where T→0 and ⟨σv⟩→0 rapidly
        integrand = np.nan_to_num(sv_arr * n_hat**2 * Vprime,
                                   nan=0.0, posinf=0.0)
        # Normalised reactivity integral I_fus  [m³ s⁻¹]
        I_fus = np.trapezoid(integrand, rho_grid) / V_total
        V     = V_total
    else:
        # Academic mode: cylindrical weight 2ρ dρ.
        # Vectorised trapezoid replaces the former scalar quad() call,
        # giving a ~10–50x speedup in the hot solver loop with negligible
        # loss of accuracy (<0.1 % for smooth DT profiles on 200 points).
        _rho_acad = np.linspace(0.0, 1.0, 200)
        _T_acad   = f_Tprof(Tbar, nu_T, _rho_acad, rho_ped, T_ped_frac)
        _n_acad   = f_nprof(1.0,  nu_n, _rho_acad, rho_ped, n_ped_frac)
        # Reactivity uses the ion temperature T_i = tau_i_e * T_e.
        _sv_acad  = f_sigmav(tau_i_e * _T_acad)
        _intgd    = np.nan_to_num(_sv_acad * _n_acad**2 * 2.0 * _rho_acad,
                                   nan=0.0, posinf=0.0)
        I_fus     = float(np.trapezoid(_intgd, _rho_acad))
        V         = 2.0 * np.pi**2 * R0 * kappa * a**2   # Wesson volume

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
           Vprime_data=None, tau_i_e=1.0):
    """
    Volume-averaged plasma pressure  p̄ = (1 + τ_ie)⟨nT⟩_vol  [MPa].

    The total pressure is p = p_e + p_i = n_e T_e + n_i T_i.  With n_i ≈ n_e
    and the prescribed ratio T_i = τ_ie · T_e, the single-temperature prefactor
    2 generalises to (1 + τ_ie); τ_ie = 1 recovers p̄ = 2⟨nT⟩ (T_i = T_e).
    The integral ⟨nT⟩_vol = ∫ n(ρ)·T(ρ)·w(ρ) dρ uses:

      Academic mode:
        Parabolic (analytical): ⟨n̂T̂⟩ = (1+ν_n)(1+ν_T)/(1+ν_n+ν_T)
        Pedestal  (numerical):  cylindrical weight 2ρ dρ
      refined mode:
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
    Vprime_data : tuple or None  Precomputed Miller data (refined mode).

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
        # refined mode: Miller V'(ρ) integration
        rho_grid, Vprime, V_total = Vprime_data[:3]  # safe: 5-tuple (rho, V', V, dA, Lp)
        n_hat = f_nprof(1.0, nu_n, rho_grid, rho_ped, n_ped_frac,
                        Vprime_data)
        T_hat = f_Tprof(1.0, nu_T, rho_grid, rho_ped, T_ped_frac,
                        Vprime_data)
        C_vol = float(np.trapezoid(n_hat * T_hat * Vprime, rho_grid)) / V_total
        # p = p_e + p_i = (1 + tau_i_e) n_e T_e  (n_i ~ n_e, T_i = tau_i_e T_e)
        profile_factor = (1.0 + tau_i_e) * C_vol

    elif rho_ped >= 1.0:
        # Academic mode, parabolic: closed-form analytical result.
        # Prefactor (1 + tau_i_e): p = p_e + p_i with T_i = tau_i_e T_e.
        profile_factor = (1.0 + tau_i_e) * (1.0 + nu_n) * (1.0 + nu_T) / (1.0 + nu_n + nu_T)

    else:
        # Academic mode, pedestal: numerical with cylindrical weight 2ρ dρ
        # C_vol = <n̂ T̂>_vol = 2 ∫₀¹ n̂(ρ) T̂(ρ) ρ dρ   (cylindrical volume average)
        # profile_factor = (1 + tau_i_e) × C_vol   (p = p_e + p_i, T_i = tau_i_e T_e)
        rho_arr = np.linspace(0.0, 1.0, 2000)
        n_hat   = f_nprof(1.0, nu_n, rho_arr, rho_ped, n_ped_frac)
        T_hat   = f_Tprof(1.0, nu_T, rho_arr, rho_ped, T_ped_frac)
        C_vol = 2.0 * float(np.trapezoid(n_hat * T_hat * rho_arr, rho_arr))
        profile_factor = (1.0 + tau_i_e) * C_vol
        
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


def f_n_limit_giacomin(P_sol, R0, a, kappa, B0, q_edge, A_ion=2.0, alpha_GR=3.3):
    """
    First-principles edge density limit — Giacomin et al. (2022), Eq. (12).

    Power-dependent maximum density achievable near the separatrix before
    the edge pressure-gradient collapse leading to MARFE onset and a
    density-limit disruption (L-mode disruptive limit):

        n_lim [10²⁰ m⁻³] = α · A^(1/6) · a^(3/14) · P_SOL^(10/21)
                           · R0^(−43/42) · q^(−22/21) · (1+κ²)^(−1/3) · B_T^(2/3)

    The fitted prefactor α = 3.3 ± 0.3 (R² ≈ 0.8) is a single value across
    the AUG / JET / TCV multi-machine database of the reference paper.

    Parameters
    ----------
    P_sol : float
        Power crossing the separatrix [MW].
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    kappa : float
        Plasma elongation [-].
    B0 : float
        On-axis toroidal field [T].
    q_edge : float
        Edge safety factor [-]. The reference paper uses q ≈ q95 in its
        machine examples (q = 3 for ITER and SPARC, q = 4 for C-Mod).
    A_ion : float, optional
        Main-ion mass number [-]. Default 2.0 (deuterium), the value of
        the fitting database; reproduces the paper predictions exactly
        (ITER: 2.5e20 m⁻³ at P_SOL = 50 MW; SPARC: 8.7e20 at 28 MW).
        Use 2.5 for a D-T plasma (+4% via the A^(1/6) dependence).
    alpha_GR : float, optional
        Fitted numerical prefactor. Default 3.3.

    Returns
    -------
    n_lim : float
        Maximum near-separatrix electron density [10²⁰ m⁻³].

    Notes
    -----
    This is an EDGE density limit (validated against edge Thomson data at
    MARFE onset), NOT a line-averaged one. Comparison with the D0FUS
    line-averaged density requires an edge-to-average conversion, e.g.
    through the n_sep / n̄ anchoring factor f_n_sep (see f_density_limit).
    Derived for L-mode disruptive limits; for an H-mode reactor it bounds
    the post H-L back-transition operating point.

    References
    ----------
    Giacomin M., Pau A., Ricci P., Sauter O., Eich T. et al.,
        Phys. Rev. Lett. 128 (2022) 185003 (arXiv:2204.02911), Eq. (12).
    """
    return (alpha_GR * A_ion**(1.0/6.0) * a**(3.0/14.0) * P_sol**(10.0/21.0)
            * R0**(-43.0/42.0) * q_edge**(-22.0/21.0)
            * (1.0 + kappa**2)**(-1.0/3.0) * B0**(2.0/3.0))


def f_n_limit_zanca(P_tot, Ip, a, Z_eff, Z_i=1.0, f0=0.5):
    """
    Power-balance density limit — Zanca et al. (2019), Eq. (20).

    Radiative-collapse limit on the LINE-AVERAGED density for additionally
    heated L-mode tokamaks, derived from a 1D power balance with impurity
    and neutral radiation losses:

        n_lim [10²⁰ m⁻³] = 0.4 · Z_eff^(4/9) · (f0 + Z_eff − Z_i)^(−5/9)
                           · (P_tot / Ip)^(4/9) · n_GW^(8/9)

    with P_tot the TOTAL heating power [MW] (not P_sep; includes alpha,
    auxiliary and Ohmic heating), Ip in MA, and n_GW = Ip/(πa²) the
    Greenwald density [10²⁰ m⁻³].

    Parameters
    ----------
    P_tot : float
        Total plasma heating power [MW].
    Ip : float
        Plasma current [MA].
    a : float
        Minor radius [m].
    Z_eff : float
        Effective plasma charge [-].
    Z_i : float, optional
        Main-ion charge number [-]. Default 1.0 (hydrogen isotopes).
    f0 : float, optional
        Effective neutral-deuterium concentration parameter [%].
        Default 0.5, the value assumed in the original paper and in
        subsequent multi-machine comparisons.

    Returns
    -------
    n_lim : float
        Maximum line-averaged electron density [10²⁰ m⁻³].

    References
    ----------
    Zanca P., Sattin F., Escande D.F. & JET Contributors,
        Nucl. Fusion 59 (2019) 126011, Eq. (20).
    Manz P., Eich T., Grover O. et al., Nucl. Fusion 63 (2023) 076026,
        Eq. (7) — formula restatement and AUG validation; source of the
        f0 = 0.5 convention.
    """
    nGW = f_nG(Ip, a)
    return (0.4 * Z_eff**(4.0/9.0) * (f0 + Z_eff - Z_i)**(-5.0/9.0)
            * (P_tot / Ip)**(4.0/9.0) * nGW**(8.0/9.0))


def f_density_limit(model, Ip, a, P_sol=None, P_tot=None, R0=None, kappa=None,
                    B0=None, q_edge=None, Z_eff=None, f_n_sep_line=0.20,
                    A_ion=2.0, alpha_GR=3.3, f0=0.5, Z_i=1.0):
    """
    Model-selectable density limit dispatcher.

    Returns the density limit of the selected model expressed as an
    equivalent cap on the LINE-AVERAGED density (the quantity disciplined
    in D0FUS and used by the feasibility constraints), together with the
    native-convention value and metadata.

    Models
    ------
    'greenwald' : n̄_line < n_GW = Ip/(πa²)
                  (Greenwald, PPCF 44, R27 (2002)).
    'giacomin'  : edge limit of Giacomin et al. PRL 128 (2022) 185003,
                  Eq. (12); converted to a line-averaged cap through the
                  edge anchoring n_sep = f_n_sep_line · n̄_line:
                      n̄_line_max = n_lim_edge / f_n_sep_line.
                  Requires P_sol, R0, kappa, B0, q_edge.
    'zanca'     : line-averaged limit of Zanca et al. NF 59 (2019) 126011,
                  Eq. (20). Requires P_tot and Z_eff.

    Parameters
    ----------
    model : str
        'greenwald', 'giacomin' or 'zanca'.
    Ip : float          Plasma current [MA].
    a : float           Minor radius [m].
    P_sol : float       Power crossing the separatrix [MW] (giacomin).
    P_tot : float       Total heating power [MW] (zanca).
    R0, kappa, B0, q_edge : float
        Geometry, on-axis field and edge safety factor (giacomin).
    Z_eff : float       Effective charge (zanca).
    f_n_sep_line : float
        Separatrix-to-line-average density ratio n_sep/n̄_line [-] used to
        convert the edge limit into a line-averaged cap (giacomin only).
    A_ion, alpha_GR, f0, Z_i : float
        Model coefficients (see the individual functions).

    Returns
    -------
    n_limit_line : float
        Equivalent cap on the line-averaged density [10²⁰ m⁻³].
    n_limit_native : float
        Limit in the model native convention [10²⁰ m⁻³]
        (edge density for 'giacomin', line-averaged otherwise).
    convention : str
        'line-averaged' or 'edge (near-separatrix)'.

    Caution
    -------
    The edge-to-line conversion for 'giacomin' uses the OPERATING-POINT
    separatrix anchoring n_sep = f_n_sep_line · n̄_line. Near the density
    limit the edge density profile flattens and n_sep/n̄ rises (typically
    towards 0.4-0.5), so the converted line-averaged cap returned here is
    an OPTIMISTIC upper bound. For a conservative assessment, pass a
    density-limit-relevant f_n_sep_line rather than the nominal one.
    """
    if model == 'greenwald':
        nl = f_nG(Ip, a)
        return nl, nl, 'line-averaged'
    elif model == 'giacomin':
        if any(v is None for v in (P_sol, R0, kappa, B0, q_edge)):
            raise ValueError("f_density_limit('giacomin') requires "
                             "P_sol, R0, kappa, B0 and q_edge.")
        nl_edge = f_n_limit_giacomin(P_sol, R0, a, kappa, B0, q_edge,
                                     A_ion=A_ion, alpha_GR=alpha_GR)
        return nl_edge / f_n_sep_line, nl_edge, 'edge (near-separatrix)'
    elif model == 'zanca':
        if P_tot is None or Z_eff is None:
            raise ValueError("f_density_limit('zanca') requires P_tot and Z_eff.")
        nl = f_n_limit_zanca(P_tot, Ip, a, Z_eff, Z_i=Z_i, f0=f0)
        return nl, nl, 'line-averaged'
    else:
        raise ValueError(f"Unknown density_limit_model '{model}'. "
                         "Choose from: 'greenwald', 'giacomin', 'zanca'.")


if __name__ == "__main__":
    # ── Published anchors - operational density limits ───────────────────
    # Greenwald, PPCF 44 (2002) R27: n_GW = Ip/(pi a^2), exact identity at
    # the ITER 15 MA point. Giacomin et al., PRL 128 (2022) 185003,
    # Eq. 12: the paper's own machine predictions (A = 2) are 2.5e20 for
    # ITER and 8.7e20 for SPARC (rounding of the quoted inputs). Zanca et
    # al., NF 59 (2019) 126011, Eq. 20 (as in Manz, NF 63 (2023) 076026,
    # Eq. 7): algebraic identity at an ITER-like point.
    _zref = (0.4 * 1.7**(4 / 9) * (0.5 + 1.7 - 1)**(-5 / 9) * (50 / 15)**(4 / 9)
             * f_nG(15., 2.)**(8 / 9))
    _bench("Published anchors - density limits", [
        ("n_GW ITER identity [1e20 m-3]", f_nG(15., 2.), 15 / (4 * np.pi),
         1e-12, "Greenwald 2002"),
        ("n_lim Giacomin ITER [1e20 m-3]",
         f_n_limit_giacomin(50., 6.2, 2., 1.8, 5.3, 3.), 2.5, 0.06,
         "Giacomin 2022"),
        ("n_lim Giacomin SPARC [1e20 m-3]",
         f_n_limit_giacomin(28., 1.85, .57, 2., 12.2, 3.), 8.7, 0.06,
         "Giacomin 2022"),
        ("n_lim Zanca identity [1e20 m-3]",
         f_n_limit_zanca(50., 15., 2., Z_eff=1.7), _zref, 1e-9,
         "Zanca 2019"),
    ])

if __name__ == "__main__":
    # ── ITER chain (2/12) - operating point: density and pressure ───────
    # The deck specifies the operating point through the Greenwald
    # fraction (Tbar_mode = greenwald, f_GW_target = 0.85): resolve_Tbar
    # solves Tbar = 7.754 keV, used here as the chain input. Two forward
    # references are taken from FROZEN and closed downstream:
    #   - f_alpha (helium ash fraction), closed by chain 12;
    #   - f_imp_dil = sum_j <Z_j> c_j (fuel dilution by the W + Ne mix),
    #     closed by chain 4 where get_Z_mean is defined.
    _n = f_nbar(ITER['P_fus'], ITER['nu_n'], ITER['nu_T'], FROZEN['f_alpha'],
                ITER['Tbar'], ITER['R0'], ITER['a'], ITER['kappa'],
                rho_ped=ITER['rho_ped'], n_ped_frac=ITER['n_ped_frac'],
                T_ped_frac=ITER['T_ped_frac'],
                Vprime_data=ITER_Vpd, f_imp=FROZEN['f_imp_dil'], tau_i_e=1.0)
    _p = f_pbar(ITER['nu_n'], ITER['nu_T'], _n, ITER['Tbar'],
                rho_ped=ITER['rho_ped'], n_ped_frac=ITER['n_ped_frac'],
                T_ped_frac=ITER['T_ped_frac'],
                Vprime_data=ITER_Vpd, tau_i_e=1.0)
    _nl = f_nbar_line(_n, ITER['nu_n'], ITER['rho_ped'], ITER['n_ped_frac'])
    _nl_flat = f_nbar_line(1.0, 0.0)   # analytic identity for a flat profile
    ITER.update(nbar=_n, pbar=_p, nbar_line=_nl)
    _bench("ITER chain 2/12 - operating point (density, pressure)", [
        ("nbar volume-avg [1e20 m-3]", _n, FROZEN['nbar'], 1e-3, "deck frozen"),
        ("nbar line-avg [1e20 m-3]", _nl, FROZEN['nbar_line'], 1e-3,
         "deck frozen"),
        ("nbar line-avg [1e20 m-3]", _nl, 1.01, 0.01, "Shimada 2007"),
        ("pbar [MPa]", _p, FROZEN['pbar'], 1e-3, "deck frozen"),
        ("flat-profile identity n_line/n_vol", _nl_flat, 1.0, 1e-9,
         "analytic"),
        ("f_GW at frozen Ip", _nl / f_nG(FROZEN['Ip'], ITER['a']), 0.85,
         5e-3, "deck target"),
    ], notes=[
        "f_GW is re-closed in chain 11 with the chain Ip from the "
        "scaling-law inversion.",
    ])

#%% B and beta

def f_B0(Bmax, a, b, R0, b_cover=0.0):
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
    B0 = Bmax * (1.0 - (a + b + b_cover) / R0)
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
            L_pol ≈ πa √(2(1+κ²))   [RMS approximation]
        Ampere's law: B_pol = μ₀ I_p / L_pol.
        Using the circular-plasma relation I_p = π a² B_T / (μ₀ R₀ q) to
        eliminate I_p gives exactly the same result as Route 1:
            B_pol = (a B_T / (R₀ q)) × √((1+κ²)/2)

    Note: this function uses the Route 1 (q-inversion) result directly.
    The √((1+κ²)/2) factor is a physics correction, not a perimeter
    approximation. The Ramanujan perimeter (used elsewhere in D0FUS)
    would give a slightly different Route 2 result (~2%), but Route 1
    is the more fundamental derivation.

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
    Sauter, Fusion Eng. Des. 112 (2016) 633 — shaped q₉₅ formula.
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

    For an ellipse with semi-axes a and κa, the Ramanujan perimeter gives:
        L_pol ≈ π [3(a + κa) − √((3a + κa)(a + 3κa))]

    Note: this is the Ramanujan approximation to the ellipse perimeter,
    consistent with f_first_wall_surface and the li(3) fallback.
    Accurate to better than 0.04% for any eccentricity.

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

    # Effective poloidal circumference — Ramanujan ellipse perimeter
    # (consistent with f_first_wall_surface and li(3) fallback)
    L_pol = _ramanujan_perimeter(a, κ * a)

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


def f_beta_fast_alpha(P_alpha_MW, Te_keV, ne_20, B0, V_m3, Z_eff=1.65,
                      A_DT=2.5):
    """
    Fast-alpha contribution to toroidal beta from slowing-down pressure.

    Fusion-born alpha particles (E_alpha = 3.52 MeV) slow down on thermal
    electrons and ions over a characteristic time tau_se.  During this
    transient, they carry a stored energy W_fast that exerts a real
    MHD-relevant pressure but is NOT included in the thermal energy W_th
    (which enters the confinement scaling law).

    The fast-alpha beta must be added to the thermal beta when comparing
    against the MHD stability limit (beta_N < beta_N_crit), because
    kink, ballooning, and NTM modes respond to the total pressure.

    Model
    -----
    The Stix (1972) isotropic slowing-down distribution gives the
    steady-state fast-particle energy content as:

        W_fast = P_alpha * tau_se * G_eff(E_alpha/E_c)

    where tau_se is the Spitzer electron-drag time defined via
    dv/dt = -v/tau_se, E_alpha is the alpha birth energy, and E_c
    is the critical velocity (electron/ion drag crossover).
    G_eff is the dimensionless Cordey & Core (1974) energy-integral
    function. It rises monotonically from zero to its asymptotic
    value 1/2 as E_alpha/E_c -> infinity, taking values close to
    0.40 for E_alpha/E_c ~ 10 (ITER and EU-DEMO operating
    conditions). The simple textbook formula W_fast = P_alpha
    * tau_se / 3 is an order-of-magnitude estimate, not an
    asymptotic limit; the analytical Cordey-Core integral evaluated
    here gives the correct value at finite E_alpha/E_c.

    The Spitzer electron-drag time (NRL Plasma Formulary):

        tau_se [s] = 6.32e14 * A_alpha / Z_alpha^2
                     * T_e[eV]^{3/2} / (n_e[m^-3] * ln Lambda)

    Parameters
    ----------
    P_alpha_MW : float
        Total alpha heating power deposited in the plasma [MW].
        Typically P_alpha = 0.2 * P_fus (D-T: E_alpha/E_n = 3.52/14.06).
    Te_keV : float
        Volume-averaged electron temperature [keV].
    ne_20 : float
        Volume-averaged electron density [10^20 m^-3].
    B0 : float
        On-axis toroidal magnetic field [T].
    V_m3 : float
        Plasma volume [m^3].
    Z_eff : float, optional
        Effective charge (enters Coulomb logarithm and critical energy).
        Default 1.65.
    A_DT : float, optional
        Effective fuel mass number (2.5 for 50/50 D-T). Default 2.5.

    Returns
    -------
    beta_fast : float
        Fast-alpha contribution to toroidal beta (dimensionless).
    tau_se : float
        Spitzer electron-drag time [s].
    W_fast : float
        Stored fast-alpha energy [MJ].

    References
    ----------
    Stix, Plasma Physics 14 (1972) 367 — slowing-down distribution.
    Cordey & Core, Phys. Fluids 17 (1974) 1626 — energy integral.
    Wesson, Tokamaks 4th ed. (2011), §14.5 — critical velocity.
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054, §17 — PROCESS model.
    """
    # ── Alpha parameters ──
    A_alpha = 4.0           # alpha mass number
    Z_alpha = 2.0           # alpha charge
    E_alpha = 3520.0        # alpha birth energy [keV]

    # ── Coulomb logarithm (alpha-electron) ──
    ln_Lambda = max(15.2 - 0.5 * np.log(max(ne_20, 1e-3))
                    + np.log(max(Te_keV, 0.1)), 10.0)

    # ── Spitzer electron-drag time ──
    # tau_se = (3/(4 sqrt(2 pi))) * (4 pi eps_0)^2 * m_alpha * (kT_e)^1.5
    #          / (Z_alpha^2 * e^4 * n_e * sqrt(m_e) * ln Lambda)
    # In practical units (derived from SI, verified numerically):
    #   tau_se = 6.32e14 * A_alpha/Z_alpha^2 * Te[eV]^1.5 / (ne[m^-3] * lnL)
    ne_m3 = ne_20 * 1e20
    Te_eV = Te_keV * 1e3
    tau_se = (6.32e14 * A_alpha / Z_alpha**2
              * Te_eV**1.5 / (ne_m3 * ln_Lambda))

    # ── Critical energy (electron/ion drag crossover) ──
    # E_c = 14.8 * A_alpha * Te[keV] * (sum_j n_j Z_j^2 / (n_e A_j))^{2/3}
    # For DT plasma: sum = (1/A_DT) approximately (fuel dominates)
    # Wesson (2011) Eq. 14.5.5; Stix (1972) Eq. 18.
    sigma_Zi = 1.0 / A_DT   # simplified: pure fuel, Z=1, <1/A> = 1/A_DT
    E_c = 14.8 * A_alpha * Te_keV * sigma_Zi**(2.0/3.0)   # [keV]

    # ── Stix energy integral (Cordey & Core 1974) ──
    # W_fast = (P_α/E_α) × (τ_se/3) × E_c × ∫₀^u₀ w^{2/3}/(w+1) dw
    # where u₀ = (E₀/E_c)^{3/2}. Substituting w = t³:
    #   ∫ = 3 × ∫₀^y t⁴/(t³+1) dt,  y = sqrt(E₀/E_c) = v₀/v_c
    # Analytical result (partial fractions):
    #   ∫₀^y t⁴/(t³+1) dt = y²/2 + (1/3)ln(y+1) - (1/6)ln(y²-y+1)
    #                        - (1/√3)[arctan((2y-1)/√3) + π/6]
    y = np.sqrt(max(E_alpha / max(E_c, 1.0), 1.01))  # v₀/v_c, clamp > 1
    I_t4 = (y**2 / 2.0
            + np.log(y + 1.0) / 3.0
            - np.log(y**2 - y + 1.0) / 6.0
            - (np.arctan((2.0*y - 1.0) / np.sqrt(3.0)) + np.pi/6.0) / np.sqrt(3.0))
    I_w = 3.0 * I_t4   # = ∫₀^{u₀} w^{2/3}/(w+1) dw

    # Effective G factor: W_fast = P_α × τ_se × G_eff
    G_eff = (E_c / E_alpha) * I_w / 3.0

    # ── Fast-alpha stored energy and pressure ──
    P_alpha_W = P_alpha_MW * 1e6               # MW → W
    W_fast_J  = P_alpha_W * tau_se * G_eff     # [J]
    p_fast    = (2.0 / 3.0) * W_fast_J / V_m3  # [Pa], isotropic

    # ── Fast-alpha beta ──
    μ0 = 4.0e-7 * np.pi
    beta_fast = 2.0 * μ0 * p_fast / B0**2

    return beta_fast, tau_se, W_fast_J / 1e6  # beta [-], tau [s], W [MJ]


if __name__ == "__main__":
    # ── ITER chain (3/12) - on-axis field, stored energy and beta ───────
    # Field convention of the deck: Bmax_TF = 10.60 T is the peak field at
    # the winding-pack FRONT FACE, at radius R0 - a - b with the published
    # inboard stack b = 1.10 m. ITER quotes 11.8 T on the conductor,
    # deeper inside the winding pack; both describe the same machine and
    # the front-face convention reproduces B0 = 5.30 T exactly.
    # W_th = (3/2) pbar V is evaluated inline (definition of f_W_th,
    # re-checked through f_W_th in chain 11). Ip is a forward reference
    # (FROZEN), closed by the scaling-law inversion of chain 11.
    _B0 = f_B0(ITER['Bmax_TF'], ITER['a'], ITER['b'], ITER['R0'])
    _W = 1.5 * ITER['pbar'] * 1e6 * ITER['V'] / 1e6        # [MJ]
    _bT = f_beta_T(ITER['pbar'], _B0)
    _bP = f_beta_P(ITER['a'], ITER['kappa'], ITER['pbar'], FROZEN['Ip'])
    _bN = f_beta_N(f_beta(_bT, _bP), ITER['a'], _B0, FROZEN['Ip'])
    # Fast-alpha pressure (Stix slowing-down), to be added to the thermal
    # beta for MHD-limit comparisons. P_alpha = P_fus/5 = f_P_alpha
    # (defined in the next section, re-checked in chain 7).
    _bfa, _tau_se, _W_fast = f_beta_fast_alpha(
        ITER['P_fus'] / 5.0, ITER['Tbar'], ITER['nbar'], _B0, ITER['V'],
        Z_eff=ITER['Zeff'])
    ITER.update(B0=_B0, W_th=_W, betaN=_bN)
    _bench("ITER chain 3/12 - field, stored energy and beta", [
        ("B0 on axis [T]", _B0, 5.30, 1e-3, "Shimada 2007"),
        ("W_th [MJ]", _W, FROZEN['W_th'], 1e-3, "deck frozen"),
        ("W_th [MJ]", _W, 350.0, 0.05, "Shimada 2007"),
        ("beta_T [-]", _bT, FROZEN['betaT'], 2e-3, "deck frozen"),
        ("beta_P [-]", _bP, FROZEN['betaP'], 2e-3, "deck frozen"),
        ("beta_N [% m T/MA]", _bN, FROZEN['betaN'], 2e-3, "deck frozen"),
        ("beta_N [% m T/MA]", _bN, "1.8", None, "Shimada 2007"),
        ("Troyon margin beta_N", _bN, (0.0, 2.8), 1, "deck limit"),
        ("beta_fast_alpha / beta_T [-]", _bfa / _bT, None, None,
         "Stix slowing-down"),
    ], notes=[
        "beta_N solves below the published 1.8 because the deck-solved "
        "temperature (7.75 keV) sits under the 8.9 keV design assumption; "
        "the deviation follows the stored energy and density directly.",
    ])

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
                    tbeta=2.0,
                    rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                    Vprime_data=None):
    """
    Total synchrotron (electron-cyclotron) radiation power — Albajar-Fidone.

    Gyrating relativistic electrons emit cyclotron radiation at harmonics of
    the electron-cyclotron frequency.  At reactor temperatures (T_e > 5 keV),
    the relativistic correction and the plasma opacity become important.

    The formula combines:
      - Albajar et al., Nucl. Fusion 41 (2001) 665 — base formula
      - Fidone, Giruzzi & Granata, Nucl. Fusion 41 (2001) 1755 — modified
        wall-reflection treatment with exponents 0.62 and 0.41.

    Key terms:
      - Opacity parameter (Albajar Eq. 7):
            p_a0 = 6.04e3 × a × ne0[10²⁰] / B₀
      - Profile factor K (Albajar Eq. 13), depends on (αn, αT, βT):
            The temperature profile is T ∝ (1 − ρ^βT)^αT.
            αT = nu_T (peaking), βT = tbeta (radial shape, default 2).
      - Aspect-ratio factor G (Albajar Eq. 15):
            G = 0.93 (1 + 0.85 exp(−0.82 A))
      - Wall reflection (Fidone 2001):
            prefactor (1 − R)^0.62, opacity correction (1 − R)^0.41.

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
        Density and temperature profile peaking exponents (αn, αT).
    r_synch : float
        First-wall reflectivity for synchrotron photons [0, 1).
    tbeta : float, optional
        Temperature profile radial exponent βT in T ∝ (1−ρ^βT)^αT.
        Default 2.0 (standard parabolic: (1−ρ²)^αT).
        This enters the K factor only (Albajar Eq. 13).
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
    Fidone, Giruzzi & Granata, Nucl. Fusion 41 (2001) 1755.
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054, §10.
    """
    T0  = float(f_Tprof(Tbar, nu_T, 0.0, rho_ped, T_ped_frac,
                        Vprime_data))                              # on-axis T [keV]
    ne0 = float(f_nprof(nbar,  nu_n,  0.0, rho_ped, n_ped_frac,
                        Vprime_data))                              # on-axis n [1e20 m-3]
    A   = R0 / a

    pa0 = 6.04e3 * a * ne0 / B0                                    # opacity parameter (Eq. 7)

    # Profile factor K (Albajar 2001, Eq. 13).
    # K depends on αn (= nu_n), αT (= nu_T), and βT (= tbeta).
    # The denominator (βT^1.53 + 1.87·αT − 0.16) vanishes for small βT;
    # clamp βT to 0.5 minimum for numerical safety.
    bT = max(tbeta, 0.5)
    nu_T_K = max(nu_T, 0.1)
    if nu_T < 0.1:
        warnings.warn(
            f"f_P_synchrotron: nu_T = {nu_T:.3f} < 0.1; clamped to 0.1 "
            "for the Albajar K-factor (valid for nu_T >= 0.5).",
            RuntimeWarning, stacklevel=2
        )

    K = ((nu_n + 3.87*nu_T_K + 1.46)**(-0.79)                     # Albajar Eq. 13
         * (1.98 + nu_T_K)**1.36 * bT**2.14
         / (bT**1.53 + 1.87*nu_T_K - 0.16)**1.33)

    G = 0.93 * (1.0 + 0.85 * math.exp(-0.82 * A))                 # Albajar Eq. 15

    # Fidone (2001) wall-reflection correction:
    #   prefactor:  (1 − R)^0.62  (not 0.5 as in original Albajar)
    #   opacity:    includes (1 − R)^0.41 in reabsorption term
    Rw = r_synch
    dum = (1.0 + 0.12 * (T0 / pa0**0.41)
           * (1.0 - Rw)**0.41)**(-1.51)                            # Fidone Eq.

    return (3.84e-8 * (1.0 - Rw)**0.62                             # Albajar Eq. 16 + Fidone
            * R0 * a**1.38 * kappa**0.79 * B0**2.62 * ne0**0.38
            * T0 * (16.0 + T0)**2.61
            * dum * K * G)


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
      refined  (Vprime_data provided):
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
        Plasma volume [m³].  Pass V_ac (Academic) or V_d0 (refined).
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

    if Vprime_data is not None or rho_ped < 1.0:
        # Numerical profile integration with pedestal+tanh profiles.
        #   refined mode  (Vprime_data provided): Miller V'(ρ)/V weight
        #   Academic mode (Vprime_data = None):  cylindrical 2ρ weight
        # The analytical f_peak is only valid for purely parabolic profiles;
        # with a pedestal, the n²T^0.5 integral must be computed numerically.
        if Vprime_data is not None:
            rho_grid, Vprime, V_total = Vprime_data[:3]  # safe: 5-tuple (rho, V', V, dA, Lp)
            w_vol = Vprime / V_total
        else:
            rho_grid = np.linspace(0, 1, 300)
            w_vol = 2.0 * rho_grid             # cylindrical, ∫ w dρ = 1
        n_hat = f_nprof(1.0, nu_n, rho_grid, rho_ped, n_ped_frac,
                        Vprime_data)
        T_hat = f_Tprof(1.0, nu_T, rho_grid, rho_ped, T_ped_frac,
                        Vprime_data)
        integrand = n_hat**2 * np.maximum(T_hat, 0.0)**0.5 * w_vol
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0)
        f_peak = float(np.trapezoid(integrand, rho_grid))
    else:
        # Purely parabolic (rho_ped = 1.0, no Vprime): analytical formula
        f_peak = ((1.0 + nu_n)**2 * (1.0 + nu_T)**0.5
                  / (1.0 + 2.0*nu_n + 0.5*nu_T))

    return C_B * Z_eff * nbar**2 * Tbar**0.5 * V * f_peak * 1e-6


if __name__ == "__main__":
    # ── Published anchor - bremsstrahlung constant ───────────────────────
    # Wesson, Tokamaks, Sec. 4.25 (5.35e-37 W m3 in SI; same constant in
    # the NRL Formulary). Flat unit plasma (n = 1e20 m-3, T = 1 keV,
    # Z_eff = 1, V = 1 m3) radiates 5.35e3 W = 5.35e-3 MW.
    _bench("Published anchor - bremsstrahlung constant (Wesson)", [
        ("P_brem flat unit plasma [MW]",
         f_P_bremsstrahlung(1., 1., 1., 1., nu_n=0., nu_T=0.),
         5.35e-3, 0.02, "Wesson 2004"),
    ])

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
        Plasma volume [m³].  Pass V_ac (Academic) or V_d0 (refined).

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
      refined    : w(ρ) = V'_Miller(ρ)     (flux-surface Jacobian)

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
        # refined mode: Miller V'(ρ) integration
        rho_grid, Vprime, V_total = Vprime_data[:3]  # safe: 5-tuple (rho, V', V, dA, Lp)
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

# ── SOL-range radiative cooling rates for the Lengyel detachment model ──────
# Non-coronal equilibrium Lz(Te) for the standard seeding species, valid from
# 1 eV to 2 keV, i.e. covering the scrape-off-layer temperature range where
# the Mavrin (2018) core tables above (validity 0.1-100 keV) cannot be used:
# clamping them below 100 eV would miss the low-Z radiation peaks
# (N: ~10 eV, Ar: ~18 eV, Ne: ~33 eV) and bias the Lengyel integral by an
# order of magnitude.
# Provenance: computed from OpenADAS ADF11 rate coefficients through the
# open-source `radas` package (the atomic-data generator of cfspopcon),
# NON-CORONAL equilibrium ionisation balance at the cfspopcon reference
# residence parameter ne·τ = 5e16 m⁻³ s (SPARC PRD convention), evaluated
# at n_e = 1e20 m⁻³ (log-interpolated on the 17-point ADF11 density grid).
# The finite-ne·τ balance enhances low-Z radiation above the coronal peak,
# which is essential for the Lengyel integral: the coronal limit
# underestimates L_INT by a factor 1.5-3.5 for these species. The coronal
# limit of the same pipeline was cross-validated against the Mavrin (2018)
# tables on the 0.1-2 keV overlap (agreement within 1-5%), validating the
# atomic-data chain. 40-point log-spaced grid; log-log interpolation.
_LZ_SOL_TE_EV = np.array([
        1.0000e+00, 1.2152e+00, 1.4767e+00, 1.7944e+00,
        2.1806e+00, 2.6498e+00, 3.2200e+00, 3.9128e+00,
        4.7548e+00, 5.7780e+00, 7.0213e+00, 8.5322e+00,
        1.0368e+01, 1.2599e+01, 1.5310e+01, 1.8605e+01,
        2.2608e+01, 2.7473e+01, 3.3385e+01, 4.0569e+01,
        4.9299e+01, 5.9907e+01, 7.2798e+01, 8.8463e+01,
        1.0750e+02, 1.3063e+02, 1.5874e+02, 1.9290e+02,
        2.3441e+02, 2.8485e+02, 3.4614e+02, 4.2063e+02,
        5.1114e+02, 6.2113e+02, 7.5478e+02, 9.1720e+02,
        1.1146e+03, 1.3544e+03, 1.6458e+03, 2.0000e+03,
])

_LZ_SOL_TABLES = {
    "N": np.array([
        5.0025e-36, 2.4353e-35, 7.7385e-35, 2.0810e-34,
        5.4433e-34, 1.3119e-33, 2.8754e-33, 5.8246e-33,
        1.0667e-32, 1.7410e-32, 2.5772e-32, 3.5402e-32,
        4.6050e-32, 5.5408e-32, 5.7195e-32, 4.8543e-32,
        3.5276e-32, 2.3917e-32, 1.6212e-32, 1.1381e-32,
        8.3808e-33, 6.4927e-33, 5.3281e-33, 4.7196e-33,
        4.5987e-33, 4.9061e-33, 5.5201e-33, 6.2480e-33,
        6.8995e-33, 7.3460e-33, 7.5283e-33, 7.4650e-33,
        7.2343e-33, 6.9200e-33, 6.5775e-33, 6.2418e-33,
        5.9343e-33, 5.6604e-33, 5.4153e-33, 5.1949e-33,
    ]),
    "Ne": np.array([
        9.8517e-40, 1.8370e-38, 2.1898e-37, 1.5661e-36,
        6.5299e-36, 1.7311e-35, 4.3615e-35, 1.4131e-34,
        4.8459e-34, 1.3634e-33, 3.0661e-33, 5.8667e-33,
        9.8720e-33, 1.5104e-32, 2.2047e-32, 3.0526e-32,
        3.8913e-32, 4.5949e-32, 5.1670e-32, 5.5754e-32,
        5.6399e-32, 5.1888e-32, 4.3802e-32, 3.5452e-32,
        2.8413e-32, 2.2840e-32, 1.8641e-32, 1.5594e-32,
        1.3512e-32, 1.2277e-32, 1.1747e-32, 1.1657e-32,
        1.1688e-32, 1.1647e-32, 1.1592e-32, 1.1931e-32,
        1.3563e-32, 1.3806e-32, 1.0545e-32, 9.5416e-33,
    ]),
    "Ar": np.array([
        2.0544e-37, 1.8181e-36, 1.1460e-35, 5.8612e-35,
        2.5789e-34, 8.6531e-34, 2.2022e-33, 4.9495e-33,
        1.1326e-32, 2.4691e-32, 4.6854e-32, 7.7604e-32,
        1.1441e-31, 1.5159e-31, 1.8221e-31, 2.0334e-31,
        2.1204e-31, 1.9803e-31, 1.6204e-31, 1.1892e-31,
        8.3509e-32, 5.9904e-32, 4.5672e-32, 3.7829e-32,
        3.4596e-32, 3.4943e-32, 3.7724e-32, 4.1327e-32,
        4.4139e-32, 4.5117e-32, 4.3905e-32, 4.0746e-32,
        3.6349e-32, 3.1582e-32, 2.7162e-32, 2.3401e-32,
        2.0366e-32, 1.7990e-32, 1.6155e-32, 1.4728e-32,
    ]),
}


def get_Lz_SOL(impurity, Te_eV):
    """
    SOL-range non-coronal radiative cooling rate Lz(Te) [W m³].

    Non-coronal equilibrium (ne·τ = 5e16 m⁻³ s, n_e = 1e20 m⁻³), matching
    the cfspopcon SPARC PRD reference convention; complements get_Lz
    (Mavrin 2018 coronal, core range 0.1-100 keV) below 100 eV, where the
    Lengyel detachment integral lives. See the provenance note on the
    _LZ_SOL_TABLES block above. Temperatures are clipped to the
    [1 eV, 2 keV] table range.

    Parameters
    ----------
    impurity : str
        Seeding species: 'N', 'Ne' or 'Ar'.
    Te_eV : float or array
        Electron temperature [eV].

    Returns
    -------
    Lz : float or array
        Radiative cooling rate [W m³].
    """
    if impurity not in _LZ_SOL_TABLES:
        raise ValueError(
            f"get_Lz_SOL: unsupported seeding species '{impurity}'. "
            f"Supported: {sorted(_LZ_SOL_TABLES)}.")
    Te = np.clip(np.asarray(Te_eV, dtype=float), _LZ_SOL_TE_EV[0], _LZ_SOL_TE_EV[-1])
    lz = np.exp(np.interp(np.log(Te), np.log(_LZ_SOL_TE_EV),
                          np.log(_LZ_SOL_TABLES[impurity])))
    return float(lz) if np.ndim(Te_eV) == 0 else lz


def f_L_INT(impurity, T_start_eV, T_stop_eV, n_pts=300):
    """
    Lengyel weighted cooling integral L_INT = ∫ Lz(T) √T dT  [W m³ eV^1.5].

    Evaluated by trapezoidal quadrature of the SOL-range coronal cooling
    rate (get_Lz_SOL) on a log-spaced temperature grid between the target
    and upstream temperatures.

    Parameters
    ----------
    impurity : str
        Seeding species: 'N', 'Ne' or 'Ar'.
    T_start_eV, T_stop_eV : float
        Lower (target) and upper (upstream separatrix) electron
        temperatures [eV].
    n_pts : int, optional
        Quadrature points. Default 300.

    Returns
    -------
    L_INT : float
        Weighted cooling integral [W m³ eV^1.5].
    """
    T_lo = max(float(T_start_eV), _LZ_SOL_TE_EV[0])
    T_hi = min(float(T_stop_eV), _LZ_SOL_TE_EV[-1])
    if T_hi <= T_lo:
        return 0.0
    T = np.logspace(np.log10(T_lo), np.log10(T_hi), n_pts)
    return float(np.trapezoid(get_Lz_SOL(impurity, T) * np.sqrt(T), T))


def f_lengyel_concentration(q_par_u, n_u, T_u_eV, T_t_eV, impurity,
                            f_pwr_loss, kappa_e0=2600.0,
                            lengyel_factor=4.3):
    """
    Required SOL impurity concentration for a given power-loss fraction —
    Lengyel model.

    Integrating the parallel heat-conduction equation with impurity line
    radiation as the only volumetric loss, under constant total pressure
    along the flux tube (n T = n_u T_u), yields:

        c_z = [q_∥u² − ((1 − f_loss) q_∥u)²]
              / [2 κ_e0 (n_u T_u)² L_INT] / C_Lengyel

    with L_INT = ∫_{T_t}^{T_u} Lz(T) √T dT (see f_L_INT) and
    q_∥t = (1 − f_loss) q_∥u the residual parallel flux at the target.

    Parameters
    ----------
    q_par_u : float
        Upstream peak parallel heat flux [MW m⁻²] (as returned by
        f_heat_two_point / f_heat_PFU_Eich; converted internally to SI).
    n_u : float
        Upstream (separatrix) electron density [m⁻³].
    T_u_eV : float
        Upstream separatrix electron temperature [eV].
    T_t_eV : float
        Target electron temperature [eV].
    impurity : str
        Seeding species: 'N', 'Ne' or 'Ar'.
    f_pwr_loss : float
        SOL power-loss fraction to be radiated, 0 ≤ f < 1 (e.g. the
        f_pwr_loss_req diagnostic of f_heat_two_point).
    kappa_e0 : float, optional
        Spitzer-Härm parallel electron heat conductivity constant
        [W m⁻¹ eV⁻⁷ᐟ²], q_∥ = −κ_e0 T^{5/2} ∇_∥T. Default 2600
        (cfspopcon SPARC PRD reference value).
    lengyel_factor : float, optional
        Calibration divisor C_Lengyel correcting the known systematic
        overestimation of the Lengyel model against SOLPS (Moulton et al.
        2021). Default 4.3 (cfspopcon SPARC PRD reference value).

    Returns
    -------
    c_z : float
        Required impurity concentration n_z/n_e in the SOL [-].

    Notes
    -----
    Non-coronal Lz at ne·τ = 5e16 m⁻³ s is used (cfspopcon reference
    convention, consistent with the Moulton calibration factor); this is
    the standard 0D / system-code level of fidelity. The model does not enforce consistency between c_z and the
    assumed (T_u, T_t) pair; D0FUS uses the two-point-model temperatures.

    References
    ----------
    Lengyel L.L., IPP Report 1/191, Max-Planck-Institut für Plasmaphysik
        (1981) — "Analysis of Radiating Plasma Boundary Layers".
    Moulton D. et al., Nucl. Fusion 61 (2021) 046029 — Lengyel/SOLPS
        comparison motivating the calibration factor.
    Body T. et al., cfspopcon (github.com/cfs-energy/cfspopcon) —
        reference implementation and default coefficients.
    """
    if not (0.0 <= f_pwr_loss < 1.0):
        raise ValueError("f_pwr_loss must satisfy 0 <= f < 1.")
    q_u_SI = q_par_u * 1e6                              # [W m^-2]
    L_int = f_L_INT(impurity, T_t_eV, T_u_eV)           # [W m^3 eV^1.5]
    if L_int <= 0.0:
        return np.inf
    num = q_u_SI**2 - ((1.0 - f_pwr_loss) * q_u_SI)**2
    den = 2.0 * kappa_e0 * (n_u * T_u_eV)**2 * L_int
    return num / den / lengyel_factor


# Mavrin 2018 average charge state <Z>(Te) polynomial coefficients.
# Same reference as the Lz tables below:
#   A.A. Mavrin, "Improved fits of coronal radiative cooling rates for
#   high-temperature plasmas", Radiat. Eff. Defects Solids 173 (2018) 388,
#   DOI: 10.1080/10420150.2018.1462361.
# Coefficients cross-checked against the open-source TORAX implementation
# (google-deepmind/torax, torax/_src/physics/charge_states.py).
# Convention: per species, a list of rows of 5 polynomial coefficients in
# DESCENDING degree (numpy.polyval order) evaluated on X = log10(Te[keV]);
# the row is selected by the temperature interval boundaries [keV] in
# _MAVRIN_ZAVG_TBOUNDS via searchsorted. Validity range: 0.1-100 keV.
# He, Li and Be are fully stripped above 0.1 keV and handled separately.
_MAVRIN_ZAVG_COEFFS = {
    "C": [
        [-7.2007e00, -1.2217e01, -7.3521e00, -1.7632e00, 5.8588e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 6.0000e00],
    ],
    "N": [
        [0.0000e00, 3.3818e00, 1.8861e00, 1.5668e-01, 6.9728e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 7.0000e00],
    ],
    "O": [
        [0.0000e00, -1.8560e01, -3.8664e01, -2.2093e01, 4.0451e00],
        [-4.3092e00, -4.6261e-01, -3.7050e-02, 8.0180e-02, 7.9878e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 8.0000e00],
    ],
    "Ne": [
        [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
        [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
    ],
    "Ar": [
        [6.8717e00, -1.1595e01, -4.3776e01, -2.0781e01, 1.3171e01],
        [-4.8830e-02, 1.8455e00, 2.5023e00, 1.1413e00, 1.5986e01],
        [-5.9213e-01, 3.5667e00, -8.0048e00, 7.9986e00, 1.4948e01],
    ],
    "Kr": [
        [1.3630e02, 4.6320e02, 5.6890e02, 3.0638e02, 7.7040e01],
        [-1.0279e02, 6.8446e01, 1.5744e01, 1.5186e00, 2.4728e01],
        [-2.4682e00, 1.3215e01, -2.5703e01, 2.3443e01, 2.5368e01],
    ],
    "Xe": [
        [5.8178e02, 1.9967e03, 2.5189e03, 1.3973e03, 3.0532e02],
        [8.6824e01, -2.9061e01, -4.8384e01, 1.6271e01, 3.2616e01],
        [4.0756e02, -9.0008e02, 6.6739e02, -1.7259e02, 4.8066e01],
        [-1.0019e01, 7.3261e01, -1.9931e02, 2.4056e02, -5.7527e01],
    ],
    "W": [
        [1.6823e01, 3.4582e01, 2.1027e01, 1.6518e01, 2.6703e01],
        [-2.5887e02, -1.0577e01, 2.5532e02, -7.9611e01, 3.6902e01],
        [1.5119e01, -8.4207e01, 1.5985e02, -1.0011e02, 6.3795e01],
    ],
}

# Temperature interval boundaries [keV] separating the coefficient rows above.
_MAVRIN_ZAVG_TBOUNDS = {
    "C": [0.7], "N": [0.7], "O": [0.3, 1.5], "Ne": [0.5, 2.0],
    "Ar": [0.6, 3.0], "Kr": [0.447, 4.117], "Xe": [0.3, 1.5, 8.0],
    "W": [1.5, 4.0],
}

# Light species fully stripped above 0.1 keV (Mavrin 2018 provides no fit).
_FULLY_STRIPPED_Z = {"He": 2.0, "Li": 3.0, "Be": 4.0}


def get_Z_mean(impurity, Te_keV):
    """
    Coronal-equilibrium average charge state <Z>(Te) of an impurity species.

    Piecewise polynomial fits to ADAS coronal-equilibrium data from
    Mavrin (2018), evaluated on X = log10(Te[keV]). Same reference and
    species coverage as the Lz cooling-rate tables (get_Lz), making the
    Z_eff / dilution / radiation chain fully consistent.

    Parameters
    ----------
    impurity : str
        Species symbol. Fitted: 'C','N','O','Ne','Ar','Kr','Xe','W'.
        'He','Li','Be' are treated as fully stripped (<Z> = Z_nuclear).
    Te_keV : float or array
        Electron temperature [keV]. Clipped to the fit validity range
        [0.1, 100] keV to avoid extrapolation.

    Returns
    -------
    Z_mean : float or array
        Average charge state [-].

    References
    ----------
    Mavrin A.A., Radiat. Eff. Defects Solids 173 (2018) 388,
        DOI: 10.1080/10420150.2018.1462361.
    Coefficients cross-checked against the TORAX implementation
        (google-deepmind/torax, charge_states.py).
    """
    if impurity in _FULLY_STRIPPED_Z:
        Z_full = _FULLY_STRIPPED_Z[impurity]
        out = np.full_like(np.asarray(Te_keV, dtype=float), Z_full)
        return float(out) if np.ndim(Te_keV) == 0 else out
    if impurity not in _MAVRIN_ZAVG_COEFFS:
        raise ValueError(
            f"Unknown impurity '{impurity}' for get_Z_mean. Supported: "
            f"{sorted(_FULLY_STRIPPED_Z) + sorted(_MAVRIN_ZAVG_COEFFS)}.")
    Te = np.clip(np.asarray(Te_keV, dtype=float), 0.1, 100.0)
    X = np.log10(Te)
    bounds = np.asarray(_MAVRIN_ZAVG_TBOUNDS[impurity])
    coeffs = np.asarray(_MAVRIN_ZAVG_COEFFS[impurity])
    idx = np.searchsorted(bounds, Te)
    Z = np.empty_like(X)
    for k in np.unique(idx):
        m = (idx == k)
        Z[m] = np.polyval(coeffs[k], X[m])
    # Physical clamping: 0 <= <Z> <= Z_nuclear (last row constant term for
    # fully-stripped high-T rows of light species; for heavy species use a
    # nuclear-charge chart).
    Z_nuc = {"C": 6, "N": 7, "O": 8, "Ne": 10, "Ar": 18, "Kr": 36,
             "Xe": 54, "W": 74}[impurity]
    Z = np.clip(Z, 0.0, Z_nuc)
    return float(Z) if np.ndim(Te_keV) == 0 else Z


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
    # ── ITER chain (4/12) - radiation budget ─────────────────────────────
    # Deck composition: W (2e-5) + Ne (7e-3) with the Zeff = 1.65
    # override, wall reflectivity r_synch = 0.6, profile-integrated
    # radiation with the core/edge split at rho_rad_core = 0.75 and
    # coreradiationfraction = 1 (deck defaults).
    # Convention: the bremsstrahlung term uses the FUEL effective charge
    # Z_eff,fuel = 1 + 2 f_alpha - f_imp_dil (D, T, He only); the
    # impurity contribution (line + recombination + impurity
    # bremsstrahlung) is carried by the Mavrin cooling rates inside
    # P_line. Term-by-term comparisons with published decompositions are
    # therefore convention-dependent; the underlying pieces are anchored
    # separately (Wesson constant above, Mavrin/TORAX block below).
    # This block also closes the f_imp_dil forward reference of chain 2.
    _fdil = sum(get_Z_mean(s, ITER['Tbar']) * c for s, c in ITER['imp'].items())
    _zimp2 = sum(get_Z_mean(s, ITER['Tbar'])**2 * c
                 for s, c in ITER['imp'].items())
    _zf = 1.0 + 2.0 * FROZEN['f_alpha'] - _fdil
    _Pb = f_P_bremsstrahlung(ITER['nbar'], ITER['Tbar'], _zf, ITER['V'],
                             ITER['nu_n'], ITER['nu_T'],
                             rho_ped=ITER['rho_ped'],
                             n_ped_frac=ITER['n_ped_frac'],
                             T_ped_frac=ITER['T_ped_frac'],
                             Vprime_data=ITER_Vpd)
    _Ps = f_P_synchrotron(ITER['Tbar'], ITER['R0'], ITER['a'], ITER['B0'],
                          ITER['nbar'], ITER['kappa'], ITER['nu_n'],
                          ITER['nu_T'], ITER['r_synch'],
                          rho_ped=ITER['rho_ped'],
                          n_ped_frac=ITER['n_ped_frac'],
                          T_ped_frac=ITER['T_ped_frac'],
                          Vprime_data=ITER_Vpd)
    _Plc, _Plt = 0.0, 0.0
    for _sp, _c in ITER['imp'].items():
        _pc, _pt = f_P_line_radiation_profile(
            _sp, _c, ITER['nbar'], ITER['Tbar'], ITER['nu_n'], ITER['nu_T'],
            ITER['V'], rho_ped=ITER['rho_ped'],
            n_ped_frac=ITER['n_ped_frac'], T_ped_frac=ITER['T_ped_frac'],
            Vprime_data=ITER_Vpd, rho_core=ITER['rho_rad_core'], N=150)
        _Plc += _pc
        _Plt += _pt
    _Prc = _Pb + _Ps + _Plc        # coreradiationfraction = 1 (deck default)
    _Prt = _Pb + _Ps + _Plt
    ITER.update(P_rad_core=_Prc, P_rad_tot=_Prt)
    _bench("ITER chain 4/12 - radiation budget (W + Ne, pedestal profiles)", [
        ("fuel dilution sum <Z_j> c_j [-]", _fdil, FROZEN['f_imp_dil'], 2e-3,
         "deck frozen"),
        ("Z_eff,fuel [-]", _zf, None, None, "convention"),
        ("Z_eff computed (fuel + imp) [-]", _zf + _zimp2, "1.65", None,
         "deck override"),
        ("P_brem (fuel) [MW]", _Pb, FROZEN['P_Brem'], 2e-3, "deck frozen"),
        ("P_synchrotron [MW]", _Ps, FROZEN['P_syn'], 2e-3, "deck frozen"),
        ("P_line core (rho < 0.75) [MW]", _Plc, FROZEN['P_line_core'], 2e-3,
         "deck frozen"),
        ("P_line total [MW]", _Plt, FROZEN['P_line'], 2e-3, "deck frozen"),
        ("P_rad,core -> tau_E, Ip [MW]", _Prc, None, None, "chain 11"),
        ("P_rad,total -> P_sep [MW]", _Prt, None, None, "chain 6"),
    ])

#%% Current drive
"""
Current Drive : non-inductive figure of merit γ [MA/(MW m²)].

Definition
----------
    γ = I_CD [MA] × R₀ [m] × n̄_e [10²⁰ m⁻³] / P_CD [MW]

equivalently η_CD [10²⁰ A m⁻² W⁻¹]  (Fisch 1987 & Freidberg 2015 convention)

LHCD models
-----------
METIS mode 0 (Artaud et al. NF 58, 2018, 105001):
    gamma_LH = 2.4 / (5 + Zeff) * tanh(Te_keV / 6)
Semi-empirical formula from METIS (CEA-IRFM):
The 1/(5+Zeff) is the Fisch (1987) Spitzer conductivity factor; the
tanh(Te/6keV) captures non-relativistic to relativistic saturation.
Note: the specific tanh form and prefactor 2.4 are not published in
peer-reviewed literature. They originate from ITER Physics Basis
p.2515 combined with Tore Supra calibration (METIS zicd0.m comments).

ECCD and NBCD models
--------------------
- ECCD :  Giruzzi, NF 27 (1987) + Lin-Liu, GA-A24257 (2003).
          Physics-based model from METIS (CEA-IRFM, zicd0.m lhmode=5).
          No calibration constant. Depends on Te, Zeff, eps, theta_p.
- NBCD :  Stix/Cordey/Lin-Liu physics-based model from METIS.
          No calibration constant. Depends on Te, ne, Eb, Zeff, angle.

Trapped-particle correction (Kim et al. 1991):
    f_trap(ρ) = 1.46 √ε (1 − 0.54 √ε),   ε = ρ a / R₀.

No calibration constants remain in the CD models.

Expected ordering at ITER Q=10:
    γ_LH ≈ 0.33 > γ_NBI ≈ 0.34 > γ_EC ≈ 0.17 MA/(MW m²)

References
----------
Artaud et al., Nucl. Fusion 58 (2018) 105001 — METIS code, LHCD model.
Fisch, Rev. Mod. Phys. 59 (1987) 175 — theoretical basis.
Fisch & Boozer, Phys. Rev. Lett. 45 (1980) 720 — current drive mechanism.
Giruzzi, Nucl. Fusion 27 (1987) 1934 — ECCD trapped-electron correction.
Lin-Liu, Chan, Prater, GA-A24257 (2003) — ECCD Zeff correction.
Lin-Liu, Chan, Prater, Phys. Plasmas 10 (2003) 4064 — relativistic ECCD.
Stix, Plasma Phys. 14 (1972) 367 — critical energy, slowing-down.
Cordey, Start, Jones, Nucl. Fusion 19 (1979) 249 — base NBCD formula.
Cordey, Nucl. Fusion 26 (1986) 123 — trapped-particle effects on NBCD.
Kim et al., Phys. Fluids B 3 (1991) 2050 — trapped particle correction.
Gormezano et al. (ITER PIPB Ch. 6), Nucl. Fusion 47 (2007) S285.
"""


if __name__ == "__main__":
    # ── Published anchors - atomic data (Mavrin, SOL tables, Lengyel) ───
    # Mavrin (2018) <Z>(T_e): coefficients cross-checked against TORAX
    # (google-deepmind/torax, charge_states.py); Ne fully stripped above
    # 5 keV; W(8.9 keV) = 53.06 (hand evaluation of the interval-3
    # polynomial). Frozen non-coronal SOL L_z tables (OpenADAS/radas,
    # ne tau = 5e16 m-3 s, cfspopcon SPARC-PRD convention): peak
    # integrity at freezing time, and the one-sided invariant
    # Lz_SOL >= Lz_coronal on the 0.1-2 keV overlap (the finite residence
    # time keeps line-radiating charge states alive). Lengyel: frozen
    # cfspopcon SPARC-PRD regression point (chain validated to 0.3 % over
    # five decades of q_par, 2026-06 review).
    _rows = [
        ("<Z> Ne at 8.9 keV [-]", get_Z_mean('Ne', 8.9), 10.0, 1e-9,
         "Mavrin 2018 / TORAX"),
        ("<Z> W at 8.9 keV [-]", get_Z_mean('W', 8.9), 53.06, 1e-3,
         "Mavrin 2018 / TORAX"),
    ]
    for _sp, (_Tp, _Lp) in {'N': (13.2, 5.745e-32), 'Ne': (49.8, 5.642e-32),
                            'Ar': (22.6, 2.121e-31)}.items():
        _Tt = np.logspace(0, 3, 600)
        _lz = get_Lz_SOL(_sp, _Tt)
        _kk = int(np.argmax(_lz))
        _rows.append((f"Lz_SOL {_sp} peak value [W m3]", float(_lz[_kk]),
                      _Lp, 0.05, "frozen table"))
        _rows.append((f"Lz_SOL {_sp} peak position ratio", float(_Tt[_kk]) / _Tp,
                      (1 / 1.6, 1.6), 1, "frozen table"))
        _rmin = min(get_Lz_SOL(_sp, _Tk * 1e3) / get_Lz(_sp, _Tk)
                    for _Tk in (0.1, 0.5, 2.0))
        _rows.append((f"invariant Lz_SOL/Lz_coronal {_sp}", float(_rmin),
                      (0.95, 1e9), 1, "one-sided physics"))
    _rows.append(("Lengyel c_z (Ar, SPARC point) [-]",
                  f_lengyel_concentration(9101., 2.1e19, 280.1, 25.0, 'Ar',
                                          0.988),
                  0.7569, 0.02, "cfspopcon ref."))
    _bench("Published anchors - atomic data and Lengyel chain", _rows)

def _ln_Lambda_CD(Te_keV, ne_20):
    """
    Coulomb logarithm for electron-electron collisions — NRL Formulary (2022).

    Piecewise formula in D0FUS units (T in keV, n in 10²⁰ m⁻³):

        T_e < 10 eV :  lnΛ = 23.0  − ln(√n_e [cm⁻³] / T_e [eV]^{3/2})
        T_e ≥ 10 eV :  lnΛ = 24.0 − ln(√n_e [cm⁻³] / T_e [eV])

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
        lnL = 24.0 - np.log(ne_cm3**0.5 * Te_eV**(-1.0))
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


def f_etaCD_LH_physics(Tbar, Z_eff):
    """
    LHCD figure of merit — METIS mode 0 (Artaud et al. 2018).

    No calibration constant. Replaces Ehst-Karney + C_LH = 1.52:

        gamma_LH = 2.4 / (5 + Zeff) * tanh(Te_keV / 6)

    The 1/(5+Zeff) factor is the Spitzer parallel conductivity correction
    for Landau-damped current drive (Fisch 1987). The tanh(Te/6keV) term
    interpolates between the linear non-relativistic limit (Te << 6 keV)
    and relativistic saturation (Te >> 6 keV).

    Origin: METIS integrated tokamak simulator (CEA-IRFM), zicd0.m
    lhmode=0. Described as derived from ITER Physics Basis p.2515
    (Gormezano et al. NF 47 S285, 2007) with tanh saturation calibrated
    on Tore Supra V_loop = 0 measurements. The specific functional form
    tanh(Te/6keV) and prefactor 2.4 are not published in the peer-reviewed
    literature; the formula should be understood as semi-empirical.

    Future development directions:
    - Fisch (1978) + cold-wave accessibility (Stix): replace tanh with
      explicit n_parallel dependence. Requires f_LH and n_par_launched.
      Would capture density limit and launcher design effects.
    - Hot conductivity correction (Giruzzi NF 37, 1997): additional
      current from resistivity reduction in LH deposition zone.

    Parameters
    ----------
    Tbar  : float  Volume-averaged electron temperature [keV].
    Z_eff : float  Effective ion charge.

    Returns
    -------
    float  gamma_CD^LH [MA/(MW m^2)].

    References
    ----------
    Artaud et al., Nucl. Fusion 58 (2018) 105001.
    Fisch, Rev. Mod. Phys. 59 (1987) 175.
    Gormezano et al., Nucl. Fusion 47 (2007) S285.
    """
    return 2.4 / (5.0 + Z_eff) * np.tanh(Tbar / 6.0)


def _mu_trapped_EC(eps, theta_p_rad):
    """
    Pitch-angle boundary for trapped/passing transition at poloidal angle theta_p.

    mu_t = sqrt(eps * (1 + cos(theta_p)) / (1 + eps * cos(theta_p)))

    At theta_p = pi (HFS): mu_t = 0   (no trapping effect)
    At theta_p = 0  (LFS): mu_t = sqrt(2*eps/(1+eps))  (max trapping)

    Parameters
    ----------
    eps         : float  Local inverse aspect ratio.
    theta_p_rad : float  Poloidal angle of EC deposition [rad].

    Returns
    -------
    float  mu_t in [0, 1).

    References
    ----------
    Giruzzi, Nucl. Fusion 27 (1987) 1934.
    """
    num = eps * (1.0 + np.cos(theta_p_rad))
    den = 1.0 + eps * np.cos(theta_p_rad)
    return np.sqrt(max(num / max(den, 1e-10), 0.0))


def _f_circ_simple(eps):
    """
    Approximate effective circulating fraction for circular geometry.

    f_c = 1 - sqrt(2*eps/(1+eps))

    Consistent with Lin-Liu GA-A24257 Eq. 32 in the large-aspect-ratio limit.

    Parameters
    ----------
    eps : float  Local inverse aspect ratio.

    Returns
    -------
    float  f_c in (0, 1].
    """
    return 1.0 - np.sqrt(2.0 * eps / (1.0 + eps))


def f_etaCD_EC_physics(a, R0, Tbar, nbar, Z_eff, nu_T, nu_n, rho_EC,
                        theta_EC_pol_deg=0.0,
                        rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                        Vprime_data=None):
    """
    ECCD figure of merit — Giruzzi (1987) + Lin-Liu (GA-A24257) physics model.

    Replaces f_etaCD_EC (C_EC formula) with first-principles computation.
    No free calibration constant: efficiency depends on local Te, Zeff,
    inverse aspect ratio eps, and poloidal angle of EC deposition theta_p.

    Based on METIS zicd0.m (CEA-IRFM), lhmode=5.

    gamma_EC = [Te_loc / (Te_loc + 100)] x G_trap(eps, theta_p, Zeff) x G_Zeff(eps, Zeff)

    Three physical factors:

    1. Temperature: Te / (Te + 100 keV)
       Giruzzi (1987) saturating scaling. Linear at low Te, saturates
       in the relativistic regime.

    2. Trapped-particle correction: Giruzzi (1987) generalised formula
       G_trap = 1 - (1 + rho_hat/3) x (sqrt(2) x mu_t)^rho_hat
       where rho_hat = (5+Zeff)/(1+Zeff) and mu_t depends on the
       poloidal angle of EC deposition. HFS (theta_p ~ 180 deg)
       minimises trapping, giving maximum gamma.

    3. Zeff correction: Lin-Liu GA-A24257, Eq. 34
       6 / (1 + 4*f_c + Zeff) where f_c is the circulating fraction.
       Reduces to 6/(5+Zeff) at large aspect ratio.

    Parameters
    ----------
    a, R0       : float  Minor/major radius [m].
    Tbar, nbar  : float  Volume-averaged Te [keV] and ne [10^20 m^-3].
    Z_eff       : float  Effective ion charge.
    nu_T, nu_n  : float  Profile peaking exponents.
    rho_EC      : float  Normalised EC wave deposition radius.
    theta_EC_pol_deg : float  Poloidal angle of EC deposition [deg].
                              0 = outboard midplane (LFS), 180 = HFS.
    rho_ped, n_ped_frac, T_ped_frac : float  Pedestal parameters.
    Vprime_data : tuple or None  Miller geometry data.

    Returns
    -------
    float  gamma_CD^EC  [MA / (MW m^2)].

    References
    ----------
    Giruzzi, Nucl. Fusion 27 (1987) 1934.
    Lin-Liu, Chan, Prater, GA-A24257 (2003).
    Lin-Liu, Chan, Prater, Phys. Plasmas 10 (2003) 4064.
    Taguchi, Plasma Phys. Controlled Fusion 31 (1989) 241.
    Fisch & Boozer, Phys. Rev. Lett. 45 (1980) 720.
    """
    # Local Te at deposition radius
    Te_loc = max(float(f_Tprof(Tbar, nu_T, rho_EC, rho_ped, T_ped_frac,
                                Vprime_data)), 0.1)

    # Local inverse aspect ratio
    eps = abs(rho_EC) * a / R0

    # Poloidal angle in radians
    theta_p = np.radians(theta_EC_pol_deg)

    # Factor 1: temperature dependence (Giruzzi 1987)
    T_crit = 100.0  # [keV]
    f_Te = Te_loc / (Te_loc + T_crit)

    # Factor 2: trapped-particle correction (Giruzzi 1987)
    rho_hat = (5.0 + Z_eff) / (1.0 + Z_eff)
    mu_t = _mu_trapped_EC(eps, theta_p)
    sqrt2_mu = np.sqrt(2.0) * mu_t

    if sqrt2_mu < 1e-12:
        G_trap = 1.0
    else:
        G_trap = 1.0 - (1.0 + rho_hat / 3.0) * sqrt2_mu ** rho_hat

    # Factor 3: Zeff + circulating fraction (Lin-Liu GA-A24257, Eq. 34)
    f_c = _f_circ_simple(eps)
    G_Zeff = 6.0 / (1.0 + 4.0 * f_c + Z_eff)

    return f_Te * G_trap * G_Zeff


# ── NBCD physics-based model (Stix/Cordey/Lin-Liu) ──────────────────────
# Replaces the calibrated C_NBI formula. No free constant.
# Validated against METIS zicd0.m to < 1.5% (see test_NBCD_physics.py).
#
# References
# ----------
# Stix, Plasma Phys. 14 (1972) 367.
# Cordey, Jones & Start, Nucl. Fusion 19 (1979) 249.
# Lin-Liu & Hinton, Phys. Plasmas 4 (1997) 4179.
# Artaud et al., Nucl. Fusion 58 (2018) 105001  (METIS).


def _stix_critical_energy(Te_keV, A_beam, sum_nZ2_over_A):
    """
    Stix critical energy for beam slowing-down [keV].

        E_c = 14.8 * T_e * (A_b^{3/2} * sum(n_i Z_i^2 / A_i) / n_e)^{2/3}

    Parameters
    ----------
    Te_keV         : float  Local electron temperature [keV].
    A_beam         : int    Beam ion mass number.
    sum_nZ2_over_A : float  sum(n_i Z_i^2 / A_i) / n_e  (dimensionless).

    Returns
    -------
    float  Critical energy [keV], floored at 0.03 keV.

    References
    ----------
    Stix, Plasma Phys. 14 (1972) 367, eq. 15.
    """
    E_c = 14.8 * Te_keV * (A_beam**1.5 * sum_nZ2_over_A)**(2.0 / 3.0)
    return max(E_c, 0.03)


def _stix_critical_energy_gamma(Te_keV, A_beam, Z_eff):
    """
    Modified critical energy for current-drive efficiency [keV].

        E_{c,gamma} = 14.8 * T_e * (2 sqrt(A_b) Z_eff)^{2/3}

    References
    ----------
    METIS zicd0.m, l.783-813 (Artaud et al. 2018).
    """
    E_c_g = 14.8 * Te_keV * (2.0 * np.sqrt(A_beam) * Z_eff)**(2.0 / 3.0)
    return max(E_c_g, 0.03)


def _slowing_down_time(Te_keV, ne_20, A_beam, lnL):
    """
    Spitzer slowing-down time on electrons [s].

        tau_s = 6.27e14 * A_b * T_e[eV]^{3/2} / (n_e[m^-3] * lnL)

    References
    ----------
    NRL Plasma Formulary (2022), p. 35.
    """
    Te_eV = Te_keV * 1.0e3
    ne    = ne_20 * 1.0e20
    return 6.27e14 * A_beam * Te_eV**1.5 / (ne * lnL)


def _cordey_velocity_integral(v_0, v_c, v_g):
    """
    Slowing-down-averaged beam current integral [m/s].

    Numerically evaluates the Cordey (1979) velocity-space integral.
    Replaces the sqrt(E_b/A_b) approximation of the old C_NBI formula.

    References
    ----------
    Cordey, Jones & Start, Nucl. Fusion 19 (1979) 249.
    METIS zicd0.m, l.892-897.
    """
    ev = 1.0 + 2.0 * (v_g / v_c)**3 / 3.0
    x0 = v_0 / v_c
    u  = np.linspace(0.0, 1.0, 501)
    xu = x0 * u
    integrand = xu * (xu**3 / (1.0 + xu**3))**ev
    F_int = np.trapezoid(integrand, u)
    prefactor = v_c * ((v_0**3 + v_c**3) / v_0**3)**(ev - 1.0)
    return min(prefactor * F_int, v_0)


def _linliu_GZ(f_trap, Z_eff):
    """
    Lin-Liu & Hinton trapped-particle CD correction G(Z, f_trap).

    The full correction to driven current is:
        j_CD = j_free * (1 - (1 - G) / Z_eff)

    References
    ----------
    Lin-Liu & Hinton, Phys. Plasmas 4 (1997) 4179, eq. 5-7.
    """
    f_pass = max(1.0 - f_trap, 0.01)
    xt = f_trap / f_pass
    D = (1.414 * Z_eff + Z_eff**2
         + xt * (0.754 + 2.657 * Z_eff + 2.0 * Z_eff**2)
         + xt**2 * (0.348 + 1.243 * Z_eff + Z_eff**2))
    return xt * ((0.754 + 2.21 * Z_eff + Z_eff**2)
                 + xt * (0.348 + 1.243 * Z_eff + Z_eff**2)) / D


def _orbit_trapping_screening(pitch, eps_loc):
    """
    Orbit trapping screening for finite injection angle.

    fi ~ 1 for passing (pitch > mu_trap), fi ~ 0 for trapped.

    References
    ----------
    METIS zicd0.m, l.827.
    """
    mu_trap = np.sqrt(2.0 * eps_loc / (1.0 + eps_loc))
    return min(1.0, max(0.0, 1.0 + np.tanh(10.0 * (pitch - mu_trap))))


def _sum_nZ2_over_A_DT(f_alpha, Z_eff):
    """
    Ion composition factor for DT plasma with helium ash.

    Computes sum(n_i Z_i^2 / A_i) / n_e assuming 50-50 D-T fuel.
    Impurity contribution inferred from Z_eff (carbon-like Z~6, A~12).
    """
    Z_imp, A_imp = 6.0, 12.0
    n_imp_frac = max(0.0, (Z_eff - 1.0 - 2.0 * f_alpha)
                     / (Z_imp * (Z_imp - 1.0)))
    f_fuel = max(1.0 - 2.0 * f_alpha - Z_imp * n_imp_frac, 0.01)
    f_D = f_fuel / 2.0
    f_T = f_fuel / 2.0
    return (f_D / 2.0 + f_T / 3.0 + f_alpha
            + n_imp_frac * Z_imp**2 / A_imp)


def f_etaCD_NBI_physics(A_beam, E_beam_keV, a, R0, Tbar, nbar, Z_eff,
                        nu_T, nu_n, rho_NBI,
                        f_alpha=0.04, angle_NBI_deg=20.0,
                        rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                        Vprime_data=None):
    """
    NBCD figure of merit -- physics-based model without calibration constant.

    Replaces f_etaCD_NBI (C_NBI formula) with first-principles computation:
    Stix critical energy, NRL slowing-down time, Cordey velocity integral,
    Lin-Liu trapped-particle correction.

    Validated against METIS zicd0.m to < 1.5% (see test_NBCD_physics.py).

    Parameters
    ----------
    A_beam        : int    Beam ion mass number (1=H, 2=D, 3=T).
    E_beam_keV    : float  Injection energy [keV].
    a, R0         : float  Minor / major radius [m].
    Tbar, nbar    : float  Vol-averaged T_e [keV] and n_e [10^20 m^-3].
    Z_eff         : float  Effective ion charge.
    nu_T, nu_n    : float  Temperature / density peaking exponents.
    rho_NBI       : float  Normalised deposition radius.
    f_alpha       : float  Helium ash fraction n_He/n_e (default 0.04).
    angle_NBI_deg : float  Injection angle from tangential [deg] (default 20).
    rho_ped, n_ped_frac, T_ped_frac : float  Pedestal parameters.
    Vprime_data   : array or None  Volume derivative data.

    Returns
    -------
    float  gamma_CD [MA/(MW m^2)].

    References
    ----------
    Stix, Plasma Phys. 14 (1972) 367.
    Cordey, Jones & Start, Nucl. Fusion 19 (1979) 249.
    Lin-Liu & Hinton, Phys. Plasmas 4 (1997) 4179.
    Artaud et al., Nucl. Fusion 58 (2018) 105001.
    """
    # Local plasma at deposition point
    Te_keV = max(float(f_Tprof(Tbar, nu_T, rho_NBI, rho_ped,
                                T_ped_frac, Vprime_data)), 0.1)
    ne_20  = float(f_nprof(nbar, nu_n, rho_NBI, rho_ped,
                            n_ped_frac, Vprime_data))

    lnL = _ln_Lambda_CD(Te_keV, ne_20)

    # Ion composition and critical energies
    sum_nZ2A  = _sum_nZ2_over_A_DT(f_alpha, Z_eff)
    E_c_slow  = _stix_critical_energy(Te_keV, A_beam, sum_nZ2A)
    E_c_gamma = _stix_critical_energy_gamma(Te_keV, A_beam, Z_eff)

    # Velocities [m/s]  (M_I = 2*m_p in D0FUS, so M_I/2 = m_p)
    keV_to_J = 1.0e3 * E_ELEM
    m_proton = M_I / 2.0
    v_0 = np.sqrt(2.0 * E_beam_keV * keV_to_J / (A_beam * m_proton))
    v_c = np.sqrt(2.0 * E_c_slow   * keV_to_J / (A_beam * m_proton))
    v_g = np.sqrt(2.0 * E_c_gamma  * keV_to_J / (A_beam * m_proton))

    # Cordey velocity integral
    v_eff = _cordey_velocity_integral(v_0, v_c, v_g)

    # Lin-Liu trapped-particle correction
    f_trap  = _f_trap_CD(rho_NBI, a, R0)
    GZ      = _linliu_GZ(f_trap, Z_eff)
    cd_corr = 1.0 - (1.0 - GZ) / Z_eff

    # Orbit screening
    pitch   = np.cos(np.radians(angle_NBI_deg))
    eps_loc = rho_NBI * a / R0
    fi_trap = _orbit_trapping_screening(abs(pitch), eps_loc)

    # Density-independent assembly (tau_s * n_e cancels):
    #   tau_n = 6.27e14 * A_b * Te[eV]^1.5 / lnL  (= tau_s * n_e)
    Te_eV = Te_keV * 1.0e3
    tau_n = 6.27e14 * A_beam * Te_eV**1.5 / lnL
    E_b_J = E_beam_keV * keV_to_J

    gamma = (tau_n * E_ELEM * abs(pitch) * v_eff
             * cd_corr * fi_trap / (E_b_J * 2.0 * np.pi))

    return gamma / 1.0e20


if __name__ == "__main__":
    # ── Published anchors - NBCD physics model vs METIS ──────────────────
    # Expected values from the METIS zicd0.m benchmark
    # (test_NBCD_physics.py). Tolerance 2 % (lnLambda formula difference,
    # NRL vs METIS). Perpendicular injection must drive zero current.
    _nbcd_cases = [
        ("ITER 1 MeV D", 0.3336, dict(A_beam=2, E_beam_keV=1000, a=2.0, R0=6.2,
                              Tbar=8.9, nbar=1.0, Z_eff=1.65, nu_T=1.5,
                              nu_n=0.3, rho_NBI=0.3, f_alpha=0.04,
                              angle_NBI_deg=20.0)),
        ("ARC 150 keV D", 0.1147, dict(A_beam=2, E_beam_keV=150, a=1.13,
                              R0=3.3, Tbar=14.0, nbar=1.8, Z_eff=1.5,
                              nu_T=1.2, nu_n=0.5, rho_NBI=0.4, f_alpha=0.06,
                              angle_NBI_deg=25.0)),
        ("EU-DEMO 1 MeV", 0.4006, dict(A_beam=2, E_beam_keV=1000, a=2.88,
                              R0=9.07, Tbar=12.0, nbar=0.8, Z_eff=1.6,
                              nu_T=1.5, nu_n=0.3, rho_NBI=0.3, f_alpha=0.05,
                              angle_NBI_deg=20.0)),
    ]
    _rows = [(f"gamma_NBI {_nm}", f_etaCD_NBI_physics(**_kw), _ref, 0.02,
              "METIS zicd0.m")
             for _nm, _ref, _kw in _nbcd_cases]
    _g_perp = f_etaCD_NBI_physics(2, 500, 2.0, 6.2, 10.0, 1.0, 1.65,
                                  1.5, 0.3, 0.3, angle_NBI_deg=90.0)
    _rows.append(("gamma_NBI perpendicular limit", float(_g_perp),
                  (-1e-6, 1e-6), 1, "physics limit"))
    _bench("Published anchors - NBCD (Stix/Cordey/Lin-Liu) vs METIS", _rows)

if __name__ == "__main__":
    # ── Published anchors - ECCD physics model vs METIS ──────────────────
    # The formula is identical to METIS zicd0.m lhmode = 5 (Giruzzi 1987 +
    # Lin-Liu trapped-particle correction): the reference is re-evaluated
    # inline at the same local T_e, so the agreement must hold to machine
    # precision. Physics ordering: HFS > top > LFS > 0.
    _ec_cases = [
        ("ITER HFS 160 deg", 160.0, dict(a=2.0, R0=6.2, Tbar=8.9, nbar=1.0,
                                         Z_eff=1.65, nu_T=1.0, nu_n=0.3,
                                         rho_EC=0.3)),
        ("ITER LFS 0 deg", 0.0, dict(a=2.0, R0=6.2, Tbar=8.9, nbar=1.0,
                                     Z_eff=1.65, nu_T=1.0, nu_n=0.3,
                                     rho_EC=0.3)),
        ("ITER top 90 deg", 90.0, dict(a=2.0, R0=6.2, Tbar=8.9, nbar=1.0,
                                       Z_eff=1.65, nu_T=1.0, nu_n=0.3,
                                       rho_EC=0.3)),
        ("EU-DEMO HFS 160 deg", 160.0, dict(a=2.88, R0=9.07, Tbar=12.0,
                                            nbar=0.8, Z_eff=1.6, nu_T=1.5,
                                            nu_n=0.3, rho_EC=0.3)),
    ]
    _rows = []
    _g_by_case = {}
    for _nm, _theta, _kw in _ec_cases:
        _g = f_etaCD_EC_physics(**_kw, theta_EC_pol_deg=_theta)
        _g_by_case[_nm] = _g
        # METIS reference evaluated inline at the same local T_e
        _Te_loc = max(float(f_Tprof(_kw['Tbar'], _kw['nu_T'], _kw['rho_EC'])),
                      0.1)
        _eps = abs(_kw['rho_EC']) * _kw['a'] / _kw['R0']
        _theta_r = np.radians(_theta)
        _mut = np.sqrt(max(_eps * (1 + np.cos(_theta_r))
                           / max(1 + _eps * np.cos(_theta_r), 1e-10), 0.0))
        _rh = (5.0 + _kw['Z_eff']) / (1.0 + _kw['Z_eff'])
        _s2m = np.sqrt(2.0) * _mut
        _Gt = 1.0 - (1.0 + _rh / 3.0) * _s2m ** _rh if _s2m > 1e-12 else 1.0
        _fc = 1.0 - np.sqrt(2.0 * _eps / (1.0 + _eps))
        _g_metis = (_Te_loc / (_Te_loc + 100.0) * _Gt
                    * 6.0 / (1.0 + 4.0 * _fc + _kw['Z_eff']))
        _rows.append((f"gamma_EC {_nm}", _g, _g_metis, 1e-9, "METIS zicd0.m"))
    _g_hfs = f_etaCD_EC_physics(a=2.0, R0=6.2, Tbar=8.9, nbar=1.0, Z_eff=1.65,
                                nu_T=1.0, nu_n=0.3, rho_EC=0.3,
                                theta_EC_pol_deg=180.0)
    _g_top = f_etaCD_EC_physics(a=2.0, R0=6.2, Tbar=8.9, nbar=1.0, Z_eff=1.65,
                                nu_T=1.0, nu_n=0.3, rho_EC=0.3,
                                theta_EC_pol_deg=90.0)
    _g_lfs = f_etaCD_EC_physics(a=2.0, R0=6.2, Tbar=8.9, nbar=1.0, Z_eff=1.65,
                                nu_T=1.0, nu_n=0.3, rho_EC=0.3,
                                theta_EC_pol_deg=0.0)
    _rows.append(("ordering gamma_HFS/gamma_top", _g_hfs / _g_top,
                  (1.0 + 1e-9, 100.0), 1, "trapping physics"))
    _rows.append(("ordering gamma_top/gamma_LFS", _g_top / _g_lfs,
                  (1.0 + 1e-9, 100.0), 1, "trapping physics"))
    _bench("Published anchors - ECCD (Giruzzi/Lin-Liu) vs METIS", _rows)

def f_etaCD_effective(config, a, R0, B0, nbar, Tbar, nu_n, nu_T, Z_eff,
                      rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0):
    """
    Effective CD figure of merit γ_CD [MA/(MW·m²)] for the active heating mix.

    Routes to the appropriate CD model based on ``config.CD_source``:

    * ``'Academic'``: Fixed user-specified γ_CD (config.gamma_CD_acad).
                      No plasma-physics CD model is evaluated — simplest option.
    * ``'LHCD'``  : f_etaCD_LH_physics  — METIS mode 0 (Artaud 2018)
    * ``'ECCD'``  : f_etaCD_EC_physics — Giruzzi/Lin-Liu physics model
    * ``'NBCD'``  : f_etaCD_NBI_physics — Stix/Cordey/Lin-Liu physics model
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
                            rho_EC, theta_EC_pol_deg, rho_NBI, A_beam, E_beam_keV,
                            angle_NBI_deg).
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
        return f_etaCD_LH_physics(Tbar, Z_eff)

    elif CD_source == 'ECCD':
        return f_etaCD_EC_physics(a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                          config.rho_EC,
                          theta_EC_pol_deg=config.theta_EC_pol_deg,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac)

    elif CD_source == 'NBCD':
        return f_etaCD_NBI_physics(
            config.A_beam, config.E_beam_keV,
            a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
            config.rho_NBI,
            f_alpha=getattr(config, '_f_alpha', 0.04),
            angle_NBI_deg=config.angle_NBI_deg,
            rho_ped=rho_ped, n_ped_frac=n_ped_frac,
            T_ped_frac=T_ped_frac)


    elif CD_source == 'Multi':
        # Individual efficiencies
        gamma_LH  = f_etaCD_LH_physics(Tbar, Z_eff)
        gamma_EC  = f_etaCD_EC_physics(a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                                config.rho_EC,
                                theta_EC_pol_deg=config.theta_EC_pol_deg,
                                rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                T_ped_frac=T_ped_frac)
        gamma_NBI = f_etaCD_NBI_physics(
                        config.A_beam, config.E_beam_keV,
                        a, R0, Tbar, nbar, Z_eff, nu_T, nu_n,
                        config.rho_NBI,
                        f_alpha=getattr(config, '_f_alpha', 0.04),
                        angle_NBI_deg=config.angle_NBI_deg,
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
    # ── ITER chain (5/12) - current drive and current decomposition ─────
    # Part A: PIPB-style figures of merit at the historical comparison
    # point (rho = 0.3, Tbar = 8.9 keV) against the ITER Chapter 6
    # projections (Gormezano et al., NF 47 (2007) S285), kept as
    # informative context: gamma_LH ~ 0.24-0.33, gamma_EC ~ 0.20,
    # gamma_NBI ~ 0.15-0.40 MA/(MW m2) across the ITER scenarios.
    _kwq = dict(a=2.0, R0=6.2, Tbar=8.9, nbar=1.01, nu_n=0.1, nu_T=1.0,
                Z_eff=1.6, rho_ped=0.94, n_ped_frac=0.80, T_ped_frac=0.40)
    _gLH_pub = f_etaCD_LH_physics(_kwq['Tbar'], _kwq['Z_eff'])
    _gEC_pub = f_etaCD_EC_physics(_kwq['a'], _kwq['R0'], _kwq['Tbar'],
                                  _kwq['nbar'], _kwq['Z_eff'], _kwq['nu_T'],
                                  _kwq['nu_n'], rho_EC=0.3,
                                  rho_ped=_kwq['rho_ped'],
                                  n_ped_frac=_kwq['n_ped_frac'],
                                  T_ped_frac=_kwq['T_ped_frac'])
    _gNBI_pub = f_etaCD_NBI_physics(2, 1000., _kwq['a'], _kwq['R0'],
                                    _kwq['Tbar'], _kwq['nbar'], _kwq['Z_eff'],
                                    _kwq['nu_T'], _kwq['nu_n'], rho_NBI=0.3,
                                    rho_ped=_kwq['rho_ped'],
                                    n_ped_frac=_kwq['n_ped_frac'],
                                    T_ped_frac=_kwq['T_ped_frac'])
    # Part B: chain evaluation at the deck point (Tbar = 7.754 keV,
    # rho_EC = 0.40 for NTM control, rho_NBI = 0.30, 1 MeV D, 20 deg).
    _eLH = f_etaCD_LH_physics(ITER['Tbar'], ITER['Zeff'])
    _eEC = f_etaCD_EC_physics(ITER['a'], ITER['R0'], ITER['Tbar'],
                              ITER['nbar'], ITER['Zeff'], ITER['nu_T'],
                              ITER['nu_n'], ITER['rho_EC'],
                              rho_ped=ITER['rho_ped'],
                              n_ped_frac=ITER['n_ped_frac'],
                              T_ped_frac=ITER['T_ped_frac'])
    _eNBI = f_etaCD_NBI_physics(ITER['A_beam'], ITER['E_beam_keV'], ITER['a'],
                                ITER['R0'], ITER['Tbar'], ITER['nbar'],
                                ITER['Zeff'], ITER['nu_T'], ITER['nu_n'],
                                ITER['rho_NBI'], f_alpha=FROZEN['f_alpha'],
                                angle_NBI_deg=ITER['angle_NBI_deg'],
                                rho_ped=ITER['rho_ped'],
                                n_ped_frac=ITER['n_ped_frac'],
                                T_ped_frac=ITER['T_ped_frac'])
    _IEC = f_I_CD(ITER['R0'], ITER['nbar'], _eEC, ITER['P_ECRH'])
    _INBI = f_I_CD(ITER['R0'], ITER['nbar'], _eNBI, ITER['P_NBI'])
    _ICD = _IEC + _INBI            # ICRH drives no current (gamma_ICR = 0)
    ITER.update(I_CD=_ICD)
    _bench("ITER chain 5/12 - current drive (Multi: NBI + EC + IC)", [
        ("gamma_LH, PIPB point [MA/(MW m2)]", _gLH_pub, "~0.24-0.33", None,
         "Gormezano 2007"),
        ("gamma_EC, PIPB point (rho=0.3)", _gEC_pub, "~0.20", None,
         "Gormezano 2007"),
        ("gamma_NBI, PIPB point", _gNBI_pub, "0.15-0.40", None,
         "Gormezano 2007"),
        ("eta_LH chain [MA/(MW m2)]", _eLH, FROZEN['eta_LH'], 2e-3,
         "deck frozen"),
        ("eta_EC chain (rho=0.40)", _eEC, FROZEN['eta_EC'], 2e-3,
         "deck frozen"),
        ("eta_NBI chain (1 MeV, 20 deg)", _eNBI, FROZEN['eta_NBI'], 2e-3,
         "deck frozen"),
        ("I_EC [MA]", _IEC, None, None, "P_EC = 6.7 MW"),
        ("I_NBI [MA]", _INBI, None, None, "P_NBI = 33 MW"),
        ("I_CD total [MA]", _ICD, FROZEN['I_CD'], 2e-3, "deck frozen"),
    ], notes=[
        "Chain efficiencies are lower than the PIPB-point values because "
        "the deck deposits EC off-axis (rho = 0.40, NTM control) at the "
        "solved Tbar = 7.75 keV.",
        "Current budget context (Eriksson et al., NF 64 (2024) 126033): "
        "inductive ~10 MA, non-inductive ~5 MA with bootstrap dominant.",
    ])

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

    where S = 2πR₀ × P_e is the plasma surface area [m²], with P_e the
    Ramanujan ellipse perimeter (semi-axes a and κa).

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
    S = 2.0 * np.pi * R0 * _ramanujan_perimeter(a, kappa * a)
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
    S     = 2.0 * np.pi * R0 * _ramanujan_perimeter(a, kappa * a)
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
    S = 2.0 * np.pi * R0 * _ramanujan_perimeter(a, kappa * a)
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
    # ── Published anchors - L-H power threshold ──────────────────────────
    # Martin, JPCS 123 (2008) 012033, against the PUBLISHED ITER
    # predictions of Table 5 in the ITPA TC-26 paper (NF 2026,
    # 10.1088/1741-4326/ae39f2): 52.3 MW at 0.5e20 m-3 and 86.0 MW at
    # 1.0e20 m-3 (deuterium, full field). Tolerance 10 %: D0FUS evaluates
    # the plasma surface from the Ramanujan ellipse perimeter, the
    # reference from the true ITER LCFS (~680 m2). The 'New_S' option
    # (TC-26 draft 2017, Delabie) must sit within the published 1-sigma
    # intervals of TC-26(Bt): P/S = (0.0441+-0.0025) B^(0.580+-0.039)
    # n^(1.08+-0.03) (2/M)^(0.975+-0.032); updating to the published
    # central values (2-4 % on P_LH) is a deliberate maintainer decision.
    _b_ref = P_Thresh_New_S(1., 1., 2., 6.2, 1.7, 2.)
    _bench("Published anchors - L-H threshold (Martin, New_S)", [
        ("P_LH Martin, 0.5e20, D [MW]",
         P_Thresh_Martin(0.5, 5.3, 2.0, 6.2, 1.85, 2.0), 52.3, 0.10,
         "ITPA TC-26 2026"),
        ("P_LH Martin, 1.0e20, D [MW]",
         P_Thresh_Martin(1.0, 5.3, 2.0, 6.2, 1.85, 2.0), 86.0, 0.10,
         "ITPA TC-26 2026"),
        ("New_S exponent on B [-]",
         float(np.log2(P_Thresh_New_S(1., 2., 2., 6.2, 1.7, 2.) / _b_ref)),
         0.580, 0.07, "TC-26 1-sigma"),
        ("New_S exponent on n [-]",
         float(np.log2(P_Thresh_New_S(2., 1., 2., 6.2, 1.7, 2.) / _b_ref)),
         1.08, 0.037, "TC-26 1-sigma"),
    ])

if __name__ == "__main__":
    # ── ITER chain (6/12) - L-H threshold and separatrix power ──────────
    # The Martin scaling is evaluated with the LINE-averaged chain density
    # and the D-T isotope mass M = 2.5 (P_LH proportional to 1/M), as in
    # the production solver. P_sep = P_alpha + P_CD - P_rad,total
    # (f_P_sep; P_Ohm is not included, matching D0FUS_run.py). The
    # metal-wall correction (W/Be vs C, factor ~0.70: Ryter et al., NF 53
    # (2013) 113003; Maggi et al., NF 54 (2014) 023007) yields the ~50 MW
    # best estimate usually quoted for ITER.
    _PLH = P_Thresh_Martin(ITER['nbar_line'], ITER['B0'], ITER['a'],
                           ITER['R0'], ITER['kappa'], ITER['M'])
    _Psep = f_P_sep(ITER['P_fus'], ITER['P_aux'], ITER['P_rad_tot'])
    ITER.update(P_sep=_Psep, P_LH_th=_PLH)
    _bench("ITER chain 6/12 - L-H threshold and P_sep", [
        ("P_LH Martin (D-T, M=2.5) [MW]", _PLH, FROZEN['P_LH_th'], 2e-3,
         "deck frozen"),
        ("P_LH x 0.70 metal wall [MW]", 0.70 * _PLH, "~50", None,
         "Ryter 2013 / Maggi 2014"),
        ("P_sep [MW]", _Psep, FROZEN['P_sep'], 2e-3, "deck frozen"),
        ("H-mode access P_sep/P_LH [-]", _Psep / _PLH, (1.0, 3.0), 1,
         "operational"),
    ])

def f_P_LH_thresh(nbar, B0, a, R0, kappa, M_ion, Ip=None,
                  Option_PLH='Martin'):
    """Alias for f_P_thresh — kept for backward compatibility."""
    return f_P_thresh(nbar, B0, a, R0, kappa, M_ion, Ip, Option_PLH)


#%% Plasma resistivity
# ─────────────────────────────────────────────────────────────────────────────
# Classical and neoclassical parallel resistivity models.
# Used by f_Reff (loop voltage / flux consumption), f_q_profile_refined
# (Ohmic current density via local sigma_neo(T, q, eps, Z_eff)), and all
# functions requiring local η(ρ).
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
        24.0 - np.log(ne_cm3**0.5 * T_safe**(-1.0))
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
        T > 10 eV:  ln(L) = 24.0 - ln(ne_cm3^0.5 * T_eV^-1)

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

    Key relations (Sauter Eqs. 13a-b):
        sigma_neo/sigma_Sp = 1 - (1+0.36/Z)*X + 0.59/Z*X^2 - 0.23/Z*X^3
        f_t_eff = f_t / [1 + (0.55 - 0.1*f_t)*sqrt(nu*)
                         + 0.45*(1-f_t)*nu*/Z^1.5]

    Collisionality (Eq. 18b, with ln Lambda_e of Eq. 18d):
        nu*_e = 6.921e-18 * q*R*n_e*Z_eff*ln(L_e) / (T_e[eV]^2 * eps^1.5)

    Verified line by line against sigmaneo.m / nustar.m of Sauter's NEOS
    repository (https://gitlab.epfl.ch/spc/public/NEOS).  Note: the 2002
    erratum does not modify Eqs. 13 or 18; it only fixes the sign of the
    alpha bootstrap coefficient and clarifies Z conventions.

    All four physics inputs (T_keV, ne, epsilon, q) accept either scalars
    or numpy arrays of identical shape; vectorisation is element-wise.

    References:
        [1] O. Sauter et al., Phys. Plasmas 6, 2834 (1999)
        [2] O. Sauter et al., Erratum, Phys. Plasmas 9, 5140 (2002)
        [3] https://crppwww.epfl.ch/~sauter/neoclassical/
    """
    # Geometric trapped fraction (Kim et al. PoF 1991).
    # Regularise epsilon at the magnetic axis: epsilon(rho=0) = 0 would
    # cause a division by zero in nu_star.  At epsilon -> 0: f_t -> 0
    # and the neoclassical correction vanishes (sigma_neo -> sigma_Spitzer).
    # np.maximum is array-safe; the function therefore accepts both
    # scalar and ndarray inputs.
    eps_reg  = np.maximum(epsilon, 1e-6)
    sqrt_eps = np.sqrt(eps_reg)
    f_t      = 1.46 * sqrt_eps * (1 - 0.54 * sqrt_eps)

    # Electron collisionality — Sauter (1999) Eq. (18b) with the Coulomb
    # logarithm of Eq. (18d): ln Lambda_e = 31.3 - ln(sqrt(n_e)/T_e),
    # n_e in m^-3 and T_e in eV.  Identical to _nu_e_star() used by the
    # bootstrap module and to nustar.m in Sauter's open-source NEOS code
    # (https://gitlab.epfl.ch/spc/public/NEOS).  T_keV is clamped from
    # below to avoid the 1/T^2 divergence when T -> 0 in the cold edge.
    T_safe    = np.maximum(T_keV, 1e-4)
    Te_eV     = T_safe * 1.0e3
    ln_e      = 31.3 - np.log(np.sqrt(np.maximum(ne, 1.0)) / Te_eV)
    nu_star_e = (6.921e-18 * q * R0 * ne * Z_eff * ln_e
                 / (Te_eV**2 * eps_reg**1.5))

    # Effective trapped fraction (Eq. 13b).
    sqrt_nu = np.sqrt(nu_star_e)
    denom = (1.0 + (0.55 - 0.1*f_t)*sqrt_nu
             + 0.45*(1.0 - f_t)*nu_star_e / Z_eff**1.5)
    f_t_eff = f_t / denom

    # Conductivity ratio (Eq. 13a).
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

    All four physics inputs (T_keV, ne, epsilon, q) accept either scalars
    or numpy arrays of identical shape; vectorisation is element-wise.

    References:
        [1] A. Redl et al., Phys. Plasmas 28, 022502 (2021)
        [2] Data: https://doi.org/10.5281/zenodo.4072358
    """
    # Geometric trapped fraction.  Regularise epsilon at the magnetic
    # axis (see eta_sauter for rationale).  np.maximum makes the call
    # array-safe for the q-profile Picard iteration.
    eps_reg  = np.maximum(epsilon, 1e-6)
    sqrt_eps = np.sqrt(eps_reg)
    f_t      = 1.46 * sqrt_eps * (1 - 0.54 * sqrt_eps)

    # Electron collisionality — Sauter (1999) Eq. (18b) with ln Lambda_e
    # of Eq. (18d); Redl (2021) keeps the same nu*_e definition.
    # Identical to _nu_e_star() (bootstrap module) and NEOS nustar.m.
    # T_keV is clamped from below to avoid 1/T^2 divergence at the edge.
    T_safe    = np.maximum(T_keV, 1e-4)
    Te_eV     = T_safe * 1.0e3
    ln_e      = 31.3 - np.log(np.sqrt(np.maximum(ne, 1.0)) / Te_eV)
    nu_star_e = (6.921e-18 * q * R0 * ne * Z_eff * ln_e
                 / (Te_eV**2 * eps_reg**1.5))

    # Effective trapped fraction (Eq. 18).
    sqrt_nu = np.sqrt(nu_star_e)
    denom = (1.0 + 0.25*(1-0.7*f_t)*sqrt_nu*(1 + 0.45*(Z_eff-1)**0.5)
             + 0.61*(1-0.41*f_t)*nu_star_e/np.sqrt(Z_eff))
    f_t_eff = f_t / denom

    # Conductivity ratio (Eq. 17).
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
    In steady-state flat-top, Maxwell's equation (nabla x E = 0) in
    axisymmetric geometry gives E_phi(R) = E_phi_0 * R0 / R.  The
    Ohmic current density at normalised radius rho is therefore:

        j_Ohm(rho) = E_phi_0 * R0 / (R * eta_neo(rho))

    Integrating over the cross-section with dA = V'(rho) drho / (2 pi R0)
    and using V_loop = 2 pi R0 * E_phi_0 yields V_loop = R_eff * I_Ohm:

        R_eff = (2 pi R0)^2
                / integral_0^1 <R0/R>_rho * V'(rho) / eta_neo(rho) drho

    where <R0/R>_rho is the flux-surface average of R0/R.  For concentric
    circular surfaces:  <R0/R>_rho = 1 / sqrt(1 - (rho * a / R0)^2).

    The current profile j(rho) ~ sigma_neo(rho) emerges self-consistently
    from T(rho) and n(rho) via the neoclassical conductivity — no
    prescribed alpha_J exponent is used.

    Parameters
    ----------
    a : float
        Plasma minor radius [m].
    kappa : float
        Plasma elongation [-].
    R0 : float
        Plasma major radius [m].
    Tbar : float
        Volume-averaged electron temperature [keV].
    nbar : float
        Volume-averaged electron density [1e20 m^-3].
    Z_eff : float
        Effective ionic charge [-].
    q95 : float
        Safety factor at 95% flux surface [-].
    nu_T : float
        Temperature profile peaking exponent [-].
    nu_n : float
        Density profile peaking exponent [-].
    eta_model : str, optional
        Resistivity model: 'old', 'spitzer', 'sauter', 'redl' (default).
    rho_ped : float, optional
        Normalised pedestal radius (1.0 = no pedestal).
    n_ped_frac : float, optional
        n_ped / nbar.
    T_ped_frac : float, optional
        T_ped / Tbar.
    Vprime_data : tuple or None, optional
        (rho_grid, Vprime, V_total) from precompute_Vprime().
        When None, cylindrical V' = 4 pi^2 R0 a^2 kappa rho is used.

    Returns
    -------
    R_eff : float
        Effective plasma resistance [Ohm].

    References
    ----------
    Sauter O. et al., Phys. Plasmas 6, 2834 (1999).
    Redl A. et al., Phys. Plasmas 28, 022502 (2021).
    Johner J., Fusion Sci. Technol. 59, 308 (2011), Eq. 31.
        HELIOS — only other 0D code including the 1/R correction.
    """

    def _eta_local(rho):
        """Neoclassical resistivity [Ohm.m] at normalised radius rho."""
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

    def _R0_over_R_fsa(rho):
        """Flux-surface average <R0/R> for concentric circular surfaces."""
        eps_rho = rho * a / R0
        return 1.0 / math.sqrt(1.0 - eps_rho**2)

    def _integrand(rho):
        """Shaped conductance integrand with 1/R correction."""
        return _R0_over_R_fsa(rho) * _Vprime_local(rho) / max(_eta_local(rho), 1e-12)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=IntegrationWarning)
        sigma_integral, _ = quad(
            _integrand, 0, 1.0,
            limit=200, epsabs=1e-8, epsrel=1e-4)

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

if __name__ == "__main__":
    # ── ITER chain (7/12) - ohmic power and resistivity models ──────────
    # P_Ohm closes the power balance at the 0.4 MW level, negligible for
    # ITER but the natural place to exercise the resistivity chain. The
    # frozen I_Ohm and q95 are forward references, both closed by
    # chain 12 (I_Ohm = Ip - I_bs - I_CD and the q95 inversion). The
    # four implemented resistivity models are compared at the chain
    # point: the neoclassical values (Sauter 1999; Redl 2021) must exceed
    # Spitzer because trapped electrons cannot carry parallel current.
    _Palpha = f_P_alpha(ITER['P_fus'])
    _po = {m: f_P_Ohm(FROZEN['I_Ohm'], ITER['Tbar'], ITER['R0'], ITER['a'],
                      ITER['kappa'], Z_eff=ITER['Zeff'], nbar=ITER['nbar'],
                      eta_model=m, q95=FROZEN['q95'])
           for m in ('old', 'spitzer', 'sauter', 'redl')}
    ITER.update(P_alpha=_Palpha, P_Ohm=_po['sauter'])   # deck: eta_model=sauter
    _bench("ITER chain 7/12 - ohmic power and resistivity models", [
        ("P_alpha = P_fus Ea/(Ea+En) [MW]", _Palpha,
         ITER['P_fus'] * E_ALPHA / (E_ALPHA + E_N), 1e-12, "definition"),
        ("P_alpha approx P_fus/5 [MW]", _Palpha, 100.0, 1e-3, "rule of thumb"),
        ("P_Ohm Sauter (deck) [MW]", _po['sauter'], FROZEN['P_Ohm'], 5e-3,
         "deck frozen"),
        ("P_Ohm Spitzer [MW]", _po['spitzer'], None, None, "model comparison"),
        ("P_Ohm Redl [MW]", _po['redl'], None, None, "model comparison"),
        ("P_Ohm Wesson 'old' [MW]", _po['old'], None, None, "model comparison"),
        ("neoclassical enhancement Sauter/Spitzer",
         _po['sauter'] / _po['spitzer'], (1.0, 5.0), 1, "trapped electrons"),
        ("Sauter vs Redl consistency",
         _po['sauter'] / _po['redl'], (0.7, 1.4), 1, "model agreement"),
    ])


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
    appropriate figure-of-merit γ from f_etaCD_LH_physics / f_etaCD_EC_physics / f_etaCD_NBI_physics.

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
    # ── ITER chain (8/12) - fusion gain ──────────────────────────────────
    # Q = P_fus / (P_aux + P_Ohm) with the flat-top heating mix of the
    # deck (33 NBI + 6.7 EC + 10 IC = 49.7 MW [Kim 2018]) and the chain
    # ohmic power from chain 4. The published design target is Q = 10
    # with ~50 MW of auxiliary power (Shimada 2007).
    _Q = f_Q_multiaux(P_fus=ITER['P_fus'], P_LH=ITER['P_LH'],
                      P_ECRH=ITER['P_ECRH'], P_NBI=ITER['P_NBI'],
                      P_ICRH=ITER['P_ICRH'], P_Ohm=ITER['P_Ohm'])
    ITER.update(Q=_Q)
    _bench("ITER chain 8/12 - fusion gain", [
        ("Q = P_fus/(P_aux + P_Ohm) [-]", _Q, FROZEN['Q'], 2e-3,
         "deck frozen"),
        ("Q [-]", _Q, 10.0, 0.05, "Shimada 2007"),
    ])

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

Implementation follows Sauter Eq. 5 directly:

    <j_bs . B> = -I(psi) * p * [ L31 * d(ln p)/d(psi_hat)
                                + L32 * R_pe * d(ln Te)/d(psi_hat)
                                + L34 * alpha * (1-R_pe) * d(ln Ti)/d(psi_hat) ]

    j_bs [A/m^2] = <j_bs . B> / <B^2>
    I_bs [MA]    = integral( j_bs * 2*pi*rho*a^2*kappa(rho) drho )

where I(psi) = R0*B0, p is the LOCAL total pressure, and <B^2> = B0^2*(1+eps^2/2)
in the large aspect ratio approximation.  This is consistent with the recommendation
on the NEOS page (https://crppwww.epfl.ch/~sauter/neoclassical/):
  "Total pressure should be used for p and pe/p should be used where stated
   and not some approximations with Te and Ti for example."

Validation against NEOS (Sauter's reference Fortran code)
---------------------------------------------------------
The Sauter coefficients L31, L32, L34, and alpha implemented in D0FUS have been
verified term-by-term against the NEOS Fortran 90 module (neobscoeffmod.f90),
the reference implementation by O. Sauter himself, distributed under Apache 2.0
license by SPC-EPFL (2025):
    https://gitlab.epfl.ch/spc/public/NEOS

All effective trapped fractions (f31_eff, f32ee_eff, f32ei_eff, f34_eff), the
polynomial F31(X,Z), the L32 = L32_ee + L32_ei decomposition, and the alpha
coefficient (with errata +0.315) match NEOS to machine precision (< 1e-14
relative error) across the full range of f_t, nu*, and Zeff tested.

Comparison with PROCESS (Fable formulation)
-------------------------------------------
PROCESS v1.0.10 (ibss=4, "Sauter" option) uses an unpublished reformulation
attributed to E. Fable (IPP Garching, private communication).  While the Sauter
coefficients (L31/L32/L34/alpha) are identical to NEOS, the j_bs assembly differs:

  PROCESS (Fable):
    - Decomposes d(ln p) into d(ln n) + d(ln T) separately
    - Absorbs pressure into local beta_poloidal: beta_p(rho) = 2*mu0*p/Bp^2
    - Uses circularized coordinates rho_circ = a*sqrt(kappa)*rho_norm
    - Prefactor: 0.5 * 1e6 * (-B0*rho_circ)/(0.2*pi*R0*q)
    - Factor 0.5 compensates a convention in beta_p averaging (sum vs mean)

  refined  (Sauter Eq. 5 direct):
    - Uses d(ln p)/dr directly (total pressure gradient)
    - Multiplies by p(rho) / <B^2>(rho) with <B^2> = B0^2*(1+eps^2/2)
    - Prefactor: -R0*B0 (= I(psi))
    - No additional numerical factors

The PROCESS documentation itself acknowledges (April 2025):
  "In its current state the several base functions called by the Sauter scaling
   have no reference and cannot be verified.  The ad-hoc adaption of the Sauter
   scaling for use in PROCESS is done knowing that PROCESS does not calculate
   flux surfaces across the plasma."
  (https://ukaea.github.io/PROCESS/physics-models/plasma_current/bootstrap_current/)

A dedicated comparison (bootstrap_comparison.py) feeding IDENTICAL profiles and
IDENTICAL Sauter coefficients into both assembly formulas yields:
    I_bs(Sauter Eq.5) / I_bs(Fable) = 1.40 (+40%)
for EU-DEMO 2017 conditions.  The ratio is approximately constant (~1.43) from
rho = 0.3 to the pedestal top, indicating a systematic geometric prefactor
difference rather than a localised numerical artifact.

References for this validation
------------------------------
[5] NEOS — neobscoeffmod.f90, O. Sauter, SPC-EPFL, Apache 2.0, 2025.
    https://gitlab.epfl.ch/spc/public/NEOS
    https://crppwww.epfl.ch/~sauter/neoclassical/

[6] PROCESS bootstrap_current.py, E. Fable (private comm.), UKAEA.
    https://github.com/ukaea/PROCESS
    Documentation: https://ukaea.github.io/PROCESS/physics-models/plasma_current/bootstrap_current/

[7] A. Redl et al., Phys. Plasmas 28 (2021) 022502.
    Improved coefficients at high collisionality (pedestal), refitted against NEO.
    DOI: 10.1063/5.0012664

[8] E.A. Belli, J. Candy, Plasma Phys. Control. Fusion 54 (2012) 015015.
    NEO drift-kinetic code used as reference for Sauter/Redl validation.

"""

# ==============================================================================
# Internal Functions — Profile logarithmic gradients (used by Sauter and Redl)
# ==============================================================================
# Internal Functions (Sauter model)
# ==============================================================================

def _trapped_fraction(epsilon, fit='Sauter2002'):
    """
    Geometric trapped particle fraction f_t(ε).

    The trapped fraction enters the Sauter/Redl bootstrap coefficients
    via the effective trapped fractions f_t_eff(ν*).  At low collisionality
    (banana regime, ν*_e < 0.1, typical of reactor-grade plasmas), the
    bootstrap current is approximately proportional to f_t.

    Shaping effects (elongation, triangularity) on the bootstrap current
    are captured by the shaped area element dA = 2π ρ a² κ(ρ) dρ and the
    pressure gradients, not by modifying the trapped fraction formula.
    This is consistent with NEOS, PROCESS, ASTRA, JINTRAC, and CRONOS.

    Parameters
    ----------
    epsilon : float or ndarray
        Local inverse aspect ratio ε = ρ a / R₀.
    fit : str, optional
        Trapped fraction formula.  Options:

        'Sauter2002' (default, recommended)
            Equation 4 of Sauter et al., PPCF 44 (2002) 1999.
            f_t = 1 - (1-ε)² / [(1+1.46√ε)√(1-ε²)]
            Standard in NEOS (Sauter's own code), JINTRAC, and CRONOS.

        'ASTRA'
            ASTRA formula from Fable (IPP Garching, private comm.).
            f_t = 1 - (1-ε)^{3/2} / (1+1.46√ε)
            Used by PROCESS (fit=0) and ASTRA transport code.

    Returns
    -------
    f_t : float or ndarray
        Geometric trapped particle fraction (0 < f_t < 1).

    References
    ----------
    Sauter et al., PPCF 44 (2002) 1999 — Eq. 4 ('Sauter2002').
    Lin-Liu & Miller, Phys. Plasmas 2 (1995) 1666 — derivation.
    Fable, IPP Garching (private comm.) — ASTRA formula ('ASTRA').
    """
    sqrt_eps = np.sqrt(np.maximum(epsilon, 0.0))

    if fit == 'Sauter2002':
        # Sauter, PPCF 44 (2002) 1999, Eq. 4
        return 1.0 - ((1.0 - epsilon)**2
                       / ((1.0 + 1.46 * sqrt_eps) * np.sqrt(1.0 - epsilon**2)))

    elif fit == 'ASTRA':
        # ASTRA/Fable — PROCESS default (fit=0 in bootstrap_current.py)
        return 1.0 - (1.0 - epsilon)**1.5 / (1.0 + 1.46 * sqrt_eps)

    else:
        raise ValueError(
            f"Unknown trapped fraction model '{fit}'. "
            f"Options: 'Sauter2002', 'ASTRA'."
        )


def _nu_e_star(n_e, T_e, q, R0, epsilon, Z_eff):
    """Electron collisionality [Eq. 18b]."""
    ln_Lambda = 31.3 - np.log(np.sqrt(n_e) / T_e)
    return 6.921e-18 * q * R0 * n_e * Z_eff * ln_Lambda / (T_e**2 * epsilon**1.5)


if __name__ == "__main__":
    # ── Published anchor - electron collisionality identity ──────────────
    # Sauter, Angioni & Lin-Liu, Phys. Plasmas 6 (1999) 2834, Eq. 18b
    # with the Coulomb logarithm of Eq. 18d (helper takes T_e in eV).
    # The ITER core sits deep in the banana regime, nu*_e ~ 0.03.
    _ne, _Te = 1.0e20, 8900.0
    _lnL = 31.3 - np.log(np.sqrt(_ne) / _Te)
    _nref = 6.921e-18 * 3.0 * 6.2 * _ne * 1.7 * _lnL / (_Te**2 * (2 / 6.2)**1.5)
    _bench("Published anchor - Sauter collisionality (Eq. 18b)", [
        ("nu*_e identity, ITER core [-]",
         float(_nu_e_star(_ne, _Te, 3.0, 6.2, 2 / 6.2, 1.7)), _nref, 1e-6,
         "Sauter 1999"),
        ("banana-regime check nu*_e", float(_nref), (0.0, 0.1), 1,
         "ITER core"),
    ])

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
# NOTE: moved here (before f_Sauter_Redl_Ib) so that the inline
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

    On axis : q(0) = q_edge / (αJ + 1).

    This profile is consistent with the j-profile model used in f_Reff()
    for the loop voltage calculation.  The internal inductance l_i(3) is
    computed numerically inside f_q_profile_academic(), which calls this
    function as its parametric q(rho) builder.

    The internal inductance l_i increases monotonically with αJ:
        l_i(αJ=0) = 0.50, l_i(αJ=1.5) ≈ 1.08, l_i(αJ=3) ≈ 1.55.
    Computed in f_q_profile_academic() from the integral definition of
    l_i(3) (Luce 2014, ITER convention).

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


# ─────────────────────────────────────────────────────────────────────────────
# Academic q,j profile builder (Mode A pedagogical entry point)
# ─────────────────────────────────────────────────────────────────────────────


def f_q_profile_academic(
        Ip, q95,
        R0, a, kappa,
        alpha_J=1.5,
        rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
        Vprime_data=None, kappa_95=None, rho_95=0.95,
        n_rho=200):
    """
    Parametric safety-factor and current-density profiles (Mode 'academic').

    The current density follows the standard PROCESS / Uckan ansatz

        j(rho) = j0 * (1 - rho^2)^alpha_J

    from which the cylindrical Ampère integral yields

        I_enc(rho)/Ip = 1 - (1 - rho^2)^(alpha_J + 1)
        q(rho)        = q_edge * rho^2 / I_enc(rho)/Ip
        q_0           = q_edge / (alpha_J + 1)

    The edge normalisation is q(rho_95) = q95, with q95 supplied by the
    caller from any external scaling (Sauter 2016, Uckan 1989, MHD limit).
    The user owns alpha_J as a pedagogical input; no coupling to bootstrap,
    current-drive, or Ohmic physics takes place inside this function.  It is
    deterministic and runs in a single pass, suitable for tutorials,
    benchmarks, or cross-checks of more elaborate self-consistent solvers.

    The internal inductance l_i(3) (ITER / EFIT convention, Luce 2014) is
    integrated numerically on the same Miller-aware geometric weights that
    f_q_profile_refined uses, so the two modes share a common diagnostic.

    Parameters
    ----------
    Ip : float
        Plasma current [MA].
    q95 : float
        Safety factor at rho = rho_95, imposed as edge normalisation.
    R0, a, kappa : float
        Major radius [m], minor radius [m], edge elongation [-].
    alpha_J : float
        Current density peaking exponent.  Default 1.5 (IPDG89 / PROCESS).
    rho_ped, n_ped_frac, T_ped_frac : float
        Profile pedestal parameters.  Unused by the academic q-profile per
        se but accepted for signature compatibility with the refined branch.
    Vprime_data : tuple or None
        Miller geometry data (rho_grid, Vprime, V_total, dA_grid, Lp_grid).
        When None, cylindrical-torus weights are used.
    kappa_95, rho_95 : float
        95%-surface elongation and radial position.
    n_rho : int
        Radial grid resolution.

    Returns
    -------
    dict with keys (matching f_q_profile_refined for caller compatibility):
        q_arr     : ndarray   Safety-factor profile q(rho).
        rho       : ndarray   Radial grid.
        j_total   : ndarray   Parametric j(rho) [A/m^2], normalised to Ip.
        j_Ohm     : ndarray   Zeros (no current decomposition in this mode).
        j_CD      : ndarray   Zeros.
        j_non_bs  : ndarray   Zeros.
        j_bs      : ndarray   Zeros.
        I_enc     : ndarray   Enclosed current Ip * I_frac(rho) [A].
        li        : float     Internal inductance l_i(3) [-].
        q0        : float     Central safety factor (analytic).
        alpha_J   : float     Echo of the input exponent.
        I_bs      : float     0.0 (bootstrap not modelled here).
        f_bs      : float     0.0.
        n_iter    : int       1 (deterministic).
        converged : bool      True.
        kappa_arr : ndarray   Local elongation profile.
        q_at_95   : float     q(rho_95).  Equals q95 by construction here
                              (kept for return-shape symmetry with refined).

    References
    ----------
    Wesson, Tokamaks, 4th ed. (2011) ch. 3 — cylindrical j–q relation.
    Uckan et al., ITER Physics Design Guidelines, IAEA/ITER/DS/10 (1990).
    Kovari et al., Fus. Eng. Des. 89 (2014) 3054 — PROCESS j ∝ (1−ρ²)^αJ.
    Hender et al., AEA FUS 172 (1992) — l_i(α_J) analytical relation.
    Luce et al., Nucl. Fusion 54 (2014) 093005 — l_i(3) integral definition.
    """
    # == Radial grid =========================================================
    # Same layout as the refined mode for diagnostic comparability.
    rho_inner = np.linspace(0.001, 0.05, 30, endpoint=False)
    rho_core  = np.linspace(0.05, rho_ped, n_rho, endpoint=False)
    rho_edge  = np.linspace(rho_ped, 0.99, n_rho)
    rho = np.concatenate([rho_inner, rho_core, rho_edge])

    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa)

    use_miller = (Vprime_data is not None)
    if use_miller:
        kappa_arr = kappa_profile(rho, kappa, kappa_95, rho_95)
    else:
        kappa_arr = np.full_like(rho, kappa)

    # == Poloidal area element (Miller-consistent when available) ============
    drho = np.zeros_like(rho)
    drho[0]    = (rho[0] + rho[1]) / 2.0
    drho[-1]   = rho[-1] - rho[-2]
    drho[1:-1] = (rho[2:] - rho[:-2]) / 2.0
    if (Vprime_data is not None and len(Vprime_data) >= 5
            and Vprime_data[3] is not None):
        dA_per_drho = interpolate_dA(rho, Vprime_data[0], Vprime_data[3])
    else:
        dA_per_drho = 2.0 * np.pi * rho * a**2 * kappa_arr
    dA = dA_per_drho * drho

    # == Geometric weights for li(3) =========================================
    _lp_from_miller = (
        use_miller and Vprime_data is not None
        and len(Vprime_data) >= 5 and Vprime_data[4] is not None
    )
    if _lp_from_miller:
        Lp_arr = np.interp(rho, Vprime_data[0], Vprime_data[4])
    else:
        Lp_arr = _ramanujan_perimeter(rho * a, kappa_arr * rho * a)
    Lp_arr = np.where(rho > 1e-8, Lp_arr, 1.0)

    if use_miller and Vprime_data is not None:
        Vprime_arr = interpolate_Vprime(rho, Vprime_data[0], Vprime_data[1])
    else:
        Vprime_arr = 4.0 * np.pi**2 * R0 * a**2 * kappa_arr * rho

    # == Parametric q(rho) and j(rho) ========================================
    # j(rho) = j0 (1 - rho^2)^alpha_J, normalised so that integral(j dA) = Ip.
    Ip_A      = Ip * 1e6
    j_shape   = np.maximum(1.0 - rho**2, 0.0)**alpha_J
    j_dA_int  = float(np.sum(j_shape * dA))
    j_total   = (Ip_A * j_shape / j_dA_int) if j_dA_int > 0.0 else np.zeros_like(rho)

    # q(rho) from the analytical formula consistent with the same j ansatz.
    q_arr = f_q_profile(rho, q95=q95, rho95=rho_95, alpha_J=alpha_J)

    # Enclosed current (cumulative Ampère, monotonic by construction).
    I_enc = np.cumsum(j_total * dA)
    I_enc = np.maximum(I_enc, 1e-3)

    # == Internal inductance l_i(3) numerical integral =======================
    # Luce et al. NF 54 (2014) 093005, ITER/EFIT convention:
    #   l_i(3) = (2 / R0) * integral( (I_enc/Ip)^2 * V'(rho) / Lp(rho)^2 d rho )
    I_norm       = I_enc / np.maximum(I_enc[-1], 1e-3)
    li_integrand = np.where(rho > 1e-8,
                            I_norm**2 * Vprime_arr / Lp_arr**2,
                            0.0)
    li = (2.0 / R0) * np.trapezoid(li_integrand, rho)

    # == Analytic central safety factor ======================================
    ap1       = alpha_J + 1.0
    I_frac_95 = 1.0 - (1.0 - rho_95**2)**ap1
    q_edge    = q95 * I_frac_95 / rho_95**2
    q0_val    = q_edge / ap1

    return {
        'q_arr':     q_arr,
        'rho':       rho,
        'j_total':   j_total,
        'j_Ohm':     np.zeros_like(rho),
        'j_CD':      np.zeros_like(rho),
        'j_non_bs':  np.zeros_like(rho),
        'j_bs':      np.zeros_like(rho),
        'I_enc':     I_enc,
        'li':        li,
        'q0':        q0_val,
        'alpha_J':   alpha_J,
        'I_bs':      0.0,
        'f_bs':      0.0,
        'n_iter':    1,
        'converged': True,
        'kappa_arr': kappa_arr,
        'q_at_95':   float(q95),       # imposed by construction in this mode
    }



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
# Internal Functions (Redl model : NEO-fitted coefficients)
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

def f_Sauter_Redl_Ib(R0, a, kappa, B0, nbar, Tbar, q95, Z_eff, nu_n, nu_T, n_rho=100,
                     rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                     Vprime_data=None, kappa_95=None, rho_95=0.95,
                     return_profile=False, q_profile=None,
                     trapped_fraction_model='Sauter2002', tau_i_e=1.0):
    """
    Bootstrap current using the Sauter-Redl neoclassical model.

    Combines the analytical structure of Sauter (1999, 2002) with the
    refitted polynomial coefficients of Redl et al. (2021), giving
    improved accuracy at high collisionality (pedestal region) and for
    plasmas with impurities.  Supports parabolic and parabola-with-pedestal
    profile models.

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
        If provided, activates refined mode: area element dA and trapped
        fraction use the PCHIP local kappa(rho) instead of constant
        kappa_edge.
    kappa_95 : float or None, optional
        Elongation at the 95% flux surface.  Used only in refined mode.
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

    # Radial grid — core + pedestal edge (equal density).
    # Pedestal resolution tested: <1% effect on I_bs (60 to 800 pts).
    rho_core = np.linspace(0.05, rho_ped, n_rho, endpoint=False)
    rho_edge = np.linspace(rho_ped, 0.99, n_rho)
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
    # Safety factor profile for collisionality — see f_Sauter_Redl_Ib for rationale.
    if q_profile is not None and 'rho' in q_profile and 'q_arr' in q_profile:
        q_arr = np.interp(rho_arr, q_profile['rho'], q_profile['q_arr'],
                          left=q_profile['q_arr'][0], right=q95)
    else:
        q_arr = 1.0 + (q95 - 1.0) * rho_arr**2

    # SI units
    n_e  = n_arr * 1e20           # [m^-3]
    T_eV = T_arr * 1e3            # [eV]  electron temperature
    Ti_eV = tau_i_e * T_eV        # [eV]  ion temperature, T_i = tau_i_e * T_e
    n_i  = n_e / Z_eff

    # Pressure [Pa]
    p_e   = n_e * T_eV * E_ELEM
    p_i   = n_i * Ti_eV * E_ELEM
    p_tot = p_e + p_i
    R_pe  = np.where(p_tot > 0, p_e / p_tot, 0.5)

    # Local elongation: kappa(rho) in refined mode, kappa_edge in Academic
    if use_miller:
        kappa_arr = kappa_profile(rho_arr, kappa, kappa_95, rho_95)
    else:
        kappa_arr = np.full_like(rho_arr, kappa)

    # Trapped fraction and collisionalities (vectorised)
    f_t  = _trapped_fraction(eps_arr, fit=trapped_fraction_model)
    nu_e = _nu_e_star(n_e, T_eV, q_arr, R0, eps_arr, Z_eff)
    nu_i_arr = _nu_i_star(n_i, Ti_eV, q_arr, R0, eps_arr)   # ion collisionality on T_i

    # Redl coefficients (vectorised — all use only np operations)
    # Redl Eq. 19: L34 = L31 (simplification validated by NEO)
    L31 = _L31_Redl(f_t, nu_e, Z_eff)
    L32 = _L32_Redl(f_t, nu_e, Z_eff)
    L34 = L31                                    # Redl Eq. 19
    alp = _alpha_Redl(f_t, nu_i_arr, Z_eff)     # Redl α depends on Z_eff

    # Logarithmic gradients
    dln_p  = dln_n + dln_T   # T_i = tau_i_e T_e (const ratio): dln(p_tot) = dln_n + dln_Te
    dln_Ti = dln_T           # dln(T_i) = dln(tau_i_e T_e) = dln(T_e) for constant ratio

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

    # Poloidal cross-section area element dA = (dA_pol/dρ) × dρ.
    # Use the precomputed Miller-consistent profile from Vprime_data
    # when available (includes κ(ρ) and δ(ρ)); otherwise fall back to
    # the exact ellipse expression 2πρa²κ (valid at constant κ, no δ).
    if (Vprime_data is not None and len(Vprime_data) >= 5
            and Vprime_data[3] is not None):
        dA_per_drho = interpolate_dA(rho_arr, Vprime_data[0], Vprime_data[3])
    else:
        dA_per_drho = 2.0 * np.pi * rho_arr * a**2 * kappa_arr
    dA = dA_per_drho * drho

    # Mask out unphysical points (eps too small, n or T too low)
    valid = (eps_arr >= 0.01) & (n_arr >= 1e-3) & (T_arr >= 0.1)
    j_bs = np.where(valid, j_bs, 0.0)

    I_bs = np.sum(j_bs * dA) / 1e6   # [MA]

    if return_profile:
        return {'I_bs': I_bs, 'rho': rho_arr, 'j_bs': j_bs,
                'dA': dA, 'kappa_arr': kappa_arr, 'q_arr': q_arr,
                'drho': drho}
    return I_bs



# ─────────────────────────────────────────────────────────────────────────────
# Refined q,j profile builder (Picard self-consistency on q(rho) itself)
# ─────────────────────────────────────────────────────────────────────────────


def f_q_profile_refined(
        Ip, I_CD,
        R0, a, B0, kappa, nbar, Tbar, Z_eff,
        nu_n, nu_T,
        trapped_fraction_model='Sauter2002',
        eta_model='redl',
        rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
        Vprime_data=None, kappa_95=None, rho_95=0.95,
        delta=0.0, delta_95=None,
        rho_CD=0.3, delta_CD=0.15,
        q_init=None,
        n_rho=200, tol=1e-3, max_iter=15, damping=0.5,
        N_theta_inv_R2=400, tau_i_e=1.0):
    """
    Self-consistent safety-factor and current-density profiles (Mode 'refined').

    Picard iteration on the radial profile q(rho) itself (not on a scalar
    parametric exponent).  Each iteration evaluates the composite current
    density

        j_total(rho) = j_Ohm(rho) + j_CD(rho) + j_bs(rho)

    on the current q(rho), then rebuilds q(rho) from the rigorous local
    formula

        q(rho) = F * Lp(rho)^2 * <1/R^2>(rho) / (2 pi mu0 I_enc(rho))

    derived from the axisymmetric definition

        q(psi) = (F / 2 pi) * oint dl_p / (R^2 B_p)

    under the single approximation B_p ~ mu0 I_enc / Lp constant around
    the surface.  F = R B_phi ~ R0 B0 (vacuum-field, low-beta limit).

    The shape factor is captured by Lp(rho) and <1/R^2>(rho), both
    integrated on the Miller flux surface (R(rho, theta), Z(rho, theta)).
    There is no empirical correction term: the elongation kappa(rho), the
    triangularity delta(rho), and the finite aspect ratio epsilon = rho a/R0
    enter automatically via the geometric weights.  In the limits
    kappa = 1, delta = 0, epsilon -> 0 the formula reduces to the standard
    cylindrical q = 2 pi r^2 B_phi / (mu0 R0 I_enc).  For an ellipse at
    constant kappa with delta = 0 and epsilon -> 0, it reduces exactly to
    the Freidberg q* = pi a^2 B0 (1+kappa^2) / (mu0 R0 Ip) at rho = 1.

    Component physics
    -----------------
    j_bs : Sauter (1999/2002) structure with Redl (2021) refitted
           coefficients (f_Sauter_Redl_Ib).  Collisionality nu*_e and the
           L_31, L_32 coefficients see the local q(rho).
    j_CD : Gaussian deposition centred on rho_CD with width delta_CD,
           normalised to I_CD.  No physical-source decomposition here;
           the caller selects the deposition centre via the Multi-source
           current-weighted average.
    j_Ohm: Distributed according to the local neoclassical conductivity
           sigma_neo(T(rho), q(rho), eps(rho), Z_eff), normalised so that
           integral(j_Ohm dA) = Ip - I_CD - I_bs.

    Edge condition
    --------------
    q(rho_95) is *not* constrained to match the externally computed q95
    from f_q95.  The latter is an MHD-stability scaling (Sauter 2016,
    Uckan 1989) fitted on full Grad-Shafranov equilibria and includes
    Shafranov-shift and B_p-anisotropy effects that are absent here.
    The two values typically differ by 10-20% at ITER aspect ratio; this
    is a diagnostic of the consistency between the global scaling and
    the geometric reconstruction, not a numerical defect.

    Reversed shear (q with off-axis minimum) is permitted.  q(0) below 1
    is permitted and not corrected; in real tokamaks, sawtooth oscillations
    redistribute current to keep q(0) ~ 1, which is not modelled in this
    stationary 0D solver.  A 0D q(0) below 1 should be read as a signal
    that the prescribed current-drive deposition is too off-axis for
    sawtooth-free operation, not as a numerical bug.

    Pedestal bootstrap peak
    -----------------------
    In H-mode operation with rho_ped close to 1, the Sauter-Redl bootstrap
    current density develops a sharp peak in the pedestal layer (typically
    near rho ~ rho_ped + (1 - rho_ped) / 2), where the pressure gradient
    is maximal.  This peak is fully resolved by the dense radial grid used
    here (n_rho points in [rho_ped, 0.99]) and produces a localised
    minimum in q(rho) past rho_ped, followed by a recovery toward the LCFS.
    The minimum is not an artefact: it reflects the steady-state Ampere
    closure on the prescribed pedestal pressure profile.

    Reference 1.5D codes (CORSICA, ASTRA, METIS) typically produce
    monotonic q(rho) at the same operating point because they include
    resistive current diffusion, which redistributes the pedestal
    bootstrap toward the core over the resistive time scale tau_R ~
    R0^2 / eta_neo (a few hundred seconds for ITER).  D0FUS being a
    stationary 0D code, no time-dependent relaxation is captured: the
    profile returned here is the instantaneous Ampere balance for the
    prescribed pedestal at t = 0, before any resistive smoothing.  The
    overall q(0), q(rho_95), q(1), l_i(3), and bootstrap fraction
    integrals remain physically meaningful and benchmark against
    transient codes within typical 0D model uncertainty (~10 %).

    For applications where the pedestal q-dip is undesirable (e.g.
    visual presentation against transient simulations), the user can
    soften the pedestal pressure profile through input parameters
    (T_ped_frac smaller, n_ped_frac smaller, rho_ped lower); the dip
    flattens accordingly.  The dip cannot be eliminated by changing the
    solver itself without modelling resistive diffusion, which is out
    of scope for a 0D code.

    Parameters
    ----------
    Ip : float
        Plasma current [MA].
    I_CD : float
        Externally driven current (NBI, LHCD, ECCD) [MA].
    R0, a, B0, kappa : float
        Major radius [m], minor radius [m], on-axis field [T], edge
        elongation [-].
    nbar, Tbar, Z_eff : float
        Volume-averaged density [1e20 m^-3], temperature [keV], and
        effective charge [-].
    nu_n, nu_T : float
        Density and temperature peaking exponents.
    trapped_fraction_model : str
        Trapped-particle-fraction model passed to f_Sauter_Redl_Ib.
    eta_model : str
        Resistivity model used to shape j_Ohm: 'redl' (default), 'sauter',
        or 'spitzer'.  The neoclassical models couple j_Ohm to q(rho) via
        the trapped-particle correction.
    rho_ped, n_ped_frac, T_ped_frac : float
        Pedestal parameters passed to f_Tprof / f_nprof.
    Vprime_data : tuple or None
        Miller geometry data.  Accepted formats:
            6-tuple: (rho, Vprime, V_total, dA, Lp, inv_R2)  [preferred]
            5-tuple: (rho, Vprime, V_total, dA, Lp)          [legacy]
        When None, or when inv_R2 is missing, Lp(rho) and <1/R^2>(rho)
        are recomputed on the fly from the Miller parameterisation using
        the kappa, delta, kappa_95, delta_95 inputs.
    kappa_95, rho_95 : float
        95%-surface elongation and radial position.
    delta, delta_95 : float
        Edge and 95%-surface triangularity.  Used when Vprime_data does
        not provide inv_R2 or Lp; ignored otherwise.
    rho_CD, delta_CD : float
        Current-drive Gaussian deposition centre and width.
    q_init : dict or None
        Optional warm start from a previous solution (must contain 'rho'
        and 'q_arr' keys).  When None, the iteration starts from a
        parametric q with alpha_J = 1.5 and q(rho_95) = 3.0.
    n_rho : int
        Radial grid resolution.
    tol : float
        Relative L2 tolerance on || q_new - q_old || / || q_old ||.
    max_iter : int
        Maximum Picard iterations.
    damping : float
        Mixing parameter for the q update: q <- damping * q_new + (1-damping) * q.
    N_theta_inv_R2 : int
        Theta resolution for on-the-fly Lp and <1/R^2> integration when
        Vprime_data does not provide them.  Default 400.

    Returns
    -------
    dict with keys:
        q_arr     : ndarray   Self-consistent q(rho).
        rho       : ndarray   Radial grid.
        j_total   : ndarray   Composite j_Ohm + j_CD + j_bs [A/m^2].
        j_Ohm     : ndarray   Ohmic current density [A/m^2].
        j_CD      : ndarray   CD current density [A/m^2].
        j_non_bs  : ndarray   j_Ohm + j_CD [A/m^2].
        j_bs      : ndarray   Bootstrap current density [A/m^2].
        I_enc     : ndarray   Cumulative enclosed current [A].
        Lp_arr    : ndarray   Poloidal perimeter Lp(rho) [m].
        inv_R2_arr: ndarray   Flux-surface-perimeter <1/R^2>(rho) [m^-2].
        li        : float     Internal inductance l_i(3) [-].
        q0        : float     Central safety factor (may be < 1).
        alpha_J   : float     NaN (no parametric exponent in this mode).
        I_bs      : float     Integrated bootstrap current [MA].
        f_bs      : float     Bootstrap fraction I_bs / Ip.
        n_iter    : int       Number of Picard iterations performed.
        converged : bool      Convergence flag.
        kappa_arr : ndarray   Local elongation profile.
        q_at_95   : float     Diagnostic: q(rho = rho_95) of the converged
                              profile, for comparison with the f_q95 scaling.

    References
    ----------
    Wesson, Tokamaks, 4th ed. (2011) ch. 3 — q definition and cylindrical limit.
    Freidberg, Ideal MHD (Cambridge, 2014) ch. 6 — axisymmetric q integral.
    Miller et al., Phys. Plasmas 5 (1998) 973 — flux-surface parameterisation.
    Artaud et al., Nucl. Fusion 58 (2018) 105001 — METIS current diffusion
        philosophy: prescribed non-inductive sources, Ohmic share closes
        the balance, q(rho) reconstructed from j_total.
    Sauter et al., Phys. Plasmas 6 (1999) 2834 — neoclassical bootstrap.
    Redl et al., Phys. Plasmas 28 (2021) 022502 — improved Sauter coefs.
    Luce et al., Nucl. Fusion 54 (2014) 093005 — l_i(3) integral definition.
    """
    # == Radial grid =========================================================
    rho_inner = np.linspace(0.001, 0.05, 30, endpoint=False)
    rho_core  = np.linspace(0.05, rho_ped, n_rho, endpoint=False)
    rho_edge  = np.linspace(rho_ped, 0.99, n_rho)
    rho = np.concatenate([rho_inner, rho_core, rho_edge])

    if kappa_95 is None:
        kappa_95 = f_Kappa_95(kappa)
    if delta_95 is None:
        delta_95 = f_Delta_95(delta)

    use_miller = (Vprime_data is not None)
    if use_miller:
        kappa_arr = kappa_profile(rho, kappa, kappa_95, rho_95)
    else:
        kappa_arr = np.full_like(rho, kappa)

    # == Geometric weights: dA_per_drho, Lp, <1/R^2> =========================
    # Three quantities all derived from the same Miller surfaces R(rho,theta),
    # Z(rho,theta).  Self-consistency requires that they all come from the
    # same source (precomputed Vprime_data, or on-the-fly).  Mixing
    # precomputed-Lp with ellipse-dA introduces ~10 % drift in the final q
    # at the edge.
    drho = np.zeros_like(rho)
    drho[0]    = (rho[0] + rho[1]) / 2.0
    drho[-1]   = rho[-1] - rho[-2]
    drho[1:-1] = (rho[2:] - rho[:-2]) / 2.0

    have_dA     = (Vprime_data is not None and len(Vprime_data) >= 5
                   and Vprime_data[3] is not None)
    have_Lp     = (Vprime_data is not None and len(Vprime_data) >= 5
                   and Vprime_data[4] is not None)
    have_invR2  = (Vprime_data is not None and len(Vprime_data) >= 6
                   and Vprime_data[5] is not None)

    if have_dA and have_Lp and have_invR2:
        # Preferred path: full 6-tuple Vprime_data, all weights consistent.
        dA_per_drho = interpolate_dA(rho, Vprime_data[0], Vprime_data[3])
        Lp_arr      = np.interp(rho, Vprime_data[0], Vprime_data[4])
        inv_R2_arr  = np.interp(rho, Vprime_data[0], Vprime_data[5])
        # Pass Vprime_data through to sub-functions as is.
        Vprime_data_internal = Vprime_data
    else:
        # On-the-fly Miller integration.  np.gradient on the Picard rho
        # grid produces artifacts at the rho_inner/rho_core/rho_edge
        # junctions where the spacing jumps, so we integrate on an
        # internal uniform grid first and interpolate to the Picard
        # grid afterwards.  Equivalent to a self-contained
        # precompute_Vprime call but kept inside the function for the
        # Vprime_data=None case.  The full 6-tuple is also assembled and
        # passed downstream to the bootstrap and Ohmic sub-routines, so
        # that all geometry-dependent integrals see the same Miller
        # surfaces.
        rho_uni = np.linspace(1e-6, 1.0, max(n_rho * 2, 400))
        theta_u = np.linspace(0.0, 2.0*np.pi, N_theta_inv_R2, endpoint=False)
        dtheta_u = theta_u[1] - theta_u[0]
        drho_uni = rho_uni[1] - rho_uni[0]
        RHO_u, THETA_u = np.meshgrid(rho_uni, theta_u, indexing='ij')
        R_u, Z_u = miller_RZ(RHO_u, THETA_u, R0, a, kappa, delta,
                             kappa_95, delta_95, rho_95)
        dR_dt_u = np.gradient(R_u, dtheta_u, axis=1)
        dZ_dt_u = np.gradient(Z_u, dtheta_u, axis=1)
        dR_drho_u = np.gradient(R_u, drho_uni, axis=0)
        dZ_drho_u = np.gradient(Z_u, drho_uni, axis=0)
        jac_u = np.abs(dR_drho_u * dZ_dt_u - dR_dt_u * dZ_drho_u)
        # V'(rho) = 2 pi oint R |J2D| dtheta
        Vprime_uni = np.sum(2.0 * np.pi * R_u * jac_u, axis=1) * dtheta_u
        V_total_uni = float(np.trapezoid(Vprime_uni, rho_uni))
        # dA_pol/drho = oint |J2D| dtheta
        dA_uni = np.sum(jac_u, axis=1) * dtheta_u
        # Lp = oint sqrt((dR/dtheta)^2 + (dZ/dtheta)^2) dtheta
        arc_u  = np.sqrt(dR_dt_u**2 + dZ_dt_u**2)
        Lp_uni = np.sum(arc_u, axis=1) * dtheta_u
        Lp_safe_u = np.where(Lp_uni > 1e-12, Lp_uni, 1.0)
        invR2_uni = np.sum(arc_u / R_u**2, axis=1) * dtheta_u / Lp_safe_u
        invR2_uni[Lp_uni <= 1e-12] = 1.0 / R0**2
        # Interpolate to the Picard grid for local use.
        dA_per_drho = np.interp(rho, rho_uni, dA_uni)
        Lp_arr      = np.interp(rho, rho_uni, Lp_uni)
        inv_R2_arr  = np.interp(rho, rho_uni, invR2_uni)
        # Assemble a full 6-tuple Vprime_data for downstream consumers
        # (f_Sauter_Redl_Ib, f_Tprof, f_nprof) so that they see the same
        # Miller geometry as the q solver itself.
        Vprime_data_internal = (rho_uni, Vprime_uni, V_total_uni,
                                dA_uni, Lp_uni, invR2_uni)

    dA = dA_per_drho * drho

    # == Volume metric for li(3) and Lp for the geometric weights ============
    # All downstream consumers (l_i integral, T/n profiles, bootstrap) use
    # Vprime_data_internal so that geometry is consistent whether or not
    # the caller supplied a precomputed Vprime_data.
    Lp_li = np.interp(rho, Vprime_data_internal[0], Vprime_data_internal[4])
    Lp_li = np.where(rho > 1e-8, Lp_li, 1.0)

    Vprime_arr = interpolate_Vprime(rho,
                                    Vprime_data_internal[0],
                                    Vprime_data_internal[1])

    # == Profiles of T(rho), n(rho), epsilon(rho) ============================
    T_arr   = f_Tprof(Tbar, nu_T, rho, rho_ped, T_ped_frac, Vprime_data_internal)
    n_arr   = f_nprof(nbar, nu_n, rho, rho_ped, n_ped_frac, Vprime_data_internal)
    eps_arr = rho * a / R0

    # == Current-drive deposition profile (Gaussian, geometric only) =========
    j_CD_shape = np.exp(-0.5 * ((rho - rho_CD) / max(delta_CD, 0.01))**2)
    I_CD_A     = max(I_CD * 1e6, 0.0)
    jCD_dA     = float(np.sum(j_CD_shape * dA))
    if jCD_dA > 0.0 and I_CD_A > 0.0:
        j_CD_arr = I_CD_A * j_CD_shape / jCD_dA
    else:
        j_CD_arr = np.zeros_like(rho)

    Ip_A = Ip * 1e6
    F_vacuum = R0 * B0   # F = R B_phi ~ R0 B0 in low-beta vacuum approximation

    # == Initial q(rho) ======================================================
    if q_init is not None and 'rho' in q_init and 'q_arr' in q_init:
        # Warm start from a previous solution (e.g. solver cache).
        q_arr = np.interp(rho, q_init['rho'], q_init['q_arr'],
                          left=q_init['q_arr'][0], right=q_init['q_arr'][-1])
    else:
        # Generic parametric guess (alpha_J = 1.5, q(rho_95) = 3.0).
        q_arr = f_q_profile(rho, q95=3.0, rho95=rho_95, alpha_J=1.5)

    # == Helper: q(rho) from cumulative I_enc(rho) ==========================
    # Rigorous local q from Miller geometry:
    #     q(rho) = F * Lp(rho)^2 * <1/R^2>(rho) / (2 pi mu0 I_enc(rho))
    # with F = R0 B0, Lp the poloidal perimeter, <1/R^2> the perimeter
    # average of 1/R^2.  Captures kappa, delta, and finite aspect ratio
    # without any empirical shape factor.
    rho_ext = np.concatenate([[0.0], rho])

    def _cum_trap_from_zero(f_ext):
        """Trapezoidal cumulative integral on rho_ext, evaluated at rho."""
        steps = 0.5 * (f_ext[:-1] + f_ext[1:]) * np.diff(rho_ext)
        return np.cumsum(steps)

    def _q_from_Ienc(j_total_arr, j_axis_for_limit):
        # Cumulative enclosed current via trapezoidal rule.  The integrand
        # 2 pi rho a^2 kappa(rho) j_total vanishes at rho = 0, so the
        # prepended virtual point at rho_ext[0] = 0 is exactly zero.
        integrand     = j_total_arr * dA_per_drho
        integrand_ext = np.concatenate([[0.0], integrand])
        I_enc_arr     = _cum_trap_from_zero(integrand_ext)
        I_safe        = np.maximum(I_enc_arr, 1.0)   # zero/negative guard

        # Rigorous Miller-based q formula
        q_calc = (F_vacuum * Lp_arr**2 * inv_R2_arr
                  / (2.0 * np.pi * μ0 * I_safe))

        # On-axis analytic limit.  As rho -> 0:
        #   I_enc       -> j(0) * pi (rho a)^2 * kappa(0)
        #   Lp          -> 2 pi (rho a) sqrt((1 + kappa(0)^2) / 2)
        #   <1/R^2>     -> 1/R0^2
        # so q(0) -> (1 + kappa(0)^2) / kappa(0) * B0 / (mu0 R0 j(0)).
        # For kappa(0) = 1 this reduces to 2 B0 / (mu0 R0 j(0)).
        if rho[0] < 0.005 and j_axis_for_limit > 0.0:
            k0 = float(kappa_arr[0])
            q_calc[0] = ((1.0 + k0**2) / k0) * B0 / (μ0 * R0 * j_axis_for_limit)
        return q_calc, I_enc_arr

    # == Picard iteration on q(rho) =========================================
    converged = False
    n_iter = max_iter
    for iter_idx in range(1, max_iter + 1):
        q_old = q_arr.copy()

        # Wrap current q(rho) in the dict form expected by f_Sauter_Redl_Ib.
        q_dict = {'rho': rho, 'q_arr': q_old}

        # Bootstrap on current q(rho) (collisionality and L_31, L_32, L_34
        # coefficients see the local q via nu*_e ~ q).
        q95_cur = float(np.interp(rho_95, rho, q_old))
        bs_res = f_Sauter_Redl_Ib(
            R0, a, kappa, B0, nbar, Tbar, q95_cur, Z_eff, nu_n, nu_T,
            n_rho=len(rho), rho_ped=rho_ped,
            n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
            Vprime_data=Vprime_data_internal, kappa_95=kappa_95, rho_95=rho_95,
            return_profile=True, q_profile=q_dict,
            trapped_fraction_model=trapped_fraction_model, tau_i_e=tau_i_e)
        j_bs   = np.interp(rho, bs_res['rho'], bs_res['j_bs'])
        I_bs_A = float(np.clip(bs_res['I_bs'] * 1e6, 0.0, Ip_A))

        # Ohmic share closes the current balance.
        I_Ohm_A = max(Ip_A - I_CD_A - I_bs_A, 0.0)

        # Local conductivity sigma_neo(T, q, eps, Z_eff)
        if eta_model == 'redl':
            eta_loc = eta_redl(T_arr, n_arr * 1e20, Z_eff,
                               eps_arr, q=q_old, R0=R0)
        elif eta_model == 'sauter':
            eta_loc = eta_sauter(T_arr, n_arr * 1e20, Z_eff,
                                 eps_arr, q=q_old, R0=R0)
        else:  # 'spitzer' or any unknown -> classical Spitzer-Harm
            eta_loc = eta_spitzer(T_arr, n_arr * 1e20, Z_eff)
        sigma_loc   = 1.0 / np.maximum(eta_loc, 1e-12)
        sigma_loc   = np.where(np.isfinite(sigma_loc) & (sigma_loc > 0.0),
                               sigma_loc, 0.0)
        sig_dA_int  = float(np.sum(sigma_loc * dA))
        if sig_dA_int > 0.0 and I_Ohm_A > 0.0:
            j_Ohm = I_Ohm_A * sigma_loc / sig_dA_int
        else:
            j_Ohm = np.zeros_like(rho)

        # Composite current density and rebuild q(rho) from the rigorous
        # Miller-based formula.
        j_total = j_Ohm + j_CD_arr + j_bs
        q_new, I_enc = _q_from_Ienc(j_total,
                                    j_axis_for_limit=float(j_total[0]))

        # Damped update.
        q_damped = damping * q_new + (1.0 - damping) * q_old

        # L2 relative convergence test on q.
        rel_change = (float(np.sqrt(np.mean((q_damped - q_old)**2)))
                      / max(float(np.sqrt(np.mean(q_old**2))), 1e-6))
        q_arr = q_damped

        if rel_change < tol:
            converged = True
            n_iter = iter_idx
            break

    # == Final consistent pass on the converged q(rho) ======================
    # Re-evaluate the components without damping to return self-consistent
    # j_Ohm, j_bs, j_CD on the converged q.
    q_dict_final = {'rho': rho, 'q_arr': q_arr}
    q95_cur = float(np.interp(rho_95, rho, q_arr))
    bs_res = f_Sauter_Redl_Ib(
        R0, a, kappa, B0, nbar, Tbar, q95_cur, Z_eff, nu_n, nu_T,
        n_rho=len(rho), rho_ped=rho_ped,
        n_ped_frac=n_ped_frac, T_ped_frac=T_ped_frac,
        Vprime_data=Vprime_data_internal, kappa_95=kappa_95, rho_95=rho_95,
        return_profile=True, q_profile=q_dict_final,
        trapped_fraction_model=trapped_fraction_model, tau_i_e=tau_i_e)
    j_bs   = np.interp(rho, bs_res['rho'], bs_res['j_bs'])
    I_bs_A = float(np.clip(bs_res['I_bs'] * 1e6, 0.0, Ip_A))
    I_Ohm_A = max(Ip_A - I_CD_A - I_bs_A, 0.0)

    if eta_model == 'redl':
        eta_loc = eta_redl(T_arr, n_arr * 1e20, Z_eff,
                           eps_arr, q=q_arr, R0=R0)
    elif eta_model == 'sauter':
        eta_loc = eta_sauter(T_arr, n_arr * 1e20, Z_eff,
                             eps_arr, q=q_arr, R0=R0)
    else:
        eta_loc = eta_spitzer(T_arr, n_arr * 1e20, Z_eff)
    sigma_loc  = 1.0 / np.maximum(eta_loc, 1e-12)
    sigma_loc  = np.where(np.isfinite(sigma_loc) & (sigma_loc > 0.0),
                          sigma_loc, 0.0)
    sig_dA_int = float(np.sum(sigma_loc * dA))
    if sig_dA_int > 0.0 and I_Ohm_A > 0.0:
        j_Ohm = I_Ohm_A * sigma_loc / sig_dA_int
    else:
        j_Ohm = np.zeros_like(rho)

    j_total = j_Ohm + j_CD_arr + j_bs
    _q_unused, I_enc = _q_from_Ienc(j_total,
                                    j_axis_for_limit=float(j_total[0]))

    # == Internal inductance l_i(3) (Luce convention) =======================
    I_norm       = I_enc / np.maximum(I_enc[-1], 1.0)
    li_integrand = np.where(rho > 1e-8,
                            I_norm**2 * Vprime_arr / Lp_li**2,
                            0.0)
    li = (2.0 / R0) * np.trapezoid(li_integrand, rho)

    # == Output ==============================================================
    q_at_95 = float(np.interp(rho_95, rho, q_arr))

    return {
        'q_arr':      q_arr,
        'rho':        rho,
        'j_total':    j_total,
        'j_Ohm':      j_Ohm,
        'j_CD':       j_CD_arr,
        'j_non_bs':   j_Ohm + j_CD_arr,
        'j_bs':       j_bs,
        'I_enc':      I_enc,
        'Lp_arr':     Lp_arr,
        'inv_R2_arr': inv_R2_arr,
        'li':         li,
        'q0':         float(q_arr[0]),
        'alpha_J':    float('nan'),       # no parametric exponent here
        'I_bs':       I_bs_A / 1e6,
        'f_bs':       I_bs_A / max(Ip_A, 1.0),
        'n_iter':     n_iter,
        'converged':  converged,
        'kappa_arr':  kappa_arr,
        'q_at_95':    q_at_95,
    }



if __name__ == "__main__":
    # ── ITER chain (9/12) - bootstrap current ────────────────────────────
    # Sauter-Redl evaluated at the chain operating point with the deck
    # pedestal profiles. q95 is a forward reference (FROZEN), closed by
    # chain 12 after f_q95 is defined. The production solver additionally
    # refines the collisionality with the Picard q(rho) cache, worth
    # ~0.2 % on I_bs, hence the 1 % tolerance. The Segal (2021) model is
    # reported for model comparison at the same point.
    _Ib = f_Sauter_Redl_Ib(ITER['R0'], ITER['a'], ITER['kappa'], ITER['B0'],
                           ITER['nbar'], ITER['Tbar'], FROZEN['q95'],
                           ITER['Zeff'], ITER['nu_n'], ITER['nu_T'],
                           rho_ped=ITER['rho_ped'],
                           n_ped_frac=ITER['n_ped_frac'],
                           T_ped_frac=ITER['T_ped_frac'],
                           Vprime_data=ITER_Vpd, kappa_95=ITER['kappa95'],
                           tau_i_e=1.0)
    _Ib_seg = f_Segal_Ib(ITER['nu_n'], ITER['nu_T'], ITER['a'] / ITER['R0'],
                         ITER['kappa'], ITER['nbar'], ITER['Tbar'],
                         ITER['R0'], FROZEN['Ip'],
                         rho_ped=ITER['rho_ped'],
                         n_ped_frac=ITER['n_ped_frac'],
                         T_ped_frac=ITER['T_ped_frac'])
    ITER.update(Ib=_Ib)
    _bench("ITER chain 9/12 - bootstrap current (Sauter-Redl)", [
        ("I_bs Sauter-Redl [MA]", _Ib, FROZEN['Ib'], 0.01, "deck frozen"),
        ("bootstrap fraction I_bs/Ip [-]", _Ib / FROZEN['Ip'], None, None,
         "chain"),
        ("I_bs Segal (2021) [MA]", _Ib_seg, None, None, "model comparison"),
    ])

#%% Other parameters
# ─────────────────────────────────────────────────────────────────────────────
# Engineering figures of merit: heat load proxies, plasma current, neutron
# wall loading, MHD safety factors, scaling-law registry, global energy
# descriptors, helium ash accumulation model, and cost proxy.
#
# Profile convention (applies to f_He_fraction, f_tau_alpha):
#   Academic  : rho_ped=1.0, T_ped_frac=0.0  (pure power-law, no pedestal)
#   refined     : 0 < rho_ped < 1, T_ped_frac > 0  (H-mode pedestal profile)
#
# Geometry convention (applies to f_Gamma_n, f_tauE):
#   Academic  : S_wall=None → elliptical-torus approximation
#               V must be passed as 2π²R₀a²κ by the caller
#   refined     : S_wall / V supplied by the Miller-geometry module
# ─────────────────────────────────────────────────────────────────────────────

def f_heat_refined(R0, P_sep):
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
        Poloidal heat load parameter P_sep·B₀/(q₉₅·A·R₀) [MW T m⁻¹]

    Notes
    -----
    This is the EU-DEMO divertor-protection figure of merit (PROCESS
    constraint 'psepbqarmax'); values ≲ 9.2 MW T m⁻¹ are considered
    compatible with an ITER-class divertor.

    References
    ----------
    M. Siccinio et al., Nuclear Fusion 59 (2019) 106026.
    """
    A = R0 / a
    return (P_sep * B0) / (q95 * A * R0)


def f_heat_PFU_Eich(P_sol, B_pol, R, eps, theta_deg,
                    B0=None, f_outer_target=0.65):
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
        Total field-line grazing angle on the divertor target [°]
        Typical range: 1-5° for vertical-target configurations.
    B0 : float or None
        On-axis toroidal field [T], used for the field-line pitch B/B_θ at
        the outer midplane.  If None, the pitch factor is set to 1 (legacy
        poloidal-projection behaviour) and a UserWarning is issued, since
        q_∥ is then NOT a parallel flux and is inconsistent with theta_deg.
    f_outer_target : float
        Fraction of P_sol carried to the analysed (outer) target [-].
        Default 0.65, typical of lower single-null in/out asymmetry.
        Set to 1.0 for a single-target worst-case estimate.

    Returns
    -------
    lambda_q : float
        SOL power e-folding decay length [m]
    q_par_u : float
        Peak parallel heat flux upstream (outer midplane separatrix) [MW m⁻²]
    q_target : float
        Peak perpendicular heat flux on the divertor target [MW m⁻²]

    Notes
    -----
    Eich (2013) regression #15 (all devices incl. spherical tokamaks):
        λ_q [mm] = 1.35 · P_sol^{-0.02} · R^{0.04} · B_pol^{-0.92} · ε^{0.42}
    (The conventional-tokamak alternative in (B_tor, q95) form is
        λ_q ∝ B_tor^{-0.77} · q95^{1.05} · P_sol^{0.09};   not used here.)

    Upstream peak parallel flux.  The SOL power flows along B through the
    annular cross-section 2π R λ_q (B_θ/B)|omp, hence
        q_∥u = f_out · P_sol / (2π R λ_q) · (B/B_θ)|omp
    with B_tor,omp = B0/(1+ε) and B = sqrt(B_tor,omp² + B_pol²).
    Omitting the pitch factor B/B_θ (≈ 3-6 for conventional tokamaks)
    underestimates q_∥u and is inconsistent with the use of the total
    grazing angle θ below.

    Projection onto the target surface (grazing incidence, attached):
        q_⊥ = q_∥u · sin θ

    References
    ----------
    T. Eich et al., Nuclear Fusion 53 (2013) 093031.
    P.C. Stangeby, Plasma Phys. Control. Fusion 60 (2018) 044022.
    """
    θ = np.deg2rad(theta_deg)

    # SOL power decay length: empirical multi-machine regression [mm → m]
    lambda_q = 1.35 * R**0.04 * B_pol**(-0.92) * eps**0.42 * P_sol**(-0.02) * 1e-3

    # Outer-midplane poloidal-projection heat flux [MW m⁻²]
    q_pol_omp = f_outer_target * P_sol / (2 * np.pi * R * lambda_q)

    # Field-line pitch factor (B/B_θ)|omp
    if B0 is None:
        warnings.warn(
            "f_heat_PFU_Eich: B0 not provided — pitch factor B/B_theta set "
            "to 1 (legacy poloidal projection). q_par_u is then NOT a "
            "parallel heat flux; pass B0 for the physical estimate.",
            UserWarning, stacklevel=2
        )
        pitch = 1.0
    else:
        B_tor_omp = B0 / (1.0 + eps)
        pitch = np.sqrt(B_tor_omp**2 + B_pol**2) / B_pol

    # Peak parallel heat flux at the upstream separatrix [MW m⁻²]
    q_par_u = q_pol_omp * pitch

    # Perpendicular target heat flux accounting for grazing incidence [MW m⁻²]
    q_target = q_par_u * np.sin(θ)

    return lambda_q, q_par_u, q_target


if __name__ == "__main__":
    # ── SOL width: published anchor and ITER chain (10/12) ──────────────
    # Published anchor: Eich et al., NF 53 (2013) 093031, Table 6,
    # regression #15: lambda_q [mm] = 1.35 P_SOL^-0.02 R^0.04 Bpol^-0.92
    # (a/R)^0.42. Closed worked example: the paper's own ITER evaluation
    # gives lambda_q = 0.73 mm (P_SOL = 100 MW, Bpol,MP = 1.185 T,
    # R = 6.2 m, a = 2 m); tolerance 8 % (rounding of the quoted
    # midplane poloidal field).
    # Chain: B_pol from the q-inversion at the frozen q95 (closed by
    # chain 12) and lambda_q at the chain P_sep.
    _lam_pub, _, _ = f_heat_PFU_Eich(100., 1.185, 6.2, 2 / 6.2, 3.0, B0=5.3)
    _Bpol = f_Bpol(FROZEN['q95'], ITER['B0'], ITER['a'], ITER['R0'],
                   kappa=ITER['kappa'])
    _lam, _, _ = f_heat_PFU_Eich(ITER['P_sep'], _Bpol, ITER['R0'],
                                 ITER['a'] / ITER['R0'], FROZEN['q95'],
                                 B0=ITER['B0'])
    _bench("ITER chain 10/12 - SOL heat-flux width (Eich #15)", [
        ("lambda_q paper inputs [mm]", _lam_pub * 1e3, 0.73, 0.08,
         "Eich 2013"),
        ("B_pol outer midplane [T]", _Bpol, FROZEN['B_pol'], 2e-3,
         "deck frozen"),
        ("lambda_q chain point [mm]", _lam * 1e3, FROZEN['lambda_q_mm'],
         2e-3, "deck frozen"),
    ])

# =============================================================================
# Refined divertor exhaust — two-point model (Stangeby 2018)
# =============================================================================
#
# Fidelity ladder of the D0FUS divertor models:
#   figure_of_merit : H, H_par, H_pol  (f_heat_refined / f_heat_par / f_heat_pol)
#                     Nested 0D scalars to RANK designs vs ITER, NOT predictions.
#   Academic        : f_heat_PFU_Eich — peak ATTACHED parallel flux (upper bound).
#   refined         : f_heat_two_point — two-point model with volumetric losses,
#                     giving the target temperature, the detachment state and the
#                     SOL power-loss fraction required for target survival.
#
# The seeding-impurity concentration closure (Lengyel / extended-Lengyel,
# Kallenbach 2016; Body, Kallenbach & Eich 2025, arXiv:2504.05486) is out of
# scope for a lean 0D code (needs OpenADAS atomic data); it is the natural
# future upgrade of f_heat_two_point.

# Average DT fuel-ion mass, m_f = 2.5 u, Stangeby (2018) convention.
M_F_DT = 2.5 * 1.67e-27   # [kg]


def f_heat_dissipation_required(q_par_u, theta_deg,
                                q_dep_limit=5.0, flux_expansion=1.0):
    """
    Minimum SOL volumetric power-loss fraction needed for target survival.

    From a flux-tube power balance (Stangeby 2018, Eq. 14), the fraction of the
    upstream parallel power that must be dissipated before the target to keep the
    deposited plasma heat flux below an engineering limit is:

        1 − f_pwr_loss = (q_dep_limit / sinθ) (R_t/R_u) / q_∥u

    This is the single most informative 0D exhaust output: it needs neither the
    upstream density nor any 2PM closure, only q_∥u (from Eich) and the target
    geometry.

    Parameters
    ----------
    q_par_u : float    Upstream parallel heat flux density [MW/m²].
    theta_deg : float  Field-line incidence angle on the target [deg].
    q_dep_limit : float  Tolerable plasma-only deposited target flux [MW/m²].
        Default 5.0 (Stangeby), leaving headroom under the ~10 MW/m² limit.
    flux_expansion : float  Target/upstream flux expansion R_t/R_u [-].
        Default 1.0; >1 for Super-X type long legs, which relax the requirement.

    Returns
    -------
    f_pwr_loss : float   Minimum required SOL power-loss fraction, clamped to
        [0, 1). Zero means the target survives with no dissipation.

    References
    ----------
    P.C. Stangeby, Plasma Phys. Control. Fusion 60 (2018) 044022, Eq. (14).
    """
    s = np.sin(np.deg2rad(theta_deg))
    one_minus = (q_dep_limit / s) * flux_expansion / q_par_u
    return float(np.clip(1.0 - one_minus, 0.0, 1.0 - 1e-12))


def _two_point_core(q_par_u_SI, p_u, f_cooling, f_mom, gamma_sheath, m_f):
    """
    Bare 2PM target quantities (Stangeby 2018, Eqs. 15a–17a), UNCAPPED.

    T_et   = (8 m_f / e γ²) (q_∥u/p_u)²  [(1−f_cooling)/(1−f_mom)]²        [eV]
    n_et   = (γ² / 32 m_f) (p_u³/q_∥u²) (1−f_mom)³/(1−f_cooling)²          [m⁻³]
    Γ_et   = (γ / 8 m_f)  (p_u²/q_∥u)  (1−f_mom)²/(1−f_cooling)            [m⁻²s⁻¹]

    Self-consistent by construction with the Bohm condition Γ_et = n_et c_st,
    the pressure balance p_t = (1−f_mom) p_u and the power balance
    q_∥t = (1−f_cooling) q_∥u. Kept separate so the conduction-limited cap in
    f_heat_two_point does not hide this consistency.

    Returns (T_et [eV], n_et [m⁻³], Γ_et [m⁻²s⁻¹]).
    """
    ratio = (1.0 - f_cooling) / (1.0 - f_mom)
    T_et = (8.0 * m_f / (E_ELEM * gamma_sheath**2)) * (q_par_u_SI / p_u)**2 * ratio**2
    n_et = (gamma_sheath**2 / (32.0 * m_f)) * (p_u**3 / q_par_u_SI**2) \
           * (1.0 - f_mom)**3 / (1.0 - f_cooling)**2
    Gamma_et = (gamma_sheath / (8.0 * m_f)) * (p_u**2 / q_par_u_SI) \
               * (1.0 - f_mom)**2 / (1.0 - f_cooling)
    return T_et, n_et, Gamma_et


def f_heat_two_point(P_sol, B_pol, R0, eps, q95, n_sep, theta_deg,
                     B0=None, f_outer_target=0.65,
                     f_cooling=0.0, f_mom=0.0,
                     gamma_sheath=7.0, kappa0e=2000.0, eps_pot=15.0,
                     q_dep_limit=5.0, flux_expansion=1.0,
                     m_f=M_F_DT):
    """
    Refined 0D divertor exhaust estimate via the two-point model.

    Chain (Stangeby 2018):
      1. λ_q from Eich (2013) regression #15 → upstream parallel flux
         q_∥u = f_out · P_sol / (2π R₀ λ_q) · (B/B_θ)|omp,
         with B_tor,omp = B0/(1+ε) and B = sqrt(B_tor,omp² + B_pol²).
      2. Upstream electron temperature from Spitzer–Härm conduction (Eq. 38),
         T_eu = (7 q_∥u L / 2 κ₀ₑ)^(2/7), connection length L = π R₀ q₉₅.
      3. Upstream total pressure p_u = 2 n_sep e T_eu  (T_i=T_e, M_u=0, Z=1; Eq. 20).
      4. Target quantities from the 2PM with volumetric losses (Eqs. 15a–17a).
      5. Deposited target heat flux (Eqs. 1–3):
         q_dep_t = (γ_sheath + ε_pot/T_et) e T_et Γ_et sinθ.
      6. Required dissipation fraction (Eq. 14), independent of the closure.

    The 2PM is CONDUCTION-LIMITED: valid only for T_et ≪ T_eu. When the inputs
    (typically a reactor with q_∥u ~ GW/m² and no dissipation, f_cooling=f_mom=0)
    drive T_et above T_eu, the regime is sheath-limited and the attached target
    is unviable; this is flagged via 'regime' and T_et is capped at T_eu. The
    physical reading is that strong dissipation is mandatory, which is exactly
    what f_pwr_loss_req quantifies.

    Parameters
    ----------
    P_sol : float     Power crossing the separatrix (= divertor power) [MW].
    B_pol : float     Outer-midplane poloidal field [T].
    R0 : float        Major radius [m].
    eps : float       Inverse aspect ratio a/R₀ [-].
    q95 : float       Safety factor at ψ_N = 0.95 [-].
    n_sep : float     Upstream (separatrix) electron density [m⁻³].
    theta_deg : float Total field-line incidence angle on the target [deg].
    B0 : float or None  On-axis toroidal field [T] for the pitch (B/B_θ)|omp.
        If None, pitch = 1 (legacy poloidal projection) with a UserWarning;
        the whole 2PM chain is then biased optimistic by (B/B_θ)^(2/7) on T_eu
        and ~(B/B_θ)² on T_et.
    f_outer_target : float  Fraction of P_sol routed to the analysed (outer)
        target [-]. Default 0.65 (typical LSN asymmetry); 1.0 = worst case.
    f_cooling : float Volumetric power-loss fraction along the flux tube (Eq. 18),
        input not predicted. 0 = attached.
    f_mom : float     Volumetric momentum-loss fraction (Eq. 19). 0 = attached.
    gamma_sheath : float  Sheath heat transmission coefficient [-], default 7.0.
    kappa0e : float   Electron Spitzer–Härm coefficient [W m⁻¹ eV⁻⁷ᐟ²], default
        2000. Stangeby uses ~3000; T_eu ∝ κ₀ₑ^(−2/7) makes this nearly immaterial.
    eps_pot : float   Hydrogenic potential energy deposited per ion [eV], default 15.
    q_dep_limit : float  Tolerable deposited target flux [MW/m²], default 5.0.
    flux_expansion : float  Target/upstream flux expansion R_t/R_u [-].
    m_f : float       Fuel-ion mass [kg], default DT (2.5 u).

    Returns
    -------
    dict with keys:
        'lambda_q'        SOL power width [m].
        'q_pol_omp'       Poloidal-projection flux f_out·P_sol/(2πR₀λ_q) [MW/m²].
        'pitch_B_over_Bpol'  Field-line pitch (B/B_θ)|omp applied to q_∥u [-].
        'f_outer_target'  Outer-target power-sharing fraction used [-].
        'q_par_u'         Upstream parallel heat flux [MW/m²].
        'T_eu'            Upstream electron temperature [eV].
        'T_et'            Target electron temperature [eV] (capped at T_eu).
        'n_et'            Target electron density [m⁻³].
        'Gamma_et'        Target parallel particle flux [m⁻²s⁻¹].
        'q_dep_t'         Deposited perpendicular target flux [MW/m²].
        'f_pwr_loss_req'  Min. SOL power-loss fraction for survival [-].
        'regime'          'sheath-limited' | 'conduction-limited'.
        'detached'        bool, T_et < 10 eV.
        'sputtering_safe' bool, T_et < 5 eV (gross W erosion strongly suppressed).

    References
    ----------
    P.C. Stangeby, Plasma Phys. Control. Fusion 60 (2018) 044022.
    V. Kotov & D. Reiter, Plasma Phys. Control. Fusion 51 (2009) 115002.
    """
    s = np.sin(np.deg2rad(theta_deg))

    # 1. SOL width and upstream parallel flux.
    #    q_∥u = f_out · P_sol/(2π R λ_q) · (B/B_θ)|omp — the pitch factor is
    #    mandatory: the Spitzer-Härm relation in step 2 requires the TRUE
    #    parallel flux (Stangeby 2018). B_tor,omp = B0/(1+ε).
    lambda_q = 1.35 * R0**0.04 * B_pol**(-0.92) * eps**0.42 * P_sol**(-0.02) * 1e-3
    q_pol_omp = f_outer_target * P_sol / (2.0 * np.pi * R0 * lambda_q)  # [MW/m²]
    if B0 is None:
        warnings.warn(
            "f_heat_two_point: B0 not provided — pitch factor B/B_theta set "
            "to 1 (legacy poloidal projection). T_eu and all downstream 2PM "
            "quantities are then underestimated; pass B0 for the physical "
            "estimate.",
            UserWarning, stacklevel=2
        )
        pitch = 1.0
    else:
        B_tor_omp = B0 / (1.0 + eps)
        pitch = np.sqrt(B_tor_omp**2 + B_pol**2) / B_pol
    q_par_u = q_pol_omp * pitch                                  # [MW/m²]
    q_par_u_SI = q_par_u * 1e6                                   # [W/m²]

    # 2. Upstream temperature, Spitzer conduction (Eq. 38), L = π R₀ q₉₅
    L_conn = np.pi * R0 * q95
    T_eu = (7.0 * q_par_u_SI * L_conn / (2.0 * kappa0e))**(2.0/7.0)

    # 3. Upstream total pressure (Eq. 20; T_i=T_e, M_u=0, Z=1)
    p_u = 2.0 * n_sep * E_ELEM * T_eu                           # [Pa]

    # 4. Target quantities, 2PM with losses (Eqs. 15a–17a)
    T_et, n_et, Gamma_et = _two_point_core(
        q_par_u_SI, p_u, f_cooling, f_mom, gamma_sheath, m_f)

    # Conduction-limited validity: cap T_et at T_eu, flag sheath-limited.
    if T_et >= T_eu:
        regime = 'sheath-limited'
        T_et = T_eu
    else:
        regime = 'conduction-limited'

    # 5. Deposited perpendicular target heat flux (Eqs. 1–3)
    gamma_target = gamma_sheath + eps_pot / T_et
    q_dep_t = gamma_target * E_ELEM * T_et * Gamma_et * s * 1e-6  # [MW/m²]

    # 6. Required dissipation fraction (Eq. 14)
    f_pwr_loss_req = f_heat_dissipation_required(
        q_par_u, theta_deg, q_dep_limit, flux_expansion)

    return {
        'lambda_q':        lambda_q,
        'q_pol_omp':       q_pol_omp,
        'pitch_B_over_Bpol': pitch,
        'f_outer_target':  f_outer_target,
        'q_par_u':         q_par_u,
        'T_eu':            T_eu,
        'T_et':            T_et,
        'n_et':            n_et,
        'Gamma_et':        Gamma_et,
        'q_dep_t':         q_dep_t,
        'f_pwr_loss_req':  f_pwr_loss_req,
        'regime':          regime,
        'detached':        bool(T_et < 10.0),
        'sputtering_safe': bool(T_et < 5.0),
    }


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
    geometry_model : 'Academic' | 'refined'
    S_cross : float or None  Externally provided cross-section area [m²].

    Returns
    -------
    kappa_x : float  Cross-section elongation [-].
    """
    if S_cross is not None:
        return S_cross / (np.pi * a**2)
    if geometry_model == 'Academic':
        return kappa_edge
    elif geometry_model == 'refined':
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

    **refined / Miller (S_wall provided)**
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
        # Academic: elliptical torus first-wall area (Ramanujan perimeter)
        S_wall = 2 * np.pi * R0 * _ramanujan_perimeter(a, κ * a)

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
# f_q_profile() defined earlier (before f_Sauter_Redl_Ib) due to forward dependency.


# ── Scaling law coefficient registry ─────────────────────────────────────────

def f_Get_parameter_scaling_law(Scaling_Law):
    """
    Return the pre-factor and exponents of a global τ_E scaling law.

    The general form (ITER Engineering Parameters convention) is:

        τ_E = C · R₀^α_R · ε^α_ε · κ^α_κ · (1+δ)^α_δ
                · (10 n)^α_n · B₀^α_B · M^α_M · P^α_P · I_p^α_I

    with n̄  in 10²⁰ m⁻³ (factor 10 converts to 10¹⁹ m⁻³), I_p in MA, P in MW.

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
        # Yushmanov et al. (1990), NF 30, 1999 — ITER89-P L-mode scaling.
        # Published form: 0.048 I^0.85 R^1.2 a^0.3 κ^0.5 n̄20^0.1 B^0.2 M^0.5 P^-0.5
        # Converted to the D0FUS (R, ε, n̄19) convention:
        #   R^1.2 a^0.3 = R^1.5 ε^0.3   and   n̄20^0.1 = 10^-0.1 · n̄19^0.1
        #   → C_SL = 0.048 × 10^-0.1 = 0.0381
        'ITER89-P':   dict(C_SL=0.0381, α_δ=0,    α_M=0.5,  α_κ=0.5,
                           α_ε=0.3,   α_R=1.5,   α_B=0.2,
                           α_n=0.1,   α_I=0.85,  α_P=-0.5),
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

if __name__ == "__main__":
    # ── Published anchors - confinement-scaling registry ─────────────────
    # The registry must store the published exponents EXACTLY:
    # IPB98(y,2): ITER Physics Basis, NF 39 (1999) 2175 / Doyle, NF 47
    # (2007) S18. ITPA20: Verdoolaege, NF 61 (2021) 076006. ITER89-P:
    # Yushmanov, NF 30 (1990) 1999, with the prefactor converted to the
    # D0FUS n19 convention: C(n19) = 0.048 x 10^-0.1 = 0.0381.
    # Exact tuple equality is asserted (structural identity).
    assert f_Get_parameter_scaling_law('IPB98(y,2)') == \
        (0.0562, 0, 0.19, 0.78, 0.58, 1.97, 0.15, 0.41, 0.93, -0.69)
    assert f_Get_parameter_scaling_law('ITPA20') == \
        (0.053, 0.36, 0.2, 0.8, 0.35, 1.71, 0.22, 0.24, 0.98, -0.669)
    # IPB98 at the ITER Q=10 point with the PUBLISHED loss-power
    # convention (P = 87 MW, radiation NOT subtracted; kappa_x = 1.70,
    # n19 = 10.1): tolerance 8 % covers the kappa-convention and P_loss
    # definition spread between sources. The production (PROCESS-like)
    # convention, which subtracts the core radiation, is exercised by
    # chain 11.
    _Csl, _ad, _aM, _ak, _ae, _aR, _aB, _an, _aI, _aP = \
        f_Get_parameter_scaling_law('IPB98(y,2)')
    _tau98 = (_Csl * 15**_aI * 5.3**_aB * 10.1**_an * 2.5**_aM * 6.2**_aR
              * (2 / 6.2)**_ae * 1.70**_ak * 87.0**_aP)
    _bench("Published anchors - confinement scaling laws", [
        ("IPB98(y,2) / ITPA20 exponents", "exact", "published", None,
         "registry assert"),
        ("ITER89-P prefactor C(n19) [-]",
         f_Get_parameter_scaling_law('ITER89-P')[0], 0.048 * 10**-0.1, 5e-3,
         "Yushmanov 1990"),
        ("tau_IPB98, published convention [s]", _tau98, 3.7, 0.08,
         "IPB 1999"),
    ])

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
    - **refined / Miller** : V numerically integrated from the Miller profile
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


if __name__ == "__main__":
    # ── ITER chain (11/12) - stored energy, tau_E and plasma current ────
    # Loss-power convention: D0FUS subtracts the CORE radiated power from
    # the heating power, P_loss = P_alpha + P_aux + P_Ohm - P_rad,core,
    # in BOTH tau_E and the scaling-law inversion for Ip (PROCESS-like
    # convention; see the note in f_tauE). With this convention and the
    # chain radiation budget, the module-level tau_E and Ip must land on
    # the full-device values. The published tau_E = 3.7 s instead uses
    # P_loss = 87 MW with no radiation subtracted: the same stored energy
    # then gives W_th/87 = 3.91 s, bracketing the published value.
    # The IPB98 inversion uses the area elongation kappa_a = V/(2 pi^2 R0
    # a^2) and the LINE-averaged density (fitting conventions of the law).
    _W = f_W_th(ITER['pbar'], ITER['V']) / 1e6                       # [MJ]
    _tau = f_tauE(ITER['pbar'], ITER['V'], ITER['P_alpha'], ITER['P_aux'],
                  ITER['P_Ohm'], ITER['P_rad_core'])
    _Ploss = (ITER['P_alpha'] + ITER['P_aux'] + ITER['P_Ohm']
              - ITER['P_rad_core'])
    _Csl, _ad, _aM, _ak, _ae, _aR, _aB, _an, _aI, _aP = \
        f_Get_parameter_scaling_law('IPB98(y,2)')
    _Ip = f_Ip(_tau, ITER['R0'], ITER['a'], ITER['kappa_a'], ITER['delta'],
               ITER['nbar_line'], ITER['B0'], ITER['M'],
               ITER['P_alpha'], ITER['P_Ohm'], ITER['P_aux'],
               ITER['P_rad_core'], ITER['H'], _Csl,
               _ad, _aM, _ak, _ae, _aR, _aB, _an, _aI, _aP)
    _nG = f_nG(_Ip, ITER['a'])
    ITER.update(tauE=_tau, Ip=_Ip, W_th=_W)
    _bench("ITER chain 11/12 - tau_E and plasma current (IPB98 inversion)", [
        ("W_th [MJ]", _W, FROZEN['W_th'], 1e-3, "deck frozen"),
        ("P_loss = P_heat - P_rad,core [MW]", _Ploss, None, None,
         "convention"),
        ("tau_E, core radiation subtracted [s]", _tau, FROZEN['tauE'], 1e-3,
         "deck frozen"),
        ("energy-balance closure W/(tau P_loss)",
         f_W_th(ITER['pbar'], ITER['V']) / 1e6 / (_tau * _Ploss), 1.0, 1e-9,
         "identity"),
        ("tau_E, published convention [s]", _W / 87.0, "3.7", None,
         "IPB 1999 (P=87 MW)"),
        ("Ip from IPB98 inversion [MA]", _Ip, FROZEN['Ip'], 5e-3,
         "deck frozen"),
        ("Ip [MA]", _Ip, 15.0, 0.01, "Shimada 2007"),
        ("n_GW at chain Ip [1e20 m-3]", _nG, FROZEN['nG'], 2e-3,
         "deck frozen"),
        ("f_GW = n_line/n_GW [-]", ITER['nbar_line'] / _nG, 0.85, 5e-3,
         "deck target"),
    ], notes=[
        "Neither Ip nor the density is imposed: the deck receives the "
        "geometry, B at the front face, P_fus, P_aux and f_GW = 0.85, and "
        "the chain closes on the published 15 MA / 1.01e20 m-3 point.",
    ])


# ── Helium ash accumulation model ─────────────────────────────────────────────

def _sigmav_vol(T_bar, nu_T, rho_ped=1.0, T_ped_frac=0.0, N=200,
                Vprime_data=None, tau_i_e=1.0):
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
        # refined mode: Miller weight V'(ρ)/V
        rho_grid, Vprime, V_total = Vprime_data[:3]  # safe: 5-tuple (rho, V', V, dA, Lp)
        T   = f_Tprof(T_bar, nu_T, rho_grid, rho_ped, T_ped_frac,
                       Vprime_data)
        sv  = f_sigmav(tau_i_e * T)   # reactivity on T_i = tau_i_e * T_e
        return float(np.trapezoid(sv * Vprime, rho_grid)) / V_total
    else:
        # Academic mode: cylindrical weight 2ρ dρ
        rho = np.linspace(0.0, 1.0, N)
        T   = f_Tprof(T_bar, nu_T, rho, rho_ped, T_ped_frac)
        sv  = f_sigmav(tau_i_e * T)   # reactivity on T_i = tau_i_e * T_e
        return float(np.trapezoid(sv * 2.0 * rho, rho))


def f_He_fraction(n_bar, T_bar, tauE, C_Alpha, nu_T,
                  rho_ped=1.0, T_ped_frac=0.0, Vprime_data=None, tau_i_e=1.0):
    """
    Estimate the equilibrium helium ash fraction f_α = n_α / n_e.

    Alpha particles (⁴He²⁺) produced by D–T fusion must be expelled to
    prevent fuel dilution and Q degradation.  This function solves the
    steady-state particle balance between alpha production (∝ n² ⟨σv⟩)
    and alpha removal (∝ n_α / τ_α).

    Two temperature profile models are supported via `f_Tprof`:

    **Academic** : rho_ped = 1.0, T_ped_frac = 0.0  (default)
        Pure power-law core profile T(ρ) = T̄ (1 − ρ²)^ν_T; no pedestal.

    **refined / H-mode** : 0 < rho_ped < 1, T_ped_frac > 0
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
    in Academic mode, Miller V'(ρ)/V in refined mode).

    References
    ----------
    Y. Sarazin et al., Nuclear Fusion (2021). Appendix B.
    """
    # Volume-averaged reactivity via shared helper (uses T_i = tau_i_e * T_e)
    sigmav_vol = _sigmav_vol(T_bar, nu_T, rho_ped, T_ped_frac,
                             Vprime_data=Vprime_data, tau_i_e=tau_i_e)
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

def f_volume(a, b, c, d, R0, κ, Delta_TF, H_TF):
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
    Delta_TF : float
        Outboard port-access radial clearance [m].
    H_TF : float
        Total TF coil height from Princeton-D cross-section [m].

    Returns
    -------
    V_blanket : float
        Volume of FW + BB + neutron shield + VV + gaps [m³].
    V_TF_Pappus : float
        Volume of TF coil winding packs (Pappus approximation) [m³].
    V_CS_geom : float
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
    - FI  : solid cylinder, height H_TF,
            outer radius R₀ + a + b + Delta_TF + c.
    """
    # Blanket shell (Pappus): A_cross = 4[(a+b)(κa+b) − a·κa] = 4b[a(1+κ)+b]
    V_blanket = 8 * np.pi * b * R0 * (a * (1 + κ) + b)

    # TF shell (Pappus): A_cross = 4c[a(1+κ) + 2b + c]
    V_TF_Pappus = 8 * np.pi * c * R0 * (a * (1 + κ) + 2 * b + c)

    # Central solenoid: π h (R_out² − R_in²), h = 2(κa + b + c)
    R_i = R0 - a - b - c
    V_CS_geom = 2 * np.pi * (κ * a + b + c) * (2 * d * R_i - d**2)

    # Fusion-island bounding cylinder: π R² H_TF, R = R0 + a + b + Delta_TF + c
    V_FI = 2 * np.pi * H_TF * (R0 + a + b + Delta_TF + c)**2

    return (V_blanket, V_TF_Pappus, V_CS_geom, V_FI)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT LIFETIME & PLANT AVAILABILITY
# ══════════════════════════════════════════════════════════════════════════════

def f_blanket_lifetime_fpy(P_fus: float, A_FW: float,
                           dpa_lim: float, C_dpa: float) -> float:
    """
    Blanket structural lifetime based on neutron displacement damage.

    t_bl = dpa_lim * A_FW / (0.8 * C_dpa * P_fus)

    Derivation: neutron wall loading q_n = 0.8*P_fus/A_FW [MW/m²];
    dpa rate = C_dpa * q_n [dpa/fpy]; lifetime = dpa_lim / dpa_rate.

    Ref: Gilbert et al. (2013), EUROfusion (2015).

    Parameters
    ----------
    P_fus   : float  Fusion power [MW].
    A_FW    : float  First-wall area [m²].
    dpa_lim : float  Allowable structural damage [dpa].
    C_dpa   : float  dpa conversion coefficient [dpa fpy⁻¹ / (MW m⁻²)].

    Returns
    -------
    float  Blanket lifetime [fpy].
    """
    return dpa_lim * A_FW / (0.8 * C_dpa * P_fus)


def f_divertor_lifetime_fpy(P_sep: float, A_div: float,
                            epsilon_div: float, f_peak: float) -> float:
    """
    Divertor lifetime based on integrated heat exposure.

    t_div = epsilon_div * A_div / (f_peak * P_sep)

    Derivation: peak heat flux q_div = f_peak * P_sep / A_div [MW/m²];
    integrated limit epsilon_div [MW yr/m²]; lifetime = epsilon_div / q_div.

    Ref: ITER Organization (2025), CEA IRFM (2017).

    Parameters
    ----------
    P_sep       : float  Power crossing the separatrix [MW].
    A_div       : float  Divertor wetted area [m²].
    epsilon_div : float  Integrated heat limit [MW yr / m²].
    f_peak      : float  Heat flux peaking factor [-].

    Returns
    -------
    float  Divertor lifetime [fpy].
    """
    if P_sep <= 0.0:
        return np.inf
    return epsilon_div * A_div / (f_peak * P_sep)


def f_lifetime_to_years(t_fpy: float, Util_factor: float,
                        Dwell_factor: float) -> float:
    """
    Convert a lifetime in full-power years (fpy) to calendar years.

    t_yr = t_fpy / (Util_factor * Dwell_factor)

    Parameters
    ----------
    t_fpy        : float  Lifetime [fpy].
    Util_factor  : float  Utilisation factor [-].
    Dwell_factor : float  Dwell factor (1.0 for steady-state) [-].

    Returns
    -------
    float  Lifetime [calendar years].
    """
    return t_fpy / (Util_factor * Dwell_factor)


def f_availability_schedule(t_life_bl_fpy: float, t_life_div_fpy: float,
                             dt_rep_bl: float, dt_rep_div: float,
                             Util_factor: float, Dwell_factor: float) -> tuple:
    """
    Effective plant availability from a two-component replacement schedule.

    Blanket (bl) and divertor (div) replacements are performed in parallel
    whenever they coincide, avoiding additive downtime.

    Algorithm
    ---------
    Let A = longer-lived component, B = shorter-lived.
    n = floor(t_A / t_B)  — how many B-cycles fit inside one A-cycle.

    Every t_B calendar years a replacement occurs:
      - (n−1) out of n times: only B replaced  → downtime = dt_B
      - 1 out of n times:     both replaced    → downtime = max(dt_A, dt_B)

    Effective average downtime per t_B cycle:
        dt_eff = [(n−1)*dt_B + max(dt_A, dt_B)] / n

    Availability: Av = t_B_yr / (t_B_yr + dt_eff)
    Capacity factor: CF = Av * Util_factor * Dwell_factor

    Parameters
    ----------
    t_life_bl_fpy  : float  Blanket lifetime [fpy].
    t_life_div_fpy : float  Divertor lifetime [fpy].
    dt_rep_bl      : float  Blanket replacement downtime [yr].
    dt_rep_div     : float  Divertor replacement downtime [yr].
    Util_factor    : float  Utilisation factor [-].
    Dwell_factor   : float  Dwell factor [-].

    Returns
    -------
    T_op_limit : float  Calendar years of operation per replacement cycle [yr].
    dt_rep_eff : float  Effective average replacement downtime per cycle [yr].
    Av         : float  Plant availability [-].
    CF         : float  Capacity factor [-].
    """
    UD = Util_factor * Dwell_factor
    t_bl_yr  = t_life_bl_fpy  / UD
    t_div_yr = t_life_div_fpy / UD

    # Identify longer (A) and shorter (B) component
    if t_bl_yr >= t_div_yr:
        t_A, t_B   = t_bl_yr,  t_div_yr
        dt_A, dt_B = dt_rep_bl, dt_rep_div
    else:
        t_A, t_B   = t_div_yr,  t_bl_yr
        dt_A, dt_B = dt_rep_div, dt_rep_bl

    # Number of B-cycles per A-cycle (floor, never exceed lifetime)
    n = max(1, int(t_A / t_B))

    dt_rep_eff = ((n - 1) * dt_B + max(dt_A, dt_B)) / n
    T_op_limit = t_B

    Av = T_op_limit / (T_op_limit + dt_rep_eff)
    CF = Av * UD
    return (T_op_limit, dt_rep_eff, Av, CF)


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
    ∫ J dA = Ip.  This is the same approach as used in
    f_q_profile_refined() for the j_Ohm composite share.

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
        rho_grid, Vprime, V_total = Vprime_data[:3]  # safe: 5-tuple (rho, V', V, dA, Lp)
        # Fetch the precomputed poloidal area element when present
        dA_grid = Vprime_data[3] if len(Vprime_data) >= 5 else None
        # Subsample if Vprime grid is finer than N_rho
        if len(rho_grid) > N_rho:
            idx = np.linspace(0, len(rho_grid) - 1, N_rho, dtype=int)
            rho   = rho_grid[idx]
            Vp    = Vprime[idx]
            dA_pol_per_drho = dA_grid[idx] if dA_grid is not None else None
        else:
            rho = rho_grid.copy()
            Vp  = Vprime.copy()
            dA_pol_per_drho = dA_grid.copy() if dA_grid is not None else None
        use_miller = True
    else:
        rho = np.linspace(1e-4, 0.98, N_rho)
        Vp  = None
        dA_pol_per_drho = None
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

    # Current density profile normalised to integral(J dA) = Ip.
    # In Miller mode with dA_pol_per_drho available, integrate directly
    # with the shaped Jacobian; otherwise fall back to the cylindrical
    # approximation dA = 2 pi a kappa rho drho (constant kappa, no delta).
    if use_miller and dA_pol_per_drho is not None:
        sigma_area_int = np.trapezoid(sigma_rho * dA_pol_per_drho, rho)
        if sigma_area_int < 1e-20:
            # Degenerate case — uniform J over total poloidal area
            A_pol_tot = float(np.trapezoid(dA_pol_per_drho, rho))
            J_rho = Ip_A / max(A_pol_tot, 1e-12) * np.ones_like(rho)
        else:
            J_rho = Ip_A * sigma_rho / sigma_area_int
    else:
        # Cylindrical fallback: dA = 2 pi a kappa rho drho
        # integral(J dA) = Ip  =>  J(rho) = Ip * sigma(rho) / (2 pi a kappa * integral(sigma * rho drho))
        sigma_area_int = np.trapezoid(sigma_rho * rho, rho)
        if sigma_area_int < 1e-20:
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
    # N_RE  = integral( f_RE(rho) * ne(rho) * V'(rho) drho )          (total RE count)
    # I_RE  = e * c * integral( f_RE(rho) * ne(rho) * dA_pol(rho) drho )  (seed current)
    if use_miller:
        integrand_N = f_RE * ne_rho * Vp
        N_RE = float(np.trapezoid(integrand_N, rho))

        # Poloidal cross-section area weight for I_RE.
        # Priority: precomputed dA_pol/drho (Miller-consistent, includes
        # kappa(rho) and delta(rho)). Fallback: large aspect ratio
        # approximation dA_pol/drho ~ V'(rho) / (2 pi R0).
        if dA_pol_per_drho is not None:
            integrand_I = f_RE * ne_rho * dA_pol_per_drho
        else:
            integrand_I = f_RE * ne_rho * Vp / (2.0 * np.pi * R0)
        I_RE = E_ELEM * C_LIGHT * float(np.trapezoid(integrand_I, rho))
    else:
        # Cylindrical: dV = V * 2rho drho,  dA = pi a^2 kappa * 2rho drho
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
                           pellet_dilution=10.0,
                           pellet_dilution_cools=False):
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
    pellet_dilution_cools : bool
        If True, isobaric energy conservation is enforced: the pre-TQ
        temperature is divided by pellet_dilution (n·T ≈ const).  This
        models a slow assimilation where the cold pellet material has
        thermally equilibrated with the bulk plasma before the TQ.
        Default False (rapid injection: density rises, temperature unchanged).

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
        'Tbar_diluted'   : float  Pre-TQ temperature used [keV]
                                  (= Tbar / pellet_dilution if isobaric).
        'pellet_dilution': float  Density multiplication factor used.
        'pellet_dilution_cools': bool  Whether isobaric cooling was applied.

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

    # ── Optional isobaric cooling ──────────────────────────────────────────────
    # When pellet_dilution_cools is True, the pre-TQ bulk temperature is
    # reduced assuming isobaric energy conservation:  n · T ≈ const  →
    # T_diluted = T_pre / pellet_dilution.  This applies to the hot-tail
    # seed only (the post-TQ residual Te_final_eV is set by the radiation
    # balance during the TQ itself and is not affected by pre-TQ dilution).
    Tbar_diluted = Tbar / pellet_dilution if pellet_dilution_cools else Tbar

    # ── Step 1: Hot-tail seed — evaluated at post-pellet density ─────────────
    # Both n_e and the Coulomb logarithm in the Smith model use nbar_diluted.
    # If pellet_dilution_cools, Tbar_diluted < Tbar (isobaric limit).
    ht = f_hot_tail_seed_profile(
        nbar_diluted, Tbar_diluted, Ip, a, R0, κ, Z_eff,
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
        'Tbar_diluted':    Tbar_diluted,        # [keV]  — pre-TQ temperature (cooled if isobaric)
        'pellet_dilution': pellet_dilution,     # [-]   — assimilation factor used
        'pellet_dilution_cools': pellet_dilution_cools,  # [-] — isobaric cooling flag
    }


# ── Validation ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── ITER chain (12/12) - q95, wall load and helium ash closure ───────
    # q95: the Sauter (2016) formula at the chain Ip and LCFS shaping
    # closes the forward reference used by chains 4, 9 and 10; the
    # ITER-1989 guideline formula at the PUBLISHED 95 % shaping and 15 MA
    # reproduces the ITER design value q95 = 3.0.
    # Helium ash: f_alpha solves the Sarazin steady-state balance with
    # the deck removal efficiency C_alpha = 5 (tau*_alpha = C_alpha
    # tau_E) at the chain tau_E. The value must close on the f_alpha
    # forward reference injected in chain 2, which is the convergence
    # criterion of the production solver. The production solver evaluates
    # the ash balance with the cylindrical volume weight even in refined
    # geometry (see D0FUS_run.py); the chain mirrors that call exactly.
    _q95 = f_q95(ITER['B0'], ITER['Ip'], ITER['R0'], ITER['a'],
                 ITER['kappa'], ITER['delta'], ITER['kappa95'],
                 ITER['delta95'], Option_q95='Sauter')
    _q95_pub = f_q95(5.3, 15.0, 6.2, 2.0, 1.85, 0.485, 1.70, 0.33,
                     Option_q95='ITER_1989')
    _Gam = f_Gamma_n(ITER['a'], ITER['P_fus'], ITER['R0'], ITER['kappa'],
                     S_wall=ITER['S'])
    _fa = f_He_fraction(ITER['nbar'], ITER['Tbar'], ITER['tauE'],
                        ITER['C_Alpha'], ITER['nu_T'],
                        rho_ped=ITER['rho_ped'],
                        T_ped_frac=ITER['T_ped_frac'], tau_i_e=1.0)
    _ta = f_tau_alpha(ITER['nbar'], ITER['Tbar'], ITER['tauE'],
                      ITER['C_Alpha'], ITER['nu_T'],
                      rho_ped=ITER['rho_ped'],
                      T_ped_frac=ITER['T_ped_frac'])
    _IOhm = f_I_Ohm(ITER['Ip'], ITER['Ib'], ITER['I_CD'])
    _bench("ITER chain 12/12 - q95, neutron wall load, helium ash", [
        ("q95 Sauter, chain Ip [-]", _q95, FROZEN['q95'], 2e-3,
         "deck frozen"),
        ("q95 ITER-1989, published point [-]", _q95_pub, 3.0, 0.01,
         "Uckan 1990"),
        ("Gamma_n, Miller wall [MW/m2]", _Gam, FROZEN['Gamma_n'], 2e-3,
         "deck frozen"),
        ("Gamma_n [MW/m2]", _Gam, 0.57, 0.05, "Shimada 2007"),
        ("f_He closure (C_alpha = 5) [-]", _fa, FROZEN['f_alpha'], 1e-3,
         "solver fixed point"),
        ("f_He, IPB exhaust assumption [%]", _fa * 100, "4.4", None,
         "IPB 1999"),
        ("tau*_alpha = C_alpha tau_E [s]", _ta, FROZEN['tau_alpha'], 1e-3,
         "deck frozen"),
        ("I_Ohm = Ip - I_bs - I_CD [MA]", _IOhm, FROZEN['I_Ohm'], 5e-3,
         "deck frozen"),
    ], notes=[
        "The f_He closure is the global loop of the chain: chain 2 "
        "injected the frozen f_alpha into the density solve, and the same "
        "value re-emerges from the ash balance at the chain tau_E.",
        "The IPB exhaust value (4.4 %) corresponds to the ITER Physics "
        "Basis assumption at its own operating point.",
        "The I_Ohm row closes the forward reference of chain 7 "
        "(0.2 % residual inherited from the Picard q-profile cache of "
        "the bootstrap step).",
    ])

if __name__ == "__main__":
    # ── Published anchors - runaway electrons ────────────────────────────
    # Hot-tail seed: Smith & Verwichte model evaluated at the Stahl
    # (2016) Fig. 2(b) point (informative: the figure quotes 4-5e-4).
    # Avalanche: Breizman et al., NF 59 (2019) 083001, Eq. 99 at the
    # Fig. 17 conditions (li = 1, Z = 4, Ip = 15 MA); the implementation
    # must reproduce the analytic Eq. 99 values, and the Fig. 17 ranges
    # are quoted for context.
    _f_RE = _hot_tail_fraction_local(2.8e19, 3.1e3, 1.4e6, Te_final_eV=31.0,
                                     tau_TQ=0.3e-3, Z_eff=1.0)
    _re_rows = [
        ("hot-tail f_RE (local) [-]", float(_f_RE), "4-5e-4", None,
         "Stahl 2016 Fig. 2b"),
        ("relativistic lnLambda (1e20, 5 eV)",
         float(_coulomb_log_relativistic(1e20, 5.0)), None, None,
         "Breizman 2019"),
    ]
    for _ire0, _ref99, _fig17 in ((1.0, 3.306, "~2-3"), (1e3, 7.999, "~7-9"),
                                  (1e6, 13.002, "~12-14")):
        _ire = f_RE_avalanche(15e6, _ire0, 1e20, 5.0, 1.0, 4) / 1e6
        _re_rows.append((f"I_RE avalanche, seed {_ire0:.0e} A [MA]",
                         float(_ire), _ref99, 1e-3, "Breizman Eq. 99"))
        _re_rows.append((f"  Fig. 17 range, seed {_ire0:.0e} A [MA]",
                         float(_ire), _fig17, None, "Breizman Fig. 17"))
    _bench("Published anchors - runaway electrons (hot tail, avalanche)",
           _re_rows)

    import D0FUS_BIB.D0FUS_figures as figs
    # plot_He_fraction takes separate ITER/DEMO removal efficiencies
    # (C_Alpha_ITER=5.0, C_Alpha_DEMO=7.0 by default); defaults match the
    # chain above (C_alpha = 5).
    figs.plot_He_fraction(C_Alpha_ITER=ITER['C_Alpha'])

#%%

# print("D0FUS_physical_functions loaded")


if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────
    # Full-device regression: the shipped reference deck must reproduce
    # the frozen 2026-06 values (anti-drift guard; intentional physics
    # changes must update these anchors AND the FROZEN dict at the top of
    # this file). Skipped gracefully if the deck is absent. Indices
    # follow the save_run_output tuple map of D0FUS_EXE/D0FUS_run.py.
    # ─────────────────────────────────────────────────────────────────────
    try:
        from D0FUS_EXE.D0FUS_run import load_config_from_file, run
        _deck = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'D0FUS_INPUTS', '1_run_ITER.txt')
        _res = run(load_config_from_file(_deck), verbose=0)
        _frozen_idx = [
            (0, "B0 [T]", 5.300), (3, "tau_E [s]", 3.144),
            (5, "Q [-]", 9.982), (8, "Ip [MA]", 14.968),
            (9, "I_bs [MA]", 4.816), (13, "n_line [1e20 m-3]", 1.012),
            (14, "n_GW [1e20 m-3]", 1.191), (16, "beta_N [-]", 1.635),
            (20, "q95 [-]", 3.598), (23, "P_LH [MW]", 73.522),
            (40, "f_alpha [-]", 0.029762),
        ]
        _rows = [(f"deck[{_i}] {_nm}", float(_res[_i]), _v, 5e-3,
                  "frozen 2026-06") for _i, _nm, _v in _frozen_idx]
        _rows.append(("deck[4] W_th [MJ]", float(_res[4]) / 1e6, 340.15,
                      5e-3, "frozen 2026-06"))
        _bench("Full-device regression - shipped ITER deck", _rows, notes=[
            "Closes the chain: every forward reference (f_alpha, "
            "f_imp_dil, q95, I_Ohm) and every chain output is "
            "re-produced by the assembled solver on the shipped deck.",
        ])
    except FileNotFoundError:
        print("--  ITER deck not found: full-device regression skipped")

    _bench_summary()