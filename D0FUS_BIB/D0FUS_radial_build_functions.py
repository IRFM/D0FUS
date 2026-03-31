"""
Radial Build functions definition for the D0FUS - Design 0-dimensional for FUsion Systems project
Created on: Dec 2023
Author: Auclair Timothe
"""

#%% Import

# When imported as a module (normal usage in production)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_parameterization import *
    from .D0FUS_physical_functions import (f_nprof, f_Tprof, f_q_profile,
                                           eta_old, eta_spitzer, eta_sauter,
                                           eta_redl, _coulomb_logarithm,
                                           f_Reff, f_Vloop)

# When executed directly (for testing and development)
else:
    import sys
    import os
    
    # Add parent directory to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import using absolute paths for standalone execution
    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_parameterization import *
    from D0FUS_BIB.D0FUS_physical_functions import (f_nprof, f_Tprof, f_q_profile,
                                                    eta_old, eta_spitzer, eta_sauter,
                                                    eta_redl, _coulomb_logarithm,
                                                    f_Reff, f_Vloop)
    
    # Initialisation of the default config paramters
    cfg = DEFAULT_CONFIG
    
    # Suppress numpy divide/invalid warnings — these are handled via NaN returns
    warnings.filterwarnings('ignore', category=RuntimeWarning)

from scipy.special import ellipk, ellipe
    
#%% ========================================================================
# MODULE-LEVEL UTILITIES
# =========================================================================

def gamma_func(alpha_val, n_val):
    """
    Effective steel area fraction for transverse (axial-face) loading.

    Derived from square-conductor geometry with n_val turns sharing
    the inter-turn load path.  Used by TF winding pack, CS D0FUS,
    and CS CIRCE solvers.

    Parameters
    ----------
    alpha_val : float
        Superconductor packing fraction [-] (0 < alpha < 1).
    n_val : float
        Conductor shape / turn-count factor [-].
        n = 1: square jacket;  n → ∞: limit of many thin conductors.

    Returns
    -------
    gamma : float
        Effective steel fraction [-].  NaN if inputs are out of range.
    """
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


# ── TF winding pack grading helpers ──────────────────────────────────

def _build_gamma_inverse_table(n_val, n_pts=2000):
    """
    Pre-compute lookup table for fast inversion of γ(α, n).

    γ is monotonically decreasing in α: α=0 → γ≈1, α→1 → γ→0.
    Table sorted by increasing γ for use with np.interp.

    Parameters
    ----------
    n_val : float
        Conductor geometry factor (same as in gamma_func).
    n_pts : int
        Number of tabulation points.

    Returns
    -------
    gamma_tab : ndarray   γ values, sorted ascending.
    alpha_tab : ndarray   Corresponding α values.
    """
    alpha_arr = np.linspace(0.002, 0.998, n_pts)
    gamma_arr = np.array([gamma_func(a, n_val) for a in alpha_arr])
    valid = np.isfinite(gamma_arr) & (gamma_arr > 0) & (gamma_arr < 1)
    alpha_arr = alpha_arr[valid]
    gamma_arr = gamma_arr[valid]
    idx = np.argsort(gamma_arr)
    return gamma_arr[idx], alpha_arr[idx]


def _invert_gamma(gamma_target, gamma_tab, alpha_tab, alpha_min=0.01):
    """
    Find α such that γ(α, n) = gamma_target via table interpolation.

    Parameters
    ----------
    gamma_target : float    Desired γ value.
    gamma_tab, alpha_tab : ndarray   From _build_gamma_inverse_table.
    alpha_min : float       Hard lower bound on α.

    Returns
    -------
    alpha : float    Conductor fraction satisfying γ(α,n) ≈ gamma_target.
    """
    if gamma_target <= gamma_tab[0]:
        return 0.999
    if gamma_target >= gamma_tab[-1]:
        return alpha_min
    alpha = float(np.interp(gamma_target, gamma_tab, alpha_tab))
    return max(alpha, alpha_min)


# Module-level storage for the last graded WP profile (used by diagnostic plots)
_last_graded_profile = {}


def _solve_graded_wp(R_ext, B_max, J_max, sigma_max, omega, n, ln_term,
                     dR=5e-4, alpha_min=0.01, max_iter=60, tol=1e-4):
    """
    Graded TF winding pack: α(R) varies to saturate Tresca everywhere.

    Integrates from R_ext inward using cylindrical Ampere's law.
    σ_z is self-consistent via Picard iteration.

    After convergence, the last profile (R, α arrays and σ_z) is stored
    in the module-level dict _last_graded_profile for diagnostic plotting.

    Parameters
    ----------
    R_ext : float       WP outer radius (= R_0 - a - b) [m].
    B_max : float       Peak toroidal field at R_ext [T].
    J_max : float       Engineering current density (non-steel) [A/m²].
    sigma_max : float   Allowable Tresca stress [Pa].
    omega : float       Fraction of vertical tension on inboard leg [-].
    n : float           Conductor geometry factor for γ(α, n).
    ln_term : float     ln((R_0 + a + b) / R_ext).
    dR : float          Radial integration step [m].
    alpha_min : float   Minimum conductor fraction [-].
    max_iter : int      Maximum Picard iterations for σ_z.
    tol : float         Relative convergence tolerance on σ_z.

    Returns
    -------
    c_WP : float            Winding pack thickness [m].
    sigma_r_peak : float    Peak radial stress (at bore) [Pa].
    sigma_z : float         Converged vertical stress [Pa].
    sigma_theta : float     Hoop stress [Pa] (= 0).
    Steel_fraction : float  Area-weighted average ⟨1 − α⟩ [-].
    """
    gamma_tab, alpha_tab = _build_gamma_inverse_table(n)

    def sigma_z_fn(R_sep, f_steel):
        denom = R_ext**2 - R_sep**2
        if denom <= 0 or f_steel <= 0:
            return 0.0
        return (omega / f_steel) * B_max**2 * R_ext**2 \
               / (2 * μ0 * denom) * ln_term

    def integrate(sigma_z):
        sigma_r_budget = sigma_max - sigma_z
        if sigma_r_budget <= 0:
            return None

        R   = R_ext
        NI  = B_max * 2 * np.pi * R_ext / μ0
        B   = B_max
        sigma_r = 0.0

        R_list     = [R]
        alpha_list = []
        sr_list    = [sigma_r]

        while NI > 0 and R > dR:
            gamma_target = sigma_r / sigma_r_budget
            if gamma_target >= 1.0:
                return None
            alpha_val = _invert_gamma(gamma_target, gamma_tab, alpha_tab,
                                      alpha_min)
            alpha_list.append(alpha_val)

            dNI = alpha_val * J_max * 2 * np.pi * R * dR
            if dNI > NI:
                dR_last = NI / (alpha_val * J_max * 2 * np.pi * R)
                dNI = NI
                R_new = R - dR_last
            else:
                R_new = R - dR

            NI_new = NI - dNI
            if R_new <= 0:
                R_new = 1e-4
                NI_new = 0.0

            B_new = μ0 * NI_new / (2 * np.pi * R_new) if R_new > 0 else 0.0

            dR_eff = R - R_new
            sigma_r_new = sigma_r + alpha_val * J_max \
                          * 0.5 * (B + B_new) * dR_eff

            R   = R_new
            NI  = NI_new
            B   = B_new
            sigma_r = sigma_r_new

            R_list.append(R)
            sr_list.append(sigma_r)
            if NI <= 0:
                break

        if len(alpha_list) == 0:
            return None

        R_sep = R_list[-1]

        # Area-weighted average steel fraction
        R_arr     = np.array(R_list)
        alpha_arr = np.array(alpha_list)
        n_seg     = min(len(R_arr) - 1, len(alpha_arr))
        R_mid     = 0.5 * (R_arr[:n_seg] + R_arr[1:n_seg + 1])
        dR_seg    = np.abs(np.diff(R_arr[:n_seg + 1]))
        alpha_seg = alpha_arr[:n_seg]
        denom_area = R_ext**2 - R_sep**2
        f_steel = np.sum((1 - alpha_seg) * 2 * R_mid * dR_seg) \
                  / denom_area if denom_area > 0 else 0.5

        return {'R_sep': R_sep, 'delta_R': R_ext - R_sep,
                'sigma_r_peak': sr_list[-1], 'f_steel': f_steel,
                'R': np.array(R_list), 'alpha': np.array(alpha_list)}

    # Initial σ_z guess
    dR_guess = B_max / (0.3 * μ0 * J_max)
    R_sep_guess = max(R_ext - dR_guess, 0.05)
    sigma_z = sigma_z_fn(R_sep_guess, 0.7)

    # Picard iteration
    result = None
    for it in range(max_iter):
        result = integrate(sigma_z)
        if result is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        R_sep_new = result['R_sep']
        if R_sep_new <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        sigma_z_new = sigma_z_fn(R_sep_new, result['f_steel'])
        if abs(sigma_z_new - sigma_z) / max(abs(sigma_z), 1e6) < tol:
            sigma_z = sigma_z_new
            break
        sigma_z = 0.5 * sigma_z + 0.5 * sigma_z_new

    # Final integration with converged σ_z
    result = integrate(sigma_z)
    if result is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Store profile for diagnostic plots (mutate in place for importers)
    _last_graded_profile.clear()
    _last_graded_profile.update({
        'R': result['R'], 'alpha': result['alpha'], 'sigma_z': sigma_z})

    return (result['delta_R'], result['sigma_r_peak'],
            sigma_z, 0.0, result['f_steel'])


def _find_bracket(residual_fn, d_lo, d_hi, n_pts):
    """
    Hybrid log+linear probe for root bracketing on a potentially partial domain.

    Scans a hybrid grid (2/3 log-spaced over full range + 1/3 linearly-spaced
    in the upper 60% of the range) and returns ALL nan→finite transitions and
    ALL sign changes of residual_fn.

    The hybrid grid addresses the failure mode where large-d solutions (thick
    LTS windings) fall between consecutive log-spaced points whose spacing
    exceeds 0.5 m.  The linear overlay in the upper portion of the range
    guarantees sub-0.3 m spacing there, at negligible extra cost (+5–7 pts).

    Parameters
    ----------
    residual_fn : callable
        Scalar function f(d) → float or NaN.
    d_lo, d_hi : float
        Search domain bounds (must satisfy d_lo < d_hi, both > 0).
    n_pts : int
        Target number of probe points (actual count may differ slightly
        after deduplication of log and linear grids).

    Returns
    -------
    nan_transitions : list of (float, float)
        All brackets (d_left, d_right) around nan→finite transitions,
        ordered by increasing d_left.
    sign_changes : list of (float, float)
        All brackets (d_left, d_right) around sign changes,
        ordered by increasing d_left.
    """
    # Hybrid grid: 2/3 log-spaced (resolution at small d for thin HTS WP)
    # + 1/3 linear in the upper 60% (fills gaps at large d for thick LTS WP)
    if n_pts >= 8:
        n_log = max(n_pts * 2 // 3, 5)
        n_lin = n_pts - n_log
        d_log = np.logspace(np.log10(d_lo), np.log10(d_hi), n_log)
        d_lin = np.linspace(d_hi * 0.4, d_hi, n_lin + 2)[1:-1]
        d_vals = np.unique(np.sort(np.concatenate([d_log, d_lin])))
    else:
        d_vals = np.logspace(np.log10(d_lo), np.log10(d_hi), n_pts)

    y_vals = np.array([residual_fn(d) for d in d_vals])

    nan_transitions = []
    sign_changes    = []
    for i in range(1, len(d_vals)):
        fp, fc = y_vals[i - 1], y_vals[i]
        finite_p, finite_c = np.isfinite(fp), np.isfinite(fc)
        if not finite_p and finite_c:
            nan_transitions.append((d_vals[i - 1], d_vals[i]))
        if finite_p and finite_c and fp * fc < 0:
            sign_changes.append((d_vals[i - 1], d_vals[i]))

    return nan_transitions, sign_changes


def _bisect_valid_boundary(residual_fn, lo, hi, n_iter=25, tol=1e-6):
    """
    Bisection on the nan/finite boundary of residual_fn.

    Finds the smallest d in [lo, hi] where residual_fn(d) is finite,
    assuming lo is in the nan region and hi is in the finite region.

    Parameters
    ----------
    residual_fn : callable
        Scalar function f(d) → float or NaN.
    lo, hi : float
        Bracket endpoints (lo → NaN, hi → finite).
    n_iter : int
        Maximum bisection iterations.
    tol : float
        Convergence tolerance on hi - lo [m].

    Returns
    -------
    d_min_valid : float
        Approximate boundary between nan and finite regions.
    """
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if np.isfinite(residual_fn(mid)):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def _adaptive_root_search(residual_fn, d_lo, d_hi,
                          n_probe_1=20, n_probe_2=25,
                          select='smallest', brentq_xtol=1e-4):
    """
    Adaptive 3-pass root search for residual functions with partial NaN domains.

    Factorises the search pattern shared by Winding_Pack_D0FUS, f_CS_D0FUS,
    and f_CS_CIRCE.  Handles non-monotone residuals (e.g. CIRCE where the
    Tresca maximum can jump between bore and outer radius) by collecting
    ALL sign-change brackets and returning the root selected by ``select``.

    Algorithm
    ---------
    Pass 1 — Hybrid probe (n_probe_1 pts) over [d_lo, d_hi]:
        Collects all nan→finite transitions and sign-change brackets.

    Pass 2 — Adaptive refinement (triggered only if Pass 1 finds no bracket):
        For each nan→finite transition detected in Pass 1:
            a. Bisect to pin d_min_valid (~25 bisection steps).
            b. Fine hybrid probe (n_probe_2 pts) on [d_min_valid, d_hi].
            c. Accumulate any new sign-change brackets.
        Handles islands where the entire valid domain fits between two
        consecutive Pass 1 points (the primary failure mode at high field).

    Pass 3 — brentq on every bracket; return best root per ``select``.

    Worst-case evaluation budget: n_probe_1 + 25 + n_probe_2 + 15 ~ 85.
    Typical (Pass 1 succeeds): n_probe_1 + 15 ~ 35.

    Parameters
    ----------
    residual_fn : callable
        Scalar function f(d) → float or NaN.
    d_lo, d_hi : float
        Search domain bounds (both > 0, d_lo < d_hi).
    n_probe_1 : int
        Number of probe points for Pass 1 (default: 20).
    n_probe_2 : int
        Number of probe points for Pass 2 per nan transition (default: 25).
    select : str
        Root selection: 'smallest' (default — thinnest feasible WP/CS)
        or 'largest'.
    brentq_xtol : float
        Absolute tolerance for brentq [m] (default: 1e-4 = 0.1 mm).

    Returns
    -------
    d_root : float
        Root of residual_fn, or NaN if no root exists in [d_lo, d_hi].
    """
    if d_lo >= d_hi:
        return np.nan

    # ── Pass 1: coarse hybrid probe ──────────────────────────────────
    nan_transitions, sign_changes = _find_bracket(
        residual_fn, d_lo, d_hi, n_probe_1)

    # ── Pass 2: adaptive refinement (only if no bracket from Pass 1) ─
    if not sign_changes:
        if not nan_transitions:
            # Residual is NaN everywhere in [d_lo, d_hi]:
            # no valid operating point exists (B > B_max_CS or J_wost = 0
            # for all thicknesses).
            return np.nan

        # Probe each nan→finite transition: the root may sit in a narrow
        # valid island that Pass 1 straddled entirely.
        for nt_lo, nt_hi in nan_transitions:
            d_min_valid = _bisect_valid_boundary(residual_fn, nt_lo, nt_hi)
            if d_min_valid >= d_hi:
                continue
            _, new_scs = _find_bracket(
                residual_fn, d_min_valid, d_hi, n_probe_2)
            sign_changes.extend(new_scs)

        if not sign_changes:
            # No sign change found even after refinement.  Two sub-cases:
            #
            # (a) Residual < 0 everywhere in the valid domain:
            #     stress constraint is satisfied for ALL valid d.
            #     Return the thinnest valid winding pack (d_min_valid).
            #     This can happen for conservative σ_CS with high-Jc HTS.
            #
            # (b) Residual > 0 everywhere: stress limit exceeded for all d.
            #     Design is infeasible → return NaN.
            if select == 'smallest' and nan_transitions:
                d_boundary = _bisect_valid_boundary(
                    residual_fn,
                    nan_transitions[0][0],
                    nan_transitions[0][1])
                y_boundary = residual_fn(d_boundary)
                if np.isfinite(y_boundary) and y_boundary < 0:
                    return d_boundary
            return np.nan

    # ── Pass 3: brentq on every bracket, collect all roots ───────────
    roots = []
    for lo, hi in sign_changes:
        try:
            d_root = brentq(residual_fn, lo, hi,
                            xtol=brentq_xtol, full_output=False)
            roots.append(d_root)
        except ValueError:
            continue

    if not roots:
        return np.nan

    return min(roots) if select == 'smallest' else max(roots)


def _unpack_CS_config(config):
    """
    Extract CS-relevant parameters from GlobalConfig.

    Centralises the config → local variable unpacking shared by
    f_CS_ACAD, f_CS_D0FUS, and f_CS_CIRCE.

    Returns
    -------
    dict
        Keys: Gap, I_cond, V_max, f_He_pipe, f_void, f_In,
        T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn, Marge_T_NbTi,
        Marge_T_REBCO, Eps, Tet, n_shape_CS.
    """
    return dict(
        Gap            = config.Gap,
        I_cond         = config.I_cond,
        V_max          = config.V_max,
        f_He_pipe      = config.f_He_pipe,
        f_void         = config.f_void,
        f_In           = config.f_In,
        T_hotspot      = config.T_hotspot,
        RRR            = config.RRR,
        Marge_T_He     = config.Marge_T_He,
        Marge_T_Nb3Sn  = config.Marge_T_Nb3Sn,
        Marge_T_NbTi   = config.Marge_T_NbTi,
        Marge_T_REBCO  = config.Marge_T_REBCO,
        Eps            = config.Eps,
        Tet            = config.Tet,
        n_shape_CS     = config.n_shape_CS,
        # Fatigue knockdown factor and operation mode — consumed by
        # the stress residual / f_sigma_diff closures in each CS solver.
        fatigue_CS     = config.fatigue_CS,
        Operation_mode = config.Operation_mode,
    )


#%% print

if __name__ == "__main__":
    print("##################################################### Steel Model ##########################################################")
    
#%% Steel

def Steel(Chosen_Steel, σ_manual):
    if Chosen_Steel == '316L':
        σ = 660*1e6        # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'N50H':
        σ = 1000*1e6       # Mechanical limit of the steel considered in [Pa]
    elif Chosen_Steel == 'Manual':
        σ = σ_manual*1e6   # Mechanical limit of the steel considered in [Pa]
    else : 
        raise ValueError(
            f"Unknown steel '{Chosen_Steel}'. "
            "Valid options: '316L', 'N50H', 'Manual'."
        )
    return(σ)

#%% print

if __name__ == "__main__":
    print("##################################################### TF coils number ##########################################################")

#%% Number of TF to satisfy ripple

def Number_TF_coils(R0, a, b, ripple_adm, L_min):
    """
    Find the minimum number of toroidal field (TF) coils required
    to keep the magnetic field ripple below a target value and satisfy
    a minimum toroidal access.

    Model (Wesson, 'Tokamaks', p.169):
        Ripple ≈ ((R0 - a - b)/(R0 + a))**N_TF + ((R0 + a)/(R0 + a + b + Delta))**N_TF
        L_access = 2 * pi * r2 / N_TF

    Limitations
    -----------
    This is a filamentary-coil model (point conductors on the midplane).
    For real D-shaped TF coils, the poloidal extent of the conductor
    provides partial cancellation of the ripple harmonic, reducing the
    actual ripple by typically 20-40% compared to this model (Goldston &
    Rutherford, "Introduction to Plasma Physics", 1995, ch. 14).
    Consequently, this function tends to overestimate the required N_TF
    by 1-2 coils, which is conservative for design purposes.
    For detailed studies, a 3D Biot-Savart calculation with realistic
    coil geometry is recommended.

    Parameters
    ----------
    R0 : float
        Major radius of the plasma [m]
    a : float
        Minor radius of the plasma [m]
    b : float
        Base radial distance between plasma edge and TF coil [m]
    ripple_adm : float
        Maximum admissible ripple (fraction, e.g. 0.01 for 1%)
    L_min : float
        Minimum toroidal access [m]

    Returns
    -------
    N_TF : int
        Minimum integer number of TF coils satisfying ripple <= ripple_adm
        and L_access >= L_min
    ripple_val : float
        Corresponding ripple value
    Delta : float
        Additional margin added to r2 to satisfy both constraints
    """

    N_min = 1
    N_max = 200
    delta_step = 0.01
    delta_max = 6

    if ripple_adm <= 0 or ripple_adm >= 1:
        raise ValueError("ripple_adm must be a fraction between 0 and 1.")
    if b <= 0:
        raise ValueError("b must be positive (coil must be outside the plasma).")
    if L_min <= 0:
        raise ValueError("L_min must be positive.")

    # Scan Delta from 0 to delta_max
    Delta = 0.0
    while Delta <= delta_max:
        r2 = R0 + a + b + Delta
        for N_TF in range(N_min, N_max + 1):
            ripple = ((R0 - a - b) / (R0 + a)) ** N_TF + ((R0 + a) / r2) ** N_TF
            L_access = 2 * math.pi * r2 / N_TF
            if ripple <= ripple_adm and L_access >= L_min:
                return N_TF, ripple, Delta
        Delta += delta_step

    raise ValueError(f"No N_TF and Delta combination found up to Delta_max={delta_max} m "
                     f"satisfying ripple ≤ {ripple_adm} and L_access ≥ {L_min} m")
    
#%% TF coil number test

if __name__ == "__main__":
    
    print("="*70)
    print("ITER prediction for the number of TF")
    print("Considering ripple and port minimal size")
    print("="*70)
    
    R0 = 6.2           # [m] major radius
    a = 2.0            # [m] minor radius
    b = 1.2            # [m] base radial distance
    ripple_adm = 0.01  # 1% ripple
    L_min = 3.6        # [m] minimum toroidal access

    N_TF, ripple, Delta = Number_TF_coils(R0, a, b, ripple_adm, L_min)
    r2 = R0 + a + b + Delta

    print(f"Calculated number of TF coils: {N_TF}")
    print(f"ITER TF coils: 18")
    print(f"Additional Delta = {Delta:.3f} m")
    print(f"Ripple = {ripple*100:.3f}%")
    print(f"L_access = {2*math.pi*r2/N_TF:.3f} m")
    print("="*70)

#%% print

if __name__ == "__main__":
    print("##################################################### Jc Model ##########################################################")

#%% Critical Current Density Scaling Laws for Superconductors

"""
Critical Current Density Scaling Laws for Superconductors
==========================================================

The critical current density Jc​(B,T) of a superconductor decreases as the 
magnetic field and temperature approach the material's critical surface, 
bounded by the upper critical field Bc2​ and critical temperature Tc​.
Empirical scaling laws capture this behavior through fits to experimental
data on production-grade conductors.
Nb₃Sn scaling additionally includes strain dependence ε due to the brittleness of
its crystal structure.
REBCO scaling also accounts for the anisotropy of Jc​ with respect to field orientation.

Scaling laws for Nb3Sn, NbTi, and REBCO superconductors used in fusion magnet design:

References
----------
[1] Corato, V. et al. (2016). "Common operating values for DEMO magnets design 
    for 2016". EUROfusion, IDM Ref. EFDA_D_2MMDTG.
    
[2] Fleiter, J. & Ballarino, A. (2014). "Parameterization of the critical surface 
    of REBCO conductors from Fujikura". CERN EDMS 1426239.
    
[3] Bajas, H. & Tommasini, D. (2022). "The SHiP spectrometer magnet – 
    Superconducting options". CERN-SHiP-NOTE-2022-001, EDMS 2440157.
    
[4] Tsuchiya, K. et al. (2017). "Critical current measurement of commercial 
    REBCO conductors at 4.2 K". Cryogenics 85, 1-7.
    
[5] Senatore, C. et al. (2024). "REBCO tapes for applications in ultra-high 
    fields: critical current surface and scaling relations". 
    Supercond. Sci. Technol. 37, 115013.
"""

def J_non_Cu_Nb3Sn(B, T, Eps=-0.003):
    """
    Critical current density for Nb3Sn on non-Cu cross-section.
    
    Formula:
        Jc = (C/B) · s(ε) · (1-t^1.52) · (1-t²) · b^p · (1-b)^q
    
    Parameters
    ----------
    B : float or array
        Magnetic field [T]
    T : float or array
        Temperature [K]
    Eps : float
        Applied strain [-] (default: -0.003, typical operating strain)
        
    Returns
    -------
    J_non_Cu : float or array
        Critical current density [A/m²] on superconducting (non-Cu) area
        
    Reference
    ---------
    Corato et al. (2016), Table 2.1 - EU-DEMO WST strand parameters
    """
    B = np.atleast_1d(np.array(B, dtype=float))
    T = np.atleast_1d(np.array(T, dtype=float))
    
    # EU-DEMO WST strand parameters [Ref. 1]
    Ca1, Ca2 = 50.06, 0.0
    Eps0a = 0.00312
    Bc2m, Tcm = 33.24, 16.34      # [T], [K]
    C = 83075                      # [AT/mm²] on SC area
    p, q = 0.593, 2.156
    
    # Strain function s(ε)  [Eps0a > 0 by definition]
    s_eps = 1 + (Ca1 / (1 - Ca1 * Eps0a)) * (
        Eps0a - np.sqrt(Eps**2 + Eps0a**2)
    )
    
    # Critical temperature and field
    Tc0_eps = Tcm * s_eps**(1/3)
    t = np.clip(T / Tc0_eps, 0, 1 - 1e-10)
    
    Bc2_T_eps = Bc2m * s_eps * (1 - t**1.52)
    b = np.clip(B / Bc2_T_eps, 0, 1 - 1e-10)
    
    # Jc formula
    Jc = (C / B) * s_eps * (1 - t**1.52) * (1 - t**2) * b**p * (1 - b)**q
    Jc = Jc * 1e6  # AT/mm² → A/m²
    Jc = np.where((t >= 1) | (b >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)


def J_non_Cu_NbTi(B, T):
    """
    Critical current density for NbTi on non-Cu cross-section.
    
    Formula:
        Jc = (C0/B) · (1 - t^1.7)^γ · b^α · (1-b)^β
    
    Parameters
    ----------
    B : float or array
        Magnetic field [T]
    T : float or array
        Temperature [K]
        
    Returns
    -------
    J_non_Cu : float or array
        Critical current density [A/m²] on NbTi (non-Cu) area
        
    Reference
    ---------
    Corato et al. (2016), Section 2.2 - ITER/EU-DEMO parameters
    """
    B = np.atleast_1d(np.array(B, dtype=float))
    T = np.atleast_1d(np.array(T, dtype=float))
    
    # ITER/EU-DEMO parameters [Ref. 1]
    Tc0, Bc20 = 9.03, 14.61       # [K], [T]
    C0 = 168512                    # [A/mm²] on NbTi area
    alpha, beta, gamma = 1.0, 1.54, 2.1
    
    t = np.clip(T / Tc0, 0, 1 - 1e-10)
    Bc2_T = Bc20 * (1 - t**1.7)
    b = np.clip(B / Bc2_T, 0, 1 - 1e-10)
    
    Jc = (C0 / B) * (1 - t**1.7)**gamma * b**alpha * (1 - b)**beta
    Jc = Jc * 1e6  # A/mm² → A/m²
    Jc = np.where((t >= 1) | (b >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)


def J_non_Cu_REBCO(B, T, Tet=0, dataset='Fujikura_2019'):
    """
    Engineering current density for REBCO (non-copper cross-section).

    Multiple datasets are available, selected by the ``dataset`` parameter.
    The default 'Fujikura_2019' reflects the state of the art of modern
    high-performance tapes as of 2024.

    Parameters
    ----------
    B : float or array
        Magnetic field [T].
    T : float or array
        Temperature [K].
    Tet : float
        Field angle [rad]: 0 = B⊥tape (B//c-axis, minimum Ic),
        π/2 = B//tape (B//ab-plane, maximum Ic).
    dataset : str, optional
        REBCO tape dataset:

        'Fujikura_2019' (default)
            Senatore et al., Supercond. Sci. Technol. 37 (2024) 115013.
            Fujikura FESC 19-0008: EuBCO 2.5 µm, IBAD/PLD, BHO columns.
            Pinning-force scaling (Dew-Hughes) with exponential T dependence.
            p=0.77, q=4.5 (Fig 8a); T*=22 K (Table 4); Bpeak(20K)=12.5 T.
            Non-Cu Jc ~2000 A/mm² at 4.2 K, 19 T, B⊥tape.
            Representative of modern high-performance APC tapes.

        'SuperOx_2019'
            Senatore et al., Supercond. Sci. Technol. 37 (2024) 115013.
            SuperOx #337-R: YBCO 2.7 µm, IBAD/PLD, native Y₂O₃ particles.
            p=0.64, q=2.2 (Fig 8b); T*=25 K (Table 4); Bpeak(20K)=17 T.
            Non-Cu Jc ~2000 A/mm² at 4.2 K, 19 T, B⊥tape.
            Higher T*: retains more Jc at 20 K than Fujikura.
            Representative of modern large-batch native-pinning tapes.

        'Fleiter_2014'
            Fleiter & Ballarino (2014), CERN EDMS 1426239.
            Calibrated on Fujikura 12 mm tape (~2012–2014 vintage).
            Uses irreversibility-field form with angular interpolation.
            OUTDATED: underestimates modern (2019+) tapes by ~2× at 19 T.
            Retained for backward compatibility and conservative bounding.

    Returns
    -------
    Jc : float or array
        Non-copper engineering critical current density [A/m²].

    Notes
    -----
    Batch-to-batch variability: Senatore 2024 reports ±30% spread in Jc
    at 20 K / 20 T across batches of the same manufacturer.  The datasets
    here correspond to specific tape samples tested at UNIGE; actual tape
    performance should be confirmed by the manufacturer for each order.

    References
    ----------
    [2] Fleiter & Ballarino (2014), CERN EDMS 1426239.
    [3] Bajas & Tommasini (2022), CERN-SHiP-NOTE-2022-001, Table 3.
    [5] Senatore et al. (2024), Supercond. Sci. Technol. 37, 115013.
        Open Access: https://doi.org/10.1088/1361-6668/ad7f95
    """
    if dataset == 'Fleiter_2014':
        return _J_REBCO_Fleiter2014(B, T, Tet)
    elif dataset in ('Fujikura_2019', 'SuperOx_2019'):
        return _J_REBCO_Senatore2024(B, T, Tet, dataset)
    else:
        raise ValueError(
            f"Unknown REBCO dataset '{dataset}'. Choose from: "
            "'Fleiter_2014', 'Fujikura_2019', 'SuperOx_2019'."
        )


def _J_REBCO_Fleiter2014(B, T, Tet=0):
    """
    Fleiter & Ballarino (2014) REBCO scaling — Fujikura 12 mm tape (~2012).

    Irreversibility-field model with c-axis / ab-plane angular interpolation.
    Parameters from CERN EDMS 1426239 and Bajas & Tommasini (2022).

    Data vintage caveat: calibrated on ~2012–2014 Fujikura tapes.  Modern
    tapes (2019+) achieve 2–2.5× higher Jc at 19 T / 4.2 K due to improved
    artificial pinning (BHO nanorods).  Use 'Fujikura_2019' or 'SuperOx_2019'
    for modern tape performance.
    """
    B = np.atleast_1d(np.array(B, dtype=float))
    T = np.atleast_1d(np.array(T, dtype=float))
    
    # Geometry (Fujikura 12mm tape)
    w_tape = 12e-3      # Tape width [m]
    t_tape = 100e-6     # Total tape thickness [m]
    # 50 µm Hasteloy, 2 µm Ag, 2 µm REBCO, 40 µm Cu  -> ~ 100 µm in total
    t_tape_wCu = 60e-6  # tape thickness without Copper
    A_tape_non_Cu = w_tape * t_tape_wCu  # Cross-section [m²]
    
    # Fleiter 2014 parameters (Table 3, page 49 of Ref. [3])
    Tc0 = 93.0          # Critical temperature [K]
    n = 1.0
    
    # c-axis (B perpendicular to tape) parameters
    # Note: α scaled to match Tsuchiya (2017) Fujikura data at 12T, 4.2K
    Bi0c = 140.0        # Irreversibility field at T=0 [T]
    Alfc = 3.0e6 * w_tape    # [A·T] - calibrated to give Ic (~800-1000 A/mm² @ 12T)
    pc = 0.5
    qc = 2.5
    gamc = 2.44
    
    # ab-plane (B parallel to tape) parameters
    Bi0ab = 250.0       # [T]
    Alfab = 110e6 * w_tape   # [A·T] - scaled proportionally
    pab = 1.0
    qab = 5.0
    gamab = 1.63
    
    # Temperature exponents for ab-plane irreversibility field
    n1 = 1.4
    n2 = 4.45
    a_Bi = 0.1              # Additive correction in Biab(T) expression
    
    # Angular interpolation parameters
    Nu = 0.857
    g0, g1, g2, g3 = 0.03, 0.25, 0.06, 0.058
    
    # Reduced temperature
    tred = T / Tc0
    
    # Irreversibility fields
    Bic = Bi0c * (1 - tred**n)
    Biab = Bi0ab * ((1 - tred**n1)**n2 + a_Bi * (1 - tred**n))
    
    # Reduced fields (clipped to avoid singularities)
    bredc = np.clip(B / Bic, 1e-10, 1 - 1e-10)
    bredab = np.clip(B / Biab, 1e-10, 1 - 1e-10)
    
    # Critical currents [A] for each orientation
    Icc = (Alfc / B) * bredc**pc * (1 - bredc)**qc * (1 - tred**n)**gamc
    Icab = (Alfab / B) * bredab**pab * (1 - bredab)**qab * \
           ((1 - tred**n1)**n2 + a_Bi * (1 - tred**n))**gamab
    
    # Angular interpolation between c-axis and ab-plane
    g = g0 + g1 * np.exp(-g2 * np.exp(g3 * T) * B)
    Ic = Icc + (Icab - Icc) / (1 + (np.abs(Tet - np.pi/2) / g)**Nu)
    
    # Convert to engineering Jc on tape cross-section
    Jc = Ic / A_tape_non_Cu
    
    # Zero outside valid range
    Jc = np.where((tred >= 1) | (B <= 0), 0.0, Jc)
    
    return np.squeeze(Jc)


def _J_REBCO_Senatore2024(B, T, Tet=0, dataset='Fujikura_2019'):
    """
    Senatore et al. (2024) REBCO scaling — modern high-performance tapes.

    Model: Dew-Hughes pinning-force scaling with exponential T dependence.

        Jc(B,T) = Jc_ref × exp(-(T - T_ref) / T*) × fp(B,T) / fp(B_ref, T_ref)

    where the pinning-force function (proportional to Fp/B) is:
        fp(B,T) = b^(p-1) × (1-b)^q,   b = B / Birr(T)

    and the irreversibility field decreases with temperature as:
        Birr(T) = Birr0 × (1 - (T/Tc)^n1)^n2

    The model is anchored at a reference point (B_ref, T_ref) where
    the non-Cu Jc is known from transport measurements (Fig. 9 of [5]).

    Angular dependence (simplified): at angles other than B⊥tape (Tet≠0),
    the c-axis Jc is multiplied by an empirical anisotropy factor
    Γ(B) = 1 + k × B^m, fitted from Table 2 of [5]:
        Jc(B,T,θ) = Jc_c(B,T) × [1 + (Γ(B)-1) × sin²(θ)]
    This is a simplified model; for detailed angular studies, the full
    Hilton model (Eq. 1-2 of [5]) should be used.

    Accuracy (validated against Senatore 2024 Fig. 9 and text data):
      - At 4.2 K / 14-19 T: ±10% (anchored at B_ref).
      - At 4.2 K / 6-10 T:  ±20% (fp shape extrapolation from 20-77 K data).
      - At 20 K:  tape-dependent bias due to the fp(B,T)/fp(B_ref,T_ref)
        ratio picking up extra T-dependence through Birr(T).
        SuperOx (Birr0=84 T, small Birr drop 4K→20K):  ±5% at 19 T.
        Fujikura (Birr0=187 T, large Birr drop 4K→20K): −20 to −25% at 19 T.
        This bias is conservative for magnet design (underestimates Jc →
        overestimates required winding-pack thickness).

    Note on the fp/fp_ref factorisation and absence of explicit 1/Birr(T):
        In the standard Dew-Hughes formalism, Jc ∝ Fp_max(T)/Birr(T) × fp(B,T).
        Here, the combined [Fp_max(T)/Birr(T)] variation is implicitly captured
        by exp(-ΔT/T*).  However, T* was measured from total Jc(T) at fixed
        B = 1-6 T (Table 4 of [5]), where the fp ratio is close to 1.
        At higher B (where b = B/Birr deviates more between 4.2 K and 20 K),
        the fp ratio introduces additional T-suppression not absorbed by T*.
        The effect scales with the fractional Birr drop: negligible for tapes
        with low Birr0, significant for tapes with high Birr0.

    Parameters fitted from Senatore et al. SST 37 (2024) 115013:
    ---------------------------------------------------------------
    Fujikura_2019 (FESC 19-0008, EuBCO, BHO artificial pinning):
        p=0.77, q=4.5         [Fig 8a: pinning force shape]
        Birr0=187 T            [Fitted from Bpeak(T) in Table 5]
        n1=0.40, n2=1.0       [Birr temperature dependence]
        T*=22 K                [Table 4: average over 1-6 T]
        Jc_ref=2000 A/mm²     [Fig 9a: non-Cu Jc at 19T, 4.2K]

    SuperOx_2019 (#337-R, YBCO, native Y₂O₃ pinning):
        p=0.64, q=2.2         [Fig 8b: pinning force shape]
        Birr0=83.6 T           [Fitted from Bpeak(T) in Table 5]
        n1=2.0, n2=2.18       [Birr temperature dependence]
        T*=25 K                [Table 4: average over 1-6 T]
        Jc_ref=2000 A/mm²     [Fig 9a: non-Cu Jc at 19T, 4.2K]

    Reference
    ---------
    [5] Senatore C., Bonura M., Bagni T., Supercond. Sci. Technol. 37
        (2024) 115013. https://doi.org/10.1088/1361-6668/ad7f95
    """
    B = np.atleast_1d(np.asarray(B, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))

    # ── Parameter sets ────────────────────────────────────────────────
    _PARAMS = {
        'Fujikura_2019': dict(
            p=0.77, q=4.5,
            Birr0=187.0, n1=0.40, n2=1.0, Tc=93.0,
            T_star=22.0,
            Jc_ref=2000e6, B_ref=19.0, T_ref=4.2,
            k_Gamma=0.224, m_Gamma=0.80,
        ),
        'SuperOx_2019': dict(
            p=0.64, q=2.2,
            Birr0=83.6, n1=2.0, n2=2.18, Tc=93.0,
            T_star=25.0,
            Jc_ref=2000e6, B_ref=19.0, T_ref=4.2,
            k_Gamma=0.25, m_Gamma=0.75,
        ),
    }

    par = _PARAMS[dataset]
    p, q = par['p'], par['q']
    Birr0, n1, n2, Tc = par['Birr0'], par['n1'], par['n2'], par['Tc']
    T_star = par['T_star']
    Jc_ref, B_ref, T_ref = par['Jc_ref'], par['B_ref'], par['T_ref']

    # ── Irreversibility field Birr(T) ─────────────────────────────────
    def _Birr(T_):
        t = np.clip(T_ / Tc, 0, 1 - 1e-10)
        return Birr0 * np.maximum((1 - t**n1)**n2, 0.0)

    # ── Pinning-force function fp(B,T) ∝ Fp/B ────────────────────────
    def _fp(B_, T_):
        Bi = _Birr(T_)
        b = np.clip(B_ / np.maximum(Bi, 1e-10), 1e-10, 1 - 1e-10)
        return b**(p - 1) * (1 - b)**q

    # Normalisation: fp at the reference (B_ref, T_ref) point
    fp_ref = _fp(B_ref, T_ref)

    # ── c-axis Jc (B ⊥ tape, minimum Ic) ─────────────────────────────
    Jc_c = Jc_ref * np.exp(-(T - T_ref) / T_star) * _fp(B, T) / fp_ref

    # ── Angular dependence (simplified) ───────────────────────────────
    # Anisotropy ratio Γ(B) = Ic(ab-plane) / Ic(c-axis)
    # Fitted from Table 2 of Senatore (2024) at 4.2 K.
    # Γ increases with B and with T; the T-dependence is neglected here
    # for simplicity (Γ(20K) ≈ 1.3 × Γ(4.2K) at 18 T).
    if np.any(Tet != 0):
        k, m = par['k_Gamma'], par['m_Gamma']
        Gamma = 1.0 + k * np.maximum(B, 0.0)**m
        Jc = Jc_c * (1.0 + (Gamma - 1.0) * np.sin(Tet)**2)
    else:
        Jc = Jc_c

    # ── Zero outside valid range ──────────────────────────────────────
    Jc = np.where((T >= Tc) | (B <= 0) | (B >= _Birr(T)), 0.0, Jc)

    return np.squeeze(Jc)


#%% Current density validation

if __name__ == "__main__":
    # Superconductor Jc scaling laws: NbTi, Nb3Sn, REBCO — MAGLAB benchmark
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_Jc_scaling()

#%% Print
        
if __name__ == "__main__":
    print("##################################################### Cu Model ##########################################################")

#%% Copper Stabilizer Sizing for CICC Conductors
"""
Copper Stabilizer Sizing for CICC Conductors
=============================================

This module provides functions to size the copper stabilizer fraction in 
Cable-In-Conduit Conductors (CICC) based on quench protection requirements.

The approach follows the Maddock (adiabatic hot-spot) criterion: during a quench,
the conductor must survive the thermal transient without exceeding a maximum 
temperature. This sets a maximum allowable current density in the copper stabilizer,
which in turn determines the required Cu/SC ratio.

Physical basis:
    - During quench, current transfers from SC to Cu stabilizer
    - Adiabatic assumption: no heat removal during the fast transient
    - Energy balance: ∫ρJ²dt = ∫Cp·dT  →  J²·t_dump = Z(T_hotspot)
    - Joule integral: Z(T) = ∫(Cp/ρ)dT from T_op to T_hotspot

References
----------
[1] Maddock, B. J., et al. (1969). "Superconductive composites: heat transfer 
    and steady state stabilization". Cryogenics 9(4), 261-273.
[2] Mitchell, N., et al. (2008). "The ITER Magnet System". 
    IEEE Trans. Appl. Supercond. 18(2), 435-440.
[3] Torre, A. "Superconductors for Fusion". CEA Lecture Notes.
"""

# =============================================================================
# COPPER MATERIAL PROPERTIES
# =============================================================================

def get_copper_properties(T, B, RRR):
    """
    Copper resistivity and volumetric heat capacity.
    
    Parameters
    ----------
    T : float
        Temperature [K]
    B : float
        Magnetic field [T]
    RRR : float
        Residual Resistivity Ratio (default: 100)
        
    Returns
    -------
    rho : float
        Electrical resistivity [Ω·m]
    cp_vol : float
        Volumetric heat capacity [J/(m³·K)]
        
    Notes
    -----
    Source: CryoSoft/THEA library (rCopper.m, MAGRCU.m, cCopper.m)
    Valid range: 0.1-1000 K, 0-30 T, RRR 1.5-3000
    
    Includes magnetoresistance effect (Kohler's rule).
    """
    # Clamp inputs to valid range
    TT = max(0.1, min(T, 1000.0))
    R = max(1.5, min(RRR, 3000.0))
    BB = max(0.0, min(B, 30.0))
    
    # --- Resistivity ---
    RHO273 = 1.54e-8  # Resistivity at 273 K [Ω·m]
    P1, P2, P3, P4, P5, P6, P7 = 0.1171e-16, 4.49, 3.841e10, -1.14, 50.0, 6.428, 0.4531
    
    # Zero-temperature resistivity (impurity scattering)
    rhoZero = RHO273 / (R - 1.0)
    
    # Ideal resistivity (phonon scattering)
    arg = min((P5 / TT)**P6, 30.0)
    rhoI = P1 * TT**P2 / (1.0 + P1 * P3 * TT**(P2 + P4) * np.exp(-arg))
    
    # Matthiessen's rule with deviation term
    rhoI0 = P7 * rhoI * rhoZero / (rhoI + rhoZero)
    rho0 = rhoZero + rhoI + rhoI0
    
    # Magnetoresistance (Kohler's rule)
    RHORRR = 2.37e-8
    A1, A2, A3, A4 = 0.382806e-3, 1.32407, 0.167634e-2, 0.789953
    rhoIce = RHO273 + RHORRR / R
    brr = min(BB * rhoIce / rho0, 40.0e3)
    magR = A1 * brr**A2 / (1.0 + A3 * brr**A4) + 1.0 if brr > 1 else 1.0
    
    rho = magR * rho0
    
    # --- Heat capacity ---
    DENSITY = 8900.0  # kg/m³
    T0, T1 = 10.4529369, 48.26583891
    
    if TT <= T0:
        Cp = 0.01188007*TT - 0.00050323*TT**2 + 0.00075762*TT**3
    elif TT <= T1:
        Cp = -5.03283229 + 1.27426834*TT - 0.11610718*TT**2 + 0.00522402*TT**3 - 5.2996e-5*TT**4
    else:
        Cp = (-65.07570094*TT / (1.833505318 + TT)**0.518553624 +
              624.7552517*TT**3 / (16.55124429 + TT)**2.855560719 +
              0.529512119*TT**4 / (-0.000101401 + TT)**2.983928329)
    
    cp_vol = max(Cp, 0) * DENSITY
    
    return rho, cp_vol


def compute_quench_integral(T_op, T_hotspot, B, RRR, n_steps=200):
    """
    Joule integral Z(T) = ∫(Cp/ρ)dT for Maddock criterion.
    
    Parameters
    ----------
    T_op : float
        Operating temperature [K]
    T_hotspot : float
        Maximum allowable hot-spot temperature [K]
    B : float
        Magnetic field [T]
    RRR : float
        Residual Resistivity Ratio (default: 100)
    n_steps : int
        Number of integration steps (default: 200)
        
    Returns
    -------
    Z : float
        Joule integral [A²·s/m⁴]
        
    Reference
    ---------
    Maddock et al. (1969), Eq. 12
    """
    temp_array = np.linspace(T_op, T_hotspot, n_steps)
    
    # Vectorized integration: evaluate Cp/rho at all temperature points
    # and use trapezoidal rule (replaces scalar for-loop, ~5x faster)
    properties = np.array([get_copper_properties(T, B, RRR) for T in temp_array])
    rho_arr = properties[:, 0]   # Electrical resistivity [Ohm.m]
    cp_arr  = properties[:, 1]   # Volumetric heat capacity [J/(m³·K)]
    Z = np.trapezoid(cp_arr / rho_arr, temp_array)
    
    return Z


# =============================================================================
# MAGNETIC ENERGY CALCULATIONS
# =============================================================================

def calculate_E_mag_TF(B_max, r_bore_in, r_bore_out, H_TF):
    """
    Magnetic energy stored in toroidal field.
    
    The toroidal field varies as B(r) = B_max × r_bore_in / r.
    Integration over the toroidal volume yields:
        E_mag = (B_max² × r_bore_in² / 2μ₀) × 2π × H_TF × ln(r_bore_out / r_bore_in)
    
    Parameters
    ----------
    B_max : float
        Peak toroidal field at inner bore radius [T]
    r_bore_in : float
        Inner radius of TF bore (where B = B_max) [m]
        Typically: R_0 - a - gaps, or inner radius of TF inner leg
    r_bore_out : float
        Outer radius of toroidal volume [m]
        Typically: R_0 + a + gaps, or outer radius of TF outer leg
    H_TF : float
        Effective height of toroidal field region [m]
        
    Returns
    -------
    E_mag : float
        Magnetic energy [J]
    """
    E_mag = (B_max**2 * r_bore_in**2 / (2 * μ0)) * 2 * np.pi * H_TF * np.log(r_bore_out / r_bore_in)
    return E_mag



def calculate_E_mag_CS(B_max, r_in_CS, r_out_CS, H_CS):
    """
    Magnetic energy stored in Central Solenoid (bore + winding pack).

    Assumes a long thick solenoid with uniform engineering current density J,
    giving a linear radial field profile:
        B(r) = B_max                           for r < r_in   (bore)
        B(r) = B_max × (r_out - r) / d         for r_in ≤ r ≤ r_out  (winding)
    where d = r_out - r_in.

    The total stored energy (bore + winding) is computed analytically:
        E_mag = (π H B_max² / μ₀) × [r_in²/2 + r_out×d/3 - d²/4]

    Parameters
    ----------
    B_max : float
        Peak magnetic field at CS inner radius [T].
        For symmetric operation (-B_max to +B_max), this is related to the
        total flux swing by: B_max = 3Ψ_CS / (2π(r_out² + r_out r_in + r_in²)).
    r_in_CS : float
        Inner radius of CS winding pack [m].
    r_out_CS : float
        Outer radius of CS winding pack [m].
    H_CS : float
        Total height of CS [m].

    Returns
    -------
    E_mag : float
        Total magnetic energy [J], including bore + winding contributions.

    Notes
    -----
    The bore energy (B uniform at B_max) can dominate for thin CS (r_in >> d).
    For quench protection, the total stored energy is needed because the CS
    self-inductance includes flux linkage through the bore.

    Derivation
    ----------
    E_bore    = B_max²/(2μ₀) × π r_in² H
    E_winding = (πH B_max²)/(μ₀ d²) × ∫_{r_in}^{r_out} (r_out-r)² r dr
              = (πH B_max²/μ₀) × [r_out d/3 - d²/4]
              (using ∫ (Ro-r)² r dr = Ro d³/3 - d⁴/4 via substitution u = Ro-r)
    E_total   = E_bore + E_winding
    """
    d = r_out_CS - r_in_CS
    if d <= 0 or H_CS <= 0:
        return 0.0
    return (np.pi * H_CS * B_max**2 / μ0) * (r_in_CS**2 / 2.0
                                               + r_out_CS * d / 3.0
                                               - d**2 / 4.0)


# =============================================================================
# QUENCH PROTECTION
# =============================================================================

def calculate_t_dump(E_mag, I_cond, V_max, N_sub, tau_h):
    """
    Effective discharge time for hot-spot criterion.
    
    The dump circuit is characterized by:
        - Dump resistor: R_dump = V_max / I_cond
        - Coil inductance: L = 2 × E_mag / I_cond²
        - Discharge time constant: τ_dis = L / (N_sub × R_dump)
    
    The effective time for Maddock criterion:
        t_dump = τ_h + τ_dis / 2
    
    Parameters
    ----------
    E_mag : float
        Total magnetic energy [J]
    I_cond : float
        Operating current per conductor [A]
    V_max : float
        Maximum voltage to ground [V]
        Typical: 5-10 kV
    N_sub : int or float
        Number of subdivisions in protection circuit.
        Each subdivision has its own dump resistor.
    tau_h : float
        Detection + holding time before discharge [s]
        Typical: 2-5 s (LTS), 10-20 s (HTS)
        
    Returns
    -------
    t_dump : float
        Effective discharge time for hot-spot calculation [s]
        
    Reference
    ---------
    Torre, A. CEA Lecture Notes
    
    Note
    ---------
    Voltage convention :
    V_max here refers to the terminal voltage across the dump resistor
    (i.e. V_terminal = I_cond × R_dump), NOT the voltage to ground.
    For a symmetrically grounded circuit, the voltage to ground is:
        V_to_ground = V_terminal / 2
    ITER TF:  V_terminal ≈ 10–12 kV  →  V_to_ground ≈ 5–6 kV  [Fink 2005]
    ITER CS:  V_terminal ≈  6.3 kV   →  V_to_ground ≈ 3.2 kV  [Duchateau 2009]
    EU DEMO:  V_terminal ≈ 10 kV     →  V_to_ground ≈ 5 kV    [Novello 2019]
    A conservative default of 10 kV (terminal) is chosen
    """
    tau_dis = 2 * E_mag / (I_cond * V_max * N_sub)
    t_dump = tau_h + tau_dis / 2
    return t_dump


# =============================================================================
# CABLE FRACTION SIZING
# =============================================================================

def size_cable_fractions(J_non_Cu, B_peak, T_op, t_dump,
                         T_hotspot, f_He_pipe, f_void, f_In, RRR):
    """
    Calculate cable composition based on quench protection (Maddock criterion).

    The non-steel cross-section ("wost") of a CICC-like conductor is
    decomposed into three zones at the top level:

        wost = Insulation + He pipe + Active zone
             = f_In      + f_He_pipe + f_active
             with f_active = 1 - f_In - f_He_pipe

    Inside the active zone, strands (SC + Cu) are packed with an
    interstitial void fraction used for helium cooling in LTS conductors:

        Active zone = Void (interstitial He) + Strands (SC + Cu)
                    = f_void * f_active      + (1 - f_void) * f_active

    Typical values
    --------------
    LTS (Nb3Sn, NbTi) : f_He_pipe ~ 0.05-0.10, f_void = 0.33
    HTS (REBCO)        : f_He_pipe ~ 0.05-0.10, f_void = 0.00

    Parameters
    ----------
    J_non_Cu : float
        Critical current density on non-Cu cross-section [A/m²].
    B_peak : float
        Peak magnetic field at conductor [T].
    T_op : float
        Operating temperature [K].
    t_dump : float
        Effective discharge time [s] (from calculate_t_dump).
    T_hotspot : float
        Maximum allowable hot-spot temperature [K].
    f_He_pipe : float
        Fraction of wost dedicated to the helium cooling pipe/channel [-].
    f_void : float
        Interstitial void fraction inside the active strand bundle [-].
        LTS (NbTi, Nb3Sn): ~0.33;  HTS (REBCO): 0.00.
    f_In : float
        Insulation fraction in wost [-].
    RRR : float
        Copper Residual Resistivity Ratio.

    Returns
    -------
    dict
        f_sc      : float - Superconductor volume fraction (wost) [-]
        f_cu      : float - Copper volume fraction (wost) [-]
        f_He_pipe : float - Helium pipe fraction (wost) [-]
        f_void    : float - Interstitial void fraction (wost) [-]
        f_He      : float - Total helium fraction (pipe + void) (wost) [-]
        f_In      : float - Insulation fraction (wost) [-]
        J_wost    : float - Current density on non-steel area [A/m²]

    Notes
    -----
    "wost" = Without Steel.  All fractions are relative to the non-steel
    cross-section and sum to 1.0:
        f_In + f_He_pipe + f_void_wost + f_sc + f_cu = 1.0

    References
    ----------
    [1] Maddock et al. (1969) - Adiabatic hot-spot criterion
    [2] Mitchell et al. (2008) - ITER magnet design
    """
    # Joule integral from T_op to T_hotspot
    Z = compute_quench_integral(T_op, T_hotspot, B_peak, RRR)

    # Maximum Cu current density (adiabatic criterion)
    J_cu_max = np.sqrt(Z / t_dump)

    # Required Cu/SC ratio from quench protection
    ratio_cu_sc = J_non_Cu / J_cu_max

    # ── Hierarchical area decomposition in wost ──
    # Level 1: insulation + He pipe + active zone
    f_active = 1.0 - f_In - f_He_pipe

    # Level 2: inside active zone — interstitial void + strands
    f_strand_in_active = 1.0 - f_void

    # Level 3: inside strands — SC + Cu split by quench protection
    f_sc_in_strand = 1.0 / (1.0 + ratio_cu_sc)
    f_cu_in_strand = 1.0 - f_sc_in_strand

    # ── Convert all fractions to wost reference ──
    f_strand_wost = f_strand_in_active * f_active
    f_void_wost   = f_void * f_active

    f_sc = f_sc_in_strand * f_strand_wost
    f_cu = f_cu_in_strand * f_strand_wost

    # Total helium = dedicated pipe + interstitial void
    f_He_total = f_He_pipe + f_void_wost

    # Current density on non-steel area
    J_wost = J_non_Cu * f_sc

    return {
        "f_sc":      f_sc,
        "f_cu":      f_cu,
        "f_He_pipe": f_He_pipe,
        "f_void":    f_void_wost,
        "f_He":      f_He_total,
        "f_In":      f_In,
        "J_wost":    J_wost,
    }

#%% E_mag and t_dump test
    
if __name__ == "__main__":
    """
    Validation of magnetic energy and quench discharge time calculations.
    
    References
    ----------
    [1] Mitchell, N. et al., "The ITER Magnet System", IEEE Trans. Appl. Supercond. 18(2), 2008.
    [2] Fink, S. et al., "Transient electrical behaviour of the ITER TF coils during fast discharge", 
        Fusion Eng. Des. 75-79, 2005.
    [3] Takahashi, Y. et al., "Quench Detection Using Pick-Up Coils for the ITER Central Solenoid",
        IEEE Trans. Appl. Supercond. 15(2), 2005.
    [4] Duchateau, J.-L. et al., "Detailed design of the ITER central solenoid", 
        Fusion Eng. Des. 84(2-6), 2009.
    """
    
    print("\n" + "="*65)
    print("MAGNETIC ENERGY & DISCHARGE TIME VALIDATION")
    print("="*65)
    
    # =========================================================================
    # ITER TF SYSTEM
    # =========================================================================
    # The TF system consists of 18 coils with 9 Fast Discharge Units (FDU).
    # Reference values [1,2]: E_mag = 41 GJ, tau_dis = 11 s, I = 68 kA
    print("\n--- ITER TF ---")
    # Geometry: r_bore_in/out are the inner/outer radii of the toroidal volume
    # where B_tor exists, i.e., the TF inner and outer leg positions.
    B_max_TF = 11.8       # Peak field at inner leg [T]
    r_bore_in_TF = 2.7    # Inner leg radius [m]
    r_bore_out_TF = 9.5   # Outer leg radius [m]
    H_TF = 12.0           # Effective height [m]
    E_TF = calculate_E_mag_TF(B_max_TF, r_bore_in_TF, r_bore_out_TF, H_TF)
    E_TF_ref = 41e9
    # Discharge parameters [2]: 9 FDU, V_max ~ 6 kV, tau_dis ~ 11 s
    I_TF = 68e3
    V_max_TF = 10e3
    N_FDU_TF = 9
    tau_h_TF = 2.0
    t_dump_TF = calculate_t_dump(E_TF, I_TF, V_max_TF, N_FDU_TF, tau_h_TF)
    tau_dis_TF = 2 * E_TF / (I_TF * V_max_TF * N_FDU_TF)
    print(f"  E_mag   = {E_TF/1e9:.1f} GJ   (ref: 41 GJ)")
    print(f"  tau_dis = {tau_dis_TF:.1f} s    (ref: ~11 s)")
    
    # =========================================================================
    # ITER CS SYSTEM
    # =========================================================================
    # The CS consists of 6 independent modules, each with its own FDU.
    # Reference values [3,4]: E_mag = 6.4 GJ, tau_dis = 7.5 s, I = 45 kA
    print("\n--- ITER CS ---")
    # Geometry: thick solenoid with nearly uniform axial field
    B_max_CS = 13.0       # Peak field [T]
    r_in_CS = 1.3         # Inner winding pack radius [m]
    r_out_CS = 2.1        # Outer winding pack radius [m]
    H_CS = 12.0           # Total height [m]
    E_CS = calculate_E_mag_CS(B_max_CS, r_in_CS, r_out_CS, H_CS)
    E_CS_ref = 6.4e9
    # Discharge parameters [3]: 6 modules, tau_dis = 7.5 s
    # V_max derived from: V = 2*E / (I * tau_dis * N) ≈ 6.3 kV
    I_CS = 45e3
    V_max_CS = 6.3e3
    N_FDU_CS = 6
    tau_h_CS = 2.0
    t_dump_CS = calculate_t_dump(E_CS, I_CS, V_max_CS, N_FDU_CS, tau_h_CS)
    tau_dis_CS = 2 * E_CS / (I_CS * V_max_CS * N_FDU_CS)
    print(f"  E_mag   = {E_CS/1e9:.1f} GJ   (ref: 6.4 GJ)")
    print(f"  tau_dis = {tau_dis_CS:.1f} s    (ref: 7.5 s)")
    
#%% Print
        
if __name__ == "__main__":
    print("##################################################### J_cable Model ##########################################################")

#%% Final cable current density

# =============================================================================
# CABLE CURRENT DENSITY
# =============================================================================

def _compute_cable_current_density_core(
    sc_type,
    B_peak,
    T_op,
    E_mag,
    I_cond,
    V_max,
    N_sub,
    tau_h,
    f_He_pipe,
    f_void,
    f_In,
    T_hotspot,
    RRR,
    Marge_T_He,
    Marge_T_Nb3Sn,
    Marge_T_NbTi,
    Marge_T_REBCO,
    Eps,
    Tet,
    J_wost_Manual=None,
):
    """
    Core computation function for cable current density calculations.
    
    This function contains the actual physics calculations for determining
    the overall current density in superconducting cables, including the
    fraction of superconductor, copper, helium, and insulation.
    
    Parameters
    ----------
    sc_type : str
        Superconductor type ('Nb3Sn', 'NbTi', 'REBCO', 'REBCO_Fujikura2019', 'REBCO_SuperOx2019', 'REBCO_Fleiter2014', 'Manual'). 'REBCO' defaults to Fujikura_2019 (Senatore 2024).
    B_peak : float
        Peak magnetic field at conductor location [T]
    T_op : float
        Operating temperature [K]
    E_mag : float
        Magnetic energy stored in the coil system [J]
    I_cond : float
        Conductor current [A]
    V_max : float
        Maximum terminal voltage during quench [V]
    N_sub : int
        Number of subcables in the cable
    tau_h : float
         Detection + holding time before discharge [s]
    f_He : float
        Helium fraction in cable [-]
    f_In : float
        Insulation fraction in cable [-]
    T_hotspot : float
        Maximum allowable hotspot temperature [K]
    RRR : float
        Residual resistivity ratio of copper [-]
    Marge_T_He : float
        Temperature margin for helium cooling [K]
    Marge_T_Nb3Sn : float
        Temperature margin for Nb3Sn superconductor [K]
    Marge_T_NbTi : float
        Temperature margin for NbTi superconductor [K]
    Marge_T_REBCO : float
        Temperature margin for REBCO superconductor [K]
    Eps : float
        Strain for Nb3Sn superconductor [-]
    Tet : float
        Field angle for REBCO [rad]
    J_wost_Manual : float, optional
        Manual override for overall current density [A/m²]
        If provided, skips automatic calculation
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'J_non_Cu': Critical current density on SC area [A/m²]
        - 'J_wost': Overall current density in cable [A/m²]
        - 'f_sc': Superconductor fraction [-]
        - 'f_cu': Copper fraction [-]
        - 'f_He': Helium fraction [-]
        - 'f_In': Insulation fraction [-]
        - 't_dump': Effective discharge time [s]
    
    Notes
    -----
    This is the core computational function that should NOT be called directly.
    Use calculate_cable_current_density() instead, which handles caching.
    
    The function calculates the required current density based on:
    - Critical current density of the superconductor (field and temperature dependent)
    - Thermal stability requirements (hotspot temperature limit)
    - Quench protection constraints (maximum terminal voltage)
    
    "wost" = Without Steel. The steel jacket fraction is sized separately
    based on mechanical stress requirements (Lorentz forces, thermal stress).
    """
    
    # Guard against invalid inputs (NaN propagation prevention)
    if np.isnan(B_peak) or np.isnan(E_mag) or np.isnan(N_sub):
        f_active = 1.0 - f_In - f_He_pipe
        return {
            'J_non_Cu': 0,
            'J_wost': 0,
            'f_sc': 0,
            'f_cu': 0,
            'f_He_pipe': f_He_pipe,
            'f_void': f_void * f_active,
            'f_He': f_He_pipe + f_void * f_active,
            'f_In': f_In,
            't_dump': np.nan,
        }
    
    # 1. Superconductor critical current density
    if sc_type == "Manual":
        return {
            'J_non_Cu': np.nan,
            'J_wost': J_wost_Manual,
            'f_sc': np.nan,
            'f_cu': np.nan,
            'f_He_pipe': np.nan,
            'f_void': np.nan,
            'f_He': np.nan,
            'f_In': np.nan,
            't_dump': np.nan,
        }
    elif sc_type == "Nb3Sn":
        T_eff = T_op + Marge_T_He + Marge_T_Nb3Sn
        J_non_Cu = J_non_Cu_Nb3Sn(B_peak, T_eff, Eps)
        
    elif sc_type == "NbTi":
        T_eff = T_op + Marge_T_He + Marge_T_NbTi
        J_non_Cu = J_non_Cu_NbTi(B_peak, T_eff)
        
    elif sc_type == "REBCO":
        T_eff = T_op + Marge_T_He + Marge_T_REBCO
        J_non_Cu = J_non_Cu_REBCO(B_peak, T_eff, Tet, dataset='Fujikura_2019')
        
    elif sc_type == "REBCO_Fujikura2019":
        T_eff = T_op + Marge_T_He + Marge_T_REBCO
        J_non_Cu = J_non_Cu_REBCO(B_peak, T_eff, Tet, dataset='Fujikura_2019')
        
    elif sc_type == "REBCO_SuperOx2019":
        T_eff = T_op + Marge_T_He + Marge_T_REBCO
        J_non_Cu = J_non_Cu_REBCO(B_peak, T_eff, Tet, dataset='SuperOx_2019')
        
    elif sc_type == "REBCO_Fleiter2014":
        T_eff = T_op + Marge_T_He + Marge_T_REBCO
        J_non_Cu = J_non_Cu_REBCO(B_peak, T_eff, Tet, dataset='Fleiter_2014')
        
    else:
        raise ValueError(f"Unknown superconductor: {sc_type}")
    
    # Handle case where SC is beyond critical surface
    if J_non_Cu <= 0:
        f_active = 1.0 - f_In - f_He_pipe
        return {
            'J_non_Cu': 0,
            'J_wost': 0,
            'f_sc': 0,
            'f_cu': 0,
            'f_He_pipe': f_He_pipe,
            'f_void': f_void * f_active,
            'f_He': f_He_pipe + f_void * f_active,
            'f_In': f_In,
            't_dump': np.inf,
        }
    
    # 2. Effective discharge time
    t_dump = calculate_t_dump(E_mag, I_cond, V_max, N_sub, tau_h)
    
    # 3. Size all non-steel fractions (SC, Cu, He pipe, void, Ins)
    result = size_cable_fractions(
        J_non_Cu=J_non_Cu,
        B_peak=B_peak,
        T_op=T_op,
        t_dump=t_dump,
        T_hotspot=T_hotspot,
        f_He_pipe=f_He_pipe,
        f_void=f_void,
        f_In=f_In,
        RRR=RRR,
    )
    
    return {
        'J_non_Cu': J_non_Cu,
        'J_wost': result['J_wost'],
        'f_sc': result['f_sc'],
        'f_cu': result['f_cu'],
        'f_He_pipe': result['f_He_pipe'],
        'f_void': result['f_void'],
        'f_He': result['f_He'],
        'f_In': result['f_In'],
        't_dump': t_dump,
    }


@lru_cache(maxsize=2000)
def _cached_cable_current_density(
    sc_type: str,
    B_peak_rounded: float,
    T_op: float,
    E_mag_rounded: float,
    I_cond: float,
    V_max: float,
    N_sub: int,
    tau_h: float,
    f_He_pipe: float,
    f_void: float,
    f_In: float,
    T_hotspot: float,
    RRR: float,
    Marge_T_He: float,
    Marge_T_Nb3Sn: float,
    Marge_T_NbTi: float,
    Marge_T_REBCO: float,
    Eps: float,
    Tet: float,
    J_wost_Manual: float,
):
    """
    LRU-cached wrapper for cable current density calculations.
    
    This function provides memoization of expensive cable current density
    calculations. It uses rounded input parameters to improve cache hit rates
    while maintaining sufficient numerical accuracy for engineering design.
    
    Parameters
    ----------
    sc_type : str
        Superconductor type ('Nb3Sn', 'NbTi', 'REBCO', 'REBCO_Fujikura2019', 'REBCO_SuperOx2019', 'REBCO_Fleiter2014', 'Manual'). 'REBCO' defaults to Fujikura_2019 (Senatore 2024).
    B_peak_rounded : float
        Peak magnetic field rounded to 0.01 T precision [T]
    T_op : float
        Operating temperature [K]
    E_mag_rounded : float
        Magnetic energy rounded to 100 kJ precision [J]
    I_cond : float
        Conductor current [A]
    V_max : float
        Maximum terminal voltage during quench [V]
    N_sub : int
        Number of subcables in the cable
    tau_h : float
        Hydraulic time constant [s]
    f_He : float
        Helium fraction in cable [-]
    f_In : float
        Insulation fraction in cable [-]
    T_hotspot : float
        Maximum allowable hotspot temperature [K]
    RRR : float
        Residual resistivity ratio of copper [-]
    Marge_T_He : float
        Temperature margin for helium cooling [K]
    Marge_T_Nb3Sn : float
        Temperature margin for Nb3Sn superconductor [K]
    Marge_T_NbTi : float
        Temperature margin for NbTi superconductor [K]
    Marge_T_REBCO : float
        Temperature margin for REBCO superconductor [K]
    Eps : float
        Strain for Nb3Sn superconductor [-]
    Tet : float
        Field angle for REBCO [rad]
    J_wost_Manual : float
        Manual override for current density [A/m²]
        Use -1.0 to indicate no manual override
    
    Returns
    -------
    dict
        Dictionary containing cable current density results
        (see _compute_cable_current_density_core for details)
    
    Notes
    -----
    This function should NOT be called directly by users. Use the public
    calculate_cable_current_density() function instead.
    
    The cache uses LRU (Least Recently Used) eviction policy with a maximum
    of 2000 entries. This provides significant speedup for iterative design
    optimization where similar cable configurations are evaluated repeatedly.
    
    Cache key considerations:
    - All parameters must be hashable (hence conversion of J_wost_Manual)
    - Rounding is applied to continuous variables to increase cache hits
    - String and integer parameters are used directly
    """
    return _compute_cable_current_density_core(
        sc_type=sc_type,
        B_peak=B_peak_rounded,
        T_op=T_op,
        E_mag=E_mag_rounded,
        I_cond=I_cond,
        V_max=V_max,
        N_sub=N_sub,
        tau_h=tau_h,
        f_He_pipe=f_He_pipe,
        f_void=f_void,
        f_In=f_In,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet,
        J_wost_Manual=J_wost_Manual if J_wost_Manual > 0 else None,
    )


def calculate_cable_current_density(
    sc_type,
    B_peak,
    T_op,
    E_mag,
    I_cond,
    V_max,
    N_sub,
    tau_h,
    f_He_pipe,
    f_void,
    f_In,
    T_hotspot,
    RRR,
    Marge_T_He,
    Marge_T_Nb3Sn,
    Marge_T_NbTi,
    Marge_T_REBCO,
    Eps,
    Tet,
    J_wost_Manual=None,
):
    """
    Calculate current density on the non-steel part of the conductor.
    
    This function integrates:
    1. Superconductor J_non_Cu calculation (with technology-specific temperature margins)
    2. Discharge time from quench protection circuit
    3. Copper fraction sizing (Maddock criterion)
    4. Final composition and current density (excluding steel jacket)
    
    Conductor hierarchy (CICC-like model):
        - Strand = SC + Cu
        - Active zone = Strands + Interstitial void (f_void)
        - Non-steel (wost) = Insulation + He pipe + Active zone
        - Conductor = Non-steel + Steel jacket (steel sized separately)
    
    Parameters
    ----------
    sc_type : str
        Superconductor type: "NbTi", "Nb3Sn", "REBCO" (=Fujikura_2019),
        "REBCO_Fujikura2019", "REBCO_SuperOx2019", "REBCO_Fleiter2014",
        or "Manual"
    B_peak : float
        Peak magnetic field at conductor [T]
        Automatically rounded to 0.01 T for caching
    T_op : float
        Operating temperature [K]
    E_mag : float
        Total magnetic energy of the coil [J]
        Automatically rounded to 100 kJ for caching
    I_cond : float
        Operating current per conductor [A]
    V_max : float, optional
        Maximum voltage to ground during dump [V] (default: 10 kV)
    N_sub : int or float, optional
        Number of protection subdivisions (default: 6)
        Automatically converted to integer
    tau_h : float, optional
        Detection + delay time [s] (default: 2.0 s for LTS)
    f_He : float, optional
        Helium void fraction in cable (default: 0.30)
    f_In : float, optional
        Insulation fraction in non-steel area (default: 0.10)
    T_hotspot : float, optional
        Maximum hot-spot temperature [K] (default: 250 K)
    RRR : float, optional
        Copper residual resistivity ratio (default: 100)
    Marge_T_He : float, optional
        Helium temperature margin [K] (default: 0.0)
    Marge_T_Nb3Sn : float, optional
        Nb3Sn-specific temperature margin [K] (default: 1.5)
    Marge_T_NbTi : float, optional
        NbTi-specific temperature margin [K] (default: 1.0)
    Marge_T_REBCO : float, optional
        REBCO-specific temperature margin [K] (default: 2.0)
    Eps : float, optional
        Strain for Nb3Sn (default: -0.003)
    Tet : float, optional
        Field angle for REBCO [rad] (default: 0)
    J_wost_Manual : float, optional
        Manual override for overall current density [A/m²]
        If None, current density is calculated automatically
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'J_non_Cu': Critical current density on SC area [A/m²]
        - 'J_wost': Current density on non-steel area [A/m²]
        - 'f_sc': SC volume fraction (wost) [-]
        - 'f_cu': Cu volume fraction (wost) [-]
        - 'f_He': He volume fraction (wost) [-]
        - 'f_In': Insulation volume fraction (wost) [-]
        - 't_dump': Effective discharge time [s]
        
    Notes
    -----
    Parameter rounding strategy for caching:
    - B_peak: Rounded to 0.01 T (10 mT precision)
    - E_mag: Rounded to 100 kJ (0.1 MJ precision)
    - N_sub: Converted to integer
    
    These rounding levels balance cache efficiency with numerical accuracy
    requirements for tokamak magnet design.
    
    "wost" = Without Steel. The steel jacket fraction is sized separately
    based on mechanical stress requirements (Lorentz forces, thermal stress).
    
    The function returns zero current density if any critical input
    (B_peak, E_mag, N_sub) is NaN, preventing error propagation in
    iterative calculations.
    
    See Also
    --------
    clear_cable_cache : Clear the calculation cache
    get_cache_stats : Retrieve cache performance statistics
    """
    
    # Guard against invalid inputs (NaN propagation prevention)
    if np.isnan(B_peak) or np.isnan(E_mag) or np.isnan(N_sub):
        f_active = 1.0 - f_In - f_He_pipe
        return {
            'J_non_Cu': 0,
            'J_wost': 0,
            'f_sc': 0,
            'f_cu': 0,
            'f_He_pipe': f_He_pipe,
            'f_void': f_void * f_active,
            'f_He': f_He_pipe + f_void * f_active,
            'f_In': f_In,
            't_dump': np.nan,
        }
    
    # Round continuous parameters for improved cache hit rate.
    # NOTE: rounding introduces step discontinuities at bin boundaries.
    # B_peak: ±0.005 T bins → negligible impact (< 0.1% on Jc).
    # E_mag:  ±50 kJ bins → can cause J_wost jumps of ~1-2% at boundaries
    # (e.g. E_mag = 1.049 GJ vs 1.051 GJ map to different cache keys).
    # This is acceptable for design-point evaluations but may create
    # artefacts in fine-grained sensitivity scans over E_mag.
    B_peak_rounded = round(B_peak, 2)
    E_mag_rounded = round(E_mag / 1e5) * 1e5

    # Convert manual override to cache-compatible format
    J_wost_Manual_val = J_wost_Manual if J_wost_Manual is not None else -1.0
    
    # Call cached computation with rounded parameters
    return _cached_cable_current_density(
        sc_type=sc_type,
        B_peak_rounded=B_peak_rounded,
        T_op=T_op,
        E_mag_rounded=E_mag_rounded,
        I_cond=I_cond,
        V_max=V_max,
        N_sub=int(N_sub),
        tau_h=tau_h,
        f_He_pipe=f_He_pipe,
        f_void=f_void,
        f_In=f_In,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet,
        J_wost_Manual=J_wost_Manual_val,
    )


def clear_cable_cache():
    """
    Clear the cable current density calculation cache.
    
    This function should be called when global parameters or material
    properties have been modified, ensuring that subsequent calculations
    use updated values rather than stale cached results.
    
    Examples
    --------
    >>> # After changing material properties or global constants
    >>> clear_cable_cache()
    >>> # Next calculation will use fresh values
    
    Notes
    -----
    Typical scenarios requiring cache clearing:
    - Modification of superconductor critical surface parameterization
    - Changes to material property databases (Cu resistivity, He properties)
    - Updates to quench protection algorithms
    - Switching between different design standards or safety factors
    
    The cache will automatically rebuild as new calculations are performed.
    """
    _cached_cable_current_density.cache_clear()


def get_cache_stats():
    """
    Retrieve cache performance statistics.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'hits': Number of cache hits (reused calculations)
        - 'misses': Number of cache misses (new calculations required)
        - 'hit_rate': Cache hit rate as a fraction [0, 1]
    
    Notes
    -----
    A high hit rate (>80%) indicates effective cache utilization and
    significant computational savings. Low hit rates may suggest:
    - Parameter ranges too broad (consider tighter rounding)
    - Sequential rather than iterative calculations
    - Cache size too small (increase maxsize in @lru_cache)
    
    For optimization studies with >10,000 evaluations, cache hit rates
    above 95% are typical and can reduce computation time by 10-30x.
    """
    info = _cached_cable_current_density.cache_info()
    total = info.hits + info.misses
    return {
        'hits': info.hits,
        'misses': info.misses,
        'hit_rate': info.hits / total if total > 0 else 0
    }

#%% Without Steel current density test

"""
Validation benchmark for cable current density calculations.

Definitions
---------------
J_wost : Current density in winding pack without steel [A/m²]
f_sc   : Fraction of non-copper material (SC filaments + matrix/substrate + solder)
f_cu   : Fraction of copper (stabilizer + matrix + structural Cu)
f_He   : Fraction of helium (void)
f_In   : Fraction of insulation/gaps (cable level only)

References
----------
[1] JT-60SA TF Conductor
    - Sborchia et al., "The toroidal field coils for the JT-60SA tokamak",
      Fusion Engineering and Design 86 (2011) 572-579
    - JT-60SA Technical Design Report (2009)
    
[2] ITER TF Conductor
    - Mitchell et al., "The ITER magnet system", 
      Supercond. Sci. Technol. 25 (2012) 095004
    - ITER Design Description Document (DDD 1.1 - Magnet)
    
[3] SPARC CSMC (PIT VIPER cable)
    - Hartwig et al., "The SPARC Toroidal Field Model Coil Program" (Context on VIPER),
      IEEE Trans. Appl. Supercond. (2024)
    - CFS Press Release, "CFS Second Breakthrough... PIT VIPER", Oct 2024
    - Salazar et al., "Fiber optic quench detection... on VIPER cable",
      Supercond. Sci. Technol. 34 (2021)
    - SPARC CSMC Test parameters (nominal design values)
"""

if __name__ == "__main__":
    
    results = []
    
    # -- Unpack superconductor parameters from global configuration --
    T_hotspot     = cfg.T_hotspot
    RRR           = cfg.RRR
    Marge_T_He    = cfg.Marge_T_He
    Marge_T_Nb3Sn = cfg.Marge_T_Nb3Sn
    Marge_T_NbTi  = cfg.Marge_T_NbTi
    Marge_T_REBCO = cfg.Marge_T_REBCO
    Eps           = cfg.Eps
    Tet           = cfg.Tet
    
    # =========================================================================
    # JT-60SA TF (NbTi CICC) - [1]
    # =========================================================================
    # Conductor geometry [Sborchia 2011, Table 1]:
    #   - Rectangular jacket: 22 × 26 mm² (outer dimensions)
    #   - Wall thickness: 2 mm (316LN stainless steel)
    #   - Cable space: 18 × 22 mm²
    jacket_w = 22.0              # mm [Sborchia 2011]
    jacket_h = 26.0              # mm [Sborchia 2011]
    wall = 2.0                   # mm [Sborchia 2011]
    cable_w = jacket_w - 2*wall  # 18 mm
    cable_h = jacket_h - 2*wall  # 22 mm
    A_cable = cable_w * cable_h  # 396 mm²
    # Strand composition [Sborchia 2011, Table 2]:
    #   - Total: 486 strands
    #   - 324 NbTi strands (2/3) + 162 pure Cu strands (1/3)
    #   - Strand diameter: 0.81 mm
    #   - Cu:non-Cu ratio in NbTi strand: ~2.0
    n_NbTi = 324                 # [Sborchia 2011]
    n_Cu = 162                   # [Sborchia 2011]
    d_strand = 0.81              # mm [Sborchia 2011]
    Cu_nonCu = 2.0               # Typical value
    A_strand = np.pi * (d_strand/2)**2  # 0.515 mm²
    A_NbTi = n_NbTi * A_strand          # 166.9 mm²
    A_Cu_pure = n_Cu * A_strand         # 83.4 mm²
    f_NbTi = 1/(1+Cu_nonCu)             # 33.3% (NbTi in SC strand)
    A_nonCu = A_NbTi * f_NbTi           # 55.6 mm² (NbTi filaments)
    A_Cu_matrix = A_NbTi * (1-f_NbTi)   # 111.3 mm² (Cu in SC strand)
    A_Cu_tot = A_Cu_matrix + A_Cu_pure  # 194.7 mm²
    # Void fraction [Sborchia 2011]:
    void = 0.33                  # [Sborchia 2011, measured]
    A_He = void * A_cable        # 130.7 mm²
    # Operating parameters:
    I_op = 25.7e3                # A [Sborchia 2011]
    B_peak = 5.65                # T [JT-60SA TDR]
    J_wost_ref  = I_op / (A_cable * 1e-6)  # 64.9 MA/m²
    
    # JT-60SA: NbTi CICC — rectangular cable, no central pipe, LTS void = 0.33
    calc = calculate_cable_current_density(
            sc_type="NbTi",
            B_peak=B_peak,
            T_op=4.2,
            E_mag=1.06e9,
            I_cond=I_op,
            V_max=5e3,
            N_sub=3,
            tau_h=1.0,
            f_He_pipe=0.00,
            f_void=0.33,
            f_In=0.05,
            T_hotspot=T_hotspot,
            RRR=RRR,
            Marge_T_He=Marge_T_He,
            Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi,
            Marge_T_REBCO=Marge_T_REBCO,
            Eps=Eps,
            Tet=Tet
            )

    
    results.append({
        "name": "JT-60SA TF", "ref": "[1]", "sc": "NbTi", "B": B_peak,
        "J_ref": J_wost_ref, "f_sc_ref": A_nonCu/A_cable, "f_cu_ref": A_Cu_tot/A_cable,
        "f_He_ref": void, "f_In_ref": 1 - A_nonCu/A_cable - A_Cu_tot/A_cable - void,
        "J_calc": calc['J_wost'], "f_sc_calc": calc['f_sc'], "f_cu_calc": calc['f_cu'],
        "f_He_calc": calc['f_He'], "f_In_calc": calc['f_In'],
    })
    
    # =========================================================================
    # ITER TF (Nb3Sn CICC) - [2]
    # =========================================================================
    # Conductor geometry [Mitchell 2012]:
    #   - Circular jacket: Ø 43.7 mm
    #   - Wall thickness: 2 mm
    #   - Central cooling spiral: Ø 10 mm
    D_jacket = 43.7              # mm [Mitchell 2012]
    wall = 2.0                   # mm [Mitchell 2012]
    D_spiral = 10.0              # mm [Mitchell 2012]
    D_int = D_jacket - 2*wall    # 39.7 mm
    A_jacket = np.pi * (D_int/2)**2      # 1237 mm²
    A_spiral = np.pi * (D_spiral/2)**2   # 78.5 mm²
    A_cable = A_jacket - A_spiral        # 1159 mm² (cable space)
    A_wost = A_jacket                    # 1237 mm²
    # Strand composition [Mitchell 2012]:
    #   - 900 Nb3Sn strands + 522 pure Cu strands
    #   - Strand diameter: 0.82 mm
    #   - Cu:non-Cu ratio: 1.0
    n_Nb3Sn = 900                # [Mitchell 2012]
    n_Cu = 522                   # [Mitchell 2012]
    d_strand = 0.82              # mm [Mitchell 2012]
    Cu_nonCu = 1.0               # [ITER DDD]
    A_strand = np.pi * (d_strand/2)**2  # 0.528 mm²
    A_Nb3Sn = n_Nb3Sn * A_strand        # 475.3 mm²
    A_Cu_pure = n_Cu * A_strand         # 275.7 mm²
    f_Nb3Sn = 1/(1+Cu_nonCu)            # 50%
    A_nonCu = A_Nb3Sn * f_Nb3Sn         # 237.6 mm²
    A_Cu_matrix = A_Nb3Sn * (1-f_Nb3Sn) # 237.6 mm²
    A_Cu_tot = A_Cu_matrix + A_Cu_pure  # 513.3 mm²
    # Void fraction:
    void = 0.33                  # [Mitchell 2012]
    A_He = void * A_cable        # 382.5 mm² (interstitial)
    # He pipe (central spiral) fraction in wost:
    f_He_pipe_ITER = A_spiral / A_wost   # ~0.063
    # Operating parameters:
    I_op = 68.0e3                # A [Mitchell 2012]
    B_peak = 11.8                # T [ITER DDD]
    J_wost_ref  = I_op / (A_wost * 1e-6)  # 54.9 MA/m²
    
    # ITER TF: Nb3Sn CICC — central spiral pipe + LTS interstitial void
    calc = calculate_cable_current_density(
        sc_type="Nb3Sn",
        B_peak=B_peak,
        T_op=4.2,
        E_mag=41e9,
        I_cond=I_op,
        V_max=10e3,
        N_sub=9,
        tau_h=5.0,
        f_He_pipe=f_He_pipe_ITER,
        f_void=0.33,
        f_In=0.05,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet
    )
    
    # Reference He: central spiral + interstitial void (total)
    f_He_ref_ITER = (A_spiral + A_He) / A_wost
    results.append({
        "name": "ITER TF", "ref": "[2]", "sc": "Nb3Sn", "B": B_peak,
        "J_ref": J_wost_ref, "f_sc_ref": A_nonCu/A_wost, "f_cu_ref": A_Cu_tot/A_wost,
        "f_He_ref": f_He_ref_ITER,
        "f_In_ref": 1 - A_nonCu/A_wost - A_Cu_tot/A_wost - f_He_ref_ITER,
        "J_calc": calc['J_wost'], "f_sc_calc": calc['f_sc'], "f_cu_calc": calc['f_cu'],
        "f_He_calc": calc['f_He'], "f_In_calc": calc['f_In'],
    })
    
    # =========================================================================
    # SPARC CS (PIT VIPER REBCO) - [3]
    # =========================================================================
    # Geometry:
    #   - Copper Jacket OD: ~26 mm
    #   - Turn Insulation: Essential for CS stack. Assumed 0.5 mm (fiberglass tape).
    #   - NOTE: This creates a ~5-8% insulation fraction area.
    D_core_out = 26.0            # mm (Conductor OD)
    t_ins = 0.5                  # mm (Turn insulation)
    D_wost = D_core_out + 2*t_ins # 27.0 mm
    A_wost = np.pi * (D_wost/2)**2 # 572.5 mm²
    # Internal components (same as before)
    D_core = 22.0
    D_cool = 6.0
    A_core = np.pi * (D_core/2)**2
    A_cool = np.pi * (D_cool/2)**2
    A_jacket_cu = np.pi/4 * (D_core_out**2 - D_core**2)
    # Tape Stack (PIT VIPER 50kA Class)
    n_tapes = 120
    w_tape = 4.0
    t_tape = 0.100
    A_tapes = n_tapes * (w_tape * t_tape) # 48.0 mm²
    # Fractions within tape
    f_nonCu_tape = 0.55
    f_Cu_tape = 0.45
    A_nonCu_tape = A_tapes * f_nonCu_tape
    A_Cu_tape = A_tapes * f_Cu_tape
    # Channels & Solder
    A_channels = 160.0
    A_solder = A_channels - A_tapes
    # Aggregates
    A_Cu_core = A_core - A_cool - A_channels
    A_Cu_tot = A_Cu_core + A_jacket_cu + A_Cu_tape
    A_nonCu_tot = A_nonCu_tape + A_solder # Solder + SC/Substrate
    A_He = A_cool # Central channel
    # Operating Parameters (Design Values)
    # Using the FULL SPARC CS design requirement, not just the model coil limit.
    I_op = 50.0e3    # A
    B_peak = 24.0    # T [Design requirement for SPARC CS]
    J_wost_ref  = I_op / (A_wost * 1e-6) # 87.3 MA/m²
    
    # SPARC CS: REBCO — central cooling pipe only, no interstitial void (HTS)
    # sc_type="REBCO" → Fujikura_2019 (Senatore 2024), consistent with SPARC tape choice
    calc = calculate_cable_current_density(
        sc_type="REBCO",
        B_peak=B_peak,
        T_op=20.0,
        E_mag=0.5e9,
        I_cond=I_op,
        V_max=5e3,
        N_sub=6,
        tau_h=10,
        f_He_pipe=0.05,
        f_void=0.00,
        f_In= 0.07,
        T_hotspot=T_hotspot,
        RRR=RRR,
        Marge_T_He=Marge_T_He,
        Marge_T_Nb3Sn=Marge_T_Nb3Sn,
        Marge_T_NbTi=Marge_T_NbTi,
        Marge_T_REBCO=Marge_T_REBCO,
        Eps=Eps,
        Tet=Tet
    )
    
    results.append({
        "name": "SPARC CS", "ref": "[3]", "sc": "REBCO", "B": B_peak,
        "J_ref": J_wost_ref, "f_sc_ref": A_nonCu_tot/A_wost, "f_cu_ref": A_Cu_tot/A_wost,
        "f_He_ref": A_He/A_wost, "f_In_ref": 1 - (A_nonCu_tot + A_Cu_tot + A_He)/A_wost,
        "J_calc": calc['J_wost'], "f_sc_calc": calc['f_sc'], "f_cu_calc": calc['f_cu'],
        "f_He_calc": calc['f_He'], "f_In_calc": calc['f_In'],
    })
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*90)
    print("CABLE CURRENT DENSITY BENCHMARK")
    print("="*90)
    print(f"\n{'Machine':<14} {'Type':<6} {'SC':<7} {'B[T]':<6} {'J_wost[MA/m²]':<14} "
          f"{'f_sc[%]':<9} {'f_cu[%]':<9} {'f_He[%]':<9} {'f_In[%]':<9}")
    print("-"*90)
    
    for r in results:
        # Ligne référence
        print(f"{r['name']:<14} {'Ref':<6} {r['sc']:<7} {r['B']:<6.2f} "
              f"{r['J_ref']/1e6:<14.1f} "
              f"{r['f_sc_ref']*100:<9.1f} {r['f_cu_ref']*100:<9.1f} "
              f"{r['f_He_ref']*100:<9.1f} {r['f_In_ref']*100:<9.1f}")
        # Ligne calculée
        print(f"{'':<14} {'Calc':<6} {'':<7} {'':<6} "
              f"{r['J_calc']/1e6:<14.1f} "
              f"{r['f_sc_calc']*100:<9.1f} {r['f_cu_calc']*100:<9.1f} "
              f"{r['f_He_calc']*100:<9.1f} {r['f_In_calc']*100:<9.1f}")
        print("-"*90)

if __name__ == "__main__":
    # Worst-case cable current density vs field under quench protection
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_cable_current_density(cfg=cfg)

#%% Print
        
if __name__ == "__main__":
    print("##################################################### CIRCE Model ##########################################################")
    
#%% CIRCE 0D module

"""
F_CIRCE0D - Analytical stress solver for multilayer thick cylinders
Developed by B.Boudes
Addapted by T.Auclair
====================================================================

This module implements the analytical solution for the elasticity problem of
concentric cylinder stacks subjected to:
- Internal and external pressures (Pi, Pe)
- Electromagnetic body loads (J × B) with linear radial profile

Theory
------
For each layer, the Lorentz body force is modeled as:
    f_r(r) = J(r) × B(r) ≈ K1·r + K2

where K1 and K2 depend on the field configuration (increasing/decreasing).

The axisymmetric equilibrium equation:
    dσr/dr + (σr - σθ)/r + f_r = 0

is solved analytically with Hooke's law, yielding expressions for σr, σθ, 
and ur as functions of integration constants.

Boundary conditions are:
- σr(ri) = -Pi  (internal pressure)
- σr(re) = -Pe  (external pressure)
- Continuity of ur at interfaces

The code handles multilayer structures where each layer can have different:
- Young's modulus E (e.g., conductor vs structural steel)
- Current density J (can be zero for passive layers)
- Magnetic field B
- Field profile configuration

References
----------
- Timoshenko & Goodier, "Theory of Elasticity"
- CIRCE B.Boudes (CEA)

"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class LayerResult:
    """Results for a single layer."""
    r: np.ndarray          # Radial positions [m]
    sigma_r: np.ndarray    # Radial stress [Pa]
    sigma_t: np.ndarray    # Hoop stress [Pa]
    u_r: np.ndarray        # Radial displacement [m]


def compute_body_load_coefficients(
    J: float, 
    B: float, 
    r_inner: float, 
    r_outer: float, 
    config: int
) -> Tuple[float, float]:
    """
    Compute K1 and K2 coefficients for the linear body load profile.

    The loading function f(r) = K1·r + K2 enters the equilibrium equation as:
        dσr/dr + (σr - σθ)/r = f(r)
    i.e. f(r) appears on the RHS.  The physical (outward) Lorentz body force
    is b_r(r) = -f(r).

    Parameters
    ----------
    J : float
        Average current density in the layer [A/m²]
    B : float
        Characteristic magnetic field [T]
    r_inner : float
        Inner radius of the layer [m]
    r_outer : float
        Outer radius of the layer [m]
    config : int
        Field profile configuration (see Notes for sign details):
        - 1 : f(r) = J·B·(r - r_inner)/dr  → net outward body force b_r
              is INWARD (compressive loading, max at r_outer).
        - else (0 or 2) : f(r) = J·B·(r - r_outer)/dr  → net outward body
              force b_r is OUTWARD (expansive loading, max at r_inner).
              This is the standard CS solenoid profile (J × B pushes outward).

    Returns
    -------
    K1, K2 : float
        Linear profile coefficients for f(r) = K1·r + K2

    Notes
    -----
    For an infinite solenoid with uniform J, B(r) is linear.

    Config 1:
        f(r) = J·B·(r - r_inner)/dr   → f = 0 at r_inner, J·B at r_outer
        b_r  = -f  → max inward force at r_outer.

    Config 2 (or 0, default):
        f(r) = J·B·(r - r_outer)/dr   → f = -J·B at r_inner, 0 at r_outer
        b_r  = -f  → max outward force J·B at r_inner, zero at r_outer.
        Physically: the CS winding is pushed outward by J × B where B is
        strongest (inner radius).  This is the correct loading for a
        solenoid with B decreasing outward.

    For passive layers (J=0 or B=0), both K1 and K2 are zero.
    """
    dr = r_outer - r_inner

    # Handle passive layers (no electromagnetic load)
    if J == 0 or B == 0 or dr == 0:
        return 0.0, 0.0

    K1 = J * B / dr

    if config == 1:
        # f(r) zero at r_inner, positive at r_outer → inward body force
        K2 = -J * B * r_inner / dr
    else:
        # f(r) negative at r_inner, zero at r_outer → outward body force (CS)
        K2 = -J * B * r_outer / dr
    
    return K1, K2


def _build_continuity_rhs_core(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    config: List[int]
) -> float:
    """
    Core body-load and jump terms of the radial displacement continuity equation.

    Computes: term_ext - term_int + jump_term, which is common to all interface
    types (first, middle, last).  Boundary pressure contributions are added
    by the calling function.

    Parameters
    ----------
    ri, r0, re : float
        Inner, interface, and outer radii of the two adjacent layers [m].
    E_prev, E_curr : float
        Young's modulus of inner and outer layer [Pa].
    nu : float
        Poisson's ratio (assumed equal in both layers) [-].
    J_prev, B_prev, J_curr, B_curr : float
        Current density [A/m²] and magnetic field [T] in each layer.
    config : List[int]
        Configuration flags for body-load computation in each layer.
    """
    K1_prev, K2_prev = compute_body_load_coefficients(J_prev, B_prev, ri, r0, config[0])
    K1_curr, K2_curr = compute_body_load_coefficients(J_curr, B_curr, r0, re, config[1])

    # Contribution from outer layer
    term_ext = (
        (1 + nu) / E_curr * re**2 * (K1_curr * (nu + 3) / 8 + K2_curr * (nu + 2) / (3 * (re + r0))) +
        (1 - nu) / E_curr * (
            (K2_curr * (nu + 2) / 3) * (re**2 + r0**2 + re * r0) / (re + r0) +
            (re**2 + r0**2) * K1_curr * (nu + 3) / 8
        )
    )

    # Contribution from inner layer
    term_int = (
        (1 + nu) / E_prev * ri**2 * (K1_prev * (nu + 3) / 8 + K2_prev * (nu + 2) / (3 * (r0 + ri))) +
        (1 - nu) / E_prev * (
            (K2_prev * (nu + 2) / 3) * (r0**2 + ri**2 + r0 * ri) / (r0 + ri) +
            (r0**2 + ri**2) * K1_prev * (nu + 3) / 8
        )
    )

    # Jump terms at interface (discontinuity in material properties)
    jump_term = (
        (1 - nu**2) / 8 * r0**2 * (K1_prev / E_prev - K1_curr / E_curr) +
        (1 - nu**2) / 3 * r0 * (K2_prev / E_prev - K2_curr / E_curr)
    )

    return term_ext - term_int + jump_term


def _build_continuity_rhs_first(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    Pi: float, config: List[int]
) -> float:
    """
    RHS of the continuity equation for the first interface.

    Adds internal pressure Pi contribution to the common core terms.
    """
    rhs = _build_continuity_rhs_core(ri, r0, re, E_prev, E_curr, nu,
                                      J_prev, B_prev, J_curr, B_curr, config)
    # Internal pressure term: Pi acts on the inner surface of the first layer
    pressure_term = -Pi * (-2 * ri**2) / (E_prev * (r0**2 - ri**2))
    return rhs + pressure_term


def _build_continuity_rhs_last(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    Pe: float, config: List[int]
) -> float:
    """
    RHS of the continuity equation for the last interface.

    Adds external pressure Pe contribution to the common core terms.
    """
    rhs = _build_continuity_rhs_core(ri, r0, re, E_prev, E_curr, nu,
                                      J_prev, B_prev, J_curr, B_curr, config)
    # External pressure term: Pe acts on the outer surface of the last layer
    pressure_term = -Pe * (-2 * re**2) / (E_curr * (re**2 - r0**2))
    return rhs + pressure_term


def _build_continuity_rhs_middle(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float,
    J_prev: float, B_prev: float, J_curr: float, B_curr: float,
    config: List[int]
) -> float:
    """
    RHS of the continuity equation for internal interfaces.

    No pressure terms (pressures are at boundaries only).
    """
    return _build_continuity_rhs_core(ri, r0, re, E_prev, E_curr, nu,
                                       J_prev, B_prev, J_curr, B_curr, config)


def _build_stiffness_row_first(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float
) -> List[float]:
    """
    Build the first row of the stiffness matrix.
    
    Coefficients for [P1, P2] (P0 = Pi is not a variable).
    """
    # Diagonal coefficient (continuity at interface 1)
    diag = (
        ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
        ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
    )
    
    # Off-diagonal coefficient (coupling with P2)
    off_diag = (-2 * re**2) / (E_curr * (re**2 - r0**2))
    
    return [diag, off_diag]


def _build_stiffness_row_last(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float
) -> List[float]:
    """
    Build the last row of the stiffness matrix.
    
    Coefficients for [P_{n-2}, P_{n-1}] (P_n = Pe is not a variable).
    """
    # Off-diagonal coefficient (coupling with P_{n-2})
    off_diag = (-2 * ri**2) / (E_prev * (r0**2 - ri**2))
    
    # Diagonal coefficient
    diag = (
        ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
        ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
    )
    
    return [off_diag, diag]


def _build_stiffness_row_middle(
    ri: float, r0: float, re: float,
    E_prev: float, E_curr: float, nu: float
) -> List[float]:
    """
    Build an intermediate row of the stiffness matrix.
    
    Tridiagonal coefficients for [P_{i-1}, P_i, P_{i+1}].
    """
    # Sub-diagonal coefficient
    sub_diag = (-2 * ri**2) / (E_prev * (r0**2 - ri**2))
    
    # Diagonal coefficient
    diag = (
        ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
        ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
    )
    
    # Super-diagonal coefficient
    super_diag = (-2 * re**2) / (E_curr * (re**2 - r0**2))
    
    return [sub_diag, diag, super_diag]


def compute_layer_stresses(
    r: np.ndarray,
    ri: float, re: float,
    P_inner: float, P_outer: float,
    K1: float, K2: float,
    E: float, nu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stresses and displacements within a single layer.
    
    Parameters
    ----------
    r : np.ndarray
        Radial positions [m]
    ri, re : float
        Inner and outer radii of the layer [m]
    P_inner, P_outer : float
        Pressures at inner and outer interfaces [Pa]
    K1, K2 : float
        Body load profile coefficients
    E : float
        Young's modulus [Pa]
    nu : float
        Poisson's ratio [-]
        
    Returns
    -------
    sigma_r, sigma_t, u_r : np.ndarray
        Radial stress, hoop stress [Pa] and radial displacement [m]
        
    Notes
    -----
    Generalized Lamé solution with linear body load.
    For passive layers (K1=K2=0), this reduces to the classical Lamé solution.
    """
    # Integration constants (modified Lamé equations)
    C1 = (
        K1 * (nu + 3) / 8 + 
        K2 * (nu + 2) / (3 * (re + ri)) - 
        (P_inner - P_outer) / (re**2 - ri**2)
    )
    
    C2 = (
        (K2 * (nu + 2) / 3) * (re**2 + ri**2 + re * ri) / (re + ri) +
        (P_outer * re**2 - P_inner * ri**2) / (re**2 - ri**2) +
        (re**2 + ri**2) * K1 * (nu + 3) / 8
    )
    
    # Radial stress: σr = A/r² + B·r² + C·r + D
    sigma_r = (
        re**2 * ri**2 / r**2 * C1 + 
        K1 * (nu + 3) / 8 * r**2 + 
        K2 * (nu + 2) / 3 * r - 
        C2
    )
    
    # Hoop stress: σθ = -A/r² + B'·r² + C'·r + D
    sigma_t = (
        -re**2 * ri**2 / r**2 * C1 + 
        K1 * (3 * nu + 1) / 8 * r**2 + 
        K2 * (2 * nu + 1) / 3 * r - 
        C2
    )
    
    # Radial displacement (generalized plane strain Hooke's law)
    u_r = r / E * (
        -re**2 * ri**2 / r**2 * C1 * (1 + nu) + 
        (1 - nu**2) / 8 * K1 * r**2 +
        (1 - nu**2) / 3 * K2 * r - 
        C2 * (1 - nu)
    )
    
    return sigma_r, sigma_t, u_r


def F_CIRCE0D(
    n_points: int,
    R: List[float],
    J: List[float],
    B: List[float],
    Pi: float,
    Pe: float,
    E: List[float],
    nu: float,
    config: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main stress solver for multilayer cylinders.
    
    Solves the axisymmetric elasticity problem for a stack of n concentric
    cylinders subjected to pressures and electromagnetic body loads (J × B).
    
    This function supports heterogeneous structures where:
    - Active layers (conductors) have J > 0 and experience Lorentz forces
    - Passive layers (structural) have J = 0 and only transmit stresses
    
    Parameters
    ----------
    n_points : int
        Number of radial discretization points per layer
    R : List[float]
        Interface radii [m], length (n_layers + 1)
        R = [r0, r1, r2, ..., rn] where r0 is the inner radius
    J : List[float]
        Current densities per layer [A/m²]
        Use J=0 for passive/structural layers
    B : List[float]
        Characteristic magnetic fields per layer [T]
        For passive layers, this value is irrelevant (multiplied by J=0)
    Pi : float
        Internal pressure (at r = R[0]) [Pa]
    Pe : float
        External pressure (at r = R[-1]) [Pa]
    E : List[float]
        Young's moduli per layer [Pa]
        Typical values: ~50 GPa for smeared conductor, ~200 GPa for steel
    nu : float
        Poisson's ratio (assumed identical for all layers) [-]
    config : List[int]
        Field profile configuration per layer:
        - 1 : Decreasing field (max at r_inner)
        - 2 : Increasing field (max at r_outer)
        For passive layers, this value is irrelevant
        
    Returns
    -------
    sigma_r : np.ndarray
        Radial stress over the entire domain [Pa]
    sigma_t : np.ndarray
        Hoop stress over the entire domain [Pa]
    u_r : np.ndarray
        Radial displacement over the entire domain [m]
    r_vec : np.ndarray
        Corresponding radial positions [m]
    P : np.ndarray
        Pressures at interfaces (including Pi and Pe) [Pa]
        
    Raises
    ------
    ValueError
        If input dimensions are inconsistent
        
    Examples
    --------
    >>> # Two-layer case: conductor + steel jacket
    >>> R = [1.0, 1.5, 1.7]           # radii [m]
    >>> J = [50e6, 0.0]               # current density: conductor, passive steel
    >>> B = [13.0, 0.0]               # magnetic field [T]
    >>> E = [50e9, 200e9]             # Young's modulus: smeared conductor, steel
    >>> sigma_r, sigma_t, u_r, r, P = F_CIRCE0D(50, R, J, B, 0, 0, E, 0.3, [1, 1])
    """
    n_layers = len(E)
    
    # Input validation
    if len(R) != n_layers + 1:
        raise ValueError(f"R must have {n_layers + 1} elements, got {len(R)}")
    if len(J) != n_layers or len(B) != n_layers or len(config) != n_layers:
        raise ValueError("J, B, and config must have the same number of elements as E")
    
    # --- Solve for interface pressures ---
    
    if n_layers == 1:
        # Trivial case: single layer, no internal interfaces
        P = np.array([Pi, Pe])
        
    elif n_layers == 2:
        # Two-layer case: single unknown pressure P1
        # Scalar system: MG * P1 = MD
        
        ri, r0, re = R[0], R[1], R[2]
        E_prev, E_curr = E[0], E[1]
        
        # Stiffness matrix (scalar)
        MG = (
            ((1 + nu) * re**2 + (1 - nu) * r0**2) / (E_curr * (re**2 - r0**2)) +
            ((1 + nu) * ri**2 + (1 - nu) * r0**2) / (E_prev * (r0**2 - ri**2))
        )
        
        # Right-hand side (body loads + boundary pressures)
        MD = _build_continuity_rhs_first(
            ri, r0, re, E_prev, E_curr, nu,
            J[0], B[0], J[1], B[1], Pi, [config[0], config[1]]
        )
        MD += -Pe * (-2 * re**2) / (E_curr * (re**2 - r0**2))
        
        P1 = MD / MG
        P = np.array([Pi, P1, Pe])
        
    else:
        # General case: n_layers - 1 unknown pressures
        # Tridiagonal matrix system: MG @ P_vec = MD
        
        n_unknowns = n_layers - 1
        MG = np.zeros((n_unknowns, n_unknowns))
        MD = np.zeros(n_unknowns)
        
        for i in range(n_unknowns):
            # Interface i+1 is between layer i and layer i+1
            ri = R[i]
            r0 = R[i + 1]
            re = R[i + 2]
            E_prev = E[i]
            E_curr = E[i + 1]
            
            if i == 0:
                # First interface
                row = _build_stiffness_row_first(ri, r0, re, E_prev, E_curr, nu)
                MG[i, i:i+2] = row
                MD[i] = _build_continuity_rhs_first(
                    ri, r0, re, E_prev, E_curr, nu,
                    J[i], B[i], J[i+1], B[i+1], Pi,
                    [config[i], config[i+1]]
                )
                
            elif i == n_unknowns - 1:
                # Last interface
                row = _build_stiffness_row_last(ri, r0, re, E_prev, E_curr, nu)
                MG[i, i-1:i+1] = row
                MD[i] = _build_continuity_rhs_last(
                    ri, r0, re, E_prev, E_curr, nu,
                    J[i], B[i], J[i+1], B[i+1], Pe,
                    [config[i], config[i+1]]
                )
                
            else:
                # Intermediate interface
                row = _build_stiffness_row_middle(ri, r0, re, E_prev, E_curr, nu)
                MG[i, i-1:i+2] = row
                MD[i] = _build_continuity_rhs_middle(
                    ri, r0, re, E_prev, E_curr, nu,
                    J[i], B[i], J[i+1], B[i+1],
                    [config[i], config[i+1]]
                )
        
        # Solve linear system
        P_internal = np.linalg.solve(MG, MD)
        P = np.concatenate([[Pi], P_internal, [Pe]])
    
    # --- Compute stresses and displacements per layer ---
    
    sigma_r_list = []
    sigma_t_list = []
    u_r_list = []
    r_list = []
    
    for i in range(n_layers):
        ri = R[i]
        re = R[i + 1]
        
        K1, K2 = compute_body_load_coefficients(J[i], B[i], ri, re, config[i])
        
        r = np.linspace(ri, re, n_points)
        
        sigma_r, sigma_t, u_r = compute_layer_stresses(
            r, ri, re, P[i], P[i+1], K1, K2, E[i], nu
        )
        
        sigma_r_list.append(sigma_r)
        sigma_t_list.append(sigma_t)
        u_r_list.append(u_r)
        r_list.append(r)
    
    # Concatenate results
    sigma_r_total = np.concatenate(sigma_r_list)
    sigma_t_total = np.concatenate(sigma_t_list)
    u_r_total = np.concatenate(u_r_list)
    r_vec = np.concatenate(r_list)
    
    return sigma_r_total, sigma_t_total, u_r_total, r_vec, P


def compute_von_mises_stress(sigma_r: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    """
    Compute von Mises equivalent stress in axisymmetric conditions.
    
    For an axisymmetric stress state (σr, σθ, σz=0):
        σ_VM = sqrt(σr² + σθ² - σr·σθ)
    
    Parameters
    ----------
    sigma_r : np.ndarray
        Radial stress [Pa]
    sigma_t : np.ndarray
        Hoop stress [Pa]
        
    Returns
    -------
    sigma_vm : np.ndarray
        von Mises stress [Pa]
    """
    return np.sqrt(sigma_r**2 + sigma_t**2 - sigma_r * sigma_t)


def compute_tresca_stress(sigma_r: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    """
    Compute Tresca equivalent stress in axisymmetric conditions.
    
    Parameters
    ----------
    sigma_r : np.ndarray
        Radial stress [Pa]
    sigma_t : np.ndarray
        Hoop stress [Pa]
        
    Returns
    -------
    sigma_tresca : np.ndarray
        Tresca stress [Pa]
    """
    sigma_z = np.zeros_like(sigma_r)  # Zero axial stress
    
    # Principal stress differences
    diff1 = np.abs(sigma_r - sigma_t)
    diff2 = np.abs(sigma_t - sigma_z)
    diff3 = np.abs(sigma_z - sigma_r)
    
    return np.maximum(np.maximum(diff1, diff2), diff3)


#%% TEST CASES CIRCE

if __name__ == "__main__":
    # CIRCE0D stress solver validation: Lame, CS 3-layer, composite
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_CIRCE_stress_validation()

#%% print

if __name__ == "__main__":
    print("##################################################### TF Model ##########################################################")
    
#%% Academic model

def f_TF_academic(a, b, R0, σ_TF, J_max_TF, B_max_TF, Choice_Buck_Wedg,
                  coef_inboard_tension, F_CClamp):
    """
    Calculate the thickness of the TF coil using a 2-layer thin cylinder model.

    Layer 1 (WP):   conductor annulus sized by NI / J_max_TF.
    Layer 2 (Nose): steel casing sized by Tresca (radial pressure or hoop).

    The vertical tension at the midplane is obtained from the Maxwell stress
    integral through the toroidal bore:
        T_separating = π B0² R0² / μ0 × ln(R2/R1)

    Parameters
    ----------
    a : float
        Minor radius [m].
    b : float
        First Wall + Breeding Blanket + Neutron Shield + Gaps [m].
    R0 : float
        Major radius [m].
    σ_TF : float
        Yield strength of the TF steel [Pa].
    J_max_TF : float
        Maximum current density of the chosen Supra + Cu + He [A/m²].
    B_max_TF : float
        Maximum magnetic field at TF inner conductor [T].
    Choice_Buck_Wedg : str
        Mechanical option: 'Bucking', 'Plug', or 'Wedging'.
    coef_inboard_tension : float
        Fraction of total tension carried by inboard leg [-].
    F_CClamp : float
        Clamping force subtracted from tension [N].

    Returns
    -------
    c : float
        Total TF inboard radial thickness (WP + Nose) [m].
    c_WP : float
        Winding pack thickness [m].
    c_Nose : float
        Structural nose thickness [m].
    σ_z : float
        Axial (vertical) stress in the nose [Pa].
    σ_theta : float
        Hoop stress in the nose [Pa].
    σ_r : float
        Radial stress in the nose [Pa].
    Steel_fraction : float
        Nose / total thickness ratio [-].
    """
    
    def f_B0(Bmax, a, b, R0):
        """
        
        Estimate the magnetic field in the centre of the plasma
        
        Parameters
        ----------
        Bmax : The magnetic field at the inboard of the Toroidal Field (TF) coil [T]
        a : Minor radius [m]
        b : Thickness of the First Wall+ the Breeding Blanket+ The Neutron shield+ The Vacuum Vessel + Gaps [m]
        R0 : Major radius [m]

        Returns
        -------
        B0 : The estimated central magnetic field [T]
        
        """
        B0 = Bmax*(1-((a+b)/R0))
        return B0

    # 1. Calculate the central magnetic field B0 based on geometry and maximum field
    B0 = f_B0(B_max_TF, a, b, R0)

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

    # 7. Vertical separating force from Maxwell stress integral through midplane:
    #    T_sep = π B0² R0² / μ0 × ln(R2/R1),  corrected by clamping and inboard fraction.
    if (R2 > 0) and (R1 > 0) and (R2 / R1 > 0):
        T = abs(((np.pi * B0**2 * R0**2) / μ0 * math.log(R2 / R1) - F_CClamp) * coef_inboard_tension)
    else:
        # Invalid geometric conditions
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 8. Radial pressure P due to the magnetic field B_max_TF
    P = B_max_TF**2 / (2 * μ0)

    # 9. Mechanical option choice: "bucking" or "wedging"
    if Choice_Buck_Wedg in ("Bucking", "Plug"):
        # Thickness c2 for bucking, valid if R1 >> c
        c_Nose = (B0**2 * R0**2) * math.log(R2 / R1) / (2 * μ0 * 2 * R1 * (σ_TF - P))
        σ_r = P  # Radial stress
        σ_z = T / (2 * np.pi * R1 * c_Nose)  # Axial stress
        σ_theta = 0

    elif Choice_Buck_Wedg == "Wedging":
        # Thickness c2 for wedging, valid if R1 >> c
        c_Nose = (B0**2 * R0**2) / (2 * μ0 * R1 * σ_TF) * (1 + math.log(R2 / R1) / 2)
        σ_theta = P * R1 / c_Nose  # Circumferential stress
        σ_z = T / (2 * np.pi * R1 * c_Nose)  # Axial stress
        σ_r = 0
        
    else:
        raise ValueError("Choose 'Bucking' or 'Wedging' as the mechanical option.")

    # 10. Total thickness c (sum of the two layers)
    c = c_WP + c_Nose

    # Verify that c_WP is valid
    if c is None or np.isnan(c) or c < 0 or c > (c_WP + c_Nose) or c > R0 - a - b:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    Steel_fraction = c_Nose / c

    return c, c_WP, c_Nose, σ_z, σ_theta, σ_r, Steel_fraction

    
#%% D0FUS model

def Winding_Pack_D0FUS(R_0, a, b, sigma_max, J_max, B_max, omega, n,
                       grading=False):
    
    """
    Computes the winding pack thickness and stress ratio under Tresca criterion.
    
    When grading=False (default):
        Uniform α. Log-spaced adaptive bracket search + brentq root-finding.
        Tresca is satisfied at the most loaded point only.
    
    When grading=True:
        Radially varying α(R). Tresca is saturated at every radius.
        Uses inward radial integration with self-consistent σ_z (Picard).
        Typically reduces c_WP by 8 to 25% depending on B_max.
    
    Args:
        R_0: Major radius [m]
        a: Plasma minor radius [m]
        b: Radial build from plasma edge to TF inner face [m]
        sigma_max: Maximum allowable Tresca stress [Pa]
        J_max: Maximum engineering current density [A/m²]
        B_max: Peak magnetic field [T]
        omega: Scaling factor for axial load [dimensionless].
            Fraction of total vertical tension borne by the inboard leg.
            Typical: 0.4-0.6 depending on coil shape and support structure.
        n: Geometric factor for gamma (steel area fraction) [dimensionless]
        grading: bool, optional
            If True, use radially graded α(R) model. Default: False.
    
    Returns:
        winding_pack_thickness: R_ext - R_sep [m]
        sigma_r: Radial stress at solution [Pa]
        sigma_z: Axial stress at solution [Pa]
        sigma_theta: Hoop stress at solution [Pa]
        Steel_fraction: 1 - alpha (structural fraction) [-]
            For graded: area-weighted average ⟨1 - α⟩.
    
    Limitations:
        The axial stress σ_z accounts for the vertical component of the
        in-plane Lorentz force (centering force), but does NOT include:
        - The overturning moment (out-of-plane torque from I_TF × B_poloidal).
          For high aspect ratio (A > 3), this contributes < 5% of σ_z.
          For low aspect ratio (A < 2.5, e.g. spherical tokamaks), the
          overturning bending stress can reach 10-20% of σ_z and should
          be evaluated with a dedicated structural model.
        - Thermal stresses from cool-down (typically 30-50 MPa for LTS,
          higher for HTS due to CTE mismatch).
        These omissions make the present model slightly non-conservative
        for compact / low-A designs.
    """
    
    R_ext = R_0 - a - b

    # Validate J_max before proceeding
    if J_max is None or J_max <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if R_ext <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        # raise ValueError("R_ext must be positive. Check R_0, a, and b.")

    ln_term = np.log((R_0 + a + b) / (R_ext))
    if ln_term <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        # raise ValueError("Invalid logarithmic term: ensure R_0 + a + b > R_0 - a - b")

    # Graded branch: delegate to radial integration solver
    if grading:
        return _solve_graded_wp(R_ext, B_max, J_max, sigma_max,
                                omega, n, ln_term)

    # Ungraded branch: analytical α, brentq root-finding (original model)
    def alpha(R_sep):
        denom = R_ext**2 - R_sep**2
        if denom <= 0:
            return np.nan
        val = (2 * B_max / (μ0 * J_max)) * (R_ext / denom)
        if np.iscomplex(val) or val < 0 or val > 1 :
            return np.nan
        return val

    def tresca_residual(R_sep):
        a_val = alpha(R_sep)
        if np.isnan(a_val):
            return np.inf
        g_val = gamma_func(a_val, n)
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
    # Search variable: WP thickness d = R_ext - R_sep (not R_sep directly).
    # Hybrid log+linear probing with exhaustive bracket collection handles
    # both thin HTS (d ~ cm) and thick LTS (d ~ m) solutions, including
    # narrow valid islands at high field that pure log spacing can miss.
    # Typical call budget: ~35 evaluations (Pass 1 success) to ~85 (worst case).

    R_sep_solution = None

    d_lo = 1e-3                # Minimum WP thickness [m]
    d_hi = R_ext - 1e-3        # Maximum WP thickness (nearly solid cylinder)

    if d_lo >= d_hi:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    def residual_vs_d(d):
        """Tresca residual as a function of WP thickness d = R_ext - R_sep."""
        return tresca_residual(R_ext - d)

    d_solution = _adaptive_root_search(
        residual_vs_d, d_lo, d_hi,
        n_probe_1=20, n_probe_2=25,
        select='smallest')

    if np.isnan(d_solution):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    R_sep_solution = R_ext - d_solution

    # === Final stress calculation ===
    a_val = alpha(R_sep_solution)
    g_val = gamma_func(a_val, n)

    if np.isnan(a_val) or np.isnan(g_val):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        sigma_r = B_max**2 / (2 * μ0 * g_val)
        sigma_z = (omega / (1 - a_val)) * B_max**2 * R_ext**2 / (2 * μ0 * (R_ext**2 - R_sep_solution**2)) * ln_term
        sigma_theta = 0
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    winding_pack_thickness = R_ext - R_sep_solution
    Steel_fraction = (1-a_val)

    return winding_pack_thickness, sigma_r, sigma_z, sigma_theta, Steel_fraction


def Nose_D0FUS(R_ext_Nose, sigma_max, omega, B_max, R_0, a, b,
               coef_inboard_tension):
    """
    Compute the inner radius of the TF nose (inner structural casing).

    Analytical thick-cylinder model: finds the inner radius Ri such that
    the Tresca stress at Ri equals sigma_max.

    The caller computes the nose thickness as:
        c_Nose = R_ext_Nose - Nose_D0FUS(R_ext_Nose, ...)

    Parameters
    ----------
    R_ext_Nose : float
        External radius of the nose region [m].
        Typically R_ext_Nose = R0 - a - b - c_WP.
    sigma_max : float
        Maximum allowable Tresca stress [Pa].
    omega : float
        Fraction of vertical tension carried by the inboard leg [-].
        Typical: 0.4–0.6 depending on coil shape and support structure.
    B_max : float
        Peak magnetic field at the TF inboard conductor (R0 - a - b) [T].
    R_0 : float
        Plasma major radius [m].
    a : float
        Plasma minor radius [m].
    b : float
        Radial build from plasma edge to TF inner face [m].
    coef_inboard_tension : float
        Correction factor for inboard tension distribution [-].
        Accounts for non-uniform current distribution across the WP.

    Returns
    -------
    Ri : float
        Inner radius of the nose [m].
        Returns NaN if no valid solution exists (stress too high).
    """
    
    # Centering pressure transmitted from WP to nose outer surface.
    # The magnetic pressure B_max²/(2μ0) acts at R_TF.  By cylindrical membrane
    # force balance (P × r = const for the centering force), the contact pressure
    # at R_nose is amplified by the circumference ratio R_TF / R_nose.
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

def f_TF_D0FUS(a, b, R0, σ_TF, J_max_TF, B_max_TF, Choice_Buck_Wedg, omega, n,
               c_BP, coef_inboard_tension, F_CClamp, TF_grading=False):
    
    """
    Calculate the thickness of the TF coil using a 2 layer thick cylinder model 

    Parameters:
    a : Minor radius (m)
    b : 1rst Wall + Breeding Blanket + Neutron Shield + Gaps (m)
    R0 : Major radius (m)
    σ_TF : Yield strength of the TF steel (Pa)
    J_max_TF : Maximum current density of the chosen Supra + Cu + He (A/m²)
    B_max_TF : Maximum magnetic field (T)
    Choice_Buck_Wedg : Mechanical configuration ('Bucking', 'Wedging', 'Plug')
    omega : Fraction of vertical tension on inboard leg [-]
    n : Conductor geometry factor for γ(α, n) [-]
    c_BP : Backplate thickness (m)
    coef_inboard_tension : Inboard tension correction [-]
    F_CClamp : Clamping force [N]
    TF_grading : bool, optional
        If True, use radially graded α(R) in the WP. Default: False.

    Returns:
    c : TF total inboard radial thickness [m]
    c_WP : Winding pack thickness [m]
    c_Nose : Nose thickness [m]
    σ_z, σ_theta, σ_r : Stress components [Pa]
    Steel_fraction : Average structural fraction [-]
    
    """
    
    debuging = 'Off'
    
    if Choice_Buck_Wedg == "Wedging":
        
        c_WP, σ_r, σ_z, σ_theta, Steel_fraction  = Winding_Pack_D0FUS( R0, a, b, σ_TF, J_max_TF, B_max_TF, omega, n, grading=TF_grading)
        
        # Vérification que c_WP est valide
        if c_WP is None or np.isnan(c_WP) or c_WP < 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        c_Nose = R0 - a - b - c_WP - Nose_D0FUS(R0 - a - b - c_WP, σ_TF, omega, B_max_TF, R0, a, b,
                                          coef_inboard_tension)

        # Vérification que c_Nose est valide
        if c_Nose is None or np.isnan(c_Nose) or c_Nose < 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Vérification que la somme ne dépasse pas R0 - a - b
        if (c_WP + c_Nose) > (R0 - a - b):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Epaisseur totale de la bobine
        c  = c_WP + c_Nose + c_BP
        
        if __name__ == "__main__" and debuging == 'On':
            
            print(f'Winding pack width : {c_WP}')
            print(f'Nose width : {c_Nose}')
            print(f'Backplate width : {c_BP}')
    
    elif Choice_Buck_Wedg == "Bucking" or Choice_Buck_Wedg == "Plug":
        
        c_WP, σ_r, σ_z, σ_theta, Steel_fraction = Winding_Pack_D0FUS(R0, a, b, σ_TF, J_max_TF, B_max_TF, omega, n, grading=TF_grading)
        
        c = c_WP
        c_Nose = 0
        
        # Vérification que c est valide
        if c is None or np.isnan(c) or c < 0 or c > R0 - a - b :
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    else : 
        print( "Choose a valid mechanical configuration" )
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return c, c_WP, c_Nose, σ_z, σ_theta, σ_r, Steel_fraction


#%% TF Benchmark
if __name__ == "__main__":
    # TF coil benchmark table: D0FUS vs published references
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_TF_benchmark_table(cfg=cfg)

#%% TF plot
    
if __name__ == "__main__":
    # TF winding-pack thickness vs B_max — Academic vs D0FUS
    # No explicit parameters: uses plot_TF_thickness_vs_field defaults
    # (Giannini 2023: a=2.0, b=2.7, R0=9.0, σ_TF=867 MPa, J=60 MA/m²)
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_TF_thickness_vs_field(cfg=cfg)

#%% TF Grading

if __name__ == "__main__":
    print("##################################################### TF Grading ##########################################################")
    # TF winding-pack grading: thickness, reduction, and α(R) profile
    # No explicit parameters: uses figures.py defaults
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_TF_grading_thickness_vs_field(cfg=cfg)
    figs.plot_TF_grading_reduction(cfg=cfg)
    figs.plot_TF_grading_alpha_profile(cfg=cfg)

#%% Print
        
if __name__ == "__main__":
    print("##################################################### CS Model ##########################################################")

#%% Resistivity and effective resistance
# NOTE: All plasma resistivity functions (eta_old, eta_spitzer, eta_sauter,
# eta_redl, _coulomb_logarithm) and the effective resistance / loop voltage
# functions (f_Reff, f_Vloop) now live in D0FUS_physical_functions.py.
# They are imported at module level.

if __name__ == "__main__":
    # Plasma resistivity: Wesson / Spitzer / Sauter / Redl — ITER-like parameters
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_resistivity_models()
    
#%% Magnetic flux calculation

def f_Psi_PI(R0, E_BD=0.25):
    """
    Plasma initiation flux from Faraday's law.

    During plasma breakdown, the central solenoid imposes a toroidal
    loop voltage V_loop = 2 pi R0 E_phi to ionise the filling gas and
    burn through low-Z impurities.  Integrating Faraday's law over the
    breakdown duration t_BD gives the consumed flux:

        Psi_PI = V_loop * t_BD = 2 pi R0 * (E_phi * t_BD)

    Because only the product E_phi * t_BD is unknown ,
    this product is lumped into a single calibration
    parameter E_BD [V.s/m].  The formula reduces to:

        Psi_PI = 2 pi R0 * E_BD

    The geometric factor 2 pi R0 captures the machine-size scaling
    (larger toroidal circumference requires more flux for the same
    breakdown conditions).  All breakdown physics (fill pressure,
    error field level, pre-ionisation, impurity content) is absorbed
    into E_BD.

    Calibration:
        ITER (R0=6.2 m): Psi_PI ~ 10 Wb  =>  E_BD ~ 0.26 V.s/m.
        Default E_BD = 0.25 is calibrated on this reference.

    Note: Psi_PI is typically < 5% of the total CS flux budget,
    so the result is weakly sensitive to the choice of E_BD.

    Parameters
    ----------
    R0 : float
        Plasma major radius [m].
    E_BD : float, optional
        Breakdown calibration parameter [V.s/m]. Default: 0.25.
        Product of the toroidal electric field at breakdown [V/m]
        and the breakdown duration [s].

    Returns
    -------
    Psi_PI : float
        Flux consumed during plasma initiation [Wb].

    References
    ----------
    Lloyd et al., Plasma Phys. Control. Fusion 33(11), 1441 (1991).
        Breakdown electric field requirements in tokamaks.
    Shimada et al., Nucl. Fusion 47, S1 (2007).
        ITER reference flux budget (Psi_PI ~ 10 Wb).
    """
    return 2.0 * np.pi * R0 * E_BD


def f_Psi_ind(R0, a, kappa, li, Ip):
    """
    Inductive flux consumed to build the plasma current (self-inductance).

    The flux stored in the poloidal magnetic field of a current-carrying
    torus is Psi_ind = Lp * Ip, where Lp = L_ext + L_int.

    External inductance: Hirshman & Neilson (1986) fit
    --------------------------------------------------
    L_ext is computed from the Hirshman & Neilson fit, which is exact
    for an axisymmetric current ring at finite inverse aspect ratio
    epsilon = a/R0 and accounts for elongation:

        aeps = (1 + 1.81 sqrt(eps) + 2.05 eps) ln(8/eps)
               - (2 + 9.25 sqrt(eps) - 1.21 eps)
        beps = 0.73 sqrt(eps) (1 + 2 eps^4 - 6 eps^5 + 3.7 eps^6)

        L_ext = mu0 R0 * aeps * (1 - eps) / (1 - eps + beps * kappa)

    This replaces the Neumann large-aspect-ratio formula
    L_ext = mu0 R0 [ln(8 R0 / (a sqrt(kappa))) - 2], which underestimates
    L_ext by ~8% for ITER (eps = 0.32) and fails qualitatively for
    spherical tokamaks (eps > 0.4).

    Internal inductance
    -------------------
        L_int = mu0 R0 li / 2

    where li = li(3) is the normalised internal inductance from
    f_q_profile_selfconsistent().

    Parameters
    ----------
    R0 : float
        Plasma major radius [m].
    a : float
        Plasma minor radius [m].
    kappa : float
        Plasma elongation [-].
    li : float
        Normalised internal inductance li(3) [-].
    Ip : float
        Plasma current [A].

    Returns
    -------
    Psi_ind : float
        Inductive flux [Wb].
    Lp : float
        Total plasma self-inductance [H].

    References
    ----------
    Hirshman S.P. & Neilson G.H., Phys. Fluids 29, 790 (1986).
        External inductance fit (exact at finite epsilon).
    Freidberg, Plasma Physics and Fusion Energy, Cambridge (2007), ch. 11.
    Wesson, Tokamaks, 4th ed., Oxford (2011), Sec. 3.9.
    """
    if a <= 0.0 or R0 <= 0.0:
        return (np.nan, np.nan)

    eps = a / R0

    # Hirshman & Neilson (1986) fit for external inductance
    sqrt_eps = math.sqrt(eps)
    aeps = ((1.0 + 1.81 * sqrt_eps + 2.05 * eps) * math.log(8.0 / eps)
            - (2.0 + 9.25 * sqrt_eps - 1.21 * eps))
    beps = 0.73 * sqrt_eps * (1.0 + 2.0*eps**4 - 6.0*eps**5 + 3.7*eps**6)

    L_ext = μ0 * R0 * aeps * (1.0 - eps) / (1.0 - eps + beps * kappa)

    # Internal inductance
    L_int = μ0 * R0 * li / 2.0

    Lp = L_ext + L_int
    return Lp * Ip, Lp


def f_Psi_res(R0, Ip, Ce=0.45):
    """
    Resistive flux consumed during plasma current ramp-up (Ejima formula).

    During current ramp-up, the toroidal electric field must overcome
    both the inductive back-EMF (accounted for in Psi_ind = Lp * Ip)
    and the Ohmic dissipation in the resistive plasma.  The flux lost
    to Ohmic dissipation is Psi_res.

    Dimensional analysis shows that this flux is independent of the
    plasma resistivity eta and minor radius a.  The resistive loop
    voltage scales as V_res ~ 2 pi R0 * eta * Ip / (pi a^2), and the
    ramp-up duration scales as the resistive diffusion time
    tau_R ~ mu0 a^2 / eta.  Their product:

        Psi_res ~ V_res * tau_R  ~  mu0 * R0 * Ip

    where eta and a cancel.  The result depends only on R0 and Ip,
    up to a dimensionless coefficient Ce:

        Psi_res = Ce * mu0 * R0 * Ip

    Ce is an empirical calibration coefficient.  It absorbs the effects
    of current profile evolution during the ramp (skin effect), ramp
    rate, early auxiliary heating (which reduces eta and hence Ce),
    and sawtooth-induced reconnection.

    Calibration:
        Doublet III (Ejima 1982):  Ce ~ 0.4  (original measurement).
        ITER (design basis):      Ce ~ 0.45 (default).

    Parameters
    ----------
    R0 : float
        Plasma major radius [m].
    Ip : float
        Flat-top plasma current [A].
    Ce : float, optional
        Ejima coefficient [-]. Default: 0.45 (ITER design basis).
        Lower values (~ 0.30) are sometimes assumed for reactor
        scenarios with early auxiliary heating during ramp-up,
        but lack experimental validation at reactor scale.

    Returns
    -------
    Psi_res : float
        Resistive flux consumed during current ramp-up [Wb].

    References
    ----------
    Ejima et al., Nucl. Fusion 22(10), 1341 (1982).
        Original measurement on Doublet III and dimensional argument.
    Shimada et al., Nucl. Fusion 47, S1 (2007).
        ITER flux budget with Ce = 0.45.
    """
    return Ce * μ0 * R0 * Ip


def f_Psi_PF(Ip, R0, a, kappa, beta_p, li, RCS_ext):
    """
    PF flux contribution from Shafranov equilibrium vertical field.

    Physical model
    --------------
    Toroidal force balance (Shafranov) requires a vertical field B_V to
    counteract the hoop force, tyre-tube force, and 1/R field gradient:

        B_V = (mu0 Ip) / (4 pi R0)
              * [beta_p + li/2 - 3/2 + ln(8 R0 / (a sqrt(kappa)))]

    The flux linked to the plasma is the integral of B_V over the
    annular area between the plasma axis and the CS outer face:

        Psi_PF = pi * (R0^2 - RCS_ext^2) * B_V

    The CS bore is excluded because flux threading the solenoid is
    already counted in Psi_CS (HELIOS convention, Johner 2011).

    Parameters
    ----------
    Ip : float
        Plasma current [A].
    R0 : float
        Plasma major radius [m].
    a : float
        Plasma minor radius [m].
    kappa : float
        Plasma elongation [-].
    beta_p : float
        Poloidal beta [-].  Use f_beta_P() from D0FUS_physical_functions.
    li : float
        Normalised internal inductance li(3) [-].
    RCS_ext : float
        CS outer radius [m].

    Returns
    -------
    Psi_PF : float
        PF flux contribution [Wb].
    B_V : float
        Shafranov vertical field [T].

    References
    ----------
    Shafranov V.D., Reviews of Plasma Physics, vol. 2 (1966).
    Johner J., Fusion Sci. Technol. 59, 308 (2011)
    Duchateau et al., Fusion Eng. Des. 89, 2606 (2014)
    """
    a_eff = a * math.sqrt(kappa)
    B_V = (μ0 * Ip) / (4.0 * np.pi * R0) * (
        beta_p + li / 2.0 - 1.5 + math.log(8.0 * R0 / a_eff)
    )
    Psi_PF = np.pi * (R0**2 - RCS_ext**2) * B_V
    return Psi_PF, B_V


def Magnetic_flux(Ip, I_Ohm, R0, a, κ, li, Ce, Temps_Plateau,
                  E_BD, beta_p, RCS_ext,
                  nbar, Tbar, Z_eff, q, nu_T, nu_n, eta_model,
                  rho_ped=1.0, n_ped_frac=0.0, T_ped_frac=0.0,
                  Vprime_data=None):
    """
    Assemble the four magnetic flux components for a tokamak plasma scenario.

    Flux budget:  Ψ_CS = Ψ_PI + Ψ_RampUp + Ψ_plateau - Ψ_PF

    Each term is computed by a dedicated function with its own docstring:
        f_Psi_PI   — plasma initiation (Faraday + calibration E_BD)
        f_Psi_ind  — inductive ramp-up (Hirshman & Neilson 1986)
        f_Psi_res  — resistive ramp-up (Ejima scaling)
        f_Vloop    — flat-top loop voltage (neoclassical conductance)
        f_Psi_PF   — PF coil contribution (Shafranov vertical field)

    Parameters
    ----------
    Ip : float
        Plasma current [MA].
    I_Ohm : float
        Ohmic plasma current at flat-top [MA].
    R0 : float
        Plasma major radius [m].
    a : float
        Plasma minor radius [m].
    κ : float
        Plasma elongation [-].
    li : float
        Normalised internal inductance li(3) [-].
    Ce : float
        Ejima coefficient [-].
    Temps_Plateau : float
        Flat-top (burn) duration [s].
    E_BD : float
        Breakdown calibration parameter [V.s/m].
    beta_p : float
        Poloidal beta [-]. Use f_beta_P() from D0FUS_physical_functions.
    RCS_ext : float
        CS outer radius [m].
    nbar : float
        Volume-averaged electron density [1e20 m^-3].
    Tbar : float
        Volume-averaged electron temperature [keV].
    Z_eff : float
        Effective ionic charge [-].
    q : float
        Safety factor at 95%% flux surface [-].
    nu_T : float
        Temperature profile peaking exponent [-].
    nu_n : float
        Density profile peaking exponent [-].
    eta_model : str
        Resistivity model: 'spitzer', 'sauter', 'redl' (recommended).
    rho_ped, n_ped_frac, T_ped_frac : float, optional
        Pedestal parameters (passed to f_Vloop).
    Vprime_data : tuple or None, optional
        Miller volume derivative data (passed to f_Vloop).

    Returns
    -------
    ΨPI : float
        Plasma initiation flux [Wb].
    ΨRampUp : float
        Total ramp-up flux (inductive + resistive) [Wb].
    Ψplateau : float
        Flat-top flux [Wb].
    ΨPF : float
        PF coil flux contribution [Wb].
    Vloop : float
        Steady-state loop voltage [V].
    """

    # --- Unit conversion: MA -> A ---
    Ip_A    = Ip    * 1e6
    I_Ohm_A = I_Ohm * 1e6

    # --- 1. Plasma initiation flux ---
    ΨPI = f_Psi_PI(R0, E_BD=E_BD)

    # --- 2. Ramp-up flux = inductive + resistive ---
    Ψind, _Lp = f_Psi_ind(R0, a, κ, li, Ip_A)
    Ψres      = f_Psi_res(R0, Ip_A, Ce)
    ΨRampUp   = Ψind + Ψres

    # --- 3. Flat-top flux = V_loop × t_burn ---
    Vloop    = f_Vloop(I_Ohm, a, κ, R0, Tbar, nbar, Z_eff, q,
                       nu_T, nu_n, eta_model,
                       rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                       T_ped_frac=T_ped_frac, Vprime_data=Vprime_data)
    Ψplateau = Vloop * Temps_Plateau

    # --- 4. PF coil contribution (Shafranov vertical field) ---
    ΨPF, _BV = f_Psi_PF(Ip_A, R0, a, κ, beta_p, li, RCS_ext)

    return (ΨPI, ΨRampUp, Ψplateau, ΨPF, Vloop)

#%% Flux generation benchmark — ITER Q=10

if __name__ == "__main__":

    p = dict(a=2.00, b=1.25, c=0.90, R0=6.2, Ip=15.0, κ=1.70,
             nbar=1.0, Tbar=8.0, Li=0.80, beta_p=0.65,
             Z_eff=1.7, q=3.0, nu_T=1.0, nu_n=0.1,
             I_Ohm=10.0, Ce=0.45, Temps_Plateau=400)

    RCS_ext = p["R0"] - p["a"] - p["b"] - p["c"] - 0.1
    ΨPI, ΨRampUp, Ψplateau, ΨPF, Vloop = Magnetic_flux(
        p["Ip"], p["I_Ohm"], p["R0"], p["a"], p["κ"], p["Li"],
        p["Ce"], p["Temps_Plateau"],
        E_BD=0.25, beta_p=p["beta_p"], RCS_ext=RCS_ext,
        nbar=p["nbar"], Tbar=p["Tbar"],
        Z_eff=p["Z_eff"], q=p["q"], nu_T=p["nu_T"], nu_n=p["nu_n"],
        eta_model='redl')
    Ψtot = ΨPI + ΨRampUp + Ψplateau

    # ref = 0 means no published reference available
    data = [
        ("Ψ_PI",       ΨPI,          0,   ""),
        ("Ψ_rampup",   ΨRampUp,      0,   ""),
        ("Ψ_PI+rampup",ΨPI+ΨRampUp, 210,  "Polevoi NF 55 (2015)"),
        ("Ψ_plateau",  Ψplateau,     30,  "Polevoi NF 55 (2015)"),
        ("Ψ_total",    Ψtot,        240,  "Polevoi NF 55 (2015)"),
        ("Ψ_PF",       ΨPF,          73,  "Duchateau FED 89 (2014)"),
        ("Ψ_CS",       Ψtot - ΨPF,   0,   ""),
        ("V_loop [mV]",Vloop*1e3,     0,   ""),
    ]

    print("\n  ITER Q=10 Flux Benchmark [Wb]")
    print(f"  {'':>16} {'D0FUS':>8} {'ref':>8}  source")
    print("  " + "-" * 58)
    for name, val, ref, src in data:
        r = f"{ref:.0f}" if ref else "—"
        print(f"  {name:>16} {val:>8.1f} {r:>8}  {src}")
    print()
        
#%% ===========================================================================
# CS ACADEMIC MODEL
# =============================================================================


def f_return_flux_correction(R_e, H_CS, R0, R_i=None):
    """
    Finite-solenoid return flux correction factor f_corr > 1.

    A finite-length solenoid has a return field B_z < 0 outside its
    winding (div B = 0 requires the field lines to close through the
    exterior).  This reduces the net poloidal flux linking the plasma
    at R0 compared to the total flux through the CS cross-section.
    The CS must therefore produce more flux than the plasma requires:

        Ψ_CS_solenoid = f_corr × Ψ_plasma,    f_corr = Ψ(R_e) / Ψ(R0).

    The correction depends only on geometry (R_e/R0, H/R_e); it is
    independent of the current density.  It is evaluated via
    Gauss-Legendre quadrature (20 × 40 nodes) over the solenoid
    cross-section using the exact elliptic integral kernel for A_phi
    of a current loop (Callaghan & Maslen 1960).

    Cross-checked against BOBOZ on ITER,
    EU-DEMO, JT-60SA and EAST: agreement < 1% on all four machines.

    Parameters
    ----------
    R_e : float
        CS outer radius [m].
    H_CS : float
        CS total height [m].
    R0 : float
        Plasma major radius [m].
    R_i : float, optional
        CS inner radius [m].  If None, estimated as 0.3 × R_e.

    Returns
    -------
    f_corr : float
        Correction factor Ψ(R_e) / Ψ(R0), >= 1.  Typical values
        1.3 (ITER, JT-60SA) to 1.5 (EAST).

    References
    ----------
    Derby N. & Olbert S., Am. J. Phys. 78(3), 229-235 (2010).
    Callaghan E.E. & Maslen S.H., NASA Technical Note D-465 (1960).
    """
    if R_i is None:
        R_i = max(0.3 * R_e, 0.01)
    if R_e <= R_i or H_CS <= 0 or R0 <= R_e:
        return 1.0

    h = H_CS / 2.0

    # Gauss-Legendre quadrature nodes and weights
    n_r, n_z = 20, 40
    r_pts, r_wts = np.polynomial.legendre.leggauss(n_r)
    z_pts, z_wts = np.polynomial.legendre.leggauss(n_z)

    # Map to integration domains [R_i, R_e] and [-h, +h]
    r_mid, r_half = (R_e + R_i) / 2.0, (R_e - R_i) / 2.0
    a_vals = r_mid + r_half * r_pts      # source radii
    z_vals = h * z_pts                    # source heights

    def _Aphi_ratio(R_obs):
        """Weighted A_phi integral at R_obs (arbitrary normalisation)."""
        total = 0.0
        for ir in range(n_r):
            a = a_vals[ir]
            for iz in range(n_z):
                z = z_vals[iz]
                denom = (a + R_obs)**2 + z**2
                k2 = 4.0 * a * R_obs / denom
                if k2 >= 1.0:
                    k2 = 1.0 - 1e-14
                if k2 < 1e-15:
                    continue
                K_val = ellipk(k2)
                E_val = ellipe(k2)
                sk2 = np.sqrt(k2)
                kernel = ((2.0 - k2) * K_val - 2.0 * E_val) / (2.0 * sk2)
                total += r_wts[ir] * z_wts[iz] * np.sqrt(a / R_obs) * kernel
        # Jacobian factors from quadrature domain mapping
        return total * r_half * h

    # Ψ(R) = 2πR × A_phi(R); the ratio cancels prefactors
    Psi_Re = R_e * _Aphi_ratio(R_e)
    Psi_R0 = R0  * _Aphi_ratio(R0)

    if abs(Psi_R0) < 1e-20:
        return 1.0
    return max(Psi_Re / Psi_R0, 1.0)


def _CS_geometry_init(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, κ,
                      Choice_Buck_Wedg, Gap, config=None):
    """
    Common geometry initialization for CS models (ACAD, D0FUS, CIRCE).

    Computes the CS outer radius, total flux requirement, and height.

    When config.cs_return_flux_correction is True, the plasma flux
    demand (ΨPI + ΨRampUp + Ψplateau - ΨPF, evaluated at R0) is
    multiplied by f_corr > 1 to account for the return field of the
    finite-length CS solenoid.  See f_return_flux_correction() for details.

    Parameters
    ----------
    ΨPI, ΨRampUp, Ψplateau, ΨPF : float
        Flux components [Wb] (plasma convention, at R0).
    a, b, c, R0 : float
        Radial build geometry [m].
    κ : float
        Plasma elongation [-].
    Choice_Buck_Wedg : str
        Mechanical configuration: 'Bucking', 'Wedging', or 'Plug'.
    Gap : float
        Radial gap between TF and CS [m], used only for 'Wedging'.
    config : GlobalConfig or None
        If provided, cs_return_flux_correction and H_CS override are read from it.

    Returns
    -------
    RCS_ext : float or None
        CS outer radius [m].  None if invalid geometry.
    ΨCS : float
        Required CS flux swing [Wb] (solenoid convention, at R_e).
    H_CS : float
        CS total height [m].

    References
    ----------
    Derby N. & Olbert S., Am. J. Phys. 78(3), 229-235 (2010).
    Callaghan E.E. & Maslen S.H., NASA Technical Note D-465 (1960).
    Auclair T. (2025), BOBOZ validation note, CEA-IRFM.
    """
    if Choice_Buck_Wedg in ('Bucking', 'Plug'):
        RCS_ext = R0 - a - b - c
    elif Choice_Buck_Wedg == 'Wedging':
        RCS_ext = R0 - a - b - c - Gap
    else:
        return None, np.nan, np.nan

    if RCS_ext <= 0.0:
        return None, np.nan, np.nan

    ΨCS = ΨPI + ΨRampUp + Ψplateau - ΨPF

    # CS height: use override if provided, otherwise default formula
    H_CS_override = getattr(config, 'H_CS', None) if config is not None else None
    if H_CS_override is not None and H_CS_override > 0:
        H_CS = H_CS_override
    else:
        H_CS = 2 * (κ * a + b + 1)

    # Finite-solenoid return flux correction (Derby & Olbert 2010, AJP 78(3);
    # Callaghan & Maslen 1960, NASA TN D-465).
    # Enabled by default. Benchmark scripts may disable via dynamic attribute
    # cfg.cs_return_flux_correction = False when Ψ is already CS flux.
    # Validated against BOBOZ (Auclair 2025) on ITER, EU-DEMO, JT-60SA, EAST (<1%).
    if config is not None and getattr(config, 'cs_return_flux_correction', True):
        f_corr = f_return_flux_correction(RCS_ext, H_CS, R0)
        ΨCS *= f_corr

    return RCS_ext, ΨCS, H_CS

def f_CS_ACAD(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
              config: GlobalConfig):
    """
    Calculate the Central Solenoid (CS) thickness using the thin-solenoid 
    approximation and a 2-layer model (superconductor + steel structure).
    
    Flux inversion uses the exact thick-solenoid integral:
        Ψ = (2/3) π μ₀ J (R_e³ − R_i³)
    Peak field from Ampere's law (exact for uniform-J solenoid):
        B_CS = μ₀ J_wost (R_ext − R_int)
    
    Iteration on J_wost(B_CS, E_mag) since B_CS and E_mag depend on R_int.

    Parameters
    ----------
    ΨPI : float
        Plasma initiation flux (Wb)
    ΨRampUp : float
        Current ramp-up flux swing (Wb)
    Ψplateau : float
        Flat-top operational flux (Wb)
    ΨPF : float
        Poloidal field coil contribution (Wb)
    a : float
        Plasma minor radius (m)
    b : float
        Cumulative radial build: 1st wall + breeding blanket + neutron shield + gaps (m)
    c : float
        Toroidal field (TF) coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T)
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier for Jc calculation
    Jc_manual : float
        If needed, manual current density
    T_Helium : float
        Helium coolant temperature (K)
    Choice_Buck_Wedg : str
        Mechanical support configuration: 'Bucking', 'Wedging', or 'Plug'
    κ : float
        Plasma elongation
    
    Returns
    -------
    tuple of float
        (d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS)
        
        Returns (nan, nan, nan, nan, nan, nan, nan) if no physically valid solution exists.

    References
    ----------
    [Auclair et al. NF 2026]

    config : GlobalConfig
        Global design configuration. Used to access: Gap, I_cond, V_max,
        f_He_pipe, f_void, f_In, T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn,
        Marge_T_NbTi, Marge_T_REBCO, Eps, Tet, n_shape_CS.
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    _c = _unpack_CS_config(config)
    Gap, I_cond, V_max   = _c['Gap'], _c['I_cond'], _c['V_max']
    f_He_pipe, f_void, f_In = _c['f_He_pipe'], _c['f_void'], _c['f_In']
    T_hotspot, RRR       = _c['T_hotspot'], _c['RRR']
    Marge_T_He, Marge_T_Nb3Sn = _c['Marge_T_He'], _c['Marge_T_Nb3Sn']
    Marge_T_NbTi, Marge_T_REBCO = _c['Marge_T_NbTi'], _c['Marge_T_REBCO']
    Eps, Tet, n_shape_CS       = _c['Eps'], _c['Tet'], _c['n_shape_CS']
    fatigue_CS     = _c['fatigue_CS']
    Operation_mode = _c['Operation_mode']
    debug = False
    Tol_CS = 1e-3

    # Common geometry (shared with f_CS_D0FUS via _CS_geometry_init)
    RCS_ext, ΨCS, H_CS = _CS_geometry_init(
        ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, κ, Choice_Buck_Wedg, Gap,
        config=config)
    if RCS_ext is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Total flux for CS
    
    # ------------------------------------------------------------------
    # STEP 1: Determine B_CS, J_max_CS, and d_SU with self-consistent
    #         iteration on RCS_int.
    #
    #   Flux integral (exact for uniform-J thick solenoid):
    #     ΨCS = (2/3)π μ₀ J (R_e³ − R_i³)
    #   Inversion:
    #     R_i³ = R_e³ − 3 ΨCS / (2π μ₀ J)
    #
    #   Peak field (Ampere's law, exact):
    #     B_CS = μ₀ J (R_e − R_i) = μ₀ J d
    #
    #   The iteration is on J_wost(B_CS, E_mag) since both B_CS and E_mag
    #   depend on R_int, which depends on J_wost.
    # ------------------------------------------------------------------
    
    # Conservative B estimate for Jc evaluation (overestimates B by up to
    # ~50% for wide WP → J_wost conservative → d_SU is an upper bound).
    # The true B_CS = μ₀Jd is computed after convergence.
    B_CS_est = ΨCS / (np.pi * RCS_ext**2)
    
    if debug:
        print(f"[STEP 1] Conservative B_CS estimate for Jc: {B_CS_est:.2f} T")
    
    # --- Analytical initial estimate for RCS_int ---
    d_est = ΨCS / (2 * np.pi * RCS_ext * B_CS_est)
    RCS_int = max(0.1 * RCS_ext, RCS_ext - d_est)
    
    # --- Iterative convergence loop ---
    max_iter_CS = 15
    converged = False
    
    for i_iter in range(max_iter_CS):
        # Magnetic energy with current R_int estimate
        E_mag_CS = calculate_E_mag_CS(B_CS_est, RCS_int, RCS_ext, H_CS)
        
        # Cable current density (cached → cheap on repeated calls)
        result_J = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS_est, T_op=T_Helium,
            E_mag=E_mag_CS, I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS,
            tau_h=tau_h, f_He_pipe=f_He_pipe, f_void=f_void, f_In=f_In,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He,
            Marge_T_Nb3Sn=Marge_T_Nb3Sn, Marge_T_NbTi=Marge_T_NbTi,
            Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=Jc_manual if Supra_choice_CS == 'Manual' else None)
        J_max_CS = result_J['J_wost']
        
        if J_max_CS < Tol_CS:
            if debug:
                print(f"[STEP 1] iter {i_iter}: Non-positive J_max_CS: {J_max_CS:.2e} A/m²")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Thick solenoid flux inversion → new R_int
        RCS_sep_cubed = RCS_ext**3 - (3 * ΨCS) / (2 * np.pi * μ0 * J_max_CS)
        
        if RCS_sep_cubed <= 0:
            if debug:
                print(f"[STEP 1] iter {i_iter}: Invalid RCS_sep³: {RCS_sep_cubed:.2e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        RCS_int_new = RCS_sep_cubed**(1/3)
        
        if debug:
            print(f"[STEP 1] iter {i_iter}: RCS_int = {RCS_int:.4f} → {RCS_int_new:.4f} m  "
                  f"(ΔR = {abs(RCS_int_new - RCS_int)*1e3:.2f} mm, "
                  f"J_wost = {J_max_CS/1e6:.1f} MA/m²)")
        
        # Convergence check
        if abs(RCS_int_new - RCS_int) < Tol_CS:
            converged = True
            RCS_int = RCS_int_new
            break
        
        RCS_int = RCS_int_new
    
    if not converged and debug:
        print(f"[STEP 1] WARNING: R_int loop did not converge after {max_iter_CS} iterations "
              f"(last ΔR = {abs(RCS_int_new - RCS_int)*1e3:.2f} mm)")
    
    # --- Final values from converged R_int ---
    RCS_sep = RCS_int
    d_SU = RCS_ext - RCS_sep
    
    if d_SU <= Tol_CS or d_SU >= RCS_ext - Tol_CS:
        if debug:
            print(f"[STEP 1] Invalid d_SU: {d_SU:.4f} m")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Peak field from Ampere's law: B = μ₀ J d  (exact for uniform-J solenoid)
    B_CS = μ0 * J_max_CS * d_SU
    
    if B_CS > B_max_CS:
        if debug:
            print(f"[STEP 1] B_CS = {B_CS:.2f} T exceeds limit {B_max_CS:.2f} T")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # ------------------------------------------------------------------
    # STEP 2: Find steel thickness d_SS using brentq
    # ------------------------------------------------------------------
    
    P_CS = B_CS**2 / (2.0 * μ0)
    P_TF = B_max_TF**2 / (2.0 * μ0)
    
    # Special case: Plug with TF dominant
    if Choice_Buck_Wedg == 'Plug' and abs(P_CS - P_TF) <= abs(P_TF):
        d = d_SU
        return d, 0, 0, 0, 0, B_CS, J_max_CS
    
    def stress_residual(d_SS):
        if d_SS <= Tol_CS:
            return np.nan
        d_total = d_SS + d_SU
        if d_total >= RCS_ext - Tol_CS:
            return np.nan
        
        if Choice_Buck_Wedg == 'Bucking':
            Sigma_CS = abs(np.nanmax([P_TF, abs(P_CS - P_TF)])) * RCS_sep / d_SS
        elif Choice_Buck_Wedg == 'Wedging':
            Sigma_CS = abs(P_CS * RCS_sep / d_SS)
        elif Choice_Buck_Wedg == 'Plug':
            if abs(P_CS - P_TF) > abs(P_TF):
                Sigma_CS = abs(abs(P_CS - P_TF) * RCS_sep / d_SS)
            else:
                return np.nan
        else:
            return np.nan
        
        if Sigma_CS < Tol_CS:
            return np.nan
        # Effective stress allowable: apply fatigue knockdown only when the
        # light case (CS own electromagnetic load dominant) governs AND the
        # scenario is pulsed. In Wedging the light case is the only case
        # (no TF back-pressure); in Bucking it governs when |P_CS-P_TF| > P_TF.
        _light_governs = (
            (Choice_Buck_Wedg == 'Wedging') or
            (Choice_Buck_Wedg == 'Bucking' and abs(P_CS - P_TF) > P_TF)
        )
        σ_eff = (σ_CS / fatigue_CS
                 if (Operation_mode == 'Pulsed' and _light_governs)
                 else σ_CS)
        return Sigma_CS - σ_eff
    
    # ------------------------------------------------------------------
    # STEP 2 — Direct two-point brentq (no linear scan required)
    # ------------------------------------------------------------------
    # For all three mechanical configurations (Wedging, Bucking, Plug),
    # stress_residual(d_SS) = K / d_SS - σ_CS, where K is a positive
    # constant (K = P_CS * RCS_sep or similar). This function is strictly
    # monotone decreasing on the valid domain (Tol_CS, d_SS_max), so:
    #   • At most one root exists.
    #   • Evaluating at both endpoints is sufficient to determine whether
    #     a root is present; no scan is needed.
    #   • If residual(d_SS_min) > 0 and residual(d_SS_max) < 0 → root
    #     exists; direct brentq.
    #   • Otherwise → design is geometrically infeasible (steel thickness
    #     required to meet stress limit exceeds available radial space).
    # Total cost: ~2 (bound checks) + ~15 (brentq) = ~17 evaluations,
    # compared to 200 for the previous linear scan.
    # ------------------------------------------------------------------

    d_SS_min = Tol_CS
    d_SS_max = RCS_ext - d_SU - Tol_CS

    if d_SS_max <= d_SS_min:
        if debug:
            print("[STEP 2] No space left for steel: d_SS_max <= d_SS_min.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # stress_residual returns nan when d_SS <= Tol_CS (guard) or when
    # d_SS + d_SU >= RCS_ext - Tol_CS (guard). Evaluating at the exact
    # bound values would therefore return nan on both sides, incorrectly
    # indicating an infeasible design. A small inward offset eps_SS avoids
    # both guards while keeping the bracket physically valid.
    eps_SS   = Tol_CS * 1e-3   # 1 μm for Tol_CS = 1 mm; negligible vs solution
    d_lo_ss  = d_SS_min + eps_SS
    d_hi_ss  = d_SS_max - eps_SS

    if d_lo_ss >= d_hi_ss:
        if debug:
            print("[STEP 2] Effective bracket collapsed after eps offset.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    y_lo = stress_residual(d_lo_ss)
    y_hi = stress_residual(d_hi_ss)

    if debug:
        print(f"[STEP 2] stress_residual: "
              f"f({d_lo_ss:.6f}) = {y_lo:.3e},  "
              f"f({d_hi_ss:.4f}) = {y_hi:.3e}")

    if not (np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo * y_hi < 0):
        # No sign change: stress constraint cannot be met in available space.
        if debug:
            print("[STEP 2] No root in valid domain — design infeasible.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        d_SS = brentq(stress_residual, d_lo_ss, d_hi_ss, xtol=1e-9)
    except ValueError:
        if debug:
            print("[STEP 2] brentq failed on [d_lo_ss, d_hi_ss].")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # ------------------------------------------------------------------
    # STEP 3: Compute final outputs
    # ------------------------------------------------------------------
    
    d_total = d_SS + d_SU
    alpha = d_SU / d_total
    
    if Choice_Buck_Wedg == 'Bucking':
        σ_theta = abs(np.nanmax([P_TF, abs(P_CS - P_TF)])) * RCS_sep / d_SS
        σ_r = 0
    elif Choice_Buck_Wedg == 'Wedging':
        σ_theta = abs(P_CS * RCS_sep / d_SS)
        σ_r = 0
    elif Choice_Buck_Wedg == 'Plug':
        if abs(P_CS - P_TF) > abs(P_TF):
            σ_r = abs(abs(P_CS - P_TF) * RCS_sep / d_SS)
            σ_theta = 0
    
    if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    d = d_total
    Steel_fraction = 1 - alpha
    σ_z = 0
    
    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

#%% ===========================================================================
# CS D0FUS MODEL
# =============================================================================


def f_sigma_z_CS_axial(J_smear, R_i, R_e, h):
    """
    Analytical axial (compressive) smeared stress at the CS midplane,
    induced by fringe-field Lorentz forces.

    Derivation: integrate the axial Lorentz body force J_smear x B_r from
    the free end (z=h, sigma_z=0) to the midplane (z=0).
    B_r is approximated from div.B=0 in the bore: B_r ~ -(r/2)*dBz/dz.

    Formula (Auclair 2026):
        sigma_z = -(mu0 * J_smear^2 * h * R_i / 2) * [L(h) - L(2h)]
        L(zeta) = ln((R_e + sqrt(R_e^2+zeta^2)) / (R_i + sqrt(R_i^2+zeta^2)))

    To convert smeared stress to steel stress:
        sigma_z_steel = sigma_z_smear / gamma
    where gamma = gamma_func(alpha, n_shape_CS), n_shape_CS being the conductor shape
    factor (1 = square jacket, 0 = optimal).

    Parameters
    ----------
    J_smear : float
        Smeared (homogenised) current density over the full CS cross-section
        [A/m^2]. Pass J_wost * alpha, where:
          alpha  = B_CS / (mu0 * J_wost * (R_e - R_i))  [-]
          J_wost = engineering current density without steel jacket [A/m^2]
        J_smear is the macroscopic J entering Maxwell: curl(B) = mu0 * J_smear.
    R_i : float
        CS inner winding radius [m]
    R_e : float
        CS outer winding radius [m]
    h : float
        CS half-height [m]  (h = H_CS / 2)

    Returns
    -------
    sigma_z : float
        Smeared axial stress at the midplane [Pa]. Negative = compressive.

    References
    ----------
    Auclair T. (2026), "Analytical axial stress at CS midplane", internal note.
    """
    def calL(zeta):
        # Logarithmic geometry factor for the on-axis field of a thick solenoid
        return np.log((R_e + np.sqrt(R_e**2 + zeta**2)) /
                      (R_i + np.sqrt(R_i**2 + zeta**2)))

    delta_L = calL(h) - calL(2.0 * h)   # > 0 since calL decreases with zeta

    sigma_z = -(μ0 * J_smear**2 * h * R_i / 2.0) * delta_L
    return float(sigma_z)


def f_CS_D0FUS(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
              config: GlobalConfig):
    """
    Parameters
    ----------
    ΨPI : float
        Plasma initiation flux (Wb)
    ΨRampUp : float
        Current ramp-up flux swing (Wb)
    Ψplateau : float
        Flat-top operational flux (Wb)
    ΨPF : float
        Poloidal field coil contribution (Wb)
    a : float
        Plasma minor radius (m)
    b : float
        Cumulative radial build (m)
    c : float
        TF coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T)
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier
    Jc_manual : float
        Manual current density if needed
    T_Helium : float
        Helium coolant temperature (K)
    Choice_Buck_Wedg : str
        Mechanical configuration: 'Bucking', 'Wedging', or 'Plug'
    κ : float
        Plasma elongation
        
    Returns
    -------
    tuple of float
        (d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS)
        
        Returns (nan, nan, nan, nan, nan, nan, nan) if no solution exists.

    References
    ----------
    [Auclair et al. NF 2026]
    
    config : GlobalConfig
        Global design configuration. Used to access: Gap, I_cond, V_max,
        f_He_pipe, f_void, f_In, T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn,
        Marge_T_NbTi, Marge_T_REBCO, Eps, Tet, n_shape_CS.
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    _c = _unpack_CS_config(config)
    Gap, I_cond, V_max   = _c['Gap'], _c['I_cond'], _c['V_max']
    f_He_pipe, f_void, f_In = _c['f_He_pipe'], _c['f_void'], _c['f_In']
    T_hotspot, RRR       = _c['T_hotspot'], _c['RRR']
    Marge_T_He, Marge_T_Nb3Sn = _c['Marge_T_He'], _c['Marge_T_Nb3Sn']
    Marge_T_NbTi, Marge_T_REBCO = _c['Marge_T_NbTi'], _c['Marge_T_REBCO']
    Eps, Tet, n_shape_CS       = _c['Eps'], _c['Tet'], _c['n_shape_CS']
    fatigue_CS     = _c['fatigue_CS']
    Operation_mode = _c['Operation_mode']
    debug = False
    Tol_CS = 1e-3

    # Common geometry (shared with f_CS_ACAD via _CS_geometry_init)
    RCS_ext, ΨCS, H_CS = _CS_geometry_init(
        ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, κ, Choice_Buck_Wedg, Gap,
        config=config)
    if RCS_ext is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # ------------------------------------------------------------------
    # Main solving function (uses CACHED cable current density)
    # ------------------------------------------------------------------

    def d_to_solve(d):
        if d < Tol_CS or d > RCS_ext - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        RCS_int = RCS_ext - d
        B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        E_mag_CS = calculate_E_mag_CS(B_CS, RCS_int, RCS_ext, H_CS)
        
        # STRATEGY C: Use cached cable current density
        result_J = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS, T_op=T_Helium, E_mag=E_mag_CS,
            I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS, tau_h=tau_h, f_He_pipe=f_He_pipe, f_void=f_void, f_In=f_In,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=Jc_manual if Supra_choice_CS == 'Manual' else None)
        J_max_CS = result_J['J_wost']
        
        if J_max_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        alpha = B_CS / (μ0 * J_max_CS * d)
        
        if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if B_CS < Tol_CS or B_CS > B_max_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Compute principal stresses and apply 3D Tresca criterion.
        # Hoop and radial stresses in the structural steel are scaled by 1/(1-alpha).
        P_CS = B_CS**2 / (2.0 * μ0)
        P_TF = B_max_TF**2 / (2.0 * μ0) * (R0 - a - b) / RCS_ext
        denom_stress = RCS_ext**2 - RCS_int**2

        if abs(denom_stress) < 1e-30:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        gamma_val = gamma_func(alpha, n_shape_CS)
        if np.isnan(gamma_val):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Axial smeared stress at the midplane (compressive, < 0).
        # Always enabled — fringe-field axial stress is part of the physical model.
        J_smear   = J_max_CS * alpha
        σ_z_smear = f_sigma_z_CS_axial(J_smear, RCS_int, RCS_ext, H_CS / 2.0)
        σ_z_steel = σ_z_smear / gamma_val                       # Peak steel axial stress [Pa]

        def tresca(s_t, s_r, s_z):
            """Tresca equivalent stress: max principal stress difference [Pa]."""
            return max(abs(s_t - s_r), abs(s_r - s_z), abs(s_t - s_z))

        if Choice_Buck_Wedg == 'Bucking':
            # Light case (CS energized): tensile hoop − TF back-pressure, non-zero σ_z.
            # Strong case (J = 0, TF only): compressive hoop, σ_z = 0.
            sigma_theta_light  = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) -
                                  (2.0 * P_TF * RCS_ext**2 / denom_stress)) / (1.0 - alpha)
            sigma_theta_strong = -(2.0 * P_TF * RCS_ext**2 / denom_stress) / (1.0 - alpha)

            tresca_light  = tresca(sigma_theta_light,  0.0, σ_z_steel)
            tresca_strong = tresca(sigma_theta_strong, 0.0, 0.0)

            Sigma_CS = max(tresca_light, tresca_strong)
            σ_theta  = sigma_theta_light
            σ_r      = 0.0
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Wedging':
            sigma_theta = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress)
                           / (1.0 - alpha))
            Sigma_CS = tresca(sigma_theta, 0.0, σ_z_steel)
            σ_theta  = sigma_theta
            σ_r      = 0.0
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Plug':
            if abs(P_CS - P_TF) > abs(P_TF):
                # Bucking-dominated: two-case Tresca as above.
                sigma_theta_light  = ((P_CS * (RCS_ext**2 + RCS_int**2) / denom_stress) -
                                      (2.0 * P_TF * RCS_ext**2 / denom_stress)) / (1.0 - alpha)
                sigma_theta_strong = -(2.0 * P_TF * RCS_ext**2 / denom_stress) / (1.0 - alpha)
                tresca_light  = tresca(sigma_theta_light,  0.0, σ_z_steel)
                tresca_strong = tresca(sigma_theta_strong, 0.0, 0.0)
                Sigma_CS = max(tresca_light, tresca_strong)
                σ_theta  = sigma_theta_light
                σ_r      = 0.0
                σ_z      = σ_z_steel
            elif abs(P_CS - P_TF) <= abs(P_TF):
                # Plug-dominated: J → 0, dominant stress is radial, σ_z = 0.
                sigma_r_signed = -(abs(P_TF) / gamma_val)
                Sigma_CS = tresca(0.0, sigma_r_signed, 0.0)
                σ_r      = sigma_r_signed
                σ_theta  = 0.0
                σ_z      = 0.0
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if Sigma_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        Steel_fraction = 1 - alpha

        return (float(Sigma_CS), float(σ_z), float(σ_theta), float(σ_r),
                float(Steel_fraction), float(B_CS), float(J_max_CS))
    
    # ------------------------------------------------------------------
    # Root function
    # ------------------------------------------------------------------
    
    def f_sigma_diff(d):
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d)
        if not np.isfinite(Sigma_CS):
            return np.nan
        # Effective stress allowable: apply fatigue knockdown only when the light
        # case (CS own electromagnetic load dominant) governs AND the scenario is
        # pulsed. The light case is identified by a net tensile hoop stress
        # (sigma_theta > 0). In Wedging, sigma_theta is always tensile (no TF
        # back-pressure); in Bucking, the sign depends on the relative magnitudes
        # of P_CS and P_TF.
        _light_governs = (
            Choice_Buck_Wedg in ('Wedging', 'Bucking')
            and np.isfinite(σ_theta) and σ_theta > 0.0
        )
        σ_eff = (σ_CS / fatigue_CS
                 if (Operation_mode == 'Pulsed' and _light_governs)
                 else σ_CS)
        return Sigma_CS - σ_eff
    
    # ------------------------------------------------------------------
    # Root-finding — adaptive multi-bracket solver
    # ------------------------------------------------------------------
    # Uses _adaptive_root_search: hybrid log+linear probing with exhaustive
    # bracket collection.  Handles non-monotone residuals (CIRCE) and narrow
    # valid islands at high field that the previous single-bracket approach
    # could miss.  Typical call budget: ~35 (Pass 1 success) to ~85 (worst).
    # ------------------------------------------------------------------

    d_lo = Tol_CS + 1e-6
    d_hi = RCS_ext - Tol_CS

    if d_lo >= d_hi:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    d_sol = _adaptive_root_search(
        f_sigma_diff, d_lo, d_hi,
        n_probe_1=20, n_probe_2=25,
        select='smallest')

    if np.isnan(d_sol):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Recover full output at solution
    Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d_sol)

    if not np.isfinite(Sigma_CS):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    d = d_sol

    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

#%% ===========================================================================
# CS CIRCE MODEL
# =============================================================================

def f_CS_CIRCE(ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, B_max_TF, B_max_CS, σ_CS,
              Supra_choice_CS, Jc_manual, T_Helium, Choice_Buck_Wedg, κ, N_sub_CS, tau_h,
              config: GlobalConfig):
    """
    Calculate the Central Solenoid (CS) thickness using CIRCE.
    
    Parameters
    ----------
    ΨPI : float
        Plasma initiation flux (Wb)
    ΨRampUp : float
        Current ramp-up flux swing (Wb)
    Ψplateau : float
        Flat-top operational flux (Wb)
    ΨPF : float
        Poloidal field coil contribution (Wb)
    a : float
        Plasma minor radius (m)
    b : float
        Cumulative radial build (m)
    c : float
        TF coil radial thickness (m)
    R0 : float
        Major radius (m)
    B_max_TF : float
        Maximum TF coil magnetic field (T)
    B_max_CS : float
        Maximum CS magnetic field (T)
    σ_CS : float
        Yield strength of CS structural steel (Pa)
    Supra_choice_CS : str
        Superconductor type identifier
    Jc_manual : float
        Manual current density if needed
    T_Helium : float
        Helium coolant temperature (K)
    Choice_Buck_Wedg : str
        Mechanical configuration: 'Bucking', 'Wedging', or 'Plug'
    κ : float
        Plasma elongation
        
    Returns
    -------
    tuple of float
        (d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS)
        
        Returns (nan, nan, nan, nan, nan, nan, nan) if no solution exists.

    References
    ----------
    [Auclair et al. NF 2026]

    config : GlobalConfig
        Global design configuration. Used to access: Gap, I_cond, V_max,
        f_He_pipe, f_void, f_In, T_hotspot, RRR, Marge_T_He, Marge_T_Nb3Sn,
        Marge_T_NbTi, Marge_T_REBCO, Eps, Tet, n_shape_CS.
    """
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    _c = _unpack_CS_config(config)
    Gap, I_cond, V_max   = _c['Gap'], _c['I_cond'], _c['V_max']
    f_He_pipe, f_void, f_In = _c['f_He_pipe'], _c['f_void'], _c['f_In']
    T_hotspot, RRR       = _c['T_hotspot'], _c['RRR']
    Marge_T_He, Marge_T_Nb3Sn = _c['Marge_T_He'], _c['Marge_T_Nb3Sn']
    Marge_T_NbTi, Marge_T_REBCO = _c['Marge_T_NbTi'], _c['Marge_T_REBCO']
    Eps, Tet, n_shape_CS       = _c['Eps'], _c['Tet'], _c['n_shape_CS']
    fatigue_CS     = _c['fatigue_CS']
    Operation_mode = _c['Operation_mode']
    Young_modul_Steel = config.Young_modul_Steel
    nu_Steel          = config.nu_Steel
    debug = False
    Tol_CS = 1e-3

    # Common geometry (shared with f_CS_ACAD, f_CS_D0FUS via _CS_geometry_init)
    RCS_ext, ΨCS, H_CS = _CS_geometry_init(
        ΨPI, ΨRampUp, Ψplateau, ΨPF, a, b, c, R0, κ, Choice_Buck_Wedg, Gap,
        config=config)
    if RCS_ext is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # ------------------------------------------------------------------
    # Main solving function (uses CACHED cable current density)
    # ------------------------------------------------------------------

    def d_to_solve(d):
        
        if d < Tol_CS or d > RCS_ext - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        RCS_int = RCS_ext - d
        B_CS = 3 * ΨCS / (2 * np.pi * (RCS_ext**2 + RCS_ext * RCS_int + RCS_int**2))
        E_mag_CS = calculate_E_mag_CS(B_CS, RCS_int, RCS_ext, H_CS)
        
        # STRATEGY C: Use cached cable current density
        result_J = calculate_cable_current_density(
            sc_type=Supra_choice_CS, B_peak=B_CS, T_op=T_Helium, E_mag=E_mag_CS,
            I_cond=I_cond, V_max=V_max, N_sub=N_sub_CS, tau_h=tau_h, f_He_pipe=f_He_pipe, f_void=f_void, f_In=f_In,
            T_hotspot=T_hotspot, RRR=RRR, Marge_T_He=Marge_T_He, Marge_T_Nb3Sn=Marge_T_Nb3Sn,
            Marge_T_NbTi=Marge_T_NbTi, Marge_T_REBCO=Marge_T_REBCO, Eps=Eps, Tet=Tet,
            J_wost_Manual=Jc_manual if Supra_choice_CS == 'Manual' else None)
        J_max_CS = result_J['J_wost']
        
        if J_max_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        alpha = B_CS / (μ0 * J_max_CS * d)
        
        if alpha < Tol_CS or alpha > 1.0 - Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if B_CS < Tol_CS or B_CS > B_max_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Compute principal stresses with CIRCE and apply 3D Tresca criterion.
        # Hoop and radial smeared stresses are converted to steel via 1/(1-alpha).
        P_CS = B_CS**2 / (2.0 * μ0)
        P_TF = B_max_TF**2 / (2.0 * μ0) * (R0 - a - b) / RCS_ext

        disR       = 20
        R_arr      = np.array([RCS_int, RCS_ext])
        E_arr      = np.array([Young_modul_Steel])
        config_arr = np.array([0])

        gamma_val = gamma_func(alpha, n_shape_CS)
        if np.isnan(gamma_val):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Axial smeared stress at the midplane (compressive, < 0).
        # Always enabled — fringe-field axial stress is part of the physical model.
        J_smear   = J_max_CS * alpha
        σ_z_smear = f_sigma_z_CS_axial(J_smear, RCS_int, RCS_ext, H_CS / 2.0)
        σ_z_steel = σ_z_smear / gamma_val                       # Peak steel axial stress [Pa]

        scale_tr = 1.0 / (1.0 - alpha)   # Hoop/radial smeared → steel (volume average)
        scale_z  = 1.0 / gamma_val        # Axial smeared → steel (minimum ligament)

        def tresca_radial(SigT, SigR, sz_smear, s_tr, s_z):
            """
            Point-wise Tresca criterion on the CIRCE radial stress profile.

            Parameters
            ----------
            SigT, SigR : ndarray
                Smeared hoop and radial stress profiles from F_CIRCE0D [Pa].
            sz_smear : float
                Smeared axial stress (scalar, uniform over profile) [Pa].
            s_tr : float
                Scale factor smeared → steel for hoop/radial: 1/(1-alpha)
                (volume-average section, body force in steel).
            s_z : float
                Scale factor smeared → steel for axial: 1/gamma_val
                (minimum ligament — cable→jacket wall transfer).

            Returns
            -------
            float
                Maximum Tresca equivalent stress in the steel [Pa].
            """
            T = SigT * s_tr
            R = SigR * s_tr
            Z = sz_smear * s_z
            return float(np.max(np.maximum(np.maximum(np.abs(T - R),
                                                      np.abs(R - Z)),
                                           np.abs(T - Z))))

        # --- CIRCE computation ---
        if Choice_Buck_Wedg == 'Bucking':

            # Light case: CS energized — non-zero σ_z.
            J_light = np.array([J_max_CS * alpha])
            B_light = np.array([B_CS])
            SigR_L, SigT_L, _, _, _ = F_CIRCE0D(
                disR, R_arr, J_light, B_light, 0, P_TF, E_arr, nu_Steel, config_arr)
            tresca_light = tresca_radial(SigT_L, SigR_L, σ_z_smear, scale_tr, scale_z)

            # Strong case: J = 0, σ_z = 0.
            J_strong = np.array([0.0])
            B_strong = np.array([0.0])
            SigR_S, SigT_S, _, _, _ = F_CIRCE0D(
                disR, R_arr, J_strong, B_strong, 0, P_TF, E_arr, nu_Steel, config_arr)
            tresca_strong = tresca_radial(SigT_S, SigR_S, 0.0, scale_tr, scale_z)

            Sigma_CS = max(tresca_light, tresca_strong)
            # Inner-radius signed hoop stress: preserves sign so that the
            # fatigue knockdown in f_sigma_diff fires only when the light
            # case genuinely produces tensile hoop (σ_θ > 0).  The previous
            # np.max(np.abs(...)) always returned a positive value, making
            # _light_governs erroneously True even when P_TF dominates and
            # the hoop is compressive — this inflated the required thickness
            # by applying the fatigue penalty to the strong (compressive)
            # case.  Fixed: use SigT_L[0] (inner radius, where σ_r = 0 and
            # σ_θ is maximum in magnitude) with its physical sign.
            σ_theta  = float(SigT_L[0]) * scale_tr
            σ_r      = float(np.max(np.abs(SigR_L))) * scale_tr
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Wedging':

            J_w = np.array([J_max_CS * alpha])
            B_w = np.array([B_CS])
            SigR_W, SigT_W, _, _, _ = F_CIRCE0D(
                disR, R_arr, J_w, B_w, 0, 0, E_arr, nu_Steel, config_arr)
            Sigma_CS = tresca_radial(SigT_W, SigR_W, σ_z_smear, scale_tr, scale_z)
            # Inner-radius signed hoop stress (consistent with Bucking fix).
            # In Wedging σ_θ is always tensile (no TF back-pressure), so
            # this change has no functional impact — only sign consistency.
            σ_theta  = float(SigT_W[0]) * scale_tr
            σ_r      = float(np.max(np.abs(SigR_W))) * scale_tr
            σ_z      = σ_z_steel

        elif Choice_Buck_Wedg == 'Plug':

            if abs(P_CS - P_TF) > abs(P_TF):
                # Bucking-dominated: two-case Tresca as Bucking.
                J_light  = np.array([J_max_CS * alpha])
                B_light  = np.array([B_CS])
                SigR_L, SigT_L, _, _, _ = F_CIRCE0D(
                    disR, R_arr, J_light, B_light, 0, P_TF, E_arr, nu_Steel, config_arr)
                tresca_light = tresca_radial(SigT_L, SigR_L, σ_z_smear, scale_tr, scale_z)

                J_strong = np.array([0.0])
                B_strong = np.array([0.0])
                SigR_S, SigT_S, _, _, _ = F_CIRCE0D(
                    disR, R_arr, J_strong, B_strong, 0, P_TF, E_arr, nu_Steel, config_arr)
                tresca_strong = tresca_radial(SigT_S, SigR_S, 0.0, scale_tr, scale_z)

                Sigma_CS = max(tresca_light, tresca_strong)
                # Inner-radius signed hoop stress (same fix as Bucking).
                σ_theta  = float(SigT_L[0]) * scale_tr
                σ_r      = float(np.max(np.abs(SigR_L))) * scale_tr
                σ_z      = σ_z_steel

            elif abs(P_CS - P_TF) <= abs(P_TF):
                # Plug-dominated: J → 0, dominant stress is radial, σ_z = 0.
                # Hoop/radial scaled by 1/gamma_val (TF plug geometry).
                gamma_val_plug = gamma_func(alpha, n_shape_CS)
                J_plug = np.array([0.0])
                B_plug = np.array([0.0])
                SigR_P, SigT_P, _, _, _ = F_CIRCE0D(
                    disR, R_arr, J_plug, B_plug, P_TF, P_TF, E_arr, nu_Steel, config_arr)
                scale_plug = 1.0 / gamma_val_plug
                Sigma_CS = tresca_radial(SigT_P, SigR_P, 0.0, scale_plug, scale_plug)
                σ_theta  = float(np.max(np.abs(SigT_P))) * scale_plug
                σ_r      = float(np.max(np.abs(SigR_P))) * scale_plug
                σ_z      = 0.0
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if Sigma_CS < Tol_CS:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        Steel_fraction = 1 - alpha

        return (float(Sigma_CS), float(σ_z), float(σ_theta), float(σ_r),
                float(Steel_fraction), float(B_CS), float(J_max_CS))
    
    # ------------------------------------------------------------------
    # Root function
    # ------------------------------------------------------------------
    
    def f_sigma_diff(d):
        Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d)
        if not np.isfinite(Sigma_CS):
            return np.nan
        # Effective stress allowable: apply fatigue knockdown only when the light
        # case (CS own electromagnetic load dominant) governs AND the scenario is
        # pulsed. The light case is identified by a net tensile hoop stress
        # (sigma_theta > 0). In Wedging, sigma_theta is always tensile (no TF
        # back-pressure); in Bucking, the sign depends on the relative magnitudes
        # of P_CS and P_TF.
        _light_governs = (
            Choice_Buck_Wedg in ('Wedging', 'Bucking')
            and np.isfinite(σ_theta) and σ_theta > 0.0
        )
        σ_eff = (σ_CS / fatigue_CS
                 if (Operation_mode == 'Pulsed' and _light_governs)
                 else σ_CS)
        return Sigma_CS - σ_eff
    
    # ------------------------------------------------------------------
    # Root-finding — adaptive multi-bracket solver
    # ------------------------------------------------------------------
    # Uses _adaptive_root_search: hybrid log+linear probing with exhaustive
    # bracket collection.  Critical for CIRCE where the Tresca maximum can
    # jump between bore and outer radius, making the residual potentially
    # non-monotone and creating multiple roots.  The solver returns the
    # smallest valid d (thinnest feasible CS winding pack).
    # ------------------------------------------------------------------

    d_lo = Tol_CS + 1e-6
    d_hi = RCS_ext - Tol_CS

    if d_lo >= d_hi:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    d_sol = _adaptive_root_search(
        f_sigma_diff, d_lo, d_hi,
        n_probe_1=20, n_probe_2=25,
        select='smallest')

    if np.isnan(d_sol):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Recover full output at solution
    Sigma_CS, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS = d_to_solve(d_sol)

    if not np.isfinite(Sigma_CS):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    d = d_sol

    return d, σ_z, σ_theta, σ_r, Steel_fraction, B_CS, J_max_CS

#%% CS Benchmark

if __name__ == "__main__":
    # CS coil benchmark table: D0FUS vs published references
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_CS_benchmark_table(cfg=cfg)

#%% CS plot

if __name__ == "__main__":
    # CS winding-pack thickness and B_CS vs volt-second — Academic/D0FUS/CIRCE
    # No explicit parameters: uses plot_CS_thickness_vs_flux defaults
    # (Sarasola 2020: a=3, b=1.2, c=2, R0=9, σ_CS=600 MPa, J=85 MA/m²)
    import D0FUS_BIB.D0FUS_figures as figs
    figs.plot_CS_thickness_vs_flux(cfg=cfg)

#%% Note:
# CIRCE TF double cylindre en wedging ? multi cylindre for grading ?
# Nécessite la résolution de R_int et R_sep en même temps
# Permettrait aussi de mettre la répartition en tension en rapport de surface
#%% Print

# print("D0FUS_radial_build_functions loaded")