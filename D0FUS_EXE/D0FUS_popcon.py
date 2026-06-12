"""
D0FUS POPCON execution mode — Plasma OPeration CONtour maps.

Produces an operating-space map in the (line-averaged density, volume-averaged
temperature) plane at FIXED machine design: the deck is a standard RUN deck
(which defines the design point: geometry, fields, current, profiles, seeding)
followed by a [POPCON] section specifying the map grid:

    [POPCON]
    nbar_line = [0.3, 1.6, 40]    # line-averaged density grid [1e20 m^-3]
    Tbar      = [4.0, 25.0, 30]   # volume-averaged temperature grid [keV]

Physics
-------
The reference RUN solves the design point; the POPCON engine then freezes the
machine (B0, Ip, V, kappa, delta, helium fraction f_alpha, ohmic current
I_Ohm) and, at each grid point (nbar_line, Tbar):

1. Fusion power. With fixed profiles and dilution, P_fus = C(Tbar) * nbar^2
   exactly; C(Tbar) is obtained per temperature column from one call to the
   existing inversion f_nbar (P_fus -> nbar), so the POPCON reuses the RUN
   reactivity/profile machinery with zero duplication.
2. Stored energy W_th and core radiation (Bremsstrahlung with the fuel
   effective charge, synchrotron, Mavrin line radiation split core/edge),
   through the same profile-integrated functions as the RUN driver.
3. Confinement power balance. The selected tau_E scaling law
       tau_sc = H * C_SL * Ip^aI R0^aR eps^ae kappa_x^ak (10*nbar_line)^an
                * B0^aB M^aM P_loss^aP (1+delta)^ad
   combined with tau_E = W/P_loss gives the closed form
       P_loss = (W[MJ] / X)^(1/(1+aP)),   X = tau_sc / P_loss^aP,
   from which the required auxiliary power follows the RUN convention
       P_aux = P_loss + coreradiationfraction * P_rad_core - P_alpha - P_Ohm.
   P_aux <= 0 marks the IGNITION region; Q = P_fus / (P_aux + P_Ohm).
4. Operational limits: L-H threshold fraction f_LH = P_sep / P_LH (deck
   L_H_Scaling_choice), Greenwald fraction, the model-selectable density
   limit (config.density_limit_model) and the normalised beta.

Simplifications vs the RUN driver (documented, v1):
- f_alpha (helium ash) and I_Ohm are frozen at the design-point values;
- the flux-surface Jacobian follows the deck geometry mode (Miller Vprime
  in refined mode, analytic integrals in Academic mode), as in the driver;
- P_Ohm(Tbar) is recomputed from f_P_Ohm at the frozen I_Ohm.

Closure against the RUN driver at the design point (ITER deck, refined
geometry): P_fus and tau_E reproduce the run exactly, P_sep within 0.4%,
Q within ~4% (auxiliary-power reconstruction).

References
----------
Houlberg W.A., Attenberger S.E. & Hively L.M., Nucl. Fusion 22 (1982) 935 —
    original POPCON contour analysis of fusion plasma performance.
Body T. et al., cfspopcon (github.com/cfs-energy/cfspopcon) — modern
    open-source 0D POPCON reference implementation that inspired this mode.

Author: Auclair Timothe (mode added 2026)
"""

# Bootstrap imports only: sys/os are required to make the package importable
# before D0FUS_import.py (which centralises every other import) is reachable.
import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from D0FUS_BIB.D0FUS_import import *
from D0FUS_BIB.D0FUS_physical_functions import *
from D0FUS_BIB.D0FUS_parameterization import GlobalConfig, DEFAULT_CONFIG

from D0FUS_EXE import D0FUS_run as RUN
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Deck parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_popcon_file(path):
    """
    Split a POPCON deck into the design section (a standard RUN deck) and the
    [POPCON] grid specification.

    Returns
    -------
    config : GlobalConfig      Design-point configuration (RUN parser).
    grid : dict                {'nbar_line': (min, max, n), 'Tbar': (min, max, n)}
    """
    section = 'design'
    design, grid_spec = [], {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('[') and s.endswith(']') and '=' not in s:
                section = s[1:-1].strip().lower()
                continue
            if section == 'design':
                design.append(line.rstrip('\n'))
                continue
            if section == 'popcon':
                body = line.split('#', 1)[0].strip()
                if not body or '=' not in body:
                    continue
                key, val = (x.strip() for x in body.split('=', 1))
                m = re.match(r'^\[\s*([^,\]]+)\s*,\s*([^,\]]+)\s*,\s*([^,\]]+)\s*\]$', val)
                if not m:
                    raise ValueError(
                        f"[POPCON] entry '{key}' must be of the form "
                        f"[min, max, n_points]; got '{val}'.")
                grid_spec[key] = (float(m.group(1)), float(m.group(2)),
                                  int(float(m.group(3))))

    for needed in ('nbar_line', 'Tbar'):
        if needed not in grid_spec:
            raise ValueError(f"[POPCON] section must define '{needed} = "
                             f"[min, max, n_points]'.")

    fd, tmp = tempfile.mkstemp(suffix='_popcon_design.txt')
    os.close(fd)
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(design) + '\n')
    config = RUN.load_config_from_file(tmp, verbose=0)
    return config, grid_spec


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

# Results-tuple indices (authoritative map: the unpack in RUN.save_run_output).
_IDX = dict(B0=0, tauE=3, W_th=4, Q=5, Volume=6, Surface=7, Ip=8, I_Ohm=11,
            nbar=12, nbar_line=13, nG=14, pbar=15, betaN=16, q95=20,
            P_sep=22, P_Thresh=23, f_alpha=40, kappa=62, delta=64)


def compute_popcon(config, grid_spec, verbose=1):
    """
    Evaluate the POPCON maps.

    Returns a dict of 2D arrays of shape (n_T, n_n) plus grid vectors and
    reference design metadata.
    """
    # ── Reference design point ────────────────────────────────────────────
    if verbose:
        print("POPCON: solving the reference design point (RUN driver)...")
    res = RUN.run(config, verbose=0)
    rd = RUN._build_run_dict(config, res)

    B0      = float(res[_IDX['B0']])
    Ip      = float(res[_IDX['Ip']])
    V       = float(res[_IDX['Volume']])
    I_Ohm   = float(res[_IDX['I_Ohm']])
    f_alpha = float(res[_IDX['f_alpha']])
    kappa   = float(res[_IDX['kappa']])
    delta   = float(res[_IDX['delta']])
    nG_raw  = f_nG(Ip, config.a)

    nu_n       = float(rd['nu_n'])
    nu_T       = float(rd['nu_T'])
    rho_ped    = float(rd['rho_ped'])
    n_ped_frac = float(rd['n_ped_frac'])
    T_ped_frac = float(rd['T_ped_frac'])

    R0, a = config.R0, config.a
    eps   = a / R0
    M     = config.Atomic_mass
    tau_ie = getattr(config, 'tau_i_e', 1.0)

    # Flux-surface Jacobian for the profile integrals, replicating the RUN
    # driver: refined geometry mode uses the Miller Vprime; Academic mode
    # falls back to the analytic profile integrals (Vprime_data = None).
    kappa_95 = float(res[_IDX['kappa'] + 1])
    delta_95 = float(res[_IDX['delta'] + 1])
    if str(getattr(config, 'Plasma_geometry', 'Academic')).lower() == 'refined':
        Vprime_data = precompute_Vprime(R0, a, kappa, delta,
                                        geometry_model='refined',
                                        kappa_95=kappa_95, delta_95=delta_95,
                                        N_rho=500, N_theta=200)
    else:
        Vprime_data = None

    # Impurity inventory (frozen species/concentrations from the deck).
    imp_species, imp_conc = RUN._parse_impurity_inventory(config)
    f_imp_dilution = sum(get_Z_mean(s, config.Tbar) * c
                         for s, c in zip(imp_species, imp_conc))

    # Scaling-law registry coefficients and kappa convention, replicating the
    # RUN driver selection: IPB98-class laws were fitted with the AREA
    # elongation kappa_a = V/(2 pi^2 R0 a^2); ITPA20-class with kappa_edge.
    (C_SL, a_d, a_M, a_k, a_e, a_R, a_B, a_n, a_I, a_P) = \
        f_Get_parameter_scaling_law(config.Scaling_Law)
    kappa_a = V / (2.0 * np.pi**2 * R0 * a**2)
    _KAPPA_EDGE_LAWS = {'ITPA20', 'ITPA20-IL'}
    kappa_SL = kappa if config.Scaling_Law in _KAPPA_EDGE_LAWS else kappa_a
    H = config.H

    # Temperature-independent part of the scaling (density factored per point).
    X_geom = (H * C_SL * Ip**a_I * R0**a_R * eps**a_e * kappa_SL**a_k
              * B0**a_B * M**a_M * (1.0 + delta)**a_d)

    # ── Grids ─────────────────────────────────────────────────────────────
    n_lo, n_hi, n_n = grid_spec['nbar_line']
    T_lo, T_hi, n_T = grid_spec['Tbar']
    nbar_line_grid = np.linspace(n_lo, n_hi, n_n)
    Tbar_grid      = np.linspace(T_lo, T_hi, n_T)

    shape = (n_T, n_n)
    out = {k: np.full(shape, np.nan) for k in
           ('P_fus', 'P_alpha', 'P_aux', 'P_Ohm', 'P_loss', 'P_rad_core',
            'P_rad_tot', 'P_sep', 'Q', 'f_LH', 'f_GW', 'f_DL', 'betaN',
            'W_th_MJ', 'tauE')}

    P_REF = 1000.0  # [MW] reference power for the quadratic P_fus inversion

    def _column(it, Tbar):
        # Forward fusion power from the existing inversion: P = C(T) n^2.
        n_ref_vol = f_nbar(P_REF, nu_n, nu_T, f_alpha, Tbar, R0, a, kappa,
                           rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                           T_ped_frac=T_ped_frac, Vprime_data=Vprime_data,
                           f_imp=f_imp_dilution, tau_i_e=tau_ie)
        # Ohmic power at the frozen inductive current.
        Zeff_eff = RUN._compute_Zeff_effective(config, f_alpha)
        P_Ohm = f_P_Ohm(I_Ohm, Tbar, R0, a, kappa, Z_eff=Zeff_eff)

        for jn, nbl in enumerate(nbar_line_grid):
            nbar_vol = f_nbar_vol_from_line(nbl, nu_n, rho_ped=rho_ped,
                                            n_ped_frac=n_ped_frac)
            P_fus = P_REF * (nbar_vol / n_ref_vol)**2
            P_alpha = f_P_alpha(P_fus)

            pbar = f_pbar(nu_n, nu_T, nbar_vol, Tbar,
                          rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                          T_ped_frac=T_ped_frac, Vprime_data=Vprime_data,
                          tau_i_e=tau_ie)
            W_MJ = f_W_th(pbar, V) / 1e6

            # Core/total radiation, mirroring the RUN driver calls.
            Zeff_fuel = 1.0 + 2.0 * f_alpha - f_imp_dilution
            P_Brem = f_P_bremsstrahlung(nbar_vol, Tbar, Zeff_fuel, V,
                                        nu_n, nu_T, rho_ped=rho_ped,
                                        n_ped_frac=n_ped_frac,
                                        T_ped_frac=T_ped_frac)
            P_syn = f_P_synchrotron(Tbar, R0, a, B0, nbar_vol, kappa,
                                    nu_n, nu_T, config.r_synch,
                                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                                    T_ped_frac=T_ped_frac)
            P_line_core, P_line_tot = 0.0, 0.0
            for sp, fc in zip(imp_species, imp_conc):
                _Pc, _Pt = f_P_line_radiation_profile(
                    sp, fc, nbar_vol, Tbar, nu_n, nu_T, V,
                    rho_ped=rho_ped, n_ped_frac=n_ped_frac,
                    T_ped_frac=T_ped_frac, Vprime_data=Vprime_data,
                    rho_core=config.rho_rad_core, N=150)
                P_line_core += _Pc
                P_line_tot += _Pt
            P_rad_core = P_Brem + P_syn + P_line_core
            P_rad_tot  = P_Brem + P_syn + P_line_tot

            # Closed-form confinement balance at fixed machine.
            X = X_geom * (nbl * 10.0)**a_n
            expo = 1.0 + a_P
            if X <= 0 or W_MJ <= 0 or expo <= 0:
                continue
            P_loss = (W_MJ / X)**(1.0 / expo)
            P_aux = (P_loss + config.coreradiationfraction * P_rad_core
                     - P_alpha - P_Ohm)

            P_heat = P_alpha + max(P_aux, 0.0) + P_Ohm
            P_sep = P_heat - P_rad_tot
            Q = P_fus / (P_aux + P_Ohm) if P_aux > 0 else np.inf

            # L-H threshold per the deck selector.
            if config.L_H_Scaling_choice == 'Martin':
                P_LH = P_Thresh_Martin(nbl, B0, a, R0, kappa, M)
            elif config.L_H_Scaling_choice == 'New_S':
                P_LH = P_Thresh_New_S(nbl, B0, a, R0, kappa, M)
            else:
                P_LH = P_Thresh_New_Ip(nbl, B0, a, R0, kappa, Ip, M)

            # Model-selectable density limit (power-dependent ones use P_sep).
            try:
                _fnsl = config.f_n_sep * (nbar_vol / nbl)
                n_DL_line, _, _ = f_density_limit(
                    config.density_limit_model, Ip, a,
                    P_sol=max(P_sep, 1e-3), P_tot=P_heat,
                    R0=R0, kappa=kappa, B0=B0, q_edge=float(res[_IDX['q95']]),
                    Z_eff=Zeff_eff, f_n_sep_line=_fnsl,
                    A_ion=M, alpha_GR=config.alpha_giacomin,
                    f0=config.f0_zanca)
            except ValueError:
                n_DL_line = np.nan

            # Normalised beta.
            betaT = f_beta_T(pbar, B0)
            betaP = f_beta_P(a, kappa, pbar, Ip)
            betaN = f_beta_N(f_beta(betaT, betaP), a, B0, Ip)

            out['P_fus'][it, jn]      = P_fus
            out['P_alpha'][it, jn]    = P_alpha
            out['P_aux'][it, jn]      = P_aux
            out['P_Ohm'][it, jn]      = P_Ohm
            out['P_loss'][it, jn]     = P_loss
            out['P_rad_core'][it, jn] = P_rad_core
            out['P_rad_tot'][it, jn]  = P_rad_tot
            out['P_sep'][it, jn]      = P_sep
            out['Q'][it, jn]          = Q
            out['f_LH'][it, jn]       = P_sep / P_LH if P_LH > 0 else np.nan
            out['f_GW'][it, jn]       = nbl / nG_raw
            out['f_DL'][it, jn]       = (nbl / n_DL_line
                                         if np.isfinite(n_DL_line) else np.nan)
            out['betaN'][it, jn]      = betaN
            out['W_th_MJ'][it, jn]    = W_MJ
            out['tauE'][it, jn]       = W_MJ / P_loss



    # Temperature columns are independent; the threading backend is adequate
    # because the cost is dominated by NumPy profile integrals, which release
    # the GIL. Workers write into disjoint rows of the preallocated arrays.
    # One updating tqdm bar instead of per-column log lines (same pattern as
    # the UQ module): return_as='generator' yields results as they complete.
    _gen = Parallel(n_jobs=-1, backend='threading', return_as='generator')(
        delayed(_column)(it, Tbar) for it, Tbar in enumerate(Tbar_grid))
    for _ in tqdm(_gen, total=len(Tbar_grid), desc="POPCON",
                  unit="col", disable=not verbose):
        pass

    out['nbar_line_grid'] = nbar_line_grid
    out['Tbar_grid']      = Tbar_grid
    out['design'] = dict(B0=B0, Ip=Ip, Volume=V, kappa=kappa, delta=delta,
                         f_alpha=f_alpha, I_Ohm=I_Ohm, nG=nG_raw,
                         R0=R0, a=a, q95=float(res[_IDX['q95']]),
                         nbar_line_design=float(res[_IDX['nbar_line']]),
                         Tbar_design=float(config.Tbar),
                         scaling_law=config.Scaling_Law, H=H,
                         density_limit_model=config.density_limit_model)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_popcon(out, save_path=None, show=True):
    """Render the POPCON map (single figure, cfspopcon-style overlay)."""
    N, T = np.meshgrid(out['nbar_line_grid'], out['Tbar_grid'])
    dsn = out['design']

    fig, ax = plt.subplots(figsize=(10.5, 7.5))

    # Ignition / Q map background.
    Q = out['Q'].copy()
    Qf = np.where(np.isinf(Q), np.nan, Q)
    pcm = ax.pcolormesh(N, T, np.log10(np.clip(Qf, 1e-2, 1e3)),
                        shading='auto', cmap='viridis', alpha=0.75)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.015)
    cbar.set_label(r'log$_{10}$ Q (driven)')

    ign = np.isinf(out['Q']) | (out['P_aux'] <= 0)
    if ign.any():
        ax.contourf(N, T, ign.astype(float), levels=[0.5, 1.5],
                    colors=['gold'], alpha=0.55)
        ax.contour(N, T, ign.astype(float), levels=[0.5],
                   colors='darkorange', linewidths=2.0)
        ax.plot([], [], color='darkorange', lw=2, label='Ignition (P_aux = 0)')

    cs = ax.contour(N, T, out['P_fus'], colors='royalblue',
                    levels=[100, 200, 500, 1000, 2000, 3000, 5000],
                    linewidths=1.1)
    ax.clabel(cs, fmt='%g MW', fontsize=8)
    ax.plot([], [], color='royalblue', lw=1.1, label='P_fus')

    Paux = np.where(out['P_aux'] > 0, out['P_aux'], np.nan)
    cs = ax.contour(N, T, Paux, colors='forestgreen',
                    levels=[10, 25, 50, 100, 200], linewidths=1.0,
                    linestyles='--')
    ax.clabel(cs, fmt='%g MW', fontsize=8)
    ax.plot([], [], color='forestgreen', lw=1.0, ls='--', label='P_aux')

    cs = ax.contour(N, T, out['Q'], colors='k', levels=[1, 2, 5, 10, 20],
                    linewidths=1.0, linestyles=':')
    ax.clabel(cs, fmt='Q=%g', fontsize=8)

    ax.contour(N, T, out['f_LH'], colors='crimson', levels=[1.0],
               linewidths=2.0)
    ax.plot([], [], color='crimson', lw=2, label='P_sep = P_LH')

    ax.contour(N, T, out['f_GW'], colors='purple', levels=[1.0],
               linewidths=2.0, linestyles='-.')
    ax.plot([], [], color='purple', lw=2, ls='-.', label='Greenwald limit')

    if np.isfinite(out['f_DL']).any() and dsn['density_limit_model'] != 'greenwald':
        ax.contour(N, T, out['f_DL'], colors='magenta', levels=[1.0],
                   linewidths=1.8, linestyles=(0, (3, 1, 1, 1)))
        ax.plot([], [], color='magenta', lw=1.8, ls=(0, (3, 1, 1, 1)),
                label=f"Density limit ({dsn['density_limit_model']})")

    ax.plot(dsn['nbar_line_design'], dsn['Tbar_design'], marker='*',
            markersize=16, color='red', markeredgecolor='k',
            label='Design point', linestyle='None')

    ax.set_xlabel(r'Line-averaged density $\bar{n}_{line}$ [$10^{20}$ m$^{-3}$]')
    ax.set_ylabel(r'Volume-averaged temperature $\bar{T}$ [keV]')
    ax.set_title(
        f"POPCON — R0={dsn['R0']:.2f} m, a={dsn['a']:.2f} m, "
        f"B0={dsn['B0']:.2f} T, Ip={dsn['Ip']:.1f} MA, "
        f"{dsn['scaling_law']}, H={dsn['H']:.2f}")
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"POPCON figure saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(input_file, save_figures=True, output_dir=None, show=None):
    """
    POPCON mode entry point (called by D0FUS.py).

    Writes a timestamped folder under D0FUS_OUTPUTS/popcon/ containing the
    figure (PNG), the raw maps (NPZ) and a copy of the deck.
    """
    config, grid_spec = parse_popcon_file(input_file)
    out = compute_popcon(config, grid_spec, verbose=1)

    if output_dir is None:
        output_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'D0FUS_OUTPUTS', 'popcon'))
    # Folder naming mirrors the other execution modes
    # (Run_D0FUS_*, Scan_D0FUS_*, Genetic_D0FUS_*, Sobol_D0FUS_*).
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(output_dir, f"Popcon_D0FUS_{timestamp}")
    os.makedirs(folder, exist_ok=True)

    np.savez(os.path.join(folder, 'popcon_maps.npz'),
             **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
             **{f"design_{k}": v for k, v in out['design'].items()
                if isinstance(v, (int, float))})
    with open(os.path.join(folder, 'deck_copy.txt'), 'w', encoding='utf-8') as f:
        f.write(open(input_file, encoding='utf-8').read())

    # The figure is displayed interactively by default (like the RUN-mode
    # figures), in addition to being saved when save_figures is True.
    if show is None:
        show = True
    fig_path = os.path.join(folder, 'popcon.png') if save_figures else None
    plot_popcon(out, save_path=fig_path, show=show)
    print(f"POPCON outputs written to {folder}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="D0FUS POPCON mode (stand-alone)")
    p.add_argument('deck', help="POPCON deck (RUN deck + [POPCON] section)")
    p.add_argument('--no-show', action='store_true',
                   help="suppress the interactive display (save only)")
    args = p.parse_args()
    main(args.deck, save_figures=True, show=not args.no_show)
