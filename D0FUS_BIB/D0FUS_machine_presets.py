"""
D0FUS_machine_presets.py
========================
Curated 3D machine presets for the D0FUS concept-view renderer.

This module complements ``D0FUS_figures.plot_tokamak_3D`` (which draws the
CONVERGED D0FUS DESIGN of a run dict) with a bank of hand-curated presets
reproducing the geometry of the major fusion machines worldwide, in three
categories:

    operating          : JET, JT-60SA, ASDEX Upgrade, WEST, TCV, EAST,
                         KSTAR, MAST-U, TJ-2, W7X, LHD
    under_construction : ITER, SPARC, BEST
    reactor_project    : EU-DEMO, ARC, CFETR

Every preset is a plain dict (schema below) so users can copy, edit and
render their own variants.  Renders share the D0FUS 3D conventions: one
third toroidal wedge cut away, poloidal cross-section with nested flux
surfaces, pastel palette with black feature edges, and a standing 2 m human
silhouette at floor level as an absolute scale reference.

Preset schema
-------------
::

    {
      "name":     display name,
      "category": "operating" | "under_construction" | "reactor_project",
      "plasma":   {R0, a, kappa, delta [, kappa_95, delta_95]},   # [m], LCFS
      "tf":       {"shape": "d",        n, R_bore, R_out, c}      # Princeton-D
                | {"shape": "circular", n, center_R, r_in, r_out, half_tor}
                | {"shape": "rect",     n, R_in_leg, R_out_leg, Z_half,
                   c, w_tor, corner_r}                            # window pane
                | {"shape": "spherical", n_limbs, rod_r, rod_z, limb_R,
                   limb_z, limb_half_r, limb_half_t},             # ST cage
      "solenoids":[{r_in, r_out, z_lo, z_hi, modules, color, label}, ...],
      "pf":       [{R, Z, dR, dZ}, ...] or {"auto": n},           # ring coils
      "pf_label": legend label for the PF group,
      "div":      [{R, Z, dR, dZ}, ...],                          # in-vessel /
      "div_label": legend label,                                  # Cu coils
      "iron":     {core_r, core_z, n_limbs, limb_R, limb_z, ...}, # JET yoke
      "sources":  [citation strings],
      "notes":    free text (assumptions and confidence flags),
    }

Geometry provenance
-------------------
All dimensions are in METRES and are documented per machine in the
``sources`` / ``notes`` fields.  Three confidence levels are used:

  * published : value taken verbatim from an accessible primary source
                (machine-description files, design papers, lab pages);
  * derived   : computed from published values (e.g. coil radius from a
                published diameter);
  * drawn     : illustrative placement where no public table exists (always
                flagged in ``notes``).

PyVista (VTK) is an OPTIONAL dependency, exactly as for plot_tokamak_3D.

Author  : Auclair Timothe
Created : June 2026
"""

import os

# Package-relative imports (production mode)
if __name__ != "__main__":
    from .D0FUS_import import *
    from .D0FUS_physical_functions import kappa_profile, delta_profile
    from .D0FUS_radial_build_functions import (_princeton_D_contour,
                                               _offset_contour)
    from .D0FUS_figures import (
        _pv3d_keep_range, _pv3d_grid, _pv3d_rect_tube, _pv3d_add_edges,
        _pv3d_add_tube, _pv3d_add_cap, _pv3d_cyl_sector, _pv3d_camera,
        _pv3d_autotrim, _pv3d_legend_strip, _pv3d_stack_legend,
        _pv3d_d_centreline, _pv3d_add_human,
        _PV3D_PLASMA_COL, _PV3D_TF_COL, _PV3D_CS_COL, _PV3D_PF_COL,
        _PV3D_EDGE_COL, _PV3D_EDGE_LW, _PV3D_FLUX_COL, _PV3D_FLUX_FRACS,
        _PV3D_HUMAN_COL, _PV3D_HUMAN_H,
        _PV3D_GAP_CENTER_DEG, _PV3D_GAP_HALF_DEG,
    )
# Standalone-execution imports (development / testing)
else:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from D0FUS_BIB.D0FUS_import import *
    from D0FUS_BIB.D0FUS_physical_functions import kappa_profile, delta_profile
    from D0FUS_BIB.D0FUS_radial_build_functions import (_princeton_D_contour,
                                                        _offset_contour)
    from D0FUS_BIB.D0FUS_figures import (
        _pv3d_keep_range, _pv3d_grid, _pv3d_rect_tube, _pv3d_add_edges,
        _pv3d_add_tube, _pv3d_add_cap, _pv3d_cyl_sector, _pv3d_camera,
        _pv3d_autotrim, _pv3d_legend_strip, _pv3d_stack_legend,
        _pv3d_d_centreline, _pv3d_add_human,
        _PV3D_PLASMA_COL, _PV3D_TF_COL, _PV3D_CS_COL, _PV3D_PF_COL,
        _PV3D_EDGE_COL, _PV3D_EDGE_LW, _PV3D_FLUX_COL, _PV3D_FLUX_FRACS,
        _PV3D_HUMAN_COL, _PV3D_HUMAN_H,
        _PV3D_GAP_CENTER_DEG, _PV3D_GAP_HALF_DEG,
    )

# Additional palette entries specific to the presets
_PV3D_DIV_COL  = "#eccfa5"      # copper tint: in-vessel / divertor coils
_PV3D_IRON_COL = "#c6cbd2"      # steel grey : iron transformer yoke (JET)

_COLOR_MAP = {"cs": None, "pf": None, "div": None, "iron": None, "tf": None}


def _col(key):
    """Resolve a schematic colour key to its palette hex value."""
    return {"cs": _PV3D_CS_COL, "pf": _PV3D_PF_COL, "div": _PV3D_DIV_COL,
            "iron": _PV3D_IRON_COL, "tf": _PV3D_TF_COL}[key]


# =============================================================================
# Preset bank
# =============================================================================

PRESETS = {

# ─────────────────────────── Operating machines ──────────────────────────────

"WEST": {
    "name": "WEST (CEA Cadarache, France)",
    "category": "operating",
    "plasma": {"R0": 2.5, "a": 0.5, "kappa": 1.4, "delta": 0.5},
    # 18 CIRCULAR superconducting NbTi coils from Tore Supra: winding
    # inner/outer diameters 2.3 / 2.8 m (ASG data sheet), centres on the
    # R = 2.4 m circle (CEA field-torus description).  Toroidal thickness of
    # the winding pack is not published: drawn equal to the radial build.
    "tf": {"shape": "circular", "n": 18, "center_R": 2.40,
           "r_in": 1.15, "r_out": 1.40, "half_tor": 0.125},
    # Central column stack Bb | A | Bh, verbatim from the tofu (CEA) machine
    # description; the visible segmentation of the column is this stack.
    "solenoids": [
        {"r_in": 0.669, "r_out": 0.812, "z_lo": -0.845, "z_hi": 0.875,
         "modules": 1, "color": "cs", "label": "Central solenoid A"},
        {"r_in": 1.082, "r_out": 1.155, "z_lo": 1.200, "z_hi": 2.400,
         "modules": 1, "color": "pf", "label": "PF coils (B, D, E, F)"},
        {"r_in": 1.082, "r_out": 1.155, "z_lo": -2.400, "z_hi": -1.200,
         "modules": 1, "color": "pf", "label": "PF coils (B, D, E, F)"},
    ],
    "pf": [  # D, E, F pairs, verbatim from tofu (published)
        {"R": 2.8815, "Z":  1.9275, "dR": 0.267, "dZ": 0.363},
        {"R": 2.8815, "Z": -1.9275, "dR": 0.267, "dZ": 0.363},
        {"R": 3.7715, "Z":  1.5410, "dR": 0.287, "dZ": 0.390},
        {"R": 3.7715, "Z": -1.5410, "dR": 0.287, "dZ": 0.390},
        {"R": 4.3745, "Z":  0.6440, "dR": 0.289, "dZ": 0.388},
        {"R": 4.3745, "Z": -0.6460, "dR": 0.289, "dZ": 0.388},
    ],
    "pf_label": "PF coils (B, D, E, F)",
    "div": [  # WEST in-vessel Cu divertor coils, element bounding boxes (tofu)
        {"R": 2.0435, "Z":  0.7602, "dR": 0.1254, "dZ": 0.0994},
        {"R": 2.1985, "Z":  0.8228, "dR": 0.1254, "dZ": 0.0994},
        {"R": 2.0435, "Z": -0.7573, "dR": 0.1254, "dZ": 0.0994},
        {"R": 2.1985, "Z": -0.8186, "dR": 0.1254, "dZ": 0.0994},
    ],
    "div_label": "Divertor coils (in-vessel)",
    "sources": [
        "IRFM/CEA, WEST main plasma parameters (R0 2.5, a 0.5, kappa "
        "1.3-1.8, delta 0.5-0.6)",
        "ASG Superconductors, Tore Supra data sheet (18 circular TFC, "
        "winding Oi 2.3 m / Oe 2.8 m)",
        "CEA Tore Supra pages (2028 turns/coil, field torus R 2.4 m; "
        "PF set A, Bb, Bh, Db, Dh, Eb, Eh, Fb, Fh)",
        "tofu (ToFuProject, CEA): TFG_CoilPF_ExpWEST_*.txt coil polygons",
    ],
    "notes": "PF/CS/divertor geometry published (tofu); TF toroidal "
             "thickness drawn equal to the radial build.",
},

"JET": {
    "name": "JET (Culham, UK)",
    "category": "operating",
    # Published plasma: R0 2.96 m, a 1.25 m (horizontal), kappa ~1.6-1.8.
    # The horizontal radius is drawn at 1.10 m so the up-down symmetric
    # Miller shape represents a DIVERTED plasma clear of the in-vessel
    # divertor coils (drawn), as in divertor operation since 1994.
    "plasma": {"R0": 2.96, "a": 1.10, "kappa": 1.60, "delta": 0.35},
    # TF contour anchored to the published effective leg radii of the JET
    # TF coils, R1 = 1.373 m and R2 = 4.733 m (JET-R(99)10), stretched
    # vertically to the published overall coil height of 5.68 m (one coil
    # is 3.86 m wide x 5.68 m tall, 12 t, EUR-JET-R7 p. 25).
    "tf": {"shape": "d", "n": 32, "R_bore": 1.16, "R_out": 4.94, "c": 0.42,
           "z_scale": 0.916},
    "solenoids": [
        # Primary winding P1: published mean diameter 1.81 m, 10 sections.
        {"r_in": 0.80, "r_out": 1.01, "z_lo": -2.5, "z_hi": 2.5,
         "modules": 10, "color": "cs",
         "label": "Primary winding P1 (10 sections)"},
    ],
    # Iron transformer: central limb + 8 return limbs (~2600 t), the JET
    # signature.  Limb dimensions drawn (machine ~15 m diameter published).
    "iron": {"core_r": 0.55, "core_z": 4.0, "n_limbs": 8, "limb_R": 6.55,
             "limb_z": 4.0, "limb_half_r": 0.30, "limb_half_t": 0.45,
             "beam_z": 3.75, "beam_half": 0.28},
    "pf": [  # P2/P3/P4: ALL outside the TF coils ("The poloidal field
             # coils are outside the toroidal field coils", EUR-JET-R7
             # p. 20, verbatim), mounted around the mechanical shell.
             # Design mean diameters 4.3 / 7.88 / 10.49 m (EUR-JET-R7);
             # vertical positions drawn along the shell (not published).
        {"R": 2.150, "Z":  3.15, "dR": 0.42, "dZ": 0.42},
        {"R": 2.150, "Z": -3.15, "dR": 0.42, "dZ": 0.42},
        {"R": 3.940, "Z":  2.65, "dR": 0.45, "dZ": 0.45},
        {"R": 3.940, "Z": -2.65, "dR": 0.45, "dZ": 0.45},
        {"R": 5.245, "Z":  1.50, "dR": 0.55, "dZ": 0.55},
        {"R": 5.245, "Z": -1.50, "dR": 0.55, "dZ": 0.55},
    ],
    "pf_label": "PF coils P2-P4 (outside the TF cage)",
    "div": [  # 4 in-vessel Cu divertor coils, published mean radii 2.1,
             # 2.5, 2.8, 3.4 m (JET divertor coils paper); Z drawn.
        {"R": 2.10, "Z": -1.72, "dR": 0.25, "dZ": 0.20},
        {"R": 2.50, "Z": -1.88, "dR": 0.25, "dZ": 0.20},
        {"R": 2.80, "Z": -1.88, "dR": 0.25, "dZ": 0.20},
        {"R": 3.40, "Z": -1.62, "dR": 0.25, "dZ": 0.20},
    ],
    "div_label": "Divertor coils (in-vessel)",
    "sources": [
        "EUR-JET-R7, The JET Project (P1/P2/P3/P4 mean diameters 1.81 / "
        "4.3 / 7.88 / 10.49 m; PF coils OUTSIDE the TF coils, p. 20; one "
        "TF coil 3.86 m wide x 5.68 m tall, 12 t, p. 25; 32 D-shaped Cu "
        "TF coils; 8-limb iron transformer; primary in 10 sections)",
        "JET-R(99)10 (TF effective leg radii R1 1.373 m, R2 4.733 m)",
        "The JET divertor coils (D1-D4 mean radii 2.1/2.5/2.8/3.4 m, "
        "in-vessel, water-cooled Cu)",
        "EPN 13(4) 1983 and EPN 23(7) 1992 (design and pumped divertor)",
    ],
    "notes": "Coil radii, TF coil size and inside/outside topology "
             "published; P2-P4 vertical positions and iron-yoke "
             "dimensions drawn from cross-section figures.  Plasma drawn "
             "slightly reduced (a 1.10) to represent the diverted shape.",
},

"JT-60SA": {
    "name": "JT-60SA (Naka, Japan)",
    "category": "operating",
    "plasma": {"R0": 2.96, "a": 1.18, "kappa": 1.90, "delta": 0.50},
    # 18 D-shaped NbTi coils, ~7.5 m x 4.6 m each (QST): contour drawn to
    # match the published width, compressed vertically (z_scale) so the
    # winding stays within the real coil height and every EF coil sits
    # OUTSIDE the cage, as on the machine.
    "tf": {"shape": "d", "n": 18, "R_bore": 1.15, "R_out": 5.75, "c": 0.50,
           "z_scale": 0.84},
    "solenoids": [
        # CS: 4 modules, winding radius 0.824 m, dR 0.340 m, module height
        # 1.585 m (Yoshida 2010, Table 4).
        {"r_in": 0.654, "r_out": 0.994, "z_lo": -3.17, "z_hi": 3.17,
         "modules": 4, "color": "cs",
         "label": "Central solenoid (4 modules, Nb3Sn)"},
    ],
    "pf": [  # EF1-EF6: winding radii and sections published (Yoshida 2010,
             # Table 5: R 5.819/4.621/1.919/1.919/3.914/5.054 m); vertical
             # positions drawn from the machine cross-section (LSN layout).
        {"R": 5.819, "Z":  1.85, "dR": 0.343, "dZ": 0.347},   # EF1
        {"R": 4.621, "Z":  3.10, "dR": 0.370, "dZ": 0.347},   # EF2
        {"R": 1.919, "Z":  4.10, "dR": 0.556, "dZ": 0.441},   # EF3
        {"R": 1.919, "Z": -4.10, "dR": 0.556, "dZ": 0.625},   # EF4
        {"R": 3.914, "Z": -3.45, "dR": 0.315, "dZ": 0.403},   # EF5
        {"R": 5.054, "Z": -2.45, "dR": 0.370, "dZ": 0.403},   # EF6
    ],
    "pf_label": "EF coils EF1-EF6 (NbTi)",
    "sources": [
        "Yoshida et al., J. Plasma Fusion Res. SERIES 9 (2010), Tables 4-5 "
        "(EF winding radii 5.819/4.621/1.919/1.919/3.914/5.054 m with "
        "sections; CS 4 modules, winding radius 0.824 m, dR 0.34 m, "
        "module height 1.585 m)",
        "QST JT-60SA device pages (18 TF NbTi 7.5 x 4.6 m)",
        "IAEA FEC OV/P-4 (R0 2.96, a 1.18, kappa ~2.0, delta ~0.53, LSN)",
    ],
    "notes": "EF radii, sections and CS build published; EF vertical "
             "positions drawn from cross-section figures.",
},

"ASDEX-U": {
    "name": "ASDEX Upgrade (IPP Garching, Germany)",
    "category": "operating",
    "plasma": {"R0": 1.65, "a": 0.50, "kappa": 1.60, "delta": 0.40},
    "tf": {"shape": "d", "n": 16, "R_bore": 0.62, "R_out": 3.05, "c": 0.38},
    "solenoids": [
        {"r_in": 0.32, "r_out": 0.56, "z_lo": -1.55, "z_hi": 1.55,
         "modules": 1, "color": "cs", "label": "OH solenoid"},
    ],
    "pf": {"auto": 8},
    "pf_label": "PF coils (illustrative positions)",
    "sources": [
        "IAEA FEC 2025 AUG overview (R0 1.65, a 0.5, Ip <= 1.4 MA, "
        "Bt <= 3.5 T)",
        "IPP device page (minor radii 0.5 / 0.8 m, 16 Cu TF coils, "
        "machine 9 m high x 10 m diameter)",
    ],
    "notes": "TF contour and coil positions drawn (no public coil table). "
             "The 2 x 8 in-vessel saddle coils (RMP) are not drawn.",
},

"TCV": {
    "name": "TCV (EPFL Lausanne, Switzerland)",
    "category": "operating",
    # kappa up to 2.8 published; a typical shaped discharge is drawn.
    "plasma": {"R0": 0.88, "a": 0.25, "kappa": 1.60, "delta": 0.35},
    # The 16 TF coils "surround all the ohmic and shaping coils" (Moret et
    # al., verbatim), so the WHOLE air-core coil set A/C/D/E/F sits inside
    # the TF bore, which must clear the C/D rings at |Z| = 1.17 m.  The
    # coil is therefore drawn as a tall rounded rectangle (~1.4 m wide x
    # 2.8 m tall overall, about twice the vessel height), not a
    # Princeton-D; the exact coil shape is not published (inferred).
    "tf": {"shape": "rect", "n": 16, "R_in_leg": 0.28, "R_out_leg": 1.55,
           "Z_half": 1.32, "c": 0.16, "corner_r": 0.30},
    "solenoids": [
        {"r_in": 0.400, "r_out": 0.460, "z_lo": -0.93, "z_hi": 0.93,
         "modules": 1, "color": "cs",
         "label": "OH circuit A, C, D (air core)"},
        {"r_in": 0.470, "r_out": 0.540, "z_lo": -0.78, "z_hi": 0.78,
         "modules": 8, "color": "pf",
         "label": "Shaping coils E1-E8 (R 0.505 m)"},
    ],
    "pf": [  # F1-F8 at R 1.3095 m (FreeGS TCV machine description) plus
             # the C and D OH-circuit rings, drawn in the OH colour.
        {"R": 1.3095, "Z":  0.77, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z":  0.61, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z":  0.31, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z":  0.15, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z": -0.15, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z": -0.31, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z": -0.61, "dR": 0.10, "dZ": 0.12},
        {"R": 1.3095, "Z": -0.77, "dR": 0.10, "dZ": 0.12},
        {"R": 0.6215, "Z":  1.110, "dR": 0.10, "dZ": 0.10, "color": "cs"},
        {"R": 0.6215, "Z": -1.110, "dR": 0.10, "dZ": 0.10, "color": "cs"},
        {"R": 1.1765, "Z":  1.170, "dR": 0.10, "dZ": 0.10, "color": "cs"},
        {"R": 1.1765, "Z": -1.170, "dR": 0.10, "dZ": 0.10, "color": "cs"},
    ],
    "pf_label": "Shaping coils F1-F8 (R 1.31 m)",
    "sources": [
        "EPFL/SPC TCV overview (R0 0.88, a 0.25, kappa up to 2.8, "
        "delta -0.7 to +1, Bt 1.5 T, Ip up to 1 MA)",
        "Moret et al. via OSTI (16 TF coils; air-core OH stacks A-D; "
        "16 shaping coils in two stacks of eight, E and F)",
        "FreeGS TCV machine description (E1-E8 at R 0.505, Z -0.7..0.7; "
        "F1-F8 at R 1.3095, Z +/-0.15/0.31/0.61/0.77; solenoid A R 0.43, "
        "Z +/-0.93; C at (0.6215, +/-1.11); D at (1.1765, +/-1.17))",
        "Moret et al. (verbatim: the toroidal field is produced by 16 "
        "coils surrounding all the ohmic and shaping coils); Hofmann 1994 "
        "(shaping coils located between the vacuum vessel and the TF "
        "coils)",
    ],
    "notes": "Coil R/Z from the FreeGS machine description (coil sections "
             "drawn).  TF coil shape inferred (rounded rectangle sized to "
             "enclose the published coil set; no public TF table).  Fast "
             "internal VS coils not drawn.  Drawn plasma is a typical "
             "shaped discharge, not the extreme kappa = 2.8.",
},

"EAST": {
    "name": "EAST (ASIPP Hefei, China)",
    "category": "operating",
    "plasma": {"R0": 1.85, "a": 0.45, "kappa": 1.80, "delta": 0.60},
    "tf": {"shape": "d", "n": 16, "R_bore": 0.60, "R_out": 3.05, "c": 0.40},
    "solenoids": [
        {"r_in": 0.28, "r_out": 0.56, "z_lo": -1.55, "z_hi": 1.55,
         "modules": 6, "color": "cs",
         "label": "Central solenoid (6 modules, NbTi)"},
    ],
    "pf": [  # three pairs of large NbTi rings; largest diameter 7.6 m
        {"R": 1.65, "Z":  2.45, "dR": 0.42, "dZ": 0.42},   # (published);
        {"R": 1.65, "Z": -2.45, "dR": 0.42, "dZ": 0.42},   # heights drawn
        {"R": 3.30, "Z":  1.55, "dR": 0.45, "dZ": 0.45},
        {"R": 3.30, "Z": -1.55, "dR": 0.45, "dZ": 0.45},
        {"R": 3.85, "Z":  0.60, "dR": 0.45, "dZ": 0.45},
        {"R": 3.85, "Z": -0.60, "dR": 0.45, "dZ": 0.45},
    ],
    "pf_label": "PF coils (3 pairs, NbTi)",
    "sources": [
        "ASIPP slides (16 NbTi D-shaped TF coils; 6 CS coils; 3 pairs of "
        "large PF rings, largest 7.6 m diameter; machine ~10 m x 7.6 m)",
        "Wikipedia EAST (R0 1.85, a 0.45, kappa 1.6-2, delta 0.6-0.8, "
        "Bt 3.5 T, Ip 1 MA)",
    ],
    "notes": "PF ring radii anchored to the published largest diameter; "
             "vertical positions drawn.  The 16 in-vessel RMP coils are "
             "not drawn.",
},

"KSTAR": {
    "name": "KSTAR (KFE Daejeon, South Korea)",
    "category": "operating",
    "plasma": {"R0": 1.80, "a": 0.50, "kappa": 2.00, "delta": 0.70},
    # 16 Nb3Sn D-shaped coils ~3.0 m x 4.2 m (published): contour drawn to
    # the published width, inner leg just outside the CS stack, compressed
    # vertically to the published coil height so PF5-PF7 sit outside the
    # cage, as on the machine.
    "tf": {"shape": "d", "n": 16, "R_bore": 0.75, "R_out": 3.75, "c": 0.42,
           "z_scale": 0.84},
    "solenoids": [
        # CS stack = PF1-PF4 U/L: R 0.570 m, dR 0.2305 m, Z up to 1.415 m
        # (Oh et al. 2012, Table 3).
        {"r_in": 0.455, "r_out": 0.685, "z_lo": -1.415, "z_hi": 1.415,
         "modules": 8, "color": "cs",
         "label": "CS stack PF1-PF4 (8 coils, Nb3Sn)"},
    ],
    "pf": [  # PF5-PF7 pairs, published table (Oh et al. 2012, Table 3)
        {"R": 1.073, "Z":  2.295, "dR": 0.326, "dZ": 0.398},   # PF5
        {"R": 1.073, "Z": -2.295, "dR": 0.326, "dZ": 0.398},
        {"R": 3.090, "Z":  1.920, "dR": 0.207, "dZ": 0.398},   # PF6
        {"R": 3.090, "Z": -1.920, "dR": 0.207, "dZ": 0.398},
        {"R": 3.730, "Z":  0.980, "dR": 0.159, "dZ": 0.302},   # PF7
        {"R": 3.730, "Z": -0.980, "dR": 0.159, "dZ": 0.302},
    ],
    "pf_label": "PF coils PF5-PF7 (design values)",
    "div": [  # in-vessel control coils (Cu), upper/lower sets at R ~2.5 m
        {"R": 2.45, "Z":  0.95, "dR": 0.20, "dZ": 0.14},
        {"R": 2.45, "Z": -0.95, "dR": 0.20, "dZ": 0.14},
    ],
    "div_label": "In-vessel control coils (Cu)",
    "sources": [
        "Oh et al., InTech 2012 (Table 3: PF1-PF4 at R 0.570 m forming the "
        "CS stack to Z 1.415 m; PF5 (1.073, 2.295); PF6 (3.090, 1.920); "
        "PF7 (3.730, 0.980); sections verbatim)",
        "KSTAR IAEA design papers via OSTI/PPPL (R0 1.8, a 0.5, kappa 2.0, "
        "delta 0.8, DN; 16 Nb3Sn TF coils 3.0 x 4.2 m)",
        "KFE device page (machine 9.4 m diameter x 9.6 m high)",
    ],
    "notes": "PF table published (PF5 vertical position read as 2.295 m; "
             "the source table digit grouping is ambiguous there).  IVCC "
             "positions drawn (coil radius 2.5 m published).",
},

"MAST-U": {
    "name": "MAST-U (UKAEA Culham, UK)",
    "category": "operating",
    # Spherical tokamak: published R0 0.85, a 0.65, kappa ~2.5.  The minor
    # radius is drawn at 0.61 m so the symmetric Miller LCFS clears the
    # centre-column solenoid and coils.
    "plasma": {"R0": 0.85, "a": 0.61, "kappa": 2.45, "delta": 0.45},
    # TF cage of a spherical tokamak: solid centre-column rod plus return
    # limbs (MAST: 24 conductors in 12 pairs).  Rod/limb sections drawn.
    "tf": {"shape": "spherical", "n_limbs": 12, "rod_r": 0.125,
           "rod_z": 2.35, "limb_R": 2.15, "limb_z": 2.15,
           "limb_half_r": 0.09, "limb_half_t": 0.14},
    "solenoids": [
        {"r_in": 0.160, "r_out": 0.205, "z_lo": -1.581, "z_hi": 1.581,
         "modules": 1, "color": "cs", "label": "Solenoid P1 (air core)"},
    ],
    "pf": [  # P4 / P5 / P6 pairs, verbatim from the FreeGS MAST-U machine
        {"R": 1.5000, "Z":  1.0591, "dR": 0.130, "dZ": 0.038},
        {"R": 1.5000, "Z": -1.0591, "dR": 0.130, "dZ": 0.038},
        {"R": 1.5975, "Z":  0.3522, "dR": 0.025, "dZ": 0.117},
        {"R": 1.5975, "Z": -0.3522, "dR": 0.025, "dZ": 0.117},
        {"R": 1.3216, "Z":  0.9435, "dR": 0.066, "dZ": 0.106},
        {"R": 1.3216, "Z": -0.9435, "dR": 0.066, "dZ": 0.106},
    ],
    "pf_label": "PF coils P4-P6 (pairs)",
    "div": [  # divertor / shaping in-vessel coils, verbatim from FreeGS
        {"R": 0.2415, "Z":  1.2335, "dR": 0.014, "dZ": 0.366},
        {"R": 0.2415, "Z": -1.2335, "dR": 0.014, "dZ": 0.366},
        {"R": 0.3896, "Z":  1.5725, "dR": 0.074, "dZ": 0.074},
        {"R": 0.3896, "Z": -1.5725, "dR": 0.074, "dZ": 0.074},
        {"R": 0.5645, "Z":  1.7350, "dR": 0.074, "dZ": 0.044},
        {"R": 0.5645, "Z": -1.7350, "dR": 0.074, "dZ": 0.044},
        {"R": 0.7992, "Z":  1.9821, "dR": 0.059, "dZ": 0.044},
        {"R": 0.7992, "Z": -1.9821, "dR": 0.059, "dZ": 0.044},
        {"R": 0.9108, "Z":  1.4937, "dR": 0.044, "dZ": 0.015},
        {"R": 0.9108, "Z": -1.4937, "dR": 0.044, "dZ": 0.015},
        {"R": 1.2777, "Z":  1.4677, "dR": 0.059, "dZ": 0.044},
        {"R": 1.2777, "Z": -1.4677, "dR": 0.059, "dZ": 0.044},
        {"R": 1.5127, "Z":  1.4677, "dR": 0.059, "dZ": 0.044},
        {"R": 1.5127, "Z": -1.4677, "dR": 0.059, "dZ": 0.044},
        {"R": 1.8927, "Z":  1.9500, "dR": 0.030, "dZ": 0.088},
        {"R": 1.8927, "Z": -1.9500, "dR": 0.030, "dZ": 0.088},
    ],
    "div_label": "Divertor coils Px, D1-D7 (in-vessel)",
    "sources": [
        "Harrison et al., Nucl. Fusion 59 (2019) 112011 (R0 0.85, a 0.65, "
        "Super-X divertor, enlarged solenoid)",
        "FreeGS machine description, MASTU() (solenoid R 0.19475 m, "
        "Z +/-1.581 m; Px, D1-D7, Dp, P4, P5, P6 coil set, verbatim R/Z)",
        "Darke et al., MAST design (TF: 24 conductors in 12 pairs, centre "
        "rod + return limbs; vessel ~4 m diameter x 4.4 m)",
    ],
    "notes": "Coil R/Z published (FreeGS); rod and return-limb sections "
             "drawn.  Coil pairs Dp and D5-D7 grouped under the divertor "
             "label.  a drawn 0.61 (published 0.65) for column clearance.",
},

"W7-X": {
    "name": "Wendelstein 7-X (IPP Greifswald, Germany)",
    "category": "operating",
    # STELLARATOR preset (the counterpoint of the tokamak bank): a single
    # set of shaped coils confines the plasma, with no CS and no PF rings.
    "type": "stellarator",
    "plasma": {"R0": 5.5, "a": 0.50},
    "stellarator": {
        # Schematic five-field-period Helias boundary: lab-frame Fourier
        # surface R = R0 + ax*cos(5u) + a*cos(t) + h1*cos(t-5u)
        # + h2*cos(2t-5u) (same closure as the D0FUS concept generator),
        # amplitudes chosen so the mean minor radius matches the published
        # effective a of about 0.53 m.
        "n_fp": 5, "a": 0.50, "h1": 0.23, "h2": 0.09, "axis_swing": 0.20,
        # 50 modular non-planar coils (published count), one identical
        # slightly elliptical loop per coil, tilted with the helical phase
        # of the field period; sections drawn.
        "n_modular": 50, "coil_rc": 1.30, "coil_ell": 0.92,
        "coil_wob": 0.05, "coil_sad": 0.06, "coil_half": 0.10,
        # 20 planar coils (published count), larger circular loops
        # interleaved between the modular coils, alternately tilted.
        "n_planar": 20, "planar_rc": 1.70, "planar_half": 0.09,
        "planar_tilt_deg": 12.0,
    },
    "sources": [
        "IPP Greifswald, W7-X pages (50 non-planar superconducting "
        "modular coils + 20 planar coils, NbTi, five-field-period Helias, "
        "B up to 3 T, cryostat 16 m diameter)",
        "IPP technical data (major plasma radius 5.5 m, effective minor "
        "radius ~0.53 m, plasma volume ~30 m3)",
    ],
    "notes": "Coil counts, R0 and a published.  The boundary shape and "
             "the coil shapes are SCHEMATIC (low-order Fourier surface "
             "and identical tilted loops, not the real five coil types); "
             "trim and divertor sweep coils not drawn.",
},

"LHD": {
    "name": "LHD (NIFS Toki, Japan)",
    "category": "operating",
    "type": "stellarator",
    "plasma": {"R0": 3.9, "a": 0.60},
    # Heliotron: TWO continuous superconducting helical coils (l = 2,
    # m = 10) at coil minor radius 0.975 m (published), rotating-ellipse
    # plasma with semi-axes chosen so the mean minor radius matches the
    # published average a of 0.6 m.
    "stellarator": {"family": "heliotron", "n_fp": 10, "n_rot": 5,
                    "a1": 0.75, "a2": 0.45,
                    "coil_ac": 0.975, "coil_half": 0.17},
    "pf": [  # 3 superconducting ring pairs; RADII published (1.80 / 2.82
             # / 5.55 m), vertical positions drawn from design figures.
        {"R": 1.80, "Z":  0.80, "dR": 0.30, "dZ": 0.25},   # IV
        {"R": 1.80, "Z": -0.80, "dR": 0.30, "dZ": 0.25},
        {"R": 2.82, "Z":  2.00, "dR": 0.32, "dZ": 0.28},   # IS
        {"R": 2.82, "Z": -2.00, "dR": 0.32, "dZ": 0.28},
        {"R": 5.55, "Z":  1.55, "dR": 0.40, "dZ": 0.30},   # OV
        {"R": 5.55, "Z": -1.55, "dR": 0.40, "dZ": 0.30},
    ],
    "pf_label": "PF coils IV / IS / OV (3 pairs, NbTi)",
    "sources": [
        "NIFS LHD pages (R0 3.9 m, average a 0.6 m, plasma volume 30 m3, "
        "B up to 3 T, machine 13.5 m diameter x 9.1 m; operation "
        "completed 12/2025)",
        "LHD engineering summary via OSTI (l/m = 2/10 superconducting "
        "helical coils, coil minor radius 0.975 m, Al-stabilised NbTi; "
        "PF pairs IV/IS/OV at R 1.80/2.82/5.55 m, NbTi CICC)",
    ],
    "notes": "Helical coil radius and PF radii published; PF vertical "
             "positions, winding-pack sections and the rotating-ellipse "
             "semi-axes are drawn (a1 x a2 = 0.75 x 0.45 m so the mean "
             "radius matches the published 0.6 m).",
},

"TJ-II": {
    "name": "TJ-II (CIEMAT Madrid, Spain)",
    "category": "operating",
    "type": "stellarator",
    "plasma": {"R0": 1.5, "a": 0.19},
    # Flexible heliac, 4 field periods: the plasma winds helically around
    # the central circular conductor; 32 planar circular TF coils follow
    # the helical axis (28 of r 0.425 m + 4 of 0.475 m, drawn identical);
    # central circular coil (CC) plus helical winding (HX, swing 0.07 m);
    # one vertical-field coil pair at R 2.25 m.
    "stellarator": {"family": "heliac", "n_fp": 4, "a": 0.19,
                    "axis_swing": 0.28,
                    "n_tf": 32, "tf_r": 0.425, "tf_half": 0.045,
                    "cc_half": 0.05, "hx_swing": 0.07, "hx_half": 0.028,
                    "vf_R": 2.25, "vf_Z": 0.65, "vf_half": 0.06},
    "sources": [
        "FusionWiki TJ-II and TJ-II:Coil system (R0 1.5 m, a < 0.22 m, "
        "4 periods, 32 TF coils: 28 x r 0.425 m + 4 x 0.475 m; CC ring "
        "R 1.5 m; HX helical winding swing 0.07 m; VF pair R 2.25 m; "
        "water-cooled Cu, B0 ~1 T)",
        "CIEMAT TJ-II pages (flexible heliac, bean-shaped plasma, "
        "operating since 1997)",
    ],
    "notes": "Coil counts and radii published (FusionWiki).  The axis "
             "helix excursion (0.28 m) is the design-paper value, not "
             "openly re-verifiable; VF vertical positions drawn; the "
             "bean-shaped section is drawn circular (a 0.19 m); the 4 "
             "larger TF coils are not distinguished.",
},

# ─────────────────────── Machines under construction ─────────────────────────

"ITER": {
    "name": "ITER (Cadarache, France)",
    "category": "under_construction",
    "plasma": {"R0": 6.2, "a": 2.0, "kappa": 1.85, "delta": 0.485,
               "kappa_95": 1.70, "delta_95": 0.33},
    # 18 D-shaped Nb3Sn coils, 17 m x 9 m envelope (published): winding
    # contour drawn inside the envelope; leg radii tuned so the Princeton-D
    # winding height (~13.1 m) matches the real ITER winding pack and the
    # published PF centroids sit outside the winding, as on the machine.
    "tf": {"shape": "d", "n": 18, "R_bore": 2.75, "R_out": 10.90, "c": 0.85},
    "solenoids": [
        {"r_in": 1.32, "r_out": 2.10, "z_lo": -6.35, "z_hi": 6.35,
         "modules": 6, "color": "cs",
         "label": "Central solenoid (6 modules, Nb3Sn)"},
    ],
    "pf": [  # PF1-PF6 centroids (equilibrium literature), cross-checked
        {"R":  3.94, "Z":  7.56, "dR": 0.96, "dZ": 0.98},   # against the
        {"R":  8.28, "Z":  6.53, "dR": 0.65, "dZ": 0.60},   # published coil
        {"R": 11.99, "Z":  3.26, "dR": 0.70, "dZ": 1.10},   # diameters
        {"R": 11.96, "Z": -2.23, "dR": 0.65, "dZ": 1.10},   # 8/17/24/24/17/
        {"R":  8.39, "Z": -6.73, "dR": 0.80, "dZ": 0.95},   # 9-10 m
        {"R":  4.33, "Z": -7.47, "dR": 1.60, "dZ": 1.00},
    ],
    "pf_label": "PF coils PF1-PF6 (NbTi)",
    "sources": [
        "ITER-FEAT ODR Table 4.1 (R0 6.2, a 2.0, kappa 1.70/1.85, delta "
        "0.33/0.49, Ip 15 MA, Bt 5.3 T, single null)",
        "iter.org machine pages (18 D-shaped Nb3Sn TF coils 17 x 9 m; CS: "
        "6 modules, stack 13 m x 4.25 m; PF diameters 8 to 24 m)",
    ],
    "notes": "PF centroids from widely reproduced equilibrium tables "
             "(medium confidence), diameters cross-checked.  VS and 27 ELM "
             "in-vessel coils not drawn.  kappa/delta at separatrix.",
},

"SPARC": {
    "name": "SPARC (CFS, Devens, USA)",
    "category": "under_construction",
    "plasma": {"R0": 1.85, "a": 0.57, "kappa": 1.97, "delta": 0.54},
    # 18 REBCO D-shaped coils, one coil 3.0 m x 4.3 m (published): contour
    # drawn to the published width.
    "tf": {"shape": "d", "n": 18, "R_bore": 0.75, "R_out": 3.75, "c": 0.45},
    "solenoids": [
        {"r_in": 0.30, "r_out": 0.60, "z_lo": -1.70, "z_hi": 1.70,
         "modules": 6, "color": "cs",
         "label": "Central solenoid (6 modules, HTS)"},
    ],
    "pf": {"auto": 8},
    "pf_label": "PF coils PF1-PF4 pairs (illustrative)",
    "div": [  # Cu divertor coil pairs (Div1, Div2) + in-vessel VS pair;
        {"R": 1.45, "Z":  1.42, "dR": 0.16, "dZ": 0.16},   # positions drawn
        {"R": 1.45, "Z": -1.42, "dR": 0.16, "dZ": 0.16},   # from published
        {"R": 2.20, "Z":  1.32, "dR": 0.16, "dZ": 0.16},   # cross-sections
        {"R": 2.20, "Z": -1.32, "dR": 0.16, "dZ": 0.16},
        {"R": 2.35, "Z":  0.60, "dR": 0.10, "dZ": 0.10},
        {"R": 2.35, "Z": -0.60, "dR": 0.10, "dZ": 0.10},
    ],
    "div_label": "Divertor + VS coils (Cu)",
    "sources": [
        "Creely et al., J. Plasma Phys. 86 (2020) (R0 1.85, a 0.57, kappa "
        "1.97, delta 0.54, Ip 8.7 MA, Bt 12.2 T; 18 TF REBCO; CS1-CS3 "
        "pairs; PF1-PF4 pairs; DN capable)",
        "CFS/MIT TFMC paper, arXiv:2308.12301 (one TF coil 3.0 x 4.3 m)",
        "SPARC PoP 2023 (6 CS modules; 8 PF; 4 Cu divertor coils; "
        "in-vessel VS pair)",
    ],
    "notes": "No public numeric coil table: PF auto-placed, divertor/VS "
             "positions drawn from published cross-section figures.",
},

"DTT": {
    "name": "DTT (ENEA Frascati, Italy)",
    "category": "under_construction",
    "plasma": {"R0": 2.19, "a": 0.70, "kappa": 1.84, "delta": 0.48,
               "kappa_95": 1.65, "delta_95": 0.33},
    # 18 D-shaped Nb3Sn coils (84 turns, 42.5 kA, 12 T peak); envelope not
    # published, drawn from the radial-build proportions around the
    # published CS/PF set.
    "tf": {"shape": "d", "n": 18, "R_bore": 0.80, "R_out": 3.78, "c": 0.34,
           "z_scale": 0.96},
    "solenoids": [
        # CS: 6 independent layer-graded Nb3Sn modules; radii and module
        # heights from the published coil table (Energies 15, 1702, 2022):
        # sub-windings R 0.490/0.596/0.694 m, module dZ 0.788 m, centres
        # Z +/-0.433/1.299/2.166 m.
        {"r_in": 0.430, "r_out": 0.746, "z_lo": -2.560, "z_hi": 2.560,
         "modules": 6, "color": "cs",
         "label": "Central solenoid (6 modules, Nb3Sn)"},
    ],
    "pf": [  # PF1-PF6, verbatim from the published table (Energies 2022)
        {"R": 1.400, "Z":  2.760, "dR": 0.510, "dZ": 0.590},   # PF1 Nb3Sn
        {"R": 3.080, "Z":  2.534, "dR": 0.279, "dZ": 0.517},   # PF2 NbTi
        {"R": 4.351, "Z":  1.015, "dR": 0.390, "dZ": 0.452},   # PF3 NbTi
        {"R": 4.351, "Z": -1.015, "dR": 0.390, "dZ": 0.452},   # PF4 NbTi
        {"R": 3.080, "Z": -2.534, "dR": 0.279, "dZ": 0.517},   # PF5 NbTi
        {"R": 1.400, "Z": -2.760, "dR": 0.510, "dZ": 0.590},   # PF6 Nb3Sn
    ],
    "pf_label": "PF coils PF1-PF6 (published table)",
    "sources": [
        "Castaldo et al., Energies 15 (2022) 1702 (Table 2: CS sub-windings "
        "R 0.490/0.596/0.694 m, module dZ 0.788 m, centres +/-0.433/1.299/"
        "2.166 m; PF1-PF6 R/Z and sections verbatim)",
        "dtt-project.it and IAEA FEC papers (R0 2.19, a 0.70, kappa95 "
        "~1.65, delta95 ~0.33, SN, Ip 5.5 MA, BT ~5.85 T on axis; 18 "
        "Nb3Sn TF coils, 84 turns, 42.5 kA, 12 T peak)",
    ],
    "notes": "CS and PF geometry published (coil table); TF envelope "
             "drawn from the radial-build proportions.  In-vessel VS and "
             "saddle coils not drawn (no published positions).",
},

"BEST": {
    "name": "BEST (ASIPP Hefei, China)",
    "category": "under_construction",
    "plasma": {"R0": 3.6, "a": 1.1, "kappa": 1.85, "delta": 0.49},
    "tf": {"shape": "d", "n": 16, "R_bore": 1.05, "R_out": 6.70, "c": 0.70},
    "solenoids": [
        {"r_in": 0.50, "r_out": 0.98, "z_lo": -3.0, "z_hi": 3.0,
         "modules": 6, "color": "cs",
         "label": "Central solenoid (6 modules, HTS/LTS)"},
    ],
    "pf": {"auto": 7},
    "pf_label": "PF coils (7, illustrative positions)",
    "sources": [
        "BEST Research Plan v1.1 (EUROfusion/ASIPP): R0 3.6, a 1.1, kappa "
        "up to 1.88, delta 0.49, LSN, Ip up to 7 MA, Bt 6.15 T; 16 TF "
        "coils (87.2 t each, Nb3Sn); CS 6 modules (180 t, Nb3Sn + YBCO); "
        "7 PF coils + 8 correction coils",
        "Xinhua (cryostat base ~18 m diameter)",
    ],
    "notes": "Coil counts and plasma published; TF/CS dimensions estimated "
             "from the radial-build proportions (not yet public).",
},

# ─────────────────────────── Reactor projects ────────────────────────────────

"EU-DEMO": {
    "name": "EU-DEMO (2018 baseline, EUROfusion)",
    "category": "reactor_project",
    "plasma": {"R0": 9.07, "a": 2.93, "kappa": 1.80, "delta": 0.47,
               "kappa_95": 1.65, "delta_95": 0.33},
    # TF bore from the published radial build: inboard SOL 0.225 m, blanket
    # 0.755 m, shield/VV 0.60 m; outboard blanket 0.982 m, shield/VV 1.1 m.
    "tf": {"shape": "d", "n": 16, "R_bore": 3.55, "R_out": 15.30, "c": 1.05},
    "solenoids": [
        {"r_in": 2.35, "r_out": 3.40, "z_lo": -7.2, "z_hi": 7.2,
         "modules": 5, "color": "cs",
         "label": "Central solenoid (5 modules, Nb3Sn)"},
    ],
    "pf": {"auto": 6},
    "pf_label": "PF coils (6, illustrative positions)",
    "sources": [
        "Siccinio et al. (EU-DEMO 2018 baseline: R0 9.07, a 2.93, A 3.1, "
        "kappa95 1.65, delta95 0.33, Ip 17.75 MA, Bt 5.86 T, LSN)",
        "Federici et al. 2018 (radial build: inboard blanket 0.755 m, "
        "outboard 0.982 m, shields 0.6/1.1 m, SOL 0.225 m; 16 TF coils; "
        "6 PF coils)",
        "EUROfusion magnet papers (CS: 5 modules CSU3-CSL3, Nb3Sn)",
    ],
    "notes": "TF contour derived from the published radial build; CS/PF "
             "dimensions drawn (not published).",
},

"ARC": {
    "name": "ARC (MIT, 2015/2018 concept)",
    "category": "reactor_project",
    "plasma": {"R0": 3.3, "a": 1.13, "kappa": 1.84, "delta": 0.38},
    # 18 demountable REBCO coils.  The published figures (Sorbom 2015)
    # show a very FULL D, close to a rounded rectangle enclosing the whole
    # FLiBe tank, so the cage is drawn with the window-pane generator and
    # large corner radii; PF1-PF3 sit INSIDE the TF volume, which the
    # demountable joints allow.
    "tf": {"shape": "rect", "n": 18, "R_in_leg": 1.00, "R_out_leg": 5.60,
           "Z_half": 3.60, "c": 0.55, "corner_r": 1.00},
    "tf_label": "TF coils (18, demountable REBCO)",
    "solenoids": [
        {"r_in": 0.50, "r_out": 0.70, "z_lo": -2.9, "z_hi": 2.9,
         "modules": 1, "color": "cs", "label": "Central solenoid (REBCO)"},
    ],
    "pf": [  # PF1-PF5 pairs, verbatim from Kuang et al. 2018 (Table)
        {"R": 1.80, "Z":  3.03, "dR": 0.20, "dZ": 0.45},
        {"R": 1.80, "Z": -3.03, "dR": 0.20, "dZ": 0.45},
        {"R": 4.85, "Z":  2.70, "dR": 0.20, "dZ": 0.45},
        {"R": 4.85, "Z": -2.70, "dR": 0.20, "dZ": 0.45},
        {"R": 4.85, "Z":  2.10, "dR": 0.20, "dZ": 0.45},
        {"R": 4.85, "Z": -2.10, "dR": 0.20, "dZ": 0.45},
        {"R": 6.13, "Z":  2.78, "dR": 0.20, "dZ": 0.20},
        {"R": 6.13, "Z": -2.78, "dR": 0.20, "dZ": 0.20},
        {"R": 6.57, "Z":  1.70, "dR": 0.20, "dZ": 0.20},
        {"R": 6.57, "Z": -1.70, "dR": 0.20, "dZ": 0.20},
    ],
    "pf_label": "PF coils PF1-PF5 (REBCO, Kuang 2018)",
    "div": [  # Cu trim coils TR1-TR3 pairs (Kuang 2018, verbatim)
        {"R": 4.20, "Z":  2.85, "dR": 0.11, "dZ": 0.11},
        {"R": 4.20, "Z": -2.85, "dR": 0.11, "dZ": 0.11},
        {"R": 3.59, "Z":  1.97, "dR": 0.11, "dZ": 0.11},
        {"R": 3.59, "Z": -1.97, "dR": 0.11, "dZ": 0.11},
        {"R": 3.98, "Z":  1.70, "dR": 0.11, "dZ": 0.11},
        {"R": 3.98, "Z": -1.70, "dR": 0.11, "dZ": 0.11},
    ],
    "div_label": "Cu trim coils TR1-TR3",
    "sources": [
        "Sorbom et al., Fus. Eng. Des. 100 (2015) 378, arXiv:1409.3540 "
        "(R0 3.3, a 1.13, kappa 1.84, B0 9.2 T, Ip 7.8 MA; 18 demountable "
        "REBCO TF; CS at R 0.5-0.7 m, Z +/-3 m; inboard build 0.85 m)",
        "Kuang et al., Fus. Eng. Des. 2018, arXiv:1809.10555 (delta 0.375, "
        "double null; PF and trim coil table, verbatim)",
    ],
    "notes": "PF/trim coils published (Kuang 2018); TF cage drawn as a "
             "rounded-rectangle full D matching the published figures "
             "(leg positions from the radial build).  Demountable joints "
             "and FLiBe blanket tank not drawn.",
},

"MANTA": {
    "name": "MANTA (MIT PSFC et al., NT pilot plant)",
    "category": "reactor_project",
    # NEGATIVE triangularity: delta = -0.5 (published, up-down symmetric
    # double null).  kappa 1.4, R0 4.55 m, a 1.2 m, B0 11 T, Ip 10 MA.
    "plasma": {"R0": 4.55, "a": 1.2, "kappa": 1.4, "delta": -0.5},
    # 18 demountable non-insulated REBCO coils, "window pane" (rectangular)
    # design: radial build 0.6 m and toroidal thickness 0.544 m published;
    # the leg positions are drawn from the radial-build proportions (CS
    # bore 3.2 m diameter, FLiBe tank between coil and plasma).
    "tf": {"shape": "rect", "n": 18, "R_in_leg": 1.90, "R_out_leg": 7.20,
           "Z_half": 3.50, "c": 0.60, "w_tor": 0.544, "corner_r": 0.45},
    "tf_label": "TF coils (18, window-pane REBCO)",
    "solenoids": [
        {"r_in": 0.60, "r_out": 1.45, "z_lo": -2.6, "z_hi": 2.6,
         "modules": 1, "color": "cs",
         "label": "Central solenoid (REBCO, 260 Wb)"},
    ],
    "pf": [  # PF1-PF3 pairs, verbatim from the paper (Table 5)
        {"R": 3.79, "Z":  2.25, "dR": 0.365, "dZ": 0.327},
        {"R": 3.79, "Z": -2.25, "dR": 0.365, "dZ": 0.327},
        {"R": 5.45, "Z":  2.90, "dR": 0.327, "dZ": 0.290},
        {"R": 5.45, "Z": -2.90, "dR": 0.327, "dZ": 0.290},
        {"R": 6.65, "Z":  1.25, "dR": 0.215, "dZ": 0.140},
        {"R": 6.65, "Z": -1.25, "dR": 0.215, "dZ": 0.140},
    ],
    "pf_label": "PF coils PF1-PF3 (REBCO pairs)",
    "sources": [
        "MANTA Collaboration, Plasma Phys. Control. Fusion 66 (2024) "
        "105006, arXiv:2405.20243 (R0 4.55, a 1.2, kappa 1.4, delta -0.5, "
        "B0 11 T, Ip 10 MA, DN; 18 window-pane REBCO TF, azimuthal "
        "thickness 544 mm; CS 260 Wb, 25 T peak; PF table R 3.79/5.45/"
        "6.65 m, Z +/-2.25/2.9/1.25 m; FLiBe immersion blanket)",
        "Companion paper arXiv:2407.06526 (B_T 11.1 T, q95 2.3, DN "
        "power handling)",
    ],
    "notes": "Plasma, PF table and TF section published; TF leg positions "
             "and CS build drawn from the radial-build proportions (3.2 m "
             "bore).  The FLiBe tank and divertor slots are not drawn.",
},

"CFETR": {
    "name": "CFETR (China, 2019+ design)",
    "category": "reactor_project",
    "plasma": {"R0": 7.2, "a": 2.2, "kappa": 2.00, "delta": 0.43},
    # 16 D-shaped coils; one coil ~21.7 m x 12.3 m, 650 t (published):
    # contour drawn to the published width.
    "tf": {"shape": "d", "n": 16, "R_bore": 2.70, "R_out": 14.50, "c": 1.10},
    "solenoids": [
        {"r_in": 1.35, "r_out": 2.25, "z_lo": -7.4, "z_hi": 7.4,
         "modules": 8, "color": "cs",
         "label": "Central solenoid (8 modules, HTS/Nb3Sn)"},
    ],
    "pf": {"auto": 6},
    "pf_label": "PF coils (6, illustrative positions)",
    "sources": [
        "CFETR TF prototype papers (16 TF coils, one coil 21.7 m x 12.3 m, "
        "650 t, graded Nb3Sn/NbTi, 14.5-14.8 T peak)",
        "MDPI magnet review (CS: 8 modules, winding radius 2.25 m, hybrid "
        "HTS + Nb3Sn, 19.9 T peak; 6 PF coils + 18 correction coils)",
        "Parameter tables (R0 7.2, a 2.2, kappa ~2.0, delta ~0.43, "
        "Ip 13-14 MA, Bt 6.5 T)",
    ],
    "notes": "TF width and CS outer radius published; PF positions drawn "
             "(auto-placed).",
},

}

# Alias keys (tolerated spellings)
_ALIASES = {"ASDEX": "ASDEX-U", "ASDEX UPGRADE": "ASDEX-U", "AUG": "ASDEX-U",
            "JT60SA": "JT-60SA", "MASTU": "MAST-U", "MAST": "MAST-U",
            "DEMO": "EU-DEMO", "W7X": "W7-X", "WENDELSTEIN": "W7-X",
            "TJ2": "TJ-II", "TJII": "TJ-II"}

CATEGORY_LABELS = {
    "operating": "Operating tokamaks",
    "under_construction": "Tokamaks under construction",
    "reactor_project": "Reactor design projects",
}


def list_presets() -> dict:
    """Preset keys grouped by category (ordered as in the bank)."""
    out = {}
    for key, p in PRESETS.items():
        out.setdefault(p["category"], []).append(key)
    return out


def get_preset(name: str) -> dict:
    """Return a deep copy of a preset by (case-insensitive, aliased) name."""
    import copy
    key = str(name).strip().upper()
    key = _ALIASES.get(key, key)
    for k in PRESETS:
        if k.upper() == key:
            return copy.deepcopy(PRESETS[k]) | {"key": k}
    raise KeyError(f"Unknown machine preset '{name}'. "
                   f"Available: {', '.join(PRESETS)}")


# =============================================================================
# Renderer
# =============================================================================

def _preset_lcfs(plasma: dict):
    """Miller LCFS closure with D0FUS PCHIP shaping profiles."""
    R0, a = plasma["R0"], plasma["a"]
    kap, dlt = plasma["kappa"], plasma["delta"]
    kap95 = plasma.get("kappa_95", kap / 1.12)
    dlt95 = plasma.get("delta_95", dlt / 1.5)

    def _lcfs(theta, frac=1.0):
        k = float(kappa_profile(frac, kap, kap95))
        d = float(delta_profile(frac, dlt, dlt95))
        r = a * frac
        R = R0 + r * np.cos(theta + np.arcsin(np.clip(d, -1, 1))
                            * np.sin(theta))
        Z = k * r * np.sin(theta)
        return R, Z
    return _lcfs


def _visible_angles(n: int):
    """Toroidal angles of the visible coils (one on the first cut plane)."""
    phi0, phi1 = _pv3d_keep_range()
    step = 2.0 * np.pi / n
    ang = phi0 + np.arange(n + 1) * step
    return ang[ang <= phi1 + 1e-9]


def _draw_plasma(pv, pl, plasma: dict):
    """Shaped plasma torus with cut-face caps and nested flux surfaces."""
    phi0, phi1 = _pv3d_keep_range()
    _lcfs = _preset_lcfs(plasma)
    phi = np.linspace(phi0, phi1, 240)
    th = np.linspace(0.0, 2.0 * np.pi, 160)
    P, T = np.meshgrid(phi, th)
    Rp, Zp = _lcfs(T)
    plasma_mesh = _pv3d_grid(pv, Rp * np.cos(P), Rp * np.sin(P), Zp)
    pl.add_mesh(plasma_mesh, color=_PV3D_PLASMA_COL, smooth_shading=True,
                specular=0.3, specular_power=16, diffuse=0.85, ambient=0.35)
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            pl.add_silhouette(plasma_mesh.extract_surface(),
                              color=_PV3D_EDGE_COL, line_width=_PV3D_EDGE_LW)
    except Exception:
        pass
    thc = np.linspace(0.0, 2.0 * np.pi, 200)
    Rcap, Zcap = _lcfs(thc)
    R0 = plasma["R0"]
    for pc, side in ((phi0, -1), (phi1, +1)):
        _pv3d_add_cap(pv, pl, Rcap, Zcap, pc, _PV3D_PLASMA_COL)
        eps = 0.008 * side
        for f in _PV3D_FLUX_FRACS:
            Rf, Zf = _lcfs(thc, frac=f)
            pts = np.stack([Rf * np.cos(pc + eps), Rf * np.sin(pc + eps), Zf],
                           axis=1)
            pl.add_mesh(pv.lines_from_points(np.vstack([pts, pts[:1]])),
                        color=_PV3D_FLUX_COL, line_width=1.8,
                        render_lines_as_tubes=True)
        pl.add_mesh(pv.Sphere(radius=max(0.03, 0.013 * R0),
                              center=(R0 * np.cos(pc + eps),
                                      R0 * np.sin(pc + eps), 0.0)),
                    color=_PV3D_FLUX_COL)
    return float(Rcap.max()), float(np.abs(Zcap).max())


def _d_contours(tf: dict):
    """Outer and inner D contours of a TF spec, with the optional vertical
    stretch ``z_scale`` applied.  Real TF coils (JET, JT-60SA, TCV...) are
    taller than the pure Princeton-D of the same leg radii; the stretch
    reproduces that so the drawn cage clears the published coil positions."""
    R_o, Z_o = _princeton_D_contour(tf["R_bore"], tf["R_out"])
    R_i, Z_i = _offset_contour(R_o, Z_o, tf["c"])
    zs = float(tf.get("z_scale", 1.0))
    return (np.asarray(R_o), zs * np.asarray(Z_o),
            np.asarray(R_i), zs * np.asarray(Z_i))


def _draw_tf_d(pv, pl, tf: dict):
    """Princeton-D TF cage; returns (contour, extents) for PF auto-placement."""
    R_o, Z_o, R_i, Z_i = _d_contours(tf)
    R_c, Z_c = _pv3d_d_centreline(R_o, Z_o, R_i, Z_i)
    n = int(tf["n"])
    w_tor = tf.get("w_tor",
                   0.96 * 2.0 * np.pi * (tf["R_bore"] + 0.5 * tf["c"]) / n)
    for ph in _visible_angles(n):
        _pv3d_add_tube(pv, pl, R_c * np.cos(ph), R_c * np.sin(ph), Z_c,
                       0.5 * tf["c"], 0.5 * w_tor,
                       (-np.sin(ph), np.cos(ph), 0.0), _PV3D_TF_COL)
    H = float(Z_o.max() - Z_o.min())
    return dict(R_out_arr=np.asarray(R_o), Z_out_arr=np.asarray(Z_o),
                R_max=float(np.max(R_o)), z_min=float(np.min(Z_o)),
                z_max=float(np.max(Z_o)), H=H)


def _draw_tf_circular(pv, pl, tf: dict):
    """Circular TF cage (Tore Supra / WEST heritage machines)."""
    n = int(tf["n"])
    rc = 0.5 * (tf["r_in"] + tf["r_out"])
    half_rad = 0.5 * (tf["r_out"] - tf["r_in"])
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R = tf["center_R"] + rc * np.cos(theta)
    Z = rc * np.sin(theta)
    for ph in _visible_angles(n):
        _pv3d_add_tube(pv, pl, R * np.cos(ph), R * np.sin(ph), Z,
                       half_rad, tf["half_tor"],
                       (-np.sin(ph), np.cos(ph), 0.0), _PV3D_TF_COL)
    R_max = tf["center_R"] + tf["r_out"]
    return dict(R_out_arr=None, Z_out_arr=None, R_max=R_max,
                z_min=-tf["r_out"], z_max=tf["r_out"], H=2.0 * tf["r_out"])


def _rect_tf_centreline(R1, R2, Zh, rc=0.35, n=480):
    """
    Rounded-rectangle (window-pane) TF centreline in the poloidal plane.

    R1 / R2 are the inner / outer leg centreline radii, Zh the half-height
    of the horizontal legs' centreline; rc is the corner rounding radius,
    which keeps the sweep frame continuous at the four corners (a sharp
    corner would twist the rectangular section).  Returns a closed,
    uniformly resampled (R, Z) polyline.
    """
    rc = min(rc, 0.45 * (R2 - R1), 0.45 * Zh)
    pts = []

    def _arc(cx, cz, a0, a1, m=18):
        for t in np.linspace(a0, a1, m, endpoint=False):
            pts.append((cx + rc * np.cos(t), cz + rc * np.sin(t)))

    # Counter-clockwise, starting on the inner leg at the midplane.
    pts.append((R1, 0.0))
    pts.append((R1, Zh - rc))
    _arc(R1 + rc, Zh - rc, np.pi, 0.5 * np.pi)          # upper inner corner
    pts.append((R2 - rc, Zh))
    _arc(R2 - rc, Zh - rc, 0.5 * np.pi, 0.0)            # upper outer corner
    pts.append((R2, -(Zh - rc)))
    _arc(R2 - rc, -(Zh - rc), 0.0, -0.5 * np.pi)        # lower outer corner
    pts.append((R1 + rc, -Zh))
    _arc(R1 + rc, -(Zh - rc), -0.5 * np.pi, -np.pi)     # lower inner corner
    pts.append((R1, -(Zh - rc)))
    pts.append((R1, 0.0))
    P = np.asarray(pts)
    # Uniform arc-length resampling for a clean sweep frame.
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]),
                                                  np.diff(P[:, 1])))])
    su = np.linspace(0.0, s[-1], n)
    return np.interp(su, s, P[:, 0]), np.interp(su, s, P[:, 1])


def _draw_tf_rect(pv, pl, tf: dict):
    """Window-pane (rounded-rectangle) TF cage, e.g. MANTA."""
    R_c, Z_c = _rect_tf_centreline(tf["R_in_leg"], tf["R_out_leg"],
                                   tf["Z_half"], tf.get("corner_r", 0.35))
    n = int(tf["n"])
    w_tor = tf.get("w_tor",
                   0.96 * 2.0 * np.pi * (tf["R_in_leg"] - 0.5 * tf["c"]) / n)
    for ph in _visible_angles(n):
        _pv3d_add_tube(pv, pl, R_c * np.cos(ph), R_c * np.sin(ph), Z_c,
                       0.5 * tf["c"], 0.5 * w_tor,
                       (-np.sin(ph), np.cos(ph), 0.0), _PV3D_TF_COL)
    z_top = tf["Z_half"] + 0.5 * tf["c"]
    return dict(R_out_arr=np.asarray(R_c), Z_out_arr=np.asarray(Z_c),
                R_max=tf["R_out_leg"] + 0.5 * tf["c"],
                z_min=-z_top, z_max=z_top, H=2.0 * z_top)


def _draw_tf_spherical(pv, pl, tf: dict):
    """Spherical-tokamak TF cage: centre-column rod + rectangular return
    limbs (vertical outer leg and two horizontal beams per limb)."""
    # Solid centre rod (full 360 degrees: the wedge cut never exposes it).
    rod = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                      radius=tf["rod_r"], height=2.0 * tf["rod_z"],
                      capping=True)
    pl.add_mesh(rod, color=_PV3D_TF_COL, smooth_shading=True,
                specular=0.2, diffuse=0.85, ambient=0.35)
    beam_z = 0.5 * (tf["rod_z"] + tf["limb_z"])
    for ph in _visible_angles(int(tf["n_limbs"])):
        ct, st = np.cos(ph), np.sin(ph)
        # Vertical outer leg.
        zz = np.linspace(-tf["limb_z"], tf["limb_z"], 30)
        _pv3d_add_tube(pv, pl, np.full_like(zz, tf["limb_R"] * ct),
                       np.full_like(zz, tf["limb_R"] * st), zz,
                       tf["limb_half_t"], tf["limb_half_r"],
                       (ct, st, 0.0), _PV3D_TF_COL, closed=False)
        # Horizontal beams joining the rod to the outer leg.
        rr = np.linspace(tf["rod_r"] * 0.5, tf["limb_R"], 30)
        for zb in (-beam_z, beam_z):
            _pv3d_add_tube(pv, pl, rr * ct, rr * st, np.full_like(rr, zb),
                           tf["limb_half_t"], tf["limb_half_r"],
                           (0.0, 0.0, 1.0), _PV3D_TF_COL, closed=False)
    z_top = beam_z + tf["limb_half_r"]
    return dict(R_out_arr=None, Z_out_arr=None,
                R_max=tf["limb_R"] + tf["limb_half_r"],
                z_min=-z_top, z_max=z_top, H=2.0 * z_top)


def _draw_iron(pv, pl, iron: dict):
    """Iron transformer yoke (JET): central limb, return limbs, beams."""
    _pv3d_cyl_sector(pv, pl, 0.06, iron["core_r"],
                     -iron["core_z"], iron["core_z"], _PV3D_IRON_COL)
    for ph in _visible_angles(int(iron["n_limbs"])):
        ct, st = np.cos(ph), np.sin(ph)
        zz = np.linspace(-iron["limb_z"], iron["limb_z"], 30)
        _pv3d_add_tube(pv, pl, np.full_like(zz, iron["limb_R"] * ct),
                       np.full_like(zz, iron["limb_R"] * st), zz,
                       iron["limb_half_t"], iron["limb_half_r"],
                       (ct, st, 0.0), _PV3D_IRON_COL, closed=False)
        rr = np.linspace(iron["core_r"] * 0.6, iron["limb_R"], 30)
        for zb in (-iron["beam_z"], iron["beam_z"]):
            _pv3d_add_tube(pv, pl, rr * ct, rr * st, np.full_like(rr, zb),
                           iron["beam_half"], iron["beam_half"],
                           (0.0, 0.0, 1.0), _PV3D_IRON_COL, closed=False)
    R_max = iron["limb_R"] + iron["limb_half_r"]
    z_max = max(iron["core_z"], iron["limb_z"])
    return dict(R_max=R_max, z_min=-z_max, z_max=z_max)


def _auto_pf_positions(n: int, tf_ctx: dict, a: float):
    """Illustrative PF ring set along the TF outer contour (generic D0FUS
    convention), for machines without a public coil table."""
    R_o, Z_o = tf_ctx["R_out_arr"], tf_ctx["Z_out_arr"]
    R_mid = 0.5 * (float(np.min(R_o)) + float(np.max(R_o)))
    ang_contour = np.arctan2(Z_o, R_o - R_mid)
    coils = []
    for psi_deg in np.linspace(80.0, -80.0, int(n)):
        s = 0.42 * a
        psi = np.deg2rad(psi_deg)
        i_near = int(np.argmin(np.abs(np.angle(
            np.exp(1j * (ang_contour - psi))))))
        off = 0.18 * a + 0.5 * s * np.sqrt(2.0)
        coils.append({"R": R_o[i_near] + off * np.cos(psi),
                      "Z": Z_o[i_near] + off * np.sin(psi),
                      "dR": s, "dZ": s})
    return coils


def _draw_ring_coils(pv, pl, coils, color, min_half):
    """Rectangular-section ring coils (open arcs over the kept range).
    A coil entry may override the group colour with a 'color' key (palette
    key such as 'cs'), e.g. the TCV OH-circuit rings C and D."""
    phi0, phi1 = _pv3d_keep_range()
    t = np.linspace(phi0, phi1, 160)
    R_max = z_min = z_max = 0.0
    for c in coils:
        hn = max(0.5 * c["dR"], min_half)
        hb = max(0.5 * c["dZ"], min_half)
        col = _col(c["color"]) if "color" in c else color
        _pv3d_add_tube(pv, pl, c["R"] * np.cos(t), c["R"] * np.sin(t),
                       np.full_like(t, c["Z"]), hn, hb,
                       (0.0, 0.0, 1.0), col, closed=False)
        R_max = max(R_max, c["R"] + hn)
        z_min = min(z_min, c["Z"] - hb)
        z_max = max(z_max, c["Z"] + hb)
    return dict(R_max=R_max, z_min=z_min, z_max=z_max)


def _stell_axis(p: dict, u):
    """Magnetic-axis position (R_ax, Z_ax) of a stellarator preset."""
    st = p["stellarator"]
    R0 = p["plasma"]["R0"]
    fam = st.get("family", "helias")
    if fam == "heliac":
        sw = st["axis_swing"]
        return (R0 + sw * np.cos(st["n_fp"] * u),
                sw * np.sin(st["n_fp"] * u))
    if fam == "heliotron":
        return R0 + 0.0 * np.asarray(u), 0.0 * np.asarray(u)
    return (R0 + st["axis_swing"] * np.cos(st["n_fp"] * u),
            0.0 * np.asarray(u))


def _stell_surface(p: dict, u, theta, s: float = 1.0):
    """
    Boundary of the stellarator preset at normalised radius ``s`` (s = 1 is
    the LCFS).  Three families, all closing toroidally by construction:

      helias    : lab-frame Fourier surface (W7-X-like), axis swing +
                  helical first/second harmonics;
      heliotron : rotating-ellipse surface (LHD-like), semi-axes a1/a2
                  rotating n_rot times per toroidal turn with the
                  continuous helical coils;
      heliac    : circular section of radius a carried on the helical
                  magnetic axis winding around the central conductor
                  (TJ-II-like).
    """
    st = p["stellarator"]
    fam = st.get("family", "helias")
    if fam == "heliotron":
        beta = st["n_rot"] * np.asarray(u)
        a1, a2 = s * st["a1"], s * st["a2"]
        dR = (a1 * np.cos(theta) * np.cos(beta)
              - a2 * np.sin(theta) * np.sin(beta))
        dZ = (a1 * np.cos(theta) * np.sin(beta)
              + a2 * np.sin(theta) * np.cos(beta))
        return p["plasma"]["R0"] + dR, dZ
    if fam == "heliac":
        Rax, Zax = _stell_axis(p, u)
        a = s * st["a"]
        return Rax + a * np.cos(theta), Zax + a * np.sin(theta)
    N = st["n_fp"]
    a, h1, h2 = s * st["a"], s * st["h1"], s * st["h2"]
    R = (p["plasma"]["R0"] + st["axis_swing"] * np.cos(N * u)
         + a * np.cos(theta) + h1 * np.cos(theta - N * u)
         + h2 * np.cos(2.0 * theta - N * u))
    Z = (a * np.sin(theta) + h1 * np.sin(theta - N * u)
         - h2 * np.sin(2.0 * theta - N * u))
    return R, Z


def _stell_modular_centreline(p: dict, u_k: float, n: int = 160):
    """
    Modular coil centreline: one identical, slightly elliptical loop per
    coil, centred on the local magnetic axis and tilted so its plane
    follows the helical phase of the field period; small first and second
    poloidal harmonics warp it out of plane (non-planar W7-X-like shape).
    Ported from the D0FUS concept-figure generator.
    """
    st = p["stellarator"]
    N, rc, ell = st["n_fp"], st["coil_rc"], st.get("coil_ell", 0.92)
    wob, sad = st.get("coil_wob", 0.05), st.get("coil_sad", 0.06)
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    axis_R = p["plasma"]["R0"] + st["axis_swing"] * np.cos(N * u_k)
    psi = 0.5 * N * u_k                          # local helical phase
    e1 = rc * np.cos(theta)
    e2 = ell * rc * np.sin(theta)
    dR = e1 * np.cos(psi) - e2 * np.sin(psi)
    dZ = e1 * np.sin(psi) + e2 * np.cos(psi)
    R = axis_R + dR
    phi = u_k + wob * np.sin(theta + psi) + sad * np.sin(2.0 * theta + psi)
    return R * np.cos(phi), R * np.sin(phi), dZ


def _plot_stellarator_preset(pv, p: dict, save_dir, human: bool) -> None:
    """Render a stellarator preset (plasma + modular + planar coils)."""
    st = p["stellarator"]
    R0 = p["plasma"]["R0"]
    key = p.get("key", p.get("name", "stellarator"))
    phi0, phi1 = _pv3d_keep_range()

    pl = pv.Plotter(off_screen=True, window_size=[1500, 1150])
    pl.set_background("white")

    # Plasma: opaque twisted surface over the kept range, plus cut-face
    # caps decorated with nested flux surfaces and the magnetic axis.
    u = np.linspace(phi0, phi1, 480)
    th = np.linspace(0.0, 2.0 * np.pi, 170)
    U, T = np.meshgrid(u, th)
    Rp, Zp = _stell_surface(p, U, T)
    plasma = _pv3d_grid(pv, Rp * np.cos(U), Rp * np.sin(U), Zp)
    pl.add_mesh(plasma, color=_PV3D_PLASMA_COL, smooth_shading=True,
                specular=0.3, specular_power=16, diffuse=0.85, ambient=0.35)
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            pl.add_silhouette(plasma.extract_surface(),
                              color=_PV3D_EDGE_COL, line_width=_PV3D_EDGE_LW)
    except Exception:
        pass
    thc = np.linspace(0.0, 2.0 * np.pi, 240)
    for pc, side in ((phi0, -1), (phi1, +1)):
        Rc, Zc = _stell_surface(p, pc, thc)
        _pv3d_add_cap(pv, pl, Rc, Zc, pc, _PV3D_PLASMA_COL)
        eps = 0.008 * side
        for f in _PV3D_FLUX_FRACS:
            Rf, Zf = _stell_surface(p, pc, thc, s=f)
            pts = np.stack([Rf * np.cos(pc + eps), Rf * np.sin(pc + eps),
                            Zf], axis=1)
            pl.add_mesh(pv.lines_from_points(np.vstack([pts, pts[:1]])),
                        color=_PV3D_FLUX_COL, line_width=1.8,
                        render_lines_as_tubes=True)
        R_ax, Z_ax = _stell_axis(p, pc)
        pl.add_mesh(pv.Sphere(radius=max(0.03, 0.013 * R0),
                              center=(float(R_ax) * np.cos(pc + eps),
                                      float(R_ax) * np.sin(pc + eps),
                                      float(Z_ax))),
                    color=_PV3D_FLUX_COL)

    fam = st.get("family", "helias")
    legend_coils = []
    ext_R = float(np.max(Rp))
    ext_z = float(np.max(np.abs(Zp)))

    if fam == "helias":
        # Modular non-planar coils: one every 2 pi / n_modular.
        n_mod = int(st["n_modular"])
        step = 2.0 * np.pi / n_mod
        ang = phi0 + np.arange(n_mod + 1) * step
        for u_k in ang[ang <= phi1 + 1e-9]:
            cx, cy, cz = _stell_modular_centreline(p, u_k)
            Rl = np.sqrt(cx ** 2 + cy ** 2)
            ref = (-cy / Rl, cx / Rl, np.zeros_like(cx))
            _pv3d_add_tube(pv, pl, cx, cy, cz, st["coil_half"],
                           st["coil_half"], ref, _PV3D_TF_COL)
        legend_coils.append((f"Modular non-planar coils ({n_mod}, NbTi)",
                             _PV3D_TF_COL))
        ext_z = max(ext_z,
                    st.get("coil_ell", 0.92) * st["coil_rc"]
                    + st["coil_half"])
        ext_R = max(ext_R, R0 + st["coil_rc"] + st["coil_half"])
        # Planar coils: larger circular loops interleaved between the
        # modular coils, alternately tilted about the local radial axis.
        n_pla = int(st.get("n_planar", 0))
        if n_pla:
            rc_p, half_p = st["planar_rc"], st["planar_half"]
            alpha0 = np.deg2rad(st.get("planar_tilt_deg", 12.0))
            step_p = 2.0 * np.pi / n_pla
            ang_p = phi0 + 0.5 * step_p + np.arange(n_pla + 1) * step_p
            theta = np.linspace(0.0, 2.0 * np.pi, 160)
            for k, ph in enumerate(ang_p[ang_p <= phi1 + 1e-9]):
                alpha = alpha0 * (1 if k % 2 == 0 else -1)
                r_hat = np.array([np.cos(ph), np.sin(ph), 0.0])
                t_hat = np.array([-np.sin(ph), np.cos(ph), 0.0])
                z_hat = np.array([0.0, 0.0, 1.0])
                e2 = np.cos(alpha) * z_hat + np.sin(alpha) * t_hat
                C = R0 * r_hat
                pts = (C[None, :] + rc_p * np.cos(theta)[:, None] * r_hat
                       + rc_p * np.sin(theta)[:, None] * e2)
                _pv3d_add_tube(pv, pl, pts[:, 0], pts[:, 1], pts[:, 2],
                               half_p, half_p, tuple(t_hat), _PV3D_PF_COL)
            legend_coils.append((f"Planar coils ({n_pla}, NbTi)",
                                 _PV3D_PF_COL))
            ext_z = max(ext_z, rc_p * np.cos(alpha0) + half_p)
            ext_R = max(ext_R, R0 + rc_p + half_p)

    elif fam == "heliotron":
        # Two continuous helical windings (l = 2): poloidal angle
        # n_rot * phi + k*pi along the torus, on the coil minor circle.
        ac, half = st["coil_ac"], st["coil_half"]
        uu = np.linspace(phi0, phi1, 700)
        for k in range(2):
            th_h = st["n_rot"] * uu + k * np.pi
            R = R0 + ac * np.cos(th_h)
            Z = ac * np.sin(th_h)
            ref = (np.cos(uu), np.sin(uu), np.zeros_like(uu))
            _pv3d_add_tube(pv, pl, R * np.cos(uu), R * np.sin(uu), Z,
                           half, half, ref, _PV3D_TF_COL, closed=False)
        legend_coils.append(("Helical coils (2, l=2 / m=10, NbTi)",
                             _PV3D_TF_COL))
        ext_R = max(ext_R, R0 + ac + half)
        ext_z = max(ext_z, ac + half)
        # PF ring pairs (IV / IS / OV) from the preset table.
        pf_coils = p.get("pf", [])
        if pf_coils:
            e = _draw_ring_coils(pv, pl, pf_coils, _PV3D_PF_COL, 0.03)
            ext_R = max(ext_R, e["R_max"])
            ext_z = max(ext_z, e["z_max"], -e["z_min"])
            legend_coils.append((p.get("pf_label", "PF coils"),
                                 _PV3D_PF_COL))

    elif fam == "heliac":
        # Planar circular TF coils whose centres follow the helical axis
        # around the central conductor.
        n_tf, r_tf, half = int(st["n_tf"]), st["tf_r"], st["tf_half"]
        step = 2.0 * np.pi / n_tf
        ang = phi0 + np.arange(n_tf + 1) * step
        theta = np.linspace(0.0, 2.0 * np.pi, 120)
        for ph in ang[ang <= phi1 + 1e-9]:
            Rc, Zc = _stell_axis(p, ph)
            r_hat = np.array([np.cos(ph), np.sin(ph), 0.0])
            t_hat = np.array([-np.sin(ph), np.cos(ph), 0.0])
            C = float(Rc) * r_hat + np.array([0.0, 0.0, float(Zc)])
            pts = (C[None, :] + r_tf * np.cos(theta)[:, None] * r_hat
                   + r_tf * np.sin(theta)[:, None]
                   * np.array([0.0, 0.0, 1.0]))
            _pv3d_add_tube(pv, pl, pts[:, 0], pts[:, 1], pts[:, 2],
                           half, half, tuple(t_hat), _PV3D_TF_COL)
        legend_coils.append((f"TF coils ({n_tf}, circular, Cu)",
                             _PV3D_TF_COL))
        # Central circular conductor + helical winding around it.
        uu = np.linspace(phi0, phi1, 400)
        _pv3d_add_tube(pv, pl, R0 * np.cos(uu), R0 * np.sin(uu),
                       np.zeros_like(uu), st["cc_half"], st["cc_half"],
                       (0.0, 0.0, 1.0), _PV3D_CS_COL, closed=False)
        hxs = st["hx_swing"]
        Rh = R0 + hxs * np.cos(st["n_fp"] * uu + np.pi)
        Zh = hxs * np.sin(st["n_fp"] * uu + np.pi)
        _pv3d_add_tube(pv, pl, Rh * np.cos(uu), Rh * np.sin(uu), Zh,
                       st["hx_half"], st["hx_half"],
                       (np.cos(uu), np.sin(uu), np.zeros_like(uu)),
                       _PV3D_CS_COL, closed=False)
        legend_coils.append(("Central circular + helical conductors",
                             _PV3D_CS_COL))
        # Vertical field coil pair.
        if st.get("vf_R"):
            vf = [{"R": st["vf_R"], "Z":  st["vf_Z"],
                   "dR": 2 * st["vf_half"], "dZ": 2 * st["vf_half"]},
                  {"R": st["vf_R"], "Z": -st["vf_Z"],
                   "dR": 2 * st["vf_half"], "dZ": 2 * st["vf_half"]}]
            e = _draw_ring_coils(pv, pl, vf, _PV3D_PF_COL, 0.02)
            legend_coils.append(("Vertical field coils (2)", _PV3D_PF_COL))
            ext_R = max(ext_R, e["R_max"])
            ext_z = max(ext_z, e["z_max"])
        ext_R = max(ext_R, float(np.max(_stell_axis(p, uu)[0])) + r_tf
                    + half)
        ext_z = max(ext_z, st["axis_swing"] + r_tf + half)

    # Human scale figure, camera, legend and output: same conventions as
    # the tokamak presets.
    if human:
        R_h = ext_R + 0.7 + 0.04 * ext_R
        ph_h = np.deg2rad(_PV3D_GAP_CENTER_DEG + _PV3D_GAP_HALF_DEG + 6.0)
        az_c = np.deg2rad(-60.0)
        _pv3d_add_human(pv, pl, R_h * np.cos(ph_h), R_h * np.sin(ph_h),
                        -ext_z, facing=(np.cos(az_c), np.sin(az_c), 0.0))
    reach = 1.05 * max(ext_R, ext_z)
    pl.camera_position = _pv3d_camera(3.35 * reach)
    pl.reset_camera(render=False)
    pl.camera.zoom(1.28)

    legend = [("Plasma", _PV3D_PLASMA_COL)] + legend_coils
    if human:
        legend.append((f"Human figure ({_PV3D_HUMAN_H:.0f} m)",
                       _PV3D_HUMAN_COL))

    import tempfile, re
    stem = "preset_" + re.sub(r"[^A-Za-z0-9]+", "_", str(key)) + "_3D"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_png = os.path.join(save_dir, stem + ".png")
    else:
        out_png = os.path.join(tempfile.gettempdir(), stem + ".png")
    machine_tmp = out_png[:-4] + "_machine.png"
    legend_tmp = out_png[:-4] + "_legend.png"
    pl.screenshot(machine_tmp)
    pl.close()
    _pv3d_autotrim(machine_tmp)
    _pv3d_legend_strip(legend, ncol=2, path=legend_tmp)
    _pv3d_stack_legend(legend_tmp, machine_tmp, out_png)
    os.remove(machine_tmp); os.remove(legend_tmp)
    print(f"  Saved {out_png}")
    if save_dir is None:
        from PIL import Image
        img = np.asarray(Image.open(out_png))
        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.imshow(img); ax.axis("off")
        ax.set_title(p.get("name", key))
        plt.tight_layout()
        plt.show()


def plot_tokamak_3D_preset(preset, save_dir: str | None = None,
                           human: bool = True) -> None:
    """
    Render one machine preset as a D0FUS-style 3D concept view (PyVista).

    Parameters
    ----------
    preset   : str or dict
        Preset key (e.g. ``"ITER"``, see :data:`PRESETS`) or a full preset
        dict following the module schema.
    save_dir : str or None
        If provided, saves ``preset_<key>_3D.png`` there; otherwise the
        rendered image is displayed via matplotlib.
    human    : bool
        Draw the standing 2 m human silhouette at floor level (default).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  [skip] plot_tokamak_3D_preset: PyVista not installed "
              "(pip install pyvista) — 3D view skipped.")
        return
    pv.OFF_SCREEN = True

    p = get_preset(preset) if isinstance(preset, str) else dict(preset)
    key = p.get("key", p.get("name", "machine"))
    if p.get("type") == "stellarator":
        return _plot_stellarator_preset(pv, p, save_dir, human)
    plasma, tf = p["plasma"], p["tf"]

    pl = pv.Plotter(off_screen=True, window_size=[1500, 1150])
    pl.set_background("white")

    # Plasma
    Rpl_max, Zpl_max = _draw_plasma(pv, pl, plasma)
    ext_R, ext_zmin, ext_zmax = Rpl_max, -Zpl_max, Zpl_max

    # TF cage
    if tf["shape"] == "d":
        tf_ctx = _draw_tf_d(pv, pl, tf)
    elif tf["shape"] == "circular":
        tf_ctx = _draw_tf_circular(pv, pl, tf)
    elif tf["shape"] == "rect":
        tf_ctx = _draw_tf_rect(pv, pl, tf)
    elif tf["shape"] == "spherical":
        tf_ctx = _draw_tf_spherical(pv, pl, tf)
    else:
        raise ValueError(f"Unknown TF shape '{tf['shape']}'")
    ext_R = max(ext_R, tf_ctx["R_max"])
    ext_zmin = min(ext_zmin, tf_ctx["z_min"])
    ext_zmax = max(ext_zmax, tf_ctx["z_max"])

    # Solenoid stacks (CS and solenoid-like coils), with module gaps
    for s in p.get("solenoids", []):
        n_mod = max(1, int(s.get("modules", 1)))
        H = s["z_hi"] - s["z_lo"]
        g = 0.02 * H if n_mod > 1 else 0.0
        h = (H - (n_mod - 1) * g) / n_mod
        for k in range(n_mod):
            z_lo = s["z_lo"] + k * (h + g)
            _pv3d_cyl_sector(pv, pl, s["r_in"], s["r_out"], z_lo, z_lo + h,
                             _col(s["color"]))
        ext_zmin = min(ext_zmin, s["z_lo"])
        ext_zmax = max(ext_zmax, s["z_hi"])

    # Minimum drawable half-width: thin published sections stay visible
    min_half = max(0.02, 0.006 * tf_ctx["R_max"])

    # PF ring coils (published table or illustrative auto-placement)
    pf_spec = p.get("pf")
    if isinstance(pf_spec, dict) and "auto" in pf_spec:
        if tf_ctx["R_out_arr"] is None:
            raise ValueError("PF auto-placement requires a D-shaped TF")
        pf_coils = _auto_pf_positions(pf_spec["auto"], tf_ctx, plasma["a"])
    else:
        pf_coils = pf_spec or []
    if pf_coils:
        e = _draw_ring_coils(pv, pl, pf_coils, _PV3D_PF_COL, min_half)
        ext_R = max(ext_R, e["R_max"])
        ext_zmin = min(ext_zmin, e["z_min"])
        ext_zmax = max(ext_zmax, e["z_max"])

    # In-vessel / copper coils
    div_coils = p.get("div", [])
    if div_coils:
        e = _draw_ring_coils(pv, pl, div_coils, _PV3D_DIV_COL, min_half)
        ext_R = max(ext_R, e["R_max"])

    # Iron yoke (JET)
    if p.get("iron"):
        e = _draw_iron(pv, pl, p["iron"])
        ext_R = max(ext_R, e["R_max"])
        ext_zmin = min(ext_zmin, e["z_min"])
        ext_zmax = max(ext_zmax, e["z_max"])

    # Human scale figure (2 m), standing at floor level beside the machine
    if human:
        R_h = ext_R + 0.7 + 0.04 * ext_R
        ph = np.deg2rad(_PV3D_GAP_CENTER_DEG + _PV3D_GAP_HALF_DEG + 6.0)
        az_c = np.deg2rad(-60.0)                 # face the camera (frontal)
        _pv3d_add_human(pv, pl, R_h * np.cos(ph), R_h * np.sin(ph),
                        ext_zmin, facing=(np.cos(az_c), np.sin(az_c), 0.0))

    # Camera: fixed orientation, refit to include every actor
    reach = 1.05 * max(ext_R, ext_zmax, -ext_zmin)
    pl.camera_position = _pv3d_camera(3.35 * reach)
    pl.reset_camera(render=False)
    pl.camera.zoom(1.28)

    # Legend (deduplicated, in drawing order)
    legend = [("Plasma", _PV3D_PLASMA_COL)]
    for s in p.get("solenoids", []):
        entry = (s["label"], _col(s["color"]))
        if entry not in legend:
            legend.append(entry)
    tf_lab = {"d": f"TF coils ({tf.get('n', '?')}, D-shaped)",
              "circular": f"TF coils ({tf.get('n', '?')}, circular)",
              "rect": f"TF coils ({tf.get('n', '?')}, window pane)",
              "spherical": (f"TF centre rod + {tf.get('n_limbs', '?')} "
                            "return limbs")}[tf["shape"]]
    legend.append((p.get("tf_label", tf_lab), _PV3D_TF_COL))
    if pf_coils and p.get("pf_label"):
        entry = (p["pf_label"], _PV3D_PF_COL)
        if entry not in legend:
            legend.append(entry)
    if div_coils:
        legend.append((p.get("div_label", "In-vessel coils"), _PV3D_DIV_COL))
    if p.get("iron"):
        legend.append(("Iron transformer yoke", _PV3D_IRON_COL))
    if human:
        legend.append((f"Human figure ({_PV3D_HUMAN_H:.0f} m)",
                       _PV3D_HUMAN_COL))

    import tempfile, re
    stem = "preset_" + re.sub(r"[^A-Za-z0-9]+", "_", str(key)) + "_3D"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_png = os.path.join(save_dir, stem + ".png")
    else:
        out_png = os.path.join(tempfile.gettempdir(), stem + ".png")

    machine_tmp = out_png[:-4] + "_machine.png"
    legend_tmp = out_png[:-4] + "_legend.png"
    pl.screenshot(machine_tmp)
    pl.close()
    _pv3d_autotrim(machine_tmp)
    _pv3d_legend_strip(legend, ncol=2, path=legend_tmp)
    _pv3d_stack_legend(legend_tmp, machine_tmp, out_png)
    os.remove(machine_tmp); os.remove(legend_tmp)
    print(f"  Saved {out_png}")

    if save_dir is None:
        from PIL import Image
        img = np.asarray(Image.open(out_png))
        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.imshow(img); ax.axis("off")
        ax.set_title(p.get("name", key))
        plt.tight_layout()
        plt.show()


def plot_all_presets(save_dir: str | None = None, human: bool = True) -> None:
    """Render every preset of the bank, grouped by category."""
    groups = list_presets()
    for cat in ("operating", "under_construction", "reactor_project"):
        for key in groups.get(cat, []):
            print(f"[{CATEGORY_LABELS[cat]}] {key}")
            plot_tokamak_3D_preset(key, save_dir=save_dir, human=human)


# =============================================================================
# Self-test: geometric consistency of every preset
# =============================================================================

def _rect_pt_dist(rc, zc, dr, dz, rp, zp):
    """Poloidal-plane distance from a point to a rectangle (0 if inside)."""
    ddr = max(abs(rp - rc) - 0.5 * dr, 0.0)
    ddz = max(abs(zp - zc) - 0.5 * dz, 0.0)
    return float(np.hypot(ddr, ddz))


def _selftest() -> bool:
    """Check every preset for geometric consistency (no rendering)."""
    ok_all = True
    th = np.linspace(0.0, 2.0 * np.pi, 720)
    for key, p in PRESETS.items():
        msgs = []
        if p.get("type") == "stellarator":
            # Dedicated checks per family: toroidal closure of the
            # boundary, then coil-set clearance over the plasma envelope.
            st = p["stellarator"]
            fam = st.get("family", "helias")
            Ra, Za = _stell_surface(p, 0.0, 0.4)
            Rb, Zb = _stell_surface(p, 2.0 * np.pi, 0.4)
            if not (abs(Ra - Rb) < 1e-9 and abs(Za - Zb) < 1e-9):
                msgs.append("boundary does not close toroidally")
            # Largest plasma displacement from the local magnetic axis.
            d_max = 0.0
            thh = np.linspace(0.0, 2.0 * np.pi, 300)
            for u_k in np.linspace(0.0, 2.0 * np.pi, 40):
                Rpl, Zpl = _stell_surface(p, u_k, thh)
                aR, aZ = _stell_axis(p, u_k)
                d_max = max(d_max, float(np.hypot(Rpl - float(aR),
                                                  Zpl - float(aZ)).max()))
            if fam == "helias":
                semi_min = st.get("coil_ell", 0.92) * st["coil_rc"]
                if semi_min - st["coil_half"] <= d_max:
                    msgs.append("modular coils do not clear the plasma")
                if st.get("n_planar", 0) and \
                   st["planar_rc"] <= st["coil_rc"] + st["coil_half"]:
                    msgs.append("planar coils inside the modular coils")
            elif fam == "heliotron":
                # Rotating ellipse is centred on the axis: the helical
                # winding must clear its larger semi-axis.
                if st["coil_ac"] - st["coil_half"] <= max(st["a1"],
                                                          st["a2"]):
                    msgs.append("helical coils do not clear the plasma")
            elif fam == "heliac":
                if st["tf_r"] - st["tf_half"] <= st["a"]:
                    msgs.append("TF coils do not clear the plasma")
                # The plasma winds around the central conductor without
                # touching it.
                if st["axis_swing"] - st["a"] <= st["cc_half"]:
                    msgs.append("plasma touches the central conductor")
            status = "PASS" if not msgs else "FAIL: " + "; ".join(msgs)
            print(f"  {key:9s} {status}")
            ok_all &= not msgs
            continue
        plasma, tf = p["plasma"], p["tf"]
        _lcfs = _preset_lcfs(plasma)
        R, Z = _lcfs(th)

        # Plasma inside the TF bore.
        if tf["shape"] == "d":
            R_o, Z_o, R_i, Z_i = _d_contours(tf)
            if not (R.max() < np.max(R_i) and np.abs(Z).max() < np.max(Z_i)):
                msgs.append("plasma exceeds TF bore")
            cs_limit = tf["R_bore"]
            H_half = float(np.max(Z_o))
        elif tf["shape"] == "circular":
            d = np.hypot(R - tf["center_R"], Z).max()
            if d >= tf["r_in"] - 0.02:
                msgs.append("plasma exceeds circular TF bore")
            cs_limit = tf["center_R"] - tf["r_out"]
            H_half = tf["r_out"]
        elif tf["shape"] == "rect":
            if not (R.min() > tf["R_in_leg"] + 0.5 * tf["c"] and
                    R.max() < tf["R_out_leg"] - 0.5 * tf["c"] and
                    np.abs(Z).max() < tf["Z_half"] - 0.5 * tf["c"]):
                msgs.append("plasma exceeds window-pane TF bore")
            cs_limit = tf["R_in_leg"] - 0.5 * tf["c"]
            H_half = tf["Z_half"]
        else:  # spherical
            if R.min() <= tf["rod_r"]:
                msgs.append("plasma touches the centre rod")
            if R.max() >= tf["limb_R"] - tf["limb_half_r"]:
                msgs.append("plasma reaches the return limbs")
            cs_limit = None
            H_half = tf["rod_z"]

        # Solenoid stacks must not overlap the TF inner-leg winding band:
        # either fully inboard of it (classical CS) or fully outboard of
        # it, inside the bore (TCV air-core coil set).  Circular cages
        # (WEST) have their column outside each coil circle.
        if tf["shape"] == "d":
            band = (tf["R_bore"], tf["R_bore"] + tf["c"])
        elif tf["shape"] == "rect":
            band = (tf["R_in_leg"] - 0.5 * tf["c"],
                    tf["R_in_leg"] + 0.5 * tf["c"])
        else:
            band = None
        for s in p.get("solenoids", []):
            if band is not None and not (s["r_out"] <= band[0] + 1e-9 or
                                         s["r_in"] >= band[1] - 1e-9):
                msgs.append(f"solenoid '{s['label']}' overlaps the TF "
                            f"inner-leg winding ({band[0]:.2f}-"
                            f"{band[1]:.2f} m)")

        # Explicit coils clear of the plasma.
        coils = []
        if isinstance(p.get("pf"), list):
            coils += p["pf"]
        coils += p.get("div", [])
        for c in coils:
            dmin = min(_rect_pt_dist(c["R"], c["Z"], c["dR"], c["dZ"],
                                     rp, zp) for rp, zp in zip(R[::6], Z[::6]))
            inside = _rect_pt_dist(c["R"], c["Z"], c["dR"], c["dZ"],
                                   plasma["R0"], 0.0)
            if dmin <= 0.0 and inside == 0.0:
                msgs.append(f"coil at ({c['R']:.2f}, {c['Z']:.2f}) "
                            "overlaps the plasma")

        # Coil-to-coil overlap: no two drawn conductors may interpenetrate
        # in the poloidal plane (rectangles from the ring coils and the
        # solenoid stacks; real coils never overlap).
        rects = [(c["R"] - 0.5 * c["dR"], c["R"] + 0.5 * c["dR"],
                  c["Z"] - 0.5 * c["dZ"], c["Z"] + 0.5 * c["dZ"],
                  f"({c['R']:.2f},{c['Z']:+.2f})") for c in coils]
        rects += [(s["r_in"], s["r_out"], s["z_lo"], s["z_hi"],
                   s["label"]) for s in p.get("solenoids", [])]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if (a[0] < b[1] - 1e-9 and b[0] < a[1] - 1e-9 and
                        a[2] < b[3] - 1e-9 and b[2] < a[3] - 1e-9):
                    msgs.append(f"coils {a[4]} and {b[4]} overlap")

        # Informative only: ring coils crossing the drawn TF winding in the
        # poloidal plane.  Real coil positions take precedence over the
        # schematic TF contour, so this never fails the test; it flags
        # places where the drawn contour and published coils disagree.
        warns = []
        if tf["shape"] == "d" and coils:
            from matplotlib.path import Path as _MplPath
            inner = _MplPath(np.column_stack([R_i, Z_i]))
            outer = _MplPath(np.column_stack([R_o, Z_o]))
            for c in coils:
                pts = [(c["R"] + sx * 0.5 * c["dR"],
                        c["Z"] + sz * 0.5 * c["dZ"])
                       for sx in (-1, 0, 1) for sz in (-1, 0, 1)]
                ins = all(inner.contains_point(q) for q in pts)
                outs = not any(outer.contains_point(q) for q in pts)
                if not (ins or outs):
                    warns.append(f"({c['R']:.2f},{c['Z']:+.2f})")
        status = "PASS" if not msgs else "FAIL: " + "; ".join(msgs)
        if warns:
            status += (f"  [info: {len(warns)} coil(s) cross the drawn TF "
                       f"winding: {' '.join(warns)}]")
        print(f"  {key:9s} {status}")
        ok_all &= not msgs
    print("Preset self-test:", "PASSED" if ok_all else "FAILED")
    return ok_all


if __name__ == "__main__":
    _selftest()
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "D0FUS_OUTPUTS", "presets")
    plot_all_presets(save_dir=out)
