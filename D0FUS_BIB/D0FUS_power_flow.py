"""Phase-resolved D0FUS power-flow Sankey diagrams (matplotlib only).

Pulse plasma-exhaust accounting: P_alpha + P_CD = P_rad + P_sep.
The divertor diagnostic P_div = P_sep + P_Ohm is annotated separately,
because D0FUS does *not* add P_Ohm to the reactor-thermal equation.
Thus the flows entering the reactor-thermal collector sum to P_th.

Electrical accounting: gross = min(gross, recirculation) + grid export;
recirculation = min(gross, recirculation) + grid import.  Imports are
connected directly to the consumption bus, not to electrical generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch

# ----------------------------------------------------------------- palette
# One colour per energy carrier / physical meaning (no colour is reused for
# two different meanings).
FUSION = '#0879bc'     # total fusion power
ALPHA = '#4fb0d8'      # alpha / charged-particle power
PLASMA = '#2a5d9f'     # plasma node
NEUT = '#dd511d'       # neutrons and blanket nuclear gain
RAD = '#df4e97'        # radiation channels / first wall
CDCOL = '#b3175f'      # auxiliary heating / current drive power
SEP = '#812c91'        # separatrix / divertor exhaust
HEAT = '#efae18'       # reactor heat, turbine
STORE = '#1f9e8f'      # thermal storage
ELEC = '#73a62b'       # electricity (generation, bus, loads)
IMPORT = '#d65361'     # grid import
LOSS = '#8d99a7'       # losses / rejected heat
TEXT = '#162d3e'
MUTED = '#5a6d77'
BG = '#ffffff'

# ---------------------------------------------------------------- geometry
UPI = 1.29              # data units per inch (figure size follows the content)
X_MARG = .9             # left / right margin
PITCH = 2.42            # distance between columns
NODE_W = .20            # node bar thickness (data units, ~1 inch per unit)
GAP = .74               # min vertical gap between nodes (room for labels)
MIN_W = .04             # visibility floor for very thin flows
REF_H = 4.2             # drawn height of a flow equal to the reference MW
MAIN = (1.55, 8.55)     # y-range available to nodes (single row, all columns)
Y_FLOOR = .70           # centreline of the feedback (P_CD) return band
REGIONS = {'top': MAIN}


# ----------------------------------------------------- data access (unchanged)
def _get(run, key, required=False):
    try:
        v = float(run[key])
        if np.isfinite(v):
            return v
    except (KeyError, ValueError, TypeError):
        pass
    if required:
        raise ValueError(f'Power-flow figure requires a valid D0FUS result: {key}')
    return None


def _approx(a, b, scale, where):
    if abs(a - b) > max(1e-5, 1e-5 * scale):
        raise ValueError(f'D0FUS {where} values are inconsistent: {a:.6g} vs {b:.6g} MW')


def _mw(v):
    return f'{v:,.1f} MW' if abs(v) < 100 else f'{v:,.0f} MW'


def _load_data(run, phase):
    pulse = phase == 'pulse'
    gross = _get(run, f'P_e_gross_{phase}', True)
    net = _get(run, f'P_e_net_{phase}', True)
    var = _get(run, 'P_recirc_var', True) if pulse else 0.
    fix = _get(run, 'P_recirc_fix', True)
    if gross < -1e-8 or var < -1e-8 or fix < -1e-8:
        raise ValueError('Negative gross generation or recirculating load: invalid run data')
    gross = max(0., gross)
    recirc = var + fix
    _approx(net, gross - recirc, max(gross, recirc, 1.), 'phase electrical balance')
    if pulse:
        keys = [('Heating / CD', 'P_wallplug_calculated'),
                ('CS supply', 'P_coil_CS'),
                ('BoP pumps', 'P_BoP'),
                ('Cryogenics', 'P_cryo_electric'),
                ('House loads', 'P_house_load')]
    else:
        keys = [('Cryogenics', 'P_cryo_electric'), ('House loads', 'P_house_load')]
    loads = [(n, _get(run, k, True)) for n, k in keys]
    if any(v < -1e-8 for _, v in loads):
        raise ValueError('Negative D0FUS electrical-load components')
    _approx(sum(v for _, v in loads), recirc, max(recirc, 1.), 'recirculation decomposition')
    house = [(n, _get(run, k)) for n, k in (
        ('Tritium plant', 'P_tritium_balance'), ('Vacuum', 'P_vacuum_balance'),
        ('Facilities', 'P_facilities_balance'), ('TF supplies', 'P_TF_supplies_balance'))]
    if all(v is not None for _, v in house):
        _approx(sum(v for _, v in house), loads[-1][1], max(recirc, 1.),
                'house-load decomposition')
    else:
        house = []
    return gross, net, loads, recirc, house


# ---------------------------------------------------------- small helpers
def _shade(hex_, f=.62):
    h = hex_.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % (int(r * f), int(g * f), int(b * f))


def _round_path(pts, r, n=14):
    """Polyline with quadratic-Bezier filleted corners (for the feedback band)."""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        v1, v2 = p0 - p1, p2 - p1
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        rr = min(r, .5 * l1, .5 * l2)
        a, b = p1 + v1 / l1 * rr, p1 + v2 / l2 * rr
        t = np.linspace(0, 1, n)[:, None]
        out.extend((1 - t) ** 2 * a + 2 * (1 - t) * t * p1 + t ** 2 * b)
    out.append(pts[-1])
    out = np.asarray(out)
    keep = np.r_[True, np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9]
    return out[keep]


def _offset_band(xy, w):
    """Constant-width band around a smooth centreline (normal offsets)."""
    d = np.gradient(xy, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return np.vstack([xy + nrm * w / 2, (xy - nrm * w / 2)[::-1]])


# ------------------------------------------------------------ Sankey engine
@dataclass
class Node:
    key: str
    label: str
    col: int
    color: str
    region: str = 'top'
    order: float = 0.
    gap: float = GAP        # min clearance to the neighbour node (room for labels)
    lab: str = 'c'          # label alignment: 'c' centred on the bar, 'r' ends at the bar
    value: float = 0.
    x: float = 0.
    y1: float = 0.          # top edge (y axis points up)
    h: float = 0.
    ins: list = field(default_factory=list)
    outs: list = field(default_factory=list)

    @property
    def yc(self):
        return self.y1 - self.h / 2


@dataclass
class Link:
    src: str
    tgt: str
    mw: float
    color: str
    feedback: bool = False
    label: str | None = None
    label_t: float = .5
    w: float = 0.
    ys: float = 0.
    yt: float = 0.


class Sankey:
    """Columns are given, vertical positions are computed.

    Band thickness = max(K * MW, MIN_W); K is common to every band of the
    figure.  Links entering / leaving a node are stacked along its face in
    the order of the other end's height, so bands never cross at a node.
    """

    def __init__(self, k_ref):
        self.nodes: dict[str, Node] = {}
        self.links: list[Link] = []
        self.k_ref = k_ref
        self.K = k_ref
        self.floored = []            # links drawn at the minimum width

    # -- construction ------------------------------------------------------
    def node(self, key, label, col, color, region='top', order=0., **kw):
        self.nodes[key] = Node(key, label, col, color, region, order, **kw)

    def link(self, src, tgt, mw, color, **kw):
        if mw is None or mw <= 1e-8:
            return
        l = Link(src, tgt, float(mw), color, **kw)
        self.links.append(l)
        self.nodes[src].outs.append(l)
        self.nodes[tgt].ins.append(l)

    # -- computation -------------------------------------------------------
    def _groups(self):
        groups = {}
        for n in self.nodes.values():
            groups.setdefault((n.col, n.region), []).append(n)
        for ns in groups.values():
            ns.sort(key=lambda n: n.order)
        return groups

    def _prepare(self):
        self.nodes = {k: n for k, n in self.nodes.items() if n.ins or n.outs}
        for n in self.nodes.values():
            n.value = max(sum(l.mw for l in n.ins), sum(l.mw for l in n.outs))
        K = self.k_ref
        for (_, reg), ns in self._groups().items():
            lo, hi = REGIONS[reg]
            avail = (hi - lo) - sum(self._gaps(ns))
            tot = sum(n.value for n in ns)
            if tot > 0:
                K = min(K, .9 * avail / tot)
        self.K = K
        for l in self.links:
            l.w = max(K * l.mw, MIN_W)
            if K * l.mw < MIN_W:
                self.floored.append(l)
        for n in self.nodes.values():
            n.h = max(sum(l.w for l in n.ins), sum(l.w for l in n.outs))

    @staticmethod
    def _gaps(ns):
        return [max(a.gap, b.gap) for a, b in zip(ns[:-1], ns[1:])]

    def _collide(self, ns, reg):
        """Smallest total displacement that removes overlaps (isotonic fit)."""
        lo, hi = REGIONS[reg]
        gaps = self._gaps(ns) + [0.]
        cum = np.r_[0., np.cumsum([n.h + g for n, g in zip(ns, gaps)])[:-1]]
        t = [n.y1 + c for n, c in zip(ns, cum)]       # must be non-increasing
        blocks = []
        for v in t:
            blocks.append([v, 1])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
                sm, ct = blocks.pop()
                blocks[-1][0] += sm
                blocks[-1][1] += ct
        fit = [sm / ct for sm, ct in blocks for _ in range(ct)]
        for n, f, c in zip(ns, fit, cum):
            n.y1 = f - c
        over = ns[0].y1 - hi
        if over > 0:
            for n in ns:
                n.y1 -= over
        under = lo - (ns[-1].y1 - ns[-1].h)
        if under > 0:
            for n in ns:
                n.y1 += under

    def _layout(self, iters=60):
        cols = sorted({n.col for n in self.nodes.values()})
        ncols = max(cols) + 1
        pitch = PITCH
        self.pitch = pitch
        self.x0 = X_MARG
        for n in self.nodes.values():
            n.x = self.x0 + n.col * pitch
        groups = self._groups()
        for (_, reg), ns in groups.items():     # centred initial stacks
            lo, hi = REGIONS[reg]
            gaps = self._gaps(ns) + [0.]
            tot = sum(n.h for n in ns) + sum(gaps)
            y = hi - max(0., (hi - lo - tot) / 2)
            for n, gp in zip(ns, gaps):
                n.y1 = y
                y -= n.h + gp

        def desired(n, use_ins):
            ls = [l for l in (n.ins if use_ins else n.outs) if not l.feedback]
            if not ls:
                return None
            tot = sum(l.w for l in ls)
            return sum(l.w * self.nodes[l.src if use_ins else l.tgt].yc
                       for l in ls) / tot

        by_col = {c: [n for n in self.nodes.values() if n.col == c] for c in cols}
        for _ in range(iters):
            for c in cols[1:]:
                for n in by_col[c]:
                    d = desired(n, True)
                    if d is not None:
                        n.y1 += (d - n.yc) * .8
                for (cc, reg), ns in groups.items():
                    if cc == c:
                        self._collide(ns, reg)
            for c in reversed(cols[:-1]):
                for n in by_col[c]:
                    d = desired(n, False)
                    if d is not None:
                        n.y1 += (d - n.yc) * .8
                for (cc, reg), ns in groups.items():
                    if cc == c:
                        self._collide(ns, reg)

    def _centre_top(self):
        ns = [n for n in self.nodes.values() if n.region == 'top']
        if not ns:
            return
        lo, hi = MAIN
        ymin = min(n.y1 - n.h for n in ns)
        ymax = max(n.y1 for n in ns)
        shift = (lo + hi) / 2 - (ymin + ymax) / 2
        for n in ns:
            n.y1 += shift

    def _ports(self):
        for n in self.nodes.values():
            outs = sorted(n.outs, key=lambda l: (l.feedback, -self.nodes[l.tgt].yc))
            y = n.y1 - (n.h - sum(l.w for l in outs)) / 2
            for l in outs:
                l.ys = y - l.w / 2
                y -= l.w
            ins = sorted(n.ins, key=lambda l: (l.feedback, -self.nodes[l.src].yc))
            y = n.y1 - (n.h - sum(l.w for l in ins)) / 2
            for l in ins:
                l.yt = y - l.w / 2
                y -= l.w

    # -- geometry helpers --------------------------------------------------
    def fig_width(self):
        cols = [n.col for n in self.nodes.values() if n.ins or n.outs]
        return 2 * X_MARG + PITCH * max(cols) + NODE_W

    # -- drawing -----------------------------------------------------------
    def _draw_link(self, ax, l):
        s, t = self.nodes[l.src], self.nodes[l.tgt]
        halo = [pe.withStroke(linewidth=3.2, foreground=BG)]
        if l.feedback:
            r = max(.6, 1.25 * l.w)
            xa = s.x + NODE_W + .95
            x_up = t.x - .95
            path = _round_path([(s.x + NODE_W / 2, l.ys), (xa, l.ys), (xa, Y_FLOOR),
                                (x_up, Y_FLOOR), (x_up, l.yt),
                                (t.x + NODE_W / 2, l.yt)], r)
            ax.add_patch(Polygon(_offset_band(path, l.w), closed=True, fc=l.color,
                                 ec='none', alpha=.62, zorder=2))
            if l.label:
                ax.text(.5 * (x_up + xa), Y_FLOOR + l.w / 2 + .16, l.label, ha='center',
                        va='bottom', fontsize=9.6, color=_shade(l.color, .8),
                        weight='bold', path_effects=halo, zorder=8)
            return
        xs, xt = s.x + NODE_W / 2, t.x + NODE_W / 2
        xm = .5 * (xs + xt)
        u = np.linspace(0, 1, 90)
        x = (1-u)**3*xs + 3*(1-u)**2*u*xm + 3*(1-u)*u**2*xm + u**3*xt
        y = (1-u)**3*l.ys + 3*(1-u)**2*u*l.ys + 3*(1-u)*u**2*l.yt + u**3*l.yt
        poly = np.vstack([np.c_[x, y + l.w / 2], np.c_[x, y - l.w / 2][::-1]])
        ax.add_patch(Polygon(poly, closed=True, fc=l.color, ec='none', alpha=.62, zorder=2))
        if l.label:
            i = int(l.label_t * (len(u) - 1))
            ax.text(x[i], y[i] + l.w / 2 + .12, l.label, ha='center', va='bottom',
                    fontsize=9.4, color=_shade(l.color, .75), weight='bold',
                    path_effects=halo, zorder=8)

    def render(self, ax):
        self._prepare()
        self._layout()
        self._centre_top()
        self._ports()
        for l in sorted(self.links, key=lambda l: -l.w):    # thin bands on top
            self._draw_link(ax, l)
        halo = [pe.withStroke(linewidth=3.4, foreground=BG)]
        for n in self.nodes.values():
            ax.add_patch(Rectangle((n.x, n.y1 - n.h), NODE_W, n.h, fc=n.color,
                                   ec='none', zorder=4))
            xc, ha = ((n.x + NODE_W, 'right') if n.lab == 'r'
                      else (n.x + NODE_W / 2, 'center'))
            ax.text(xc, n.y1 + .36, n.label, ha=ha, va='bottom', fontsize=10.5,
                    weight='bold', color=TEXT, path_effects=halo, zorder=8)
            ax.text(xc, n.y1 + .08, _mw(n.value), ha=ha, va='bottom',
                    fontsize=10, color=_shade(n.color), path_effects=halo, zorder=8)


# ------------------------------------------------------------ figure frame
def _base(width):
    fig = plt.figure(figsize=(width / UPI, 10. / UPI))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, width)
    ax.axis('off')
    return fig, ax


def _finish(fig, ax, width, phase, y_bottom, y_content_top, net_text=None, net_color=None):
    """Title / NET tag just above the content, then crop the figure to the content."""
    y_title = y_content_top + .8
    ax.text(X_MARG, y_title, 'Sankey Diagram \u2013 ' + ('Pulse' if phase == 'pulse' else 'Dwell'),
            fontsize=26, color=TEXT, weight='bold', ha='left', va='center')
    if net_text:
        ax.text(width - X_MARG, y_title, net_text, fontsize=20, weight='bold',
                color=net_color, ha='right', va='center')
    y_top = y_title + .45
    ax.set_ylim(y_bottom, y_top)
    fig.set_size_inches(width / UPI, (y_top - y_bottom) / UPI)


# --------------------------------------------------------- graph builders
def _turbine_block(g, heat_in, gross, col_t, col_e, heat_color, order=0.):
    g.node('turbine', 'TURBINE', col_t, heat_color, 'top', order)
    g.node('elec', 'ELECTRICITY', col_e, ELEC, 'top', 0)
    g.node('rejected', 'Rejected heat', col_e, LOSS, 'top', 1, lab='r')
    g.link('turbine', 'elec', gross, ELEC)
    if heat_in > gross + 1e-8:
        g.link('turbine', 'rejected', heat_in - gross, LOSS)


def _pulse_graph(g, run):
    pf = _get(run, 'P_fus', True)
    pn = _get(run, 'P_neutron_balance', True)
    pa = _get(run, 'P_alpha_balance', True)
    pcd = _get(run, 'P_CD', True)
    mult = _get(run, 'P_blanket_nuclear_gain', True)
    prad = _get(run, 'P_rad_total_balance', True)
    sep = _get(run, 'P_sep', True)
    ohm = _get(run, 'P_Ohm')
    div = _get(run, 'P_div_balance')
    pth = _get(run, 'P_th', True)
    to_turb = _get(run, 'P_th_reactor_to_turbine_pulse', True)
    to_store = _get(run, 'P_th_reactor_to_storage_pulse', True)
    gross = _get(run, 'P_e_gross_pulse', True)
    _approx(pn + pa, pf, max(1, pf), 'fusion branch split')
    _approx(pa + pcd, prad + sep, max(1, pf), 'plasma radiation / separatrix split')
    _approx(pn + mult + prad + sep, pth, max(1, pth), 'reactor thermal closure')
    _approx(to_turb + to_store, max(0., pth), max(1, pth), 'thermal storage / turbine split')
    if min(pf, pn, pa, pcd, prad, sep, to_turb, to_store, gross) < -1e-7:
        raise ValueError('A physical power stream is negative; cannot represent as positive Sankey ribbon')

    g.node('fusion', 'FUSION', 0, FUSION)
    g.node('gain', 'BLANKET GAIN', 1, NEUT, order=0)
    g.node('neut', 'NEUTRONS', 1, NEUT, order=1)
    g.node('plasma', 'PLASMA', 1, PLASMA, order=2)
    g.link('fusion', 'neut', pn, NEUT)
    g.link('fusion', 'plasma', pa, ALPHA, label='\u03b1 / charged', label_t=.45)
    # The multiplier is an *additional* nuclear energy source, not a second
    # independent neutron transport stream (original comment preserved).
    g.node('blanket', 'BLANKET', 3, HEAT, order=0)
    g.link('gain', 'blanket', mult, NEUT)
    g.link('neut', 'blanket', pn, NEUT)

    g.node('fw', 'FIRST WALL', 3, RAD, order=1)
    rad_parts = [(name, _get(run, key)) for name, key in (
        ('Brems', 'P_Brem'), ('Synch.', 'P_syn'), ('Line', 'P_line'))]
    if all(v is not None for _, v in rad_parts) and all(v >= 0 for _, v in rad_parts):
        _approx(sum(v for _, v in rad_parts), prad, max(1, prad), 'radiation channel split')
        names = {'Brems': 'Bremsstrahlung', 'Synch.': 'Synchrotron rad.', 'Line': 'Line radiation'}
        for i, (name, p) in enumerate(rad_parts):
            k = f'rad_{i}'
            g.node(k, names[name], 2, RAD, order=i, gap=1.1)
            g.link('plasma', k, p, RAD)
            g.link(k, 'fw', p, RAD)
    else:
        g.link('plasma', 'fw', prad, RAD)

    g.node('div', 'DIVERTOR', 2, SEP, order=3)
    g.link('plasma', 'div', sep, SEP)
    if div is not None and ohm is not None:
        _approx(sep + ohm, div, max(1, pf), 'divertor diagnostic')

    g.node('heat', 'REACTOR HEAT', 4, HEAT)
    g.link('blanket', 'heat', pn + mult, HEAT)
    g.link('fw', 'heat', prad, RAD)
    g.link('div', 'heat', sep, SEP)

    _turbine_block(g, to_turb, gross, 5, 6, HEAT, order=0)
    g.node('storage', 'STORAGE CHARGE', 5, STORE, order=1)
    g.link('heat', 'turbine', to_turb, HEAT)
    g.link('heat', 'storage', to_store, STORE)
    return 6      # column of the ELECTRICITY node


def _dwell_graph(g, run):
    dis = _get(run, 'P_th_storage_to_turbine_dwell', True)
    gross = _get(run, 'P_e_gross_dwell', True)
    charge = _get(run, 'P_th_reactor_to_storage_pulse', True)
    tp = _get(run, 't_pulse_s')
    df = _get(run, 'Dwell_factor')
    if dis < -1e-8 or gross < -1e-8:
        raise ValueError('Negative D0FUS dwell heat or electricity')
    # Pulse/dwell heat rates cannot be subtracted directly. Derive energy
    # input and loss on a *dwell-equivalent* power basis from D0FUS data.
    dwell_equiv = None
    storage_loss = None
    if df is not None and 0 < df < 1 and tp is not None and tp > 0:
        td = tp * (1 - df) / df
        dwell_equiv = charge * tp / td
        storage_loss = dwell_equiv - dis
        if storage_loss < -1e-5 * max(1, dis, dwell_equiv):
            raise ValueError('Computed D0FUS storage return exceeds energy charged')
        storage_loss = max(0., storage_loss)
    g.node('storage', 'STORAGE', 0, STORE)
    g.node('loss', 'Storage losses', 1, LOSS, order=1)
    _turbine_block(g, dis, gross, 1, 2, STORE, order=0)
    g.link('storage', 'turbine', dis, STORE)
    if dwell_equiv is not None and dwell_equiv > 0:
        g.link('storage', 'loss', storage_loss, LOSS)
    return 2


def _electric_graph(g, run, phase, gross, net, loads, recirc, house, c_el):
    """Bus is fed by gross electricity and (when needed) grid imports."""
    pulse = phase == 'pulse'
    rec_gross = min(gross, recirc)
    grid_in = max(0., recirc - gross)
    exported = max(0., gross - recirc)

    g.node('export', 'GRID EXPORT', c_el + 1, ELEC, 'top', 0)
    g.node('import', 'GRID IMPORT', c_el, IMPORT, 'top', 2)
    g.node('bus', 'RECIRCULATION', c_el + 1, ELEC, 'top', 1)
    g.link('elec', 'export', exported, ELEC)
    g.link('elec', 'bus', rec_gross, ELEC)
    g.link('import', 'bus', grid_in, IMPORT)

    pref = ['CS supply', 'BoP pumps', 'Cryogenics', 'House loads', 'Heating / CD']
    ordered = sorted(loads, key=lambda t: pref.index(t[0]))
    keymap = {}
    for i, (name, mw) in enumerate(ordered):
        k = 'load_' + name
        keymap[name] = k
        g.node(k, name, c_el + 2, ELEC, 'top', i)
        g.link('bus', k, mw, ELEC)

    if house:
        for i, (name, amount) in enumerate(house):
            k = 'house_' + name
            g.node(k, name, c_el + 3, ELEC, 'top', i)
            g.link(keymap['House loads'], k, amount, ELEC)

    if pulse:
        wallplug = dict(loads)['Heating / CD']
        cd = _get(run, 'P_CD', True)
        _approx(wallplug, _get(run, 'P_wallplug_calculated', True), max(recirc, 1.),
                'H&CD supply')
        g.node('hcd_loss', 'H&CD LOSS', c_el + 3, LOSS, 'top', 10)
        g.link(keymap['Heating / CD'], 'hcd_loss', wallplug - cd, LOSS)
        if cd > 0:
            g.link(keymap['Heating / CD'], 'plasma', cd, CDCOL, feedback=True,
                   label=f'\u25c0  Auxiliary heat / P_CD back to plasma   {_mw(cd)}')
    return grid_in, exported


# --------------------------------------------------------------- public API
def plot_plant_power_balance(run, phase='pulse', save_dir=None):
    """Draw and save a connected, width-scaled D0FUS pulse/dwell flow map.

    Saves `plant_power_balance_<phase>.png` in the existing plot_run folder.
    Does not change any calculation, balance or phase-dependent allocation.
    """
    if phase not in ('pulse', 'dwell'):
        raise ValueError('phase must be pulse or dwell')
    gross, net, loads, recirc, house = _load_data(run, phase)
    # Same reference MW for both phases -> ribbon widths comparable.
    scale = max(1., gross, recirc, _get(run, 'P_fus') or 0., _get(run, 'P_th') or 0.)
    df = _get(run, 'Dwell_factor')
    # D0FUS sets P_e_net_dwell = P_e_net_pulse when Dwell_factor = 1
    # (convenience), but NO dwell interval actually exists.
    no_dwell = phase == 'dwell' and df is not None and df >= 1.0

    g = None
    if not no_dwell:
        g = Sankey(k_ref=REF_H / scale)
        c_el = _pulse_graph(g, run) if phase == 'pulse' else _dwell_graph(g, run)
        grid_in, exported = _electric_graph(g, run, phase, gross, net, loads,
                                            recirc, house, c_el)
    width = g.fig_width() if g is not None else 14.0
    fig, ax = _base(width)

    def _save():
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f'plant_power_balance_{phase}.png'),
                        dpi=170, facecolor=fig.get_facecolor())
            plt.close(fig)
        else:
            plt.show()

    try:
        if no_dwell:
            ax.add_patch(FancyBboxPatch((width / 2 - 4.8, 4.5), 9.6, 1.6,
                                        boxstyle='round,pad=.1,rounding_size=.25',
                                        fc='white', ec=SEP, lw=2.2))
            ax.text(width / 2, 5.3, 'NO DWELL INTERVAL', ha='center', va='center',
                    fontsize=22, weight='bold', color=SEP)
            _finish(fig, ax, width, phase, 3.8, 6.6)
            _save()
            return
        g.render(ax)
        y_top = max(n.y1 for n in g.nodes.values()) + .65      # label height
        y_bot = min(n.y1 - n.h for n in g.nodes.values()) - .45
        if any(l.feedback for l in g.links):
            y_bot = Y_FLOOR - .35
        if grid_in > 0:
            _finish(fig, ax, width, phase, y_bot, y_top, 'NET  \u2212' + _mw(grid_in), IMPORT)
        else:
            _finish(fig, ax, width, phase, y_bot, y_top, 'NET  +' + _mw(exported),
                    _shade(ELEC, .8))
        _save()
    except Exception:
        plt.close(fig)
        raise
