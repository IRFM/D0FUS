"""Phase-resolved D0FUS power-flow Sankey diagrams (matplotlib only).

Ribbons encode actual connected transfers, unlike a box-based KPI plot.
Every number comes from the converged run dict or from an explicit balance
of those numbers.  No equation in the D0FUS power model is modified.

Pulse plasma-exhaust accounting: P_alpha + P_CD = P_rad + P_sep.
The divertor diagnostic P_div = P_sep + P_Ohm is annotated separately,
because D0FUS does *not* add P_Ohm to the reactor-thermal equation.
Thus the flows entering the reactor-thermal collector sum to P_th.

Electrical accounting: gross = min(gross, recirculation) + grid export;
recirculation = min(gross, recirculation) + grid import.  Imports are
connected directly to the consumption bus, not to electrical generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

BLUE = '#0879bc'
NEUT = '#dd511d'
ALPHA = '#138cb3'
HEAT = '#efae18'
PURPLE = '#812c91'
ELEC = '#73a62b'
RECUR = '#6b9e24'
PINK = '#df4e97'
LOSS = '#8d99a7'
IMPORT = '#d65361'
TEXT = '#162d3e'
MUTED = '#5a6d77'
BG = '#fbfcfb'


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


def _catmull(points, n=16):
    """Smooth joined centreline through the supplied waypoints."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    out = []
    padded = np.vstack([pts[0], pts, pts[-1]])
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = padded[i:i+4]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        curve = .5 * (2*p1 + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t**2
                       + (-p0+3*p1-3*p2+p3)*t**3)
        out.extend(curve)
    out.append(pts[-1])
    return np.asarray(out)


@dataclass
class Canvas:
    ax: object
    scale: float

    def ribbon(self, pts, mw, color, arrow=True, z=2, alpha=.93, dash=False):
        """Continuous, MW-width-linked ribbon travelling between named units."""
        if mw is None or mw <= 1e-8:
            return
        xs = _catmull(pts)
        # Linear absolute width (pt / MW); sub-40 MW streams use a visibility
        # floor.  The floor is signalled in the figure legend.
        lw = max(1.85, min(70, 61 * mw / self.scale))
        self.ax.plot(xs[:,0], xs[:,1], color=color, linewidth=lw, alpha=alpha,
                     solid_capstyle='butt', solid_joinstyle='round', zorder=z,
                     linestyle='--' if dash else '-')
        if arrow and len(xs)>6:
            # Small triangle drawn on top of the flow in the actual direction.
            j = max(2, min(len(xs)-3, int(.76 * len(xs))))
            d = xs[j+2] - xs[j-2]
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                unit = d / norm
                ortho = np.array([-unit[1], unit[0]])
                size = min(.14, max(.065, lw / 450))
                apex = xs[j] + unit * size
                tail = xs[j] - unit * size
                tri = np.vstack([apex, tail + ortho*size*.60, tail-ortho*size*.60])
                self.ax.add_patch(Polygon(tri, color='#ffffff', alpha=.82,
                                          ec='none', zorder=z+.2))

    def unit(self, x, y, title, mw=None, color=HEAT, *, align='center',
             note=None, fs=10, dy=.52, width=None, filled=False):
        """Rounded equipment symbol overlaying adjoining flow endpoints."""
        if width is None:
            width = max(1.25, min(2.55, 0.15*len(title)+.70))
        height = .75 if note is None else .99
        x0 = x-width/2
        box = FancyBboxPatch((x0, y-height/2), width, height,
                 boxstyle='round,pad=.06,rounding_size=.16',
                 facecolor=(color if filled else '#ffffff'),
                 edgecolor=color, linewidth=2.1, zorder=9)
        self.ax.add_patch(box)
        col = 'white' if filled else TEXT
        self.ax.text(x, y+.16 if mw is not None else y, title, ha='center',
                     va='center', fontsize=fs, color=col, weight='bold', zorder=11)
        if mw is not None:
            self.ax.text(x, y-.19, _mw(mw), ha='center', va='center',
                         fontsize=fs, weight='semibold', color=col if filled else color, zorder=11)
        if note is not None:
            self.ax.text(x, y-.38, note, fontsize=7.4, ha='center', va='center',
                         color=col, zorder=11)

    def tag(self, x, y, text, color=MUTED, *, fs=9, ha='center', bold=False, rotation=0):
        self.ax.text(x,y,text,ha=ha,va='center',fontsize=fs,
                     color=color,rotation=rotation,
                     weight='bold' if bold else 'normal', zorder=12,
                     bbox={'fc':BG, 'alpha':.88, 'ec':'none', 'pad':1.35})

    def joint(self, x, y, mw, color, z=6, alpha=.97, edge=False):
        if mw is None or mw <= 1e-8:
            return
        lw = max(1.85, min(70, 61 * mw / self.scale))
        radius = 0.020 + lw / 205.0
        self.ax.add_patch(Circle((x, y), radius=radius,
                                 facecolor=color,
                                 edgecolor=(color if not edge else '#ffffff'),
                                 linewidth=(0 if not edge else 1.0),
                                 alpha=alpha, zorder=z))


def _mw(v):
    return f'{v:,.1f} MW' if abs(v) < 100 else f'{v:,.0f} MW'


def _base(phase, scale):
    fig, ax = plt.subplots(figsize=(24, 15.0))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(.0,24); ax.set_ylim(0,15.5); ax.set_aspect('auto'); ax.axis('off')
    phase_title = 'Pulse' if phase == 'pulse' else 'Dwell'
    ax.text(.55,14.85, f'Sankey Diagram - {phase_title}',
            fontsize=24, color=TEXT, weight='bold', ha='left')
    ax.axhline(6.08, xmin=.02,xmax=.98, color='#d8dedb', lw=.9, zorder=0)
    ax.text(.52,13.9,'FUSION PHYSICS   HEAT COLLECTION   ELECTRICITY',
            color=MUTED, weight='bold', fontsize=10.5)
    ax.text(.52,5.76,'ELECTRICITY RECIRCULATION',
            color=MUTED, weight='bold', fontsize=10.5)
    return fig, ax, Canvas(ax,scale)


def _load_data(run, phase):
    pulse = phase == 'pulse'
    gross = _get(run, f'P_e_gross_{phase}', True)
    net = _get(run, f'P_e_net_{phase}', True)
    var = _get(run, 'P_recirc_var', True) if pulse else 0.
    fix = _get(run, 'P_recirc_fix', True)
    if gross < -1e-8 or var < -1e-8 or fix < -1e-8:
        raise ValueError('Negative gross generation or recirculating load: invalid run data')
    gross = max(0.,gross)
    recirc = var+fix
    _approx(net,gross-recirc,max(gross,recirc,1.),'phase electrical balance')
    if pulse:
        keys = [ ('Heating / CD', 'P_wallplug_calculated'),
                 ('CS supply', 'P_coil_CS'),
                 ('BoP pumps', 'P_BoP'),
                 ('Cryogenics', 'P_cryo_electric'),
                 ('House loads', 'P_house_load') ]
    else:
        keys = [('Cryogenics', 'P_cryo_electric'),('House loads', 'P_house_load')]
    loads = [(n, _get(run,k,True)) for n,k in keys]
    if any(v < -1e-8 for _,v in loads):
        raise ValueError('Negative D0FUS electrical-load components')
    _approx(sum(v for _,v in loads),recirc,max(recirc,1.),'recirculation decomposition')
    house = [(n, _get(run,k)) for n,k in (
        ('Tritium plant','P_tritium_balance'),('Vacuum','P_vacuum_balance'),
        ('Facilities','P_facilities_balance'),('TF supplies','P_TF_supplies_balance'))]
    if all(v is not None for _,v in house):
        _approx(sum(v for _,v in house),loads[-1][1],max(recirc,1.),'house-load decomposition')
    else:
        house = []
    return gross, net, loads, recirc, house


def _electric_network(c, run, phase, gross, net, loads, recirc, house):
    """Bus is fed by gross electricity and (when needed) grid imports.

    Actual downstream branches connect to H&CD, CS, BoP, cryogenics, house;
    house is further split into the four existing PROCESS load contributions.
    """
    pulse = phase=='pulse'
    rec_gross = min(gross,recirc)
    grid_in = max(0.,recirc-gross)
    exported = max(0.,gross-recirc)
    if exported > 0:
        c.ribbon([(20.8,11.55),(22.5,11.55),(23.3,12.1)],exported,ELEC)
        c.unit(22.8,12.46,'GRID EXPORT',exported,ELEC,width=1.86,fs=9)
    if rec_gross > 0:
        c.ribbon([(21.1,11.35),(21.1,9.0),(21.1,6.2),(21.1,5.00)],rec_gross,ELEC)
        c.joint(21.1,5.0,rec_gross,ELEC,z=5)
    if grid_in > 0:
        c.ribbon([(23.35,5.00),(22.6,5.00),(21.1,5.00)],grid_in,IMPORT)
        c.unit(22.65,4.03,'GRID IMPORT',grid_in,IMPORT,width=2.22,fs=10)
        c.joint(21.1,5.0,grid_in,IMPORT,z=6)
    # For legibility distribute loads from RIGHT TO LEFT, with the ribbon
    # becoming thinner after every physical branch into a consumer.
    xs = ([5.2,8.25,11.3,14.35,17.4] if pulse else [14.5,17.65])
    remain = recirc
    prev_x = 21.1
    for (name,mw),x in reversed(list(zip(loads,xs))):
        if remain > 1e-8:
            c.ribbon([(prev_x,5.0),(x,5.0)],remain,RECUR,arrow=True,z=3)
        if mw > 1e-8:
            c.ribbon([(x,5.0),(x,3.65)],mw,RECUR,z=4)
        remain -= mw
        prev_x = x
        c.unit(x,3.19,name,mw,RECUR,width=2.15,fs=9.4)
    c.tag(19.25,5.46, f'Recirculation  {_mw(recirc)}',RECUR,bold=True,fs=10.2)
    # A named house load includes 4 MODELLED consumers, not extra loads.
    if house:
        xh = xs[-1]
        totalh = loads[-1][1]
        if totalh > 1e-8:
            c.ribbon([(xh,2.74),(xh,2.23)],totalh,RECUR,arrow=False,z=2)
            c.joint(xh,2.23,totalh,RECUR,z=4)
        hxs = [13.72,16.12,18.52,20.92]
        # thin distribution spine, each tap is one physical load
        # Split around the house node so EVERY house consumer is actually
        # connected.  Each section carries the sum of its downstream taps.
        hvalues = [amount for _,amount in house]
        c.ribbon([(xh,2.23),(hxs[1],2.23)],
                 hvalues[0]+hvalues[1],RECUR,arrow=False,z=1)
        c.ribbon([(hxs[1],2.23),(hxs[0],2.23)],
                 hvalues[0],RECUR,arrow=False,z=1)
        c.ribbon([(xh,2.23),(hxs[2],2.23)],
                 hvalues[2]+hvalues[3],RECUR,arrow=False,z=1)
        c.ribbon([(hxs[2],2.23),(hxs[3],2.23)],
                 hvalues[3],RECUR,arrow=False,z=1)
        for (name,amount),x in zip(house,hxs):
            if amount is not None and amount > 1e-8:
                c.ribbon([(x,2.23),(x,1.65)],amount,RECUR,z=2)
            c.unit(x,1.27,name,amount,RECUR,fs=8.4,width=2.12)
    if pulse:
        # H&CD is the uniquely represented explicit electricity-to-plasma
        # recirculation loop. Other loads feed separate plant auxiliaries.
        wallplug = dict(loads)['Heating / CD']
        cd = _get(run,'P_CD',True)
        _approx(wallplug, _get(run,'P_wallplug_calculated',True), max(recirc,1.),'H&CD supply')
        if cd > 0:
            c.ribbon([(5.2,3.58),(3.65,3.78),(2.72,4.35),(2.50,6.77),
                      (3.05,8.75),(5.35,10.0)],cd,PINK,arrow=True,z=5)
            c.tag(2.20,8.05,'Auxiliary heat / P_CD\n'+_mw(cd),PINK,fs=10,bold=True,ha='left')
        wp_loss = wallplug-cd
        if wp_loss > 1e-8:
            c.ribbon([(4.75,2.77),(3.9,1.92),(3.1,1.35)],wp_loss,LOSS,z=1)
            c.unit(3.58,.95,'H&CD LOSS',wp_loss,LOSS,width=2.02,fs=8.3)
    else:
        pass
    if grid_in > 0:
        c.tag(22.72,3.39,'NET  −'+_mw(grid_in),IMPORT,fs=10,bold=True)
    else:
        c.tag(22.6,10.53,'NET  +'+_mw(exported),ELEC,fs=10,bold=True)
    return grid_in,exported


def _pulse_physics(c, run):
    pf = _get(run,'P_fus',True)
    pn = _get(run,'P_neutron_balance',True)
    pa = _get(run,'P_alpha_balance',True)
    pcd = _get(run,'P_CD',True)
    mult = _get(run,'P_blanket_nuclear_gain',True)
    prad = _get(run,'P_rad_total_balance',True)
    sep = _get(run,'P_sep',True)
    ohm = _get(run,'P_Ohm')
    div = _get(run,'P_div_balance')
    pth = _get(run,'P_th',True)
    to_turb = _get(run,'P_th_reactor_to_turbine_pulse',True)
    to_store = _get(run,'P_th_reactor_to_storage_pulse',True)
    gross = _get(run,'P_e_gross_pulse',True)
    _approx(pn+pa,pf,max(1,pf),'fusion branch split')
    _approx(pa+pcd,prad+sep,max(1,pf),'plasma radiation / separatrix split')
    _approx(pn+mult+prad+sep,pth,max(1,pth),'reactor thermal closure')
    _approx(to_turb+to_store,max(0.,pth),max(1,pth),'thermal storage / turbine split')
    if min(pf,pn,pa,pcd,prad,sep,to_turb,to_store,gross) < -1e-7:
        raise ValueError('A physical power stream is negative; cannot represent as positive Sankey ribbon')
    c.ribbon([(1.15,11.63),(3.5,11.63)],pf,BLUE,arrow=True)
    c.joint(3.5,11.63,pf,BLUE,z=6)
    c.ribbon([(3.5,11.63),(4.55,12.22),(5.95,12.37),(8.7,12.37)],pn,NEUT)
    c.ribbon([(3.5,11.63),(4.02,11.18),(4.7,10.62),(5.45,10.15)],pa,ALPHA)
    c.unit(1.21,11.63,'FUSION',pf,BLUE,filled=True,width=1.69)
    c.unit(7.15,12.37,'NEUTRONS',pn,NEUT,width=1.95)
    c.tag(4.05,10.53,'α / charged  '+_mw(pa),BLUE,fs=10,bold=True)
    # Multiplier is an *additional* nuclear energy source; do not count it
    # as a second independent neutron transport stream.
    if mult > 1e-8:
        c.ribbon([(8.1,13.4),(8.65,12.92),(9.1,12.43)],mult,NEUT,z=3)
        c.tag(7.9,13.25,'Blanket gain  +'+_mw(mult),NEUT,bold=True,fs=9)
        c.joint(9.1,12.43,pn+mult,HEAT,z=6)
    c.ribbon([(8.85,12.37),(10.15,12.30),(11.85,11.95),(14.1,11.12)],pn+mult,HEAT)
    c.unit(9.1,12.36,'BLANKET',pn+mult,HEAT,width=2.08)
    c.unit(5.8,10.06,'PLASMA',pa+pcd,BLUE,width=1.98)
    c.ribbon([(5.8,9.82),(6.85,9.35),(7.65,9.20),(8.35,9.17)],prad,PINK)
    c.ribbon([(5.8,9.79),(6.55,8.62),(7.82,7.84),(10.82,7.72)],sep,PURPLE)
    c.joint(5.8,9.98,pa+pcd,BLUE,z=6)
    rad_parts = [(name,_get(run,key)) for name,key in (
        ('Brems','P_Brem'),('Synch.','P_syn'),('Line','P_line'))]
    if all(val is not None for _,val in rad_parts) and all(v >= 0 for _,v in rad_parts):
        _approx(sum(v for _,v in rad_parts),prad,max(1,prad),'radiation channel split')
        ys = [9.88,9.16,8.44]
        for (name,p),y in zip(rad_parts,ys):
            c.ribbon([(8.35,9.17),(9.0,y),(10.25,y),(11.5,9.17)],p,PINK,z=2)
            c.tag(9.62,y+.24,f'{name} {_mw(p)}',PINK,fs=8.5)
        c.joint(8.35,9.17,prad,PINK,z=6)
        c.joint(11.5,9.17,prad,PINK,z=6)
    else:
        c.ribbon([(8.35,9.17),(11.5,9.17)],prad,PINK)
        c.joint(8.35,9.17,prad,PINK,z=6)
        c.joint(11.5,9.17,prad,PINK,z=6)
    c.unit(8.10,9.17,'RADIATION',prad,PINK,width=1.68,fs=9)
    c.unit(11.65,9.17,'FIRST WALL',prad,PINK,width=2.05)
    c.unit(10.82,7.72,'DIVERTOR',sep,PURPLE,width=1.99)
    if div is not None and ohm is not None:
        _approx(sep+ohm,div,max(1,pf),'divertor diagnostic')
    c.ribbon([(11.65,9.35),(12.45,9.78),(13.15,10.28),(14.1,11.04)],prad,PINK)
    c.ribbon([(10.88,8.06),(11.78,8.18),(12.78,8.92),(14.1,10.91)],sep,PURPLE)
    c.joint(14.1,11.0,pth,HEAT,z=6)
    c.unit(14.25,10.99,'REACTOR HEAT',pth,HEAT,width=2.26)
    c.ribbon([(14.4,10.99),(15.24,10.99),(17.21,11.65)],to_turb,PURPLE)
    c.ribbon([(14.4,10.93),(15.55,9.43),(16.62,7.74)],to_store,PURPLE)
    c.tag(15.57,11.83,'To turbine  '+_mw(to_turb),PURPLE,fs=9.0)
    if to_store > 1e-8:
        c.unit(16.72,7.52,'STORAGE CHARGE',to_store,PURPLE,width=2.68,fs=9.5)
    c.unit(17.40,11.65,'TURBINE',to_turb,PURPLE,width=1.88)
    if to_turb > gross + 1e-8:
        rejected = to_turb-gross
        c.ribbon([(17.64,11.35),(18.15,10.20),(18.4,8.94)],rejected,LOSS,z=1)
        c.unit(19.01,8.68,'Rejected heat',rejected,LOSS,width=2.19,fs=8.7)
    c.ribbon([(17.4,11.65),(20.5,11.65)],gross,ELEC)
    c.joint(20.5,11.65,gross,ELEC,z=6)
    c.unit(19.61,11.65,'ELECTRICITY',gross,ELEC,width=2.11)


def _dwell_physics(c, run):
    dis = _get(run,'P_th_storage_to_turbine_dwell',True)
    gross = _get(run,'P_e_gross_dwell',True)
    charge = _get(run,'P_th_reactor_to_storage_pulse',True)
    tp = _get(run,'t_pulse_s')
    df = _get(run,'Dwell_factor')
    if dis < -1e-8 or gross < -1e-8:
        raise ValueError('Negative D0FUS dwell heat or electricity')
    # Pulse/dwell heat rates cannot be subtracted directly. Derive energy
    # input and loss on a *dwell-equivalent* power basis from D0FUS data.
    dwell_equiv = None
    storage_loss = None
    if (df is not None and 0 < df < 1 and tp is not None and tp > 0):
        td = tp * (1-df)/df
        dwell_equiv = charge * tp/td
        storage_loss = dwell_equiv - dis
        if storage_loss < -1e-5*max(1,dis,dwell_equiv):
            raise ValueError('Computed D0FUS storage return exceeds energy charged')
        storage_loss = max(0.,storage_loss)
    if dwell_equiv is not None and dwell_equiv > 0:
        c.ribbon([(2.0,11.45),(5.8,11.45)],dwell_equiv,PURPLE)
        c.unit(2.4,11.45,'STORED HEAT',dwell_equiv,PURPLE,width=2.48,fs=9.1)
        if storage_loss > 0:
            c.ribbon([(5.7,11.43),(6.65,9.9),(8.08,9.5)],storage_loss,LOSS)
            c.unit(8.1,9.23,'Storage losses',storage_loss,LOSS,width=2.31,fs=9.1)
        c.ribbon([(5.8,11.45),(10.0,11.45),(13.2,11.45)],dis,PURPLE)
        c.unit(5.8,11.45,'STORAGE',dwell_equiv,PURPLE,width=1.78,fs=8.7)
    else:
        c.ribbon([(2.25,11.45),(13.2,11.45)],dis,PURPLE)
        c.unit(3.5,11.45,'STORAGE',dis,PURPLE,width=2.05)
    c.unit(13.25,11.45,'TURBINE',dis,PURPLE,width=2.15)
    c.joint(13.25,11.45,dis,PURPLE,z=6)
    if dis>gross+1e-8:
        rejected = dis-gross
        c.ribbon([(13.4,11.1),(14.45,9.7),(16.1,9.10)],rejected,LOSS)
        c.unit(16.1,8.85,'Rejected heat',rejected,LOSS,width=2.15,fs=9)
    c.ribbon([(13.3,11.45),(20.55,11.55)],gross,ELEC)
    c.joint(20.55,11.55,gross,ELEC,z=6)
    c.unit(19.3,11.55,'ELECTRICITY',gross,ELEC,width=2.19)


def plot_plant_power_balance(run, phase='pulse', save_dir=None):
    """Draw and save a connected, width-scaled D0FUS pulse/dwell flow map.

    Saves `plant_power_balance_<phase>.png` in the existing plot_run folder.
    Does not change any calculation, balance or phase-dependent allocation.
    """
    if phase not in ('pulse','dwell'):
        raise ValueError('phase must be pulse or dwell')
    gross, net, loads, recirc, house = _load_data(run,phase)
    scale=max(1., gross, recirc, _get(run,'P_fus') or 0.,
              _get(run,'P_th') or 0.)
    fig, ax, c = _base(phase,scale)
    try:
        if phase=='pulse':
            _pulse_physics(c,run)
        elif (_get(run,'Dwell_factor') is not None
              and _get(run,'Dwell_factor') >= 1.0):
            # The D0FUS model assigns P_e_net_dwell=P_e_net_pulse when
            # Dwell_factor=1 (as a convenience), but NO dwell actually exists.
            c.unit(11.6,10.5,'NO DWELL INTERVAL',None,PURPLE,width=6.,fs=16)
            
            # Still emit a valid, explicitly labelled figure.
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,f'plant_power_balance_{phase}.png'),
                            dpi=170,bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            return
        else:
            _dwell_physics(c,run)
        _electric_network(c,run,phase,gross,net,loads,recirc,house)
        if save_dir is not None:
            os.makedirs(save_dir,exist_ok=True)
            fig.savefig(os.path.join(save_dir,f'plant_power_balance_{phase}.png'),dpi=170,bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    except Exception:
        plt.close(fig)
        raise
