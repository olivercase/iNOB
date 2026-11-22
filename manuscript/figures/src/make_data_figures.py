# Data figures of the iNOB paper at Nature size, as vector PDF (TrueType, live text) + 300 dpi PNG + editable SVG.
#   fig_spine_source_strength, fig_spine_trials_to_detect, fig_spine_best_channel, fig_spine_electrode_coverage,
#   fig5_spine_source_models, fig6_spine_sensor_design.
# Data come from the saved leadfields and package functions (scripts/spine_paper_figures.py, spine_figures.py), and both
# noise floors from configs/spine.yaml: SQUID magnetoneurograph 280 fT (100-5,000 Hz), electrodes 2.19 uV (30-500 Hz).
# The master copy lives in the iNOB repository, manuscript/figures/src/; the paper folder holds an identical copy.
#   python make_data_figures.py [name ...]     (the python that can import inob; INOB_REPO = package checkout)
import os, sys, json, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
# house style: the manuscripts repo's house/figures when run from the paper folder, the vendored copy in the package
sys.path.insert(0, os.path.join(HERE, 'style'))
sys.path.insert(0, os.path.join(ROOT, 'house', 'figures'))
INOB = os.environ.get('INOB_REPO', os.path.expanduser('~/iNOB'))
sys.path.insert(0, os.path.join(INOB, 'scripts'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
from matplotlib.transforms import ScaledTranslation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nature_style as ns

MM = ns.MM
NAT = ns.NAT
BLUE, RED, TEAL, PURPLE = ns.BLUE, ns.RED, ns.TEAL, ns.PURPLE
GREY_BAND, GREY_LINE, GREY_TXT = NAT['grey'][0], NAT['grey'][1], NAT['grey'][4]
BUDGET = (500, 2000)
SHOWN = ("C1", "C4", "C7", "T4", "T8", "T12")
Q_SPINE = 5.11
CACHE = os.path.join(HERE, 'cache_array_info.json')


def plain(v, _=None):
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def new_fig(w_mm, h_mm):
    ns.apply()
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def ax_mm(fig, l, b, w, h):
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([l / W, b / H, w / W, h / H])


def letter(fig, ch, x_mm, y_mm_from_top):
    W, H = fig.get_size_inches() / MM
    fig.text(x_mm / W, 1 - y_mm_from_top / H, ch, fontsize=8, fontweight='bold', ha='left', va='top', color='#000000')


def logaxis(ax, axis='y'):
    f = FuncFormatter(plain)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(f)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_minor_formatter(FuncFormatter(lambda v, _: ''))


def levels_top(ax, levels, regions=True, zrange=None):
    """Vertebral level labels above the axes (grey guide lines inside) and lumbar/thoracic/cervical bands."""
    fig = ax.figure
    for z, name in levels:
        if name in SHOWN:
            ax.axvline(z, color=GREY_LINE, lw=0.5, zorder=0)
            ax.annotate(name, xy=(z, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(0, 1.5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=7, color='#000000')
    if regions:
        d = dict((n, z) for z, n in levels)
        z0, z1 = zrange
        tr = ax.get_xaxis_transform() + ScaledTranslation(0, 13 / 72, fig.dpi_scale_trans)
        for lo, hi, nm in ((z0, d['T12'], 'lumbar'), (d['T12'], d['C7'], 'thoracic'), (d['C7'], z1, 'cervical')):
            ax.annotate('', xy=(hi - 2, 1), xytext=(lo + 2, 1), xycoords=tr, textcoords=tr,
                        arrowprops=dict(arrowstyle='<->', color=GREY_TXT, lw=0.5, shrinkA=0, shrinkB=0,
                                        mutation_scale=5), annotation_clip=False)
            ax.annotate(nm, xy=((lo + hi) / 2, 1), xycoords=tr, xytext=(0, 1.5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, color='#000000', annotation_clip=False)


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def save(fig, stem):
    ns.save(fig, stem, FIG)
    print('saved', stem)


_state = {}


def load():
    if _state:
        return _state
    import spine_paper_figures as spf
    import spine_figures as sf
    from inob.cli._common import apply_source_target
    from inob.config import load_config
    cfg = apply_source_target(load_config(os.path.join(INOB, 'configs', 'spine.yaml')), 'spine')
    meg, eeg, wb, _opm_floors, idx = spf._load(cfg)
    from inob.analysis.snr import compute_noise_floors
    floors = compute_noise_floors(cfg)        # spine.yaml: SQUID 280 fT, electrodes 2.19 uV
    print('noise floors: magnetic %.1f fT, electric %.3f uV (%s)' % (floors.meg_per_channel_fT, floors.eeg_per_channel_uV,
                                                                     cfg.noise.band_label))
    from inob.sensors.layout import radial_channels
    from inob.sources.cap import longitudinal_leadfield
    z = meg.source_pos[:, 2]
    long_meg = np.abs(longitudinal_leadfield(meg.L_fT_per_nAm, meg.source_pos)[0])
    rad = radial_channels(list(meg.channel_names))
    _state.update(spf=spf, sf=sf, cfg=cfg, meg=meg, eeg=eeg, wb=wb, floors=floors, idx=idx, z=z, order=np.argsort(z),
                  levels=sf._levels(cfg, z), long_meg=long_meg, rad=rad)
    return _state


def curves():
    S = load()
    spf, sf = S['spf'], S['sf']
    sm, se = S['floors'].meg_per_channel_fT, S['floors'].eeg_per_channel_uV
    lm, rad = S['long_meg'], S['rad']
    return {
        'EXG, 32-contact paddle': (sf._trials(spf._bipolar(S['eeg']) * Q_SPINE, se), RED, '-'),
        'MXG, SQUID triaxial': (sf._trials(lm.max(0) * Q_SPINE, sm), BLUE, '-'),
        'MXG, SQUID radial only': (sf._trials(lm[rad].max(0) * Q_SPINE, sm), BLUE, '--'),
        'EXG, 1,000-contact whole body': (sf._trials(spf._bipolar(S['wb']) * Q_SPINE, se), RED, '-.'),
    }


# ------------------------------------------------------------------ source strength
def fig_source_strength():
    from inob.physiology.scenarios import dorsal_column_population
    f = dorsal_column_population()
    d, w = f.diameters_um, f.weights
    sig, dV = 1.0, 0.080
    Qpa = lambda x: np.pi * (x * 1e-6) ** 2 * sig * dV / 4 * 1e12          # pA.m
    mean_d = float((d * w).sum()); mean_Q = float((Qpa(d) * w).sum())
    print('mean d', round(mean_d, 2), 'mean Q', round(mean_Q, 2), 'x800 nA.m', round(mean_Q * 800 / 1e3, 3))
    fig = new_fig(120, 52)
    a = ax_mm(fig, 14, 12, 41, 34); b = ax_mm(fig, 74, 12, 42, 34)
    a.bar(d, w, width=0.62, color=NAT['grey'][3], edgecolor='white', lw=0.3)
    a.axvline(mean_d, color='#000000', lw=0.75, ls=':')
    a.text(mean_d + 0.4, w.max() * 1.02, 'mean 9.8 µm', fontsize=7, va='bottom', ha='left')
    a.set_ylim(0, w.max() * 1.22); a.set_xlim(2.5, 17); a.set_yticks([0, 0.05, 0.10])
    a.set_xlabel('fibre diameter (µm)'); a.set_ylabel('fraction of fibres')
    clean(a)
    x = np.linspace(d.min(), d.max(), 200)
    b.plot(x, Qpa(x), color='#000000', lw=1.25)
    xd = float(np.sqrt(4 * mean_Q * 1e-12 / (np.pi * sig * dV)) * 1e6)
    b.plot([xd], [mean_Q], 'o', mfc='white', mec='#000000', ms=4, mew=0.75, zorder=5)
    b.annotate('6.4 pA·m per fibre\n× 800 fibres = 5.11 nA·m', xy=(xd, mean_Q), xytext=(17, 2.6), textcoords='data',
               ha='right', va='center', fontsize=7,
               arrowprops=dict(arrowstyle='-', color=GREY_TXT, lw=0.5, shrinkA=0, shrinkB=2))
    b.set_xlim(2.5, 17); b.set_ylim(0, 16)
    b.set_xlabel('fibre diameter (µm)'); b.set_ylabel('moment per fibre (pA·m)')
    clean(b)
    letter(fig, 'a', 1, 1); letter(fig, 'b', 61, 1)
    save(fig, 'fig_spine_source_strength')


# ------------------------------------------------------------------ trials to detect
def _band(ax):
    ax.axhspan(*BUDGET, color=GREY_BAND, lw=0, zorder=0)


def fig_trials():
    S = load(); z, o, idx = S['z'], S['order'], S['idx']
    C = curves()
    fig = new_fig(120, 88)
    ax = ax_mm(fig, 17, 30, 98, 44)
    _band(ax)
    hs = []
    for nm, (n, col, ls) in C.items():
        h, = ax.plot(z[o], n[o], color=col, ls=ls, lw=1.0); hs.append((h, nm))
    ax.axvline(z[idx], color='#000000', lw=0.75, ls=':')
    ax.set_yscale('log'); ax.set_ylim(0.7, 1e5)
    ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5]); logaxis(ax)
    ax.set_xlabel('source height z (mm)'); ax.set_ylabel(f'trials to SNR 3 at {Q_SPINE} nA·m')
    levels_top(ax, S['levels'], True, (z.min(), z.max()))
    clean(ax)
    handles = [h for h, _ in hs] + [Patch(fc=GREY_BAND, ec='none'), Line2D([], [], color='#000000', lw=0.75, ls=':')]
    labels = [n for _, n in hs] + ['clinical SSEP budget (500–2,000 trials)', 'default source #72 (C6)']
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(17 / 120, 1 / 88), ncol=2, fontsize=7,
               frameon=False, handlelength=2.6, columnspacing=1.2, labelspacing=0.3, borderaxespad=0)
    save(fig, 'fig_spine_trials_to_detect')


# ------------------------------------------------------------------ best channel
def fig_best_channel():
    from inob.analysis.snr import per_source_peak
    S = load(); z, o, idx = S['z'], S['order'], S['idx']
    meg, rad = S['meg'], S['rad']
    tri = per_source_peak(meg.L_fT_per_nAm); ro = per_source_peak(meg.L_fT_per_nAm[rad])
    print('max channel field at #72', round(float(tri[idx]), 1))
    fig = new_fig(120, 70)
    ax = ax_mm(fig, 17, 12, 98, 40)
    ax.plot(z[o], tri[o], color=BLUE, lw=1.0, label='triaxial')
    ax.plot(z[o], ro[o], color=BLUE, lw=1.0, ls='--', label='radial only')
    ax.plot(z[idx], tri[idx], 'o', mfc='white', mec='#000000', ms=4, mew=0.75, zorder=5)
    ax.annotate('#72', xy=(z[idx], tri[idx]), xytext=(4, -12), textcoords='offset points', fontsize=7, ha='left')
    ax.set_ylim(0, 70)
    ax.set_xlabel('source height z (mm)'); ax.set_ylabel('MXG maximum channel field\n(fT per nA·m)')
    ax.legend(loc='lower right', fontsize=7, frameon=False, handlelength=2.6)
    levels_top(ax, S['levels'], True, (z.min(), z.max()))
    clean(ax)
    save(fig, 'fig_spine_best_channel')


# ------------------------------------------------------------------ electrode coverage
def fig_coverage():
    S = load(); z, o = S['z'], S['order']; spf = S['spf']
    pad, whole = spf._bipolar(S['eeg']), spf._bipolar(S['wb'])
    pz = S['eeg'].coil_pos[:, 2]
    fig = new_fig(120, 70)
    ax = ax_mm(fig, 19, 12, 96, 40)
    ax.axvspan(pz.min(), pz.max(), color=GREY_BAND, lw=0, zorder=0)
    ax.plot(z[o], whole[o], color=RED, ls='-.', lw=1.0)
    ax.plot(z[o], pad[o], color=RED, lw=1.0)
    ax.set_yscale('log')
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1]); logaxis(ax)
    ax.text(z.min() + 6, 0.6, 'EXG, 1,000-contact whole body (dash-dotted)', fontsize=7, va='center')
    ax.text(z.min() + 6, 1.1e-2, 'EXG, 32-contact paddle (solid)', fontsize=7, va='center')
    ax.set_ylim(top=3.0)
    ax.text((pz.min() + pz.max()) / 2 + 12, 1.6, 'paddle extent', fontsize=7, ha='center', va='center', bbox=dict(fc=GREY_BAND, ec='none', pad=0.6))
    ax.set_xlabel('source height z (mm)'); ax.set_ylabel('EXG maximum bipolar difference\n(µV per nA·m)')
    levels_top(ax, S['levels'], True, (z.min(), z.max()))
    clean(ax)
    save(fig, 'fig_spine_electrode_coverage')


# ------------------------------------------------------------------ source models (fig 5)
def source_models_json():
    p = os.path.join(tempfile.mkdtemp(), 'sm.json')
    subprocess.run([sys.executable, '-m', 'inob.cli.main', 'source-models', '--config', 'configs/spine.yaml',
                    '--source-target', 'spine', '--json-out', p, '--no-figure'], check=True, cwd=INOB,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(p))


def fig_source_models():
    r = source_models_json()
    M = [r['models'][k] for k in ('stationary', 'ascending')]
    S = load(); sf = S['sf']
    snr_m = [m['meg_fT'] / S['floors'].meg_per_channel_fT for m in M]; snr_e = [m['eeg_single_trial_SNR'] for m in M]
    tr_m = [float(sf._trials(m['meg_fT'], S['floors'].meg_per_channel_fT)[0]) for m in M]; tr_e = [m['eeg_trials_for_SNR3'] for m in M]
    print('amps', [m['meg_fT'] for m in M], [m['eeg_uV'] for m in M])
    print('trials MXG', [round(t, 1) for t in tr_m], 'EXG', [round(t, 1) for t in tr_e], 'SNR', snr_m, snr_e)
    fig = new_fig(120, 104)
    a = ax_mm(fig, 14, 62, 40, 34); b = ax_mm(fig, 71, 62, 45, 34); c = ax_mm(fig, 14, 8, 102, 36)
    x = np.arange(2); w = 0.38
    labs = ['stationary\n(one segment)', 'ascending\n(travelling volley)']
    hat = dict(hatch='////', edgecolor='white', lw=0)
    # a
    a.bar(x - w / 2, snr_m, w, color=BLUE, lw=0)
    a.bar(x + w / 2, snr_e, w, color=RED, **hat)
    a.axhline(3, color='#000000', lw=0.75, ls='--')
    a.text(1.5, 3.4, 'SNR = 3 (Rose criterion)', ha='right', va='bottom', fontsize=7)
    a.set_yscale('log'); a.set_ylim(0.03, 10); a.set_yticks([0.1, 1, 10]); logaxis(a)
    a.set_ylabel('single-trial SNR')
    a.set_xticks(x, labs); a.set_xlim(-0.55, 1.5)
    a.legend([Patch(fc=BLUE), Patch(fc=RED, hatch='////', ec='white', lw=0)], ['MXG (SQUID)', 'EXG (paddle)'],
             loc='center left', bbox_to_anchor=(0.5, 0.68), fontsize=7, frameon=False, handlelength=1.2)
    clean(a)
    # b
    b.axhspan(*BUDGET, color=GREY_BAND, lw=0, zorder=0)
    b.bar(x - w / 2, tr_m, w, color=BLUE, lw=0, zorder=3)
    b.bar(x + w / 2, tr_e, w, color=RED, zorder=3, **hat)
    b.text(-0.5, 2300, 'clinical budget\n500–2,000', fontsize=7, va='bottom', ha='left')
    b.set_yscale('log'); b.set_ylim(10, 1e4); b.set_yticks([10, 100, 1e3, 1e4]); logaxis(b)
    b.set_ylabel('trials to reach SNR 3')
    b.set_xticks(x, labs); b.set_xlim(-0.55, 1.5)
    clean(b)
    # c
    pm = M[1]['meg_fT'] / M[0]['meg_fT']; pe = M[1]['eeg_uV'] / M[0]['eeg_uV']
    ratios = [snr_m[0] / snr_e[0], snr_m[1] / snr_e[1], pm, pe]
    cols = [PURPLE, PURPLE, BLUE, RED]
    for i, (v, col) in enumerate(zip(ratios, cols)):
        c.bar(i, v, 0.6, color=col, lw=0, **({'hatch': '////', 'edgecolor': 'white'} if i == 3 else {}))
        c.text(i, v * 1.08, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    c.axhline(1, color='#000000', lw=0.75, ls=':')
    c.set_yscale('log'); c.set_ylim(0.2, 8); c.set_yticks([0.2, 0.5, 1, 2, 4, 8]); logaxis(c)
    c.set_ylabel('ratio (×)')
    c.set_xticks(range(4), ['MXG/EXG\nstationary', 'MXG/EXG\nascending', 'propagation\npenalty (MXG)',
                            'propagation\npenalty (EXG)'])
    clean(c)
    letter(fig, 'a', 1, 1); letter(fig, 'b', 58, 1); letter(fig, 'c', 1, 52)
    print('ratios', ratios)
    save(fig, 'fig5_spine_source_models')


# ------------------------------------------------------------------ sensor design (fig 6)
def array_info():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    S = load(); sf = S['sf']
    from inob.analysis.placement import nearest_array_info, optimise_array
    meg = S['meg']; sig = S['floors'].meg_per_channel_fT; L = meg.L_fT_per_nAm
    opt = optimise_array(L, meg.coil_pos, n_sensors=max(sf.ARRAY_SIZES), noise_floor=sig, moment_nAm=Q_SPINE)
    near = [nearest_array_info(L, meg.coil_pos, meg.source_pos, n_sensors=n, noise_floor=sig, moment_nAm=Q_SPINE)
            for n in sf.ARRAY_SIZES]
    d = {'sizes': list(sf.ARRAY_SIZES), 'optimised': [float(opt.info_bits[n - 1]) for n in sf.ARRAY_SIZES],
         'nearest': [float(v) for v in near]}
    json.dump(d, open(CACHE, 'w'), indent=1)
    return d


STANDOFF_JSON = os.environ.get('STANDOFF_JSON', os.path.join(INOB, 'outputs', 'paper', 'standoff.json'))


def standoff():
    """Stand-off (mm) and maximum-amplitude-channel field at source #72 (fT per nA.m).

    Read from `scripts/sensor_sweep.py --json outputs/paper/standoff.json` when that file exists (the sweep needs
    DUNEuro); otherwise the values of the paper's sweep, 2026-10-22, are used and the fact is printed."""
    if os.path.exists(STANDOFF_JSON):
        d = json.load(open(STANDOFF_JSON))
        rows = sorted(d['rows'] if 'rows' in d else d, key=lambda r: r['depth_mm'])
        return [r['depth_mm'] for r in rows], [r['peak_fT_per_nAm'] for r in rows]
    print('stand-off: no', STANDOFF_JSON, '- using the values of the 2026-10-22 sweep')
    return [6.5, 10, 20, 30], [43.7, 40.7, 31.4, 24.6]


def fig_sensor_design():
    S = load(); z, o, idx = S['z'], S['order'], S['idx']; sf = S['sf']
    C = curves()
    H = 80
    fig = new_fig(183, H)
    bot, ah = 31, 41
    a = ax_mm(fig, 15, bot, 54, ah); b = ax_mm(fig, 83, bot, 50, ah); c = ax_mm(fig, 147, bot, 34, ah)
    _band(a)
    hs = []
    for nm, (n, col, ls) in C.items():
        h, = a.plot(z[o], n[o], color=col, ls=ls, lw=1.0); hs.append((h, nm))
    a.axvline(z[idx], color='#000000', lw=0.75, ls=':')
    a.set_yscale('log'); a.set_ylim(0.7, 1e5); a.set_yticks([1, 10, 100, 1e3, 1e4, 1e5]); logaxis(a)
    a.set_xlabel('source height z (mm)'); a.set_ylabel(f'trials to SNR 3 at {Q_SPINE} nA·m')
    levels_top(a, S['levels'], False); clean(a)
    handles = [h for h, _ in hs] + [Patch(fc=GREY_BAND, ec='none'), Line2D([], [], color='#000000', lw=0.75, ls=':')]
    labels = [n for _, n in hs] + ['clinical SSEP budget (500–2,000)', 'default source #72 (C6)']
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(2 / 183, 0), ncol=1, fontsize=7, frameon=False,
               handlelength=2.6, labelspacing=0.15, borderaxespad=0)
    # b: stand-off of the SQUID array (maximum-amplitude channel at source #72, fT per nA.m from scripts/sensor_sweep.py)
    so, amp = standoff()
    n = [int(sf._trials(np.array([a_ * Q_SPINE]), S['floors'].meg_per_channel_fT)[0]) for a_ in amp]
    print('stand-off trials', n)
    bars = b.bar(range(4), n, 0.7, color=BLUE, lw=0)
    for bb, v in zip(bars, n):
        b.text(bb.get_x() + bb.get_width() / 2, v + 0.4, str(v), ha='center', va='bottom', fontsize=7)
    b.set_xticks(range(4), [f'{v:g}' for v in so]); b.set_ylim(0, max(n) * 1.25)
    b.set_ylabel('trials to SNR 3, source #72')
    b.set_xlabel('stand-off from skin (mm)')
    clean(b)
    # c
    d = array_info()
    c.plot(d['sizes'], d['optimised'], 'o-', color=RED, ms=3, lw=1.0, label='information-optimised')
    c.plot(d['sizes'], d['nearest'], '^--', color=BLUE, ms=3, lw=1.0, label='nearest the cord')
    c.set_xscale('log', base=2); c.set_xticks(d['sizes'], [str(v) for v in d['sizes']]); c.xaxis.set_minor_locator(NullLocator())
    c.set_ylim(10, 65); c.set_yticks([20, 40, 60])
    c.set_xlabel('triaxial SQUIDs in the array'); c.set_ylabel('cord information (bits)')
    c.legend(loc='upper left', fontsize=7, frameon=False, handlelength=2.0, borderaxespad=0.1)
    clean(c)
    letter(fig, 'a', 1, 1); letter(fig, 'b', 69, 1); letter(fig, 'c', 133, 1)
    save(fig, 'fig6_spine_sensor_design')

def fig_validation():
    """Predicted vs measured SQUID magnetospinography. Values are the output of ~/iNOB/scripts/validate_miyano2020.py
    (Miyano 2020 for a; Hashimoto 2021 Table 1 for b); hardcoded here, not recomputed."""
    fig = new_fig(183, 70)
    a = ax_mm(fig, 14, 14, 70, 50)
    sites = [('Elbow', 3.81, 64.0, 12.0), ('Wrist', 1.70, 30.0, 7.8)]
    free, fem = 20.7, 40.6   # fT per nA.m, left-right moment at T1-C7, cranio-caudal channels
    lim = 190
    a.plot([0, lim], [0, lim], color=GREY_LINE, lw=0.5, zorder=1)
    a.text(96, 94, '1:1', fontsize=7, color='#000000')
    for lab, q, m, sd in sites:
        a.errorbar(q * free, m, yerr=sd, fmt='o', ms=4, mfc=BLUE, mec=BLUE, ecolor=BLUE, elinewidth=0.8, capsize=2, zorder=3)
        a.errorbar(q * fem, m, yerr=sd, fmt='s', ms=4, mfc='white', mec=RED, ecolor=RED, elinewidth=0.8, capsize=2, zorder=3)
        a.text(q * free - 4, m + sd + 5, lab, fontsize=7, ha='center', va='bottom')
    a.set_xlim(0, lim); a.set_ylim(0, 100)
    a.set_xlabel('predicted peak field (fT)'); a.set_ylabel('measured peak field (fT)')
    a.legend(handles=[Line2D([], [], marker='o', ls='', mfc=BLUE, mec=BLUE, ms=4, label='free space'),
                      Line2D([], [], marker='s', ls='', mfc='white', mec=RED, ms=4, label='full FEM')],
             loc='upper left', frameon=False, fontsize=7, borderaxespad=0.1, handletextpad=0.2)
    clean(a)
    b = ax_mm(fig, 108, 14, 70, 50)
    levels = ['T10', 'T6', 'T2']
    bil = [0.71, 0.47, 0.45]; uni = [0.37, 0.21, 0.22]
    y = np.arange(3)[::-1]
    b.axvline(Q_SPINE, color=GREY_TXT, lw=0.8)
    b.text(Q_SPINE * 1.08, 2.45, 'default 5.11', fontsize=7, va='center')
    b.plot(bil, y, 'o', ms=4, mfc=BLUE, mec=BLUE, ls='')
    b.plot(uni, y, 's', ms=4, mfc='white', mec=BLUE, ls='')
    b.set_xscale('log'); b.set_xlim(0.1, 12); b.set_ylim(-0.6, 2.8)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    b.set_yticks(y); b.set_yticklabels(levels)
    b.set_xlabel('moment reproducing the measured field (nA\u00b7m)')
    b.legend(handles=[Line2D([], [], marker='o', ls='', mfc=BLUE, mec=BLUE, ms=4, label='bilateral'),
                      Line2D([], [], marker='s', ls='', mfc='white', mec=BLUE, ms=4, label='unilateral')],
             loc='lower right', frameon=False, fontsize=7, borderaxespad=0.1, handletextpad=0.2)
    clean(b)
    letter(fig, 'a', 1, 1); letter(fig, 'b', 95, 1)
    save(fig, 'fig_spine_validation')


ALL = {'source_strength': fig_source_strength, 'trials': fig_trials, 'best_channel': fig_best_channel,
       'coverage': fig_coverage, 'source_models': fig_source_models, 'sensor_design': fig_sensor_design,
       'validation': fig_validation}
if __name__ == '__main__':
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
