# Figure 3 of the iNOB paper: field of one cord dipole on the skin (MXG beside EXG).
# The package's own `inob torso` pipeline is run unchanged (scripts of the package are not edited); its figure object is
# captured instead of saved. Every piece of text, the legend, the title and the colour bars are then removed from the captured
# figure, the remaining artwork (skin mesh coloured by the field, sensor circles, cord dots, dipole arrow, scale-bar line,
# body-axis arrows) is rasterised at 450 dpi of the original 10.6 in canvas (about 660 dpi at the 183 mm print width), and the
# text, panel letters and colour bars are redrawn as live vector Helvetica 7 pt using the positions and the true colour limits
# read from the captured figure.
#   python3 make_torso.py     (python that can import inob; INOB_REPO = package checkout)
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, 'style'))   # vendored copy, used in the iNOB repository
sys.path.insert(0, os.path.join(ROOT, 'house', 'figures'))
INOB = os.environ.get('INOB_REPO', os.path.expanduser('~/iNOB'))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.text as mt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import nature_style as ns

os.chdir(INOB)
import inob.viz.torso_topoplot as tt
cap = {}
def _capture(fig, out, dpi=300, **k):
    cap['fig'] = fig
    return Path(out)
tt.save_figure = _capture
import inob.cli.torso as ct
ct.main(['--source-target', 'spine', '--esg-npz', 'outputs/forward/duneuro_eeg_leadfield_wholebody_spine.npz',
         '--out', os.path.join(os.environ.get('TMPDIR', '/tmp'), 'torso_unused.png')])
fig = cap['fig']
fig.canvas.draw()
inv = fig.transFigure.inverted()

# ---- read what has to be redrawn
axes3d = [a for a in fig.axes if a.name == '3d']
cbs = [a for a in fig.axes if a.get_label() == '<colorbar>']
info = {'cb': [], 'text': []}
for cax in cbs:
    mappable = cax.collections[0] if cax.collections else cax.images[0]
    lo, hi = cax.get_xlim()
    ticks = [t for t in cax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9]
    labs = [l.get_text() for l, t in zip(cax.get_xticklabels(), cax.get_xticks()) if lo - 1e-9 <= t <= hi + 1e-9]
    info['cb'].append(dict(pos=cax.get_position().bounds, lim=(lo, hi), cmap=tt.divergent_cmap(), ticks=ticks, labs=labs,
                           label=cax.get_xlabel()))
print('colour limits', [c['lim'] for c in info['cb']], [c['label'] for c in info['cb']])
print('tick labels', [c['labs'] for c in info['cb']])
for t in fig.findobj(mt.Text):
    s = t.get_text().strip()
    if not s:
        continue
    if t.axes is not None and t.axes.get_label() == '<colorbar>':
        continue
    ax_, tr = t.axes, t.get_transform()
    cx, cy = inv.transform(tr.transform(t.get_position())) if ax_ is not None else t.get_position()
    info['text'].append((s, cx, cy, t.get_ha(), t.get_va()))
    print(repr(s), round(cx, 3), round(cy, 3), t.get_ha(), t.get_va())

# ---- strip all text, legends, colour bars
for t in fig.findobj(mt.Text):
    if isinstance(t, mt.Annotation):
        t.set_text('')          # keep the arrow, drop the words
    else:
        t.set_visible(False)
for lg in fig.legends:
    lg.set_visible(False)
for cax in cbs:
    cax.set_visible(False)
fig.suptitle('')
buf_dpi = 450
fig.canvas.draw()
img = np.asarray(fig.canvas.buffer_rgba()) if False else None
from io import BytesIO
b = BytesIO(); fig.savefig(b, dpi=buf_dpi, format='png', facecolor='white'); b.seek(0)
img = plt.imread(b)
plt.close(fig)
H, W = img.shape[:2]
YTOP, YBOT = 0.835, 0.0                       # crop the removed title block off the top, empty margin off the bottom
img = img[int(round(H * (1 - YTOP))):int(round(H * (1 - YBOT))), :, :]
print('raster', img.shape, 'dpi at 183 mm', round(W / (183 / 25.4)))

# ---- compose at 183 mm
Wmm = 183.0; Hmm = Wmm * (7.4 * (YTOP - YBOT)) / 10.6
ns.apply()
out = plt.figure(figsize=(Wmm * ns.MM, Hmm * ns.MM))
bg = out.add_axes([0, 0, 1, 1]); bg.set_axis_off()
bg.imshow(img, extent=(0, 1, 0, 1), aspect='auto', interpolation='lanczos')
bg.set_xlim(0, 1); bg.set_ylim(0, 1)
SCALE_BAR_X, SCALE_BAR_TEXT_Y = (0.1357, 0.579), 0.292
fy = lambda y: (y - YBOT) / (YTOP - YBOT)
for (s, cx, cy, ha, va) in info['text']:
    if s in ('a', 'b'):
        out.text(cx, fy(cy), s, fontsize=8, fontweight='bold', ha='center', va='center', color='#000000')
    elif s.startswith('MXG'):
        out.text(cx, fy(cy), 'MXG, magnetic field on the skin', fontsize=7, ha='center', va='center')
    elif s.startswith('EXG'):
        out.text(cx, fy(cy), 'EXG, electric potential on the skin', fontsize=7, ha='center', va='center')
    elif s == '50 mm':
        pass                                           # placed below, from the scale-bar position in the raster
    elif s[:2] in ('x ', 'y ', 'z '):
        out.text(0.017 if s[0] == 'x' else cx, fy(cy), s[0], fontsize=7, ha='right' if s[0] == 'x' else ha, va=va)
for c in info['cb']:
    x, y, w, h = c['pos']
    cax = out.add_axes([x, fy(y), w, h / (YTOP - YBOT)])
    cb = out.colorbar(ScalarMappable(Normalize(*c['lim']), c['cmap']), cax=cax, orientation='horizontal')
    if len(c['ticks']) > 6:
        tk = [t for t in c['ticks'] if abs(round(t / 0.1) * 0.1 - t) < 1e-6]
        cb.set_ticks(tk); cb.set_ticklabels([f'{t:g}'.replace('-', '\u2212') for t in tk])
    else:
        cb.set_ticks(c['ticks']); cb.set_ticklabels(c['labs'])
    cb.outline.set_linewidth(0.5); cax.tick_params(width=0.5, length=2.5, labelsize=7)
    cb.set_label(c['label'].replace('$\\mu$', '\u00b5'), fontsize=7)
for xb in SCALE_BAR_X:
    out.text(xb, SCALE_BAR_TEXT_Y, '50 mm', fontsize=7, ha='center', va='center')
ns.save(out, 'fig3_spinal_cord', FIG)
