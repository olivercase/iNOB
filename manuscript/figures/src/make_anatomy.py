# Figure 2 of the iNOB paper: exploded anatomy. The BodyParts3D meshes are re-rendered with the package's own layout
# (scripts/exploded_anatomy.py: load_groups, pack_columns, stack_labels, STYLE, lighting) but WITHOUT any text, leader line or
# scale bar baked into the raster. Labels, leaders and the 500 mm bar are added here as live vector text/lines at 7 pt on a
# 183 mm wide canvas. The image is embedded at its native render resolution (>= 300 dpi at 183 mm).
#   python3 make_anatomy.py        (python that can import pyvista, trimesh, matplotlib; INOB_REPO = package checkout)
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, 'style'))   # vendored copy, used in the iNOB repository
sys.path.insert(0, os.path.join(ROOT, 'house', 'figures'))
INOB = os.environ.get('INOB_REPO', os.path.expanduser('~/iNOB'))
sys.path.insert(0, os.path.join(INOB, 'scripts'))
from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import nature_style as ns
import exploded_anatomy as ea

H_PX = 3000          # render height; width follows from the scene aspect (>= 300 dpi at 183 mm)
RENDER = os.environ.get('FIG2_RENDER', os.path.join(HERE, '_fig2_render.png'))


def render_plain(groups, out, height):
    """exploded_anatomy.render without labels, leaders or scale bar; returns the layout needed to draw them."""
    mm_per_pt = (groups['skin'].bounds[5] - groups['skin'].bounds[4]) / height
    font_px = ea.label_font_px(2600)                      # layout assumes the paper's reference label size
    mm_per_pt = (groups['skin'].bounds[5] - groups['skin'].bounds[4]) / 2600
    headroom = ea.LABEL_RISE_MM + font_px * ea._LINE_HEIGHT * mm_per_pt
    dx = ea.pack_columns(groups, label_headroom_mm=headroom)
    named = {g: m for g, m in groups.items() if g != 'skin'}
    label_z = ea.stack_labels({g: (float(m.center[0]) + dx[g], float(m.bounds[5]) + ea.LABEL_RISE_MM) for g, m in named.items()},
                              {g: ea.LABEL[g] for g in named}, mm_per_pt=mm_per_pt, font_px=font_px)
    xs, zs = [], []
    for g, m in groups.items():
        b = m.bounds; xs += [b[0] + dx[g], b[1] + dx[g]]; zs += [b[4], label_z.get(g, b[5])]
    bar_z = min(zs) - 120.0; zs.append(bar_z - 160.0)
    x_lo, x_hi = min(xs) - ea.MARGIN_MM, max(xs) + ea.MARGIN_MM
    z_lo, z_hi = min(zs) - ea.MARGIN_MM, max(zs) + ea.MARGIN_MM
    aspect = (x_hi - x_lo) / (z_hi - z_lo)
    pl = pv.Plotter(off_screen=True, window_size=(round(height * aspect), height))
    pl.set_background('white')
    leaders, labels = [], []
    for g, mesh in groups.items():
        colour, opacity, rough = ea.STYLE[g]; ghost = g == 'skin'
        pl.add_mesh(mesh, color=colour, opacity=opacity if ghost else ea.GHOST_OPACITY, smooth_shading=True,
                    specular=0.45 if ghost else 0.10, specular_power=30 if ghost else 10, diffuse=0.85, ambient=0.22,
                    show_scalar_bar=False)
        if ghost:
            continue
        moved = mesh.copy(); moved.translate((dx[g], 0.0, 0.0), inplace=True)
        pl.add_mesh(moved, color=colour, opacity=opacity, smooth_shading=True, specular=0.55 * (1 - rough),
                    specular_power=12 + 60 * (1 - rough), diffuse=0.90, ambient=0.20, show_scalar_bar=False)
        c = np.asarray(mesh.center, float)
        leaders.append(((c[0], c[2]), (c[0] + dx[g], c[2])))
        labels.append((c[0] + dx[g], label_z[g], ea.LABEL[g]))
    pl.enable_parallel_projection()
    centre = ((x_lo + x_hi) / 2, (z_lo + z_hi) / 2)
    pl.camera_position = [(centre[0], -8000.0, centre[1]), (centre[0], 0.0, centre[1]), (0.0, 0.0, 1.0)]
    pl.camera.parallel_scale = (z_hi - z_lo) / 2
    ea.light_the_scene(pl)
    pl.enable_anti_aliasing('ssaa')
    pl.screenshot(str(out)); pl.close()
    return dict(extent=(x_lo, x_hi, z_lo, z_hi), leaders=leaders, labels=labels, bar=(x_lo + ea.MARGIN_MM, bar_z),
                px=(round(height * aspect), height))


def main():
    groups = ea.load_groups(Path(INOB) / 'data' / 'bodyparts3d' / 'subset')
    info = render_plain(groups, RENDER, H_PX)
    x_lo, x_hi, z_lo, z_hi = info['extent']
    aspect = (x_hi - x_lo) / (z_hi - z_lo)
    W_mm = 183.0; Hh = W_mm / aspect
    print('canvas mm', W_mm, round(Hh, 1), 'render px', info['px'], 'dpi at 183 mm', round(info['px'][0] / (W_mm / 25.4)))
    ns.apply()
    fig = plt.figure(figsize=(W_mm * ns.MM, Hh * ns.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    img = plt.imread(RENDER)
    ax.imshow(img, extent=(x_lo, x_hi, z_lo, z_hi), aspect='auto', interpolation='lanczos', zorder=0)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(z_lo, z_hi)
    LC = ns.NAT['grey'][3]
    for (a, b) in info['leaders']:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LC, lw=0.5, solid_capstyle='butt', zorder=2)
    for x, z, t in info['labels']:
        ax.text(x, z, t, ha='center', va='center', fontsize=7, color='#000000', zorder=3)
    x0, z0 = info['bar']
    ax.plot([x0, x0 + 500], [z0, z0], color='#000000', lw=1.0, solid_capstyle='butt', zorder=3)
    ax.text(x0 + 250, z0 - 70, '500 mm', ha='center', va='top', fontsize=7, color='#000000', zorder=3)
    ns.save(fig, 'fig2_anatomy', FIG)


if __name__ == '__main__':
    main()
