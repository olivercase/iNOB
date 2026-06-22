"""4-panel FEM visualisation (PyVista).

Three orthogonal slices through the vagus centroid + a 3-D view with the
skin rendered as a wireframe cage. Output written to ``cfg.outputs.fem_png``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from vagus_fm.config import Config
from vagus_fm.io.hdf5 import load_fem

logger = logging.getLogger(__name__)

TISSUE_RGBA = {
    "vagus_left":  (0.20, 0.80, 0.40, 1.00),
    "vagus_right": (0.10, 0.55, 0.95, 1.00),
    "skin":        (0.95, 0.78, 0.65, 0.30),
    "bone":        (0.82, 0.71, 0.55, 0.85),
    "muscle":      (0.78, 0.39, 0.31, 0.85),
}
EXTRA = [(1.00, 0.65, 0.00, 1.00), (0.78, 1.00, 0.39, 1.00)]


def _build_grid(pos: np.ndarray, tet: np.ndarray, tissue: np.ndarray) -> pv.UnstructuredGrid:
    n = len(tet)
    cells = np.empty((n, 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tet
    grid = pv.UnstructuredGrid(
        cells.ravel(), np.full(n, pv.CellType.TETRA, dtype=np.uint8), pos
    )
    grid["tissue"] = tissue.astype(np.float32)
    return grid


def _build_lut(uid: np.ndarray, id_to_label: dict[int, str]) -> tuple[pv.LookupTable, int]:
    n = int(uid.max()) + 1
    lut = pv.LookupTable(n_values=n)
    lut.SetNumberOfTableValues(n)
    for i in range(n):
        lut.SetTableValue(i, 0.3, 0.3, 0.3, 0.0)
    for k, tid in enumerate(uid):
        lbl = id_to_label.get(int(tid), f"t{tid}")
        rgba = TISSUE_RGBA.get(lbl, EXTRA[k % len(EXTRA)])
        lut.SetTableValue(int(tid), *rgba)
    lut.SetRange(0, n - 1)
    return lut, n


def render_fem(cfg: Config, *, interactive: bool = False) -> Path | None:
    """Render the FEM. Off-screen by default → ``cfg.outputs.fem_png``.

    With ``interactive=True``, opens a PyVista window and returns ``None``.
    """
    fem_path = cfg.outputs.fem_mat
    if not fem_path.exists():
        raise FileNotFoundError(f"{fem_path} not found. Run build_fem first.")

    fem = load_fem(fem_path)
    pos = fem.nodes
    tet = fem.tets.astype(np.int64)
    tissue = fem.tissue
    uid = np.unique(tissue)
    id_to_label = {i + 1: lab for i, lab in enumerate(fem.tissue_labels)}
    logger.info("FEM: %d nodes, %d tets, tissues: %s",
                len(pos), len(tet),
                {id_to_label.get(int(t), "?"): int((tissue == t).sum()) for t in uid})

    grid = _build_grid(pos, tet, tissue)
    lut, n_lut = _build_lut(uid, id_to_label)
    bounds = grid.bounds
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])

    inv = {lbl: tid for tid, lbl in id_to_label.items()}
    vagus_ids = [inv[k] for k in ("vagus_left", "vagus_right") if k in inv]
    if vagus_ids:
        vmask = np.isin(tissue, vagus_ids)
        node_idx = np.unique(tet[np.where(vmask)[0]])
        vc = pos[node_idx].mean(axis=0)
        x_slice, y_slice, z_slice = float(vc[0]), float(vc[1]), float(vc[2])
    else:
        x_slice, y_slice, z_slice = cx, cy, cz

    MESH_KWARGS = dict(
        scalars="tissue", cmap=lut, clim=[0, n_lut - 1], show_scalar_bar=False
    )

    def tissue_surface(tid: int):
        return grid.extract_cells(tissue == tid).extract_surface()

    if interactive:
        pl = pv.Plotter(window_size=(1600, 1000))
        pl.set_background("white")
        for t in uid:
            lbl = id_to_label.get(int(t), "")
            rgba = TISSUE_RGBA.get(lbl, EXTRA[0])
            surf = tissue_surface(int(t))
            if lbl == "skin":
                pl.add_mesh(surf, color=rgba[:3], opacity=0.10,
                            style="wireframe", line_width=0.4)
            else:
                pl.add_mesh(surf, color=rgba[:3], opacity=float(rgba[3]),
                            smooth_shading=True)
        pl.add_axes()
        pl.show_bounds(grid="back", location="outer", color="grey")
        pl.show()
        return None

    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(2000, 1600), border=False)
    pl.set_background("white")
    outer_lbl = max(id_to_label.values(), key=lambda k: (tissue == inv[k]).sum())
    outer_id = inv[outer_lbl]
    outer_surf = tissue_surface(int(outer_id))

    def add_panel(row, col, title, origin, normal, cam_fn, legend=False):
        pl.subplot(row, col)
        slc = grid.slice(normal=normal, origin=origin)
        pl.add_mesh(slc, **MESH_KWARGS)
        pl.add_mesh(slc, style="wireframe", color="black", opacity=0.18, line_width=0.4)
        pl.add_mesh(outer_surf, style="wireframe", color="#666666",
                    opacity=0.06, line_width=0.4)
        cam_fn()
        pl.reset_camera()
        pl.add_title(title, font_size=12, color="black")
        if legend:
            leg = []
            for i, t in enumerate(uid):
                lbl = id_to_label.get(int(t), f"t{t}")
                rgba = TISSUE_RGBA.get(lbl, EXTRA[i % len(EXTRA)])
                leg.append([lbl, list(rgba[:3])])
            pl.add_legend(leg, bcolor="white", border=True,
                          size=(0.30, 0.40), loc="lower right", face="rectangle")

    add_panel(0, 0, f"Coronal (Y={y_slice:.0f} mm)",
              [cx, y_slice, cz], [0, 1, 0], pl.view_xz)
    add_panel(0, 1, f"Sagittal (X={x_slice:.0f} mm)",
              [x_slice, cy, cz], [1, 0, 0], pl.view_yz)
    add_panel(1, 0, f"Axial (Z={z_slice:.0f} mm)",
              [cx, cy, z_slice], [0, 0, 1], pl.view_xy, legend=True)

    pl.subplot(1, 1)
    for i, t in enumerate(uid):
        lbl = id_to_label.get(int(t), "")
        rgba = TISSUE_RGBA.get(lbl, EXTRA[i % len(EXTRA)])
        if lbl == "skin":
            pl.add_mesh(tissue_surface(int(t)), style="wireframe",
                        color=rgba[:3], opacity=0.10, line_width=0.5)
        else:
            pl.add_mesh(tissue_surface(int(t)), color=rgba[:3],
                        opacity=float(rgba[3]), smooth_shading=True)
    pl.view_isometric()
    pl.reset_camera()
    pl.add_title("3-D (skin = wire cage)", font_size=12, color="black")

    out = cfg.outputs.fem_png
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out))
    logger.info("[saved] %s", out)
    return out
