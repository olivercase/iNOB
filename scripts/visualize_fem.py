#!/usr/bin/env python3
"""Render the multi-tissue FEM (outputs/fem/fem_vagus.mat) coloured by tissue.

Produces, under ``--out`` (default ``outputs/fem``):
  fem_<view>.png          full + clipped (interior-exposing) renders
  fem_overview.png        montage with a colour key + per-tissue tet counts

Uses VTK offscreen. A sagittal clipping plane through the neck exposes the
deep tissues (vagus, vessels) inside the skin/muscle/bone envelope.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("fem_viz")

TISSUE_STYLE = {
    "vagus_left":   ((1.00, 0.85, 0.10), 1.0),   # yellow
    "vagus_right":  ((1.00, 0.65, 0.00), 1.0),   # amber
    "blood_vessel": ((0.85, 0.10, 0.10), 1.0),   # red
    "muscle":       ((0.72, 0.34, 0.32), 1.0),   # brick
    "bone":         ((0.94, 0.92, 0.84), 1.0),   # ivory
    "skin":         ((0.90, 0.78, 0.68), 0.18),  # translucent beige
}


def build_grid(nodes, tets, tissue):
    import vtk
    from vtk.util import numpy_support as ns
    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(nodes, dtype=np.float64)))
    grid.SetPoints(pts)
    cells = np.empty((len(tets), 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tets
    idarr = ns.numpy_to_vtkIdTypeArray(cells.ravel())
    cellarr = vtk.vtkCellArray()
    cellarr.SetCells(len(tets), idarr)
    grid.SetCells(vtk.VTK_TETRA, cellarr)
    tarr = ns.numpy_to_vtk(np.ascontiguousarray(tissue, dtype=np.int32))
    tarr.SetName("tissue")
    grid.GetCellData().SetScalars(tarr)
    return grid


def tissue_actor(grid, tid, rgb, opac, clip_plane=None):
    import vtk
    thr = vtk.vtkThreshold()
    thr.SetInputData(grid)
    thr.SetUpperThreshold(tid)
    thr.SetLowerThreshold(tid)
    thr.SetInputArrayToProcess(0, 0, 0,
                               vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "tissue")
    thr.Update()
    geo = vtk.vtkGeometryFilter()
    geo.SetInputConnection(thr.GetOutputPort())
    geo.Update()
    src = geo.GetOutputPort()
    if clip_plane is not None:
        clip = vtk.vtkClipPolyData()
        clip.SetInputConnection(src)
        clip.SetClipFunction(clip_plane)
        clip.Update()
        src = clip.GetOutputPort()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*rgb)
    actor.GetProperty().SetOpacity(opac)
    return actor


def render(mesh, out: Path, size: int):
    import vtk
    nodes, tets, tissue = mesh.nodes, mesh.tets, mesh.tissue
    labels = mesh.tissue_labels
    grid = build_grid(nodes, tets, tissue)
    label_to_id = {lab: i + 1 for i, lab in enumerate(labels)}
    center = nodes.mean(axis=0)

    def make_renderer(clip):
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.1, 0.1, 0.12)
        plane = None
        if clip:
            plane = vtk.vtkPlane()
            plane.SetOrigin(*center)
            plane.SetNormal(1, 0, 0)  # sagittal cut → expose deep neck tissues
        for lab in labels:
            (r, g, b), opac = TISSUE_STYLE.get(lab, ((0.6, 0.6, 0.6), 1.0))
            # when clipped, render skin opaque on the cut face for context
            if clip and lab == "skin":
                opac = 0.5
            ren.AddActor(tissue_actor(grid, label_to_id[lab], (r, g, b), opac,
                                      clip_plane=plane))
        return ren

    pngs = {}
    for name, clip in (("full", False), ("clipped", True)):
        ren = make_renderer(clip)
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.AddRenderer(ren)
        win.SetSize(size, size)
        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        ren.ResetCamera()
        fp = cam.GetFocalPoint(); dist = cam.GetDistance()
        cam.SetViewUp(0, 0, 1)
        cam.SetPosition(fp[0], fp[1] - dist, fp[2])  # anterior view
        ren.ResetCamera()
        win.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
        png = out / f"fem_{name}.png"
        wr = vtk.vtkPNGWriter(); wr.SetFileName(str(png))
        wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
        pngs[name] = png
        logger.info("rendered %s -> %s", name, png.name)

    # neck-zoom anterior view (skin translucent, no clip) — shows carotid sheath
    ren = make_renderer(False)
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren)
    win.SetSize(size, size)
    cam = ren.GetActiveCamera(); cam.ParallelProjectionOn()
    ren.ResetCamera()
    # zoom to upper body (vagus/neck region): focus near top quartile in z
    # focus on the neck: vagus nodes mark the carotid-sheath level
    vmask = np.isin(mesh.tissue, [label_to_id.get("vagus_left", -1),
                                  label_to_id.get("vagus_right", -1)])
    vtet = mesh.tets[vmask]
    if len(vtet):
        vnodes = nodes[np.unique(vtet)]
        fc = vnodes.mean(axis=0)
        fz = fc[2]
        cx = fc[0]
    else:
        zlo, zhi = nodes[:, 2].min(), nodes[:, 2].max()
        fz = zlo + 0.80 * (zhi - zlo); cx = center[0]
    cam.SetFocalPoint(cx, center[1], fz)
    cam.SetViewUp(0, 0, 1)
    cam.SetPosition(cx, center[1] - 1000, fz)
    cam.SetParallelScale(90.0)
    ren.ResetCameraClippingRange()
    win.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
    png = out / "fem_neck.png"
    wr = vtk.vtkPNGWriter(); wr.SetFileName(str(png)); wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    pngs["neck"] = png
    logger.info("rendered neck zoom -> %s", png.name)

    montage(pngs, mesh, out)


def montage(pngs, mesh, out: Path):
    from PIL import Image, ImageDraw, ImageFont
    order = [k for k in ("full", "clipped", "neck") if k in pngs]
    imgs = [Image.open(pngs[k]).convert("RGB") for k in order]
    w, h = imgs[0].size
    legend_h = 120
    canvas = Image.new("RGB", (w * len(imgs), h + legend_h), (20, 20, 24))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = small = ImageFont.load_default()
    titles = {"full": "FULL (skin translucent)", "clipped": "SAGITTAL CLIP",
              "neck": "NECK ZOOM (clip)"}
    counts = {lab: int((mesh.tissue == i + 1).sum()) for i, lab in enumerate(mesh.tissue_labels)}
    for i, k in enumerate(order):
        canvas.paste(imgs[i], (i * w, 0))
        d.text((i * w + 12, 10), titles.get(k, k), fill=(255, 255, 255), font=font)
    d.text((16, h + 12), f"Vagus FEM — {len(mesh.tets)} tets, {len(mesh.nodes)} nodes, "
           f"{len(mesh.tissue_labels)} tissues", fill=(255, 255, 255), font=font)
    lx, ly = 16, h + 56
    for lab in mesh.tissue_labels:
        (r, g, b), _ = TISSUE_STYLE.get(lab, ((0.6, 0.6, 0.6), 1.0))
        d.rectangle([lx, ly, lx + 26, ly + 26], fill=(int(r*255), int(g*255), int(b*255)))
        txt = f"{lab} ({counts[lab]})"
        d.text((lx + 34, ly + 1), txt, fill=(230, 230, 230), font=small)
        lx += 34 + int(d.textlength(txt, font=small)) + 36
    p = out / "fem_overview.png"
    canvas.save(p)
    logger.info("wrote montage -> %s", p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fem", type=Path, default=Path("outputs/fem/fem_vagus.mat"))
    ap.add_argument("--out", type=Path, default=Path("outputs/fem"))
    ap.add_argument("--size", type=int, default=1200)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    sys.path.insert(0, "src")
    from vagus_fm.io.hdf5 import load_fem
    mesh = load_fem(args.fem)
    logger.info("loaded FEM: %d tets, tissues=%s", len(mesh.tets), mesh.tissue_labels)
    args.out.mkdir(parents=True, exist_ok=True)
    render(mesh, args.out, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
