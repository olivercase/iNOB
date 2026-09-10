#!/usr/bin/env python3
"""Render ALL BodyParts3D 4.3 meshes together, coloured by anatomical system.

Reads the categorised set (``internal_meshes/MANIFEST.csv`` + ``<system>/*.obj``)
and produces, under ``--out`` (default ``outputs/``):

  bodyparts3d_4.3_<view>.png   anterior / left / superior / oblique offscreen renders
  bodyparts3d_4.3_overview.png a labelled montage of the four views with a colour key
  bodyparts3d_4.3_all.glb      one interactive, per-system-coloured scene (open in any
                               glTF viewer / browser) for free rotation

Uses VTK for robust offscreen rendering (no display needed). Skin is rendered
semi-transparent so internal structures stay visible.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger("bp3d_viz")

# system -> RGB (0..1) + opacity. Anatomically suggestive, high-contrast.
SYSTEM_STYLE: dict[str, tuple[tuple[float, float, float], float]] = {
    "blood_vessel": ((0.80, 0.12, 0.12), 1.0),  # red
    "muscle": ((0.70, 0.32, 0.30), 1.0),  # brick
    "organ": ((0.85, 0.55, 0.55), 1.0),  # pink-tan
    "bone": ((0.94, 0.92, 0.84), 1.0),  # ivory
    "nerve": ((0.96, 0.86, 0.15), 1.0),  # yellow
    "brain": ((0.75, 0.70, 0.78), 1.0),  # pinkish-grey
    "ligament_tendon": ((0.55, 0.75, 0.95), 1.0),  # light blue
    "cartilage": ((0.55, 0.90, 0.90), 1.0),  # cyan
    "gland": ((0.95, 0.60, 0.20), 1.0),  # orange
    "lymphatic": ((0.35, 0.75, 0.40), 1.0),  # green
    "skin": ((0.90, 0.78, 0.68), 0.12),  # translucent beige
    "other": ((0.6, 0.6, 0.6), 1.0),  # grey
}

# camera presets: (view_up, direction-from-center). +X left, +Y post?, +Z sup.
# BodyParts3D world: X left-right, Y ant-post, Z inferior-superior (mm).
VIEWS = {
    "anterior": ((0, 0, 1), (0, -1, 0)),
    "left": ((0, 0, 1), (1, 0, 0)),
    "superior": ((0, 1, 0), (0, 0, 1)),
    "oblique": ((0, 0, 1), (0.7, -0.7, 0.2)),
}


def load_rows(internal: Path) -> list[tuple[Path, str]]:
    man = internal / "MANIFEST.csv"
    rows = []
    with man.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = internal.parent / r["path"] if r.get("path") else None
            if p is None or not p.exists():
                # fall back to <system>/<reconstructed name>
                continue
            rows.append((p, r["system"]))
    return rows


def render_vtk(rows, out: Path, size: int) -> bool:
    try:
        import vtk
    except ImportError:
        logger.error("vtk not available")
        return False

    logger.info("loading %d meshes into VTK ...", len(rows))
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.12)

    for i, (path, system) in enumerate(rows):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(str(path))
        reader.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        (r, g, b), opac = SYSTEM_STYLE.get(system, SYSTEM_STYLE["other"])
        actor.GetProperty().SetColor(r, g, b)
        actor.GetProperty().SetOpacity(opac)
        renderer.AddActor(actor)
        if (i + 1) % 500 == 0:
            logger.info("  added %d/%d", i + 1, len(rows))

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(renderer)
    win.SetSize(size, size)

    renderer.GetActiveCamera().ParallelProjectionOn()
    pngs = {}
    for name, (up, direction) in VIEWS.items():
        cam = renderer.GetActiveCamera()
        renderer.ResetCamera()
        fp = cam.GetFocalPoint()
        dist = cam.GetDistance()
        cam.SetViewUp(*up)
        cam.SetPosition(
            fp[0] + direction[0] * dist, fp[1] + direction[1] * dist, fp[2] + direction[2] * dist
        )
        renderer.ResetCamera()
        win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        w2i.Update()
        png = out / f"bodyparts3d_4.3_{name}.png"
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(png))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        pngs[name] = png
        logger.info("rendered %s -> %s", name, png.name)

    make_montage(pngs, rows, out)
    return True


def make_montage(pngs: dict, rows, out: Path) -> None:
    from collections import Counter

    from PIL import Image, ImageDraw, ImageFont

    counts = Counter(s for _, s in rows)
    imgs = {k: Image.open(v).convert("RGB") for k, v in pngs.items()}
    if not imgs:
        return
    w, h = next(iter(imgs.values())).size
    order = [k for k in ("anterior", "left", "superior", "oblique") if k in imgs]
    cols, rowsn = 2, 2
    legend_h = 150
    canvas = Image.new("RGB", (w * cols, h * rowsn + legend_h), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = small = ImageFont.load_default()
    for idx, k in enumerate(order):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(imgs[k], (x, y))
        draw.text((x + 12, y + 10), k.upper(), fill=(255, 255, 255), font=font)
    # legend
    ly = h * rowsn + 16
    draw.text(
        (16, ly - 4),
        f"BodyParts3D 4.3 — {len(rows)} meshes by system",
        fill=(255, 255, 255),
        font=font,
    )
    lx, ly2 = 16, ly + 40
    for system, ((r, g, b), _o) in SYSTEM_STYLE.items():
        if system not in counts:
            continue
        draw.rectangle(
            [lx, ly2, lx + 26, ly2 + 26], fill=(int(r * 255), int(g * 255), int(b * 255))
        )
        label = f"{system} ({counts[system]})"
        draw.text((lx + 34, ly2 + 1), label, fill=(230, 230, 230), font=small)
        lx += 34 + int(draw.textlength(label, font=small)) + 40
        if lx > w * cols - 240:
            lx = 16
            ly2 += 36
    montage = out / "bodyparts3d_4.3_overview.png"
    canvas.save(montage)
    logger.info("wrote montage -> %s", montage)


def export_glb(rows, out: Path) -> None:
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    logger.info("building combined GLB ...")
    for i, (path, system) in enumerate(rows):
        try:
            m = trimesh.load(path, process=False)
        except Exception:
            continue
        if m.is_empty or len(m.faces) == 0:
            continue
        (r, g, b), opac = SYSTEM_STYLE.get(system, SYSTEM_STYLE["other"])
        rgba = [int(r * 255), int(g * 255), int(b * 255), int(opac * 255)]
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(rgba, (len(m.faces), 1)))
        scene.add_geometry(m, node_name=f"{system}_{i}")
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d", i + 1, len(rows))
    glb = out / "bodyparts3d_4.3_all.glb"
    glb.write_bytes(scene.export(file_type="glb"))
    logger.info("wrote interactive scene -> %s (%.0f MB)", glb, glb.stat().st_size / 1e6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--internal", type=Path, default=Path("internal_meshes"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--size", type=int, default=1400, help="render size (px, square)")
    ap.add_argument("--no-glb", action="store_true", help="skip interactive GLB export")
    ap.add_argument("--no-png", action="store_true", help="skip PNG renders")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.internal)
    logger.info("%d meshes from %s", len(rows), args.internal)
    if not rows:
        logger.error("no meshes found — run categorize_bodyparts3d.py first")
        return 1
    if not args.no_png:
        render_vtk(rows, args.out, args.size)
    if not args.no_glb:
        export_glb(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
