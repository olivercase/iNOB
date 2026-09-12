#!/usr/bin/env python3
"""Exploded view of the anatomical systems iNOB can model, labelled.

One figure, one body. The skin stays where it belongs, translucent, and each
system is drawn twice: once in place inside the body, and once pulled out
sideways to a clear column with a leader line back to where it came from. That
is what makes it an exploded view rather than a row of thumbnails — you can see
both what each system is and where in the body it sits.

The systems are the ones a whole-body forward model has to represent: brain,
vertebral column, spinal cord, vagus nerve, heart, lungs, gut, great vessels
and limb muscle. Meshes come from :mod:`fetch_bodyparts3d_subset`, which
records in its ``MANIFEST.csv`` exactly which BodyParts3D concept each file is.

Nothing about the layout is hand-placed. Systems are packed into columns by
their heights, columns are ordered by width and dealt to whichever side has
room, and labels that would collide are lifted apart — so adding a system
re-flows the figure instead of breaking it.

Provenance: BodyParts3D/Anatomography, version 4.3, © The Database Center for
Life Science, licensed CC BY-SA 2.1 JP.

Usage::

    python3 scripts/fetch_bodyparts3d_subset.py        # once, ~110 MB
    python3 scripts/exploded_anatomy.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

logger = logging.getLogger("exploded")


# ── appearance ─────────────────────────────────────────────────────────────

#: Colour, opacity and surface finish per system. Anatomically suggestive and
#: separable in print: the three red things (heart, great vessels, muscle) are
#: deliberately a deep crimson, a bright arterial red and a desaturated brick,
#: so they do not read as one material.
STYLE: dict[str, tuple[tuple[float, float, float], float, float]] = {
    # name: (rgb, opacity, roughness) — roughness 0 is wet and glossy, 1 matte.
    "brain": ((0.78, 0.74, 0.82), 1.0, 0.55),
    "spine": ((0.94, 0.91, 0.83), 1.0, 0.45),
    "spinal_cord": ((0.29, 0.56, 0.80), 1.0, 0.40),
    # Darker than a highlighter yellow: the vagus is a thread on a 1.7 m body
    # and a pale colour disappears against white at figure scale.
    "vagus_nerve": ((0.83, 0.63, 0.09), 1.0, 0.50),
    "heart": ((0.60, 0.11, 0.16), 1.0, 0.30),
    "lungs": ((0.86, 0.64, 0.64), 1.0, 0.60),
    "gut": ((0.85, 0.58, 0.43), 1.0, 0.35),
    "blood_vessel": ((0.82, 0.17, 0.17), 1.0, 0.30),
    "leg_muscle": ((0.68, 0.31, 0.28), 1.0, 0.55),
    "skin": ((0.93, 0.83, 0.75), 0.11, 0.25),
}

#: Human-readable name per system, for the labels.
LABEL: dict[str, str] = {
    "brain": "Brain",
    "spine": "Vertebral column",
    "spinal_cord": "Spinal cord",
    "vagus_nerve": "Vagus nerve",
    "heart": "Heart",
    "lungs": "Lungs",
    "gut": "Gastrointestinal tract",
    "blood_vessel": "Great vessels",
    "leg_muscle": "Lower-limb muscle",
    "skin": "Skin",
}

#: Opacity of the in-place copy. Low enough that nine overlaid systems still
#: read as one body rather than as mud, high enough to locate each of them.
GHOST_OPACITY = 0.13

#: Clear space between a column and its neighbour, and between two systems
#: sharing a column, in mm.
COLUMN_GAP_MM = 95.0
STACK_GAP_MM = 60.0

#: Headroom above a system's top for its label, and margin round the scene.
LABEL_RISE_MM = 80.0
MARGIN_MM = 80.0

#: Label type size, in points of the render's own font. Width per character is
#: about 0.55 em for this face, which is what :func:`stack_labels` uses to know
#: whether two labels overlap.
LABEL_FONT_SIZE = 32
LABEL_FONT = "arial"
_EM_PER_CHAR = 0.55
_LINE_HEIGHT = 1.7


# ── geometry ───────────────────────────────────────────────────────────────


def load_groups(base: Path) -> dict[str, pv.PolyData]:
    """One merged surface per system, read from the subset manifest."""
    manifest = base / "MANIFEST.csv"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found — run scripts/fetch_bodyparts3d_subset.py first")
    paths: dict[str, list[Path]] = collections.defaultdict(list)
    with manifest.open() as f:
        for row in csv.DictReader(f):
            paths[row["group"]].append(base / row["obj"])

    unknown = sorted(set(paths) - set(STYLE))
    if unknown:
        # A system with no style would render in VTK's default white and be
        # silently unreadable; better to say so than to ship that figure.
        raise SystemExit(f"no STYLE entry for: {', '.join(unknown)}")

    groups: dict[str, pv.PolyData] = {}
    for group, files in sorted(paths.items()):
        blocks = []
        for p in files:
            m = trimesh.load(p, process=False, force="mesh")
            faces = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel()
            blocks.append(pv.PolyData(np.asarray(m.vertices, float), faces))
        merged = blocks[0] if len(blocks) == 1 else pv.merge(blocks)
        # Vertex normals, computed once: the OBJs carry none, and flat-shaded
        # viscera look like low-poly game assets.
        groups[group] = merged.compute_normals(
            cell_normals=False, point_normals=True, auto_orient_normals=True, split_vertices=False
        )
        logger.info("%-14s %3d meshes, %7d faces", group, len(files), merged.n_cells)
    return groups


def pack_columns(groups: dict[str, pv.PolyData]) -> dict[str, float]:
    """Lateral displacement per system, derived from the meshes themselves.

    Two things have to be true and neither should be hand-maintained: no two
    columns may overlap, and the figure should be no wider than its contents
    need. So systems are first packed into columns — two may share one only if
    their heights do not overlap, which is why brain and lower-limb muscle sit
    together and the four trunk-length systems cannot — and the columns are
    then dealt outward from the body silhouette, narrowest first, each to
    whichever side is currently narrower. Adding a system re-flows the figure
    rather than colliding with a neighbour.
    """
    named = [g for g in groups if g != "skin"]
    z = {g: (float(groups[g].bounds[4]), float(groups[g].bounds[5])) for g in named}
    width = {g: float(groups[g].bounds[1] - groups[g].bounds[0]) for g in named}

    # Pack by height, tallest first: a system spanning the whole trunk can
    # never share, so placing it early keeps the greedy choice honest.
    columns: list[list[str]] = []
    for g in sorted(named, key=lambda n: z[n][0] - z[n][1]):
        for col in columns:
            if all(
                z[g][0] > z[o][1] + STACK_GAP_MM or z[g][1] < z[o][0] - STACK_GAP_MM for o in col
            ):
                col.append(g)
                break
        else:
            columns.append([g])

    half = {id(c): max(width[g] for g in c) / 2 for c in columns}
    edge = {-1: float(groups["skin"].bounds[1] - groups["skin"].bounds[0]) / 2}
    edge[1] = edge[-1]

    dx: dict[str, float] = {"skin": 0.0}
    for col in sorted(columns, key=lambda c: half[id(c)]):
        side = -1 if edge[-1] <= edge[1] else 1
        hw = half[id(col)]
        centre = side * (edge[side] + COLUMN_GAP_MM + hw)
        edge[side] += COLUMN_GAP_MM + 2 * hw
        for g in col:
            # Displace the mesh's own centre onto the column, not its origin:
            # a system that sits off the midline (the vagus does) would
            # otherwise land off-centre in its column and crowd a neighbour.
            dx[g] = centre - float(groups[g].center[0])
    return dx


def stack_labels(
    placed: dict[str, tuple[float, float]],
    texts: dict[str, str],
    *,
    mm_per_pt: float,
) -> dict[str, float]:
    """Raise labels until no two of them overlap. Returns the new heights.

    Columns are packed on the meshes' widths, but a label is as wide as its
    text, not as its mesh — "Vertebral column" is far wider than the spine —
    so neighbouring labels collide even when the systems they name do not.
    Rather than widen every column to fit its caption, which would stretch the
    figure for the sake of two words, overlapping labels are lifted onto
    separate lines.

    Deterministic: labels are considered left to right and only ever move up,
    so the result does not depend on dict order and cannot oscillate.
    """
    line = LABEL_FONT_SIZE * _LINE_HEIGHT * mm_per_pt
    order = sorted(placed, key=lambda g: placed[g][0])
    out = {g: placed[g][1] for g in placed}
    for i, g in enumerate(order):
        for h in order[:i]:
            half_span = (
                (len(texts[g]) + len(texts[h])) / 2 * _EM_PER_CHAR * LABEL_FONT_SIZE * mm_per_pt
            )
            if abs(placed[g][0] - placed[h][0]) >= half_span:
                continue
            if abs(out[g] - out[h]) < line:
                out[g] = out[h] + line
    return out


# ── drawing ────────────────────────────────────────────────────────────────


def add_system(
    pl: pv.Plotter,
    group: str,
    mesh: pv.PolyData,
    *,
    dx: float,
    label_z: float,
) -> None:
    """Draw one system in place, again displaced, and the leader between them."""
    colour, opacity, roughness = STYLE[group]
    ghost = group == "skin"

    # In place. Every system keeps a copy where it actually lives, so the
    # exploded copy annotates the body rather than replacing it. The skin is
    # the exception: it is only ever shown in place, as the shell.
    pl.add_mesh(
        mesh,
        color=colour,
        opacity=opacity if ghost else GHOST_OPACITY,
        smooth_shading=True,
        specular=0.45 if ghost else 0.10,
        specular_power=30 if ghost else 10,
        diffuse=0.85,
        ambient=0.22,
        show_scalar_bar=False,
    )
    if ghost:
        return

    moved = mesh.copy()
    moved.translate((dx, 0.0, 0.0), inplace=True)
    pl.add_mesh(
        moved,
        color=colour,
        opacity=opacity,
        smooth_shading=True,
        # Glossier surfaces for the wet viscera, matter ones for bone and
        # muscle: one specular setting for all nine makes them look moulded
        # from the same plastic.
        specular=0.55 * (1.0 - roughness),
        specular_power=12 + 60 * (1.0 - roughness),
        diffuse=0.90,
        ambient=0.20,
        show_scalar_bar=False,
    )

    # The leader runs between the two copies at their common height, so it
    # reads as "this came from there" and not as an anatomical connection.
    c = np.asarray(mesh.center, float)
    pl.add_mesh(
        pv.Line((c[0], c[1], c[2]), (c[0] + dx, c[1], c[2])),
        color=(0.55, 0.55, 0.58),
        line_width=1.4,
    )
    _label(pl, (c[0] + dx, c[1], label_z), LABEL[group], size=LABEL_FONT_SIZE)


def _label(pl: pv.Plotter, at: tuple[float, float, float], text: str, *, size: int) -> None:
    """Centred text at a world point, with no marker and no halo box.

    Centred because a label anchored by its left edge drifts off its own
    column as the text gets longer, which put "Gastrointestinal tract" over
    the neighbouring one.
    """
    pl.add_point_labels(
        np.array([at], dtype=float),
        [text],
        font_size=size,
        font_family=LABEL_FONT,
        text_color=(0.10, 0.10, 0.12),
        shape=None,
        show_points=False,
        always_visible=True,
        justification_horizontal="center",
    )


def scale_bar(pl: pv.Plotter, *, x0: float, z0: float, y: float, mm: float = 500.0) -> None:
    """A bar of known length: the axes are off and the body is not a unit."""
    pl.add_mesh(pv.Line((x0, y, z0), (x0 + mm, y, z0)), color=(0.10, 0.10, 0.12), line_width=6)
    _label(pl, (x0 + mm / 2, y, z0 - 95.0), f"{mm:.0f} mm", size=LABEL_FONT_SIZE - 6)


def light_the_scene(pl: pv.Plotter) -> None:
    """Three-point lighting, which is what stops this looking like a CT render.

    VTK's default light kit is a headlight plus fills: flat, and it washes the
    translucent skin out completely. A key from the front-left with a cool
    fill opposite and a rim from behind gives each organ a lit side, a shaded
    side and an edge — the three things that make a form read as solid.
    """
    pl.remove_all_lights()
    for position, intensity, colour in (
        ((-0.55, -1.0, 0.45), 0.85, (1.00, 0.97, 0.93)),  # key, warm
        ((0.80, -0.60, 0.10), 0.35, (0.90, 0.94, 1.00)),  # fill, cool
        ((0.10, 1.00, 0.35), 0.45, (1.00, 1.00, 1.00)),  # rim, from behind
    ):
        light = pv.Light(position=position, light_type="scene light", intensity=intensity)
        light.diffuse_color = colour
        light.positional = False
        pl.add_light(light)


def render(groups: dict[str, pv.PolyData], out: Path, *, height: int) -> Path:
    dx = pack_columns(groups)

    # Labels sit above each system's own top, not on a shared line: the
    # systems occupy different heights and a shared line would float far from
    # most of them. Overlapping ones are then lifted apart.
    named = {g: m for g, m in groups.items() if g != "skin"}
    mm_per_pt = (groups["skin"].bounds[5] - groups["skin"].bounds[4]) / height
    label_z = stack_labels(
        {
            g: (float(m.center[0]) + dx[g], float(m.bounds[5]) + LABEL_RISE_MM)
            for g, m in named.items()
        },
        {g: LABEL[g] for g in named},
        mm_per_pt=mm_per_pt,
    )

    # Fit the frame to what is actually drawn — every displaced mesh, plus the
    # label above it and the scale bar below — rather than to the body alone,
    # which left a third of the canvas empty on the side with the narrow
    # systems and clipped the scale bar off the bottom.
    xs: list[float] = []
    zs: list[float] = []
    for g, m in groups.items():
        b = m.bounds
        xs += [b[0] + dx[g], b[1] + dx[g]]
        zs += [b[4], label_z.get(g, b[5])]
    bar_z = min(zs) - 120.0
    zs.append(bar_z - 160.0)
    x_lo, x_hi = min(xs) - MARGIN_MM, max(xs) + MARGIN_MM
    z_lo, z_hi = min(zs) - MARGIN_MM, max(zs) + MARGIN_MM

    aspect = (x_hi - x_lo) / (z_hi - z_lo)
    pl = pv.Plotter(off_screen=True, window_size=(round(height * aspect), height))
    # Flat white: the figure is for print, where a gradient becomes a banded
    # grey wash. The skin shell is instead given its edge by the rim light.
    pl.set_background("white")

    for group, mesh in groups.items():
        add_system(pl, group, mesh, dx=dx[group], label_z=label_z.get(group, 0.0))
    scale_bar(pl, x0=x_lo + MARGIN_MM, z0=bar_z, y=float(groups["skin"].center[1]))

    # Anterior view, orthographic: an exploded diagram is read by comparing
    # sizes across the frame, and perspective would make the near columns
    # larger than the far ones for no informational gain.
    pl.enable_parallel_projection()
    centre = ((x_lo + x_hi) / 2, (z_lo + z_hi) / 2)
    pl.camera_position = [
        (centre[0], -8000.0, centre[1]),
        (centre[0], 0.0, centre[1]),
        (0.0, 0.0, 1.0),
    ]
    # parallel_scale is half the viewport height in world units; the width
    # then follows from the window aspect, which was chosen to match.
    pl.camera.parallel_scale = (z_hi - z_lo) / 2

    light_the_scene(pl)
    # Supersampled: these are thin, high-curvature surfaces (a nerve, a vessel
    # tree) and FXAA smears them where SSAA keeps them crisp.
    pl.enable_anti_aliasing("ssaa")

    out.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out))
    pl.close()
    logger.info("[saved] %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=Path("data/bodyparts3d/subset"))
    ap.add_argument("--out", type=Path, default=Path("outputs/anatomy/exploded_anatomy.png"))
    ap.add_argument(
        "--height",
        type=int,
        default=2600,
        help="Image height in px; the width follows from the scene's aspect.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    render(load_groups(args.src), args.out, height=args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
