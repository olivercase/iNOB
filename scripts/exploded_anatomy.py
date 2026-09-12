#!/usr/bin/env python3
"""Exploded view of the anatomical systems iNOB can model, labelled.

One figure, one body. The skin stays where it belongs, translucent, and each
system is drawn twice: once in place inside the body, and once pulled out
sideways to a clear column with a leader line back to where it came from. That
is what makes it an exploded view rather than a row of thumbnails — you can see
both what each system is and where in the body it sits.

The systems are the ones the forward model has to represent: a brain, a spine
and the cord inside it, the vagus nerve, the gut, the great vessels, and limb
muscle. Meshes come from :mod:`fetch_bodyparts3d_subset`, which records in its
``MANIFEST.csv`` exactly which BodyParts3D concept each file is.

Provenance: BodyParts3D/Anatomography, version 4.3, © The Database Center for
Life Science, licensed CC BY-SA 2.1 JP.

Usage::

    python3 scripts/fetch_bodyparts3d_subset.py        # once, ~100 MB
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

#: Colour and opacity per system. Anatomically suggestive and high-contrast in
#: print; the same convention as ``visualize_bodyparts3d.py`` so the two
#: figures can sit in one paper without re-teaching the reader the key.
STYLE: dict[str, tuple[tuple[float, float, float], float]] = {
    "brain": ((0.72, 0.66, 0.76), 1.0),
    "spine": ((0.93, 0.90, 0.81), 1.0),
    "spinal_cord": ((0.35, 0.62, 0.85), 1.0),
    # Darker than a highlighter yellow: the vagus is a 2 mm thread on a 1.7 m
    # body and a pale colour disappears against white at figure scale.
    "vagus_nerve": ((0.78, 0.58, 0.04), 1.0),
    "gut": ((0.86, 0.60, 0.45), 1.0),
    "blood_vessel": ((0.78, 0.14, 0.16), 1.0),
    "leg_muscle": ((0.70, 0.32, 0.30), 1.0),
    "skin": ((0.88, 0.76, 0.66), 0.10),
}

#: Human-readable name per system, for the labels.
LABEL: dict[str, str] = {
    "brain": "Brain",
    "spine": "Vertebral column",
    "spinal_cord": "Spinal cord",
    "vagus_nerve": "Vagus nerve",
    "gut": "Gastrointestinal tract",
    "blood_vessel": "Great vessels",
    "leg_muscle": "Lower-limb muscle",
    "skin": "Skin",
}

#: Columns of displaced systems, innermost first, alternating sides.
#:
#: Four of these systems (spine, cord, vagus, vessels) span the trunk's whole
#: height, so none of them can be stacked and each needs a column of its own.
#: Brain and lower-limb muscle do not overlap in height at all, so they share
#: one. The widest column goes outermost, where it costs the least width.
COLUMN_ORDER: list[list[str]] = [
    ["spine"],
    ["blood_vessel"],
    ["spinal_cord"],
    ["vagus_nerve"],
    ["gut"],
    ["brain", "leg_muscle"],
]

#: Clear space left between a column and its neighbour, mm.
COLUMN_GAP_MM = 90.0


def load_groups(base: Path) -> dict[str, pv.PolyData]:
    """One merged surface per system, read from the subset manifest."""
    manifest = base / "MANIFEST.csv"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found — run scripts/fetch_bodyparts3d_subset.py first")
    paths: dict[str, list[Path]] = collections.defaultdict(list)
    with manifest.open() as f:
        for row in csv.DictReader(f):
            paths[row["group"]].append(base / row["obj"])

    groups: dict[str, pv.PolyData] = {}
    for group, files in paths.items():
        blocks = []
        for p in files:
            m = trimesh.load(p, process=False, force="mesh")
            faces = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel()
            blocks.append(pv.PolyData(np.asarray(m.vertices, float), faces))
        merged = blocks[0] if len(blocks) == 1 else pv.merge(blocks)
        groups[group] = merged
        logger.info("%-14s %3d meshes, %7d faces", group, len(files), merged.n_cells)
    return groups


def plan_columns(groups: dict[str, pv.PolyData]) -> dict[str, float]:
    """Lateral displacement per system, packed from the meshes' own widths.

    Hard-coded offsets have to be re-tuned whenever the cast list changes, and
    a column that silently overlaps its neighbour is the one mistake this
    figure cannot survive. Packing outward from the body silhouette, each
    column claiming exactly the width it needs plus a fixed gap, makes overlap
    impossible by construction and keeps the figure as narrow as its contents
    allow.
    """

    def half_width(names: list[str]) -> float:
        return max((groups[n].bounds[1] - groups[n].bounds[0]) / 2 for n in names)

    edge = {-1: (groups["skin"].bounds[1] - groups["skin"].bounds[0]) / 2}
    edge[1] = edge[-1]
    dx: dict[str, float] = {"skin": 0.0}
    for i, column in enumerate(COLUMN_ORDER):
        side = -1 if i % 2 == 0 else 1
        hw = half_width(column)
        centre = side * (edge[side] + COLUMN_GAP_MM + hw)
        edge[side] += COLUMN_GAP_MM + 2 * hw
        for name in column:
            # Displace the mesh's own centre onto the column, not its origin:
            # a system that sits off the midline (the vagus does) would
            # otherwise land off-centre in its column and crowd a neighbour.
            dx[name] = centre - float(groups[name].center[0])
    return dx


def add_system(
    pl: pv.Plotter,
    group: str,
    mesh: pv.PolyData,
    *,
    dx: float,
    label_z: float,
) -> None:
    """Draw one system in place, again displaced, and the leader between them."""
    colour, opacity = STYLE[group]
    ghost = group == "skin"

    # In place. Every system keeps a copy where it actually lives, so the
    # exploded copy is an annotation of the body rather than a replacement
    # for it. The skin is the exception: it is only ever shown in place.
    pl.add_mesh(
        mesh,
        color=colour,
        opacity=0.10 if not ghost else opacity,
        smooth_shading=True,
        specular=0.15,
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
        specular=0.25,
        show_scalar_bar=False,
    )

    # The leader runs between the two copies at their common height, so it
    # reads as "this came from there" and not as an anatomical connection.
    c = np.asarray(mesh.center, float)
    pl.add_mesh(
        pv.Line((c[0], c[1], c[2]), (c[0] + dx, c[1], c[2])),
        color=(0.45, 0.45, 0.45),
        line_width=1.5,
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
        text_color="black",
        shape=None,
        show_points=False,
        always_visible=True,
        justification_horizontal="center",
    )


def scale_bar(pl: pv.Plotter, *, x0: float, z0: float, y: float, mm: float = 500.0) -> None:
    """A bar of known length: the axes are off and the body is not a unit."""
    pl.add_mesh(pv.Line((x0, y, z0), (x0 + mm, y, z0)), color="black", line_width=6)
    _label(pl, (x0 + mm / 2, y, z0 - 95.0), f"{mm:.0f} mm", size=26)


#: Headroom above a system's top for its label, and margin round the whole
#: scene, both in mm.
LABEL_RISE_MM = 85.0
MARGIN_MM = 70.0

#: Label type size, in points of the render's own font. Width per character is
#: about 0.55 em for this face, which is what :func:`stack_labels` uses to know
#: whether two labels overlap.
LABEL_FONT_SIZE = 30
_EM_PER_CHAR = 0.55
_LINE_HEIGHT = 1.7


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
    Rather than widen every column to fit its caption (which would stretch the
    figure to nearly twice its width for the sake of two words), overlapping
    labels are lifted onto separate lines.

    Deterministic: labels are considered left to right, and only ever move up,
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


def render(groups: dict[str, pv.PolyData], out: Path, *, height: int) -> Path:
    pv.global_theme.background = "white"
    dx = plan_columns(groups)

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
    xs, zs = [], []
    for g, m in groups.items():
        b = m.bounds
        xs += [b[0] + dx[g], b[1] + dx[g]]
        zs += [b[4], label_z.get(g, b[5])]
    bar_z = min(zs) - 110.0
    zs.append(bar_z - 150.0)
    x_lo, x_hi = min(xs) - MARGIN_MM, max(xs) + MARGIN_MM
    z_lo, z_hi = min(zs) - MARGIN_MM, max(zs) + MARGIN_MM

    aspect = (x_hi - x_lo) / (z_hi - z_lo)
    pl = pv.Plotter(off_screen=True, window_size=(round(height * aspect), height))
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
    pl.enable_lightkit()

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
        default=2200,
        help="Image height in px; the width follows from the scene's aspect.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    render(load_groups(args.src), args.out, height=args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
