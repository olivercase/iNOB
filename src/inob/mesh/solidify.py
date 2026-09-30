"""Turn a hollow anatomical shell into the solid body it bounds.

Several BodyParts3D structures are *organ* meshes rather than region boundaries.
``FMA7163 Skin`` is the clearest case: it is the skin itself, a closed double
wall about 2 mm thick, so its surface area is twice the body's (3.50 m^2 vs
1.75 m^2), the volume it encloses is the volume of skin (3.4 L) rather than of a
person (~70 L), and ``contains()`` is False for every point in the chest. As a
volume conductor that is a 2 mm rind around a vacuum.

Filling it is not a hole-closing problem. The shell is genus > 0: the inner and
outer walls are joined by tunnels at the facial orifices, so the body cavity is
topologically *outside* and ``binary_fill_holes`` walks straight through. Nor is
it a closing problem — sealing the tunnels needs a structuring element wider than
the widest orifice, which by then has eaten real anatomy (tested to 24 mm; the
fill still leaked).

What does work is morphological **reconstruction of the exterior**:

    1. occupancy   splat the wall into a voxel grid
    2. free        the complement — true exterior plus the leaked body cavity
    3. erode       by R, severing every channel narrower than 2R
    4. seed        keep only eroded components touching the grid border: these,
                   and nothing else, are genuinely outside the body
    5. reconstruct dilate that seed back, confined to ``free`` — recovering the
                   exterior's true shape without ever re-entering the cavity
    6. solid       everything else, surfaced by marching cubes

Erosion-then-reconstruction is what distinguishes "connected to infinity through
a narrow throat" from "connected to infinity"; plain opening cannot, because it
severs the tunnel and then puts it back.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_propagation,
    label,
)
from skimage.measure import marching_cubes

from inob.mesh.repair import cheap_repair

logger = logging.getLogger(__name__)

__all__ = ["solidify_shell", "exterior_mask"]


def _border_mask(shape: tuple[int, ...]) -> np.ndarray:
    b = np.zeros(shape, dtype=bool)
    b[0], b[-1] = True, True
    b[:, 0], b[:, -1] = True, True
    b[:, :, 0], b[:, :, -1] = True, True
    return b


def exterior_mask(occ: np.ndarray, *, seal_voxels: int) -> np.ndarray:
    """Voxels genuinely outside the shell, immune to leaks through its tunnels.

    ``seal_voxels`` is the erosion radius R in voxels: any channel narrower than
    ``2 * R * pitch`` is treated as a hole in the wall rather than a way out.
    """
    free = ~occ
    if seal_voxels <= 0:
        seed = free & _border_mask(free.shape)
        return binary_propagation(seed, mask=free)

    eroded = binary_erosion(free, iterations=seal_voxels, border_value=1)
    lab, n = label(eroded)
    border = _border_mask(free.shape)
    outside_ids = set(np.unique(lab[eroded & border])) - {0}
    if not outside_ids:
        logger.warning("exterior reconstruction: no eroded component reaches the "
                       "grid border (seal_voxels=%d too large?)", seal_voxels)
        return binary_propagation(free & border, mask=free)
    seed = np.isin(lab, list(outside_ids))
    logger.debug("exterior reconstruction: %d eroded components, %d outside",
                 n, len(outside_ids))
    return binary_propagation(seed, mask=free)


def solidify_shell(
    shell: trimesh.Trimesh,
    *,
    pitch: float = 3.0,
    n_samples: int = 600_000,
    seal_mm: float = 12.0,
    smooth_iter: int = 8,
    decimate_target: int | None = 40_000,
    seed: int | None = 0,
    wall_voxels: int = 2,
    min_fill_ratio: float = 5.0,
) -> trimesh.Trimesh:
    """The solid body bounded by a hollow ``shell``.

    ``seal_mm`` is the widest orifice to treat as a hole in the wall. It must
    exceed half the largest tunnel and stay below the thinnest real anatomy.
    ``wall_voxels`` is how far the wall is thickened before the exterior is
    reconstructed, and thinned again afterwards -- it must cover the shell's own
    thickness at this pitch. ``min_fill_ratio`` is the gate: the filled body must
    enclose at least this many times the shell's own volume.
    """
    pts, _ = trimesh.sample.sample_surface(shell, count=n_samples, seed=seed)
    pad = pitch * 6
    mn, mx = pts.min(0) - pad, pts.max(0) + pad
    shape = tuple(((mx - mn) / pitch).astype(int) + 1)
    ijk = np.clip(((pts - mn) / pitch).astype(int), 0, np.array(shape) - 1)
    occ = np.zeros(shape, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    # THICKEN, then reason, then thin back. The skin wall is ~2 mm and the grid
    # is coarser than that, so the wall is *sub-voxel*: whether a splat leaves a
    # 6-connected pinhole depends on how the surface happens to cut the grid. It
    # is not a small effect and it is not stable -- solidifying this shell in the
    # atlas frame gives 78 L, and solidifying the SAME shell after a ~1 deg
    # registration rotation used to give 19 L, i.e. the wall alone, because the
    # exterior found one pinhole and poured into the chest. A single dilation is
    # not enough to guarantee closure.
    #
    # ``wall_voxels`` dilations close it with margin. That thickens the wall
    # inward and outward alike, so the reconstructed exterior sits ``wall_voxels``
    # too far out; eroding the solid by the same amount puts the surface back
    # where the anatomy is. Morphologically this is an opening of the exterior,
    # and it is exact for a wall of any thickness below 2*wall_voxels*pitch.
    occ = binary_dilation(occ, iterations=wall_voxels)

    R = max(1, int(round(seal_mm / pitch)))
    ext = exterior_mask(occ, seal_voxels=R)
    solid = ~ext
    if wall_voxels > 0:
        solid = binary_erosion(solid, iterations=wall_voxels, border_value=0)
    frac = solid.sum() / solid.size
    solid_cm3 = solid.sum() * pitch ** 3 / 1000.0
    shell_cm3 = abs(float(shell.volume)) / 1000.0
    logger.info("[solidify] pitch=%.1f seal=%.0f mm (R=%d vox) wall=%d vox -> "
                "solid fraction %.3f (%.0f cm^3, shell encloses %.0f cm^3)",
                pitch, seal_mm, R, wall_voxels, frac, solid_cm3, shell_cm3)

    # GATE. The failure this guards against is silent and produces a conductor
    # that looks fine in every topology check: watertight, one body, sane
    # bounds -- and is a hollow rind. A body must enclose far more than the
    # volume of its own skin.
    if solid_cm3 < min_fill_ratio * shell_cm3:
        raise RuntimeError(
            f"solidify_shell: filled volume {solid_cm3:.0f} cm^3 is only "
            f"{solid_cm3 / max(shell_cm3, 1e-9):.1f}x the shell's own enclosed "
            f"volume {shell_cm3:.0f} cm^3 -- the exterior leaked into the cavity "
            f"and this is the wall, not the body. Raise wall_voxels or seal_mm."
        )

    padded = np.pad(solid, 1)
    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), level=0.5)
    verts = (verts - 1.0) * pitch + mn
    m = trimesh.Trimesh(verts, faces, process=True)
    # skimage's marching_cubes winds these faces so the enclosed volume comes out
    # negative, i.e. normals point into the solid. Flip once, here, rather than
    # leaving every downstream volume report signed.
    if m.volume < 0:
        m.invert()
    m = max(m.split(only_watertight=False), key=lambda b: len(b.faces)) \
        if len(m.split(only_watertight=False)) > 1 else m
    if decimate_target and len(m.faces) > decimate_target:
        try:
            m = m.simplify_quadric_decimation(face_count=decimate_target)
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning("decimation skipped: %s", e)
    if smooth_iter:
        trimesh.smoothing.filter_taubin(m, iterations=smooth_iter)
    m = cheap_repair(m)
    logger.info("[solidify] %d V / %d F, %.0f cm^3, watertight=%s euler=%d",
                len(m.vertices), len(m.faces), m.volume / 1000.0,
                m.is_watertight, m.euler_number)
    return m
