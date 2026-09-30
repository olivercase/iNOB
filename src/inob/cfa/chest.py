"""The real 8-site triaxial chest array, placed on the iNOB torso.

**What is real here and what is not.** The array's own geometry is measured: 8
sites on a rigid planar plate, 148 x 99 mm, 48 mm nearest-neighbour spacing, each
carrying three orthonormal sensitive axes, recovered from the OptiTrack rigid-body
timeseries of record 4 (``channelLevelRbTimeseriesChestSorted``) and verified rigid
to 0.00 mm over all 864,220 samples. Its *placement on this torso* is not measured
and cannot be, for two reasons:

* the pack's 85-channel head array is a simulation helmet, not the subject's
  45-site scanner-cast, so THAT pairing carries nothing (a RANSAC match finds
  9/45 inliers -- i.e. none). **But see below: matching against the subject's own
  scanner-cast slot table does work.**
* the pilot bridged that gap with the subject's own torso STL and six named slot
  fiducials, and that STL is not in the pack.

**2026-09-03 update.** Two of those dead ends have since been reopened:

* the OptiTrack head array DOES match the subject's own scanner-cast slot table
  (``table_of_info - all.csv``, 64 slots with position and orientation in
  head/MRI space) under multi-restart ICP -- median residual 2.29 mm, max 5.50,
  45/45 one-to-one, det(R)=+1. So OptiTrack -> scanner-cast/MRI is available;
  only the last leg to a subject ANATOMY is still missing, because the subject's
  scanner-cast STL/MRI itself is not on disk.
* the pilot's six torso fiducials ARE in the pack, inline in
  ``provenance/pilot_code/CFA_HEP_ChestInversion.m`` (``sensor_pts_chest`` /
  ``sensor_pts_head``, metres, in the area-weighted-centred torso frame). They
  are hand-picked to about 16 mm -- no triple of measured sites reproduces their
  triangle better than that -- but they are enough to fix the plate's pose
  relative to the heart, which is what :func:`placed_from_cfa` uses.

A curved array would partly register itself against the chest wall. This one is
flat to numerical precision, so it carries no shape information about where it
sat. The placement is therefore an explicit anatomical assumption -- the array
over the precordium, tangent to the chest surface at the measured standoff --
and is reported as such rather than presented as a registration.

Everything that matters for a sensitivity study survives that: the number of
sites, their spacing, the plate geometry, the triaxial readout, and the standoff.
What does not survive is any claim to reproduce this subject's exact recording
geometry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

__all__ = ["ChestArray", "load_real_chest_array"]


@dataclass
class ChestArray:
    """Triaxial chest array: ``n_site`` positions, ``3 * n_site`` channels."""

    site_pos_mm: np.ndarray       # (S, 3)
    site_frame: np.ndarray        # (S, 3, 3) rows = the three sensitive axes
    label: str = "real-8"

    @property
    def n_site(self) -> int:
        return len(self.site_pos_mm)

    @property
    def n_chan(self) -> int:
        return 3 * self.n_site

    @property
    def chanpos_mm(self) -> np.ndarray:
        return np.tile(self.site_pos_mm, (3, 1))

    @property
    def chanori(self) -> np.ndarray:
        """(3S, 3), ordered all-axis-1, all-axis-2, all-axis-3."""
        return np.concatenate([self.site_frame[:, k, :] for k in range(3)], 0)

    def placed_on(self, skin: trimesh.Trimesh, target_mm: np.ndarray, *,
                  standoff_mm: float = 10.0, up=(0.0, 0.0, 1.0)) -> "ChestArray":
        """Rigidly place the plate tangent to ``skin`` above ``target_mm``.

        The plate's own centroid goes to the skin point nearest ``target_mm``,
        offset outward by ``standoff_mm``; its plane normal goes to the outward
        surface normal there; its long axis is rotated to run as close to ``up``
        as the plane allows, which is how a chest plate is actually worn.
        Relative sensor geometry and the three axes per site are carried through
        unchanged -- this is one rigid transform, not a refit.
        """
        p = np.asarray(target_mm, float).reshape(1, 3)
        closest, _, tri = trimesh.proximity.closest_point(skin, p)
        n = np.asarray(skin.face_normals[tri[0]], float)
        if np.dot(n, closest[0] - np.asarray(skin.centroid)) < 0:
            n = -n                                     # outward
        centre = closest[0] + standoff_mm * n

        c = self.site_pos_mm - self.site_pos_mm.mean(0)
        _, _, vt = np.linalg.svd(c)
        src = np.stack([vt[0], vt[1], np.cross(vt[0], vt[1])])   # plate frame
        u = np.asarray(up, float)
        e1 = u - np.dot(u, n) * n                      # in-plane, pointing up
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(n, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        dst = np.stack([e1, np.cross(n, e1), n])
        rot = dst.T @ src
        if np.linalg.det(rot) < 0:
            dst[1] *= -1
            rot = dst.T @ src
        pos = c @ rot.T + centre
        frame = np.einsum("ij,skj->ski", rot, self.site_frame)
        # A rigid flat plate on a curved chest rests on the most prominent point
        # under it, not on the tangent point of its centre. So slide it out along
        # the normal until the CLOSEST site is at the requested standoff; the
        # others then stand further off, which is what a flat plate does and is
        # why a real chest array is a poor sampler of a curved surface.
        sd = trimesh.proximity.signed_distance(skin, pos)
        pos = pos + (standoff_mm + sd.max()) * n
        sd = trimesh.proximity.signed_distance(skin, pos)
        logger.info("[chest] placed at %s mm; clearance %.1f-%.1f mm outside the "
                    "skin (%d inside)", np.round(centre, 1).tolist(),
                    float(-sd.max()), float(-sd.min()), int((sd > 0).sum()))
        return ChestArray(site_pos_mm=pos, site_frame=frame, label=self.label)


def conformal_array(skin: trimesh.Trimesh, target_mm: np.ndarray, *,
                    n_site: int, span_mm: float = 160.0,
                    standoff_mm: float = 6.5) -> ChestArray:
    """A hypothetical array of ``n_site`` triaxial sensors CONFORMAL to the chest.

    The comparator the real plate needs. The measured array is flat, so its
    corner sensors stand 35 mm off a curved chest and lose most of their signal;
    a conformal array of the same site count is the fair test of whether the
    limitation is the geometry or the count. Sites are laid on a square grid in
    the tangent plane over ``target_mm``, projected onto the skin, and lifted by
    ``standoff_mm`` along the local surface normal. Each site is triaxial, with
    its third axis radial (normal to the skin) and two tangential.
    """
    p = np.asarray(target_mm, float).reshape(1, 3)
    closest, _, tri = trimesh.proximity.closest_point(skin, p)
    n = np.asarray(skin.face_normals[tri[0]], float)
    if np.dot(n, closest[0] - np.asarray(skin.centroid)) < 0:
        n = -n
    up = np.array([0.0, 0.0, 1.0])
    e1 = up - np.dot(up, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    k = int(np.ceil(np.sqrt(n_site)))
    g = (np.arange(k) - (k - 1) / 2.0) * (span_mm / max(k - 1, 1))
    uu, vv = np.meshgrid(g, g, indexing="ij")
    seeds = closest[0] + np.outer(uu.ravel(), e1) + np.outer(vv.ravel(), e2)
    seeds = seeds[:n_site] + 200.0 * n            # start well outside, project in
    hit, _, tri2 = trimesh.proximity.closest_point(skin, seeds)
    nrm = np.asarray(skin.face_normals)[tri2]
    flip = np.einsum("ij,ij->i", nrm, hit - np.asarray(skin.centroid)) < 0
    nrm[flip] *= -1
    pos = hit + standoff_mm * nrm
    frame = np.empty((len(pos), 3, 3))
    for i, r in enumerate(nrm):
        a = np.array([0.0, 0.0, 1.0])
        t1 = a - np.dot(a, r) * r
        if np.linalg.norm(t1) < 1e-9:
            t1 = np.cross(r, [1.0, 0.0, 0.0])
        t1 /= np.linalg.norm(t1)
        frame[i] = np.stack([t1, np.cross(r, t1), r])
    logger.info("[chest] conformal array: %d sites over %.0f mm, standoff %.1f mm",
                len(pos), span_mm, standoff_mm)
    return ChestArray(site_pos_mm=pos, site_frame=frame, label=f"conformal-{n_site}")


def load_real_chest_array(mat_path: Path, frame: int = 0) -> ChestArray:
    """The measured array, in its own OptiTrack frame (mm)."""
    from scipy.io import loadmat

    d = loadmat(str(mat_path))
    pos = d["chest_pos"][:, :, frame] * 1000.0
    ori = d["chest_ori"][:, :, :, frame]
    _, ia = np.unique(np.round(pos, 4), axis=0, return_index=True)
    sel = np.sort(ia)
    c = pos[sel] - pos[sel].mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    logger.info("[chest] %d sites from %d channels; plate %.0f x %.0f mm, "
                "sag %.2f mm (flat), spacing %.1f mm",
                len(sel), len(pos), sv[0], sv[1], sv[2],
                float(np.mean(np.sort(np.linalg.norm(
                    pos[sel][:, None] - pos[sel][None], axis=-1)
                    + np.eye(len(sel)) * 1e9, axis=1)[:, 0])))
    return ChestArray(site_pos_mm=pos[sel], site_frame=ori[sel])


#: The pilot's hand-picked chest fiducials on the subject's torso STL, in the
#: AREA-WEIGHTED-CENTRED torso frame, metres. Copied verbatim from
#: ``provenance/pilot_code/CFA_HEP_ChestInversion.m`` (``sensor_pts_chest``).
#: The comments are the pilot's own labels for which plate site each one is.
CFA_CHEST_FIDUCIALS_M = np.array([
    [-0.023, 0.135,  0.092],    # second from right top   -> plate site 7
    [ 0.063, 0.120,  0.024],    # second from bottom left -> plate site 2
    [-0.056, 0.130, -0.018],    # bottom right            -> plate site 6
])
#: Which measured plate sites those fiducials are, 0-based. The pilot's own
#: selection was ``heartGradY_pos([7,2,6],:)`` over the radial channels.
CFA_FIDUCIAL_SITES = (6, 1, 5)


def placed_from_cfa(array: "ChestArray", pack_heart_centroid_mm: np.ndarray,
                    pack_torso_centroid_mm: np.ndarray,
                    skin: trimesh.Trimesh, heart_centroid_mm: np.ndarray, *,
                    standoff_mm: float = 6.5) -> tuple["ChestArray", float]:
    """Place the measured plate at the pose the CFA pilot used, on this torso.

    The pilot fixed the array by mapping three measured plate sites onto three
    hand-picked fiducials on the subject's torso STL. Those fiducials are the
    only surviving record of where the plate actually sat, so this reproduces
    that pose **relative to the heart** -- which is the part that transfers
    between two different torsos -- rather than its absolute coordinates.

    Two honest caveats, both measured rather than assumed:

    * the fiducials are hand-picked to about 16 mm (no triple of measured sites
      reproduces their triangle better), so the pose carries that error;
    * the pack's subject torso is a different shape from this BodyParts3D one,
      so the transferred pose puts 6 of 8 sites INSIDE this body. The plate is
      therefore slid outward along its own normal by the minimum distance that
      clears the skin at ``standoff_mm``. The slide is returned so it can be
      reported; it is a real departure from the CFA pose, not a free parameter.

    Returns the placed array and the slide distance in mm.
    """
    pos = array.site_pos_mm
    fid_pack = CFA_CHEST_FIDUCIALS_M * 1000.0 + pack_torso_centroid_mm
    a = pos[list(CFA_FIDUCIAL_SITES)]
    ca, cb = a.mean(0), fid_pack.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (fid_pack - cb))
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    trans = cb - ca @ rot
    # pack frame -> heart-anchored -> this torso's heart
    placed = (pos @ rot + trans) - pack_heart_centroid_mm + heart_centroid_mm
    frame = np.einsum("ij,ski->skj", rot, array.site_frame)

    c = placed.mean(0)
    normal = np.linalg.svd(placed - c, full_matrices=False)[2][2]
    normal = normal / np.linalg.norm(normal)
    if normal @ (c - heart_centroid_mm) < 0:
        normal = -normal
    pq = trimesh.proximity.ProximityQuery(skin)
    lo, hi = 0.0, 300.0
    for _ in range(60):                       # smallest slide that clears the skin
        mid = 0.5 * (lo + hi)
        if pq.signed_distance(placed + mid * normal).max() > -standoff_mm:
            lo = mid
        else:
            hi = mid
    return ChestArray(site_pos_mm=placed + hi * normal, site_frame=frame,
                      label="real-8-cfa-pose"), float(hi)
