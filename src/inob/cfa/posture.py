"""Head-on-neck rotation for the fused head+thorax conductor.

The replication pack rotates only the sensor array (``cfa_rotate_grad``) against
a torso-only BEM. That is self-consistent there precisely *because* the head is
not in the volume conductor: nothing in the model has to bend. Once head and
thorax are one continuous conductor it stops being self-consistent -- turning the
array without turning the head would slide 85 sensors through the scalp.

So the rotation is applied to the anatomy as well, as a **graded twist about a
cervical pivot**:

    s(z) = 0                    below ``z_bot``   (thorax: fixed)
    s(z) = smoothstep           between          (neck: shares the rotation)
    s(z) = 1                    above ``z_top``   (head: rigid, with the array)

and a point at height z is rotated about the pivot by the geodesic fraction
``s(z)`` of the full rotation -- i.e. by ``exp(s * log R)``, not by a linear blend
of matrices, so every intermediate transform is itself a rotation.

**The share is not a function of height alone**, and it cannot be. The chin sits
at z ~ -110 mm and the mandible's centroid at ~ -95 mm, well inside the blend
band, so a purely height-based twist turns the jaw and the face by only ~75% of
what the cranium does -- at 100 deg of yaw that shears the face off the skull by
25 deg. So the share is

    s(p) = max( smoothstep_in_z(p_z),  head_membership(p) )

where ``head_membership`` is 1 inside the **head block** -- the convex hull of the
skull and brain, which is what actually rides on the atlas -- and falls off
smoothly over ``head_falloff_mm`` outside it, to keep s continuous. Everything
cranial therefore turns rigidly whatever its height, the neck grades, and the
thorax stays put.

Two tissue rules, and the difference is anatomy, not convenience:

* **Soft tissue** (skin, muscle, cord, and any organ crossing the band) is
  deformed **per vertex**. Its cross-sections genuinely shear when the neck
  turns.
* **Bone** is deformed **per connected body**, each vertebra rotating rigidly by
  ``s`` at its own centroid. Cervical rotation really is distributed over C1-C7
  as a stack of small rigid rotations; a per-vertex twist would instead shear
  individual vertebrae, which do not shear.

Head-internal structures (brain, insula, skull vault) and the sensor array sit
above ``z_top`` and so move rigidly as one body, which is what keeps the insula
leadfield expressible in a head-fixed frame.

.. warning::
   The design's movement=high level reaches **100 deg of yaw**, and physiological
   cervical axial rotation is about 80-90 deg. At the extremes this is an
   extrapolated posture, retained for comparability with the published 2x2
   numbers. :func:`physiological_fraction` reports how far past the limit each
   orientation sits, and the build writes it into the run summary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

__all__ = ["NeckModel", "rotation_matrix", "physiological_fraction"]

#: Physiological single-side limits for the cervical spine, degrees
#: (White & Panjabi, *Clinical Biomechanics of the Spine*): axial rotation,
#: flexion/extension, lateral bending.
CERVICAL_LIMIT_DEG: tuple[float, float, float] = (90.0, 70.0, 45.0)


def rotation_matrix(ypr_deg) -> np.ndarray:
    """World-frame rotation for [yaw, pitch, roll] in degrees, as ``Rz @ Ry @ Rx``.

    Matches ``cfa_rotate_grad`` exactly so a FEM run and the pack's BEM run can be
    compared orientation by orientation.
    """
    y, p, r = np.deg2rad(np.asarray(ypr_deg, float).ravel())
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def physiological_fraction(ypr_deg) -> float:
    """Largest per-axis excursion as a fraction of the cervical range of motion.

    1.0 is exactly at the limit; >1 is extrapolated posture. Reported rather than
    enforced -- the design chooses its own angles, this only makes the cost of
    that choice visible.
    """
    a = np.abs(np.asarray(ypr_deg, float).ravel())
    return float(np.max(a / np.asarray(CERVICAL_LIMIT_DEG)))


@dataclass(frozen=True)
class NeckModel:
    """Pivot, blend band and head block for the head-on-neck twist, in mm."""

    pivot: np.ndarray
    z_bot: float
    z_top: float
    #: ``(n, 4)`` half-space equations of the head block's convex hull, as
    #: scipy's ``ConvexHull.equations``: a point is inside where every
    #: ``a . p + d <= 0``. None disables the anatomical override.
    head_hull: np.ndarray | None = None
    #: Distance over which head membership falls from 1 to 0 outside the hull.
    head_falloff_mm: float = 20.0

    def __post_init__(self) -> None:
        if not self.z_top > self.z_bot:
            raise ValueError(f"z_top ({self.z_top}) must exceed z_bot ({self.z_bot})")
        object.__setattr__(self, "pivot", np.asarray(self.pivot, float).reshape(3))
        if self.head_hull is not None:
            object.__setattr__(self, "head_hull",
                               np.asarray(self.head_hull, float).reshape(-1, 4))

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_anatomy(
        cls,
        *,
        cord: trimesh.Trimesh,
        skull_base_z: float,
        t1_z: float,
        pivot_z: float | None = None,
        head_block: trimesh.Trimesh | None = None,
        head_falloff_mm: float = 20.0,
    ) -> "NeckModel":
        """Pivot on the cord axis; blend band from T1 up to the skull base.

        The pivot is placed on the **spinal cord's own axis** at the height of the
        atlanto-axial joint rather than at a fixed offset below the array
        centroid, because the head does not pivot about its centre: it pivots
        about the top of the spine, so a turn translates the array relative to the
        thorax as well as reorienting it. Getting that translation wrong directly
        understates (or overstates) the movement factor.
        """
        z_top = float(skull_base_z)
        z_bot = float(t1_z)
        pz = float(pivot_z) if pivot_z is not None else z_top - 25.0
        v = np.asarray(cord.vertices, float)
        band = v[np.abs(v[:, 2] - pz) < 15.0]
        if len(band) < 8:                       # cord may not reach that high
            band = v[v[:, 2] > np.quantile(v[:, 2], 0.9)]
        pivot = np.array([band[:, 0].mean(), band[:, 1].mean(), pz])
        hull = None
        if head_block is not None:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.asarray(head_block.vertices, float)).equations
            logger.info("[neck] head block: %d hull faces, reaches down to z=%.0f mm",
                        len(hull), float(head_block.bounds[0, 2]))
        logger.info(
            "[neck] pivot %s mm, blend %.0f -> %.0f mm (%.0f mm of neck)",
            np.round(pivot, 1).tolist(), z_bot, z_top, z_top - z_bot,
        )
        return cls(pivot=pivot, z_bot=z_bot, z_top=z_top, head_hull=hull,
                   head_falloff_mm=head_falloff_mm)

    # -- the twist ------------------------------------------------------------
    @staticmethod
    def _smoothstep(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def head_membership(self, pts: np.ndarray) -> np.ndarray:
        """1 inside the head block, falling smoothly to 0 ``head_falloff_mm`` out."""
        if self.head_hull is None:
            return np.zeros(len(np.atleast_2d(pts)))
        p = np.asarray(pts, float).reshape(-1, 3)
        d = (self.head_hull[:, :3] @ p.T + self.head_hull[:, 3:4]).max(0)   # <0 inside
        # 1 inside and ON the hull, decaying to 0 at head_falloff_mm OUTSIDE it.
        # Measuring inward from the surface instead would give the hull's own
        # vertices -- the mandible, the occiput -- a share of 0 and shear the
        # face off the skull, which is the very failure this exists to prevent.
        return self._smoothstep((self.head_falloff_mm - d) / self.head_falloff_mm)

    def weight(self, pts_or_z: np.ndarray) -> np.ndarray:
        """Share of the rotation at each point.

        Accepts ``(n, 3)`` points -- which is what enables the anatomical head
        override -- or a bare ``(n,)`` of heights for the height-only rule.
        """
        a = np.asarray(pts_or_z, float)
        if a.ndim == 2 and a.shape[1] == 3:
            z, pts = a[:, 2], a
        else:
            z, pts = a.ravel(), None
        w = self._smoothstep((z - self.z_bot) / (self.z_top - self.z_bot))
        if pts is not None and self.head_hull is not None:
            w = np.maximum(w, self.head_membership(pts))
        return w

    def _graded(self, rotvec: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Rotation matrices ``exp(s_i * rotvec)``, one per element of ``s``."""
        return Rotation.from_rotvec(np.outer(np.asarray(s, float).ravel(), rotvec)).as_matrix()

    def deform_points(self, pts: np.ndarray, ypr_deg, *, weights=None) -> np.ndarray:
        """Twist points about the pivot, each by its own share of the rotation.

        ``weights`` overrides the height-derived share -- pass ``1.0`` for a
        head-fixed body (the array, the brain) so it moves rigidly.
        """
        p = np.asarray(pts, float).reshape(-1, 3)
        rotvec = Rotation.from_matrix(rotation_matrix(ypr_deg)).as_rotvec()
        if np.allclose(rotvec, 0.0):
            return p.copy()
        s = self.weight(p) if weights is None else np.broadcast_to(
            np.asarray(weights, float), (len(p),))
        rot = self._graded(rotvec, s)                       # (n, 3, 3)
        return np.einsum("nij,nj->ni", rot, p - self.pivot) + self.pivot

    def deform_directions(self, dirs: np.ndarray, at: np.ndarray, ypr_deg,
                          *, weights=None) -> np.ndarray:
        """Rotate unit directions by the share of the rotation at their location."""
        d = np.asarray(dirs, float).reshape(-1, 3)
        a = np.asarray(at, float).reshape(-1, 3)
        rotvec = Rotation.from_matrix(rotation_matrix(ypr_deg)).as_rotvec()
        if np.allclose(rotvec, 0.0):
            return d.copy()
        s = self.weight(a) if weights is None else np.broadcast_to(
            np.asarray(weights, float), (len(d),))
        out = np.einsum("nij,nj->ni", self._graded(rotvec, s), d)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def deform_mesh(self, mesh: trimesh.Trimesh, ypr_deg, *,
                    per_body: bool = False, rigid: bool = False) -> trimesh.Trimesh:
        """Deformed copy of ``mesh``.

        ``rigid`` moves the whole mesh as a head-fixed body; ``per_body`` rotates
        each connected component rigidly by the share at its own centroid (bone);
        otherwise the twist is applied per vertex (soft tissue).
        """
        out = mesh.copy()
        v = np.asarray(out.vertices, float)
        if rigid:
            out.vertices = self.deform_points(v, ypr_deg, weights=1.0)
            return out
        if not per_body:
            out.vertices = self.deform_points(v, ypr_deg)
            return out
        new = v.copy()
        for comp in trimesh.graph.connected_components(out.edges, nodes=np.arange(len(v))):
            s = float(self.weight(v[comp].mean(0, keepdims=True))[0])
            new[comp] = self.deform_points(v[comp], ypr_deg, weights=s)
        out.vertices = new
        return out

    def summary(self, ypr_deg) -> dict:
        return {
            "ypr_deg": np.asarray(ypr_deg, float).round(2).tolist(),
            "pivot_mm": self.pivot.round(1).tolist(),
            "blend_mm": [self.z_bot, self.z_top],
            "physiological_fraction": round(physiological_fraction(ypr_deg), 3),
        }
