"""The triaxial OPM array for the CFA study.

Every sensor site carries **three** orthogonal magnetometers -- radial plus two
tangential -- so the 85 helmet positions give 255 channels. That is what current
and next-generation OPM helmets actually are, and it matters here for two
reasons beyond channel count:

* **The precision factor is a per-channel axis tilt.** With a triaxial site the
  three nominal axes span the full field vector, so a tilted axis is still a
  linear combination of what the site measures -- the mismatch is a rotation
  within a known subspace rather than a projection onto something unmeasured.
* **The harmonic (HFC/AMM) basis gets three components per site.** The shielding
  factor those methods reach is set by how well the array samples the vector
  field, not just by how many sites it has.

Channel order follows :func:`inob.sensors.build_triaxial`: all N radial, then all
N tangent-1, then all N tangent-2, so channel ``i``, ``i+N``, ``i+2N`` share a
position. :meth:`CFAArray.site_of` makes that mapping explicit rather than
implied by arithmetic scattered through the engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inob.sensors.triaxial import build_triaxial

logger = logging.getLogger(__name__)

__all__ = ["CFAArray"]


@dataclass
class CFAArray:
    """Triaxial array: ``n_site`` positions, ``3 * n_site`` channels."""

    site_pos_mm: np.ndarray          # (S, 3)
    site_normal: np.ndarray          # (S, 3) outward unit
    chanori: np.ndarray              # (3S, 3) nominal sensitive axes
    labels: tuple[str, ...]

    @property
    def n_site(self) -> int:
        return len(self.site_pos_mm)

    @property
    def n_chan(self) -> int:
        return len(self.chanori)

    @property
    def chanpos_mm(self) -> np.ndarray:
        """(3S, 3) -- each site repeated once per axis."""
        return np.tile(self.site_pos_mm, (3, 1))

    def site_of(self, chan: np.ndarray | None = None) -> np.ndarray:
        """Site index for each channel."""
        c = np.arange(self.n_chan) if chan is None else np.asarray(chan)
        return c % self.n_site

    @classmethod
    def from_reference(cls, npz_path: Path) -> "CFAArray":
        """Build from the study's reference helmet positions.

        The reference array is single-axis: 85 positions with 85 radial
        sensitive axes. The positions and the radial direction are the physical
        helmet; the two tangential axes are added here, so this is the same
        helmet read out triaxially rather than a different array.
        """
        ref = np.load(npz_path)
        pos = np.asarray(ref["chanpos"], float) * 1000.0
        nrm = np.asarray(ref["chanori"], float)
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        # Orient outward from the array centroid, so "radial" means the same
        # thing at every site regardless of how the reference stored its signs.
        outward = pos - pos.mean(0)
        flip = np.einsum("ij,ij->i", nrm, outward) < 0
        nrm[flip] *= -1.0
        arr = build_triaxial(pos, nrm)
        logger.info("[array] %d sites x 3 axes = %d channels; %d radial axes flipped "
                    "to outward", len(pos), len(arr.coilori), int(flip.sum()))
        return cls(site_pos_mm=pos, site_normal=nrm,
                   chanori=np.asarray(arr.coilori, float), labels=tuple(arr.labels))

    # -- geometry -------------------------------------------------------------
    def transformed(self, fn) -> "CFAArray":
        """Copy with ``fn(points, directions_at_points)`` applied (posture)."""
        pos, nrm = fn(self.site_pos_mm, self.site_normal)
        ori = fn(self.chanpos_mm, self.chanori)[1]
        return CFAArray(site_pos_mm=pos, site_normal=nrm, chanori=ori,
                        labels=self.labels)

    # -- readout --------------------------------------------------------------
    def project(self, field: np.ndarray, ori: np.ndarray | None = None) -> np.ndarray:
        """Channel readings from a per-site field vector.

        ``field`` is ``(S, 3, K)`` -- the world-frame field vector at each site
        for each of K source components. Returns ``(3S, K)``. ``ori`` defaults to
        the nominal axes; pass a perturbed ``(3S, 3)`` to read the array out as
        it physically is.
        """
        f = np.asarray(field, float)
        if f.ndim != 3 or f.shape[0] != self.n_site or f.shape[1] != 3:
            raise ValueError(f"field must be (n_site, 3, K); got {f.shape}")
        o = self.chanori if ori is None else np.asarray(ori, float)
        if o.shape != (self.n_chan, 3):
            raise ValueError(f"ori must be ({self.n_chan}, 3); got {o.shape}")
        return np.einsum("sck,sc->sk", f[self.site_of()], o)
