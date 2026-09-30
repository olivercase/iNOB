"""Cardiac and cortical source models on the iNOB anatomy.

**The cardiac source is distributed over the myocardial wall.** The replication
pack generates the artefact from two point dipoles and then hand-sets a 10.2 mm
position error to give the joint GLM something to be wrong about. Here the
mismatch has a cause instead: the truth is a current-source density spread over
the anatomical myocardium with physiological (wall-normal) orientations, while
every estimator still models it as the same two equivalent dipoles the pack used.
What the estimator cannot fit is then the source's real spatial extent -- its
quadrupole and higher content -- rather than an arbitrary offset.

The distributed source is anchored to measured data at the dipole level. Site
amplitudes are the minimum-norm solution of

    sum_k  n_k a_k(t)  =  q(t)

where ``n_k`` is the wall normal at site k and ``q(t)`` is the *measured* cardiac
dipole moment from the replication pack's CDM estimate. So the net equivalent
dipole is exactly the real one, at every time sample, and everything the model
adds is higher-order structure the anatomy puts there.

.. note::
   The CDM time course is the one input here that is not iNOB anatomy: it is a
   real recorded cardiac moment and there is no atlas substitute for it. Geometry
   -- where the sources sit, which way they point, how they are weighted -- is
   entirely BodyParts3D.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

__all__ = ["MyocardialSource", "CorticalSource", "load_cdm", "equivalent_dipoles"]


# ---------------------------------------------------------------------------
# measured drive
# ---------------------------------------------------------------------------
#: The stored CDM is fit to fT-native OP-MEG data, so it carries an implicit
#: 1e-15: ``L[T/(A.m)] @ cdm`` reads directly in fT. That convention is fine as
#: long as nothing else in the chain has units, but a FEM leadfield calibrated
#: against an analytic sphere very much does. So it is made explicit here: the
#: moment is converted to SI (A.m) on load, leadfields stay in T/(A.m), and the
#: single fT conversion happens once, at the end, in the engine.
_CDM_TO_SI: float = 1e-15


def load_cdm(path: Path) -> dict:
    """Load the measured cardiac dipole moment and HEF timebase, in SI.

    Returns ``q`` (3, T) in A.m -- the *total* measured moment, i.e. the sum of
    the pack's two equivalent dipoles -- along with the per-dipole moments, the
    time axis and the HEF latency index.
    """
    from scipy.io import loadmat

    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    cdm6 = np.asarray(m["cdm_resampled"], float) * _CDM_TO_SI   # (6, T) A.m: [q1; q2]
    time = np.asarray(m["time"], float).ravel()
    latency = float(np.asarray(m["erp_latency"]).ravel()[0])
    fs = 1.0 / float(time[1] - time[0])
    out = {
        "q6": cdm6,
        "q": cdm6[:3] + cdm6[3:],
        "time": time,
        "fs": fs,
        "erp_latency": latency,
        "erp_index": int(round(latency * fs)),
    }
    logger.info(
        "[cdm] %d samples at %.0f Hz, HEF latency %.0f ms (index %d), "
        "|q| peak %.3g uA.m",
        len(time), fs, 1e3 * latency, out["erp_index"],
        1e6 * np.abs(np.linalg.norm(out["q"], axis=0)).max(),
    )
    return out


# ---------------------------------------------------------------------------
# distributed cardiac source
# ---------------------------------------------------------------------------
@dataclass
class MyocardialSource:
    """Current dipoles distributed through the myocardial wall.

    Attributes
    ----------
    pos_mm : (N, 3) site positions, pushed inside the wall so no dipole sits on
        a conductivity interface (DUNEuro's source models need an interior
        element, and a dipole exactly on a face is not well posed).
    ori : (N, 3) unit wall normals, inward-positive. Depolarisation crosses the
        wall, so the normal is the physiological orientation.
    area_mm2 : (N,) the surface area each site stands for, used to weight the
        minimum-norm fit so a densely-triangulated region does not attract more
        of the moment than its size warrants.
    """

    pos_mm: np.ndarray
    ori: np.ndarray
    area_mm2: np.ndarray

    def __post_init__(self) -> None:
        self.pos_mm = np.asarray(self.pos_mm, float).reshape(-1, 3)
        self.ori = np.asarray(self.ori, float).reshape(-1, 3)
        self.ori /= np.linalg.norm(self.ori, axis=1, keepdims=True)
        self.area_mm2 = np.asarray(self.area_mm2, float).ravel()
        if not len(self.pos_mm) == len(self.ori) == len(self.area_mm2):
            raise ValueError("pos/ori/area length mismatch")

    def __len__(self) -> int:
        return len(self.pos_mm)

    @classmethod
    def from_wall(
        cls,
        myocardium: trimesh.Trimesh,
        *,
        n_sites: int = 400,
        inset_mm: float = 3.0,
        seed: int = 0,
    ) -> "MyocardialSource":
        """Area-weighted Poisson-ish sample of the myocardial wall.

        Sites are drawn by area so the sampling is uniform over the wall, then
        moved ``inset_mm`` along the inward normal to sit strictly inside the
        heart compartment.
        """
        pts, face_idx = trimesh.sample.sample_surface(myocardium, n_sites, seed=seed)
        nrm = np.asarray(myocardium.face_normals)[face_idx]
        centre = np.asarray(myocardium.centroid, float)
        # Orient every normal inward, so "positive amplitude" means the same
        # thing at every site regardless of the atlas mesh's winding.
        inward = np.sign(np.einsum("ij,ij->i", centre - pts, nrm))[:, None]
        inward[inward == 0] = 1.0
        nrm = nrm * inward
        pos = pts + inset_mm * nrm
        area = np.full(n_sites, float(myocardium.area) / n_sites)
        logger.info(
            "[myocardium] %d sites over %.0f cm^2 of wall, inset %.1f mm; "
            "extent %s mm", n_sites, myocardium.area / 100.0, inset_mm,
            np.round(pos.max(0) - pos.min(0), 1).tolist(),
        )
        return cls(pos_mm=pos, ori=nrm, area_mm2=area)

    # -- driving the source ---------------------------------------------------
    def amplitudes(self, q: np.ndarray, *, ridge: float = 1e-9) -> np.ndarray:
        """Minimum-norm site amplitudes reproducing net moment ``q`` (3, T).

        Solves ``N W a = q`` for a, where ``N`` stacks the wall normals and ``W``
        is the area weighting, under the minimum weighted-norm constraint. The
        constraint is satisfied to machine precision -- this is an exactly
        determined 3-row underdetermined system, not a fit -- so the distributed
        source has the measured dipole moment at every sample.
        """
        q = np.asarray(q, float).reshape(3, -1)
        w = self.area_mm2 / self.area_mm2.mean()
        nw = (self.ori * w[:, None]).T                       # (3, N)
        gram = nw @ nw.T                                     # (3, 3)
        gram = gram + ridge * np.trace(gram) / 3.0 * np.eye(3)
        return (w[:, None] * (nw.T @ np.linalg.solve(gram, q)))  # (N, T)

    def net_moment(self, amps: np.ndarray) -> np.ndarray:
        """Net dipole moment (3, T) of site amplitudes (N, T) -- the round trip."""
        return self.ori.T @ np.asarray(amps, float)

    def centre_of_moment(self, amps_t: np.ndarray) -> np.ndarray:
        """Amplitude-weighted centroid at one time sample, mm."""
        w = np.abs(np.asarray(amps_t, float).ravel())
        return (self.pos_mm * w[:, None]).sum(0) / max(w.sum(), 1e-30)

    def transformed(self, fn) -> "MyocardialSource":
        """Copy with ``fn`` applied to positions and orientations (posture)."""
        pos, ori = fn(self.pos_mm, self.ori)
        return MyocardialSource(pos_mm=pos, ori=ori, area_mm2=self.area_mm2)


def equivalent_dipoles(
    src: MyocardialSource, q: np.ndarray, *, offset_mm=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """The two nominal dipole positions the ESTIMATOR models the heart with, mm.

    Not a fit to the distributed source: this is the model an experimenter would
    build from a registered template. It is placed by geometry -- the myocardial
    centroid, and a second site displaced along the wall's principal axis, which
    is the base-to-apex direction -- and then displaced by ``offset_mm`` to carry
    the residual registration error of that template.
    """
    p = src.pos_mm
    c = p.mean(0)
    axis = np.linalg.svd(p - c, full_matrices=False)[2][0]
    half = 0.25 * float(np.ptp((p - c) @ axis))
    pair = np.vstack([c - half * axis, c + half * axis]) + np.asarray(offset_mm, float)
    logger.info(
        "[cardiac model] 2 equivalent dipoles %.0f mm apart along the base-apex "
        "axis, offset %.1f mm: %s",
        2 * half, float(np.linalg.norm(offset_mm)), np.round(pair, 1).tolist(),
    )
    return pair


# ---------------------------------------------------------------------------
# cortical (HEF) source
# ---------------------------------------------------------------------------
@dataclass
class CorticalSource:
    """The heart-evoked field generator: the anatomical right insula.

    The pack puts a point dipole at MNI [35, -11, 6] (Deen et al. 2011). Here the
    position comes from the atlas structure itself (FMA72977), so it is in the
    same frame as the conductor by construction. Still 3-dof at one location --
    the HEF's spatial extent is not what this study is testing, and keeping it a
    point keeps the joint GLM's insula basis a clean 3-column span.
    """

    pos_mm: np.ndarray
    label: str = "insula_right"

    @classmethod
    def from_mesh(cls, insula: trimesh.Trimesh, *, label: str = "insula_right") -> "CorticalSource":
        c = np.asarray(insula.centroid, float)
        logger.info("[%s] centroid %s mm, volume %.1f cm^3",
                    label, np.round(c, 1).tolist(), insula.volume / 1000.0)
        return cls(pos_mm=c, label=label)
