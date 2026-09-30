"""Fit a rigid (optionally scaled) transform between two surfaces.

The atlas meshes live in the BodyParts3D frame; a head-mounted OPM array lives in
whatever frame the sensor model uses (for the CFA work, MNI mm). Something has to
carry one into the other. The existing CFA pack does it with a hand-authored 4x4
(``torso2mni.txt``) that is not rigid — it folds in a ~1000x scale and shear, and
nothing checks it. Here the transform is *fitted*, from surface to surface, and
returns its own residual so the fit can be asserted rather than trusted.

Fit the HEAD, not the whole body. The sensors are head-mounted and the target
frame is defined by the head, so head-to-head is the correspondence that is
actually meaningful; the torso then lands wherever the subject's anatomy puts it
instead of being stretched to match a template.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

__all__ = ["FitResult", "fit_surface", "apply", "residual_mm"]


class FitResult:
    """A fitted transform plus the evidence that it is a good one."""

    def __init__(self, matrix: np.ndarray, rms_mm: float, max_mm: float,
                 scale: float, n_samples: int):
        self.matrix = np.asarray(matrix, dtype=float)
        self.rms_mm = float(rms_mm)
        self.max_mm = float(max_mm)
        self.scale = float(scale)
        self.n_samples = int(n_samples)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3] / self.scale

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def __repr__(self) -> str:
        return (f"FitResult(rms={self.rms_mm:.2f} mm, max={self.max_mm:.2f} mm, "
                f"scale={self.scale:.4f}, n={self.n_samples})")


def residual_mm(source: trimesh.Trimesh, target: trimesh.Trimesh,
                matrix: np.ndarray, *, n_samples: int = 4000,
                seed: int = 0) -> tuple[float, float]:
    """(rms, max) distance from transformed ``source`` samples to ``target``."""
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(source, n_samples, seed=int(rng.integers(1 << 31)))
    pts = trimesh.transform_points(pts, matrix)
    d = trimesh.proximity.closest_point(target, pts)[1]
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(d))


def fit_surface(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    scale: bool = True,
    samples: int = 2000,
    icp_first: int = 20,
    icp_final: int = 100,
    seed: int = 0,
) -> FitResult:
    """Fit ``source`` onto ``target``; returns the 4x4 and its residual.

    Reflections are disallowed — a mirrored body is anatomically wrong however
    well it fits, and left/right matters for a cardiac source.
    """
    matrix, _ = trimesh.registration.mesh_other(
        source, target, samples=samples, scale=scale,
        icp_first=icp_first, icp_final=icp_final, reflection=False, seed=seed,
    )
    s = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    if np.linalg.det(matrix[:3, :3]) <= 0:
        raise ValueError("fitted transform is a reflection (det <= 0)")
    rms, mx = residual_mm(source, target, matrix, seed=seed)
    out = FitResult(matrix, rms, mx, s, samples)
    logger.info("[register] %s", out)
    return out


def apply(m: trimesh.Trimesh, matrix: np.ndarray) -> trimesh.Trimesh:
    """Copy of ``m`` with ``matrix`` applied (does not mutate the input)."""
    out = m.copy()
    out.apply_transform(np.asarray(matrix, dtype=float))
    return out
