"""Build a head+thorax volume conductor for cardiac-field work, in sensor space.

What this fixes, relative to the CFA replication pack's model:

* **One atlas.** Skin, bone, heart and lungs all come from BodyParts3D 4.3, so
  they are mutually registered by construction. The pack instead glues a
  ``torso_tools`` heart/lung pair to a separate ``thorax.gii`` body with a
  hand-authored 4x4 (``torso2mni.txt``) that nothing validates.
* **A fitted transform.** One rigid fit of the atlas head onto the MNI scalp,
  reporting its own residual (see :mod:`inob.geometry.register`). No scale, so
  organ sizes stay those of the real subject.
* **A continuous conductor.** The pack's body boundary stops at the neck
  (z = -215 mm) while the sensors sit at z = -46..118 mm — the array is 20+ cm
  outside its own volume conductor. Here the skin runs unbroken from the vertex
  past the heart, so the sensors are where they belong: just outside the body.
* **Bone.** Ribs, sternum, clavicles, scapulae and spine are included. At
  0.0042 S/m against 0.43 S/m for skin, bone is where the cardiac return current
  does not go, and the pack's three-compartment BEM has none of it.

Every cut is capped (:mod:`inob.mesh.clip`), so each compartment stays a closed
surface that a tetrahedral mesher will accept.
"""
from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from inob.config import ShrinkwrapParams
from inob.geometry.builder import watertighten
from inob.geometry.composites import default_manifest_path, load_composite
from inob.mesh.clip import clip_band, clip_capped, report
from inob.mesh.solidify import solidify_shell

logger = logging.getLogger(__name__)

__all__ = ["ThoraxSpec", "build_thorax", "CONDUCTIVITY_SM"]

#: Conductivities (S/m) for the compartments built here. Blood and myocardium
#: follow the pack's torso BEM (0.62 blood); the rest are the iNOB defaults.
CONDUCTIVITY_SM: dict[str, float] = {
    # One homogeneous heart at the intracavitary-blood value, which is the
    # standard cardiac-forward-modelling simplification (Klepfer et al. 1997):
    # the chambers dominate the organ's conduction, and resolving the ~10 mm
    # myocardial wall as a separate shell buys little at head-array distance.
    # The myocardial wall geometry (FMA7101) is still loaded -- but as the
    # SOURCE substrate, not as a conductivity compartment. See inob.cfa.sources.
    "heart": 0.62,
    "brain": 0.33,
    "lung_left": 0.05,
    "lung_right": 0.05,
    "bone": 0.0042,
    "spinal_cord": 0.23,
    "skin": 0.23,
}


#: Per-compartment voxel-shrinkwrap parameters.
#:
#: These are not cosmetic. Several BodyParts3D structures are *shells*, not
#: solids: FMA7163 "Skin" is the skin organ, a ~2 mm-thick closed sheet whose
#: enclosed volume is 3.4 L for a whole body (1.8 m^2 x 2 mm) rather than the
#: ~70 L of a person. Clipping such a shell yields a hollow surface that is
#: watertight and useless as a conductor. The voxel shrinkwrap is what turns it
#: into a filled body, so the pitch has to be coarse enough to bridge the wall
#: (skin) and the airways (lung) rather than reproduce them.
SHRINKWRAP_DEFAULTS: dict[str, ShrinkwrapParams] = {
    "_default":         ShrinkwrapParams(pitch=2.0, n_samples=400_000, close_iter=2,
                                         decimate_target=25_000, smooth_iter=8),
    # matches configs/default.yaml mesh_skin
    "mesh_skin":        ShrinkwrapParams(pitch=3.0, n_samples=600_000, close_iter=2,
                                         decimate_target=30_000, smooth_iter=8),
    "mesh_heart":       ShrinkwrapParams(pitch=2.0, n_samples=400_000, close_iter=3,
                                         decimate_target=25_000, smooth_iter=8),
    # 89 neuraxis components (hemispheres, cerebellum, brainstem) that overlap
    # and abut. A 2 mm pitch bridges the sulci into one conducting volume, which
    # is what a FEM compartment wants -- resolving every sulcus would produce a
    # genus-hundreds surface no tetra mesher will accept.
    "mesh_brain":       ShrinkwrapParams(pitch=2.0, n_samples=600_000, close_iter=3,
                                         decimate_target=30_000, smooth_iter=10),
    # lungs are 245-318 parts riddled with bronchi and vessels; a fine pitch
    # reproduces every airway as a handle (euler ~ -4800) and under-fills the
    # parenchyma. 3 mm + close_iter 3 fills the lobe.
    "mesh_lung_left":   ShrinkwrapParams(pitch=3.0, n_samples=500_000, close_iter=3,
                                         decimate_target=25_000, smooth_iter=10),
    "mesh_lung_right":  ShrinkwrapParams(pitch=3.0, n_samples=500_000, close_iter=3,
                                         decimate_target=25_000, smooth_iter=10),
}


@dataclass
class ThoraxSpec:
    """What to build, in the *target* (MNI mm) frame after transform."""

    project_root: Path
    transform: np.ndarray
    #: Inferior clip of the body, mm. Well below the lung bases so the artificial
    #: cap is far from the cardiac source; the field there is negligible.
    z_lo: float = -600.0
    #: Superior clip, or None to keep the head whole (the usual case).
    z_hi: float | None = None
    #: Half-width for arm removal, mm. The atlas is in a T-pose, which no seated
    #: subject is; truncating at the shoulder is closer to the recorded posture
    #: than outstretched arms. None keeps them.
    x_halfwidth: float | None = 200.0
    organs: tuple[str, ...] = ("heart", "lung_left", "lung_right", "brain")
    include_bone: bool = True
    include_cord: bool = True
    #: Skin solidification (see :mod:`inob.mesh.solidify`). The pitch has to
    #: bridge the ~2 mm skin wall; the seal has to exceed the facial orifices.
    skin_pitch_mm: float = 3.0
    skin_samples: int = 3_000_000
    skin_seal_mm: float = 12.0
    shrinkwrap: dict[str, ShrinkwrapParams] = field(default_factory=dict)

    def sw(self, label: str) -> ShrinkwrapParams:
        if label in self.shrinkwrap:
            return self.shrinkwrap[label]
        return SHRINKWRAP_DEFAULTS.get(label, SHRINKWRAP_DEFAULTS["_default"])


def _to_target(m: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    out = m.copy()
    out.apply_transform(np.asarray(T, dtype=float))
    return out


def _crop(m: trimesh.Trimesh, spec: ThoraxSpec, label: str,
          *, largest_only: bool = True) -> trimesh.Trimesh:
    """Apply the body clip band and the arm cut, capping every cut."""
    out = clip_band(m, lo=spec.z_lo, hi=spec.z_hi, label=label,
                    largest_only=largest_only)
    if len(out.faces) == 0 or spec.x_halfwidth is None:
        return out
    h = spec.x_halfwidth
    out = clip_capped(out, [-h, 0, 0], [1, 0, 0], label=f"{label}[x>-{h:g}]",
                      largest_only=largest_only)
    if len(out.faces) == 0:
        return out
    return clip_capped(out, [h, 0, 0], [-1, 0, 0], label=f"{label}[x<{h:g}]",
                       largest_only=largest_only)


def _skin(spec: ThoraxSpec) -> trimesh.Trimesh:
    """The body as a solid, from the skin shell.

    **Solidify before clipping — the order is load-bearing.** Clipping first is
    tempting (it drops the legs before the expensive voxelisation) but wrong: a
    plane through a double-walled shell has an *annular* cross-section, and
    capping an annulus does not seal the cavity behind it. Do that and the body
    interior becomes continuous with the outside — measurably so, the free space
    collapses from two components (exterior + a 54 L cavity) to one — after
    which no amount of filling recovers a body. Solidified first, the same cut
    crosses a simple disc and caps correctly.
    """
    src = spec.project_root / "data/torso/FJ2810_BP22617_FMA7163_Skin.stl"
    raw = trimesh.load(src, process=True)
    logger.info("[skin] shell: %d V / %d F, area %.2f m^2 (~2x a body: a double "
                "wall), encloses %.0f cm^3 of skin",
                len(raw.vertices), len(raw.faces), raw.area / 1e6, raw.volume / 1000.0)
    solid = solidify_shell(_to_target(raw, spec.transform),
                           pitch=spec.skin_pitch_mm, n_samples=spec.skin_samples,
                           seal_mm=spec.skin_seal_mm, seed=0)
    logger.info("[skin] whole-body solid: %.0f cm^3", solid.volume / 1000.0)
    return _crop(solid, spec, "skin")


def _bone(spec: ThoraxSpec) -> trimesh.Trimesh:
    """Every atlas bone inside the band, kept as separate bodies.

    Bones are genuinely disjoint (24 ribs, sternum, vertebrae, skull plates); a
    single watertight surface over them is neither achievable nor meaningful.
    The FEM voxelises them, so each is cheap-repaired and cropped, then merged
    as a multi-body compartment.
    """
    paths = sorted(glob.glob(str(spec.project_root / "internal_meshes/bone/*.obj")))
    if not paths:
        raise FileNotFoundError("no bones under internal_meshes/bone")
    kept: list[trimesh.Trimesh] = []
    for p in paths:
        m = _to_target(trimesh.load(p, process=True, force="mesh"), spec.transform)
        if m.bounds[0, 2] > (spec.z_hi if spec.z_hi is not None else np.inf):
            continue
        if m.bounds[1, 2] < spec.z_lo:
            continue                      # entirely below the band: skip cheaply
        c = _crop(m, spec, f"bone:{Path(p).stem[:28]}", largest_only=False)
        if len(c.faces):
            kept.append(c)
    merged = trimesh.util.concatenate(kept)
    logger.info("[bone] %d/%d bones in band -> %d V / %d F",
                len(kept), len(paths), len(merged.vertices), len(merged.faces))
    return merged


def _organ(name: str, spec: ThoraxSpec) -> trimesh.Trimesh:
    raw = load_composite(
        name, mesh_root=spec.project_root / "internal_meshes",
        manifest_path=default_manifest_path(spec.project_root),
    )
    m = _to_target(raw, spec.transform)
    label = f"mesh_{name}"
    out = watertighten(m, label, spec.sw(label), force_shrinkwrap=True, seed=0)
    logger.info("[%s] %.0f cm^3, euler=%d", name, out.volume / 1000.0, out.euler_number)
    return out


def _cord(spec: ThoraxSpec) -> trimesh.Trimesh:
    paths = sorted(glob.glob(str(spec.project_root / "data/spinal_cord/*.stl")))
    if not paths:
        raise FileNotFoundError("no spinal-cord STL")
    m = _to_target(trimesh.load(paths[0], process=True), spec.transform)
    return clip_band(m, lo=spec.z_lo, hi=spec.z_hi, label="spinal_cord")


def build_thorax(spec: ThoraxSpec) -> dict[str, trimesh.Trimesh]:
    """Build every compartment. Returns name -> mesh, all in the target frame."""
    out: dict[str, trimesh.Trimesh] = {"skin": _skin(spec)}
    for name in spec.organs:
        out[name] = _organ(name, spec)
    if spec.include_bone:
        out["bone"] = _bone(spec)
    if spec.include_cord:
        out["spinal_cord"] = _cord(spec)
    return out


def summarise(meshes: dict[str, trimesh.Trimesh]) -> dict[str, dict]:
    """Per-compartment topology + placement, for logging and for the gate."""
    s = {}
    for k, m in meshes.items():
        r = dict(report(m))
        r["centroid_mm"] = np.round(np.asarray(m.centroid), 1).tolist()
        r["bounds_mm"] = np.round(np.asarray(m.bounds), 1).tolist()
        r["conductivity_sm"] = CONDUCTIVITY_SM.get(k)
        s[k] = r
    return s


def write_summary(meshes: dict[str, trimesh.Trimesh], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarise(meshes), indent=2))
    logger.info("[saved] %s", path)
    return path
