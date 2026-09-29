"""Heart, brain and lung as modelled tissues.

These three came in together as the volume-conductor and source compartments a
concurrent brain-heart-vagus array needs. The invariants worth pinning are the
ones that fail silently: a tissue declared in the config but with no
voxelisation rule (it would be skipped and the FEM would quietly lack it), and
a bulky tissue routed to the thin-nerve Z-slab sampler (which would place its
dipoles on the body midline, outside the organ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inob.config import SOURCE_TARGETS, load_config
from inob.fem.cgal_builder import _GROUP_TISSUES, VOXELISATION
from inob.io.hdf5 import FemMesh
from inob.physiology.profiles import profile_for_tag
from inob.sources.vagus import sample_source_tissues

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"

_TET = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _bilateral_fem(label: str) -> FemMesh:
    """Two blobs either side of the midline, as heart/brain/muscle really are."""
    nodes: list[np.ndarray] = []
    tets: list[list[int]] = []
    tissue: list[int] = []
    for centre in ([-40.0, 0.0, 1200.0], [40.0, 0.0, 1200.0]):
        for dz in (0.0, 10.0, 20.0):
            base = len(nodes)
            nodes.extend(np.asarray(centre) + np.array([0.0, 0.0, dz]) + _TET)
            tets.append([base, base + 1, base + 2, base + 3])
            tissue.append(1)
    return FemMesh(
        np.asarray(nodes),
        np.asarray(tets, dtype=np.int32),
        np.asarray(tissue, dtype=np.int32),
        (label,),
        "mm",
    )


def test_every_declared_group_tissue_can_be_voxelised() -> None:
    cfg = load_config(DEFAULT_CFG)
    group = {t for t, _ in _GROUP_TISSUES}
    # skin and the vagus trunks have their own voxelisers; everything else in
    # fem.tissues must be reachable through the group path *and* carry a rule.
    for tissue in cfg.fem.tissues:
        if tissue in {"skin", "vagus_left", "vagus_right"}:
            continue
        assert tissue in group, f"{tissue} declared in fem.tissues but not a group tissue"
        assert tissue in VOXELISATION, f"{tissue} has no VOXELISATION rule"


def test_new_tissues_have_a_conductivity() -> None:
    sigma = load_config(DEFAULT_CFG).forward.conductivities_sm
    for tissue in ("heart", "brain", "lung"):
        assert sigma[tissue] > 0.0


def test_bulky_tissues_are_volume_filled_not_midline_averaged() -> None:
    """A Z-slab average would collapse the two lobes onto x = 0 — outside both."""
    for label in ("heart", "brain", "lung"):
        pos = sample_source_tissues(_bilateral_fem(label), label, spacing_mm=5.0)
        assert len(pos) >= 2
        assert (np.abs(pos[:, 0]) > 1.0).all(), f"{label} sources fell on the midline"


def test_heart_and_brain_targets_resolve_to_their_own_physiology() -> None:
    """Not the vagus fallback, which would quietly relabel vagal physiology."""
    for tag, strength_nAm in (("heart", 2000.0), ("brain", 10.0)):
        assert tag in SOURCE_TARGETS
        profile = profile_for_tag(tag)
        assert profile.name == tag
        assert profile.default_strength_nAm == strength_nAm
        # Both are equivalent-dipole figures from the literature, not products
        # of the axonal fibre model.
        assert profile.event_moment_nAm == strength_nAm
