"""``inob.anatomy`` — naming the vertebral level of a body-frame height.

Reviewers asked for "C6/C7" rather than "z = 1419 mm". The name must come from
the segmented bones the FEM was built from, and it must disappear quietly —
not crash a figure — on a checkout without the Git LFS meshes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from inob import anatomy
from inob.anatomy import VertebralBody, VertebralLevel, level_from_bodies

REPO_BONE = Path(__file__).resolve().parents[1] / "data" / "bone"

# A made-up column: C6 over C7 over T1, with a 4 mm disc between C6 and C7
# and C7/T1 abutting, as the BodyParts3D bodies nearly do.
_BODIES = (
    VertebralBody("c6", 100.0, 116.0),
    VertebralBody("c7", 80.0, 96.0),
    VertebralBody("t1", 60.0, 80.0),
)


def test_a_plane_through_a_body_names_that_body() -> None:
    lv = level_from_bodies(88.0, _BODIES)
    assert lv == VertebralLevel("c7", "c7")
    assert lv.short == "C7" and lv.long == "body of C7"


def test_a_plane_through_the_disc_names_both_neighbours() -> None:
    lv = level_from_bodies(98.0, _BODIES)
    assert lv is not None and lv.is_gap
    assert lv.short == "C6/C7"
    assert lv.long == "C6/C7 intervertebral gap"


def test_abutting_bodies_still_get_a_disc_band() -> None:
    """The disc is not segmented, so a zero-height gap would never be hit."""
    for z in (78.0, 80.0, 82.0):
        assert level_from_bodies(z, _BODIES).short == "C7/T1"
    assert level_from_bodies(84.0, _BODIES).short == "C7"


def test_no_disc_is_named_across_a_missing_level() -> None:
    bodies = (_BODIES[0], _BODIES[2])  # C7 absent
    lv = level_from_bodies(98.0, bodies)
    assert lv is not None and not lv.is_gap


def test_no_disc_between_atlas_and_axis() -> None:
    bodies = (VertebralBody("c1", 140.0, 150.0), VertebralBody("c2", 120.0, 148.0))
    assert not level_from_bodies(140.0, bodies).is_gap


def test_a_plane_off_the_column_has_no_level() -> None:
    assert level_from_bodies(500.0, _BODIES) is None
    assert level_from_bodies(-500.0, _BODIES) is None
    assert level_from_bodies(100.0, ()) is None


def test_missing_meshes_give_none_not_an_error(tmp_path: Path) -> None:
    assert anatomy.vertebral_level(tmp_path, 1400.0) is None
    assert anatomy.vertebral_level(None, 1400.0) is None
    assert anatomy.level_suffix(tmp_path, 1400.0) == ""


def test_an_lfs_pointer_is_treated_as_missing(tmp_path: Path) -> None:
    (tmp_path / "FJ1_Seventh cervical vertebra.stl").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\n"
    )
    assert anatomy.vertebral_level(tmp_path, 1400.0) is None


def test_body_is_taken_from_the_anterior_slice(tmp_path: Path) -> None:
    """A spinous process hanging below the body must not move its endplate.

    Anterior is -Y (docs/COORDINATES.md): the body sits at low Y, the process
    at high Y and 30 mm lower.
    """
    body = trimesh.creation.box(extents=(30.0, 20.0, 16.0))
    body.apply_translation((0.0, -30.0, 88.0))
    process = trimesh.creation.box(extents=(8.0, 40.0, 8.0))
    process.apply_translation((0.0, 30.0, 60.0))
    trimesh.util.concatenate([body, process]).export(tmp_path / "X_Seventh cervical vertebra.stl")
    anatomy._vertebral_bodies_cached.cache_clear()
    bodies = anatomy.vertebral_bodies(tmp_path)
    assert [b.level for b in bodies] == ["c7"]
    assert bodies[0].z_lo == pytest.approx(80.0, abs=0.5)
    assert bodies[0].z_hi == pytest.approx(96.0, abs=0.5)
    assert anatomy.level_suffix(tmp_path, 88.0) == " (C7)"


@pytest.mark.skipif(
    not any(REPO_BONE.glob("*Seventh cervical vertebra*.stl")),
    reason="vertebra meshes not present",
)
def test_real_column_is_ordered_and_names_the_neck() -> None:
    bodies = anatomy.vertebral_bodies(REPO_BONE)
    if len(bodies) < len(anatomy.VERTEBRA_LEVELS):
        pytest.skip("vertebra meshes are LFS pointers or incomplete")
    mids = np.array([0.5 * (b.z_lo + b.z_hi) for b in bodies])
    # C1 first, L5 last, each body below the one before it.
    assert bodies[0].level == "c1" and bodies[-1].level == "l5"
    assert (np.diff(mids) < 0).all()
    # The C7 body sits where the neck meets the thorax in this model
    # (head crown ~1642 mm, docs/COORDINATES.md).
    c7 = next(b for b in bodies if b.level == "c7")
    assert 1340.0 < c7.z_lo < c7.z_hi < 1410.0
    # Every plane from C2 to L5 gets a name.
    for z in np.arange(1450.0, 910.0, -5.0):
        assert anatomy.vertebral_level(REPO_BONE, float(z)) is not None
    assert anatomy.vertebral_level(REPO_BONE, 1642.0) is None
