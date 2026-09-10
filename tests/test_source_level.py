"""``forward.source_level`` — optional level/span clipping of sampled dipoles.

The load-bearing test here is the *default*: every vagus and spine run ever
made sampled the tissue's whole Z extent, so ``source_level=None`` must remain
bit-identical to calling the sampler directly. A regression there would
silently change every existing leadfield.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from inob.sources.vagus import level_span_z_band


class _Bone:
    """Stand-in for a bone dir: fixed, made-up bands, no segmentation needed."""

    BANDS: ClassVar[dict[str, tuple[float, float]]] = {
        "c1": (100.0, 110.0), "c7": (40.0, 52.0), "t1": (30.0, 43.0),
    }


def _band(bone_dir, level):
    return _Bone.BANDS[level]


@pytest.fixture(autouse=True)
def _patch_band(monkeypatch):
    monkeypatch.setattr("inob.anatomy.vertebra_z_band", _band)


def test_single_level_is_that_levels_band():
    assert level_span_z_band(None, "c7") == (40.0, 52.0)


def test_span_is_the_union_of_its_endpoints():
    # C1's top down to C7's bottom — the cervical vagus.
    assert level_span_z_band(None, "c1-c7") == (40.0, 110.0)


def test_span_order_does_not_matter():
    assert level_span_z_band(None, "c7-c1") == level_span_z_band(None, "c1-c7")


def test_case_and_whitespace_tolerated():
    assert level_span_z_band(None, " C1 - C7 ") == (40.0, 110.0)


def test_unknown_level_raises():
    with pytest.raises(KeyError):
        level_span_z_band(None, "z9")


def test_empty_spec_raises():
    with pytest.raises(ValueError, match="empty source level spec"):
        level_span_z_band(None, "-")


def test_default_none_leaves_positions_untouched():
    """The guarantee that protects every pre-existing run."""
    from inob.sources.vagus import _restrict_to_level

    class _Cfg:
        class forward:
            source_level = None
            source_tissue = "vagus_left"

        class data:
            bone_dir = None

    pos = np.array([[0.0, 0.0, z] for z in (10.0, 45.0, 105.0, 400.0)])
    out = _restrict_to_level(pos, _Cfg)
    assert out is pos, "source_level=None must not copy or filter"


def test_level_clips_to_the_band():
    from inob.sources.vagus import _restrict_to_level

    class _Cfg:
        class forward:
            source_level = "c1-c7"
            source_tissue = "vagus_left"

        class data:
            bone_dir = None

    pos = np.array([[0.0, 0.0, z] for z in (10.0, 45.0, 105.0, 400.0)])
    out = _restrict_to_level(pos, _Cfg)
    assert out[:, 2].tolist() == [45.0, 105.0]


def test_empty_selection_is_a_readable_error():
    from inob.sources.vagus import _restrict_to_level

    class _Cfg:
        class forward:
            source_level = "c7"
            source_tissue = "vagus_left"

        class data:
            bone_dir = None

    pos = np.array([[0.0, 0.0, 900.0]])
    with pytest.raises(ValueError, match="no vagus_left dipoles lie within C7"):
        _restrict_to_level(pos, _Cfg)
