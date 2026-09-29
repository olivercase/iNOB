"""Layout rules for the exploded anatomy figure (scripts/exploded_anatomy.py).

Rendering needs the BodyParts3D meshes and a GPU-ish context, so these
exercise the pure layout functions instead: the packing and label stacking
are where the figure actually goes wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from exploded_anatomy import (  # noqa: E402
    LABEL_HEIGHT_FRACTION,
    _px,
    label_font_px,
    pack_columns,
    stack_labels,
)


def _box(x: float, z_lo: float, z_hi: float, width: float = 40.0) -> pv.PolyData:
    return pv.Box(bounds=(x - width / 2, x + width / 2, -10.0, 10.0, z_lo, z_hi)).extract_surface()


def test_label_size_tracks_the_render_height() -> None:
    """Sizing text in raw pixels made it shrink whenever --height went up."""
    assert label_font_px(2600) == round(LABEL_HEIGHT_FRACTION * 2600)
    assert label_font_px(5200) == pytest.approx(2 * label_font_px(2600), abs=1)


def test_label_size_never_collapses_for_a_tiny_render() -> None:
    assert label_font_px(1) >= 8


def test_line_widths_scale_with_the_render_height() -> None:
    assert _px(3.2, 2600) == pytest.approx(3.2)
    assert _px(3.2, 5200) == pytest.approx(6.4)


def test_stacked_systems_do_not_share_a_column_when_a_label_needs_the_room() -> None:
    """The regression that appeared once the type was big enough to read.

    Two systems that merely sit above one another can share a column on mesh
    extents alone, and then the lower one's caption lands on the upper one's
    mesh. Reserving the label's headroom separates them.
    """
    groups = {
        "skin": _box(0.0, 0.0, 1000.0, width=300.0),
        "tall": _box(0.0, 400.0, 900.0),
        "short": _box(0.0, 100.0, 300.0),
    }
    # A 100 mm gap clears STACK_GAP_MM (60), so the meshes alone would share
    # a column; a 200 mm label headroom closes it.
    shared = pack_columns(groups, label_headroom_mm=0.0)
    assert shared["tall"] == pytest.approx(shared["short"])

    separated = pack_columns(groups, label_headroom_mm=200.0)
    assert separated["tall"] != pytest.approx(separated["short"])


def test_pack_columns_keeps_the_body_at_the_centre() -> None:
    groups = {
        "skin": _box(0.0, 0.0, 1000.0, width=300.0),
        "a": _box(0.0, 400.0, 900.0),
        "b": _box(0.0, 100.0, 380.0),
    }
    dx = pack_columns(groups, label_headroom_mm=50.0)
    assert dx["skin"] == 0.0
    # Systems are dealt to either side, not piled on one.
    assert min(dx.values()) < 0.0 < max(dx.values())


def test_stack_labels_lifts_overlapping_captions_apart() -> None:
    placed = {"a": (0.0, 500.0), "b": (5.0, 500.0)}
    texts = {"a": "Gastrointestinal tract", "b": "Vertebral column"}
    out = stack_labels(placed, texts, mm_per_pt=0.4, font_px=47)
    assert abs(out["a"] - out["b"]) > 0.0


def test_stack_labels_leaves_well_separated_captions_alone() -> None:
    placed = {"a": (0.0, 500.0), "b": (5000.0, 500.0)}
    texts = {"a": "Heart", "b": "Lungs"}
    out = stack_labels(placed, texts, mm_per_pt=0.4, font_px=47)
    assert out == {"a": 500.0, "b": 500.0}


def test_stack_labels_only_ever_moves_labels_up() -> None:
    """Deterministic and non-oscillating, whatever order the dict arrives in."""
    placed = {"a": (0.0, 500.0), "b": (5.0, 500.0), "c": (10.0, 500.0)}
    texts = {"a": "Alpha", "b": "Beta", "c": "Gamma"}
    out = stack_labels(placed, texts, mm_per_pt=0.4, font_px=47)
    for group, height in out.items():
        assert height >= placed[group][1]
    assert len(set(np.round(list(out.values()), 6))) == 3
