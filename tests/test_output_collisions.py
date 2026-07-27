"""No output artefact may be shared between two different source targets.

Two failure modes, both silent:

  * **Collision** — a per-target artefact written to a shared path. Running
    spine then overwrites the vagus result of the same name, and the next
    analysis reads whichever ran last while still labelling it correctly.
  * **False separation** — tagging something that is genuinely shared (the
    FEM contains every tissue) would force a needless rebuild per target.

So every path in ``cfg.outputs`` is classified here exactly once, and the
classification is asserted rather than assumed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import pytest

from inob.cli._common import add_common_args, setup
from inob.config import SOURCE_TARGETS, tag_path

# Built from all tissues at once and reused by every target. Sharing these is
# the point: they are what makes a second target cheap.
SHARED = {
    "base",
    "geometry_mat", "geometry_png",     # every compartment, incl. the cord
    "fem_mat", "fem_png",               # every tissue label
    "sensors_mat", "sensors_png",       # OPM array wraps the whole torso
    "intermediate_bone_clean", "intermediate_bone_union",
    "logs_dir",                         # timestamped per run, cannot collide
}

# Derived from one target's sources; must never share a path.
PER_TARGET = {
    "electrodes_mat", "electrodes_png",   # patch is sited over the target
    "forward_npz", "forward_eeg_npz",     # the leadfields themselves
    "forward_chunks_dir",                 # per-target solve intermediates
    "sensitivity_dir",                    # re-solves this target's sources
}


def _cfg_for(target: str | None):
    p = argparse.ArgumentParser()
    add_common_args(p)
    argv = ["--config", "configs/default.yaml"]
    if target:
        argv += ["--source-target", target]
    return setup(p.parse_args(argv), log_prefix="test")


def test_classification_covers_every_output_field() -> None:
    """A newly added output must be classified, not silently defaulted."""
    declared = {f.name for f in fields(_cfg_for(None).outputs)}
    classified = SHARED | PER_TARGET
    assert declared == classified, (
        f"unclassified outputs: {sorted(declared - classified)}; "
        f"stale entries: {sorted(classified - declared)}"
    )


@pytest.mark.parametrize("field_name", sorted(PER_TARGET))
def test_per_target_outputs_never_collide(field_name: str) -> None:
    """Every pair of targets must get a distinct path for this artefact."""
    seen: dict[Path, str] = {}
    for target in SOURCE_TARGETS:
        path = getattr(_cfg_for(target).outputs, field_name)
        assert path not in seen, (
            f"{field_name}: targets {seen[path]!r} and {target!r} both write "
            f"{path} — one silently overwrites the other"
        )
        seen[path] = target


@pytest.mark.parametrize("field_name", sorted(SHARED))
def test_shared_outputs_stay_shared(field_name: str) -> None:
    """Shared artefacts must NOT be tagged, or every target rebuilds the FEM."""
    paths = {getattr(_cfg_for(t).outputs, field_name) for t in SOURCE_TARGETS}
    assert len(paths) == 1, (
        f"{field_name} differs across targets ({sorted(map(str, paths))}) but is "
        "built from all tissues and should be shared"
    )


def test_shared_artefacts_are_not_named_after_one_target() -> None:
    """A mesh containing every tissue must not be called 'vagus' anything.

    The old fem_vagus.mat / vagus_geometry.mat names implied a per-target file
    that was in fact shared and overwritten by every run.
    """
    outputs = _cfg_for(None).outputs
    for name in sorted(SHARED):
        path = getattr(outputs, name)
        if not isinstance(path, Path):
            continue
        for tissue in ("vagus", "spine", "spinal_cord", "muscle"):
            assert tissue not in path.name, (
                f"{name} = {path.name} is shared by all targets but is named "
                f"after {tissue!r}"
            )


def test_untargeted_run_keeps_legacy_paths() -> None:
    """Omitting --source-target must not start tagging things."""
    plain = _cfg_for(None).outputs
    assert plain.forward_chunks_dir.name == "chunks"
    assert plain.sensitivity_dir.name == "sensitivity"
    assert plain.electrodes_mat.name == "electrode_array.mat"


# ── the tagging primitive ──────────────────────────────────────────────────

def test_tag_path_handles_files_and_dirs() -> None:
    assert tag_path(Path("a/b.png"), "spine") == Path("a/b_spine.png")
    assert tag_path(Path("a/chunks"), "spine") == Path("a/chunks_spine")


def test_tag_path_is_idempotent() -> None:
    """Applying a tag twice must not produce ``x_spine_spine``."""
    once = tag_path(Path("a/b.png"), "spine")
    assert tag_path(once, "spine") == once


def test_empty_tag_is_a_noop() -> None:
    assert tag_path(Path("a/b.png"), "") == Path("a/b.png")


# ── figures ────────────────────────────────────────────────────────────────

FIGURE_RENDERERS = [
    ("inob.viz.cap_compare", "cap_compare.png"),
    ("inob.viz.cross_modality_plot", "cross_modality.png"),
    ("inob.viz.surface_topoplot", "surface_topoplots.png"),
    ("inob.viz.location_optimisation", "location_optimisation.png"),
    ("inob.viz.physiology_plot", "physiology.png"),
]


@pytest.mark.parametrize(("module", "filename"), FIGURE_RENDERERS)
def test_leadfield_figures_default_to_tagged_names(module: str, filename: str) -> None:
    """Figures derived from a leadfield must carry the target in their name."""
    from inob.config import target_output
    spine = target_output(_cfg_for("spine"), filename)
    vagus = target_output(_cfg_for("vagus"), filename)
    assert spine != vagus
    assert "spine" in spine.name
    # Sanity: the module really does route through the shared helper.
    import importlib
    src = importlib.import_module(module).__file__
    assert "target_output" in Path(src).read_text(), (
        f"{module} builds its default output path without target_output()"
    )
