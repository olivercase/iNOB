"""Explicit point sources must be honoured by *every* forward path.

``cfg.forward.point_sources`` is how the GUI sends clicked dipoles to the
solver. ``forward/solve.py`` (serial) and ``forward/eeg.py`` resolved them via
``resolve_source_positions``, but ``forward/chunk.py`` and ``forward/reduce.py``
called ``sample_source_tissues`` directly and silently ignored them — and the
chunked path is the *default* local multi-core one, so in practice a clicked
source never reached the solver and the answer described the whole sampled
nerve instead.

Found by running the GUI end to end against a real DUNEuro build.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from inob.sources.vagus import resolve_source_positions

SRC = Path(__file__).resolve().parents[1] / "src" / "inob" / "forward"

# Every module that turns a config into dipole positions for a solve.
SOLVE_PATHS = ["solve.py", "chunk.py", "reduce.py", "eeg.py"]


@pytest.mark.parametrize("module", SOLVE_PATHS)
def test_solve_paths_resolve_sources_through_the_shared_helper(module: str) -> None:
    """No forward path may call the sampler directly and skip point sources."""
    text = (SRC / module).read_text()
    assert "resolve_source_positions" in text, (
        f"{module} does not use resolve_source_positions, so explicit "
        "cfg.forward.point_sources would be ignored on this path"
    )
    tree = ast.parse(text)
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "sample_source_tissues" not in called, (
        f"{module} calls sample_source_tissues directly — use "
        "resolve_source_positions so point_sources are honoured"
    )


class _Fwd:
    def __init__(self, point_sources=()):
        self.point_sources = point_sources
        self.source_tissue = "vagus_left"
        self.source_spacing_mm = 5.0


class _Cfg:
    def __init__(self, point_sources=()):
        self.forward = _Fwd(point_sources)


def _unit_cube_fem():
    """A single-tetra 'mesh' big enough to contain the points we test."""
    from inob.io.hdf5 import FemMesh

    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                      [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    tets = np.array([[0, 1, 2, 3]])
    return FemMesh(nodes=nodes, tets=tets, tissue=np.array([1]),
                   tissue_labels=("vagus_left",))


def test_explicit_point_sources_are_returned_verbatim() -> None:
    pts = ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    out = resolve_source_positions(_Cfg(pts), _unit_cube_fem())
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.array(pts))


def test_sources_outside_the_mesh_raise_a_readable_error() -> None:
    """DUNEuro's own error for this is an opaque C++ kdtree exception."""
    pts = ((1.0, 1.0, 1.0), (500.0, 500.0, 500.0))
    with pytest.raises(ValueError) as e:
        resolve_source_positions(_Cfg(pts), _unit_cube_fem())
    msg = str(e.value)
    assert "outside the FEM model" in msg
    assert "source 2" in msg          # names which one
    assert "500" in msg               # and where it was
