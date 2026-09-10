"""Leadfield NPZ schema + atomic-write tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inob.io.npz import (
    REQUIRED_KEYS,
    Leadfield,
    SchemaError,
    load_leadfield,
    save_leadfield,
    validate_leadfield,
)


def _make_lf(C: int = 12, S: int = 3, *, seed: int | None = 0) -> Leadfield:
    rng = np.random.default_rng(seed if seed is not None else 0)
    L = rng.standard_normal((C, 3 * S)) * 1e-9
    return Leadfield(
        L=L,
        L_fT_per_nAm=L * 1e6,
        source_pos=rng.standard_normal((S, 3)),
        coil_pos=rng.standard_normal((C, 3)),
        coil_orient=np.eye(3)[rng.integers(0, 3, C)],
        channel_names=tuple(f"ch{i}" for i in range(C)),
        conductivities=np.array([3e-4, 4.3e-4]),
        tissue_labels=("vagus_left", "skin"),
        seed=seed,
    )


def test_roundtrip(tmp_path: Path) -> None:
    lf = _make_lf()
    p = tmp_path / "lf.npz"
    save_leadfield(p, lf)
    loaded = load_leadfield(p)
    np.testing.assert_allclose(loaded.L, lf.L)
    np.testing.assert_allclose(loaded.source_pos, lf.source_pos)
    assert loaded.channel_names == lf.channel_names
    assert loaded.tissue_labels == lf.tissue_labels
    assert loaded.seed == 0


def test_validate_passes() -> None:
    validate_leadfield(_make_lf())


def test_validate_requires_finite() -> None:
    lf = _make_lf()
    L = lf.L.copy()
    L[0, 0] = np.nan
    bad = Leadfield(
        L=L,
        L_fT_per_nAm=L * 1e6,
        source_pos=lf.source_pos,
        coil_pos=lf.coil_pos,
        coil_orient=lf.coil_orient,
        channel_names=lf.channel_names,
        conductivities=lf.conductivities,
        tissue_labels=lf.tissue_labels,
        seed=lf.seed,
    )
    with pytest.raises(SchemaError, match="non-finite"):
        validate_leadfield(bad)


def test_validate_rejects_shape_mismatch() -> None:
    lf = _make_lf()
    bad = Leadfield(
        L=lf.L[:, :-1],  # wrong second dim
        L_fT_per_nAm=lf.L_fT_per_nAm[:, :-1],
        source_pos=lf.source_pos,
        coil_pos=lf.coil_pos,
        coil_orient=lf.coil_orient,
        channel_names=lf.channel_names,
        conductivities=lf.conductivities,
        tissue_labels=lf.tissue_labels,
    )
    with pytest.raises(SchemaError, match=r"L shape"):
        validate_leadfield(bad)


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    np.savez(p, L=np.zeros((1, 3)))
    with pytest.raises(SchemaError, match="missing keys"):
        load_leadfield(p)


def test_required_keys_includes_source_pos() -> None:
    """Regression: legacy cluster reduce.py omitted source_pos. Schema mandates it."""
    assert "source_pos" in REQUIRED_KEYS


def test_atomic_save_leaves_old_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "lf.npz"
    save_leadfield(p, _make_lf())
    sentinel_size = p.stat().st_size

    def bad_savez(*a, **kw):
        raise RuntimeError("explode")

    monkeypatch.setattr(np, "savez", bad_savez)
    with pytest.raises(RuntimeError, match="explode"):
        save_leadfield(p, _make_lf())
    # file is unchanged
    assert p.stat().st_size == sentinel_size
    siblings = [q for q in tmp_path.iterdir() if q.name.startswith(f".{p.name}.")]
    assert siblings == [], f"orphan tempfile(s) left: {siblings}"
