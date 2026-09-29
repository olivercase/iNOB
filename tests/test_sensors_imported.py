"""Importing a user's own sensor or electrode array, and single-axis arrays."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io
import trimesh

from inob.config import ConfigError, load_config
from inob.io.hdf5 import SchemaError, load_sensors, validate_sensors
from inob.sensors.electrodes import generate_electrode_array
from inob.sensors.imported import (
    RawArray,
    import_electrodes,
    import_magnetometers,
    read_array_file,
    skin_offsets,
)
from inob.sensors.layout import (
    OTHER,
    SINGLE_AXIS,
    TRIAXIAL,
    array_layout,
    position_channels,
    radial_channels,
    require_triaxial,
)
from inob.sensors.triaxial import build_radial, build_triaxial, generate_sensor_array

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CFG = REPO_ROOT / "configs" / "tiny_test.yaml"

RADIUS = 100.0


@pytest.fixture
def skin() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(radius=RADIUS, subdivisions=4)


def _ring(n: int = 4, r: float = RADIUS + 10.0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` positions 10 mm outside the sphere, with outward normals."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    nrm = np.column_stack([np.cos(th), np.sin(th), np.zeros(n)])
    return r * nrm, nrm


# ── readers ────────────────────────────────────────────────────────────────


def test_csv_with_header_and_comments(tmp_path) -> None:
    f = tmp_path / "array.csv"
    f.write_text("# my array\nx,y,z,nx,ny,nz\n1,2,3,0,0,1\n4,5,6,0,2,0  # unnormalised\n")
    raw = read_array_file(f, kind="meg")
    np.testing.assert_allclose(raw.positions, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(raw.orientations, [[0, 0, 1], [0, 2, 0]])


def test_whitespace_txt_is_read(tmp_path) -> None:
    f = tmp_path / "array.txt"
    f.write_text("1 2 3 1 0 0\n4\t5\t6\t0\t1\t0\n")
    assert read_array_file(f, kind="meg").positions.shape == (2, 3)


def test_csv_without_orientations_is_refused_for_meg(tmp_path) -> None:
    f = tmp_path / "array.csv"
    f.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(SchemaError, match="x y z nx ny nz"):
        read_array_file(f, kind="meg")
    assert read_array_file(f, kind="eeg").orientations is None


def test_non_finite_values_are_refused(tmp_path) -> None:
    f = tmp_path / "array.csv"
    f.write_text("1,2,nan,0,0,1\n")
    with pytest.raises(SchemaError, match="NaN or infinite"):
        read_array_file(f, kind="meg")


def test_npz_converts_its_unit(tmp_path) -> None:
    f = tmp_path / "array.npz"
    np.savez(f, coil_pos=np.array([[0.1, 0.0, 0.0]]), coil_orient=[[1.0, 0, 0]], unit="m")
    np.testing.assert_allclose(read_array_file(f).positions, [[100.0, 0.0, 0.0]])


def test_npz_defaults_to_mm(tmp_path) -> None:
    f = tmp_path / "array.npz"
    np.savez(f, coil_pos=np.array([[110.0, 0.0, 0.0]]), coil_orient=[[1.0, 0, 0]])
    np.testing.assert_allclose(read_array_file(f).positions, [[110.0, 0.0, 0.0]])


def test_unknown_extension_is_refused(tmp_path) -> None:
    f = tmp_path / "array.json"
    f.write_text("{}")
    with pytest.raises(SchemaError, match="unrecognised"):
        read_array_file(f)


@pytest.mark.parametrize(("unit", "scale"), [("m", 1000.0), ("cm", 10.0), ("mm", 1.0)])
def test_fieldtrip_grad_coilpos_units(tmp_path, unit, scale) -> None:
    f = tmp_path / "grad.mat"
    pos = np.array([[0.11, 0.0, 0.0], [0.0, 0.11, 0.0]])
    ori = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scipy.io.savemat(f, {"grad": {"coilpos": pos, "coilori": ori, "unit": unit}})
    raw = read_array_file(f, kind="meg")
    np.testing.assert_allclose(raw.positions, pos * scale)
    np.testing.assert_allclose(raw.orientations, ori)


def test_fieldtrip_grad_chanpos_fallback(tmp_path) -> None:
    f = tmp_path / "grad.mat"
    scipy.io.savemat(
        f,
        {"grad": {"chanpos": [[11.0, 0, 0]], "chanori": [[1.0, 0, 0]], "unit": "cm"}},
    )
    np.testing.assert_allclose(read_array_file(f).positions, [[110.0, 0.0, 0.0]])


def test_fieldtrip_grad_without_unit_is_refused(tmp_path) -> None:
    f = tmp_path / "grad.mat"
    scipy.io.savemat(f, {"grad": {"coilpos": [[1.0, 0, 0]], "coilori": [[1.0, 0, 0]]}})
    with pytest.raises(SchemaError, match="no 'unit'"):
        read_array_file(f)


def test_fieldtrip_gradiometer_is_refused(tmp_path) -> None:
    f = tmp_path / "grad.mat"
    scipy.io.savemat(
        f,
        {
            "grad": {
                "coilpos": [[0.1, 0, 0], [0.12, 0, 0]],
                "coilori": [[1.0, 0, 0], [-1.0, 0, 0]],
                "tra": [[1.0, 1.0]],
                "unit": "m",
            }
        },
    )
    with pytest.raises(SchemaError, match="gradiometer"):
        read_array_file(f)


def test_fieldtrip_magnetometer_tra_is_accepted(tmp_path) -> None:
    f = tmp_path / "grad.mat"
    scipy.io.savemat(
        f,
        {
            "grad": {
                "coilpos": [[0.1, 0, 0], [0.0, 0.1, 0]],
                "coilori": [[1.0, 0, 0], [0.0, 1.0, 0]],
                "tra": np.eye(2),
                "unit": "m",
            }
        },
    )
    assert read_array_file(f).positions.shape == (2, 3)


def test_fieldtrip_v73_grad(tmp_path) -> None:
    """A v7.3 file is HDF5 with MATLAB's column-major arrays stored transposed."""
    f = tmp_path / "grad73.mat"
    pos = np.array([[0.11, 0.0, 0.0], [0.0, 0.11, 0.0], [0.0, 0.0, 0.11], [0.05, 0.05, 0.0]])
    ori = np.eye(3)[[0, 1, 2, 0]]
    with h5py.File(f, "w") as h:
        g = h.create_group("grad")
        for key, arr in (("coilpos", pos), ("coilori", ori)):
            ds = g.create_dataset(key, data=arr.T)
            ds.attrs["MATLAB_class"] = np.bytes_(b"double")
        u = g.create_dataset("unit", data=np.array([[ord("m")]], dtype=np.uint16))
        u.attrs["MATLAB_class"] = np.bytes_(b"char")
    raw = read_array_file(f)
    np.testing.assert_allclose(raw.positions, pos * 1000.0)
    np.testing.assert_allclose(raw.orientations, ori)


def test_fieldtrip_elec(tmp_path) -> None:
    f = tmp_path / "elec.mat"
    scipy.io.savemat(f, {"elec": {"elecpos": [[10.0, 0, 0], [0, 10.0, 0]], "unit": "cm"}})
    raw = read_array_file(f, kind="eeg")
    np.testing.assert_allclose(raw.positions, [[100.0, 0, 0], [0, 100.0, 0]])
    assert raw.orientations is None


def test_fieldtrip_elec_chanpos_fallback(tmp_path) -> None:
    f = tmp_path / "elec.mat"
    scipy.io.savemat(f, {"elec": {"chanpos": [[0.1, 0, 0]], "unit": "m"}})
    np.testing.assert_allclose(read_array_file(f, kind="eeg").positions, [[100.0, 0, 0]])


# ── magnetometer import ───────────────────────────────────────────────────


def test_zero_orientation_is_refused(skin) -> None:
    pos, _ = _ring(2)
    raw = RawArray(pos, np.array([[1.0, 0, 0], [0.0, 0, 0]]), "test")
    with pytest.raises(SchemaError, match="zero length"):
        import_magnetometers(raw, skin=skin)


def test_unnormalised_orientation_is_normalised_with_a_warning(skin, caplog) -> None:
    pos, nrm = _ring(2)
    raw = RawArray(pos, 3.0 * nrm, "test")
    with caplog.at_level(logging.WARNING):
        arr = import_magnetometers(raw, skin=skin)
    np.testing.assert_allclose(np.linalg.norm(arr.coilori, axis=1), 1.0)
    assert "normalised" in caplog.text


def test_single_axis_import_is_labelled_radial(skin, caplog) -> None:
    pos, nrm = _ring(4)
    with caplog.at_level(logging.WARNING):
        arr = import_magnetometers(RawArray(pos, nrm, "test"), skin=skin)
    validate_sensors(arr)
    assert arr.labels == ("mag-0001-R", "mag-0002-R", "mag-0003-R", "mag-0004-R")
    assert array_layout(arr.labels) == SINGLE_AXIS
    assert set(arr.chantype) == {"megmag"}
    assert caplog.text == ""


def test_tangential_single_axis_is_reported(skin, caplog) -> None:
    pos, _ = _ring(4)
    tangential = np.tile([0.0, 0.0, 1.0], (4, 1))
    with caplog.at_level(logging.WARNING):
        import_magnetometers(RawArray(pos, tangential, "test"), skin=skin)
    assert "off the skin" in caplog.text


def test_triaxial_import_is_written_in_block_layout(skin) -> None:
    """Interleaved triplets become R/T1/T2 blocks, R nearest the skin normal."""
    pos, nrm = _ring(3)
    t1 = np.column_stack([-nrm[:, 1], nrm[:, 0], np.zeros(3)])
    t2 = np.tile([0.0, 0.0, 1.0], (3, 1))
    # File order per position: T2, R, T1 — R is deliberately not first.
    rows_pos = np.repeat(pos, 3, axis=0)
    rows_ori = np.stack([t2, nrm, t1], axis=1).reshape(-1, 3)
    arr = import_magnetometers(RawArray(rows_pos, rows_ori, "test"), skin=skin)
    validate_sensors(arr)
    assert array_layout(arr.labels) == TRIAXIAL
    np.testing.assert_allclose(arr.coilori[:3], nrm)
    np.testing.assert_allclose(arr.coilpos[:3], pos)
    np.testing.assert_allclose(arr.coilpos[3:6], pos)
    # The other two keep their file order: T2 came before T1 in the file.
    np.testing.assert_allclose(arr.coilori[3:6], t2)
    np.testing.assert_allclose(arr.coilori[6:9], t1)
    assert require_triaxial(arr.labels, "test") == 3


def test_dual_axis_import_keeps_file_order(skin, caplog) -> None:
    pos, nrm = _ring(2)
    rows_pos = np.repeat(pos, 2, axis=0)
    rows_ori = np.stack([nrm, np.tile([0.0, 0, 1], (2, 1))], axis=1).reshape(-1, 3)
    with caplog.at_level(logging.WARNING):
        arr = import_magnetometers(RawArray(rows_pos, rows_ori, "test"), skin=skin)
    assert arr.labels == ("mag-0001-A1", "mag-0001-A2", "mag-0002-A1", "mag-0002-A2")
    assert array_layout(arr.labels) == OTHER
    assert "will refuse" in caplog.text
    with pytest.raises(ValueError, match="triaxial"):
        require_triaxial(arr.labels, "the placement comparison")
    with pytest.raises(ValueError, match="no radial"):
        radial_channels(arr.labels)
    np.testing.assert_array_equal(position_channels(arr.labels, arr.coilpos), [0, 2])


def test_coils_inside_the_skin_are_reported(skin, caplog) -> None:
    pos, nrm = _ring(2, r=RADIUS - 20.0)
    with caplog.at_level(logging.WARNING):
        import_magnetometers(RawArray(pos, nrm, "test"), skin=skin)
    assert "inside the skin" in caplog.text


def test_coils_far_from_the_skin_are_reported(skin, caplog) -> None:
    """An array in metres read as mm lands nowhere near the body."""
    pos, nrm = _ring(2, r=RADIUS * 1000.0)
    with caplog.at_level(logging.WARNING):
        import_magnetometers(RawArray(pos, nrm, "test"), skin=skin)
    assert "from the skin" in caplog.text


def test_import_without_skin_still_builds(caplog) -> None:
    pos, nrm = _ring(2)
    with caplog.at_level(logging.WARNING):
        arr = import_magnetometers(RawArray(pos, nrm, "test"), skin=None)
    assert "checks skipped" in caplog.text
    validate_sensors(arr)


# ── electrode import ──────────────────────────────────────────────────────


def test_electrode_import_uses_skin_normals(skin) -> None:
    pos, nrm = _ring(3, r=RADIUS)
    arr = import_electrodes(RawArray(pos, None, "test"), skin=skin)
    validate_sensors(arr)
    assert arr.labels == ("elec-0001", "elec-0002", "elec-0003")
    assert set(arr.chantype) == {"eeg"} and set(arr.chanunit) == {"V"}
    assert (np.einsum("ij,ij->i", arr.coilori, nrm) > 0.95).all()


def test_electrodes_off_the_skin_are_reported(skin, caplog) -> None:
    pos, _ = _ring(2, r=RADIUS + 30.0)
    with caplog.at_level(logging.WARNING):
        import_electrodes(RawArray(pos, None, "test"), skin=skin)
    assert "from the skin" in caplog.text


# ── end to end through the config ─────────────────────────────────────────


def _cfg_with_skin(tmp_path: Path, skin: trimesh.Trimesh, overrides=None):
    stl = tmp_path / "data" / "torso" / "skin.stl"
    stl.parent.mkdir(parents=True)
    skin.export(stl)
    return load_config(TINY_CFG, project_root=tmp_path, overrides=overrides or [])


def test_generate_sensor_array_from_file_writes_the_generated_schema(tmp_path, skin) -> None:
    cfg = _cfg_with_skin(tmp_path, skin)
    pos, nrm = _ring(4)
    f = tmp_path / "mine.csv"
    np.savetxt(f, np.hstack([pos, nrm]), delimiter=",", header="x,y,z,nx,ny,nz")
    out = generate_sensor_array(cfg, from_file=f)
    assert out == cfg.outputs.sensors_mat
    arr = load_sensors(out)
    np.testing.assert_allclose(arr.coilpos, pos)
    np.testing.assert_allclose(arr.coilori, nrm)
    assert arr.unit == "mm"
    assert array_layout(arr.labels) == SINGLE_AXIS


def test_sensors_from_file_config_key_resolves_against_the_project_root(tmp_path, skin) -> None:
    pos, nrm = _ring(2)
    np.savez(tmp_path / "mine.npz", coil_pos=pos, coil_orient=nrm)
    cfg = _cfg_with_skin(tmp_path, skin, ["sensors.from_file=mine.npz"])
    arr = load_sensors(generate_sensor_array(cfg))
    np.testing.assert_allclose(arr.coilpos, pos)


def test_generate_electrode_array_from_file(tmp_path, skin) -> None:
    cfg = _cfg_with_skin(tmp_path, skin)
    pos, _ = _ring(3, r=RADIUS)
    f = tmp_path / "cap.csv"
    np.savetxt(f, pos, delimiter=",")
    arr = load_sensors(generate_electrode_array(cfg, from_file=f))
    np.testing.assert_allclose(arr.coilpos, pos)
    assert set(arr.chantype) == {"eeg"}


def test_radial_axes_generates_one_coil_per_position(tmp_path) -> None:
    box = trimesh.creation.box(extents=(200.0, 200.0, 200.0))
    tri_cfg = _cfg_with_skin(tmp_path, box)
    rad_cfg = load_config(TINY_CFG, project_root=tmp_path, overrides=["sensors.axes=radial"])
    tri = load_sensors(generate_sensor_array(tri_cfg))
    rad = load_sensors(generate_sensor_array(rad_cfg))
    n = len(rad.coilpos)
    assert len(tri.coilpos) == 3 * n
    assert array_layout(rad.labels) == SINGLE_AXIS
    # The radial array is exactly the R block of the triaxial one.
    np.testing.assert_allclose(rad.coilpos, tri.coilpos[:n])
    np.testing.assert_allclose(rad.coilori, tri.coilori[:n])
    assert rad.labels == tri.labels[:n]


def test_default_axes_is_triaxial_and_unknown_axes_is_refused(tmp_path) -> None:
    assert load_config(TINY_CFG, project_root=tmp_path).sensors.axes == "triaxial"
    with pytest.raises(ConfigError, match="axes"):
        load_config(TINY_CFG, project_root=tmp_path, overrides=["sensors.axes=biaxial"])


# ── layout helpers ────────────────────────────────────────────────────────


def test_layout_of_generated_arrays() -> None:
    pos, nrm = _ring(3)
    tri = build_triaxial(pos, nrm)
    rad = build_radial(pos, nrm)
    assert array_layout(tri.labels) == TRIAXIAL
    assert array_layout(rad.labels) == SINGLE_AXIS
    np.testing.assert_array_equal(radial_channels(tri.labels), [0, 1, 2])
    np.testing.assert_array_equal(radial_channels(rad.labels), [0, 1, 2])
    np.testing.assert_array_equal(position_channels(rad.labels, rad.coilpos), [0, 1, 2])


def test_single_axis_array_of_three_positions_is_not_mistaken_for_triaxial() -> None:
    """Three radial coils divide by three; the labels, not the count, decide."""
    pos, nrm = _ring(3)
    rad = build_radial(pos, nrm)
    with pytest.raises(ValueError, match="single-axis"):
        require_triaxial(rad.labels, "the placement comparison")


def test_sensor_field_maps_every_channel_of_a_single_axis_array() -> None:
    """The field map used to take the first third of any array as radial."""
    from types import SimpleNamespace

    from inob.viz.sensor_field import _radial_idx

    pos, nrm = _ring(6)
    rad = build_radial(pos, nrm)
    lf = SimpleNamespace(channel_names=rad.labels, coil_pos=rad.coilpos)
    np.testing.assert_array_equal(_radial_idx(lf), np.arange(6))


# ── noise presets ─────────────────────────────────────────────────────────


def test_squid_preset_is_selectable_and_marked_unverified(tmp_path) -> None:
    from inob.sensors.opm_presets import DEFAULT_OPM_SENSOR, opm_sensor

    squid = opm_sensor("megin_triux_mag")
    assert not squid.verified
    assert "NM24132A" in squid.source
    assert DEFAULT_OPM_SENSOR == "quspin_qzfm3"
    cfg = load_config(
        TINY_CFG, project_root=tmp_path, overrides=["noise.opm_sensor=megin_triux_mag"]
    )
    assert cfg.noise.opm_intrinsic_fT_sqrtHz == squid.noise_fT_sqrtHz
    assert cfg.noise.opm_bandwidth_hz == squid.bandwidth_hz


def test_skin_offsets_trusts_winding_on_an_open_concave_skin() -> None:
    """An open but consistently wound skin must not be judged by its centroid.

    An L-shaped prism, opened by deleting its end caps like the shipped skin's
    cut ends: the faces of the inner corner point back towards the centroid,
    so orienting normals away from it put a point 10 mm outside those faces
    10 mm inside.
    """
    ring = np.array([(0, 0), (200, 0), (200, 60), (60, 60), (60, 200), (0, 200)], float)
    n = len(ring)
    verts = np.vstack(
        [np.column_stack([ring, np.zeros(n)]), np.column_stack([ring, np.full(n, 100.0)])]
    )
    faces = []
    for i in range(n):  # walls only; the ring is counter-clockwise, so this winds outward
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    skin = trimesh.Trimesh(verts, np.array(faces), process=False)
    assert skin.is_winding_consistent and not skin.is_watertight
    # 10 mm outside the inner-corner walls (x = 60 and y = 60).
    pts = np.array([[70.0, 130.0, 50.0], [130.0, 70.0, 50.0]])
    signed, _ = skin_offsets(skin, pts)
    np.testing.assert_allclose(signed, [10.0, 10.0], atol=1e-6)
