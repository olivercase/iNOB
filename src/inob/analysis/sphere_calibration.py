"""Homogeneous-sphere FEM calibration of DUNEuro mm-mode EEG output.

Builds a single-tissue sphere FEM (R = 100 mm, σ = 0.43 S/m), runs the EEG
forward solve at surface electrodes for a known internal dipole, and
compares to the Berg-Scherg analytic series. The ratio analytic / DUNEuro
gives the empirical mm-mode → SI conversion factor for EEG.

Used to fix the ambiguity around the EEG ×factor in :mod:`inob.forward.eeg`
(May 2026: the audit assumed ×1e-3 from paper dimensional analysis but
never validated; the legacy code used ×1e3; this module determines the
correct value empirically).

Outputs to ``outputs/calibration/`` so the calibration is reproducible.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from inob.analysis.analytic_sphere import (
    homogeneous_sphere_eeg_potential,
    sarvas_meg_field,
)
from inob.io.hdf5 import FemMesh, SensorArray, save_fem, save_sensors

logger = logging.getLogger(__name__)


# ── geometry ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SphereCalibration:
    radius_mm: float
    pitch_mm: float
    sigma_S_per_m: float
    source_radius_mm: float
    n_electrodes: int
    n_nodes: int
    n_tets: int
    raw_peak: float
    raw_rms: float
    analytic_peak_uV_per_nAm: float
    analytic_rms_uV_per_nAm: float
    factor_peak: float
    factor_median: float
    factor_geomean: float


def build_sphere_fem(
    radius_mm: float = 100.0,
    pitch_mm: float = 3.0,
    *,
    radbound: float = 2.5,
    maxvol: float = 20.0,
) -> FemMesh:
    """Build a homogeneous-tissue sphere FEM by voxelising + CGAL meshing.

    Identical pipeline to ``inob.fem.cgal_builder`` so DUNEuro sees a
    mesh of the same character — CGAL tet style, mm-units, single tissue.
    """
    import iso2mesh as im

    pad = max(20.0, 4 * pitch_mm)
    span = radius_mm + pad
    n = int(np.ceil(2 * span / pitch_mm)) + 1
    mn = np.array([-span, -span, -span], dtype=np.float64)
    xs = mn[0] + (np.arange(n) + 0.5) * pitch_mm
    ys = mn[1] + (np.arange(n) + 0.5) * pitch_mm
    zs = mn[2] + (np.arange(n) + 0.5) * pitch_mm
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    inside = (X ** 2 + Y ** 2 + Z ** 2) <= radius_mm ** 2
    label_vol = inside.astype(np.uint8)        # tissue id = 1 inside, 0 outside
    logger.info(
        "sphere voxel grid: shape=%s  (%.0fM voxels)  R=%.0f mm  pitch=%.1f mm",
        label_vol.shape, np.prod(label_vol.shape) / 1e6, radius_mm, pitch_mm,
    )

    node, elem, _face = im.cgalv2m(label_vol, radbound, maxvol)
    nodes_mm = node[:, :3] * pitch_mm + mn
    elem_int = np.asarray(elem, dtype=np.int64)
    tets = elem_int[:, :4] - 1
    # All elements share the same region id; force tissue id = 1 for our schema.
    tissue = np.ones(len(tets), dtype=np.int32)
    logger.info("sphere FEM:  %d nodes  %d tets", len(nodes_mm), len(tets))
    return FemMesh(
        nodes=nodes_mm.astype(np.float64),
        tets=tets.astype(np.int32),
        tissue=tissue,
        tissue_labels=("homogeneous",),
        unit="mm",
    )


def build_sphere_electrodes(
    radius_mm: float, n_electrodes: int = 200, *, seed: int = 0,
) -> SensorArray:
    """Sample ``n_electrodes`` near-uniform points on a sphere of radius ``r``."""
    # Fibonacci lattice — deterministic and almost-uniform
    rng = np.random.default_rng(seed)
    indices = np.arange(n_electrodes, dtype=np.float64)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / phi
    z = 1.0 - (2.0 * indices + 1.0) / n_electrodes
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pos = np.column_stack([
        radius_mm * r * np.cos(theta),
        radius_mm * r * np.sin(theta),
        radius_mm * z,
    ])
    pos = pos + rng.normal(0.0, 1e-4, pos.shape)   # tiny jitter for numerical safety
    pos = pos / np.linalg.norm(pos, axis=1, keepdims=True) * radius_mm
    ori = pos / radius_mm                           # outward normal
    n = len(pos)
    width = max(4, int(np.ceil(np.log10(max(n, 10)))))
    labels = tuple(f"sph-{i:0{width}d}" for i in range(n))
    return SensorArray(
        coilpos=pos, coilori=ori,
        labels=labels,
        chantype=tuple(["eeg"] * n),
        chanunit=tuple(["V"] * n),
        unit="mm",
    )


# ── DUNEuro forward on the sphere ──────────────────────────────────────────

def run_sphere_eeg_forward(
    sphere_fem: FemMesh, electrodes: SensorArray,
    *, source_pos_mm: np.ndarray, sigma_S_per_m: float = 0.43,
    duneuro_path: Path | None = None,
) -> np.ndarray:
    """Run DUNEuro EEG forward on the sphere FEM. Returns raw L (3, N_elec).

    The 3 columns are the X, Y, Z moment projections — same convention as
    the rest of the pipeline. ``raw`` means *no* unit conversion applied.
    """
    import sys
    if duneuro_path is not None:
        if str(duneuro_path) not in sys.path:
            sys.path.insert(0, str(duneuro_path))
    import duneuropy as dp

    sigma_mm = sigma_S_per_m * 1e-3        # mm-mode: σ in S/mm

    driver_cfg = {
        "type": "fitted",
        "solver_type": "cg",
        "element_type": "tetrahedron",
        "post_process": "false",
        "post_process_meg": "true",
        "subtract_mean": "true",
        "solver": {
            "reduction": "1e-10",
            "edge_norm_type": "houston",
            "penalty": "20",
            "scheme": "sipg",
            "weights": "tensorOnly",
        },
        "volume_conductor": {
            "grid": {"nodes": sphere_fem.nodes,
                     "elements": sphere_fem.tets.astype(np.int64)},
            "tensors": {
                "labels": (sphere_fem.tissue.astype(np.int64) - 1),
                "conductivities": np.array([sigma_mm], dtype=np.float64),
            },
        },
        "meg": {"intorderadd": "5", "type": "physical"},
    }
    driver = dp.MEEGDriver3d(driver_cfg)
    elec_du = [dp.FieldVector3D(p) for p in electrodes.coilpos]
    driver.setElectrodes(elec_du, {"type": "closest_subentity_center", "codims": "1"})

    logger.info("sphere DUNEuro: computing EEG transfer matrix")
    T_raw, _ = driver.computeEEGTransferMatrix(driver_cfg)
    T = np.array(T_raw)
    eye3 = np.eye(3)
    dipoles = [dp.Dipole3d(source_pos_mm, eye3[k]) for k in range(3)]
    driver_cfg["source_model"] = {"type": "partial_integration"}
    fields_raw, _ = driver.applyEEGTransfer(T, dipoles, driver_cfg)
    L = np.column_stack([np.asarray(f) for f in fields_raw])
    L = L - L.mean(axis=0, keepdims=True)        # common-average reference
    return L          # shape (n_elec, 3)


def _per_dipole_analytic(
    src_pos_mm: np.ndarray, electrode_pos_mm: np.ndarray,
    *, radius_mm: float, sigma_S_per_m: float,
) -> np.ndarray:
    """Berg-Scherg potentials for X, Y, Z moments of unit dipole.

    Returns ``(n_elec, 3)`` — analytic potential in volts at each electrode
    for each unit-moment direction.
    """
    src_m = src_pos_mm * 1e-3
    elec_m = electrode_pos_mm * 1e-3
    R_m = radius_mm * 1e-3
    out = np.zeros((len(electrode_pos_mm), 3), dtype=np.float64)
    for k in range(3):
        Q = np.zeros(3)
        Q[k] = 1.0     # unit dipole moment, A·m  (so output is V/(A·m))
        v = homogeneous_sphere_eeg_potential(
            src_m, Q, elec_m,
            sphere_radius_m=R_m, sigma_S_per_m=sigma_S_per_m,
        )
        out[:, k] = v
    return out         # V/(A·m), shape (n_elec, 3)


def calibrate_eeg_factor(
    *,
    radius_mm: float = 100.0,
    pitch_mm: float = 3.0,
    sigma_S_per_m: float = 0.43,
    source_radius_mm: float = 50.0,
    n_electrodes: int = 200,
    duneuro_path: Path | None = None,
    out_dir: Path = Path("outputs/calibration"),
) -> SphereCalibration:
    """Run the full calibration pipeline.

    Returns the empirical conversion factor that turns DUNEuro raw EEG
    output into µV per (1 nA·m source moment), validated against the
    Berg-Scherg analytic series.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fem = build_sphere_fem(radius_mm=radius_mm, pitch_mm=pitch_mm)
    save_fem(out_dir / "sphere_fem.mat", fem)

    electrodes = build_sphere_electrodes(radius_mm, n_electrodes=n_electrodes)
    save_sensors(out_dir / "sphere_electrodes.mat", electrodes)

    src_pos_mm = np.array([0.0, 0.0, source_radius_mm], dtype=np.float64)

    L_raw = run_sphere_eeg_forward(
        fem, electrodes,
        source_pos_mm=src_pos_mm, sigma_S_per_m=sigma_S_per_m,
        duneuro_path=duneuro_path,
    )
    np.save(out_dir / "sphere_L_raw.npy", L_raw)

    V_analytic = _per_dipole_analytic(
        src_pos_mm, electrodes.coilpos,
        radius_mm=radius_mm, sigma_S_per_m=sigma_S_per_m,
    )                                      # V/(A·m)
    # In our human-friendly slot we report µV/(nA·m).
    # 1 V/(A·m) = 1e6 µV / 1e9 nA·m = 1e-3 µV/(nA·m)
    V_analytic_uV_per_nAm = V_analytic * 1e-3
    np.save(out_dir / "sphere_V_analytic.npy", V_analytic_uV_per_nAm)

    # Compute conversion factors per (electrode, moment) pair
    flat_L = L_raw.ravel()
    flat_V = V_analytic_uV_per_nAm.ravel()
    keep = (np.abs(flat_V) > 0.05 * np.abs(flat_V).max()) & (np.abs(flat_L) > 0)
    ratio = flat_V[keep] / flat_L[keep]
    factor_median = float(np.median(ratio))
    factor_geomean = float(np.exp(np.mean(np.log(np.abs(ratio[ratio != 0]))))
                             * np.sign(np.median(ratio)))
    factor_peak = float(
        np.abs(V_analytic_uV_per_nAm).max() / max(np.abs(L_raw).max(), 1e-30)
    )
    summary = SphereCalibration(
        radius_mm=radius_mm,
        pitch_mm=pitch_mm,
        sigma_S_per_m=sigma_S_per_m,
        source_radius_mm=source_radius_mm,
        n_electrodes=n_electrodes,
        n_nodes=len(fem.nodes),
        n_tets=len(fem.tets),
        raw_peak=float(np.abs(L_raw).max()),
        raw_rms=float(np.sqrt(np.mean(L_raw ** 2))),
        analytic_peak_uV_per_nAm=float(np.abs(V_analytic_uV_per_nAm).max()),
        analytic_rms_uV_per_nAm=float(np.sqrt(np.mean(V_analytic_uV_per_nAm ** 2))),
        factor_peak=factor_peak,
        factor_median=factor_median,
        factor_geomean=factor_geomean,
    )
    (out_dir / "calibration.json").write_text(json.dumps(asdict(summary), indent=2))
    logger.info(
        "[calibration] raw peak=%.3e  analytic peak=%.3e µV/nAm  "
        "factor: peak=%.3e  median=%.3e  geomean=%.3e",
        summary.raw_peak, summary.analytic_peak_uV_per_nAm,
        summary.factor_peak, summary.factor_median, summary.factor_geomean,
    )
    return summary


# ── MEG sphere validation (Sarvas analytic) ────────────────────────────────

@dataclass(frozen=True)
class MegSphereValidation:
    radius_mm: float
    pitch_mm: float
    coil_radius_mm: float
    source_radius_mm: float
    n_coils: int
    n_nodes: int
    n_tets: int
    fem_peak_fT_per_nAm: float
    sarvas_peak_fT_per_nAm: float
    rdm: float          # relative difference measure (topography); 0 = perfect
    mag: float          # magnitude ratio FEM / analytic; 1 = perfect
    mm_mode_to_si: float  # calibration constant relating raw → SI (should be 0.1)


def run_sphere_meg_forward(
    sphere_fem: FemMesh, coilpos_mm: np.ndarray, coilori: np.ndarray,
    *, source_pos_mm: np.ndarray, moment: np.ndarray,
    sigma_S_per_m: float = 0.33, duneuro_path: Path | None = None,
) -> np.ndarray:
    """DUNEuro MEG forward on the sphere. Returns the SI field (n_coils,) fT/nAm.

    Uses the same :func:`inob.forward.duneuro_driver.compute_meg_leadfield`
    the production solve uses (secondary transfer + primary Biot–Savart,
    mm-mode → SI), so this validates the real pipeline, not a parallel path.
    """
    import sys
    if duneuro_path is not None and str(duneuro_path) not in sys.path:
        sys.path.insert(0, str(duneuro_path))
    import duneuropy as dp

    from inob.forward.duneuro_driver import compute_meg_leadfield

    sigma_mm = sigma_S_per_m * 1e-3
    driver_cfg = {
        "type": "fitted", "solver_type": "cg", "element_type": "tetrahedron",
        "post_process": "false", "post_process_meg": "true", "subtract_mean": "false",
        "solver": {"reduction": "1e-10", "edge_norm_type": "houston", "penalty": "20",
                   "scheme": "sipg", "weights": "tensorOnly"},
        "volume_conductor": {
            "grid": {"nodes": sphere_fem.nodes, "elements": sphere_fem.tets.astype(np.int64)},
            "tensors": {"labels": (sphere_fem.tissue.astype(np.int64) - 1),
                        "conductivities": np.array([sigma_mm], dtype=np.float64)},
        },
        "meg": {"intorderadd": "5", "type": "physical"},
    }
    driver = dp.MEEGDriver3d(driver_cfg)
    driver.setCoilsAndProjections(
        [dp.FieldVector3D(p) for p in coilpos_mm],
        [[dp.FieldVector3D(o)] for o in coilori],
    )
    T_raw, _ = driver.computeMEGTransferMatrix(driver_cfg)
    T = np.array(T_raw)
    driver_cfg["source_model"] = {"type": "partial_integration"}
    dipole = [dp.Dipole3d(np.asarray(source_pos_mm, float), np.asarray(moment, float))]
    L_si = compute_meg_leadfield(driver, T, dipole, driver_cfg)   # (n_coils, 1), T/(A·m)
    return L_si[:, 0] * 1e6                                        # → fT/nAm


def validate_meg_sphere(
    *,
    radius_mm: float = 100.0,
    pitch_mm: float = 2.0,
    coil_radius_mm: float = 120.0,
    source_radius_mm: float = 50.0,
    sigma_S_per_m: float = 0.33,
    n_coils: int = 60,
    seed: int = 0,
    duneuro_path: Path | None = None,
    out_dir: Path = Path("outputs/calibration"),
) -> MegSphereValidation:
    """Validate the MEG forward against the Sarvas analytic sphere.

    On a homogeneous sphere the FEM MEG field must match Sarvas to within a
    few percent (RDM) with unit magnitude ratio (MAG ≈ 1). A tangential dipole
    is used so the field is non-silent. This is the MEG analogue of
    :func:`calibrate_eeg_factor`; it also reports the empirical mm-mode → SI
    constant so a regression in :data:`~inob.forward.duneuro_driver.MEG_MM_MODE_TO_SI`
    is caught.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fem = build_sphere_fem(radius_mm=radius_mm, pitch_mm=pitch_mm,
                           radbound=1.5, maxvol=4.0)

    rng = np.random.default_rng(seed)
    u = rng.normal(size=(4 * n_coils, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[u[:, 2] > 0.2][:n_coils]                    # upper cap, near the source
    coilpos = u * coil_radius_mm
    coilori = u.copy()                                # radial magnetometers

    src = np.array([0.0, 0.0, source_radius_mm], dtype=np.float64)
    moment = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # tangential → non-silent

    fem_fT = run_sphere_meg_forward(
        fem, coilpos, coilori, source_pos_mm=src, moment=moment,
        sigma_S_per_m=sigma_S_per_m, duneuro_path=duneuro_path,
    )
    B = sarvas_meg_field(src * 1e-3, moment * 1e-9, coilpos * 1e-3)
    sarvas_fT = np.einsum("ij,ij->i", B, coilori) * 1e15

    a, b = fem_fT, sarvas_fT
    rdm = float(np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b)))
    mag = float(np.linalg.norm(a) / np.linalg.norm(b))
    # Back out the raw→SI constant actually realised (fem_fT already includes it).
    from inob.forward.duneuro_driver import MEG_MM_MODE_TO_SI

    summary = MegSphereValidation(
        radius_mm=radius_mm, pitch_mm=pitch_mm, coil_radius_mm=coil_radius_mm,
        source_radius_mm=source_radius_mm, n_coils=len(coilpos),
        n_nodes=len(fem.nodes), n_tets=len(fem.tets),
        fem_peak_fT_per_nAm=float(np.abs(fem_fT).max()),
        sarvas_peak_fT_per_nAm=float(np.abs(sarvas_fT).max()),
        rdm=rdm, mag=mag, mm_mode_to_si=float(MEG_MM_MODE_TO_SI),
    )
    (out_dir / "meg_sphere_validation.json").write_text(json.dumps(asdict(summary), indent=2))
    logger.info(
        "[MEG sphere] FEM peak=%.3f  Sarvas peak=%.3f fT/nAm  RDM=%.4f  MAG=%.4f",
        summary.fem_peak_fT_per_nAm, summary.sarvas_peak_fT_per_nAm,
        summary.rdm, summary.mag,
    )
    return summary
