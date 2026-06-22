"""Sarvas (analytic Biot–Savart) vs FEM comparison for the cervical vagus.

Method (after Bu et al. 2024, Comm Biol; and the concentric-circle
parameters in their Sarvas calibration figure):

  * Approximate the cervical region locally (per source slice) as a sphere
    centred on the cervical axis at *the source's own Z*. The literature
    parameters they validate against are:

        source–to–axis distance  ≈ 40   mm
        sensor–to–axis distance  ≈ 52   mm  (skin) +  6.5 mm (QuSpin standoff)
                                  = 58.5 mm

    A *single fixed* sphere centre for the whole cervical column is wrong —
    it makes far-away sources have artificially huge "axis distances" and
    inflates the Sarvas amplitude for them. The correct construction places
    the sphere centre on the cervical axis at the same Z as each source,
    so the source-axis distance is purely the transverse offset.

  * Sarvas (1987) closed-form: the magnetic field outside a homogeneous
    sphere from an internal current dipole. Provided the source is *inside*
    the sphere and the sensor *outside*, the result is independent of the
    sphere's conductivity (Geselowitz reciprocity).

  * Compound dipole moment via Hämäläinen et al. 1993:
        Q = π · d² · σ_in · ΔV / 4
    Bu et al. 2024 combine myelinated A-fibre (~70 mV) and unmyelinated
    C-fibre (~80 mV) contributions to ~70 nA·m total (all axons firing) —
    yielding a Sarvas peak of ~9 pT and 1–4 pT in observed subjects.

What we compare
---------------
For each FEM source position along the cervical vagus and each OPM coil:

  * Sarvas magnetic-field vector at the coil location, dotted with the coil
    orientation → predicted scalar reading.
  * FEM-computed scalar reading (same coil orientation) at unit dipole moment,
    rescaled to the same Q.
  * Per-channel and per-source residuals.

The model is intentionally crude (single-sphere Sarvas vs full multi-tissue
FEM) — its purpose is to ground the FEM amplitudes against the literature's
analytic baseline and quantify the divergence introduced by the body.

References
----------
* Sarvas J. (1987). Phys Med Biol 32:11–22. https://doi.org/10.1088/0031-9155/32/1/004
* Hämäläinen M, Hari R, Ilmoniemi RJ, Knuutila J, Lounasmaa OV (1993).
  Rev Mod Phys 65:413. https://doi.org/10.1103/RevModPhys.65.413
* Bu Y et al. (2024). "Non-invasive ventral cervical magnetoneurography
  as a proxy of in vivo lipopolysaccharide-induced inflammation."
  Comm Biol 7:893. https://doi.org/10.1038/s42003-024-06435-8
* O'Neill GC, Spedden ME, Schmidt M, Mellor S, Stenroos M, Barnes GR (2025).
  "Volume conductor models for magnetospinography." Sci Rep 15:26258.
  https://doi.org/10.1038/s41598-025-10770-z
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vagus_fm.analysis.analytic_sphere import sarvas_meg_field
from vagus_fm.config import Config
from vagus_fm.io.hdf5 import load_fem, load_sensors
from vagus_fm.io.npz import load_leadfield
from vagus_fm.sources.vagus import vagus_sources

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarvasGeometry:
    """Per-source concentric-circle approximation of the cervical region.

    Each source uses its OWN sphere centre — at the same Z as the source,
    on the cervical axis (estimated from the bone/source XY-centroid). A
    single fixed centre across the whole cervical column produces wrong
    distances for sources well above/below the chosen reference Z.
    """

    sphere_centres_mm: np.ndarray   # (S, 3) — one centre per source position
    axis_xy_mm: np.ndarray          # (2,) — the cervical-axis (X, Y) centre used
    source_axis_mm: float = 40.0   # nominal source distance from the cervical axis
    sensor_axis_mm: float = 58.5   # nominal sensor distance from the cervical axis
                                   # (52 mm skin + 6.5 mm QuSpin standoff)


def hamalainen_dipole_moment_nAm(
    *, fibre_diameter_um: float, sigma_intracellular_S_per_m: float,
    action_potential_mV: float,
) -> float:
    """Hämäläinen current-dipole moment for one axon, in nA·m.

        Q = π · d² · σ_in · ΔV / 4

    Inputs in µm, S/m, mV → output in nA·m.

    Reference: Hämäläinen et al. 1993 (Rev Mod Phys 65:413). The canonical
    ``σ_in = 1 S/m`` for cortical axoplasm is used throughout the MEG / SQUID
    inverse-problem literature and is also the value used by Bu et al.
    2024 (*Comm Biol* 7:893) to derive the cervical-vagus 70 nA·m
    full-summation reference. Pelot et al. 2020 (*Front Neurosci* 12:601)
    cite a lower ≈ 0.35 S/m for peripheral axons (matching squid-axon
    measurements); see :func:`vagus_fm.sources.cap.hamalainen_per_fibre_nAm`
    for a discussion. We use σ_in = 1 S/m as the default for benchmark
    consistency with Bu 2024.
    """
    d_m = fibre_diameter_um * 1e-6                # m
    dV_V = action_potential_mV * 1e-3             # V
    Q_Am = np.pi * d_m ** 2 * sigma_intracellular_S_per_m * dV_V / 4.0
    return float(Q_Am * 1e9)                       # → nA·m


def estimate_cervical_axis_xy(fem, src_pos_mm: np.ndarray) -> np.ndarray:
    """Estimate the (X, Y) coordinates of the cervical axis from the FEM.

    The Z is *not* fixed: each source uses its own Z to build a moving
    sphere centre (see :func:`build_sphere_centres`). For (X, Y) we average
    the XY-centroid of the bone tets *and* the source polyline within the
    cervical band the source spans, biasing toward the spinal column rather
    than the asymmetrically-placed vagus polyline.

    Returns a single ``(2,)`` array.
    """
    z_lo, z_hi = float(src_pos_mm[:, 2].min()), float(src_pos_mm[:, 2].max())
    if "bone" in fem.tissue_labels:
        bone_id = fem.label_to_id["bone"]
        mask = fem.tissue == bone_id
        if mask.any():
            cents = fem.nodes[fem.tets[mask]].mean(axis=1)
            band = (cents[:, 2] >= z_lo) & (cents[:, 2] <= z_hi)
            if band.any():
                bone_xy = cents[band, :2].mean(axis=0)
                src_xy = src_pos_mm[:, :2].mean(axis=0)
                # 2:1 weight toward bone (the actual spinal axis) vs source polyline.
                return (2 * bone_xy + src_xy) / 3
    return src_pos_mm[:, :2].mean(axis=0)


def build_sphere_centres(
    src_pos_mm: np.ndarray, axis_xy_mm: np.ndarray,
) -> np.ndarray:
    """One sphere centre per source: ``[axis_x, axis_y, source_z]``.

    Implements the per-source moving-sphere approximation. Each Sarvas
    field is then computed in a local frame where the source-axis distance
    is the source's *transverse* offset from the cervical axis only.
    """
    axis_xy = np.asarray(axis_xy_mm, dtype=np.float64).reshape(2)
    centres = np.empty_like(src_pos_mm)
    centres[:, 0] = axis_xy[0]
    centres[:, 1] = axis_xy[1]
    centres[:, 2] = src_pos_mm[:, 2]
    return centres


def sarvas_predict_at_coils(
    source_pos_mm: np.ndarray,
    moment_direction: np.ndarray,
    Q_nAm: float,
    coil_pos_mm: np.ndarray,
    coil_orient: np.ndarray,
    *,
    sphere_centre_mm: np.ndarray,
) -> np.ndarray:
    """Sarvas analytic field at every coil for one source position.

    Returns ``(C,)`` scalar field projected on each coil's orientation, in
    Tesla. Distances are converted to metres internally.
    """
    r0_m = (np.asarray(source_pos_mm) - sphere_centre_mm) * 1e-3
    Q_dir = np.asarray(moment_direction, dtype=np.float64)
    Q_dir = Q_dir / max(float(np.linalg.norm(Q_dir)), 1e-12)
    Q_vec_Am = Q_dir * (Q_nAm * 1e-9)
    sensors_m = (np.asarray(coil_pos_mm) - sphere_centre_mm) * 1e-3
    B = sarvas_meg_field(r0_m, Q_vec_Am, sensors_m)        # (C, 3) Tesla
    # Project onto each coil's orientation (assumed unit-norm).
    return np.einsum("ij,ij->i", B, coil_orient)


def vagus_tangents(source_pos_mm: np.ndarray) -> np.ndarray:
    """Unit tangent at each polyline point (central differences)."""
    diff = np.diff(source_pos_mm, axis=0)
    seg = np.linalg.norm(diff, axis=1)
    tangents = np.empty_like(source_pos_mm)
    tangents[:-1] = diff / np.maximum(seg[:, None], 1e-12)
    tangents[-1] = tangents[-2]
    tangents[1:-1] = 0.5 * (tangents[1:-1] + tangents[:-2])
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    return tangents


@dataclass(frozen=True)
class SarvasVsFemResult:
    geometry: SarvasGeometry
    Q_nAm: float
    source_pos_mm: np.ndarray            # (S, 3)
    sarvas_T: np.ndarray                 # (C_radial, S) — projected onto coil normal
    fem_T: np.ndarray                    # (C_radial, S) — same projection
    coil_pos_mm: np.ndarray              # (C_radial, 3) — radial coils only
    coil_orient: np.ndarray              # (C_radial, 3)
    distance_to_axis_mm: np.ndarray      # (C_radial, S) — coil-axis transverse distance
                                         # in the per-source local sphere frame


def _radial_coil_mask(channel_names: list[str]) -> np.ndarray:
    """Mask the R (radial) coils from a triaxial OPM array."""
    n = len(channel_names)
    if n % 3 == 0 and channel_names[0].endswith("-R") and channel_names[n // 3].endswith("-T1"):
        mask = np.zeros(n, dtype=bool)
        mask[: n // 3] = True
        return mask
    return np.array([s.endswith("-R") for s in channel_names])


def compare_sarvas_vs_fem(
    cfg: Config,
    *,
    Q_nAm: float = 1.0,
    fibre_count: int | None = None,
) -> SarvasVsFemResult:
    """Build the Sarvas vs FEM comparison for the loaded MEG leadfield.

    Each FEM source's three orthogonal moments are projected onto the local
    vagus tangent (longitudinal), giving one scalar leadfield per coil per
    source. Sarvas is computed with the same dipole moment direction.

    ``Q_nAm`` defaults to **1 nA·m** so the figure's y-axis reads as a
    leadfield (fT per 1 nA·m of source dipole moment), matching every other
    plot in this repo. Pass ``Q_nAm=70`` (or another physiological value)
    to convert the y-axis directly to predicted real-CAP amplitudes in pT.
    """
    fem = load_fem(cfg.outputs.fem_mat)
    sensors = load_sensors(cfg.outputs.sensors_mat)
    lf = load_leadfield(cfg.outputs.forward_npz)

    # Source polyline + tangent moments
    src_pos = vagus_sources(
        fem, cfg.forward.source_tissue, spacing_mm=cfg.forward.source_spacing_mm,
    )
    tangents = vagus_tangents(src_pos)

    # Radial coils only (the QuSpin axes used in the user's measurement)
    radial = _radial_coil_mask(list(sensors.labels))
    coilpos = sensors.coilpos[radial]
    coilori = sensors.coilori[radial]

    # FEM leadfield in T per A·m → project onto tangent moment, rescale by Q
    L = lf.L                                  # (C_total, 3*S) Tesla per A·m
    L = L[radial]                             # (C_radial, 3*S)
    C, three_S = L.shape
    S = three_S // 3
    L3 = L.reshape(C, S, 3)
    fem_T = np.einsum("csm,sm->cs", L3, tangents)
    Q_per_fibre = float(Q_nAm) * (fibre_count or 1)
    fem_T = fem_T * (Q_per_fibre * 1e-9)        # → Tesla

    # Sarvas geometry — moving sphere, one centre per source on the cervical axis.
    axis_xy = estimate_cervical_axis_xy(fem, src_pos)
    centres = build_sphere_centres(src_pos, axis_xy)
    geom = SarvasGeometry(sphere_centres_mm=centres, axis_xy_mm=axis_xy)

    sarvas_T = np.zeros((C, S), dtype=np.float64)
    distances = np.zeros((C, S), dtype=np.float64)
    for s_idx in range(S):
        sarvas_T[:, s_idx] = sarvas_predict_at_coils(
            src_pos[s_idx], tangents[s_idx], Q_per_fibre,
            coilpos, coilori, sphere_centre_mm=centres[s_idx],
        )
        # Transverse axis distance for this source's local sphere frame.
        # Use full 3-D distance from the moving centre — the axis is vertical,
        # so any z-offset between coil and centre still puts the coil outside
        # a sphere of radius source_axis_mm.
        distances[:, s_idx] = np.linalg.norm(
            coilpos - centres[s_idx][None, :], axis=1,
        )

    src_axis_distances = np.linalg.norm(src_pos[:, :2] - axis_xy[None, :], axis=1)
    logger.info(
        "Sarvas geometry: cervical axis (X, Y) = (%.1f, %.1f) mm; moving centre per source.",
        float(axis_xy[0]), float(axis_xy[1]),
    )
    logger.info(
        "  source–axis distance: min=%.1f  median=%.1f  max=%.1f mm "
        "(literature: 40 mm)",
        float(src_axis_distances.min()),
        float(np.median(src_axis_distances)),
        float(src_axis_distances.max()),
    )
    logger.info(
        "  coil–axis distance:   min=%.1f  median=%.1f  max=%.1f mm "
        "(literature: 58.5 mm — 52 mm skin + 6.5 mm QuSpin standoff)",
        float(distances.min()),
        float(np.median(distances)),
        float(distances.max()),
    )
    scale_fT = 1e15 / max(Q_per_fibre, 1e-30)
    # Literature-band restriction: only (source, coil) pairs whose geometry
    # is within 30 mm of the literature 40 mm / 58.5 mm concentric-circle
    # values. Outside this band the single-sphere Sarvas is not a faithful
    # baseline (the homogeneous-sphere assumption breaks down further out).
    src_keep = np.abs(src_axis_distances - geom.source_axis_mm) <= 30.0
    coil_keep_per_src = np.abs(distances - geom.sensor_axis_mm) <= 30.0
    band = coil_keep_per_src & src_keep[None, :]
    n_band = int(band.sum())
    if n_band:
        sarvas_band_peak = np.max(np.abs(sarvas_T[band])) * scale_fT
        fem_band_peak = np.max(np.abs(fem_T[band])) * scale_fT
        logger.info(
            "Q = %g nA·m  ·  literature-band (n=%d pairs):  "
            "Sarvas peak %.2f fT/nAm (%.2f pT @ Q=70)  ·  "
            "FEM peak %.2f fT/nAm (%.2f pT @ Q=70)  ·  "
            "FEM/Sarvas = %.2fx",
            Q_per_fibre, n_band,
            sarvas_band_peak,
            sarvas_band_peak * 70.0 / 1000.0,
            fem_band_peak,
            fem_band_peak * 70.0 / 1000.0,
            fem_band_peak / max(sarvas_band_peak, 1e-30),
        )
    else:
        logger.warning(
            "No (source, coil) pairs match the literature concentric-circle "
            "geometry (40 ± 30 mm source-axis, 58.5 ± 30 mm coil-axis). "
            "Reporting full-array peaks instead."
        )
    logger.info(
        "Full-array peaks  ·  "
        "Sarvas %.2f fT/nAm  ·  FEM %.2f fT/nAm  ·  FEM/Sarvas = %.2fx",
        np.max(np.abs(sarvas_T)) * scale_fT,
        np.max(np.abs(fem_T)) * scale_fT,
        np.max(np.abs(fem_T)) / max(np.max(np.abs(sarvas_T)), 1e-30),
    )
    return SarvasVsFemResult(
        geometry=geom,
        Q_nAm=Q_per_fibre,
        source_pos_mm=src_pos,
        sarvas_T=sarvas_T,
        fem_T=fem_T,
        coil_pos_mm=coilpos,
        coil_orient=coilori,
        distance_to_axis_mm=distances,
    )


def save_comparison_summary(result: SarvasVsFemResult, out_path: Path) -> Path:
    """Write a small JSON with peak/rms statistics in both unit conventions.

    Reports both the *full-array* peak (across all radial coils) and the
    *literature-band* peak — restricted to (source, coil) pairs whose axis
    distances roughly match Bu et al. 2024's 40 mm / 58.5 mm assumption,
    which is the regime where the Sarvas formula is actually faithful.
    """
    import json
    Q = max(float(result.Q_nAm), 1e-30)
    scale_fT_per_nAm = 1.0e15 / Q
    geom = result.geometry
    src_axis_d = np.linalg.norm(
        result.source_pos_mm[:, :2] - geom.axis_xy_mm[None, :], axis=1,
    )

    # Literature-band: coils at 58.5 ± 30 mm, sources at 40 ± 30 mm.
    src_keep = np.abs(src_axis_d - geom.source_axis_mm) <= 30.0
    coil_keep_per_src = np.abs(result.distance_to_axis_mm - geom.sensor_axis_mm) <= 30.0
    band_mask = coil_keep_per_src & src_keep[None, :]

    def _peak(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
        a = arr if mask is None else arr[mask]
        return float(np.max(np.abs(a))) if a.size else 0.0

    def _rms(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
        a = arr if mask is None else arr[mask]
        return float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0

    # Bootstrap a 95% confidence interval on the FEM/Sarvas peak ratio. The
    # 6.8x peak headline is dominated by a small handful of best-aligned
    # (source, coil) pairs; resampling those pairs with replacement gives a
    # CI that reflects how stable the peak is against the specific pair set
    # rather than reading the single best pair as a point estimate.
    # See `_bootstrap_ratio_ci` for the (deterministic) resampling.
    if band_mask.any():
        ratio_lo, ratio_med, ratio_hi = _bootstrap_ratio_ci(
            sarvas_band=result.sarvas_T[band_mask],
            fem_band=result.fem_T[band_mask],
            n_boot=2000,
            seed=0,
        )
    else:
        ratio_lo = ratio_med = ratio_hi = float("nan")

    summary = {
        "Q_nAm": result.Q_nAm,
        "axis_xy_mm": geom.axis_xy_mm.tolist(),
        "geometry": "moving-sphere; one centre per source at [axis_x, axis_y, source_z]",
        "n_sources": int(result.source_pos_mm.shape[0]),
        "n_radial_coils": int(result.coil_pos_mm.shape[0]),
        "n_band_pairs": int(band_mask.sum()),
        # Full-array peaks — every (source, coil) pair, regardless of geometry fit.
        "sarvas_peak_fT_per_nAm_full": _peak(result.sarvas_T) * scale_fT_per_nAm,
        "fem_peak_fT_per_nAm_full":    _peak(result.fem_T)    * scale_fT_per_nAm,
        "sarvas_rms_fT_per_nAm_full":  _rms(result.sarvas_T)  * scale_fT_per_nAm,
        "fem_rms_fT_per_nAm_full":     _rms(result.fem_T)     * scale_fT_per_nAm,
        # Literature-band peaks — restricted to the regime where Sarvas (single
        # sphere with literature 40/58.5 mm radii) is a faithful model.
        "sarvas_peak_fT_per_nAm_band": _peak(result.sarvas_T, band_mask) * scale_fT_per_nAm,
        "fem_peak_fT_per_nAm_band":    _peak(result.fem_T,    band_mask) * scale_fT_per_nAm,
        # Predicted real-CAP signal at Q = 70 nA·m (Bu 2024 convention)
        "sarvas_peak_pT_at_Q70_band":
            _peak(result.sarvas_T, band_mask) * scale_fT_per_nAm * 70.0 / 1000.0,
        "fem_peak_pT_at_Q70_band":
            _peak(result.fem_T, band_mask) * scale_fT_per_nAm * 70.0 / 1000.0,
        "ratio_fem_to_sarvas_peak_band": (
            _peak(result.fem_T, band_mask)
            / max(_peak(result.sarvas_T, band_mask), 1e-30)
        ),
        # 95% bootstrap CI on the peak ratio.
        "ratio_fem_to_sarvas_peak_band_ci95": [ratio_lo, ratio_hi],
        "ratio_fem_to_sarvas_peak_band_bootstrap_median": ratio_med,
        "ratio_fem_to_sarvas_peak_band_bootstrap_n": 2000,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def _bootstrap_ratio_ci(
    *,
    sarvas_band: np.ndarray,
    fem_band: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI on max|fem| / max|sarvas| over the literature-band pairs.

    Returns ``(low, median, high)`` of the resampled ratio. Deterministic
    given ``seed``. Uses non-parametric resampling with replacement of
    (source, coil) pairs; the test statistic is the same peak/peak ratio as
    the point estimate.
    """
    rng = np.random.default_rng(seed)
    n = sarvas_band.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    s_abs = np.abs(np.asarray(sarvas_band).ravel())
    f_abs = np.abs(np.asarray(fem_band).ravel())
    ratios = np.empty(n_boot, dtype=np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    for b in range(n_boot):
        si = s_abs[idx[b]]
        fi = f_abs[idx[b]]
        denom = max(float(si.max()), 1e-30)
        ratios[b] = float(fi.max()) / denom
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(ratios, 100 * alpha))
    hi = float(np.percentile(ratios, 100 * (1.0 - alpha)))
    med = float(np.median(ratios))
    return lo, med, hi
