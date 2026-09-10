"""Propagating compound action potential (CAP) source model.

Each fibre is represented as a moving current dipole travelling along the
nerve at its
fibre-diameter-dependent conduction velocity (CV). The recorded signal is a
fibre-population integral of these dipoles over time, weighted by the
fibre-diameter histogram.

Approximations
--------------
* Geometry: the vagus path is the polyline of source positions
  (``vagus_sources``, sampled every ``source_spacing_mm``).
* Dipole orientation: tangent to the polyline at each location (the
  "longitudinal" propagating dipole). For OPM forward modelling this is
  the dominant component; transverse fibre asymmetry is neglected.
* Fibre population: lognormal (or arbitrary user-supplied) PDF over fibre
  diameter D (µm). CV ≈ k * D for myelinated fibres (Hursh 1939, k ≈ 6
  m/s/µm for large myelinated; Pelot et al. 2017 for vagal-specific data).
* Action-potential waveform: biphasic (Hermite-Gaussian) of width
  ``ap_width_ms``. This is the "shape" the dipole moment traces out as the
  AP passes a station.

For the leadfield we don't need the time waveform directly — the spatial
leadfield ``L`` already maps a fixed source location to all sensor channels.
The CAP module instead generates *temporal* signal predictions:

    s_c(t) = Σ_d w(d) Σ_x L_long(c, x) * shape(t - x/CV(d))

where x is arc length along the nerve, CV(d) the diameter-dependent
velocity, w(d) the fibre-diameter histogram, and ``L_long`` the
longitudinal-component leadfield.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FibreDistribution:
    diameters_um: np.ndarray  # (D,) sample points
    weights: np.ndarray  # (D,) normalised weights, sum to 1


def lognormal_fibre_distribution(
    *,
    mean_um: float = 4.0,
    sigma_log: float = 0.5,
    n_bins: int = 30,
    lo_um: float = 0.5,
    hi_um: float = 15.0,
) -> FibreDistribution:
    """A lognormal PDF over fibre diameter [µm].

    Defaults very roughly approximate vagal A-fibre population (large
    myelinated). Override with measured data per Pelot 2017 for a publishable
    figure.
    """
    edges = np.linspace(lo_um, hi_um, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mu = np.log(mean_um) - 0.5 * sigma_log**2
    pdf = (1.0 / (centres * sigma_log * np.sqrt(2.0 * np.pi))) * np.exp(
        -((np.log(centres) - mu) ** 2) / (2.0 * sigma_log**2)
    )
    weights = pdf * np.diff(edges)
    weights = weights / weights.sum()
    return FibreDistribution(diameters_um=centres, weights=weights)


def hamalainen_per_fibre_nAm(
    fibres: FibreDistribution,
    *,
    action_potential_mV: float = 70.0,
    sigma_intracellular_S_per_m: float = 1.0,
) -> float:
    """Population-averaged per-fibre dipole moment, nA·m.

        Q(d) = π · d² · σ_in · ΔV / 4         (Hämäläinen et al. 1993, eq. 31)

    Returned value is the weighted mean over the diameter histogram.

    Choice of σ_in
    --------------
    The default ``σ_in = 1.0 S/m`` is the canonical Hämäläinen / Plonsey value
    for intracellular (cortical-axoplasm) conductivity, used throughout the
    MEG / inverse-problem literature for current-dipole moment derivations
    (Hämäläinen et al. 1993 *Rev Mod Phys* 65:413, Section IV.A). For
    *peripheral* unmyelinated and myelinated axons specifically, Pelot et al.
    2017 *Front Neurosci* 12:601 cite a lower value (≈ 0.35 S/m) consistent
    with classical squid-axon measurements (Hodgkin & Huxley 1952). Switching
    to the Pelot value scales every Q by ≈ 0.35×.

    For consistency with the Sarvas / Bu et al. 2024 cervical-vagus
    benchmark (which itself uses Hämäläinen σ_in = 1 S/m to derive the
    "≈ 70 nA·m at full A+C summation" reference), this codebase uses
    σ_in = 1 S/m as the default. Pass ``sigma_intracellular_S_per_m=0.35``
    to re-derive every figure under the Pelot convention.
    """
    d_m = np.asarray(fibres.diameters_um) * 1e-6
    Q_per_diameter_Am = (
        np.pi * d_m**2 * sigma_intracellular_S_per_m * (action_potential_mV * 1e-3) / 4.0
    )
    Q_mean_Am = float(np.sum(Q_per_diameter_Am * fibres.weights))
    return Q_mean_Am * 1e9  # → nA·m


def conduction_velocity_m_per_s(
    diameters_um: np.ndarray,
    *,
    k_m_per_s_per_um: float = 6.0,
    myelinated_threshold_um: float = 1.5,
    c_unmyelinated: float = 0.5,
) -> np.ndarray:
    """Diameter → CV. Myelinated: CV ≈ k·D.  Unmyelinated (D < threshold):
    constant ``c_unmyelinated`` m/s (~C-fibre baseline)."""
    D = np.asarray(diameters_um, dtype=np.float64)
    cv = np.where(
        D >= myelinated_threshold_um, k_m_per_s_per_um * D, np.full_like(D, c_unmyelinated)
    )
    return cv


def biphasic_waveform(
    t_ms: np.ndarray,
    *,
    ap_width_ms: float = 0.5,
) -> np.ndarray:
    """Biphasic (1st derivative of Gaussian) action-potential shape.

    Normalised to unit peak. Width parameter ``ap_width_ms`` is the σ
    of the underlying Gaussian.
    """
    t = np.asarray(t_ms, dtype=np.float64)
    g = np.exp(-(t**2) / (2.0 * ap_width_ms**2))
    out = -t / ap_width_ms * g
    out /= max(np.abs(out).max(), 1e-30)
    return out


def longitudinal_leadfield(
    L: np.ndarray,
    source_pos_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a per-moment leadfield to the polyline-tangent direction.

    Inputs:
      L (C, 3*S) — leadfield with three orthogonal moments per source.
      source_pos_mm (S, 3) — source positions in arc-length order.

    Returns:
      L_long (C, S)  — leadfield for a unit longitudinal dipole at each source.
      arc_length_mm (S,) — cumulative arc length along the polyline.
      tangents (S, 3) — unit tangents at each source.
    """
    C, three_S = L.shape
    S = three_S // 3
    if 3 * S != three_S or len(source_pos_mm) != S:
        raise ValueError(f"L second dim {three_S} not 3 * len(source_pos) ({len(source_pos_mm)})")
    L3 = L.reshape(C, S, 3)

    diff = np.diff(source_pos_mm, axis=0)
    seg = np.linalg.norm(diff, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    tangents = np.empty_like(source_pos_mm)
    tangents[:-1] = diff / np.maximum(seg[:, None], 1e-12)
    tangents[-1] = tangents[-2]
    # central differences for interior nodes, smoother:
    tangents[1:-1] = 0.5 * (tangents[1:-1] + tangents[:-2])
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)

    L_long = np.einsum("csm,sm->cs", L3, tangents)
    return L_long, arc, tangents


def cap_signal(
    L: np.ndarray,
    source_pos_mm: np.ndarray,
    *,
    fibres: FibreDistribution,
    ap_width_ms: float = 0.5,
    fs_hz: float = 30_000.0,
    duration_ms: float = 30.0,
    cv_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the time-domain CAP per sensor channel.

    Returns ``(t_ms, signal)`` where ``signal`` has shape ``(C, n_samples)``.

    The arc-length integral assumes uniform fibre activation density along
    the nerve (each diameter bin contributes equally per unit length).
    Multiply by an excitation profile in arc length if you have one.
    """
    L_long, arc_mm, _ = longitudinal_leadfield(L, source_pos_mm)
    arc_m = arc_mm * 1e-3
    n = round(duration_ms * fs_hz / 1000.0)
    t_ms = np.arange(n) / fs_hz * 1000.0
    signal = np.zeros((L_long.shape[0], n), dtype=np.float64)
    centre_ms = duration_ms * 0.4  # peak AP arrival in centre of window
    for d, w in zip(fibres.diameters_um, fibres.weights, strict=True):
        v = float(conduction_velocity_m_per_s(np.array([d]), **(cv_kwargs or {}))[0])
        for s, x_m in enumerate(arc_m):
            t_arrival = centre_ms + (x_m / v) * 1000.0
            shape = biphasic_waveform(t_ms - t_arrival, ap_width_ms=ap_width_ms)
            signal += w * L_long[:, s][:, None] * shape[None, :]
    # Approximate 1/n_sources spatial-density correction so CAP magnitude
    # is independent of source spacing (per arc-length, not per source).
    if len(arc_m) > 1:
        signal *= float(np.diff(arc_m).mean())
    return t_ms, signal
