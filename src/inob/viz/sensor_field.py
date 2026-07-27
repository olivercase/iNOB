"""Sensor-field pattern analysis for a source region (e.g. the spinal cord).

Given a solved MEG leadfield, this characterises *what the field looks like on
the sensor array* for a source in the region: the spatial (dipolar) pattern, how
strong/focal it is, and how it varies along the region's axis. Modality-agnostic
over source region — driven entirely by the leadfield's ``source_pos`` — so it
works for spine, vagus, or any future target.

Sensor convention: the OPM array stores 3 orientations per physical position;
the first third of channels are the radial (outward) component, which is what a
scalar OPM topography plots. We characterise the radial field of a source's
dominant moment orientation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.anatomy import vertebra_z_band as _anatomy_z_band
from inob.config import Config, target_output
from inob.io.npz import Leadfield, load_leadfield
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    divergent_cmap,
    divergent_norm,
    save_figure,
)

logger = logging.getLogger(__name__)


# ── leadfield accessors ─────────────────────────────────────────────────────

def _radial_idx(lf: Leadfield) -> np.ndarray:
    """Channel indices of the radial OPM component (first third of the array)."""
    return np.arange(lf.coil_pos.shape[0] // 3)


def _dominant_moment(lf: Leadfield, source_idx: int) -> int:
    """Which source-moment orientation (0=x,1=y,2=z) gives the largest radial field."""
    ri = _radial_idx(lf)
    peaks = [np.max(np.abs(lf.L_fT_per_nAm[ri, 3 * source_idx + m])) for m in range(3)]
    return int(np.argmax(peaks))


def radial_field(lf: Leadfield, source_idx: int, moment: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial-channel field (fT/nA·m) and sensor positions (mm) for one source-moment."""
    ri = _radial_idx(lf)
    return lf.L_fT_per_nAm[ri, 3 * source_idx + moment], lf.coil_pos[ri]


def peak_amplitude_along_axis(lf: Leadfield) -> tuple[np.ndarray, np.ndarray]:
    """Per-source (z_mm, best-moment peak radial |field|) for every source."""
    ri = _radial_idx(lf)
    L = lf.L_fT_per_nAm[ri]
    S = lf.source_pos.shape[0]
    peaks = np.array([
        max(np.max(np.abs(L[:, 3 * s + m])) for m in range(3)) for s in range(S)
    ])
    return lf.source_pos[:, 2], peaks


def aggregate_field(
    lf: Leadfield, moment: int, *, mode: str = "coherent",
    sources: np.ndarray | None = None,
) -> np.ndarray:
    """Field on the radial array from combined sources (unit moment each).

    By default combines ALL sources; pass ``sources`` (an index array) to
    restrict to a subset (e.g. only the C7 band).

    ``coherent`` sums the signed fields (region active in phase — the net
    topography you'd measure). ``rms`` takes the per-sensor RMS across sources
    (sign-agnostic sensitivity map — which sensors see the region at all).
    """
    ri = _radial_idx(lf)
    if sources is None:
        sources = np.arange(lf.source_pos.shape[0])
    cols = lf.L_fT_per_nAm[ri][:, [3 * s + moment for s in sources]]   # (n_sens, |sources|)
    if mode == "coherent":
        return cols.sum(axis=1)
    if mode == "rms":
        return np.sqrt(np.mean(cols ** 2, axis=1))
    raise ValueError(f"unknown aggregate mode {mode!r} (use 'coherent' or 'rms')")


# ── vertebral-level source selection ────────────────────────────────────────
#
# The level→STL lookup and Z-band derivation live in :mod:`inob.anatomy` so
# electrode placement can share them. Re-exported here for back-compat with
# callers importing ``VERTEBRA_LEVELS`` / ``vertebra_z_band`` from this module.


def vertebra_z_band(cfg: Config, level: str) -> tuple[float, float]:
    """Z bounding-box (z_lo, z_hi) mm of a vertebra from its segmented STL."""
    return _anatomy_z_band(cfg.data.bone_dir, level)


def sources_in_z_band(lf: Leadfield, z_lo: float, z_hi: float) -> np.ndarray:
    """Indices of cord sources whose Z lies within ``[z_lo, z_hi]`` (inclusive)."""
    z = lf.source_pos[:, 2]
    idx = np.flatnonzero((z >= z_lo) & (z <= z_hi))
    if idx.size == 0:
        raise ValueError(
            f"no sources in Z band [{z_lo:.1f}, {z_hi:.1f}] mm "
            f"(source Z range {z.min():.1f}..{z.max():.1f} mm)"
        )
    return idx


# ── characteristics ─────────────────────────────────────────────────────────

def sensor_field_characteristics(
    lf: Leadfield, source_idx: int, *, moment: int | None = None,
) -> dict:
    """Quantify the sensor field pattern of one source (dominant moment by default)."""
    if moment is None:
        moment = _dominant_moment(lf, source_idx)
    v, pos = radial_field(lf, source_idx, moment)
    src = lf.source_pos[source_idx]

    peak = float(np.max(np.abs(v)))
    dist = np.linalg.norm(pos - src, axis=1)
    # Dipolar-lobe separation: distance between the strongest +ve and -ve sensors.
    i_pos, i_neg = int(np.argmax(v)), int(np.argmin(v))
    lobe_sep = float(np.linalg.norm(pos[i_pos] - pos[i_neg]))
    # Focality: fraction of the array carrying >50 % of the peak (smaller = tighter).
    halfmax_frac = float(np.mean(np.abs(v) > 0.5 * peak))
    return {
        "source_idx": int(source_idx),
        "source_z_mm": float(src[2]),
        "dominant_moment": "xyz"[moment],
        "peak_fT_per_nAm": peak,
        "source_to_nearest_sensor_mm": float(dist.min()),
        "dipole_lobe_separation_mm": lobe_sep,
        "halfmax_extent_frac": halfmax_frac,
        "n_radial_sensors": len(v),
    }


def region_summary(lf: Leadfield) -> dict:
    """Region-wide summary: how peak field varies across the source axis."""
    z, peaks = peak_amplitude_along_axis(lf)
    order = int(np.argmax(peaks))
    return {
        "n_sources": int(lf.source_pos.shape[0]),
        "z_range_mm": [float(z.min()), float(z.max())],
        "peak_fT_per_nAm": {
            "min": float(peaks.min()), "median": float(np.median(peaks)),
            "max": float(peaks.max()),
        },
        "strongest_source_idx": order,
        "strongest_source_z_mm": float(z[order]),
    }


# ── figure ──────────────────────────────────────────────────────────────────

def _cylinder_theta_z(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unroll sensor positions to (azimuth°, z) about the array's vertical axis."""
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    theta = np.degrees(np.arctan2(pos[:, 1] - cy, pos[:, 0] - cx))
    return theta, pos[:, 2]


def render_aggregate_field(
    cfg: Config, *, moment: int = 2, mode: str = "coherent",
    z_range: tuple[float, float] | None = None, region_label: str | None = None,
    out_path: Path | None = None, dpi: int = 300,
) -> Path:
    """Global field pattern from region sources combined (see ``aggregate_field``).

    Combines all sources by default; pass ``z_range`` to restrict to a Z band
    (e.g. a single vertebral level such as C7).

    a  3-D OPM array coloured by the aggregate field, with the source line drawn.
    b  Unrolled cylinder (azimuth vs z) topography of the aggregate field.
    """
    apply_nature_style()
    lf = load_leadfield(cfg.outputs.forward_npz)
    sources = None if z_range is None else sources_in_z_band(lf, *z_range)
    v = aggregate_field(lf, moment, mode=mode, sources=sources)
    _, pos = radial_field(lf, 0, moment)          # positions only
    src = lf.source_pos if sources is None else lf.source_pos[sources]

    if mode == "coherent":
        cmap = divergent_cmap()
        vmin, vmax = divergent_norm(v)
        cbar_label = "Σ field (fT/nA·m)"
    else:
        cmap = plt.get_cmap("magma")
        vmin, vmax = 0.0, float(np.max(v))
        cbar_label = "RMS field (fT/nA·m)"

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, left=0.05, right=0.95, top=0.88,
                  bottom=0.10, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    p = ax_a.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=v, cmap=cmap,
                     vmin=vmin, vmax=vmax, s=8, depthshade=False)
    ax_a.plot(src[:, 0], src[:, 1], src[:, 2], color="k", lw=2.0, label="source line")
    ax_a.set_title("a  aggregate field on OPM array", fontsize=10)
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("y")
    ax_a.set_zlabel("z (mm)")
    ax_a.legend(loc="upper right", fontsize=8)
    fig.colorbar(p, ax=ax_a, shrink=0.6, pad=0.08)

    ax_b = fig.add_subplot(gs[0, 1])
    theta, zc = _cylinder_theta_z(pos)
    pb = ax_b.scatter(theta, zc, c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=14)
    ax_b.axhspan(float(src[:, 2].min()), float(src[:, 2].max()),
                 color=NATURE_PALETTE["axis"], alpha=0.10)
    ax_b.set_xlabel("azimuth (°)")
    ax_b.set_ylabel("sensor z (mm)")
    ax_b.set_title("b  unrolled cylinder topography", fontsize=10)
    fig.colorbar(pb, ax=ax_b, shrink=0.85, pad=0.02, label=cbar_label)

    peak = float(np.max(np.abs(v)))
    scope = region_label or "all"
    fig.suptitle(
        f"Global field pattern — {scope} {src.shape[0]} sources ({mode}, moment "
        f"{'xyz'[moment]}), peak {peak:.0f} fT/nA·m",
        fontsize=12, fontweight="bold", y=0.97,
    )
    lvl = f"_{region_label}" if region_label else ""
    # target_output stamps the source-target slug on the default name, so a run
    # for one target never overwrites another's figure at the same level.
    out = out_path or target_output(cfg, f"sensor_field_aggregate_{mode}{lvl}.png")
    return save_figure(fig, out, dpi=dpi)


def render_sensor_field(
    cfg: Config, *, source_idx: int = -1, moment: int | None = None,
    z_range: tuple[float, float] | None = None, region_label: str | None = None,
    out_path: Path | None = None, dpi: int = 300,
) -> Path:
    """Four-panel sensor-field characterisation figure for one source.

    a  3-D sensor array coloured by the radial field (the dipolar pattern).
    b  Unrolled cylinder (azimuth vs z) — the canonical OPM topography.
    c  Peak field along the source axis (all sources) — where the region is seen best.
    d  Radial-field falloff vs sensor distance from the source.

    With ``z_range`` and no explicit ``source_idx``, the strongest source *within*
    the band (e.g. C7) is shown and the band is shaded in panel c.
    """
    apply_nature_style()
    lf = load_leadfield(cfg.outputs.forward_npz)
    n_src = lf.source_pos.shape[0]

    # Default to the strongest source — the clearest "what does it look like" view.
    # If a Z band is given, restrict the search to sources inside it.
    _, peaks = peak_amplitude_along_axis(lf)
    if source_idx < 0:
        if z_range is not None:
            band = sources_in_z_band(lf, *z_range)
            source_idx = int(band[np.argmax(peaks[band])])
        else:
            source_idx = int(np.argmax(peaks))
    if moment is None:
        moment = _dominant_moment(lf, source_idx)
    v, pos = radial_field(lf, source_idx, moment)
    src = lf.source_pos[source_idx]
    cmap = divergent_cmap()
    vmin, vmax = divergent_norm(v)

    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.92,
                  bottom=0.07, hspace=0.28, wspace=0.24)

    # a) 3-D sensor cloud coloured by radial field
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    p = ax_a.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=v, cmap=cmap,
                     vmin=vmin, vmax=vmax, s=8, depthshade=False)
    ax_a.scatter(*src, color="k", marker="*", s=120, label="source")
    ax_a.set_title("a  radial field on OPM array (fT/nA·m)", fontsize=10)
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("y")
    ax_a.set_zlabel("z (mm)")
    ax_a.legend(loc="upper right", fontsize=8)
    fig.colorbar(p, ax=ax_a, shrink=0.6, pad=0.08)

    # b) Unrolled cylinder topography
    ax_b = fig.add_subplot(gs[0, 1])
    theta, zc = _cylinder_theta_z(pos)
    pb = ax_b.scatter(theta, zc, c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=14)
    ax_b.axhline(src[2], color=NATURE_PALETTE["axis"], lw=0.8, ls="--")
    ax_b.set_xlabel("azimuth (°)")
    ax_b.set_ylabel("sensor z (mm)")
    ax_b.set_title("b  unrolled cylinder topography", fontsize=10)
    fig.colorbar(pb, ax=ax_b, shrink=0.8, pad=0.02, label="fT/nA·m")
    add_panel_label(ax_b, "")

    # c) Peak field along the source axis
    ax_c = fig.add_subplot(gs[1, 0])
    z_src = lf.source_pos[:, 2]
    order = np.argsort(z_src)
    ax_c.plot(z_src[order], peaks[order], color=NATURE_PALETTE["blue"], lw=1.6)
    if z_range is not None:
        ax_c.axvspan(z_range[0], z_range[1], color=NATURE_PALETTE["axis"],
                     alpha=0.12, label=region_label or "Z band")
    ax_c.axvline(src[2], color="k", ls="--", lw=0.8, label="shown source")
    ax_c.set_xlabel("source z (mm)")
    ax_c.set_ylabel("best-channel peak (fT/nA·m)")
    ax_c.set_title("c  field strength along the source axis", fontsize=10)
    ax_c.legend(fontsize=8)

    # d) Falloff vs sensor distance
    ax_d = fig.add_subplot(gs[1, 1])
    dist = np.linalg.norm(pos - src, axis=1)
    ax_d.scatter(dist, np.abs(v), s=8, color=NATURE_PALETTE["teal"], alpha=0.5)
    ax_d.set_xlabel("sensor distance from source (mm)")
    ax_d.set_ylabel("|radial field| (fT/nA·m)")
    ax_d.set_yscale("log")
    ax_d.set_title("d  field falloff with distance", fontsize=10)

    ch = sensor_field_characteristics(lf, source_idx, moment=moment)
    scope = f"{region_label} " if region_label else ""
    fig.suptitle(
        f"Sensor field pattern — {scope}source {source_idx}/{n_src} at z={src[2]:.0f} mm "
        f"(dominant moment {ch['dominant_moment']}, peak {ch['peak_fT_per_nAm']:.0f} fT/nA·m, "
        f"lobe sep {ch['dipole_lobe_separation_mm']:.0f} mm)",
        fontsize=12, fontweight="bold", y=0.985,
    )
    lvl = f"_{region_label}" if region_label else ""
    return save_figure(fig, out_path or target_output(cfg, f"sensor_field{lvl}.png"), dpi=dpi)
