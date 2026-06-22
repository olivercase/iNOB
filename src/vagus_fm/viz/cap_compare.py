"""Propagating-CAP vs stationary-dipole comparison figure.

The physiology simulator (:mod:`vagus_fm.physiology.simulate`) collapses every
event in a Scenario to a stationary biphasic source at the cervical hot-spot.
This is justified — but not visualised — by the argument that fibre
conduction along the ~50 mm cervical polyline takes ~1 ms (CV ≈ 50 m/s),
which is comparable to the 0.5 ms AP width, so propagation only smears each
event by a fraction of an AP width.

This module renders the side-by-side comparison a reviewer is owed: for a
single A-fibre CAP event with a realistic lognormal diameter distribution,
plot the best MEG-channel time-trace under (i) the stationary approximation
used in the manuscript and (ii) the full propagating-CAP source model
(:mod:`vagus_fm.sources.cap`).

Both signals carry the same total fibre-population dipole moment so the
amplitude ratio quantifies the propagation effect alone.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from vagus_fm.config import Config
from vagus_fm.io.npz import load_leadfield
from vagus_fm.physiology.scenarios import a_fibre_population
from vagus_fm.sources.cap import (
    biphasic_waveform,
    conduction_velocity_m_per_s,
    longitudinal_leadfield,
)
from vagus_fm.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
)
from vagus_fm.viz.topoplot import _radial_channel_mask

logger = logging.getLogger(__name__)


def _fwhm_ms(x: np.ndarray, t_ms: np.ndarray) -> float:
    half = np.abs(x).max() * 0.5
    above = np.where(np.abs(x) >= half)[0]
    if len(above) < 2:
        return float("nan")
    return float(t_ms[above[-1]] - t_ms[above[0]])


def render_cap_compare(
    cfg: Config,
    *,
    n_fibres: int = 200,
    ap_amplitude_mV: float = 70.0,
    sigma_in_Sm: float = 1.0,
    ap_width_ms: float = 0.5,
    fs_hz: float = 30_000.0,
    duration_ms: float = 30.0,
    cervical_segment_mm: float = 50.0,
    out_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render the propagating-vs-stationary comparison figure for the MEG leadfield.

    For a synchronously-firing baroreceptor bundle of ``n_fibres`` afferents,
    we compare three models *at the same total dipole moment*:

      * **Stationary at hot-spot**:  all moment lumped at the cervical
        hot-spot — the approximation used by :mod:`vagus_fm.physiology.simulate`.
      * **Propagating cervical segment (~50 mm)**: same 200 fibres, but
        spatially distributed along the top ``cervical_segment_mm`` of the
        polyline (carotid-baroreceptor afferents are anatomically localised
        to the cervical bundle, not the entire 500 mm vagus). Each source
        position carries its arc-length-dependent transit delay.
      * **Propagating whole vagus (~500 mm)**: the entire polyline, applicable
        only to pulmonary / abdominal afferents that physically traverse it.

    The cervical-segment comparison is the one that bears on the manuscript
    claim that propagation only smears each event by a fraction of an AP
    width.  All three signals are computed by the same code path (see
    ``assemble``) so units and normalisation match.
    """
    apply_nature_style()
    lf = load_leadfield(cfg.outputs.forward_npz)
    L_long, arc_mm, _ = longitudinal_leadfield(lf.L, lf.source_pos)
    arc_m = arc_mm * 1e-3                                  # (S,)

    # Cervical hot-spot = highest-Z source on the polyline (closest to neck patch)
    hot_idx = int(np.argmax(lf.source_pos[:, 2]))

    fibres = a_fibre_population()
    cv_per_d = conduction_velocity_m_per_s(fibres.diameters_um)
    cv_mean = float(np.sum(cv_per_d * fibres.weights))     # m/s
    # Per-fibre dipole moment, A·m, as a function of diameter d (Hämäläinen)
    Q_per_fibre_Am = (
        np.pi * (fibres.diameters_um * 1e-6) ** 2
        * sigma_in_Sm * (ap_amplitude_mV * 1e-3) / 4.0
    )

    n = round(duration_ms * fs_hz / 1000.0)
    t_ms = np.arange(n) / fs_hz * 1000.0
    centre_ms = duration_ms * 0.4
    arc_total_m = float(arc_m[-1] - arc_m[0])

    # Identify the cervical segment: the top `cervical_segment_mm` of the polyline,
    # measured back from the hot-spot (highest-Z source at z=1499mm).
    cerv_m = cervical_segment_mm * 1e-3
    cerv_mask = arc_m >= (arc_m[hot_idx] - cerv_m)
    cerv_idx = np.where(cerv_mask)[0]
    cerv_len_m = float(arc_m[cerv_idx[-1]] - arc_m[cerv_idx[0]])

    def stationary_signal() -> np.ndarray:
        """Manuscript approximation: all N fibres lumped at the cervical hot-spot.

        Σ_d w(d) Q(d) is the population-mean per-fibre moment <Q>_w; multiplied
        by N gives the total event moment. Signal = L_long[hot] × Q_total × shape(t).
        """
        Q_total_Am = n_fibres * float(np.sum(fibres.weights * Q_per_fibre_Am))
        shape = biphasic_waveform(t_ms - centre_ms, ap_width_ms=ap_width_ms)
        return L_long[:, hot_idx][:, None] * Q_total_Am * shape[None, :]

    def moving_wavelet(
        x_start_idx: int, x_end_idx: int, *,
        time_peak_at_hotspot: bool = True,
    ) -> np.ndarray:
        """Physically correct: a single AP wavelet of total moment N×<Q>_w
        starts at ``arc_m[x_start_idx]`` and propagates rostrally at fibre-CV.

        At each instant t and each diameter d, the wave is at position
        ``x_d(t) = x_start + CV(d)·(t − t_fire)``. The contribution at sensor
        c is N · w(d) · Q(d) · L_long[c, x_d(t)] · shape(t − t_fire), summed
        over fibre diameter d.

        If ``time_peak_at_hotspot`` is true, ``t_fire`` is chosen so the
        mean-CV wavelet's peak coincides with the wave passing through the
        cervical hot-spot — the most generous (highest-amplitude) timing for
        the propagating model.
        """
        # Snap a target arc-length to the closest source index.
        x_start_m = arc_m[x_start_idx]
        x_hot_m = arc_m[hot_idx]
        if time_peak_at_hotspot:
            t_fire_ms = centre_ms - (x_hot_m - x_start_m) / cv_mean * 1000.0
        else:
            t_fire_ms = centre_ms

        sig = np.zeros((L_long.shape[0], n), dtype=np.float64)
        shape_t = biphasic_waveform(t_ms - centre_ms, ap_width_ms=ap_width_ms)

        x_lo = arc_m[min(x_start_idx, x_end_idx)]
        x_hi = arc_m[max(x_start_idx, x_end_idx)]
        margin_m = 0.005     # 5 mm slop so AP envelope decays smoothly off-segment

        for d_idx, w in enumerate(fibres.weights):
            v = float(cv_per_d[d_idx])
            Q_d_Am = float(Q_per_fibre_Am[d_idx])
            # Position of this diameter's wavelet at every sample time
            x_d_t = x_start_m + v * (t_ms - t_fire_ms) * 1e-3
            for t_idx in range(n):
                x_now = float(x_d_t[t_idx])
                if not (x_lo - margin_m <= x_now <= x_hi + margin_m):
                    continue
                s_nearest = int(np.argmin(np.abs(arc_m - x_now)))
                sig[:, t_idx] += (
                    n_fibres * w * Q_d_Am
                    * L_long[:, s_nearest] * shape_t[t_idx]
                )
        return sig

    sig_stat = stationary_signal()
    # Propagating cervical: wave starts at cervical bottom, ends at hot-spot,
    # timed so the mean-CV wavelet peaks at the hot-spot.
    sig_cerv = moving_wavelet(
        x_start_idx=cerv_idx[0], x_end_idx=hot_idx, time_peak_at_hotspot=True,
    )
    # Propagating whole vagus: applicable to pulmonary / abdominal afferents.
    sig_whole = moving_wavelet(
        x_start_idx=0, x_end_idx=hot_idx, time_peak_at_hotspot=True,
    )

    # Pick the best radial MEG channel for the *stationary* signal — this is
    # the channel `simulate_train` would report and the SNR pipeline uses, so
    # the comparison is "at the channel the simulator highlights, what does
    # propagation do?".
    radial = _radial_channel_mask(list(lf.channel_names))
    best_c = int(np.argmax(np.sqrt(np.mean(sig_stat ** 2, axis=1)) * radial))

    sig_stat_fT = sig_stat * 1e15
    sig_cerv_fT = sig_cerv * 1e15
    sig_whole_fT = sig_whole * 1e15
    p_stat = sig_stat_fT[best_c]
    p_cerv = sig_cerv_fT[best_c]
    p_whole = sig_whole_fT[best_c]
    residual = p_cerv - p_stat

    peak_stat = float(np.abs(p_stat).max())
    peak_cerv = float(np.abs(p_cerv).max())
    peak_whole = float(np.abs(p_whole).max())
    fwhm_stat = _fwhm_ms(p_stat, t_ms)
    fwhm_cerv = _fwhm_ms(p_cerv, t_ms)
    transit_cerv_ms = cerv_len_m / cv_mean * 1000.0
    transit_whole_ms = arc_total_m / cv_mean * 1000.0
    rms_residual = float(np.sqrt(np.mean(residual ** 2)))
    rms_stat = float(np.sqrt(np.mean(p_stat ** 2)))

    logger.info(
        "[cap-compare] best MEG #%d  ·  stat=%.2f fT  ·  cerv prop=%.2f fT (ratio %.3f)  "
        "·  whole-vagus prop=%.2f fT (ratio %.3f)  ·  FWHM stat/cerv=%.2f/%.2f ms  ·  "
        "rms residual / rms stat=%.3f",
        best_c, peak_stat,
        peak_cerv, peak_cerv / max(peak_stat, 1e-30),
        peak_whole, peak_whole / max(peak_stat, 1e-30),
        fwhm_stat, fwhm_cerv,
        rms_residual / max(rms_stat, 1e-30),
    )

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = GridSpec(1, 3, figure=fig, left=0.06, right=0.98, top=0.82, bottom=0.16,
                  wspace=0.32, width_ratios=[1.3, 1.0, 0.95])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_ms, p_stat, color=NATURE_PALETTE["blue"], lw=1.6,
             label="Stationary at hot-spot (simulator)")
    ax0.plot(t_ms, p_cerv, color=NATURE_PALETTE["red"], lw=1.4, alpha=0.95,
             label=f"Propagating cervical ({cervical_segment_mm:.0f} mm)")
    ax0.plot(t_ms, p_whole, color=NATURE_PALETTE["axis"], lw=1.0, alpha=0.7,
             linestyle="--",
             label=f"Propagating whole vagus ({arc_total_m * 1000:.0f} mm)")
    ax0.axhline(0, color=NATURE_PALETTE["axis"], lw=0.5, alpha=0.4)
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel(f"Best radial MEG channel #{best_c}  (fT)")
    ax0.set_title("Single A-fibre baroreceptor event — same total moment, three source models",
                  fontsize=10)
    ax0.legend(loc="upper right", fontsize=8, handlelength=1.6)
    add_panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t_ms, residual, color=NATURE_PALETTE["axis"], lw=1.0)
    ax1.axhline(0, color=NATURE_PALETTE["axis"], lw=0.4, alpha=0.4)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Cervical-propagating − stationary  (fT)")
    ax1.set_title("Residual = propagation contribution at the\ncervical-localised baroreceptor source",
                  fontsize=10)
    add_panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    Q_total_nAm = float(
        n_fibres * np.sum(fibres.weights * Q_per_fibre_Am) * 1e9
    )
    summary = (
        f"Event: {n_fibres} A-fibres\n"
        f"Total moment Q_total: {Q_total_nAm:.2f} nA·m\n"
        f"Cervical segment: {cerv_len_m * 1000:.0f} mm\n"
        f"Whole vagus polyline: {arc_total_m * 1000:.0f} mm\n"
        f"Mean fibre CV: {cv_mean:.1f} m/s\n"
        f"Transit time, cervical: {transit_cerv_ms:.2f} ms\n"
        f"Transit time, whole: {transit_whole_ms:.2f} ms\n"
        f"AP width: {ap_width_ms:.2f} ms\n\n"
        f"Stationary peak:        {peak_stat:6.2f} fT\n"
        f"Cervical-propagating:  {peak_cerv:6.2f} fT\n"
        f"  P/S ratio:           {peak_cerv / max(peak_stat, 1e-30):6.3f}\n"
        f"Whole-vagus propagating: {peak_whole:6.2f} fT\n"
        f"  P/S ratio:           {peak_whole / max(peak_stat, 1e-30):6.3f}\n\n"
        f"FWHM stationary:  {fwhm_stat:5.2f} ms\n"
        f"FWHM cervical:    {fwhm_cerv:5.2f} ms\n"
        f"FWHM ratio:       {fwhm_cerv / max(fwhm_stat, 1e-30):6.3f}\n\n"
        f"RMS(cerv − stat) / RMS(stat): {rms_residual / max(rms_stat, 1e-30):.3f}"
    )
    ax2.text(0.02, 0.97, summary, transform=ax2.transAxes,
             va="top", ha="left", fontsize=8.5,
             family="monospace", color=NATURE_PALETTE["axis"])
    add_panel_label(ax2, "c")

    fig.suptitle(
        "Propagating CAP vs stationary-dipole approximation  —  "
        f"{n_fibres} A-fibres at the cervical baroreceptor source",
        fontsize=11.5, fontweight="bold", y=0.97,
    )

    out = out_path or cfg.outputs.base / "cap_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("[saved] %s", out)
    return out
