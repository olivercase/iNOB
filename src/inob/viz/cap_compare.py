"""Propagating-CAP vs stationary-dipole comparison figure.

The physiology simulator (:mod:`inob.physiology.simulate`) collapses every
event in a Scenario to a single stationary biphasic source. Whether that is
defensible depends entirely on the target, and this module renders the
comparison that decides it: for one synchronous event with a realistic
lognormal fibre-diameter distribution, plot the best MEG-channel time-trace
under (i) the stationary approximation and (ii) the full propagating-CAP
source model (:mod:`inob.sources.cap`).

Both signals carry the same total fibre-population dipole moment, so the
amplitude ratio quantifies the propagation effect alone.

All physiology comes from the target's
:class:`~inob.physiology.profiles.PhysiologyProfile`, selected by
``--source-target``. Nothing here is specific to any one target: whether the
stationary approximation survives is measured, and the profile's own
prediction is reported alongside so the two can be compared.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from inob.config import Config, source_region_label, source_target_tag, target_output
from inob.io.npz import load_leadfield
from inob.physiology.profiles import profile_for_tag
from inob.sources.cap import (
    biphasic_waveform,
    conduction_velocity_m_per_s,
    longitudinal_leadfield,
)
from inob.viz.style import (
    NATURE_PALETTE,
    add_panel_label,
    apply_nature_style,
    save_figure,
)
from inob.viz.topoplot import _radial_channel_mask

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
    n_fibres: int | None = None,
    ap_amplitude_mV: float | None = None,
    sigma_in_Sm: float | None = None,
    ap_width_ms: float | None = None,
    fs_hz: float = 30_000.0,
    duration_ms: float | None = None,
    segment_mm: float | None = None,
    out_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render the propagating-vs-stationary comparison figure for the MEG leadfield.

    Every physiological parameter defaults to the target's
    :class:`~inob.physiology.profiles.PhysiologyProfile` — fibre population,
    conduction velocity, AP width and amplitude, event size, and how far one
    event's activity propagates. Explicit keyword arguments override the
    profile for sensitivity checks. Three source models are compared *at the
    same total dipole moment*:

      * **Stationary at generator**: all moment lumped at one point — the
        approximation used by :mod:`inob.physiology.simulate`.
      * **Propagating over the active segment**: the same moment distributed
        along the profile's ``propagation_span_mm``, each position carrying
        its arc-length-dependent transit delay.
      * **Propagating whole polyline**: the entire modelled structure.

    What the figure demonstrates is target-dependent, which is the point:

      * For the **vagus**, baroreceptor afferents are localised to the ~50 mm
        cervical bundle, crossed in ~1 ms at 47 m/s — comparable to the 0.5 ms
        AP width. The stationary approximation is a good one, and the figure
        exists to show that it is.
      * For the **spine**, there is no localised generator: the volley enters
        at one segment and ascends the whole imaged cord, ~450 mm at 59 m/s
        = ~7.6 ms, an order of magnitude longer than the 0.7 ms AP width. The
        stationary approximation is *not* valid, and the figure exists to show
        by how much it fails.

    Targets whose profile is not physiologically validated (muscle — no
    MUAP/motor-unit model) still render, but carry a visible provisional
    banner and log a warning.
    """
    apply_nature_style()
    region = source_region_label(cfg)
    tag = source_target_tag(cfg)
    profile = profile_for_tag(tag)

    # Profile supplies the physiology; explicit kwargs override it.
    n_fibres = profile.n_fibres if n_fibres is None else n_fibres
    ap_amplitude_mV = (profile.ap_amplitude_mV if ap_amplitude_mV is None
                       else ap_amplitude_mV)
    sigma_in_Sm = profile.sigma_in_Sm if sigma_in_Sm is None else sigma_in_Sm
    ap_width_ms = profile.ap_width_ms if ap_width_ms is None else ap_width_ms

    provisional = not profile.validated
    if provisional:
        logger.warning(
            "[cap-compare] PHYSIOLOGY-TODO: no validated physiology for '%s' "
            "(%s); output reuses nerve-CAP assumptions and is NOT "
            "physiologically validated. Treat as provisional.",
            region, profile.paradigm,
        )
    logger.info("[cap-compare] physiology profile:\n%s", profile.describe())

    lf = load_leadfield(cfg.outputs.forward_npz)
    L_long, arc_mm, _ = longitudinal_leadfield(lf.L, lf.source_pos)
    arc_m = arc_mm * 1e-3                                  # (S,)

    # Rostral end of the polyline: the reference point for the stationary
    # lump and the end point of the ascending wave.
    hot_idx = int(np.argmax(lf.source_pos[:, 2]))

    fibres = profile.fibres
    cv_per_d = conduction_velocity_m_per_s(fibres.diameters_um)
    cv_mean = float(np.sum(cv_per_d * fibres.weights))     # m/s
    # Per-fibre dipole moment, A·m, as a function of diameter d (Hämäläinen)
    Q_per_fibre_Am = (
        np.pi * (fibres.diameters_um * 1e-6) ** 2
        * sigma_in_Sm * (ap_amplitude_mV * 1e-3) / 4.0
    )

    # How far one event's activity travels. None = the whole polyline.
    total_span_mm = float(arc_mm[-1] - arc_mm[0])
    if segment_mm is None:
        segment_mm = (
            total_span_mm if profile.propagation_span_mm is None
            else profile.propagation_span_mm
        )

    # The window must cover the transit or the propagating trace is cut off
    # mid-flight. Sized from the segment the event actually crosses, floored at
    # the historical 30 ms so targets with a short transit (vagus: 50 mm at
    # 47 m/s ≈ 1 ms) keep exactly the window they have always used.
    if duration_ms is None:
        transit_ms = profile.transit_ms(segment_mm)
        duration_ms = max(30.0, 2.5 * transit_ms + 10.0 * ap_width_ms)
        logger.info("[cap-compare] window %.0f ms (transit %.2f ms over %.0f mm)",
                    duration_ms, transit_ms, segment_mm)

    n = round(duration_ms * fs_hz / 1000.0)
    t_ms = np.arange(n) / fs_hz * 1000.0
    centre_ms = duration_ms * 0.4
    arc_total_m = float(arc_m[-1] - arc_m[0])

    # The active segment: the rostral-most `segment_mm` of the
    # polyline, measured back from the hot-spot. For a profile with no
    # localised generator (spine) this is the whole polyline.
    seg_m = segment_mm * 1e-3
    seg_mask = arc_m >= (arc_m[hot_idx] - seg_m)
    seg_idx = np.where(seg_mask)[0]
    seg_len_m = float(arc_m[seg_idx[-1]] - arc_m[seg_idx[0]])

    def stationary_signal() -> np.ndarray:
        """Lumped approximation: all N fibres at a single point.

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
        rostral end of the polyline — the most generous (highest-amplitude)
        timing for the propagating model.
        """
        # Snap a target arc-length to the closest source index.
        x_start_m = arc_m[x_start_idx]
        x_hot_m = arc_m[hot_idx]
        if time_peak_at_hotspot:
            t_fire_ms = centre_ms - (x_hot_m - x_start_m) / cv_mean * 1000.0
        else:
            t_fire_ms = centre_ms

        shape_t = biphasic_waveform(t_ms - centre_ms, ap_width_ms=ap_width_ms)

        x_lo = arc_m[min(x_start_idx, x_end_idx)]
        x_hi = arc_m[max(x_start_idx, x_end_idx)]
        margin_m = 0.005     # 5 mm slop so AP envelope decays smoothly off-segment

        # Wavelet position per (diameter, sample), snapped to the nearest source.
        # arc_m is a cumulative arc length and therefore sorted, so searchsorted
        # + a neighbour comparison gives the same index as an argmin over |Δ|.
        x_dt = x_start_m + cv_per_d[:, None] * (t_ms - t_fire_ms)[None, :] * 1e-3
        right = np.searchsorted(arc_m, x_dt).clip(1, len(arc_m) - 1)
        left = right - 1
        nearest = np.where(
            np.abs(x_dt - arc_m[left]) <= np.abs(arc_m[right] - x_dt), left, right,
        )

        # Weight of each (diameter, sample) contribution, zero off-segment.
        on_seg = (x_dt >= x_lo - margin_m) & (x_dt <= x_hi + margin_m)
        w_dt = (n_fibres * (fibres.weights * Q_per_fibre_Am)[:, None]
                * shape_t[None, :] * on_seg)

        # Accumulate into a (source × sample) moment map, then project through
        # the leadfield once: same arithmetic as the per-sample loop, one matmul.
        moments = np.zeros((len(arc_m), n), dtype=np.float64)
        t_idx = np.broadcast_to(np.arange(n), nearest.shape)
        np.add.at(moments, (nearest.ravel(), t_idx.ravel()), w_dt.ravel())
        return L_long @ moments

    sig_stat = stationary_signal()
    # Propagating over the active segment: wave starts at the caudal end of
    # the segment, ends at the rostral end, timed so the mean-CV wavelet peaks
    # there. For a profile with no localised generator this spans the whole
    # polyline and coincides with sig_whole below.
    sig_seg = moving_wavelet(
        x_start_idx=seg_idx[0], x_end_idx=hot_idx, time_peak_at_hotspot=True,
    )
    # Propagating over the whole polyline: for the vagus this is the
    # pulmonary/abdominal-afferent case; for a whole-span profile it is the
    # same model as sig_seg and the figure collapses the two.
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
    sig_seg_fT = sig_seg * 1e15
    sig_whole_fT = sig_whole * 1e15
    p_stat = sig_stat_fT[best_c]
    p_seg = sig_seg_fT[best_c]
    p_whole = sig_whole_fT[best_c]
    residual = p_seg - p_stat

    peak_stat = float(np.abs(p_stat).max())
    peak_seg = float(np.abs(p_seg).max())
    peak_whole = float(np.abs(p_whole).max())
    fwhm_stat = _fwhm_ms(p_stat, t_ms)
    fwhm_seg = _fwhm_ms(p_seg, t_ms)
    transit_seg_ms = seg_len_m / cv_mean * 1000.0
    transit_whole_ms = arc_total_m / cv_mean * 1000.0
    rms_residual = float(np.sqrt(np.mean(residual ** 2)))
    rms_stat = float(np.sqrt(np.mean(p_stat ** 2)))

    logger.info(
        "[cap-compare] best MEG #%d  ·  stat=%.2f fT  ·  active-segment prop=%.2f fT "
        "(ratio %.3f)  ·  whole-%s prop=%.2f fT (ratio %.3f)  ·  FWHM stat/prop=%.2f/%.2f ms "
        " ·  rms residual / rms stat=%.3f",
        best_c, peak_stat,
        peak_seg, peak_seg / max(peak_stat, 1e-30),
        region, peak_whole, peak_whole / max(peak_stat, 1e-30),
        fwhm_stat, fwhm_seg,
        rms_residual / max(rms_stat, 1e-30),
    )
    # The headline result is whether the lumped approximation survives. Say so
    # explicitly rather than leaving it to be read off the figure.
    stat_error = abs(peak_seg - peak_stat) / max(peak_stat, 1e-30)
    # A profile *predicts* whether lumping is safe (transit vs AP width); the
    # figure *measures* it. Report both, and never let the prediction overrule
    # a measurement that contradicts it.
    measured_ok = stat_error <= 0.25
    if profile.stationary_ok and measured_ok:
        logger.info(
            "[cap-compare] %s: transit %.2f ms vs AP width %.2f ms — stationary "
            "approximation holds (peak error %.0f%%)",
            region, transit_seg_ms, ap_width_ms, 100.0 * stat_error,
        )
    elif not profile.stationary_ok:
        logger.warning(
            "[cap-compare] %s: transit %.2f ms over %.0f mm vs AP width %.2f ms — "
            "the stationary approximation is NOT valid for this target; it "
            "misestimates the peak by %.0f%% and the width by %.2fx. Use the "
            "propagating model for %s figures.",
            region, transit_seg_ms, seg_len_m * 1000, ap_width_ms,
            100.0 * stat_error, fwhm_seg / max(fwhm_stat, 1e-30), region,
        )
    else:
        logger.warning(
            "[cap-compare] %s: the %s profile predicts the stationary "
            "approximation should hold (transit %.2f ms vs AP width %.2f ms), but "
            "the measured peak error is %.0f%%. Either the profile's propagation "
            "span is wrong for this target or the source geometry is not a "
            "polyline (arc length is only meaningful for a linear structure).",
            region, profile.name, transit_seg_ms, ap_width_ms, 100.0 * stat_error,
        )

    seg_label = ("whole cord" if profile.propagation_span_mm is None
                 else f"{segment_mm:.0f} mm segment")

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = GridSpec(1, 3, figure=fig, left=0.06, right=0.98, top=0.82, bottom=0.16,
                  wspace=0.32, width_ratios=[1.3, 1.0, 0.95])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_ms, p_stat, color=NATURE_PALETTE["blue"], lw=1.6,
             label="Stationary at generator")
    ax0.plot(t_ms, p_seg, color=NATURE_PALETTE["red"], lw=1.4, alpha=0.95,
             label=f"Propagating, {seg_label} ({seg_len_m * 1000:.0f} mm)")
    # When the profile has no localised generator the active segment IS the
    # whole polyline, so the third curve would be an exact duplicate of the
    # second. Draw it only when it is a distinct model.
    whole_is_distinct = abs(seg_len_m - arc_total_m) > 1e-6
    if whole_is_distinct:
        ax0.plot(t_ms, p_whole, color=NATURE_PALETTE["axis"], lw=1.0, alpha=0.7,
                 linestyle="--",
                 label=f"Propagating whole {region} ({arc_total_m * 1000:.0f} mm)")
    ax0.axhline(0, color=NATURE_PALETTE["axis"], lw=0.5, alpha=0.4)
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel(f"Best radial MEG channel #{best_c}  (fT)")
    n_models = 3 if whole_is_distinct else 2
    ax0.set_title(f"Single event, same total moment, {n_models} source models",
                  fontsize=10)
    ax0.legend(loc="upper right", fontsize=8, handlelength=1.6)
    add_panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t_ms, residual, color=NATURE_PALETTE["axis"], lw=1.0)
    ax1.axhline(0, color=NATURE_PALETTE["axis"], lw=0.4, alpha=0.4)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Propagating − stationary  (fT)")
    verdict = ("propagation is a small correction"
               if profile.stationary_ok else
               "propagation dominates")
    ax1.set_title(f"Residual — {verdict}", fontsize=10)
    add_panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    Q_total_nAm = float(
        n_fibres * np.sum(fibres.weights * Q_per_fibre_Am) * 1e9
    )
    verdict_line = (
        "Stationary approximation VALID\n  (transit ≈ AP width)"
        if profile.stationary_ok else
        "Stationary approximation INVALID\n  (transit >> AP width)"
    )
    summary = (
        f"{profile.paradigm}\n"
        f"Generator: {profile.generator}\n\n"
        f"Event: {n_fibres} fibres, mean d={np.sum(fibres.diameters_um * fibres.weights):.1f} µm\n"
        f"Total moment Q_total: {Q_total_nAm:.2f} nA·m\n"
        f"Active segment: {seg_len_m * 1000:.0f} mm\n"
        f"Whole {region} polyline: {arc_total_m * 1000:.0f} mm\n"
        f"Mean fibre CV: {cv_mean:.1f} m/s\n"
        f"Transit, active segment: {transit_seg_ms:.2f} ms\n"
        f"Transit, whole: {transit_whole_ms:.2f} ms\n"
        f"AP width: {ap_width_ms:.2f} ms\n\n"
        f"Stationary peak:     {peak_stat:8.2f} fT\n"
        f"Propagating (segment):{peak_seg:7.2f} fT\n"
        f"  P/S ratio:         {peak_seg / max(peak_stat, 1e-30):8.3f}\n"
        f"Propagating (whole): {peak_whole:8.2f} fT\n"
        f"  P/S ratio:         {peak_whole / max(peak_stat, 1e-30):8.3f}\n\n"
        f"FWHM stationary:  {fwhm_stat:5.2f} ms\n"
        f"FWHM propagating: {fwhm_seg:5.2f} ms\n"
        f"FWHM ratio:       {fwhm_seg / max(fwhm_stat, 1e-30):6.3f}\n\n"
        f"RMS(prop − stat) / RMS(stat): {rms_residual / max(rms_stat, 1e-30):.3f}\n\n"
        f"{verdict_line}"
    )
    ax2.text(0.02, 0.97, summary, transform=ax2.transAxes,
             va="top", ha="left", fontsize=7.8,
             family="monospace", color=NATURE_PALETTE["axis"])
    add_panel_label(ax2, "c")

    fig.suptitle(
        "Propagating CAP vs stationary-dipole approximation  —  "
        f"{profile.label}, {n_fibres} fibres",
        fontsize=11.5, fontweight="bold", y=0.97,
    )

    if provisional:
        # Self-documenting stamp: figures escape into slides/papers, so any
        # unvalidated export must carry its own caveat.
        fig.text(
            0.5, 0.005,
            f"PROVISIONAL — no validated physiology for '{region}': "
            f"{profile.paradigm}. Reuses nerve-CAP assumptions; NOT "
            f"physiologically validated.",
            ha="center", va="bottom", fontsize=8, style="italic",
            color=NATURE_PALETTE.get("red", "#CC3311"),
        )

    return save_figure(fig, out_path or target_output(cfg, "cap_compare.png"), dpi=dpi)
