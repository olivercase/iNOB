"""Propagating-CAP vs stationary-dipole source models.

One synchronous event can be modelled two ways, at the same total dipole
moment:

  * **stationary** — the whole moment lumped at one point on the polyline;
  * **propagating** — a wavelet of that moment travelling at fibre conduction
    velocity, each position carrying its arc-length transit delay.

Whether the first is defensible is target-dependent, and each target's
:class:`~inob.physiology.profiles.PhysiologyProfile` records its prediction in
``stationary_ok``. This module *measures* it. The vagus passes (transit ≈ AP
width over the localised cervical bundle); the spine fails badly (the volley
ascends the whole cord, ~8 ms against a 0.7 ms AP width).

Why this lives here rather than in the figure that draws it
-----------------------------------------------------------
It used to be computed inline inside :mod:`inob.viz.cap_compare`, which meant
:mod:`inob.viz.detectability` had no access to it and silently assumed the
stationary model. The two figures then disagreed: cap_compare printed
"stationary approximation INVALID" for the spine and measured a peak ratio of
~0.30, while the detectability figure beside it quoted trial counts computed as
if the ratio were 1 — an 11x difference in trials-to-detect. Both now call
:func:`compute_propagation_signals`, so they cannot drift apart again.

The signals are returned per channel; how to reduce them to one number is the
caller's business, because the two figures legitimately differ on it (cap_compare
reports the best radial MEG channel; detectability uses each modality's own
detection observable). :func:`propagation_ratio` applies a caller-supplied
reduction so that choice stays explicit.

Scope
-----
The propagating model describes *one event traversing the structure*. It has no
per-source decomposition — the wave is not parameterised by a source index, it
sweeps across all of them — so the ratio it yields is a single event-level
number, not a function of position along the polyline. Callers that plot
against source position must apply it as a scalar and say so.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from inob.io.npz import Leadfield
from inob.physiology.profiles import PhysiologyProfile
from inob.sources.cap import (
    biphasic_waveform,
    conduction_velocity_m_per_s,
    longitudinal_leadfield,
)

logger = logging.getLogger(__name__)


#: Above this ratio of cumulative arc length to bounding-box diagonal, the
#: source list is not an ordered path and "arc length along the sources" is
#: meaningless. Measured values: cervical cord polyline 1.06; muscle
#: volume-fill point cloud 57; combined spine+muscle 45. Threshold sits in the
#: wide empty gap between them.
MAX_POLYLINE_TORTUOSITY: float = 3.0


def is_ordered_polyline(
    source_pos_mm: np.ndarray, *, max_tortuosity: float = MAX_POLYLINE_TORTUOSITY,
) -> bool:
    """Whether ``source_pos_mm`` is an arc-length-ordered path.

    The propagating model advances a wavefront along cumulative arc length, so
    it only means anything when consecutive sources are actually neighbours
    along one structure — true for the vagus bundle and the spinal cord, false
    for a target whose sources are a 3-D volume fill (muscle, and therefore the
    combined spine+muscle target).

    Run over an unordered point cloud the wave "travels" tens of metres in
    random order and the resulting ratio is an artefact: the combined
    spine+muscle EEG leadfield gave 0.006, a 170x attenuation with no physical
    meaning. Callers that would otherwise present that number must check here
    first. See the PHYSIOLOGY-TODO in :mod:`inob.physiology.scenarios` — fixing
    it properly needs a fibre-ordered source path for muscle, not a threshold.
    """
    sp = np.asarray(source_pos_mm, dtype=np.float64)
    if len(sp) < 3:
        return True
    arc = float(np.linalg.norm(np.diff(sp, axis=0), axis=1).sum())
    diag = float(np.linalg.norm(sp.max(axis=0) - sp.min(axis=0)))
    if diag <= 0:
        return False
    return (arc / diag) <= max_tortuosity


@dataclass(frozen=True)
class PropagationSignals:
    """Per-channel time traces for one event under both source models.

    Signals are in the leadfield's native units (T for MEG, V for EEG) —
    the leadfield's ``L`` field is used, not the human-readable slot, so the
    caller scales as it prefers.
    """

    t_ms: np.ndarray            # (T,)
    stationary: np.ndarray      # (C, T) — all moment lumped at the hot spot
    segment: np.ndarray         # (C, T) — propagating over the active segment
    whole: np.ndarray           # (C, T) — propagating over the whole polyline

    hot_idx: int                # rostral end of the polyline (wave end point)
    stationary_idx: int         # where the stationary model lumps the moment
    n_fibres: int
    ap_width_ms: float
    Q_total_nAm: float
    cv_mean_m_per_s: float
    segment_mm: float           # active-segment length actually simulated
    total_span_mm: float        # whole-polyline arc length
    transit_segment_ms: float
    transit_whole_ms: float
    duration_ms: float
    profile: PhysiologyProfile

    @property
    def segment_is_whole(self) -> bool:
        """True when the active segment spans the entire polyline.

        Then ``segment`` and ``whole`` are the same model and a figure should
        not draw both as if they were distinct.
        """
        return abs(self.segment_mm - self.total_span_mm) < 1e-3


def compute_propagation_signals(
    leadfield: Leadfield,
    profile: PhysiologyProfile,
    *,
    n_fibres: int | None = None,
    ap_amplitude_mV: float | None = None,
    sigma_in_Sm: float | None = None,
    ap_width_ms: float | None = None,
    fs_hz: float = 30_000.0,
    duration_ms: float | None = None,
    segment_mm: float | None = None,
    stationary_idx: int | None = None,
) -> PropagationSignals:
    """Simulate one event under both source models against ``leadfield``.

    Every physiological parameter defaults to ``profile``; explicit keywords
    override it for sensitivity checks. Both models carry the *same* total
    dipole moment, so any amplitude difference is the propagation effect alone.

    The leadfield may be MEG or EEG: the source model is the same, and the two
    modalities' spatial sensitivity differs, so the resulting ratio does too.
    Pass whichever modality the answer is about.

    ``stationary_idx`` is where the stationary model puts the lumped moment,
    defaulting to the rostral end of the polyline (the wave's end point). A
    caller comparing against its *own* stationary assumption must pass the
    source index it actually used, or the ratio compares two different dipole
    positions and does not compose — for a patch array sited mid-structure that
    error is large enough to flip the ratio's direction.
    """
    n_fibres = profile.n_fibres if n_fibres is None else n_fibres
    ap_amplitude_mV = (profile.ap_amplitude_mV if ap_amplitude_mV is None
                       else ap_amplitude_mV)
    sigma_in_Sm = profile.sigma_in_Sm if sigma_in_Sm is None else sigma_in_Sm
    ap_width_ms = profile.ap_width_ms if ap_width_ms is None else ap_width_ms

    L_long, arc_mm, _ = longitudinal_leadfield(leadfield.L, leadfield.source_pos)
    arc_m = arc_mm * 1e-3

    # Rostral end of the polyline: the reference point for the stationary lump
    # and the end point of the ascending wave.
    hot_idx = int(np.argmax(leadfield.source_pos[:, 2]))
    lump_idx = hot_idx if stationary_idx is None else int(stationary_idx)

    fibres = profile.fibres
    cv_per_d = conduction_velocity_m_per_s(
        fibres.diameters_um, **(profile.cv_kwargs or {}),
    )
    cv_mean = float(np.sum(cv_per_d * fibres.weights))
    # Per-fibre dipole moment, A·m, as a function of diameter d (Hämäläinen).
    Q_per_fibre_Am = (
        np.pi * (fibres.diameters_um * 1e-6) ** 2
        * sigma_in_Sm * (ap_amplitude_mV * 1e-3) / 4.0
    )

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
        logger.debug("propagation window %.0f ms (transit %.2f ms over %.0f mm)",
                     duration_ms, transit_ms, segment_mm)

    n = round(duration_ms * fs_hz / 1000.0)
    t_ms = np.arange(n) / fs_hz * 1000.0
    centre_ms = duration_ms * 0.4
    arc_total_m = float(arc_m[-1] - arc_m[0])

    # The active segment: the rostral-most `segment_mm` of the polyline,
    # measured back from the hot spot. For a profile with no localised
    # generator (spine) this is the whole polyline.
    seg_m = segment_mm * 1e-3
    seg_idx = np.where(arc_m >= (arc_m[hot_idx] - seg_m))[0]
    seg_len_m = float(arc_m[seg_idx[-1]] - arc_m[seg_idx[0]])

    def stationary_signal() -> np.ndarray:
        """Lumped approximation: all N fibres at a single point.

        Σ_d w(d) Q(d) is the population-mean per-fibre moment <Q>_w; multiplied
        by N gives the total event moment.
        Signal = L_long[lump_idx] × Q_total × shape(t).
        """
        Q_total_Am = n_fibres * float(np.sum(fibres.weights * Q_per_fibre_Am))
        shape = biphasic_waveform(t_ms - centre_ms, ap_width_ms=ap_width_ms)
        return L_long[:, lump_idx][:, None] * Q_total_Am * shape[None, :]

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

    return PropagationSignals(
        t_ms=t_ms,
        stationary=stationary_signal(),
        # Wave starts at the caudal end of the segment and ends at the rostral
        # end, timed so the mean-CV wavelet peaks there. For a profile with no
        # localised generator this spans the whole polyline and coincides with
        # ``whole`` below.
        segment=moving_wavelet(x_start_idx=seg_idx[0], x_end_idx=hot_idx),
        whole=moving_wavelet(x_start_idx=0, x_end_idx=hot_idx),
        hot_idx=hot_idx,
        stationary_idx=lump_idx,
        n_fibres=int(n_fibres),
        ap_width_ms=float(ap_width_ms),
        Q_total_nAm=float(n_fibres * np.sum(fibres.weights * Q_per_fibre_Am) * 1e9),
        cv_mean_m_per_s=cv_mean,
        segment_mm=seg_len_m * 1000.0,
        total_span_mm=arc_total_m * 1000.0,
        transit_segment_ms=seg_len_m / cv_mean * 1000.0,
        transit_whole_ms=arc_total_m / cv_mean * 1000.0,
        duration_ms=float(duration_ms),
        profile=profile,
    )


def peak_over_channels(sig: np.ndarray) -> float:
    """Reduction for MEG: largest |value| over all channels and samples."""
    return float(np.abs(sig).max())


def peak_bipolar(sig: np.ndarray) -> float:
    """Reduction for EEG: largest potential *difference* over channel pairs.

    Mirrors :func:`inob.analysis.snr.per_source_best_bipolar` — a surface
    potential is only measurable as a difference, so the propagation ratio must
    be measured on the same observable the detectability figure uses, or the
    correction and the signal it corrects would be defined on different
    quantities.
    """
    return float(np.abs(sig.max(axis=0) - sig.min(axis=0)).max())


def propagation_ratio(
    signals: PropagationSignals,
    reduce: Callable[[np.ndarray], float] = peak_over_channels,
    *,
    model: str = "segment",
) -> float:
    """Propagating / stationary amplitude ratio under ``reduce``.

    ``model`` selects which propagating case to compare: ``"segment"`` (the
    active segment the profile defines) or ``"whole"`` (the whole polyline).
    For a profile with no localised generator the two coincide.

    A ratio below 1 means the stationary approximation *overestimates*: the
    wavefront's contributions at different arc positions partially cancel, and
    the event is spread over a transit far longer than its AP width. Returns
    1.0 if the stationary amplitude is degenerate.
    """
    prop = {"segment": signals.segment, "whole": signals.whole}
    if model not in prop:
        raise ValueError(f"model must be 'segment' or 'whole', got {model!r}")
    stat_amp = reduce(signals.stationary)
    if stat_amp <= 0:
        return 1.0
    return reduce(prop[model]) / stat_amp
