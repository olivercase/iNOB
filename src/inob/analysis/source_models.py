"""How OPM and surface electrodes compare on the *same* cord activity.

A like-for-like modality comparison, deliberately stripped of everything that
needs a physiological calibration to be believed.

The problem this exists to avoid
--------------------------------
Asking "how big is the spinal signal?" forces an absolute source strength, and
for the cord that number is contested: the travelling volley's equivalent
current dipole comes from magnetospinography, and the reported figures span
more than an order of magnitude. Any headline amplitude inherits that spread.

Asking "how do the two modalities compare, and what does propagation cost?"
does not. Both are ratios. Every quantity here is linear in the source moment
``Q``, so the modality gap (MEG SNR ÷ EEG SNR) and the propagation penalty
(propagating ÷ stationary) are **independent of Q entirely** — they survive
whatever the true source strength turns out to be. Absolute amplitudes are
reported too, but as "at this assumed Q", clearly a scaling of an assumption
rather than a prediction.

What is held fixed
------------------
For the comparison to be like-for-like, only *one* thing may vary at a time:

* **Same orientation.** All three source models use the longitudinal
  (cord-tangent) direction, via :func:`inob.sources.cap.longitudinal_leadfield`.
  Using each model's own best-aligned moment would let orientation differences
  masquerade as source-model differences.
* **Same moment per active source.** ``Q`` nA·m wherever the model puts
  current. This is the choice that makes the synchronous model the *naive
  upper bound* it is meant to be: it asks what you would predict if you assumed
  the whole cord fires at full strength at once, rather than redistributing a
  fixed total over a longer structure.
* **Same reduction per modality, throughout.** MEG is the best single channel;
  EEG is the best bipolar pair, because a surface potential exists only as a
  difference and a single referenced channel measures the reference as much as
  the source (see :func:`inob.analysis.snr.per_source_best_bipolar`).
* **Same noise floors** as every other figure in the package, from
  :func:`inob.analysis.snr.compute_noise_floors`.

The three source models
-----------------------
``synchronous``  every cord source active in phase at once. Not physiological
                 — an 8 ms transit against a sub-ms AP width says so — but it
                 is the assumption you make by default if you treat the cord
                 the way the vagus can legitimately be treated, so it belongs
                 in the comparison as the bound it is.
``stationary``   one segment's worth of current, lumped at a single source
                 under the array. The right model for a generator that does not
                 travel, and the reference the propagation correction is
                 measured against.
``ascending``    the same moment as a wavefront travelling rostrally at fibre
                 conduction velocity — the volley an SSEP actually evokes.

Nothing here needs the activity to be a physiologically faithful dorsal-column
volley. It needs current in the cord that is either stationary or ascending,
which is the distinction the modalities are being compared on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from inob.analysis.propagation import (
    compute_propagation_signals,
    is_ordered_polyline,
    peak_bipolar,
    peak_over_channels,
    propagation_ratio,
)
from inob.analysis.snr import compute_noise_floors
from inob.io.npz import Leadfield
from inob.sources.cap import longitudinal_leadfield

logger = logging.getLogger(__name__)

#: SNR a source model must reach to count as detected (Rose criterion), matching
#: :func:`inob.viz.detectability.required_trials`.
SNR_TARGET: float = 3.0


def _trials(signal: float, sigma: float, target: float = SNR_TARGET) -> float:
    """Averages to reach ``target`` SNR — see inob.viz.detectability."""
    if signal <= 0 or sigma <= 0:
        return float("inf")
    return max(1.0, float((target * sigma / signal) ** 2))


@dataclass(frozen=True)
class ModelRow:
    """One source model, measured by both modalities."""

    name: str
    description: str
    meg_fT: float
    eeg_uV: float
    meg_sigma_fT: float
    eeg_sigma_uV: float

    @property
    def meg_snr(self) -> float:
        return self.meg_fT / self.meg_sigma_fT

    @property
    def eeg_snr(self) -> float:
        return self.eeg_uV / self.eeg_sigma_uV

    @property
    def meg_trials(self) -> float:
        return _trials(self.meg_fT, self.meg_sigma_fT)

    @property
    def eeg_trials(self) -> float:
        return _trials(self.eeg_uV, self.eeg_sigma_uV)

    @property
    def modality_gap(self) -> float:
        """MEG SNR ÷ EEG SNR — how many times better the OPM array does.

        The one number in this class that does not depend on the assumed source
        strength: both SNRs are linear in Q, so it cancels. Above 1 favours
        OPMs; below 1 favours electrodes.
        """
        return self.meg_snr / self.eeg_snr if self.eeg_snr else float("inf")

    @property
    def trials_ratio(self) -> float:
        """EEG trials ÷ MEG trials — the same gap, squared, in trial counts."""
        return self.eeg_trials / self.meg_trials if self.meg_trials else float("inf")


@dataclass(frozen=True)
class SourceModelComparison:
    rows: tuple[ModelRow, ...]
    Q_nAm: float
    source_idx: int
    source_z_mm: float
    n_sources: int
    span_mm: float
    transit_ms: float
    ap_width_ms: float
    meg_propagation_factor: float
    eeg_propagation_factor: float

    def by_name(self, name: str) -> ModelRow:
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(name)

    @property
    def coherence_penalty(self) -> float:
        """Ascending ÷ synchronous, MEG — the cost of the naive assumption.

        Q-independent, like every other ratio here.
        """
        return self.by_name("ascending").meg_fT / self.by_name("synchronous").meg_fT


def compare_source_models(
    meg: Leadfield,
    eeg: Leadfield,
    profile,
    *,
    Q_nAm: float,
    source_idx: int,
    meg_sigma_fT: float,
    eeg_sigma_uV: float,
) -> SourceModelComparison:
    """Measure all three source models with both modalities.

    ``profile`` supplies only conduction velocity and AP width, which set how
    far the wavefront spreads — not the source strength, which is ``Q_nAm``.
    """
    L_meg, arc_mm, _ = longitudinal_leadfield(meg.L_fT_per_nAm, meg.source_pos)
    L_eeg, _, _ = longitudinal_leadfield(eeg.L_fT_per_nAm, eeg.source_pos)
    if L_eeg.shape[1] != L_meg.shape[1]:
        raise ValueError(
            f"MEG and EEG leadfields describe different source spaces "
            f"({L_meg.shape[1]} vs {L_eeg.shape[1]} sources) — they are from "
            "different solves and cannot be compared like for like"
        )

    def meg_amp(col: np.ndarray) -> float:
        return float(np.abs(col).max())

    def eeg_amp(col: np.ndarray) -> float:
        return float(col.max() - col.min())  # best bipolar pair

    rows: list[ModelRow] = []

    # 1. Synchronous: every source at Q, summed in phase.
    sync = L_meg.sum(axis=1) * Q_nAm
    sync_e = L_eeg.sum(axis=1) * Q_nAm
    rows.append(
        ModelRow(
            "synchronous",
            f"all {L_meg.shape[1]} cord sources active in phase, {Q_nAm:g} nA·m each",
            meg_amp(sync),
            eeg_amp(sync_e),
            meg_sigma_fT,
            eeg_sigma_uV,
        )
    )

    # 2. Stationary: one source at Q.
    stat = L_meg[:, source_idx] * Q_nAm
    stat_e = L_eeg[:, source_idx] * Q_nAm
    rows.append(
        ModelRow(
            "stationary",
            f"one segment at source #{source_idx}, {Q_nAm:g} nA·m, not moving",
            meg_amp(stat),
            eeg_amp(stat_e),
            meg_sigma_fT,
            eeg_sigma_uV,
        )
    )

    # 3. Ascending: the same moment, travelling. Measured as a ratio against
    #    the stationary case rather than in the wave simulation's own units —
    #    the simulation runs on the raw leadfield (uncalibrated for EEG), so
    #    only its ratios are meaningful, and they are exactly what is wanted.
    if not is_ordered_polyline(meg.source_pos):
        raise ValueError(
            "sources are not an ordered path, so 'ascending' has no meaning "
            "here (arc length along an unordered point cloud is not a distance)"
        )
    meg_factor = propagation_ratio(
        compute_propagation_signals(meg, profile, stationary_idx=source_idx),
        peak_over_channels,
    )
    eeg_factor = propagation_ratio(
        compute_propagation_signals(eeg, profile, stationary_idx=source_idx),
        peak_bipolar,
    )
    rows.append(
        ModelRow(
            "ascending",
            f"{Q_nAm:g} nA·m travelling rostrally at {profile.mean_cv_m_per_s:.0f} m/s",
            meg_amp(stat) * meg_factor,
            eeg_amp(stat_e) * eeg_factor,
            meg_sigma_fT,
            eeg_sigma_uV,
        )
    )

    span = float(arc_mm[-1] - arc_mm[0])
    return SourceModelComparison(
        rows=tuple(rows),
        Q_nAm=Q_nAm,
        source_idx=source_idx,
        source_z_mm=float(meg.source_pos[source_idx, 2]),
        n_sources=int(L_meg.shape[1]),
        span_mm=span,
        transit_ms=profile.transit_ms(span),
        ap_width_ms=profile.ap_width_ms,
        meg_propagation_factor=meg_factor,
        eeg_propagation_factor=eeg_factor,
    )


# ── report ─────────────────────────────────────────────────────────────────


def source_model_summary(
    cfg,
    *,
    Q_nAm: float | None = None,
    source_idx: int | None = None,
) -> dict:
    """Headline like-for-like numbers as a JSON-serialisable dict."""
    from inob.config import source_target_tag
    from inob.io.npz import load_leadfield
    from inob.physiology.profiles import profile_for_tag
    from inob.viz.detectability import default_source_idx

    meg = load_leadfield(cfg.outputs.forward_npz)
    eeg = load_leadfield(cfg.outputs.forward_eeg_npz)
    profile = profile_for_tag(source_target_tag(cfg))
    if source_idx is None:
        source_idx = default_source_idx(meg, eeg)
    if Q_nAm is None:
        Q_nAm = profile.default_strength_nAm
    floors = compute_noise_floors(cfg)

    cmp = compare_source_models(
        meg,
        eeg,
        profile,
        Q_nAm=Q_nAm,
        source_idx=source_idx,
        meg_sigma_fT=floors.meg_per_channel_fT,
        eeg_sigma_uV=floors.eeg_per_channel_uV,
    )
    for row in cmp.rows:
        logger.info(
            "[source-models] %-12s MEG %9.1f fT (SNR %6.2f, N=%s)  "
            "EEG %7.4f µV (SNR %6.3f, N=%s)  gap %.1fx",
            row.name,
            row.meg_fT,
            row.meg_snr,
            _fmt_n(row.meg_trials),
            row.eeg_uV,
            row.eeg_snr,
            _fmt_n(row.eeg_trials),
            row.modality_gap,
        )
    return {
        "Q_nAm": cmp.Q_nAm,
        "Q_note": (
            "Absolute amplitudes scale linearly with this; the modality gaps "
            "and propagation factors below do not depend on it at all."
        ),
        "source_idx": cmp.source_idx,
        "source_z_mm": cmp.source_z_mm,
        "n_sources": cmp.n_sources,
        "span_mm": cmp.span_mm,
        "transit_ms": cmp.transit_ms,
        "ap_width_ms": cmp.ap_width_ms,
        "noise": {
            "meg_fT": cmp.rows[0].meg_sigma_fT,
            "eeg_uV": cmp.rows[0].eeg_sigma_uV,
            "band_label": cfg.noise.band_label,
        },
        "models": {
            row.name: {
                "description": row.description,
                "meg_fT": row.meg_fT,
                "eeg_uV": row.eeg_uV,
                "meg_single_trial_SNR": row.meg_snr,
                "eeg_single_trial_SNR": row.eeg_snr,
                "meg_trials_for_SNR3": row.meg_trials,
                "eeg_trials_for_SNR3": row.eeg_trials,
                "modality_gap_snr": row.modality_gap,
                "modality_gap_trials": row.trials_ratio,
            }
            for row in cmp.rows
        },
        "q_independent": {
            "note": "These are ratios; they hold whatever the true source strength is.",
            "meg_propagation_factor": cmp.meg_propagation_factor,
            "eeg_propagation_factor": cmp.eeg_propagation_factor,
            "coherence_penalty_meg": cmp.coherence_penalty,
            "modality_gap_by_model": {row.name: row.modality_gap for row in cmp.rows},
        },
    }


def _fmt_n(n: float) -> str:
    return "∞" if not np.isfinite(n) else f"{n:,.0f}"
