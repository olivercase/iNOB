"""Detectability for the planning question / GUI ``Simulate`` button.

Given source positions (already solved into the leadfield) plus per-source
strengths, a noise floor and an SNR threshold, answer: **how many averaged
trials are needed to detect each source?**

    single-trial SNR_s = (peak_s · Q_s) / σ_noise
    trials_needed_s     = ceil( (threshold / SNR_s)² )   (white-noise √n averaging)

where

  * ``peak_s`` is the canonical best-channel peak leadfield amplitude
    (:func:`inob.analysis.snr.per_source_peak`) — the same definition the
    detectability figure uses, so the GUI number and the figure never diverge;
  * ``Q_s`` is the source strength in nA·m (the leadfield is stored per nA·m,
    so SNR scales linearly with strength — no re-solve needed);
  * ``σ_noise`` is the per-channel modality noise floor
    (:func:`inob.analysis.snr.compute_noise_floors`).

The result dict matches the web API contract (``gui/web/API_CONTRACT.md``).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from inob.analysis.snr import compute_noise_floors, per_source_peak
from inob.config import Config
from inob.io.npz import load_leadfield
from inob.physiology.profiles import profile_for

logger = logging.getLogger(__name__)


def _coerce_strengths(
    strengths_nAm,
    n_sources: int,
    *,
    default_nAm: float = 70.0,
) -> np.ndarray:
    """Return an (S,) strength array, broadcasting a scalar / resizing a list."""
    if strengths_nAm is None:
        return np.full(n_sources, default_nAm)
    arr = np.asarray(strengths_nAm, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(n_sources, default_nAm)
    if arr.size == 1:
        return np.full(n_sources, float(arr[0]))
    if arr.size != n_sources:
        logger.warning(
            "strengths length %d != %d sources; resizing",
            arr.size,
            n_sources,
        )
        return np.resize(arr, n_sources)
    return arr


def compute_detectability(
    cfg: Config,
    *,
    strengths_nAm=None,
    threshold_snr: float = 3.0,
    modality: str = "meg",
) -> dict:
    """Per-source SNR + trials-to-detect from a solved leadfield.

    ``modality`` is ``"meg"`` (uses ``cfg.outputs.forward_npz``) or ``"eeg"``
    (uses ``cfg.outputs.forward_eeg_npz``). The leadfield's ``source_pos`` are
    the positions that were solved — for the GUI flow these are the clicked
    ``cfg.forward.point_sources``. Raises ``FileNotFoundError`` if the required
    leadfield has not been computed yet.
    """
    modality = modality.lower()
    floors = compute_noise_floors(cfg)
    if modality == "eeg":
        lf = load_leadfield(cfg.outputs.forward_eeg_npz)
        sigma = float(floors.eeg_per_channel_uV)
        unit = "uV"
    elif modality == "meg":
        lf = load_leadfield(cfg.outputs.forward_npz)
        sigma = float(floors.meg_per_channel_fT)
        unit = "fT"
    else:
        raise ValueError(f"modality must be 'meg' or 'eeg', got {modality!r}")

    peak = per_source_peak(lf.L_fT_per_nAm)  # (S,) fT/nAm or µV/nAm
    n_sources = int(peak.shape[0])
    pos = np.asarray(lf.source_pos, dtype=float)
    # Default event strength comes from the target's physiology rather than a
    # single hardcoded figure: a spinal dorsal-column volley (~5 nA·m) and the
    # cervical-vagus A+C summation reference (70 nA·m) differ by more than an
    # order of magnitude, and trials-to-detect scales as 1/strength².
    # The vagus and muscle profiles carry the historical 70 nA·m, so their
    # results are unchanged.
    profile = profile_for(cfg)
    strengths = _coerce_strengths(
        strengths_nAm,
        n_sources,
        default_nAm=profile.default_strength_nAm,
    )
    if strengths_nAm is None:
        logger.info(
            "[detect] source strength %.2f nA·m from the %s physiology profile (%s)",
            profile.default_strength_nAm,
            profile.label,
            profile.paradigm,
        )

    snr = (peak * strengths) / sigma if sigma > 0 else np.full(n_sources, np.inf)
    thr = float(threshold_snr)

    per_source = []
    for i in range(n_sources):
        s = float(snr[i])
        trials = math.ceil((thr / s) ** 2) if s > 0 else -1  # -1 == never
        per_source.append(
            {
                "index": i,
                "x": round(float(pos[i, 0]), 2),
                "y": round(float(pos[i, 1]), 2),
                "z": round(float(pos[i, 2]), 2),
                "strength_nAm": round(float(strengths[i]), 4),
                "snr": round(s, 4),
                "trials_needed": trials,
            }
        )

    finite = snr[np.isfinite(snr)]
    return {
        "modality": modality,
        "per_source": per_source,
        "array": {
            "n_sensors": int(lf.coil_pos.shape[0]),
            "mean_snr": round(float(finite.mean()), 4) if finite.size else 0.0,
            "max_snr": round(float(finite.max()), 4) if finite.size else 0.0,
            "noise_floor_fT": round(sigma, 4),  # contract key (fT for MEG; µV for EEG)
            "noise_unit": unit,
            "threshold_snr": thr,
        },
    }
