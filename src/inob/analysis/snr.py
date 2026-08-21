"""SNR + sensitivity analysis for the dual-modality leadfields.

Given a leadfield ``L`` (channels × 3*S) and a per-modality noise floor,
compute the predicted single-CAP SNR (and post-averaging SNR) per source.

Noise floors (set in ``cfg.noise``):

  OPM (magnetic):    σ_n_meg = opm_intrinsic_fT_sqrtHz × √BW_opm / |H(f_CAP)|
  HD-EMG (electric): σ_n_eeg = √(amp_noise² + Johnson²) × √BW      [µV]

where BW is the recording passband (``noise.band_lo_hz``…``band_hi_hz``,
default 30–500 Hz — what evoked-potential recording actually uses) and the
Johnson term is the thermal noise of the electrode-skin contact impedance.

The OPM is not a flat window on that band. It is modelled as a single pole at
``noise.opm_bandwidth_hz`` (from the sensor preset — see
:mod:`inob.sensors.opm_presets`), which does two things:

  * the noise integrates over ``BW_opm = f₃dB·[atan(hi/f₃dB) − atan(lo/f₃dB)]``
    rather than the nominal ``hi − lo``;
  * the signal is scaled by ``|H(f_CAP)| = 1/√(1 + (f_CAP/f₃dB)²)`` at the CAP's
    spectral peak ``f_CAP = 1/(2πσ)``, σ = the profile's ``ap_width_ms``.

Both are folded into the returned MEG σ, so every caller — trials-to-detect,
detectability figures, source-model comparisons — sees one *effective* noise
floor and none of them can accidentally reward a narrow-band sensor for the
noise its own bandwidth removes. Dividing σ by the gain is exactly equivalent
to attenuating the signal, and keeps the amplitude pipeline untouched.

Concretely, for a 0.5 ms vagal CAP (f_CAP ≈ 318 Hz) over a 30–500 Hz band:
a QuSpin QZFM-3 (7 fT/√Hz, 135 Hz) gives σ_eff ≈ 217 fT, not the 152 fT the
un-rolled-off arithmetic used to report; a helium-4 wide-band OPM
(30 fT/√Hz, 2 kHz) gives ≈ 650 fT. The narrow-band sensor still wins here,
but by a stated factor rather than by an unstated assumption.

The "signal" we use is the dipole-moment-norm-projected leadfield amplitude:
    |L_eff(s)| = √(L[:, 3s : 3s+3] @ q_unit · |q_unit|²)
With q_unit = unit dipole moment (the moment direction with the largest
projection by default — call ``snr_per_source(L, ..., moment="rms")`` for an
isotropic average).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from inob.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoiseFloors:
    meg_per_channel_fT: float          # σ_n in fT for one OPM channel, effective
    eeg_per_channel_uV: float          # σ_n in µV for one HD electrode × bandwidth
    meg_sensor_gain: float = 1.0       # |H(f_CAP)| — the CAP fraction the OPM passes
    meg_noise_bandwidth_hz: float = 0.0  # equivalent noise BW of band ∩ sensor
    signal_hz: float = 0.0             # f_CAP the gain was evaluated at


def compute_noise_floors(
    cfg: Config, *, signal_hz: float | None = None,
) -> NoiseFloors:
    """Compute σ_n in fT (OPM) and µV (HD electrode) over the recording band.

    The MEG figure is the *effective* floor: noise integrated over the sensor-
    weighted band and then divided by the sensor's gain at the CAP frequency,
    so comparing it against an un-filtered signal amplitude gives the SNR the
    sensor really delivers. ``signal_hz`` overrides the frequency taken from
    the target's physiology profile (``1/(2π·ap_width_ms)``).
    """
    n = cfg.noise
    bw = n.effective_bandwidth_hz
    meg_bw = n.opm_noise_bandwidth_hz
    if signal_hz is None:
        signal_hz = _cap_frequency(cfg)
    gain = n.sensor.gain(signal_hz)
    meg_sigma = float(n.opm_intrinsic_fT_sqrtHz) * np.sqrt(meg_bw) / gain
    # Johnson voltage noise of the contact resistance:
    #   v_n = √(4 k_B T R)  in V/√Hz; with R in kΩ:
    R_ohm = float(n.eeg_electrode_skin_kohm) * 1e3
    k_B = 1.380649e-23
    T = 310.15        # body temp (37 °C)
    v_johnson_per_sqrtHz = np.sqrt(4.0 * k_B * T * R_ohm) * 1e6   # → µV/√Hz
    amp = float(n.eeg_amplifier_uV_sqrtHz)
    eeg_per_sqrtHz = np.sqrt(amp ** 2 + v_johnson_per_sqrtHz ** 2)
    eeg_sigma = eeg_per_sqrtHz * np.sqrt(bw)
    logger.info(
        "Noise floors: OPM=%.2f fT (%s: %.1f fT/√Hz, pole %g Hz → BW_eff %.0f Hz, "
        "gain %.2f at f_CAP=%.0f Hz)  EEG=%.3f µV "
        "(amp=%.2f + Johnson=%.3f µV/√Hz, BW=%g Hz)",
        meg_sigma, n.opm_sensor, n.opm_intrinsic_fT_sqrtHz, n.opm_bandwidth_hz,
        meg_bw, gain, signal_hz, eeg_sigma, amp, v_johnson_per_sqrtHz, bw,
    )
    return NoiseFloors(
        meg_per_channel_fT=meg_sigma,
        eeg_per_channel_uV=eeg_sigma,
        meg_sensor_gain=float(gain),
        meg_noise_bandwidth_hz=float(meg_bw),
        signal_hz=float(signal_hz),
    )


def _cap_frequency(cfg: Config) -> float:
    """Spectral peak of the target's CAP, Hz — the frequency the OPM must pass.

    Imported lazily: :mod:`inob.physiology.profiles` pulls in the source model,
    and the noise floor is wanted by modules that have no other reason to.
    """
    from inob.physiology.profiles import profile_for
    from inob.sensors.opm_presets import cap_dominant_hz
    return cap_dominant_hz(profile_for(cfg).ap_width_ms)


def per_source_amplitude(
    L_human_units: np.ndarray, *, moment: str = "rms",
) -> np.ndarray:
    """Per-source signal amplitude (across channels) from a leadfield.

    ``L_human_units`` has shape (C, 3*S). Returns an (S,) array of dipole-moment
    -averaged channel-RMS amplitudes (fT for MEG, µV for EEG).

    ``moment="rms"`` averages the three orthogonal moments (isotropic source);
    ``moment="max"`` reports the worst-case (best-aligned) moment.
    """
    C, three_S = L_human_units.shape
    if three_S % 3 != 0:
        raise ValueError(f"L second dim {three_S} not divisible by 3")
    S = three_S // 3
    L = L_human_units.reshape(C, S, 3)
    if moment == "rms":
        per_chan = np.linalg.norm(L, axis=2) / np.sqrt(3.0)
    elif moment == "max":
        per_chan = np.linalg.norm(L, axis=2)
    else:
        raise ValueError(f"unknown moment={moment!r} (use 'rms' or 'max')")
    return np.sqrt(np.mean(per_chan ** 2, axis=0))    # RMS across channels


def per_source_peak(L_human_units: np.ndarray) -> np.ndarray:
    """Best-channel, best-moment peak |L| per source — shape (C, 3*S) → (S,).

    This is the CANONICAL detectability signal used across the package
    (``inob.viz.detectability`` and ``inob.analysis.detect`` both call it):
    the single most-sensitive channel/orientation for each source. The
    rationale for the trials-to-detect question is that one well-placed sensor
    suffices to see the CAP, so the relevant signal is the best-channel peak,
    not the array RMS (which ``per_source_amplitude`` returns and which answers
    a different, array-averaged question). Keeping one definition here prevents
    the two code paths from disagreeing by an orientation/array factor.
    """
    C, three_S = L_human_units.shape
    if three_S % 3 != 0:
        raise ValueError(f"L second dim {three_S} not divisible by 3")
    S = three_S // 3
    return np.abs(L_human_units.reshape(C, S, 3)).max(axis=(0, 2))


def per_source_best_bipolar(L_human_units: np.ndarray) -> np.ndarray:
    """Largest potential difference the array can form, per source — (S,).

    For each source and moment direction this is ``max_c L − min_c L`` over the
    channels: the best bipolar pair in the array. Use for EEG; meaningless for
    MEG, where each channel is already a difference-free field measurement.

    Why this and not :func:`per_source_peak`
    ----------------------------------------
    A surface potential is only defined up to a per-source additive constant,
    so a single channel's value depends entirely on the reference. The saved
    EEG leadfield is common-average referenced (see :mod:`inob.forward.eeg`),
    which for a patch spanning a few centimetres removes most of the amplitude
    a deep source produces — the potential is nearly flat across that
    footprint, and the average takes the flat part away.

    A *difference* between two contacts is immune to that: subtracting a common
    constant from every channel leaves it unchanged. So this quantity is what
    the electrode array can actually measure, independent of how it happens to
    be referenced, and it is recoverable from an existing leadfield without
    re-solving. On the cervical spine patch it runs ~1.7x above the referenced
    single-channel peak; on the whole-body array, where contacts are metres
    rather than centimetres apart, the gap is far larger — which is the point
    the whole-body-vs-patch comparison in :mod:`inob.viz.location_optimisation`
    exists to make.
    """
    C, three_S = L_human_units.shape
    if three_S % 3 != 0:
        raise ValueError(f"L second dim {three_S} not divisible by 3")
    S = three_S // 3
    L3 = L_human_units.reshape(C, S, 3)
    # max_ij |L_i - L_j| = max_i L_i - min_i L_i, so this is O(C·S) rather than
    # the O(C²·S) all-pairs form — which matters for the 1000-contact array.
    return (L3.max(axis=0) - L3.min(axis=0)).max(axis=1)


def snr_per_source(
    L_human_units: np.ndarray, sigma_noise: float, *,
    moment: str = "rms", n_averages: int = 1,
) -> np.ndarray:
    """Predicted SNR per source after ``n_averages`` repetitions.

    Assumes uncorrelated white noise; SNR scales as √n_averages.
    """
    sig = per_source_amplitude(L_human_units, moment=moment)
    return (sig / sigma_noise) * np.sqrt(max(1, int(n_averages)))


def array_summary(snr: np.ndarray) -> dict[str, float]:
    """Quick one-liner stats for an SNR-per-source array."""
    return {
        "n_sources": len(snr),
        "min": float(snr.min()),
        "p5":  float(np.percentile(snr, 5)),
        "p50": float(np.percentile(snr, 50)),
        "p95": float(np.percentile(snr, 95)),
        "max": float(snr.max()),
    }
