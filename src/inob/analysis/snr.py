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
    meg_per_channel_fT: float  # σ_n in fT for one OPM channel, effective
    eeg_per_channel_uV: float  # σ_n in µV for one HD electrode × bandwidth
    meg_sensor_gain: float = 1.0  # |H(f_CAP)| — the CAP fraction the OPM passes
    meg_noise_bandwidth_hz: float = 0.0  # equivalent noise BW of band ∩ sensor
    signal_hz: float = 0.0  # f_CAP the gain was evaluated at


def compute_noise_floors(
    cfg: Config,
    *,
    signal_hz: float | None = None,
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
    T = float(n.eeg_contact_temperature_K)
    v_johnson_per_sqrtHz = np.sqrt(4.0 * k_B * T * R_ohm) * 1e6  # → µV/√Hz
    amp = float(n.eeg_amplifier_uV_sqrtHz)
    eeg_per_sqrtHz = np.sqrt(amp**2 + v_johnson_per_sqrtHz**2)
    eeg_sigma = eeg_per_sqrtHz * np.sqrt(bw)
    logger.info(
        "Noise floors: OPM=%.2f fT (%s: %.1f fT/√Hz, pole %g Hz → BW_eff %.0f Hz, "
        "gain %.2f at f_CAP=%.0f Hz)  EEG=%.3f µV "
        "(amp=%.2f + Johnson=%.3f µV/√Hz, BW=%g Hz)",
        meg_sigma,
        n.opm_sensor,
        n.opm_intrinsic_fT_sqrtHz,
        n.opm_bandwidth_hz,
        meg_bw,
        gain,
        signal_hz,
        eeg_sigma,
        amp,
        v_johnson_per_sqrtHz,
        bw,
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


#: Named dipole orientations, as unit vectors in model coordinates. Anterior
#: is -Y (docs/COORDINATES.md), so "posterior" -- a dipole pointing out of the
#: back -- is +Y. Every amplitude here is sign-symmetric (|L| or max - min), so
#: a direction and its opposite give the same number.
NAMED_ORIENTATIONS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "posterior": (0.0, 1.0, 0.0),
    "anterior": (0.0, -1.0, 0.0),
}


def parse_orientation(spec: str | None) -> np.ndarray | None:
    """Turn an orientation spec into a unit vector, or ``None`` for "best".

    Accepts ``best`` (or ``None``), a name from :data:`NAMED_ORIENTATIONS`, or
    three comma-separated components ``vx,vy,vz`` in model coordinates, which
    are normalised. ``None`` means "keep the per-function best-of-three-axes
    maximisation", so passing it everywhere reproduces the historic numbers.
    """
    if spec is None:
        return None
    key = str(spec).strip().lower()
    if key == "best":
        return None
    if key in NAMED_ORIENTATIONS:
        return np.asarray(NAMED_ORIENTATIONS[key], dtype=float)
    parts = key.split(",")
    try:
        vec = np.asarray([float(v) for v in parts], dtype=float)
    except ValueError:
        vec = np.empty(0)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(
            f"orientation {spec!r}: use best, one of "
            f"{', '.join(NAMED_ORIENTATIONS)}, or three numbers vx,vy,vz"
        )
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError(f"orientation {spec!r} is the zero vector")
    return vec / norm


def _moment_stack(L_human_units: np.ndarray, orientation) -> np.ndarray:
    """(C, 3S) leadfield → (C, S, k) per-moment responses.

    ``k = 3`` (the Cartesian x, y, z unit moments) when ``orientation`` is
    ``None``; ``k = 1`` for a fixed orientation, where the column is the
    linear combination ``L[:, 3s:3s+3] @ q`` -- exactly the response of a
    unit dipole pointing along ``q``.
    """
    C, three_S = L_human_units.shape
    if three_S % 3 != 0:
        raise ValueError(f"L second dim {three_S} not divisible by 3")
    L3 = L_human_units.reshape(C, three_S // 3, 3)
    if orientation is None:
        return L3
    q = np.asarray(orientation, dtype=float).reshape(3)
    return (L3 @ q)[:, :, None]


def per_source_amplitude(
    L_human_units: np.ndarray,
    *,
    moment: str = "rms",
    orientation=None,
) -> np.ndarray:
    """Per-source signal amplitude (across channels) from a leadfield.

    ``L_human_units`` has shape (C, 3*S). Returns an (S,) array of dipole-moment
    -averaged channel-RMS amplitudes (fT for MEG, µV for EEG).

    ``moment="rms"`` averages the three orthogonal moments (isotropic source);
    ``moment="max"`` takes, per channel, the norm over the three moments --
    the response to the dipole orientation that channel is most sensitive to,
    which may differ from channel to channel.

    ``orientation`` (a unit vector, see :func:`parse_orientation`) fixes one
    dipole direction for every channel instead and overrides ``moment``.
    """
    if orientation is not None:
        per_chan = np.abs(_moment_stack(L_human_units, orientation)[:, :, 0])
        return np.sqrt(np.mean(per_chan**2, axis=0))
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
    return np.sqrt(np.mean(per_chan**2, axis=0))  # RMS across channels


def per_source_peak(L_human_units: np.ndarray, *, orientation=None) -> np.ndarray:
    """Best-channel, best-moment peak |L| per source — shape (C, 3*S) → (S,).

    "Best" means precisely: the maximum of ``|L[c, 3s + k]|`` over every
    channel ``c`` and over the three Cartesian unit moments ``k`` (x, y, z).
    It is *not* maximised over oblique orientations, which can exceed it by
    up to sqrt(3). With ``orientation`` (a unit vector, see
    :func:`parse_orientation`) the moment is fixed and only the channel is
    maximised.

    This is the CANONICAL detectability signal used across the package
    (``inob.viz.detectability`` and ``inob.analysis.detect`` both call it):
    the single most-sensitive channel/orientation for each source. The
    rationale for the trials-to-detect question is that one well-placed sensor
    suffices to see the CAP, so the relevant signal is the best-channel peak,
    not the array RMS (which ``per_source_amplitude`` returns and which answers
    a different, array-averaged question). Keeping one definition here prevents
    the two code paths from disagreeing by an orientation/array factor.
    """
    return np.abs(_moment_stack(L_human_units, orientation)).max(axis=(0, 2))


def per_source_triaxial_peak(L_human_units: np.ndarray, *, orientation=None) -> np.ndarray:
    """Best-*position*, best-moment field magnitude per source — (3N, 3S) → (S,).

    For a triaxial magnetometer array laid out as this package writes it —
    all N radial channels, then all N T1, then all N T2 — this takes the
    vector magnitude over the three axes at each position before maximising.

    Why it differs from :func:`per_source_peak`
    -------------------------------------------
    ``per_source_peak`` maximises over channels, so for a triaxial sensor it
    returns the largest single *axis*. Which axis that is depends on how the
    two tangents happen to be oriented, and the tangential frame is a
    convention (see :func:`inob.sensors.triaxial.tangent_frame`), not a
    property of the sensor. On the shipped array, rotating that frame moves
    individual per-source peaks by up to 40% while leaving the field itself
    untouched.

    A triaxial sensor measures all three components at once, so any single
    orientation is recoverable from them after the fact and the physically
    available signal at a position is the vector magnitude. That quantity is
    invariant to the tangential convention, which is the reason to prefer it
    for a triaxial array. It is an upper bound on ``per_source_peak`` and
    exceeds it by at most sqrt(3).

    It is *not* the right measure for a single-axis sensor: a radial-only
    SQUID cannot form this combination. Use ``per_source_peak`` restricted to
    the radial block for that case.

    ``orientation`` fixes the dipole direction as in :func:`per_source_peak`.
    """
    C = L_human_units.shape[0]
    if C % 3 != 0:
        raise ValueError(f"L first dim {C} not divisible by 3 (needs R/T1/T2 blocks)")
    stack = _moment_stack(L_human_units, orientation)  # (C, S, k)
    n = C // 3
    axes = stack.reshape(3, n, *stack.shape[1:])  # (axis, position, source, moment)
    magnitude = np.sqrt((axes**2).sum(axis=0))  # (position, source, moment)
    return magnitude.max(axis=(0, 2))


def per_source_best_bipolar(L_human_units: np.ndarray, *, orientation=None) -> np.ndarray:
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

    "Best" is the maximum over every contact pair and over the three Cartesian
    unit moments (x, y, z); ``orientation`` fixes the moment instead. It is
    also an upper bound on the single-channel peak under *any* reference that
    is one of the array's own contacts (or an average of them), since such a
    channel is itself a difference of array potentials -- see
    :mod:`inob.analysis.reference`.
    """
    L3 = _moment_stack(L_human_units, orientation)
    # max_ij |L_i - L_j| = max_i L_i - min_i L_i, so this is O(C·S) rather than
    # the O(C²·S) all-pairs form — which matters for the 1000-contact array.
    return (L3.max(axis=0) - L3.min(axis=0)).max(axis=1)


def snr_per_source(
    L_human_units: np.ndarray,
    sigma_noise: float,
    *,
    moment: str = "rms",
    n_averages: int = 1,
    orientation=None,
) -> np.ndarray:
    """Predicted SNR per source after ``n_averages`` repetitions.

    Assumes uncorrelated white noise; SNR scales as √n_averages. A fixed
    ``orientation`` overrides ``moment`` (see :func:`per_source_amplitude`).
    """
    sig = per_source_amplitude(L_human_units, moment=moment, orientation=orientation)
    return (sig / sigma_noise) * np.sqrt(max(1, int(n_averages)))


def array_summary(snr: np.ndarray) -> dict[str, float]:
    """Quick one-liner stats for an SNR-per-source array."""
    return {
        "n_sources": len(snr),
        "min": float(snr.min()),
        "p5": float(np.percentile(snr, 5)),
        "p50": float(np.percentile(snr, 50)),
        "p95": float(np.percentile(snr, 95)),
        "max": float(snr.max()),
    }
