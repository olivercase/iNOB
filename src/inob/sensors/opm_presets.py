"""OPM sensor presets — noise floor *and* bandwidth, as a pair.

Why this module exists
----------------------
A magnetometer is described by two numbers, not one. Quoting an intrinsic
noise density without the bandwidth that goes with it produces the mistake
this table was written to stop: pairing a QuSpin QZFM-3 (7 fT/√Hz) with a
30–500 Hz recording band, when that sensor's own 3-dB point is ≈ 135 Hz. The
sensor cannot integrate noise over 470 Hz, and — more importantly — it cannot
pass a 0.5 ms compound action potential, whose Hermite-Gaussian spectrum peaks
near 1/(2πσ) ≈ 320 Hz. Noise was over-counted and signal loss was not counted
at all.

:mod:`inob.analysis.snr` therefore models the sensor as a single pole at
:attr:`OpmSensor.bandwidth_hz`, applied to signal and noise alike:

    |H(f)|² = 1 / (1 + (f / f₃dB)²)

so the noise integrates as ``f₃dB·[atan(hi/f₃dB) − atan(lo/f₃dB)]`` rather than
``hi − lo``, and the signal is scaled by ``|H(f_CAP)|``. Narrowing the band
then cannot flatter a narrow-band sensor: the noise it removes is noise the
signal was riding on.

Provenance
----------
These are manufacturer figures, entered here so they can be cited and changed
in one place. ``verified=False`` marks a preset whose numbers have not been
checked against a current spec sheet by anyone on this project — usable for
comparison, not for a published claim without re-checking.

A system not listed here needs no preset: set ``noise.opm_intrinsic_fT_sqrtHz``
and ``noise.opm_bandwidth_hz`` directly (in the YAML, or e.g.
``--set noise.opm_intrinsic_fT_sqrtHz=3 --set noise.opm_bandwidth_hz=1000``),
and either overrides whichever preset is named.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpmSensor:
    """One magnetometer's noise density and 3-dB bandwidth."""

    name: str
    label: str
    noise_fT_sqrtHz: float
    bandwidth_hz: float
    """Single-pole (3-dB) bandwidth, Hz. Bounds both signal and noise."""
    source: str
    verified: bool = True

    def gain(self, f_hz: float) -> float:
        """Single-pole amplitude response at ``f_hz`` (1.0 at DC)."""
        return 1.0 / (1.0 + (float(f_hz) / self.bandwidth_hz) ** 2) ** 0.5


OPM_SENSORS: dict[str, OpmSensor] = {
    "quspin_qzfm3": OpmSensor(
        name="quspin_qzfm3",
        label="QuSpin QZFM Gen-3",
        noise_fT_sqrtHz=7.0,
        bandwidth_hz=135.0,
        source=(
            "QuSpin QZFM Gen-3 spec: sensitivity 7-10 fT/√Hz, 3-dB "
            "bandwidth ~135 Hz (commonly quoted as ~150 Hz)."
        ),
    ),
    "quspin_qzfm2": OpmSensor(
        name="quspin_qzfm2",
        label="QuSpin QZFM Gen-2",
        noise_fT_sqrtHz=15.0,
        bandwidth_hz=135.0,
        source="QuSpin QZFM Gen-2 spec; the pre-Gen-3 baseline this project used.",
    ),
    "fieldline_v3": OpmSensor(
        name="fieldline_v3",
        label="FieldLine HEDscan v3",
        noise_fT_sqrtHz=15.0,
        bandwidth_hz=500.0,
        source="FieldLine HEDscan v3 manufacturer figures.",
        verified=False,
    ),
    "he4_wideband": OpmSensor(
        name="he4_wideband",
        label="Helium-4 OPM (wide-band)",
        noise_fT_sqrtHz=30.0,
        bandwidth_hz=2000.0,
        source=(
            "Helium-4 OPM class (e.g. Mag4Health): ~30-50 fT/√Hz with a "
            "kHz-scale bandwidth — the trade the alkali sensors do not offer."
        ),
        verified=False,
    ),
    # A SQUID magnetometer, for comparison on the same array geometry. The
    # table is named for OPMs because they are the default, but nothing in a
    # preset is OPM-specific: a sensor is a noise density and a bandwidth.
    # What does differ is geometry, and that is not in the preset — a SQUID
    # sits in a dewar a few centimetres off the skin (sensors.depth_mm) and
    # measures one axis (sensors.axes: radial).
    "megin_triux_mag": OpmSensor(
        name="megin_triux_mag",
        label="MEGIN (Elekta) TRIUX SQUID magnetometer",
        noise_fT_sqrtHz=5.0,
        bandwidth_hz=3300.0,
        source=(
            "Elekta Neuromag TRIUX Technical Manual NM24132A (2011), "
            "section 1.1: magnetometer white noise (60-70 Hz) max. "
            "5 fT/√Hz for 96% of channels, max. 10 fT/√Hz for all; "
            "acquisition low-pass a 6th-order Butterworth, adjustable "
            "330 Hz-3.3 kHz. A SQUID has no intrinsic roll-off in this "
            "band, so the bandwidth is the highest filter setting; set "
            "noise.opm_bandwidth_hz to the one you record with. MEGIN "
            "quotes < 3 fT/√Hz for the current TRIUX neo."
        ),
        verified=False,
    ),
}

DEFAULT_OPM_SENSOR = "quspin_qzfm3"


def opm_sensor(name: str) -> OpmSensor:
    """Look up a preset by name, with a listing in the error."""
    try:
        return OPM_SENSORS[name]
    except KeyError:
        known = ", ".join(sorted(OPM_SENSORS))
        raise KeyError(f"unknown OPM sensor preset {name!r}; known presets: {known}") from None


def cap_dominant_hz(ap_width_ms: float) -> float:
    """Spectral peak of the modelled CAP, Hz.

    The action-potential shape in :func:`inob.sources.cap.biphasic_waveform`
    is the first derivative of a Gaussian of σ = ``ap_width_ms``, whose
    Fourier magnitude ∝ f·exp(−2π²σ²f²) and therefore peaks at 1/(2πσ).
    """
    sigma_s = float(ap_width_ms) * 1e-3
    if sigma_s <= 0:
        raise ValueError(f"ap_width_ms must be positive, got {ap_width_ms}")
    return 1.0 / (2.0 * 3.141592653589793 * sigma_s)
