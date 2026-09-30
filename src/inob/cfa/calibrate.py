"""Calibrate the amplitude of the cardiac source against the recorded head field.

The cardiac moment in ``sources.npz`` comes from the pilot pipeline, which fitted
it by inverting the recorded head data through the replication pack's forward
model. That model places the heart ~14 cm too low and has no neck, so its
conductor attenuates the field reaching the head; the fitted moment is inflated
to compensate. Driven through the whole-body FEM, the same moment predicts a head
field ~23x larger than was recorded.

The shape of the source is right (the T/QRS ratio of the head field is 0.67 here
against 0.72 recorded), so a single amplitude scale is the correct fix. It is the
least-squares scale between the simulated and recorded array-RMS time courses of
the radial field over -50..500 ms from the R-peak. The recorded field is the
R-locked average of the head OPMs' Z (radial) channels of the pilot recording.

    python -m inob.cfa.calibrate            # writes outputs/cfa_forward/calibration.json

:class:`inob.cfa.engine.Forward` applies the saved scale on load, so every
consumer of the forward model sees the calibrated source.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FS_REC = 1500.0                 # pilot recording sample rate, Hz
WINDOW_MS = (-50.0, 500.0)      # fit window relative to the R-peak
R_PEAK_MS = 400.0               # R-peak position in the simulation window


def recorded_rms(pack: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Array-RMS time course of the recorded, R-locked head field (fT).

    Returns (time_ms relative to the R-peak, rms, number of beats averaged).
    Same processing as the manuscript's real-data figure: 2-70 Hz band-pass,
    50 Hz notch, -100..0 ms baseline.
    """
    import scipy.io as sio
    from scipy.signal import butter, filtfilt, iirnotch

    raw = np.load(pack / "data/cache/real_head_z_ft.npy").astype(np.float64)
    rpk = sio.loadmat(pack / "data/real/ecg_cardiac_timing_record4.mat")["rPeakIdx"]
    rpk = rpk.ravel().astype(int) - 1
    b, a = butter(4, [2 / (FS_REC / 2), 70 / (FS_REC / 2)], "bandpass")
    dat = filtfilt(b, a, raw, axis=1)
    bn, an = iirnotch(50, 30, FS_REC)
    dat = filtfilt(bn, an, dat, axis=1)
    pre, post = int(0.4 * FS_REC), int(0.6 * FS_REC)
    peaks = rpk[(rpk > pre) & (rpk < dat.shape[1] - post)]
    ep = np.stack([dat[:, i - pre:i + post] for i in peaks])
    t = np.arange(-pre, post) / FS_REC * 1000.0
    ep -= ep[:, :, (t >= -100) & (t < 0)].mean(2, keepdims=True)
    return t, np.sqrt((ep.mean(0) ** 2).mean(0)), len(peaks)


def simulated_rms(fwd) -> tuple[np.ndarray, np.ndarray]:
    """Array-RMS time course of the simulated radial head field (fT), still head."""
    arr = fwd.array
    cfa = (arr.project(fwd.b_myo[0], arr.chanori) @ fwd.amps) * 1e15
    radial = cfa[:arr.n_site]                    # channel order: all radial first
    return fwd.time_s * 1000.0 - R_PEAK_MS, np.sqrt((radial ** 2).mean(0))


def fit_scale(fwd, pack: Path) -> dict:
    """Least-squares amplitude scale mapping the simulated field onto the recording."""
    tr, rec, n_beats = recorded_rms(pack)
    ts, sim = simulated_rms(fwd)
    tg = ts[(ts >= WINDOW_MS[0]) & (ts <= WINDOW_MS[1])]
    recg = np.interp(tg, tr, rec)
    simg = np.interp(tg, ts, sim)
    k = float(recg @ simg / (simg @ simg))

    def at(x, t, q):
        return float(x[np.argmin(np.abs(t - q))])

    def t_over_qrs(x):
        return float(x[(tg > 150) & (tg < 450)].max() / x[(tg > -30) & (tg < 60)].max())

    return {"amp_scale": k, "window_ms": list(WINDOW_MS), "n_beats_recorded": n_beats,
            "recorded_rms_ft": {"r_peak": at(recg, tg, 0), "hef_latency": at(recg, tg, 250)},
            "simulated_rms_ft_uncalibrated": {"r_peak": at(simg, tg, 0),
                                              "hef_latency": at(simg, tg, 250)},
            "t_over_qrs": {"recorded": t_over_qrs(recg), "simulated": t_over_qrs(simg)},
            "shape_corr": float(np.corrcoef(recg, simg)[0, 1])}


def main() -> int:
    import sys
    logging.basicConfig(level="INFO", format="%(message)s")
    root = Path(__file__).resolve().parents[3]
    pack = Path("/Users/olivercase/Desktop/CFA/replication pack")
    sys.path.insert(0, str(root / "src"))
    from inob.cfa.array import CFAArray
    from inob.cfa.engine import Forward

    d = root / "outputs/cfa_forward"
    cal = d / "calibration.json"
    if cal.exists():
        cal.unlink()                             # fit against the RAW source
    fwd = Forward.load(d, CFAArray.from_reference(root / "outputs/_cfa_ref.npz"))
    out = fit_scale(fwd, pack)
    cal.write_text(json.dumps(out, indent=2))
    logger.info("[calibrate] amplitude scale %.4f (1/%.1f); T/QRS recorded %.2f, "
                "simulated %.2f -> %s", out["amp_scale"], 1 / out["amp_scale"],
                out["t_over_qrs"]["recorded"], out["t_over_qrs"]["simulated"], cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
