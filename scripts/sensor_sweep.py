#!/usr/bin/env python3
"""What do the sensor's noise and its distance from the skin cost?

Two numbers a user sets for any magnetometer array are its noise density
and its stand-off from the skin: a wearable OPM sits a few millimetres out,
a second-generation cell a little further, a cryogenic dewar tens of
millimetres. This script moves both for one source and reports what they do
to the amplitude and the averaging budget, so the choice can be made on the
numbers rather than the spec sheet.

For each stand-off it rebuilds the magnetometer array exactly as ``inob
sensors`` would (same cylinder, spacing and tangent basis), cropped to a band
around the source so the solve stays cheap, and solves the forward problem
for that one source on the current FEM with DUNEuro. Each amplitude is then
judged against each noise preset's effective floor for the target's own CAP,
the same way ``inob detect`` judges it:

    python3 scripts/sensor_sweep.py                       # spine, C7 source
    python3 scripts/sensor_sweep.py --standoff-mm 10 20 --preset quspin_qzfm3 quspin_qzfm2

The source defaults to the one ``inob detect`` reports for the target (the
source nearest the electrode patch); ``--source-idx`` picks another from the
target's leadfield. Needs duneuropy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Row:
    standoff_mm: float
    preset: str
    noise_fT_sqrtHz: float
    bandwidth_hz: float
    n_positions: int
    peak_fT_per_nAm: float  # best channel and best moment, as inob detect
    radial_peak_fT_per_nAm: float  # radial channels only (a single-axis array)
    floor_fT: float
    moment_nAm: float
    snr_single_trial: float
    trials: float
    radial_trials: float


def solve_one_source(cfg, fem, src_mm: np.ndarray, sensors) -> np.ndarray:
    """MEG leadfield (channels, 3) in fT/(nA·m) for one source position."""
    from inob.forward.duneuro_driver import (
        attach_coils,
        build_driver,
        build_orthogonal_dipoles,
        build_source_model_config,
        compute_meg_leadfield,
        import_duneuro,
    )

    dp = import_duneuro(cfg)
    driver, driver_cfg, _ = build_driver(cfg, fem)
    attach_coils(driver, dp, sensors.coilpos, sensors.coilori)
    T_raw, _ = driver.computeMEGTransferMatrix(driver_cfg)
    dipoles = build_orthogonal_dipoles(dp, src_mm[None, :])
    driver_cfg["source_model"] = build_source_model_config(cfg)
    return compute_meg_leadfield(driver, np.array(T_raw), dipoles, driver_cfg) * 1e6


def build_band_array(cfg, standoff_mm: float, z_lo: float, z_hi: float):
    """The ``inob sensors`` array at ``standoff_mm``, cropped to [z_lo, z_hi]."""
    from inob.io.stl import load_stl
    from inob.sensors.triaxial import build_triaxial, cylindrical_raycast

    mesh = load_stl(cfg.data.torso_skin)
    positions, normals = cylindrical_raycast(
        mesh,
        resolution_mm=cfg.sensors.resolution_mm,
        z_min=z_lo,
        z_max=z_hi,
        cylinder_radius_factor=cfg.sensors.cylinder_radius_factor,
        angular_margin_deg=cfg.sensors.angular_margin_deg,
    )
    return build_triaxial(
        positions + normals * standoff_mm, normals, tangent_basis=cfg.sensors.tangent_basis
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--target", default="spine", help="source target (default spine)")
    ap.add_argument("--source-idx", type=int, default=-1)
    ap.add_argument("--standoff-mm", type=float, nargs="+", default=[6.5, 10.0, 20.0, 30.0])
    ap.add_argument(
        "--preset",
        nargs="+",
        default=["quspin_qzfm3", "quspin_qzfm2"],
        help="noise presets (see inob.sensors.opm_presets)",
    )
    ap.add_argument(
        "--band-mm", type=float, default=150.0, help="array half-height about the source"
    )
    ap.add_argument("--json", type=Path, help="also write the rows here")
    args = ap.parse_args(argv)

    from inob.analysis.snr import compute_noise_floors, per_source_peak
    from inob.cli._common import apply_source_target
    from inob.config import load_config
    from inob.io.hdf5 import load_fem
    from inob.io.npz import load_leadfield
    from inob.physiology.profiles import profile_for
    from inob.sensors.layout import radial_channels
    from inob.viz.detectability import default_source_idx, required_trials

    base = apply_source_target(load_config(args.config), args.target)
    meg_lf = load_leadfield(base.outputs.forward_npz)
    eeg_path = base.outputs.forward_eeg_npz
    eeg_lf = load_leadfield(eeg_path) if eeg_path.exists() else None
    idx = args.source_idx if args.source_idx >= 0 else default_source_idx(meg_lf, eeg_lf)
    src = np.asarray(meg_lf.source_pos, float)[idx]
    Q = profile_for(base).event_moment_nAm
    fem = load_fem(base.outputs.fem_mat)
    print(f"{args.target}: source #{idx} at z = {src[2]:.0f} mm, Q = {Q:.2f} nA·m", file=sys.stderr)

    rows: list[Row] = []
    for d in args.standoff_mm:
        arr = build_band_array(base, d, src[2] - args.band_mm, src[2] + args.band_mm)
        t0 = time.time()
        L = solve_one_source(base, fem, src, arr)
        peak = float(per_source_peak(L)[0])
        radial = float(per_source_peak(L[radial_channels(arr.labels)])[0])
        n_pos = len(arr.coilpos) // 3
        print(f"  stand-off {d:g} mm: {n_pos} positions, {time.time() - t0:.0f} s", file=sys.stderr)
        for name in args.preset:
            cfg = apply_source_target(
                load_config(args.config, overrides=[f"noise.opm_sensor={name}"]), args.target
            )
            sigma = compute_noise_floors(cfg).meg_per_channel_fT
            rows.append(
                Row(
                    standoff_mm=d,
                    preset=name,
                    noise_fT_sqrtHz=float(cfg.noise.opm_intrinsic_fT_sqrtHz),
                    bandwidth_hz=float(cfg.noise.opm_bandwidth_hz),
                    n_positions=n_pos,
                    peak_fT_per_nAm=peak,
                    radial_peak_fT_per_nAm=radial,
                    floor_fT=sigma,
                    moment_nAm=Q,
                    snr_single_trial=peak * Q / sigma,
                    trials=required_trials(peak * Q, sigma),
                    radial_trials=required_trials(radial * Q, sigma),
                )
            )

    print(
        f"{'stand-off':>9}  {'preset':<14}{'fT/√Hz':>7}{'fT/nA·m':>9}{'radial':>8}"
        f"{'floor fT':>9}{'signal fT':>10}{'SNR':>6}{'trials':>8}{'radial':>8}"
    )
    for r in rows:
        print(
            f"{r.standoff_mm:>7g}mm  {r.preset:<14}{r.noise_fT_sqrtHz:>7g}"
            f"{r.peak_fT_per_nAm:>9.2f}{r.radial_peak_fT_per_nAm:>8.2f}{r.floor_fT:>9.1f}"
            f"{r.peak_fT_per_nAm * r.moment_nAm:>10.1f}{r.snr_single_trial:>6.2f}"
            f"{r.trials:>8.0f}{r.radial_trials:>8.0f}"
        )
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
