"""CLI: empirical DUNEuro mm-mode EEG calibration via a sphere FEM."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vagus_fm.analysis.sphere_calibration import calibrate_eeg_factor
from vagus_fm.cli._common import add_common_args, setup


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--radius-mm", type=float, default=100.0,
                   help="Sphere radius (default 100 mm — neck-scale).")
    p.add_argument("--pitch-mm", type=float, default=3.0,
                   help="Voxelisation pitch for the sphere FEM.")
    p.add_argument("--sigma", type=float, default=0.43,
                   help="Homogeneous conductivity in S/m (default 0.43 = skin).")
    p.add_argument("--source-radius-mm", type=float, default=50.0,
                   help="Source eccentricity along Z (mm; must be < radius).")
    p.add_argument("--n-electrodes", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/calibration"))
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="calibrate")
    summary = calibrate_eeg_factor(
        radius_mm=args.radius_mm,
        pitch_mm=args.pitch_mm,
        sigma_S_per_m=args.sigma,
        source_radius_mm=args.source_radius_mm,
        n_electrodes=args.n_electrodes,
        duneuro_path=cfg.forward.duneuro_path,
        out_dir=args.out_dir,
    )
    sys.stdout.write(
        f"\n[calibration] empirical EEG factor (raw → µV/nA·m):\n"
        f"  peak    : {summary.factor_peak:.4e}\n"
        f"  median  : {summary.factor_median:.4e}\n"
        f"  geomean : {summary.factor_geomean:.4e}\n"
        f"  (n_elec={summary.n_electrodes}, R={summary.radius_mm:.0f} mm,\n"
        f"   src_r={summary.source_radius_mm:.0f} mm, σ={summary.sigma_S_per_m} S/m,\n"
        f"   n_tets={summary.n_tets})\n"
        f"\nApply to forward/eeg.py: L_uV_per_nAm = L * {summary.factor_median:.3e}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
