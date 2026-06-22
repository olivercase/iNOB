"""CLI: whole-body EEG sensitivity comparison vs the cervical paddle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vagus_fm.cli._common import add_common_args, setup
from vagus_fm.viz.location_optimisation import render_location_optimisation


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--paddle-mat", type=Path,
        default=Path("outputs/sensors/electrode_array.mat"),
        help="32-channel cervical paddle electrode positions.",
    )
    p.add_argument(
        "--paddle-npz", type=Path,
        default=Path("outputs/forward/duneuro_eeg_leadfield_vagus.npz"),
        help="Paddle EEG leadfield NPZ.",
    )
    p.add_argument(
        "--wholebody-mat", type=Path,
        default=Path("outputs/sensors/electrode_array_wholebody.mat"),
        help="Whole-body electrode positions.",
    )
    p.add_argument(
        "--wholebody-npz", type=Path,
        default=Path("outputs/forward/duneuro_eeg_leadfield_wholebody.npz"),
        help="Whole-body EEG leadfield NPZ.",
    )
    p.add_argument("--source-idx", type=int, default=-1)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="location")
    render_location_optimisation(
        cfg,
        paddle_mat=args.paddle_mat, paddle_npz=args.paddle_npz,
        wholebody_mat=args.wholebody_mat, wholebody_npz=args.wholebody_npz,
        source_idx=args.source_idx, out_path=args.out, dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
