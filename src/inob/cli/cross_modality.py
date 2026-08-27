"""CLI: render the cross-modality MEG↔EEG coupling figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.viz.cross_modality_plot import render_cross_modality


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob cross",
        description=__doc__,
        epilog="""\
examples:
  inob cross                             MEG/EEG coupling for the same source
  inob cross --source-idx 40             at one point on the polyline
  inob cross --noise-uV 0.5               bootstrap with 0.5 µV of EEG noise

Not an inverse solution: the source position is assumed known.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument("--source-idx", type=int, default=-1,
                   help="Source index along the vagus polyline (default: middle).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: cfg.outputs.base/cross_modality.png).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--noise-uV", type=float, default=None,
        help="Override the EEG noise floor (RMS µV) used to corrupt the EEG "
             "observation before the lstsq inversion. Default = derive from "
             "cfg.noise (HD-EMG amplifier + Johnson noise integrated over "
             "the cfg.noise recording band).",
    )
    p.add_argument("--noise-seed", type=int, default=0,
                   help="RNG seed for the EEG-noise sample.")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="cross_modality")
    render_cross_modality(
        cfg, source_idx=args.source_idx, out_path=args.out, dpi=args.dpi,
        noise_uV=args.noise_uV, noise_seed=args.noise_seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
