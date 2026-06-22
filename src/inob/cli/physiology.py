"""CLI: moving-dipole physiology simulation (baroreceptor + slow breathing)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.physiology.scenarios import (
    baroreceptor_scenario,
    respiratory_scenario,
)
from inob.viz.physiology_plot import render_physiology


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--hr-bpm", type=float, default=70.0,
                   help="Heart rate for the baroreceptor scenario.")
    p.add_argument("--breath-bpm", type=float, default=6.0,
                   help="Respiratory rate for the deep-breathing scenario.")
    p.add_argument("--baro-duration-s", type=float, default=6.0)
    p.add_argument("--resp-duration-s", type=float, default=12.0)
    p.add_argument("--n-fibres-baro", type=int, default=200,
                   help="Fibres per cardiac burst (baroreceptor).")
    p.add_argument("--n-fibres-rar", type=int, default=600,
                   help="RAR (rapidly-adapting receptor) phasic burst fibres at insp-onset.")
    p.add_argument("--n-fibres-sar", type=int, default=80,
                   help="SAR (slowly-adapting receptor) tonic-train fibres.")
    p.add_argument("--fs-hz", type=float, default=30_000.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="physiology")

    scenarios = (
        baroreceptor_scenario(
            duration_s=args.baro_duration_s, hr_bpm=args.hr_bpm,
            n_fibres_per_burst=args.n_fibres_baro,
        ),
        respiratory_scenario(
            duration_s=args.resp_duration_s, breath_bpm=args.breath_bpm,
            n_phasic_fibres_RAR=args.n_fibres_rar,
            n_tonic_fibres_SAR=args.n_fibres_sar,
        ),
    )
    render_physiology(
        cfg, scenarios=scenarios, fs_hz=args.fs_hz,
        out_path=args.out, dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
