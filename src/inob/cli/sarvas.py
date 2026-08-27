"""CLI: Sarvas analytic vs FEM benchmark for the cervical vagus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.analysis.sarvas_compare import (
    compare_sarvas_vs_fem,
    hamalainen_dipole_moment_nAm,
    save_comparison_summary,
)
from inob.cli._common import add_common_args, setup
from inob.config import source_region_label, target_output
from inob.viz.sarvas_plot import render_sarvas_vs_fem


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob sarvas",
        description=__doc__,
        epilog="""\
examples:
  inob sarvas                            FEM against the analytic sphere
  inob sarvas --source-idx 40            benchmark one source
  inob sarvas --show-hamalainen          mark the Hamalainen dipole moment

A validation, not a result: the sphere is the reference the FEM is judged by.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--Q-nAm", type=float, default=1.0,
        help="Source dipole moment in nA·m. Default 1 nA·m so the figure "
             "y-axis reads as a leadfield in fT — matches every other plot "
             "in this repo. Pass --Q-nAm 70 to scale the y-axis directly to "
             "the predicted real-CAP amplitude (full A+C fibre summation, "
             "Hämäläinen 1993).",
    )
    p.add_argument("--source-idx", type=int, default=-1)
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: cfg.outputs.base/sarvas/sarvas_vs_fem.png).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--show-hamalainen", action="store_true",
        help="Print the Hämäläinen Q for representative A and C fibres.",
    )
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="sarvas")

    if args.show_hamalainen:
        print("Hämäläinen Q = π · d² · σ_in · ΔV / 4")
        for d_um, dV_mV, label in (
            (10.0, 70.0, "A-fibre  d=10 µm  ΔV=70 mV"),
            (1.0,  80.0, "C-fibre  d=1 µm   ΔV=80 mV"),
        ):
            Q = hamalainen_dipole_moment_nAm(
                fibre_diameter_um=d_um,
                sigma_intracellular_S_per_m=1.0,
                action_potential_mV=dV_mV,
            )
            print(f"  {label}  →  Q = {Q:.4f} nA·m")

    result = compare_sarvas_vs_fem(cfg, Q_nAm=args.Q_nAm)
    out = args.out or (cfg.outputs.base / "sarvas" / target_output(cfg, "sarvas_vs_fem.png").name)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_sarvas_vs_fem(result, source_idx=args.source_idx, out_path=out,
                          dpi=args.dpi, region=source_region_label(cfg))
    summary_path = out.with_suffix(".json")
    save_comparison_summary(result, summary_path)
    print(f"figure: {out}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
