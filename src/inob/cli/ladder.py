"""CLI: run rungs of the Biot–Savart → Sarvas → FEM forward-model ladder.

Each rung predicts the OPM radial-coil field for one source:

  * biot    free-space Biot–Savart (no volume conductor) — analytic, seconds
  * sarvas  single homogeneous sphere (Sarvas 1987)       — analytic, seconds
  * fem     full multi-tissue FEM via DUNEuro             — needs the solve

The analytic rungs need only the geometry (FEM mesh + sensor array); the fem
rung additionally needs the forward leadfield. Run one rung, a subset, or all
three to compare them side by side.
"""

from __future__ import annotations

import argparse
import json
import sys

from inob.analysis.sarvas_compare import LADDER_RUNGS, run_ladder
from inob.cli import _ui
from inob.cli._common import add_common_args, setup


def _parse_rungs(arg: str) -> list[str]:
    if arg.lower() == "all":
        return list(LADDER_RUNGS)
    out: list[str] = []
    for r in arg.split(","):
        r = r.strip().lower()
        if not r:
            continue
        if r not in LADDER_RUNGS:
            raise ValueError(f"unknown rung {r!r}; choose from {', '.join(LADDER_RUNGS)} or 'all'")
        if r not in out:
            out.append(r)
    if not out:
        raise ValueError("no rungs selected")
    return out


def _render(result: dict, out) -> None:
    print(
        f"{_ui.heading('Forward-model ladder')}  "
        f"source #{result['source_index'] + 1} of {result['n_sources']} "
        f"at {tuple(result['source_pos_mm'])} mm  ·  "
        f"{result['n_radial_coils']} radial coils",
        file=out,
    )
    print(f"{_ui.paint('peak field, fT per nA·m', 'dim')}\n", file=out)

    order = [r for r in LADDER_RUNGS if r in result["rungs"]]
    for name in order:
        rung = result["rungs"][name]
        print(
            f"  {_ui.mark('ok')} {name:<7}"
            f"{rung['peak_fT_per_nAm']:>10.2f}   "
            f"{_ui.paint(rung['label'], 'dim')}",
            file=out,
        )

    ratios = result["ratios"]
    if ratios:
        print(f"\n{_ui.heading('Ladder ratios')}", file=out)
        pretty = {
            "sarvas_to_biot": "Sarvas / Biot–Savart  (volume-current effect)",
            "fem_to_biot": "FEM / Biot–Savart     (total body effect)",
            "fem_to_sarvas": "FEM / Sarvas          (multi-tissue vs sphere)",
        }
        for key, label in pretty.items():
            if key in ratios:
                print(f"  {label}: {_ui.paint(f'{ratios[key]:.2f}x', 'cyan')}", file=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob ladder",
        description="Compare rungs of the Biot–Savart → Sarvas → FEM ladder.",
        epilog=(
            "examples:\n"
            "  inob ladder                      all three rungs\n"
            "  inob ladder --rungs biot,sarvas  analytic only (no DUNEuro)\n"
            "  inob ladder --rungs fem          just the FEM rung\n"
            "  inob ladder --json               machine-readable output"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--rungs",
        default="all",
        help="Comma-separated rungs (biot,sarvas,fem) or 'all' (default).",
    )
    p.add_argument(
        "--Q-nAm", type=float, default=1.0, help="Source dipole moment in nA·m (default 1)."
    )
    p.add_argument(
        "--source-idx",
        type=int,
        default=-1,
        help="Source index along the polyline (default: middle).",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = p.parse_args(argv)

    cfg = setup(args, log_prefix="ladder")
    try:
        rungs = _parse_rungs(args.rungs)
    except ValueError as e:
        print(_ui.paint(str(e), "red"), file=sys.stderr)
        return 2

    result = run_ladder(cfg, rungs=rungs, Q_nAm=args.Q_nAm, source_idx=args.source_idx)

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _render(result, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
