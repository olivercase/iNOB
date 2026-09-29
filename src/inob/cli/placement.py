"""CLI: model-informed sensor placement against naive nearest-to-source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.analysis.placement import format_sweep_table
from inob.cli._common import add_common_args, setup
from inob.viz.placement_comparison import render_placement_comparison

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Test whether the forward model picks better sensor positions than
simply putting the sensors as close to the source as you can.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob placement",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob placement                       20 sensors at the best-coupled source
  inob placement --n-sensors 64        a larger array
  inob placement --source-idx 40       judge it at one source
  inob placement --optimise --n-sensors 32 --write-array opt.npz
                                       design a 32-sensor array for the
                                       whole target, then use it with
                                       inob sensors --from opt.npz

Both arms come from the same solved leadfield, so the only difference
is which positions were chosen. On the shipped anatomy they come out
within a few percent: the nearest positions are also the best-coupled
ones. The model earns its keep on the amplitude and trial count, not
on picking the positions.

--optimise answers the question for a whole structure instead of one
source: it adds positions one at a time, each the one that adds most
information about every source in the target (a nerve, the cord) given
the sensors already chosen, and compares the result with simply taking
the positions nearest the structure.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--forward-npz",
        type=Path,
        default=None,
        help="MEG leadfield NPZ (default: this target's outputs.forward_npz).",
    )
    p.add_argument(
        "--source-idx",
        type=int,
        default=-1,
        help="Source to judge placement at (default: the best-coupled one).",
    )
    p.add_argument(
        "--n-sensors",
        type=int,
        default=20,
        help="How many sensor positions each arm may place (default: 20).",
    )
    p.add_argument(
        "--min-spacing-mm",
        type=float,
        default=30.0,
        help="Closest two sensors may sit, in either arm (default: 30).",
    )
    p.add_argument(
        "--optimise",
        action="store_true",
        help="Design an array for every source in the target by total "
        "information, instead of comparing arms at one source.",
    )
    p.add_argument(
        "--write-array",
        type=Path,
        default=None,
        help="With --optimise, save the chosen positions here (.npz), ready "
        "for inob sensors --from.",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="placement")
    if args.optimise:
        return _optimise(cfg, args)
    _, sweep = render_placement_comparison(
        cfg,
        forward_npz=args.forward_npz,
        source_idx=args.source_idx,
        n_sensors=args.n_sensors,
        min_spacing_mm=args.min_spacing_mm,
        out_path=args.out,
        dpi=args.dpi,
    )
    print(format_sweep_table(sweep))
    return 0


def _optimise(cfg, args) -> int:
    import numpy as np

    from inob.analysis.placement import nearest_array_info, optimise_array
    from inob.analysis.snr import compute_noise_floors
    from inob.io.npz import load_leadfield
    from inob.physiology.profiles import profile_for

    lf = load_leadfield(args.forward_npz or cfg.outputs.forward_npz)
    noise = compute_noise_floors(cfg).meg_per_channel_fT
    q = profile_for(cfg).event_moment_nAm
    res = optimise_array(
        lf.L_fT_per_nAm,
        lf.coil_pos,
        n_sensors=args.n_sensors,
        noise_floor=noise,
        moment_nAm=q,
        min_spacing_mm=args.min_spacing_mm,
    )
    print(
        f"{res.n_sources} sources, {q:.2f} nA·m each, noise {noise:.0f} fT, "
        f"sensors at least {args.min_spacing_mm:.0f} mm apart"
    )
    print(f"whole candidate array ({len(res.positions)} positions): {res.full_info_bits:.1f} bits")
    print(f"{'sensors':>8}  {'optimised':>10}  {'nearest':>8}  {'gain':>6}")
    for n in sorted(
        {k for k in (4, 8, 16, 32, 64, 128) if k < len(res.info_bits)} | {len(res.info_bits)}
    ):
        near = nearest_array_info(
            lf.L_fT_per_nAm,
            lf.coil_pos,
            np.asarray(lf.source_pos, dtype=float),
            n_sensors=n,
            noise_floor=noise,
            moment_nAm=q,
            min_spacing_mm=args.min_spacing_mm,
        )
        opt = res.info_bits[n - 1]
        print(f"{n:>8}  {opt:>10.1f}  {near:>8.1f}  {opt / max(near, 1e-12):>5.2f}x")
    print("Information is in bits about the target's sources; 'nearest' takes")
    print("the positions closest to the structure under the same spacing.")
    if args.write_array:
        ch = res.channel_idx
        np.savez(
            args.write_array,
            coil_pos=np.asarray(lf.coil_pos)[ch],
            coil_orient=np.asarray(lf.coil_orient)[ch],
        )
        print(f"[saved] {args.write_array} ({len(ch)} channels); use: inob sensors --from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
