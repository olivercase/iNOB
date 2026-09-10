"""CLI: render the propagating-CAP vs stationary-dipole comparison figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.viz.cap_compare import render_cap_compare

# The user-facing summary in `--help`. Kept separate from the module
# docstring, which is written for whoever maintains the code.
_DESCRIPTION = """\
Compare a propagating action potential against the stationary
dipole usually assumed in its place.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob cap-compare",
        description=_DESCRIPTION,
        epilog="""\
examples:
  inob cap-compare                       propagating volley vs lumped dipole
  inob cap-compare --ap-width-ms 0.3     a sharper action potential
  inob cap-compare --segment-mm 100      a longer stretch of nerve

Measures whether the stationary approximation holds — it does for the
cervical vagus, and does not for the ascending cord.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument("--ap-width-ms", type=float, default=None,
                   help="Width parameter (σ) of the biphasic action-potential "
                        "shape. Default: the target's physiology profile.")
    p.add_argument("--n-fibres", type=int, default=None,
                   help="Fibres in the synchronous event. Default: profile.")
    p.add_argument("--ap-amplitude-mV", type=float, default=None,
                   help="Intracellular AP amplitude (mV). Default: profile.")
    p.add_argument("--segment-mm", type=float, default=None,
                   help="Arc length over which the event propagates. Default: "
                        "the profile's span (whole polyline for the spine).")
    p.add_argument("--fs-hz", type=float, default=30_000.0,
                   help="Sampling frequency (Hz). Default 30 kHz captures sub-AP detail.")
    p.add_argument("--duration-ms", type=float, default=None,
                   help="Trace duration (ms). Default: sized to the target's "
                        "transit time (30 ms vagus, ~50 ms spine).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: cfg.outputs.base/cap_compare_<target>.png).")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="cap_compare")
    render_cap_compare(
        cfg,
        n_fibres=args.n_fibres,
        ap_amplitude_mV=args.ap_amplitude_mV,
        ap_width_ms=args.ap_width_ms,
        segment_mm=args.segment_mm,
        fs_hz=args.fs_hz,
        duration_ms=args.duration_ms,
        out_path=args.out,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
