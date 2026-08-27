"""CLI: run the local DUNEuro forward solve."""
from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.forward.local import run_forward_local


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        # Reached as the `inob-forward` console script, so sys.argv is ours to
        # re-exec. Dispatch through `inob forward` handles its own switch.
        from inob.duneuro_env import reexec_with_duneuro
        reexec_with_duneuro()
    p = argparse.ArgumentParser(
        prog="inob forward",
        description=run_forward_local.__doc__,
        epilog="""\
examples:
  inob forward                           MEG leadfield, every core
  inob forward --workers 1               serial, for debugging
  inob forward --source-target spine     solve the cord, write a tagged NPZ
  inob forward --source-model venant     St. Venant instead of partial integration

Needs DUNEuro. Writes fT/nA·m into outputs/forward/. Slow: this is the solve.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="forward")
    # The multi-core path, the same one `inob run` uses for this stage: the coil
    # array is split into `forward.local_workers` chunks (0 = every core), each
    # solved in its own process, then stitched. It falls back to the serial
    # solve when that resolves to one chunk. Calling the serial solve directly
    # here — as this did — meant `inob forward --workers 12` quietly ran on one
    # core, and the transfer matrix is per-coil, so that is the whole cost.
    run_forward_local(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
