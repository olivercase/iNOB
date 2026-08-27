"""CLI: DUNEuro EEG forward solve."""
from __future__ import annotations

import argparse
import sys

from inob.cli._common import add_common_args, setup
from inob.forward.eeg import run_eeg_forward


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        # `inob-eeg` console script — see inob.cli.run_forward for why.
        from inob.duneuro_env import reexec_with_duneuro
        reexec_with_duneuro()
    p = argparse.ArgumentParser(
        prog="inob eeg",
        description=run_eeg_forward.__doc__,
        epilog="""\
examples:
  inob eeg                               surface-potential leadfield (µV/nA·m)
  inob eeg --source-target spine         the cord, with its own electrode patch

Same FEM and same sources as `inob forward`, so the two are comparable.
Needs the electrode array (`inob electrodes`) and DUNEuro.

Every command also takes --config, --set, --source-target, --log-level;
see `inob --help` for the full list.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="eeg_forward")
    run_eeg_forward(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
