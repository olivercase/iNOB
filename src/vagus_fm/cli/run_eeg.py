"""CLI: DUNEuro EEG forward solve."""
from __future__ import annotations

import argparse
import sys

from vagus_fm.cli._common import add_common_args, setup
from vagus_fm.forward.eeg import run_eeg_forward


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=run_eeg_forward.__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="eeg_forward")
    run_eeg_forward(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
