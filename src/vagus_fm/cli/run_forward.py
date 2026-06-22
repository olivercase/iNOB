"""CLI: run the local DUNEuro forward solve."""
from __future__ import annotations

import argparse
import sys

from vagus_fm.cli._common import add_common_args, setup
from vagus_fm.forward.solve import run_forward


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=run_forward.__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="forward")
    run_forward(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
