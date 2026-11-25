#!/usr/bin/env python3
"""Run the precision x movement 2x2 once and cache it for every downstream script.

    python scripts/run_cfa_engine.py      ->  outputs/cfa_forward/sim_2x2.pkl
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inob.cfa.array import CFAArray
from inob.cfa.design import DESIGN
from inob.cfa.engine import Forward, run_2x2


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    root = Path(__file__).resolve().parents[1]
    d = root / "outputs/cfa_forward"
    fwd = Forward.load(d, CFAArray.from_reference(root / "outputs/_cfa_ref.npz"))
    res = run_2x2(fwd, DESIGN)
    with open(d / "sim_2x2.pkl", "wb") as f:
        pickle.dump(res, f)
    print("wrote", d / "sim_2x2.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
