#!/usr/bin/env python3
"""Does the model get the brain right? MG against EG for a cortical source.

A reviewer's first question on seeing electrography lose to magnetography
by an order of magnitude at the neck is whether the package reproduces what
is known about the one target where both have been measured for decades.
For the brain the empirical position is that evoked-response SNR is broadly
comparable between MEG and EEG (Goldenholz et al. 2009, Hum Brain Mapp
30:1077), with a 10 nA·m cortical dipole giving on the order of a picotesla
at an on-scalp OPM and a few microvolts on the scalp.

This script reads the two brain leadfields — solved on the same whole-body
FEM, the same conductivities and the same noise model as the vagus, cord and
muscle — and prints, by source depth, the signals the paper's own metrics
report (best channel and moment for MG, largest contact difference for EG)
and the SNR ratio under the paper's noise floors. It separates the two
questions a disagreement could come from: are the leadfield magnitudes
wrong, or are the noise floors?

    inob forward --source-target brain
    inob electrodes --source-target brain --set electrodes.shape=whole_body
    inob eeg --source-target brain --set electrodes.shape=whole_body
    python3 scripts/brain_sanity.py

The noise floors are the ones every other target is judged against (taken
from the config, vagal CAP band), deliberately: the point is to ask whether
the comparison the paper makes elsewhere would come out right for the brain.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent

# Source depth bins below the skin, mm. Cortex sits roughly 12-30 mm under
# the scalp; the deeper bins are sulcal and subcortical.
DEPTH_EDGES_MM = (0.0, 20.0, 30.0, 40.0, 60.0, np.inf)


@dataclass(frozen=True)
class DepthRow:
    lo_mm: float
    hi_mm: float
    n: int
    mg_fT_per_nAm: float  # median best-channel, best-moment |B|
    eg_uV_per_nAm: float  # median largest contact difference
    snr_ratio_eg_over_mg: float  # median over sources in the bin


def compare(
    L_mg: np.ndarray,
    pos_mg: np.ndarray,
    L_eg: np.ndarray,
    pos_eg: np.ndarray,
    coil_pos: np.ndarray,
    *,
    standoff_mm: float,
    sigma_fT: float,
    sigma_uV: float,
    match_tol_mm: float = 1.0,
) -> list[DepthRow]:
    """Bin matched sources by depth and report both modalities' signals.

    The two solves need not share a source grid, so each MG source is paired
    with the nearest EG source and dropped if none lies within
    ``match_tol_mm``. Depth is the distance to the nearest magnetometer less
    its stand-off: the array is dense and conformal, so this is the distance
    to the skin to within a few millimetres.
    """
    from inob.analysis.snr import per_source_best_bipolar, per_source_peak

    mg = per_source_peak(L_mg)
    eg_all = per_source_best_bipolar(L_eg)
    dist, idx = cKDTree(pos_eg).query(pos_mg)
    keep = dist <= match_tol_mm
    mg = mg[keep]
    eg = eg_all[idx[keep]]
    depth = cKDTree(coil_pos).query(pos_mg[keep])[0] - standoff_mm
    ratio = (eg / sigma_uV) / (mg / sigma_fT)

    rows = []
    for lo, hi in itertools.pairwise(DEPTH_EDGES_MM):
        k = (depth >= lo) & (depth < hi)
        if not k.any():
            continue
        rows.append(
            DepthRow(
                lo_mm=lo,
                hi_mm=hi,
                n=int(k.sum()),
                mg_fT_per_nAm=float(np.median(mg[k])),
                eg_uV_per_nAm=float(np.median(eg[k])),
                snr_ratio_eg_over_mg=float(np.median(ratio[k])),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--mg", default=str(REPO_ROOT / "outputs/forward/duneuro_leadfield_brain.npz"))
    ap.add_argument(
        "--eg", default=str(REPO_ROOT / "outputs/forward/duneuro_eeg_leadfield_wholebody_brain.npz")
    )
    ap.add_argument(
        "--moment-nAm",
        type=float,
        default=10.0,
        help="Dipole moment for the absolute column (default 10, a typical "
        "cortical evoked equivalent dipole).",
    )
    args = ap.parse_args(argv)

    from inob.analysis.snr import compute_noise_floors
    from inob.config import load_config

    for p in (args.mg, args.eg):
        if not Path(p).exists():
            print(f"missing {p}; solve the brain leadfields first (see --help)", file=sys.stderr)
            return 1
    cfg = load_config(args.config)
    floors = compute_noise_floors(cfg)
    m = np.load(args.mg, allow_pickle=True)
    e = np.load(args.eg, allow_pickle=True)
    rows = compare(
        m["L_fT_per_nAm"],
        m["source_pos"],
        # The EEG file keeps its µV leadfield in the L_fT_per_nAm slot; see
        # inob.forward.eeg.
        e["L_fT_per_nAm"],
        e["source_pos"],
        m["coil_pos"],
        standoff_mm=float(cfg.sensors.depth_mm),
        sigma_fT=floors.meg_per_channel_fT,
        sigma_uV=floors.eeg_per_channel_uV,
    )
    q = args.moment_nAm
    print(
        f"Noise floors (paper's): MG {floors.meg_per_channel_fT:.0f} fT, "
        f"EG {floors.eeg_per_channel_uV:.2f} µV per channel"
    )
    print(
        f"{'depth mm':>10} {'n':>5} {'MG fT/nAm':>10} {'EG µV/nAm':>10} "
        f"{'MG @' + format(q, 'g'):>9} {'EG @' + format(q, 'g'):>9} {'SNR EG/MG':>10}"
    )
    for r in rows:
        hi = "∞" if np.isinf(r.hi_mm) else f"{r.hi_mm:g}"
        print(
            f"{f'{r.lo_mm:g}-{hi}':>10} {r.n:>5} {r.mg_fT_per_nAm:>10.1f} "
            f"{r.eg_uV_per_nAm:>10.3f} {r.mg_fT_per_nAm * q / 1e3:>7.2f}pT "
            f"{r.eg_uV_per_nAm * q:>7.2f}µV {r.snr_ratio_eg_over_mg:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
