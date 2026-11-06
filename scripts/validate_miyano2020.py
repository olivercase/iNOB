#!/usr/bin/env python3
"""Compare the spine leadfield with SQUID magnetospinography (Miyano 2020; Hashimoto 2022).

Miyano et al. (Clin Neurophysiol 131:2460-2468, doi 10.1016/j.clinph.2020.07.009)
report, after ulnar-nerve stimulation, the largest Y-axis (cranio-caudal) peak of
64 +/- 12.0 fT (elbow) and 30 +/- 7.8 fT (wrist), and reconstructed foraminal
inflow currents at C7/T1 of 3.81 +/- 0.96 and 1.70 +/- 0.45 nA.m. Their
beamformer assumes no volume conductor, so those currents are free-space
equivalents. This script places a left-right moment in the T1-C7 region and reads
the largest field on the cranio-caudal (body-z) channels, in free space
(Biot-Savart) and in the full FEM, from the saved leadfield.

Hashimoto et al. (Clin Neurophysiol 133:39-47, doi 10.1016/j.clinph.2021.09.023)
report the largest baseline-to-peak field on the left-right (X) channels over
the thoracic cord after unilateral and synchronised bilateral sciatic-nerve
stimulation (their Table 1). The second part places a cranio-caudal moment at
the T10, T6 and T2 levels, reads the largest field on the left-right channels,
and reports the moment that would reproduce each measured field.

    python scripts/validate_miyano2020.py [--config configs/spine.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

#: (label, reconstructed current nA.m, measured peak fT, SD fT) from Miyano et al. 2020.
MEASURED = (("elbow", 3.81, 64.0, 12.0), ("wrist", 1.70, 30.0, 7.8))
#: (level, bilateral fT, unilateral fT): largest baseline-to-peak left-right-axis field after
#: sciatic-nerve stimulation, mean of five participants, Hashimoto et al. 2022 Table 1.
HASHIMOTO = (("T10", 35.9, 18.5), ("T6", 21.9, 9.7), ("T2", 13.0, 6.4))
#: Source-height window (mm) spanning the T1 and C7 levels, where the cord source is
#: nearest the C7/T1 foramen (levels from inob.anatomy).
Z_WINDOW_MM = (1349.0, 1384.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/spine.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    from inob.cli._common import apply_source_target
    from inob.config import load_config
    from inob.io.npz import load_leadfield

    cfg = apply_source_target(load_config(args.config), "spine")
    meg = load_leadfield(cfg.outputs.forward_npz)
    n_ch = meg.L_fT_per_nAm.shape[0]
    lead = meg.L_fT_per_nAm.reshape(n_ch, -1, 3)
    z = meg.source_pos[:, 2]
    idx = np.where((z >= Z_WINDOW_MM[0]) & (z < Z_WINDOW_MM[1]))[0]
    axial = np.where(np.abs(meg.coil_orient[:, 2]) > 0.9)[0]  # channels along body z

    def biot_savart(src: np.ndarray, moment: np.ndarray) -> np.ndarray:
        d = (meg.coil_pos[axial] - src) / 1000.0  # m
        r = np.linalg.norm(d, axis=1)
        b = 1e-7 * np.cross(moment[None, :], d) / r[:, None] ** 3 * 1e-9  # T per nA.m
        return (b * meg.coil_orient[axial]).sum(1) * 1e15  # fT per nA.m

    x = np.array([1.0, 0.0, 0.0])  # left-right moment, into the foramen
    free = max(np.abs(biot_savart(meg.source_pos[i], x)).max() for i in idx)
    fem = float(np.abs(lead[axial][:, idx, 0]).max())
    print(
        f"{len(idx)} sources, z = {Z_WINDOW_MM[0]:.0f}-{Z_WINDOW_MM[1]:.0f} mm; {len(axial)} cranio-caudal channels"
    )
    print(
        f"left-right moment: free space {free:.1f}, FEM {fem:.1f} fT per nA.m (ratio {fem / free:.2f})"
    )
    for label, q, meas, sd in MEASURED:
        print(
            f"{label:>5}: q = {q:.2f} nA.m -> free space {q * free:5.0f} fT, FEM {q * fem:5.0f} fT; "
            f"measured {meas:.0f} +/- {sd:.1f} fT; FEM-implied current {meas / fem:.2f} nA.m"
        )

    # --- Hashimoto 2022: thoracic cord, cranio-caudal moment, left-right channels
    lateral = np.where(np.abs(meg.coil_orient[:, 0]) > 0.9)[0]
    levels = _levels(cfg, z)
    names = sorted(levels, key=levels.get)

    def window(name: str) -> np.ndarray:
        i = names.index(name)
        hi = levels[names[i + 1]] if i + 1 < len(names) else z.max() + 1
        return np.where((z >= levels[name]) & (z < hi))[0]

    def biot_savart_ch(src: np.ndarray, moment: np.ndarray, chans: np.ndarray) -> np.ndarray:
        d = (meg.coil_pos[chans] - src) / 1000.0
        r = np.linalg.norm(d, axis=1)
        b = 1e-7 * np.cross(moment[None, :], d) / r[:, None] ** 3 * 1e-9
        return (b * meg.coil_orient[chans]).sum(1) * 1e15

    zdir = np.array([0.0, 0.0, 1.0])
    print("\nThoracic cord, cranio-caudal moment, left-right channels (Hashimoto 2022)")
    for level, bil, uni in HASHIMOTO:
        idx = window(level)
        free = max(np.abs(biot_savart_ch(meg.source_pos[i], zdir, lateral)).max() for i in idx)
        fem = float(np.abs(lead[lateral][:, idx, 2]).max())
        print(
            f"{level:>3}: free space {free:5.1f}, FEM {fem:5.1f} fT per nA.m; measured "
            f"bilateral {bil:5.1f} / unilateral {uni:4.1f} fT -> FEM moment {bil / fem:4.2f} / {uni / fem:4.2f} nA.m; "
            f"stationary 5.11 nA.m would give {5.11 * fem:4.0f} fT ({5.11 * fem / bil:3.1f}x bilateral)"
        )
    return 0


def _levels(cfg, z):
    from inob.anatomy import vertebral_level

    out: dict[str, float] = {}
    for zz in np.sort(z):
        lv = vertebral_level(cfg.data.bone_dir, float(zz))
        if lv is not None:
            out.setdefault(lv.short.split("/")[0], float(zz))
    return out


if __name__ == "__main__":
    sys.exit(main())
