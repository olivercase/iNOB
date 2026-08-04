"""Stitch per-chunk leadfields into the canonical NPZ.

Produces an output identical in schema to the local
:mod:`inob.forward.solve` (``source_pos`` included).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inob.config import Config
from inob.forward.duneuro_driver import build_conductivity_vector
from inob.io.hdf5 import load_fem, load_sensors, validate_fem, validate_sensors
from inob.io.npz import Leadfield, save_leadfield, validate_leadfield
from inob.sources.vagus import resolve_source_positions

logger = logging.getLogger(__name__)


def reduce_chunks(cfg: Config) -> Path:
    """Stitch all ``L_chunk_*.npy`` in ``cfg.outputs.forward_chunks_dir`` into one NPZ."""
    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks = sorted(chunks_dir.glob("L_chunk_*.npy"))
    if not chunks:
        raise FileNotFoundError(f"no L_chunk_*.npy in {chunks_dir}")
    logger.info("found %d chunks in %s", len(chunks), chunks_dir)

    fem = load_fem(cfg.outputs.fem_mat)
    validate_fem(fem)
    sensors = load_sensors(cfg.outputs.sensors_mat)
    validate_sensors(sensors)

    n_chan = len(sensors.coilpos)
    n_cols = int(np.load(chunks[0], mmap_mode="r").shape[1])
    L = np.full((n_chan, n_cols), np.nan, dtype=np.float64)

    for L_path in chunks:
        cid = L_path.stem.split("_")[-1]
        idx_path = chunks_dir / f"coil_idx_{cid}.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"missing index for {L_path.name}: {idx_path}")
        Lc = np.load(L_path)
        idx = np.load(idx_path)
        if Lc.shape != (len(idx), n_cols):
            raise ValueError(
                f"chunk {cid}: shape {Lc.shape} != expected {(len(idx), n_cols)}"
            )
        L[idx] = Lc

    if np.isnan(L).any():
        missing = np.where(np.isnan(L).any(axis=1))[0]
        raise RuntimeError(
            f"missing rows: {len(missing)} channels (e.g. {missing[:5].tolist()})"
        )

    # Must resolve sources exactly as the chunk workers did, or the stitched
    # leadfield's columns get labelled with the wrong positions. Both sides use
    # resolve_source_positions so explicit point sources are honoured here too.
    src_pos = resolve_source_positions(cfg, fem)
    if 3 * len(src_pos) != n_cols:
        raise RuntimeError(
            f"source count mismatch: 3 * {len(src_pos)} != {n_cols} (chunks contain "
            f"a different source set than the FEM/cfg implies). Re-run chunks "
            f"with the same config.")

    cond = build_conductivity_vector(cfg, fem)

    lf = Leadfield(
        L=L,
        L_fT_per_nAm=L * 1e6,
        source_pos=src_pos,
        coil_pos=sensors.coilpos,
        coil_orient=sensors.coilori,
        channel_names=tuple(sensors.labels),
        conductivities=cond,
        tissue_labels=tuple(fem.tissue_labels),
        seed=cfg.reproducibility.seed,
    )
    validate_leadfield(
        lf,
        require_finite=cfg.forward.validate.require_finite,
        max_abs_fT_per_nAm=cfg.forward.validate.max_abs_fT_per_nAm,
    )
    out = cfg.outputs.forward_npz
    save_leadfield(out, lf)
    logger.info("[reduce] saved %s (%.1f MB); L shape %s",
                out, out.stat().st_size / 1e6, L.shape)
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m inob.forward.reduce`` entry point."""
    import argparse

    from inob.cli._common import add_common_args, setup

    p = argparse.ArgumentParser(description=reduce_chunks.__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix="reduce")
    reduce_chunks(cfg)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
