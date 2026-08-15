"""Cluster array-job worker: forward solve for one slice of the sensor array.

Slices the configured sensor array into ``n_chunks`` contiguous pieces and
solves the forward problem for one chunk only. The transfer matrix is local
to the coil chunk, so chunks are fully independent.

Outputs (per chunk, into ``cfg.outputs.forward_chunks_dir``):

    L_chunk_<id>.npy        (n_chans_in_chunk, 3 * n_src) float64, T per A·m
    coil_idx_<id>.npy       int64 indices into the full coil array

The companion :mod:`inob.forward.reduce` stitches all chunks back into a
single leadfield NPZ identical in schema to the local
:mod:`inob.forward.solve` output (``source_pos`` included).

# TODO: source-first cluster mode for sensor optimisation
#
# The current reciprocal (sensor-first) approach commits to a fixed sensor
# geometry before the FEM runs — T must be fully recomputed for any change
# to the array. This makes sensor optimisation loops (e.g. greedy placement,
# gradient-based layout search) prohibitively expensive.
#
# An alternative is a source-first cluster mode: chunk by SOURCE instead of
# by sensor. For each source chunk, solve ∇·σ∇φ = f_source once and store
# φ on the mesh (or the Biot-Savart integral pre-projected onto a dense
# candidate sensor set). The full volumetric field is then available and any
# sensor position can be evaluated cheaply as a post-hoc inner product,
# without re-running the FEM.
#
# Cost trade-off: n_source solves (cheap if n_src is small, e.g. ~150 vagus
# positions) vs n_sensor reciprocal solves (8190 currently). For optimisation
# loops with a large candidate sensor set this flips the scaling decisively.
#
# The 2730-position / 8190-channel current array is far beyond physically
# viable hardware — the intent is to be able to vary sensor count and
# placement freely as a knob during optimisation, then commit to a realistic
# subset for the final DUNEuro run.
#
# Implementation sketch:
#   - New ``run_chunk_source(cfg, chunk_id, n_chunks)`` that chunks over
#     source positions rather than coil indices.
#   - Store per-source φ or pre-projected B at a dense candidate surface grid.
#   - New ``reduce_sources.py`` that assembles the full source × candidate
#     leadfield and exposes a ``sample_at(sensor_pos, sensor_ori)`` API.
#   - Wrap in a new CLI entry point ``inob-forward-source`` alongside the
#     existing ``inob-forward``.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from inob.config import Config
from inob.forward.duneuro_driver import (
    attach_coils,
    build_driver,
    build_orthogonal_dipoles,
    build_source_model_config,
    compute_meg_leadfield,
    import_duneuro,
)
from inob.io.hdf5 import load_fem, load_sensors, validate_fem, validate_sensors
from inob.sources.vagus import resolve_source_positions

logger = logging.getLogger(__name__)


def run_chunk(cfg: Config, *, chunk_id: int, n_chunks: int) -> tuple[Path, Path]:
    """Run forward solve for ``chunk_id`` of ``n_chunks``; return saved file paths."""
    if not (0 <= chunk_id < n_chunks):
        raise ValueError(f"chunk_id {chunk_id} out of range [0, {n_chunks})")

    chunks_dir = cfg.outputs.forward_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nodename = os.uname().nodename
    logger.info("[chunk %d/%d] starting on %s", chunk_id, n_chunks, nodename)

    fem = load_fem(cfg.outputs.fem_mat)
    validate_fem(fem)
    sensors = load_sensors(cfg.outputs.sensors_mat)
    validate_sensors(sensors)

    # resolve_source_positions, NOT sample_source_tissues: the former honours
    # cfg.forward.point_sources (explicit dipoles, e.g. clicked in the GUI) and
    # falls back to tissue sampling only when none are set. Calling the sampler
    # directly here silently ignored explicit sources on the chunked path —
    # which is the default local multi-core path — so the GUI's clicked sources
    # never reached the solver while the serial path honoured them.
    src_pos_mm = resolve_source_positions(cfg, fem)
    n_src = len(src_pos_mm)
    logger.info("FEM=%dt sensors=%d sources=%d", len(fem.tets), len(sensors.coilpos), n_src)

    n_chan_total = len(sensors.coilpos)
    chunk_idx = np.array_split(np.arange(n_chan_total), n_chunks)[chunk_id]
    coilpos = sensors.coilpos[chunk_idx]
    coilori = sensors.coilori[chunk_idx]
    logger.info("chunk %d channels (global %d..%d)",
                len(chunk_idx), int(chunk_idx[0]), int(chunk_idx[-1]))

    dp = import_duneuro(cfg)
    # limit_threads: this worker is one of `n_chunks` processes, so it must not
    # also claim every core for TBB (see SolverCfg.threads_per_process).
    driver, driver_cfg, _cond = build_driver(cfg, fem, limit_threads=True)
    attach_coils(driver, dp, coilpos, coilori)

    logger.info("computing transfer matrix…")
    t_T = time.time()
    T_raw, _ = driver.computeMEGTransferMatrix(driver_cfg)
    T = np.array(T_raw)
    logger.info("  T %s (%.0fs)", T.shape, time.time() - t_T)

    dipoles_du = build_orthogonal_dipoles(dp, src_pos_mm)
    driver_cfg["source_model"] = build_source_model_config(cfg)
    logger.info("source model: %s", cfg.forward.source_model.type)

    logger.info("applying transfer to dipoles…")
    t_a = time.time()
    # SI leadfield (T per A·m): secondary (transfer) + primary (Biot–Savart),
    # unit-corrected from DUNEuro mm-mode — same as the single-machine path.
    L = compute_meg_leadfield(driver, T, dipoles_du, driver_cfg)
    logger.info("  L %s (%.0fs)", L.shape, time.time() - t_a)

    out_L = chunks_dir / f"L_chunk_{chunk_id:03d}.npy"
    out_idx = chunks_dir / f"coil_idx_{chunk_id:03d}.npy"
    np.save(out_L, L.astype(np.float64))
    np.save(out_idx, chunk_idx.astype(np.int64))
    logger.info("[chunk %d] saved %s (%.1f MB) total %.0fs",
                chunk_id, out_L.name, out_L.stat().st_size / 1e6, time.time() - t0)
    return out_L, out_idx


def main(argv: list[str] | None = None) -> int:
    """``python -m inob.forward.chunk`` entry point (used by submit_array.sh)."""
    import argparse

    from inob.cli._common import add_common_args, setup

    p = argparse.ArgumentParser(description=run_chunk.__doc__)
    add_common_args(p)
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--n-chunks", type=int, default=None,
                   help="Defaults to cluster.n_chunks from the config.")
    args = p.parse_args(argv)
    cfg = setup(args, log_prefix=f"chunk_{args.chunk_id:03d}")
    n_chunks = args.n_chunks if args.n_chunks is not None else cfg.cluster.n_chunks
    run_chunk(cfg, chunk_id=args.chunk_id, n_chunks=n_chunks)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
