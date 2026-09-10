"""Local multi-core forward solve.

The single-process DUNEuro MEG solve (:mod:`inob.forward.solve`) computes the
transfer matrix one coil at a time on a single core. The coils are independent,
so this module splits the array into ``forward.local_workers`` chunks (default:
all CPU cores), solves each in its own process via the same worker the cluster
array job uses (:func:`inob.forward.chunk.run_chunk`), and stitches the result
with :func:`inob.forward.reduce.reduce_chunks`. The output NPZ is byte-for-byte
the same schema as the serial path.

This is the default forward path used by ``inob run``; it falls back to the
serial solve when only one worker is requested or the array is too small to
split.
"""

from __future__ import annotations

import logging
import os
import shutil
from multiprocessing import get_context
from pathlib import Path

from inob.config import Config
from inob.io.hdf5 import load_sensors

logger = logging.getLogger(__name__)


def effective_workers(cfg: Config) -> int:
    """Resolve ``forward.local_workers`` (0 → all cores) to a positive count."""
    n = cfg.forward.local_workers
    if n and int(n) > 0:
        return int(n)
    return os.cpu_count() or 1


def run_forward_local(cfg: Config) -> Path:
    """Forward solve across local cores; returns the leadfield NPZ path.

    Splits the coil array into ``min(workers, n_channels)`` process-parallel
    chunks and reduces them. Falls back to the serial single-process solve when
    that resolves to one chunk (e.g. ``local_workers=1`` or a tiny array).
    """
    # Imported lazily so importing this module (and the CLI) never triggers the
    # heavy forward/DUNEuro import graph until a solve is actually requested.
    from inob.forward.chunk import run_chunk
    from inob.forward.reduce import reduce_chunks
    from inob.forward.solve import run_forward

    workers = effective_workers(cfg)
    sensors = load_sensors(cfg.outputs.sensors_mat)
    n_chan = len(sensors.coilpos)
    n_chunks = max(1, min(workers, n_chan))

    if n_chunks <= 1:
        logger.info("forward: serial solve (workers=%d, channels=%d)", workers, n_chan)
        return run_forward(cfg)

    logger.info(
        "forward: local parallel solve — %d chunks across %d cores, %d channels",
        n_chunks,
        workers,
        n_chan,
    )
    chunks_dir = cfg.outputs.forward_chunks_dir
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Pin each worker to a single BLAS/OpenMP thread so N processes map to N
    # cores without oversubscribing. Children inherit this environment.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # 'spawn' gives each worker a clean interpreter — required for the compiled
    # duneuropy extension, which is not fork-safe once a driver has been built.
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_chunks, mp_context=ctx) as pool:
        futures = {
            pool.submit(run_chunk, cfg, chunk_id=i, n_chunks=n_chunks): i for i in range(n_chunks)
        }
        done = 0
        for fut in as_completed(futures):
            fut.result()  # re-raise the first worker error, cancelling the rest
            done += 1
            logger.info("forward: chunk %d/%d complete", done, n_chunks)

    return reduce_chunks(cfg)
