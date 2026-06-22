"""Conductivity sensitivity sweep.

For each (tissue, perturbation) in ``cfg.sensitivity``, recompute the
forward solution with that tissue's conductivity scaled, then summarise the
per-channel relative change. This is the cleanest finding for the paper:
the magnetic forward solution is insensitive to bone/skin conductivity
uncertainty, while the electric (surface-potential) one is dominated by it.

Output: ``cfg.outputs.sensitivity_dir/sensitivity_<modality>.json`` with
per-tissue, per-perturbation summaries (rms, p50, p95 of the relative change
in the leadfield's channel-RMS amplitude per source). Optionally save the
perturbed leadfields too — but those are large.

Recomputing the leadfield is the expensive step (~7 min per chunk × 32
chunks for the full 8190-channel array). For research-paper figures we
typically run on a *coarse* sensor subset or use the cluster.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from inob.config import Config
from inob.io.npz import load_leadfield

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    tissue: str
    factor: float
    rms_rel_change: float
    p50_rel_change: float
    p95_rel_change: float


def _channel_amplitude(L: np.ndarray) -> np.ndarray:
    """RMS amplitude per channel (across all source moments)."""
    return np.sqrt(np.mean(L ** 2, axis=1))


def relative_change(L_perturbed: np.ndarray, L_baseline: np.ndarray) -> np.ndarray:
    """|ΔL|/|L_baseline| per channel, clipped to avoid div-by-zero."""
    a_b = _channel_amplitude(L_baseline)
    a_p = _channel_amplitude(L_perturbed)
    eps = max(1e-30, 1e-6 * a_b.max())
    return np.abs(a_p - a_b) / np.maximum(a_b, eps)


def summarise_perturbation(
    rel: np.ndarray, *, tissue: str, factor: float,
) -> PerturbationResult:
    return PerturbationResult(
        tissue=tissue,
        factor=factor,
        rms_rel_change=float(np.sqrt(np.mean(rel ** 2))),
        p50_rel_change=float(np.percentile(rel, 50)),
        p95_rel_change=float(np.percentile(rel, 95)),
    )


def sweep(
    cfg: Config,
    *,
    baseline_path: Path,
    forward_fn: Callable[[Config, Path], np.ndarray],
    out_path: Path,
    modality: str = "meg",
) -> dict[str, list[dict]]:
    """Run a conductivity sensitivity sweep.

    ``forward_fn(cfg, out_npz_path)`` should run the chosen modality's forward
    solve and write the leadfield NPZ to ``out_npz_path``. The function uses
    ``cfg.forward.conductivities_sm`` — the caller swaps that field through
    ``Config`` overrides for each perturbation.

    Returns the per-modality summary dict written to ``out_path``.
    """
    baseline = load_leadfield(baseline_path)
    L0 = baseline.L
    sweep_dir = cfg.outputs.sensitivity_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results: list[PerturbationResult] = []
    for tissue in cfg.sensitivity.tissues:
        if tissue not in cfg.forward.conductivities_sm:
            logger.warning("sensitivity.tissues: %r not in conductivities_sm; skipping",
                           tissue)
            continue
        sigma0 = cfg.forward.conductivities_sm[tissue]
        for factor in cfg.sensitivity.perturbations:
            new_sigma = float(sigma0) * float(factor)
            logger.info("[sensitivity] %s × %.2f (σ %.4g → %.4g S/m)",
                        tissue, factor, sigma0, new_sigma)

            from dataclasses import replace
            new_cond = dict(cfg.forward.conductivities_sm)
            new_cond[tissue] = new_sigma
            cfg_p = replace(cfg, forward=replace(cfg.forward, conductivities_sm=new_cond))

            tag = f"{modality}_{tissue}_{int(factor * 100):03d}"
            out_npz = sweep_dir / f"L_{tag}.npz"
            L = forward_fn(cfg_p, out_npz)
            rel = relative_change(L, L0)
            res = summarise_perturbation(rel, tissue=tissue, factor=factor)
            logger.info(
                "  rel change rms=%.3f%% p50=%.3f%% p95=%.3f%%",
                100 * res.rms_rel_change, 100 * res.p50_rel_change,
                100 * res.p95_rel_change,
            )
            results.append(res)

    summary = {
        "modality": modality,
        "baseline_path": str(baseline_path),
        "results": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("[saved] %s", out_path)
    return summary
