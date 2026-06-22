"""End-to-end pipeline orchestrator.

Chains the geometry, FEM, sensor, forward, and visualisation stages with
file-based skip logic. Each stage's output is validated via its schema before
the orchestrator declares it "done"; a failed stage drops a ``.FAILED``
marker in its output dir so reruns know to retry it.

Usage
-----
    vagus-fm-pipeline                                    # run all stages
    vagus-fm-pipeline --stages geom,fem                  # just two
    vagus-fm-pipeline --force                            # ignore existing outputs
    vagus-fm-pipeline --force --stages forward           # rebuild forward only
    vagus-fm-pipeline --skip-viz                         # skip the PNG render
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vagus_fm.cli._common import add_common_args, setup
from vagus_fm.config import Config

logger = logging.getLogger(__name__)

ALL_STAGES: tuple[str, ...] = ("geom", "fem", "sensors", "forward", "viz")


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    output_paths: tuple[str, ...]    # config field names whose paths must exist
    run: Callable[[Config], object]


def _stage_geom(cfg: Config) -> object:
    from vagus_fm.geometry.builder import build_geometry
    return build_geometry(cfg)


def _stage_fem(cfg: Config) -> object:
    from vagus_fm.fem.cgal_builder import build_fem
    return build_fem(cfg)


def _stage_sensors(cfg: Config) -> object:
    from vagus_fm.sensors.triaxial import generate_sensor_array
    return generate_sensor_array(cfg)


def _stage_forward(cfg: Config) -> object:
    from vagus_fm.forward.solve import run_forward
    return run_forward(cfg)


def _stage_viz(cfg: Config) -> object:
    from vagus_fm.viz.fem import render_fem
    from vagus_fm.viz.geometry import render_geometry
    render_geometry(cfg)
    render_fem(cfg)
    return cfg.outputs.geometry_png


STAGES: dict[str, Stage] = {
    "geom":    Stage("geom",    "build watertight geometry from STLs",
                     ("geometry_mat",), _stage_geom),
    "fem":     Stage("fem",     "build CGAL multi-tissue FEM mesh",
                     ("fem_mat",), _stage_fem),
    "sensors": Stage("sensors", "place triaxial OPM sensor array",
                     ("sensors_mat",), _stage_sensors),
    "forward": Stage("forward", "DUNEuro forward solve (local)",
                     ("forward_npz",), _stage_forward),
    "viz":     Stage("viz",     "render geometry + FEM PNGs",
                     ("geometry_png", "fem_png"), _stage_viz),
}


def _output_paths(cfg: Config, stage: Stage) -> list[Path]:
    return [getattr(cfg.outputs, f) for f in stage.output_paths]


def _all_outputs_exist(cfg: Config, stage: Stage) -> bool:
    return all(p.exists() for p in _output_paths(cfg, stage))


def _failed_marker(cfg: Config, stage: Stage) -> Path:
    primary = _output_paths(cfg, stage)[0]
    return primary.parent / f".{stage.name}.FAILED"


def run_pipeline(
    cfg: Config, *, stages: list[str], force: bool = False,
) -> dict[str, str]:
    """Run the requested stages in dependency order.

    Returns a per-stage status map (``"skipped"`` / ``"ran"`` / ``"failed"``).
    Stages whose primary outputs already exist are skipped unless ``force``.
    """
    statuses: dict[str, str] = {}
    for name in ALL_STAGES:
        if name not in stages:
            continue
        stage = STAGES[name]
        marker = _failed_marker(cfg, stage)

        if _all_outputs_exist(cfg, stage) and not force and not marker.exists():
            logger.info("[skip] %s: %s exist (use --force to rebuild)",
                        name, ", ".join(p.name for p in _output_paths(cfg, stage)))
            statuses[name] = "skipped"
            continue

        logger.info("[run]  %s: %s", name, stage.description)
        try:
            stage.run(cfg)
        except Exception as e:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{type(e).__name__}: {e}\n")
            logger.error("[FAIL] %s: %s", name, e)
            statuses[name] = "failed"
            raise
        else:
            if marker.exists():
                marker.unlink()
            statuses[name] = "ran"
            logger.info("[ok]   %s", name)

    return statuses


def _parse_stages(arg: str | None) -> list[str]:
    if not arg or arg.lower() == "all":
        return list(ALL_STAGES)
    out: list[str] = []
    for s in arg.split(","):
        s = s.strip()
        if not s:
            continue
        if s not in STAGES:
            raise ValueError(f"unknown stage {s!r}; valid: {', '.join(STAGES)}")
        out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=run_pipeline.__doc__)
    add_common_args(p)
    p.add_argument(
        "--stages", default="all",
        help="Comma-separated stage list (geom,fem,sensors,forward,viz) "
             "or 'all' (default).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild stages even if their outputs already exist.",
    )
    p.add_argument(
        "--skip-viz", action="store_true",
        help="Convenience flag: drop 'viz' from --stages.",
    )
    args = p.parse_args(argv)

    cfg = setup(args, log_prefix="pipeline")
    try:
        stages = _parse_stages(args.stages)
    except ValueError as e:
        logger.error("%s", e)
        return 2
    if args.skip_viz and "viz" in stages:
        stages.remove("viz")

    logger.info("running stages: %s", stages)
    statuses = run_pipeline(cfg, stages=stages, force=args.force)
    logger.info("[summary] %s",
                " ".join(f"{k}={v}" for k, v in statuses.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
