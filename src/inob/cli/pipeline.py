"""End-to-end pipeline orchestrator.

Chains the geometry, FEM, sensor, forward, and visualisation stages with
file-based skip logic. Each stage's output is validated via its schema before
the orchestrator declares it "done"; a failed stage drops a ``.FAILED``
marker in its output dir so reruns know to retry it.

Usage
-----
    inob-pipeline                                    # run all stages
    inob-pipeline --stages geom,fem                  # just two
    inob-pipeline --force                            # ignore existing outputs
    inob-pipeline --force --stages forward           # rebuild forward only
    inob-pipeline --skip-viz                         # skip the PNG render
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inob.cli._common import add_common_args, setup
from inob.config import Config

logger = logging.getLogger(__name__)

ALL_STAGES: tuple[str, ...] = ("geom", "fem", "sensors", "forward", "viz")


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    output_paths: tuple[str, ...]    # config field names whose paths must exist
    run: Callable[[Config], object]


def _stage_geom(cfg: Config) -> object:
    from inob.geometry.builder import build_geometry
    return build_geometry(cfg)


def _stage_fem(cfg: Config) -> object:
    from inob.fem.cgal_builder import build_fem
    return build_fem(cfg)


def _stage_sensors(cfg: Config) -> object:
    from inob.sensors.triaxial import generate_sensor_array
    return generate_sensor_array(cfg)


def _stage_forward(cfg: Config) -> object:
    # Default forward path: local, split across all cores (forward.local_workers,
    # 0 = all). Falls back to the serial solve for a single worker.
    from inob.forward.local import run_forward_local
    return run_forward_local(cfg)


def _stage_viz(cfg: Config) -> object:
    from inob.viz.fem import render_fem
    from inob.viz.geometry import render_geometry
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
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, str]:
    """Run the requested stages in dependency order.

    Returns a per-stage status map (``"skipped"`` / ``"ran"`` / ``"failed"`` /
    ``"cancelled"``). Stages whose primary outputs already exist are skipped
    unless ``force``.

    ``should_cancel``, when given, is polled between stages and stops the run
    at the next stage boundary. It deliberately cannot interrupt a stage that
    is already executing — a DUNEuro solve is opaque C++ — so a cancel takes
    effect once the current stage finishes. Callers that need the distinction
    should look for ``"cancelled"`` in the returned map.
    """
    statuses: dict[str, str] = {}
    for name in ALL_STAGES:
        if name not in stages:
            continue
        if should_cancel is not None and should_cancel():
            logger.info("[stop] cancelled before %s", name)
            statuses[name] = "cancelled"
            break
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


# What each stage needs on disk before it can start. Checked at the CLI layer
# so `--stages forward` reports a missing mesh up front instead of failing
# several minutes deep inside the solver.
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "geom": (),
    "fem": ("geom",),
    "sensors": ("geom",),
    "forward": ("fem", "sensors"),
    "viz": ("geom", "fem"),
}


def missing_prerequisites(
    cfg: Config, stages: list[str],
) -> list[tuple[str, str]]:
    """Return ``(stage, unmet_prerequisite)`` pairs for this stage selection.

    A prerequisite is satisfied if it is also being run now, or if its outputs
    already exist on disk.
    """
    requested = set(stages)
    unmet: list[tuple[str, str]] = []
    for name in stages:
        for prereq in STAGE_REQUIRES.get(name, ()):
            if prereq in requested:
                continue
            if not _all_outputs_exist(cfg, STAGES[prereq]):
                unmet.append((name, prereq))
    return unmet


def _parse_stages(arg: str | None) -> list[str]:
    # `None` means "flag omitted" → run everything. But an *explicit* empty or
    # whitespace value (e.g. `--stages ""` from an unset shell variable) is a
    # mistake: silently running the entire pipeline, including a slow forward
    # solve, is the wrong thing to do on what is almost certainly a typo.
    if arg is None or arg.lower() == "all":
        return list(ALL_STAGES)
    out: list[str] = []
    for s in arg.split(","):
        s = s.strip()
        if not s:
            continue
        if s not in STAGES:
            raise ValueError(f"unknown stage {s!r}; valid: {', '.join(STAGES)}")
        out.append(s)
    if not out:
        raise ValueError(
            f"no stages selected from {arg!r}; pass a comma-separated subset "
            f"of {', '.join(STAGES)}, or 'all'"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build anatomy, mesh, sensors and leadfield, in order. "
                    "Stages whose outputs already exist are skipped.",
        epilog=(
            "examples:\n"
            "  inob run                          everything that's missing\n"
            "  inob run --stages geom,fem        anatomy and mesh only\n"
            "  inob run --force --stages fem     rebuild the mesh\n"
            "  inob run --set fem.pitch_mm=2.0   finer mesh\n"
            "\nCheck progress at any time with `inob status`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    unmet = missing_prerequisites(cfg, stages)
    if unmet:
        for stage, prereq in unmet:
            logger.error("stage %r needs %r, which has not been built",
                         stage, prereq)
        # Expand transitively: 'fem' is no use as a suggestion if 'geom' is
        # missing too. Report them in pipeline order.
        needed: set[str] = set()
        pending = [prereq for _, prereq in unmet]
        while pending:
            name = pending.pop()
            if name in needed:
                continue
            needed.add(name)
            pending.extend(
                req for req in STAGE_REQUIRES.get(name, ())
                if not _all_outputs_exist(cfg, STAGES[req])
            )
        ordered = [s for s in ALL_STAGES if s in needed]
        logger.error("build it first: inob run --stages %s",
                     ",".join(ordered))
        logger.error("or let the pipeline sort it out: inob run")
        return 2

    logger.info("running stages: %s", stages)
    statuses = run_pipeline(cfg, stages=stages, force=args.force)
    logger.info("[summary] %s",
                " ".join(f"{k}={v}" for k, v in statuses.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
