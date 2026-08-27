"""``inob status`` — what has been built, what is missing, what to run next.

Read-only: it never loads meshes, never configures file logging, and never
touches ``outputs/``. Safe to run at any time, including mid-solve.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from inob.cli import _ui
from inob.cli._common import DEFAULT_CONFIG, apply_source_target
from inob.cli.pipeline import DEFAULT_STAGES, STAGES
from inob.config import SOURCE_TARGETS, Config, load_config

# Stage name -> the `inob` subcommand that builds it, for the "next step" hint.
STAGE_COMMAND: dict[str, str] = {
    "geom": "inob build-geom",
    "fem": "inob build-fem",
    "sensors": "inob sensors",
    "forward": "inob forward",
    "viz": "inob visualise",
}

# Artifacts outside the pipeline stages a bare run builds, shown as optional
# extras. Figures live here rather than under "Pipeline" because `inob run`
# does not draw them: listing an undrawn PNG as a missing stage would report a
# complete run as incomplete, and point "what to run next" at a render nobody
# asked for. Each entry names one or more config fields — `viz` writes two.
EXTRA_ARTIFACTS: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("electrodes", ("electrodes_mat",), "inob electrodes",
     "HD surface-electrode array"),
    ("eeg", ("forward_eeg_npz",), "inob eeg",
     "EEG leadfield via DUNEuro"),
    ("figures", ("geometry_png", "fem_png"), "inob run --with-viz",
     "geometry + FEM overview PNGs"),
)


@dataclass(frozen=True)
class Artifact:
    name: str
    description: str
    command: str
    paths: tuple[Path, ...]
    built: bool
    failed: bool

    @property
    def state(self) -> str:
        if self.failed:
            return "failed"
        return "built" if self.built else "missing"


def _failed_marker(paths: tuple[Path, ...], name: str) -> Path:
    return paths[0].parent / f".{name}.FAILED"


def collect(cfg: Config) -> tuple[list[Artifact], list[Artifact]]:
    """Return ``(pipeline_stages, extras)`` with their on-disk state."""
    stages: list[Artifact] = []
    for name in DEFAULT_STAGES:
        stage = STAGES[name]
        paths = tuple(getattr(cfg.outputs, f) for f in stage.output_paths)
        stages.append(Artifact(
            name=name,
            description=stage.description,
            command=STAGE_COMMAND.get(name, f"inob {name}"),
            paths=paths,
            built=all(p.exists() for p in paths),
            failed=_failed_marker(paths, name).exists(),
        ))

    extras: list[Artifact] = []
    for name, fields, command, description in EXTRA_ARTIFACTS:
        paths = tuple(getattr(cfg.outputs, f) for f in fields)
        extras.append(Artifact(
            name=name, description=description, command=command,
            paths=paths, built=all(p.exists() for p in paths), failed=False,
        ))
    return stages, extras


def forward_physics(cfg: Config) -> list[tuple[str, str]]:
    """The forward-solve choices a leadfield on disk cannot tell you about.

    Source model and muscle anisotropy change the numbers in the NPZ without
    changing its shape, name or schema, so two leadfields solved with different
    settings are indistinguishable once written. Showing them here means the
    setting is visible next to the artefact it produced.
    """
    sm = cfg.forward.source_model
    detail = sm.type.replace("_", " ")
    if sm.type.endswith("venant"):
        patch = "one tissue" if sm.restrict else "across tissues"
        detail += f" ({sm.number_of_moments} moments, {patch})"

    aniso = cfg.forward.muscle_anisotropy
    active = aniso.active_for(cfg.forward.source_tissue)
    ratio = aniso.sigma_long_sm / max(aniso.sigma_trans_sm, 1e-12)
    aniso_detail = (
        f"{aniso.mode} — {'on' if active else 'off'} for {cfg.forward.source_tissue}"
    )
    if active:
        aniso_detail += f", {ratio:.1f}:1"

    return [
        ("source model", detail),
        ("muscle anisotropy", aniso_detail),
        ("solver", f"{cfg.forward.solver.type}, reduction {cfg.forward.solver.reduction:g}"),
    ]


def next_step(stages: list[Artifact]) -> str | None:
    """The command that makes the most progress from here, if any."""
    for stage in stages:
        if stage.failed:
            return f"{stage.command} --force    # retry the failed stage"
        if not stage.built:
            remaining = [s for s in stages if not s.built]
            if len(remaining) > 1:
                return "inob run    # builds everything still missing"
            return stage.command
    return None


def _describe(path: Path, root: Path) -> str:
    shown = _ui.rel(path, root)
    if not path.exists():
        return _ui.paint(shown, "dim")
    if path.is_dir():
        return shown
    stat = path.stat()
    detail = f"{_ui.human_bytes(stat.st_size)}, {_ui.human_age(stat.st_mtime)}"
    return f"{shown}  {_ui.paint(detail, 'dim')}"


def _render(cfg: Config, config_path: Path, stages: list[Artifact],
            extras: list[Artifact], out) -> None:
    root = cfg.project_root
    # One column width across both tables so the two line up.
    pad = max(len(a.name) for a in (*stages, *extras)) + 2
    indent = " " * (pad + 4)

    print(f"{_ui.heading('Project')}  {root}", file=out)
    print(f"{_ui.heading('Config')}   {_ui.rel(config_path, root)}\n", file=out)

    print(_ui.heading("Pipeline"), file=out)
    for stage in stages:
        glyph = _ui.mark(
            "fail" if stage.failed else "ok" if stage.built else "miss")
        print(f"  {glyph} {stage.name:<{pad}}"
              f"{_describe(stage.paths[0], root)}", file=out)
        for extra_path in stage.paths[1:]:
            print(f"{indent}{_describe(extra_path, root)}", file=out)
        if stage.failed:
            print(f"{indent}{_ui.paint('last run failed', 'red')}", file=out)

    print(f"\n{_ui.heading('Optional')}", file=out)
    for extra in extras:
        glyph = _ui.mark("ok" if extra.built else "miss")
        detail = (_describe(extra.paths[0], root) if extra.built
                  else _ui.paint(f"not built — {extra.command}", "dim"))
        print(f"  {glyph} {extra.name:<{pad}}{detail}", file=out)
        if extra.built:
            for extra_path in extra.paths[1:]:
                print(f"{indent}{_describe(extra_path, root)}", file=out)

    print(f"\n{_ui.heading('Forward physics')}", file=out)
    physics = forward_physics(cfg)
    phys_pad = max(len(label) for label, _ in physics) + 2
    for label, detail in physics:
        print(f"    {label:<{phys_pad}}{detail}", file=out)

    step = next_step(stages)
    if step is None:
        print(f"\nEverything is built. Analyse it with "
              f"{_ui.paint('inob detect', 'cyan')} or "
              f"{_ui.paint('inob topoplot', 'cyan')}.", file=out)
    else:
        print(f"\n{_ui.heading('Next')}", file=out)
        print(_ui.hint(step), file=out)


def _as_json(cfg: Config, config_path: Path, stages: list[Artifact],
             extras: list[Artifact]) -> dict:
    def encode(a: Artifact) -> dict:
        return {
            "name": a.name,
            "state": a.state,
            "description": a.description,
            "command": a.command,
            "paths": [str(p) for p in a.paths],
        }
    return {
        "project_root": str(cfg.project_root),
        "config": str(config_path),
        "stages": [encode(s) for s in stages],
        "optional": [encode(e) for e in extras],
        "next": next_step(stages),
        "forward_physics": dict(forward_physics(cfg)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inob status",
        description="Show which pipeline outputs exist and what to run next.",
        epilog="""\
examples:
  inob status                             what's built, and the next command
  inob status --source-target spine       the cord run's own tagged outputs
  inob status --json                      machine-readable, for scripts

Figures are listed but never demanded: a run does not draw them unless asked
(`inob run --with-viz`), so a missing PNG is not a missing stage.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG),
                   help=f"Path to YAML config (default {DEFAULT_CONFIG}).")
    p.add_argument("--project-root", type=Path, default=None,
                   help="Override the project root used to resolve paths.")
    p.add_argument("--source-target", default=None, metavar="TAG",
                   choices=list(SOURCE_TARGETS),
                   help="Report the outputs of one target's run rather than "
                        "the config's defaults. A targeted solve writes tagged "
                        "files (duneuro_leadfield_<TAG>.npz and friends), so "
                        "without this a spine run reads as unbuilt. Same flag, "
                        f"same meaning as on every other command. One of: "
                        f"{', '.join(SOURCE_TARGETS)}.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(argv)

    cfg = load_config(args.config, project_root=args.project_root)
    if args.source_target:
        cfg = apply_source_target(cfg, args.source_target)
    stages, extras = collect(cfg)

    if args.json:
        json.dump(_as_json(cfg, args.config, stages, extras),
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _render(cfg, args.config, stages, extras, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
