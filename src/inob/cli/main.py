"""``inob`` — the single entry point for the whole toolkit.

Dispatches to the per-stage modules in :mod:`inob.cli`, each of which keeps its
own ``main(argv) -> int``. Everything here is presentation and routing: command
discovery, grouped help, and turning exceptions into readable messages. The
legacy ``inob-*`` console scripts still call those modules directly, so both
spellings stay in step.

Modules are imported lazily, on dispatch, because several pull in matplotlib or
pyvista — ``inob --help`` should not pay for that.
"""
from __future__ import annotations

import difflib
import importlib
import os
import sys
from dataclasses import dataclass

from inob import __version__
from inob.cli import _ui


@dataclass(frozen=True)
class Command:
    name: str
    module: str
    group: str
    summary: str
    aliases: tuple[str, ...] = ()
    needs_duneuro: bool = False
    """Whether this stage calls into the compiled ``duneuropy`` extension.

    A DUNEuro build only works with the one interpreter it was compiled for,
    which is usually not the one on your PATH. Commands flagged here re-run
    themselves under a capable interpreter when the current one cannot import
    it (see :func:`inob.duneuro_env.reexec_with_duneuro`), so "which Python am
    I using" stops being something the user has to track.
    """


# Ordered: the groups render in this sequence, and so do commands within them.
COMMANDS: tuple[Command, ...] = (
    # --- Start here -------------------------------------------------------
    Command("doctor", "inob.cli.doctor", "Start here",
            "Check this machine can run the pipeline"),
    Command("status", "inob.cli.status", "Start here",
            "Show what's built and what to run next"),
    Command("run", "inob.cli.pipeline", "Start here",
            "Run the whole pipeline (geometry to leadfield)",
            aliases=("pipeline",), needs_duneuro=True),

    # --- Build the model --------------------------------------------------
    Command("build-geom", "inob.cli.build_geom", "Build the model",
            "STLs to watertight anatomical geometry"),
    Command("build-fem", "inob.cli.build_fem", "Build the model",
            "Geometry to multi-tissue tetrahedral FEM mesh"),
    Command("sensors", "inob.cli.generate_sensors", "Build the model",
            "Place the triaxial OPM sensor array"),
    Command("electrodes", "inob.cli.generate_electrodes", "Build the model",
            "Place the HD surface-electrode array"),

    # --- Solve ------------------------------------------------------------
    Command("forward", "inob.cli.run_forward", "Solve",
            "MEG leadfield via DUNEuro", needs_duneuro=True),
    Command("eeg", "inob.cli.run_eeg", "Solve",
            "EEG leadfield via DUNEuro", needs_duneuro=True),
    Command("volume-field", "inob.cli.volume_field", "Solve",
            "The solved field inside the body, not just at the sensors",
            needs_duneuro=True),

    # --- Analyse ----------------------------------------------------------
    Command("detect", "inob.cli.detectability", "Analyse",
            "Trials needed to detect each source"),
    Command("snr", "inob.cli.snr", "Analyse",
            "Predicted signal-to-noise per source"),
    Command("sensitivity", "inob.cli.sensitivity", "Analyse",
            "How much conductivity uncertainty matters"),
    Command("location", "inob.cli.location", "Analyse",
            "Whole-body EEG vs cervical paddle placement"),
    Command("cross", "inob.cli.cross_modality", "Analyse",
            "MEG and EEG coupling for the same source"),
    Command("physiology", "inob.cli.physiology", "Analyse",
            "Simulate moving-dipole physiological activity"),
    Command("cap-compare", "inob.cli.cap_compare", "Analyse",
            "Propagating action potential vs stationary dipole"),
    Command("source-models", "inob.cli.source_models", "Analyse",
            "OPM vs electrodes on stationary and ascending cord activity"),

    # --- Validate ---------------------------------------------------------
    Command("ladder", "inob.cli.ladder", "Validate",
            "Compare Biot–Savart, Sarvas and FEM rungs"),
    Command("sarvas", "inob.cli.sarvas", "Validate",
            "Benchmark the FEM against the analytic sphere"),
    Command("calibrate", "inob.cli.calibrate", "Validate",
            "Calibrate DUNEuro EEG output to absolute units",
            needs_duneuro=True),

    # --- Figures ----------------------------------------------------------
    Command("topoplot", "inob.cli.topoplot", "Figures",
            "MEG/EEG field maps on the body surface"),
    Command("torso", "inob.cli.torso", "Figures",
            "Four-panel field map painted on the body itself"),
    Command("sensor-field", "inob.cli.sensor_field", "Figures",
            "Dipolar pattern, falloff and along-axis strength at the array"),
    Command("visualise", "inob.cli.visualise", "Figures",
            "Render geometry and FEM mesh PNGs"),
    Command("muscle-sources", "inob.cli.muscle_sources", "Figures",
            "Muscle source dipoles: L/R pairing + fibre orientation (pre-solve)"),
)

BY_NAME: dict[str, Command] = {}
for _cmd in COMMANDS:
    BY_NAME[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        BY_NAME[_alias] = _cmd

GROUPS: tuple[str, ...] = (
    "Start here", "Build the model", "Solve", "Analyse", "Validate", "Figures",
)

TAGLINE = "imaging neuroscience outside the brain"

# Shown by `inob` with no arguments: the shortest path to a real result.
QUICKSTART = """\
New here? Three commands:

  inob doctor      check your setup is complete
  inob run         build anatomy, mesh, sensors and solve the leadfield
  inob detect      how many trials to detect your source

`inob status` tells you where you are at any point.
"""


def _usage(out) -> None:
    print(f"{_ui.heading('inob')} — {TAGLINE}  "
          f"{_ui.paint(f'v{__version__}', 'dim')}\n", file=out)
    print(f"{_ui.heading('Usage')}  inob <command> [options]\n", file=out)

    pad = max(len(c.name) for c in COMMANDS) + 2
    for group in GROUPS:
        members = [c for c in COMMANDS if c.group == group]
        if not members:
            continue
        print(_ui.heading(group), file=out)
        for cmd in members:
            print(f"  {cmd.name:<{pad}}{cmd.summary}", file=out)
        print("", file=out)

    print(f"{_ui.heading('Common options')}  "
          f"(accepted by every command except doctor/status)", file=out)
    print("  --config PATH        YAML config to use "
          "(default configs/default.yaml)", file=out)
    print("  --set KEY=VALUE      Override one config field; repeatable", file=out)
    print("  --log-level LEVEL    DEBUG / INFO / WARNING / ERROR\n", file=out)

    print(f"{_ui.heading('Examples')}", file=out)
    print("  inob run --stages geom,fem        build anatomy and mesh only",
          file=out)
    print("  inob run --set fem.pitch_mm=2.0   finer mesh than the default",
          file=out)
    print("  inob detect --snr-threshold 5     stricter detection criterion\n",
          file=out)

    print(f"Detailed help for any command: "
          f"{_ui.paint('inob <command> --help', 'cyan')}", file=out)


def _greeting(out) -> None:
    """`inob` with no arguments — orient rather than error out."""
    print(f"{_ui.heading('inob')} — {TAGLINE}  "
          f"{_ui.paint(f'v{__version__}', 'dim')}\n", file=out)
    print(QUICKSTART, file=out)
    print(f"All {len(COMMANDS)} commands: "
          f"{_ui.paint('inob --help', 'cyan')}", file=out)


def _unknown(name: str, out) -> int:
    print(f"{_ui.paint('Unknown command', 'red')}: {name!r}\n", file=out)
    close = difflib.get_close_matches(name, sorted(BY_NAME), n=3, cutoff=0.5)
    if close:
        print("Did you mean:", file=out)
        for match in close:
            print(f"  {_ui.paint(f'inob {match}', 'cyan')}"
                  f"   {BY_NAME[match].summary}", file=out)
        print("", file=out)
    print(f"See all commands: {_ui.paint('inob --help', 'cyan')}", file=out)
    return 2


def _dispatch(cmd: Command, argv: list[str]) -> int:
    module = importlib.import_module(cmd.module)
    # argparse derives its `usage:` line from argv[0]; make it read
    # "inob build-fem" rather than the bare "inob" wrapper.
    original = sys.argv[0]
    sys.argv[0] = f"inob {cmd.name}"
    try:
        return int(module.main(argv))
    finally:
        sys.argv[0] = original


def _maybe_reexec(cmd: Command, rest: list[str]) -> None:
    """Hand a DUNEuro stage to an interpreter that can actually run it.

    Does nothing for commands that never touch ``duneuropy``, for ``--help``
    (which must stay instant and must not depend on a build existing), and
    whenever the current interpreter can already import the extension — which
    is the common case and costs one ``find_spec``.
    """
    if not cmd.needs_duneuro:
        return
    if any(a in ("-h", "--help") for a in rest):
        return
    from inob.duneuro_env import reexec_with_duneuro

    reexec_with_duneuro()      # replaces this process when a switch is needed


def _debug_enabled(argv: list[str]) -> bool:
    return "--debug" in argv or bool(os.environ.get("INOB_DEBUG"))


def main(argv: list[str] | None = None) -> int:
    # Only a real command line may re-exec (see _maybe_reexec): when a caller
    # passes argv explicitly — a test, the GUI backend — sys.argv belongs to
    # something else, and replacing that process would be wrong.
    from_command_line = argv is None
    argv = list(sys.argv[1:] if argv is None else argv)
    debug = _debug_enabled(argv)
    argv = [a for a in argv if a != "--debug"]

    if not argv:
        _greeting(sys.stdout)
        return 0

    head, rest = argv[0], argv[1:]

    if head in ("-h", "--help", "help"):
        if rest and rest[0] in BY_NAME:
            return _dispatch(BY_NAME[rest[0]], ["--help"])
        _usage(sys.stdout)
        return 0

    if head in ("-V", "--version", "version"):
        print(f"inob {__version__}")
        return 0

    if head not in BY_NAME:
        return _unknown(head, sys.stderr)

    if from_command_line:
        _maybe_reexec(BY_NAME[head], rest)

    try:
        return _dispatch(BY_NAME[head], rest)
    except KeyboardInterrupt:
        print(f"\n{_ui.paint('Interrupted.', 'yellow')} Finished stages are "
              f"kept — rerun to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        if debug:
            raise
        _report(exc, head, sys.stderr)
        return 1


def _report(exc: Exception, command: str, out) -> None:
    """Turn the exceptions users actually hit into an actionable message."""
    from inob.config import ConfigError

    label = type(exc).__name__
    hints: list[str] = []

    if isinstance(exc, ConfigError):
        label = "Config error"
        hints = ["Check the field named above in your --config file.",
                 "Compare against configs/default.yaml."]
    elif isinstance(exc, FileNotFoundError):
        label = "Missing input"
        hints = ["`inob status` shows which stages have been built.",
                 "`inob run` builds everything that's missing."]
    elif isinstance(exc, ModuleNotFoundError) and "duneuro" in str(exc).lower():
        label = "DUNEuro not available"
        hints = ["`inob doctor` explains how to build duneuropy.",
                 "Every stage except forward/eeg runs without it."]
    elif isinstance(exc, MemoryError):
        label = "Out of memory"
        hints = ["Coarsen the mesh: --set fem.pitch_mm=4.0",
                 "Or run the forward solve on the cluster (see cluster/)."]

    print(f"{_ui.paint(label, 'red')}: {exc}\n", file=out)
    for h in hints:
        print(f"  {_ui.arrow()} {_ui.paint(h, 'cyan')}", file=out)
    if hints:
        print("", file=out)
    print(_ui.paint(f"Full traceback: inob {command} --debug ...", "dim"),
          file=out)


if __name__ == "__main__":
    sys.exit(main())
