"""What this machine can bring to a local solve.

The forward stage splits the sensor array into ``forward.local_workers``
process-parallel chunks (0 == every core, see :mod:`inob.forward.local`). Both
the CLI (``inob doctor``, ``--workers``) and the GUI backend (``/api/system``)
need the real core count to show what "all cores" actually means here, so the
detection lives here once rather than in either front-end.
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CpuInfo:
    """Core counts for the machine this process is running on."""

    platform: str
    machine: str
    logical: int
    physical: int
    # Apple silicon is heterogeneous: the performance-core count is only
    # meaningful there, so it is None everywhere else.
    performance: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sysctl_int(key: str) -> int | None:
    """Read an integer ``sysctl`` value, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return int(value) if out.returncode == 0 and value.isdigit() else None


def cpu_info() -> CpuInfo:
    """Detect the core counts available to a local solve.

    ``logical`` always falls back to ``os.cpu_count()``, so this never raises
    and never returns zero — a capability report must not be able to break the
    caller that is only trying to describe the machine.
    """
    logical = os.cpu_count() or 1
    physical = logical
    performance: int | None = None

    if platform.system() == "Darwin":
        # perflevel0 is the performance-core cluster. Solving on those alone is
        # often faster than oversubscribing across the efficiency cores too, so
        # it is offered as a choice rather than assumed either way.
        physical = _sysctl_int("hw.physicalcpu") or logical
        performance = _sysctl_int("hw.perflevel0.logicalcpu")
        # A homogeneous Mac (Intel) reports every core as perflevel0, which is
        # not a distinct choice — drop it so front-ends don't offer a preset
        # identical to "all cores".
        if performance is not None and performance >= logical:
            performance = None

    return CpuInfo(
        platform=platform.system(),
        machine=platform.machine(),
        logical=logical,
        physical=physical,
        performance=performance,
    )


@dataclass(frozen=True)
class WorkerPreset:
    """A named ``forward.local_workers`` choice, with why you'd pick it."""

    value: int          # 0 == the config's "every core" sentinel
    label: str
    note: str


def worker_presets(info: CpuInfo | None = None) -> list[WorkerPreset]:
    """The worker-count choices worth offering for this machine.

    Shared by the GUI chooser and ``inob doctor`` so both name the same options
    with the same descriptions. ``0`` is kept as the first choice because it is
    the config's own sentinel for "every core" and so keeps following the
    machine if the config moves to a different one.
    """
    info = info or cpu_info()
    presets = [
        WorkerPreset(0, f"All cores ({info.logical})",
                     "Fastest. Uses every core, so the machine will be busy."),
    ]
    if info.performance:
        presets.append(WorkerPreset(
            info.performance, f"Performance cores only ({info.performance})",
            "Skips the efficiency cores. Often nearly as fast as all cores on "
            "Apple silicon, and leaves the machine usable.",
        ))
    half = max(1, info.logical // 2)
    presets.append(WorkerPreset(
        half, f"Half ({half})",
        "Leaves plenty of headroom for other work while it solves.",
    ))
    presets.append(WorkerPreset(
        1, "Single core",
        "Slowest, but the most predictable — and easiest to debug.",
    ))
    return presets
