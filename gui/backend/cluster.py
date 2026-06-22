"""Cluster job submission for the GUI — drives the existing ``cluster/`` scripts.

The flow mirrors the documented manual steps (``cluster/README.md``):

  1. ``cluster/stage.sh`` (runs locally) rsyncs the FEM + sensors + ``src/`` +
     ``cluster/`` + ``configs/`` to ``<host>:${HOME}/Scratch/inob/...``.
  2. The resolved run config (with any clicked ``forward.point_sources``) is
     rsynced to ``<base>/configs/default.yaml`` — the body scripts read that
     name specifically.
  3. ``cluster/submit.sh`` calls ``qsub`` and therefore runs ON the cluster, so
     we invoke it over SSH: ``ssh <host> "cd <base>/code && CLUSTER_PROFILE=<p>
     bash cluster/submit.sh array"`` then ``reduce`` (which ``-hold_jid``s on the
     array job).

Live submission needs SSH access + credentials to UCL Myriad/Kathleen. When
``INOB_CLUSTER_DRYRUN=1`` or SSH is unreachable, ``submit``/``status``/``fetch``
return a ``state="dryrun"`` response listing the exact commands that WOULD run,
so the endpoint never fails opaquely.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# gui/backend/cluster.py → project root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_DIR = PROJECT_ROOT / "cluster"
PROFILES_DIR = CLUSTER_DIR / "profiles"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
WORKING_CONFIG = PROJECT_ROOT / "configs" / "gui_working.yaml"

# Remote project base (literal; the remote shell expands ${HOME}).
REMOTE_BASE = "${HOME}/Scratch/inob"

_JOB_ID_RE = re.compile(r"Your job(?:-array)?\s+(\d+)")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ClusterError(ValueError):
    """Bad cluster request (unknown profile, malformed input)."""


# ── pure helpers (unit-tested, no IO) ────────────────────────────────────────

def list_profiles() -> list[str]:
    """Available cluster profiles, from ``cluster/profiles/*.env`` stems."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.env"))


def validate_profile(profile: str) -> None:
    if not profile or not _NAME_RE.match(profile) or profile not in list_profiles():
        raise ClusterError(
            f"unknown cluster profile {profile!r} (have {list_profiles()})"
        )


def profile_vars(profile: str) -> dict[str, str]:
    """Parse the simple ``KEY=VALUE`` lines of a profile .env (ignores arrays)."""
    out: dict[str, str] = {}
    for line in (PROFILES_DIR / f"{profile}.env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not _NAME_RE.match(key):
            continue
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        out[key] = val.replace("\\$", "$")  # REMOTE_BASE="\${HOME}/..." → ${HOME}/...
    return out


def parse_job_id(text: str) -> str | None:
    """Extract the SGE job id from qsub output (handles ``Your job`` and
    ``Your job-array``)."""
    m = _JOB_ID_RE.search(text or "")
    return m.group(1) if m else None


def map_sge_state(code: str) -> str:
    """Map an SGE qstat state code to our coarse state vocabulary."""
    code = (code or "").strip()
    if not code:
        return "done"            # absent from qstat → finished (or never existed)
    if code.startswith("E"):
        return "failed"
    if code in ("qw", "hqw", "hRwq"):
        return "queued"
    if code.startswith("r") or code in ("t", "Rr"):
        return "running"
    return "unknown"


def map_slurm_state(state: str) -> str:
    return {
        "running": "running", "pending": "queued",
        "completed": "done", "failed": "failed", "cancelled": "failed",
    }.get((state or "").strip().lower(), "unknown" if state else "done")


def build_commands(profile: str) -> dict[str, list[str]]:
    """The exact argv lists for the submission flow (pure — used by dry-run + tests)."""
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    return {
        "stage": ["bash", str(CLUSTER_DIR / "stage.sh")],
        "push_config": ["rsync", "-avh", str(WORKING_CONFIG),
                        f"{host}:{REMOTE_BASE}/configs/default.yaml"],
        "submit_array": ["ssh", host,
                         f"cd {REMOTE_BASE}/code && CLUSTER_PROFILE={profile} "
                         f"bash cluster/submit.sh array"],
        "submit_reduce": ["ssh", host,
                          f"cd {REMOTE_BASE}/code && CLUSTER_PROFILE={profile} "
                          f"bash cluster/submit.sh reduce"],
    }


def write_run_config(sources, *, out_path: Path = WORKING_CONFIG) -> Path:
    """Write the run config (default.yaml + injected explicit point sources)."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if sources:
        raw.setdefault("forward", {})["point_sources"] = [
            [float(s["x"]), float(s["y"]), float(s["z"])] for s in sources
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out_path


# ── IO ───────────────────────────────────────────────────────────────────────

def _dryrun() -> bool:
    return os.environ.get("INOB_CLUSTER_DRYRUN") == "1"


def _ssh_reachable(host: str) -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "true"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def _run(argv: list[str], *, env: dict | None = None, timeout: int = 900):
    return subprocess.run(
        argv, cwd=PROJECT_ROOT, env={**os.environ, **(env or {})},
        capture_output=True, text=True, timeout=timeout,
    )


def submit(profile, *, stages=None, sources=None, threshold_snr=3.0,
           modality="meg") -> dict:
    """Stage inputs and submit the array + reduce jobs (or dry-run)."""
    validate_profile(profile)
    cmds = build_commands(profile)
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    write_run_config(sources or [])

    if _dryrun() or not _ssh_reachable(host):
        return {
            "profile": profile, "state": "dryrun", "job_id": None,
            "message": ("dry-run: SSH unreachable or INOB_CLUSTER_DRYRUN=1; "
                        "commands not executed"),
            "commands": [" ".join(c) for c in
                         (cmds["stage"], cmds["push_config"],
                          cmds["submit_array"], cmds["submit_reduce"])],
        }

    env = {"CLUSTER_PROFILE": profile}
    log: list[str] = []
    for key in ("stage", "push_config"):
        cp = _run(cmds[key], env=env)
        log.append(cp.stdout + cp.stderr)
        if cp.returncode != 0:
            return {"profile": profile, "state": "failed", "job_id": None,
                    "message": f"{key} failed", "log": "\n".join(log)[-4000:]}
    cp = _run(cmds["submit_array"], env=env, timeout=180)
    log.append(cp.stdout + cp.stderr)
    job_id = parse_job_id(cp.stdout + cp.stderr)
    _run(cmds["submit_reduce"], env=env, timeout=180)  # holds on vagus_fwd
    return {"profile": profile, "state": "submitted", "job_id": job_id,
            "message": f"submitted array+reduce to {host}",
            "log": "\n".join(log)[-4000:]}


def status(profile, job_id) -> dict:
    validate_profile(profile)
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    sched = pv.get("SCHEDULER", "sge").lower()
    if sched == "sge":
        cmd = ["ssh", host, f"qstat | awk '$1==\"{job_id}\"{{print $5}}'"]
    else:
        cmd = ["ssh", host, f"squeue -h -j {job_id} -o %T"]
    if _dryrun() or not _ssh_reachable(host):
        return {"job_id": job_id, "state": "dryrun", "detail": " ".join(cmd)}
    cp = _run(cmd, timeout=60)
    out = cp.stdout.strip()
    state = map_sge_state(out) if sched == "sge" else map_slurm_state(out)
    return {"job_id": job_id, "state": state, "detail": out}


def fetch(profile, job_id=None) -> dict:
    validate_profile(profile)
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    dest = PROJECT_ROOT / "outputs" / "forward"
    cmd = ["rsync", "-avh",
           f"{host}:{REMOTE_BASE}/duneuro_leadfield_vagus.npz", f"{dest}/"]
    if _dryrun() or not _ssh_reachable(host):
        return {"state": "dryrun", "command": " ".join(cmd)}
    dest.mkdir(parents=True, exist_ok=True)
    cp = _run(cmd, timeout=900)
    return {"state": "done" if cp.returncode == 0 else "failed",
            "log": (cp.stdout + cp.stderr)[-2000:]}
