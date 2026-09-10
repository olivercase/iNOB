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
# The config actually shipped to the cluster. Kept separate from WORKING_CONFIG
# so submitting a job cannot overwrite the settings the user is editing in the
# GUI — it previously wrote the *defaults* back over the working copy, silently
# discarding every advanced edit both locally and on the cluster.
SUBMIT_CONFIG = PROJECT_ROOT / "configs" / "gui_submit.yaml"

# Remote project base (literal; the remote shell expands ${HOME}).
REMOTE_BASE = "${HOME}/Scratch/inob"

_JOB_ID_RE = re.compile(r"Your job(?:-array)?\s+(\d+)")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# A job id we are willing to interpolate into a remote shell command. SGE and
# Slurm ids are integers, optionally with an array-task suffix (``12345.7``,
# ``12345_7``). Anything else is rejected outright rather than quoted, because
# there is no legitimate job id that needs shell metacharacters.
_JOB_ID_SAFE_RE = re.compile(r"^\d{1,15}(?:[._]\d{1,6})?$")


class ClusterError(ValueError):
    """Bad cluster request (unknown profile, malformed input)."""


def validate_job_id(job_id: object) -> str:
    """Return ``job_id`` if it is a syntactically valid scheduler job id.

    ``status()`` interpolates the job id into a command string that the REMOTE
    shell executes over SSH. Without this check a request like
    ``?job_id=1"; rm -rf ~ ;#`` would run arbitrary commands on the cluster
    under the user's credentials, so the id is validated (not merely quoted)
    before it can reach a shell.
    """
    text = "" if job_id is None else str(job_id).strip()
    if not _JOB_ID_SAFE_RE.match(text):
        raise ClusterError(
            f"invalid job id {text!r} — expected a scheduler job id such as '12345' or '12345.7'"
        )
    return text


# ── pure helpers (unit-tested, no IO) ────────────────────────────────────────


def list_profiles() -> list[str]:
    """Available cluster profiles, from ``cluster/profiles/*.env`` stems."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.env"))


def validate_profile(profile: str) -> None:
    if not profile or not _NAME_RE.match(profile) or profile not in list_profiles():
        raise ClusterError(f"unknown cluster profile {profile!r} (have {list_profiles()})")


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
        return "done"  # absent from qstat → finished (or never existed)
    if code.startswith("E"):
        return "failed"
    if code in ("qw", "hqw", "hRwq"):
        return "queued"
    if code.startswith("r") or code in ("t", "Rr"):
        return "running"
    return "unknown"


def map_slurm_state(state: str) -> str:
    return {
        "running": "running",
        "pending": "queued",
        "completed": "done",
        "failed": "failed",
        "cancelled": "failed",
    }.get((state or "").strip().lower(), "unknown" if state else "done")


MODALITIES = ("meg", "eeg")


def validate_modality(modality: object) -> str:
    m = str(modality or "meg").strip().lower()
    if m not in MODALITIES:
        raise ClusterError(f"unknown modality {m!r} (have {list(MODALITIES)})")
    return m


def build_commands(profile: str, *, modality: str = "meg") -> dict[str, list[str]]:
    """The exact argv lists for the submission flow (pure — used by dry-run + tests).

    ``modality`` selects which ``cluster/submit.sh`` task(s) to queue: MEG is
    the chunked ``array`` solve plus the ``reduce`` that stitches the chunks,
    EEG is the single ``eeg`` job. It used to be accepted and ignored, so an
    EEG request silently queued a MEG solve.
    """
    modality = validate_modality(modality)
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)

    def _submit(task: str) -> list[str]:
        return [
            "ssh",
            host,
            f"cd {REMOTE_BASE}/code && CLUSTER_PROFILE={profile} bash cluster/submit.sh {task}",
        ]

    cmds = {
        "stage": ["bash", str(CLUSTER_DIR / "stage.sh")],
        "push_config": [
            "rsync",
            "-avh",
            str(SUBMIT_CONFIG),
            f"{host}:{REMOTE_BASE}/configs/default.yaml",
        ],
    }
    if modality == "eeg":
        cmds["submit_eeg"] = _submit("eeg")
    else:
        cmds["submit_array"] = _submit("array")
        cmds["submit_reduce"] = _submit("reduce")
    return cmds


def submit_tasks(modality: str = "meg") -> tuple[str, ...]:
    """Which ``build_commands`` keys are the actual job submissions, in order."""
    return (
        ("submit_eeg",)
        if validate_modality(modality) == "eeg"
        else ("submit_array", "submit_reduce")
    )


def _active_config_path() -> Path:
    """The config the GUI currently edits — working copy if present, else default.

    Mirrors ``app._active_config_path`` so a cluster run solves exactly what the
    GUI shows, rather than the shipped defaults.
    """
    return WORKING_CONFIG if WORKING_CONFIG.exists() else DEFAULT_CONFIG


def write_run_config(sources, *, out_path: Path | None = None) -> Path:
    """Write the config to submit: the *active* GUI config + explicit sources.

    Reads whatever the GUI is currently editing (not ``default.yaml``) and
    writes to a dedicated submission file, so a submit neither ships stale
    defaults nor clobbers the user's working copy.
    """
    out_path = SUBMIT_CONFIG if out_path is None else out_path
    raw = yaml.safe_load(_active_config_path().read_text()) or {}
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
            capture_output=True,
            timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def _run(argv: list[str], *, env: dict | None = None, timeout: int = 900):
    return subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def submit(profile, *, sources=None, modality="meg") -> dict:
    """Stage inputs and submit the solve jobs for ``modality`` (or dry-run).

    Note there is deliberately no ``stages`` or ``threshold_snr`` parameter.
    The cluster only ever runs the forward solve; stage selection and the
    detection threshold are local post-processing applied once the leadfield
    is fetched back. Both used to be accepted here and silently ignored, which
    made the GUI look like it was honouring settings it wasn't.
    """
    validate_profile(profile)
    modality = validate_modality(modality)
    cmds = build_commands(profile, modality=modality)
    tasks = submit_tasks(modality)
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    write_run_config(sources or [])

    if _dryrun() or not _ssh_reachable(host):
        return {
            "profile": profile,
            "state": "dryrun",
            "job_id": None,
            "modality": modality,
            "message": ("dry-run: SSH unreachable or INOB_CLUSTER_DRYRUN=1; commands not executed"),
            "commands": [" ".join(cmds[k]) for k in ("stage", "push_config", *tasks)],
        }

    env = {"CLUSTER_PROFILE": profile}
    log: list[str] = []
    for key in ("stage", "push_config"):
        cp = _run(cmds[key], env=env)
        log.append(cp.stdout + cp.stderr)
        if cp.returncode != 0:
            return {
                "profile": profile,
                "state": "failed",
                "job_id": None,
                "modality": modality,
                "message": f"{key} failed",
                "log": "\n".join(log)[-4000:],
            }

    # The first task carries the job id we report back; later tasks (reduce)
    # -hold_jid on it. A non-zero exit on any of them is a failed submission,
    # not a success with a missing id — the old code ignored reduce entirely.
    job_id = None
    for i, key in enumerate(tasks):
        cp = _run(cmds[key], env=env, timeout=180)
        log.append(cp.stdout + cp.stderr)
        if cp.returncode != 0:
            return {
                "profile": profile,
                "state": "failed",
                "job_id": job_id,
                "modality": modality,
                "message": f"{key} failed",
                "log": "\n".join(log)[-4000:],
            }
        if i == 0:
            job_id = parse_job_id(cp.stdout + cp.stderr)
    return {
        "profile": profile,
        "state": "submitted",
        "job_id": job_id,
        "modality": modality,
        "message": f"submitted {'+'.join(t[7:] for t in tasks)} to {host}",
        "log": "\n".join(log)[-4000:],
    }


def status(profile, job_id) -> dict:
    validate_profile(profile)
    job_id = validate_job_id(job_id)  # never interpolate unvalidated input
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    sched = pv.get("SCHEDULER", "sge").lower()
    # Safe to interpolate: validate_job_id has restricted job_id to digits with
    # an optional .N/_N task suffix, so it cannot carry shell or awk syntax.
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


def leadfield_name(modality: str = "meg") -> str:
    """Filename of the leadfield the cluster reduce step writes.

    ``cluster/run_reduce.sh`` names it ``duneuro_leadfield_${TARGET_TAG}.npz``,
    so it tracks the configured source target (vagus / spine / muscle / …).
    This used to be hardcoded to ``duneuro_leadfield_vagus.npz``, which fetched
    nothing at all for any non-vagus run.
    """
    modality = validate_modality(modality)
    try:
        from inob.config import load_config

        cfg = load_config(_active_config_path(), project_root=PROJECT_ROOT)
        npz = cfg.outputs.forward_eeg_npz if modality == "eeg" else cfg.outputs.forward_npz
        return Path(npz).name
    except Exception as e:
        # An unreadable config must not make fetching impossible; fall back to
        # the untagged default rather than a target that may well be wrong.
        logger.warning("could not resolve leadfield name from config: %s", e)
        return "duneuro_eeg_leadfield.npz" if modality == "eeg" else "duneuro_leadfield.npz"


def fetch(profile, job_id=None, *, modality: str = "meg") -> dict:
    validate_profile(profile)
    if job_id is not None:
        validate_job_id(job_id)
    modality = validate_modality(modality)
    pv = profile_vars(profile)
    host = pv.get("REMOTE_HOST", profile)
    dest = PROJECT_ROOT / "outputs" / "forward"
    name = leadfield_name(modality)
    cmd = ["rsync", "-avh", f"{host}:{REMOTE_BASE}/{name}", f"{dest}/"]
    if _dryrun() or not _ssh_reachable(host):
        return {
            "state": "dryrun",
            "modality": modality,
            "leadfield": name,
            "command": " ".join(cmd),
        }
    dest.mkdir(parents=True, exist_ok=True)
    cp = _run(cmd, timeout=900)
    return {
        "state": "done" if cp.returncode == 0 else "failed",
        "modality": modality,
        "leadfield": name,
        "log": (cp.stdout + cp.stderr)[-2000:],
    }
