"""FastAPI backend for the iNOB GUI.

Wraps the existing ``inob`` Python pipeline so a browser front-end
(React + Blueprint + react-three-fiber) can:

  * read / edit / validate the YAML config            (``/api/config``)
  * fetch tissue surface meshes for the 3-D viewer     (``/api/meshes``)
  * launch the pipeline and stream logs live           (``ws /api/run``)

There is no MATLAB here — the compute is pure Python (``inob``,
``iso2mesh`` python port, ``duneuropy``). The backend simply imports and
calls the same functions the CLI uses, so the GUI and CLI stay in lock-step.

Run with::

    uvicorn gui.backend.app:app --reload --port 8000

The frontend (Next.js, ``gui/web``, port 3000) proxies ``/api`` here via the
rewrites in ``gui/web/next.config.mjs``; the ``/api/run`` WebSocket connects
straight to this port.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Anything in this process that touches matplotlib must stay off the GUI path:
# its macOS backend refuses to build a figure outside the main thread. The
# pipeline itself now runs in a child process (see /api/run — PyVista's VTK
# window is Cocoa, and Cocoa is main-thread-only), but pin the non-interactive
# backend here too, before anything imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

import yaml
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from gui.backend import cluster, duneuro_setup
from inob import __version__ as _inob_version
from inob.analysis.detect import compute_detectability
from inob.cli.pipeline import ALL_STAGES
from inob.config import ConfigError, load_config

logger = logging.getLogger(__name__)

# Single-flight guard: a DUNEuro solve writes shared cfg.outputs artefacts, so
# only one run may be in flight at a time (two concurrent runs would race on the
# same files and the shared "inob" logger). Acquired non-blocking by /api/run.
_RUN_LOCK = threading.Lock()

# The pipeline child currently in flight, so shutdown can take it down with us
# rather than leaving a headless solve behind.
_ACTIVE_CHILD: dict[str, Any] = {"proc": None}

# ── paths ───────────────────────────────────────────────────────────────────
# gui/backend/app.py  →  project root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
# The GUI edits a working copy so the canonical default.yaml is never clobbered.
WORKING_CONFIG = PROJECT_ROOT / "configs" / "gui_working.yaml"


def _active_config_path() -> Path:
    """The config the GUI currently edits — working copy if present, else default."""
    return WORKING_CONFIG if WORKING_CONFIG.exists() else DEFAULT_CONFIG


def _read_raw(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_raw_safe(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Read a config dict, never raising — a hand-corrupted working file must
    not 500 every endpoint and soft-lock the UI. Returns (dict, errors)."""
    try:
        return _read_raw(path), []
    except (yaml.YAMLError, OSError) as e:
        return {}, [f"could not read {path.name}: {e}"]


def _validate_raw(raw: dict[str, Any]) -> list[str]:
    """Validate a raw config dict by constructing a ``Config``.

    Returns a list of human-readable error strings (empty == valid). Reuses
    the exact same loader the CLI uses, so GUI validation can never drift from
    pipeline validation.
    """
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="inob_cfg_")
    tmp_path = Path(tmp)
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        try:
            load_config(tmp_path, project_root=PROJECT_ROOT)
        except ConfigError as e:
            return [str(e)]
        except Exception as e:  # bad types etc. → a clean 422, not a 500
            return [f"{type(e).__name__}: {e}"]
        return []
    finally:
        os.close(fd)
        tmp_path.unlink(missing_ok=True)


# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="iNOB GUI backend", version=_inob_version)

# In dev the React app runs on a different origin (Vite :5173); allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # legacy Vite frontend
        "http://localhost:3000", "http://127.0.0.1:3000",  # Next.js frontend (gui/web)
        # …and the port Next falls back to when 3000 is taken by something else.
        "http://localhost:3001", "http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def _stop_running_pipeline() -> None:
    _stop_child(_ACTIVE_CHILD.get("proc"))


@app.get("/api/system")
def system_info() -> dict[str, Any]:
    """What this machine can bring to a local solve.

    The forward stage splits the sensor array into ``forward.local_workers``
    process-parallel chunks (0 == every core, see ``inob.forward.local``). The
    GUI needs the real core count to show what "all cores" actually means here
    and to bound the chooser, rather than asking the user to guess.
    """
    import platform

    logical = os.cpu_count() or 1
    physical = logical
    performance: int | None = None
    if platform.system() == "Darwin":
        # Apple silicon is heterogeneous: perflevel0 is the performance core
        # cluster. Solving on those alone is often faster than oversubscribing
        # across the efficiency cores too, so offer it as a choice.
        for key, target in (("hw.physicalcpu", "physical"),
                            ("hw.perflevel0.logicalcpu", "performance")):
            try:
                out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip().isdigit():
                    value = int(out.stdout.strip())
                    if target == "physical":
                        physical = value
                    else:
                        performance = value
            except (OSError, subprocess.SubprocessError):
                pass

    configured = None
    try:
        raw, _ = _read_raw_safe(_active_config_path())
        configured = (raw.get("forward") or {}).get("local_workers")
    except Exception:  # a broken config must not break the capability report
        pass

    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_logical": logical,
        "cpu_physical": physical,
        "cpu_performance": performance,
        "configured_workers": configured,
        # What `local_workers: 0` resolves to right now.
        "all_cores": logical,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "active_config": str(_active_config_path()),
        "stages": list(ALL_STAGES),
    }


# ── which interpreter runs a pipeline ─────────────────────────────────────────
#
# A DUNEuro build is a compiled extension tied to one Python version, and the
# backend does not have to be that version: the pipeline runs as a child
# process, so the solve can use whichever interpreter can actually load
# duneuropy while the API keeps serving on its own. Without this a machine
# whose only build targets 3.11 could never solve from a 3.14 backend, even
# though the CLI ran fine under 3.11.
#
# Order: an explicit INOB_PIPELINE_PYTHON, then this interpreter if it can
# solve, then the venv beside a discovered build — each candidate verified by
# actually importing duneuropy and inob before it is used.

_PIPELINE_PY: dict[str, str] = {}
_TAGS: dict[str, str] = {}


def _can_solve(python: str) -> bool:
    """Can ``python`` run a real solve against *this* checkout?

    Two things have to be true, and both have bitten us: the interpreter must
    load the compiled ``duneuropy`` (it is built for one Python version), and
    its ``inob`` must be the source in this project — a stale editable install
    elsewhere on the machine silently ran a different copy of the pipeline.
    The configured ``forward.duneuro_path`` is honoured here exactly as the
    solver honours it, so a build that only needs a sys.path entry counts.
    """
    duneuro_path = _active_duneuro_path() or ""
    # Never hand a build to an interpreter it was not compiled for: duneuropy
    # has no ABI tag in its filename, so a mismatched interpreter dlopens it
    # and segfaults inside PyInit_duneuropy rather than raising ImportError.
    if duneuro_path:
        build_tag = duneuro_setup._python_tag_for(Path(duneuro_path) / "x")
        if duneuro_setup._tags_conflict(build_tag, _interpreter_tag(python)):
            return False
    probe = (
        "import sys, json\n"
        f"p = {duneuro_path!r}\n"
        "if p: sys.path.insert(0, p)\n"
        "import duneuropy, inob\n"
        "print(inob.__file__)\n"
    )
    try:
        out = subprocess.run(
            [python, "-c", probe], capture_output=True, text=True, timeout=90,
            env={**os.environ,
                 "PYTHONPATH": os.pathsep.join(filter(None, [
                     str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]))},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    where = Path(out.stdout.strip() or "/nonexistent").resolve()
    return str(where).startswith(str((PROJECT_ROOT / "src").resolve()))


def _interpreter_tag(python: str) -> str:
    """Ask an interpreter for its own ``python3.X`` tag. Cheap and crash-free."""
    cached = _TAGS.get(python)
    if cached:
        return cached
    try:
        out = subprocess.run(
            [python, "-c",
             "import sys; print(f'python3.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=30,
        )
        tag = out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        tag = "unknown"
    _TAGS[python] = tag
    return tag


def _venv_python_for(site_packages: str) -> str | None:
    """`.../venv/lib/python3.11/site-packages` → `.../venv/bin/python`."""
    path = Path(site_packages)
    for parent in path.parents:
        candidate = parent / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return None


def _solver_candidates() -> list[str]:
    """Interpreters worth trying, best first.

    This interpreter comes first (nothing to install). Then a plain
    ``pythonX.Y`` matching a build's tag, which is usually the one that already
    has this project installed. The build's own venv is last: it can load
    duneuropy, but it often carries its own copy of ``inob``.
    """
    import shutil

    out: list[str] = [sys.executable]
    try:
        found = duneuro_setup.discover(_active_duneuro_path())
        candidates = found.get("candidates", [])  # type: ignore[union-attr]
    except Exception as e:
        logger.warning("could not enumerate duneuro builds: %s", e)
        return out

    tail: list[str] = []
    for cand in candidates:
        tag = str(cand.get("python_tag") or "")
        if tag.startswith("python3"):
            on_path = shutil.which(tag)
            if on_path and on_path not in out:
                out.append(on_path)
        venv_py = _venv_python_for(str(cand["path"]))
        if venv_py and venv_py not in tail:
            tail.append(venv_py)
    return out + tail


def _pipeline_python() -> str:
    """The interpreter a pipeline child should run under."""
    explicit = os.environ.get("INOB_PIPELINE_PYTHON")
    if explicit:
        return explicit
    cached = _PIPELINE_PY.get("path")
    if cached:
        return cached

    chosen = sys.executable
    for candidate in _solver_candidates():
        if _can_solve(candidate):
            chosen = candidate
            break
    if chosen != sys.executable:
        logger.info("pipeline will run under %s (this backend cannot solve here)",
                    chosen)

    _PIPELINE_PY["path"] = chosen
    return chosen


# ── config ────────────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Return the active config as a raw dict plus its validation status."""
    path = _active_config_path()
    raw, read_errors = _read_raw_safe(path)
    errors = read_errors or _validate_raw(raw)
    return {"source": str(path), "config": raw, "errors": errors}


@app.post("/api/config/validate")
def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a candidate config without persisting it."""
    raw = payload.get("config", payload)
    errors = _validate_raw(raw)
    return {"valid": not errors, "errors": errors}


@app.put("/api/config")
def put_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist a config to the working copy."""
    raw = payload.get("config", payload)
    errors = _validate_raw(raw)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    WORKING_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with WORKING_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return {"saved": str(WORKING_CONFIG), "valid": True, "errors": []}


@app.post("/api/config/reset")
def reset_config() -> dict[str, Any]:
    """Discard the working copy, reverting to configs/default.yaml."""
    WORKING_CONFIG.unlink(missing_ok=True)
    return get_config()


# ── DUNEuro solver engine setup ────────────────────────────────────────────────
#
# The forward solve needs the compiled ``duneuropy`` extension. This lets the
# user point the solver at a local build from the GUI and see, honestly, whether
# this backend can actually load it (a build is tied to one Python version).

def _active_duneuro_path() -> str | None:
    raw, _ = _read_raw_safe(_active_config_path())
    value = (raw.get("forward") or {}).get("duneuro_path")
    return str(value) if value else None


# ── forward-model ladder (Biot–Savart → Sarvas → FEM) ──────────────────────

@app.post("/api/ladder")
def run_ladder_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Run selected rungs of the analytic ladder for the current config.

    Same principle as the pipeline run — it loads the working config and calls
    the same ``run_ladder`` the CLI uses. The ladder is the *validation* path:
    it benchmarks the configured vagus source polyline (picked by
    ``source_idx``), not the clicked planning sources. The analytic rungs need
    no DUNEuro; the fem rung needs the leadfield and 409s cleanly without it.
    """
    from inob.analysis.sarvas_compare import LADDER_RUNGS, run_ladder

    rungs = payload.get("rungs") or list(LADDER_RUNGS)
    rungs = [r for r in rungs if r in LADDER_RUNGS]
    if not rungs:
        raise HTTPException(status_code=422,
                            detail={"errors": ["no valid rungs requested"]})

    try:
        cfg = load_config(_active_config_path(), project_root=PROJECT_ROOT)
    except ConfigError as e:
        raise HTTPException(status_code=422, detail={"errors": [str(e)]}) from e

    try:
        return run_ladder(
            cfg, rungs=rungs,
            Q_nAm=float(payload.get("Q_nAm", 1.0)),
            source_idx=int(payload.get("source_idx", -1)),
        )
    except FileNotFoundError as e:
        # Almost always: fem rung requested but geometry/mesh/leadfield missing.
        raise HTTPException(
            status_code=409,
            detail={"errors": [str(e)],
                    "hint": "Build the model first (Run), or drop the fem rung "
                            "to compare only the analytic rungs."},
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"errors": [str(e)]}) from e


@app.get("/api/duneuro")
def duneuro_status() -> dict[str, Any]:
    """Report the running interpreter and any local DUNEuro builds found."""
    return duneuro_setup.discover(_active_duneuro_path())


@app.post("/api/duneuro")
def set_duneuro_path(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist ``forward.duneuro_path`` into the working config (or clear it).

    Same principle as every other GUI edit: it writes the YAML the pipeline
    reads. An empty/null path means "use whatever duneuropy the interpreter
    already has".
    """
    path = (payload or {}).get("path")
    path = str(path).strip() if path else None

    raw, errors = _read_raw_safe(_active_config_path())
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    raw.setdefault("forward", {})["duneuro_path"] = path
    validation = _validate_raw(raw)
    if validation:
        raise HTTPException(status_code=422, detail={"errors": validation})
    WORKING_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with WORKING_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return duneuro_setup.discover(path)


# ── meshes (for the 3-D viewer) ────────────────────────────────────────────────
#
# The viewer shows the whole multi-tissue anatomy, one toggle per tissue. Each
# tissue category may be backed by many source STLs (74 bones, 12 muscles, …);
# they are merged into a single binary STL and cached on disk so the browser
# fetches one file per tissue instead of dozens. The cache key includes source
# mtimes so an edited/added STL invalidates it automatically.

# Cache of merged per-tissue STLs (survives reloads; keyed by content mtimes).
_MESH_CACHE = Path(tempfile.gettempdir()) / "inob_gui_meshes"

# Viewer render order / default visibility. Skin and bone are large and occlude
# the interior structures the user actually plans around, so they load hidden.
_TISSUE_ORDER = [
    "skin", "bone", "muscle", "blood_vessel",
    "spinal_cord", "vagus_left", "vagus_right",
]
_HIDDEN_BY_DEFAULT = {"skin", "bone"}


def _abs_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _safe_stl(p: Path) -> Path | None:
    """Only serve existing ``.stl`` files inside the project root — a config
    path like ``data.torso_skin: /etc/passwd`` must not become readable."""
    try:
        rp = p.resolve()
    except OSError:
        return None
    if rp.suffix.lower() != ".stl":
        return None
    if not rp.is_relative_to(PROJECT_ROOT.resolve()):
        return None
    return rp if rp.is_file() else None


def _tissue_sources() -> dict[str, list[Path]]:
    """Map each tissue category → the list of source STL paths from the config.

    Single-file tissues (skin) yield one path; grouped tissues (bone, muscle,
    vessel, spinal cord) yield every ``*.stl`` in their directory; vagus
    left/right yield every glob match. Only project-local ``.stl`` files pass.
    """
    raw, _ = _read_raw_safe(_active_config_path())
    data = raw.get("data", {})
    out: dict[str, list[Path]] = {}

    def _add(name: str, paths: list[Path]) -> None:
        safe = [s for p in paths if (s := _safe_stl(p))]
        if safe:
            out[name] = safe

    skin = data.get("torso_skin")
    if skin:
        _add("skin", [_abs_path(skin)])

    for name, key in (("bone", "bone_dir"), ("muscle", "muscle_dir"),
                      ("blood_vessel", "vessel_dir"),
                      ("spinal_cord", "spinal_cord_dir")):
        d = data.get(key)
        if d:
            _add(name, sorted(_abs_path(d).glob("*.stl")))

    for name, key in (("vagus_left", "vagus_left_glob"),
                      ("vagus_right", "vagus_right_glob")):
        pat = data.get(key)
        if pat:
            _add(name, [Path(m) for m in sorted(glob.glob(str(_abs_path(pat))))])

    # Preserve the documented render order, dropping tissues with no meshes.
    return {t: out[t] for t in _TISSUE_ORDER if t in out}


def _merged_stl(name: str, paths: list[Path]) -> Path | None:
    """Return a cached single binary STL merging ``paths``; build it if stale.

    The cache filename embeds a hash of the source paths and their mtimes, so
    any edit/add/remove of a source STL produces a fresh merge automatically.
    Returns ``None`` (never raises) if the merge fails, so one broken tissue
    cannot 500 the whole ``/api/meshes`` payload.
    """
    import hashlib

    sig = "|".join(f"{p}:{p.stat().st_mtime_ns}" for p in paths)
    digest = hashlib.sha1(sig.encode()).hexdigest()[:16]
    _MESH_CACHE.mkdir(parents=True, exist_ok=True)
    cached = _MESH_CACHE / f"{name}-{digest}.stl"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    try:
        from inob.io.stl import concat_stls
        mesh = concat_stls(list(paths), check_units_mm=False)
        mesh.export(cached, file_type="stl")  # binary STL
        return cached
    except Exception as e:
        logger.warning("mesh merge failed for %s (%d files): %s", name, len(paths), e)
        return None


# In-process memo of the last built index, keyed by the config's mtime and the
# tissue source signature. Serving one mesh used to rebuild (and re-stat) the
# merge for all seven tissues, so a viewer loading the anatomy did that work
# once per tissue fetched.
_MESH_INDEX_MEMO: dict[str, Any] = {"key": None, "index": {}}


def _mesh_index() -> dict[str, tuple[Path, int]]:
    """Build (and cache-back) the merged STL for every tissue category.

    Returns ``name -> (merged_path, n_source_parts)``.
    """
    sources = _tissue_sources()
    # Same key discipline as the on-disk merge cache: paths plus mtimes, so an
    # edited/added/removed STL still invalidates immediately.
    try:
        key = "|".join(
            f"{n}:{p}:{p.stat().st_mtime_ns}" for n, ps in sources.items() for p in ps
        )
    except OSError:
        key = None          # a source vanished mid-scan; rebuild rather than memo
    if key is not None and _MESH_INDEX_MEMO["key"] == key:
        return _MESH_INDEX_MEMO["index"]

    out: dict[str, tuple[Path, int]] = {}
    for name, paths in sources.items():
        merged = _merged_stl(name, paths)
        if merged:
            out[name] = (merged, len(paths))
    if key is not None:
        _MESH_INDEX_MEMO["key"] = key
        _MESH_INDEX_MEMO["index"] = out
    return out


@app.get("/api/meshes")
def list_meshes() -> dict[str, Any]:
    index = _mesh_index()
    return {
        "meshes": [
            {
                "name": name,
                "url": f"/api/meshes/{name}",
                "bytes": path.stat().st_size,
                "parts": parts,
                "default_visible": name not in _HIDDEN_BY_DEFAULT,
            }
            for name, (path, parts) in index.items()
        ]
    }


@app.get("/api/meshes/{name}")
def get_mesh(name: str) -> FileResponse:
    index = _mesh_index()
    if name not in index:
        raise HTTPException(status_code=404, detail=f"no mesh named {name!r}")
    return FileResponse(index[name][0], media_type="model/stl", filename=f"{name}.stl")


# ── figures ─────────────────────────────────────────────────────────────────
#
# Every stage that draws something writes a PNG somewhere under the config's
# output paths. The GUI shows those as nodes on the journey canvas, so it needs
# (a) a list of what a run can produce — including figures that do not exist
# yet, which is how the canvas can offer them — and (b) a way to fetch the
# bytes. Files are only ever served out of the index built here, so a key from
# the client can never address a path of its own choosing.

# Config keys under ``outputs:`` that name a figure, with the stage that draws
# it and the words a planner would use for it.
_FIGURE_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("geometry_png", "geom", "Geometry", "Tissue surfaces after shrinkwrap and repair"),
    ("fem_png", "fem", "FEM mesh", "Multi-tissue tetrahedral mesh, cut through"),
    ("sensors_png", "sensors", "Sensor array", "OPM positions and orientations on the body"),
    ("electrodes_png", "sensors", "Electrode array", "EEG electrode positions, when solved"),
)

# Directories under ``outputs:`` whose PNGs are all figures of one stage.
_FIGURE_DIRS: tuple[tuple[str, str, str], ...] = (
    ("sensitivity_dir", "forward", "Sensitivity"),
)

_FIGURE_SCAN_LIMIT = 60


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _figure_index() -> "OrderedDict[str, dict[str, Any]]":
    """Map figure key → metadata, in journey order (declared first, found after).

    Built fresh per request: figures appear mid-run, and the whole point of the
    canvas is that a node lights up the moment its PNG lands.
    """
    raw, _ = _read_raw_safe(_active_config_path())
    outputs = raw.get("outputs") or {}
    index: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add(key: str, path: Path, stage: str, label: str, blurb: str = "") -> None:
        if key in index:
            return
        try:
            st = path.stat()
            exists, size, mtime = True, st.st_size, st.st_mtime
        except OSError:
            exists, size, mtime = False, 0, 0.0
        index[key] = {
            "key": key,
            "label": label,
            "blurb": blurb,
            "stage": stage,
            "path": path,
            "url": f"/api/figures/{key}",
            "exists": exists,
            "bytes": size,
            "mtime": mtime,
        }

    for key, stage, label, blurb in _FIGURE_KEYS:
        value = outputs.get(key)
        if value:
            add(key, Path(str(value)), stage, label, blurb)

    for dir_key, stage, label in _FIGURE_DIRS:
        value = outputs.get(dir_key)
        if not value:
            continue
        directory = Path(str(value))
        if not directory.is_dir():
            continue
        for png in sorted(directory.glob("*.png"))[:_FIGURE_SCAN_LIMIT]:
            add(_slug(f"{dir_key}_{png.stem}"), png, stage,
                f"{label} — {png.stem.replace('_', ' ')}")

    # Anything else a run dropped under the output base: viz figures the config
    # does not name individually (topoplots, comparisons, physiology panels).
    base = outputs.get("base")
    if base:
        base_path = Path(str(base))
        if base_path.is_dir():
            for png in sorted(base_path.rglob("*.png"))[:_FIGURE_SCAN_LIMIT]:
                rel = png.relative_to(base_path)
                add(_slug(str(rel.with_suffix(""))), png, "viz",
                    png.stem.replace("_", " "))

    return index


@app.get("/api/figures")
def list_figures() -> dict[str, Any]:
    return {
        "figures": [
            {k: v for k, v in meta.items() if k != "path"}
            for meta in _figure_index().values()
        ]
    }


@app.get("/api/figures/{key}")
def get_figure(key: str) -> FileResponse:
    meta = _figure_index().get(key)
    if meta is None or not meta["exists"]:
        raise HTTPException(status_code=404, detail=f"no figure named {key!r}")
    return FileResponse(meta["path"], media_type="image/png")


# ── cluster submission ──────────────────────────────────────────────────────────

@app.get("/api/cluster/profiles")
def cluster_profiles() -> dict[str, Any]:
    return {"profiles": cluster.list_profiles()}


@app.post("/api/cluster/submit")
def cluster_submit(payload: dict[str, Any]) -> dict[str, Any]:
    # `stages` / `threshold_snr` are intentionally not forwarded: the cluster
    # only runs the forward solve, and both are local post-processing applied
    # after the leadfield is fetched back. See cluster.submit.
    try:
        return cluster.submit(
            payload.get("profile"),
            sources=payload.get("sources"),
            modality=payload.get("modality", "meg"),
        )
    except cluster.ClusterError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/api/cluster/status")
def cluster_status(profile: str, job_id: str) -> dict[str, Any]:
    try:
        return cluster.status(profile, job_id)
    except cluster.ClusterError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/cluster/fetch")
def cluster_fetch(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return cluster.fetch(
            payload.get("profile"), payload.get("job_id"),
            modality=payload.get("modality", "meg"),
        )
    except cluster.ClusterError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# ── run (streaming logs over WebSocket) ─────────────────────────────────────────

class _QueueLogHandler(logging.Handler):
    """A logging handler that pushes formatted records onto a thread-safe queue."""

    def __init__(self, q: queue.Queue[str]) -> None:
        super().__init__()
        self._q = q
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put_nowait(self.format(record))
        except Exception:
            pass


@app.websocket("/api/run")
async def run_ws(ws: WebSocket) -> None:
    """Run pipeline stages and stream every ``inob`` log line to the client.

    Client sends ``{"stages": ["geom", ...] | "all", "force": bool}``.
    Server streams ``{"type": "log"|"status"|"done"|"error", ...}`` messages.
    """
    await ws.accept()
    try:
        req = await ws.receive_json()
    except WebSocketDisconnect:
        return

    # Single-flight: refuse a second concurrent run rather than race on the
    # shared output artefacts / logger.
    if not _RUN_LOCK.acquire(blocking=False):
        await ws.send_json({"type": "error",
                            "message": "a pipeline run is already in progress"})
        await ws.close()
        return

    stages_in = req.get("stages", "all")
    force = bool(req.get("force", False))
    sources = req.get("sources") or []
    threshold_snr = float(req.get("threshold_snr", 3.0))
    modality = str(req.get("modality", "meg")).lower()

    # Clicked sources reach the solver as a config override. They only change
    # the dipole positions, so geom/fem/sensors do not need rebuilding — but
    # they DO need to exist. Restricting the run to ["forward"] outright (what
    # this used to do) meant that on a machine without those artefacts already
    # on disk the run died with a bare "sensor file not found" instead of just
    # building them. So: run every stage, but only *force* the forward solve —
    # the upstream stages skip themselves when their outputs are already there.
    overrides: list[str] = []
    strengths: list[float] = []
    resolve_sources = False
    if sources:
        positions = [[float(s["x"]), float(s["y"]), float(s["z"])] for s in sources]
        strengths = [float(s.get("strength_nAm", 70.0)) for s in sources]
        overrides.append(f"forward.point_sources={json.dumps(positions)}")
        resolve_sources = True

    log_q: queue.Queue[str] = queue.Queue()
    handler = _QueueLogHandler(log_q)
    vagus_logger = logging.getLogger("inob")
    prev_level = vagus_logger.level
    vagus_logger.addHandler(handler)
    vagus_logger.setLevel(logging.INFO)

    result: dict[str, Any] = {}
    error: dict[str, Any] = {}
    done = threading.Event()
    cancel = threading.Event()
    # The running child, so a cancel can stop it rather than only asking the
    # next stage boundary not to start.
    child: dict[str, Any] = {"proc": None}

    # A run happens in a child process, not on a thread here.
    #
    # The viz stage draws with PyVista, which builds a VTK render window; on
    # macOS that is Cocoa, and Cocoa raises an uncatchable NSException when it
    # is touched from any thread but the main one. Running the pipeline on a
    # worker thread therefore killed the whole backend process partway through
    # "viz" — the API server included — while the identical CLI run succeeded.
    # A child process gets its own main thread, so the same code that works in
    # the CLI works here, and a crash costs us the run rather than the server.
    def _pipeline(stages: list[str], force_run: bool) -> dict[str, str]:
        """Run one pipeline invocation as a child, streaming its log lines.

        Returns the per-stage statuses parsed from the markers the CLI already
        prints ([run]/[ok]/[skip]/[FAIL]), so the client sees exactly what the
        CLI reports.
        """
        argv = [
            _pipeline_python(), "-u", "-m", "inob.cli.pipeline",
            "--config", str(_active_config_path()),
            "--project-root", str(PROJECT_ROOT),
            "--stages", ",".join(stages),
        ]
        if force_run:
            argv.append("--force")
        for override in overrides:
            argv += ["--set", override]

        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"          # matplotlib stays off any GUI path
        # This project's src first, so the child can never pick up a stale
        # editable install of inob from elsewhere on the machine.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")])
        )

        statuses: dict[str, str] = {}
        # Its own process group: the forward stage fans out into a pool of
        # chunk workers, and terminating only the parent orphaned twelve of
        # them to keep burning every core with nothing to report to. Killing
        # the group takes the whole solve down together.
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        child["proc"] = proc
        _ACTIVE_CHILD["proc"] = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            log_q.put(line)
            m = re.search(r"\[(ok|skip|FAIL)]\s+(\w+)", line)
            if m:
                statuses[m.group(2)] = {
                    "ok": "ran", "skip": "skipped", "FAIL": "failed",
                }[m.group(1)]
        code = proc.wait()
        child["proc"] = None
        _ACTIVE_CHILD["proc"] = None
        if cancel.is_set():
            for name in stages:
                statuses.setdefault(name, "cancelled")
            return statuses
        if code != 0:
            failed = [k for k, v in statuses.items() if v == "failed"]
            raise RuntimeError(
                f"the pipeline exited with code {code}"
                + (f" while running {failed[0]!r}" if failed else "")
                + " — see the run log for the traceback"
            )
        return statuses

    def _work() -> None:
        try:
            if stages_in == "all" or not stages_in:
                stages = list(ALL_STAGES)
            else:
                stages = [s for s in stages_in if s in ALL_STAGES]

            if resolve_sources:
                # Two passes so "reuse the model, re-solve the sources" is
                # expressible with the CLI's single force flag: build any
                # missing upstream stages (skipped when already present), then
                # force the forward solve for the new dipole positions.
                upstream = [s for s in stages if s != "forward"]
                statuses: dict[str, str] = {}
                if upstream:
                    statuses.update(_pipeline(upstream, False))
                if not cancel.is_set() and "forward" in stages:
                    statuses.update(_pipeline(["forward"], True))
            else:
                statuses = _pipeline(stages, force)

            result["statuses"] = statuses
            if cancel.is_set():
                result["cancelled"] = True
                return

            # Turn the solved leadfield into the planning answer. This is numpy
            # and h5py only — no rendering — so it is safe on this thread.
            cfg = load_config(_active_config_path(), overrides=overrides,
                              project_root=PROJECT_ROOT)
            try:
                result["detect"] = compute_detectability(
                    cfg, strengths_nAm=strengths or None,
                    threshold_snr=threshold_snr, modality=modality,
                )
            except FileNotFoundError as e:
                logger.info("detectability skipped (no leadfield yet): %s", e)
                result["detect_unavailable"] = {
                    "reason": "No leadfield has been computed yet, so "
                              "trials-to-detect cannot be calculated.",
                    "detail": str(e),
                    "hint": "Run the full pipeline including the forward "
                            "solve. That stage needs DUNEuro (duneuropy) "
                            "installed, or submit it to the cluster.",
                }
        except Exception as e:  # surfaced to the client, not swallowed
            error["message"] = f"{type(e).__name__}: {e}"
        finally:
            # Detach the log handler and drop the single-flight lock HERE, on
            # the worker itself, rather than in the socket handler's `finally`.
            # The pipeline outlives the socket (a browser disconnect does not
            # stop it), so releasing on disconnect let a second run start while
            # the first was still writing the same outputs/ artefacts.
            vagus_logger.removeHandler(handler)
            vagus_logger.setLevel(prev_level)
            done.set()
            _RUN_LOCK.release()

    worker = threading.Thread(target=_work, name="inob-run", daemon=True)
    try:
        worker.start()
    except BaseException:
        vagus_logger.removeHandler(handler)
        vagus_logger.setLevel(prev_level)
        _RUN_LOCK.release()
        raise

    async def _watch_for_cancel() -> None:
        """A client message during a run is a cancel request.

        The pipeline runs as a child process, so cancelling terminates it and
        the run stops within a stage rather than only at the next boundary.
        Whatever a stage had already written to disk stays there.
        """
        try:
            while not done.is_set():
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "cancel":
                    cancel.set()
                    # The work is a child process now, so a cancel can really
                    # stop it: ask it to terminate rather than waiting out a
                    # solve the user has already abandoned.
                    _stop_child(child.get("proc"))
                    await ws.send_json({
                        "type": "log",
                        "line": "— cancel requested; stopping the run —",
                    })
        except (WebSocketDisconnect, RuntimeError, ValueError):
            # Client vanished or sent junk. The run continues (it holds the
            # lock and owns the artefacts); we simply stop listening.
            pass

    watcher = asyncio.create_task(_watch_for_cancel())
    try:
        # Drain the log queue to the socket until the worker finishes.
        while not (done.is_set() and log_q.empty()):
            try:
                line = log_q.get_nowait()
                await ws.send_json({"type": "log", "line": line})
            except queue.Empty:
                await asyncio.sleep(0.1)
        if error:
            await ws.send_json({"type": "error", **error})
        elif result.get("cancelled"):
            await ws.send_json({
                "type": "done", "cancelled": True,
                "statuses": result.get("statuses", {}),
            })
        else:
            if "detect" in result:
                await ws.send_json({"type": "result", "detect": result["detect"]})
            await ws.send_json({
                "type": "done",
                "statuses": result.get("statuses", {}),
                "detect_unavailable": result.get("detect_unavailable"),
            })
    except WebSocketDisconnect:
        pass
    finally:
        watcher.cancel()
        try:
            await ws.close()
        except Exception:
            pass
