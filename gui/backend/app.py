"""FastAPI backend for the Vagus-FM GUI.

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

The frontend dev server (Vite, port 5173) proxies ``/api`` here.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from gui.backend import cluster
from inob.analysis.detect import compute_detectability
from inob.cli.pipeline import ALL_STAGES, run_pipeline
from inob.config import ConfigError, load_config

logger = logging.getLogger(__name__)

# Single-flight guard: a DUNEuro solve writes shared cfg.outputs artefacts, so
# only one run may be in flight at a time (two concurrent runs would race on the
# same files and the shared "inob" logger). Acquired non-blocking by /api/run.
_RUN_LOCK = threading.Lock()

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
        import os
        os.close(fd)
        tmp_path.unlink(missing_ok=True)


# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Vagus-FM GUI backend", version="0.1.0")

# In dev the React app runs on a different origin (Vite :5173); allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # legacy Vite frontend
        "http://localhost:3000", "http://127.0.0.1:3000",  # Next.js frontend (gui/web)
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "active_config": str(_active_config_path()),
        "stages": list(ALL_STAGES),
    }


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


# ── meshes (for the 3-D viewer) ────────────────────────────────────────────────

def _resolve_mesh_paths() -> dict[str, Path]:
    """Map a viewer mesh name → an STL path on disk, from the active config.

    Skin is a single file; vagus left/right are matched by glob (first hit).
    Bones are 77 separate STLs — too heavy to stream individually here, so
    they are intentionally excluded from the default viewer payload.
    """
    raw, _ = _read_raw_safe(_active_config_path())
    data = raw.get("data", {})
    out: dict[str, Path] = {}

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _safe(p: Path) -> Path | None:
        """Only serve existing .stl files inside the project root — a config
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

    skin = data.get("torso_skin")
    if skin:
        safe = _safe(_abs(skin))
        if safe:
            out["skin"] = safe
    for name, key in (("vagus_left", "vagus_left_glob"),
                      ("vagus_right", "vagus_right_glob")):
        pat = data.get(key)
        if not pat:
            continue
        for match in sorted(glob.glob(str(_abs(pat)))):
            safe = _safe(Path(match))
            if safe:
                out[name] = safe
                break
    return out


@app.get("/api/meshes")
def list_meshes() -> dict[str, Any]:
    meshes = _resolve_mesh_paths()
    return {
        "meshes": [
            {"name": n, "url": f"/api/meshes/{n}", "bytes": p.stat().st_size}
            for n, p in meshes.items()
        ]
    }


@app.get("/api/meshes/{name}")
def get_mesh(name: str) -> FileResponse:
    meshes = _resolve_mesh_paths()
    if name not in meshes:
        raise HTTPException(status_code=404, detail=f"no mesh named {name!r}")
    return FileResponse(meshes[name], media_type="model/stl", filename=f"{name}.stl")


# ── cluster submission ──────────────────────────────────────────────────────────

@app.get("/api/cluster/profiles")
def cluster_profiles() -> dict[str, Any]:
    return {"profiles": cluster.list_profiles()}


@app.post("/api/cluster/submit")
def cluster_submit(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return cluster.submit(
            payload.get("profile"),
            stages=payload.get("stages"),
            sources=payload.get("sources"),
            threshold_snr=float(payload.get("threshold_snr", 3.0)),
            modality=str(payload.get("modality", "meg")),
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
        return cluster.fetch(payload.get("profile"), payload.get("job_id"))
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
    # the dipole positions, so re-running geom/fem/sensors is both wasteful and
    # would rebuild the (cached) geometry unnecessarily — restrict the run to
    # the forward solve and force only that.
    overrides: list[str] = []
    strengths: list[float] = []
    sources_only_forward = False
    if sources:
        positions = [[float(s["x"]), float(s["y"]), float(s["z"])] for s in sources]
        strengths = [float(s.get("strength_nAm", 70.0)) for s in sources]
        overrides.append(f"forward.point_sources={json.dumps(positions)}")
        force = True
        sources_only_forward = True

    log_q: queue.Queue[str] = queue.Queue()
    handler = _QueueLogHandler(log_q)
    vagus_logger = logging.getLogger("inob")
    prev_level = vagus_logger.level
    vagus_logger.addHandler(handler)
    vagus_logger.setLevel(logging.INFO)

    result: dict[str, Any] = {}
    error: dict[str, Any] = {}
    done = threading.Event()

    def _work() -> None:
        try:
            cfg = load_config(_active_config_path(), overrides=overrides,
                              project_root=PROJECT_ROOT)
            if sources_only_forward:
                stages = ["forward"]      # explicit sources → re-solve only
            elif stages_in == "all" or not stages_in:
                stages = list(ALL_STAGES)
            else:
                stages = [s for s in stages_in if s in ALL_STAGES]
            statuses = run_pipeline(cfg, stages=stages, force=force)
            result["statuses"] = statuses
            # Turn the solved leadfield into the planning answer. Best-effort:
            # if the required leadfield was not produced, the client falls back
            # to its own estimate (see API_CONTRACT.md).
            try:
                result["detect"] = compute_detectability(
                    cfg, strengths_nAm=strengths or None,
                    threshold_snr=threshold_snr, modality=modality,
                )
            except FileNotFoundError as e:
                logger.info("detectability skipped (no leadfield yet): %s", e)
        except Exception as e:  # surfaced to the client, not swallowed
            error["message"] = f"{type(e).__name__}: {e}"
        finally:
            done.set()

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()

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
        else:
            if "detect" in result:
                await ws.send_json({"type": "result", "detect": result["detect"]})
            await ws.send_json({"type": "done", "statuses": result.get("statuses", {})})
    except WebSocketDisconnect:
        pass
    finally:
        vagus_logger.removeHandler(handler)
        vagus_logger.setLevel(prev_level)
        _RUN_LOCK.release()
        try:
            await ws.close()
        except Exception:
            pass
