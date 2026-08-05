// Typed client for the iNOB backend. HTTP calls go same-origin to /api/* and
// are proxied to FastAPI by Next rewrites (see next.config.mjs). The pipeline
// run is a WebSocket connected directly to the backend.

import type { Cfg } from "./config";

// The run socket connects straight to FastAPI (Next rewrites proxy HTTP only).
// Derive it from wherever the page is actually served so the GUI keeps working
// when it isn't on localhost — a hardcoded 127.0.0.1 meant HTTP went through
// the rewrite and the WebSocket silently pointed at the viewer's own machine.
const DEFAULT_BACKEND_PORT = "8000";

function defaultWsBase(): string {
  if (typeof window === "undefined") return `ws://127.0.0.1:${DEFAULT_BACKEND_PORT}`;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.hostname}:${DEFAULT_BACKEND_PORT}`;
}

const WS_BASE = process.env.NEXT_PUBLIC_INOB_WS || defaultWsBase();

export interface MeshInfo {
  name: string;
  url: string;
  bytes: number;
  parts?: number;
  default_visible?: boolean;
}

export interface PointSource {
  x: number;
  y: number;
  z: number;
  strength_nAm: number;
}

export interface PerSourceDetect {
  index: number;
  x: number;
  y: number;
  z: number;
  strength_nAm: number;
  snr: number;
  trials_needed: number;
}

export interface DetectResult {
  modality: string;
  per_source: PerSourceDetect[];
  array: {
    n_sensors: number;
    mean_snr: number;
    max_snr: number;
    noise_floor_fT: number;
    noise_unit?: string;
    threshold_snr: number;
  };
}

// Sent instead of a result when the backend could not compute detectability —
// almost always because no leadfield exists yet. The UI shows this verbatim
// rather than substituting an estimate of its own.
export interface DetectUnavailable {
  reason: string;
  detail?: string;
  hint?: string;
}

export async function getHealth(): Promise<{ status: string } | null> {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

// What the local machine can bring to a solve. Drives the compute-resources
// chooser, so the user picks from their real core count rather than guessing.
export interface SystemInfo {
  platform: string;
  machine: string;
  cpu_logical: number;
  cpu_physical: number;
  cpu_performance: number | null;
  configured_workers: number | null;
  all_cores: number;
}

export async function getSystem(): Promise<SystemInfo | null> {
  try {
    const r = await fetch("/api/system", { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function getConfig(): Promise<{ config: Cfg; errors: string[] }> {
  const r = await fetch("/api/config", { cache: "no-store" });
  if (!r.ok) throw new Error(`GET /api/config failed: ${r.status}`);
  return r.json();
}

export async function putConfig(
  config: Cfg,
): Promise<{ ok: boolean; errors: string[] }> {
  const r = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  const body = await r.json().catch(() => ({}));
  // FastAPI wraps a rejected payload's detail, so the real validation
  // messages arrive as body.detail.errors. Reading only body.errors turned
  // every invalid config into an opaque "PUT failed: 422".
  const errors: string[] =
    body.detail?.errors ?? body.errors ?? (r.ok ? [] : [`PUT failed: ${r.status}`]);
  return { ok: r.ok, errors };
}

export async function validateConfig(config: Cfg): Promise<string[]> {
  const r = await fetch("/api/config/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  const body = await r.json().catch(() => ({}));
  return body.detail?.errors ?? body.errors ?? [];
}

export async function getMeshes(): Promise<MeshInfo[]> {
  const r = await fetch("/api/meshes", { cache: "no-store" });
  if (!r.ok) return [];
  const body = await r.json();
  return body.meshes ?? body ?? [];
}

// A figure a stage draws. `exists` is false for figures a run can produce but
// hasn't yet — the canvas offers those as nodes you can add ahead of the run.
export interface FigureInfo {
  key: string;
  label: string;
  blurb?: string;
  stage: string;
  url: string;
  exists: boolean;
  bytes: number;
  mtime: number;
}

export async function getFigures(): Promise<FigureInfo[]> {
  try {
    const r = await fetch("/api/figures", { cache: "no-store" });
    if (!r.ok) return [];
    const body = await r.json();
    return body.figures ?? [];
  } catch {
    return [];
  }
}

// What the next run will solve with. The backend searches the machine for a
// DUNEuro build and pairs it with an interpreter that can load it, so this
// reports a decision already made rather than asking anyone to make one.
export interface SolverInfo {
  python: string;
  python_version: string;
  duneuro_path: string | null;
  found: boolean;
  searched: string[];
}

export async function getSolver(): Promise<SolverInfo | null> {
  try {
    const r = await fetch("/api/solver", { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function rescanSolver(): Promise<SolverInfo | null> {
  try {
    const r = await fetch("/api/solver/rescan", { method: "POST" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export interface RunHandlers {
  onLog: (line: string) => void;
  onStatuses?: (statuses: Record<string, string>) => void;
  onResult?: (detect: DetectResult) => void;
  onDone: (
    statuses: Record<string, string>,
    unavailable?: DetectUnavailable,
    cancelled?: boolean,
  ) => void;
  onError: (message: string) => void;
}

// A live run. `cancel()` asks the backend to stop at the next stage boundary
// (a solve already in flight cannot be interrupted); `close()` just detaches
// this client and leaves the run going.
export interface RunHandle {
  cancel: () => void;
  close: () => void;
}

// Opens the run WebSocket. Streams {type: log|result|done|error}.
// `extra` (sources, threshold_snr) is merged into the init payload.
export function runSimulation(
  stages: string[],
  force: boolean,
  handlers: RunHandlers,
  extra: Record<string, unknown> = {},
): RunHandle {
  const ws = new WebSocket(`${WS_BASE}/api/run`);
  let lastStatuses: Record<string, string> = {};
  // The run is over exactly once — whether by `done`, by `error`, or because
  // the socket dropped. Without this latch a backend crash mid-run produced no
  // terminal message at all and the UI span forever on "Solving…".
  let settled = false;
  const settle = (fn: () => void) => {
    if (settled) return;
    settled = true;
    fn();
  };

  ws.onopen = () => ws.send(JSON.stringify({ stages, force, ...extra }));
  ws.onmessage = (ev) => {
    let msg: { type: string; [k: string]: unknown };
    try {
      msg = JSON.parse(ev.data as string);
    } catch {
      handlers.onLog(String(ev.data));
      return;
    }
    switch (msg.type) {
      case "log":
        handlers.onLog(String(msg.line ?? ""));
        break;
      case "result":
        if (msg.detect) handlers.onResult?.(msg.detect as DetectResult);
        break;
      case "done":
        lastStatuses = (msg.statuses as Record<string, string>) ?? lastStatuses;
        handlers.onStatuses?.(lastStatuses);
        settle(() =>
          handlers.onDone(
            lastStatuses,
            (msg.detect_unavailable as DetectUnavailable | null) ?? undefined,
            Boolean(msg.cancelled),
          ),
        );
        ws.close();
        break;
      case "error":
        settle(() => handlers.onError(String(msg.message ?? "unknown error")));
        ws.close();
        break;
    }
  };
  ws.onerror = () =>
    settle(() =>
      handlers.onError(
        `WebSocket error — is the backend running at ${WS_BASE}?`,
      ),
    );
  // A close without a prior done/error means the backend went away mid-run.
  // Report it rather than leaving the caller stuck in a "running" state.
  ws.onclose = (ev) =>
    settle(() =>
      handlers.onError(
        `Connection to the backend closed before the run finished` +
          `${ev.reason ? ` (${ev.reason})` : ""}. The run may still be going ` +
          `on the server — check the backend log.`,
      ),
    );

  return {
    cancel: () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "cancel" }));
      }
    },
    // Detach without reporting an error: used on unmount, where the caller is
    // going away and does not want a spurious failure message.
    close: () => {
      settled = true;
      ws.close();
    },
  };
}

// ── forward-model ladder ────────────────────────────────────────────────────

export type LadderRung = "biot" | "sarvas" | "fem";

export interface LadderResult {
  Q_nAm: number;
  source_index: number;
  n_sources: number;
  source_pos_mm: number[];
  n_radial_coils: number;
  rungs: Record<
    string,
    { label: string; peak_fT_per_nAm: number; rms_fT_per_nAm: number }
  >;
  ratios: Record<string, number>;
}

// Runs the analytic ladder on the configured vagus polyline. Returns {result}
// on success, or {error, hint} when the backend can't (e.g. the fem rung with
// no leadfield yet).
export async function runLadder(
  rungs: LadderRung[],
): Promise<{ result?: LadderResult; error?: string; hint?: string }> {
  try {
    const r = await fetch("/api/ladder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rungs }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      const detail = body.detail ?? body;
      return {
        error: (detail.errors ?? [`request failed: ${r.status}`]).join("; "),
        hint: detail.hint,
      };
    }
    return { result: body as LadderResult };
  } catch (e) {
    return { error: `could not reach the backend: ${String(e)}` };
  }
}

// ── DUNEuro solver engine setup ────────────────────────────────────────────

export interface DuneuroCandidate {
  path: string;
  module: string;
  label: string;
  python_tag: string;
  importable: boolean;
  note: string;
}

export interface DuneuroStatus {
  running: {
    executable: string;
    python: string;
    duneuropy_importable: boolean;
    duneuropy_location: string | null;
  };
  candidates: DuneuroCandidate[];
  active_path: string | null;
  hint: string;
}

export async function getDuneuro(): Promise<DuneuroStatus | null> {
  try {
    const r = await fetch("/api/duneuro", { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

// Persist forward.duneuro_path (empty string clears it). Returns fresh status.
export async function setDuneuroPath(
  path: string | null,
): Promise<DuneuroStatus | null> {
  try {
    const r = await fetch("/api/duneuro", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path ?? "" }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

// Revert the working config back to the shipped defaults. The backend has
// always supported this; the old UI just never offered a way to call it.
export async function resetConfig(): Promise<{ config: Cfg } | null> {
  try {
    const r = await fetch("/api/config/reset", { method: "POST" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}
