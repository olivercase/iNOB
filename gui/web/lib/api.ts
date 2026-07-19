// Typed client for the iNOB backend. HTTP calls go same-origin to /api/* and
// are proxied to FastAPI by Next rewrites (see next.config.mjs). The pipeline
// run is a WebSocket connected directly to the backend.

import type { Cfg } from "./config";

const WS_BASE =
  process.env.NEXT_PUBLIC_INOB_WS || "ws://127.0.0.1:8000";

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
  return { ok: r.ok, errors: body.errors ?? (r.ok ? [] : [`PUT failed: ${r.status}`]) };
}

export async function validateConfig(config: Cfg): Promise<string[]> {
  const r = await fetch("/api/config/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  const body = await r.json().catch(() => ({}));
  return body.errors ?? [];
}

export async function getMeshes(): Promise<MeshInfo[]> {
  const r = await fetch("/api/meshes", { cache: "no-store" });
  if (!r.ok) return [];
  const body = await r.json();
  return body.meshes ?? body ?? [];
}

export interface RunHandlers {
  onLog: (line: string) => void;
  onStatuses?: (statuses: Record<string, string>) => void;
  onResult?: (detect: DetectResult) => void;
  onDone: (
    statuses: Record<string, string>,
    unavailable?: DetectUnavailable,
  ) => void;
  onError: (message: string) => void;
}

// Opens the run WebSocket; returns a closer. Streams {type: log|result|done|error}.
// `extra` (sources, threshold_snr) is merged into the init payload; the current
// backend ignores unknown fields, the future backend reads them (see API_CONTRACT).
export function runSimulation(
  stages: string[],
  force: boolean,
  handlers: RunHandlers,
  extra: Record<string, unknown> = {},
): () => void {
  const ws = new WebSocket(`${WS_BASE}/api/run`);
  let lastStatuses: Record<string, string> = {};
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
        handlers.onDone(
          lastStatuses,
          (msg.detect_unavailable as DetectUnavailable | null) ?? undefined,
        );
        ws.close();
        break;
      case "error":
        handlers.onError(String(msg.message ?? "unknown error"));
        ws.close();
        break;
    }
  };
  ws.onerror = () =>
    handlers.onError(
      `WebSocket error — is the backend running at ${WS_BASE}?`,
    );
  return () => ws.close();
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
