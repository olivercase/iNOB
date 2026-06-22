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
  mocked?: boolean;
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
  onDone: (statuses: Record<string, string>) => void;
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
        handlers.onDone(lastStatuses);
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

// Dev fallback: ask the local mock route for a detectability estimate when the
// backend does not (yet) emit a {type:"result"} message. See app/api/detect.
export async function detectFallback(
  sources: PointSource[],
  thresholdSnr: number,
  noiseFloorFt: number,
): Promise<DetectResult | null> {
  try {
    const r = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sources,
        threshold_snr: thresholdSnr,
        noise_floor_fT: noiseFloorFt,
      }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}
