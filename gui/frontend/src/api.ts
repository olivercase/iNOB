// Typed client for the Vagus-FM FastAPI backend. All URLs are relative so the
// Vite dev proxy routes them to :8000.

export type Stage = "geom" | "fem" | "sensors" | "forward" | "viz";

export interface HealthResponse {
  status: string;
  project_root: string;
  active_config: string | null;
  stages: Stage[];
}

// The raw config is an arbitrary nested dict; we keep it loosely typed.
export type ConfigDict = Record<string, unknown>;

export interface ConfigResponse {
  source: string;
  config: ConfigDict;
  errors: string[];
}

export interface ValidateResponse {
  valid: boolean;
  errors: string[];
}

export interface SaveResponse {
  saved: boolean;
  valid: boolean;
  errors: string[];
}

export interface MeshInfo {
  name: string;
  url: string;
  bytes: number;
}

export interface MeshesResponse {
  meshes: MeshInfo[];
}

export interface RunDoneMessage {
  type: "done";
  statuses: Record<string, string>;
}

export interface RunLogMessage {
  type: "log";
  line: string;
}

export interface RunErrorMessage {
  type: "error";
  message: string;
}

export type RunMessage = RunLogMessage | RunDoneMessage | RunErrorMessage;

async function asJson<T>(res: Response): Promise<T> {
  const body = (await res.json()) as T;
  return body;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`health failed: ${res.status}`);
  return asJson<HealthResponse>(res);
}

export async function getConfig(): Promise<ConfigResponse> {
  const res = await fetch("/api/config");
  if (!res.ok) throw new Error(`config failed: ${res.status}`);
  return asJson<ConfigResponse>(res);
}

export async function validateConfig(
  config: ConfigDict
): Promise<ValidateResponse> {
  const res = await fetch("/api/config/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) throw new Error(`validate failed: ${res.status}`);
  return asJson<ValidateResponse>(res);
}

export async function saveConfig(config: ConfigDict): Promise<SaveResponse> {
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (res.status === 422) {
    const detail = (await res.json()) as { detail?: { errors?: string[] } };
    return {
      saved: false,
      valid: false,
      errors: detail.detail?.errors ?? ["Validation failed"],
    };
  }
  if (!res.ok) throw new Error(`save failed: ${res.status}`);
  return asJson<SaveResponse>(res);
}

export async function resetConfig(): Promise<ConfigResponse> {
  const res = await fetch("/api/config/reset", { method: "POST" });
  if (!res.ok) throw new Error(`reset failed: ${res.status}`);
  return asJson<ConfigResponse>(res);
}

export async function getMeshes(): Promise<MeshesResponse> {
  const res = await fetch("/api/meshes");
  if (!res.ok) throw new Error(`meshes failed: ${res.status}`);
  return asJson<MeshesResponse>(res);
}

export interface RunCallbacks {
  onLog: (line: string) => void;
  onDone: (statuses: Record<string, string>) => void;
  onError: (message: string) => void;
}

export function runPipeline(
  stages: "all" | Stage[],
  force: boolean,
  cb: RunCallbacks
): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/run`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ stages, force }));
  };
  ws.onmessage = (ev: MessageEvent<string>) => {
    let msg: RunMessage;
    try {
      msg = JSON.parse(ev.data) as RunMessage;
    } catch {
      cb.onLog(ev.data);
      return;
    }
    if (msg.type === "log") cb.onLog(msg.line);
    else if (msg.type === "done") cb.onDone(msg.statuses);
    else if (msg.type === "error") cb.onError(msg.message);
  };
  ws.onerror = () => cb.onError("WebSocket connection error");
  return ws;
}
