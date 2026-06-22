// Nested-config get/set helpers. The backend config is an arbitrary nested
// dict (validated server-side by inob.config.load_config); we edit it in place
// by dotted path, e.g. "forward.conductivities_sm.bone".

export type Cfg = Record<string, unknown>;

export function getPath<T = unknown>(obj: Cfg, path: string, fallback?: T): T {
  const parts = path.split(".");
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur && typeof cur === "object" && p in (cur as object)) {
      cur = (cur as Record<string, unknown>)[p];
    } else {
      return fallback as T;
    }
  }
  return (cur as T) ?? (fallback as T);
}

// Returns a deep-cloned config with `path` set to `value`.
export function setPath(obj: Cfg, path: string, value: unknown): Cfg {
  const next = structuredClone(obj);
  const parts = path.split(".");
  let cur: Record<string, unknown> = next;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    if (typeof cur[p] !== "object" || cur[p] === null) cur[p] = {};
    cur = cur[p] as Record<string, unknown>;
  }
  cur[parts[parts.length - 1]] = value;
  return next;
}
