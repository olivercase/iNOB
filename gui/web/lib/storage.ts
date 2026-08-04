// Small localStorage wrapper for the bits of UI state that are expensive to
// recreate by hand — placed sources above all. Previously a page reload threw
// all of it away.
//
// Everything is best-effort: private browsing, a full quota, or a stale schema
// must never stop the app from starting, so failures fall back to defaults.

const KEY = "inob.session.v1";

export interface Session {
  sources: unknown[];
  threshold: number;
  target: string | null;
  visible: Record<string, boolean>;
  /** Figure keys the user pinned to the journey canvas as output nodes. */
  outputs: string[];
  /** Where the user dragged each node, by node id. */
  positions: Record<string, { x: number; y: number }>;
}

export function loadSession(): Partial<Session> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<Session>;
    // Guard the shapes we actually index into; a hand-edited or older payload
    // should degrade to defaults rather than crash the first render.
    return {
      sources: Array.isArray(parsed.sources) ? parsed.sources : undefined,
      threshold:
        typeof parsed.threshold === "number" && parsed.threshold > 0
          ? parsed.threshold
          : undefined,
      target: typeof parsed.target === "string" ? parsed.target : undefined,
      visible:
        parsed.visible && typeof parsed.visible === "object"
          ? (parsed.visible as Record<string, boolean>)
          : undefined,
      outputs: Array.isArray(parsed.outputs)
        ? parsed.outputs.filter((k): k is string => typeof k === "string")
        : undefined,
      positions:
        parsed.positions && typeof parsed.positions === "object"
          ? (parsed.positions as Record<string, { x: number; y: number }>)
          : undefined,
    };
  } catch {
    return {};
  }
}

export function saveSession(session: Session): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    /* quota or private mode — the app works fine without persistence */
  }
}

export function clearSession(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}
