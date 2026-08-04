// Cluster submission API client (backend: /api/cluster/*). Degrades gracefully
// when the endpoints are absent (returns nulls) so the UI still works locally.
import type { PointSource } from "./api";

export interface ClusterSubmit {
  profile: string;
  state: string;          // submitted | dryrun | failed
  job_id: string | null;
  message: string;
  modality?: string;
  commands?: string[];
  log?: string;
}

export interface ClusterFetch {
  state: string;          // done | failed | dryrun
  leadfield?: string;
  command?: string;
  log?: string;
}

export async function getProfiles(): Promise<string[]> {
  try {
    const r = await fetch("/api/cluster/profiles", { cache: "no-store" });
    if (!r.ok) return [];
    const b = await r.json();
    return b.profiles ?? [];
  } catch {
    return [];
  }
}

// Note there is no `stages` / `thresholdSnr` here: the cluster only runs the
// forward solve, and both are local post-processing applied after the leadfield
// comes back. They used to be sent and silently ignored by the backend, which
// made the GUI look like it was honouring settings it wasn't.
export async function submitCluster(
  profile: string,
  sources: PointSource[],
  modality: string,
): Promise<ClusterSubmit | null> {
  try {
    const r = await fetch("/api/cluster/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, sources, modality }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

// Pull the finished leadfield back from the cluster into outputs/forward.
// The endpoint has always existed; the UI simply never offered a way to call it.
export async function fetchCluster(
  profile: string,
  jobId: string | null,
  modality: string,
): Promise<ClusterFetch | null> {
  try {
    const r = await fetch("/api/cluster/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, job_id: jobId, modality }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function clusterStatus(
  profile: string,
  jobId: string,
): Promise<{ state: string; detail?: string } | null> {
  try {
    const r = await fetch(
      `/api/cluster/status?profile=${encodeURIComponent(profile)}&job_id=${encodeURIComponent(jobId)}`,
      { cache: "no-store" },
    );
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}
