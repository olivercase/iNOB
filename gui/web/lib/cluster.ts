// Cluster submission API client (backend: /api/cluster/*). Degrades gracefully
// when the endpoints are absent (returns nulls) so the UI still works locally.
import type { PointSource } from "./api";

export interface ClusterSubmit {
  profile: string;
  state: string;          // submitted | dryrun | failed
  job_id: string | null;
  message: string;
  commands?: string[];
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

export async function submitCluster(
  profile: string,
  sources: PointSource[],
  thresholdSnr: number,
  modality: string,
  stages: string[],
): Promise<ClusterSubmit | null> {
  try {
    const r = await fetch("/api/cluster/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, sources, threshold_snr: thresholdSnr, modality, stages }),
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
