"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Note, Select, Tag } from "@/components/ui";
import {
  type ClusterFetch,
  type ClusterSubmit,
  clusterStatus,
  fetchCluster,
  getProfiles,
  submitCluster,
} from "@/lib/cluster";
import type { PointSource } from "@/lib/api";

interface Props {
  sources: PointSource[];
  modality: string;
}

// Terminal states stop the poller; anything else means the job is still moving.
const SETTLED = new Set(["done", "failed", "unknown"]);
const POLL_MS = 15_000;

export default function ClusterPanel({ sources, modality }: Props) {
  const [profiles, setProfiles] = useState<string[] | null>(null);
  const [profile, setProfile] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<ClusterSubmit | null>(null);
  const [state, setState] = useState<string>("");
  const [fetching, setFetching] = useState(false);
  const [fetched, setFetched] = useState<ClusterFetch | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    // A null list means "not loaded yet"; an empty one means the backend
    // genuinely has no profiles. Inventing myriad/kathleen when the backend
    // reported none only produced a confusing 422 on submit.
    getProfiles().then((p) => {
      setProfiles(p);
      setProfile((cur) => cur || p[0] || "");
    });
  }, []);

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  // Track the job until it settles. It used to be checked exactly once, right
  // after submit, so the panel always read "queued" no matter what happened next.
  useEffect(() => {
    const jobId = res?.job_id;
    if (!jobId || res?.state !== "submitted") return;
    let alive = true;
    const tick = async () => {
      const s = await clusterStatus(profile, jobId);
      if (!alive || !s) return;
      setState(s.state);
      if (SETTLED.has(s.state)) stopPolling();
    };
    tick();
    timer.current = window.setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      stopPolling();
    };
  }, [res, profile, stopPolling]);

  const submit = async () => {
    setBusy(true);
    setState("");
    setFetched(null);
    stopPolling();
    const r = await submitCluster(profile, sources, modality);
    setRes(
      r ?? {
        profile,
        state: "failed",
        job_id: null,
        message: "the backend rejected the submission (is it running?)",
      },
    );
    setBusy(false);
  };

  const doFetch = async () => {
    setFetching(true);
    const f = await fetchCluster(profile, res?.job_id ?? null, modality);
    setFetched(f ?? { state: "failed", log: "could not reach the backend" });
    setFetching(false);
  };

  const noProfiles = profiles !== null && profiles.length === 0;

  return (
    <div className="cluster">

      {noProfiles ? (
        <Note tone="warn">
          No cluster profiles found. Add one under <code>cluster/profiles/</code>{" "}
          (e.g. <code>myriad.env</code>) and reload.
        </Note>
      ) : (
        <div className="cluster-submit">
          <Select
            value={profile}
            options={(profiles ?? []).map((x) => ({ value: x, label: x }))}
            disabled={!profiles}
            ariaLabel="cluster profile"
            onChange={setProfile}
          />
          <Button
            icon="cloud"
            variant="primary"
            loading={busy}
            disabled={!profile}
            onClick={submit}
          >
            Submit {modality.toUpperCase()} to {profile || "cluster"}
          </Button>
        </div>
      )}

      {res && (
        <Note
          tone={
            res.state === "submitted"
              ? "ok"
              : res.state === "failed"
                ? "danger"
                : "warn"
          }
          title={`${res.state}${res.job_id ? ` · job ${res.job_id}` : ""}`}
        >
          <div aria-live="polite">
            {res.message}
            {state && (
              <div className="cluster-state">
                <Tag tone={state === "failed" ? "danger" : "signal"}>{state}</Tag>
                {!SETTLED.has(state) && (
                  <span className="ui-dim">
                    checking every {POLL_MS / 1000}s
                  </span>
                )}
              </div>
            )}
          </div>

          {res.commands && (
            <pre className="ui-pre mono">{res.commands.join("\n")}</pre>
          )}

          {res.state === "submitted" && (
            <div className="cluster-fetch">
              <Button size="sm" icon="cloud" loading={fetching} onClick={doFetch}>
                Fetch the finished leadfield
              </Button>
              {state && !SETTLED.has(state) && (
                <span className="ui-dim">
                  (the job is still {state})
                </span>
              )}
            </div>
          )}
        </Note>
      )}

      {fetched && (
        <Note
          tone={fetched.state === "done" ? "ok" : "warn"}
          title={`fetch: ${fetched.state}`}
        >
          {fetched.leadfield && <div className="ui-pre mono">{fetched.leadfield}</div>}
          {fetched.command && <pre className="ui-pre mono">{fetched.command}</pre>}
          {fetched.log && <pre className="ui-pre mono">{fetched.log}</pre>}
        </Note>
      )}
    </div>
  );
}
