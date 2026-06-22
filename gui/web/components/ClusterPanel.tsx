"use client";

import { useEffect, useState } from "react";
import { Button, Callout, Card, HTMLSelect } from "@blueprintjs/core";
import { type ClusterSubmit, clusterStatus, getProfiles, submitCluster } from "@/lib/cluster";
import type { PointSource } from "@/lib/api";

interface Props {
  sources: PointSource[];
  threshold: number;
  modality: string;
  stages: string[];
}

export default function ClusterPanel({ sources, threshold, modality, stages }: Props) {
  const [profiles, setProfiles] = useState<string[]>([]);
  const [profile, setProfile] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<ClusterSubmit | null>(null);
  const [state, setState] = useState<string>("");

  useEffect(() => {
    getProfiles().then((p) => {
      const list = p.length ? p : ["myriad", "kathleen"];
      setProfiles(list);
      setProfile((cur) => cur || list[0]);
    });
  }, []);

  const submit = async () => {
    setBusy(true);
    setState("");
    const r = await submitCluster(profile, sources, threshold, modality, stages);
    setRes(r);
    setBusy(false);
    if (r?.job_id) {
      const s = await clusterStatus(profile, r.job_id);
      if (s) setState(s.state);
    }
  };

  return (
    <Card className="panel" compact>
      <h3 className="section-title">Run on cluster</h3>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <HTMLSelect value={profile} options={profiles}
                    onChange={(e) => setProfile(e.currentTarget.value)} />
        <Button icon="cloud-upload" intent="primary" loading={busy}
                disabled={!profile} onClick={submit}>
          Submit to {profile || "cluster"}
        </Button>
      </div>
      {res && (
        <Callout
          compact
          intent={res.state === "submitted" ? "success" : res.state === "failed" ? "danger" : "warning"}
          style={{ marginTop: 8 }}
          title={`${res.state}${res.job_id ? ` · job ${res.job_id}` : ""}${state ? ` · ${state}` : ""}`}
        >
          {res.message}
          {res.commands && (
            <pre className="mono" style={{ fontSize: 11, whiteSpace: "pre-wrap", marginTop: 6 }}>
              {res.commands.join("\n")}
            </pre>
          )}
        </Callout>
      )}
    </Card>
  );
}
