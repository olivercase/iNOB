"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Button, Navbar, Tag } from "@blueprintjs/core";
import {
  getHealth, getConfig, putConfig, getMeshes, runSimulation, detectFallback,
  type MeshInfo, type PointSource, type DetectResult,
} from "@/lib/api";
import { type Cfg, getPath } from "@/lib/config";
import SourceList from "@/components/SourceList";
import ParamPanels from "@/components/ParamPanels";
import RunConsole from "@/components/RunConsole";
import ResultsPanel from "@/components/ResultsPanel";
import ClusterPanel from "@/components/ClusterPanel";

const Viewer3D = dynamic(() => import("@/components/Viewer3D"), { ssr: false });
const ALL_STAGES = ["geom", "fem", "sensors", "forward", "viz"];

export default function Page() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [config, setConfig] = useState<Cfg | null>(null);
  const [meshes, setMeshes] = useState<MeshInfo[]>([]);
  const [visible, setVisible] = useState<Record<string, boolean>>({});
  const [sources, setSources] = useState<PointSource[]>([]);
  const [threshold, setThreshold] = useState(3);
  const [logs, setLogs] = useState<string[]>([]);
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [saveMsg, setSaveMsg] = useState("");

  useEffect(() => {
    getHealth().then((h) => setHealthy(!!h));
    getConfig().then((r) => setConfig(r.config)).catch(() => setHealthy(false));
    getMeshes().then((m) => {
      setMeshes(m);
      setVisible(Object.fromEntries(m.map((x) => [x.name, true])));
    });
  }, []);

  const noiseFloorFt = useMemo(() => {
    if (!config) return 221;
    const opm = getPath<number>(config, "noise.opm_intrinsic_fT_sqrtHz", 7);
    const bw = getPath<number>(config, "noise.bandwidth_hz", 1000);
    return +(opm * Math.sqrt(bw)).toFixed(2);
  }, [config]);

  const save = async () => {
    if (!config) return;
    setSaveMsg("saving…");
    const { ok, errors } = await putConfig(config);
    setSaveMsg(ok ? "config saved" : `invalid: ${errors.join("; ")}`);
    setTimeout(() => setSaveMsg(""), 4000);
  };

  const onSimulate = async (stages: string[], force: boolean) => {
    if (!config) return;
    setRunning(true);
    setLogs([]);
    setStatuses({});
    setResult(null);
    await putConfig(config);
    let gotResult = false;
    runSimulation(stages, force, {
      onLog: (line) => setLogs((l) => [...l, line]),
      onStatuses: (s) => setStatuses(s),
      onResult: (d) => { gotResult = true; setResult(d); },
      onDone: async () => {
        if (!gotResult) {
          const d = await detectFallback(sources, threshold, noiseFloorFt);
          if (d) setResult(d);
        }
        setRunning(false);
      },
      onError: (m) => { setLogs((l) => [...l, `ERROR: ${m}`]); setRunning(false); },
    }, { sources, threshold_snr: threshold });
  };

  return (
    <div className="app">
      <Navbar>
        <Navbar.Group align="left">
          <Navbar.Heading>
            <b>iNOB</b> <span className="bp6-text-muted">· vagus nerve forward model</span>
          </Navbar.Heading>
        </Navbar.Group>
        <Navbar.Group align="right">
          {saveMsg && <Tag minimal style={{ marginRight: 8 }}>{saveMsg}</Tag>}
          <Button icon="floppy-disk" minimal disabled={!config} onClick={save}>Save config</Button>
          <Navbar.Divider />
          <Tag
            minimal
            intent={healthy ? "success" : healthy === false ? "danger" : "none"}
            icon={healthy ? "dot" : "offline"}
          >
            {healthy == null ? "connecting…" : healthy ? "backend online" : "backend offline"}
          </Tag>
        </Navbar.Group>
      </Navbar>

      <div className="main">
        <div className="col">
          <div className="viewerWrap" style={{ height: 420 }}>
            <Viewer3D
              meshes={meshes}
              visible={visible}
              sources={sources}
              onAddSource={(p) => setSources((s) => [...s, { ...p, strength_nAm: 70 }])}
            />
          </div>
          <div className="panel">
            <h3 className="section-title">Mesh visibility</h3>
            <div className="chips">
              {meshes.map((m) => (
                <Tag key={m.name} interactive minimal={!visible[m.name]}
                     intent={visible[m.name] ? "primary" : "none"}
                     icon={visible[m.name] ? "eye-open" : "eye-off"}
                     onClick={() => setVisible((v) => ({ ...v, [m.name]: !v[m.name] }))}>
                  {m.name}
                </Tag>
              ))}
              {meshes.length === 0 && <span className="bp6-text-muted">no meshes from backend</span>}
            </div>
          </div>
          <SourceList sources={sources} onChange={setSources} />
          <ClusterPanel sources={sources} threshold={threshold} modality="meg" stages={ALL_STAGES} />
        </div>

        <div className="col">
          {config ? (
            <ParamPanels config={config} onChange={setConfig}
                         threshold={threshold} onThreshold={setThreshold} />
          ) : (
            <div className="panel">
              <span className="bp6-text-muted">
                Waiting for backend config (start the FastAPI backend on :8000).
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="bottom">
        <RunConsole running={running} logs={logs} statuses={statuses} onSimulate={onSimulate} />
        <ResultsPanel result={result} />
      </div>
    </div>
  );
}
