"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Button, HTMLSelect, Navbar, Tag } from "@blueprintjs/core";
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
  const [selected, setSelected] = useState<number | null>(null);
  const [target, setTarget] = useState<string | null>(null);
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
      // Default the imaging target to the first non-skin structure available.
      const t = m.find((x) => x.name !== "skin") ?? m[0];
      if (t) setTarget(t.name);
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
            <b className="brand">iNOB</b>{" "}
            <span className="bp6-text-muted">· imaging neuroscience outside the brain</span>
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
              selected={selected}
              target={target}
              onSelect={setSelected}
              onAddSource={(p) =>
                setSources((s) => {
                  setSelected(s.length);
                  return [...s, { ...p, strength_nAm: 70 }];
                })
              }
            />
          </div>
          <div className="panel">
            <h3 className="section-title">Anatomy</h3>
            <div className="field-row">
              <label title="Structure that sources snap to and is highlighted in the viewer">
                Imaging target
              </label>
              <HTMLSelect
                value={target ?? ""}
                disabled={meshes.length === 0}
                onChange={(e) => setTarget(e.currentTarget.value || null)}
                options={meshes.map((m) => m.name)}
              />
            </div>
            <div className="chips" style={{ marginTop: 6 }}>
              {meshes.map((m) => (
                <Tag key={m.name} interactive minimal={!visible[m.name]}
                     intent={visible[m.name] ? "primary" : "none"}
                     icon={visible[m.name] ? "eye-open" : "eye-off"}
                     onClick={() => setVisible((v) => ({ ...v, [m.name]: !v[m.name] }))}>
                  {m.name === target ? `★ ${m.name}` : m.name}
                </Tag>
              ))}
              {meshes.length === 0 && <span className="bp6-text-muted">no meshes from backend</span>}
            </div>
          </div>
          <SourceList sources={sources} selected={selected} onSelect={setSelected} onChange={setSources} />
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
