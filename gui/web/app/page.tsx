"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import {
  getHealth,
  getConfig,
  putConfig,
  getMeshes,
  runSimulation,
  detectFallback,
  type MeshInfo,
  type PointSource,
  type DetectResult,
} from "@/lib/api";
import { type Cfg, getPath } from "@/lib/config";
import SourceList from "@/components/SourceList";
import ParamPanels from "@/components/ParamPanels";
import RunConsole from "@/components/RunConsole";
import ResultsPanel from "@/components/ResultsPanel";

// three.js cannot render server-side.
const Viewer3D = dynamic(() => import("@/components/Viewer3D"), { ssr: false });

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
    getConfig()
      .then((r) => setConfig(r.config))
      .catch(() => setHealthy(false));
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
    // Persist the (valid) config edits first; ignore failures so a run can
    // still proceed against the last-saved config.
    await putConfig(config);

    let gotResult = false;
    runSimulation(
      stages,
      force,
      {
        onLog: (line) => setLogs((l) => [...l, line]),
        onStatuses: (s) => setStatuses(s),
        onResult: (d) => {
          gotResult = true;
          setResult(d);
        },
        onDone: async () => {
          if (!gotResult) {
            // backend did not emit a detectability result — use the dev mock.
            const d = await detectFallback(sources, threshold, noiseFloorFt);
            if (d) setResult(d);
          }
          setRunning(false);
        },
        onError: (m) => {
          setLogs((l) => [...l, `ERROR: ${m}`]);
          setRunning(false);
        },
      },
      { sources, threshold_snr: threshold },
    );
  };

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          iNOB <span className="dim">· vagus nerve forward model</span>
        </div>
        <div className="spacer" />
        {saveMsg && <span className="pill">{saveMsg}</span>}
        <button className="ghost" onClick={save} disabled={!config}>
          Save config
        </button>
        <span className={`pill ${healthy ? "ok" : healthy === false ? "bad" : ""}`}>
          {healthy == null ? "connecting…" : healthy ? "backend online" : "backend offline"}
        </span>
      </div>

      <div className="main">
        <div className="col">
          <div style={{ height: 440, borderBottom: "1px solid var(--border)" }}>
            <Viewer3D
              meshes={meshes}
              visible={visible}
              sources={sources}
              onAddSource={(p) =>
                setSources((s) => [...s, { ...p, strength_nAm: 70 }])
              }
            />
          </div>
          <div className="panel">
            <h3>Mesh visibility</h3>
            <div>
              {meshes.map((m) => (
                <span
                  key={m.name}
                  className={`chip ${visible[m.name] ? "on" : ""}`}
                  onClick={() => setVisible((v) => ({ ...v, [m.name]: !v[m.name] }))}
                >
                  {visible[m.name] ? "✓" : "○"} {m.name}
                </span>
              ))}
              {meshes.length === 0 && <span className="hint">no meshes from backend</span>}
            </div>
          </div>
          <SourceList sources={sources} onChange={setSources} />
        </div>

        <div className="col">
          {config ? (
            <ParamPanels
              config={config}
              onChange={setConfig}
              threshold={threshold}
              onThreshold={setThreshold}
            />
          ) : (
            <div className="panel">
              <p className="hint">
                Waiting for backend config (start the FastAPI backend on :8000).
              </p>
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
