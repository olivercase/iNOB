"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import {
  Button,
  Drawer,
  DrawerSize,
  HTMLSelect,
  Navbar,
  Spinner,
  SpinnerSize,
  Tag,
} from "@blueprintjs/core";
import {
  getHealth,
  getConfig,
  putConfig,
  resetConfig,
  getMeshes,
  runSimulation,
  type MeshInfo,
  type PointSource,
  type DetectResult,
  type DetectUnavailable,
} from "@/lib/api";
import { type Cfg } from "@/lib/config";
import { loadSession, saveSession } from "@/lib/storage";
import SourceList from "@/components/SourceList";
import ParamPanels from "@/components/ParamPanels";
import ResultsPanel from "@/components/ResultsPanel";
import ClusterPanel from "@/components/ClusterPanel";
import DuneuroSetup from "@/components/DuneuroSetup";
import LadderPanel from "@/components/LadderPanel";
import { Collapse } from "@blueprintjs/core";
import { Step, StepRail, type StepState } from "@/components/StepRail";

const Viewer3D = dynamic(() => import("@/components/Viewer3D"), { ssr: false });
const ALL_STAGES = ["geom", "fem", "sensors", "forward", "viz"];

// Detection thresholds most planners actually pick, with the textbook name.
const THRESHOLDS = [
  { value: 3, label: "3 — Rose criterion (standard)" },
  { value: 5, label: "5 — conservative" },
  { value: 2, label: "2 — lenient" },
];

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
  const [unavailable, setUnavailable] = useState<DetectUnavailable | null>(null);

  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const [ladderOpen, setLadderOpen] = useState(false);
  const [restored, setRestored] = useState(false);

  // Restore the previous session (sources, threshold, target) before the first
  // paint that matters, so placed work survives a reload.
  useEffect(() => {
    const s = loadSession();
    if (s.sources) setSources(s.sources as PointSource[]);
    if (typeof s.threshold === "number") setThreshold(s.threshold);
    if (s.target) setTarget(s.target);
    setRestored(true);
  }, []);

  useEffect(() => {
    getHealth().then((h) => setHealthy(!!h));
    getConfig()
      .then((r) => setConfig(r.config))
      .catch(() => setHealthy(false));
    getMeshes().then((m) => {
      setMeshes(m);
      setVisible((cur) =>
        Object.fromEntries(
          m.map((x) => [x.name, cur[x.name] ?? x.default_visible !== false]),
        ),
      );
      // Only pick a default target if the restored session didn't have one.
      setTarget((cur) => {
        if (cur) return cur;
        const t =
          m.find((x) => x.name.startsWith("vagus")) ??
          m.find((x) => x.name !== "skin" && x.name !== "bone") ??
          m[0];
        return t ? t.name : null;
      });
    });
  }, []);

  // Persist the session whenever the user-facing choices change.
  useEffect(() => {
    if (!restored) return;
    saveSession({ sources, threshold, target, visible });
  }, [restored, sources, threshold, target, visible]);

  const onRun = async () => {
    if (!config || sources.length === 0) return;
    setRunning(true);
    setLogs([]);
    setStatuses({});
    setResult(null);
    setUnavailable(null);
    // Save any advanced edits first so the run uses them.
    await putConfig(config);
    runSimulation(
      ALL_STAGES,
      false,
      {
        onLog: (line) => setLogs((l) => [...l, line]),
        onStatuses: (s) => setStatuses(s),
        onResult: (d) => setResult(d),
        onDone: (_s, unavail) => {
          if (unavail) setUnavailable(unavail);
          setRunning(false);
        },
        onError: (m) => {
          setLogs((l) => [...l, `ERROR: ${m}`]);
          setUnavailable({ reason: m });
          setRunning(false);
        },
      },
      { sources, threshold_snr: threshold },
    );
  };

  const onReset = async () => {
    const r = await resetConfig();
    if (r) setConfig(r.config);
  };

  // Step states drive the visual "where am I" cues in the rail.
  const step1: StepState = sources.length > 0 ? "done" : "active";
  const step2: StepState = running
    ? "active"
    : sources.length === 0
      ? "todo"
      : result || unavailable
        ? "done"
        : "active";
  const step3: StepState = result || unavailable ? "active" : "todo";

  const targetLabel = useMemo(
    () => (target ? target.replace(/_/g, " ") : "the anatomy"),
    [target],
  );

  return (
    <div className="app">
      <Navbar>
        <Navbar.Group align="left">
          <Navbar.Heading>
            <b className="brand">iNOB</b>{" "}
            <span className="bp6-text-muted subtitle">
              trials-to-detect planner
            </span>
          </Navbar.Heading>
        </Navbar.Group>
        <Navbar.Group align="right">
          <Button
            icon="cog"
            minimal
            disabled={!config}
            onClick={() => setAdvancedOpen(true)}
          >
            Advanced settings
          </Button>
          <Navbar.Divider />
          <Tag
            minimal
            intent={healthy ? "success" : healthy === false ? "danger" : "none"}
            icon={healthy ? "dot" : "offline"}
          >
            {healthy == null
              ? "connecting…"
              : healthy
                ? "backend online"
                : "backend offline"}
          </Tag>
        </Navbar.Group>
      </Navbar>

      <div className="workspace">
        <div className="canvas">
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
          <div className="viewer-legend">
            <HTMLSelect
              minimal
              value={target ?? ""}
              disabled={meshes.length === 0}
              onChange={(e) => setTarget(e.currentTarget.value || null)}
              options={meshes.map((m) => m.name)}
            />
            <span className="bp6-text-muted">
              is your imaging target — sources snap to it.
            </span>
            <span className="legend-spacer" />
            {meshes.map((m) => (
              <Tag
                key={m.name}
                interactive
                minimal={!visible[m.name]}
                intent={visible[m.name] ? "primary" : "none"}
                icon={visible[m.name] ? "eye-open" : "eye-off"}
                onClick={() =>
                  setVisible((v) => ({ ...v, [m.name]: !v[m.name] }))
                }
              >
                {m.name}
              </Tag>
            ))}
          </div>
        </div>

        <StepRail>
          <Step index={1} title="Place a source" state={step1}>
            <p className="step-lead">
              Click anywhere on <b>{targetLabel}</b> in the 3-D view to drop a
              current source. It marks a spot where nerve activity might occur.
            </p>
            <SourceList
              sources={sources}
              selected={selected}
              onSelect={setSelected}
              onChange={setSources}
            />
          </Step>

          <Step index={2} title="Run the simulation" state={step2}>
            <div className="run-controls">
              <label className="run-threshold">
                <span>Detection confidence</span>
                <HTMLSelect
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.currentTarget.value))}
                  options={THRESHOLDS}
                />
              </label>
              <Button
                large
                fill
                intent="primary"
                icon="play"
                loading={running}
                disabled={sources.length === 0 || !config}
                onClick={onRun}
              >
                {sources.length === 0
                  ? "Place a source first"
                  : running
                    ? "Solving…"
                    : "Run"}
              </Button>
            </div>

            {running && (
              <div className="run-progress">
                <Spinner size={SpinnerSize.SMALL} />
                <span>
                  Building the model and solving the leadfield. This can take a
                  few minutes.
                </span>
              </div>
            )}
            {Object.keys(statuses).length > 0 && (
              <div className="chips" style={{ marginTop: 8 }}>
                {Object.entries(statuses).map(([k, v]) => (
                  <Tag
                    key={k}
                    minimal
                    intent={
                      v === "ran" ? "success" : v === "failed" ? "danger" : "none"
                    }
                  >
                    {k}: {v}
                  </Tag>
                ))}
              </div>
            )}
            {(logs.length > 0 || running) && (
              <>
                <Button
                  minimal
                  small
                  icon={logOpen ? "chevron-down" : "chevron-right"}
                  onClick={() => setLogOpen((o) => !o)}
                  style={{ marginTop: 6 }}
                >
                  {logOpen ? "Hide" : "Show"} run log
                </Button>
                {logOpen && (
                  <pre className="console">
                    {logs.length === 0 ? "— starting —" : logs.join("\n")}
                  </pre>
                )}
              </>
            )}
          </Step>

          <Step index={3} title="Read the answer" state={step3}>
            <ResultsPanel result={result} unavailable={unavailable} />
          </Step>

          <div className="rail-extra">
            <Button
              minimal
              fill
              alignText="left"
              icon="comparison"
              rightIcon={ladderOpen ? "chevron-up" : "chevron-down"}
              onClick={() => setLadderOpen((o) => !o)}
            >
              Compare forward models (Biot–Savart → Sarvas → FEM)
            </Button>
            <Collapse isOpen={ladderOpen}>
              <div className="rail-extra-body">
                <LadderPanel />
              </div>
            </Collapse>
          </div>
        </StepRail>
      </div>

      <Drawer
        isOpen={advancedOpen}
        onClose={() => setAdvancedOpen(false)}
        title="Advanced settings"
        size={DrawerSize.SMALL}
        icon="cog"
      >
        <div className="drawer-body">
          {config ? (
            <>
              <div className="drawer-actions">
                <Button icon="reset" onClick={onReset}>
                  Reset all to defaults
                </Button>
              </div>
              <section className="adv-group">
                <h3 className="adv-title">Solver engine (DUNEuro)</h3>
                <p className="adv-blurb">
                  The forward FEM solve needs a local DUNEuro build. Point it at
                  one here.
                </p>
                <DuneuroSetup />
              </section>
              <ParamPanels config={config} onChange={setConfig} />
              <hr className="drawer-rule" />
              <h3 className="adv-title">Run on the cluster</h3>
              <p className="adv-blurb">
                Offload the DUNEuro solve to UCL Myriad or Kathleen instead of
                this machine.
              </p>
              <ClusterPanel
                sources={sources}
                threshold={threshold}
                modality="meg"
                stages={ALL_STAGES}
              />
            </>
          ) : (
            <span className="bp6-text-muted">
              Waiting for the backend config (start the FastAPI backend on
              :8000).
            </span>
          )}
        </div>
      </Drawer>
    </div>
  );
}
