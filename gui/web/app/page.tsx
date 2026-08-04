"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Button, Disclosure, Icon, Note, Sheet, Tag } from "@/components/ui";
import {
  getFigures,
  getHealth,
  getConfig,
  putConfig,
  resetConfig,
  getMeshes,
  runSimulation,
  type FigureInfo,
  type MeshInfo,
  type PointSource,
  type DetectResult,
  type DetectUnavailable,
  type RunHandle,
} from "@/lib/api";
import { getPath, type Cfg } from "@/lib/config";
import { loadSession, saveSession } from "@/lib/storage";
import {
  BASE_EDGES,
  BASE_NODES,
  addableById,
  figureParent,
  figurePosition,
  type AddableSpec,
  type JourneyEdge,
  type JourneyNode,
  type NodeState,
} from "@/lib/journey";
import JourneyCanvas from "@/components/JourneyCanvas";
import StepView from "@/components/StepView";
import AddOutputPanel from "@/components/AddOutputPanel";
import ParamPanels from "@/components/ParamPanels";
import ClusterPanel from "@/components/ClusterPanel";
import ComputePanel from "@/components/ComputePanel";
import DuneuroSetup from "@/components/DuneuroSetup";
import LadderPanel from "@/components/LadderPanel";

const Viewer3D = dynamic(() => import("@/components/Viewer3D"), { ssr: false });
const ALL_STAGES = ["geom", "fem", "sensors", "forward", "viz"];

// Stage names in the words a planner would use, not the CLI's.
const STAGE_LABEL: Record<string, string> = {
  geom: "Building geometry",
  fem: "Meshing tissues",
  sensors: "Placing sensors",
  forward: "Solving leadfield",
  viz: "Rendering figures",
};

// Nodes whose work happens in the 3-D view, so it opens alongside them.
const VIEWER_NODES = new Set(["anatomy", "sources"]);

export default function Page() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [config, setConfig] = useState<Cfg | null>(null);
  const [meshes, setMeshes] = useState<MeshInfo[]>([]);
  const [visible, setVisible] = useState<Record<string, boolean>>({});
  const [sources, setSources] = useState<PointSource[]>([]);
  const [selectedSource, setSelectedSource] = useState<number | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(3);
  // Which physics the run reports. The sensor-array step owns this; the EEG
  // node is a shortcut to it, so the two can never disagree.
  const [modality, setModality] = useState<"meg" | "eeg">("meg");

  const [figures, setFigures] = useState<FigureInfo[]>([]);
  const [outputs, setOutputs] = useState<string[]>([]);
  // Optional nodes the user added (EEG, the analytic rungs), by spec id.
  const [extras, setExtras] = useState<string[]>([]);
  // Where the user dragged each card. Empty means "use the authored layout".
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>(
    {},
  );
  // Two different ideas, kept apart on purpose: `marked` is the card
  // highlighted on the canvas (select, drag, delete); `selected` is the step
  // page currently open over it. Conflating them meant a single click both
  // highlighted a card and navigated away from the canvas.
  const [marked, setMarked] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);

  const [logs, setLogs] = useState<string[]>([]);
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [unavailable, setUnavailable] = useState<DetectUnavailable | null>(null);

  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const [ladderOpen, setLadderOpen] = useState(false);
  const [restored, setRestored] = useState(false);
  const [meshError, setMeshError] = useState(false);
  const [configErrors, setConfigErrors] = useState<string[]>([]);
  const [cancelling, setCancelling] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  // Which stages this run was asked for — the HUD narrates only those.
  const [runningStages, setRunningStages] = useState<string[]>(ALL_STAGES);
  const [elapsed, setElapsed] = useState(0);
  const runRef = useRef<RunHandle | null>(null);

  // The 3-D well is expensive to build (tens of MB of STL), so it is mounted on
  // first use and then only hidden — never unmounted — when another node is
  // selected. Re-mounting would refetch the whole anatomy every time.
  const [wellMounted, setWellMounted] = useState(false);
  const viewerOpen = selected !== null && VIEWER_NODES.has(selectedKind(selected));

  useEffect(() => {
    if (viewerOpen) setWellMounted(true);
  }, [viewerOpen]);

  useEffect(
    () => () => {
      runRef.current?.close();
    },
    [],
  );

  useEffect(() => {
    const s = loadSession();
    if (s.sources) setSources(s.sources as PointSource[]);
    if (typeof s.threshold === "number") setThreshold(s.threshold);
    if (s.target) setTarget(s.target);
    if (s.visible) setVisible(s.visible);
    if (s.outputs) setOutputs(s.outputs);
    if (s.extras) setExtras(s.extras);
    if (s.modality === "eeg" || s.modality === "meg") setModality(s.modality);
    if (s.positions) setPositions(s.positions);
    setRestored(true);
  }, []);

  useEffect(() => {
    getConfig()
      .then((r) => setConfig(r.config))
      .catch(() => setHealthy(false));
    getMeshes()
      .then((m) => {
        setMeshes(m);
        setVisible((cur) =>
          Object.fromEntries(
            m.map((x) => [x.name, cur[x.name] ?? x.default_visible !== false]),
          ),
        );
        setTarget((cur) => {
          if (cur) return cur;
          const t =
            m.find((x) => x.name.startsWith("vagus")) ??
            m.find((x) => x.name !== "skin" && x.name !== "bone") ??
            m[0];
          return t ? t.name : null;
        });
      })
      .catch(() => setMeshError(true));
  }, []);

  const refreshFigures = useCallback(() => {
    getFigures().then(setFigures);
  }, []);

  useEffect(() => {
    refreshFigures();
  }, [refreshFigures]);

  // A run's own clock, so "still going" always has a number attached.
  useEffect(() => {
    if (!running || startedAt === null) return;
    const tick = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [running, startedAt]);

  // While a run is in flight, figures land on disk one stage at a time — poll
  // so an output node fills in the moment its PNG is written, not at the end.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(refreshFigures, 4000);
    return () => window.clearInterval(id);
  }, [running, refreshFigures]);

  useEffect(() => {
    let alive = true;
    const check = () => {
      getHealth().then((h) => {
        if (alive) setHealthy(!!h);
      });
    };
    check();
    const id = window.setInterval(check, 10_000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!restored) return;
    saveSession({
      sources,
      threshold,
      target,
      visible,
      outputs,
      positions,
      extras,
      modality,
    });
  }, [
    restored,
    sources,
    threshold,
    target,
    visible,
    outputs,
    positions,
    extras,
    modality,
  ]);

  // ── the graph ──────────────────────────────────────────────────────────────

  const figureByKey = useMemo(
    () => new Map(figures.map((f) => [f.key, f])),
    [figures],
  );

  const nodes: JourneyNode[] = useMemo(() => {
    const list = [...BASE_NODES];

    for (const id of extras) {
      const spec = addableById(id);
      if (!spec) continue;
      list.push({
        id: spec.id,
        kind: spec.kind,
        title: spec.title,
        caption: spec.caption,
        icon: spec.icon,
        x: spec.x,
        y: spec.y,
      });
    }

    const perParent = new Map<string, number>();
    for (const key of outputs) {
      const fig = figureByKey.get(key);
      if (!fig) continue;
      const parentId = figureParent(fig.stage);
      const parent = BASE_NODES.find((n) => n.id === parentId) ?? BASE_NODES[0];
      const i = perParent.get(parentId) ?? 0;
      perParent.set(parentId, i + 1);
      const { x, y } = figurePosition(parent, i);
      list.push({
        id: `fig:${key}`,
        kind: "figure",
        title: fig.label,
        caption: fig.exists ? "figure" : "not drawn yet",
        stage: fig.stage,
        icon: "figure",
        figureKey: key,
        x,
        y,
      });
    }
    // A dragged card keeps where the user put it; everything else sits where
    // the journey layout says it should.
    return list.map((n) => {
      const at = positions[n.id];
      return at ? { ...n, x: at.x, y: at.y } : n;
    });
  }, [outputs, extras, figureByKey, positions]);

  const edges: JourneyEdge[] = useMemo(() => {
    const list = [...BASE_EDGES];
    for (const id of extras) {
      const spec = addableById(id);
      if (!spec) continue;
      list.push({ from: spec.from, to: spec.id });
      if (spec.to) list.push({ from: spec.id, to: spec.to });
    }
    for (const n of nodes) {
      if (n.kind !== "figure" || !n.stage) continue;
      list.push({ from: figureParent(n.stage), to: n.id });
    }
    return list;
  }, [nodes, extras]);

  const stageState = useCallback(
    (name: string, fallback: NodeState): NodeState => {
      if (running && stage === name) return "active";
      const st = statuses[name];
      if (st === "ran") return "done";
      if (st === "skipped") return "skipped";
      if (st === "failed") return "failed";
      if (st === "cancelled") return "idle";
      return fallback;
    },
    [running, stage, statuses],
  );

  const states: Record<string, NodeState> = useMemo(() => {
    const s: Record<string, NodeState> = {
      anatomy: stageState("geom", meshes.length ? "ready" : "idle"),
      // Conductivities aren't a pipeline stage — they're settings the solve
      // reads — so the node is "ready" as soon as the config has values.
      conductivity: config ? "ready" : "idle",
      mesh: stageState("fem", "idle"),
      sensors: stageState("sensors", "idle"),
      sources: sources.length ? "done" : "ready",
      solve: stageState("forward", "idle"),
      detect: result
        ? "done"
        : unavailable
          ? "failed"
          : sources.length
            ? "ready"
            : "idle",
    };
    // The optional nodes: EEG follows the chosen modality; the analytic rungs
    // are always available, since neither needs a leadfield.
    if (extras.includes("eeg")) s.eeg = modality === "eeg" ? "done" : "ready";
    if (extras.includes("biot")) s.biot = "ready";
    if (extras.includes("sarvas")) s.sarvas = "ready";

    for (const n of nodes) {
      if (n.kind !== "figure") continue;
      const fig = n.figureKey ? figureByKey.get(n.figureKey) : undefined;
      s[n.id] =
        running && stage === n.stage ? "active" : fig?.exists ? "done" : "idle";
    }
    return s;
  }, [
    stageState,
    config,
    extras,
    modality,
    meshes.length,
    sources.length,
    result,
    unavailable,
    nodes,
    figureByKey,
    running,
    stage,
  ]);

  const badges: Record<string, string | undefined> = useMemo(() => {
    const workers = config ? getPath<number>(config, "forward.local_workers", 0) : 0;
    const tissues = config ? getPath<string[]>(config, "fem.tissues", []) : [];
    const pitch = config ? getPath<number>(config, "fem.pitch_mm", 0) : 0;
    const b: Record<string, string | undefined> = {
      anatomy: target ? target.replace(/_/g, " ") : undefined,
      conductivity: tissues.length ? `${tissues.length} tissues` : undefined,
      mesh: pitch ? `${pitch} mm pitch` : undefined,
      eeg: modality === "eeg" ? "solving EEG" : "not selected",
      sources: sources.length
        ? `${sources.length} placed`
        : undefined,
      sensors: result
        ? `${result.array.n_sensors} ${modality.toUpperCase()} sensors`
        : modality.toUpperCase(),
      solve: workers === 0 ? "all cores" : `${workers}×`,
      detect: result
        ? bestTrials(result)
        : undefined,
    };
    return b;
  }, [config, target, sources.length, result, modality]);

  const previews: Record<string, string | undefined> = useMemo(() => {
    const p: Record<string, string | undefined> = {};
    for (const n of nodes) {
      if (n.kind !== "figure" || !n.figureKey) continue;
      const fig = figureByKey.get(n.figureKey);
      if (fig?.exists) p[n.id] = `${fig.url}?v=${Math.round(fig.mtime)}`;
    }
    return p;
  }, [nodes, figureByKey]);

  const selectedNode = nodes.find((n) => n.id === selected) ?? null;
  const selectedFigure =
    selectedNode?.figureKey ? figureByKey.get(selectedNode.figureKey) : undefined;

  function selectedKind(id: string): string {
    if (id.startsWith("fig:")) return "figure";
    return (
      BASE_NODES.find((n) => n.id === id)?.kind ?? addableById(id)?.kind ?? ""
    );
  }

  // ── running ────────────────────────────────────────────────────────────────

  // One run path for everything. `stages` is what the backend is asked to do,
  // so the same code covers "run the whole journey" and "just re-mesh" — the
  // stages a step doesn't ask for are never touched, and the ones it does ask
  // for skip themselves when their outputs are already on disk.
  const runStages = async (stages: string[]) => {
    if (!config || running) return;
    // Only a solve needs dipoles; rebuilding geometry or the mesh does not.
    const needsSources = stages.includes("forward");
    if (needsSources && sources.length === 0) return;
    setLogs([]);
    setStatuses({});
    setResult(null);
    setUnavailable(null);
    setConfigErrors([]);
    setStage(null);

    const saved = await putConfig(config);
    if (!saved.ok) {
      setConfigErrors(
        saved.errors.length ? saved.errors : ["the config could not be saved"],
      );
      return;
    }

    setRunning(true);
    setCancelling(false);
    setStartedAt(Date.now());
    setElapsed(0);
    setRunningStages(stages);
    runRef.current = runSimulation(
      stages,
      false,
      {
        onLog: (line) => {
          setLogs((l) => [...l, line]);
          const started = line.match(/\[run]\s+(\w+):/);
          if (started) setStage(started[1]);
          const settled = line.match(/\[(ok|skip)]\s+(\w+)/);
          if (settled) {
            setStatuses((s) => ({
              ...s,
              [settled[2]]: settled[1] === "ok" ? "ran" : "skipped",
            }));
          }
        },
        onStatuses: (s) => setStatuses(s),
        onResult: (d) => setResult(d),
        onDone: (_s, unavail, cancelled) => {
          if (cancelled) {
            setUnavailable({
              reason: "Run cancelled before the solve finished.",
              hint: "Press Run again to start over.",
            });
          } else if (unavail) {
            setUnavailable(unavail);
          }
          setRunning(false);
          setCancelling(false);
          setStage(null);
          runRef.current = null;
          refreshFigures();
        },
        onError: (m) => {
          setLogs((l) => [...l, `ERROR: ${m}`]);
          setUnavailable({ reason: m });
          setRunning(false);
          setCancelling(false);
          runRef.current = null;
        },
      },
      // Sources are only sent when this run actually solves; sending them
      // otherwise would force a re-solve the user didn't ask for.
      needsSources
        ? { sources, threshold_snr: threshold, modality }
        : { threshold_snr: threshold, modality },
    );
  };

  const onRun = () => runStages(ALL_STAGES);

  const onCancel = () => {
    if (!runRef.current) return;
    setCancelling(true);
    runRef.current.cancel();
  };

  // "Save and return" from a step. Placed sources and view choices are already
  // held in state (and localStorage); what needs persisting is the config the
  // pipeline will read, so write it and surface any validation error in the
  // step rather than failing silently at run time.
  const onSaveStep = async (): Promise<string[]> => {
    if (!config) return [];
    const saved = await putConfig(config);
    if (saved.ok) return [];
    return saved.errors.length ? saved.errors : ["the config could not be saved"];
  };

  // Only pinned output figures can be removed; the five core stages are what
  // the FEM run is made of, so the canvas never offers to delete them.
  const removeNode = useCallback((id: string) => {
    if (id.startsWith("fig:")) {
      const key = id.slice(4);
      setOutputs((o) => o.filter((k) => k !== key));
    } else if (addableById(id)) {
      setExtras((e) => e.filter((k) => k !== id));
      // Removing the EEG node hands the run back to the OPM array, so the
      // canvas and the next run can never disagree about what is being solved.
      if (id === "eeg") setModality("meg");
    } else {
      return; // a core stage: the run is made of exactly these
    }
    setPositions((p) => {
      const next = { ...p };
      delete next[id];
      return next;
    });
    setMarked((m) => (m === id ? null : m));
    setSelected((sel) => (sel === id ? null : sel));
  }, []);

  const onReset = async () => {
    const r = await resetConfig();
    if (r) setConfig(r.config);
  };

  const computeSummary = useMemo(() => {
    if (!config) return undefined;
    const w = getPath<number>(config, "forward.local_workers", 0);
    return w === 0 ? "all cores" : `${w} core${w === 1 ? "" : "s"}`;
  }, [config]);

  const hudStages = running || Object.keys(statuses).length ? runningStages : ALL_STAGES;
  const doneCount = hudStages.filter(
    (s) => statuses[s] === "ran" || statuses[s] === "skipped",
  ).length;

  return (
    <div className="jshell">
      <main className="jstage">
        <JourneyCanvas
          nodes={nodes}
          edges={edges}
          states={states}
          badges={badges}
          previews={previews}
          selected={marked}
          onSelect={setMarked}
          onOpen={(id) => {
            setSelected(id);
            setMarked(id);
            setAddOpen(false);
          }}
          onMove={(id, x, y) =>
            setPositions((p) => ({ ...p, [id]: { x: Math.round(x), y: Math.round(y) } }))
          }
          onRemove={removeNode}
          onRunNode={(stages) => runStages(stages)}
          running={running}
        />

        <div className="jchrome jchrome--tl" hidden={!!selectedNode}>
          <span className="jbrand">
            <span className="jbrand-mark" aria-hidden />
            <span className="jbrand-name">iNOB</span>
          </span>
          <span className="jbrand-sub">trials-to-detect journey</span>
        </div>

        <div className="jchrome jchrome--tr" hidden={!!selectedNode}>
          {running ? (
            <Button variant="danger" icon="stop" disabled={cancelling} onClick={onCancel}>
              {cancelling ? "Stopping…" : "Stop"}
            </Button>
          ) : (
            <Button
              variant="primary"
              icon="play"
              disabled={sources.length === 0 || !config}
              onClick={onRun}
              title={sources.length === 0 ? "Place a current source first" : "Run every stage"}
            >
              Run journey
            </Button>
          )}
          <Button icon="cog" disabled={!config} onClick={() => setAdvancedOpen(true)}>
            Advanced
          </Button>
          <span
            className={`jhealth${healthy ? " jhealth--up" : healthy === false ? " jhealth--down" : ""}`}
            title={healthy ? "backend online" : "backend offline"}
          >
            <span className="jhealth-dot" aria-hidden />
            {healthy == null ? "connecting" : healthy ? "online" : "offline"}
          </span>
        </div>

        {/* The 3-D well: where anatomy is chosen and sources are placed. */}
        {wellMounted && (
          <section
            className={`jwell${viewerOpen ? "" : " jwell--hidden"}`}
            aria-hidden={!viewerOpen}
            aria-label="3-D anatomy view"
          >
            <Viewer3D
              meshes={meshes}
              visible={visible}
              sources={sources}
              selected={selectedSource}
              target={target}
              onSelect={setSelectedSource}
              onAddSource={(p) => {
                setSelectedSource(sources.length);
                setSources((s) => [...s, { ...p, strength_nAm: 70 }]);
              }}
            />
            {meshError && (
              <div className="jwell-error">
                <Tag tone="danger" icon="alert">
                  could not load the anatomy — is the backend running?
                </Tag>
              </div>
            )}
          </section>
        )}

        <div className="jtools" hidden={!!selectedNode}>
          <button
            type="button"
            className={`jtool jtool--wide${addOpen ? " jtool--on" : ""}`}
            onClick={() => {
              setAddOpen((o) => !o);
              setSelected(null);
              refreshFigures();
            }}
          >
            <Icon name="plus" size={14} /> Add output
          </button>
          <span className="jtool-sep" aria-hidden />
          <button
            type="button"
            className={`jtool${logOpen ? " jtool--on" : ""}`}
            onClick={() => setLogOpen((o) => !o)}
            title="Show the run log"
            aria-label="Show the run log"
          >
            <Icon name="terminal" size={15} />
          </button>
          <button
            type="button"
            className={`jtool${ladderOpen ? " jtool--on" : ""}`}
            onClick={() => setLadderOpen((o) => !o)}
            title="Compare forward models"
            aria-label="Compare forward models"
          >
            <Icon name="compare" size={15} />
          </button>
        </div>

        {addOpen && !selectedNode && (
          <AddOutputPanel
            figures={figures}
            added={
              new Set([...outputs.map((k) => `fig:${k}`), ...extras])
            }
            onAddFigure={(f) =>
              setOutputs((o) => (o.includes(f.key) ? o : [...o, f.key]))
            }
            onAddNode={(spec: AddableSpec) => {
              setExtras((e) => (e.includes(spec.id) ? e : [...e, spec.id]));
              if (spec.id === "eeg") setModality("eeg");
            }}
            onClose={() => setAddOpen(false)}
          />
        )}

        {selectedNode && (
          <StepView
            node={selectedNode}
            state={states[selectedNode.id] ?? "idle"}
            usesViewer={viewerOpen}
            figure={selectedFigure}
            config={config}
            onConfigChange={setConfig}
            meshes={meshes}
            visible={visible}
            onVisibleChange={setVisible}
            target={target}
            onTargetChange={setTarget}
            sources={sources}
            onSourcesChange={setSources}
            selectedSource={selectedSource}
            onSelectSource={setSelectedSource}
            threshold={threshold}
            onThresholdChange={setThreshold}
            modality={modality}
            onModalityChange={setModality}
            result={result}
            unavailable={unavailable}
            running={running}
            onRunStage={(stages) => {
              setSelected(null);
              runStages(stages);
            }}
            onOpenAdvanced={() => setAdvancedOpen(true)}
            onSave={onSaveStep}
            onBack={() => setSelected(null)}
          />
        )}

        {ladderOpen && (
          <section className="jpanel jpanel--ladder" aria-label="Forward model comparison">
            <header className="jpanel-head">
              <h2>Compare forward models</h2>
              <button
                type="button"
                className="jpanel-x"
                onClick={() => setLadderOpen(false)}
                aria-label="Close"
              >
                <Icon name="close" size={13} />
              </button>
            </header>
            <div className="jpanel-scroll">
              <LadderPanel />
            </div>
          </section>
        )}

        {/* The run, narrated. Bottom centre, because during a run this is
            the only thing anyone is looking at. */}
        <div className="jrun" aria-live="polite">
          {configErrors.length > 0 && !running && (
            <div className="jrun-card jrun-card--bad">
              <Note tone="danger" title="Nothing was run">
                <ul className="ui-errlist">
                  {configErrors.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </Note>
            </div>
          )}

          {(running || doneCount > 0) && configErrors.length === 0 && (
            <div className={`jrun-card${running ? " jrun-card--live" : ""}`}>
              <div className="jrun-top">
                <span className="jrun-badge mono">
                  {running
                    ? `${Math.min(doneCount + 1, hudStages.length)} / ${hudStages.length}`
                    : "done"}
                </span>
                <span className="jrun-name">
                  {cancelling
                    ? "Stopping after this stage"
                    : running
                      ? (STAGE_LABEL[stage ?? ""] ?? "Starting the pipeline")
                      : unavailable
                        ? "Finished — no result"
                        : "Journey complete"}
                </span>
                <span className="jrun-clock mono">
                  {String(Math.floor(elapsed / 60)).padStart(2, "0")}:
                  {String(elapsed % 60).padStart(2, "0")}
                </span>
              </div>

              <ol className="jrun-steps">
                {hudStages.map((sname) => {
                  const st = statuses[sname];
                  const cls =
                    running && stage === sname
                      ? "active"
                      : st === "ran"
                        ? "done"
                        : st === "skipped"
                          ? "skipped"
                          : st === "failed"
                            ? "failed"
                            : "todo";
                  return (
                    <li key={sname} className={`jrun-step jrun-step--${cls}`}>
                      <span className="jrun-step-mark" aria-hidden>
                        {cls === "done" || cls === "skipped" ? (
                          <Icon name="check" size={11} />
                        ) : cls === "failed" ? (
                          <Icon name="alert" size={11} />
                        ) : cls === "active" ? (
                          <span className="jrun-step-spin" />
                        ) : null}
                      </span>
                      <span className="jrun-step-label">{STAGE_LABEL[sname]}</span>
                      {st === "skipped" && (
                        <span className="jrun-step-note">reused</span>
                      )}
                    </li>
                  );
                })}
              </ol>

              <div className="jrun-track" aria-hidden>
                <span
                  className="jrun-fill"
                  style={{ width: `${(doneCount / hudStages.length) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {logOpen && (
          <section className="jlog" aria-label="Run log">
            <header>
              <span className="mono">run log</span>
              <button type="button" onClick={() => setLogOpen(false)} aria-label="Close">
                <Icon name="close" size={12} />
              </button>
            </header>
            <pre>{logs.length === 0 ? "— nothing yet —" : logs.join("\n")}</pre>
          </section>
        )}

        {sources.length === 0 && !running && !selectedNode && (
          <p className="jhint">
            Open <b>Sources</b>, then click the target in the 3-D view to drop
            one. Drag any card to rearrange the journey.
          </p>
        )}
      </main>

      <Sheet
        open={advancedOpen}
        onClose={() => setAdvancedOpen(false)}
        title="Advanced settings"
        icon="cog"
      >
        <div className="drawer-body">
          {config ? (
            <>
              <p className="drawer-lead ui-dim">
                Every setting here already has a working default. Open a layer only
                when you want to change what it covers.
              </p>
              <div className="drawer-actions">
                <Button icon="reset" onClick={onReset}>
                  Reset all to defaults
                </Button>
              </div>

              <div className="adv">
                <Disclosure
                  title="Compute"
                  icon="solve"
                  blurb={
                    "The forward solve splits the sensor array into independent " +
                    "chunks and solves them in parallel, one process per worker."
                  }
                  summary={computeSummary}
                  defaultOpen
                >
                  <ComputePanel config={config} onChange={setConfig} />
                </Disclosure>

                <Disclosure
                  title="Solver engine (DUNEuro)"
                  icon="cog"
                  blurb="The forward FEM solve needs a local DUNEuro build. Point it at one here."
                >
                  <DuneuroSetup />
                </Disclosure>
              </div>

              <ParamPanels config={config} onChange={setConfig} />

              <div className="adv">
                <Disclosure
                  title="Run on the cluster"
                  icon="cloud"
                  blurb={
                    "Offload the DUNEuro solve to UCL Myriad or Kathleen instead " +
                    "of this machine. Only the forward solve runs there; the " +
                    "result is fetched back and analysed locally."
                  }
                >
                  <ClusterPanel sources={sources} modality={modality} />
                </Disclosure>
              </div>
            </>
          ) : (
            <span className="ui-dim">
              Waiting for the backend config (start the FastAPI backend on :8000).
            </span>
          )}
        </div>
      </Sheet>
    </div>
  );
}

// The headline number on the detect node: the easiest source to detect.
function bestTrials(r: DetectResult): string {
  const trials = r.per_source
    .map((p) => p.trials_needed)
    .filter((n) => n >= 0 && Number.isFinite(n));
  if (trials.length === 0) return "∞";
  return `${Math.min(...trials).toLocaleString()} trials`;
}
