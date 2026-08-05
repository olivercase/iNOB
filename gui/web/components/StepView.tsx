"use client";

import { useEffect, useState } from "react";
import { Button, Icon, Note, Select, Tag } from "@/components/ui";
import type { IconName } from "@/components/ui/Icon";
import { getPath, setPath, type Cfg } from "@/lib/config";
import {
  getFieldMap,
  getLevels,
  getSensorArray,
  runLadder,
  suggestSources,
  type LadderResult,
  type LadderRung,
  type FieldMap,
  type LevelInfo,
  type SensorArrayInfo,
  type SuggestedSources,
} from "@/lib/api";
import type {
  DetectResult,
  DetectUnavailable,
  FigureInfo,
  MeshInfo,
  PointSource,
} from "@/lib/api";
import type { JourneyNode, NodeState } from "@/lib/journey";
import SourceList from "@/components/SourceList";
import ClusterPanel from "@/components/ClusterPanel";
import DuneuroSetup from "@/components/DuneuroSetup";
import ResultsPanel from "@/components/ResultsPanel";
import ComputePanel from "@/components/ComputePanel";
import FigureView from "@/components/FigureView";
import { BarChart, StatTile } from "@/components/Chart";

const STATE_WORD: Record<NodeState, string> = {
  idle: "not run yet",
  ready: "ready",
  active: "running now",
  done: "done",
  skipped: "reused",
  failed: "failed",
};

// Idle and ready have no mark: nothing has happened to report yet.
const STATE_ICON: Partial<Record<NodeState, IconName>> = {
  active: "play",
  done: "check",
  skipped: "check",
  failed: "alert",
};

// The vertebral levels this anatomy carries, grouped the way a clinician says
// them. Mirrors inob.anatomy.VERTEBRA_LEVELS.
const VERTEBRAE = [
  { name: "cervical", levels: ["c1", "c2", "c3", "c4", "c5", "c6", "c7"] },
  {
    name: "thoracic",
    levels: ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"],
  },
  { name: "lumbar", levels: ["l1", "l2", "l3", "l4", "l5"] },
];

const THRESHOLDS = [
  { value: 3, label: "3 — Rose criterion (standard)" },
  { value: 5, label: "5 — conservative" },
  { value: 2, label: "2 — lenient" },
];

// Meshing presets, in the terms that actually matter: how fine the elements
// are. Each writes the three fem knobs together, because setting one without
// the others produces a mesh that is fine in one dimension and coarse in
// another.
const MESH_PRESETS = [
  {
    key: "coarse",
    label: "Coarse — fastest",
    note: "Quick to build and solve. Good for checking a layout works.",
    values: { pitch_mm: 4, radbound: 3.5, maxvol: 40 },
  },
  {
    key: "standard",
    label: "Standard",
    note: "The default. Sensible everywhere, and what the published runs use.",
    values: { pitch_mm: 3, radbound: 2.5, maxvol: 20 },
  },
  {
    key: "fine",
    label: "Fine — slowest",
    note: "Smaller elements near thin structures. Costs memory and solve time.",
    values: { pitch_mm: 2, radbound: 1.8, maxvol: 10 },
  },
];

interface Props {
  node: JourneyNode;
  state: NodeState;
  /** True when this step's work happens in the 3-D view, which sits behind. */
  usesViewer: boolean;
  figure?: FigureInfo;

  config: Cfg | null;
  onConfigChange: (c: Cfg) => void;

  meshes: MeshInfo[];
  visible: Record<string, boolean>;
  onVisibleChange: (v: Record<string, boolean>) => void;
  target: string | null;
  onTargetChange: (t: string | null) => void;

  sources: PointSource[];
  onSourcesChange: (s: PointSource[]) => void;
  selectedSource: number | null;
  onSelectSource: (i: number | null) => void;

  threshold: number;
  onThresholdChange: (n: number) => void;
  modality: "meg" | "eeg";
  onModalityChange: (m: "meg" | "eeg") => void;

  result: DetectResult | null;
  unavailable: DetectUnavailable | null;

  /** True while any run is in flight, so a step can't start a second one. */
  running: boolean;
  /** Run only these stages and return to the canvas to watch them. */
  onRunStage: (stages: string[]) => void;
  /** Hand a sensor cloud to the 3-D well behind this step, or clear it. */
  onSensorCloud: (
    cloud: { positions: [number, number, number][]; values?: number[] } | null,
  ) => void;
  onOpenAdvanced: () => void;
  /** Persist the config and return to the journey. Resolves to any errors. */
  onSave: () => Promise<string[]>;
  onBack: () => void;
}

/* One labelled number, with its unit and a line saying what it does. */
function NumField({
  label,
  unit,
  help,
  path,
  config,
  onChange,
}: {
  label: string;
  unit?: string;
  help: string;
  path: string;
  config: Cfg;
  onChange: (c: Cfg) => void;
}) {
  const value = getPath<number>(config, path, 0);
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="stepfield">
      <div className="stepfield-head">
        <label htmlFor={path}>
          {label}
          {unit && <span className="stepfield-unit"> ({unit})</span>}
        </label>
        <input
          id={path}
          className="ui-input mono stepfield-input"
          inputMode="decimal"
          value={draft ?? String(Number.isFinite(value) ? value : 0)}
          onChange={(e) => {
            const v = e.currentTarget.value;
            setDraft(v);
            const n = Number(v);
            if (v.trim() !== "" && Number.isFinite(n)) onChange(setPath(config, path, n));
          }}
          onBlur={() => setDraft(null)}
        />
      </div>
      <p className="stepfield-help">{help}</p>
    </div>
  );
}

export default function StepView(p: Props) {
  const { node, config } = p;
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const save = async () => {
    setSaving(true);
    const errs = await p.onSave();
    setSaving(false);
    setErrors(errs);
    if (errs.length === 0) p.onBack();
  };

  // Tissues in the FEM mesh — the set every later step works from.
  const femTissues = config ? getPath<string[]>(config, "fem.tissues", []) : [];
  const conductivities = config
    ? getPath<Record<string, number>>(config, "forward.conductivities_sm", {})
    : {};

  const toggleTissue = (name: string) => {
    if (!config) return;
    const on = femTissues.includes(name);
    const next = on
      ? femTissues.filter((t) => t !== name)
      : [...femTissues, name];
    p.onConfigChange(setPath(config, "fem.tissues", next));
  };

  // Which steps actually own settings the pipeline will read. On the rest —
  // a rendered figure, the two analytic rungs — "Save and return" was the
  // green primary calling putConfig on an unchanged config, while the real
  // action sat below it as a quiet link. So those steps just close.
  const WRITES_CONFIG = new Set([
    "anatomy",
    "conductivity",
    "mesh",
    "sources",
    "sensors",
    "eeg",
    "noise",
    "anisotropy",
    "solve",
    "detect",
  ]);
  const writesConfig = WRITES_CONFIG.has(node.kind);

  const activeMeshPreset = MESH_PRESETS.find(
    (m) =>
      config &&
      getPath<number>(config, "fem.pitch_mm", 0) === m.values.pitch_mm &&
      getPath<number>(config, "fem.radbound", 0) === m.values.radbound &&
      getPath<number>(config, "fem.maxvol", 0) === m.values.maxvol,
  );

  return (
    <div className={`jstep${p.usesViewer ? " jstep--viewer" : ""}`}>
      <header className="jstep-bar">
        <button type="button" className="jbtn jbtn--ghost" onClick={p.onBack}>
          <Icon name="arrow-left" size={14} /> Journey
        </button>
        <span className="jstep-crumb" aria-hidden>
          /
        </span>
        <span className="jstep-icon" aria-hidden>
          <Icon name={node.icon as IconName} size={15} />
        </span>
        <h1 className="jstep-title">{node.title}</h1>
        {/* Same glyphs as the canvas mark and the run HUD's step marks, so
            "done", "reused" and "failed" look the same wherever they appear. */}
        <span className={`jstate jstate--${p.state}`}>
          {STATE_ICON[p.state] && (
            <Icon name={STATE_ICON[p.state] as IconName} size={11} />
          )}
          {STATE_WORD[p.state]}
        </span>

        <span className="jstep-spacer" />
        {node.stage && (
          <button
            type="button"
            className="jbtn"
            disabled={p.running}
            title={`Run only this step (${node.stage})`}
            onClick={() => p.onRunStage([node.stage as string])}
          >
            <Icon name="play" size={13} /> Run this step
          </button>
        )}
        {node.kind === "figure" && p.figure?.exists && (
          <a
            className="jbtn"
            href={p.figure.url}
            target="_blank"
            rel="noreferrer"
          >
            <Icon name="external" size={13} /> Open full size
          </a>
        )}
        {writesConfig ? (
          <button type="button" className="jbtn jbtn--go" onClick={save} disabled={saving}>
            <Icon name="check" size={14} /> {saving ? "Saving…" : "Save and return"}
          </button>
        ) : (
          <button type="button" className="jbtn jbtn--go" onClick={p.onBack}>
            <Icon name="check" size={14} /> Done
          </button>
        )}
      </header>

      <div className="jstep-body">
        {p.usesViewer && <div className="jstep-viewslot" aria-hidden />}

        <section className="jstep-side">
          {errors.length > 0 && (
            <Note tone="danger" title="Not saved">
              <ul className="ui-errlist">
                {errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </Note>
          )}

          {/* ── modality: the first decision ──────────────────────────── */}
          {node.kind === "modality" && (
            <>
              <p className="jlead">
                What you record with. It decides which sensors are placed, which
                leadfield is solved, and which noise floor the answer is measured
                against — so it comes first.
              </p>
              <div className="jstack">
                {(
                  [
                    {
                      key: "meg" as const,
                      title: "MEG · OPM magnetometers",
                      note: "Triaxial optically-pumped sensors standing off the skin. No contact, no skin impedance.",
                    },
                    {
                      key: "eeg" as const,
                      title: "EEG · surface electrodes",
                      note: "Contacts on the skin. Cheaper and lighter, and blurred by the tissue between source and contact.",
                    },
                  ]
                ).map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    className={`pick${p.modality === m.key ? " pick--on" : ""}`}
                    aria-pressed={p.modality === m.key}
                    onClick={() => p.onModalityChange(m.key)}
                  >
                    <span className="pick-dot" aria-hidden />
                    <span className="pick-text">
                      <span className="pick-title">{m.title}</span>
                      <span className="pick-note">{m.note}</span>
                    </span>
                  </button>
                ))}
              </div>

              {config && p.modality === "eeg" && (
                <>
                  <h2 className="jsub">Electrode patch</h2>
                  <NumField
                    label="Rows"
                    path="electrodes.rows"
                    config={config}
                    onChange={p.onConfigChange}
                    help="Contacts along the body axis."
                  />
                  <NumField
                    label="Columns"
                    path="electrodes.cols"
                    config={config}
                    onChange={p.onConfigChange}
                    help="Contacts around the circumference."
                  />
                  <NumField
                    label="Contact pitch"
                    unit="mm"
                    path="electrodes.contact_pitch_mm"
                    config={config}
                    onChange={p.onConfigChange}
                    help="Centre-to-centre spacing within the patch."
                  />
                </>
              )}
            </>
          )}

          {/* ── anatomy: which tissues are in the model ────────────────── */}
          {node.kind === "anatomy" && (
            <>
              <p className="jlead">
                Pick the tissues that go into the model, and the one you are
                trying to image. Sources snap to the target in the 3-D view.
              </p>

              <label className="jfield">
                <span>Imaging target</span>
                <Select
                  value={p.target ?? ""}
                  disabled={p.meshes.length === 0}
                  ariaLabel="Imaging target"
                  onChange={(v) => p.onTargetChange(v || null)}
                  options={p.meshes.map((m) => ({
                    value: m.name,
                    label: m.name.replace(/_/g, " "),
                  }))}
                />
              </label>

              <h2 className="jsub">Tissues in the mesh</h2>
              <p className="stepfield-help" style={{ marginBottom: 10 }}>
                Each tissue included here becomes its own conductive compartment.
                Leaving one out makes it part of the surrounding tissue.
              </p>
              <div className="jstack">
                {p.meshes.map((m) => {
                  const inMesh = femTissues.includes(m.name);
                  const shown = p.visible[m.name] !== false;
                  return (
                    <div key={m.name} className={`tissue${inMesh ? " tissue--on" : ""}`}>
                      <button
                        type="button"
                        className="tissue-main"
                        aria-pressed={inMesh}
                        onClick={() => toggleTissue(m.name)}
                        disabled={!config}
                      >
                        <span className="tissue-box" aria-hidden>
                          {inMesh && <Icon name="check" size={11} />}
                        </span>
                        <span className="tissue-name">{m.name.replace(/_/g, " ")}</span>
                        {inMesh && (
                          <span className="tissue-sigma mono">
                            {conductivities[m.name] ?? "—"} S/m
                          </span>
                        )}
                      </button>
                      <button
                        type="button"
                        className="tissue-eye"
                        aria-label={`${shown ? "Hide" : "Show"} ${m.name} in the 3-D view`}
                        onClick={() =>
                          p.onVisibleChange({ ...p.visible, [m.name]: !shown })
                        }
                      >
                        <Icon name={shown ? "eye" : "eye-off"} size={13} />
                      </button>
                    </div>
                  );
                })}
                {p.meshes.length === 0 && (
                  <span className="jempty">
                    No anatomy loaded — check the backend is running.
                  </span>
                )}
              </div>
            </>
          )}

          {/* ── conductivities ─────────────────────────────────────────── */}
          {node.kind === "conductivity" && (
            <>
              <p className="jlead">
                How well each tissue carries current, in siemens per metre. These
                set the volume-conductor the solver sees — the field at the
                sensors depends on them as much as on the source.
              </p>
              {!config && (
                <Note tone="warn" title="No config loaded">
                  The backend hasn’t sent its config yet. It’s retried
                  automatically — start it on :8000 and this fills in.
                </Note>
              )}
              {config &&
                Object.keys(conductivities).length === 0 && (
                  <Note tone="warn">No conductivities are defined in this config.</Note>
                )}
              {config &&
                Object.keys(conductivities)
                  .sort()
                  .map((tissue) => (
                    <NumField
                      key={tissue}
                      label={tissue.replace(/_/g, " ")}
                      unit="S/m"
                      path={`forward.conductivities_sm.${tissue}`}
                      config={config}
                      onChange={p.onConfigChange}
                      help={
                        femTissues.includes(tissue)
                          ? "In the mesh — this value is used."
                          : "Not currently meshed, so this value is unused."
                      }
                    />
                  ))}
            </>
          )}

          {/* ── FEM meshing ────────────────────────────────────────────── */}
          {node.kind === "mesh" && config && (
            <>
              <p className="jlead">
                The tissue surfaces are cut into tetrahedra. Smaller elements
                resolve thin structures better and cost solve time and memory.
              </p>

              <h2 className="jsub">Element size</h2>
              <div className="jstack">
                {MESH_PRESETS.map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    className={`pick${activeMeshPreset?.key === m.key ? " pick--on" : ""}`}
                    onClick={() => {
                      let next = config;
                      for (const [k, v] of Object.entries(m.values)) {
                        next = setPath(next, `fem.${k}`, v);
                      }
                      p.onConfigChange(next);
                    }}
                  >
                    <span className="pick-dot" aria-hidden />
                    <span className="pick-text">
                      <span className="pick-title">{m.label}</span>
                      <span className="pick-note">{m.note}</span>
                    </span>
                  </button>
                ))}
              </div>

              <h2 className="jsub">Exact values</h2>
              <NumField
                label="Voxel pitch"
                unit="mm"
                path="fem.pitch_mm"
                config={config}
                onChange={p.onConfigChange}
                help="Grid step used to rasterise the surfaces before meshing."
              />
              <NumField
                label="Surface bound"
                unit="mm"
                path="fem.radbound"
                config={config}
                onChange={p.onConfigChange}
                help="Largest triangle allowed on a tissue boundary."
              />
              <NumField
                label="Max element volume"
                unit="mm³"
                path="fem.maxvol"
                config={config}
                onChange={p.onConfigChange}
                help="Ceiling on a single tetrahedron. Lower means more elements."
              />
              <NumField
                label="Padding around the body"
                unit="mm"
                path="fem.pad_mm"
                config={config}
                onChange={p.onConfigChange}
                help="Air margin so the outer boundary is far from the sources."
              />
            </>
          )}

          {/* ── sources: what fires, and where along the spine ────────── */}
          {node.kind === "sources" && (
            <SourcesStep
              config={config}
              onConfigChange={p.onConfigChange}
              femTissues={femTissues}
              meshes={p.meshes}
              target={p.target}
              sources={p.sources}
              onSourcesChange={p.onSourcesChange}
              selectedSource={p.selectedSource}
              onSelectSource={p.onSelectSource}
            />
          )}

          {/* ── sensor array ───────────────────────────────────────────── */}
          {node.kind === "sensors" && config && (
            <>
              <SensorArrayPreview modality={p.modality} onCloud={p.onSensorCloud} />
              <p className="jlead">
                Where the sensors sit on the body. You are solving{" "}
                <b>{p.modality.toUpperCase()}</b> — change that in the Modality
                step at the head of the journey.
              </p>

              <h2 className="jsub">OPM array</h2>
              <NumField
                label="Sensor spacing"
                unit="mm"
                path="sensors.resolution_mm"
                config={config}
                onChange={p.onConfigChange}
                help="Distance between neighbouring sensors over the body surface."
              />
              <NumField
                label="Stand-off"
                unit="mm"
                path="sensors.depth_mm"
                config={config}
                onChange={p.onConfigChange}
                help="Gap from skin to sensor. Field falls off fast, so this matters."
              />
              <NumField
                label="Angular coverage"
                unit="°"
                path="sensors.angular_margin_deg"
                config={config}
                onChange={p.onConfigChange}
                help="How far around the body the array wraps from the target."
              />
              {p.result && (
                <Tag tone="signal">{p.result.array.n_sensors} sensors in the last solve</Tag>
              )}
            </>
          )}

          {/* ── field map: the topography, live ───────────────────────── */}
          {node.kind === "fieldmap" && (
            <FieldMapStep
              modality={p.modality}
              sourceCount={p.result?.per_source.length ?? 0}
              onCloud={p.onSensorCloud}
            />
          )}

          {/* ── analytic rungs (optional nodes) ────────────────────────── */}
          {(node.kind === "biot" || node.kind === "sarvas") && (
            <LadderRungStep kind={node.kind} />
          )}

          {/* ── forward solve ──────────────────────────────────────────── */}
          {node.kind === "solve" && (
            <>
              <p className="jlead">
                DUNEuro solves the leadfield: what every sensor reads for a unit
                current at each source. This is the expensive stage.
              </p>
              <Tag>{p.modality === "eeg" ? "solving EEG" : "solving MEG"}</Tag>
              {config ? (
                <div style={{ marginTop: 14 }}>
                  <ComputePanel config={config} onChange={p.onConfigChange} />
                  <NumField
                    label="Solver tolerance"
                    path="forward.solver.reduction"
                    config={config}
                    onChange={p.onConfigChange}
                    help="Residual reduction the iterative solver must reach. Smaller is stricter and slower."
                  />
                </div>
              ) : (
                <Note tone="warn" title="No config loaded">
                  The backend hasn’t sent its config yet. It’s retried
                  automatically — start it on :8000 and this fills in.
                </Note>
              )}
              <Button variant="ghost" icon="cog" onClick={p.onOpenAdvanced}>
                Solver engine and cluster
              </Button>
            </>
          )}

          {/* ── the answer ─────────────────────────────────────────────── */}
          {node.kind === "detect" && (
            <>
              <label className="jfield">
                <span>Detection confidence</span>
                <Select
                  value={p.threshold}
                  ariaLabel="Detection confidence"
                  onChange={(v) => p.onThresholdChange(Number(v))}
                  options={THRESHOLDS}
                />
              </label>
              <div aria-live="polite">
                <ResultsPanel result={p.result} unavailable={p.unavailable} />
              </div>
            </>
          )}

          {/* ── a figure ───────────────────────────────────────────────── */}
          {node.kind === "figure" && (
            <>
              {p.figure?.exists ? (
                /* "Open full size" is the real verb of this step, so it lives
                   in the bar with the other actions rather than as a link
                   underneath a byte count. */
                <p className="jmeta mono">
                  {(p.figure.bytes / 1024).toFixed(0)} kB · drawn{" "}
                  {new Date(p.figure.mtime * 1000).toLocaleString()}
                </p>
              ) : (
                <p className="jlead">
                  Not drawn yet. Run the journey and this fills in when the{" "}
                  {node.stage} stage writes it.
                </p>
              )}
            </>
          )}
        </section>

        {node.kind === "figure" && p.figure?.exists && (
          <section className="jstep-main jstep-main--figure">
            <FigureView figure={p.figure} />
          </section>
        )}

        {/* The answer, drawn rather than listed: one bar per source, the
            detection threshold marked, so "which source is hard" is a glance. */}
        {node.kind === "detect" && p.result && (
          <section className="jstep-main jstep-main--chart">
            <div className="chart-wrap">
              <StatTile
                value={detectHeadline(p.result)}
                unit="averaged trials to detect"
                note={`${p.result.array.n_sensors} ${p.result.modality.toUpperCase()} sensors · noise floor ${p.result.array.noise_floor_fT} ${p.result.array.noise_unit ?? "fT"}`}
              />

              <h2 className="chart-title">Trials needed, per source</h2>
              <BarChart
                unit="trials"
                bars={p.result.per_source.map((src) => ({
                  label: `source ${src.index + 1}`,
                  values: [src.trials_needed],
                  display: [
                    src.trials_needed >= 0 && Number.isFinite(src.trials_needed)
                      ? src.trials_needed.toLocaleString()
                      : "never reaches threshold",
                  ],
                }))}
                series={["trials"]}
              />

              <h2 className="chart-title">Single-trial SNR</h2>
              <BarChart
                bars={p.result.per_source.map((src) => ({
                  label: `source ${src.index + 1}`,
                  values: [src.snr],
                  display: [src.snr.toFixed(2)],
                }))}
                series={["SNR"]}
                reference={{
                  value: p.result.array.threshold_snr,
                  label: `threshold ${p.result.array.threshold_snr}`,
                }}
              />
            </div>
          </section>
        )}

        {!p.usesViewer &&
          node.kind !== "figure" &&
          !(node.kind === "detect" && p.result) &&
          !(node.kind === "biot" || node.kind === "sarvas") && (
            <section className="jstep-main jstep-main--quiet">
              <div className="jstep-mark" aria-hidden>
                <Icon name={node.icon as IconName} size={72} />
              </div>
              <p className="jstep-caption">{node.caption}</p>
            </section>
          )}
      </div>
    </div>
  );
}

/* Sources: the step that decides what fires and where.
 *
 * A vertebral level is not decoration here — it *is* the placement. Saying
 * "C7" means the sources sit inside the source tissue within that vertebra's
 * own measured Z band, and the electrode patch is centred on the same level,
 * so the study and the array are describing one place rather than two. The
 * points come back from the mesh as tetrahedron centroids, which is what makes
 * them solvable: a click that lands a millimetre outside the volume is the
 * commonest way a first run fails.
 */
function SourcesStep({
  config,
  onConfigChange,
  femTissues,
  meshes,
  target,
  sources,
  onSourcesChange,
  selectedSource,
  onSelectSource,
}: {
  config: Cfg | null;
  onConfigChange: (c: Cfg) => void;
  femTissues: string[];
  meshes: MeshInfo[];
  target: string | null;
  sources: PointSource[];
  onSourcesChange: (s: PointSource[]) => void;
  selectedSource: number | null;
  onSelectSource: (i: number | null) => void;
}) {
  const [levels, setLevels] = useState<LevelInfo[]>([]);
  const [count, setCount] = useState(3);
  const [busy, setBusy] = useState(false);
  const [placed, setPlaced] = useState<SuggestedSources | null>(null);
  const [error, setError] = useState<{ msg: string; hint?: string } | null>(null);

  useEffect(() => {
    getLevels().then(setLevels);
  }, []);

  const tissue = config
    ? getPath<string>(config, "forward.source_tissue", "") ||
      (femTissues[0] ?? "")
    : "";
  const level = config
    ? getPath<string>(config, "electrodes.target_level", "") || ""
    : "";

  const setLevel = (next: string | null) => {
    if (!config) return;
    onConfigChange(setPath(config, "electrodes.target_level", next));
    setPlaced(null);
    setError(null);
  };

  const place = async () => {
    setBusy(true);
    setError(null);
    const { result, error: err, hint } = await suggestSources(
      tissue,
      level || null,
      count,
    );
    if (result) {
      setPlaced(result);
      onSourcesChange(
        result.sources.map((pt) => ({ ...pt, strength_nAm: 70 })),
      );
      onSelectSource(null);
    } else if (err) {
      setError({ msg: err, hint });
    }
    setBusy(false);
  };

  const bands = new Map(levels.map((l) => [l.level, l]));
  const chosen = level ? bands.get(level) : undefined;

  return (
    <>
      <p className="jlead">
        Click <b>{target?.replace(/_/g, " ") ?? "the target"}</b> in the 3-D view
        to drop a source, or place a set along a vertebral level below. Each
        source is solved independently.
      </p>

      {config && (
        <label className="jfield">
          <span>Source tissue</span>
          <Select
            value={tissue}
            ariaLabel="Source tissue"
            onChange={(v) => onConfigChange(setPath(config, "forward.source_tissue", v))}
            options={(femTissues.length ? femTissues : meshes.map((m) => m.name)).map(
              (t) => ({ value: t, label: t.replace(/_/g, " ") }),
            )}
          />
        </label>
      )}

      <h2 className="jsub">Vertebral level</h2>
      <p className="stepfield-help" style={{ marginBottom: 10 }}>
        The level places the sources and centres the electrode patch. Each band
        is measured from that vertebra&rsquo;s own segmented STL, so it follows
        this anatomy rather than an assumed proportion.
      </p>

      <div className="levels">
        <div className="levels-group">
          <button
            type="button"
            className={`level${!level ? " level--on" : ""}`}
            aria-pressed={!level}
            onClick={() => setLevel(null)}
          >
            any
          </button>
          <span className="levels-any-note">whole length of the tissue</span>
        </div>
        {VERTEBRAE.map((group) => {
          const available = group.levels.filter((l) => bands.has(l));
          if (available.length === 0) return null;
          return (
            <div key={group.name} className="levels-group">
              <span className="levels-group-name">{group.name}</span>
              {available.map((lvl) => (
                <button
                  key={lvl}
                  type="button"
                  className={`level${level === lvl ? " level--on" : ""}`}
                  aria-pressed={level === lvl}
                  title={`${lvl.toUpperCase()} · ${bands.get(lvl)!.z_lo_mm}–${bands.get(lvl)!.z_hi_mm} mm`}
                  onClick={() => setLevel(lvl)}
                >
                  {lvl.toUpperCase()}
                </button>
              ))}
            </div>
          );
        })}
      </div>

      {chosen && (
        <p className="levels-band mono">
          {chosen.level.toUpperCase()} spans {chosen.z_lo_mm}–{chosen.z_hi_mm} mm
        </p>
      )}

      <div className="levels-place">
        <label className="levels-count">
          <span>How many</span>
          <Select
            value={count}
            ariaLabel="How many sources"
            onChange={(v) => setCount(Number(v))}
            options={[1, 2, 3, 5, 8].map((n) => ({ value: n, label: String(n) }))}
          />
        </label>
        <Button
          variant="primary"
          icon="bolt"
          loading={busy}
          disabled={!config || !tissue}
          onClick={place}
        >
          Place sources
        </Button>
      </div>

      {error && (
        <Note tone="warn" title="Could not place sources">
          <p>{error.msg}</p>
          {error.hint && <p className="ui-dim">{error.hint}</p>}
        </Note>
      )}
      {placed && !error && (
        <Note tone="ok">
          {placed.sources.length} source
          {placed.sources.length === 1 ? "" : "s"} placed inside{" "}
          {placed.tissue.replace(/_/g, " ")}
          {placed.level ? ` at ${placed.level.toUpperCase()}` : ""}, chosen from{" "}
          {placed.available} tetrahedra. Every one is inside the mesh, so the
          solve will evaluate them.
        </Note>
      )}

      {config && (
        <NumField
          label="Source spacing"
          unit="mm"
          path="forward.source_spacing_mm"
          config={config}
          onChange={onConfigChange}
          help="Spacing of the dipole grid laid along the source tissue."
        />
      )}

      <h2 className="jsub">Placed sources — {sources.length}</h2>
      <SourceList
        sources={sources}
        selected={selectedSource}
        onSelect={onSelectSource}
        onChange={onSourcesChange}
      />
    </>
  );
}

/* The headline: the easiest source to detect, since that is the number a
   planner acts on. Sources that never reach threshold are excluded from it —
   the per-source chart below says which those are. */
function detectHeadline(result: DetectResult): string {
  const trials = result.per_source
    .map((s) => s.trials_needed)
    .filter((n) => n >= 0 && Number.isFinite(n));
  if (trials.length === 0) return "∞";
  return Math.min(...trials).toLocaleString();
}

/* The array itself, drawn in the well rather than rendered to a PNG: every
   sensor that was placed, where it sits on the body. */
function SensorArrayPreview({
  modality,
  onCloud,
}: {
  modality: "meg" | "eeg";
  onCloud: (
    cloud: { positions: [number, number, number][]; values?: number[] } | null,
  ) => void;
}) {
  const [info, setInfo] = useState<SensorArrayInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getSensorArray(modality).then(({ result, error: err }) => {
      if (!live) return;
      if (result) {
        setInfo(result);
        setError(null);
        onCloud({ positions: result.positions });
      } else {
        setInfo(null);
        setError(err ?? "not built yet");
        onCloud(null);
      }
    });
    return () => {
      live = false;
    };
  }, [modality, onCloud]);

  useEffect(() => () => onCloud(null), [onCloud]);

  if (error) {
    return (
      <Note tone="warn">
        No {modality.toUpperCase()} array on disk yet — run this step and it
        appears in the view.
      </Note>
    );
  }
  if (!info) return null;
  return (
    <Note tone="ok">
      {info.count} {info.types.join(", ")} channels, drawn in the view. Drag to
      orbit.
    </Note>
  );
}

/* The field map.
 *
 * The same topography the PNG topoplot draws, except the numbers come back per
 * channel and are painted onto the array in the 3-D well — so it can be
 * orbited, and a hot spot can be traced to the sensor that reads it. Nothing is
 * rendered server-side.
 */
function FieldMapStep({
  modality,
  sourceCount,
  onCloud,
}: {
  modality: "meg" | "eeg";
  sourceCount: number;
  onCloud: (
    cloud: { positions: [number, number, number][]; values?: number[] } | null,
  ) => void;
}) {
  const [source, setSource] = useState(0);
  const [map, setMap] = useState<FieldMap | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ msg: string; hint?: string } | null>(null);

  useEffect(() => {
    let live = true;
    setBusy(true);
    setError(null);
    getFieldMap(source, modality).then(({ result, error: err, hint }) => {
      if (!live) return;
      if (result) {
        setMap(result);
        onCloud({ positions: result.positions, values: result.values });
      } else {
        setMap(null);
        onCloud(null);
        // No leadfield yet is the ordinary case before a solve, not a fault.
        setError({ msg: err ?? "could not be read", hint });
      }
      setBusy(false);
    });
    return () => {
      live = false;
    };
  }, [source, modality, onCloud]);

  // The cloud belongs to this step; leaving it should not leave the well
  // painted with a map the next step knows nothing about.
  useEffect(() => () => onCloud(null), [onCloud]);

  return (
    <>
      <p className="jlead">
        What every sensor reads for one source, painted onto the array in the
        view. Drag to orbit it — brighter is a stronger reading.
      </p>

      {sourceCount > 1 && (
        <label className="jfield">
          <span>Source</span>
          <Select
            value={source}
            ariaLabel="Which source"
            onChange={(v) => setSource(Number(v))}
            options={Array.from({ length: sourceCount }, (_, i) => ({
              value: i,
              label: `source ${i + 1}`,
            }))}
          />
        </label>
      )}

      {busy && <p className="ui-dim">reading the leadfield…</p>}

      {error && !busy && (
        <Note tone="warn" title="No field map yet">
          <p>{error.msg}</p>
          {error.hint && <p className="ui-dim">{error.hint}</p>}
        </Note>
      )}

      {map && !busy && (
        <>
          <StatTile value={map.peak.toFixed(2)} unit={`${map.unit}, peak`} />
          <div className="fieldscale">
            <span className="fieldscale-bar" aria-hidden />
            <span className="fieldscale-ends mono">
              <span>0</span>
              <span>
                {map.peak.toFixed(1)} {map.unit.split(" ")[0]}
              </span>
            </span>
          </div>
          <dl className="fieldfacts mono">
            <div>
              <dt>channels</dt>
              <dd>{map.count}</dd>
            </div>
            <div>
              <dt>RMS</dt>
              <dd>{map.rms.toFixed(2)}</dd>
            </div>
            <div>
              <dt>source</dt>
              <dd>{map.source_pos.map((v) => v.toFixed(1)).join(", ")} mm</dd>
            </div>
          </dl>
        </>
      )}
    </>
  );
}

/* An analytic rung, run on demand. Cheap enough to run right here — neither
   rung needs DUNEuro or a leadfield. */
function LadderRungStep({ kind }: { kind: "biot" | "sarvas" }) {
  const rung: LadderRung = kind === "biot" ? "biot" : "sarvas";
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<LadderResult | null>(null);
  const [err, setErr] = useState<{ msg: string; hint?: string } | null>(null);

  const run = async () => {
    setBusy(true);
    setErr(null);
    const { result, error, hint } = await runLadder([rung]);
    if (result) setRes(result);
    else if (error) setErr({ msg: error, hint });
    setBusy(false);
  };

  const value = res?.rungs?.[rung];

  return (
    <>
      <p className="jlead">
        {kind === "biot"
          ? "The primary current in free space — no volume conductor at all. The floor every other model is compared against."
          : "A single homogeneous sphere, solved analytically. Adding the sphere's return currents to Biot–Savart shows what the volume conductor does before any real geometry is involved."}
      </p>
      <Button variant="primary" icon="play" loading={busy} onClick={run}>
        Run this rung
      </Button>

      {err && (
        <Note tone="warn" title="Could not run">
          <p>{err.msg}</p>
          {err.hint && <p className="ui-dim">{err.hint}</p>}
        </Note>
      )}

      {value && (
        <>
          <StatTile
            value={value.peak_fT_per_nAm.toFixed(2)}
            unit="fT per nA·m, peak"
            note={`RMS ${value.rms_fT_per_nAm.toFixed(2)} across ${res?.n_radial_coils} radial coils`}
          />
          <h2 className="chart-title">Peak against RMS</h2>
          <BarChart
            unit="fT/nA·m"
            series={["peak", "RMS"]}
            bars={[
              {
                label: kind === "biot" ? "Biot–Savart" : "Sarvas",
                values: [value.peak_fT_per_nAm, value.rms_fT_per_nAm],
                display: [
                  value.peak_fT_per_nAm.toFixed(2),
                  value.rms_fT_per_nAm.toFixed(2),
                ],
              },
            ]}
          />
        </>
      )}
    </>
  );
}
