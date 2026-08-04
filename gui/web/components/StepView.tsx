"use client";

import { useState } from "react";
import { Button, Icon, Note, Select, Tag } from "@/components/ui";
import type { IconName } from "@/components/ui/Icon";
import { getPath, setPath, type Cfg } from "@/lib/config";
import { runLadder, type LadderResult, type LadderRung } from "@/lib/api";
import type {
  DetectResult,
  DetectUnavailable,
  FigureInfo,
  MeshInfo,
  PointSource,
} from "@/lib/api";
import type { JourneyNode, NodeState } from "@/lib/journey";
import SourceList from "@/components/SourceList";
import ResultsPanel from "@/components/ResultsPanel";
import ComputePanel from "@/components/ComputePanel";

const STATE_WORD: Record<NodeState, string> = {
  idle: "not run yet",
  ready: "ready",
  active: "running now",
  done: "done",
  skipped: "reused",
  failed: "failed",
};

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
        <span className={`jstate jstate--${p.state}`}>{STATE_WORD[p.state]}</span>

        <span className="jstep-spacer" />
        <button type="button" className="jbtn jbtn--go" onClick={save} disabled={saving}>
          <Icon name="check" size={14} /> {saving ? "Saving…" : "Save and return"}
        </button>
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
              {!config && <span className="jempty">Waiting for the backend config.</span>}
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

          {/* ── sources ────────────────────────────────────────────────── */}
          {node.kind === "sources" && (
            <>
              <p className="jlead">
                Click <b>{p.target?.replace(/_/g, " ") ?? "the target"}</b> in the
                3-D view to drop a current source where nerve activity might
                occur. Each one is solved independently.
              </p>

              {config && (
                <>
                  <label className="jfield">
                    <span>Source tissue</span>
                    <Select
                      value={getPath<string>(config, "forward.source_tissue", "")}
                      ariaLabel="Source tissue"
                      onChange={(v) =>
                        p.onConfigChange(setPath(config, "forward.source_tissue", v))
                      }
                      options={(femTissues.length ? femTissues : p.meshes.map((m) => m.name)).map(
                        (t) => ({ value: t, label: t.replace(/_/g, " ") }),
                      )}
                    />
                  </label>
                  <NumField
                    label="Source spacing"
                    unit="mm"
                    path="forward.source_spacing_mm"
                    config={config}
                    onChange={p.onConfigChange}
                    help="Spacing of the dipole grid laid along the source tissue."
                  />
                </>
              )}

              <h2 className="jsub">Placed sources — {p.sources.length}</h2>
              <SourceList
                sources={p.sources}
                selected={p.selectedSource}
                onSelect={p.onSelectSource}
                onChange={p.onSourcesChange}
              />
            </>
          )}

          {/* ── sensor array ───────────────────────────────────────────── */}
          {node.kind === "sensors" && config && (
            <>
              <p className="jlead">
                Where the sensors sit on the body, and what kind they are. The
                solve reports whichever modality is selected here.
              </p>

              <h2 className="jsub">Modality</h2>
              <div className="jstack">
                {(
                  [
                    {
                      key: "meg" as const,
                      title: "OPM magnetometers",
                      note: "Triaxial optically-pumped sensors, standing off the skin.",
                    },
                    {
                      key: "eeg" as const,
                      title: "EEG electrodes",
                      note: "Contacts on the surface. Needs the electrode array built.",
                    },
                  ]
                ).map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    className={`pick${p.modality === m.key ? " pick--on" : ""}`}
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

          {/* ── EEG electrodes (optional node) ─────────────────────────── */}
          {node.kind === "eeg" && config && (
            <>
              <p className="jlead">
                A contact array on the body surface, solved instead of the OPMs.
                Selecting EEG here is what makes the next run an EEG run.
              </p>
              <div className="jstack">
                <button
                  type="button"
                  className={`pick${p.modality === "eeg" ? " pick--on" : ""}`}
                  onClick={() => p.onModalityChange("eeg")}
                >
                  <span className="pick-dot" aria-hidden />
                  <span className="pick-text">
                    <span className="pick-title">Solve EEG</span>
                    <span className="pick-note">
                      {p.modality === "eeg"
                        ? "The next run reports electrode potentials."
                        : "Currently solving OPM magnetometers instead."}
                    </span>
                  </span>
                </button>
              </div>

              <h2 className="jsub">Contact layout</h2>
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
                help="Centre-to-centre spacing within the paddle."
              />
            </>
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
                <span className="jempty">Waiting for the backend config.</span>
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
                <>
                  <p className="jmeta mono">
                    {(p.figure.bytes / 1024).toFixed(0)} kB · drawn{" "}
                    {new Date(p.figure.mtime * 1000).toLocaleString()}
                  </p>
                  <a className="jlink" href={p.figure.url} target="_blank" rel="noreferrer">
                    Open full size
                  </a>
                </>
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
          <section className="jstep-main">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              className="jfigure"
              src={`${p.figure.url}?v=${Math.round(p.figure.mtime)}`}
              alt={p.figure.label}
            />
          </section>
        )}

        {!p.usesViewer && node.kind !== "figure" && (
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
          <div className="result-headline">
            <div className="result-number">{value.peak_fT_per_nAm.toFixed(2)}</div>
            <div className="result-unit">fT per nA·m, peak</div>
          </div>
          <p className="result-verdict">
            RMS across the array is {value.rms_fT_per_nAm.toFixed(2)} fT per nA·m,
            over {res?.n_radial_coils} radial coils.
          </p>
        </>
      )}
    </>
  );
}
