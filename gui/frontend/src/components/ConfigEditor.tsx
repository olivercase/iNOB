import { useEffect, useMemo, useState } from "react";
import {
  Button,
  Callout,
  Card,
  Collapse,
  Divider,
  FormGroup,
  H4,
  H5,
  HTMLSelect,
  InputGroup,
  Intent,
  NumericInput,
  Spinner,
  Switch,
  TextArea,
} from "@blueprintjs/core";
import {
  ConfigDict,
  getConfig,
  resetConfig,
  saveConfig,
  validateConfig,
} from "../api";
import { AppToaster } from "../toaster";

type Dict = Record<string, unknown>;

function asDict(v: unknown): Dict {
  return v && typeof v === "object" ? (v as Dict) : {};
}

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" ? v : fallback;
}

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

// Immutably set a nested path on the config object.
function setPath(root: ConfigDict, path: (string | number)[], value: unknown): ConfigDict {
  const clone: Dict = Array.isArray(root) ? [...(root as unknown[])] as unknown as Dict : { ...root };
  let cur: Dict = clone;
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i];
    const next = cur[key];
    cur[key] = Array.isArray(next) ? [...(next as unknown[])] : { ...asDict(next) };
    cur = cur[key] as Dict;
  }
  cur[path[path.length - 1]] = value;
  return clone;
}

interface Props {
  onConfigChange: (cfg: ConfigDict) => void;
}

const RAW_SECTIONS = [
  "outputs",
  "data",
  "noise",
  "cluster",
  "sensitivity",
  "reproducibility",
  "geometry",
];

export default function ConfigEditor({ onConfigChange }: Props) {
  const [config, setConfig] = useState<ConfigDict | null>(null);
  const [loadErrors, setLoadErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [openRaw, setOpenRaw] = useState<Record<string, boolean>>({});

  useEffect(() => {
    getConfig()
      .then((r) => {
        setConfig(r.config);
        setLoadErrors(r.errors);
        onConfigChange(r.config);
      })
      .catch((e: unknown) => setLoadErrors([String(e)]));
  }, [onConfigChange]);

  function update(path: (string | number)[], value: unknown) {
    setConfig((prev) => {
      if (!prev) return prev;
      const next = setPath(prev, path, value);
      onConfigChange(next);
      return next;
    });
  }

  const forward = useMemo(() => asDict(config?.forward), [config]);
  const conductivities = useMemo(
    () => asDict(forward.conductivities_sm),
    [forward]
  );
  const tissueNames = Object.keys(conductivities);
  const solver = asDict(forward.solver);
  const sensors = asDict(config?.sensors);
  const electrodes = asDict(config?.electrodes);
  const fem = asDict(config?.fem);

  async function onValidate() {
    if (!config) return;
    setBusy(true);
    try {
      const r = await validateConfig(config);
      const t = await AppToaster;
      if (r.valid) t.show({ intent: Intent.SUCCESS, message: "Config is valid." });
      else
        t.show({
          intent: Intent.WARNING,
          message: `Invalid: ${r.errors.join("; ")}`,
        });
    } catch (e) {
      (await AppToaster).show({ intent: Intent.DANGER, message: String(e) });
    } finally {
      setBusy(false);
    }
  }

  async function onSave() {
    if (!config) return;
    setBusy(true);
    try {
      const r = await saveConfig(config);
      const t = await AppToaster;
      if (r.saved)
        t.show({ intent: Intent.SUCCESS, message: "Config saved." });
      else
        t.show({
          intent: Intent.DANGER,
          message: `Save rejected: ${r.errors.join("; ")}`,
        });
    } catch (e) {
      (await AppToaster).show({ intent: Intent.DANGER, message: String(e) });
    } finally {
      setBusy(false);
    }
  }

  async function onReset() {
    setBusy(true);
    try {
      const r = await resetConfig();
      setConfig(r.config);
      onConfigChange(r.config);
      (await AppToaster).show({
        intent: Intent.PRIMARY,
        message: "Reverted to defaults.",
      });
    } catch (e) {
      (await AppToaster).show({ intent: Intent.DANGER, message: String(e) });
    } finally {
      setBusy(false);
    }
  }

  if (!config) {
    return (
      <div style={{ padding: 24, textAlign: "center" }}>
        <Spinner />
        {loadErrors.length > 0 && (
          <Callout intent={Intent.DANGER} style={{ marginTop: 12 }}>
            {loadErrors.join("; ")}
          </Callout>
        )}
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <Button icon="tick" onClick={onValidate} loading={busy}>
          Validate
        </Button>
        <Button icon="floppy-disk" intent={Intent.PRIMARY} onClick={onSave} loading={busy}>
          Save
        </Button>
        <Button icon="reset" onClick={onReset} loading={busy}>
          Reset
        </Button>
      </div>

      {loadErrors.length > 0 && (
        <Callout intent={Intent.WARNING} style={{ marginBottom: 12 }}>
          {loadErrors.join("; ")}
        </Callout>
      )}

      {/* Forward / conductivities */}
      <Card className="section-card">
        <H4>Forward model</H4>
        <H5>Conductivities (S/m)</H5>
        {tissueNames.map((tissue) => (
          <div className="cond-row" key={tissue}>
            <FormGroup label={tissue} inline>
              <NumericInput
                value={num(conductivities[tissue])}
                minorStepSize={0.0001}
                stepSize={0.01}
                onValueChange={(v) =>
                  update(["forward", "conductivities_sm", tissue], v)
                }
                fill
              />
            </FormGroup>
          </div>
        ))}
        <Divider />
        <FormGroup label="Source tissue">
          <HTMLSelect
            value={str(forward.source_tissue)}
            options={tissueNames}
            onChange={(e) =>
              update(["forward", "source_tissue"], e.currentTarget.value)
            }
            fill
          />
        </FormGroup>
        <FormGroup label="Source spacing (mm)">
          <NumericInput
            value={num(forward.source_spacing_mm)}
            stepSize={0.5}
            onValueChange={(v) => update(["forward", "source_spacing_mm"], v)}
            fill
          />
        </FormGroup>
        <H5>Solver</H5>
        <FormGroup label="Type">
          <HTMLSelect
            value={str(solver.type, "cg")}
            options={["cg"]}
            onChange={(e) => update(["forward", "solver", "type"], e.currentTarget.value)}
            fill
          />
        </FormGroup>
        <FormGroup label="Scheme">
          <HTMLSelect
            value={str(solver.scheme, "sipg")}
            options={["sipg", "dg"]}
            onChange={(e) => update(["forward", "solver", "scheme"], e.currentTarget.value)}
            fill
          />
        </FormGroup>
        <FormGroup label="Penalty">
          <NumericInput
            value={num(solver.penalty)}
            onValueChange={(v) => update(["forward", "solver", "penalty"], v)}
            fill
          />
        </FormGroup>
        <FormGroup label="Reduction">
          <NumericInput
            value={num(solver.reduction)}
            minorStepSize={1e-12}
            stepSize={1e-10}
            onValueChange={(v) => update(["forward", "solver", "reduction"], v)}
            fill
          />
        </FormGroup>
        <FormGroup label="Int order add">
          <NumericInput
            value={num(solver.intorderadd)}
            onValueChange={(v) => update(["forward", "solver", "intorderadd"], v)}
            fill
          />
        </FormGroup>
        <Switch
          label="post_process_meg"
          checked={solver.post_process_meg === true}
          onChange={(e) =>
            update(["forward", "solver", "post_process_meg"], e.currentTarget.checked)
          }
        />
        <Switch
          label="subtract_mean"
          checked={solver.subtract_mean === true}
          onChange={(e) =>
            update(["forward", "solver", "subtract_mean"], e.currentTarget.checked)
          }
        />
      </Card>

      {/* Sensors */}
      <Card className="section-card">
        <H4>Sensors</H4>
        {(
          [
            ["resolution_mm", "Resolution (mm)"],
            ["depth_mm", "Depth (mm)"],
            ["angular_margin_deg", "Angular margin (deg)"],
            ["cylinder_radius_factor", "Cylinder radius factor"],
            ["z_crop_low_factor", "Z crop low factor"],
          ] as [string, string][]
        ).map(([key, label]) => (
          <FormGroup label={label} key={key}>
            <NumericInput
              value={num(sensors[key])}
              stepSize={0.05}
              onValueChange={(v) => update(["sensors", key], v)}
              fill
            />
          </FormGroup>
        ))}
      </Card>

      {/* Electrodes */}
      <Card className="section-card">
        <H4>Electrodes</H4>
        <FormGroup label="Shape">
          <HTMLSelect
            value={str(electrodes.shape, "paddle32")}
            options={["rectangular", "paddle32", "whole_body"]}
            onChange={(e) => update(["electrodes", "shape"], e.currentTarget.value)}
            fill
          />
        </FormGroup>
        <FormGroup label="Target tissue">
          <HTMLSelect
            value={str(electrodes.target_tissue, tissueNames[0] ?? "")}
            options={tissueNames}
            onChange={(e) => update(["electrodes", "target_tissue"], e.currentTarget.value)}
            fill
          />
        </FormGroup>
        {(
          [
            ["rows", "Rows"],
            ["cols", "Cols"],
            ["contact_pitch_mm", "Contact pitch (mm)"],
            ["head_offset_mm", "Head offset (mm)"],
            ["foot_offset_mm", "Foot offset (mm)"],
            ["target_z_low_factor", "Target z low factor"],
            ["target_z_high_factor", "Target z high factor"],
            ["n_contacts", "N contacts (whole_body)"],
            ["sample_seed", "Sample seed"],
          ] as [string, string][]
        ).map(([key, label]) => (
          <FormGroup label={label} key={key}>
            <NumericInput
              value={num(electrodes[key])}
              onValueChange={(v) => update(["electrodes", key], v)}
              fill
            />
          </FormGroup>
        ))}
        <FormGroup label="Label prefix">
          <InputGroup
            value={str(electrodes.label_prefix, "elec")}
            onChange={(e) => update(["electrodes", "label_prefix"], e.currentTarget.value)}
          />
        </FormGroup>
      </Card>

      {/* FEM */}
      <Card className="section-card">
        <H4>FEM</H4>
        {(
          [
            ["pitch_mm", "Pitch (mm)"],
            ["pad_mm", "Pad (mm)"],
            ["radbound", "Radbound"],
            ["maxvol", "Maxvol"],
            ["vagus_dilate_voxels", "Vagus dilate (voxels)"],
            ["bone_closing_mm", "Bone closing (mm)"],
          ] as [string, string][]
        ).map(([key, label]) => (
          <FormGroup label={label} key={key}>
            <NumericInput
              value={num(fem[key])}
              stepSize={0.5}
              onValueChange={(v) => update(["fem", key], v)}
              fill
            />
          </FormGroup>
        ))}
        <FormGroup label="Tissues (comma separated)">
          <InputGroup
            value={(Array.isArray(fem.tissues) ? (fem.tissues as string[]) : []).join(", ")}
            onChange={(e) =>
              update(
                ["fem", "tissues"],
                e.currentTarget.value
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean)
              )
            }
          />
        </FormGroup>
      </Card>

      {/* Raw editors */}
      <Card className="section-card">
        <H4>Advanced (raw JSON)</H4>
        {RAW_SECTIONS.map((section) => (
          <RawSection
            key={section}
            section={section}
            value={config[section]}
            open={openRaw[section] ?? false}
            onToggle={() =>
              setOpenRaw((p) => ({ ...p, [section]: !(p[section] ?? false) }))
            }
            onCommit={(parsed) => update([section], parsed)}
          />
        ))}
      </Card>
    </div>
  );
}

interface RawProps {
  section: string;
  value: unknown;
  open: boolean;
  onToggle: () => void;
  onCommit: (parsed: unknown) => void;
}

function RawSection({ section, value, open, onToggle, onCommit }: RawProps) {
  const [text, setText] = useState(() => JSON.stringify(value ?? {}, null, 2));
  const [error, setError] = useState<string | null>(null);

  // Keep local text in sync when the upstream value changes (e.g. reset).
  useEffect(() => {
    setText(JSON.stringify(value ?? {}, null, 2));
    setError(null);
  }, [value]);

  function handleBlur() {
    try {
      const parsed = JSON.parse(text) as unknown;
      setError(null);
      onCommit(parsed);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div style={{ marginBottom: 8 }}>
      <Button
        minimal
        fill
        alignText="left"
        icon={open ? "chevron-down" : "chevron-right"}
        onClick={onToggle}
      >
        {section}
      </Button>
      <Collapse isOpen={open}>
        <TextArea
          value={text}
          onChange={(e) => setText(e.currentTarget.value)}
          onBlur={handleBlur}
          fill
          style={{ fontFamily: "monospace", fontSize: 12, minHeight: 160 }}
        />
        {error && (
          <Callout intent={Intent.DANGER} style={{ marginTop: 4 }}>
            {error}
          </Callout>
        )}
      </Collapse>
    </div>
  );
}
