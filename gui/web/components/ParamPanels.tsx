"use client";

import { Card, HTMLSelect, NumericInput, Tag } from "@blueprintjs/core";
import { Cfg, getPath, setPath } from "@/lib/config";

interface Props {
  config: Cfg;
  onChange: (c: Cfg) => void;
}

interface PanelProps extends Props {
  threshold: number;
  onThreshold: (n: number) => void;
}

const ALL_TISSUES = [
  "vagus_left", "vagus_right", "blood_vessel", "spinal_cord", "muscle", "bone", "skin",
];

function NumRow({
  config, onChange, path, label, stepSize,
}: Props & { path: string; label: string; stepSize?: number }) {
  const v = getPath<number>(config, path, 0);
  return (
    <div className="field-row">
      <label title={path}>{label}</label>
      <NumericInput
        value={Number.isFinite(v) ? v : 0}
        onValueChange={(n) => onChange(setPath(config, path, n))}
        buttonPosition="none"
        fill={false}
        style={{ width: 96 }}
        minorStepSize={null}
        stepSize={stepSize ?? 1}
      />
    </div>
  );
}

export default function ParamPanels({ config, onChange, threshold, onThreshold }: PanelProps) {
  const cond = getPath<Record<string, number>>(config, "forward.conductivities_sm", {});
  const tissues = getPath<string[]>(config, "fem.tissues", []);

  const toggleTissue = (t: string) => {
    const next = tissues.includes(t) ? tissues.filter((x) => x !== t) : [...tissues, t];
    onChange(setPath(config, "fem.tissues", next));
  };

  return (
    <>
      <Card className="panel" compact>
        <h3 className="section-title">Conductivities (S/m)</h3>
        <div className="grid2">
          {Object.entries(cond).map(([k, v]) => (
            <div className="field-row" key={k}>
              <label>{k}</label>
              <NumericInput
                value={v}
                onValueChange={(n) => onChange(setPath(config, `forward.conductivities_sm.${k}`, n))}
                buttonPosition="none" style={{ width: 96 }} minorStepSize={null} stepSize={0.01}
              />
            </div>
          ))}
          {Object.keys(cond).length === 0 && <span className="bp6-text-muted">no config loaded</span>}
        </div>
      </Card>

      <Card className="panel" compact>
        <h3 className="section-title">Meshes / tissues included</h3>
        <div className="chips">
          {ALL_TISSUES.map((t) => {
            const on = tissues.includes(t);
            return (
              <Tag
                key={t}
                interactive
                minimal={!on}
                intent={on ? "primary" : "none"}
                icon={on ? "tick" : "disable"}
                onClick={() => toggleTissue(t)}
              >
                {t}
              </Tag>
            );
          })}
        </div>
      </Card>

      <Card className="panel" compact>
        <h3 className="section-title">OPM sensor array</h3>
        <NumRow config={config} onChange={onChange} path="sensors.resolution_mm" label="resolution (mm)" />
        <NumRow config={config} onChange={onChange} path="sensors.depth_mm" label="stand-off depth (mm)" />
        <NumRow config={config} onChange={onChange} path="sensors.angular_margin_deg" label="angular margin (°)" />
      </Card>

      <Card className="panel" compact>
        <h3 className="section-title">HD electrode array</h3>
        <NumRow config={config} onChange={onChange} path="electrodes.rows" label="rows" />
        <NumRow config={config} onChange={onChange} path="electrodes.cols" label="cols" />
        <NumRow config={config} onChange={onChange} path="electrodes.contact_pitch_mm" label="contact pitch (mm)" />
        <div className="field-row">
          <label>shape</label>
          <HTMLSelect
            value={getPath<string>(config, "electrodes.shape", "paddle32")}
            onChange={(e) => onChange(setPath(config, "electrodes.shape", e.currentTarget.value))}
            options={["rectangular", "paddle32", "whole_body"]}
          />
        </div>
      </Card>

      <Card className="panel" compact>
        <h3 className="section-title">Noise floor</h3>
        <NumRow config={config} onChange={onChange} path="noise.opm_intrinsic_fT_sqrtHz" label="OPM intrinsic (fT/√Hz)" />
        <NumRow config={config} onChange={onChange} path="noise.eeg_amplifier_uV_sqrtHz" label="EEG amplifier (µV/√Hz)" />
        <NumRow config={config} onChange={onChange} path="noise.eeg_electrode_skin_kohm" label="electrode (kΩ)" />
        <NumRow config={config} onChange={onChange} path="noise.bandwidth_hz" label="bandwidth (Hz)" />
      </Card>

      <Card className="panel" compact>
        <h3 className="section-title">Detection criterion</h3>
        <div className="field-row">
          <label title="SNR a source must reach to count as detected">SNR threshold</label>
          <NumericInput
            value={threshold}
            onValueChange={(n) => onThreshold(n)}
            buttonPosition="none" style={{ width: 96 }} minorStepSize={null} stepSize={0.5}
          />
        </div>
        <span className="bp6-text-muted">trials-needed = (threshold / single-trial SNR)²</span>
      </Card>
    </>
  );
}
