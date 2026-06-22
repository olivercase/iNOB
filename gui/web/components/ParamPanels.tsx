"use client";

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
  "vagus_left",
  "vagus_right",
  "blood_vessel",
  "muscle",
  "bone",
  "skin",
];

function NumberField({
  config,
  onChange,
  path,
  label,
  step = "any",
}: Props & { path: string; label: string; step?: string }) {
  const v = getPath<number>(config, path, 0);
  return (
    <div className="row">
      <label title={path}>{label}</label>
      <input
        type="number"
        step={step}
        value={Number.isFinite(v) ? v : 0}
        onChange={(e) => onChange(setPath(config, path, parseFloat(e.target.value)))}
      />
    </div>
  );
}

export default function ParamPanels({ config, onChange, threshold, onThreshold }: PanelProps) {
  const cond = getPath<Record<string, number>>(config, "forward.conductivities_sm", {});
  const tissues = getPath<string[]>(config, "fem.tissues", []);

  const toggleTissue = (t: string) => {
    const next = tissues.includes(t)
      ? tissues.filter((x) => x !== t)
      : [...tissues, t];
    onChange(setPath(config, "fem.tissues", next));
  };

  return (
    <>
      <div className="panel">
        <h3>Conductivities (S/m)</h3>
        <div className="grid2">
          {Object.keys(cond).length === 0 && <p className="hint">no config loaded</p>}
          {Object.entries(cond).map(([k, v]) => (
            <div className="row" key={k}>
              <label>{k}</label>
              <input
                type="number"
                step="any"
                value={v}
                onChange={(e) =>
                  onChange(
                    setPath(config, `forward.conductivities_sm.${k}`, parseFloat(e.target.value)),
                  )
                }
              />
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>Meshes / tissues included</h3>
        <div>
          {ALL_TISSUES.map((t) => (
            <span
              key={t}
              className={`chip ${tissues.includes(t) ? "on" : ""}`}
              onClick={() => toggleTissue(t)}
            >
              {tissues.includes(t) ? "✓" : "○"} {t}
            </span>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>OPM sensor array</h3>
        <NumberField config={config} onChange={onChange} path="sensors.resolution_mm" label="resolution (mm)" />
        <NumberField config={config} onChange={onChange} path="sensors.depth_mm" label="stand-off depth (mm)" />
        <NumberField config={config} onChange={onChange} path="sensors.angular_margin_deg" label="angular margin (°)" />
      </div>

      <div className="panel">
        <h3>HD electrode array</h3>
        <NumberField config={config} onChange={onChange} path="electrodes.rows" label="rows" step="1" />
        <NumberField config={config} onChange={onChange} path="electrodes.cols" label="cols" step="1" />
        <NumberField config={config} onChange={onChange} path="electrodes.contact_pitch_mm" label="contact pitch (mm)" />
        <div className="row">
          <label>shape</label>
          <select
            value={getPath<string>(config, "electrodes.shape", "paddle32")}
            onChange={(e) => onChange(setPath(config, "electrodes.shape", e.target.value))}
          >
            <option value="rectangular">rectangular</option>
            <option value="paddle32">paddle32</option>
            <option value="whole_body">whole_body</option>
          </select>
        </div>
      </div>

      <div className="panel">
        <h3>Noise floor</h3>
        <NumberField config={config} onChange={onChange} path="noise.opm_intrinsic_fT_sqrtHz" label="OPM intrinsic (fT/√Hz)" />
        <NumberField config={config} onChange={onChange} path="noise.eeg_amplifier_uV_sqrtHz" label="EEG amplifier (µV/√Hz)" />
        <NumberField config={config} onChange={onChange} path="noise.eeg_electrode_skin_kohm" label="electrode (kΩ)" />
        <NumberField config={config} onChange={onChange} path="noise.bandwidth_hz" label="bandwidth (Hz)" />
      </div>

      <div className="panel">
        <h3>Detection criterion</h3>
        <div className="row">
          <label title="SNR a source must reach to count as detected">SNR threshold</label>
          <input
            type="number"
            step="any"
            value={threshold}
            onChange={(e) => onThreshold(parseFloat(e.target.value) || 0)}
          />
        </div>
        <p className="hint">trials-needed = (threshold / single-trial SNR)²</p>
      </div>
    </>
  );
}
