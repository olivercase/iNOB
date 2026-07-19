"use client";

import { NumericInput, Callout } from "@blueprintjs/core";
import { Cfg, getPath, setPath } from "@/lib/config";
import {
  PARAM_GROUPS,
  CONDUCTIVITY_ROOT,
  CONDUCTIVITY_BLURB,
  conductivityHelp,
  type Param,
} from "@/lib/params";

interface Props {
  config: Cfg;
  onChange: (c: Cfg) => void;
}

// One labelled numeric field with its explanation underneath — the help text
// is the whole reason this panel exists, so it is always visible, not a tooltip.
function Field({
  config,
  onChange,
  param,
}: Props & { param: Param }) {
  const value = getPath<number>(config, param.path, 0);
  return (
    <div className="adv-field">
      <div className="adv-field-head">
        <label>
          {param.label}
          {param.unit && <span className="adv-unit"> ({param.unit})</span>}
        </label>
        <NumericInput
          value={Number.isFinite(value) ? value : 0}
          onValueChange={(n) => onChange(setPath(config, param.path, n))}
          buttonPosition="none"
          style={{ width: 96 }}
          minorStepSize={null}
          stepSize={param.step ?? 1}
          min={param.min}
          max={param.max}
        />
      </div>
      <p className="adv-help">{param.help}</p>
    </div>
  );
}

export default function ParamPanels({ config, onChange }: Props) {
  const cond = getPath<Record<string, number>>(config, CONDUCTIVITY_ROOT, {});

  return (
    <div className="adv">
      <Callout icon="info-sign" className="bp6-text-muted" style={{ marginBottom: 16 }}>
        Every setting here already has a sensible default. You only need to touch
        these to explore a different sensor, a different noise floor, or to check
        that your result is mesh-converged.
      </Callout>

      {PARAM_GROUPS.map((group) => (
        <section className="adv-group" key={group.title}>
          <h3 className="adv-title">{group.title}</h3>
          <p className="adv-blurb">{group.blurb}</p>
          {group.params.map((param) => (
            <Field key={param.path} config={config} onChange={onChange} param={param} />
          ))}
        </section>
      ))}

      <section className="adv-group">
        <h3 className="adv-title">Tissue conductivities</h3>
        <p className="adv-blurb">{CONDUCTIVITY_BLURB}</p>
        {Object.entries(cond).map(([tissue, v]) => (
          <Field
            key={tissue}
            config={config}
            onChange={onChange}
            param={{
              path: `${CONDUCTIVITY_ROOT}.${tissue}`,
              label: tissue.replace(/_/g, " "),
              unit: "S/m",
              help: conductivityHelp(tissue),
              step: 0.01,
              min: 0,
            }}
          />
        ))}
        {Object.keys(cond).length === 0 && (
          <span className="bp6-text-muted">no config loaded</span>
        )}
      </section>
    </div>
  );
}
