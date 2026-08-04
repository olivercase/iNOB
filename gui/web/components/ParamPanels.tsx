"use client";

import { Disclosure, TextInput } from "@/components/ui";
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
        <TextInput
          mono
          width={92}
          value={String(Number.isFinite(value) ? value : 0)}
          ariaLabel={param.label}
          onChange={(v) => {
            const n = Number(v);
            if (v.trim() !== "" && Number.isFinite(n)) {
              onChange(setPath(config, param.path, n));
            }
          }}
        />
      </div>
      <p className="adv-help">{param.help}</p>
    </div>
  );
}

export default function ParamPanels({ config, onChange }: Props) {
  const cond = getPath<Record<string, number>>(config, CONDUCTIVITY_ROOT, {});
  const nTissues = Object.keys(cond).length;

  return (
    <div className="adv">
      {PARAM_GROUPS.map((group) => (
        <Disclosure
          key={group.title}
          title={group.title}
          blurb={group.blurb}
          icon={group.icon}
          summary={group.summary?.(config)}
        >
          {group.params.map((param) => (
            <Field key={param.path} config={config} onChange={onChange} param={param} />
          ))}
        </Disclosure>
      ))}

      <Disclosure
        title="Tissue conductivities"
        blurb={CONDUCTIVITY_BLURB}
        icon="mesh"
        summary={nTissues ? `${nTissues} tissues` : undefined}
      >
        {Object.entries(cond).map(([tissue]) => (
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
        {nTissues === 0 && (
          <span className="ui-dim">no config loaded</span>
        )}
      </Disclosure>
    </div>
  );
}
