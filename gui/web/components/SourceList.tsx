"use client";

import type { PointSource } from "@/lib/api";

interface Props {
  sources: PointSource[];
  onChange: (s: PointSource[]) => void;
}

export default function SourceList({ sources, onChange }: Props) {
  const update = (i: number, key: keyof PointSource, v: number) => {
    const next = sources.slice();
    next[i] = { ...next[i], [key]: v };
    onChange(next);
  };
  const remove = (i: number) => onChange(sources.filter((_, j) => j !== i));
  const add = () =>
    onChange([...sources, { x: 0, y: 0, z: 0, strength_nAm: 70 }]);

  return (
    <div className="panel">
      <h3>Sources ({sources.length})</h3>
      {sources.length === 0 && (
        <p className="hint">
          Click <b>Place source</b> in the viewer to drop dipoles, or add one manually.
        </p>
      )}
      {sources.map((s, i) => (
        <div className="srcrow" key={i}>
          <span style={{ color: "var(--muted)", width: 16 }}>{i + 1}</span>
          {(["x", "y", "z"] as const).map((k) => (
            <input
              key={k}
              type="number"
              value={s[k]}
              title={`${k} (mm)`}
              onChange={(e) => update(i, k, parseFloat(e.target.value) || 0)}
            />
          ))}
          <input
            type="number"
            value={s.strength_nAm}
            title="strength (nA·m)"
            style={{ width: 78 }}
            onChange={(e) => update(i, "strength_nAm", parseFloat(e.target.value) || 0)}
          />
          <button className="ghost" onClick={() => remove(i)} title="remove">
            ✕
          </button>
        </div>
      ))}
      <div className="row" style={{ marginTop: 8 }}>
        <button className="ghost" onClick={add}>+ Add source</button>
        <span className="hint">x, y, z in mm · strength in nA·m</span>
      </div>
    </div>
  );
}
