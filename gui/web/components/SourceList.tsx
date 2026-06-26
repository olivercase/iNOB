"use client";

import { Button, Card, InputGroup } from "@blueprintjs/core";
import type { PointSource } from "@/lib/api";

interface Props {
  sources: PointSource[];
  selected: number | null;
  onSelect: (i: number | null) => void;
  onChange: (s: PointSource[]) => void;
}

export default function SourceList({ sources, selected, onSelect, onChange }: Props) {
  const update = (i: number, key: keyof PointSource, v: number) => {
    const next = sources.slice();
    next[i] = { ...next[i], [key]: v };
    onChange(next);
  };
  const remove = (i: number) => {
    onChange(sources.filter((_, j) => j !== i));
    onSelect(null);
  };
  const add = () => {
    onChange([...sources, { x: 0, y: 0, z: 0, strength_nAm: 70 }]);
    onSelect(sources.length);
  };

  return (
    <Card className="panel" compact>
      <h3 className="section-title">Sources ({sources.length})</h3>
      {sources.length === 0 && (
        <p className="bp6-text-muted">
          Click <b>Place source</b> in the viewer to drop dipoles, or add one manually.
        </p>
      )}
      {sources.map((s, i) => (
        <div
          className={`srcrow${i === selected ? " srcrow--sel" : ""}`}
          key={i}
          onClick={() => onSelect(i)}
        >
          <span className="srcnum" style={{ width: 16 }}>{i + 1}</span>
          {(["x", "y", "z"] as const).map((k) => (
            <InputGroup
              key={k}
              type="number"
              value={String(s[k])}
              title={`${k} (mm)`}
              small
              onValueChange={(v) => update(i, k, parseFloat(v) || 0)}
            />
          ))}
          <InputGroup
            type="number"
            value={String(s.strength_nAm)}
            title="strength (nA·m)"
            small
            style={{ width: 78 }}
            onValueChange={(v) => update(i, "strength_nAm", parseFloat(v) || 0)}
          />
          <Button
            icon="cross"
            minimal
            small
            title="remove"
            onClick={(e) => {
              e.stopPropagation();
              remove(i);
            }}
          />
        </div>
      ))}
      <Button icon="add" minimal small onClick={add} style={{ marginTop: 6 }}>
        Add source
      </Button>
      <span className="bp6-text-muted" style={{ marginLeft: 8 }}>x, y, z mm · strength nA·m</span>
    </Card>
  );
}
