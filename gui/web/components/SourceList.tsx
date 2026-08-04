"use client";

import { useEffect, useRef, useState } from "react";
import { Button, IconButton, TextInput } from "@/components/ui";
import type { PointSource } from "@/lib/api";

interface Props {
  sources: PointSource[];
  selected: number | null;
  onSelect: (i: number | null) => void;
  onChange: (s: PointSource[]) => void;
}

// Anatomical coordinates are signed about the origin, so a field has to accept
// intermediate text that isn't yet a number: "-", "-1.", "". Parsing on every
// keystroke and writing the result straight back turned a typed "-" into "0"
// instantly and made negatives unenterable. So: hold the raw text while
// focused, commit the parsed number when it parses.
function NumField({
  value,
  onCommit,
  title,
  width,
}: {
  value: number;
  onCommit: (n: number) => void;
  title: string;
  width?: number;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? String(value);

  return (
    <TextInput
      mono
      width={width}
      value={shown}
      title={title}
      ariaLabel={title}
      onChange={(v) => {
        setDraft(v);
        const n = Number(v);
        // Commit only genuinely numeric text; "" and "-" are valid things to
        // be *typing*, so keep them on screen without touching the model.
        if (v.trim() !== "" && Number.isFinite(n)) onCommit(n);
      }}
      // Snap back to the model on blur so a half-typed "-" doesn't linger.
      onBlur={() => setDraft(null)}
    />
  );
}

export default function SourceList({ sources, selected, onSelect, onChange }: Props) {
  // Stable identity per row so React reconciles correctly across removals —
  // with index keys, deleting row 1 left row 2's input values behind in it.
  const idsRef = useRef<number[]>([]);
  const nextIdRef = useRef(0);
  if (idsRef.current.length < sources.length) {
    while (idsRef.current.length < sources.length) {
      idsRef.current.push(nextIdRef.current++);
    }
  }
  useEffect(() => {
    if (idsRef.current.length > sources.length) {
      idsRef.current.length = sources.length;
    }
  }, [sources.length]);

  const update = (i: number, key: keyof PointSource, v: number) => {
    const next = sources.slice();
    next[i] = { ...next[i], [key]: v };
    onChange(next);
  };
  const remove = (i: number) => {
    idsRef.current.splice(i, 1);
    onChange(sources.filter((_, j) => j !== i));
    onSelect(null);
  };
  const add = () => {
    onChange([...sources, { x: 0, y: 0, z: 0, strength_nAm: 70 }]);
    onSelect(sources.length);
  };

  return (
    <div className="srcpanel">
      {sources.length > 0 && (
        <div className="srchdr">
          <span>{sources.length} source{sources.length > 1 ? "s" : ""}</span>
          <span className="srchdr-cols mono">x y z mm · nA·m</span>
        </div>
      )}
      {sources.length === 0 && (
        <p className="srcempty">
          No sources yet. Click the target in the view, or add one by hand.
        </p>
      )}
      {sources.map((s, i) => (
        <div
          className={`srcrow${i === selected ? " srcrow--sel" : ""}`}
          key={idsRef.current[i] ?? i}
          onClick={() => onSelect(i)}
        >
          <span className="srcnum mono">{i + 1}</span>
          {(["x", "y", "z"] as const).map((k) => (
            <NumField
              key={k}
              value={s[k]}
              title={`source ${i + 1} ${k} (mm)`}
              onCommit={(n) => update(i, k, n)}
            />
          ))}
          <NumField
            value={s.strength_nAm}
            title={`source ${i + 1} strength (nA·m)`}
            width={64}
            onCommit={(n) => update(i, "strength_nAm", n)}
          />
          <IconButton
            name="trash"
            size={13}
            label={`Remove source ${i + 1}`}
            onClick={() => remove(i)}
          />
        </div>
      ))}
      <Button icon="plus" variant="ghost" size="sm" onClick={add} className="srcadd">
        Add source by hand
      </Button>
    </div>
  );
}
