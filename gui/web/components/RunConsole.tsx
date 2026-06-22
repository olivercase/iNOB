"use client";

import { useEffect, useRef, useState } from "react";

const STAGES = ["geom", "fem", "sensors", "forward", "viz"];

interface Props {
  running: boolean;
  logs: string[];
  statuses: Record<string, string>;
  onSimulate: (stages: string[], force: boolean) => void;
}

export default function RunConsole({ running, logs, statuses, onSimulate }: Props) {
  const [selected, setSelected] = useState<string[]>(STAGES);
  const [force, setForce] = useState(false);
  const consoleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [logs]);

  const toggle = (s: string) =>
    setSelected((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));

  return (
    <div>
      <div className="panel" style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <button
          className="btn"
          disabled={running || selected.length === 0}
          onClick={() => onSimulate(STAGES.filter((s) => selected.includes(s)), force)}
        >
          {running ? "Simulating…" : "▶ Simulate"}
        </button>
        {STAGES.map((s) => (
          <span key={s} className={`chip ${selected.includes(s) ? "on" : ""}`} onClick={() => toggle(s)}>
            {s}
          </span>
        ))}
        <label className="chip" onClick={() => setForce((f) => !f)}>
          {force ? "✓" : "○"} force
        </label>
      </div>
      {Object.keys(statuses).length > 0 && (
        <div className="statuses">
          {Object.entries(statuses).map(([k, v]) => (
            <span key={k} className={`tag ${v}`}>{k}: {v}</span>
          ))}
        </div>
      )}
      <div className="console" ref={consoleRef}>
        {logs.length === 0 ? "— no output yet —" : logs.join("\n")}
      </div>
    </div>
  );
}
