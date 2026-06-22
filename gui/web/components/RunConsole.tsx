"use client";

import { useEffect, useRef, useState } from "react";
import { Button, Card, Switch, Tag } from "@blueprintjs/core";

const STAGES = ["geom", "fem", "sensors", "forward", "viz"];

interface Props {
  running: boolean;
  logs: string[];
  statuses: Record<string, string>;
  onSimulate: (stages: string[], force: boolean) => void;
}

const INTENT: Record<string, "success" | "none" | "danger"> = {
  ran: "success", skipped: "none", failed: "danger",
};

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
      <Card className="panel" compact>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <Button
            intent="primary" icon="play" loading={running}
            disabled={selected.length === 0}
            onClick={() => onSimulate(STAGES.filter((s) => selected.includes(s)), force)}
          >
            Simulate
          </Button>
          {STAGES.map((s) => (
            <Tag
              key={s} interactive minimal={!selected.includes(s)}
              intent={selected.includes(s) ? "primary" : "none"}
              onClick={() => toggle(s)}
            >
              {s}
            </Tag>
          ))}
          <Switch checked={force} label="force" onChange={() => setForce((f) => !f)}
                  style={{ margin: 0 }} />
        </div>
        {Object.keys(statuses).length > 0 && (
          <div className="chips" style={{ marginTop: 8 }}>
            {Object.entries(statuses).map(([k, v]) => (
              <Tag key={k} intent={INTENT[v] ?? "none"} minimal>{k}: {v}</Tag>
            ))}
          </div>
        )}
      </Card>
      <div className="console" ref={consoleRef}>
        {logs.length === 0 ? "— no output yet —" : logs.join("\n")}
      </div>
    </div>
  );
}
