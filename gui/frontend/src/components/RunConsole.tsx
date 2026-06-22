import { useEffect, useRef, useState } from "react";
import {
  Button,
  Callout,
  Checkbox,
  Collapse,
  Divider,
  H5,
  Intent,
  Switch,
  Tag,
} from "@blueprintjs/core";
import { runPipeline, Stage } from "../api";

const ALL_STAGES: Stage[] = ["geom", "fem", "sensors", "forward", "viz"];

function statusIntent(status: string): Intent {
  if (status === "ran" || status === "ok") return Intent.SUCCESS;
  if (status === "failed" || status === "error") return Intent.DANGER;
  return Intent.NONE;
}

interface Props {
  open: boolean;
  onToggle: () => void;
}

export default function RunConsole({ open, onToggle }: Props) {
  const [selected, setSelected] = useState<Set<Stage>>(new Set(ALL_STAGES));
  const [force, setForce] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [statuses, setStatuses] = useState<Record<string, string> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const preRef = useRef<HTMLPreElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [lines]);

  useEffect(() => () => wsRef.current?.close(), []);

  function toggleStage(s: Stage) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  function start() {
    setLines([]);
    setStatuses(null);
    setError(null);
    setRunning(true);
    const stages: "all" | Stage[] =
      selected.size === ALL_STAGES.length
        ? "all"
        : ALL_STAGES.filter((s) => selected.has(s));
    wsRef.current = runPipeline(stages, force, {
      onLog: (line) => setLines((prev) => [...prev, line]),
      onDone: (s) => {
        setStatuses(s);
        setRunning(false);
      },
      onError: (msg) => {
        setError(msg);
        setRunning(false);
      },
    });
  }

  return (
    <div className="run-console" style={{ borderTop: "1px solid rgba(255,255,255,0.15)" }}>
      <Button
        minimal
        fill
        alignText="left"
        icon={open ? "chevron-down" : "chevron-up"}
        onClick={onToggle}
      >
        Run Console {running ? "(running...)" : ""}
      </Button>
      <Collapse isOpen={open}>
        <div style={{ padding: "8px 12px" }}>
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            {ALL_STAGES.map((s) => (
              <Checkbox
                key={s}
                label={s}
                checked={selected.has(s)}
                disabled={running}
                onChange={() => toggleStage(s)}
                inline
              />
            ))}
            <Switch
              label="force"
              checked={force}
              disabled={running}
              onChange={(e) => setForce(e.currentTarget.checked)}
              inline
            />
            <Button
              intent={Intent.SUCCESS}
              icon="play"
              loading={running}
              disabled={running || selected.size === 0}
              onClick={start}
            >
              Run
            </Button>
          </div>

          {error && (
            <Callout intent={Intent.DANGER} style={{ margin: "8px 0" }}>
              {error}
            </Callout>
          )}

          {statuses && (
            <div style={{ margin: "8px 0", display: "flex", gap: 6, flexWrap: "wrap" }}>
              {Object.entries(statuses).map(([stage, status]) => (
                <Tag key={stage} intent={statusIntent(status)} minimal>
                  {stage}: {status}
                </Tag>
              ))}
            </div>
          )}

          <Divider />
          <H5 style={{ marginTop: 8 }}>Log</H5>
          <pre ref={preRef}>{lines.join("\n") || "(no output yet)"}</pre>
        </div>
      </Collapse>
    </div>
  );
}
