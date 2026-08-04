"use client";

import { useEffect, useMemo, useState } from "react";
import { Note, Select, Slider, Spinner, Tag } from "@/components/ui";
import { getSystem, type SystemInfo } from "@/lib/api";
import { getPath, setPath, type Cfg } from "@/lib/config";

const WORKERS_PATH = "forward.local_workers";

interface Props {
  config: Cfg;
  onChange: (c: Cfg) => void;
}

// Presets cover what people actually want; the slider underneath covers the
// rest. `0` is the config's own sentinel for "every core", so choosing it keeps
// working if the machine changes.
type PresetValue = number;

function presets(sys: SystemInfo): { value: PresetValue; label: string; note: string }[] {
  const out = [
    {
      value: 0,
      label: `All cores (${sys.all_cores})`,
      note: "Fastest. Uses every core, so the machine will be busy.",
    },
  ];
  if (sys.cpu_performance && sys.cpu_performance < sys.cpu_logical) {
    out.push({
      value: sys.cpu_performance,
      label: `Performance cores only (${sys.cpu_performance})`,
      note:
        "Skips the efficiency cores. Often nearly as fast as all cores on " +
        "Apple silicon, and leaves the machine usable.",
    });
  }
  const half = Math.max(1, Math.floor(sys.cpu_logical / 2));
  out.push({
    value: half,
    label: `Half (${half})`,
    note: "Leaves plenty of headroom for other work while it solves.",
  });
  out.push({
    value: 1,
    label: "Single core",
    note: "Slowest, but the most predictable — and easiest to debug.",
  });
  return out;
}

export default function ComputePanel({ config, onChange }: Props) {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSystem()
      .then(setSys)
      .finally(() => setLoading(false));
  }, []);

  const workers = getPath<number>(config, WORKERS_PATH, 0);
  const options = useMemo(() => (sys ? presets(sys) : []), [sys]);
  const matched = options.find((o) => o.value === workers);

  // What `0` actually resolves to, so "all cores" is never abstract.
  const effective = workers === 0 ? (sys?.all_cores ?? null) : workers;

  if (loading) {
    return (
      <div className="adv-field">
        <Spinner size={16} />
      </div>
    );
  }

  if (!sys) {
    return (
      <Note tone="warn">
        Could not read this machine&apos;s core count. The solve will still run
        using the value in the config ({String(workers)}).
      </Note>
    );
  }

  return (
    <>
      <div className="compute-machine">
        <Tag icon="machine">
          {sys.machine} · {sys.cpu_logical} cores
          {sys.cpu_performance && sys.cpu_performance < sys.cpu_logical
            ? ` (${sys.cpu_performance} performance)`
            : ""}
        </Tag>
        {effective !== null && (
          <Tag tone="signal" icon="solve">
            solving on {effective} core{effective === 1 ? "" : "s"}
          </Tag>
        )}
      </div>

      <div className="adv-field">
        <div className="adv-field-head">
          <label htmlFor="compute-preset">How much of this machine to use</label>
          <Select
            id="compute-preset"
            value={matched ? String(workers) : "custom"}
            onChange={(v) => {
              if (v === "custom") return;
              onChange(setPath(config, WORKERS_PATH, Number(v)));
            }}
            options={[
              ...options.map((o) => ({ value: String(o.value), label: o.label })),
              ...(matched ? [] : [{ value: "custom", label: `Custom (${workers})` }]),
            ]}
          />
        </div>
        <p className="adv-help">
          {matched?.note ??
            "A custom worker count set below. The forward solve splits the " +
              "sensor array into this many chunks and solves them in parallel."}
        </p>
      </div>

      <div className="adv-field">
        <div className="adv-field-head">
          <label id="compute-slider-label">Exact worker count</label>
          <span className="mono compute-count">
            {workers === 0 ? `auto (${sys.all_cores})` : workers}
          </span>
        </div>
        <Slider
          min={0}
          max={sys.cpu_logical}
          value={workers}
          ariaLabel="Worker count"
          onChange={(n) => onChange(setPath(config, WORKERS_PATH, n))}
        />
        <p className="adv-help">
          0 means &ldquo;every core&rdquo; and follows the machine. More workers
          means more memory in flight — each one loads the FEM mesh, so if a big
          model runs out of memory, lower this first.
        </p>
      </div>
    </>
  );
}
