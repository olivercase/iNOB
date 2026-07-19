"use client";

import { useEffect, useState } from "react";
import { Button, Callout, HTMLSelect, InputGroup } from "@blueprintjs/core";
import {
  getDuneuro,
  setDuneuroPath,
  type DuneuroStatus,
} from "@/lib/api";

const CUSTOM = "__custom__";
const NONE = "__none__";

// Lets the user point the forward solver at a local DUNEuro build. The known
// build shows up pre-selected in the dropdown; the status line says plainly
// whether this backend can actually run the solve.
export default function DuneuroSetup() {
  const [status, setStatus] = useState<DuneuroStatus | null>(null);
  const [choice, setChoice] = useState<string>(NONE);
  const [custom, setCustom] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getDuneuro().then((s) => {
      setStatus(s);
      if (s?.active_path) {
        const known = s.candidates.some((c) => c.path === s.active_path);
        setChoice(known ? s.active_path : CUSTOM);
        if (!known) setCustom(s.active_path);
      }
      setLoaded(true);
    });
  }, []);

  const apply = async (path: string | null) => {
    setBusy(true);
    const s = await setDuneuroPath(path);
    if (s) setStatus(s);
    setBusy(false);
  };

  const onSelect = (value: string) => {
    setChoice(value);
    if (value === CUSTOM) return; // wait for the Apply button
    apply(value === NONE ? null : value);
  };

  const ready = status?.running.duneuropy_importable;

  return (
    <div className="dun">
      <div className="dun-status">
        <span className={`led ${ready ? "led--on" : "led--off"}`} aria-hidden />
        <div>
          <div className="dun-line">
            {loaded
              ? ready
                ? "DUNEuro is ready"
                : "DUNEuro not loaded"
              : "Checking…"}
          </div>
          {status && (
            <div className="dun-sub">
              backend interpreter: Python {status.running.python}
            </div>
          )}
        </div>
      </div>

      {status && (
        <Callout
          compact
          intent={ready ? "success" : "warning"}
          style={{ margin: "10px 0" }}
        >
          {status.hint}
        </Callout>
      )}

      <label className="dun-field">
        <span>DUNEuro build</span>
        <HTMLSelect
          value={choice}
          disabled={busy}
          onChange={(e) => onSelect(e.currentTarget.value)}
        >
          <option value={NONE}>Use the interpreter&apos;s own duneuropy</option>
          {status?.candidates.map((c) => (
            <option key={c.path} value={c.path}>
              {c.label}
              {c.importable ? "  ✓ loads here" : "  ✗ wrong Python"}
            </option>
          ))}
          <option value={CUSTOM}>Custom path…</option>
        </HTMLSelect>
      </label>

      {choice === CUSTOM && (
        <div className="dun-custom">
          <InputGroup
            placeholder="/path/to/dir/containing/duneuropy"
            value={custom}
            onValueChange={setCustom}
            disabled={busy}
          />
          <Button intent="primary" disabled={busy || !custom} onClick={() => apply(custom)}>
            Apply
          </Button>
        </div>
      )}

      {choice !== CUSTOM && status?.active_path && (
        <div className="dun-sub" style={{ marginTop: 6 }}>
          using: {status.active_path}
        </div>
      )}

      <p className="adv-help" style={{ marginTop: 10 }}>
        A build is compiled for one Python version. If the one you pick says
        “wrong Python”, restart the backend with that interpreter — for the
        local build that is{" "}
        <code className="mono">/Volumes/UCL/duneuro_build/venv/bin/python</code>.
      </p>
    </div>
  );
}
