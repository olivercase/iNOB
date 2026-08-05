"use client";

import { useEffect, useState } from "react";
import { Button, Note, Select, TextInput } from "@/components/ui";
import {
  getDuneuro,
  getSolver,
  rescanSolver,
  setDuneuroPath,
  type DuneuroStatus,
  type SolverInfo,
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
  const [solver, setSolver] = useState<SolverInfo | null>(null);
  const [scanning, setScanning] = useState(false);

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
    getSolver().then(setSolver);
  }, []);

  const rescan = async () => {
    setScanning(true);
    setSolver(await rescanSolver());
    setScanning(false);
  };

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
      {/* What the next run will actually use. The backend searches the machine
          and pairs a build with an interpreter that can load it, so this
          reports a decision rather than asking for one. */}
      <div className="solver">
        <div className="solver-head">
          <span className={`led ${solver?.found ? "led--on" : "led--off"}`} aria-hidden />
          <span className="solver-title">
            {solver === null
              ? "Looking for a DUNEuro build…"
              : solver.found
                ? "Found a working solver"
                : "No working DUNEuro found"}
          </span>
          <Button
            size="sm"
            variant="ghost"
            icon="reset"
            loading={scanning}
            onClick={rescan}
          >
            Search again
          </Button>
        </div>

        {solver && (
          <dl className="solver-facts mono">
            <div>
              <dt>runs on</dt>
              <dd>{solver.python_version}</dd>
            </div>
            <div>
              <dt>python</dt>
              <dd title={solver.python}>{solver.python}</dd>
            </div>
            <div>
              <dt>build</dt>
              <dd title={solver.duneuro_path ?? ""}>
                {solver.duneuro_path ?? "none needed"}
              </dd>
            </div>
          </dl>
        )}

        {solver && !solver.found && (
          <Note tone="warn">
            Nothing importable turned up under {solver.searched.length} searched
            location{solver.searched.length === 1 ? "" : "s"}. Build one with{" "}
            <code>cluster/build_duneuro.sh</code>, or point at it below.
          </Note>
        )}
      </div>

      <h3 className="jsub">This backend&apos;s own interpreter</h3>
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
        <Note tone={ready ? "ok" : "warn"}>{status.hint}</Note>
      )}

      <label className="dun-field">
        <span>DUNEuro build</span>
        <Select value={choice} disabled={busy} onChange={onSelect}>
          <option value={NONE}>Use the interpreter&apos;s own duneuropy</option>
          {status?.candidates.map((c) => (
            <option key={c.path} value={c.path}>
              {c.label}
              {c.importable ? "  ✓ loads here" : "  ✗ wrong Python"}
            </option>
          ))}
          <option value={CUSTOM}>Custom path…</option>
        </Select>
      </label>

      {choice === CUSTOM && (
        <div className="dun-custom">
          <TextInput
            placeholder="/path/to/dir/containing/duneuropy"
            value={custom}
            onChange={setCustom}
            disabled={busy}
          />
          <Button variant="primary" disabled={busy || !custom} onClick={() => apply(custom)}>
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
