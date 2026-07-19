"use client";

import { useState } from "react";
import { Button, Callout, Checkbox, HTMLTable } from "@blueprintjs/core";
import { runLadder, type LadderRung, type LadderResult } from "@/lib/api";

// The three rungs, in ladder order. Analytic rungs are instant and need no
// DUNEuro; the fem rung needs the leadfield, so it is off by default.
const RUNGS: { key: LadderRung; name: string; blurb: string; needsSolve: boolean }[] =
  [
    { key: "biot", name: "Biot–Savart", blurb: "free space, no boundary", needsSolve: false },
    { key: "sarvas", name: "Sarvas", blurb: "single homogeneous sphere", needsSolve: false },
    { key: "fem", name: "FEM", blurb: "full multi-tissue (DUNEuro)", needsSolve: true },
  ];

const RATIO_LABELS: Record<string, string> = {
  sarvas_to_biot: "Sarvas / Biot–Savart",
  fem_to_biot: "FEM / Biot–Savart",
  fem_to_sarvas: "FEM / Sarvas",
};

export default function LadderPanel() {
  const [picked, setPicked] = useState<Record<LadderRung, boolean>>({
    biot: true,
    sarvas: true,
    fem: false,
  });
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<LadderResult | null>(null);
  const [error, setError] = useState<{ msg: string; hint?: string } | null>(null);

  const chosen = RUNGS.filter((r) => picked[r.key]).map((r) => r.key);

  const toggle = (k: LadderRung) =>
    setPicked((p) => ({ ...p, [k]: !p[k] }));

  const run = async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    const { result: res, error: err, hint } = await runLadder(chosen);
    if (res) setResult(res);
    else if (err) setError({ msg: err, hint });
    setRunning(false);
  };

  return (
    <div className="ladder">
      <p className="adv-blurb">
        Compare the same source through three forward models of rising fidelity.
        Biot–Savart and Sarvas are analytic and run in seconds; FEM needs the
        DUNEuro solve. Pick any combination.
      </p>

      <div className="ladder-rungs">
        {RUNGS.map((r) => (
          <label
            key={r.key}
            className={`ladder-rung${picked[r.key] ? " ladder-rung--on" : ""}`}
          >
            <Checkbox
              checked={picked[r.key]}
              onChange={() => toggle(r.key)}
              style={{ margin: 0 }}
            />
            <span className="ladder-rung-name">{r.name}</span>
            <span className="ladder-rung-blurb">{r.blurb}</span>
            {r.needsSolve && <span className="ladder-tag">needs solve</span>}
          </label>
        ))}
      </div>

      <Button
        intent="primary"
        icon="chart"
        loading={running}
        disabled={chosen.length === 0}
        onClick={run}
        style={{ marginTop: 10 }}
      >
        Run {chosen.length} rung{chosen.length === 1 ? "" : "s"}
      </Button>

      {error && (
        <Callout intent="warning" style={{ marginTop: 12 }} title="Could not run">
          <p>{error.msg}</p>
          {error.hint && <p className="bp6-text-muted">{error.hint}</p>}
        </Callout>
      )}

      {result && (
        <div className="ladder-result">
          <div className="bp6-text-muted" style={{ margin: "10px 0 6px" }}>
            source #{result.source_index + 1} of {result.n_sources} ·{" "}
            {result.n_radial_coils} radial coils · peak field in fT per nA·m
          </div>
          <HTMLTable compact striped className="mono" style={{ width: "100%" }}>
            <thead>
              <tr>
                <th>rung</th>
                <th>peak</th>
                <th>RMS</th>
              </tr>
            </thead>
            <tbody>
              {RUNGS.filter((r) => result.rungs[r.key]).map((r) => (
                <tr key={r.key}>
                  <td>{r.name}</td>
                  <td>{result.rungs[r.key].peak_fT_per_nAm.toFixed(2)}</td>
                  <td>{result.rungs[r.key].rms_fT_per_nAm.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </HTMLTable>

          {Object.keys(result.ratios).length > 0 && (
            <div className="ladder-ratios">
              {Object.entries(result.ratios).map(([k, v]) => (
                <span key={k} className="ladder-ratio">
                  {RATIO_LABELS[k] ?? k}
                  <b>{v.toFixed(2)}×</b>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
