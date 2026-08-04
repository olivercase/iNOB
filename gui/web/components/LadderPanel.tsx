"use client";

import { useState } from "react";
import { Button, Checkbox, Note } from "@/components/ui";
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
      <p className="ui-lead">
        Compare the same source through three forward models of rising fidelity.
        Biot–Savart and Sarvas are analytic and run in seconds; FEM needs the
        DUNEuro solve. Pick any combination.
      </p>

      <div className="ladder-rungs">
        {RUNGS.map((r) => (
          <div
            key={r.key}
            className={`ladder-rung${picked[r.key] ? " ladder-rung--on" : ""}`}
          >
            <Checkbox
              checked={picked[r.key]}
              onChange={() => toggle(r.key)}
              label={<span className="ladder-rung-name">{r.name}</span>}
            />
            <span className="ladder-rung-blurb">{r.blurb}</span>
            {r.needsSolve && <span className="ladder-tag">needs solve</span>}
          </div>
        ))}
      </div>

      <Button
        variant="primary"
        icon="compare"
        loading={running}
        disabled={chosen.length === 0}
        onClick={run}
        className="ladder-run"
      >
        Run {chosen.length} rung{chosen.length === 1 ? "" : "s"}
      </Button>

      {error && (
        <Note tone="warn" title="Could not run">
          <p>{error.msg}</p>
          {error.hint && <p className="ui-dim">{error.hint}</p>}
        </Note>
      )}

      {result && (
        <div className="ladder-result">
          <div className="ladder-meta ui-dim">
            source #{result.source_index + 1} of {result.n_sources} ·{" "}
            {result.n_radial_coils} radial coils · peak field in fT per nA·m
          </div>
          <table className="ui-table mono">
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
          </table>

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
