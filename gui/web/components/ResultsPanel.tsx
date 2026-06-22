"use client";

import type { DetectResult } from "@/lib/api";

function fmtTrials(n: number): string {
  if (n < 0 || !Number.isFinite(n)) return "∞";
  return n.toLocaleString();
}

export default function ResultsPanel({ result }: { result: DetectResult | null }) {
  if (!result) {
    return (
      <div className="panel">
        <h3>Detectability</h3>
        <p className="hint">Run a simulation to estimate trials-to-detect.</p>
      </div>
    );
  }

  const trials = result.per_source.map((p) => p.trials_needed).filter((n) => n >= 0);
  const worst = trials.length ? Math.max(...trials) : -1;
  const best = trials.length ? Math.min(...trials) : -1;

  return (
    <div className="panel">
      <h3>Detectability {result.mocked && <span className="mock">· MOCK (no backend result)</span>}</h3>
      <div className="row">
        <div>
          <div className="hint">trials to detect (best → worst source)</div>
          <div className="big">{fmtTrials(best)} – {fmtTrials(worst)}</div>
        </div>
      </div>
      <p className="hint">
        threshold SNR {result.array.threshold_snr} · noise floor {result.array.noise_floor_fT} fT ·
        {" "}{result.array.n_sensors} sensors · mean single-trial SNR {result.array.mean_snr}
      </p>
      <table className="results">
        <thead>
          <tr>
            <th>#</th><th>x</th><th>y</th><th>z</th><th>Q (nA·m)</th><th>SNR</th><th>trials</th>
          </tr>
        </thead>
        <tbody>
          {result.per_source.map((p) => (
            <tr key={p.index}>
              <td>{p.index + 1}</td>
              <td>{p.x}</td><td>{p.y}</td><td>{p.z}</td>
              <td>{p.strength_nAm}</td>
              <td>{p.snr}</td>
              <td>{fmtTrials(p.trials_needed)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
