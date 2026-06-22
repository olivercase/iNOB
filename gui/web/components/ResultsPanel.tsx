"use client";

import { Callout, Card, HTMLTable, Tag } from "@blueprintjs/core";
import type { DetectResult } from "@/lib/api";

function fmtTrials(n: number): string {
  if (n < 0 || !Number.isFinite(n)) return "∞";
  return n.toLocaleString();
}

export default function ResultsPanel({ result }: { result: DetectResult | null }) {
  if (!result) {
    return (
      <Card className="panel" compact>
        <h3 className="section-title">Detectability</h3>
        <p className="bp6-text-muted">Run a simulation to estimate trials-to-detect.</p>
      </Card>
    );
  }

  const trials = result.per_source.map((p) => p.trials_needed).filter((n) => n >= 0);
  const worst = trials.length ? Math.max(...trials) : -1;
  const best = trials.length ? Math.min(...trials) : -1;

  return (
    <Card className="panel" compact>
      <h3 className="section-title">
        Detectability {result.mocked && <Tag intent="warning" minimal>MOCK</Tag>}
      </h3>
      <div className="bp6-text-muted">trials to detect (best → worst source)</div>
      <div className="big">{fmtTrials(best)} – {fmtTrials(worst)}</div>
      <Callout intent="primary" compact style={{ margin: "8px 0" }}>
        {result.modality.toUpperCase()} · threshold SNR {result.array.threshold_snr} ·
        noise {result.array.noise_floor_fT} {result.array.noise_unit ?? "fT"} ·
        {" "}{result.array.n_sensors} sensors · mean single-trial SNR {result.array.mean_snr}
      </Callout>
      <HTMLTable compact striped className="mono" style={{ width: "100%" }}>
        <thead>
          <tr><th>#</th><th>x</th><th>y</th><th>z</th><th>Q</th><th>SNR</th><th>trials</th></tr>
        </thead>
        <tbody>
          {result.per_source.map((p) => (
            <tr key={p.index}>
              <td>{p.index + 1}</td>
              <td>{p.x}</td><td>{p.y}</td><td>{p.z}</td>
              <td>{p.strength_nAm}</td><td>{p.snr}</td><td>{fmtTrials(p.trials_needed)}</td>
            </tr>
          ))}
        </tbody>
      </HTMLTable>
    </Card>
  );
}
