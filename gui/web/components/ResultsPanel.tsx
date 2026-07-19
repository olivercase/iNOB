"use client";

import { Callout, HTMLTable } from "@blueprintjs/core";
import type { DetectResult, DetectUnavailable } from "@/lib/api";

function fmtTrials(n: number): string {
  if (n < 0 || !Number.isFinite(n)) return "∞";
  return n.toLocaleString();
}

// One plain sentence a planner can act on, instead of leaving them to read the
// number cold.
function verdict(best: number, worst: number): string {
  if (best < 0) return "Not detectable at this threshold, even averaging forever.";
  if (worst <= 1) return "Detectable in a single trial — no averaging needed.";
  if (best === worst) return `Detectable after about ${fmtTrials(best)} averaged trials.`;
  return `Detectable after ${fmtTrials(best)} to ${fmtTrials(worst)} averaged ` +
    "trials, depending on the source.";
}

export default function ResultsPanel({
  result,
  unavailable,
}: {
  result: DetectResult | null;
  unavailable?: DetectUnavailable | null;
}) {
  if (unavailable && !result) {
    return (
      <Callout intent="warning" title="No result yet" icon="info-sign">
        <p>{unavailable.reason}</p>
        {unavailable.hint && <p className="bp6-text-muted">{unavailable.hint}</p>}
      </Callout>
    );
  }

  if (!result) {
    return (
      <Callout icon="pulse" className="bp6-text-muted">
        Place a source and run the simulation to see how many averaged trials it
        takes to detect it.
      </Callout>
    );
  }

  const trials = result.per_source
    .map((p) => p.trials_needed)
    .filter((n) => n >= 0);
  const worst = trials.length ? Math.max(...trials) : -1;
  const best = trials.length ? Math.min(...trials) : -1;

  return (
    <div>
      <div className="result-headline">
        <div className="result-number">
          {fmtTrials(best)}
          {worst !== best && <span className="result-range"> – {fmtTrials(worst)}</span>}
        </div>
        <div className="result-unit">averaged trials to detect</div>
      </div>

      <p className="result-verdict">{verdict(best, worst)}</p>

      <Callout compact intent="primary" style={{ margin: "10px 0" }}>
        {result.modality.toUpperCase()} · {result.array.n_sensors} sensors ·
        detection threshold SNR {result.array.threshold_snr} · noise floor{" "}
        {result.array.noise_floor_fT} {result.array.noise_unit ?? "fT"} · mean
        single-trial SNR {result.array.mean_snr}
      </Callout>

      {result.per_source.length > 1 && (
        <HTMLTable compact striped className="mono" style={{ width: "100%" }}>
          <thead>
            <tr>
              <th>source</th>
              <th>single-trial SNR</th>
              <th>trials to detect</th>
            </tr>
          </thead>
          <tbody>
            {result.per_source.map((p) => (
              <tr key={p.index}>
                <td>#{p.index + 1}</td>
                <td>{p.snr}</td>
                <td>{fmtTrials(p.trials_needed)}</td>
              </tr>
            ))}
          </tbody>
        </HTMLTable>
      )}
    </div>
  );
}
