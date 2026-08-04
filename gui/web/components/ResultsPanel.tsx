"use client";

import { Note } from "@/components/ui";
import type { DetectResult, DetectUnavailable } from "@/lib/api";

function fmtTrials(n: number): string {
  if (n < 0 || !Number.isFinite(n)) return "∞";
  return n.toLocaleString();
}

// One plain sentence a planner can act on, instead of leaving them to read the
// number cold. `nUndetectable` matters: a source that never reaches threshold
// is excluded from the min/max, so reporting the range alone would quietly
// describe only the sources that happen to work.
function verdict(
  best: number,
  worst: number,
  nUndetectable: number,
  nTotal: number,
): string {
  if (nTotal > 0 && nUndetectable === nTotal) {
    return "Not detectable at this threshold, even averaging forever.";
  }
  const tail =
    nUndetectable > 0
      ? ` ${nUndetectable} of ${nTotal} source${nTotal === 1 ? "" : "s"} ` +
        "cannot be detected at this threshold at all."
      : "";
  if (best < 0) return "Not detectable at this threshold, even averaging forever.";
  if (worst <= 1) return `Detectable in a single trial — no averaging needed.${tail}`;
  if (best === worst) {
    return `Detectable after about ${fmtTrials(best)} averaged trials.${tail}`;
  }
  return (
    `Detectable after ${fmtTrials(best)} to ${fmtTrials(worst)} averaged ` +
    `trials, depending on the source.${tail}`
  );
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
      <Note tone="warn" title="No result yet">
        <p>{unavailable.reason}</p>
        {unavailable.hint && <p className="ui-dim">{unavailable.hint}</p>}
      </Note>
    );
  }

  if (!result) {
    return (
      <Note>
        Place a source and run the journey to see how many averaged trials it
        takes to detect it.
      </Note>
    );
  }

  const all = result.per_source.map((p) => p.trials_needed);
  const trials = all.filter((n) => n >= 0 && Number.isFinite(n));
  const nUndetectable = all.length - trials.length;
  const worst = trials.length ? Math.max(...trials) : -1;
  const best = trials.length ? Math.min(...trials) : -1;

  return (
    <div>
      <div className="result-headline">
        <div className="result-number">
          {fmtTrials(best)}
          {worst !== best && <span className="result-range"> – {fmtTrials(worst)}</span>}
          {/* Undetectable sources are excluded from the range above, so say
              so right where the number is read rather than only in prose. */}
          {nUndetectable > 0 && trials.length > 0 && (
            <span className="result-range"> + ∞</span>
          )}
        </div>
        <div className="result-unit">averaged trials to detect</div>
      </div>

      <p className="result-verdict">
        {verdict(best, worst, nUndetectable, all.length)}
      </p>

      <div className="result-facts mono">
        {result.modality.toUpperCase()} · {result.array.n_sensors} sensors ·
        detection threshold SNR {result.array.threshold_snr} · noise floor{" "}
        {result.array.noise_floor_fT} {result.array.noise_unit ?? "fT"} · mean
        single-trial SNR {result.array.mean_snr}
      </div>

      {result.per_source.length > 1 && (
        <table className="ui-table mono">
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
        </table>
      )}
    </div>
  );
}
