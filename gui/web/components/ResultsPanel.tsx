"use client";

import { Button, Note } from "@/components/ui";
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
  sourceCount = 0,
  onBack,
}: {
  result: DetectResult | null;
  unavailable?: DetectUnavailable | null;
  /** How many sources are placed — decides which empty state is the true one. */
  sourceCount?: number;
  /** Return to the journey, where both missing things are done. */
  onBack?: () => void;
}) {
  if (unavailable && !result) {
    return (
      <Note tone="warn" title="No result yet">
        <p>{unavailable.reason}</p>
        {unavailable.hint && <p className="ui-dim">{unavailable.hint}</p>}
      </Note>
    );
  }

  /* An empty screen is an invitation, so it names the one thing missing and
     carries the verb for it. Which thing that is depends on the run: with no
     source there is nothing to solve; with sources placed the solve is simply
     the step not taken yet. Saying both at once would be true of neither. */
  if (!result) {
    return (
      <div className="jempty">
        <p className="jempty-head">
          {sourceCount === 0 ? "No source placed" : "Not run yet"}
        </p>
        <p className="jempty-line">
          {sourceCount === 0
            ? "Drop a dipole on the anatomy, then run the journey and the answer lands here."
            : `${sourceCount} source${sourceCount === 1 ? "" : "s"} placed. Run the journey and the answer lands here.`}
        </p>
        {onBack && (
          /* Outline, not filled: the bar already carries one filled accent,
             and two greens on one screen is two primaries, which is none. */
          <Button icon="arrow-left" onClick={onBack}>
            Back to journey
          </Button>
        )}
      </div>
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
