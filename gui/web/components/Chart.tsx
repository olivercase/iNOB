"use client";

import { useState, type ReactNode } from "react";

/**
 * The chart set: plain SVG in the console's own language.
 *
 * The figures a run writes are rasters, and a raster cannot answer a question.
 * Anything the app already holds as numbers — the detection result, the ladder
 * rungs — is drawn here instead, so a value can be hovered, read exactly, and
 * compared without leaving the page.
 *
 * Two series colours, and only two, both validated against this surface for
 * lightness, chroma, colour-vision separation and contrast. Identity never
 * rests on colour alone: every series is also directly labelled.
 */

export const SERIES = ["#23a95c", "#2f86c4"] as const;

export interface Bar {
  label: string;
  /** Bars in order; one entry per series. */
  values: number[];
  /** Shown in the tooltip instead of the raw number, when given. */
  display?: string[];
  /** Marks a bar as beyond measurement (e.g. never detectable). */
  unbounded?: boolean;
}

/* A horizontal bar chart. Horizontal because the labels are words — source
   numbers, model names — and words read along the axis they are written on. */
export function BarChart({
  bars,
  series,
  unit,
  max,
  reference,
}: {
  bars: Bar[];
  /** Series names, in the same order as each bar's values. */
  series: string[];
  unit?: string;
  /** Force the axis top; otherwise the largest value sets it. */
  max?: number;
  /** A line across the plot, e.g. the detection threshold. */
  reference?: { value: number; label: string };
}) {
  const [hover, setHover] = useState<{ bar: number; series: number } | null>(null);

  const values = bars.flatMap((b) => b.values.filter(Number.isFinite));
  const top = Math.max(max ?? 0, ...values, reference?.value ?? 0) || 1;
  const rowH = series.length > 1 ? 34 : 26;
  const gap = 10;
  const height = bars.length * (rowH + gap);

  return (
    <figure className="chart">
      {series.length > 1 && (
        <figcaption className="chart-legend">
          {series.map((name, i) => (
            <span key={name} className="chart-key">
              <span
                className="chart-swatch"
                style={{ background: SERIES[i % SERIES.length] }}
                aria-hidden
              />
              {name}
            </span>
          ))}
        </figcaption>
      )}

      <div className="chart-plot" style={{ height }}>
        {reference && (
          <span
            className="chart-ref"
            style={{ left: `${(reference.value / top) * 100}%` }}
          >
            <span className="chart-ref-label mono">{reference.label}</span>
          </span>
        )}

        {bars.map((bar, i) => (
          <div key={bar.label} className="chart-row" style={{ height: rowH }}>
            <span className="chart-row-label">{bar.label}</span>
            <div className="chart-row-track">
              {bar.values.map((value, s) => {
                const finite = Number.isFinite(value) && value >= 0;
                const pct = finite ? Math.max((value / top) * 100, 1.5) : 100;
                const shown =
                  bar.display?.[s] ??
                  (finite ? `${value}${unit ? ` ${unit}` : ""}` : "∞");
                const on = hover?.bar === i && hover?.series === s;
                return (
                  <div key={s} className="chart-mark-line">
                    <div
                      className={`chart-mark${finite ? "" : " chart-mark--unbounded"}${
                        on ? " chart-mark--on" : ""
                      }`}
                      style={{
                        width: `${pct}%`,
                        background: finite ? SERIES[s % SERIES.length] : undefined,
                      }}
                      onMouseEnter={() => setHover({ bar: i, series: s })}
                      onMouseLeave={() => setHover(null)}
                      tabIndex={0}
                      role="img"
                      aria-label={`${bar.label}${
                        series.length > 1 ? ` ${series[s]}` : ""
                      }: ${shown}`}
                    />
                    <span className="chart-value mono">{shown}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}

/* The one number that answers the question, with its unit and a line of
   context under it. Not every result needs a plot. */
export function StatTile({
  value,
  unit,
  note,
  tone = "signal",
}: {
  value: string;
  unit: string;
  note?: ReactNode;
  tone?: "signal" | "flat";
}) {
  return (
    <div className={`stat stat--${tone}`}>
      <div className="stat-value mono">{value}</div>
      <div className="stat-unit">{unit}</div>
      {note && <p className="stat-note">{note}</p>}
    </div>
  );
}
