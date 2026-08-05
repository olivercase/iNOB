"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Icon, IconButton } from "@/components/ui";
import type { FigureInfo } from "@/lib/api";

/**
 * A run's figures are PNGs on disk, so the picture cannot be restyled — but
 * looking at one can be. This gives a rendered figure the same handling as the
 * canvas: wheel to zoom about the pointer, drag to pan, fit or 1:1, and a
 * plate-invert for the white-background plots matplotlib writes, so a figure
 * can sit in a dark room without being the only bright thing in it.
 *
 * The invert is a viewing aid and says so. It never changes the file, and the
 * download is always the original.
 */

const MIN = 0.2;
const MAX = 8;

export default function FigureView({ figure }: { figure: FigureInfo }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const [fitted, setFitted] = useState(true);
  const [inverted, setInverted] = useState(false);
  const [dragging, setDragging] = useState(false);
  const from = useRef<{ px: number; py: number; vx: number; vy: number } | null>(null);

  const src = `${figure.url}?v=${Math.round(figure.mtime)}`;

  const fit = useCallback(() => {
    setView({ x: 0, y: 0, scale: 1 });
    setFitted(true);
  }, []);

  const zoomTo = useCallback((scale: number) => {
    setView((v) => ({ ...v, scale: Math.min(MAX, Math.max(MIN, scale)) }));
    setFitted(false);
  }, []);

  // Wheel zooms about the pointer, so the detail under the cursor stays put.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left - rect.width / 2;
      const py = e.clientY - rect.top - rect.height / 2;
      setView((v) => {
        const scale = Math.min(MAX, Math.max(MIN, v.scale * Math.exp(-e.deltaY * 0.0015)));
        const k = scale / v.scale;
        return { scale, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
      });
      setFitted(false);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div className="figview">
      <div
        className={`figview-stage${dragging ? " figview-stage--dragging" : ""}`}
        ref={wrapRef}
        onPointerDown={(e) => {
          from.current = { px: e.clientX, py: e.clientY, vx: view.x, vy: view.y };
          setDragging(true);
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const start = from.current;
          if (!start) return;
          setView((v) => ({
            ...v,
            x: start.vx + (e.clientX - start.px),
            y: start.vy + (e.clientY - start.py),
          }));
          setFitted(false);
        }}
        onPointerUp={() => {
          from.current = null;
          setDragging(false);
        }}
        onDoubleClick={() => (fitted ? zoomTo(2) : fit())}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          className={`figview-img${fitted ? " figview-img--fit" : ""}${
            inverted ? " figview-img--inverted" : ""
          }`}
          src={src}
          alt={figure.label}
          draggable={false}
          style={
            fitted
              ? undefined
              : {
                  transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
                }
          }
        />
      </div>

      <div className="figview-bar">
        <span className="figview-name">{figure.label}</span>
        <span className="figview-meta mono">
          {(figure.bytes / 1024).toFixed(0)} kB ·{" "}
          {new Date(figure.mtime * 1000).toLocaleString()}
        </span>

        <span className="figview-tools">
          <IconButton
            name="minus"
            label="Zoom out"
            size={14}
            onClick={() => zoomTo(view.scale / 1.3)}
          />
          <span className="figview-zoom mono">
            {fitted ? "fit" : `${Math.round(view.scale * 100)}%`}
          </span>
          <IconButton
            name="plus"
            label="Zoom in"
            size={14}
            onClick={() => zoomTo(view.scale * 1.3)}
          />
          <IconButton name="fit" label="Fit to view" size={14} onClick={fit} />
          <IconButton
            name="figure"
            label={inverted ? "Show original colours" : "Invert for dark viewing"}
            size={14}
            active={inverted}
            onClick={() => setInverted((i) => !i)}
          />
          <a
            className="figview-open"
            href={figure.url}
            target="_blank"
            rel="noreferrer"
            title="Open the original file"
          >
            <Icon name="external" size={14} />
          </a>
        </span>
      </div>

      {inverted && (
        <p className="figview-note">
          Colours inverted for viewing only — the file itself is unchanged.
        </p>
      )}
    </div>
  );
}
