"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Icon, { type IconName } from "@/components/ui/Icon";
import {
  NODE_H,
  NODE_W,
  isCore,
  bounds,
  nodeCentre,
  wirePath,
  type JourneyEdge,
  type JourneyNode,
  type NodeState,
} from "@/lib/journey";

export interface CanvasView {
  x: number;
  y: number;
  scale: number;
}

interface Props {
  nodes: JourneyNode[];
  edges: JourneyEdge[];
  states: Record<string, NodeState>;
  /** Short readout under the title, e.g. "3 placed", "480 sensors". */
  badges: Record<string, string | undefined>;
  /** Figure nodes: preview image URL, once the figure exists on disk. */
  previews: Record<string, string | undefined>;
  selected: string | null;
  onSelect: (id: string | null) => void;
  /** Open a node's step page (a press that wasn't a drag). */
  onOpen: (id: string) => void;
  /** Commit a dragged position, in canvas units. */
  onMove: (id: string, x: number, y: number) => void;
  /** Remove a node. Only offered for nodes that are deletable. */
  onRemove: (id: string) => void;
  /** Run just this node's stage(s), leaving the rest of the journey alone. */
  onRunNode: (stages: string[]) => void;
  /** True while the pipeline runs — live wires carry a travelling pulse. */
  running: boolean;
  /**
   * The one card the user should touch next, if there is an obvious one. It is
   * flagged on the canvas so a first run has somewhere to start rather than six
   * equally-weighted cards and a disabled Run button.
   */
  next?: string | null;
  /**
   * Bumped by the shell to refit the view (the F shortcut and the palette both
   * ask for it). A counter rather than a callback ref keeps the canvas's own
   * view state private.
   */
  fitSignal?: number;
}

const MIN_SCALE = 0.35;
const MAX_SCALE = 1.8;
// Pointer travel (screen px) past which a press counts as a drag, not a click.
const DRAG_SLOP = 4;

// Core pipeline stages cannot be deleted — a FEM run is made of exactly
// those. Everything else on the canvas the user added, so it can go.
export function isDeletable(n: JourneyNode): boolean {
  return !isCore(n.id);
}

export default function JourneyCanvas({
  nodes,
  edges,
  states,
  badges,
  previews,
  selected,
  onSelect,
  onOpen,
  onMove,
  onRemove,
  onRunNode,
  running,
  next = null,
  fitSignal = 0,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<CanvasView>({ x: 0, y: 0, scale: 1 });
  const [panning, setPanning] = useState(false);
  const panFrom = useRef<{ px: number; py: number; vx: number; vy: number } | null>(
    null,
  );
  // The card under the pointer, with enough state to tell a click from a drag
  // and to place the card while it moves.
  const dragRef = useRef<{
    id: string;
    px: number;
    py: number;
    ox: number;
    oy: number;
    moved: boolean;
  } | null>(null);
  const [dragging, setDragging] = useState<{ id: string; x: number; y: number } | null>(
    null,
  );
  const fitted = useRef(false);

  // Nodes that just finished, so the canvas can mark the moment rather than
  // silently swapping a spinner for a tick.
  const prevStates = useRef<Record<string, NodeState>>({});
  const [justDone, setJustDone] = useState<Set<string>>(new Set());

  useEffect(() => {
    const finished: string[] = [];
    for (const [id, st] of Object.entries(states)) {
      const before = prevStates.current[id];
      if ((st === "done" || st === "skipped") && before && before !== st) {
        finished.push(id);
      }
    }
    prevStates.current = { ...states };
    if (finished.length === 0) return;
    setJustDone((cur) => new Set([...cur, ...finished]));
    const timer = window.setTimeout(() => {
      setJustDone((cur) => {
        const next = new Set(cur);
        for (const id of finished) next.delete(id);
        return next;
      });
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [states]);

  const fit = useCallback(() => {
    const el = wrapRef.current;
    if (!el || nodes.length === 0) return;
    const b = bounds(nodes);
    const pad = 130;
    const w = el.clientWidth || 1;
    const h = el.clientHeight || 1;
    const scale = Math.min(
      MAX_SCALE,
      Math.max(
        MIN_SCALE,
        Math.min(w / (b.maxX - b.minX + pad * 2), h / (b.maxY - b.minY + pad * 2)),
      ),
    );
    setView({
      scale,
      x: w / 2 - ((b.minX + b.maxX) / 2) * scale,
      y: h / 2 - ((b.minY + b.maxY) / 2) * scale,
    });
  }, [nodes]);

  useEffect(() => {
    if (fitted.current) return;
    fitted.current = true;
    // One frame later the wrapper has been laid out and has a real size.
    const id = window.requestAnimationFrame(fit);
    return () => window.cancelAnimationFrame(id);
  }, [fit]);

  // Wheel zooms about the pointer so the card under the cursor stays put.
  // Registered non-passively: React's onWheel cannot preventDefault, which let
  // a trackpad pinch zoom the whole page instead of the canvas.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      setView((v) => {
        const factor = Math.exp(-e.deltaY * 0.0015);
        const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
        const k = scale / v.scale;
        return { scale, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Delete removes the selected node, when that node is one of the removable
  // ones. Ignored while typing, so a Backspace in a field never eats a node.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const node = nodes.find((n) => n.id === selected);
      if (node && isDeletable(node)) {
        e.preventDefault();
        onRemove(node.id);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [nodes, selected, onRemove]);

  /* ── pointer ─────────────────────────────────────────────────────────── */

  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest(".jnode-wrap")) return;
    panFrom.current = { px: e.clientX, py: e.clientY, vx: view.x, vy: view.y };
    setPanning(true);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const from = panFrom.current;
    if (!from) return;
    setView((v) => ({
      ...v,
      x: from.vx + (e.clientX - from.px),
      y: from.vy + (e.clientY - from.py),
    }));
  };

  const endPan = () => {
    panFrom.current = null;
    setPanning(false);
  };

  /* ── dragging a card ─────────────────────────────────────────────────── */
  //
  // The card captures the pointer on press, so the whole gesture is delivered
  // to it — dragging keeps working when the cursor outruns the card, crosses
  // another card, or leaves the window. The canvas never sees these events, so
  // a card drag can't be mistaken for a pan.

  const startDrag = (e: React.PointerEvent, n: JourneyNode) => {
    e.stopPropagation();
    onSelect(n.id);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = {
      id: n.id,
      px: e.clientX,
      py: e.clientY,
      ox: n.x,
      oy: n.y,
      moved: false,
    };
  };

  const moveDrag = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (
      !drag.moved &&
      Math.hypot(e.clientX - drag.px, e.clientY - drag.py) < DRAG_SLOP
    ) {
      return;
    }
    drag.moved = true;
    setDragging({
      id: drag.id,
      x: drag.ox + (e.clientX - drag.px) / view.scale,
      y: drag.oy + (e.clientY - drag.py) / view.scale,
    });
  };

  const endDrag = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
    // A press that never travelled is a click: open the step.
    if (drag.moved) {
      if (dragging) onMove(drag.id, dragging.x, dragging.y);
    } else {
      onOpen(drag.id);
    }
    dragRef.current = null;
    setDragging(null);
  };

  const zoomBy = (factor: number) =>
    setView((v) => {
      const el = wrapRef.current;
      const px = (el?.clientWidth ?? 0) / 2;
      const py = (el?.clientHeight ?? 0) / 2;
      const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
      const k = scale / v.scale;
      return { scale, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
    });

  // The card being dragged is drawn at its live position, so its wires follow
  // it in real time rather than snapping at drop.
  const placed = nodes.map((n) =>
    dragging && dragging.id === n.id ? { ...n, x: dragging.x, y: dragging.y } : n,
  );
  const byId = new Map(placed.map((n) => [n.id, n]));
  const transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;

  return (
    <div
      className={`jcanvas${panning ? " jcanvas--panning" : ""}${
        dragging ? " jcanvas--dragging" : ""
      }${running ? " jcanvas--running" : ""}`}
      ref={wrapRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      onClick={(e) => {
        if (!(e.target as HTMLElement).closest(".jnode-wrap")) onSelect(null);
      }}
    >
      <div
        className="jcanvas-grid"
        style={{
          backgroundPosition: `${view.x}px ${view.y}px`,
          backgroundSize: `${26 * view.scale}px ${26 * view.scale}px`,
        }}
      />

      <svg className="jwires" aria-hidden>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
          {edges.map((e) => {
            const a = byId.get(e.from);
            const b = byId.get(e.to);
            if (!a || !b) return null;
            const d = wirePath(a, b);
            const settled = (id: string) =>
              states[id] === "done" || states[id] === "skipped";
            const live = settled(e.from) && (settled(e.to) || states[e.to] === "active");
            const charging = settled(e.from) && states[e.to] === "active";
            const mid = {
              x: (nodeCentre(a).x + nodeCentre(b).x) / 2,
              y: (nodeCentre(a).y + nodeCentre(b).y) / 2,
            };
            return (
              <g
                key={`${e.from}-${e.to}`}
                className={`jwire ${live ? "jwire--live" : "jwire--idle"}${
                  charging ? " jwire--charging" : ""
                }`}
              >
                <path className="jwire-glow" d={d} />
                <path className="jwire-line" d={d} />
                {running && live && <path className="jwire-pulse" d={d} />}
                {e.label && (
                  <text className="jwire-label" x={mid.x} y={mid.y - 9}>
                    {e.label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      <div className="jnodes" style={{ transform }}>
        {placed.map((n) => {
          const state = states[n.id] ?? "idle";
          const preview = previews[n.id];
          const isFigure = n.kind === "figure";
          const removable = isDeletable(n);
          return (
            <div
              key={n.id}
              className={`jnode-wrap${
                dragging?.id === n.id ? " jnode-wrap--dragging" : ""
              }`}
              style={{ left: n.x, top: n.y, width: NODE_W }}
            >
              <span className="jnode-ghost jnode-ghost--2" aria-hidden />
              <span className="jnode-ghost jnode-ghost--1" aria-hidden />

              <div
                role="button"
                tabIndex={0}
                className={`jnode jnode--${state}${
                  selected === n.id ? " jnode--selected" : ""
                }${isFigure ? " jnode--figure" : ""}${
                  justDone.has(n.id) ? " jnode--justdone" : ""
                }`}
                style={{ height: NODE_H }}
                aria-label={`${n.title}, ${state}. Enter opens this step; arrow keys move it.`}
                onPointerDown={(e) => startDrag(e, n)}
                onPointerMove={moveDrag}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
                // Keyboard parity: open with Enter/Space, nudge with arrows, so
                // the canvas works without a pointer at all.
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(n.id);
                    onOpen(n.id);
                    return;
                  }
                  const step = e.shiftKey ? 40 : 10;
                  const nudge: Record<string, [number, number]> = {
                    ArrowLeft: [-step, 0],
                    ArrowRight: [step, 0],
                    ArrowUp: [0, -step],
                    ArrowDown: [0, step],
                  };
                  const d = nudge[e.key];
                  if (d) {
                    e.preventDefault();
                    onMove(n.id, n.x + d[0], n.y + d[1]);
                  }
                }}
              >
                <span className="jnode-bloom jnode-bloom--in" aria-hidden />
                <span className="jnode-bloom jnode-bloom--out" aria-hidden />
                {justDone.has(n.id) && <span className="jnode-ring" aria-hidden />}
                {state === "active" && <span className="jnode-sweep" aria-hidden />}

                {isFigure && preview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img className="jnode-thumb" src={preview} alt="" />
                ) : (
                  <>
                    <span className="jnode-icon" aria-hidden>
                      <Icon name={n.icon as IconName} size={20} />
                    </span>
                    <span className="jnode-text">
                      <span className="jnode-title">{n.title}</span>
                      <span className="jnode-caption">
                        {badges[n.id] ?? n.caption}
                      </span>
                    </span>
                  </>
                )}

                {state === "done" && (
                  <span className="jnode-tick" aria-hidden>
                    <Icon name="check" size={12} />
                  </span>
                )}
                {state === "skipped" && (
                  <span className="jnode-tick jnode-tick--soft" aria-hidden>
                    <Icon name="check" size={12} />
                  </span>
                )}
                {state === "active" && <span className="jnode-spin" aria-hidden />}
                {state === "failed" && (
                  <span className="jnode-tick jnode-tick--bad" aria-hidden>
                    <Icon name="alert" size={12} />
                  </span>
                )}
              </div>

              {isFigure && (
                <div className="jnode-sublabel">
                  <span className="jnode-title">{n.title}</span>
                  <span className="jnode-caption">{n.caption}</span>
                </div>
              )}

              {n.stage && (
                <button
                  type="button"
                  className="jnode-run"
                  title={`Run just this step (${n.stage})`}
                  aria-label={`Run just the ${n.title} step`}
                  disabled={running}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRunNode([n.stage as string]);
                  }}
                >
                  <Icon name="play" size={10} />
                </button>
              )}

              {removable && (
                <button
                  type="button"
                  className="jnode-remove"
                  title={`Remove the ${n.title} node`}
                  aria-label={`Remove the ${n.title} node`}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(n.id);
                  }}
                >
                  <Icon name="close" size={11} />
                </button>
              )}
              {!removable && (
                <span className="jnode-core" title="Core stage — every run needs it">
                  core
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div className="jzoom">
        <button type="button" onClick={() => zoomBy(1.2)} aria-label="Zoom in">
          <Icon name="plus" size={13} />
        </button>
        <button type="button" onClick={() => zoomBy(1 / 1.2)} aria-label="Zoom out">
          <Icon name="minus" size={13} />
        </button>
        <button type="button" onClick={fit} aria-label="Fit the journey to the view">
          <Icon name="fit" size={13} />
        </button>
        <span className="jzoom-read mono">{Math.round(view.scale * 100)}%</span>
      </div>
    </div>
  );
}
