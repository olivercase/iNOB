"use client";

import { useMemo, useState } from "react";
import Icon from "@/components/ui/Icon";
import type { FigureInfo } from "@/lib/api";
import { ADDABLE, BASE_NODES, type AddableSpec } from "@/lib/journey";

const STAGE_LABEL: Record<string, string> = {
  geom: "Anatomy figures",
  fem: "FEM mesh figures",
  sensors: "Sensor figures",
  forward: "Forward-solve figures",
  viz: "Rendered figures",
};

const GROUP_LABEL: Record<AddableSpec["group"], string> = {
  sensing: "Sensing",
  model: "Forward models",
  physics: "Physics",
  compute: "Where it runs",
};

interface Props {
  figures: FigureInfo[];
  /** Ids already on the canvas — offered as "on canvas", not again. */
  added: Set<string>;
  onAddFigure: (figure: FigureInfo) => void;
  onAddNode: (spec: AddableSpec) => void;
  onClose: () => void;
}

export default function AddOutputPanel({
  figures,
  added,
  onAddFigure,
  onAddNode,
  onClose,
}: Props) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();

  const matches = (...text: string[]) =>
    !q || text.some((t) => t.toLowerCase().includes(q));

  const specs = useMemo(
    () =>
      ADDABLE.filter((a) =>
        matches(a.title, a.caption, a.blurb, GROUP_LABEL[a.group]),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [q],
  );

  const figureGroups = useMemo(() => {
    const hits = figures.filter((f) =>
      matches(f.label, f.blurb ?? "", STAGE_LABEL[f.stage] ?? f.stage),
    );
    const out = new Map<string, FigureInfo[]>();
    for (const f of hits) {
      const list = out.get(f.stage) ?? [];
      list.push(f);
      out.set(f.stage, list);
    }
    return [...out.entries()];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [figures, q]);

  const specGroups = useMemo(() => {
    const out = new Map<AddableSpec["group"], AddableSpec[]>();
    for (const a of specs) {
      const list = out.get(a.group) ?? [];
      list.push(a);
      out.set(a.group, list);
    }
    return [...out.entries()];
  }, [specs]);

  const nothing = specGroups.length === 0 && figureGroups.length === 0;

  return (
    <aside className="jpanel jpanel--add" aria-label="Add to the journey">
      <header className="jpanel-head">
        <h2>Add to the journey</h2>
        <button type="button" className="jpanel-x" onClick={onClose} aria-label="Close">
          <Icon name="close" size={13} />
        </button>
      </header>

      <label className="jsearch">
        <Icon name="search" size={14} />
        <input
          value={query}
          onChange={(e) => setQuery(e.currentTarget.value)}
          placeholder="Search models and outputs…"
          aria-label="Search models and outputs"
        />
      </label>

      <div className="jpanel-scroll">
        {nothing && <p className="jempty">Nothing matches “{query}”.</p>}

        {!q && (
          <section className="jgroup">
            <h3 className="jgroup-head">Already in the journey · core</h3>
            <p className="jgroup-note">
              Every run is made of these seven steps, so they can’t be removed.
              Open one from the canvas to set it.
            </p>
            <div className="jcorelist">
              {BASE_NODES.map((n) => (
                <span key={n.id} className="jcorechip">
                  <Icon name={n.icon} size={12} />
                  {n.title}
                </span>
              ))}
            </div>
          </section>
        )}

        {specGroups.map(([group, list]) => (
          <section key={group} className="jgroup">
            <h3 className="jgroup-head">{GROUP_LABEL[group]}</h3>
            {list.map((a) => {
              const on = added.has(a.id);
              return (
                <button
                  key={a.id}
                  type="button"
                  className={`jrow${on ? " jrow--on" : ""}`}
                  disabled={on}
                  onClick={() => onAddNode(a)}
                >
                  <span className="jrow-icon" aria-hidden>
                    <Icon name={a.icon} size={16} />
                  </span>
                  <span className="jrow-text">
                    <span className="jrow-title">{a.title}</span>
                    <span className="jrow-blurb">{a.blurb}</span>
                  </span>
                  <span className="jrow-go" aria-hidden>
                    <Icon name={on ? "check" : "arrow-right"} size={13} />
                  </span>
                </button>
              );
            })}
          </section>
        ))}

        {figureGroups.map(([stage, list]) => (
          <section key={stage} className="jgroup">
            <h3 className="jgroup-head">{STAGE_LABEL[stage] ?? stage}</h3>
            {list.map((f) => {
              const on = added.has(`fig:${f.key}`);
              return (
                <button
                  key={f.key}
                  type="button"
                  className={`jrow${on ? " jrow--on" : ""}`}
                  disabled={on}
                  onClick={() => onAddFigure(f)}
                >
                  <span className="jrow-icon" aria-hidden>
                    <Icon name="figure" size={16} />
                  </span>
                  <span className="jrow-text">
                    <span className="jrow-title">{f.label}</span>
                    <span className="jrow-blurb">
                      {f.exists
                        ? (f.blurb || "Ready to view")
                        : "Drawn by the next run"}
                    </span>
                  </span>
                  <span className="jrow-go" aria-hidden>
                    <Icon name={on ? "check" : "arrow-right"} size={13} />
                  </span>
                </button>
              );
            })}
          </section>
        ))}
      </div>

      <footer className="jpanel-foot">
        Anything you add can be dragged anywhere, and removed with its ✕ or the
        Delete key. A run draws only the figures pinned here — pin none and it
        builds the model and solves it, nothing more.
      </footer>
    </aside>
  );
}
