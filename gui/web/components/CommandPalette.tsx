"use client";

// Every action in the app, reachable from the keyboard in one gesture.
//
// The canvas hides most of its verbs behind a card you must first find and
// open. The palette is the flat index of the same verbs, so a user who knows
// what they want ("compare", "cluster", "log") never has to hunt the graph
// for it.

import { useEffect, useMemo, useRef, useState } from "react";
import Icon, { type IconName } from "@/components/ui/Icon";

export interface Command {
  id: string;
  title: string;
  /** Where this lives, e.g. "Steps" or "Run". Groups the list. */
  group: string;
  icon: IconName;
  /** Right-aligned trailing text: the shortcut, or why it is unavailable. */
  hint?: string;
  blurb?: string;
  disabled?: boolean;
  run: () => void;
}

interface Props {
  open: boolean;
  onClose: () => void;
  commands: Command[];
}

/**
 * Subsequence match, the behaviour every palette has trained users to expect:
 * "cmp fw" finds "Compare forward models". Returns a rank (lower is better) so
 * tighter, earlier matches float up, or null when it does not match at all.
 */
function score(query: string, text: string): number | null {
  if (!query) return 0;
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const direct = t.indexOf(q);
  if (direct >= 0) return direct; // contiguous hits always beat scattered ones

  let at = 0;
  let spread = 0;
  let last = -1;
  for (const ch of q) {
    if (ch === " ") continue;
    const found = t.indexOf(ch, at);
    if (found < 0) return null;
    if (last >= 0) spread += found - last;
    last = found;
    at = found + 1;
  }
  return 1000 + spread;
}

export default function CommandPalette({ open, onClose, commands }: Props) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Each opening starts clean — a stale query from last time is never what
  // the user meant by pressing the shortcut again.
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCursor(0);
    inputRef.current?.focus();
  }, [open]);

  const matches = useMemo(() => {
    const scored = commands
      .map((c) => ({
        c,
        rank: score(query, `${c.title} ${c.group} ${c.blurb ?? ""}`),
      }))
      .filter((m): m is { c: Command; rank: number } => m.rank !== null);
    // Disabled commands stay listed but sink, so the palette still answers
    // "can I do X?" with "yes, but not yet — here is why".
    scored.sort(
      (a, b) =>
        Number(a.c.disabled) - Number(b.c.disabled) || a.rank - b.rank,
    );
    return scored.map((m) => m.c);
  }, [commands, query]);

  const clamped = Math.min(cursor, Math.max(0, matches.length - 1));

  useEffect(() => {
    listRef.current
      ?.querySelector("[data-active='true']")
      ?.scrollIntoView({ block: "nearest" });
  }, [clamped, matches.length]);

  if (!open) return null;

  const choose = (c: Command) => {
    if (c.disabled) return;
    onClose();
    c.run();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || (e.key === "n" && e.ctrlKey)) {
      e.preventDefault();
      setCursor((i) => (matches.length ? (i + 1) % matches.length : 0));
    } else if (e.key === "ArrowUp" || (e.key === "p" && e.ctrlKey)) {
      e.preventDefault();
      setCursor((i) =>
        matches.length ? (i - 1 + matches.length) % matches.length : 0,
      );
    } else if (e.key === "Enter") {
      e.preventDefault();
      const c = matches[clamped];
      if (c) choose(c);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  // Group headers are drawn inline as the list is walked, so filtering never
  // leaves an empty heading behind.
  let lastGroup = "";

  return (
    <div className="jpal-scrim" onMouseDown={onClose}>
      <div
        className="jpal"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="jpal-search">
          <Icon name="search" size={15} />
          <input
            ref={inputRef}
            value={query}
            placeholder="Search actions…"
            aria-label="Search actions"
            role="combobox"
            aria-expanded
            aria-controls="jpal-list"
            onChange={(e) => {
              setQuery(e.currentTarget.value);
              setCursor(0);
            }}
          />
          <kbd className="jkbd">esc</kbd>
        </div>

        <div className="jpal-list" id="jpal-list" role="listbox" ref={listRef}>
          {matches.length === 0 && (
            <p className="jempty">No action matches “{query}”.</p>
          )}
          {matches.map((c, i) => {
            const header = c.group !== lastGroup ? c.group : null;
            lastGroup = c.group;
            const active = i === clamped;
            return (
              <div key={c.id}>
                {header && <p className="jgroup-head">{header}</p>}
                <button
                  type="button"
                  role="option"
                  aria-selected={active}
                  data-active={active}
                  className={`jpal-row${active ? " jpal-row--active" : ""}${
                    c.disabled ? " jpal-row--off" : ""
                  }`}
                  // Pointer-down would fire before the click that dismisses
                  // the scrim, so use click and stop it reaching the scrim.
                  onClick={(e) => {
                    e.stopPropagation();
                    choose(c);
                  }}
                  onMouseMove={() => setCursor(i)}
                >
                  <span className="jrow-icon" aria-hidden>
                    <Icon name={c.icon} size={15} />
                  </span>
                  <span className="jrow-text">
                    <span className="jrow-title">{c.title}</span>
                    {c.blurb && <span className="jrow-blurb">{c.blurb}</span>}
                  </span>
                  {c.hint && <span className="jpal-hint mono">{c.hint}</span>}
                </button>
              </div>
            );
          })}
        </div>

        <div className="jpal-foot">
          <span>
            <kbd className="jkbd">↑</kbd>
            <kbd className="jkbd">↓</kbd> move
          </span>
          <span>
            <kbd className="jkbd">↵</kbd> run
          </span>
          <span>
            <kbd className="jkbd">?</kbd> all shortcuts
          </span>
        </div>
      </div>
    </div>
  );
}
