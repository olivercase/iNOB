"use client";

// Transient feedback, and the only place the app can offer to undo something.
//
// The store is a module singleton rather than a React context: a toast is
// raised from event handlers, promise callbacks and the run stream alike, and
// threading a provider through all of those only to reach one host at the
// shell would be plumbing for its own sake. There is exactly one <ToastHost/>.

import { useEffect, useState } from "react";
import Icon, { type IconName } from "./Icon";

export type ToastTone = "info" | "ok" | "warn" | "danger";

export interface ToastSpec {
  tone?: ToastTone;
  title: string;
  /** One line of detail under the title. Keep it to a clause. */
  body?: string;
  /** An offer to reverse what just happened. Extends the dwell time. */
  action?: { label: string; icon?: IconName; onAct: () => void };
  /** Milliseconds on screen. Defaults are longer when an action is offered. */
  ms?: number;
}

interface Item extends ToastSpec {
  id: number;
}

// Stacking more than this is noise rather than feedback, so the oldest is
// dropped to make room.
const MAX_STACK = 3;

let seq = 0;
let items: Item[] = [];
const listeners = new Set<(v: Item[]) => void>();
const timers = new Map<number, number>();

function emit() {
  for (const l of listeners) l(items);
}

export function dismissToast(id: number) {
  const t = timers.get(id);
  if (t !== undefined) window.clearTimeout(t);
  timers.delete(id);
  items = items.filter((i) => i.id !== id);
  emit();
}

export function toast(spec: ToastSpec): number {
  const id = ++seq;
  items = [...items, { ...spec, id }];
  while (items.length > MAX_STACK) {
    const oldest = items[0];
    const t = timers.get(oldest.id);
    if (t !== undefined) window.clearTimeout(t);
    timers.delete(oldest.id);
    items = items.slice(1);
  }
  emit();
  const ms = spec.ms ?? (spec.action ? 9000 : 4200);
  timers.set(
    id,
    window.setTimeout(() => dismissToast(id), ms),
  );
  return id;
}

const TONE_ICON: Record<ToastTone, IconName> = {
  info: "info",
  ok: "check",
  warn: "alert",
  danger: "alert",
};

export function ToastHost() {
  const [list, setList] = useState<Item[]>(items);

  useEffect(() => {
    listeners.add(setList);
    // The store may have been written to between module load and mount.
    setList(items);
    return () => {
      listeners.delete(setList);
    };
  }, []);

  if (list.length === 0) return null;

  return (
    <div className="jtoasts" role="region" aria-label="Notifications">
      {list.map((t) => {
        const tone = t.tone ?? "info";
        return (
          <div
            key={t.id}
            className={`jtoast jtoast--${tone}`}
            role="status"
            aria-live="polite"
          >
            <span className="jtoast-icon" aria-hidden>
              <Icon name={TONE_ICON[tone]} size={14} />
            </span>
            <span className="jtoast-text">
              <strong>{t.title}</strong>
              {t.body && <span>{t.body}</span>}
            </span>
            {t.action && (
              <button
                type="button"
                className="jtoast-act"
                onClick={() => {
                  t.action?.onAct();
                  dismissToast(t.id);
                }}
              >
                {t.action.icon && <Icon name={t.action.icon} size={12} />}
                {t.action.label}
              </button>
            )}
            <button
              type="button"
              className="jtoast-x"
              onClick={() => dismissToast(t.id)}
              aria-label="Dismiss"
            >
              <Icon name="close" size={11} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
