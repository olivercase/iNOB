"use client";

import type { ReactNode } from "react";

export type StepState = "todo" | "active" | "done";

interface StepProps {
  index: number;
  title: string;
  state: StepState;
  children: ReactNode;
}

// A single numbered step. The badge reflects state so a first-time user can see
// at a glance where they are: dim = not yet, highlighted = do this now,
// ticked = handled.
export function Step({ index, title, state, children }: StepProps) {
  return (
    <section className={`step step--${state}`}>
      <div className="step-head">
        <span className="step-badge" aria-hidden>
          {state === "done" ? "✓" : index}
        </span>
        <h2 className="step-title">{title}</h2>
      </div>
      <div className="step-body">{children}</div>
    </section>
  );
}

export function StepRail({ children }: { children: ReactNode }) {
  return <div className="step-rail">{children}</div>;
}
