"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "@/components/ui/Icon";
import { PRESETS, type PresetId } from "@/lib/presets";
import {
  getConfig,
  getFigures,
  getHealth,
  getMeshes,
  getSolver,
  getSystem,
} from "@/lib/api";

/**
 * The opening sequence: the app checking itself before it says it is ready.
 *
 * Every row is a real request, and what it reports is what came back — the
 * DUNEuro row in particular is the machine-wide search finishing, not a
 * decorative delay. A step only claims to be done when it is, and a failure
 * stays on screen with what it means instead of sliding past.
 */

type Status = "waiting" | "checking" | "ok" | "warn" | "fail";

interface Check {
  key: string;
  label: string;
  /** What it is looking for, shown while the check is in flight. */
  seeking: string;
  run: () => Promise<{ status: Status; detail: string }>;
}

// A step never flashes past: a check that answers instantly still holds the
// screen long enough to be read, which is the difference between a sequence
// and a flicker.
const MIN_STEP_MS = 420;

const CHECKS: Check[] = [
  {
    key: "backend",
    label: "Backend",
    seeking: "connecting to the API",
    run: async () => {
      const h = await getHealth();
      return h
        ? { status: "ok", detail: "online" }
        : { status: "fail", detail: "no answer on :8000" };
    },
  },
  {
    key: "config",
    label: "Configuration",
    seeking: "reading and validating the config",
    run: async () => {
      try {
        const r = await getConfig();
        const sections = Object.keys(r.config).length;
        if (r.errors.length) {
          return { status: "warn", detail: `${r.errors.length} problem(s): ${r.errors[0]}` };
        }
        return { status: "ok", detail: `${sections} sections, valid` };
      } catch {
        return { status: "fail", detail: "could not be read" };
      }
    },
  },
  {
    key: "anatomy",
    label: "Anatomy",
    seeking: "loading tissue surfaces",
    run: async () => {
      const m = await getMeshes();
      return m.length
        ? { status: "ok", detail: `${m.length} tissues` }
        : { status: "warn", detail: "no meshes found" };
    },
  },
  {
    key: "machine",
    label: "This machine",
    seeking: "counting cores",
    run: async () => {
      const s = await getSystem();
      if (!s) return { status: "warn", detail: "could not be read" };
      const perf =
        s.cpu_performance && s.cpu_performance < s.cpu_logical
          ? `, ${s.cpu_performance} performance`
          : "";
      return { status: "ok", detail: `${s.machine}, ${s.cpu_logical} cores${perf}` };
    },
  },
  {
    key: "solver",
    label: "Solver engine",
    seeking: "searching this machine for a DUNEuro build",
    run: async () => {
      const s = await getSolver();
      if (!s) return { status: "warn", detail: "could not be checked" };
      if (!s.found) {
        return {
          status: "warn",
          detail: `no build found in ${s.searched.length} searched location${
            s.searched.length === 1 ? "" : "s"
          }`,
        };
      }
      const where = s.duneuro_path
        ? s.duneuro_path.replace(/^.*\/(?=[^/]*\/[^/]*$)/, "…/")
        : "already importable";
      return { status: "ok", detail: `${s.python_version}, ${where}` };
    },
  },
  {
    key: "figures",
    label: "Outputs",
    seeking: "listing figures this config can produce",
    run: async () => {
      const f = await getFigures();
      const drawn = f.filter((x) => x.exists).length;
      return f.length
        ? { status: "ok", detail: `${f.length} available, ${drawn} already drawn` }
        : { status: "warn", detail: "none defined" };
    },
  },
];

interface Result {
  status: Status;
  detail: string;
}

/** What the user chose to start from once the checks were done. */
export type BootChoice = { kind: "resume" } | { kind: "preset"; id: PresetId };

export default function Boot({
  onDone,
  canResume,
  resumeSummary,
}: {
  onDone: (choice: BootChoice) => void;
  /** True when a previous session is worth offering back. */
  canResume: boolean;
  /** One line describing that session, e.g. "2 sources · vagus left". */
  resumeSummary?: string;
}) {
  const [results, setResults] = useState<Record<string, Result>>({});
  const [current, setCurrent] = useState(0);
  const [finished, setFinished] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const started = useRef(false);

  const dismiss = useCallback(
    (choice: BootChoice) => {
      if (leaving) return;
      setLeaving(true);
      window.setTimeout(() => onDone(choice), 480);
    },
    [leaving, onDone],
  );

  // The default when someone just presses on: pick up where they left off if
  // there is anything to pick up, otherwise start clean.
  const carryOn = useCallback(
    () => dismiss(canResume ? { kind: "resume" } : { kind: "preset", id: "blank" }),
    [dismiss, canResume],
  );

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let live = true;

    (async () => {
      for (let i = 0; i < CHECKS.length; i++) {
        if (!live) return;
        setCurrent(i);
        const check = CHECKS[i];
        const began = Date.now();
        let outcome: Result;
        try {
          outcome = await check.run();
        } catch {
          outcome = { status: "fail", detail: "check could not run" };
        }
        const held = Date.now() - began;
        if (held < MIN_STEP_MS) {
          await new Promise((r) => setTimeout(r, MIN_STEP_MS - held));
        }
        if (!live) return;
        setResults((r) => ({ ...r, [check.key]: outcome }));
      }
      if (!live) return;
      setCurrent(CHECKS.length);
      setFinished(true);
    })();

    return () => {
      live = false;
    };
  }, []);

  const anyBad = Object.values(results).some(
    (r) => r.status === "fail" || r.status === "warn",
  );

  // Esc carries on with the sensible default — nobody should be held by an
  // animation, and the choice below is an offer, not a gate.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") carryOn();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [carryOn]);

  const done = Object.keys(results).length;
  const pct = Math.round((done / CHECKS.length) * 100);

  return (
    <div
      className={`boot${leaving ? " boot--leaving" : ""}`}
      role="status"
      aria-live="polite"
      aria-label="Starting iNOB"
    >
      <div className="boot-scan" aria-hidden />
      <div className="boot-panel">
        <header className="boot-head">
          <span className="boot-mark" aria-hidden>
            <span className="boot-mark-core" />
            <span className="boot-mark-ring" />
            <span className="boot-mark-ring boot-mark-ring--2" />
          </span>
          <div className="boot-title">
            <h1>iNOB</h1>
            <p>
              {finished
                ? anyBad
                  ? "Ready, with notes — choose how to start"
                  : "Ready — choose how to start"
                : (CHECKS[current]?.seeking ?? "starting up")}
            </p>
          </div>
          <span className="boot-pct mono">{pct}%</span>
        </header>

        <ol className="boot-list">
          {CHECKS.map((check, i) => {
            const res = results[check.key];
            const state: Status = res
              ? res.status
              : i === current
                ? "checking"
                : "waiting";
            return (
              <li key={check.key} className={`boot-row boot-row--${state}`}>
                <span className="boot-dot" aria-hidden>
                  {state === "ok" && <Icon name="check" size={11} />}
                  {(state === "warn" || state === "fail") && (
                    <Icon name="alert" size={11} />
                  )}
                  {state === "checking" && <span className="boot-spin" />}
                </span>
                <span className="boot-label">{check.label}</span>
                <span className="boot-detail mono">
                  {res ? res.detail : state === "checking" ? check.seeking : ""}
                </span>
              </li>
            );
          })}
        </ol>

        <div className="boot-track" aria-hidden>
          <span className="boot-fill" style={{ width: `${pct}%` }} />
        </div>

        {finished ? (
          <section className="boot-start">
            <h2 className="boot-start-head">Start from</h2>
            <div className="boot-choices">
              {canResume && (
                <button
                  type="button"
                  className="boot-choice boot-choice--resume"
                  onClick={() => dismiss({ kind: "resume" })}
                >
                  <span className="boot-choice-icon" aria-hidden>
                    <Icon name="reset" size={17} />
                  </span>
                  <span className="boot-choice-text">
                    <span className="boot-choice-title">Where you left off</span>
                    <span className="boot-choice-blurb">
                      {resumeSummary ?? "Your last session, restored."}
                    </span>
                  </span>
                  <span className="boot-choice-go" aria-hidden>
                    <Icon name="arrow-right" size={13} />
                  </span>
                </button>
              )}

              {PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className="boot-choice"
                  onClick={() => dismiss({ kind: "preset", id: preset.id })}
                >
                  <span className="boot-choice-icon" aria-hidden>
                    <Icon name={preset.icon} size={17} />
                  </span>
                  <span className="boot-choice-text">
                    <span className="boot-choice-title">
                      {preset.title}
                      <span className="boot-choice-detail mono">{preset.detail}</span>
                    </span>
                    <span className="boot-choice-blurb">{preset.blurb}</span>
                  </span>
                  <span className="boot-choice-go" aria-hidden>
                    <Icon name="arrow-right" size={13} />
                  </span>
                </button>
              ))}
            </div>
          </section>
        ) : (
          <footer className="boot-foot">
            <button type="button" className="boot-skip" onClick={carryOn}>
              skip
            </button>
          </footer>
        )}
      </div>
    </div>
  );
}
