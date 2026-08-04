"use client";

// The interface's own control set. Small on purpose: every control here is one
// element with one job, styled from the journey token set in globals.css, so
// the app has a single voice instead of a framework's.

import {
  useEffect,
  useId,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";
import Icon, { type IconName } from "./Icon";

export { default as Icon } from "./Icon";
export type { IconName } from "./Icon";

/* ── button ───────────────────────────────────────────────────────────────── */

export function Button({
  children,
  onClick,
  icon,
  iconRight,
  variant = "default",
  size = "md",
  fill,
  disabled,
  loading,
  title,
  type = "button",
  className = "",
  ariaLabel,
}: {
  children?: ReactNode;
  onClick?: () => void;
  icon?: IconName;
  iconRight?: IconName;
  variant?: "default" | "primary" | "ghost" | "danger";
  size?: "sm" | "md";
  fill?: boolean;
  disabled?: boolean;
  loading?: boolean;
  title?: string;
  type?: "button" | "submit";
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <button
      type={type}
      className={`ui-btn ui-btn--${variant} ui-btn--${size}${fill ? " ui-btn--fill" : ""} ${className}`}
      onClick={onClick}
      disabled={disabled || loading}
      title={title}
      aria-label={ariaLabel}
    >
      {loading ? <Spinner size={13} /> : icon && <Icon name={icon} size={size === "sm" ? 13 : 15} />}
      {children && <span>{children}</span>}
      {iconRight && <Icon name={iconRight} size={size === "sm" ? 13 : 15} />}
    </button>
  );
}

/* ── icon-only button ─────────────────────────────────────────────────────── */

export function IconButton({
  name,
  label,
  onClick,
  active,
  size = 16,
  className = "",
}: {
  name: IconName;
  label: string;
  onClick?: () => void;
  active?: boolean;
  size?: number;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={`ui-iconbtn${active ? " ui-iconbtn--on" : ""} ${className}`}
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
    >
      <Icon name={name} size={size} />
    </button>
  );
}

/* ── select ───────────────────────────────────────────────────────────────── */

export function Select({
  value,
  onChange,
  options,
  children,
  disabled,
  id,
  ariaLabel,
}: {
  value: string | number;
  onChange: (value: string) => void;
  options?: { value: string | number; label: string }[];
  children?: ReactNode;
  disabled?: boolean;
  id?: string;
  ariaLabel?: string;
}) {
  return (
    <span className="ui-select">
      <select
        id={id}
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(e.currentTarget.value)}
      >
        {options
          ? options.map((o) => (
              <option key={String(o.value)} value={o.value}>
                {o.label}
              </option>
            ))
          : children}
      </select>
      <Icon name="chevron-down" size={13} />
    </span>
  );
}

/* ── text input ───────────────────────────────────────────────────────────── */

export function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  mono,
  ariaLabel,
  title,
  onBlur,
  onKeyDown,
  width,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  mono?: boolean;
  ariaLabel?: string;
  title?: string;
  onBlur?: () => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  width?: number;
}) {
  return (
    <input
      className={`ui-input${mono ? " mono" : ""}`}
      style={width ? { width } : undefined}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      aria-label={ariaLabel}
      title={title}
      onChange={(e) => onChange(e.currentTarget.value)}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
    />
  );
}

/* ── field wrapper ────────────────────────────────────────────────────────── */

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="ui-field">
      <span className="ui-field-label">{label}</span>
      {children}
      {hint && <span className="ui-field-hint">{hint}</span>}
    </label>
  );
}

/* ── note (was: callout) ──────────────────────────────────────────────────── */

export function Note({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "ok" | "warn" | "danger";
  title?: string;
  children?: ReactNode;
}) {
  const icon: IconName = tone === "danger" || tone === "warn" ? "alert" : tone === "ok" ? "check" : "info";
  return (
    <div className={`ui-note ui-note--${tone}`}>
      <span className="ui-note-icon">
        <Icon name={icon} size={14} />
      </span>
      <div className="ui-note-body">
        {title && <strong>{title}</strong>}
        {children}
      </div>
    </div>
  );
}

/* ── tag ──────────────────────────────────────────────────────────────────── */

export function Tag({
  children,
  tone = "neutral",
  icon,
}: {
  children: ReactNode;
  tone?: "neutral" | "signal" | "danger";
  icon?: IconName;
}) {
  return (
    <span className={`ui-tag ui-tag--${tone}`}>
      {icon && <Icon name={icon} size={12} />}
      {children}
    </span>
  );
}

/* ── spinner ──────────────────────────────────────────────────────────────── */

export function Spinner({ size = 16 }: { size?: number }) {
  return (
    <span
      className="ui-spinner"
      style={{ width: size, height: size }}
      role="progressbar"
      aria-label="working"
    />
  );
}

/* ── switch ───────────────────────────────────────────────────────────────── */

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="ui-toggle">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
      <span className="ui-toggle-track" aria-hidden>
        <span className="ui-toggle-knob" />
      </span>
      <span className="ui-toggle-label">{label}</span>
    </label>
  );
}

/* ── checkbox ─────────────────────────────────────────────────────────────── */

export function Checkbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
}) {
  return (
    <label className="ui-check">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
      <span className="ui-check-box" aria-hidden>
        <Icon name="check" size={11} />
      </span>
      <span>{label}</span>
    </label>
  );
}

/* ── slider ───────────────────────────────────────────────────────────────── */

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  ariaLabel,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
  ariaLabel?: string;
}) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <input
      type="range"
      className="ui-slider"
      style={{ ["--pct" as string]: `${pct}%` }}
      value={value}
      min={min}
      max={max}
      step={step}
      aria-label={ariaLabel}
      onChange={(e) => onChange(Number(e.currentTarget.value))}
    />
  );
}

/* ── disclosure ───────────────────────────────────────────────────────────── */

export function Disclosure({
  title,
  blurb,
  icon,
  summary,
  defaultOpen = false,
  children,
}: {
  title: string;
  blurb?: string;
  icon?: IconName;
  summary?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const panelId = useId();
  return (
    <section className={`ui-disc${open ? " ui-disc--open" : ""}`}>
      <button
        type="button"
        className="ui-disc-head"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon name={open ? "chevron-down" : "chevron-right"} size={13} />
        {icon && <span className="ui-disc-icon"><Icon name={icon} size={14} /></span>}
        <span className="ui-disc-title">{title}</span>
        {summary && <span className="ui-disc-summary">{summary}</span>}
      </button>
      {open && (
        <div className="ui-disc-body" id={panelId}>
          {blurb && <p className="ui-disc-blurb">{blurb}</p>}
          {children}
        </div>
      )}
    </section>
  );
}

/* ── sheet (right-hand drawer) ────────────────────────────────────────────── */

export function Sheet({
  open,
  title,
  icon,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  icon?: IconName;
  onClose: () => void;
  children: ReactNode;
}) {
  const panel = useRef<HTMLDivElement>(null);

  // Escape closes, and focus moves into the sheet when it opens so keyboard
  // users are not left behind on the canvas.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    panel.current?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="ui-sheet-scrim" onClick={onClose}>
      <div
        className="ui-sheet"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        ref={panel}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="ui-sheet-head">
          {icon && <span className="ui-sheet-icon"><Icon name={icon} size={14} /></span>}
          <h2>{title}</h2>
          <IconButton name="close" label="Close" onClick={onClose} size={13} />
        </header>
        <div className="ui-sheet-body">{children}</div>
      </div>
    </div>
  );
}
