"use client";

// The icon set, drawn in-house so the interface owns its own line quality:
// one 24-unit grid, 1.6 stroke, round caps, no fills. Everything is
// currentColor, so an icon takes the colour of whatever it sits in.

export type IconName =
  | "anatomy"
  | "mesh"
  | "sensor"
  | "bolt"
  | "solve"
  | "pulse"
  | "figure"
  | "plus"
  | "minus"
  | "fit"
  | "terminal"
  | "compare"
  | "cog"
  | "play"
  | "stop"
  | "check"
  | "close"
  | "arrow-right"
  | "arrow-left"
  | "eye"
  | "eye-off"
  | "search"
  | "alert"
  | "trash"
  | "reset"
  | "cloud"
  | "chevron-down"
  | "chevron-right"
  | "machine"
  | "info"
  | "external";

const PATHS: Record<IconName, string> = {
  // A body volume: the thing being imaged.
  anatomy: "M12 3.2 20 7.6v8.8L12 20.8 4 16.4V7.6z M4 7.6l8 4.4 8-4.4 M12 12v8.8",
  // Tetrahedra: the FEM mesh.
  mesh: "M4.2 5h15.6v14H4.2z M9.4 5v14 M14.6 5v14 M4.2 9.6h15.6 M4.2 14.4h15.6",
  // A sensor on a surface.
  sensor: "M5 15.5a7 7 0 0 1 14 0 M12 15.5v3.7 M8.4 19.2h7.2 M12 5.2V3.4 M17.8 7.4 19 6.2 M6.2 7.4 5 6.2",
  // Current: a dipole arrow.
  bolt: "M13.4 3 6 13.6h4.8L10 21l7.6-10.8H12.8z",
  // Parallel workers chewing through chunks.
  solve: "M4.5 6.5h6v11h-6z M13.5 6.5h6v6h-6z M13.5 15.2h6v2.3h-6z",
  // The readout trace.
  pulse: "M3 12.5h4l2.4-6 3.4 12 2.6-8 1.8 2h4",
  figure: "M4 5.5h16v13H4z M4 15l4.4-4.2 3.6 3.4 3-2.8L20 15.4 M9 9.4a1 1 0 1 0 0-.1z",
  plus: "M12 5.5v13 M5.5 12h13",
  minus: "M5.5 12h13",
  fit: "M4 9V4.6h4.4 M20 9V4.6h-4.4 M4 15v4.4h4.4 M20 15v4.4h-4.4",
  terminal: "M4 5.5h16v13H4z M7.6 10 10 12.4l-2.4 2.4 M12.6 15h4",
  compare: "M12 4v16 M4 8.4h4.6 M4 15.6h4.6 M15.4 8.4H20 M15.4 15.6H20",
  cog: "M12 9.2a2.8 2.8 0 1 0 0 5.6 2.8 2.8 0 0 0 0-5.6z M19.2 12c0-.6-.07-1.2-.2-1.7l1.7-1.2-1.8-3-2 .8a7.3 7.3 0 0 0-2.9-1.7L13.6 3h-3.2l-.4 2.2a7.3 7.3 0 0 0-2.9 1.7l-2-.8-1.8 3 1.7 1.2a7 7 0 0 0 0 3.4L3.3 15l1.8 3 2-.8a7.3 7.3 0 0 0 2.9 1.7l.4 2.1h3.2l.4-2.1a7.3 7.3 0 0 0 2.9-1.7l2 .8 1.8-3-1.7-1.2c.13-.5.2-1.1.2-1.8z",
  play: "M8 5.4 19 12 8 18.6z",
  stop: "M6.5 6.5h11v11h-11z",
  check: "M5 12.8 9.6 17.4 19 6.9",
  close: "M6.2 6.2 17.8 17.8 M17.8 6.2 6.2 17.8",
  "arrow-right": "M4.5 12h14 M13 6.5 18.6 12 13 17.5",
  "arrow-left": "M19.5 12h-14 M11 6.5 5.4 12 11 17.5",
  eye: "M2.8 12S6.6 5.8 12 5.8 21.2 12 21.2 12 17.4 18.2 12 18.2 2.8 12 2.8 12z M12 9.4a2.6 2.6 0 1 0 0 5.2 2.6 2.6 0 0 0 0-5.2z",
  "eye-off": "M4 4.4 20 19.6 M9.6 9.7A2.6 2.6 0 0 0 12 14.6c.7 0 1.3-.26 1.8-.7 M6.4 7.2C4.2 8.8 2.8 12 2.8 12S6.6 18.2 12 18.2c1.5 0 2.8-.4 4-1 M17.3 15.2c2.3-1.6 3.9-3.2 3.9-3.2S17.4 5.8 12 5.8c-.7 0-1.4.1-2 .3",
  search: "M11 4.6a6.4 6.4 0 1 0 0 12.8 6.4 6.4 0 0 0 0-12.8z M15.8 15.8 20 20",
  alert: "M12 4.4 21 19.6H3z M12 10v4.2 M12 16.6v.9",
  trash: "M5 7.2h14 M9.6 7.2V5h4.8v2.2 M7.2 7.2l.9 12h7.8l.9-12 M10.4 10.4v5.6 M13.6 10.4v5.6",
  reset: "M20 12a8 8 0 1 1-2.6-5.9 M20 4v4.6h-4.6",
  cloud: "M7.4 18.4a4 4 0 0 1-.4-8 5.4 5.4 0 0 1 10.3 1.2 3.4 3.4 0 0 1-.6 6.8z",
  "chevron-down": "M6 9.5 12 15.5 18 9.5",
  "chevron-right": "M9.5 6 15.5 12 9.5 18",
  machine: "M3.4 5.5h17.2v10H3.4z M8.4 19.4h7.2 M12 15.5v3.9 M8 9.2h8 M8 12h5",
  info: "M12 4.4a7.6 7.6 0 1 0 0 15.2 7.6 7.6 0 0 0 0-15.2z M12 11v5.2 M12 8.2v.9",
  external: "M14.4 4.6H19.4v5 M19.4 4.6 11.6 12.4 M17 14.4v4.2a.8.8 0 0 1-.8.8H5.4a.8.8 0 0 1-.8-.8V7.8a.8.8 0 0 1 .8-.8h4.2",
};

export default function Icon({
  name,
  size = 16,
  className,
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
