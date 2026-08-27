// How a session can start.
//
// Two honest options, plus resuming what you left. The vagus preset is not a
// demo fixture: its sources are sampled from the cervical vagus in this
// project's own FEM mesh, so they sit inside the model and solve rather than
// landing outside it and being rejected — the single most common way a first
// click fails.
//
// No preset pins a figure. A preset chooses a target and a modality; deciding
// what to draw is the planner's, and pinning two PNGs on their behalf both
// clutters the canvas and quietly adds a render stage to their first run. Every
// figure is one click away in "Add to the journey", which is where a decision
// about output belongs.

import type { Session } from "./storage";

export type PresetId = "vagus" | "spine" | "blank";

export interface Preset {
  id: PresetId;
  title: string;
  blurb: string;
  /** The one line that says what you get, on the card. */
  detail: string;
  icon: "anatomy" | "mesh" | "bolt";
  session: Omit<Session, "visible">;
}

export const PRESETS: Preset[] = [
  {
    id: "vagus",
    title: "Cervical vagus",
    blurb:
      "The whole left vagus sampled at the configured dipole spacing, with " +
      "the OPM array wrapped around the torso.",
    detail: "full vagus · MEG · ready to run",
    icon: "anatomy",
    session: {
      // Empty on purpose: an empty list is what makes the solve *sample* the
      // source tissue at forward.source_spacing_mm. The three hand-picked
      // points that used to live here silently overrode that, so the headline
      // preset modelled 3 dipoles where the CLI modelled ~88.
      sources: [],
      threshold: 3,
      target: "vagus_left",
      outputs: [],
      positions: {},
      extras: [],
      modality: "meg",
      sourcesLevel: null,
    },
  },
  {
    id: "spine",
    title: "Cervical spine at C7",
    blurb:
      "The spinal cord within the C7 vertebral band, sampled at the " +
      "configured dipole spacing, with the electrode patch centred there.",
    detail: "C7 · 5 mm sampling · EEG patch",
    icon: "mesh",
    session: {
      // Empty for the same reason as the vagus preset: sampling is the base
      // operation. sourcesLevel below clips it to C7's own measured Z band, so
      // the cord is sampled at full density *within* that level.
      sources: [],
      threshold: 3,
      target: "spinal_cord",
      outputs: [],
      positions: {},
      extras: [],
      modality: "eeg",
      sourcesLevel: "c7",
    },
  },
  {
    id: "blank",
    title: "Blank journey",
    blurb:
      "The seven core steps and nothing else. Choose your own target, place " +
      "your own sources, add what you need.",
    detail: "no target · nothing pinned",
    icon: "bolt",
    session: {
      sources: [],
      threshold: 3,
      target: null,
      outputs: [],
      positions: {},
      extras: [],
      modality: "meg",
      sourcesLevel: null,
    },
  },
];

export function presetById(id: PresetId): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}
