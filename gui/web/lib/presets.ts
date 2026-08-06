// How a session can start.
//
// Two honest options, plus resuming what you left. The vagus preset is not a
// demo fixture: its three sources are tetrahedron centroids taken from the
// cervical vagus in this project's own FEM mesh, so they sit inside the model
// and solve rather than landing outside it and being rejected — the single
// most common way a first click fails.

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
      "The whole left vagus sampled at the configured dipole spacing, the " +
      "OPM array, and the geometry and mesh figures already on the canvas.",
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
      outputs: ["geometry_png", "fem_png"],
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
      outputs: ["geometry_png"],
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
    detail: "no sources · nothing pinned",
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
