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
      "Three sources along the left vagus, the OPM array, and the geometry " +
      "and mesh figures already on the canvas.",
    detail: "3 sources · MEG · ready to run",
    icon: "anatomy",
    session: {
      // Interior points along the nerve, low to high, 70 nA·m each — the
      // strength a compound action potential is usually modelled at.
      sources: [
        { x: 13.23, y: -78.46, z: 1209.6, strength_nAm: 70 },
        { x: 30.09, y: -117.73, z: 1299.19, strength_nAm: 70 },
        { x: 32.83, y: -82.51, z: 1415.04, strength_nAm: 70 },
      ],
      threshold: 3,
      target: "vagus_left",
      outputs: ["geometry_png", "fem_png"],
      positions: {},
      extras: [],
      modality: "meg",
    },
  },
  {
    id: "spine",
    title: "Cervical spine at C7",
    blurb:
      "Three sources in the spinal cord within the C7 vertebral band, with " +
      "the electrode patch centred on that level.",
    detail: "3 sources · C7 · EEG patch",
    icon: "mesh",
    session: {
      // Tet centroids inside spinal_cord, taken from within C7's own Z band
      // (1364.7–1391.8 mm) as measured from the segmented vertebra — so the
      // sources and the patch are describing the same place.
      sources: [
        { x: 6.09, y: -64.0, z: 1369.63, strength_nAm: 70 },
        { x: -3.05, y: -65.73, z: 1376.89, strength_nAm: 70 },
        { x: -1.83, y: -67.85, z: 1384.95, strength_nAm: 70 },
      ],
      threshold: 3,
      target: "spinal_cord",
      outputs: ["geometry_png"],
      positions: {},
      extras: [],
      modality: "eeg",
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
    },
  },
];

export function presetById(id: PresetId): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}
