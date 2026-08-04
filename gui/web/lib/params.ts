// Plain-English descriptions of the config fields the GUI exposes.
//
// The old UI labelled inputs with their raw dotted config path and left the
// user to infer the rest. Every entry here instead says what the number means
// and which way the answer moves when you change it — the help text is the
// point, not decoration.

import type { IconName } from "@/components/ui/Icon";
import { getPath, type Cfg } from "./config";

export interface Param {
  path: string; // dotted path into the config, e.g. "sensors.resolution_mm"
  label: string;
  unit?: string;
  help: string;
  min?: number;
  max?: number;
  step?: number;
}

export interface ParamGroup {
  title: string;
  blurb: string;
  icon?: IconName;
  /** One-line state shown on the collapsed header, so a folded layer still
   *  tells you what it is currently set to. */
  summary?: (cfg: Cfg) => string | undefined;
  params: Param[];
}

const num = (cfg: Cfg, path: string): number | undefined => {
  const v = getPath<number>(cfg, path, NaN);
  return Number.isFinite(v) ? v : undefined;
};

export const PARAM_GROUPS: ParamGroup[] = [
  {
    title: "Sensor array",
    blurb: "Where the OPM sensors sit and how many of them there are.",
    icon: "sensor",
    summary: (c) => {
      const r = num(c, "sensors.resolution_mm");
      const d = num(c, "sensors.depth_mm");
      return r === undefined ? undefined : `${r} mm spacing · ${d ?? "?"} mm stand-off`;
    },
    params: [
      {
        path: "sensors.resolution_mm",
        label: "Sensor spacing",
        unit: "mm",
        help:
          "Distance between neighbouring OPMs. Smaller spacing packs in more " +
          "sensors, which raises SNR and lowers trials-to-detect, but makes " +
          "the forward solve slower.",
        min: 5,
        max: 100,
        step: 1,
      },
      {
        path: "sensors.depth_mm",
        label: "Stand-off from skin",
        unit: "mm",
        help:
          "Gap between the skin surface and the sensing cell. Field falls off " +
          "steeply with distance, so a larger stand-off costs real signal.",
        min: 0,
        max: 60,
        step: 1,
      },
      {
        path: "sensors.angular_margin_deg",
        label: "Angular coverage margin",
        unit: "°",
        help:
          "How far around the body the array wraps beyond the target. Wider " +
          "coverage catches more of the field pattern.",
        min: 0,
        max: 180,
        step: 5,
      },
    ],
  },
  {
    title: "Noise floor",
    blurb:
      "The sensor noise you are competing against. This sets the scale of the " +
      "trials-to-detect answer more than anything else here.",
    icon: "pulse",
    summary: (c) => {
      const n = num(c, "noise.opm_intrinsic_fT_sqrtHz");
      return n === undefined ? undefined : `${n} fT/√Hz`;
    },
    params: [
      {
        path: "noise.opm_intrinsic_fT_sqrtHz",
        label: "OPM noise density",
        unit: "fT/√Hz",
        help:
          "Intrinsic noise of one magnetometer. 7 fT/√Hz matches a QuSpin " +
          "Gen-3 sensor. Halving it halves the trials needed by a factor of four.",
        min: 0.1,
        max: 100,
        step: 0.5,
      },
      {
        path: "noise.bandwidth_hz",
        label: "Recording bandwidth",
        unit: "Hz",
        help:
          "Bandwidth of the measurement. Total noise scales with its square " +
          "root, so a narrower band around your signal helps.",
        min: 1,
        max: 10000,
        step: 10,
      },
      {
        path: "noise.eeg_amplifier_uV_sqrtHz",
        label: "EEG amplifier noise",
        unit: "µV/√Hz",
        help: "Only affects the EEG comparison, not the MEG answer.",
        min: 0.01,
        max: 10,
        step: 0.1,
      },
      {
        path: "noise.eeg_electrode_skin_kohm",
        label: "Electrode–skin impedance",
        unit: "kΩ",
        help:
          "Contact impedance of the surface electrodes. Only affects the EEG " +
          "comparison.",
        min: 0.1,
        max: 200,
        step: 1,
      },
    ],
  },
  {
    title: "Mesh accuracy",
    blurb:
      "Finer meshes are more faithful but cost a lot of solve time. Change " +
      "these only if you are checking that your result is mesh-converged.",
    icon: "anatomy",
    summary: (c) => {
      const p = num(c, "fem.pitch_mm");
      return p === undefined ? undefined : `${p} mm voxels`;
    },
    params: [
      {
        path: "fem.pitch_mm",
        label: "Voxel size",
        unit: "mm",
        help:
          "Resolution the anatomy is sampled at before meshing. Halving it " +
          "roughly multiplies mesh size by eight.",
        min: 0.5,
        max: 10,
        step: 0.5,
      },
      {
        path: "fem.maxvol",
        label: "Max element volume",
        unit: "mm³",
        help:
          "Upper bound on a single tetrahedron. Lower values give a denser, " +
          "more accurate, slower mesh.",
        min: 1,
        max: 200,
        step: 1,
      },
    ],
  },
];

// Conductivities are a free-form map in the config (one entry per tissue), so
// their group is built at render time from whatever tissues are present.
export const CONDUCTIVITY_ROOT = "forward.conductivities_sm";

export const CONDUCTIVITY_BLURB =
  "Electrical conductivity of each tissue, in siemens per metre. These " +
  "come from the literature; the sensitivity analysis tells you how much " +
  "the uncertainty on them matters.";

export function conductivityHelp(tissue: string): string {
  return `Conductivity assumed for ${tissue.replace(/_/g, " ")}.`;
}
