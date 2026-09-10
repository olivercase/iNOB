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
  /** Numbers are the common case, so "number" is the default. */
  kind?: "number" | "choice" | "toggle";
  /** kind "choice" only: the fixed set of values. */
  choices?: { value: string; label: string }[];
  /** Rendered only when this holds — e.g. St. Venant knobs are meaningless
   *  unless a Venant source model is selected, and showing them anyway would
   *  imply they do something. */
  visibleIf?: (cfg: Cfg) => boolean;
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

const SOURCE_MODEL = "forward.source_model";
const isVenant = (cfg: Cfg): boolean =>
  getPath<string>(cfg, `${SOURCE_MODEL}.type`, "").endsWith("venant");

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
      if (n === undefined) return undefined;
      const bw = num(c, "noise.opm_bandwidth_hz");
      return bw === undefined ? `${n} fT/√Hz` : `${n} fT/√Hz · ${bw} Hz`;
    },
    params: [
      {
        path: "noise.opm_intrinsic_fT_sqrtHz",
        label: "OPM noise density",
        unit: "fT/√Hz",
        help:
          "Intrinsic noise of one magnetometer. 7 fT/√Hz matches a QuSpin " +
          "Gen-3 sensor. Trials-to-detect goes as the square of it, so halving " +
          "the noise cuts the trials needed by a factor of four. It never " +
          "travels alone: read it with the sensor bandwidth below.",
        min: 0.1,
        max: 100,
        step: 0.5,
      },
      {
        path: "noise.opm_bandwidth_hz",
        label: "OPM sensor bandwidth",
        unit: "Hz",
        help:
          "The magnetometer's own 3 dB point — 135 Hz for a QuSpin Gen-3, " +
          "a few kHz for a helium-4 sensor. It bounds the noise you integrate, " +
          "and it also rolls off the signal: a 0.5 ms action potential peaks near " +
          "318 Hz, so a 135 Hz sensor passes under 40% of it. Narrowing this " +
          "always makes detection harder, never easier.",
        min: 10,
        max: 5000,
        step: 5,
      },
      {
        path: "noise.bandwidth_hz",
        label: "Recording bandwidth",
        unit: "Hz",
        help:
          "Bandwidth of the measurement. Total noise scales with its square " +
          "root, so a narrower band around your signal helps — up to the " +
          "sensor bandwidth above, past which there is little left to gain.",
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
  {
    title: "Discretisation",
    blurb:
      "How the FEM represents the potential. Changes the numbers, not the " +
      "geometry — every leadfield here was solved continuous (CG).",
    icon: "mesh",
    summary: (c) => {
      const t = getPath<string>(c, "forward.solver.type", "");
      return t ? t.toUpperCase() : undefined;
    },
    params: [
      {
        path: "forward.solver.type",
        label: "Galerkin method",
        kind: "choice",
        choices: [
          { value: "cg", label: "Continuous (CG)" },
          { value: "dg", label: "Discontinuous (DG)" },
        ],
        help:
          "CG puts one unknown on each node and holds the potential continuous " +
          "across element faces. DG puts them inside the elements and couples " +
          "neighbours by flux, so a conductivity jump stays a jump instead of " +
          "being smeared over the elements either side — the argument for it in " +
          "a mesh with a thin, high-contrast compartment like a nerve inside " +
          "muscle. Costs roughly four times the degrees of freedom. DG only works with " +
          "partial integration here.",
      },
      {
        path: "forward.solver.penalty",
        label: "Interior penalty",
        visibleIf: (c) => getPath<string>(c, "forward.solver.type", "") === "dg",
        help: "How hard DG penalises a jump across a face. Too low loses stability, too high stiffens the system.",
        min: 1,
        step: 1,
      },
      {
        path: "forward.solver.reduction",
        label: "Solver tolerance",
        help: "Residual reduction the iterative solve must reach. Smaller is stricter and slower.",
        min: 0,
      },
    ],
  },
  {
    title: "Source model",
    blurb:
      "How a point dipole — a singularity with no exact representation on a " +
      "mesh — becomes a finite-element load. Affects the volume-current field " +
      "only; the dipole's own field is analytic either way.",
    icon: "solve",
    summary: (c) => {
      const t = getPath<string>(c, `${SOURCE_MODEL}.type`, "");
      return t ? t.replace(/_/g, " ") : undefined;
    },
    params: [
      {
        path: `${SOURCE_MODEL}.type`,
        label: "Model",
        kind: "choice",
        choices: [
          { value: "partial_integration", label: "Partial integration" },
          { value: "venant", label: "St. Venant" },
          { value: "multipolar_venant", label: "St. Venant (multipolar)" },
        ],
        help:
          "Partial integration loads only the containing element's nodes and is " +
          "what every leadfield here was solved with. St. Venant spreads the " +
          "dipole over a patch of neighbouring nodes fitted to its moment, which " +
          "behaves better near a conductivity jump.",
      },
      {
        path: `${SOURCE_MODEL}.restrict`,
        label: "Keep patch inside one tissue",
        kind: "toggle",
        visibleIf: isVenant,
        help:
          "Stops monopoles crossing a conductivity jump. Watch it on thin " +
          "structures: restricting to a nerve two elements across can leave too " +
          "few nodes to fit the moments.",
      },
      {
        path: `${SOURCE_MODEL}.number_of_moments`,
        label: "Moments matched",
        visibleIf: isVenant,
        help: "How many moments of the dipole the patch reproduces. 3 is standard.",
        min: 1,
        max: 6,
        step: 1,
      },
      {
        path: `${SOURCE_MODEL}.reference_length_mm`,
        label: "Reference length",
        unit: "mm",
        visibleIf: isVenant,
        help: "Length scale the moment fit is normalised by.",
        min: 0.1,
        step: 1,
      },
      {
        path: `${SOURCE_MODEL}.weighting_exponent`,
        label: "Weighting exponent",
        visibleIf: isVenant,
        help:
          "How sharply the regularisation penalises nodes far from the dipole. " +
          "Must be below the moment count.",
        min: 0,
        max: 5,
        step: 1,
      },
      {
        path: `${SOURCE_MODEL}.relaxation_factor`,
        label: "Relaxation factor",
        visibleIf: isVenant,
        help:
          "Weight on that penalty. Larger trades moment accuracy for a smoother, " +
          "better-conditioned fit.",
        min: 0,
      },
      {
        path: `${SOURCE_MODEL}.mixed_moments`,
        label: "Mixed moments",
        kind: "toggle",
        visibleIf: isVenant,
        help: "Include cross terms (xy, xz, …) in the moment conditions.",
      },
      {
        path: `${SOURCE_MODEL}.initialization`,
        label: "Patch starts from",
        kind: "choice",
        visibleIf: isVenant,
        choices: [
          { value: "closest_vertex", label: "Elements at the nearest node" },
          { value: "single_element", label: "The containing element only" },
        ],
        help: "Which elements seed the monopole patch before it is grown.",
      },
      {
        path: `${SOURCE_MODEL}.extensions`,
        label: "Patch grown by",
        kind: "choice",
        visibleIf: isVenant,
        choices: [
          { value: "vertex", label: "Elements sharing a node" },
          { value: "intersection", label: "Elements sharing a face" },
          { value: "", label: "Not grown" },
        ],
        help:
          "How far the patch spreads from that seed — this, not a fixed count, " +
          "is what sets how many nodes the dipole lands on.",
      },
      {
        path: `${SOURCE_MODEL}.intorderadd`,
        label: "Extra quadrature order",
        visibleIf: isVenant,
        help: "Integration order added when assembling the patch. Raise only if the fit looks under-integrated.",
        min: 0,
        max: 10,
        step: 1,
      },
    ],
  },
  {
    title: "Muscle fibre anisotropy",
    blurb:
      "Muscle carries current better along its fibres than across them. When " +
      "active, each muscle element gets a conductivity tensor aligned to its " +
      "muscle's fibre axis; every other tissue stays isotropic.",
    icon: "mesh",
    summary: (c) => {
      const mode = getPath<string>(c, "forward.muscle_anisotropy.mode", "");
      const l = num(c, "forward.muscle_anisotropy.sigma_long_sm");
      const t = num(c, "forward.muscle_anisotropy.sigma_trans_sm");
      if (!mode) return undefined;
      const ratio = l && t ? ` · ${(l / t).toFixed(1)}:1` : "";
      return `${mode}${ratio}`;
    },
    params: [
      {
        path: "forward.muscle_anisotropy.mode",
        label: "When to apply it",
        kind: "choice",
        choices: [
          { value: "auto", label: "Automatic — on for muscle sources" },
          { value: "on", label: "Always on" },
          { value: "off", label: "Always off" },
        ],
        help:
          "Automatic switches the tensor on exactly when muscle is a source " +
          "tissue, so vagus and spine runs stay comparable with ones already " +
          "solved. Force it for a like-for-like A/B.",
      },
      {
        path: "forward.muscle_anisotropy.sigma_long_sm",
        label: "Along the fibre",
        unit: "S/m",
        help: "Conductivity parallel to the fibre direction.",
        min: 0,
        step: 0.05,
      },
      {
        path: "forward.muscle_anisotropy.sigma_trans_sm",
        label: "Across the fibre",
        unit: "S/m",
        help:
          "Conductivity perpendicular to it. The ratio of the two is what the " +
          "solver actually feels.",
        min: 0,
        step: 0.05,
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
