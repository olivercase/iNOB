// The journey graph: what the pipeline is, expressed as nodes a planner walks
// through rather than as five opaque CLI stages.
//
// Positions are authored in canvas units (the canvas pans/zooms around them)
// so the layout reads left-to-right as the physics does: anatomy becomes a
// mesh, sensors are placed over it, a source is dropped in, the leadfield is
// solved, and the answer falls out of the right-hand end. Figure nodes hang
// below the stage that draws them.

import type { IconName } from "@/components/ui/Icon";

export type NodeKind =
  | "anatomy"
  | "conductivity"
  | "mesh"
  | "sensors"
  | "sources"
  | "solve"
  | "detect"
  | "figure"
  // Optional nodes the user adds: a second sensing modality, and the two
  // analytic forward models the FEM solve is checked against.
  | "eeg"
  | "biot"
  | "sarvas"
  | "noise"
  | "anisotropy"
  | "engine"
  | "cluster";

// What a node is doing right now. Drives every visual cue on the canvas.
export type NodeState =
  | "idle" // nothing to show yet
  | "ready" // has what it needs
  | "active" // running this instant
  | "done"
  | "skipped"
  | "failed";

export interface JourneyNode {
  id: string;
  kind: NodeKind;
  title: string;
  /** The one line under the title on the card. */
  caption: string;
  /**
   * What this step actually does, shown on hover before the user commits to
   * opening it. The caption has to fit a card; this does not, so it is where
   * the physics gets explained.
   */
  hint?: string;
  /** Pipeline stage this node reports the state of, if any. */
  stage?: string;
  /** Icon name for the card glyph (see components/ui/Icon). */
  icon: IconName;
  x: number;
  y: number;
  /** Figure nodes only: the /api/figures key they render. */
  figureKey?: string;
}

export interface JourneyEdge {
  from: string;
  to: string;
  /** Small label drawn on the wire — what actually travels along it. */
  label?: string;
}

export const NODE_W = 208;
export const NODE_H = 96;

const COL = 300; // horizontal pitch between stages
const ROW = 200; // vertical pitch between a stage and its figures

// The spine of the journey. Every node here always exists; figure nodes are
// added by the user from the "What happens next?" panel.
export const BASE_NODES: JourneyNode[] = [
  {
    id: "anatomy",
    kind: "anatomy",
    title: "Anatomy",
    caption: "Which tissues to model",
    stage: "geom",
    icon: "anatomy",
    x: 0,
    y: 0,
  },
  {
    id: "conductivity",
    kind: "conductivity",
    title: "Conductivities",
    caption: "S/m per tissue",
    icon: "bolt",
    x: COL,
    y: 0,
  },
  {
    id: "mesh",
    kind: "mesh",
    title: "FEM meshing",
    caption: "Element size",
    stage: "fem",
    icon: "mesh",
    x: COL * 2,
    y: 0,
  },
  {
    id: "sources",
    kind: "sources",
    title: "Sources",
    caption: "Where the nerve fires",
    icon: "bolt",
    x: COL * 3,
    y: -ROW,
  },
  {
    id: "sensors",
    kind: "sensors",
    title: "Sensor array",
    caption: "OPM placement",
    stage: "sensors",
    icon: "sensor",
    x: COL * 3,
    y: 0,
  },
  {
    id: "solve",
    kind: "solve",
    title: "Forward solve",
    caption: "DUNEuro leadfield",
    stage: "forward",
    icon: "solve",
    x: COL * 4,
    y: 0,
  },
  {
    id: "detect",
    kind: "detect",
    title: "Trials to detect",
    caption: "The answer",
    icon: "pulse",
    x: COL * 5,
    y: 0,
  },
];

export const BASE_EDGES: JourneyEdge[] = [
  { from: "anatomy", to: "conductivity", label: "surfaces" },
  { from: "conductivity", to: "mesh", label: "sigma" },
  { from: "mesh", to: "sensors", label: "volume" },
  { from: "sensors", to: "solve", label: "sensors" },
  { from: "sources", to: "solve", label: "dipoles" },
  { from: "solve", to: "detect", label: "leadfield" },
];

// Which spine node a figure hangs off, by the stage that draws it.
const FIGURE_PARENT: Record<string, string> = {
  geom: "anatomy",
  fem: "mesh",
  sensors: "sensors",
  forward: "solve",
  viz: "detect",
};

export function figureParent(stage: string): string {
  return FIGURE_PARENT[stage] ?? "detect";
}

/** Place a figure node under its parent, stacking further down for each sibling. */
export function figurePosition(
  parent: JourneyNode,
  siblingIndex: number,
): { x: number; y: number } {
  return {
    x: parent.x + (siblingIndex % 2 === 0 ? -24 : 24),
    y: parent.y + ROW + Math.floor(siblingIndex / 2) * (NODE_H + 64),
  };
}

/** Centre of a node, in canvas units — where wires attach. */
export function nodeCentre(n: JourneyNode): { x: number; y: number } {
  return { x: n.x + NODE_W / 2, y: n.y + NODE_H / 2 };
}

/**
 * A cubic bezier between two nodes. Wires leave the right edge and arrive at
 * the left edge when the nodes are side by side, and leave the bottom/top when
 * they are stacked, so a wire never cuts back across its own card.
 */
export function wirePath(a: JourneyNode, b: JourneyNode): string {
  const ac = nodeCentre(a);
  const bc = nodeCentre(b);
  const horizontal = Math.abs(bc.x - ac.x) >= Math.abs(bc.y - ac.y);

  if (horizontal) {
    const x1 = ac.x + (bc.x > ac.x ? NODE_W / 2 : -NODE_W / 2);
    const x2 = bc.x + (bc.x > ac.x ? -NODE_W / 2 : NODE_W / 2);
    const k = Math.max(40, Math.abs(x2 - x1) * 0.5);
    const dir = bc.x > ac.x ? 1 : -1;
    return `M ${x1} ${ac.y} C ${x1 + k * dir} ${ac.y}, ${x2 - k * dir} ${bc.y}, ${x2} ${bc.y}`;
  }

  const y1 = ac.y + (bc.y > ac.y ? NODE_H / 2 : -NODE_H / 2);
  const y2 = bc.y + (bc.y > ac.y ? -NODE_H / 2 : NODE_H / 2);
  const k = Math.max(40, Math.abs(y2 - y1) * 0.55);
  const dir = bc.y > ac.y ? 1 : -1;
  return `M ${ac.x} ${y1} C ${ac.x} ${y1 + k * dir}, ${bc.x} ${y2 - k * dir}, ${bc.x} ${y2}`;
}

/* ── the optional nodes ───────────────────────────────────────────────────
 *
 * Everything here can be added to the journey and removed again. The five
 * core stages cannot, because a FEM run is made of exactly those.
 */

export interface AddableSpec {
  /** Node id once added. */
  id: string;
  group: "sensing" | "model" | "physics" | "compute";
  kind: NodeKind;
  title: string;
  caption: string;
  /** One line in the add panel: what it is and when you'd want it. */
  blurb: string;
  icon: IconName;
  /** Wires in from here… */
  from: string;
  /** …and, when it feeds the rest of the pipeline, out to here. */
  to?: string;
  x: number;
  y: number;
}

export const ADDABLE: AddableSpec[] = [
  {
    id: "eeg",
    group: "sensing",
    kind: "eeg",
    title: "EEG electrodes",
    caption: "Contact array",
    blurb: "Solve for surface electrodes instead of OPM magnetometers",
    icon: "sensor",
    from: "mesh",
    to: "solve",
    x: COL * 3,
    y: ROW,
  },
  {
    id: "biot",
    group: "model",
    kind: "biot",
    title: "Biot–Savart",
    caption: "Free space",
    blurb: "The primary current alone — no volume conductor at all",
    icon: "pulse",
    from: "sources",
    x: COL * 4,
    y: -ROW,
  },
  {
    id: "sarvas",
    group: "model",
    kind: "sarvas",
    title: "Sarvas sphere",
    caption: "Homogeneous sphere",
    blurb: "Analytic single-sphere model — the FEM sanity check",
    icon: "anatomy",
    from: "sources",
    x: COL * 5,
    y: -ROW,
  },
  {
    id: "noise",
    group: "physics",
    kind: "noise",
    title: "Noise floor",
    caption: "What the sensor hears",
    blurb: "Sensor noise and bandwidth — the denominator of every SNR",
    icon: "pulse",
    from: "sensors",
    to: "detect",
    x: COL * 4,
    y: ROW,
  },
  {
    id: "anisotropy",
    group: "physics",
    kind: "anisotropy",
    title: "Muscle anisotropy",
    caption: "Direction-dependent sigma",
    blurb: "Muscle conducts better along its fibres than across them",
    icon: "mesh",
    from: "conductivity",
    x: COL,
    y: ROW,
  },
  {
    id: "engine",
    group: "compute",
    kind: "engine",
    title: "Solver engine",
    caption: "DUNEuro build",
    blurb: "Point the forward solve at a compiled DUNEuro, and check it loads",
    icon: "cog",
    from: "solve",
    x: COL * 5,
    y: ROW,
  },
  {
    id: "cluster",
    group: "compute",
    kind: "cluster",
    title: "Cluster solve",
    caption: "Myriad / Kathleen",
    blurb: "Send the forward solve to UCL's cluster instead of this machine",
    icon: "cloud",
    from: "solve",
    x: COL * 6,
    y: ROW,
  },
];

// The core spine: every stage a FEM run is made of. These can be moved and
// opened, never removed — deleting one would leave a journey that cannot run.
export const CORE_IDS = new Set(BASE_NODES.map((n) => n.id));

export function isCore(id: string): boolean {
  return CORE_IDS.has(id);
}

export function addableById(id: string): AddableSpec | undefined {
  return ADDABLE.find((a) => a.id === id);
}

/** Bounding box of a node set, for "fit to view". */
export function bounds(nodes: JourneyNode[]) {
  const xs = nodes.map((n) => n.x);
  const ys = nodes.map((n) => n.y);
  return {
    minX: Math.min(...xs),
    minY: Math.min(...ys),
    maxX: Math.max(...xs) + NODE_W,
    maxY: Math.max(...ys) + NODE_H,
  };
}
