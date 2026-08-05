"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, ThreeEvent, useThree } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, Html, OrbitControls } from "@react-three/drei";
import { Button, IconButton, Tag, Toggle } from "@/components/ui";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { MeshInfo, PointSource } from "@/lib/api";

// Whole-body anatomical palette. No structure is privileged — iNOB images any
// target, so colours are per tissue category and the *selected target* (which
// the user picks) is what gets emphasised, whatever it is.
const TISSUE: Record<string, { color: string; opacity: number }> = {
  skin: { color: "#e8c4a0", opacity: 0.09 },
  bone: { color: "#e4e9f0", opacity: 0.28 },
  muscle: { color: "#c1566a", opacity: 0.4 },
  blood_vessel: { color: "#d0424e", opacity: 0.62 },
  spinal_cord: { color: "#7ad6b0", opacity: 0.7 },
  vagus_left: { color: "#ffcf4d", opacity: 0.9 },
  vagus_right: { color: "#ffb24d", opacity: 0.9 },
};
// Anything the backend serves that we don't have a colour for: stable hash hue.
function tissueStyle(name: string): { color: string; opacity: number } {
  if (TISSUE[name]) return TISSUE[name];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360;
  return { color: `hsl(${h}, 55%, 62%)`, opacity: 0.55 };
}

interface LoadedMesh {
  name: string;
  geometry: THREE.BufferGeometry;
}

// Lazily load only the meshes that are currently visible, caching parsed
// geometries so toggling a tissue back on is instant. Loads are independent:
// a single STL that fails to fetch/parse is skipped, never taking the rest of
// the anatomy down with it (a whole-body payload of ~30 MB across 7 tissues
// must degrade gracefully). `loading` is true while any visible mesh is still
// in flight; `failed` names meshes that could not be loaded at all.
function useStlMeshes(
  meshes: MeshInfo[],
  visible: Record<string, boolean>,
): {
  loaded: LoadedMesh[];
  loading: boolean;
  failed: string[];
  retryFailed: () => void;
} {
  const cache = useRef<Map<string, LoadedMesh>>(new Map());
  const failedRef = useRef<Set<string>>(new Set());
  const [, bump] = useState(0);
  const [loading, setLoading] = useState(false);

  const want = useMemo(
    () => meshes.filter((m) => visible[m.name] !== false),
    [meshes, visible],
  );

  // Free the GPU-side buffers when this viewer goes away. Parsed geometries
  // were cached and never disposed, so every mount leaked its whole-body
  // anatomy (tens of MB of vertex buffers) for the life of the page.
  useEffect(() => {
    const cached = cache.current;
    return () => {
      cached.forEach((m) => m.geometry.dispose());
      cached.clear();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loader = new STLLoader();
    const todo = want.filter(
      (m) => !cache.current.has(m.name) && !failedRef.current.has(m.name),
    );
    if (todo.length === 0) return;
    setLoading(true);
    Promise.allSettled(
      todo.map(async (m) => {
        const buf = await fetch(m.url).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.arrayBuffer();
        });
        const geometry = loader.parse(buf);
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();   // snapping rejects on the box first
        if (cancelled) {
          geometry.dispose();            // nobody will ever render this one
          return;
        }
        cache.current.set(m.name, { name: m.name, geometry });
      }),
    ).then((results) => {
      if (cancelled) return;
      results.forEach((r, i) => {
        if (r.status === "rejected") failedRef.current.add(todo[i].name);
      });
      setLoading(false);
      bump((n) => n + 1);
    });
    return () => {
      cancelled = true;
    };
  }, [want]);

  // Clear the failure set so the loader effect picks those tissues up again.
  // Failures were previously permanent for the session, so a single blip while
  // the backend was starting hid a tissue with no way back short of a reload.
  const retryFailed = useCallback(() => {
    if (failedRef.current.size === 0) return;
    failedRef.current.clear();
    bump((n) => n + 1);
  }, []);

  const loaded = useMemo(
    () => want.map((m) => cache.current.get(m.name)).filter(Boolean) as LoadedMesh[],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [want, loading],
  );
  const failed = want.map((m) => m.name).filter((n) => failedRef.current.has(n));
  return { loaded, loading, failed, retryFailed };
}

// Nearest point on the target anatomy, used to snap a dropped dipole onto the
// chosen structure.
//
// This runs on every pointermove while placing. A brute-force scan of every
// vertex in the target (whole-body meshes are hundreds of thousands of verts,
// times however many tissues are shown) stalls the interaction visibly, so the
// search is bounded two ways:
//
//   * a coarse uniform stride finds the neighbourhood, then a full scan runs
//     only within a small window around it;
//   * the bounding box is checked first, so geometries nowhere near the cursor
//     cost nothing.
//
// The result is identical to the exhaustive scan for any realistic mesh, at a
// fraction of the work. `MAX_EXACT` is the size below which we just do the
// exhaustive thing, because it is already cheap.
const MAX_EXACT = 20_000;
const COARSE_TARGET = 4_000;

function nearestVertex(
  point: THREE.Vector3,
  g: THREE.BufferGeometry,
  best: { d: number; p: THREE.Vector3 | null },
): void {
  const pos = g.getAttribute("position");
  if (!pos) return;

  // Cheap reject: if the box is already further away than the best hit, skip.
  if (!g.boundingBox) g.computeBoundingBox();
  if (g.boundingBox) {
    const dBox = g.boundingBox.distanceToPoint(point);
    if (dBox * dBox > best.d) return;
  }

  const v = new THREE.Vector3();
  const n = pos.count;

  const consider = (i: number) => {
    v.fromBufferAttribute(pos, i);
    const d = v.distanceToSquared(point);
    if (d < best.d) {
      best.d = d;
      best.p = v.clone();
    }
  };

  if (n <= MAX_EXACT) {
    for (let i = 0; i < n; i++) consider(i);
    return;
  }

  // Coarse pass over a strided sample to locate the neighbourhood…
  const stride = Math.max(1, Math.floor(n / COARSE_TARGET));
  let coarseBestI = 0;
  let coarseBestD = Infinity;
  for (let i = 0; i < n; i += stride) {
    v.fromBufferAttribute(pos, i);
    const d = v.distanceToSquared(point);
    if (d < coarseBestD) {
      coarseBestD = d;
      coarseBestI = i;
    }
  }
  // …then an exact pass over a window around it. Vertices in an STL are
  // spatially coherent enough that the true nearest sits within a few strides.
  const half = stride * 8;
  const lo = Math.max(0, coarseBestI - half);
  const hi = Math.min(n - 1, coarseBestI + half);
  for (let i = lo; i <= hi; i++) consider(i);
}

function snapToGeometries(
  point: THREE.Vector3,
  geoms: THREE.BufferGeometry[],
): THREE.Vector3 | null {
  const best = { d: Infinity, p: null as THREE.Vector3 | null };
  for (const g of geoms) nearestVertex(point, g, best);
  return best.p;
}

// Fit the camera to the model exactly once (when meshes first load) and again
// whenever `recenter` changes. Toggling tissue visibility must NOT move the view.
function CameraRig({
  meshes,
  recenter,
  loading,
  controls,
  onFit,
}: {
  meshes: LoadedMesh[];
  recenter: number;
  loading: boolean;
  controls: React.RefObject<{ target: THREE.Vector3; update: () => void } | null>;
  onFit: (size: number) => void;
}) {
  const { camera } = useThree();
  // We lock the view once the anatomy is fully loaded so toggling a tissue
  // doesn't move it. Until then we keep refitting as meshes stream in — fitting
  // to the first mesh that happens to arrive would frame the wrong thing.
  const locked = useRef(false);
  const lastRecenter = useRef(recenter);
  useEffect(() => {
    if (!meshes.length) {
      locked.current = false;
      return;
    }
    const recenterRequested = recenter !== lastRecenter.current;
    if (locked.current && !recenterRequested) return;

    const box = new THREE.Box3();
    for (const m of meshes) {
      m.geometry.computeBoundingBox();
      if (m.geometry.boundingBox) box.union(m.geometry.boundingBox);
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 200;
    camera.position.set(center.x + size * 0.55, center.y + size * 0.18, center.z + size * 0.55);
    camera.near = size / 1000;
    camera.far = size * 10;
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    if (controls.current) {
      controls.current.target.copy(center);
      controls.current.update();
    }
    onFit(size);
    lastRecenter.current = recenter;
    // Only consider the view settled once nothing else is still loading.
    if (!loading) locked.current = true;
  }, [meshes, recenter, loading, camera, controls, onFit]);
  return null;
}

/**
 * The sensor array drawn in the well itself, one point per channel.
 *
 * When `values` is present each point is coloured by what that sensor reads —
 * a topography you can orbit, rather than a flat picture of one. The ramp is a
 * single hue, dark to light, because the quantity is magnitude: there is no
 * midpoint here for a diverging pair to sit on.
 */
function SensorCloud({
  positions,
  values,
  size,
}: {
  positions: [number, number, number][];
  values?: number[];
  size: number;
}) {
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const xyz = new Float32Array(positions.length * 3);
    positions.forEach((p, i) => {
      xyz[i * 3] = p[0];
      xyz[i * 3 + 1] = p[1];
      xyz[i * 3 + 2] = p[2];
    });
    geom.setAttribute("position", new THREE.BufferAttribute(xyz, 3));

    if (values && values.length === positions.length) {
      const top = Math.max(...values) || 1;
      const rgb = new Float32Array(positions.length * 3);
      const low = new THREE.Color("#08301c");   // one hue…
      const high = new THREE.Color("#7dffab");  // …dark to light
      const c = new THREE.Color();
      values.forEach((v, i) => {
        // Field falls off steeply, so a linear ramp puts every sensor at the
        // dark end and shows nothing. The square root keeps the near sensors
        // separated without pretending the far ones read more than they do.
        c.copy(low).lerp(high, Math.sqrt(Math.max(v, 0) / top));
        rgb[i * 3] = c.r;
        rgb[i * 3 + 1] = c.g;
        rgb[i * 3 + 2] = c.b;
      });
      geom.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
    }
    return geom;
  }, [positions, values]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={size}
        sizeAttenuation
        vertexColors={!!values}
        color={values ? undefined : "#8fb3a2"}
        transparent
        opacity={0.95}
      />
    </points>
  );
}

interface Props {
  meshes: MeshInfo[];
  visible: Record<string, boolean>;
  sources: PointSource[];
  selected: number | null;
  target: string | null;
  onSelect: (i: number | null) => void;
  onAddSource: (p: { x: number; y: number; z: number }) => void;
  /**
   * True while the step whose whole job is placing sources is open. Placement
   * is a mode, and the step's instruction is "click the target" — so the mode
   * arms itself rather than making the user find a button the copy never
   * mentioned and clicking the anatomy do nothing until they did.
   */
  armPlacing?: boolean;
  /** Sensor positions to draw in the well, and optionally what each reads. */
  sensorCloud?: {
    positions: [number, number, number][];
    values?: number[];
  } | null;
}

export default function Viewer3D({
  meshes,
  visible,
  sources,
  selected,
  target,
  onSelect,
  onAddSource,
  armPlacing = false,
  sensorCloud = null,
}: Props) {
  const { loaded, loading, failed, retryFailed } = useStlMeshes(meshes, visible);
  const [placing, setPlacing] = useState(false);
  const [snap, setSnap] = useState(true);
  const [recenter, setRecenter] = useState(0);
  const [hover, setHover] = useState<THREE.Vector3 | null>(null);
  const [missed, setMissed] = useState(false);
  const [modelSize, setModelSize] = useState(300);
  const controls = useRef<{ target: THREE.Vector3; update: () => void } | null>(null);

  const shown = useMemo(() => loaded.filter((m) => visible[m.name] !== false), [loaded, visible]);
  // Snap onto the chosen target if one is set; otherwise fall back to whatever
  // is currently shown so placement always lands on visible anatomy.
  const snapGeoms = useMemo(() => {
    if (target) {
      const t = loaded.find((m) => m.name === target);
      if (t) return [t.geometry];
    }
    return shown.map((m) => m.geometry);
  }, [loaded, shown, target]);
  const canSnap = snap && snapGeoms.length > 0;
  const markerR = modelSize * 0.011;

  // Opening the Sources step arms placement; leaving it disarms, so the
  // crosshair never survives into a step that has nothing to place.
  useEffect(() => {
    setPlacing(armPlacing);
    setMissed(false);
  }, [armPlacing]);

  useEffect(() => {
    if (!placing) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setPlacing(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [placing]);

  const resolvePoint = (raw: THREE.Vector3): THREE.Vector3 => {
    if (canSnap) {
      const s = snapToGeometries(raw, snapGeoms);
      if (s) return s;
    }
    return raw;
  };

  const handleMove = (e: ThreeEvent<PointerEvent>) => {
    if (!placing) return;
    setMissed(false);
    setHover(resolvePoint(e.point.clone()));
  };

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (!placing) return;
    e.stopPropagation();
    const p = resolvePoint(e.point.clone());
    onAddSource({ x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2) });
  };

  return (
    <div className="viewerWrap" style={{ cursor: placing ? "crosshair" : "default" }}>
      <Canvas camera={{ fov: 42, position: [200, 80, 200] }} dpr={[1, 2]}>
        <color attach="background" args={["#0a0d13"]} />
        <fog attach="fog" args={["#0a0d13", modelSize * 1.6, modelSize * 4.5]} />
        <hemisphereLight args={["#aeb9ff", "#161a22", 0.7]} />
        <directionalLight position={[1, 1.2, 0.8]} intensity={1.1} />
        <directionalLight position={[-1, -0.4, -1]} intensity={0.35} />

        <CameraRig meshes={loaded} recenter={recenter} loading={loading} controls={controls} onFit={setModelSize} />

        <group onClick={handleClick} onPointerMove={handleMove} onPointerMissed={() => placing && setMissed(true)}>
          {shown.map((m) => {
            const t = tissueStyle(m.name);
            const isTarget = m.name === target;
            const opacity = isTarget ? 1 : t.opacity;
            return (
              <mesh key={m.name} geometry={m.geometry}>
                <meshStandardMaterial
                  color={t.color}
                  emissive={isTarget ? t.color : "#000000"}
                  emissiveIntensity={isTarget ? 0.5 : 0}
                  transparent={opacity < 1}
                  opacity={opacity}
                  depthWrite={opacity > 0.6}
                  roughness={isTarget ? 0.35 : 0.7}
                  metalness={0}
                  side={THREE.DoubleSide}
                />
              </mesh>
            );
          })}
        </group>

        {/* Live placement preview — a ghost marker tracking the cursor. */}
        {placing && hover && (
          <mesh position={hover} renderOrder={10}>
            <sphereGeometry args={[markerR, 20, 20]} />
            <meshBasicMaterial color="#df472a" transparent opacity={0.45} depthTest={false} />
          </mesh>
        )}

        {sources.map((s, i) => {
          const sel = i === selected;
          return (
            <group key={i} position={[s.x, s.y, s.z]}>
              <mesh
                renderOrder={11}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(i);
                }}
              >
                <sphereGeometry args={[markerR * (sel ? 1.45 : 1), 24, 24]} />
                <meshStandardMaterial
                  color={sel ? "#ff7a5c" : "#df472a"}
                  emissive={sel ? "#c0361a" : "#7d2410"}
                  emissiveIntensity={sel ? 1.1 : 0.6}
                  depthTest={false}
                />
              </mesh>
              {sel && (
                <Html center distanceFactor={modelSize * 1.4} zIndexRange={[20, 0]}>
                  <div className="srcLabel">
                    #{i + 1} · {s.strength_nAm} nA·m
                  </div>
                </Html>
              )}
            </group>
          );
        })}

        {sensorCloud && sensorCloud.positions.length > 0 && (
          <SensorCloud
            positions={sensorCloud.positions}
            values={sensorCloud.values}
            size={Math.max(modelSize * 0.006, 1.5)}
          />
        )}

        <OrbitControls ref={controls as never} makeDefault enableDamping />
        <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
          <GizmoViewport axisColors={["#e06666", "#7bd88f", "#6f9bff"]} labelColor="#0a0d13" />
        </GizmoHelper>
      </Canvas>

      {/* Top-left controls, floating over the well. */}
      <div className="viewerHud viewerHud--tl">
        <Button
          icon="bolt"
          size="sm"
          variant={placing ? "primary" : "default"}
          onClick={() => {
            setPlacing((p) => !p);
            setMissed(false);
          }}
        >
          {placing ? "Placing — Esc to stop" : "Place source"}
        </Button>
        {placing && snapGeoms.length > 0 && (
          <Toggle
            checked={snap}
            label={target ? `Snap to ${target.replace(/_/g, " ")}` : "Snap to anatomy"}
            onChange={setSnap}
          />
        )}
        <IconButton
          name="reset"
          label="Recentre the view"
          onClick={() => setRecenter((n) => n + 1)}
        />
      </div>

      {/* Bottom-left status: load state, errors, live coordinate readout. */}
      <div className="viewerHud viewerHud--bl">
        {loading && !loaded.length ? (
          <Tag icon="cloud">
            loading anatomy…
          </Tag>
        ) : loaded.length ? (
          <Tag icon="anatomy">
            {loaded.length} tissue{loaded.length === 1 ? "" : "s"} shown
            {loading ? " · loading…" : ""}
          </Tag>
        ) : (
          <Tag icon="eye-off">
            no tissues visible — turn one on in the panel
          </Tag>
        )}
        {failed.length > 0 && (
          <>
            <Tag tone="danger" icon="alert">
              failed: {failed.join(", ")}
            </Tag>
            <Button size="sm" variant="ghost" icon="reset" onClick={retryFailed}>
              Retry
            </Button>
          </>
        )}
        {placing && hover && (
          <Tag>
            {hover.x.toFixed(1)}, {hover.y.toFixed(1)}, {hover.z.toFixed(1)} mm
            {canSnap ? " · snapped" : ""}
          </Tag>
        )}
        {missed && (
          <Tag tone="danger">
            click landed off-model
          </Tag>
        )}
      </div>
    </div>
  );
}
