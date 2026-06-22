import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, ThreeEvent, useThree } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Button, Callout, Card, H5, Tooltip } from "@blueprintjs/core";
import { getMeshes, MeshInfo } from "../api";

interface TissueStyle {
  color: string;
  opacity: number;
}

const TISSUE_STYLES: Record<string, TissueStyle> = {
  skin: { color: "#d8b89a", opacity: 0.18 },
  vagus_left: { color: "#37cc6a", opacity: 1.0 },
  vagus_right: { color: "#e0524f", opacity: 1.0 },
};

function styleFor(name: string): TissueStyle {
  return TISSUE_STYLES[name] ?? { color: "#8a9ba8", opacity: 0.6 };
}

interface Marker {
  id: number;
  point: [number, number, number];
}

interface LoadedMeshProps {
  info: MeshInfo;
  onLoad: (geom: THREE.BufferGeometry) => void;
  onClickPoint: (p: THREE.Vector3) => void;
}

function LoadedMesh({ info, onLoad, onClickPoint }: LoadedMeshProps) {
  const [geom, setGeom] = useState<THREE.BufferGeometry | null>(null);
  const style = styleFor(info.name);

  useEffect(() => {
    const loader = new STLLoader();
    let cancelled = false;
    loader.load(info.url, (g) => {
      if (cancelled) return;
      g.computeVertexNormals();
      g.computeBoundingBox();
      setGeom(g);
      onLoad(g);
    });
    return () => {
      cancelled = true;
    };
  }, [info.url, info.name, onLoad]);

  if (!geom) return null;

  return (
    <mesh
      geometry={geom}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        onClickPoint(e.point.clone());
      }}
    >
      <meshStandardMaterial
        color={style.color}
        transparent={style.opacity < 1}
        opacity={style.opacity}
        roughness={0.6}
        metalness={0.05}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// Fits the camera to the union of loaded bounding boxes once.
function CameraFit({ box }: { box: THREE.Box3 | null }) {
  const { camera } = useThree();
  const fitted = useRef(false);
  useEffect(() => {
    if (!box || fitted.current || box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 1.6 || 100;
    camera.position.set(center.x + radius, center.y + radius, center.z + radius);
    camera.near = radius / 1000;
    camera.far = radius * 100;
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    fitted.current = true;
  }, [box, camera]);
  return null;
}

export default function Viewer3D() {
  const [meshes, setMeshes] = useState<MeshInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [box, setBox] = useState<THREE.Box3 | null>(null);
  const [markers, setMarkers] = useState<Marker[]>([]);
  const nextId = useRef(0);

  useEffect(() => {
    getMeshes()
      .then((r) => setMeshes(r.meshes))
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const target = useMemo(() => box?.getCenter(new THREE.Vector3()), [box]);

  function handleLoad(geom: THREE.BufferGeometry) {
    if (!geom.boundingBox) return;
    setBox((prev) => {
      const next = prev ? prev.clone() : new THREE.Box3();
      next.union(geom.boundingBox as THREE.Box3);
      return next;
    });
  }

  function handleClickPoint(p: THREE.Vector3) {
    setMarkers((prev) => [
      ...prev,
      { id: nextId.current++, point: [p.x, p.y, p.z] },
    ]);
  }

  return (
    <>
      <Card className="viewer-overlay" elevation={2}>
        <H5>Viewer</H5>
        {error && <Callout intent="danger">{error}</Callout>}
        {!error && meshes.length === 0 && (
          <Callout intent="primary">
            No meshes available yet. Run the geometry stage to generate STLs.
          </Callout>
        )}
        {meshes.length > 0 && (
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            Loaded: {meshes.map((m) => m.name).join(", ")}
          </div>
        )}
        <Tooltip content="Placed coordinates are not yet wired into the pipeline (needs a backend/config extension).">
          <div style={{ fontSize: 12, fontWeight: 600 }}>
            Placed markers: {markers.length}
          </div>
        </Tooltip>
        {markers.length > 0 && (
          <div style={{ fontSize: 11, fontFamily: "monospace", marginTop: 4 }}>
            {markers.map((m) => (
              <div key={m.id}>
                #{m.id}: ({m.point[0].toFixed(1)}, {m.point[1].toFixed(1)},{" "}
                {m.point[2].toFixed(1)})
              </div>
            ))}
          </div>
        )}
        <div style={{ marginTop: 6 }}>
          <Button
            small
            icon="trash"
            disabled={markers.length === 0}
            onClick={() => setMarkers([])}
          >
            Clear markers
          </Button>
        </div>
        <Callout intent="warning" style={{ marginTop: 8, fontSize: 11 }}>
          Click a mesh to drop a source marker. Viewer-only: placed coordinates
          are not yet wired into the pipeline.
        </Callout>
      </Card>

      <Canvas camera={{ position: [200, 200, 200], fov: 45 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[100, 200, 150]} intensity={1.0} />
        <directionalLight position={[-100, -50, -150]} intensity={0.3} />
        <Grid
          infiniteGrid
          cellSize={10}
          sectionSize={50}
          fadeDistance={1200}
          sectionColor="#5c7080"
          cellColor="#394b59"
        />
        <Suspense fallback={null}>
          {meshes.map((m) => (
            <LoadedMesh
              key={m.name}
              info={m}
              onLoad={handleLoad}
              onClickPoint={handleClickPoint}
            />
          ))}
        </Suspense>
        {markers.map((m) => (
          <mesh key={m.id} position={m.point}>
            <sphereGeometry args={[2.5, 16, 16]} />
            <meshStandardMaterial color="#fdb515" emissive="#7a5400" />
          </mesh>
        ))}
        <CameraFit box={box} />
        <OrbitControls target={target} makeDefault />
      </Canvas>
    </>
  );
}
