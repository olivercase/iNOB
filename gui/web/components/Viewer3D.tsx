"use client";

import { useEffect, useMemo, useState } from "react";
import { Canvas, ThreeEvent, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { MeshInfo, PointSource } from "@/lib/api";

const TISSUE_COLOR: Record<string, string> = {
  skin: "#d9a06b",
  vagus_left: "#ffd23f",
  vagus_right: "#ffa23f",
};

interface LoadedMesh {
  name: string;
  geometry: THREE.BufferGeometry;
}

function useStlMeshes(meshes: MeshInfo[]): LoadedMesh[] {
  const [loaded, setLoaded] = useState<LoadedMesh[]>([]);
  useEffect(() => {
    let cancelled = false;
    const loader = new STLLoader();
    Promise.all(
      meshes.map(async (m) => {
        const buf = await fetch(m.url).then((r) => r.arrayBuffer());
        const geometry = loader.parse(buf);
        geometry.computeVertexNormals();
        return { name: m.name, geometry } as LoadedMesh;
      }),
    )
      .then((res) => !cancelled && setLoaded(res))
      .catch(() => !cancelled && setLoaded([]));
    return () => {
      cancelled = true;
    };
  }, [meshes]);
  return loaded;
}

function FitCamera({ meshes }: { meshes: LoadedMesh[] }) {
  const { camera } = useThree();
  useEffect(() => {
    if (!meshes.length) return;
    const box = new THREE.Box3();
    for (const m of meshes) {
      m.geometry.computeBoundingBox();
      if (m.geometry.boundingBox) box.union(m.geometry.boundingBox);
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 200;
    camera.position.set(center.x + size * 0.6, center.y + size * 0.2, center.z + size * 0.6);
    camera.near = size / 1000;
    camera.far = size * 10;
    camera.lookAt(center);
    camera.updateProjectionMatrix();
  }, [meshes, camera]);
  return null;
}

interface Props {
  meshes: MeshInfo[];
  visible: Record<string, boolean>;
  sources: PointSource[];
  onAddSource: (p: { x: number; y: number; z: number }) => void;
}

export default function Viewer3D({ meshes, visible, sources, onAddSource }: Props) {
  const loaded = useStlMeshes(meshes);
  const [placing, setPlacing] = useState(false);
  const shown = useMemo(
    () => loaded.filter((m) => visible[m.name] !== false),
    [loaded, visible],
  );

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (!placing) return;
    e.stopPropagation();
    const p = e.point;
    onAddSource({ x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2) });
  };

  return (
    <div className="viewerWrap">
      <Canvas camera={{ fov: 45, position: [200, 80, 200] }} dpr={[1, 2]}>
        <color attach="background" args={["#0b0e14"]} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[1, 1, 1]} intensity={1.1} />
        <directionalLight position={[-1, -0.5, -1]} intensity={0.4} />
        <FitCamera meshes={shown} />
        {shown.map((m) => (
          <mesh key={m.name} geometry={m.geometry} onClick={handleClick}>
            <meshStandardMaterial
              color={TISSUE_COLOR[m.name] ?? "#7aa2ff"}
              transparent
              opacity={m.name === "skin" ? 0.28 : 0.9}
              roughness={0.6}
              metalness={0.0}
              side={THREE.DoubleSide}
            />
          </mesh>
        ))}
        {sources.map((s, i) => (
          <mesh key={i} position={[s.x, s.y, s.z]}>
            <sphereGeometry args={[3, 16, 16]} />
            <meshStandardMaterial color="#5ad1c4" emissive="#2a6f66" />
          </mesh>
        ))}
        <OrbitControls makeDefault enableDamping />
      </Canvas>
      <div style={{ position: "absolute", top: 10, left: 10, display: "flex", gap: 8 }}>
        <button
          className={placing ? "btn" : "ghost"}
          onClick={() => setPlacing((p) => !p)}
        >
          {placing ? "● Click a mesh to drop a source" : "Place source"}
        </button>
        {!loaded.length && <span className="pill">loading meshes…</span>}
      </div>
    </div>
  );
}
