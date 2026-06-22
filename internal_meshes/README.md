# internal_meshes — BodyParts3D 4.3, sorted by anatomical system

3210 full-resolution OBJ meshes from BodyParts3D / Anatomography **version 4.3**, sorted into per-system subfolders.

**Provenance:** downloaded 2026-05-26 from the Anatomography viewer's internal download API (`lifesciencedb.jp/bp3d`) — the full-res 4.3 set is not available from the public bulk archive (which only ships 4.0, polygon-reduced). See `scripts/download_bodyparts3d_4.3.py`.

**License:** BodyParts3D, © Database Center for Life Science (DBCLS), CC-BY-SA 2.1 Japan. Attribute DBCLS / BodyParts3D on redistribution.

**Filename:** `FJ<file>_BP<rep>_FMA<concept>_<name>.obj` — the FMA id ties each mesh to the Foundational Model of Anatomy concept.

## Counts by system

| System | Meshes |
|--------|-------:|
| blood_vessel | 1479 |
| muscle | 447 |
| organ | 420 |
| bone | 337 |
| nerve | 238 |
| brain | 180 |
| ligament_tendon | 46 |
| cartilage | 37 |
| gland | 18 |
| skin | 5 |
| lymphatic | 3 |
| **total** | **3210** |

## How each mesh was classified

- `isa` (FMA IS-A hierarchy walk to nearest system root): 1
- `name` (name-keyword fallback): 3168
- `other` (unresolved): 0

See `MANIFEST.csv` for the per-mesh fma_id / system / method / vertex+face counts.
