# BodyParts3D 4.3 mesh acquisition — handoff (continue tomorrow)

_Last updated: 2026-05-25 (late night), by Claude Code session._

## TL;DR — where we are

- **The 4.3 download API is CRACKED and proven working.** We can download genuine
  version-4.3 OBJ meshes directly from the Anatomography viewer backend
  (`lifesciencedb.jp/bp3d`). They are **full-resolution** (≈300 KB/part), which is
  **better** than the only public bulk archive (`dbarchive` only has the
  **4.0**, 99%-polygon-reduced set).
- **Proof preserved in the repo:** `data/bodyparts3d/raw_4.3/_poc_BP22970_4.3.zip`
  — a real download for part `BP22970`, containing 3 OBJ files dated 2014-03-27
  (the `4.3.1403…` build), e.g. `FJ4039_BP23210_FMA50881_Right trochlear nerve.obj`.
  Verified loadable in `trimesh` (1405 verts / 2758 faces for one part).
- **What's NOT done yet:** (1) enumerate the *complete* list of 4.3 part/element
  ids, (2) bulk-download them all, (3) convert/verify OBJ, (4) categorize into
  `internal_meshes/<system>/` by anatomical system. See "Resume checklist".

Why this matters: these are the same meshes the existing `data/{bone,torso,vagus}`
STLs derive from (identical `FJ…_BP…_FMA…_<name>` naming), so once downloaded +
converted to STL they drop straight into the FEM pipeline as new tissues
(muscle, vessel, organ, fat, …).

---

## The 4.3 download API (verified working)

All requests go to `https://lifesciencedb.jp/bp3d/`. Use a browser-like User-Agent
and a Referer of the viewer; a session cookie helps (saved at
`data/bodyparts3d/raw_4.3/_session_cookies.txt`, but a fresh one works too).

Identifier scheme: `BP…` = "representation"/concept part id; `FJ…` = the actual
**element file** id (one OBJ per FJ); `FMA…` = Foundational Model of Anatomy concept id.

### Step 0 — get a session cookie (optional but polite)
```bash
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
REF='https://lifesciencedb.jp/bp3d/?lng=en'
curl -A "$UA" -c cookies.txt -s -o /dev/null "$REF"
```

### Step 1 — resolve a BP concept id → element file ids (`FJ…`)
POST `download-pallet-art_file.cgi`. **`rep_ids` is a JSON array of OBJECTS**
(not strings — sending strings yields `"Can't use string as a HASH ref"`).
```bash
curl -A "$UA" -e "$REF" -b cookies.txt -s \
  --data-urlencode 'rep_ids=[{"rep_id":"BP22970","opacity":1,"exclude":false}]' \
  "https://lifesciencedb.jp/bp3d/download-pallet-art_file.cgi"
# → {"art_ids":["FJ4039","FJ1381","FJ3981"],"rep_ids":["BP23034","BP23210"],"success":true}
```

### Step 2 — download the OBJ zip
POST `download.cgi` with the resolved ids. Returns `application/zip`
(`Content-Disposition: filename=<filename>.zip`) containing the OBJ files.
```bash
curl -A "$UA" -e "$REF" -b cookies.txt -s -o out.zip \
  --data-urlencode 'rep_id=["BP23034","BP23210"]' \
  --data-urlencode 'ids=["FJ4039","FJ1381","FJ3981"]' \
  --data-urlencode 'filename=bp22970' \
  --data-urlencode 'type=art_file' \
  --data-urlencode 'all_downloads=1' \
  "https://lifesciencedb.jp/bp3d/download.cgi"
```
You can batch many ids in one call (`ids` / `rep_id` arrays can be long). Be polite:
sequential calls, small delay, retries. License: **CC-BY-SA 2.1 Japan**, ©
Database Center for Life Science (DBCLS) — attribution required.

### Confirming version
`GET get-version.cgi?lng=en` returns all renderer data-versions as JSON records
(`tgi_objects_set`, `tgi_renderer_version`, `tgi_tree_version`). The 4.3 build is
`4.3.1403311232` (objects set "4.3", tree FMA3.0); the downloaded OBJs' 2014-03-27
timestamps confirm we're pulling 4.3.

---

## THE OPEN PROBLEM: enumerate ALL 4.3 ids

We can download any part by id, but we still need the **full id list** for 4.3.
Things tried tonight (without the right params they return empty/need a category):
- `get-convert-id-list.cgi` → "There is not Category information" (needs params)
- `get-pallet-element.cgi` → empty (needs a tree/category arg)
- `get-tree_type.cgi` → empty

### Candidate routes (try these tomorrow, in order)
1. **Walk the IS-A / PART-OF tree from the root** via the tree CGIs the viewer uses:
   `get-isa.cgi`, `get-partof.cgi`, `get-tree.cgi`, `get-haspart.cgi`,
   `get-hasmember.cgi`, `get-conventional_root.cgi` / `get-conventional_child.cgi`.
   Inspect how `/tmp/ag.js` (the viewer JS, 2.4 MB) calls them — grep for each
   `.cgi` name to get exact params. Collect every `BP…` node, then run Step 1 on
   all of them to gather the full `FJ…` set, dedupe, then Step 2 in batches.
   - Re-fetch the JS if `/tmp` was wiped:
     `curl -s 'https://lifesciencedb.jp/bp3d/ag_js/ag-all-en.js?240423220011' -o ag.js`
   - Full list of available CGIs is in that JS (already enumerated): includes
     `get-tree.cgi`, `get-isa.cgi`, `get-partof.cgi`, `get-table.cgi`,
     `download-pallet-art_file.cgi`, `download.cgi`, `get-version.cgi`, etc.
2. **Seed from the 4.0 parts list** (small, fast, authoritative id↔FMA↔name map):
   `https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_parts_list_e.txt`
   (128 KB) and `isa_element_parts.txt` (1.1 MB) + the `partof_*` equivalents.
   These give a near-complete element-id list; feed those ids through Step 1/2 to
   pull the **4.3** geometry for each. (4.3 has some extra parts vs 4.0, so prefer
   route 1 for completeness, but route 2 is a reliable ≥95% fallback.)
3. **`get-incremental-search-*.cgi`** can enumerate by term if the tree walk stalls.

---

## Categorization plan (once downloaded)

Bucket every OBJ into `internal_meshes/<system>/` using the FMA IS-A hierarchy +
name heuristics:
- Buckets: `organ/`, `muscle/`, `blood_vessel/` (artery+vein), `bone/`,
  `fat/` (adipose), `nerve/`, `cartilage/`, `ligament_tendon/`, `skin/`,
  `gland/`, `lymphatic/`, `other/`.
- Primary signal: walk each FMA id up the IS-A tree
  (`isa_inclusion_relation_list.txt` / `isa_parts_list_e.txt`) to its top-level
  system. Secondary signal: the human-readable name already in the filename
  (`…_Right trochlear nerve.obj` → nerve; `…vertebra` → bone; `…artery`/`…vein`
  → blood_vessel; `…muscle` → muscle; fat/adipose → fat).
- Keep filenames in the existing `FJ…_BP…_FMA…_<name>.obj` form (traceable).
- Optionally also export STL (mm) per part with `trimesh` so they slot into the
  existing `data/`-style FEM inputs.

### Deliverables to produce
- `internal_meshes/<system>/*.obj` (+ optional `.stl`)
- `internal_meshes/MANIFEST.csv` — fma_id, fj_id, bp_id, name, system, path,
  verts, faces, source_url, data_version=4.3
- `internal_meshes/README.md` — provenance (BodyParts3D/Anatomography **4.3**,
  retrieved 2026-05-26, source `lifesciencedb.jp/bp3d` download API, license
  CC-BY-SA 2.1 JP, attribution DBCLS), per-system counts, caveats.

---

## Environment / repo state at handoff

- **caffeinate** was started for 4h to keep the Mac awake — it has likely expired
  by now; restart with `caffeinate -dimsu &` if running an overnight batch.
  ⚠️ Reminder: `caffeinate` does NOT survive a closed lid — leave the lid open or
  `sudo pmset -a disablesleep 1` (undo with `…disablesleep 0`).
- **GUI** (separate workstream, already built & verified): FastAPI backend
  `gui/backend/app.py` + React/Blueprint/three.js frontend in `gui/frontend/`.
  Run: `uvicorn gui.backend.app:app --port 8000` and `cd gui/frontend && npm run dev`
  (→ http://localhost:5173). Dev servers may have been left running in the prior
  session; `pkill -f "uvicorn gui.backend.app"` / stop the vite process to clean up.
- **Do NOT touch** `src/vagus_fm/`, `gui/`, or existing `data/{bone,torso,vagus}/`
  during the mesh download — write only under `data/bodyparts3d/` and `internal_meshes/`.
- Subagents run under a locked-down permission layer that **denies** `curl`/`python3`/
  `WebFetch` — so the bulk download must run in the **main session**, not a spawned agent.

## Resume checklist (tomorrow)
1. `caffeinate -dimsu &` (lid open) if doing a long run.
2. Re-fetch `/tmp/ag.js`; grep it for the tree CGIs to learn their exact params.
3. Build a tree-walk (route 1) or parts-list seed (route 2) to get the full id set.
4. Batch Step-1 → Step-2 downloads into `data/bodyparts3d/raw_4.3/`, resumable
   (skip ids already fetched; log to `data/bodyparts3d/raw_4.3/download.log`).
5. Unzip, dedupe by FJ id, validate each OBJ in trimesh.
6. Categorize into `internal_meshes/<system>/`; write MANIFEST.csv + README.md.
