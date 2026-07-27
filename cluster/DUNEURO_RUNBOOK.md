# DUNEuro on UCL Myriad — build, run, and multi-region targeting

Compressed field notes from getting the DUNEuro (DUNE 2.10) FEM forward solver
built and running on Myriad, and generalising the pipeline to different source
regions (vagus, spine, both). Companion to [README.md](README.md).

---

## 1. Building DUNEuro on the cluster

One-time build: `SOURCE_TARGET`-agnostic. Run via
`CLUSTER_PROFILE=myriad bash cluster/submit.sh build` (batch, survives
disconnect) or interactively on a compute node with `cluster/build_duneuro.sh`.
Builds under `${INOB_REMOTE_BASE}` (see §4) into `duneuro-src/` + a venv.

Hard-won fixes now baked into `build_duneuro.sh` and the patch:

- **Patch minimalism.** `scripts/patches/duneuro-eigen5-dune210.patch` keeps
  *only* the `geometry_adaption.hh` hunk (consolidates the two `evaluate()`
  overloads into one template for DUNE 2.10). The old whitney_source_model.hh
  hunk was a no-op comment change that only ever caused "patch does not apply" —
  dropped. The runtime-flag `jacobiSvd(flags)` compiles fine on Eigen 3.4.0
  (deprecated, not removed), so no code change is needed there.
- **Resilient apply.** `git reset --hard <pin>` + `clean -fdq` first (idempotent,
  kills stale half-patched trees), then try `git apply`, falling back to
  `patch -p1 --fuzz=3` (git apply has no fuzz and rejects blank-context lines).
- **PythonLibs.** duneuro-py uses the deprecated `find_package(PythonLibs)`,
  which can't find the module/venv Python alone. `build_duneuro.sh` computes the
  venv include dir + `libpython*.so` and passes explicit `-DPython3_*` **and**
  `-DPYTHON_*` hints in `release.opts`.
- **venv vs pip --user.** `-DDUNE_ENABLE_PYTHONBINDINGS=ON` makes each DUNE
  module try `pip install --user`, which pip forbids inside a venv. Fix:
  `-DDUNE_PYTHON_INSTALL_LOCATION=none`.
- **Don't rebuild duneuro-py.** `dunecontrol all` already builds
  `duneuropy.so` under `duneuro-py/build-cmake`. A standalone `cmake` afterward
  fails on `dune-commonConfig.cmake`; instead just `cp` the artefact into the
  venv site-packages.

Success looks like: `import duneuropy` exposes `Dipole3d`, `FieldVector3D`, etc.

---

## 2. Staging from a Mac

`cluster/stage.sh` rsyncs inputs + code to `${INOB_REMOTE_BASE}`.

- **openrsync.** macOS `/usr/bin/rsync` is Apple's openrsync and segfaults
  against the remote (`error: unexpected end of file`, `status 11`). Install GNU
  rsync: `brew install rsync` (lands at `/opt/homebrew/bin`, ahead in PATH).
- **`RSYNC_OLD_ARGS=1`.** rsync ≥3.4 "secluded args" stops the remote shell
  expanding `${HOME}` in the destination; the script exports this to restore it.
- **Nested dirs.** rsync only `mkdir`s one missing level and the remote rsync
  (3.1.2) lacks `--mkpath`, so `stage.sh` pre-creates `code/scripts/patches`,
  `outputs/fem`, `outputs/sensors`, etc.
- **Input placement.** Inputs must land where the config reads them
  (`outputs/fem/fem.mat`, `outputs/sensors/sensor_array.mat`) — NOT a flat
  `inputs/`, or `inob.forward.chunk` can't find them.

SSH quality-of-life: `~/.ssh/config` with `Host myriad …` and a
`ControlMaster auto` / `ControlPersist 10m` block → one password per stage.

---

## 3. Running the forward solve

**Interactive smoke test** (always validate before a full array). On a compute
node, load modules into the *current* shell (source, don't `bash`), then relax
strict flags so the venv activate + unset vars don't kill the session:

```bash
qrsh -pe smp 4 -l mem=8G,h_rt=1:00:00 -now no
cd ~/Scratch/inob/code
export CLUSTER_PROFILE=myriad
export PYTHONPATH=~/Scratch/inob/code/src
source cluster/lib.sh && inob__init && inob__load_modules
set +eu
source ~/Scratch/inob/venv/bin/activate
python -u -m inob.forward.chunk --config ~/Scratch/inob/configs/default.yaml \
    --chunk-id 0 --n-chunks 32 \
    --set forward.source_tissue=spinal_cord \
    --set outputs.forward_chunks_dir=~/Scratch/inob/out_spine
```

Gotchas: `libpython3.11.so.1.0: cannot open shared object file` = python module
not loaded (`inob__load_modules` in the current shell fixes it). `PYTHONPATH:
unbound variable` / session drops = `lib.sh`'s `set -eu` leaked in — `set +eu`.

**Batch** (from the login node, not qrsh — `submit.sh` only queues):

```bash
SOURCE_TARGET=spine CLUSTER_PROFILE=myriad bash cluster/submit.sh array
SOURCE_TARGET=spine CLUSTER_PROFILE=myriad bash cluster/submit.sh reduce
```

`reduce` is submitted with `-hold_jid fwd_<tag>` so it auto-waits (state `hqw`)
for the array, then stitches. Submit both back-to-back. Monitor:
`qstat`; logs at `~/Scratch/inob/logs/fwd_<tag>.*.log`; cancel with `qdel <id>`.

Throughput comes from the **32-task array**, not the 4 cores/task — up to 32×4
cores when nodes are free. Running locally is a non-starter (no ARM/macOS
duneuropy build) and FEM CG solves are bandwidth-bound, so more cores/task give
diminishing returns.

---

## 4. Which base directory (`INOB_REMOTE_BASE`)

`cluster/lib.sh` prefers `INOB_REMOTE_BASE` (from cluster `~/.inob.env`) over the
profile default (`~/Scratch/inob`). A mismatch — Mac stages to `inob`, cluster
`~/.inob.env` pointed the build at `~/Scratch/duneuro` — silently split the two.
Keep both sides on **one** base. Everything (stage, build, array, reduce) then
lives under it.

---

## 5. Targeting different regions (`SOURCE_TARGET`)

`cluster/lib.sh:inob__source_target` maps a target to source tissues + a slug:

| `SOURCE_TARGET` | `TISSUES` (solver) | chunks dir | leadfield |
|---|---|---|---|
| `vagus` (default) | `vagus_left` | `out_vagus/` | `duneuro_leadfield_vagus.npz` |
| `spine` | `spinal_cord` | `out_spine/` | `duneuro_leadfield_spine.npz` |
| `spine_vagus` | `spinal_cord,vagus_left` | `out_spine_vagus/` | `duneuro_leadfield_spine_vagus.npz` |
| `muscle` | `muscle` | `out_muscle/` | `duneuro_leadfield_muscle.npz` |

Add a region by extending that `case` (any mesh tissue label works — the mesh
already carries all compartments; only the *source placement* changes).

**Muscle is special.** The nerve/cord targets are thin midline structures, so
`vagus_sources()` (one dipole per axial Z-slab) is faithful. Muscle is a bulky
*bilateral* tissue: Z-slab averaging would collapse left+right into a midline
point outside the muscle. So `source_tissue=muscle` dispatches to
`inob.sources.muscle.muscle_sources()`, which **volume-fills** the muscle interior
(one dipole per occupied voxel of a grid at `source_spacing_mm`, each taken from a
real muscle tet centroid → guaranteed inside muscle). At the muscle default 15 mm
spacing the current 40-muscle FEM yields **~736 sources** (`inob__source_target`
sets `SOURCE_SPACING=15`; a 5 mm grid would give thousands). Override with
`SOURCE_SPACING_OVERRIDE=<mm>` — pass the *same* value to `array`, `reduce`, and
`eeg` so MEG/EEG share source positions. Per-muscle long-axis fibre orientation
(from the 40 `data/muscle/*.stl`) is a viz-time projection — the solve stores all
3 moment components, and `snr`/`detectability` use the orientation-free norm.

The FEM voxelises muscle with `method="surface"` (surface-splat, not convex hull)
because the expanded set includes thin/curved sheets (platysma, splenius) whose
hull would over-fill the neck; see `fem/cgal_builder.py`. Adding/removing muscle
STLs needs a `inob-build-fem` rebuild (the FEM reads the STLs directly, not
`geometry.mat`).

Mechanism:
- **Solver.** `inob.sources.vagus.sample_source_tissues()` splits a
  comma-separated `forward.source_tissue` and concatenates dipoles across
  tissues. Single-tissue is byte-identical to the old `vagus_sources()` (kept for
  `sarvas_compare`). Wired into `chunk.py`, `reduce.py`, `resolve_source_positions`.
- **Batch.** `run_array.sh` / `run_reduce.sh` derive per-target chunk dir + NPZ
  and inject `--set forward.source_tissue=${TISSUES}`; `submit.sh` names jobs
  `fwd_<tag>` / `reduce_<tag>` so concurrent regions never collide.
- **Analysis.** `inob.cli._common` adds `--source-target <tag>`, repointing
  `outputs.forward_npz` / `forward_eeg_npz` to `duneuro_leadfield_<tag>.npz`.
  Analyses (`snr`, `detectability`, `sensitivity`, `cross_modality`) are
  leadfield-agnostic — they read `source_pos` from the NPZ, so they "just work"
  per region. `sarvas_compare` is vagus/analytic-sphere-specific only.

**Backward compatible:** no `SOURCE_TARGET` / `--source-target` → defaults to
vagus, same output filename as before; only internal renames (`out/`→`out_vagus/`,
`vagus_fwd`→`fwd_vagus`). Vagus users just re-stage (for the input-path fix).

### 5.1 Running the muscle target end-to-end (without clobbering prior runs)

Every muscle artifact is namespaced by `TARGET_TAG=muscle` (`out_muscle/`,
`duneuro_leadfield_muscle.npz`, jobs `fwd_muscle`/`reduce_muscle`/`eeg_muscle`),
so it **cannot** touch existing `out_vagus/` / `out_spine/` chunk dirs, their
`.npz`, or their jobs. The one shared input staging overwrites is the FEM
(`outputs/fem/fem.mat`) — required, since the cluster needs the
muscle-expanded FEM to sample muscle sources. Completed leadfields are already
saved and are unaffected; but the muscle rebuild re-ran CGAL globally, so
vagus/spine tets differ slightly from the original FEM — back it up first if you
want bit-exact reproduction of the earlier vagus/spine runs.

```bash
# 0. (once) preserve the FEM the previous runs used
ssh myriad 'cp -n ~/Scratch/inob/outputs/fem/fem.mat \
                  ~/Scratch/inob/outputs/fem/fem.pre_muscle.mat'

# 1. stage the muscle-expanded FEM + code (from the laptop, iNOB repo root)
CLUSTER_PROFILE=myriad bash cluster/stage.sh

# 2. MEG array + reduce (reduce auto-holds on fwd_muscle)
ssh myriad "cd ~/Scratch/inob/code && SOURCE_TARGET=muscle CLUSTER_PROFILE=myriad bash cluster/submit.sh array"
ssh myriad "cd ~/Scratch/inob/code && SOURCE_TARGET=muscle CLUSTER_PROFILE=myriad bash cluster/submit.sh reduce"

# 3. (optional) EEG leadfield
ssh myriad "cd ~/Scratch/inob/code && SOURCE_TARGET=muscle CLUSTER_PROFILE=myriad bash cluster/submit.sh eeg"

# 4. pull back + analyse (isolated by tag)
scp myriad:~/Scratch/inob/duneuro_leadfield_muscle.npz outputs/forward/
python -m inob.cli.snr           --source-target muscle
python -m inob.cli.detectability --source-target muscle --target detect --print-summary
```

At ~736 sources (vs 39 for vagus) each chunk applies the transfer to ~2200
dipoles; the 4 h array walltime is still ample (the per-chunk transfer-matrix
solve dominates, not dipole application). To lighten/densify, prepend
`SOURCE_SPACING_OVERRIDE=20` (or `=10`) to **all** of `array`/`reduce`/`eeg`.

**Muscle anisotropy is on by default for this target.** The config default
`forward.muscle_anisotropy.mode="auto"` enables the fibre-aligned tensor iff
`muscle` is one of the `source_tissue` labels — so the muscle solve uses the
tensor (§6) while vagus/spine stay isotropic, with no flag to pass.
`run_array.sh`/`run_eeg.sh` only emit an explicit
`--set forward.muscle_anisotropy.mode=on|off` when you force an A/B via
`MUSCLE_ANISOTROPY_OVERRIDE`.
`run_reduce.sh` needs no flag (it only stitches chunks; it never builds a
driver). Confirm it took effect in the chunk logs:

```bash
ssh myriad 'grep -m1 "Muscle anisotropy" ~/Scratch/inob/logs/chunk_muscle_0.log'
# Muscle anisotropy ENABLED: sigma_long=0.400 sigma_trans=0.100 S/m (4.0:1) over 40 muscle STLs (23386 muscle tets)
```

To solve isotropic muscle instead (e.g. to quantify what anisotropy changed),
prepend `MUSCLE_ANISOTROPY_OVERRIDE=0` to `array`/`eeg`, and write to a
different tag so you keep both.

---

## 6. Physics captured (and a caveat)

Solves quasi-static `∇·(σ∇φ) = ∇·Jᵖ`.
- **Primary current** Jᵖ = the dipole, via the `partial_integration` source model.
- **Volume/return currents** −σ∇φ = implicit in the σ-weighted PDE solution over
  the heterogeneous conductivities (bone/muscle/skin/…) — this is the field being
  attenuated/smeared as it penetrates tissue.
- **MEG B** = Biot–Savart over *both* primary + volume currents
  (`computeMEGTransferMatrix` / `applyMEGTransfer`).

Conductivity is **isotropic per tissue by default**, with one opt-in exception:
**muscle anisotropy**. When `forward.muscle_anisotropy` is active — `mode="auto"`
(the default) resolves to on whenever `muscle` is a source tissue, i.e. for
`SOURCE_TARGET=muscle`; see §5.1 — the muscle compartment gets a fibre-aligned
conductivity tensor

```text
σ = σ_⊥·I + (σ_∥ − σ_⊥)·ê⊗ê        σ_∥ = 0.40, σ_⊥ = 0.10 S/m (≈4:1)
```

with `ê` the per-muscle fibre axis — the *same* long-axis proxy used to orient
the muscle source dipoles, so sources and conductivity share one fibre field.
Implementation: each muscle tet is assigned to its nearest muscle STL, giving one
tensor per muscle; the driver then switches from duneuro's scalar
`{labels, conductivities}` volume-conductor description to `{labels, tensors}`
(a label→3×3 table, `tensors[labels[e]]` per element). Other tissues stay
isotropic σ·I, and targets that do not set the flag take the unchanged scalar
path — so **vagus/spine leadfields are byte-identical to before**.

Remaining caveat: the spinal cord (white matter) is also strongly anisotropic and
still uses a scalar σ. The tensor machinery above now makes that a small change
(supply a cord fibre axis and append its tensors) — the next physics refinement.

---

## 7. Analysing a leadfield (local)

Pull the leadfield back, then run analyses with `--source-target <tag>` (repoints
`outputs.forward_npz` to `duneuro_leadfield_<tag>.npz`):

```bash
scp myriad:~/Scratch/inob/duneuro_leadfield_spine.npz outputs/forward/
python -m inob.cli.snr           --source-target spine            # JSON only (per-source SNR)
python -m inob.cli.detectability --source-target spine --target detect --print-summary
python -m inob.cli.sensor_field  --source-target spine --out outputs/sensor_field_spine.png --print-summary
```

Analyses are **leadfield-agnostic** (they read `source_pos` from the NPZ), so they
work for any region. Notes:

- **`snr`** — numeric only (stdout / `--out file.json`); no figure.
- **`detectability`** — `--target detect` gives the SNR-vs-trials chart + summary.
  EEG is optional now: with no `duneuro_leadfield_<tag>_eeg.npz` it renders a
  MEG-only 3-panel figure and drops EEG keys from the summary. `--target
  all`/`surface` still needs EEG (the 3-D skin topoplot isn't MEG-only yet).
  Default figure name is untagged — pass `--out-detect outputs/detect_<tag>.png`.
- **`sensor_field`** — characterises the field *pattern on the OPM array* for a
  region: 3-D dipole, unrolled θ–z topography, strength along the source axis,
  and falloff. Flags: `--source-idx` (default = strongest source), `--moment
  {dominant,x,y,z}`, `--no-figure`. **Use `--moment z` for the physiologically
  real longitudinal cord CAP** — for the spine it peaks ~80 fT/nA·m (half the
  transverse-dominant 160) and is much less focal (lobe sep ~137 mm vs 53 mm),
  so real cord detectability is weaker/broader than the naive peak suggests.
  `--aggregate {coherent,rms}` shows the **global** field from all sources at
  once instead of one dipole: `coherent` = in-phase sum (upper bound; a uniform
  longitudinal cord largely self-cancels, field only at ends/bends), `rms` =
  sign-agnostic **sensitivity map** (where to place sensors — cord band, near
  side). A real travelling CAP lies between these (moving single-dipole).
- **`sarvas`** — vagus/analytic-sphere only; not meaningful for the cord.

Figure filenames are not auto-tagged by target — pass explicit `--out` paths per
region so vagus/spine outputs don't overwrite each other.
