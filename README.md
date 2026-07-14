# iNOB — imaging neuroscience outside the brain

A whole-body **forward-modelling and sensor-planning** package for non-invasive
recording of peripheral and autonomic nerves. You pick a target structure and a
source configuration, choose which anatomical meshes to include and their
conductivities, generate an OPM and/or surface-electrode array, and run the
pipeline from atlas meshes through a multi-tissue tetrahedral FEM to calibrated
**MEG and EEG leadfields** (via DUNEuro). It runs from a single command or a
browser GUI.

It answers one planning question: **given a sensor noise floor, how many
averaged trials are needed to detect a given source?** Both leadfields come
from the same FEM and the same conductivities, and are calibrated to absolute
units (fT, µV) against analytic solutions.

```
data/{bone,torso,vagus,muscle,vessel}/*.stl
   │  geom      watertighten + shrinkwrap
   ▼
 outputs/geometry        │  fem       iso2mesh + CGAL multi-tissue tetra mesh
   ▼                     ▼
 outputs/fem ── sensors (OPM array) / electrodes (HD-EMG patch)
   │  forward   DUNEuro  →  MEG + EEG leadfields (fT, µV per nA·m)
   ▼
 outputs/forward  ── viz (topoplots) / snr / detect (trials-to-detect)
```

## Install

The anatomical meshes (`*.stl`, `*.obj`) are stored with **Git LFS**, so you
need Git LFS installed *before* cloning. Without it the clone pulls 132-byte
pointer files instead of meshes and the pipeline fails.

```bash
git lfs install                      # once per machine (brew install git-lfs)
git clone https://github.com/olivercase/iNOB.git
cd iNOB
git lfs pull                         # fetch the meshes if the clone didn't
python3 -m pip install -e .[dev]     # editable install with dev deps (Python 3.11+)
make test                            # 140 tests, no DUNEuro required
```

The DUNEuro forward solve needs the `duneuropy` extension. Every other stage
runs without it, so you only need this to run `inob-forward` / `inob-eeg`.

**Locally (macOS / Apple Silicon):** `scripts/build_duneuro_local.sh` builds
DUNE 2.10 + duneuro + duneuro-py against Homebrew + `python@3.11` into a self-
contained venv (default `/Volumes/UCL/duneuro_build/venv`). Because the solve
runs from that venv, install this package into it too, then call the CLI with
that interpreter:

```bash
brew install eigen gmp metis superlu cmake python@3.11   # one-time prerequisites
bash scripts/build_duneuro_local.sh                      # builds duneuropy (~30 min)
BASE=/Volumes/UCL/duneuro_build
"$BASE/venv/bin/pip" install --no-deps -e .               # put inob in the same venv
"$BASE/venv/bin/inob-forward"                             # real DUNEuro solve
```

The build recipe (scripts, Eigen-5/DUNE-2.10 patch, reference logs) also lives
standalone at [`olivercase/duneuro-build`](https://github.com/olivercase/duneuro-build).

**On a cluster:** `cluster/build_duneuro.sh` (see [Cluster](#cluster-ucl-myriad--kathleen-sge)).

## Run the full pipeline

`configs/default.yaml` is the single source of truth for paths, mesh sizes,
sensor params, conductivities, and solver settings. Override any field with
`--set key.path=value`.

```bash
# Build anatomy → FEM → sensor array → forward solve → figures, in order.
# Stages whose outputs already exist are skipped (use --force to rebuild).
inob-pipeline --stages all                       # needs duneuropy for 'forward'
inob-pipeline --stages geom,fem,sensors          # everything except the solve
```

Individual stages (same effect, finer control):

```bash
inob-build-geom        # STLs → watertight geometry
inob-build-fem         # geometry → multi-tissue tetrahedral FEM
inob-sensors           # place the OPM triaxial array
inob-electrodes        # place the HD-EMG surface patch
inob-forward           # MEG leadfield via DUNEuro
inob-eeg               # EEG leadfield via DUNEuro
inob-topoplot --target dual    # MEG + EEG field topoplots
inob-detect            # trials-to-detect for the configured sources
```

## Example: field from a neck muscle (around C7)

Model the magnetic field of a single current dipole placed at the centre of a
neck muscle, pointing along the muscle's long axis.

**Quick look (analytic, no DUNEuro, seconds):** paints the radial field on the
torso skin using the Sarvas single-sphere solution.

```bash
inob-pipeline --stages geom                    # build the skin surface once
python scripts/muscle_field_sarvas.py          # default: scalene group at C7
python scripts/muscle_field_sarvas.py sternocleido   # or any muscle substring
# writes outputs/muscle_skin_topoplot.png
```

**Full FEM (DUNEuro, head-to-head MEG/EEG, absolute units):** place explicit
dipoles with `forward.point_sources` (mm, in the atlas frame). The coordinates
below are the centroids of the left/right scalenus anterior and medius:

```bash
inob-pipeline --stages all --set \
  'forward.point_sources=[[33.8,-88.4,1379.8],[34.8,-78.4,1391.0],[-34.7,-88.1,1381.0],[-36.1,-78.4,1392.6]]'
inob-topoplot --target meg --source-idx 0      # field map for source 0
```

Muscle is a FEM tissue by default (`fem.tissues` includes `muscle`,
σ = 0.35 S/m). The solve returns all three moment components per source, so the
"along the muscle axis" projection is applied at the visualisation step.

## Browser GUI

```bash
# backend (FastAPI): serves config, meshes, runs, cluster submission
uvicorn gui.backend.app:app --port 8000
# frontend (Next.js): 3-D viewer, click-to-place sources, live run console
cd gui/web && npm install && npm run dev        # http://localhost:3000
```

Pick an imaging target, toggle tissues, click sources onto the anatomy (they
snap to the target), edit conductivities/arrays, run, and read trials-to-detect.

## Cluster (UCL Myriad / Kathleen, SGE)

The DUNEuro solve fans the 8190-channel leadfield over an array job.

```bash
# one-time: build DUNE + duneuro on a compute node
ssh myriad
cd ~/Scratch/inob/code && CLUSTER_PROFILE=myriad bash cluster/build_duneuro.sh

# from your laptop: stage inputs + submit the forward solve
CLUSTER_PROFILE=myriad bash cluster/stage.sh
ssh myriad "cd ~/Scratch/inob/code && CLUSTER_PROFILE=myriad bash cluster/submit.sh array"
ssh myriad "cd ~/Scratch/inob/code && CLUSTER_PROFILE=myriad bash cluster/submit.sh reduce"
```

Profiles live in `cluster/profiles/*.env`. The GUI's "Run on cluster" button
drives the same flow.

## Repository layout

```
src/inob/        the package: geometry, fem, sensors, forward, analysis, viz, cli
configs/         default.yaml, the single source of truth
data/            anatomical meshes (BodyParts3D-derived STLs)
gui/             web/ (Next.js frontend) + backend/ (FastAPI)
cluster/         SGE job scripts + DUNEuro build for UCL Myriad/Kathleen
scripts/         standalone tools (muscle field, FEM/atlas viewers)
tests/           pytest suite (no DUNEuro required)
```

## License

Code under the MIT License (`LICENSE`). Anatomical data under `LICENSE-DATA`.
Cite via `CITATION.cff`.
