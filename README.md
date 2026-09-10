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
make test                            # the full suite, no DUNEuro required
```

The DUNEuro forward solve needs the `duneuropy` extension. Every other stage
runs without it, so you only need this to run `inob-forward` / `inob-eeg`.

**Locally (macOS or Linux), the whole build is three commands.** conda supplies
the C++ toolchain, so there is nothing to install by hand:

```bash
conda env create -f environment.yml     # python 3.11 + eigen, gmp, metis, suitesparse, tbb
conda activate inob
bash scripts/build_duneuro_local.sh     # builds DUNE 2.10 + duneuro + duneuro-py (~30 min)
```

The script installs `duneuropy` into whatever environment is active, so
`inob-forward` then just works — no second interpreter to remember:

```bash
pip install -e .        # once, into the same env
inob-forward            # real DUNEuro solve
inob doctor             # confirms duneuropy is importable
```

Prefer Homebrew and a plain venv? That works too — the script takes the C++
dependencies from Homebrew whenever no conda env is active:

```bash
brew install eigen gmp metis suite-sparse tbb cmake pkg-config python@3.11
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pip install -e .
bash scripts/build_duneuro_local.sh     # installs duneuropy into .venv
```

Re-running the script is safe: clones, the source patch and the CMake
configure all no-op once they are in the state they want. Sources are built
under `~/.local/share/inob-duneuro` (override with `INOB_DUNEURO_BASE`), and
`INOB_VENV` picks the target environment explicitly. Already built duneuropy
somewhere else? Point at it with
`INOB_DUNEURO_PYTHON=/path/to/env/bin/python` and skip the build entirely.

DUNEuro needs one source patch for Eigen 5 / DUNE 2.10; it is versioned in
[`olivercase/duneuro-build`](https://github.com/olivercase/duneuro-build) and
fetched by tag, so the local, cluster and Docker builds all apply byte-identical
sources. You never clone that repo yourself.

**On a cluster:** `cluster/build_duneuro.sh` (see [Cluster](#cluster-ucl-myriad--kathleen-sge)).

## Run the full pipeline

**The whole workflow is two steps: edit one file, run one command.**

```bash
#  1. Edit the config — this one file controls the entire run:
#        anatomy, mesh resolution, sensors, conductivities, solver, noise.
$EDITOR configs/default.yaml

#  2. Run everything, start to finish, in one command:
inob run
```

That's it. `inob run` takes the STL meshes in `data/` all the way to a
calibrated leadfield in `outputs/forward/`, running each stage in order
(anatomy → FEM → sensors → forward solve → figures) and skipping any whose
output already exists. Nothing else needs editing — every knob lives in the
config file.

Two conveniences that don't change the model, only how you invoke it:

- **Try a change without editing the file:** `inob run --set fem.pitch_mm=2.0`
  overrides any config field for one run (repeatable). Good for sweeps.
- **Keep separate setups side by side:** `inob run --config configs/mine.yaml`
  points at a different file. Copy `default.yaml`, edit the copy, keep both.

```bash
inob status            # (optional) shows what's built and what's stale
inob run               # the one command — builds everything that's missing
inob run --force       # rebuild from scratch, ignoring cached outputs
```

Everything past this point is detail: individual stages, analysis commands,
and how to inspect results.

---

`configs/default.yaml` is the single source of truth for paths, mesh sizes,
sensor params, conductivities, and solver settings. Override any field with
`--set key.path=value`.

The pipeline runs through one command, `inob`:

```bash
inob doctor            # check this machine can run the pipeline
inob status            # what's built, and what to run next
inob run               # anatomy → FEM → sensors → leadfield → figures
inob detect            # trials-to-detect for the configured sources
```

`inob run` skips stages whose outputs already exist; `--force` rebuilds them,
`--stages geom,fem,sensors` runs a subset (everything except the solve).

| Group | Commands |
| --- | --- |
| Start here | `doctor` · `status` · `run` |
| Build the model | `build-geom` · `build-fem` · `sensors` · `electrodes` |
| Solve | `forward` (MEG) · `eeg` · `volume-field` |
| Analyse | `detect` · `snr` · `sensitivity` · `location` · `cross` · `physiology` · `cap-compare` · `source-models` |
| Validate | `ladder` · `sarvas` · `calibrate` |
| Figures | `topoplot` · `torso` · `sensor-field` · `visualise` · `muscle-sources` |

`inob --help` lists them all; `inob <command> --help` documents one. The older
`inob-*` binaries (`inob-pipeline`, `inob-build-fem`, …) still work unchanged —
`inob run` and `inob-pipeline` are the same code.

## Example: field from a neck muscle (around C7)

Model the magnetic field of a single current dipole placed at the centre of a
neck muscle, pointing along the muscle's long axis.

**Quick look (analytic, no DUNEuro, seconds):** paints the radial field on the
torso skin using the Sarvas single-sphere solution.

```bash
inob run --stages geom                         # build the skin surface once
python scripts/muscle_field_sarvas.py          # default: scalene group at C7
python scripts/muscle_field_sarvas.py sternocleido   # or any muscle substring
# writes outputs/muscle_skin_topoplot.png
```

**Full FEM (DUNEuro, head-to-head MEG/EEG, absolute units):** place explicit
dipoles with `forward.point_sources` (mm, in the atlas frame). The coordinates
below are the centroids of the left/right scalenus anterior and medius:

```bash
inob run --stages all --set \
  'forward.point_sources=[[33.8,-88.4,1379.8],[34.8,-78.4,1391.0],[-34.7,-88.1,1381.0],[-36.1,-78.4,1392.6]]'
inob topoplot --target meg --source-idx 0      # field map for source 0
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

## Documentation

| Read | For |
| --- | --- |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | fresh clone to a leadfield and a trials-to-detect number |
| [`configs/default.yaml`](configs/default.yaml) | every field of the reference configuration, commented |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | regenerate the paper's figures from `data/` |
| [`docs/COMPARISON.md`](docs/COMPARISON.md) | what iNOB reuses and how it differs from MNE, FieldTrip, Brainstorm, SimNIBS, OpenMEEG, DUNEuro, ASCENT |
| [`docs/CITING.md`](docs/CITING.md) | what to cite: iNOB and the packages under it |
| [`docs/COORDINATES.md`](docs/COORDINATES.md) | the axis convention, fixed once |
| [`cluster/README.md`](cluster/README.md) | SGE array jobs and the cluster DUNEuro build |
| [`gui/web/API_CONTRACT.md`](gui/web/API_CONTRACT.md) | the GUI backend API |

`inob <command> --help` is the reference for each command and is tested to
stay readable.

## Validation

Every leadfield magnitude is checked against closed-form solutions: `inob
sarvas` and `inob ladder` compare the FEM with the Sarvas sphere and
Biot–Savart, and `inob calibrate` fixes the absolute units (fT, µV per nA·m).
The same comparisons run as tests (`tests/test_meg_sphere_validation.py`,
`tests/test_venant_sphere_validation.py`) every week in CI against a real
DUNEuro build ([duneuro-validation](.github/workflows/duneuro-validation.yml)).
The rest of the suite (`make test`, 880+ tests) runs on every push.

## Repository layout

```
src/inob/        the package: geometry, fem, sensors, forward, analysis, viz, cli
configs/         default.yaml, the single source of truth
data/            anatomical meshes (BodyParts3D-derived STLs)
gui/             web/ (Next.js frontend) + backend/ (FastAPI)
cluster/         SGE job scripts + DUNEuro build for UCL Myriad/Kathleen
scripts/         standalone tools (muscle field, FEM/atlas viewers)
tests/           pytest suite (no DUNEuro required)
docs/            quickstart, reproduction, comparison, citing, coordinates
```

## Contributing

Bug reports about numbers are as welcome as bug reports about crashes; see
[`CONTRIBUTING.md`](CONTRIBUTING.md). `make lint && make test` before a pull
request.

## License and citation

Code under the MIT License (`LICENSE`). Anatomical data are BodyParts3D
derivatives under CC BY-SA 2.1 JP (`LICENSE-DATA`). Cite via `CITATION.cff` —
software DOI: [10.17605/OSF.IO/U4MDS](https://doi.org/10.17605/OSF.IO/U4MDS)
— and cite the packages iNOB is built on; [`docs/CITING.md`](docs/CITING.md)
lists them. DUNEuro (LGPL) does the forward solves and must be cited with any
leadfield produced here.
